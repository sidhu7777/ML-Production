"""
Phase 26: corrected outdoor building obstruction profile.

This test-only phase starts from the Phase 19/22 path, but replaces the
outdoor building obstruction calculation. The old path summed many crossed
building knife-edge losses with diminishing weights; in dense urban grids that
created 45-60 dB outdoor obstruction and false no-coverage holes near sites.

Phase 26 keeps:
  - Phase 9 COST-231 + directional antenna candidates
  - Phase 22 DEM terrain diffraction
  - indoor entry/depth logic unchanged
  - final no-coverage threshold as NaN below -140 dBm

Phase 26 changes only outdoor building obstruction:
  - one dominant building obstacle per ray
  - median of the small fan of rays
  - no multi-building summed outdoor loss
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
for path in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit_project210_phase13_beam_check as phase13
import streamlit_project210_phase15_radius_progression as phase15
import test_project210_phase17_full_polygon_geo_dt_comparison as phase17
import test_project210_phase22_terrain_diffraction_comparison as phase22
from phase_rsrp_guard import valid_model_rsrp


PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE20_DIR = PROJECT_DIR / "cost231_phase20_5g_real_dt_match"
OUT_DIR = PROJECT_DIR / "cost231_phase26_corrected_obstruction_profile"
IMAGE_DIR = OUT_DIR / "images"

UE_HEIGHT_M = phase22.UE_HEIGHT_M
DEM_PATH = phase22.DEM_PATH
MIN_DT_FOR_REPRESENTATIVE_CLASS = phase17.MIN_DT_FOR_REPRESENTATIVE_CLASS
RAW_SERVING_MARGIN_DB = 20.0
RAW_MIN_FOR_EXPENSIVE_GEOMETRY_DBM = -145.0


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _dominant_path_diffraction_loss_db(
    site_pt: Point,
    target_pt: Point,
    total_dist_m: float,
    buildings_gdf: gpd.GeoDataFrame,
    sindex,
    tx_height_m: float,
    rx_height_m: float,
    wavelength_m: float,
) -> tuple[float, int]:
    line = LineString([site_pt, target_pt])
    candidate_idx = list(sindex.query(line, predicate="intersects"))
    if not candidate_idx:
        return 0.0, 0

    dominant_loss = 0.0
    obstacle_count = 0
    for j in candidate_idx:
        poly = buildings_gdf.geometry.iloc[j]
        height = buildings_gdf["height"].iloc[j]
        if not np.isfinite(height) or height <= 0:
            continue
        inter = line.intersection(poly)
        if inter.is_empty:
            continue
        coords: list = []
        if inter.geom_type == "Point":
            coords = [(inter.x, inter.y)]
        elif inter.geom_type == "LineString":
            coords = list(inter.coords)
        elif inter.geom_type in ("MultiLineString", "GeometryCollection"):
            for geom in getattr(inter, "geoms", []):
                if hasattr(geom, "coords"):
                    coords.extend(list(geom.coords))
        if not coords:
            continue

        entry_lon, entry_lat = min(coords, key=lambda coord: site_pt.distance(Point(coord)))
        d1_m = max(phase15._haversine_m(site_pt.y, site_pt.x, entry_lat, entry_lon), 1.0)
        d2_m = max(total_dist_m - d1_m, 1.0)
        frac = d1_m / max(d1_m + d2_m, 1.0)
        los_height_here = tx_height_m + frac * (rx_height_m - tx_height_m)
        h_obstruction_m = float(height) - los_height_here
        loss = phase15._knife_edge_loss_db(h_obstruction_m, d1_m, d2_m, wavelength_m)
        if loss > 0:
            obstacle_count += 1
            dominant_loss = max(dominant_loss, loss)

    return dominant_loss, obstacle_count


def _median_fan_dominant_diffraction_loss_db(
    site_lat: float,
    site_lon: float,
    site_pt: Point,
    target_lat: float,
    target_lon: float,
    total_dist_m: float,
    buildings_gdf: gpd.GeoDataFrame,
    sindex,
    tx_height_m: float,
    rx_height_m: float,
    wavelength_m: float,
) -> tuple[float, int]:
    if total_dist_m <= 1.0:
        return 0.0, 0
    bearing = phase15._bearing_deg(site_lat, site_lon, target_lat, target_lon)
    losses: list[float] = []
    obstacle_total = 0
    for offset in phase15.FAN_OFFSETS_DEG:
        lat_i, lon_i = phase15._destination_point(site_lat, site_lon, bearing + offset, total_dist_m)
        loss, n_obstacles = _dominant_path_diffraction_loss_db(
            site_pt,
            Point(lon_i, lat_i),
            total_dist_m,
            buildings_gdf,
            sindex,
            tx_height_m,
            rx_height_m,
            wavelength_m,
        )
        losses.append(loss)
        obstacle_total += n_obstacles
    return float(np.median(losses)), obstacle_total


def _geo_correction_with_branch_phase26(
    grid_df: pd.DataFrame,
    clutter_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    center_lat: float,
    center_lon: float,
    tx_height_m: float,
    rx_height_m: float,
    freq_mhz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(grid_df)
    correction = np.zeros(n, dtype=float)
    branch = np.array(["clear"] * n, dtype=object)
    wavelength_m = phase15.LIGHT_SPEED_M_S / (freq_mhz * 1e6)
    clutter_weights = dict(phase15.DEFAULT_CLUTTER_WEIGHTS)
    building_area_weight = phase15.DEFAULT_BUILDING_AREA_WEIGHT

    clutter_lookup = phase15._lookup_clutter(grid_df, clutter_gdf)
    clutter_classes = clutter_lookup["clutter_class"].to_numpy()
    building_area_ratios = clutter_lookup["building_area_ratio"].to_numpy()

    def _env_adj(cls) -> float:
        if not cls or cls in phase15.OBSTRUCTION_PROXY_CLUTTER_CLASSES:
            return 0.0
        return clutter_weights.get(cls, 0.0)

    if buildings_gdf.empty:
        for i in range(n):
            cls = clutter_classes[i]
            bar = building_area_ratios[i]
            proxy = clutter_weights.get(cls, 0.0) if cls else 0.0
            correction[i] = proxy + (float(bar) if pd.notna(bar) else 0.0) * building_area_weight
        return correction, branch, clutter_classes

    site_pt = Point(center_lon, center_lat)
    sindex = buildings_gdf.sindex
    for i in range(n):
        cls = clutter_classes[i]
        bar = building_area_ratios[i]
        env_adj = _env_adj(cls)
        lat_i = float(grid_df["lat"].iloc[i])
        lon_i = float(grid_df["lon"].iloc[i])
        pt = Point(lon_i, lat_i)

        candidate_idx = list(sindex.query(pt, predicate="intersects"))
        containing = [j for j in candidate_idx if buildings_gdf.geometry.iloc[j].contains(pt)]
        if containing:
            depth_m = max(
                (phase15._indoor_depth_m(site_pt, pt, buildings_gdf.geometry.iloc[j]) for j in containing),
                default=0.0,
            )
            correction[i] = env_adj - 15.0 + depth_m * -0.5
            branch[i] = "indoor"
            continue

        total_dist_m = phase15._haversine_m(center_lat, center_lon, lat_i, lon_i)
        diffraction_loss, n_obstacles = _median_fan_dominant_diffraction_loss_db(
            center_lat,
            center_lon,
            site_pt,
            lat_i,
            lon_i,
            total_dist_m,
            buildings_gdf,
            sindex,
            tx_height_m,
            rx_height_m,
            wavelength_m,
        )
        if n_obstacles > 0:
            correction[i] = env_adj - diffraction_loss
            branch[i] = "obstructed"
        else:
            proxy = clutter_weights.get(cls, 0.0) if cls else 0.0
            correction[i] = env_adj + proxy + (float(bar) if pd.notna(bar) else 0.0) * building_area_weight
            branch[i] = "clear"

    return correction, branch, clutter_classes


def _score_with_phase26_obstruction(
    points: pd.DataFrame,
    identity: pd.DataFrame,
    clutter_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    dem: phase22.TerrainSampler,
    key_col: str,
    raw_col: str,
) -> pd.DataFrame:
    original = phase22.phase19._geo_correction_with_branch
    phase22.phase19._geo_correction_with_branch = _geo_correction_with_branch_phase26
    try:
        out = phase22._score_points(points, identity, clutter_gdf, buildings_gdf, dem, key_col, raw_col)
    finally:
        phase22.phase19._geo_correction_with_branch = original
    out["phase26_obstruction_model"] = "dominant_obstacle_median_fan"
    return out


def _metrics(measured: pd.Series, predicted: pd.Series) -> dict:
    err = pd.to_numeric(measured, errors="coerce") - pd.to_numeric(predicted, errors="coerce")
    arr = err.dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mae": math.nan, "rmse": math.nan, "bias": math.nan, "p90_abs": math.nan}
    return {
        "mae": float(np.mean(np.abs(arr))),
        "rmse": float(np.sqrt(np.mean(np.square(arr)))),
        "bias": float(np.mean(arr)),
        "p90_abs": float(np.quantile(np.abs(arr), 0.90)),
    }


def _rf_plausible_candidates(surface: pd.DataFrame) -> pd.DataFrame:
    out = surface.copy()
    raw = pd.to_numeric(out["raw_cost231_rsrp"], errors="coerce")
    best_raw = out.assign(_raw_cost231=raw).groupby(["technology", "grid_id"])["_raw_cost231"].transform("max")
    keep = raw.notna() & (raw >= best_raw - RAW_SERVING_MARGIN_DB) & (raw >= RAW_MIN_FOR_EXPENSIVE_GEOMETRY_DBM)
    filtered = out.loc[keep].copy()
    coverage = filtered.groupby("technology")["grid_id"].nunique().to_dict()
    print(
        "[PHASE26] RF-plausible candidate filter: "
        f"{len(surface)} -> {len(filtered)} rows, "
        f"margin={RAW_SERVING_MARGIN_DB:.1f} dB, min_raw={RAW_MIN_FOR_EXPENSIVE_GEOMETRY_DBM:.1f} dBm, "
        f"grid coverage={coverage}"
    )
    return filtered


def _rename_phase22_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: col.replace("phase22_", "phase26_") for col in df.columns if col.startswith("phase22_")})


def _aggregate_phase26_by_grid(
    candidates: pd.DataFrame,
    replacements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stale_replacement_cols = [
        col
        for col in ["dt_replacement_rsrp", "dt_replacement_count"]
        if col in candidates.columns
    ]
    out = candidates.drop(columns=stale_replacement_cols).merge(
        replacements,
        on=["technology", "grid_id"],
        how="left",
    )
    lock = out["dt_replacement_rsrp"].notna()

    bias = pd.to_numeric(out["phase22_phase19_bias_db"], errors="coerce").fillna(0.0)
    out["phase22_no_terrain_calibrated_no_lock_unclipped"] = (
        pd.to_numeric(out["phase22_physical_no_terrain_rsrp_unclipped"], errors="coerce") + bias
    )
    out["phase22_with_terrain_calibrated_no_lock_unclipped"] = (
        pd.to_numeric(out["phase22_physical_with_terrain_rsrp_unclipped"], errors="coerce") + bias
    )
    out["phase22_no_terrain_calibrated_no_lock"] = valid_model_rsrp(
        out["phase22_no_terrain_calibrated_no_lock_unclipped"]
    )
    out["phase22_with_terrain_calibrated_no_lock"] = valid_model_rsrp(
        out["phase22_with_terrain_calibrated_no_lock_unclipped"]
    )
    out["phase22_no_terrain_calibrated_rsrp"] = out["phase22_no_terrain_calibrated_no_lock"].where(
        ~lock, out["dt_replacement_rsrp"]
    )
    out["phase22_with_terrain_calibrated_rsrp"] = out["phase22_with_terrain_calibrated_no_lock"].where(
        ~lock, out["dt_replacement_rsrp"]
    )

    agg = out.groupby(["technology", "grid_id"], dropna=False).agg(
        {
            "phase22_physical_no_terrain_rsrp": ["max", "mean"],
            "phase22_physical_with_terrain_rsrp": ["max", "mean"],
            "phase22_no_terrain_calibrated_rsrp": ["max", "mean"],
            "phase22_with_terrain_calibrated_rsrp": ["max", "mean"],
            "terrain_diffraction_loss_db": ["mean", "max"],
            "building_geo_correction_db": ["mean"],
            "terrain_obstructed": ["mean"],
        }
    )
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    agg = agg.rename(
        columns={
            "phase22_physical_no_terrain_rsrp_max": "phase22_physical_no_terrain_best_rsrp",
            "phase22_physical_no_terrain_rsrp_mean": "phase22_physical_no_terrain_mean_rsrp",
            "phase22_physical_with_terrain_rsrp_max": "phase22_physical_with_terrain_best_rsrp",
            "phase22_physical_with_terrain_rsrp_mean": "phase22_physical_with_terrain_mean_rsrp",
            "phase22_no_terrain_calibrated_rsrp_max": "phase22_no_terrain_best_rsrp",
            "phase22_no_terrain_calibrated_rsrp_mean": "phase22_no_terrain_mean_rsrp",
            "phase22_with_terrain_calibrated_rsrp_max": "phase22_with_terrain_best_rsrp",
            "phase22_with_terrain_calibrated_rsrp_mean": "phase22_with_terrain_mean_rsrp",
            "terrain_obstructed_mean": "terrain_obstructed_share",
        }
    )
    return agg, out


def main() -> None:
    _ensure_dirs()
    surface = phase22._read_frame(PHASE9_DIR / "phase9_directional_raw_corrected_surface_project210")
    grid = phase22._read_frame(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210")
    dt_source = (
        PHASE20_DIR / "phase9_dt_match_project210_corrected"
        if (PHASE20_DIR / "phase9_dt_match_project210_corrected.parquet").exists()
        else PHASE9_DIR / "phase9_dt_match_project210"
    )
    dt = phase22._read_frame(dt_source)
    identity = phase13.load_identity()
    clutter_gdf, buildings_gdf = phase17._load_clutter_and_buildings()
    dt = phase17._classify_dt_clutter(dt, clutter_gdf)
    freq_lookup = phase22._frequency_lookup(surface)
    dt_points = dt.merge(freq_lookup, left_on="assigned_strict_cell_key", right_on="strict_cell_key", how="left")
    dt_points["technology"] = dt_points["assigned_technology"]
    surface_for_scoring = _rf_plausible_candidates(surface)

    dem = phase22.TerrainSampler(DEM_PATH)
    try:
        print(f"[PHASE26] surface rows={len(surface_for_scoring)} dt rows={len(dt_points)} identity rows={len(identity)}")
        candidates = _score_with_phase26_obstruction(
            surface_for_scoring, identity, clutter_gdf, buildings_gdf, dem, "strict_cell_key", "raw_cost231_rsrp"
        )
        dt_scored = _score_with_phase26_obstruction(
            dt_points, identity, clutter_gdf, buildings_gdf, dem, "assigned_strict_cell_key", "raw_cost231_at_dt_rsrp"
        )
    finally:
        dem.close()

    dt_scored["dt_minus_no_terrain_physical_db"] = (
        dt_scored["rsrp_measured"] - dt_scored["phase22_physical_no_terrain_rsrp"]
    )
    dt_scored["dt_minus_with_terrain_physical_db"] = (
        dt_scored["rsrp_measured"] - dt_scored["phase22_physical_with_terrain_rsrp"]
    )
    bias = phase22._bias_table(dt_scored, "dt_minus_no_terrain_physical_db")
    bias["bias_source"] = "phase26_no_terrain_physical_residual"
    bias.to_csv(OUT_DIR / "phase26_bias_by_condition.csv", index=False)

    candidates = phase22._attach_bias(candidates, bias, "phase22_phase19_bias_db")
    replacements = phase22._corrected_dt_replacements(dt)
    grid_agg, scored_candidates = _aggregate_phase26_by_grid(candidates, replacements)
    scored_candidates["phase22_terrain_delta_db"] = (
        scored_candidates["phase22_physical_with_terrain_rsrp"]
        - scored_candidates["phase22_physical_no_terrain_rsrp"]
    )

    grid_bounds = grid[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]].copy()
    all_tech_grid = pd.concat(
        [grid_bounds[["grid_id"]].assign(technology=technology) for technology in ["4G", "5G"]],
        ignore_index=True,
    )
    grid_agg = all_tech_grid.merge(grid_agg, on=["technology", "grid_id"], how="left")

    summary = {
        "base_candidate_file": str(PHASE9_DIR / "phase9_directional_raw_corrected_surface_project210.parquet"),
        "outdoor_obstruction_model": "dominant building obstacle per ray, median fan loss, no multi-building summation",
        "coverage_threshold_dbm": -140.0,
        "technology": {},
    }
    for technology in ["4G", "5G"]:
        serving = grid_agg[grid_agg["technology"].astype(str) == technology].merge(grid_bounds, on="grid_id", how="left")
        serving = _rename_phase22_cols(serving)
        dt_tech = dt_scored[dt_scored["assigned_technology"].astype(str) == technology].copy()
        dt_tech = phase22._attach_bias(dt_tech, bias, "phase22_phase19_bias_db")
        dt_tech["phase22_no_terrain_calibrated_rsrp_unclipped"] = (
            dt_tech["phase22_physical_no_terrain_rsrp_unclipped"] + dt_tech["phase22_phase19_bias_db"]
        )
        dt_tech["phase22_with_terrain_calibrated_rsrp_unclipped"] = (
            dt_tech["phase22_physical_with_terrain_rsrp_unclipped"] + dt_tech["phase22_phase19_bias_db"]
        )
        dt_tech["phase22_no_terrain_calibrated_rsrp"] = valid_model_rsrp(
            dt_tech["phase22_no_terrain_calibrated_rsrp_unclipped"]
        )
        dt_tech["phase22_with_terrain_calibrated_rsrp"] = valid_model_rsrp(
            dt_tech["phase22_with_terrain_calibrated_rsrp_unclipped"]
        )
        dt_tech = _rename_phase22_cols(dt_tech)

        phase22._save_frame(serving, OUT_DIR / f"phase26_serving_grid_{technology.lower()}_project210")
        phase22._plot_cdf(
            [
                ("Physical before terrain", serving["phase26_physical_no_terrain_best_rsrp"], "#ef4444"),
                ("Physical after terrain", serving["phase26_physical_with_terrain_best_rsrp"], "#2563eb"),
                ("Calibrated before terrain", serving["phase26_no_terrain_best_rsrp"], "#f97316"),
                ("Same bias + terrain", serving["phase26_with_terrain_best_rsrp"], "#16a34a"),
            ],
            f"Project 210 {technology}: Phase 26 corrected obstruction",
            IMAGE_DIR / f"phase26_{technology.lower()}_full_polygon_cdf.png",
        )
        phase22._plot_cdf(
            [
                ("DT measured", dt_tech["rsrp_measured"], "#111827"),
                ("Physical after terrain", dt_tech["phase26_physical_with_terrain_rsrp"], "#2563eb"),
                ("Calibrated after terrain", dt_tech["phase26_with_terrain_calibrated_rsrp"], "#16a34a"),
            ],
            f"Project 210 {technology}: Phase 26 DT comparison",
            IMAGE_DIR / f"phase26_{technology.lower()}_dt_cdf.png",
        )

        building = pd.to_numeric(serving["building_geo_correction_db_mean"], errors="coerce")
        value = pd.to_numeric(serving["phase26_with_terrain_best_rsrp"], errors="coerce")
        dt_value = dt_tech["phase26_with_terrain_calibrated_rsrp"]
        summary["technology"][technology] = {
            "grid_rows": int(len(serving)),
            "candidate_rows": int((surface_for_scoring["technology"].astype(str) == technology).sum()),
            "valid_grid_rows": int(value.notna().sum()),
            "no_coverage_grid_rows": int(value.isna().sum()),
            "mean_building_geo_correction_db": float(building.mean()),
            "building_geo_correction_db": {
                "p10": float(building.quantile(0.10)),
                "p50": float(building.quantile(0.50)),
                "p90": float(building.quantile(0.90)),
                "min": float(building.min()),
                "max": float(building.max()),
            },
            "mean_phase26_with_terrain_best_rsrp": float(value.mean()),
            "dt_phase26_calibrated_with_terrain": _metrics(dt_tech["rsrp_measured"], dt_value),
            "images": {
                "full_polygon_cdf": str((IMAGE_DIR / f"phase26_{technology.lower()}_full_polygon_cdf.png").relative_to(THIS_DIR)),
                "dt_cdf": str((IMAGE_DIR / f"phase26_{technology.lower()}_dt_cdf.png").relative_to(THIS_DIR)),
            },
        }
        print(f"[PHASE26] wrote {technology} serving rows={len(serving)}")

    scored_candidates = _rename_phase22_cols(scored_candidates)
    dt_scored = _rename_phase22_cols(dt_scored)
    phase22._save_frame(scored_candidates, OUT_DIR / "phase26_scored_candidates_project210")
    phase22._save_frame(dt_scored, OUT_DIR / "phase26_dt_scored_project210")
    (OUT_DIR / "phase26_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("[PHASE26] summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
