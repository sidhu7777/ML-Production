"""
Phase 19: DT-calibrated correction gated by BOTH clutter class AND
obstruction branch (indoor / obstructed / clear) - not just clutter class
the way Phase 17's IDW residual was.

Why: Phase 18 (test_project210_phase18_dt_point_diagnostic.py) measured
real DT error per DT point and showed clutter class alone is too coarse -
the same "4G Dense Urban" cell has wildly different real-world bias
depending on the obstruction condition of the actual path to it:
    clear:       -21.0 dB median (model under-corrects, no buildings at all)
    obstructed:  +15.4 dB median (model over-corrects, diffraction too big)
    indoor:      -14.9 dB median (reasonably close already)
One flat clutter-class weight/IDW-residual could never capture that split -
it was always going to land somewhere in between and be wrong for all
three conditions at once.

This phase replaces the Phase 17 IDW dt_residual mechanism with a bias
correction read directly from Phase 18's own measured
[technology, clutter_class, obstruction_branch] error medians, applied
ONLY where that exact combination had >= MIN_DT_FOR_REPRESENTATIVE_CLASS
real DT points behind it (same threshold Phase 11/12/17 already used) -
never extrapolated to a combination DT didn't actually cover. Everywhere
else falls back to the physical model alone, honestly uncorrected.

Does not modify Phase 15, 16, or 17 in any way - only imports and reuses
their already-written, already-verified functions/output read-only.
Phase 17's own saved output is left untouched for side-by-side comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
for p in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit_project210_phase13_beam_check as phase13
import streamlit_project210_phase15_radius_progression as phase15
import test_project210_phase17_full_polygon_geo_dt_comparison as phase17  # reused read-only, not modified

MIN_DT_FOR_REPRESENTATIVE_CLASS = phase17.MIN_DT_FOR_REPRESENTATIVE_CLASS  # 8, same threshold as Phase 11/12/17
RSRP_MIN, RSRP_MAX = phase17.RSRP_MIN, phase17.RSRP_MAX
N78_TECHNOLOGY_OFFSET_DB = phase17.N78_TECHNOLOGY_OFFSET_DB
PHASE18_DIR = THIS_DIR / "data" / "project_210_taiwan" / "cost231_phase18_dt_point_diagnostic"
OUT_DIR = THIS_DIR / "data" / "project_210_taiwan" / "cost231_phase19_branch_calibrated_comparison"


def _load_bias_table() -> pd.DataFrame:
    """Phase 18's own measured error, filtered to combinations with enough
    real DT points to trust, kept as-is (not smoothed/re-fit) - the bias
    applied here is exactly what DT itself showed for that condition."""
    dt = pd.read_parquet(PHASE18_DIR / "phase18_dt_point_diagnostic_project210.parquet")
    table = (
        dt.groupby(["assigned_technology", "clutter_class", "obstruction_branch"])
        .agg(n=("phase18_error_db", "size"), bias_db=("phase18_error_db", "median"))
        .reset_index()
        .rename(columns={"assigned_technology": "technology"})
    )
    representative = table[table["n"] >= MIN_DT_FOR_REPRESENTATIVE_CLASS].copy()
    print("[PHASE19] bias table (representative combinations only, n >= "
          f"{MIN_DT_FOR_REPRESENTATIVE_CLASS}):")
    print(representative.to_string(index=False))
    dropped = table[table["n"] < MIN_DT_FOR_REPRESENTATIVE_CLASS]
    if not dropped.empty:
        print(f"[PHASE19] {len(dropped)} combination(s) had too few DT points and get NO bias "
              f"(physical model only): {dropped[['technology', 'clutter_class', 'obstruction_branch', 'n']].to_dict('records')}")
    return representative[["technology", "clutter_class", "obstruction_branch", "bias_db"]]


def _geo_correction_with_branch(
    grid_df: pd.DataFrame,
    clutter_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    center_lat: float,
    center_lon: float,
    tx_height_m: float,
    rx_height_m: float,
    freq_mhz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identical physics to phase15._geo_correction_db (same de-duplicated
    single-obstruction-term logic Phase 17 already uses) - duplicated here,
    not imported, ONLY so the obstruction branch can be returned per point
    alongside the correction and the point's own clutter class, which the
    shared function's aggregated counts dict can't give us without changing
    its return signature (and Phase 15/16/17 must stay untouched)."""
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
        diffraction_loss, n_obstacles = phase15._multi_ray_diffraction_loss_db(
            center_lat, center_lon, site_pt, lat_i, lon_i, total_dist_m,
            buildings_gdf, sindex, tx_height_m, rx_height_m, wavelength_m,
        )
        if n_obstacles > 0:
            correction[i] = env_adj - diffraction_loss * 1.0
            branch[i] = "obstructed"
        else:
            proxy = clutter_weights.get(cls, 0.0) if cls else 0.0
            correction[i] = env_adj + proxy + (float(bar) if pd.notna(bar) else 0.0) * building_area_weight
            branch[i] = "clear"

    return correction, branch, clutter_classes


def _compute_phase19_candidates(
    candidates: pd.DataFrame, identity: pd.DataFrame, clutter_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame, bias_table: pd.DataFrame, technology: str,
) -> pd.DataFrame:
    candidates = candidates.merge(
        identity[["Node_Cell_ID", "Etilt", "Mtilt", "Height", "tx_power"]],
        left_on="strict_cell_key", right_on="Node_Cell_ID", how="left",
    )
    candidates["physical_rsrp"] = np.nan
    candidates["geo_correction_db"] = 0.0
    candidates["obstruction_branch"] = "unknown"
    candidates["clutter_class"] = None
    candidates["bias_db"] = 0.0

    params_common = {"ue_height": 1.5, "k1": 0, "k2": 0, "cable_loss": 2.0, "antenna_gain": 18.0}
    bias_lookup = bias_table[bias_table["technology"] == technology].set_index(
        ["clutter_class", "obstruction_branch"]
    )["bias_db"]

    n_cells = candidates["strict_cell_key"].nunique()
    for idx, (cell_key, group) in enumerate(candidates.groupby("strict_cell_key", dropna=False)):
        row0 = group.iloc[0]
        if pd.isna(row0.get("Etilt")):
            continue
        site_row = pd.Series({
            "lat": row0["lat"], "lon": row0["lon"], "azimuth": row0["azimuth"],
            "Etilt": row0["Etilt"], "Mtilt": row0["Mtilt"], "Height": row0["Height"], "tx_power": row0["tx_power"],
        })
        site_dict = phase15._row_to_site_dict_fixed(site_row)
        freq = float(row0["frequency_mhz"])
        tx_height_m = float(row0["Height"]) if pd.notna(row0["Height"]) else 30.0
        center_lat, center_lon = float(row0["lat"]), float(row0["lon"])

        grid_lats = group["lat"].to_numpy(dtype=float)
        grid_lons = group["lon"].to_numpy(dtype=float)
        physical = np.array(
            [phase15.compute_sector_rsrp(site_dict, la, lo, freq, params_common) for la, lo in zip(grid_lats, grid_lons)],
            dtype=float,
        )
        if str(row0["band"]) == "78":
            physical = physical + N78_TECHNOLOGY_OFFSET_DB

        grid_df = pd.DataFrame({"lat": grid_lats, "lon": grid_lons})
        correction, branch, cls = _geo_correction_with_branch(
            grid_df, clutter_gdf, buildings_gdf, center_lat, center_lon,
            tx_height_m=tx_height_m, rx_height_m=1.5, freq_mhz=freq,
        )
        bias = np.array([
            float(bias_lookup.get((c, b), 0.0)) if c else 0.0 for c, b in zip(cls, branch)
        ], dtype=float)

        candidates.loc[group.index, "physical_rsrp"] = np.clip(physical + correction, RSRP_MIN, RSRP_MAX)
        candidates.loc[group.index, "geo_correction_db"] = correction
        candidates.loc[group.index, "obstruction_branch"] = branch
        candidates.loc[group.index, "clutter_class"] = cls
        candidates.loc[group.index, "bias_db"] = bias
        if (idx + 1) % 10 == 0 or idx == n_cells - 1:
            print(f"[PHASE19] geo-corrected+branch-biased cells {idx + 1}/{n_cells} ({len(group)} points)", flush=True)

    missing = candidates["physical_rsrp"].isna()
    if missing.any():
        candidates.loc[missing, "physical_rsrp"] = np.clip(candidates.loc[missing, "raw_cost231_rsrp"], RSRP_MIN, RSRP_MAX)
        print(f"[PHASE19] {int(missing.sum())} candidate rows had no identity match - fell back to raw_cost231_rsrp only")

    candidates["phase19_rsrp_no_lock"] = np.clip(
        candidates["physical_rsrp"] + candidates["bias_db"], RSRP_MIN, RSRP_MAX
    )
    lock = candidates["dt_replaced"].fillna(False).astype(bool) if "dt_replaced" in candidates.columns else pd.Series(False, index=candidates.index)
    candidates["candidate_phase19_rsrp"] = np.where(
        lock, candidates.get("dt_replacement_rsrp", np.nan), candidates["phase19_rsrp_no_lock"]
    )
    candidates["candidate_phase19_rsrp"] = np.clip(candidates["candidate_phase19_rsrp"].astype(float), RSRP_MIN, RSRP_MAX)
    return candidates


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bias_table = _load_bias_table()

    serving_all = phase17._build_serving_grid()
    surface_all = pd.read_parquet(phase17.PHASE9_DIR / "phase9_directional_raw_corrected_surface_project210.parquet")
    identity = phase13.load_identity()
    clutter_gdf, buildings_gdf = phase17._load_clutter_and_buildings()

    summary = {}
    for technology in ["4G", "5G"]:
        print(f"\n[PHASE19] ==== {technology} ====")
        serving = serving_all[serving_all["technology"] == technology].copy().reset_index(drop=True)
        candidates = surface_all[surface_all["technology"] == technology].copy().reset_index(drop=True)

        candidates = _compute_phase19_candidates(candidates, identity, clutter_gdf, buildings_gdf, bias_table, technology)

        agg = candidates.groupby("grid_id")["candidate_phase19_rsrp"].agg(["max", "mean"]).reset_index()
        agg = agg.rename(columns={"max": "phase19_rsrp_agg", "mean": "phase19_frontend_mean_rsrp"})

        winner_detail = serving[["grid_id", "strict_cell_key"]].merge(
            candidates[["grid_id", "strict_cell_key", "physical_rsrp", "geo_correction_db", "bias_db", "obstruction_branch"]],
            on=["grid_id", "strict_cell_key"], how="left",
        )
        serving = serving.merge(
            winner_detail[["grid_id", "physical_rsrp", "geo_correction_db", "bias_db", "obstruction_branch"]],
            on="grid_id", how="left",
        )
        serving = serving.merge(agg, on="grid_id", how="left")
        serving["phase19_rsrp"] = np.clip(serving["phase19_rsrp_agg"].astype(float), RSRP_MIN, RSRP_MAX)
        serving["phase19_frontend_mean_rsrp"] = np.clip(serving["phase19_frontend_mean_rsrp"].astype(float), RSRP_MIN, RSRP_MAX)

        out_path = OUT_DIR / f"phase19_serving_grid_{technology.lower()}_project210.parquet"
        serving.to_parquet(out_path, index=False)
        serving.to_csv(out_path.with_suffix(".csv"), index=False)
        print(f"[PHASE19] wrote {out_path} ({len(serving)} rows)")

        summary[technology] = {
            "grid_rows": int(len(serving)),
            "mean_corrected_rsrp_phase9": float(serving["corrected_rsrp"].mean()),
            "mean_phase19_rsrp": float(serving["phase19_rsrp"].mean()),
            "mean_phase19_frontend_mean_rsrp": float(serving["phase19_frontend_mean_rsrp"].mean()),
            "mean_geo_correction_db": float(serving["geo_correction_db"].mean()),
            "mean_bias_db": float(serving["bias_db"].mean()),
            "branch_share": serving["obstruction_branch"].value_counts(normalize=True).round(3).to_dict(),
            "mean_frontend_mean_rsrp_phase9": float(serving["frontend_mean_rsrp"].mean()),
            "mean_phase19_frontend_vs_serving_gap_db": float((serving["phase19_frontend_mean_rsrp"] - serving["phase19_rsrp"]).mean()),
        }

    (OUT_DIR / "phase19_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
    print("\n[PHASE19] summary:")
    print(pd.Series(summary).to_json(indent=2))


if __name__ == "__main__":
    main()
