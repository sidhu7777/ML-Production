"""
Phase 17: full-polygon comparison of Phase 9 (production flat-offset
baseline) vs. a new geo-corrected + representative-DT-residual surface,
built on Phase 9's own serving grid (same grid_id per technology, same
frontend-shaped output) so the two are directly comparable.

Architecture (as agreed):
  Physical_RSRP = tilt-fixed COST231 antenna model
                + ONE non-overlapping path-obstruction term per point:
                  - indoor (target inside a building): real depth-scaled
                    building entry loss only
                  - path crosses a building (target outdoor): real
                    per-path knife-edge diffraction only (diminishing-
                    weight combined across every building actually
                    crossed, multi-ray averaged)
                  - no building geometry found on the path at all: the
                    coarse clutter-class/footprint weight, as a fallback
                    only (never stacked on top of the real geometric terms
                    above - that was the earlier double-counting bug)
                + independent environmental adjustment (Water/Vegetation/
                  Rural-Open clutter weight - a separate physical effect
                  from building obstruction, so always applies)
  5G (n78) also gets the -2.58dB technology offset established early in
  this project's baseline work.
  DT residual is then IDW-interpolated ONLY from DT points whose OWN
  clutter class has enough real DT samples to be representative (>=8,
  same threshold Phase 12 used), AND only applied to target cells whose
  OWN clutter class is also representative - never extrapolated into
  clutter classes/locations DT never actually measured (e.g. a building
  25m from a representative road DT point does not borrow that road's
  residual). Phase 9's own DT-pixel-replacement is preserved as-is for
  grid cells it already locks.

Does not touch ML/tests/baseline or production - reuses their functions
read-only. 4G and 5G are computed and saved fully separately throughout.
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
from test_project210_cost231_phase11_12_residual_blending import (  # reused, not reimplemented
    _idw_residual,
    _serving_by_technology,
)

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
OUT_DIR = PROJECT_DIR / "cost231_phase17_geo_dt_comparison"
BASELINE_DATA_DIR = BASELINE_DIR / "data" / "project_210_taiwan"
CLUTTER_TILES_PATH = BASELINE_DATA_DIR / "clutter_tiles_final_v2.geojson"

N78_TECHNOLOGY_OFFSET_DB = -2.58  # established Phase 2/3 baseline logic for Taiwan 5G n78
MIN_DT_FOR_REPRESENTATIVE_CLASS = 8
RSRP_MIN, RSRP_MAX = -147.0, -44.0


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_clutter_and_buildings():
    # Same whole-polygon files Phase 15/16 already load (via their own
    # st.cache_data-wrapped loaders) - called directly here since this is a
    # plain batch script, not a Streamlit page.
    clutter_gdf = gpd.read_file(CLUTTER_TILES_PATH)
    buildings_gdf = phase15.load_building_gdf(BASELINE_DATA_DIR)
    buildings_gdf = phase15.impute_building_heights(buildings_gdf, phase15.OBAT_CSV_PATH)
    return clutter_gdf, buildings_gdf


def _classify_dt_clutter(dt: pd.DataFrame, clutter_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    points = gpd.GeoDataFrame(
        {"row_idx": range(len(dt))}, geometry=gpd.points_from_xy(dt["lon"], dt["lat"]), crs="EPSG:4326"
    )
    joined = gpd.sjoin(points, clutter_gdf[["clutter_class", "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates(subset="row_idx").sort_values("row_idx")
    dt = dt.copy()
    dt["clutter_class"] = joined.set_index("row_idx").reindex(range(len(dt)))["clutter_class"].to_numpy()
    return dt


def _representative_classes(dt: pd.DataFrame, technology: str) -> set:
    sub = dt[dt["assigned_technology"] == technology]
    counts = sub.groupby("clutter_class")["id"].count()
    return set(counts[counts >= MIN_DT_FOR_REPRESENTATIVE_CLASS].index)


def _compute_physical_and_geo_all_candidates(
    candidates: pd.DataFrame, identity: pd.DataFrame, clutter_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Same geo-correction as before, but run for EVERY candidate cell of
    every grid_id (not just the single winning serving cell), so both a
    best-server (max) and a frontend-style (mean) aggregate can be derived
    afterward - matching how Phase 9's own frontend_mean_rsrp is computed
    (mean across all real candidates), instead of only ever having a mean
    version of the OLD baseline and never of the new geo-corrected surface."""
    candidates = candidates.merge(
        identity[["Node_Cell_ID", "Etilt", "Mtilt", "Height", "tx_power"]],
        left_on="strict_cell_key", right_on="Node_Cell_ID", how="left",
    )
    candidates["physical_rsrp"] = np.nan
    candidates["geo_correction_db"] = 0.0

    params_common = {"ue_height": 1.5, "k1": 0, "k2": 0, "cable_loss": 2.0, "antenna_gain": 18.0}
    clutter_weights = dict(phase15.DEFAULT_CLUTTER_WEIGHTS)

    n_cells = candidates["strict_cell_key"].nunique()
    for idx, (cell_key, group) in enumerate(candidates.groupby("strict_cell_key", dropna=False)):
        row0 = group.iloc[0]
        if pd.isna(row0.get("Etilt")):
            continue  # no identity match, skip (kept as NaN, filled from raw_cost231 fallback below)
        site_row = pd.Series(
            {
                "lat": row0["lat"], "lon": row0["lon"], "azimuth": row0["azimuth"],
                "Etilt": row0["Etilt"], "Mtilt": row0["Mtilt"], "Height": row0["Height"],
                "tx_power": row0["tx_power"],
            }
        )
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
        correction, counts = phase15._geo_correction_db(
            grid_df, clutter_gdf, buildings_gdf, center_lat, center_lon,
            tx_height_m=tx_height_m, rx_height_m=1.5, freq_mhz=freq,
            clutter_weights=clutter_weights, building_area_weight=phase15.DEFAULT_BUILDING_AREA_WEIGHT,
            diffraction_multiplier=1.0, entry_loss_db=-15.0, entry_depth_slope_db_per_m=-0.5,
        )
        candidates.loc[group.index, "physical_rsrp"] = np.clip(physical + correction, RSRP_MIN, RSRP_MAX)
        candidates.loc[group.index, "geo_correction_db"] = correction
        if (idx + 1) % 10 == 0 or idx == n_cells - 1:
            print(f"[PHASE17] geo-corrected cells {idx + 1}/{n_cells} (all-candidates pass, {len(group)} points this cell)", flush=True)

    missing = candidates["physical_rsrp"].isna()
    if missing.any():
        candidates.loc[missing, "physical_rsrp"] = np.clip(candidates.loc[missing, "raw_cost231_rsrp"], RSRP_MIN, RSRP_MAX)
        print(f"[PHASE17] {int(missing.sum())} candidate rows had no identity match - fell back to raw_cost231_rsrp only")
    return candidates


def _dt_residual_by_grid_id(
    grid_ids: pd.DataFrame, dt: pd.DataFrame, technology: str, clutter_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """DT residual only ever depends on the TARGET location (grid_id's own
    lat/lon), never on which candidate cell is serving it - computed once
    per unique grid_id here, then broadcast to every candidate row for that
    grid_id, instead of recomputing the same IDW interpolation redundantly
    per candidate.

    DT residual is a CALIBRATION of the physical model for environments DT
    actually measured - it must not be carried into a target cell whose OWN
    clutter class isn't in that representative set, e.g. a building 25m
    from a road DT point borrowing that road's residual just because IDW
    distance-weighting alone would blend it in. So after interpolating,
    any target cell whose own clutter class is not representative for this
    technology is reset to a zero residual (physical model only)."""
    representative = _representative_classes(dt, technology)
    print(f"[PHASE17] {technology} representative clutter classes (>= {MIN_DT_FOR_REPRESENTATIVE_CLASS} DT points): {sorted(representative)}")
    dt_tech = dt[(dt["assigned_technology"] == technology) & dt["clutter_class"].isin(representative)].dropna(
        subset=["lat", "lon", "dt_minus_cost231_db"]
    )
    print(f"[PHASE17] {technology} representative DT points used as residual source: {len(dt_tech)} / {(dt['assigned_technology'] == technology).sum()}")
    out = grid_ids.drop_duplicates(subset=["grid_id"])[["grid_id", "lat", "lon"]].copy()
    out["dt_residual_db"] = 0.0
    if not dt_tech.empty:
        out["dt_residual_db"] = _idw_residual(
            out["lat"].to_numpy(dtype=float), out["lon"].to_numpy(dtype=float),
            dt_tech["lat"].to_numpy(dtype=float), dt_tech["lon"].to_numpy(dtype=float),
            dt_tech["dt_minus_cost231_db"].to_numpy(dtype=float),
            decay_m=350.0, max_distance_m=1200.0, k=16,
        )

    target_clutter = phase15._lookup_clutter(out[["lat", "lon"]].reset_index(drop=True), clutter_gdf)
    target_cls = target_clutter["clutter_class"].to_numpy()
    not_representative = ~pd.Series(target_cls).isin(representative).to_numpy()
    n_gated = int(not_representative.sum())
    out.loc[not_representative, "dt_residual_db"] = 0.0
    print(
        f"[PHASE17] {technology} DT residual gated to 0 (target cell's own clutter class not "
        f"representative) for {n_gated} / {len(out)} grid cells"
    )
    return out[["grid_id", "dt_residual_db"]]


def _aggregate_candidates(candidates: pd.DataFrame, dt_residual_by_grid: pd.DataFrame) -> pd.DataFrame:
    """From every candidate cell's own physical+geo value, derive both the
    best-server (max) and frontend-style (mean) aggregate per grid_id, using
    the SAME per-candidate DT residual and DT lock as the single-cell path -
    so 'Phase 17 frontend-mean' is a real mean of fully-corrected candidate
    values, not a mean of just the raw physical prediction."""
    candidates = candidates.merge(dt_residual_by_grid, on="grid_id", how="left")
    candidates["dt_residual_db"] = candidates["dt_residual_db"].fillna(0.0)
    candidates["candidate_rsrp_no_lock"] = np.clip(
        candidates["physical_rsrp"] + candidates["dt_residual_db"], RSRP_MIN, RSRP_MAX
    )
    lock = candidates["dt_replaced"].fillna(False).astype(bool) if "dt_replaced" in candidates.columns else pd.Series(False, index=candidates.index)
    candidates["candidate_phase17_rsrp"] = np.where(
        lock, candidates.get("dt_replacement_rsrp", np.nan), candidates["candidate_rsrp_no_lock"]
    )
    candidates["candidate_phase17_rsrp"] = np.clip(candidates["candidate_phase17_rsrp"].astype(float), RSRP_MIN, RSRP_MAX)

    agg = candidates.groupby("grid_id")["candidate_phase17_rsrp"].agg(["max", "mean"]).reset_index()
    agg = agg.rename(columns={"max": "phase17_rsrp_agg", "mean": "phase17_frontend_mean_rsrp"})
    return agg


def _build_serving_grid() -> pd.DataFrame:
    """Phase 9's own 'directional_serving_grid' file assigns exactly ONE
    technology per grid_id (mutually exclusive - confirmed 10,234 grid
    tiles split into 6,923 4G + 3,311 5G, zero overlap), which understates
    real coverage since a location can genuinely have both 4G and 5G.
    Re-derive it the way Phase 11/12 already does correctly:
    group by [technology, grid_id] independently, so each technology gets
    its own full pass over all 10,234 grid cells."""
    surface = pd.read_parquet(PHASE9_DIR / f"phase9_directional_raw_corrected_surface_project210.parquet")
    grid = pd.read_parquet(PHASE9_DIR / f"phase9_gridanalytics_compatible_grid_project210.parquet")
    serving = _serving_by_technology(surface, grid, "corrected_rsrp")
    extra_cols = surface[["technology", "grid_id", "strict_cell_key", "azimuth", "frequency_mhz"]].drop_duplicates(
        subset=["technology", "grid_id", "strict_cell_key"]
    )
    serving = serving.merge(extra_cols, on=["technology", "grid_id", "strict_cell_key"], how="left")
    serving["lat"] = serving["center_lat"]
    serving["lon"] = serving["center_lon"]

    # "Frontend" value: the production map's current default aggregation
    # (lteGridAggregationMethod = "mean" in UnifiedMapView.jsx) - the MEAN
    # of every candidate cell's corrected_rsrp assigned to this grid_id, not
    # just the single best server. This is deliberately kept alongside the
    # serving-cell (best server, idxmax) value above so the two can be shown
    # stacked and compared directly - this is the same "mean vs max" root
    # cause diagnosed early in this project's debugging, now on real data.
    frontend_mean = (
        surface.groupby(["technology", "grid_id"], dropna=False)["corrected_rsrp"]
        .mean()
        .rename("frontend_mean_rsrp")
        .reset_index()
    )
    frontend_candidates = (
        surface.groupby(["technology", "grid_id"], dropna=False)["strict_cell_key"]
        .nunique()
        .rename("frontend_candidate_count")
        .reset_index()
    )
    serving = serving.merge(frontend_mean, on=["technology", "grid_id"], how="left")
    serving = serving.merge(frontend_candidates, on=["technology", "grid_id"], how="left")
    return serving


def main() -> None:
    _ensure_dirs()
    serving_all = _build_serving_grid()
    surface_all = pd.read_parquet(PHASE9_DIR / "phase9_directional_raw_corrected_surface_project210.parquet")
    # phase9_dt_match_project210.parquet's own 5G rows are proximity-mislabeled
    # LTE-anchor readings (136 of them) - the real network='5G NSA'/'5G SA' DT
    # data (8,033 rows, confirmed correctly inside the project polygon in the
    # live Taiwan DB) was being silently dropped upstream by
    # fetch_project_propagation_cache.py's _network_like_4g() filter. Phase 20
    # (test_project210_phase20_5g_real_dt_match.py) fetched the real rows
    # directly from the DB and merged them with the (always-correct) 4G rows
    # from this same file - use that corrected file here so the fix flows
    # through Phase 17/18/19 without duplicating this pipeline.
    corrected_dt_path = PROJECT_DIR / "cost231_phase20_5g_real_dt_match" / "phase9_dt_match_project210_corrected.parquet"
    dt_source_path = corrected_dt_path if corrected_dt_path.exists() else PHASE9_DIR / "phase9_dt_match_project210.parquet"
    print(f"[PHASE17] DT source: {dt_source_path.name}")
    dt_all = pd.read_parquet(dt_source_path)
    identity = phase13.load_identity()
    clutter_gdf, buildings_gdf = _load_clutter_and_buildings()
    dt_all = _classify_dt_clutter(dt_all, clutter_gdf)
    print(f"[PHASE17] serving grid rows: {len(serving_all)}  surface (all candidates) rows: {len(surface_all)}  DT rows: {len(dt_all)}")

    summary = {}
    for technology in ["4G", "5G"]:
        print(f"\n[PHASE17] ==== {technology} ====")
        serving = serving_all[serving_all["technology"] == technology].copy().reset_index(drop=True)
        candidates = surface_all[surface_all["technology"] == technology].copy().reset_index(drop=True)

        candidates = _compute_physical_and_geo_all_candidates(candidates, identity, clutter_gdf, buildings_gdf)
        dt_res_by_grid = _dt_residual_by_grid_id(candidates[["grid_id", "lat", "lon"]], dt_all, technology, clutter_gdf)
        agg = _aggregate_candidates(candidates, dt_res_by_grid)

        # pull the winning candidate's own physical/geo/residual values too,
        # for display/summary consistency with the pre-aggregation numbers
        winner_detail = candidates.merge(dt_res_by_grid, on="grid_id", how="left")
        winner_detail = serving[["grid_id", "strict_cell_key"]].merge(
            winner_detail[["grid_id", "strict_cell_key", "physical_rsrp", "geo_correction_db", "dt_residual_db"]],
            on=["grid_id", "strict_cell_key"], how="left",
        )
        serving = serving.merge(winner_detail[["grid_id", "physical_rsrp", "geo_correction_db", "dt_residual_db"]], on="grid_id", how="left")
        serving = serving.merge(agg, on="grid_id", how="left")
        serving["phase17_rsrp"] = np.clip(serving["phase17_rsrp_agg"].astype(float), RSRP_MIN, RSRP_MAX)
        serving["phase17_frontend_mean_rsrp"] = np.clip(serving["phase17_frontend_mean_rsrp"].astype(float), RSRP_MIN, RSRP_MAX)

        out_path = OUT_DIR / f"phase17_serving_grid_{technology.lower()}_project210.parquet"
        serving.to_parquet(out_path, index=False)
        serving.to_csv(out_path.with_suffix(".csv"), index=False)
        print(f"[PHASE17] wrote {out_path} ({len(serving)} rows)")

        summary[technology] = {
            "grid_rows": int(len(serving)),
            "mean_corrected_rsrp_phase9": float(serving["corrected_rsrp"].mean()),
            "mean_phase17_rsrp": float(serving["phase17_rsrp"].mean()),
            "mean_phase17_frontend_mean_rsrp": float(serving["phase17_frontend_mean_rsrp"].mean()),
            "mean_geo_correction_db": float(serving["geo_correction_db"].mean()),
            "mean_dt_residual_db": float(serving["dt_residual_db"].mean()),
            "dt_locked_rows": int(serving["dt_replaced"].fillna(False).astype(bool).sum()),
            "mean_frontend_mean_rsrp": float(serving["frontend_mean_rsrp"].mean()),
            "mean_frontend_vs_serving_gap_db": float((serving["frontend_mean_rsrp"] - serving["corrected_rsrp"]).mean()),
            "mean_phase17_frontend_vs_serving_gap_db": float((serving["phase17_frontend_mean_rsrp"] - serving["phase17_rsrp"]).mean()),
            "mean_candidates_per_grid": float(serving["frontend_candidate_count"].mean()),
        }

    dt_all.to_parquet(OUT_DIR / "phase17_dt_with_clutter_project210.parquet", index=False)
    (OUT_DIR / "phase17_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
    print("\n[PHASE17] summary:")
    print(pd.Series(summary).to_json(indent=2))


if __name__ == "__main__":
    main()
