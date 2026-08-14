"""
Project-wide (ALL real sites/sectors) pipeline trace - complement to
test_single_site_pipeline.py's single-site debug trace.

Why this exists (from this session's investigation): single-site scoping was
correct for fast antenna/coverage-shape debugging (Step 1), but is the wrong
scope for calibration validation (Step 2) - real production propagation-
model calibration is always done "market by market"/project-wide, never per
site, specifically because (a) with only one site's grid as the match
target, every real drive-test point in the whole project gets force-matched
onto that one site regardless of which real site actually served it, and
(b) a single site only exposes a handful of clutter classes with too few
real samples each to fit/validate properly. Running the whole project fixes
both: real DT points nearest-match their own real serving site's grid, and
every clutter class gets a real, adequately-sized sample.

Uses the REAL project polygon (not the single-site debug's generous bounding
box - that override existed specifically to avoid clipping one site's
circular antenna pattern to the survey boundary; for the whole project,
clipping predictions to the real survey polygon is correct, standard
production behavior) and the real production default radius (500m per
site/sector, per tools/lte_prediction/routes.py's default), not the 2500m
single-site debug radius.

Test-case only - tools/lte_prediction/ is not touched. Local cache only -
no DB/production writes.

Usage:
    python tests/baseline/run_project_wide_trace.py --project-id 210 --region taiwan
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from tools.lte_prediction.ml_engine import engine, run_rf_prediction_fast
from tools.lte_prediction.geo_correction_pipeline import (
    normalize_site_for_geo,
    load_geo_weights,
    align_project_polygon_to_points,
    building_df_to_gdf,
    align_building_geometries_to_project,
    _enrich_buildings_with_osm_heights,
    _osm_enrichment_enabled,
    create_analysis_grid,
    attach_building_features,
    _attach_osm_context_features,
    build_grid_feature_frame,
    augment_grid_with_advanced_geo_features,
    assign_points_to_tiles,
    _attach_missing_grid_features_by_grid_id,
    attach_fixed_serving_sinr_rsrq_proxy,
    apply_experimental_geo_adjustments,
    split_drive_train_holdout,
    evaluate_geo_against_dt,
    fit_dt_holdout_calibration,
    apply_dt_holdout_calibration,
    preserve_calibrated_kpis,
    DEFAULT_TILE_SIZE_M,
)
# Reuse the exact same test-case helpers/metrics as the single-site trace -
# not reimplemented, so both traces are guaranteed consistent.
from tests.baseline.test_single_site_pipeline import (
    haversine_m, bearing_deg, off_axis_deg,
    detect_indoor_loss, compute_sector_rsrp_unclipped,
    attach_pred_cols_to_dt_points, fit_spm_style_clutter_offsets, apply_spm_style_stage2,
    build_dt_calibration_features_no_clutter, STAGE3_FEATURE_COLS,
    fit_stage3_rewired, apply_stage3_rewired,
    mae_rmse, binned_mae_rmse, RSRP_BIN_LABELS,
)

DEFAULT_RADIUS_M = 500.0  # real production default (routes.py), not the 2500m single-site debug radius
MAX_POINTS_PER_SECTOR_FOR_MAP = 300  # downsampled, for the combined dashboard map only - not used for any metric


def downsample_for_map(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].head(max_points)


def run_project_wide_trace(project_id: int, region: str, data_dir: Path, output_path: Path,
                            radius_m: float = DEFAULT_RADIUS_M, grid_resolution_m: float = 25.0):
    print(f"[PWTRACE] loading FULL site/drive/building data from local cache: {data_dir}")
    site_df = pd.read_csv(data_dir / "site_df.csv", low_memory=False)
    drive_cache_path = data_dir / "drive_df_n78_fixed.csv"
    if drive_cache_path.exists():
        drive_df = pd.read_csv(drive_cache_path, low_memory=False)
        print(f"[PWTRACE] using n78-fixed drive cache: {len(drive_df)} rows, "
              f"band_source={drive_df['band_source'].value_counts().to_dict()}")
    else:
        drive_df = pd.read_csv(data_dir / "drive_df.csv", low_memory=False)
    building_df = pd.read_csv(data_dir / "building_df.csv", low_memory=False)
    print(f"[PWTRACE] site_df={len(site_df)} (ALL sites, not filtered) drive_df={len(drive_df)} building_df={len(building_df)}")

    # ---- REAL project polygon, not the single-site debug's bounding-box
    # override. Clipping the full-project prediction to the real survey
    # polygon is correct, standard production behaviour here - unlike the
    # single-site case, there's no risk of chopping one site's antenna
    # pattern, because with all real sites present, coverage right up to the
    # polygon edge is still contributed by whichever real neighbouring site
    # actually reaches there.
    #
    # IMPORTANT: align_project_polygon_to_points must run BEFORE Stage 1,
    # not after - a first attempt passed the RAW (unaligned) polygon's WKT
    # into run_rf_prediction_fast's params and it silently clipped away
    # ALL 140,742 points (polygon_removed=140742), because this project's
    # real polygon needs an x/y swap to align with the real site
    # coordinates (confirmed: polygon_alignment=swapped_xy,
    # swapped_hits=112/112) - exactly the same real coordinate-convention
    # quirk already known from the single-site trace's building/polygon
    # alignment, just not corrected yet at the point Stage 1 ran. ----
    real_polygon_gdf = gpd.read_file(data_dir / "project_polygon.geojson")
    if real_polygon_gdf.crs is None:
        real_polygon_gdf = real_polygon_gdf.set_crs("EPSG:4326")
    else:
        real_polygon_gdf = real_polygon_gdf.to_crs("EPSG:4326")
    if real_polygon_gdf.geometry.name != "geometry":
        real_polygon_gdf = real_polygon_gdf.rename_geometry("geometry")

    site_norm = normalize_site_for_geo(site_df)
    polygon_gdf, polygon_alignment = align_project_polygon_to_points(real_polygon_gdf[["geometry"]], site_norm)
    print(f"[PWTRACE] polygon_alignment={polygon_alignment} (aligned BEFORE Stage 1 this time)")
    real_polygon_wkt = polygon_gdf.geometry.iloc[0].wkt

    params = {
        "project_id": project_id, "region": region, "radius": radius_m, "grid": grid_resolution_m,
        "workers": 8, "max_interference_sites": 10,
        "use_frontend_grid_sampling": False,
        "samples_per_grid_axis": 3, "max_cells_per_grid": 3, "min_cells_per_grid": 1,
        "ensure_all_cells": True, "min_grids_per_cell": 1,
        "min_candidate_rsrp_dbm": -128, "candidate_safety_cap": 20,
        "polygon_wkt": real_polygon_wkt,
    }

    # ---- STAGE 1: COST-231 raw, real production function, ALL sites ----
    print("[PWTRACE] STAGE 1: running COST-231 (run_rf_prediction_fast) for ALL sites/sectors...")
    raw_pred_df = run_rf_prediction_fast(site_df, drive_df, building_df, params)
    print(f"[PWTRACE] STAGE 1 done: {len(raw_pred_df)} points across all sites, "
          f"rsrp_range=({raw_pred_df['pred_rsrp'].min():.2f},{raw_pred_df['pred_rsrp'].max():.2f})")

    pred_df = raw_pred_df.copy()
    pred_df["stage1_raw_rsrp"] = pred_df["pred_rsrp"]

    current_engine = engine.get(region.lower(), engine["india"])
    weights, weights_summary = load_geo_weights(project_id=project_id, weights_path=None)
    print(f"[PWTRACE] geo weights source={weights_summary}")

    osm_enabled = _osm_enrichment_enabled(params)
    building_gdf = building_df_to_gdf(building_df)
    building_gdf, building_alignment = align_building_geometries_to_project(building_gdf, polygon_gdf)
    print(f"[PWTRACE] building_alignment={building_alignment} building_rows={len(building_gdf)}")
    building_gdf = _enrich_buildings_with_osm_heights(building_gdf, polygon_gdf, enabled=osm_enabled)
    building_geoms_list = list(building_gdf.geometry)
    building_sindex = building_gdf.sindex

    tile_size_m = float(params.get("tile_size_m") or max(float(params.get("grid", 25.0)), DEFAULT_TILE_SIZE_M))
    cluster_count = 5

    grid_gdf = create_analysis_grid(polygon_gdf, tile_size_m)
    grid_gdf = attach_building_features(grid_gdf, building_gdf)
    grid_gdf = _attach_osm_context_features(grid_gdf, polygon_gdf, enabled=osm_enabled)
    grid_df, _ = build_grid_feature_frame(grid_gdf, site_norm, cluster_count)
    grid_df, geo_status = augment_grid_with_advanced_geo_features(grid_df, building_gdf, site_norm, dem_raster_path=None)
    print(f"[PWTRACE] OLD production clutter distribution (whole project, for comparison only): {grid_df['clutter_class'].value_counts().to_dict()}")

    # ---- Same corrected clutter classification as the single-site trace, applied project-wide ----
    print("[PWTRACE] overriding clutter_class with corrected Overture+GHS-OBAT+cited-threshold classification (test-case only)...")
    from tests.baseline.compute_clutter_final_v2 import (
        GREEN_LC_SUBTYPES, GREEN_LU_SUBTYPES, GREEN_LU_CLASS,
        attach_building_features_fixed, attach_area_ratio, attach_road_length,
        impute_building_heights, attach_surrounding_height, classify as corrected_classify,
    )
    obat_csv = Path(r"C:\Users\PC\AppData\Local\Temp\claude\c--Users-PC-Desktop-S-Tracer-Exe-S-Tracer-Exe\bee613e1-06ad-457d-9fd7-900823b4b9ed\scratchpad\ghsobat_project210_bbox.csv")
    corrected_grid = attach_building_features_fixed(grid_gdf[["grid_id", "geometry", "cell_area_m2"]].copy(), building_gdf)
    roads = gpd.read_file(data_dir / "roads_segment.geojson")
    corrected_grid = attach_road_length(corrected_grid, roads)
    water = gpd.read_file(data_dir / "water.geojson")
    corrected_grid = attach_area_ratio(corrected_grid, water, "water_ratio")
    land_cover = gpd.read_file(data_dir / "land_cover.geojson")
    land_use = gpd.read_file(data_dir / "land_use.geojson")
    green_lc = land_cover[land_cover["subtype"].isin(GREEN_LC_SUBTYPES)]
    green_lu = land_use[land_use["subtype"].isin(GREEN_LU_SUBTYPES) | land_use["class"].isin(GREEN_LU_CLASS)]
    green = gpd.GeoDataFrame(pd.concat([green_lc, green_lu], ignore_index=True), crs="EPSG:4326")
    corrected_grid = attach_area_ratio(corrected_grid, green, "green_ratio")
    if obat_csv.exists():
        building_gdf_h = impute_building_heights(building_gdf, obat_csv)
        project_mean_height = float(building_gdf_h["height"].mean())
        surrounding_df = attach_surrounding_height(corrected_grid, building_gdf_h, radius_m=100.0)
        corrected_grid = corrected_grid.merge(surrounding_df, on="grid_id", how="left")
    else:
        print(f"[PWTRACE] GHS-OBAT csv not found at {obat_csv} - falling back to building presence only")
        corrected_grid["surrounding_height_m"] = np.nan
        project_mean_height = 20.0
    corrected_grid["clutter_class"] = corrected_grid.apply(lambda r: corrected_classify(r, project_mean_height), axis=1)
    print(f"[PWTRACE] CORRECTED clutter distribution (whole project): {corrected_grid['clutter_class'].value_counts().to_dict()}")
    grid_df = grid_df.drop(columns=["clutter_class"]).merge(corrected_grid[["grid_id", "clutter_class"]], on="grid_id", how="left")
    grid_gdf = grid_gdf.merge(grid_df[["grid_id", "clutter_class", "morphology_cluster"]], on="grid_id", how="left")

    pred_work = pred_df.copy()
    pred_work = assign_points_to_tiles(pred_work, grid_gdf)
    pred_work = _attach_missing_grid_features_by_grid_id(pred_work, grid_df)
    if "Node_Cell_ID" not in pred_work.columns and "node_cell_id" in pred_work.columns:
        pred_work["Node_Cell_ID"] = pred_work["node_cell_id"].astype(str)
    pred_work = attach_fixed_serving_sinr_rsrq_proxy(pred_work, site_norm)

    # ---- Whole-project real pre-clip Stage 1 RSRP (same real formula
    # production uses internally, same replica already validated against
    # production's real clipped output in the single-site trace) ----
    pred_work["_cell_id_key"] = pred_work["Node_Cell_ID"].astype(str).str.split("_").str[1]
    site_lookup = site_norm.set_index("cell_id")
    site_rf_cols = site_lookup[[
        "lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "antenna_height", "tx_power", "frequency_mhz",
    ]].rename(columns={
        "lat": "_s_lat", "lon": "_s_lon", "azimuth": "_s_az", "electrical_tilt": "_s_etilt",
        "mechanical_tilt": "_s_mtilt", "antenna_height": "_s_htx", "tx_power": "_s_txpwr", "frequency_mhz": "_s_freq",
    })
    pred_work = pred_work.merge(site_rf_cols, left_on="_cell_id_key", right_index=True, how="left")
    print(f"[PWTRACE] computing indoor loss for {len(pred_work)} points (whole project)...")
    indoor_losses_all = np.array([
        detect_indoor_loss(lat, lon, building_sindex, building_geoms_list)
        for lat, lon in zip(pred_work["lat"].astype(float), pred_work["lon"].astype(float))
    ])
    pred_work["stage1_raw_rsrp_unclipped"] = compute_sector_rsrp_unclipped(
        s_lat=pred_work["_s_lat"], s_lon=pred_work["_s_lon"], s_az=pred_work["_s_az"],
        s_etilt=pred_work["_s_etilt"], s_mtilt=pred_work["_s_mtilt"], s_htx=pred_work["_s_htx"],
        tx_pwr=pred_work["_s_txpwr"], freq=pred_work["_s_freq"],
        p_lat=pred_work["lat"].astype(float), p_lon=pred_work["lon"].astype(float),
        indoor_loss=indoor_losses_all,
    )

    # ---- STAGE 2: current, hand-tuned (real production function) ----
    print("[PWTRACE] STAGE 2: apply_experimental_geo_adjustments...")
    pred_work, geo_summary = apply_experimental_geo_adjustments(pred_work, weights=weights)
    pred_work["stage2_geo_rsrp"] = pred_work["pred_rsrp_geo"].astype(float).copy()
    print(f"[PWTRACE] STAGE 2 done: rsrp_range=({pred_work['stage2_geo_rsrp'].min():.2f},{pred_work['stage2_geo_rsrp'].max():.2f})")

    drive_train_df, drive_holdout_df, split_summary = split_drive_train_holdout(drive_df, validation_fraction=0.25)
    print(f"[PWTRACE] DT split: {split_summary}")
    train_eval, _, _ = evaluate_geo_against_dt(drive_train_df, pred_work)
    dt_calibration_models, dt_calibration_debug = fit_dt_holdout_calibration(train_eval)

    # ---- STAGE 2b: SPM-style, project-wide fit - real per-class sample
    # sizes now, and real DT points match their OWN real serving site's
    # grid (all 112 sites present), not forced onto one unrelated site. ----
    print("[PWTRACE] STAGE 2b: SPM-style clutter-only geo correction, project-wide fit (test-case only)...")
    dt_train_matched = attach_pred_cols_to_dt_points(
        drive_train_df, pred_work, ["stage1_raw_rsrp_unclipped", "clutter_class"],
    )
    clutter_offsets, offsets_debug = fit_spm_style_clutter_offsets(dt_train_matched)
    print(f"[PWTRACE] STAGE 2b fitted per-class offsets (dB, PROJECT-WIDE real DT train split, n={offsets_debug['matched_dt_rows_used']}): {offsets_debug['per_class']}")
    pred_work = apply_spm_style_stage2(pred_work, clutter_offsets, offsets_debug["global_mean_residual_db"])
    print(f"[PWTRACE] STAGE 2b done: rsrp_range=({pred_work['stage2b_spm_rsrp'].min():.2f},{pred_work['stage2b_spm_rsrp'].max():.2f})")

    holdout_matched = attach_pred_cols_to_dt_points(
        drive_holdout_df, pred_work, ["pred_rsrp", "stage2_geo_rsrp", "stage2b_spm_rsrp"],
    )
    mae_s1, rmse_s1 = mae_rmse(holdout_matched, "pred_rsrp")
    mae_s2, rmse_s2 = mae_rmse(holdout_matched, "stage2_geo_rsrp")
    mae_s2b, rmse_s2b = mae_rmse(holdout_matched, "stage2b_spm_rsrp")
    stage2b_validation = {
        "holdout_dt_rows_matched": int(len(holdout_matched.dropna(subset=["RSRP_meas"]))) if "RSRP_meas" in holdout_matched.columns else 0,
        "mae_stage1_raw_physics_only": mae_s1, "rmse_stage1_raw_physics_only": rmse_s1,
        "mae_stage2_current_hand_tuned_weights": mae_s2, "rmse_stage2_current_hand_tuned_weights": rmse_s2,
        "mae_stage2b_spm_style_clutter_only": mae_s2b, "rmse_stage2b_spm_style_clutter_only": rmse_s2b,
    }
    print(f"[PWTRACE] PROJECT-WIDE STAGE 2b MAE/RMSE validation: {stage2b_validation}")

    # ---- STAGE 3-rewired (test-case) + STAGE 3 (production, unchanged) ----
    print("[PWTRACE] STAGE 3-rewired: DT residual calibration on top of Stage 2b, project-wide (test-case only)...")
    train_eval_2b = attach_pred_cols_to_dt_points(drive_train_df, pred_work, ["stage2b_spm_rsrp"] + STAGE3_FEATURE_COLS)
    stage3_rewired_bundle, stage3_rewired_debug = fit_stage3_rewired(train_eval_2b, pred_col="stage2b_spm_rsrp")
    print(f"[PWTRACE] STAGE 3-rewired fit debug: {stage3_rewired_debug}")
    pred_work["stage3_rewired_rsrp"] = apply_stage3_rewired(pred_work, stage3_rewired_bundle)
    print(f"[PWTRACE] STAGE 3-rewired done: rsrp_range=({pred_work['stage3_rewired_rsrp'].min():.2f},{pred_work['stage3_rewired_rsrp'].max():.2f})")

    print("[PWTRACE] STAGE 3 (production, unchanged): apply_dt_holdout_calibration + preserve_calibrated_kpis...")
    pred_work = apply_dt_holdout_calibration(pred_work, dt_calibration_models)
    pred_work = preserve_calibrated_kpis(pred_work)
    print(f"[PWTRACE] STAGE 3 done: rsrp_range=({pred_work['pred_rsrp_calibrated'].min():.2f},{pred_work['pred_rsrp_calibrated'].max():.2f})")

    full_holdout_matched = attach_pred_cols_to_dt_points(
        drive_holdout_df, pred_work,
        ["pred_rsrp", "stage2_geo_rsrp", "stage2b_spm_rsrp", "stage3_rewired_rsrp", "pred_rsrp_calibrated"],
    )
    mae_f1, rmse_f1 = mae_rmse(full_holdout_matched, "pred_rsrp")
    mae_f2, rmse_f2 = mae_rmse(full_holdout_matched, "stage2_geo_rsrp")
    mae_f2b, rmse_f2b = mae_rmse(full_holdout_matched, "stage2b_spm_rsrp")
    mae_f3old, rmse_f3old = mae_rmse(full_holdout_matched, "pred_rsrp_calibrated")
    mae_f3r, rmse_f3r = mae_rmse(full_holdout_matched, "stage3_rewired_rsrp")
    stage3_validation = {
        "holdout_dt_rows_matched": int(len(full_holdout_matched.dropna(subset=["RSRP_meas"]))) if "RSRP_meas" in full_holdout_matched.columns else 0,
        "mae_stage1_raw_physics": mae_f1, "rmse_stage1_raw_physics": rmse_f1,
        "mae_stage2_current_hand_tuned": mae_f2, "rmse_stage2_current_hand_tuned": rmse_f2,
        "mae_stage2b_spm_style": mae_f2b, "rmse_stage2b_spm_style": rmse_f2b,
        "mae_stage3_old_calibrated_on_stage2": mae_f3old, "rmse_stage3_old_calibrated_on_stage2": rmse_f3old,
        "mae_stage3_rewired_calibrated_on_stage2b": mae_f3r, "rmse_stage3_rewired_calibrated_on_stage2b": rmse_f3r,
    }
    print(f"[PWTRACE] PROJECT-WIDE FULL STAGE VALIDATION: {stage3_validation}")

    stage3_validation_by_rsrp_bin = {
        "stage1_raw_physics": binned_mae_rmse(full_holdout_matched, "pred_rsrp"),
        "stage2_current_hand_tuned": binned_mae_rmse(full_holdout_matched, "stage2_geo_rsrp"),
        "stage2b_spm_style": binned_mae_rmse(full_holdout_matched, "stage2b_spm_rsrp"),
        "stage3_old_calibrated_on_stage2": binned_mae_rmse(full_holdout_matched, "pred_rsrp_calibrated"),
        "stage3_rewired_calibrated_on_stage2b": binned_mae_rmse(full_holdout_matched, "stage3_rewired_rsrp"),
    }
    print(f"[PWTRACE] PROJECT-WIDE VALIDATION BY RSRP BIN: {stage3_validation_by_rsrp_bin}")

    # ---- Raw arrays for the CDF plots (kept separately as .npy, not baked
    # into the JSON - they're too large/plain for that and the dashboard
    # only needs the PNGs, not to redraw these itself). ----
    cdf_dir = Path(__file__).parent / "output" / "cdf_graphs"
    cdf_dir.mkdir(parents=True, exist_ok=True)
    dt_all = drive_df.copy()
    rcol = next((c for c in dt_all.columns if "rsrp" in c.lower()), None)
    dt_rsrp_all = pd.to_numeric(dt_all[rcol], errors="coerce").dropna().to_numpy()
    baseline_full_grid_rsrp = pd.to_numeric(pred_work["stage1_raw_rsrp"], errors="coerce").dropna().to_numpy()
    baseline_at_dt_points = pd.to_numeric(full_holdout_matched["pred_rsrp"], errors="coerce").dropna().to_numpy()
    np.save(cdf_dir / "_raw_dt_measured_rsrp.npy", dt_rsrp_all)
    np.save(cdf_dir / "_raw_baseline_full_grid_rsrp.npy", baseline_full_grid_rsrp)
    np.save(cdf_dir / "_raw_baseline_at_dt_points_rsrp.npy", baseline_at_dt_points)
    print(f"[PWTRACE] CDF raw arrays saved to {cdf_dir}: "
          f"dt_measured n={len(dt_rsrp_all)}, baseline_full_grid n={len(baseline_full_grid_rsrp)}, "
          f"baseline_at_dt_points n={len(baseline_at_dt_points)}")

    # ---- Lightweight per-site/sector summary + downsampled points for the
    # new "combined project" dashboard page. NOT full per-point grids for
    # all 112 sites (that JSON would be gigabytes) - summary stats plus a
    # capped, evenly-spaced sample per sector, enough for a combined map. ----
    pred_work["_cell_id_key"] = pred_work["Node_Cell_ID"].astype(str).str.split("_").str[1]
    sites_summary = {}
    map_points = []
    for site_id, site_rows in site_norm.groupby("Site ID"):
        sectors_summary = {}
        for _, srow in site_rows.iterrows():
            cell_id = srow["cell_id"]
            sector_pts = pred_work[pred_work["_cell_id_key"] == cell_id].copy()
            if sector_pts.empty:
                continue
            site_lat, site_lon = float(srow["lat"]), float(srow["lon"])
            azimuth = float(srow["azimuth"]) if pd.notna(srow["azimuth"]) else 0.0
            sectors_summary[str(srow["sector"])] = {
                "cell_id": cell_id,
                "band": None if pd.isna(srow.get("band")) else (int(srow["band"]) if float(srow["band"]).is_integer() else float(srow["band"])),
                "azimuth": azimuth,
                "point_count": int(len(sector_pts)),
                "stage_summary": {
                    "stage1_raw_rsrp": {"mean": float(sector_pts["stage1_raw_rsrp"].mean()), "std": float(sector_pts["stage1_raw_rsrp"].std())},
                    "stage2_geo_rsrp": {"mean": float(sector_pts["stage2_geo_rsrp"].mean()), "std": float(sector_pts["stage2_geo_rsrp"].std())},
                    "stage2b_spm_rsrp": {"mean": float(sector_pts["stage2b_spm_rsrp"].mean()), "std": float(sector_pts["stage2b_spm_rsrp"].std())},
                    "stage3_rewired_rsrp": {"mean": float(sector_pts["stage3_rewired_rsrp"].mean()), "std": float(sector_pts["stage3_rewired_rsrp"].std())},
                    "pred_rsrp_calibrated": {"mean": float(sector_pts["pred_rsrp_calibrated"].mean()), "std": float(sector_pts["pred_rsrp_calibrated"].std())},
                },
                "clutter_distribution": sector_pts["clutter_class"].value_counts().to_dict() if "clutter_class" in sector_pts.columns else {},
            }
            sampled = downsample_for_map(sector_pts, MAX_POINTS_PER_SECTOR_FOR_MAP)
            for _, prow in sampled.iterrows():
                map_points.append({
                    "site_id": site_id, "cell_id": cell_id, "sector": str(srow["sector"]),
                    "lat": float(prow["lat"]), "lon": float(prow["lon"]),
                    "stage1_raw_rsrp": float(prow["stage1_raw_rsrp"]),
                    "stage2_geo_rsrp": float(prow["stage2_geo_rsrp"]),
                    "stage2b_spm_rsrp": float(prow["stage2b_spm_rsrp"]),
                    "stage3_rewired_rsrp": float(prow["stage3_rewired_rsrp"]),
                    "pred_rsrp_calibrated": float(prow["pred_rsrp_calibrated"]),
                    "clutter_class": prow.get("clutter_class"),
                })
        if sectors_summary:
            first = site_rows.iloc[0]
            sites_summary[site_id] = {
                "site_id": site_id, "lat": float(first["lat"]), "lon": float(first["lon"]),
                "sectors": sectors_summary,
            }
        print(f"[PWTRACE] site={site_id} sectors={len(sectors_summary)} done "
              f"({len(sites_summary)}/{site_norm['Site ID'].nunique()} sites)")

    output = {
        "project_id": project_id, "region": region,
        "scope": "project_wide_all_sites",
        "radius_m": radius_m, "grid_resolution_m": grid_resolution_m,
        "weights_summary": weights_summary,
        "clutter_distribution_corrected": grid_df["clutter_class"].value_counts().to_dict(),
        "stage2b_spm_offsets": offsets_debug,
        "stage2b_validation": stage2b_validation,
        "stage3_rewired_fit_debug": stage3_rewired_debug,
        "stage3_validation": stage3_validation,
        "stage3_validation_by_rsrp_bin": stage3_validation_by_rsrp_bin,
        "sites_summary": sites_summary,
        "map_points": map_points,
        "cdf_raw_arrays_dir": str(cdf_dir),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output), encoding="utf-8")
    print(f"[PWTRACE] wrote {output_path} ({output_path.stat().st_size} bytes), "
          f"{len(sites_summary)} sites, {len(map_points)} downsampled map points")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Project-wide (all sites) LTE prediction trace")
    parser.add_argument("--project-id", type=int, default=210)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    parser.add_argument("--grid-resolution-m", type=float, default=25.0)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "project_wide_trace.json")
    args = parser.parse_args(argv)
    run_project_wide_trace(
        project_id=args.project_id, region=args.region, data_dir=args.data_dir,
        output_path=args.output, radius_m=args.radius_m, grid_resolution_m=args.grid_resolution_m,
    )


if __name__ == "__main__":
    main()
