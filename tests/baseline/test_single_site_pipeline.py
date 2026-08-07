"""
Site -> sector, stage-by-stage pipeline trace for debugging prediction quality.

Runs the REAL production functions (imported directly from
tools.lte_prediction.ml_engine / geo_correction_pipeline - not a
reimplementation) for one project, then organizes every physical site's
sectors (site + sector + cell + band, matching how production actually
calculates - per site+sector+cell+band, not per whole site) so a dashboard
can pick a site, then one of its sectors, and inspect every stage the value
passes through:

  1. COST-231 raw          (run_rf_prediction_fast output, untouched)
  2. Geo-corrected          (apply_experimental_geo_adjustments output,
                             captured BEFORE DT calibration overwrites it -
                             the production code never persists this value
                             on its own, so it must be snapshotted here)
  3. DT-calibrated          (preserve_calibrated_kpis output)
  4. Smoothed / DT-blended  (apply_demo_dt_overlay output - what actually
                             gets saved as pred_rsrp_smoothed in production)

Each sector also carries its azimuth and the antenna's 3dB half-beamwidth
(65 degrees, the same constant COST-231's own antenna gain formula uses -
see grid_sampling.py:_antenna_gain_estimate) so a dashboard can draw the
true directional beam instead of treating the full 500m circular grid as
uniform coverage.

Writes one JSON file consumable by the companion Streamlit dashboard.

Usage:
    python tests/baseline/test_single_site_pipeline.py --project-id 210 --region taiwan
"""
from __future__ import annotations

import argparse
import json
import os
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

from tools.lte_prediction.ml_engine import (
    engine,
    fetch_site_data,
    fetch_drive_data,
    fetch_building_data,
    run_rf_prediction_fast,
    _resolve_prediction_polygons,
)
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
    apply_demo_dt_overlay,
    DEFAULT_TILE_SIZE_M,
)

ANTENNA_3DB_BEAMWIDTH_DEG = 65.0  # matches grid_sampling.py::_antenna_gain_estimate


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2r - lon1r
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def off_axis_deg(bearing, azimuth):
    return np.abs((bearing - azimuth + 180.0) % 360.0 - 180.0)


def run_trace(project_id: int, region: str, session_ids, operator, radius_m, grid_resolution_m,
              polygon_ids, output_path: Path, only_sites=None):
    print(f"[TRACE] fetching site/drive/building data for project_id={project_id} region={region}")
    site_df, resolved_operator = fetch_site_data(project_id, region=region, operator=operator, polygon_ids=polygon_ids)
    drive_df = fetch_drive_data(session_ids, resolved_operator, project_id, region=region)
    building_df = fetch_building_data(project_id, region=region)
    print(f"[TRACE] site_df={len(site_df)} drive_df={len(drive_df)} building_df={len(building_df)}")

    dem_path = None
    candidate_dem = PROJECT_ROOT / "data" / "dem" / f"project_{project_id}_dem.tif"
    if candidate_dem.exists():
        dem_path = str(candidate_dem)

    # These are EXACTLY the defaults routes.py uses for a real /api/lte-prediction/run
    # call (see ML/tools/lte_prediction/routes.py) - no overrides, so this runs the
    # same code path a real production job would, including the frontend-grid-sampling
    # attempt (which may itself fall back to the circular grid if GetGridAnalytics
    # has no data for this project/region - that is production's real behavior, not
    # something this script is choosing).
    params = {
        "project_id": project_id,
        "region": region,
        "radius": radius_m,
        "grid": grid_resolution_m,
        "workers": 8,
        "max_interference_sites": 10,
        "use_frontend_grid_sampling": True,
        "samples_per_grid_axis": 3,
        "max_cells_per_grid": 3,
        "min_cells_per_grid": 1,
        "ensure_all_cells": True,
        "min_grids_per_cell": 1,
        "min_candidate_rsrp_dbm": -128,
        "candidate_safety_cap": 20,
    }

    # ---- STAGE 1: COST-231 raw (real production function, unmodified) ----
    print("[TRACE] STAGE 1: running COST-231 (run_rf_prediction_fast) for ALL sectors...")
    raw_pred_df = run_rf_prediction_fast(site_df, drive_df, building_df, params)
    print(f"[TRACE] STAGE 1 done: {len(raw_pred_df)} points across all sectors, "
          f"rsrp_range=({raw_pred_df['pred_rsrp'].min():.2f},{raw_pred_df['pred_rsrp'].max():.2f})")

    pred_df = raw_pred_df.copy()
    pred_df["stage1_raw_rsrp"] = pred_df["pred_rsrp"]
    pred_df["stage1_raw_rsrq"] = pred_df["pred_rsrq"]
    pred_df["stage1_raw_sinr"] = pred_df["pred_sinr"]

    # ---- Replicate apply_full_display_correction's real sequence, with a
    # snapshot inserted before the point where production code overwrites
    # the geo-only value with the DT-calibrated one. ----
    current_engine = engine.get(region.lower(), engine["india"])
    polygons = _resolve_prediction_polygons(params, current_engine)

    site_norm = normalize_site_for_geo(site_df)
    weights, weights_summary = load_geo_weights(project_id=project_id, weights_path=None)
    print(f"[TRACE] geo weights source={weights_summary}")

    polygon_list = list(polygons)
    polygon_gdf = gpd.GeoDataFrame({"geometry": polygon_list}, crs="EPSG:4326")
    polygon_gdf, polygon_alignment = align_project_polygon_to_points(polygon_gdf, site_norm)
    print(f"[TRACE] polygon_alignment={polygon_alignment}")

    osm_enabled = _osm_enrichment_enabled(params)
    building_gdf = building_df_to_gdf(building_df)
    building_gdf, building_alignment = align_building_geometries_to_project(building_gdf, polygon_gdf)
    print(f"[TRACE] building_alignment={building_alignment} building_rows={len(building_gdf)}")
    building_gdf = _enrich_buildings_with_osm_heights(building_gdf, polygon_gdf, enabled=osm_enabled)

    tile_size_m = float(params.get("tile_size_m") or max(float(params.get("grid", 25.0)), DEFAULT_TILE_SIZE_M))
    cluster_count = 5

    grid_gdf = create_analysis_grid(polygon_gdf, tile_size_m)
    grid_gdf = attach_building_features(grid_gdf, building_gdf)
    grid_gdf = _attach_osm_context_features(grid_gdf, polygon_gdf, enabled=osm_enabled)
    grid_df, _ = build_grid_feature_frame(grid_gdf, site_norm, cluster_count)
    grid_df, geo_status = augment_grid_with_advanced_geo_features(grid_df, building_gdf, site_norm, dem_raster_path=dem_path)
    grid_gdf = grid_gdf.merge(grid_df[["grid_id", "clutter_class", "morphology_cluster"]], on="grid_id", how="left")
    print(f"[TRACE] clutter distribution (all sectors): {grid_df['clutter_class'].value_counts().to_dict()}")

    pred_work = pred_df.copy()
    pred_work = assign_points_to_tiles(pred_work, grid_gdf)
    pred_work = _attach_missing_grid_features_by_grid_id(pred_work, grid_df)
    if "Node_Cell_ID" not in pred_work.columns and "node_cell_id" in pred_work.columns:
        pred_work["Node_Cell_ID"] = pred_work["node_cell_id"].astype(str)
    pred_work = attach_fixed_serving_sinr_rsrq_proxy(pred_work, site_norm)

    # ---- STAGE 2: geo-corrected (captured BEFORE DT calibration overwrites pred_rsrp_geo) ----
    print("[TRACE] STAGE 2: apply_experimental_geo_adjustments...")
    pred_work, geo_summary = apply_experimental_geo_adjustments(pred_work, weights=weights)
    pred_work["stage2_geo_rsrp"] = pred_work["pred_rsrp_geo"].astype(float).copy()
    print(f"[TRACE] STAGE 2 done: rsrp_range=({pred_work['stage2_geo_rsrp'].min():.2f},{pred_work['stage2_geo_rsrp'].max():.2f})")

    drive_train_df, drive_holdout_df, split_summary = split_drive_train_holdout(drive_df, validation_fraction=0.25)
    train_eval, _, train_geo_metrics = evaluate_geo_against_dt(drive_train_df, pred_work)
    dt_calibration_models, dt_calibration_debug = fit_dt_holdout_calibration(train_eval)

    # ---- STAGE 3: DT-calibrated ----
    print("[TRACE] STAGE 3: apply_dt_holdout_calibration + preserve_calibrated_kpis...")
    pred_work = apply_dt_holdout_calibration(pred_work, dt_calibration_models)
    pred_work = preserve_calibrated_kpis(pred_work)
    print(f"[TRACE] STAGE 3 done: rsrp_range=({pred_work['pred_rsrp_calibrated'].min():.2f},{pred_work['pred_rsrp_calibrated'].max():.2f})")

    # ---- STAGE 4: smoothed / DT-anchor-blended ----
    print("[TRACE] STAGE 4: apply_demo_dt_overlay...")
    pred_work, demo_summary = apply_demo_dt_overlay(
        pred_work, drive_df, replace_radius_m=20.0, blend_sigma_m=60.0, blend_radius_m=140.0,
    )
    print(f"[TRACE] STAGE 4 done: demo_summary={demo_summary}")

    # ---- Robust site/sector matching: pred_work's Node_Cell_ID has no
    # region suffix ("LA200267_LA200267A2_A2_28"); site_norm's does
    # ("...A2_28_Taiwan"). Match on the parsed cell_id component instead of
    # the raw string - an earlier version of this script matched the raw
    # string, silently failed, and fell back to an arbitrary wrong site for
    # the bearing/distance reference. ----
    pred_work["_cell_id_key"] = pred_work["Node_Cell_ID"].astype(str).str.split("_").str[1]
    site_lookup = site_norm.set_index("cell_id")

    sites_out = {}
    for site_id, site_rows in site_norm.groupby("Site ID"):
        if only_sites and site_id not in only_sites:
            continue
        sectors_out = {}
        for _, srow in site_rows.iterrows():
            cell_id = srow["cell_id"]
            sector_pts = pred_work[pred_work["_cell_id_key"] == cell_id].copy()
            if sector_pts.empty:
                continue
            site_lat = float(srow["lat"])
            site_lon = float(srow["lon"])
            azimuth = float(srow["azimuth"]) if pd.notna(srow["azimuth"]) else 0.0
            sector_pts["distance_m"] = haversine_m(site_lat, site_lon, sector_pts["lat"].astype(float), sector_pts["lon"].astype(float))
            sector_pts["bearing_deg"] = bearing_deg(site_lat, site_lon, sector_pts["lat"].astype(float), sector_pts["lon"].astype(float))
            sector_pts["off_axis_deg"] = off_axis_deg(sector_pts["bearing_deg"], azimuth)

            radial = sector_pts.sort_values("distance_m").copy()
            radial["spacing_from_prev_m"] = radial["distance_m"].diff()
            for col in ["stage1_raw_rsrp", "stage2_geo_rsrp", "pred_rsrp_calibrated", "pred_rsrp_demo"]:
                radial[f"delta_{col}"] = radial[col].diff()
            # keep only the on-axis half of the grid for a cleaner radial profile
            radial_onaxis = radial[radial["off_axis_deg"] <= ANTENNA_3DB_BEAMWIDTH_DEG].sort_values("distance_m")

            keep_cols = [
                "lat", "lon", "distance_m", "bearing_deg", "off_axis_deg",
                "stage1_raw_rsrp", "stage1_raw_rsrq", "stage1_raw_sinr",
                "stage2_geo_rsrp", "pred_rsrp_calibrated", "pred_rsrq_calibrated", "pred_sinr_calibrated",
                "pred_rsrp_demo", "pred_rsrq_demo", "pred_sinr_demo",
                "clutter_class", "morphology_cluster", "building_count", "building_area_ratio",
                "road_length_m", "green_ratio", "water_ratio",
                "los_blocker_count", "los_blocked_ratio", "max_blocker_height_m",
                "demo_dt_distance_m", "demo_blend_weight", "demo_dt_anchor",
            ]
            keep_cols = [c for c in keep_cols if c in sector_pts.columns]

            def df_to_records(df):
                return json.loads(df.replace({np.nan: None}).to_json(orient="records"))

            radial_cols = keep_cols + [
                "spacing_from_prev_m", "delta_stage1_raw_rsrp", "delta_stage2_geo_rsrp",
                "delta_pred_rsrp_calibrated", "delta_pred_rsrp_demo",
            ]
            radial_cols = [c for c in radial_cols if c in radial_onaxis.columns]

            sectors_out[srow["sector"]] = {
                "cell_id": cell_id,
                "band": None if pd.isna(srow.get("band")) else (int(srow["band"]) if float(srow["band"]).is_integer() else float(srow["band"])),
                "azimuth": azimuth,
                "beamwidth_3db_deg": ANTENNA_3DB_BEAMWIDTH_DEG,
                "point_count": int(len(sector_pts)),
                "stage_summary": {
                    "stage1_raw_rsrp": {"min": float(sector_pts["stage1_raw_rsrp"].min()), "max": float(sector_pts["stage1_raw_rsrp"].max()), "mean": float(sector_pts["stage1_raw_rsrp"].mean()), "std": float(sector_pts["stage1_raw_rsrp"].std())},
                    "stage2_geo_rsrp": {"min": float(sector_pts["stage2_geo_rsrp"].min()), "max": float(sector_pts["stage2_geo_rsrp"].max()), "mean": float(sector_pts["stage2_geo_rsrp"].mean()), "std": float(sector_pts["stage2_geo_rsrp"].std())},
                    "stage3_calibrated_rsrp": {"min": float(sector_pts["pred_rsrp_calibrated"].min()), "max": float(sector_pts["pred_rsrp_calibrated"].max()), "mean": float(sector_pts["pred_rsrp_calibrated"].mean()), "std": float(sector_pts["pred_rsrp_calibrated"].std())},
                    "stage4_smoothed_rsrp": {"min": float(sector_pts["pred_rsrp_demo"].min()), "max": float(sector_pts["pred_rsrp_demo"].max()), "mean": float(sector_pts["pred_rsrp_demo"].mean()), "std": float(sector_pts["pred_rsrp_demo"].std())},
                },
                "clutter_distribution": sector_pts["clutter_class"].value_counts().to_dict() if "clutter_class" in sector_pts.columns else {},
                "all_points": df_to_records(sector_pts[keep_cols]),
                "radial_profile": df_to_records(radial_onaxis[radial_cols]),
            }
            print(f"[TRACE] site={site_id} sector={srow['sector']} cell_id={cell_id} azimuth={azimuth} "
                  f"points={len(sector_pts)} on_axis_radial_points={len(radial_onaxis)}")

        if sectors_out:
            first = site_rows.iloc[0]
            sites_out[site_id] = {
                "site_id": site_id,
                "lat": float(first["lat"]),
                "lon": float(first["lon"]),
                "sectors": sectors_out,
            }

    output = {
        "project_id": project_id,
        "region": region,
        "demo_summary": demo_summary,
        "dt_calibration_debug": dt_calibration_debug,
        "weights_summary": weights_summary,
        "sites": sites_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output), encoding="utf-8")
    print(f"[TRACE] wrote {output_path} ({output_path.stat().st_size} bytes), {len(sites_out)} sites")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Site -> sector stage-by-stage LTE prediction trace")
    parser.add_argument("--project-id", type=int, default=210)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--session-ids", type=int, nargs="+", default=None)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--polygon-ids", type=str, default=None)
    parser.add_argument("--radius-m", type=float, default=500.0)
    parser.add_argument("--grid-resolution-m", type=float, default=25.0)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "site_sector_trace.json")
    parser.add_argument("--only-site", type=str, nargs="+", default=None,
                         help="Restrict the OUTPUT to these physical Site IDs (e.g. LA201565). "
                              "The full site set is still fetched and used for COST-231 interference "
                              "candidates, exactly like production - only what gets written is scoped down.")
    args = parser.parse_args(argv)

    session_ids = args.session_ids
    if session_ids is None:
        from sqlalchemy import create_engine, text
        db_url = os.getenv("DATABASE_URL_Taiwan") if args.region == "taiwan" else os.getenv("DATABASE_URL")
        eng = create_engine(db_url)
        with eng.connect() as conn:
            ref = conn.execute(text("SELECT ref_session_id FROM tbl_project WHERE id=:pid"), {"pid": args.project_id}).scalar()
        session_ids = [int(s.strip()) for s in (ref or "").split(",") if s.strip()]
        print(f"[TRACE] resolved {len(session_ids)} session_ids from tbl_project.ref_session_id")

    run_trace(
        project_id=args.project_id,
        region=args.region,
        session_ids=session_ids,
        operator=args.operator,
        radius_m=args.radius_m,
        grid_resolution_m=args.grid_resolution_m,
        polygon_ids=args.polygon_ids,
        output_path=args.output,
        only_sites=set(args.only_site) if args.only_site else None,
    )


if __name__ == "__main__":
    main()
