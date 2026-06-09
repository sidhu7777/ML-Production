import os

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from shapely.geometry import Point
from shapely.ops import transform
from shapely.wkt import loads as load_wkt

from .geo_correction_pipeline import (
    apply_full_display_correction,
    align_building_geometries_to_project,
    align_project_polygon_to_points,
    attach_site_identity_to_predictions,
    building_df_to_gdf,
    filter_sites_to_project_polygon,
    prepare_building_df_for_rf,
    prepare_site_df_for_source_rf_export,
)
from .grid_sampling import (
    assign_samples_to_relevant_cells,
    build_grid_sample_points,
    fetch_frontend_grid_cells,
)
from .Sector_wise_prediction_code_copy import run_prediction_from_api


load_dotenv()

engine = {
    "india": create_engine(
        os.getenv("DATABASE_URL"),
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL") else None,
    "taiwan": create_engine(
        os.getenv("DATABASE_URL_Taiwan"),
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL_Taiwan") else None
}


def _safe_nunique(df, col):
    return int(df[col].nunique(dropna=True)) if col in df.columns else "n/a"


def _safe_non_null(df, col):
    return int(df[col].notna().sum()) if col in df.columns else "n/a"


def _safe_minmax(df, col):
    if col not in df.columns:
        return "n/a"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return "n/a"
    return f"{series.min():.4f}..{series.max():.4f}"


def _print_fetch_summary(stage, table_name, filters, df, extra=None):
    print(f"[LTE][{stage}] source_table={table_name}")
    print(f"[LTE][{stage}] filters={filters}")
    print(f"[LTE][{stage}] row_count={len(df)}")
    print(f"[LTE][{stage}] columns={list(df.columns)}")
    if extra:
        for key, value in extra.items():
            print(f"[LTE][{stage}] {key}={value}")


def _print_df_profile(stage, df):
    pci_col = "PCI" if "PCI" in df.columns else "pci"
    print(f"[LTE][{stage}] df_shape={df.shape}")
    print(
        f"[LTE][{stage}] unique_cell_id={_safe_nunique(df, 'cell_id')} "
        f"unique_pci={_safe_nunique(df, pci_col)} "
        f"unique_nodeb_id={_safe_nunique(df, 'nodeb_id')}"
    )
    print(
        f"[LTE][{stage}] lat_range={_safe_minmax(df, 'lat')} "
        f"lon_range={_safe_minmax(df, 'lon')}"
    )
    if "network" in df.columns:
        top_network = df["network"].astype(str).value_counts(dropna=False).head(5).to_dict()
        print(f"[LTE][{stage}] top_network_counts={top_network}")


def _load_project_polygons(project_id, current_engine):
    query = f"""
    SELECT ST_AsText(region) AS region_wkt
    FROM map_regions
    WHERE tbl_project_id = {project_id}
      AND status = 1
    """

    region_df = pd.read_sql(query, current_engine)
    polygons = []
    for raw_region in region_df.get("region_wkt", pd.Series(dtype=str)).dropna():
        raw_region = str(raw_region).strip()
        if not raw_region:
            continue
        try:
            polygons.append(load_wkt(raw_region))
        except Exception:
            continue
    return polygons


def _resolve_prediction_polygons(params, current_engine):
    polygon_wkt = str(params.get("polygon_wkt") or "").strip() if params else ""
    if polygon_wkt:
        try:
            return [load_wkt(polygon_wkt)]
        except Exception as exc:
            print(f"[LTE][POLYGON_OVERRIDE] invalid_wkt={exc}")
    return _load_project_polygons(params["project_id"], current_engine)


def _swap_polygon_coords(polygons):
    def _swap_xy(x, y, z=None):
        return (y, x) if z is None else (y, x, z)

    return [transform(_swap_xy, poly) for poly in polygons]


def _filter_df_by_polygons(df, polygons):
    if df.empty or not polygons:
        return df

    mask = []
    for _, row in df.iterrows():
        point = Point(row["lon"], row["lat"])
        mask.append(any(poly.contains(point) for poly in polygons))
    return df.loc[mask].copy()


def _apply_drive_polygon_filter(df, project_id, current_engine):
    polygons = _load_project_polygons(project_id, current_engine)
    if not polygons:
        print("[LTE][DRIVE_FETCH_POLYGON] polygons_found=0 skipped=True")
        return df, {
            "polygons_found": 0,
            "rows_before": len(df),
            "rows_after": len(df),
            "swapped": False,
            "skipped": True,
        }

    filtered_df = _filter_df_by_polygons(df, polygons)
    if not filtered_df.empty:
        stats = {
            "polygons_found": len(polygons),
            "rows_before": len(df),
            "rows_after": len(filtered_df),
            "swapped": False,
            "skipped": False,
        }
        print(
            f"[LTE][DRIVE_FETCH_POLYGON] polygons_found={stats['polygons_found']} "
            f"rows_before={stats['rows_before']} rows_after={stats['rows_after']} "
            f"swapped={stats['swapped']}"
        )
        return filtered_df, stats

    swapped_polygons = _swap_polygon_coords(polygons)
    swapped_df = _filter_df_by_polygons(df, swapped_polygons)
    stats = {
        "polygons_found": len(polygons),
        "rows_before": len(df),
        "rows_after": len(swapped_df),
        "swapped": True,
        "skipped": False,
    }
    print(
        f"[LTE][DRIVE_FETCH_POLYGON] polygons_found={stats['polygons_found']} "
        f"rows_before={stats['rows_before']} rows_after={stats['rows_after']} "
        f"swapped={stats['swapped']}"
    )
    return swapped_df, stats


def _apply_prediction_polygon_filter(df, project_id, current_engine):
    polygons = _load_project_polygons(project_id, current_engine)
    if not polygons:
        print("[LTE][RF_OUTPUT_POLYGON] polygons_found=0 skipped=True")
        return df, {
            "polygons_found": 0,
            "rows_before": len(df),
            "rows_after": len(df),
            "swapped": False,
            "skipped": True,
        }

    filtered_df = _filter_df_by_polygons(df, polygons)
    if not filtered_df.empty:
        stats = {
            "polygons_found": len(polygons),
            "rows_before": len(df),
            "rows_after": len(filtered_df),
            "swapped": False,
            "skipped": False,
        }
        print(
            f"[LTE][RF_OUTPUT_POLYGON] polygons_found={stats['polygons_found']} "
            f"rows_before={stats['rows_before']} rows_after={stats['rows_after']} "
            f"swapped={stats['swapped']}"
        )
        return filtered_df, stats

    swapped_polygons = _swap_polygon_coords(polygons)
    swapped_df = _filter_df_by_polygons(df, swapped_polygons)
    stats = {
        "polygons_found": len(polygons),
        "rows_before": len(df),
        "rows_after": len(swapped_df),
        "swapped": True,
        "skipped": False,
    }
    print(
        f"[LTE][RF_OUTPUT_POLYGON] polygons_found={stats['polygons_found']} "
        f"rows_before={stats['rows_before']} rows_after={stats['rows_after']} "
        f"swapped={stats['swapped']}"
    )
    return swapped_df, stats


def _resolve_operator(df):
    for col in ("network", "cluster", "operator", "Technology"):
        if col in df.columns:
            series = df[col].dropna().astype(str).str.strip()
            series = series[series != ""]
            if not series.empty:
                operator = series.mode().iloc[0]
                print(f"[LTE][SITE_FETCH] resolved_operator_from={col} value={operator}")
                return operator
    raise ValueError("Unable to resolve operator from site_prediction data")


def fetch_site_data(project_id, region="india"):
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT *
    FROM site_prediction
    WHERE tbl_project_id = {project_id}
    """

    df = pd.read_sql(query, current_engine)

    if df.empty:
        raise ValueError("No site data found")

    _print_fetch_summary(
        "SITE_FETCH_RAW",
        "site_prediction",
        {"project_id": project_id, "region": region},
        df,
        extra={
            "distinct_pci": _safe_nunique(df, "pci"),
            "distinct_cell_id": _safe_nunique(df, "cell_id"),
            "distinct_nodeb_id": _safe_nunique(df, "nodeb_id"),
            "non_null_cluster": _safe_non_null(df, "cluster"),
        }
    )

    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "latitude": "lat",
        "longitude": "lon",
        "e_tilt": "Etilt",
        "m_tilt": "Mtilt",
        "height": "Height",
        "cell_id": "cell_id",
        "nodeb_id": "nodeb_id",
        "azimuth": "azimuth",
        "pci": "PCI",
        "rsrp": "rsrp",
        "tx_power": "tx_power",
        "band": "band",
        "earfcn": "earfcn",
        "reference_signal_power": "reference_signal_power",
        "site": "Site ID",
        "cluster": "network"
    })

    required = ["lat", "lon", "Etilt", "Mtilt", "Height", "tx_power"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["Etilt"] = pd.to_numeric(df["Etilt"], errors="coerce").fillna(3)
    df["Mtilt"] = pd.to_numeric(df["Mtilt"], errors="coerce").fillna(0)
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce").fillna(30)
    df["tx_power"] = pd.to_numeric(df["tx_power"], errors="coerce").fillna(46)
    if "frequency_mhz" not in df.columns:
        if "frequency" in df.columns:
            df["frequency_mhz"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(1800)
        else:
            df["frequency_mhz"] = 1800
    if "Node_Cell_ID" not in df.columns and "cell_id" in df.columns:
        df["Node_Cell_ID"] = df["cell_id"].astype(str).str.strip()

    df = df.dropna(subset=["lat", "lon"])

    print("[LTE][SITE_FETCH_READY] converted_to_prediction_engine_format=True")
    _print_df_profile("SITE_FETCH_READY", df)
    return df, _resolve_operator(df)


def fetch_drive_data(session_ids, operator, project_id, region="india"):
    session_str = ",".join(map(str, session_ids))
    key = f"{project_id}_{operator}_{session_str}"
    path = f"cache/drive_{key}.parquet"
    required_drive_cols = {"cell_id", "nodeb_id"}

    if os.path.exists(path):
        print("[LTE][DRIVE_FETCH_CACHE] cache_hit=True")
        cached = pd.read_parquet(path)
        missing_cache_cols = sorted(required_drive_cols - set(cached.columns))
        if not missing_cache_cols:
            cached["cell_id"] = cached["cell_id"].astype(str).str.strip()
            cached["nodeb_id"] = cached["nodeb_id"].astype(str).str.strip()
            _print_fetch_summary(
                "DRIVE_FETCH_CACHE",
                "cache/drive parquet",
                {
                    "session_ids": session_ids,
                    "operator": operator,
                    "project_id": project_id,
                    "region": region
                },
                cached,
                extra={
                    "distinct_sessions_requested": len(session_ids),
                    "lat_range": _safe_minmax(cached, "lat"),
                    "lon_range": _safe_minmax(cached, "lon"),
                }
            )
            return cached

        print(
            f"[LTE][DRIVE_FETCH_CACHE] refreshing_cache_missing_columns={missing_cache_cols}"
        )

    current_engine = engine.get(region.lower(), engine["india"])

    main_query = f"""
    SELECT lat, lon, rsrp, rsrq, sinr, cell_id, nodeb_id, pci, earfcn
    FROM tbl_network_log
    WHERE session_id IN ({session_str})
    AND LOWER(COALESCE(m_alpha_long, m_alpha_short)) = LOWER('{operator}')
    AND LOWER(COALESCE(`primary`, '')) = 'yes'
    """

    neighbour_query = f"""
    SELECT lat, lon, rsrp, rsrq, sinr, cell_id, nodeb_id, pci, earfcn
    FROM tbl_network_log_neighbour
    WHERE session_id IN ({session_str})
    AND LOWER(COALESCE(m_alpha_long, m_alpha_short)) = LOWER('{operator}')
    AND LOWER(COALESCE(`primary`, '')) = 'yes'
    """

    raw_main_query = f"""
    SELECT lat, lon, rsrp, rsrq, sinr, cell_id, nodeb_id, pci, earfcn
    FROM tbl_network_log
    WHERE session_id IN ({session_str})
    AND LOWER(COALESCE(m_alpha_long, m_alpha_short)) = LOWER('{operator}')
    """

    raw_neighbour_query = f"""
    SELECT lat, lon, rsrp, rsrq, sinr, cell_id, nodeb_id, pci, earfcn
    FROM tbl_network_log_neighbour
    WHERE session_id IN ({session_str})
    AND LOWER(COALESCE(m_alpha_long, m_alpha_short)) = LOWER('{operator}')
    """

    raw_main_df = pd.read_sql(raw_main_query, current_engine)
    raw_neighbour_df = pd.read_sql(raw_neighbour_query, current_engine)
    main_df = pd.read_sql(main_query, current_engine)
    neighbour_df = pd.read_sql(neighbour_query, current_engine)

    _print_fetch_summary(
        "DRIVE_FETCH_MAIN",
        "tbl_network_log",
        {"session_ids": session_ids, "operator": operator, "project_id": project_id, "region": region},
        main_df,
        extra={"distinct_sessions_requested": len(session_ids)}
    )
    _print_fetch_summary(
        "DRIVE_FETCH_NEIGHBOUR",
        "tbl_network_log_neighbour",
        {"session_ids": session_ids, "operator": operator, "project_id": project_id, "region": region},
        neighbour_df,
        extra={"distinct_sessions_requested": len(session_ids)}
    )

    raw_total_rows = len(raw_main_df) + len(raw_neighbour_df)
    primary_filtered_rows = len(main_df) + len(neighbour_df)

    df = pd.concat([main_df, neighbour_df], ignore_index=True)
    for col in ["cell_id", "nodeb_id", "pci", "earfcn"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df, polygon_stats = _apply_drive_polygon_filter(df, project_id, current_engine)
    print(
        f"[LTE][DRIVE_FETCH_COUNTS] raw_total_rows={raw_total_rows} "
        f"after_primary_rows={primary_filtered_rows} "
        f"after_polygon_rows={len(df)} "
        f"primary_removed={raw_total_rows - primary_filtered_rows} "
        f"polygon_removed={primary_filtered_rows - len(df)} "
        f"polygon_swapped={polygon_stats['swapped']}"
    )
    _print_df_profile("DRIVE_FETCH_COMBINED", df)

    os.makedirs("cache", exist_ok=True)
    df.to_parquet(path)
    return df


def fetch_building_data(project_id, region="india"):
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT
        id,
        name,
        region,
        project_id,
        area,
        geometry,
        ST_AsText(region) AS region_wkt,
        ST_AsText(geometry) AS geometry_wkt
    FROM tbl_savepolygon
    WHERE project_id = {project_id}
    """

    df = pd.read_sql(query, current_engine)
    _print_fetch_summary(
        "BUILDING_FETCH",
        "tbl_savepolygon",
        {"project_id": project_id, "region": region},
        df,
        extra={
            "distinct_project_id": _safe_nunique(df, "project_id"),
            "non_null_region": _safe_non_null(df, "region"),
            "non_null_region_wkt": _safe_non_null(df, "region_wkt"),
            "non_null_geometry_wkt": _safe_non_null(df, "geometry_wkt"),
        }
    )
    return df


def fetch_polygon_data(project_id):
    return {
        "type": "Polygon",
        "coordinates": [[[77.1, 28.6], [77.2, 28.6], [77.2, 28.7], [77.1, 28.7], [77.1, 28.6]]]
    }


def run_rf_prediction_fast(site_df, drive_df, building_df, params):
    temp_dir = "temp_rf"
    os.makedirs(temp_dir, exist_ok=True)

    site_path = f"{temp_dir}/site.csv"
    building_path = f"{temp_dir}/building.csv"

    current_engine = engine.get(params.get("region", "india").lower(), engine["india"])
    polygons = _resolve_prediction_polygons(params, current_engine)
    serving_site_df = site_df.copy()
    building_export_df = building_df.copy()
    aligned_prediction_polygons = polygons
    if polygons:
        import geopandas as gpd

        polygon_gdf = gpd.GeoDataFrame({"geometry": polygons}, crs="EPSG:4326")
        polygon_gdf, polygon_alignment = align_project_polygon_to_points(polygon_gdf, site_df)
        aligned_prediction_polygons = list(polygon_gdf.geometry)
        serving_site_df = filter_sites_to_project_polygon(site_df, polygon_gdf)
        if serving_site_df.empty:
            raise ValueError("No serving LTE sites found inside the aligned project polygon")
        print(
            f"[LTE][RF_SITE_POLYGON] alignment={polygon_alignment} "
            f"all_project_site_rows={len(site_df)} serving_site_rows={len(serving_site_df)}"
        )
        try:
            building_gdf = building_df_to_gdf(building_df)
            building_gdf, _ = align_building_geometries_to_project(building_gdf, polygon_gdf)
            building_export_df = prepare_building_df_for_rf(building_df, building_gdf)
        except Exception as exc:
            print(f"[LTE][RF_INPUT] building_geometry_precheck_failed={exc}")
    site_export_df = prepare_site_df_for_source_rf_export(serving_site_df)
    expected_unique_frontend = (
        int(site_export_df["frontend_site_sector_key"].nunique(dropna=True))
        if "frontend_site_sector_key" in site_export_df.columns
        else _safe_nunique(site_export_df, "cell_id")
    )
    site_export_df.to_csv(site_path, index=False)
    building_export_df.to_csv(building_path, index=False)

    print(
        f"[LTE][RF_INPUT] site_rows={len(site_export_df)} drive_rows={len(drive_df)} "
        f"building_rows={len(building_export_df)} radius={params['radius']} "
        f"grid={params['grid']} workers={params['workers']} "
        f"max_interference_sites={params.get('max_interference_sites', 50)}"
    )
    print(
        f"[LTE][RF_INPUT] unique_cells={_safe_nunique(site_export_df, 'cell_id')} "
        f"unique_frontend_site_sector={expected_unique_frontend} "
        f"unique_pci={_safe_nunique(site_export_df, 'PCI') if 'PCI' in site_export_df.columns else _safe_nunique(site_export_df, 'pci')}"
    )

    prediction_points_df = pd.DataFrame()
    use_frontend_grid_sampling = bool(params.get("use_frontend_grid_sampling", True))
    if use_frontend_grid_sampling:
        grid_df, frontend_grid_scenario_id = fetch_frontend_grid_cells(
            current_engine,
            int(params["project_id"]),
            scenario_id=params.get("grid_analytics_scenario_id"),
            grid_size_meters=params.get("frontend_grid_size_meters"),
        )
        if not grid_df.empty:
            sample_df = build_grid_sample_points(
                grid_df,
                samples_per_axis=int(params.get("samples_per_grid_axis", 3) or 3),
                clip_polygons=aligned_prediction_polygons,
            )
            prediction_points_df = assign_samples_to_relevant_cells(
                sample_df,
                site_export_df,
                radius_m=float(params.get("radius", 500) or 500),
                max_cells_per_grid=int(params.get("max_cells_per_grid", 3) or 3),
                min_cells_per_grid=int(params.get("min_cells_per_grid", 1) or 1),
                ensure_all_cells=bool(params.get("ensure_all_cells", True)),
                min_grids_per_cell=int(params.get("min_grids_per_cell", 1) or 1),
            )
            if not prediction_points_df.empty:
                print(
                    f"[LTE][RF_SAMPLE_SOURCE] source=frontend_grid scenario_id={frontend_grid_scenario_id} "
                    f"rows={len(prediction_points_df)} grids={prediction_points_df['frontend_grid_id'].nunique()} "
                    f"cells={prediction_points_df['Node_Cell_ID'].nunique()}"
                )
        if prediction_points_df.empty:
            print("[LTE][RF_SAMPLE_SOURCE] source=circular_cell_grid reason=no_frontend_grid_samples")
    else:
        print("[LTE][RF_SAMPLE_SOURCE] source=circular_cell_grid reason=frontend_grid_sampling_disabled")

    run_prediction_from_api({
        "site": site_path,
        "drive": None,
        "building": building_path,
        "polygon_area": None,
        "radius": params["radius"],
        "grid_resolution": params["grid"],
        "frequency": params.get("frequency_mhz", 1800),
        "bandwidth": params.get("bandwidth_mhz", 10),
        "antenna_gain": params.get("antenna_gain", 18),
        "cable_loss": params.get("cable_loss", 2),
        "ue_height": params.get("ue_height", 1.5),
        "outdir": temp_dir,
        "n_workers": params["workers"],
        "max_interference_sites": params.get("max_interference_sites", 50),
        "calibrate": False,
        "prediction_points_df": prediction_points_df,
    })

    pred_df = pd.read_csv(f"{temp_dir}/prediction_ALL_SITES.csv")
    override_polygon_wkt = str(params.get("polygon_wkt") or "").strip()
    if not prediction_points_df.empty:
        polygon_stats = {
            "polygons_found": len(polygons or []),
            "rows_before": len(pred_df),
            "rows_after": len(pred_df),
            "swapped": False,
            "skipped": True,
            "reason": "frontend_grid_samples_already_define_prediction_surface",
        }
        print(
            f"[LTE][RF_OUTPUT_POLYGON] skipped=True reason={polygon_stats['reason']} "
            f"rows_before={polygon_stats['rows_before']} rows_after={polygon_stats['rows_after']}"
        )
    elif override_polygon_wkt:
        polygons = _resolve_prediction_polygons(params, current_engine)
        pred_df = _filter_df_by_polygons(pred_df, polygons)
        polygon_stats = {
            "polygons_found": len(polygons),
            "rows_before": len(pd.read_csv(f"{temp_dir}/prediction_ALL_SITES.csv")),
            "rows_after": len(pred_df),
            "swapped": False,
            "skipped": False,
        }
        print(
            f"[LTE][RF_OUTPUT_POLYGON] polygons_found={polygon_stats['polygons_found']} "
            f"rows_before={polygon_stats['rows_before']} rows_after={polygon_stats['rows_after']} "
            f"swapped={polygon_stats['swapped']} override=True"
        )
    else:
        pred_df, polygon_stats = _apply_prediction_polygon_filter(
            pred_df, params["project_id"], current_engine
        )
    pred_df = attach_site_identity_to_predictions(pred_df, site_df)
    print(
        f"[LTE][RF_OUTPUT_COUNTS] rows_before_polygon={polygon_stats['rows_before']} "
        f"rows_after_polygon={len(pred_df)} "
        f"polygon_removed={polygon_stats['rows_before'] - len(pred_df)} "
        f"polygon_swapped={polygon_stats['swapped']}"
    )
    _print_fetch_summary(
        "RF_OUTPUT",
        "temp_rf/prediction_ALL_SITES.csv",
        {
            "radius": params["radius"],
            "grid": params["grid"],
            "project_id": params["project_id"],
            "region": params.get("region", "india"),
            "calibrate": False,
        },
        pred_df,
        extra={
            "unique_predicted_cells": _safe_nunique(pred_df, "Node_Cell_ID"),
            "unique_frontend_site_sector": _safe_nunique(pred_df, "frontend_site_sector_key"),
            "pred_rsrp_range": _safe_minmax(pred_df, "pred_rsrp"),
            "pred_rsrq_range": _safe_minmax(pred_df, "pred_rsrq"),
            "pred_sinr_range": _safe_minmax(pred_df, "pred_sinr"),
        }
    )
    return pred_df

def run_ml_fast(pred_df, drive_df, site_df=None, building_df=None, params=None):
    print(
        f"[LTE][POST_INPUT] pred_rows={len(pred_df)} drive_rows={len(drive_df)} "
        f"pred_cols={list(pred_df.columns)} drive_cols={list(drive_df.columns)}"
    )
    if site_df is None or building_df is None or params is None:
        raise ValueError("Geo/display correction requires site_df, building_df, and params")

    current_engine = engine.get(params.get("region", "india").lower(), engine["india"])
    polygons = _resolve_prediction_polygons(params, current_engine)
    final_df, summary = apply_full_display_correction(
        pred_df,
        drive_df,
        site_df,
        building_df,
        polygons,
        params=params,
    )
    final_df.attrs["production_summary"] = summary
    print(f"[LTE][POST_OUTPUT] summary={summary}")
    return final_df
