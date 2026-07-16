import numpy as np
import pandas as pd
from shapely import contains as shp_contains, points as shp_points
from shapely.wkt import loads as load_wkt

from .db import (
    get_engine,
    get_project_by_id,
    get_network_logs_for_sessions,
    get_project_regions,
)
from .map_generator import normalize_band_name


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------

def _parse_session_ids(ref_session_id: str) -> list[int]:
    """
    Convert '3187,3189,3191' -> [3187, 3189, 3191]
    """
    return [
        int(s.strip())
        for s in ref_session_id.split(",")
        if s.strip().isdigit()
    ]


def _parse_polygons(region_rows) -> list:
    """
    Convert region BLOB/TEXT -> shapely polygons
    """
    polygons = []

    for row in region_rows:
        if "region_wkt" in row:
            raw_region = row["region_wkt"]
        elif "region" in row:
            raw_region = row["region"]
        else:
            raise KeyError(
                f"Polygon column not found. Available keys: {list(row.keys())}"
            )

        if not raw_region:
            continue

        polygon = load_wkt(raw_region)
        polygons.append(polygon)

    return polygons


def _filter_df_by_polygons(df: pd.DataFrame, polygons: list) -> pd.DataFrame:
    """
    Keep rows where (lon, lat) lies inside ANY polygon.

    Vectorized with shapely 2.0: builds all points once and tests them against
    each polygon in a single call — ~50-100x faster than the previous
    row-by-row df.iterrows()/poly.contains(Point(...)) loop, with identical
    results.
    """
    if not polygons:
        return df
    if df.empty or not {"lat", "lon"}.issubset(df.columns):
        return df

    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=float)
    pts = shp_points(lon, lat)

    inside = np.zeros(len(df), dtype=bool)
    for poly in polygons:
        inside |= shp_contains(poly, pts)
    inside &= ~(np.isnan(lon) | np.isnan(lat))

    return df.loc[inside].reset_index(drop=True)


def _swap_polygon_coords(polygons: list):
    """
    Swap (x, y) -> (y, x) for each polygon, to handle WKT stored as lat/lon.
    """
    from shapely.ops import transform

    def _swap_xy(x, y, z=None):
        return (y, x) if z is None else (y, x, z)

    return [transform(_swap_xy, poly) for poly in polygons]


def _polygons_to_wkt(polygons: list) -> list[str]:
    from shapely.wkt import dumps as dump_wkt
    return [dump_wkt(poly) for poly in polygons]


def _filter_df_by_polygons_swapped(df: pd.DataFrame, polygons: list) -> pd.DataFrame:
    """
    Fallback: swap polygon coordinates (lat/lon) and keep rows where (lon, lat) lies inside.
    """
    if not polygons:
        return df
    return _filter_df_by_polygons(df, _swap_polygon_coords(polygons))


# -----------------------------------------------------
# Report dataset helpers (shared by main pipeline)
# -----------------------------------------------------

def filter_known_band_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows whose band normalizes to a known value.

    normalize_band_name() converts raw values like "0", "-1", "Unknown",
    "N/A" to "Unknown"; those rows are dropped so band %, maps, CDF and
    metadata are all computed over the same known-band population.
    """
    if "band" not in df.columns:
        return df.iloc[0:0].copy()
    normalized = df["band"].apply(normalize_band_name)
    return df.loc[normalized.ne("Unknown")].reset_index(drop=True)


def polygon_filter_all_cells(raw_df: pd.DataFrame, polygon_wkt: str | None) -> pd.DataFrame:
    """
    Spatially filter raw rows by the project polygon WITHOUT any primary-cell
    restriction — this matches the C# GetNetworkLog backend (which returns all
    cells for the session IDs inside the polygon) and is used for handover
    detection so the report count matches the frontend.  Vectorized.
    """
    if not polygon_wkt:
        return raw_df

    geo = raw_df.dropna(subset=["lat", "lon"])
    if geo.empty:
        return raw_df

    poly = load_wkt(polygon_wkt)
    lon = pd.to_numeric(geo["lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(geo["lat"], errors="coerce").to_numpy(dtype=float)
    pts = shp_points(lon, lat)
    mask = shp_contains(poly, pts)
    mask = mask & ~(np.isnan(lon) | np.isnan(lat))
    return geo.loc[mask].reset_index(drop=True)


def _filter_primary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only primary-serving rows.

    The frontend uses the registered-cell marker inside `primary_cell_info_1`
    (mRegistered=YES). Some rows (e.g. NR/5G rows whose `primary_cell_info_1`
    is a placeholder like "nr_from_signalstrength" rather than structured cell
    text) never contain that marker even though they are the serving cell, so
    the explicit `primary` column is checked as well and either signal being
    true keeps the row.
    """
    if df.empty:
        return df

    has_cell_info = "primary_cell_info_1" in df.columns
    has_primary_col = "primary" in df.columns

    if not has_cell_info and not has_primary_col:
        return df

    primary_mask = pd.Series(False, index=df.index)

    if has_cell_info:
        primary_info = df["primary_cell_info_1"].fillna("").astype(str)
        primary_mask |= primary_info.str.contains("mRegistered=YES", case=False, na=False)

    if has_primary_col:
        primary_values = df["primary"].fillna("").astype(str).str.strip().str.lower()
        primary_mask |= primary_values.isin({"yes", "y", "true", "1"})

    return df.loc[primary_mask].reset_index(drop=True)


# -----------------------------------------------------
# MAIN LOADER (THIS IS WHAT YOU IMPORT)
# -----------------------------------------------------

def load_project_data(project_id: int):
    """
    DB-based replacement for Excel loading.

    Returns:
        raw_df       : DataFrame (all project session data)
        filtered_df  : DataFrame (polygon-filtered data)
        project_meta : dict (tbl_project row)
    """
    engine = get_engine()

    with engine.connect() as conn:
        # 1 Project
        project = get_project_by_id(project_id, conn)
        if not project:
            raise ValueError(f"No project found for id={project_id}")

        ref_session_id = project["ref_session_id"]
        session_ids = _parse_session_ids(ref_session_id)

        if not session_ids:
            raise ValueError("No valid session IDs found for project")

        # 2 Raw network logs
        raw_df = get_network_logs_for_sessions(
            session_ids,
            conn,
            project_id=project_id,
            provider=project.get("provider"),
            start_date=project.get("from_date"),
            end_date=project.get("to_date"),
        )

        valid_geo_rows = 0
        if {"lat", "lon"}.issubset(raw_df.columns):
            valid_geo_rows = int((raw_df["lat"].notna() & raw_df["lon"].notna()).sum())
        print(
            "[ReportLogs] load_project_data received raw logs "
            f"project_id={project_id} sessions={session_ids} rows={len(raw_df)} "
            f"valid_lat_lon_rows={valid_geo_rows} columns={list(raw_df.columns)}"
        )

        

        # 3 Primary row filtering
        primary_df = _filter_primary_rows(raw_df)

        # 4 Regions / polygons
        region_rows = get_project_regions(project_id, conn)
        polygons = _parse_polygons(region_rows)

        # 5 Spatial filtering
        filtered_df = _filter_df_by_polygons(primary_df, polygons)
        print(
            "[ReportLogs] polygon filter result "
            f"project_id={project_id} polygons={len(polygons)} rows_before={len(primary_df)} "
            f"rows_after={len(filtered_df)}"
        )
        used_polygons = polygons
        used_region_wkts = None

        # Fallback: try swapped polygon coords if first pass returns 0 rows
        if filtered_df.empty and polygons:
            print("WARNING: Polygon filter returned 0 rows, retrying with swapped polygon coordinates.")
            swapped_polygons = _swap_polygon_coords(polygons)
            filtered_df = _filter_df_by_polygons(primary_df, swapped_polygons)
            print(
                "[ReportLogs] swapped polygon filter result "
                f"project_id={project_id} rows_after={len(filtered_df)}"
            )
            if not filtered_df.empty:
                used_polygons = swapped_polygons
                used_region_wkts = _polygons_to_wkt(swapped_polygons)

        # 6 Add region WKT to project metadata for map generation
        if used_region_wkts:
            project["region"] = used_region_wkts[0]
        elif region_rows and len(region_rows) > 0:
            project["region"] = region_rows[0]["region_wkt"]
        else:
            project["region"] = None

        return raw_df, filtered_df, project
