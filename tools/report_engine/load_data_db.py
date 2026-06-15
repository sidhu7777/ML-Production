import pandas as pd
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt

from .db import (
    get_engine,
    get_project_by_id,
    get_network_logs_for_sessions,
    get_project_regions,
)


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
    Keep rows where (lon, lat) lies inside ANY polygon
    """
    if not polygons:
        return df

    mask = []

    for _, row in df.iterrows():
        point = Point(row["lon"], row["lat"])
        inside = any(poly.contains(point) for poly in polygons)
        mask.append(inside)

    return df.loc[mask].reset_index(drop=True)


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

    polygons = _swap_polygon_coords(polygons)

    mask = []
    for _, row in df.iterrows():
        point = Point(row["lon"], row["lat"])
        inside = any(poly.contains(point) for poly in polygons)
        mask.append(inside)

    return df.loc[mask].reset_index(drop=True)


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

        

        # 3 Regions / polygons
        region_rows = get_project_regions(project_id, conn)
        polygons = _parse_polygons(region_rows)

        # 4 Spatial filtering
        filtered_df = _filter_df_by_polygons(raw_df, polygons)
        used_polygons = polygons
        used_region_wkts = None

        # Fallback: try swapped polygon coords if first pass returns 0 rows
        if filtered_df.empty and polygons:
            print("WARNING: Polygon filter returned 0 rows, retrying with swapped polygon coordinates.")
            swapped_polygons = _swap_polygon_coords(polygons)
            filtered_df = _filter_df_by_polygons(raw_df, swapped_polygons)
            if not filtered_df.empty:
                used_polygons = swapped_polygons
                used_region_wkts = _polygons_to_wkt(swapped_polygons)

        # 5 Add region WKT to project metadata for map generation
        if used_region_wkts:
            project["region"] = used_region_wkts[0]
        elif region_rows and len(region_rows) > 0:
            project["region"] = region_rows[0]["region_wkt"]
        else:
            project["region"] = None

        return raw_df, filtered_df, project
