from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sqlalchemy import text
from shapely.geometry import Point, box
from shapely.ops import unary_union


def _safe_minmax(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return "n/a"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return "n/a"
    return f"{series.min():.6f}..{series.max():.6f}"


def haversine_m(lat1, lon1, lat2, lon2):
    radius_m = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_m * np.arcsin(np.sqrt(a))


def fetch_frontend_grid_cells(
    current_engine,
    project_id: int,
    scenario_id=None,
    grid_size_meters=None,
) -> tuple[pd.DataFrame, object]:
    filters = ["project_id = :project_id"]
    params = {"project_id": int(project_id)}
    selected_scenario_id = scenario_id
    if selected_scenario_id is None:
        query = text(
            """
            SELECT scenario_id, MAX(created_at) AS max_created, COUNT(*) AS row_count
            FROM grid_analytics_results
            WHERE project_id = :project_id
            GROUP BY scenario_id
            ORDER BY row_count DESC, max_created DESC
            LIMIT 1
            """
        )
        with current_engine.connect() as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            print(f"[LTE][FRONTEND_GRID_FETCH] rows=0 project_id={project_id} reason=no_grid_analytics")
            return pd.DataFrame(), None
        selected_scenario_id = row[0]

    if selected_scenario_id is None:
        filters.append("scenario_id IS NULL")
    else:
        filters.append("scenario_id = :scenario_id")
        params["scenario_id"] = int(selected_scenario_id)
    if grid_size_meters is not None:
        filters.append("grid_size_meters = :grid_size_meters")
        params["grid_size_meters"] = float(grid_size_meters)

    query = text(
        f"""
        SELECT
            grid_id,
            center_lat,
            center_lon,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            grid_size_meters,
            scenario_id
        FROM grid_analytics_results
        WHERE {" AND ".join(filters)}
        ORDER BY grid_id
        """
    )
    with current_engine.connect() as conn:
        grid_df = pd.read_sql(query, conn, params=params)

    required = ["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]
    for col in required:
        if col not in grid_df.columns:
            print(f"[LTE][FRONTEND_GRID_FETCH] rows=0 reason=missing_column_{col}")
            return pd.DataFrame(), selected_scenario_id
    for col in ["center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]:
        grid_df[col] = pd.to_numeric(grid_df[col], errors="coerce")
    grid_df = grid_df.dropna(subset=required).drop_duplicates(subset=["grid_id"], keep="first").copy()
    print(
        f"[LTE][FRONTEND_GRID_FETCH] rows={len(grid_df)} scenario_id={selected_scenario_id} "
        f"lat_range={_safe_minmax(grid_df, 'center_lat')} lon_range={_safe_minmax(grid_df, 'center_lon')}"
    )
    return grid_df, selected_scenario_id


def build_grid_sample_points(
    grid_df: pd.DataFrame,
    samples_per_axis: int = 3,
    clip_polygons=None,
) -> pd.DataFrame:
    if grid_df.empty:
        return pd.DataFrame()
    samples_per_axis = max(1, int(samples_per_axis or 1))
    if samples_per_axis == 1:
        fractions = [0.5]
    else:
        margin = 0.5 / samples_per_axis
        fractions = np.linspace(margin, 1.0 - margin, samples_per_axis)

    clip_union = None
    if clip_polygons:
        valid_polygons = [poly for poly in clip_polygons if poly is not None and not poly.is_empty]
        if valid_polygons:
            clip_union = unary_union(valid_polygons)

    rows = []
    clipped_out_samples = 0
    representative_samples = 0
    skipped_grids = 0
    for row in grid_df.itertuples(index=False):
        lat_span = float(row.max_lat) - float(row.min_lat)
        lon_span = float(row.max_lon) - float(row.min_lon)
        sample_idx = 0
        grid_rows = []
        for fy in fractions:
            for fx in fractions:
                lat = float(row.min_lat) + lat_span * float(fy)
                lon = float(row.min_lon) + lon_span * float(fx)
                if clip_union is not None and not clip_union.covers(Point(lon, lat)):
                    clipped_out_samples += 1
                    continue
                grid_rows.append(
                    {
                        "frontend_grid_id": str(row.grid_id),
                        "grid_id": str(row.grid_id),
                        "sample_index": sample_idx,
                        "lat": lat,
                        "lon": lon,
                        "grid_center_lat": float(row.center_lat),
                        "grid_center_lon": float(row.center_lon),
                        "sample_source": "regular_grid",
                    }
                )
                sample_idx += 1
        if not grid_rows and clip_union is not None:
            grid_box = box(float(row.min_lon), float(row.min_lat), float(row.max_lon), float(row.max_lat))
            clipped = grid_box.intersection(clip_union)
            if not clipped.is_empty:
                point = clipped.representative_point()
                grid_rows.append(
                    {
                        "frontend_grid_id": str(row.grid_id),
                        "grid_id": str(row.grid_id),
                        "sample_index": 0,
                        "lat": float(point.y),
                        "lon": float(point.x),
                        "grid_center_lat": float(row.center_lat),
                        "grid_center_lon": float(row.center_lon),
                        "sample_source": "polygon_intersection_representative",
                    }
                )
                representative_samples += 1
            else:
                skipped_grids += 1
        rows.extend(grid_rows)
    samples = pd.DataFrame(rows)
    print(
        f"[LTE][FRONTEND_GRID_SAMPLES] grids={grid_df['grid_id'].nunique()} "
        f"samples_per_axis={samples_per_axis} rows={len(samples)} "
        f"polygon_clipped={bool(clip_union is not None)} clipped_out_samples={clipped_out_samples} "
        f"representative_samples={representative_samples} skipped_grids={skipped_grids}"
    )
    return samples


def assign_samples_to_relevant_cells(
    samples_df: pd.DataFrame,
    site_df: pd.DataFrame,
    radius_m: float,
    max_cells_per_grid: int = 3,
    min_cells_per_grid: int = 1,
    ensure_all_cells: bool = True,
    min_grids_per_cell: int = 1,
) -> pd.DataFrame:
    if samples_df.empty or site_df.empty:
        return pd.DataFrame()

    site_work = site_df.copy()
    if "Node_Cell_ID" not in site_work.columns:
        site_work["Node_Cell_ID"] = site_work["cell_id"].astype(str)
    site_work["Node_Cell_ID"] = site_work["Node_Cell_ID"].astype(str)
    site_work["lat"] = pd.to_numeric(site_work["lat"], errors="coerce")
    site_work["lon"] = pd.to_numeric(site_work["lon"], errors="coerce")
    valid_sites = site_work.dropna(subset=["lat", "lon", "Node_Cell_ID"]).copy()
    if valid_sites.empty:
        return pd.DataFrame()

    if "canonical_sector_id" in valid_sites.columns:
        sector_key = valid_sites["canonical_sector_id"].astype(str).str.strip()
    elif "frontend_site_sector_key" in valid_sites.columns:
        sector_key = valid_sites["frontend_site_sector_key"].astype(str).str.strip()
    elif {"site_identity_key", "sector_identity"}.issubset(valid_sites.columns):
        sector_key = valid_sites["site_identity_key"].astype(str).str.strip() + "|" + valid_sites["sector_identity"].astype(str).str.strip()
    elif {"site", "sector"}.issubset(valid_sites.columns):
        sector_key = valid_sites["site"].astype(str).str.strip() + "|" + valid_sites["sector"].astype(str).str.strip()
    elif {"Site ID", "sector"}.issubset(valid_sites.columns):
        sector_key = valid_sites["Site ID"].astype(str).str.strip() + "|" + valid_sites["sector"].astype(str).str.strip()
    else:
        sector_key = valid_sites["Node_Cell_ID"].astype(str)
    invalid_sector = sector_key.isna() | sector_key.eq("") | sector_key.str.lower().isin({"nan", "none", "<na>"})
    valid_sites["_assignment_sector_key"] = sector_key.mask(invalid_sector, valid_sites["Node_Cell_ID"].astype(str))

    sector_centers = (
        valid_sites
        .groupby("_assignment_sector_key", dropna=False)
        .agg(site_lat=("lat", "mean"), site_lon=("lon", "mean"))
        .reset_index()
    )
    if sector_centers.empty:
        return pd.DataFrame()

    sector_to_cells = (
        valid_sites[["_assignment_sector_key", "Node_Cell_ID"]]
        .drop_duplicates()
        .groupby("_assignment_sector_key")["Node_Cell_ID"]
        .apply(lambda s: sorted(set(s.astype(str))))
        .to_dict()
    )

    grid_centers = (
        samples_df[["frontend_grid_id", "grid_center_lat", "grid_center_lon"]]
        .drop_duplicates(subset=["frontend_grid_id"])
        .copy()
    )
    max_cells_per_grid = max(1, int(max_cells_per_grid or 1))
    min_cells_per_grid = max(1, int(min_cells_per_grid or 1))
    radius_m = float(radius_m or 0.0)

    assignments = []
    site_lat = sector_centers["site_lat"].to_numpy(dtype=float)
    site_lon = sector_centers["site_lon"].to_numpy(dtype=float)
    sector_ids = sector_centers["_assignment_sector_key"].astype(str).to_numpy()
    for grid in grid_centers.itertuples(index=False):
        dist = haversine_m(float(grid.grid_center_lat), float(grid.grid_center_lon), site_lat, site_lon)
        within = np.where(dist <= radius_m)[0] if radius_m > 0 else np.array([], dtype=int)
        if within.size:
            selected = within[np.argsort(dist[within])[:max_cells_per_grid]]
            source = "within_radius_sector_all_bands"
        else:
            nearest_count = min(min_cells_per_grid, len(sector_ids))
            selected = np.argsort(dist)[:nearest_count]
            source = "nearest_sector_fallback_all_bands"
        for idx in selected:
            for node_cell_id in sector_to_cells.get(str(sector_ids[idx]), []):
                assignments.append(
                    {
                        "frontend_grid_id": str(grid.frontend_grid_id),
                        "Node_Cell_ID": str(node_cell_id),
                        "distance_m": float(dist[idx]),
                        "assignment_source": source,
                    }
                )

    assignment_df = pd.DataFrame(assignments)
    if ensure_all_cells and not assignment_df.empty:
        min_grids_per_cell = max(1, int(min_grids_per_cell or 1))
        assigned_cells = set(assignment_df["Node_Cell_ID"].astype(str))
        cell_centers = (
            valid_sites.groupby("Node_Cell_ID", dropna=False)
            .agg(site_lat=("lat", "mean"), site_lon=("lon", "mean"))
            .reset_index()
        )
        missing_sites = cell_centers.loc[~cell_centers["Node_Cell_ID"].astype(str).isin(assigned_cells)].copy()
        if not missing_sites.empty:
            grid_lat = pd.to_numeric(grid_centers["grid_center_lat"], errors="coerce").to_numpy(dtype=float)
            grid_lon = pd.to_numeric(grid_centers["grid_center_lon"], errors="coerce").to_numpy(dtype=float)
            backfill_rows = []
            for site in missing_sites.itertuples(index=False):
                dist = haversine_m(float(site.site_lat), float(site.site_lon), grid_lat, grid_lon)
                nearest = np.argsort(dist)[: min(min_grids_per_cell, len(grid_centers))]
                for idx in nearest:
                    backfill_rows.append(
                        {
                            "frontend_grid_id": str(grid_centers.iloc[idx]["frontend_grid_id"]),
                            "Node_Cell_ID": str(site.Node_Cell_ID),
                            "distance_m": float(dist[idx]),
                            "assignment_source": "cell_coverage_backfill",
                        }
                    )
            if backfill_rows:
                assignment_df = pd.concat([assignment_df, pd.DataFrame(backfill_rows)], ignore_index=True)
                print(
                    f"[LTE][FRONTEND_GRID_ASSIGN_BACKFILL] missing_cells={len(missing_sites)} "
                    f"backfill_rows={len(backfill_rows)}"
                )

    if assignment_df.empty:
        return pd.DataFrame()
    assignment_df = (
        assignment_df.sort_values(["frontend_grid_id", "distance_m", "Node_Cell_ID"])
        .drop_duplicates(subset=["frontend_grid_id", "Node_Cell_ID"], keep="first")
        .copy()
    )
    out = samples_df.merge(assignment_df[["frontend_grid_id", "Node_Cell_ID", "assignment_source"]], on="frontend_grid_id", how="inner")
    print(
        f"[LTE][FRONTEND_GRID_ASSIGN] grids={grid_centers['frontend_grid_id'].nunique()} "
        f"assigned_grids={assignment_df['frontend_grid_id'].nunique()} "
        f"cells={assignment_df['Node_Cell_ID'].nunique()} rows={len(out)} "
        f"fallback_grids={int((assignment_df['assignment_source'] == 'nearest_sector_fallback_all_bands').sum())} "
        f"sector_expansion=True"
    )
    return out.reset_index(drop=True)
