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


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def _numeric_series(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
    else:
        series = pd.Series(np.nan, index=df.index, dtype=float)
    return series.fillna(default).astype(float)


def _derive_frequency_series(df: pd.DataFrame, default: float = 1800.0) -> pd.Series:
    for col in ("frequency_mhz", "frequency", "band"):
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(default).astype(float)
        extracted = (
            df[col]
            .astype(str)
            .str.extract(r"(\d{3,4}(?:\.\d+)?)", expand=False)
        )
        numeric = pd.to_numeric(extracted, errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _antenna_gain_estimate(az_diff, elev_diff, max_gain=18.0):
    ah = np.minimum(12.0 * np.square(az_diff / 65.0), 30.0)
    av = np.minimum(12.0 * np.square(elev_diff / 6.0), 20.0)
    return max_gain - np.minimum(ah + av, 30.0)


def _estimate_candidate_rsrp(
    site_lat: np.ndarray,
    site_lon: np.ndarray,
    freq: np.ndarray,
    height: np.ndarray,
    tx_power: np.ndarray,
    azimuth: np.ndarray,
    etilt: np.ndarray,
    mtilt: np.ndarray,
    point_lat: float,
    point_lon: float,
    distance_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d_m = np.maximum(distance_m, 1.0)
    d_km = np.maximum(d_m / 1000.0, 0.001)
    h_rx = 1.5
    a_hm = (1.1 * np.log10(freq) - 0.7) * h_rx - (1.56 * np.log10(freq) - 0.8)
    base_pl = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(height) - a_hm + 3.0
    slope = 44.9 - 6.55 * np.log10(height)
    pathloss = base_pl + slope * np.log10(d_km)

    bearing = bearing_deg(
        site_lat,
        site_lon,
        point_lat,
        point_lon,
    )
    az_diff = np.abs((bearing - azimuth + 180.0) % 360.0 - 180.0)
    elev_angle = np.degrees(np.arctan2(h_rx - height, d_m))
    elev_diff = elev_angle + etilt + mtilt
    gain = _antenna_gain_estimate(az_diff, elev_diff)
    return tx_power + gain - pathloss - 2.0, az_diff


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
    min_candidate_rsrp_dbm: float = -128.0,
    candidate_safety_cap: int = 20,
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

    valid_sites = (
        valid_sites
        .sort_values(["Node_Cell_ID", "lat", "lon"])
        .drop_duplicates(subset=["Node_Cell_ID"], keep="first")
        .reset_index(drop=True)
    )

    grid_centers = (
        samples_df[["frontend_grid_id", "grid_center_lat", "grid_center_lon"]]
        .drop_duplicates(subset=["frontend_grid_id"])
        .copy()
    )
    candidate_safety_cap = max(1, int(candidate_safety_cap or max_cells_per_grid or 20))
    min_cells_per_grid = max(1, int(min_cells_per_grid or 1))
    radius_m = float(radius_m or 0.0)
    min_candidate_rsrp_dbm = float(min_candidate_rsrp_dbm)

    assignments = []
    site_lat = valid_sites["lat"].to_numpy(dtype=float)
    site_lon = valid_sites["lon"].to_numpy(dtype=float)
    node_cell_ids = valid_sites["Node_Cell_ID"].astype(str).to_numpy()
    site_freq = np.clip(_derive_frequency_series(valid_sites).to_numpy(dtype=float), 450.0, 3800.0)
    site_height = np.clip(_numeric_series(valid_sites, "antenna_height", 30.0).to_numpy(dtype=float), 5.0, 120.0)
    site_tx_power = _numeric_series(valid_sites, "tx_power", 46.0).to_numpy(dtype=float)
    site_azimuth = _numeric_series(valid_sites, "azimuth", 0.0).to_numpy(dtype=float)
    site_etilt = _numeric_series(valid_sites, "electrical_tilt", 3.0).to_numpy(dtype=float)
    site_mtilt = _numeric_series(valid_sites, "mechanical_tilt", 0.0).to_numpy(dtype=float)
    for grid in grid_centers.itertuples(index=False):
        dist = haversine_m(float(grid.grid_center_lat), float(grid.grid_center_lon), site_lat, site_lon)
        estimated_rsrp, az_diff = _estimate_candidate_rsrp(
            site_lat,
            site_lon,
            site_freq,
            site_height,
            site_tx_power,
            site_azimuth,
            site_etilt,
            site_mtilt,
            float(grid.grid_center_lat),
            float(grid.grid_center_lon),
            dist,
        )
        viable = np.where(estimated_rsrp >= min_candidate_rsrp_dbm)[0]
        if viable.size:
            selected = viable[np.argsort(estimated_rsrp[viable])[::-1][:candidate_safety_cap]]
            source = "rf_viable_cell_band"
        else:
            nearest_count = min(min_cells_per_grid, len(node_cell_ids))
            selected = np.argsort(estimated_rsrp)[::-1][:nearest_count]
            source = "rf_best_fallback_cell_band"
        for idx in selected:
            assignments.append(
                {
                    "frontend_grid_id": str(grid.frontend_grid_id),
                    "Node_Cell_ID": str(node_cell_ids[idx]),
                    "distance_m": float(dist[idx]),
                    "estimated_rsrp_dbm": float(estimated_rsrp[idx]),
                    "azimuth_delta_deg": float(az_diff[idx]),
                    "assignment_source": source,
                }
            )

    assignment_df = pd.DataFrame(assignments)
    if ensure_all_cells and not assignment_df.empty:
        min_grids_per_cell = max(1, int(min_grids_per_cell or 1))
        assigned_cells = set(assignment_df["Node_Cell_ID"].astype(str))
        missing_sites = valid_sites.loc[~valid_sites["Node_Cell_ID"].astype(str).isin(assigned_cells)].copy()
        if not missing_sites.empty:
            grid_lat = pd.to_numeric(grid_centers["grid_center_lat"], errors="coerce").to_numpy(dtype=float)
            grid_lon = pd.to_numeric(grid_centers["grid_center_lon"], errors="coerce").to_numpy(dtype=float)
            backfill_rows = []
            for site in missing_sites.itertuples(index=False):
                dist = haversine_m(float(site.lat), float(site.lon), grid_lat, grid_lon)
                nearest = np.argsort(dist)[: min(min_grids_per_cell, len(grid_centers))]
                for idx in nearest:
                    backfill_rows.append(
                        {
                            "frontend_grid_id": str(grid_centers.iloc[idx]["frontend_grid_id"]),
                            "Node_Cell_ID": str(site.Node_Cell_ID),
                            "distance_m": float(dist[idx]),
                            "estimated_rsrp_dbm": np.nan,
                            "azimuth_delta_deg": np.nan,
                            "assignment_source": "cell_band_coverage_backfill",
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
        assignment_df.sort_values(["frontend_grid_id", "estimated_rsrp_dbm", "distance_m", "Node_Cell_ID"], ascending=[True, False, True, True])
        .drop_duplicates(subset=["frontend_grid_id", "Node_Cell_ID"], keep="first")
        .copy()
    )
    out = samples_df.merge(assignment_df[["frontend_grid_id", "Node_Cell_ID", "assignment_source"]], on="frontend_grid_id", how="inner")
    print(
        f"[LTE][FRONTEND_GRID_ASSIGN] grids={grid_centers['frontend_grid_id'].nunique()} "
        f"assigned_grids={assignment_df['frontend_grid_id'].nunique()} "
        f"cells={assignment_df['Node_Cell_ID'].nunique()} rows={len(out)} "
        f"fallback_grids={int((assignment_df['assignment_source'] == 'rf_best_fallback_cell_band').sum())} "
        f"cell_band_assignment=True min_candidate_rsrp_dbm={min_candidate_rsrp_dbm} "
        f"candidate_safety_cap={candidate_safety_cap}"
    )
    return out.reset_index(drop=True)
