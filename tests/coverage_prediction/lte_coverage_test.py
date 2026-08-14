from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from pyproj import CRS, Transformer
from sqlalchemy import text
from shapely.geometry import box
from shapely.ops import transform
from shapely.wkt import loads as load_wkt
from tools.lte_tilt_recommandation.cell_identity import (
    build_rf_identity,
    build_sector_identity,
    build_site_sector_band_identity,
)

from tests.baseline.lte_rf_debug_lab import DEFAULT_PROJECT_ID, DEFAULT_REGION, _write_json
from tests.baseline.lte_rf_debug_lab import (
    BUILDING_TAGS,
    GREEN_TAGS,
    ROAD_TAGS,
    WATER_TAGS,
    _attach_line_density,
    _attach_polygon_area_ratio,
    _fetch_osm_layer,
)
from tests.coverage_prediction.coverage_artifact_locator import write_latest_coverage_artifact
from tools.lte_prediction import ml_engine as base_ml
from tools.lte_prediction_optimised import ml_engine as opt_ml


OUTPUT_ROOT = Path("tests/output")
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_POLYGON_WKT = (
    "POLYGON(("
    "77.3493010211505 28.6451999446618,"
    "77.3760801959551 28.6563475183659,"
    "77.3798996615923 28.6493804236681,"
    "77.3790413547076 28.6309248012639,"
    "77.3383146930255 28.6320924980605,"
    "77.3493010211505 28.6451999446618"
    "))"
)
DEFAULT_BUCKETS: Tuple[Tuple[str, str, str], ...] = (
    ("PART_1", "2025-08-06 00:00:00", "2025-11-07 23:59:59"),
    ("PART_2", "2025-11-08 00:00:00", "2026-02-10 23:59:59"),
    ("PART_3", "2026-02-11 00:00:00", "2026-05-16 23:59:59"),
)
DEFAULT_GEO_SNAPSHOT_TS_BY_BUCKET: Dict[str, str] = {
    "PART_1": "2023-05-31 23:59:59",
    "PART_2": "2024-05-31 23:59:59",
    "PART_3": "2026-05-31 23:59:59",
}
ENABLE_COVERAGE_GREEN_LAYER = False
REDUCED_COVERAGE_GREEN_TAGS = {
    "landuse": ["grass"],
    "natural": ["grassland"],
}
COVERAGE_CLUTTER_FORMULA_VERSION = "v3_park_open_driven_suburban_split"
TOPOLOGY_SELECTION_VERSION = "v2_site_cell_sector_growth_db_topology"
BAND_DIVERSITY_VERSION = "v2_bucket_band_mix_plus_multi_band_sectors"
BAND_MIX_PLAN = {
    "PART_1": {900: 0.15, 1800: 0.69, 850: 0.15, 2300: 0.01},
    "PART_2": {900: 0.11, 1800: 0.76, 850: 0.10, 2300: 0.03},
    "PART_3": {900: 0.08, 1800: 0.80, 850: 0.07, 2300: 0.05},
}
MULTI_BAND_SECTOR_SHARE_PLAN = {
    "PART_1": 0.06,
    "PART_2": 0.14,
    "PART_3": 0.18,
}
TRIPLE_BAND_SECTOR_SHARE_PLAN = {
    "PART_1": 0.01,
    "PART_2": 0.02,
    "PART_3": 0.04,
}
MULTI_BAND_SECONDARY_BAND_PLAN = {
    "PART_1": {700: 0.10, 850: 0.18, 900: 0.28, 2100: 0.24, 2300: 0.20},
    "PART_2": {700: 0.12, 850: 0.18, 900: 0.24, 2100: 0.24, 2300: 0.22},
    "PART_3": {700: 0.14, 850: 0.16, 900: 0.20, 2100: 0.24, 2300: 0.26},
}
SYNTHETIC_BAND_TO_EARFCN = {
    700: 700,
    850: 850,
    900: 900,
    1800: 1750,
    2100: 2100,
    2300: 2300,
}
CARRIER_LOAD_SHARE_BAND_WEIGHT = {
    700: 1.15,
    850: 1.10,
    900: 1.00,
    1800: 0.90,
    2100: 0.75,
    2300: 0.65,
}


@dataclass
class CoverageTestConfig:
    project_id: int = DEFAULT_PROJECT_ID
    site_project_id: Optional[int] = None
    region: str = DEFAULT_REGION
    polygon_wkt: str = DEFAULT_POLYGON_WKT
    buckets: Sequence[Tuple[str, str, str]] = DEFAULT_BUCKETS
    chunk_size: int = 10000
    grid_size_m: float = 50.0
    baseline_radius_m: float = 500.0
    baseline_workers: int = 3
    max_interference_sites: int = 50
    geo_cache_mode: str = "prebuilt"
    geo_base_run_dir: Optional[str] = None
    topology_operator: Optional[str] = "Airtel"
    use_frontend_grid_sampling: bool = True
    grid_analytics_scenario_id: Optional[int] = None
    output_root: Path = OUTPUT_ROOT


def _config_fingerprint(config: CoverageTestConfig) -> Dict[str, object]:
    return {
        "project_id": int(config.project_id),
        "site_project_id": int(config.site_project_id if config.site_project_id is not None else config.project_id),
        "region": str(config.region).lower(),
        "polygon_wkt": str(config.polygon_wkt),
        "buckets": [[str(label), str(start_ts), str(end_ts)] for label, start_ts, end_ts in config.buckets],
        "chunk_size": int(config.chunk_size),
        "grid_size_m": float(config.grid_size_m),
        "baseline_radius_m": float(config.baseline_radius_m),
        "baseline_workers": int(config.baseline_workers),
        "max_interference_sites": int(config.max_interference_sites),
        "topology_operator": str(config.topology_operator or ""),
        "use_frontend_grid_sampling": bool(config.use_frontend_grid_sampling),
        "grid_analytics_scenario_id": (
            int(config.grid_analytics_scenario_id) if config.grid_analytics_scenario_id is not None else None
        ),
        "topology_plan": _default_topology_plan(),
        "band_mix_plan": BAND_MIX_PLAN,
    }


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _make_cache_key(payload: object) -> str:
    return hashlib.sha1(_stable_json_dumps(payload).encode("utf-8")).hexdigest()[:16]


def _cache_root_for_project(project_id: int) -> Path:
    return _ensure_dir(OUTPUT_ROOT / f"project_{project_id}" / "coverage_cache")


def _write_json_file(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json_file(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache_df(path_base: Path, df: pd.DataFrame) -> Path:
    parquet_path = path_base.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        pickle_path = path_base.with_suffix(".pkl")
        df.to_pickle(pickle_path)
        return pickle_path


def _read_cache_df(path_base: Path) -> Optional[pd.DataFrame]:
    parquet_path = path_base.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    pickle_path = path_base.with_suffix(".pkl")
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    return None


def _load_bucket_summary_from_run_dir(run_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if run_dir is None:
        return None
    path = run_dir / "bucket_summary.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    print(f"[COVERAGE_TEST][RUN_REUSE] artifact=bucket_summary.csv rows={len(df)} path={path}")
    return df


def _load_coverage_rows_from_run_dir(run_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if run_dir is None:
        return None
    path = run_dir / "coverage_rows.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    df = _coerce_numeric_columns(df)
    print(f"[COVERAGE_TEST][RUN_REUSE] artifact=coverage_rows.csv rows={len(df)} path={path}")
    return df


def _load_project_sites_from_run_dir(run_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if run_dir is None:
        return None
    path = run_dir / "project_sites.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    print(f"[COVERAGE_TEST][RUN_REUSE] artifact=project_sites.csv rows={len(df)} path={path}")
    return df


def _resolve_engine(region: str):
    current_engine = opt_ml.engine.get(region.lower(), opt_ml.engine["india"])
    if current_engine is None:
        raise RuntimeError(f"No DB engine configured for region={region}")
    return current_engine


def _bucket_case_sql(buckets: Sequence[Tuple[str, str, str]]) -> str:
    parts: List[str] = ["CASE"]
    for label, start_ts, end_ts in buckets:
        parts.append(
            "WHEN `timestamp` >= '{start}' AND `timestamp` <= '{end}' THEN '{label}'".format(
                start=start_ts,
                end=end_ts,
                label=label,
            )
        )
    parts.append("END")
    return "\n        ".join(parts)


def _overall_time_bounds(buckets: Sequence[Tuple[str, str, str]]) -> Tuple[str, str]:
    starts = [start_ts for _, start_ts, _ in buckets]
    ends = [end_ts for _, _, end_ts in buckets]
    return min(starts), max(ends)


def _fetch_bucket_summary(engine, polygon_wkt: str, buckets: Sequence[Tuple[str, str, str]]) -> pd.DataFrame:
    bucket_case = _bucket_case_sql(buckets)
    overall_start, overall_end = _overall_time_bounds(buckets)
    sql = text(
        f"""
        SELECT
            time_bucket,
            MIN(`timestamp`) AS start_time,
            MAX(`timestamp`) AS end_time,
            COUNT(*) AS total_rows
        FROM (
            SELECT
                {bucket_case} AS time_bucket,
                `timestamp`
            FROM Stracer.tbl_network_log
            WHERE lat IS NOT NULL
              AND lon IS NOT NULL
              AND `timestamp` >= :overall_start
              AND `timestamp` <= :overall_end
              AND ST_CONTAINS(ST_GeomFromText(:polygon_wkt), POINT(lon, lat))
        ) bucketed
        WHERE time_bucket IS NOT NULL
        GROUP BY time_bucket
        ORDER BY time_bucket
        """
    )
    return pd.read_sql(
        sql,
        engine,
        params={
            "polygon_wkt": polygon_wkt,
            "overall_start": overall_start,
            "overall_end": overall_end,
        },
    )


def _fetch_detail_rows_chunked(
    engine,
    polygon_wkt: str,
    buckets: Sequence[Tuple[str, str, str]],
    chunk_size: int,
) -> pd.DataFrame:
    bucket_case = _bucket_case_sql(buckets)
    overall_start, overall_end = _overall_time_bounds(buckets)
    sql = text(
        f"""
        SELECT
            id,
            {bucket_case} AS time_bucket,
            session_id,
            `timestamp`,
            lat,
            lon,
            network,
            m_alpha_long,
            m_alpha_short,
            cell_id,
            nodeb_id,
            pci,
            earfcn,
            band,
            rsrp,
            rsrq,
            sinr,
            rssi,
            cqi,
            dl_tpt,
            ul_tpt,
            latency,
            jitter,
            packet_loss,
            mos,
            speed,
            bw,
            phone_heading,
            ta,
            csi_rsrp,
            csi_rsrq,
            csi_sinr
        FROM Stracer.tbl_network_log
        WHERE lat IS NOT NULL
          AND lon IS NOT NULL
          AND `timestamp` >= :overall_start
          AND `timestamp` <= :overall_end
          AND id > :last_id
          AND ST_CONTAINS(ST_GeomFromText(:polygon_wkt), POINT(lon, lat))
        ORDER BY id
        LIMIT :chunk_size
        """
    )
    chunks: List[pd.DataFrame] = []
    last_id = 0
    total_rows = 0
    chunk_index = 0
    while True:
        chunk_df = pd.read_sql(
            sql,
            engine,
            params={
                "polygon_wkt": polygon_wkt,
                "overall_start": overall_start,
                "overall_end": overall_end,
                "last_id": int(last_id),
                "chunk_size": int(chunk_size),
            },
        )
        if chunk_df.empty:
            break
        chunk_index += 1
        total_rows += int(len(chunk_df))
        last_id = int(pd.to_numeric(chunk_df["id"], errors="coerce").max())
        chunks.append(chunk_df)
        print(
            f"[COVERAGE_TEST][CHUNK] index={chunk_index} rows={len(chunk_df)} "
            f"total_rows={total_rows} last_id={last_id}"
        )
        if len(chunk_df) < int(chunk_size):
            break
    if not chunks:
        return pd.DataFrame()
    detail_df = pd.concat(chunks, ignore_index=True)
    detail_df = detail_df.loc[detail_df["time_bucket"].notna()].copy()
    return detail_df.sort_values(["time_bucket", "timestamp", "id"]).reset_index(drop=True)


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "lat",
        "lon",
        "rsrp",
        "rsrq",
        "sinr",
        "rssi",
        "cqi",
        "dl_tpt",
        "ul_tpt",
        "bw",
        "latency",
        "jitter",
        "packet_loss",
        "mos",
        "speed",
        "phone_heading",
        "ta",
        "csi_rsrp",
        "csi_rsrq",
        "csi_sinr",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out


def _utm_crs_for_polygon(polygon_wkt: str) -> CRS:
    polygon = load_wkt(polygon_wkt)
    centroid = polygon.centroid
    zone = int((centroid.x + 180.0) // 6.0) + 1
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def _build_polygon_grid(polygon_wkt: str, grid_size_m: float):
    polygon_wgs84 = load_wkt(polygon_wkt)
    utm_crs = _utm_crs_for_polygon(polygon_wkt)
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    polygon_utm = transform(to_utm.transform, polygon_wgs84)
    minx, miny, maxx, maxy = polygon_utm.bounds

    cols = int(np.ceil((maxx - minx) / float(grid_size_m)))
    rows = int(np.ceil((maxy - miny) / float(grid_size_m)))
    grid_rows: List[Dict[str, object]] = []
    grid_id = 1

    for row_idx in range(rows):
        y0 = miny + row_idx * float(grid_size_m)
        y1 = y0 + float(grid_size_m)
        for col_idx in range(cols):
            x0 = minx + col_idx * float(grid_size_m)
            x1 = x0 + float(grid_size_m)
            clipped = box(x0, y0, x1, y1).intersection(polygon_utm)
            if clipped.is_empty or clipped.area <= 0:
                continue
            centroid_utm = clipped.centroid
            centroid_lon, centroid_lat = to_wgs84.transform(centroid_utm.x, centroid_utm.y)
            geom_wgs84 = transform(to_wgs84.transform, clipped)
            grid_rows.append(
                {
                    "grid_id": int(grid_id),
                    "grid_row": int(row_idx),
                    "grid_col": int(col_idx),
                    "grid_size_m": float(grid_size_m),
                    "centroid_lat": float(centroid_lat),
                    "centroid_lon": float(centroid_lon),
                    "grid_area_m2": float(clipped.area),
                    "geometry_wkt": geom_wgs84.wkt,
                }
            )
            grid_id += 1

    return pd.DataFrame(grid_rows), utm_crs, (minx, miny)


def _assign_points_to_grid(
    detail_df: pd.DataFrame,
    grid_size_m: float,
    utm_crs: CRS,
    origin_xy: Tuple[float, float],
    grid_cells_df: pd.DataFrame,
) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df.copy()

    out = detail_df.copy()
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    minx, miny = origin_xy
    cell_lookup = {
        (int(row.grid_row), int(row.grid_col)): (
            int(row.grid_id),
            float(row.centroid_lat),
            float(row.centroid_lon),
        )
        for row in grid_cells_df.itertuples(index=False)
    }

    grid_ids: List[object] = []
    grid_rows: List[object] = []
    grid_cols: List[object] = []
    centroid_lats: List[object] = []
    centroid_lons: List[object] = []

    for lon, lat in zip(pd.to_numeric(out["lon"], errors="coerce"), pd.to_numeric(out["lat"], errors="coerce")):
        if pd.isna(lon) or pd.isna(lat):
            grid_ids.append(pd.NA)
            grid_rows.append(pd.NA)
            grid_cols.append(pd.NA)
            centroid_lats.append(pd.NA)
            centroid_lons.append(pd.NA)
            continue
        x, y = to_utm.transform(float(lon), float(lat))
        col_idx = int(np.floor((x - minx) / float(grid_size_m)))
        row_idx = int(np.floor((y - miny) / float(grid_size_m)))
        match = cell_lookup.get((row_idx, col_idx))
        if match is None:
            grid_ids.append(pd.NA)
            grid_rows.append(pd.NA)
            grid_cols.append(pd.NA)
            centroid_lats.append(pd.NA)
            centroid_lons.append(pd.NA)
            continue
        grid_id, centroid_lat, centroid_lon = match
        grid_ids.append(grid_id)
        grid_rows.append(row_idx)
        grid_cols.append(col_idx)
        centroid_lats.append(centroid_lat)
        centroid_lons.append(centroid_lon)

    out["grid_id"] = grid_ids
    out["grid_row"] = grid_rows
    out["grid_col"] = grid_cols
    out["grid_centroid_lat"] = centroid_lats
    out["grid_centroid_lon"] = centroid_lons
    return out


def _build_bucket_grid_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty or "grid_id" not in detail_df.columns:
        return pd.DataFrame()
    work = detail_df.loc[detail_df["grid_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for bucket_name, bucket_df in work.groupby("time_bucket", dropna=True):
        rows.append(
            {
                "time_bucket": str(bucket_name),
                "mapped_rows": int(len(bucket_df)),
                "unique_grids": int(bucket_df["grid_id"].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _derive_coverage_clutter_class(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    for col in [
        "building_area_ratio",
        "road_length_m",
        "road_density",
        "building_count",
        "green_ratio",
        "water_ratio",
        "park_open_area",
        "open_area_ratio",
    ]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    effective_green = (
        work["green_ratio"].clip(lower=0.0)
        + (0.65 * work["park_open_area"].clip(lower=0.0))
        + (0.35 * work["open_area_ratio"].clip(lower=0.0))
    )

    building_hi = work["building_area_ratio"] >= work["building_area_ratio"].quantile(0.88)
    building_mid = work["building_area_ratio"] >= work["building_area_ratio"].quantile(0.58)
    building_low = work["building_area_ratio"] >= work["building_area_ratio"].quantile(0.18)

    road_hi = work["road_density"] >= work["road_density"].quantile(0.72)
    road_mid = work["road_density"] >= work["road_density"].quantile(0.42)
    road_low = work["road_density"] >= work["road_density"].quantile(0.12)

    building_count_hi = work["building_count"] >= work["building_count"].quantile(0.70)
    building_count_mid = work["building_count"] >= work["building_count"].quantile(0.35)

    clutter = np.full(len(work), "Rural/Open", dtype=object)
    clutter = np.where(work["water_ratio"] >= 0.12, "Water", clutter)
    clutter = np.where(
        (effective_green >= 0.25) & (work["building_area_ratio"] < 0.05) & (work["water_ratio"] < 0.12),
        "Vegetation",
        clutter,
    )
    clutter = np.where(
        (clutter == "Rural/Open") & building_hi & road_hi & building_count_hi,
        "Dense Urban",
        clutter,
    )
    clutter = np.where(
        (clutter == "Rural/Open") & ((building_mid & road_mid) | (building_hi & building_count_mid) | (road_hi & (effective_green < 0.18))),
        "Urban",
        clutter,
    )
    clutter = np.where(
        (clutter == "Rural/Open") & (
            ((building_low & road_low) | (work["building_count"] > 0) | (work["road_length_m"] > 0))
            & (effective_green < 0.32)
        ),
        "Suburban",
        clutter,
    )
    clutter = np.where(
        (clutter == "Rural/Open") & (effective_green >= 0.12) & (work["building_area_ratio"] < work["building_area_ratio"].quantile(0.45)),
        "Suburban",
        clutter,
    )
    return pd.Series(clutter, index=work.index)


def _attach_building_features(grid_gdf: gpd.GeoDataFrame, building_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid_utm = grid_gdf.to_crs(_utm_crs_for_polygon(grid_gdf.geometry.union_all().wkt))
    grid_utm["building_count"] = 0.0
    grid_utm["building_area_sum_m2"] = 0.0
    grid_utm["avg_building_area_m2"] = 0.0
    if building_gdf.empty:
        grid_utm["building_area_ratio"] = 0.0
        return grid_utm.to_crs("EPSG:4326")

    bld_utm = building_gdf.to_crs(grid_utm.crs).copy()
    bld_utm = bld_utm[bld_utm.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if bld_utm.empty:
        grid_utm["building_area_ratio"] = 0.0
        return grid_utm.to_crs("EPSG:4326")

    bld_utm["building_area_m2"] = bld_utm.geometry.area
    centroids = bld_utm.copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin(
        centroids[["building_area_m2", "geometry"]],
        grid_utm[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    )
    if not joined.empty:
        agg = joined.groupby("grid_id").agg(
            building_count=("building_area_m2", "size"),
            building_area_sum_m2=("building_area_m2", "sum"),
            avg_building_area_m2=("building_area_m2", "mean"),
        )
        grid_utm = grid_utm.merge(agg, on="grid_id", how="left", suffixes=("", "_calc"))
        for col in ["building_count", "building_area_sum_m2", "avg_building_area_m2"]:
            calc_col = f"{col}_calc"
            if calc_col in grid_utm.columns:
                grid_utm[col] = pd.to_numeric(grid_utm[calc_col], errors="coerce").fillna(0.0)
            else:
                grid_utm[col] = pd.to_numeric(grid_utm[col], errors="coerce").fillna(0.0)
        grid_utm = grid_utm.drop(columns=[f"{col}_calc" for col in ["building_count", "building_area_sum_m2", "avg_building_area_m2"]], errors="ignore")
    grid_utm["building_area_ratio"] = (
        pd.to_numeric(grid_utm["building_area_sum_m2"], errors="coerce").fillna(0.0)
        / pd.to_numeric(grid_utm["cell_area_m2"], errors="coerce").replace(0, np.nan)
    ).fillna(0.0)
    return grid_utm.to_crs("EPSG:4326")


def _build_building_features_for_snapshot(
    grid_cells_df: pd.DataFrame,
    polygon_wkt: str,
    cache_dir: Path,
    snapshot_ts_utc: str,
) -> pd.DataFrame:
    if grid_cells_df.empty:
        return pd.DataFrame(
            columns=[
                "grid_id",
                "building_count",
                "building_area_sum_m2",
                "avg_building_area_m2",
                "building_area_ratio",
                "coverage_buildings_mode",
            ]
        )

    grid_gdf = gpd.GeoDataFrame(
        grid_cells_df.drop(columns=["geometry_wkt"]),
        geometry=gpd.GeoSeries.from_wkt(grid_cells_df["geometry_wkt"]),
        crs="EPSG:4326",
    )
    if "cell_area_m2" not in grid_gdf.columns and "grid_area_m2" in grid_gdf.columns:
        grid_gdf["cell_area_m2"] = pd.to_numeric(grid_gdf["grid_area_m2"], errors="coerce")
    polygon_gdf = gpd.GeoDataFrame(geometry=[load_wkt(polygon_wkt)], crs="EPSG:4326")
    building_gdf, building_mode = _fetch_osm_layer_snapshot_with_fallback(
        "coverage_buildings",
        polygon_gdf,
        BUILDING_TAGS,
        cache_dir,
        snapshot_ts_utc,
    )
    grid_gdf = _attach_building_features(grid_gdf, building_gdf)
    out = pd.DataFrame(grid_gdf.drop(columns="geometry"))[
        ["grid_id", "building_count", "building_area_sum_m2", "avg_building_area_m2", "building_area_ratio"]
    ].copy()
    out["coverage_buildings_mode"] = building_mode
    return out


def _attach_point_presence(grid_gdf: gpd.GeoDataFrame, point_gdf: gpd.GeoDataFrame, out_col: str) -> gpd.GeoDataFrame:
    out = grid_gdf.copy()
    out[out_col] = 0.0
    if point_gdf.empty:
        return out
    pts = point_gdf.copy()
    pts = pts[pts.geometry.notnull() & ~pts.geometry.is_empty].copy()
    poly_mask = pts.geometry.type.isin(["Polygon", "MultiPolygon"])
    if poly_mask.any():
        pts.loc[poly_mask, "geometry"] = pts.loc[poly_mask, "geometry"].centroid
    pts = pts[pts.geometry.type.isin(["Point", "MultiPoint"])].copy()
    if pts.empty:
        return out
    joined = gpd.sjoin(
        pts[["geometry"]],
        out[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    )
    if joined.empty:
        return out
    counts = joined.groupby("grid_id").size().rename(f"{out_col}_count").reset_index()
    out = out.merge(counts, on="grid_id", how="left")
    out[out_col] = np.where(pd.to_numeric(out.get(f"{out_col}_count"), errors="coerce").fillna(0) > 0, 1.0, 0.0)
    return out.drop(columns=[f"{out_col}_count"], errors="ignore")


def _build_static_grid_geo_features(
    grid_cells_df: pd.DataFrame,
    polygon_wkt: str,
    cache_dir: Path,
) -> pd.DataFrame:
    if grid_cells_df.empty:
        return pd.DataFrame()

    grid_gdf = gpd.GeoDataFrame(
        grid_cells_df.drop(columns=["geometry_wkt"]),
        geometry=gpd.GeoSeries.from_wkt(grid_cells_df["geometry_wkt"]),
        crs="EPSG:4326",
    )
    if "cell_area_m2" not in grid_gdf.columns and "grid_area_m2" in grid_gdf.columns:
        grid_gdf["cell_area_m2"] = pd.to_numeric(grid_gdf["grid_area_m2"], errors="coerce")
    polygon_gdf = gpd.GeoDataFrame(geometry=[load_wkt(polygon_wkt)], crs="EPSG:4326")

    roads_gdf = _fetch_osm_layer("coverage_roads", polygon_gdf, ROAD_TAGS, cache_dir)
    green_gdf = (
        _fetch_osm_layer("coverage_green", polygon_gdf, REDUCED_COVERAGE_GREEN_TAGS, cache_dir)
        if ENABLE_COVERAGE_GREEN_LAYER
        else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    )
    water_gdf = _fetch_osm_layer("coverage_water", polygon_gdf, WATER_TAGS, cache_dir)
    building_gdf = _fetch_osm_layer("coverage_buildings", polygon_gdf, BUILDING_TAGS, cache_dir)
    mall_gdf = _fetch_osm_layer("coverage_mall", polygon_gdf, {"shop": ["mall"], "building": ["retail", "commercial"]}, cache_dir)
    metro_gdf = _fetch_osm_layer("coverage_metro", polygon_gdf, {"railway": ["station"], "station": ["subway"], "public_transport": ["station"]}, cache_dir)
    park_gdf = _fetch_osm_layer("coverage_park", polygon_gdf, {"leisure": ["park", "garden"], "landuse": ["recreation_ground"]}, cache_dir)
    open_gdf = _fetch_osm_layer("coverage_open", polygon_gdf, {"landuse": ["grass", "meadow", "farmland"], "natural": ["grassland", "scrub"]}, cache_dir)

    grid_gdf["road_length_m"] = 0.0
    grid_gdf["green_ratio"] = 0.0
    grid_gdf["water_ratio"] = 0.0
    grid_gdf = _attach_line_density(grid_gdf, roads_gdf, "road_length_m")
    grid_gdf = _attach_polygon_area_ratio(grid_gdf, green_gdf, "green_ratio")
    grid_gdf = _attach_polygon_area_ratio(grid_gdf, water_gdf, "water_ratio")
    grid_gdf = _attach_building_features(grid_gdf, building_gdf)
    grid_gdf = _attach_polygon_area_ratio(grid_gdf, park_gdf, "park_open_area")
    grid_gdf = _attach_polygon_area_ratio(grid_gdf, open_gdf, "open_area_ratio")
    grid_gdf = _attach_point_presence(grid_gdf, mall_gdf, "mall_presence")
    grid_gdf = _attach_point_presence(grid_gdf, metro_gdf, "metro_presence")
    grid_gdf["road_density"] = (
        pd.to_numeric(grid_gdf["road_length_m"], errors="coerce").fillna(0.0)
        / pd.to_numeric(grid_gdf["cell_area_m2"], errors="coerce").replace(0, np.nan)
    ).fillna(0.0)
    grid_gdf["clutter_class"] = _derive_coverage_clutter_class(pd.DataFrame(grid_gdf.drop(columns="geometry")))

    out = pd.DataFrame(grid_gdf.drop(columns="geometry"))
    out["geo_snapshot_mode"] = "current_static_reused"
    return out


def _snapshot_utc_string(ts_text: str) -> str:
    cleaned = str(ts_text).strip()
    if not cleaned:
        return ""
    return cleaned.replace(" ", "T") + "Z"


def _split_polygon_for_osm_snapshot(
    polygon_wkt: str,
    tile_side_m: float = 1000.0,
) -> List[object]:
    polygon_wgs84 = load_wkt(polygon_wkt)
    utm_crs = _utm_crs_for_polygon(polygon_wkt)
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    polygon_utm = transform(to_utm.transform, polygon_wgs84)
    minx, miny, maxx, maxy = polygon_utm.bounds
    cols = int(np.ceil((maxx - minx) / float(tile_side_m)))
    rows = int(np.ceil((maxy - miny) / float(tile_side_m)))
    pieces: List[object] = []
    for row_idx in range(rows):
        y0 = miny + row_idx * float(tile_side_m)
        y1 = y0 + float(tile_side_m)
        for col_idx in range(cols):
            x0 = minx + col_idx * float(tile_side_m)
            x1 = x0 + float(tile_side_m)
            clipped = polygon_utm.intersection(box(x0, y0, x1, y1))
            if clipped.is_empty or clipped.area <= 0:
                continue
            pieces.append(transform(to_wgs84.transform, clipped))
    return pieces


def _fetch_osm_layer_snapshot(
    name: str,
    polygon_gdf: gpd.GeoDataFrame,
    tags: Dict,
    cache_dir: Path,
    snapshot_ts_utc: str,
) -> gpd.GeoDataFrame:
    snapshot_key = snapshot_ts_utc.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    cache_path = cache_dir / f"{name}_{snapshot_key}.geojson"
    if cache_path.exists():
        return gpd.read_file(cache_path)

    geom = polygon_gdf.geometry.union_all()
    polygon_wkt = geom.wkt
    pieces = _split_polygon_for_osm_snapshot(polygon_wkt, tile_side_m=1000.0)
    previous_timeout = ox.settings.requests_timeout
    previous_use_cache = ox.settings.use_cache
    previous_cache_folder = ox.settings.cache_folder
    previous_overpass_settings = ox.settings.overpass_settings
    fetched_parts: List[gpd.GeoDataFrame] = []
    try:
        ox.settings.requests_timeout = 120
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(cache_dir)
        ox.settings.overpass_settings = f'[out:json][timeout:{{timeout}}]{{maxsize}}[date:"{snapshot_ts_utc}"]'
        for idx, piece in enumerate(pieces, start=1):
            try:
                part_gdf = ox.features_from_polygon(piece, tags=tags)
            except Exception as exc:
                print(
                    f"[TEST][OSM][SNAPSHOT] layer={name} snapshot={snapshot_ts_utc} "
                    f"tile={idx}/{len(pieces)} skipped reason={exc}"
                )
                continue
            if part_gdf is None or part_gdf.empty:
                print(
                    f"[TEST][OSM][SNAPSHOT] layer={name} snapshot={snapshot_ts_utc} "
                    f"tile={idx}/{len(pieces)} rows=0"
                )
                continue
            part_gdf = part_gdf.reset_index(drop=False)
            if part_gdf.crs is None:
                part_gdf = part_gdf.set_crs("EPSG:4326")
            else:
                part_gdf = part_gdf.to_crs("EPSG:4326")
            fetched_parts.append(part_gdf)
            print(
                f"[TEST][OSM][SNAPSHOT] layer={name} snapshot={snapshot_ts_utc} "
                f"tile={idx}/{len(pieces)} rows={len(part_gdf)}"
            )
    finally:
        ox.settings.requests_timeout = previous_timeout
        ox.settings.use_cache = previous_use_cache
        ox.settings.cache_folder = previous_cache_folder
        ox.settings.overpass_settings = previous_overpass_settings

    if not fetched_parts:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    gdf = pd.concat(fetched_parts, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    dedup_cols = [col for col in ["element", "id", "osmid", "geometry"] if col in gdf.columns]
    if dedup_cols:
        gdf = gdf.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    else:
        gdf = gdf.drop_duplicates().reset_index(drop=True)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(cache_path, driver="GeoJSON")
    return gdf


def _fetch_osm_layer_snapshot_with_fallback(
    name: str,
    polygon_gdf: gpd.GeoDataFrame,
    tags: Dict,
    cache_dir: Path,
    snapshot_ts_utc: str,
) -> tuple[gpd.GeoDataFrame, str]:
    snapshot_gdf = _fetch_osm_layer_snapshot(name, polygon_gdf, tags, cache_dir, snapshot_ts_utc)
    if not snapshot_gdf.empty:
        return snapshot_gdf, "historical_snapshot"

    print(
        f"[TEST][OSM][FALLBACK] layer={name} snapshot={snapshot_ts_utc} "
        f"reason=empty_or_timeout_using_current_static=True"
    )
    fallback_dir = _ensure_dir(cache_dir / "fallback_current_static")
    fallback_gdf = _fetch_osm_layer(name, polygon_gdf, tags, fallback_dir)
    if fallback_gdf is None or fallback_gdf.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326"), "empty_after_fallback"
    if fallback_gdf.crs is None:
        fallback_gdf = fallback_gdf.set_crs("EPSG:4326")
    else:
        fallback_gdf = fallback_gdf.to_crs("EPSG:4326")
    return fallback_gdf, "current_static_fallback"


def _build_grid_geo_features_for_snapshot(
    grid_cells_df: pd.DataFrame,
    polygon_wkt: str,
    cache_dir: Path,
    snapshot_ts_utc: str,
) -> pd.DataFrame:
    if grid_cells_df.empty:
        return pd.DataFrame()

    grid_gdf = gpd.GeoDataFrame(
        grid_cells_df.drop(columns=["geometry_wkt"]),
        geometry=gpd.GeoSeries.from_wkt(grid_cells_df["geometry_wkt"]),
        crs="EPSG:4326",
    )
    if "cell_area_m2" not in grid_gdf.columns and "grid_area_m2" in grid_gdf.columns:
        grid_gdf["cell_area_m2"] = pd.to_numeric(grid_gdf["grid_area_m2"], errors="coerce")
    polygon_gdf = gpd.GeoDataFrame(geometry=[load_wkt(polygon_wkt)], crs="EPSG:4326")

    layer_modes: Dict[str, str] = {}
    roads_gdf, layer_modes["coverage_roads"] = _fetch_osm_layer_snapshot_with_fallback("coverage_roads", polygon_gdf, ROAD_TAGS, cache_dir, snapshot_ts_utc)
    if ENABLE_COVERAGE_GREEN_LAYER:
        green_gdf, layer_modes["coverage_green"] = _fetch_osm_layer_snapshot_with_fallback("coverage_green", polygon_gdf, REDUCED_COVERAGE_GREEN_TAGS, cache_dir, snapshot_ts_utc)
    else:
        green_gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        layer_modes["coverage_green"] = "disabled_for_latency"
    water_gdf, layer_modes["coverage_water"] = _fetch_osm_layer_snapshot_with_fallback("coverage_water", polygon_gdf, WATER_TAGS, cache_dir, snapshot_ts_utc)
    building_gdf, layer_modes["coverage_buildings"] = _fetch_osm_layer_snapshot_with_fallback("coverage_buildings", polygon_gdf, BUILDING_TAGS, cache_dir, snapshot_ts_utc)
    mall_gdf, layer_modes["coverage_mall"] = _fetch_osm_layer_snapshot_with_fallback("coverage_mall", polygon_gdf, {"shop": ["mall"], "building": ["retail", "commercial"]}, cache_dir, snapshot_ts_utc)
    metro_gdf, layer_modes["coverage_metro"] = _fetch_osm_layer_snapshot_with_fallback("coverage_metro", polygon_gdf, {"railway": ["station"], "station": ["subway"], "public_transport": ["station"]}, cache_dir, snapshot_ts_utc)
    park_gdf, layer_modes["coverage_park"] = _fetch_osm_layer_snapshot_with_fallback("coverage_park", polygon_gdf, {"leisure": ["park", "garden"], "landuse": ["recreation_ground"]}, cache_dir, snapshot_ts_utc)
    open_gdf, layer_modes["coverage_open"] = _fetch_osm_layer_snapshot_with_fallback("coverage_open", polygon_gdf, {"landuse": ["grass", "meadow", "farmland"], "natural": ["grassland", "scrub"]}, cache_dir, snapshot_ts_utc)

    grid_gdf["road_length_m"] = 0.0
    grid_gdf["green_ratio"] = 0.0
    grid_gdf["water_ratio"] = 0.0
    grid_gdf = _attach_line_density(grid_gdf, roads_gdf, "road_length_m")
    grid_gdf = _attach_polygon_area_ratio(grid_gdf, green_gdf, "green_ratio")
    grid_gdf = _attach_polygon_area_ratio(grid_gdf, water_gdf, "water_ratio")
    grid_gdf = _attach_building_features(grid_gdf, building_gdf)
    grid_gdf = _attach_polygon_area_ratio(grid_gdf, park_gdf, "park_open_area")
    grid_gdf = _attach_polygon_area_ratio(grid_gdf, open_gdf, "open_area_ratio")
    grid_gdf = _attach_point_presence(grid_gdf, mall_gdf, "mall_presence")
    grid_gdf = _attach_point_presence(grid_gdf, metro_gdf, "metro_presence")
    grid_gdf["road_density"] = (
        pd.to_numeric(grid_gdf["road_length_m"], errors="coerce").fillna(0.0)
        / pd.to_numeric(grid_gdf["cell_area_m2"], errors="coerce").replace(0, np.nan)
    ).fillna(0.0)
    grid_gdf["clutter_class"] = _derive_coverage_clutter_class(pd.DataFrame(grid_gdf.drop(columns="geometry")))

    out = pd.DataFrame(grid_gdf.drop(columns="geometry"))
    out["geo_snapshot_mode"] = "osm_overpass_attic"
    out["geo_snapshot_ts_utc"] = snapshot_ts_utc
    out["geo_layer_modes_json"] = json.dumps(layer_modes, sort_keys=True)
    return out


def _build_bucket_grid_geo_features(
    grid_cells_df: pd.DataFrame,
    polygon_wkt: str,
    cache_root: Path,
    buckets: Sequence[Tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if grid_cells_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    bucket_rows: List[pd.DataFrame] = []
    latest_grid_geo_df = pd.DataFrame()
    for label, start_ts, end_ts in buckets:
        snapshot_source_ts = DEFAULT_GEO_SNAPSHOT_TS_BY_BUCKET.get(str(label), end_ts)
        snapshot_ts_utc = _snapshot_utc_string(snapshot_source_ts)
        bucket_cache_dir = _ensure_dir(cache_root / str(label).lower())
        bucket_geo_df = _build_grid_geo_features_for_snapshot(
            grid_cells_df=grid_cells_df,
            polygon_wkt=polygon_wkt,
            cache_dir=bucket_cache_dir,
            snapshot_ts_utc=snapshot_ts_utc,
        )
        if bucket_geo_df.empty:
            continue
        bucket_geo_df["time_bucket"] = str(label)
        bucket_geo_df["bucket_start"] = str(start_ts)
        bucket_geo_df["bucket_end"] = str(end_ts)
        bucket_geo_df["geo_snapshot_source_ts"] = str(snapshot_source_ts)
        bucket_rows.append(bucket_geo_df)
        latest_grid_geo_df = bucket_geo_df.drop(columns=["time_bucket", "bucket_start", "bucket_end"], errors="ignore").copy()

    return latest_grid_geo_df, (pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame())


def _load_or_build_df(
    cache_dir: Path,
    cache_name: str,
    cache_key_payload: object,
    builder,
) -> pd.DataFrame:
    cache_key = _make_cache_key(cache_key_payload)
    path_base = cache_dir / f"{cache_name}_{cache_key}"
    cached = _read_cache_df(path_base)
    if cached is not None:
        print(f"[COVERAGE_TEST][CACHE_HIT] name={cache_name} key={cache_key} rows={len(cached)}")
        return cached
    print(f"[COVERAGE_TEST][CACHE_MISS] name={cache_name} key={cache_key}")
    built = builder()
    _write_cache_df(path_base, built)
    return built


def _load_or_build_df_from_any_cache_dir(
    cache_dirs: Sequence[Path],
    cache_name: str,
    cache_key_payload: object,
    builder,
) -> pd.DataFrame:
    cache_key = _make_cache_key(cache_key_payload)
    for cache_dir in cache_dirs:
        path_base = cache_dir / f"{cache_name}_{cache_key}"
        cached = _read_cache_df(path_base)
        if cached is not None:
            print(f"[COVERAGE_TEST][CACHE_HIT] name={cache_name} key={cache_key} rows={len(cached)} source_cache_dir={cache_dir}")
            return cached
    print(f"[COVERAGE_TEST][CACHE_MISS] name={cache_name} key={cache_key}")
    built = builder()
    primary_path_base = cache_dirs[0] / f"{cache_name}_{cache_key}"
    _write_cache_df(primary_path_base, built)
    return built


def _load_or_build_grid(
    cache_dir: Path,
    cache_key_payload: object,
    polygon_wkt: str,
    grid_size_m: float,
) -> tuple[pd.DataFrame, CRS, Tuple[float, float]]:
    cache_key = _make_cache_key(cache_key_payload)
    data_base = cache_dir / f"grid_cells_{cache_key}"
    meta_path = cache_dir / f"grid_cells_{cache_key}.json"
    cached_df = _read_cache_df(data_base)
    if cached_df is not None and meta_path.exists():
        meta = _read_json_file(meta_path)
        print(f"[COVERAGE_TEST][CACHE_HIT] name=grid_cells key={cache_key} rows={len(cached_df)}")
        return (
            cached_df,
            CRS.from_user_input(meta["utm_crs"]),
            (float(meta["origin_xy"][0]), float(meta["origin_xy"][1])),
        )
    print(f"[COVERAGE_TEST][CACHE_MISS] name=grid_cells key={cache_key}")
    grid_cells_df, utm_crs, origin_xy = _build_polygon_grid(polygon_wkt, grid_size_m)
    _write_cache_df(data_base, grid_cells_df)
    _write_json_file(
        meta_path,
        {
            "utm_crs": utm_crs.to_string(),
            "origin_xy": [float(origin_xy[0]), float(origin_xy[1])],
        },
    )
    return grid_cells_df, utm_crs, origin_xy


def _grid_cells_to_frontend_grid_df(grid_cells_df: pd.DataFrame) -> pd.DataFrame:
    """Expose the local polygon grid in the same shape as GridAnalyticsController."""
    if grid_cells_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for row in grid_cells_df.itertuples(index=False):
        geometry_wkt = getattr(row, "geometry_wkt", None)
        if not geometry_wkt:
            continue
        try:
            geometry = load_wkt(str(geometry_wkt))
        except Exception:
            continue
        if geometry.is_empty:
            continue
        min_lon, min_lat, max_lon, max_lat = geometry.bounds
        rows.append(
            {
                "grid_id": str(getattr(row, "grid_id")),
                "center_lat": float(getattr(row, "centroid_lat")),
                "center_lon": float(getattr(row, "centroid_lon")),
                "min_lat": float(min_lat),
                "max_lat": float(max_lat),
                "min_lon": float(min_lon),
                "max_lon": float(max_lon),
                "grid_size_meters": float(getattr(row, "grid_size_m")),
                "scenario_id": None,
            }
        )
    return pd.DataFrame(rows)


def _time_step(timings: Dict[str, float], step_name: str, builder):
    started = time.perf_counter()
    result = builder()
    elapsed = round(float(time.perf_counter() - started), 4)
    timings[step_name] = elapsed
    print(f"[COVERAGE_TEST][TIMING] step={step_name} elapsed_sec={elapsed}")
    return result


def _geo_bundle_cache_key_payload(
    grid_cells_df: pd.DataFrame,
    polygon_wkt: str,
    buckets: Sequence[Tuple[str, str, str]],
    grid_size_m: float,
) -> Dict[str, object]:
    return {
        "stage": "bucket_grid_geo_features_bundle",
        "polygon_wkt": str(polygon_wkt),
        "grid_size_m": float(grid_size_m),
        "grid_cell_count": int(len(grid_cells_df)),
        "geo_snapshot_timestamps": DEFAULT_GEO_SNAPSHOT_TS_BY_BUCKET,
        "enable_green_layer": bool(ENABLE_COVERAGE_GREEN_LAYER),
        "clutter_formula_version": COVERAGE_CLUTTER_FORMULA_VERSION,
        "buckets": [[str(label), str(start_ts), str(end_ts)] for label, start_ts, end_ts in buckets],
        "layers": [
            "roads",
            "green",
            "water",
            "buildings",
            "mall",
            "metro",
            "park",
            "open",
        ],
    }


def _precomputed_geo_bundle_path(cache_dir: Path, cache_key_payload: object) -> Path:
    cache_key = _make_cache_key(cache_key_payload)
    return cache_dir / f"bucket_grid_geo_features_bundle_{cache_key}"


def _load_precomputed_geo_bundle(cache_dir: Path, cache_key_payload: object) -> Optional[pd.DataFrame]:
    return _read_cache_df(_precomputed_geo_bundle_path(cache_dir, cache_key_payload))


def _build_and_save_precomputed_geo_bundle(
    cache_dir: Path,
    grid_cells_df: pd.DataFrame,
    polygon_wkt: str,
    buckets: Sequence[Tuple[str, str, str]],
    grid_size_m: float,
) -> pd.DataFrame:
    payload = _geo_bundle_cache_key_payload(grid_cells_df, polygon_wkt, buckets, grid_size_m)
    path_base = _precomputed_geo_bundle_path(cache_dir, payload)
    existing = _read_cache_df(path_base)
    if existing is not None:
        print(f"[COVERAGE_TEST][GEO_PREBUILD] cache_hit=True rows={len(existing)}")
        return existing
    print("[COVERAGE_TEST][GEO_PREBUILD] cache_hit=False building_yearly_geo_bundle=True")
    geo_snapshot_cache_dir = _ensure_dir(cache_dir / "geo_snapshot_cache")
    latest_df, bucket_df = _build_bucket_grid_geo_features(
        grid_cells_df=grid_cells_df,
        polygon_wkt=polygon_wkt,
        cache_root=geo_snapshot_cache_dir,
        buckets=buckets,
    )
    bundle = (
        bucket_df.assign(__is_latest=0)
        if latest_df.empty else pd.concat(
            [
                bucket_df.assign(__is_latest=0),
                latest_df.assign(
                    time_bucket="__LATEST__",
                    bucket_start=pd.NA,
                    bucket_end=pd.NA,
                    geo_snapshot_source_ts=pd.NA,
                    __is_latest=1,
                ),
            ],
            ignore_index=True,
        )
    )
    _write_cache_df(path_base, bundle)
    return bundle


def _find_latest_geo_bundle(cache_dir: Path, exclude_key_payload: Optional[object] = None) -> Optional[pd.DataFrame]:
    exclude_name = None
    if exclude_key_payload is not None:
        exclude_name = _precomputed_geo_bundle_path(cache_dir, exclude_key_payload).with_suffix(".parquet").name
    candidates = sorted(cache_dir.glob("bucket_grid_geo_features_bundle_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if exclude_name and candidate.name == exclude_name:
            continue
        try:
            df = pd.read_parquet(candidate)
            print(f"[COVERAGE_TEST][GEO_REUSE] base_bundle={candidate.name} rows={len(df)}")
            return df
        except Exception:
            continue
    return None


def _load_geo_bundle_from_run_dir(run_dir: Path) -> Optional[pd.DataFrame]:
    bucket_path = run_dir / "bucket_grid_geo_features.csv"
    latest_path = run_dir / "grid_geo_features.csv"
    if not bucket_path.exists():
        return None
    try:
        bucket_df = pd.read_csv(bucket_path)
    except Exception:
        return None
    if bucket_df.empty:
        return None
    bucket_df = bucket_df.copy()
    bucket_df["__is_latest"] = 0

    latest_df = pd.DataFrame()
    if latest_path.exists():
        try:
            latest_df = pd.read_csv(latest_path)
        except Exception:
            latest_df = pd.DataFrame()
    if not latest_df.empty:
        latest_df = latest_df.copy()
        latest_df["time_bucket"] = "__LATEST__"
        latest_df["bucket_start"] = pd.NA
        latest_df["bucket_end"] = pd.NA
        latest_df["geo_snapshot_source_ts"] = pd.NA
        latest_df["__is_latest"] = 1
        for col in bucket_df.columns:
            if col not in latest_df.columns:
                latest_df[col] = pd.NA
        for col in latest_df.columns:
            if col not in bucket_df.columns:
                bucket_df[col] = pd.NA
        latest_df = latest_df[bucket_df.columns]
        bundle = pd.concat([bucket_df, latest_df], ignore_index=True)
    else:
        bundle = bucket_df
    print(f"[COVERAGE_TEST][GEO_REUSE] base_run_dir={run_dir} rows={len(bundle)}")
    return bundle


def _build_green_ratio_for_snapshot(
    grid_cells_df: pd.DataFrame,
    polygon_wkt: str,
    cache_dir: Path,
    snapshot_ts_utc: str,
) -> pd.DataFrame:
    if grid_cells_df.empty:
        return pd.DataFrame(columns=["grid_id", "green_ratio"])

    grid_gdf = gpd.GeoDataFrame(
        grid_cells_df.drop(columns=["geometry_wkt"]),
        geometry=gpd.GeoSeries.from_wkt(grid_cells_df["geometry_wkt"]),
        crs="EPSG:4326",
    )
    if "cell_area_m2" not in grid_gdf.columns and "grid_area_m2" in grid_gdf.columns:
        grid_gdf["cell_area_m2"] = pd.to_numeric(grid_gdf["grid_area_m2"], errors="coerce")
    polygon_gdf = gpd.GeoDataFrame(geometry=[load_wkt(polygon_wkt)], crs="EPSG:4326")
    green_gdf, green_mode = _fetch_osm_layer_snapshot_with_fallback("coverage_green", polygon_gdf, REDUCED_COVERAGE_GREEN_TAGS, cache_dir, snapshot_ts_utc)
    grid_gdf["green_ratio"] = 0.0
    grid_gdf = _attach_polygon_area_ratio(grid_gdf, green_gdf, "green_ratio")
    out = pd.DataFrame(grid_gdf.drop(columns="geometry"))[["grid_id", "green_ratio"]].copy()
    out["coverage_green_mode"] = green_mode
    return out


def _rebuild_geo_bundle_from_base_with_buildings(
    base_bundle_df: pd.DataFrame,
    grid_cells_df: pd.DataFrame,
    polygon_wkt: str,
    cache_dir: Path,
    buckets: Sequence[Tuple[str, str, str]],
) -> pd.DataFrame:
    if base_bundle_df.empty:
        return base_bundle_df

    work = base_bundle_df.copy()
    geo_snapshot_cache_dir = _ensure_dir(cache_dir / "geo_snapshot_cache")
    for label, _, end_ts in buckets:
        snapshot_source_ts = DEFAULT_GEO_SNAPSHOT_TS_BY_BUCKET.get(str(label), end_ts)
        snapshot_ts_utc = _snapshot_utc_string(snapshot_source_ts)
        bucket_cache_dir = _ensure_dir(geo_snapshot_cache_dir / str(label).lower())
        building_df = _build_building_features_for_snapshot(
            grid_cells_df=grid_cells_df,
            polygon_wkt=polygon_wkt,
            cache_dir=bucket_cache_dir,
            snapshot_ts_utc=snapshot_ts_utc,
        )
        if building_df.empty:
            continue
        building_map = building_df.set_index("grid_id")
        bucket_mask = work["time_bucket"].astype(str) == str(label)
        for col in ["building_count", "building_area_sum_m2", "avg_building_area_m2", "building_area_ratio"]:
            work.loc[bucket_mask, col] = (
                work.loc[bucket_mask, "grid_id"].map(building_map[col]).fillna(0.0).to_numpy()
            )
        if "geo_layer_modes_json" in work.columns:
            def _update_modes(raw_json: object, mode_value: object) -> str:
                try:
                    payload = json.loads(str(raw_json)) if pd.notna(raw_json) and str(raw_json).strip() else {}
                except Exception:
                    payload = {}
                payload["coverage_buildings"] = str(mode_value)
                return json.dumps(payload, sort_keys=True)
            mode_value = (
                str(building_df["coverage_buildings_mode"].dropna().iloc[0])
                if "coverage_buildings_mode" in building_df.columns and not building_df["coverage_buildings_mode"].dropna().empty
                else "historical_snapshot"
            )
            work.loc[bucket_mask, "geo_layer_modes_json"] = work.loc[bucket_mask, "geo_layer_modes_json"].apply(lambda raw: _update_modes(raw, mode_value))

    non_latest_mask = work.get("__is_latest", pd.Series(0, index=work.index)).fillna(0).astype(int) == 0
    if non_latest_mask.any():
        work.loc[non_latest_mask, "clutter_class"] = _derive_coverage_clutter_class(work.loc[non_latest_mask].copy()).to_numpy()

    latest_mask = work.get("__is_latest", pd.Series(0, index=work.index)).fillna(0).astype(int) == 1
    if latest_mask.any():
        part3_rows = work.loc[(work.get("__is_latest", 0) == 0) & (work["time_bucket"].astype(str) == "PART_3")].copy()
        if not part3_rows.empty:
            latest_cols = work.loc[latest_mask].columns
            refreshed = part3_rows.copy()
            refreshed["time_bucket"] = "__LATEST__"
            refreshed["bucket_start"] = pd.NA
            refreshed["bucket_end"] = pd.NA
            refreshed["geo_snapshot_source_ts"] = pd.NA
            refreshed["__is_latest"] = 1
            work = pd.concat([work.loc[~latest_mask], refreshed[latest_cols]], ignore_index=True)
    return work


def _expand_grid_geo_features_by_bucket(grid_geo_df: pd.DataFrame, buckets: Sequence[Tuple[str, str, str]]) -> pd.DataFrame:
    if grid_geo_df.empty:
        return pd.DataFrame()
    rows = []
    for label, start_ts, end_ts in buckets:
        bucket_df = grid_geo_df.copy()
        bucket_df["time_bucket"] = str(label)
        bucket_df["bucket_start"] = str(start_ts)
        bucket_df["bucket_end"] = str(end_ts)
        rows.append(bucket_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _mode_or_na(series: pd.Series):
    cleaned = series.dropna()
    if cleaned.empty:
        return pd.NA
    mode = cleaned.mode(dropna=True)
    if mode.empty:
        return cleaned.iloc[0]
    return mode.iloc[0]


def _estimate_bandwidth_mhz(series: pd.Series) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    allowed = {1.4, 3.0, 5.0, 10.0, 15.0, 20.0}
    for value in cleaned:
        v = float(value)
        if v > 1000:
            v = v / 1000.0
        if v in allowed:
            return v
    return 10.0


def _normalize_operator_label(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return "Unknown"
    lowered = text.lower()
    if "airtel" in lowered:
        return "Airtel"
    if "jio" in lowered:
        return "Jio"
    if lowered.startswith("vi") or "vodafone" in lowered or "idea" in lowered:
        return "Vi"
    return text


def _fetch_project_sites(project_id: int, region: str) -> pd.DataFrame:
    site_df, _ = base_ml.fetch_site_data(project_id, region=region)
    if site_df.empty:
        return pd.DataFrame()
    out = site_df.copy()
    if "network" in out.columns:
        out["site_operator"] = out["network"].apply(_normalize_operator_label)
    elif "operator" in out.columns:
        out["site_operator"] = out["operator"].apply(_normalize_operator_label)
    else:
        out["site_operator"] = "Unknown"
    keep_cols = [
        col for col in [
            "lat", "lon", "site_operator", "network", "operator", "Site ID",
            "nodeb_id", "cell_id", "Node_Cell_ID", "PCI", "earfcn", "band", "azimuth"
        ] if col in out.columns
    ]
    out = out[keep_cols].copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return out


def _build_project_site_summary(site_df: pd.DataFrame) -> pd.DataFrame:
    if site_df.empty or "site_operator" not in site_df.columns:
        return pd.DataFrame()
    rows = []
    for operator_name, group_df in site_df.groupby("site_operator", dropna=False):
        rows.append(
            {
                "site_operator": str(operator_name),
                "rows": int(len(group_df)),
                "unique_site_id": int(group_df["Site ID"].astype(str).nunique()) if "Site ID" in group_df.columns else 0,
                "unique_nodeb_id": int(group_df["nodeb_id"].astype(str).nunique()) if "nodeb_id" in group_df.columns else 0,
                "unique_cell_id": int(group_df["cell_id"].astype(str).nunique()) if "cell_id" in group_df.columns else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("site_operator").reset_index(drop=True)


def _fetch_project_sites_raw(project_id: int, region: str) -> pd.DataFrame:
    site_df, _ = base_ml.fetch_site_data(project_id, region=region)
    return site_df.copy()


def _default_topology_plan() -> Dict[str, Dict[str, int]]:
    return {
        "PART_1": {"site_count": 56, "cell_count": 166, "cell_sector_count": 166},
        "PART_2": {"site_count": 59, "cell_count": 176, "cell_sector_count": 176},
    }


def _normalize_site_operator_series(df: pd.DataFrame) -> pd.Series:
    if "network" in df.columns:
        return df["network"].apply(_normalize_operator_label)
    if "operator" in df.columns:
        return df["operator"].apply(_normalize_operator_label)
    if "cluster" in df.columns:
        return df["cluster"].apply(_normalize_operator_label)
    return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")


def _filter_rows_to_operator(df: pd.DataFrame, operator_name: Optional[str]) -> pd.DataFrame:
    if df.empty or not operator_name:
        return df.copy()
    normalized_target = _normalize_operator_label(operator_name)
    if "m_alpha_long" in df.columns or "m_alpha_short" in df.columns:
        long_match = (
            df.get("m_alpha_long", pd.Series([""] * len(df), index=df.index))
            .apply(_normalize_operator_label)
            .eq(normalized_target)
        )
        short_match = (
            df.get("m_alpha_short", pd.Series([""] * len(df), index=df.index))
            .apply(_normalize_operator_label)
            .eq(normalized_target)
        )
        filtered = df.loc[long_match | short_match].copy()
        print(
            f"[COVERAGE_TEST][OPERATOR_FILTER] target={normalized_target} "
            f"source=drive_rows before={len(df)} after={len(filtered)}"
        )
        return filtered

    if {"network", "operator", "cluster"}.intersection(df.columns):
        op_series = _normalize_site_operator_series(df)
        filtered = df.loc[op_series.eq(normalized_target)].copy()
        print(
            f"[COVERAGE_TEST][OPERATOR_FILTER] target={normalized_target} "
            f"source=site_rows before={len(df)} after={len(filtered)}"
        )
        return filtered
    return df.copy()


def _derive_topology_keys(site_df: pd.DataFrame) -> pd.DataFrame:
    out = site_df.copy()
    site_key_col = next((col for col in ["Site ID", "site_id", "site", "site_name", "nodeb_id"] if col in out.columns), None)
    cell_source_col = "cell_id" if "cell_id" in out.columns else None
    if site_key_col is None:
        raise ValueError("Cannot derive site topology keys: missing raw site identifier column")
    if cell_source_col is None:
        raise ValueError("Cannot derive site topology keys: missing raw cell identifier column")

    sector_series = (
        out["sector"].astype("string").str.strip()
        if "sector" in out.columns
        else pd.Series(pd.NA, index=out.index, dtype="string")
    )
    sector_series = sector_series.mask(sector_series.isin(["", "nan", "NaN", "None", "<NA>"]))
    cell_sector_fallback = (
        out[cell_source_col]
        .astype("string")
        .str.strip()
        .str.extract(r"_([^_]+)$", expand=False)
        .astype("string")
    )
    cell_sector_fallback = cell_sector_fallback.mask(cell_sector_fallback.isin(["", "nan", "NaN", "None", "<NA>"]))
    sector_series = cell_sector_fallback.where(cell_sector_fallback.notna(), sector_series)
    if "sector" in out.columns:
        out["sector"] = sector_series
    if "sec_id" in out.columns:
        out["sec_id"] = cell_sector_fallback.where(cell_sector_fallback.notna(), out["sec_id"].astype("string"))
    out["_topo_site_key"] = out[site_key_col].astype(str).str.strip().values
    out["_topo_cell_key"] = (
        out[site_key_col].astype(str).str.strip()
        + "|"
        + out[cell_source_col].astype(str).str.strip()
    ).values
    out["_topo_cell_sector_key"] = (
        out[site_key_col].astype(str).str.strip()
        + "|"
        + out[cell_source_col].astype(str).str.strip()
        + "|"
        + sector_series.fillna("__NULL__").astype(str).str.strip()
    ).values
    return out


def _select_topology_subset(
    sector_catalog: pd.DataFrame,
    prior_sector_keys: List[str],
    target_site_count: int,
    target_cell_count: int,
    target_sector_count: int,
) -> List[str]:
    selected: List[str] = []
    selected_set = set()
    selected_sites = set()
    selected_cells = set()

    sector_lookup = sector_catalog.set_index("_topo_cell_sector_key")[["_topo_site_key", "_topo_cell_key"]].to_dict("index")

    for key in prior_sector_keys:
        if key not in sector_lookup or key in selected_set:
            continue
        selected.append(key)
        selected_set.add(key)
        selected_sites.add(sector_lookup[key]["_topo_site_key"])
        selected_cells.add(sector_lookup[key]["_topo_cell_key"])

    if (
        len(selected_sites) > target_site_count
        or len(selected_cells) > target_cell_count
        or len(selected) > target_sector_count
    ):
        raise ValueError("Previous bucket topology already exceeds target counts")

    def _try_add(key: str) -> bool:
        if key in selected_set:
            return False
        meta = sector_lookup.get(key)
        if meta is None:
            return False
        new_site = 0 if meta["_topo_site_key"] in selected_sites else 1
        new_cell = 0 if meta["_topo_cell_key"] in selected_cells else 1
        if len(selected) + 1 > target_sector_count:
            return False
        if len(selected_sites) + new_site > target_site_count:
            return False
        if len(selected_cells) + new_cell > target_cell_count:
            return False
        selected.append(key)
        selected_set.add(key)
        selected_sites.add(meta["_topo_site_key"])
        selected_cells.add(meta["_topo_cell_key"])
        return True

    ordered_site_keys = sector_catalog["_topo_site_key"].drop_duplicates().tolist()
    for site_key in ordered_site_keys:
        if len(selected_sites) >= target_site_count:
            break
        selected_sites.add(site_key)

    candidate_catalog = sector_catalog.loc[sector_catalog["_topo_site_key"].isin(selected_sites)].copy()

    per_site_first = candidate_catalog.drop_duplicates(subset=["_topo_site_key"], keep="first")
    for row in per_site_first.to_dict("records"):
        _try_add(str(row["_topo_cell_sector_key"]))

    per_cell_first = candidate_catalog.drop_duplicates(subset=["_topo_cell_key"], keep="first")
    for row in per_cell_first.to_dict("records"):
        if len(selected_cells) >= target_cell_count:
            break
        _try_add(str(row["_topo_cell_sector_key"]))

    preference_masks = [
        (candidate_catalog["_topo_cell_key"].isin(selected_cells)),
        pd.Series([True] * len(candidate_catalog), index=candidate_catalog.index),
    ]
    for mask in preference_masks:
        for row in candidate_catalog.loc[mask].to_dict("records"):
            if len(selected) >= target_sector_count:
                break
            _try_add(str(row["_topo_cell_sector_key"]))
        if len(selected) >= target_sector_count:
            break

    if len(selected_sites) != target_site_count or len(selected_cells) != target_cell_count or len(selected) != target_sector_count:
        raise ValueError(
            f"Unable to build requested topology subset exactly: "
            f"sites={len(selected_sites)}/{target_site_count} "
            f"cells={len(selected_cells)}/{target_cell_count} "
            f"cell_sector={len(selected)}/{target_sector_count}"
        )
    return selected


def _select_topology_subset_relaxed(
    sector_catalog: pd.DataFrame,
    prior_sector_keys: List[str],
    target_site_count: int,
    target_cell_count: int,
    target_sector_count: int,
) -> tuple[List[str], int]:
    last_error: Optional[Exception] = None
    for relaxed_cell_count in range(int(target_cell_count), 0, -1):
        try:
            selected = _select_topology_subset(
                sector_catalog=sector_catalog,
                prior_sector_keys=prior_sector_keys,
                target_site_count=target_site_count,
                target_cell_count=relaxed_cell_count,
                target_sector_count=target_sector_count,
            )
            if relaxed_cell_count != int(target_cell_count):
                print(
                    f"[COVERAGE_TEST][TOPOLOGY_RELAX] "
                    f"sites={target_site_count} sectors={target_sector_count} "
                    f"cell_target={target_cell_count} relaxed_cell_target={relaxed_cell_count}"
                )
            return selected, int(relaxed_cell_count)
        except ValueError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError("Unable to build requested topology subset")


def _assign_band_mix_to_bucket(bucket_df: pd.DataFrame, bucket_label: str) -> tuple[pd.DataFrame, Dict[str, int]]:
    out = bucket_df.copy()
    band_plan = BAND_MIX_PLAN.get(str(bucket_label), BAND_MIX_PLAN.get("PART_3", {1800: 1.0}))
    if out.empty or "_topo_cell_sector_key" not in out.columns:
        return out, {}

    unique_keys = (
        out["_topo_cell_sector_key"]
        .astype(str)
        .drop_duplicates()
        .sort_values(
            key=lambda s: s.map(lambda value: hashlib.sha1(f"{bucket_label}|{value}".encode("utf-8")).hexdigest())
        )
        .tolist()
    )
    total_keys = len(unique_keys)
    if total_keys <= 0:
        return out, {}

    bands = list(band_plan.keys())
    raw_counts = [float(band_plan[band]) * float(total_keys) for band in bands]
    base_counts = [int(np.floor(value)) for value in raw_counts]
    remainder = total_keys - int(sum(base_counts))
    fractional_order = sorted(
        range(len(bands)),
        key=lambda idx: (raw_counts[idx] - base_counts[idx], -int(bands[idx])),
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        base_counts[idx] += 1

    assignments: Dict[str, int] = {}
    cursor = 0
    for band_value, band_count in zip(bands, base_counts):
        for key in unique_keys[cursor:cursor + int(band_count)]:
            assignments[str(key)] = int(band_value)
        cursor += int(band_count)

    if cursor < total_keys:
        fallback_band = int(max(band_plan.items(), key=lambda item: item[1])[0])
        for key in unique_keys[cursor:]:
            assignments[str(key)] = fallback_band

    assigned_band = out["_topo_cell_sector_key"].astype(str).map(assignments).astype(int)
    out["band"] = assigned_band.astype(float)
    out["frequency_mhz"] = assigned_band.astype(float)
    if "frequency" in out.columns:
        out["frequency"] = assigned_band.astype(float)
    if "downlink_frequency" in out.columns:
        out["downlink_frequency"] = assigned_band.astype(float)
    if "uplink_center_frequency" in out.columns:
        out["uplink_center_frequency"] = assigned_band.astype(float)

    band_counts = {str(int(band)): int(count) for band, count in zip(bands, base_counts)}
    return out, band_counts


def _secondary_band_for_sector(bucket_label: str, sector_key: object, primary_band: object, assigned_band: int) -> int:
    try:
        primary = int(float(primary_band))
    except (TypeError, ValueError):
        primary = None
    if assigned_band != primary:
        return int(assigned_band)
    plan = MULTI_BAND_SECONDARY_BAND_PLAN.get(str(bucket_label), MULTI_BAND_SECONDARY_BAND_PLAN["PART_3"])
    candidates = [int(band) for band in plan.keys() if int(band) != primary]
    if not candidates:
        candidates = [int(band) for band in SYNTHETIC_BAND_TO_EARFCN.keys() if int(band) != primary]
    if not candidates:
        return int(assigned_band)
    digest = hashlib.sha1(f"{bucket_label}|{sector_key}|{primary}|fallback".encode("utf-8")).hexdigest()
    return int(candidates[int(digest, 16) % len(candidates)])


def _assign_secondary_band_mix(
    sector_keys: List[str],
    primary_band_by_sector: Dict[str, object],
    bucket_label: str,
) -> Dict[str, int]:
    plan = MULTI_BAND_SECONDARY_BAND_PLAN.get(str(bucket_label), MULTI_BAND_SECONDARY_BAND_PLAN["PART_3"])
    if not sector_keys:
        return {}
    ordered_keys = sorted(
        sector_keys,
        key=lambda value: hashlib.sha1(f"{bucket_label}|secondary_band|{value}".encode("utf-8")).hexdigest(),
    )
    bands = list(plan.keys())
    raw_counts = [float(plan[band]) * float(len(ordered_keys)) for band in bands]
    base_counts = [int(np.floor(value)) for value in raw_counts]
    remainder = len(ordered_keys) - int(sum(base_counts))
    fractional_order = sorted(
        range(len(bands)),
        key=lambda idx: (raw_counts[idx] - base_counts[idx], -int(bands[idx])),
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        base_counts[idx] += 1

    assigned: Dict[str, int] = {}
    cursor = 0
    for band_value, band_count in zip(bands, base_counts):
        for key in ordered_keys[cursor:cursor + int(band_count)]:
            assigned[str(key)] = int(band_value)
        cursor += int(band_count)
    if cursor < len(ordered_keys):
        fallback_band = int(max(plan.items(), key=lambda item: item[1])[0])
        for key in ordered_keys[cursor:]:
            assigned[str(key)] = fallback_band

    normalized: Dict[str, int] = {}
    for key in ordered_keys:
        normalized[str(key)] = _secondary_band_for_sector(
            str(bucket_label),
            key,
            primary_band_by_sector.get(str(key)),
            assigned.get(str(key), int(max(plan.items(), key=lambda item: item[1])[0])),
        )
    return normalized


def _tertiary_band_for_sector(
    bucket_label: str,
    sector_key: object,
    primary_band: object,
    secondary_band: object,
) -> int:
    try:
        primary = int(float(primary_band))
    except (TypeError, ValueError):
        primary = None
    try:
        secondary = int(float(secondary_band))
    except (TypeError, ValueError):
        secondary = None

    plan = MULTI_BAND_SECONDARY_BAND_PLAN.get(str(bucket_label), MULTI_BAND_SECONDARY_BAND_PLAN["PART_3"])
    candidates = [int(band) for band in plan.keys() if int(band) not in {primary, secondary}]
    if not candidates:
        candidates = [int(band) for band in SYNTHETIC_BAND_TO_EARFCN.keys() if int(band) not in {primary, secondary}]
    if not candidates:
        return int(secondary if secondary is not None else (primary if primary is not None else 1800))
    digest = hashlib.sha1(f"{bucket_label}|{sector_key}|{primary}|{secondary}|tertiary".encode("utf-8")).hexdigest()
    return int(candidates[int(digest, 16) % len(candidates)])


def _expand_multi_band_sectors(bucket_df: pd.DataFrame, bucket_label: str) -> tuple[pd.DataFrame, Dict[str, object]]:
    out = bucket_df.copy()
    if out.empty or "_topo_cell_sector_key" not in out.columns:
        return out, {
            "multi_band_sector_share_target": 0.0,
            "multi_band_sector_count": 0,
            "triple_band_sector_share_target": 0.0,
            "triple_band_sector_count": 0,
            "multi_band_carrier_rows_added": 0,
            "triple_band_carrier_rows_added": 0,
            "multi_band_secondary_band_counts_json": json.dumps({}, sort_keys=True),
            "multi_band_tertiary_band_counts_json": json.dumps({}, sort_keys=True),
        }

    sector_share_target = float(MULTI_BAND_SECTOR_SHARE_PLAN.get(str(bucket_label), 0.0))
    triple_share_target = float(TRIPLE_BAND_SECTOR_SHARE_PLAN.get(str(bucket_label), 0.0))
    sector_keys = (
        out["_topo_cell_sector_key"]
        .astype(str)
        .drop_duplicates()
        .sort_values(
            key=lambda s: s.map(lambda value: hashlib.sha1(f"{bucket_label}|multi_band|{value}".encode("utf-8")).hexdigest())
        )
        .tolist()
    )
    sector_target_count = min(len(sector_keys), int(round(len(sector_keys) * sector_share_target)))
    triple_target_count = min(sector_target_count, int(round(len(sector_keys) * triple_share_target)))
    selected_sector_keys = sector_keys[:sector_target_count]
    triple_sector_keys = selected_sector_keys[:triple_target_count]
    dual_sector_keys = selected_sector_keys[triple_target_count:]
    if not selected_sector_keys:
        out["carrier_variant"] = 1
        return out, {
            "multi_band_sector_share_target": sector_share_target,
            "multi_band_sector_count": 0,
            "triple_band_sector_share_target": triple_share_target,
            "triple_band_sector_count": 0,
            "multi_band_carrier_rows_added": 0,
            "triple_band_carrier_rows_added": 0,
            "multi_band_secondary_band_counts_json": json.dumps({}, sort_keys=True),
            "multi_band_tertiary_band_counts_json": json.dumps({}, sort_keys=True),
        }

    selected_mask = out["_topo_cell_sector_key"].astype(str).isin(selected_sector_keys)
    selected_rows = out.loc[selected_mask].copy()
    if selected_rows.empty:
        out["carrier_variant"] = 1
        return out, {
            "multi_band_sector_share_target": sector_share_target,
            "multi_band_sector_count": 0,
            "triple_band_sector_share_target": triple_share_target,
            "triple_band_sector_count": 0,
            "multi_band_carrier_rows_added": 0,
            "triple_band_carrier_rows_added": 0,
            "multi_band_secondary_band_counts_json": json.dumps({}, sort_keys=True),
            "multi_band_tertiary_band_counts_json": json.dumps({}, sort_keys=True),
        }

    primary_rows = out.copy()
    primary_rows["carrier_variant"] = 1
    duplicate_rows = selected_rows.loc[selected_rows["_topo_cell_sector_key"].astype(str).isin(dual_sector_keys)].copy()
    triple_rows = selected_rows.loc[selected_rows["_topo_cell_sector_key"].astype(str).isin(triple_sector_keys)].copy()
    duplicate_rows["carrier_variant"] = 2
    triple_rows["carrier_variant"] = 3
    for col in ["sector", "sec_id"]:
        if col in duplicate_rows.columns:
            duplicate_rows[col] = duplicate_rows[col].astype("string")
        if col in triple_rows.columns:
            triple_rows[col] = triple_rows[col].astype("string")

    secondary_band_counts: Dict[str, int] = {}
    tertiary_band_counts: Dict[str, int] = {}
    primary_band_by_sector = (
        selected_rows.groupby("_topo_cell_sector_key", dropna=False)["band"].first().to_dict()
        if "band" in selected_rows.columns
        else {}
    )
    secondary_band_by_sector = _assign_secondary_band_mix(
        [str(key) for key in selected_sector_keys],
        {str(key): value for key, value in primary_band_by_sector.items()},
        str(bucket_label),
    )
    tertiary_band_by_sector = {
        str(key): _tertiary_band_for_sector(
            str(bucket_label),
            key,
            primary_band_by_sector.get(str(key)),
            secondary_band_by_sector.get(str(key)),
        )
        for key in triple_sector_keys
    }
    node_col = "Node_Cell_ID" if "Node_Cell_ID" in duplicate_rows.columns else ("cell_id" if "cell_id" in duplicate_rows.columns else None)
    for row_idx, row in duplicate_rows.iterrows():
        sector_key = row["_topo_cell_sector_key"]
        secondary_band = int(secondary_band_by_sector.get(str(sector_key), row.get("band", 1800)))
        secondary_band_counts[str(secondary_band)] = secondary_band_counts.get(str(secondary_band), 0) + 1
        if "sector" in row.index and pd.notna(row["sector"]) and str(row["sector"]).strip() not in {"", "nan", "None", "__NULL__"}:
            sector_value = str(row["sector"]).strip()
        else:
            base_sector_source = str(row["cell_id"]).strip() if "cell_id" in row.index else str(sector_key)
            sector_value = base_sector_source.rsplit("_", 1)[-1] if "_" in base_sector_source else str(sector_key).split("|")[-1]
        base_cell = ""
        if node_col is not None:
            base_cell = str(row[node_col]).strip()
        if not base_cell and "cell_id" in row.index:
            base_cell = str(row["cell_id"]).strip()
        if not base_cell:
            base_cell = f"{row_idx}"
        synthetic_cell_id = f"{base_cell}__MB{secondary_band}"

        if "Node_Cell_ID" in duplicate_rows.columns:
            duplicate_rows.at[row_idx, "Node_Cell_ID"] = synthetic_cell_id
        if "cell_id" in duplicate_rows.columns:
            duplicate_rows.at[row_idx, "cell_id"] = synthetic_cell_id
        if "sector" in duplicate_rows.columns:
            duplicate_rows.at[row_idx, "sector"] = sector_value
        if "sec_id" in duplicate_rows.columns:
            duplicate_rows.at[row_idx, "sec_id"] = sector_value
        if "band" in duplicate_rows.columns:
            duplicate_rows.at[row_idx, "band"] = float(secondary_band)
        synthetic_earfcn = float(SYNTHETIC_BAND_TO_EARFCN.get(int(secondary_band), int(secondary_band)))
        for col in ["frequency_mhz", "frequency", "downlink_frequency", "uplink_center_frequency"]:
            if col in duplicate_rows.columns:
                duplicate_rows.at[row_idx, col] = float(secondary_band)
        if "earfcn" in duplicate_rows.columns:
            duplicate_rows.at[row_idx, "earfcn"] = synthetic_earfcn
        if "cell_id_representative" in duplicate_rows.columns:
            duplicate_rows.at[row_idx, "cell_id_representative"] = synthetic_cell_id
        if "_topo_site_key" in duplicate_rows.columns and "_topo_cell_key" in duplicate_rows.columns:
            site_value = row["_topo_site_key"]
            duplicate_rows.at[row_idx, "_topo_cell_key"] = f"{site_value}|{synthetic_cell_id}"
            duplicate_rows.at[row_idx, "_topo_cell_sector_key"] = f"{site_value}|{synthetic_cell_id}|{sector_value}"
            if "rf_identity_key" in duplicate_rows.columns:
                duplicate_rows.at[row_idx, "rf_identity_key"] = build_rf_identity(
                    site_value,
                    synthetic_cell_id,
                    sector_value,
                    secondary_band,
                    fallback=synthetic_cell_id,
                )
            if "sector_identity_key" in duplicate_rows.columns:
                duplicate_rows.at[row_idx, "sector_identity_key"] = build_sector_identity(
                    site_value,
                    synthetic_cell_id,
                    sector_value,
                    fallback=synthetic_cell_id,
                )
        if "site_sector_band_key" in duplicate_rows.columns:
            duplicate_rows.at[row_idx, "site_sector_band_key"] = build_site_sector_band_identity(
                site_value,
                sector_value,
                secondary_band,
            )

    node_col_triple = "Node_Cell_ID" if "Node_Cell_ID" in triple_rows.columns else ("cell_id" if "cell_id" in triple_rows.columns else None)
    for row_idx, row in triple_rows.iterrows():
        sector_key = row["_topo_cell_sector_key"]
        primary_band = row.get("band", 1800)
        secondary_band = secondary_band_by_sector.get(str(sector_key))
        tertiary_band = int(tertiary_band_by_sector.get(str(sector_key), row.get("band", 1800)))
        tertiary_band_counts[str(tertiary_band)] = tertiary_band_counts.get(str(tertiary_band), 0) + 1
        if "sector" in row.index and pd.notna(row["sector"]) and str(row["sector"]).strip() not in {"", "nan", "None", "__NULL__"}:
            sector_value = str(row["sector"]).strip()
        else:
            base_sector_source = str(row["cell_id"]).strip() if "cell_id" in row.index else str(sector_key)
            sector_value = base_sector_source.rsplit("_", 1)[-1] if "_" in base_sector_source else str(sector_key).split("|")[-1]
        base_cell = ""
        if node_col_triple is not None:
            base_cell = str(row[node_col_triple]).strip()
        if not base_cell and "cell_id" in row.index:
            base_cell = str(row["cell_id"]).strip()
        if not base_cell:
            base_cell = f"{row_idx}"
        synthetic_cell_id = f"{base_cell}__MB{tertiary_band}"

        if "Node_Cell_ID" in triple_rows.columns:
            triple_rows.at[row_idx, "Node_Cell_ID"] = synthetic_cell_id
        if "cell_id" in triple_rows.columns:
            triple_rows.at[row_idx, "cell_id"] = synthetic_cell_id
        if "sector" in triple_rows.columns:
            triple_rows.at[row_idx, "sector"] = sector_value
        if "sec_id" in triple_rows.columns:
            triple_rows.at[row_idx, "sec_id"] = sector_value
        if "band" in triple_rows.columns:
            triple_rows.at[row_idx, "band"] = float(tertiary_band)
        synthetic_earfcn = float(SYNTHETIC_BAND_TO_EARFCN.get(int(tertiary_band), int(tertiary_band)))
        for col in ["frequency_mhz", "frequency", "downlink_frequency", "uplink_center_frequency"]:
            if col in triple_rows.columns:
                triple_rows.at[row_idx, col] = float(tertiary_band)
        if "earfcn" in triple_rows.columns:
            triple_rows.at[row_idx, "earfcn"] = synthetic_earfcn
        if "cell_id_representative" in triple_rows.columns:
            triple_rows.at[row_idx, "cell_id_representative"] = synthetic_cell_id
        if "_topo_site_key" in triple_rows.columns and "_topo_cell_key" in triple_rows.columns:
            site_value = row["_topo_site_key"]
            triple_rows.at[row_idx, "_topo_cell_key"] = f"{site_value}|{synthetic_cell_id}"
            triple_rows.at[row_idx, "_topo_cell_sector_key"] = f"{site_value}|{synthetic_cell_id}|{sector_value}"
            if "rf_identity_key" in triple_rows.columns:
                triple_rows.at[row_idx, "rf_identity_key"] = build_rf_identity(
                    site_value,
                    synthetic_cell_id,
                    sector_value,
                    tertiary_band,
                    fallback=synthetic_cell_id,
                )
            if "sector_identity_key" in triple_rows.columns:
                triple_rows.at[row_idx, "sector_identity_key"] = build_sector_identity(
                    site_value,
                    synthetic_cell_id,
                    sector_value,
                    fallback=synthetic_cell_id,
                )
            if "site_sector_band_key" in triple_rows.columns:
                triple_rows.at[row_idx, "site_sector_band_key"] = build_site_sector_band_identity(
                    site_value,
                    sector_value,
                    tertiary_band,
                )

    out = pd.concat([primary_rows, duplicate_rows, triple_rows], ignore_index=True)
    sort_cols = [col for col in ["_topo_site_key", "_topo_cell_key", "_topo_cell_sector_key", "carrier_variant", "band", "Node_Cell_ID"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    summary = {
        "multi_band_sector_share_target": sector_share_target,
        "multi_band_sector_count": int(len(selected_sector_keys)),
        "triple_band_sector_share_target": triple_share_target,
        "triple_band_sector_count": int(len(triple_sector_keys)),
        "multi_band_carrier_rows_added": int(len(duplicate_rows) + len(triple_rows)),
        "triple_band_carrier_rows_added": int(len(triple_rows)),
        "multi_band_secondary_band_counts_json": json.dumps(secondary_band_counts, sort_keys=True),
        "multi_band_tertiary_band_counts_json": json.dumps(tertiary_band_counts, sort_keys=True),
    }
    return out, summary


def _build_bucket_site_topologies(
    site_df: pd.DataFrame,
    buckets: Sequence[Tuple[str, str, str]],
    operator_name: Optional[str],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    if site_df.empty:
        return {}, pd.DataFrame()

    filtered = _filter_rows_to_operator(site_df, operator_name)
    if filtered.empty:
        raise ValueError(f"No site rows found for operator={operator_name}")
    filtered = _derive_topology_keys(filtered)
    filtered = filtered.sort_values(["_topo_site_key", "_topo_cell_key", "_topo_cell_sector_key", "lat", "lon"]).reset_index(drop=True)

    sector_catalog = (
        filtered[["_topo_site_key", "_topo_cell_key", "_topo_cell_sector_key"]]
        .drop_duplicates(subset=["_topo_cell_sector_key"], keep="first")
        .sort_values(["_topo_site_key", "_topo_cell_key", "_topo_cell_sector_key"])
        .reset_index(drop=True)
    )

    full_site_count = int(filtered["_topo_site_key"].nunique())
    full_cell_count = int(filtered["_topo_cell_key"].nunique())
    full_sector_count = int(filtered["_topo_cell_sector_key"].nunique())
    plan = _default_topology_plan()

    bucket_frames: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, object]] = []
    previous_sector_keys: List[str] = []
    for idx, (label, _, _) in enumerate(buckets):
        bucket_plan = dict(plan.get(str(label), {}))
        site_target = int(bucket_plan.get("site_count", full_site_count if idx == len(buckets) - 1 else full_site_count))
        cell_target = int(bucket_plan.get("cell_count", full_cell_count if idx == len(buckets) - 1 else full_cell_count))
        sector_target = int(bucket_plan.get("cell_sector_count", full_sector_count if idx == len(buckets) - 1 else full_sector_count))
        if idx == len(buckets) - 1:
            site_target = full_site_count
            cell_target = full_cell_count
            sector_target = full_sector_count

        selected_sector_keys, effective_cell_target = _select_topology_subset_relaxed(
            sector_catalog=sector_catalog,
            prior_sector_keys=previous_sector_keys,
            target_site_count=site_target,
            target_cell_count=cell_target,
            target_sector_count=sector_target,
        )
        bucket_df = filtered.loc[filtered["_topo_cell_sector_key"].isin(selected_sector_keys)].copy()
        bucket_df, band_counts = _assign_band_mix_to_bucket(bucket_df, str(label))
        bucket_df, multi_band_summary = _expand_multi_band_sectors(bucket_df, str(label))
        actual_site_count = int(bucket_df["_topo_site_key"].nunique())
        actual_cell_count = int(bucket_df["_topo_cell_key"].nunique())
        actual_sector_count = int(bucket_df["_topo_cell_sector_key"].nunique())
        bucket_df = bucket_df.drop(columns=["_topo_site_key", "_topo_cell_key", "_topo_cell_sector_key"], errors="ignore")
        bucket_frames[str(label)] = bucket_df.reset_index(drop=True)
        previous_sector_keys = selected_sector_keys
        summary_rows.append(
            {
                "time_bucket": str(label),
                "operator": _normalize_operator_label(operator_name),
                "site_count": actual_site_count,
                "cell_count": actual_cell_count,
                "cell_sector_count": actual_sector_count,
                "requested_cell_count": int(cell_target),
                "effective_cell_count_target": int(effective_cell_target),
                "band_mix_version": BAND_DIVERSITY_VERSION,
                "band_counts_json": json.dumps(band_counts, sort_keys=True),
                "multi_band_sector_share_target": float(multi_band_summary["multi_band_sector_share_target"]),
                "multi_band_sector_count": int(multi_band_summary["multi_band_sector_count"]),
                "multi_band_carrier_rows_added": int(multi_band_summary["multi_band_carrier_rows_added"]),
                "multi_band_secondary_band_counts_json": str(multi_band_summary["multi_band_secondary_band_counts_json"]),
                "site_rows": int(len(bucket_df)),
            }
        )

    return bucket_frames, pd.DataFrame(summary_rows)


def _attach_carrier_load_share(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()
    out = pred_df.copy()
    if "Node_Cell_ID" in out.columns:
        carrier_id = out["Node_Cell_ID"].astype(str).str.strip()
    elif "cell_id" in out.columns:
        carrier_id = out["cell_id"].astype(str).str.strip()
    else:
        out["carrier_load_share"] = 1.0
        return out

    out["_carrier_base_key"] = carrier_id.str.replace(r"__MB\d+$", "", regex=True)
    out["_carrier_band_value"] = pd.to_numeric(out.get("band"), errors="coerce").fillna(1800.0)
    out["_carrier_band_weight"] = out["_carrier_band_value"].round().astype(int).map(CARRIER_LOAD_SHARE_BAND_WEIGHT).fillna(0.85)
    rsrp_col = next((col for col in ["pred_rsrp_calibrated", "pred_rsrp_geo", "pred_rsrp"] if col in out.columns), None)
    if rsrp_col is not None:
        rsrp = pd.to_numeric(out[rsrp_col], errors="coerce")
        group_keys = [col for col in ["time_bucket", "grid_id", "_carrier_base_key"] if col in out.columns]
        rsrp_rel = rsrp - rsrp.groupby([out[col] for col in group_keys], dropna=False).transform("max")
        strength_weight = np.exp((rsrp_rel.fillna(-12.0)).clip(-18.0, 0.0) / 6.0)
    else:
        strength_weight = pd.Series(1.0, index=out.index, dtype="float64")

    raw_share = (out["_carrier_band_weight"] * strength_weight).clip(lower=1e-6)
    group_keys = [col for col in ["time_bucket", "grid_id", "_carrier_base_key"] if col in out.columns]
    if not group_keys:
        out["carrier_load_share"] = 1.0
        return out.drop(columns=["_carrier_base_key", "_carrier_band_value", "_carrier_band_weight"], errors="ignore")

    denom = raw_share.groupby([out[col] for col in group_keys], dropna=False).transform("sum").replace(0.0, np.nan)
    out["carrier_load_share"] = (raw_share / denom).fillna(1.0).clip(0.0, 1.0)
    return out.drop(columns=["_carrier_base_key", "_carrier_band_value", "_carrier_band_weight"], errors="ignore")


def _run_project_baseline_prediction(
    project_id: int,
    region: str,
    site_df: pd.DataFrame,
    drive_df: pd.DataFrame,
    building_df: pd.DataFrame,
    baseline_radius_m: float,
    grid_size_m: float,
    workers: int,
    max_interference_sites: int,
    polygon_wkt: Optional[str] = None,
    use_frontend_grid_sampling: bool = True,
    grid_analytics_scenario_id: Optional[int] = None,
    frontend_grid_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if site_df.empty:
        return pd.DataFrame()
    params = {
        "project_id": int(project_id),
        "region": str(region).lower(),
        "polygon_wkt": str(polygon_wkt or "").strip(),
        "radius": float(baseline_radius_m),
        "grid": float(grid_size_m),
        "workers": int(workers),
        "max_interference_sites": int(max_interference_sites),
        "max_cells_per_grid": 3,
        "min_grids_per_cell": 3,
        "candidate_safety_cap": 20,
        "use_frontend_grid_sampling": bool(use_frontend_grid_sampling),
        "grid_analytics_scenario_id": (
            int(grid_analytics_scenario_id) if grid_analytics_scenario_id is not None else None
        ),
    }
    if use_frontend_grid_sampling and frontend_grid_df is not None and not frontend_grid_df.empty:
        original_get_bridge_client = base_ml.get_bridge_client
        original_fetch_frontend_grid_cells = base_ml.fetch_frontend_grid_cells

        def _fetch_local_frontend_grid(*args, **kwargs):
            print(
                f"[COVERAGE_TEST][FRONTEND_GRID_LOCAL] rows={len(frontend_grid_df)} "
                f"grid_size_m={grid_size_m} source=coverage_grid_cells"
            )
            return frontend_grid_df.copy(), None

        try:
            base_ml.get_bridge_client = lambda: None
            base_ml.fetch_frontend_grid_cells = _fetch_local_frontend_grid
            pred_df = base_ml.run_rf_prediction_fast(
                site_df=site_df,
                drive_df=drive_df if not drive_df.empty else pd.DataFrame(columns=["lat", "lon"]),
                building_df=building_df,
                params=params,
            )
        finally:
            base_ml.get_bridge_client = original_get_bridge_client
            base_ml.fetch_frontend_grid_cells = original_fetch_frontend_grid_cells
    else:
        pred_df = base_ml.run_rf_prediction_fast(
            site_df=site_df,
            drive_df=drive_df if not drive_df.empty else pd.DataFrame(columns=["lat", "lon"]),
            building_df=building_df,
            params=params,
        )
    if pred_df.empty:
        return pd.DataFrame()
    out = pred_df.copy()
    for col in ["lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr", "pci", "earfcn", "band"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return _attach_carrier_load_share(out)


def _run_bucket_baseline_predictions(
    detail_df: pd.DataFrame,
    site_df_by_bucket: Dict[str, pd.DataFrame],
    building_df: pd.DataFrame,
    project_id: int,
    region: str,
    baseline_radius_m: float,
    grid_size_m: float,
    workers: int,
    max_interference_sites: int,
    buckets: Sequence[Tuple[str, str, str]],
    polygon_wkt: Optional[str] = None,
    use_frontend_grid_sampling: bool = True,
    grid_analytics_scenario_id: Optional[int] = None,
    frontend_grid_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    out_rows: List[pd.DataFrame] = []
    for label, _, _ in buckets:
        bucket_site_df = site_df_by_bucket.get(str(label), pd.DataFrame())
        bucket_dt_df = (
            detail_df.loc[detail_df["time_bucket"].astype(str) == str(label)].copy()
            if (not detail_df.empty and "time_bucket" in detail_df.columns)
            else pd.DataFrame()
        )
        baseline_df = _run_project_baseline_prediction(
            project_id=project_id,
            region=region,
            site_df=bucket_site_df,
            drive_df=bucket_dt_df,
            building_df=building_df,
            baseline_radius_m=baseline_radius_m,
            grid_size_m=grid_size_m,
            workers=workers,
            max_interference_sites=max_interference_sites,
            polygon_wkt=polygon_wkt,
            use_frontend_grid_sampling=use_frontend_grid_sampling,
            grid_analytics_scenario_id=grid_analytics_scenario_id,
            frontend_grid_df=frontend_grid_df,
        )
        if baseline_df.empty:
            continue
        baseline_df = baseline_df.copy()
        baseline_df["time_bucket"] = str(label)
        baseline_df = _attach_carrier_load_share(baseline_df)
        out_rows.append(baseline_df)
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def _run_bucket_corrected_predictions(
    baseline_pred_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    site_df_by_bucket: Dict[str, pd.DataFrame],
    building_df: pd.DataFrame,
    project_id: int,
    region: str,
    grid_size_m: float,
    buckets: Sequence[Tuple[str, str, str]],
    polygon_wkt: Optional[str] = None,
) -> pd.DataFrame:
    if baseline_pred_df.empty:
        return pd.DataFrame()
    out_rows: List[pd.DataFrame] = []
    for label, _, _ in buckets:
        bucket_site_df = site_df_by_bucket.get(str(label), pd.DataFrame())
        bucket_baseline_df = (
            baseline_pred_df.loc[baseline_pred_df["time_bucket"].astype(str) == str(label)].copy()
            if "time_bucket" in baseline_pred_df.columns
            else baseline_pred_df.copy()
        )
        if bucket_baseline_df.empty:
            continue
        bucket_dt_df = detail_df.loc[detail_df["time_bucket"].astype(str) == str(label)].copy() if (not detail_df.empty and "time_bucket" in detail_df.columns) else pd.DataFrame()
        corrected_df = base_ml.run_ml_fast(
            pred_df=bucket_baseline_df,
            drive_df=bucket_dt_df,
            site_df=bucket_site_df,
            building_df=building_df,
            params={
                "project_id": int(project_id),
                "region": str(region).lower(),
                "polygon_wkt": str(polygon_wkt or "").strip(),
                "grid": float(grid_size_m),
                "tile_size_m": 100.0,
                "cluster_count": 5,
                "dt_replace_radius_m": 20.0,
                "dt_blend_sigma_m": 60.0,
                "dt_blend_radius_m": 140.0,
            },
        )
        corrected_df = corrected_df.copy()
        corrected_df["time_bucket"] = str(label)
        corrected_df = _attach_carrier_load_share(corrected_df)
        out_rows.append(corrected_df)
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def _build_corrected_grid_surface(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty or "grid_id" not in pred_df.columns or "time_bucket" not in pred_df.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    work = pred_df.loc[pred_df["grid_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    # Match production baseline persistence: keep calibrated pre-smoothing KPI
    # values as the main baseline surface, while preserving display/demo values
    # separately for audit.
    def _resolve_metric_series(group_df: pd.DataFrame, metric: str) -> pd.Series:
        candidates = [
            f"pred_{metric}_calibrated",
            f"pred_{metric}_geo",
            f"pred_{metric}",
        ]
        for col in candidates:
            if col in group_df.columns:
                return pd.to_numeric(group_df[col], errors="coerce")
        return pd.Series(np.nan, index=group_df.index, dtype="float64")

    def _resolve_display_series(group_df: pd.DataFrame, metric: str) -> pd.Series:
        candidates = [
            f"pred_{metric}_demo",
            f"pred_{metric}",
        ]
        for col in candidates:
            if col in group_df.columns:
                return pd.to_numeric(group_df[col], errors="coerce")
        return pd.Series(np.nan, index=group_df.index, dtype="float64")

    group_cols = ["time_bucket", "grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon"]
    for keys, group_df in work.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rsrp_series = _resolve_metric_series(group_df, "rsrp")
        rsrq_series = _resolve_metric_series(group_df, "rsrq")
        sinr_series = _resolve_metric_series(group_df, "sinr")
        rsrp_display_series = _resolve_display_series(group_df, "rsrp")
        rsrq_display_series = _resolve_display_series(group_df, "rsrq")
        sinr_display_series = _resolve_display_series(group_df, "sinr")
        row.update(
            {
                "prediction_point_count": int(len(group_df)),
                "corrected_rsrp_mean": float(rsrp_series.dropna().mean()) if rsrp_series.dropna().size else np.nan,
                "corrected_rsrq_mean": float(rsrq_series.dropna().mean()) if rsrq_series.dropna().size else np.nan,
                "corrected_sinr_mean": float(sinr_series.dropna().mean()) if sinr_series.dropna().size else np.nan,
                "display_rsrp_mean": float(rsrp_display_series.dropna().mean()) if rsrp_display_series.dropna().size else np.nan,
                "display_rsrq_mean": float(rsrq_display_series.dropna().mean()) if rsrq_display_series.dropna().size else np.nan,
                "display_sinr_mean": float(sinr_display_series.dropna().mean()) if sinr_display_series.dropna().size else np.nan,
                "corrected_dominant_pci": _mode_or_na(pd.to_numeric(group_df["pci"], errors="coerce")) if "pci" in group_df.columns else pd.NA,
                "corrected_dominant_earfcn": _mode_or_na(pd.to_numeric(group_df["earfcn"], errors="coerce")) if "earfcn" in group_df.columns else pd.NA,
                "corrected_bandwidth_mhz_est": _estimate_bandwidth_mhz(group_df["band"]) if "band" in group_df.columns else 10.0,
                "correction_source": _mode_or_na(group_df["demo_visual_source"]) if "demo_visual_source" in group_df.columns else pd.NA,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _approx_cqi_from_sinr(series: pd.Series) -> pd.Series:
    sinr = pd.to_numeric(series, errors="coerce")
    cqi = np.floor((sinr + 6.7) / 1.8)
    cqi = cqi.clip(lower=1, upper=15)
    return cqi


def _build_grid_kpi_timeseries(
    detail_df: pd.DataFrame,
    corrected_grid_df: pd.DataFrame,
    grid_cells_df: pd.DataFrame,
    buckets: Sequence[Tuple[str, str, str]],
) -> pd.DataFrame:
    dt_work = detail_df.loc[detail_df["grid_id"].notna()].copy() if (not detail_df.empty and "grid_id" in detail_df.columns) else pd.DataFrame()
    if not dt_work.empty:
        raw_cqi = pd.to_numeric(dt_work.get("cqi"), errors="coerce")
        valid_observed_cqi = raw_cqi.where((raw_cqi >= 1) & (raw_cqi <= 15))
        dt_work["cqi_filled"] = valid_observed_cqi
        dt_work["cqi_source"] = np.where(valid_observed_cqi.notna(), "observed_valid", "estimated_from_sinr")
        missing_cqi_mask = dt_work["cqi_filled"].isna()
        if "sinr" in dt_work.columns:
            dt_work.loc[missing_cqi_mask, "cqi_filled"] = _approx_cqi_from_sinr(dt_work.loc[missing_cqi_mask, "sinr"])
        dt_work.loc[dt_work["cqi_filled"].isna(), "cqi_source"] = "missing"
        dt_work["bandwidth_mhz_est"] = 10.0
        if "bw" in dt_work.columns:
            for _, bucket_idx in dt_work.groupby(["time_bucket", "grid_id"]).groups.items():
                dt_work.loc[list(bucket_idx), "bandwidth_mhz_est"] = _estimate_bandwidth_mhz(dt_work.loc[list(bucket_idx), "bw"])
        elif "band" in dt_work.columns:
            for _, bucket_idx in dt_work.groupby(["time_bucket", "grid_id"]).groups.items():
                dt_work.loc[list(bucket_idx), "bandwidth_mhz_est"] = _estimate_bandwidth_mhz(dt_work.loc[list(bucket_idx), "band"])
        dt_work["estimated_prb"] = np.nan
        valid_prb_mask = (
            pd.to_numeric(dt_work.get("dl_tpt"), errors="coerce").notna()
            & (pd.to_numeric(dt_work.get("dl_tpt"), errors="coerce") > 0)
            & pd.to_numeric(dt_work["cqi_filled"], errors="coerce").notna()
        )
        dt_work.loc[valid_prb_mask, "estimated_prb"] = (
            pd.to_numeric(dt_work.loc[valid_prb_mask, "cqi_filled"], errors="coerce")
            * pd.to_numeric(dt_work.loc[valid_prb_mask, "bandwidth_mhz_est"], errors="coerce")
        ) / pd.to_numeric(dt_work.loc[valid_prb_mask, "dl_tpt"], errors="coerce")

    dt_rows: List[Dict[str, object]] = []
    if not dt_work.empty:
        group_cols = ["time_bucket", "grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon"]
        for keys, group_df in dt_work.groupby(group_cols, dropna=False):
            row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            row.update(
                {
                    "sample_count": int(len(group_df)),
                    "dt_rsrp_mean": float(pd.to_numeric(group_df["rsrp"], errors="coerce").dropna().mean()) if "rsrp" in group_df.columns and pd.to_numeric(group_df["rsrp"], errors="coerce").dropna().size else np.nan,
                    "dt_rsrq_mean": float(pd.to_numeric(group_df["rsrq"], errors="coerce").dropna().mean()) if "rsrq" in group_df.columns and pd.to_numeric(group_df["rsrq"], errors="coerce").dropna().size else np.nan,
                    "dt_sinr_mean": float(pd.to_numeric(group_df["sinr"], errors="coerce").dropna().mean()) if "sinr" in group_df.columns and pd.to_numeric(group_df["sinr"], errors="coerce").dropna().size else np.nan,
                    "rssi_mean": float(pd.to_numeric(group_df["rssi"], errors="coerce").dropna().mean()) if "rssi" in group_df.columns and pd.to_numeric(group_df["rssi"], errors="coerce").dropna().size else np.nan,
                    "dl_tpt_mean": float(pd.to_numeric(group_df["dl_tpt"], errors="coerce").dropna().mean()) if "dl_tpt" in group_df.columns and pd.to_numeric(group_df["dl_tpt"], errors="coerce").dropna().size else np.nan,
                    "ul_tpt_mean": float(pd.to_numeric(group_df["ul_tpt"], errors="coerce").dropna().mean()) if "ul_tpt" in group_df.columns and pd.to_numeric(group_df["ul_tpt"], errors="coerce").dropna().size else np.nan,
                    "dt_cqi_mean": float(pd.to_numeric(group_df["cqi_filled"], errors="coerce").dropna().mean()) if pd.to_numeric(group_df["cqi_filled"], errors="coerce").dropna().size else np.nan,
                    "dt_estimated_prb_mean": float(pd.to_numeric(group_df["estimated_prb"], errors="coerce").dropna().mean()) if pd.to_numeric(group_df["estimated_prb"], errors="coerce").dropna().size else np.nan,
                    "dt_dominant_pci": _mode_or_na(pd.to_numeric(group_df["pci"], errors="coerce")) if "pci" in group_df.columns else pd.NA,
                    "dt_dominant_earfcn": _mode_or_na(pd.to_numeric(group_df["earfcn"], errors="coerce")) if "earfcn" in group_df.columns else pd.NA,
                    "dt_bandwidth_mhz_est": float(pd.to_numeric(group_df["bandwidth_mhz_est"], errors="coerce").dropna().median()) if pd.to_numeric(group_df["bandwidth_mhz_est"], errors="coerce").dropna().size else np.nan,
                    "dt_cqi_source": _mode_or_na(group_df["cqi_source"]),
                }
            )
            dt_rows.append(row)
    dt_grid_df = pd.DataFrame(dt_rows)

    if grid_cells_df.empty:
        return dt_grid_df
    base_grid = grid_cells_df[["grid_id", "grid_row", "grid_col", "centroid_lat", "centroid_lon"]].copy()
    base_grid = base_grid.rename(columns={"centroid_lat": "grid_centroid_lat", "centroid_lon": "grid_centroid_lon"})
    all_rows: List[pd.DataFrame] = []
    for label, _, _ in buckets:
        bucket_base = base_grid.copy()
        bucket_base["time_bucket"] = str(label)
        if not corrected_grid_df.empty:
            bucket_pred_df = corrected_grid_df.loc[corrected_grid_df["time_bucket"].astype(str) == str(label)].copy()
            bucket_base = bucket_base.merge(bucket_pred_df, on=["time_bucket", "grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon"], how="left")
        if not dt_grid_df.empty:
            bucket_dt_df = dt_grid_df.loc[dt_grid_df["time_bucket"].astype(str) == str(label)].copy()
            bucket_base = bucket_base.merge(bucket_dt_df, on=["time_bucket", "grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon"], how="left")

        for col in [
            "sample_count", "dt_rsrp_mean", "dt_rsrq_mean", "dt_sinr_mean", "rssi_mean", "dl_tpt_mean", "ul_tpt_mean",
            "dt_cqi_mean", "dt_estimated_prb_mean", "dt_bandwidth_mhz_est",
            "corrected_rsrp_mean", "corrected_rsrq_mean", "corrected_sinr_mean",
            "display_rsrp_mean", "display_rsrq_mean", "display_sinr_mean",
            "corrected_bandwidth_mhz_est",
            "prediction_point_count",
        ]:
            if col not in bucket_base.columns:
                bucket_base[col] = np.nan
        for col in ["dt_dominant_pci", "dt_dominant_earfcn", "corrected_dominant_pci", "corrected_dominant_earfcn", "correction_source"]:
            if col not in bucket_base.columns:
                bucket_base[col] = pd.Series([pd.NA] * len(bucket_base), index=bucket_base.index, dtype="object")

        bucket_base["sample_count"] = pd.to_numeric(bucket_base.get("sample_count"), errors="coerce").fillna(0).astype(int)
        bucket_base["rsrp_mean"] = pd.to_numeric(bucket_base.get("corrected_rsrp_mean"), errors="coerce")
        bucket_base["rsrq_mean"] = pd.to_numeric(bucket_base.get("corrected_rsrq_mean"), errors="coerce")
        bucket_base["sinr_mean"] = pd.to_numeric(bucket_base.get("corrected_sinr_mean"), errors="coerce")
        bucket_base["rssi_mean"] = pd.to_numeric(bucket_base.get("rssi_mean"), errors="coerce")
        bucket_base["dl_tpt_mean"] = pd.to_numeric(bucket_base.get("dl_tpt_mean"), errors="coerce")
        bucket_base["ul_tpt_mean"] = pd.to_numeric(bucket_base.get("ul_tpt_mean"), errors="coerce")
        dt_cqi_series = pd.to_numeric(bucket_base.get("dt_cqi_mean"), errors="coerce")
        est_cqi_series = _approx_cqi_from_sinr(bucket_base["sinr_mean"])
        bucket_base["cqi_mean"] = dt_cqi_series.combine_first(est_cqi_series)
        bucket_base["bandwidth_mhz_est"] = pd.to_numeric(bucket_base.get("dt_bandwidth_mhz_est"), errors="coerce").combine_first(pd.to_numeric(bucket_base.get("corrected_bandwidth_mhz_est"), errors="coerce")).fillna(10.0)
        bucket_base["estimated_prb_mean"] = pd.to_numeric(bucket_base.get("dt_estimated_prb_mean"), errors="coerce")
        bucket_base["dominant_pci"] = bucket_base["dt_dominant_pci"].combine_first(bucket_base["corrected_dominant_pci"])
        bucket_base["dominant_earfcn"] = bucket_base["dt_dominant_earfcn"].combine_first(bucket_base["corrected_dominant_earfcn"])
        bucket_base["cqi_source"] = np.where(
            dt_cqi_series.notna(),
            "dt_observed_or_estimated",
            np.where(bucket_base["sinr_mean"].notna(), "corrected_prediction_from_sinr", "missing"),
        )
        bucket_base["kpi_source"] = bucket_base["correction_source"].fillna("prediction_only")
        all_rows.append(bucket_base)
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    keep_cols = [
        "time_bucket", "grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon",
        "sample_count", "rsrp_mean", "rsrq_mean", "sinr_mean", "rssi_mean", "dl_tpt_mean", "ul_tpt_mean",
        "cqi_mean", "estimated_prb_mean", "dominant_pci", "dominant_earfcn", "bandwidth_mhz_est",
        "cqi_source", "kpi_source", "corrected_rsrp_mean", "corrected_rsrq_mean", "corrected_sinr_mean",
        "display_rsrp_mean", "display_rsrq_mean", "display_sinr_mean",
        "prediction_point_count",
    ]
    return out[[col for col in keep_cols if col in out.columns]].copy()


def _build_kpi_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metric_cols = [
        "rsrp",
        "rsrq",
        "sinr",
        "rssi",
        "dl_tpt",
        "ul_tpt",
        "cqi",
        "latency",
        "jitter",
        "packet_loss",
        "mos",
        "speed",
        "csi_rsrp",
        "csi_rsrq",
        "csi_sinr",
    ]
    for bucket_name, bucket_df in detail_df.groupby("time_bucket", dropna=True):
        for metric in metric_cols:
            if metric not in bucket_df.columns:
                continue
            series = pd.to_numeric(bucket_df[metric], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "time_bucket": bucket_name,
                    "metric": metric,
                    "sample_count": int(series.shape[0]),
                    "min": float(series.min()),
                    "p25": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "p75": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                }
            )
    return pd.DataFrame(rows)


def run_coverage_test(config: CoverageTestConfig) -> Path:
    start = time.perf_counter()
    site_project_id = int(config.site_project_id if config.site_project_id is not None else config.project_id)
    run_dir = _ensure_dir(config.output_root / f"project_{config.project_id}" / f"coverage_{_timestamp()}")
    stable_cache_dir = _cache_root_for_project(config.project_id)
    base_run_dir = Path(str(config.geo_base_run_dir)) if config.geo_base_run_dir else None
    engine = _resolve_engine(config.region)
    config_fp = _config_fingerprint(config)
    timings: Dict[str, float] = {}
    reused_summary_df = _load_bucket_summary_from_run_dir(base_run_dir)
    reused_detail_df = _load_coverage_rows_from_run_dir(base_run_dir)
    reused_project_sites_df = _load_project_sites_from_run_dir(base_run_dir)
    print(
        f"[COVERAGE_TEST][START] project_id={config.project_id} region={config.region} "
        f"site_project_id={site_project_id} "
        f"bucket_count={len(config.buckets)} chunk_size={config.chunk_size} "
        f"geo_cache_mode={config.geo_cache_mode}"
    )

    summary_df = _time_step(
        timings,
        "bucket_summary",
        lambda: _load_or_build_df(
            stable_cache_dir,
            "bucket_summary",
            {"stage": "bucket_summary", **config_fp},
            lambda: reused_summary_df.copy() if reused_summary_df is not None else _fetch_bucket_summary(engine, config.polygon_wkt, config.buckets),
        ),
    )
    raw_detail_df = _time_step(
        timings,
        "coverage_rows_raw",
        lambda: _load_or_build_df(
            stable_cache_dir,
            "coverage_rows_raw",
            {"stage": "coverage_rows_raw", **config_fp},
            lambda: reused_detail_df.copy()
            if reused_detail_df is not None
            else _coerce_numeric_columns(
                _fetch_detail_rows_chunked(engine, config.polygon_wkt, config.buckets, config.chunk_size)
            ),
        ),
    )
    grid_cells_df, utm_crs, origin_xy = _time_step(
        timings,
        "grid_cells",
        lambda: _load_or_build_grid(
            stable_cache_dir,
            {"stage": "grid_cells", "polygon_wkt": config_fp["polygon_wkt"], "grid_size_m": config_fp["grid_size_m"]},
            config.polygon_wkt,
            config.grid_size_m,
        ),
    )
    local_frontend_grid_df = _time_step(
        timings,
        "local_frontend_grid",
        lambda: _grid_cells_to_frontend_grid_df(grid_cells_df),
    )
    detail_df = _time_step(
        timings,
        "coverage_rows_mapped",
        lambda: _load_or_build_df(
            stable_cache_dir,
            "coverage_rows_mapped",
            {
                "stage": "coverage_rows_mapped",
                **config_fp,
                "grid_key": {"polygon_wkt": config_fp["polygon_wkt"], "grid_size_m": config_fp["grid_size_m"]},
            },
            lambda: raw_detail_df.copy()
            if ("grid_id" in raw_detail_df.columns and raw_detail_df["grid_id"].notna().any())
            else _assign_points_to_grid(
                detail_df=raw_detail_df,
                grid_size_m=config.grid_size_m,
                utm_crs=utm_crs,
                origin_xy=origin_xy,
                grid_cells_df=grid_cells_df,
            ),
        ),
    )
    detail_df = _filter_rows_to_operator(detail_df, config.topology_operator)
    project_sites_raw_df = _time_step(
        timings,
        "project_sites_raw",
        lambda: reused_project_sites_df.copy() if reused_project_sites_df is not None else _fetch_project_sites_raw(site_project_id, config.region),
    )
    bucket_site_df_map, bucket_topology_summary_df = _time_step(
        timings,
        "bucket_site_topology",
        lambda: _build_bucket_site_topologies(
            site_df=project_sites_raw_df,
            buckets=config.buckets,
            operator_name=config.topology_operator,
        ),
    )
    bucket_topology_key = {
        "topology_selection_version": TOPOLOGY_SELECTION_VERSION,
        "band_diversity_version": BAND_DIVERSITY_VERSION,
        "band_mix_plan": BAND_MIX_PLAN,
        "bucket_topology_summary": bucket_topology_summary_df.to_dict(orient="records"),
    }
    building_df = _time_step(
        timings,
        "building_rows",
        lambda: _load_or_build_df_from_any_cache_dir(
            [
                stable_cache_dir,
                PROJECT_ROOT.parent / "tests" / "output" / f"project_{config.project_id}" / "coverage_cache",
            ],
            "building_rows",
            {"stage": "building_rows", "site_project_id": site_project_id, "region": config.region},
            lambda: base_ml.fetch_building_data(site_project_id, region=config.region),
        ),
    )
    baseline_pred_df = _time_step(
        timings,
        "baseline_prediction_grid_mapped",
        lambda: _load_or_build_df(
            stable_cache_dir,
            "baseline_prediction_grid_mapped",
            {
                "stage": "baseline_prediction_grid_mapped",
                "project_id": config.project_id,
                "site_project_id": site_project_id,
                "region": config.region,
                "grid_size_m": config.grid_size_m,
                "baseline_radius_m": config.baseline_radius_m,
                "baseline_workers": config.baseline_workers,
                "max_interference_sites": config.max_interference_sites,
                "polygon_wkt": config.polygon_wkt,
                "topology_operator": config.topology_operator,
                "use_frontend_grid_sampling": config.use_frontend_grid_sampling,
                "grid_analytics_scenario_id": config.grid_analytics_scenario_id,
                "buckets": [[str(label), str(start_ts), str(end_ts)] for label, start_ts, end_ts in config.buckets],
                "baseline_mode": "per_bucket_local_frontend_grid_samples_v1",
                "frontend_grid_rows": int(len(local_frontend_grid_df)),
                "samples_per_grid_axis": 3,
                "max_cells_per_grid": 3,
                "candidate_safety_cap": 20,
                "bucket_topology_key": bucket_topology_key,
            },
            lambda: _assign_points_to_grid(
                detail_df=_run_bucket_baseline_predictions(
                    detail_df=detail_df,
                    site_df_by_bucket=bucket_site_df_map,
                    building_df=building_df,
                    project_id=config.project_id,
                    region=config.region,
                    baseline_radius_m=config.baseline_radius_m,
                    grid_size_m=config.grid_size_m,
                    workers=config.baseline_workers,
                    max_interference_sites=config.max_interference_sites,
                    buckets=config.buckets,
                    polygon_wkt=config.polygon_wkt,
                    use_frontend_grid_sampling=config.use_frontend_grid_sampling,
                    grid_analytics_scenario_id=config.grid_analytics_scenario_id,
                    frontend_grid_df=local_frontend_grid_df,
                ),
                grid_size_m=config.grid_size_m,
                utm_crs=utm_crs,
                origin_xy=origin_xy,
                grid_cells_df=grid_cells_df,
            ),
        ),
    )
    corrected_pred_df = _time_step(
        timings,
        "bucket_corrected_prediction_grid_mapped",
        lambda: _load_or_build_df(
            stable_cache_dir,
            "bucket_corrected_prediction_grid_mapped",
            {
                "stage": "bucket_corrected_prediction_grid_mapped",
                **config_fp,
                "baseline_key": {
                    "grid_size_m": config.grid_size_m,
                    "baseline_radius_m": config.baseline_radius_m,
                    "baseline_workers": config.baseline_workers,
                    "max_interference_sites": config.max_interference_sites,
                },
                "bucket_topology_key": bucket_topology_key,
            },
            lambda: _assign_points_to_grid(
                detail_df=_run_bucket_corrected_predictions(
                    baseline_pred_df=baseline_pred_df,
                    detail_df=detail_df,
                    site_df_by_bucket=bucket_site_df_map,
                    building_df=building_df,
                    project_id=config.project_id,
                    region=config.region,
                    grid_size_m=config.grid_size_m,
                    buckets=config.buckets,
                    polygon_wkt=config.polygon_wkt,
                ),
                grid_size_m=config.grid_size_m,
                utm_crs=utm_crs,
                origin_xy=origin_xy,
                grid_cells_df=grid_cells_df,
            ),
        ),
    )
    corrected_grid_df = _time_step(timings, "corrected_grid_surface", lambda: _build_corrected_grid_surface(corrected_pred_df))
    kpi_summary_df = _time_step(timings, "kpi_summary", lambda: _build_kpi_summary(detail_df))
    bucket_grid_summary_df = _time_step(timings, "bucket_grid_summary", lambda: _build_bucket_grid_summary(detail_df))
    grid_kpi_timeseries_df = _time_step(
        timings,
        "grid_kpi_timeseries",
        lambda: _build_grid_kpi_timeseries(detail_df, corrected_grid_df, grid_cells_df, config.buckets),
    )
    project_sites_df = _time_step(
        timings,
        "project_sites",
        lambda: _filter_rows_to_operator(reused_project_sites_df.copy(), config.topology_operator)
        if reused_project_sites_df is not None
        else _filter_rows_to_operator(_fetch_project_sites(site_project_id, config.region), config.topology_operator),
    )
    project_site_summary_df = _time_step(timings, "project_site_summary", lambda: _build_project_site_summary(project_sites_df))

    geo_bundle_key = _geo_bundle_cache_key_payload(grid_cells_df, config.polygon_wkt, config.buckets, config.grid_size_m)
    def _geo_bundle_builder():
        if str(config.geo_cache_mode).lower() == "prebuilt":
            cached = _load_precomputed_geo_bundle(stable_cache_dir, geo_bundle_key)
            if cached is None and config.geo_base_run_dir:
                cached = _load_geo_bundle_from_run_dir(Path(str(config.geo_base_run_dir)))
                if cached is not None:
                    print(f"[COVERAGE_TEST][GEO_PREBUILT] reused_from_base_run=True rows={len(cached)}")
                    return cached
            if cached is None:
                raise FileNotFoundError(
                    "Prebuilt yearly geo bundle not found. Run once with --geo-cache-mode build to create it."
                )
            print(f"[COVERAGE_TEST][GEO_PREBUILT] reused=True rows={len(cached)}")
            return cached
        if str(config.geo_cache_mode).lower() == "build":
            if config.geo_base_run_dir:
                base_bundle = _load_geo_bundle_from_run_dir(Path(str(config.geo_base_run_dir)))
                if base_bundle is not None:
                    print(f"[COVERAGE_TEST][GEO_REUSE] base_run_dir={config.geo_base_run_dir} rows={len(base_bundle)}")
                    return base_bundle
            base_bundle = None
            current = _load_precomputed_geo_bundle(stable_cache_dir, geo_bundle_key)
            if current is not None:
                print(f"[COVERAGE_TEST][GEO_PREBUILD] cache_hit=True rows={len(current)}")
                return current
            if base_bundle is None:
                base_bundle = _find_latest_geo_bundle(stable_cache_dir, exclude_key_payload=geo_bundle_key)
            if base_bundle is not None:
                print("[COVERAGE_TEST][GEO_PREBUILD] cache_hit=False incremental_update_from_existing_bundle=True")
                updated = _rebuild_geo_bundle_from_base_with_buildings(
                    base_bundle_df=base_bundle,
                    grid_cells_df=grid_cells_df,
                    polygon_wkt=config.polygon_wkt,
                    cache_dir=stable_cache_dir,
                    buckets=config.buckets,
                )
                _write_cache_df(_precomputed_geo_bundle_path(stable_cache_dir, geo_bundle_key), updated)
                return updated
            return _build_and_save_precomputed_geo_bundle(
                cache_dir=stable_cache_dir,
                grid_cells_df=grid_cells_df,
                polygon_wkt=config.polygon_wkt,
                buckets=config.buckets,
                grid_size_m=config.grid_size_m,
            )
        cached = _load_precomputed_geo_bundle(stable_cache_dir, geo_bundle_key)
        if cached is not None:
            print(f"[COVERAGE_TEST][GEO_PREBUILT] reused=True rows={len(cached)}")
            return cached
        return _build_and_save_precomputed_geo_bundle(
            cache_dir=stable_cache_dir,
            grid_cells_df=grid_cells_df,
            polygon_wkt=config.polygon_wkt,
            buckets=config.buckets,
            grid_size_m=config.grid_size_m,
        )

    geo_bundle_df = _time_step(timings, "bucket_grid_geo_features_bundle", _geo_bundle_builder)
    bucket_grid_geo_features_df = geo_bundle_df.loc[geo_bundle_df.get("__is_latest", 0) == 0].drop(columns=["__is_latest"], errors="ignore").copy()
    grid_geo_features_df = geo_bundle_df.loc[geo_bundle_df.get("__is_latest", 0) == 1].drop(columns=["__is_latest"], errors="ignore").copy()
    if not grid_geo_features_df.empty:
        grid_geo_features_df = grid_geo_features_df.drop(columns=["time_bucket", "bucket_start", "bucket_end", "geo_snapshot_source_ts"], errors="ignore").reset_index(drop=True)
    summary_df.to_csv(run_dir / "bucket_summary.csv", index=False)
    detail_df.to_csv(run_dir / "coverage_rows.csv", index=False)
    kpi_summary_df.to_csv(run_dir / "kpi_summary.csv", index=False)
    bucket_grid_summary_df.to_csv(run_dir / "bucket_grid_summary.csv", index=False)
    grid_kpi_timeseries_df.to_csv(run_dir / "grid_kpi_timeseries.csv", index=False)
    baseline_pred_df.to_csv(run_dir / "baseline_prediction_grid.csv", index=False)
    corrected_pred_df.to_csv(run_dir / "bucket_corrected_prediction_grid.csv", index=False)
    corrected_grid_df.to_csv(run_dir / "corrected_grid_surface.csv", index=False)
    project_sites_df.to_csv(run_dir / "project_sites.csv", index=False)
    project_site_summary_df.to_csv(run_dir / "project_site_summary.csv", index=False)
    bucket_topology_summary_df.to_csv(run_dir / "bucket_topology_summary.csv", index=False)
    grid_geo_features_df.to_csv(run_dir / "grid_geo_features.csv", index=False)
    bucket_grid_geo_features_df.to_csv(run_dir / "bucket_grid_geo_features.csv", index=False)
    grid_cells_df.to_csv(run_dir / "coverage_grid_cells.csv", index=False)
    if not grid_cells_df.empty:
        grid_gdf = gpd.GeoDataFrame(
            grid_cells_df.drop(columns=["geometry_wkt"]),
            geometry=gpd.GeoSeries.from_wkt(grid_cells_df["geometry_wkt"]),
            crs="EPSG:4326",
        )
        grid_gdf.to_file(run_dir / "coverage_grid_cells.geojson", driver="GeoJSON")

    summary = {
        "run_type": "coverage_test",
        "project_id": int(config.project_id),
        "site_project_id": int(site_project_id),
        "region": config.region,
        "polygon_wkt": config.polygon_wkt,
        "grid_size_m": float(config.grid_size_m),
        "grid_cell_count": int(len(grid_cells_df)),
        "bucket_ranges": [
            {"label": label, "start": start_ts, "end": end_ts}
            for label, start_ts, end_ts in config.buckets
        ],
        "chunk_size": int(config.chunk_size),
        "baseline_radius_m": float(config.baseline_radius_m),
        "baseline_workers": int(config.baseline_workers),
        "max_interference_sites": int(config.max_interference_sites),
        "bucket_row_counts": {
            str(row["time_bucket"]): int(row["total_rows"])
            for _, row in summary_df.iterrows()
        },
        "total_rows": int(len(detail_df)),
        "timings_sec": {
            **timings,
            "total_runtime_sec": round(float(time.perf_counter() - start), 4),
        },
        "grid_kpi_timeseries_rows": int(len(grid_kpi_timeseries_df)),
        "baseline_prediction_rows": int(len(baseline_pred_df)),
        "bucket_corrected_prediction_rows": int(len(corrected_pred_df)),
        "corrected_grid_surface_rows": int(len(corrected_grid_df)),
        "project_site_rows": int(len(project_sites_df)),
        "bucket_topology_summary_rows": int(len(bucket_topology_summary_df)),
        "grid_geo_features_rows": int(len(grid_geo_features_df)),
        "bucket_grid_geo_features_rows": int(len(bucket_grid_geo_features_df)),
        "topology_operator": str(config.topology_operator or ""),
        "use_frontend_grid_sampling": bool(config.use_frontend_grid_sampling),
        "grid_analytics_scenario_id": (
            int(config.grid_analytics_scenario_id) if config.grid_analytics_scenario_id is not None else None
        ),
        "geo_snapshot_strategy": "per_bucket_osm_overpass_attic_custom_snapshot_ts",
        "geo_snapshot_timestamps": DEFAULT_GEO_SNAPSHOT_TS_BY_BUCKET,
        "geo_cache_mode": str(config.geo_cache_mode),
        "artifacts": {
            "bucket_summary_csv": "bucket_summary.csv",
            "bucket_grid_summary_csv": "bucket_grid_summary.csv",
            "coverage_rows_csv": "coverage_rows.csv",
            "kpi_summary_csv": "kpi_summary.csv",
            "grid_kpi_timeseries_csv": "grid_kpi_timeseries.csv",
            "baseline_prediction_grid_csv": "baseline_prediction_grid.csv",
            "bucket_corrected_prediction_grid_csv": "bucket_corrected_prediction_grid.csv",
            "corrected_grid_surface_csv": "corrected_grid_surface.csv",
            "project_sites_csv": "project_sites.csv",
            "project_site_summary_csv": "project_site_summary.csv",
            "bucket_topology_summary_csv": "bucket_topology_summary.csv",
            "grid_geo_features_csv": "grid_geo_features.csv",
            "bucket_grid_geo_features_csv": "bucket_grid_geo_features.csv",
            "coverage_grid_cells_csv": "coverage_grid_cells.csv",
            "coverage_grid_cells_geojson": "coverage_grid_cells.geojson",
        },
    }
    _write_json(run_dir / "summary.json", summary)
    archive_path = DATA_ROOT / f"{run_dir.name}.7z"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "tar",
            "-cf",
            str(archive_path),
            "-C",
            str(run_dir.parent),
            str(run_dir.name),
        ]
    )
    write_latest_coverage_artifact(archive_path)
    print(
        f"[COVERAGE_TEST][DONE] rows={len(detail_df)} "
        f"bucket_counts={summary['bucket_row_counts']} grid_cells={len(grid_cells_df)}"
    )
    return run_dir


def _parse_args() -> CoverageTestConfig:
    parser = argparse.ArgumentParser(description="Run test-only polygon coverage analysis from tbl_network_log.")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--site-project-id", type=int, default=None)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--polygon-wkt", type=str, default=DEFAULT_POLYGON_WKT)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--grid-size-m", type=float, default=50.0)
    parser.add_argument("--baseline-radius-m", type=float, default=500.0)
    parser.add_argument("--baseline-workers", type=int, default=3)
    parser.add_argument("--max-interference-sites", type=int, default=50)
    parser.add_argument("--geo-cache-mode", type=str, default="prebuilt", choices=["prebuilt", "build", "auto"])
    parser.add_argument("--geo-base-run-dir", type=str, default=None)
    parser.add_argument("--topology-operator", type=str, default="Airtel")
    parser.add_argument("--use-frontend-grid-sampling", type=int, default=1)
    parser.add_argument("--grid-analytics-scenario-id", type=int, default=None)
    args = parser.parse_args()
    return CoverageTestConfig(
        project_id=int(args.project_id),
        site_project_id=(int(args.site_project_id) if args.site_project_id is not None else None),
        region=str(args.region),
        polygon_wkt=str(args.polygon_wkt),
        buckets=DEFAULT_BUCKETS,
        chunk_size=max(1000, int(args.chunk_size)),
        grid_size_m=max(5.0, float(args.grid_size_m)),
        baseline_radius_m=max(100.0, float(args.baseline_radius_m)),
        baseline_workers=max(1, int(args.baseline_workers)),
        max_interference_sites=max(1, int(args.max_interference_sites)),
        geo_cache_mode=str(args.geo_cache_mode).lower(),
        geo_base_run_dir=str(args.geo_base_run_dir) if args.geo_base_run_dir else None,
        topology_operator=str(args.topology_operator).strip() or None,
        use_frontend_grid_sampling=bool(int(args.use_frontend_grid_sampling)),
        grid_analytics_scenario_id=(int(args.grid_analytics_scenario_id) if args.grid_analytics_scenario_id is not None else None),
        output_root=OUTPUT_ROOT,
    )


def main() -> None:
    config = _parse_args()
    run_dir = run_coverage_test(config)
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
