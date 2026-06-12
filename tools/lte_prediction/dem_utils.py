from __future__ import annotations

import gzip
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv
from rasterio.merge import merge
from sqlalchemy import create_engine
from shapely.wkt import loads as load_wkt

from utils.python_bridge import get_bridge_client

try:
    import rasterio
except Exception:  # pragma: no cover - optional dependency
    rasterio = None

from .geo_correction_pipeline import align_project_polygon_to_points


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

SKADI_BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"


def _engine_for_region(region: str):
    env_key = "DATABASE_URL" if str(region).lower() == "india" else "DATABASE_URL_Taiwan"
    url = os.getenv(env_key)
    if not url:
        raise RuntimeError(f"Missing {env_key} in environment")
    return create_engine(url)


def _load_project_polygon(project_id: int, region: str, site_df: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
    bridge = get_bridge_client()
    if bridge:
        df = bridge.get_rows("GetProjectRegions", {"projectId": int(project_id)})
        source = "python_bridge"
    else:
        engine = _engine_for_region(region)
        query = """
        SELECT ST_AsText(region) AS region_wkt
        FROM map_regions
        WHERE tbl_project_id = %(project_id)s
          AND status = 1
        """
        df = pd.read_sql(query, engine, params={"project_id": int(project_id)})
        source = "database"
    print(f"[DEM][PROJECT_REGION_FETCH] source={source} rows={len(df)}")
    polygons = []
    for raw in df.get("region_wkt", pd.Series(dtype=str)).dropna():
        raw = str(raw).strip()
        if not raw:
            continue
        try:
            polygons.append(load_wkt(raw))
        except Exception:
            continue
    if not polygons:
        raise ValueError(f"No project polygons found for project_id={project_id}")

    polygon_gdf = gpd.GeoDataFrame({"geometry": polygons}, crs="EPSG:4326")
    if site_df is not None and not site_df.empty and {"lat", "lon"}.issubset(site_df.columns):
        polygon_gdf, alignment = align_project_polygon_to_points(polygon_gdf, site_df)
        print(f"[DEM][POLYGON_ALIGNMENT] project_id={project_id} region={region} {alignment}")
    return polygon_gdf


def _project_bbox(project_id: int, region: str, site_df: Optional[pd.DataFrame] = None) -> tuple[float, float, float, float]:
    polygon_gdf = _load_project_polygon(project_id, region, site_df=site_df)
    min_lon, min_lat, max_lon, max_lat = polygon_gdf.geometry.union_all().bounds
    return float(min_lat), float(min_lon), float(max_lat), float(max_lon)


def _tile_name(lat_deg: int, lon_deg: int) -> tuple[str, str]:
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    folder = f"{ns}{abs(lat_deg):02d}"
    filename = f"{ns}{abs(lat_deg):02d}{ew}{abs(lon_deg):03d}.hgt.gz"
    return folder, filename


def _tile_range(min_lat: float, min_lon: float, max_lat: float, max_lon: float):
    lat_start = math.floor(min_lat)
    lat_end = math.floor(max_lat)
    lon_start = math.floor(min_lon)
    lon_end = math.floor(max_lon)
    for lat_deg in range(lat_start, lat_end + 1):
        for lon_deg in range(lon_start, lon_end + 1):
            yield lat_deg, lon_deg


def _download_tile(lat_deg: int, lon_deg: int, temp_dir: Path, timeout_sec: int) -> Path:
    folder, filename = _tile_name(lat_deg, lon_deg)
    url = f"{SKADI_BASE_URL}/{folder}/{filename}"
    gz_path = temp_dir / filename
    hgt_path = temp_dir / filename[:-3]
    print(f"[DEM][DOWNLOAD] url={url}")
    resp = requests.get(url, stream=True, timeout=timeout_sec)
    resp.raise_for_status()
    with gz_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                fh.write(chunk)
    with gzip.open(gz_path, "rb") as src, hgt_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return hgt_path


def _sample_raster_value(src, lon: float, lat: float) -> float:
    x, y = lon, lat
    if src.crs is not None and str(src.crs).upper() != "EPSG:4326":
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
    value = float(next(src.sample([(x, y)]))[0])
    if src.nodata is not None and math.isclose(value, float(src.nodata), rel_tol=0.0, abs_tol=1e-9):
        return float("nan")
    return value


def _validate_dem_coverage(dem_path: Path, min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> tuple[bool, str]:
    if rasterio is None:
        return False, "rasterio_unavailable"
    if not dem_path.exists():
        return False, "dem_file_missing"

    with rasterio.open(dem_path) as src:
        if src.crs is None:
            return False, "dem_crs_missing"

        left, bottom, right, top = src.bounds
        if str(src.crs).upper() != "EPSG:4326":
            from pyproj import Transformer

            to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            xs = [left, right, right, left]
            ys = [bottom, bottom, top, top]
            lon_lat = [to_wgs84.transform(x, y) for x, y in zip(xs, ys)]
            lons = [pt[0] for pt in lon_lat]
            lats = [pt[1] for pt in lon_lat]
            left, right = min(lons), max(lons)
            bottom, top = min(lats), max(lats)

        bbox_ok = (left <= min_lon <= right) and (left <= max_lon <= right) and (bottom <= min_lat <= top) and (bottom <= max_lat <= top)
        if not bbox_ok:
            return False, (
                f"bbox_mismatch raster_lon=({left:.6f},{right:.6f}) raster_lat=({bottom:.6f},{top:.6f}) "
                f"project_lon=({min_lon:.6f},{max_lon:.6f}) project_lat=({min_lat:.6f},{max_lat:.6f})"
            )

        centroid_lon = (min_lon + max_lon) / 2.0
        centroid_lat = (min_lat + max_lat) / 2.0
        centroid_value = _sample_raster_value(src, centroid_lon, centroid_lat)
        if not math.isfinite(centroid_value):
            return False, "centroid_nodata"

    return True, "ok"


def _build_dem(project_id: int, region: str, site_df: Optional[pd.DataFrame], output_path: Path, timeout_sec: int) -> Path:
    if rasterio is None:
        raise RuntimeError("rasterio is required for DEM generation")

    min_lat, min_lon, max_lat, max_lon = _project_bbox(project_id, region, site_df=site_df)
    print(
        f"[DEM][BBOX] project_id={project_id} region={region} "
        f"min_lat={min_lat:.6f} min_lon={min_lon:.6f} max_lat={max_lat:.6f} max_lon={max_lon:.6f}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"dem_{project_id}_") as tmp:
        temp_dir = Path(tmp)
        tile_paths = [
            _download_tile(lat_deg, lon_deg, temp_dir, timeout_sec)
            for lat_deg, lon_deg in _tile_range(min_lat, min_lon, max_lat, max_lon)
        ]

        datasets = [rasterio.open(path) for path in tile_paths]
        try:
            mosaic, transform = merge(datasets)
            meta = datasets[0].meta.copy()
            meta.update(
                {
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": transform,
                }
            )
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(mosaic)
        finally:
            for ds in datasets:
                ds.close()

    is_valid, reason = _validate_dem_coverage(output_path, min_lat, min_lon, max_lat, max_lon)
    if not is_valid:
        raise RuntimeError(f"Generated DEM failed validation: {reason}")

    with rasterio.open(output_path) as src:
        left, bottom, right, top = src.bounds
        print(
            f"[DEM][OUTPUT] path={output_path} crs={src.crs} "
            f"bounds_lon=({left:.6f},{right:.6f}) bounds_lat=({bottom:.6f},{top:.6f})"
        )
    return output_path


def ensure_project_dem(
    project_id: int,
    region: str,
    site_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str | Path] = None,
    timeout_sec: int = 60,
    force: bool = False,
) -> Path:
    dem_path = Path(output_path) if output_path else PROJECT_ROOT / "data" / "dem" / f"project_{int(project_id)}_dem.tif"
    min_lat, min_lon, max_lat, max_lon = _project_bbox(project_id, region, site_df=site_df)

    if not force and dem_path.exists():
        is_valid, reason = _validate_dem_coverage(dem_path, min_lat, min_lon, max_lat, max_lon)
        print(f"[DEM][CACHE_CHECK] path={dem_path} valid={is_valid} reason={reason}")
        if is_valid:
            return dem_path
        try:
            dem_path.unlink()
        except Exception:
            pass

    return _build_dem(project_id, region, site_df=site_df, output_path=dem_path, timeout_sec=timeout_sec)
