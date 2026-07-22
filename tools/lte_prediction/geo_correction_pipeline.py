from __future__ import annotations

import ast
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely import wkb
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import transform
from shapely.wkt import loads as load_wkt
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

from tools.lte_tilt_recommandation.cell_identity import (
    build_rf_identity,
    build_sector_identity,
    build_site_sector_band_identity,
)

try:
    import rasterio
except Exception:  # pragma: no cover - optional dependency
    rasterio = None

try:
    import osmnx as ox
except Exception:  # pragma: no cover - optional dependency
    ox = None


DEFAULT_TILE_SIZE_M = 100.0
DEFAULT_CLUSTER_COUNT = 5
DEFAULT_ENABLE_OSM_ENRICHMENT = False
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZER_ROOT = PROJECT_ROOT / "tests" / "output"
DEFAULT_GEO_WEIGHTS = {
    "clutter_Dense Urban": -4.5,
    "clutter_Urban": -2.5,
    "clutter_Suburban": -1.0,
    "clutter_Vegetation": -1.8,
    "clutter_Water": 1.0,
    "clutter_Rural/Open": 0.8,
    "morphology_cluster": -0.35,
    "building_area_ratio": -9.0,
    "building_count": -0.08,
    "road_length_m": -0.003,
    "green_ratio": -2.0,
    "water_ratio": 1.2,
    "avg_building_area_m2": -0.0008,
    "site_count_250m": 0.15,
    "site_count_500m": 0.08,
    "serving_distance_m": -0.0035,
    "nearest_site_distance_m": -0.0015,
    "azimuth_delta_deg": -0.018,
    "mean_nearest3_site_distance_m": 0.0008,
    "dense_urban_far_base": -2.8,
    "dense_urban_far_slope": -0.004,
    "urban_off_axis_slope": -0.015,
    "far_serving_off_axis_base": -1.2,
    "far_serving_distance_slope": -0.004,
    "far_serving_azimuth_slope": -0.010,
    "high_building_far_base": -1.1,
    "high_building_area_slope": -0.0012,
    "high_building_distance_slope": -0.0030,
    "vegetation_far_base": -0.8,
    "vegetation_green_slope": -2.2,
    "water_open_base": 0.9,
    "water_open_distance_slope": 0.0015,
    "dense_site_base": 0.7,
    "dense_site_count_slope": 0.06,
    "cluster_dense_urban_base": -1.4,
    "cluster_dense_urban_slope": -0.35,
    "nlos_flag": -2.4,
    "los_blocker_count": -0.9,
    "los_blocked_ratio": -5.5,
    "max_blocker_height_m": -0.05,
    "diffraction_proxy_db": -0.55,
    "terrain_slope_deg": -0.08,
    "terrain_relief_to_site_m": -0.028,
    "interference_gap_penalty_slope": -0.55,
    "interference_gap_bonus_slope": 0.10,
    "interference_ratio_linear": -1.6,
    "rsrp_phys_weight": 0.28,
    "rsrp_geo_weight": 0.55,
    "rsrq_phys_weight": 0.24,
    "rsrq_geo_weight": 0.18,
    "rsrq_geo_fallback_weight": 0.22,
    "sinr_phys_weight": 0.32,
    "sinr_geo_weight": 0.24,
    "sinr_geo_fallback_weight": 0.35,
}


def _choose_utm_crs(gdf_4326: gpd.GeoDataFrame) -> str:
    centroid = gdf_4326.to_crs("EPSG:4326").geometry.union_all().centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": south}).to_string()


def _swap_geometry_xy(geom):
    return transform(lambda x, y, z=None: (y, x) if z is None else (y, x, z), geom)


def _fallback_polygon_from_points(points_df: pd.DataFrame) -> gpd.GeoDataFrame:
    points = points_df.copy()
    points["lat"] = pd.to_numeric(points["lat"], errors="coerce")
    points["lon"] = pd.to_numeric(points["lon"], errors="coerce")
    points = points.dropna(subset=["lat", "lon"]).copy()
    if points.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    min_lon = float(points["lon"].min())
    max_lon = float(points["lon"].max())
    min_lat = float(points["lat"].min())
    max_lat = float(points["lat"].max())
    lon_pad = max((max_lon - min_lon) * 0.05, 0.001)
    lat_pad = max((max_lat - min_lat) * 0.05, 0.001)
    polygon = Polygon(
        [
            (min_lon - lon_pad, min_lat - lat_pad),
            (max_lon + lon_pad, min_lat - lat_pad),
            (max_lon + lon_pad, max_lat + lat_pad),
            (min_lon - lon_pad, max_lat + lat_pad),
        ]
    )
    return gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")


def _parse_geometry_value(raw_value):
    if raw_value is None:
        return None

    if isinstance(raw_value, memoryview):
        raw_value = raw_value.tobytes()

    if isinstance(raw_value, (bytes, bytearray)):
        candidates = [bytes(raw_value)]
        text_candidate = None
    else:
        text_candidate = str(raw_value).strip()
        if text_candidate.lower() in ("", "none", "nan"):
            return None
        candidates = []

    if text_candidate:
        try:
            return load_wkt(text_candidate)
        except Exception:
            pass

        if text_candidate.startswith(("b'", 'b"', "bytearray(")):
            try:
                literal = ast.literal_eval(text_candidate)
                if isinstance(literal, memoryview):
                    literal = literal.tobytes()
                if isinstance(literal, bytearray):
                    literal = bytes(literal)
                if isinstance(literal, bytes):
                    candidates.append(literal)
            except Exception:
                pass

        hex_candidate = text_candidate.lower()
        if hex_candidate.startswith("0x"):
            hex_candidate = hex_candidate[2:]
        if hex_candidate and all(ch in "0123456789abcdef" for ch in hex_candidate):
            try:
                candidates.append(bytes.fromhex(hex_candidate))
            except Exception:
                pass

    for candidate in candidates:
        try:
            return wkb.loads(candidate)
        except Exception:
            continue
    return None


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _osm_enrichment_enabled(params: Optional[Dict[str, object]] = None) -> bool:
    params = params or {}
    if params.get("enable_osm_enrichment") is not None:
        return _truthy(params.get("enable_osm_enrichment"))
    if "LTE_PREDICTION_ENABLE_OSM" in os.environ:
        return _truthy(os.getenv("LTE_PREDICTION_ENABLE_OSM"))
    return DEFAULT_ENABLE_OSM_ENRICHMENT


def _log_geo_stage(stage: str, started_at: float, **details):
    elapsed = time.perf_counter() - started_at
    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[LTE][GEO_STAGE] stage={stage} elapsed_sec={elapsed:.2f} {suffix}".rstrip(), flush=True)


def _metric_bundle(y_true: pd.Series, y_pred: pd.Series, metric_key: Optional[str] = None) -> Dict[str, float]:
    y_true_num = pd.to_numeric(y_true, errors="coerce")
    y_pred_num = pd.to_numeric(y_pred, errors="coerce")
    valid = pd.DataFrame({"actual": y_true_num, "pred": y_pred_num}).dropna()
    if valid.empty:
        return {}
    err = valid["actual"] - valid["pred"]
    abs_err = err.abs()
    metrics = {
        "mae": round(float(mean_absolute_error(valid["actual"], valid["pred"])), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(valid["actual"], valid["pred"]))), 4),
        "r2": round(float(r2_score(valid["actual"], valid["pred"])), 4),
        "bias": round(float(err.mean()), 4),
        "p50_abs_err": round(float(abs_err.quantile(0.50)), 4),
        "p90_abs_err": round(float(abs_err.quantile(0.90)), 4),
    }
    thresholds = {
        "RSRP_meas": (3.0, 6.0, 10.0),
        "RSRQ_meas": (1.0, 2.0, 3.0),
        "SINR_meas": (2.0, 4.0, 6.0),
    }.get(metric_key or "")
    if thresholds:
        for threshold in thresholds:
            metrics[f"within_{str(threshold).replace('.', '_')}"] = round(float((abs_err <= threshold).mean()), 4)
    return metrics


def _discover_best_weights_path(project_id: int, output_root: Path = OPTIMIZER_ROOT) -> Optional[Path]:
    project_dir = output_root / f"project_{project_id}"
    if not project_dir.exists():
        return None
    candidates = list(project_dir.glob("**/best_weights.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_geo_weights(project_id: Optional[int] = None, weights_path: Optional[str | Path] = None) -> tuple[Dict[str, float], Dict[str, object]]:
    weights = dict(DEFAULT_GEO_WEIGHTS)
    summary: Dict[str, object] = {"source": "defaults", "path": None, "loaded_count": 0}

    resolved_path: Optional[Path] = None
    if weights_path:
        resolved_path = Path(weights_path)
    elif project_id is not None:
        resolved_path = _discover_best_weights_path(project_id)

    if resolved_path is None or not resolved_path.exists():
        return weights, summary

    weight_df = pd.read_csv(resolved_path)
    if not {"parameter", "value"}.issubset(weight_df.columns):
        return weights, summary

    loaded_count = 0
    for row in weight_df.itertuples(index=False):
        key = str(getattr(row, "parameter"))
        if key in weights:
            try:
                weights[key] = float(getattr(row, "value"))
                loaded_count += 1
            except Exception:
                continue

    summary["source"] = "optimizer_csv"
    summary["path"] = str(resolved_path)
    summary["loaded_count"] = loaded_count
    return weights, summary


def _derive_frequency_mhz(df: pd.DataFrame, default: float = 1800.0) -> pd.Series:
    frequency = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for col in ("frequency_mhz", "frequency", "band"):
        if col not in df.columns:
            continue
        candidate = pd.to_numeric(df[col], errors="coerce")
        frequency = frequency.where(frequency.notna(), candidate)
    return frequency.fillna(float(default)).astype(float)


def normalize_site_for_geo(site_df: pd.DataFrame) -> pd.DataFrame:
    out = site_df.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    if "Node_Cell_ID" not in out.columns:
        if "cell_id" in out.columns:
            out["Node_Cell_ID"] = out["cell_id"].astype(str).str.strip()
        else:
            raise ValueError("site_df is missing both Node_Cell_ID and cell_id")

    alias_map = {
        "Etilt": "electrical_tilt",
        "Mtilt": "mechanical_tilt",
        "Height": "antenna_height",
        "PCI": "pci",
    }
    for src_col, dst_col in alias_map.items():
        if dst_col not in out.columns and src_col in out.columns:
            out[dst_col] = out[src_col]

    numeric_defaults = {
        "lat": None,
        "lon": None,
        "azimuth": 0,
        "electrical_tilt": 3,
        "mechanical_tilt": 0,
        "antenna_height": 30,
        "tx_power": 46,
    }
    for col, default in numeric_defaults.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if default is not None:
                out[col] = out[col].fillna(default)
        elif default is not None:
            out[col] = default

    out["frequency_mhz"] = _derive_frequency_mhz(out)

    required_cols = [
        "lat",
        "lon",
        "azimuth",
        "electrical_tilt",
        "mechanical_tilt",
        "antenna_height",
        "tx_power",
        "frequency_mhz",
        "Node_Cell_ID",
    ]
    missing = [col for col in required_cols if col not in out.columns]
    if missing:
        raise ValueError(f"site_df is missing RF-required columns after normalization: {missing}")
    return out


def _clean_identity_part(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.isin(["", "nan", "NaN", "None", "<NA>"]))


def _canonical_identity_token(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _canonical_sector_token(value: object) -> Optional[str]:
    text = _canonical_identity_token(value)
    if not text:
        return None
    if "|" in text:
        text = text.split("|", 1)[1]
    if "_" in text:
        text = text.rsplit("_", 1)[-1]
    text = _canonical_identity_token(text)
    if text and text.endswith(".0"):
        text = text[:-2]
    return text or None


def _canonical_site_token(value: object) -> Optional[str]:
    text = _canonical_identity_token(value)
    if not text:
        return None
    if "|" in text:
        text = text.split("|", 1)[0]
    return _canonical_identity_token(text)


def _canonical_sector_ids(site_key: pd.Series, sector_or_cell: pd.Series, fallback_cell: pd.Series) -> pd.Series:
    site = site_key.map(_canonical_site_token)
    sector = sector_or_cell.map(_canonical_sector_token)
    fallback_sector = fallback_cell.map(_canonical_sector_token)
    sector = sector.where(sector.notna(), fallback_sector)
    canonical = site.astype("string") + "|" + sector.astype("string")
    canonical = canonical.mask(site.isna() | sector.isna())
    return canonical.astype("string")


def add_sector_identity_columns(site_df: pd.DataFrame, use_as_node_cell_id: bool = False) -> pd.DataFrame:
    out = site_df.copy()
    if out.empty:
        return out

    if "original_node_cell_id" not in out.columns:
        if "Node_Cell_ID" in out.columns:
            out["original_node_cell_id"] = _clean_identity_part(out["Node_Cell_ID"])
        elif "cell_id" in out.columns:
            out["original_node_cell_id"] = _clean_identity_part(out["cell_id"])
        else:
            out["original_node_cell_id"] = pd.Series(pd.NA, index=out.index, dtype="string")

    if "original_cell_id" not in out.columns:
        out["original_cell_id"] = _clean_identity_part(out["cell_id"]) if "cell_id" in out.columns else out["original_node_cell_id"]

    site_col = next((col for col in ["site", "Site ID", "site_id", "site_name"] if col in out.columns), None)
    if site_col:
        site_key = _clean_identity_part(out[site_col]).fillna("unknown-site")
    elif "nodeb_id" in out.columns:
        site_key = _clean_identity_part(out["nodeb_id"]).fillna("unknown-site")
    else:
        site_key = out["original_node_cell_id"].fillna("unknown-site")

    sector_key = _clean_identity_part(out["sector"]) if "sector" in out.columns else pd.Series(pd.NA, index=out.index, dtype="string")
    cell_key = _clean_identity_part(out["cell_id"]) if "cell_id" in out.columns else out["original_node_cell_id"]
    sector_or_cell = sector_key.fillna(cell_key)
    band_col = next((col for col in ["band", "Band", "carrier", "frequency_mhz", "frequency"] if col in out.columns), None)
    band_key = _clean_identity_part(out[band_col]) if band_col else pd.Series(pd.NA, index=out.index, dtype="string")

    out["site_identity_key"] = site_key
    out["sector_identity"] = sector_or_cell
    out["frontend_site_sector_key"] = site_key.astype(str) + "|" + sector_or_cell.astype(str)
    out["node_cell_sector_key"] = out["original_node_cell_id"].astype(str) + "|" + sector_or_cell.astype(str)
    out["canonical_sector_id"] = _canonical_sector_ids(site_key, sector_or_cell, out["original_cell_id"])
    out["sector_identity_key"] = [
        build_sector_identity(site, cell, sector, fallback=fallback)
        for site, cell, sector, fallback in zip(site_key, cell_key, sector_key, out["original_node_cell_id"])
    ]
    out["site_sector_band_key"] = [
        build_site_sector_band_identity(site, sector, band)
        for site, sector, band in zip(site_key, sector_key, band_key)
    ]
    out["rf_identity_key"] = [
        build_rf_identity(site, cell, sector, band, fallback=fallback)
        for site, cell, sector, band, fallback in zip(site_key, cell_key, sector_key, band_key, out["original_node_cell_id"])
    ]
    out.loc[sector_or_cell.isna(), ["frontend_site_sector_key", "node_cell_sector_key"]] = pd.NA

    if use_as_node_cell_id:
        out["Node_Cell_ID"] = out["rf_identity_key"].where(out["rf_identity_key"].astype("string").str.strip().ne(""), out["frontend_site_sector_key"])
    return out


def filter_sites_to_project_polygon(site_df: pd.DataFrame, polygon_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if site_df.empty or polygon_gdf.empty or not {"lat", "lon"}.issubset(site_df.columns):
        return site_df.copy()

    work = site_df.copy()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    valid = work["lat"].notna() & work["lon"].notna()
    if not valid.any():
        return work.iloc[0:0].copy()

    point_gdf = gpd.GeoDataFrame(
        work.loc[valid].copy(),
        geometry=gpd.points_from_xy(work.loc[valid, "lon"], work.loc[valid, "lat"]),
        crs=polygon_gdf.crs or "EPSG:4326",
    )
    polygon_union = polygon_gdf.geometry.union_all()
    inside_index = point_gdf.loc[point_gdf.geometry.apply(lambda geom: bool(polygon_union.contains(geom)))].index
    return work.loc[inside_index].copy()


def prepare_site_df_for_source_rf_export(site_df: pd.DataFrame) -> pd.DataFrame:
    rf_df = normalize_site_for_geo(site_df)
    rf_df = add_sector_identity_columns(rf_df, use_as_node_cell_id=True)
    if "cell_id" in rf_df.columns and "rf_source_cell_id" not in rf_df.columns:
        rf_df["rf_source_cell_id"] = rf_df["cell_id"]
    if "Node_Cell_ID" in rf_df.columns:
        rf_df["cell_id"] = rf_df["Node_Cell_ID"]
    duplicate_aliases = {
        "Etilt": "electrical_tilt",
        "Mtilt": "mechanical_tilt",
        "Height": "antenna_height",
        "PCI": "pci",
    }
    for legacy_col, normalized_col in duplicate_aliases.items():
        if legacy_col in rf_df.columns and normalized_col in rf_df.columns:
            rf_df = rf_df.drop(columns=[legacy_col])
    return rf_df.loc[:, ~rf_df.columns.duplicated()].copy()


def attach_site_identity_to_predictions(pred_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    if out.empty or site_df.empty or "Node_Cell_ID" not in out.columns:
        return out

    site_identity = add_sector_identity_columns(site_df, use_as_node_cell_id=True)
    keep_cols = [
        col
        for col in [
            "Node_Cell_ID",
            "original_node_cell_id",
            "original_cell_id",
            "site_identity_key",
            "sector_identity",
            "frontend_site_sector_key",
            "node_cell_sector_key",
            "sector_identity_key",
            "site_sector_band_key",
            "rf_identity_key",
            "site",
            "Site ID",
            "sector",
            "band",
            "nodeb_id",
            "pci",
            "PCI",
            "earfcn",
            "azimuth",
            "canonical_sector_id",
        ]
        if col in site_identity.columns
    ]
    site_identity = site_identity[keep_cols].drop_duplicates(subset=["Node_Cell_ID"], keep="first")
    out["Node_Cell_ID"] = out["Node_Cell_ID"].astype(str).str.strip()
    site_identity["Node_Cell_ID"] = site_identity["Node_Cell_ID"].astype(str).str.strip()
    overlap_cols = [col for col in site_identity.columns if col != "Node_Cell_ID" and col in out.columns]
    if overlap_cols:
        out = out.drop(columns=overlap_cols, errors="ignore")
    return out.merge(site_identity, on="Node_Cell_ID", how="left")


def _candidate_building_height_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "height_m",
        "building_height_m",
        "building_height",
        "height",
        "bldg_height",
        "roof_height",
    ]
    return [col for col in candidates if col in df.columns]


def _candidate_building_level_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "building_levels",
        "levels",
        "floors",
        "num_floors",
        "storeys",
    ]
    return [col for col in candidates if col in df.columns]


def building_df_to_gdf(building_df: pd.DataFrame) -> gpd.GeoDataFrame:
    geom_col = None
    for candidate in ("region_wkt", "geometry_wkt", "geometry", "region"):
        if candidate not in building_df.columns:
            continue
        sample_series = building_df[candidate].dropna()
        if sample_series.empty:
            continue
        if any(_parse_geometry_value(value) is not None for value in sample_series.head(10).tolist()):
            geom_col = candidate
            break

    if geom_col is None:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    geometries = []
    records = []
    height_cols = _candidate_building_height_columns(building_df)
    level_cols = _candidate_building_level_columns(building_df)
    for _, row in building_df.iterrows():
        geom = _parse_geometry_value(row.get(geom_col))
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiPolygon":
            pieces = list(geom.geoms)
            if not pieces:
                continue
            geom = max(pieces, key=lambda g: g.area)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty or not geom.is_valid:
            continue
        geometries.append(geom)
        records.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "project_id": row.get("project_id"),
                "area_db": row.get("area"),
                "building_height_m": next(
                    (
                        pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                        for col in height_cols
                        if pd.notna(pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0])
                    ),
                    np.nan,
                ),
                "building_levels": next(
                    (
                        pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                        for col in level_cols
                        if pd.notna(pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0])
                    ),
                    np.nan,
                ),
            }
        )

    if not geometries:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    return gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")


def align_project_polygon_to_points(
    polygon_gdf: gpd.GeoDataFrame,
    points_df: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, str]:
    if polygon_gdf.empty or points_df.empty or not {"lat", "lon"}.issubset(points_df.columns):
        return polygon_gdf, "empty"

    points = points_df.copy()
    points["lat"] = pd.to_numeric(points["lat"], errors="coerce")
    points["lon"] = pd.to_numeric(points["lon"], errors="coerce")
    points = points.dropna(subset=["lat", "lon"]).copy()
    if points.empty:
        return polygon_gdf, "empty_points"

    point_gdf = gpd.GeoDataFrame(
        points[["lat", "lon"]],
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs="EPSG:4326",
    )
    project_union = polygon_gdf.geometry.union_all()
    direct_hits = int(point_gdf.geometry.within(project_union).sum())

    swapped = polygon_gdf.copy()
    swapped["geometry"] = swapped.geometry.apply(_swap_geometry_xy)
    swapped_hits = int(point_gdf.geometry.within(swapped.geometry.union_all()).sum())

    if swapped_hits > direct_hits:
        return swapped, f"swapped_xy direct_hits={direct_hits} swapped_hits={swapped_hits}"
    return polygon_gdf, f"original direct_hits={direct_hits} swapped_hits={swapped_hits}"


def align_building_geometries_to_project(
    building_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, str]:
    if building_gdf.empty or polygon_gdf.empty:
        return building_gdf, "empty"

    project_union = polygon_gdf.geometry.union_all()
    direct = building_gdf.copy()
    direct["_intersects"] = direct.geometry.intersects(project_union)
    direct_hits = int(direct["_intersects"].sum())

    swapped = building_gdf.copy()
    swapped["geometry"] = swapped.geometry.apply(_swap_geometry_xy)
    swapped["_intersects"] = swapped.geometry.intersects(project_union)
    swapped_hits = int(swapped["_intersects"].sum())

    if swapped_hits > direct_hits:
        return swapped.drop(columns=["_intersects"]), f"swapped_xy direct_hits={direct_hits} swapped_hits={swapped_hits}"
    return direct.drop(columns=["_intersects"]), f"original direct_hits={direct_hits} swapped_hits={swapped_hits}"


def prepare_building_df_for_rf(building_df: pd.DataFrame, building_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if building_gdf.empty:
        return building_df.copy()

    rf_df = building_df.copy().reset_index(drop=True)
    geom_count = min(len(rf_df), len(building_gdf))
    rf_df = rf_df.iloc[:geom_count].copy()
    geometry_wkt = building_gdf.geometry.to_wkt().reset_index(drop=True)
    rf_df["region_wkt"] = geometry_wkt
    rf_df["geometry_wkt"] = geometry_wkt
    rf_df["geometry"] = geometry_wkt
    rf_df["region"] = geometry_wkt
    return rf_df


def create_analysis_grid(mask_gdf: gpd.GeoDataFrame, cell_size_m: float) -> gpd.GeoDataFrame:
    utm_crs = _choose_utm_crs(mask_gdf)
    mask_utm = mask_gdf.to_crs(utm_crs)
    xmin, ymin, xmax, ymax = mask_utm.total_bounds

    polygons = []
    grid_ids = []
    idx = 1
    y = ymin
    while y < ymax:
        x = xmin
        while x < xmax:
            polygons.append(
                Polygon([(x, y), (x + cell_size_m, y), (x + cell_size_m, y + cell_size_m), (x, y + cell_size_m)])
            )
            grid_ids.append(idx)
            idx += 1
            x += cell_size_m
        y += cell_size_m

    grid_utm = gpd.GeoDataFrame({"grid_id": grid_ids, "geometry": polygons}, crs=utm_crs)
    clipped = gpd.overlay(grid_utm, mask_utm[["geometry"]], how="intersection", keep_geom_type=False)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notnull()].copy()
    clipped["cell_area_m2"] = clipped.geometry.area
    return clipped.to_crs("EPSG:4326")


def attach_building_features(grid_gdf: gpd.GeoDataFrame, building_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid_utm = grid_gdf.to_crs(_choose_utm_crs(grid_gdf))
    grid_utm["building_count"] = 0.0
    grid_utm["building_area_sum_m2"] = 0.0
    grid_utm["avg_building_area_m2"] = 0.0

    if building_gdf.empty:
        grid_utm["building_area_ratio"] = 0.0
        return grid_utm.to_crs("EPSG:4326")

    bld_utm = building_gdf.to_crs(grid_utm.crs).copy()
    bld_utm["building_area_m2"] = bld_utm.geometry.area
    centroids = bld_utm.copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin(
        centroids[["building_area_m2", "geometry"]],
        grid_utm[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    )
    agg = joined.groupby("grid_id").agg(
        building_count=("building_area_m2", "size"),
        building_area_sum_m2=("building_area_m2", "sum"),
        avg_building_area_m2=("building_area_m2", "mean"),
    )
    agg = agg.rename(
        columns={
            "building_count": "building_count_calc",
            "building_area_sum_m2": "building_area_sum_m2_calc",
            "avg_building_area_m2": "avg_building_area_m2_calc",
        }
    )
    grid_utm = grid_utm.merge(agg, on="grid_id", how="left")
    grid_utm["building_count"] = pd.to_numeric(grid_utm["building_count_calc"], errors="coerce").fillna(0.0)
    grid_utm["building_area_sum_m2"] = pd.to_numeric(grid_utm["building_area_sum_m2_calc"], errors="coerce").fillna(0.0)
    grid_utm["avg_building_area_m2"] = pd.to_numeric(grid_utm["avg_building_area_m2_calc"], errors="coerce").fillna(0.0)
    grid_utm["building_area_ratio"] = (
        grid_utm["building_area_sum_m2"] / grid_utm["cell_area_m2"].replace(0, np.nan)
    ).fillna(0)
    grid_utm = grid_utm.drop(
        columns=["building_count_calc", "building_area_sum_m2_calc", "avg_building_area_m2_calc"],
        errors="ignore",
    )
    return grid_utm.to_crs("EPSG:4326")


def _derive_clutter_class(df: pd.DataFrame) -> pd.Series:
    building_dense = df["building_area_ratio"] >= df["building_area_ratio"].quantile(0.75)
    building_mid = df["building_area_ratio"] >= df["building_area_ratio"].quantile(0.40)
    road_dense = df["road_length_m"] >= df["road_length_m"].quantile(0.75)
    road_mid = df["road_length_m"] >= df["road_length_m"].quantile(0.40)

    clutter = np.full(len(df), "Rural/Open", dtype=object)
    clutter = np.where(df["water_ratio"] >= 0.15, "Water", clutter)
    clutter = np.where(
        (df["green_ratio"] >= 0.30) & (df["water_ratio"] < 0.15) & (df["building_area_ratio"] < 0.08),
        "Vegetation",
        clutter,
    )
    clutter = np.where(building_dense & road_dense, "Dense Urban", clutter)
    clutter = np.where((clutter == "Rural/Open") & (building_mid | road_mid), "Urban", clutter)
    clutter = np.where(
        (clutter == "Rural/Open") & ((df["building_count"] > 0) | (df["road_length_m"] > 0)),
        "Suburban",
        clutter,
    )
    return pd.Series(clutter, index=df.index)


def _fit_morphology_clusters(grid_df: pd.DataFrame, cluster_count: int) -> pd.DataFrame:
    feature_cols = [
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
    ]
    work = grid_df.copy()
    for col in feature_cols:
        work[col] = pd.to_numeric(work.get(col, pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0)

    usable = work[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    if usable.empty or len(usable) < 2 or int(usable.drop_duplicates().shape[0]) <= 1:
        work["morphology_cluster"] = 0
        return work

    n_clusters = max(2, min(cluster_count, len(usable), int(usable.drop_duplicates().shape[0])))
    scaler = StandardScaler()
    X = scaler.fit_transform(usable)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    work["morphology_cluster"] = model.fit_predict(X)
    return work


def _fetch_osm_features(mask_gdf: gpd.GeoDataFrame, tags: Dict[str, object]) -> gpd.GeoDataFrame:
    if ox is None or mask_gdf.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    try:
        polygon = mask_gdf.to_crs("EPSG:4326").geometry.union_all()
        features = ox.features_from_polygon(polygon, tags=tags)
        if features is None or len(features) == 0:
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        if not isinstance(features, gpd.GeoDataFrame):
            features = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
        if features.crs is None:
            features = features.set_crs("EPSG:4326")
        else:
            features = features.to_crs("EPSG:4326")
        features = features[features.geometry.notnull() & ~features.geometry.is_empty].copy()
        return features
    except Exception as exc:
        print(f"[LTE][OSM_FETCH] tags={tags} failed={exc}")
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")


def _attach_osm_context_features(
    grid_gdf: gpd.GeoDataFrame,
    mask_gdf: gpd.GeoDataFrame,
    enabled: bool = DEFAULT_ENABLE_OSM_ENRICHMENT,
) -> gpd.GeoDataFrame:
    out = grid_gdf.copy()
    for col in ["road_length_m", "green_ratio", "water_ratio"]:
        if col not in out.columns:
            out[col] = 0.0

    if not enabled:
        print(f"[LTE][OSM_CONTEXT] skipped=True reason=disabled grid_rows={len(out)}", flush=True)
        return out

    if mask_gdf.empty:
        return out

    grid_utm = out.to_crs(_choose_utm_crs(out))
    mask_wgs84 = mask_gdf.to_crs("EPSG:4326")

    road_tags = {"highway": True}
    green_tags = {"landuse": ["forest", "grass", "meadow", "recreation_ground", "village_green"], "natural": ["wood", "grassland", "scrub"], "leisure": ["park", "garden", "nature_reserve", "pitch"]}
    water_tags = {"natural": ["water", "wetland"], "waterway": True, "landuse": ["reservoir", "basin"], "leisure": ["swimming_pool"]}

    roads = _fetch_osm_features(mask_wgs84, road_tags)
    green = _fetch_osm_features(mask_wgs84, green_tags)
    water = _fetch_osm_features(mask_wgs84, water_tags)

    if not roads.empty:
        roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].to_crs(grid_utm.crs).copy()
        if not roads.empty:
            road_join = gpd.overlay(
                roads[["geometry"]],
                grid_utm[["grid_id", "geometry"]],
                how="intersection",
                keep_geom_type=False,
            )
            if not road_join.empty:
                road_join["road_seg_m"] = road_join.geometry.length
                road_agg = road_join.groupby("grid_id")["road_seg_m"].sum()
                out["road_length_m"] = out["grid_id"].map(road_agg).fillna(0.0)

    def _area_ratio(layer: gpd.GeoDataFrame, out_col: str):
        nonlocal out
        if layer.empty:
            return
        layer = layer[layer.geometry.type.isin(["Polygon", "MultiPolygon"])].to_crs(grid_utm.crs).copy()
        if layer.empty:
            return
        area_join = gpd.overlay(
            layer[["geometry"]],
            grid_utm[["grid_id", "geometry", "cell_area_m2"]],
            how="intersection",
            keep_geom_type=False,
        )
        if area_join.empty:
            return
        area_join["clip_area_m2"] = area_join.geometry.area
        area_agg = area_join.groupby("grid_id")["clip_area_m2"].sum()
        ratios = (area_agg / grid_utm.set_index("grid_id")["cell_area_m2"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[out_col] = out["grid_id"].map(ratios).fillna(0.0).clip(0.0, 1.0)

    _area_ratio(green, "green_ratio")
    _area_ratio(water, "water_ratio")

    print(
        f"[LTE][OSM_CONTEXT] grid_rows={len(out)} roads={len(roads)} green={len(green)} water={len(water)} "
        f"road_non_zero={int((pd.to_numeric(out['road_length_m'], errors='coerce').fillna(0) > 0).sum())} "
        f"green_non_zero={int((pd.to_numeric(out['green_ratio'], errors='coerce').fillna(0) > 0).sum())} "
        f"water_non_zero={int((pd.to_numeric(out['water_ratio'], errors='coerce').fillna(0) > 0).sum())}"
    )
    return out


def _enrich_buildings_with_osm_heights(
    building_gdf: gpd.GeoDataFrame,
    mask_gdf: gpd.GeoDataFrame,
    enabled: bool = DEFAULT_ENABLE_OSM_ENRICHMENT,
) -> gpd.GeoDataFrame:
    out = _normalize_building_height_gdf(building_gdf)
    if not enabled:
        print(f"[LTE][OSM_BUILDING_HEIGHT] skipped=True reason=disabled local_rows={len(out)}", flush=True)
        return out
    if mask_gdf.empty or ox is None:
        return out

    missing_height = out.empty or pd.to_numeric(out.get("building_height_m"), errors="coerce").isna().all()
    if not missing_height:
        return out

    osm_buildings = _fetch_osm_features(mask_gdf, {"building": True})
    if osm_buildings.empty:
        return out

    osm_buildings = _normalize_building_height_gdf(osm_buildings)
    osm_buildings = osm_buildings[
        osm_buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
        & osm_buildings.geometry.notnull()
        & ~osm_buildings.geometry.is_empty
    ].copy()
    if osm_buildings.empty:
        return out

    height_series = pd.to_numeric(osm_buildings.get("building_height_m"), errors="coerce")
    if height_series.notna().sum() == 0:
        return out

    if out.empty:
        print(f"[LTE][OSM_BUILDING_HEIGHT] local_rows=0 osm_rows={len(osm_buildings)} height_rows={int(height_series.notna().sum())}")
        return osm_buildings

    utm_crs = _choose_utm_crs(mask_gdf)
    local_utm = out.to_crs(utm_crs).copy()
    osm_utm = osm_buildings.to_crs(utm_crs).copy()
    local_utm["geometry"] = local_utm.geometry.centroid
    osm_utm["geometry"] = osm_utm.geometry.centroid
    osm_utm = osm_utm[height_series.notna().to_numpy()].copy()
    if osm_utm.empty:
        return out

    joined = gpd.sjoin_nearest(
        local_utm[["geometry"]],
        osm_utm[["geometry", "building_height_m"]],
        how="left",
        distance_col="_height_match_m",
        max_distance=35.0,
    )
    matched_heights = pd.to_numeric(joined["building_height_m"], errors="coerce")
    matched_heights.index = out.index
    out["building_height_m"] = pd.to_numeric(out.get("building_height_m"), errors="coerce")
    out["building_height_m"] = out["building_height_m"].fillna(matched_heights)
    print(
        f"[LTE][OSM_BUILDING_HEIGHT] local_rows={len(out)} osm_rows={len(osm_buildings)} "
        f"height_rows={int(pd.to_numeric(out['building_height_m'], errors='coerce').notna().sum())}"
    )
    return out


def _safe_angle_delta_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _haversine_m_np(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * 6371000.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _bearing_deg_np(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _compute_proxy_rsrp_arrays(
    point_lat,
    point_lon,
    site_lat,
    site_lon,
    site_azimuth,
    site_height,
    site_tx_power,
    site_frequency_mhz,
    site_electrical_tilt,
    site_mechanical_tilt,
    site_elevation_m=None,
    point_elevation_m=None,
    local_k2_adjust_db=0.0,
):
    distance_m = np.maximum(_haversine_m_np(site_lat, site_lon, point_lat, point_lon), 1.0)
    distance_km = np.maximum(distance_m / 1000.0, 0.001)
    freq = np.clip(np.asarray(site_frequency_mhz, dtype=float), 700.0, 3500.0)
    h_tx = np.asarray(site_height, dtype=float)
    if site_elevation_m is not None and point_elevation_m is not None:
        h_tx = h_tx + (np.asarray(site_elevation_m, dtype=float) - np.asarray(point_elevation_m, dtype=float))
    h_tx = np.clip(h_tx, 5.0, 180.0)
    h_rx = 1.5
    a_hm = (1.1 * np.log10(freq) - 0.7) * h_rx - (1.56 * np.log10(freq) - 0.8)
    slope_term = (44.9 - 6.55 * np.log10(h_tx)) + np.asarray(local_k2_adjust_db, dtype=float)
    pathloss = (
        46.3
        + 33.9 * np.log10(freq)
        - 13.82 * np.log10(h_tx)
        - a_hm
        + 3.0
        + slope_term * np.log10(distance_km)
    )
    bearing = _bearing_deg_np(site_lat, site_lon, point_lat, point_lon)
    az_diff = np.abs((bearing - np.asarray(site_azimuth, dtype=float) + 180.0) % 360.0 - 180.0)
    elev_angle = np.degrees(np.arctan2(h_rx - h_tx, distance_m))
    total_tilt = np.asarray(site_electrical_tilt, dtype=float) + np.asarray(site_mechanical_tilt, dtype=float)
    elev_diff = np.abs(elev_angle + total_tilt)
    ah = np.where(
        az_diff <= 90.0,
        np.minimum(12.0 * (az_diff / 65.0) ** 2, 25.0),
        np.minimum(22.0 + 8.0 * np.sin(np.radians(az_diff - 90.0)) ** 2, 32.0),
    )
    av = np.minimum(12.0 * (elev_diff / 6.0) ** 2, 20.0)
    gain = 18.0 - np.minimum(ah + av, 30.0)
    return np.asarray(site_tx_power, dtype=float) + gain - pathloss - 2.0


def attach_site_context_features(points_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    points = points_df.copy()
    if points.empty or site_df.empty or not {"lat", "lon"}.issubset(points.columns):
        return points

    site_work = normalize_site_for_geo(site_df)
    site_work = site_work.dropna(subset=["lat", "lon"]).copy()
    if site_work.empty:
        return points

    def _series_or_default(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index, dtype=float)

    point_lat = pd.to_numeric(points["lat"], errors="coerce").to_numpy(dtype=float)
    point_lon = pd.to_numeric(points["lon"], errors="coerce").to_numpy(dtype=float)
    site_lat = site_work["lat"].to_numpy(dtype=float)
    site_lon = site_work["lon"].to_numpy(dtype=float)

    point_rad = np.radians(np.c_[point_lat, point_lon])
    site_rad = np.radians(np.c_[site_lat, site_lon])
    tree = BallTree(site_rad, metric="haversine")
    k = min(4, len(site_work))
    dist_rad, idx = tree.query(point_rad, k=k)
    earth_radius_m = 6371000.0
    dist_m = dist_rad * earth_radius_m
    points["nearest_site_distance_m"] = dist_m[:, 0]
    points["mean_nearest3_site_distance_m"] = dist_m.mean(axis=1)
    points["site_count_250m"] = np.array([len(x) for x in tree.query_radius(point_rad, r=250.0 / earth_radius_m)])
    points["site_count_500m"] = np.array([len(x) for x in tree.query_radius(point_rad, r=500.0 / earth_radius_m)])

    nearest_rows = site_work.iloc[idx[:, 0]].reset_index(drop=True)
    points["_proxy_site_id"] = nearest_rows["Node_Cell_ID"].astype(str).values
    points["_proxy_site_lat"] = pd.to_numeric(nearest_rows["lat"], errors="coerce").values
    points["_proxy_site_lon"] = pd.to_numeric(nearest_rows["lon"], errors="coerce").values
    points["_proxy_site_azimuth"] = _series_or_default(nearest_rows, "azimuth", 0).values
    points["_proxy_site_height_m"] = _series_or_default(nearest_rows, "antenna_height", 30).values
    points["_proxy_site_tx_power"] = _series_or_default(nearest_rows, "tx_power", 46).values
    points["_proxy_site_frequency_mhz"] = _series_or_default(nearest_rows, "frequency_mhz", 1800).values
    points["_proxy_site_etilt"] = _series_or_default(nearest_rows, "electrical_tilt", 3).values
    points["_proxy_site_mtilt"] = _series_or_default(nearest_rows, "mechanical_tilt", 0).values

    proxy_bearing = _bearing_deg_np(
        points["_proxy_site_lat"].to_numpy(dtype=float),
        points["_proxy_site_lon"].to_numpy(dtype=float),
        point_lat,
        point_lon,
    )
    points["serving_distance_m"] = dist_m[:, 0]
    points["azimuth_delta_deg"] = np.abs(
        (proxy_bearing - points["_proxy_site_azimuth"].to_numpy(dtype=float) + 180.0) % 360.0 - 180.0
    )
    points["serving_proxy_rsrp_dbm"] = _compute_proxy_rsrp_arrays(
        point_lat,
        point_lon,
        points["_proxy_site_lat"].to_numpy(dtype=float),
        points["_proxy_site_lon"].to_numpy(dtype=float),
        points["_proxy_site_azimuth"].to_numpy(dtype=float),
        points["_proxy_site_height_m"].to_numpy(dtype=float),
        points["_proxy_site_tx_power"].to_numpy(dtype=float),
        points["_proxy_site_frequency_mhz"].to_numpy(dtype=float),
        points["_proxy_site_etilt"].to_numpy(dtype=float),
        points["_proxy_site_mtilt"].to_numpy(dtype=float),
    )

    if k >= 2:
        interferer_rows = site_work.iloc[idx[:, 1]].reset_index(drop=True)
        points["best_interferer_distance_m"] = dist_m[:, 1]
        interferer_azimuth = _series_or_default(interferer_rows, "azimuth", 0).to_numpy(dtype=float)
        interferer_height = _series_or_default(interferer_rows, "antenna_height", 30).to_numpy(dtype=float)
        interferer_tx = _series_or_default(interferer_rows, "tx_power", 46).to_numpy(dtype=float)
        interferer_freq = _series_or_default(interferer_rows, "frequency_mhz", 1800).to_numpy(dtype=float)
        interferer_etilt = _series_or_default(interferer_rows, "electrical_tilt", 3).to_numpy(dtype=float)
        interferer_mtilt = _series_or_default(interferer_rows, "mechanical_tilt", 0).to_numpy(dtype=float)
        interferer_bearing = _bearing_deg_np(
            pd.to_numeric(interferer_rows["lat"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(interferer_rows["lon"], errors="coerce").to_numpy(dtype=float),
            point_lat,
            point_lon,
        )
        points["best_interferer_azimuth_delta_deg"] = np.abs((interferer_bearing - interferer_azimuth + 180.0) % 360.0 - 180.0)
        points["best_interferer_proxy_rsrp_dbm"] = _compute_proxy_rsrp_arrays(
            point_lat,
            point_lon,
            pd.to_numeric(interferer_rows["lat"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(interferer_rows["lon"], errors="coerce").to_numpy(dtype=float),
            interferer_azimuth,
            interferer_height,
            interferer_tx,
            interferer_freq,
            interferer_etilt,
            interferer_mtilt,
        )
        points["interference_gap_db"] = (
            pd.to_numeric(points["serving_proxy_rsrp_dbm"], errors="coerce")
            - pd.to_numeric(points["best_interferer_proxy_rsrp_dbm"], errors="coerce")
        )
        points["interference_ratio_linear"] = np.power(
            10.0,
            (
                pd.to_numeric(points["best_interferer_proxy_rsrp_dbm"], errors="coerce")
                - pd.to_numeric(points["serving_proxy_rsrp_dbm"], errors="coerce")
            ) / 10.0,
        )
    else:
        points["best_interferer_distance_m"] = np.nan
        points["best_interferer_azimuth_delta_deg"] = np.nan
        points["best_interferer_proxy_rsrp_dbm"] = np.nan
        points["interference_gap_db"] = np.nan
        points["interference_ratio_linear"] = np.nan

    site_freq_all = _series_or_default(site_work, "frequency_mhz", 1800).to_numpy(dtype=float)
    site_az_all = _series_or_default(site_work, "azimuth", 0).to_numpy(dtype=float)
    site_height_all = _series_or_default(site_work, "antenna_height", 30).to_numpy(dtype=float)
    site_tx_all = _series_or_default(site_work, "tx_power", 46).to_numpy(dtype=float)
    site_etilt_all = _series_or_default(site_work, "electrical_tilt", 3).to_numpy(dtype=float)
    site_mtilt_all = _series_or_default(site_work, "mechanical_tilt", 0).to_numpy(dtype=float)
    serving_proxy_phys = np.full(len(points), np.nan, dtype=float)
    best_interferer_phys = np.full(len(points), np.nan, dtype=float)
    interference_sum_dbm = np.full(len(points), np.nan, dtype=float)
    sinr_proxy_db = np.full(len(points), np.nan, dtype=float)
    rsrq_proxy_db = np.full(len(points), np.nan, dtype=float)
    effective_tx_height = np.full(len(points), np.nan, dtype=float)
    noise_linear = 10 ** (-104.0 / 10.0)
    n_rb = 50.0
    max_candidates = min(len(site_work), 24)
    _, all_idx = tree.query(point_rad, k=max_candidates)
    for row_idx in range(len(points)):
        candidate_idx = np.unique(all_idx[row_idx])
        cand_lat = site_lat[candidate_idx]
        cand_lon = site_lon[candidate_idx]
        cand_az = site_az_all[candidate_idx]
        cand_height = site_height_all[candidate_idx]
        cand_tx = site_tx_all[candidate_idx]
        cand_freq = site_freq_all[candidate_idx]
        cand_etilt = site_etilt_all[candidate_idx]
        cand_mtilt = site_mtilt_all[candidate_idx]
        local_distances = _haversine_m_np(
            cand_lat,
            cand_lon,
            np.full(len(candidate_idx), point_lat[row_idx], dtype=float),
            np.full(len(candidate_idx), point_lon[row_idx], dtype=float),
        )
        local_k2_adjust = np.where(local_distances > 250.0, 2.5, 0.8)
        rsrp_all = _compute_proxy_rsrp_arrays(
            np.full(len(candidate_idx), point_lat[row_idx], dtype=float),
            np.full(len(candidate_idx), point_lon[row_idx], dtype=float),
            cand_lat,
            cand_lon,
            cand_az,
            cand_height,
            cand_tx,
            cand_freq,
            cand_etilt,
            cand_mtilt,
            local_k2_adjust_db=local_k2_adjust,
        )
        order = np.argsort(rsrp_all)[::-1]
        if len(order) == 0:
            continue
        best_idx = int(order[0])
        serving_proxy_phys[row_idx] = float(rsrp_all[best_idx])
        effective_tx_height[row_idx] = float(cand_height[best_idx])
        if len(order) > 1:
            best_interferer_phys[row_idx] = float(rsrp_all[int(order[1])])
        linear = np.power(10.0, rsrp_all / 10.0)
        best_linear = linear[best_idx]
        total_linear = float(np.sum(linear))
        interference_linear = max(total_linear - float(best_linear) + noise_linear, noise_linear)
        interference_sum_dbm[row_idx] = float(10.0 * np.log10(interference_linear))
        sinr_proxy_db[row_idx] = float(10.0 * np.log10(best_linear / interference_linear))
        rssi_dbm = float(10.0 * np.log10(total_linear + noise_linear))
        rsrq_proxy_db[row_idx] = float(serving_proxy_phys[row_idx] - rssi_dbm + 10.0 * np.log10(n_rb))

    points["serving_proxy_rsrp_phys_dbm"] = serving_proxy_phys
    points["best_interferer_proxy_phys_dbm"] = best_interferer_phys
    points["interference_sum_proxy_dbm"] = interference_sum_dbm
    points["sinr_proxy_db"] = sinr_proxy_db
    points["rsrq_proxy_db"] = rsrq_proxy_db
    points["effective_tx_height_m"] = effective_tx_height
    if "best_interferer_proxy_phys_dbm" in points.columns:
        points["interference_gap_db"] = (
            pd.to_numeric(points["serving_proxy_rsrp_phys_dbm"], errors="coerce")
            - pd.to_numeric(points["best_interferer_proxy_phys_dbm"], errors="coerce")
        )
        points["interference_ratio_linear"] = np.power(
            10.0,
            (
                pd.to_numeric(points["best_interferer_proxy_phys_dbm"], errors="coerce")
                - pd.to_numeric(points["serving_proxy_rsrp_phys_dbm"], errors="coerce")
            ) / 10.0,
        )
    return points


def build_grid_feature_frame(
    grid_gdf: gpd.GeoDataFrame,
    site_df: pd.DataFrame,
    cluster_count: int,
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    grid_df = pd.DataFrame(grid_gdf.drop(columns="geometry"))
    grid_centroids = grid_gdf.to_crs(_choose_utm_crs(grid_gdf)).copy()
    grid_centroids["geometry"] = grid_centroids.geometry.centroid
    grid_centroids = grid_centroids.to_crs("EPSG:4326")
    grid_centroids["lat"] = grid_centroids.geometry.y
    grid_centroids["lon"] = grid_centroids.geometry.x

    grid_df["lat"] = grid_centroids["lat"].values
    grid_df["lon"] = grid_centroids["lon"].values
    for col in ["road_length_m", "green_ratio", "water_ratio"]:
        if col not in grid_df.columns:
            grid_df[col] = 0.0
        else:
            grid_df[col] = pd.to_numeric(grid_df[col], errors="coerce").fillna(0.0)
    grid_site_context = attach_site_context_features(grid_centroids[["grid_id", "lat", "lon"]], site_df).drop(
        columns=["lat", "lon"],
        errors="ignore",
    )
    grid_df = grid_df.merge(grid_site_context, on="grid_id", how="left")
    grid_df["clutter_class"] = _derive_clutter_class(grid_df)
    grid_df = _fit_morphology_clusters(grid_df, cluster_count)
    return grid_df, grid_centroids


def _normalize_building_height_gdf(building_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = building_gdf.copy()
    if out.empty:
        out["building_height_m"] = pd.Series(dtype=float)
        return out
    out["building_height_m"] = pd.to_numeric(out.get("building_height_m"), errors="coerce")
    for col in [c for c in ["height_m", "height", "building:height", "building_height", "roof_height"] if c in out.columns]:
        out["building_height_m"] = out["building_height_m"].fillna(pd.to_numeric(out[col], errors="coerce"))
    for col in [c for c in ["building_levels", "levels", "building:levels", "floors", "num_floors"] if c in out.columns]:
        out["building_height_m"] = out["building_height_m"].fillna(pd.to_numeric(out[col], errors="coerce") * 3.0)
    return out


def _attach_building_path_features(points_df: pd.DataFrame, building_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    out = points_df.copy()
    default_cols = {
        "los_blocker_count": 0.0,
        "los_blocked_length_m": 0.0,
        "los_blocked_ratio": 0.0,
        "mean_blocker_height_m": 0.0,
        "max_blocker_height_m": 0.0,
        "nlos_flag": 0.0,
        "diffraction_proxy_db": 0.0,
    }
    for col, default in default_cols.items():
        if col not in out.columns:
            out[col] = default

    required = {"lat", "lon", "_proxy_site_lat", "_proxy_site_lon"}
    if out.empty or building_gdf.empty or not required.issubset(out.columns):
        return out

    building_gdf = _normalize_building_height_gdf(building_gdf)
    utm_crs = _choose_utm_crs(building_gdf)
    building_utm = building_gdf.to_crs(utm_crs).copy()
    building_utm["building_height_m"] = pd.to_numeric(building_utm["building_height_m"], errors="coerce")
    sindex = building_utm.sindex
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    blocker_count = np.zeros(len(out), dtype=float)
    blocked_length = np.zeros(len(out), dtype=float)
    mean_height = np.zeros(len(out), dtype=float)
    max_height = np.zeros(len(out), dtype=float)

    for row_idx, (_, row) in enumerate(out.iterrows()):
        if any(pd.isna(row.get(col)) for col in ["lat", "lon", "_proxy_site_lat", "_proxy_site_lon"]):
            continue
        site_x, site_y = transformer.transform(float(row["_proxy_site_lon"]), float(row["_proxy_site_lat"]))
        point_x, point_y = transformer.transform(float(row["lon"]), float(row["lat"]))
        path = LineString([(site_x, site_y), (point_x, point_y)])
        if path.length <= 0:
            continue
        candidate_idx = list(sindex.intersection(path.bounds))
        if not candidate_idx:
            continue
        hits = building_utm.iloc[candidate_idx]
        hits = hits[hits.geometry.intersects(path)].copy()
        if hits.empty:
            continue
        lengths = np.array(
            [geom.length for geom in hits.geometry.intersection(path) if geom is not None and not geom.is_empty],
            dtype=float,
        )
        blocker_count[row_idx] = float(len(hits))
        blocked_length[row_idx] = float(lengths.sum()) if len(lengths) else 0.0
        valid_heights = pd.to_numeric(hits.get("building_height_m", pd.Series(dtype=float)), errors="coerce").dropna()
        if not valid_heights.empty:
            mean_height[row_idx] = float(valid_heights.mean())
            max_height[row_idx] = float(valid_heights.max())

    out["los_blocker_count"] = blocker_count
    out["los_blocked_length_m"] = blocked_length
    out["los_blocked_ratio"] = blocked_length / np.maximum(
        pd.to_numeric(out["serving_distance_m"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
        1.0,
    )
    out["mean_blocker_height_m"] = mean_height
    out["max_blocker_height_m"] = max_height
    out["nlos_flag"] = (out["los_blocker_count"] > 0).astype(float)
    out["diffraction_proxy_db"] = (
        1.4 * out["los_blocker_count"].clip(0, 8)
        + 9.0 * out["los_blocked_ratio"].clip(0, 1.0)
        + 0.04 * out["max_blocker_height_m"].clip(0, 80.0)
    )
    return out


def _attach_dem_features(points_df: pd.DataFrame, dem_raster_path: Optional[str | Path]) -> tuple[pd.DataFrame, Dict[str, object]]:
    out = points_df.copy()
    status = {"enabled": dem_raster_path is not None, "sampled": False, "reason": None}
    for col in ["terrain_elevation_m", "terrain_slope_deg", "proxy_site_elevation_m", "terrain_relief_to_site_m"]:
        if col not in out.columns:
            out[col] = np.nan
    if dem_raster_path is None:
        status["reason"] = "dem_not_configured"
        return out, status
    if rasterio is None:
        status["reason"] = "rasterio_unavailable"
        return out, status
    dem_path = Path(dem_raster_path)
    if not dem_path.exists():
        status["reason"] = "dem_file_missing"
        return out, status

    with rasterio.open(dem_path) as src:
        if src.crs is None:
            status["reason"] = "dem_crs_missing"
            return out, status
        to_dem = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        point_xy = [to_dem.transform(float(lon), float(lat)) for lat, lon in zip(out["lat"], out["lon"])]
        point_samples = np.array([sample[0] for sample in src.sample(point_xy)], dtype=float)
        nodata = src.nodata
        if nodata is not None:
            point_samples = np.where(np.isclose(point_samples, nodata), np.nan, point_samples)
        out["terrain_elevation_m"] = point_samples

        dx = abs(float(src.transform.a)) or 1.0
        dy = abs(float(src.transform.e)) or 1.0
        west = np.array([sample[0] for sample in src.sample([(x - dx, y) for x, y in point_xy])], dtype=float)
        east = np.array([sample[0] for sample in src.sample([(x + dx, y) for x, y in point_xy])], dtype=float)
        south = np.array([sample[0] for sample in src.sample([(x, y - dy) for x, y in point_xy])], dtype=float)
        north = np.array([sample[0] for sample in src.sample([(x, y + dy) for x, y in point_xy])], dtype=float)
        if nodata is not None:
            for arr in (west, east, south, north):
                arr[np.isclose(arr, nodata)] = np.nan
        grad_x = (east - west) / max(2.0 * dx, 1.0)
        grad_y = (north - south) / max(2.0 * dy, 1.0)
        out["terrain_slope_deg"] = np.degrees(np.arctan(np.sqrt(np.square(grad_x) + np.square(grad_y))))

        if {"_proxy_site_lat", "_proxy_site_lon"}.issubset(out.columns):
            site_xy = [
                to_dem.transform(float(lon), float(lat)) if pd.notna(lat) and pd.notna(lon) else (np.nan, np.nan)
                for lat, lon in zip(out["_proxy_site_lat"], out["_proxy_site_lon"])
            ]
            valid_mask = np.array([np.isfinite(x) and np.isfinite(y) for x, y in site_xy], dtype=bool)
            site_samples = np.full(len(out), np.nan, dtype=float)
            if valid_mask.any():
                sampled = np.array([sample[0] for sample in src.sample([site_xy[i] for i in np.where(valid_mask)[0]])], dtype=float)
                if nodata is not None:
                    sampled = np.where(np.isclose(sampled, nodata), np.nan, sampled)
                site_samples[valid_mask] = sampled
            out["proxy_site_elevation_m"] = site_samples
            out["terrain_relief_to_site_m"] = out["terrain_elevation_m"] - out["proxy_site_elevation_m"]

    status["sampled"] = True
    return out, status


def _refine_experimental_forward_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "serving_proxy_rsrp_phys_dbm",
        "best_interferer_proxy_phys_dbm",
        "serving_proxy_rsrp_dbm",
        "best_interferer_proxy_rsrp_dbm",
        "terrain_elevation_m",
        "proxy_site_elevation_m",
        "terrain_relief_to_site_m",
        "terrain_slope_deg",
        "los_blocker_count",
        "los_blocked_ratio",
        "diffraction_proxy_db",
        "max_blocker_height_m",
        "azimuth_delta_deg",
        "best_interferer_azimuth_delta_deg",
        "serving_distance_m",
        "best_interferer_distance_m",
        "effective_tx_height_m",
    ]:
        out[col] = pd.to_numeric(out.get(col, pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)

    relief_penalty = -0.022 * out["terrain_relief_to_site_m"].clip(lower=0.0, upper=180.0)
    slope_penalty = -0.06 * out["terrain_slope_deg"].clip(0.0, 30.0)
    obstruction_penalty = (
        -0.85 * out["los_blocker_count"].clip(0.0, 8.0)
        - 4.8 * out["los_blocked_ratio"].clip(0.0, 1.0)
        - 0.035 * out["max_blocker_height_m"].clip(0.0, 80.0)
        - 0.45 * out["diffraction_proxy_db"].clip(0.0, 25.0)
    )
    off_axis_penalty = -0.010 * out["azimuth_delta_deg"].clip(0.0, 180.0)
    height_bonus = 0.12 * np.log1p(out["effective_tx_height_m"].clip(5.0, 180.0) - 4.0)
    out["serving_proxy_rsrp_phys_dbm"] = (
        out["serving_proxy_rsrp_phys_dbm"] + relief_penalty + slope_penalty + obstruction_penalty + off_axis_penalty + height_bonus
    )
    out["best_interferer_proxy_phys_dbm"] = (
        out["best_interferer_proxy_phys_dbm"]
        - 0.010 * out["terrain_relief_to_site_m"].clip(lower=0.0, upper=180.0)
        - 0.006 * out["best_interferer_azimuth_delta_deg"].clip(0.0, 180.0)
        - 0.0012 * out["best_interferer_distance_m"].clip(0.0, 1200.0)
    )
    out["interference_gap_db"] = out["serving_proxy_rsrp_phys_dbm"] - out["best_interferer_proxy_phys_dbm"]
    out["interference_ratio_linear"] = np.power(
        10.0, (out["best_interferer_proxy_phys_dbm"] - out["serving_proxy_rsrp_phys_dbm"]) / 10.0
    )
    noise_linear = 10 ** (-104.0 / 10.0)
    serving_linear = np.power(10.0, out["serving_proxy_rsrp_phys_dbm"] / 10.0)
    interference_linear = np.power(10.0, pd.to_numeric(out.get("interference_sum_proxy_dbm", -120.0), errors="coerce").fillna(-120.0) / 10.0)
    interference_linear = np.maximum(interference_linear, noise_linear)
    out["sinr_proxy_db"] = 10.0 * np.log10(np.maximum(serving_linear, noise_linear) / interference_linear)
    rssi_linear = serving_linear + interference_linear
    out["rsrq_proxy_db"] = out["serving_proxy_rsrp_phys_dbm"] - (10.0 * np.log10(rssi_linear)) + 10.0 * np.log10(50.0)
    return out


def augment_grid_with_advanced_geo_features(
    grid_df: pd.DataFrame,
    building_gdf: gpd.GeoDataFrame,
    site_df: pd.DataFrame,
    dem_raster_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    out = attach_site_context_features(grid_df, site_df)
    out = _attach_building_path_features(out, building_gdf)
    out, dem_status = _attach_dem_features(out, dem_raster_path)
    out = _refine_experimental_forward_features(out)
    helper_cols = [
        "_proxy_site_id",
        "_proxy_site_lat",
        "_proxy_site_lon",
        "_proxy_site_azimuth",
        "_proxy_site_height_m",
        "_proxy_site_tx_power",
        "_proxy_site_frequency_mhz",
        "_proxy_site_etilt",
        "_proxy_site_mtilt",
    ]
    out = out.drop(columns=helper_cols, errors="ignore")
    return out, {"dem_status": dem_status}


def assign_points_to_tiles(points_df: pd.DataFrame, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    points = points_df.copy()
    grid_cols = [col for col in grid_gdf.columns if col != "geometry"]
    overlap_cols = [col for col in grid_cols if col in points.columns]
    if overlap_cols:
        points = points.drop(columns=overlap_cols, errors="ignore")
    point_gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
    joined = gpd.sjoin(point_gdf, grid_gdf, how="left", predicate="within")
    missing = joined["grid_id"].isna()
    if missing.any():
        utm_crs = _choose_utm_crs(grid_gdf)
        point_missing_utm = point_gdf.loc[missing, ["geometry"]].to_crs(utm_crs)
        grid_utm = grid_gdf.to_crs(utm_crs)
        nearest = gpd.sjoin_nearest(point_missing_utm, grid_utm, how="left", distance_col="_tile_distance").to_crs("EPSG:4326")
        for col in grid_gdf.columns:
            if col != "geometry":
                joined.loc[missing, col] = nearest[col].values
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


def _attach_missing_grid_features_by_grid_id(pred_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    if "grid_id" not in out.columns or "grid_id" not in grid_df.columns:
        return out
    feature_cols = [col for col in grid_df.columns if col not in {"grid_id", "lat", "lon"}]
    missing_cols = [col for col in feature_cols if col not in out.columns]
    if not missing_cols:
        return out
    return out.merge(grid_df[["grid_id"] + missing_cols], on="grid_id", how="left")


def attach_fixed_serving_sinr_rsrq_proxy(points_df: pd.DataFrame, site_df: pd.DataFrame, max_interferers: int = 24) -> pd.DataFrame:
    out = points_df.copy()
    if out.empty or site_df.empty or not {"lat", "lon", "Node_Cell_ID"}.issubset(out.columns):
        return out

    site_work = normalize_site_for_geo(site_df)
    site_work["Node_Cell_ID"] = site_work["Node_Cell_ID"].astype(str).str.strip()
    site_work = site_work.dropna(subset=["lat", "lon"]).copy()
    serving_sites = site_work.sort_values("Node_Cell_ID").drop_duplicates(subset=["Node_Cell_ID"], keep="first").reset_index(drop=True)
    if serving_sites.empty:
        return out

    def _series_or_default(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index, dtype=float)

    serving_lookup = {cell_id: idx for idx, cell_id in enumerate(serving_sites["Node_Cell_ID"].tolist())}
    site_lat = serving_sites["lat"].to_numpy(dtype=float)
    site_lon = serving_sites["lon"].to_numpy(dtype=float)
    site_az = _series_or_default(serving_sites, "azimuth", 0.0).to_numpy(dtype=float)
    site_height = _series_or_default(serving_sites, "antenna_height", 30.0).to_numpy(dtype=float)
    site_tx = _series_or_default(serving_sites, "tx_power", 46.0).to_numpy(dtype=float)
    site_freq = _series_or_default(serving_sites, "frequency_mhz", 1800.0).to_numpy(dtype=float)
    site_etilt = _series_or_default(serving_sites, "electrical_tilt", 3.0).to_numpy(dtype=float)
    site_mtilt = _series_or_default(serving_sites, "mechanical_tilt", 0.0).to_numpy(dtype=float)

    point_lat = pd.to_numeric(out["lat"], errors="coerce").to_numpy(dtype=float)
    point_lon = pd.to_numeric(out["lon"], errors="coerce").to_numpy(dtype=float)
    point_cells = out["Node_Cell_ID"].astype(str).str.strip().to_numpy(dtype=object)
    point_rad = np.radians(np.c_[point_lat, point_lon])
    site_rad = np.radians(np.c_[site_lat, site_lon])
    tree = BallTree(site_rad, metric="haversine")
    candidate_k = max(1, min(int(max_interferers), len(serving_sites)))
    _, candidate_idx = tree.query(point_rad, k=candidate_k)
    if candidate_idx.ndim == 1:
        candidate_idx = candidate_idx.reshape(-1, 1)

    sinr_proxy_db = pd.to_numeric(out.get("sinr_proxy_db"), errors="coerce").to_numpy(dtype=float)
    rsrq_proxy_db = pd.to_numeric(out.get("rsrq_proxy_db"), errors="coerce").to_numpy(dtype=float)
    best_interferer_proxy = pd.to_numeric(out.get("best_interferer_proxy_phys_dbm"), errors="coerce").to_numpy(dtype=float)
    best_interferer_distance = pd.to_numeric(out.get("best_interferer_distance_m"), errors="coerce").to_numpy(dtype=float)
    best_interferer_az_delta = pd.to_numeric(out.get("best_interferer_azimuth_delta_deg"), errors="coerce").to_numpy(dtype=float)
    interference_sum_dbm = pd.to_numeric(out.get("interference_sum_proxy_dbm"), errors="coerce").to_numpy(dtype=float)
    interference_gap_db = pd.to_numeric(out.get("interference_gap_db"), errors="coerce").to_numpy(dtype=float)
    interference_ratio = pd.to_numeric(out.get("interference_ratio_linear"), errors="coerce").to_numpy(dtype=float)

    noise_linear = 10 ** (-104.0 / 10.0)
    n_rb = 50.0

    for row_idx, cell_id in enumerate(point_cells):
        serving_idx = serving_lookup.get(str(cell_id))
        if serving_idx is None or not np.isfinite(point_lat[row_idx]) or not np.isfinite(point_lon[row_idx]):
            continue
        local_candidates = np.unique(candidate_idx[row_idx]).astype(int).tolist()
        if serving_idx not in local_candidates:
            local_candidates.append(serving_idx)
        candidate_arr = np.array(local_candidates, dtype=int)
        rsrp_all = _compute_proxy_rsrp_arrays(
            np.full(len(candidate_arr), point_lat[row_idx], dtype=float),
            np.full(len(candidate_arr), point_lon[row_idx], dtype=float),
            site_lat[candidate_arr],
            site_lon[candidate_arr],
            site_az[candidate_arr],
            site_height[candidate_arr],
            site_tx[candidate_arr],
            site_freq[candidate_arr],
            site_etilt[candidate_arr],
            site_mtilt[candidate_arr],
        )
        serving_mask = candidate_arr == serving_idx
        if not serving_mask.any():
            continue
        linear = np.power(10.0, rsrp_all / 10.0)
        serving_linear = float(np.sum(linear[serving_mask]))
        interference_linear_raw = float(np.sum(linear[~serving_mask]))
        interference_linear = max(interference_linear_raw + noise_linear, noise_linear)
        if serving_linear <= 0.0:
            continue

        sinr_proxy_db[row_idx] = float(10.0 * np.log10(serving_linear / interference_linear))
        rssi_linear = max(serving_linear + interference_linear_raw + noise_linear, noise_linear)
        rsrq_proxy_db[row_idx] = float(rsrp_all[serving_mask][0] - (10.0 * np.log10(rssi_linear)) + 10.0 * np.log10(n_rb))
        interference_sum_dbm[row_idx] = float(10.0 * np.log10(interference_linear))

        if (~serving_mask).any():
            interferer_rsrp = rsrp_all[~serving_mask]
            interferer_linear = linear[~serving_mask]
            best_local_idx = int(np.argmax(interferer_linear))
            best_interferer_proxy[row_idx] = float(interferer_rsrp[best_local_idx])
            interferer_candidate_idx = candidate_arr[~serving_mask][best_local_idx]
            best_interferer_distance[row_idx] = float(
                _haversine_m_np(
                    site_lat[interferer_candidate_idx],
                    site_lon[interferer_candidate_idx],
                    point_lat[row_idx],
                    point_lon[row_idx],
                )
            )
            interferer_bearing = float(
                _bearing_deg_np(
                    site_lat[interferer_candidate_idx],
                    site_lon[interferer_candidate_idx],
                    point_lat[row_idx],
                    point_lon[row_idx],
                )
            )
            best_interferer_az_delta[row_idx] = float(abs((interferer_bearing - site_az[interferer_candidate_idx] + 180.0) % 360.0 - 180.0))
            serving_rsrp = float(rsrp_all[serving_mask][0])
            interference_gap_db[row_idx] = serving_rsrp - best_interferer_proxy[row_idx]
            interference_ratio[row_idx] = float(np.power(10.0, (best_interferer_proxy[row_idx] - serving_rsrp) / 10.0))

    out["sinr_proxy_db"] = sinr_proxy_db
    out["rsrq_proxy_db"] = rsrq_proxy_db
    out["best_interferer_proxy_phys_dbm"] = best_interferer_proxy
    out["best_interferer_distance_m"] = best_interferer_distance
    out["best_interferer_azimuth_delta_deg"] = best_interferer_az_delta
    out["interference_sum_proxy_dbm"] = interference_sum_dbm
    out["interference_gap_db"] = interference_gap_db
    out["interference_ratio_linear"] = interference_ratio
    return out


def _copy_first_present(df: pd.DataFrame, target: str, candidates: list[str]) -> None:
    if target in df.columns:
        return
    by_lower = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        source = by_lower.get(str(candidate).strip().lower())
        if source is not None:
            df[target] = df[source]
            return


def _normalize_lat_lon_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()
    _copy_first_present(out, "lat", ["latitude", "Lat", "Latitude", "LAT", "lat_deg", "y"])
    _copy_first_present(out, "lon", ["longitude", "lng", "Lng", "Longitude", "LON", "lon_deg", "long", "x"])
    for col in ["lat", "lon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prepare_drive_measurements(drive_df: pd.DataFrame) -> pd.DataFrame:
    dt = _normalize_lat_lon_columns(drive_df)
    if not {"lat", "lon"}.issubset(dt.columns):
        print(f"[LTE][DRIVE_MEASUREMENTS] skipped=True reason=missing_lat_lon columns={list(dt.columns)}")
        return pd.DataFrame()
    dt = dt.dropna(subset=["lat", "lon"]).copy()
    if dt.empty:
        return dt
    rcol = next((c for c in dt.columns if "rsrp" in c.lower()), None)
    qcol = next((c for c in dt.columns if "rsrq" in c.lower()), None)
    scol = next((c for c in dt.columns if "sinr" in c.lower()), None)
    if rcol is None:
        raise ValueError("Drive-test data is missing an RSRP column")
    dt["RSRP_meas"] = pd.to_numeric(dt[rcol], errors="coerce")
    if qcol:
        dt["RSRQ_meas"] = pd.to_numeric(dt[qcol], errors="coerce")
    if scol:
        dt["SINR_meas"] = pd.to_numeric(dt[scol], errors="coerce")
    return dt.dropna(subset=["RSRP_meas"]).copy()


def _attach_prediction_grid_to_points(points_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    points = points_df.copy()
    if points.empty or pred_df.empty or not {"lat", "lon"}.issubset(pred_df.columns):
        print(
            f"[LTE][DT_EVAL_GRID_ATTACH] skipped=True reason=missing_prediction_lat_lon "
            f"pred_cols={list(pred_df.columns)}"
        )
        return points
    keep_cols = [
        "lat",
        "lon",
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "pred_rsrp_geo",
        "pred_rsrq_geo",
        "pred_sinr_geo",
        "pred_rsrp_demo",
        "pred_rsrq_demo",
        "pred_sinr_demo",
        "morphology_cluster",
        "grid_id",
        "clutter_class",
    ]
    pred_keep_cols = [col for col in keep_cols if col in pred_df.columns]
    if not {"lat", "lon"}.issubset(pred_keep_cols):
        print(
            f"[LTE][DT_EVAL_GRID_ATTACH] skipped=True reason=missing_keep_lat_lon "
            f"pred_keep_cols={pred_keep_cols}"
        )
        return points
    preds = pred_df[pred_keep_cols].dropna(subset=["lat", "lon"]).copy()
    if points.empty or preds.empty:
        return points

    points_gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
    preds_gdf = gpd.GeoDataFrame(preds, geometry=gpd.points_from_xy(preds["lon"], preds["lat"]), crs="EPSG:4326")
    preds_gdf = preds_gdf.rename(columns={"lat": "grid_lat", "lon": "grid_lon"})
    utm_crs = _choose_utm_crs(preds_gdf)
    joined = gpd.sjoin_nearest(
        points_gdf.to_crs(utm_crs),
        preds_gdf.to_crs(utm_crs),
        how="left",
        distance_col="grid_match_distance_m",
    )
    joined = joined.to_crs("EPSG:4326")
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    return pd.DataFrame(joined)


def evaluate_geo_against_dt(
    drive_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    holdout = _prepare_drive_measurements(drive_df)
    holdout = _attach_prediction_grid_to_points(holdout, pred_df)

    baseline_metrics: Dict[str, Dict[str, float]] = {}
    for meas_col, base_col in [("RSRP_meas", "pred_rsrp"), ("RSRQ_meas", "pred_rsrq"), ("SINR_meas", "pred_sinr")]:
        if meas_col in holdout.columns and base_col in holdout.columns:
            valid = holdout.dropna(subset=[meas_col, base_col])
            if not valid.empty:
                baseline_metrics[meas_col] = _metric_bundle(valid[meas_col], valid[base_col], metric_key=meas_col)

    geo_metrics: Dict[str, Dict[str, float]] = {}
    for meas_col, geo_col in [("RSRP_meas", "pred_rsrp_geo"), ("RSRQ_meas", "pred_rsrq_geo"), ("SINR_meas", "pred_sinr_geo")]:
        if meas_col in holdout.columns and geo_col in holdout.columns:
            valid = holdout.dropna(subset=[meas_col, geo_col])
            if not valid.empty:
                geo_metrics[meas_col] = _metric_bundle(valid[meas_col], valid[geo_col], metric_key=meas_col)

    return holdout, baseline_metrics, geo_metrics


def split_drive_train_holdout(
    drive_df: pd.DataFrame,
    validation_fraction: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    dt = _prepare_drive_measurements(drive_df)
    summary: Dict[str, object] = {
        "enabled": False,
        "strategy": "deterministic_row_hash",
        "validation_fraction": float(validation_fraction),
        "dt_rows": int(len(dt)),
        "train_rows": 0,
        "holdout_rows": 0,
    }
    if dt.empty:
        summary["reason"] = "empty_drive_measurements"
        return dt.copy(), dt.copy(), summary

    holdout_frac = float(np.clip(validation_fraction, 0.1, 0.5))
    if len(dt) < 40:
        holdout_size = max(1, int(round(len(dt) * holdout_frac)))
        holdout_df = dt.iloc[:holdout_size].copy()
        train_df = dt.iloc[holdout_size:].copy()
    else:
        split_key = pd.DataFrame(
            {
                "session_id": dt["session_id"] if "session_id" in dt.columns else pd.Series(0, index=dt.index),
                "lat_5dp": pd.to_numeric(dt["lat"], errors="coerce").round(5),
                "lon_5dp": pd.to_numeric(dt["lon"], errors="coerce").round(5),
                "node_cell_id": dt["Node_Cell_ID"] if "Node_Cell_ID" in dt.columns else pd.Series("", index=dt.index),
            },
            index=dt.index,
        )
        hashed = pd.util.hash_pandas_object(split_key, index=False).astype("uint64")
        dt = dt.assign(_split_rand=(hashed % 10_000) / 10_000.0)
        holdout_mask = dt["_split_rand"] < holdout_frac
        if holdout_mask.sum() < max(30, int(0.1 * len(dt))):
            cutoff = np.quantile(dt["_split_rand"], holdout_frac)
            holdout_mask = dt["_split_rand"] <= cutoff
        if (~holdout_mask).sum() < max(30, int(0.2 * len(dt))):
            order = dt["_split_rand"].sort_values().index
            holdout_count = max(30, min(len(dt) - 30, int(round(len(dt) * holdout_frac))))
            holdout_mask = pd.Series(False, index=dt.index)
            holdout_mask.loc[order[:holdout_count]] = True
        train_df = dt.loc[~holdout_mask].drop(columns=["_split_rand"], errors="ignore").copy()
        holdout_df = dt.loc[holdout_mask].drop(columns=["_split_rand"], errors="ignore").copy()

    summary.update(
        {
            "enabled": not train_df.empty and not holdout_df.empty,
            "validation_fraction": holdout_frac,
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
        }
    )
    if not summary["enabled"]:
        summary["reason"] = "insufficient_train_or_holdout_rows"
    return train_df, holdout_df, summary


def _build_dt_calibration_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "pred_rsrp_geo",
        "pred_rsrq_geo",
        "pred_sinr_geo",
        "morphology_cluster",
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "los_blocker_count",
        "los_blocked_ratio",
        "max_blocker_height_m",
        "diffraction_proxy_db",
        "nlos_flag",
        "terrain_elevation_m",
        "terrain_slope_deg",
        "terrain_relief_to_site_m",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "azimuth_delta_deg",
        "serving_proxy_rsrp_phys_dbm",
        "rsrq_proxy_db",
        "sinr_proxy_db",
        "interference_gap_db",
        "interference_ratio_linear",
    ]
    numeric = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        if col in df.columns:
            numeric[col] = pd.to_numeric(df[col], errors="coerce")

    categorical = pd.DataFrame(index=df.index)
    if "clutter_class" in df.columns:
        categorical = pd.get_dummies(df["clutter_class"].fillna("unknown").astype(str), prefix="clutter")

    features = pd.concat([numeric, categorical], axis=1)
    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def fit_dt_holdout_calibration(
    train_eval: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    model_specs = {
        "RSRP": ("RSRP_meas", "pred_rsrp_geo", -12.0, 12.0),
        "RSRQ": ("RSRQ_meas", "pred_rsrq_geo", -8.0, 8.0),
        "SINR": ("SINR_meas", "pred_sinr_geo", -10.0, 10.0),
    }
    model_bundle: Dict[str, Dict[str, object]] = {}
    debug: Dict[str, object] = {
        "enabled": False,
        "train_rows": int(len(train_eval)),
        "models": {},
    }
    if train_eval.empty:
        debug["reason"] = "empty_train_eval"
        return model_bundle, debug

    for metric_name, (meas_col, pred_col, low_clip, high_clip) in model_specs.items():
        valid = train_eval.dropna(subset=[meas_col, pred_col]).copy()
        if len(valid) < 60:
            debug["models"][metric_name] = {
                "used": False,
                "reason": "train_rows_lt_60",
                "rows": int(len(valid)),
            }
            continue

        features = _build_dt_calibration_feature_frame(valid)
        if features.empty or features.shape[1] == 0:
            debug["models"][metric_name] = {
                "used": False,
                "reason": "no_features",
                "rows": int(len(valid)),
            }
            continue

        residual = (
            pd.to_numeric(valid[meas_col], errors="coerce")
            - pd.to_numeric(valid[pred_col], errors="coerce")
        ).clip(lower=low_clip, upper=high_clip)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(features)
        ridge = Ridge(alpha=3.0, random_state=42)
        ridge.fit(x_scaled, residual.to_numpy(dtype=float))

        model_bundle[metric_name] = {
            "metric_name": metric_name,
            "pred_col": pred_col,
            "scaler": scaler,
            "model": ridge,
            "feature_columns": features.columns.tolist(),
            "low_clip": float(low_clip),
            "high_clip": float(high_clip),
        }
        debug["models"][metric_name] = {
            "used": True,
            "rows": int(len(valid)),
            "feature_count": int(features.shape[1]),
            "residual_mae": round(float(np.abs(residual).mean()), 4),
            "residual_bias": round(float(residual.mean()), 4),
        }

    debug["enabled"] = bool(model_bundle)
    if not model_bundle:
        debug["reason"] = "no_metric_models_fit"
    return model_bundle, debug


def apply_dt_holdout_calibration(
    pred_df: pd.DataFrame,
    model_bundle: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    if pred_df.empty or not model_bundle:
        return pred_df

    out = pred_df.copy()
    metric_ranges = {
        "RSRP": (-140.0, -44.0),
        "RSRQ": (-20.0, -3.0),
        "SINR": (-10.0, 30.0),
    }
    for metric_name, bundle in model_bundle.items():
        pred_col = str(bundle["pred_col"])
        if pred_col not in out.columns:
            continue
        features = _build_dt_calibration_feature_frame(out)
        feature_columns = list(bundle["feature_columns"])
        for col in feature_columns:
            if col not in features.columns:
                features[col] = 0.0
        features = features.reindex(columns=feature_columns, fill_value=0.0)
        x_scaled = bundle["scaler"].transform(features)
        residual_pred = pd.Series(bundle["model"].predict(x_scaled), index=out.index, dtype=float)
        residual_pred = residual_pred.clip(lower=float(bundle["low_clip"]), upper=float(bundle["high_clip"]))
        clip_min, clip_max = metric_ranges[metric_name]
        base_values = pd.to_numeric(out[pred_col], errors="coerce")
        out[pred_col] = (base_values + residual_pred).clip(lower=clip_min, upper=clip_max)
    return out


def preserve_calibrated_kpis(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    calibrated_specs = [
        ("pred_rsrp_calibrated", "pred_rsrp_geo", "pred_rsrp", -140.0, -44.0),
        ("pred_rsrq_calibrated", "pred_rsrq_geo", "pred_rsrq", -20.0, -3.0),
        ("pred_sinr_calibrated", "pred_sinr_geo", "pred_sinr", -10.0, 30.0),
    ]
    for out_col, geo_col, raw_col, clip_min, clip_max in calibrated_specs:
        source_col = geo_col if geo_col in out.columns else raw_col
        if source_col in out.columns:
            out[out_col] = pd.to_numeric(out[source_col], errors="coerce").clip(clip_min, clip_max)
    return out


def apply_demo_dt_overlay(
    pred_df: pd.DataFrame,
    drive_df: pd.DataFrame,
    replace_radius_m: float = 20.0,
    blend_sigma_m: float = 60.0,
    blend_radius_m: float = 140.0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    pred_out = pred_df.copy()
    dt = _prepare_drive_measurements(drive_df)
    if pred_out.empty or dt.empty or not {"lat", "lon"}.issubset(pred_out.columns):
        return pred_out, {"enabled": False, "pred_rows": len(pred_out), "dt_rows": len(dt)}

    pred_points = pred_out.dropna(subset=["lat", "lon"]).copy()
    if pred_points.empty:
        return pred_out, {"enabled": False, "pred_rows": len(pred_out), "dt_rows": len(dt)}

    pred_gdf = gpd.GeoDataFrame(pred_points, geometry=gpd.points_from_xy(pred_points["lon"], pred_points["lat"]), crs="EPSG:4326")
    dt_gdf = gpd.GeoDataFrame(dt, geometry=gpd.points_from_xy(dt["lon"], dt["lat"]), crs="EPSG:4326")
    utm_crs = _choose_utm_crs(pred_gdf)
    pred_utm = pred_gdf.to_crs(utm_crs)
    dt_utm = dt_gdf.to_crs(utm_crs)
    pred_coords = np.c_[pred_utm.geometry.x.to_numpy(dtype=float), pred_utm.geometry.y.to_numpy(dtype=float)]
    dt_coords = np.c_[dt_utm.geometry.x.to_numpy(dtype=float), dt_utm.geometry.y.to_numpy(dtype=float)]
    if len(pred_coords) == 0 or len(dt_coords) == 0:
        return pred_out, {"enabled": False, "pred_rows": len(pred_out), "dt_rows": len(dt)}

    dt_tree = BallTree(dt_coords, metric="euclidean")
    pred_tree = BallTree(pred_coords, metric="euclidean")
    nearest_dt_dist, nearest_dt_idx = dt_tree.query(pred_coords, k=1)
    nearest_pred_dist, nearest_pred_idx = pred_tree.query(dt_coords, k=1)
    nearest_dt_dist = nearest_dt_dist[:, 0]
    nearest_dt_idx = nearest_dt_idx[:, 0]
    nearest_pred_dist = nearest_pred_dist[:, 0]
    nearest_pred_idx = nearest_pred_idx[:, 0]

    pred_out["demo_dt_distance_m"] = np.nan
    pred_out.loc[pred_points.index, "demo_dt_distance_m"] = nearest_dt_dist
    pred_out["demo_blend_weight"] = 0.0
    pred_out["demo_dt_anchor"] = False

    kpi_specs = [
        ("RSRP_meas", "pred_rsrp_geo" if "pred_rsrp_geo" in pred_out.columns else "pred_rsrp", "pred_rsrp_demo", -140.0, -44.0),
        ("RSRQ_meas", "pred_rsrq_geo" if "pred_rsrq_geo" in pred_out.columns else "pred_rsrq", "pred_rsrq_demo", -20.0, -3.0),
        ("SINR_meas", "pred_sinr_geo" if "pred_sinr_geo" in pred_out.columns else "pred_sinr", "pred_sinr_demo", -10.0, 30.0),
    ]

    blend_weight = np.exp(-0.5 * np.square(nearest_dt_dist / max(blend_sigma_m, 1.0)))
    blend_weight = np.where(nearest_dt_dist <= blend_radius_m, blend_weight, 0.0)
    blend_weight = np.clip(blend_weight, 0.0, 1.0)
    anchor_mask = nearest_dt_dist <= replace_radius_m
    blend_weight = np.where(anchor_mask, 1.0, blend_weight)
    pred_out.loc[pred_points.index, "demo_blend_weight"] = blend_weight

    anchor_hits = pd.Series(False, index=pred_points.index)
    for dt_row_pos, pred_row_pos in enumerate(nearest_pred_idx):
        if nearest_pred_dist[dt_row_pos] <= replace_radius_m:
            anchor_hits.iloc[int(pred_row_pos)] = True

    for meas_col, base_col, out_col, clip_min, clip_max in kpi_specs:
        if base_col not in pred_out.columns:
            continue
        pred_out[out_col] = pd.to_numeric(pred_out[base_col], errors="coerce")
        if meas_col not in dt.columns:
            continue
        dt_meas = pd.to_numeric(dt[meas_col], errors="coerce").to_numpy(dtype=float)
        base_vals = pd.to_numeric(pred_points[base_col], errors="coerce").to_numpy(dtype=float)
        nearest_vals = dt_meas[nearest_dt_idx]
        blended_vals = ((1.0 - blend_weight) * base_vals) + (blend_weight * nearest_vals)
        pred_out.loc[pred_points.index, out_col] = np.clip(blended_vals, clip_min, clip_max)

        anchored_indices = []
        anchored_values = []
        for dt_row_pos, pred_row_pos in enumerate(nearest_pred_idx):
            dt_value = dt_meas[dt_row_pos]
            if np.isnan(dt_value) or nearest_pred_dist[dt_row_pos] > replace_radius_m:
                continue
            anchored_indices.append(int(pred_points.index[int(pred_row_pos)]))
            anchored_values.append(float(np.clip(dt_value, clip_min, clip_max)))
        if anchored_indices:
            pred_out.loc[anchored_indices, out_col] = anchored_values

    pred_out.loc[pred_points.index, "demo_dt_anchor"] = anchor_hits.to_numpy(dtype=bool)
    pred_out["demo_visual_source"] = np.where(
        pred_out["demo_dt_anchor"],
        "dt_anchor",
        np.where(pred_out["demo_blend_weight"] > 0.0, "dt_blend", "prediction_only"),
    )
    return pred_out, {
        "enabled": True,
        "replace_radius_m": float(replace_radius_m),
        "blend_sigma_m": float(blend_sigma_m),
        "blend_radius_m": float(blend_radius_m),
        "pred_rows": int(len(pred_out)),
        "dt_rows": int(len(dt)),
        "anchor_cells": int(pred_out["demo_dt_anchor"].sum()),
        "blended_cells": int((pred_out["demo_blend_weight"] > 0.0).sum()),
    }


def geo_offset_from_features(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
    work = df.copy()
    weights = dict(DEFAULT_GEO_WEIGHTS if weights is None else weights)

    clutter_penalty = pd.Series(0.0, index=work.index)
    if "clutter_class" in work.columns:
        clutter_penalty = work["clutter_class"].astype(str).map(
            {
                "Dense Urban": weights["clutter_Dense Urban"],
                "Urban": weights["clutter_Urban"],
                "Suburban": weights["clutter_Suburban"],
                "Vegetation": weights["clutter_Vegetation"],
                "Water": weights["clutter_Water"],
                "Rural/Open": weights["clutter_Rural/Open"],
            }
        ).fillna(0.0)

    cluster_center = float(_safe_numeric(work, "morphology_cluster").mean()) if len(work) else 0.0
    geo_offset = clutter_penalty + ((_safe_numeric(work, "morphology_cluster") - cluster_center) * float(weights["morphology_cluster"]))
    geo_offset = geo_offset + (_safe_numeric(work, "building_area_ratio").clip(0, 0.8) * float(weights["building_area_ratio"]))
    geo_offset = geo_offset + (_safe_numeric(work, "building_count").clip(0, 30) * float(weights["building_count"]))
    geo_offset = geo_offset + (_safe_numeric(work, "road_length_m").clip(0, 400) * float(weights["road_length_m"]))
    geo_offset = geo_offset + (_safe_numeric(work, "green_ratio").clip(0, 1.0) * float(weights["green_ratio"]))
    geo_offset = geo_offset + (_safe_numeric(work, "water_ratio").clip(0, 1.0) * float(weights["water_ratio"]))
    geo_offset = geo_offset + (_safe_numeric(work, "avg_building_area_m2").clip(0, 3000) * float(weights["avg_building_area_m2"]))
    geo_offset = geo_offset + (_safe_numeric(work, "site_count_250m").clip(0, 12) * float(weights["site_count_250m"]))
    geo_offset = geo_offset + (_safe_numeric(work, "site_count_500m").clip(0, 25) * float(weights["site_count_500m"]))
    geo_offset = geo_offset + (_safe_numeric(work, "serving_distance_m").clip(0, 1200) * float(weights["serving_distance_m"]))
    geo_offset = geo_offset + (_safe_numeric(work, "nearest_site_distance_m").clip(0, 1000) * float(weights["nearest_site_distance_m"]))
    geo_offset = geo_offset + (((_safe_numeric(work, "azimuth_delta_deg").clip(0, 180) / 10.0) ** 1.2) * float(weights["azimuth_delta_deg"]))
    geo_offset = geo_offset + (_safe_numeric(work, "mean_nearest3_site_distance_m").clip(0, 1500) * float(weights["mean_nearest3_site_distance_m"]))

    clutter_series = work.get("clutter_class", pd.Series("", index=work.index)).astype(str)
    nearest_site = _safe_numeric(work, "nearest_site_distance_m")
    serving_distance = _safe_numeric(work, "serving_distance_m")
    azimuth_delta = _safe_numeric(work, "azimuth_delta_deg")
    avg_building_area = _safe_numeric(work, "avg_building_area_m2")
    green_ratio = _safe_numeric(work, "green_ratio")
    site_count_250m = _safe_numeric(work, "site_count_250m")
    morphology_cluster = _safe_numeric(work, "morphology_cluster")

    dense_urban_far_penalty = np.where(
        (clutter_series == "Dense Urban") & (nearest_site > 180.0),
        float(weights["dense_urban_far_base"]) + (float(weights["dense_urban_far_slope"]) * (nearest_site.clip(180.0, 700.0) - 180.0)),
        0.0,
    )
    urban_off_axis_penalty = np.where(
        azimuth_delta > 45.0,
        float(weights["urban_off_axis_slope"]) * (azimuth_delta.clip(45.0, 180.0) - 45.0),
        0.0,
    )
    far_serving_off_axis_penalty = np.where(
        (serving_distance > 250.0) & (azimuth_delta > 35.0),
        float(weights["far_serving_off_axis_base"])
        + (float(weights["far_serving_distance_slope"]) * (serving_distance.clip(250.0, 1200.0) - 250.0))
        + (float(weights["far_serving_azimuth_slope"]) * (azimuth_delta.clip(35.0, 180.0) - 35.0)),
        0.0,
    )
    high_building_far_penalty = np.where(
        (avg_building_area > 250.0) & (nearest_site > 160.0),
        float(weights["high_building_far_base"])
        + (float(weights["high_building_area_slope"]) * (avg_building_area.clip(250.0, 3000.0) - 250.0))
        + (float(weights["high_building_distance_slope"]) * (nearest_site.clip(160.0, 1000.0) - 160.0)),
        0.0,
    )
    vegetation_far_penalty = np.where(
        (clutter_series == "Vegetation") & (serving_distance > 220.0),
        float(weights["vegetation_far_base"]) + (float(weights["vegetation_green_slope"]) * green_ratio.clip(0.2, 1.0)),
        0.0,
    )
    water_open_bonus = np.where(
        clutter_series.isin(["Water", "Rural/Open"]) & (azimuth_delta < 20.0) & (nearest_site < 220.0),
        float(weights["water_open_base"]) + (float(weights["water_open_distance_slope"]) * (220.0 - nearest_site.clip(0.0, 220.0))),
        0.0,
    )
    dense_site_bonus = np.where(
        (site_count_250m >= 4.0) & (nearest_site < 120.0),
        float(weights["dense_site_base"]) + (float(weights["dense_site_count_slope"]) * site_count_250m.clip(4.0, 12.0)),
        0.0,
    )
    cluster_dense_urban_penalty = np.where(
        (morphology_cluster >= (cluster_center + 1.0)) & (clutter_series == "Dense Urban"),
        float(weights["cluster_dense_urban_base"])
        + (float(weights["cluster_dense_urban_slope"]) * (morphology_cluster - cluster_center).clip(lower=0.0, upper=4.0)),
        0.0,
    )
    nlos_penalty = float(weights["nlos_flag"]) * _safe_numeric(work, "nlos_flag").clip(0, 1)
    blocker_penalty = (
        float(weights["los_blocker_count"]) * _safe_numeric(work, "los_blocker_count").clip(0, 10)
        + float(weights["los_blocked_ratio"]) * _safe_numeric(work, "los_blocked_ratio").clip(0, 1.0)
        + float(weights["max_blocker_height_m"]) * _safe_numeric(work, "max_blocker_height_m").clip(0, 80.0)
    )
    diffraction_penalty = float(weights["diffraction_proxy_db"]) * _safe_numeric(work, "diffraction_proxy_db").clip(0, 25.0)
    terrain_penalty = (
        float(weights["terrain_slope_deg"]) * _safe_numeric(work, "terrain_slope_deg").clip(0, 35.0)
        + float(weights["terrain_relief_to_site_m"]) * _safe_numeric(work, "terrain_relief_to_site_m").clip(lower=0.0, upper=180.0)
    )
    interference_gap = _safe_numeric(work, "interference_gap_db")
    interference_penalty = np.where(
        interference_gap < 6.0,
        float(weights["interference_gap_penalty_slope"]) * (6.0 - interference_gap.clip(-15.0, 6.0)),
        float(weights["interference_gap_bonus_slope"]) * (interference_gap.clip(6.0, 18.0) - 6.0),
    )
    interference_ratio_penalty = float(weights["interference_ratio_linear"]) * _safe_numeric(work, "interference_ratio_linear").clip(0.0, 2.5)

    return (
        geo_offset
        + pd.Series(dense_urban_far_penalty, index=work.index, dtype=float)
        + pd.Series(urban_off_axis_penalty, index=work.index, dtype=float)
        + pd.Series(far_serving_off_axis_penalty, index=work.index, dtype=float)
        + pd.Series(high_building_far_penalty, index=work.index, dtype=float)
        + pd.Series(vegetation_far_penalty, index=work.index, dtype=float)
        + pd.Series(water_open_bonus, index=work.index, dtype=float)
        + pd.Series(dense_site_bonus, index=work.index, dtype=float)
        + pd.Series(cluster_dense_urban_penalty, index=work.index, dtype=float)
        + nlos_penalty
        + blocker_penalty
        + diffraction_penalty
        + terrain_penalty
        + pd.Series(interference_penalty, index=work.index, dtype=float)
        + interference_ratio_penalty
    )


def apply_experimental_geo_adjustments(pred_df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    pred_out = pred_df.copy()
    weights = dict(DEFAULT_GEO_WEIGHTS if weights is None else weights)
    geo_offset = geo_offset_from_features(pred_out, weights=weights)
    rsrp_base = pd.to_numeric(pred_out["pred_rsrp"], errors="coerce")
    rsrq_base = pd.to_numeric(pred_out["pred_rsrq"], errors="coerce")
    sinr_base = pd.to_numeric(pred_out["pred_sinr"], errors="coerce")
    rsrp_phys = pd.to_numeric(pred_out.get("serving_proxy_rsrp_phys_dbm"), errors="coerce")
    rsrq_phys = pd.to_numeric(pred_out.get("rsrq_proxy_db"), errors="coerce")
    sinr_phys = pd.to_numeric(pred_out.get("sinr_proxy_db"), errors="coerce")

    pred_out["pred_rsrp_geo"] = rsrp_base.astype(float).copy()
    has_rsrp_phys = rsrp_phys.notna()
    rsrp_base_weight = max(0.0, 1.0 - float(weights["rsrp_phys_weight"]))
    pred_out.loc[has_rsrp_phys, "pred_rsrp_geo"] = (
        (rsrp_base_weight * rsrp_base[has_rsrp_phys])
        + (float(weights["rsrp_phys_weight"]) * rsrp_phys[has_rsrp_phys])
        + (float(weights["rsrp_geo_weight"]) * geo_offset[has_rsrp_phys])
    )
    pred_out.loc[~has_rsrp_phys, "pred_rsrp_geo"] = rsrp_base[~has_rsrp_phys] + geo_offset[~has_rsrp_phys]

    pred_out["pred_rsrq_geo"] = rsrq_base.astype(float).copy()
    has_rsrq_phys = rsrq_phys.notna()
    rsrq_base_weight = max(0.0, 1.0 - float(weights["rsrq_phys_weight"]))
    pred_out.loc[has_rsrq_phys, "pred_rsrq_geo"] = (
        (rsrq_base_weight * rsrq_base[has_rsrq_phys])
        + (float(weights["rsrq_phys_weight"]) * rsrq_phys[has_rsrq_phys])
        + (float(weights["rsrq_geo_weight"]) * geo_offset[has_rsrq_phys])
    )
    pred_out.loc[~has_rsrq_phys, "pred_rsrq_geo"] = rsrq_base[~has_rsrq_phys] + (
        geo_offset[~has_rsrq_phys] * float(weights["rsrq_geo_fallback_weight"])
    )

    pred_out["pred_sinr_geo"] = sinr_base.astype(float).copy()
    has_sinr_phys = sinr_phys.notna()
    sinr_base_weight = max(0.0, 1.0 - float(weights["sinr_phys_weight"]))
    pred_out.loc[has_sinr_phys, "pred_sinr_geo"] = (
        (sinr_base_weight * sinr_base[has_sinr_phys])
        + (float(weights["sinr_phys_weight"]) * sinr_phys[has_sinr_phys])
        + (float(weights["sinr_geo_weight"]) * geo_offset[has_sinr_phys])
    )
    pred_out.loc[~has_sinr_phys, "pred_sinr_geo"] = sinr_base[~has_sinr_phys] + (
        geo_offset[~has_sinr_phys] * float(weights["sinr_geo_fallback_weight"])
    )

    pred_out["pred_rsrp_geo"] = pred_out["pred_rsrp_geo"].clip(-140, -44)
    pred_out["pred_rsrq_geo"] = pred_out["pred_rsrq_geo"].clip(-20, -3)
    pred_out["pred_sinr_geo"] = pred_out["pred_sinr_geo"].clip(-10, 30)

    return pred_out, {"mode": "weighted_geo_adjustment", "weights": weights}


def apply_full_display_correction(
    pred_df: pd.DataFrame,
    drive_df: pd.DataFrame,
    site_df: pd.DataFrame,
    building_df: pd.DataFrame,
    polygon_geoms: Iterable,
    params: Optional[Dict[str, object]] = None,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    params = params or {}
    correction_started_at = time.perf_counter()
    if pred_df.empty:
        return pred_df.copy(), {"enabled": False, "reason": "empty_prediction"}

    print(
        f"[LTE][GEO_START] pred_rows={len(pred_df)} drive_rows={len(drive_df)} "
        f"site_rows={len(site_df)} building_rows={len(building_df)} "
        f"osm_enabled={_osm_enrichment_enabled(params)}",
        flush=True,
    )
    site_norm = normalize_site_for_geo(site_df)
    weights, weights_summary = load_geo_weights(
        project_id=params.get("project_id"),
        weights_path=params.get("optimizer_weights_path"),
    )
    _log_geo_stage("normalize_site_and_weights", correction_started_at, site_rows=len(site_norm))
    polygon_list = list(polygon_geoms)
    if polygon_list:
        polygon_gdf = gpd.GeoDataFrame({"geometry": polygon_list}, crs="EPSG:4326")
        polygon_gdf, polygon_alignment = align_project_polygon_to_points(polygon_gdf, site_norm)
    else:
        polygon_gdf = _fallback_polygon_from_points(pred_df if not pred_df.empty else site_norm)
        polygon_alignment = "fallback_from_points"
    _log_geo_stage("polygon_alignment", correction_started_at, polygon_rows=len(polygon_gdf), alignment=polygon_alignment)

    osm_enabled = _osm_enrichment_enabled(params)
    building_gdf = building_df_to_gdf(building_df)
    building_gdf, building_alignment = align_building_geometries_to_project(building_gdf, polygon_gdf)
    _log_geo_stage("building_alignment", correction_started_at, building_rows=len(building_gdf), alignment=building_alignment)
    building_gdf = _enrich_buildings_with_osm_heights(building_gdf, polygon_gdf, enabled=osm_enabled)
    _log_geo_stage("building_height_enrichment", correction_started_at, building_rows=len(building_gdf), osm_enabled=osm_enabled)

    tile_size_m = float(params.get("tile_size_m") or max(float(params.get("grid", 25.0)), DEFAULT_TILE_SIZE_M))
    cluster_count = int(params.get("cluster_count", 5))
    dem_raster_path = params.get("dem_raster_path")
    if polygon_gdf.empty:
        pred_work = pred_df.copy()
        pred_work = attach_site_context_features(pred_work, site_norm)
        _log_geo_stage("fallback_site_context", correction_started_at, pred_rows=len(pred_work))
        pred_work = _attach_building_path_features(pred_work, building_gdf)
        _log_geo_stage("fallback_building_path", correction_started_at, pred_rows=len(pred_work))
        pred_work, dem_status = _attach_dem_features(pred_work, dem_raster_path)
        _log_geo_stage("fallback_dem", correction_started_at, sampled=dem_status.get("sampled"), reason=dem_status.get("reason"))
        pred_work = _refine_experimental_forward_features(pred_work)
        pred_work = attach_fixed_serving_sinr_rsrq_proxy(pred_work, site_norm)
        pred_work, geo_summary = apply_experimental_geo_adjustments(pred_work, weights=weights)
        _log_geo_stage("fallback_geo_adjustment", correction_started_at, pred_rows=len(pred_work))
        drive_train_df, drive_holdout_df, split_summary = split_drive_train_holdout(
            drive_df,
            validation_fraction=float(params.get("dt_validation_fraction", 0.25)),
        )
        train_eval, _, train_geo_metrics = evaluate_geo_against_dt(drive_train_df, pred_work)
        dt_calibration_models, dt_calibration_debug = fit_dt_holdout_calibration(train_eval)
        pred_work = apply_dt_holdout_calibration(pred_work, dt_calibration_models)
        pred_work = preserve_calibrated_kpis(pred_work)
        _, baseline_metrics, geo_metrics = evaluate_geo_against_dt(drive_holdout_df, pred_work)
        _log_geo_stage("fallback_dt_calibration", correction_started_at, train_rows=len(drive_train_df), holdout_rows=len(drive_holdout_df))
        pred_work, demo_summary = apply_demo_dt_overlay(
            pred_work,
            drive_df,
            replace_radius_m=float(params.get("dt_replace_radius_m", 20.0)),
            blend_sigma_m=float(params.get("dt_blend_sigma_m", 60.0)),
            blend_radius_m=float(params.get("dt_blend_radius_m", 140.0)),
        )
        pred_work["pred_rsrp"] = pd.to_numeric(pred_work.get("pred_rsrp_demo", pred_work.get("pred_rsrp_geo", pred_work["pred_rsrp"])), errors="coerce").clip(-140, -44)
        pred_work["pred_rsrq"] = pd.to_numeric(pred_work.get("pred_rsrq_demo", pred_work.get("pred_rsrq_geo", pred_work["pred_rsrq"])), errors="coerce").clip(-20, -3)
        pred_work["pred_sinr"] = pd.to_numeric(pred_work.get("pred_sinr_demo", pred_work.get("pred_sinr_geo", pred_work["pred_sinr"])), errors="coerce").clip(-10, 30)
        _log_geo_stage("fallback_done", correction_started_at, pred_rows=len(pred_work))
        return pred_work, {
            "enabled": True,
            "polygon_alignment": polygon_alignment,
            "building_alignment": building_alignment,
            "weights_summary": weights_summary,
            "geo_summary": geo_summary,
            "geo_status": {"dem_status": dem_status},
            "baseline_validation_metrics": baseline_metrics,
            "geo_validation_metrics": geo_metrics,
            "train_geo_metrics": train_geo_metrics,
            "dt_calibration_summary": {
                "split": split_summary,
                "calibration": dt_calibration_debug,
            },
            "demo_summary": demo_summary,
            "grid_rows": 0,
            "building_rows": int(len(building_gdf)),
        }

    grid_gdf = create_analysis_grid(polygon_gdf, tile_size_m)
    _log_geo_stage("create_analysis_grid", correction_started_at, grid_rows=len(grid_gdf), tile_size_m=tile_size_m)
    grid_gdf = attach_building_features(grid_gdf, building_gdf)
    _log_geo_stage("attach_building_features", correction_started_at, grid_rows=len(grid_gdf))
    grid_gdf = _attach_osm_context_features(grid_gdf, polygon_gdf, enabled=osm_enabled)
    _log_geo_stage("attach_osm_context", correction_started_at, grid_rows=len(grid_gdf), osm_enabled=osm_enabled)
    grid_df, _ = build_grid_feature_frame(grid_gdf, site_norm, cluster_count)
    _log_geo_stage("build_grid_feature_frame", correction_started_at, grid_rows=len(grid_df))
    grid_df, geo_status = augment_grid_with_advanced_geo_features(grid_df, building_gdf, site_norm, dem_raster_path=dem_raster_path)
    _log_geo_stage("advanced_geo_features", correction_started_at, grid_rows=len(grid_df))
    grid_gdf = grid_gdf.merge(grid_df[["grid_id", "clutter_class", "morphology_cluster"]], on="grid_id", how="left")

    pred_work = pred_df.copy()
    pred_work = assign_points_to_tiles(pred_work, grid_gdf)
    _log_geo_stage("assign_points_to_tiles", correction_started_at, pred_rows=len(pred_work))
    pred_work = _attach_missing_grid_features_by_grid_id(pred_work, grid_df)
    if "Node_Cell_ID" not in pred_work.columns and "node_cell_id" in pred_work.columns:
        pred_work["Node_Cell_ID"] = pred_work["node_cell_id"].astype(str)
    pred_work = attach_fixed_serving_sinr_rsrq_proxy(pred_work, site_norm)
    _log_geo_stage("serving_sinr_rsrq_proxy", correction_started_at, pred_rows=len(pred_work))
    pred_work, geo_summary = apply_experimental_geo_adjustments(pred_work, weights=weights)
    drive_train_df, drive_holdout_df, split_summary = split_drive_train_holdout(
        drive_df,
        validation_fraction=float(params.get("dt_validation_fraction", 0.25)),
    )
    train_eval, _, train_geo_metrics = evaluate_geo_against_dt(drive_train_df, pred_work)
    dt_calibration_models, dt_calibration_debug = fit_dt_holdout_calibration(train_eval)
    pred_work = apply_dt_holdout_calibration(pred_work, dt_calibration_models)
    pred_work = preserve_calibrated_kpis(pred_work)
    _, baseline_metrics, geo_metrics = evaluate_geo_against_dt(drive_holdout_df, pred_work)
    _log_geo_stage("dt_calibration", correction_started_at, train_rows=len(drive_train_df), holdout_rows=len(drive_holdout_df))
    pred_work, demo_summary = apply_demo_dt_overlay(
        pred_work,
        drive_df,
        replace_radius_m=float(params.get("dt_replace_radius_m", 20.0)),
        blend_sigma_m=float(params.get("dt_blend_sigma_m", 60.0)),
        blend_radius_m=float(params.get("dt_blend_radius_m", 140.0)),
    )

    pred_work["pred_rsrp"] = pd.to_numeric(
        pred_work.get("pred_rsrp_demo", pred_work.get("pred_rsrp_geo", pred_work["pred_rsrp"])),
        errors="coerce",
    )
    pred_work["pred_rsrq"] = pd.to_numeric(
        pred_work.get("pred_rsrq_demo", pred_work.get("pred_rsrq_geo", pred_work["pred_rsrq"])),
        errors="coerce",
    )
    pred_work["pred_sinr"] = pd.to_numeric(
        pred_work.get("pred_sinr_demo", pred_work.get("pred_sinr_geo", pred_work["pred_sinr"])),
        errors="coerce",
    )
    pred_work["pred_rsrp"] = pred_work["pred_rsrp"].clip(-140, -44)
    pred_work["pred_rsrq"] = pred_work["pred_rsrq"].clip(-20, -3)
    pred_work["pred_sinr"] = pred_work["pred_sinr"].clip(-10, 30)
    _log_geo_stage("done", correction_started_at, pred_rows=len(pred_work))

    return pred_work, {
        "enabled": True,
        "polygon_alignment": polygon_alignment,
        "building_alignment": building_alignment,
        "weights_summary": weights_summary,
        "geo_summary": geo_summary,
        "geo_status": geo_status,
        "baseline_validation_metrics": baseline_metrics,
        "geo_validation_metrics": geo_metrics,
        "train_geo_metrics": train_geo_metrics,
        "dt_calibration_summary": {
            "split": split_summary,
            "calibration": dt_calibration_debug,
        },
        "demo_summary": demo_summary,
        "grid_rows": int(len(grid_df)),
        "building_rows": int(len(building_gdf)),
    }
