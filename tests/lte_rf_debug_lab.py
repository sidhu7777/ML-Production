from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import transform, unary_union
from shapely import wkb
from shapely.wkt import loads as load_wkt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

try:
    import rasterio
except Exception:  # pragma: no cover - optional dependency
    rasterio = None

from tools.lte_prediction import ml_engine
from tools.lte_prediction.dem_utils import ensure_project_dem
from tools.lte_prediction.Sector_wise_prediction_code_copy import (
    compute_predictions_parallel,
    load_building_polygons,
    run_prediction_from_api,
    select_nearest_site_rows,
)


DEFAULT_PROJECT_ID = 196
DEFAULT_SESSION_IDS = [4187, 4178, 4180]
DEFAULT_REGION = "india"
DEFAULT_OPERATOR = "Airtel"
DEFAULT_RADIUS_M = 500.0
DEFAULT_GRID_RESOLUTION_M = 25.0
DEFAULT_WORKERS = 3
DEFAULT_MAX_INTERFERENCE_SITES = 5
DEFAULT_TILE_SIZE_M = 50.0
DEFAULT_CLUSTER_COUNT = 5
DEFAULT_VALIDATION_FRACTION = 0.3
DEFAULT_REUSE_RUN_DIR = Path("tests/output/project_196/20260508_022650")
MAX_MAP_POINTS = 18000
DEFAULT_DEM_RASTER_PATH: Optional[Path] = None
DEFAULT_TERRAIN_API_URL = "https://api.opentopodata.org/v1/aster30m"
DEFAULT_TERRAIN_API_BATCH_SIZE = 75
DEFAULT_TERRAIN_SAMPLE_STEP_M = 30.0
DEFAULT_SINR_MIN_DISTANCE_M = 20.0
METRIC_THRESHOLDS = {
    "RSRP_meas": (3.0, 6.0, 10.0),
    "RSRQ_meas": (1.0, 2.0, 3.0),
    "SINR_meas": (2.0, 4.0, 6.0),
}

GREEN_TAGS = {
    "landuse": ["forest", "grass", "meadow", "farmland", "recreation_ground"],
    "leisure": ["park", "garden", "nature_reserve"],
    "natural": ["wood", "grassland", "scrub", "heath"],
}

WATER_TAGS = {
    "natural": ["water", "wetland"],
    "water": True,
    "waterway": True,
}

ROAD_TAGS = {"highway": True}
BUILDING_TAGS = {"building": True}


class TeeStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            try:
                stream.write(text)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(safe_text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class RunConfig:
    project_id: int = DEFAULT_PROJECT_ID
    session_ids: Tuple[int, ...] = tuple(DEFAULT_SESSION_IDS)
    region: str = DEFAULT_REGION
    operator: str = DEFAULT_OPERATOR
    radius_m: float = DEFAULT_RADIUS_M
    grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M
    workers: int = DEFAULT_WORKERS
    max_interference_sites: int = DEFAULT_MAX_INTERFERENCE_SITES
    tile_size_m: float = DEFAULT_TILE_SIZE_M
    cluster_count: int = DEFAULT_CLUSTER_COUNT
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    enable_osm: bool = False
    output_root: Path = Path("tests/output")
    reuse_run_dir: Optional[Path] = DEFAULT_REUSE_RUN_DIR
    reuse_cached_artifacts: bool = True
    dem_raster_path: Optional[Path] = DEFAULT_DEM_RASTER_PATH
    require_advanced_geo_on_miss: bool = True
    terrain_api_url: str = DEFAULT_TERRAIN_API_URL
    terrain_api_batch_size: int = DEFAULT_TERRAIN_API_BATCH_SIZE
    terrain_sample_step_m: float = DEFAULT_TERRAIN_SAMPLE_STEP_M
    sinr_min_distance_m: float = DEFAULT_SINR_MIN_DISTANCE_M


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metric_bundle(y_true: pd.Series, y_pred: pd.Series, metric_key: Optional[str] = None) -> Dict[str, float]:
    y_true_num = pd.to_numeric(y_true, errors="coerce")
    y_pred_num = pd.to_numeric(y_pred, errors="coerce")
    err = y_true_num - y_pred_num
    abs_err = err.abs()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pearson = y_true_num.corr(y_pred_num, method="pearson")
    spearman = y_true_num.corr(y_pred_num, method="spearman")
    metrics = {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(rmse, 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "bias": round(float(err.mean()), 4),
        "p50_abs_err": round(float(abs_err.quantile(0.50)), 4),
        "p90_abs_err": round(float(abs_err.quantile(0.90)), 4),
        "pearson": round(float(pearson), 4) if pd.notna(pearson) else None,
        "spearman": round(float(spearman), 4) if pd.notna(spearman) else None,
    }
    thresholds = METRIC_THRESHOLDS.get(metric_key or "")
    if thresholds:
        for threshold in thresholds:
            metrics[f"within_{str(threshold).replace('.', '_')}"] = round(float((abs_err <= threshold).mean()), 4)
    return metrics


def _resolve_dem_path_for_test(
    project_id: int,
    region: str,
    site_df: pd.DataFrame,
    requested_path: Optional[Path | str] = None,
) -> Optional[Path]:
    try:
        resolved_path = ensure_project_dem(
            project_id=int(project_id),
            region=str(region).lower(),
            site_df=site_df,
            output_path=requested_path,
            timeout_sec=60,
            force=False,
        )
        print(f"[TEST][DEM] auto_resolved=True path={resolved_path}")
        return Path(resolved_path)
    except Exception as exc:
        print(f"[TEST][DEM] auto_resolved=False reason={exc}")
        return _coerce_optional_path(requested_path)


def _choose_utm_crs(gdf_4326: gpd.GeoDataFrame) -> str:
    centroid = gdf_4326.to_crs("EPSG:4326").geometry.union_all().centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": south}).to_string()


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _safe_sample(df: pd.DataFrame, limit: int = MAX_MAP_POINTS) -> pd.DataFrame:
    if len(df) <= limit:
        return df.copy()
    step = max(1, math.ceil(len(df) / limit))
    return df.iloc[::step].copy()


KPI_VISUAL_RANGES = {
    "RSRP": {
        "unit": "dBm",
        "columns": ("RSRP_meas", "rsrp", "pred_rsrp"),
        "bins": [
            {"label": "-140 to -120", "min": -140.0, "max": -120.0, "color": "#ef2f2f"},
            {"label": "-120 to -115", "min": -120.0, "max": -115.0, "color": "#d86b2d"},
            {"label": "-115 to -100", "min": -115.0, "max": -100.0, "color": "#f29a38"},
            {"label": "-100 to -95", "min": -100.0, "max": -95.0, "color": "#f3ea4e"},
            {"label": "-95 to -90", "min": -95.0, "max": -90.0, "color": "#8bd3ee"},
            {"label": "-90 to -85", "min": -90.0, "max": -85.0, "color": "#76df58"},
            {"label": "-85 to -44", "min": -85.0, "max": -44.0, "color": "#159a28"},
        ],
    },
    "RSRQ": {
        "unit": "dB",
        "columns": ("RSRQ_meas", "rsrq", "pred_rsrq"),
        "bins": [
            {"label": "-20 to -17", "min": -20.0, "max": -17.0, "color": "#ef2f2f"},
            {"label": "-17 to -15", "min": -17.0, "max": -15.0, "color": "#d86b2d"},
            {"label": "-15 to -12", "min": -15.0, "max": -12.0, "color": "#f29a38"},
            {"label": "-12 to -10", "min": -12.0, "max": -10.0, "color": "#f3ea4e"},
            {"label": "-10 to -8", "min": -10.0, "max": -8.0, "color": "#8bd3ee"},
            {"label": "-8 to -6", "min": -8.0, "max": -6.0, "color": "#76df58"},
            {"label": "-6 to -3", "min": -6.0, "max": -3.0, "color": "#159a28"},
        ],
    },
    "SINR": {
        "unit": "dB",
        "columns": ("SINR_meas", "sinr", "pred_sinr"),
        "bins": [
            {"label": "-10 to 0", "min": -10.0, "max": 0.0, "color": "#ef2f2f"},
            {"label": "0 to 5", "min": 0.0, "max": 5.0, "color": "#d86b2d"},
            {"label": "5 to 10", "min": 5.0, "max": 10.0, "color": "#f29a38"},
            {"label": "10 to 15", "min": 10.0, "max": 15.0, "color": "#f3ea4e"},
            {"label": "15 to 20", "min": 15.0, "max": 20.0, "color": "#8bd3ee"},
            {"label": "20 to 25", "min": 20.0, "max": 25.0, "color": "#76df58"},
            {"label": "25 to 30", "min": 25.0, "max": 30.0, "color": "#159a28"},
        ],
    },
}


def _first_present_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_lookup = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        matched = lower_lookup.get(str(candidate).lower())
        if matched is not None:
            return matched
    return None


def _records_for_dashboard(df: pd.DataFrame, value_columns: Dict[str, str], limit: int) -> List[Dict[str, object]]:
    if df.empty or not {"lat", "lon"}.issubset(df.columns):
        return []
    sample = _safe_sample(df.dropna(subset=["lat", "lon"]).copy(), limit)
    records: List[Dict[str, object]] = []
    for row in sample.itertuples(index=False):
        row_map = row._asdict()
        record = {
            "lat": float(row_map["lat"]),
            "lon": float(row_map["lon"]),
        }
        for kpi, col in value_columns.items():
            value = pd.to_numeric(pd.Series([row_map.get(col)]), errors="coerce").iloc[0]
            record[kpi] = None if pd.isna(value) else float(value)
        if "demo_visual_source" in row_map:
            record["source"] = str(row_map.get("demo_visual_source") or "")
        records.append(record)
    return records


def _write_rf_debug_dashboard(run_dir: Path, drive_df: pd.DataFrame, pred_df: pd.DataFrame, polygon_gdf: gpd.GeoDataFrame) -> Path:
    dt = _prepare_drive_measurements(drive_df)
    dt_cols = {
        kpi: _first_present_col(dt, spec["columns"][:2])
        for kpi, spec in KPI_VISUAL_RANGES.items()
    }
    pred_cols = {
        kpi: _first_present_col(pred_df, [spec["columns"][2]])
        for kpi, spec in KPI_VISUAL_RANGES.items()
    }
    dt_cols = {kpi: col for kpi, col in dt_cols.items() if col}
    pred_cols = {kpi: col for kpi, col in pred_cols.items() if col}
    dt_records = _records_for_dashboard(dt, dt_cols, MAX_MAP_POINTS)
    pred_records = _records_for_dashboard(pred_df, pred_cols, MAX_MAP_POINTS)
    polygon_geojson = json.loads(polygon_gdf.to_json()) if not polygon_gdf.empty else None
    center_lat = next((r["lat"] for r in dt_records or pred_records), 28.6)
    center_lon = next((r["lon"] for r in dt_records or pred_records), 77.2)
    payload = {
        "dt": dt_records,
        "prediction": pred_records,
        "ranges": KPI_VISUAL_RANGES,
        "polygon": polygon_geojson,
        "center": {"lat": center_lat, "lon": center_lon},
    }
    dashboard_path = run_dir / "rf_debug_dashboard.html"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RF Debug LTE Visualization</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; font-family: Segoe UI, sans-serif; }}
    .panel {{ position: absolute; z-index: 1000; top: 16px; right: 16px; width: 280px; background: #1f2533; color: #f7f9ff; border-radius: 8px; box-shadow: 0 12px 32px rgba(0,0,0,.28); overflow: hidden; }}
    .panel header {{ padding: 12px 14px; font-weight: 700; border-bottom: 1px solid rgba(255,255,255,.12); }}
    .controls {{ display: grid; gap: 10px; padding: 12px 14px; }}
    .seg {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
    .seg.kpi {{ grid-template-columns: 1fr 1fr 1fr; }}
    button {{ border: 0; border-radius: 6px; padding: 8px 6px; background: #30384c; color: #dce4f7; cursor: pointer; font-weight: 650; }}
    button.active {{ background: #e8edf8; color: #1e2533; }}
    .legend {{ padding: 0 14px 12px; max-height: 360px; overflow: auto; }}
    .row {{ display: grid; grid-template-columns: 16px 1fr auto; gap: 9px; align-items: center; padding: 7px 0; font-size: 12px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; }}
    .meta {{ color: #aeb8cc; font-size: 11px; padding: 0 14px 14px; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <aside class="panel">
    <header id="title">RSRP</header>
    <div class="controls">
      <div class="seg">
        <button data-layer="dt" class="active">DT Original</button>
        <button data-layer="prediction">LTE Prediction</button>
      </div>
      <div class="seg kpi">
        <button data-kpi="RSRP" class="active">RSRP</button>
        <button data-kpi="RSRQ">RSRQ</button>
        <button data-kpi="SINR">SINR</button>
      </div>
    </div>
    <div class="legend" id="legend"></div>
    <div class="meta" id="meta"></div>
  </aside>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const DATA = {json.dumps(payload)};
    let activeLayer = 'dt';
    let activeKpi = 'RSRP';
    const map = L.map('map', {{ preferCanvas: true }}).setView([DATA.center.lat, DATA.center.lon], 15);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 20, attribution: '&copy; OpenStreetMap' }}).addTo(map);
    if (DATA.polygon) {{
      const poly = L.geoJSON(DATA.polygon, {{ style: {{ color: '#2563eb', weight: 1, fillOpacity: 0 }} }}).addTo(map);
      try {{ map.fitBounds(poly.getBounds(), {{ padding: [20, 20] }}); }} catch (e) {{}}
    }}
    const pointLayer = L.layerGroup().addTo(map);
    function binFor(value, bins) {{
      if (value === null || Number.isNaN(value)) return null;
      return bins.find((bin, idx) => value >= bin.min && (value < bin.max || idx === bins.length - 1 && value <= bin.max));
    }}
    function render() {{
      pointLayer.clearLayers();
      const rows = DATA[activeLayer] || [];
      const spec = DATA.ranges[activeKpi];
      const counts = spec.bins.map(() => 0);
      let plotted = 0;
      for (const row of rows) {{
        const value = row[activeKpi];
        const bin = binFor(value, spec.bins);
        if (!bin) continue;
        const idx = spec.bins.indexOf(bin);
        counts[idx] += 1;
        plotted += 1;
        L.circleMarker([row.lat, row.lon], {{
          radius: activeLayer === 'dt' ? 4 : 3,
          color: bin.color,
          fillColor: bin.color,
          fillOpacity: activeLayer === 'dt' ? 0.82 : 0.58,
          opacity: 0.9,
          weight: 1
        }}).bindTooltip(`${{activeLayer === 'dt' ? 'DT' : 'LTE'}} ${{activeKpi}}: ${{value.toFixed(2)}} ${{spec.unit}}${{row.source ? '<br>' + row.source : ''}}`).addTo(pointLayer);
      }}
      document.getElementById('title').textContent = `${{activeKpi}} (${{spec.unit}})`;
      document.getElementById('legend').innerHTML = spec.bins.map((bin, idx) => {{
        const pct = plotted ? ((counts[idx] / plotted) * 100).toFixed(1) : '0.0';
        return `<div class="row"><span class="dot" style="background:${{bin.color}}"></span><span>${{bin.label}}</span><strong>${{counts[idx].toLocaleString()}} (${{pct}}%)</strong></div>`;
      }}).join('');
      document.getElementById('meta').textContent = `${{activeLayer === 'dt' ? 'DT Original' : 'LTE Prediction'}} plotted: ${{plotted.toLocaleString()}} of ${{rows.length.toLocaleString()}} sampled points`;
    }}
    document.querySelectorAll('[data-layer]').forEach(btn => btn.addEventListener('click', () => {{
      activeLayer = btn.dataset.layer;
      document.querySelectorAll('[data-layer]').forEach(b => b.classList.toggle('active', b === btn));
      render();
    }}));
    document.querySelectorAll('[data-kpi]').forEach(btn => btn.addEventListener('click', () => {{
      activeKpi = btn.dataset.kpi;
      document.querySelectorAll('[data-kpi]').forEach(b => b.classList.toggle('active', b === btn));
      render();
    }}));
    render();
  </script>
</body>
</html>
"""
    dashboard_path.write_text(html, encoding="utf-8")
    print(f"[TEST][DASHBOARD] written={dashboard_path}")
    return dashboard_path


def _normalize_session_ids(session_ids: Iterable[int]) -> List[int]:
    return [int(session_id) for session_id in session_ids]


def _coerce_optional_path(path: Optional[Path | str]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path)


def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_optional_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _read_optional_gdf(path: Path) -> Optional[gpd.GeoDataFrame]:
    if not path.exists():
        return None
    return gpd.read_file(path)


def _read_optional_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cached_run_artifacts(reuse_run_dir: Optional[Path | str]) -> Dict[str, object]:
    base_dir = _coerce_optional_path(reuse_run_dir)
    if base_dir is None or not base_dir.exists():
        return {}

    building_df = _read_optional_csv(base_dir / "building_debug.csv")
    if building_df is None:
        building_df = _read_optional_csv(base_dir / "building_df.csv")

    return {
        "base_dir": base_dir,
        "summary": _read_optional_json(base_dir / "summary.json"),
        "site_df": _read_optional_csv(base_dir / "site_df.csv"),
        "drive_df": _read_optional_csv(base_dir / "drive_df.csv"),
        "building_df": building_df,
        "polygon_gdf": _read_optional_gdf(base_dir / "project_polygon.geojson"),
        "building_gdf": _read_optional_gdf(base_dir / "buildings.geojson"),
        "grid_gdf": _read_optional_gdf(base_dir / "analysis_grid.geojson"),
        "grid_df": _read_optional_csv(base_dir / "analysis_grid_features.csv"),
        "pred_df": _read_optional_parquet(base_dir / "rf_prediction_grid.parquet"),
        "rf_accuracy_points": _read_optional_csv(base_dir / "rf_accuracy_points.csv"),
        "building_debug_csv": (base_dir / "building_debug.csv") if (base_dir / "building_debug.csv").exists() else None,
        "rf_log_path": next(iter(sorted(base_dir.glob("run_log_*.txt"))), None),
    }


def _cached_config_matches(
    config: RunConfig,
    cached_summary: Optional[Dict[str, object]],
    fields: Iterable[str],
) -> tuple[bool, List[str]]:
    if not cached_summary:
        return False, ["summary_missing"]

    cached_config = cached_summary.get("config")
    if not isinstance(cached_config, dict):
        return False, ["config_missing"]

    mismatches: List[str] = []
    for field in fields:
        current_value = getattr(config, field)
        cached_value = cached_config.get(field)
        if field == "session_ids":
            current_value = _normalize_session_ids(current_value)
            cached_value = _normalize_session_ids(cached_value or [])
        elif isinstance(current_value, Path):
            current_value = str(current_value)
        if current_value != cached_value:
            mismatches.append(field)
    return not mismatches, mismatches


def _grid_required_feature_columns() -> List[str]:
    return [
        "grid_id",
        "lat",
        "lon",
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "azimuth_delta_deg",
        "clutter_class",
        "morphology_cluster",
        "best_interferer_distance_m",
        "best_interferer_azimuth_delta_deg",
        "serving_proxy_rsrp_dbm",
        "best_interferer_proxy_rsrp_dbm",
        "serving_proxy_rsrp_phys_dbm",
        "best_interferer_proxy_phys_dbm",
        "interference_gap_db",
        "interference_ratio_linear",
        "interference_sum_proxy_dbm",
        "sinr_proxy_db",
        "rsrq_proxy_db",
        "effective_tx_height_m",
        "los_blocker_count",
        "los_blocked_length_m",
        "los_blocked_ratio",
        "mean_blocker_height_m",
        "max_blocker_height_m",
        "nlos_flag",
        "diffraction_proxy_db",
        "terrain_elevation_m",
        "terrain_slope_deg",
        "proxy_site_elevation_m",
        "terrain_relief_to_site_m",
    ]


def _advanced_geo_feature_columns() -> List[str]:
    return [col for col in _grid_required_feature_columns() if col not in {"grid_id", "lat", "lon", "clutter_class", "morphology_cluster"}]


def _prediction_required_columns() -> List[str]:
    return [
        "lat",
        "lon",
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "grid_id",
    ]


def _cached_grid_artifacts_are_usable(
    grid_gdf: Optional[gpd.GeoDataFrame],
    grid_df: Optional[pd.DataFrame],
) -> tuple[bool, List[str]]:
    issues: List[str] = []
    if grid_gdf is None or grid_gdf.empty:
        issues.append("grid_geometry_missing")
    if grid_df is None or grid_df.empty:
        issues.append("grid_features_missing")
    if issues:
        return False, issues

    missing_cols = [col for col in _grid_required_feature_columns() if col not in grid_df.columns]
    if missing_cols:
        issues.append(f"grid_feature_columns_missing={missing_cols}")
    if "grid_id" not in grid_gdf.columns:
        issues.append("grid_geometry_missing_grid_id")
    elif "grid_id" in grid_df.columns:
        cached_ids = pd.Index(pd.to_numeric(grid_df["grid_id"], errors="coerce").dropna().astype(int))
        geom_ids = pd.Index(pd.to_numeric(grid_gdf["grid_id"], errors="coerce").dropna().astype(int))
        if set(cached_ids.tolist()) != set(geom_ids.tolist()):
            issues.append("grid_id_mismatch")
    return not issues, issues


def _cached_prediction_is_usable(pred_df: Optional[pd.DataFrame]) -> tuple[bool, List[str]]:
    issues: List[str] = []
    if pred_df is None or pred_df.empty:
        issues.append("prediction_missing")
        return False, issues
    missing_cols = [col for col in _prediction_required_columns() if col not in pred_df.columns]
    if missing_cols:
        issues.append(f"prediction_columns_missing={missing_cols}")
    return not issues, issues


def _attach_missing_grid_features_by_grid_id(pred_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    if "grid_id" not in out.columns or "grid_id" not in grid_df.columns:
        return out

    feature_cols = [col for col in _grid_required_feature_columns() if col != "grid_id"]
    missing_cols = [col for col in feature_cols if col not in out.columns]
    if not missing_cols:
        return out

    available_missing_cols = [col for col in missing_cols if col in grid_df.columns]
    if not available_missing_cols:
        return out

    grid_features = grid_df[["grid_id"] + available_missing_cols].copy()
    out = out.merge(grid_features, on="grid_id", how="left")
    return out


def _load_project_polygon_gdf(project_id: int, region: str) -> gpd.GeoDataFrame:
    current_engine = ml_engine.engine.get(region.lower(), ml_engine.engine["india"])
    polygons = ml_engine._load_project_polygons(project_id, current_engine)
    if not polygons:
        raise ValueError(f"No project polygons found for project_id={project_id}")
    return gpd.GeoDataFrame({"geometry": polygons}, crs="EPSG:4326")


def _swap_geometry_xy(geom):
    return transform(lambda x, y, z=None: (y, x) if z is None else (y, x, z), geom)


def _fetch_building_data_for_test(project_id: int, region: str) -> pd.DataFrame:
    current_engine = ml_engine.engine.get(region.lower(), ml_engine.engine["india"])
    query = f"""
    SELECT
        t.*,
        ST_AsText(t.region) AS region_wkt,
        ST_AsText(t.geometry) AS geometry_wkt
    FROM tbl_savepolygon AS t
    WHERE t.project_id = {project_id}
    """
    df = pd.read_sql(query, current_engine)
    for raw_col in ["region", "geometry"]:
        wkt_col = f"{raw_col}_wkt"
        if raw_col in df.columns and wkt_col in df.columns:
            parsed_from_raw = df[raw_col].apply(_parse_geometry_value)
            needs_fill = df[wkt_col].isna() | (df[wkt_col].astype(str).str.strip() == "")
            if needs_fill.any():
                df.loc[needs_fill, wkt_col] = parsed_from_raw.loc[needs_fill].apply(
                    lambda geom: geom.wkt if geom is not None and not geom.is_empty else None
                )
    print(f"[TEST][BUILDING_FETCH] row_count={len(df)} project_id={project_id} region={region}")
    print(f"[TEST][BUILDING_FETCH] columns={list(df.columns)}")
    if "region_wkt" in df.columns:
        print(f"[TEST][BUILDING_FETCH] non_null_region_wkt={int(df['region_wkt'].notna().sum())}")
    if "geometry_wkt" in df.columns:
        print(f"[TEST][BUILDING_FETCH] non_null_geometry_wkt={int(df['geometry_wkt'].notna().sum())}")
    if "region" in df.columns:
        print(f"[TEST][BUILDING_FETCH] non_null_region={int(df['region'].notna().sum())}")
    height_cols = _candidate_building_height_columns(df)
    level_cols = _candidate_building_level_columns(df)
    if height_cols:
        print(f"[TEST][BUILDING_FETCH] building_height_columns={height_cols}")
    if level_cols:
        print(f"[TEST][BUILDING_FETCH] building_level_columns={level_cols}")
    return df


def _fetch_drive_data_for_test(
    session_ids: Iterable[int],
    operator: str,
    project_id: int,
    region: str = "india",
) -> pd.DataFrame:
    session_ids = tuple(int(session_id) for session_id in session_ids)
    session_str = ",".join(map(str, session_ids))
    current_engine = ml_engine.engine.get(region.lower(), ml_engine.engine["india"])

    query = f"""
    SELECT session_id, lat, lon, rsrp, rsrq, sinr, cell_id, nodeb_id, pci, earfcn,
           'serving' AS measurement_role
    FROM tbl_network_log
    WHERE session_id IN ({session_str})
      AND LOWER(COALESCE(m_alpha_long, m_alpha_short)) = LOWER('{operator}')
      AND LOWER(COALESCE(`primary`, '')) = 'yes'
    UNION ALL
    SELECT session_id, lat, lon, rsrp, rsrq, sinr, cell_id, nodeb_id, pci, earfcn,
           'neighbor' AS measurement_role
    FROM tbl_network_log_neighbour
    WHERE session_id IN ({session_str})
      AND LOWER(COALESCE(m_alpha_long, m_alpha_short)) = LOWER('{operator}')
      AND LOWER(COALESCE(`primary`, '')) = 'yes'
    """
    df = pd.read_sql(query, current_engine)
    for col in ["cell_id", "nodeb_id", "pci", "earfcn"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df, polygon_stats = ml_engine._apply_drive_polygon_filter(df, project_id, current_engine)
    ml_engine._print_fetch_summary(
        "DRIVE_FETCH_TEST",
        "tbl_network_log + tbl_network_log_neighbour",
        {"session_ids": session_ids, "operator": operator, "project_id": project_id, "region": region},
        df,
        extra={
            "distinct_session_id": int(df["session_id"].nunique()) if "session_id" in df.columns else 0,
            "lat_range": ml_engine._safe_minmax(df, "lat"),
            "lon_range": ml_engine._safe_minmax(df, "lon"),
            "polygon_swapped": polygon_stats["swapped"],
        },
    )
    return df


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

        if text_candidate.startswith(("b'", 'b"', 'bytearray(')):
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


def _normalize_site_for_rf(site_df: pd.DataFrame) -> pd.DataFrame:
    out = site_df.copy()
    duplicate_cols = out.columns[out.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"[TEST][SITE_NORMALIZE] dropping_duplicate_columns={duplicate_cols}")
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
        "tx_power": None,
        "frequency_mhz": 1800,
    }
    for col, default in numeric_defaults.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if default is not None:
                out[col] = out[col].fillna(default)
        elif default is not None:
            out[col] = default

    if "frequency_mhz" not in out.columns:
        if "frequency" in out.columns:
            out["frequency_mhz"] = pd.to_numeric(out["frequency"], errors="coerce").fillna(1800)
        else:
            out["frequency_mhz"] = 1800
    original_tx_missing = int(pd.to_numeric(out["tx_power"], errors="coerce").isna().sum()) if "tx_power" in out.columns else len(out)
    out["tx_power"] = _apply_tx_power_fallback(out)
    if original_tx_missing:
        print(
            f"[TEST][TX_POWER_FALLBACK] missing_tx_power_rows={original_tx_missing} "
            f"fallback_counts={out['tx_power'].value_counts(dropna=False).sort_index().to_dict()}"
        )

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
    out = _add_sector_identity_columns(out, use_as_node_cell_id=True)
    return out


def _valid_earfcn_score(series: pd.Series) -> pd.Series:
    earfcn = pd.to_numeric(series, errors="coerce")
    return earfcn.between(1.0, 65535.0).astype(int)


def _site_canonical_duplicate_audit(site_df: pd.DataFrame) -> pd.DataFrame:
    if site_df.empty or "canonical_sector_id" not in site_df.columns:
        return pd.DataFrame()
    work = site_df.copy()
    for col in ["earfcn", "frequency_mhz", "azimuth", "electrical_tilt", "mechanical_tilt", "tx_power"]:
        if col not in work.columns:
            work[col] = np.nan
    grouped = (
        work.groupby("canonical_sector_id", dropna=False)
        .agg(
            raw_row_count=("canonical_sector_id", "size"),
            node_cell_id_values=("Node_Cell_ID", lambda s: "|".join(sorted(set(s.dropna().astype(str).str.strip()))[:20])),
            earfcn_values=("earfcn", lambda s: "|".join(sorted(set(s.dropna().astype(str).str.strip()))[:20])),
            frequency_values=("frequency_mhz", lambda s: "|".join(sorted(set(s.dropna().astype(str).str.strip()))[:20])),
            azimuth_values=("azimuth", lambda s: "|".join(sorted(set(s.dropna().astype(str).str.strip()))[:20])),
            etilt_values=("electrical_tilt", lambda s: "|".join(sorted(set(s.dropna().astype(str).str.strip()))[:20])),
            mtilt_values=("mechanical_tilt", lambda s: "|".join(sorted(set(s.dropna().astype(str).str.strip()))[:20])),
            tx_power_values=("tx_power", lambda s: "|".join(sorted(set(s.dropna().astype(str).str.strip()))[:20])),
        )
        .reset_index()
        .sort_values(["raw_row_count", "canonical_sector_id"], ascending=[False, True])
    )
    return grouped


def _deduplicate_site_df_for_rf_matching(site_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if site_df.empty or "canonical_sector_id" not in site_df.columns:
        return site_df.copy(), {"enabled": False, "reason": "missing_canonical_sector_id"}
    work = site_df.copy()
    before_rows = len(work)
    before_node_cell_ids = int(work["Node_Cell_ID"].nunique(dropna=True)) if "Node_Cell_ID" in work.columns else 0
    before_canonical = int(work["canonical_sector_id"].nunique(dropna=True))
    for col in ["lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "antenna_height", "tx_power", "frequency_mhz", "earfcn"]:
        if col not in work.columns:
            work[col] = np.nan
    completeness_cols = ["lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "antenna_height", "tx_power", "frequency_mhz"]
    work["_rf_completeness_score"] = work[completeness_cols].notna().sum(axis=1)
    work["_valid_earfcn_score"] = _valid_earfcn_score(work["earfcn"])
    work["_canonical_preferred_id_score"] = work["Node_Cell_ID"].astype(str).str.contains("_", regex=False).astype(int)
    work["_valid_frequency_score"] = pd.to_numeric(work["frequency_mhz"], errors="coerce").between(600.0, 4000.0).astype(int)
    work = work.sort_values(
        [
            "canonical_sector_id",
            "_valid_earfcn_score",
            "_valid_frequency_score",
            "_rf_completeness_score",
            "_canonical_preferred_id_score",
        ],
        ascending=[True, False, False, False, False],
    )
    deduped = work.drop_duplicates(subset=["canonical_sector_id"], keep="first").copy()
    deduped = deduped.drop(
        columns=[
            "_rf_completeness_score",
            "_valid_earfcn_score",
            "_canonical_preferred_id_score",
            "_valid_frequency_score",
        ],
        errors="ignore",
    )
    summary = {
        "enabled": True,
        "before_rows": int(before_rows),
        "after_rows": int(len(deduped)),
        "before_node_cell_id_count": int(before_node_cell_ids),
        "before_canonical_sector_count": int(before_canonical),
        "duplicate_physical_sector_count": int((work.groupby("canonical_sector_id").size() > 1).sum()),
        "max_rows_per_physical_sector": int(work.groupby("canonical_sector_id").size().max()) if before_canonical else 0,
    }
    return deduped.reset_index(drop=True), summary


def _filter_site_df_for_operator(site_df: pd.DataFrame, operator: str) -> pd.DataFrame:
    operator_clean = str(operator or "").strip()
    if not operator_clean:
        return site_df
    for col in ["network", "cluster", "operator", "Technology"]:
        if col not in site_df.columns:
            continue
        values = site_df[col].astype(str).str.strip()
        filtered = site_df.loc[values.str.lower() == operator_clean.lower()].copy()
        print(
            f"[TEST][SITE_OPERATOR_FILTER] operator={operator_clean} column={col} "
            f"before_rows={len(site_df)} after_rows={len(filtered)}"
        )
        if filtered.empty:
            raise ValueError(f"No site rows found for operator={operator_clean} using column={col}")
        return filtered
    print(f"[TEST][SITE_OPERATOR_FILTER] no_operator_column_found requested_operator={operator_clean}; using all site rows")
    return site_df


def _infer_frequency_for_power_fallback(site_df: pd.DataFrame) -> pd.Series:
    fallback = pd.Series(np.nan, index=site_df.index, dtype=float)
    for col in ["band", "frequency", "frequency_mhz", "downlink_frequency", "uplink_center_frequency"]:
        if col in site_df.columns:
            values = pd.to_numeric(site_df[col], errors="coerce")
            fallback = fallback.where(fallback.notna(), values)
    return fallback.fillna(1800.0)


def _tx_power_from_frequency_mhz(freq_mhz: pd.Series) -> pd.Series:
    freq = pd.to_numeric(freq_mhz, errors="coerce").fillna(1800.0)
    return pd.Series(
        np.select(
            [
                freq <= 900.0,
                (freq > 900.0) & (freq <= 1800.0),
                (freq > 1800.0) & (freq <= 2700.0),
                freq > 2700.0,
            ],
            [45.0, 46.0, 49.0, 51.0],
            default=46.0,
        ),
        index=freq.index,
        dtype=float,
    )


def _apply_tx_power_fallback(site_df: pd.DataFrame) -> pd.Series:
    tx_power = (
        pd.to_numeric(site_df["tx_power"], errors="coerce")
        if "tx_power" in site_df.columns
        else pd.Series(np.nan, index=site_df.index, dtype=float)
    )
    fallback_power = _tx_power_from_frequency_mhz(_infer_frequency_for_power_fallback(site_df))
    return tx_power.fillna(fallback_power)


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


def _add_sector_identity_columns(site_df: pd.DataFrame, use_as_node_cell_id: bool = False) -> pd.DataFrame:
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

    out["site_identity_key"] = site_key
    out["sector_identity"] = sector_or_cell
    out["frontend_site_sector_key"] = site_key.astype(str) + "|" + sector_or_cell.astype(str)
    out["node_cell_sector_key"] = out["original_node_cell_id"].astype(str) + "|" + sector_or_cell.astype(str)
    out["canonical_sector_id"] = _canonical_sector_ids(site_key, sector_or_cell, out["original_cell_id"])
    out.loc[sector_or_cell.isna(), ["frontend_site_sector_key", "node_cell_sector_key"]] = pd.NA

    if use_as_node_cell_id:
        out["Node_Cell_ID"] = out["frontend_site_sector_key"]
    return out


def _feature_diagnostics(grid_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    for col in [
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "azimuth_delta_deg",
        "best_interferer_distance_m",
        "interference_gap_db",
        "sinr_proxy_db",
        "rsrq_proxy_db",
        "effective_tx_height_m",
        "los_blocker_count",
        "los_blocked_ratio",
        "diffraction_proxy_db",
        "terrain_elevation_m",
        "terrain_slope_deg",
        "terrain_relief_to_site_m",
    ]:
        series = pd.to_numeric(grid_df.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0)
        report[col] = {
            "non_zero": int((series != 0).sum()),
            "nunique": int(series.nunique(dropna=True)),
            "min": float(series.min()) if len(series) else 0.0,
            "max": float(series.max()) if len(series) else 0.0,
            "mean": float(series.mean()) if len(series) else 0.0,
        }
    return report


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
    min_distance_m: float = 1.0,
):
    distance_floor = max(float(min_distance_m or 1.0), 1.0)
    distance_m = np.maximum(_haversine_m_np(site_lat, site_lon, point_lat, point_lon), distance_floor)
    distance_km = np.maximum(distance_m / 1000.0, 0.001)
    freq = np.clip(np.asarray(site_frequency_mhz, dtype=float), 700.0, 3500.0)
    h_tx = np.asarray(site_height, dtype=float)
    if site_elevation_m is not None and point_elevation_m is not None:
        elev_delta = np.asarray(site_elevation_m, dtype=float) - np.asarray(point_elevation_m, dtype=float)
        h_tx = h_tx + elev_delta
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
    tx_power = np.asarray(site_tx_power, dtype=float)
    return tx_power + gain - pathloss - 2.0


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


def _offset_latlon(lat: float, lon: float, north_m: float = 0.0, east_m: float = 0.0) -> tuple[float, float]:
    dlat = north_m / 111320.0
    cos_lat = math.cos(math.radians(lat))
    dlon = east_m / max(111320.0 * max(abs(cos_lat), 1e-6), 1e-6)
    return lat + dlat, lon + dlon


def _project_shared_cache_dir(output_root: Path, project_id: int) -> Path:
    return _ensure_dir(output_root / f"project_{project_id}" / "shared_cache")


def _building_df_to_gdf(building_df: pd.DataFrame) -> gpd.GeoDataFrame:
    geom_col = None
    for candidate in ("region_wkt", "geometry_wkt", "geometry", "region"):
        if candidate not in building_df.columns:
            continue
        sample_series = building_df[candidate].dropna()
        if sample_series.empty:
            continue
        sample_values = sample_series.head(10).tolist()
        if any(_parse_geometry_value(value) is not None for value in sample_values):
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
        if geom is None:
            continue
        if geom.is_empty:
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
        records.append({
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
        })

    if not geometries:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    return gdf


def _align_building_geometries_to_project(
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
        aligned = swapped.drop(columns=["_intersects"])
        return aligned, f"swapped_xy direct_hits={direct_hits} swapped_hits={swapped_hits}"

    aligned = direct.drop(columns=["_intersects"])
    return aligned, f"original direct_hits={direct_hits} swapped_hits={swapped_hits}"


def _align_project_polygon_to_points(
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
    swapped_union = swapped.geometry.union_all()
    swapped_hits = int(point_gdf.geometry.within(swapped_union).sum())

    if swapped_hits > direct_hits:
        return swapped, f"swapped_xy direct_hits={direct_hits} swapped_hits={swapped_hits}"
    return polygon_gdf, f"original direct_hits={direct_hits} swapped_hits={swapped_hits}"


def _frontend_site_sector_summary(site_df: pd.DataFrame, polygon_gdf: gpd.GeoDataFrame) -> Dict[str, int]:
    if site_df.empty or polygon_gdf.empty or not {"lat", "lon"}.issubset(site_df.columns):
        return {"inside_polygon_site_rows": 0, "frontend_site_sector_count": 0}

    work = site_df.copy()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    valid = work["lat"].notna() & work["lon"].notna()
    if not valid.any():
        return {"inside_polygon_site_rows": 0, "frontend_site_sector_count": 0}

    point_gdf = gpd.GeoDataFrame(
        work.loc[valid].copy(),
        geometry=gpd.points_from_xy(work.loc[valid, "lon"], work.loc[valid, "lat"]),
        crs=polygon_gdf.crs or "EPSG:4326",
    )
    polygon_union = polygon_gdf.geometry.union_all()
    inside = point_gdf.loc[point_gdf.geometry.apply(lambda geom: bool(polygon_union.contains(geom)))].copy()
    if inside.empty:
        return {"inside_polygon_site_rows": 0, "frontend_site_sector_count": 0}

    def _clean_key_part(series: pd.Series) -> pd.Series:
        out = series.astype("string").str.strip()
        return out.mask(out.isin(["", "nan", "NaN", "None", "<NA>"]))

    site_col = next((col for col in ["site", "Site ID", "site_id", "site_name"] if col in inside.columns), None)
    site_key = (
        _clean_key_part(inside[site_col]).fillna("unknown-site")
        if site_col
        else pd.Series("unknown-site", index=inside.index, dtype="string")
    )
    sector_key = _clean_key_part(inside["sector"]) if "sector" in inside.columns else pd.Series(pd.NA, index=inside.index, dtype="string")
    cell_key = _clean_key_part(inside["cell_id"]) if "cell_id" in inside.columns else pd.Series(pd.NA, index=inside.index, dtype="string")
    sector_or_cell = sector_key.fillna(cell_key)
    valid_key = sector_or_cell.notna()
    distinct_key = site_key.loc[valid_key].astype(str) + "|" + sector_or_cell.loc[valid_key].astype(str)
    return {
        "inside_polygon_site_rows": int(len(inside)),
        "frontend_site_sector_count": int(distinct_key.nunique(dropna=True)),
    }


def _filter_sites_to_project_polygon(site_df: pd.DataFrame, polygon_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
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


def _prepare_building_df_for_rf(building_df: pd.DataFrame, building_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
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


def _prepare_site_df_for_source_rf_export(site_df: pd.DataFrame) -> pd.DataFrame:
    rf_df = _add_sector_identity_columns(site_df, use_as_node_cell_id=True)
    rf_df["tx_power"] = _apply_tx_power_fallback(rf_df)
    tx_counts = pd.to_numeric(rf_df["tx_power"], errors="coerce").value_counts(dropna=False).sort_index().to_dict()
    band_counts = (
        pd.to_numeric(rf_df["band"], errors="coerce").value_counts(dropna=False).sort_index().to_dict()
        if "band" in rf_df.columns
        else {}
    )
    print(f"[TEST][RF_EXPORT_POWER] tx_power_counts={tx_counts} band_counts={band_counts}")
    if "cell_id" in rf_df.columns and "rf_source_cell_id" not in rf_df.columns:
        rf_df["rf_source_cell_id"] = rf_df["cell_id"]
    if "Node_Cell_ID" in rf_df.columns:
        rf_df["cell_id"] = rf_df["Node_Cell_ID"]

    # The source predictor renames Etilt/Mtilt/Height -> electrical_tilt/mechanical_tilt/antenna_height.
    # If we export both versions, pandas will recreate duplicate column names inside the source path.
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


def _attach_site_identity_to_predictions(pred_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    if out.empty or site_df.empty or "Node_Cell_ID" not in out.columns:
        return out

    site_identity = _add_sector_identity_columns(site_df, use_as_node_cell_id=True)
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
            "site",
            "sector",
            "nodeb_id",
            "pci",
            "earfcn",
            "azimuth",
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


def _create_analysis_grid(mask_gdf: gpd.GeoDataFrame, cell_size_m: float) -> gpd.GeoDataFrame:
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
                Polygon(
                    [(x, y), (x + cell_size_m, y), (x + cell_size_m, y + cell_size_m), (x, y + cell_size_m)]
                )
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


def _attach_building_features(grid_gdf: gpd.GeoDataFrame, building_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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


def _normalize_building_height_gdf(building_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = building_gdf.copy()
    if out.empty:
        if "building_height_m" not in out.columns:
            out["building_height_m"] = pd.Series(dtype=float)
        return out

    if "building_height_m" in out.columns:
        out["building_height_m"] = pd.to_numeric(out["building_height_m"], errors="coerce")
    else:
        out["building_height_m"] = np.nan

    height_source_cols = [
        col for col in ["height_m", "height", "building:height", "building_height", "roof_height"]
        if col in out.columns
    ]
    for col in height_source_cols:
        series = pd.to_numeric(out[col], errors="coerce")
        out["building_height_m"] = out["building_height_m"].fillna(series)

    level_source_cols = [col for col in ["building_levels", "levels", "building:levels", "floors", "num_floors"] if col in out.columns]
    for col in level_source_cols:
        levels = pd.to_numeric(out[col], errors="coerce")
        out["building_height_m"] = out["building_height_m"].fillna(levels * 3.0)
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
    utm_crs = _choose_utm_crs(building_gdf if not building_gdf.empty else gpd.GeoDataFrame(geometry=[]))
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
        candidates = building_utm.iloc[candidate_idx]
        hits = candidates[candidates.geometry.intersects(path)].copy()
        if hits.empty:
            continue
        path_geoms = hits.geometry.intersection(path)
        lengths = np.array([geom.length for geom in path_geoms if geom is not None and not geom.is_empty], dtype=float)
        blocker_count[row_idx] = float(len(hits))
        blocked_length[row_idx] = float(lengths.sum()) if len(lengths) else 0.0
        valid_heights = pd.to_numeric(hits.get("building_height_m", pd.Series(dtype=float)), errors="coerce").dropna()
        if not valid_heights.empty:
            mean_height[row_idx] = float(valid_heights.mean())
            max_height[row_idx] = float(valid_heights.max())

    out["los_blocker_count"] = blocker_count
    out["los_blocked_length_m"] = blocked_length
    out["los_blocked_ratio"] = (
        blocked_length / np.maximum(pd.to_numeric(out["serving_distance_m"], errors="coerce").fillna(1.0).to_numpy(dtype=float), 1.0)
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


def _attach_dem_features(points_df: pd.DataFrame, dem_raster_path: Optional[Path | str]) -> tuple[pd.DataFrame, Dict[str, object]]:
    out = points_df.copy()
    dem_path = _coerce_optional_path(dem_raster_path)
    status = {
        "enabled": dem_path is not None,
        "path": str(dem_path) if dem_path is not None else None,
        "sampled": False,
        "reason": None,
    }
    default_cols = ["terrain_elevation_m", "terrain_slope_deg", "proxy_site_elevation_m", "terrain_relief_to_site_m"]
    for col in default_cols:
        if col not in out.columns:
            out[col] = np.nan

    if dem_path is None:
        status["reason"] = "dem_not_configured"
        return out, status
    if rasterio is None:
        status["reason"] = "rasterio_unavailable"
        return out, status
    if not dem_path.exists():
        status["reason"] = "dem_file_missing"
        return out, status
    if out.empty or not {"lat", "lon"}.issubset(out.columns):
        status["reason"] = "points_missing"
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
        west_xy = [(x - dx, y) for x, y in point_xy]
        east_xy = [(x + dx, y) for x, y in point_xy]
        south_xy = [(x, y - dy) for x, y in point_xy]
        north_xy = [(x, y + dy) for x, y in point_xy]
        west = np.array([sample[0] for sample in src.sample(west_xy)], dtype=float)
        east = np.array([sample[0] for sample in src.sample(east_xy)], dtype=float)
        south = np.array([sample[0] for sample in src.sample(south_xy)], dtype=float)
        north = np.array([sample[0] for sample in src.sample(north_xy)], dtype=float)
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


def _load_terrain_sample_cache(cache_path: Path) -> Dict[tuple[float, float], float]:
    if not cache_path.exists():
        return {}
    try:
        df = pd.read_csv(cache_path)
    except Exception:
        return {}
    if not {"lat", "lon", "elevation_m"}.issubset(df.columns):
        return {}
    cache: Dict[tuple[float, float], float] = {}
    for row in df.itertuples(index=False):
        try:
            cache[(round(float(row.lat), 7), round(float(row.lon), 7))] = float(row.elevation_m)
        except Exception:
            continue
    return cache


def _write_terrain_sample_cache(cache_path: Path, cache: Dict[tuple[float, float], float]) -> None:
    rows = [
        {"lat": key[0], "lon": key[1], "elevation_m": value}
        for key, value in sorted(cache.items())
    ]
    pd.DataFrame(rows).to_csv(cache_path, index=False)


def _fetch_remote_elevation_map(
    coordinates: List[tuple[float, float]],
    api_url: str,
    batch_size: int,
    cache_path: Path,
) -> tuple[Dict[tuple[float, float], float], Dict[str, object]]:
    status: Dict[str, object] = {
        "used_remote_api": False,
        "api_url": api_url,
        "cache_path": str(cache_path),
        "requested_points": len(coordinates),
        "fetched_points": 0,
        "reason": None,
    }
    coord_keys = [(round(float(lat), 7), round(float(lon), 7)) for lat, lon in coordinates]
    sample_cache = _load_terrain_sample_cache(cache_path)
    missing = [key for key in coord_keys if key not in sample_cache]
    if not missing:
        status["fetched_points"] = len(coord_keys)
        status["reason"] = "cache_hit"
        return sample_cache, status

    if not api_url:
        status["reason"] = "terrain_api_missing"
        return sample_cache, status

    status["used_remote_api"] = True
    batch_size = max(1, int(batch_size))
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        locations = "|".join(f"{lat:.7f},{lon:.7f}" for lat, lon in batch)
        query = urllib.parse.urlencode({"locations": locations})
        request_url = f"{api_url}?{query}"
        try:
            with urllib.request.urlopen(request_url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            status["reason"] = f"terrain_api_request_failed:{exc}"
            return sample_cache, status

        results = payload.get("results")
        if not isinstance(results, list) or len(results) != len(batch):
            status["reason"] = "terrain_api_invalid_response"
            return sample_cache, status

        for key, item in zip(batch, results):
            elevation = item.get("elevation") if isinstance(item, dict) else None
            if elevation is None:
                status["reason"] = "terrain_api_missing_elevation"
                return sample_cache, status
            sample_cache[key] = float(elevation)

    status["fetched_points"] = len(coord_keys)
    status["reason"] = "remote_fetch_ok"
    _write_terrain_sample_cache(cache_path, sample_cache)
    return sample_cache, status


def _attach_terrain_features_from_remote_api(
    points_df: pd.DataFrame,
    cache_dir: Path,
    project_id: int,
    api_url: str,
    batch_size: int,
    sample_step_m: float,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    out = points_df.copy()
    status: Dict[str, object] = {
        "enabled": True,
        "path": None,
        "sampled": False,
        "reason": None,
        "source": "remote_api",
    }
    default_cols = ["terrain_elevation_m", "terrain_slope_deg", "proxy_site_elevation_m", "terrain_relief_to_site_m"]
    for col in default_cols:
        if col not in out.columns:
            out[col] = np.nan
    if out.empty or not {"lat", "lon", "_proxy_site_lat", "_proxy_site_lon"}.issubset(out.columns):
        status["reason"] = "points_missing"
        return out, status

    sample_step_m = max(float(sample_step_m), 5.0)
    cache_path = cache_dir / f"terrain_samples_project_{project_id}.csv"
    point_coords = [
        (float(lat), float(lon))
        for lat, lon in zip(pd.to_numeric(out["lat"], errors="coerce"), pd.to_numeric(out["lon"], errors="coerce"))
        if pd.notna(lat) and pd.notna(lon)
    ]
    site_coords = [
        (float(lat), float(lon))
        for lat, lon in zip(
            pd.to_numeric(out["_proxy_site_lat"], errors="coerce"),
            pd.to_numeric(out["_proxy_site_lon"], errors="coerce"),
        )
        if pd.notna(lat) and pd.notna(lon)
    ]
    unique_point_coords = sorted({(round(lat, 7), round(lon, 7)) for lat, lon in point_coords})
    unique_site_coords = sorted({(round(lat, 7), round(lon, 7)) for lat, lon in site_coords})
    all_coords = [(lat, lon) for lat, lon in unique_point_coords] + [(lat, lon) for lat, lon in unique_site_coords]

    sample_cache, remote_status = _fetch_remote_elevation_map(
        all_coords,
        api_url=api_url,
        batch_size=batch_size,
        cache_path=cache_path,
    )
    status.update(remote_status)
    if remote_status.get("reason") not in {"cache_hit", "remote_fetch_ok"}:
        return out, status

    terrain_vals = [
        sample_cache.get((round(float(lat), 7), round(float(lon), 7)), np.nan)
        for lat, lon in zip(pd.to_numeric(out["lat"], errors="coerce"), pd.to_numeric(out["lon"], errors="coerce"))
    ]
    site_vals = [
        sample_cache.get((round(float(lat), 7), round(float(lon), 7)), np.nan)
        for lat, lon in zip(
            pd.to_numeric(out["_proxy_site_lat"], errors="coerce"),
            pd.to_numeric(out["_proxy_site_lon"], errors="coerce"),
        )
    ]
    out["terrain_elevation_m"] = terrain_vals
    out["proxy_site_elevation_m"] = site_vals

    slope_vals = np.full(len(out), np.nan, dtype=float)
    if {"grid_id", "lat", "lon"}.issubset(out.columns):
        slope_frame = out[["grid_id", "lat", "lon", "terrain_elevation_m"]].copy()
        slope_frame["lat"] = pd.to_numeric(slope_frame["lat"], errors="coerce")
        slope_frame["lon"] = pd.to_numeric(slope_frame["lon"], errors="coerce")
        slope_frame["terrain_elevation_m"] = pd.to_numeric(slope_frame["terrain_elevation_m"], errors="coerce")
        slope_frame = slope_frame.dropna(subset=["lat", "lon", "terrain_elevation_m"]).copy()
        if len(slope_frame) >= 3:
            slope_gdf = gpd.GeoDataFrame(
                slope_frame,
                geometry=gpd.points_from_xy(slope_frame["lon"], slope_frame["lat"]),
                crs="EPSG:4326",
            )
            utm_crs = _choose_utm_crs(slope_gdf)
            slope_utm = slope_gdf.to_crs(utm_crs)
            coords = np.c_[slope_utm.geometry.x.to_numpy(dtype=float), slope_utm.geometry.y.to_numpy(dtype=float)]
            tree = BallTree(coords, metric="euclidean")
            neighbor_k = min(5, len(slope_utm))
            distances, indices = tree.query(coords, k=neighbor_k)
            slope_by_grid: Dict[int, float] = {}
            for row_pos, (dist_row, idx_row) in enumerate(zip(distances, indices)):
                valid_neighbors = [
                    (float(dist), int(idx))
                    for dist, idx in zip(dist_row[1:], idx_row[1:])
                    if dist > 0
                ]
                if not valid_neighbors:
                    slope_by_grid[int(slope_utm.iloc[row_pos]["grid_id"])] = 0.0
                    continue
                elevation_diffs = [
                    abs(
                        float(slope_utm.iloc[row_pos]["terrain_elevation_m"])
                        - float(slope_utm.iloc[idx]["terrain_elevation_m"])
                    ) / max(dist, 1.0)
                    for dist, idx in valid_neighbors
                ]
                grade = float(np.mean(elevation_diffs)) if elevation_diffs else 0.0
                slope_by_grid[int(slope_utm.iloc[row_pos]["grid_id"])] = float(np.degrees(np.arctan(grade)))
            slope_vals = pd.to_numeric(out["grid_id"], errors="coerce").map(slope_by_grid).to_numpy(dtype=float)
    out["terrain_slope_deg"] = slope_vals
    out["terrain_relief_to_site_m"] = (
        pd.to_numeric(out["terrain_elevation_m"], errors="coerce")
        - pd.to_numeric(out["proxy_site_elevation_m"], errors="coerce")
    )
    status["sampled"] = True
    return out, status


def _refine_experimental_forward_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_sinr_proxy_db = pd.to_numeric(out.get("sinr_proxy_db"), errors="coerce").copy()
    raw_rsrq_proxy_db = pd.to_numeric(out.get("rsrq_proxy_db"), errors="coerce").copy()
    out["sinr_proxy_db_raw"] = raw_sinr_proxy_db
    out["rsrq_proxy_db_raw"] = raw_rsrq_proxy_db
    raw_serving_phys = pd.to_numeric(out.get("serving_proxy_rsrp_phys_dbm"), errors="coerce").fillna(-120.0)
    raw_best_interferer_phys = pd.to_numeric(out.get("best_interferer_proxy_phys_dbm"), errors="coerce").fillna(-120.0)
    raw_interference_sum = pd.to_numeric(out.get("interference_sum_proxy_dbm"), errors="coerce").fillna(-120.0)
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
        series = out[col] if col in out.columns else pd.Series(0.0, index=out.index, dtype=float)
        out[col] = pd.to_numeric(series, errors="coerce").fillna(0.0)

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
    # Keep coverage-side serving refinement, but cap it so the later SINR path
    # does not recursively re-apply the full geo degradation.
    serving_physics_delta = (relief_penalty + slope_penalty + obstruction_penalty + off_axis_penalty + height_bonus).clip(lower=-6.0, upper=2.0)
    out["serving_proxy_rsrp_phys_dbm"] = raw_serving_phys + serving_physics_delta

    interferer_relief_penalty = -0.010 * out["terrain_relief_to_site_m"].clip(lower=0.0, upper=180.0)
    interferer_off_axis = -0.006 * out["best_interferer_azimuth_delta_deg"].clip(0.0, 180.0)
    interferer_distance_penalty = -0.0012 * out["best_interferer_distance_m"].clip(0.0, 1200.0)
    interferer_physics_delta = (
        interferer_relief_penalty
        + interferer_off_axis
        + interferer_distance_penalty
    )
    out["best_interferer_proxy_phys_dbm"] = (
        raw_best_interferer_phys
        + interferer_relief_penalty
        + interferer_off_axis
        + interferer_distance_penalty
    )
    # Decouple SINR dominance from full coverage penalties. We keep only a mild
    # serving-side local adjustment and a small interferer attenuation effect.
    sinr_serving_phys_dbm = raw_serving_phys + (0.35 * serving_physics_delta)
    sinr_interference_sum_dbm = raw_interference_sum + np.minimum(0.0, 0.20 * interferer_physics_delta)
    out["interference_sum_proxy_dbm"] = sinr_interference_sum_dbm
    sinr_best_interferer_dbm = raw_best_interferer_phys + np.minimum(0.0, 0.35 * interferer_physics_delta)
    out["interference_gap_db"] = sinr_serving_phys_dbm - sinr_best_interferer_dbm
    out["interference_ratio_linear"] = np.power(
        10.0,
        (sinr_best_interferer_dbm - sinr_serving_phys_dbm) / 10.0,
    )

    noise_linear = 10 ** (-104.0 / 10.0)
    serving_linear = np.power(10.0, sinr_serving_phys_dbm / 10.0)
    interference_linear = np.power(10.0, sinr_interference_sum_dbm / 10.0)
    interference_linear = np.maximum(interference_linear, noise_linear)
    out["sinr_proxy_db"] = 10.0 * np.log10(np.maximum(serving_linear, noise_linear) / interference_linear)
    rssi_linear = serving_linear + interference_linear
    out["rsrq_proxy_db"] = sinr_serving_phys_dbm - (10.0 * np.log10(rssi_linear)) + 10.0 * np.log10(50.0)
    return out


def _ensure_required_building_source(
    building_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    cache_dir: Path,
) -> tuple[gpd.GeoDataFrame, Dict[str, object]]:
    status: Dict[str, object] = {
        "source": "db",
        "fetched_from_osm": False,
        "height_coverage_ratio": 0.0,
    }
    normalized = _normalize_building_height_gdf(building_gdf)
    if not normalized.empty:
        non_null_heights = pd.to_numeric(normalized.get("building_height_m", pd.Series(dtype=float)), errors="coerce").notna().mean()
        status["height_coverage_ratio"] = float(non_null_heights) if pd.notna(non_null_heights) else 0.0
        if status["height_coverage_ratio"] > 0.0:
            return normalized, status

        osm_buildings = _fetch_osm_layer("buildings", polygon_gdf, BUILDING_TAGS, cache_dir)
        osm_buildings = _normalize_building_height_gdf(osm_buildings)
        osm_height_rows = pd.to_numeric(osm_buildings.get("building_height_m", pd.Series(dtype=float)), errors="coerce").notna().sum()
        if osm_buildings.empty or int(osm_height_rows) == 0:
            return normalized, status

        utm_crs = _choose_utm_crs(polygon_gdf)
        local_utm = normalized.to_crs(utm_crs).copy()
        osm_utm = osm_buildings.to_crs(utm_crs).copy()
        local_utm["geometry"] = local_utm.geometry.centroid
        osm_utm["geometry"] = osm_utm.geometry.centroid
        osm_utm = osm_utm[pd.to_numeric(osm_utm.get("building_height_m"), errors="coerce").notna()].copy()
        if osm_utm.empty:
            return normalized, status

        joined = gpd.sjoin_nearest(
            local_utm[["geometry"]],
            osm_utm[["geometry", "building_height_m"]],
            how="left",
            distance_col="_height_match_m",
            max_distance=35.0,
        )
        matched_heights = pd.to_numeric(joined["building_height_m"], errors="coerce")
        matched_heights.index = normalized.index
        normalized["building_height_m"] = pd.to_numeric(normalized.get("building_height_m"), errors="coerce")
        normalized["building_height_m"] = normalized["building_height_m"].fillna(matched_heights)
        non_null_heights = pd.to_numeric(normalized.get("building_height_m", pd.Series(dtype=float)), errors="coerce").notna().mean()
        status["source"] = "db+osm_height_backfill"
        status["fetched_from_osm"] = True
        status["height_coverage_ratio"] = float(non_null_heights) if pd.notna(non_null_heights) else 0.0
        return normalized, status

    osm_buildings = _fetch_osm_layer("buildings", polygon_gdf, BUILDING_TAGS, cache_dir)
    osm_buildings = _normalize_building_height_gdf(osm_buildings)
    status["source"] = "osm"
    status["fetched_from_osm"] = not osm_buildings.empty
    if not osm_buildings.empty:
        non_null_heights = pd.to_numeric(osm_buildings.get("building_height_m", pd.Series(dtype=float)), errors="coerce").notna().mean()
        status["height_coverage_ratio"] = float(non_null_heights) if pd.notna(non_null_heights) else 0.0
    return osm_buildings, status


def _advanced_geo_required_columns() -> Dict[str, List[str]]:
    return {
        "site_context": [
            "serving_distance_m",
            "azimuth_delta_deg",
            "best_interferer_distance_m",
            "best_interferer_azimuth_delta_deg",
            "serving_proxy_rsrp_dbm",
            "best_interferer_proxy_rsrp_dbm",
            "interference_gap_db",
            "interference_ratio_linear",
        ],
        "building": [
            "los_blocker_count",
            "los_blocked_length_m",
            "los_blocked_ratio",
            "mean_blocker_height_m",
            "max_blocker_height_m",
            "nlos_flag",
            "diffraction_proxy_db",
        ],
        "terrain": [
            "terrain_elevation_m",
            "terrain_slope_deg",
            "proxy_site_elevation_m",
            "terrain_relief_to_site_m",
        ],
    }


def _columns_missing_or_empty(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for col in columns:
        if col not in df.columns:
            missing.append(col)
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            missing.append(col)
    return missing


def _validate_advanced_geo_requirements(
    grid_df: pd.DataFrame,
    advanced_geo_status: Dict[str, object],
    require_advanced_geo_on_miss: bool,
) -> None:
    if not require_advanced_geo_on_miss:
        return

    required = _advanced_geo_required_columns()
    site_missing = _columns_missing_or_empty(grid_df, required["site_context"])
    building_missing = _columns_missing_or_empty(grid_df, required["building"])
    terrain_missing = _columns_missing_or_empty(grid_df, required["terrain"])

    if site_missing:
        raise ValueError(
            f"Advanced geo feature generation failed for site context columns: {site_missing}"
        )

    building_status = advanced_geo_status.get("building_source_status") or {}
    if building_status.get("source") == "osm" and not building_status.get("fetched_from_osm"):
        raise ValueError(
            "Advanced geo feature generation failed: no building source was available from DB cache or OSM "
            f"for LOS/NLOS enrichment. building_source_status={building_status}"
        )
    if building_missing:
        raise ValueError(
            f"Advanced geo feature generation failed for building/LOS columns: {building_missing}. "
            f"building_source_status={building_status}"
        )

    dem_status = advanced_geo_status.get("dem_status") or {}
    if terrain_missing:
        raise ValueError(
            f"Advanced geo feature generation failed for terrain columns: {terrain_missing}. "
            f"Provide --dem-raster-path or reuse a cache that already contains terrain features. "
            f"dem_status={dem_status}"
        )


def _augment_grid_with_advanced_geo_features(
    grid_df: pd.DataFrame,
    building_gdf: gpd.GeoDataFrame,
    site_df: pd.DataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    cache_dir: Path,
    project_id: int,
    dem_raster_path: Optional[Path | str] = None,
    terrain_api_url: str = DEFAULT_TERRAIN_API_URL,
    terrain_api_batch_size: int = DEFAULT_TERRAIN_API_BATCH_SIZE,
    terrain_sample_step_m: float = DEFAULT_TERRAIN_SAMPLE_STEP_M,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    out = grid_df.copy()
    status: Dict[str, object] = {
        "site_context_refreshed": False,
        "building_path_enriched": False,
        "building_source_status": None,
        "dem_status": None,
        "new_columns_added": [],
    }

    before_cols = set(out.columns)
    out = _attach_site_context_features(out, site_df)
    status["site_context_refreshed"] = True

    building_needed = any(col not in before_cols for col in [
        "los_blocker_count",
        "los_blocked_length_m",
        "los_blocked_ratio",
        "mean_blocker_height_m",
        "max_blocker_height_m",
        "nlos_flag",
        "diffraction_proxy_db",
    ]) or bool(_columns_missing_or_empty(out, _advanced_geo_required_columns()["building"]))
    if building_needed:
        building_source_gdf, building_source_status = _ensure_required_building_source(
            building_gdf,
            polygon_gdf,
            cache_dir,
        )
        status["building_source_status"] = building_source_status
        out = _attach_building_path_features(out, building_source_gdf)
        status["building_path_enriched"] = True

    terrain_missing_before = bool(_columns_missing_or_empty(out, _advanced_geo_required_columns()["terrain"]))
    out, dem_status = _attach_dem_features(out, dem_raster_path)
    terrain_missing_after_raster = bool(_columns_missing_or_empty(out, _advanced_geo_required_columns()["terrain"]))
    if terrain_missing_before and terrain_missing_after_raster:
        out, dem_status = _attach_terrain_features_from_remote_api(
            out,
            cache_dir=cache_dir,
            project_id=project_id,
            api_url=terrain_api_url,
            batch_size=terrain_api_batch_size,
            sample_step_m=terrain_sample_step_m,
        )
    status["dem_status"] = dem_status
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
    status["new_columns_added"] = sorted(set(out.columns) - before_cols)
    return out, status


def _empty_gdf(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _fetch_osm_layer(
    name: str,
    polygon_gdf: gpd.GeoDataFrame,
    tags: Dict,
    cache_dir: Path,
) -> gpd.GeoDataFrame:
    cache_path = cache_dir / f"{name}.geojson"
    if cache_path.exists():
        return gpd.read_file(cache_path)

    ox.settings.timeout = 120
    ox.settings.use_cache = True
    geom = polygon_gdf.geometry.union_all()
    try:
        gdf = ox.features_from_polygon(geom, tags=tags)
    except Exception as exc:
        print(f"[TEST][OSM] layer={name} skipped reason={exc}")
        return _empty_gdf()

    if gdf.empty:
        return _empty_gdf()

    gdf = gdf.reset_index(drop=True)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(cache_path, driver="GeoJSON")
    return gdf


def _attach_line_density(grid_gdf: gpd.GeoDataFrame, line_gdf: gpd.GeoDataFrame, out_col: str) -> gpd.GeoDataFrame:
    grid_utm = grid_gdf.to_crs(_choose_utm_crs(grid_gdf))
    grid_utm[out_col] = 0.0
    if line_gdf.empty:
        return grid_utm.to_crs("EPSG:4326")

    line_utm = line_gdf.to_crs(grid_utm.crs)
    line_utm = line_utm[line_utm.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    if line_utm.empty:
        return grid_utm.to_crs("EPSG:4326")

    clipped = gpd.overlay(
        grid_utm[["grid_id", "geometry"]],
        line_utm[["geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if clipped.empty:
        return grid_utm.to_crs("EPSG:4326")

    clipped[out_col] = clipped.geometry.length
    agg = clipped.groupby("grid_id")[out_col].sum().rename(f"{out_col}_calc").reset_index()
    grid_utm = grid_utm.merge(agg, on="grid_id", how="left")
    grid_utm[out_col] = pd.to_numeric(grid_utm[f"{out_col}_calc"], errors="coerce").fillna(0.0)
    grid_utm = grid_utm.drop(columns=[f"{out_col}_calc"], errors="ignore")
    return grid_utm.to_crs("EPSG:4326")


def _attach_polygon_area_ratio(grid_gdf: gpd.GeoDataFrame, poly_gdf: gpd.GeoDataFrame, out_col: str) -> gpd.GeoDataFrame:
    grid_utm = grid_gdf.to_crs(_choose_utm_crs(grid_gdf))
    grid_utm[out_col] = 0.0
    if poly_gdf.empty:
        return grid_utm.to_crs("EPSG:4326")

    poly_utm = poly_gdf.to_crs(grid_utm.crs)
    poly_utm = poly_utm[poly_utm.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if poly_utm.empty:
        return grid_utm.to_crs("EPSG:4326")

    clipped = gpd.overlay(
        grid_utm[["grid_id", "geometry"]],
        poly_utm[["geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if clipped.empty:
        return grid_utm.to_crs("EPSG:4326")

    clipped["_area_m2"] = clipped.geometry.area
    agg = clipped.groupby("grid_id")["_area_m2"].sum().reset_index()
    grid_utm = grid_utm.merge(agg, on="grid_id", how="left")
    grid_utm["_area_m2"] = grid_utm["_area_m2"].fillna(0)
    grid_utm[out_col] = (
        grid_utm["_area_m2"] / grid_utm["cell_area_m2"].replace(0, np.nan)
    ).fillna(0)
    grid_utm = grid_utm.drop(columns=["_area_m2"])
    return grid_utm.to_crs("EPSG:4326")


def _derive_clutter_class(df: pd.DataFrame) -> pd.Series:
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


def _fit_morphology_clusters(grid_df: pd.DataFrame, cluster_count: int) -> Tuple[pd.DataFrame, Optional[KMeans], Optional[StandardScaler]]:
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
        series = work[col] if col in work.columns else pd.Series(0.0, index=work.index, dtype=float)
        work[col] = pd.to_numeric(series, errors="coerce").fillna(0.0)

    usable = work[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    if usable.empty or len(usable) < 2:
        work["morphology_cluster"] = 0
        return work, None, None

    distinct_rows = int(usable.drop_duplicates().shape[0])
    if distinct_rows <= 1:
        print("[TEST][CLUSTER] feature table is constant; assigning a single morphology cluster")
        work["morphology_cluster"] = 0
        return work, None, None

    n_clusters = max(2, min(cluster_count, len(usable), distinct_rows))
    print(
        f"[TEST][CLUSTER] requested_clusters={cluster_count} "
        f"usable_rows={len(usable)} distinct_feature_rows={distinct_rows} "
        f"effective_clusters={n_clusters}"
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(usable)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    work["morphology_cluster"] = model.fit_predict(X)
    return work, model, scaler


def _build_grid_feature_frame(
    grid_gdf: gpd.GeoDataFrame,
    site_df: pd.DataFrame,
    cluster_count: int,
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, Dict[str, Dict[str, float]]]:
    grid_df = pd.DataFrame(grid_gdf.drop(columns="geometry"))
    grid_centroids = grid_gdf.to_crs(_choose_utm_crs(grid_gdf)).copy()
    grid_centroids["geometry"] = grid_centroids.geometry.centroid
    grid_centroids = grid_centroids.to_crs("EPSG:4326")
    grid_centroids["lat"] = grid_centroids.geometry.y
    grid_centroids["lon"] = grid_centroids.geometry.x

    grid_df["lat"] = grid_centroids["lat"].values
    grid_df["lon"] = grid_centroids["lon"].values
    grid_site_context = _attach_site_context_features(
        grid_centroids[["grid_id", "lat", "lon"]],
        site_df,
    ).drop(columns=["lat", "lon"], errors="ignore")
    grid_df = grid_df.merge(grid_site_context, on="grid_id", how="left")
    grid_df["clutter_class"] = _derive_clutter_class(grid_df)
    grid_df, _, _ = _fit_morphology_clusters(grid_df, cluster_count)
    feature_stats = _feature_diagnostics(grid_df)
    return grid_df, grid_centroids, feature_stats


def _run_post_rf_smoke_test(
    pred_df: pd.DataFrame,
    drive_df: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
    grid_df: Optional[pd.DataFrame] = None,
) -> None:
    smoke_pred = pred_df.head(min(len(pred_df), 3000)).copy()
    smoke_holdout = drive_df.head(min(len(drive_df), 500)).copy()
    smoke_pred = _assign_points_to_tiles(smoke_pred, grid_gdf)
    if grid_df is not None:
        smoke_pred = _attach_missing_grid_features_by_grid_id(smoke_pred, grid_df)
    smoke_pred, _ = _apply_experimental_geo_adjustments(smoke_pred)
    _evaluate_prediction_grid_against_holdout(smoke_holdout, smoke_pred)
    required_cols = {"pred_rsrp_geo", "pred_rsrq_geo", "pred_sinr_geo", "grid_id"}
    missing = sorted(required_cols.difference(smoke_pred.columns))
    if missing:
        raise ValueError(f"Post-RF smoke test missing expected columns: {missing}")
    print(
        f"[TEST][SMOKE] post_rf_pipeline_ok pred_rows={len(smoke_pred)} "
        f"holdout_rows={len(smoke_holdout)}"
    )


def _run_post_rf_integrity_checks(
    pred_df: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
    grid_df: pd.DataFrame,
    holdout_eval: pd.DataFrame,
) -> None:
    if grid_gdf.empty:
        raise ValueError("Post-RF integrity check failed: analysis grid is empty")
    if grid_df.empty:
        raise ValueError("Post-RF integrity check failed: analysis grid feature frame is empty")
    if pred_df.empty:
        raise ValueError("Post-RF integrity check failed: prediction grid is empty")

    required_pred_cols = {
        "lat",
        "lon",
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "grid_id",
        "pred_rsrp_geo",
        "pred_rsrq_geo",
        "pred_sinr_geo",
    }
    missing_pred_cols = sorted(required_pred_cols.difference(pred_df.columns))
    if missing_pred_cols:
        raise ValueError(f"Post-RF integrity check missing prediction columns: {missing_pred_cols}")

    required_grid_cols = {"grid_id", "lat", "lon", "clutter_class", "morphology_cluster"}
    missing_grid_cols = sorted(required_grid_cols.difference(grid_df.columns))
    if missing_grid_cols:
        raise ValueError(f"Post-RF integrity check missing grid columns: {missing_grid_cols}")

    if pred_df[["lat", "lon"]].isna().any().any():
        raise ValueError("Post-RF integrity check failed: prediction coordinates contain nulls")
    if grid_df[["lat", "lon"]].isna().any().any():
        raise ValueError("Post-RF integrity check failed: grid centroid coordinates contain nulls")
    if grid_df["grid_id"].duplicated().any():
        raise ValueError("Post-RF integrity check failed: duplicate grid_id values in grid_df")
    if "grid_id" in pred_df.columns and pred_df["grid_id"].isna().all():
        raise ValueError("Post-RF integrity check failed: all prediction rows are missing grid_id")

    expected_holdout_cols = {"RSRP_pred", "RSRP_pred_geo"}
    missing_holdout_cols = sorted(expected_holdout_cols.difference(holdout_eval.columns))
    if missing_holdout_cols:
        raise ValueError(f"Post-RF integrity check missing holdout columns: {missing_holdout_cols}")

    print(
        f"[TEST][SMOKE] integrity_ok pred_rows={len(pred_df)} "
        f"grid_rows={len(grid_df)} holdout_rows={len(holdout_eval)}"
    )


def _run_artifact_write_smoke(
    run_dir: Path,
    pred_df: pd.DataFrame,
    holdout_eval: pd.DataFrame,
    grid_df: pd.DataFrame,
) -> None:
    smoke_dir = _ensure_dir(run_dir / "smoke_artifacts")
    pred_sample = _safe_sample(pred_df, limit=1000)
    holdout_sample = holdout_eval.head(min(len(holdout_eval), 300)).copy()
    grid_sample = grid_df.head(min(len(grid_df), 300)).copy()

    csv_path = smoke_dir / "pred_sample.csv"
    parquet_path = smoke_dir / "pred_sample.parquet"
    holdout_path = smoke_dir / "holdout_sample.csv"
    grid_path = smoke_dir / "grid_sample.csv"

    pred_sample.to_csv(csv_path, index=False)
    holdout_sample.to_csv(holdout_path, index=False)
    grid_sample.to_csv(grid_path, index=False)
    pred_sample.to_parquet(parquet_path, index=False)

    csv_reload = pd.read_csv(csv_path)
    parquet_reload = pd.read_parquet(parquet_path)
    if len(csv_reload) != len(pred_sample):
        raise ValueError("Artifact smoke failed: CSV round-trip row count mismatch")
    if len(parquet_reload) != len(pred_sample):
        raise ValueError("Artifact smoke failed: Parquet round-trip row count mismatch")

    print(
        f"[TEST][SMOKE] artifact_write_ok csv_rows={len(csv_reload)} "
        f"parquet_rows={len(parquet_reload)}"
    )


def _assign_points_to_tiles(points_df: pd.DataFrame, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    points = points_df.copy()
    grid_cols = [col for col in grid_gdf.columns if col != "geometry"]
    overlap_cols = [col for col in grid_cols if col in points.columns]
    if overlap_cols:
        points = points.drop(columns=overlap_cols, errors="ignore")
    point_gdf = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(point_gdf, grid_gdf, how="left", predicate="within")

    missing = joined["grid_id"].isna()
    if missing.any():
        utm_crs = _choose_utm_crs(grid_gdf)
        point_missing_utm = point_gdf.loc[missing, ["geometry"]].to_crs(utm_crs)
        grid_utm = grid_gdf.to_crs(utm_crs)
        nearest = gpd.sjoin_nearest(
            point_missing_utm,
            grid_utm,
            how="left",
            distance_col="_tile_distance",
        ).to_crs("EPSG:4326")
        for col in grid_gdf.columns:
            if col == "geometry":
                continue
            joined.loc[missing, col] = nearest[col].values

    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    return pd.DataFrame(joined)


def _attach_site_context_features(points_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    points = points_df.copy()
    if points.empty or site_df.empty or not {"lat", "lon"}.issubset(points.columns):
        return points

    site_work = site_df.copy()
    for col in ["lat", "lon", "azimuth"]:
        if col in site_work.columns:
            site_work[col] = pd.to_numeric(site_work[col], errors="coerce")
    site_work = site_work.dropna(subset=["lat", "lon"]).copy()
    if site_work.empty:
        return points

    def _series_or_default(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index, dtype=float)

    def _string_series_or_default(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
        if col in frame.columns:
            return frame[col].fillna(default).astype(str).str.strip()
        return pd.Series(default, index=frame.index, dtype=object)

    def _out_numeric_series_or_default(col: str, default: float = np.nan) -> np.ndarray:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        return pd.Series(default, index=out.index, dtype=float).to_numpy(dtype=float)

    def _out_string_series_or_default(col: str, default: str = "") -> np.ndarray:
        if col in out.columns:
            return out[col].fillna(default).astype(str).str.strip().to_numpy(dtype=object)
        return pd.Series(default, index=out.index, dtype=object).to_numpy(dtype=object)

    point_lat = pd.to_numeric(points["lat"], errors="coerce").to_numpy(dtype=float)
    point_lon = pd.to_numeric(points["lon"], errors="coerce").to_numpy(dtype=float)
    site_lat = site_work["lat"].to_numpy(dtype=float)
    site_lon = site_work["lon"].to_numpy(dtype=float)

    point_rad = np.radians(np.c_[point_lat, point_lon])
    site_rad = np.radians(np.c_[site_lat, site_lon])
    tree = BallTree(site_rad, metric="haversine")
    k = min(4, len(site_work))
    dist_rad, _ = tree.query(point_rad, k=k)
    _, idx = tree.query(point_rad, k=k)
    earth_radius_m = 6371000.0
    dist_m = dist_rad * earth_radius_m
    points["nearest_site_distance_m"] = dist_m[:, 0]
    points["mean_nearest3_site_distance_m"] = dist_m.mean(axis=1)
    points["site_count_250m"] = np.array([len(x) for x in tree.query_radius(point_rad, r=250.0 / earth_radius_m)])
    points["site_count_500m"] = np.array([len(x) for x in tree.query_radius(point_rad, r=500.0 / earth_radius_m)])

    nearest_rows = site_work.iloc[idx[:, 0]].reset_index(drop=True)
    points["_proxy_site_id"] = nearest_rows["Node_Cell_ID"].astype(str).values if "Node_Cell_ID" in nearest_rows.columns else ""
    points["_proxy_site_lat"] = pd.to_numeric(nearest_rows["lat"], errors="coerce").values
    points["_proxy_site_lon"] = pd.to_numeric(nearest_rows["lon"], errors="coerce").values
    points["_proxy_site_azimuth"] = _series_or_default(nearest_rows, "azimuth", 0).values
    points["_proxy_site_height_m"] = _series_or_default(nearest_rows, "antenna_height", 30).values
    points["_proxy_site_tx_power"] = _series_or_default(nearest_rows, "tx_power", 46).values
    if "frequency_mhz" in nearest_rows.columns:
        points["_proxy_site_frequency_mhz"] = _series_or_default(nearest_rows, "frequency_mhz", 1800).values
    else:
        points["_proxy_site_frequency_mhz"] = _series_or_default(nearest_rows, "frequency", 1800).values
    points["_proxy_site_pci"] = _string_series_or_default(nearest_rows, "pci").values
    points["_proxy_site_earfcn"] = _string_series_or_default(nearest_rows, "earfcn").values
    points["_proxy_site_etilt"] = _series_or_default(nearest_rows, "electrical_tilt", 3).values
    points["_proxy_site_mtilt"] = _series_or_default(nearest_rows, "mechanical_tilt", 0).values
    points["serving_pci"] = points["_proxy_site_pci"]
    points["serving_earfcn"] = points["_proxy_site_earfcn"]
    points["serving_frequency_mhz"] = pd.to_numeric(points["_proxy_site_frequency_mhz"], errors="coerce")

    proxy_bearing = _bearing_deg_np(
        points["_proxy_site_lat"].to_numpy(dtype=float),
        points["_proxy_site_lon"].to_numpy(dtype=float),
        point_lat,
        point_lon,
    )
    points["serving_distance_m"] = dist_m[:, 0]
    points["azimuth_delta_deg"] = np.abs((proxy_bearing - points["_proxy_site_azimuth"].to_numpy(dtype=float) + 180.0) % 360.0 - 180.0)
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
    points["best_interferer_cell_id"] = ""
    points["best_interferer_pci"] = ""
    points["best_interferer_earfcn"] = ""
    for neighbor_rank in (1, 2):
        points[f"neighbor_{neighbor_rank}_cell_id"] = ""
        points[f"neighbor_{neighbor_rank}_pci"] = ""
        points[f"neighbor_{neighbor_rank}_earfcn"] = ""
        points[f"neighbor_{neighbor_rank}_proxy_rsrp_dbm"] = np.nan
        points[f"neighbor_{neighbor_rank}_distance_m"] = np.nan
        points[f"neighbor_{neighbor_rank}_azimuth_delta_deg"] = np.nan

    if k >= 2:
        interferer_rows = site_work.iloc[idx[:, 1]].reset_index(drop=True)
        points["best_interferer_distance_m"] = dist_m[:, 1]
        interferer_azimuth = _series_or_default(interferer_rows, "azimuth", 0).to_numpy(dtype=float)
        interferer_height = _series_or_default(interferer_rows, "antenna_height", 30).to_numpy(dtype=float)
        interferer_tx = _series_or_default(interferer_rows, "tx_power", 46).to_numpy(dtype=float)
        if "frequency_mhz" in interferer_rows.columns:
            interferer_freq = _series_or_default(interferer_rows, "frequency_mhz", 1800).to_numpy(dtype=float)
        else:
            interferer_freq = _series_or_default(interferer_rows, "frequency", 1800).to_numpy(dtype=float)
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

    # Build a stronger forward-style proxy from multiple nearby sectors.
    site_freq_all = (
        _series_or_default(site_work, "frequency_mhz", 1800).to_numpy(dtype=float)
        if "frequency_mhz" in site_work.columns
        else _series_or_default(site_work, "frequency", 1800).to_numpy(dtype=float)
    )
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

    if "Node_Cell_ID" in points.columns and "Node_Cell_ID" in site_work.columns:
        serving_site = (
            site_work.sort_values("Node_Cell_ID")
            .drop_duplicates(subset=["Node_Cell_ID"], keep="first")
            [["Node_Cell_ID", "lat", "lon"] + ([ "azimuth"] if "azimuth" in site_work.columns else [])]
            .rename(columns={"lat": "serving_lat", "lon": "serving_lon", "azimuth": "serving_azimuth"})
        )
        points["Node_Cell_ID"] = points["Node_Cell_ID"].astype(str)
        serving_site["Node_Cell_ID"] = serving_site["Node_Cell_ID"].astype(str)
        points = points.merge(serving_site, on="Node_Cell_ID", how="left")
        has_serving = points["serving_lat"].notna() & points["serving_lon"].notna()
        if has_serving.any():
            src_lat = pd.to_numeric(points.loc[has_serving, "serving_lat"], errors="coerce").to_numpy(dtype=float)
            src_lon = pd.to_numeric(points.loc[has_serving, "serving_lon"], errors="coerce").to_numpy(dtype=float)
            dst_lat = pd.to_numeric(points.loc[has_serving, "lat"], errors="coerce").to_numpy(dtype=float)
            dst_lon = pd.to_numeric(points.loc[has_serving, "lon"], errors="coerce").to_numpy(dtype=float)
            phi1 = np.radians(src_lat)
            phi2 = np.radians(dst_lat)
            dphi = np.radians(dst_lat - src_lat)
            dlambda = np.radians(dst_lon - src_lon)
            a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
            points.loc[has_serving, "serving_distance_m"] = 2.0 * earth_radius_m * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
            if "serving_azimuth" in points.columns:
                y = np.sin(np.radians(dst_lon - src_lon)) * np.cos(np.radians(dst_lat))
                x = (
                    np.cos(np.radians(src_lat)) * np.sin(np.radians(dst_lat))
                    - np.sin(np.radians(src_lat)) * np.cos(np.radians(dst_lat)) * np.cos(np.radians(dst_lon - src_lon))
                )
                bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
                points.loc[has_serving, "azimuth_delta_deg"] = _safe_angle_delta_deg(
                    pd.Series(bearing, index=points.index[has_serving]),
                    pd.to_numeric(points.loc[has_serving, "serving_azimuth"], errors="coerce"),
                )
        points = points.drop(columns=["serving_lat", "serving_lon", "serving_azimuth"], errors="ignore")
    return points


def _attach_fixed_serving_sinr_rsrq_proxy(
    points_df: pd.DataFrame,
    site_df: pd.DataFrame,
    max_interferers: int = 18,
    min_distance_m: float = DEFAULT_SINR_MIN_DISTANCE_M,
) -> pd.DataFrame:
    out = points_df.copy()
    required_point_cols = {"lat", "lon", "Node_Cell_ID"}
    required_site_cols = {"lat", "lon", "Node_Cell_ID"}
    if out.empty or site_df.empty or not required_point_cols.issubset(out.columns):
        return out
    if not required_site_cols.issubset(site_df.columns):
        return out

    site_work = site_df.copy()
    if "canonical_sector_id" not in site_work.columns:
        site_work = _add_sector_identity_columns(site_work, use_as_node_cell_id=False)
    site_work, canonical_dedup_summary = _deduplicate_site_df_for_rf_matching(site_work)
    print(f"[TEST][SINR_CANONICAL_SITE_DEDUP] {canonical_dedup_summary}")
    for col in ["lat", "lon", "azimuth"]:
        if col in site_work.columns:
            site_work[col] = pd.to_numeric(site_work[col], errors="coerce")
    site_work["Node_Cell_ID"] = site_work["Node_Cell_ID"].astype(str).str.strip()
    site_work["canonical_sector_id"] = site_work["canonical_sector_id"].astype(str).str.strip()
    site_work = site_work.dropna(subset=["lat", "lon"]).copy()
    if site_work.empty:
        return out

    def _series_or_default(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index, dtype=float)

    def _string_series_or_default(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
        if col in frame.columns:
            return frame[col].fillna(default).astype(str).str.strip()
        return pd.Series(default, index=frame.index, dtype=object)

    def _out_numeric_series_or_default(col: str, default: float = np.nan) -> np.ndarray:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        return pd.Series(default, index=out.index, dtype=float).to_numpy(dtype=float)

    def _out_string_series_or_default(col: str, default: str = "") -> np.ndarray:
        if col in out.columns:
            return out[col].fillna(default).astype(str).str.strip().to_numpy(dtype=object)
        return pd.Series(default, index=out.index, dtype=object).to_numpy(dtype=object)

    serving_sites = (
        site_work.sort_values(["canonical_sector_id", "Node_Cell_ID"])
        .drop_duplicates(subset=["canonical_sector_id"], keep="first")
        .reset_index(drop=True)
    )
    if serving_sites.empty:
        return out

    serving_lookup = {
        cell_id: idx
        for idx, cell_id in enumerate(serving_sites["canonical_sector_id"].astype(str).str.strip().tolist())
    }
    site_lat = serving_sites["lat"].to_numpy(dtype=float)
    site_lon = serving_sites["lon"].to_numpy(dtype=float)
    site_az = _series_or_default(serving_sites, "azimuth", 0.0).to_numpy(dtype=float)
    site_height = _series_or_default(serving_sites, "antenna_height", 30.0).to_numpy(dtype=float)
    site_tx = _series_or_default(serving_sites, "tx_power", 46.0).to_numpy(dtype=float)
    if "frequency_mhz" in serving_sites.columns:
        site_freq = _series_or_default(serving_sites, "frequency_mhz", 1800.0).to_numpy(dtype=float)
    else:
        site_freq = _series_or_default(serving_sites, "frequency", 1800.0).to_numpy(dtype=float)
    site_pci = _string_series_or_default(serving_sites, "pci").to_numpy(dtype=object)
    site_earfcn = _string_series_or_default(serving_sites, "earfcn").to_numpy(dtype=object)
    site_canonical = _string_series_or_default(serving_sites, "canonical_sector_id").to_numpy(dtype=object)
    site_etilt = _series_or_default(serving_sites, "electrical_tilt", 3.0).to_numpy(dtype=float)
    site_mtilt = _series_or_default(serving_sites, "mechanical_tilt", 0.0).to_numpy(dtype=float)

    point_lat = pd.to_numeric(out["lat"], errors="coerce").to_numpy(dtype=float)
    point_lon = pd.to_numeric(out["lon"], errors="coerce").to_numpy(dtype=float)
    if "canonical_sector_id" not in out.columns:
        point_identity = pd.DataFrame(
            {
                "Node_Cell_ID": out["Node_Cell_ID"],
                "cell_id": out["Node_Cell_ID"],
            },
            index=out.index,
        )
        point_identity = _add_sector_identity_columns(point_identity, use_as_node_cell_id=False)
        out["canonical_sector_id"] = point_identity["canonical_sector_id"].to_numpy(dtype=object)
    point_cells = out["canonical_sector_id"].astype(str).str.strip().to_numpy(dtype=object)
    point_rad = np.radians(np.c_[point_lat, point_lon])
    site_rad = np.radians(np.c_[site_lat, site_lon])
    tree = BallTree(site_rad, metric="haversine")
    candidate_k = max(1, min(int(max_interferers), len(serving_sites)))
    _, candidate_idx = tree.query(point_rad, k=candidate_k)
    if candidate_idx.ndim == 1:
        candidate_idx = candidate_idx.reshape(-1, 1)

    sinr_proxy_db = _out_numeric_series_or_default("sinr_proxy_db")
    rsrq_proxy_db = _out_numeric_series_or_default("rsrq_proxy_db")
    best_interferer_proxy = _out_numeric_series_or_default("best_interferer_proxy_phys_dbm")
    best_interferer_distance = _out_numeric_series_or_default("best_interferer_distance_m")
    best_interferer_az_delta = _out_numeric_series_or_default("best_interferer_azimuth_delta_deg")
    interference_sum_dbm = _out_numeric_series_or_default("interference_sum_proxy_dbm")
    interference_gap_db = _out_numeric_series_or_default("interference_gap_db")
    interference_ratio = _out_numeric_series_or_default("interference_ratio_linear")
    serving_pci = _out_string_series_or_default("serving_pci")
    serving_earfcn = _out_string_series_or_default("serving_earfcn")
    serving_frequency = _out_numeric_series_or_default("serving_frequency_mhz")
    best_interferer_cell_id = _out_string_series_or_default("best_interferer_cell_id")
    best_interferer_pci = _out_string_series_or_default("best_interferer_pci")
    best_interferer_earfcn = _out_string_series_or_default("best_interferer_earfcn")
    neighbor_1_cell_id = _out_string_series_or_default("neighbor_1_cell_id")
    neighbor_1_pci = _out_string_series_or_default("neighbor_1_pci")
    neighbor_1_earfcn = _out_string_series_or_default("neighbor_1_earfcn")
    neighbor_1_rsrp = _out_numeric_series_or_default("neighbor_1_proxy_rsrp_dbm")
    neighbor_1_distance = _out_numeric_series_or_default("neighbor_1_distance_m")
    neighbor_1_az_delta = _out_numeric_series_or_default("neighbor_1_azimuth_delta_deg")
    neighbor_2_cell_id = _out_string_series_or_default("neighbor_2_cell_id")
    neighbor_2_pci = _out_string_series_or_default("neighbor_2_pci")
    neighbor_2_earfcn = _out_string_series_or_default("neighbor_2_earfcn")
    neighbor_2_rsrp = _out_numeric_series_or_default("neighbor_2_proxy_rsrp_dbm")
    neighbor_2_distance = _out_numeric_series_or_default("neighbor_2_distance_m")
    neighbor_2_az_delta = _out_numeric_series_or_default("neighbor_2_azimuth_delta_deg")

    noise_linear = 10 ** (-104.0 / 10.0)
    n_rb = 50.0
    updated_rows = 0
    dominance_window_db = 15.0
    strong_dominance_window_db = 12.0
    strongest_interferer_cap = 3
    same_earfcn_interferer_count = np.zeros(len(out), dtype=float)
    dominant_interferer_count = np.zeros(len(out), dtype=float)
    interference_selection_mode = np.full(len(out), "", dtype=object)
    clutter_labels = (
        out["clutter_class"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "clutter_class" in out.columns
        else np.full(len(out), "", dtype=object)
    )
    row_nlos_flag = _out_numeric_series_or_default("nlos_flag", default=0.0)
    row_los_blocked_ratio = _out_numeric_series_or_default("los_blocked_ratio", default=0.0)
    row_diffraction_proxy_db = _out_numeric_series_or_default("diffraction_proxy_db", default=0.0)

    for row_idx, cell_id in enumerate(point_cells):
        serving_idx = serving_lookup.get(str(cell_id))
        if serving_idx is None or not np.isfinite(point_lat[row_idx]) or not np.isfinite(point_lon[row_idx]):
            continue

        local_candidates = np.unique(candidate_idx[row_idx]).astype(int).tolist()
        if serving_idx not in local_candidates:
            local_candidates.append(serving_idx)

        candidate_arr = np.array(local_candidates, dtype=int)
        cand_lat = site_lat[candidate_arr]
        cand_lon = site_lon[candidate_arr]
        cand_az = site_az[candidate_arr]
        cand_height = site_height[candidate_arr]
        cand_tx = site_tx[candidate_arr]
        cand_freq = site_freq[candidate_arr]
        cand_etilt = site_etilt[candidate_arr]
        cand_mtilt = site_mtilt[candidate_arr]
        rsrp_all = _compute_proxy_rsrp_arrays(
            np.full(len(candidate_arr), point_lat[row_idx], dtype=float),
            np.full(len(candidate_arr), point_lon[row_idx], dtype=float),
            cand_lat,
            cand_lon,
            cand_az,
            cand_height,
            cand_tx,
            cand_freq,
            cand_etilt,
            cand_mtilt,
            min_distance_m=min_distance_m,
        )
        serving_mask = candidate_arr == serving_idx
        if not serving_mask.any():
            continue
        clutter_label = str(clutter_labels[row_idx] or "").strip()
        base_interferer_loss_db = {
            "Dense Urban": 6.5,
            "Urban": 3.5,
            "Suburban": 1.5,
            "Vegetation": 2.0,
            "Water": 0.5,
            "Rural/Open": 0.5,
        }.get(clutter_label, 2.5)
        serving_pci[row_idx] = str(site_pci[serving_idx] or "").strip()
        serving_earfcn[row_idx] = str(site_earfcn[serving_idx] or "").strip()
        serving_frequency[row_idx] = float(site_freq[serving_idx]) if np.isfinite(site_freq[serving_idx]) else np.nan
        serving_rsrp = float(rsrp_all[serving_mask][0])
        non_serving_mask = ~serving_mask
        rsrp_interference_all = rsrp_all.copy()
        selection_mode = f"nearest_fallback_clutter_{clutter_label or 'unknown'}"
        serving_earfcn_value = str(site_earfcn[serving_idx] or "").strip()
        if non_serving_mask.any():
            non_serving_idx = np.flatnonzero(non_serving_mask)
            non_serving_candidate_idx = candidate_arr[non_serving_idx]
            non_serving_distance = np.array(
                [
                    _haversine_m_np(
                        site_lat[int(site_idx)],
                        site_lon[int(site_idx)],
                        point_lat[row_idx],
                        point_lon[row_idx],
                    )
                    for site_idx in non_serving_candidate_idx
                ],
                dtype=float,
            )
            non_serving_bearing = np.array(
                [
                    _bearing_deg_np(
                        site_lat[int(site_idx)],
                        site_lon[int(site_idx)],
                        point_lat[row_idx],
                        point_lon[row_idx],
                    )
                    for site_idx in non_serving_candidate_idx
                ],
                dtype=float,
            )
            non_serving_az_delta = np.abs(
                (non_serving_bearing - site_az[non_serving_candidate_idx] + 180.0) % 360.0 - 180.0
            )
            angle_loss_db = np.where(
                non_serving_az_delta <= 25.0,
                0.0,
                np.where(
                    non_serving_az_delta >= 120.0,
                    14.0 + 0.05 * (non_serving_az_delta.clip(120.0, 180.0) - 120.0),
                    np.where(
                        non_serving_az_delta <= 60.0,
                        6.0 * ((non_serving_az_delta - 25.0) / 35.0),
                        6.0 + 8.0 * ((non_serving_az_delta - 60.0) / 60.0),
                    ),
                ),
            )
            distance_loss_db = np.where(
                non_serving_distance <= 100.0,
                0.0,
                np.where(
                    non_serving_distance >= 450.0,
                    8.0,
                    np.minimum(8.0, (non_serving_distance - 100.0) / 55.0),
                ),
            )
            blockage_loss_db = (
                2.0 * float(np.clip(row_nlos_flag[row_idx], 0.0, 1.0))
                + 4.0 * float(np.clip(row_los_blocked_ratio[row_idx], 0.0, 1.0))
                + 0.10 * float(np.clip(row_diffraction_proxy_db[row_idx], 0.0, 20.0))
            )
            total_interferer_loss_db = (
                float(base_interferer_loss_db)
                + angle_loss_db
                + distance_loss_db
                + blockage_loss_db
            )
            total_interferer_loss_db = np.clip(total_interferer_loss_db, 0.0, 18.0)
            rsrp_interference_all[non_serving_idx] = rsrp_interference_all[non_serving_idx] - total_interferer_loss_db
            selection_mode = f"{selection_mode}_suppressed"

        linear = np.power(10.0, rsrp_interference_all / 10.0)
        serving_linear = float(np.sum(linear[serving_mask]))
        eligible_mask = non_serving_mask.copy()
        if serving_earfcn_value:
            same_earfcn_mask = np.array(
                [str(site_earfcn[idx] or "").strip() == serving_earfcn_value for idx in candidate_arr],
                dtype=bool,
            )
            if (non_serving_mask & same_earfcn_mask).any():
                eligible_mask = non_serving_mask & same_earfcn_mask
                selection_mode = "same_earfcn_only"
            else:
                eligible_mask = np.zeros(len(candidate_arr), dtype=bool)
                selection_mode = "no_same_earfcn_interferer"
        elif np.isfinite(site_freq[serving_idx]):
            same_freq_mask = np.isclose(cand_freq, site_freq[serving_idx], atol=0.5)
            if (non_serving_mask & same_freq_mask).any():
                eligible_mask = non_serving_mask & same_freq_mask
                selection_mode = "same_frequency_only"

        strict_geometry_mask = np.zeros(len(candidate_arr), dtype=bool)
        if non_serving_mask.any():
            strict_geometry_mask = (
                (non_serving_distance <= 420.0)
                & (non_serving_az_delta <= 95.0)
            )
            strict_geometry_mask_full = np.zeros(len(candidate_arr), dtype=bool)
            strict_geometry_mask_full[np.flatnonzero(non_serving_mask)] = strict_geometry_mask
        else:
            strict_geometry_mask_full = np.zeros(len(candidate_arr), dtype=bool)

        strong_dominance_mask = rsrp_interference_all >= (serving_rsrp - strong_dominance_window_db)
        within_dominance_mask = rsrp_interference_all >= (serving_rsrp - dominance_window_db)
        if (eligible_mask & strict_geometry_mask_full & strong_dominance_mask).any():
            eligible_mask = eligible_mask & strict_geometry_mask_full & strong_dominance_mask
            selection_mode = f"{selection_mode}_geo_dom12"
        elif (eligible_mask & strict_geometry_mask_full & within_dominance_mask).any():
            eligible_mask = eligible_mask & strict_geometry_mask_full & within_dominance_mask
            selection_mode = f"{selection_mode}_geo_dom15"
        elif (eligible_mask & within_dominance_mask).any():
            eligible_mask = eligible_mask & within_dominance_mask
            if selection_mode:
                selection_mode = f"{selection_mode}_dominant"
        elif eligible_mask.any():
            selection_mode = f"{selection_mode}_all"

        eligible_idx = np.flatnonzero(eligible_mask)
        if eligible_idx.size > strongest_interferer_cap:
            strongest_order = np.argsort(rsrp_interference_all[eligible_idx])[::-1][:strongest_interferer_cap]
            eligible_idx = eligible_idx[strongest_order]
            selection_mode = f"{selection_mode}_top{strongest_interferer_cap}"

        interference_linear_raw = 0.0
        if eligible_idx.size:
            interference_linear_raw = float(np.sum(linear[eligible_idx]))
        interference_linear = max(interference_linear_raw + noise_linear, noise_linear)
        if serving_linear <= 0.0:
            continue
        same_earfcn_interferer_count[row_idx] = float(np.sum(eligible_mask))
        dominant_interferer_count[row_idx] = float(len(eligible_idx))
        interference_selection_mode[row_idx] = selection_mode

        sinr_proxy_db[row_idx] = float(10.0 * np.log10(serving_linear / interference_linear))
        rssi_linear = max(serving_linear + interference_linear_raw + noise_linear, noise_linear)
        rsrq_proxy_db[row_idx] = float(serving_rsrp - (10.0 * np.log10(rssi_linear)) + 10.0 * np.log10(n_rb))
        interference_sum_dbm[row_idx] = float(10.0 * np.log10(interference_linear))
        best_interferer_proxy[row_idx] = np.nan
        best_interferer_distance[row_idx] = np.nan
        best_interferer_az_delta[row_idx] = np.nan
        best_interferer_cell_id[row_idx] = ""
        best_interferer_pci[row_idx] = ""
        best_interferer_earfcn[row_idx] = ""
        neighbor_1_cell_id[row_idx] = ""
        neighbor_1_pci[row_idx] = ""
        neighbor_1_earfcn[row_idx] = ""
        neighbor_1_rsrp[row_idx] = np.nan
        neighbor_1_distance[row_idx] = np.nan
        neighbor_1_az_delta[row_idx] = np.nan
        neighbor_2_cell_id[row_idx] = ""
        neighbor_2_pci[row_idx] = ""
        neighbor_2_earfcn[row_idx] = ""
        neighbor_2_rsrp[row_idx] = np.nan
        neighbor_2_distance[row_idx] = np.nan
        neighbor_2_az_delta[row_idx] = np.nan
        interference_gap_db[row_idx] = np.nan
        interference_ratio[row_idx] = np.nan

        if eligible_idx.size:
            eligible_rsrp = rsrp_interference_all[eligible_idx]
            eligible_rank = np.argsort(eligible_rsrp)[::-1]
            ranked_idx = eligible_idx[eligible_rank]
            top_two_idx = ranked_idx[:2]
            for neighbor_rank, local_idx in enumerate(top_two_idx, start=1):
                interferer_candidate_idx = int(candidate_arr[local_idx])
                interferer_cell_id = str(serving_sites.iloc[interferer_candidate_idx]["Node_Cell_ID"]).strip()
                interferer_distance = float(
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
                interferer_az_delta = float(
                    abs((interferer_bearing - site_az[interferer_candidate_idx] + 180.0) % 360.0 - 180.0)
                )
                interferer_pci = str(site_pci[interferer_candidate_idx] or "").strip()
                interferer_earfcn = str(site_earfcn[interferer_candidate_idx] or "").strip()
                interferer_rsrp = float(rsrp_interference_all[local_idx])
                if neighbor_rank == 1:
                    best_interferer_proxy[row_idx] = interferer_rsrp
                    best_interferer_distance[row_idx] = interferer_distance
                    best_interferer_az_delta[row_idx] = interferer_az_delta
                    best_interferer_cell_id[row_idx] = interferer_cell_id
                    best_interferer_pci[row_idx] = interferer_pci
                    best_interferer_earfcn[row_idx] = interferer_earfcn
                    neighbor_1_cell_id[row_idx] = interferer_cell_id
                    neighbor_1_pci[row_idx] = interferer_pci
                    neighbor_1_earfcn[row_idx] = interferer_earfcn
                    neighbor_1_rsrp[row_idx] = interferer_rsrp
                    neighbor_1_distance[row_idx] = interferer_distance
                    neighbor_1_az_delta[row_idx] = interferer_az_delta
                    interference_gap_db[row_idx] = serving_rsrp - interferer_rsrp
                    interference_ratio[row_idx] = float(np.power(10.0, (interferer_rsrp - serving_rsrp) / 10.0))
                else:
                    neighbor_2_cell_id[row_idx] = interferer_cell_id
                    neighbor_2_pci[row_idx] = interferer_pci
                    neighbor_2_earfcn[row_idx] = interferer_earfcn
                    neighbor_2_rsrp[row_idx] = interferer_rsrp
                    neighbor_2_distance[row_idx] = interferer_distance
                    neighbor_2_az_delta[row_idx] = interferer_az_delta
        updated_rows += 1

    out["sinr_proxy_db"] = sinr_proxy_db
    out["rsrq_proxy_db"] = rsrq_proxy_db
    out["best_interferer_proxy_phys_dbm"] = best_interferer_proxy
    out["best_interferer_cell_id"] = best_interferer_cell_id
    out["best_interferer_pci"] = best_interferer_pci
    out["best_interferer_earfcn"] = best_interferer_earfcn
    out["best_interferer_distance_m"] = best_interferer_distance
    out["best_interferer_azimuth_delta_deg"] = best_interferer_az_delta
    out["interference_sum_proxy_dbm"] = interference_sum_dbm
    out["interference_gap_db"] = interference_gap_db
    out["interference_ratio_linear"] = interference_ratio
    out["serving_pci"] = serving_pci
    out["serving_earfcn"] = serving_earfcn
    out["serving_frequency_mhz"] = serving_frequency
    out["neighbor_1_cell_id"] = neighbor_1_cell_id
    out["neighbor_1_pci"] = neighbor_1_pci
    out["neighbor_1_earfcn"] = neighbor_1_earfcn
    out["neighbor_1_proxy_rsrp_dbm"] = neighbor_1_rsrp
    out["neighbor_1_distance_m"] = neighbor_1_distance
    out["neighbor_1_azimuth_delta_deg"] = neighbor_1_az_delta
    out["neighbor_2_cell_id"] = neighbor_2_cell_id
    out["neighbor_2_pci"] = neighbor_2_pci
    out["neighbor_2_earfcn"] = neighbor_2_earfcn
    out["neighbor_2_proxy_rsrp_dbm"] = neighbor_2_rsrp
    out["neighbor_2_distance_m"] = neighbor_2_distance
    out["neighbor_2_azimuth_delta_deg"] = neighbor_2_az_delta
    out["same_earfcn_interferer_count"] = same_earfcn_interferer_count
    out["dominant_interferer_count"] = dominant_interferer_count
    out["interference_selection_mode"] = interference_selection_mode
    print(
        f"[TEST][SINR_FIX] fixed_serving_rows={updated_rows} "
        f"candidate_pool={candidate_k} top_interferers={strongest_interferer_cap} "
        f"dominance_window_db={dominance_window_db:.1f} total_rows={len(out)}"
    )
    return out


def _resolve_validation_sessions(session_ids: Iterable[int]) -> Tuple[int, ...]:
    session_ids = tuple(int(session_id) for session_id in session_ids)
    if not session_ids:
        raise ValueError("At least one DT session is required for validation.")
    return session_ids


def _prepare_drive_measurements(drive_df: pd.DataFrame) -> pd.DataFrame:
    dt = drive_df.dropna(subset=["lat", "lon"]).copy()
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
    dt = dt.dropna(subset=["RSRP_meas"]).copy()
    return dt


def _split_drive_train_holdout(
    drive_df: pd.DataFrame,
    validation_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dt = _prepare_drive_measurements(drive_df)
    if dt.empty:
        return dt.copy(), dt.copy()

    holdout_frac = float(np.clip(validation_fraction, 0.1, 0.5))
    if len(dt) < 40:
        holdout_size = max(1, int(round(len(dt) * holdout_frac)))
        return dt.iloc[holdout_size:].copy(), dt.iloc[:holdout_size].copy()

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
    return train_df, holdout_df


def _attach_prediction_grid_to_points(points_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    points = points_df.copy()
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
        "serving_proxy_rsrp_phys_dbm",
        "rsrq_proxy_db_raw",
        "rsrq_proxy_db",
        "sinr_proxy_db_raw",
        "sinr_proxy_db",
        "coverage_offset",
        "sinr_structural_offset",
    ]
    pred_keep_cols = [col for col in keep_cols if col in pred_df.columns]
    preds = pred_df[pred_keep_cols].dropna(subset=["lat", "lon"]).copy()
    if points.empty or preds.empty:
        print(
            f"[TEST][GRID_MATCH] skipped points_empty={points.empty} preds_empty={preds.empty} "
            f"point_rows={len(points)} pred_rows={len(preds)}"
        )
        return points

    points_gdf = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs="EPSG:4326",
    )
    preds_gdf = gpd.GeoDataFrame(
        preds,
        geometry=gpd.points_from_xy(preds["lon"], preds["lat"]),
        crs="EPSG:4326",
    )
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
    out = pd.DataFrame(joined)
    expected_cols = ["pred_rsrp", "pred_rsrq", "pred_sinr", "morphology_cluster"]
    missing_cols = [col for col in expected_cols if col not in out.columns]
    if missing_cols:
        print(
            f"[TEST][GRID_MATCH] missing_prediction_columns={missing_cols} "
            f"joined_columns={list(out.columns)}"
        )
    else:
        print(
            f"[TEST][GRID_MATCH] matched_points={len(out)} "
            f"pred_rsrp_non_null={int(out['pred_rsrp'].notna().sum())} "
            f"cluster_non_null={int(out['morphology_cluster'].notna().sum())}"
        )
    return out


def _build_serving_view_prediction_grid(pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if pred_df.empty or not {"lat", "lon"}.issubset(pred_df.columns):
        return pred_df.copy(), {
            "enabled": False,
            "reason": "missing_prediction_coordinates",
            "input_rows": int(len(pred_df)),
            "output_rows": int(len(pred_df)),
        }
    work = pred_df.dropna(subset=["lat", "lon"]).copy()
    if work.empty:
        return work, {
            "enabled": False,
            "reason": "empty_prediction_coordinates",
            "input_rows": int(len(pred_df)),
            "output_rows": 0,
        }
    if "serving_proxy_rsrp_phys_dbm" in work.columns:
        rank_col = "serving_proxy_rsrp_phys_dbm"
    elif "pred_rsrp_geo" in work.columns:
        rank_col = "pred_rsrp_geo"
    else:
        rank_col = "pred_rsrp"
    work["_serving_view_rank_rsrp"] = pd.to_numeric(work.get(rank_col), errors="coerce")
    work["_serving_view_rank_rsrp"] = work["_serving_view_rank_rsrp"].fillna(-999.0)
    work["_serving_view_original_order"] = np.arange(len(work), dtype=int)
    serving_view = (
        work.sort_values(
            ["lat", "lon", "_serving_view_rank_rsrp", "_serving_view_original_order"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(subset=["lat", "lon"], keep="first")
        .drop(columns=["_serving_view_rank_rsrp", "_serving_view_original_order"], errors="ignore")
        .copy()
    )
    rows_per_location = work.groupby(["lat", "lon"], dropna=False).size()
    summary = {
        "enabled": True,
        "rank_col": rank_col,
        "input_rows": int(len(pred_df)),
        "usable_rows": int(len(work)),
        "output_rows": int(len(serving_view)),
        "unique_locations": int(len(rows_per_location)),
        "mean_rows_per_location": round(float(rows_per_location.mean()), 4) if len(rows_per_location) else 0.0,
        "p50_rows_per_location": round(float(rows_per_location.quantile(0.50)), 4) if len(rows_per_location) else 0.0,
        "max_rows_per_location": int(rows_per_location.max()) if len(rows_per_location) else 0,
        "dropped_non_serving_rows": int(len(work) - len(serving_view)),
    }
    return serving_view.reset_index(drop=True), summary


def _sinr_distribution_summary(df: pd.DataFrame, pred_col: str = "pred_sinr_geo") -> Dict[str, object]:
    if df.empty or pred_col not in df.columns:
        return {"available": False, "column": pred_col, "rows": int(len(df))}
    values = pd.to_numeric(df[pred_col], errors="coerce").dropna()
    if values.empty:
        return {"available": False, "column": pred_col, "rows": int(len(df)), "non_null": 0}
    return {
        "available": True,
        "column": pred_col,
        "rows": int(len(df)),
        "non_null": int(len(values)),
        "mean": round(float(values.mean()), 4),
        "min": round(float(values.min()), 4),
        "p10": round(float(values.quantile(0.10)), 4),
        "p50": round(float(values.quantile(0.50)), 4),
        "p90": round(float(values.quantile(0.90)), 4),
        "max": round(float(values.max()), 4),
        "lt_0_count": int((values < 0.0).sum()),
        "lt_5_count": int((values < 5.0).sum()),
        "lt_6_count": int((values < 6.0).sum()),
        "gte_6_count": int((values >= 6.0).sum()),
    }


def _apply_demo_dt_overlay(
    pred_df: pd.DataFrame,
    drive_df: pd.DataFrame,
    replace_radius_m: float = 20.0,
    blend_sigma_m: float = 60.0,
    blend_radius_m: float = 140.0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    pred_out = pred_df.copy()
    dt = _prepare_drive_measurements(drive_df)
    required_pred_cols = {"lat", "lon"}
    if pred_out.empty or dt.empty or not required_pred_cols.issubset(pred_out.columns):
        return pred_out, {
            "enabled": False,
            "reason": "missing_prediction_or_dt_points",
            "pred_rows": len(pred_out),
            "dt_rows": len(dt),
        }

    pred_points = pred_out.dropna(subset=["lat", "lon"]).copy()
    if pred_points.empty:
        return pred_out, {
            "enabled": False,
            "reason": "prediction_coordinates_missing",
            "pred_rows": len(pred_out),
            "dt_rows": len(dt),
        }

    pred_gdf = gpd.GeoDataFrame(
        pred_points,
        geometry=gpd.points_from_xy(pred_points["lon"], pred_points["lat"]),
        crs="EPSG:4326",
    )
    dt_gdf = gpd.GeoDataFrame(
        dt,
        geometry=gpd.points_from_xy(dt["lon"], dt["lat"]),
        crs="EPSG:4326",
    )
    utm_crs = _choose_utm_crs(pred_gdf)
    pred_utm = pred_gdf.to_crs(utm_crs)
    dt_utm = dt_gdf.to_crs(utm_crs)

    pred_coords = np.c_[pred_utm.geometry.x.to_numpy(dtype=float), pred_utm.geometry.y.to_numpy(dtype=float)]
    dt_coords = np.c_[dt_utm.geometry.x.to_numpy(dtype=float), dt_utm.geometry.y.to_numpy(dtype=float)]
    if len(pred_coords) == 0 or len(dt_coords) == 0:
        return pred_out, {
            "enabled": False,
            "reason": "empty_metric_coordinate_frame",
            "pred_rows": len(pred_out),
            "dt_rows": len(dt),
        }

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
        ("SINR_meas", "pred_sinr_geo" if "pred_sinr_geo" in pred_out.columns else "pred_sinr", "pred_sinr_demo", -25.0, 30.0),
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
        blended_vals = np.clip(blended_vals, clip_min, clip_max)
        pred_out.loc[pred_points.index, out_col] = blended_vals

        anchored_indices: List[int] = []
        anchored_values: List[float] = []
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

    summary = {
        "enabled": True,
        "replace_radius_m": float(replace_radius_m),
        "blend_sigma_m": float(blend_sigma_m),
        "blend_radius_m": float(blend_radius_m),
        "pred_rows": int(len(pred_out)),
        "dt_rows": int(len(dt)),
        "anchor_cells": int(pred_out["demo_dt_anchor"].sum()),
        "blended_cells": int((pred_out["demo_blend_weight"] > 0.0).sum()),
        "visual_source_counts": pred_out["demo_visual_source"].astype(str).value_counts(dropna=False).to_dict(),
    }
    print(
        f"[TEST][DEMO_OVERLAY] anchor_cells={summary['anchor_cells']} "
        f"blended_cells={summary['blended_cells']} replace_radius_m={replace_radius_m} "
        f"blend_sigma_m={blend_sigma_m} blend_radius_m={blend_radius_m}"
    )
    return pred_out, summary


def _evaluate_prediction_grid_against_holdout(
    holdout_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    offsets_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    holdout = _prepare_drive_measurements(holdout_df)
    holdout = _attach_prediction_grid_to_points(holdout, pred_df)

    baseline_metrics = {}
    metric_specs = [
        ("RSRP_meas", "pred_rsrp"),
        ("RSRQ_meas", "pred_rsrq"),
        ("SINR_meas", "pred_sinr"),
    ]
    rename_map = {
        "pred_rsrp": "RSRP_pred",
        "pred_rsrq": "RSRQ_pred",
        "pred_sinr": "SINR_pred",
        "pred_rsrp_geo": "RSRP_pred_geo",
        "pred_rsrq_geo": "RSRQ_pred_geo",
        "pred_sinr_geo": "SINR_pred_geo",
    }
    for src_col, out_col in rename_map.items():
        if src_col in holdout.columns and out_col not in holdout.columns:
            holdout[out_col] = holdout[src_col]

    for meas_col, base_col in metric_specs:
        if meas_col in holdout.columns and base_col in holdout.columns:
            valid = holdout.dropna(subset=[meas_col, base_col])
            if not valid.empty:
                baseline_metrics[meas_col] = _metric_bundle(valid[meas_col], valid[base_col], metric_key=meas_col)
    experimental_metric_specs = [
        ("RSRP_meas", "pred_rsrp_geo"),
        ("RSRQ_meas", "pred_rsrq_geo"),
        ("SINR_meas", "pred_sinr_geo"),
    ]
    experimental_metrics = {}
    for meas_col, exp_col in experimental_metric_specs:
        if meas_col in holdout.columns and exp_col in holdout.columns:
            valid = holdout.dropna(subset=[meas_col, exp_col])
            if not valid.empty:
                experimental_metrics[meas_col] = _metric_bundle(valid[meas_col], valid[exp_col], metric_key=meas_col)
    return holdout, baseline_metrics, experimental_metrics


def _build_experimental_feature_frame(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    pred_col = {
        "RSRP": "pred_rsrp",
        "RSRQ": "pred_rsrq",
        "SINR": "pred_sinr",
    }[metric_name]
    work = df.copy()
    feature_cols = [
        pred_col,
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "grid_match_distance_m",
        "serving_distance_m",
        "azimuth_delta_deg",
        "serving_proxy_rsrp_phys_dbm",
        "best_interferer_proxy_phys_dbm",
        "sinr_proxy_db",
        "rsrq_proxy_db",
        "effective_tx_height_m",
        "best_interferer_distance_m",
        "interference_gap_db",
        "los_blocker_count",
        "los_blocked_ratio",
        "diffraction_proxy_db",
        "terrain_elevation_m",
        "terrain_slope_deg",
        "terrain_relief_to_site_m",
    ]
    available_numeric = [col for col in feature_cols if col in work.columns]
    available_categorical = [col for col in ["clutter_class", "morphology_cluster"] if col in work.columns]
    if not available_numeric and not available_categorical:
        return pd.DataFrame(index=work.index)

    numeric = work[available_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0) if available_numeric else pd.DataFrame(index=work.index)
    categorical = pd.get_dummies(work[available_categorical].fillna("missing").astype(str), prefix=available_categorical) if available_categorical else pd.DataFrame(index=work.index)
    features = pd.concat([numeric, categorical], axis=1)
    return features.replace([np.inf, -np.inf], 0).fillna(0.0)


def _fit_train_only_residual_calibration(
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
        features = _build_experimental_feature_frame(valid, metric_name)
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

        clutter_bias: Dict[str, float] = {}
        if "clutter_class" in valid.columns:
            clutter_bias = (
                valid.assign(_residual=residual)
                .groupby("clutter_class", dropna=False)["_residual"]
                .median()
                .clip(lower=low_clip / 2.0, upper=high_clip / 2.0)
                .to_dict()
            )

        model_bundle[metric_name] = {
            "metric_name": metric_name,
            "pred_col": pred_col,
            "scaler": scaler,
            "model": ridge,
            "feature_columns": features.columns.tolist(),
            "low_clip": float(low_clip),
            "high_clip": float(high_clip),
            "clutter_bias": clutter_bias,
        }
        debug["models"][metric_name] = {
            "used": True,
            "rows": int(len(valid)),
            "feature_count": int(features.shape[1]),
            "residual_mae": round(float(np.abs(residual).mean()), 4),
            "residual_bias": round(float(residual.mean()), 4),
            "clutter_bias_classes": int(len(clutter_bias)),
            "bad_interference_uplift_cap": metric_name == "SINR",
        }

    debug["enabled"] = bool(model_bundle)
    return model_bundle, debug


def _apply_bad_interference_sinr_uplift_cap(
    df: pd.DataFrame,
    calibrated_sinr: pd.Series,
) -> pd.Series:
    def _num(col: str, default: float = np.nan) -> pd.Series:
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    raw_anchor = _num("sinr_proxy_db_raw").combine_first(_num("sinr_proxy_db")).combine_first(_num("pred_sinr"))
    raw_anchor = raw_anchor.fillna(pd.to_numeric(calibrated_sinr, errors="coerce")).clip(lower=-25.0, upper=30.0)
    interference_gap = _num("interference_gap_db", 6.0).fillna(6.0)
    site_density = _num("site_count_250m", 0.0).fillna(0.0)
    los_blocked = _num("los_blocked_ratio", 0.0).fillna(0.0)
    nlos = _num("nlos_flag", 0.0).fillna(0.0)
    best_interferer_distance = _num("best_interferer_distance_m", 250.0).fillna(250.0)

    low_raw_sinr = raw_anchor <= -2.0
    weak_server_gap = interference_gap <= 3.0
    dense_or_blocked = (
        (site_density >= 15.0)
        | (los_blocked >= 0.25)
        | (nlos >= 0.5)
        | (best_interferer_distance <= 150.0)
    )
    bad_interference = (low_raw_sinr & weak_server_gap) | (low_raw_sinr & dense_or_blocked) | (weak_server_gap & dense_or_blocked & (raw_anchor <= 2.0))

    capped = np.minimum(calibrated_sinr, raw_anchor)
    return calibrated_sinr.where(~bad_interference, capped).clip(lower=-25.0, upper=30.0)


def _apply_train_only_residual_calibration(
    pred_df: pd.DataFrame,
    model_bundle: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    if pred_df.empty or not model_bundle:
        return pred_df

    out = pred_df.copy()
    metric_ranges = {
        "RSRP": (-140.0, -44.0),
        "RSRQ": (-20.0, -3.0),
        "SINR": (-25.0, 30.0),
    }
    for metric_name, bundle in model_bundle.items():
        pred_col = str(bundle["pred_col"])
        if pred_col not in out.columns:
            continue
        features = _build_experimental_feature_frame(out, metric_name)
        feature_columns = list(bundle["feature_columns"])
        for col in feature_columns:
            if col not in features.columns:
                features[col] = 0.0
        features = features.reindex(columns=feature_columns, fill_value=0.0)
        x_scaled = bundle["scaler"].transform(features)
        residual_pred = pd.Series(bundle["model"].predict(x_scaled), index=out.index, dtype=float)
        residual_pred = residual_pred.clip(lower=float(bundle["low_clip"]), upper=float(bundle["high_clip"]))
        clutter_bias = bundle.get("clutter_bias") or {}
        if clutter_bias and "clutter_class" in out.columns:
            residual_pred = residual_pred + (
                out["clutter_class"].astype(str).map(clutter_bias).fillna(0.0) * 0.35
            )
        clip_min, clip_max = metric_ranges[metric_name]
        base_values = pd.to_numeric(out[pred_col], errors="coerce").fillna(0.0)
        out[pred_col] = (base_values + residual_pred).clip(lower=clip_min, upper=clip_max)
    return out


def _geo_offsets_from_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    work = df.copy()
    for col in [
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "morphology_cluster",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "azimuth_delta_deg",
        "best_interferer_distance_m",
        "best_interferer_azimuth_delta_deg",
        "serving_proxy_rsrp_dbm",
        "best_interferer_proxy_rsrp_dbm",
        "serving_proxy_rsrp_phys_dbm",
        "best_interferer_proxy_phys_dbm",
        "interference_gap_db",
        "interference_ratio_linear",
        "interference_sum_proxy_dbm",
        "sinr_proxy_db",
        "rsrq_proxy_db",
        "effective_tx_height_m",
        "los_blocker_count",
        "los_blocked_length_m",
        "los_blocked_ratio",
        "mean_blocker_height_m",
        "max_blocker_height_m",
        "nlos_flag",
        "diffraction_proxy_db",
        "terrain_elevation_m",
        "terrain_slope_deg",
        "proxy_site_elevation_m",
        "terrain_relief_to_site_m",
    ]:
        series = work[col] if col in work.columns else pd.Series(0.0, index=work.index, dtype=float)
        work[col] = pd.to_numeric(series, errors="coerce").fillna(0.0)

    clutter_penalty = pd.Series(0.0, index=work.index)
    if "clutter_class" in work.columns:
        clutter_penalty = work["clutter_class"].astype(str).map(
            {
                "Dense Urban": -5.748911279913842,
                "Urban": -4.570568952812136,
                "Suburban": -1.899177902584371,
                "Vegetation": -3.9552529876524924,
                "Water": 2.4531907113373754,
                "Rural/Open": -0.33255324361378014,
            }
        ).fillna(0.0)

    cluster_center = work["morphology_cluster"].mean() if len(work) else 0.0
    cluster_offset = (work["morphology_cluster"] - cluster_center) * 0.4297518853707454
    building_offset = (-12.964392127510385 * work["building_area_ratio"].clip(0, 0.75)) + (-0.15073959763804912 * work["building_count"].clip(0, 25))
    road_offset = -0.004980672169259599 * work["road_length_m"].clip(0, 400)
    green_water_offset = (-2.3779716792919743 * work["green_ratio"].clip(0, 1.0)) + (2.592578657419375 * work["water_ratio"].clip(0, 1.0))
    size_offset = 0.0003519269280539811 * work["avg_building_area_m2"].clip(0, 3000)
    site_density_offset = (0.32517669019050915 * work["site_count_250m"].clip(0, 12)) + (-0.08856885322089737 * work["site_count_500m"].clip(0, 25))
    serving_distance_offset = -0.0010098019128563871 * work["serving_distance_m"].clip(0, 1200)
    nearest_site_offset = -0.003260265433476443 * work["nearest_site_distance_m"].clip(0, 1000)
    boresight_offset = -0.0006302280734634055 * (work["azimuth_delta_deg"].clip(0, 180) / 10.0) ** 1.2
    isolation_offset = 0.0005536357270057354 * work["mean_nearest3_site_distance_m"].clip(0, 1500)
    dense_urban_far_penalty = np.where(
        (work.get("clutter_class", pd.Series("", index=work.index)).astype(str) == "Dense Urban")
        & (work["nearest_site_distance_m"] > 180.0),
        -3.3180945597164664 - 0.008115694031989687 * (work["nearest_site_distance_m"].clip(180.0, 700.0) - 180.0),
        0.0,
    )
    urban_off_axis_penalty = np.where(
        work["azimuth_delta_deg"] > 45.0,
        -0.039504018296481806 * (work["azimuth_delta_deg"].clip(45.0, 180.0) - 45.0),
        0.0,
    )
    far_serving_off_axis_penalty = np.where(
        (work["serving_distance_m"] > 250.0) & (work["azimuth_delta_deg"] > 35.0),
        -2.6321855291869034
        + 5.422069609443102e-05 * (work["serving_distance_m"].clip(250.0, 1200.0) - 250.0)
        - 0.005284872039560948 * (work["azimuth_delta_deg"].clip(35.0, 180.0) - 35.0),
        0.0,
    )
    high_building_far_penalty = np.where(
        (work["avg_building_area_m2"] > 250.0) & (work["nearest_site_distance_m"] > 160.0),
        -0.09061887674784153
        - 0.0035465731549538214 * (work["avg_building_area_m2"].clip(250.0, 3000.0) - 250.0)
        - 0.0037821231931904706 * (work["nearest_site_distance_m"].clip(160.0, 1000.0) - 160.0),
        0.0,
    )
    vegetation_far_penalty = np.where(
        (work.get("clutter_class", pd.Series("", index=work.index)).astype(str) == "Vegetation")
        & (work["serving_distance_m"] > 220.0),
        -1.3571238149230584 - 1.5567623240241666 * work["green_ratio"].clip(0.2, 1.0),
        0.0,
    )
    water_open_bonus = np.where(
        (
            work.get("clutter_class", pd.Series("", index=work.index)).astype(str).isin(["Water", "Rural/Open"])
            & (work["azimuth_delta_deg"] < 20.0)
            & (work["nearest_site_distance_m"] < 220.0)
        ),
        0.4613646566903654 + (-1.5741148296497583e-05) * (220.0 - work["nearest_site_distance_m"].clip(0.0, 220.0)),
        0.0,
    )
    dense_site_bonus = np.where(
        (work["site_count_250m"] >= 4.0) & (work["nearest_site_distance_m"] < 120.0),
        1.2215359060737925 + (-0.004170459532614926) * work["site_count_250m"].clip(4.0, 12.0),
        0.0,
    )
    cluster_dense_urban_penalty = np.where(
        (work["morphology_cluster"] >= (cluster_center + 1.0))
        & (work.get("clutter_class", pd.Series("", index=work.index)).astype(str) == "Dense Urban"),
        -1.5925996890836496 - 0.2190252819471583 * (work["morphology_cluster"] - cluster_center).clip(lower=0.0, upper=4.0),
        0.0,
    )
    nlos_penalty = -1.082159675021917 * work["nlos_flag"].clip(0, 1)
    blocker_penalty_raw = (
        0.024825012042379857 * work["los_blocker_count"].clip(0, 10)
        - 2.6902531306299293 * work["los_blocked_ratio"].clip(0, 1.0)
        - 0.04969984402837008 * work["max_blocker_height_m"].clip(0, 80.0)
    )
    blocker_penalty = blocker_penalty_raw.clip(lower=-3.772306016024911, upper=0.0)
    diffraction_penalty = (-0.3069439421687738 * work["diffraction_proxy_db"].clip(0, 25.0)).clip(lower=-2.362916825053845, upper=0.0)
    terrain_penalty_raw = (
        -0.048179164367668385 * work["terrain_slope_deg"].clip(0, 35.0)
        - 0.026085764830383558 * work["terrain_relief_to_site_m"].clip(lower=0.0, upper=180.0)
    )
    terrain_penalty = pd.Series(terrain_penalty_raw, index=work.index, dtype=float).clip(lower=-2.4865370074364375, upper=0.0)
    interference_penalty = pd.Series(
        np.where(
            work["interference_gap_db"] < 4.388916682071409,
            -0.45533279144346844 * (4.388916682071409 - work["interference_gap_db"].clip(-12.0, 4.388916682071409)),
            0.18987445793696894 * (work["interference_gap_db"].clip(4.388916682071409, 15.043309431816219) - 4.388916682071409),
        ),
        index=work.index,
        dtype=float,
    ).clip(lower=-3.5, upper=2.0)
    # Avoid double-counting the same interference geometry through both gap and ratio.
    interference_ratio_support = pd.Series(0.0, index=work.index, dtype=float)
    combined_rf_penalty = (nlos_penalty + blocker_penalty + diffraction_penalty + terrain_penalty).clip(lower=-3.4116957929762926, upper=0.5)
    structural_geo_offset = (
        clutter_penalty
        + cluster_offset
        + building_offset
        + road_offset
        + green_water_offset
        + size_offset
        + site_density_offset
        + serving_distance_offset
        + nearest_site_offset
        + boresight_offset
        + isolation_offset
        + pd.Series(dense_urban_far_penalty, index=work.index, dtype=float)
        + pd.Series(urban_off_axis_penalty, index=work.index, dtype=float)
        + pd.Series(far_serving_off_axis_penalty, index=work.index, dtype=float)
        + pd.Series(high_building_far_penalty, index=work.index, dtype=float)
        + pd.Series(vegetation_far_penalty, index=work.index, dtype=float)
        + pd.Series(water_open_bonus, index=work.index, dtype=float)
        + pd.Series(dense_site_bonus, index=work.index, dtype=float)
        + pd.Series(cluster_dense_urban_penalty, index=work.index, dtype=float)
        + combined_rf_penalty
    )
    coverage_offset = structural_geo_offset.clip(lower=-10.0, upper=4.5) + (interference_penalty + interference_ratio_support).clip(lower=-3.0, upper=1.5)
    coverage_offset = coverage_offset.clip(lower=-11.0, upper=5.0)

    sinr_structural_offset = (
        (0.18 * site_density_offset)
        + (0.10 * serving_distance_offset)
        + (0.08 * nearest_site_offset)
        + (0.08 * boresight_offset)
        + (0.12 * interference_penalty)
        + (0.08 * combined_rf_penalty.clip(lower=-3.0, upper=0.5))
    ).clip(lower=-2.5, upper=1.5)
    return coverage_offset, sinr_structural_offset


def _default_geo_coefficients() -> Dict[str, Dict[str, float]]:
    return {
        "RSRP": {
            "intercept": 0.0,
            "phys_weight": 0.08972745122796283,
            "offset_weight": 0.930594270876904,
            "offset_only_weight": 1.0,
        },
        "RSRQ": {
            "intercept": 0.0,
            "phys_weight": 0.015271496882229606,
            "offset_weight": 0.5372747809311423,
            "offset_only_weight": 0.5960584217154146,
        },
        "SINR": {
            "intercept": 0.0,
            "phys_weight": 0.17841479695610893,
            "offset_weight": 0.6146672501874981,
            "offset_only_weight": 0.6097922919738781,
        },
    }


def _sinr_geo_context_features(
    df: pd.DataFrame,
    base: pd.Series,
    phys: pd.Series,
    offset: pd.Series,
) -> pd.DataFrame:
    def _num(col: str, default: float = 0.0) -> pd.Series:
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce").fillna(default)

    phys_filled = phys.combine_first(_num("sinr_proxy_db", np.nan)).combine_first(base).fillna(0.0)
    interference_gap = _num("interference_gap_db", 6.0)
    sinr_proxy = phys_filled
    site_density = _num("site_count_250m", 0.0)
    best_interferer_distance = _num("best_interferer_distance_m", 250.0)
    los_blocked_ratio = _num("los_blocked_ratio", 0.0).clip(lower=0.0, upper=1.5)
    nlos_flag = _num("nlos_flag", 0.0).clip(lower=0.0, upper=1.0)
    azimuth_delta = _num("azimuth_delta_deg", 0.0)
    serving_distance = _num("serving_distance_m", 0.0)
    rf_penalty = _num("diffraction_proxy_db", 0.0)

    poor_interference = ((6.0 - interference_gap) / 12.0).clip(lower=0.0, upper=1.0)
    clean_interference = ((interference_gap - 6.0) / 12.0).clip(lower=0.0, upper=1.0)
    low_proxy = ((3.0 - sinr_proxy) / 12.0).clip(lower=0.0, upper=1.0)
    good_proxy = ((sinr_proxy - 3.0) / 12.0).clip(lower=0.0, upper=1.0)
    dense_sites = ((site_density - 10.0) / 45.0).clip(lower=0.0, upper=1.0)
    close_interferer = ((160.0 - best_interferer_distance) / 160.0).clip(lower=0.0, upper=1.0)
    off_axis = ((azimuth_delta - 45.0) / 90.0).clip(lower=0.0, upper=1.0)
    far_serving = ((serving_distance - 180.0) / 220.0).clip(lower=0.0, upper=1.0)
    blockage = ((0.7 * los_blocked_ratio) + (0.3 * nlos_flag)).clip(lower=0.0, upper=1.0)

    features = pd.DataFrame(
        {
            "base_sinr": base,
            "phys_delta": phys_filled - base,
            "sinr_structural_offset": offset,
            "interference_gap_db": interference_gap,
            "sinr_proxy_db": sinr_proxy,
            "poor_interference": poor_interference,
            "clean_interference": clean_interference,
            "low_proxy": low_proxy,
            "good_proxy": good_proxy,
            "dense_sites": dense_sites,
            "close_interferer": close_interferer,
            "blockage": blockage,
            "off_axis": off_axis,
            "far_serving": far_serving,
            "rf_penalty": rf_penalty,
            "poor_x_low_proxy": poor_interference * low_proxy,
            "clean_x_good_proxy": clean_interference * good_proxy,
            "poor_x_dense": poor_interference * dense_sites,
            "poor_x_blockage": poor_interference * blockage,
        },
        index=df.index,
    )
    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _fit_scaled_ridge_payload(features: pd.DataFrame, target: pd.Series, alpha: float = 8.0) -> Dict[str, object]:
    feature_columns = features.columns.tolist()
    mean = features.mean(axis=0)
    scale = features.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    x_scaled = ((features - mean) / scale).to_numpy(dtype=float)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_scaled, target.to_numpy(dtype=float))
    return {
        "feature_columns": feature_columns,
        "mean": mean.to_list(),
        "scale": scale.to_list(),
        "coef": ridge.coef_.astype(float).tolist(),
        "intercept": float(ridge.intercept_),
    }


def _predict_scaled_ridge_payload(features: pd.DataFrame, payload: Dict[str, object]) -> pd.Series:
    columns = list(payload.get("feature_columns", []))
    if not columns:
        return pd.Series(0.0, index=features.index, dtype=float)
    work = features.copy()
    for col in columns:
        if col not in work.columns:
            work[col] = 0.0
    work = work.reindex(columns=columns, fill_value=0.0)
    mean = np.asarray(payload.get("mean", [0.0] * len(columns)), dtype=float)
    scale = np.asarray(payload.get("scale", [1.0] * len(columns)), dtype=float)
    scale = np.where(scale == 0.0, 1.0, scale)
    coef = np.asarray(payload.get("coef", [0.0] * len(columns)), dtype=float)
    x_scaled = (work.to_numpy(dtype=float) - mean) / scale
    pred = float(payload.get("intercept", 0.0)) + np.dot(x_scaled, coef)
    return pd.Series(pred, index=features.index, dtype=float)


def _fit_scaled_logistic_payload(features: pd.DataFrame, target: pd.Series) -> Dict[str, object]:
    feature_columns = features.columns.tolist()
    mean = features.mean(axis=0)
    scale = features.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    y = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
    if y.nunique(dropna=True) < 2:
        return {
            "feature_columns": feature_columns,
            "mean": mean.to_list(),
            "scale": scale.to_list(),
            "coef": [0.0] * len(feature_columns),
            "intercept": 20.0 if int(y.iloc[0]) == 1 else -20.0,
            "classes": sorted(y.unique().astype(int).tolist()),
            "single_class": int(y.iloc[0]),
        }
    x_scaled = ((features - mean) / scale).to_numpy(dtype=float)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(x_scaled, y.to_numpy(dtype=int))
    positive_idx = int(np.where(clf.classes_ == 1)[0][0])
    coef_idx = positive_idx if clf.coef_.shape[0] > 1 else 0
    coef = clf.coef_[coef_idx].astype(float).tolist()
    intercept = float(clf.intercept_[coef_idx])
    if clf.coef_.shape[0] == 1 and int(clf.classes_[-1]) != 1:
        coef = (-np.asarray(coef, dtype=float)).tolist()
        intercept = -intercept
    return {
        "feature_columns": feature_columns,
        "mean": mean.to_list(),
        "scale": scale.to_list(),
        "coef": coef,
        "intercept": intercept,
        "classes": clf.classes_.astype(int).tolist(),
        "single_class": None,
    }


def _predict_scaled_logistic_probability(features: pd.DataFrame, payload: Dict[str, object]) -> pd.Series:
    columns = list(payload.get("feature_columns", []))
    if not columns:
        return pd.Series(0.0, index=features.index, dtype=float)
    single_class = payload.get("single_class")
    if single_class is not None:
        return pd.Series(1.0 if int(single_class) == 1 else 0.0, index=features.index, dtype=float)
    work = features.copy()
    for col in columns:
        if col not in work.columns:
            work[col] = 0.0
    work = work.reindex(columns=columns, fill_value=0.0)
    mean = np.asarray(payload.get("mean", [0.0] * len(columns)), dtype=float)
    scale = np.asarray(payload.get("scale", [1.0] * len(columns)), dtype=float)
    scale = np.where(scale == 0.0, 1.0, scale)
    coef = np.asarray(payload.get("coef", [0.0] * len(columns)), dtype=float)
    logits = float(payload.get("intercept", 0.0)) + np.dot((work.to_numpy(dtype=float) - mean) / scale, coef)
    logits = np.clip(logits, -50.0, 50.0)
    probability = 1.0 / (1.0 + np.exp(-logits))
    return pd.Series(probability, index=features.index, dtype=float)


def _fit_train_only_geo_coefficients(train_eval: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, object]]:
    defaults = _default_geo_coefficients()
    debug: Dict[str, object] = {
        "enabled": False,
        "train_rows": int(len(train_eval)),
        "models": {},
    }
    if train_eval.empty:
        return defaults, debug

    specs = {
        "RSRP": {
            "meas_col": "RSRP_meas",
            "base_col": "pred_rsrp",
            "phys_col": "serving_proxy_rsrp_phys_dbm",
            "offset_col": "coverage_offset",
            "low_clip": -12.0,
            "high_clip": 12.0,
            "phys_bounds": (0.0, 0.50),
            "offset_bounds": (-1.50, 1.50),
        },
        "RSRQ": {
            "meas_col": "RSRQ_meas",
            "base_col": "pred_rsrq",
            "phys_col": "rsrq_proxy_db_raw",
            "fallback_phys_col": "rsrq_proxy_db",
            "offset_col": "coverage_offset",
            "low_clip": -8.0,
            "high_clip": 8.0,
            "phys_bounds": (0.0, 0.50),
            "offset_bounds": (-1.00, 1.00),
        },
        "SINR": {
            "meas_col": "SINR_meas",
            "base_col": "pred_sinr",
            "phys_col": "sinr_proxy_db_raw",
            "fallback_phys_col": "sinr_proxy_db",
            "offset_col": "sinr_structural_offset",
            "low_clip": -10.0,
            "high_clip": 10.0,
            "phys_bounds": (0.0, 0.50),
            "offset_bounds": (-1.00, 1.00),
        },
    }

    learned = {metric: values.copy() for metric, values in defaults.items()}
    for metric_name, spec in specs.items():
        required = [str(spec["meas_col"]), str(spec["base_col"]), str(spec["offset_col"])]
        if not set(required).issubset(train_eval.columns):
            debug["models"][metric_name] = {"used": False, "reason": "missing_required_columns"}
            continue

        work = train_eval.copy()
        base = pd.to_numeric(work[spec["base_col"]], errors="coerce")
        measured = pd.to_numeric(work[spec["meas_col"]], errors="coerce")
        offset = pd.to_numeric(work[spec["offset_col"]], errors="coerce")
        phys_col = str(spec["phys_col"])
        phys = pd.to_numeric(work[phys_col], errors="coerce") if phys_col in work.columns else pd.Series(np.nan, index=work.index)
        fallback_phys_col = spec.get("fallback_phys_col")
        if fallback_phys_col and str(fallback_phys_col) in work.columns:
            phys = phys.combine_first(pd.to_numeric(work[str(fallback_phys_col)], errors="coerce"))

        valid_mask = measured.notna() & base.notna() & offset.notna()
        valid_rows = int(valid_mask.sum())
        if valid_rows < 60:
            debug["models"][metric_name] = {
                "used": False,
                "reason": "train_rows_lt_60",
                "rows": valid_rows,
            }
            continue

        residual = (measured[valid_mask] - base[valid_mask]).clip(
            lower=float(spec["low_clip"]),
            upper=float(spec["high_clip"]),
        )
        phys_delta = (phys[valid_mask] - base[valid_mask]).fillna(0.0)
        offset_fit = offset[valid_mask].fillna(0.0)

        if metric_name == "SINR":
            sinr_features = _sinr_geo_context_features(
                work.loc[valid_mask],
                base.loc[valid_mask],
                phys.loc[valid_mask],
                offset.loc[valid_mask],
            )
            context_payload = _fit_scaled_ridge_payload(sinr_features, residual, alpha=8.0)
            train_pred = _predict_scaled_ridge_payload(sinr_features, context_payload).clip(
                lower=float(spec["low_clip"]),
                upper=float(spec["high_clip"]),
            )
            measured_sinr = pd.to_numeric(work.loc[valid_mask, spec["meas_col"]], errors="coerce")
            good_target = (measured_sinr >= 6.0).astype(int)
            good_classifier = _fit_scaled_logistic_payload(sinr_features, good_target)
            train_good_probability = _predict_scaled_logistic_probability(sinr_features, good_classifier)
            gated_train_pred = train_pred.where(train_pred <= 0.0, train_pred * train_good_probability)
            learned[metric_name] = {
                "intercept": 0.0,
                "phys_weight": 0.0,
                "offset_weight": 0.0,
                "offset_only_weight": 0.0,
                "context_model": context_payload,
                "good_sinr_classifier": good_classifier,
                "good_sinr_threshold": 6.0,
                "context_correction_clip": [float(spec["low_clip"]), float(spec["high_clip"])],
            }
            debug["models"][metric_name] = {
                "used": True,
                "mode": "classifier_gated_context_residual",
                "rows": valid_rows,
                "feature_count": len(context_payload.get("feature_columns", [])),
                "residual_bias": round(float(residual.mean()), 4),
                "residual_mae": round(float(np.abs(residual).mean()), 4),
                "context_train_pred_mean": round(float(train_pred.mean()), 4),
                "context_train_pred_min": round(float(train_pred.min()), 4),
                "context_train_pred_max": round(float(train_pred.max()), 4),
                "gated_train_pred_mean": round(float(gated_train_pred.mean()), 4),
                "good_probability_mean": round(float(train_good_probability.mean()), 4),
                "good_probability_min": round(float(train_good_probability.min()), 4),
                "good_probability_max": round(float(train_good_probability.max()), 4),
                "good_target_share": round(float(good_target.mean()), 4),
                "top_features": {
                    name: round(float(value), 6)
                    for name, value in sorted(
                        zip(context_payload.get("feature_columns", []), context_payload.get("coef", [])),
                        key=lambda item: abs(float(item[1])),
                        reverse=True,
                    )[:8]
                },
                "classifier_top_features": {
                    name: round(float(value), 6)
                    for name, value in sorted(
                        zip(good_classifier.get("feature_columns", []), good_classifier.get("coef", [])),
                        key=lambda item: abs(float(item[1])),
                        reverse=True,
                    )[:8]
                },
            }
            continue

        x = pd.DataFrame({"phys_delta": phys_delta, "geo_offset": offset_fit}, index=residual.index)
        ridge = Ridge(alpha=3.0, fit_intercept=True)
        ridge.fit(x.to_numpy(dtype=float), residual.to_numpy(dtype=float))

        phys_low, phys_high = spec["phys_bounds"]
        offset_low, offset_high = spec["offset_bounds"]
        phys_weight = float(np.clip(ridge.coef_[0], phys_low, phys_high))
        offset_weight = float(np.clip(ridge.coef_[1], offset_low, offset_high))
        intercept_low, intercept_high = spec.get(
            "intercept_bounds",
            (float(spec["low_clip"]) / 2.0, float(spec["high_clip"]) / 2.0),
        )
        intercept = float(np.clip(ridge.intercept_, float(intercept_low), float(intercept_high)))
        learned[metric_name] = {
            "intercept": intercept,
            "phys_weight": phys_weight,
            "offset_weight": offset_weight,
            "offset_only_weight": offset_weight,
        }
        debug["models"][metric_name] = {
            "used": True,
            "rows": valid_rows,
            "intercept": round(intercept, 6),
            "phys_weight": round(phys_weight, 6),
            "offset_weight": round(offset_weight, 6),
            "raw_phys_weight": round(float(ridge.coef_[0]), 6),
            "raw_offset_weight": round(float(ridge.coef_[1]), 6),
            "residual_bias": round(float(residual.mean()), 4),
            "residual_mae": round(float(np.abs(residual).mean()), 4),
            "intercept_bounds": [float(intercept_low), float(intercept_high)],
        }

    debug["enabled"] = any(model.get("used") for model in debug["models"].values())
    return learned, debug


def _apply_experimental_geo_adjustments(
    pred_df: pd.DataFrame,
    geo_coefficients: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    pred_out = pred_df.copy()
    coverage_offset, sinr_structural_offset = _geo_offsets_from_features(pred_out)
    pred_out["coverage_offset"] = coverage_offset
    pred_out["sinr_structural_offset"] = sinr_structural_offset
    coeffs = geo_coefficients or _default_geo_coefficients()
    rsrp_base = pd.to_numeric(pred_out["pred_rsrp"], errors="coerce")
    rsrq_base = pd.to_numeric(pred_out["pred_rsrq"], errors="coerce")
    sinr_base = pd.to_numeric(pred_out["pred_sinr"], errors="coerce")
    rsrp_phys = pd.to_numeric(pred_out.get("serving_proxy_rsrp_phys_dbm"), errors="coerce")
    rsrq_phys = pd.to_numeric(pred_out.get("rsrq_proxy_db_raw", pred_out.get("rsrq_proxy_db")), errors="coerce")
    sinr_phys = pd.to_numeric(pred_out.get("sinr_proxy_db_raw", pred_out.get("sinr_proxy_db")), errors="coerce")

    pred_out["pred_rsrp_geo"] = rsrp_base.copy()
    has_rsrp_phys = rsrp_phys.notna()
    rsrp_coeff = coeffs.get("RSRP", _default_geo_coefficients()["RSRP"])
    pred_out.loc[has_rsrp_phys, "pred_rsrp_geo"] = (
        rsrp_base[has_rsrp_phys]
        + float(rsrp_coeff.get("intercept", 0.0))
        + (float(rsrp_coeff.get("phys_weight", 0.0)) * (rsrp_phys[has_rsrp_phys] - rsrp_base[has_rsrp_phys]))
        + (float(rsrp_coeff.get("offset_weight", 0.0)) * coverage_offset[has_rsrp_phys])
    )
    pred_out.loc[~has_rsrp_phys, "pred_rsrp_geo"] = (
        rsrp_base[~has_rsrp_phys]
        + float(rsrp_coeff.get("intercept", 0.0))
        + (float(rsrp_coeff.get("offset_only_weight", rsrp_coeff.get("offset_weight", 0.0))) * coverage_offset[~has_rsrp_phys])
    )

    pred_out["pred_rsrq_geo"] = rsrq_base.copy()
    has_rsrq_phys = rsrq_phys.notna()
    rsrq_coeff = coeffs.get("RSRQ", _default_geo_coefficients()["RSRQ"])
    pred_out.loc[has_rsrq_phys, "pred_rsrq_geo"] = (
        rsrq_base[has_rsrq_phys]
        + float(rsrq_coeff.get("intercept", 0.0))
        + (float(rsrq_coeff.get("phys_weight", 0.0)) * (rsrq_phys[has_rsrq_phys] - rsrq_base[has_rsrq_phys]))
        + (float(rsrq_coeff.get("offset_weight", 0.0)) * coverage_offset[has_rsrq_phys])
    )
    pred_out.loc[~has_rsrq_phys, "pred_rsrq_geo"] = (
        rsrq_base[~has_rsrq_phys]
        + float(rsrq_coeff.get("intercept", 0.0))
        + (float(rsrq_coeff.get("offset_only_weight", rsrq_coeff.get("offset_weight", 0.0))) * coverage_offset[~has_rsrq_phys])
    )

    pred_out["pred_sinr_geo"] = sinr_base.copy()
    has_sinr_phys = sinr_phys.notna()
    sinr_coeff = coeffs.get("SINR", _default_geo_coefficients()["SINR"])
    sinr_offset = sinr_structural_offset.clip(lower=-14.291166799136963, upper=4.656856193336386)
    sinr_context_model = sinr_coeff.get("context_model") if isinstance(sinr_coeff, dict) else None
    if sinr_context_model:
        sinr_features = _sinr_geo_context_features(pred_out, sinr_base, sinr_phys, sinr_offset)
        context_correction = _predict_scaled_ridge_payload(sinr_features, sinr_context_model)
        clip_low, clip_high = sinr_coeff.get("context_correction_clip", [-10.0, 10.0])
        context_correction = context_correction.clip(lower=float(clip_low), upper=float(clip_high))
        good_classifier = sinr_coeff.get("good_sinr_classifier")
        if good_classifier:
            good_probability = _predict_scaled_logistic_probability(sinr_features, good_classifier)
            context_correction = context_correction.where(
                context_correction <= 0.0,
                context_correction * good_probability,
            )
        pred_out["pred_sinr_geo"] = sinr_base + context_correction
    else:
        pred_out.loc[has_sinr_phys, "pred_sinr_geo"] = (
            sinr_base[has_sinr_phys]
            + float(sinr_coeff.get("intercept", 0.0))
            + (float(sinr_coeff.get("phys_weight", 0.0)) * (sinr_phys[has_sinr_phys] - sinr_base[has_sinr_phys]))
            + (float(sinr_coeff.get("offset_weight", 0.0)) * sinr_offset[has_sinr_phys])
        )
        pred_out.loc[~has_sinr_phys, "pred_sinr_geo"] = (
            sinr_base[~has_sinr_phys]
            + float(sinr_coeff.get("intercept", 0.0))
            + (float(sinr_coeff.get("offset_only_weight", sinr_coeff.get("offset_weight", 0.0))) * sinr_offset[~has_sinr_phys])
        )

    pred_out["pred_rsrp_geo"] = pred_out["pred_rsrp_geo"].clip(-140, -44)
    pred_out["pred_rsrq_geo"] = pred_out["pred_rsrq_geo"].clip(-20, -3)
    pred_out["pred_sinr_geo"] = pred_out["pred_sinr_geo"].clip(-25.0, 30.0)

    summary = {
        "mode": {
            "train_rows": 0,
            "feature_count": 33,
            "top_features": {
                "blend_rsrp": "train_dt_learned: baseline + phys_weight * (forward_proxy_physics - baseline) + offset_weight * geo_offset",
                "blend_rsrq": "train_dt_learned: baseline + phys_weight * (rsrq_proxy_db - baseline) + offset_weight * geo_offset",
                "blend_sinr": "train_dt_classifier_gated: baseline + positive_residual * P(DT_SINR>=6) + negative_residual",
                "geo_coefficients": coeffs,
                "building_area_ratio": -9.0,
                "clutter_class": -4.5,
                "serving_distance_m": -0.0035,
                "azimuth_delta_deg": -0.018,
                "site_count_250m": 0.15,
                "green_ratio": -2.0,
                "water_ratio": 1.2,
                "morphology_cluster": -0.35,
                "dense_urban_far_penalty": "if Dense Urban and nearest_site_distance_m > 180",
                "far_serving_off_axis_penalty": "if serving_distance_m > 250 and azimuth_delta_deg > 35",
                "high_building_far_penalty": "if avg_building_area_m2 > 250 and nearest_site_distance_m > 160",
                "water_open_bonus": "if Water/Rural/Open and azimuth_delta_deg < 20 and nearest_site_distance_m < 220",
                "nlos_flag": -2.4,
                "los_blocked_ratio": -5.5,
                "diffraction_proxy_db": -0.55,
                "terrain_slope_deg": -0.08,
                "terrain_relief_to_site_m": -0.028,
                "interference_gap_db": "penalize below 6 dB, reward above 6 dB",
                "dt_training_used": False,
            },
        }
    }
    print("[TEST][EXPERIMENTAL] mode=train_dt_geo_coefficients_plus_residual_calibration")
    return pred_out, summary


def _run_rf_prediction_without_dt_calibration(
    site_df: pd.DataFrame,
    building_df: pd.DataFrame,
    params: Dict[str, object],
    polygon_gdf: Optional[gpd.GeoDataFrame] = None,
) -> pd.DataFrame:
    temp_dir = "temp_rf"
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    site_path = f"{temp_dir}/site.csv"
    building_path = f"{temp_dir}/building.csv"
    serving_site_df = _filter_sites_to_project_polygon(site_df, polygon_gdf) if polygon_gdf is not None else site_df.copy()
    site_export_df = _prepare_site_df_for_source_rf_export(serving_site_df)
    expected_unique = int(site_export_df["frontend_site_sector_key"].nunique(dropna=True)) if "frontend_site_sector_key" in site_export_df.columns else 0
    site_export_df.to_csv(site_path, index=False)
    building_df.to_csv(building_path, index=False)

    print(
        f"[TEST][RF_BASELINE] mode=cost231_no_dt_calibration serving_site_rows={len(site_export_df)} "
        f"expected_unique_frontend_site_sector={expected_unique} all_project_site_rows={len(site_df)} "
        f"building_rows={len(building_df)} radius={params['radius']} grid={params['grid']}"
    )
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
    })

    pred_df = pd.read_csv(f"{temp_dir}/prediction_ALL_SITES.csv")
    current_engine = ml_engine.engine.get(params.get("region", "india").lower(), ml_engine.engine["india"])
    pred_df, polygon_stats = ml_engine._apply_prediction_polygon_filter(
        pred_df, params["project_id"], current_engine
    )
    pred_df = _attach_site_identity_to_predictions(pred_df, site_df)
    print(
        f"[LTE][RF_OUTPUT_COUNTS] rows_before_polygon={polygon_stats['rows_before']} "
        f"rows_after_polygon={len(pred_df)} "
        f"polygon_removed={polygon_stats['rows_before'] - len(pred_df)} "
        f"polygon_swapped={polygon_stats['swapped']}"
    )
    ml_engine._print_fetch_summary(
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
            "unique_predicted_cells": ml_engine._safe_nunique(pred_df, "Node_Cell_ID"),
            "unique_frontend_site_sector": ml_engine._safe_nunique(pred_df, "frontend_site_sector_key"),
            "pred_rsrp_range": ml_engine._safe_minmax(pred_df, "pred_rsrp"),
            "pred_rsrq_range": ml_engine._safe_minmax(pred_df, "pred_rsrq"),
            "pred_sinr_range": ml_engine._safe_minmax(pred_df, "pred_sinr"),
        }
    )
    return pred_df


def _build_rf_accuracy_frame(
    site_df: pd.DataFrame,
    drive_df: pd.DataFrame,
    building_polygons,
    building_meta,
    workers: int,
    max_interference_sites: int,
) -> pd.DataFrame:
    site_rf = _normalize_site_for_rf(site_df)
    dt = drive_df.dropna(subset=["lat", "lon"]).copy()
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
    dt = dt.dropna(subset=["RSRP_meas"])

    params = {
        "k1": 0,
        "k2": 0,
        "polygons": building_polygons,
        "meta": building_meta,
        "antenna_gain": 18,
        "cable_loss": 2,
        "ue_height": 1.5,
        "frequency_mhz": 1800,
        "bandwidth_mhz": 10,
        "all_sites_rows": select_nearest_site_rows(site_rf, site_rf, max_interference_sites),
        "n_workers": workers,
    }

    rsrp_pred, rsrq_pred, sinr_pred = compute_predictions_parallel(
        dt,
        site_rf,
        params,
        n_workers=workers,
        use_shared_pool=True,
    )
    dt["RSRP_pred"] = rsrp_pred
    dt["RSRQ_pred"] = rsrq_pred
    dt["SINR_pred"] = sinr_pred
    return dt




def _find_latest_rf_log(before: Iterable[Path], after: Iterable[Path]) -> Optional[Path]:
    before_set = {p.resolve() for p in before}
    new_logs = [p for p in after if p.resolve() not in before_set]
    if not new_logs:
        return None
    return max(new_logs, key=lambda p: p.stat().st_mtime)


def _collect_rf_logs() -> List[Path]:
    root = Path("temp_rf")
    if not root.exists():
        return []
    return sorted(root.glob("run_log_*.txt"))


def run_rf_debug_lab(config: RunConfig) -> Path:
    run_dir = _ensure_dir(config.output_root / f"project_{config.project_id}" / _timestamp())
    cache_dir = _ensure_dir(run_dir / "cache")
    shared_cache_dir = _project_shared_cache_dir(config.output_root, config.project_id)
    log_path = run_dir / "run.log"
    timings: Dict[str, float] = {}
    summary: Dict[str, object] = {"config": config.__dict__.copy(), "run_dir": str(run_dir)}
    cached_artifacts = _load_cached_run_artifacts(config.reuse_run_dir) if config.reuse_cached_artifacts else {}
    cached_summary = cached_artifacts.get("summary") if cached_artifacts else None
    cache_reuse: Dict[str, object] = {
        "enabled": bool(config.reuse_cached_artifacts),
        "base_dir": str(cached_artifacts.get("base_dir")) if cached_artifacts else None,
        "inputs": False,
        "rf_prediction": False,
        "geo_enrichment": False,
        "advanced_geo_append": False,
        "reasons": {},
    }
    summary["cache_reuse"] = cache_reuse

    with log_path.open("w", encoding="utf-8") as log_file:
        tee = TeeStream(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"[TEST] Starting RF debug lab for project_id={config.project_id}")
            start_all = time.perf_counter()

            step = time.perf_counter()
            input_match, input_mismatches = _cached_config_matches(
                config,
                cached_summary,
                ["project_id", "session_ids", "region"],
            )
            can_reuse_inputs = (
                input_match
                and isinstance(cached_artifacts.get("site_df"), pd.DataFrame)
                and isinstance(cached_artifacts.get("drive_df"), pd.DataFrame)
                and isinstance(cached_artifacts.get("building_df"), pd.DataFrame)
                and isinstance(cached_artifacts.get("polygon_gdf"), gpd.GeoDataFrame)
            )
            if can_reuse_inputs:
                site_df = _normalize_site_for_rf(cached_artifacts["site_df"])
                drive_df = cached_artifacts["drive_df"].copy()
                building_df = cached_artifacts["building_df"].copy()
                polygon_gdf = cached_artifacts["polygon_gdf"].copy()
                operator = str((cached_summary or {}).get("operator") or "cached")
                polygon_alignment = str((cached_summary or {}).get("project_polygon_alignment") or "cached")
                cache_reuse["inputs"] = True
                print(f"[TEST][CACHE] Reusing saved input artifacts from {cached_artifacts['base_dir']}")
            else:
                cache_reuse["reasons"]["inputs"] = input_mismatches or ["artifacts_missing"]
                operator = str(config.operator or DEFAULT_OPERATOR)
                site_df, resolved_operator = ml_engine.fetch_site_data(config.project_id, region=config.region)
                site_df = _filter_site_df_for_operator(site_df, operator or resolved_operator)
                site_df = _normalize_site_for_rf(site_df)
                drive_df = _fetch_drive_data_for_test(
                    config.session_ids,
                    operator,
                    config.project_id,
                    region=config.region,
                )
                building_df = _fetch_building_data_for_test(config.project_id, config.region)
                polygon_gdf = _load_project_polygon_gdf(config.project_id, config.region)
                polygon_gdf, polygon_alignment = _align_project_polygon_to_points(polygon_gdf, site_df)
            timings["fetch_inputs_sec"] = round(time.perf_counter() - step, 2)
            validation_sessions = _resolve_validation_sessions(config.session_ids)
            print(
                f"[TEST] Inputs fetched site_rows={len(site_df)} drive_rows={len(drive_df)} "
                f"building_rows={len(building_df)} operator={operator}"
            )
            print(f"[TEST] Project polygon alignment={polygon_alignment}")
            print(
                f"[TEST] Validation mode=dt_validation_only "
                f"validation_sessions={list(validation_sessions)} validation_rows={len(drive_df)}"
            )
            resolved_dem_path = _resolve_dem_path_for_test(
                project_id=config.project_id,
                region=config.region,
                site_df=site_df,
                requested_path=config.dem_raster_path,
            )
            summary["resolved_dem_raster_path"] = str(resolved_dem_path) if resolved_dem_path else None

            step = time.perf_counter()
            cached_building_gdf = cached_artifacts.get("building_gdf")
            if cache_reuse["inputs"] and isinstance(cached_building_gdf, gpd.GeoDataFrame):
                building_gdf = cached_building_gdf.copy()
                building_alignment = str((cached_summary or {}).get("building_alignment") or "cached")
            else:
                building_gdf = _building_df_to_gdf(building_df)
                building_gdf, building_alignment = _align_building_geometries_to_project(building_gdf, polygon_gdf)
            building_df_for_rf = _prepare_building_df_for_rf(building_df, building_gdf)
            building_csv = run_dir / "building_debug.csv"
            export_building_df = building_df_for_rf.copy()
            export_building_df.to_csv(building_csv, index=False)
            building_polygons, building_meta = load_building_polygons(str(building_csv))
            timings["parse_buildings_sec"] = round(time.perf_counter() - step, 2)
            print(
                f"[TEST] Building geometry parsed db_polygons={len(building_gdf)} "
                f"rf_polygons={len(building_polygons)}"
            )
            if not building_gdf.empty:
                print(f"[TEST] Building bounds={building_gdf.total_bounds.tolist()}")
                print(f"[TEST] Building alignment={building_alignment}")
                print(
                    f"[TEST] RF building export prepared rows={len(building_df_for_rf)} "
                    f"non_null_geometry_wkt={int(building_df_for_rf['geometry_wkt'].notna().sum())}"
                )
            else:
                print("[TEST] Building geometry still empty after parsing; geo building features will remain zero")

            step = time.perf_counter()
            pred_match, pred_mismatches = _cached_config_matches(
                config,
                cached_summary,
                ["project_id", "region", "radius_m", "grid_resolution_m", "max_interference_sites"],
            )
            cached_pred_ok, cached_pred_issues = _cached_prediction_is_usable(cached_artifacts.get("pred_df"))
            if pred_match and cached_pred_ok:
                pred_df = cached_artifacts["pred_df"].copy()
                cache_reuse["rf_prediction"] = True
                print(f"[TEST][CACHE] Reusing saved RF prediction grid from {cached_artifacts['base_dir']}")
            else:
                cache_reuse["reasons"]["rf_prediction"] = pred_mismatches or cached_pred_issues
                pre_logs = _collect_rf_logs()
                pred_df = _run_rf_prediction_without_dt_calibration(
                    site_df,
                    building_df_for_rf,
                    {
                        "project_id": config.project_id,
                        "region": config.region,
                        "radius": config.radius_m,
                        "grid": config.grid_resolution_m,
                        "workers": config.workers,
                        "max_interference_sites": config.max_interference_sites,
                    },
                    polygon_gdf=polygon_gdf,
                )
                post_logs = _collect_rf_logs()
                rf_log_path = _find_latest_rf_log(pre_logs, post_logs)
                if rf_log_path:
                    rf_log_copy = run_dir / rf_log_path.name
                    rf_log_copy.write_text(rf_log_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                    summary["rf_log_path"] = str(rf_log_copy)
                    print(f"[TEST] RF source log captured at {rf_log_copy}")
            timings["rf_prediction_sec"] = round(time.perf_counter() - step, 2)
            rf_log_path = cached_artifacts.get("rf_log_path") if cache_reuse["rf_prediction"] else None
            if rf_log_path:
                rf_log_copy = run_dir / rf_log_path.name
                rf_log_copy.write_text(rf_log_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                summary["rf_log_path"] = str(rf_log_copy)
                print(f"[TEST] RF source log copied from cached run to {rf_log_copy}")

            step = time.perf_counter()
            geo_match, geo_mismatches = _cached_config_matches(
                config,
                cached_summary,
                ["project_id", "region", "tile_size_m", "cluster_count", "enable_osm"],
            )
            cached_grid_ok, cached_grid_issues = _cached_grid_artifacts_are_usable(
                cached_artifacts.get("grid_gdf"),
                cached_artifacts.get("grid_df"),
            )
            if geo_match and cached_grid_ok:
                grid_gdf = cached_artifacts["grid_gdf"].copy()
                grid_df = cached_artifacts["grid_df"].copy()
                feature_stats = dict((cached_summary or {}).get("feature_diagnostics") or _feature_diagnostics(grid_df))
                osm_status = dict((cached_summary or {}).get("osm_status") or {
                    "enabled": config.enable_osm,
                    "roads": False,
                    "green": False,
                    "water": False,
                })
                if "clutter_class" not in grid_gdf.columns or "morphology_cluster" not in grid_gdf.columns:
                    grid_gdf = grid_gdf.merge(
                        grid_df[["grid_id", "clutter_class", "morphology_cluster"]],
                        on="grid_id",
                        how="left",
                    )
                cache_reuse["geo_enrichment"] = True
                print(f"[TEST][CACHE] Reusing saved geo-enriched grid from {cached_artifacts['base_dir']}")
            else:
                cache_reuse["reasons"]["geo_enrichment"] = geo_mismatches or cached_grid_issues
                grid_gdf = _create_analysis_grid(polygon_gdf, config.tile_size_m)
                grid_gdf = _attach_building_features(grid_gdf, building_gdf)
                grid_gdf["road_length_m"] = 0.0
                grid_gdf["green_ratio"] = 0.0
                grid_gdf["water_ratio"] = 0.0

                osm_status = {"enabled": config.enable_osm, "roads": False, "green": False, "water": False}
                if config.enable_osm:
                    roads_gdf = _fetch_osm_layer("roads", polygon_gdf, ROAD_TAGS, cache_dir)
                    green_gdf = _fetch_osm_layer("green", polygon_gdf, GREEN_TAGS, cache_dir)
                    water_gdf = _fetch_osm_layer("water", polygon_gdf, WATER_TAGS, cache_dir)
                    osm_status["roads"] = not roads_gdf.empty
                    osm_status["green"] = not green_gdf.empty
                    osm_status["water"] = not water_gdf.empty
                    grid_gdf = _attach_line_density(grid_gdf, roads_gdf, "road_length_m")
                    grid_gdf = _attach_polygon_area_ratio(grid_gdf, green_gdf, "green_ratio")
                    grid_gdf = _attach_polygon_area_ratio(grid_gdf, water_gdf, "water_ratio")
                grid_df, _, feature_stats = _build_grid_feature_frame(
                    grid_gdf,
                    site_df,
                    config.cluster_count,
                )
                grid_gdf = grid_gdf.merge(
                    grid_df[["grid_id", "clutter_class", "morphology_cluster"]],
                    on="grid_id",
                    how="left",
                )
            advanced_geo_before = set(grid_df.columns)
            grid_df, advanced_geo_status = _augment_grid_with_advanced_geo_features(
                grid_df,
                building_gdf,
                site_df,
                polygon_gdf,
                shared_cache_dir,
                project_id=config.project_id,
                dem_raster_path=resolved_dem_path,
                terrain_api_url=config.terrain_api_url,
                terrain_api_batch_size=config.terrain_api_batch_size,
                terrain_sample_step_m=config.terrain_sample_step_m,
            )
            new_advanced_cols = sorted(set(grid_df.columns) - advanced_geo_before)
            cache_reuse["advanced_geo_append"] = bool(new_advanced_cols)
            if new_advanced_cols:
                print(f"[TEST][CACHE] Appended advanced geo features columns={new_advanced_cols}")
            _validate_advanced_geo_requirements(
                grid_df,
                advanced_geo_status,
                require_advanced_geo_on_miss=config.require_advanced_geo_on_miss,
            )
            for feature_name, stats in feature_stats.items():
                print(
                    f"[TEST][FEATURE] {feature_name} non_zero={stats['non_zero']} "
                    f"nunique={stats['nunique']} min={stats['min']:.4f} "
                    f"max={stats['max']:.4f} mean={stats['mean']:.4f}"
                )
            advanced_feature_stats = _feature_diagnostics(grid_df)
            for feature_name in [
                "best_interferer_distance_m",
                "interference_gap_db",
                "los_blocker_count",
                "los_blocked_ratio",
                "diffraction_proxy_db",
                "terrain_elevation_m",
                "terrain_slope_deg",
                "terrain_relief_to_site_m",
            ]:
                stats = advanced_feature_stats.get(feature_name)
                if stats:
                    print(
                        f"[TEST][FEATURE_ADV] {feature_name} non_zero={stats['non_zero']} "
                        f"nunique={stats['nunique']} min={stats['min']:.4f} "
                        f"max={stats['max']:.4f} mean={stats['mean']:.4f}"
                    )
            timings["geo_enrichment_sec"] = round(time.perf_counter() - step, 2)
            summary["osm_status"] = osm_status
            summary["advanced_geo_status"] = advanced_geo_status
            summary["project_polygon_alignment"] = polygon_alignment
            summary["building_alignment"] = building_alignment
            summary["feature_diagnostics"] = advanced_feature_stats
            frontend_site_sector_summary = _frontend_site_sector_summary(site_df, polygon_gdf)
            summary["site_identity"] = frontend_site_sector_summary
            print(f"[TEST][SITE_IDENTITY] {frontend_site_sector_summary}")
            site_canonical_duplicate_audit = _site_canonical_duplicate_audit(site_df)
            if not site_canonical_duplicate_audit.empty:
                duplicate_physical_sector_count = int((site_canonical_duplicate_audit["raw_row_count"] > 1).sum())
                max_rows_per_physical_sector = int(site_canonical_duplicate_audit["raw_row_count"].max())
            else:
                duplicate_physical_sector_count = 0
                max_rows_per_physical_sector = 0
            summary["site_canonical_identity"] = {
                "canonical_sector_count": int(site_df["canonical_sector_id"].nunique(dropna=True))
                if "canonical_sector_id" in site_df.columns
                else 0,
                "duplicate_physical_sector_count": duplicate_physical_sector_count,
                "max_rows_per_physical_sector": max_rows_per_physical_sector,
            }
            print(f"[TEST][SITE_CANONICAL_IDENTITY] {summary['site_canonical_identity']}")
            drive_train_df, drive_holdout_df = _split_drive_train_holdout(drive_df, config.validation_fraction)
            summary["holdout_strategy"] = "row_split_within_validation_sessions"
            summary["train_sessions"] = list(validation_sessions)
            summary["holdout_sessions"] = list(validation_sessions)
            summary["cluster_counts"] = (
                grid_df["morphology_cluster"].value_counts(dropna=False).sort_index().to_dict()
                if "morphology_cluster" in grid_df.columns
                else {}
            )
            print(f"[TEST][CLUSTER] counts={summary['cluster_counts']}")
            _run_post_rf_smoke_test(pred_df, drive_df, grid_gdf, grid_df)

            step = time.perf_counter()
            pred_df = _assign_points_to_tiles(pred_df, grid_gdf)
            pred_df = _attach_missing_grid_features_by_grid_id(pred_df, grid_df)
            pred_df = _attach_site_identity_to_predictions(pred_df, site_df)
            pred_df = _attach_fixed_serving_sinr_rsrq_proxy(
                pred_df,
                site_df,
                max_interferers=config.max_interference_sites,
                min_distance_m=config.sinr_min_distance_m,
            )
            geo_train_pred_df = pred_df.copy()
            coverage_offset, sinr_structural_offset = _geo_offsets_from_features(geo_train_pred_df)
            geo_train_pred_df["coverage_offset"] = coverage_offset
            geo_train_pred_df["sinr_structural_offset"] = sinr_structural_offset
            geo_train_eval = _attach_prediction_grid_to_points(drive_train_df, geo_train_pred_df)
            geo_coefficients, geo_calibration_debug = _fit_train_only_geo_coefficients(geo_train_eval)
            pred_df, experimental_model_debug = _apply_experimental_geo_adjustments(pred_df, geo_coefficients)
            train_eval, _, train_experimental_metrics = _evaluate_prediction_grid_against_holdout(
                drive_train_df,
                pred_df,
            )
            residual_models, residual_calibration_debug = _fit_train_only_residual_calibration(train_eval)
            pred_df = _apply_train_only_residual_calibration(pred_df, residual_models)
            pred_serving_view_df, serving_view_summary = _build_serving_view_prediction_grid(pred_df)
            holdout_eval, baseline_metrics, experimental_metrics = _evaluate_prediction_grid_against_holdout(
                drive_holdout_df,
                pred_df,
            )
            full_eval, full_baseline_metrics, full_experimental_metrics = _evaluate_prediction_grid_against_holdout(
                drive_df,
                pred_df,
            )
            serving_holdout_eval, serving_baseline_metrics, serving_experimental_metrics = _evaluate_prediction_grid_against_holdout(
                drive_holdout_df,
                pred_serving_view_df,
            )
            serving_full_eval, serving_full_baseline_metrics, serving_full_experimental_metrics = _evaluate_prediction_grid_against_holdout(
                drive_df,
                pred_serving_view_df,
            )
            pred_df, demo_overlay_summary = _apply_demo_dt_overlay(pred_df, drive_df)
            final_serving_view_df, final_serving_view_summary = _build_serving_view_prediction_grid(pred_df)
            _run_post_rf_integrity_checks(pred_df, grid_gdf, grid_df, holdout_eval)
            _run_artifact_write_smoke(run_dir, pred_df, holdout_eval, grid_df)
            metrics = {
                "baseline": baseline_metrics,
                "experimental": experimental_metrics,
            }
            timings["evaluation_sec"] = round(time.perf_counter() - step, 2)
            summary["production_style_prediction"] = True
            summary["site_identity"]["predicted_frontend_site_sector_count"] = (
                int(pred_df["frontend_site_sector_key"].nunique(dropna=True))
                if "frontend_site_sector_key" in pred_df.columns
                else 0
            )
            experimental_model_debug["mode"]["train_rows"] = int(len(train_eval))
            experimental_model_debug["mode"]["top_features"]["dt_training_used"] = bool(residual_models)
            experimental_model_debug["dt_geo_calibration"] = geo_calibration_debug
            experimental_model_debug["dt_residual_calibration"] = residual_calibration_debug
            experimental_model_debug["train_metrics"] = {
                "experimental": train_experimental_metrics,
            }
            summary["experimental_model"] = experimental_model_debug
            summary["demo_visualization"] = demo_overlay_summary
            summary["serving_view"] = {
                "pre_demo": serving_view_summary,
                "post_demo": final_serving_view_summary,
                "sinr_distribution_full_grid": _sinr_distribution_summary(pred_df, "pred_sinr_geo"),
                "sinr_distribution_serving_view": _sinr_distribution_summary(final_serving_view_df, "pred_sinr_geo"),
                "validation_metrics": {
                    "baseline": serving_baseline_metrics,
                    "experimental": serving_experimental_metrics,
                },
                "full_metrics": {
                    "baseline": serving_full_baseline_metrics,
                    "experimental": serving_full_experimental_metrics,
                },
            }

            step = time.perf_counter()
            grid_gdf.to_file(run_dir / "analysis_grid.geojson", driver="GeoJSON")
            if not building_gdf.empty:
                building_gdf.to_file(run_dir / "buildings.geojson", driver="GeoJSON")
            polygon_gdf.to_file(run_dir / "project_polygon.geojson", driver="GeoJSON")
            grid_df.to_csv(run_dir / "analysis_grid_features.csv", index=False)
            site_df.to_csv(run_dir / "site_df.csv", index=False)
            if not site_canonical_duplicate_audit.empty:
                site_canonical_duplicate_audit.to_csv(run_dir / "site_canonical_duplicate_audit.csv", index=False)
            drive_df.to_csv(run_dir / "drive_df.csv", index=False)
            drive_train_df.to_csv(run_dir / "drive_train.csv", index=False)
            drive_holdout_df.to_csv(run_dir / "drive_holdout.csv", index=False)
            holdout_eval.to_csv(run_dir / "rf_accuracy_points.csv", index=False)
            serving_holdout_eval.to_csv(run_dir / "rf_accuracy_points_serving_view.csv", index=False)
            pred_df.to_parquet(run_dir / "rf_prediction_grid.parquet", index=False)
            pred_df.to_csv(run_dir / "rf_prediction_grid_full.csv", index=False)
            final_serving_view_df.to_csv(run_dir / "rf_prediction_serving_view.csv", index=False)
            _safe_sample(pred_df).to_csv(run_dir / "rf_prediction_grid_sample.csv", index=False)
            _safe_sample(final_serving_view_df).to_csv(run_dir / "rf_prediction_serving_view_sample.csv", index=False)
            dashboard_path = _write_rf_debug_dashboard(run_dir, drive_df, final_serving_view_df, polygon_gdf)
            timings["artifact_write_sec"] = round(time.perf_counter() - step, 2)

            summary.update(
                {
                    "operator": operator,
                    "rows": {
                        "site_df": len(site_df),
                        "drive_df": len(drive_df),
                        "building_df": len(building_df),
                        "building_polygons": len(building_polygons),
                        "analysis_grid": len(grid_gdf),
                        "rf_prediction_grid": len(pred_df),
                        "rf_accuracy_points": len(holdout_eval),
                        "drive_train_df": len(drive_train_df),
                        "drive_holdout_df": len(drive_holdout_df),
                    },
                    "timings_sec": timings,
                    "validation_metrics": metrics,
                    "full_metrics": {
                        "baseline": full_baseline_metrics,
                        "experimental": full_experimental_metrics,
                    },
                    "artifacts": {
                        "analysis_grid": str(run_dir / "analysis_grid.geojson"),
                        "analysis_grid_features": str(run_dir / "analysis_grid_features.csv"),
                        "buildings": str(run_dir / "buildings.geojson"),
                        "project_polygon": str(run_dir / "project_polygon.geojson"),
                        "rf_accuracy_points": str(run_dir / "rf_accuracy_points.csv"),
                        "rf_accuracy_points_serving_view": str(run_dir / "rf_accuracy_points_serving_view.csv"),
                        "rf_prediction_grid": str(run_dir / "rf_prediction_grid.parquet"),
                        "rf_prediction_grid_full_csv": str(run_dir / "rf_prediction_grid_full.csv"),
                        "rf_prediction_grid_sample": str(run_dir / "rf_prediction_grid_sample.csv"),
                        "rf_prediction_serving_view": str(run_dir / "rf_prediction_serving_view.csv"),
                        "rf_prediction_serving_view_sample": str(run_dir / "rf_prediction_serving_view_sample.csv"),
                        "rf_debug_dashboard": str(dashboard_path),
                        "site_df": str(run_dir / "site_df.csv"),
                        "site_canonical_duplicate_audit": str(run_dir / "site_canonical_duplicate_audit.csv"),
                        "drive_df": str(run_dir / "drive_df.csv"),
                        "drive_train": str(run_dir / "drive_train.csv"),
                        "drive_holdout": str(run_dir / "drive_holdout.csv"),
                        "run_log": str(log_path),
                    },
                }
            )
            summary["total_runtime_sec"] = round(time.perf_counter() - start_all, 2)
            _write_json(run_dir / "summary.json", summary)
            print(f"[TEST] Completed run in {summary['total_runtime_sec']} sec")

    return run_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Test-only LTE RF debug lab for project 196")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--session-ids", type=int, nargs="+", default=DEFAULT_SESSION_IDS)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=DEFAULT_OPERATOR)
    parser.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    parser.add_argument("--grid-resolution-m", type=float, default=DEFAULT_GRID_RESOLUTION_M)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-interference-sites", type=int, default=DEFAULT_MAX_INTERFERENCE_SITES)
    parser.add_argument("--tile-size-m", type=float, default=DEFAULT_TILE_SIZE_M)
    parser.add_argument("--cluster-count", type=int, default=DEFAULT_CLUSTER_COUNT)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--enable-osm", action="store_true")
    parser.add_argument("--dem-raster-path", type=Path, default=DEFAULT_DEM_RASTER_PATH)
    parser.add_argument("--allow-missing-advanced-geo", action="store_true")
    parser.add_argument("--terrain-api-url", type=str, default=DEFAULT_TERRAIN_API_URL)
    parser.add_argument("--terrain-api-batch-size", type=int, default=DEFAULT_TERRAIN_API_BATCH_SIZE)
    parser.add_argument("--terrain-sample-step-m", type=float, default=DEFAULT_TERRAIN_SAMPLE_STEP_M)
    parser.add_argument("--sinr-min-distance-m", type=float, default=DEFAULT_SINR_MIN_DISTANCE_M)
    parser.add_argument("--output-root", type=Path, default=Path("tests/output"))
    parser.add_argument("--reuse-run-dir", type=Path, default=DEFAULT_REUSE_RUN_DIR)
    parser.add_argument("--disable-reuse-cache", action="store_true")
    args = parser.parse_args(argv)

    config = RunConfig(
        project_id=args.project_id,
        session_ids=tuple(args.session_ids),
        region=args.region,
        operator=args.operator,
        radius_m=args.radius_m,
        grid_resolution_m=args.grid_resolution_m,
        workers=args.workers,
        max_interference_sites=args.max_interference_sites,
        tile_size_m=args.tile_size_m,
        cluster_count=args.cluster_count,
        validation_fraction=args.validation_fraction,
        enable_osm=args.enable_osm,
        dem_raster_path=args.dem_raster_path,
        require_advanced_geo_on_miss=not args.allow_missing_advanced_geo,
        terrain_api_url=args.terrain_api_url,
        terrain_api_batch_size=args.terrain_api_batch_size,
        terrain_sample_step_m=args.terrain_sample_step_m,
        sinr_min_distance_m=args.sinr_min_distance_m,
        output_root=args.output_root,
        reuse_run_dir=args.reuse_run_dir,
        reuse_cached_artifacts=not args.disable_reuse_cache,
    )
    run_dir = run_rf_debug_lab(config)
    print(f"[TEST] Artifacts saved under {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
