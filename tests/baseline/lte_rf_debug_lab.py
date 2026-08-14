"""
RF debug lab - runs the REAL production LTE prediction pipeline end to end
(tools.lte_prediction.ml_engine + tools.lte_prediction.geo_correction_pipeline)
and writes debug artifacts (CSV/GeoJSON/parquet exports, an HTML map dashboard,
a run summary with production's own validation metrics) for inspection.

This file used to contain its own separately-written duplicate of the entire
pipeline (site/drive/building fetch, COST-231 wiring, clutter classification,
geo correction, DT calibration, an experimental Ridge/logistic geo-coefficient
fitting subsystem that never existed in production). That duplicate had drifted
from production over time and was deleted; this file now only calls the real
production functions and keeps the debug-only value-adds (artifact export,
HTML dashboard, run timing/summary).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tools.lte_prediction import ml_engine
from tools.lte_prediction.dem_utils import ensure_project_dem
from tools.lte_prediction.geo_correction_pipeline import (
    _choose_utm_crs,
    _prepare_drive_measurements,
    building_df_to_gdf,
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
MAX_MAP_POINTS = 18000
DEFAULT_DEM_RASTER_PATH: Optional[Path] = None
METRIC_THRESHOLDS = {
    "RSRP_meas": (3.0, 6.0, 10.0),
    "RSRQ_meas": (1.0, 2.0, 3.0),
    "SINR_meas": (2.0, 4.0, 6.0),
}

# Not used by run_rf_debug_lab itself (production's own run_ml_fast handles OSM
# context internally) - kept only because tests/coverage_prediction/lte_coverage_test.py
# (a different, non-baseline feature) imports these directly.
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
    dem_raster_path: Optional[Path] = DEFAULT_DEM_RASTER_PATH


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


def _empty_gdf(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _fetch_osm_layer(layer_name: str, polygon_gdf: gpd.GeoDataFrame, tags: Dict[str, object], cache_dir: Path) -> gpd.GeoDataFrame:
    """Fetches an OSM feature layer for a polygon, cached to disk under cache_dir.
    Not used by run_rf_debug_lab itself - kept for lte_coverage_test.py."""
    cache_path = cache_dir / f"osm_{layer_name}.geojson"
    if cache_path.exists():
        try:
            return gpd.read_file(cache_path)
        except Exception:
            pass
    if ox is None or polygon_gdf.empty:
        return _empty_gdf()
    try:
        polygon = polygon_gdf.to_crs("EPSG:4326").geometry.union_all()
        features = ox.features_from_polygon(polygon, tags=tags)
        if features is None or len(features) == 0:
            features = _empty_gdf()
        else:
            if not isinstance(features, gpd.GeoDataFrame):
                features = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
            features = features.set_crs("EPSG:4326") if features.crs is None else features.to_crs("EPSG:4326")
            features = features[features.geometry.notnull() & ~features.geometry.is_empty].copy()
    except Exception as exc:
        print(f"[TEST][OSM_FETCH] layer={layer_name} tags={tags} failed={exc}")
        features = _empty_gdf()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not features.empty:
            features[["geometry"]].to_file(cache_path, driver="GeoJSON")
    except Exception:
        pass
    return features


def _attach_line_density(grid_gdf: gpd.GeoDataFrame, line_gdf: gpd.GeoDataFrame, out_col: str) -> gpd.GeoDataFrame:
    out = grid_gdf.copy()
    if out_col not in out.columns:
        out[out_col] = 0.0
    if line_gdf.empty:
        return out
    grid_utm = out.to_crs(_choose_utm_crs(out))
    lines = line_gdf[line_gdf.geometry.type.isin(["LineString", "MultiLineString"])].to_crs(grid_utm.crs).copy()
    if lines.empty:
        return out
    joined = gpd.overlay(lines[["geometry"]], grid_utm[["grid_id", "geometry"]], how="intersection", keep_geom_type=False)
    if joined.empty:
        return out
    joined["seg_m"] = joined.geometry.length
    agg = joined.groupby("grid_id")["seg_m"].sum()
    out[out_col] = out["grid_id"].map(agg).fillna(0.0)
    return out


def _attach_polygon_area_ratio(grid_gdf: gpd.GeoDataFrame, poly_gdf: gpd.GeoDataFrame, out_col: str) -> gpd.GeoDataFrame:
    out = grid_gdf.copy()
    if out_col not in out.columns:
        out[out_col] = 0.0
    if poly_gdf.empty:
        return out
    grid_utm = out.to_crs(_choose_utm_crs(out))
    polys = poly_gdf[poly_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].to_crs(grid_utm.crs).copy()
    if polys.empty:
        return out
    if "cell_area_m2" not in grid_utm.columns:
        grid_utm["cell_area_m2"] = grid_utm.geometry.area
    joined = gpd.overlay(polys[["geometry"]], grid_utm[["grid_id", "geometry", "cell_area_m2"]], how="intersection", keep_geom_type=False)
    if joined.empty:
        return out
    joined["clip_area_m2"] = joined.geometry.area
    agg = joined.groupby("grid_id")["clip_area_m2"].sum()
    ratios = (agg / grid_utm.set_index("grid_id")["cell_area_m2"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[out_col] = out["grid_id"].map(ratios).fillna(0.0).clip(0.0, 1.0)
    return out


def run_rf_debug_lab(config: RunConfig) -> Path:
    """Runs the real production pipeline (ml_engine + geo_correction_pipeline)
    for one project and writes debug artifacts under output_root."""
    run_dir = _ensure_dir(config.output_root / f"project_{config.project_id}" / _timestamp())
    log_path = run_dir / "run.log"
    timings: Dict[str, float] = {}
    summary: Dict[str, object] = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in config.__dict__.items()},
        "run_dir": str(run_dir),
    }

    with log_path.open("w", encoding="utf-8") as log_file:
        tee = TeeStream(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(
                f"[TEST] Starting RF debug lab for project_id={config.project_id} "
                f"region={config.region} (real production pipeline)"
            )
            start_all = time.perf_counter()

            step = time.perf_counter()
            session_ids = _normalize_session_ids(config.session_ids)
            site_df, resolved_operator = ml_engine.fetch_site_data(
                config.project_id, region=config.region, operator=config.operator or None,
            )
            drive_df = ml_engine.fetch_drive_data(
                session_ids, resolved_operator, config.project_id, region=config.region,
            )
            building_df = ml_engine.fetch_building_data(config.project_id, region=config.region)
            timings["fetch_inputs_sec"] = round(time.perf_counter() - step, 2)
            print(
                f"[TEST] Inputs fetched site_rows={len(site_df)} drive_rows={len(drive_df)} "
                f"building_rows={len(building_df)} operator={resolved_operator}"
            )

            resolved_dem_path = _resolve_dem_path_for_test(
                project_id=config.project_id,
                region=config.region,
                site_df=site_df,
                requested_path=config.dem_raster_path,
            )
            summary["resolved_dem_raster_path"] = str(resolved_dem_path) if resolved_dem_path else None

            step = time.perf_counter()
            rf_params = {
                "project_id": config.project_id,
                "region": config.region,
                "radius": config.radius_m,
                "grid": config.grid_resolution_m,
                "workers": config.workers,
                "max_interference_sites": config.max_interference_sites,
            }
            pred_df = ml_engine.run_rf_prediction_fast(site_df, drive_df, building_df, rf_params)
            timings["rf_prediction_sec"] = round(time.perf_counter() - step, 2)
            print(
                f"[TEST] COST-231 prediction done rows={len(pred_df)} "
                f"rsrp_range=({pred_df['pred_rsrp'].min():.2f},{pred_df['pred_rsrp'].max():.2f})"
            )

            step = time.perf_counter()
            ml_params = {
                "project_id": config.project_id,
                "region": config.region,
                "grid": config.grid_resolution_m,
                "tile_size_m": config.tile_size_m,
                "cluster_count": config.cluster_count,
                "dem_raster_path": str(resolved_dem_path) if resolved_dem_path else None,
                "enable_osm_enrichment": config.enable_osm,
                "dt_validation_fraction": config.validation_fraction,
            }
            final_df = ml_engine.run_ml_fast(
                pred_df, drive_df, site_df=site_df, building_df=building_df, params=ml_params,
            )
            production_summary = dict(final_df.attrs.get("production_summary") or {})
            timings["geo_correction_and_calibration_sec"] = round(time.perf_counter() - step, 2)
            baseline_metrics = production_summary.get("baseline_validation_metrics") or {}
            geo_metrics = production_summary.get("geo_validation_metrics") or {}
            print(f"[TEST] Geo correction + DT calibration done rows={len(final_df)}")
            if geo_metrics:
                print(f"[TEST] geo_validation_metrics={geo_metrics}")

            current_engine = ml_engine.engine.get(config.region.lower(), ml_engine.engine["india"])
            polygons = ml_engine._resolve_prediction_polygons(
                {"project_id": config.project_id, "region": config.region}, current_engine,
            )
            polygon_gdf = gpd.GeoDataFrame({"geometry": list(polygons)}, crs="EPSG:4326")
            building_gdf = building_df_to_gdf(building_df)

            step = time.perf_counter()
            polygon_gdf.to_file(run_dir / "project_polygon.geojson", driver="GeoJSON")
            if not building_gdf.empty:
                building_gdf.to_file(run_dir / "buildings.geojson", driver="GeoJSON")
            site_df.to_csv(run_dir / "site_df.csv", index=False)
            drive_df.to_csv(run_dir / "drive_df.csv", index=False)
            final_df.to_parquet(run_dir / "rf_prediction_grid.parquet", index=False)
            final_df.to_csv(run_dir / "rf_prediction_grid_full.csv", index=False)
            _safe_sample(final_df).to_csv(run_dir / "rf_prediction_grid_sample.csv", index=False)
            dashboard_path = _write_rf_debug_dashboard(run_dir, drive_df, final_df, polygon_gdf)
            timings["artifact_write_sec"] = round(time.perf_counter() - step, 2)

            summary.update(
                {
                    "operator": resolved_operator,
                    "rows": {
                        "site_df": len(site_df),
                        "drive_df": len(drive_df),
                        "building_df": len(building_df),
                        "rf_prediction_grid": len(final_df),
                    },
                    "timings_sec": timings,
                    "production_summary": production_summary,
                    "baseline_validation_metrics": baseline_metrics,
                    "geo_validation_metrics": geo_metrics,
                    "artifacts": {
                        "project_polygon": str(run_dir / "project_polygon.geojson"),
                        "buildings": str(run_dir / "buildings.geojson"),
                        "rf_prediction_grid": str(run_dir / "rf_prediction_grid.parquet"),
                        "rf_prediction_grid_full_csv": str(run_dir / "rf_prediction_grid_full.csv"),
                        "rf_prediction_grid_sample": str(run_dir / "rf_prediction_grid_sample.csv"),
                        "rf_debug_dashboard": str(dashboard_path),
                        "site_df": str(run_dir / "site_df.csv"),
                        "drive_df": str(run_dir / "drive_df.csv"),
                        "run_log": str(log_path),
                    },
                }
            )
            summary["total_runtime_sec"] = round(time.perf_counter() - start_all, 2)
            _write_json(run_dir / "summary.json", summary)
            print(f"[TEST] Completed run in {summary['total_runtime_sec']} sec")

    return run_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="RF debug lab - runs the real production LTE prediction pipeline end to end"
    )
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
    parser.add_argument("--output-root", type=Path, default=Path("tests/output"))
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
        output_root=args.output_root,
    )
    run_dir = run_rf_debug_lab(config)
    print(f"[TEST] Artifacts saved under {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
