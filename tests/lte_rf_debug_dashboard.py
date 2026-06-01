from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
from shapely.ops import unary_union

from tests.lte_rf_debug_lab import (
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_GRID_RESOLUTION_M,
    DEFAULT_MAX_INTERFERENCE_SITES,
    DEFAULT_PROJECT_ID,
    DEFAULT_RADIUS_M,
    DEFAULT_REGION,
    DEFAULT_SESSION_IDS,
    DEFAULT_TILE_SIZE_M,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_WORKERS,
    RunConfig,
    run_rf_debug_lab,
)
from tests.coverage_prediction.lte_coverage_test import CoverageTestConfig, run_coverage_test
from tests.lte_rf_optimization_test import OptimizationTestConfig, run_optimization_test
from tests.lte_tilt_recommendation_test import TiltRecommendationTestConfig, run_tilt_recommendation_test
from tests.lte_tilt_recommendation_optimization_test import (
    TiltOptimizationTestConfig,
    run_tilt_recommendation_optimization_test,
)


OUTPUT_ROOT = Path("tests/output")
MAX_SITE_POINTS = 250
MAX_DRIVE_POINTS = 1200
MAX_PRED_POINTS = 3500
MAX_COMPARE_PRED_POINTS = 100000
MAX_BUILDING_POLYGONS = 250

KPI_LIMITS = {
    "RSRP": (-140, -44),
    "RSRQ": (-20, -3),
    "SINR": (-10, 30),
}

DEFAULT_COVERAGE_POLYGON_WKT = (
    "POLYGON(("
    "77.3493010211505 28.6451999446618,"
    "77.3760801959551 28.6563475183659,"
    "77.3798996615923 28.6493804236681,"
    "77.3790413547076 28.6309248012639,"
    "77.3383146930255 28.6320924980605,"
    "77.3493010211505 28.6451999446618"
    "))"
)

OPENCELLID_OPERATOR_LABELS = {
    (404, 2): "Airtel",
    (404, 3): "Airtel",
    (404, 5): "Airtel",
    (404, 7): "Airtel",
    (404, 45): "Airtel",
    (404, 49): "Airtel",
    (404, 51): "Airtel",
    (404, 52): "Airtel",
    (404, 53): "Airtel",
    (404, 54): "Airtel",
    (404, 55): "Airtel",
    (404, 56): "Airtel",
    (404, 57): "Airtel",
    (404, 58): "Airtel",
    (404, 59): "Airtel",
    (404, 62): "Airtel",
    (404, 64): "Airtel",
    (404, 66): "Airtel",
    (404, 68): "Airtel",
    (404, 70): "Airtel",
    (404, 10): "Airtel",
    (404, 90): "Airtel",
    (404, 92): "Airtel",
    (404, 93): "Airtel",
    (404, 94): "Airtel",
    (404, 95): "Airtel",
    (404, 96): "Airtel",
    (404, 97): "Airtel",
    (404, 98): "Airtel",
    (404, 1): "Vi",
    (404, 11): "Vi",
    (404, 12): "Vi",
    (404, 13): "Vi",
    (404, 14): "Vi",
    (404, 15): "Vi",
    (404, 16): "Vi",
    (404, 17): "Vi",
    (404, 18): "Vi",
    (404, 19): "Vi",
    (404, 4): "Vi",
    (404, 20): "Vi",
    (404, 21): "Vi",
    (404, 22): "Vi",
    (404, 24): "Vi",
    (404, 25): "Vi",
    (404, 27): "Vi",
    (404, 30): "Vi",
    (404, 34): "Vi",
    (404, 36): "Vi",
    (404, 43): "Vi",
    (404, 46): "Vi",
    (404, 60): "Vi",
    (404, 84): "Vi",
    (404, 86): "Vi",
    (405, 66): "Vi",
    (405, 67): "Vi",
    (405, 750): "Jio",
    (405, 751): "Jio",
    (405, 752): "Jio",
    (405, 753): "Jio",
    (405, 754): "Jio",
    (405, 755): "Jio",
    (405, 756): "Jio",
    (405, 799): "Jio",
    (405, 800): "Jio",
    (405, 801): "Jio",
    (405, 802): "Jio",
    (405, 803): "Jio",
    (405, 804): "Jio",
    (405, 805): "Jio",
    (405, 806): "Jio",
    (405, 807): "Jio",
    (405, 808): "Jio",
    (405, 809): "Jio",
    (405, 810): "Jio",
    (405, 811): "Jio",
    (405, 812): "Jio",
    (405, 813): "Jio",
    (405, 814): "Jio",
    (405, 815): "Jio",
    (405, 816): "Jio",
    (405, 817): "Jio",
    (405, 818): "Jio",
    (405, 819): "Jio",
    (405, 820): "Jio",
    (405, 821): "Jio",
    (405, 822): "Jio",
    (405, 823): "Jio",
    (405, 824): "Jio",
    (405, 825): "Jio",
    (405, 826): "Jio",
    (405, 827): "Jio",
    (405, 828): "Jio",
    (405, 829): "Jio",
    (405, 830): "Jio",
    (405, 831): "Jio",
    (405, 832): "Jio",
    (405, 833): "Jio",
    (405, 834): "Jio",
    (405, 835): "Jio",
    (405, 836): "Jio",
    (405, 837): "Jio",
    (405, 838): "Jio",
    (405, 839): "Jio",
    (405, 840): "Jio",
    (405, 841): "Jio",
    (405, 842): "Jio",
    (405, 843): "Jio",
    (405, 844): "Jio",
    (405, 845): "Jio",
    (405, 846): "Jio",
    (405, 847): "Jio",
    (405, 848): "Jio",
    (405, 849): "Jio",
    (405, 850): "Jio",
    (405, 851): "Jio",
    (405, 852): "Jio",
    (405, 853): "Jio",
    (405, 854): "Jio",
    (405, 855): "Jio",
    (405, 856): "Jio",
    (405, 872): "Jio",
    (405, 857): "Jio",
    (405, 858): "Jio",
    (405, 859): "Jio",
    (405, 860): "Jio",
    (405, 861): "Jio",
    (405, 862): "Jio",
    (405, 863): "Jio",
    (405, 864): "Jio",
    (405, 865): "Jio",
    (405, 866): "Jio",
    (405, 867): "Jio",
    (405, 868): "Jio",
    (405, 869): "Jio",
    (405, 870): "Jio",
    (405, 871): "Jio",
}

OPENCELLID_OPERATOR_GROUPS = {
    "Airtel": {
        (404, 2), (404, 3), (404, 5), (404, 7), (404, 10), (404, 45), (404, 49),
        (404, 51), (404, 52), (404, 53), (404, 54), (404, 55), (404, 56), (404, 57),
        (404, 58), (404, 59), (404, 62), (404, 64), (404, 66), (404, 68), (404, 70),
        (404, 90), (404, 92), (404, 93), (404, 94), (404, 95), (404, 96), (404, 97),
        (404, 98),
    },
    "Vi": {
        (404, 1), (404, 4), (404, 11), (404, 12), (404, 13), (404, 14), (404, 15),
        (404, 16), (404, 17), (404, 18), (404, 19), (404, 20), (404, 21), (404, 22),
        (404, 24), (404, 25), (404, 27), (404, 30), (404, 34), (404, 36), (404, 43),
        (404, 46), (404, 60), (404, 84), (404, 86), (405, 66), (405, 67),
    },
}

COVERAGE_METRIC_SPECS = {
    "RSRP": {
        "column": "rsrp",
        "valid_range": (-125.0, -45.0),
        "bands": [
            ("All Values", None, None),
            ("Better than -75", -75.0, None),
            ("-90 to -75", -90.0, -75.0),
            ("-105 to -90", -105.0, -90.0),
            ("-125 to -105", -125.0, -105.0),
        ],
    },
    "RSRQ": {
        "column": "rsrq",
        "valid_range": (-20.0, -3.0),
        "bands": [
            ("All Values", None, None),
            ("Better than -10", -10.0, None),
            ("-15 to -10", -15.0, -10.0),
            ("-20 to -15", -20.0, -15.0),
        ],
    },
    "SINR": {
        "column": "sinr",
        "valid_range": (-10.0, 30.0),
        "bands": [
            ("All Values", None, None),
            ("20 and above", 20.0, None),
            ("13 to 20", 13.0, 20.0),
            ("5 to 13", 5.0, 13.0),
            ("0 to 5", 0.0, 5.0),
            ("-10 to 0", -10.0, 0.0),
        ],
    },
    "RSSI": {
        "column": "rssi",
        "valid_range": (-110.0, -45.0),
        "bands": [
            ("All Values", None, None),
            ("Better than -65", -65.0, None),
            ("-75 to -65", -75.0, -65.0),
            ("-85 to -75", -85.0, -75.0),
            ("-95 to -85", -95.0, -85.0),
            ("-110 to -95", -110.0, -95.0),
        ],
    },
    "DL Throughput": {
        "column": "dl_tpt",
        "valid_range": (0.0, None),
        "bands": [
            ("All Values", None, None),
            ("50 Mbps and above", 50.0, None),
            ("20 to 50 Mbps", 20.0, 50.0),
            ("5 to 20 Mbps", 5.0, 20.0),
            ("0 to 5 Mbps", 0.0, 5.0),
        ],
    },
    "UL Throughput": {
        "column": "ul_tpt",
        "valid_range": (0.0, None),
        "bands": [
            ("All Values", None, None),
            ("20 Mbps and above", 20.0, None),
            ("10 to 20 Mbps", 10.0, 20.0),
            ("3 to 10 Mbps", 3.0, 10.0),
            ("0 to 3 Mbps", 0.0, 3.0),
        ],
    },
    "CQI": {
        "column": "cqi",
        "valid_range": (0.0, 15.0),
        "bands": [
            ("All Values", None, None),
            ("12 and above", 12.0, None),
            ("9 to 12", 9.0, 12.0),
            ("6 to 9", 6.0, 9.0),
            ("3 to 6", 3.0, 6.0),
            ("Below 3", None, 3.0),
        ],
    },
}

KPI_VISUAL_BANDS = {
    "RSRP": [
        {"label": "Best (-44 to -75)", "min": -75.0, "max": -44.0, "color": "#16a34a"},
        {"label": "-75 to -85", "min": -85.0, "max": -75.0, "color": "#2563eb"},
        {"label": "-85 to -95", "min": -95.0, "max": -85.0, "color": "#facc15"},
        {"label": "-95 to -105", "min": -105.0, "max": -95.0, "color": "#92400e"},
        {"label": "Below -105", "min": None, "max": -105.0, "color": "#dc2626"},
    ],
    "RSRQ": [
        {"label": "Best (-3 to -8)", "min": -8.0, "max": -3.0, "color": "#16a34a"},
        {"label": "-8 to -11", "min": -11.0, "max": -8.0, "color": "#2563eb"},
        {"label": "-11 to -14", "min": -14.0, "max": -11.0, "color": "#facc15"},
        {"label": "-14 to -17", "min": -17.0, "max": -14.0, "color": "#92400e"},
        {"label": "Below -17", "min": None, "max": -17.0, "color": "#dc2626"},
    ],
    "SINR": [
        {"label": "Best (20 to 30)", "min": 20.0, "max": 30.0, "color": "#16a34a"},
        {"label": "13 to 20", "min": 13.0, "max": 20.0, "color": "#2563eb"},
        {"label": "5 to 13", "min": 5.0, "max": 13.0, "color": "#facc15"},
        {"label": "0 to 5", "min": 0.0, "max": 5.0, "color": "#92400e"},
        {"label": "Below 0", "min": None, "max": 0.0, "color": "#dc2626"},
    ],
}

RF_DEBUG_COMPARE_BANDS = {
    "RSRP": {
        "unit": "dBm",
        "dt_candidates": ("RSRP_meas", "rsrp"),
        "pred_candidates": ("pred_rsrp",),
        "bands": [
            {"label": "-140 to -120", "min": -140.0, "max": -120.0, "color": "#ef2f2f"},
            {"label": "-120 to -100", "min": -120.0, "max": -100.0, "color": "#9a5a22"},
            {"label": "-100 to -95", "min": -100.0, "max": -95.0, "color": "#f3ea4e"},
            {"label": "-95 to -90", "min": -95.0, "max": -90.0, "color": "#76df58"},
            {"label": "-90 to -44", "min": -90.0, "max": -44.0, "color": "#159a28"},
        ],
    },
    "RSRQ": {
        "unit": "dB",
        "dt_candidates": ("RSRQ_meas", "rsrq"),
        "pred_candidates": ("pred_rsrq",),
        "bands": [
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
        "dt_candidates": ("SINR_meas", "sinr"),
        "pred_candidates": ("pred_sinr",),
        "bands": [
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


def _list_runs(project_id: int) -> List[Path]:
    root = OUTPUT_ROOT / f"project_{project_id}"
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _list_baseline_runs(project_id: int) -> List[Path]:
    out: List[Path] = []
    for run in _list_runs(project_id):
        summary = _load_summary(run)
        run_type = summary.get("run_type")
        if run_type in {"optimization_test", "tilt_recommendation_test", "coverage_test", "opencellid_probe_test"}:
            continue
        if (
            run.name.startswith("optimization_")
            or run.name.startswith("tilt_")
            or run.name.startswith("coverage_")
            or run.name.startswith("opencellid_probe_")
        ):
            continue
        out.append(run)
    return out


def _list_optimization_runs(project_id: int) -> List[Path]:
    out: List[Path] = []
    for run in _list_runs(project_id):
        summary = _load_summary(run)
        if summary.get("run_type") == "optimization_test" or run.name.startswith("optimization_"):
            out.append(run)
    return out


def _list_tilt_optimization_runs(project_id: int) -> List[Path]:
    out: List[Path] = []
    for run in _list_runs(project_id):
        summary = _load_summary(run)
        if summary.get("run_type") == "tilt_recommendation_optimization_test" or run.name.startswith("tilt_optimization_"):
            out.append(run)
    return out


def _list_tilt_runs(project_id: int) -> List[Path]:
    out: List[Path] = []
    for run in _list_runs(project_id):
        summary = _load_summary(run)
        if summary.get("run_type") == "tilt_recommendation_test" or run.name.startswith("tilt_"):
            out.append(run)
    return out


def _list_coverage_runs(project_id: int) -> List[Path]:
    out: List[Path] = []
    for run in _list_runs(project_id):
        summary = _load_summary(run)
        if summary.get("run_type") == "coverage_test" or run.name.startswith("coverage_"):
            out.append(run)
    return out


def _list_opencellid_probe_runs(project_id: int) -> List[Path]:
    out: List[Path] = []
    for run in _list_runs(project_id):
        summary = _load_summary(run)
        if summary.get("run_type") == "opencellid_probe_test" or run.name.startswith("opencellid_probe_"):
            out.append(run)
    return out


def _load_summary(run_dir: Path) -> Dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _metric_row(title: str, metrics: Dict[str, float]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{title} MAE", metrics.get("mae"))
    c2.metric(f"{title} RMSE", metrics.get("rmse"))
    c3.metric(f"{title} R2", metrics.get("r2"))


def _render_metric_detail_table(summary: Dict, metric_name: str) -> None:
    rows = []
    for series_name in ["baseline", "experimental"]:
        metrics = summary.get("full_metrics", {}).get(series_name, {}).get(metric_name)
        if metrics:
            row = {"series": "Baseline RF" if series_name == "baseline" else "Experimental Geo"}
            row.update(metrics)
            rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _build_map(
    polygon_gdf: gpd.GeoDataFrame,
    site_df: pd.DataFrame,
    drive_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    grid_gdf: Optional[gpd.GeoDataFrame] = None,
    show_geo: bool = False,
    kpi_col: str = "pred_rsrp",
    selected_sector: Optional[str] = None,
    selected_nodeb: Optional[str] = None,
    show_site_markers: bool = True,
) -> folium.Map:
    center_source = site_df if not site_df.empty else pred_df
    center = [float(pd.to_numeric(center_source["lat"], errors="coerce").median()), float(pd.to_numeric(center_source["lon"], errors="coerce").median())]
    fmap = folium.Map(
        location=center,
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
        width="100%",
        height=680,
    )

    folium.GeoJson(
        polygon_gdf,
        name="Project Polygon",
        style_function=lambda _: {"color": "#ef4444", "weight": 3, "fillOpacity": 0.0},
    ).add_to(fmap)

    if buildings_gdf is not None and not buildings_gdf.empty:
        step = max(1, len(buildings_gdf) // MAX_BUILDING_POLYGONS)
        b_sample = buildings_gdf.iloc[::step].head(MAX_BUILDING_POLYGONS)
        folium.GeoJson(
            b_sample,
            name="Buildings",
            style_function=lambda _: {"color": "#6b7280", "weight": 1, "fillColor": "#9ca3af", "fillOpacity": 0.2},
        ).add_to(fmap)

    if grid_gdf is not None and not grid_gdf.empty:
        geo_col = "morphology_cluster" if show_geo else "clutter_class"
        if geo_col in grid_gdf.columns:
            color_map = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
            colors = {}
            for idx, key in enumerate(sorted(grid_gdf[geo_col].dropna().astype(str).unique())):
                colors[key] = color_map[idx % len(color_map)]

            def _style(feature):
                key = str(feature["properties"].get(geo_col))
                color = colors.get(key, "#888888")
                return {"color": color, "weight": 1, "fillColor": color, "fillOpacity": 0.18}

            folium.GeoJson(grid_gdf, name=geo_col, style_function=_style).add_to(fmap)

    site_work = _prepare_site_selection_df(site_df)
    for col in ["lat", "lon", "nodeb_id", "cell_id", "Node_Cell_ID", "dashboard_nodeb_id"]:
        if col in site_work.columns:
            site_work[col] = site_work[col].astype(str) if col in {"nodeb_id", "cell_id", "Node_Cell_ID", "dashboard_nodeb_id"} else pd.to_numeric(site_work[col], errors="coerce")
    if selected_sector:
        site_work = site_work[site_work.get("Node_Cell_ID", "").astype(str) == str(selected_sector)].copy()
    elif selected_nodeb and "dashboard_nodeb_id" in site_work.columns:
        site_work = site_work[site_work["dashboard_nodeb_id"].astype(str) == str(selected_nodeb)].copy()

    if show_site_markers and not site_work.empty:
        site_sample = site_work.iloc[::max(1, len(site_work) // MAX_SITE_POINTS)].head(MAX_SITE_POINTS)
        for _, row in site_sample.iterrows():
            tooltip_label = f"Site cell={row.get('Node_Cell_ID', row.get('cell_id'))} nodeb={row.get('dashboard_nodeb_id')}"
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4 if selected_sector or selected_nodeb else 3,
                color="#111827",
                weight=1,
                fill=True,
                fill_color="#111827",
                fill_opacity=0.85,
                tooltip=tooltip_label,
            ).add_to(fmap)

    drive_sample = drive_df.copy()
    rsrp_col = next((c for c in drive_sample.columns if "rsrp" in c.lower()), None)
    if rsrp_col:
        drive_sample["_rsrp"] = pd.to_numeric(drive_sample[rsrp_col], errors="coerce")
        drive_sample = drive_sample.dropna(subset=["_rsrp"])
        drive_sample = drive_sample.iloc[::max(1, len(drive_sample) // MAX_DRIVE_POINTS)].head(MAX_DRIVE_POINTS)
        color_scale = px.colors.sample_colorscale("Turbo", [0.1, 0.4, 0.7, 0.9])
        bins = [-140, -110, -95, -80, -44]
        for _, row in drive_sample.iterrows():
            value = row["_rsrp"]
            color = color_scale[0]
            for idx in range(len(bins) - 1):
                if bins[idx] <= value < bins[idx + 1]:
                    color = color_scale[min(idx, len(color_scale) - 1)]
                    break
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.65,
                weight=0,
                tooltip=f"DT RSRP={value:.2f}",
            ).add_to(fmap)

    pred_work = pred_df.copy()
    for col in ["lat", "lon", kpi_col]:
        if col in pred_work.columns:
            pred_work[col] = pd.to_numeric(pred_work[col], errors="coerce")
    if "Node_Cell_ID" in pred_work.columns:
        pred_work["Node_Cell_ID"] = pred_work["Node_Cell_ID"].astype(str)
    if selected_sector and "Node_Cell_ID" in pred_work.columns:
        pred_work = pred_work[pred_work["Node_Cell_ID"] == str(selected_sector)].copy()
    elif selected_nodeb and "Node_Cell_ID" in pred_work.columns and not site_df.empty:
        site_lookup = _prepare_site_selection_df(site_df)
        nodeb_cells = set(
            site_lookup.loc[
                site_lookup["dashboard_nodeb_id"].astype(str) == str(selected_nodeb),
                "Node_Cell_ID" if "Node_Cell_ID" in site_lookup.columns else "cell_id",
            ].astype(str).tolist()
        )
        pred_work = pred_work[pred_work["Node_Cell_ID"].isin(nodeb_cells)].copy()

    pred_sample = pred_work.dropna(subset=["lat", "lon", kpi_col]).copy()
    pred_sample[kpi_col] = pd.to_numeric(pred_sample[kpi_col], errors="coerce")
    pred_sample = pred_sample.dropna(subset=[kpi_col])
    if not polygon_gdf.empty and not pred_sample.empty:
        polygon_union = unary_union(polygon_gdf.geometry)
        pred_points = gpd.GeoDataFrame(
            pred_sample,
            geometry=gpd.points_from_xy(pred_sample["lon"], pred_sample["lat"]),
            crs=polygon_gdf.crs or "EPSG:4326",
        )
        pred_sample = pd.DataFrame(pred_points[pred_points.geometry.within(polygon_union)].drop(columns="geometry"))
    pred_sample = pred_sample.iloc[::max(1, len(pred_sample) // MAX_PRED_POINTS)].head(MAX_PRED_POINTS)
    for _, row in pred_sample.iterrows():
        value = float(row[kpi_col])
        metric_name = "RSRP" if "rsrp" in kpi_col.lower() else "RSRQ" if "rsrq" in kpi_col.lower() else "SINR"
        low, high = KPI_LIMITS[metric_name]
        clipped_value = min(max(value, float(low)), float(high))
        scale_position = (clipped_value - low) / (high - low)
        color = px.colors.sample_colorscale("Viridis", [scale_position])[0]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2 if selected_sector or selected_nodeb else 1,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6 if selected_sector or selected_nodeb else 0.45,
            weight=0,
            tooltip=f"{kpi_col}={value:.2f} cell={row.get('Node_Cell_ID')}",
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def _first_present_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    lower_lookup = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        matched = lower_lookup.get(candidate.lower())
        if matched is not None:
            return matched
    return None


def _band_for_value(value: float, bands: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if pd.isna(value):
        return None
    for idx, band in enumerate(bands):
        low = band.get("min")
        high = band.get("max")
        low_ok = low is None or value >= float(low)
        high_ok = high is None or value < float(high) or (idx == len(bands) - 1 and value <= float(high))
        if low_ok and high_ok:
            return band
    return None


def _sample_map_points(df: pd.DataFrame, value_col: str, max_points: int) -> pd.DataFrame:
    if df.empty or not {"lat", "lon", value_col}.issubset(df.columns):
        return pd.DataFrame()
    work = df.copy()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["lat", "lon", value_col])
    if work.empty:
        return work
    step = max(1, len(work) // max_points)
    return work.iloc[::step].head(max_points).copy()


def _render_kpi_legend(title: str, bands: List[Dict[str, object]], counts: Dict[str, int], total: int) -> None:
    rows = []
    for band in bands:
        label = str(band["label"])
        count = int(counts.get(label, 0))
        pct = (count / total * 100.0) if total else 0.0
        rows.append(
            {
                "Range": label,
                "Count": count,
                "Percent": f"{pct:.1f}%",
                "Color": str(band["color"]),
            }
        )
    st.caption(f"{title} legend, total plotted: {total:,}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _build_dt_prediction_compare_map(
    polygon_gdf: gpd.GeoDataFrame,
    drive_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    metric_name: str,
    layer_choice: str,
) -> tuple[folium.Map, Dict[str, object]]:
    spec = RF_DEBUG_COMPARE_BANDS[metric_name]
    source_df = drive_df if layer_choice == "DT Original" else pred_df
    value_col = _first_present_col(
        source_df,
        spec["dt_candidates"] if layer_choice == "DT Original" else spec["pred_candidates"],
    )
    center_source = source_df if not source_df.empty else pred_df
    center = [
        float(pd.to_numeric(center_source.get("lat", pd.Series([28.6])), errors="coerce").median()),
        float(pd.to_numeric(center_source.get("lon", pd.Series([77.2])), errors="coerce").median()),
    ]
    fmap = folium.Map(
        location=center,
        zoom_start=15,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
        width="100%",
        height=680,
    )
    if not polygon_gdf.empty:
        folium.GeoJson(
            polygon_gdf,
            name="Project Polygon",
            style_function=lambda _: {"color": "#2563eb", "weight": 2, "fillOpacity": 0.0},
        ).add_to(fmap)
    if not value_col:
        return fmap, {"title": f"{metric_name} ({spec['unit']})", "bands": spec["bands"], "counts": {}, "total": 0}

    max_points = MAX_DRIVE_POINTS if layer_choice == "DT Original" else MAX_COMPARE_PRED_POINTS
    sample = _sample_map_points(source_df, value_col, max_points)
    counts = {str(band["label"]): 0 for band in spec["bands"]}
    for _, row in sample.iterrows():
        value = float(row[value_col])
        band = _band_for_value(value, spec["bands"])
        if not band:
            continue
        counts[str(band["label"])] += 1
        tooltip_parts = [
            f"{layer_choice} {metric_name}: {value:.2f} {spec['unit']}",
        ]
        if "demo_visual_source" in row:
            tooltip_parts.append(f"source={row.get('demo_visual_source')}")
        if "Node_Cell_ID" in row:
            tooltip_parts.append(f"cell={row.get('Node_Cell_ID')}")
        is_dt = layer_choice == "DT Original"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4 if is_dt else 5,
            color=str(band["color"]),
            fill=True,
            fill_color=str(band["color"]),
            fill_opacity=0.82 if is_dt else 0.72,
            opacity=0.95 if is_dt else 0.72,
            weight=1 if is_dt else 0,
            tooltip="<br>".join(tooltip_parts),
        ).add_to(fmap)
    return fmap, {
        "title": f"{metric_name} ({spec['unit']})",
        "bands": spec["bands"],
        "counts": counts,
        "total": int(sum(counts.values())),
    }


def _render_map(fmap: folium.Map, key: str) -> None:
    html = fmap.get_root().render()
    html = html.replace(
        "<style>",
        "<style>html, body {height: 100%; margin: 0;} .folium-map {width: 100% !important; height: 680px !important;}",
        1,
    )
    components.html(html, height=700, scrolling=False)


def _coverage_polygon_gdf(polygon_wkt: str) -> gpd.GeoDataFrame:
    geom = gpd.GeoSeries.from_wkt([polygon_wkt], crs="EPSG:4326")
    return gpd.GeoDataFrame({"geometry": geom}, crs="EPSG:4326")


def _coverage_metric_options(df: pd.DataFrame) -> List[str]:
    return [
        label
        for label, spec in COVERAGE_METRIC_SPECS.items()
        if spec["column"] in df.columns and _apply_coverage_valid_range(df, label)[spec["column"]].notna().any()
    ]


def _apply_coverage_valid_range(df: pd.DataFrame, metric_label: str) -> pd.DataFrame:
    spec = COVERAGE_METRIC_SPECS[metric_label]
    out = df.copy()
    out[spec["column"]] = pd.to_numeric(out[spec["column"]], errors="coerce")
    lower, upper = spec.get("valid_range", (None, None))
    mask = out[spec["column"]].notna()
    if lower is not None:
        mask = mask & (out[spec["column"]] >= float(lower))
    if upper is not None:
        mask = mask & (out[spec["column"]] <= float(upper))
    return out.loc[mask].copy()


def _filter_coverage_band(df: pd.DataFrame, metric_label: str, band_label: str) -> pd.DataFrame:
    spec = COVERAGE_METRIC_SPECS[metric_label]
    out = _apply_coverage_valid_range(df, metric_label)
    if band_label == "All Values":
        return out
    series = pd.to_numeric(out[spec["column"]], errors="coerce")
    for label, lower, upper in spec["bands"]:
        if label != band_label:
            continue
        mask = series.notna()
        if lower is not None and upper is not None:
            mask = mask & (series >= float(lower)) & (series < float(upper))
        elif lower is not None:
            mask = mask & (series >= float(lower))
        elif upper is not None:
            mask = mask & (series < float(upper))
        return out.loc[mask].copy()
    return out


def _coverage_band_counts(df: pd.DataFrame, metric_label: str) -> pd.DataFrame:
    spec = COVERAGE_METRIC_SPECS[metric_label]
    rows = []
    for label, _, _ in spec["bands"]:
        if label == "All Values":
            continue
        band_df = _filter_coverage_band(df, metric_label, label)
        rows.append({"band": label, "rows": int(len(band_df))})
    return pd.DataFrame(rows)


def _matplotlib_color(color: str):
    color_text = str(color).strip()
    if color_text.startswith("rgb(") and color_text.endswith(")"):
        parts = [part.strip() for part in color_text[4:-1].split(",")]
        if len(parts) == 3:
            try:
                r, g, b = [max(0, min(255, int(float(part)))) / 255.0 for part in parts]
                return (r, g, b, 1.0)
            except ValueError:
                pass
    return to_rgba(color_text)


def _build_coverage_map(
    polygon_gdf: gpd.GeoDataFrame,
    point_df: pd.DataFrame,
    metric_label: str,
):
    spec = COVERAGE_METRIC_SPECS[metric_label]
    work = _apply_coverage_valid_range(point_df, metric_label)
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work = work.dropna(subset=["lat", "lon", spec["column"]]).copy()

    fig, ax = plt.subplots(figsize=(10, 8))
    polygon_gdf.boundary.plot(ax=ax, color="#dc2626", linewidth=2.0)
    if not polygon_gdf.empty:
        polygon_gdf.plot(ax=ax, color="#fee2e2", alpha=0.15)
    if work.empty:
        ax.set_title(f"{metric_label} Coverage Map")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="datalim")
        return fig

    if not polygon_gdf.empty:
        minx, miny, maxx, maxy = polygon_gdf.total_bounds
        span_x = max(maxx - minx, 1e-6)
        span_y = max(maxy - miny, 1e-6)
        gridsize = max(35, int(max(span_x, span_y) / min(span_x, span_y) * 45))
    else:
        gridsize = 55

    hexbin = ax.hexbin(
        work["lon"],
        work["lat"],
        gridsize=gridsize,
        mincnt=1,
        cmap="viridis",
        linewidths=0,
        bins="log",
    )
    cbar = fig.colorbar(hexbin, ax=ax, shrink=0.82)
    cbar.set_label("KPI point density (log count)")

    ax.set_title(f"{metric_label} Coverage Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if not polygon_gdf.empty:
        pad_x = max((maxx - minx) * 0.02, 0.0005)
        pad_y = max((maxy - miny) * 0.02, 0.0005)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def _build_site_reference_map(
    polygon_gdf: gpd.GeoDataFrame,
    site_df: pd.DataFrame,
    title: str,
    lat_col: str,
    lon_col: str,
    color: str,
    marker: str,
):
    fig, ax = plt.subplots(figsize=(8, 7))
    polygon_gdf.boundary.plot(ax=ax, color="#dc2626", linewidth=2.0)
    if not polygon_gdf.empty:
        polygon_gdf.plot(ax=ax, color="#fee2e2", alpha=0.15)

    work = site_df.copy()
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=[lat_col, lon_col]).copy()

    if not work.empty:
        ax.scatter(
            work[lon_col],
            work[lat_col],
            s=40,
            marker=marker,
            c=color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.85,
            rasterized=True,
        )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if not polygon_gdf.empty:
        minx, miny, maxx, maxy = polygon_gdf.total_bounds
        pad_x = max((maxx - minx) * 0.02, 0.0005)
        pad_y = max((maxy - miny) * 0.02, 0.0005)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def _sample_dt_points_for_map(point_df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    if point_df.empty:
        return point_df
    work = point_df.copy()
    if "lat" in work.columns:
        work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    if "lon" in work.columns:
        work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work = work.dropna(subset=[col for col in ["lat", "lon"] if col in work.columns]).copy()
    if len(work) <= max_points:
        return work
    return work.sample(n=max_points, random_state=42).copy()


def _build_dt_points_map(
    polygon_gdf: gpd.GeoDataFrame,
    point_df: pd.DataFrame,
    title: str,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    polygon_gdf.boundary.plot(ax=ax, color="#dc2626", linewidth=2.0)
    if not polygon_gdf.empty:
        polygon_gdf.plot(ax=ax, color="#fee2e2", alpha=0.12)
    work = _sample_dt_points_for_map(point_df)
    if not work.empty:
        ax.scatter(
            work["lon"],
            work["lat"],
            s=8,
            marker="o",
            c="#111827",
            edgecolors="none",
            alpha=0.45,
            rasterized=True,
            label=f"DT points shown ({len(work)})",
        )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if not polygon_gdf.empty:
        minx, miny, maxx, maxy = polygon_gdf.total_bounds
        pad_x = max((maxx - minx) * 0.02, 0.0005)
        pad_y = max((maxy - miny) * 0.02, 0.0005)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def _build_grid_metric_map(
    polygon_gdf: gpd.GeoDataFrame,
    grid_cells_df: pd.DataFrame,
    grid_metric_df: pd.DataFrame,
    metric_col: str,
    title: str,
    site_df: Optional[pd.DataFrame] = None,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    polygon_gdf.boundary.plot(ax=ax, color="#dc2626", linewidth=2.0)
    if not polygon_gdf.empty:
        polygon_gdf.plot(ax=ax, color="#fee2e2", alpha=0.10)

    if grid_cells_df.empty or grid_metric_df.empty or metric_col not in grid_metric_df.columns:
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if not polygon_gdf.empty:
            minx, miny, maxx, maxy = polygon_gdf.total_bounds
            pad_x = max((maxx - minx) * 0.02, 0.0005)
            pad_y = max((maxy - miny) * 0.02, 0.0005)
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        return fig

    work_cells = grid_cells_df.copy()
    if "geometry_wkt" in work_cells.columns:
        work_cells = gpd.GeoDataFrame(
            work_cells.drop(columns=["geometry_wkt"]),
            geometry=gpd.GeoSeries.from_wkt(work_cells["geometry_wkt"]),
            crs="EPSG:4326",
        )
    elif "geometry" not in work_cells.columns:
        work_cells = gpd.GeoDataFrame(work_cells, geometry=[], crs="EPSG:4326")
    else:
        work_cells = gpd.GeoDataFrame(work_cells, geometry="geometry", crs="EPSG:4326")

    metric_df = grid_metric_df.copy()
    metric_df["grid_id"] = pd.to_numeric(metric_df["grid_id"], errors="coerce")
    work_cells["grid_id"] = pd.to_numeric(work_cells["grid_id"], errors="coerce")
    merged = work_cells.merge(metric_df[["grid_id", metric_col]], on="grid_id", how="left")
    merged[metric_col] = pd.to_numeric(merged[metric_col], errors="coerce")
    plot_df = merged.loc[merged[metric_col].notna()].copy()

    if not plot_df.empty:
        plot_df.plot(
            ax=ax,
            column=metric_col,
            cmap="viridis",
            linewidth=0.05,
            edgecolor="none",
            legend=True,
            legend_kwds={"shrink": 0.82, "label": metric_col},
        )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if not polygon_gdf.empty:
        minx, miny, maxx, maxy = polygon_gdf.total_bounds
        pad_x = max((maxx - minx) * 0.02, 0.0005)
        pad_y = max((maxy - miny) * 0.02, 0.0005)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    if site_df is not None and not site_df.empty:
        site_work = site_df.copy()
        if "lat" in site_work.columns and "lon" in site_work.columns:
            site_work["lat"] = pd.to_numeric(site_work["lat"], errors="coerce")
            site_work["lon"] = pd.to_numeric(site_work["lon"], errors="coerce")
            site_work = site_work.dropna(subset=["lat", "lon"]).copy()
            if not site_work.empty:
                ax.scatter(
                    site_work["lon"],
                    site_work["lat"],
                    s=42,
                    marker="^",
                    c="#111827",
                    edgecolors="white",
                    linewidths=0.45,
                    alpha=0.9,
                    rasterized=True,
                    label=f"Project Sites ({len(site_work)})",
                )
    fig.tight_layout()
    return fig


def _merge_grid_metric_geometry(grid_cells_df: pd.DataFrame, grid_metric_df: pd.DataFrame, metric_col: str) -> gpd.GeoDataFrame:
    work_cells = grid_cells_df.copy()
    if "geometry_wkt" in work_cells.columns:
        work_cells = gpd.GeoDataFrame(
            work_cells.drop(columns=["geometry_wkt"]),
            geometry=gpd.GeoSeries.from_wkt(work_cells["geometry_wkt"]),
            crs="EPSG:4326",
        )
    elif "geometry" not in work_cells.columns:
        work_cells = gpd.GeoDataFrame(work_cells, geometry=[], crs="EPSG:4326")
    else:
        work_cells = gpd.GeoDataFrame(work_cells, geometry="geometry", crs="EPSG:4326")
    metric_df = grid_metric_df.copy()
    metric_df["grid_id"] = pd.to_numeric(metric_df["grid_id"], errors="coerce")
    work_cells["grid_id"] = pd.to_numeric(work_cells["grid_id"], errors="coerce")
    merged = work_cells.merge(metric_df[["grid_id", metric_col]], on="grid_id", how="left")
    return merged


def _build_grid_metric_triptych(
    polygon_gdf: gpd.GeoDataFrame,
    grid_cells_df: pd.DataFrame,
    bucket_frames: Dict[str, pd.DataFrame],
    metric_col: str,
    title_prefix: str,
    discrete_bands: Optional[List[Dict[str, object]]] = None,
    bucket_site_frames: Optional[Dict[str, pd.DataFrame]] = None,
):
    buckets = list(bucket_frames.keys())
    fig, axes = plt.subplots(1, len(buckets), figsize=(6.2 * len(buckets), 7.2))
    if len(buckets) == 1:
        axes = [axes]

    merged_frames: Dict[str, gpd.GeoDataFrame] = {}
    global_min = None
    global_max = None
    for bucket_name, bucket_df in bucket_frames.items():
        merged = _merge_grid_metric_geometry(grid_cells_df, bucket_df, metric_col)
        merged[metric_col] = pd.to_numeric(merged[metric_col], errors="coerce")
        merged_frames[bucket_name] = merged
        vals = merged[metric_col].dropna()
        if not vals.empty:
            local_min = float(vals.min())
            local_max = float(vals.max())
            global_min = local_min if global_min is None else min(global_min, local_min)
            global_max = local_max if global_max is None else max(global_max, local_max)

    for ax, bucket_name in zip(axes, buckets):
        polygon_gdf.boundary.plot(ax=ax, color="#991b1b", linewidth=1.5)
        if not polygon_gdf.empty:
            polygon_gdf.plot(ax=ax, color="#fee2e2", alpha=0.08)
        merged = merged_frames[bucket_name]
        plot_df = merged.loc[merged[metric_col].notna()].copy()
        if not plot_df.empty:
            if discrete_bands:
                def _pick_band_color(value):
                    numeric_value = pd.to_numeric(value, errors="coerce")
                    if pd.isna(numeric_value):
                        return None
                    for band in discrete_bands:
                        lower = band.get("min")
                        upper = band.get("max")
                        lower_ok = True if lower is None else float(numeric_value) >= float(lower)
                        upper_ok = True if upper is None else float(numeric_value) < float(upper)
                        if lower_ok and upper_ok:
                            return str(band["color"])
                    return str(discrete_bands[-1]["color"])

                plot_df["__band_color"] = plot_df[metric_col].apply(_pick_band_color)
                plot_df = plot_df.loc[plot_df["__band_color"].notna()].copy()
                if not plot_df.empty:
                    plot_df.plot(
                        ax=ax,
                        color=plot_df["__band_color"],
                        linewidth=0.03,
                        edgecolor="none",
                    )
            else:
                plot_df.plot(
                    ax=ax,
                    column=metric_col,
                    cmap="viridis",
                    linewidth=0.03,
                    edgecolor="none",
                    vmin=global_min,
                    vmax=global_max,
                    legend=True,
                    legend_kwds={"shrink": 0.72, "label": metric_col},
                )
        ax.set_title(f"{title_prefix}\n{bucket_name}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if not polygon_gdf.empty:
            minx, miny, maxx, maxy = polygon_gdf.total_bounds
            pad_x = max((maxx - minx) * 0.02, 0.0005)
            pad_y = max((maxy - miny) * 0.02, 0.0005)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.12)
        if bucket_site_frames is not None and bucket_name in bucket_site_frames:
            bucket_site_df = bucket_site_frames[bucket_name]
            if bucket_site_df is not None and not bucket_site_df.empty and "lat" in bucket_site_df.columns and "lon" in bucket_site_df.columns:
                site_work = bucket_site_df.copy()
                site_work["lat"] = pd.to_numeric(site_work["lat"], errors="coerce")
                site_work["lon"] = pd.to_numeric(site_work["lon"], errors="coerce")
                site_work = site_work.dropna(subset=["lat", "lon"]).copy()
                if not site_work.empty:
                    ax.scatter(
                        site_work["lon"],
                        site_work["lat"],
                        s=34,
                        marker="^",
                        c="#111827",
                        edgecolors="white",
                        linewidths=0.45,
                        alpha=0.85,
                        rasterized=True,
                    )
        if discrete_bands:
            legend_handles = [
                Patch(facecolor=str(band["color"]), edgecolor="none", label=str(band["label"]))
                for band in discrete_bands
            ]
            ax.legend(handles=legend_handles, fontsize=8, loc="upper right", frameon=True)
    fig.tight_layout()
    return fig


def _build_clutter_triptych(
    polygon_gdf: gpd.GeoDataFrame,
    grid_cells_df: pd.DataFrame,
    bucket_frames: Dict[str, pd.DataFrame],
):
    buckets = list(bucket_frames.keys())
    fig, axes = plt.subplots(1, len(buckets), figsize=(6.4 * len(buckets), 7.4))
    if len(buckets) == 1:
        axes = [axes]

    color_map = {
        "Dense Urban": "#7f1d1d",
        "Urban": "#ea580c",
        "Suburban": "#facc15",
        "Vegetation": "#16a34a",
        "Water": "#2563eb",
        "Rural/Open": "#94a3b8",
    }
    categories = ["Dense Urban", "Urban", "Suburban", "Vegetation", "Water", "Rural/Open"]

    for ax, bucket_name in zip(axes, buckets):
        polygon_gdf.boundary.plot(ax=ax, color="#111827", linewidth=1.2)
        if not polygon_gdf.empty:
            polygon_gdf.plot(ax=ax, color="#f8fafc", alpha=0.1)
        merged = _merge_grid_metric_geometry(grid_cells_df, bucket_frames[bucket_name], "clutter_class")
        plot_df = merged.loc[merged["clutter_class"].notna()].copy()
        if not plot_df.empty:
            plot_df["__clutter_color"] = plot_df["clutter_class"].astype(str).map(color_map).fillna("#6b7280")
            plot_df.plot(
                ax=ax,
                color=plot_df["__clutter_color"],
                linewidth=0.03,
                edgecolor="none",
            )
        ax.set_title(f"Clutter Class\n{bucket_name}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if not polygon_gdf.empty:
            minx, miny, maxx, maxy = polygon_gdf.total_bounds
            pad_x = max((maxx - minx) * 0.02, 0.0005)
            pad_y = max((maxy - miny) * 0.02, 0.0005)
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.12)
        legend_handles = [Patch(facecolor=color_map[key], edgecolor="none", label=key) for key in categories]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right", frameon=True)
    fig.tight_layout()
    return fig


def _build_grid_delta_map(
    polygon_gdf: gpd.GeoDataFrame,
    grid_cells_df: pd.DataFrame,
    start_df: pd.DataFrame,
    end_df: pd.DataFrame,
    metric_col: str,
    start_label: str,
    end_label: str,
    title_metric_label: Optional[str] = None,
):
    start_work = start_df[["grid_id", metric_col]].copy().rename(columns={metric_col: "__start"})
    end_work = end_df[["grid_id", metric_col]].copy().rename(columns={metric_col: "__end"})
    delta_df = start_work.merge(end_work, on="grid_id", how="outer")
    delta_df["delta"] = pd.to_numeric(delta_df["__end"], errors="coerce") - pd.to_numeric(delta_df["__start"], errors="coerce")
    merged = _merge_grid_metric_geometry(grid_cells_df, delta_df.rename(columns={"delta": metric_col}), metric_col)
    merged[metric_col] = pd.to_numeric(merged[metric_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 8))
    polygon_gdf.boundary.plot(ax=ax, color="#111827", linewidth=1.5)
    if not polygon_gdf.empty:
        polygon_gdf.plot(ax=ax, color="#f8fafc", alpha=0.08)
    plot_df = merged.loc[merged[metric_col].notna()].copy()
    if not plot_df.empty:
        max_abs = max(abs(float(plot_df[metric_col].min())), abs(float(plot_df[metric_col].max())))
        plot_df.plot(
            ax=ax,
            column=metric_col,
            cmap="RdYlGn",
            linewidth=0.03,
            edgecolor="none",
            vmin=-max_abs,
            vmax=max_abs,
            legend=True,
            legend_kwds={"shrink": 0.78, "label": f"{metric_col} delta"},
        )
    display_label = title_metric_label or metric_col
    ax.set_title(f"{display_label} Change Map\n{start_label} -> {end_label}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if not polygon_gdf.empty:
        minx, miny, maxx, maxy = polygon_gdf.total_bounds
        pad_x = max((maxx - minx) * 0.02, 0.0005)
        pad_y = max((maxy - miny) * 0.02, 0.0005)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.12)
    fig.tight_layout()
    return fig


def _build_clutter_transition_map(
    polygon_gdf: gpd.GeoDataFrame,
    grid_cells_df: pd.DataFrame,
    start_df: pd.DataFrame,
    end_df: pd.DataFrame,
    start_label: str,
    end_label: str,
):
    trans_df = start_df[["grid_id", "clutter_class"]].copy().rename(columns={"clutter_class": "__start"})
    trans_df = trans_df.merge(
        end_df[["grid_id", "clutter_class"]].copy().rename(columns={"clutter_class": "__end"}),
        on="grid_id",
        how="outer",
    )
    trans_df["transition"] = np.where(
        trans_df["__start"].astype(str) == trans_df["__end"].astype(str),
        "Unchanged",
        trans_df["__start"].astype(str) + " -> " + trans_df["__end"].astype(str),
    )
    merged = _merge_grid_metric_geometry(grid_cells_df, trans_df.rename(columns={"transition": "transition_label"}), "transition_label")
    color_map = {
        "Unchanged": "#cbd5e1",
        "Suburban -> Urban": "#fb923c",
        "Urban -> Dense Urban": "#b91c1c",
        "Suburban -> Dense Urban": "#7f1d1d",
        "Urban -> Suburban": "#facc15",
        "Dense Urban -> Urban": "#fdba74",
        "Vegetation -> Urban": "#84cc16",
        "Urban -> Vegetation": "#22c55e",
        "Water -> Urban": "#60a5fa",
        "Urban -> Water": "#1d4ed8",
    }
    merged["__transition_color"] = merged["transition_label"].astype(str).map(color_map).fillna("#6b7280")

    fig, ax = plt.subplots(figsize=(10, 8))
    polygon_gdf.boundary.plot(ax=ax, color="#111827", linewidth=1.5)
    if not polygon_gdf.empty:
        polygon_gdf.plot(ax=ax, color="#f8fafc", alpha=0.08)
    plot_df = merged.loc[merged["transition_label"].notna()].copy()
    if not plot_df.empty:
        plot_df.plot(ax=ax, color=plot_df["__transition_color"], linewidth=0.03, edgecolor="none")
    ax.set_title(f"Clutter Transition Map\n{start_label} -> {end_label}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if not polygon_gdf.empty:
        minx, miny, maxx, maxy = polygon_gdf.total_bounds
        pad_x = max((maxx - minx) * 0.02, 0.0005)
        pad_y = max((maxy - miny) * 0.02, 0.0005)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.12)
    top_labels = plot_df["transition_label"].astype(str).value_counts().head(8).index.tolist() if not plot_df.empty else ["Unchanged"]
    legend_handles = [Patch(facecolor=color_map.get(label, "#6b7280"), edgecolor="none", label=label) for label in top_labels]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", frameon=True)
    fig.tight_layout()
    return fig


def _build_transition_kpi_delta_map(
    polygon_gdf: gpd.GeoDataFrame,
    grid_cells_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    transition_label: str,
    kpi_label: str,
):
    plot_source = transition_df.copy()
    if transition_label != "All Changed":
        plot_source = plot_source.loc[plot_source["transition"].astype(str) == str(transition_label)].copy()
    else:
        plot_source = plot_source.loc[plot_source["transition"].astype(str) != "Unchanged"].copy()
    plot_source["delta_kpi"] = pd.to_numeric(plot_source["delta_kpi"], errors="coerce")
    merged = _merge_grid_metric_geometry(
        grid_cells_df,
        plot_source[["grid_id", "delta_kpi"]],
        "delta_kpi",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    polygon_gdf.boundary.plot(ax=ax, color="#111827", linewidth=1.5)
    if not polygon_gdf.empty:
        polygon_gdf.plot(ax=ax, color="#f8fafc", alpha=0.08)
    plot_df = merged.loc[merged["delta_kpi"].notna()].copy()
    if not plot_df.empty:
        max_abs = max(abs(float(plot_df["delta_kpi"].min())), abs(float(plot_df["delta_kpi"].max())))
        max_abs = max(max_abs, 1.0)
        plot_df.plot(
            ax=ax,
            column="delta_kpi",
            cmap="RdYlGn",
            linewidth=0.03,
            edgecolor="none",
            vmin=-max_abs,
            vmax=max_abs,
            legend=True,
            legend_kwds={"shrink": 0.78, "label": f"{kpi_label} delta"},
        )
    ax.set_title(f"{transition_label}\n{kpi_label} Change")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if not polygon_gdf.empty:
        minx, miny, maxx, maxy = polygon_gdf.total_bounds
        pad_x = max((maxx - minx) * 0.02, 0.0005)
        pad_y = max((maxy - miny) * 0.02, 0.0005)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.12)
    fig.tight_layout()
    return fig


def _opencellid_operator_label(mcc, mnc) -> str:
    try:
        key = (int(float(mcc)), int(float(mnc)))
    except Exception:
        return f"{mcc}/{mnc}"
    if key in OPENCELLID_OPERATOR_LABELS:
        return OPENCELLID_OPERATOR_LABELS[key]
    for operator_name, operator_keys in OPENCELLID_OPERATOR_GROUPS.items():
        if key in operator_keys:
            return operator_name
    return f"{key[0]}/{key[1]}"


def _render_opencellid_probe_page(project_id: int) -> None:
    runs = _list_opencellid_probe_runs(int(project_id))
    st.subheader("OpenCellID Probe Runs")
    if not runs:
        st.info("No OpenCellID probe runs found yet. Use the sidebar OpenCellID form to launch one.")
        return

    latest_by_bucket: Dict[str, Path] = {}
    for run in runs:
        summary = _load_summary(run)
        bucket = str(summary.get("bucket_label") or "")
        if bucket and bucket not in latest_by_bucket:
            latest_by_bucket[bucket] = run

    ordered_buckets = ["PART_1", "PART_2", "PART_3"]
    available_buckets = [bucket for bucket in ordered_buckets if bucket in latest_by_bucket]
    if not available_buckets:
        st.info("No bucket-labelled OpenCellID runs found.")
        return

    tabs = st.tabs(available_buckets)
    for tab, bucket_name in zip(tabs, available_buckets):
        with tab:
            run_dir = latest_by_bucket[bucket_name]
            summary = _load_summary(run_dir)
            result_csv = summary.get("artifacts", {}).get("result_csv")
            if not result_csv:
                st.warning(f"{bucket_name} probe returned no polygon rows.")
                continue
            result_path = run_dir / result_csv
            probe_df = _safe_read_csv(result_path)
            if probe_df.empty:
                st.warning(f"{bucket_name} result CSV is empty.")
                continue

            for col in ["lat", "lon", "mcc", "mnc", "samples", "range"]:
                if col in probe_df.columns:
                    probe_df[col] = pd.to_numeric(probe_df[col], errors="coerce")
            probe_df["operator_label"] = [
                _opencellid_operator_label(row.get("mcc"), row.get("mnc"))
                for _, row in probe_df.iterrows()
            ]

            polygon_gdf = _coverage_polygon_gdf(summary.get("polygon_wkt", DEFAULT_COVERAGE_POLYGON_WKT))
            st.caption(
                f"{bucket_name}: {summary.get('bucket_start')} to {summary.get('bucket_end')} | "
                f"rows={summary.get('result_rows')} | tiles={summary.get('tile_count')}"
            )

            control_cols = st.columns(3)
            operator_options = ["All"] + sorted(probe_df["operator_label"].dropna().astype(str).unique().tolist())
            operator_choice = control_cols[0].selectbox(
                f"{bucket_name} Operator",
                options=operator_options,
                index=0,
                key=f"opencellid_operator_{bucket_name}",
            )
            radio_options = ["All"] + sorted(probe_df["radio"].dropna().astype(str).unique().tolist()) if "radio" in probe_df.columns else ["All"]
            radio_choice = control_cols[1].selectbox(
                f"{bucket_name} Radio",
                options=radio_options,
                index=0,
                key=f"opencellid_radio_{bucket_name}",
            )
            show_full_table = control_cols[2].checkbox(
                f"{bucket_name} Show Raw Rows",
                value=False,
                key=f"opencellid_table_{bucket_name}",
            )

            filtered_df = probe_df.copy()
            if operator_choice != "All":
                filtered_df = filtered_df[filtered_df["operator_label"].astype(str) == operator_choice].copy()
            if radio_choice != "All" and "radio" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["radio"].astype(str) == radio_choice].copy()

            m1, m2, m3 = st.columns(3)
            m1.metric("Reference Rows", int(len(filtered_df)))
            m2.metric("Unique Operators", int(filtered_df["operator_label"].astype(str).nunique()))
            m3.metric("Unique Radios", int(filtered_df["radio"].astype(str).nunique()) if "radio" in filtered_df.columns else 0)

            map_fig = _build_site_reference_map(
                polygon_gdf=polygon_gdf,
                site_df=filtered_df,
                title=f"{bucket_name} OpenCellID Reference",
                lat_col="lat",
                lon_col="lon",
                color="#f59e0b",
                marker="s",
            )
            st.pyplot(map_fig, use_container_width=True)
            plt.close(map_fig)

            if show_full_table:
                show_cols = [col for col in ["operator_label", "mcc", "mnc", "radio", "lac", "cellid", "tac", "lat", "lon", "samples", "range"] if col in filtered_df.columns]
                st.dataframe(filtered_df[show_cols], use_container_width=True)


def _load_latest_opencellid_probe_frames(project_id: int) -> Dict[str, Dict]:
    runs = _list_opencellid_probe_runs(int(project_id))
    latest_by_bucket: Dict[str, Dict] = {}
    for run in runs:
        summary = _load_summary(run)
        bucket = str(summary.get("bucket_label") or "")
        if not bucket or bucket in latest_by_bucket:
            continue
        result_csv = summary.get("artifacts", {}).get("result_csv")
        if not result_csv:
            continue
        result_path = run / result_csv
        if not result_path.exists():
            continue
        probe_df = _safe_read_csv(result_path)
        if probe_df.empty:
            continue
        for col in ["lat", "lon", "mcc", "mnc", "samples", "range"]:
            if col in probe_df.columns:
                probe_df[col] = pd.to_numeric(probe_df[col], errors="coerce")
        probe_df["operator_label"] = [
            _opencellid_operator_label(row.get("mcc"), row.get("mnc"))
            for _, row in probe_df.iterrows()
        ]
        latest_by_bucket[bucket] = {
            "run_dir": run,
            "summary": summary,
            "df": probe_df,
        }
    return latest_by_bucket


def _render_opencellid_compare_page(project_id: int) -> None:
    st.subheader("OpenCellID API Comparison")
    latest_by_bucket = _load_latest_opencellid_probe_frames(int(project_id))
    ordered_buckets = ["PART_1", "PART_2", "PART_3"]
    available_buckets = [bucket for bucket in ordered_buckets if bucket in latest_by_bucket]
    if not available_buckets:
        st.info("No OpenCellID probe results found yet. Run the probe from the sidebar first.")
        return

    frames = [latest_by_bucket[bucket]["df"] for bucket in available_buckets]
    combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    operator_options = ["All"] + sorted(combined_df["operator_label"].dropna().astype(str).unique().tolist())
    radio_options = ["All"] + sorted(combined_df["radio"].dropna().astype(str).unique().tolist()) if "radio" in combined_df.columns else ["All"]

    control_cols = st.columns(3)
    operator_choice = control_cols[0].selectbox("Operator Filter", options=operator_options, index=0, key="opencellid_compare_operator")
    radio_choice = control_cols[1].selectbox("Radio Filter", options=radio_options, index=0, key="opencellid_compare_radio")
    show_table = control_cols[2].checkbox("Show API Rows", value=False, key="opencellid_compare_table")

    st.caption(
        "This page shows only OpenCellID API probe results by bucket. "
        "If PART_1, PART_2, and PART_3 look the same, the API is behaving like a current reference source rather than true history."
    )

    summary_cols = st.columns(len(available_buckets))
    for col, bucket_name in zip(summary_cols, available_buckets):
        bucket_payload = latest_by_bucket[bucket_name]
        bucket_summary = bucket_payload["summary"]
        bucket_df = bucket_payload["df"].copy()
        if operator_choice != "All":
            bucket_df = bucket_df[bucket_df["operator_label"].astype(str) == operator_choice].copy()
        if radio_choice != "All" and "radio" in bucket_df.columns:
            bucket_df = bucket_df[bucket_df["radio"].astype(str) == radio_choice].copy()
        with col:
            st.markdown(f"**{bucket_name}**")
            st.caption(f"{bucket_summary.get('bucket_start')} to {bucket_summary.get('bucket_end')}")
            st.metric("API Rows", int(len(bucket_df)))
            st.metric("Unique Cells", int(bucket_df[["mcc", "mnc", "lac", "cellid"]].drop_duplicates().shape[0]) if {"mcc", "mnc", "lac", "cellid"}.issubset(bucket_df.columns) else int(len(bucket_df)))
            st.metric("Unique Points", int(bucket_df[["lat", "lon"]].drop_duplicates().shape[0]) if {"lat", "lon"}.issubset(bucket_df.columns) else int(len(bucket_df)))

    map_cols = st.columns(len(available_buckets))
    for col, bucket_name in zip(map_cols, available_buckets):
        bucket_payload = latest_by_bucket[bucket_name]
        bucket_summary = bucket_payload["summary"]
        bucket_df = bucket_payload["df"].copy()
        if operator_choice != "All":
            bucket_df = bucket_df[bucket_df["operator_label"].astype(str) == operator_choice].copy()
        if radio_choice != "All" and "radio" in bucket_df.columns:
            bucket_df = bucket_df[bucket_df["radio"].astype(str) == radio_choice].copy()
        polygon_gdf = _coverage_polygon_gdf(bucket_summary.get("polygon_wkt", DEFAULT_COVERAGE_POLYGON_WKT))
        with col:
            fig = _build_site_reference_map(
                polygon_gdf=polygon_gdf,
                site_df=bucket_df,
                title=f"{bucket_name} OpenCellID",
                lat_col="lat",
                lon_col="lon",
                color="#f59e0b",
                marker="s",
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    if show_table:
        for bucket_name in available_buckets:
            bucket_df = latest_by_bucket[bucket_name]["df"].copy()
            if operator_choice != "All":
                bucket_df = bucket_df[bucket_df["operator_label"].astype(str) == operator_choice].copy()
            if radio_choice != "All" and "radio" in bucket_df.columns:
                bucket_df = bucket_df[bucket_df["radio"].astype(str) == radio_choice].copy()
            st.markdown(f"**{bucket_name} API Rows**")
            show_cols = [col for col in ["operator_label", "mcc", "mnc", "radio", "lac", "cellid", "tac", "lat", "lon", "samples", "range"] if col in bucket_df.columns]
            st.dataframe(bucket_df[show_cols], use_container_width=True)


def _prepare_site_selection_df(site_df: pd.DataFrame) -> pd.DataFrame:
    work = site_df.copy()
    if "Node_Cell_ID" not in work.columns and "cell_id" in work.columns:
        work["Node_Cell_ID"] = work["cell_id"].astype(str)
    for col in ["nodeb_id", "cell_id", "Node_Cell_ID", "Site ID"]:
        if col in work.columns:
            work[col] = work[col].astype(str)
    dashboard_nodeb = pd.Series(index=work.index, dtype=object)
    if "nodeb_id" in work.columns:
        nodeb_series = work["nodeb_id"].astype(str).str.strip()
        dashboard_nodeb = nodeb_series.where(~nodeb_series.isin(["", "nan", "None"]))
    if "Site ID" in work.columns:
        site_id_series = work["Site ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        dashboard_nodeb = dashboard_nodeb.fillna(site_id_series.where(~site_id_series.isin(["", "nan", "None"])))
    if "Node_Cell_ID" in work.columns:
        derived_series = work["Node_Cell_ID"].astype(str).str.split("_").str[0].str.strip()
        dashboard_nodeb = dashboard_nodeb.fillna(derived_series.where(~derived_series.isin(["", "nan", "None"])))
    work["dashboard_nodeb_id"] = dashboard_nodeb.astype(str)
    return work


def _render_metric_compare(summary: Dict, metric_name: str) -> None:
    baseline = summary.get("full_metrics", {}).get("baseline", {}).get(metric_name)
    if baseline:
        st.markdown(f"**{metric_name} Baseline**")
        _metric_row(metric_name.replace("_meas", ""), baseline)
    experimental = summary.get("full_metrics", {}).get("experimental", {}).get(metric_name)
    if experimental:
        st.markdown(f"**{metric_name} Experimental Geo Model**")
        _metric_row(metric_name.replace("_meas", ""), experimental)


def _clip_series(series: pd.Series, metric_name: str) -> pd.Series:
    low, high = KPI_LIMITS[metric_name]
    return pd.to_numeric(series, errors="coerce").clip(lower=low, upper=high)


def _prepare_kpi_eval(dt_eval: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    meas_col = f"{metric_name}_meas"
    pred_col = f"{metric_name}_pred"
    geo_pred_col = f"{metric_name}_pred_geo"
    cols = [c for c in ["lat", "lon", meas_col, pred_col, geo_pred_col, "morphology_cluster"] if c in dt_eval.columns]
    work = dt_eval[cols].copy()
    if meas_col in work.columns:
        work[meas_col] = _clip_series(work[meas_col], metric_name)
    if pred_col in work.columns:
        work[pred_col] = _clip_series(work[pred_col], metric_name)
    if geo_pred_col in work.columns:
        work[geo_pred_col] = _clip_series(work[geo_pred_col], metric_name)
    if meas_col in work.columns and pred_col in work.columns:
        work["baseline_error"] = (work[meas_col] - work[pred_col]).abs()
    if meas_col in work.columns and geo_pred_col in work.columns:
        work["experimental_error"] = (work[meas_col] - work[geo_pred_col]).abs()
    return work.dropna()


def _render_range_summary(dt_eval: pd.DataFrame) -> None:
    rows = []
    for metric_name in ("RSRP", "RSRQ", "SINR"):
        work = _prepare_kpi_eval(dt_eval, metric_name)
        meas_col = f"{metric_name}_meas"
        pred_col = f"{metric_name}_pred"
        geo_col = f"{metric_name}_pred_geo"
        if meas_col in work.columns and not work.empty:
            rows.append({
                "metric": metric_name,
                "series": "DT Measured",
                "min": round(float(work[meas_col].min()), 4),
                "max": round(float(work[meas_col].max()), 4),
                "mean": round(float(work[meas_col].mean()), 4),
            })
        if pred_col in work.columns and not work.empty:
            rows.append({
                "metric": metric_name,
                "series": "Baseline RF",
                "min": round(float(work[pred_col].min()), 4),
                "max": round(float(work[pred_col].max()), 4),
                "mean": round(float(work[pred_col].mean()), 4),
            })
        if geo_col in work.columns and not work.empty:
            rows.append({
                "metric": metric_name,
                "series": "Experimental Geo",
                "min": round(float(work[geo_col].min()), 4),
                "max": round(float(work[geo_col].max()), 4),
                "mean": round(float(work[geo_col].mean()), 4),
            })
    if rows:
        st.markdown("**KPI Range Summary**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_kpi_distribution(dt_eval: pd.DataFrame, metric_name: str) -> None:
    work = _prepare_kpi_eval(dt_eval, metric_name)
    if work.empty:
        st.info(f"No data available for {metric_name} distribution.")
        return
    meas_col = f"{metric_name}_meas"
    pred_col = f"{metric_name}_pred"
    geo_pred_col = f"{metric_name}_pred_geo"

    fig = go.Figure()
    for col, label, color in [
        (meas_col, "DT Measured", "#111827"),
        (pred_col, "Baseline RF", "#2563eb"),
        (geo_pred_col, "Experimental Geo", "#dc2626"),
    ]:
        if col in work.columns:
            fig.add_trace(
                go.Histogram(
                    x=work[col],
                    name=label,
                    opacity=0.55,
                    nbinsx=45,
                    marker_color=color,
                )
            )
    fig.update_layout(
        title=f"{metric_name} Distribution",
        barmode="overlay",
        xaxis_title=metric_name,
        yaxis_title="Count",
        legend_title="Series",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_scatter_validation(dt_eval: pd.DataFrame, metric_name: str) -> None:
    work = _prepare_kpi_eval(dt_eval, metric_name)
    if work.empty:
        st.info(f"No data available for {metric_name} validation scatter.")
        return
    meas_col = f"{metric_name}_meas"
    pred_col = f"{metric_name}_pred"
    geo_pred_col = f"{metric_name}_pred_geo"
    row = st.columns(2)
    for idx, (col, title) in enumerate([
        (pred_col, f"{metric_name}: DT vs Baseline RF"),
        (geo_pred_col, f"{metric_name}: DT vs Experimental Geo"),
    ]):
        if col not in work.columns:
            continue
        scatter = px.scatter(
            work,
            x=meas_col,
            y=col,
            color="morphology_cluster" if "morphology_cluster" in work.columns else None,
            opacity=0.5,
            title=title,
        )
        scatter.add_shape(
            type="line",
            x0=work[meas_col].min(),
            y0=work[meas_col].min(),
            x1=work[meas_col].max(),
            y1=work[meas_col].max(),
            line=dict(color="black", dash="dash"),
        )
        with row[idx]:
            st.plotly_chart(scatter, use_container_width=True)


def _render_error_distribution(dt_eval: pd.DataFrame, metric_name: str) -> None:
    work = _prepare_kpi_eval(dt_eval, metric_name)
    if work.empty or ("baseline_error" not in work.columns and "experimental_error" not in work.columns):
        st.info(f"No data available for {metric_name} error distribution.")
        return

    fig = go.Figure()
    for col, label, color in [
        ("baseline_error", "Baseline Abs Error", "#2563eb"),
        ("experimental_error", "Experimental Abs Error", "#dc2626"),
    ]:
        if col in work.columns:
            fig.add_trace(
                go.Histogram(
                    x=work[col],
                    name=label,
                    opacity=0.6,
                    nbinsx=45,
                    marker_color=color,
                )
            )
    fig.update_layout(
        title=f"{metric_name} Absolute Error Distribution",
        barmode="overlay",
        xaxis_title="Absolute Error",
        yaxis_title="Count",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_error_image(dt_eval: pd.DataFrame, metric_name: str) -> None:
    work = _prepare_kpi_eval(dt_eval, metric_name)
    if work.empty or ("baseline_error" not in work.columns and "experimental_error" not in work.columns):
        st.info(f"No data available for {metric_name} error image.")
        return

    row = st.columns(2)
    for idx, (col, title) in enumerate([
        ("baseline_error", f"{metric_name} Baseline Abs Error"),
        ("experimental_error", f"{metric_name} Experimental Abs Error"),
    ]):
        if col not in work.columns:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5), dpi=140)
        hb = ax.hexbin(
            work["lon"],
            work["lat"],
            C=work[col],
            gridsize=36,
            reduce_C_function=np.mean,
            cmap="turbo",
            mincnt=1,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("Mean Abs Error")
        fig.tight_layout()
        with row[idx]:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def _render_feature_map(feature_df: pd.DataFrame, feature_name: str) -> None:
    if feature_name not in feature_df.columns or feature_df.empty:
        st.info(f"No data available for feature {feature_name}.")
        return

    work = feature_df.dropna(subset=["lat", "lon", feature_name]).copy()
    if work.empty:
        st.info(f"No plottable data available for feature {feature_name}.")
        return

    if pd.api.types.is_numeric_dtype(work[feature_name]) or pd.to_numeric(work[feature_name], errors="coerce").notna().sum() > 0:
        work[feature_name] = pd.to_numeric(work[feature_name], errors="coerce")
        row = st.columns(2)
        fig_map = px.scatter(
            work,
            x="lon",
            y="lat",
            color=feature_name,
            title=f"{feature_name} Spatial View",
            opacity=0.7,
            color_continuous_scale="Turbo",
        )
        with row[0]:
            st.plotly_chart(fig_map, use_container_width=True)
        fig_hist = px.histogram(work, x=feature_name, nbins=40, title=f"{feature_name} Distribution")
        with row[1]:
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        row = st.columns(2)
        fig_map = px.scatter(
            work,
            x="lon",
            y="lat",
            color=feature_name,
            title=f"{feature_name} Spatial View",
            opacity=0.7,
        )
        with row[0]:
            st.plotly_chart(fig_map, use_container_width=True)
        counts = work[feature_name].astype(str).value_counts().reset_index()
        counts.columns = [feature_name, "count"]
        fig_bar = px.bar(counts, x=feature_name, y="count", title=f"{feature_name} Counts")
        with row[1]:
            st.plotly_chart(fig_bar, use_container_width=True)


def _render_signal_image(
    holdout_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    metric_name: str,
) -> None:
    work = _prepare_kpi_eval(holdout_df, metric_name)
    if work.empty and pred_df.empty:
        st.info(f"No data available for {metric_name} signal image.")
        return

    meas_col = f"{metric_name}_meas"
    grid_pred_col = {
        "RSRP": "pred_rsrp",
        "RSRQ": "pred_rsrq",
        "SINR": "pred_sinr",
    }[metric_name]
    vmin, vmax = KPI_LIMITS[metric_name]
    panels = [
        ("holdout_dt", meas_col, f"{metric_name} Holdout DT Measured"),
        ("baseline_grid", grid_pred_col, f"{metric_name} Source RF Full Polygon"),
        ("experimental_grid", f"{grid_pred_col}_geo", f"{metric_name} Experimental Geo Full Polygon"),
    ]

    def _plot_panel(panel_kind: str, col: str, title: str) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2), dpi=140)
        hb = None
        if panel_kind == "holdout_dt":
            if col in work.columns:
                hb = ax.hexbin(
                    work["lon"],
                    work["lat"],
                    C=work[col],
                    gridsize=36,
                    reduce_C_function=np.mean,
                    cmap="viridis",
                    mincnt=1,
                    vmin=vmin,
                    vmax=vmax,
                )
        elif panel_kind == "baseline_grid":
            if col in pred_df.columns:
                grid_plot = pred_df.dropna(subset=["lat", "lon", col]).copy()
                if not grid_plot.empty:
                    hb = ax.hexbin(
                        grid_plot["lon"],
                        grid_plot["lat"],
                        C=pd.to_numeric(grid_plot[col], errors="coerce"),
                        gridsize=48,
                        reduce_C_function=np.mean,
                        cmap="viridis",
                        mincnt=1,
                        vmin=vmin,
                        vmax=vmax,
                    )
        elif panel_kind == "experimental_grid":
            if col in pred_df.columns:
                grid_plot = pred_df.dropna(subset=["lat", "lon", col]).copy()
                if not grid_plot.empty:
                    hb = ax.hexbin(
                        grid_plot["lon"],
                        grid_plot["lat"],
                        C=pd.to_numeric(grid_plot[col], errors="coerce"),
                        gridsize=48,
                        reduce_C_function=np.mean,
                        cmap="viridis",
                        mincnt=1,
                        vmin=vmin,
                        vmax=vmax,
                    )
        if hb is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            cbar = fig.colorbar(hb, ax=ax)
            cbar.set_label(metric_name)
        ax.set_title(title)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    row1 = st.columns(3)
    with row1[0]:
        _plot_panel(*panels[0])
    with row1[1]:
        _plot_panel(*panels[1])
    with row1[2]:
        _plot_panel(*panels[2])


def _prepare_optimization_compare(opt_run_dir: Path) -> Dict[str, pd.DataFrame]:
    baseline_df = pd.read_parquet(opt_run_dir / "baseline_smoothed_latest.parquet") if (opt_run_dir / "baseline_smoothed_latest.parquet").exists() else pd.DataFrame()
    optimized_affected_df = pd.read_parquet(opt_run_dir / "optimized_affected_predictions.parquet") if (opt_run_dir / "optimized_affected_predictions.parquet").exists() else pd.DataFrame()
    optimized_merged_df = pd.read_parquet(opt_run_dir / "optimized_merged_predictions.parquet") if (opt_run_dir / "optimized_merged_predictions.parquet").exists() else pd.DataFrame()
    compare_df = pd.DataFrame()
    if not baseline_df.empty and not optimized_affected_df.empty:
        before = baseline_df.copy()
        after = optimized_affected_df.copy()
        for frame in (before, after):
            for col in ["lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr"]:
                if col in frame.columns:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
            if "Node_Cell_ID" in frame.columns:
                frame["Node_Cell_ID"] = frame["Node_Cell_ID"].astype(str)
        compare_df = before.merge(
            after[["Node_Cell_ID", "lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr"]],
            on=["Node_Cell_ID", "lat", "lon"],
            how="inner",
            suffixes=("_before", "_after"),
        )
        if not compare_df.empty:
            compare_df["site_id"] = compare_df["Node_Cell_ID"].astype(str).str.split("_").str[0]
            compare_df["delta_rsrp"] = compare_df["pred_rsrp_after"] - compare_df["pred_rsrp_before"]
            compare_df["delta_rsrq"] = compare_df["pred_rsrq_after"] - compare_df["pred_rsrq_before"]
            compare_df["delta_sinr"] = compare_df["pred_sinr_after"] - compare_df["pred_sinr_before"]
    return {"baseline_df": baseline_df, "optimized_affected_df": optimized_affected_df, "optimized_merged_df": optimized_merged_df, "compare_df": compare_df}


def _render_optimization_site_changes(site_before_df: pd.DataFrame, changed_df: pd.DataFrame) -> None:
    if site_before_df.empty or changed_df.empty:
        return
    before = site_before_df.copy()
    after = changed_df.copy()
    for frame in (before, after):
        if "Node_Cell_ID" in frame.columns:
            frame["Node_Cell_ID"] = frame["Node_Cell_ID"].astype(str)
    keep_cols = ["Node_Cell_ID", "lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "tx_power", "antenna_height"]
    before = before[[c for c in keep_cols if c in before.columns]].copy()
    after = after[[c for c in keep_cols if c in after.columns]].copy()
    merged = before.merge(after, on="Node_Cell_ID", how="inner", suffixes=("_before", "_after"))
    st.markdown("**Changed Site Parameters**")
    st.dataframe(merged, use_container_width=True)


def _render_optimization_kpi_summary(compare_df: pd.DataFrame) -> None:
    if compare_df.empty:
        st.info("No optimization KPI comparison rows available yet.")
        return
    rows = []
    for metric, before_col, after_col, delta_col in [
        ("RSRP", "pred_rsrp_before", "pred_rsrp_after", "delta_rsrp"),
        ("RSRQ", "pred_rsrq_before", "pred_rsrq_after", "delta_rsrq"),
        ("SINR", "pred_sinr_before", "pred_sinr_after", "delta_sinr"),
    ]:
        rows.append({
            "metric": metric,
            "before_mean": round(float(compare_df[before_col].mean()), 4),
            "after_mean": round(float(compare_df[after_col].mean()), 4),
            "delta_mean": round(float(compare_df[delta_col].mean()), 4),
            "delta_min": round(float(compare_df[delta_col].min()), 4),
            "delta_max": round(float(compare_df[delta_col].max()), 4),
        })
    st.markdown("**Affected KPI Summary**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    by_site = compare_df.groupby("site_id", dropna=False).agg(
        rows=("site_id", "size"),
        rsrp_delta_mean=("delta_rsrp", "mean"),
        rsrq_delta_mean=("delta_rsrq", "mean"),
        sinr_delta_mean=("delta_sinr", "mean"),
    ).reset_index()
    for col in ["rsrp_delta_mean", "rsrq_delta_mean", "sinr_delta_mean"]:
        by_site[col] = by_site[col].round(4)
    st.markdown("**Affected Site KPI Delta**")
    st.dataframe(by_site, use_container_width=True)


def _render_optimization_visuals(compare_df: pd.DataFrame) -> None:
    if compare_df.empty:
        return
    metric_choice = st.selectbox(
        "Optimization KPI View",
        options=[
            ("RSRP", "pred_rsrp_before", "pred_rsrp_after", "delta_rsrp"),
            ("RSRQ", "pred_rsrq_before", "pred_rsrq_after", "delta_rsrq"),
            ("SINR", "pred_sinr_before", "pred_sinr_after", "delta_sinr"),
        ],
        format_func=lambda item: item[0],
        index=0,
        key="opt_metric_choice",
    )
    _, before_col, after_col, delta_col = metric_choice
    row = st.columns(3)
    before_fig = px.scatter(compare_df, x="lon", y="lat", color=before_col, title=f"{metric_choice[0]} Before", color_continuous_scale="Viridis", opacity=0.7)
    after_fig = px.scatter(compare_df, x="lon", y="lat", color=after_col, title=f"{metric_choice[0]} After", color_continuous_scale="Viridis", opacity=0.7)
    delta_limit = max(abs(float(compare_df[delta_col].min())), abs(float(compare_df[delta_col].max())), 0.1)
    delta_fig = px.scatter(compare_df, x="lon", y="lat", color=delta_col, title=f"{metric_choice[0]} Delta", color_continuous_scale="RdBu", range_color=[-delta_limit, delta_limit], opacity=0.75)
    with row[0]:
        st.plotly_chart(before_fig, use_container_width=True)
    with row[1]:
        st.plotly_chart(after_fig, use_container_width=True)
    with row[2]:
        st.plotly_chart(delta_fig, use_container_width=True)
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=compare_df[before_col], name="Before", opacity=0.55, marker_color="#2563eb"))
    hist.add_trace(go.Histogram(x=compare_df[after_col], name="After", opacity=0.55, marker_color="#dc2626"))
    hist.update_layout(title=f"{metric_choice[0]} Distribution Before vs After", barmode="overlay", height=380)
    st.plotly_chart(hist, use_container_width=True)
    delta_hist = px.histogram(compare_df, x=delta_col, nbins=40, title=f"{metric_choice[0]} Delta Distribution")
    st.plotly_chart(delta_hist, use_container_width=True)


def _render_coverage_page(project_id: int) -> None:
    runs = _list_coverage_runs(int(project_id))
    st.subheader("Coverage Test Runs")
    if not runs:
        st.info("No coverage test runs found yet. Use the sidebar coverage button to launch one.")
        return

    run_labels = [run.name for run in runs]
    selected_label = st.selectbox("Available Coverage Runs", options=run_labels, index=0)
    run_dir = next(run for run in runs if run.name == selected_label)
    summary = _load_summary(run_dir)
    bucket_summary_df = _safe_read_csv(run_dir / "bucket_summary.csv")
    bucket_grid_summary_df = _safe_read_csv(run_dir / "bucket_grid_summary.csv")
    coverage_df = _safe_read_csv(run_dir / "coverage_rows.csv")
    kpi_summary_df = _safe_read_csv(run_dir / "kpi_summary.csv")
    grid_kpi_timeseries_df = _safe_read_csv(run_dir / "grid_kpi_timeseries.csv")
    baseline_prediction_grid_df = _safe_read_csv(run_dir / "baseline_prediction_grid.csv")
    bucket_corrected_prediction_grid_df = _safe_read_csv(run_dir / "bucket_corrected_prediction_grid.csv")
    corrected_grid_surface_df = _safe_read_csv(run_dir / "corrected_grid_surface.csv")
    bucket_grid_geo_features_df = _safe_read_csv(run_dir / "bucket_grid_geo_features.csv")
    grid_cells_df = _safe_read_csv(run_dir / "coverage_grid_cells.csv")
    project_sites_df = _safe_read_csv(run_dir / "project_sites.csv")
    project_site_summary_df = _safe_read_csv(run_dir / "project_site_summary.csv")
    if coverage_df.empty:
        st.warning("Coverage run exists but `coverage_rows.csv` is empty.")
        return

    if "timestamp" in coverage_df.columns:
        coverage_df["timestamp"] = pd.to_datetime(coverage_df["timestamp"], errors="coerce")
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
        "latency",
        "jitter",
        "packet_loss",
        "mos",
        "speed",
        "csi_rsrp",
        "csi_rsrq",
        "csi_sinr",
    ]
    for col in numeric_cols:
        if col in coverage_df.columns:
            coverage_df[col] = pd.to_numeric(coverage_df[col], errors="coerce")

    polygon_gdf = _coverage_polygon_gdf(summary.get("polygon_wkt", DEFAULT_COVERAGE_POLYGON_WKT))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Buckets", len(summary.get("bucket_ranges", [])))
    c2.metric("Coverage Rows", int(summary.get("total_rows", len(coverage_df))))
    c3.metric("Runtime (sec)", summary.get("timings_sec", {}).get("total_runtime_sec"))
    c4.metric("Chunk Size", summary.get("chunk_size"))
    c5.metric(f"{int(summary.get('grid_size_m', 50))}m Grid Cells", int(summary.get("grid_cell_count", len(grid_cells_df))))
    if not project_site_summary_df.empty:
        st.markdown("**Project Sites By Operator**")
        st.dataframe(project_site_summary_df, use_container_width=True)

    if not bucket_summary_df.empty:
        st.markdown("**Bucket Row Counts**")
        st.dataframe(bucket_summary_df, use_container_width=True)

    if not bucket_grid_summary_df.empty:
        st.markdown("**Bucket Grid Mapping Summary**")
        st.dataframe(bucket_grid_summary_df, use_container_width=True)

    if not kpi_summary_df.empty:
        with st.expander("KPI Distribution Summary", expanded=False):
            st.dataframe(kpi_summary_df, use_container_width=True)

    bucket_names = [bucket["label"] for bucket in summary.get("bucket_ranges", []) if bucket.get("label")]
    available_buckets = [bucket for bucket in bucket_names if bucket in coverage_df["time_bucket"].astype(str).unique().tolist()]
    if not available_buckets:
        available_buckets = sorted(coverage_df["time_bucket"].dropna().astype(str).unique().tolist())
    tabs = st.tabs(available_buckets)

    for tab, bucket_name in zip(tabs, available_buckets):
        with tab:
            bucket_df = coverage_df.loc[coverage_df["time_bucket"].astype(str) == str(bucket_name)].copy()
            bucket_grid_df = grid_kpi_timeseries_df.loc[grid_kpi_timeseries_df["time_bucket"].astype(str) == str(bucket_name)].copy() if not grid_kpi_timeseries_df.empty else pd.DataFrame()
            bucket_geo_df = bucket_grid_geo_features_df.loc[bucket_grid_geo_features_df["time_bucket"].astype(str) == str(bucket_name)].copy() if not bucket_grid_geo_features_df.empty else pd.DataFrame()
            metric_options = _coverage_metric_options(bucket_df)
            if bucket_df.empty or not metric_options:
                st.info(f"No plottable KPI rows found for {bucket_name}.")
                continue

            bucket_ranges = next((item for item in summary.get("bucket_ranges", []) if item.get("label") == bucket_name), {})
            st.caption(f"{bucket_name}: {bucket_ranges.get('start')} to {bucket_ranges.get('end')}")

            control_cols = st.columns(2)
            metric_label = control_cols[0].selectbox(
                f"{bucket_name} KPI",
                options=metric_options,
                index=0,
                key=f"coverage_metric_{bucket_name}",
            )
            band_options = [label for label, _, _ in COVERAGE_METRIC_SPECS[metric_label]["bands"]]
            band_label = control_cols[1].selectbox(
                f"{bucket_name} Band Filter",
                options=band_options,
                index=0,
                key=f"coverage_band_{bucket_name}",
            )
            map_view_options = [
                "Full Baseline Surface",
                "DT Aggregated Grid KPI",
                "Bucket Corrected Surface",
            ]
            map_view_choice = st.selectbox(
                f"{bucket_name} Map View",
                options=map_view_options,
                index=0,
                key=f"coverage_map_view_{bucket_name}",
            )
            site_operator_options = ["All"]
            if not project_sites_df.empty and "site_operator" in project_sites_df.columns:
                site_operator_options += sorted(project_sites_df["site_operator"].dropna().astype(str).unique().tolist())
            site_operator_choice = st.selectbox(
                f"{bucket_name} Site Operator",
                options=site_operator_options,
                index=0,
                key=f"coverage_site_operator_{bucket_name}",
            )
            bucket_site_df = project_sites_df.copy()
            if not bucket_site_df.empty and site_operator_choice != "All" and "site_operator" in bucket_site_df.columns:
                bucket_site_df = bucket_site_df[bucket_site_df["site_operator"].astype(str) == site_operator_choice].copy()

            filtered_df = _filter_coverage_band(bucket_df, metric_label, band_label)
            metric_col = COVERAGE_METRIC_SPECS[metric_label]["column"]

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Bucket Rows", int(len(bucket_df)))
            m2.metric("Filtered Rows", int(len(filtered_df)))
            m3.metric("Mapped 25m Grids", int(bucket_grid_df["grid_id"].nunique()) if "grid_id" in bucket_grid_df.columns else 0)
            m4.metric(
                f"{metric_label} In Range",
                int(len(_apply_coverage_valid_range(bucket_df, metric_label))),
            )
            m5.metric("Project Sites", int(len(bucket_site_df)))

            band_counts_df = _coverage_band_counts(bucket_df, metric_label)
            chart_cols = st.columns([1.2, 1.0])
            with chart_cols[0]:
                if not band_counts_df.empty:
                    fig = px.bar(
                        band_counts_df,
                        x="band",
                        y="rows",
                        title=f"{bucket_name} {metric_label} Band Coverage",
                    )
                    fig.update_layout(xaxis_title="", yaxis_title="Rows", height=360)
                    st.plotly_chart(fig, use_container_width=True)
            with chart_cols[1]:
                stat_rows = []
                series = pd.to_numeric(filtered_df[metric_col], errors="coerce").dropna()
                if not series.empty:
                    stat_rows.append(
                        {
                            "metric": metric_label,
                            "band": band_label,
                            "min": round(float(series.min()), 4),
                            "median": round(float(series.median()), 4),
                            "mean": round(float(series.mean()), 4),
                            "max": round(float(series.max()), 4),
                        }
                    )
                if stat_rows:
                    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)

            st.markdown("**25m Grid View**")
            grid_metric_map = {
                "RSRP": "rsrp_mean",
                "RSRQ": "rsrq_mean",
                "SINR": "sinr_mean",
                "RSSI": "rssi_mean",
                "DL Throughput": "dl_tpt_mean",
                "UL Throughput": "ul_tpt_mean",
                "CQI": "cqi_mean",
            }
            baseline_metric_map = {
                "RSRP": "pred_rsrp",
                "RSRQ": "pred_rsrq",
                "SINR": "pred_sinr",
            }
            corrected_metric_map = {
                "RSRP": "pred_rsrp",
                "RSRQ": "pred_rsrq",
                "SINR": "pred_sinr",
            }
            grid_metric_col = grid_metric_map.get(metric_label)
            baseline_metric_col = baseline_metric_map.get(metric_label)
            corrected_metric_col = corrected_metric_map.get(metric_label)
            bucket_baseline_view_df = (
                baseline_prediction_grid_df.loc[
                    baseline_prediction_grid_df["time_bucket"].astype(str) == str(bucket_name)
                ].copy()
                if not baseline_prediction_grid_df.empty and "time_bucket" in baseline_prediction_grid_df.columns
                else baseline_prediction_grid_df.copy()
            )
            bucket_corrected_view_df = (
                bucket_corrected_prediction_grid_df.loc[
                    bucket_corrected_prediction_grid_df["time_bucket"].astype(str) == str(bucket_name)
                ].copy()
                if not bucket_corrected_prediction_grid_df.empty and "time_bucket" in bucket_corrected_prediction_grid_df.columns
                else pd.DataFrame()
            )
            if (
                map_view_choice == "Full Baseline Surface"
                and baseline_metric_col
                and not bucket_baseline_view_df.empty
                and not grid_cells_df.empty
            ):
                fig = _build_grid_metric_map(
                    polygon_gdf=polygon_gdf,
                    grid_cells_df=grid_cells_df,
                    grid_metric_df=bucket_baseline_view_df,
                    metric_col=baseline_metric_col,
                    title=f"{bucket_name} {metric_label} Part-Wise Baseline Surface",
                    site_df=bucket_site_df,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            elif (
                map_view_choice == "Bucket Corrected Surface"
                and corrected_metric_col
                and not bucket_corrected_view_df.empty
                and not grid_cells_df.empty
            ):
                fig = _build_grid_metric_map(
                    polygon_gdf=polygon_gdf,
                    grid_cells_df=grid_cells_df,
                    grid_metric_df=bucket_corrected_view_df,
                    metric_col=corrected_metric_col,
                    title=f"{bucket_name} {metric_label} Bucket Corrected Surface",
                    site_df=bucket_site_df,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            elif grid_metric_col and not bucket_grid_df.empty and not grid_cells_df.empty:
                fig = _build_grid_metric_map(
                    polygon_gdf=polygon_gdf,
                    grid_cells_df=grid_cells_df,
                    grid_metric_df=bucket_grid_df,
                    metric_col=grid_metric_col,
                    title=f"{bucket_name} {metric_label} DT Aggregated Grid KPI",
                    site_df=bucket_site_df,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                fig = _build_coverage_map(
                    polygon_gdf=polygon_gdf,
                    point_df=filtered_df,
                    metric_label=metric_label,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.markdown("**DT Points Used For This Part**")
            dt_points_fig = _build_dt_points_map(
                polygon_gdf=polygon_gdf,
                point_df=bucket_df,
                title=f"{bucket_name} DT Points Used",
            )
            st.pyplot(dt_points_fig, use_container_width=True)
            plt.close(dt_points_fig)

            if not bucket_baseline_view_df.empty and not corrected_grid_surface_df.empty and metric_label in {"RSRP", "RSRQ", "SINR"}:
                st.markdown("**Baseline Vs Part-Wise DT Corrected Surface**")
                corrected_surface_bucket_df = corrected_grid_surface_df.loc[
                    corrected_grid_surface_df["time_bucket"].astype(str) == str(bucket_name)
                ].copy()
                corrected_surface_metric_map = {
                    "RSRP": "corrected_rsrp_mean",
                    "RSRQ": "corrected_rsrq_mean",
                    "SINR": "corrected_sinr_mean",
                }
                baseline_surface_metric_map = {
                    "RSRP": "pred_rsrp",
                    "RSRQ": "pred_rsrq",
                    "SINR": "pred_sinr",
                }
                corrected_surface_metric_col = corrected_surface_metric_map.get(metric_label)
                baseline_surface_metric_col = baseline_surface_metric_map.get(metric_label)
                if not corrected_surface_bucket_df.empty and corrected_surface_metric_col and baseline_surface_metric_col:
                    compare_cols = st.columns(4)
                    with compare_cols[0]:
                        dt_compare_fig = _build_dt_points_map(
                            polygon_gdf=polygon_gdf,
                            point_df=bucket_df,
                            title=f"{bucket_name} DT Points Used",
                        )
                        st.pyplot(dt_compare_fig, use_container_width=True)
                        plt.close(dt_compare_fig)
                    with compare_cols[1]:
                        before_fig = _build_grid_metric_map(
                            polygon_gdf=polygon_gdf,
                            grid_cells_df=grid_cells_df,
                            grid_metric_df=bucket_baseline_view_df,
                            metric_col=baseline_surface_metric_col,
                            title=f"{bucket_name} {metric_label} Part-Wise Baseline",
                            site_df=bucket_site_df,
                        )
                        st.pyplot(before_fig, use_container_width=True)
                        plt.close(before_fig)
                    with compare_cols[2]:
                        after_fig = _build_grid_metric_map(
                            polygon_gdf=polygon_gdf,
                            grid_cells_df=grid_cells_df,
                            grid_metric_df=corrected_surface_bucket_df,
                            metric_col=corrected_surface_metric_col,
                            title=f"{bucket_name} {metric_label} DT Corrected Surface",
                            site_df=bucket_site_df,
                        )
                        st.pyplot(after_fig, use_container_width=True)
                        plt.close(after_fig)
                    with compare_cols[3]:
                        delta_fig = _build_grid_delta_map(
                            polygon_gdf=polygon_gdf,
                            grid_cells_df=grid_cells_df,
                            start_df=bucket_baseline_view_df.rename(columns={baseline_surface_metric_col: "__metric_value"}),
                            end_df=corrected_surface_bucket_df.rename(columns={corrected_surface_metric_col: "__metric_value"}),
                            metric_col="__metric_value",
                            start_label="Baseline",
                            end_label=bucket_name,
                            title_metric_label=metric_label,
                        )
                        st.pyplot(delta_fig, use_container_width=True)
                        plt.close(delta_fig)

            if not bucket_grid_df.empty:
                st.markdown("**Grid KPI Timeseries Sample**")
                show_cols = [col for col in [
                    "grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon",
                    "sample_count", "rsrp_mean", "rsrq_mean", "sinr_mean", "cqi_mean",
                    "estimated_prb_mean", "dominant_pci", "dominant_earfcn", "bandwidth_mhz_est", "cqi_source"
                ] if col in bucket_grid_df.columns]
                st.dataframe(bucket_grid_df[show_cols].head(500), use_container_width=True)

            if not bucket_geo_df.empty:
                st.markdown("**Historical Geo Feature View**")
                snapshot_mode = None
                if "geo_snapshot_mode" in bucket_geo_df.columns:
                    snapshot_mode = str(bucket_geo_df["geo_snapshot_mode"].dropna().iloc[0]) if not bucket_geo_df["geo_snapshot_mode"].dropna().empty else None
                snapshot_ts = None
                if "geo_snapshot_ts_utc" in bucket_geo_df.columns:
                    snapshot_ts = str(bucket_geo_df["geo_snapshot_ts_utc"].dropna().iloc[0]) if not bucket_geo_df["geo_snapshot_ts_utc"].dropna().empty else None
                if snapshot_mode or snapshot_ts:
                    st.caption(
                        "Geo snapshot: "
                        + (snapshot_mode or "unknown")
                        + (f" | as_of={snapshot_ts}" if snapshot_ts else "")
                    )
                geo_metric_options = [
                    col for col in [
                        "building_count",
                        "building_area_ratio",
                        "road_density",
                        "mall_presence",
                        "metro_presence",
                        "office_density",
                        "residential_density",
                        "industrial_density",
                        "park_open_area",
                        "open_area_ratio",
                        "green_ratio",
                    ] if col in bucket_geo_df.columns
                ]
                geo_metric_col = st.selectbox(
                    f"{bucket_name} Geo Feature",
                    options=geo_metric_options if geo_metric_options else ["building_count"],
                    index=0,
                    key=f"coverage_geo_metric_{bucket_name}",
                )
                geo_fig = _build_grid_metric_map(
                    polygon_gdf=polygon_gdf,
                    grid_cells_df=grid_cells_df,
                    grid_metric_df=bucket_geo_df,
                    metric_col=geo_metric_col,
                    title=f"{bucket_name} {geo_metric_col} 25m Grid Geo Map",
                    site_df=bucket_site_df,
                )
                st.pyplot(geo_fig, use_container_width=True)
                plt.close(geo_fig)
                geo_show_cols = [col for col in [
                    "grid_id", "grid_row", "grid_col", "centroid_lat", "centroid_lon",
                    "building_count", "building_area_ratio", "road_density", "mall_presence",
                    "metro_presence", "office_density", "residential_density", "industrial_density",
                    "park_open_area", "open_area_ratio", "green_ratio", "water_ratio", "clutter_class",
                    "geo_snapshot_mode", "geo_snapshot_ts_utc"
                ] if col in bucket_geo_df.columns]
                st.dataframe(bucket_geo_df[geo_show_cols].head(500), use_container_width=True)

    if not grid_cells_df.empty and not baseline_prediction_grid_df.empty and not corrected_grid_surface_df.empty and len(available_buckets) >= 1:
        baseline_buckets = [
            bucket for bucket in available_buckets
            if "time_bucket" in baseline_prediction_grid_df.columns
            and bucket in baseline_prediction_grid_df["time_bucket"].astype(str).unique().tolist()
        ]
        corrected_buckets = [
            bucket for bucket in available_buckets
            if "time_bucket" in corrected_grid_surface_df.columns
            and bucket in corrected_grid_surface_df["time_bucket"].astype(str).unique().tolist()
        ]
        compare_buckets = [bucket for bucket in available_buckets if bucket in baseline_buckets and bucket in corrected_buckets]
        if compare_buckets:
            st.markdown("**Part-Wise Baseline And Corrected Comparison**")
            st.caption("First row: baseline for PART_1, PART_2, PART_3. Second row: corrected surface for PART_1, PART_2, PART_3. Use this to check whether the baseline itself differs across the parts before correction.")
            triptych_metric = st.selectbox(
                "Triptych KPI",
                options=["RSRP", "RSRQ", "SINR"],
                index=0,
                key="coverage_triptych_metric",
            )
            baseline_surface_metric_map = {
                "RSRP": "pred_rsrp",
                "RSRQ": "pred_rsrq",
                "SINR": "pred_sinr",
            }
            corrected_surface_metric_map = {
                "RSRP": "corrected_rsrp_mean",
                "RSRQ": "corrected_rsrq_mean",
                "SINR": "corrected_sinr_mean",
            }
            site_frames = {bucket: project_sites_df.copy() for bucket in compare_buckets}
            baseline_frames = {
                bucket: baseline_prediction_grid_df.loc[
                    baseline_prediction_grid_df["time_bucket"].astype(str) == str(bucket)
                ].copy()
                for bucket in compare_buckets
            }
            corrected_frames = {
                bucket: corrected_grid_surface_df.loc[
                    corrected_grid_surface_df["time_bucket"].astype(str) == str(bucket)
                ].copy()
                for bucket in compare_buckets
            }
            baseline_triptych = _build_grid_metric_triptych(
                polygon_gdf=polygon_gdf,
                grid_cells_df=grid_cells_df,
                bucket_frames=baseline_frames,
                metric_col=baseline_surface_metric_map[triptych_metric],
                title_prefix=f"{triptych_metric} Baseline",
                discrete_bands=KPI_VISUAL_BANDS.get(triptych_metric),
                bucket_site_frames=site_frames,
            )
            st.pyplot(baseline_triptych, use_container_width=True)
            plt.close(baseline_triptych)

            corrected_triptych = _build_grid_metric_triptych(
                polygon_gdf=polygon_gdf,
                grid_cells_df=grid_cells_df,
                bucket_frames=corrected_frames,
                metric_col=corrected_surface_metric_map[triptych_metric],
                title_prefix=f"{triptych_metric} Corrected",
                discrete_bands=KPI_VISUAL_BANDS.get(triptych_metric),
                bucket_site_frames=site_frames,
            )
            st.pyplot(corrected_triptych, use_container_width=True)
            plt.close(corrected_triptych)

    if not bucket_grid_geo_features_df.empty and not grid_cells_df.empty and len(available_buckets) >= 2:
        st.info("Clutter and static geo comparison moved to the `Coverage - Clutter` page.")


def _render_coverage_clutter_page(project_id: int) -> None:
    runs = _list_coverage_runs(int(project_id))
    st.subheader("Coverage - Clutter")
    if not runs:
        st.info("No coverage test runs found yet.")
        return

    run_labels = [run.name for run in runs]
    selected_label = st.selectbox("Available Coverage Runs", options=run_labels, index=0, key="coverage_clutter_run")
    run_dir = next(run for run in runs if run.name == selected_label)
    summary = _load_summary(run_dir)
    bucket_grid_geo_features_df = _safe_read_csv(run_dir / "bucket_grid_geo_features.csv")
    grid_kpi_timeseries_df = _safe_read_csv(run_dir / "grid_kpi_timeseries.csv")
    grid_cells_df = _safe_read_csv(run_dir / "coverage_grid_cells.csv")
    if bucket_grid_geo_features_df.empty or grid_cells_df.empty:
        st.warning("This run does not contain clutter geo outputs.")
        return

    polygon_gdf = _coverage_polygon_gdf(summary.get("polygon_wkt", DEFAULT_COVERAGE_POLYGON_WKT))
    bucket_names = [bucket["label"] for bucket in summary.get("bucket_ranges", []) if bucket.get("label")]
    available_buckets = [bucket for bucket in bucket_names if bucket in bucket_grid_geo_features_df["time_bucket"].astype(str).unique().tolist()]
    if not available_buckets:
        available_buckets = sorted(bucket_grid_geo_features_df["time_bucket"].dropna().astype(str).unique().tolist())
    compare_frames = {
        bucket: bucket_grid_geo_features_df.loc[
            bucket_grid_geo_features_df["time_bucket"].astype(str) == str(bucket)
        ].copy()
        for bucket in available_buckets
    }

    st.caption("This page is only for clutter and geo analysis across PART_1, PART_2, PART_3.")

    static_geo_options = [
        col for col in [
            "building_count",
            "building_area_ratio",
            "building_area_sum_m2",
            "avg_building_area_m2",
            "road_density",
            "park_open_area",
            "open_area_ratio",
            "green_ratio",
            "water_ratio",
            "mall_presence",
            "metro_presence",
        ]
        if col in bucket_grid_geo_features_df.columns
    ]
    static_feature = st.selectbox(
        "Geo Feature Triptych",
        options=static_geo_options,
        index=0 if static_geo_options else None,
        key="coverage_clutter_static_feature",
    )
    if static_feature:
        feature_triptych = _build_grid_metric_triptych(
            polygon_gdf=polygon_gdf,
            grid_cells_df=grid_cells_df,
            bucket_frames=compare_frames,
            metric_col=static_feature,
            title_prefix=static_feature,
        )
        st.pyplot(feature_triptych, use_container_width=True)
        plt.close(feature_triptych)

    clutter_triptych = _build_clutter_triptych(
        polygon_gdf=polygon_gdf,
        grid_cells_df=grid_cells_df,
        bucket_frames=compare_frames,
    )
    st.pyplot(clutter_triptych, use_container_width=True)
    plt.close(clutter_triptych)

    if not grid_kpi_timeseries_df.empty:
        kpi_choice = st.selectbox(
            "Clutter KPI",
            options=["RSRP", "RSRQ", "SINR"],
            index=0,
            key="coverage_clutter_kpi",
        )
        kpi_col_map = {
            "RSRP": "rsrp_mean",
            "RSRQ": "rsrq_mean",
            "SINR": "sinr_mean",
        }
        kpi_col = kpi_col_map[kpi_choice]
        clutter_kpi_frames: List[pd.DataFrame] = []
        for bucket in available_buckets:
            geo_df = compare_frames[bucket][["grid_id", "clutter_class"]].copy()
            kpi_df = grid_kpi_timeseries_df.loc[
                grid_kpi_timeseries_df["time_bucket"].astype(str) == str(bucket)
            ].copy()
            if kpi_col not in kpi_df.columns:
                continue
            merged = geo_df.merge(kpi_df[["grid_id", kpi_col]], on="grid_id", how="inner")
            merged[kpi_col] = pd.to_numeric(merged[kpi_col], errors="coerce")
            merged = merged.dropna(subset=[kpi_col, "clutter_class"]).copy()
            if merged.empty:
                continue
            summary_df = merged.groupby("clutter_class", dropna=False)[kpi_col].agg(["count", "mean", "median"]).reset_index()
            summary_df["time_bucket"] = bucket
            clutter_kpi_frames.append(summary_df)
        if clutter_kpi_frames:
            clutter_kpi_df = pd.concat(clutter_kpi_frames, ignore_index=True)
            st.markdown("**KPI Average By Clutter Class**")
            chart = px.bar(
                clutter_kpi_df,
                x="clutter_class",
                y="mean",
                color="time_bucket",
                barmode="group",
                title=f"{kpi_choice} average by clutter class",
            )
            st.plotly_chart(chart, use_container_width=True)
            st.dataframe(clutter_kpi_df, use_container_width=True)

            if len(available_buckets) >= 2:
                start_bucket = available_buckets[0]
                end_bucket = available_buckets[-1]
                start_geo = compare_frames[start_bucket][["grid_id", "clutter_class"]].rename(columns={"clutter_class": "start_clutter"})
                end_geo = compare_frames[end_bucket][["grid_id", "clutter_class"]].rename(columns={"clutter_class": "end_clutter"})
                start_kpi = grid_kpi_timeseries_df.loc[
                    grid_kpi_timeseries_df["time_bucket"].astype(str) == str(start_bucket),
                    ["grid_id", kpi_col],
                ].rename(columns={kpi_col: "start_kpi"})
                end_kpi = grid_kpi_timeseries_df.loc[
                    grid_kpi_timeseries_df["time_bucket"].astype(str) == str(end_bucket),
                    ["grid_id", kpi_col],
                ].rename(columns={kpi_col: "end_kpi"})
                transition_df = start_geo.merge(end_geo, on="grid_id", how="outer")
                transition_df = transition_df.merge(start_kpi, on="grid_id", how="left").merge(end_kpi, on="grid_id", how="left")
                transition_df["transition"] = np.where(
                    transition_df["start_clutter"].astype(str) == transition_df["end_clutter"].astype(str),
                    "Unchanged",
                    transition_df["start_clutter"].astype(str) + " -> " + transition_df["end_clutter"].astype(str),
                )
                transition_df["delta_kpi"] = pd.to_numeric(transition_df["end_kpi"], errors="coerce") - pd.to_numeric(transition_df["start_kpi"], errors="coerce")
                transition_summary = transition_df.groupby("transition", dropna=False).agg(
                    grids=("grid_id", "count"),
                    start_kpi_mean=("start_kpi", "mean"),
                    end_kpi_mean=("end_kpi", "mean"),
                    delta_kpi_mean=("delta_kpi", "mean"),
                ).reset_index().sort_values(["grids", "transition"], ascending=[False, True])
                st.markdown(f"**Clutter Transition Impact: {start_bucket} -> {end_bucket}**")
                st.dataframe(transition_summary, use_container_width=True)
                transition_options = [
                    item for item in transition_summary["transition"].dropna().astype(str).tolist()
                    if item != "Unchanged"
                ]
                if transition_options:
                    selected_transition = st.selectbox(
                        "Transition KPI Map",
                        options=["All Changed"] + transition_options,
                        index=0,
                        key="coverage_clutter_transition_kpi_map",
                    )
                    transition_map_cols = st.columns(2)
                    with transition_map_cols[0]:
                        transition_map = _build_clutter_transition_map(
                            polygon_gdf=polygon_gdf,
                            grid_cells_df=grid_cells_df,
                            start_df=compare_frames[start_bucket],
                            end_df=compare_frames[end_bucket],
                            start_label=start_bucket,
                            end_label=end_bucket,
                        )
                        st.pyplot(transition_map, use_container_width=True)
                        plt.close(transition_map)
                    with transition_map_cols[1]:
                        delta_map = _build_transition_kpi_delta_map(
                            polygon_gdf=polygon_gdf,
                            grid_cells_df=grid_cells_df,
                            transition_df=transition_df,
                            transition_label=selected_transition,
                            kpi_label=kpi_choice,
                        )
                        st.pyplot(delta_map, use_container_width=True)
                        plt.close(delta_map)


def _render_optimization_page(project_id: int) -> None:
    opt_runs = _list_optimization_runs(int(project_id))
    st.subheader("Optimization Test Runs")
    if not opt_runs:
        st.info("No optimization test runs found yet. Use the sidebar optimization button to launch one.")
        return
    opt_labels = [run.name for run in opt_runs]
    selected_opt_label = st.selectbox("Available Optimization Runs", options=opt_labels, index=0)
    opt_run_dir = next(run for run in opt_runs if run.name == selected_opt_label)
    opt_summary = _load_summary(opt_run_dir)
    latency_df = pd.read_csv(opt_run_dir / "latency_log.csv") if (opt_run_dir / "latency_log.csv").exists() else pd.DataFrame()
    changed_df = pd.read_csv(opt_run_dir / "site_changed_rows.csv") if (opt_run_dir / "site_changed_rows.csv").exists() else pd.DataFrame()
    site_before_df = pd.read_csv(opt_run_dir / "site_before.csv") if (opt_run_dir / "site_before.csv").exists() else pd.DataFrame()
    opt_site_df = pd.read_csv(opt_run_dir / "site_after.csv") if (opt_run_dir / "site_after.csv").exists() else pd.DataFrame()
    prepared = _prepare_optimization_compare(opt_run_dir)
    compare_df = prepared["compare_df"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimization Runtime (sec)", opt_summary.get("total_runtime_sec"))
    c2.metric("Affected Sites", opt_summary.get("counts", {}).get("affected_sites"))
    c3.metric("Affected Cells", opt_summary.get("counts", {}).get("affected_cells"))
    c4.metric("K1/K2 Cells", opt_summary.get("counts", {}).get("k1k2_cells"))
    st.json({
        "baseline_job_id": opt_summary.get("baseline_job_id"),
        "target_type": opt_summary.get("target_type"),
        "target_id": opt_summary.get("target_id"),
        "impact_radius_m": opt_summary.get("impact_radius_m"),
        "neighbor_site_count": opt_summary.get("neighbor_site_count"),
        "max_interference_sites": opt_summary.get("max_interference_sites"),
        "changes": opt_summary.get("changes", {}),
        "affected_sites": opt_summary.get("affected_sites", []),
        "affected_cells": opt_summary.get("affected_cells", []),
    })
    if not latency_df.empty:
        st.markdown("**Optimization Latency Log**")
        st.dataframe(latency_df, use_container_width=True)
    _render_optimization_site_changes(site_before_df, changed_df)
    _render_optimization_kpi_summary(compare_df)
    st.subheader("Optimization Visualizations")
    _render_optimization_visuals(compare_df)
    if not compare_df.empty:
        st.markdown("**Affected Point-Level KPI Changes**")
        st.dataframe(compare_df.head(500), use_container_width=True)
    if not opt_site_df.empty and not prepared["optimized_merged_df"].empty:
        polygon_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        baseline_runs = _list_baseline_runs(int(project_id))
        if baseline_runs:
            polygon_path = baseline_runs[0] / "project_polygon.geojson"
            if polygon_path.exists():
                polygon_gdf = gpd.read_file(polygon_path)
        st.subheader("Optimization Map")
        _render_map(
            _build_map(
                polygon_gdf,
                _prepare_site_selection_df(opt_site_df),
                pd.DataFrame(columns=["lat", "lon"]),
                prepared["optimized_merged_df"],
                buildings_gdf=None,
                grid_gdf=None,
                show_geo=False,
                kpi_col="pred_rsrp",
            ),
            "optimization_map",
        )


def _render_single_optimization_panel(run_dir: Path, title: str, metric_key: str) -> None:
    summary = _load_summary(run_dir)
    changed_df = _safe_read_csv(run_dir / "site_changed_rows.csv")
    site_before_df = _safe_read_csv(run_dir / "site_before.csv")
    prepared = _prepare_optimization_compare(run_dir)
    compare_df = prepared["compare_df"]

    st.markdown(f"**{title}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Runtime (sec)", summary.get("total_runtime_sec"))
    c2.metric("Affected Cells", summary.get("counts", {}).get("affected_cells"))
    c3.metric("Optimized Rows", summary.get("counts", {}).get("optimized_rows"))
    st.json({
        "run_type": summary.get("run_type"),
        "baseline_job_id": summary.get("baseline_job_id"),
        "recommendation_scenario_id": summary.get("recommendation_scenario_id"),
        "affected_cells": summary.get("affected_cells", []),
        "parameter_change_counts": summary.get("parameter_change_counts", {}),
    })
    _render_optimization_site_changes(site_before_df, changed_df)
    _render_optimization_kpi_summary(compare_df)
    if not compare_df.empty:
        metric_choice = st.selectbox(
            f"{title} KPI View",
            options=[
                ("RSRP", "pred_rsrp_before", "pred_rsrp_after", "delta_rsrp"),
                ("RSRQ", "pred_rsrq_before", "pred_rsrq_after", "delta_rsrq"),
                ("SINR", "pred_sinr_before", "pred_sinr_after", "delta_sinr"),
            ],
            format_func=lambda item: item[0],
            index=0,
            key=f"{metric_key}_metric_choice",
        )
        _, before_col, after_col, delta_col = metric_choice
        delta_limit = max(abs(float(compare_df[delta_col].min())), abs(float(compare_df[delta_col].max())), 0.1)
        fig = px.scatter(
            compare_df,
            x="lon",
            y="lat",
            color=delta_col,
            title=f"{metric_choice[0]} Delta",
            color_continuous_scale="RdBu",
            range_color=[-delta_limit, delta_limit],
            opacity=0.75,
        )
        st.plotly_chart(fig, use_container_width=True)
        dist = go.Figure()
        dist.add_trace(go.Histogram(x=compare_df[before_col], name="Before", opacity=0.55, marker_color="#2563eb"))
        dist.add_trace(go.Histogram(x=compare_df[after_col], name="After", opacity=0.55, marker_color="#dc2626"))
        dist.update_layout(title=f"{metric_choice[0]} Before vs After", barmode="overlay", height=330)
        st.plotly_chart(dist, use_container_width=True)


def _render_tilt_optimization_full_polygon_images(run_dir: Path) -> None:
    prepared = _prepare_optimization_compare(run_dir)
    baseline_df = prepared["baseline_df"].copy()
    optimized_merged_df = prepared["optimized_merged_df"].copy()
    if baseline_df.empty or optimized_merged_df.empty:
        st.info("Full-polygon before/after files are not available for this tilt optimization run.")
        return

    metric_choice = st.selectbox(
        "Full Polygon KPI Image",
        options=[
            ("RSRP", "pred_rsrp"),
            ("RSRQ", "pred_rsrq"),
            ("SINR", "pred_sinr"),
        ],
        format_func=lambda item: item[0],
        index=0,
        key="tilt_opt_full_polygon_metric",
    )
    metric_label, value_col = metric_choice
    needed_cols = ["Node_Cell_ID", "lat", "lon", value_col]
    if any(col not in baseline_df.columns for col in needed_cols) or any(col not in optimized_merged_df.columns for col in needed_cols):
        st.info(f"Missing columns required for {metric_label} full-polygon comparison.")
        return

    before = baseline_df[needed_cols].copy()
    after = optimized_merged_df[needed_cols].copy()
    for frame in (before, after):
        frame["Node_Cell_ID"] = frame["Node_Cell_ID"].astype(str)
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce").round(6)
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce").round(6)
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    before = before.dropna(subset=["lat", "lon", value_col])
    after = after.dropna(subset=["lat", "lon", value_col])
    compare_df = before.merge(
        after,
        on=["Node_Cell_ID", "lat", "lon"],
        how="inner",
        suffixes=("_before", "_after"),
    )
    if compare_df.empty:
        st.info("No overlapping baseline and tilt-optimized points found for full-polygon image comparison.")
        return

    before_col = f"{value_col}_before"
    after_col = f"{value_col}_after"
    delta_col = f"delta_{value_col}"
    compare_df[delta_col] = compare_df[after_col] - compare_df[before_col]
    kpi_min, kpi_max = KPI_LIMITS[metric_label]
    delta_limit = max(abs(float(compare_df[delta_col].min())), abs(float(compare_df[delta_col].max())), 0.1)

    def _rsrp_quality(value: float) -> str:
        if pd.isna(value):
            return "Unknown"
        value = float(value)
        if value >= -85.0:
            return "Good (-85 to -44)"
        if value >= -95.0:
            return "Fair (-95 to -85)"
        if value >= -105.0:
            return "Weak (-105 to -95)"
        return "Poor (< -105)"

    st.markdown("**Full Polygon Before / After / Delta**")
    image_cols = st.columns(3)
    panels = [
        (before_col, f"Before LTE Prediction {metric_label}", False),
        (after_col, f"After Tilt Optimization {metric_label}", False),
        (delta_col, f"Tilt Optimization Delta {metric_label}", True),
    ]
    for idx, (col, title, is_delta) in enumerate(panels):
        if metric_label == "RSRP" and not is_delta:
            plot_df = compare_df.copy()
            quality_col = f"{col}_quality"
            plot_df[quality_col] = plot_df[col].map(_rsrp_quality)
            fig = px.scatter(
                plot_df,
                x="lon",
                y="lat",
                color=quality_col,
                title=title,
                category_orders={
                    quality_col: [
                        "Good (-85 to -44)",
                        "Fair (-95 to -85)",
                        "Weak (-105 to -95)",
                        "Poor (< -105)",
                        "Unknown",
                    ]
                },
                color_discrete_map={
                    "Good (-85 to -44)": "#16a34a",
                    "Fair (-95 to -85)": "#facc15",
                    "Weak (-105 to -95)": "#4a2c16",
                    "Poor (< -105)": "#dc2626",
                    "Unknown": "#9ca3af",
                },
                opacity=0.72,
                render_mode="webgl",
            )
        else:
            color_scale = "RdBu" if is_delta else "Viridis"
            color_range = [-delta_limit, delta_limit] if is_delta else [kpi_min, kpi_max]
            fig = px.scatter(
                compare_df,
                x="lon",
                y="lat",
                color=col,
                title=title,
                color_continuous_scale=color_scale,
                range_color=color_range,
                opacity=0.72,
                render_mode="webgl",
            )
        fig.update_traces(marker={"size": 4})
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10))
        with image_cols[idx]:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Full Polygon KPI Delta Summary**")
    st.dataframe(
        pd.DataFrame([
            {
                "metric": metric_label,
                "before_mean": round(float(compare_df[before_col].mean()), 4),
                "after_mean": round(float(compare_df[after_col].mean()), 4),
                "delta_mean": round(float(compare_df[delta_col].mean()), 4),
                "delta_min": round(float(compare_df[delta_col].min()), 4),
                "delta_max": round(float(compare_df[delta_col].max()), 4),
                "points": int(len(compare_df)),
            }
        ]),
        use_container_width=True,
    )


def _render_tilt_optimization_page(project_id: int) -> None:
    opt_runs = _list_optimization_runs(int(project_id))
    tilt_opt_runs = _list_tilt_optimization_runs(int(project_id))
    st.subheader("Tilt Recommendation Optimisation")
    if not tilt_opt_runs:
        st.info("No tilt recommendation optimization runs found yet. Use the sidebar form to launch one.")
        return

    selection_cols = st.columns(2)
    opt_label = None
    if opt_runs:
        opt_label = selection_cols[0].selectbox(
            "Normal Optimization Run",
            options=[run.name for run in opt_runs],
            index=0,
            key="tilt_opt_compare_normal_run",
        )
    else:
        selection_cols[0].info("No normal optimization runs available.")
    tilt_opt_label = selection_cols[1].selectbox(
        "Tilt Optimization Run",
        options=[run.name for run in tilt_opt_runs],
        index=0,
        key="tilt_opt_compare_tilt_run",
    )

    cols = st.columns(2)
    if opt_label:
        opt_run_dir = next(run for run in opt_runs if run.name == opt_label)
        with cols[0]:
            _render_single_optimization_panel(opt_run_dir, "Normal LTE Optimization", "normal_opt_compare")
    with cols[1]:
        tilt_opt_run_dir = next(run for run in tilt_opt_runs if run.name == tilt_opt_label)
        _render_single_optimization_panel(tilt_opt_run_dir, "Tilt Recommendation Optimization", "tilt_opt_compare")
    st.subheader("Tilt Optimization Full Polygon Images")
    _render_tilt_optimization_full_polygon_images(tilt_opt_run_dir)


def _prepare_tilt_recommendation_compare(run_dir: Path) -> Dict[str, pd.DataFrame]:
    bad_summary = _safe_read_csv(run_dir / "bad_summary.csv")
    forecast_df = _safe_read_csv(run_dir / "forecast.csv")
    reco_df = _safe_read_csv(run_dir / "recommendations.csv")
    bad_geo_df = _safe_read_csv(run_dir / "bad_samples_with_geo.csv")
    antenna_df = _safe_read_csv(run_dir / "antenna_input.csv")
    geo_summary_df = _safe_read_csv(run_dir / "bad_geo_cell_summary.csv")
    bearing_df = _safe_read_csv(run_dir / "dominant_bearing_summary.csv")
    candidate_df = _safe_read_csv(run_dir / "candidate_validation_results.csv")
    site_candidate_df = _safe_read_csv(run_dir / "site_candidate_evaluations.csv")

    changed_reco_df = pd.DataFrame()
    if not reco_df.empty:
        changed_reco_df = reco_df.copy()
        changed_reco_df["Current Value"] = pd.to_numeric(changed_reco_df["Current Value"], errors="coerce")
        changed_reco_df["Recommended Value"] = pd.to_numeric(changed_reco_df["Recommended Value"], errors="coerce")
        changed_reco_df["delta"] = changed_reco_df["Recommended Value"] - changed_reco_df["Current Value"]
        changed_reco_df = changed_reco_df[
            changed_reco_df["delta"].notna() & (changed_reco_df["delta"].abs() > 1e-9)
        ].copy()

    forecast_compare_df = pd.DataFrame()
    if not forecast_df.empty:
        forecast_compare_df = forecast_df.copy()
        forecast_compare_df["Pre-Change"] = pd.to_numeric(forecast_compare_df["Pre-Change"], errors="coerce")
        forecast_compare_df["Est. Post-Change"] = pd.to_numeric(forecast_compare_df["Est. Post-Change"], errors="coerce")
        forecast_compare_df["estimated_delta_bad_samples"] = (
            forecast_compare_df["Est. Post-Change"] - forecast_compare_df["Pre-Change"]
        )

    candidate_compare_df = _prepare_tilt_candidate_validation_df(candidate_df)
    site_candidate_compare_df = _prepare_tilt_candidate_validation_df(site_candidate_df)

    return {
        "bad_summary": bad_summary,
        "forecast_df": forecast_df,
        "recommendations_df": reco_df,
        "changed_recommendations_df": changed_reco_df,
        "bad_samples_with_geo_df": bad_geo_df,
        "antenna_df": antenna_df,
        "geo_summary_df": geo_summary_df,
        "bearing_df": bearing_df,
        "forecast_compare_df": forecast_compare_df,
        "candidate_validation_df": candidate_compare_df,
        "site_candidate_evaluations_df": site_candidate_compare_df,
    }


def _summarize_tilt_candidate_actions(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None"}:
        return "HOLD"
    try:
        payload = json.loads(text)
    except Exception:
        return text[:220]
    if not isinstance(payload, list) or not payload:
        return "HOLD"
    parts = []
    for item in payload[:6]:
        if not isinstance(item, dict):
            continue
        cell = str(item.get("cell_id", "")).strip()
        param = str(item.get("parameter", "")).strip()
        target = item.get("target_value", "")
        try:
            target_text = f"{float(target):.2f}".rstrip("0").rstrip(".")
        except Exception:
            target_text = str(target)
        parts.append(f"{cell} {param}->{target_text}".strip())
    extra = f" +{len(payload) - 6} more" if len(payload) > 6 else ""
    return "; ".join(parts) + extra


def _prepare_tilt_candidate_validation_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    numeric_cols = [
        "score",
        "baseline_bad_count",
        "candidate_bad_count",
        "recovered_bad_samples",
        "new_bad_samples",
        "net_bad_reduction",
        "mean_rsrp_delta",
        "mean_rsrq_delta",
        "mean_sinr_delta",
        "rsrp_recovered_bad",
        "rsrp_new_bad",
        "rsrq_recovered_bad",
        "rsrq_new_bad",
        "sinr_recovered_bad",
        "sinr_new_bad",
        "rsrp_severity_reduction_per_sample",
        "rsrq_severity_reduction_per_sample",
        "sinr_severity_reduction_per_sample",
        "total_severity_reduction_per_sample",
        "good_area_loss_pct",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "candidate_name" in out.columns:
        out["candidate_name"] = out["candidate_name"].astype(str)
        out["is_hold"] = out["candidate_name"].str.lower().eq("hold")
    else:
        out["candidate_name"] = ""
        out["is_hold"] = False
    if "target_value" in out.columns:
        out["Action Summary"] = out["target_value"].map(_summarize_tilt_candidate_actions)
    else:
        out["Action Summary"] = ""
    if "site_id" in out.columns:
        out["site_id"] = out["site_id"].astype(str)
    return out


def _render_tilt_recommendation_summary(summary: Dict, changed_reco_df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tilt Runtime (sec)", summary.get("total_runtime_sec"))
    c2.metric("Bad Samples", summary.get("counts", {}).get("bad_samples"))
    c3.metric("Bad Cells", summary.get("counts", {}).get("bad_cells"))
    c4.metric("Swap Sector Yes", summary.get("counts", {}).get("swap_sector_yes"))
    st.json(
        {
            "project_id": summary.get("project_id"),
            "region": summary.get("region"),
            "operator": summary.get("operator"),
            "thresholds": summary.get("thresholds", {}),
        }
    )
    if not changed_reco_df.empty:
        st.markdown("**Changed Recommendation Counts**")
        counts = (
            changed_reco_df.groupby("Parameter", dropna=False)
            .size()
            .reset_index(name="changed_rows")
            .sort_values("changed_rows", ascending=False)
        )
        st.dataframe(counts, use_container_width=True)
    if not forecast_df.empty:
        total_pre = pd.to_numeric(forecast_df["Pre-Change"], errors="coerce").sum()
        total_post = pd.to_numeric(forecast_df["Est. Post-Change"], errors="coerce").sum()
        total_delta = total_post - total_pre
        st.metric("Estimated Total Bad-Sample Delta", round(float(total_delta), 2))


def _render_tilt_recommendation_tables(changed_reco_df: pd.DataFrame, geo_summary_df: pd.DataFrame, bearing_df: pd.DataFrame) -> None:
    if not changed_reco_df.empty:
        st.markdown("**Recommended Parameter Changes**")
        st.dataframe(
            changed_reco_df[
                ["Cell ID", "Technology", "Parameter", "Current Value", "Recommended Value", "delta", "Swap Sector Detected", "Reason"]
            ],
            use_container_width=True,
        )
    if not geo_summary_df.empty:
        st.markdown("**Bad-Sample Geo Context by Cell**")
        st.dataframe(geo_summary_df, use_container_width=True)
    if not bearing_df.empty:
        st.markdown("**Dominant Bearing Summary**")
        st.dataframe(bearing_df, use_container_width=True)


def _render_tilt_recommendation_visuals(forecast_compare_df: pd.DataFrame, changed_reco_df: pd.DataFrame) -> None:
    if not forecast_compare_df.empty:
        st.subheader("Before / After Forecast")
        kpi_choice = st.selectbox(
            "Forecast KPI",
            options=sorted(forecast_compare_df["KPI"].dropna().astype(str).unique().tolist()),
            key="tilt_forecast_kpi",
        )
        work = forecast_compare_df[forecast_compare_df["KPI"].astype(str) == str(kpi_choice)].copy()
        if not work.empty:
            work = work.sort_values("Pre-Change", ascending=False)
            bar_fig = go.Figure()
            bar_fig.add_trace(
                go.Bar(
                    x=work["Cell ID"],
                    y=work["Pre-Change"],
                    name="Before",
                    marker_color="#2563eb",
                )
            )
            bar_fig.add_trace(
                go.Bar(
                    x=work["Cell ID"],
                    y=work["Est. Post-Change"],
                    name="Estimated After",
                    marker_color="#dc2626",
                )
            )
            bar_fig.update_layout(
                title=f"{kpi_choice} Bad Samples Before vs Estimated After",
                barmode="group",
                xaxis_title="Cell ID",
                yaxis_title="Bad Sample Count",
                height=420,
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            delta_fig = px.bar(
                work,
                x="Cell ID",
                y="estimated_delta_bad_samples",
                color="estimated_delta_bad_samples",
                color_continuous_scale="RdBu",
                title=f"{kpi_choice} Estimated Change in Bad Sample Count",
            )
            st.plotly_chart(delta_fig, use_container_width=True)

    if not changed_reco_df.empty:
        st.subheader("Recommendation Mix")
        mix = changed_reco_df.copy()
        mix["label"] = mix["Parameter"].astype(str) + " " + mix["delta"].round(2).astype(str)
        pie_fig = px.pie(
            mix,
            names="label",
            title="Changed Recommendations Distribution",
        )
        st.plotly_chart(pie_fig, use_container_width=True)


def _render_tilt_candidate_validation_visuals(candidate_df: pd.DataFrame, antenna_df: pd.DataFrame) -> None:
    st.subheader("Candidate Before / After Validation")
    if candidate_df.empty:
        st.info("No candidate validation file found yet. Run with --validate-candidates to populate this section.")
        return

    work = candidate_df.copy()
    if "site_id" not in work.columns:
        st.info("Candidate validation file does not contain site_id.")
        return

    non_hold = work[~work["is_hold"].astype(bool)].copy() if "is_hold" in work.columns else work.copy()
    rank_source = non_hold if not non_hold.empty else work.copy()
    rank_source = rank_source.sort_values(
        ["site_id", "score", "net_bad_reduction", "mean_rsrp_delta"],
        ascending=[True, False, False, False],
    )
    best_by_site = rank_source.groupby("site_id", as_index=False, dropna=False).head(1).copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validated Candidates", int(len(work)))
    c2.metric("Non-HOLD Candidates", int(len(non_hold)))
    c3.metric("Best Score", round(float(pd.to_numeric(work.get("score"), errors="coerce").max()), 4) if "score" in work.columns else None)
    c4.metric(
        "Best Net Bad Reduction",
        round(float(pd.to_numeric(work.get("net_bad_reduction"), errors="coerce").max()), 2)
        if "net_bad_reduction" in work.columns
        else None,
    )

    display_cols = [
        "site_id",
        "candidate_name",
        "Action Summary",
        "score",
        "baseline_bad_count",
        "candidate_bad_count",
        "net_bad_reduction",
        "recovered_bad_samples",
        "new_bad_samples",
        "mean_rsrp_delta",
        "rsrp_recovered_bad",
        "rsrp_new_bad",
        "rsrp_severity_reduction_per_sample",
        "mean_rsrq_delta",
        "mean_sinr_delta",
        "constraints_passed",
    ]
    display_cols = [col for col in display_cols if col in best_by_site.columns]
    st.markdown("**Best Candidate Per Site, Even If Negative**")
    st.dataframe(best_by_site[display_cols], use_container_width=True, hide_index=True)

    sites = sorted(work["site_id"].dropna().astype(str).unique().tolist())
    selected_site = st.selectbox("Candidate Site", options=sites, key="tilt_candidate_site") if sites else None
    selected = work[work["site_id"].astype(str) == str(selected_site)].copy() if selected_site else work.copy()
    selected = selected.sort_values("score", ascending=False).head(30)

    if selected.empty:
        st.info("No candidates available for selected site.")
        return

    st.markdown("**Selected Site Candidate Table**")
    table_cols = [
        "candidate_name",
        "Action Summary",
        "score",
        "baseline_bad_count",
        "candidate_bad_count",
        "net_bad_reduction",
        "recovered_bad_samples",
        "new_bad_samples",
        "mean_rsrp_delta",
        "rsrp_recovered_bad",
        "rsrp_new_bad",
        "rsrp_severity_reduction_per_sample",
        "mean_rsrq_delta",
        "mean_sinr_delta",
    ]
    table_cols = [col for col in table_cols if col in selected.columns]
    st.dataframe(selected[table_cols], use_container_width=True, hide_index=True)

    if {"candidate_name", "baseline_bad_count", "candidate_bad_count"}.issubset(selected.columns):
        before_after = selected.melt(
            id_vars=["candidate_name"],
            value_vars=["baseline_bad_count", "candidate_bad_count"],
            var_name="Stage",
            value_name="Bad Samples",
        )
        before_after["Stage"] = before_after["Stage"].map(
            {"baseline_bad_count": "Before Baseline", "candidate_bad_count": "After Candidate"}
        )
        fig = px.bar(
            before_after,
            x="candidate_name",
            y="Bad Samples",
            color="Stage",
            barmode="group",
            title=f"Site {selected_site}: Baseline vs Candidate Bad Samples",
        )
        fig.update_layout(xaxis_tickangle=-35, height=430)
        st.plotly_chart(fig, use_container_width=True)

    delta_cols = [
        col
        for col in [
            "net_bad_reduction",
            "recovered_bad_samples",
            "new_bad_samples",
            "rsrp_recovered_bad",
            "rsrp_new_bad",
            "mean_rsrp_delta",
            "rsrp_severity_reduction_per_sample",
        ]
        if col in selected.columns
    ]
    if delta_cols:
        metric = st.selectbox(
            "Candidate Delta Metric",
            options=delta_cols,
            index=delta_cols.index("mean_rsrp_delta") if "mean_rsrp_delta" in delta_cols else 0,
            key="tilt_candidate_delta_metric",
        )
        fig = px.bar(
            selected,
            x="candidate_name",
            y=metric,
            color=metric,
            color_continuous_scale="RdYlGn",
            title=f"Site {selected_site}: {metric} by Candidate",
        )
        fig.update_layout(xaxis_tickangle=-35, height=430)
        st.plotly_chart(fig, use_container_width=True)

    scatter_cols = {"score", "mean_rsrp_delta", "net_bad_reduction"}
    if scatter_cols.issubset(work.columns):
        fig = px.scatter(
            work,
            x="mean_rsrp_delta",
            y="score",
            color="net_bad_reduction",
            hover_name="candidate_name",
            hover_data=[col for col in ["site_id", "Action Summary", "recovered_bad_samples", "new_bad_samples"] if col in work.columns],
            color_continuous_scale="RdYlGn",
            title="All Candidates: RSRP Delta vs Score",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#111827")
        fig.add_vline(x=0, line_dash="dash", line_color="#111827")
        st.plotly_chart(fig, use_container_width=True)

    _render_tilt_candidate_site_map(best_by_site, antenna_df)


def _render_tilt_candidate_site_map(best_by_site: pd.DataFrame, antenna_df: pd.DataFrame) -> None:
    if best_by_site.empty or antenna_df.empty:
        return
    site_df = _prepare_site_selection_df(antenna_df.copy())
    if "dashboard_nodeb_id" not in site_df.columns:
        return
    site_points = site_df.drop_duplicates(subset=["dashboard_nodeb_id"]).copy()
    plot_df = site_points.merge(
        best_by_site,
        left_on="dashboard_nodeb_id",
        right_on="site_id",
        how="inner",
        suffixes=("", "_candidate"),
    )
    if plot_df.empty:
        return

    center = [
        float(pd.to_numeric(plot_df["lat"], errors="coerce").median()),
        float(pd.to_numeric(plot_df["lon"], errors="coerce").median()),
    ]
    fmap = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron", control_scale=True)
    for _, row in plot_df.iterrows():
        score = float(row.get("score", 0.0) or 0.0)
        net_bad = float(row.get("net_bad_reduction", 0.0) or 0.0)
        mean_rsrp = float(row.get("mean_rsrp_delta", 0.0) or 0.0)
        color = "#16a34a" if net_bad > 0 or score > 0 else "#dc2626" if net_bad < 0 or score < 0 else "#64748b"
        tooltip = (
            f"Site {row.get('site_id')}<br>"
            f"Best: {row.get('candidate_name')}<br>"
            f"Score={score:.4f}<br>"
            f"Net bad reduction={net_bad:.0f}<br>"
            f"Mean RSRP delta={mean_rsrp:.3f}<br>"
            f"{row.get('Action Summary', '')}"
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=7,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=tooltip,
        ).add_to(fmap)
    st.markdown("**Best Candidate Site Map: Green Improves, Red Degrades**")
    _render_map(fmap, "tilt_candidate_site_map")


def _render_tilt_bad_sample_map(
    antenna_df: pd.DataFrame,
    bad_geo_df: pd.DataFrame,
    changed_reco_df: pd.DataFrame,
) -> None:
    if antenna_df.empty:
        st.info("No antenna data available for tilt visualization.")
        return

    site_df = antenna_df.copy()
    site_df = _prepare_site_selection_df(site_df)
    polygon_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if changed_reco_df.empty and bad_geo_df.empty:
        st.info("No tilt visualization data available yet.")
        return

    metric_choice = st.selectbox(
        "Bad Sample KPI Layer",
        options=[
            ("RSRP", "Bad RSRP"),
            ("RSRQ", "Bad RSRQ"),
            ("SINR", "Bad SINR"),
        ],
        format_func=lambda item: item[0],
        key="tilt_bad_sample_metric",
    )[1]

    bad_map_df = bad_geo_df.copy()
    if not bad_map_df.empty:
        lat_col = next((c for c in ["lat", "latitude"] if c in bad_map_df.columns), None)
        lon_col = next((c for c in ["lon", "longitude"] if c in bad_map_df.columns), None)
        if lat_col and lon_col:
            bad_map_df = bad_map_df[
                bad_map_df[metric_choice].astype(bool)
            ].copy()
            metric_name = metric_choice.split()[-1]
            value_col = next(
                (
                    c for c in ["RSRP_eval", "RSRQ_eval", "SINR_eval"]
                    if c in bad_map_df.columns and metric_name in c
                ),
                None,
            )
            plot_kpi_col = {
                "RSRP": "pred_rsrp",
                "RSRQ": "pred_rsrq",
                "SINR": "pred_sinr",
            }.get(metric_name, "pred_rsrp")
            bad_map_df = pd.DataFrame(
                {
                    "lat": pd.to_numeric(bad_map_df[lat_col], errors="coerce"),
                    "lon": pd.to_numeric(bad_map_df[lon_col], errors="coerce"),
                    plot_kpi_col: (
                        pd.to_numeric(bad_map_df[value_col], errors="coerce")
                        if value_col else pd.Series(-100.0, index=bad_map_df.index)
                    ),
                    "Node_Cell_ID": bad_map_df["Cell ID"].astype(str) if "Cell ID" in bad_map_df.columns else "",
                }
            ).dropna(subset=["lat", "lon", plot_kpi_col])
        else:
            plot_kpi_col = "pred_rsrp"
            bad_map_df = pd.DataFrame(columns=["lat", "lon", plot_kpi_col])
    else:
        plot_kpi_col = "pred_rsrp"

    st.markdown("**Tilt Recommendation Map**")
    fmap = _build_map(
        polygon_gdf,
        site_df,
        pd.DataFrame(columns=["lat", "lon"]),
        bad_map_df if not bad_map_df.empty else pd.DataFrame(columns=["lat", "lon", plot_kpi_col]),
        buildings_gdf=None,
        grid_gdf=None,
        show_geo=False,
        kpi_col=plot_kpi_col,
        show_site_markers=True,
    )

    if not changed_reco_df.empty:
        site_reco = changed_reco_df.copy()
        site_reco["site_id"] = site_reco["Cell ID"].astype(str).str.split("_").str[0]
        pivot = (
            site_reco.pivot_table(
                index="site_id",
                columns="Parameter",
                values="delta",
                aggfunc="first",
            )
            .reset_index()
        )
        site_points = site_df.drop_duplicates(subset=["dashboard_nodeb_id"]).copy()
        site_points = site_points.merge(
            pivot,
            left_on="dashboard_nodeb_id",
            right_on="site_id",
            how="inner",
        )
        for _, row in site_points.iterrows():
            popup_lines = [f"Site {row.get('dashboard_nodeb_id')}"]
            for param in ["ETilt", "Azimuth", "TX Power"]:
                if param in row and pd.notna(row[param]) and abs(float(row[param])) > 1e-9:
                    popup_lines.append(f"{param} delta={float(row[param]):.2f}")
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lon"])],
                radius=6,
                color="#dc2626",
                weight=2,
                fill=True,
                fill_color="#f97316",
                fill_opacity=0.9,
                tooltip=" | ".join(popup_lines),
            ).add_to(fmap)

    _render_map(fmap, "tilt_map")


def _render_tilt_recommendation_page(project_id: int) -> None:
    tilt_runs = _list_tilt_runs(int(project_id))
    st.subheader("Tilt Recommendation Test Runs")
    if not tilt_runs:
        st.info("No tilt recommendation test runs found yet. Use the sidebar tilt button to launch one.")
        return
    tilt_labels = [run.name for run in tilt_runs]
    selected_label = st.selectbox("Available Tilt Runs", options=tilt_labels, index=0)
    run_dir = next(run for run in tilt_runs if run.name == selected_label)
    summary = _load_summary(run_dir)
    prepared = _prepare_tilt_recommendation_compare(run_dir)

    _render_tilt_recommendation_summary(
        summary,
        prepared["changed_recommendations_df"],
        prepared["forecast_df"],
    )
    _render_tilt_recommendation_tables(
        prepared["changed_recommendations_df"],
        prepared["geo_summary_df"],
        prepared["bearing_df"],
    )
    _render_tilt_recommendation_visuals(
        prepared["forecast_compare_df"],
        prepared["changed_recommendations_df"],
    )
    _render_tilt_candidate_validation_visuals(
        prepared["candidate_validation_df"],
        prepared["antenna_df"],
    )
    _render_tilt_bad_sample_map(
        prepared["antenna_df"],
        prepared["bad_samples_with_geo_df"],
        prepared["changed_recommendations_df"],
    )


def main() -> None:
    st.set_page_config(page_title="LTE RF Debug Dashboard", layout="wide")
    st.sidebar.header("Run Controls")
    dashboard_page = st.sidebar.radio(
        "Dashboard Page",
        options=["RF Debug", "Coverage", "Coverage - Clutter", "Optimization", "Tilt Recommendation", "Tilt Recommendation Optimisation"],
        index=0,
    )
    page_titles = {
        "RF Debug": "LTE RF Debug Dashboard",
        "Coverage": "Coverage Dashboard",
        "Coverage - Clutter": "Coverage - Clutter",
        "Optimization": "Optimization Dashboard",
        "Tilt Recommendation": "Tilt Recommendation Dashboard",
        "Tilt Recommendation Optimisation": "Tilt Recommendation Optimisation Dashboard",
    }
    page_captions = {
        "RF Debug": "Test-only RF lab. Reads from DB, uses DT for validation, and compares baseline RF against geo-adjusted output.",
        "Coverage": "Coverage baseline, DT aggregation, and part-wise corrected surfaces.",
        "Coverage - Clutter": "Cluster class, geo-feature, and KPI transition analysis across PART_1, PART_2, and PART_3.",
        "Optimization": "Optimization test runs and before/after KPI comparison.",
        "Tilt Recommendation": "Tilt recommendation outputs and impacted bad-sample view.",
        "Tilt Recommendation Optimisation": "Side-by-side comparison of manual LTE optimization and tilt recommendation driven optimization.",
    }
    st.title(page_titles.get(dashboard_page, "LTE RF Debug Dashboard"))
    st.caption(page_captions.get(dashboard_page, ""))
    with st.sidebar.form("rf_debug_run_form"):
        project_id = st.number_input("Project ID", value=DEFAULT_PROJECT_ID, step=1)
        session_input = st.text_input("Session IDs", value=",".join(map(str, DEFAULT_SESSION_IDS)))
        region = st.text_input("Region", value=DEFAULT_REGION)
        radius_m = st.number_input("Radius (m)", value=DEFAULT_RADIUS_M, step=50.0)
        grid_resolution_m = st.number_input("Grid Resolution (m)", value=DEFAULT_GRID_RESOLUTION_M, step=5.0)
        workers = st.number_input("Workers", value=DEFAULT_WORKERS, step=1, min_value=1)
        max_interference_sites = st.number_input(
            "Max Interference Sites",
            value=DEFAULT_MAX_INTERFERENCE_SITES,
            step=5,
            min_value=1,
        )
        tile_size_m = st.number_input("Tile Size (m)", value=DEFAULT_TILE_SIZE_M, step=25.0)
        cluster_count = st.number_input("Morphology Clusters", value=DEFAULT_CLUSTER_COUNT, step=1, min_value=2)
        validation_fraction = st.slider(
            "Validation Fraction",
            min_value=0.1,
            max_value=0.5,
            value=float(DEFAULT_VALIDATION_FRACTION),
            step=0.05,
        )
        enable_osm = st.checkbox("Enable OSM Enrichment", value=False)
        run_triggered = st.form_submit_button("Run RF Debug Lab", type="primary")
    if run_triggered:
        session_ids = tuple(int(part.strip()) for part in session_input.split(",") if part.strip())
        config = RunConfig(
            project_id=int(project_id),
            session_ids=session_ids,
            region=region,
            radius_m=float(radius_m),
            grid_resolution_m=float(grid_resolution_m),
            workers=int(workers),
            max_interference_sites=int(max_interference_sites),
            tile_size_m=float(tile_size_m),
            cluster_count=int(cluster_count),
            validation_fraction=float(validation_fraction),
            enable_osm=enable_osm,
            output_root=OUTPUT_ROOT,
        )
        with st.spinner("Running RF debug lab. This uses DB input only and does not save results back to DB."):
            run_dir = run_rf_debug_lab(config)
        st.success(f"Run completed: {run_dir}")

    st.sidebar.header("Coverage Test")
    with st.sidebar.form("coverage_test_form"):
        coverage_polygon_wkt = st.text_area("Coverage Polygon WKT", value=DEFAULT_COVERAGE_POLYGON_WKT, height=140)
        coverage_chunk_size = st.number_input("Coverage Chunk Size", value=10000, step=5000, min_value=1000)
        coverage_run_triggered = st.form_submit_button("Run Coverage Test", type="primary")
    if coverage_run_triggered:
        coverage_config = CoverageTestConfig(
            project_id=int(project_id),
            region=region,
            polygon_wkt=coverage_polygon_wkt.strip(),
            chunk_size=int(coverage_chunk_size),
            output_root=OUTPUT_ROOT,
        )
        with st.spinner("Running coverage test using tbl_network_log rows inside the polygon buckets."):
            coverage_run_dir = run_coverage_test(coverage_config)
        st.success(f"Coverage test completed: {coverage_run_dir}")

    st.sidebar.header("Optimization Test")
    with st.sidebar.form("rf_optimization_test_form"):
        opt_target_type = st.selectbox("Target Type", options=["site", "cell"], index=0)
        opt_target_id = st.text_input("Target ID", value="")
        opt_impact_radius_m = st.number_input("Impact Radius (m)", value=1200.0, step=100.0, min_value=100.0)
        opt_delta_lat = st.number_input("Delta Latitude", value=0.0, step=0.0001, format="%.6f")
        opt_delta_lon = st.number_input("Delta Longitude", value=0.0, step=0.0001, format="%.6f")
        opt_delta_azimuth = st.number_input("Delta Azimuth", value=0.0, step=1.0)
        opt_delta_etilt = st.number_input("Delta Electrical Tilt", value=0.0, step=0.5)
        opt_delta_mtilt = st.number_input("Delta Mechanical Tilt", value=0.0, step=0.5)
        opt_delta_tx = st.number_input("Delta TX Power", value=0.0, step=0.5)
        opt_delta_height = st.number_input("Delta Antenna Height", value=0.0, step=0.5)
        opt_neighbor_site_count = st.number_input("Neighbor Site Count", value=2, step=1, min_value=0)
        opt_max_interference_sites = st.number_input("Max Optimization Interference Sites", value=20, step=1, min_value=1)
        opt_workers = st.number_input("Optimization Workers", value=DEFAULT_WORKERS, step=1, min_value=1)
        opt_grid_resolution_m = st.number_input("Optimization Grid Resolution (m)", value=DEFAULT_GRID_RESOLUTION_M, step=5.0)
        opt_radius_m = st.number_input("Optimization Radius (m)", value=DEFAULT_RADIUS_M, step=50.0)
        opt_run_triggered = st.form_submit_button("Run Optimization Test", type="primary")
    if opt_run_triggered:
        if not opt_target_id.strip():
            st.error("Target ID is required for the optimization test.")
        else:
            opt_config = OptimizationTestConfig(
                project_id=int(project_id),
                region=region,
                target_type=opt_target_type,
                target_id=opt_target_id.strip(),
                impact_radius_m=float(opt_impact_radius_m),
                delta_lat=float(opt_delta_lat),
                delta_lon=float(opt_delta_lon),
                delta_azimuth=float(opt_delta_azimuth),
                delta_electrical_tilt=float(opt_delta_etilt),
                delta_mechanical_tilt=float(opt_delta_mtilt),
                delta_tx_power=float(opt_delta_tx),
                delta_antenna_height=float(opt_delta_height),
                neighbor_site_count=int(opt_neighbor_site_count),
                max_interference_sites=int(opt_max_interference_sites),
                workers=int(opt_workers),
                grid_resolution_m=float(opt_grid_resolution_m),
                radius_m=float(opt_radius_m),
                output_root=OUTPUT_ROOT,
            )
            with st.spinner("Running optimization test using the saved smoothed baseline results."):
                opt_run_dir = run_optimization_test(opt_config)
            st.success(f"Optimization test completed: {opt_run_dir}")

    st.sidebar.header("Tilt Recommendation Test")
    with st.sidebar.form("tilt_recommendation_test_form"):
        tilt_operator = st.text_input("Tilt Operator", value="Airtel")
        tilt_rsrp = st.number_input("Tilt RSRP Threshold", value=-105.0, step=1.0)
        tilt_rsrq = st.number_input("Tilt RSRQ Threshold", value=-15.0, step=1.0)
        tilt_sinr = st.number_input("Tilt SINR Threshold", value=0.0, step=1.0)
        tilt_run_triggered = st.form_submit_button("Run Tilt Recommendation Test", type="primary")
    if tilt_run_triggered:
        tilt_config = TiltRecommendationTestConfig(
            project_id=int(project_id),
            region=region,
            operator=tilt_operator.strip() or None,
            rsrp_threshold=float(tilt_rsrp),
            rsrq_threshold=float(tilt_rsrq),
            sinr_threshold=float(tilt_sinr),
            output_root=OUTPUT_ROOT,
        )
        with st.spinner("Running tilt recommendation test using baseline + geo context."):
            tilt_run_dir = run_tilt_recommendation_test(tilt_config)
        st.success(f"Tilt recommendation test completed: {tilt_run_dir}")

    st.sidebar.header("Tilt Recommendation Optimisation Test")
    with st.sidebar.form("tilt_recommendation_optimization_test_form"):
        tilt_opt_operator = st.text_input("Tilt Optimization Operator", value="Airtel")
        tilt_opt_scenario_id = st.number_input("Recommendation Scenario ID (0 = latest)", value=0, step=1, min_value=0)
        tilt_opt_impact_radius_m = st.number_input("Tilt Optimization Impact Radius (m)", value=500.0, step=100.0, min_value=100.0)
        tilt_opt_neighbor_site_count = st.number_input("Tilt Optimization Neighbor Site Count", value=2, step=1, min_value=0)
        tilt_opt_max_interference_sites = st.number_input("Tilt Optimization Max Interference Sites", value=10, step=1, min_value=1)
        tilt_opt_workers = st.number_input("Tilt Optimization Workers", value=DEFAULT_WORKERS, step=1, min_value=1)
        tilt_opt_grid_resolution_m = st.number_input("Tilt Optimization Grid Resolution (m)", value=10.0, step=5.0)
        tilt_opt_radius_m = st.number_input("Tilt Optimization Radius (m)", value=DEFAULT_RADIUS_M, step=50.0)
        tilt_opt_run_triggered = st.form_submit_button("Run Tilt Optimization Test", type="primary")
    if tilt_opt_run_triggered:
        tilt_opt_config = TiltOptimizationTestConfig(
            project_id=int(project_id),
            region=region,
            operator=tilt_opt_operator.strip() or None,
            recommendation_scenario_id=int(tilt_opt_scenario_id) if int(tilt_opt_scenario_id) > 0 else None,
            impact_radius_m=float(tilt_opt_impact_radius_m),
            neighbor_site_count=int(tilt_opt_neighbor_site_count),
            max_interference_sites=int(tilt_opt_max_interference_sites),
            workers=int(tilt_opt_workers),
            grid_resolution_m=float(tilt_opt_grid_resolution_m),
            radius_m=float(tilt_opt_radius_m),
            output_root=OUTPUT_ROOT,
        )
        with st.spinner("Running optimization from saved tilt recommendation rows."):
            tilt_opt_run_dir = run_tilt_recommendation_optimization_test(tilt_opt_config)
        st.success(f"Tilt recommendation optimization test completed: {tilt_opt_run_dir}")

    if dashboard_page == "Coverage":
        _render_coverage_page(int(project_id))
        return
    if dashboard_page == "Coverage - Clutter":
        _render_coverage_clutter_page(int(project_id))
        return
    if dashboard_page == "Optimization":
        _render_optimization_page(int(project_id))
        return
    if dashboard_page == "Tilt Recommendation":
        _render_tilt_recommendation_page(int(project_id))
        return
    if dashboard_page == "Tilt Recommendation Optimisation":
        _render_tilt_optimization_page(int(project_id))
        return

    runs = _list_baseline_runs(int(project_id))
    if not runs:
        st.info("No RF debug runs found yet. Use the sidebar to launch one.")
    else:
        run_labels = [run.name for run in runs]
        selected_label = st.selectbox("Available RF Debug Runs", options=run_labels, index=0)
        run_dir = next(run for run in runs if run.name == selected_label)
        summary = _load_summary(run_dir)

        st.subheader("Run Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runtime (sec)", summary.get("total_runtime_sec"))
        c2.metric("RF Grid Rows", summary.get("rows", {}).get("rf_prediction_grid"))
        c3.metric("Accuracy Points", summary.get("rows", {}).get("rf_accuracy_points"))
        c4.metric("Building Polygons", summary.get("rows", {}).get("building_polygons"))

        st.markdown("**Timing Breakdown**")
        timings_df = pd.DataFrame(
            [{"step": key, "seconds": value} for key, value in summary.get("timings_sec", {}).items()]
        )
        if not timings_df.empty:
            st.dataframe(timings_df, use_container_width=True)

        if summary.get("building_alignment"):
            st.markdown(f"**Building Alignment**: `{summary['building_alignment']}`")
        if summary.get("production_style_prediction"):
            st.markdown("**Prediction Mode**: `production_style_rf_polygon`")
        if summary.get("holdout_strategy"):
            if summary.get("holdout_strategy") == "validation_only_sessions":
                st.markdown(
                    f"**Validation Mode**: `dt_validation_only` | "
                    f"validation_sessions=`{summary.get('holdout_sessions', [])}`"
                )
            else:
                st.markdown(
                    f"**Validation Split**: `{summary['holdout_strategy']}` | "
                    f"train_sessions=`{summary.get('train_sessions', [])}` | "
                    f"holdout_sessions=`{summary.get('holdout_sessions', [])}`"
                )

        feature_diag = summary.get("feature_diagnostics", {})
        if feature_diag:
            st.markdown("**Feature Diagnostics**")
            feature_diag_df = pd.DataFrame(
                [{"feature": key, **value} for key, value in feature_diag.items()]
            )
            st.dataframe(feature_diag_df, use_container_width=True)

        cluster_counts = summary.get("cluster_counts", {})
        if cluster_counts:
            st.markdown("**Cluster Counts**")
            cluster_df = pd.DataFrame(
                [{"morphology_cluster": key, "count": value} for key, value in cluster_counts.items()]
            )
            st.dataframe(cluster_df, use_container_width=True)

        experimental_model = summary.get("experimental_model", {})
        if experimental_model:
            st.markdown("**Experimental Model**")
            experimental_df = pd.DataFrame(
                [
                    {
                        "metric": metric,
                        "train_rows": info.get("train_rows"),
                        "feature_count": info.get("feature_count"),
                        "top_features": ", ".join(
                            f"{name}={value}" for name, value in info.get("top_features", {}).items()
                        ),
                    }
                    for metric, info in experimental_model.items()
                ]
            )
            st.dataframe(experimental_df, use_container_width=True)

        artifacts = summary.get("artifacts", {})
        required_artifacts = [
            "project_polygon",
            "analysis_grid",
            "analysis_grid_features",
            "site_df",
            "drive_df",
            "rf_prediction_grid",
            "rf_accuracy_points",
        ]
        missing_artifacts = [key for key in required_artifacts if key not in artifacts]
        if missing_artifacts:
            st.error(
                "Selected run does not contain RF Debug artifacts. "
                f"Missing: {', '.join(missing_artifacts)}"
            )
            st.info("Pick an RF Debug run on the RF Debug page, or open the Coverage page for coverage runs.")
            return
        polygon_gdf = gpd.read_file(artifacts["project_polygon"])
        grid_gdf = gpd.read_file(artifacts["analysis_grid"])
        buildings_path = Path(artifacts.get("buildings", ""))
        buildings_gdf = gpd.read_file(buildings_path) if buildings_path.exists() else None
        analysis_features_df = pd.read_csv(artifacts["analysis_grid_features"])
        site_df = pd.read_csv(artifacts["site_df"])
        site_df = _prepare_site_selection_df(site_df)
        drive_df = pd.read_csv(artifacts["drive_df"])
        pred_df = pd.read_parquet(artifacts["rf_prediction_grid"])
        dt_eval = pd.read_csv(artifacts["rf_accuracy_points"])
        st.subheader("Metric Comparison")
        tabs = st.tabs(["RSRP", "RSRQ", "SINR"])
        metric_names = ["RSRP_meas", "RSRQ_meas", "SINR_meas"]
        for tab, metric_name in zip(tabs, metric_names):
            with tab:
                _render_metric_compare(summary, metric_name)
                _render_metric_detail_table(summary, metric_name)

        _render_range_summary(dt_eval)

        st.subheader("RF Comparison Images")
        image_tabs = st.tabs(["RSRP Images", "RSRQ Images", "SINR Images"])
        for tab, metric_name in zip(image_tabs, ("RSRP", "RSRQ", "SINR")):
            with tab:
                st.markdown(f"**{metric_name}: Holdout DT vs Source RF Full Polygon**")
                _render_signal_image(dt_eval, pred_df, metric_name)
                _render_error_image(dt_eval, metric_name)

        st.subheader("Validation Charts")
        chart_tabs = st.tabs(["RSRP Charts", "RSRQ Charts", "SINR Charts"])
        for tab, metric_name in zip(chart_tabs, ("RSRP", "RSRQ", "SINR")):
            with tab:
                _render_scatter_validation(dt_eval, metric_name)
                _render_error_distribution(dt_eval, metric_name)

        st.subheader("Maps")
        map_control_cols = st.columns(4)
        kpi_map_choice = map_control_cols[0].selectbox(
            "RF KPI",
            options=[
                ("RSRP", "pred_rsrp"),
                ("RSRQ", "pred_rsrq"),
                ("SINR", "pred_sinr"),
                ("RSRP Geo", "pred_rsrp_geo"),
                ("RSRQ Geo", "pred_rsrq_geo"),
                ("SINR Geo", "pred_sinr_geo"),
            ],
            format_func=lambda item: item[0],
            index=0,
        )[1]
        selection_mode = map_control_cols[1].radio("Coverage Scope", options=["All", "Sector", "NodeB"], horizontal=True)
        available_sectors = sorted(site_df["Node_Cell_ID"].dropna().astype(str).unique().tolist()) if "Node_Cell_ID" in site_df.columns else []
        available_nodebs = (
            sorted(
                [
                    value
                    for value in site_df["dashboard_nodeb_id"].dropna().astype(str).unique().tolist()
                    if value not in {"", "nan", "None"}
                ]
            )
            if "dashboard_nodeb_id" in site_df.columns
            else []
        )
        selected_sector = None
        selected_nodeb = None
        if selection_mode == "Sector" and available_sectors:
            selected_sector = map_control_cols[2].selectbox("Sector", options=available_sectors, index=0)
        elif selection_mode == "NodeB" and available_nodebs:
            selected_nodeb = map_control_cols[2].selectbox("NodeB/Site", options=available_nodebs, index=0)
        show_site_markers = map_control_cols[3].checkbox("Show Site Markers", value=True)

        map_tabs = st.tabs(["DT vs LTE Prediction", "RF Full Polygon", "Clutter Tiles", "Morphology Clusters"])
        with map_tabs[0]:
            compare_kpi = st.selectbox(
                "Compare KPI",
                options=["RSRP", "RSRQ", "SINR"],
                index=0,
                key="rf_debug_compare_kpi",
            )
            st.caption(
                "Left is raw DT from drive data. Right is final LTE prediction after geo correction and prediction-side smoothing."
            )
            compare_cols = st.columns(2)
            with compare_cols[0]:
                st.markdown("**DT Original**")
                dt_map, dt_legend = _build_dt_prediction_compare_map(
                    polygon_gdf,
                    drive_df,
                    pred_df,
                    compare_kpi,
                    "DT Original",
                )
                _render_map(
                    dt_map,
                    "dt_original_compare_map",
                )
                _render_kpi_legend(dt_legend["title"], dt_legend["bands"], dt_legend["counts"], dt_legend["total"])
            with compare_cols[1]:
                st.markdown("**LTE Prediction**")
                lte_map, lte_legend = _build_dt_prediction_compare_map(
                    polygon_gdf,
                    drive_df,
                    pred_df,
                    compare_kpi,
                    "LTE Prediction",
                )
                _render_map(
                    lte_map,
                    "lte_prediction_compare_map",
                )
                _render_kpi_legend(lte_legend["title"], lte_legend["bands"], lte_legend["counts"], lte_legend["total"])
        with map_tabs[1]:
            _render_map(
                _build_map(
                    polygon_gdf,
                    site_df,
                    drive_df,
                    pred_df,
                    buildings_gdf,
                    grid_gdf,
                    show_geo=False,
                    kpi_col=kpi_map_choice,
                    selected_sector=selected_sector,
                    selected_nodeb=selected_nodeb,
                    show_site_markers=show_site_markers,
                ),
                "baseline_map",
            )
        with map_tabs[2]:
            _render_map(
                _build_map(
                    polygon_gdf,
                    site_df.iloc[:50],
                    drive_df.iloc[:1],
                    pred_df.iloc[:1],
                    buildings_gdf,
                    grid_gdf,
                    show_geo=False,
                    show_site_markers=show_site_markers,
                ),
                "clutter_map",
            )
        with map_tabs[3]:
            _render_map(
                _build_map(
                    polygon_gdf,
                    site_df.iloc[:50],
                    drive_df.iloc[:1],
                    pred_df.iloc[:1],
                    buildings_gdf,
                    grid_gdf,
                    show_geo=True,
                    show_site_markers=show_site_markers,
                ),
                "cluster_map",
            )

        if {"RSRP_meas", "RSRP_pred"}.issubset(dt_eval.columns):
            scatter = px.scatter(
                dt_eval,
                x="RSRP_meas",
                y="RSRP_pred",
                color="morphology_cluster" if "morphology_cluster" in dt_eval.columns else None,
                title="Baseline RF: Measured vs Predicted RSRP",
                opacity=0.65,
            )
            st.plotly_chart(scatter, use_container_width=True)

        feature_candidates = [
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
        ]
        available_features = [col for col in feature_candidates if col in analysis_features_df.columns]
        if available_features:
            st.subheader("Feature Visualization")
            selected_feature = st.selectbox("Feature Parameter", available_features, index=0)
            _render_feature_map(analysis_features_df, selected_feature)

        st.subheader("Run Logs")
        run_log_path = Path(artifacts["run_log"])
        if run_log_path.exists():
            st.text_area("Test Lab Log", run_log_path.read_text(encoding="utf-8", errors="ignore"), height=320)
        rf_log_path = summary.get("rf_log_path")
        if rf_log_path and Path(rf_log_path).exists():
            st.text_area("Source RF Log", Path(rf_log_path).read_text(encoding="utf-8", errors="ignore"), height=320)


if __name__ == "__main__":
    main()
