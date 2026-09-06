from __future__ import annotations

import base64
import math
import re
import sys
from io import BytesIO
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import transform
from streamlit_folium import st_folium

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import streamlit_project210_phase11_12_dashboard as phase11_12_view
import streamlit_project210_phase13_beam_check as phase13_view
import streamlit_project210_phase14_tilt_scale_fix as phase14_view
import streamlit_project210_phase15_radius_progression as phase15_view
import streamlit_project210_phase16_site_local_geo as phase16_view
import streamlit_project210_phase17_full_polygon_comparison as phase17_view
import streamlit_project210_phase19_branch_calibrated_comparison as phase19_view
import streamlit_project210_phase20_21_22_dashboard as phase20_21_22_view
import streamlit_taiwan_mapdata_dashboard as mapdata_view
import streamlit_debug_frequency_only as debug_freq_view

DATA_DIR = THIS_DIR / "data"

PROJECT_CONFIG = {
    "Project 196 India": {
        "project_id": 196,
        "region": "India",
        "dir": DATA_DIR / "project_196_india",
    },
    "Project 210 Taiwan": {
        "project_id": 210,
        "region": "Taiwan",
        "dir": DATA_DIR / "project_210_taiwan",
    },
    "Taiwan Debug (frequency-only 4G/5G)": {
        "project_id": 210,
        "region": "Taiwan",
        "dir": DATA_DIR / "project_210_taiwan",
        "debug_frequency_only": True,
    },
}

MODEL_CONFIG = {
    "Cost231": {
        "label": "Cost231",
        "subdir": "cost231",
        "surface_pattern": "cost231_offset_corrected_surface_project{project_id}",
        "dt_pattern": "cost231_dt_match_project{project_id}",
        "offset_patterns": [
            "cost231_offsets_cells_project{project_id}",
            "cost231_offsets_102_cells_project{project_id}",
        ],
        "raw_col": "raw_cost231_rsrp",
        "dt_raw_col": "raw_cost231_at_dt_rsrp",
        "dt_delta_col": "dt_minus_cost231_db",
    },
    "Cost231 Phase 9 GridAnalytics compatible": {
        "label": "Cost231 Phase 9 GridAnalytics compatible",
        "subdir": "cost231_phase9_gridanalytics_compatible",
        "surface_pattern": "phase9_directional_raw_corrected_surface_project{project_id}",
        "dt_pattern": "phase9_dt_match_project{project_id}",
        "offset_patterns": [
            "phase9_offsets_project{project_id}",
        ],
        "raw_col": "raw_cost231_rsrp",
        "dt_raw_col": "raw_cost231_at_dt_rsrp",
        "dt_delta_col": "dt_minus_cost231_db",
    },
    "Cost231 Phase 10 site technology refresh": {
        "label": "Cost231 Phase 10 site technology refresh",
        "subdir": "cost231_phase10_site_technology_refresh",
        "surface_pattern": "phase10_directional_raw_corrected_surface_project{project_id}",
        "dt_pattern": "phase10_dt_match_project{project_id}",
        "offset_patterns": [
            "phase10_offsets_project{project_id}",
        ],
        "raw_col": "raw_cost231_rsrp",
        "dt_raw_col": "raw_cost231_at_dt_rsrp",
        "dt_delta_col": "dt_minus_cost231_db",
    },
    "P1812": {
        "label": "P1812",
        "subdir": "p1812",
        "surface_pattern": "p1812_offset_corrected_surface_project{project_id}",
        "dt_pattern": "p1812_dt_match_project{project_id}",
        "offset_patterns": [
            "p1812_offsets_cells_project{project_id}",
            "p1812_offsets_102_cells_project{project_id}",
        ],
        "raw_col": "plain_p1812_rsrp",
        "dt_raw_col": "plain_p1812_at_dt_rsrp",
        "dt_delta_col": "dt_minus_p1812_db",
    },
}

RSRP_BINS = [
    (-140, -120, "#b91c1c", "-140 to -120"),
    (-120, -110, "#ef4444", "-120 to -110"),
    (-110, -100, "#f97316", "-110 to -100"),
    (-100, -95, "#facc15", "-100 to -95"),
    (-95, -85, "#84cc16", "-95 to -85"),
    (-85, -44, "#16a34a", "-85 to -44"),
]

BAND_COLORS = {
    "3": "#0891b2",
    "28": "#dc2626",
    "78": "#7c3aed",
    "850": "#8b5cf6",
    "900": "#06b6d4",
    "1800": "#ef4444",
    "2100": "#f59e0b",
    "2300": "#2563eb",
}
BAND_SIZE_M = {
    "3": 55,
    "28": 75,
    "78": 105,
    "850": 45,
    "900": 55,
    "1800": 70,
    "2100": 85,
    "2300": 105,
}


st.set_page_config(
    page_title="Project Propagation Offset",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _read_frame(parquet_path: Path, csv_path: Path | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path, columns=columns)
        except Exception:
            return pd.read_parquet(parquet_path)
    if csv_path and csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False, usecols=lambda col: columns is None or col in columns)
    raise FileNotFoundError(parquet_path)


def _first_existing_stem(base_dir: Path, stems: list[str]) -> tuple[Path, Path]:
    for stem in stems:
        parquet_path = base_dir / f"{stem}.parquet"
        csv_path = base_dir / f"{stem}.csv"
        if parquet_path.exists() or csv_path.exists():
            return parquet_path, csv_path
    first = stems[0] if stems else "missing"
    return base_dir / f"{first}.parquet", base_dir / f"{first}.csv"


def _project_cfg(project_key: str) -> dict:
    return PROJECT_CONFIG[project_key]


def _model_paths(project_key: str, model_key: str) -> dict[str, Path | str]:
    project = _project_cfg(project_key)
    cfg = MODEL_CONFIG[model_key].copy()
    project_id = int(project["project_id"])
    base_dir = Path(project["dir"]) / str(cfg["subdir"])
    cfg["base_dir"] = base_dir
    for key in ["surface", "dt"]:
        stem = str(cfg[f"{key}_pattern"]).format(project_id=project_id)
        cfg[f"{key}_parquet"] = base_dir / f"{stem}.parquet"
        cfg[f"{key}_csv"] = base_dir / f"{stem}.csv"
    offset_stems = [pattern.format(project_id=project_id) for pattern in cfg["offset_patterns"]]
    cfg["offsets_parquet"], cfg["offsets_csv"] = _first_existing_stem(base_dir, offset_stems)
    return cfg


def _model_available(project_key: str, model_key: str) -> bool:
    cfg = _model_paths(project_key, model_key)
    needed = ["surface", "dt", "offsets"]
    return all(
        Path(cfg[f"{key}_parquet"]).exists() or Path(cfg[f"{key}_csv"]).exists()
        for key in needed
    )


def _model_display_name(project_key: str, model_key: str) -> str:
    if project_key == "Project 210 Taiwan" and model_key == "Cost231":
        return "Cost231 Phase 1"
    return str(MODEL_CONFIG[model_key]["label"])


@st.cache_data(show_spinner=False)
def load_surface(project_key: str, model_key: str) -> pd.DataFrame:
    cfg = _model_paths(project_key, model_key)
    raw_col = str(cfg["raw_col"])
    columns = [
        "grid_id",
        "lat",
        "lon",
        raw_col,
        "corrected_rsrp",
        "offset_corrected_rsrp",
        "offset_db",
        "dt_replacement_count",
        "dt_replaced",
        "site",
        "sector",
        "band",
        "strict_cell_key",
        "site_sector_band_key",
    ]
    df = _read_frame(Path(cfg["surface_parquet"]), Path(cfg["surface_csv"]), columns=columns)
    for col in [
        "lat",
        "lon",
        str(cfg["raw_col"]),
        "corrected_rsrp",
        "offset_corrected_rsrp",
        "offset_db",
        "dt_replacement_count",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["site", "sector", "band", "strict_cell_key", "site_sector_band_key", "grid_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["raw_model_rsrp"] = pd.to_numeric(df[str(cfg["raw_col"])], errors="coerce")
    return df.dropna(subset=["lat", "lon", "raw_model_rsrp", "corrected_rsrp"]).copy()


@st.cache_data(show_spinner=False)
def load_dt(project_key: str, model_key: str) -> pd.DataFrame:
    cfg = _model_paths(project_key, model_key)
    columns = [
        "lat",
        "lon",
        "rsrp_measured",
        str(cfg["dt_raw_col"]),
        "after_at_dt_pixel_rsrp",
        str(cfg["dt_delta_col"]),
        "nearest_grid_distance_m",
        "assigned_strict_cell_key",
        "assigned_site",
        "assigned_sector",
        "assigned_band",
    ]
    df = _read_frame(Path(cfg["dt_parquet"]), Path(cfg["dt_csv"]), columns=columns)
    for col in [
        "lat",
        "lon",
        "rsrp_measured",
        str(cfg["dt_raw_col"]),
        "after_at_dt_pixel_rsrp",
        str(cfg["dt_delta_col"]),
        "nearest_grid_distance_m",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["assigned_strict_cell_key", "assigned_site", "assigned_sector", "assigned_band"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["raw_at_dt_rsrp"] = pd.to_numeric(df[str(cfg["dt_raw_col"])], errors="coerce")
    df["dt_minus_model_db"] = pd.to_numeric(df[str(cfg["dt_delta_col"])], errors="coerce")
    return df.dropna(subset=["lat", "lon", "rsrp_measured"]).copy()


@st.cache_data(show_spinner=False)
def load_offsets(project_key: str, model_key: str) -> pd.DataFrame:
    cfg = _model_paths(project_key, model_key)
    columns = [
        "strict_cell_key",
        "site_sector_band_key",
        "site_key",
        "sector_key",
        "band_key",
        "operator_key",
        "offset_db",
        "dt_count",
        "offset_mean_db",
        "offset_source",
    ]
    df = _read_frame(Path(cfg["offsets_parquet"]), Path(cfg["offsets_csv"]), columns=columns)
    for col in ["offset_db", "dt_count", "offset_mean_db"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["strict_cell_key", "site_sector_band_key", "site_key", "sector_key", "band_key", "offset_source"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_sites(project_key: str) -> pd.DataFrame:
    project = _project_cfg(project_key)
    project_id = int(project["project_id"])
    baseline_scope = Path(project["dir"]) / "baseline_fetch_scope"
    candidates = [
        baseline_scope / f"site_identity_strict_cells_project{project_id}.csv",
        baseline_scope / "site_identity_102_strict_cells.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    df = pd.read_csv(path, low_memory=False)
    for col in ["lat", "lon", "azimuth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["site", "band", "Node_Cell_ID", "rf_identity_key"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["strict_cell_key"] = df["Node_Cell_ID"].astype(str)
    df["band"] = df["band"].astype(str)
    df["sector_label"] = df["sector"].astype(str)
    return df.dropna(subset=["lat", "lon"]).copy()


@st.cache_data(show_spinner=False)
def load_project_polygon_coords(project_key: str) -> list[list[float]]:
    project = _project_cfg(project_key)
    project_id = int(project["project_id"])
    region_path = Path(project["dir"]) / "geo_db" / f"map_regions_project_{project_id}_active.csv"
    regions = pd.read_csv(region_path)
    if regions.empty or "region_wkt" not in regions.columns:
        return []
    geom = wkt.loads(str(regions.iloc[0]["region_wkt"]))
    geom = transform(lambda x, y, z=None: (y, x) if z is None else (y, x, z), geom)
    if not geom.is_valid:
        geom = geom.buffer(0)
    coords = list(geom.exterior.coords)
    return [[lat, lon] for lon, lat in coords]


def clean_options(values: pd.Series) -> list[str]:
    out = sorted([str(v) for v in values.dropna().unique().tolist()], key=lambda x: (len(x), x))
    return ["All"] + out


def band_group_for_band(series: pd.Series) -> pd.Series:
    return np.where(series.astype(str).eq("78"), "5G", "4G")


def filter_band_group(df: pd.DataFrame, band_group: str, band_col: str = "band") -> pd.DataFrame:
    if band_group == "All" or band_col not in df.columns:
        return df
    groups = band_group_for_band(df[band_col])
    return df.loc[groups == band_group].copy()


def color_for_rsrp(value: float) -> str:
    if pd.isna(value):
        return "#6b7280"
    for low, high, color, _ in RSRP_BINS:
        if low <= float(value) < high or (float(value) == high and high == -44):
            return color
    return "#6b7280"


def filter_surface(df: pd.DataFrame, band_group: str, band: str, sector: str, site: str, cell: str) -> pd.DataFrame:
    out = filter_band_group(df, band_group, "band")
    if band != "All":
        out = out[out["band"].astype(str) == str(band)]
    if sector != "All":
        out = out[out["sector"].astype(str) == str(sector)]
    if site != "All":
        out = out[out["site"].astype(str) == str(site)]
    if cell != "All":
        out = out[out["strict_cell_key"].astype(str) == str(cell)]
    return out.copy()


def filter_sites(df: pd.DataFrame, band_group: str, band: str, sector: str, site: str, cell: str) -> pd.DataFrame:
    out = filter_band_group(df, band_group, "band")
    if band != "All":
        out = out[out["band"].astype(str) == str(band)]
    if sector != "All":
        out = out[out["sector_label"].astype(str) == str(sector)]
    if site != "All":
        out = out[out["site"].astype(str) == str(site)]
    if cell != "All":
        out = out[out["strict_cell_key"].astype(str) == str(cell)]
    return out.copy()


def filter_dt(df: pd.DataFrame, band_group: str, band: str, sector: str, site: str, cell: str) -> pd.DataFrame:
    out = filter_band_group(df, band_group, "assigned_band")
    if band != "All" and "assigned_band" in out.columns:
        out = out[out["assigned_band"].astype(str) == str(band)]
    if sector != "All" and "assigned_sector" in out.columns:
        out = out[out["assigned_sector"].astype(str) == str(sector)]
    if site != "All" and "assigned_site" in out.columns:
        out = out[out["assigned_site"].astype(str) == str(site)]
    if cell != "All" and "assigned_strict_cell_key" in out.columns:
        out = out[out["assigned_strict_cell_key"].astype(str) == str(cell)]
    return out.copy()


def aggregate_grid(df: pd.DataFrame, value_col: str, mode: str) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["grid_id", "lat", "lon"]
    agg_name = {"Best server": "max", "Mean": "mean", "Worst": "min"}[mode]
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            rsrp=(value_col, agg_name),
            cell_count=("strict_cell_key", "nunique"),
            dt_replaced=("dt_replaced", "sum"),
            offset_db=("offset_db", "median"),
        )
        .reset_index()
    )
    return out


def triangle_points(lat: float, lon: float, azimuth: float, radius_m: float, beamwidth_deg: float = 58) -> list[list[float]]:
    def dest(angle_deg: float, dist_m: float) -> tuple[float, float]:
        brng = math.radians(angle_deg)
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        angular = dist_m / 6371000.0
        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular)
            + math.cos(lat1) * math.sin(angular) * math.cos(brng)
        )
        lon2 = lon1 + math.atan2(
            math.sin(brng) * math.sin(angular) * math.cos(lat1),
            math.cos(angular) - math.sin(lat1) * math.sin(lat2),
        )
        return math.degrees(lat2), math.degrees(lon2)

    left = dest(azimuth - beamwidth_deg / 2.0, radius_m)
    right = dest(azimuth + beamwidth_deg / 2.0, radius_m)
    return [[lat, lon], [left[0], left[1]], [right[0], right[1]], [lat, lon]]


def add_project_polygon(fmap: folium.Map, coords: list[list[float]], project_label: str) -> None:
    if not coords:
        return
    folium.Polygon(
        locations=coords,
        color="#2563eb",
        weight=2,
        fill=False,
        tooltip=f"{project_label} polygon",
    ).add_to(fmap)


def add_grid_layer(fmap: folium.Map, grid_df: pd.DataFrame, layer_name: str) -> None:
    if grid_df.empty:
        return
    layer = folium.FeatureGroup(name=layer_name, show=True)
    work = grid_df[["lat", "lon", "rsrp"]].dropna().copy()
    if work.empty:
        return
    work["lat_key"] = work["lat"].round(7)
    work["lon_key"] = work["lon"].round(7)
    lat_values = np.sort(work["lat_key"].unique())[::-1]
    lon_values = np.sort(work["lon_key"].unique())
    if len(lat_values) * len(lon_values) > 2_500_000:
        work = work.sample(2_500_000, random_state=7)
    lat_index = {value: idx for idx, value in enumerate(lat_values)}
    lon_index = {value: idx for idx, value in enumerate(lon_values)}
    image = np.zeros((len(lat_values), len(lon_values), 4), dtype=np.uint8)
    color_lookup = {
        label: tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        for _, _, color, label in RSRP_BINS
    }
    for row in work.itertuples(index=False):
        rgba = (107, 114, 128)
        for low, high, color, label in RSRP_BINS:
            if low <= float(row.rsrp) < high or (float(row.rsrp) == high and high == -44):
                rgba = color_lookup[label]
                break
        image[lat_index[row.lat_key], lon_index[row.lon_key]] = [rgba[0], rgba[1], rgba[2], 172]
    pil_image = Image.fromarray(image, mode="RGBA").resize(
        (max(len(lon_values) * 3, 64), max(len(lat_values) * 3, 64)),
        resample=Image.Resampling.NEAREST,
    )
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    bounds = [[float(work["lat"].min()), float(work["lon"].min())], [float(work["lat"].max()), float(work["lon"].max())]]
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{encoded}",
        bounds=bounds,
        opacity=0.78,
        name=layer_name,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(layer)
    layer.add_to(fmap)


def add_site_triangles(fmap: folium.Map, site_df: pd.DataFrame) -> None:
    layer = folium.FeatureGroup(name=f"{len(site_df):,} site-sector-band triangles", show=True)
    for row in site_df.itertuples(index=False):
        band = str(row.band)
        color = BAND_COLORS.get(band, "#111827")
        size_m = BAND_SIZE_M.get(band, 65)
        label = (
            f"<b>Site:</b> {row.site}<br>"
            f"<b>Sector:</b> {row.sector_label}<br>"
            f"<b>Band:</b> {band}<br>"
            f"<b>Azimuth:</b> {float(row.azimuth):.1f}<br>"
            f"<b>Key:</b> {row.strict_cell_key}"
        )
        folium.Polygon(
            locations=triangle_points(float(row.lat), float(row.lon), float(row.azimuth), size_m),
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.62,
            popup=folium.Popup(label, max_width=320),
        ).add_to(layer)
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)],
            radius=3,
            color="#ffffff",
            fill=True,
            fill_color=color,
            fill_opacity=1,
            weight=1,
        ).add_to(layer)
    layer.add_to(fmap)


def add_dt_points(fmap: folium.Map, dt_df: pd.DataFrame, model_label: str) -> None:
    if dt_df.empty:
        return
    layer = folium.FeatureGroup(name="DT pixels", show=False)
    sample = dt_df if len(dt_df) <= 2500 else dt_df.sample(2500, random_state=7)
    for row in sample.itertuples(index=False):
        value = float(row.rsrp_measured)
        raw_at_dt = pd.to_numeric(getattr(row, "raw_at_dt_rsrp", np.nan), errors="coerce")
        after_at_pixel = pd.to_numeric(getattr(row, "after_at_dt_pixel_rsrp", np.nan), errors="coerce")
        offset_db = pd.to_numeric(getattr(row, "dt_minus_model_db", np.nan), errors="coerce")
        after_label = f"{float(after_at_pixel):.2f} dBm" if np.isfinite(after_at_pixel) else "N/A"
        raw_label = f"{float(raw_at_dt):.2f} dBm" if np.isfinite(raw_at_dt) else "N/A"
        offset_label = f"{float(offset_db):+.2f} dB" if np.isfinite(offset_db) else "N/A"
        popup = (
            f"<b>DT RSRP:</b> {value:.2f} dBm<br>"
            f"<b>{model_label} at DT:</b> {raw_label}<br>"
            f"<b>After at pixel:</b> {after_label}<br>"
            f"<b>Offset:</b> {offset_label}"
        )
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)],
            radius=3,
            color="#111827",
            fill=True,
            fill_color=color_for_rsrp(value),
            fill_opacity=0.9,
            weight=1,
            popup=folium.Popup(popup, max_width=260),
        ).add_to(layer)
    layer.add_to(fmap)


def _legend_rows(grid_df: pd.DataFrame) -> tuple[str, int]:
    values = pd.to_numeric(grid_df["rsrp"], errors="coerce").dropna() if "rsrp" in grid_df.columns else pd.Series(dtype=float)
    total = int(len(values))
    rows = []
    for low, high, color, label in RSRP_BINS:
        if total:
            mask = (values >= low) & (values < high)
            if high == -44:
                mask = (values >= low) & (values <= high)
            count = int(mask.sum())
            pct = count / total * 100.0
        else:
            count = 0
            pct = 0.0
        rows.append(
            "<div style='display:grid;grid-template-columns:16px 1fr auto;gap:8px;align-items:center;margin:8px 0;'>"
            f"<span style='background:{color};width:11px;height:11px;border-radius:50%;display:inline-block;'></span>"
            f"<span>{label}</span>"
            f"<span style='font-variant-numeric:tabular-nums;'>{count:,} ({pct:.1f}%)</span>"
            "</div>"
        )
    return "".join(rows), total


def add_legend(fmap: folium.Map, title: str, grid_df: pd.DataFrame) -> None:
    rsrp_rows, total = _legend_rows(grid_df)
    band_rows = "".join(
        f"<div><span style='background:{color};width:10px;height:10px;display:inline-block;margin-right:6px'></span>Band {band} size {BAND_SIZE_M.get(band, 65)}m</div>"
        for band, color in BAND_COLORS.items()
    )
    html = f"""
    <div style="position: fixed; top: 78px; right: 18px; z-index: 9999;
         min-width: 265px; background: #1f2937; color: #f9fafb; padding: 12px 14px;
         border: 1px solid rgba(255,255,255,.12); border-radius: 8px; font-size: 12px;
         box-shadow: 0 12px 28px rgba(0,0,0,.28); font-family: Arial, sans-serif;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
        <b>{title} RSRP (dBm)</b>
        <span style="color:#9ca3af;font-size:11px;">Total</span>
      </div>
      <div style="margin-top:8px">{rsrp_rows}</div>
      <div style="border-top:1px solid rgba(255,255,255,.14);margin-top:8px;padding-top:8px;
           display:flex;justify-content:space-between;color:#9ca3af;font-size:11px;">
        <span>Total</span><span>{total:,}</span>
      </div>
      <div style="border-top:1px solid rgba(255,255,255,.14);margin-top:8px;padding-top:8px;color:#d1d5db;">
        {band_rows}
      </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(html))


def make_map(
    grid_df: pd.DataFrame,
    site_df: pd.DataFrame,
    dt_df: pd.DataFrame,
    polygon_coords: list[list[float]],
    title: str,
    model_label: str,
    project_label: str,
) -> folium.Map:
    if polygon_coords:
        center = [np.mean([p[0] for p in polygon_coords]), np.mean([p[1] for p in polygon_coords])]
    elif not grid_df.empty:
        center = [float(grid_df["lat"].mean()), float(grid_df["lon"].mean())]
    else:
        center = [28.64, 77.35]
    fmap = folium.Map(location=center, zoom_start=15, tiles="CartoDB positron", control_scale=True)
    add_project_polygon(fmap, polygon_coords, project_label)
    add_grid_layer(fmap, grid_df, title)
    add_site_triangles(fmap, site_df)
    add_dt_points(fmap, dt_df, model_label)
    folium.LayerControl(collapsed=False).add_to(fmap)
    add_legend(fmap, title, grid_df)
    return fmap


def cdf_trace(values: pd.Series, name: str, color: str) -> go.Scatter:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    y = np.arange(1, len(arr) + 1) / max(len(arr), 1) * 100
    return go.Scatter(x=arr, y=y, mode="lines", name=f"{name} (n={len(arr):,})", line=dict(color=color, width=3))


def render_cdf(surface_df: pd.DataFrame, dt_df: pd.DataFrame, model_label: str) -> None:
    fig = go.Figure()
    fig.add_trace(cdf_trace(surface_df["raw_model_rsrp"], f"{model_label} before", "#e34b3b"))
    fig.add_trace(cdf_trace(surface_df["corrected_rsrp"], "After offset + DT replacement", "#168a52"))
    if not dt_df.empty:
        fig.add_trace(cdf_trace(dt_df["rsrp_measured"], "DT measured", "#2563eb"))
    fig.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=35, b=20),
        title="CDF for current selection",
        xaxis_title="RSRP (dBm)",
        yaxis_title="Cumulative percentage (%)",
        yaxis=dict(range=[0, 100]),
        xaxis=dict(range=[-140, -44]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def _display_key(value: str) -> str:
    if len(value) <= 80:
        return value
    return value[:77] + "..."


def main() -> None:
    st.title("Project Propagation Offset Dashboard")

    with st.sidebar:
        st.header("Filters")
        project_key = st.selectbox("Project", list(PROJECT_CONFIG.keys()), index=0)
        project = _project_cfg(project_key)

    if project.get("debug_frequency_only"):
        debug_freq_view.render()
        return

    with st.sidebar:
        section_options = ["Propagation model comparison"]
        if project_key == "Project 210 Taiwan":
            if (phase11_12_view.OUT_DIR / "phase11_12_summary.json").exists():
                section_options.append("Phase 11/12 residual blending")
            if mapdata_view.MAPDATA_DIR.exists():
                section_options.append("Mapdata inventory")
            if phase13_view.IDENTITY_PATH.exists():
                section_options.append("Phase 13 beam check")
                section_options.append("Phase 14 tilt scale fix")
                section_options.append("Phase 15 radius progression")
            if phase16_view.LOCAL_CLUTTER_PATH.exists():
                section_options.append("Phase 16 site-local geo data")
            if phase17_view.OUT_DIR.exists():
                section_options.append("Phase 17 full-polygon comparison")
            if phase19_view.OUT_DIR.exists():
                section_options.append("Phase 19 branch-calibrated comparison")
            if phase20_21_22_view.PHASE22_DIR.exists():
                section_options.append("Phase 20/22/24/25 validation")
        section = st.selectbox("Section", section_options, index=0)

    if section == "Phase 11/12 residual blending":
        phase11_12_view.render()
        return
    if section == "Mapdata inventory":
        mapdata_view.render()
        return
    if section == "Phase 13 beam check":
        phase13_view.render()
        return
    if section == "Phase 14 tilt scale fix":
        phase14_view.render()
        return
    if section == "Phase 15 radius progression":
        phase15_view.render()
        return
    if section == "Phase 16 site-local geo data":
        phase16_view.render()
        return
    if section == "Phase 17 full-polygon comparison":
        phase17_view.render()
        return
    if section == "Phase 19 branch-calibrated comparison":
        phase19_view.render()
        return
    if section == "Phase 20/22/24/25 validation":
        phase20_21_22_view.render()
        return

    with st.sidebar:
        available_models = [model for model in MODEL_CONFIG if _model_available(project_key, model)]
        if not available_models:
            st.error(f"No completed model outputs found for {project_key}.")
            st.stop()
        default_model_key = (
            "Cost231 Phase 10 site technology refresh"
            if project_key == "Project 196 India"
            and "Cost231 Phase 10 site technology refresh" in available_models
            else available_models[0]
        )
        model_key = st.selectbox(
            "Model",
            available_models,
            index=available_models.index(default_model_key),
            format_func=lambda key: _model_display_name(project_key, key),
        )
        model_label = _model_display_name(project_key, model_key)
        missing_models = [model for model in MODEL_CONFIG if model not in available_models]
        if missing_models:
            st.caption(
                "Unavailable until output files exist: "
                + ", ".join(_model_display_name(project_key, model) for model in missing_models)
            )
        surface = load_surface(project_key, model_key)
        sites = load_sites(project_key)
        dt = load_dt(project_key, model_key)
        offsets = load_offsets(project_key, model_key)
        polygon_coords = load_project_polygon_coords(project_key)
        band_group = "All"
        if project_key == "Project 210 Taiwan":
            band_group = st.selectbox("Band group", ["All", "4G", "5G"], index=0)
        band = st.selectbox("Band", clean_options(filter_band_group(surface, band_group, "band")["band"]), index=0)
        sector_options_df = filter_surface(surface, band_group, band, "All", "All", "All")
        sector = st.selectbox("Sector", clean_options(sector_options_df["sector"]), index=0)
        site_options_df = filter_surface(surface, band_group, band, sector, "All", "All")
        site = st.selectbox("Site", clean_options(site_options_df["site"]), index=0)
        cell_options_df = filter_surface(surface, band_group, band, sector, site, "All")
        cell_values = ["All"] + sorted(cell_options_df["strict_cell_key"].dropna().astype(str).unique().tolist())
        cell_labels = [_display_key(v) for v in cell_values]
        cell_label = st.selectbox("Strict cell key", cell_labels, index=0)
        cell = cell_values[cell_labels.index(cell_label)]
        agg_mode = st.selectbox("Grid aggregation", ["Best server", "Mean", "Worst"], index=0)
        show_tables = st.checkbox("Show tables", value=False)

    selected_surface = filter_surface(surface, band_group, band, sector, site, cell)
    selected_sites = filter_sites(sites, band_group, band, sector, site, cell)
    selected_dt = filter_dt(dt, band_group, band, sector, site, cell)

    before_grid = aggregate_grid(selected_surface, "raw_model_rsrp", agg_mode)
    after_grid = aggregate_grid(selected_surface, "corrected_rsrp", agg_mode)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Strict cells", f"{selected_surface['strict_cell_key'].nunique():,}")
    metric_cols[1].metric("Site triangles", f"{len(selected_sites):,}")
    metric_cols[2].metric("Surface rows", f"{len(selected_surface):,}")
    metric_cols[3].metric("Grid pixels", f"{before_grid['grid_id'].nunique() if not before_grid.empty else 0:,}")
    metric_cols[4].metric("DT rows", f"{len(selected_dt):,}")
    metric_cols[5].metric("DT replaced pixels", f"{int(selected_surface['dt_replaced'].sum()) if not selected_surface.empty else 0:,}")

    left, right = st.columns(2)
    with left:
        st.subheader(f"{model_label} Before Offset")
        st_folium(
            make_map(
                before_grid,
                selected_sites,
                selected_dt,
                polygon_coords,
                f"{model_label} before",
                model_label,
                project_key,
            ),
            height=620,
            use_container_width=True,
            returned_objects=[],
            key=f"{project_key}_{model_key}_before_map_{band_group}_{band}_{sector}_{site}_{cell}_{agg_mode}",
        )
    with right:
        st.subheader("After Offset + DT Pixel Replacement")
        st_folium(
            make_map(
                after_grid,
                selected_sites,
                selected_dt,
                polygon_coords,
                "After offset + DT replacement",
                model_label,
                project_key,
            ),
            height=620,
            use_container_width=True,
            returned_objects=[],
            key=f"{project_key}_{model_key}_after_map_{band_group}_{band}_{sector}_{site}_{cell}_{agg_mode}",
        )

    render_cdf(selected_surface, selected_dt, model_label)

    comparison_dir = Path(project["dir"]) / "comparison"
    metrics_path = comparison_dir / "cost231_vs_p1812_dt_error_metrics.csv"
    if metrics_path.exists():
        st.subheader("Cost231 vs P1812 DT Error Comparison")
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True, height=180)
        image_cols = st.columns(2)
        complete_png = comparison_dir / "cdf_complete_polygon_cost231_vs_p1812.png"
        dt_png = comparison_dir / "cdf_dt_locations_cost231_vs_p1812.png"
        if complete_png.exists():
            image_cols[0].image(str(complete_png), caption="Complete polygon CDF")
        if dt_png.exists():
            image_cols[1].image(str(dt_png), caption="DT locations CDF")

    st.subheader("Cell Offset Summary")
    selected_offsets = offsets.copy()
    if band_group != "All":
        selected_offsets = filter_band_group(selected_offsets.rename(columns={"band_key": "band"}), band_group, "band").rename(
            columns={"band": "band_key"}
        )
    if band != "All":
        selected_offsets = selected_offsets[selected_offsets["band_key"].astype(str) == str(band)]
    if sector != "All":
        selected_offsets = selected_offsets[selected_offsets["sector_key"].astype(str) == str(sector)]
    if site != "All":
        selected_offsets = selected_offsets[selected_offsets["site_key"].astype(str) == str(site)]
    if cell != "All":
        selected_offsets = selected_offsets[selected_offsets["strict_cell_key"].astype(str) == str(cell)]
    st.dataframe(
        selected_offsets[
            [
                "site_key",
                "sector_key",
                "band_key",
                "strict_cell_key",
                "dt_count",
                "offset_db",
                "offset_source",
            ]
        ].sort_values(["site_key", "sector_key", "band_key"]),
        use_container_width=True,
        height=260,
    )

    if show_tables:
        st.subheader("Current Surface Rows")
        st.dataframe(selected_surface.head(5000), use_container_width=True, height=360)
        st.subheader("Current DT Rows")
        st.dataframe(selected_dt.head(5000), use_container_width=True, height=320)


if __name__ == "__main__":
    main()
