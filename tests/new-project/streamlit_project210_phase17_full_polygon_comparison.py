from __future__ import annotations

import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle as MplRectangle

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import streamlit_project210_phase15_radius_progression as phase15

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
OUT_DIR = PROJECT_DIR / "cost231_phase17_geo_dt_comparison"

RSRP_BINS = [
    (-147, -115, "#991b1b", "-147 to -115"),
    (-115, -105, "#d97706", "-115 to -105"),
    (-105, -95, "#fef08a", "-105 to -95"),
    (-95, -85, "#22c55e", "-95 to -85"),
    (-85, 0, "#15803d", "-85 to 0"),
]


@st.cache_data(show_spinner=False)
def load_serving(tech: str) -> pd.DataFrame:
    path = OUT_DIR / f"phase17_serving_grid_{tech.lower()}_project210.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_dt() -> pd.DataFrame:
    path = OUT_DIR / "phase17_dt_with_clutter_project210.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_grid_bounds() -> pd.DataFrame:
    path = PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)[["grid_id", "min_lat", "max_lat", "min_lon", "max_lon"]]


def _color_for(value: float) -> str:
    if not np.isfinite(value):
        return "#9ca3af"
    for lo, hi, color, _label in RSRP_BINS:
        if lo <= value < hi:
            return color
    return "#9ca3af"


def _build_grid_map(serving: pd.DataFrame, grid_bounds: pd.DataFrame, value_col: str, title: str) -> folium.Map:
    # Real grid-cell rectangles (min/max lat/lon), same rendering pattern as
    # Phase 11/12's proven map - fixed-pixel CircleMarkers were leaving
    # visible gaps and made real grid cells look sparse/broken instead of a
    # filled surface. Phase 17's own output already carries min/max lat/lon
    # (from the corrected per-technology serving grid build) - only merge
    # the standalone bounds table as a fallback if they're not already there.
    if {"min_lat", "max_lat", "min_lon", "max_lon"}.issubset(serving.columns):
        df = serving.dropna(subset=["min_lat", "max_lat", "min_lon", "max_lon"])
    else:
        df = serving.merge(grid_bounds, on="grid_id", how="left").dropna(subset=["min_lat", "max_lat", "min_lon", "max_lon"])
    center = [float(df["lat"].mean()), float(df["lon"].mean())]
    fmap = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron", control_scale=True)
    layer = folium.FeatureGroup(name=title, show=True)
    for row in df.itertuples(index=False):
        val = float(getattr(row, value_col))
        popup = (
            f"<b>Grid:</b> {row.grid_id}<br>"
            f"<b>Serving cell:</b> {row.strict_cell_key}<br>"
            f"<b>Site/Sector/Band:</b> {row.site} / {row.sector} / {row.band}<br>"
            f"<b>Phase 9 (production offset):</b> {row.corrected_rsrp:.1f} dBm<br>"
            f"<b>Phase 17 (geo + DT residual):</b> {row.phase17_rsrp:.1f} dBm<br>"
            f"<b>Geo correction:</b> {row.geo_correction_db:.1f} dB<br>"
            f"<b>DT residual:</b> {row.dt_residual_db:.1f} dB<br>"
            f"<b>DT locked:</b> {bool(row.dt_replaced)}"
        )
        folium.Rectangle(
            bounds=[[row.min_lat, row.min_lon], [row.max_lat, row.max_lon]],
            color=_color_for(val),
            weight=0,
            fill=True,
            fill_color=_color_for(val),
            fill_opacity=0.85,
            tooltip=f"{val:.1f} dBm",  # shows on hover, no click needed
            popup=folium.Popup(popup, max_width=320),
        ).add_to(layer)
    layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def _build_static_image(serving: pd.DataFrame, grid_bounds: pd.DataFrame, value_col: str, title: str):
    # Flat PNG, real grid-cell rectangles - avoids the Leaflet re-tiling
    # noise you get zooming an interactive map with 10k+ shapes, and is
    # easy to drop straight into a slide/report.
    if {"min_lat", "max_lat", "min_lon", "max_lon"}.issubset(serving.columns):
        df = serving.dropna(subset=["min_lat", "max_lat", "min_lon", "max_lon"])
    else:
        df = serving.merge(grid_bounds, on="grid_id", how="left").dropna(subset=["min_lat", "max_lat", "min_lon", "max_lon"])

    fig, ax = plt.subplots(figsize=(7, 9))
    for row in df.itertuples(index=False):
        val = float(getattr(row, value_col))
        ax.add_patch(
            MplRectangle(
                (row.min_lon, row.min_lat), row.max_lon - row.min_lon, row.max_lat - row.min_lat,
                facecolor=_color_for(val), edgecolor="none",
            )
        )
    ax.set_xlim(df["min_lon"].min(), df["max_lon"].max())
    ax.set_ylim(df["min_lat"].min(), df["max_lat"].max())
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c, _ in RSRP_BINS]
    labels = [lbl for _, _, _, lbl in RSRP_BINS]
    ax.legend(handles, labels, loc="lower left", fontsize=8, title="RSRP (dBm)", framealpha=0.9)
    fig.tight_layout()
    return fig


def _render_map(serving: pd.DataFrame, grid_bounds: pd.DataFrame, value_col: str, title: str, view_mode: str, height: int = 460) -> None:
    if view_mode == "Static image":
        fig = _build_static_image(serving, grid_bounds, value_col, title)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        components.html(_build_grid_map(serving, grid_bounds, value_col, title)._repr_html_(), height=height, scrolling=False)


def _cdf_trace(values: pd.Series, name: str, color: str) -> go.Scatter:
    arr = pd.to_numeric(values, errors="coerce").dropna().sort_values().to_numpy()
    if len(arr) == 0:
        return go.Scatter(x=[], y=[], mode="lines", name=name)
    y = (pd.Series(range(1, len(arr) + 1)) / len(arr) * 100.0).to_numpy()
    return go.Scatter(x=arr, y=y, mode="lines", name=f"{name} (n={len(arr):,})", line=dict(color=color, width=2.5))


def render() -> None:
    st.title("Project 210 Taiwan - Phase 17: Full-Polygon Geo + DT-Residual vs. Production Offset (Phase 9)")

    serving_4g = load_serving("4G")
    serving_5g = load_serving("5G")
    dt = load_dt()
    grid_bounds = load_grid_bounds()
    if serving_4g.empty or serving_5g.empty:
        st.error(
            f"Phase 17 output not found under {OUT_DIR}. Run "
            "test_project210_phase17_full_polygon_geo_dt_comparison.py first."
        )
        return

    with st.sidebar:
        view_mode = st.radio(
            "Map view", ["Interactive (folium)", "Static image"], index=0, key="phase17_view_mode",
            help="Interactive is a real Leaflet map (zoom/pan/hover); Static renders a flat PNG (no re-tiling noise on zoom, easy to paste into a report).",
        )
        st.subheader("Clutter weights used (same as Phase 15/16 - reused from ML/tests/baseline)")
        st.caption("Read-only here: this run used these fixed values across the whole polygon. Adjusting them requires re-running the batch script, not a live recompute, because this covers 10,234 grid cells vs. one site.")
        for cls, w in phase15.DEFAULT_CLUTTER_WEIGHTS.items():
            st.text(f"{cls}: {w:+.1f} dB")
        st.text(f"Building footprint weight: {phase15.DEFAULT_BUILDING_AREA_WEIGHT:+.1f} dB (x building_area_ratio)")
        st.text("Building entry (wall): -15.0 dB")
        st.text("Indoor depth slope: -0.5 dB/m")
        st.text("5G n78 technology offset: -2.58 dB")

    mean_shift_4g = float((serving_4g["phase17_rsrp"] - serving_4g["corrected_rsrp"]).mean())
    mean_shift_5g = float((serving_5g["phase17_rsrp"] - serving_5g["corrected_rsrp"]).mean())
    st.warning(
        f"**Known over-correction, not yet fixed:** mean Phase17 vs Phase9 shift is "
        f"{mean_shift_4g:+.1f}dB (4G) / {mean_shift_5g:+.1f}dB (5G) across the WHOLE polygon - "
        f"25% of 4G grid points hit the -145dBm floor. Summing every real diffraction "
        "obstacle (instead of only the worst one) fixed under-counting on a single site, "
        "but naive summation over-penalizes multi-building paths at full-polygon scale - "
        "this needs a diminishing-weight or Deygout-style correction before the numbers "
        "below should be trusted as calibration-ready, not just a bug-free run."
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("4G grid cells", f"{len(serving_4g):,}")
    metric_cols[1].metric("5G grid cells", f"{len(serving_5g):,}")
    metric_cols[2].metric("4G mean shift (Phase17 - Phase9)", f"{mean_shift_4g:+.1f} dB")
    metric_cols[3].metric("5G mean shift (Phase17 - Phase9)", f"{mean_shift_5g:+.1f} dB")

    st.header("Frontend (mean of candidates) on top, then Serving cell (best server) below - Phase 9 vs. Phase 17, both aggregations")
    st.caption(
        "'Frontend' reproduces the production map's current default "
        "(lteGridAggregationMethod = \"mean\" in UnifiedMapView.jsx) - averaging every "
        "candidate cell assigned to a grid box. Shown for BOTH Phase 9 (old baseline) "
        "and Phase 17 (new geo+DT values), so you can see the mean-aggregation effect "
        "on each. Below that, 'Serving cell' is the single best-server value (correct "
        "aggregation) for Phase 9 vs. Phase 17 - the real methodology comparison."
    )
    for tech, serving in [("4G", serving_4g), ("5G", serving_5g)]:
        gap9 = float((serving["frontend_mean_rsrp"] - serving["corrected_rsrp"]).mean())
        gap17 = float((serving["phase17_frontend_mean_rsrp"] - serving["phase17_rsrp"]).mean())
        avg_candidates = float(serving["frontend_candidate_count"].mean())
        st.subheader(f"{tech} - mean candidates per grid cell: {avg_candidates:.1f}")

        frontend_col_a, frontend_col_b = st.columns(2)
        with frontend_col_a:
            st.caption(f"Frontend (Phase 9 mean, current production default) - gap vs. serving cell: {gap9:+.1f} dB")
            _render_map(serving, grid_bounds, "frontend_mean_rsrp", "Frontend (Phase 9 mean)", view_mode)
        with frontend_col_b:
            st.caption(f"Frontend (Phase 17 mean, same mean-of-candidates aggregation, new geo+DT values) - gap vs. serving cell: {gap17:+.1f} dB")
            _render_map(serving, grid_bounds, "phase17_frontend_mean_rsrp", "Frontend (Phase 17 mean)", view_mode)

        map_col_a, map_col_b = st.columns(2)
        with map_col_a:
            st.caption("Serving cell (Phase 9 - best server, production offset baseline)")
            _render_map(serving, grid_bounds, "corrected_rsrp", "Phase 9", view_mode)
        with map_col_b:
            st.caption("Phase 17 (geo correction + representative-DT residual)")
            _render_map(serving, grid_bounds, "phase17_rsrp", "Phase 17", view_mode)

    st.subheader("CDF: DT measured vs. Phase 9 predicted vs. Phase 17 predicted (at each DT point's nearest grid)")
    for tech, serving in [("4G", serving_4g), ("5G", serving_5g)]:
        dt_tech = dt[dt["assigned_technology"] == tech].copy()
        joined = dt_tech.merge(
            serving[["grid_id", "corrected_rsrp", "phase17_rsrp"]],
            left_on="nearest_grid_id", right_on="grid_id", how="inner",
        )
        fig = go.Figure()
        fig.add_trace(_cdf_trace(joined["rsrp_measured"], "DT measured", "#2563eb"))
        fig.add_trace(_cdf_trace(joined["corrected_rsrp"], "Phase 9 predicted", "#ef4444"))
        fig.add_trace(_cdf_trace(joined["phase17_rsrp"], "Phase 17 predicted", "#16a34a"))
        fig.update_layout(
            title=f"{tech} - DT vs Phase 9 vs Phase 17 (n_dt_joined={len(joined):,})",
            height=420, xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
            yaxis_range=[0, 100], xaxis_range=[-147, -45],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Serving cell detail (per grid cell, not an average)")
    tech_choice = st.selectbox("Technology", ["4G", "5G"], index=0, key="phase17_serving_table_tech")
    serving_show = serving_4g if tech_choice == "4G" else serving_5g
    st.dataframe(
        serving_show[
            ["grid_id", "strict_cell_key", "site", "sector", "band", "technology",
             "corrected_rsrp", "phase17_rsrp", "geo_correction_db", "dt_residual_db", "dt_replaced"]
        ].sort_values("grid_id").head(500),
        use_container_width=True, height=350,
    )


def main() -> None:
    st.set_page_config(page_title="Project 210 Phase 17", layout="wide")
    render()


if __name__ == "__main__":
    main()
