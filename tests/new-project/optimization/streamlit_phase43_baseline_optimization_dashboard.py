from __future__ import annotations

import json
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
from matplotlib.patches import Rectangle


ML_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ML_ROOT / "tests" / "output"

RUNS = {
    "Project 210 Taiwan": {
        "production": OUTPUT_ROOT / "baseline_210_profile_20260907_165242",
        "phase43": OUTPUT_ROOT / "baseline_210_profile_20260907_174241",
        "phase43_v2": OUTPUT_ROOT / "baseline_210_profile_20260907_183149",
    },
    "Project 193 India": {
        "production": OUTPUT_ROOT / "baseline_193_profile_20260907_170610",
        "phase43": OUTPUT_ROOT / "baseline_193_profile_20260907_174519",
        "phase43_v2": OUTPUT_ROOT / "baseline_193_profile_20260907_183325",
        "phase43_v3": OUTPUT_ROOT / "baseline_193_profile_20260907_190114",
    },
}

RSRP_BINS = [
    (-140, -115, "#991b1b", "-140 to -115"),
    (-115, -105, "#d97706", "-115 to -105"),
    (-105, -95, "#fef08a", "-105 to -95"),
    (-95, -85, "#22c55e", "-95 to -85"),
    (-85, 0, "#15803d", "-85 to 0"),
]

SINR_BINS = [
    (-60, -20, "#991b1b", "-60 to -20"),
    (-20, 0, "#dc2626", "-20 to 0"),
    (0, 10, "#facc15", "0 to 10"),
    (10, 20, "#22c55e", "10 to 20"),
    (20, 45, "#15803d", "20 to 45"),
]

RSRQ_BINS = [
    (-50, -20, "#991b1b", "-50 to -20"),
    (-20, -16, "#dc2626", "-20 to -16"),
    (-16, -13, "#facc15", "-16 to -13"),
    (-13, -10, "#22c55e", "-13 to -10"),
    (-10, 0, "#15803d", "-10 to 0"),
]


@st.cache_data(show_spinner=False)
def load_summary(path: str) -> dict:
    summary_path = Path(path) / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}


@st.cache_data(show_spinner=False)
def load_predictions(path: str) -> pd.DataFrame:
    parquet_path = Path(path) / "baseline_predictions.parquet"
    if not parquet_path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(parquet_path)
    if "technology" not in frame.columns and "Technology" in frame.columns:
        frame["technology"] = frame["Technology"]
    if "operator" not in frame.columns:
        frame["operator"] = "unknown"
    if "center_lat" not in frame.columns and "lat" in frame.columns:
        frame["center_lat"] = frame["lat"]
    if "center_lon" not in frame.columns and "lon" in frame.columns:
        frame["center_lon"] = frame["lon"]
    return frame


def color_for(value: float, bins: list[tuple[float, float, str, str]]) -> str:
    if not np.isfinite(value):
        return "#9ca3af"
    for lo, hi, color, _label in bins:
        if lo <= value < hi:
            return color
    return "#9ca3af"


def metric_bins(metric: str):
    if metric == "pred_sinr":
        return SINR_BINS
    if metric == "pred_rsrq":
        return RSRQ_BINS
    return RSRP_BINS


def filtered(frame: pd.DataFrame, tech: str, operator: str) -> pd.DataFrame:
    out = frame.copy()
    if tech != "All":
        out = out[out["technology"].astype(str).eq(tech)]
    if operator != "All":
        out = out[out["operator"].astype(str).eq(operator)]
    return out


def cdf_trace(values: pd.Series, name: str, color: str) -> go.Scatter:
    arr = pd.to_numeric(values, errors="coerce").dropna().sort_values().to_numpy()
    if len(arr) == 0:
        return go.Scatter(x=[], y=[], mode="lines", name=name)
    y = np.arange(1, len(arr) + 1, dtype=float) / len(arr) * 100.0
    return go.Scatter(x=arr, y=y, mode="lines", name=f"{name} (n={len(arr):,})", line=dict(color=color, width=2.5))


def map_frame(frame: pd.DataFrame, metric: str, title: str, view_mode: str) -> None:
    needed_bounds = {"min_lat", "max_lat", "min_lon", "max_lon"}
    bins = metric_bins(metric)
    df = frame.dropna(subset=[metric]).copy()
    if df.empty:
        st.warning("No rows for this filter.")
        return
    if view_mode == "Static":
        fig, ax = plt.subplots(figsize=(7, 8))
        if needed_bounds.issubset(df.columns):
            plot_df = df.dropna(subset=list(needed_bounds))
            for row in plot_df.itertuples(index=False):
                value = float(getattr(row, metric))
                ax.add_patch(
                    Rectangle(
                        (row.min_lon, row.min_lat),
                        row.max_lon - row.min_lon,
                        row.max_lat - row.min_lat,
                        facecolor=color_for(value, bins),
                        edgecolor="none",
                    )
                )
            ax.set_xlim(plot_df["min_lon"].min(), plot_df["max_lon"].max())
            ax.set_ylim(plot_df["min_lat"].min(), plot_df["max_lat"].max())
        else:
            ax.scatter(df["center_lon"], df["center_lat"], c=[color_for(float(v), bins) for v in df[metric]], s=4)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color, _ in bins]
        labels = [label for _, _, _, label in bins]
        ax.legend(handles, labels, loc="lower left", fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    fmap = folium.Map(
        location=[float(df["center_lat"].mean()), float(df["center_lon"].mean())],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )
    layer = folium.FeatureGroup(name=title, show=True)
    for row in df.itertuples(index=False):
        value = float(getattr(row, metric))
        popup = (
            f"<b>Grid:</b> {getattr(row, 'grid_id', '')}<br>"
            f"<b>Cell:</b> {getattr(row, 'strict_cell_key', '')}<br>"
            f"<b>Operator:</b> {getattr(row, 'operator', '')}<br>"
            f"<b>Technology:</b> {getattr(row, 'technology', '')}<br>"
            f"<b>{metric}:</b> {value:.2f}"
        )
        if needed_bounds.issubset(df.columns) and all(np.isfinite([row.min_lat, row.max_lat, row.min_lon, row.max_lon])):
            folium.Rectangle(
                bounds=[[row.min_lat, row.min_lon], [row.max_lat, row.max_lon]],
                color=color_for(value, bins),
                weight=0,
                fill=True,
                fill_color=color_for(value, bins),
                fill_opacity=0.82,
                tooltip=f"{value:.1f}",
                popup=folium.Popup(popup, max_width=320),
            ).add_to(layer)
        else:
            folium.CircleMarker(
                location=[row.center_lat, row.center_lon],
                radius=2,
                color=color_for(value, bins),
                fill=True,
                fill_color=color_for(value, bins),
                fill_opacity=0.85,
                tooltip=f"{value:.1f}",
                popup=folium.Popup(popup, max_width=320),
            ).add_to(layer)
    layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    components.html(fmap._repr_html_(), height=620, scrolling=False)


def scorer_seconds(timings: list[dict]) -> float:
    return sum(float(row.get("wall_s", 0.0)) for row in timings if row.get("stage") == "services.score_candidates")


@st.cache_data(show_spinner=False)
def load_timings(path: str) -> list[dict]:
    timings_path = Path(path) / "timings.json"
    return json.loads(timings_path.read_text(encoding="utf-8")) if timings_path.exists() else []


def render() -> None:
    st.title("Phase 43 Baseline Optimization")

    with st.sidebar:
        project = st.selectbox("Project", list(RUNS), index=0)
        available_runs = [key for key in ["phase43", "phase43_v2", "phase43_v3"] if key in RUNS[project]]
        optimized_run = st.selectbox("Optimized run", available_runs, index=len(available_runs) - 1)
        metric = st.selectbox("Metric", ["pred_rsrp", "pred_rsrq", "pred_sinr"], index=0)
        view_mode = st.radio("Map view", ["Static", "Interactive"], index=0)

    prod_dir = RUNS[project]["production"]
    phase43_dir = RUNS[project][optimized_run]
    prod = load_predictions(str(prod_dir))
    opt = load_predictions(str(phase43_dir))
    prod_summary = load_summary(str(prod_dir))
    opt_summary = load_summary(str(phase43_dir))
    prod_timings = load_timings(str(prod_dir))
    opt_timings = load_timings(str(phase43_dir))

    if prod.empty or opt.empty:
        st.error("Missing production or Phase 43 output parquet.")
        return

    techs = ["All"] + sorted(set(prod["technology"].dropna().astype(str)) | set(opt["technology"].dropna().astype(str)))
    operators = ["All"] + sorted(set(prod["operator"].dropna().astype(str)) | set(opt["operator"].dropna().astype(str)))
    tech = st.sidebar.radio("Technology", techs, index=0, horizontal=False)
    operator = st.sidebar.selectbox("Operator", operators, index=0)

    prod_f = filtered(prod, tech, operator)
    opt_f = filtered(opt, tech, operator)

    cols = st.columns(5)
    cols[0].metric("Production wall", f"{prod_summary.get('wall_s', 0.0):.1f}s")
    cols[1].metric(f"{optimized_run} wall", f"{opt_summary.get('wall_s', 0.0):.1f}s")
    cols[2].metric("Production scorer", f"{scorer_seconds(prod_timings):.1f}s")
    cols[3].metric(f"{optimized_run} scorer", f"{scorer_seconds(opt_timings):.1f}s")
    cols[4].metric("Rows", f"{len(opt):,}")

    if len(prod_f) and len(opt_f):
        aligned = prod_f.sort_values(["technology", "operator", "grid_id", "strict_cell_key"]).reset_index(drop=True)
        aligned_opt = opt_f.sort_values(["technology", "operator", "grid_id", "strict_cell_key"]).reset_index(drop=True)
        if len(aligned) == len(aligned_opt):
            diff = (pd.to_numeric(aligned[metric], errors="coerce") - pd.to_numeric(aligned_opt[metric], errors="coerce")).abs()
            st.metric(f"Max abs diff for {metric}", f"{float(diff.max()):.6f}")

    st.subheader("CDF")
    fig = go.Figure()
    fig.add_trace(cdf_trace(prod_f[metric], "Production", "#ef4444"))
    fig.add_trace(cdf_trace(opt_f[metric], optimized_run, "#22c55e"))
    fig.update_layout(height=420, xaxis_title=metric, yaxis_title="Cumulative %", yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Maps")
    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption("Production baseline")
        map_frame(prod_f, metric, f"{project} production {metric}", view_mode)
    with map_cols[1]:
        st.caption(f"{optimized_run} optimized baseline")
        map_frame(opt_f, metric, f"{project} {optimized_run} {metric}", view_mode)

    st.subheader("Operator / Technology Counts")
    count_cols = st.columns(2)
    with count_cols[0]:
        st.dataframe(
            prod.groupby(["technology", "operator"], dropna=False).size().reset_index(name="production_rows"),
            use_container_width=True,
            hide_index=True,
        )
    with count_cols[1]:
        st.dataframe(
            opt.groupby(["technology", "operator"], dropna=False).size().reset_index(name=f"{optimized_run}_rows"),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(page_title="Phase 43 Baseline Optimization", layout="wide")
    render()


if __name__ == "__main__":
    main()



