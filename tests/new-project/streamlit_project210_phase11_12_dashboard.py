from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "data" / "project_210_taiwan" / "cost231_phase11_12_residual_blending"
IMAGE_DIR = OUT_DIR / "images"
HTML_DIR = OUT_DIR / "html"


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    path = OUT_DIR / "phase11_12_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_serving() -> pd.DataFrame:
    path = OUT_DIR / "phase11_12_serving_grid_by_technology_project210.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def cdf_trace(values: pd.Series, name: str) -> go.Scatter:
    arr = pd.to_numeric(values, errors="coerce").dropna().sort_values().to_numpy()
    if len(arr) == 0:
        return go.Scatter(x=[], y=[], mode="lines", name=name)
    y = (pd.Series(range(1, len(arr) + 1)) / len(arr) * 100.0).to_numpy()
    return go.Scatter(x=arr, y=y, mode="lines", name=f"{name} (n={len(arr):,})")


def render() -> None:
    st.title("Project 210 Taiwan - Phase 11 / Phase 12 Residual Blending")

    summary = load_summary()
    serving = load_serving()
    if serving.empty:
        st.error(f"Phase 11/12 output not found under {OUT_DIR}")
        st.stop()

    with st.sidebar:
        technology = st.selectbox("Technology", ["4G", "5G"])
        view = st.radio("View", ["Interactive map", "Static comparison image", "CDF image", "Plotly CDF", "Output files"])

    tech_df = serving[serving["technology"].astype(str) == technology].copy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grid pixels", f"{tech_df['grid_id'].nunique():,}")
    c2.metric("Phase 9 mean", f"{tech_df['corrected_rsrp'].mean():.2f} dBm")
    c3.metric("Phase 11 mean", f"{tech_df['phase11_rsrp'].mean():.2f} dBm")
    c4.metric("Phase 12 mean", f"{tech_df['phase12_rsrp'].mean():.2f} dBm")

    if view == "Interactive map":
        html_path = HTML_DIR / f"project210_{technology.lower()}_phase11_phase12_interactive_map.html"
        st.caption(str(html_path))
        if not html_path.exists():
            st.error("Interactive HTML map not found.")
        else:
            html = html_path.read_text(encoding="utf-8")
            components.html(html, height=820, scrolling=True)

    elif view == "Static comparison image":
        image_path = IMAGE_DIR / f"project210_{technology.lower()}_phase11_phase12_map_comparison.png"
        st.caption(str(image_path))
        st.image(str(image_path), use_container_width=True)

    elif view == "CDF image":
        image_path = IMAGE_DIR / f"project210_{technology.lower()}_phase11_phase12_cdf_comparison.png"
        st.caption(str(image_path))
        st.image(str(image_path), use_container_width=True)

    elif view == "Plotly CDF":
        fig = go.Figure()
        fig.add_trace(cdf_trace(tech_df["corrected_rsrp"], "Phase 9 offset baseline"))
        fig.add_trace(cdf_trace(tech_df["phase11_rsrp"], "Phase 11 residual blending"))
        fig.add_trace(cdf_trace(tech_df["phase12_rsrp"], "Phase 12 clutter-aware residual"))
        fig.update_layout(
            height=650,
            xaxis_title="RSRP (dBm)",
            yaxis_title="Cumulative percentage (%)",
            yaxis_range=[0, 100],
            xaxis_range=[-147, -45],
            legend_title="Surface",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("Generated Outputs")
        rows = []
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "file": path.name,
                        "folder": str(path.parent),
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=500)
        st.json(summary)

    st.subheader("Sample Grid Rows")
    st.dataframe(
        tech_df[
            [
                "grid_id",
                "technology",
                "site",
                "sector",
                "band",
                "corrected_rsrp",
                "phase11_rsrp",
                "phase12_rsrp",
                "clutter_class",
                "clutter_height_m",
                "dt_replaced",
            ]
        ].head(300),
        use_container_width=True,
        height=320,
    )


def main() -> None:
    st.set_page_config(page_title="Project 210 Phase 11/12", layout="wide")
    render()


if __name__ == "__main__":
    main()
