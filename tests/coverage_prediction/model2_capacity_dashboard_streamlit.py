from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium


ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

MODEL_ROOT = ML_ROOT / "models" / "model2"
TARGETS = {
    "model2a_demand": "demand_index",
    "model2b_users": "active_users_est",
    "model2c_traffic": "traffic_demand_est",
}


@st.cache_data(show_spinner=False)
def load_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(path_text: str) -> dict:
    path = Path(path_text)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def metric_cards(metrics: dict) -> None:
    cols = st.columns(3)
    for col, name in zip(cols, ["mae", "rmse", "r2"]):
        col.metric(name.upper(), fmt(metrics.get(name)))


def risk_color(score: float, min_score: float, max_score: float) -> str:
    if pd.isna(score):
        return "#6b7280"
    if max_score <= min_score:
        max_score = min_score + 1.0
    ratio = max(0.0, min(1.0, (float(score) - min_score) / (max_score - min_score)))
    if ratio >= 0.80:
        return "#b91c1c"
    if ratio >= 0.50:
        return "#f97316"
    return "#16a34a"


def risk_map(df: pd.DataFrame, value_col: str, title: str, key: str) -> None:
    required = {"grid_centroid_lat", "grid_centroid_lon", value_col}
    if df.empty or not required.issubset(df.columns):
        st.info("No map-ready grid prediction rows found.")
        return

    map_df = df.dropna(subset=["grid_centroid_lat", "grid_centroid_lon", value_col]).copy()
    if map_df.empty:
        st.info("No non-null rows for this map.")
        return

    center_lat = float(map_df["grid_centroid_lat"].median())
    center_lon = float(map_df["grid_centroid_lon"].median())
    min_score = float(map_df[value_col].min())
    max_score = float(map_df[value_col].max())
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
        width="100%",
        height=540,
    )

    for _, row in map_df.iterrows():
        color = risk_color(row[value_col], min_score, max_score)
        tooltip = (
            f"Grid: {row.get('grid_id', 'N/A')}<br>"
            f"Risk: {fmt(row.get('capacity_risk_score'))}<br>"
            f"Demand: {fmt(row.get('demand_index_pred'))}<br>"
            f"Users: {fmt(row.get('active_users_est_pred'))}<br>"
            f"Traffic: {fmt(row.get('traffic_demand_est_pred'))}<br>"
            f"Cell: {row.get('dominant_pci', 'N/A')}"
        )
        folium.CircleMarker(
            location=[float(row["grid_centroid_lat"]), float(row["grid_centroid_lon"])],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.82,
            opacity=0.9,
            weight=1,
            tooltip=tooltip,
        ).add_to(fmap)

    st.caption(title)
    st_folium(fmap, width=None, height=560, key=key)


def shap_block() -> None:
    st.subheader("SHAP Feature Importance")
    target_key = st.selectbox("Target", list(TARGETS.keys()), format_func=lambda k: f"{k} / {TARGETS[k]}")
    shap_df = load_csv(str(MODEL_ROOT / target_key / "shap_importance.csv"))
    image_path = MODEL_ROOT / target_key / "shap_summary.png"
    if shap_df.empty:
        st.info("SHAP importance CSV is missing.")
        return
    cols = st.columns([1, 1])
    with cols[0]:
        fig = px.bar(shap_df.head(20), x="mean_abs_shap", y="feature", orientation="h", title="Top SHAP Features")
        fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with cols[1]:
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.dataframe(shap_df.head(25), use_container_width=True, hide_index=True)


def temporal_performance_block() -> None:
    st.subheader("Temporal Prediction Performance")
    rows = []
    for target_key, target in TARGETS.items():
        metrics = load_json(str(MODEL_ROOT / target_key / "metrics.json"))
        for split_name in ["TRAIN", "VALID", "TEST"]:
            split = metrics.get(split_name, {})
            if split:
                rows.append({"model": target_key, "target": target, "split": split_name, **split})
    perf_df = pd.DataFrame(rows)
    if perf_df.empty:
        st.info("Metrics are missing.")
        return
    fig = px.bar(perf_df, x="target", y="rmse", color="split", barmode="group", title="RMSE by Temporal Split")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(perf_df.round(4), use_container_width=True, hide_index=True)


def top_tables_block() -> None:
    st.subheader("Congestion Leaderboards")
    tabs = st.tabs(["Cells", "Sectors", "Sites"])
    files = ["top_congested_cells.csv", "top_congested_sectors.csv", "top_congested_sites.csv"]
    for tab, filename in zip(tabs, files):
        with tab:
            df = load_csv(str(MODEL_ROOT / filename))
            if df.empty:
                st.info(f"{filename} is missing.")
            else:
                st.dataframe(df.head(20).round(3), use_container_width=True, hide_index=True)


def drilldown_block(grid_df: pd.DataFrame) -> None:
    st.subheader("Grid -> Cell -> Site Drill-Down")
    if grid_df.empty:
        st.info("Top grid output is missing.")
        return
    cell_col = "dominant_pci" if "dominant_pci" in grid_df.columns else None
    if cell_col:
        cells = grid_df[cell_col].dropna().astype(str).unique().tolist()
        selected = st.selectbox("Cell / sector / site key", cells)
        view = grid_df[grid_df[cell_col].astype(str) == selected].copy()
    else:
        view = grid_df.copy()
    st.dataframe(view.head(100).round(3), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Model 2 Capacity Dashboard", layout="wide")
    st.title("Model 2 Capacity / Congestion / Demand Dashboard")

    summary = load_json(str(MODEL_ROOT / "model2_training_summary.json"))
    grid_df = load_csv(str(MODEL_ROOT / "top_congested_grids.csv"))
    if summary:
        st.caption(summary.get("business_outputs", {}).get("aggregation_key_note", "Model 2 grid-level outputs."))

    st.subheader("Demand Risk Map")
    risk_map(grid_df, "demand_index_pred" if "demand_index_pred" in grid_df.columns else "capacity_risk_score", "Predicted demand pressure by grid", "demand_risk_map")

    st.subheader("Capacity Risk Map")
    risk_map(grid_df, "capacity_risk_score", "Combined capacity risk by grid", "capacity_risk_map")

    top_tables_block()
    shap_block()
    temporal_performance_block()
    drilldown_block(grid_df)


if __name__ == "__main__":
    main()
