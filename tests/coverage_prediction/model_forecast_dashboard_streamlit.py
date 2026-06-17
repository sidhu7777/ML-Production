from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import folium
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_DIR = Path("data")
MODEL2_CSV = DATA_DIR / "model2_capacity_training.csv"
MODEL2_SUMMARY_JSON = DATA_DIR / "model2_capacity_training.summary.json"
COVERAGE_ARCHIVE = DATA_DIR / "coverage_20260521_104406.7z"
BUCKET_ORDER = ["PART_1", "PART_2", "PART_3"]
MAP_VALUE_RANGES = {
    "rsrp_mean": (-125.0, -55.0),
    "rsrq_mean": (-20.0, -3.0),
    "sinr_mean": (-10.0, 30.0),
    "cqi_mean": (0.0, 15.0),
    "demand_index": (0.0, 100.0),
    "traffic_demand_est": (0.0, 80.0),
    "active_users_est": (0.0, 50.0),
}


def _archive_root(archive_path: Path) -> str:
    listed = subprocess.check_output(["tar", "-tf", str(archive_path)], text=True)
    first_file = next((line for line in listed.splitlines() if "/" in line and not line.endswith("/")), "")
    if not first_file:
        raise RuntimeError(f"No files found in archive: {archive_path}")
    return first_file.split("/", 1)[0]


@st.cache_data(show_spinner=False)
def load_coverage_summary(archive_path_text: str) -> dict:
    archive_path = Path(archive_path_text)
    root = _archive_root(archive_path)
    raw = subprocess.check_output(["tar", "-xOf", str(archive_path), f"{root}/summary.json"], text=True)
    return json.loads(raw)


@st.cache_data(show_spinner=False)
def load_dataset(csv_path_text: str, summary_path_text: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(Path(csv_path_text))
    summary_path = Path(summary_path_text)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    numeric_cols = [
        "grid_id",
        "grid_centroid_lat",
        "grid_centroid_lon",
        "sample_count",
        "rsrp_mean",
        "rsrq_mean",
        "sinr_mean",
        "cqi_mean",
        "dl_tpt_mean",
        "estimated_prb_mean",
        "prb_pressure_est",
        "prb_outlier_flag",
        "growth_rate",
        "demand_index",
        "active_users_est",
        "traffic_demand_est",
        "geo_demand_score",
        "kpi_demand_score",
        "building_count",
        "building_area_ratio",
        "road_density",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, summary


def fmt(value, suffix: str = "", digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}{suffix}"
    return f"{number:,.{digits}f}{suffix}"


def section(title: str, help_text: str | None = None) -> None:
    st.markdown(f"### {title}")
    if help_text:
        st.caption(help_text)


def metric_row(items: list[tuple[str, object, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, suffix) in zip(cols, items):
        col.metric(label, fmt(value, suffix=suffix))


def date_range_label(coverage_summary: dict) -> str:
    ranges = coverage_summary.get("bucket_ranges") or []
    if not ranges:
        return "N/A"
    start = str(ranges[0].get("start", ""))[:7]
    end = str(ranges[-1].get("end", ""))[:7]
    return f"{start} -> {end}"


def bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("time_bucket", as_index=False)
        .agg(
            rows=("grid_id", "count"),
            grids=("grid_id", "nunique"),
            samples=("sample_count", "sum"),
            avg_rsrp=("rsrp_mean", "mean"),
            avg_rsrq=("rsrq_mean", "mean"),
            avg_sinr=("sinr_mean", "mean"),
            avg_cqi=("cqi_mean", "mean"),
            avg_dl_tpt=("dl_tpt_mean", "mean"),
            avg_prb=("estimated_prb_mean", "mean"),
            avg_prb_pressure=("prb_pressure_est", "mean"),
            prb_outliers=("prb_outlier_flag", "sum"),
            avg_demand=("demand_index", "mean"),
            avg_users=("active_users_est", "mean"),
            avg_traffic=("traffic_demand_est", "mean"),
            total_traffic=("traffic_demand_est", "sum"),
            avg_growth=("growth_rate", "mean"),
        )
    )
    out["time_bucket"] = pd.Categorical(out["time_bucket"], BUCKET_ORDER, ordered=True)
    return out.sort_values("time_bucket").reset_index(drop=True)


def clutter_distribution(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("clutter_class", dropna=False, as_index=False)
        .agg(rows=("grid_id", "count"), grids=("grid_id", "nunique"))
        .sort_values("rows", ascending=False)
    )
    out["pct_rows"] = (out["rows"] / max(out["rows"].sum(), 1) * 100).round(2)
    out["clutter_class"] = out["clutter_class"].fillna("Unknown")
    return out


def trend_chart(df: pd.DataFrame, value_cols: list[str], title: str, chart_key: str) -> None:
    bucket_df = bucket_summary(df)
    chart_df = bucket_df.melt(id_vars=["time_bucket"], value_vars=value_cols, var_name="metric", value_name="value")
    fig = px.line(chart_df, x="time_bucket", y="value", color="metric", markers=True, title=title)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def distribution_block(df: pd.DataFrame, chart_key: str) -> None:
    dist = clutter_distribution(df)
    cols = st.columns([1, 1])
    with cols[0]:
        st.dataframe(dist.round(2), use_container_width=True, hide_index=True)
    with cols[1]:
        fig = px.bar(dist, x="clutter_class", y="pct_rows", title="Cluster / Clutter Share")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10), xaxis_title="", yaxis_title="% rows")
        st.plotly_chart(fig, use_container_width=True, key=chart_key)


def bucket_distribution_block(df: pd.DataFrame, chart_key: str) -> None:
    bucket_df = bucket_summary(df)[["time_bucket", "rows", "grids", "samples"]]
    cols = st.columns([1, 1])
    with cols[0]:
        st.dataframe(bucket_df, use_container_width=True, hide_index=True)
    with cols[1]:
        fig = px.bar(bucket_df, x="time_bucket", y="rows", title="Rows by Bucket")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10), xaxis_title="", yaxis_title="Rows")
        st.plotly_chart(fig, use_container_width=True, key=chart_key)


def _color_for_value(value: float, color_col: str, observed_min: float, observed_max: float) -> str:
    if pd.isna(value):
        return "#6b7280"
    low, high = MAP_VALUE_RANGES.get(color_col, (observed_min, observed_max))
    if not pd.notna(low):
        low = observed_min
    if not pd.notna(high):
        high = observed_max
    if high <= low:
        high = low + 1.0
    clipped = min(max(float(value), float(low)), float(high))
    scale_position = (clipped - low) / (high - low)
    return pc.sample_colorscale("Turbo", [scale_position])[0]


def map_block(df: pd.DataFrame, color_col: str, title: str, chart_key: str) -> None:
    map_df = (
        df.dropna(subset=["grid_centroid_lat", "grid_centroid_lon"])
        .groupby("grid_id", as_index=False)
        .agg(
            lat=("grid_centroid_lat", "mean"),
            lon=("grid_centroid_lon", "mean"),
            clutter=("clutter_class", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
            samples=("sample_count", "sum"),
            value=(color_col, "mean"),
        )
        .dropna(subset=["value"])
    )
    if map_df.empty:
        st.info("No map-ready rows after filters.")
        return
    center_lat = float(map_df["lat"].median())
    center_lon = float(map_df["lon"].median())
    observed_min = float(map_df["value"].min())
    observed_max = float(map_df["value"].max())

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
        width="100%",
        height=520,
    )

    for _, row in map_df.iterrows():
        color = _color_for_value(float(row["value"]), color_col, observed_min, observed_max)
        tooltip = (
            f"Grid: {int(row['grid_id'])}<br>"
            f"Cluster: {row['clutter']}<br>"
            f"Samples: {int(row['samples'])}<br>"
            f"{color_col}: {float(row['value']):.3f}"
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
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
    st_folium(fmap, width=None, height=560, key=chart_key)


def model1_dataset_page(df: pd.DataFrame, coverage_summary: dict) -> None:
    st.subheader("Model 1 Dataset Validation: Coverage Forecast")
    st.caption("Purpose: decide whether the coverage dataset is trustworthy before training.")

    section("1. Dataset Summary")
    metric_row(
        [
            ("Total Grids", df["grid_id"].nunique(), ""),
            ("Total Buckets", df["time_bucket"].nunique(), ""),
            ("Total Rows", len(df), ""),
            ("Date Range", date_range_label(coverage_summary), ""),
        ]
    )

    section("2. Bucket Distribution", "Each bucket should contain the same grid surface. Samples can differ, but grid-bucket rows should not be missing.")
    bucket_distribution_block(df, "model1_bucket_distribution")

    section("3. Geo Distribution", "Checks whether the coverage grid contains sensible Dense Urban / Urban / Suburban / Open style distribution.")
    distribution_block(df, "model1_geo_distribution")

    section("4. KPI Trend by Bucket", "Coverage training needs visible KPI movement across PART_1 -> PART_2 -> PART_3, not a flat or broken surface.")
    trend_chart(df, ["avg_rsrp", "avg_sinr", "avg_cqi", "avg_prb_pressure", "avg_dl_tpt"], "Model 1 KPI Trends", "model1_kpi_trends")

    section("5. Coverage Feature Validation by Bucket")
    st.dataframe(
        bucket_summary(df)[
            ["time_bucket", "avg_rsrp", "avg_rsrq", "avg_sinr", "avg_cqi", "avg_prb", "avg_dl_tpt"]
        ].round(3),
        use_container_width=True,
        hide_index=True,
    )

    section("6. Cluster vs Coverage Validation", "Dense and urban zones can have lower SINR/RSRQ; the pattern should look explainable, not random.")
    cluster_kpi = (
        df.groupby("clutter_class", as_index=False)
        .agg(
            grids=("grid_id", "nunique"),
            avg_rsrp=("rsrp_mean", "mean"),
            avg_sinr=("sinr_mean", "mean"),
            avg_cqi=("cqi_mean", "mean"),
            avg_prb_pressure=("prb_pressure_est", "mean"),
            prb_outliers=("prb_outlier_flag", "sum"),
        )
        .sort_values("avg_sinr")
    )
    st.dataframe(cluster_kpi.round(3), use_container_width=True, hide_index=True)

    section("7. Top 20 Weak Coverage Grids", "Quick sanity check for bad coverage locations before training Model 1.")
    weak_cols = ["time_bucket", "grid_id", "clutter_class", "sample_count", "rsrp_mean", "rsrq_mean", "sinr_mean", "cqi_mean"]
    st.dataframe(
        df[weak_cols].sort_values(["sinr_mean", "rsrp_mean"], ascending=[True, True]).head(20).round(3),
        use_container_width=True,
        hide_index=True,
    )

    section("8. Coverage Map", "Single visual map for coverage quality. Red/yellow clusters in dense areas should make RF sense.")
    coverage_metric = st.selectbox("Coverage map metric", ["rsrp_mean", "sinr_mean", "cqi_mean"], index=1, key="model1_map_metric")
    map_block(df, coverage_metric, f"Model 1 Coverage Map: {coverage_metric}", "model1_map")


def model2_dataset_page(df: pd.DataFrame, model2_summary: dict) -> None:
    st.subheader("Model 2 Dataset Validation: Capacity / Congestion")
    st.caption("Purpose: decide whether the synthetic demand dataset is trustworthy before training.")

    section("1. Dataset Summary")
    metric_row(
        [
            ("Total Grids", df["grid_id"].nunique(), ""),
            ("Total Buckets", df["time_bucket"].nunique(), ""),
            ("Total Rows", len(df), ""),
            ("Date Range", "2025-08 -> 2026-05", ""),
        ]
    )

    section("2. Bucket Distribution", "Rows should stay balanced across PART_1, PART_2, PART_3.")
    bucket_distribution_block(df, "model2_bucket_distribution")

    section("3. Geo Distribution", "If the geo mix is strange, the demand logic will be strange too.")
    distribution_block(df, "model2_geo_distribution")

    section("4. KPI Trend by Bucket", "Demand is based on actual network observations plus geo context, so KPI trend must be visible here.")
    trend_chart(df, ["avg_rsrp", "avg_sinr", "avg_cqi", "avg_prb_pressure", "avg_dl_tpt"], "Model 2 KPI Inputs by Bucket", "model2_kpi_trends")

    section("5. Demand Feature Trends by Bucket", "This tells you immediately whether demand grows differently across buckets.")
    demand_bucket = bucket_summary(df)[["time_bucket", "avg_users", "avg_demand", "avg_traffic", "avg_growth", "samples"]]
    st.dataframe(demand_bucket.round(3), use_container_width=True, hide_index=True)
    trend_chart(df, ["avg_demand", "avg_users", "avg_traffic", "avg_growth"], "Demand Feature Trends", "model2_demand_trends")

    section("6. Cluster vs Demand Validation", "This is the main synthetic-feature sanity check. Dense Urban should generally outrank Open/Rural.")
    cluster_demand = (
        df.groupby("clutter_class", as_index=False)
        .agg(
            grids=("grid_id", "nunique"),
            avg_users=("active_users_est", "mean"),
            avg_demand=("demand_index", "mean"),
            avg_traffic=("traffic_demand_est", "mean"),
            avg_geo_score=("geo_demand_score", "mean"),
            avg_kpi_score=("kpi_demand_score", "mean"),
            prb_outliers=("prb_outlier_flag", "sum"),
        )
        .sort_values("avg_users", ascending=False)
    )
    cols = st.columns([1, 1])
    with cols[0]:
        st.dataframe(cluster_demand.round(3), use_container_width=True, hide_index=True)
    with cols[1]:
        fig = px.bar(cluster_demand, x="clutter_class", y="avg_users", color="avg_demand", title="Cluster vs Avg Users")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), xaxis_title="", yaxis_title="Avg Users")
        st.plotly_chart(fig, use_container_width=True, key="model2_cluster_demand")

    section("7. Top 20 Demand Hotspots", "These rows should line up with urban/high-activity grid locations.")
    hot_cols = [
        "time_bucket",
        "grid_id",
        "clutter_class",
        "building_count",
        "demand_index",
        "active_users_est",
        "traffic_demand_est",
        "dl_tpt_mean",
        "estimated_prb_mean",
        "prb_pressure_est",
        "prb_outlier_flag",
    ]
    st.dataframe(
        df[hot_cols].sort_values(["demand_index", "traffic_demand_est"], ascending=False).head(20).round(3),
        use_container_width=True,
        hide_index=True,
    )

    section("8. Demand Map", "Single map to verify high demand falls in dense/urban zones and low demand falls in open areas.")
    demand_metric = st.selectbox("Demand map metric", ["demand_index", "traffic_demand_est", "active_users_est"], key="model2_map_metric")
    map_block(df, demand_metric, f"Model 2 Demand Map: {demand_metric}", "model2_map")


def data_health_page(df: pd.DataFrame, coverage_summary: dict, model2_summary: dict) -> None:
    st.subheader("Data Health Checks")
    duplicate_rows = int(df.duplicated(["grid_id", "time_bucket"]).sum())
    metric_row(
        [
            ("Expected Rows", model2_summary.get("expected_rows_from_source_summary"), ""),
            ("Actual Rows", len(df), ""),
            ("Duplicate Grid+Bucket", duplicate_rows, ""),
            ("Coverage Source Rows", coverage_summary.get("total_rows"), ""),
        ]
    )
    required = ["demand_index", "active_users_est", "traffic_demand_est", "growth_rate", "clutter_class"]
    nulls = df[required].isna().sum().reset_index()
    nulls.columns = ["required_column", "null_count"]
    st.dataframe(nulls, use_container_width=True, hide_index=True)

    if "prb_outlier_flag" in df.columns:
        st.caption(
            f"Raw estimated_prb_mean outlier rows over 100: {int(df['prb_outlier_flag'].fillna(0).sum()):,}. "
            "Use prb_pressure_est for training."
        )

    kpi_cols = ["dl_tpt_mean", "estimated_prb_mean", "prb_pressure_est", "cqi_mean", "sinr_mean"]
    raw_nulls = df[kpi_cols].isna().sum().reset_index()
    raw_nulls.columns = ["raw_kpi_column", "missing_rows_preserved"]
    st.dataframe(raw_nulls, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Dataset Validation Dashboard", layout="wide")
    st.title("Dataset Validation Dashboard")
    st.caption("Question answered here: Do I trust my Model 1 and Model 2 datasets before training?")

    if not MODEL2_CSV.exists():
        st.error(f"Missing Model 2 dataset: {MODEL2_CSV}")
        st.stop()
    if not COVERAGE_ARCHIVE.exists():
        st.error(f"Missing coverage archive: {COVERAGE_ARCHIVE}")
        st.stop()

    df, model2_summary = load_dataset(str(MODEL2_CSV), str(MODEL2_SUMMARY_JSON))
    coverage_summary = load_coverage_summary(str(COVERAGE_ARCHIVE))

    with st.sidebar:
        st.header("Validation Filters")
        buckets = sorted(df["time_bucket"].dropna().astype(str).unique().tolist())
        selected_buckets = st.multiselect("Bucket", buckets, default=buckets)
        clusters = sorted(df["clutter_class"].dropna().astype(str).unique().tolist())
        selected_clusters = st.multiselect("Cluster / clutter", clusters, default=clusters)
        demand_min, demand_max = st.slider(
            "Demand index",
            min_value=float(df["demand_index"].min()),
            max_value=float(df["demand_index"].max()),
            value=(float(df["demand_index"].min()), float(df["demand_index"].max())),
        )

    filtered = df[
        df["time_bucket"].astype(str).isin(selected_buckets)
        & df["clutter_class"].astype(str).isin(selected_clusters)
        & df["demand_index"].between(demand_min, demand_max)
    ].copy()

    st.caption(f"Filtered rows: {len(filtered):,} / {len(df):,}")
    model1_tab, model2_tab, health_tab = st.tabs(["Model 1 Dataset", "Model 2 Dataset", "Data Health"])

    with model1_tab:
        model1_dataset_page(filtered, coverage_summary)
    with model2_tab:
        model2_dataset_page(filtered, model2_summary)
    with health_tab:
        data_health_page(filtered, coverage_summary, model2_summary)


if __name__ == "__main__":
    main()
