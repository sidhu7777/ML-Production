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
PHASE39_DIR = ML_ROOT / "tests" / "new-project" / "data" / "project_210_taiwan" / "cost231_phase39_equal_power_diagnostic"

RUNS = {
    "Project 210 Taiwan": {
        "production": OUTPUT_ROOT / "baseline_210_profile_20260908_115936",
        "phase43": OUTPUT_ROOT / "baseline_210_profile_20260907_174241",
        "phase43_v2": OUTPUT_ROOT / "baseline_210_profile_20260908_120413",
    },
    "Project 193 India": {
        "production": OUTPUT_ROOT / "baseline_193_profile_20260907_170610",
        "phase43": OUTPUT_ROOT / "baseline_193_profile_20260907_174519",
        "phase43_v2": OUTPUT_ROOT / "baseline_193_profile_20260907_183325",
        "phase43_v3": OUTPUT_ROOT / "baseline_193_profile_20260907_190114",
    },
}

RSRP_BINS = [
    (-147, -115, "#991b1b", "-147 to -115"),
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


@st.cache_data(show_spinner=False)
def load_dt_scored(path: str) -> pd.DataFrame:
    run = Path(path)
    scored_path = run / "dt_calibrated_predictions.parquet"
    if not scored_path.exists():
        scored_path = run / "dt_scored_predictions.parquet"
    drive_path = run / "drive_test_rows.parquet"
    if not scored_path.exists():
        return pd.DataFrame()
    scored = pd.read_parquet(scored_path)
    if "technology" not in scored.columns and "Technology" in scored.columns:
        scored["technology"] = scored["Technology"]
    if drive_path.exists():
        drive = pd.read_parquet(drive_path)
        measured_cols = [col for col in ["rsrp_measured", "RSRP", "rsrp", "lte_rsrp", "reference_signal_power"] if col in drive.columns]
        if measured_cols and len(drive) == len(scored):
            scored["rsrp_measured"] = pd.to_numeric(drive[measured_cols[0]], errors="coerce").to_numpy()
    return scored


def environment_mask(frame: pd.DataFrame, indoor: bool) -> pd.Series:
    branch = frame.get("obstruction_branch", pd.Series("", index=frame.index)).astype(str).str.lower()
    clutter = frame.get("clutter_class", pd.Series("", index=frame.index)).astype(str).str.lower()
    mask = branch.eq("indoor") | clutter.eq("indoor")
    return mask if indoor else ~mask






@st.cache_data(show_spinner=False)
def load_phase39_serving_reference(tech: str) -> pd.DataFrame:
    path = PHASE39_DIR / f"phase39_serving_grid_{tech.lower()}_project210.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_phase39_validation_reference(tech: str) -> pd.DataFrame:
    path = PHASE39_DIR / "phase39_validation_dt_project210.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    out = frame[frame["technology"].astype(str).eq(tech)].copy()
    out = out[out.get("obstruction_branch", pd.Series("", index=out.index)).astype(str).ne("indoor")].copy()
    for excluded_col in ["p36_backlobe", "p38_excluded"]:
        if excluded_col in out.columns:
            out = out[~out[excluded_col].astype(bool)].copy()
    return out


def phase39_reference_cdf_figure(tech: str, aggregation: str) -> go.Figure:
    agg_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    agg_label = "frontend mean" if agg_suffix == "mean" else "serving cell"
    final_col = f"phase39_final_{agg_suffix}_rsrp"
    serv = load_phase39_serving_reference(tech)
    val = load_phase39_validation_reference(tech)
    fig = go.Figure()
    if not val.empty:
        fig.add_trace(cdf_trace(val["rsrp_measured"], "1 - DT measured (outdoor)", "#e5e7eb"))
        fig.add_trace(cdf_trace(val["phase39_final_rsrp"], "2 - Phase 39 calibrated predicted at DT", "#3b82f6"))
    if not serv.empty and final_col in serv.columns:
        fig.add_trace(cdf_trace(serv.loc[serv["serving_environment"].eq("outdoor"), final_col], f"3 - Phase 39 calibrated outdoor polygon ({agg_label})", "#22c55e"))
        fig.add_trace(cdf_trace(serv.loc[serv["serving_environment"].eq("indoor"), final_col], f"4 - Phase 39 calibrated indoor polygon ({agg_label})", "#f59e0b"))
    fig.update_layout(
        title=f"{tech} canonical Phase 39 production CDF reference ({agg_label})",
        height=470,
        xaxis_title="RSRP (dBm)",
        yaxis_title="Cumulative %",
        yaxis_range=[0, 100],
        xaxis_range=[-140, -45],
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#f8fafc"),
    )
    fig.update_xaxes(gridcolor="#374151", zerolinecolor="#6b7280")
    fig.update_yaxes(gridcolor="#374151", zerolinecolor="#6b7280")
    return fig


def serving_grid_for_phase39_cdf(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror Phase 39 CDF polygon source: one serving row per technology/grid."""
    if frame.empty or "grid_id" not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    if "technology" not in work.columns and "Technology" in work.columns:
        work["technology"] = work["Technology"]
    value_col = "final_rsrp" if "final_rsrp" in work.columns else "pred_rsrp"
    sort_col = "final_rsrp_unclipped" if "final_rsrp_unclipped" in work.columns else value_col
    work["_phase43_cdf_sort"] = pd.to_numeric(work[sort_col], errors="coerce")
    best = work.sort_values("_phase43_cdf_sort").groupby(["technology", "grid_id"], dropna=False).tail(1).copy()
    best["phase43_cdf_rsrp"] = pd.to_numeric(best[value_col], errors="coerce")
    branch = best.get("obstruction_branch", pd.Series("", index=best.index)).astype(str)
    best["serving_environment"] = np.where(branch.eq("indoor"), "indoor", "outdoor")
    return best.reset_index(drop=True)


def validation_dt_for_phase39_cdf(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror Phase 39 DT CDF intent: held-out outdoor DT rows only."""
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if "technology" not in out.columns and "Technology" in out.columns:
        out["technology"] = out["Technology"]
    if "split" in out.columns:
        out = out[out["split"].astype(str).str.lower().eq("validation")].copy()
    elif "phase25_split" in out.columns:
        out = out[out["phase25_split"].astype(str).str.lower().eq("validation")].copy()
    out = out[out.get("obstruction_branch", pd.Series("", index=out.index)).astype(str).ne("indoor")].copy()
    for excluded_col in ["p36_backlobe", "p38_excluded"]:
        if excluded_col in out.columns:
            out = out[~out[excluded_col].astype(bool)].copy()
    return out

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
            f"<b>{metric}:</b> {value:.2f}<br>"
            f"<b>Samples:</b> {getattr(row, 'samples', '')}"
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



def grid_display_frame(frame: pd.DataFrame, metric: str, mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    lat_col = "center_lat" if "center_lat" in frame.columns else "lat"
    lon_col = "center_lon" if "center_lon" in frame.columns else "lon"
    min_lat_col = "min_lat" if "min_lat" in frame.columns else "grid_min_lat"
    max_lat_col = "max_lat" if "max_lat" in frame.columns else "grid_max_lat"
    min_lon_col = "min_lon" if "min_lon" in frame.columns else "grid_min_lon"
    max_lon_col = "max_lon" if "max_lon" in frame.columns else "grid_max_lon"
    work = frame.dropna(subset=[metric, lat_col, lon_col]).copy()
    if work.empty:
        return work
    if mode == "Frontend API avg by lat/lon/site":
        work["lat_6dp"] = pd.to_numeric(work[lat_col], errors="coerce").round(6)
        work["lon_6dp"] = pd.to_numeric(work[lon_col], errors="coerce").round(6)
        site_col = "site" if "site" in work.columns else "strict_cell_key"
        work["frontend_site_id"] = work.get(site_col, pd.Series("", index=work.index)).astype(str)
        agg = {
            metric: (metric, "mean"),
            "center_lat": ("lat_6dp", "first"),
            "center_lon": ("lon_6dp", "first"),
            "samples": (metric, "size"),
        }
        if "grid_id" in work.columns:
            agg["grid_id"] = ("grid_id", "first")
        else:
            work["grid_id"] = work["lat_6dp"].astype(str) + ":" + work["lon_6dp"].astype(str)
            agg["grid_id"] = ("grid_id", "first")
        for col in [min_lat_col, max_lat_col, min_lon_col, max_lon_col]:
            if col in work.columns:
                agg[col] = (col, "first")
        for col in ["operator", "technology", "strict_cell_key", "site", "sector", "band", "clutter_class", "obstruction_branch"]:
            if col in work.columns:
                agg[col] = (col, "first")
        out = work.groupby(["lat_6dp", "lon_6dp", "frontend_site_id"], as_index=False).agg(**agg)
    else:
        if "grid_id" not in work.columns:
            return work.copy()
        idx = work.groupby("grid_id")[metric].idxmax()
        out = work.loc[idx].copy()
        out["samples"] = work.groupby("grid_id")[metric].size().reindex(out["grid_id"]).to_numpy()
    rename = {}
    if lat_col in out.columns:
        rename[lat_col] = "center_lat"
    if lon_col in out.columns:
        rename[lon_col] = "center_lon"
    if min_lat_col in out.columns:
        rename[min_lat_col] = "min_lat"
    if max_lat_col in out.columns:
        rename[max_lat_col] = "max_lat"
    if min_lon_col in out.columns:
        rename[min_lon_col] = "min_lon"
    if max_lon_col in out.columns:
        rename[max_lon_col] = "max_lon"
    out = out.rename(columns=rename)
    return out.reset_index(drop=True)

def render() -> None:
    st.title("Phase 43 Baseline Optimization")
    st.info(
        "Phase 43 is a speed optimization check. Production and optimized maps/CDF should overlap exactly. "
        "These plots use the captured diagnostic prediction parquet, not the live frontend DB/API layer unless that export is added."
    )

    with st.sidebar:
        project = st.selectbox("Project", list(RUNS), index=0)
        available_runs = [key for key in ["phase43", "phase43_v2", "phase43_v3"] if key in RUNS[project]]
        optimized_run = st.selectbox("Optimized run", available_runs, index=len(available_runs) - 1)
        metric = st.selectbox("Metric", ["pred_rsrp", "pred_rsrq", "pred_sinr"], index=0)
        display_mode = st.radio("Aggregation", ["Serving cell (best server)", "Frontend (mean of candidates)", "Both"], index=0)
        view_mode = st.radio("Map view", ["Interactive", "Static"], index=0)

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

    prod_raw_f = filtered(prod, tech, operator)
    opt_raw_f = filtered(opt, tech, operator)
    display_modes = ["Serving cell (best server)", "Frontend (mean of candidates)"] if display_mode == "Both" else [display_mode]

    def mode_to_grid_name(mode: str) -> str:
        return "Frontend API avg by lat/lon/site" if mode.startswith("Frontend") else "Best server / MAX per grid"

    cols = st.columns(5)
    cols[0].metric("Production wall", f"{prod_summary.get('wall_s', 0.0):.1f}s")
    cols[1].metric(f"{optimized_run} wall", f"{opt_summary.get('wall_s', 0.0):.1f}s")
    cols[2].metric("Production scorer", f"{scorer_seconds(prod_timings):.1f}s")
    cols[3].metric(f"{optimized_run} scorer", f"{scorer_seconds(opt_timings):.1f}s")
    cols[4].metric("Raw rows", f"{len(opt_raw_f):,}")
    st.caption("Phase 43 compares production-before vs optimized-after. Serving cell is best-server by grid_id. Frontend mode mirrors GetLtePredictionLocationStats: avg by rounded lat/lon/site with sampleCount.")
    st.caption(f"Production source: {prod_dir.name}; optimized source: {phase43_dir.name}")

    for selected_mode in display_modes:
        grid_mode = mode_to_grid_name(selected_mode)
        prod_f = grid_display_frame(prod_raw_f, metric, grid_mode)
        opt_f = grid_display_frame(opt_raw_f, metric, grid_mode)
        st.divider()
        st.header(selected_mode)
        st.metric("Display grid rows", f"{len(opt_f):,}", delta=f"raw {len(opt_raw_f):,}")

        aligned = pd.DataFrame()
        aligned_opt = pd.DataFrame()
        if len(prod_f) and len(opt_f):
            aligned = prod_f.sort_values(["grid_id"]).reset_index(drop=True)
            aligned_opt = opt_f.sort_values(["grid_id"]).reset_index(drop=True)
            if len(aligned) == len(aligned_opt) and aligned["grid_id"].astype(str).equals(aligned_opt["grid_id"].astype(str)):
                diff_rows = []
                for diff_metric in ["pred_rsrp", "pred_rsrq", "pred_sinr", "building_obstruction_loss_db", "terrain_diffraction_loss_db", "physical_rsrp_unclipped"]:
                    if diff_metric in aligned.columns and diff_metric in aligned_opt.columns:
                        diff = (pd.to_numeric(aligned[diff_metric], errors="coerce") - pd.to_numeric(aligned_opt[diff_metric], errors="coerce")).abs()
                        diff_rows.append({"field": diff_metric, "max_abs_diff": float(diff.max()), "changed_rows": int((diff > 1e-9).sum())})
                label_rows = []
                for label_col in ["obstruction_branch", "clutter_class"]:
                    if label_col in aligned.columns and label_col in aligned_opt.columns:
                        label_rows.append({"field": label_col, "mismatch_rows": int((aligned[label_col].astype(str) != aligned_opt[label_col].astype(str)).sum())})
                st.subheader("Production vs optimized equality check")
                st.caption("For Phase 43, these values should be zero. Zero means latency improved without changing RF/model output.")
                if diff_rows:
                    st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
                if label_rows:
                    st.dataframe(pd.DataFrame(label_rows), use_container_width=True, hide_index=True)
                selected_diff = (pd.to_numeric(aligned[metric], errors="coerce") - pd.to_numeric(aligned_opt[metric], errors="coerce")).abs()
                st.metric(f"Selected metric max abs diff: {metric}", f"{float(selected_diff.max()):.6f}")
                if float(selected_diff.max()) == 0.0:
                    st.success("Production and optimized values are exactly identical for this aggregation and filter.")
            else:
                st.error(f"Cannot align rows for exact comparison: production={len(aligned):,}, optimized={len(aligned_opt):,}")

        st.subheader("Phase 37/39-style RSRP CDF guardrail")
        prod_dt = load_dt_scored(str(prod_dir))
        opt_dt = load_dt_scored(str(phase43_dir))
        prod_dt_f = filtered(validation_dt_for_phase39_cdf(prod_dt), tech, operator) if not prod_dt.empty else pd.DataFrame()
        opt_dt_f = filtered(validation_dt_for_phase39_cdf(opt_dt), tech, operator) if not opt_dt.empty else pd.DataFrame()
        prod_serving_cdf = filtered(serving_grid_for_phase39_cdf(prod_raw_f), tech, operator)
        opt_serving_cdf = filtered(serving_grid_for_phase39_cdf(opt_raw_f), tech, operator)

        def four_curve_guard(title: str, serving_frame: pd.DataFrame, dt_frame: pd.DataFrame, predicted_label: str) -> go.Figure:
            fig = go.Figure()
            if not dt_frame.empty and "rsrp_measured" in dt_frame.columns:
                fig.add_trace(cdf_trace(dt_frame["rsrp_measured"], "1 - DT measured (outdoor)", "#e5e7eb"))
                pred_col = "final_rsrp" if "final_rsrp" in dt_frame.columns else ("pred_rsrp" if "pred_rsrp" in dt_frame.columns else None)
                if pred_col:
                    fig.add_trace(cdf_trace(dt_frame[pred_col], f"2 - {predicted_label} calibrated predicted at DT", "#3b82f6"))
            if "phase43_cdf_rsrp" in serving_frame.columns:
                fig.add_trace(cdf_trace(serving_frame.loc[serving_frame["serving_environment"].eq("outdoor"), "phase43_cdf_rsrp"], f"3 - {predicted_label} calibrated outdoor polygon (serving cell)", "#22c55e"))
                fig.add_trace(cdf_trace(serving_frame.loc[serving_frame["serving_environment"].eq("indoor"), "phase43_cdf_rsrp"], f"4 - {predicted_label} calibrated indoor polygon (serving cell)", "#f59e0b"))
            fig.update_layout(
                title=title,
                height=470,
                xaxis_title="RSRP (dBm)",
                yaxis_title="Cumulative %",
                yaxis_range=[0, 100],
                xaxis_range=[-140, -45],
                legend=dict(orientation="h", yanchor="bottom", y=-0.35),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#f8fafc"),
            )
            fig.update_xaxes(gridcolor="#374151", zerolinecolor="#6b7280")
            fig.update_yaxes(gridcolor="#374151", zerolinecolor="#6b7280")
            return fig

        guard_cols = st.columns(2)
        with guard_cols[0]:
            st.plotly_chart(
                four_curve_guard(
                    f"Production baseline CDF guardrail - {selected_mode}",
                    prod_serving_cdf,
                    prod_dt_f,
                    "Production baseline",
                ),
                use_container_width=True,
            )
        with guard_cols[1]:
            st.plotly_chart(
                four_curve_guard(
                    f"{optimized_run} CDF guardrail - {selected_mode}",
                    opt_serving_cdf,
                    opt_dt_f,
                    optimized_run,
                ),
                use_container_width=True,
            )
        if prod_dt_f.empty or opt_dt_f.empty or "rsrp_measured" not in prod_dt_f.columns or "rsrp_measured" not in opt_dt_f.columns:
            st.warning("DT measured/predicted-at-DT curves need drive_test_rows.parquet plus dt_calibrated_predictions.parquet. Older runs may only show outdoor/indoor polygon curves until rerun with the updated harness.")
        st.caption("This is the valid Phase43 optimization guardrail: production baseline run and optimized run are drawn from the same pipeline, same DT selection, same serving-grid aggregation. DT counts must match; if they do not, the comparison is invalid.")
        if project == "Project 210 Taiwan" and tech in ["4G", "5G"]:
            with st.expander("Canonical Phase39 equal-power reference only ? not used as the Phase43 comparison baseline"):
                st.plotly_chart(phase39_reference_cdf_figure(tech, selected_mode), use_container_width=True)

        st.subheader("CDF: production before vs optimized after")
        fig = go.Figure()
        fig.add_trace(cdf_trace(prod_f[metric], "Production", "#ef4444"))
        fig.add_trace(cdf_trace(opt_f[metric], optimized_run, "#22c55e"))
        fig.update_layout(height=420, xaxis_title=metric, yaxis_title="Cumulative %", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Maps: production before vs optimized after")
        map_cols = st.columns(2)
        with map_cols[0]:
            st.caption("Production baseline / before")
            map_frame(prod_f, metric, f"{project} production {metric} - {selected_mode}", view_mode)
        with map_cols[1]:
            st.caption(f"{optimized_run} optimized baseline / after")
            map_frame(opt_f, metric, f"{project} {optimized_run} {metric} - {selected_mode}", view_mode)

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





