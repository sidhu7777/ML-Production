from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


OUTPUT_ROOT = Path("tests/output")
KPI_CONFIG = {
    "RSRP": {"pred": "pred_rsrp", "avg": "avg_rsrp", "baseline": "baseline_avg_rsrp", "threshold": -90.0, "unit": "dBm", "vmin": -120.0, "vmax": -70.0},
    "RSRQ": {"pred": "pred_rsrq", "avg": "avg_rsrq", "baseline": "baseline_avg_rsrq", "threshold": -14.0, "unit": "dB", "vmin": -20.0, "vmax": -3.0},
    "SINR": {"pred": "pred_sinr", "avg": "avg_sinr", "baseline": "baseline_avg_sinr", "threshold": 6.0, "unit": "dB", "vmin": -10.0, "vmax": 20.0},
}
BINS = {
    "RSRP": {
        "order": [">= -90", "-90 to -95", "-95 to -100", "-100 to -105", "< -105"],
        "colors": {">= -90": "#16a34a", "-90 to -95": "#a3e635", "-95 to -100": "#facc15", "-100 to -105": "#f97316", "< -105": "#dc2626"},
    },
    "RSRQ": {
        "order": [">= -10", "-10 to -12", "-12 to -14", "-14 to -16", "< -16"],
        "colors": {">= -10": "#16a34a", "-10 to -12": "#a3e635", "-12 to -14": "#facc15", "-14 to -16": "#f97316", "< -16": "#dc2626"},
    },
    "SINR": {
        "order": [">= 12", "6 to 12", "0 to 6", "-5 to 0", "< -5"],
        "colors": {">= 12": "#16a34a", "6 to 12": "#a3e635", "0 to 6": "#facc15", "-5 to 0": "#f97316", "< -5": "#dc2626"},
    },
}
DELTA_ORDER = ["< 1", "1 to 2", "2 to 3", "3 to 4", "4 to 5", ">= 5"]
DELTA_COLORS = {"< 1": "#cbd5e1", "1 to 2": "#93c5fd", "2 to 3": "#38bdf8", "3 to 4": "#22c55e", "4 to 5": "#facc15", ">= 5": "#dc2626"}


def _available_project_ids() -> List[int]:
    if not OUTPUT_ROOT.exists():
        return []
    ids: List[int] = []
    for path in OUTPUT_ROOT.iterdir():
        if path.is_dir() and path.name.startswith("project_"):
            try:
                ids.append(int(path.name.split("_", 1)[1]))
            except Exception:
                pass
    return sorted(set(ids), reverse=True)


def _list_runs(project_id: int) -> List[Path]:
    root = OUTPUT_ROOT / f"project_{project_id}"
    if not root.exists():
        return []
    runs: List[Path] = []
    for path in root.iterdir():
        summary_path = path / "summary.json"
        if not path.is_dir() or not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        if summary.get("run_type") == "tilt_combined_weighted_test" or path.name.startswith("tilt_combined_weighted_"):
            runs.append(path)
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    read_path = path
    if not read_path.exists() and path.suffix != ".gz":
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            read_path = gz_path
    if not read_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(read_path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _load_summary(run_dir: Path) -> Dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _format_number(value, digits: int = 2) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return "-"
    if abs(float(parsed) - int(parsed)) < 1e-9:
        return str(int(parsed))
    return f"{float(parsed):.{digits}f}"


def _format_runtime(seconds) -> str:
    parsed = pd.to_numeric(pd.Series([seconds]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return "-"
    minutes, secs = divmod(int(round(float(parsed))), 60)
    return f"{minutes}m {secs:02d}s"


def _parse_updates(best_summary: Dict) -> pd.DataFrame:
    updates = best_summary.get("selected_updates", []) if isinstance(best_summary, dict) else []
    if not isinstance(updates, list) or not updates:
        return pd.DataFrame()
    df = pd.DataFrame([row for row in updates if isinstance(row, dict)])
    for col in ["current_value", "target_value", "actual_delta", "requested_delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"current_value", "target_value"}.issubset(df.columns):
        df["tilt_direction"] = np.where(
            df["target_value"] < df["current_value"],
            "Uptilt",
            np.where(df["target_value"] > df["current_value"], "Downtilt", "No change"),
        )
    return df


def _grid_points(df: pd.DataFrame, kpi: str) -> pd.DataFrame:
    cfg = KPI_CONFIG[kpi]
    col = cfg["pred"]
    if df.empty or not {"grid_id", "lat", "lon", col}.issubset(df.columns):
        return pd.DataFrame()
    work = df[["grid_id", "lat", "lon", col]].copy()
    work["grid_id"] = work["grid_id"].astype(str)
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["grid_id", "lat", "lon", col])
    if work.empty:
        return pd.DataFrame()
    return (
        work.groupby("grid_id", dropna=False)
        .agg(lat=("lat", "mean"), lon=("lon", "mean"), value=(col, "mean"), sample_count=(col, "count"))
        .reset_index()
    )


def _frontend_baseline_grid(grid_analytics_df: pd.DataFrame, fallback_before_df: pd.DataFrame, kpi: str) -> pd.DataFrame:
    cfg = KPI_CONFIG[kpi]
    if not grid_analytics_df.empty and {"grid_id", cfg["baseline"]}.issubset(grid_analytics_df.columns):
        lat_col = "center_lat" if "center_lat" in grid_analytics_df.columns else "lat"
        lon_col = "center_lon" if "center_lon" in grid_analytics_df.columns else "lon"
        if {lat_col, lon_col}.issubset(grid_analytics_df.columns):
            out = grid_analytics_df[["grid_id", lat_col, lon_col, cfg["baseline"]]].copy()
            out = out.rename(columns={lat_col: "lat", lon_col: "lon", cfg["baseline"]: "value"})
            out["grid_id"] = out["grid_id"].astype(str)
            for col in ["lat", "lon", "value"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            return out.dropna(subset=["grid_id", "lat", "lon", "value"])
    return _grid_points(fallback_before_df, kpi)


def _comparison_grid(before_df: pd.DataFrame, after_df: pd.DataFrame, grid_analytics_df: pd.DataFrame, kpi: str) -> pd.DataFrame:
    before_grid = _frontend_baseline_grid(grid_analytics_df, before_df, kpi)
    model_before_grid = _grid_points(before_df, kpi)
    model_after_grid = _grid_points(after_df, kpi)
    if before_grid.empty or model_before_grid.empty or model_after_grid.empty:
        return pd.DataFrame()
    delta_grid = model_before_grid[["grid_id", "value"]].merge(
        model_after_grid[["grid_id", "value"]],
        on="grid_id",
        how="inner",
        suffixes=("_model_before", "_model_after"),
    )
    if delta_grid.empty:
        return pd.DataFrame()
    delta_grid["model_delta"] = (
        pd.to_numeric(delta_grid["value_model_after"], errors="coerce")
        - pd.to_numeric(delta_grid["value_model_before"], errors="coerce")
    )
    merged = before_grid.merge(delta_grid[["grid_id", "model_delta"]], on="grid_id", how="left")
    merged = merged.rename(columns={"value": "value_before"})
    merged["model_delta"] = pd.to_numeric(merged["model_delta"], errors="coerce").fillna(0.0)
    merged["value_after"] = pd.to_numeric(merged["value_before"], errors="coerce") + merged["model_delta"]
    merged["delta"] = merged["value_after"] - pd.to_numeric(merged["value_before"], errors="coerce")
    threshold = float(KPI_CONFIG[kpi]["threshold"])
    merged["before_bad"] = pd.to_numeric(merged["value_before"], errors="coerce") < threshold
    merged["after_bad"] = pd.to_numeric(merged["value_after"], errors="coerce") < threshold
    merged["transition"] = np.select(
        [
            merged["before_bad"] & ~merged["after_bad"],
            ~merged["before_bad"] & merged["after_bad"],
            merged["before_bad"] & merged["after_bad"],
        ],
        ["bad_to_good", "good_to_bad", "still_bad"],
        default="still_good",
    )
    return merged.dropna(subset=["lat", "lon", "value_before", "value_after"])


def _kpi_bin(value: float, kpi: str) -> str:
    if pd.isna(value):
        return "No data"
    value = float(value)
    if kpi == "RSRP":
        if value >= -90.0:
            return ">= -90"
        if value >= -95.0:
            return "-90 to -95"
        if value >= -100.0:
            return "-95 to -100"
        if value >= -105.0:
            return "-100 to -105"
        return "< -105"
    if kpi == "RSRQ":
        if value >= -10.0:
            return ">= -10"
        if value >= -12.0:
            return "-10 to -12"
        if value >= -14.0:
            return "-12 to -14"
        if value >= -16.0:
            return "-14 to -16"
        return "< -16"
    if value >= 12.0:
        return ">= 12"
    if value >= 6.0:
        return "6 to 12"
    if value >= 0.0:
        return "0 to 6"
    if value >= -5.0:
        return "-5 to 0"
    return "< -5"


def _delta_bin(value: float) -> str:
    if pd.isna(value):
        return "No data"
    value = abs(float(value))
    if value < 1.0:
        return "< 1"
    if value < 2.0:
        return "1 to 2"
    if value < 3.0:
        return "2 to 3"
    if value < 4.0:
        return "3 to 4"
    if value < 5.0:
        return "4 to 5"
    return ">= 5"


def _plot_kpi_maps(compare_df: pd.DataFrame, antenna_df: pd.DataFrame, kpi: str) -> None:
    if compare_df.empty:
        st.info(f"No mapped before/after grid rows for {kpi}.")
        return
    plot_df = compare_df.copy()
    plot_df["before_bin"] = plot_df["value_before"].map(lambda value: _kpi_bin(value, kpi))
    plot_df["after_bin"] = plot_df["value_after"].map(lambda value: _kpi_bin(value, kpi))
    plot_df["delta_bin"] = plot_df["delta"].map(_delta_bin)
    x_pad = max((compare_df["lon"].max() - compare_df["lon"].min()) * 0.08, 0.0015)
    y_pad = max((compare_df["lat"].max() - compare_df["lat"].min()) * 0.08, 0.0015)
    x0, x1 = compare_df["lon"].min() - x_pad, compare_df["lon"].max() + x_pad
    y0, y1 = compare_df["lat"].min() - y_pad, compare_df["lat"].max() + y_pad

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=140)
    panels = [
        (f"Before {kpi}", "before_bin", BINS[kpi]["order"], BINS[kpi]["colors"]),
        (f"After {kpi}", "after_bin", BINS[kpi]["order"], BINS[kpi]["colors"]),
        (f"Delta {kpi}", "delta_bin", DELTA_ORDER, DELTA_COLORS),
    ]
    for ax, (title, col, order, colors) in zip(axes, panels):
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#f8fafc")
        ax.grid(color="#cbd5e1", alpha=0.35, linewidth=0.6)
        for label in order:
            group = plot_df.loc[plot_df[col] == label]
            if group.empty:
                continue
            ax.scatter(
                group["lon"],
                group["lat"],
                s=18,
                c=colors[label],
                alpha=0.86,
                label=f"{label} ({len(group)})",
                linewidths=0.0,
            )
        if not antenna_df.empty and {"lat", "lon"}.issubset(antenna_df.columns):
            ant = antenna_df.copy()
            ant["lat"] = pd.to_numeric(ant["lat"], errors="coerce")
            ant["lon"] = pd.to_numeric(ant["lon"], errors="coerce")
            ant = ant.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["Node_Cell_ID"] if "Node_Cell_ID" in ant.columns else ["lat", "lon"])
            ax.scatter(ant["lon"], ant["lat"], s=12, c="#111827", marker="^", alpha=0.45)
        ax.legend(loc="best", fontsize=7, frameon=True)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.suptitle("Before / After / Delta Map", fontsize=12)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _plot_cdf(compare_df: pd.DataFrame, kpi: str) -> None:
    if compare_df.empty:
        return
    cfg = KPI_CONFIG[kpi]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for col, label in [("value_before", "Before"), ("value_after", "After")]:
        values = pd.to_numeric(compare_df[col], errors="coerce").dropna().sort_values()
        if values.empty:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, linewidth=2, label=label)
    ax.axvline(float(cfg["threshold"]), color="#dc2626", linestyle="--", linewidth=1.2, label=f"Threshold {cfg['threshold']}")
    ax.grid(color="#cbd5e1", alpha=0.4)
    ax.set_title(f"{kpi} CDF")
    ax.set_xlabel(f"{kpi} ({cfg['unit']})")
    ax.set_ylabel("Cumulative Share")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _kpi_impact_table(comparisons: Dict[str, pd.DataFrame], grid_analytics_df: pd.DataFrame, weights: Dict) -> pd.DataFrame:
    rows = []
    for kpi, compare_df in comparisons.items():
        threshold = float(KPI_CONFIG[kpi]["threshold"])
        weight = pd.to_numeric(pd.Series([weights.get(kpi.lower())]), errors="coerce").iloc[0] if isinstance(weights, dict) else np.nan
        total_frontend_bad = np.nan
        baseline_col = KPI_CONFIG[kpi]["baseline"]
        if not grid_analytics_df.empty and baseline_col in grid_analytics_df.columns:
            total_frontend_bad = int((pd.to_numeric(grid_analytics_df[baseline_col], errors="coerce") < threshold).sum())
        if compare_df.empty:
            rows.append({"KPI": kpi, "Frontend Before Bad Grids": total_frontend_bad})
            continue
        before_bad = int(compare_df["before_bad"].sum())
        after_bad = int(compare_df["after_bad"].sum())
        bad_to_good = int((compare_df["transition"] == "bad_to_good").sum())
        good_to_bad = int((compare_df["transition"] == "good_to_bad").sum())
        frontend_before = int(total_frontend_bad) if pd.notna(total_frontend_bad) else before_bad
        frontend_after = frontend_before - bad_to_good + good_to_bad
        rows.append(
            {
                "KPI": kpi,
                "Weight": weight,
                "Threshold": threshold,
                "Frontend Before Bad Grids": frontend_before,
                "Frontend After Bad Grids": frontend_after,
                "Delta Bad Grids": frontend_after - frontend_before,
                "Net Improvement": frontend_before - frontend_after,
                "Bad To Good": bad_to_good,
                "Good To Bad": good_to_bad,
                "Mean Before": float(compare_df["value_before"].mean()),
                "Mean After": float(compare_df["value_after"].mean()),
                "Mean Delta": float(compare_df["delta"].mean()),
                "Common Before Bad": before_bad,
                "Common After Bad": after_bad,
                "Common Grids": int(len(compare_df)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Tilt Per-KPI Frontend Dashboard", layout="wide")
    st.title("Tilt Per-KPI Frontend Dashboard")

    project_ids = _available_project_ids()
    if project_ids:
        project_id = st.sidebar.selectbox("Project ID", project_ids, index=0)
    else:
        project_id = st.sidebar.number_input("Project ID", value=196, min_value=1, step=1)

    runs = _list_runs(int(project_id))
    if not runs:
        st.info(f"No combined weighted runs found for project {int(project_id)}.")
        return
    run_name = st.selectbox("Available Runs", [run.name for run in runs], index=0)
    run_dir = next(run for run in runs if run.name == run_name)

    summary = _load_summary(run_dir)
    best_summary_path = run_dir / "best_candidate_summary.json"
    best_summary = json.loads(best_summary_path.read_text(encoding="utf-8")) if best_summary_path.exists() else summary.get("best_candidate", {})
    before_df = _safe_read_csv(run_dir / "best_candidate_before_scope.csv")
    after_df = _safe_read_csv(run_dir / "best_candidate_after_scope.csv")
    grid_analytics_df = _safe_read_csv(run_dir / "grid_analytics_input.csv")
    antenna_df = _safe_read_csv(run_dir / "antenna_input.csv")
    recommendations_df = _safe_read_csv(run_dir / "recommendations.csv")
    candidate_df = _safe_read_csv(run_dir / "candidate_validation_results.csv")
    saved_impact_df = _safe_read_csv(run_dir / "combined_kpi_grid_impact.csv")

    thresholds = summary.get("thresholds", {})
    for kpi, cfg in KPI_CONFIG.items():
        key = kpi.lower()
        if key in thresholds and thresholds[key] is not None:
            cfg["threshold"] = float(thresholds[key])

    weights = summary.get("weights", {})
    counts = summary.get("counts", {})
    updates_df = _parse_updates(best_summary)
    uptilt_count = int((updates_df.get("tilt_direction") == "Uptilt").sum()) if not updates_df.empty else 0
    downtilt_count = int((updates_df.get("tilt_direction") == "Downtilt").sum()) if not updates_df.empty else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Score", _format_number(best_summary.get("score"), 4))
    c2.metric("Frontend RSRP Bad", _format_number((pd.to_numeric(grid_analytics_df.get("baseline_avg_rsrp"), errors="coerce") < KPI_CONFIG["RSRP"]["threshold"]).sum()) if not grid_analytics_df.empty and "baseline_avg_rsrp" in grid_analytics_df.columns else "-")
    c3.metric("Frontend RSRQ Bad", _format_number((pd.to_numeric(grid_analytics_df.get("baseline_avg_rsrq"), errors="coerce") < KPI_CONFIG["RSRQ"]["threshold"]).sum()) if not grid_analytics_df.empty and "baseline_avg_rsrq" in grid_analytics_df.columns else "-")
    c4.metric("Frontend SINR Bad", _format_number((pd.to_numeric(grid_analytics_df.get("baseline_avg_sinr"), errors="coerce") < KPI_CONFIG["SINR"]["threshold"]).sum()) if not grid_analytics_df.empty and "baseline_avg_sinr" in grid_analytics_df.columns else "-")
    c5.metric("Tilt Changes", len(updates_df))
    c6.metric("Runtime", _format_runtime(summary.get("total_runtime_sec")))

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("RSRP Weight", _format_number(weights.get("rsrp"), 2))
    w2.metric("RSRQ Weight", _format_number(weights.get("rsrq"), 2))
    w3.metric("SINR Weight", _format_number(weights.get("sinr"), 2))
    w4.metric("Uptilt / Downtilt", f"{uptilt_count} / {downtilt_count}")

    comparisons = {kpi: _comparison_grid(before_df, after_df, grid_analytics_df, kpi) for kpi in KPI_CONFIG}
    impact_df = _kpi_impact_table(comparisons, grid_analytics_df, weights)
    st.markdown("**Per-KPI Frontend Before / After / Delta**")
    st.dataframe(impact_df, use_container_width=True, hide_index=True)

    if not saved_impact_df.empty:
        with st.expander("Saved combined_kpi_grid_impact.csv"):
            st.dataframe(saved_impact_df, use_container_width=True, hide_index=True)

    st.markdown("**Tilt Applied**")
    if updates_df.empty:
        st.info("No ETilt action selected. Best result is HOLD.")
    else:
        st.dataframe(updates_df, use_container_width=True, hide_index=True)

    tabs = st.tabs(["RSRP", "SINR", "RSRQ", "Details"])
    for tab, kpi in zip(tabs[:3], ["RSRP", "SINR", "RSRQ"]):
        with tab:
            compare_df = comparisons[kpi]
            _plot_kpi_maps(compare_df, antenna_df, kpi)
            _plot_cdf(compare_df, kpi)
            if not compare_df.empty:
                transition_counts = compare_df["transition"].value_counts().rename_axis("transition").reset_index(name="grid_count")
                st.markdown(f"**{kpi} Grid Transitions**")
                st.dataframe(transition_counts, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown("**Best Candidate Summary**")
        overview_keys = [
            "candidate_name",
            "score",
            "baseline_bad_count",
            "candidate_bad_count",
            "net_bad_reduction",
            "good_area_loss_pct",
            "priority_kpi_worsened",
            "constraints_passed",
            "grid_scoring_source",
            "validation_scope",
        ]
        overview = pd.DataFrame([{key: best_summary.get(key) for key in overview_keys if key in best_summary}])
        st.dataframe(overview, use_container_width=True, hide_index=True)

        if not recommendations_df.empty:
            st.markdown("**Recommendations**")
            st.dataframe(recommendations_df, use_container_width=True, hide_index=True)
        if not candidate_df.empty:
            st.markdown("**Candidate Validation Results**")
            show_cols = [
                col
                for col in [
                    "candidate_name",
                    "score",
                    "baseline_bad_count",
                    "candidate_bad_count",
                    "net_bad_reduction",
                    "frontend_rsrp_net_bad_grid_reduction",
                    "frontend_rsrq_net_bad_grid_reduction",
                    "frontend_sinr_net_bad_grid_reduction",
                    "priority_kpi_worsened",
                    "constraints_passed",
                ]
                if col in candidate_df.columns
            ]
            st.dataframe(candidate_df[show_cols].head(200) if show_cols else candidate_df.head(200), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
