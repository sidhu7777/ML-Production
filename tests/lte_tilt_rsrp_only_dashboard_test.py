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

from tests import lte_tilt_rsrp_only_recommendation_test as rsrp_test


OUTPUT_ROOT = Path("tests/output")
MAP_IMAGE_NAME = "best_candidate_before_after_map.png"
ATTEMPT_MAP_IMAGE_NAME = "best_attempt_before_after_map.png"


def _available_project_ids() -> List[int]:
    if not OUTPUT_ROOT.exists():
        return []
    ids: List[int] = []
    for path in OUTPUT_ROOT.iterdir():
        if not path.is_dir():
            continue
        name = path.name.strip()
        if not name.startswith("project_"):
            continue
        try:
            ids.append(int(name.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(ids), reverse=True)


def _list_runs(project_id: int) -> List[Path]:
    root = OUTPUT_ROOT / f"project_{project_id}"
    if not root.exists():
        return []
    runs = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        summary_path = path / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if summary.get("run_type") == "tilt_rsrp_only_test" or path.name.startswith("tilt_rsrp_only_"):
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
        return pd.read_csv(read_path)
    except Exception:
        return pd.DataFrame()


def _load_summary(run_dir: Path) -> Dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _parse_update_payload(value) -> List[Dict]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None"}:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _cell_metadata(antenna_df: pd.DataFrame) -> pd.DataFrame:
    if antenna_df.empty:
        return pd.DataFrame()
    meta_cols = [
        col
        for col in [
            "Node_Cell_ID",
            "cell_id",
            "dashboard_site_id",
            "site",
            "nodeb_id",
            "electrical_tilt",
            "mechanical_tilt",
            "azimuth",
            "tx_power",
            "band",
            "earfcn",
        ]
        if col in antenna_df.columns
    ]
    if not meta_cols or "Node_Cell_ID" not in meta_cols:
        return pd.DataFrame()
    out = antenna_df[meta_cols].copy()
    out["Node_Cell_ID"] = out["Node_Cell_ID"].astype(str)
    return out.drop_duplicates(subset=["Node_Cell_ID"])


def _build_candidate_action_df(candidate_df: pd.DataFrame, antenna_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty or "target_value" not in candidate_df.columns:
        return pd.DataFrame()

    rows: List[Dict] = []
    for _, row in candidate_df.iterrows():
        updates = _parse_update_payload(row.get("target_value"))
        if not updates:
            continue
        for update in updates:
            out = {
                "site_id": row.get("site_id"),
                "candidate_name": row.get("candidate_name"),
                "score": row.get("score"),
                "baseline_bad_count": row.get("baseline_bad_count"),
                "candidate_bad_count": row.get("candidate_bad_count"),
                "net_bad_reduction": row.get("net_bad_reduction"),
                "recovered_bad_samples": row.get("recovered_bad_samples"),
                "new_bad_samples": row.get("new_bad_samples"),
                "changed_cell_id": update.get("cell_id"),
                "parameter": update.get("parameter"),
                "recommended_value": update.get("target_value"),
            }
            rows.append(out)

    if not rows:
        return pd.DataFrame()
    action_df = pd.DataFrame(rows)
    for col in ["score", "baseline_bad_count", "candidate_bad_count", "net_bad_reduction", "recovered_bad_samples", "new_bad_samples", "recommended_value"]:
        if col in action_df.columns:
            action_df[col] = pd.to_numeric(action_df[col], errors="coerce")

    meta = _cell_metadata(antenna_df)
    if not meta.empty:
        action_df["changed_cell_id"] = action_df["changed_cell_id"].astype(str)
        action_df = action_df.merge(
            meta.rename(columns={"Node_Cell_ID": "changed_cell_id", "electrical_tilt": "current_etilt"}),
            on="changed_cell_id",
            how="left",
        )
        if "current_etilt" in action_df.columns:
            action_df["etilt_delta"] = action_df["recommended_value"] - pd.to_numeric(action_df["current_etilt"], errors="coerce")
    return action_df


def _add_candidate_change_columns(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    out = candidate_df.copy()
    if "target_value" not in out.columns:
        return out

    changed_cells: List[str] = []
    changed_counts: List[int] = []
    changed_values: List[str] = []
    for value in out["target_value"]:
        updates = _parse_update_payload(value)
        cell_ids = [str(item.get("cell_id", "")).strip() for item in updates if str(item.get("cell_id", "")).strip()]
        values = [
            f"{item.get('cell_id')} {item.get('parameter')}->{item.get('target_value')}"
            for item in updates
            if str(item.get("cell_id", "")).strip()
        ]
        changed_cells.append(",".join(cell_ids))
        changed_counts.append(len(cell_ids))
        changed_values.append("; ".join(values))

    out["changed_cell_ids"] = changed_cells
    out["changed_cell_count_from_target"] = changed_counts
    out["changed_cell_values"] = changed_values
    return out


def _build_fixed_extent(before_df: pd.DataFrame, after_df: pd.DataFrame, antenna_df: pd.DataFrame) -> tuple[float, float, float, float]:
    frames = []
    for df in [before_df, after_df, antenna_df]:
        if not df.empty and {"lat", "lon"}.issubset(df.columns):
            frames.append(df[["lat", "lon"]].copy())
    if not frames:
        return 0.0, 1.0, 0.0, 1.0
    coords = pd.concat(frames, ignore_index=True)
    coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
    coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
    coords = coords.dropna(subset=["lat", "lon"])
    if coords.empty:
        return 0.0, 1.0, 0.0, 1.0
    min_lon = float(coords["lon"].min())
    max_lon = float(coords["lon"].max())
    min_lat = float(coords["lat"].min())
    max_lat = float(coords["lat"].max())
    lon_pad = max((max_lon - min_lon) * 0.08, 0.0015)
    lat_pad = max((max_lat - min_lat) * 0.08, 0.0015)
    return min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad


def _render_static_map_image(
    run_dir: Path,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    summary: Dict,
    image_name: str = MAP_IMAGE_NAME,
    title: str = "Tilt-Only RSRP Recommendation Map",
) -> Path:
    out_path = run_dir / image_name
    if before_df.empty and after_df.empty:
        return out_path

    before = before_df.copy()
    after = after_df.copy()
    antenna = antenna_df.copy()
    for df in [before, after, antenna]:
        if not df.empty:
            if "lat" in df.columns:
                df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            if "lon" in df.columns:
                df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    x0, x1, y0, y1 = _build_fixed_extent(before, after, antenna)
    best = summary.get("best_candidate", {})
    rsrp_threshold = float(summary.get("thresholds", {}).get("rsrp", -90.0))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=160)
    panels = [
        (axes[0], before, f"Before Baseline\nRSRP < {rsrp_threshold:.0f} count: {int(summary.get('counts', {}).get('before_bad_rsrp_count', 0))}"),
        (axes[1], after, f"After Best Tilt\n{best.get('candidate_name', 'hold')} | score={best.get('score', 'nan')}"),
    ]
    for ax, df, title in panels:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#f8fafc")
        ax.grid(color="#cbd5e1", alpha=0.35, linewidth=0.6)
        if not antenna.empty and {"lat", "lon"}.issubset(antenna.columns):
            site_points = antenna.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["Node_Cell_ID"] if "Node_Cell_ID" in antenna.columns else ["lat", "lon"])
            ax.scatter(
                site_points["lon"],
                site_points["lat"],
                s=18,
                c="#111827",
                marker="^",
                alpha=0.7,
                label="Cells",
            )
        if not df.empty and {"lat", "lon", "pred_rsrp"}.issubset(df.columns):
            plot_df = df.dropna(subset=["lat", "lon", "pred_rsrp"]).copy()
            if not plot_df.empty:
                sc = ax.scatter(
                    plot_df["lon"],
                    plot_df["lat"],
                    c=plot_df["pred_rsrp"],
                    cmap="RdYlGn",
                    vmin=-120,
                    vmax=-70,
                    s=np.where(plot_df.get("is_bad_rsrp", False).fillna(False), 28, 18),
                    edgecolors=np.where(plot_df.get("is_bad_rsrp", False).fillna(False), "#7f1d1d", "none"),
                    linewidths=np.where(plot_df.get("is_bad_rsrp", False).fillna(False), 0.45, 0.0),
                    alpha=0.9,
                )
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="Predicted RSRP (dBm)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle(
        f"{title}\nFixed extent for before/after comparison",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _grid_points_for_map(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"grid_id", "pred_rsrp"}.issubset(df.columns):
        return pd.DataFrame()
    work = df.copy()
    work["grid_id"] = work["grid_id"].astype(str)
    work["pred_rsrp"] = pd.to_numeric(work["pred_rsrp"], errors="coerce")
    for col in ["lat", "lon"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["grid_id", "pred_rsrp"])
    if work.empty:
        return pd.DataFrame()
    agg_map = {
        "avg_rsrp": ("pred_rsrp", "mean"),
        "sample_count": ("pred_rsrp", "count"),
    }
    if "lat" in work.columns:
        agg_map["lat"] = ("lat", "mean")
    if "lon" in work.columns:
        agg_map["lon"] = ("lon", "mean")
    out = work.groupby("grid_id", dropna=False).agg(**agg_map).reset_index()
    return out.dropna(subset=["lat", "lon"]) if {"lat", "lon"}.issubset(out.columns) else pd.DataFrame()


def _rsrp_bin_label(value: float) -> str:
    if pd.isna(value):
        return "No data"
    if value >= -90.0:
        return "-44 to -90"
    if value >= -95.0:
        return "-90 to -95"
    if value >= -100.0:
        return "-95 to -100"
    if value >= -105.0:
        return "-100 to -105"
    return "< -105"


def _delta_bin_label(value: float) -> str:
    if pd.isna(value):
        return "No data"
    if value < 1.0:
        return "< 1 dB"
    if value < 2.0:
        return "1 to 2 dB"
    if value < 3.0:
        return "2 to 3 dB"
    if value < 4.0:
        return "3 to 4 dB"
    if value < 5.0:
        return "4 to 5 dB"
    return ">= 5 dB"


def _render_binned_rsrp_maps(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    best_summary: Dict,
    trusted_grid_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    trusted = trusted_grid_df.copy() if isinstance(trusted_grid_df, pd.DataFrame) else pd.DataFrame()
    use_trusted = (
        not trusted.empty
        and "grid_id" in trusted.columns
        and "baseline_avg_rsrp" in trusted.columns
        and ({"center_lat", "center_lon"}.issubset(trusted.columns) or {"lat", "lon"}.issubset(trusted.columns))
    )
    if use_trusted:
        transition_df = _build_transition_df(before_df, after_df, -90.0)
        trusted["grid_id"] = trusted["grid_id"].astype(str)
        trusted["avg_rsrp_before"] = pd.to_numeric(trusted["baseline_avg_rsrp"], errors="coerce")
        trusted["lat"] = pd.to_numeric(trusted["center_lat"] if "center_lat" in trusted.columns else trusted["lat"], errors="coerce")
        trusted["lon"] = pd.to_numeric(trusted["center_lon"] if "center_lon" in trusted.columns else trusted["lon"], errors="coerce")
        if not transition_df.empty and {"grid_id", "rsrp_delta"}.issubset(transition_df.columns):
            delta_df = transition_df[["grid_id", "rsrp_delta"]].copy()
            delta_df["grid_id"] = delta_df["grid_id"].astype(str)
            delta_df["rsrp_delta"] = pd.to_numeric(delta_df["rsrp_delta"], errors="coerce")
            merged = trusted.merge(delta_df, on="grid_id", how="left")
        else:
            merged = trusted.copy()
            merged["rsrp_delta"] = 0.0
        merged["rsrp_delta"] = pd.to_numeric(merged["rsrp_delta"], errors="coerce").fillna(0.0)
        merged["avg_rsrp_after"] = merged["avg_rsrp_before"] + merged["rsrp_delta"]
        source_label = "Trusted grid analytics + candidate RF delta"
    else:
        before_grid = _grid_points_for_map(before_df)
        after_grid = _grid_points_for_map(after_df)
        if before_grid.empty or after_grid.empty:
            st.info("No before/after grid rows available for map visualization.")
            return pd.DataFrame()
        merged = before_grid.merge(after_grid, on="grid_id", how="outer", suffixes=("_before", "_after"))
        merged["lat"] = pd.to_numeric(merged.get("lat_after"), errors="coerce").fillna(pd.to_numeric(merged.get("lat_before"), errors="coerce"))
        merged["lon"] = pd.to_numeric(merged.get("lon_after"), errors="coerce").fillna(pd.to_numeric(merged.get("lon_before"), errors="coerce"))
        merged["avg_rsrp_before"] = pd.to_numeric(merged.get("avg_rsrp_before"), errors="coerce")
        merged["avg_rsrp_after"] = pd.to_numeric(merged.get("avg_rsrp_after"), errors="coerce")
        merged["rsrp_delta"] = merged["avg_rsrp_after"] - merged["avg_rsrp_before"]
        source_label = "Recomputed before/after scope"
    merged = merged.dropna(subset=["lat", "lon"])
    if merged.empty:
        st.info("No mapped before/after grid rows available.")
        return pd.DataFrame()

    rsrp_order = ["-44 to -90", "-90 to -95", "-95 to -100", "-100 to -105", "< -105"]
    rsrp_colors = {
        "-44 to -90": "#16a34a",
        "-90 to -95": "#a3e635",
        "-95 to -100": "#facc15",
        "-100 to -105": "#f97316",
        "< -105": "#dc2626",
    }
    delta_order = ["< 1 dB", "1 to 2 dB", "2 to 3 dB", "3 to 4 dB", "4 to 5 dB", ">= 5 dB"]
    delta_colors = {
        "< 1 dB": "#cbd5e1",
        "1 to 2 dB": "#93c5fd",
        "2 to 3 dB": "#38bdf8",
        "3 to 4 dB": "#22c55e",
        "4 to 5 dB": "#facc15",
        ">= 5 dB": "#dc2626",
    }
    merged["before_bin"] = merged["avg_rsrp_before"].map(_rsrp_bin_label)
    merged["after_bin"] = merged["avg_rsrp_after"].map(_rsrp_bin_label)
    merged["delta_bin"] = merged["rsrp_delta"].map(_delta_bin_label)

    x0, x1, y0, y1 = _build_fixed_extent(before_df, after_df, antenna_df)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    panels = [
        (axes[0], "Before RSRP", "before_bin", rsrp_order, rsrp_colors),
        (axes[1], "After RSRP", "after_bin", rsrp_order, rsrp_colors),
        (axes[2], "Delta RSRP", "delta_bin", delta_order, delta_colors),
    ]
    for ax, title, col, order, colors in panels:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#f8fafc")
        ax.grid(color="#cbd5e1", alpha=0.35, linewidth=0.6)
        for label in order:
            group = merged.loc[merged[col] == label]
            if group.empty:
                continue
            ax.scatter(group["lon"], group["lat"], s=18, c=colors[label], alpha=0.85, label=f"{label} ({len(group)})")
        if not antenna_df.empty and {"lat", "lon"}.issubset(antenna_df.columns):
            sites = antenna_df.copy()
            sites["lat"] = pd.to_numeric(sites["lat"], errors="coerce")
            sites["lon"] = pd.to_numeric(sites["lon"], errors="coerce")
            sites = sites.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["Node_Cell_ID"] if "Node_Cell_ID" in sites.columns else ["lat", "lon"])
            ax.scatter(sites["lon"], sites["lat"], s=12, c="#111827", marker="^", alpha=0.45)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="best", fontsize=7, frameon=True)

    selected_updates = best_summary.get("selected_updates", []) if isinstance(best_summary, dict) else []
    candidate = str(best_summary.get("candidate_name", "selected candidate")) if isinstance(best_summary, dict) else "selected candidate"
    update_count = len(selected_updates or [])
    title_text = f"Best Score Recommendation: {update_count} ETilt changes"
    if update_count <= 1:
        title_text = f"Best Score Recommendation: {candidate}"
    fig.suptitle(f"{title_text}\n{source_label}", fontsize=12)
    fig.tight_layout()
    st.markdown("**Before / After / Delta Map**")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    rows = []
    for state, col, order in [("before", "before_bin", rsrp_order), ("after", "after_bin", rsrp_order), ("delta", "delta_bin", delta_order)]:
        counts = merged[col].value_counts()
        for label in order:
            rows.append({"map": state, "range": label, "grid_count": int(counts.get(label, 0))})
    return pd.DataFrame(rows)


def _build_transition_df(before_df: pd.DataFrame, after_df: pd.DataFrame, rsrp_threshold: float | None = None) -> pd.DataFrame:
    if before_df.empty and after_df.empty:
        return pd.DataFrame()
    before = before_df.copy()
    after = after_df.copy()
    required = {"grid_id", "pred_rsrp"}
    if not required.issubset(before.columns) or not required.issubset(after.columns):
        return pd.DataFrame()

    threshold = float(rsrp_threshold) if rsrp_threshold is not None else -90.0
    if rsrp_threshold is None and "is_bad_rsrp" in before.columns:
        bad_before_values = pd.to_numeric(before["pred_rsrp"], errors="coerce")
        threshold_candidates = bad_before_values.loc[before["is_bad_rsrp"].fillna(False).astype(bool)]
        if not threshold_candidates.empty:
            threshold = float(np.ceil(threshold_candidates.max()))

    def _grid_agg(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        work = df.copy()
        work["grid_id"] = work["grid_id"].astype(str)
        work["pred_rsrp"] = pd.to_numeric(work["pred_rsrp"], errors="coerce")
        for col in ["lat", "lon"]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.dropna(subset=["grid_id", "pred_rsrp"])
        if work.empty:
            return pd.DataFrame()
        agg = (
            work.groupby("grid_id", dropna=False)
            .agg(
                **{
                    f"grid_avg_rsrp_{suffix}": ("pred_rsrp", "mean"),
                    f"sample_count_{suffix}": ("pred_rsrp", "count"),
                    f"lat_{suffix}": ("lat", "mean") if "lat" in work.columns else ("pred_rsrp", "size"),
                    f"lon_{suffix}": ("lon", "mean") if "lon" in work.columns else ("pred_rsrp", "size"),
                    f"serving_cells_{suffix}": ("Node_Cell_ID", lambda s: ",".join(sorted(set(s.astype(str)))) if "Node_Cell_ID" in work.columns else ""),
                }
            )
            .reset_index()
        )
        return agg

    before_grid = _grid_agg(before, "before")
    after_grid = _grid_agg(after, "after")
    if before_grid.empty or after_grid.empty:
        return pd.DataFrame()

    merged = before_grid.merge(after_grid, on="grid_id", how="outer")
    merged["lat"] = pd.to_numeric(merged.get("lat_after"), errors="coerce").fillna(pd.to_numeric(merged.get("lat_before"), errors="coerce"))
    merged["lon"] = pd.to_numeric(merged.get("lon_after"), errors="coerce").fillna(pd.to_numeric(merged.get("lon_before"), errors="coerce"))
    merged["grid_avg_rsrp_before"] = pd.to_numeric(merged.get("grid_avg_rsrp_before"), errors="coerce")
    merged["grid_avg_rsrp_after"] = pd.to_numeric(merged.get("grid_avg_rsrp_after"), errors="coerce")
    before_bad = merged["grid_avg_rsrp_before"] < float(threshold)
    after_bad = merged["grid_avg_rsrp_after"] < float(threshold)
    merged["transition"] = np.select(
        [
            before_bad & ~after_bad,
            ~before_bad & after_bad,
            before_bad & after_bad,
            ~before_bad & ~after_bad,
        ],
        [
            "bad_to_good",
            "good_to_bad",
            "bad_to_bad",
            "good_to_good",
        ],
        default="unknown",
    )
    merged["rsrp_delta"] = merged["grid_avg_rsrp_after"] - merged["grid_avg_rsrp_before"]
    return merged


def _build_threshold_sensitivity_df(before_df: pd.DataFrame, after_df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        transition_df = _build_transition_df(before_df, after_df, float(threshold))
        if transition_df.empty:
            continue
        counts = transition_df["transition"].value_counts(dropna=False)
        before_bad = int(((pd.to_numeric(transition_df.get("grid_avg_rsrp_before"), errors="coerce") < float(threshold))).fillna(False).sum())
        after_bad = int(((pd.to_numeric(transition_df.get("grid_avg_rsrp_after"), errors="coerce") < float(threshold))).fillna(False).sum())
        rows.append(
            {
                "threshold_dbm": float(threshold),
                "before_bad_grids": before_bad,
                "after_bad_grids": after_bad,
                "net_bad_grid_reduction": before_bad - after_bad,
                "bad_to_good": int(counts.get("bad_to_good", 0)),
                "good_to_bad": int(counts.get("good_to_bad", 0)),
                "still_bad": int(counts.get("bad_to_bad", 0)),
                "still_good": int(counts.get("good_to_good", 0)),
                "grid_count": int(len(transition_df)),
            }
        )
    return pd.DataFrame(rows)


def _trusted_grid_threshold_counts(grid_df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    if grid_df.empty:
        return pd.DataFrame()
    rsrp_col = None
    for candidate_col in ["baseline_avg_rsrp", "avg_rsrp", "grid_avg_rsrp"]:
        if candidate_col in grid_df.columns:
            rsrp_col = candidate_col
            break
    if rsrp_col is None:
        return pd.DataFrame()
    rsrp = pd.to_numeric(grid_df[rsrp_col], errors="coerce").dropna()
    if rsrp.empty:
        return pd.DataFrame()
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "threshold_dbm": float(threshold),
                "bad_grid_count": int((rsrp < float(threshold)).sum()),
                "total_grid_count": int(rsrp.size),
                "mean_rsrp": float(rsrp.mean()),
            }
        )
    return pd.DataFrame(rows)


def _render_trusted_grid_threshold_counts(grid_df: pd.DataFrame) -> None:
    counts_df = _trusted_grid_threshold_counts(grid_df, [-90.0, -95.0, -100.0, -105.0])
    if counts_df.empty:
        return
    st.markdown("**Trusted Baseline Grid RSRP Counts**")
    st.caption("Uses saved frontend/grid analytics grid averages. This is the trusted baseline count, not recomputed best-attempt scope.")
    st.dataframe(counts_df, use_container_width=True, hide_index=True)


def _rsrp_cdf_values(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.array([]), np.array([])
    values = np.sort(values)
    cdf = np.arange(1, values.size + 1, dtype=float) / float(values.size)
    return values, cdf


def _grid_rsrp_series(df: pd.DataFrame) -> pd.Series:
    if df.empty or not {"grid_id", "pred_rsrp"}.issubset(df.columns):
        return pd.Series(dtype=float)
    work = df.copy()
    work["grid_id"] = work["grid_id"].astype(str)
    work["pred_rsrp"] = pd.to_numeric(work["pred_rsrp"], errors="coerce")
    work = work.dropna(subset=["grid_id", "pred_rsrp"])
    if work.empty:
        return pd.Series(dtype=float)
    return work.groupby("grid_id", dropna=False)["pred_rsrp"].mean()


def _render_rsrp_cdf(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    title: str = "Best Attempt RSRP CDF",
) -> None:
    if before_df.empty or after_df.empty or "pred_rsrp" not in before_df.columns or "pred_rsrp" not in after_df.columns:
        st.info("No before/after RSRP rows available for CDF.")
        return

    before_series = _grid_rsrp_series(before_df)
    after_series = _grid_rsrp_series(after_df)
    x_label = "Grid average RSRP (dBm)"

    before_x, before_y = _rsrp_cdf_values(before_series)
    after_x, after_y = _rsrp_cdf_values(after_series)
    if before_x.size == 0 or after_x.size == 0:
        st.info("No valid RSRP values available for CDF.")
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(before_x, before_y, color="#2563eb", linewidth=2.0, label=f"Before ({before_x.size:,})")
    ax.plot(after_x, after_y, color="#dc2626", linewidth=2.0, label=f"After ({after_x.size:,})")
    for threshold in [-90.0, -95.0, -100.0, -105.0]:
        ax.axvline(threshold, color="#64748b", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.text(threshold, 0.02, f"{threshold:.0f}", rotation=90, va="bottom", ha="right", fontsize=8, color="#475569")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("CDF")
    ax.grid(color="#cbd5e1", alpha=0.4, linewidth=0.6)
    ax.legend(loc="lower right")
    st.markdown("**Best Attempt Before/After RSRP CDF**")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    rows = []
    for label, values in [("before", before_series), ("after", after_series)]:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            continue
        row = {
            "state": label,
            "count": int(numeric.size),
            "mean_rsrp": float(numeric.mean()),
            "p05_rsrp": float(numeric.quantile(0.05)),
            "p50_rsrp": float(numeric.quantile(0.50)),
            "p95_rsrp": float(numeric.quantile(0.95)),
        }
        for threshold in [-90.0, -95.0, -100.0, -105.0]:
            row[f"count_lt_{int(abs(threshold))}"] = int((numeric < threshold).sum())
        rows.append(row)
    if rows:
        st.markdown("**CDF Threshold Counts**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _grid_state_by_cell(df: pd.DataFrame, rsrp_threshold: float) -> pd.DataFrame:
    required = {"Node_Cell_ID", "grid_id", "pred_rsrp"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    work = df.copy()
    work["Node_Cell_ID"] = work["Node_Cell_ID"].astype(str)
    work["grid_id"] = work["grid_id"].astype(str)
    work["pred_rsrp"] = pd.to_numeric(work["pred_rsrp"], errors="coerce")
    work = work.dropna(subset=["Node_Cell_ID", "grid_id", "pred_rsrp"])
    if work.empty:
        return pd.DataFrame()
    grid_state = (
        work.groupby(["Node_Cell_ID", "grid_id"], dropna=False)
        .agg(
            grid_avg_rsrp=("pred_rsrp", "mean"),
            sample_count=("pred_rsrp", "count"),
        )
        .reset_index()
    )
    grid_state["is_bad_grid"] = grid_state["grid_avg_rsrp"] < float(rsrp_threshold)
    return grid_state


def _grid_state_global(df: pd.DataFrame, rsrp_threshold: float) -> pd.DataFrame:
    required = {"grid_id", "pred_rsrp"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    work = df.copy()
    work["grid_id"] = work["grid_id"].astype(str)
    work["pred_rsrp"] = pd.to_numeric(work["pred_rsrp"], errors="coerce")
    work = work.dropna(subset=["grid_id", "pred_rsrp"])
    if work.empty:
        return pd.DataFrame()
    grid_state = (
        work.groupby("grid_id", dropna=False)
        .agg(
            grid_avg_rsrp=("pred_rsrp", "mean"),
            sample_count=("pred_rsrp", "count"),
            after_serving_cells=("Node_Cell_ID", lambda s: ",".join(sorted(set(s.astype(str)))) if "Node_Cell_ID" in work.columns else ""),
        )
        .reset_index()
    )
    grid_state["is_bad_grid"] = grid_state["grid_avg_rsrp"] < float(rsrp_threshold)
    return grid_state


def _build_per_cell_bad_grid_impact(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    rsrp_threshold: float,
) -> pd.DataFrame:
    # Attribute impact to the original/before serving cell footprint. If we group the
    # after side by after-serving cell, handover/serving-cell changes look like fake
    # bad-grid recovery for the original cell.
    before_grid = _grid_state_by_cell(before_df, rsrp_threshold)
    after_grid = _grid_state_global(after_df, rsrp_threshold)
    if before_grid.empty or after_grid.empty:
        return pd.DataFrame()

    merged = before_grid.merge(
        after_grid,
        on="grid_id",
        how="left",
        suffixes=("_before", "_after"),
    )
    for col in ["is_bad_grid_before", "is_bad_grid_after"]:
        merged[col] = merged.get(col, False)
        merged[col] = pd.Series(merged[col], index=merged.index).fillna(False).astype(bool)
    merged["grid_transition"] = np.select(
        [
            merged["is_bad_grid_before"] & ~merged["is_bad_grid_after"],
            ~merged["is_bad_grid_before"] & merged["is_bad_grid_after"],
            merged["is_bad_grid_before"] & merged["is_bad_grid_after"],
            ~merged["is_bad_grid_before"] & ~merged["is_bad_grid_after"],
        ],
        ["bad_to_good", "good_to_bad", "bad_to_bad", "good_to_good"],
        default="unknown",
    )
    merged["grid_rsrp_delta"] = pd.to_numeric(merged.get("grid_avg_rsrp_after"), errors="coerce") - pd.to_numeric(
        merged.get("grid_avg_rsrp_before"), errors="coerce"
    )

    cell_impact = (
        merged.groupby("Node_Cell_ID", dropna=False)
        .agg(
            footprint_grid_count=("grid_id", lambda s: int(s.notna().sum())),
            artifact_before_bad_grid_count=("is_bad_grid_before", "sum"),
            candidate_after_bad_grid_count=("is_bad_grid_after", "sum"),
            bad_to_good_grid_count=("grid_transition", lambda s: int((s == "bad_to_good").sum())),
            good_to_bad_grid_count=("grid_transition", lambda s: int((s == "good_to_bad").sum())),
            still_bad_grid_count=("grid_transition", lambda s: int((s == "bad_to_bad").sum())),
            still_good_grid_count=("grid_transition", lambda s: int((s == "good_to_good").sum())),
            missing_after_grid_count=("grid_avg_rsrp_after", lambda s: int(s.isna().sum())),
            artifact_mean_grid_rsrp_before=("grid_avg_rsrp_before", "mean"),
            candidate_mean_grid_rsrp_after=("grid_avg_rsrp_after", "mean"),
            candidate_mean_grid_rsrp_delta=("grid_rsrp_delta", "mean"),
            artifact_before_sample_count=("sample_count_before", "sum"),
            candidate_after_sample_count=("sample_count_after", "sum"),
        )
        .reset_index()
    )
    cell_impact["artifact_net_bad_grid_reduction"] = (
        cell_impact["artifact_before_bad_grid_count"] - cell_impact["candidate_after_bad_grid_count"]
    )
    cell_impact["before_bad_grid_count"] = cell_impact["artifact_before_bad_grid_count"]
    cell_impact["after_bad_grid_count"] = cell_impact["candidate_after_bad_grid_count"]
    cell_impact["net_bad_grid_reduction"] = cell_impact["artifact_net_bad_grid_reduction"]

    meta = _cell_metadata(antenna_df)
    if not meta.empty:
        cell_impact = cell_impact.merge(meta, on="Node_Cell_ID", how="left")
    return cell_impact


def _build_per_site_bad_grid_impact(cell_impact_df: pd.DataFrame) -> pd.DataFrame:
    if cell_impact_df.empty:
        return pd.DataFrame()
    work = cell_impact_df.copy()
    if "dashboard_site_id" in work.columns:
        site_col = "dashboard_site_id"
    elif "site" in work.columns:
        site_col = "site"
    else:
        work["site_id"] = work["Node_Cell_ID"].astype(str).str.split("_").str[0]
        site_col = "site_id"
    numeric_cols = [
        "footprint_grid_count",
        "artifact_before_bad_grid_count",
        "candidate_after_bad_grid_count",
        "artifact_net_bad_grid_reduction",
        "bad_to_good_grid_count",
        "good_to_bad_grid_count",
        "still_bad_grid_count",
        "still_good_grid_count",
        "artifact_before_sample_count",
        "candidate_after_sample_count",
    ]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    site_impact = (
        work.groupby(site_col, dropna=False)
        .agg(
            cell_count=("Node_Cell_ID", "nunique"),
            before_bad_grid_count=("before_bad_grid_count", "sum"),
            after_bad_grid_count=("after_bad_grid_count", "sum"),
            net_bad_grid_reduction=("net_bad_grid_reduction", "sum"),
            bad_to_good_grid_count=("bad_to_good_grid_count", "sum"),
            good_to_bad_grid_count=("good_to_bad_grid_count", "sum"),
            still_bad_grid_count=("still_bad_grid_count", "sum"),
            candidate_mean_grid_rsrp_delta=("candidate_mean_grid_rsrp_delta", "mean"),
            artifact_before_sample_count=("artifact_before_sample_count", "sum"),
            candidate_after_sample_count=("candidate_after_sample_count", "sum"),
        )
        .reset_index()
        .rename(columns={site_col: "site_id"})
    )
    return site_impact


def _render_per_cell_grid_impact(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    summary: Dict,
    best_summary: Dict,
) -> None:
    rsrp_threshold = float(summary.get("thresholds", {}).get("rsrp", -90.0))
    cell_impact = _build_per_cell_bad_grid_impact(before_df, after_df, antenna_df, rsrp_threshold)
    if cell_impact.empty:
        st.info("No per-cell bad grid impact could be calculated from this run.")
        return

    selected_updates = best_summary.get("selected_updates", []) if isinstance(best_summary, dict) else []
    changed_cells = {str(item.get("cell_id", "")).strip() for item in selected_updates}
    cell_impact["selected_changed_cell"] = cell_impact["Node_Cell_ID"].astype(str).isin(changed_cells)
    cell_impact["selected_candidate_score"] = best_summary.get("score") if isinstance(best_summary, dict) else np.nan
    cell_impact["selected_candidate_name"] = best_summary.get("candidate_name") if isinstance(best_summary, dict) else ""
    overall_before = pd.to_numeric(pd.Series([best_summary.get("baseline_bad_count") if isinstance(best_summary, dict) else np.nan]), errors="coerce").iloc[0]
    overall_after = pd.to_numeric(pd.Series([best_summary.get("candidate_bad_count") if isinstance(best_summary, dict) else np.nan]), errors="coerce").iloc[0]
    cell_impact["overall_before_bad_grid_count"] = overall_before
    cell_impact["overall_after_bad_grid_count"] = overall_after
    cell_impact["overall_net_bad_grid_reduction"] = overall_before - overall_after
    cell_impact["overall_improved"] = cell_impact["overall_net_bad_grid_reduction"] > 0
    if "electrical_tilt" in cell_impact.columns:
        cell_impact["before_etilt"] = pd.to_numeric(cell_impact["electrical_tilt"], errors="coerce")
    else:
        cell_impact["before_etilt"] = np.nan
    update_map = {
        str(item.get("cell_id", "")).strip(): item.get("target_value")
        for item in selected_updates
        if str(item.get("cell_id", "")).strip()
    }
    cell_impact["after_etilt"] = cell_impact["Node_Cell_ID"].astype(str).map(update_map)
    cell_impact["after_etilt"] = pd.to_numeric(cell_impact["after_etilt"], errors="coerce")
    cell_impact["after_etilt"] = cell_impact["after_etilt"].fillna(cell_impact["before_etilt"])
    cell_impact["etilt_delta"] = cell_impact["after_etilt"] - cell_impact["before_etilt"]

    st.markdown("**Per-Cell Bad Grid Impact**")
    st.caption(
        f"Cell-wise bad grid count grouped by `Node_Cell_ID + grid_id`. Bad grid means average RSRP < {rsrp_threshold:.0f} dBm."
    )
    cell_cols = [
        "Node_Cell_ID",
        "dashboard_site_id",
        "cell_id",
        "selected_changed_cell",
        "selected_candidate_name",
        "selected_candidate_score",
        "before_etilt",
        "after_etilt",
        "etilt_delta",
        "mechanical_tilt",
        "azimuth",
        "before_bad_grid_count",
        "after_bad_grid_count",
        "net_bad_grid_reduction",
        "overall_before_bad_grid_count",
        "overall_after_bad_grid_count",
        "overall_net_bad_grid_reduction",
        "overall_improved",
        "bad_to_good_grid_count",
        "good_to_bad_grid_count",
        "still_bad_grid_count",
        "artifact_mean_grid_rsrp_before",
        "candidate_mean_grid_rsrp_after",
    ]
    cell_cols = [col for col in cell_cols if col in cell_impact.columns]
    st.dataframe(
        cell_impact.sort_values(
            ["selected_changed_cell", "net_bad_grid_reduction", "before_bad_grid_count"],
            ascending=[False, False, False],
        )[cell_cols],
        use_container_width=True,
        hide_index=True,
    )

    site_impact = _build_per_site_bad_grid_impact(cell_impact)
    if not site_impact.empty:
        st.markdown("**Per-Site Bad Grid Impact**")
        site_cols = [
            "site_id",
            "cell_count",
            "before_bad_grid_count",
            "after_bad_grid_count",
            "net_bad_grid_reduction",
            "bad_to_good_grid_count",
            "good_to_bad_grid_count",
            "still_bad_grid_count",
        ]
        site_cols = [col for col in site_cols if col in site_impact.columns]
        st.dataframe(
            site_impact.sort_values(["net_bad_grid_reduction", "before_bad_grid_count"], ascending=[False, False])[site_cols],
            use_container_width=True,
            hide_index=True,
        )


def _render_transition_map(run_dir: Path, before_df: pd.DataFrame, after_df: pd.DataFrame, antenna_df: pd.DataFrame, summary: Dict) -> None:
    default_threshold = float(summary.get("thresholds", {}).get("rsrp", -90.0))
    threshold_options = sorted({default_threshold, -90.0, -95.0, -100.0, -105.0}, reverse=True)
    selected_threshold = st.selectbox(
        "RSRP bad-grid threshold for transition map",
        options=threshold_options,
        index=threshold_options.index(default_threshold) if default_threshold in threshold_options else 0,
        format_func=lambda value: f"{value:.0f} dBm",
        help="Map/counts are recomputed from saved before/after artifacts. No recommendation rerun is needed.",
    )
    sensitivity_df = _build_threshold_sensitivity_df(before_df, after_df, threshold_options)
    if not sensitivity_df.empty:
        st.markdown("**Best-Attempt Scope Threshold Counts**")
        st.caption(
            "Diagnostic only: recomputed from the before/after best-attempt scope. "
            "Use Trusted Baseline Grid RSRP Counts for real baseline totals."
        )
        st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)

    transition_df = _build_transition_df(before_df, after_df, float(selected_threshold))
    if transition_df.empty:
        st.info("No before/after transition rows available for map comparison.")
        return

    transition_counts = (
        transition_df["transition"]
        .value_counts(dropna=False)
        .rename_axis("transition")
        .reset_index(name="count")
    )
    label_map = {
        "bad_to_good": "Improved: bad -> good",
        "good_to_bad": "Worsened: good -> bad",
        "bad_to_bad": "Still bad",
        "good_to_good": "Still good",
        "unknown": "Unknown",
    }
    transition_counts["label"] = transition_counts["transition"].map(label_map).fillna(transition_counts["transition"])

    st.markdown(f"**Before/After Transition Counts @ {float(selected_threshold):.0f} dBm**")
    count_cols = st.columns(4)
    metric_map = {
        "bad_to_good": "Improved",
        "good_to_bad": "Worsened",
        "bad_to_bad": "Still Bad",
        "good_to_good": "Still Good",
    }
    for idx, key in enumerate(["bad_to_good", "good_to_bad", "bad_to_bad", "good_to_good"]):
        value = int(transition_counts.loc[transition_counts["transition"] == key, "count"].sum())
        count_cols[idx].metric(metric_map[key], value)
    st.dataframe(transition_counts[["label", "count"]], use_container_width=True, hide_index=True)

    x0, x1, y0, y1 = _build_fixed_extent(before_df, after_df, antenna_df)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f8fafc")
    ax.grid(color="#cbd5e1", alpha=0.35, linewidth=0.6)

    if not antenna_df.empty and {"lat", "lon"}.issubset(antenna_df.columns):
        site_points = antenna_df.copy()
        site_points["lat"] = pd.to_numeric(site_points["lat"], errors="coerce")
        site_points["lon"] = pd.to_numeric(site_points["lon"], errors="coerce")
        site_points = site_points.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["Node_Cell_ID"] if "Node_Cell_ID" in site_points.columns else ["lat", "lon"])
        ax.scatter(site_points["lon"], site_points["lat"], s=12, c="#111827", marker="^", alpha=0.5, label="Cells")

    color_map = {
        "bad_to_good": "#16a34a",
        "good_to_bad": "#dc2626",
        "bad_to_bad": "#f59e0b",
        "good_to_good": "#94a3b8",
    }
    order = ["bad_to_good", "good_to_bad", "bad_to_bad", "good_to_good"]
    for key in order:
        group = transition_df.loc[transition_df["transition"] == key].copy()
        if group.empty:
            continue
        group = group.dropna(subset=["lat", "lon"])
        if group.empty:
            continue
        ax.scatter(
            group["lon"],
            group["lat"],
            s=26 if key in {"bad_to_good", "good_to_bad"} else 16,
            c=color_map[key],
            alpha=0.85 if key in {"bad_to_good", "good_to_bad"} else 0.5,
            label=label_map[key],
        )

    ax.set_title(f"Grid-Level RSRP Threshold Transition Map @ {float(selected_threshold):.0f} dBm", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    st.markdown("**Grid-Level Transition Map: Where It Improved Or Worsened**")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    detail_cols = [
        col for col in [
            "grid_id",
            "lat",
            "lon",
            "grid_avg_rsrp_before",
            "grid_avg_rsrp_after",
            "rsrp_delta",
            "sample_count_before",
            "sample_count_after",
            "transition",
            "serving_cells_before",
            "serving_cells_after",
        ] if col in transition_df.columns
    ]
    if detail_cols:
        st.markdown("**Transition Detail Table**")
        st.dataframe(
            transition_df[detail_cols].sort_values(["transition", "rsrp_delta"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )


def _render_candidate_change_explainer(
    summary: Dict,
    best_summary: Dict,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
) -> None:
    selected_updates = best_summary.get("selected_updates", []) if isinstance(best_summary, dict) else []
    candidate_name = str(best_summary.get("candidate_name", "")) if isinstance(best_summary, dict) else ""
    site_id = str(best_summary.get("site_id", "")) if isinstance(best_summary, dict) else ""

    st.markdown("**Selected Recommendation Change**")
    if selected_updates:
        updates_df = pd.DataFrame(selected_updates)
        if not updates_df.empty:
            st.caption("These are the cells that were actually changed for the selected best candidate.")
            st.dataframe(updates_df, use_container_width=True, hide_index=True)
    else:
        st.info("No ETilt change was selected for the best candidate. The run stayed on hold.")

    st.caption(
        "Cell/site bad-grid impact is shown only in the source-separated `Per-Cell Bad Grid Impact` "
        "and `Per-Site Bad Grid Impact` sections above. The older serving-cell transition summary was removed "
        "because it duplicated confusing raw transition counts."
    )


def _best_candidate_overview(best_summary: Dict) -> pd.DataFrame:
    if not isinstance(best_summary, dict) or not best_summary:
        return pd.DataFrame()
    before_bad = pd.to_numeric(pd.Series([best_summary.get("baseline_bad_count")]), errors="coerce").iloc[0]
    after_bad = pd.to_numeric(pd.Series([best_summary.get("candidate_bad_count")]), errors="coerce").iloc[0]
    return pd.DataFrame(
        [
            {
                "site_id": best_summary.get("site_id"),
                "candidate_name": best_summary.get("candidate_name"),
                "score": best_summary.get("score"),
                "before_bad_grids": before_bad,
                "after_bad_grids": after_bad,
                "net_bad_grid_reduction": before_bad - after_bad,
                "mean_rsrp_delta": best_summary.get("mean_rsrp_delta"),
                "changed_cell_count": len(best_summary.get("selected_updates", []) or []),
            }
        ]
    )


def _selected_action_table(best_summary: Dict, antenna_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(best_summary, dict):
        return pd.DataFrame()
    updates = best_summary.get("selected_updates", []) or []
    if not updates:
        return pd.DataFrame()
    action_df = pd.DataFrame(updates)
    if action_df.empty or "cell_id" not in action_df.columns:
        return pd.DataFrame()
    action_df["cell_id"] = action_df["cell_id"].astype(str)
    action_df = action_df.rename(
        columns={
            "current_value": "before_etilt",
            "target_value": "after_etilt",
            "actual_delta": "etilt_delta",
            "requested_delta": "requested_etilt_delta",
        }
    )
    meta = _cell_metadata(antenna_df)
    if not meta.empty:
        meta = meta.drop(columns=["cell_id"], errors="ignore")
        meta = meta.rename(columns={"Node_Cell_ID": "cell_id", "electrical_tilt": "antenna_before_etilt"})
        meta = meta.loc[:, ~meta.columns.duplicated()].copy()
        meta["cell_id"] = meta["cell_id"].astype(str)
        keep_cols = [col for col in ["cell_id", "dashboard_site_id", "antenna_before_etilt", "mechanical_tilt", "azimuth", "band", "earfcn"] if col in meta.columns]
        action_df = action_df.merge(meta[keep_cols], on="cell_id", how="left")
    action_df["after_etilt"] = pd.to_numeric(action_df.get("after_etilt"), errors="coerce")
    action_df["before_etilt"] = pd.to_numeric(action_df.get("before_etilt"), errors="coerce")
    if "antenna_before_etilt" in action_df.columns:
        action_df["before_etilt"] = action_df["before_etilt"].fillna(pd.to_numeric(action_df["antenna_before_etilt"], errors="coerce"))
    action_df["etilt_delta"] = pd.to_numeric(action_df.get("etilt_delta"), errors="coerce")
    action_df["etilt_delta"] = action_df["etilt_delta"].fillna(action_df["after_etilt"] - action_df["before_etilt"])
    action_df["tilt_direction"] = np.select(
        [action_df["etilt_delta"] < 0, action_df["etilt_delta"] > 0],
        ["uptilt", "downtilt"],
        default="unchanged",
    )
    cols = [
        "cell_id",
        "dashboard_site_id",
        "before_etilt",
        "after_etilt",
        "etilt_delta",
        "tilt_direction",
        "requested_etilt_delta",
        "mechanical_tilt",
        "azimuth",
        "band",
        "earfcn",
    ]
    return action_df[[col for col in cols if col in action_df.columns]]


def _format_metric_value(value, decimals: int = 2) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    if float(numeric).is_integer():
        return f"{int(numeric):,}"
    return f"{float(numeric):,.{decimals}f}"


def _tilt_direction_counts(actions_df: pd.DataFrame) -> tuple[int, int, int]:
    if actions_df.empty or "etilt_delta" not in actions_df.columns:
        return 0, 0, 0
    delta = pd.to_numeric(actions_df["etilt_delta"], errors="coerce").fillna(0.0)
    uptilt = int((delta < 0).sum())
    downtilt = int((delta > 0).sum())
    unchanged = int((delta == 0).sum())
    return uptilt, downtilt, unchanged


def _format_runtime_minutes_seconds(value) -> str:
    seconds = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(seconds):
        return "-"
    total_seconds = int(round(float(seconds)))
    minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60
    return f"{minutes}min, {remaining_seconds}sec"


def _candidate_results_table(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()
    work = _add_candidate_change_columns(candidate_df)
    numeric_cols = [
        "score",
        "baseline_bad_count",
        "candidate_bad_count",
        "net_bad_reduction",
        "recovered_bad_samples",
        "new_bad_samples",
        "mean_rsrp_delta",
        "constraints_passed",
    ]
    for threshold_suffix in ["90", "95", "100", "105"]:
        numeric_cols.extend(
            [
                f"overall_before_bad_grid_count_lt_{threshold_suffix}",
                f"overall_after_bad_grid_count_lt_{threshold_suffix}",
                f"overall_net_bad_grid_reduction_lt_{threshold_suffix}",
                f"overall_bad_to_good_grid_count_lt_{threshold_suffix}",
                f"overall_good_to_bad_grid_count_lt_{threshold_suffix}",
            ]
        )
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    cols = [
        "site_id",
        "candidate_name",
        "score",
        "baseline_bad_count",
        "candidate_bad_count",
        "net_bad_reduction",
        "recovered_bad_samples",
        "new_bad_samples",
        "mean_rsrp_delta",
        "overall_before_bad_grid_count_lt_90",
        "overall_after_bad_grid_count_lt_90",
        "overall_net_bad_grid_reduction_lt_90",
        "overall_before_bad_grid_count_lt_95",
        "overall_after_bad_grid_count_lt_95",
        "overall_net_bad_grid_reduction_lt_95",
        "overall_before_bad_grid_count_lt_100",
        "overall_after_bad_grid_count_lt_100",
        "overall_net_bad_grid_reduction_lt_100",
        "overall_before_bad_grid_count_lt_105",
        "overall_after_bad_grid_count_lt_105",
        "overall_net_bad_grid_reduction_lt_105",
        "constraints_passed",
        "changed_cell_ids",
        "changed_cell_values",
    ]
    cols = [col for col in cols if col in work.columns]
    if not cols:
        return pd.DataFrame()
    sort_cols = [col for col in ["score", "net_bad_reduction", "mean_rsrp_delta"] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return work[cols]


def _best_attempt_candidate_row(candidate_df: pd.DataFrame) -> pd.Series | None:
    if candidate_df.empty or "target_value" not in candidate_df.columns:
        return None
    work = candidate_df.copy()
    work["_updates"] = work["target_value"].apply(_parse_update_payload)
    work = work.loc[work["_updates"].apply(bool)].copy()
    if "candidate_name" in work.columns:
        work = work.loc[work["candidate_name"].astype(str).str.strip().str.lower() != "hold"].copy()
    if work.empty:
        return None
    for col in [
        "score",
        "net_bad_reduction",
        "rsrp_severity_reduction",
        "recovered_bad_samples",
        "new_bad_samples",
        "mean_rsrp_delta",
        "constraints_passed",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["_recovered_minus_new"] = pd.to_numeric(work.get("recovered_bad_samples"), errors="coerce").fillna(0) - pd.to_numeric(
        work.get("new_bad_samples"), errors="coerce"
    ).fillna(0)
    sort_cols = [
        col for col in [
            "constraints_passed",
            "score",
            "net_bad_reduction",
            "rsrp_severity_reduction",
            "_recovered_minus_new",
            "mean_rsrp_delta",
        ] if col in work.columns
    ]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return work.iloc[0]


def _candidate_meta_from_row(candidate_row: pd.Series | None) -> Dict:
    if candidate_row is None:
        return {}
    updates = _parse_update_payload(candidate_row.get("target_value"))
    meta = {
        "site_id": str(candidate_row.get("site_id", "")),
        "candidate_name": str(candidate_row.get("candidate_name", "")),
        "score": float(pd.to_numeric(pd.Series([candidate_row.get("score")]), errors="coerce").fillna(np.nan).iloc[0]),
        "baseline_bad_count": float(pd.to_numeric(pd.Series([candidate_row.get("baseline_bad_count")]), errors="coerce").fillna(np.nan).iloc[0]),
        "candidate_bad_count": float(pd.to_numeric(pd.Series([candidate_row.get("candidate_bad_count")]), errors="coerce").fillna(np.nan).iloc[0]),
        "mean_rsrp_delta": float(pd.to_numeric(pd.Series([candidate_row.get("mean_rsrp_delta")]), errors="coerce").fillna(np.nan).iloc[0]),
        "constraints_passed": int(pd.to_numeric(pd.Series([candidate_row.get("constraints_passed")]), errors="coerce").fillna(0).iloc[0]),
        "selected_updates": updates,
        "reject_reason": str(candidate_row.get("reject_reason", "")),
        "net_bad_reduction": float(pd.to_numeric(pd.Series([candidate_row.get("net_bad_reduction")]), errors="coerce").fillna(np.nan).iloc[0]),
        "bad_to_good_grid_count": float(pd.to_numeric(pd.Series([candidate_row.get("recovered_bad_samples")]), errors="coerce").fillna(np.nan).iloc[0]),
        "good_to_bad_grid_count": float(pd.to_numeric(pd.Series([candidate_row.get("new_bad_samples")]), errors="coerce").fillna(np.nan).iloc[0]),
    }
    for threshold_suffix in ["90", "95", "100", "105"]:
        for prefix in ["before", "after", "net_bad_grid_reduction", "bad_to_good", "good_to_bad"]:
            if prefix == "before":
                source_col = f"overall_before_bad_grid_count_lt_{threshold_suffix}"
            elif prefix == "after":
                source_col = f"overall_after_bad_grid_count_lt_{threshold_suffix}"
            elif prefix == "net_bad_grid_reduction":
                source_col = f"overall_net_bad_grid_reduction_lt_{threshold_suffix}"
            elif prefix == "bad_to_good":
                source_col = f"overall_bad_to_good_grid_count_lt_{threshold_suffix}"
            else:
                source_col = f"overall_good_to_bad_grid_count_lt_{threshold_suffix}"
            if source_col in candidate_row.index:
                meta[source_col] = float(pd.to_numeric(pd.Series([candidate_row.get(source_col)]), errors="coerce").fillna(np.nan).iloc[0])
    if "overall_mean_rsrp_delta" in candidate_row.index:
        meta["overall_mean_rsrp_delta"] = float(pd.to_numeric(pd.Series([candidate_row.get("overall_mean_rsrp_delta")]), errors="coerce").fillna(np.nan).iloc[0])
    return meta


def _candidate_overall_threshold_table(candidate_meta: Dict) -> pd.DataFrame:
    if not candidate_meta:
        return pd.DataFrame()
    rows = []
    for threshold_suffix in ["90", "95", "100", "105"]:
        before = candidate_meta.get(f"overall_before_bad_grid_count_lt_{threshold_suffix}")
        after = candidate_meta.get(f"overall_after_bad_grid_count_lt_{threshold_suffix}")
        if pd.isna(before) or pd.isna(after):
            continue
        rows.append(
            {
                "threshold_dbm": -float(threshold_suffix),
                "before_bad_grids": int(before),
                "after_bad_grids": int(after),
                "net_bad_grid_reduction": int(candidate_meta.get(f"overall_net_bad_grid_reduction_lt_{threshold_suffix}", before - after)),
                "bad_to_good": int(candidate_meta.get(f"overall_bad_to_good_grid_count_lt_{threshold_suffix}", 0)),
                "good_to_bad": int(candidate_meta.get(f"overall_good_to_bad_grid_count_lt_{threshold_suffix}", 0)),
            }
        )
    return pd.DataFrame(rows)


def _build_attempt_action_table(candidate_meta: Dict, antenna_df: pd.DataFrame) -> pd.DataFrame:
    updates = candidate_meta.get("selected_updates", []) if isinstance(candidate_meta, dict) else []
    if not updates:
        return pd.DataFrame()
    rows = []
    for update in updates:
        cell_id = str(update.get("cell_id", "")).strip()
        parameter = str(update.get("parameter", "ETilt")).strip() or "ETilt"
        rows.append(
            {
                "cell_id": cell_id,
                "parameter": parameter,
                "after_value": update.get("target_value"),
            }
        )
    out = pd.DataFrame(rows)
    meta = _cell_metadata(antenna_df)
    if not meta.empty:
        meta = meta.drop(columns=["cell_id"], errors="ignore")
        meta = meta.rename(columns={"Node_Cell_ID": "cell_id", "electrical_tilt": "before_etilt", "azimuth": "before_azimuth"})
        meta = meta.loc[:, ~meta.columns.duplicated()].copy()
        meta["cell_id"] = meta["cell_id"].astype(str)
        keep_cols = [col for col in ["cell_id", "dashboard_site_id", "before_etilt", "before_azimuth", "mechanical_tilt", "band", "earfcn"] if col in meta.columns]
        out = out.merge(meta[keep_cols], on="cell_id", how="left")
    out["after_value"] = pd.to_numeric(out["after_value"], errors="coerce")
    out["before_etilt"] = pd.to_numeric(out.get("before_etilt"), errors="coerce")
    out["before_azimuth"] = pd.to_numeric(out.get("before_azimuth"), errors="coerce")
    out["before_value"] = np.where(out["parameter"].eq("Azimuth"), out["before_azimuth"], out["before_etilt"])
    out["delta"] = out["after_value"] - pd.to_numeric(out["before_value"], errors="coerce")
    out["tilt_direction"] = np.select(
        [out["parameter"].eq("ETilt") & (out["delta"] > 0), out["parameter"].eq("ETilt") & (out["delta"] < 0)],
        ["downtilt", "uptilt"],
        default=np.where(out["parameter"].eq("Azimuth"), "azimuth_change", "unchanged"),
    )
    cols = [
        "cell_id",
        "dashboard_site_id",
        "parameter",
        "before_value",
        "after_value",
        "delta",
        "tilt_direction",
        "before_etilt",
        "before_azimuth",
        "mechanical_tilt",
        "band",
        "earfcn",
    ]
    return out[[col for col in cols if col in out.columns]]


def _materialize_best_attempt_scope(
    summary: Dict,
    run_dir: Path,
    candidate_row: pd.Series | None,
    baseline_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    grid_analytics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if candidate_row is None or baseline_df.empty or antenna_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    candidate_meta = _candidate_meta_from_row(candidate_row)
    try:
        config = rsrp_test.TiltRsrpOnlyRecommendationTestConfig(
            project_id=int(summary.get("project_id")),
            region=str(summary.get("region", "india")),
            operator=summary.get("operator"),
            rsrp_threshold=float(summary.get("thresholds", {}).get("rsrp", -90.0)),
        )
        base_config = rsrp_test._config_as_base(config)
        baseline_job_id = rsrp_test.base._fetch_latest_baseline_job_id(config.project_id, config.region)
        before_scope, after_scope, updates, meta = rsrp_test._materialize_candidate_scope(
            baseline_df=baseline_df,
            antenna_df=antenna_df,
            candidate_row=candidate_row,
            config=base_config,
            baseline_job_id=baseline_job_id,
            grid_analytics_df=grid_analytics_df,
        )
        before_scope = rsrp_test._prepare_scope_export(before_scope, config.rsrp_threshold, "before_attempt")
        after_scope = rsrp_test._prepare_scope_export(after_scope, config.rsrp_threshold, "after_attempt")
        merged_meta = {**candidate_meta, **meta}
        if not merged_meta.get("selected_updates"):
            merged_meta["selected_updates"] = updates
        return before_scope, after_scope, merged_meta
    except Exception as exc:
        st.warning(f"Could not materialize best attempted candidate map: {type(exc).__name__}: {exc}")
        return pd.DataFrame(), pd.DataFrame(), candidate_meta


def _simple_cell_level_table(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    best_summary: Dict,
    rsrp_threshold: float,
) -> pd.DataFrame:
    if before_df.empty or after_df.empty or "Node_Cell_ID" not in before_df.columns or "Node_Cell_ID" not in after_df.columns:
        return pd.DataFrame()

    before = before_df.copy()
    after = after_df.copy()
    before["Node_Cell_ID"] = before["Node_Cell_ID"].astype(str)
    after["Node_Cell_ID"] = after["Node_Cell_ID"].astype(str)
    before["pred_rsrp"] = pd.to_numeric(before.get("pred_rsrp"), errors="coerce")
    after["pred_rsrp"] = pd.to_numeric(after.get("pred_rsrp"), errors="coerce")
    before = before.dropna(subset=["Node_Cell_ID", "pred_rsrp"])
    after = after.dropna(subset=["Node_Cell_ID", "pred_rsrp"])
    if before.empty or after.empty:
        return pd.DataFrame()

    before_summary = (
        before.groupby("Node_Cell_ID", dropna=False)
        .agg(
            before_sample_count=("pred_rsrp", "count"),
            before_bad_sample_count=("pred_rsrp", lambda s: int((s < float(rsrp_threshold)).sum())),
            before_avg_rsrp=("pred_rsrp", "mean"),
        )
        .reset_index()
    )
    after_summary = (
        after.groupby("Node_Cell_ID", dropna=False)
        .agg(
            after_sample_count=("pred_rsrp", "count"),
            after_bad_sample_count=("pred_rsrp", lambda s: int((s < float(rsrp_threshold)).sum())),
            after_avg_rsrp=("pred_rsrp", "mean"),
        )
        .reset_index()
    )
    out = before_summary.merge(after_summary, on="Node_Cell_ID", how="outer")

    meta = _cell_metadata(antenna_df)
    if not meta.empty:
        meta = meta.drop(columns=["cell_id"], errors="ignore").rename(
            columns={"Node_Cell_ID": "cell_id", "electrical_tilt": "before_etilt"}
        )
        meta = meta.loc[:, ~meta.columns.duplicated()].copy()
        meta["cell_id"] = meta["cell_id"].astype(str)
        out = out.rename(columns={"Node_Cell_ID": "cell_id"})
        keep_cols = [col for col in ["cell_id", "dashboard_site_id", "before_etilt", "mechanical_tilt", "azimuth"] if col in meta.columns]
        out = out.merge(meta[keep_cols], on="cell_id", how="left")
    else:
        out = out.rename(columns={"Node_Cell_ID": "cell_id"})
        out["before_etilt"] = np.nan

    updates = best_summary.get("selected_updates", []) if isinstance(best_summary, dict) else []
    update_map = {
        str(item.get("cell_id", "")).strip(): item.get("target_value")
        for item in updates
        if str(item.get("cell_id", "")).strip()
    }
    out["after_etilt"] = out["cell_id"].astype(str).map(update_map)
    out["before_etilt"] = pd.to_numeric(out.get("before_etilt"), errors="coerce")
    out["after_etilt"] = pd.to_numeric(out["after_etilt"], errors="coerce").fillna(out["before_etilt"])
    out["etilt_delta"] = out["after_etilt"] - out["before_etilt"]

    for col in ["before_bad_sample_count", "after_bad_sample_count", "before_avg_rsrp", "after_avg_rsrp"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out["bad_sample_delta"] = out["after_bad_sample_count"] - out["before_bad_sample_count"]
    out["bad_sample_reduction"] = out["before_bad_sample_count"] - out["after_bad_sample_count"]
    out["avg_rsrp_delta"] = out["after_avg_rsrp"] - out["before_avg_rsrp"]
    out["selected_changed_cell"] = out["etilt_delta"].fillna(0).abs() > 0

    cols = [
        "cell_id",
        "dashboard_site_id",
        "selected_changed_cell",
        "before_etilt",
        "after_etilt",
        "etilt_delta",
        "before_bad_sample_count",
        "after_bad_sample_count",
        "bad_sample_reduction",
        "before_avg_rsrp",
        "after_avg_rsrp",
        "avg_rsrp_delta",
        "before_sample_count",
        "after_sample_count",
        "mechanical_tilt",
        "azimuth",
    ]
    cols = [col for col in cols if col in out.columns]
    return out[cols].sort_values(
        ["selected_changed_cell", "bad_sample_reduction", "before_bad_sample_count"],
        ascending=[False, False, False],
    )


def _greedy_debug_table(candidate_df: pd.DataFrame, antenna_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()

    meta = _cell_metadata(antenna_df)
    if not meta.empty:
        meta = meta.drop(columns=["cell_id"], errors="ignore").rename(
            columns={"Node_Cell_ID": "changed_cell_id", "electrical_tilt": "before_etilt"}
        )
        meta = meta.loc[:, ~meta.columns.duplicated()].copy()
        meta["changed_cell_id"] = meta["changed_cell_id"].astype(str)

    rows: List[Dict] = []
    for _, row in candidate_df.iterrows():
        updates = _parse_update_payload(row.get("target_value"))
        local_metrics = _parse_update_payload(row.get("changed_cell_local_metrics"))
        local_by_cell = {
            str(item.get("cell_id", "")).strip(): item
            for item in local_metrics
            if str(item.get("cell_id", "")).strip()
        }
        stage = str(row.get("selection_stage", ""))
        if str(row.get("candidate_name", "")).strip().lower() == "hold":
            rows.append(
                {
                    "step_type": "hold",
                    "site_id": row.get("site_id"),
                    "candidate_name": row.get("candidate_name"),
                    "changed_cell_id": "",
                    "after_etilt": np.nan,
                    "changed_cell_count": 0,
                    "scope_before_bad_grids": row.get("baseline_bad_count"),
                    "scope_after_bad_grids": row.get("candidate_bad_count"),
                    "scope_net_bad_reduction": row.get("net_bad_reduction"),
                    "scope_bad_to_good": row.get("recovered_bad_samples"),
                    "scope_good_to_bad": row.get("new_bad_samples"),
                    "mean_rsrp_delta": row.get("mean_rsrp_delta"),
                    "score": row.get("score"),
                    "constraints_passed": row.get("constraints_passed"),
                    "decision": "baseline",
                }
            )
            continue
        if not updates:
            continue
        changed_ids = [str(item.get("cell_id", "")).strip() for item in updates if str(item.get("cell_id", "")).strip()]
        step_type = "greedy_accumulated" if "greedy" in stage.lower() else "single_cell_try"
        decision = "accepted" if step_type == "greedy_accumulated" and float(pd.to_numeric(pd.Series([row.get("net_bad_reduction")]), errors="coerce").fillna(0).iloc[0]) > 0 else "tested"
        for update in updates:
            changed_cell_id = str(update.get("cell_id", "")).strip()
            if not changed_cell_id:
                continue
            local = local_by_cell.get(changed_cell_id, {})
            rows.append(
                {
                    "step_type": step_type,
                    "site_id": row.get("site_id"),
                    "candidate_name": row.get("candidate_name"),
                    "changed_cell_id": changed_cell_id,
                    "after_etilt": update.get("target_value"),
                    "changed_cell_count": len(changed_ids),
                    "all_changed_cells": ",".join(changed_ids),
                    "cell_before_bad_samples": local.get("before_bad_sample_count"),
                    "cell_after_bad_samples": local.get("after_bad_sample_count"),
                    "cell_bad_sample_reduction": local.get("bad_sample_reduction"),
                    "cell_before_avg_rsrp": local.get("before_avg_rsrp"),
                    "cell_after_avg_rsrp": local.get("after_avg_rsrp"),
                    "cell_avg_rsrp_delta": local.get("avg_rsrp_delta"),
                    "scope_before_bad_grids": row.get("baseline_bad_count"),
                    "scope_after_bad_grids": row.get("candidate_bad_count"),
                    "scope_net_bad_reduction": row.get("net_bad_reduction"),
                    "scope_bad_to_good": row.get("recovered_bad_samples"),
                    "scope_good_to_bad": row.get("new_bad_samples"),
                    "mean_rsrp_delta": row.get("mean_rsrp_delta"),
                    "score": row.get("score"),
                    "constraints_passed": row.get("constraints_passed"),
                    "decision": decision,
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if not meta.empty:
        keep_cols = [col for col in ["changed_cell_id", "dashboard_site_id", "before_etilt", "mechanical_tilt", "azimuth"] if col in meta.columns]
        out = out.merge(meta[keep_cols], on="changed_cell_id", how="left")
    out["after_etilt"] = pd.to_numeric(out.get("after_etilt"), errors="coerce")
    out["before_etilt"] = pd.to_numeric(out.get("before_etilt"), errors="coerce")
    out["etilt_delta"] = out["after_etilt"] - out["before_etilt"]
    for col in [
        "cell_before_bad_samples",
        "cell_after_bad_samples",
        "cell_bad_sample_reduction",
        "cell_before_avg_rsrp",
        "cell_after_avg_rsrp",
        "cell_avg_rsrp_delta",
        "scope_before_bad_grids",
        "scope_after_bad_grids",
        "scope_net_bad_reduction",
        "scope_bad_to_good",
        "scope_good_to_bad",
        "mean_rsrp_delta",
        "score",
    ]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    cols = [
        "step_type",
        "decision",
        "site_id",
        "changed_cell_id",
        "dashboard_site_id",
        "before_etilt",
        "after_etilt",
        "etilt_delta",
        "changed_cell_count",
        "all_changed_cells",
        "cell_before_bad_samples",
        "cell_after_bad_samples",
        "cell_bad_sample_reduction",
        "cell_before_avg_rsrp",
        "cell_after_avg_rsrp",
        "cell_avg_rsrp_delta",
        "scope_before_bad_grids",
        "scope_after_bad_grids",
        "scope_net_bad_reduction",
        "scope_bad_to_good",
        "scope_good_to_bad",
        "mean_rsrp_delta",
        "score",
        "constraints_passed",
        "candidate_name",
    ]
    cols = [col for col in cols if col in out.columns]
    local_metric_cols = [
        "cell_before_bad_samples",
        "cell_after_bad_samples",
        "cell_bad_sample_reduction",
        "cell_before_avg_rsrp",
        "cell_after_avg_rsrp",
        "cell_avg_rsrp_delta",
    ]
    for col in local_metric_cols:
        if col in cols and out[col].isna().all():
            cols.remove(col)
    return out[cols].sort_values(
        ["step_type", "score", "scope_net_bad_reduction"],
        ascending=[True, False, False],
    )


def main() -> None:
    st.set_page_config(page_title="Tilt RSRP Only Dashboard", layout="wide")
    st.title("Tilt RSRP Only Dashboard")
    available_project_ids = _available_project_ids()
    default_project_id = available_project_ids[0] if available_project_ids else 1
    if available_project_ids:
        project_id = st.sidebar.selectbox(
            "Project ID",
            options=available_project_ids,
            index=0,
        )
    else:
        project_id = st.sidebar.number_input("Project ID", value=default_project_id, step=1, min_value=1)

    runs = _list_runs(int(project_id))
    if not runs:
        st.info(
            f"No tilt RSRP only runs found for project {int(project_id)} yet. "
            "Run `tests/lte_tilt_rsrp_only_recommendation_test.py` first or pick a different project ID."
        )
        return

    run_names = [run.name for run in runs]
    selected_run_name = st.selectbox(
        "Available Runs",
        run_names,
        index=0,
        key="tilt_rsrp_only_run_select",
    )
    run_dir = next(run for run in runs if run.name == selected_run_name)
    summary = _load_summary(run_dir)
    reco_df = _safe_read_csv(run_dir / "recommendations.csv")
    candidate_df = _safe_read_csv(run_dir / "candidate_validation_results.csv")
    before_df = _safe_read_csv(run_dir / "best_candidate_before_scope.csv")
    after_df = _safe_read_csv(run_dir / "best_candidate_after_scope.csv")
    baseline_df = _safe_read_csv(run_dir / "baseline_log_input.csv")
    grid_analytics_df = _safe_read_csv(run_dir / "grid_analytics_input.csv")
    baseline_grid_metrics_df = _safe_read_csv(run_dir / "baseline_grid_metrics.csv")
    antenna_df = _safe_read_csv(run_dir / "antenna_input.csv")
    best_summary_path = run_dir / "best_candidate_summary.json"
    best_summary = json.loads(best_summary_path.read_text(encoding="utf-8")) if best_summary_path.exists() else {}
    if best_summary and not candidate_df.empty and "candidate_name" in candidate_df.columns:
        candidate_name = str(best_summary.get("candidate_name", ""))
        selected_candidate_rows = candidate_df.loc[candidate_df["candidate_name"].astype(str) == candidate_name]
        if not selected_candidate_rows.empty:
            selected_candidate_row = selected_candidate_rows.iloc[0]
            for col in ["changed_cell_avg_rsrp_delta_mean", "changed_cell_bad_sample_reduction_sum"]:
                if col in selected_candidate_row.index and col not in best_summary:
                    value = pd.to_numeric(pd.Series([selected_candidate_row.get(col)]), errors="coerce").iloc[0]
                    if pd.notna(value):
                        best_summary[col] = float(value)

    selected_actions = _selected_action_table(best_summary, antenna_df)
    uptilt_count, downtilt_count, unchanged_count = _tilt_direction_counts(selected_actions)
    counts = summary.get("counts", {})

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Score", _format_metric_value(best_summary.get("score"), 4))
    c2.metric("Before Bad Grids", _format_metric_value(best_summary.get("baseline_bad_count")))
    c3.metric("After Bad Grids", _format_metric_value(best_summary.get("candidate_bad_count")))
    c4.metric("Grid Reduction", _format_metric_value(float(best_summary.get("baseline_bad_count", 0) or 0) - float(best_summary.get("candidate_bad_count", 0) or 0)))
    c5.metric("Overall RSRP Delta", f"{_format_metric_value(best_summary.get('mean_rsrp_delta'), 3)} dB")
    c6.metric("Runtime", _format_runtime_minutes_seconds(summary.get("total_runtime_sec")))

    changed_cell_delta = best_summary.get("changed_cell_avg_rsrp_delta_mean")
    c7, c8, c9, c10, c11 = st.columns(5)
    c7.metric("Tilt Changes", len(best_summary.get("selected_updates", []) or []))
    c8.metric("Uptilt", uptilt_count)
    c9.metric("Downtilt", downtilt_count)
    c10.metric("Changed-Cell RSRP Delta", f"{_format_metric_value(changed_cell_delta, 3)} dB")
    c11.metric("Bad Samples", _format_metric_value(counts.get("bad_samples")))

    best_overview = _best_candidate_overview(best_summary)
    if not best_overview.empty:
        st.markdown("**Best Score Recommendation**")
        st.dataframe(best_overview, use_container_width=True, hide_index=True)

    st.markdown("**Tilt Applied**")
    if selected_actions.empty:
        st.info("No ETilt action selected. Best result is HOLD.")
    else:
        st.dataframe(selected_actions, use_container_width=True, hide_index=True)

    trusted_grid_df = grid_analytics_df if not grid_analytics_df.empty else baseline_grid_metrics_df
    bin_counts = _render_binned_rsrp_maps(before_df, after_df, antenna_df, best_summary, trusted_grid_df)
    if not bin_counts.empty:
        st.markdown("**Map Range Counts**")
        st.dataframe(bin_counts, use_container_width=True, hide_index=True)

    _render_rsrp_cdf(before_df, after_df, title="Best Score Recommendation RSRP CDF")

    threshold = float(summary.get("thresholds", {}).get("rsrp", -90.0))
    transition_df = _build_transition_df(before_df, after_df, threshold)
    if not transition_df.empty:
        transition_counts = transition_df["transition"].value_counts(dropna=False)
        grid_change_df = pd.DataFrame(
            [
                {
                    "threshold_dbm": threshold,
                    "bad_to_good": int(transition_counts.get("bad_to_good", 0)),
                    "good_to_bad": int(transition_counts.get("good_to_bad", 0)),
                    "still_bad": int(transition_counts.get("bad_to_bad", 0)),
                    "still_good": int(transition_counts.get("good_to_good", 0)),
                    "mean_grid_rsrp_delta": float(pd.to_numeric(transition_df.get("rsrp_delta"), errors="coerce").mean()),
                }
            ]
        )
        st.markdown("**Grid Changes**")
        st.dataframe(grid_change_df, use_container_width=True, hide_index=True)

    cell_summary = _simple_cell_level_table(before_df, after_df, antenna_df, best_summary, threshold)
    if not cell_summary.empty:
        keep_cols = [
            "cell_id",
            "dashboard_site_id",
            "selected_changed_cell",
            "before_etilt",
            "after_etilt",
            "etilt_delta",
            "before_bad_sample_count",
            "after_bad_sample_count",
            "bad_sample_reduction",
            "before_avg_rsrp",
            "after_avg_rsrp",
            "avg_rsrp_delta",
        ]
        keep_cols = [col for col in keep_cols if col in cell_summary.columns]
        st.markdown("**Cell Summary**")
        st.dataframe(cell_summary[keep_cols].head(80), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
