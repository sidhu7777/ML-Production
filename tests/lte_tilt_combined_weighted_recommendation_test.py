from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import lte_tilt_rsrp_only_recommendation_test as fast_runtime


DEFAULT_RSRP_THRESHOLD = -90.0
DEFAULT_RSRQ_THRESHOLD = -14.0
DEFAULT_SINR_THRESHOLD = 6.0


@dataclass(frozen=True)
class KpiWeights:
    rsrp: float
    rsrq: float
    sinr: float


_ACTIVE_THRESHOLDS = {
    "rsrp": DEFAULT_RSRP_THRESHOLD,
    "rsrq": DEFAULT_RSRQ_THRESHOLD,
    "sinr": DEFAULT_SINR_THRESHOLD,
}
_ACTIVE_WEIGHTS = KpiWeights(rsrp=0.60, rsrq=0.20, sinr=0.20)

_SEVERITY_CAPS = {
    "rsrp": 25.0,
    "rsrq": 8.0,
    "sinr": 15.0,
}
_MEAN_DELTA_SCALES = {
    "rsrp": 10.0,
    "rsrq": 3.0,
    "sinr": 10.0,
}


def _normalise_weights(rsrp: float, rsrq: float, sinr: float) -> KpiWeights:
    raw = [float(rsrp), float(rsrq), float(sinr)]
    if any(value < 0.0 for value in raw):
        raise ValueError("KPI weights must be zero or positive.")
    total = sum(raw)
    if total <= 0.0:
        raise ValueError("At least one KPI weight must be greater than zero.")
    return KpiWeights(*(value / total for value in raw))


def _active_threshold(config, kpi: str) -> float:
    return float(_ACTIVE_THRESHOLDS[kpi])


def _severity(values: pd.Series, threshold: float) -> pd.Series:
    return (float(threshold) - pd.to_numeric(values, errors="coerce")).clip(lower=0.0).fillna(0.0)


def _safe_float(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else float(default)


def _normalised_component(metrics: Dict[str, object], kpi: str) -> float:
    severity_term = _safe_float(metrics.get(f"{kpi}_severity_reduction_per_sample")) / _SEVERITY_CAPS[kpi]
    recovery_term = _safe_float(metrics.get(f"{kpi}_recovered_bad_share")) - _safe_float(metrics.get(f"{kpi}_new_bad_share"))
    mean_delta_term = _safe_float(metrics.get(f"mean_{kpi}_delta")) / _MEAN_DELTA_SCALES[kpi]
    return float(severity_term + recovery_term + mean_delta_term)


def _frontend_grid_component(metrics: Dict[str, object], kpi: str) -> float:
    severity_term = _safe_float(metrics.get(f"{kpi}_severity_reduction_per_grid")) / _SEVERITY_CAPS[kpi]
    recovery_term = _safe_float(metrics.get(f"{kpi}_grid_recovered_share")) - _safe_float(metrics.get(f"{kpi}_grid_new_bad_share"))
    mean_delta_term = _safe_float(metrics.get(f"mean_{kpi}_delta")) / _MEAN_DELTA_SCALES[kpi]
    return float(severity_term + recovery_term + mean_delta_term)


def _weighted_bad_mask(frame: pd.DataFrame, prefix: str = "avg") -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for kpi, weight in [("rsrp", _ACTIVE_WEIGHTS.rsrp), ("rsrq", _ACTIVE_WEIGHTS.rsrq), ("sinr", _ACTIVE_WEIGHTS.sinr)]:
        if weight <= 0.0:
            continue
        col = f"{prefix}_{kpi}"
        if col in frame.columns:
            mask = mask | (pd.to_numeric(frame[col], errors="coerce") < _ACTIVE_THRESHOLDS[kpi])
    return mask.fillna(False)


def _weighted_severity(frame: pd.DataFrame, prefix: str = "avg") -> pd.Series:
    total = pd.Series(0.0, index=frame.index)
    for kpi, weight in [("rsrp", _ACTIVE_WEIGHTS.rsrp), ("rsrq", _ACTIVE_WEIGHTS.rsrq), ("sinr", _ACTIVE_WEIGHTS.sinr)]:
        col = f"{prefix}_{kpi}"
        if weight > 0.0 and col in frame.columns:
            total = total + _severity(frame[col], _ACTIVE_THRESHOLDS[kpi]) * float(weight)
    return total.fillna(0.0)


def _kpi_improves_with_higher_value(kpi: str) -> bool:
    return kpi in {"rsrp", "rsrq", "sinr"}


def _add_combined_grid_fields(out: pd.DataFrame) -> pd.DataFrame:
    out["is_bad_combined"] = _weighted_bad_mask(out)
    out["combined_severity"] = _weighted_severity(out)
    out["combined_score_basis"] = out["combined_severity"]
    # Compatibility aliases required by the imported fast coordinate-search
    # runtime. Combined-owned reports must use the native combined columns.
    out["is_bad_rsrp"] = out["is_bad_combined"]
    out["rsrp_severity"] = out["combined_severity"]
    return out


def _weighted_kpi_cell_order(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    total_cells = int(len(summary))
    kpi_weights = [("rsrp", _ACTIVE_WEIGHTS.rsrp), ("rsrq", _ACTIVE_WEIGHTS.rsrq), ("sinr", _ACTIVE_WEIGHTS.sinr)]
    active = [(kpi, float(weight)) for kpi, weight in kpi_weights if float(weight) > 0.0]
    if not active:
        return summary
    max_weight = max(weight for _, weight in active)
    priority_kpis = [kpi for kpi, weight in active if weight == max_weight]

    if len(priority_kpis) == 1:
        kpi = priority_kpis[0]
        bad_col = f"Bad {kpi.upper()}"
        sev_col = f"{kpi}_bad_severity"
        priority = summary.loc[pd.to_numeric(summary.get(bad_col), errors="coerce").fillna(0) > 0].copy()
        if priority.empty:
            return priority
        priority["_selection_kpi"] = kpi.upper()
        priority["_selection_weight"] = float(max_weight)
        priority = priority.sort_values(
            [bad_col, sev_col, "Bad Grid Count", "combined_grid_severity", "Cell ID"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
        priority["_selection_rank"] = np.arange(1, len(priority) + 1)
        return priority

    selected: list[pd.DataFrame] = []
    selected_cells: set[str] = set()
    remaining_slots = total_cells
    for index, (kpi, weight) in enumerate(active):
        if remaining_slots <= 0:
            break
        if index == len(active) - 1:
            budget = remaining_slots
        else:
            budget = int(round(total_cells * weight))
            budget = max(1, min(budget, remaining_slots))
        bad_col = f"Bad {kpi.upper()}"
        sev_col = f"{kpi}_bad_severity"
        candidates = summary.loc[
            (~summary["Cell ID"].astype(str).isin(selected_cells))
            & (pd.to_numeric(summary.get(bad_col), errors="coerce").fillna(0) > 0)
        ].copy()
        if candidates.empty:
            continue
        candidates["_selection_kpi"] = kpi.upper()
        candidates["_selection_weight"] = weight
        candidates = candidates.sort_values(
            [bad_col, sev_col, "Bad Grid Count", "combined_grid_severity", "Cell ID"],
            ascending=[False, False, False, False, True],
        ).head(budget)
        selected.append(candidates)
        selected_cells.update(candidates["Cell ID"].astype(str).tolist())
        remaining_slots = total_cells - len(selected_cells)

    if len(selected_cells) < total_cells:
        fill = summary.loc[~summary["Cell ID"].astype(str).isin(selected_cells)].copy()
        if not fill.empty:
            fill["_selection_kpi"] = "COMBINED_FILL"
            fill["_selection_weight"] = 0.0
            fill = fill.sort_values(
                ["combined_grid_severity", "Bad Grid Count", "Bad Samples", "Cell ID"],
                ascending=[False, False, False, True],
            )
            selected.append(fill)

    if not selected:
        return summary
    out = pd.concat(selected, ignore_index=True)
    out["_selection_rank"] = np.arange(1, len(out) + 1)
    return out.drop_duplicates(subset=["Cell ID"], keep="first").reset_index(drop=True)


def _filter_bad_samples_combined(log_df: pd.DataFrame, allowed_techs) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = log_df.copy()
    if "Cell ID" not in work.columns and "Node_Cell_ID" in work.columns:
        work["Cell ID"] = work["Node_Cell_ID"].astype(str)
    if "Technology" not in work.columns:
        work["Technology"] = "4G"
    tech_mask = work["Technology"].astype(str).isin(allowed_techs) if allowed_techs else pd.Series(True, index=work.index)
    for kpi in ["rsrp", "rsrq", "sinr"]:
        pred_col = f"pred_{kpi}"
        bad_col = f"Bad {kpi.upper()}"
        if pred_col in work.columns and getattr(_ACTIVE_WEIGHTS, kpi) > 0.0:
            work[bad_col] = tech_mask & (pd.to_numeric(work[pred_col], errors="coerce") < _ACTIVE_THRESHOLDS[kpi])
        else:
            work[bad_col] = False
    bad_mask = work["Bad RSRP"] | work["Bad RSRQ"] | work["Bad SINR"]
    bad_df = work.loc[bad_mask].copy()
    if bad_df.empty:
        return bad_df, pd.DataFrame(columns=["Cell ID", "Technology", "Bad RSRP", "Bad RSRQ", "Bad SINR"])
    summary = (
        bad_df.groupby(["Cell ID", "Technology"], dropna=False)
        .agg(**{"Bad RSRP": ("Bad RSRP", "sum"), "Bad RSRQ": ("Bad RSRQ", "sum"), "Bad SINR": ("Bad SINR", "sum")})
        .reset_index()
    )
    summary["Bad Samples"] = summary[["Bad RSRP", "Bad RSRQ", "Bad SINR"]].sum(axis=1)
    return bad_df, summary


def _aggregate_grid_metrics_combined(pred_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if pred_df.empty or "grid_id" not in pred_df.columns:
        return pd.DataFrame()
    work = pred_df.copy()
    work["grid_id"] = fast_runtime._normalize_grid_id_series(work["grid_id"])
    agg: Dict[str, tuple] = {"point_count": ("grid_id", "count")}
    for kpi in ["rsrp", "rsrq", "sinr"]:
        col = f"pred_{kpi}"
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            agg[f"avg_{kpi}"] = (col, "mean")
    for col in ["lat", "lon"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            agg[col] = (col, "mean")
    if "Node_Cell_ID" in work.columns:
        agg["distinct_cells"] = ("Node_Cell_ID", "nunique")
    out = work.dropna(subset=["grid_id"]).groupby("grid_id", dropna=False).agg(**agg).reset_index()
    for kpi in ["rsrp", "rsrq", "sinr"]:
        if f"avg_{kpi}" not in out.columns:
            out[f"avg_{kpi}"] = np.nan
        out[f"is_bad_{kpi}"] = (
            pd.to_numeric(out[f"avg_{kpi}"], errors="coerce") < _ACTIVE_THRESHOLDS[kpi]
            if getattr(_ACTIVE_WEIGHTS, kpi) > 0.0
            else False
        )
        out[f"is_bad_{kpi}_kpi"] = out[f"is_bad_{kpi}"]
        out[f"{kpi}_severity"] = _severity(out[f"avg_{kpi}"], _ACTIVE_THRESHOLDS[kpi])
    out["weighted_bad_severity"] = _weighted_severity(out)
    return _add_combined_grid_fields(out)


def _grid_reference_metrics_combined(grid_analytics_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if grid_analytics_df.empty or "grid_id" not in grid_analytics_df.columns:
        return pd.DataFrame()
    work = grid_analytics_df.copy()
    work["grid_id"] = fast_runtime._normalize_grid_id_series(work["grid_id"])
    for kpi in ["rsrp", "rsrq", "sinr"]:
        source = f"baseline_avg_{kpi}"
        work[f"avg_{kpi}"] = pd.to_numeric(work[source], errors="coerce") if source in work.columns else np.nan
    work["point_count"] = pd.to_numeric(work.get("baseline_point_count"), errors="coerce") if "baseline_point_count" in work.columns else np.nan
    keep_cols = [
        c
        for c in ["grid_id", "avg_rsrp", "avg_rsrq", "avg_sinr", "point_count", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]
        if c in work.columns
    ]
    out = work.loc[work["grid_id"].notna(), keep_cols].drop_duplicates(subset=["grid_id"], keep="first").copy()
    for kpi in ["rsrp", "rsrq", "sinr"]:
        out[f"is_bad_{kpi}"] = (
            pd.to_numeric(out[f"avg_{kpi}"], errors="coerce") < _ACTIVE_THRESHOLDS[kpi]
            if getattr(_ACTIVE_WEIGHTS, kpi) > 0.0
            else False
        )
        out[f"is_bad_{kpi}_kpi"] = out[f"is_bad_{kpi}"]
        out[f"{kpi}_severity"] = _severity(out[f"avg_{kpi}"], _ACTIVE_THRESHOLDS[kpi])
    out["weighted_bad_severity"] = _weighted_severity(out)
    return _add_combined_grid_fields(out)


def _build_grid_ranked_summary_combined(
    baseline_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    threshold: float,
    grid_analytics_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_metrics = _grid_reference_metrics_combined(grid_analytics_df, threshold) if grid_analytics_df is not None else pd.DataFrame()
    if grid_metrics.empty:
        grid_metrics = _aggregate_grid_metrics_combined(baseline_df, threshold)
    if grid_metrics.empty:
        return pd.DataFrame(), grid_metrics

    work = baseline_df.copy()
    work["grid_id"] = fast_runtime._normalize_grid_id_series(work.get("grid_id"))
    work = work.loc[work["grid_id"].notna()].copy()
    if work.empty or "Node_Cell_ID" not in work.columns:
        return pd.DataFrame(), grid_metrics
    work["Cell ID"] = work["Node_Cell_ID"].astype(str)

    cell_rows: Dict[str, Dict[str, object]] = {}
    cell_grid_sets: Dict[str, set[str]] = {}
    for kpi in ["rsrp", "rsrq", "sinr"]:
        weight = float(getattr(_ACTIVE_WEIGHTS, kpi))
        bad_col = f"Bad {kpi.upper()}"
        sev_col = f"{kpi}_bad_severity"
        if weight <= 0.0:
            continue
        grid_flag_col = f"is_bad_{kpi}_kpi" if f"is_bad_{kpi}_kpi" in grid_metrics.columns else f"is_bad_{kpi}"
        if grid_flag_col not in grid_metrics.columns:
            continue
        kpi_bad_grid_ids = set(grid_metrics.loc[grid_metrics[grid_flag_col].fillna(False), "grid_id"].astype(str).tolist())
        if not kpi_bad_grid_ids:
            continue
        col = f"pred_{kpi}"
        if col not in work.columns:
            continue
        kpi_work = work.loc[work["grid_id"].astype(str).isin(kpi_bad_grid_ids)].copy()
        if kpi_work.empty:
            continue
        kpi_work[col] = pd.to_numeric(kpi_work[col], errors="coerce")
        kpi_work["_is_bad"] = kpi_work[col] < _ACTIVE_THRESHOLDS[kpi]
        kpi_work["_raw_severity"] = _severity(kpi_work[col], _ACTIVE_THRESHOLDS[kpi])
        kpi_work = kpi_work.loc[kpi_work["_is_bad"].fillna(False)].copy()
        if kpi_work.empty:
            continue
        grouped = (
            kpi_work.groupby("Cell ID", dropna=False)
            .agg(
                bad_samples=("_is_bad", "sum"),
                bad_grid_count=("grid_id", "nunique"),
                bad_severity=("_raw_severity", "sum"),
            )
            .reset_index()
        )
        for _, row in grouped.iterrows():
            cell_id = str(row["Cell ID"])
            record = cell_rows.setdefault(
                cell_id,
                {
                    "Cell ID": cell_id,
                    "Bad RSRP": 0,
                    "Bad RSRQ": 0,
                    "Bad SINR": 0,
                    "rsrp_bad_severity": 0.0,
                    "rsrq_bad_severity": 0.0,
                    "sinr_bad_severity": 0.0,
                },
            )
            record[bad_col] = int(row["bad_samples"])
            record[sev_col] = float(row["bad_severity"])
        for cell_id, cell_grids in kpi_work.groupby("Cell ID")["grid_id"]:
            cell_grid_sets.setdefault(str(cell_id), set()).update(cell_grids.astype(str).tolist())

    if not cell_rows:
        return pd.DataFrame(), grid_metrics

    summary = pd.DataFrame(cell_rows.values())
    summary["Bad Grid Count"] = summary["Cell ID"].astype(str).map(lambda cell_id: len(cell_grid_sets.get(cell_id, set())))
    summary["combined_grid_severity"] = (
        summary["rsrp_bad_severity"] * float(_ACTIVE_WEIGHTS.rsrp)
        + summary["rsrq_bad_severity"] * float(_ACTIVE_WEIGHTS.rsrq)
        + summary["sinr_bad_severity"] * float(_ACTIVE_WEIGHTS.sinr)
    )
    summary["Bad Samples"] = summary[["Bad RSRP", "Bad RSRQ", "Bad SINR"]].sum(axis=1)
    summary["total_bad_samples"] = summary["Bad Samples"]
    site_map = fast_runtime.base._build_cell_site_map(antenna_df)[["Cell ID", "Site ID"]].copy()
    if not site_map.empty:
        summary = summary.merge(site_map, on="Cell ID", how="left")
    summary = summary.sort_values(["Bad Grid Count", "combined_grid_severity", "Bad Samples", "Cell ID"], ascending=[False, False, False, True]).reset_index(drop=True)
    return _weighted_kpi_cell_order(summary), grid_metrics


def _grid_validation_payload_combined(
    baseline_df: pd.DataFrame,
    grid_analytics_df: pd.DataFrame,
    threshold: float,
) -> Dict[str, object]:
    recomputed = _aggregate_grid_metrics_combined(baseline_df, threshold)
    mapped_rows = int(fast_runtime._normalize_grid_id_series(baseline_df.get("grid_id")).notna().sum()) if "grid_id" in baseline_df.columns else 0
    payload: Dict[str, object] = {
        "kpi_mode": "combined_weighted",
        "thresholds": dict(_ACTIVE_THRESHOLDS),
        "weights": _ACTIVE_WEIGHTS.__dict__,
        "comparison_operator": "weighted any-active KPI below threshold",
        "baseline_rows": int(len(baseline_df)),
        "grid_mapped_rows": mapped_rows,
        "grid_unmapped_rows": int(len(baseline_df) - mapped_rows),
        "grid_mapped_pct": float(mapped_rows / max(len(baseline_df), 1) * 100.0),
        "recomputed_grid_count": int(len(recomputed)),
        "recomputed_bad_grid_count": int(recomputed["is_bad_combined"].fillna(False).sum()) if not recomputed.empty else 0,
        "grid_analytics_rows": int(len(grid_analytics_df)),
        "grid_analytics_bad_grid_count": None,
        "common_grid_count": 0,
    }
    reference = _grid_reference_metrics_combined(grid_analytics_df, threshold)
    if reference.empty:
        return payload
    payload["grid_analytics_bad_grid_count"] = int(reference["is_bad_combined"].fillna(False).sum())
    if not recomputed.empty:
        compare_cols = ["grid_id", "avg_rsrp", "avg_rsrq", "avg_sinr"]
        compare = recomputed[compare_cols].merge(reference[compare_cols], on="grid_id", how="inner", suffixes=("_recomputed", "_analytics"))
        payload["common_grid_count"] = int(len(compare))
        for kpi in ["rsrp", "rsrq", "sinr"]:
            delta = (
                pd.to_numeric(compare[f"avg_{kpi}_recomputed"], errors="coerce")
                - pd.to_numeric(compare[f"avg_{kpi}_analytics"], errors="coerce")
            ).abs()
            payload[f"avg_abs_{kpi}_delta_vs_grid_analytics"] = float(delta.mean()) if len(delta) else np.nan
            payload[f"max_abs_{kpi}_delta_vs_grid_analytics"] = float(delta.max()) if len(delta) else np.nan
    return payload


def _changed_cell_local_metrics_combined(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    updates: Sequence[Dict[str, object]],
    threshold: float,
) -> Dict[str, object]:
    rows = []
    for update in updates:
        cell_id = str(update.get("cell_id", "")).strip()
        if not cell_id:
            continue
        before_cell = before_df.loc[before_df.get("Node_Cell_ID", "").astype(str) == cell_id].copy() if "Node_Cell_ID" in before_df.columns else pd.DataFrame()
        after_cell = after_df.loc[after_df.get("Node_Cell_ID", "").astype(str) == cell_id].copy() if "Node_Cell_ID" in after_df.columns else pd.DataFrame()
        row: Dict[str, object] = {
            "cell_id": cell_id,
            "parameter": update.get("parameter", "ETilt"),
            "target_value": update.get("target_value"),
            "before_sample_count": int(len(before_cell)),
            "after_sample_count": int(len(after_cell)),
        }
        bad_reduction_sum = 0
        for kpi in ["rsrp", "rsrq", "sinr"]:
            before_values = pd.to_numeric(before_cell.get(f"pred_{kpi}"), errors="coerce") if not before_cell.empty else pd.Series(dtype=float)
            after_values = pd.to_numeric(after_cell.get(f"pred_{kpi}"), errors="coerce") if not after_cell.empty else pd.Series(dtype=float)
            before_bad = int((before_values < _ACTIVE_THRESHOLDS[kpi]).fillna(False).sum()) if len(before_values) else 0
            after_bad = int((after_values < _ACTIVE_THRESHOLDS[kpi]).fillna(False).sum()) if len(after_values) else 0
            before_avg = float(before_values.mean()) if len(before_values.dropna()) else np.nan
            after_avg = float(after_values.mean()) if len(after_values.dropna()) else np.nan
            row[f"before_bad_{kpi}_count"] = before_bad
            row[f"after_bad_{kpi}_count"] = after_bad
            row[f"{kpi}_bad_sample_reduction"] = before_bad - after_bad
            row[f"before_avg_{kpi}"] = before_avg
            row[f"after_avg_{kpi}"] = after_avg
            row[f"avg_{kpi}_delta"] = after_avg - before_avg if pd.notna(before_avg) and pd.notna(after_avg) else np.nan
            bad_reduction_sum += before_bad - after_bad
        row["combined_bad_sample_reduction"] = bad_reduction_sum
        rows.append(row)
    if not rows:
        return {"changed_cell_local_metrics": "[]", "changed_cell_bad_sample_reduction_sum": 0.0}
    local_df = pd.DataFrame(rows)
    out: Dict[str, object] = {
        "changed_cell_local_metrics": json.dumps(rows, sort_keys=True),
        "changed_cell_bad_sample_reduction_sum": float(pd.to_numeric(local_df["combined_bad_sample_reduction"], errors="coerce").fillna(0).sum()),
    }
    for kpi in ["rsrp", "rsrq", "sinr"]:
        out[f"changed_cell_avg_{kpi}_delta_mean"] = float(pd.to_numeric(local_df[f"avg_{kpi}_delta"], errors="coerce").mean())
    return out


def _prepare_scope_export_combined(df: pd.DataFrame, threshold: float, stage: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["lat"] = pd.to_numeric(out.get("lat"), errors="coerce")
    out["lon"] = pd.to_numeric(out.get("lon"), errors="coerce")
    for kpi in ["rsrp", "rsrq", "sinr"]:
        col = f"pred_{kpi}"
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out[f"is_bad_{kpi}"] = (
            out[col] < _ACTIVE_THRESHOLDS[kpi]
            if getattr(_ACTIVE_WEIGHTS, kpi) > 0.0
            else False
        )
    out["is_bad_combined"] = out["is_bad_rsrp"] | out["is_bad_rsrq"] | out["is_bad_sinr"]
    out["combined_severity"] = _weighted_severity(out, prefix="pred")
    out["stage"] = stage
    return out


def _score_candidate_on_frontend_grids(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    config,
    grid_analytics_df: Optional[pd.DataFrame],
) -> Optional[Dict[str, float]]:
    if grid_analytics_df is None or grid_analytics_df.empty:
        return None
    frontend_grid = _grid_reference_metrics_combined(grid_analytics_df, threshold=0.0)
    model_baseline_grid = _aggregate_grid_metrics_combined(baseline_df, threshold=0.0)
    candidate_grid = _aggregate_grid_metrics_combined(candidate_df, threshold=0.0)
    if frontend_grid.empty or model_baseline_grid.empty or candidate_grid.empty:
        return None

    compare_cols = ["grid_id", "avg_rsrp", "avg_rsrq", "avg_sinr"]
    delta_source = model_baseline_grid[compare_cols].merge(
        candidate_grid[compare_cols],
        on="grid_id",
        how="inner",
        suffixes=("_model_base", "_model_cand"),
    )
    if delta_source.empty:
        return None
    for kpi in ["rsrp", "rsrq", "sinr"]:
        delta_source[f"avg_{kpi}_delta"] = (
            pd.to_numeric(delta_source[f"avg_{kpi}_model_cand"], errors="coerce")
            - pd.to_numeric(delta_source[f"avg_{kpi}_model_base"], errors="coerce")
        )

    merged = frontend_grid[compare_cols].rename(
        columns={
            "avg_rsrp": "avg_rsrp_base",
            "avg_rsrq": "avg_rsrq_base",
            "avg_sinr": "avg_sinr_base",
        }
    )
    merged = merged.merge(
        delta_source[["grid_id", "avg_rsrp_delta", "avg_rsrq_delta", "avg_sinr_delta"]],
        on="grid_id",
        how="left",
    )
    for kpi in ["rsrp", "rsrq", "sinr"]:
        base_col = f"avg_{kpi}_base"
        delta_col = f"avg_{kpi}_delta"
        cand_col = f"avg_{kpi}_cand"
        merged[base_col] = pd.to_numeric(merged[base_col], errors="coerce")
        merged[cand_col] = merged[base_col] + pd.to_numeric(merged[delta_col], errors="coerce").fillna(0.0)
    merged = merged.dropna(subset=["grid_id"]).copy()
    if merged.empty:
        return None

    evaluation_grid_count = max(float(len(merged)), 1.0)
    base_bad_any = pd.Series(False, index=merged.index)
    cand_bad_any = pd.Series(False, index=merged.index)
    total_severity_reduction = 0.0
    weighted_severity_reduction = 0.0
    metrics: Dict[str, float] = {
        "evaluation_grid_count": float(evaluation_grid_count),
        "frontend_common_grid_count": float(len(merged)),
    }

    for kpi in ["rsrp", "rsrq", "sinr"]:
        weight = float(getattr(_ACTIVE_WEIGHTS, kpi))
        base_values = pd.to_numeric(merged[f"avg_{kpi}_base"], errors="coerce")
        cand_values = pd.to_numeric(merged[f"avg_{kpi}_cand"], errors="coerce").fillna(base_values)
        active = weight > 0.0
        base_bad = (base_values < _ACTIVE_THRESHOLDS[kpi]) if active else pd.Series(False, index=merged.index)
        cand_bad = (cand_values < _ACTIVE_THRESHOLDS[kpi]) if active else pd.Series(False, index=merged.index)
        base_bad_any = base_bad_any | base_bad
        cand_bad_any = cand_bad_any | cand_bad

        recovered = int((base_bad & ~cand_bad).sum())
        new_bad = int((~base_bad & cand_bad).sum())
        severity_reduction = float((_severity(base_values, _ACTIVE_THRESHOLDS[kpi]) - _severity(cand_values, _ACTIVE_THRESHOLDS[kpi])).sum())
        severity_per_grid = severity_reduction / evaluation_grid_count
        mean_delta = float((cand_values - base_values).mean())
        before_bad_count = int(base_bad.sum())
        after_bad_count = int(cand_bad.sum())
        net_reduction = before_bad_count - after_bad_count

        total_severity_reduction += severity_reduction
        weighted_severity_reduction += severity_per_grid * weight / _SEVERITY_CAPS[kpi]
        metrics.update(
            {
                f"frontend_{kpi}_before_bad_grid_count": float(before_bad_count),
                f"frontend_{kpi}_after_bad_grid_count": float(after_bad_count),
                f"frontend_{kpi}_net_bad_grid_reduction": float(net_reduction),
                f"{kpi}_recovered_bad": float(recovered),
                f"{kpi}_new_bad": float(new_bad),
                f"{kpi}_severity_reduction": float(severity_reduction),
                f"{kpi}_severity_reduction_per_grid": float(severity_per_grid),
                f"{kpi}_grid_recovered_share": float(recovered) / evaluation_grid_count,
                f"{kpi}_grid_new_bad_share": float(new_bad) / evaluation_grid_count,
                f"mean_{kpi}_delta": float(mean_delta),
                f"combined_{kpi}_component": float(_frontend_grid_component({**metrics, f'{kpi}_severity_reduction_per_grid': severity_per_grid, f'{kpi}_grid_recovered_share': float(recovered) / evaluation_grid_count, f'{kpi}_grid_new_bad_share': float(new_bad) / evaluation_grid_count, f'mean_{kpi}_delta': mean_delta}, kpi)),
            }
        )

    recovered_combined = int((base_bad_any & ~cand_bad_any).sum())
    new_bad_combined = int((~base_bad_any & cand_bad_any).sum())
    combined_baseline_bad_count = int(base_bad_any.sum())
    combined_candidate_bad_count = int(cand_bad_any.sum())
    combined_net_bad_reduction = combined_baseline_bad_count - combined_candidate_bad_count

    active_weights = {
        kpi: float(getattr(_ACTIVE_WEIGHTS, kpi))
        for kpi in ["rsrp", "rsrq", "sinr"]
        if float(getattr(_ACTIVE_WEIGHTS, kpi)) > 0.0
    }
    weight_sum = max(float(sum(active_weights.values())), 1.0)
    weighted_before_bad_count = sum(
        (weight / weight_sum) * _safe_float(metrics.get(f"frontend_{kpi}_before_bad_grid_count"))
        for kpi, weight in active_weights.items()
    )
    weighted_after_bad_count = sum(
        (weight / weight_sum) * _safe_float(metrics.get(f"frontend_{kpi}_after_bad_grid_count"))
        for kpi, weight in active_weights.items()
    )
    weighted_recovered_bad = sum(
        (weight / weight_sum) * _safe_float(metrics.get(f"{kpi}_recovered_bad"))
        for kpi, weight in active_weights.items()
    )
    weighted_new_bad = sum(
        (weight / weight_sum) * _safe_float(metrics.get(f"{kpi}_new_bad"))
        for kpi, weight in active_weights.items()
    )
    weighted_net_bad_reduction = weighted_before_bad_count - weighted_after_bad_count

    max_weight = max(_ACTIVE_WEIGHTS.rsrp, _ACTIVE_WEIGHTS.rsrq, _ACTIVE_WEIGHTS.sinr)
    priority_kpis = [kpi for kpi in ["rsrp", "rsrq", "sinr"] if float(getattr(_ACTIVE_WEIGHTS, kpi)) == float(max_weight) and max_weight > 0.0]
    if len(priority_kpis) == 1:
        decision_kpi = priority_kpis[0]
        baseline_bad_count = _safe_float(metrics.get(f"frontend_{decision_kpi}_before_bad_grid_count"))
        candidate_bad_count = _safe_float(metrics.get(f"frontend_{decision_kpi}_after_bad_grid_count"))
        recovered_bad_count = _safe_float(metrics.get(f"{decision_kpi}_recovered_bad"))
        new_bad_count = _safe_float(metrics.get(f"{decision_kpi}_new_bad"))
        net_bad_reduction = baseline_bad_count - candidate_bad_count
        decision_scope = f"priority_kpi_frontend_{decision_kpi}"
    else:
        baseline_bad_count = weighted_before_bad_count
        candidate_bad_count = weighted_after_bad_count
        recovered_bad_count = weighted_recovered_bad
        new_bad_count = weighted_new_bad
        net_bad_reduction = weighted_net_bad_reduction
        decision_scope = "weighted_frontend_kpi_blend"

    good_area_loss_pct = (float(weighted_new_bad) / evaluation_grid_count) * 100.0
    net_bad_reduction_share = float(weighted_net_bad_reduction) / evaluation_grid_count
    weighted_score = (
        _ACTIVE_WEIGHTS.rsrp * _frontend_grid_component(metrics, "rsrp")
        + _ACTIVE_WEIGHTS.rsrq * _frontend_grid_component(metrics, "rsrq")
        + _ACTIVE_WEIGHTS.sinr * _frontend_grid_component(metrics, "sinr")
        + net_bad_reduction_share * 2.0
        - good_area_loss_pct * 0.0025
    )

    priority_worsened = any(_safe_float(metrics.get(f"frontend_{kpi}_net_bad_grid_reduction")) < 0.0 for kpi in priority_kpis)
    constraints_passed = good_area_loss_pct <= max(float(config.max_good_area_loss_pct), 15.0) and not priority_worsened
    if not constraints_passed:
        weighted_score -= float(fast_runtime.base.SEVERE_CONSTRAINT_PENALTY)

    metrics.update(
        {
            "baseline_bad_count": float(baseline_bad_count),
            "candidate_bad_count": float(candidate_bad_count),
            "recovered_bad_samples": float(recovered_bad_count),
            "new_bad_samples": float(new_bad_count),
            "net_bad_reduction": float(net_bad_reduction),
            "net_bad_reduction_share": float(net_bad_reduction_share),
            "recovered_bad_share": float(recovered_bad_count) / evaluation_grid_count,
            "new_bad_share": float(new_bad_count) / evaluation_grid_count,
            "weighted_frontend_before_bad_grid_count": float(weighted_before_bad_count),
            "weighted_frontend_after_bad_grid_count": float(weighted_after_bad_count),
            "weighted_frontend_net_bad_grid_reduction": float(weighted_net_bad_reduction),
            "combined_any_before_bad_grid_count": float(combined_baseline_bad_count),
            "combined_any_after_bad_grid_count": float(combined_candidate_bad_count),
            "combined_any_net_bad_grid_reduction": float(combined_net_bad_reduction),
            "combined_any_recovered_bad": float(recovered_combined),
            "combined_any_new_bad": float(new_bad_combined),
            "decision_scope": decision_scope,
            "total_severity_reduction": float(total_severity_reduction),
            "combined_weighted_severity_reduction": float(weighted_severity_reduction),
            "good_area_loss_pct": float(good_area_loss_pct),
            "score": float(weighted_score),
            "constraints_passed": float(1 if constraints_passed else 0),
            "priority_kpi_worsened": float(1 if priority_worsened else 0),
            "frontend_scoring_used": 1.0,
            "grid_scoring_source": "frontend_grid_analytics_plus_candidate_rf_delta_weighted_kpi_formula",
            "validation_scope": "frontend_grid_analytics_population_with_candidate_rf_delta",
        }
    )
    return metrics


def _score_candidate_combined(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    evaluation_cells: Sequence[str],
    config,
    grid_analytics_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    frontend_metrics = _score_candidate_on_frontend_grids(baseline_df, candidate_df, config, grid_analytics_df)
    if frontend_metrics is not None:
        frontend_metrics["combined_rsrp_weight"] = float(_ACTIVE_WEIGHTS.rsrp)
        frontend_metrics["combined_rsrq_weight"] = float(_ACTIVE_WEIGHTS.rsrq)
        frontend_metrics["combined_sinr_weight"] = float(_ACTIVE_WEIGHTS.sinr)
        frontend_metrics["combined_weighted_tie_break"] = float(frontend_metrics.get("combined_weighted_severity_reduction", 0.0))
        # Compatibility fields consumed by the fast search/reporting runtime.
        frontend_metrics["baseline_bad_grid_count"] = frontend_metrics.get("baseline_bad_count", 0.0)
        frontend_metrics["candidate_bad_grid_count"] = frontend_metrics.get("candidate_bad_count", 0.0)
        frontend_metrics["bad_to_good_grid_count"] = frontend_metrics.get("recovered_bad_samples", 0.0)
        frontend_metrics["good_to_bad_grid_count"] = frontend_metrics.get("new_bad_samples", 0.0)
        frontend_metrics["grid_mean_rsrp_delta_bad_baseline"] = frontend_metrics.get("mean_rsrp_delta", 0.0)
        return frontend_metrics

    core_config = fast_runtime.base.TiltRecommendationTestConfig(
        project_id=int(config.project_id),
        region=str(config.region),
        operator=config.operator,
        rsrp_threshold=_ACTIVE_THRESHOLDS["rsrp"],
        rsrq_threshold=_ACTIVE_THRESHOLDS["rsrq"],
        sinr_threshold=_ACTIVE_THRESHOLDS["sinr"],
        validate_candidates=True,
        radius_m=float(config.radius_m),
        grid_resolution_m=float(config.grid_resolution_m),
        workers=int(config.workers),
        impact_radius_m=float(config.impact_radius_m),
        neighbor_site_count=int(config.neighbor_site_count),
        max_interference_sites=int(config.max_interference_sites),
        max_good_area_loss_pct=float(config.max_good_area_loss_pct),
        max_mean_sinr_drop_db=float(config.max_mean_sinr_drop_db),
        min_score_gain=float(config.min_score_gain),
        min_recovered_bad_samples=int(config.min_recovered_bad_samples),
        max_ranked_cells=0,
        max_ranked_sites=0,
        threshold_file_path=config.threshold_file_path,
        threshold_constraint_count=int(config.threshold_constraint_count),
        threshold_optimised_count=int(config.threshold_optimised_count),
        output_root=config.output_root,
    )
    metrics = fast_runtime.base._score_candidate_vs_baseline(baseline_df, candidate_df, evaluation_cells, core_config)
    rsrp_component = _normalised_component(metrics, "rsrp")
    rsrq_component = _normalised_component(metrics, "rsrq")
    sinr_component = _normalised_component(metrics, "sinr")
    combined_weighted_severity = (
        _ACTIVE_WEIGHTS.rsrp * (_safe_float(metrics.get("rsrp_severity_reduction_per_sample")) / _SEVERITY_CAPS["rsrp"])
        + _ACTIVE_WEIGHTS.rsrq * (_safe_float(metrics.get("rsrq_severity_reduction_per_sample")) / _SEVERITY_CAPS["rsrq"])
        + _ACTIVE_WEIGHTS.sinr * (_safe_float(metrics.get("sinr_severity_reduction_per_sample")) / _SEVERITY_CAPS["sinr"])
    )
    weighted_score = (
        _ACTIVE_WEIGHTS.rsrp * rsrp_component
        + _ACTIVE_WEIGHTS.rsrq * rsrq_component
        + _ACTIVE_WEIGHTS.sinr * sinr_component
        + metrics.get("net_bad_reduction_share", 0.0) * 2.0
        - metrics.get("good_area_loss_pct", 0.0) * 0.0025
    )
    if not bool(metrics.get("constraints_passed", 0.0)):
        weighted_score -= float(fast_runtime.base.SEVERE_CONSTRAINT_PENALTY)
    metrics["score"] = float(weighted_score)
    metrics["combined_rsrp_component"] = float(rsrp_component)
    metrics["combined_rsrq_component"] = float(rsrq_component)
    metrics["combined_sinr_component"] = float(sinr_component)
    metrics["combined_weighted_severity_reduction"] = float(combined_weighted_severity)
    metrics["combined_rsrp_weight"] = float(_ACTIVE_WEIGHTS.rsrp)
    metrics["combined_rsrq_weight"] = float(_ACTIVE_WEIGHTS.rsrq)
    metrics["combined_sinr_weight"] = float(_ACTIVE_WEIGHTS.sinr)
    metrics["combined_weighted_tie_break"] = float(combined_weighted_severity)
    metrics["baseline_bad_sample_count"] = metrics.get("baseline_bad_count", 0.0)
    metrics["candidate_bad_sample_count"] = metrics.get("candidate_bad_count", 0.0)
    metrics["bad_to_good_sample_count"] = metrics.get("recovered_bad_samples", 0.0)
    metrics["good_to_bad_sample_count"] = metrics.get("new_bad_samples", 0.0)
    metrics["grid_mean_rsrp_delta_bad_baseline"] = metrics.get("mean_rsrp_delta", 0.0)
    metrics["grid_scoring_source"] = "combined_weighted_core_kpi_formula"
    metrics["validation_scope"] = "combined_weighted_best_serving_population"
    return metrics


def _activate_combined_mode(thresholds: Dict[str, float], weights: KpiWeights) -> None:
    global _ACTIVE_THRESHOLDS, _ACTIVE_WEIGHTS
    _ACTIVE_THRESHOLDS = {key: float(value) for key, value in thresholds.items()}
    _ACTIVE_WEIGHTS = weights
    fast_runtime.base.TILT_SRC.filter_bad_samples = _filter_bad_samples_combined
    fast_runtime._aggregate_grid_metrics_from_predictions = _aggregate_grid_metrics_combined
    fast_runtime._grid_reference_metrics_from_analytics = _grid_reference_metrics_combined
    fast_runtime._grid_validation_payload = _grid_validation_payload_combined
    fast_runtime._build_grid_ranked_summary = _build_grid_ranked_summary_combined
    fast_runtime._score_candidate_vs_baseline_grids = _score_candidate_combined
    fast_runtime._changed_cell_local_metrics_global = _changed_cell_local_metrics_combined
    fast_runtime._prepare_scope_export = _prepare_scope_export_combined


def _replace_path_values(payload: Any, old: str, new: str) -> Any:
    if isinstance(payload, dict):
        return {key: _replace_path_values(value, old, new) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_replace_path_values(value, old, new) for value in payload]
    if isinstance(payload, str):
        return payload.replace(old, new)
    return payload


def _truthy_series(values: pd.Series) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    return values.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _build_per_kpi_grid_impact_report(run_dir: Path) -> Optional[pd.DataFrame]:
    frontend_path = run_dir / "grid_analytics_input.csv"
    before_path = frontend_path if frontend_path.exists() else run_dir / "best_candidate_before_grid_metrics.csv"
    after_path = run_dir / "best_candidate_after_grid_metrics.csv"
    if not before_path.exists() or not after_path.exists():
        return None
    before = pd.read_csv(before_path)
    after = pd.read_csv(after_path)
    if before.empty or after.empty or "grid_id" not in before.columns or "grid_id" not in after.columns:
        return None
    merged = before.merge(after, on="grid_id", how="inner", suffixes=("_before", "_after"))
    rows = []
    for kpi in ["rsrp", "rsrq", "sinr"]:
        before_source_col = f"baseline_avg_{kpi}_before" if frontend_path.exists() else f"avg_{kpi}_before"
        before_avg = before_source_col if before_source_col in merged.columns else f"avg_{kpi}_before"
        after_avg = f"avg_{kpi}_after"
        if before_avg not in merged.columns or after_avg not in merged.columns:
            continue
        before_values = pd.to_numeric(merged.get(before_avg), errors="coerce")
        after_values = pd.to_numeric(merged.get(after_avg), errors="coerce")
        if getattr(_ACTIVE_WEIGHTS, kpi) > 0.0:
            was_bad = before_values < _ACTIVE_THRESHOLDS[kpi]
            is_bad = after_values < _ACTIVE_THRESHOLDS[kpi]
        else:
            was_bad = pd.Series(False, index=merged.index)
            is_bad = pd.Series(False, index=merged.index)
        rows.append(
            {
                "kpi": kpi.upper(),
                "baseline_scope": "frontend_grid_analytics" if frontend_path.exists() else "recomputed_candidate_before_grid",
                "threshold": _ACTIVE_THRESHOLDS[kpi],
                "weight": getattr(_ACTIVE_WEIGHTS, kpi),
                "frontend_total_before_bad_grids": int((pd.to_numeric(before.get(f"baseline_avg_{kpi}" if frontend_path.exists() else f"avg_{kpi}"), errors="coerce") < _ACTIVE_THRESHOLDS[kpi]).sum()),
                "before_bad_grids": int(was_bad.sum()),
                "after_bad_grids": int(is_bad.sum()),
                "net_bad_grid_reduction": int(was_bad.sum() - is_bad.sum()),
                "bad_to_good_grids": int((was_bad & ~is_bad).sum()),
                "good_to_bad_grids": int((~was_bad & is_bad).sum()),
                "stayed_bad_grids": int((was_bad & is_bad).sum()),
                "mean_before": float(before_values.mean()) if len(before_values.dropna()) else np.nan,
                "mean_after": float(after_values.mean()) if len(after_values.dropna()) else np.nan,
                "mean_delta": float((after_values - before_values).mean()) if len(before_values.dropna()) and len(after_values.dropna()) else np.nan,
            }
        )
    if not rows:
        return None
    report = pd.DataFrame(rows)
    report.to_csv(run_dir / "combined_kpi_grid_impact.csv", index=False)
    return report


def _postprocess_combined_outputs(run_dir: Path) -> Dict[str, str]:
    renamed: Dict[str, str] = {}
    for stage in ["before", "after"]:
        scope_path = run_dir / f"best_candidate_{stage}_scope.csv.gz"
        combined_path = run_dir / f"best_candidate_{stage}_bad_combined.csv.gz"
        if not scope_path.exists():
            continue
        scope_df = pd.read_csv(scope_path)
        if "is_bad_combined" not in scope_df.columns:
            continue
        combined_bad_df = scope_df.loc[_truthy_series(scope_df["is_bad_combined"])].copy()
        combined_bad_df.to_csv(combined_path, index=False, compression="gzip")
        renamed[f"best_candidate_{stage}_bad_combined"] = str(combined_path)

    best_summary_path = run_dir / "best_candidate_summary.json"
    best_payload: Dict[str, object] = {}
    if best_summary_path.exists():
        best_payload = json.loads(best_summary_path.read_text(encoding="utf-8"))
        best_payload["root_cause"] = "global_bad_grid_combined_weighted"
        best_payload["topology_root_cause"] = "global_bad_grid_combined_weighted"
        best_payload["kpi_mode"] = "combined_weighted"
        best_payload["weights"] = _ACTIVE_WEIGHTS.__dict__
        best_payload["thresholds"] = dict(_ACTIVE_THRESHOLDS)
        best_summary_path.write_text(json.dumps(best_payload, indent=2, default=str), encoding="utf-8")

    grid_impact = _build_per_kpi_grid_impact_report(run_dir)
    if grid_impact is not None:
        renamed["combined_kpi_grid_impact"] = str(run_dir / "combined_kpi_grid_impact.csv")

    for csv_name in ["candidate_validation_results.csv", "site_candidate_evaluations.csv"]:
        csv_path = run_dir / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "root_cause" in df.columns:
            df["root_cause"] = "global_bad_grid_combined_weighted"
        if "topology_root_cause" in df.columns:
            df["topology_root_cause"] = "global_bad_grid_combined_weighted"
        df["kpi_mode"] = "combined_weighted"
        df["combined_rsrp_weight"] = _ACTIVE_WEIGHTS.rsrp
        df["combined_rsrq_weight"] = _ACTIVE_WEIGHTS.rsrq
        df["combined_sinr_weight"] = _ACTIVE_WEIGHTS.sinr
        df.to_csv(csv_path, index=False)

    reason = (
        "Global combined weighted KPI optimization. "
        f"weights RSRP={_ACTIVE_WEIGHTS.rsrp:.2f}, RSRQ={_ACTIVE_WEIGHTS.rsrq:.2f}, SINR={_ACTIVE_WEIGHTS.sinr:.2f}."
    )
    for csv_name in ["recommendations.csv", "recommendations_all.csv"]:
        csv_path = run_dir / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "Reason" in df.columns:
            df["Reason"] = df["Reason"].astype(str).str.replace("Global RSRP bad-grid cell optimization.", reason, regex=False)
        if "Root Cause Category" in df.columns:
            df["Root Cause Category"] = "global_bad_grid_combined_weighted"
        for label, metric_key in [
            ("Mean RSRQ Delta", "mean_rsrq_delta"),
            ("Mean SINR Delta", "mean_sinr_delta"),
            ("RSRQ Recovered Bad", "rsrq_recovered_bad"),
            ("RSRQ New Bad", "rsrq_new_bad"),
            ("SINR Recovered Bad", "sinr_recovered_bad"),
            ("SINR New Bad", "sinr_new_bad"),
            ("Combined Weighted Severity Reduction", "combined_weighted_severity_reduction"),
            ("Combined RSRP Component", "combined_rsrp_component"),
            ("Combined RSRQ Component", "combined_rsrq_component"),
            ("Combined SINR Component", "combined_sinr_component"),
        ]:
            if label not in df.columns:
                df[label] = _safe_float(best_payload.get(metric_key), np.nan) if best_payload else np.nan
        df["KPI Mode"] = "combined_weighted"
        df["RSRP Weight"] = _ACTIVE_WEIGHTS.rsrp
        df["RSRQ Weight"] = _ACTIVE_WEIGHTS.rsrq
        df["SINR Weight"] = _ACTIVE_WEIGHTS.sinr
        df.to_csv(csv_path, index=False)
    return renamed


def _move_run_dir_to_combined_name(run_dir: Path) -> Path:
    target = run_dir.with_name(run_dir.name.replace("tilt_rsrp_only", "tilt_combined_weighted"))
    if target == run_dir:
        target = run_dir.with_name(f"tilt_combined_weighted_{run_dir.name}")
    if target.exists():
        target = run_dir.with_name(target.name + "_combined")
    shutil.move(str(run_dir), str(target))
    renamed_artifacts = _postprocess_combined_outputs(target)
    summary_path = target / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload = _replace_path_values(payload, str(run_dir), str(target))
        for old_name, new_path in renamed_artifacts.items():
            payload = _replace_path_values(payload, old_name, Path(new_path).name)
        payload["run_type"] = "tilt_combined_weighted_test"
        payload["thresholds"] = {
            "rsrp": _ACTIVE_THRESHOLDS["rsrp"],
            "rsrq": _ACTIVE_THRESHOLDS["rsrq"],
            "sinr": _ACTIVE_THRESHOLDS["sinr"],
            "kpi_mode": "Combined weighted RSRP/RSRQ/SINR",
        }
        payload["weights"] = {
            "rsrp": _ACTIVE_WEIGHTS.rsrp,
            "rsrq": _ACTIVE_WEIGHTS.rsrq,
            "sinr": _ACTIVE_WEIGHTS.sinr,
        }
        payload.setdefault("search", {})["kpi_formula_source"] = "tests.lte_tilt_combined_weighted_recommendation_test._score_candidate_combined"
        payload.setdefault("search", {})["ranking_source"] = "weighted_kpi_cell_prioritization_plus_frontend_weighted_score"
        payload.setdefault("search", {})["cell_prioritization"] = "weighted_per_kpi_bad_cell_slices_from_cli_weights"
        if isinstance(payload.get("best_candidate"), dict):
            payload["best_candidate"]["root_cause"] = "global_bad_grid_combined_weighted"
            payload["best_candidate"]["topology_root_cause"] = "global_bad_grid_combined_weighted"
            payload["best_candidate"]["kpi_mode"] = "combined_weighted"
        payload["artifacts"] = payload.get("artifacts", {})
        if (target / "best_candidate_before_bad_combined.csv.gz").exists():
            payload["artifacts"]["best_candidate_before_bad_combined"] = str(target / "best_candidate_before_bad_combined.csv.gz")
        if (target / "best_candidate_after_bad_combined.csv.gz").exists():
            payload["artifacts"]["best_candidate_after_bad_combined"] = str(target / "best_candidate_after_bad_combined.csv.gz")
        if (target / "combined_kpi_grid_impact.csv").exists():
            payload["artifacts"]["combined_kpi_grid_impact"] = str(target / "combined_kpi_grid_impact.csv")
        counts = payload.setdefault("counts", {})
        for stage in ["before", "after"]:
            combined_path = target / f"best_candidate_{stage}_bad_combined.csv.gz"
            if combined_path.exists():
                counts[f"{stage}_bad_combined_count"] = int(len(pd.read_csv(combined_path, low_memory=False)))
        summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def _parse_args():
    parser = argparse.ArgumentParser()
    fixture_available = fast_runtime.PROJECT_196_RSRP_TILT_BASELINE_POINTS.exists()
    parser.add_argument("--project-id", type=int, default=fast_runtime.DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=fast_runtime.DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--rsrp", type=float, default=DEFAULT_RSRP_THRESHOLD)
    parser.add_argument("--rsrq", type=float, default=DEFAULT_RSRQ_THRESHOLD)
    parser.add_argument("--sinr", type=float, default=DEFAULT_SINR_THRESHOLD)
    parser.add_argument("--rsrp-weight", type=float, default=60.0)
    parser.add_argument("--rsrq-weight", type=float, default=20.0)
    parser.add_argument("--sinr-weight", type=float, default=20.0)
    parser.add_argument("--radius-m", type=float, default=500.0)
    parser.add_argument("--grid-resolution-m", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--impact-radius-m", type=float, default=None)
    parser.add_argument("--neighbor-site-count", type=int, default=3)
    parser.add_argument("--max-interference-sites", type=int, default=10)
    parser.add_argument("--max-good-area-loss-pct", type=float, default=2.0)
    parser.add_argument("--max-mean-sinr-drop-db", type=float, default=1.0)
    parser.add_argument("--min-score-gain", type=float, default=0.0)
    parser.add_argument("--min-recovered-bad-samples", type=int, default=0)
    parser.add_argument("--bad-grid-coverage-pct", type=float, default=80.0)
    parser.add_argument("--max-group-cells", type=int, default=0)
    parser.add_argument("--candidate-workers", type=int, default=2)
    parser.add_argument("--coordinate-passes", type=int, default=2)
    parser.add_argument("--max-neighbors-per-update-cell", type=int, default=2)
    parser.add_argument("--threshold-file", "--threshold-file-path", dest="threshold_file_path", type=str, default=str(fast_runtime.PROJECT_196_RSRP_TILT_THRESHOLD_FILE) if fixture_available else None)
    parser.add_argument("--baseline-points", "--baseline-points-path", dest="baseline_points_path", type=str, default=str(fast_runtime.PROJECT_196_RSRP_TILT_BASELINE_POINTS) if fixture_available else None)
    parser.add_argument("--antenna-input", "--antenna-input-path", dest="antenna_input_path", type=str, default=str(fast_runtime.PROJECT_196_RSRP_TILT_ANTENNA_INPUT) if fixture_available else None)
    parser.add_argument("--geo-features", "--geo-features-path", dest="geo_features_path", type=str, default=str(fast_runtime.PROJECT_196_RSRP_TILT_GEO_FEATURES) if fixture_available else None)
    parser.add_argument("--grid-analytics", "--grid-analytics-path", dest="grid_analytics_path", type=str, default=str(fast_runtime.PROJECT_196_RSRP_TILT_GRID_ANALYTICS) if fixture_available else None)
    parser.add_argument("--local-baseline-kpi-stage", choices=["geo", "demo", "raw"], default="geo")
    parser.add_argument("--session-ids", type=str, default=",".join(str(value) for value in fast_runtime.DEFAULT_SESSION_IDS))
    parser.add_argument("--validation-fraction", type=float, default=fast_runtime.DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--no-residual-calibration", action="store_true")
    parser.add_argument("--recompute-k1k2-from-baseline", action="store_true")
    parser.add_argument("--output-root", type=Path, default=fast_runtime.OUTPUT_ROOT)
    args = parser.parse_args()
    session_ids = tuple(int(value.strip()) for value in str(args.session_ids).split(",") if value.strip())
    weights = _normalise_weights(args.rsrp_weight, args.rsrq_weight, args.sinr_weight)
    config = fast_runtime.TiltRsrpOnlyRecommendationTestConfig(
        project_id=args.project_id,
        region=args.region,
        operator=args.operator,
        rsrp_threshold=args.rsrp,
        radius_m=args.radius_m,
        grid_resolution_m=args.grid_resolution_m,
        workers=args.workers,
        impact_radius_m=args.impact_radius_m,
        neighbor_site_count=args.neighbor_site_count,
        max_interference_sites=args.max_interference_sites,
        max_good_area_loss_pct=args.max_good_area_loss_pct,
        max_mean_sinr_drop_db=args.max_mean_sinr_drop_db,
        min_score_gain=args.min_score_gain,
        min_recovered_bad_samples=args.min_recovered_bad_samples,
        bad_grid_coverage_pct=args.bad_grid_coverage_pct,
        max_group_cells=args.max_group_cells,
        candidate_workers=args.candidate_workers,
        coordinate_passes=args.coordinate_passes,
        max_neighbors_per_update_cell=args.max_neighbors_per_update_cell,
        threshold_file_path=args.threshold_file_path,
        baseline_points_path=args.baseline_points_path,
        antenna_input_path=args.antenna_input_path,
        geo_features_path=args.geo_features_path,
        grid_analytics_path=args.grid_analytics_path,
        local_baseline_kpi_stage=args.local_baseline_kpi_stage,
        session_ids=session_ids,
        validation_fraction=args.validation_fraction,
        apply_residual_calibration=not bool(args.no_residual_calibration),
        fixed_k1k2_for_local_inputs=not bool(args.recompute_k1k2_from_baseline),
        output_root=args.output_root,
    )
    thresholds = {"rsrp": float(args.rsrp), "rsrq": float(args.rsrq), "sinr": float(args.sinr)}
    return config, thresholds, weights


if __name__ == "__main__":
    config, thresholds, weights = _parse_args()
    _activate_combined_mode(thresholds, weights)
    run_dir = fast_runtime.run_tilt_rsrp_only_recommendation_test(config)
    run_dir = _move_run_dir_to_combined_name(Path(run_dir))
    print(json.dumps({"run_dir": str(run_dir), "thresholds": thresholds, "weights": weights.__dict__}, indent=2))
