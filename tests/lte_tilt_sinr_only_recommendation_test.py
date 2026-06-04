from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import lte_tilt_recommendation_test as core_tilt
from tests import lte_tilt_rsrp_only_recommendation_test as fast_runtime


KPI_NAME = "sinr"
KPI_LABEL = "SINR"
PRED_COL = "pred_sinr"
AVG_COL = "avg_sinr"
BASELINE_AVG_COL = "baseline_avg_sinr"
BAD_COL = "Bad SINR"
DEFAULT_THRESHOLD = 0.0
DISABLED_THRESHOLD = -999.0


def _active_threshold(config) -> float:
    return float(getattr(config, "rsrp_threshold", DEFAULT_THRESHOLD))


def _convert_args_for_fast_parser(argv: list[str]) -> list[str]:
    converted: list[str] = []
    threshold_was_set = False
    skip_next = False
    for idx, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--sinr":
            converted.append("--rsrp")
            if idx + 1 < len(argv):
                converted.append(argv[idx + 1])
                threshold_was_set = True
                skip_next = True
            continue
        if token.startswith("--sinr="):
            converted.append("--rsrp=" + token.split("=", 1)[1])
            threshold_was_set = True
            continue
        converted.append(token)
    if not threshold_was_set and "--rsrp" not in converted and not any(arg.startswith("--rsrp=") for arg in converted):
        converted.extend(["--rsrp", str(DEFAULT_THRESHOLD)])
    return converted


def _core_config(config) -> core_tilt.TiltRecommendationTestConfig:
    return core_tilt.TiltRecommendationTestConfig(
        project_id=int(config.project_id),
        region=str(config.region),
        operator=config.operator,
        rsrp_threshold=DISABLED_THRESHOLD,
        rsrq_threshold=DISABLED_THRESHOLD,
        sinr_threshold=_active_threshold(config),
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


def _filter_bad_samples_active(log_df: pd.DataFrame, allowed_techs) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = log_df.copy()
    if PRED_COL not in work.columns:
        return pd.DataFrame(), pd.DataFrame(columns=["Cell ID", "Technology", "Bad RSRP", "Bad RSRQ", "Bad SINR"])
    if "Cell ID" not in work.columns and "Node_Cell_ID" in work.columns:
        work["Cell ID"] = work["Node_Cell_ID"].astype(str)
    if "Technology" not in work.columns:
        work["Technology"] = "4G"
    tech_mask = work["Technology"].astype(str).isin(allowed_techs) if allowed_techs else pd.Series(True, index=work.index)
    values = pd.to_numeric(work[PRED_COL], errors="coerce")
    active_bad = tech_mask & values.notna() & (values < float(fast_runtime.base.TILT_SRC.RSRP_THRESH))
    work["Bad RSRP"] = active_bad
    work["Bad RSRQ"] = False
    work["Bad SINR"] = active_bad
    bad_df = work.loc[work["Bad SINR"]].copy()
    if bad_df.empty:
        return bad_df, pd.DataFrame(columns=["Cell ID", "Technology", "Bad RSRP", "Bad RSRQ", "Bad SINR"])
    summary = (
        bad_df.groupby(["Cell ID", "Technology"], dropna=False)
        .agg(**{"Bad RSRP": ("Bad RSRP", "sum"), "Bad RSRQ": ("Bad RSRQ", "sum"), "Bad SINR": ("Bad SINR", "sum")})
        .reset_index()
    )
    return bad_df, summary


def _aggregate_grid_metrics_active(pred_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if pred_df.empty or "grid_id" not in pred_df.columns or PRED_COL not in pred_df.columns:
        return pd.DataFrame()
    work = pred_df.copy()
    work["grid_id"] = fast_runtime._normalize_grid_id_series(work["grid_id"])
    work[PRED_COL] = pd.to_numeric(work[PRED_COL], errors="coerce")
    work = work.dropna(subset=["grid_id", PRED_COL])
    if work.empty:
        return pd.DataFrame()
    agg: Dict[str, tuple] = {"point_count": (PRED_COL, "count"), AVG_COL: (PRED_COL, "mean")}
    for col in ["pred_rsrp", "pred_rsrq", "lat", "lon"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "pred_rsrp" in work.columns:
        agg["avg_rsrp"] = ("pred_rsrp", "mean")
    if "pred_rsrq" in work.columns:
        agg["avg_rsrq"] = ("pred_rsrq", "mean")
    if "lat" in work.columns:
        agg["lat"] = ("lat", "mean")
    if "lon" in work.columns:
        agg["lon"] = ("lon", "mean")
    if "Node_Cell_ID" in work.columns:
        agg["distinct_cells"] = ("Node_Cell_ID", "nunique")
    out = work.groupby("grid_id", dropna=False).agg(**agg).reset_index()
    out["is_bad_sinr"] = pd.to_numeric(out[AVG_COL], errors="coerce") < float(threshold)
    out["sinr_severity"] = (float(threshold) - pd.to_numeric(out[AVG_COL], errors="coerce")).clip(lower=0.0)
    out["is_bad_rsrp"] = out["is_bad_sinr"]
    out["rsrp_severity"] = out["sinr_severity"]
    if "avg_rsrp" not in out.columns:
        out["avg_rsrp"] = np.nan
    if "avg_rsrq" not in out.columns:
        out["avg_rsrq"] = np.nan
    return out


def _grid_reference_metrics_active(grid_analytics_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if grid_analytics_df.empty or "grid_id" not in grid_analytics_df.columns or BASELINE_AVG_COL not in grid_analytics_df.columns:
        return pd.DataFrame()
    work = grid_analytics_df.copy()
    work["grid_id"] = fast_runtime._normalize_grid_id_series(work["grid_id"])
    work[AVG_COL] = pd.to_numeric(work[BASELINE_AVG_COL], errors="coerce")
    work["avg_rsrp"] = pd.to_numeric(work.get("baseline_avg_rsrp"), errors="coerce") if "baseline_avg_rsrp" in work.columns else np.nan
    work["avg_rsrq"] = pd.to_numeric(work.get("baseline_avg_rsrq"), errors="coerce") if "baseline_avg_rsrq" in work.columns else np.nan
    work["point_count"] = pd.to_numeric(work.get("baseline_point_count"), errors="coerce") if "baseline_point_count" in work.columns else np.nan
    keep_cols = [c for c in ["grid_id", "avg_rsrp", "avg_rsrq", "avg_sinr", "point_count", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"] if c in work.columns]
    out = work.loc[work["grid_id"].notna(), keep_cols].drop_duplicates(subset=["grid_id"], keep="first").copy()
    out["is_bad_sinr"] = pd.to_numeric(out[AVG_COL], errors="coerce") < float(threshold)
    out["sinr_severity"] = (float(threshold) - pd.to_numeric(out[AVG_COL], errors="coerce")).clip(lower=0.0)
    out["is_bad_rsrp"] = out["is_bad_sinr"]
    out["rsrp_severity"] = out["sinr_severity"]
    return out


def _build_grid_ranked_summary_active(
    baseline_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    threshold: float,
    grid_analytics_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_metrics = _grid_reference_metrics_active(grid_analytics_df, threshold) if grid_analytics_df is not None else pd.DataFrame()
    if grid_metrics.empty:
        grid_metrics = _aggregate_grid_metrics_active(baseline_df, threshold)
    if grid_metrics.empty:
        return pd.DataFrame(), grid_metrics
    bad_grid_ids = set(grid_metrics.loc[grid_metrics["is_bad_sinr"].fillna(False), "grid_id"].tolist())
    if not bad_grid_ids:
        return pd.DataFrame(), grid_metrics
    work = baseline_df.copy()
    work["grid_id"] = fast_runtime._normalize_grid_id_series(work.get("grid_id"))
    work = work.loc[work["grid_id"].isin(bad_grid_ids)].copy()
    if work.empty:
        return pd.DataFrame(), grid_metrics
    work["Cell ID"] = work["Node_Cell_ID"].astype(str)
    work[PRED_COL] = pd.to_numeric(work[PRED_COL], errors="coerce")
    work["_severity"] = (float(threshold) - work[PRED_COL]).clip(lower=0.0).fillna(0.0)
    site_map = fast_runtime.base._build_cell_site_map(antenna_df)[["Cell ID", "Site ID"]].copy()
    work = work.merge(site_map, on="Cell ID", how="left")
    summary = (
        work.groupby("Cell ID", dropna=False)
        .agg(**{"Bad SINR": ("grid_id", "count"), "Bad Grid Count": ("grid_id", "nunique"), "mean_bad_grid_sinr": (PRED_COL, "mean"), "bad_grid_severity": ("_severity", "sum")})
        .reset_index()
    )
    summary["Bad RSRP"] = summary["Bad SINR"]
    summary["Bad RSRQ"] = 0
    summary["Bad Samples"] = pd.to_numeric(summary["Bad SINR"], errors="coerce").fillna(0).astype(int)
    summary["total_bad_samples"] = summary["Bad Samples"]
    return summary.sort_values(["Bad Grid Count", "bad_grid_severity", "Bad SINR", "Cell ID"], ascending=[False, False, False, True]).reset_index(drop=True), grid_metrics


def _grid_validation_payload_active(
    baseline_df: pd.DataFrame,
    grid_analytics_df: pd.DataFrame,
    threshold: float,
) -> Dict[str, object]:
    recomputed = _aggregate_grid_metrics_active(baseline_df, threshold)
    mapped_rows = int(fast_runtime._normalize_grid_id_series(baseline_df.get("grid_id")).notna().sum()) if "grid_id" in baseline_df.columns else 0
    payload: Dict[str, object] = {
        "kpi_mode": "sinr",
        "threshold": float(threshold),
        "comparison_operator": "avg_sinr < threshold",
        "baseline_rows": int(len(baseline_df)),
        "grid_mapped_rows": mapped_rows,
        "grid_unmapped_rows": int(len(baseline_df) - mapped_rows),
        "grid_mapped_pct": float(mapped_rows / max(len(baseline_df), 1) * 100.0),
        "recomputed_grid_count": int(len(recomputed)),
        "recomputed_bad_grid_count": int(recomputed["is_bad_sinr"].fillna(False).sum()) if not recomputed.empty else 0,
        "grid_analytics_rows": int(len(grid_analytics_df)),
        "grid_analytics_bad_grid_count": None,
        "common_grid_count": 0,
    }
    reference = _grid_reference_metrics_active(grid_analytics_df, threshold)
    if reference.empty:
        return payload
    payload["grid_analytics_bad_grid_count"] = int(reference["is_bad_sinr"].fillna(False).sum())
    if not recomputed.empty:
        compare = recomputed[["grid_id", "avg_sinr"]].merge(reference[["grid_id", "avg_sinr"]], on="grid_id", how="inner", suffixes=("_recomputed", "_analytics"))
        payload["common_grid_count"] = int(len(compare))
        if not compare.empty:
            delta = (pd.to_numeric(compare["avg_sinr_recomputed"], errors="coerce") - pd.to_numeric(compare["avg_sinr_analytics"], errors="coerce")).abs()
            payload["avg_abs_sinr_delta_vs_grid_analytics"] = float(delta.mean())
            payload["max_abs_sinr_delta_vs_grid_analytics"] = float(delta.max())
    return payload


def _score_candidate_active(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    evaluation_cells: Sequence[str],
    config,
    grid_analytics_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    metrics = core_tilt._score_candidate_vs_baseline(baseline_df, candidate_df, evaluation_cells, _core_config(config))
    metrics["baseline_bad_grid_count"] = metrics.get("baseline_bad_count", 0.0)
    metrics["candidate_bad_grid_count"] = metrics.get("candidate_bad_count", 0.0)
    metrics["bad_to_good_grid_count"] = metrics.get("recovered_bad_samples", 0.0)
    metrics["good_to_bad_grid_count"] = metrics.get("new_bad_samples", 0.0)
    metrics["grid_mean_rsrp_delta_bad_baseline"] = metrics.get("mean_sinr_delta", 0.0)
    metrics["grid_mean_sinr_delta_bad_baseline"] = metrics.get("mean_sinr_delta", 0.0)
    metrics["grid_scoring_source"] = "core_tilt_sinr_formula_fast_runtime"
    metrics["validation_scope"] = "core_kpi_point_population"
    return metrics


def _changed_cell_local_metrics_active(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    updates: Sequence[Dict[str, object]],
    threshold: float,
) -> Dict[str, object]:
    rows: list[Dict[str, object]] = []
    for update in updates:
        cell_id = str(update.get("cell_id", "")).strip()
        if not cell_id:
            continue
        before_cell = before_df.loc[before_df.get("Node_Cell_ID", "").astype(str) == cell_id].copy() if "Node_Cell_ID" in before_df.columns else pd.DataFrame()
        after_cell = after_df.loc[after_df.get("Node_Cell_ID", "").astype(str) == cell_id].copy() if "Node_Cell_ID" in after_df.columns else pd.DataFrame()
        before_values = pd.to_numeric(before_cell.get(PRED_COL), errors="coerce") if not before_cell.empty else pd.Series(dtype=float)
        after_values = pd.to_numeric(after_cell.get(PRED_COL), errors="coerce") if not after_cell.empty else pd.Series(dtype=float)
        before_bad = int((before_values < float(threshold)).fillna(False).sum()) if len(before_values) else 0
        after_bad = int((after_values < float(threshold)).fillna(False).sum()) if len(after_values) else 0
        before_avg = float(before_values.mean()) if len(before_values.dropna()) else np.nan
        after_avg = float(after_values.mean()) if len(after_values.dropna()) else np.nan
        rows.append({
            "cell_id": cell_id,
            "parameter": update.get("parameter", "ETilt"),
            "target_value": update.get("target_value"),
            "before_bad_sample_count": before_bad,
            "after_bad_sample_count": after_bad,
            "bad_sample_reduction": before_bad - after_bad,
            "before_avg_sinr": before_avg,
            "after_avg_sinr": after_avg,
            "avg_sinr_delta": after_avg - before_avg if pd.notna(before_avg) and pd.notna(after_avg) else np.nan,
            "before_sample_count": int(len(before_values)),
            "after_sample_count": int(len(after_values)),
        })
    if not rows:
        return {"changed_cell_local_metrics": "[]", "changed_cell_bad_sample_reduction_sum": 0.0, "changed_cell_avg_sinr_delta_mean": np.nan}
    local_df = pd.DataFrame(rows)
    return {
        "changed_cell_local_metrics": json.dumps(rows, sort_keys=True),
        "changed_cell_bad_sample_reduction_sum": float(pd.to_numeric(local_df["bad_sample_reduction"], errors="coerce").fillna(0).sum()),
        "changed_cell_avg_sinr_delta_mean": float(pd.to_numeric(local_df["avg_sinr_delta"], errors="coerce").mean()),
    }


def _prepare_scope_export_active(df: pd.DataFrame, threshold: float, stage: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["lat"] = pd.to_numeric(out.get("lat"), errors="coerce")
    out["lon"] = pd.to_numeric(out.get("lon"), errors="coerce")
    out[PRED_COL] = pd.to_numeric(out.get(PRED_COL), errors="coerce")
    out["is_bad_sinr"] = out[PRED_COL] < float(threshold)
    out["is_bad_rsrp"] = out["is_bad_sinr"]
    out["stage"] = stage
    return out


def _activate_sinr_mode() -> None:
    fast_runtime.base.TILT_SRC.filter_bad_samples = _filter_bad_samples_active
    fast_runtime._aggregate_grid_metrics_from_predictions = _aggregate_grid_metrics_active
    fast_runtime._grid_reference_metrics_from_analytics = _grid_reference_metrics_active
    fast_runtime._grid_validation_payload = _grid_validation_payload_active
    fast_runtime._build_grid_ranked_summary = _build_grid_ranked_summary_active
    fast_runtime._score_candidate_vs_baseline_grids = _score_candidate_active
    fast_runtime._changed_cell_local_metrics_global = _changed_cell_local_metrics_active
    fast_runtime._prepare_scope_export = _prepare_scope_export_active


def _replace_path_values(payload: Any, old: str, new: str) -> Any:
    if isinstance(payload, dict):
        return {key: _replace_path_values(value, old, new) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_replace_path_values(value, old, new) for value in payload]
    if isinstance(payload, str):
        return payload.replace(old, new)
    return payload


def _move_run_dir_to_sinr_name(run_dir: Path) -> Path:
    target = run_dir.with_name(run_dir.name.replace("tilt_rsrp_only", "tilt_sinr_only"))
    if target == run_dir:
        return run_dir
    if target.exists():
        target = run_dir.with_name(target.name + "_sinr")
    shutil.move(str(run_dir), str(target))
    summary_path = target / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload = _replace_path_values(payload, str(run_dir), str(target))
        payload["run_type"] = "tilt_sinr_only_test"
        payload.setdefault("thresholds", {})["sinr"] = payload.get("thresholds", {}).get("rsrp")
        payload.setdefault("thresholds", {})["rsrp"] = None
        payload.setdefault("thresholds", {})["kpi_mode"] = "SINR only"
        payload.setdefault("search", {})["kpi_formula_source"] = "tests.lte_tilt_recommendation_test._score_candidate_vs_baseline"
        summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def _parse_args():
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *_convert_args_for_fast_parser(old_argv[1:])]
        return fast_runtime._parse_args()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    _activate_sinr_mode()
    run_dir = fast_runtime.run_tilt_rsrp_only_recommendation_test(_parse_args())
    run_dir = _move_run_dir_to_sinr_name(Path(run_dir))
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
