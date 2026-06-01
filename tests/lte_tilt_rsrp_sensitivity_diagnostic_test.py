from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.lte_rf_debug_lab import DEFAULT_PROJECT_ID, DEFAULT_REGION, DEFAULT_SESSION_IDS, DEFAULT_VALIDATION_FRACTION, _write_json
from tests import lte_rf_debug_lab as rf_lab
from tests import lte_tilt_recommendation_test as base
from tests import lte_tilt_rsrp_only_recommendation_test as rsrp_only


OUTPUT_ROOT = Path("tests/output")


@dataclass
class TiltRsrpSensitivityDiagnosticConfig:
    project_id: int = DEFAULT_PROJECT_ID
    region: str = DEFAULT_REGION
    operator: Optional[str] = None
    rsrp_threshold: float = -90.0
    radius_m: float = 500.0
    grid_resolution_m: float = 30.0
    workers: int = 1
    impact_radius_m: float = 1.0
    neighbor_site_count: int = 0
    max_interference_sites: int = 10
    max_bad_grids: int = 20
    max_cells_per_grid: int = 0
    deltas: Sequence[float] = (-2.0, -1.0, 1.0, 2.0)
    session_ids: Sequence[int] = tuple(DEFAULT_SESSION_IDS)
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    apply_residual_calibration: bool = True
    baseline_points_path: Optional[str] = None
    antenna_input_path: Optional[str] = None
    geo_features_path: Optional[str] = None
    grid_analytics_path: Optional[str] = None
    local_baseline_kpi_stage: str = "geo"
    fixed_k1k2_for_local_inputs: bool = True
    fixed_dt_calibration_k1: float = 170.0
    fixed_dt_calibration_k2: float = 35.2
    use_fixed_dt_calibration_fallback: bool = True
    threshold_file_path: Optional[str] = None
    output_root: Path = OUTPUT_ROOT


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def _load_inputs(config: TiltRsrpSensitivityDiagnosticConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, bool]:
    using_local_inputs = bool(
        config.baseline_points_path
        or config.antenna_input_path
        or config.geo_features_path
        or config.grid_analytics_path
    )
    if using_local_inputs:
        print("[TILT_RSRP_SENSITIVITY_LOCAL_INPUT] enabled=True db_fetch_for_inputs=False")
        baseline_df = rsrp_only._prepare_local_baseline_points(
            rsrp_only._read_local_table(config.baseline_points_path),
            config.local_baseline_kpi_stage,
        )
        antenna_df = rsrp_only._read_local_table(config.antenna_input_path)
        geo_df = rsrp_only._read_local_table(config.geo_features_path)
        grid_analytics_df = rsrp_only._read_local_table(config.grid_analytics_path)
        baseline_df, antenna_df = rsrp_only._prepare_rsrp_only_inputs(baseline_df, antenna_df)
        baseline_df = base._enrich_log_with_antenna_context(baseline_df, antenna_df)
        if "nodeb_id_cell_id" in geo_df.columns and "Node_Cell_ID" not in geo_df.columns:
            geo_df["Node_Cell_ID"] = geo_df["nodeb_id_cell_id"].astype(str)
        baseline_df["grid_id"] = pd.NA
        baseline_df = rsrp_only._attach_grid_context_to_predictions(baseline_df, geo_df, grid_analytics_df)
        return baseline_df, antenna_df, geo_df, grid_analytics_df, "local_artifact_baseline", True

    baseline_job_id = base._fetch_latest_baseline_job_id(config.project_id, config.region)
    baseline_df = base._fetch_baseline_log_df(config.project_id, config.region, config.operator)
    baseline_df = rsrp_only._attach_baseline_identity_columns_from_db(
        baseline_df,
        config.project_id,
        config.region,
        baseline_job_id,
    )
    antenna_df = base._fetch_antenna_df(config.project_id, config.region, config.operator)
    baseline_df, antenna_df = rsrp_only._prepare_rsrp_only_inputs(baseline_df, antenna_df)
    baseline_df, antenna_df, _ = rsrp_only._filter_run_inputs_to_project_polygon(
        log_df=baseline_df,
        antenna_df=antenna_df,
        project_id=config.project_id,
        region=config.region,
    )
    baseline_df = base._enrich_log_with_antenna_context(baseline_df, antenna_df)
    geo_df = base._fetch_geo_df(config.project_id, config.region, config.operator, antenna_df)
    grid_analytics_df = rsrp_only._fetch_grid_analytics_df(config.project_id, config.region, config.operator)
    baseline_df = rsrp_only._attach_grid_context_to_predictions(baseline_df, geo_df, grid_analytics_df)
    return baseline_df, antenna_df, geo_df, grid_analytics_df, str(baseline_job_id), False


def _prepare_residual_prediction_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    metric_pairs = [
        ("pred_rsrp", "pred_rsrp_geo"),
        ("pred_rsrq", "pred_rsrq_geo"),
        ("pred_sinr", "pred_sinr_geo"),
    ]
    for base_col, geo_col in metric_pairs:
        if geo_col not in out.columns and base_col in out.columns:
            out[geo_col] = pd.to_numeric(out[base_col], errors="coerce")
    return out


def _fit_fixed_residual_calibration(
    baseline_df: pd.DataFrame,
    config: TiltRsrpSensitivityDiagnosticConfig,
) -> tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    if not bool(config.apply_residual_calibration):
        return {}, {"enabled": False, "reason": "disabled"}
    operator = config.operator or "Airtel"
    try:
        drive_df = rf_lab._fetch_drive_data_for_test(
            session_ids=config.session_ids,
            operator=operator,
            project_id=int(config.project_id),
            region=str(config.region),
        )
        drive_train_df, _ = rf_lab._split_drive_train_holdout(drive_df, float(config.validation_fraction))
        residual_source_df = _prepare_residual_prediction_frame(baseline_df)
        train_eval, _, train_metrics = rf_lab._evaluate_prediction_grid_against_holdout(
            drive_train_df,
            residual_source_df,
        )
        residual_models, residual_debug = rf_lab._fit_train_only_residual_calibration(train_eval)
        residual_debug["source"] = "diagnostic_baseline_prediction_frame"
        residual_debug["operator"] = operator
        residual_debug["session_ids"] = [int(value) for value in config.session_ids]
        residual_debug["train_metrics"] = train_metrics
        print(
            f"[TILT_RSRP_SENSITIVITY_RESIDUAL] enabled={bool(residual_models)} "
            f"drive_rows={len(drive_df)} train_rows={len(drive_train_df)} "
            f"eval_rows={len(train_eval)} models={list(residual_models.keys())}"
        )
        return residual_models, residual_debug
    except Exception as exc:
        print(f"[TILT_RSRP_SENSITIVITY_RESIDUAL] enabled=False reason={exc}")
        return {}, {"enabled": False, "reason": str(exc)}


def _apply_fixed_residual_calibration_to_rf(
    rf_df: pd.DataFrame,
    residual_models: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    out = _prepare_residual_prediction_frame(rf_df)
    if residual_models:
        out = rf_lab._apply_train_only_residual_calibration(out, residual_models)
    for base_col, geo_col in [
        ("pred_rsrp", "pred_rsrp_geo"),
        ("pred_rsrq", "pred_rsrq_geo"),
        ("pred_sinr", "pred_sinr_geo"),
    ]:
        if geo_col in out.columns:
            out[base_col] = pd.to_numeric(out[geo_col], errors="coerce")
    return out


def _fixed_dt_calibration_k1k2_map(
    cells: Sequence[str],
    config: TiltRsrpSensitivityDiagnosticConfig,
) -> Dict[str, tuple[float, float]]:
    return {
        str(cell): (float(config.fixed_dt_calibration_k1), float(config.fixed_dt_calibration_k2))
        for cell in cells
        if str(cell).strip()
    }


def _cell_current_etilt_map(antenna_df: pd.DataFrame, baseline_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    ant_use = base._build_cell_site_map(antenna_df)
    if ant_use.empty or "Cell ID" not in ant_use.columns:
        return {}
    value_col = "electrical_tilt" if "electrical_tilt" in ant_use.columns else "ETilt"
    if value_col not in ant_use.columns:
        return {}
    out: Dict[str, float] = {}
    for _, row in ant_use.drop_duplicates(subset=["Cell ID"], keep="last").iterrows():
        raw_cell_id = str(row["Cell ID"]).strip()
        value = pd.to_numeric(pd.Series([row.get(value_col)]), errors="coerce").iloc[0]
        if pd.notna(value):
            out[raw_cell_id] = float(value)
            if baseline_df is not None and not baseline_df.empty:
                canonical = rsrp_only._canonicalize_cell_list([raw_cell_id], baseline_df)
                for canonical_cell_id in canonical:
                    out[str(canonical_cell_id)] = float(value)
    return out


def _build_bad_grid_contributors(
    baseline_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    grid_analytics_df: pd.DataFrame,
    config: TiltRsrpSensitivityDiagnosticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_grid = rsrp_only._grid_reference_metrics_from_analytics(grid_analytics_df, config.rsrp_threshold)
    if reference_grid.empty:
        reference_grid = rsrp_only._aggregate_grid_metrics_from_predictions(baseline_df, config.rsrp_threshold)
    if reference_grid.empty:
        return pd.DataFrame(), pd.DataFrame()

    bad_grid_df = reference_grid.loc[reference_grid["is_bad_rsrp"].fillna(False)].copy()
    bad_grid_df = bad_grid_df.sort_values(["rsrp_severity", "avg_rsrp"], ascending=[False, True]).head(
        int(config.max_bad_grids)
    )
    bad_grid_ids = set(bad_grid_df["grid_id"].astype(str).tolist())
    if not bad_grid_ids:
        return bad_grid_df, pd.DataFrame()

    ant_use = base._build_cell_site_map(antenna_df)
    if not ant_use.empty:
        tunable_cells = set(
            rsrp_only._canonicalize_cell_list(
                ant_use["Cell ID"].astype(str).str.strip().dropna().tolist(),
                baseline_df,
            )
        )
    else:
        tunable_cells = set()
    work = baseline_df.copy()
    work["grid_id"] = rsrp_only._normalize_grid_id_series(work.get("grid_id"))
    work["Node_Cell_ID"] = work["Node_Cell_ID"].astype(str).str.strip()
    work["pred_rsrp"] = pd.to_numeric(work["pred_rsrp"], errors="coerce")
    work = work.loc[work["grid_id"].astype(str).isin(bad_grid_ids)].copy()
    if tunable_cells:
        work = work.loc[work["Node_Cell_ID"].isin(tunable_cells)].copy()
    work = work.dropna(subset=["grid_id", "Node_Cell_ID", "pred_rsrp"])
    if work.empty:
        return bad_grid_df, pd.DataFrame()

    threshold = float(config.rsrp_threshold)
    contributors = (
        work.groupby(["grid_id", "Node_Cell_ID"], dropna=False)
        .agg(
            grid_sample_count=("pred_rsrp", "count"),
            before_cell_mean_rsrp=("pred_rsrp", "mean"),
            before_cell_min_rsrp=("pred_rsrp", "min"),
            before_cell_p10_rsrp=("pred_rsrp", lambda values: float(pd.to_numeric(values, errors="coerce").quantile(0.10))),
        )
        .reset_index()
    )
    contributors["mean_weakness_db"] = (
        threshold - pd.to_numeric(contributors["before_cell_mean_rsrp"], errors="coerce")
    ).clip(lower=0.0).fillna(0.0)
    contributors["p10_weakness_db"] = (
        threshold - pd.to_numeric(contributors["before_cell_p10_rsrp"], errors="coerce")
    ).clip(lower=0.0).fillna(0.0)
    contributors["severity_sum"] = (
        contributors["mean_weakness_db"] * pd.to_numeric(contributors["grid_sample_count"], errors="coerce").fillna(0.0)
    )
    contributors["bad_sample_count"] = pd.to_numeric(contributors["grid_sample_count"], errors="coerce").fillna(0).astype(int)
    contributors = contributors.loc[pd.to_numeric(contributors["grid_sample_count"], errors="coerce").fillna(0) > 0].copy()
    contributors = contributors.sort_values(
        ["grid_id", "severity_sum", "mean_weakness_db", "p10_weakness_db", "grid_sample_count", "Node_Cell_ID"],
        ascending=[True, False, False, False, False, True],
    )
    if int(config.max_cells_per_grid) > 0:
        contributors = contributors.groupby("grid_id", group_keys=False).head(int(config.max_cells_per_grid)).copy()
    print(
        f"[TILT_RSRP_SENSITIVITY_CONTRIBUTORS] bad_grids_selected={len(bad_grid_df)} "
        f"bad_grid_rows={len(work)} contributor_pairs={len(contributors)} "
        f"unique_cells={contributors['Node_Cell_ID'].astype(str).nunique() if not contributors.empty else 0}"
    )
    return bad_grid_df, contributors.reset_index(drop=True)


def _make_etilt_update(
    cell_id: str,
    requested_delta: float,
    current_etilt: float,
    constraint_map: Dict[str, Dict[str, object]],
) -> Optional[Dict[str, object]]:
    target = float(np.clip(float(current_etilt) + float(requested_delta), float(base.MIN_SAFE_ETILT_DEG), float(base.MAX_SAFE_ETILT_DEG)))
    target = base._clip_target_to_user_constraint_test_only(cell_id, "ETilt", target, constraint_map)
    if pd.isna(target) or np.isclose(float(target), float(current_etilt), equal_nan=True):
        return None
    return {
        "cell_id": str(cell_id),
        "parameter": "ETilt",
        "current_value": float(current_etilt),
        "target_value": float(target),
        "requested_delta": float(requested_delta),
        "actual_delta": float(target) - float(current_etilt),
    }


def _grid_counts(df: pd.DataFrame, thresholds: Sequence[float] = (-90.0, -95.0, -100.0, -105.0)) -> Dict[str, float]:
    metrics = rsrp_only._aggregate_grid_metrics_from_predictions(df, rsrp_threshold=-90.0)
    out: Dict[str, float] = {}
    if metrics.empty:
        for threshold in thresholds:
            out[f"bad_grid_count_lt_{abs(int(threshold))}"] = 0.0
        return out
    rsrp = pd.to_numeric(metrics["avg_rsrp"], errors="coerce")
    for threshold in thresholds:
        out[f"bad_grid_count_lt_{abs(int(threshold))}"] = float((rsrp < float(threshold)).fillna(False).sum())
    return out


def _evaluate_sensitivity_candidate(
    grid_id: str,
    cell_id: str,
    update: Dict[str, object],
    baseline_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    baseline_job_id: str,
    config: TiltRsrpSensitivityDiagnosticConfig,
    residual_models: Dict[str, Dict[str, object]],
    geo_features_df: Optional[pd.DataFrame] = None,
    use_fixed_raw_k1k2: bool = False,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    grid_scope = baseline_df.loc[baseline_df["grid_id"].astype(str) == str(grid_id)].copy()
    if grid_scope.empty:
        raise ValueError(f"No baseline rows for grid_id={grid_id}")

    modified_site_df = base._apply_multiple_parameter_targets(antenna_df, [update])
    affected_cells, affected_sites, _ = base.opt_ml._compute_affected_cells(
        modified_site_df,
        float(config.impact_radius_m),
        int(config.neighbor_site_count),
    )
    update_cells = rsrp_only._canonicalize_cell_list([cell_id], baseline_df)
    affected_cells = rsrp_only._canonicalize_cell_list(set(affected_cells).union(update_cells), baseline_df)
    prediction_points_df = grid_scope.loc[grid_scope["Node_Cell_ID"].astype(str).isin(set(affected_cells))].copy()
    if prediction_points_df.empty:
        prediction_points_df = grid_scope.loc[grid_scope["Node_Cell_ID"].astype(str) == str(cell_id)].copy()
    if prediction_points_df.empty:
        raise ValueError(f"No prediction points for grid_id={grid_id} cell_id={cell_id}")

    use_fixed_dt_fallback = bool(use_fixed_raw_k1k2 and not residual_models and config.use_fixed_dt_calibration_fallback)
    if use_fixed_dt_fallback:
        k1k2_map = _fixed_dt_calibration_k1k2_map(affected_cells, config)
        calibration_mode = f"fixed_dt_calibration_k1k2_{float(config.fixed_dt_calibration_k1):g}_{float(config.fixed_dt_calibration_k2):g}"
    elif use_fixed_raw_k1k2:
        k1k2_map = rsrp_only._fixed_raw_k1k2_map(affected_cells)
        calibration_mode = "fixed_raw_k1k2_zero_plus_fixed_residual"
    else:
        k1k2_map = base.opt_ml.compute_k1k2_for_cells(baseline_df, antenna_df, affected_cells)
        calibration_mode = "fixed_baseline_k1k2_plus_fixed_residual" if residual_models else "fixed_baseline_k1k2"
    if not k1k2_map:
        raise ValueError("No K1/K2 map for diagnostic candidate")

    baseline_prediction_site_df = rsrp_only._mark_recompute_cells_for_prediction(antenna_df, affected_cells, [], baseline_df)
    candidate_prediction_site_df = rsrp_only._mark_recompute_cells_for_prediction(modified_site_df, affected_cells, update_cells, baseline_df)
    prediction_params = {
        "project_id": int(config.project_id),
        "region": str(config.region),
        "radius": float(config.radius_m),
        "grid_resolution": float(config.grid_resolution_m),
        "n_workers": int(config.workers),
        "impact_radius_m": float(config.impact_radius_m),
        "neighbor_site_count": int(config.neighbor_site_count),
        "max_interference_sites": int(config.max_interference_sites),
        "baseline_job_id": baseline_job_id,
        "prediction_points_df": prediction_points_df,
        "geo_features_df": geo_features_df,
    }
    baseline_rf_df = base.opt_ml.run_prediction_only_optimized(
        baseline_prediction_site_df,
        k1k2_map,
        prediction_params,
    )
    candidate_rf_df = base.opt_ml.run_prediction_only_optimized(
        candidate_prediction_site_df,
        k1k2_map,
        prediction_params,
    )
    baseline_rf_df = _apply_fixed_residual_calibration_to_rf(baseline_rf_df, residual_models)
    candidate_rf_df = _apply_fixed_residual_calibration_to_rf(candidate_rf_df, residual_models)
    after_grid_scope, rf_delta_metrics = rsrp_only._apply_rf_delta_to_baseline_points(
        grid_scope,
        baseline_rf_df,
        candidate_rf_df,
        grid_scope,
    )
    after_grid_scope = rsrp_only._attach_grid_context_to_predictions(after_grid_scope, grid_scope)
    meta = {
        "affected_cell_count": float(len(affected_cells)),
        "affected_site_count": float(len(affected_sites)),
        "prediction_point_count": float(len(prediction_points_df)),
        "rf_baseline_row_count": float(len(baseline_rf_df)),
        "rf_candidate_row_count": float(len(candidate_rf_df)),
        "residual_calibration_applied": float(1 if residual_models else 0),
        "fixed_raw_k1k2_used": float(1 if use_fixed_raw_k1k2 else 0),
        "fixed_dt_calibration_fallback_used": float(1 if use_fixed_dt_fallback else 0),
        "fixed_dt_calibration_k1": float(config.fixed_dt_calibration_k1) if use_fixed_dt_fallback else np.nan,
        "fixed_dt_calibration_k2": float(config.fixed_dt_calibration_k2) if use_fixed_dt_fallback else np.nan,
        "calibration_mode": calibration_mode,
        **rf_delta_metrics,
    }
    return after_grid_scope, meta


def run_tilt_rsrp_sensitivity_diagnostic(config: TiltRsrpSensitivityDiagnosticConfig) -> Path:
    start = time.perf_counter()
    run_dir = _ensure_dir(config.output_root / f"project_{config.project_id}" / f"tilt_rsrp_sensitivity_{_timestamp()}")
    base.TILT_SRC.RSRP_THRESH = float(config.rsrp_threshold)
    base.TILT_SRC.RSRQ_THRESH = -999.0
    base.TILT_SRC.SINR_THRESH = -999.0

    threshold_path = base._resolve_threshold_file_path_test_only(config.threshold_file_path)
    constraint_df = base._load_threshold_constraints_test_only(threshold_path) if threshold_path else pd.DataFrame()
    constraint_map = base._constraint_map_test_only(constraint_df)

    baseline_df, antenna_df, geo_df, grid_analytics_df, baseline_job_id, using_local_inputs = _load_inputs(config)
    residual_models, residual_debug = _fit_fixed_residual_calibration(baseline_df, config)
    use_fixed_raw_k1k2 = bool(using_local_inputs and config.fixed_k1k2_for_local_inputs)
    print(
        f"[TILT_RSRP_SENSITIVITY_RF_PIPELINE] baseline_anchor=stored_calibrated_baseline "
        f"candidate_delta_residual_calibration={bool(residual_models)} "
        f"fixed_raw_k1k2={use_fixed_raw_k1k2} "
        f"fixed_dt_calibration_fallback={bool(use_fixed_raw_k1k2 and not residual_models and config.use_fixed_dt_calibration_fallback)} "
        f"fixed_dt_k1={float(config.fixed_dt_calibration_k1):g} fixed_dt_k2={float(config.fixed_dt_calibration_k2):g}"
    )
    grid_validation = rsrp_only._grid_validation_payload(baseline_df, grid_analytics_df, config.rsrp_threshold)
    rsrp_only._log_grid_validation("SENSITIVITY_BASELINE_VALIDATE", grid_validation)

    bad_grid_df, contributors_df = _build_bad_grid_contributors(baseline_df, antenna_df, grid_analytics_df, config)
    etilt_map = _cell_current_etilt_map(antenna_df, baseline_df)
    rows: List[Dict[str, object]] = []

    for idx, contributor in contributors_df.iterrows():
        grid_id = str(contributor["grid_id"])
        cell_id = str(contributor["Node_Cell_ID"])
        current_etilt = etilt_map.get(cell_id)
        if current_etilt is None or pd.isna(current_etilt):
            rows.append(
                {
                    "grid_id": grid_id,
                    "cell_id": cell_id,
                    "status": "skipped_missing_current_etilt",
                    "current_etilt": np.nan,
                }
            )
            continue
        before_grid_scope = baseline_df.loc[baseline_df["grid_id"].astype(str) == grid_id].copy()
        before_grid_avg_rsrp = float(pd.to_numeric(before_grid_scope["pred_rsrp"], errors="coerce").mean())
        before_bad = bool(before_grid_avg_rsrp < float(config.rsrp_threshold))

        for delta in config.deltas:
            update = _make_etilt_update(cell_id, float(delta), float(current_etilt), constraint_map)
            row: Dict[str, object] = {
                "grid_id": grid_id,
                "cell_id": cell_id,
                "current_etilt": float(current_etilt),
                "requested_delta_etilt": float(delta),
                "test_etilt": np.nan if update is None else float(update["target_value"]),
                "actual_delta_etilt": 0.0 if update is None else float(update["actual_delta"]),
                "before_grid_avg_rsrp": before_grid_avg_rsrp,
                "before_bad": before_bad,
                "contributor_bad_sample_count": int(contributor.get("bad_sample_count", 0)),
                "contributor_sample_count": int(contributor.get("grid_sample_count", 0)),
                "contributor_severity_sum": float(contributor.get("severity_sum", 0.0)),
                "status": "skipped_no_effective_delta" if update is None else "evaluated",
            }
            if update is None:
                row.update(
                    {
                        "after_grid_avg_rsrp": before_grid_avg_rsrp,
                        "grid_rsrp_delta": 0.0,
                        "after_bad": before_bad,
                        "bad_to_good": False,
                        "good_to_bad": False,
                    }
                )
                rows.append(row)
                continue
            try:
                after_grid_scope, meta = _evaluate_sensitivity_candidate(
                    grid_id=grid_id,
                    cell_id=cell_id,
                    update=update,
                    baseline_df=baseline_df,
                    antenna_df=antenna_df,
                    baseline_job_id=baseline_job_id,
                    config=config,
                    residual_models=residual_models,
                    geo_features_df=geo_df,
                    use_fixed_raw_k1k2=use_fixed_raw_k1k2,
                )
                after_grid_avg_rsrp = float(pd.to_numeric(after_grid_scope["pred_rsrp"], errors="coerce").mean())
                after_bad = bool(after_grid_avg_rsrp < float(config.rsrp_threshold))
                row.update(
                    {
                        "after_grid_avg_rsrp": after_grid_avg_rsrp,
                        "grid_rsrp_delta": after_grid_avg_rsrp - before_grid_avg_rsrp,
                        "after_bad": after_bad,
                        "bad_to_good": bool(before_bad and not after_bad),
                        "good_to_bad": bool((not before_bad) and after_bad),
                        **meta,
                    }
                )
                row.update({f"before_{k}": v for k, v in _grid_counts(before_grid_scope).items()})
                row.update({f"after_{k}": v for k, v in _grid_counts(after_grid_scope).items()})
            except Exception as exc:
                row.update(
                    {
                        "status": "error",
                        "error": str(exc),
                        "after_grid_avg_rsrp": np.nan,
                        "grid_rsrp_delta": np.nan,
                        "after_bad": np.nan,
                        "bad_to_good": False,
                        "good_to_bad": False,
                    }
                )
            rows.append(row)
            print(
                f"[TILT_RSRP_SENSITIVITY] grid={grid_id} cell={cell_id} "
                f"delta={float(delta):+.1f} actual={float(row.get('actual_delta_etilt', 0.0)):+.1f} "
                f"before={before_grid_avg_rsrp:.4f} after={float(row.get('after_grid_avg_rsrp', np.nan)):.4f} "
                f"rsrp_delta={float(row.get('grid_rsrp_delta', np.nan)):.4f} status={row.get('status')}"
            )

    sensitivity_df = pd.DataFrame(rows)
    if not sensitivity_df.empty:
        sensitivity_df = sensitivity_df.sort_values(
            ["grid_rsrp_delta", "contributor_severity_sum", "contributor_bad_sample_count"],
            ascending=[False, False, False],
            na_position="last",
        )
    artifact_paths = {
        "bad_grids": _write_csv(bad_grid_df, run_dir / "sensitivity_bad_grids.csv"),
        "contributors": _write_csv(contributors_df, run_dir / "sensitivity_contributors.csv"),
        "tilt_sensitivity_by_bad_grid": _write_csv(sensitivity_df, run_dir / "tilt_sensitivity_by_bad_grid.csv"),
    }
    positive_df = sensitivity_df.loc[pd.to_numeric(sensitivity_df.get("grid_rsrp_delta"), errors="coerce") > 0].copy() if not sensitivity_df.empty else pd.DataFrame()
    summary = {
        "project_id": int(config.project_id),
        "region": str(config.region),
        "operator": config.operator,
        "rsrp_threshold": float(config.rsrp_threshold),
        "baseline_job_id": baseline_job_id,
        "bad_grid_count_selected": int(len(bad_grid_df)),
        "contributor_pair_count": int(len(contributors_df)),
        "evaluated_candidate_count": int((sensitivity_df.get("status") == "evaluated").sum()) if not sensitivity_df.empty else 0,
        "positive_grid_rsrp_delta_count": int(len(positive_df)),
        "best_grid_rsrp_delta": float(pd.to_numeric(sensitivity_df.get("grid_rsrp_delta"), errors="coerce").max()) if not sensitivity_df.empty else None,
        "best_row": sensitivity_df.head(1).to_dict("records")[0] if not sensitivity_df.empty else None,
        "grid_validation": grid_validation,
        "residual_calibration": residual_debug,
        "candidate_rf_pipeline": {
            "baseline_anchor": "stored_calibrated_baseline",
            "candidate_delta": "raw_rf_plus_geo_correction_plus_fixed_dt_residual_calibration",
            "local_inputs": bool(using_local_inputs),
            "fixed_raw_k1k2_for_local_inputs": bool(use_fixed_raw_k1k2),
            "residual_calibration_applied": bool(residual_models),
            "fixed_dt_calibration_fallback_used": bool(use_fixed_raw_k1k2 and not residual_models and config.use_fixed_dt_calibration_fallback),
            "fixed_dt_calibration_k1": float(config.fixed_dt_calibration_k1),
            "fixed_dt_calibration_k2": float(config.fixed_dt_calibration_k2),
        },
        "artifacts": artifact_paths,
        "total_runtime_sec": round(float(time.perf_counter() - start), 4),
    }
    _write_json(run_dir / "summary.json", summary)
    print(
        f"[TILT_RSRP_SENSITIVITY_DONE] run_dir={run_dir} "
        f"evaluated={summary['evaluated_candidate_count']} positive={summary['positive_grid_rsrp_delta_count']} "
        f"best_delta={summary['best_grid_rsrp_delta']} runtime_sec={summary['total_runtime_sec']}"
    )
    return run_dir


def _parse_deltas(raw: str) -> Sequence[float]:
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return tuple(values or [-2.0, -1.0, 1.0, 2.0])


def _parse_args() -> TiltRsrpSensitivityDiagnosticConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--rsrp", type=float, default=-90.0)
    parser.add_argument("--radius-m", type=float, default=500.0)
    parser.add_argument("--grid-resolution-m", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--impact-radius-m", type=float, default=1.0)
    parser.add_argument("--neighbor-site-count", type=int, default=0)
    parser.add_argument("--max-interference-sites", type=int, default=10)
    parser.add_argument("--max-bad-grids", type=int, default=20)
    parser.add_argument("--max-cells-per-grid", type=int, default=0)
    parser.add_argument("--deltas", type=str, default="-2,-1,1,2")
    parser.add_argument("--session-ids", type=str, default=",".join(str(value) for value in DEFAULT_SESSION_IDS))
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--no-residual-calibration", action="store_true")
    parser.add_argument("--baseline-points", "--baseline-points-path", dest="baseline_points_path", type=str, default=None)
    parser.add_argument("--antenna-input", "--antenna-input-path", dest="antenna_input_path", type=str, default=None)
    parser.add_argument("--geo-features", "--geo-features-path", dest="geo_features_path", type=str, default=None)
    parser.add_argument("--grid-analytics", "--grid-analytics-path", dest="grid_analytics_path", type=str, default=None)
    parser.add_argument("--local-baseline-kpi-stage", choices=["geo", "demo", "raw"], default="geo")
    parser.add_argument("--recompute-k1k2-from-baseline", action="store_true")
    parser.add_argument("--fixed-dt-calibration-k1", type=float, default=170.0)
    parser.add_argument("--fixed-dt-calibration-k2", type=float, default=35.2)
    parser.add_argument("--no-fixed-dt-calibration-fallback", action="store_true")
    parser.add_argument("--threshold-file", "--threshold-file-path", dest="threshold_file_path", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    return TiltRsrpSensitivityDiagnosticConfig(
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
        max_bad_grids=args.max_bad_grids,
        max_cells_per_grid=args.max_cells_per_grid,
        deltas=_parse_deltas(args.deltas),
        session_ids=tuple(int(part.strip()) for part in str(args.session_ids).split(",") if part.strip()),
        validation_fraction=args.validation_fraction,
        apply_residual_calibration=not bool(args.no_residual_calibration),
        baseline_points_path=args.baseline_points_path,
        antenna_input_path=args.antenna_input_path,
        geo_features_path=args.geo_features_path,
        grid_analytics_path=args.grid_analytics_path,
        local_baseline_kpi_stage=args.local_baseline_kpi_stage,
        fixed_k1k2_for_local_inputs=not bool(args.recompute_k1k2_from_baseline),
        fixed_dt_calibration_k1=float(args.fixed_dt_calibration_k1),
        fixed_dt_calibration_k2=float(args.fixed_dt_calibration_k2),
        use_fixed_dt_calibration_fallback=not bool(args.no_fixed_dt_calibration_fallback),
        threshold_file_path=args.threshold_file_path,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    run_dir = run_tilt_rsrp_sensitivity_diagnostic(_parse_args())
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
