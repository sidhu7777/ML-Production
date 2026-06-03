from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import lte_tilt_rsrp_only_recommendation_test as rsrp_test


KPI_NAME = "rsrq"
DEFAULT_RSRQ_THRESHOLD = -12.0


def _alias_selected_kpi_to_rsrp(df: pd.DataFrame, kpi_name: str = KPI_NAME) -> pd.DataFrame:
    out = df.copy()
    pred_col = f"pred_{kpi_name}"
    geo_col = f"pred_{kpi_name}_geo"
    demo_col = f"pred_{kpi_name}_demo"
    if pred_col in out.columns:
        if "pred_rsrp_original" not in out.columns and "pred_rsrp" in out.columns:
            out["pred_rsrp_original"] = out["pred_rsrp"]
        out["pred_rsrp"] = pd.to_numeric(out[pred_col], errors="coerce")
    if geo_col in out.columns:
        out["pred_rsrp_geo"] = pd.to_numeric(out[geo_col], errors="coerce")
    if demo_col in out.columns:
        out["pred_rsrp_demo"] = pd.to_numeric(out[demo_col], errors="coerce")
    return out


def _alias_grid_analytics(df: pd.DataFrame, kpi_name: str = KPI_NAME) -> pd.DataFrame:
    out = df.copy()
    source_col = f"baseline_avg_{kpi_name}"
    if source_col in out.columns:
        if "baseline_avg_rsrp_original" not in out.columns and "baseline_avg_rsrp" in out.columns:
            out["baseline_avg_rsrp_original"] = out["baseline_avg_rsrp"]
        out["baseline_avg_rsrp"] = pd.to_numeric(out[source_col], errors="coerce")
    return out


def _activate_rsrq_mode() -> None:
    original_prepare_local = rsrp_test._prepare_local_baseline_points
    original_read_local = rsrp_test._read_local_table
    original_apply_residual = rsrp_test._apply_fixed_residual_calibration_to_rf
    original_fetch_grid = rsrp_test._fetch_grid_analytics_df
    original_fetch_baseline = rsrp_test.base._fetch_baseline_log_df

    def prepare_local_wrapper(df: pd.DataFrame, kpi_stage: str) -> pd.DataFrame:
        return _alias_selected_kpi_to_rsrp(original_prepare_local(df, kpi_stage))

    def read_local_wrapper(path_value: str | None) -> pd.DataFrame:
        return _alias_grid_analytics(original_read_local(path_value))

    def apply_residual_wrapper(rf_df: pd.DataFrame, residual_models):
        return _alias_selected_kpi_to_rsrp(original_apply_residual(rf_df, residual_models))

    def fetch_grid_wrapper(project_id: int, region: str, operator: str | None) -> pd.DataFrame:
        return _alias_grid_analytics(original_fetch_grid(project_id, region, operator))

    def fetch_baseline_wrapper(project_id: int, region: str, operator: str | None) -> pd.DataFrame:
        return _alias_selected_kpi_to_rsrp(original_fetch_baseline(project_id, region, operator))

    rsrp_test._prepare_local_baseline_points = prepare_local_wrapper
    rsrp_test._read_local_table = read_local_wrapper
    rsrp_test._apply_fixed_residual_calibration_to_rf = apply_residual_wrapper
    rsrp_test._fetch_grid_analytics_df = fetch_grid_wrapper
    rsrp_test.base._fetch_baseline_log_df = fetch_baseline_wrapper


def _convert_args_for_rsrp_parser(argv: list[str]) -> list[str]:
    converted = []
    threshold_was_set = False
    skip_next = False
    for idx, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--rsrq":
            converted.append("--rsrp")
            if idx + 1 < len(argv):
                converted.append(argv[idx + 1])
                threshold_was_set = True
                skip_next = True
            continue
        if token.startswith("--rsrq="):
            converted.append("--rsrp=" + token.split("=", 1)[1])
            threshold_was_set = True
            continue
        converted.append(token)
    if not threshold_was_set and "--rsrp" not in converted and not any(arg.startswith("--rsrp=") for arg in converted):
        converted.extend(["--rsrp", str(DEFAULT_RSRQ_THRESHOLD)])
    return converted


def _replace_path_values(payload: Any, old: str, new: str) -> Any:
    if isinstance(payload, dict):
        return {key: _replace_path_values(value, old, new) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_replace_path_values(value, old, new) for value in payload]
    if isinstance(payload, str):
        return payload.replace(old, new)
    return payload


def _move_run_dir_to_rsrq_name(run_dir: Path) -> Path:
    target = run_dir.with_name(run_dir.name.replace("tilt_rsrp_only", "tilt_rsrq_only"))
    if target == run_dir:
        return run_dir
    if target.exists():
        target = run_dir.with_name(target.name + "_rsrq")
    shutil.move(str(run_dir), str(target))
    summary_path = target / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload = _replace_path_values(payload, str(run_dir), str(target))
        payload["run_type"] = "tilt_rsrq_only_test"
        payload.setdefault("thresholds", {})["rsrq"] = payload.get("thresholds", {}).get("rsrp")
        payload.setdefault("thresholds", {})["rsrp"] = None
        payload.setdefault("thresholds", {})["kpi_mode"] = "RSRQ only"
        summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def _parse_args():
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *_convert_args_for_rsrp_parser(old_argv[1:])]
        return rsrp_test._parse_args()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    _activate_rsrq_mode()
    run_dir = rsrp_test.run_tilt_rsrp_only_recommendation_test(_parse_args())
    run_dir = _move_run_dir_to_rsrq_name(Path(run_dir))
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
