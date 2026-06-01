from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.lte_rf_debug_lab import (
    DEFAULT_GRID_RESOLUTION_M,
    DEFAULT_PROJECT_ID,
    DEFAULT_RADIUS_M,
    DEFAULT_REGION,
    DEFAULT_WORKERS,
    _write_json,
)
from tools.lte_prediction_optimised import ml_engine as opt_ml


OUTPUT_ROOT = Path("tests/output")


@dataclass
class TiltOptimizationTestConfig:
    project_id: int = DEFAULT_PROJECT_ID
    region: str = DEFAULT_REGION
    operator: Optional[str] = None
    recommendation_scenario_id: Optional[int] = None
    radius_m: float = DEFAULT_RADIUS_M
    grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M
    workers: int = DEFAULT_WORKERS
    impact_radius_m: float = 1200.0
    neighbor_site_count: int = 3
    max_interference_sites: int = 10
    output_root: Path = OUTPUT_ROOT


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_engine(region: str):
    current_engine = opt_ml.engine.get(region.lower(), opt_ml.engine["india"])
    if current_engine is None:
        raise RuntimeError(f"No DB engine configured for region={region}")
    return current_engine


def _clean_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _cell_suffix(value) -> str:
    text = _clean_id(value)
    return text.rsplit("_", 1)[-1] if text else ""


def _values_changed(current_value, recommended_value) -> bool:
    current_num = pd.to_numeric(pd.Series([current_value]), errors="coerce").iloc[0]
    recommended_num = pd.to_numeric(pd.Series([recommended_value]), errors="coerce").iloc[0]
    if pd.notna(current_num) and pd.notna(recommended_num):
        return not np.isclose(float(current_num), float(recommended_num), equal_nan=True)
    return str(current_value).strip() != str(recommended_value).strip()


def _latest_recommendation_scenario_id(project_id: int, region: str, operator: Optional[str]) -> int:
    current_engine = _resolve_engine(region)
    where_parts = ["project_id = :project_id"]
    params: Dict[str, object] = {"project_id": int(project_id)}
    if operator:
        where_parts.append("LOWER(TRIM(operator)) = :operator")
        params["operator"] = str(operator).strip().lower()
    query = text(
        f"""
        SELECT MAX(scenario_id)
        FROM rf_optimization_results
        WHERE {' AND '.join(where_parts)}
        """
    )
    with current_engine.connect() as conn:
        scenario_id = conn.execute(query, params).scalar()
    if scenario_id is None:
        op_msg = f" operator={operator}" if operator else ""
        raise FileNotFoundError(f"No tilt recommendation rows found for project_id={project_id}{op_msg}")
    return int(scenario_id)


def _fetch_recommendation_rows(config: TiltOptimizationTestConfig) -> Tuple[int, pd.DataFrame]:
    scenario_id = config.recommendation_scenario_id
    if scenario_id is None:
        scenario_id = _latest_recommendation_scenario_id(config.project_id, config.region, config.operator)

    current_engine = _resolve_engine(config.region)
    where_parts = ["project_id = :project_id", "scenario_id = :scenario_id"]
    params: Dict[str, object] = {
        "project_id": int(config.project_id),
        "scenario_id": int(scenario_id),
    }
    if config.operator:
        where_parts.append("LOWER(TRIM(operator)) = :operator")
        params["operator"] = str(config.operator).strip().lower()

    query = text(
        f"""
        SELECT
            project_id,
            scenario_id,
            operator,
            cell_id,
            technology,
            parameter,
            current_value,
            recommended_value,
            reason,
            swap_sector_detected,
            rsrp_threshold,
            rsrq_threshold,
            sinr_threshold,
            created_at
        FROM rf_optimization_results
        WHERE {' AND '.join(where_parts)}
        ORDER BY cell_id, parameter, id
        """
    )
    with current_engine.connect() as conn:
        reco_df = pd.read_sql(query, conn, params=params)
    if reco_df.empty:
        raise FileNotFoundError(
            f"No rows found in rf_optimization_results for project_id={config.project_id} "
            f"scenario_id={scenario_id}"
        )
    return int(scenario_id), reco_df


def _actionable_recommendations(reco_df: pd.DataFrame) -> pd.DataFrame:
    work = reco_df.copy()
    work["cell_id_clean"] = work["cell_id"].map(_clean_id)
    work["parameter_norm"] = work["parameter"].astype(str).str.strip().str.lower()
    changed_mask = work.apply(lambda row: _values_changed(row["current_value"], row["recommended_value"]), axis=1)
    work = work.loc[changed_mask].copy()
    supported = {
        "etilt",
        "e tilt",
        "electrical tilt",
        "azimuth",
        "tx power",
        "power",
        "mechanical tilt",
        "mtilt",
        "height",
        "antenna height",
    }
    work = work.loc[work["parameter_norm"].isin(supported)].copy()
    if work.empty:
        raise ValueError("Latest tilt recommendation scenario has no actionable supported parameter changes")
    return work


def _site_match_mask(site_df: pd.DataFrame, recommendation_cell_id: str) -> pd.Series:
    rec_id = _clean_id(recommendation_cell_id)
    rec_suffix = _cell_suffix(rec_id)
    node_cell = site_df["Node_Cell_ID"].astype(str).map(_clean_id)
    mask = node_cell == rec_id
    if not mask.any() and "cell_id" in site_df.columns:
        cell_id = site_df["cell_id"].astype(str).map(_clean_id)
        mask = cell_id == rec_id
    if not mask.any() and rec_suffix and "cell_id" in site_df.columns:
        cell_suffix = site_df["cell_id"].astype(str).map(_cell_suffix)
        mask = cell_suffix == rec_suffix
    return mask


def _apply_tilt_recommendations_to_sites(site_df: pd.DataFrame, actionable_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    modified = opt_ml._normalize_site_df(site_df, log_stage="TILT_OPT_TEST_INPUT")
    compare_cols = [
        "lat",
        "lon",
        "azimuth",
        "electrical_tilt",
        "mechanical_tilt",
        "tx_power",
        "antenna_height",
    ]
    for col in compare_cols:
        modified[f"orig_{col}"] = pd.to_numeric(modified[col], errors="coerce")

    parameter_map = {
        "etilt": "electrical_tilt",
        "e tilt": "electrical_tilt",
        "electrical tilt": "electrical_tilt",
        "azimuth": "azimuth",
        "tx power": "tx_power",
        "power": "tx_power",
        "mechanical tilt": "mechanical_tilt",
        "mtilt": "mechanical_tilt",
        "height": "antenna_height",
        "antenna height": "antenna_height",
    }

    applied_rows: List[Dict[str, object]] = []
    modified["optimization_applied"] = False
    for _, row in actionable_df.iterrows():
        target_col = parameter_map.get(str(row["parameter_norm"]))
        if not target_col:
            continue
        mask = _site_match_mask(modified, row["cell_id_clean"])
        if not mask.any():
            applied_rows.append(
                {
                    "recommendation_cell_id": row["cell_id"],
                    "parameter": row["parameter"],
                    "status": "not_matched_to_site",
                    "recommended_value": row["recommended_value"],
                }
            )
            continue

        rec_value = pd.to_numeric(pd.Series([row["recommended_value"]]), errors="coerce").iloc[0]
        if pd.isna(rec_value):
            applied_rows.append(
                {
                    "recommendation_cell_id": row["cell_id"],
                    "parameter": row["parameter"],
                    "status": "invalid_recommended_value",
                    "recommended_value": row["recommended_value"],
                }
            )
            continue

        before_values = modified.loc[mask, target_col].tolist()
        modified.loc[mask, target_col] = float(rec_value)
        modified.loc[mask, "optimization_applied"] = True
        for node_cell_id, before_value in zip(modified.loc[mask, "Node_Cell_ID"].astype(str), before_values):
            applied_rows.append(
                {
                    "recommendation_cell_id": row["cell_id"],
                    "matched_node_cell_id": node_cell_id,
                    "parameter": row["parameter"],
                    "target_column": target_col,
                    "current_value": before_value,
                    "recommended_value": float(rec_value),
                    "status": "applied",
                    "reason": row.get("reason"),
                }
            )

    applied_df = pd.DataFrame(applied_rows)
    if applied_df.empty or not (applied_df["status"].astype(str) == "applied").any():
        raise ValueError("No tilt recommendation rows could be applied to site_prediction rows")
    return modified, applied_df


def _fetch_latest_baseline_job_id(project_id: int, region: str) -> str:
    current_engine = _resolve_engine(region)
    query = text(
        """
        SELECT job_id
        FROM lte_prediction_baseline_results
        WHERE project_id = :project_id
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    with current_engine.connect() as conn:
        row = conn.execute(query, {"project_id": int(project_id)}).fetchone()
    if not row or row[0] is None:
        raise FileNotFoundError(f"No baseline results found for project_id={project_id}")
    return str(row[0])


def _compute_affected_cells(site_df: pd.DataFrame, impact_radius_m: float, neighbor_site_count: int):
    return opt_ml._compute_affected_cells(site_df, impact_radius_m, neighbor_site_count)


def _compute_k1k2_for_cells(baseline_df: pd.DataFrame, site_df: pd.DataFrame, target_cells: Sequence[str]):
    return opt_ml.compute_k1k2_for_cells(baseline_df, site_df, target_cells)


def _run_prediction(modified_site_df: pd.DataFrame, k1k2_map, config: TiltOptimizationTestConfig, baseline_job_id: str):
    return opt_ml.run_prediction_only_optimized(
        modified_site_df,
        k1k2_map,
        {
            "project_id": int(config.project_id),
            "region": config.region,
            "radius": float(config.radius_m),
            "grid_resolution": float(config.grid_resolution_m),
            "n_workers": int(config.workers),
            "impact_radius_m": float(config.impact_radius_m),
            "neighbor_site_count": int(config.neighbor_site_count),
            "max_interference_sites": int(config.max_interference_sites),
            "baseline_job_id": baseline_job_id,
        },
    )


def run_tilt_recommendation_optimization_test(config: TiltOptimizationTestConfig) -> Path:
    start = time.perf_counter()
    run_dir = _ensure_dir(config.output_root / f"project_{config.project_id}" / f"tilt_optimization_{_timestamp()}")

    print(
        f"[TILT_OPT_TEST][START] project_id={config.project_id} region={config.region} "
        f"operator={config.operator} recommendation_scenario_id={config.recommendation_scenario_id or 'latest'}"
    )

    recommendation_scenario_id, reco_df = _fetch_recommendation_rows(config)
    actionable_df = _actionable_recommendations(reco_df)
    baseline_job_id = _fetch_latest_baseline_job_id(config.project_id, config.region)
    baseline_df = opt_ml.fetch_baseline(config.project_id, region=config.region)
    site_df = opt_ml.fetch_site_data(config.project_id, region=config.region, operator=config.operator)
    modified_site_df, applied_df = _apply_tilt_recommendations_to_sites(site_df, actionable_df)

    affected_cells, affected_sites, changed_rows = _compute_affected_cells(
        modified_site_df,
        config.impact_radius_m,
        config.neighbor_site_count,
    )
    calibration_cells = sorted(changed_rows["Node_Cell_ID"].astype(str).unique().tolist())
    k1k2_map = _compute_k1k2_for_cells(baseline_df, modified_site_df, calibration_cells)
    if not k1k2_map:
        raise ValueError("No calibrated cells found after applying tilt recommendation changes")

    optimized_df = _run_prediction(modified_site_df, k1k2_map, config, baseline_job_id)
    if optimized_df.empty:
        raise RuntimeError("Tilt recommendation optimization produced no prediction rows")
    merged_df = opt_ml.replace_cells(baseline_df, optimized_df)

    reco_df.to_csv(run_dir / "rf_optimization_results_source.csv", index=False)
    actionable_df.to_csv(run_dir / "recommendations_actionable.csv", index=False)
    applied_df.to_csv(run_dir / "recommendations_applied_to_sites.csv", index=False)
    site_df.to_csv(run_dir / "site_before.csv", index=False)
    modified_site_df.to_csv(run_dir / "site_after_tilt_recommendation.csv", index=False)
    changed_rows.to_csv(run_dir / "site_changed_rows.csv", index=False)
    baseline_df.to_parquet(run_dir / "baseline_smoothed_latest.parquet", index=False)
    optimized_df.to_parquet(run_dir / "optimized_affected_predictions.parquet", index=False)
    merged_df.to_parquet(run_dir / "optimized_merged_predictions.parquet", index=False)

    summary = {
        "run_type": "tilt_recommendation_optimization_test",
        "project_id": int(config.project_id),
        "region": config.region,
        "operator": config.operator,
        "baseline_job_id": baseline_job_id,
        "recommendation_scenario_id": int(recommendation_scenario_id),
        "source_table": "rf_optimization_results",
        "counts": {
            "recommendation_rows": int(len(reco_df)),
            "actionable_recommendation_rows": int(len(actionable_df)),
            "applied_recommendation_rows": int((applied_df["status"].astype(str) == "applied").sum()),
            "baseline_rows": int(len(baseline_df)),
            "site_rows": int(len(site_df)),
            "changed_rows": int(len(changed_rows)),
            "changed_cells": int(changed_rows["Node_Cell_ID"].nunique()),
            "affected_sites": int(len(affected_sites)),
            "affected_cells": int(len(affected_cells)),
            "optimized_rows": int(len(optimized_df)),
            "merged_rows": int(len(merged_df)),
            "k1k2_cells": int(len(k1k2_map)),
        },
        "affected_sites": list(affected_sites),
        "affected_cells": list(affected_cells),
        "calibration_cells": calibration_cells,
        "parameter_change_counts": actionable_df["parameter"].astype(str).value_counts().to_dict(),
        "artifacts": {
            "rf_optimization_results_source": str(run_dir / "rf_optimization_results_source.csv"),
            "recommendations_actionable": str(run_dir / "recommendations_actionable.csv"),
            "recommendations_applied_to_sites": str(run_dir / "recommendations_applied_to_sites.csv"),
            "site_before": str(run_dir / "site_before.csv"),
            "site_after_tilt_recommendation": str(run_dir / "site_after_tilt_recommendation.csv"),
            "site_changed_rows": str(run_dir / "site_changed_rows.csv"),
            "baseline_smoothed_latest": str(run_dir / "baseline_smoothed_latest.parquet"),
            "optimized_affected_predictions": str(run_dir / "optimized_affected_predictions.parquet"),
            "optimized_merged_predictions": str(run_dir / "optimized_merged_predictions.parquet"),
        },
        "total_runtime_sec": round(time.perf_counter() - start, 4),
    }
    _write_json(run_dir / "summary.json", summary)
    print(
        f"[TILT_OPT_TEST][DONE] run_dir={run_dir} "
        f"recommendation_scenario_id={recommendation_scenario_id} "
        f"optimized_rows={len(optimized_df)} total_runtime_sec={summary['total_runtime_sec']}"
    )
    return run_dir


def _parse_args() -> TiltOptimizationTestConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--recommendation-scenario-id", type=int, default=None)
    parser.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    parser.add_argument("--grid-resolution-m", type=float, default=DEFAULT_GRID_RESOLUTION_M)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--impact-radius-m", type=float, default=1200.0)
    parser.add_argument("--neighbor-site-count", type=int, default=3)
    parser.add_argument("--max-interference-sites", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    return TiltOptimizationTestConfig(
        project_id=args.project_id,
        region=args.region,
        operator=args.operator,
        recommendation_scenario_id=args.recommendation_scenario_id,
        radius_m=args.radius_m,
        grid_resolution_m=args.grid_resolution_m,
        workers=args.workers,
        impact_radius_m=args.impact_radius_m,
        neighbor_site_count=args.neighbor_site_count,
        max_interference_sites=args.max_interference_sites,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    run_dir = run_tilt_recommendation_optimization_test(_parse_args())
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
