from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.lte_rf_debug_lab import DEFAULT_PROJECT_ID, DEFAULT_REGION, _write_json
from tools.lte_prediction_optimised.services import _resolve_engine


OUTPUT_ROOT = Path("tests/output")


def _metric_delta_summary(merged: pd.DataFrame, metric: str) -> dict:
    before = pd.to_numeric(merged[f"{metric}_base"], errors="coerce")
    after = pd.to_numeric(merged[f"{metric}_opt"], errors="coerce")
    delta = after - before
    changed = delta.abs() > 1e-6
    return {
        "changed_count": int(changed.sum()),
        "mean_delta": float(delta.mean()) if len(delta) else 0.0,
        "min_delta": float(delta.min()) if len(delta) else 0.0,
        "max_delta": float(delta.max()) if len(delta) else 0.0,
        "mean_abs_delta_changed": float(delta.loc[changed].abs().mean()) if bool(changed.any()) else 0.0,
    }


def run_diagnostic(args: argparse.Namespace) -> Path:
    engine = _resolve_engine(args.region)
    run_dir = OUTPUT_ROOT / f"project_{args.project_id}" / f"recommendation_optimized_db_scenario_{args.recommendation_scenario_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scenario_df = pd.read_sql(
        text(
            """
            SELECT id, scenario_id, project_id, baseline_job_id, scenario_name,
                   target_type, target_id, operator, status, created_at, updated_at
            FROM lte_optimization_scenarios
            WHERE project_id = :project_id
              AND target_type = 'recommendation'
              AND target_id = :target_id
            ORDER BY id DESC
            LIMIT 1
            """
        ),
        engine,
        params={
            "project_id": int(args.project_id),
            "target_id": f"rf_scenario_{int(args.recommendation_scenario_id)}",
        },
    )
    if scenario_df.empty:
        raise FileNotFoundError(
            f"No lte_optimization_scenarios row found for rf_scenario_{int(args.recommendation_scenario_id)}"
        )

    scenario = scenario_df.iloc[0]
    scenario_row_id = int(scenario["id"])
    public_scenario_id = int(scenario["scenario_id"])
    baseline_job_id = str(scenario["baseline_job_id"])
    operator = str(scenario.get("operator") or args.operator or "")

    count_df = pd.read_sql(
        text(
            """
            SELECT scenario_id, COUNT(*) AS result_count, COUNT(DISTINCT nodeb_id_cell_id) AS cell_count
            FROM lte_prediction_optimised_results
            WHERE project_id = :project_id
              AND scenario_id IN (:scenario_row_id, :public_scenario_id, :recommendation_scenario_id)
            GROUP BY scenario_id
            ORDER BY scenario_id
            """
        ),
        engine,
        params={
            "project_id": int(args.project_id),
            "scenario_row_id": scenario_row_id,
            "public_scenario_id": public_scenario_id,
            "recommendation_scenario_id": int(args.recommendation_scenario_id),
        },
    )

    opt_df = pd.read_sql(
        text(
            """
            SELECT lat, lon, nodeb_id_cell_id, pred_rsrp, pred_rsrq, pred_sinr
            FROM lte_prediction_optimised_results
            WHERE project_id = :project_id
              AND scenario_id = :scenario_row_id
            """
        ),
        engine,
        params={"project_id": int(args.project_id), "scenario_row_id": scenario_row_id},
    )
    if opt_df.empty:
        raise FileNotFoundError(
            f"No optimized result rows found under scenario row id {scenario_row_id}. "
            "The recommendation-optimized job may not have been run."
        )

    base_where = "project_id = :project_id AND job_id = :baseline_job_id"
    params = {"project_id": int(args.project_id), "baseline_job_id": baseline_job_id}
    if operator:
        base_where += " AND LOWER(TRIM(operator)) = :operator"
        params["operator"] = operator.strip().lower()
    base_df = pd.read_sql(
        text(
            f"""
            SELECT lat, lon, nodeb_id_cell_id, pred_rsrp, pred_rsrq, pred_sinr
            FROM lte_prediction_baseline_results
            WHERE {base_where}
            """
        ),
        engine,
        params=params,
    )

    for frame in [base_df, opt_df]:
        frame["_lat_key"] = pd.to_numeric(frame["lat"], errors="coerce").round(6)
        frame["_lon_key"] = pd.to_numeric(frame["lon"], errors="coerce").round(6)
    key_cols = ["nodeb_id_cell_id", "_lat_key", "_lon_key"]
    merged = (
        base_df.drop_duplicates(key_cols)
        .merge(opt_df.drop_duplicates(key_cols), on=key_cols, suffixes=("_base", "_opt"))
    )
    if not merged.empty:
        rsrp_delta = pd.to_numeric(merged["pred_rsrp_opt"], errors="coerce") - pd.to_numeric(
            merged["pred_rsrp_base"], errors="coerce"
        )
        sample_changed = merged.loc[
            rsrp_delta.abs() > 1e-6,
            [
                "nodeb_id_cell_id",
                "lat_base",
                "lon_base",
                "pred_rsrp_base",
                "pred_rsrp_opt",
                "pred_rsrq_base",
                "pred_rsrq_opt",
                "pred_sinr_base",
                "pred_sinr_opt",
            ],
        ].head(25)
    else:
        sample_changed = pd.DataFrame()

    scenario_df.to_csv(run_dir / "optimization_scenario_row.csv", index=False)
    count_df.to_csv(run_dir / "optimized_result_counts_by_scenario_id.csv", index=False)
    sample_changed.to_csv(run_dir / "sample_changed_points.csv", index=False)

    summary = {
        "project_id": int(args.project_id),
        "region": args.region,
        "operator": operator,
        "recommendation_scenario_id": int(args.recommendation_scenario_id),
        "optimization_scenario_row_id": scenario_row_id,
        "optimization_public_scenario_id": public_scenario_id,
        "baseline_job_id": baseline_job_id,
        "status": str(scenario["status"]),
        "important": (
            "lte_prediction_optimised_results.scenario_id references lte_optimization_scenarios.id, "
            "so fetch optimized rows with optimization_scenario_row_id, not the public scenario_id."
        ),
        "counts_by_scenario_id": count_df.to_dict("records"),
        "baseline_rows": int(len(base_df)),
        "optimized_rows": int(len(opt_df)),
        "matched_rows": int(len(merged)),
        "metric_deltas": {
            "pred_rsrp": _metric_delta_summary(merged, "pred_rsrp") if not merged.empty else {},
            "pred_rsrq": _metric_delta_summary(merged, "pred_rsrq") if not merged.empty else {},
            "pred_sinr": _metric_delta_summary(merged, "pred_sinr") if not merged.empty else {},
        },
        "artifacts": {
            "optimization_scenario_row": str(run_dir / "optimization_scenario_row.csv"),
            "optimized_result_counts_by_scenario_id": str(run_dir / "optimized_result_counts_by_scenario_id.csv"),
            "sample_changed_points": str(run_dir / "sample_changed_points.csv"),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check saved recommendation-optimized DB results and KPI deltas.")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--recommendation-scenario-id", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_diagnostic(_parse_args())
