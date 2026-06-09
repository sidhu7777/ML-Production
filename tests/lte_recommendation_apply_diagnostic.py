from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.lte_rf_debug_lab import DEFAULT_PROJECT_ID, DEFAULT_REGION, _write_json
from tools.lte_prediction_optimised.ml_engine import fetch_site_data
from tools.lte_prediction_optimised.services import (
    _actionable_recommendations,
    _apply_recommendations_to_sites,
    _fetch_recommendation_rows,
)


OUTPUT_ROOT = Path("tests/output")


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _status_counts(df: pd.DataFrame) -> dict:
    if df.empty or "status" not in df.columns:
        return {}
    return df["status"].astype(str).value_counts(dropna=False).to_dict()


def run_diagnostic(args: argparse.Namespace) -> Path:
    run_dir = OUTPUT_ROOT / f"project_{args.project_id}" / f"recommendation_apply_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scenario_id, reco_df = _fetch_recommendation_rows(
        args.project_id,
        args.region,
        operator=args.operator,
        recommendation_scenario_id=args.recommendation_scenario_id,
    )
    actionable_df = _actionable_recommendations(reco_df)
    site_df = fetch_site_data(args.project_id, region=args.region, operator=args.operator)
    modified_site_df, applied_df = _apply_recommendations_to_sites(site_df, actionable_df)

    changed_site_df = modified_site_df.loc[modified_site_df["optimization_applied"].fillna(False)].copy()
    applied_mask = applied_df["status"].astype(str) == "applied" if "status" in applied_df.columns else pd.Series(dtype=bool)
    dropped_df = applied_df.loc[~applied_mask].copy() if not applied_df.empty else pd.DataFrame()

    reco_df.to_csv(run_dir / "scenario_recommendation_rows.csv", index=False)
    actionable_df.to_csv(run_dir / "scenario_actionable_rows.csv", index=False)
    applied_df.to_csv(run_dir / "recommendation_apply_results.csv", index=False)
    dropped_df.to_csv(run_dir / "recommendation_dropped_rows.csv", index=False)
    changed_site_df.to_csv(run_dir / "site_rows_changed_by_recommendation.csv", index=False)

    summary = {
        "project_id": int(args.project_id),
        "region": args.region,
        "operator": args.operator,
        "recommendation_scenario_id": int(scenario_id),
        "counts": {
            "recommendation_rows": int(len(reco_df)),
            "actionable_recommendation_rows": int(len(actionable_df)),
            "site_rows": int(len(site_df)),
            "apply_result_rows": int(len(applied_df)),
            "applied_rows": int(applied_mask.sum()) if len(applied_mask) else 0,
            "dropped_rows": int((~applied_mask).sum()) if len(applied_mask) else 0,
            "changed_site_rows": int(len(changed_site_df)),
            "changed_cells": int(changed_site_df["Node_Cell_ID"].nunique()) if "Node_Cell_ID" in changed_site_df.columns else 0,
        },
        "apply_status_counts": _status_counts(applied_df),
        "dropped_status_counts": _status_counts(dropped_df),
        "parameter_counts_actionable": actionable_df["parameter"].astype(str).value_counts(dropna=False).to_dict()
        if "parameter" in actionable_df.columns
        else {},
        "sample_dropped_rows": dropped_df.head(20).to_dict("records"),
        "sample_applied_rows": applied_df.loc[applied_mask].head(20).to_dict("records") if len(applied_mask) else [],
        "artifacts": {
            "scenario_recommendation_rows": str(run_dir / "scenario_recommendation_rows.csv"),
            "scenario_actionable_rows": str(run_dir / "scenario_actionable_rows.csv"),
            "recommendation_apply_results": str(run_dir / "recommendation_apply_results.csv"),
            "recommendation_dropped_rows": str(run_dir / "recommendation_dropped_rows.csv"),
            "site_rows_changed_by_recommendation": str(run_dir / "site_rows_changed_by_recommendation.csv"),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether production recommendation rows apply to site rows.")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--recommendation-scenario-id", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_diagnostic(_parse_args())
