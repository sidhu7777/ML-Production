from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from ML.tools.lte_prediction_optimised import ml_engine as opt_ml
from ML.tools.lte_prediction_optimised.services import (
    LTEPredictionService_optimised,
    _actionable_recommendations,
    _fetch_recommendation_rows,
    _resolve_engine,
    _apply_recommendations_to_sites,
    _save_recommendation_site_prediction_scenario,
)


OUTPUT_ROOT = Path("ML/tests/output")
DEFAULT_PROJECT_ID = 196
DEFAULT_REGION = "india"


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_debug_rows(region: str, project_id: int, scenario_row_id: int, public_scenario_id: int) -> None:
    engine = _resolve_engine(region)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM site_prediction_optimized
                WHERE tbl_project_id = :project_id
                  AND scenario = :scenario
                """
            ),
            {"project_id": int(project_id), "scenario": int(public_scenario_id)},
        )
        conn.execute(
            text(
                """
                DELETE FROM lte_optimization_scenarios
                WHERE id = :scenario_row_id
                """
            ),
            {"scenario_row_id": int(scenario_row_id)},
        )


def run_verification(args: argparse.Namespace) -> Path:
    run_dir = _ensure_dir(
        OUTPUT_ROOT
        / f"project_{int(args.project_id)}"
        / f"verify_recommendation_site_save_{_timestamp()}"
    )

    recommendation_scenario_id, reco_df = _fetch_recommendation_rows(
        int(args.project_id),
        args.region,
        operator=args.operator,
        recommendation_scenario_id=args.recommendation_scenario_id,
    )
    actionable_df = _actionable_recommendations(reco_df)
    site_df = opt_ml.fetch_site_data(
        int(args.project_id),
        region=args.region,
        operator=args.operator,
    )
    modified_site_df, applied_df = _apply_recommendations_to_sites(site_df, actionable_df)

    service = LTEPredictionService_optimised()
    debug_tag = f"recommendation_site_save_debug_{uuid.uuid4().hex[:10]}"
    cfg = {
        "project_id": int(args.project_id),
        "region": args.region,
        "operator": args.operator or "Airtel",
        "recommendation_scenario_id": int(recommendation_scenario_id),
        "target_type": "recommendation_debug",
        "target_id": debug_tag,
        "scenario_name": f"Debug Recommendation Site Save {recommendation_scenario_id}",
        "scenario_description": f"Debug verification for recommendation scenario {recommendation_scenario_id}",
        "created_by": "debug_test",
    }
    scenario_row_id, public_scenario_id = service._create_scenario(
        cfg,
        job_id=f"debug-{uuid.uuid4()}",
        region=args.region,
    )

    cleanup_done = False
    try:
        saved_rows = _save_recommendation_site_prediction_scenario(
            int(args.project_id),
            int(public_scenario_id),
            modified_site_df,
            applied_df,
            args.region,
        )

        engine = _resolve_engine(args.region)
        with engine.connect() as conn:
            saved_df = pd.read_sql(
                text(
                    """
                    SELECT id, site_prediction_id, scenario, site, sector, cell_id,
                           latitude, longitude, azimuth, e_tilt, m_tilt, height,
                           tx_power, cluster, updated_at
                    FROM site_prediction_optimized
                    WHERE tbl_project_id = :project_id
                      AND scenario = :scenario
                    ORDER BY id
                    """
                ),
                conn,
                params={
                    "project_id": int(args.project_id),
                    "scenario": int(public_scenario_id),
                },
            )

        summary = {
            "project_id": int(args.project_id),
            "region": args.region,
            "operator": args.operator,
            "recommendation_scenario_id": int(recommendation_scenario_id),
            "debug_target_id": debug_tag,
            "scenario_row_id": int(scenario_row_id),
            "public_scenario_id": int(public_scenario_id),
            "actionable_recommendation_rows": int(len(actionable_df)),
            "applied_recommendation_rows": int((applied_df["status"].astype(str) == "applied").sum()),
            "saved_site_prediction_rows_reported": int(saved_rows),
            "saved_site_prediction_rows_found": int(len(saved_df)),
            "saved_distinct_source_rows": int(saved_df["site_prediction_id"].nunique()) if not saved_df.empty else 0,
            "status": "pass" if len(saved_df) > 0 else "fail",
        }

        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
        saved_df.to_csv(run_dir / "site_prediction_optimized_rows.csv", index=False)
        print(json.dumps(summary, indent=2))
        return run_dir
    finally:
        if args.cleanup:
            _cleanup_debug_rows(args.region, int(args.project_id), int(scenario_row_id), int(public_scenario_id))
            cleanup_done = True
        (run_dir / "cleanup.txt").write_text(
            "cleanup_done=" + ("true" if cleanup_done else "false"),
            encoding="ascii",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--recommendation-scenario-id", type=int, default=None)
    parser.add_argument("--cleanup", action="store_true", default=False)
    parser.add_argument("--no-cleanup", action="store_false", dest="cleanup")
    return parser.parse_args()


if __name__ == "__main__":
    run_dir = run_verification(_parse_args())
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
