from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _print_df(name: str, df: pd.DataFrame) -> None:
    print(f"--- {name} ---")
    if df.empty:
        print("EMPTY")
    else:
        print(df.to_string(index=False))


def main() -> int:
    _load_env()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    engine = create_engine(database_url, pool_pre_ping=True)
    queries = [
        (
            "rf_parameter_counts_project196_scenario19",
            """
            SELECT parameter, COUNT(*) AS cnt
            FROM rf_optimization_results
            WHERE project_id = 196 AND scenario_id = 19
            GROUP BY parameter
            ORDER BY parameter
            """,
        ),
        (
            "rf_rows_project196_scenario19",
            """
            SELECT cell_id, parameter, current_value, recommended_value, created_at
            FROM rf_optimization_results
            WHERE project_id = 196 AND scenario_id = 19
            ORDER BY id
            """,
        ),
        (
            "latest_optimized_scenarios_project196",
            """
            SELECT id, scenario_id, status, created_at, updated_at
            FROM lte_optimization_scenarios
            WHERE project_id = 196
            ORDER BY id DESC
            LIMIT 5
            """,
        ),
        (
            "site_prediction_optimized_project196_scenario1",
            """
            SELECT
                scenario,
                COUNT(*) AS cnt,
                SUM(CASE WHEN is_updated = 1 THEN 1 ELSE 0 END) AS updated_cnt,
                MIN(created_at) AS min_created,
                MAX(updated_at) AS max_updated
            FROM site_prediction_optimized
            WHERE tbl_project_id = 196 AND scenario = 1
            GROUP BY scenario
            """,
        ),
        (
            "site_prediction_optimized_project196_scenario1_recent",
            """
            SELECT id, site_prediction_id, scenario, azimuth, e_tilt, updated_at
            FROM site_prediction_optimized
            WHERE tbl_project_id = 196 AND scenario = 1
            ORDER BY updated_at DESC, id DESC
            LIMIT 20
            """,
        ),
        (
            "lte_prediction_optimised_results_project196_row69_public1",
            """
            SELECT
                scenario_id,
                public_scenario_id,
                COUNT(*) AS cnt,
                MIN(created_at) AS min_created,
                MAX(created_at) AS max_created
            FROM lte_prediction_optimised_results
            WHERE project_id = 196
              AND scenario_id = 69
              AND public_scenario_id = 1
            GROUP BY scenario_id, public_scenario_id
            """,
        ),
    ]

    with engine.connect() as conn:
        for name, query in queries:
            _print_df(name, pd.read_sql(text(query), conn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
