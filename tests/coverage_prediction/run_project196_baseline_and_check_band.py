from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
sys.path.insert(0, str(ML_ROOT))
load_dotenv(ML_ROOT / ".env")
for bridge_env in ["PYTHON_BRIDGE_BASE_URL", "SIGNAL_TRACKERS_BRIDGE_URL"]:
    os.environ[bridge_env] = ""

from app import create_app  # noqa: E402
from tools.lte_prediction.services import JOBS, LTEPredictionService  # noqa: E402


def main() -> None:
    project_id = 196
    session_ids = [4187, 4178, 4180]
    job_id = uuid.uuid4().hex
    cfg = {
        "project_id": project_id,
        "session_ids": session_ids,
        "region": "india",
        "operator": "Airtel",
        "radius_m": 500.0,
        "grid_resolution": 25.0,
        "building": True,
        "n_workers": 2,
        "max_interference_sites": 10,
        "use_frontend_grid_sampling": True,
        "samples_per_grid_axis": 3,
        "max_cells_per_grid": 3,
        "min_cells_per_grid": 1,
        "ensure_all_cells": True,
        "min_grids_per_cell": 1,
        "output_folder": str(ML_ROOT / "outputs"),
    }
    JOBS[job_id] = {"status": "running", "progress": "Starting", "error": None}
    app = create_app()
    with app.app_context():
        LTEPredictionService()._run(job_id, cfg)
    print("\nJOB_STATUS")
    print(json.dumps(JOBS.get(job_id, {}), indent=2, default=str))
    if JOBS.get(job_id, {}).get("status") != "done":
        raise SystemExit(2)

    engine = create_engine(os.environ.get("DATABASE_URL") or os.environ.get("DB_URL"))
    summary = pd.read_sql(
        text(
            """
            SELECT
              job_id,
              COUNT(*) AS rows_count,
              COUNT(DISTINCT nodeb_id_cell_id) AS distinct_nodeb_id_cell_id,
              SUM(CASE WHEN band IS NULL OR TRIM(CAST(band AS CHAR)) = '' THEN 1 ELSE 0 END) AS null_or_blank_band_rows,
              COUNT(DISTINCT NULLIF(TRIM(CAST(band AS CHAR)), '')) AS distinct_band_count
            FROM lte_prediction_baseline_results
            WHERE project_id = :project_id AND job_id = :job_id
            GROUP BY job_id
            """
        ),
        engine,
        params={"project_id": project_id, "job_id": job_id},
    )
    dist = pd.read_sql(
        text(
            """
            SELECT COALESCE(NULLIF(TRIM(CAST(band AS CHAR)), ''), '<blank/null>') AS band_value,
                   COUNT(*) AS rows_count,
                   COUNT(DISTINCT nodeb_id_cell_id) AS distinct_nodeb_id_cell_id
            FROM lte_prediction_baseline_results
            WHERE project_id = :project_id AND job_id = :job_id
            GROUP BY band_value
            ORDER BY rows_count DESC
            """
        ),
        engine,
        params={"project_id": project_id, "job_id": job_id},
    )
    print("\nSAVED_BASELINE_BAND_SUMMARY")
    print(summary.to_string(index=False))
    print("\nSAVED_BASELINE_BAND_DISTRIBUTION")
    print(dist.to_string(index=False))


if __name__ == "__main__":
    main()
