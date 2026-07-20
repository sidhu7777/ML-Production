from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
sys.path.insert(0, str(ML_ROOT))
load_dotenv(ML_ROOT / ".env")

JOB_ID = "a0fa8a57-7e38-4a48-b764-9ebc82273575"


def main() -> None:
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    queries = {
        "baseline_operator_tech": """
            SELECT operator, Technology, COUNT(*) rows_count,
                   COUNT(DISTINCT cell_id) cell_id_count,
                   COUNT(DISTINCT nodeb_id_cell_id) nodeb_cell_count,
                   COUNT(DISTINCT legacy_nodeb_id_cell_id) legacy_count,
                   COUNT(DISTINCT rf_identity_key) rf_count,
                   COUNT(DISTINCT site_sector_band_key) ssb_count,
                   COUNT(DISTINCT frontend_site_sector_key) sector_count
            FROM lte_prediction_baseline_results
            WHERE project_id = 196 AND job_id = :job
            GROUP BY operator, Technology
            ORDER BY rows_count DESC
        """,
        "baseline_sample_ids": """
            SELECT operator, Technology, node_b_id, cell_id, nodeb_id_cell_id,
                   legacy_nodeb_id_cell_id, rf_identity_key, sector,
                   frontend_site_sector_key, site_sector_band_key, band, serving_earfcn
            FROM lte_prediction_baseline_results
            WHERE project_id = 196 AND job_id = :job
            LIMIT 20
        """,
        "site_operator_tech": """
            SELECT cluster, Technology, COUNT(*) rows_count,
                   COUNT(DISTINCT cell_id) cell_count,
                   COUNT(DISTINCT CONCAT(COALESCE(site,''), '_', COALESCE(cell_id,''))) site_cell_count,
                   COUNT(DISTINCT CONCAT(COALESCE(site,''), '_', COALESCE(sector,''), '_', COALESCE(band,''))) ssb_count
            FROM site_prediction
            WHERE tbl_project_id = 196
            GROUP BY cluster, Technology
            ORDER BY rows_count DESC
        """,
        "site_sample": """
            SELECT site, cell_id, sector, band, earfcn, cluster, Technology, latitude, longitude
            FROM site_prediction
            WHERE tbl_project_id = 196
            LIMIT 30
        """,
    }
    with engine.connect() as conn:
        for name, query in queries.items():
            print(f"\n{name}")
            print(pd.read_sql(text(query), conn, params={"job": JOB_ID}).to_string(index=False))


if __name__ == "__main__":
    main()
