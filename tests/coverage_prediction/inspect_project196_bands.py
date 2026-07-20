from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main() -> None:
    engine = create_engine(os.environ.get("DATABASE_URL") or os.environ.get("DB_URL"))

    cols = pd.read_sql(
        text(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'site_prediction' "
            "ORDER BY ORDINAL_POSITION"
        ),
        engine,
    )["COLUMN_NAME"].tolist()
    print("SITE_COLUMNS", cols)

    band_col = "band" if "band" in cols else ("frequency" if "frequency" in cols else None)
    operator_col = next((c for c in ["operator", "operator_name", "network_operator", "mno"] if c in cols), None)
    tech_col = next((c for c in ["technology", "tech", "network_type"] if c in cols), None)

    select_parts = []
    group_parts = []
    if operator_col:
        select_parts.append(f"COALESCE(CAST({operator_col} AS CHAR), '<null>') AS operator_value")
        group_parts.append("operator_value")
    if tech_col:
        select_parts.append(f"COALESCE(CAST({tech_col} AS CHAR), '<null>') AS tech_value")
        group_parts.append("tech_value")
    if band_col:
        select_parts.append(f"COALESCE(CAST({band_col} AS CHAR), '<null>') AS band_value")
        group_parts.append("band_value")
    else:
        select_parts.append("'<no-band-column>' AS band_value")
        group_parts.append("band_value")

    project_col = "project_id" if "project_id" in cols else "tbl_project_id"
    query = (
        "SELECT "
        + ", ".join(select_parts)
        + ", COUNT(*) AS rows_count, COUNT(DISTINCT cell_id) AS distinct_cell_id "
        + f"FROM site_prediction WHERE {project_col} = :pid GROUP BY "
        + ", ".join(group_parts)
        + " ORDER BY rows_count DESC LIMIT 100"
    )
    site_dist = pd.read_sql(text(query), engine, params={"pid": 196})
    print("\nSITE_BAND_DISTRIBUTION")
    print(site_dist.to_string(index=False))

    baseline_cols = pd.read_sql(
        text(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'lte_prediction_baseline_results' "
            "ORDER BY ORDINAL_POSITION"
        ),
        engine,
    )["COLUMN_NAME"].tolist()
    baseline_band_col = "band" if "band" in baseline_cols else None
    band_null_sql = (
        f"SUM(CASE WHEN {baseline_band_col} IS NULL THEN 1 ELSE 0 END) AS band_null_rows"
        if baseline_band_col
        else "NULL AS band_null_rows"
    )
    baseline = pd.read_sql(
        text(
            "SELECT COUNT(*) AS rows_count, "
            "COUNT(DISTINCT nodeb_id_cell_id) AS distinct_nodeb_id_cell_id, "
            "COUNT(DISTINCT cell_id) AS distinct_cell_id, "
            f"{band_null_sql} "
            "FROM lte_prediction_baseline_results "
            "WHERE project_id = :pid AND job_id = :job_id"
        ),
        engine,
        params={
            "pid": 196,
            "job_id": "a0fa8a57-7e38-4a48-b764-9ebc82273575",
        },
    )
    print("\nBASELINE_COUNTS")
    print(baseline.to_string(index=False))


if __name__ == "__main__":
    main()
