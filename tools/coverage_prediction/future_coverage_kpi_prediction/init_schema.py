from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from tools.coverage_prediction.future_coverage_kpi_prediction.schema import CREATE_TABLE_SQL, EXPECTED_COLUMNS, TABLE_NAME


ML_ROOT = Path(__file__).resolve().parents[3]


def _database_url() -> str:
    load_dotenv(ML_ROOT / ".env")
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is missing in ML/.env")
    return db_url


def create_model1_results_table() -> list[str]:
    engine = create_engine(
        _database_url(),
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"connect_timeout": 30},
    )
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        rows = conn.execute(text(f"DESCRIBE {TABLE_NAME}")).fetchall()
    return [str(row[0]) for row in rows]


def main() -> None:
    columns = create_model1_results_table()
    missing = [col for col in EXPECTED_COLUMNS if col not in columns]
    print(f"table={TABLE_NAME}")
    print(f"columns={columns}")
    if missing:
        raise RuntimeError(f"Table {TABLE_NAME} missing expected columns: {missing}")
    print("status=ok")


if __name__ == "__main__":
    main()
