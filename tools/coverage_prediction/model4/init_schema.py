from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from tools.coverage_prediction.model4.schema import (
    CREATE_RECOMMENDATION_TABLE_SQL,
    CREATE_RF_SURFACE_TABLE_SQL,
    RECOMMENDATION_EXPECTED_COLUMNS,
    RECOMMENDATION_TABLE_NAME,
    RF_SURFACE_EXPECTED_COLUMNS,
    RF_SURFACE_TABLE_NAME,
)


ML_ROOT = Path(__file__).resolve().parents[3]


def _database_url(region: str = "india") -> str:
    load_dotenv(ML_ROOT / ".env")
    key = str(region or "india").lower()
    if key == "taiwan":
        db_url = os.getenv("DATABASE_URL_Taiwan") or os.getenv("DATABASE_URL")
    else:
        db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is missing in ML/.env")
    return db_url


def _describe(conn, table_name: str) -> list[str]:
    rows = conn.execute(text(f"DESCRIBE {table_name}")).fetchall()
    return [str(row[0]) for row in rows]


def create_model4_results_tables(region: str = "india") -> dict[str, list[str]]:
    engine = create_engine(
        _database_url(region),
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"connect_timeout": 30},
    )
    with engine.begin() as conn:
        conn.execute(text(CREATE_RECOMMENDATION_TABLE_SQL))
        conn.execute(text(CREATE_RF_SURFACE_TABLE_SQL))
        return {
            RECOMMENDATION_TABLE_NAME: _describe(conn, RECOMMENDATION_TABLE_NAME),
            RF_SURFACE_TABLE_NAME: _describe(conn, RF_SURFACE_TABLE_NAME),
        }


def main() -> None:
    region = os.getenv("MODEL4_SCHEMA_REGION", "india")
    tables = create_model4_results_tables(region)
    expected = {
        RECOMMENDATION_TABLE_NAME: RECOMMENDATION_EXPECTED_COLUMNS,
        RF_SURFACE_TABLE_NAME: RF_SURFACE_EXPECTED_COLUMNS,
    }
    for table_name, columns in tables.items():
        missing = [col for col in expected[table_name] if col not in columns]
        print(f"table={table_name}")
        print(f"columns={columns}")
        if missing:
            raise RuntimeError(f"Table {table_name} missing expected columns: {missing}")
    print("status=ok")


if __name__ == "__main__":
    main()
