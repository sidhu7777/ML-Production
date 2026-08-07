from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from tools.pci_optimization.schema import (
    CREATE_RESULT_TABLE_SQL,
    RESULT_EXPECTED_COLUMNS,
    RESULT_TABLE_NAME,
)


ML_ROOT = Path(__file__).resolve().parents[2]


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


def create_pci_optimization_results_table(region: str = "india") -> list[str]:
    """CREATE TABLE IF NOT EXISTS only -- additive, never drops or alters
    any other table in the database."""
    engine = create_engine(
        _database_url(region),
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"connect_timeout": 30},
    )
    with engine.begin() as conn:
        conn.execute(text(CREATE_RESULT_TABLE_SQL))
        return _describe(conn, RESULT_TABLE_NAME)


def main() -> None:
    region = os.getenv("PCI_OPTIMIZATION_SCHEMA_REGION", "india")
    columns = create_pci_optimization_results_table(region)
    print(f"table={RESULT_TABLE_NAME}")
    print(f"columns={columns}")
    missing = [col for col in RESULT_EXPECTED_COLUMNS if col not in columns]
    if missing:
        raise RuntimeError(f"Table {RESULT_TABLE_NAME} missing expected columns: {missing}")
    print("status=ok")


if __name__ == "__main__":
    main()
