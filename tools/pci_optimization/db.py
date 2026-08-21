from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from .schema import CREATE_RESULT_TABLE_SQL, RESULT_COLUMNS, RESULT_TABLE_NAME


ML_ROOT = Path(__file__).resolve().parents[2]

_ENGINES: dict[str, Any] = {}


def _load_env() -> None:
    load_dotenv(ML_ROOT / ".env")


def get_engine(region: str = "india"):
    """Direct DB engine, same convention as every other tools/* db.py:
    DATABASE_URL_Taiwan for region=taiwan, DATABASE_URL otherwise -- no
    python bridge involved. Saving results is always a direct DB write in
    this codebase (the bridge is only ever used, optionally, for reads)."""
    _load_env()
    key = str(region or "india").lower()
    if key in _ENGINES:
        return _ENGINES[key]
    db_url = os.getenv("DATABASE_URL_Taiwan") if key == "taiwan" else os.getenv("DATABASE_URL")
    db_url = db_url or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is missing in ML/.env")
    _ENGINES[key] = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
        connect_args={"connect_timeout": 30},
    )
    return _ENGINES[key]


def _ensure_column(conn, column: str, ddl: str) -> None:
    """Additive ALTER only -- adds `column` if it's missing, never drops or
    changes anything else. Checked via information_schema first (portable
    across MySQL versions, unlike `ADD COLUMN IF NOT EXISTS`)."""
    exists = conn.execute(
        text(
            """
            SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table_name AND COLUMN_NAME = :column_name
            """
        ),
        {"table_name": RESULT_TABLE_NAME, "column_name": column},
    ).scalar()
    if not exists:
        conn.execute(text(f"ALTER TABLE {RESULT_TABLE_NAME} ADD COLUMN {ddl}"))


def ensure_results_table(region: str = "india") -> None:
    """CREATE TABLE IF NOT EXISTS, plus additive ALTERs for columns added
    after the table already existed in production (e.g. resolved/reason) --
    never drops or touches any other table."""
    with get_engine(region).begin() as conn:
        conn.execute(text(CREATE_RESULT_TABLE_SQL))
        _ensure_column(conn, "resolved", "resolved TINYINT(1) NULL AFTER changed_flag")
        _ensure_column(conn, "reason", "reason TEXT NULL AFTER resolved")


def save_results(df: pd.DataFrame, region: str = "india") -> int:
    """Replace, not upsert: deletes EVERY prior row for a project (any
    older job_id included) before inserting the new run's rows, so a
    project always reflects only its latest PCI optimization run -- no
    accumulating job history, no ON DUPLICATE KEY UPDATE. Plain DELETE
    then INSERT, same as every other tools/* results table in this repo."""
    if df.empty:
        return 0
    ensure_results_table(region)
    out = df.loc[:, RESULT_COLUMNS].copy()
    project_ids = out["project_id"].drop_duplicates().tolist()
    with get_engine(region).begin() as conn:
        for project_id in project_ids:
            conn.execute(
                text(f"DELETE FROM {RESULT_TABLE_NAME} WHERE project_id = :project_id"),
                {"project_id": int(project_id)},
            )
        out.to_sql(RESULT_TABLE_NAME, con=conn, if_exists="append", index=False, method="multi", chunksize=1000)
    return int(len(out))
