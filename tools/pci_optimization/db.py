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


def ensure_results_table(region: str = "india") -> None:
    """CREATE TABLE IF NOT EXISTS only -- never touches any other table."""
    with get_engine(region).begin() as conn:
        conn.execute(text(CREATE_RESULT_TABLE_SQL))


def save_results(df: pd.DataFrame, region: str = "india") -> int:
    """Idempotent per (project_id, job_id): deletes any prior rows for the
    same job before inserting, so re-running/re-saving a job never
    duplicates rows -- same pattern as lte_current_capacity_recommendation_results."""
    if df.empty:
        return 0
    ensure_results_table(region)
    out = df.loc[:, RESULT_COLUMNS].copy()
    keys = out[["project_id", "job_id"]].drop_duplicates()
    with get_engine(region).begin() as conn:
        for _, row in keys.iterrows():
            conn.execute(
                text(
                    f"""
                    DELETE FROM {RESULT_TABLE_NAME}
                    WHERE project_id = :project_id AND job_id = :job_id
                    """
                ),
                {"project_id": int(row["project_id"]), "job_id": str(row["job_id"])},
            )
        out.to_sql(RESULT_TABLE_NAME, con=conn, if_exists="append", index=False, method="multi", chunksize=1000)
    return int(len(out))
