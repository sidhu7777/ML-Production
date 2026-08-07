from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from utils.python_bridge import PythonBridgeError, get_bridge_client

from .schema import CREATE_TABLE_SQL, RESULT_COLUMNS, TABLE_NAME


ML_ROOT = Path(__file__).resolve().parents[3]

_ENGINES: dict[str, Any] = {}


def _load_env() -> None:
    load_dotenv(ML_ROOT / ".env")


def get_engine(region: str = "india"):
    _load_env()
    key = str(region or "india").lower()
    if key in _ENGINES:
        return _ENGINES[key]
    if key == "taiwan":
        db_url = os.getenv("DATABASE_URL_Taiwan") or os.getenv("DATABASE_URL")
    else:
        db_url = os.getenv("DATABASE_URL")
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
    with get_engine(region).begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))


def latest_baseline_job_id(project_id: int, region: str = "india", operator: str | None = None) -> str:
    try:
        bridge = get_bridge_client()
    except PythonBridgeError:
        bridge = None
    if bridge:
        try:
            params = {"projectId": int(project_id), "region": str(region or "india").lower()}
            if operator and str(operator).strip().lower() != "all":
                params["operator"] = str(operator).strip()
            payload = bridge._request("GET", "GetLatestLteBaselineJobId", params=params)
            job_id = payload.get("JobId") or payload.get("jobId")
            if job_id:
                return str(job_id)
        except PythonBridgeError:
            pass

    filters = ["project_id = :project_id"]
    params: dict[str, Any] = {"project_id": int(project_id)}
    if operator and str(operator).strip().lower() != "all":
        filters.append("LOWER(TRIM(operator)) = :operator")
        params["operator"] = str(operator).strip().lower()
    query = text(
        f"""
        SELECT job_id
        FROM lte_prediction_baseline_results
        WHERE {" AND ".join(filters)}
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    with get_engine(region).connect() as conn:
        row = conn.execute(query, params).fetchone()
    if not row or row[0] is None:
        raise FileNotFoundError(f"No baseline job found for project_id={project_id}")
    return str(row[0])


def fetch_baseline_rows(
    project_id: int,
    region: str = "india",
    operator: str | None = None,
    baseline_job_id: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    source = "python_bridge_baseline"
    try:
        bridge = get_bridge_client()
    except PythonBridgeError:
        bridge = None

    if bridge:
        try:
            job_id = baseline_job_id or latest_baseline_job_id(project_id, region, operator)
            params = {"projectId": int(project_id), "region": str(region or "india").lower(), "jobId": str(job_id)}
            if operator and str(operator).strip().lower() != "all":
                params["operator"] = str(operator).strip()
            page_limit = int(os.getenv("PYTHON_BRIDGE_BASELINE_PAGE_SIZE", "50000"))
            df = bridge.get_rows("GetLteBaselineRows", params, limit=page_limit, progress_label="model1_baseline")
            if "job_id" in df.columns:
                df = df.loc[df["job_id"].astype(str) == str(job_id)].copy()
            if operator and str(operator).strip().lower() != "all" and "operator" in df.columns:
                op = str(operator).strip().lower()
                df = df.loc[df["operator"].astype(str).str.strip().str.lower() == op].copy()
            if not df.empty:
                return df, str(job_id), source
        except PythonBridgeError:
            pass

    source = "direct_db_baseline_fallback"
    job_id = baseline_job_id or latest_baseline_job_id(project_id, region, operator)
    filters = ["project_id = :project_id", "job_id = :job_id"]
    params = {"project_id": int(project_id), "job_id": str(job_id)}
    if operator and str(operator).strip().lower() != "all":
        filters.append("LOWER(TRIM(operator)) = :operator")
        params["operator"] = str(operator).strip().lower()
    query = text(f"SELECT * FROM lte_prediction_baseline_results WHERE {' AND '.join(filters)}")
    with get_engine(region).connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    if df.empty:
        raise FileNotFoundError(f"No baseline rows found for project_id={project_id} job_id={job_id}")
    return df, str(job_id), source


def fetch_geo_rows(
    project_id: int,
    region: str = "india",
    baseline_job_id: str | None = None,
) -> pd.DataFrame:
    try:
        bridge = get_bridge_client()
    except PythonBridgeError:
        bridge = None
    if bridge:
        try:
            params = {"projectId": int(project_id), "region": str(region or "india").lower()}
            df = bridge.get_rows("GetLtePredictionGeoFeatures", params, limit=50000, progress_label="model1_geo")
            if baseline_job_id and "baseline_job_id" in df.columns:
                exact = df.loc[df["baseline_job_id"].astype(str) == str(baseline_job_id)].copy()
                if not exact.empty:
                    df = exact
            return df
        except PythonBridgeError:
            pass

    filters = ["project_id = :project_id"]
    params: dict[str, Any] = {"project_id": int(project_id)}
    if baseline_job_id:
        filters.append("baseline_job_id = :baseline_job_id")
        params["baseline_job_id"] = str(baseline_job_id)
    query = text(f"SELECT * FROM lte_prediction_geo_features WHERE {' AND '.join(filters)}")
    with get_engine(region).connect() as conn:
        return pd.read_sql(query, conn, params=params)


def save_results(df: pd.DataFrame, region: str = "india") -> int:
    if df.empty:
        return 0
    ensure_results_table(region)
    out = df.loc[:, RESULT_COLUMNS].copy()
    keys = out[["project_id", "baseline_job_id", "model_run_id"]].drop_duplicates()
    with get_engine(region).begin() as conn:
        for _, row in keys.iterrows():
            conn.execute(
                text(
                    f"""
                    DELETE FROM {TABLE_NAME}
                    WHERE project_id = :project_id
                      AND baseline_job_id = :baseline_job_id
                      AND model_run_id = :model_run_id
                    """
                ),
                {
                    "project_id": int(row["project_id"]),
                    "baseline_job_id": str(row["baseline_job_id"]),
                    "model_run_id": str(row["model_run_id"]),
                },
            )
        out.to_sql(TABLE_NAME, con=conn, if_exists="append", index=False, method="multi", chunksize=5000)
    return int(len(out))
