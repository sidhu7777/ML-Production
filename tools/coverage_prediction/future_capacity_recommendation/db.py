from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from utils.python_bridge import PythonBridgeError, get_bridge_client

from .schema import (
    CREATE_RECOMMENDATION_TABLE_SQL,
    CREATE_RF_SURFACE_TABLE_SQL,
    RECOMMENDATION_RESULT_COLUMNS,
    RECOMMENDATION_TABLE_NAME,
    RF_SURFACE_RESULT_COLUMNS,
    RF_SURFACE_TABLE_NAME,
)
from tools.coverage_prediction.future_demand_capacity_forecast.schema import TABLE_NAME as FUTURE_DEMAND_CAPACITY_TABLE_NAME


ML_ROOT = Path(__file__).resolve().parents[3]

_ENGINES: dict[str, Any] = {}


def _load_env() -> None:
    load_dotenv(ML_ROOT / ".env")


def get_engine(region: str = "india"):
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


def ensure_results_tables(region: str = "india") -> None:
    with get_engine(region).begin() as conn:
        conn.execute(text(CREATE_RECOMMENDATION_TABLE_SQL))
        conn.execute(text(CREATE_RF_SURFACE_TABLE_SQL))


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


def fetch_model2_results(model2_run_id: str, region: str = "india") -> pd.DataFrame:
    query = text(
        f"""
        SELECT *
        FROM {FUTURE_DEMAND_CAPACITY_TABLE_NAME}
        WHERE model_run_id = :model_run_id
        """
    )
    with get_engine(region).connect() as conn:
        df = pd.read_sql(query, conn, params={"model_run_id": str(model2_run_id)})
    if df.empty:
        raise FileNotFoundError(f"No future demand/capacity rows found for model_run_id={model2_run_id}")
    return df


def _safe_str(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    text_value = str(value).strip()
    if not text_value or text_value.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return text_value


def _safe_num(value: Any) -> float | None:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value) or np.isinf(float(value)):
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    num = _safe_num(value)
    if num is None:
        return None
    return int(num)


def _safe_bool(value: Any) -> int | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, str):
        text_value = value.strip().lower()
        if text_value in {"", "nan", "none", "null", "<na>"}:
            return None
        return 1 if text_value in {"true", "1", "yes", "y", "resolved"} else 0
    if pd.isna(value):
        return None
    return 1 if bool(value) else 0


def _recommendation_level(action: Any) -> str:
    text_value = str(action or "").strip().lower()
    if "load balance" in text_value or "carrier" in text_value:
        return "cell"
    if "sector" in text_value:
        return "sector"
    if "site" in text_value:
        return "site"
    return "cell"


def prepare_recommendation_results(
    recommendations: pd.DataFrame,
    *,
    project_id: int,
    baseline_job_id: str,
    model_run_id: str,
    region: str,
    operator: str | None,
    source_model1_run_id: str | None,
    source_model2_run_id: str | None,
    model_version: str | None = None,
    artifact_recommendations_path: str | None = None,
    artifact_after_rf_path: str | None = None,
    after_rf_rows: int | None = None,
    rf_runtime_sec: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in recommendations.reset_index(drop=True).iterrows():
        future_prb = _safe_num(row.get("prb_before_pct"))
        future_rrc = _safe_num(row.get("rrc_before_pct"))
        after_prb = _safe_num(row.get("projected_prb_after_pct"))
        after_rrc = _safe_num(row.get("projected_rrc_after_pct"))
        action = _safe_str(row.get("action")) or "No Action"
        status = _safe_str(row.get("status")) or "Unknown"
        node_cell_id = _safe_str(row.get("Node_Cell_ID") or row.get("node_cell_id")) or ""
        rows.append(
            {
                "project_id": int(project_id),
                "baseline_job_id": str(baseline_job_id),
                "model_run_id": str(model_run_id),
                "recommendation_id": f"{model_run_id}_{idx + 1:04d}",
                "source_model1_run_id": source_model1_run_id,
                "source_model2_run_id": source_model2_run_id,
                "region": str(region or "india").lower(),
                "operator": operator,
                "site_id": _safe_str(row.get("site_id")) or "",
                "sector_id": _safe_str(row.get("sector_id")) or "",
                "node_cell_id": node_cell_id,
                "canonical_physical_cell_id": _safe_str(row.get("canonical_physical_cell_id") or row.get("canonical_cell_id") or node_cell_id),
                "sector_congested_node_cell_ids": _safe_str(row.get("sector_congested_node_cell_ids")),
                "band": _safe_str(row.get("band")) or "",
                "earfcn": _safe_str(row.get("earfcn")),
                "grid_count": _safe_int(row.get("grid_count")),
                "congested_grid_count": _safe_int(row.get("congested_grid_count")),
                "future_prb_before_pct": future_prb,
                "future_rrc_before_pct": future_rrc,
                "future_rrc_users_before": _safe_num(row.get("rrc_users_before")),
                "future_estimated_offered_traffic_mbps": _safe_num(row.get("estimated_offered_traffic_mbps")),
                "future_pressure_before_pct": max([v for v in [future_prb, future_rrc] if v is not None], default=None),
                "recommended_action": action,
                "recommendation_level": _recommendation_level(action),
                "status": status,
                "resolved_flag": 1 if status.lower() == "resolved" else 0,
                "decision_path": _safe_str(row.get("decision_path")),
                "attempted_actions": _safe_str(row.get("attempted_actions")),
                "action_reason": _safe_str(row.get("action_reason")),
                "next_step": _safe_str(row.get("next_step")),
                "priority_score": _safe_num(row.get("priority_score")),
                "load_balance_possible": _safe_bool(row.get("load_balance_possible")),
                "selected_peer_node_cell_id": _safe_str(row.get("selected_peer_node_cell_id")),
                "selected_peer_band": _safe_str(row.get("selected_peer_band")),
                "recommended_band_to_add": _safe_str(row.get("recommended_band_to_add")),
                "available_bands_to_add": _safe_str(row.get("available_bands_to_add")),
                "carrier_addition_possible": _safe_bool(row.get("carrier_addition_possible")),
                "carrier_addition_blocked": _safe_bool(row.get("carrier_addition_blocked")),
                "max_supported_carriers": _safe_int(row.get("max_supported_carriers")),
                "existing_carrier_count": _safe_int(row.get("existing_carrier_count")),
                "existing_carriers": _safe_str(row.get("existing_carriers")),
                "new_sector_value": _safe_str(row.get("new_sector_value")),
                "new_site_value": _safe_str(row.get("new_site_value")),
                "projected_prb_after_pct": after_prb,
                "projected_rrc_after_pct": after_rrc,
                "projected_rrc_users_after": _safe_num(row.get("projected_rrc_users_after")),
                "pressure_after_pct": max([v for v in [after_prb, after_rrc] if v is not None], default=None),
                "resimulation_required": _safe_bool(row.get("resimulation_required")) or 0,
                "resimulation_flow": _safe_str(row.get("resimulation_flow")),
                "after_rf_job_id": str(model_run_id),
                "after_rf_rows": after_rf_rows,
                "affected_cells_count": None,
                "affected_sites_count": None,
                "rf_runtime_sec": rf_runtime_sec,
                "model_name": "future_capacity_recommendation",
                "model_version": model_version,
                "input_source": "future_coverage_and_demand_forecast",
                "artifact_recommendations_path": artifact_recommendations_path,
                "artifact_after_rf_path": artifact_after_rf_path,
            }
        )
    return pd.DataFrame(rows, columns=RECOMMENDATION_RESULT_COLUMNS)


def prepare_after_rf_results(
    after_rf: pd.DataFrame,
    *,
    project_id: int,
    baseline_job_id: str,
    model_run_id: str,
    region: str,
    operator: str | None,
    artifact_path: str | None = None,
) -> pd.DataFrame:
    if after_rf.empty:
        return pd.DataFrame(columns=RF_SURFACE_RESULT_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, row in after_rf.iterrows():
        lat = _safe_num(row.get("lat") if "lat" in after_rf.columns else row.get("grid_centroid_lat"))
        lon = _safe_num(row.get("lon") if "lon" in after_rf.columns else row.get("grid_centroid_lon"))
        if lat is None or lon is None:
            continue
        node_cell_id = _safe_str(row.get("Node_Cell_ID") or row.get("node_cell_id") or row.get("nodeb_id_cell_id"))
        rows.append(
            {
                "project_id": int(project_id),
                "baseline_job_id": str(baseline_job_id),
                "model_run_id": str(model_run_id),
                "recommendation_id": None,
                "rf_stage": "after",
                "region": str(region or "india").lower(),
                "operator": _safe_str(row.get("operator")) or operator,
                "grid_id": _safe_int(row.get("grid_id")),
                "grid_row": _safe_int(row.get("grid_row")),
                "grid_col": _safe_int(row.get("grid_col")),
                "lat": lat,
                "lon": lon,
                "lat_6dp": round(lat, 6),
                "lon_6dp": round(lon, 6),
                "site_id": _safe_str(row.get("site_id") or row.get("Site ID") or row.get("node_b_id")),
                "sector_id": _safe_str(row.get("sector_identity") or row.get("sector_id")),
                "node_cell_id": node_cell_id,
                "canonical_physical_cell_id": _safe_str(row.get("canonical_physical_cell_id") or row.get("canonical_cell_id") or node_cell_id),
                "band": _safe_str(row.get("band")),
                "earfcn": _safe_str(row.get("earfcn")),
                "pred_rsrp": _safe_num(row.get("pred_rsrp")),
                "pred_rsrq": _safe_num(row.get("pred_rsrq")),
                "pred_sinr": _safe_num(row.get("pred_sinr")),
                "pred_rsrp_smoothed": _safe_num(row.get("pred_rsrp_smoothed")),
                "pred_rsrq_smoothed": _safe_num(row.get("pred_rsrq_smoothed")),
                "pred_sinr_smoothed": _safe_num(row.get("pred_sinr_smoothed")),
                "best_server_flag": _safe_bool(row.get("best_server_flag")),
                "affected_flag": 1,
                "rf_source": "future_capacity_affected_rf_rerun",
                "artifact_path": artifact_path or _safe_str(row.get("_artifact_file")),
            }
        )
    return pd.DataFrame(rows, columns=RF_SURFACE_RESULT_COLUMNS)


def save_recommendation_results(df: pd.DataFrame, region: str = "india") -> int:
    if df.empty:
        return 0
    ensure_results_tables(region)
    out = df.loc[:, RECOMMENDATION_RESULT_COLUMNS].copy()
    keys = out[["project_id", "baseline_job_id", "model_run_id"]].drop_duplicates()
    with get_engine(region).begin() as conn:
        for _, row in keys.iterrows():
            conn.execute(
                text(
                    f"""
                    DELETE FROM {RECOMMENDATION_TABLE_NAME}
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
        out.to_sql(RECOMMENDATION_TABLE_NAME, con=conn, if_exists="append", index=False, method="multi", chunksize=1000)
    return int(len(out))


def save_after_rf_results(df: pd.DataFrame, region: str = "india") -> int:
    if df.empty:
        return 0
    ensure_results_tables(region)
    out = df.loc[:, RF_SURFACE_RESULT_COLUMNS].copy()
    keys = out[["project_id", "baseline_job_id", "model_run_id"]].drop_duplicates()
    with get_engine(region).begin() as conn:
        for _, row in keys.iterrows():
            conn.execute(
                text(
                    f"""
                    DELETE FROM {RF_SURFACE_TABLE_NAME}
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
        out.to_sql(RF_SURFACE_TABLE_NAME, con=conn, if_exists="append", index=False, method="multi", chunksize=2000)
    return int(len(out))
