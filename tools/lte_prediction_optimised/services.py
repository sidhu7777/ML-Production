import uuid
import threading
import pandas as pd
import numpy as np
import os
import datetime
import traceback
from sqlalchemy import create_engine, text

# Import your ML engine functions
from .ml_engine import (
    fetch_baseline,
    fetch_site_data,
    fetch_optimized_sites,
    resolve_site_prediction_scenario_operator,
    compute_k1k2_for_cells,
    _compute_affected_cells,
    _normalize_site_df,
    run_prediction_only_optimized,
    replace_cells,
)
from ..lte_tilt_recommandation.cell_identity import canonical_cell_id
from ..lte_tilt_recommandation.candidate_validation import _apply_rf_delta as _apply_recommendation_rf_delta
from utils.python_bridge import PythonBridgeError, get_bridge_client

# Global dictionary to track job status
JOBS = {}


def _metric_range(df, col):
    if col not in df.columns:
        return "n/a"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return "n/a"
    return f"{series.min():.4f}..{series.max():.4f}"


def _df_summary(stage, df):
    print(f"[LTE_OPT][{stage}] shape={df.shape}")
    print(f"[LTE_OPT][{stage}] columns={list(df.columns)}")
    for col in ["Node_Cell_ID", "canonical_cell_id", "cell_id", "node_b_id", "site_id", "nodeb_id_cell_id", "Operator"]:
        if col in df.columns:
            print(f"[LTE_OPT][{stage}] distinct_{col}={int(df[col].nunique(dropna=True))}")
    for col in ["pred_rsrp", "pred_rsrq", "pred_sinr"]:
        if col in df.columns:
            print(f"[LTE_OPT][{stage}] range_{col}={_metric_range(df, col)}")

# Database connection
engine = {
    "india": create_engine(
        os.getenv("DATABASE_URL"),
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL") else None,
    
    "taiwan": create_engine(
        os.getenv("DATABASE_URL_Taiwan"), 
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL_Taiwan") else None
}


def _resolve_engine(region="india"):
    return engine.get(str(region).lower(), engine["india"])


def _latest_baseline_job_id(project_id, region="india", operator=None):
    bridge = get_bridge_client()
    if bridge:
        params = {"projectId": int(project_id), "region": region}
        if operator and str(operator).strip().lower() != "all":
            params["operator"] = str(operator).strip()
        payload = bridge._request("GET", "GetLatestLteBaselineJobId", params=params)
        job_id = payload.get("JobId") or payload.get("jobId")
        return str(job_id) if job_id is not None else None
    current_engine = _resolve_engine(region)
    where_parts = ["project_id = :project_id"]
    params = {"project_id": int(project_id)}
    if operator and str(operator).strip().lower() != "all":
        where_parts.append("LOWER(TRIM(operator)) = :operator")
        params["operator"] = str(operator).strip().lower()
    query = text(f"""
        SELECT job_id
        FROM lte_prediction_baseline_results
        WHERE {" AND ".join(where_parts)}
        ORDER BY created_at DESC
        LIMIT 1
    """)
    with current_engine.connect() as conn:
        row = conn.execute(query, params).fetchone()
    return str(row[0]) if row and row[0] is not None else None


def _clean_id(value):
    return canonical_cell_id(value)


def _rf_id(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null", "<na>"}:
        return ""
    text = text.replace("|", "_")
    while ".0_" in text:
        text = text.replace(".0_", "_")
    if text.endswith(".0"):
        text = text[:-2]
    return text.strip("_")


def _cell_suffix(value):
    text_value = _clean_id(value)
    return text_value.rsplit("_", 1)[-1] if text_value else ""


def _values_changed(current_value, recommended_value):
    current_num = pd.to_numeric(pd.Series([current_value]), errors="coerce").iloc[0]
    recommended_num = pd.to_numeric(pd.Series([recommended_value]), errors="coerce").iloc[0]
    if pd.notna(current_num) and pd.notna(recommended_num):
        return not np.isclose(float(current_num), float(recommended_num), equal_nan=True)
    return str(current_value).strip() != str(recommended_value).strip()


def _latest_recommendation_scenario_id(project_id, region, operator=None):
    bridge = get_bridge_client()
    if bridge:
        params = {"projectId": int(project_id)}
        if operator:
            params["operator"] = operator
        payload = bridge._request("GET", "GetLatestRfOptimizationScenarioId", params=params)
        scenario_id = payload.get("ScenarioId") or payload.get("scenarioId")
        if scenario_id is None:
            op_msg = f" operator={operator}" if operator else ""
            raise FileNotFoundError(f"No tilt recommendation rows found for project_id={project_id}{op_msg}")
        return int(scenario_id)

    current_engine = _resolve_engine(region)
    where_parts = ["project_id = :project_id"]
    params = {"project_id": int(project_id)}
    if operator:
        where_parts.append("LOWER(TRIM(operator)) = :operator")
        params["operator"] = str(operator).strip().lower()
    query = text(
        f"""
        SELECT MAX(scenario_id)
        FROM rf_optimization_results
        WHERE {' AND '.join(where_parts)}
        """
    )
    with current_engine.connect() as conn:
        scenario_id = conn.execute(query, params).scalar()
    if scenario_id is None:
        op_msg = f" operator={operator}" if operator else ""
        raise FileNotFoundError(f"No tilt recommendation rows found for project_id={project_id}{op_msg}")
    return int(scenario_id)


def _fetch_recommendation_rows(project_id, region, operator=None, recommendation_scenario_id=None):
    scenario_id = recommendation_scenario_id
    if scenario_id is None:
        scenario_id = _latest_recommendation_scenario_id(project_id, region, operator=operator)

    bridge = get_bridge_client()
    if bridge:
        params = {"projectId": int(project_id), "scenarioId": int(scenario_id)}
        if operator:
            params["operator"] = operator
        reco_df = bridge.get_rows("GetRfOptimizationRows", params, limit=50000)
        if reco_df.empty and operator:
            print(
                f"[LTE_OPT][RECOMMENDATION_ROWS] scenario_id={scenario_id} "
                f"operator={operator} rows=0 retry_without_operator=True"
            )
            reco_df = bridge.get_rows(
                "GetRfOptimizationRows",
                {"projectId": int(project_id), "scenarioId": int(scenario_id)},
                limit=50000,
            )
        if reco_df.empty:
            raise FileNotFoundError(
                f"No rows found in rf_optimization_results for project_id={project_id} "
                f"scenario_id={scenario_id} operator={operator or 'all'}"
            )
        return int(scenario_id), reco_df

    current_engine = _resolve_engine(region)
    where_parts = ["project_id = :project_id", "scenario_id = :scenario_id"]
    params = {"project_id": int(project_id), "scenario_id": int(scenario_id)}
    if operator:
        where_parts.append("LOWER(TRIM(operator)) = :operator")
        params["operator"] = str(operator).strip().lower()

    query = text(
        f"""
        SELECT
            project_id,
            scenario_id,
            operator,
            cell_id,
            technology,
            parameter,
            current_value,
            recommended_value,
            reason,
            swap_sector_detected,
            rsrp_threshold,
            rsrq_threshold,
            sinr_threshold,
            created_at
        FROM rf_optimization_results
        WHERE {' AND '.join(where_parts)}
        ORDER BY cell_id, parameter, id
        """
    )
    with current_engine.connect() as conn:
        reco_df = pd.read_sql(query, conn, params=params)
    if reco_df.empty and operator:
        fallback_query = text(
            """
            SELECT
                project_id,
                scenario_id,
                operator,
                cell_id,
                technology,
                parameter,
                current_value,
                recommended_value,
                reason,
                swap_sector_detected,
                rsrp_threshold,
                rsrq_threshold,
                sinr_threshold,
                created_at
            FROM rf_optimization_results
            WHERE project_id = :project_id AND scenario_id = :scenario_id
            ORDER BY cell_id, parameter, id
            """
        )
        print(
            f"[LTE_OPT][RECOMMENDATION_ROWS] scenario_id={scenario_id} "
            f"operator={operator} rows=0 retry_without_operator=True"
        )
        with current_engine.connect() as conn:
            reco_df = pd.read_sql(
                fallback_query,
                conn,
                params={"project_id": int(project_id), "scenario_id": int(scenario_id)},
            )
    if reco_df.empty:
        raise FileNotFoundError(
            f"No rows found in rf_optimization_results for project_id={project_id} "
            f"scenario_id={scenario_id} operator={operator or 'all'}"
        )
    return int(scenario_id), reco_df


def _actionable_recommendations(reco_df):
    work = reco_df.copy()
    work["cell_id_clean"] = work["cell_id"].map(_rf_id)
    work["parameter_norm"] = work["parameter"].astype(str).str.strip().str.lower()
    changed_mask = work.apply(lambda row: _values_changed(row["current_value"], row["recommended_value"]), axis=1)
    work = work.loc[changed_mask].copy()
    supported = {
        "etilt",
        "e tilt",
        "electrical tilt",
        "azimuth",
        "tx power",
        "power",
        "mechanical tilt",
        "mtilt",
        "height",
        "antenna height",
    }
    work = work.loc[work["parameter_norm"].isin(supported)].copy()
    if work.empty:
        raise ValueError("Recommendation scenario has no actionable supported parameter changes")
    return work


def _site_match_mask(site_df, recommendation_cell_id):
    rec_rf_id = _rf_id(recommendation_cell_id)
    rec_id = _clean_id(recommendation_cell_id)
    rec_suffix = _cell_suffix(rec_id)
    node_cell_rf = site_df["Node_Cell_ID"].astype(str).map(_rf_id)
    mask = node_cell_rf == rec_rf_id
    if not mask.any() and "cell_id" in site_df.columns:
        cell_rf = site_df["cell_id"].astype(str).map(_rf_id)
        mask = cell_rf == rec_rf_id
    if not mask.any():
        node_cell = site_df["Node_Cell_ID"].astype(str).map(_clean_id)
        mask = node_cell == rec_id
    if not mask.any() and "cell_id" in site_df.columns:
        cell_id = site_df["cell_id"].astype(str).map(_clean_id)
        mask = cell_id == rec_id
    if not mask.any() and rec_suffix and "cell_id" in site_df.columns:
        cell_suffix = site_df["cell_id"].astype(str).map(_cell_suffix)
        mask = cell_suffix == rec_suffix
    return mask


def _apply_recommendations_to_sites(site_df, actionable_df):
    modified = _normalize_site_df(site_df, log_stage="RECOMMENDATION_OPT_INPUT")
    compare_cols = ["lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "tx_power", "antenna_height"]
    for col in compare_cols:
        modified[f"orig_{col}"] = pd.to_numeric(modified[col], errors="coerce")

    parameter_map = {
        "etilt": "electrical_tilt",
        "e tilt": "electrical_tilt",
        "electrical tilt": "electrical_tilt",
        "azimuth": "azimuth",
        "tx power": "tx_power",
        "power": "tx_power",
        "mechanical tilt": "mechanical_tilt",
        "mtilt": "mechanical_tilt",
        "height": "antenna_height",
        "antenna height": "antenna_height",
    }

    applied_rows = []
    modified["optimization_applied"] = False
    for _, row in actionable_df.iterrows():
        target_col = parameter_map.get(str(row["parameter_norm"]))
        if not target_col:
            continue
        mask = _site_match_mask(modified, row["cell_id_clean"])
        if not mask.any():
            applied_rows.append({
                "recommendation_cell_id": row["cell_id"],
                "parameter": row["parameter"],
                "status": "not_matched_to_site",
                "recommended_value": row["recommended_value"],
            })
            continue

        rec_value = pd.to_numeric(pd.Series([row["recommended_value"]]), errors="coerce").iloc[0]
        if pd.isna(rec_value):
            applied_rows.append({
                "recommendation_cell_id": row["cell_id"],
                "parameter": row["parameter"],
                "status": "invalid_recommended_value",
                "recommended_value": row["recommended_value"],
            })
            continue

        before_values = modified.loc[mask, target_col].tolist()
        modified.loc[mask, target_col] = float(rec_value)
        modified.loc[mask, "optimization_applied"] = True
        for node_cell_id, before_value in zip(modified.loc[mask, "Node_Cell_ID"].astype(str), before_values):
            applied_rows.append({
                "recommendation_cell_id": row["cell_id"],
                "matched_node_cell_id": node_cell_id,
                "parameter": row["parameter"],
                "target_column": target_col,
                "current_value": before_value,
                "recommended_value": float(rec_value),
                "status": "applied",
                "reason": row.get("reason"),
            })

    applied_df = pd.DataFrame(applied_rows)
    if applied_df.empty or not (applied_df["status"].astype(str) == "applied").any():
        raise ValueError("No tilt recommendation rows could be applied to site_prediction rows")
    return modified, applied_df


def _json_safe(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_present(row, names):
    for name in names:
        if name in row.index:
            value = row.get(name)
            if value is not None and not pd.isna(value):
                return value
    return None


def _build_site_prediction_update_rows(project_id, public_scenario_id, modified_site_df, applied_df):
    applied = applied_df.loc[applied_df["status"].astype(str) == "applied"].copy()
    if applied.empty:
        return []

    applied_cells = {
        _clean_id(value)
        for value in applied["matched_node_cell_id"].dropna().astype(str)
        if str(value).strip()
    }
    work = modified_site_df.copy()
    work["_recommendation_cell_key"] = work["Node_Cell_ID"].astype(str).map(_clean_id)
    work = work.loc[work["_recommendation_cell_key"].isin(applied_cells)].copy()
    if work.empty:
        return []

    value_map = {
        "lat": ["lat", "latitude"],
        "lon": ["lon", "longitude"],
        "azimuth": ["azimuth"],
        "e_tilt": ["electrical_tilt", "Etilt", "e_tilt"],
        "m_tilt": ["mechanical_tilt", "Mtilt", "m_tilt"],
        "tx_power": ["tx_power", "maximum_transmission_power_of_resource"],
        "height": ["antenna_height", "Height", "height"],
    }

    records = []
    seen_source_ids = set()
    for _, row in work.iterrows():
        source_id = _first_present(row, ["source_id", "site_prediction_id", "original_id", "id"])
        try:
            source_id = int(source_id)
        except (TypeError, ValueError):
            source_id = None
        if not source_id or source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)

        payload = {
            "id": source_id,
            "source_id": source_id,
            "project_id": int(project_id),
            "scenario": int(public_scenario_id),
        }
        for api_key, source_names in value_map.items():
            value = _first_present(row, source_names)
            if value is not None and not pd.isna(value):
                payload[api_key] = _json_safe(value)
        records.append(payload)

    if not records:
        raise ValueError("No source site_prediction rows could be resolved for recommendation site scenario save")
    return records


def _site_prediction_api_root(bridge):
    api_root = str(getattr(bridge, "api_root_url", "") or "").rstrip("/")
    suffix = "/api/pythonbridge"
    if api_root.lower().endswith(suffix):
        api_root = api_root[: -len(suffix)]
    return api_root


def _save_site_prediction_updates_via_api(project_id, public_scenario_id, update_rows):
    bridge = get_bridge_client()
    if not bridge:
        return None
    api_root = _site_prediction_api_root(bridge)
    if not api_root:
        return None
    try:
        result = bridge._request_url(
            "POST",
            f"{api_root}/api/MapView/UpdateSitePrediction",
            json=update_rows,
            timeout=int(os.getenv("PYTHON_BRIDGE_SITE_UPDATE_TIMEOUT_SECONDS", "120")),
        )
    except PythonBridgeError as exc:
        print(
            f"[LTE_OPT][SITE_SCENARIO_SAVE] source=mapview_api project_id={project_id} "
            f"scenario={public_scenario_id} fallback=direct_db reason={exc}"
        )
        return None
    rows_affected = int(result.get("RowsAffected") or result.get("rowsAffected") or 0)
    print(
        f"[LTE_OPT][SITE_SCENARIO_SAVE] source=mapview_api project_id={project_id} "
        f"scenario={public_scenario_id} payload_rows={len(update_rows)} rows_affected={rows_affected}"
    )
    return rows_affected


def _table_columns(conn, table_name):
    rows = conn.execute(
        text(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = :table_name
            """
        ),
        {"table_name": table_name},
    ).fetchall()
    return {str(row[0]) for row in rows if row and row[0] is not None}


def _ensure_site_prediction_optimized_table(conn):
    conn.execute(text("CREATE TABLE IF NOT EXISTS site_prediction_optimized LIKE site_prediction"))
    existing = _table_columns(conn, "site_prediction_optimized")
    required = {
        "scenario": "INT NOT NULL DEFAULT 1",
        "site_prediction_id": "INT NULL",
        "is_updated": "TINYINT(1) NOT NULL DEFAULT 1",
        "version": "INT NOT NULL DEFAULT 1",
        "status": "VARCHAR(20) NULL DEFAULT 'updated'",
        "created_at": "DATETIME NULL",
        "updated_at": "DATETIME NULL",
        "updated_by": "VARCHAR(100) NULL",
    }
    for column, definition in required.items():
        if column not in existing:
            conn.execute(text(f"ALTER TABLE site_prediction_optimized ADD COLUMN `{column}` {definition}"))


def _save_site_prediction_updates_direct(project_id, public_scenario_id, update_rows, region):
    current_engine = _resolve_engine(region)
    with current_engine.begin() as conn:
        _ensure_site_prediction_optimized_table(conn)
        source_columns = _table_columns(conn, "site_prediction")
        optimized_columns = _table_columns(conn, "site_prediction_optimized")
        reserved = {
            "id",
            "site_prediction_id",
            "scenario",
            "scenario_id",
            "is_updated",
            "version",
            "status",
            "created_at",
            "updated_at",
            "updated_by",
        }
        copy_columns = [
            column
            for column in source_columns
            if column in optimized_columns and column not in reserved
        ]
        if not copy_columns:
            raise ValueError("No common site prediction columns available for optimized scenario save")

        insert_columns = ", ".join(f"`{column}`" for column in copy_columns)
        select_columns = ", ".join(f"sp.`{column}`" for column in copy_columns)
        total_updated = 0
        for row in update_rows:
            source_id = int(row["source_id"])
            conn.execute(
                text(
                    f"""
                    INSERT INTO site_prediction_optimized (
                        site_prediction_id, scenario, {insert_columns},
                        is_updated, version, status, created_at, updated_at, updated_by
                    )
                    SELECT
                        sp.id, :scenario, {select_columns},
                        1, 0, 'updated', UTC_TIMESTAMP(), UTC_TIMESTAMP(), 'backend'
                    FROM site_prediction sp
                    WHERE sp.id = :source_id
                      AND sp.tbl_project_id = :project_id
                      AND NOT EXISTS (
                          SELECT 1
                          FROM site_prediction_optimized spo
                          WHERE spo.site_prediction_id = sp.id
                            AND spo.scenario = :scenario
                      )
                    """
                ),
                {
                    "scenario": int(public_scenario_id),
                    "source_id": source_id,
                    "project_id": int(project_id),
                },
            )

            update_pairs = []
            params = {
                "scenario": int(public_scenario_id),
                "source_id": source_id,
                "project_id": int(project_id),
            }
            for key, db_column in {
                "lat": "latitude",
                "lon": "longitude",
                "azimuth": "azimuth",
                "e_tilt": "e_tilt",
                "m_tilt": "m_tilt",
                "tx_power": "tx_power",
                "height": "height",
            }.items():
                if key in row and db_column in optimized_columns:
                    param_name = f"value_{key}"
                    update_pairs.append(f"spo.`{db_column}` = :{param_name}")
                    params[param_name] = row[key]

            if not update_pairs:
                continue

            result = conn.execute(
                text(
                    f"""
                    UPDATE site_prediction_optimized spo
                    SET {", ".join(update_pairs)},
                        spo.is_updated = 1,
                        spo.status = 'updated',
                        spo.version = COALESCE(spo.version, 0) + 1,
                        spo.updated_at = UTC_TIMESTAMP(),
                        spo.updated_by = 'backend'
                    WHERE spo.site_prediction_id = :source_id
                      AND spo.scenario = :scenario
                      AND COALESCE(spo.tbl_project_id, 0) = :project_id
                    """
                ),
                params,
            )
            total_updated += int(result.rowcount or 0)

    print(
        f"[LTE_OPT][SITE_SCENARIO_SAVE] source=direct_db project_id={project_id} "
        f"scenario={public_scenario_id} payload_rows={len(update_rows)} rows_affected={total_updated}"
    )
    return total_updated


def _save_recommendation_site_prediction_scenario(project_id, public_scenario_id, modified_site_df, applied_df, region):
    update_rows = _build_site_prediction_update_rows(
        project_id,
        public_scenario_id,
        modified_site_df,
        applied_df,
    )
    rows_affected = _save_site_prediction_updates_via_api(project_id, public_scenario_id, update_rows)
    if rows_affected is None:
        rows_affected = _save_site_prediction_updates_direct(project_id, public_scenario_id, update_rows, region)
    if rows_affected <= 0:
        raise RuntimeError(
            f"Recommendation site scenario save wrote 0 rows for project_id={project_id} "
            f"scenario={public_scenario_id}"
        )
    return rows_affected


def _build_scenario_name(cfg):
    provided = str(cfg.get("scenario_name", "")).strip()
    if provided:
        return provided
    target_type = str(cfg.get("target_type", "target")).strip() or "target"
    target_id = str(cfg.get("target_id", "")).strip()
    change_labels = []
    if float(cfg.get("delta_lat", 0) or 0) or float(cfg.get("delta_lon", 0) or 0):
        change_labels.append("Site Move")
    if float(cfg.get("delta_azimuth", 0) or 0):
        change_labels.append("Azimuth Change")
    if float(cfg.get("delta_electrical_tilt", 0) or 0):
        change_labels.append("Electrical Tilt Change")
    if float(cfg.get("delta_mechanical_tilt", 0) or 0):
        change_labels.append("Mechanical Tilt Change")
    if float(cfg.get("delta_tx_power", 0) or 0):
        change_labels.append("Tx Power Change")
    if float(cfg.get("delta_antenna_height", 0) or 0):
        change_labels.append("Antenna Height Change")
    if not change_labels:
        change_labels.append("Optimization Change")
    if target_id:
        return f"{' + '.join(change_labels)} - {target_type} {target_id}"
    return " + ".join(change_labels)


def _build_scenario_description(cfg):
    provided = str(cfg.get("scenario_description", "")).strip()
    if provided:
        return provided

    target_type = str(cfg.get("target_type", "")).strip() or "target"
    target_id = str(cfg.get("target_id", "")).strip() or "unknown"
    parts = [f"Target {target_type} {target_id}"]
    if float(cfg.get("delta_lat", 0) or 0) or float(cfg.get("delta_lon", 0) or 0):
        parts.append(
            f"move(lat={float(cfg.get('delta_lat', 0) or 0):.6f}, lon={float(cfg.get('delta_lon', 0) or 0):.6f})"
        )
    if float(cfg.get("delta_azimuth", 0) or 0):
        parts.append(f"delta_azimuth={float(cfg.get('delta_azimuth', 0) or 0):.2f}")
    if float(cfg.get("delta_electrical_tilt", 0) or 0):
        parts.append(f"delta_electrical_tilt={float(cfg.get('delta_electrical_tilt', 0) or 0):.2f}")
    if float(cfg.get("delta_mechanical_tilt", 0) or 0):
        parts.append(f"delta_mechanical_tilt={float(cfg.get('delta_mechanical_tilt', 0) or 0):.2f}")
    if float(cfg.get("delta_tx_power", 0) or 0):
        parts.append(f"delta_tx_power={float(cfg.get('delta_tx_power', 0) or 0):.2f}")
    if float(cfg.get("delta_antenna_height", 0) or 0):
        parts.append(f"delta_antenna_height={float(cfg.get('delta_antenna_height', 0) or 0):.2f}")
    parts.append(
        f"impact_radius_m={float(cfg.get('impact_radius_m', cfg.get('radius', 500)) or 500):.1f}"
    )
    parts.append(f"neighbor_site_count={int(cfg.get('neighbor_site_count', 2) or 2)}")
    parts.append(f"max_interference_sites={int(cfg.get('max_interference_sites', 10) or 10)}")
    return "; ".join(parts)


class LTEPredictionService_optimised:

    def submit(self, cfg):
        job_id = str(uuid.uuid4())
        region = str(cfg.get("region", "india")).lower()
        scenario_id = cfg.get("scenario_id")
        scenario_row_id = cfg.get("scenario_row_id")
        if scenario_id and scenario_row_id:
            scenario_id = int(scenario_id)
            scenario_row_id = int(scenario_row_id)
        else:
            site_prediction_scenario_id = (
                cfg.get("site_prediction_scenario_id")
                or cfg.get("sitePredictionScenarioId")
                or cfg.get("scenario")
            )
            if site_prediction_scenario_id is not None:
                try:
                    requested_scenario_id = int(site_prediction_scenario_id)
                except (TypeError, ValueError):
                    requested_scenario_id = None
                if requested_scenario_id and requested_scenario_id > 0:
                    cfg["requested_public_scenario_id"] = requested_scenario_id
                    cfg.setdefault("target_type", "manual_site_prediction")
                    cfg.setdefault("target_id", f"site_prediction_scenario_{requested_scenario_id}")
                    cfg.setdefault("scenario_name", f"Manual Optimization - Site Scenario {requested_scenario_id}")
                    cfg.setdefault(
                        "scenario_description",
                        f"Apply saved site_prediction_optimized scenario {requested_scenario_id} and run manual LTE optimized prediction",
                    )
            scenario_row_id, scenario_id = self._create_scenario(cfg, job_id, region)
            cfg["scenario_row_id"] = scenario_row_id
            cfg["scenario_id"] = scenario_id

        JOBS[job_id] = {
            "status": "queued",
            "scenario_row_id": scenario_row_id,
            "scenario_id": scenario_id,
            "project_id": int(cfg["project_id"]),
            "region": region,
            "operator": cfg.get("operator"),
            "baseline_job_id": cfg.get("baseline_job_id"),
        }

        threading.Thread(
            target=self._run,
            args=(job_id, cfg),
            daemon=True
        ).start()

        return {"job_id": job_id, "scenario_id": scenario_id, "scenario_row_id": scenario_row_id}

    def submit_recommendation_optimization(self, cfg):
        job_id = str(uuid.uuid4())
        region = str(cfg.get("region", "india")).lower()
        recommendation_scenario_id = cfg.get("recommendation_scenario_id")
        operator = cfg.get("operator")
        if recommendation_scenario_id is None:
            recommendation_scenario_id = _latest_recommendation_scenario_id(
                cfg["project_id"],
                region,
                operator=operator,
            )
        cfg["recommendation_scenario_id"] = int(recommendation_scenario_id)
        cfg.setdefault("target_type", "recommendation")
        cfg.setdefault("target_id", f"rf_scenario_{int(recommendation_scenario_id)}")
        cfg.setdefault("scenario_name", f"Tilt Recommendation Optimization - RF Scenario {int(recommendation_scenario_id)}")
        cfg.setdefault(
            "scenario_description",
            f"Apply saved tilt recommendation scenario {int(recommendation_scenario_id)} and run cell-level optimized prediction",
        )

        scenario_row_id, scenario_id = self._create_scenario(cfg, job_id, region)
        cfg["scenario_row_id"] = scenario_row_id
        cfg["scenario_id"] = scenario_id

        JOBS[job_id] = {
            "status": "queued",
            "scenario_row_id": scenario_row_id,
            "scenario_id": scenario_id,
            "recommendation_scenario_id": int(recommendation_scenario_id),
            "project_id": int(cfg["project_id"]),
            "region": region,
            "operator": operator,
            "baseline_job_id": cfg.get("baseline_job_id"),
        }

        threading.Thread(
            target=self._run_recommendation_optimization,
            args=(job_id, cfg),
            daemon=True,
        ).start()

        return {
            "job_id": job_id,
            "scenario_id": scenario_id,
            "scenario_row_id": scenario_row_id,
            "recommendation_scenario_id": int(recommendation_scenario_id),
        }

    def get(self, job_id):
        return JOBS.get(job_id)

    def _run(self, job_id, cfg):
        scenario_id = cfg.get("scenario_id")
        scenario_row_id = cfg.get("scenario_row_id")
        region = str(cfg.get("region", "india")).lower()
        try:
            print(
                f"[LTE_OPT][JOB_START] job_id={job_id} project_id={cfg['project_id']} "
                f"region={str(cfg.get('region', 'india')).lower()} operator={cfg.get('operator') or 'all'} "
                f"radius={cfg.get('radius', 500)} grid_resolution={cfg.get('grid_resolution', 10)} "
                f"n_workers={cfg.get('n_workers')}"
            )
            if scenario_row_id:
                self._update_scenario_status(int(scenario_row_id), "running", region=region, job_id=job_id)
            self._update(job_id, "running", "Loading baseline")

            project_id = cfg["project_id"]
            operator = cfg.get("operator")
            polygon_ids = cfg.get("polygon_ids") or cfg.get("polygonIds")
            site_prediction_scenario_id = (
                cfg.get("site_prediction_scenario_id")
                or cfg.get("sitePredictionScenarioId")
                or cfg.get("scenario")
            )
            scenario_operator = resolve_site_prediction_scenario_operator(
                project_id,
                site_prediction_scenario_id,
                region=region,
            ) if site_prediction_scenario_id else None
            if scenario_operator:
                if operator and str(operator).strip().lower() != str(scenario_operator).strip().lower():
                    print(
                        f"[LTE_OPT][OPERATOR_RESOLVE] source=scenario_override "
                        f"requested_operator={operator} scenario_operator={scenario_operator} "
                        f"scenario={site_prediction_scenario_id}"
                    )
                elif not operator:
                    print(
                        f"[LTE_OPT][OPERATOR_RESOLVE] source=scenario "
                        f"operator={scenario_operator} scenario={site_prediction_scenario_id}"
                    )
                operator = scenario_operator
                cfg["operator"] = scenario_operator

            baseline_df = fetch_baseline(
                project_id,
                region=region,
                operator=operator,
                baseline_job_id=cfg.get("baseline_job_id"),
            )
            _df_summary("BASELINE_DF", baseline_df)
            baseline_job_id = None
            if "job_id" in baseline_df.columns and not baseline_df["job_id"].dropna().empty:
                baseline_job_id = str(baseline_df["job_id"].dropna().iloc[0]).strip()

            self._update(job_id, "running", "Loading site data")
            site_df = fetch_site_data(
                project_id,
                region=region,
                operator=operator,
                allowed_cells=baseline_df["Node_Cell_ID"].astype(str).unique().tolist(),
                polygon_ids=polygon_ids,
            )
            _df_summary("SITE_DF", site_df)

            self._update(job_id, "running", f"Loading optimized sites for {operator}")
            opt_sites = fetch_optimized_sites(
                project_id,
                operator,
                region=region,
                polygon_ids=polygon_ids,
                scenario=site_prediction_scenario_id,
            )
            if opt_sites.empty:
                raise ValueError(
                    f"No rows found in site_prediction_optimized for project_id={project_id} "
                    f"operator={operator} scenario={site_prediction_scenario_id or 'latest'}"
                )
            _df_summary("OPTIMIZED_SITE_DF", opt_sites)

            self._update(job_id, "running", "Calculating local K1/K2 from optimized DB changes")
            affected_cells, _, changed_rows = _compute_affected_cells(
                opt_sites,
                float(cfg.get("impact_radius_m", cfg.get("radius", 500)) or cfg.get("radius", 500) or 500),
                int(cfg.get("neighbor_site_count", 2) or 2),
                baseline_df=baseline_df,
                max_neighbors_per_update_cell=cfg.get("max_neighbors_per_update_cell", cfg.get("neighbor_site_count", 2) or 2),
            )
            changed_cells = sorted(changed_rows["Node_Cell_ID"].astype(str).unique().tolist())
            calibration_cells = sorted(affected_cells)
            print(
                f"[LTE_OPT][K1K2_LOCAL_SCOPE] changed_cells={len(changed_cells)} "
                f"affected_cells={len(affected_cells)} calibration_cells={calibration_cells}"
            )
            k1k2_map = compute_k1k2_for_cells(baseline_df, opt_sites, calibration_cells)
            if not k1k2_map:
                raise ValueError("No calibrated cells found from DB-driven optimized site changes")

            params = {
                "radius": cfg.get("radius", 500),
                "grid_resolution": cfg.get("grid_resolution", 10),
                "n_workers": cfg.get("n_workers"),
                "antenna_gain": 18,
                "cable_loss": 2,
                "ue_height": 1.5,
                "frequency_mhz": 1800,
                "bandwidth_mhz": 10,
                "project_id": project_id,
                "region": region,
                "baseline_job_id": baseline_job_id,
                "impact_radius_m": cfg.get("impact_radius_m", cfg.get("radius", 500) or 500),
                "neighbor_site_count": cfg.get("neighbor_site_count", 2) or 2,
                "max_interference_sites": cfg.get("max_interference_sites", 10) or 10,
                "max_neighbors_per_update_cell": cfg.get("max_neighbors_per_update_cell", cfg.get("neighbor_site_count", 2) or 2),
                "baseline_df": baseline_df,
                "recompute_cells": affected_cells,
            }

            self._update(job_id, "running", "Running prediction")
            optimized_df = run_prediction_only_optimized(
                opt_sites,
                k1k2_map,
                params
            )
            _df_summary("OPTIMIZED_RF_OUTPUT_DF", optimized_df)

            self._update(job_id, "running", "Saving CSV")

            # Save the CSV
            file_path = self._save_csv(optimized_df, project_id, operator)

            db_df = self._format_for_db(
                optimized_df,
                project_id,
                job_id,
                operator,
                scenario_id=scenario_row_id,
                public_scenario_id=scenario_id,
            )
            _df_summary("OPTIMIZED_DB_PAYLOAD", db_df)

            self._save_to_db(db_df, region=region)

            JOBS[job_id]["output"] = file_path
            JOBS[job_id]["rows"] = len(optimized_df)

            if scenario_row_id:
                self._update_scenario_status(int(scenario_row_id), "done", region=region, job_id=job_id)
            self._update(job_id, "done", "Completed")

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            if scenario_row_id:
                self._update_scenario_status(int(scenario_row_id), "failed", region=region, job_id=job_id)
            print(" ERROR:", traceback.format_exc())

    def _run_recommendation_optimization(self, job_id, cfg):
        scenario_id = cfg.get("scenario_id")
        scenario_row_id = cfg.get("scenario_row_id")
        recommendation_scenario_id = cfg.get("recommendation_scenario_id")
        region = str(cfg.get("region", "india")).lower()
        try:
            project_id = cfg["project_id"]
            operator = cfg.get("operator")
            polygon_ids = cfg.get("polygon_ids") or cfg.get("polygonIds")
            print(
                f"[LTE_OPT][RECOMMENDATION_JOB_START] job_id={job_id} project_id={project_id} "
                f"region={region} operator={operator} recommendation_scenario_id={recommendation_scenario_id}"
            )
            if scenario_row_id:
                self._update_scenario_status(int(scenario_row_id), "running", region=region, job_id=job_id)

            self._update(job_id, "running", "Loading recommendation rows")
            recommendation_scenario_id, reco_df = _fetch_recommendation_rows(
                project_id,
                region,
                operator=operator,
                recommendation_scenario_id=recommendation_scenario_id,
            )
            actionable_df = _actionable_recommendations(reco_df)
            _df_summary("RECOMMENDATION_ROWS", reco_df)
            _df_summary("RECOMMENDATION_ACTIONABLE_ROWS", actionable_df)

            self._update(job_id, "running", "Loading baseline")
            baseline_df = fetch_baseline(
                project_id,
                region=region,
                operator=operator,
                baseline_job_id=cfg.get("baseline_job_id"),
            )
            _df_summary("BASELINE_DF", baseline_df)
            baseline_job_id = None
            if "job_id" in baseline_df.columns and not baseline_df["job_id"].dropna().empty:
                baseline_job_id = str(baseline_df["job_id"].dropna().iloc[0]).strip()

            self._update(job_id, "running", "Loading site data")
            site_df = fetch_site_data(
                project_id,
                region=region,
                operator=operator,
                allowed_cells=baseline_df["Node_Cell_ID"].astype(str).unique().tolist(),
                polygon_ids=polygon_ids,
            )
            _df_summary("SITE_DF", site_df)

            self._update(job_id, "running", "Applying recommendation changes")
            modified_site_df, applied_df = _apply_recommendations_to_sites(site_df, actionable_df)
            _df_summary("RECOMMENDATION_APPLIED_ROWS", applied_df)
            _df_summary("RECOMMENDATION_MODIFIED_SITE_DF", modified_site_df)

            self._update(job_id, "running", "Saving recommendation site scenario")
            saved_site_rows = _save_recommendation_site_prediction_scenario(
                project_id,
                scenario_id,
                modified_site_df,
                applied_df,
                region,
            )

            affected_cells, affected_sites, changed_rows = _compute_affected_cells(
                modified_site_df,
                float(cfg.get("impact_radius_m", cfg.get("radius", 500)) or cfg.get("radius", 500) or 500),
                int(cfg.get("neighbor_site_count", 2) or 2),
                baseline_df=baseline_df,
                max_neighbors_per_update_cell=cfg.get("max_neighbors_per_update_cell", cfg.get("neighbor_site_count", 2) or 2),
            )
            changed_cells = sorted(changed_rows["Node_Cell_ID"].astype(str).unique().tolist())
            calibration_cells = sorted(affected_cells)
            print(
                f"[LTE_OPT][RECOMMENDATION_SCOPE] changed_cells={len(changed_cells)} "
                f"affected_cells={len(affected_cells)} affected_sites={len(affected_sites)} "
                f"recommendation_scenario_id={recommendation_scenario_id}"
            )

            self._update(job_id, "running", "Calculating local K1/K2")
            k1k2_map = compute_k1k2_for_cells(baseline_df, modified_site_df, calibration_cells)
            if not k1k2_map:
                raise ValueError("No calibrated cells found after applying recommendation changes")

            params = {
                "radius": cfg.get("radius", 500),
                "grid_resolution": cfg.get("grid_resolution", 10),
                "n_workers": cfg.get("n_workers"),
                "antenna_gain": 18,
                "cable_loss": 2,
                "ue_height": 1.5,
                "frequency_mhz": 1800,
                "bandwidth_mhz": 10,
                "project_id": project_id,
                "region": region,
                "baseline_job_id": baseline_job_id,
                "baseline_df": baseline_df,
                "prediction_points_df": baseline_df,
                "strict_prediction_points": True,
                "impact_radius_m": cfg.get("impact_radius_m", cfg.get("radius", 500) or 500),
                "neighbor_site_count": cfg.get("neighbor_site_count", 2) or 2,
                "max_interference_sites": cfg.get("max_interference_sites", 10) or 10,
                "max_neighbors_per_update_cell": cfg.get("max_neighbors_per_update_cell", cfg.get("neighbor_site_count", 2) or 2),
                "recompute_cells": affected_cells,
            }

            self._update(job_id, "running", "Running recommendation optimized prediction")
            baseline_rf_df = run_prediction_only_optimized(site_df, k1k2_map, params)
            optimized_df = run_prediction_only_optimized(modified_site_df, k1k2_map, params)
            if optimized_df.empty:
                raise RuntimeError("Recommendation optimization produced no prediction rows")
            merged_df, rf_delta_metrics = _apply_recommendation_rf_delta(
                baseline_df,
                baseline_rf_df,
                optimized_df,
            )
            print(f"[LTE_OPT][RECOMMENDATION_RF_DELTA] {rf_delta_metrics}")
            _df_summary("RECOMMENDATION_BASELINE_RF_OUTPUT_DF", baseline_rf_df)
            _df_summary("RECOMMENDATION_OPTIMIZED_RF_OUTPUT_DF", optimized_df)
            _df_summary("RECOMMENDATION_OPTIMIZED_DELTA_APPLIED_DF", merged_df)

            self._update(job_id, "running", "Saving CSV")
            file_path = self._save_csv(merged_df, project_id, operator or "recommendation")

            db_df = self._format_for_db(
                merged_df,
                project_id,
                job_id,
                operator or "Recommendation",
                scenario_id=scenario_row_id,
                public_scenario_id=scenario_id,
            )
            _df_summary("RECOMMENDATION_OPTIMIZED_DB_PAYLOAD", db_df)
            self._save_to_db(db_df, region=region)

            JOBS[job_id].update({
                "output": file_path,
                "rows": len(merged_df),
                "optimized_rows": len(optimized_df),
                "merged_rows": len(merged_df),
                "recommendation_scenario_id": int(recommendation_scenario_id),
                "actionable_recommendation_rows": int(len(actionable_df)),
                "applied_recommendation_rows": int((applied_df["status"].astype(str) == "applied").sum()),
                "saved_site_prediction_rows": int(saved_site_rows),
                "changed_cells": int(changed_rows["Node_Cell_ID"].nunique()),
                "affected_cells": int(len(affected_cells)),
                "affected_sites": int(len(affected_sites)),
            })

            if scenario_row_id:
                self._update_scenario_status(int(scenario_row_id), "done", region=region, job_id=job_id)
            self._update(job_id, "done", "Completed")

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            if scenario_row_id:
                self._update_scenario_status(int(scenario_row_id), "failed", region=region, job_id=job_id)
            print(" ERROR:", traceback.format_exc())

    def _save_csv(self, df, project_id, operator):
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        file_path = os.path.join(
            output_dir,
            f"optimized_{operator}_{project_id}_{timestamp}.csv"
        )

        df.to_csv(file_path, index=False)
        print(f" CSV saved: {file_path}")

        return file_path
    
    def _save_to_db(self, df, region="india"):
        bridge = get_bridge_client()
        if bridge:
            safe_df = df.replace({pd.NA: None}).where(pd.notna(df), None)
            rows = []
            for row in safe_df.to_dict(orient="records"):
                rows.append(
                    {
                        "lat": row.get("lat"),
                        "lon": row.get("lon"),
                        "pred_rsrp": row.get("pred_rsrp"),
                        "pred_rsrq": row.get("pred_rsrq"),
                        "pred_sinr": row.get("pred_sinr"),
                        "node_b_id": row.get("node_b_id"),
                        "cell_id": row.get("cell_id"),
                        "Technology": row.get("Technology"),
                        "operator": row.get("Operator"),
                        "operator_name": row.get("Operator"),
                        "created_at": row.get("created_at").isoformat() if hasattr(row.get("created_at"), "isoformat") else row.get("created_at"),
                        "site_id": row.get("site_id"),
                        "nodeb_id_cell_id": row.get("nodeb_id_cell_id"),
                        "scenario_id": row.get("scenario_id"),
                        "public_scenario_id": row.get("public_scenario_id"),
                    }
                )
            payload = bridge._request(
                "POST",
                "SaveLtePredictionOptimisedResults",
                json={
                    "ProjectId": int(df["project_id"].iloc[0]),
                    "JobId": str(df["job_id"].iloc[0]) if "job_id" in df.columns else "",
                    "Rows": rows,
                },
            )
            print(f"[LTE_OPT][DB_WRITE_DONE] source=python_bridge rows={payload.get('Inserted') or payload.get('inserted') or 0}")
            return
        current_engine = _resolve_engine(region)
        self._ensure_public_scenario_id_column(current_engine)
        print(
            f"[LTE_OPT][DB_WRITE] table=lte_prediction_optimised_results "
            f"mode=append rows={len(df)} region={region}"
        )
        
        df.to_sql(
            "lte_prediction_optimised_results",
            con=current_engine,
            if_exists="append",
            index=False,
            chunksize=15000,
            method="multi"
        )
        print(" Data saved to DB")

    def _ensure_public_scenario_id_column(self, current_engine):
        check_sql = text("""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = DATABASE()
              AND table_name = 'lte_prediction_optimised_results'
              AND column_name = 'public_scenario_id'
        """)
        alter_sql = text("""
            ALTER TABLE lte_prediction_optimised_results
            ADD COLUMN public_scenario_id INT NULL AFTER scenario_id
        """)
        with current_engine.begin() as conn:
            exists = int(conn.execute(check_sql).scalar() or 0)
            if not exists:
                conn.execute(alter_sql)
                print("[LTE_OPT][SCHEMA] added lte_prediction_optimised_results.public_scenario_id")
    
    def _format_for_db(self, df, project_id, job_id, operator, scenario_id=None, public_scenario_id=None):
        import datetime

        df = df.copy()
        if "Node_Cell_ID" not in df.columns:
            if "nodeb_id_cell_id" in df.columns:
                df["Node_Cell_ID"] = df["nodeb_id_cell_id"].astype(str)
            else:
                raise ValueError("Missing Node_Cell_ID/nodeb_id_cell_id in optimized output")

        raw_node_cell = df["nodeb_id_cell_id"].astype(str) if "nodeb_id_cell_id" in df.columns else df["Node_Cell_ID"].astype(str)
        canonical = df["canonical_cell_id"].astype(str) if "canonical_cell_id" in df.columns else df["Node_Cell_ID"].map(_clean_id)
        split_cols = canonical.astype(str).str.split("_", expand=True)
        if split_cols.shape[1] < 2:
            raise ValueError("Invalid canonical cell identity format")

        df["node_b_id"] = df["node_b_id"].astype(str) if "node_b_id" in df.columns else split_cols[0].astype(str)
        df["cell_id"] = df["cell_id"].astype(str) if "cell_id" in df.columns else canonical
        df["nodeb_id_cell_id"] = raw_node_cell.astype(str)

        df["project_id"] = project_id
        df["job_id"] = job_id
        
        df["Technology"] = "4G"
        df["Operator"] = operator  
        df["created_at"] = datetime.datetime.now()
        if "site_id" in df.columns:
            site_id_series = df["site_id"].astype("string").str.strip()
            site_id_series = site_id_series.mask(site_id_series.isin(["", "nan", "NaN", "None", "<NA>"]))
        else:
            site_id_series = pd.Series(pd.NA, index=df.index, dtype="string")

        fallback_site_series = None
        for candidate in ["dashboard_site_id", "site", "nodeb_id_raw"]:
            if candidate in df.columns:
                fallback_site_series = df[candidate].astype("string").str.strip()
                fallback_site_series = fallback_site_series.mask(
                    fallback_site_series.isin(["", "nan", "NaN", "None", "<NA>"])
                )
                break

        if fallback_site_series is None:
            fallback_site_series = df["node_b_id"].astype("string").str.strip()

        df["site_id"] = site_id_series.fillna(fallback_site_series)
        df["scenario_id"] = scenario_id
        df["public_scenario_id"] = public_scenario_id

        final_df = df[[
            "project_id",
            "job_id",
            "lat",
            "lon",
            "pred_rsrp",
            "pred_rsrq",
            "pred_sinr",
            "node_b_id",
            "cell_id",
            "Technology",
            "created_at",
            "site_id",
            "nodeb_id_cell_id",
            "Operator",
            "scenario_id",
            "public_scenario_id",
        ]]

        return final_df

    def _update(self, job_id, status, msg):
        JOBS[job_id]["status"] = status
        JOBS[job_id]["progress"] = msg
        print(f"[{job_id[:6]}] {msg}", flush=True)

    def _get_next_project_scenario_id(self, project_id, current_engine):
        query = text("""
            SELECT scenario_id
            FROM lte_optimization_scenarios
            WHERE project_id = :project_id
              AND scenario_id IS NOT NULL
            ORDER BY scenario_id ASC
        """)
        with current_engine.connect() as conn:
            used_ids = {
                int(row[0])
                for row in conn.execute(query, {"project_id": int(project_id)}).fetchall()
                if row[0] is not None
            }
        for scenario_id in range(1, 7):
            if scenario_id not in used_ids:
                return scenario_id
        raise ValueError(
            f"No available public scenario slot for project_id={int(project_id)}. "
            "Scenario pruning did not free a 1..6 slot."
        )

    def _prune_oldest_project_scenario_if_needed(self, project_id, current_engine, max_scenarios=6):
        count_query = text("""
            SELECT COUNT(*)
            FROM lte_optimization_scenarios
            WHERE project_id = :project_id
        """)
        oldest_query = text("""
            SELECT id, scenario_id
            FROM lte_optimization_scenarios
            WHERE project_id = :project_id
            ORDER BY COALESCE(created_at, updated_at, '1970-01-01') ASC, id ASC
            LIMIT 1
        """)
        delete_results_query = text("""
            DELETE FROM lte_prediction_optimised_results
            WHERE scenario_id = :scenario_row_id
        """)
        delete_scenario_query = text("""
            DELETE FROM lte_optimization_scenarios
            WHERE id = :scenario_row_id
        """)
        with current_engine.begin() as conn:
            pruned = []
            while True:
                scenario_count = int(conn.execute(count_query, {"project_id": int(project_id)}).scalar() or 0)
                if scenario_count < int(max_scenarios):
                    break
                oldest = conn.execute(oldest_query, {"project_id": int(project_id)}).fetchone()
                if not oldest:
                    break
                oldest_row_id = int(oldest[0])
                oldest_public_scenario_id = int(oldest[1]) if oldest[1] is not None else None
                results_deleted = conn.execute(
                    delete_results_query,
                    {"scenario_row_id": oldest_row_id},
                ).rowcount
                conn.execute(delete_scenario_query, {"scenario_row_id": oldest_row_id})
                pruned.append(
                    {
                        "row_id": oldest_row_id,
                        "scenario_id": oldest_public_scenario_id,
                        "result_rows": int(results_deleted or 0),
                    }
                )
        for item in pruned:
            print(
                f"[LTE_OPT][SCENARIO_PRUNE] project_id={int(project_id)} "
                f"deleted_row_id={item['row_id']} deleted_public_scenario_id={item['scenario_id']} "
                f"deleted_result_rows={item['result_rows']} max_scenarios={int(max_scenarios)}"
            )
        return pruned

    def _create_scenario(self, cfg, job_id, region):
        bridge = get_bridge_client()
        if bridge:
            baseline_job_id = cfg.get("baseline_job_id") or _latest_baseline_job_id(
                cfg["project_id"], region=region, operator=cfg.get("operator")
            )
            cfg["baseline_job_id"] = baseline_job_id
            payload = {
                "ProjectId": int(cfg["project_id"]),
                "ScenarioId": int(cfg["requested_public_scenario_id"]) if cfg.get("requested_public_scenario_id") else None,
                "BaselineJobId": baseline_job_id,
                "ScenarioName": _build_scenario_name(cfg),
                "ScenarioDescription": _build_scenario_description(cfg),
                "Region": region,
                "Operator": cfg.get("operator"),
                "TargetType": str(cfg.get("target_type", "")).strip() or None,
                "TargetId": str(cfg.get("target_id", "")).strip() or None,
                "ImpactRadiusM": float(cfg.get("impact_radius_m", cfg.get("radius", 500)) or 500),
                "NeighborSiteCount": int(cfg.get("neighbor_site_count", 2) or 2),
                "MaxInterferenceSites": int(cfg.get("max_interference_sites", 10) or 10),
                "DeltaLat": float(cfg.get("delta_lat", 0) or 0),
                "DeltaLon": float(cfg.get("delta_lon", 0) or 0),
                "DeltaAzimuth": float(cfg.get("delta_azimuth", 0) or 0),
                "DeltaElectricalTilt": float(cfg.get("delta_electrical_tilt", 0) or 0),
                "DeltaMechanicalTilt": float(cfg.get("delta_mechanical_tilt", 0) or 0),
                "DeltaTxPower": float(cfg.get("delta_tx_power", 0) or 0),
                "DeltaAntennaHeight": float(cfg.get("delta_antenna_height", 0) or 0),
                "Status": "created",
                "CreatedBy": str(cfg.get("created_by", "backend")),
            }
            result = bridge._request("POST", "CreateLteOptimizationScenario", json=payload)
            scenario_row_id = int(result.get("ScenarioRowId") or result.get("scenarioRowId"))
            public_scenario_id = int(result.get("ScenarioId") or result.get("scenarioId"))
            print(f"[LTE_OPT][SCENARIO_CREATE] source=python_bridge row_id={scenario_row_id} scenario_id={public_scenario_id} job_id={job_id}")
            return scenario_row_id, public_scenario_id
        current_engine = _resolve_engine(region)
        baseline_job_id = cfg.get("baseline_job_id") or _latest_baseline_job_id(
            cfg["project_id"], region=region, operator=cfg.get("operator")
        )
        cfg["baseline_job_id"] = baseline_job_id
        self._prune_oldest_project_scenario_if_needed(cfg["project_id"], current_engine, max_scenarios=6)
        requested_public_scenario_id = cfg.get("requested_public_scenario_id")
        if requested_public_scenario_id is not None:
            requested_public_scenario_id = int(requested_public_scenario_id)
            if requested_public_scenario_id <= 0:
                requested_public_scenario_id = None
        public_scenario_id = requested_public_scenario_id or self._get_next_project_scenario_id(cfg["project_id"], current_engine)
        payload = {
            "project_id": int(cfg["project_id"]),
            "scenario_id": public_scenario_id,
            "baseline_job_id": baseline_job_id,
            "scenario_name": _build_scenario_name(cfg),
            "scenario_description": _build_scenario_description(cfg),
            "region": region,
            "operator": cfg.get("operator"),
            "target_type": str(cfg.get("target_type", "")).strip() or None,
            "target_id": str(cfg.get("target_id", "")).strip() or None,
            "impact_radius_m": float(cfg.get("impact_radius_m", cfg.get("radius", 500)) or 500),
            "neighbor_site_count": int(cfg.get("neighbor_site_count", 2) or 2),
            "max_interference_sites": int(cfg.get("max_interference_sites", 10) or 10),
            "delta_lat": float(cfg.get("delta_lat", 0) or 0),
            "delta_lon": float(cfg.get("delta_lon", 0) or 0),
            "delta_azimuth": float(cfg.get("delta_azimuth", 0) or 0),
            "delta_electrical_tilt": float(cfg.get("delta_electrical_tilt", 0) or 0),
            "delta_mechanical_tilt": float(cfg.get("delta_mechanical_tilt", 0) or 0),
            "delta_tx_power": float(cfg.get("delta_tx_power", 0) or 0),
            "delta_antenna_height": float(cfg.get("delta_antenna_height", 0) or 0),
            "status": "created",
            "created_by": str(cfg.get("created_by", "backend")),
        }
        insert_sql = text("""
            INSERT INTO lte_optimization_scenarios (
                project_id, scenario_id, baseline_job_id, scenario_name, scenario_description,
                region, operator, target_type, target_id, impact_radius_m,
                neighbor_site_count, max_interference_sites, delta_lat, delta_lon,
                delta_azimuth, delta_electrical_tilt, delta_mechanical_tilt,
                delta_tx_power, delta_antenna_height, status, created_by
            ) VALUES (
                :project_id, :scenario_id, :baseline_job_id, :scenario_name, :scenario_description,
                :region, :operator, :target_type, :target_id, :impact_radius_m,
                :neighbor_site_count, :max_interference_sites, :delta_lat, :delta_lon,
                :delta_azimuth, :delta_electrical_tilt, :delta_mechanical_tilt,
                :delta_tx_power, :delta_antenna_height, :status, :created_by
            )
        """)
        with current_engine.begin() as conn:
            result = conn.execute(insert_sql, payload)
            scenario_row_id = int(result.lastrowid)
        print(
            f"[LTE_OPT][SCENARIO_CREATE] row_id={scenario_row_id} scenario_id={public_scenario_id} job_id={job_id} "
            f"name={payload['scenario_name']!r} description={payload['scenario_description']!r}"
        )
        return scenario_row_id, public_scenario_id

    def _update_scenario_status(self, scenario_row_id, status, region="india", job_id=None):
        bridge = get_bridge_client()
        if bridge:
            baseline_job_id = None
            try:
                baseline_job_id = _latest_baseline_job_id(
                    JOBS.get(job_id, {}).get("project_id"),
                    region,
                    operator=JOBS.get(job_id, {}).get("operator"),
                ) if job_id and JOBS.get(job_id, {}).get("project_id") else None
            except Exception:
                baseline_job_id = None
            bridge._request(
                "POST",
                "UpdateLteOptimizationScenarioStatus",
                json={"ScenarioRowId": int(scenario_row_id), "Status": status, "BaselineJobId": baseline_job_id},
            )
            print(f"[LTE_OPT][SCENARIO_STATUS] source=python_bridge row_id={scenario_row_id} status={status}")
            return
        current_engine = _resolve_engine(region)
        update_sql = text("""
            UPDATE lte_optimization_scenarios
            SET status = :status,
                baseline_job_id = COALESCE(baseline_job_id, :baseline_job_id),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :scenario_row_id
        """)
        baseline_job_id = None
        try:
            baseline_job_id = _latest_baseline_job_id(
                JOBS.get(job_id, {}).get("project_id"),
                region,
                operator=JOBS.get(job_id, {}).get("operator"),
            ) if job_id and JOBS.get(job_id, {}).get("project_id") else None
        except Exception:
            baseline_job_id = None
        with current_engine.begin() as conn:
            conn.execute(
                update_sql,
                {
                    "status": status,
                    "baseline_job_id": baseline_job_id,
                    "scenario_row_id": int(scenario_row_id),
                },
            )
        print(f"[LTE_OPT][SCENARIO_STATUS] row_id={scenario_row_id} status={status}")
