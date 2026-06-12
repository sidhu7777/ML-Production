from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
load_dotenv(ML_ROOT / ".env")

from tools.lte_tilt_recommandation.recommendation_engine import (  # noqa: E402
    TiltEngineConfig,
    run_recommendation_engine,
)
from tools.lte_tilt_recommandation.services import (  # noqa: E402
    _apply_constraint_ranges,
    _azimuth_in_range,
    _bridge_fetch_baseline_log_df,
    _bridge_fetch_grid_analytics_df,
    _clean_id_series,
    _fetch_baseline_log_df,
    _fetch_grid_analytics_df,
    _load_constraint_df,
    _prepare_tilt_antenna_df,
    _prepare_tilt_log_df,
    _resolve_threshold_file_path,
    _rf_identity_series,
    _to_threshold_cell_id,
    engine as REGION_ENGINES,
)
from utils.python_bridge import get_bridge_client  # noqa: E402


DEFAULT_PAYLOAD: dict[str, Any] = {
    "project_id": 196,
    "region": "india",
    "operator": "Airtel",
    "rsrp": -90,
    "rsrq": -14,
    "sinr": 0,
    "rsrp_weight": 20,
    "rsrq_weight": 20,
    "sinr_weight": 60,
    "validate_candidates": True,
    "radius_m": 500,
    "grid_resolution_m": 25,
    "n_workers": 3,
    "impact_radius_m": 500,
    "neighbor_site_count": 2,
    "max_interference_sites": 10,
    "candidate_workers": 2,
    "coordinate_passes": 2,
    "bad_grid_coverage_pct": 60,
    "max_group_cells": 0,
    "max_neighbors_per_update_cell": 2,
}


GEO_QUERY = text(
    """
    SELECT
        lat,
        lon,
        nodeb_id_cell_id,
        clutter_class,
        morphology_cluster,
        building_count,
        building_area_ratio,
        avg_building_area_m2,
        road_length_m,
        green_ratio,
        water_ratio,
        los_blocker_count,
        los_blocked_ratio,
        max_blocker_height_m,
        diffraction_proxy_db,
        nlos_flag,
        terrain_elevation_m,
        terrain_slope_deg,
        proxy_site_elevation_m,
        terrain_relief_to_site_m,
        site_count_250m,
        site_count_500m,
        serving_distance_m,
        nearest_site_distance_m,
        mean_nearest3_site_distance_m,
        azimuth_delta_deg
    FROM lte_prediction_geo_features
    WHERE project_id = :pid
      AND region = :region
    """
)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text_value = str(value).strip().lower()
    if text_value in {"1", "true", "yes", "y", "on"}:
        return True
    if text_value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    return str(value)


def _safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    for key in ["database_url", "database_url_taiwan", "bridge_api_key"]:
        if redacted.get(key):
            redacted[key] = "<provided>"
    return redacted


def _safe_to_csv(df: pd.DataFrame, path: Path, limit: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df if limit is None else df.head(limit)
    out.to_csv(path, index=False)


def _load_payload(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("--payload-json must contain one JSON object")
    return payload


def _effective_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(DEFAULT_PAYLOAD)
    payload.update(_load_payload(args.payload_json))
    for key in DEFAULT_PAYLOAD:
        attr = key.replace("-", "_")
        value = getattr(args, attr, None)
        if value is not None:
            payload[key] = value
    if args.threshold_file_path:
        payload["threshold_file_path"] = args.threshold_file_path
    if args.mode:
        payload["mode"] = args.mode
    for key in ["database_url", "database_url_taiwan", "bridge_base_url", "bridge_api_key"]:
        value = getattr(args, key, None)
        if value:
            payload[key] = value
    return payload


def _is_all_operator(operator: Any) -> bool:
    return not operator or str(operator).strip().lower() in {"all", "none", ""}


def _numeric_range(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in df.columns:
        return {"present": False}
    values = pd.to_numeric(df[col], errors="coerce")
    non_null = values.dropna()
    if non_null.empty:
        return {"present": True, "non_null": 0}
    return {
        "present": True,
        "non_null": int(non_null.size),
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "mean": float(non_null.mean()),
    }


def _top_counts(df: pd.DataFrame, col: str, limit: int = 10) -> dict[str, int]:
    if col not in df.columns:
        return {}
    counts = df[col].astype(str).str.strip().value_counts(dropna=False).head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def _fingerprint(df: pd.DataFrame, candidate_cols: list[str], max_rows: int = 100000) -> str:
    cols = [col for col in candidate_cols if col in df.columns]
    if not cols or df.empty:
        return "empty"
    work = df[cols].copy()
    for col in cols:
        work[col] = work[col].astype(str).fillna("")
    work = work.sort_values(cols).head(max_rows)
    hashed = pd.util.hash_pandas_object(work, index=False)
    return f"{len(df)}:{int(hashed.sum()) & 0xFFFFFFFFFFFFFFFF:016x}"


def _bad_point_summary(df: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0}
    weights_raw = {
        "rsrp": max(float(payload.get("rsrp_weight", 0) or 0), 0.0),
        "rsrq": max(float(payload.get("rsrq_weight", 0) or 0), 0.0),
        "sinr": max(float(payload.get("sinr_weight", 0) or 0), 0.0),
    }
    total = sum(weights_raw.values()) or 1.0
    weights = {key: value / total for key, value in weights_raw.items()}
    thresholds = {
        "rsrp": float(payload.get("rsrp", -105)),
        "rsrq": float(payload.get("rsrq", -15)),
        "sinr": float(payload.get("sinr", 0)),
    }
    out: dict[str, Any] = {"rows": int(len(df)), "weights": weights, "thresholds": thresholds}
    combined = pd.Series(0.0, index=df.index)
    for kpi, col in [("rsrp", "pred_rsrp"), ("rsrq", "pred_rsrq"), ("sinr", "pred_sinr")]:
        if col not in df.columns:
            out[f"{kpi}_missing"] = True
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values < thresholds[kpi]
        severity = (thresholds[kpi] - values).clip(lower=0).fillna(0)
        combined = combined + severity * weights[kpi]
        out[f"bad_{kpi}_rows"] = int(bad.fillna(False).sum())
    out["bad_combined_rows"] = int((combined > 0).sum())
    return out


def _frame_summary(name: str, df: pd.DataFrame) -> dict[str, Any]:
    identity_cols = [
        "Node_Cell_ID",
        "nodeb_id_cell_id",
        "frontend_site_sector_key",
        "cell_id",
        "nodeb_id",
        "grid_id",
        "operator",
    ]
    numeric_cols = [
        "lat",
        "lon",
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "baseline_avg_rsrp",
        "baseline_avg_rsrq",
        "baseline_avg_sinr",
        "electrical_tilt",
        "mechanical_tilt",
        "azimuth",
    ]
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "column_count": int(len(df.columns)),
        "operators": _top_counts(df, "operator"),
        "technology": _top_counts(df, "Technology") or _top_counts(df, "technology"),
        "identity_counts": {
            col: int(df[col].nunique(dropna=True)) for col in identity_cols if col in df.columns
        },
        "numeric": {col: _numeric_range(df, col) for col in numeric_cols if col in df.columns},
        "fingerprint": _fingerprint(df, identity_cols + numeric_cols),
    }


def _fetch_geo_direct(current_engine, project_id: int, region: str) -> pd.DataFrame:
    with current_engine.connect() as conn:
        geo_df = pd.read_sql(GEO_QUERY, conn, params={"pid": int(project_id), "region": region})
    if "nodeb_id_cell_id" in geo_df.columns:
        geo_df["Node_Cell_ID"] = geo_df["nodeb_id_cell_id"].astype(str).str.strip()
    return geo_df


def _fetch_source(source: str, payload: dict[str, Any]) -> dict[str, Any]:
    project_id = int(payload["project_id"])
    region = str(payload.get("region", "india")).lower()
    operator_input = payload.get("operator")
    is_all_operators = _is_all_operator(operator_input)

    if source == "direct":
        current_engine = REGION_ENGINES.get(region)
        if current_engine is None:
            db_url = (
                payload.get("database_url_taiwan")
                if region == "taiwan"
                else payload.get("database_url")
            )
            db_url = db_url or os.getenv("DATABASE_URL_Taiwan" if region == "taiwan" else "DATABASE_URL")
            if db_url:
                current_engine = create_engine(
                    db_url,
                    pool_size=5,
                    max_overflow=10,
                    pool_recycle=300,
                    pool_pre_ping=True,
                )
        if current_engine is None:
            raise RuntimeError(f"No direct DB engine configured for region={region}")
        with current_engine.connect() as conn:
            antenna_df = pd.read_sql(
                text("SELECT * FROM site_prediction WHERE tbl_project_id = :pid"),
                conn,
                params={"pid": project_id},
            )
        antenna_df = _prepare_tilt_antenna_df(antenna_df)
        log_df, baseline_job_id = _fetch_baseline_log_df(
            current_engine,
            project_id,
            operator_input,
            is_all_operators,
        )
        log_df = _prepare_tilt_log_df(log_df, antenna_df)
        geo_df = _fetch_geo_direct(current_engine, project_id, region)
        grid_df = _fetch_grid_analytics_df(current_engine, project_id, operator_input, is_all_operators)
        scenario_id = None
    elif source == "bridge":
        if payload.get("bridge_base_url"):
            os.environ["PYTHON_BRIDGE_BASE_URL"] = str(payload["bridge_base_url"])
        if payload.get("bridge_api_key"):
            os.environ["PYTHON_BRIDGE_API_KEY"] = str(payload["bridge_api_key"])
        bridge = get_bridge_client()
        if bridge is None:
            raise RuntimeError("Bridge source requested, but PYTHON_BRIDGE_BASE_URL is not configured")
        antenna_df = bridge.get_rows(
            "GetLteTiltAntennaRows",
            {"projectId": project_id},
            limit=50000,
            progress_label="debug_tilt_antenna",
        )
        antenna_df = _prepare_tilt_antenna_df(antenna_df)
        log_df, baseline_job_id = _bridge_fetch_baseline_log_df(
            bridge,
            project_id,
            region,
            operator_input,
            is_all_operators,
        )
        log_df = _prepare_tilt_log_df(log_df, antenna_df)
        geo_df = bridge.get_rows(
            "GetLtePredictionGeoFeatures",
            {"projectId": project_id, "region": region},
            limit=50000,
            progress_label="debug_geo_features",
        )
        if "nodeb_id_cell_id" in geo_df.columns:
            geo_df["Node_Cell_ID"] = geo_df["nodeb_id_cell_id"].astype(str).str.strip()
        grid_df = _bridge_fetch_grid_analytics_df(bridge, project_id, operator_input, is_all_operators)
        scenario_id = None
    else:
        raise ValueError(f"Unknown source={source}")

    return {
        "source": source,
        "baseline_job_id": baseline_job_id,
        "scenario_id": scenario_id,
        "antenna_df": antenna_df,
        "baseline_df": log_df,
        "geo_df": geo_df,
        "grid_df": grid_df,
    }


def _build_engine_config(
    payload: dict[str, Any],
    baseline_job_id: str,
    constraint_df: pd.DataFrame,
    run_dir: Path,
    source: str,
) -> TiltEngineConfig:
    operator = payload.get("operator")
    is_all_operators = _is_all_operator(operator)
    return TiltEngineConfig(
        project_id=int(payload["project_id"]),
        region=str(payload.get("region", "india")).lower(),
        operator=None if is_all_operators else operator,
        mode=payload.get("mode") or payload.get("kpi_mode") or payload.get("recommendation_mode") or "combined_weighted",
        rsrp_threshold=float(payload.get("rsrp", -105)),
        rsrq_threshold=float(payload.get("rsrq", -15)),
        sinr_threshold=float(payload.get("sinr", 0)),
        rsrp_weight=float(payload.get("rsrp_weight", 34.0)),
        rsrq_weight=float(payload.get("rsrq_weight", 33.0)),
        sinr_weight=float(payload.get("sinr_weight", 33.0)),
        validate_candidates=_parse_bool(payload.get("validate_candidates", True)),
        max_validation_candidates=int(payload.get("max_validation_candidates", payload.get("max_candidates", 25))),
        radius_m=float(payload.get("radius_m", payload.get("radius", 500.0))),
        grid_resolution_m=float(payload.get("grid_resolution_m", payload.get("grid_resolution", 30.0))),
        workers=int(payload.get("n_workers", payload.get("workers", 1))),
        impact_radius_m=float(payload.get("impact_radius_m", payload.get("radius_m", payload.get("radius", 500.0)))),
        neighbor_site_count=int(payload.get("neighbor_site_count", 3)),
        max_interference_sites=int(payload.get("max_interference_sites", 10)),
        baseline_job_id=baseline_job_id,
        coordinate_passes=int(payload.get("coordinate_passes", 2)),
        candidate_workers=int(payload.get("candidate_workers", 1)),
        bad_grid_coverage_pct=float(payload.get("bad_grid_coverage_pct", 80.0)),
        max_group_cells=int(payload.get("max_group_cells", 0)),
        max_neighbors_per_update_cell=int(payload.get("max_neighbors_per_update_cell", 2)),
        rf_debug_log_path=str(run_dir / source / "tilt_rf_debug.log"),
        constraint_map=constraint_df.set_index("cell_id").to_dict("index") if not constraint_df.empty else {},
    )


def _engine_config_summary(config: TiltEngineConfig) -> dict[str, Any]:
    return {
        "project_id": config.project_id,
        "region": config.region,
        "operator": config.operator or "all",
        "mode": config.mode,
        "thresholds": {
            "rsrp": config.rsrp_threshold,
            "rsrq": config.rsrq_threshold,
            "sinr": config.sinr_threshold,
        },
        "weights": {
            "rsrp": config.rsrp_weight,
            "rsrq": config.rsrq_weight,
            "sinr": config.sinr_weight,
        },
        "validate_candidates": config.validate_candidates,
        "radius_m": config.radius_m,
        "grid_resolution_m": config.grid_resolution_m,
        "workers": config.workers,
        "impact_radius_m": config.impact_radius_m,
        "neighbor_site_count": config.neighbor_site_count,
        "max_interference_sites": config.max_interference_sites,
        "candidate_workers": config.candidate_workers,
        "coordinate_passes": config.coordinate_passes,
        "bad_grid_coverage_pct": config.bad_grid_coverage_pct,
        "max_group_cells": config.max_group_cells,
        "max_neighbors_per_update_cell": config.max_neighbors_per_update_cell,
        "baseline_job_id": config.baseline_job_id,
    }


def _save_engine_outputs(outputs: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Any]:
    shape_summary = {}
    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            shape_summary[name] = {"rows": int(len(df)), "columns": list(df.columns)}
            if not df.empty:
                _safe_to_csv(df, out_dir / f"{name}.csv")
        else:
            shape_summary[name] = {"type": type(df).__name__}
    return shape_summary


def _constraint_columns_for_param(param: Any) -> tuple[str | None, str | None, bool]:
    param_text = str(param or "").strip().lower()
    if param_text == "etilt":
        return "min_e_tilt", "max_e_tilt", False
    if param_text == "azimuth":
        return "min_azimuth", "max_azimuth", True
    if param_text in {"tx power", "power"}:
        return "min_tx_power", "max_tx_power", False
    if param_text in {"mechanical tilt", "mtilt"}:
        return "min_m_tilt", "max_m_tilt", False
    if param_text in {"height", "antenna height"}:
        return "min_height", "max_height", False
    return None, None, False


def _recommendation_constraint_validation(reco_df: pd.DataFrame, constraint_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if reco_df.empty:
        return pd.DataFrame(rows)

    constraint_map = (
        constraint_df.set_index("cell_id").to_dict("index")
        if isinstance(constraint_df, pd.DataFrame) and not constraint_df.empty and "cell_id" in constraint_df.columns
        else {}
    )

    for _, row in reco_df.iterrows():
        cell_id = str(row.get("Cell ID", "") or "").strip()
        threshold_cell_id = _to_threshold_cell_id(cell_id)
        param = str(row.get("Parameter", "") or "").strip()
        recommended_value = pd.to_numeric(pd.Series([row.get("Recommended Value")]), errors="coerce").iloc[0]
        cfg = constraint_map.get(threshold_cell_id)
        min_col, max_col, is_azimuth = _constraint_columns_for_param(param)

        validation = {
            "Cell ID": cell_id,
            "Threshold Cell ID": threshold_cell_id,
            "Parameter": param,
            "Current Value": row.get("Current Value"),
            "Recommended Value": row.get("Recommended Value"),
            "Constraint Applied": row.get("Constraint Applied"),
            "Constraint Source": row.get("Constraint Source"),
            "Allowed Min": row.get("Allowed Min"),
            "Allowed Max": row.get("Allowed Max"),
            "Optimised Constraint Row": False,
            "Inside Allowed Range": pd.NA,
            "Validation Reason": "",
        }

        if not cfg:
            validation["Validation Reason"] = "no_constraint_row_for_cell"
            rows.append(validation)
            continue
        if not bool(cfg.get("optimised")):
            validation["Validation Reason"] = "constraint_row_not_optimised"
            rows.append(validation)
            continue
        validation["Optimised Constraint Row"] = True
        if not min_col or not max_col:
            validation["Validation Reason"] = "unsupported_recommendation_parameter"
            rows.append(validation)
            continue

        min_allowed = cfg.get(min_col)
        max_allowed = cfg.get(max_col)
        validation["Allowed Min"] = min_allowed
        validation["Allowed Max"] = max_allowed
        if pd.isna(min_allowed) or pd.isna(max_allowed):
            validation["Validation Reason"] = "missing_allowed_range"
            rows.append(validation)
            continue
        if pd.isna(recommended_value):
            validation["Validation Reason"] = "recommended_value_not_numeric"
            rows.append(validation)
            continue

        if is_azimuth:
            inside = _azimuth_in_range(float(recommended_value), float(min_allowed), float(max_allowed))
        else:
            inside = float(min_allowed) <= float(recommended_value) <= float(max_allowed)
        validation["Inside Allowed Range"] = bool(inside)
        validation["Validation Reason"] = "inside_range" if inside else "outside_range"
        rows.append(validation)

    return pd.DataFrame(rows)


def _run_one_source(source: str, payload: dict[str, Any], run_dir: Path, skip_engine: bool) -> dict[str, Any]:
    print(f"\n[DEBUG][{source}] fetching input data")
    source_dir = run_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)

    fetched = _fetch_source(source, payload)
    for frame_name in ["antenna_df", "baseline_df", "geo_df", "grid_df"]:
        _safe_to_csv(fetched[frame_name], source_dir / f"{frame_name}_sample.csv", limit=200)

    root_dir = ML_ROOT
    threshold_file_path = _resolve_threshold_file_path(
        payload,
        int(payload["project_id"]),
        str(root_dir),
    )
    constraint_df = _load_constraint_df(threshold_file_path) if threshold_file_path else pd.DataFrame()
    if not constraint_df.empty:
        _safe_to_csv(constraint_df, source_dir / "constraint_file_normalized.csv")

    config = _build_engine_config(
        payload,
        fetched["baseline_job_id"],
        constraint_df,
        run_dir,
        source,
    )

    summary: dict[str, Any] = {
        "source": source,
        "fetch_ok": True,
        "baseline_job_id": fetched["baseline_job_id"],
        "engine_config": _engine_config_summary(config),
        "constraint_file_path": threshold_file_path or None,
        "constraint_rows": int(len(constraint_df)),
        "constraint_optimised_rows": int(constraint_df["optimised"].sum())
        if "optimised" in constraint_df.columns
        else None,
        "frames": {
            "antenna": _frame_summary("antenna", fetched["antenna_df"]),
            "baseline": _frame_summary("baseline", fetched["baseline_df"]),
            "geo": _frame_summary("geo", fetched["geo_df"]),
            "grid": _frame_summary("grid", fetched["grid_df"]),
        },
        "bad_point_summary": _bad_point_summary(fetched["baseline_df"], payload),
    }

    print(
        f"[DEBUG][{source}] baseline_job_id={summary['baseline_job_id']} "
        f"antenna_rows={summary['frames']['antenna']['rows']} "
        f"baseline_rows={summary['frames']['baseline']['rows']} "
        f"geo_rows={summary['frames']['geo']['rows']} "
        f"grid_rows={summary['frames']['grid']['rows']} "
        f"constraint_rows={summary['constraint_rows']}"
    )
    print(f"[DEBUG][{source}] bad_point_summary={summary['bad_point_summary']}")

    if skip_engine:
        _safe_write_json(source_dir / "summary.json", summary)
        return summary

    print(f"[DEBUG][{source}] running production recommendation engine without DB write")
    outputs = run_recommendation_engine(
        log_df=fetched["baseline_df"],
        antenna_df=fetched["antenna_df"],
        geo_df=fetched["geo_df"],
        grid_analytics_df=fetched["grid_df"],
        config=config,
    )
    raw_reco_df = outputs.get("recommendations", pd.DataFrame()).copy()
    constrained_reco_df = _apply_constraint_ranges(raw_reco_df, constraint_df)
    if isinstance(constrained_reco_df, pd.DataFrame):
        outputs["recommendations_after_constraints"] = constrained_reco_df
        validation_df = _recommendation_constraint_validation(constrained_reco_df, constraint_df)
        outputs["recommendation_constraint_validation"] = validation_df

    summary["engine_outputs"] = _save_engine_outputs(outputs, source_dir)
    summary["raw_recommendation_rows"] = int(len(raw_reco_df))
    summary["constrained_recommendation_rows"] = int(len(constrained_reco_df))
    summary["candidate_evaluation_rows"] = int(len(outputs.get("candidate_evaluations", pd.DataFrame())))
    summary["best_candidate_metric_rows"] = int(len(outputs.get("best_candidate_metrics", pd.DataFrame())))
    validation_df = outputs.get("recommendation_constraint_validation", pd.DataFrame())
    if isinstance(validation_df, pd.DataFrame):
        out_of_range = (
            validation_df["Inside Allowed Range"].eq(False)
            if "Inside Allowed Range" in validation_df.columns
            else pd.Series(dtype=bool)
        )
        summary["constraint_validation"] = {
            "rows": int(len(validation_df)),
            "inside_range_rows": int(validation_df["Inside Allowed Range"].eq(True).sum())
            if "Inside Allowed Range" in validation_df.columns
            else 0,
            "outside_range_rows": int(out_of_range.sum()),
            "missing_or_not_optimised_rows": int(validation_df["Inside Allowed Range"].isna().sum())
            if "Inside Allowed Range" in validation_df.columns
            else 0,
        }

    print(
        f"[DEBUG][{source}] raw_recommendations={summary['raw_recommendation_rows']} "
        f"after_constraints={summary['constrained_recommendation_rows']} "
        f"candidate_evaluations={summary['candidate_evaluation_rows']} "
        f"best_candidate_metrics={summary['best_candidate_metric_rows']}"
    )

    _safe_write_json(source_dir / "summary.json", summary)
    return summary


def _compare_summaries(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "direct" not in results or "bridge" not in results:
        return {"available_sources": list(results)}

    direct = results["direct"]
    bridge = results["bridge"]
    comparisons = []

    def add(name: str, direct_value: Any, bridge_value: Any) -> None:
        comparisons.append(
            {
                "name": name,
                "direct": direct_value,
                "bridge": bridge_value,
                "same": direct_value == bridge_value,
            }
        )

    add("baseline_job_id", direct.get("baseline_job_id"), bridge.get("baseline_job_id"))
    add("constraint_file_path", direct.get("constraint_file_path"), bridge.get("constraint_file_path"))
    add("constraint_rows", direct.get("constraint_rows"), bridge.get("constraint_rows"))
    add("constraint_optimised_rows", direct.get("constraint_optimised_rows"), bridge.get("constraint_optimised_rows"))
    for frame in ["antenna", "baseline", "geo", "grid"]:
        add(f"{frame}_rows", direct["frames"][frame]["rows"], bridge["frames"][frame]["rows"])
        add(f"{frame}_fingerprint", direct["frames"][frame]["fingerprint"], bridge["frames"][frame]["fingerprint"])
    for key in [
        "raw_recommendation_rows",
        "constrained_recommendation_rows",
        "candidate_evaluation_rows",
        "best_candidate_metric_rows",
    ]:
        if key in direct or key in bridge:
            add(key, direct.get(key), bridge.get(key))

    diagnosis = []
    first_diff = next((item for item in comparisons if not item["same"]), None)
    if first_diff:
        diagnosis.append(f"First direct-vs-bridge mismatch: {first_diff['name']}")
    if direct.get("raw_recommendation_rows", 0) > 0 and bridge.get("raw_recommendation_rows", 0) == 0:
        diagnosis.append("Direct creates recommendations but bridge does not: inspect bridge data/baseline/grid differences above.")
    if bridge.get("raw_recommendation_rows", 0) > 0 and bridge.get("constrained_recommendation_rows", 0) == 0:
        diagnosis.append("Bridge recommendations are removed after constraints: threshold file is the cause.")
    if bridge.get("raw_recommendation_rows", 0) == 0 and bridge.get("candidate_evaluation_rows", 0) == 0:
        diagnosis.append("Bridge has no candidate evaluations: target selection or required input rows are missing/filtered.")
    if not diagnosis:
        diagnosis.append("No obvious direct-vs-bridge mismatch detected in summarized stages.")

    return {"comparisons": comparisons, "diagnosis": diagnosis}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug LTE tilt recommendation input and engine differences between direct DB and Python bridge.",
    )
    parser.add_argument("--source", choices=["direct", "bridge", "both"], default="both")
    parser.add_argument("--payload-json", help="Optional JSON file copied from frontend console payload.")
    parser.add_argument("--output-root", default=str(ML_ROOT / "outputs" / "debug"))
    parser.add_argument("--skip-engine", action="store_true", help="Only fetch/summarize inputs; do not run recommendation engine.")

    parser.add_argument("--project-id", dest="project_id", type=int)
    parser.add_argument("--region")
    parser.add_argument("--operator")
    parser.add_argument("--mode")
    parser.add_argument("--rsrp", type=float)
    parser.add_argument("--rsrq", type=float)
    parser.add_argument("--sinr", type=float)
    parser.add_argument("--rsrp-weight", dest="rsrp_weight", type=float)
    parser.add_argument("--rsrq-weight", dest="rsrq_weight", type=float)
    parser.add_argument("--sinr-weight", dest="sinr_weight", type=float)
    parser.add_argument("--validate-candidates", dest="validate_candidates", type=_parse_bool)
    parser.add_argument("--radius-m", dest="radius_m", type=float)
    parser.add_argument("--grid-resolution-m", dest="grid_resolution_m", type=float)
    parser.add_argument("--n-workers", dest="n_workers", type=int)
    parser.add_argument("--impact-radius-m", dest="impact_radius_m", type=float)
    parser.add_argument("--neighbor-site-count", dest="neighbor_site_count", type=int)
    parser.add_argument("--max-interference-sites", dest="max_interference_sites", type=int)
    parser.add_argument("--candidate-workers", dest="candidate_workers", type=int)
    parser.add_argument("--coordinate-passes", dest="coordinate_passes", type=int)
    parser.add_argument("--bad-grid-coverage-pct", dest="bad_grid_coverage_pct", type=float)
    parser.add_argument("--max-group-cells", dest="max_group_cells", type=int)
    parser.add_argument("--max-neighbors-per-update-cell", dest="max_neighbors_per_update_cell", type=int)
    parser.add_argument("--threshold-file-path", dest="threshold_file_path")
    parser.add_argument("--database-url", dest="database_url", help="Optional direct DB URL for india.")
    parser.add_argument("--database-url-taiwan", dest="database_url_taiwan", help="Optional direct DB URL for taiwan.")
    parser.add_argument("--bridge-base-url", dest="bridge_base_url", help="Optional C# API base URL, e.g. http://localhost:5224.")
    parser.add_argument("--bridge-api-key", dest="bridge_api_key", help="Optional PythonBridge API key.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = _effective_payload(args)
    run_dir = Path(args.output_root) / f"lte_tilt_recommendation_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    redacted_payload = _redact_payload(payload)
    _safe_write_json(run_dir / "effective_payload.json", redacted_payload)

    print(f"[DEBUG] output_dir={run_dir}")
    print(f"[DEBUG] effective_payload={json.dumps(redacted_payload, default=_json_default)}")

    sources = ["direct", "bridge"] if args.source == "both" else [args.source]
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}

    for source in sources:
        try:
            results[source] = _run_one_source(source, payload, run_dir, args.skip_engine)
        except Exception as exc:
            failures[source] = str(exc)
            print(f"[DEBUG][{source}] FAILED: {exc}")
            traceback.print_exc()

    comparison = _compare_summaries(results)
    if failures:
        comparison["failures"] = failures

    _safe_write_json(run_dir / "summary.json", {"payload": redacted_payload, "results": results, "failures": failures})
    _safe_write_json(run_dir / "comparison.json", comparison)

    print("\n[DEBUG] comparison diagnosis")
    for line in comparison.get("diagnosis", []):
        print(f"[DEBUG] - {line}")
    if failures:
        print(f"[DEBUG] failures={failures}")
    print(f"[DEBUG] wrote reports to {run_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
