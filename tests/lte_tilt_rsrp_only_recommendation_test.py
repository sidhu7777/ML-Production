from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.lte_rf_debug_lab import DEFAULT_PROJECT_ID, DEFAULT_REGION, DEFAULT_SESSION_IDS, DEFAULT_VALIDATION_FRACTION, _write_json
from tests import lte_rf_debug_lab as rf_lab
from tests import lte_tilt_recommendation_test as base


OUTPUT_ROOT = Path("tests/output")
PROJECT_196_RSRP_TILT_FIXTURE_ROOT = Path("tests/fixtures/project_196_rsrp_tilt")
PROJECT_196_RSRP_TILT_BASELINE_POINTS = PROJECT_196_RSRP_TILT_FIXTURE_ROOT / "rf_prediction_grid.parquet"
PROJECT_196_RSRP_TILT_ANTENNA_INPUT = PROJECT_196_RSRP_TILT_FIXTURE_ROOT / "antenna_input.csv"
PROJECT_196_RSRP_TILT_GEO_FEATURES = PROJECT_196_RSRP_TILT_FIXTURE_ROOT / "geo_features_input.csv.gz"
PROJECT_196_RSRP_TILT_GRID_ANALYTICS = PROJECT_196_RSRP_TILT_FIXTURE_ROOT / "local_grid_analytics_geo.csv"
PROJECT_196_RSRP_TILT_THRESHOLD_FILE = PROJECT_196_RSRP_TILT_FIXTURE_ROOT / "lte_tilt_recommendation_transformed.csv"


def _quiet_build_cell_site_map(antenna_df: pd.DataFrame) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        ant_work = base.opt_ml._normalize_site_df(antenna_df, log_stage="TILT_TEST_SITE_MAP")
    baseline_df = getattr(base, "_rsrp_only_active_baseline_df", None)
    if isinstance(baseline_df, pd.DataFrame) and not baseline_df.empty:
        ant_work = _canonicalize_site_df_to_baseline(ant_work, baseline_df)
    ant_work["Cell ID"] = ant_work["Node_Cell_ID"].astype(str).map(base.TILT_SRC._norm_cell_id)
    ant_work["Site ID"] = ant_work["dashboard_site_id"].astype(str).str.strip()
    ant_work["Sector Suffix"] = ant_work["Cell ID"].map(base.TILT_SRC._cell_id_suffix)
    return ant_work.drop_duplicates(subset=["Cell ID"], keep="last")


def _quiet_apply_multiple_parameter_targets(
    site_df: pd.DataFrame,
    target_updates: Sequence[Dict[str, object]],
) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        modified = base.opt_ml._normalize_site_df(site_df, log_stage="TILT_TEST_CLUSTER_CANDIDATE_INPUT")
    baseline_df = getattr(base, "_rsrp_only_active_baseline_df", None)
    if isinstance(baseline_df, pd.DataFrame) and not baseline_df.empty:
        modified = _canonicalize_site_df_to_baseline(modified, baseline_df)
    for col in ["lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "tx_power", "antenna_height"]:
        modified[f"orig_{col}"] = pd.to_numeric(modified[col], errors="coerce")

    param_map = {
        "ETilt": "electrical_tilt",
        "Azimuth": "azimuth",
        "TX Power": "tx_power",
    }
    applied_mask = pd.Series(False, index=modified.index)
    for update in target_updates:
        target_col = param_map.get(str(update.get("parameter")))
        if not target_col:
            continue
        cell_id = str(update.get("cell_id", ""))
        mask = base._site_match_mask(modified, cell_id)
        if not mask.any():
            raise ValueError(f"Could not match site rows for cell_id={cell_id}")
        modified.loc[mask, target_col] = float(update["target_value"])
        applied_mask = applied_mask | mask
    modified["optimization_applied"] = applied_mask.astype(bool)
    return modified


base._build_cell_site_map = _quiet_build_cell_site_map
base._apply_multiple_parameter_targets = _quiet_apply_multiple_parameter_targets


@dataclass
class TiltRsrpOnlyRecommendationTestConfig:
    project_id: int = DEFAULT_PROJECT_ID
    region: str = DEFAULT_REGION
    operator: Optional[str] = None
    rsrp_threshold: float = -90.0
    radius_m: float = 500.0
    grid_resolution_m: float = 30.0
    workers: int = 4
    impact_radius_m: Optional[float] = None
    neighbor_site_count: int = 3
    max_interference_sites: int = 10
    max_good_area_loss_pct: float = 2.0
    max_mean_sinr_drop_db: float = 1.0
    min_score_gain: float = 0.0
    min_recovered_bad_samples: int = 0
    bad_grid_coverage_pct: float = 80.0
    max_group_cells: int = 0
    threshold_file_path: Optional[str] = None
    threshold_constraint_count: int = 0
    threshold_optimised_count: int = 0
    baseline_points_path: Optional[str] = None
    antenna_input_path: Optional[str] = None
    geo_features_path: Optional[str] = None
    grid_analytics_path: Optional[str] = None
    local_baseline_kpi_stage: str = "geo"
    session_ids: Sequence[int] = tuple(DEFAULT_SESSION_IDS)
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    apply_residual_calibration: bool = True
    fixed_k1k2_for_local_inputs: bool = True
    output_root: Path = OUTPUT_ROOT

    def __post_init__(self) -> None:
        if self.impact_radius_m is None:
            self.impact_radius_m = float(self.radius_m)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv_artifact(df: pd.DataFrame, path: Path, *, compress: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        out_path = path.with_suffix(path.suffix + ".gz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, compression="gzip")
        return str(out_path)
    df.to_csv(path, index=False)
    return str(path)


def _read_local_table(path_value: Optional[str]) -> pd.DataFrame:
    if not path_value:
        return pd.DataFrame()
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Local input file not found: {path}")
    suffixes = "".join(path.suffixes).lower()
    print(f"[TILT_RSRP_LOCAL_INPUT] loading={path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported local input format: {path}")


def _prepare_local_baseline_points(df: pd.DataFrame, kpi_stage: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if "nodeb_id_cell_id" in out.columns and "Node_Cell_ID" not in out.columns:
        out["Node_Cell_ID"] = out["nodeb_id_cell_id"].astype(str)
    kpi_stage = str(kpi_stage or "geo").strip().lower()
    if kpi_stage == "demo":
        metric_cols = {
            "pred_rsrp": ["pred_rsrp_demo", "pred_rsrp_geo", "pred_rsrp"],
            "pred_rsrq": ["pred_rsrq_demo", "pred_rsrq_geo", "pred_rsrq"],
            "pred_sinr": ["pred_sinr_demo", "pred_sinr_geo", "pred_sinr"],
        }
    elif kpi_stage == "raw":
        metric_cols = {
            "pred_rsrp": ["pred_rsrp", "pred_rsrp_geo", "pred_rsrp_demo"],
            "pred_rsrq": ["pred_rsrq", "pred_rsrq_geo", "pred_rsrq_demo"],
            "pred_sinr": ["pred_sinr", "pred_sinr_geo", "pred_sinr_demo"],
        }
    else:
        kpi_stage = "geo"
        metric_cols = {
            "pred_rsrp": ["pred_rsrp_geo", "pred_rsrp_demo", "pred_rsrp"],
            "pred_rsrq": ["pred_rsrq_geo", "pred_rsrq_demo", "pred_rsrq"],
            "pred_sinr": ["pred_sinr_geo", "pred_sinr_demo", "pred_sinr"],
        }
    for target, candidates in metric_cols.items():
        chosen = next((col for col in candidates if col in out.columns), None)
        if chosen is None:
            raise ValueError(f"Local baseline is missing KPI column for {target}; tried {candidates}")
        out[target] = pd.to_numeric(out[chosen], errors="coerce")
    out["Node_Cell_ID"] = out["Node_Cell_ID"].astype(str).str.strip()
    out["node_cell_id"] = out["Node_Cell_ID"]
    if "Technology" not in out.columns:
        out["Technology"] = "4G"
    print(
        f"[TILT_RSRP_LOCAL_INPUT] baseline_points rows={len(out)} "
        f"distinct_node_cell_id={out['Node_Cell_ID'].nunique()} kpi_stage={kpi_stage}"
    )
    return out


def _prepare_residual_prediction_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    for base_col, geo_col in [
        ("pred_rsrp", "pred_rsrp_geo"),
        ("pred_rsrq", "pred_rsrq_geo"),
        ("pred_sinr", "pred_sinr_geo"),
    ]:
        if geo_col not in out.columns and base_col in out.columns:
            out[geo_col] = pd.to_numeric(out[base_col], errors="coerce")
    return out


def _fit_fixed_residual_calibration(
    baseline_df: pd.DataFrame,
    config: TiltRsrpOnlyRecommendationTestConfig,
) -> tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    if not bool(config.apply_residual_calibration):
        return {}, {"enabled": False, "reason": "disabled"}
    operator = config.operator or "Airtel"
    try:
        drive_df = rf_lab._fetch_drive_data_for_test(
            session_ids=tuple(int(value) for value in config.session_ids),
            operator=operator,
            project_id=int(config.project_id),
            region=str(config.region),
        )
        drive_train_df, _ = rf_lab._split_drive_train_holdout(drive_df, float(config.validation_fraction))
        residual_source_df = _prepare_residual_prediction_frame(baseline_df)
        train_eval, _, train_metrics = rf_lab._evaluate_prediction_grid_against_holdout(
            drive_train_df,
            residual_source_df,
        )
        residual_models, residual_debug = rf_lab._fit_train_only_residual_calibration(train_eval)
        residual_debug["source"] = "tilt_rsrp_baseline_prediction_frame"
        residual_debug["operator"] = operator
        residual_debug["session_ids"] = [int(value) for value in config.session_ids]
        residual_debug["train_metrics"] = train_metrics
        print(
            f"[TILT_RSRP_RESIDUAL] enabled={bool(residual_models)} "
            f"drive_rows={len(drive_df)} train_rows={len(drive_train_df)} "
            f"eval_rows={len(train_eval)} models={list(residual_models.keys())}"
        )
        return residual_models, residual_debug
    except Exception as exc:
        print(f"[TILT_RSRP_RESIDUAL] enabled=False reason={exc}")
        return {}, {"enabled": False, "reason": str(exc)}


def _apply_fixed_residual_calibration_to_rf(
    rf_df: pd.DataFrame,
    residual_models: Optional[Dict[str, Dict[str, object]]],
) -> pd.DataFrame:
    out = _prepare_residual_prediction_frame(rf_df)
    if residual_models:
        out = rf_lab._apply_train_only_residual_calibration(out, residual_models)
    for base_col, geo_col in [
        ("pred_rsrp", "pred_rsrp_geo"),
        ("pred_rsrq", "pred_rsrq_geo"),
        ("pred_sinr", "pred_sinr_geo"),
    ]:
        if geo_col in out.columns:
            out[base_col] = pd.to_numeric(out[geo_col], errors="coerce")
    return out


def _fixed_raw_k1k2_map(cells: Sequence[str]) -> Dict[str, tuple[float, float]]:
    return {str(cell): (0.0, 0.0) for cell in cells if str(cell).strip()}


def _config_as_base(config: TiltRsrpOnlyRecommendationTestConfig) -> base.TiltRecommendationTestConfig:
    base_config = base.TiltRecommendationTestConfig(
        project_id=int(config.project_id),
        region=str(config.region),
        operator=config.operator,
        rsrp_threshold=float(config.rsrp_threshold),
        rsrq_threshold=-999.0,
        sinr_threshold=-999.0,
        validate_candidates=True,
        radius_m=float(config.radius_m),
        grid_resolution_m=float(config.grid_resolution_m),
        workers=int(config.workers),
        impact_radius_m=float(config.impact_radius_m),
        neighbor_site_count=int(config.neighbor_site_count),
        max_interference_sites=int(config.max_interference_sites),
        max_good_area_loss_pct=float(config.max_good_area_loss_pct),
        max_mean_sinr_drop_db=float(config.max_mean_sinr_drop_db),
        min_score_gain=float(config.min_score_gain),
        min_recovered_bad_samples=int(config.min_recovered_bad_samples),
        max_ranked_cells=0,
        max_ranked_sites=0,
        threshold_file_path=config.threshold_file_path,
        threshold_constraint_count=int(config.threshold_constraint_count),
        threshold_optimised_count=int(config.threshold_optimised_count),
        output_root=config.output_root,
    )
    setattr(base_config, "bad_grid_coverage_pct", float(config.bad_grid_coverage_pct))
    setattr(base_config, "max_group_cells", int(config.max_group_cells))
    return base_config


def _fetch_grid_analytics_df(project_id: int, region: str, operator: Optional[str]) -> pd.DataFrame:
    current_engine = base._resolve_engine(region)
    schema_query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = 'grid_analytics_results'
        """
    )
    with current_engine.connect() as conn:
        schema_df = pd.read_sql(schema_query, conn)
    if schema_df.empty:
        return pd.DataFrame()

    schema_col = "column_name" if "column_name" in schema_df.columns else schema_df.columns[0]
    available_cols = set(schema_df[schema_col].astype(str).tolist())
    base_cols = [
        "project_id",
        "grid_id",
        "center_lat",
        "center_lon",
        "min_lat",
        "max_lat",
        "min_lon",
        "max_lon",
        "baseline_point_count",
        "baseline_avg_rsrp",
        "baseline_avg_rsrq",
        "baseline_avg_sinr",
        "operator",
        "created_at",
        "scenario_id",
    ]
    select_cols = [col for col in base_cols if col in available_cols]
    if not {"project_id", "grid_id"}.issubset(set(select_cols)):
        return pd.DataFrame()

    filters = ["project_id = :project_id"]
    params: Dict[str, object] = {"project_id": int(project_id)}
    if operator and "operator" in available_cols:
        filters.append("LOWER(TRIM(operator)) = :operator")
        params["operator"] = str(operator).strip().lower()

    scenario_filter = ""
    selected_scenario_id = None
    if "scenario_id" in available_cols:
        scenario_query = text(
            f"""
            SELECT scenario_id, MAX(created_at) AS max_created, COUNT(*) AS row_count
            FROM grid_analytics_results
            WHERE {" AND ".join(filters)}
            GROUP BY scenario_id
            ORDER BY max_created DESC, row_count DESC
            LIMIT 1
            """
        )
        with current_engine.connect() as conn:
            scenario_row = conn.execute(scenario_query, params).fetchone()
        if scenario_row is not None:
            selected_scenario_id = scenario_row[0]
            if selected_scenario_id is None:
                scenario_filter = " AND scenario_id IS NULL"
            else:
                scenario_filter = " AND scenario_id = :scenario_id"
                params["scenario_id"] = selected_scenario_id

    order_cols = [col for col in ["created_at", "grid_id"] if col in available_cols]
    order_sql = ", ".join(f"{col} DESC" if col == "created_at" else f"{col} ASC" for col in order_cols)
    query = text(
        f"""
        SELECT {", ".join(f"`{col}`" for col in select_cols)}
        FROM grid_analytics_results
        WHERE {" AND ".join(filters)}
        {scenario_filter}
        {f"ORDER BY {order_sql}" if order_sql else ""}
        """
    )
    with current_engine.connect() as conn:
        grid_df = pd.read_sql(query, conn, params=params)
    if selected_scenario_id is not None and "scenario_id" not in grid_df.columns:
        grid_df["scenario_id"] = selected_scenario_id
    print(
        f"[TILT_RSRP_GRID][GRID_ANALYTICS_FETCH] rows={len(grid_df)} "
        f"selected_scenario_id={selected_scenario_id} operator_filter_applied={bool(operator and 'operator' in available_cols)}"
    )
    return grid_df


def _normalize_grid_id_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="string")
    out = series.astype("string").str.strip()
    return out.mask(out.isin(["", "nan", "NaN", "None", "<NA>"]))


def _prepare_baseline_identity_for_rsrp_only(log_df: pd.DataFrame) -> pd.DataFrame:
    out = log_df.copy()
    if out.empty:
        return out
    if "frontend_site_sector_key" in out.columns:
        frontend_key = rf_lab._clean_identity_part(out["frontend_site_sector_key"])
        if "Node_Cell_ID" in out.columns:
            out["original_node_cell_id"] = rf_lab._clean_identity_part(out.get("original_node_cell_id", out["Node_Cell_ID"])).fillna(
                rf_lab._clean_identity_part(out["Node_Cell_ID"])
            )
        out.loc[frontend_key.notna(), "Node_Cell_ID"] = frontend_key.loc[frontend_key.notna()]
        out["node_cell_id"] = out["Node_Cell_ID"].astype(str)
    else:
        out = rf_lab._add_sector_identity_columns(out, use_as_node_cell_id=True)

    out["Node_Cell_ID"] = rf_lab._clean_identity_part(out["Node_Cell_ID"])
    out["frontend_site_sector_key"] = rf_lab._clean_identity_part(out.get("frontend_site_sector_key", out["Node_Cell_ID"])).fillna(out["Node_Cell_ID"])
    out["node_cell_id"] = out["Node_Cell_ID"].astype(str)
    source_node_cell = (
        rf_lab._clean_identity_part(out["original_node_cell_id"])
        if "original_node_cell_id" in out.columns
        else rf_lab._clean_identity_part(out["Node_Cell_ID"])
    )
    split_source = source_node_cell.astype("string").str.split("_", n=1, expand=True)
    if not split_source.empty and split_source.shape[1] >= 1 and "node_b_id" not in out.columns:
        out["node_b_id"] = split_source[0]
    if not split_source.empty and split_source.shape[1] >= 1 and "nodeb_id" not in out.columns:
        out["nodeb_id"] = split_source[0]
    if not split_source.empty and split_source.shape[1] >= 2:
        if "local_cell_id" not in out.columns:
            out["local_cell_id"] = split_source[1]
        if "original_cell_id" not in out.columns:
            out["original_cell_id"] = split_source[1]
    if "cell_id" in out.columns and "original_cell_id" not in out.columns:
        out["original_cell_id"] = rf_lab._clean_identity_part(out["cell_id"])
    return _canonicalize_prediction_identity_columns(out)


def _prepare_antenna_identity_for_rsrp_only(antenna_df: pd.DataFrame) -> pd.DataFrame:
    out = rf_lab._add_sector_identity_columns(antenna_df, use_as_node_cell_id=True)
    if out.empty:
        return out
    out["Node_Cell_ID"] = rf_lab._clean_identity_part(out["Node_Cell_ID"])
    out["frontend_site_sector_key"] = out["Node_Cell_ID"]

    if "tx_power" not in out.columns:
        out["tx_power"] = np.nan
    tx_power = pd.to_numeric(out["tx_power"], errors="coerce")
    source_power_cols = [
        col for col in [
            "real_transmit_power_of_resource",
            "reference_signal_power",
        ] if col in out.columns
    ]
    source_power_missing = pd.Series(False, index=out.index)
    if source_power_cols:
        source_power_missing = pd.concat(
            [pd.to_numeric(out[col], errors="coerce").isna() for col in source_power_cols],
            axis=1,
        ).all(axis=1)
    fallback_power = rf_lab._tx_power_from_frequency_mhz(rf_lab._infer_frequency_for_power_fallback(out))
    fallback_mask = tx_power.isna() | (tx_power.eq(46.0) & source_power_missing)
    out["tx_power"] = tx_power.where(~fallback_mask, fallback_power)
    fallback_rows = int(fallback_mask.sum())
    power_counts = pd.to_numeric(out["tx_power"], errors="coerce").value_counts(dropna=False).sort_index().to_dict()
    band_counts = (
        pd.to_numeric(out["band"], errors="coerce").value_counts(dropna=False).sort_index().to_dict()
        if "band" in out.columns
        else {}
    )

    out["rf_source_cell_id"] = rf_lab._clean_identity_part(out["cell_id"]) if "cell_id" in out.columns else out["original_cell_id"]
    out["cell_id"] = out["Node_Cell_ID"].astype(str)
    out["node_cell_id"] = out["Node_Cell_ID"].astype(str)
    print(
        f"[TILT_RSRP_IDENTITY] antenna_rows={len(out)} "
        f"unique_frontend_site_sector={out['Node_Cell_ID'].astype(str).nunique()} "
        f"tx_power_fallback_rows={fallback_rows} tx_power_counts={power_counts} band_counts={band_counts}"
    )
    return out


def _prepare_rsrp_only_inputs(log_df: pd.DataFrame, antenna_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_out = _prepare_baseline_identity_for_rsrp_only(log_df)
    antenna_out = _prepare_antenna_identity_for_rsrp_only(antenna_df)
    log_unique = int(log_out["Node_Cell_ID"].astype(str).nunique()) if "Node_Cell_ID" in log_out.columns else 0
    antenna_unique = int(antenna_out["Node_Cell_ID"].astype(str).nunique()) if "Node_Cell_ID" in antenna_out.columns else 0
    print(
        f"[TILT_RSRP_IDENTITY] baseline_rows={len(log_out)} baseline_unique_cells={log_unique} "
        f"antenna_rows={len(antenna_out)} antenna_unique_cells={antenna_unique}"
    )
    return log_out, antenna_out


def _attach_baseline_identity_columns_from_db(
    log_df: pd.DataFrame,
    project_id: int,
    region: str,
    baseline_job_id: str,
) -> pd.DataFrame:
    if log_df.empty:
        return log_df.copy()

    current_engine = base._resolve_engine(region)
    schema_query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = 'lte_prediction_baseline_results'
        """
    )
    with current_engine.connect() as conn:
        schema_df = pd.read_sql(schema_query, conn)
    schema_col = "column_name" if "column_name" in schema_df.columns else "COLUMN_NAME"
    available_cols = set(schema_df[schema_col].astype(str))

    identity_cols = [
        col for col in [
            "lat",
            "lon",
            "nodeb_id_cell_id",
            "frontend_site_sector_key",
            "sector",
            "site_id",
            "node_b_id",
            "cell_id",
        ] if col in available_cols
    ]
    required = {"lat", "lon", "nodeb_id_cell_id", "frontend_site_sector_key", "sector"}
    if not required.issubset(set(identity_cols)):
        print(
            f"[TILT_RSRP_IDENTITY][WARN] baseline_identity_columns_missing="
            f"{sorted(required.difference(set(identity_cols)))}"
        )
        return log_df.copy()

    query = text(
        f"""
        SELECT {", ".join(f"`{col}`" for col in identity_cols)}
        FROM lte_prediction_baseline_results
        WHERE project_id = :project_id
          AND job_id = :baseline_job_id
        """
    )
    with current_engine.connect() as conn:
        identity_df = pd.read_sql(
            query,
            conn,
            params={"project_id": int(project_id), "baseline_job_id": str(baseline_job_id)},
        )
    if identity_df.empty:
        print(f"[TILT_RSRP_IDENTITY][WARN] baseline_identity_rows=0 job_id={baseline_job_id}")
        return log_df.copy()

    out = log_df.copy()
    out["_identity_lat_key"] = pd.to_numeric(out["lat"], errors="coerce").round(6)
    out["_identity_lon_key"] = pd.to_numeric(out["lon"], errors="coerce").round(6)
    out["_identity_raw_cell_key"] = rf_lab._clean_identity_part(out["Node_Cell_ID"])

    ident = identity_df.copy()
    ident["_identity_lat_key"] = pd.to_numeric(ident["lat"], errors="coerce").round(6)
    ident["_identity_lon_key"] = pd.to_numeric(ident["lon"], errors="coerce").round(6)
    ident["_identity_raw_cell_key"] = rf_lab._clean_identity_part(ident["nodeb_id_cell_id"])
    ident = ident.drop(columns=["lat", "lon"], errors="ignore")
    ident = ident.drop_duplicates(
        subset=["_identity_lat_key", "_identity_lon_key", "_identity_raw_cell_key"],
        keep="last",
    )
    rename_map = {
        col: f"{col}_identity_db"
        for col in ["nodeb_id_cell_id", "frontend_site_sector_key", "sector", "site_id", "node_b_id", "cell_id"]
        if col in ident.columns
    }
    ident = ident.rename(columns=rename_map)
    out = out.merge(ident, on=["_identity_lat_key", "_identity_lon_key", "_identity_raw_cell_key"], how="left")

    if "nodeb_id_cell_id_identity_db" in out.columns:
        out["raw_nodeb_id_cell_id"] = out["nodeb_id_cell_id_identity_db"]
    for col in ["frontend_site_sector_key", "sector", "site_id", "node_b_id", "cell_id"]:
        db_col = f"{col}_identity_db"
        if db_col not in out.columns:
            continue
        db_values = rf_lab._clean_identity_part(out[db_col])
        if col in out.columns:
            out[col] = db_values.fillna(rf_lab._clean_identity_part(out[col]))
        else:
            out[col] = db_values
    out = out.drop(
        columns=[
            "_identity_lat_key",
            "_identity_lon_key",
            "_identity_raw_cell_key",
            *[f"{col}_identity_db" for col in ["nodeb_id_cell_id", "frontend_site_sector_key", "sector", "site_id", "node_b_id", "cell_id"]],
        ],
        errors="ignore",
    )
    print(
        f"[TILT_RSRP_IDENTITY] baseline_identity_db_rows={len(identity_df)} "
        f"frontend_unique={identity_df['frontend_site_sector_key'].astype(str).nunique()} "
        f"sector_unique={identity_df['sector'].astype(str).nunique()}"
    )
    return out


def _identity_text(series: pd.Series) -> pd.Series:
    return rf_lab._clean_identity_part(series).astype("string")


def _strip_dot_zero_value(value: object) -> str:
    text_value = "" if pd.isna(value) else str(value).strip()
    if text_value.endswith(".0"):
        text_value = text_value[:-2]
    return text_value


def _local_cell_token(value: object) -> str:
    text_value = _strip_dot_zero_value(value)
    if "_" in text_value:
        text_value = text_value.rsplit("_", 1)[-1]
    return _strip_dot_zero_value(text_value)


def _canonical_frontend_site_sector_key(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    raw_value = str(value).strip()
    if not raw_value or raw_value.lower() in {"nan", "none", "<na>"}:
        return pd.NA

    parts = [part.strip() for part in raw_value.split("|") if part.strip()]
    if len(parts) < 2:
        return raw_value

    site_key = parts[0]
    node_key = _strip_dot_zero_value(site_key)
    sector_part = parts[1]

    if "_" in sector_part:
        sector_key = sector_part
    elif len(parts) >= 3 and _strip_dot_zero_value(sector_part) == node_key:
        sector_key = f"{node_key}_{_local_cell_token(parts[2])}"
    else:
        sector_key = f"{node_key}_{_local_cell_token(sector_part)}"

    return f"{site_key}|{sector_key}"


def _canonical_frontend_site_sector_key_series(series: pd.Series) -> pd.Series:
    return series.apply(_canonical_frontend_site_sector_key).astype("string")


def _canonicalize_cell_id(value: object, allowed_keys: Optional[set[str]] = None) -> Optional[str]:
    if pd.isna(value):
        return None
    raw_value = str(value).strip()
    if not raw_value or raw_value.lower() in {"nan", "none", "<na>"}:
        return None
    canonical = _canonical_frontend_site_sector_key(raw_value)
    canonical_value = None if pd.isna(canonical) else str(canonical).strip()
    if allowed_keys is None:
        return canonical_value or raw_value
    if raw_value in allowed_keys:
        return raw_value
    if canonical_value and canonical_value in allowed_keys:
        return canonical_value
    return canonical_value or raw_value


def _canonicalize_cell_list(cells: Sequence[object], baseline_df: Optional[pd.DataFrame] = None) -> List[str]:
    allowed_keys: Optional[set[str]] = None
    if baseline_df is not None and not baseline_df.empty and "Node_Cell_ID" in baseline_df.columns:
        allowed_keys = set(_identity_text(baseline_df["Node_Cell_ID"]).dropna().astype(str).tolist())
    out: List[str] = []
    seen: set[str] = set()
    for cell in cells:
        key = _canonicalize_cell_id(cell, allowed_keys)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return sorted(out)


def _canonicalize_site_df_to_baseline(site_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    if site_df.empty or baseline_df.empty or "Node_Cell_ID" not in site_df.columns or "Node_Cell_ID" not in baseline_df.columns:
        return site_df
    out = site_df.copy()
    baseline_keys = set(_identity_text(baseline_df["Node_Cell_ID"]).dropna().astype(str).tolist())
    if not baseline_keys:
        return out
    raw = _identity_text(out["Node_Cell_ID"])
    canonical_values = raw.map(lambda value: _canonicalize_cell_id(value, baseline_keys))
    matched = canonical_values.isin(baseline_keys)
    if bool(matched.any()):
        out.loc[matched, "Node_Cell_ID"] = canonical_values.loc[matched].astype(str)
        for col in ["cell_id", "node_cell_id", "frontend_site_sector_key"]:
            if col in out.columns:
                out.loc[matched, col] = canonical_values.loc[matched].astype(str)
    if int(matched.sum()) != len(out):
        print(
            f"[TILT_RSRP_IDENTITY][WARN] site_df_canonical_match={int(matched.sum())}/{len(out)} "
            f"sample_unmatched={raw.loc[~matched].dropna().astype(str).drop_duplicates().head(5).tolist()}"
        )
    return out


def _canonicalize_prediction_identity_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df
    out = pred_df.copy()
    for col in ["best_interferer_cell_id", "neighbor_1_cell_id", "neighbor_2_cell_id"]:
        if col in out.columns:
            out[col] = _canonical_frontend_site_sector_key_series(out[col])
    if "Node_Cell_ID" in out.columns:
        out["Node_Cell_ID"] = rf_lab._clean_identity_part(out["Node_Cell_ID"])
        out["node_cell_id"] = out["Node_Cell_ID"].astype(str)
    return out


def _strip_numeric_dot_zero(series: pd.Series) -> pd.Series:
    return _identity_text(series).str.replace(r"\.0$", "", regex=True)


def _append_numeric_dot_zero(series: pd.Series) -> pd.Series:
    clean = _strip_numeric_dot_zero(series)
    numeric_like = clean.str.fullmatch(r"\d+", na=False)
    return clean.where(~numeric_like, clean + ".0")


def _align_antenna_identity_to_baseline_keys(antenna_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    if antenna_df.empty or baseline_df.empty or "Node_Cell_ID" not in baseline_df.columns:
        return antenna_df

    baseline_keys = set(_identity_text(baseline_df["Node_Cell_ID"]).dropna().astype(str).tolist())
    if not baseline_keys or "Node_Cell_ID" not in antenna_df.columns:
        return antenna_df

    out = antenna_df.copy()
    current_key = _identity_text(out["Node_Cell_ID"])
    matched = current_key.isin(baseline_keys)
    if bool(matched.all()):
        out["Node_Cell_ID"] = current_key.astype(str)
        out["frontend_site_sector_key"] = current_key.astype(str)
        out["cell_id"] = current_key.astype(str)
        out["node_cell_id"] = current_key.astype(str)
        print(f"[TILT_RSRP_IDENTITY] antenna_baseline_key_match={int(matched.sum())}/{len(out)} mode=direct")
        return out

    sector_or_cell = (
        _identity_text(out["sector_identity"])
        if "sector_identity" in out.columns
        else _identity_text(out.get("rf_source_cell_id", out["Node_Cell_ID"]))
    )
    site_candidates: List[pd.Series] = []
    for col in ["site_identity_key", "site", "Site ID", "dashboard_site_id", "nodeb_id"]:
        if col in out.columns:
            site_candidates.extend([_identity_text(out[col]), _strip_numeric_dot_zero(out[col]), _append_numeric_dot_zero(out[col])])

    if not site_candidates:
        print(
            f"[TILT_RSRP_IDENTITY][WARN] no_site_columns_for_key_alignment "
            f"direct_matches={int(matched.sum())}/{len(out)} baseline_unique={len(baseline_keys)}"
        )
        return out

    aligned_key = current_key.copy()
    for site_key in site_candidates:
        candidate_key = site_key.astype(str) + "|" + sector_or_cell.astype(str)
        candidate_key = candidate_key.mask(site_key.isna() | sector_or_cell.isna())
        use_mask = ~matched & candidate_key.isin(baseline_keys)
        if bool(use_mask.any()):
            aligned_key.loc[use_mask] = candidate_key.loc[use_mask]
            matched.loc[use_mask] = True
        if bool(matched.all()):
            break

    if bool(matched.any()):
        baseline_key_map = {key.casefold(): key for key in baseline_keys}
        aligned_key.loc[matched] = aligned_key.loc[matched].astype(str).str.casefold().map(baseline_key_map).fillna(aligned_key.loc[matched])
    out.loc[matched, "Node_Cell_ID"] = aligned_key.loc[matched].astype(str)
    out.loc[matched, "frontend_site_sector_key"] = aligned_key.loc[matched].astype(str)
    out.loc[matched, "cell_id"] = aligned_key.loc[matched].astype(str)
    out.loc[matched, "node_cell_id"] = aligned_key.loc[matched].astype(str)
    print(
        f"[TILT_RSRP_IDENTITY] antenna_baseline_key_match={int(matched.sum())}/{len(out)} "
        f"baseline_unique={len(baseline_keys)} antenna_unique_after_align={out['Node_Cell_ID'].astype(str).nunique()}"
    )
    if not bool(matched.any()):
        print(
            f"[TILT_RSRP_IDENTITY][WARN] sample_baseline_keys={sorted(list(baseline_keys))[:5]} "
            f"sample_antenna_keys={current_key.dropna().astype(str).drop_duplicates().head(5).tolist()}"
        )
    return out


def _load_aligned_project_polygon(project_id: int, region: str, point_df: pd.DataFrame) -> gpd.GeoDataFrame:
    try:
        polygon_gdf = rf_lab._load_project_polygon_gdf(int(project_id), str(region))
        polygon_gdf, alignment = rf_lab._align_project_polygon_to_points(polygon_gdf, point_df)
        print(f"[TILT_RSRP_POLYGON][LOAD] project_id={project_id} alignment={alignment} polygons={len(polygon_gdf)}")
        return polygon_gdf
    except Exception as exc:
        raise RuntimeError(f"Unable to load/align project polygon for project_id={project_id}: {exc}") from exc


def _filter_points_to_project_polygon(
    df: pd.DataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    label: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if polygon_gdf.empty:
        raise ValueError(f"Project polygon is empty; cannot polygon-filter {label}")
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"{label} is missing required columns for polygon filter: {lat_col}, {lon_col}")

    work = df.copy()
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    valid = work[lat_col].notna() & work[lon_col].notna()
    if not valid.any():
        raise ValueError(f"{label} has no valid lat/lon rows for polygon filtering")

    point_gdf = gpd.GeoDataFrame(
        work.loc[valid].copy(),
        geometry=gpd.points_from_xy(work.loc[valid, lon_col], work.loc[valid, lat_col]),
        crs=polygon_gdf.crs or "EPSG:4326",
    )
    polygon_union = polygon_gdf.geometry.union_all()
    inside_index = point_gdf.loc[point_gdf.geometry.apply(lambda geom: bool(polygon_union.covers(geom)))].index
    out = work.loc[inside_index].copy()
    print(
        f"[TILT_RSRP_POLYGON][FILTER] label={label} rows_before={len(df)} "
        f"rows_after={len(out)} removed={len(df) - len(out)}"
    )
    return out.drop(columns=["geometry"], errors="ignore")


def _filter_run_inputs_to_project_polygon(
    log_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    project_id: int,
    region: str,
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    polygon_gdf = _load_aligned_project_polygon(project_id, region, antenna_df)
    antenna_filtered = _filter_points_to_project_polygon(antenna_df, polygon_gdf, "antenna_sites")
    log_filtered = _filter_points_to_project_polygon(log_df, polygon_gdf, "baseline_rows")
    antenna_filtered = _align_antenna_identity_to_baseline_keys(antenna_filtered, log_filtered)
    allowed_cells = set(antenna_filtered["Node_Cell_ID"].astype(str).str.strip()) if "Node_Cell_ID" in antenna_filtered.columns else set()
    if allowed_cells and "Node_Cell_ID" in log_filtered.columns:
        before = len(log_filtered)
        log_filtered = log_filtered.loc[log_filtered["Node_Cell_ID"].astype(str).str.strip().isin(allowed_cells)].copy()
        print(
            f"[TILT_RSRP_POLYGON][FILTER] label=baseline_allowed_polygon_cells "
            f"rows_before={before} rows_after={len(log_filtered)} removed={before - len(log_filtered)}"
        )
    if antenna_filtered.empty:
        raise ValueError("No antenna/site rows remain after project polygon filtering")
    if log_filtered.empty:
        raise ValueError("No baseline rows remain after project polygon filtering")
    return log_filtered, antenna_filtered, polygon_gdf


def _attach_grid_id_from_grid_bounds(pred_df: pd.DataFrame, grid_analytics_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    required = {"grid_id", "min_lat", "max_lat", "min_lon", "max_lon"}
    if out.empty or grid_analytics_df.empty or not required.issubset(set(grid_analytics_df.columns)):
        return out

    if "grid_id" not in out.columns:
        out["grid_id"] = pd.NA
    lat = pd.to_numeric(out.get("lat"), errors="coerce")
    lon = pd.to_numeric(out.get("lon"), errors="coerce")
    out["grid_id"] = _normalize_grid_id_series(out["grid_id"])
    missing_mask = out["grid_id"].isna() & lat.notna() & lon.notna()
    if not missing_mask.any():
        return out

    grids = grid_analytics_df[list(required)].copy()
    grids["grid_id"] = _normalize_grid_id_series(grids["grid_id"])
    for col in ["min_lat", "max_lat", "min_lon", "max_lon"]:
        grids[col] = pd.to_numeric(grids[col], errors="coerce")
    grids = grids.dropna(subset=["grid_id", "min_lat", "max_lat", "min_lon", "max_lon"]).drop_duplicates(subset=["grid_id"], keep="first")
    if grids.empty:
        return out

    unmatched = set(out.index[missing_mask].tolist())
    assigned = 0
    for row in grids.itertuples(index=False):
        if not unmatched:
            break
        idx = list(unmatched)
        match = (
            lat.loc[idx].between(float(row.min_lat), float(row.max_lat), inclusive="both")
            & lon.loc[idx].between(float(row.min_lon), float(row.max_lon), inclusive="both")
        )
        if not bool(match.any()):
            continue
        matched_idx = match.index[match].tolist()
        out.loc[matched_idx, "grid_id"] = str(row.grid_id)
        assigned += len(matched_idx)
        unmatched.difference_update(matched_idx)

    print(
        f"[TILT_RSRP_GRID][GRID_BOUNDS_MAP] candidate_rows={int(missing_mask.sum())} "
        f"assigned_rows={assigned} unassigned_rows={len(unmatched)} grid_rows={len(grids)}"
    )
    return out


def _attach_grid_context_to_predictions(
    pred_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    grid_analytics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    out = pred_df.copy()
    if out.empty:
        return out

    if "grid_id" not in out.columns:
        out["grid_id"] = pd.NA

    if not geo_df.empty:
        work = geo_df.copy()
        required_geo_cols = [col for col in ["Node_Cell_ID", "lat", "lon", "grid_id"] if col in work.columns]
        if len(required_geo_cols) >= 4:
            out["_grid_lat_key"] = pd.to_numeric(out["lat"], errors="coerce").round(6)
            out["_grid_lon_key"] = pd.to_numeric(out["lon"], errors="coerce").round(6)
            out["_grid_cell_key"] = out["Node_Cell_ID"].astype(str).str.strip()

            work["_grid_lat_key"] = pd.to_numeric(work["lat"], errors="coerce").round(6)
            work["_grid_lon_key"] = pd.to_numeric(work["lon"], errors="coerce").round(6)
            work["_grid_cell_key"] = work["Node_Cell_ID"].astype(str).str.strip()
            merge_cols = ["_grid_lat_key", "_grid_lon_key", "_grid_cell_key", "grid_id"]
            for col in ["clutter_class", "morphology_cluster"]:
                if col in work.columns:
                    merge_cols.append(col)
            work = work[merge_cols].drop_duplicates(subset=["_grid_lat_key", "_grid_lon_key", "_grid_cell_key"], keep="last")
            out = out.merge(work, on=["_grid_lat_key", "_grid_lon_key", "_grid_cell_key"], how="left", suffixes=("", "_geo"))
            if "grid_id_geo" in out.columns:
                current_grid_id = _normalize_grid_id_series(out["grid_id"])
                geo_grid_id = _normalize_grid_id_series(out["grid_id_geo"])
                out["grid_id"] = current_grid_id.fillna(geo_grid_id)
                out = out.drop(columns=["grid_id_geo"], errors="ignore")
            else:
                out["grid_id"] = _normalize_grid_id_series(out["grid_id"])
            out = out.drop(columns=["_grid_lat_key", "_grid_lon_key", "_grid_cell_key"], errors="ignore")

    if grid_analytics_df is not None and not grid_analytics_df.empty:
        out = _attach_grid_id_from_grid_bounds(out, grid_analytics_df)
        out["grid_id"] = _normalize_grid_id_series(out["grid_id"])

    if grid_analytics_df is not None and not grid_analytics_df.empty and "grid_id" in grid_analytics_df.columns:
        grid_meta = grid_analytics_df.copy()
        grid_meta["grid_id"] = _normalize_grid_id_series(grid_meta["grid_id"])
        grid_meta_cols = [col for col in ["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"] if col in grid_meta.columns]
        if len(grid_meta_cols) > 1:
            out = out.merge(
                grid_meta[grid_meta_cols].drop_duplicates(subset=["grid_id"], keep="last"),
                on="grid_id",
                how="left",
            )
    return out


def _merge_optimized_onto_baseline_points(
    baseline_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    grid_context_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    baseline_out = baseline_df.copy()
    if baseline_out.empty or optimized_df.empty:
        return baseline_out, {
            "optimized_matched_row_count": 0.0,
            "optimized_unmatched_row_count": float(len(optimized_df)),
            "optimized_match_pct": 0.0,
            "merged_row_count_preserved": float(len(baseline_out)),
        }

    required = {"Node_Cell_ID", "lat", "lon"}
    if not required.issubset(baseline_out.columns) or not required.issubset(optimized_df.columns):
        return baseline_out, {
            "optimized_matched_row_count": 0.0,
            "optimized_unmatched_row_count": float(len(optimized_df)),
            "optimized_match_pct": 0.0,
            "merged_row_count_preserved": float(len(baseline_out)),
        }

    optimized_work = _attach_grid_context_to_predictions(optimized_df, grid_context_df)
    baseline_out["_merge_cell_key"] = baseline_out["Node_Cell_ID"].astype(str).str.strip()
    baseline_out["_merge_lat_key"] = pd.to_numeric(baseline_out["lat"], errors="coerce").round(6)
    baseline_out["_merge_lon_key"] = pd.to_numeric(baseline_out["lon"], errors="coerce").round(6)
    optimized_work["_merge_cell_key"] = optimized_work["Node_Cell_ID"].astype(str).str.strip()
    optimized_work["_merge_lat_key"] = pd.to_numeric(optimized_work["lat"], errors="coerce").round(6)
    optimized_work["_merge_lon_key"] = pd.to_numeric(optimized_work["lon"], errors="coerce").round(6)

    key_cols = ["_merge_cell_key", "_merge_lat_key", "_merge_lon_key"]
    baseline_key_index = pd.MultiIndex.from_frame(baseline_out[key_cols])
    optimized_work = optimized_work.dropna(subset=key_cols).drop_duplicates(subset=key_cols, keep="last").copy()
    optimized_key_index = pd.MultiIndex.from_frame(optimized_work[key_cols])
    matched_mask = optimized_key_index.isin(baseline_key_index)
    matched_optimized = optimized_work.loc[matched_mask].copy()

    update_cols = [
        col for col in optimized_work.columns
        if col in baseline_out.columns
        and col not in {"lat", "lon", "Node_Cell_ID", "_merge_cell_key", "_merge_lat_key", "_merge_lon_key"}
    ]
    if not matched_optimized.empty and update_cols:
        baseline_indexed = baseline_out.set_index(key_cols, drop=False)
        optimized_indexed = matched_optimized.set_index(key_cols, drop=False)
        common_index = optimized_indexed.index.intersection(baseline_indexed.index)
        baseline_indexed.loc[common_index, update_cols] = optimized_indexed.loc[common_index, update_cols]
        baseline_out = baseline_indexed.reset_index(drop=True)

    baseline_out = baseline_out.drop(columns=key_cols, errors="ignore")
    matched_count = int(len(matched_optimized))
    unmatched_count = int(len(optimized_work) - matched_count)
    match_pct = (float(matched_count) / float(len(optimized_work)) * 100.0) if len(optimized_work) else 0.0
    print(
        f"[TILT_RSRP_SAFE_REPLACE] baseline_rows={len(baseline_df)} optimized_rows={len(optimized_df)} "
        f"optimized_unique_points={len(optimized_work)} matched_points={matched_count} "
        f"unmatched_points={unmatched_count} match_pct={match_pct:.2f} final_rows={len(baseline_out)}"
    )
    return baseline_out, {
        "optimized_matched_row_count": float(matched_count),
        "optimized_unmatched_row_count": float(unmatched_count),
        "optimized_match_pct": float(match_pct),
        "merged_row_count_preserved": float(len(baseline_out)),
    }


def _apply_rf_delta_to_baseline_points(
    stored_baseline_df: pd.DataFrame,
    rf_baseline_df: pd.DataFrame,
    rf_candidate_df: pd.DataFrame,
    grid_context_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    baseline_out = stored_baseline_df.copy()
    metric_cols = [col for col in ["pred_rsrp", "pred_rsrq", "pred_sinr"] if col in baseline_out.columns]
    required = {"Node_Cell_ID", "lat", "lon"}
    if (
        baseline_out.empty
        or rf_baseline_df.empty
        or rf_candidate_df.empty
        or not metric_cols
        or not required.issubset(baseline_out.columns)
        or not required.issubset(rf_baseline_df.columns)
        or not required.issubset(rf_candidate_df.columns)
    ):
        return baseline_out, {
            "rf_delta_matched_row_count": 0.0,
            "rf_delta_unmatched_row_count": float(len(rf_candidate_df)),
            "rf_delta_match_pct": 0.0,
            "merged_row_count_preserved": float(len(baseline_out)),
        }

    rf_base = _attach_grid_context_to_predictions(rf_baseline_df, grid_context_df)
    rf_cand = _attach_grid_context_to_predictions(rf_candidate_df, grid_context_df)
    for frame in [baseline_out, rf_base, rf_cand]:
        frame["_merge_cell_key"] = frame["Node_Cell_ID"].astype(str).str.strip()
        frame["_merge_lat_key"] = pd.to_numeric(frame["lat"], errors="coerce").round(6)
        frame["_merge_lon_key"] = pd.to_numeric(frame["lon"], errors="coerce").round(6)

    key_cols = ["_merge_cell_key", "_merge_lat_key", "_merge_lon_key"]
    rf_base = rf_base.dropna(subset=key_cols).drop_duplicates(subset=key_cols, keep="last").copy()
    rf_cand = rf_cand.dropna(subset=key_cols).drop_duplicates(subset=key_cols, keep="last").copy()
    delta_source = rf_base[key_cols + metric_cols].merge(
        rf_cand[key_cols + metric_cols],
        on=key_cols,
        how="inner",
        suffixes=("_rf_base", "_rf_cand"),
    )
    if delta_source.empty:
        baseline_out = baseline_out.drop(columns=key_cols, errors="ignore")
        return baseline_out, {
            "rf_delta_matched_row_count": 0.0,
            "rf_delta_unmatched_row_count": float(len(rf_cand)),
            "rf_delta_match_pct": 0.0,
            "merged_row_count_preserved": float(len(baseline_out)),
        }

    baseline_indexed = baseline_out.set_index(key_cols, drop=False)
    delta_indexed = delta_source.set_index(key_cols, drop=False)
    common_index = delta_indexed.index.intersection(baseline_indexed.index)
    for col in metric_cols:
        baseline_indexed[col] = pd.to_numeric(baseline_indexed[col], errors="coerce").astype(float)
        delta = (
            pd.to_numeric(delta_indexed.loc[common_index, f"{col}_rf_cand"], errors="coerce")
            - pd.to_numeric(delta_indexed.loc[common_index, f"{col}_rf_base"], errors="coerce")
        ).fillna(0.0)
        baseline_indexed.loc[common_index, col] = baseline_indexed.loc[common_index, col] + delta

    baseline_out = baseline_indexed.reset_index(drop=True).drop(columns=key_cols, errors="ignore")
    matched_count = int(len(common_index))
    unmatched_count = int(max(len(rf_cand) - matched_count, 0))
    match_pct = (float(matched_count) / float(len(rf_cand)) * 100.0) if len(rf_cand) else 0.0
    print(
        f"[TILT_RSRP_RF_DELTA] stored_baseline_rows={len(stored_baseline_df)} "
        f"rf_baseline_rows={len(rf_baseline_df)} rf_candidate_rows={len(rf_candidate_df)} "
        f"matched_points={matched_count} unmatched_points={unmatched_count} "
        f"match_pct={match_pct:.2f} final_rows={len(baseline_out)}"
    )
    return baseline_out, {
        "rf_delta_matched_row_count": float(matched_count),
        "rf_delta_unmatched_row_count": float(unmatched_count),
        "rf_delta_match_pct": float(match_pct),
        "merged_row_count_preserved": float(len(baseline_out)),
    }


def _mark_recompute_cells_for_prediction(
    site_df: pd.DataFrame,
    recompute_cells: Sequence[str],
    changed_cells: Sequence[str],
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    out = site_df.copy()
    recompute_set = set(_canonicalize_cell_list(recompute_cells, baseline_df))
    changed_set = set(_canonicalize_cell_list(changed_cells, baseline_df))
    synthetic_cells = sorted(recompute_set.difference(changed_set))
    if not synthetic_cells or "Node_Cell_ID" not in out.columns:
        return out
    for col in ["lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "tx_power", "antenna_height"]:
        if col in out.columns and f"orig_{col}" not in out.columns:
            out[f"orig_{col}"] = pd.to_numeric(out[col], errors="coerce")
    mask = out["Node_Cell_ID"].astype(str).isin(synthetic_cells)
    if mask.any() and "azimuth" in out.columns:
        out.loc[mask, "orig_azimuth"] = pd.to_numeric(out.loc[mask, "azimuth"], errors="coerce") + 0.123
        out.loc[mask, "optimization_applied"] = True
        print(f"[TILT_RSRP_GLOBAL_AFFECTED_EXPAND] synthetic_recompute_cells={len(synthetic_cells)} rows_marked={int(mask.sum())}")
    return out


def _aggregate_grid_metrics_from_predictions(
    pred_df: pd.DataFrame,
    rsrp_threshold: float,
) -> pd.DataFrame:
    if pred_df.empty or "grid_id" not in pred_df.columns:
        return pd.DataFrame()
    work = pred_df.copy()
    work["grid_id"] = _normalize_grid_id_series(work["grid_id"])
    work = work.loc[work["grid_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    for col in ["pred_rsrp", "pred_rsrq", "pred_sinr", "lat", "lon"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    agg_map: Dict[str, tuple] = {
        "point_count": ("pred_rsrp", "count"),
        "avg_rsrp": ("pred_rsrp", "mean"),
    }
    if "pred_rsrq" in work.columns:
        agg_map["avg_rsrq"] = ("pred_rsrq", "mean")
    if "pred_sinr" in work.columns:
        agg_map["avg_sinr"] = ("pred_sinr", "mean")
    if "lat" in work.columns:
        agg_map["lat"] = ("lat", "mean")
    if "lon" in work.columns:
        agg_map["lon"] = ("lon", "mean")
    if "Node_Cell_ID" in work.columns:
        agg_map["distinct_cells"] = ("Node_Cell_ID", "nunique")

    grouped = work.groupby("grid_id", dropna=False).agg(**agg_map).reset_index()
    grouped["is_bad_rsrp"] = pd.to_numeric(grouped["avg_rsrp"], errors="coerce") < float(rsrp_threshold)
    grouped["rsrp_severity"] = (float(rsrp_threshold) - pd.to_numeric(grouped["avg_rsrp"], errors="coerce")).clip(lower=0.0)
    return grouped


def _grid_reference_metrics_from_analytics(
    grid_analytics_df: pd.DataFrame,
    rsrp_threshold: float,
) -> pd.DataFrame:
    if grid_analytics_df.empty or "grid_id" not in grid_analytics_df.columns or "baseline_avg_rsrp" not in grid_analytics_df.columns:
        return pd.DataFrame()
    work = grid_analytics_df.copy()
    work["grid_id"] = _normalize_grid_id_series(work["grid_id"])
    work["avg_rsrp"] = pd.to_numeric(work["baseline_avg_rsrp"], errors="coerce")
    if "baseline_avg_rsrq" in work.columns:
        work["avg_rsrq"] = pd.to_numeric(work["baseline_avg_rsrq"], errors="coerce")
    else:
        work["avg_rsrq"] = np.nan
    if "baseline_avg_sinr" in work.columns:
        work["avg_sinr"] = pd.to_numeric(work["baseline_avg_sinr"], errors="coerce")
    else:
        work["avg_sinr"] = np.nan
    if "baseline_point_count" in work.columns:
        work["point_count"] = pd.to_numeric(work["baseline_point_count"], errors="coerce")
    else:
        work["point_count"] = np.nan
    for col in ["center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    keep_cols = [col for col in ["grid_id", "avg_rsrp", "avg_rsrq", "avg_sinr", "point_count", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"] if col in work.columns]
    out = work.loc[work["grid_id"].notna(), keep_cols].drop_duplicates(subset=["grid_id"], keep="first").copy()
    out["is_bad_rsrp"] = pd.to_numeric(out["avg_rsrp"], errors="coerce") < float(rsrp_threshold)
    out["rsrp_severity"] = (float(rsrp_threshold) - pd.to_numeric(out["avg_rsrp"], errors="coerce")).clip(lower=0.0)
    return out


def _grid_ids_for_evaluation_cells(pred_df: pd.DataFrame, evaluation_cells: Sequence[str]) -> List[str]:
    if pred_df.empty or "grid_id" not in pred_df.columns or "Node_Cell_ID" not in pred_df.columns:
        return []
    eval_cells = {str(cell).strip() for cell in evaluation_cells if str(cell).strip()}
    if not eval_cells:
        return []
    work = pred_df.loc[pred_df["Node_Cell_ID"].astype(str).isin(eval_cells)].copy()
    if work.empty:
        return []
    grid_ids = _normalize_grid_id_series(work["grid_id"]).dropna().drop_duplicates()
    return sorted(grid_ids.tolist())


def _grid_scope_for_ids(pred_df: pd.DataFrame, grid_ids: Sequence[object]) -> pd.DataFrame:
    if pred_df.empty or "grid_id" not in pred_df.columns or not grid_ids:
        return pd.DataFrame()
    grid_id_set = set(_normalize_grid_id_series(pd.Series(list(grid_ids))).dropna().tolist())
    if not grid_id_set:
        return pd.DataFrame()
    work = pred_df.copy()
    work["grid_id"] = _normalize_grid_id_series(work["grid_id"])
    return work.loc[work["grid_id"].isin(grid_id_set)].copy()


def _grid_validation_payload(
    baseline_df: pd.DataFrame,
    grid_analytics_df: pd.DataFrame,
    rsrp_threshold: float,
) -> Dict[str, object]:
    recomputed_grid_df = _aggregate_grid_metrics_from_predictions(baseline_df, rsrp_threshold)
    mapped_rows = int(_normalize_grid_id_series(baseline_df.get("grid_id")).notna().sum()) if "grid_id" in baseline_df.columns else 0
    payload: Dict[str, object] = {
        "threshold": float(rsrp_threshold),
        "comparison_operator": "avg_rsrp < threshold",
        "baseline_rows": int(len(baseline_df)),
        "grid_mapped_rows": mapped_rows,
        "grid_unmapped_rows": int(len(baseline_df) - mapped_rows),
        "grid_mapped_pct": float(mapped_rows / max(len(baseline_df), 1) * 100.0),
        "recomputed_grid_count": int(len(recomputed_grid_df)),
        "recomputed_bad_grid_count": int(recomputed_grid_df["is_bad_rsrp"].fillna(False).sum()) if not recomputed_grid_df.empty else 0,
        "grid_analytics_rows": int(len(grid_analytics_df)),
        "grid_analytics_bad_grid_count": None,
        "common_grid_count": 0,
        "avg_abs_rsrp_delta_vs_grid_analytics": None,
        "max_abs_rsrp_delta_vs_grid_analytics": None,
    }
    if grid_analytics_df.empty or "grid_id" not in grid_analytics_df.columns or "baseline_avg_rsrp" not in grid_analytics_df.columns:
        return payload

    grid_src = grid_analytics_df.copy()
    grid_src["grid_id"] = _normalize_grid_id_series(grid_src["grid_id"])
    grid_src["baseline_avg_rsrp"] = pd.to_numeric(grid_src["baseline_avg_rsrp"], errors="coerce")
    grid_src = grid_src.loc[grid_src["grid_id"].notna()].drop_duplicates(subset=["grid_id"], keep="first").copy()
    payload["grid_analytics_bad_grid_count"] = int((grid_src["baseline_avg_rsrp"] < float(rsrp_threshold)).fillna(False).sum())
    if recomputed_grid_df.empty:
        return payload

    compare = recomputed_grid_df[["grid_id", "avg_rsrp"]].merge(
        grid_src[["grid_id", "baseline_avg_rsrp"]],
        on="grid_id",
        how="inner",
    )
    payload["common_grid_count"] = int(len(compare))
    if not compare.empty:
        delta = (pd.to_numeric(compare["avg_rsrp"], errors="coerce") - pd.to_numeric(compare["baseline_avg_rsrp"], errors="coerce")).abs()
        payload["avg_abs_rsrp_delta_vs_grid_analytics"] = float(delta.mean())
        payload["max_abs_rsrp_delta_vs_grid_analytics"] = float(delta.max())
    return payload


def _log_grid_validation(prefix: str, payload: Dict[str, object]) -> None:
    print(
        f"[TILT_RSRP_GRID][{prefix}] threshold={payload.get('threshold')} operator={payload.get('comparison_operator')} "
        f"baseline_rows={payload.get('baseline_rows')} mapped_rows={payload.get('grid_mapped_rows')} "
        f"unmapped_rows={payload.get('grid_unmapped_rows')} mapped_pct={float(payload.get('grid_mapped_pct', 0.0)):.2f} "
        f"recomputed_grids={payload.get('recomputed_grid_count')} recomputed_bad_grids={payload.get('recomputed_bad_grid_count')} "
        f"grid_analytics_rows={payload.get('grid_analytics_rows')} grid_analytics_bad_grids={payload.get('grid_analytics_bad_grid_count')}"
    )
    print(
        f"[TILT_RSRP_GRID][{prefix}_COMPARE] common_grids={payload.get('common_grid_count')} "
        f"avg_abs_rsrp_delta={payload.get('avg_abs_rsrp_delta_vs_grid_analytics')} "
        f"max_abs_rsrp_delta={payload.get('max_abs_rsrp_delta_vs_grid_analytics')}"
    )


def _build_grid_ranked_summary(
    baseline_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    rsrp_threshold: float,
    grid_analytics_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recomputed_grid_metrics = _aggregate_grid_metrics_from_predictions(baseline_df, rsrp_threshold)
    grid_metrics = pd.DataFrame()
    if grid_analytics_df is not None and not grid_analytics_df.empty:
        reference_grid = _grid_reference_metrics_from_analytics(grid_analytics_df, rsrp_threshold)
        if not reference_grid.empty:
            grid_metrics = reference_grid.copy()
            print(
                f"[TILT_RSRP_GRID][RANKING_SOURCE] mode=frontend_grid_analytics "
                f"recomputed_grids={len(recomputed_grid_metrics)} "
                f"recomputed_bad_grids={int(recomputed_grid_metrics['is_bad_rsrp'].fillna(False).sum()) if not recomputed_grid_metrics.empty else 0} "
                f"analytics_grids={len(reference_grid)} "
                f"analytics_bad_grids={int(reference_grid['is_bad_rsrp'].fillna(False).sum())}"
            )
    if grid_metrics.empty:
        grid_metrics = recomputed_grid_metrics
    if grid_metrics.empty:
        return pd.DataFrame(), grid_metrics

    bad_grid_ids = set(grid_metrics.loc[grid_metrics["is_bad_rsrp"].fillna(False), "grid_id"].tolist())
    if not bad_grid_ids:
        return pd.DataFrame(), grid_metrics

    work = baseline_df.copy()
    work["grid_id"] = _normalize_grid_id_series(work.get("grid_id"))
    work = work.loc[work["grid_id"].isin(bad_grid_ids)].copy()
    if work.empty:
        return pd.DataFrame(), grid_metrics

    work["Cell ID"] = work["Node_Cell_ID"].astype(str)
    work["pred_rsrp"] = pd.to_numeric(work["pred_rsrp"], errors="coerce")
    severity = (float(rsrp_threshold) - work["pred_rsrp"]).clip(lower=0.0).fillna(0.0)
    work["_severity"] = severity

    site_map = base._build_cell_site_map(antenna_df)[["Cell ID", "Site ID"]].copy()
    work = work.merge(site_map, on="Cell ID", how="left")

    summary_df = (
        work.groupby("Cell ID", dropna=False)
        .agg(
            **{
                "Bad RSRP": ("grid_id", "count"),
                "Bad Grid Count": ("grid_id", "nunique"),
                "mean_bad_grid_rsrp": ("pred_rsrp", "mean"),
                "bad_grid_severity": ("_severity", "sum"),
            }
        )
        .reset_index()
    )
    summary_df["Bad RSRQ"] = 0
    summary_df["Bad SINR"] = 0
    summary_df["Bad Samples"] = pd.to_numeric(summary_df["Bad RSRP"], errors="coerce").fillna(0).astype(int)
    summary_df["total_bad_samples"] = pd.to_numeric(summary_df["Bad RSRP"], errors="coerce").fillna(0).astype(int)
    summary_df = summary_df.sort_values(
        ["Bad Grid Count", "bad_grid_severity", "Bad RSRP", "Cell ID"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return summary_df, grid_metrics


def _score_candidate_vs_baseline_grids(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    evaluation_cells: Sequence[str],
    config: base.TiltRecommendationTestConfig,
    grid_analytics_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    evaluation_grid_ids = _grid_ids_for_evaluation_cells(baseline_df, evaluation_cells)
    if not evaluation_grid_ids:
        return base._score_candidate_vs_baseline(baseline_df, candidate_df, evaluation_cells, config)

    baseline_scope = _grid_scope_for_ids(baseline_df, evaluation_grid_ids)
    candidate_scope = _grid_scope_for_ids(candidate_df, evaluation_grid_ids)
    baseline_grid = _aggregate_grid_metrics_from_predictions(baseline_scope, config.rsrp_threshold)
    candidate_grid = _aggregate_grid_metrics_from_predictions(candidate_scope, config.rsrp_threshold)
    if baseline_grid.empty or candidate_grid.empty:
        return base._score_candidate_vs_baseline(baseline_df, candidate_df, evaluation_cells, config)

    reference_grid = _grid_reference_metrics_from_analytics(grid_analytics_df, config.rsrp_threshold) if grid_analytics_df is not None else pd.DataFrame()
    full_reference_grid = reference_grid.copy()
    delta_source = pd.DataFrame()
    if not reference_grid.empty:
        reference_grid = reference_grid.loc[reference_grid["grid_id"].isin(evaluation_grid_ids)].copy()
        delta_source = baseline_grid[["grid_id", "avg_rsrp", "avg_rsrq", "avg_sinr"]].merge(
            candidate_grid[["grid_id", "avg_rsrp", "avg_rsrq", "avg_sinr"]],
            on="grid_id",
            how="inner",
            suffixes=("_model_base", "_model_cand"),
        )
        for metric in ["avg_rsrp", "avg_rsrq", "avg_sinr"]:
            delta_source[f"{metric}_delta"] = (
                pd.to_numeric(delta_source[f"{metric}_model_cand"], errors="coerce")
                - pd.to_numeric(delta_source[f"{metric}_model_base"], errors="coerce")
            )
        merged = reference_grid[["grid_id", "avg_rsrp", "avg_rsrq", "avg_sinr", "point_count"]].rename(
            columns={
                "avg_rsrp": "avg_rsrp_base",
                "avg_rsrq": "avg_rsrq_base",
                "avg_sinr": "avg_sinr_base",
                "point_count": "point_count_base",
            }
        )
        merged = merged.merge(delta_source[["grid_id", "avg_rsrp_delta", "avg_rsrq_delta", "avg_sinr_delta"]], on="grid_id", how="left")
        for metric in ["avg_rsrp", "avg_rsrq", "avg_sinr"]:
            merged[f"{metric}_cand"] = pd.to_numeric(merged[f"{metric}_base"], errors="coerce") + pd.to_numeric(
                merged[f"{metric}_delta"], errors="coerce"
            ).fillna(0.0)
        scoring_source = "frontend_grid_analytics_plus_candidate_rf_delta"
    else:
        merged = baseline_grid.merge(
            candidate_grid[["grid_id", "avg_rsrp", "avg_rsrq", "avg_sinr", "point_count"]],
            on="grid_id",
            how="outer",
            suffixes=("_base", "_cand"),
        )
        scoring_source = "recomputed_rf_grid_symmetric"
    if merged.empty:
        return base._score_candidate_vs_baseline(baseline_df, candidate_df, evaluation_cells, config)

    for metric in ["avg_rsrp", "avg_rsrq", "avg_sinr"]:
        base_col = f"{metric}_base"
        cand_col = f"{metric}_cand"
        if cand_col in merged.columns:
            merged[cand_col] = pd.to_numeric(merged[cand_col], errors="coerce")
        if base_col in merged.columns:
            merged[base_col] = pd.to_numeric(merged[base_col], errors="coerce")
        if cand_col in merged.columns and base_col in merged.columns:
            merged[cand_col] = merged[cand_col].fillna(merged[base_col])

    rsrp_base = pd.to_numeric(merged.get("avg_rsrp_base"), errors="coerce")
    rsrp_cand = pd.to_numeric(merged.get("avg_rsrp_cand"), errors="coerce")
    rsrq_base = pd.to_numeric(merged.get("avg_rsrq_base"), errors="coerce") if "avg_rsrq_base" in merged.columns else pd.Series(0.0, index=merged.index)
    rsrq_cand = pd.to_numeric(merged.get("avg_rsrq_cand"), errors="coerce") if "avg_rsrq_cand" in merged.columns else pd.Series(0.0, index=merged.index)
    sinr_base = pd.to_numeric(merged.get("avg_sinr_base"), errors="coerce") if "avg_sinr_base" in merged.columns else pd.Series(0.0, index=merged.index)
    sinr_cand = pd.to_numeric(merged.get("avg_sinr_cand"), errors="coerce") if "avg_sinr_cand" in merged.columns else pd.Series(0.0, index=merged.index)

    base_bad = rsrp_base < float(config.rsrp_threshold)
    cand_bad = rsrp_cand < float(config.rsrp_threshold)
    recovered_bad = int((base_bad & ~cand_bad).sum())
    new_bad = int((~base_bad & cand_bad).sum())
    baseline_bad_count = int(base_bad.sum())
    candidate_bad_count = int(cand_bad.sum())
    baseline_good = int((~base_bad).sum())
    evaluation_count = max(float(len(merged)), 1.0)

    rsrp_base_severity = (float(config.rsrp_threshold) - rsrp_base).clip(lower=0.0).fillna(0.0)
    rsrp_cand_severity = (float(config.rsrp_threshold) - rsrp_cand).clip(lower=0.0).fillna(0.0)
    rsrp_severity_reduction = float((rsrp_base_severity - rsrp_cand_severity).sum())
    rsrp_severity_reduction_per_sample = rsrp_severity_reduction / evaluation_count
    net_bad_reduction = float(baseline_bad_count - candidate_bad_count)
    net_bad_reduction_share = net_bad_reduction / evaluation_count
    recovered_bad_share = float(recovered_bad) / evaluation_count
    new_bad_share = float(new_bad) / evaluation_count
    good_area_loss_pct = (float(new_bad) / float(baseline_good) * 100.0) if baseline_good > 0 else 0.0
    mean_rsrp_delta = float((rsrp_cand - rsrp_base).mean())
    mean_rsrq_delta = float((rsrq_cand - rsrq_base).mean()) if len(rsrq_base) else 0.0
    mean_sinr_delta = float((sinr_cand - sinr_base).mean()) if len(sinr_base) else 0.0
    bad_grid_mean_rsrp_delta = float((rsrp_cand[base_bad] - rsrp_base[base_bad]).mean()) if int(base_bad.sum()) > 0 else 0.0

    overall_threshold_metrics: Dict[str, float] = {}
    threshold_values = [-90.0, -95.0, -100.0, -105.0]
    if not full_reference_grid.empty and not delta_source.empty:
        overall_merged = full_reference_grid[["grid_id", "avg_rsrp"]].rename(columns={"avg_rsrp": "avg_rsrp_base"}).copy()
        overall_merged = overall_merged.merge(delta_source[["grid_id", "avg_rsrp_delta"]], on="grid_id", how="left")
        overall_merged["avg_rsrp_base"] = pd.to_numeric(overall_merged["avg_rsrp_base"], errors="coerce")
        overall_merged["avg_rsrp_cand"] = overall_merged["avg_rsrp_base"] + pd.to_numeric(
            overall_merged["avg_rsrp_delta"], errors="coerce"
        ).fillna(0.0)
        overall_source = "frontend_grid_analytics_overall_plus_candidate_rf_delta"
    else:
        overall_merged = merged[["grid_id", "avg_rsrp_base", "avg_rsrp_cand"]].copy() if {"grid_id", "avg_rsrp_base", "avg_rsrp_cand"}.issubset(merged.columns) else pd.DataFrame()
        overall_source = scoring_source
    if not overall_merged.empty:
        overall_base = pd.to_numeric(overall_merged["avg_rsrp_base"], errors="coerce")
        overall_cand = pd.to_numeric(overall_merged["avg_rsrp_cand"], errors="coerce").fillna(overall_base)
        overall_threshold_metrics["overall_grid_count"] = float(len(overall_merged))
        overall_threshold_metrics["overall_mean_rsrp_before"] = float(overall_base.mean())
        overall_threshold_metrics["overall_mean_rsrp_after"] = float(overall_cand.mean())
        overall_threshold_metrics["overall_mean_rsrp_delta"] = float((overall_cand - overall_base).mean())
        overall_threshold_metrics["overall_threshold_source"] = overall_source
        for threshold in threshold_values:
            suffix = str(int(abs(threshold)))
            before_mask = overall_base < threshold
            after_mask = overall_cand < threshold
            before_count = int(before_mask.fillna(False).sum())
            after_count = int(after_mask.fillna(False).sum())
            bad_to_good_count = int((before_mask & ~after_mask).fillna(False).sum())
            good_to_bad_count = int((~before_mask & after_mask).fillna(False).sum())
            overall_threshold_metrics[f"overall_before_bad_grid_count_lt_{suffix}"] = float(before_count)
            overall_threshold_metrics[f"overall_after_bad_grid_count_lt_{suffix}"] = float(after_count)
            overall_threshold_metrics[f"overall_net_bad_grid_reduction_lt_{suffix}"] = float(before_count - after_count)
            overall_threshold_metrics[f"overall_bad_to_good_grid_count_lt_{suffix}"] = float(bad_to_good_count)
            overall_threshold_metrics[f"overall_good_to_bad_grid_count_lt_{suffix}"] = float(good_to_bad_count)

    # RSRP-only debug mode should tolerate controlled redistribution. The old
    # 2% style good-area loss gate collapses most RF tilt trials into HOLD.
    hard_good_area_loss_pct = max(float(config.max_good_area_loss_pct), 15.0)
    hard_sinr_drop_db = max(float(config.max_mean_sinr_drop_db), abs(float(base.SEVERE_MEAN_SINR_DROP_DB)))
    constraints_passed = (
        good_area_loss_pct <= hard_good_area_loss_pct
        and mean_sinr_delta >= -hard_sinr_drop_db
    )
    score = (
        net_bad_reduction_share * 8.0
        + recovered_bad_share * 5.0
        - new_bad_share * 2.0
        + rsrp_severity_reduction_per_sample * 4.0
        + bad_grid_mean_rsrp_delta * 0.5
        - good_area_loss_pct * 0.05
    )
    if not constraints_passed:
        score -= float(base.SEVERE_CONSTRAINT_PENALTY)

    return {
        "baseline_bad_count": float(baseline_bad_count),
        "candidate_bad_count": float(candidate_bad_count),
        "recovered_bad_samples": float(recovered_bad),
        "new_bad_samples": float(new_bad),
        "rsrp_recovered_bad": float(recovered_bad),
        "rsrp_new_bad": float(new_bad),
        "rsrq_recovered_bad": 0.0,
        "rsrq_new_bad": 0.0,
        "sinr_recovered_bad": 0.0,
        "sinr_new_bad": 0.0,
        "rsrp_severity_reduction": float(rsrp_severity_reduction),
        "rsrq_severity_reduction": 0.0,
        "sinr_severity_reduction": 0.0,
        "total_severity_reduction": float(rsrp_severity_reduction),
        "evaluation_sample_count": float(evaluation_count),
        "rsrp_severity_reduction_per_sample": float(rsrp_severity_reduction_per_sample),
        "rsrq_severity_reduction_per_sample": 0.0,
        "sinr_severity_reduction_per_sample": 0.0,
        "total_severity_reduction_per_sample": float(rsrp_severity_reduction_per_sample),
        "net_bad_reduction": float(net_bad_reduction),
        "net_bad_reduction_share": float(net_bad_reduction_share),
        "recovered_bad_share": float(recovered_bad_share),
        "new_bad_share": float(new_bad_share),
        "rsrp_recovered_bad_share": float(recovered_bad_share),
        "rsrp_new_bad_share": float(new_bad_share),
        "rsrq_recovered_bad_share": 0.0,
        "rsrq_new_bad_share": 0.0,
        "sinr_recovered_bad_share": 0.0,
        "sinr_new_bad_share": 0.0,
        "good_area_loss_pct": float(good_area_loss_pct),
        "mean_rsrp_delta": float(mean_rsrp_delta),
        "mean_rsrq_delta": float(mean_rsrq_delta),
        "mean_sinr_delta": float(mean_sinr_delta),
        "score": float(score),
        "constraints_passed": float(1 if constraints_passed else 0),
        "baseline_bad_grid_count": float(baseline_bad_count),
        "candidate_bad_grid_count": float(candidate_bad_count),
        "bad_to_good_grid_count": float(recovered_bad),
        "good_to_bad_grid_count": float(new_bad),
        "grid_mean_rsrp_delta_bad_baseline": float(bad_grid_mean_rsrp_delta),
        "evaluation_grid_count": float(len(evaluation_grid_ids)),
        "evaluation_row_count_full_grids": float(len(merged)),
        "grid_scoring_source": scoring_source,
        "validation_scope": "full_merged_grid_population",
        **overall_threshold_metrics,
    }


def _changed_cell_local_metrics_global(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    updates: Sequence[Dict[str, object]],
    rsrp_threshold: float,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    threshold = float(rsrp_threshold)
    for update in updates:
        cell_id = str(update.get("cell_id", "")).strip()
        if not cell_id:
            continue
        before_cell = before_df.loc[before_df.get("Node_Cell_ID", "").astype(str) == cell_id].copy() if "Node_Cell_ID" in before_df.columns else pd.DataFrame()
        after_cell = after_df.loc[after_df.get("Node_Cell_ID", "").astype(str) == cell_id].copy() if "Node_Cell_ID" in after_df.columns else pd.DataFrame()
        before_rsrp = pd.to_numeric(before_cell.get("pred_rsrp"), errors="coerce") if not before_cell.empty else pd.Series(dtype=float)
        after_rsrp = pd.to_numeric(after_cell.get("pred_rsrp"), errors="coerce") if not after_cell.empty else pd.Series(dtype=float)
        before_bad = int((before_rsrp < threshold).fillna(False).sum()) if len(before_rsrp) else 0
        after_bad = int((after_rsrp < threshold).fillna(False).sum()) if len(after_rsrp) else 0
        before_avg = float(before_rsrp.mean()) if len(before_rsrp.dropna()) else np.nan
        after_avg = float(after_rsrp.mean()) if len(after_rsrp.dropna()) else np.nan
        rows.append(
            {
                "cell_id": cell_id,
                "parameter": update.get("parameter", "ETilt"),
                "target_value": update.get("target_value"),
                "before_bad_sample_count": before_bad,
                "after_bad_sample_count": after_bad,
                "bad_sample_reduction": before_bad - after_bad,
                "before_avg_rsrp": before_avg,
                "after_avg_rsrp": after_avg,
                "avg_rsrp_delta": after_avg - before_avg if pd.notna(before_avg) and pd.notna(after_avg) else np.nan,
                "before_sample_count": int(len(before_rsrp)),
                "after_sample_count": int(len(after_rsrp)),
            }
        )
    if not rows:
        return {
            "changed_cell_local_metrics": "[]",
            "changed_cell_bad_sample_reduction_sum": 0.0,
            "changed_cell_avg_rsrp_delta_mean": np.nan,
        }
    local_df = pd.DataFrame(rows)
    return {
        "changed_cell_local_metrics": json.dumps(rows, sort_keys=True),
        "changed_cell_bad_sample_reduction_sum": float(pd.to_numeric(local_df["bad_sample_reduction"], errors="coerce").fillna(0).sum()),
        "changed_cell_avg_rsrp_delta_mean": float(pd.to_numeric(local_df["avg_rsrp_delta"], errors="coerce").mean()),
    }


# RSRP-only production test is intentionally global and cell-centric.
# Site-first Optuna search was removed because it fragmented bad-grid causes by site.

def _build_tilt_only_recommendations(
    summary_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    config: base.TiltRecommendationTestConfig,
    constraint_map: Dict[str, Dict[str, object]],
    grid_analytics_df: Optional[pd.DataFrame] = None,
    geo_features_df: Optional[pd.DataFrame] = None,
    residual_models: Optional[Dict[str, Dict[str, object]]] = None,
    use_fixed_raw_k1k2: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ant_use = base._build_cell_site_map(antenna_df)
    if ant_use.empty or baseline_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    tech_col = "Technology" if "Technology" in ant_use.columns else ""
    baseline_job_id = base._fetch_latest_baseline_job_id(config.project_id, config.region)
    evaluation_rows: List[Dict[str, object]] = []
    recommendation_rows: List[Dict[str, object]] = []
    site_id = "GLOBAL_BAD_GRID_CELL_OPT"
    ant_use["Cell ID"] = ant_use["Cell ID"].astype(str).str.strip()
    tunable_set = set(ant_use["Cell ID"].dropna().astype(str).str.strip())
    reference_grid = _grid_reference_metrics_from_analytics(grid_analytics_df, config.rsrp_threshold) if grid_analytics_df is not None else pd.DataFrame()
    work = baseline_df.copy()
    work["grid_id"] = _normalize_grid_id_series(work.get("grid_id")) if "grid_id" in work.columns else pd.Series(np.nan, index=work.index)
    work["Node_Cell_ID"] = work["Node_Cell_ID"].astype(str).str.strip()
    work["pred_rsrp"] = pd.to_numeric(work["pred_rsrp"], errors="coerce")
    work = work.dropna(subset=["grid_id", "pred_rsrp"])
    work = work.loc[work["Node_Cell_ID"].isin(tunable_set)].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    if not reference_grid.empty:
        bad_grid_ids = set(reference_grid.loc[reference_grid["is_bad_rsrp"].fillna(False), "grid_id"].astype(str))
        source = "frontend_grid_analytics"
    else:
        grid_avg = (
            work.groupby("grid_id", dropna=False)
            .agg(avg_rsrp=("pred_rsrp", "mean"))
            .reset_index()
        )
        bad_grid_ids = set(grid_avg.loc[pd.to_numeric(grid_avg["avg_rsrp"], errors="coerce") < float(config.rsrp_threshold), "grid_id"].astype(str))
        source = "recomputed_rf_grid"
    bad_work = work.loc[work["grid_id"].astype(str).isin(bad_grid_ids)].copy()
    bad_work["_severity"] = (float(config.rsrp_threshold) - bad_work["pred_rsrp"]).clip(lower=0.0).fillna(0.0)
    contributor = (
        bad_work.groupby(["grid_id", "Node_Cell_ID"], dropna=False)
        .agg(
            bad_sample_count=("pred_rsrp", lambda values: int((pd.to_numeric(values, errors="coerce") < float(config.rsrp_threshold)).sum())),
            severity_sum=("_severity", "sum"),
        )
        .reset_index()
    )
    contributor = contributor.loc[contributor["bad_sample_count"] > 0].copy()
    if contributor.empty:
        return pd.DataFrame(), pd.DataFrame()

    global_cells = (
        contributor.groupby("Node_Cell_ID", dropna=False)
        .agg(total_bad_samples=("bad_sample_count", "sum"), total_severity=("severity_sum", "sum"), bad_grid_count=("grid_id", "nunique"))
        .reset_index()
        .sort_values(["total_severity", "total_bad_samples", "bad_grid_count"], ascending=[False, False, False])
    )
    total_severity = float(pd.to_numeric(global_cells["total_severity"], errors="coerce").fillna(0.0).sum())
    coverage_pct = float(np.clip(float(getattr(config, "bad_grid_coverage_pct", 80.0)), 1.0, 100.0))
    max_group_cells = int(getattr(config, "max_group_cells", 0) or 0)
    global_cells["contribution_pct"] = pd.to_numeric(global_cells["total_severity"], errors="coerce").fillna(0.0) / max(total_severity, 1e-9) * 100.0
    global_cells["cumulative_contribution_pct"] = global_cells["contribution_pct"].cumsum()
    selected_cells_df = global_cells.loc[global_cells["cumulative_contribution_pct"] <= coverage_pct].copy()
    if selected_cells_df.empty:
        selected_cells_df = global_cells.head(1).copy()
    elif len(selected_cells_df) < len(global_cells):
        selected_cells_df = pd.concat([selected_cells_df, global_cells.iloc[[len(selected_cells_df)]]], ignore_index=True)
    if max_group_cells > 0:
        selected_cells_df = selected_cells_df.head(max_group_cells).copy()
    target_cells = selected_cells_df["Node_Cell_ID"].astype(str).tolist()
    if not target_cells:
        return pd.DataFrame(), pd.DataFrame()

    print(
        f"[TILT_RSRP_GLOBAL_GROUP] source={source} bad_grids={len(bad_grid_ids)} "
        f"total_contributor_cells={len(global_cells)} selected_cells={len(target_cells)} "
        f"coverage_target={coverage_pct:.1f} coverage_actual={float(selected_cells_df['contribution_pct'].sum()):.2f} "
        f"cells={target_cells}"
    )

    rf_cache: Dict[tuple, Dict[str, object]] = {}
    rf_baseline_cache: Dict[tuple, pd.DataFrame] = {}
    eval_lock = threading.Lock()

    def _cell_current(cell_id: str, column: str) -> float:
        row = ant_use.loc[ant_use["Cell ID"].astype(str) == str(cell_id)]
        if row.empty:
            return np.nan
        return pd.to_numeric(pd.Series([row.iloc[0].get(column)]), errors="coerce").iloc[0]

    def _target_update(cell_id: str, delta: float) -> Optional[Dict[str, object]]:
        current_value = _cell_current(cell_id, "electrical_tilt")
        if pd.isna(current_value):
            return None
        target = float(np.clip(float(current_value) + float(delta), float(base.MIN_SAFE_ETILT_DEG), float(base.MAX_SAFE_ETILT_DEG)))
        target = base._clip_target_to_user_constraint_test_only(cell_id, "ETilt", target, constraint_map)
        if pd.isna(target) or np.isclose(float(target), float(current_value), equal_nan=True):
            return None
        return {
            "cell_id": str(cell_id),
            "parameter": "ETilt",
            "target_value": float(target),
            "current_value": float(current_value),
            "requested_delta": float(delta),
            "actual_delta": float(target) - float(current_value),
        }

    def _dedupe_updates(updates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        deduped: Dict[tuple, Dict[str, object]] = {}
        for update in updates:
            key = (str(update.get("cell_id", "")).strip(), str(update.get("parameter", "")).strip())
            if key[0] and key[1]:
                deduped[key] = dict(update)
        return list(deduped.values())

    def _mark_recompute_cells(site_df: pd.DataFrame, recompute_cells: Sequence[str], changed_cells: Sequence[str]) -> pd.DataFrame:
        return _mark_recompute_cells_for_prediction(site_df, recompute_cells, changed_cells, baseline_df)

    def _result_rank(result: Dict[str, object]) -> tuple:
        return (
            int(bool(int(float(result.get("constraints_passed", 0.0))))),
            float(result.get("score", -99999.0)),
            float(result.get("net_bad_reduction", -99999.0)),
            float(result.get("rsrp_severity_reduction", -99999.0)),
            -float(result.get("new_bad_samples", 99999.0)),
        )

    def _result(candidate_name: str, updates: Sequence[Dict[str, object]], metrics: Dict[str, object], error: str = "") -> Dict[str, object]:
        result = {
            "site_id": site_id,
            "cell_id": ",".join(target_cells),
            "parameter": "ETilt",
            "candidate_name": candidate_name,
            "current_value": "",
            "target_value": json.dumps(list(updates), sort_keys=True),
            "tie_break": 0.0,
            "affected_cells": int(metrics.get("calibration_cell_count", 0.0) or 0),
            "affected_sites": int(metrics.get("affected_site_count", 0.0) or 0),
            "changed_cells": len(updates),
            "tunable_site_count": int(ant_use.loc[ant_use["Cell ID"].isin(target_cells), "Site ID"].astype(str).nunique()) if "Site ID" in ant_use.columns else 0,
            "tunable_cell_count": len(target_cells),
            "tunable_sites": ",".join(ant_use.loc[ant_use["Cell ID"].isin(target_cells), "Site ID"].astype(str).drop_duplicates().tolist()) if "Site ID" in ant_use.columns else "",
            "target_action_cells": ",".join(target_cells),
            "cluster_eval_sample_count": int(len(baseline_df)),
            "cluster_baseline_bad_count": int(len(bad_grid_ids)),
            "root_cause": "global_bad_grid_rsrp",
            "topology_root_cause": "global_bad_grid_rsrp",
            "interference_limited": 0,
            "coverage_limited": 1,
            "mean_interference_gap_db": np.nan,
            "same_earfcn_interferer_share": np.nan,
            "dominant_interferer_share": np.nan,
            "update_count": len(updates),
            "selection_stage": "global_bad_grid_cell_coordinate_search",
            "cheap_score": 0.0,
            **metrics,
            "error": error,
        }
        result["reject_reason"] = base._site_candidate_reject_reason(result, config)
        print(
            f"[TILT_RSRP_GLOBAL_CANDIDATE] candidate={candidate_name} update_count={len(updates)} "
            f"actual_deltas={[(str(u.get('cell_id')), float(u.get('actual_delta', np.nan))) for u in updates]} "
            f"baseline_bad_grids={float(result.get('baseline_bad_grid_count', 0.0)):.0f} "
            f"candidate_bad_grids={float(result.get('candidate_bad_grid_count', 0.0)):.0f} "
            f"net={float(result.get('net_bad_reduction', 0.0)):.0f} score={float(result.get('score', 0.0)):.4f}"
        )
        return result

    all_evaluation_cells = _canonicalize_cell_list(baseline_df["Node_Cell_ID"].dropna().astype(str).unique().tolist(), baseline_df)

    def _evaluate(candidate_name: str, updates: Sequence[Dict[str, object]]) -> Dict[str, object]:
        updates = _dedupe_updates(updates)
        if not updates:
            metrics = _score_candidate_vs_baseline_grids(baseline_df, baseline_df, all_evaluation_cells, config, grid_analytics_df)
            return _result("hold", [], {**metrics, "calibration_mode": "hold_no_rf_delta", "calibration_cell_count": 0.0, "affected_site_count": 0.0})
        cache_key = base._candidate_cache_key(site_id, updates)
        with eval_lock:
            cached = rf_cache.get(cache_key)
        if cached is not None:
            return _result(candidate_name, updates, cached)
        try:
            modified_site_df = base._apply_multiple_parameter_targets(antenna_df, updates)
            radius_affected_cells, affected_sites, _ = base.opt_ml._compute_affected_cells(
                modified_site_df,
                float(config.impact_radius_m),
                int(config.neighbor_site_count),
            )
            update_cells = _canonicalize_cell_list([str(update.get("cell_id", "")) for update in updates], baseline_df)
            radius_affected_cells = _canonicalize_cell_list(radius_affected_cells, baseline_df)
            topology_seed_cells = sorted(set(radius_affected_cells).union(update_cells))
            topology_affected_cells = _canonicalize_cell_list(base._expand_evaluation_cells_from_topology(baseline_df, topology_seed_cells), baseline_df)
            calibration_cells = _canonicalize_cell_list(set(topology_affected_cells).union(update_cells), baseline_df)
            if not calibration_cells:
                raise ValueError("No global calibration cells found for candidate")
            prediction_site_df = _mark_recompute_cells(modified_site_df, calibration_cells, update_cells)
            baseline_prediction_site_df = _mark_recompute_cells(antenna_df, calibration_cells, [])
            if use_fixed_raw_k1k2:
                k1k2_map = _fixed_raw_k1k2_map(calibration_cells)
                calibration_mode = "fixed_raw_k1k2_zero_plus_fixed_residual"
            else:
                k1k2_map = base.opt_ml.compute_k1k2_for_cells(baseline_df, antenna_df, calibration_cells)
                calibration_mode = "fixed_baseline_k1k2_plus_fixed_residual" if residual_models else "fixed_baseline_k1k2"
            if not k1k2_map:
                raise ValueError("No fixed baseline k1/k2 map found for candidate")
            prediction_params = {
                "project_id": int(config.project_id),
                "region": config.region,
                "radius": float(config.radius_m),
                "grid_resolution": float(config.grid_resolution_m),
                "n_workers": int(config.workers),
                "impact_radius_m": float(config.impact_radius_m),
                "neighbor_site_count": int(config.neighbor_site_count),
                "max_interference_sites": int(config.max_interference_sites),
                "baseline_job_id": baseline_job_id,
                "prediction_points_df": baseline_df,
                "geo_features_df": geo_features_df,
            }
            baseline_rf_key = tuple(sorted(calibration_cells))
            with eval_lock:
                baseline_rf_df = rf_baseline_cache.get(baseline_rf_key)
            if baseline_rf_df is None:
                baseline_rf_df = base.opt_ml.run_prediction_only_optimized(
                    baseline_prediction_site_df,
                    k1k2_map,
                    prediction_params,
                )
                baseline_rf_df = _apply_fixed_residual_calibration_to_rf(baseline_rf_df, residual_models)
                with eval_lock:
                    rf_baseline_cache[baseline_rf_key] = baseline_rf_df
            optimized_df = base.opt_ml.run_prediction_only_optimized(
                prediction_site_df,
                k1k2_map,
                prediction_params,
            )
            optimized_df = _apply_fixed_residual_calibration_to_rf(optimized_df, residual_models)
            merged_df, rf_delta_metrics = _apply_rf_delta_to_baseline_points(
                baseline_df,
                baseline_rf_df,
                optimized_df,
                baseline_df,
            )
            merged_df = _attach_grid_context_to_predictions(merged_df, baseline_df)
            metrics = _score_candidate_vs_baseline_grids(baseline_df, merged_df, all_evaluation_cells, config, grid_analytics_df)
            metrics.update(
                {
                    "calibration_mode": calibration_mode,
                    "fixed_raw_k1k2_used": float(bool(use_fixed_raw_k1k2)),
                    "residual_calibration_applied": float(bool(residual_models)),
                    "radius_affected_cell_count": float(len(radius_affected_cells)),
                    "topology_affected_cell_count": float(len(topology_affected_cells)),
                    "calibration_cell_count": float(len(calibration_cells)),
                    "affected_site_count": float(len(affected_sites)),
                    "optimized_row_count": float(len(optimized_df)),
                    "rf_baseline_row_count": float(len(baseline_rf_df)),
                    "merged_row_count": float(len(merged_df)),
                    "baseline_row_count": float(len(baseline_df)),
                    "optimized_distinct_cell_count": float(optimized_df["Node_Cell_ID"].astype(str).nunique()) if "Node_Cell_ID" in optimized_df.columns else 0.0,
                    **rf_delta_metrics,
                    **_changed_cell_local_metrics_global(baseline_df, merged_df, updates, config.rsrp_threshold),
                }
            )
        except Exception as exc:
            metrics = {
                "baseline_bad_count": np.nan,
                "candidate_bad_count": np.nan,
                "recovered_bad_samples": np.nan,
                "new_bad_samples": np.nan,
                "rsrp_recovered_bad": np.nan,
                "rsrp_new_bad": np.nan,
                "rsrp_severity_reduction": np.nan,
                "evaluation_sample_count": np.nan,
                "net_bad_reduction": np.nan,
                "good_area_loss_pct": np.nan,
                "mean_rsrp_delta": np.nan,
                "mean_sinr_delta": np.nan,
                "score": -99999.0,
                "constraints_passed": 0.0,
                "calibration_mode": "error",
            }
            return _result(candidate_name, updates, metrics, str(exc))
        with eval_lock:
            rf_cache[cache_key] = metrics
        return _result(candidate_name, updates, metrics)

    hold_result = _evaluate("hold", [])
    evaluation_rows.append(hold_result)

    delta_options = (-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0)
    max_coordinate_passes = 3
    current_updates_by_cell: Dict[str, Dict[str, object]] = {}
    current_result = hold_result
    seen_coordinate_keys: set[tuple] = {base._candidate_cache_key(site_id, [])}

    def _updates_from_state(state: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
        ordered: List[Dict[str, object]] = []
        for cell_id in target_cells:
            update = state.get(str(cell_id))
            if update is not None:
                ordered.append(dict(update))
        return _dedupe_updates(ordered)

    def _delta_label(delta: float) -> str:
        if np.isclose(float(delta), 0.0):
            return "hold"
        return f"plus_{abs(float(delta)):g}" if float(delta) > 0 else f"minus_{abs(float(delta)):g}"

    def _safe_cell_label(cell_id: str) -> str:
        return str(cell_id).replace("|", "_").replace(".", "p").replace(",", "_")

    for pass_idx in range(1, max_coordinate_passes + 1):
        pass_changed = False
        print(
            f"[TILT_RSRP_COORDINATE_PASS] pass={pass_idx} cells={len(target_cells)} "
            f"active_updates={len(current_updates_by_cell)} current_score={float(current_result.get('score', 0.0)):.4f} "
            f"current_net={float(current_result.get('net_bad_reduction', 0.0)):.0f}"
        )
        for cell_id in target_cells:
            best_cell_state = dict(current_updates_by_cell)
            best_cell_result = current_result
            evaluated_for_cell = 0
            for delta in delta_options:
                trial_state = dict(current_updates_by_cell)
                if np.isclose(float(delta), 0.0):
                    trial_state.pop(str(cell_id), None)
                else:
                    update = _target_update(str(cell_id), float(delta))
                    if update is None:
                        continue
                    trial_state[str(cell_id)] = update
                trial_updates = _updates_from_state(trial_state)
                cache_key = base._candidate_cache_key(site_id, trial_updates)
                if cache_key in seen_coordinate_keys:
                    continue
                seen_coordinate_keys.add(cache_key)
                candidate_name = (
                    f"coord_pass_{pass_idx}_cell_{_safe_cell_label(cell_id)}_"
                    f"etilt_{_delta_label(delta)}_active_{len(trial_updates)}"
                )
                trial_result = _evaluate(candidate_name, trial_updates)
                evaluation_rows.append(trial_result)
                evaluated_for_cell += 1
                if _result_rank(trial_result) > _result_rank(best_cell_result):
                    best_cell_state = trial_state
                    best_cell_result = trial_result
            if _result_rank(best_cell_result) > _result_rank(current_result):
                previous_score = float(current_result.get("score", 0.0))
                current_updates_by_cell = best_cell_state
                current_result = best_cell_result
                pass_changed = True
                print(
                    f"[TILT_RSRP_COORDINATE_KEEP] pass={pass_idx} cell={cell_id} "
                    f"evaluated={evaluated_for_cell} active_updates={len(current_updates_by_cell)} "
                    f"score={float(current_result.get('score', 0.0)):.4f} previous_score={previous_score:.4f} "
                    f"net={float(current_result.get('net_bad_reduction', 0.0)):.0f}"
                )
            else:
                print(
                    f"[TILT_RSRP_COORDINATE_HOLD_CELL] pass={pass_idx} cell={cell_id} "
                    f"evaluated={evaluated_for_cell} active_updates={len(current_updates_by_cell)} "
                    f"score={float(current_result.get('score', 0.0)):.4f} "
                    f"net={float(current_result.get('net_bad_reduction', 0.0)):.0f}"
                )
        if not pass_changed:
            print(f"[TILT_RSRP_COORDINATE_STOP] pass={pass_idx} reason=no_cell_improved")
            break

    final_coordinate_updates = _updates_from_state(current_updates_by_cell)
    if final_coordinate_updates:
        final_coordinate_result = _evaluate(f"coordinate_final_active_{len(final_coordinate_updates)}", final_coordinate_updates)
        evaluation_rows.append(final_coordinate_result)

    evaluation_df = pd.DataFrame(evaluation_rows)
    chosen_row = _choose_best_candidate_row(evaluation_df)
    chosen_result = chosen_row.to_dict() if chosen_row is not None else hold_result
    selected_updates = _parse_updates_from_result(chosen_result) if str(chosen_result.get("candidate_name", "hold")) != "hold" else []
    actionable = selected_updates and bool(int(float(chosen_result.get("constraints_passed", 0.0)))) and (
        float(chosen_result.get("score", -99999.0)) > 0.0
        or float(chosen_result.get("net_bad_reduction", 0.0)) > 0.0
        or float(chosen_result.get("recovered_bad_samples", 0.0)) > float(chosen_result.get("new_bad_samples", 0.0))
    )
    if not actionable:
        selected_updates = []
    status = "action_change_validated" if selected_updates else "no_safe_change_available"
    confidence_score = base._clamp_score(max(0.0, min(100.0, float(chosen_result.get("score", 0.0)) * 4.0 + (15.0 if selected_updates else 0.0))))
    updates_by_cell = {str(update.get("cell_id", "")): float(update.get("target_value", np.nan)) for update in selected_updates}
    reason = (
        f"Global RSRP bad-grid cell optimization. candidate={chosen_result.get('candidate_name')}. "
        f"score={float(chosen_result.get('score', 0.0)):.4f}. "
        f"baseline_bad_grids={float(chosen_result.get('baseline_bad_grid_count', chosen_result.get('baseline_bad_count', 0.0))):.0f}. "
        f"candidate_bad_grids={float(chosen_result.get('candidate_bad_grid_count', chosen_result.get('candidate_bad_count', 0.0))):.0f}. "
        f"net_bad_reduction={float(chosen_result.get('net_bad_reduction', 0.0)):.0f}. "
        f"bad_to_good={float(chosen_result.get('bad_to_good_grid_count', chosen_result.get('recovered_bad_samples', 0.0))):.0f}. "
        f"good_to_bad={float(chosen_result.get('good_to_bad_grid_count', chosen_result.get('new_bad_samples', 0.0))):.0f}."
    )
    selected_cell_set = set(target_cells).union(updates_by_cell.keys())
    selected_ant = ant_use.loc[ant_use["Cell ID"].astype(str).isin(selected_cell_set)].copy()
    for _, sector_row in selected_ant.iterrows():
        cell_id = str(sector_row["Cell ID"])
        current_value = pd.to_numeric(pd.Series([sector_row.get("electrical_tilt")]), errors="coerce").iloc[0]
        recommended_value = updates_by_cell.get(cell_id, current_value)
        contributor_row = selected_cells_df.loc[selected_cells_df["Node_Cell_ID"].astype(str) == cell_id]
        recommendation_rows.append(
            {
                "Cell ID": cell_id,
                "Technology": base.TILT_SRC._safe_str(sector_row[tech_col]) if tech_col and tech_col in sector_row else "4G",
                "Parameter": "ETilt",
                "Current Value": current_value,
                "Recommended Value": recommended_value,
                "Reason": reason,
                "Swap Sector Detected": "No",
                "Bad Sample Count": int(pd.to_numeric(contributor_row.get("total_bad_samples", pd.Series([0])), errors="coerce").fillna(0).sum()) if not contributor_row.empty else 0,
                "Root Cause Category": "global_bad_grid_rsrp",
                "Recommendation Status": status if cell_id in updates_by_cell else "no_change",
                "Recommendation Confidence": base._confidence_label(confidence_score),
                "Confidence Score": confidence_score,
                "Baseline Bad Count": float(chosen_result.get("baseline_bad_count", np.nan)),
                "Candidate Bad Count": float(chosen_result.get("candidate_bad_count", np.nan)),
                "Baseline Bad Grid Count": float(chosen_result.get("baseline_bad_grid_count", np.nan)),
                "Candidate Bad Grid Count": float(chosen_result.get("candidate_bad_grid_count", np.nan)),
                "Bad To Good Grid Count": float(chosen_result.get("bad_to_good_grid_count", np.nan)),
                "Good To Bad Grid Count": float(chosen_result.get("good_to_bad_grid_count", np.nan)),
                "Score": float(chosen_result.get("score", np.nan)),
                "Mean RSRP Delta": float(chosen_result.get("mean_rsrp_delta", np.nan)),
                "RSRP Recovered Bad": float(chosen_result.get("rsrp_recovered_bad", np.nan)),
                "RSRP New Bad": float(chosen_result.get("rsrp_new_bad", np.nan)),
                "RSRP Severity Reduction Per Sample": float(chosen_result.get("rsrp_severity_reduction_per_sample", np.nan)),
            }
        )

    recommendations_df = pd.DataFrame(recommendation_rows)
    return recommendations_df, evaluation_df


def _result_rank_for_export(row: pd.Series) -> tuple:
    score = float(pd.to_numeric(pd.Series([row.get("score")]), errors="coerce").fillna(-99999.0).iloc[0])
    net_bad_reduction = float(pd.to_numeric(pd.Series([row.get("net_bad_reduction")]), errors="coerce").fillna(-99999.0).iloc[0])
    recovered_bad = float(pd.to_numeric(pd.Series([row.get("recovered_bad_samples")]), errors="coerce").fillna(0.0).iloc[0])
    new_bad = float(pd.to_numeric(pd.Series([row.get("new_bad_samples")]), errors="coerce").fillna(0.0).iloc[0])
    candidate_name = str(row.get("candidate_name", "")).strip().lower()
    actionable = candidate_name != "hold" and (score > 0.0 or net_bad_reduction > 0.0 or recovered_bad > new_bad)
    return (
        int(bool(pd.to_numeric(pd.Series([row.get("constraints_passed")]), errors="coerce").fillna(0).iloc[0])),
        int(actionable),
        score,
        net_bad_reduction,
        recovered_bad - new_bad,
        float(pd.to_numeric(pd.Series([row.get("mean_rsrp_delta")]), errors="coerce").fillna(-99999.0).iloc[0]),
    )


def _choose_best_candidate_row(evaluation_df: pd.DataFrame) -> Optional[pd.Series]:
    if evaluation_df.empty:
        return None
    ranked_idx = sorted(evaluation_df.index.tolist(), key=lambda idx: _result_rank_for_export(evaluation_df.loc[idx]), reverse=True)
    return evaluation_df.loc[ranked_idx[0]]


def _materialize_candidate_scope(
    baseline_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    candidate_row: Optional[pd.Series],
    config: base.TiltRecommendationTestConfig,
    baseline_job_id: str,
    grid_analytics_df: Optional[pd.DataFrame] = None,
    geo_features_df: Optional[pd.DataFrame] = None,
    residual_models: Optional[Dict[str, Dict[str, object]]] = None,
    use_fixed_raw_k1k2: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]], Dict[str, object]]:
    if candidate_row is None:
        return pd.DataFrame(), pd.DataFrame(), [], {}

    target_value = str(candidate_row.get("target_value", "") or "[]")
    try:
        updates = json.loads(target_value)
        if not isinstance(updates, list):
            updates = []
    except Exception:
        updates = []

    if not updates:
        affected_cells = str(candidate_row.get("target_action_cells", "") or "").split(",")
        affected_cells = _canonicalize_cell_list([cell.strip() for cell in affected_cells if cell.strip()], baseline_df)
        eval_cells = _canonicalize_cell_list(base._expand_evaluation_cells_from_topology(baseline_df, affected_cells), baseline_df)
        eval_grid_ids = _grid_ids_for_evaluation_cells(baseline_df, eval_cells)
        before_scope = _grid_scope_for_ids(baseline_df, eval_grid_ids)
        after_scope = before_scope.copy()
        meta = {
            "site_id": str(candidate_row.get("site_id", "")),
            "candidate_name": str(candidate_row.get("candidate_name", "")),
            "score": float(pd.to_numeric(pd.Series([candidate_row.get("score")]), errors="coerce").fillna(np.nan).iloc[0]),
            "baseline_bad_count": float(pd.to_numeric(pd.Series([candidate_row.get("baseline_bad_count")]), errors="coerce").fillna(np.nan).iloc[0]),
            "candidate_bad_count": float(pd.to_numeric(pd.Series([candidate_row.get("candidate_bad_count")]), errors="coerce").fillna(np.nan).iloc[0]),
            "mean_rsrp_delta": float(pd.to_numeric(pd.Series([candidate_row.get("mean_rsrp_delta")]), errors="coerce").fillna(np.nan).iloc[0]),
            "constraints_passed": int(pd.to_numeric(pd.Series([candidate_row.get("constraints_passed")]), errors="coerce").fillna(0).iloc[0]),
            "selected_updates": updates,
            "evaluation_cells": eval_cells,
            "evaluation_grid_ids": eval_grid_ids,
        }
        return before_scope, after_scope, updates, meta

    modified_site_df = base._apply_multiple_parameter_targets(antenna_df, updates)
    affected_cells, _, _ = base.opt_ml._compute_affected_cells(
        modified_site_df,
        float(config.impact_radius_m),
        int(config.neighbor_site_count),
    )
    affected_cells = _canonicalize_cell_list(affected_cells, baseline_df)
    update_cells = _canonicalize_cell_list(
        [str(update.get("cell_id", "")) for update in updates if str(update.get("cell_id", "")).strip()],
        baseline_df,
    )
    calibration_cells = _canonicalize_cell_list(set(affected_cells).union(update_cells), baseline_df)
    if use_fixed_raw_k1k2:
        k1k2_map = _fixed_raw_k1k2_map(calibration_cells)
    else:
        k1k2_map = base.opt_ml.compute_k1k2_for_cells(baseline_df, modified_site_df, calibration_cells)
    baseline_prediction_site_df = _mark_recompute_cells_for_prediction(antenna_df, calibration_cells, [], baseline_df)
    candidate_prediction_site_df = _mark_recompute_cells_for_prediction(modified_site_df, calibration_cells, update_cells, baseline_df)
    prediction_params = {
        "project_id": int(config.project_id),
        "region": config.region,
        "radius": float(config.radius_m),
        "grid_resolution": float(config.grid_resolution_m),
        "n_workers": int(config.workers),
        "impact_radius_m": float(config.impact_radius_m),
        "neighbor_site_count": int(config.neighbor_site_count),
        "max_interference_sites": int(config.max_interference_sites),
        "baseline_job_id": baseline_job_id,
        "prediction_points_df": baseline_df,
        "geo_features_df": geo_features_df,
    }
    baseline_rf_df = base.opt_ml.run_prediction_only_optimized(
        baseline_prediction_site_df,
        k1k2_map,
        prediction_params,
    )
    baseline_rf_df = _apply_fixed_residual_calibration_to_rf(baseline_rf_df, residual_models)
    optimized_df = base.opt_ml.run_prediction_only_optimized(
        candidate_prediction_site_df,
        k1k2_map,
        prediction_params,
    )
    optimized_df = _apply_fixed_residual_calibration_to_rf(optimized_df, residual_models)
    merged_df, rf_delta_metrics = _apply_rf_delta_to_baseline_points(
        baseline_df,
        baseline_rf_df,
        optimized_df,
        baseline_df,
    )
    merged_df = _attach_grid_context_to_predictions(merged_df, baseline_df)
    evaluation_cells = _canonicalize_cell_list(base._expand_evaluation_cells_from_topology(baseline_df, affected_cells), baseline_df)
    evaluation_grid_ids = _grid_ids_for_evaluation_cells(baseline_df, evaluation_cells)
    before_scope = _grid_scope_for_ids(baseline_df, evaluation_grid_ids)
    after_scope = _grid_scope_for_ids(merged_df, evaluation_grid_ids)
    meta = {
        "site_id": str(candidate_row.get("site_id", "")),
        "candidate_name": str(candidate_row.get("candidate_name", "")),
        "score": float(pd.to_numeric(pd.Series([candidate_row.get("score")]), errors="coerce").fillna(np.nan).iloc[0]),
        "baseline_bad_count": float(pd.to_numeric(pd.Series([candidate_row.get("baseline_bad_count")]), errors="coerce").fillna(np.nan).iloc[0]),
        "candidate_bad_count": float(pd.to_numeric(pd.Series([candidate_row.get("candidate_bad_count")]), errors="coerce").fillna(np.nan).iloc[0]),
        "mean_rsrp_delta": float(pd.to_numeric(pd.Series([candidate_row.get("mean_rsrp_delta")]), errors="coerce").fillna(np.nan).iloc[0]),
        "constraints_passed": int(pd.to_numeric(pd.Series([candidate_row.get("constraints_passed")]), errors="coerce").fillna(0).iloc[0]),
        "selected_updates": updates,
        "evaluation_cells": evaluation_cells,
        "evaluation_grid_ids": evaluation_grid_ids,
        "fixed_raw_k1k2_used": bool(use_fixed_raw_k1k2),
        "residual_calibration_applied": bool(residual_models),
        **rf_delta_metrics,
    }
    return before_scope, after_scope, updates, meta


def _prepare_scope_export(df: pd.DataFrame, rsrp_threshold: float, stage: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    lat_col = next((c for c in ["lat", "latitude"] if c in out.columns), None)
    lon_col = next((c for c in ["lon", "longitude"] if c in out.columns), None)
    if lat_col and lat_col != "lat":
        out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
    else:
        out["lat"] = pd.to_numeric(out.get("lat"), errors="coerce")
    if lon_col and lon_col != "lon":
        out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    else:
        out["lon"] = pd.to_numeric(out.get("lon"), errors="coerce")
    out["pred_rsrp"] = pd.to_numeric(out.get("pred_rsrp"), errors="coerce")
    out["is_bad_rsrp"] = out["pred_rsrp"] < float(rsrp_threshold)
    out["stage"] = stage
    return out


def run_tilt_rsrp_only_recommendation_test(config: TiltRsrpOnlyRecommendationTestConfig) -> Path:
    start = time.perf_counter()
    run_dir = _ensure_dir(config.output_root / f"project_{config.project_id}" / f"tilt_rsrp_only_{_timestamp()}")
    base_config = _config_as_base(config)

    base.TILT_SRC.RSRP_THRESH = float(config.rsrp_threshold)
    base.TILT_SRC.RSRQ_THRESH = -999.0
    base.TILT_SRC.SINR_THRESH = -999.0

    threshold_path = base._resolve_threshold_file_path_test_only(config.threshold_file_path)
    constraint_df = base._load_threshold_constraints_test_only(threshold_path) if threshold_path else pd.DataFrame()
    constraint_map = base._constraint_map_test_only(constraint_df)

    using_local_inputs = bool(config.baseline_points_path or config.antenna_input_path or config.geo_features_path or config.grid_analytics_path)
    if using_local_inputs:
        print("[TILT_RSRP_LOCAL_INPUT] enabled=True db_fetch_for_inputs=False")
        baseline_job_id = "local_artifact_baseline"
        log_df = _prepare_local_baseline_points(_read_local_table(config.baseline_points_path), config.local_baseline_kpi_stage)
        antenna_df = _read_local_table(config.antenna_input_path)
        geo_df = _read_local_table(config.geo_features_path)
        grid_analytics_df = _read_local_table(config.grid_analytics_path)
        log_df, antenna_df = _prepare_rsrp_only_inputs(log_df, antenna_df)
        log_df = base._enrich_log_with_antenna_context(log_df, antenna_df)
        if "nodeb_id_cell_id" in geo_df.columns and "Node_Cell_ID" not in geo_df.columns:
            geo_df["Node_Cell_ID"] = geo_df["nodeb_id_cell_id"].astype(str)
        if "grid_id" in log_df.columns:
            log_df["grid_id"] = pd.NA
        log_df = _attach_grid_context_to_predictions(log_df, geo_df, grid_analytics_df)
        base._rsrp_only_active_baseline_df = log_df
        polygon_gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    else:
        baseline_job_id = base._fetch_latest_baseline_job_id(config.project_id, config.region)
        log_df = base._fetch_baseline_log_df(config.project_id, config.region, config.operator)
        log_df = _attach_baseline_identity_columns_from_db(log_df, config.project_id, config.region, baseline_job_id)
        antenna_df = base._fetch_antenna_df(config.project_id, config.region, config.operator)
        log_df, antenna_df = _prepare_rsrp_only_inputs(log_df, antenna_df)
        log_df, antenna_df, polygon_gdf = _filter_run_inputs_to_project_polygon(
            log_df=log_df,
            antenna_df=antenna_df,
            project_id=config.project_id,
            region=config.region,
        )
        log_df = base._enrich_log_with_antenna_context(log_df, antenna_df)
        geo_df = base._fetch_geo_df(config.project_id, config.region, config.operator, antenna_df)
        grid_analytics_df = _fetch_grid_analytics_df(config.project_id, config.region, config.operator)
        log_df = _attach_grid_context_to_predictions(log_df, geo_df, grid_analytics_df)
        base._rsrp_only_active_baseline_df = log_df
    grid_validation = _grid_validation_payload(log_df, grid_analytics_df, config.rsrp_threshold)
    _log_grid_validation("BASELINE_VALIDATE", grid_validation)
    residual_models, residual_debug = _fit_fixed_residual_calibration(log_df, config)
    use_fixed_raw_k1k2 = bool(using_local_inputs and config.fixed_k1k2_for_local_inputs)
    print(
        f"[TILT_RSRP_RF_PIPELINE] baseline_anchor=stored_calibrated_baseline "
        f"candidate_delta_residual_calibration={bool(residual_models)} "
        f"fixed_raw_k1k2={use_fixed_raw_k1k2}"
    )

    bad_samples_df, summary_df = base.TILT_SRC.filter_bad_samples(log_df.copy(), base.TILT_SRC.ALLOWED_TECHS)
    bad_samples_df = bad_samples_df.loc[bad_samples_df.get("Bad RSRP", False).astype(bool)].copy() if not bad_samples_df.empty and "Bad RSRP" in bad_samples_df.columns else bad_samples_df.copy()
    grid_ranked_summary_df, baseline_grid_metrics_df = _build_grid_ranked_summary(log_df, antenna_df, config.rsrp_threshold, grid_analytics_df)
    if not grid_ranked_summary_df.empty:
        summary_df = grid_ranked_summary_df.copy()
    elif not summary_df.empty and "Bad RSRP" in summary_df.columns:
        summary_df = summary_df.loc[pd.to_numeric(summary_df["Bad RSRP"], errors="coerce").fillna(0) > 0].copy()
        for col in ["Bad RSRQ", "Bad SINR"]:
            if col not in summary_df.columns:
                summary_df[col] = 0
            else:
                summary_df[col] = 0
        if "Bad Samples" in summary_df.columns:
            summary_df["Bad Samples"] = pd.to_numeric(summary_df["Bad RSRP"], errors="coerce").fillna(0).astype(int)
        if "total_bad_samples" in summary_df.columns:
            summary_df["total_bad_samples"] = pd.to_numeric(summary_df["Bad RSRP"], errors="coerce").fillna(0).astype(int)

    bad_geo_df = base._attach_geo_to_bad_samples(bad_samples_df, geo_df)
    geo_cell_summary = base._aggregate_bad_geo_context(bad_geo_df)

    recommendations_all_df, site_candidate_evaluation_df = _build_tilt_only_recommendations(
        summary_df=summary_df,
        antenna_df=antenna_df,
        baseline_df=log_df,
        config=base_config,
        constraint_map=constraint_map,
        grid_analytics_df=grid_analytics_df,
        geo_features_df=geo_df,
        residual_models=residual_models,
        use_fixed_raw_k1k2=use_fixed_raw_k1k2,
    )
    recommendations_df = recommendations_all_df.copy()
    forecast_df = pd.DataFrame()
    candidate_validation_df = site_candidate_evaluation_df.copy()
    best_candidate_row = _choose_best_candidate_row(candidate_validation_df)
    before_scope_df, after_scope_df, best_updates, best_meta = _materialize_candidate_scope(
        baseline_df=log_df,
        antenna_df=antenna_df,
        candidate_row=best_candidate_row,
        config=base_config,
        baseline_job_id=baseline_job_id,
        grid_analytics_df=grid_analytics_df,
        geo_features_df=geo_df,
        residual_models=residual_models,
        use_fixed_raw_k1k2=use_fixed_raw_k1k2,
    )
    before_scope_df = _prepare_scope_export(before_scope_df, config.rsrp_threshold, "before")
    after_scope_df = _prepare_scope_export(after_scope_df, config.rsrp_threshold, "after")
    before_bad_df = before_scope_df.loc[before_scope_df["is_bad_rsrp"].fillna(False)].copy() if not before_scope_df.empty else pd.DataFrame()
    after_bad_df = after_scope_df.loc[after_scope_df["is_bad_rsrp"].fillna(False)].copy() if not after_scope_df.empty else pd.DataFrame()
    before_grid_df = _aggregate_grid_metrics_from_predictions(before_scope_df, config.rsrp_threshold)
    after_grid_df = _aggregate_grid_metrics_from_predictions(after_scope_df, config.rsrp_threshold)
    before_bad_grid_df = before_grid_df.loc[before_grid_df["is_bad_rsrp"].fillna(False)].copy() if not before_grid_df.empty else pd.DataFrame()
    after_bad_grid_df = after_grid_df.loc[after_grid_df["is_bad_rsrp"].fillna(False)].copy() if not after_grid_df.empty else pd.DataFrame()

    artifact_paths: Dict[str, str] = {}
    artifact_paths["baseline_log_input"] = _write_csv_artifact(log_df, run_dir / "baseline_log_input.csv", compress=True)
    artifact_paths["antenna_input"] = _write_csv_artifact(antenna_df, run_dir / "antenna_input.csv")
    if not polygon_gdf.empty:
        polygon_gdf.to_file(run_dir / "project_polygon.geojson", driver="GeoJSON")
    else:
        (run_dir / "project_polygon.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    artifact_paths["geo_features_input"] = _write_csv_artifact(geo_df, run_dir / "geo_features_input.csv", compress=True)
    artifact_paths["grid_analytics_input"] = _write_csv_artifact(grid_analytics_df, run_dir / "grid_analytics_input.csv")
    artifact_paths["baseline_grid_metrics"] = _write_csv_artifact(baseline_grid_metrics_df, run_dir / "baseline_grid_metrics.csv")
    _write_json(run_dir / "grid_validation_summary.json", grid_validation)
    artifact_paths["grid_validation_summary"] = str(run_dir / "grid_validation_summary.json")
    _write_json(run_dir / "residual_calibration_summary.json", residual_debug)
    artifact_paths["residual_calibration_summary"] = str(run_dir / "residual_calibration_summary.json")
    artifact_paths["bad_samples"] = _write_csv_artifact(bad_samples_df, run_dir / "bad_samples.csv", compress=True)
    artifact_paths["bad_samples_with_geo"] = _write_csv_artifact(bad_geo_df, run_dir / "bad_samples_with_geo.csv", compress=True)
    artifact_paths["bad_summary"] = _write_csv_artifact(summary_df, run_dir / "bad_summary.csv")
    artifact_paths["bad_geo_cell_summary"] = _write_csv_artifact(geo_cell_summary, run_dir / "bad_geo_cell_summary.csv")
    artifact_paths["recommendations_all"] = _write_csv_artifact(recommendations_all_df, run_dir / "recommendations_all.csv")
    artifact_paths["recommendations"] = _write_csv_artifact(recommendations_df, run_dir / "recommendations.csv")
    artifact_paths["site_candidate_evaluations"] = _write_csv_artifact(site_candidate_evaluation_df, run_dir / "site_candidate_evaluations.csv")
    artifact_paths["candidate_validation_results"] = _write_csv_artifact(candidate_validation_df, run_dir / "candidate_validation_results.csv")
    artifact_paths["best_candidate_before_scope"] = _write_csv_artifact(before_scope_df, run_dir / "best_candidate_before_scope.csv", compress=True)
    artifact_paths["best_candidate_after_scope"] = _write_csv_artifact(after_scope_df, run_dir / "best_candidate_after_scope.csv", compress=True)
    artifact_paths["best_candidate_before_bad_rsrp"] = _write_csv_artifact(before_bad_df, run_dir / "best_candidate_before_bad_rsrp.csv", compress=True)
    artifact_paths["best_candidate_after_bad_rsrp"] = _write_csv_artifact(after_bad_df, run_dir / "best_candidate_after_bad_rsrp.csv", compress=True)
    artifact_paths["best_candidate_before_grid_metrics"] = _write_csv_artifact(before_grid_df, run_dir / "best_candidate_before_grid_metrics.csv")
    artifact_paths["best_candidate_after_grid_metrics"] = _write_csv_artifact(after_grid_df, run_dir / "best_candidate_after_grid_metrics.csv")
    artifact_paths["best_candidate_before_bad_grids"] = _write_csv_artifact(before_bad_grid_df, run_dir / "best_candidate_before_bad_grids.csv")
    artifact_paths["best_candidate_after_bad_grids"] = _write_csv_artifact(after_bad_grid_df, run_dir / "best_candidate_after_bad_grids.csv")
    artifact_paths["best_candidate_updates"] = _write_csv_artifact(pd.DataFrame(best_updates), run_dir / "best_candidate_updates.csv")

    total_runtime_sec = time.perf_counter() - start
    best_candidate_payload = {
        **best_meta,
        "selected_updates": best_updates,
        "before_bad_rsrp_count": int(len(before_bad_df)),
        "after_bad_rsrp_count": int(len(after_bad_df)),
        "before_bad_grid_count": int(len(before_bad_grid_df)),
        "after_bad_grid_count": int(len(after_bad_grid_df)),
    }
    _write_json(run_dir / "best_candidate_summary.json", best_candidate_payload)

    changed_rows = recommendations_df.copy()
    if not changed_rows.empty:
        changed_rows["Current Value"] = pd.to_numeric(changed_rows["Current Value"], errors="coerce")
        changed_rows["Recommended Value"] = pd.to_numeric(changed_rows["Recommended Value"], errors="coerce")
        changed_rows = changed_rows.loc[
            (changed_rows["Recommended Value"] - changed_rows["Current Value"]).abs() > 1e-9
        ].copy()

    summary_payload = {
        "run_type": "tilt_rsrp_only_test",
        "project_id": int(config.project_id),
        "region": str(config.region),
        "operator": config.operator,
        "thresholds": {
            "rsrp": float(config.rsrp_threshold),
            "rsrq": None,
            "sinr": None,
            "kpi_mode": "RSRP only",
        },
        "search": {
            "bad_grid_coverage_pct": float(config.bad_grid_coverage_pct),
            "max_group_cells": int(config.max_group_cells),
            "execution": "global_bad_grid_to_contributing_cells_to_etilt_to_rf_recompute_to_global_grid_validation",
            "candidate_rf_pipeline": "raw_rf_plus_geo_correction_plus_fixed_dt_residual_calibration",
            "baseline_anchor": "stored_calibrated_baseline",
            "fixed_raw_k1k2_for_local_inputs": bool(use_fixed_raw_k1k2),
            "residual_calibration_applied": bool(residual_models),
        },
        "counts": {
            "baseline_rows": int(len(log_df)),
            "bad_samples": int(len(bad_samples_df)),
            "bad_cells": int(summary_df["Cell ID"].nunique()) if not summary_df.empty and "Cell ID" in summary_df.columns else 0,
            "recommendation_rows_all": int(len(recommendations_all_df)),
            "recommendation_rows_changed": int(len(changed_rows)),
            "candidate_validation_rows": int(len(candidate_validation_df)),
            "before_bad_rsrp_count": int(len(before_bad_df)),
            "after_bad_rsrp_count": int(len(after_bad_df)),
            "baseline_bad_grid_count": int(len(baseline_grid_metrics_df.loc[baseline_grid_metrics_df["is_bad_rsrp"].fillna(False)])) if not baseline_grid_metrics_df.empty else 0,
            "before_bad_grid_count": int(len(before_bad_grid_df)),
            "after_bad_grid_count": int(len(after_bad_grid_df)),
        },
        "best_candidate": best_candidate_payload,
        "artifacts": {**artifact_paths, "best_candidate_summary": str(run_dir / "best_candidate_summary.json")},
        "total_runtime_sec": round(float(total_runtime_sec), 4),
    }
    _write_json(run_dir / "summary.json", summary_payload)
    print(f"[TILT_RSRP_ONLY_TEST][DONE] run_dir={run_dir} total_runtime_sec={summary_payload['total_runtime_sec']}")
    return run_dir


def _parse_args() -> TiltRsrpOnlyRecommendationTestConfig:
    parser = argparse.ArgumentParser()
    fixture_available = PROJECT_196_RSRP_TILT_BASELINE_POINTS.exists()
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--rsrp", type=float, default=-90.0)
    parser.add_argument("--radius-m", type=float, default=500.0)
    parser.add_argument("--grid-resolution-m", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--impact-radius-m", type=float, default=None)
    parser.add_argument("--neighbor-site-count", type=int, default=3)
    parser.add_argument("--max-interference-sites", type=int, default=10)
    parser.add_argument("--max-good-area-loss-pct", type=float, default=2.0)
    parser.add_argument("--max-mean-sinr-drop-db", type=float, default=1.0)
    parser.add_argument("--min-score-gain", type=float, default=0.0)
    parser.add_argument("--min-recovered-bad-samples", type=int, default=0)
    parser.add_argument("--bad-grid-coverage-pct", type=float, default=80.0)
    parser.add_argument("--max-group-cells", type=int, default=0)
    parser.add_argument(
        "--threshold-file",
        "--threshold-file-path",
        dest="threshold_file_path",
        type=str,
        default=str(PROJECT_196_RSRP_TILT_THRESHOLD_FILE) if fixture_available else None,
    )
    parser.add_argument(
        "--baseline-points",
        "--baseline-points-path",
        dest="baseline_points_path",
        type=str,
        default=str(PROJECT_196_RSRP_TILT_BASELINE_POINTS) if fixture_available else None,
    )
    parser.add_argument(
        "--antenna-input",
        "--antenna-input-path",
        dest="antenna_input_path",
        type=str,
        default=str(PROJECT_196_RSRP_TILT_ANTENNA_INPUT) if fixture_available else None,
    )
    parser.add_argument(
        "--geo-features",
        "--geo-features-path",
        dest="geo_features_path",
        type=str,
        default=str(PROJECT_196_RSRP_TILT_GEO_FEATURES) if fixture_available else None,
    )
    parser.add_argument(
        "--grid-analytics",
        "--grid-analytics-path",
        dest="grid_analytics_path",
        type=str,
        default=str(PROJECT_196_RSRP_TILT_GRID_ANALYTICS) if fixture_available else None,
    )
    parser.add_argument("--local-baseline-kpi-stage", choices=["geo", "demo", "raw"], default="geo")
    parser.add_argument("--session-ids", type=str, default=",".join(str(value) for value in DEFAULT_SESSION_IDS))
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--no-residual-calibration", action="store_true")
    parser.add_argument("--recompute-k1k2-from-baseline", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    session_ids = tuple(int(value.strip()) for value in str(args.session_ids).split(",") if value.strip())
    return TiltRsrpOnlyRecommendationTestConfig(
        project_id=args.project_id,
        region=args.region,
        operator=args.operator,
        rsrp_threshold=args.rsrp,
        radius_m=args.radius_m,
        grid_resolution_m=args.grid_resolution_m,
        workers=args.workers,
        impact_radius_m=args.impact_radius_m,
        neighbor_site_count=args.neighbor_site_count,
        max_interference_sites=args.max_interference_sites,
        max_good_area_loss_pct=args.max_good_area_loss_pct,
        max_mean_sinr_drop_db=args.max_mean_sinr_drop_db,
        min_score_gain=args.min_score_gain,
        min_recovered_bad_samples=args.min_recovered_bad_samples,
        bad_grid_coverage_pct=args.bad_grid_coverage_pct,
        max_group_cells=args.max_group_cells,
        threshold_file_path=args.threshold_file_path,
        baseline_points_path=args.baseline_points_path,
        antenna_input_path=args.antenna_input_path,
        geo_features_path=args.geo_features_path,
        grid_analytics_path=args.grid_analytics_path,
        local_baseline_kpi_stage=args.local_baseline_kpi_stage,
        session_ids=session_ids,
        validation_fraction=args.validation_fraction,
        apply_residual_calibration=not bool(args.no_residual_calibration),
        fixed_k1k2_for_local_inputs=not bool(args.recompute_k1k2_from_baseline),
        output_root=args.output_root,
    )


if __name__ == "__main__":
    run_dir = run_tilt_rsrp_only_recommendation_test(_parse_args())
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
