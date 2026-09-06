from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tools.lte_tilt_recommandation.candidate_validation import (
    CandidateValidationConfig,
    coordinate_search_recommendations,
)
from tools.lte_tilt_recommandation import etilt_optimizer_cd2 as TILT_SRC


DISABLED_THRESHOLD = -999999.0


@dataclass
class TiltEngineConfig:
    project_id: int
    region: str
    operator: Optional[str]
    mode: str
    rsrp_threshold: float
    rsrq_threshold: float
    sinr_threshold: float
    rsrp_weight: float = 34.0
    rsrq_weight: float = 33.0
    sinr_weight: float = 33.0
    validate_candidates: bool = True
    max_validation_candidates: int = 25
    radius_m: float = 500.0
    grid_resolution_m: float = 30.0
    workers: int = 1
    impact_radius_m: float = 500.0
    neighbor_site_count: int = 3
    max_interference_sites: int = 10
    baseline_job_id: Optional[str] = None
    coordinate_passes: int = 2
    candidate_workers: int = 1
    bad_grid_coverage_pct: float = 60.0
    max_group_cells: int = 20  # applied per technology (4G and 5G each get this budget)
    max_neighbors_per_update_cell: int = 2
    etilt_candidate_max_delta_deg: float = 4.0
    azimuth_fallback_max_delta_deg: float = 30.0
    azimuth_fallback_step_deg: float = 5.0
    azimuth_fallback_steps_deg: Optional[tuple[float, ...]] = None
    rf_debug_log_path: Optional[str] = None
    constraint_map: Optional[Dict[str, Dict[str, object]]] = None

def _normalise_mode(mode: object) -> str:
    text = str(mode or "combined_weighted").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "combined": "combined_weighted",
        "weighted": "combined_weighted",
        "rsrp": "rsrp_only",
        "rsrq": "rsrq_only",
        "sinr": "sinr_only",
    }
    text = aliases.get(text, text)
    if text not in {"rsrp_only", "rsrq_only", "sinr_only", "combined_weighted"}:
        raise ValueError(f"Unsupported tilt recommendation mode: {mode}")
    return text


def _active_thresholds(config: TiltEngineConfig) -> Dict[str, float]:
    mode = _normalise_mode(config.mode)
    if mode == "rsrp_only":
        return {"rsrp": float(config.rsrp_threshold), "rsrq": DISABLED_THRESHOLD, "sinr": DISABLED_THRESHOLD}
    if mode == "rsrq_only":
        return {"rsrp": DISABLED_THRESHOLD, "rsrq": float(config.rsrq_threshold), "sinr": DISABLED_THRESHOLD}
    if mode == "sinr_only":
        return {"rsrp": DISABLED_THRESHOLD, "rsrq": DISABLED_THRESHOLD, "sinr": float(config.sinr_threshold)}
    return {
        "rsrp": float(config.rsrp_threshold),
        "rsrq": float(config.rsrq_threshold),
        "sinr": float(config.sinr_threshold),
    }


def _normalise_weights(config: TiltEngineConfig) -> Dict[str, float]:
    mode = _normalise_mode(config.mode)
    if mode == "rsrp_only":
        return {"rsrp": 1.0, "rsrq": 0.0, "sinr": 0.0}
    if mode == "rsrq_only":
        return {"rsrp": 0.0, "rsrq": 1.0, "sinr": 0.0}
    if mode == "sinr_only":
        return {"rsrp": 0.0, "rsrq": 0.0, "sinr": 1.0}
    raw = {
        "rsrp": max(float(config.rsrp_weight), 0.0),
        "rsrq": max(float(config.rsrq_weight), 0.0),
        "sinr": max(float(config.sinr_weight), 0.0),
    }
    total = sum(raw.values())
    if total <= 0:
        return {"rsrp": 0.6, "rsrq": 0.2, "sinr": 0.2}
    return {key: value / total for key, value in raw.items()}


def _coerce_metric_columns(log_df: pd.DataFrame) -> pd.DataFrame:
    out = log_df.copy()
    for col in ["pred_rsrp", "pred_rsrq", "pred_sinr"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _severity(values: pd.Series, threshold: float) -> pd.Series:
    return (float(threshold) - pd.to_numeric(values, errors="coerce")).clip(lower=0.0).fillna(0.0)


def _score_grid_analytics(grid_analytics_df: pd.DataFrame, config: TiltEngineConfig) -> pd.DataFrame:
    if grid_analytics_df is None or grid_analytics_df.empty or "grid_id" not in grid_analytics_df.columns:
        return pd.DataFrame()
    thresholds = _active_thresholds(config)
    weights = _normalise_weights(config)
    out = grid_analytics_df.copy()
    score = pd.Series(0.0, index=out.index)
    for kpi, col in [
        ("rsrp", "baseline_avg_rsrp"),
        ("rsrq", "baseline_avg_rsrq"),
        ("sinr", "baseline_avg_sinr"),
    ]:
        if weights[kpi] <= 0 or col not in out.columns:
            continue
        sev = _severity(out[col], thresholds[kpi])
        out[f"{kpi}_severity"] = sev
        out[f"is_bad_{kpi}"] = sev > 0
        score = score + sev * float(weights[kpi])
    out["combined_weighted_severity"] = score
    out["is_bad_combined"] = score > 0
    sort_cols = ["combined_weighted_severity"]
    ascending = [False]
    if "baseline_point_count" in out.columns:
        sort_cols.append("baseline_point_count")
        ascending.append(False)
    return out.sort_values(sort_cols, ascending=ascending, na_position="last")


def _score_prediction_points(log_df: pd.DataFrame, config: TiltEngineConfig) -> pd.DataFrame:
    if log_df.empty:
        return pd.DataFrame()
    thresholds = _active_thresholds(config)
    weights = _normalise_weights(config)
    out = log_df.copy()
    score = pd.Series(0.0, index=out.index)
    for kpi, col in [
        ("rsrp", "pred_rsrp"),
        ("rsrq", "pred_rsrq"),
        ("sinr", "pred_sinr"),
    ]:
        if weights[kpi] <= 0 or col not in out.columns:
            continue
        sev = _severity(out[col], thresholds[kpi])
        out[f"{kpi}_severity"] = sev
        out[f"is_bad_{kpi}"] = sev > 0
        score = score + sev * float(weights[kpi])
    out["combined_weighted_severity"] = score
    out["is_bad_combined"] = score > 0
    return out


def _ensure_bad_geo_columns(bad_geo_df: pd.DataFrame) -> pd.DataFrame:
    out = bad_geo_df.copy()
    for col, default in {
        "nlos_flag": 0.0,
        "building_area_ratio": np.nan,
        "los_blocked_ratio": np.nan,
        "site_count_250m": np.nan,
        "serving_distance_m": np.nan,
        "nearest_site_distance_m": np.nan,
        "mean_nearest3_site_distance_m": np.nan,
        "azimuth_delta_deg": np.nan,
        "clutter_class": "",
        "building_count": np.nan,
        "road_length_m": np.nan,
        "green_ratio": np.nan,
        "water_ratio": np.nan,
        "los_blocker_count": np.nan,
        "terrain_slope_deg": np.nan,
        "terrain_relief_to_site_m": np.nan,
        "site_count_500m": np.nan,
    }.items():
        if col not in out.columns:
            out[col] = default
    return out


def run_recommendation_engine(
    *,
    log_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    grid_analytics_df: pd.DataFrame,
    config: TiltEngineConfig,
) -> Dict[str, pd.DataFrame]:
    thresholds = _active_thresholds(config)
    TILT_SRC.RSRP_THRESH = float(thresholds["rsrp"])
    TILT_SRC.RSRQ_THRESH = float(thresholds["rsrq"])
    TILT_SRC.SINR_THRESH = float(thresholds["sinr"])

    log_work = _coerce_metric_columns(log_df)
    scored_points_df = _score_prediction_points(log_work, config)
    grid_score_df = _score_grid_analytics(grid_analytics_df, config)
    bad_samples_df, summary_df = TILT_SRC.filter_bad_samples(log_work.copy(), TILT_SRC.ALLOWED_TECHS)
    bad_geo_df = _ensure_bad_geo_columns(TILT_SRC.attach_geo_to_bad_samples(bad_samples_df, geo_df))

    if not bool(config.validate_candidates):
        raise ValueError("Production tilt recommendation requires validate_candidates=true; geo-aware fallback is disabled.")

    validation_config = CandidateValidationConfig(
        project_id=int(config.project_id),
        region=str(config.region),
        radius_m=float(config.radius_m),
        grid_resolution_m=float(config.grid_resolution_m),
        workers=int(config.workers),
        impact_radius_m=float(config.impact_radius_m),
        neighbor_site_count=int(config.neighbor_site_count),
        max_interference_sites=int(config.max_interference_sites),
        max_candidates=int(config.max_validation_candidates),
        baseline_job_id=config.baseline_job_id,
        coordinate_passes=int(config.coordinate_passes),
        candidate_workers=int(config.candidate_workers),
        bad_grid_coverage_pct=float(config.bad_grid_coverage_pct),
        max_group_cells=int(config.max_group_cells),
        max_neighbors_per_update_cell=int(config.max_neighbors_per_update_cell),
        etilt_candidate_max_delta_deg=float(config.etilt_candidate_max_delta_deg),
        azimuth_fallback_max_delta_deg=float(config.azimuth_fallback_max_delta_deg),
        azimuth_fallback_step_deg=float(config.azimuth_fallback_step_deg),
        azimuth_fallback_steps_deg=config.azimuth_fallback_steps_deg,
        rf_debug_log_path=config.rf_debug_log_path,
        constraint_map=config.constraint_map,
    )
    validation_outputs = coordinate_search_recommendations(
        baseline_df=log_work,
        antenna_df=antenna_df,
        geo_df=geo_df,
        grid_analytics_df=grid_analytics_df,
        thresholds=_active_thresholds(config),
        weights=_normalise_weights(config),
        config=validation_config,
    )
    recommendations_df = validation_outputs.get("recommendations", pd.DataFrame()).copy()
    recommendations_all_df = recommendations_df.copy()
    forecast_df = TILT_SRC.build_forecast(summary_df, recommendations_all_df) if not recommendations_all_df.empty else pd.DataFrame()

    return {
        "recommendations": recommendations_df,
        "recommendations_all": recommendations_all_df,
        "forecast": forecast_df,
        "bad_samples": bad_geo_df,
        "summary": summary_df,
        "scored_points": scored_points_df,
        "grid_scores": grid_score_df,
        "candidate_evaluations": validation_outputs.get("candidate_evaluations", pd.DataFrame()),
        "best_candidate_after": validation_outputs.get("best_candidate_after", pd.DataFrame()),
        "best_candidate_metrics": validation_outputs.get("best_candidate_metrics", pd.DataFrame()),
    }


def export_engine_report(outputs: Dict[str, pd.DataFrame], output_file: str) -> str:
    return TILT_SRC.export_to_excel(
        summary_df=outputs.get("summary", pd.DataFrame()),
        recommendations_df=outputs.get("recommendations", pd.DataFrame()),
        forecast_df=outputs.get("forecast", pd.DataFrame()),
        bad_samples_df=outputs.get("bad_samples", pd.DataFrame()),
        output_path=output_file,
    )
