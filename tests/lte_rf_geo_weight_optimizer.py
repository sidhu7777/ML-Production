from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import GroupKFold, RepeatedKFold

from tests.lte_rf_debug_lab import DEFAULT_PROJECT_ID, _metric_bundle


OUTPUT_ROOT = Path("tests/output")

CLUTTER_FEATURE = "clutter_class"
NUMERIC_FEATURES = [
    "morphology_cluster",
    "building_area_ratio",
    "building_count",
    "road_length_m",
    "green_ratio",
    "water_ratio",
    "avg_building_area_m2",
    "site_count_250m",
    "site_count_500m",
    "serving_distance_m",
    "nearest_site_distance_m",
    "azimuth_delta_deg",
    "mean_nearest3_site_distance_m",
    "best_interferer_distance_m",
    "best_interferer_azimuth_delta_deg",
    "serving_proxy_rsrp_dbm",
    "best_interferer_proxy_rsrp_dbm",
    "serving_proxy_rsrp_phys_dbm",
    "best_interferer_proxy_phys_dbm",
    "interference_gap_db",
    "interference_ratio_linear",
    "interference_sum_proxy_dbm",
    "sinr_proxy_db",
    "rsrq_proxy_db",
    "same_earfcn_interferer_count",
    "dominant_interferer_count",
    "effective_tx_height_m",
    "los_blocker_count",
    "los_blocked_length_m",
    "los_blocked_ratio",
    "mean_blocker_height_m",
    "max_blocker_height_m",
    "nlos_flag",
    "diffraction_proxy_db",
    "terrain_elevation_m",
    "terrain_slope_deg",
    "proxy_site_elevation_m",
    "terrain_relief_to_site_m",
]

CURRENT_WEIGHTS = {
    "clutter_Dense Urban": -4.5,
    "clutter_Urban": -2.5,
    "clutter_Suburban": -1.0,
    "clutter_Vegetation": -1.8,
    "clutter_Water": 1.0,
    "clutter_Rural/Open": 0.8,
    "morphology_cluster": -0.35,
    "building_area_ratio": -9.0,
    "building_count": -0.08,
    "road_length_m": -0.003,
    "green_ratio": -2.0,
    "water_ratio": 1.2,
    "avg_building_area_m2": -0.0008,
    "site_count_250m": 0.15,
    "site_count_500m": 0.08,
    "serving_distance_m": -0.0035,
    "nearest_site_distance_m": -0.0015,
    "azimuth_delta_deg": -0.018,
    "mean_nearest3_site_distance_m": 0.0008,
    "dense_urban_far_base": -2.8,
    "dense_urban_far_slope": -0.004,
    "urban_off_axis_slope": -0.015,
    "far_serving_off_axis_base": -1.2,
    "far_serving_distance_slope": -0.004,
    "far_serving_azimuth_slope": -0.010,
    "high_building_far_base": -1.1,
    "high_building_area_slope": -0.0012,
    "high_building_distance_slope": -0.0030,
    "vegetation_far_base": -0.8,
    "vegetation_green_slope": -2.2,
    "water_open_base": 0.9,
    "water_open_distance_slope": 0.0015,
    "dense_site_base": 0.7,
    "dense_site_count_slope": 0.06,
    "cluster_dense_urban_base": -1.4,
    "cluster_dense_urban_slope": -0.35,
    "nlos_flag": -1.2,
    "los_blocker_count": -0.35,
    "los_blocked_ratio": -2.2,
    "max_blocker_height_m": -0.02,
    "diffraction_proxy_db": -0.20,
    "terrain_slope_deg": -0.02,
    "terrain_relief_to_site_m": -0.008,
    "blocker_penalty_cap": -4.0,
    "diffraction_penalty_cap": -3.0,
    "terrain_penalty_cap": -2.0,
    "combined_rf_penalty_cap": -6.0,
    "interference_gap_threshold_db": 3.0,
    "interference_gap_penalty_slope": -0.22,
    "interference_gap_bonus_slope": 0.08,
    "interference_gap_bonus_high_db": 15.0,
    "interference_ratio_linear": -0.45,
    "interference_ratio_clip_max": 2.0,
    "rsrp_phys_weight": 0.28,
    "rsrp_geo_weight": 0.55,
    "rsrq_phys_weight": 0.24,
    "rsrq_geo_weight": 0.18,
    "rsrq_geo_fallback_weight": 0.22,
    "sinr_phys_weight": 0.18,
    "sinr_geo_weight": 0.08,
    "sinr_geo_fallback_weight": 0.12,
    "sinr_geo_offset_clip_low": -8.0,
    "sinr_geo_offset_clip_high": 6.0,
    "sinr_clip_min": -25.0,
}

SEARCH_SPACE = {
    "clutter_Dense Urban": (-8.0, -1.0),
    "clutter_Urban": (-5.0, 0.0),
    "clutter_Suburban": (-3.0, 1.5),
    "clutter_Vegetation": (-4.0, 0.0),
    "clutter_Water": (-1.0, 2.5),
    "clutter_Rural/Open": (-1.0, 2.0),
    "morphology_cluster": (-1.2, 0.8),
    "building_area_ratio": (-16.0, 2.0),
    "building_count": (-0.25, 0.1),
    "road_length_m": (-0.01, 0.002),
    "green_ratio": (-4.0, 1.0),
    "water_ratio": (-1.0, 3.5),
    "avg_building_area_m2": (-0.003, 0.0005),
    "site_count_250m": (-0.2, 0.4),
    "site_count_500m": (-0.15, 0.25),
    "serving_distance_m": (-0.008, 0.001),
    "nearest_site_distance_m": (-0.004, 0.001),
    "azimuth_delta_deg": (-0.06, 0.005),
    "mean_nearest3_site_distance_m": (-0.001, 0.002),
    "dense_urban_far_base": (-5.0, 0.0),
    "dense_urban_far_slope": (-0.01, 0.002),
    "urban_off_axis_slope": (-0.04, 0.002),
    "far_serving_off_axis_base": (-3.5, 0.0),
    "far_serving_distance_slope": (-0.01, 0.001),
    "far_serving_azimuth_slope": (-0.03, 0.001),
    "high_building_far_base": (-3.0, 0.0),
    "high_building_area_slope": (-0.004, 0.0005),
    "high_building_distance_slope": (-0.01, 0.0005),
    "vegetation_far_base": (-2.0, 0.5),
    "vegetation_green_slope": (-4.0, 0.5),
    "water_open_base": (-0.5, 2.5),
    "water_open_distance_slope": (-0.002, 0.005),
    "dense_site_base": (-0.5, 2.5),
    "dense_site_count_slope": (-0.05, 0.20),
    "cluster_dense_urban_base": (-3.0, 0.0),
    "cluster_dense_urban_slope": (-1.0, 0.2),
    "nlos_flag": (-3.0, 0.0),
    "los_blocker_count": (-1.0, 0.05),
    "los_blocked_ratio": (-5.0, 0.5),
    "max_blocker_height_m": (-0.08, 0.01),
    "diffraction_proxy_db": (-0.8, 0.05),
    "terrain_slope_deg": (-0.08, 0.01),
    "terrain_relief_to_site_m": (-0.03, 0.005),
    "blocker_penalty_cap": (-6.0, -1.5),
    "diffraction_penalty_cap": (-5.0, -0.5),
    "terrain_penalty_cap": (-4.0, -0.5),
    "combined_rf_penalty_cap": (-8.0, -2.0),
    "interference_gap_threshold_db": (1.0, 6.0),
    "interference_gap_penalty_slope": (-0.8, -0.05),
    "interference_gap_bonus_slope": (0.0, 0.2),
    "interference_gap_bonus_high_db": (8.0, 20.0),
    "interference_ratio_linear": (-1.2, 0.0),
    "interference_ratio_clip_max": (1.0, 3.0),
    "rsrp_phys_weight": (0.0, 0.7),
    "rsrp_geo_weight": (0.0, 1.0),
    "rsrq_phys_weight": (0.0, 0.7),
    "rsrq_geo_weight": (0.0, 0.8),
    "rsrq_geo_fallback_weight": (0.0, 0.8),
    "sinr_phys_weight": (0.0, 0.8),
    "sinr_geo_weight": (0.0, 0.8),
    "sinr_geo_fallback_weight": (0.0, 0.9),
    "sinr_geo_offset_clip_low": (-15.0, -2.0),
    "sinr_geo_offset_clip_high": (2.0, 12.0),
    "sinr_clip_min": (-30.0, -8.0),
}

COVERAGE_PARAM_NAMES = [
    "clutter_Dense Urban",
    "clutter_Urban",
    "clutter_Suburban",
    "clutter_Vegetation",
    "clutter_Water",
    "clutter_Rural/Open",
    "morphology_cluster",
    "building_area_ratio",
    "building_count",
    "road_length_m",
    "green_ratio",
    "water_ratio",
    "avg_building_area_m2",
    "site_count_250m",
    "site_count_500m",
    "serving_distance_m",
    "nearest_site_distance_m",
    "azimuth_delta_deg",
    "mean_nearest3_site_distance_m",
    "dense_urban_far_base",
    "dense_urban_far_slope",
    "urban_off_axis_slope",
    "far_serving_off_axis_base",
    "far_serving_distance_slope",
    "far_serving_azimuth_slope",
    "high_building_far_base",
    "high_building_area_slope",
    "high_building_distance_slope",
    "vegetation_far_base",
    "vegetation_green_slope",
    "water_open_base",
    "water_open_distance_slope",
    "dense_site_base",
    "dense_site_count_slope",
    "cluster_dense_urban_base",
    "cluster_dense_urban_slope",
    "rsrp_phys_weight",
    "rsrp_geo_weight",
]

INTERFERENCE_PARAM_NAMES = [name for name in SEARCH_SPACE.keys() if name not in set(COVERAGE_PARAM_NAMES)]


@dataclass
class OptimizationConfig:
    project_id: int = DEFAULT_PROJECT_ID
    run_dir: Path | None = None
    run_dirs: Tuple[Path, ...] = tuple()
    output_root: Path = OUTPUT_ROOT
    iterations: int = 40
    seed: int = 42
    holdout_fraction: float = 0.2
    cv_splits: int = 5
    cv_repeats: int = 3
    regularization_lambda: float = 0.05
    patience: int = 12
    min_improvement: float = 0.002
    warmup_random: int = 12
    candidate_pool_size: int = 256
    search_method: str = "bayes"
    top_k: int = 5
    rsrp_objective_weight: float = 0.2
    rsrq_objective_weight: float = 0.2
    sinr_objective_weight: float = 0.6
    sinr_guard_tolerance: float = 0.15
    sinr_guard_penalty: float = 0.5
    rsrp_guard_tolerance: float = 0.25
    rsrp_guard_penalty: float = 0.4
    stage1_iterations: int = 24
    stage2_iterations: int = 24
    recent_runs: int = 1
    use_multi_run: bool = False
    per_run_rsrp_guard_tolerance: float = 0.35
    per_run_sinr_guard_tolerance: float = 0.35
    run_stability_penalty: float = 0.25
    target_finetune_iterations: int = 16
    target_drift_fraction: float = 0.10


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _list_runs(project_id: int, output_root: Path) -> List[Path]:
    root = output_root / f"project_{project_id}"
    if not root.exists():
        return []
    runs = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if not (p / "summary.json").exists():
            continue
        if not (p / "rf_accuracy_points.csv").exists():
            continue
        if not (p / "analysis_grid_features.csv").exists():
            continue
        runs.append(p)
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _resolve_run_dirs(config: OptimizationConfig) -> List[Path]:
    def _normalize_runs(paths: List[Path]) -> List[Path]:
        deduped: List[Path] = []
        seen: set[str] = set()
        for p in paths:
            key = str(Path(p))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(Path(p))
        return sorted(deduped, key=lambda p: p.name, reverse=True)

    if config.run_dirs:
        return _normalize_runs([Path(p) for p in config.run_dirs])
    if config.run_dir is not None:
        return [config.run_dir]
    runs = _list_runs(config.project_id, config.output_root)
    if not runs:
        raise FileNotFoundError(f"No saved runs found for project_id={config.project_id}")
    if bool(config.use_multi_run):
        if int(config.recent_runs) <= 0:
            return runs
        take_n = max(1, int(config.recent_runs))
        return runs[:take_n]
    take_n = max(1, int(config.recent_runs))
    return runs[:take_n]


def _load_summary(run_dir: Path) -> Dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _prepare_optimizer_frame(run_dir: Path, run_idx: int = 0) -> pd.DataFrame:
    dt_eval = pd.read_csv(run_dir / "rf_accuracy_points.csv")
    grid_features = pd.read_csv(run_dir / "analysis_grid_features.csv")
    feature_cols = ["grid_id", CLUTTER_FEATURE] + [col for col in NUMERIC_FEATURES if col in grid_features.columns]
    frame = dt_eval.merge(grid_features[feature_cols], on="grid_id", how="left", suffixes=("", "_grid"))
    for col in NUMERIC_FEATURES:
        frame[col] = _safe_numeric(frame, col)
    if CLUTTER_FEATURE not in frame.columns:
        frame[CLUTTER_FEATURE] = "missing"
    frame[CLUTTER_FEATURE] = frame[CLUTTER_FEATURE].fillna("missing").astype(str)
    required_cols = ["RSRP_meas", "RSRQ_meas", "SINR_meas", "pred_rsrp", "pred_rsrq", "pred_sinr"]
    frame = frame.dropna(subset=required_cols).copy()
    if frame.empty:
        raise ValueError("Optimizer input frame is empty after merging DT evaluation with grid features.")
    frame["source_run_name"] = str(run_dir.name)
    frame["source_run_idx"] = int(run_idx)
    return frame.reset_index(drop=True)


def _parameter_names() -> List[str]:
    return list(SEARCH_SPACE.keys())


def _weights_to_vector(weights: Dict[str, float]) -> np.ndarray:
    return np.array([float(weights[name]) for name in _parameter_names()], dtype=float)


def _vector_to_weights(vector: np.ndarray) -> Dict[str, float]:
    weights = dict(CURRENT_WEIGHTS)
    for idx, name in enumerate(_parameter_names()):
        low, high = SEARCH_SPACE[name]
        weights[name] = float(np.clip(vector[idx], low, high))
    return weights


def _normalized_l2_penalty(weights: Dict[str, float]) -> float:
    parts = []
    for name in _parameter_names():
        low, high = SEARCH_SPACE[name]
        width = max(high - low, 1e-9)
        parts.append(((float(weights[name]) - float(CURRENT_WEIGHTS[name])) / width) ** 2)
    return float(np.mean(parts)) if parts else 0.0


def _sample_weights(rng: np.random.Generator) -> Dict[str, float]:
    weights = dict(CURRENT_WEIGHTS)
    for key, (low, high) in SEARCH_SPACE.items():
        weights[key] = float(rng.uniform(low, high))
    return weights


def _sample_weights_subset(
    rng: np.random.Generator,
    base_weights: Dict[str, float],
    mutable_params: List[str],
) -> Dict[str, float]:
    weights = dict(base_weights)
    for key in mutable_params:
        low, high = SEARCH_SPACE[key]
        weights[key] = float(rng.uniform(low, high))
    return weights


def _sample_weights_subset_bounded(
    rng: np.random.Generator,
    base_weights: Dict[str, float],
    mutable_params: List[str],
    drift_fraction: float,
) -> Dict[str, float]:
    weights = dict(base_weights)
    drift = max(0.0, float(drift_fraction))
    for key in mutable_params:
        low, high = SEARCH_SPACE[key]
        center = float(base_weights[key])
        width = float(high - low)
        local_half_span = max(width * drift, 1e-9)
        local_low = max(float(low), center - local_half_span)
        local_high = min(float(high), center + local_half_span)
        weights[key] = float(rng.uniform(local_low, local_high))
    return weights


def _apply_weighted_adjustment(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    clutter_penalty = out[CLUTTER_FEATURE].map(
        {
            "Dense Urban": weights["clutter_Dense Urban"],
            "Urban": weights["clutter_Urban"],
            "Suburban": weights["clutter_Suburban"],
            "Vegetation": weights["clutter_Vegetation"],
            "Water": weights["clutter_Water"],
            "Rural/Open": weights["clutter_Rural/Open"],
        }
    ).fillna(0.0)

    cluster_center = float(out["morphology_cluster"].mean()) if len(out) else 0.0
    cluster_offset = (out["morphology_cluster"] - cluster_center) * float(weights["morphology_cluster"])
    geo_offset = clutter_penalty + cluster_offset
    geo_offset = geo_offset + (_safe_numeric(out, "building_area_ratio").clip(0, 0.8) * float(weights["building_area_ratio"]))
    geo_offset = geo_offset + (_safe_numeric(out, "building_count").clip(0, 30) * float(weights["building_count"]))
    geo_offset = geo_offset + (_safe_numeric(out, "road_length_m").clip(0, 400) * float(weights["road_length_m"]))
    geo_offset = geo_offset + (_safe_numeric(out, "green_ratio").clip(0, 1.0) * float(weights["green_ratio"]))
    geo_offset = geo_offset + (_safe_numeric(out, "water_ratio").clip(0, 1.0) * float(weights["water_ratio"]))
    geo_offset = geo_offset + (_safe_numeric(out, "avg_building_area_m2").clip(0, 3000) * float(weights["avg_building_area_m2"]))
    geo_offset = geo_offset + (_safe_numeric(out, "site_count_250m").clip(0, 12) * float(weights["site_count_250m"]))
    geo_offset = geo_offset + (_safe_numeric(out, "site_count_500m").clip(0, 25) * float(weights["site_count_500m"]))
    geo_offset = geo_offset + (_safe_numeric(out, "serving_distance_m").clip(0, 1200) * float(weights["serving_distance_m"]))
    geo_offset = geo_offset + (_safe_numeric(out, "nearest_site_distance_m").clip(0, 1000) * float(weights["nearest_site_distance_m"]))
    geo_offset = geo_offset + (((_safe_numeric(out, "azimuth_delta_deg").clip(0, 180) / 10.0) ** 1.2) * float(weights["azimuth_delta_deg"]))
    geo_offset = geo_offset + (_safe_numeric(out, "mean_nearest3_site_distance_m").clip(0, 1500) * float(weights["mean_nearest3_site_distance_m"]))

    clutter_series = out[CLUTTER_FEATURE].astype(str)
    nearest_site = _safe_numeric(out, "nearest_site_distance_m")
    serving_distance = _safe_numeric(out, "serving_distance_m")
    azimuth_delta = _safe_numeric(out, "azimuth_delta_deg")
    avg_building_area = _safe_numeric(out, "avg_building_area_m2")
    green_ratio = _safe_numeric(out, "green_ratio")
    site_count_250m = _safe_numeric(out, "site_count_250m")
    morphology_cluster = _safe_numeric(out, "morphology_cluster")

    dense_urban_far_penalty = np.where(
        (clutter_series == "Dense Urban") & (nearest_site > 180.0),
        float(weights["dense_urban_far_base"])
        + (float(weights["dense_urban_far_slope"]) * (nearest_site.clip(180.0, 700.0) - 180.0)),
        0.0,
    )
    urban_off_axis_penalty = np.where(
        azimuth_delta > 45.0,
        float(weights["urban_off_axis_slope"]) * (azimuth_delta.clip(45.0, 180.0) - 45.0),
        0.0,
    )
    far_serving_off_axis_penalty = np.where(
        (serving_distance > 250.0) & (azimuth_delta > 35.0),
        float(weights["far_serving_off_axis_base"])
        + (float(weights["far_serving_distance_slope"]) * (serving_distance.clip(250.0, 1200.0) - 250.0))
        + (float(weights["far_serving_azimuth_slope"]) * (azimuth_delta.clip(35.0, 180.0) - 35.0)),
        0.0,
    )
    high_building_far_penalty = np.where(
        (avg_building_area > 250.0) & (nearest_site > 160.0),
        float(weights["high_building_far_base"])
        + (float(weights["high_building_area_slope"]) * (avg_building_area.clip(250.0, 3000.0) - 250.0))
        + (float(weights["high_building_distance_slope"]) * (nearest_site.clip(160.0, 1000.0) - 160.0)),
        0.0,
    )
    vegetation_far_penalty = np.where(
        (clutter_series == "Vegetation") & (serving_distance > 220.0),
        float(weights["vegetation_far_base"])
        + (float(weights["vegetation_green_slope"]) * green_ratio.clip(0.2, 1.0)),
        0.0,
    )
    water_open_bonus = np.where(
        clutter_series.isin(["Water", "Rural/Open"]) & (azimuth_delta < 20.0) & (nearest_site < 220.0),
        float(weights["water_open_base"])
        + (float(weights["water_open_distance_slope"]) * (220.0 - nearest_site.clip(0.0, 220.0))),
        0.0,
    )
    dense_site_bonus = np.where(
        (site_count_250m >= 4.0) & (nearest_site < 120.0),
        float(weights["dense_site_base"])
        + (float(weights["dense_site_count_slope"]) * site_count_250m.clip(4.0, 12.0)),
        0.0,
    )
    cluster_dense_urban_penalty = np.where(
        (morphology_cluster >= (cluster_center + 1.0)) & (clutter_series == "Dense Urban"),
        float(weights["cluster_dense_urban_base"])
        + (float(weights["cluster_dense_urban_slope"]) * (morphology_cluster - cluster_center).clip(lower=0.0, upper=4.0)),
        0.0,
    )
    nlos_penalty = float(weights["nlos_flag"]) * _safe_numeric(out, "nlos_flag").clip(0, 1)
    blocker_penalty_raw = (
        float(weights["los_blocker_count"]) * _safe_numeric(out, "los_blocker_count").clip(0, 10)
        + float(weights["los_blocked_ratio"]) * _safe_numeric(out, "los_blocked_ratio").clip(0, 1.0)
        + float(weights["max_blocker_height_m"]) * _safe_numeric(out, "max_blocker_height_m").clip(0, 80.0)
    )
    blocker_penalty = pd.Series(blocker_penalty_raw, index=out.index, dtype=float).clip(
        lower=float(weights["blocker_penalty_cap"]),
        upper=0.0,
    )
    diffraction_penalty = pd.Series(
        float(weights["diffraction_proxy_db"]) * _safe_numeric(out, "diffraction_proxy_db").clip(0, 25.0),
        index=out.index,
        dtype=float,
    ).clip(lower=float(weights["diffraction_penalty_cap"]), upper=0.0)
    terrain_penalty_raw = (
        float(weights["terrain_slope_deg"]) * _safe_numeric(out, "terrain_slope_deg").clip(0, 35.0)
        + float(weights["terrain_relief_to_site_m"]) * _safe_numeric(out, "terrain_relief_to_site_m").clip(lower=0.0, upper=180.0)
    )
    terrain_penalty = pd.Series(terrain_penalty_raw, index=out.index, dtype=float).clip(
        lower=float(weights["terrain_penalty_cap"]),
        upper=0.0,
    )
    interference_gap = _safe_numeric(out, "interference_gap_db")
    gap_threshold = float(weights["interference_gap_threshold_db"])
    gap_bonus_high = max(gap_threshold, float(weights["interference_gap_bonus_high_db"]))
    interference_penalty = pd.Series(
        np.where(
            interference_gap < gap_threshold,
            float(weights["interference_gap_penalty_slope"])
            * (gap_threshold - interference_gap.clip(gap_threshold - 15.0, gap_threshold)),
            float(weights["interference_gap_bonus_slope"])
            * (interference_gap.clip(gap_threshold, gap_bonus_high) - gap_threshold),
        ),
        index=out.index,
        dtype=float,
    )
    interference_ratio_penalty = pd.Series(
        float(weights["interference_ratio_linear"])
        * _safe_numeric(out, "interference_ratio_linear").clip(0.0, float(weights["interference_ratio_clip_max"])),
        index=out.index,
        dtype=float,
    )
    combined_rf_penalty = (nlos_penalty + blocker_penalty + diffraction_penalty + terrain_penalty).clip(
        lower=float(weights["combined_rf_penalty_cap"]),
        upper=0.5,
    )

    geo_offset = (
        geo_offset
        + pd.Series(dense_urban_far_penalty, index=out.index, dtype=float)
        + pd.Series(urban_off_axis_penalty, index=out.index, dtype=float)
        + pd.Series(far_serving_off_axis_penalty, index=out.index, dtype=float)
        + pd.Series(high_building_far_penalty, index=out.index, dtype=float)
        + pd.Series(vegetation_far_penalty, index=out.index, dtype=float)
        + pd.Series(water_open_bonus, index=out.index, dtype=float)
        + pd.Series(dense_site_bonus, index=out.index, dtype=float)
        + pd.Series(cluster_dense_urban_penalty, index=out.index, dtype=float)
        + combined_rf_penalty
        + interference_penalty
        + interference_ratio_penalty
    )

    rsrp_base = pd.to_numeric(out["pred_rsrp"], errors="coerce")
    rsrq_base = pd.to_numeric(out["pred_rsrq"], errors="coerce")
    sinr_base = pd.to_numeric(out["pred_sinr"], errors="coerce")
    rsrp_phys = pd.to_numeric(out.get("serving_proxy_rsrp_phys_dbm"), errors="coerce")
    rsrq_phys = pd.to_numeric(out.get("rsrq_proxy_db"), errors="coerce")
    sinr_phys = pd.to_numeric(out.get("sinr_proxy_db"), errors="coerce")

    rsrp_base_weight = max(0.0, 1.0 - float(weights["rsrp_phys_weight"]))
    rsrq_base_weight = max(0.0, 1.0 - float(weights["rsrq_phys_weight"]))
    sinr_base_weight = max(0.0, 1.0 - float(weights["sinr_phys_weight"]))

    out["tuned_geo_offset"] = geo_offset
    out["pred_rsrp_tuned"] = rsrp_base.copy()
    has_rsrp_phys = rsrp_phys.notna()
    out.loc[has_rsrp_phys, "pred_rsrp_tuned"] = (
        (rsrp_base_weight * rsrp_base[has_rsrp_phys])
        + (float(weights["rsrp_phys_weight"]) * rsrp_phys[has_rsrp_phys])
        + (float(weights["rsrp_geo_weight"]) * geo_offset[has_rsrp_phys])
    )
    out.loc[~has_rsrp_phys, "pred_rsrp_tuned"] = rsrp_base[~has_rsrp_phys] + geo_offset[~has_rsrp_phys]

    out["pred_rsrq_tuned"] = rsrq_base.copy()
    has_rsrq_phys = rsrq_phys.notna()
    out.loc[has_rsrq_phys, "pred_rsrq_tuned"] = (
        (rsrq_base_weight * rsrq_base[has_rsrq_phys])
        + (float(weights["rsrq_phys_weight"]) * rsrq_phys[has_rsrq_phys])
        + (float(weights["rsrq_geo_weight"]) * geo_offset[has_rsrq_phys])
    )
    out.loc[~has_rsrq_phys, "pred_rsrq_tuned"] = rsrq_base[~has_rsrq_phys] + (
        geo_offset[~has_rsrq_phys] * float(weights["rsrq_geo_fallback_weight"])
    )

    out["pred_sinr_tuned"] = sinr_base.copy()
    has_sinr_phys = sinr_phys.notna()
    sinr_geo_offset = geo_offset.clip(
        lower=float(weights["sinr_geo_offset_clip_low"]),
        upper=float(weights["sinr_geo_offset_clip_high"]),
    )
    out.loc[has_sinr_phys, "pred_sinr_tuned"] = (
        (sinr_base_weight * sinr_base[has_sinr_phys])
        + (float(weights["sinr_phys_weight"]) * sinr_phys[has_sinr_phys])
        + (float(weights["sinr_geo_weight"]) * sinr_geo_offset[has_sinr_phys])
    )
    out.loc[~has_sinr_phys, "pred_sinr_tuned"] = sinr_base[~has_sinr_phys] + (
        sinr_geo_offset[~has_sinr_phys] * float(weights["sinr_geo_fallback_weight"])
    )

    out["pred_rsrp_tuned"] = out["pred_rsrp_tuned"].clip(-140, -44)
    out["pred_rsrq_tuned"] = out["pred_rsrq_tuned"].clip(-20, -3)
    out["pred_sinr_tuned"] = out["pred_sinr_tuned"].clip(float(weights["sinr_clip_min"]), 30.0)
    return out


def _evaluate_frame(df: pd.DataFrame, pred_suffix: str) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    specs = [
        ("RSRP_meas", f"pred_rsrp{pred_suffix}"),
        ("RSRQ_meas", f"pred_rsrq{pred_suffix}"),
        ("SINR_meas", f"pred_sinr{pred_suffix}"),
    ]
    for meas_col, pred_col in specs:
        valid = df.dropna(subset=[meas_col, pred_col])
        if not valid.empty:
            metrics[meas_col] = _metric_bundle(valid[meas_col], valid[pred_col], metric_key=meas_col)
    return metrics


def _score_metrics(
    metrics: Dict[str, Dict[str, float]],
    config: OptimizationConfig,
) -> float:
    norms = {"RSRP_meas": 10.0, "RSRQ_meas": 3.0, "SINR_meas": 6.0}
    weights = {
        "RSRP_meas": float(config.rsrp_objective_weight),
        "RSRQ_meas": float(config.rsrq_objective_weight),
        "SINR_meas": float(config.sinr_objective_weight),
    }
    weighted_parts: List[float] = []
    used_weights: List[float] = []
    for metric_name, values in metrics.items():
        mae = float(values.get("mae", 1e9))
        within_key = f"within_{str(norms[metric_name]).replace('.', '_')}"
        within_score = float(values.get(within_key, 0.0))
        metric_score = (mae / norms[metric_name]) - within_score
        weight = weights.get(metric_name, 0.0)
        if weight > 0.0:
            weighted_parts.append(metric_score * weight)
            used_weights.append(weight)
    return (float(np.sum(weighted_parts)) / float(np.sum(used_weights))) if used_weights else float("inf")


def _metrics_by_run(df: pd.DataFrame, pred_suffix: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    if "source_run_name" not in df.columns:
        return {}
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for run_name, run_df in df.groupby("source_run_name", dropna=False):
        out[str(run_name)] = _evaluate_frame(run_df.reset_index(drop=True), pred_suffix)
    return out


def _run_metric_stability_penalty(
    metrics_by_run: Dict[str, Dict[str, Dict[str, float]]],
    config: OptimizationConfig,
) -> float:
    if len(metrics_by_run) <= 1:
        return 0.0
    penalties: List[float] = []
    for metric_name in ("RSRP_meas", "RSRQ_meas", "SINR_meas"):
        maes = [
            float(run_metrics.get(metric_name, {}).get("mae", np.nan))
            for run_metrics in metrics_by_run.values()
        ]
        maes = [v for v in maes if np.isfinite(v)]
        if len(maes) > 1:
            penalties.append(float(np.std(maes)))
    if not penalties:
        return 0.0
    return float(config.run_stability_penalty) * float(np.mean(penalties))


def _per_run_guard_penalty(
    metrics_by_run: Dict[str, Dict[str, Dict[str, float]]],
    reference_by_run: Dict[str, Dict[str, Dict[str, float]]] | None,
    config: OptimizationConfig,
) -> float:
    if not metrics_by_run or not reference_by_run:
        return 0.0
    penalty = 0.0
    for run_name, candidate_metrics in metrics_by_run.items():
        reference_metrics = reference_by_run.get(run_name)
        if not reference_metrics:
            continue
        candidate_rsrp = float(candidate_metrics.get("RSRP_meas", {}).get("mae", np.nan))
        reference_rsrp = float(reference_metrics.get("RSRP_meas", {}).get("mae", np.nan))
        if np.isfinite(candidate_rsrp) and np.isfinite(reference_rsrp):
            allowed_rsrp = reference_rsrp + float(config.per_run_rsrp_guard_tolerance)
            if candidate_rsrp > allowed_rsrp:
                penalty += float(config.rsrp_guard_penalty) + (candidate_rsrp - allowed_rsrp)
        candidate_sinr = float(candidate_metrics.get("SINR_meas", {}).get("mae", np.nan))
        reference_sinr = float(reference_metrics.get("SINR_meas", {}).get("mae", np.nan))
        if np.isfinite(candidate_sinr) and np.isfinite(reference_sinr):
            allowed_sinr = reference_sinr + float(config.per_run_sinr_guard_tolerance)
            if candidate_sinr > allowed_sinr:
                penalty += float(config.sinr_guard_penalty) + (candidate_sinr - allowed_sinr)
    return float(penalty)


def _sinr_guard_adjusted_score(
    base_score: float,
    metrics: Dict[str, Dict[str, float]],
    reference_metrics: Dict[str, Dict[str, float]] | None,
    config: OptimizationConfig,
) -> float:
    if not reference_metrics:
        return base_score
    candidate_sinr = float(metrics.get("SINR_meas", {}).get("mae", np.nan))
    reference_sinr = float(reference_metrics.get("SINR_meas", {}).get("mae", np.nan))
    if not np.isfinite(candidate_sinr) or not np.isfinite(reference_sinr):
        return base_score
    allowed = reference_sinr + float(config.sinr_guard_tolerance)
    if candidate_sinr <= allowed:
        return base_score
    overrun = candidate_sinr - allowed
    return float(base_score + float(config.sinr_guard_penalty) + (0.5 * overrun))


def _rsrp_guard_adjusted_score(
    base_score: float,
    metrics: Dict[str, Dict[str, float]],
    reference_metrics: Dict[str, Dict[str, float]] | None,
    config: OptimizationConfig,
) -> float:
    if not reference_metrics:
        return base_score
    candidate_rsrp = float(metrics.get("RSRP_meas", {}).get("mae", np.nan))
    reference_rsrp = float(reference_metrics.get("RSRP_meas", {}).get("mae", np.nan))
    if not np.isfinite(candidate_rsrp) or not np.isfinite(reference_rsrp):
        return base_score
    allowed = reference_rsrp + float(config.rsrp_guard_tolerance)
    if candidate_rsrp <= allowed:
        return base_score
    overrun = candidate_rsrp - allowed
    return float(base_score + float(config.rsrp_guard_penalty) + (0.5 * overrun))


def _score_candidate_cv(
    train_df: pd.DataFrame,
    weights: Dict[str, float],
    config: OptimizationConfig,
    cv_splits: int,
    cv_repeats: int,
    regularization_lambda: float,
    seed: int,
    reference_metrics: Dict[str, Dict[str, float]] | None = None,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    use_group_cv = "source_run_name" in train_df.columns and train_df["source_run_name"].nunique() > 1
    if use_group_cv:
        group_splits = min(int(cv_splits), int(train_df["source_run_name"].nunique()))
        splitter = GroupKFold(n_splits=max(2, group_splits))
        split_iter = splitter.split(train_df, groups=train_df["source_run_name"].astype(str))
    else:
        splitter = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=seed)
        split_iter = splitter.split(train_df)
    fold_scores: List[float] = []
    metric_rows: Dict[str, List[Dict[str, float]]] = {"RSRP_meas": [], "RSRQ_meas": [], "SINR_meas": []}

    for _, valid_idx in split_iter:
        fold_valid = train_df.iloc[valid_idx].reset_index(drop=True)
        tuned_valid = _apply_weighted_adjustment(fold_valid, weights)
        valid_metrics = _evaluate_frame(tuned_valid, "_tuned")
        fold_score = _score_metrics(valid_metrics, config)
        fold_score = _sinr_guard_adjusted_score(fold_score, valid_metrics, reference_metrics, config)
        fold_score = _rsrp_guard_adjusted_score(fold_score, valid_metrics, reference_metrics, config)
        valid_metrics_by_run = _metrics_by_run(tuned_valid, "_tuned")
        reference_by_run = _metrics_by_run(fold_valid, "") if reference_metrics is None else {}
        fold_score += _run_metric_stability_penalty(valid_metrics_by_run, config)
        fold_score += _per_run_guard_penalty(valid_metrics_by_run, reference_by_run, config)
        fold_scores.append(fold_score)
        for metric_name in metric_rows:
            if metric_name in valid_metrics:
                metric_rows[metric_name].append(valid_metrics[metric_name])

    averaged_metrics: Dict[str, Dict[str, float]] = {}
    for metric_name, rows in metric_rows.items():
        if not rows:
            continue
        averaged_metrics[metric_name] = {
            key: round(float(np.mean([row[key] for row in rows if row.get(key) is not None])), 4)
            for key in rows[0].keys()
            if any(row.get(key) is not None for row in rows)
        }

    cv_score = float(np.mean(fold_scores)) if fold_scores else float("inf")
    penalty = regularization_lambda * _normalized_l2_penalty(weights)
    return cv_score + penalty, averaged_metrics


def _propose_bayesian_candidate(
    tried_vectors: List[np.ndarray],
    tried_scores: List[float],
    rng: np.random.Generator,
    candidate_pool_size: int,
) -> Dict[str, float]:
    if len(tried_vectors) < 5:
        return _sample_weights(rng)

    X = np.vstack(tried_vectors)
    y = np.array(tried_scores, dtype=float)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
        normalize_y=True,
        random_state=0,
    )
    gp.fit(X, y)

    pool = np.vstack([_weights_to_vector(_sample_weights(rng)) for _ in range(candidate_pool_size)])
    mean_pred, std_pred = gp.predict(pool, return_std=True)
    acquisition = mean_pred - (0.35 * std_pred)
    return _vector_to_weights(pool[int(np.argmin(acquisition))])


def _weights_to_vector_subset(weights: Dict[str, float], mutable_params: List[str]) -> np.ndarray:
    return np.array([float(weights[name]) for name in mutable_params], dtype=float)


def _vector_to_weights_subset(base_weights: Dict[str, float], vector: np.ndarray, mutable_params: List[str]) -> Dict[str, float]:
    weights = dict(base_weights)
    for idx, name in enumerate(mutable_params):
        low, high = SEARCH_SPACE[name]
        weights[name] = float(np.clip(vector[idx], low, high))
    return weights


def _propose_bayesian_candidate_subset(
    tried_vectors: List[np.ndarray],
    tried_scores: List[float],
    rng: np.random.Generator,
    candidate_pool_size: int,
    base_weights: Dict[str, float],
    mutable_params: List[str],
) -> Dict[str, float]:
    if len(tried_vectors) < 5:
        return _sample_weights_subset(rng, base_weights, mutable_params)

    X = np.vstack(tried_vectors)
    y = np.array(tried_scores, dtype=float)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
        normalize_y=True,
        random_state=0,
    )
    gp.fit(X, y)

    pool = np.vstack([
        _weights_to_vector_subset(_sample_weights_subset(rng, base_weights, mutable_params), mutable_params)
        for _ in range(candidate_pool_size)
    ])
    mean_pred, std_pred = gp.predict(pool, return_std=True)
    acquisition = mean_pred - (0.35 * std_pred)
    return _vector_to_weights_subset(base_weights, pool[int(np.argmin(acquisition))], mutable_params)


def _propose_bayesian_candidate_subset_bounded(
    tried_vectors: List[np.ndarray],
    tried_scores: List[float],
    rng: np.random.Generator,
    candidate_pool_size: int,
    base_weights: Dict[str, float],
    mutable_params: List[str],
    drift_fraction: float,
) -> Dict[str, float]:
    if len(tried_vectors) < 5:
        return _sample_weights_subset_bounded(rng, base_weights, mutable_params, drift_fraction)

    X = np.vstack(tried_vectors)
    y = np.array(tried_scores, dtype=float)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
        normalize_y=True,
        random_state=0,
    )
    gp.fit(X, y)

    pool = np.vstack([
        _weights_to_vector_subset(
            _sample_weights_subset_bounded(rng, base_weights, mutable_params, drift_fraction),
            mutable_params,
        )
        for _ in range(candidate_pool_size)
    ])
    mean_pred, std_pred = gp.predict(pool, return_std=True)
    acquisition = mean_pred - (0.35 * std_pred)
    return _vector_to_weights_subset(base_weights, pool[int(np.argmin(acquisition))], mutable_params)


def _format_weight_table(weights: Dict[str, float]) -> pd.DataFrame:
    rows = [{"parameter": key, "value": value} for key, value in weights.items() if key != "cluster_center_mode"]
    return pd.DataFrame(rows).sort_values("parameter").reset_index(drop=True)


def _run_stage_search(
    *,
    stage_name: str,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    start_weights: Dict[str, float],
    mutable_params: List[str],
    config: OptimizationConfig,
    rng: np.random.Generator,
    iterations: int,
    reference_train_metrics: Dict[str, Dict[str, float]] | None = None,
    reference_holdout_metrics: Dict[str, Dict[str, float]] | None = None,
    bounded_drift_fraction: float | None = None,
) -> tuple[Dict[str, float], float, Dict[str, Dict[str, float]], pd.DataFrame]:
    current_train_metrics = _evaluate_frame(_apply_weighted_adjustment(train_df, start_weights), "_tuned")
    current_holdout_metrics = _evaluate_frame(_apply_weighted_adjustment(holdout_df, start_weights), "_tuned")
    current_train_by_run = _metrics_by_run(_apply_weighted_adjustment(train_df, start_weights), "_tuned")
    current_holdout_by_run = _metrics_by_run(_apply_weighted_adjustment(holdout_df, start_weights), "_tuned")
    current_cv_score, current_cv_metrics = _score_candidate_cv(
        train_df,
        start_weights,
        config,
        cv_splits=config.cv_splits,
        cv_repeats=config.cv_repeats,
        regularization_lambda=config.regularization_lambda,
        seed=config.seed,
        reference_metrics=reference_train_metrics,
    )

    tried_vectors: List[np.ndarray] = [_weights_to_vector_subset(start_weights, mutable_params)]
    tried_scores: List[float] = [current_cv_score]
    best_weights = dict(start_weights)
    best_cv_score = current_cv_score
    best_cv_metrics = current_cv_metrics
    rows: List[Dict[str, float]] = []
    no_improvement_rounds = 0

    for idx in range(iterations):
        if idx < min(config.warmup_random, iterations) or config.search_method == "random":
            if bounded_drift_fraction is None:
                weights = _sample_weights_subset(rng, start_weights, mutable_params)
            else:
                weights = _sample_weights_subset_bounded(
                    rng,
                    start_weights,
                    mutable_params,
                    bounded_drift_fraction,
                )
            search_label = "random"
        else:
            if bounded_drift_fraction is None:
                weights = _propose_bayesian_candidate_subset(
                    tried_vectors,
                    tried_scores,
                    rng,
                    config.candidate_pool_size,
                    start_weights,
                    mutable_params,
                )
            else:
                weights = _propose_bayesian_candidate_subset_bounded(
                    tried_vectors,
                    tried_scores,
                    rng,
                    config.candidate_pool_size,
                    start_weights,
                    mutable_params,
                    bounded_drift_fraction,
                )
            search_label = "bayes"

        tuned_train = _apply_weighted_adjustment(train_df, weights)
        tuned_holdout = _apply_weighted_adjustment(holdout_df, weights)
        train_metrics = _evaluate_frame(tuned_train, "_tuned")
        holdout_metrics = _evaluate_frame(tuned_holdout, "_tuned")
        train_metrics_by_run = _metrics_by_run(tuned_train, "_tuned")
        holdout_metrics_by_run = _metrics_by_run(tuned_holdout, "_tuned")
        cv_score, cv_metrics = _score_candidate_cv(
            train_df,
            weights,
            config,
            cv_splits=config.cv_splits,
            cv_repeats=config.cv_repeats,
            regularization_lambda=config.regularization_lambda,
            seed=config.seed + idx + 1,
            reference_metrics=reference_train_metrics,
        )

        train_score = _score_metrics(train_metrics, config)
        holdout_score = _score_metrics(holdout_metrics, config)
        if reference_train_metrics:
            train_score = _sinr_guard_adjusted_score(train_score, train_metrics, reference_train_metrics, config)
            train_score = _rsrp_guard_adjusted_score(train_score, train_metrics, reference_train_metrics, config)
        if reference_holdout_metrics:
            holdout_score = _sinr_guard_adjusted_score(holdout_score, holdout_metrics, reference_holdout_metrics, config)
            holdout_score = _rsrp_guard_adjusted_score(holdout_score, holdout_metrics, reference_holdout_metrics, config)
        train_score += _run_metric_stability_penalty(train_metrics_by_run, config)
        holdout_score += _run_metric_stability_penalty(holdout_metrics_by_run, config)
        train_score += _per_run_guard_penalty(train_metrics_by_run, current_train_by_run if current_train_by_run else None, config)
        holdout_score += _per_run_guard_penalty(holdout_metrics_by_run, current_holdout_by_run if current_holdout_by_run else None, config)

        row = {
            "stage": stage_name,
            "candidate": idx + 1,
            "search_method": search_label,
            "train_score": train_score,
            "cv_score": cv_score,
            "holdout_score": holdout_score,
            "train_rsrp_mae": train_metrics.get("RSRP_meas", {}).get("mae"),
            "holdout_rsrp_mae": holdout_metrics.get("RSRP_meas", {}).get("mae"),
            "train_rsrq_mae": train_metrics.get("RSRQ_meas", {}).get("mae"),
            "holdout_rsrq_mae": holdout_metrics.get("RSRQ_meas", {}).get("mae"),
            "train_sinr_mae": train_metrics.get("SINR_meas", {}).get("mae"),
            "holdout_sinr_mae": holdout_metrics.get("SINR_meas", {}).get("mae"),
        }
        for key in mutable_params:
            row[key] = weights[key]
        rows.append(row)

        tried_vectors.append(_weights_to_vector_subset(weights, mutable_params))
        tried_scores.append(cv_score)
        if cv_score < (best_cv_score - config.min_improvement):
            best_cv_score = cv_score
            best_weights = dict(weights)
            best_cv_metrics = cv_metrics
            no_improvement_rounds = 0
        else:
            no_improvement_rounds += 1
            if no_improvement_rounds >= config.patience:
                print(f"[OPT][{stage_name}] Early stop at candidate {idx + 1} due to no CV improvement.")
                break

    return best_weights, best_cv_score, best_cv_metrics, pd.DataFrame(rows)


def run_geo_weight_optimizer(config: OptimizationConfig) -> Path:
    run_dirs = _resolve_run_dirs(config)
    primary_run_dir = run_dirs[0]
    summary = _load_summary(primary_run_dir)
    optimizer_dir = primary_run_dir / f"optimizer_{_timestamp()}"
    optimizer_dir.mkdir(parents=True, exist_ok=True)

    frames = [_prepare_optimizer_frame(run_dir, run_idx=idx) for idx, run_dir in enumerate(run_dirs)]
    frame = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    rng = np.random.default_rng(config.seed)
    if "source_run_name" in frame.columns and frame["source_run_name"].nunique() > 1:
        holdout_parts: List[pd.DataFrame] = []
        train_parts: List[pd.DataFrame] = []
        for _, run_part in frame.groupby("source_run_name", dropna=False):
            run_part = run_part.sample(frac=1.0, random_state=int(config.seed)).reset_index(drop=True)
            holdout_size = max(1, int(len(run_part) * config.holdout_fraction))
            holdout_parts.append(run_part.iloc[:holdout_size].copy())
            train_parts.append(run_part.iloc[holdout_size:].copy())
        train_df = pd.concat(train_parts, ignore_index=True).reset_index(drop=True)
        holdout_df = pd.concat(holdout_parts, ignore_index=True).reset_index(drop=True)
    else:
        shuffled_idx = rng.permutation(len(frame))
        holdout_size = max(1, int(len(frame) * config.holdout_fraction))
        holdout_idx = shuffled_idx[:holdout_size]
        train_idx = shuffled_idx[holdout_size:]
        train_df = frame.iloc[train_idx].reset_index(drop=True)
        holdout_df = frame.iloc[holdout_idx].reset_index(drop=True)
    if train_df.empty or holdout_df.empty:
        raise ValueError("Optimizer split failed: train or holdout split is empty.")

    baseline_train = _evaluate_frame(train_df, "")
    baseline_holdout = _evaluate_frame(holdout_df, "")
    baseline_full = _evaluate_frame(frame, "")

    current_train = _evaluate_frame(_apply_weighted_adjustment(train_df, CURRENT_WEIGHTS), "_tuned")
    current_holdout = _evaluate_frame(_apply_weighted_adjustment(holdout_df, CURRENT_WEIGHTS), "_tuned")
    current_full = _evaluate_frame(_apply_weighted_adjustment(frame, CURRENT_WEIGHTS), "_tuned")
    current_cv_score, current_cv_metrics = _score_candidate_cv(
        train_df,
        CURRENT_WEIGHTS,
        config,
        cv_splits=config.cv_splits,
        cv_repeats=config.cv_repeats,
        regularization_lambda=config.regularization_lambda,
        seed=config.seed,
    )

    coverage_stage_config = copy.deepcopy(config)
    coverage_stage_config.rsrp_objective_weight = 0.75
    coverage_stage_config.rsrq_objective_weight = 0.15
    coverage_stage_config.sinr_objective_weight = 0.10
    coverage_stage_config.sinr_guard_tolerance = 1.5
    coverage_stage_config.sinr_guard_penalty = 0.0

    stage1_weights, stage1_cv_score, stage1_cv_metrics, stage1_leaderboard = _run_stage_search(
        stage_name="coverage_lock",
        train_df=train_df,
        holdout_df=holdout_df,
        start_weights=CURRENT_WEIGHTS,
        mutable_params=COVERAGE_PARAM_NAMES,
        config=coverage_stage_config,
        rng=rng,
        iterations=max(1, int(config.stage1_iterations)),
        reference_train_metrics=None,
        reference_holdout_metrics=None,
    )
    stage1_train = _evaluate_frame(_apply_weighted_adjustment(train_df, stage1_weights), "_tuned")
    stage1_holdout = _evaluate_frame(_apply_weighted_adjustment(holdout_df, stage1_weights), "_tuned")
    stage1_full = _evaluate_frame(_apply_weighted_adjustment(frame, stage1_weights), "_tuned")

    stage2_weights, stage2_cv_score, stage2_cv_metrics, stage2_leaderboard = _run_stage_search(
        stage_name="interference_tune",
        train_df=train_df,
        holdout_df=holdout_df,
        start_weights=stage1_weights,
        mutable_params=INTERFERENCE_PARAM_NAMES,
        config=config,
        rng=rng,
        iterations=max(1, int(config.stage2_iterations)),
        reference_train_metrics=stage1_train,
        reference_holdout_metrics=stage1_holdout,
    )

    best_weights = dict(stage2_weights)
    best_cv_score = stage2_cv_score
    leaderboard_df = (
        pd.concat([stage1_leaderboard, stage2_leaderboard], ignore_index=True)
        .sort_values(["cv_score", "holdout_score", "train_score"])
        .reset_index(drop=True)
    )
    global_train = _evaluate_frame(_apply_weighted_adjustment(train_df, best_weights), "_tuned")
    global_holdout = _evaluate_frame(_apply_weighted_adjustment(holdout_df, best_weights), "_tuned")
    global_full_df = _apply_weighted_adjustment(frame, best_weights)
    global_full = _evaluate_frame(global_full_df, "_tuned")
    global_cv_score_final, global_cv_metrics = _score_candidate_cv(
        train_df,
        best_weights,
        config,
        cv_splits=config.cv_splits,
        cv_repeats=config.cv_repeats,
        regularization_lambda=config.regularization_lambda,
        seed=config.seed,
        reference_metrics=stage1_train,
    )

    target_frame = frames[0].copy().reset_index(drop=True)
    target_rng = np.random.default_rng(config.seed + 1000)
    target_idx = target_rng.permutation(len(target_frame))
    target_holdout_size = max(1, int(len(target_frame) * config.holdout_fraction))
    target_holdout_df = target_frame.iloc[target_idx[:target_holdout_size]].reset_index(drop=True)
    target_train_df = target_frame.iloc[target_idx[target_holdout_size:]].reset_index(drop=True)
    if target_train_df.empty or target_holdout_df.empty:
        raise ValueError("Target run split failed: train or holdout split is empty.")

    target_reference_train = _evaluate_frame(_apply_weighted_adjustment(target_train_df, best_weights), "_tuned")
    target_reference_holdout = _evaluate_frame(_apply_weighted_adjustment(target_holdout_df, best_weights), "_tuned")
    hybrid_weights, hybrid_cv_score, hybrid_cv_metrics, hybrid_leaderboard = _run_stage_search(
        stage_name="target_finetune",
        train_df=target_train_df,
        holdout_df=target_holdout_df,
        start_weights=best_weights,
        mutable_params=_parameter_names(),
        config=config,
        rng=np.random.default_rng(config.seed + 2000),
        iterations=max(1, int(config.target_finetune_iterations)),
        reference_train_metrics=target_reference_train,
        reference_holdout_metrics=target_reference_holdout,
        bounded_drift_fraction=float(config.target_drift_fraction),
    )

    best_weights = dict(hybrid_weights)
    best_cv_score = hybrid_cv_score
    leaderboard_df = (
        pd.concat([leaderboard_df, hybrid_leaderboard], ignore_index=True)
        .sort_values(["cv_score", "holdout_score", "train_score"])
        .reset_index(drop=True)
    )
    leaderboard_df.to_csv(optimizer_dir / "leaderboard.csv", index=False)

    best_train = _evaluate_frame(_apply_weighted_adjustment(train_df, best_weights), "_tuned")
    best_holdout = _evaluate_frame(_apply_weighted_adjustment(holdout_df, best_weights), "_tuned")
    best_full_df = _apply_weighted_adjustment(frame, best_weights)
    best_full = _evaluate_frame(best_full_df, "_tuned")
    best_cv_score_final, best_cv_metrics = _score_candidate_cv(
        train_df,
        best_weights,
        config,
        cv_splits=config.cv_splits,
        cv_repeats=config.cv_repeats,
        regularization_lambda=config.regularization_lambda,
        seed=config.seed,
        reference_metrics=stage1_train,
    )
    hybrid_train = _evaluate_frame(_apply_weighted_adjustment(target_train_df, best_weights), "_tuned")
    hybrid_holdout = _evaluate_frame(_apply_weighted_adjustment(target_holdout_df, best_weights), "_tuned")
    hybrid_full = _evaluate_frame(_apply_weighted_adjustment(target_frame, best_weights), "_tuned")

    export_cols = [
        "session_id",
        "lat",
        "lon",
        "grid_id",
        "Node_Cell_ID",
        "RSRP_meas",
        "RSRQ_meas",
        "SINR_meas",
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "pred_rsrp_tuned",
        "pred_rsrq_tuned",
        "pred_sinr_tuned",
        "tuned_geo_offset",
        CLUTTER_FEATURE,
    ] + [col for col in NUMERIC_FEATURES if col in best_full_df.columns]
    best_full_df[[col for col in export_cols if col in best_full_df.columns]].to_csv(optimizer_dir / "tuned_eval_points.csv", index=False)
    _format_weight_table(best_weights).to_csv(optimizer_dir / "best_weights.csv", index=False)

    result_summary = {
        "source_run_dir": str(primary_run_dir),
        "source_run_name": primary_run_dir.name,
        "source_run_dirs": [str(p) for p in run_dirs],
        "optimizer_dir": str(optimizer_dir),
        "project_id": config.project_id,
        "iterations_requested": config.iterations,
        "iterations_completed": len(leaderboard_df),
        "seed": config.seed,
        "holdout_fraction": config.holdout_fraction,
        "cv_splits": config.cv_splits,
        "cv_repeats": config.cv_repeats,
        "regularization_lambda": config.regularization_lambda,
        "patience": config.patience,
        "min_improvement": config.min_improvement,
        "warmup_random": config.warmup_random,
        "candidate_pool_size": config.candidate_pool_size,
        "search_method": config.search_method,
        "rows": {"full": len(frame), "train": len(train_df), "holdout": len(holdout_df), "runs": len(run_dirs)},
        "baseline_metrics": {"train": baseline_train, "holdout": baseline_holdout, "full": baseline_full},
        "current_fixed_weight_metrics": {
            "train": current_train,
            "cv": current_cv_metrics,
            "cv_score": current_cv_score,
            "holdout": current_holdout,
            "holdout_score": _score_metrics(current_holdout, config),
            "full": current_full,
        },
        "stage1_coverage_lock_metrics": {
            "train": stage1_train,
            "cv": stage1_cv_metrics,
            "cv_score": stage1_cv_score,
            "holdout": stage1_holdout,
            "holdout_score": _score_metrics(stage1_holdout, coverage_stage_config),
            "full": stage1_full,
        },
        "stage2_global_metrics": {
            "train": global_train,
            "cv": global_cv_metrics,
            "cv_score": global_cv_score_final,
            "holdout": global_holdout,
            "full": global_full,
        },
        "target_finetune_metrics": {
            "train": hybrid_train,
            "cv": hybrid_cv_metrics,
            "cv_score": hybrid_cv_score,
            "holdout": hybrid_holdout,
            "full": hybrid_full,
            "target_run_name": str(primary_run_dir.name),
            "target_rows": {"full": len(target_frame), "train": len(target_train_df), "holdout": len(target_holdout_df)},
        },
        "best_tuned_metrics": {
            "train": best_train,
            "cv": best_cv_metrics,
            "cv_score": best_cv_score_final,
            "holdout": best_holdout,
            "holdout_score": _sinr_guard_adjusted_score(
                _score_metrics(best_holdout, config),
                best_holdout,
                current_holdout,
                config,
            ),
            "full": best_full,
        },
        "best_weights": {key: value for key, value in best_weights.items() if key != "cluster_center_mode"},
        "top_candidates": leaderboard_df.head(config.top_k).to_dict(orient="records"),
        "source_summary_metrics": summary.get("full_metrics", {}),
    }
    (optimizer_dir / "summary.json").write_text(json.dumps(result_summary, indent=2, default=str), encoding="utf-8")

    print(f"[OPT] Source run: {primary_run_dir}")
    print(f"[OPT] Source run count: {len(run_dirs)}")
    print(f"[OPT] Optimizer output: {optimizer_dir}")
    print(f"[OPT] Baseline RSRP MAE(full)={baseline_full.get('RSRP_meas', {}).get('mae')}")
    print(f"[OPT] Current Geo RSRP MAE(holdout)={current_holdout.get('RSRP_meas', {}).get('mae')}")
    print(f"[OPT] Tuned Geo RSRP MAE(holdout)={best_holdout.get('RSRP_meas', {}).get('mae')}")
    print(f"[OPT] Current Geo CV score={round(current_cv_score, 6)}")
    print(f"[OPT] Tuned Geo CV score={round(best_cv_score_final, 6)}")
    return optimizer_dir


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test-only optimizer for LTE RF geo-adjustment weights")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--regularization-lambda", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-improvement", type=float, default=0.002)
    parser.add_argument("--warmup-random", type=int, default=12)
    parser.add_argument("--candidate-pool-size", type=int, default=256)
    parser.add_argument("--search-method", type=str, choices=["random", "bayes"], default="bayes")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rsrp-objective-weight", type=float, default=0.2)
    parser.add_argument("--rsrq-objective-weight", type=float, default=0.2)
    parser.add_argument("--sinr-objective-weight", type=float, default=0.6)
    parser.add_argument("--sinr-guard-tolerance", type=float, default=0.15)
    parser.add_argument("--sinr-guard-penalty", type=float, default=0.5)
    parser.add_argument("--rsrp-guard-tolerance", type=float, default=0.25)
    parser.add_argument("--rsrp-guard-penalty", type=float, default=0.4)
    parser.add_argument("--stage1-iterations", type=int, default=24)
    parser.add_argument("--stage2-iterations", type=int, default=24)
    parser.add_argument("--target-finetune-iterations", type=int, default=16)
    parser.add_argument("--target-drift-fraction", type=float, default=0.10)
    parser.add_argument("--recent-runs", type=int, default=0)
    parser.add_argument("--use-multi-run", action="store_true")
    parser.add_argument("--per-run-rsrp-guard-tolerance", type=float, default=0.35)
    parser.add_argument("--per-run-sinr-guard-tolerance", type=float, default=0.35)
    parser.add_argument("--run-stability-penalty", type=float, default=0.25)
    args = parser.parse_args(argv)

    resolved_run_dirs: Tuple[Path, ...] = tuple(args.run_dirs or [])
    if args.use_multi_run and not resolved_run_dirs and args.run_dir is None:
        available_runs = _list_runs(args.project_id, args.output_root)
        if int(args.recent_runs) <= 0:
            resolved_run_dirs = tuple(available_runs)
        else:
            resolved_run_dirs = tuple(available_runs[: max(1, int(args.recent_runs))])

    config = OptimizationConfig(
        project_id=args.project_id,
        run_dir=args.run_dir,
        run_dirs=resolved_run_dirs,
        output_root=args.output_root,
        iterations=args.iterations,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
        regularization_lambda=args.regularization_lambda,
        patience=args.patience,
        min_improvement=args.min_improvement,
        warmup_random=args.warmup_random,
        candidate_pool_size=args.candidate_pool_size,
        search_method=args.search_method,
        top_k=args.top_k,
        rsrp_objective_weight=args.rsrp_objective_weight,
        rsrq_objective_weight=args.rsrq_objective_weight,
        sinr_objective_weight=args.sinr_objective_weight,
        sinr_guard_tolerance=args.sinr_guard_tolerance,
        sinr_guard_penalty=args.sinr_guard_penalty,
        rsrp_guard_tolerance=args.rsrp_guard_tolerance,
        rsrp_guard_penalty=args.rsrp_guard_penalty,
        stage1_iterations=args.stage1_iterations,
        stage2_iterations=args.stage2_iterations,
        target_finetune_iterations=args.target_finetune_iterations,
        target_drift_fraction=args.target_drift_fraction,
        recent_runs=args.recent_runs,
        use_multi_run=args.use_multi_run,
        per_run_rsrp_guard_tolerance=args.per_run_rsrp_guard_tolerance,
        per_run_sinr_guard_tolerance=args.per_run_sinr_guard_tolerance,
        run_stability_penalty=args.run_stability_penalty,
    )
    optimizer_dir = run_geo_weight_optimizer(config)
    print(f"[OPT] Artifacts saved under {optimizer_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
