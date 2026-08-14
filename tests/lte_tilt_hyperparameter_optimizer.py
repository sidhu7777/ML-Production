from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from tests.baseline.lte_rf_debug_lab import DEFAULT_PROJECT_ID, DEFAULT_REGION, _write_json
from tests import lte_tilt_recommendation_test as tilt_test


OUTPUT_ROOT = Path("tests/output")

TUNABLE_DEFAULTS: Dict[str, float] = {
    "HIGH_NLOS_SHARE_GATE": float(tilt_test.HIGH_NLOS_SHARE_GATE),
    "HIGH_LOS_BLOCKED_RATIO_GATE": float(tilt_test.HIGH_LOS_BLOCKED_RATIO_GATE),
    "HIGH_BUILDING_AREA_RATIO_GATE": float(tilt_test.HIGH_BUILDING_AREA_RATIO_GATE),
    "MEDIUM_NLOS_SHARE_GATE": float(tilt_test.MEDIUM_NLOS_SHARE_GATE),
    "MEDIUM_LOS_BLOCKED_RATIO_GATE": float(tilt_test.MEDIUM_LOS_BLOCKED_RATIO_GATE),
    "MIN_PEAK_SHARE_FOR_AZIMUTH": float(tilt_test.MIN_PEAK_SHARE_FOR_AZIMUTH),
    "RELAXED_AZIMUTH_BAD_SAMPLE_COUNT": float(tilt_test.RELAXED_AZIMUTH_BAD_SAMPLE_COUNT),
    "RELAXED_AZIMUTH_PEAK_SHARE": float(tilt_test.RELAXED_AZIMUTH_PEAK_SHARE),
    "RELAXED_AZIMUTH_MAX_SPREAD_DEG": float(tilt_test.RELAXED_AZIMUTH_MAX_SPREAD_DEG),
    "MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH": float(tilt_test.MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH),
    "STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH": float(tilt_test.STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH),
    "MIN_CANDIDATE_SCORE_GAP": float(tilt_test.MIN_CANDIDATE_SCORE_GAP),
    "SITE_SYMMETRY_PENALTY_BAD_SAMPLE_RATIO": float(tilt_test.SITE_SYMMETRY_PENALTY_BAD_SAMPLE_RATIO),
    "COVERAGE_ETILT_ACTION_SCORE": float(tilt_test.COVERAGE_ETILT_ACTION_SCORE),
    "OVERLAP_ETILT_ACTION_SCORE": float(tilt_test.OVERLAP_ETILT_ACTION_SCORE),
    "AZIMUTH_ACTION_SCORE": float(tilt_test.AZIMUTH_ACTION_SCORE),
    "TX_POWER_ACTION_SCORE": float(tilt_test.TX_POWER_ACTION_SCORE),
}

SEARCH_SPACE: Dict[str, Tuple[float, float]] = {
    "HIGH_NLOS_SHARE_GATE": (0.45, 0.85),
    "HIGH_LOS_BLOCKED_RATIO_GATE": (0.12, 0.35),
    "HIGH_BUILDING_AREA_RATIO_GATE": (0.08, 0.30),
    "MEDIUM_NLOS_SHARE_GATE": (0.35, 0.75),
    "MEDIUM_LOS_BLOCKED_RATIO_GATE": (0.08, 0.25),
    "MIN_PEAK_SHARE_FOR_AZIMUTH": (0.24, 0.55),
    "RELAXED_AZIMUTH_BAD_SAMPLE_COUNT": (80.0, 320.0),
    "RELAXED_AZIMUTH_PEAK_SHARE": (0.30, 0.62),
    "RELAXED_AZIMUTH_MAX_SPREAD_DEG": (22.0, 45.0),
    "MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH": (1.0, 2.2),
    "STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH": (1.1, 3.0),
    "MIN_CANDIDATE_SCORE_GAP": (0.5, 8.0),
    "SITE_SYMMETRY_PENALTY_BAD_SAMPLE_RATIO": (0.05, 0.30),
    "COVERAGE_ETILT_ACTION_SCORE": (42.0, 72.0),
    "OVERLAP_ETILT_ACTION_SCORE": (42.0, 72.0),
    "AZIMUTH_ACTION_SCORE": (48.0, 84.0),
    "TX_POWER_ACTION_SCORE": (38.0, 68.0),
}


@dataclass
class TiltHyperTuneConfig:
    project_id: int = DEFAULT_PROJECT_ID
    region: str = DEFAULT_REGION
    operator: str | None = None
    rsrp_threshold: float = -105.0
    rsrq_threshold: float = -15.0
    sinr_threshold: float = 0.0
    output_root: Path = OUTPUT_ROOT
    iterations: int = 40
    warmup_random: int = 10
    candidate_pool_size: int = 192
    search_method: str = "bayes"
    patience: int = 10
    min_improvement: float = 0.5
    seed: int = 42
    target_actionable: int = 24
    min_actionable: int = 10
    max_actionable: int = 40
    target_azimuth_share: float = 0.25
    max_etilt_share: float = 0.80


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parameter_names() -> List[str]:
    return list(SEARCH_SPACE.keys())


def _params_to_vector(params: Dict[str, float]) -> np.ndarray:
    return np.array([float(params[name]) for name in _parameter_names()], dtype=float)


def _vector_to_params(vector: np.ndarray) -> Dict[str, float]:
    params = dict(TUNABLE_DEFAULTS)
    for idx, name in enumerate(_parameter_names()):
        low, high = SEARCH_SPACE[name]
        params[name] = float(np.clip(vector[idx], low, high))
    return _normalize_params(params)


def _sample_params(rng: np.random.Generator) -> Dict[str, float]:
    params = {}
    for name, (low, high) in SEARCH_SPACE.items():
        if name == "RELAXED_AZIMUTH_BAD_SAMPLE_COUNT":
            params[name] = float(rng.integers(int(low), int(high) + 1))
        else:
            params[name] = float(rng.uniform(low, high))
    return _normalize_params(params)


def _normalize_params(params: Dict[str, float]) -> Dict[str, float]:
    out = dict(params)
    for name, (low, high) in SEARCH_SPACE.items():
        out[name] = float(np.clip(out[name], low, high))

    out["MEDIUM_NLOS_SHARE_GATE"] = min(out["MEDIUM_NLOS_SHARE_GATE"], out["HIGH_NLOS_SHARE_GATE"] - 0.01)
    out["MEDIUM_LOS_BLOCKED_RATIO_GATE"] = min(
        out["MEDIUM_LOS_BLOCKED_RATIO_GATE"],
        out["HIGH_LOS_BLOCKED_RATIO_GATE"] - 0.01,
    )
    out["RELAXED_AZIMUTH_PEAK_SHARE"] = max(
        out["RELAXED_AZIMUTH_PEAK_SHARE"],
        out["MIN_PEAK_SHARE_FOR_AZIMUTH"] + 0.02,
    )
    out["STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH"] = max(
        out["STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH"],
        out["MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH"] + 0.05,
    )
    out["RELAXED_AZIMUTH_BAD_SAMPLE_COUNT"] = float(int(round(out["RELAXED_AZIMUTH_BAD_SAMPLE_COUNT"])))
    return out


@contextlib.contextmanager
def _patched_tilt_params(params: Dict[str, float]) -> Iterator[None]:
    old_values = {name: getattr(tilt_test, name) for name in params}
    try:
        for name, value in params.items():
            setattr(tilt_test, name, value)
        yield
    finally:
        for name, value in old_values.items():
            setattr(tilt_test, name, value)


def _positive_forecast_gain(forecast_df: pd.DataFrame) -> Tuple[float, float]:
    if forecast_df.empty or "Improvement %" not in forecast_df.columns:
        return 0.0, 0.0
    vals = pd.to_numeric(forecast_df["Improvement %"], errors="coerce").fillna(0.0)
    pos = vals[vals > 0]
    if pos.empty:
        return 0.0, 0.0
    return float(pos.sum()), float(pos.mean())


def _build_objective_score(
    metrics: Dict[str, float],
    config: TiltHyperTuneConfig,
) -> float:
    actionable = float(metrics["actionable_count"])
    azimuth_changes = float(metrics["azimuth_changes"])
    etilt_changes = float(metrics["etilt_changes"])
    tx_changes = float(metrics["tx_changes"])
    high_conf = float(metrics["high_confidence_actions"])
    medium_conf = float(metrics["medium_confidence_actions"])
    blocked = float(metrics["blocked_rows"])
    improvement_sum = float(metrics["forecast_positive_sum"])
    improvement_mean = float(metrics["forecast_positive_mean"])

    total_changes = max(actionable, 1.0)
    azimuth_share = azimuth_changes / total_changes
    etilt_share = etilt_changes / total_changes

    score = 0.0
    score += 45.0 * np.exp(-abs(actionable - float(config.target_actionable)) / max(6.0, float(config.target_actionable) * 0.35))
    if actionable < float(config.min_actionable):
        score -= (float(config.min_actionable) - actionable) * 3.5
    if actionable > float(config.max_actionable):
        score -= (actionable - float(config.max_actionable)) * 2.5

    score += min(20.0, azimuth_changes * 4.0)
    score += min(8.0, tx_changes * 2.5)
    score += max(0.0, 15.0 - abs(azimuth_share - float(config.target_azimuth_share)) * 60.0)
    if etilt_share > float(config.max_etilt_share):
        score -= (etilt_share - float(config.max_etilt_share)) * 35.0
    if azimuth_changes <= 0:
        score -= 20.0
    if etilt_changes <= 0:
        score -= 8.0

    score += min(25.0, improvement_sum * 0.15)
    score += min(10.0, improvement_mean * 0.40)
    score += min(12.0, high_conf * 3.0 + medium_conf * 0.5)
    score += min(6.0, blocked * 0.15)
    return float(score)


def _prepare_context(config: TiltHyperTuneConfig) -> Dict[str, object]:
    tilt_test.TILT_SRC.RSRP_THRESH = float(config.rsrp_threshold)
    tilt_test.TILT_SRC.RSRQ_THRESH = float(config.rsrq_threshold)
    tilt_test.TILT_SRC.SINR_THRESH = float(config.sinr_threshold)

    log_df = tilt_test._fetch_baseline_log_df(config.project_id, config.region, config.operator)
    antenna_df = tilt_test._fetch_antenna_df(config.project_id, config.region, config.operator)
    log_df = tilt_test._enrich_log_with_antenna_context(log_df, antenna_df)
    geo_df = tilt_test._fetch_geo_df(config.project_id, config.region, config.operator, antenna_df)

    bad_samples_df, summary_df = tilt_test.TILT_SRC.filter_bad_samples(log_df.copy(), tilt_test.TILT_SRC.ALLOWED_TECHS)
    swap_dict = tilt_test.TILT_SRC.detect_swap_sector(log_df.copy(), antenna_df.copy())
    bearing_summary = tilt_test._compute_dominant_bearing_summary(log_df, antenna_df)
    bad_geo_df = tilt_test._attach_geo_to_bad_samples(bad_samples_df, geo_df)
    geo_cell_summary = tilt_test._aggregate_bad_geo_context(bad_geo_df)

    return {
        "log_df": log_df,
        "antenna_df": antenna_df,
        "geo_df": geo_df,
        "bad_samples_df": bad_samples_df,
        "summary_df": summary_df,
        "swap_dict": swap_dict,
        "bearing_summary": bearing_summary,
        "bad_geo_df": bad_geo_df,
        "geo_cell_summary": geo_cell_summary,
    }


def _evaluate_candidate(
    context: Dict[str, object],
    params: Dict[str, float],
    config: TiltHyperTuneConfig,
) -> Dict[str, object]:
    with _patched_tilt_params(params):
        recommendations_all_df = tilt_test._build_geo_aware_recommendations(
            context["summary_df"],
            context["antenna_df"],
            context["swap_dict"],
            context["geo_cell_summary"],
            context["bearing_summary"],
        )
        recommendations_all_df, recommendations_df = tilt_test._prepare_recommendation_exports(recommendations_all_df)
        forecast_df = tilt_test.TILT_SRC.build_forecast(context["summary_df"], recommendations_all_df)

    changed_mask = tilt_test._changed_recommendation_mask(recommendations_df) if not recommendations_df.empty else pd.Series(dtype=bool)
    changed_rows = recommendations_df.loc[changed_mask].copy() if not recommendations_df.empty else pd.DataFrame()
    param_counts = changed_rows["Parameter"].astype(str).value_counts().to_dict() if not changed_rows.empty else {}
    conf_counts = (
        changed_rows["Recommendation Confidence"].astype(str).str.strip().str.lower().value_counts().to_dict()
        if not changed_rows.empty and "Recommendation Confidence" in changed_rows.columns
        else {}
    )
    status_counts = (
        recommendations_all_df["Recommendation Status"].astype(str).str.strip().str.lower().value_counts().to_dict()
        if not recommendations_all_df.empty
        else {}
    )
    forecast_sum, forecast_mean = _positive_forecast_gain(forecast_df)

    metrics = {
        "actionable_count": int(len(changed_rows)),
        "exported_count": int(len(recommendations_df)),
        "all_rows": int(len(recommendations_all_df)),
        "azimuth_changes": int(param_counts.get("Azimuth", 0)),
        "etilt_changes": int(param_counts.get("ETilt", 0)),
        "tx_changes": int(param_counts.get("TX Power", 0)),
        "high_confidence_actions": int(conf_counts.get("high", 0)),
        "medium_confidence_actions": int(conf_counts.get("medium", 0)),
        "blocked_rows": int(status_counts.get("blocked_by_blockage", 0)) + int(status_counts.get("hold_swap", 0)),
        "forecast_positive_sum": round(forecast_sum, 4),
        "forecast_positive_mean": round(forecast_mean, 4),
    }
    objective_score = _build_objective_score(metrics, config)
    return {
        "score": objective_score,
        "metrics": metrics,
        "recommendations_all_df": recommendations_all_df,
        "recommendations_df": recommendations_df,
        "forecast_df": forecast_df,
    }


def _propose_bayesian_candidate(
    tried_vectors: List[np.ndarray],
    tried_scores: List[float],
    rng: np.random.Generator,
    candidate_pool_size: int,
) -> Dict[str, float]:
    if len(tried_vectors) < 5:
        return _sample_params(rng)

    X = np.vstack(tried_vectors)
    y = np.array(tried_scores, dtype=float)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
        normalize_y=True,
        random_state=0,
    )
    gp.fit(X, y)

    pool = np.vstack([_params_to_vector(_sample_params(rng)) for _ in range(candidate_pool_size)])
    mean_pred, std_pred = gp.predict(pool, return_std=True)
    acquisition = mean_pred + (0.35 * std_pred)
    return _vector_to_params(pool[int(np.argmax(acquisition))])


def run_tilt_hyperparameter_optimizer(config: TiltHyperTuneConfig) -> Path:
    start = time.perf_counter()
    context = _prepare_context(config)
    run_dir = _ensure_dir(config.output_root / f"project_{config.project_id}" / f"tilt_tuning_{_timestamp()}")

    rng = np.random.default_rng(config.seed)
    baseline_params = dict(TUNABLE_DEFAULTS)
    baseline_eval = _evaluate_candidate(context, baseline_params, config)
    best_params = dict(baseline_params)
    best_eval = baseline_eval
    best_score = float(baseline_eval["score"])

    leaderboard: List[Dict[str, object]] = []
    tried_vectors: List[np.ndarray] = [_params_to_vector(baseline_params)]
    tried_scores: List[float] = [best_score]
    no_improvement_rounds = 0

    for idx in range(config.iterations):
        if idx < config.warmup_random or config.search_method == "random":
            params = _sample_params(rng)
            search_label = "random"
        else:
            params = _propose_bayesian_candidate(tried_vectors, tried_scores, rng, config.candidate_pool_size)
            search_label = "bayes"

        eval_result = _evaluate_candidate(context, params, config)
        score = float(eval_result["score"])
        metrics = dict(eval_result["metrics"])
        row: Dict[str, object] = {
            "candidate": idx + 1,
            "search_method": search_label,
            "objective_score": round(score, 6),
            **metrics,
        }
        row.update({key: params[key] for key in _parameter_names()})
        leaderboard.append(row)

        tried_vectors.append(_params_to_vector(params))
        tried_scores.append(score)

        if score > (best_score + float(config.min_improvement)):
            best_score = score
            best_params = dict(params)
            best_eval = eval_result
            no_improvement_rounds = 0
        else:
            no_improvement_rounds += 1
            if no_improvement_rounds >= config.patience:
                print(f"[TILT_TUNE] Early stop at candidate {idx + 1} due to no improvement.")
                break

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("objective_score", ascending=False).reset_index(drop=True)
    leaderboard_df.to_csv(run_dir / "leaderboard.csv", index=False)

    best_eval["recommendations_all_df"].to_csv(run_dir / "best_recommendations_all.csv", index=False)
    best_eval["recommendations_df"].to_csv(run_dir / "best_recommendations.csv", index=False)
    best_eval["forecast_df"].to_csv(run_dir / "best_forecast.csv", index=False)
    pd.DataFrame([{"parameter": key, "value": value} for key, value in best_params.items()]).to_csv(
        run_dir / "best_parameters.csv",
        index=False,
    )

    summary_payload = {
        "run_type": "tilt_hyperparameter_optimizer",
        "project_id": int(config.project_id),
        "region": config.region,
        "operator": config.operator,
        "thresholds": {
            "rsrp": float(config.rsrp_threshold),
            "rsrq": float(config.rsrq_threshold),
            "sinr": float(config.sinr_threshold),
        },
        "search": {
            "iterations_requested": int(config.iterations),
            "iterations_completed": int(len(leaderboard_df)),
            "warmup_random": int(config.warmup_random),
            "candidate_pool_size": int(config.candidate_pool_size),
            "search_method": config.search_method,
            "seed": int(config.seed),
            "patience": int(config.patience),
            "min_improvement": float(config.min_improvement),
        },
        "objective_profile": {
            "target_actionable": int(config.target_actionable),
            "min_actionable": int(config.min_actionable),
            "max_actionable": int(config.max_actionable),
            "target_azimuth_share": float(config.target_azimuth_share),
            "max_etilt_share": float(config.max_etilt_share),
            "note": (
                "This objective favors balanced actionable output, azimuth presence, forecast uplift, "
                "and fewer ETilt-only explosions while keeping safety rules outside the tuning loop."
            ),
        },
        "input_counts": {
            "baseline_rows": int(len(context["log_df"])),
            "antenna_rows": int(len(context["antenna_df"])),
            "geo_rows": int(len(context["geo_df"])),
            "bad_samples": int(len(context["bad_samples_df"])),
            "bad_cells": int(context["summary_df"]["Cell ID"].nunique()) if not context["summary_df"].empty else 0,
        },
        "baseline_metrics": {
            "score": round(float(baseline_eval["score"]), 6),
            **baseline_eval["metrics"],
        },
        "best_metrics": {
            "score": round(float(best_eval["score"]), 6),
            **best_eval["metrics"],
        },
        "baseline_parameters": baseline_params,
        "best_parameters": best_params,
        "artifacts": {
            "leaderboard": str(run_dir / "leaderboard.csv"),
            "best_parameters_csv": str(run_dir / "best_parameters.csv"),
            "best_recommendations_all": str(run_dir / "best_recommendations_all.csv"),
            "best_recommendations": str(run_dir / "best_recommendations.csv"),
            "best_forecast": str(run_dir / "best_forecast.csv"),
        },
        "total_runtime_sec": round(float(time.perf_counter() - start), 4),
    }
    _write_json(run_dir / "summary.json", summary_payload)
    print(f"[TILT_TUNE][DONE] run_dir={run_dir} best_score={summary_payload['best_metrics']['score']}")
    return run_dir


def _parse_args() -> TiltHyperTuneConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--rsrp", type=float, default=-105.0)
    parser.add_argument("--rsrq", type=float, default=-15.0)
    parser.add_argument("--sinr", type=float, default=0.0)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmup-random", type=int, default=10)
    parser.add_argument("--candidate-pool-size", type=int, default=192)
    parser.add_argument("--search-method", type=str, choices=["random", "bayes"], default="bayes")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-improvement", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-actionable", type=int, default=24)
    parser.add_argument("--min-actionable", type=int, default=10)
    parser.add_argument("--max-actionable", type=int, default=40)
    parser.add_argument("--target-azimuth-share", type=float, default=0.25)
    parser.add_argument("--max-etilt-share", type=float, default=0.80)
    args = parser.parse_args()
    return TiltHyperTuneConfig(
        project_id=args.project_id,
        region=args.region,
        operator=args.operator,
        rsrp_threshold=args.rsrp,
        rsrq_threshold=args.rsrq,
        sinr_threshold=args.sinr,
        output_root=args.output_root,
        iterations=args.iterations,
        warmup_random=args.warmup_random,
        candidate_pool_size=args.candidate_pool_size,
        search_method=args.search_method,
        patience=args.patience,
        min_improvement=args.min_improvement,
        seed=args.seed,
        target_actionable=args.target_actionable,
        min_actionable=args.min_actionable,
        max_actionable=args.max_actionable,
        target_azimuth_share=args.target_azimuth_share,
        max_etilt_share=args.max_etilt_share,
    )


if __name__ == "__main__":
    run_dir = run_tilt_hyperparameter_optimizer(_parse_args())
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
