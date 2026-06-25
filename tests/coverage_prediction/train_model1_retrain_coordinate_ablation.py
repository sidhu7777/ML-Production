"""
Model 1 coordinate ablation retraining.

This script keeps the existing Model 1 trainer and artifacts untouched. It
trains two variants into models/model1_retrain:
    - with_coordinates: current Model 1 feature set
    - remove_coordinates: excludes grid_centroid_lat/grid_centroid_lon
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tests.coverage_prediction import train_model1_coverage_xgboost as base


COORDINATE_FEATURES = ("grid_centroid_lat", "grid_centroid_lon")
RETRAIN_ROOT = Path("models") / "model1_retrain"
DATASET_CSV = Path("data") / "model1_coverage_training.csv"
SUMMARY_JSON = RETRAIN_ROOT / "coordinate_ablation_summary.json"
SUMMARY_CSV = RETRAIN_ROOT / "coordinate_ablation_metrics.csv"


def setup_logging() -> logging.Logger:
    RETRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("model1_coordinate_ablation")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(RETRAIN_ROOT / "training.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


log = setup_logging()


def variant_features(remove_coordinates: bool) -> tuple[list[str], list[str]]:
    numeric = list(base.NUMERIC_FEATURES)
    if remove_coordinates:
        numeric = [feature for feature in numeric if feature not in COORDINATE_FEATURES]
    return numeric, list(base.CATEGORICAL_FEATURES)


@contextmanager
def patched_base_context(
    model_root: Path,
    numeric_features: list[str],
    categorical_features: list[str],
    optuna_trials: int,
    optuna_timeout: int,
) -> Iterator[None]:
    original = {
        "MODEL_ROOT": base.MODEL_ROOT,
        "LOG_PATH": base.LOG_PATH,
        "NUMERIC_FEATURES": base.NUMERIC_FEATURES,
        "CATEGORICAL_FEATURES": base.CATEGORICAL_FEATURES,
        "ALL_FEATURES": base.ALL_FEATURES,
        "OPTUNA_TRIALS": base.OPTUNA_TRIALS,
        "OPTUNA_TIMEOUT": base.OPTUNA_TIMEOUT,
        "log": base.log,
    }
    try:
        base.MODEL_ROOT = model_root
        base.LOG_PATH = model_root / "training.log"
        base.NUMERIC_FEATURES = numeric_features
        base.CATEGORICAL_FEATURES = categorical_features
        base.ALL_FEATURES = numeric_features + categorical_features
        base.OPTUNA_TRIALS = int(optuna_trials)
        base.OPTUNA_TIMEOUT = int(optuna_timeout)
        base.log = log
        yield
    finally:
        base.MODEL_ROOT = original["MODEL_ROOT"]
        base.LOG_PATH = original["LOG_PATH"]
        base.NUMERIC_FEATURES = original["NUMERIC_FEATURES"]
        base.CATEGORICAL_FEATURES = original["CATEGORICAL_FEATURES"]
        base.ALL_FEATURES = original["ALL_FEATURES"]
        base.OPTUNA_TRIALS = original["OPTUNA_TRIALS"]
        base.OPTUNA_TIMEOUT = original["OPTUNA_TIMEOUT"]
        base.log = original["log"]


def load_dataset(rebuild_dataset: bool) -> pd.DataFrame:
    if rebuild_dataset or not DATASET_CSV.exists():
        log.info("Building Model 1 dataset via existing trainer")
        return base.build_dataset()

    df = pd.read_csv(DATASET_CSV)
    required = set(base.NUMERIC_FEATURES + base.CATEGORICAL_FEATURES + base.TARGETS + ["time_bucket"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"Existing Model 1 dataset is missing required columns: {missing}")

    for col in base.NUMERIC_FEATURES + base.TARGETS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in base.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    df["time_bucket"] = df["time_bucket"].astype(str)
    log.info("Loaded existing Model 1 dataset rows=%d columns=%d", len(df), len(df.columns))
    return df


def coordinate_shap_summary(target_dir: Path) -> dict[str, object]:
    shap_path = target_dir / "shap_importance.csv"
    if not shap_path.exists():
        return {
            "available": False,
            "coordinate_total_mean_abs_shap": None,
            "coordinate_top_rank": None,
            "coordinate_features": [],
        }

    shap_df = pd.read_csv(shap_path)
    coord_rows = shap_df[
        shap_df["feature"].astype(str).str.contains("grid_centroid_lat|grid_centroid_lon", regex=True, na=False)
    ].copy()
    if coord_rows.empty:
        return {
            "available": True,
            "coordinate_total_mean_abs_shap": 0.0,
            "coordinate_top_rank": None,
            "coordinate_features": [],
        }

    shap_df = shap_df.reset_index(drop=True)
    coord_rows["rank"] = coord_rows.index + 1
    return {
        "available": True,
        "coordinate_total_mean_abs_shap": float(coord_rows["mean_abs_shap"].sum()),
        "coordinate_top_rank": int(coord_rows["rank"].min()),
        "coordinate_features": coord_rows[["feature", "mean_abs_shap", "rank"]].to_dict(orient="records"),
    }


def normalized_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "train": metrics.get("train") or metrics.get("TRAIN") or {},
        "valid": metrics.get("valid") or metrics.get("VALID") or {},
        "test": metrics.get("test") or metrics.get("TEST") or {},
        "generalisation_gap_pct": metrics.get("generalisation_gap_pct"),
        "residual_mean": metrics.get("residual_mean"),
        "residual_std": metrics.get("residual_std"),
    }


def read_target_result(variant_root: Path, target: str) -> dict[str, object]:
    target_dir = variant_root / target
    metrics_path = target_dir / "metrics.json"
    evolution_path = target_dir / "future_evolution_summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    evolution = json.loads(evolution_path.read_text(encoding="utf-8")) if evolution_path.exists() else {}
    return {
        "metrics": normalized_metrics(metrics),
        "future_evolution": {
            "mean_abs_gap": evolution.get("mean_abs_gap"),
            "median_abs_gap": evolution.get("median_abs_gap"),
            "mean_residual": evolution.get("mean_residual"),
            "mean_grid_delta_alignment_gap": (evolution.get("grid_alignment") or {}).get("mean_grid_delta_alignment_gap"),
            "mean_group_delta_alignment_gap": (evolution.get("clutter_alignment") or {}).get("mean_group_delta_alignment_gap"),
        },
        "coordinate_shap": coordinate_shap_summary(target_dir),
    }


def read_variant_result(variant_name: str, remove_coordinates: bool) -> dict[str, object]:
    numeric_features, categorical_features = variant_features(remove_coordinates)
    variant_root = RETRAIN_ROOT / variant_name
    return {
        "variant": variant_name,
        "remove_coordinates": remove_coordinates,
        "removed_features": list(COORDINATE_FEATURES) if remove_coordinates else [],
        "feature_count": len(numeric_features) + len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "targets": {target: read_target_result(variant_root, target) for target in base.TARGETS},
    }


def train_variant(
    variant_name: str,
    remove_coordinates: bool,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    optuna_trials: int,
    optuna_timeout: int,
) -> dict[str, object]:
    numeric_features, categorical_features = variant_features(remove_coordinates)
    variant_root = RETRAIN_ROOT / variant_name
    variant_root.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("VARIANT: %s | features=%d | removed=%s", variant_name, len(numeric_features) + len(categorical_features), list(COORDINATE_FEATURES) if remove_coordinates else [])
    log.info("=" * 70)

    result: dict[str, object] = {
        "variant": variant_name,
        "remove_coordinates": remove_coordinates,
        "removed_features": list(COORDINATE_FEATURES) if remove_coordinates else [],
        "feature_count": len(numeric_features) + len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "targets": {},
    }

    with patched_base_context(variant_root, numeric_features, categorical_features, optuna_trials, optuna_timeout):
        for target in base.TARGETS:
            log.info("TARGET: %s", target)
            base.train_target(train_df, valid_df, test_df, target)
            target_dir = variant_root / target
            result["targets"][target] = read_target_result(variant_root, target)

    return result


def metric_delta(with_value: object, without_value: object) -> float | None:
    try:
        return float(without_value) - float(with_value)
    except (TypeError, ValueError):
        return None


def build_comparison(with_coords: dict[str, object], without_coords: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    comparison: dict[str, object] = {}
    with_targets = with_coords.get("targets", {})
    without_targets = without_coords.get("targets", {})

    for target in base.TARGETS:
        with_metrics = (with_targets.get(target, {}) or {}).get("metrics", {})
        without_metrics = (without_targets.get(target, {}) or {}).get("metrics", {})
        target_summary = {}
        for split in ["train", "valid", "test"]:
            for metric in ["mae", "rmse", "r2"]:
                with_value = (with_metrics.get(split, {}) or {}).get(metric)
                without_value = (without_metrics.get(split, {}) or {}).get(metric)
                delta = metric_delta(with_value, without_value)
                rows.append(
                    {
                        "target": target,
                        "split": split,
                        "metric": metric,
                        "with_coordinates": with_value,
                        "remove_coordinates": without_value,
                        "delta_remove_minus_with": delta,
                    }
                )
                target_summary[f"{split}_{metric}_delta_remove_minus_with"] = delta

        with_shap = (with_targets.get(target, {}) or {}).get("coordinate_shap", {})
        without_shap = (without_targets.get(target, {}) or {}).get("coordinate_shap", {})
        target_summary["with_coordinates_coordinate_shap_total"] = with_shap.get("coordinate_total_mean_abs_shap")
        target_summary["with_coordinates_coordinate_top_rank"] = with_shap.get("coordinate_top_rank")
        target_summary["remove_coordinates_coordinate_shap_total"] = without_shap.get("coordinate_total_mean_abs_shap")
        comparison[target] = target_summary

    return comparison, pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain Model 1 with/without centroid coordinates for ablation.")
    parser.add_argument("--trials", type=int, default=base.OPTUNA_TRIALS, help="Optuna trials per target and variant")
    parser.add_argument("--timeout", type=int, default=base.OPTUNA_TIMEOUT, help="Optuna timeout seconds per target and variant")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild data/model1_coverage_training.csv before retraining")
    parser.add_argument("--only-target", choices=base.TARGETS, help="Train only one target for a faster focused check")
    parser.add_argument("--summarize-only", action="store_true", help="Regenerate comparison files from existing model1_retrain artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_targets = list(base.TARGETS)
    if args.only_target:
        base.TARGETS = [args.only_target]

    try:
        log.info("Model 1 coordinate ablation started at %s", datetime.utcnow().isoformat() + "Z")
        log.info("Output root: %s", RETRAIN_ROOT.resolve())
        if args.summarize_only:
            with_coords = read_variant_result("with_coordinates", False)
            without_coords = read_variant_result("remove_coordinates", True)
        else:
            df = load_dataset(args.rebuild_dataset)
            train_df, valid_df, test_df = base.temporal_split(df)

            with_coords = train_variant(
                "with_coordinates",
                False,
                train_df,
                valid_df,
                test_df,
                args.trials,
                args.timeout,
            )
            without_coords = train_variant(
                "remove_coordinates",
                True,
                train_df,
                valid_df,
                test_df,
                args.trials,
                args.timeout,
            )
        comparison, comparison_df = build_comparison(with_coords, without_coords)
        comparison_df.to_csv(SUMMARY_CSV, index=False)

        summary = {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "dataset_csv": str(DATASET_CSV),
            "output_root": str(RETRAIN_ROOT),
            "optuna_trials": int(args.trials),
            "optuna_timeout": int(args.timeout),
            "coordinate_features": list(COORDINATE_FEATURES),
            "variants": {
                "with_coordinates": with_coords,
                "remove_coordinates": without_coords,
            },
            "comparison": comparison,
            "comparison_csv": str(SUMMARY_CSV),
            "interpretation_rule": "For MAE/RMSE, positive delta means removing coordinates is worse. For R2, negative delta means removing coordinates is worse.",
        }
        base.save_json(summary, SUMMARY_JSON)
        log.info("Saved ablation summary to %s", SUMMARY_JSON)
        log.info("Saved ablation metric table to %s", SUMMARY_CSV)
    finally:
        base.TARGETS = original_targets


if __name__ == "__main__":
    main()
