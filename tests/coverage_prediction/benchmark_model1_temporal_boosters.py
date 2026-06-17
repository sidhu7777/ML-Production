"""
Benchmark temporal coverage boosters for Model 1.

This script compares XGBoost, LightGBM, and CatBoost on the exact same:
    - observed bucket targets
    - feature set
    - temporal split

Artifacts are saved under models/model1/benchmark_boosters.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.coverage_prediction.train_model1_coverage_xgboost import (
    ALL_FEATURES,
    MODEL_ROOT,
    RANDOM_SEED,
    TARGETS,
    build_dataset,
    build_preprocessor,
    evaluate,
    run_future_evolution_analysis,
    save_json,
    temporal_split,
)


BENCHMARK_ROOT = MODEL_ROOT / "benchmark_boosters"
BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)


def build_models() -> dict[str, object]:
    return {
        "xgboost": XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        ),
        "lightgbm": LGBMRegressor(
            objective="regression",
            random_state=RANDOM_SEED,
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbosity=-1,
        ),
        "catboost": CatBoostRegressor(
            loss_function="RMSE",
            random_seed=RANDOM_SEED,
            iterations=400,
            depth=5,
            learning_rate=0.05,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def fit_pipeline(model: object, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline([("prep", build_preprocessor()), ("model", model)])
    pipeline.fit(X_train, y_train)
    return pipeline


def benchmark_target(
    model_name: str,
    model: object,
    target: str,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> dict[str, object]:
    out_dir = BENCHMARK_ROOT / model_name / target
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = df_train[ALL_FEATURES], df_train[target]
    X_valid, y_valid = df_valid[ALL_FEATURES], df_valid[target]
    X_test, y_test = df_test[ALL_FEATURES], df_test[target]

    pipeline = fit_pipeline(model, X_train, y_train)
    metrics = evaluate(pipeline, X_train, y_train, X_valid, y_valid, X_test, y_test, target, out_dir)
    evolution = run_future_evolution_analysis(pipeline, df_train, df_test, target, out_dir)

    summary = {
        "model_name": model_name,
        "target": target,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "future_evolution": {
            "mean_abs_gap": evolution["mean_abs_gap"],
            "median_abs_gap": evolution["median_abs_gap"],
            "mean_residual": evolution["mean_residual"],
            "mean_group_delta_alignment_gap": evolution["clutter_alignment"]["mean_group_delta_alignment_gap"],
            "mean_grid_delta_alignment_gap": evolution["grid_alignment"]["mean_grid_delta_alignment_gap"],
        },
    }
    save_json(summary, out_dir / "benchmark_summary.json")
    return summary


def main() -> None:
    df = build_dataset()
    train_df, valid_df, test_df = temporal_split(df)
    models = build_models()

    all_results: dict[str, dict[str, object]] = {}
    for model_name, model in models.items():
        all_results[model_name] = {}
        for target in TARGETS:
            all_results[model_name][target] = benchmark_target(model_name, model, target, train_df, valid_df, test_df)

    save_json(all_results, BENCHMARK_ROOT / "all_benchmark_results.json")


if __name__ == "__main__":
    main()
