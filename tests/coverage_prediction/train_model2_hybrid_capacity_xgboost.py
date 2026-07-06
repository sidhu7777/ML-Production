"""
Train Model 2 capacity models on the Model 1 hybrid-target coverage surface.

This is a test-side experiment. It writes under
models/model2_hybrid_target_experiment and does not overwrite the production
or baseline Model 2 artifacts under models/model2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tests.coverage_prediction.build_model3_hybrid_load_balancing_dataset import (
    HYBRID_MODEL2_CSV,
    build_hybrid_model2_dataset,
)


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MODEL_ROOT = ML_ROOT / "models" / "model2_hybrid_target_experiment"
FULL_PREDICTIONS_CSV = MODEL_ROOT / "model2_hybrid_full_predictions.csv"
SUMMARY_JSON = MODEL_ROOT / "model2_hybrid_training_summary.json"

TARGET_CONFIG = {
    "model2a_demand": {
        "target": "demand_index",
        "model_file": "model2a_demand_xgb.pkl",
    },
    "model2b_users": {
        "target": "active_users_est",
        "model_file": "model2b_users_xgb.pkl",
    },
    "model2c_traffic": {
        "target": "traffic_demand_est",
        "model_file": "model2c_traffic_xgb.pkl",
    },
}
TARGETS = [cfg["target"] for cfg in TARGET_CONFIG.values()]

NUMERIC_FEATURES = [
    "bucket_seq",
    "building_count",
    "building_area_ratio",
    "road_density",
    "mall_presence",
    "metro_presence",
    "park_open_area",
    "open_area_ratio",
    "green_ratio",
    "water_ratio",
    "rsrp_mean",
    "rsrq_mean",
    "sinr_mean",
    "corrected_rsrp_mean",
    "corrected_rsrq_mean",
    "corrected_sinr_mean",
    "cqi_mean",
    "dl_tpt_mean",
    "ul_tpt_mean",
    "sample_count",
    "bandwidth_mhz_est",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "carrier_count",
    "prb_pressure_est",
    "prb_outlier_flag",
    "growth_rate",
    "geo_demand_score",
    "kpi_demand_score",
    "development_pressure_score",
    "growth_zone_score",
    "clutter_transition_flag",
    "clutter_upgrade_score",
    "building_growth_ratio",
    "road_growth_ratio",
    "activity_anchor_score",
    "capacity_context_score",
    "capacity_gap_score",
]
CATEGORICAL_FEATURES = ["time_bucket", "clutter_class", "dominant_band_class"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def load_dataset() -> pd.DataFrame:
    build_hybrid_model2_dataset()
    df = pd.read_csv(HYBRID_MODEL2_CSV)
    required = set(ALL_FEATURES + TARGETS + ["grid_id", "time_bucket"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"Hybrid Model 2 dataset is missing required columns: {missing}")

    for col in NUMERIC_FEATURES + TARGETS + ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time_bucket"] = df["time_bucket"].astype(str)
    df["clutter_class"] = df["clutter_class"].fillna("Unknown").astype(str)
    df["dominant_band_class"] = df["dominant_band_class"].fillna("UNKNOWN").astype(str)
    df = df.dropna(subset=TARGETS).sort_values(["grid_id", "bucket_seq"]).reset_index(drop=True)
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["time_bucket"].isin(["PART_1", "PART_2"])].copy()
    part3 = df[df["time_bucket"] == "PART_3"].sort_values(["grid_id", "grid_row", "grid_col"]).reset_index(drop=True)
    split_idx = int(len(part3) * 0.60)
    split_idx = max(1, min(split_idx, len(part3) - 1))
    valid_df = part3.iloc[:split_idx].copy()
    test_df = part3.iloc[split_idx:].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise RuntimeError(f"Temporal split failed: train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")
    return train_df, valid_df, test_df


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    pipe = Pipeline([("prep", build_preprocessor()), ("model", XGBRegressor(**params))])
    pipe.fit(X_train, y_train)
    return rmse(y_valid.to_numpy(dtype=float), pipe.predict(X_valid))


def tune_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    trials: int,
    timeout: int,
) -> dict[str, Any]:
    if trials <= 0:
        return {
            "n_estimators": 550,
            "max_depth": 5,
            "learning_rate": 0.035,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "reg_alpha": 0.001,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        }
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=trials, timeout=timeout)
    return dict(study.best_params)


def fit_pipeline(X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    model_params = {
        **params,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    pipe = Pipeline([("prep", build_preprocessor()), ("model", XGBRegressor(**model_params))])
    pipe.fit(X_train, y_train)
    return pipe


def train_all(trials: int, timeout: int) -> dict[str, Any]:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    train_df, valid_df, test_df = temporal_split(df)
    full_predictions = df[
        [
            col
            for col in [
                "grid_id",
                "time_bucket",
                "bucket_seq",
                "grid_row",
                "grid_col",
                "grid_centroid_lat",
                "grid_centroid_lon",
                "clutter_class",
                "dominant_band_class",
                "dominant_pci",
                "demand_index",
                "active_users_est",
                "traffic_demand_est",
                "estimated_prb_mean",
                "prb_pressure_est",
            ]
            if col in df.columns
        ]
    ].copy()

    summary: dict[str, Any] = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "source_dataset": str(HYBRID_MODEL2_CSV),
        "output_root": str(MODEL_ROOT),
        "rows": int(len(df)),
        "split_rows": {"train": int(len(train_df)), "valid": int(len(valid_df)), "test": int(len(test_df))},
        "optuna_trials": int(trials),
        "optuna_timeout": int(timeout),
        "targets": {},
    }

    for target_key, cfg in TARGET_CONFIG.items():
        target = cfg["target"]
        out_dir = MODEL_ROOT / target_key
        out_dir.mkdir(parents=True, exist_ok=True)
        X_train, y_train = train_df[ALL_FEATURES], train_df[target]
        X_valid, y_valid = valid_df[ALL_FEATURES], valid_df[target]
        X_test, y_test = test_df[ALL_FEATURES], test_df[target]

        params = tune_params(X_train, y_train, X_valid, y_valid, trials, timeout)
        pipe = fit_pipeline(X_train, y_train, params)
        joblib.dump(pipe, out_dir / cfg["model_file"])
        joblib.dump(pipe, MODEL_ROOT / cfg["model_file"])

        metrics = {
            "TRAIN": compute_metrics(y_train.to_numpy(dtype=float), pipe.predict(X_train)),
            "VALID": compute_metrics(y_valid.to_numpy(dtype=float), pipe.predict(X_valid)),
            "TEST": compute_metrics(y_test.to_numpy(dtype=float), pipe.predict(X_test)),
        }
        test_pred = test_df.copy()
        test_pred[f"{target}_pred"] = pipe.predict(X_test)
        test_pred.to_csv(out_dir / "test_predictions.csv", index=False)
        full_predictions[f"{target}_pred"] = pipe.predict(df[ALL_FEATURES])

        target_summary = {
            "target": target,
            "model_file": str(out_dir / cfg["model_file"]),
            "root_model_file": str(MODEL_ROOT / cfg["model_file"]),
            "best_params": params,
            "metrics": metrics,
        }
        save_json(target_summary, out_dir / "metadata.json")
        summary["targets"][target_key] = target_summary

    full_predictions.to_csv(FULL_PREDICTIONS_CSV, index=False)
    summary["full_predictions_csv"] = str(FULL_PREDICTIONS_CSV)
    save_json(summary, SUMMARY_JSON)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid Model 2 capacity models.")
    parser.add_argument("--trials", type=int, default=8, help="Optuna trials per target. Use 0 for fixed params.")
    parser.add_argument("--timeout", type=int, default=120, help="Optuna timeout seconds per target.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_all(args.trials, args.timeout)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
