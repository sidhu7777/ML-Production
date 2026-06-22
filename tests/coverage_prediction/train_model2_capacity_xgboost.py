"""
Model 2 capacity / congestion / demand training pipeline.

Model 2 is a grid-level network capacity model, not a subscriber forecasting
model. It trains three separate XGBoost regressors from the saved
model2_capacity_training.csv dataset:
    - Model 2A: demand_index
    - Model 2B: active_users_est
    - Model 2C: traffic_demand_est
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

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

warnings.filterwarnings("ignore")


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ML_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ML_ROOT / "data"
MODEL2_DATASET_CSV = DATA_DIR / "model2_capacity_training.csv"
MODEL2_DATASET_SUMMARY_JSON = DATA_DIR / "model2_capacity_training.summary.json"

MODEL_ROOT = ML_ROOT / "models" / "model2"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = MODEL_ROOT / "training.log"

TARGET_CONFIG = {
    "model2a_demand": {
        "target": "demand_index",
        "model_file": "model2a_demand_xgb.pkl",
        "purpose": "Predict future demand pressure score.",
        "business_meaning": "Which grids are likely to require future capacity attention?",
    },
    "model2b_users": {
        "target": "active_users_est",
        "model_file": "model2b_users_xgb.pkl",
        "purpose": "Predict relative user concentration.",
        "business_meaning": "Which grids are expected to carry more users?",
        "note": "Estimated proxy, not a real subscriber count.",
    },
    "model2c_traffic": {
        "target": "traffic_demand_est",
        "model_file": "model2c_traffic_xgb.pkl",
        "purpose": "Predict relative traffic demand.",
        "business_meaning": "Which grids are expected to generate higher traffic load?",
        "note": "Traffic-demand proxy, not a real OSS traffic counter.",
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

DEFAULT_OPTUNA_TRIALS = 50
DEFAULT_OPTUNA_TIMEOUT = 600
MAX_GENERALISATION_GAP = 0.10


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model2_train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


log = setup_logging()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true)
    mask = denom > 1e-9
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom[mask])) * 100.0)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom <= 1e-9:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": mape(y_true, y_pred),
        "wape_pct": wape(y_true, y_pred),
    }


def load_dataset() -> pd.DataFrame:
    if not MODEL2_DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing Model 2 dataset: {MODEL2_DATASET_CSV}")

    df = pd.read_csv(MODEL2_DATASET_CSV)
    if "estimated_prb_mean" in df.columns:
        log.info("estimated_prb_mean is present but excluded from training by design.")

    required = set(ALL_FEATURES + TARGETS + ["grid_id", "time_bucket"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"Model 2 dataset is missing required columns: {missing}")

    for col in NUMERIC_FEATURES + TARGETS + ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time_bucket"] = df["time_bucket"].astype(str)
    df["clutter_class"] = df["clutter_class"].fillna("Unknown").astype(str)
    df = df.dropna(subset=TARGETS).copy()
    df = df.sort_values(["grid_id", "bucket_seq"]).reset_index(drop=True)

    log.info("Loaded Model 2 dataset rows=%d columns=%d", len(df), len(df.columns))
    return df


def temporal_split(df: pd.DataFrame, valid_fraction_of_part3: float = 0.60) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["time_bucket"].isin(["PART_1", "PART_2"])].copy()
    part3 = df[df["time_bucket"] == "PART_3"].copy()

    sort_cols = [col for col in ["geo_snapshot_source_ts", "grid_id", "grid_row", "grid_col"] if col in part3.columns]
    if not sort_cols:
        sort_cols = ["grid_id"]
    part3 = part3.sort_values(sort_cols).reset_index(drop=True)

    split_idx = int(len(part3) * valid_fraction_of_part3)
    split_idx = max(1, min(split_idx, len(part3) - 1)) if len(part3) > 1 else len(part3)
    valid_df = part3.iloc[:split_idx].copy()
    test_df = part3.iloc[split_idx:].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise RuntimeError(
            f"Temporal split failed: train={len(train_df)} valid={len(valid_df)} test={len(test_df)}. "
            "Expected PART_1/PART_2 for train and PART_3 for validation/test."
        )

    log.info("Temporal split -> TRAIN:%d | VALID:%d | TEST:%d", len(train_df), len(valid_df), len(test_df))
    return train_df, valid_df, test_df


def limited_walk_forward_splits(df: pd.DataFrame) -> list[dict[str, object]]:
    part1 = df[df["time_bucket"] == "PART_1"].copy()
    part2 = df[df["time_bucket"] == "PART_2"].copy()
    part3 = df[df["time_bucket"] == "PART_3"].copy()
    splits: list[dict[str, object]] = []
    if not part1.empty and not part2.empty:
        splits.append(
            {
                "name": "PART1_to_PART2",
                "train_label": "PART_1",
                "eval_label": "PART_2",
                "train_df": part1,
                "eval_df": part2,
            }
        )
    if not part1.empty and not part2.empty and not part3.empty:
        splits.append(
            {
                "name": "PART1PART2_to_PART3",
                "train_label": "PART_1 + PART_2",
                "eval_label": "PART_3",
                "train_df": pd.concat([part1, part2], ignore_index=True),
                "eval_df": part3,
            }
        )
    return splits


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


def _optuna_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    pipe = Pipeline([("prep", build_preprocessor()), ("model", XGBRegressor(**params))])
    pipe.fit(X_train, y_train)
    return rmse(y_valid.values, pipe.predict(X_valid))


def run_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    target_key: str,
    out_dir: Path,
    trials: int,
    timeout: int,
) -> dict:
    log.info("[%s] Optuna HPO: %d trials, timeout=%ds", target_key, trials, timeout)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name=f"model2_{target_key}",
    )
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, X_valid, y_valid),
        n_trials=trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    best = study.best_params
    save_json({"best_params": best, "best_rmse": study.best_value}, out_dir / "optuna_best_params.json")
    return best


def fit_pipeline(X_train: pd.DataFrame, y_train: pd.Series, best_params: dict) -> Pipeline:
    model_params = {
        **best_params,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    pipeline = Pipeline([("prep", build_preprocessor()), ("model", XGBRegressor(**model_params))])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_key: str,
    out_dir: Path,
) -> dict:
    results: dict[str, object] = {}
    for split_name, X, y in [("TRAIN", X_train, y_train), ("VALID", X_valid, y_valid), ("TEST", X_test, y_test)]:
        pred = pipeline.predict(X)
        metrics = compute_metrics(y.values, pred)
        results[split_name] = metrics
        log.info("[%s] %s -> MAE=%.4f | RMSE=%.4f | R2=%.4f", target_key, split_name, metrics["mae"], metrics["rmse"], metrics["r2"])

    gap = abs(results["TRAIN"]["mae"] - results["VALID"]["mae"]) / (results["VALID"]["mae"] + 1e-9)
    results["generalisation_gap_pct"] = round(gap * 100, 2)
    if gap > MAX_GENERALISATION_GAP:
        log.warning("[%s] Generalisation gap %.1f%% exceeds threshold %.0f%%", target_key, gap * 100, MAX_GENERALISATION_GAP * 100)

    residuals = y_test.values - pipeline.predict(X_test)
    results["residual_mean"] = round(float(np.mean(residuals)), 4)
    results["residual_std"] = round(float(np.std(residuals)), 4)
    save_json(results, out_dir / "metrics.json")
    return results


def run_walk_forward_validation(
    df: pd.DataFrame,
    target_key: str,
    best_params: dict,
    out_dir: Path,
) -> dict[str, object]:
    target = TARGET_CONFIG[target_key]["target"]
    folds: list[dict[str, object]] = []
    for split in limited_walk_forward_splits(df):
        train_df = split["train_df"]
        eval_df = split["eval_df"]
        pipeline = fit_pipeline(train_df[ALL_FEATURES], train_df[target], best_params)
        pred = pipeline.predict(eval_df[ALL_FEATURES])
        metrics = compute_metrics(eval_df[target].to_numpy(dtype=float), pred)
        folds.append(
            {
                "name": split["name"],
                "train": split["train_label"],
                "eval": split["eval_label"],
                "train_rows": int(len(train_df)),
                "eval_rows": int(len(eval_df)),
                "metrics": metrics,
            }
        )
    summary = {
        "target": target,
        "fold_count": len(folds),
        "folds": folds,
    }
    save_json(summary, out_dir / "walk_forward_validation.json")
    return summary


def run_clutter_transition_validation(
    pipeline: Pipeline,
    df_test: pd.DataFrame,
    target: str,
    target_key: str,
    out_dir: Path,
) -> dict[str, object]:
    log.info("[%s] Evaluating clutter-transition behaviour", target_key)
    work = df_test.copy()
    work["prediction"] = pipeline.predict(df_test[ALL_FEATURES])

    segments = {
        "all_test": work,
        "clutter_changed": work[work.get("clutter_transition_flag", 0) > 0],
        "clutter_static": work[work.get("clutter_transition_flag", 0) <= 0],
        "clutter_upgraded": work[work.get("clutter_upgrade_score", 0) > 0],
    }
    segment_metrics: dict[str, object] = {}
    for name, segment in segments.items():
        if segment.empty:
            segment_metrics[name] = {"rows": 0}
            continue
        metrics = compute_metrics(
            segment[target].to_numpy(dtype=float),
            segment["prediction"].to_numpy(dtype=float),
        )
        metrics["rows"] = int(len(segment))
        metrics["unique_grids"] = int(segment["grid_id"].nunique()) if "grid_id" in segment.columns else 0
        metrics["mean_actual"] = round(float(segment[target].mean()), 4)
        metrics["mean_prediction"] = round(float(segment["prediction"].mean()), 4)
        segment_metrics[name] = metrics

    by_clutter = (
        work.groupby("clutter_class", dropna=False, as_index=False)
        .apply(
            lambda part: pd.Series(
                {
                    "rows": int(len(part)),
                    "unique_grids": int(part["grid_id"].nunique()) if "grid_id" in part.columns else 0,
                    "mae": compute_metrics(
                        part[target].to_numpy(dtype=float),
                        part["prediction"].to_numpy(dtype=float),
                    )["mae"],
                    "wape_pct": compute_metrics(
                        part[target].to_numpy(dtype=float),
                        part["prediction"].to_numpy(dtype=float),
                    )["wape_pct"],
                }
            )
        )
        .reset_index(drop=True)
    )
    by_clutter.to_csv(out_dir / "clutter_transition_metrics.csv", index=False)
    summary = {
        "target": target,
        "segments": segment_metrics,
        "artifact_csv": "clutter_transition_metrics.csv",
    }
    save_json(summary, out_dir / "clutter_transition_validation.json")
    return summary


def run_shap(pipeline: Pipeline, X_test: pd.DataFrame, target_key: str, out_dir: Path) -> None:
    log.info("[%s] Computing SHAP values", target_key)
    try:
        import matplotlib
        import shap

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        prep = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
        X_t = prep.transform(X_test)
        try:
            feature_names = list(prep.get_feature_names_out())
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_t.shape[1])]

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_t)
        importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_vals).mean(axis=0)}).sort_values(
            "mean_abs_shap", ascending=False
        )
        importance.to_csv(out_dir / "shap_importance.csv", index=False)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals, X_t, feature_names=feature_names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:
        log.warning("[%s] SHAP step failed: %s", target_key, exc)


def run_yellowbrick(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_key: str,
    out_dir: Path,
) -> None:
    log.info("[%s] Generating Yellowbrick plots", target_key)
    try:
        import matplotlib
        from yellowbrick.regressor import PredictionError, ResidualsPlot

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        prep = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
        X_tr_t = prep.transform(X_train)
        X_te_t = prep.transform(X_test)

        fig, ax = plt.subplots(figsize=(8, 5))
        viz = ResidualsPlot(model, ax=ax, is_fitted=True)
        viz.fit(X_tr_t, y_train.values)
        viz.score(X_te_t, y_test.values)
        viz.finalize()
        fig.savefig(out_dir / "yb_residuals.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        fig, ax = plt.subplots(figsize=(8, 5))
        viz2 = PredictionError(model, ax=ax, is_fitted=True)
        viz2.fit(X_tr_t, y_train.values)
        viz2.score(X_te_t, y_test.values)
        viz2.finalize()
        fig.savefig(out_dir / "yb_prediction_error.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:
        log.warning("[%s] Yellowbrick step failed: %s", target_key, exc)


def run_evidently(pipeline: Pipeline, df_train: pd.DataFrame, df_test: pd.DataFrame, target: str, target_key: str, out_dir: Path) -> None:
    log.info("[%s] Generating Evidently report", target_key)
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, RegressionPreset
        from evidently.report import Report

        ref = df_train[ALL_FEATURES + [target]].copy()
        cur = df_test[ALL_FEATURES + [target]].copy()
        ref["prediction"] = pipeline.predict(df_train[ALL_FEATURES])
        cur["prediction"] = pipeline.predict(df_test[ALL_FEATURES])

        col_mapping = ColumnMapping(
            target=target,
            prediction="prediction",
            numerical_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
        report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
        report.run(reference_data=ref, current_data=cur, column_mapping=col_mapping)
        report.save_html(str(out_dir / "evidently_report.html"))
    except Exception as exc:
        log.warning("[%s] Evidently step failed: %s", target_key, exc)


def save_prediction_outputs(
    pipeline: Pipeline,
    df: pd.DataFrame,
    target: str,
    target_key: str,
    out_dir: Path,
) -> pd.DataFrame:
    pred_df = df.copy()
    pred_df["prediction"] = pipeline.predict(pred_df[ALL_FEATURES])
    pred_df["residual"] = pred_df[target] - pred_df["prediction"]
    pred_df["abs_error"] = pred_df["residual"].abs()

    keep_cols = [
        "time_bucket",
        "bucket_seq",
        "grid_id",
        "grid_row",
        "grid_col",
        "grid_centroid_lat",
        "grid_centroid_lon",
        "clutter_class",
        "dominant_pci",
        target,
        "prediction",
        "residual",
        "abs_error",
        "demand_index",
        "active_users_est",
        "traffic_demand_est",
        "prb_pressure_est",
        "growth_rate",
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
    keep_cols = [col for col in keep_cols if col in pred_df.columns]
    pred_df[keep_cols].to_csv(out_dir / "test_predictions.csv", index=False)
    return pred_df


def risk_label(series: pd.Series) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    hi = valid.quantile(0.80)
    med = valid.quantile(0.50)
    return pd.Series(np.where(valid >= hi, "High", np.where(valid >= med, "Medium", "Low")), index=series.index)


def build_business_outputs(all_test_predictions: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, object]:
    log.info("Generating Model 2 business aggregation outputs")
    base = None
    for target_key, pred_df in all_test_predictions.items():
        target = TARGET_CONFIG[target_key]["target"]
        cols = ["grid_id", "time_bucket", "bucket_seq", "prediction", target]
        optional = [
            "grid_centroid_lat",
            "grid_centroid_lon",
            "clutter_class",
            "dominant_pci",
            "prb_pressure_est",
            "growth_rate",
            "demand_index",
            "active_users_est",
            "traffic_demand_est",
        ]
        cols += [col for col in optional if col in pred_df.columns and col not in cols]
        current = pred_df[cols].copy().rename(columns={"prediction": f"{target}_pred"})
        if base is None:
            base = current
        else:
            merge_keys = [col for col in ["grid_id", "time_bucket", "bucket_seq"] if col in current.columns and col in base.columns]
            base = base.merge(current[merge_keys + [f"{target}_pred"]], on=merge_keys, how="left")

    if base is None or base.empty:
        return {"status": "empty"}

    score_cols = ["demand_index_pred", "active_users_est_pred", "traffic_demand_est_pred"]
    available_score_cols = [col for col in score_cols if col in base.columns]
    norm_parts = []
    for col in available_score_cols:
        values = pd.to_numeric(base[col], errors="coerce")
        denom = values.max() - values.min()
        if pd.isna(denom) or denom == 0:
            norm_parts.append(pd.Series(0.0, index=base.index))
        else:
            norm_parts.append((values - values.min()) / denom)
    base["capacity_risk_score"] = np.mean(norm_parts, axis=0) * 100.0 if norm_parts else 0.0
    base["capacity_risk_level"] = risk_label(base["capacity_risk_score"])

    base = base.sort_values(["capacity_risk_score"], ascending=False)
    grid_csv = out_dir / "top_congested_grids.csv"
    base.head(100).to_csv(grid_csv, index=False)

    grouping_outputs: dict[str, str] = {"top_congested_grids": str(grid_csv)}
    group_specs = []
    if "dominant_pci" in base.columns:
        group_specs.append(("cell", ["dominant_pci"]))
        group_specs.append(("sector", ["dominant_pci"]))
        group_specs.append(("site", ["dominant_pci"]))

    if not group_specs:
        summary = {
            "status": "grid_only",
            "reason": "No true cell/sector/site identifier exists in model2_capacity_training.csv.",
            "grid_output": str(grid_csv),
        }
        save_json(summary, out_dir / "business_aggregation_summary.json")
        return summary

    for name, keys in group_specs:
        agg = (
            base.dropna(subset=keys)
            .groupby(keys, as_index=False)
            .agg(
                grid_count=("grid_id", "nunique"),
                row_count=("grid_id", "size"),
                risk_score=("capacity_risk_score", "mean"),
                demand_pred=("demand_index_pred", "mean") if "demand_index_pred" in base.columns else ("capacity_risk_score", "mean"),
                users_pred=("active_users_est_pred", "mean") if "active_users_est_pred" in base.columns else ("capacity_risk_score", "mean"),
                traffic_pred=("traffic_demand_est_pred", "mean") if "traffic_demand_est_pred" in base.columns else ("capacity_risk_score", "mean"),
                prb_pressure=("prb_pressure_est", "mean") if "prb_pressure_est" in base.columns else ("capacity_risk_score", "mean"),
            )
            .sort_values("risk_score", ascending=False)
        )
        agg["risk_level"] = risk_label(agg["risk_score"])
        path = out_dir / f"top_congested_{name}s.csv"
        agg.head(100).to_csv(path, index=False)
        grouping_outputs[f"top_congested_{name}s"] = str(path)

    summary = {
        "status": "ok",
        "aggregation_key_note": "dominant_pci was used as the available cell-like identifier. Add real cell/sector/site columns for true planner aggregation.",
        "outputs": grouping_outputs,
    }
    save_json(summary, out_dir / "business_aggregation_summary.json")
    return summary


def save_model(
    pipeline: Pipeline,
    target_key: str,
    metrics: dict,
    best_params: dict,
    out_dir: Path,
    trials: int,
    timeout: int,
    walk_forward_summary: dict[str, object],
    clutter_validation_summary: dict[str, object],
) -> None:
    cfg = TARGET_CONFIG[target_key]
    target = cfg["target"]
    target_model_path = out_dir / cfg["model_file"]
    root_model_path = MODEL_ROOT / cfg["model_file"]
    joblib.dump(pipeline, target_model_path)
    joblib.dump(pipeline, root_model_path)

    metadata = {
        "model": target_key,
        "target": target,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "XGBRegressor",
        "purpose": cfg["purpose"],
        "business_meaning": cfg["business_meaning"],
        "note": cfg.get("note"),
        "training_granularity": "grid_id + time_bucket",
        "training_dataset_csv": str(MODEL2_DATASET_CSV),
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "excluded": ["estimated_prb_mean"],
            "total": len(ALL_FEATURES),
        },
        "hyperparameters": best_params,
        "metrics": metrics,
        "split_logic": {
            "train": "PART_1 + PART_2",
            "valid": "first 60% of PART_3 in stable timestamp/grid order",
            "test": "last 40% of PART_3 in stable timestamp/grid order",
            "leakage_rule": "No PART_3 rows are used for training.",
            "walk_forward_reference": "PART_1 -> PART_2, then PART_1 + PART_2 -> PART_3",
        },
        "walk_forward_validation": walk_forward_summary,
        "clutter_transition_validation": clutter_validation_summary,
        "artifacts": [
            cfg["model_file"],
            "metrics.json",
            "metadata.json",
            "optuna_best_params.json",
            "shap_importance.csv",
            "shap_summary.png",
            "yb_residuals.png",
            "yb_prediction_error.png",
            "evidently_report.html",
            "test_predictions.csv",
            "walk_forward_validation.json",
            "clutter_transition_validation.json",
            "clutter_transition_metrics.csv",
        ],
        "random_seed": RANDOM_SEED,
        "optuna_trials": trials,
        "optuna_timeout": timeout,
    }
    save_json(metadata, out_dir / "metadata.json")


def train_target(
    df_full: pd.DataFrame,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    target_key: str,
    trials: int,
    timeout: int,
) -> pd.DataFrame:
    cfg = TARGET_CONFIG[target_key]
    target = cfg["target"]
    out_dir = MODEL_ROOT / target_key
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = df_train[ALL_FEATURES], df_train[target]
    X_valid, y_valid = df_valid[ALL_FEATURES], df_valid[target]
    X_test, y_test = df_test[ALL_FEATURES], df_test[target]

    best_params = run_optuna(X_train, y_train, X_valid, y_valid, target_key, out_dir, trials, timeout)
    pipeline = fit_pipeline(X_train, y_train, best_params)
    metrics = evaluate(pipeline, X_train, y_train, X_valid, y_valid, X_test, y_test, target_key, out_dir)
    walk_forward_summary = run_walk_forward_validation(df_full, target_key, best_params, out_dir)
    clutter_validation_summary = run_clutter_transition_validation(pipeline, df_test, target, target_key, out_dir)
    run_shap(pipeline, X_test, target_key, out_dir)
    run_yellowbrick(pipeline, X_train, y_train, X_test, y_test, target_key, out_dir)
    run_evidently(pipeline, df_train, df_test, target, target_key, out_dir)
    pred_df = save_prediction_outputs(pipeline, df_test, target, target_key, out_dir)
    save_model(
        pipeline,
        target_key,
        metrics,
        best_params,
        out_dir,
        trials,
        timeout,
        walk_forward_summary,
        clutter_validation_summary,
    )
    return pred_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model 2 capacity/congestion/demand XGBoost models.")
    parser.add_argument("--trials", type=int, default=DEFAULT_OPTUNA_TRIALS, help="Optuna trials per target")
    parser.add_argument("--timeout", type=int, default=DEFAULT_OPTUNA_TIMEOUT, help="Optuna timeout seconds per target")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log.info("Model 2 training started at %s", datetime.utcnow().isoformat())
    df = load_dataset()
    train_df, valid_df, test_df = temporal_split(df)

    dataset_summary = {
        "source_dataset": str(MODEL2_DATASET_CSV),
        "source_summary": str(MODEL2_DATASET_SUMMARY_JSON),
        "rows": int(len(df)),
        "unique_grids": int(df["grid_id"].nunique()),
        "bucket_counts": {str(k): int(v) for k, v in df["time_bucket"].value_counts().sort_index().items()},
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "excluded": ["estimated_prb_mean"],
        },
        "targets": TARGET_CONFIG,
        "split_rows": {"train": int(len(train_df)), "valid": int(len(valid_df)), "test": int(len(test_df))},
    }
    save_json(dataset_summary, MODEL_ROOT / "dataset_training_summary.json")

    all_metrics: dict[str, object] = {}
    all_test_predictions: dict[str, pd.DataFrame] = {}
    for target_key in TARGET_CONFIG:
        log.info("=" * 70)
        log.info("TARGET: %s (%s)", target_key, TARGET_CONFIG[target_key]["target"])
        log.info("=" * 70)
        all_test_predictions[target_key] = train_target(df, train_df, valid_df, test_df, target_key, args.trials, args.timeout)
        metrics_path = MODEL_ROOT / target_key / "metrics.json"
        if metrics_path.exists():
            all_metrics[target_key] = json.loads(metrics_path.read_text(encoding="utf-8"))

    business_summary = build_business_outputs(all_test_predictions, MODEL_ROOT)
    save_json(all_metrics, MODEL_ROOT / "all_metrics_summary.json")
    save_json({"metrics": all_metrics, "business_outputs": business_summary}, MODEL_ROOT / "model2_training_summary.json")
    log.info("All Model 2 artifacts saved under %s", MODEL_ROOT.resolve())


if __name__ == "__main__":
    main()
