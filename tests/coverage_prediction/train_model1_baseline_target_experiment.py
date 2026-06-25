"""
Model 1 baseline-target experiment.

This is a separate teacher-surface experiment. It does not modify the existing
DT-based Model 1 artifacts. The goal is to check whether training on the dense
LTE baseline prediction surface is useful, and whether that surface agrees with
held-out measured DT labels where they exist.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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

from tests.coverage_prediction import train_model1_coverage_xgboost as base


OUTPUT_ROOT = Path("models") / "model1_baseline_target_experiment"
DATASET_CSV = OUTPUT_ROOT / "baseline_target_training.csv"
SUMMARY_JSON = OUTPUT_ROOT / "baseline_target_experiment_summary.json"
SUMMARY_CSV = OUTPUT_ROOT / "baseline_target_metrics.csv"
DT_AGREEMENT_CSV = OUTPUT_ROOT / "dt_agreement_metrics.csv"
TARGETS = ["pred_rsrp", "pred_rsrq", "pred_sinr"]
COORDINATE_FEATURES = {"grid_centroid_lat", "grid_centroid_lon"}
TEACHER_SUMMARY_FEATURES = {
    "bucket_seq",
    "pred_rsrp_min",
    "pred_rsrp_max",
    "pred_rsrp_std",
    "pred_sinr_std",
    "prev_obs_rsrp",
    "prev_obs_rsrq",
    "prev_obs_sinr",
    "prev2_obs_rsrp",
    "prev2_obs_rsrq",
    "prev2_obs_sinr",
    "prev_trend_rsrp",
    "prev_trend_rsrq",
    "prev_trend_sinr",
}
HOLE_THRESHOLDS = {"pred_rsrp": -105.0, "pred_rsrq": -14.0, "pred_sinr": 0.0}


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred).correlation),
    }


def clean_geo_columns(geo_df: pd.DataFrame) -> pd.DataFrame:
    return geo_df.drop(
        columns=[
            "centroid_lat",
            "centroid_lon",
            "geo_snapshot_mode",
            "geo_snapshot_ts_utc",
            "geo_layer_modes_json",
            "bucket_start",
            "bucket_end",
            "geo_snapshot_source_ts",
            "building_count_calc",
            "building_area_sum_m2_calc",
            "avg_building_area_m2_calc",
            "building_area_sum_m2",
        ],
        errors="ignore",
    )


def build_baseline_target_dataset() -> pd.DataFrame:
    pred_df = base._read_csv_from_archive(base.COVERAGE_ARCHIVE, "baseline_prediction_grid.csv")
    corrected_pred_df = base._read_csv_from_archive(base.COVERAGE_ARCHIVE, "bucket_corrected_prediction_grid.csv")
    kpi_df = base._read_csv_from_archive(base.COVERAGE_ARCHIVE, "grid_kpi_timeseries.csv")
    geo_df = base._read_csv_from_archive(base.COVERAGE_ARCHIVE, "bucket_grid_geo_features.csv")
    site_df = base._read_csv_from_archive(base.COVERAGE_ARCHIVE, "project_sites.csv")

    pred_df = base._add_rf_topology_features(pred_df, site_df)
    group_cols = ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"]
    dense_df = (
        pred_df.groupby(group_cols, as_index=False)
        .agg(
            pred_rsrp=("pred_rsrp", "mean"),
            pred_rsrq=("pred_rsrq", "mean"),
            pred_sinr=("pred_sinr", "mean"),
            pred_rsrp_min=("pred_rsrp", "min"),
            pred_rsrp_max=("pred_rsrp", "max"),
            pred_rsrp_std=("pred_rsrp", "std"),
            pred_sinr_std=("pred_sinr", "std"),
            measurement_count=("pred_rsrp", "count"),
            unique_cells=("Node_Cell_ID", "nunique"),
            unique_sites=("site_id", "nunique"),
            serving_distance_m=("serving_distance_m", "mean"),
            nearest_site_distance_m=("nearest_site_distance_m", "mean"),
            site_count_250m=("site_count_250m", "mean"),
            site_count_500m=("site_count_500m", "mean"),
            azimuth_delta_deg=("azimuth_delta_deg", "mean"),
        )
    )

    # There is no measured timestamp for pseudo-label rows. Use the bucket order
    # as a stable synthetic temporal position for the existing split code.
    bucket_seq_map = {"PART_1": 1, "PART_2": 2, "PART_3": 3}
    dense_df["bucket_seq"] = dense_df["time_bucket"].map(bucket_seq_map).astype(float)
    dense_df["bucket_min_timestamp"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(dense_df["bucket_seq"], unit="D")
    dense_df["bucket_max_timestamp"] = dense_df["bucket_min_timestamp"]
    dense_df["bucket_mid_timestamp"] = dense_df["bucket_min_timestamp"]

    corrected_features = base._derive_corrected_surface_features(corrected_pred_df)
    if not corrected_features.empty:
        dense_df = dense_df.merge(corrected_features, on=group_cols, how="left", validate="one_to_one")

    band_features = base._derive_band_features_from_kpi(kpi_df)
    if not band_features.empty:
        dense_df = dense_df.merge(band_features, on=group_cols, how="left", validate="one_to_one")

    geo_df = clean_geo_columns(geo_df)
    dense_df = dense_df.merge(
        geo_df,
        on=["grid_id", "grid_row", "grid_col", "time_bucket"],
        how="left",
        validate="one_to_one",
    )

    dense_df = base._add_temporal_history_features(dense_df)
    dense_df = base._add_clutter_evolution_features(dense_df)
    dense_df = dense_df.sort_values(["grid_id", "bucket_seq"]).reset_index(drop=True)
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    dense_df.to_csv(DATASET_CSV, index=False)
    return dense_df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["time_bucket"].isin(["PART_1", "PART_2"])].copy()
    part3 = df[df["time_bucket"] == "PART_3"].sort_values(["grid_row", "grid_col", "grid_id"]).reset_index(drop=True)
    split_idx = int(len(part3) * 0.60)
    valid_df = part3.iloc[:split_idx].copy()
    test_df = part3.iloc[split_idx:].copy()
    return train_df, part3, valid_df, test_df


def variant_features(variant: str) -> tuple[list[str], list[str]]:
    numeric = [feature for feature in base.NUMERIC_FEATURES if feature not in COORDINATE_FEATURES]
    categorical = list(base.CATEGORICAL_FEATURES)
    if variant == "physical_no_teacher_summary":
        numeric = [feature for feature in numeric if feature not in TEACHER_SUMMARY_FEATURES]
    return numeric, categorical


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def fit_model(train_df: pd.DataFrame, target: str, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    params = {
        "n_estimators": 450,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_alpha": 0.001,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": base.RANDOM_SEED,
        "n_jobs": -1,
    }
    features = numeric_features + categorical_features
    pipe = Pipeline([("prep", build_preprocessor(numeric_features, categorical_features)), ("model", XGBRegressor(**params))])
    pipe.fit(train_df[features], train_df[target])
    return pipe


def bad_counts(df: pd.DataFrame, prefix: str = "") -> dict[str, int]:
    rsrp = df[f"{prefix}pred_rsrp"] <= HOLE_THRESHOLDS["pred_rsrp"]
    rsrq = df[f"{prefix}pred_rsrq"] < HOLE_THRESHOLDS["pred_rsrq"]
    sinr = df[f"{prefix}pred_sinr"] < HOLE_THRESHOLDS["pred_sinr"]
    any_bad = rsrp | rsrq | sinr
    return {
        "rows": int(len(df)),
        "bad_rsrp": int(rsrp.sum()),
        "bad_rsrq": int(rsrq.sum()),
        "bad_sinr": int(sinr.sum()),
        "bad_any": int(any_bad.sum()),
        "good": int((~any_bad).sum()),
    }


def binary_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual = actual.astype(bool)
    predicted = predicted.astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def observed_dt_grid_labels() -> pd.DataFrame:
    coverage_rows = base._read_csv_from_archive(base.COVERAGE_ARCHIVE, "coverage_rows.csv")
    keys = ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"]
    return (
        coverage_rows.groupby(keys, as_index=False)
        .agg(dt_rsrp=("rsrp", "mean"), dt_rsrq=("rsrq", "mean"), dt_sinr=("sinr", "mean"), dt_samples=("grid_id", "size"))
        .dropna(subset=["dt_rsrp", "dt_rsrq", "dt_sinr"])
    )


def evaluate_dt_agreement(
    variant: str,
    target: str,
    model: Pipeline,
    dense_df: pd.DataFrame,
    dt_df: pd.DataFrame,
    features: list[str],
) -> list[dict[str, Any]]:
    joined = dense_df.merge(dt_df, on=["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"], how="inner")
    joined = joined.dropna(subset=[target, f"dt_{target.replace('pred_', '')}"]).copy()
    joined[f"{target}_model_pred"] = model.predict(joined[features])
    dt_col = f"dt_{target.replace('pred_', '')}"
    rows = []
    for split_name, frame in [("all_dt_overlap", joined), ("part3_dt_overlap", joined[joined["time_bucket"] == "PART_3"])]:
        if frame.empty:
            continue
        threshold = HOLE_THRESHOLDS[target]
        if target == "pred_rsrp":
            dt_bad = frame[dt_col] <= threshold
            baseline_bad = frame[target] <= threshold
            model_bad = frame[f"{target}_model_pred"] <= threshold
        else:
            dt_bad = frame[dt_col] < threshold
            baseline_bad = frame[target] < threshold
            model_bad = frame[f"{target}_model_pred"] < threshold
        rows.append(
            {
                "variant": variant,
                "target": target,
                "split": split_name,
                "rows": int(len(frame)),
                "baseline_vs_dt": compute_metrics(frame[dt_col], frame[target].to_numpy()),
                "model_vs_dt": compute_metrics(frame[dt_col], frame[f"{target}_model_pred"].to_numpy()),
                "baseline_hole_vs_dt": binary_metrics(dt_bad, baseline_bad),
                "model_hole_vs_dt": binary_metrics(dt_bad, model_bad),
                "dt_bad_count": int(dt_bad.sum()),
                "baseline_bad_count": int(baseline_bad.sum()),
                "model_bad_count": int(model_bad.sum()),
            }
        )
    return rows


def create_plots(metrics_df: pd.DataFrame, dt_df: pd.DataFrame) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, str] = {}
    holdout = metrics_df[metrics_df["split"] == "baseline_test_part3"].copy()
    if not holdout.empty:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        labels = holdout["variant"] + " / " + holdout["target"].str.replace("pred_", "", regex=False)
        ax.bar(labels, holdout["r2"], color="#2f6f73")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Baseline-Target Holdout R2 on PART_3", fontsize=14, fontweight="bold")
        ax.set_ylabel("R2 against baseline target")
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.grid(axis="y", alpha=0.25)
        output = OUTPUT_ROOT / "baseline_holdout_r2.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths["baseline_holdout_r2"] = str(output)

    flat_rows = []
    for row in dt_df.to_dict(orient="records"):
        flat_rows.append(
            {
                "variant": row["variant"],
                "target": row["target"],
                "split": row["split"],
                "baseline_mae": row["baseline_vs_dt"]["mae"],
                "model_mae": row["model_vs_dt"]["mae"],
            }
        )
    flat = pd.DataFrame(flat_rows)
    if not flat.empty:
        plot_df = flat[flat["split"] == "part3_dt_overlap"].copy()
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        x = np.arange(len(plot_df))
        width = 0.35
        ax.bar(x - width / 2, plot_df["baseline_mae"], width, label="baseline vs DT", color="#7a5c22")
        ax.bar(x + width / 2, plot_df["model_mae"], width, label="model vs DT", color="#5b5f97")
        ax.set_xticks(x, labels=(plot_df["variant"] + " / " + plot_df["target"].str.replace("pred_", "", regex=False)), rotation=35, ha="right")
        ax.set_title("DT Agreement on PART_3 Overlap", fontsize=14, fontweight="bold")
        ax.set_ylabel("MAE against measured DT")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        output = OUTPUT_ROOT / "dt_agreement_mae.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths["dt_agreement_mae"] = str(output)
    return paths


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dense_df = build_baseline_target_dataset()
    train_df, part3_df, valid_df, test_df = temporal_split(dense_df)
    dt_df = observed_dt_grid_labels()

    variants = ["with_existing_features", "physical_no_teacher_summary"]
    metric_rows: list[dict[str, Any]] = []
    dt_rows: list[dict[str, Any]] = []
    model_paths: dict[str, dict[str, str]] = {}

    for variant in variants:
        numeric_features, categorical_features = variant_features(variant)
        features = numeric_features + categorical_features
        variant_root = OUTPUT_ROOT / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        model_paths[variant] = {}

        for target in TARGETS:
            model = fit_model(train_df, target, numeric_features, categorical_features)
            model_path = variant_root / f"{target}.joblib"
            joblib.dump(model, model_path)
            model_paths[variant][target] = str(model_path)

            for split_name, frame in [
                ("baseline_train_part1_part2", train_df),
                ("baseline_valid_part3_first60", valid_df),
                ("baseline_test_part3", test_df),
            ]:
                pred = model.predict(frame[features])
                row = {
                    "variant": variant,
                    "target": target,
                    "split": split_name,
                    **compute_metrics(frame[target], pred),
                    "actual_mean": float(frame[target].mean()),
                    "pred_mean": float(np.mean(pred)),
                }
                metric_rows.append(row)
            dt_rows.extend(evaluate_dt_agreement(variant, target, model, dense_df, dt_df, features))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(SUMMARY_CSV, index=False)
    dt_agreement_df = pd.DataFrame(dt_rows)
    dt_agreement_df.to_csv(DT_AGREEMENT_CSV, index=False)
    plot_paths = create_plots(metrics_df, dt_agreement_df)

    bucket_bad_counts = {
        "baseline_dense_surface": {bucket: bad_counts(group) for bucket, group in dense_df.groupby("time_bucket")},
        "dt_measured_overlap": {
            bucket: {
                "rows": int(len(group)),
                "bad_rsrp": int((group["dt_rsrp"] <= HOLE_THRESHOLDS["pred_rsrp"]).sum()),
                "bad_rsrq": int((group["dt_rsrq"] < HOLE_THRESHOLDS["pred_rsrq"]).sum()),
                "bad_sinr": int((group["dt_sinr"] < HOLE_THRESHOLDS["pred_sinr"]).sum()),
                "bad_any": int(
                    (
                        (group["dt_rsrp"] <= HOLE_THRESHOLDS["pred_rsrp"])
                        | (group["dt_rsrq"] < HOLE_THRESHOLDS["pred_rsrq"])
                        | (group["dt_sinr"] < HOLE_THRESHOLDS["pred_sinr"])
                    ).sum()
                ),
            }
            for bucket, group in dt_df.groupby("time_bucket")
        },
    }

    summary = {
        "dataset_csv": str(DATASET_CSV),
        "output_root": str(OUTPUT_ROOT),
        "row_counts": {
            "baseline_dense_rows": int(len(dense_df)),
            "train_part1_part2": int(len(train_df)),
            "part3_total": int(len(part3_df)),
            "valid_first60_part3": int(len(valid_df)),
            "test_last40_part3": int(len(test_df)),
            "dt_labeled_rows": int(len(dt_df)),
        },
        "variants": {
            "with_existing_features": {
                "description": "Coordinate-free existing Model 1 feature set, including baseline-derived summary/history features.",
                "numeric_features": variant_features("with_existing_features")[0],
                "categorical_features": variant_features("with_existing_features")[1],
            },
            "physical_no_teacher_summary": {
                "description": "Coordinate-free feature set with bucket and direct baseline-summary/history features removed.",
                "removed_features": sorted(TEACHER_SUMMARY_FEATURES),
                "numeric_features": variant_features("physical_no_teacher_summary")[0],
                "categorical_features": variant_features("physical_no_teacher_summary")[1],
            },
        },
        "model_paths": model_paths,
        "baseline_target_metrics_csv": str(SUMMARY_CSV),
        "dt_agreement_metrics_csv": str(DT_AGREEMENT_CSV),
        "plots": plot_paths,
        "bucket_bad_counts": bucket_bad_counts,
        "key_results": {
            "baseline_holdout_part3": metrics_df[metrics_df["split"] == "baseline_test_part3"].to_dict(orient="records"),
            "dt_agreement": dt_rows,
        },
        "answer": (
            "This experiment tells whether the baseline surface is learnable and whether the learned baseline-target model "
            "agrees with measured DT labels at overlap locations. High baseline holdout R2 alone is not enough; DT agreement "
            "and hole detection against measured DT decide whether the baseline is useful supervision."
        ),
    }
    save_json(summary, SUMMARY_JSON)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
