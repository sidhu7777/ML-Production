"""
Model 1 hybrid-target experiment.

Builds a dense grid dataset where measured DT labels replace baseline labels
when available, and baseline labels fill the remaining grids. This keeps the
existing DT-only and baseline-only experiments untouched.
"""

from __future__ import annotations

import argparse
import json
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


OUTPUT_ROOT = Path("models") / "model1_hybrid_target_experiment"
DATASET_CSV = OUTPUT_ROOT / "hybrid_target_training.csv"
SUMMARY_JSON = OUTPUT_ROOT / "hybrid_target_experiment_summary.json"
METRICS_CSV = OUTPUT_ROOT / "hybrid_target_metrics.csv"
PLANNING_CSV = OUTPUT_ROOT / "hybrid_planning_metrics.csv"
DIAGNOSTICS_JSON = OUTPUT_ROOT / "hybrid_diagnostics_summary.json"
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
WEAK_THRESHOLDS = {"pred_rsrp": -95.0, "pred_rsrq": -10.0, "pred_sinr": 3.0}
SEVERITY_SCALES = {"pred_rsrp": 20.0, "pred_rsrq": 5.0, "pred_sinr": 6.0}


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    corr = spearmanr(y_true, y_pred).correlation
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(corr) if np.isfinite(corr) else None,
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


def observed_dt_grid_labels() -> pd.DataFrame:
    coverage_rows = base._read_csv_from_archive(base.COVERAGE_ARCHIVE, "coverage_rows.csv")
    keys = ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"]
    return (
        coverage_rows.groupby(keys, as_index=False)
        .agg(dt_rsrp=("rsrp", "mean"), dt_rsrq=("rsrq", "mean"), dt_sinr=("sinr", "mean"), dt_samples=("grid_id", "size"))
        .dropna(subset=["dt_rsrp", "dt_rsrq", "dt_sinr"])
    )


def build_dense_baseline_features() -> pd.DataFrame:
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
            baseline_rsrp=("pred_rsrp", "mean"),
            baseline_rsrq=("pred_rsrq", "mean"),
            baseline_sinr=("pred_sinr", "mean"),
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

    dense_df = dense_df.merge(
        clean_geo_columns(geo_df),
        on=["grid_id", "grid_row", "grid_col", "time_bucket"],
        how="left",
        validate="one_to_one",
    )
    return dense_df


def build_hybrid_dataset() -> pd.DataFrame:
    dense_df = build_dense_baseline_features()
    dt_df = observed_dt_grid_labels()
    keys = ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"]
    hybrid = dense_df.merge(dt_df, on=keys, how="left")

    has_dt = hybrid[["dt_rsrp", "dt_rsrq", "dt_sinr"]].notna().all(axis=1)
    hybrid["label_source"] = np.where(has_dt, "DT", "baseline")
    hybrid["pred_rsrp"] = np.where(has_dt, hybrid["dt_rsrp"], hybrid["baseline_rsrp"])
    hybrid["pred_rsrq"] = np.where(has_dt, hybrid["dt_rsrq"], hybrid["baseline_rsrq"])
    hybrid["pred_sinr"] = np.where(has_dt, hybrid["dt_sinr"], hybrid["baseline_sinr"])

    hybrid = base._add_temporal_history_features(hybrid)
    hybrid = base._add_clutter_evolution_features(hybrid)
    hybrid = hybrid.sort_values(["grid_id", "bucket_seq"]).reset_index(drop=True)
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    hybrid.to_csv(DATASET_CSV, index=False)
    return hybrid


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
        "n_estimators": 550,
        "max_depth": 5,
        "learning_rate": 0.035,
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


def is_bad(frame: pd.DataFrame, prefix: str = "") -> pd.Series:
    rsrp = frame[f"{prefix}pred_rsrp"] <= HOLE_THRESHOLDS["pred_rsrp"]
    rsrq = frame[f"{prefix}pred_rsrq"] < HOLE_THRESHOLDS["pred_rsrq"]
    sinr = frame[f"{prefix}pred_sinr"] < HOLE_THRESHOLDS["pred_sinr"]
    return rsrp | rsrq | sinr


def severity(frame: pd.DataFrame, prefix: str = "") -> pd.Series:
    rsrp = np.maximum(0.0, (WEAK_THRESHOLDS["pred_rsrp"] - frame[f"{prefix}pred_rsrp"]) / SEVERITY_SCALES["pred_rsrp"])
    rsrq = np.maximum(0.0, (WEAK_THRESHOLDS["pred_rsrq"] - frame[f"{prefix}pred_rsrq"]) / SEVERITY_SCALES["pred_rsrq"])
    sinr = np.maximum(0.0, (WEAK_THRESHOLDS["pred_sinr"] - frame[f"{prefix}pred_sinr"]) / SEVERITY_SCALES["pred_sinr"])
    return rsrp + rsrq + sinr


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


def top_overlap(actual_scores: pd.Series, pred_scores: pd.Series, ids: pd.Series, k: int = 20) -> dict[str, Any]:
    actual_top = set(ids.iloc[np.argsort(-actual_scores.to_numpy())[:k]].astype(str))
    pred_top = set(ids.iloc[np.argsort(-pred_scores.to_numpy())[:k]].astype(str))
    overlap = actual_top.intersection(pred_top)
    corr = spearmanr(actual_scores, pred_scores).correlation
    return {
        "top_k": k,
        "overlap_count": int(len(overlap)),
        "overlap_ratio": float(len(overlap) / k),
        "spearman": float(corr) if np.isfinite(corr) else None,
    }


def evaluate_planning(pred_frame: pd.DataFrame, split: str, variant: str) -> dict[str, Any]:
    actual_bad = is_bad(pred_frame)
    pred_bad = is_bad(pred_frame, "model_")
    actual_severity = severity(pred_frame)
    pred_severity = severity(pred_frame, "model_")

    grid_scores = (
        pd.DataFrame(
            {
                "grid_id": pred_frame["grid_id"],
                "actual_severity": actual_severity,
                "pred_severity": pred_severity,
            }
        )
        .groupby("grid_id", as_index=False)
        .agg(actual_severity=("actual_severity", "mean"), pred_severity=("pred_severity", "mean"))
    )

    return {
        "variant": variant,
        "split": split,
        "rows": int(len(pred_frame)),
        "actual_bad_count": int(actual_bad.sum()),
        "pred_bad_count": int(pred_bad.sum()),
        "hole_metrics": binary_metrics(actual_bad, pred_bad),
        "severity_mae": float(np.mean(np.abs(pred_severity - actual_severity))),
        "severity_mean_actual": float(actual_severity.mean()),
        "severity_mean_pred": float(pred_severity.mean()),
        "top20_grid_overlap": top_overlap(grid_scores["actual_severity"], grid_scores["pred_severity"], grid_scores["grid_id"]),
    }


def create_plots(metrics_df: pd.DataFrame, planning_df: pd.DataFrame) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, str] = {}
    test_df = metrics_df[metrics_df["split"] == "test_last40_part3"].copy()
    if not test_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        labels = test_df["variant"] + " / " + test_df["target"].str.replace("pred_", "", regex=False)
        ax.bar(labels, test_df["r2"], color="#2f6f73")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Hybrid-Target Test R2 on PART_3", fontsize=14, fontweight="bold")
        ax.set_ylabel("R2")
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.grid(axis="y", alpha=0.25)
        output = OUTPUT_ROOT / "hybrid_test_r2.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths["hybrid_test_r2"] = str(output)

    if not planning_df.empty:
        plot_df = planning_df[planning_df["split"] == "test_last40_part3"].copy()
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        ax.bar(plot_df["variant"], [item["f1"] for item in plot_df["hole_metrics"]], color="#5b5f97")
        ax.set_title("Hybrid Coverage-Hole F1 on PART_3 Test", fontsize=14, fontweight="bold")
        ax.set_ylabel("F1")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        output = OUTPUT_ROOT / "hybrid_coverage_hole_f1.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths["hybrid_coverage_hole_f1"] = str(output)
    return paths


def load_or_build_hybrid_dataset() -> pd.DataFrame:
    if DATASET_CSV.exists():
        return pd.read_csv(DATASET_CSV)
    return build_hybrid_dataset()


def load_model(variant: str, target: str) -> Pipeline:
    model_path = OUTPUT_ROOT / variant / f"{target}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing hybrid model: {model_path}")
    return joblib.load(model_path)


def transformed_feature_names(model: Pipeline) -> list[str]:
    prep = model.named_steps["prep"]
    try:
        return list(prep.get_feature_names_out())
    except Exception:
        return []


def run_shap_diagnostics(model: Pipeline, frame: pd.DataFrame, features: list[str], target: str, out_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    try:
        import matplotlib
        import shap

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample = frame.sample(n=min(len(frame), 800), random_state=base.RANDOM_SEED) if len(frame) > 800 else frame
        transformed = model.named_steps["prep"].transform(sample[features])
        feature_names = transformed_feature_names(model) or [f"feature_{idx}" for idx in range(transformed.shape[1])]
        explainer = shap.TreeExplainer(model.named_steps["model"])
        shap_values = explainer.shap_values(transformed)
        importance = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        importance.to_csv(out_dir / "shap_importance.csv", index=False)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False, max_display=20)
        fig = plt.gcf()
        fig.suptitle(f"Hybrid Model 1 - {target} SHAP Summary", fontsize=16, y=0.98)
        plt.tight_layout()
        fig.savefig(out_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close("all")
        artifacts["shap_importance_csv"] = str(out_dir / "shap_importance.csv")
        artifacts["shap_summary_png"] = str(out_dir / "shap_summary.png")
    except Exception as exc:
        (out_dir / "shap_failed.txt").write_text(str(exc), encoding="utf-8")
        artifacts["shap_error"] = str(out_dir / "shap_failed.txt")
    return artifacts


def run_prediction_diagnostics(model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], target: str, out_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pred = model.predict(test_df[features])
        residual = test_df[target].to_numpy(dtype=float) - pred

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.scatter(test_df[target], pred, s=12, alpha=0.45, color="#2f6f73")
        lo = float(min(test_df[target].min(), pred.min()))
        hi = float(max(test_df[target].max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{target} Prediction Error", fontsize=14, fontweight="bold")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.25)
        fig.savefig(out_dir / "yb_prediction_error.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.scatter(pred, residual, s=12, alpha=0.45, color="#5b5f97")
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{target} Residuals", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual - predicted")
        ax.grid(alpha=0.25)
        fig.savefig(out_dir / "yb_residuals.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        artifacts["prediction_error_png"] = str(out_dir / "yb_prediction_error.png")
        artifacts["residuals_png"] = str(out_dir / "yb_residuals.png")
    except Exception as exc:
        (out_dir / "prediction_diagnostics_failed.txt").write_text(str(exc), encoding="utf-8")
        artifacts["prediction_diagnostics_error"] = str(out_dir / "prediction_diagnostics_failed.txt")
    return artifacts


def run_evidently_report(model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], target: str, out_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, RegressionPreset
        from evidently.report import Report

        ref = train_df[features + [target]].copy()
        cur = test_df[features + [target]].copy()
        ref["prediction"] = model.predict(train_df[features])
        cur["prediction"] = model.predict(test_df[features])
        report_features = [feature for feature in features if not ref[feature].isna().all()]
        numeric_features = [feature for feature in report_features if feature not in base.CATEGORICAL_FEATURES]
        categorical_features = [feature for feature in report_features if feature in base.CATEGORICAL_FEATURES]
        ref = ref[report_features + [target, "prediction"]]
        cur = cur[report_features + [target, "prediction"]]
        mapping = ColumnMapping(
            target=target,
            prediction="prediction",
            numerical_features=numeric_features,
            categorical_features=categorical_features,
        )
        report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
        report.run(reference_data=ref, current_data=cur, column_mapping=mapping)
        report.save_html(str(out_dir / "evidently_report.html"))
        artifacts["evidently_report_html"] = str(out_dir / "evidently_report.html")
    except Exception as exc:
        (out_dir / "evidently_failed.txt").write_text(str(exc), encoding="utf-8")
        artifacts["evidently_error"] = str(out_dir / "evidently_failed.txt")
    return artifacts


def clean_feature_name(feature: str) -> str:
    name = str(feature)
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for categorical in base.CATEGORICAL_FEATURES:
        if name.startswith(f"{categorical}_"):
            return categorical
    return name


def create_combined_shap_image(variant: str) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for target in TARGETS:
        shap_path = OUTPUT_ROOT / variant / target / "shap_importance.csv"
        if not shap_path.exists():
            continue
        shap_df = pd.read_csv(shap_path).head(12).copy()
        shap_df["target"] = target
        shap_df["clean_feature"] = shap_df["feature"].map(clean_feature_name)
        rows.append(shap_df)
    if not rows:
        return None

    shap_all = pd.concat(rows, ignore_index=True)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18, 8), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        top = shap_all[shap_all["target"] == target].head(12).iloc[::-1].copy()
        if top.empty:
            ax.set_axis_off()
            continue
        ax.barh(top["clean_feature"], top["mean_abs_shap"], color="#2f6f73")
        ax.set_title(target.replace("pred_", "").upper(), fontsize=14, fontweight="bold")
        ax.set_xlabel("mean |SHAP|")
        ax.grid(axis="x", alpha=0.25)
    output = OUTPUT_ROOT / variant / "combined_shap_top_features.png"
    fig.suptitle(f"Hybrid Model 1 SHAP - {variant}", fontsize=18, fontweight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def run_diagnostics_only() -> dict[str, Any]:
    hybrid_df = load_or_build_hybrid_dataset()
    train_df, _, _, test_df = temporal_split(hybrid_df)
    diagnostics: dict[str, Any] = {}
    for variant in ["with_existing_features", "physical_no_teacher_summary"]:
        numeric_features, categorical_features = variant_features(variant)
        features = numeric_features + categorical_features
        diagnostics[variant] = {}
        for target in TARGETS:
            model = load_model(variant, target)
            out_dir = OUTPUT_ROOT / variant / target
            out_dir.mkdir(parents=True, exist_ok=True)
            artifacts: dict[str, str] = {}
            artifacts.update(run_shap_diagnostics(model, test_df, features, target, out_dir))
            artifacts.update(run_prediction_diagnostics(model, train_df, test_df, features, target, out_dir))
            artifacts.update(run_evidently_report(model, train_df, test_df, features, target, out_dir))
            diagnostics[variant][target] = artifacts
        diagnostics[variant]["combined_shap_image"] = create_combined_shap_image(variant)

    save_json({"output_root": str(OUTPUT_ROOT), "diagnostics": diagnostics}, DIAGNOSTICS_JSON)
    if SUMMARY_JSON.exists():
        summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        summary["diagnostics_json"] = str(DIAGNOSTICS_JSON)
        summary["diagnostics"] = diagnostics
        save_json(summary, SUMMARY_JSON)
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or diagnose the Model 1 hybrid-target experiment.")
    parser.add_argument("--diagnostics-only", action="store_true", help="Generate SHAP, prediction, and Evidently artifacts from existing hybrid models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.diagnostics_only:
        diagnostics = run_diagnostics_only()
        print(json.dumps({"diagnostics_json": str(DIAGNOSTICS_JSON), "diagnostics": diagnostics}, indent=2, default=str))
        return

    hybrid_df = build_hybrid_dataset()
    train_df, part3_df, valid_df, test_df = temporal_split(hybrid_df)

    variants = ["with_existing_features", "physical_no_teacher_summary"]
    metric_rows: list[dict[str, Any]] = []
    planning_rows: list[dict[str, Any]] = []
    model_paths: dict[str, dict[str, str]] = {}

    for variant in variants:
        numeric_features, categorical_features = variant_features(variant)
        features = numeric_features + categorical_features
        variant_root = OUTPUT_ROOT / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        model_paths[variant] = {}

        split_frames = {
            "train_part1_part2": train_df,
            "valid_first60_part3": valid_df,
            "test_last40_part3": test_df,
            "test_dt_only_last40_part3": test_df[test_df["label_source"] == "DT"],
            "test_baseline_only_last40_part3": test_df[test_df["label_source"] == "baseline"],
        }

        prediction_cache: dict[str, pd.DataFrame] = {name: frame.copy() for name, frame in split_frames.items()}
        for target in TARGETS:
            model = fit_model(train_df, target, numeric_features, categorical_features)
            model_path = variant_root / f"{target}.joblib"
            joblib.dump(model, model_path)
            model_paths[variant][target] = str(model_path)

            for split_name, frame in split_frames.items():
                if frame.empty:
                    continue
                pred = model.predict(frame[features])
                metric_rows.append(
                    {
                        "variant": variant,
                        "target": target,
                        "split": split_name,
                        **compute_metrics(frame[target], pred),
                        "rows": int(len(frame)),
                        "dt_rows": int((frame["label_source"] == "DT").sum()),
                        "baseline_rows": int((frame["label_source"] == "baseline").sum()),
                        "actual_mean": float(frame[target].mean()),
                        "pred_mean": float(np.mean(pred)),
                    }
                )
                prediction_cache[split_name][f"model_{target}"] = pred

        for split_name in ["valid_first60_part3", "test_last40_part3", "test_dt_only_last40_part3", "test_baseline_only_last40_part3"]:
            frame = prediction_cache[split_name]
            if not frame.empty and all(f"model_{target}" in frame.columns for target in TARGETS):
                planning_rows.append(evaluate_planning(frame, split_name, variant))

    metrics_df = pd.DataFrame(metric_rows)
    planning_df = pd.DataFrame(planning_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    planning_df.to_csv(PLANNING_CSV, index=False)
    plot_paths = create_plots(metrics_df, planning_df)
    diagnostics = run_diagnostics_only()

    label_counts = (
        hybrid_df.groupby(["time_bucket", "label_source"], as_index=False)
        .agg(rows=("grid_id", "size"), unique_grids=("grid_id", "nunique"))
        .to_dict(orient="records")
    )

    summary = {
        "dataset_csv": str(DATASET_CSV),
        "output_root": str(OUTPUT_ROOT),
        "row_counts": {
            "hybrid_dense_rows": int(len(hybrid_df)),
            "train_part1_part2": int(len(train_df)),
            "part3_total": int(len(part3_df)),
            "valid_first60_part3": int(len(valid_df)),
            "test_last40_part3": int(len(test_df)),
            "train_dt_rows": int((train_df["label_source"] == "DT").sum()),
            "train_baseline_rows": int((train_df["label_source"] == "baseline").sum()),
            "test_dt_rows": int((test_df["label_source"] == "DT").sum()),
            "test_baseline_rows": int((test_df["label_source"] == "baseline").sum()),
        },
        "label_counts": label_counts,
        "variants": {
            "with_existing_features": "Coordinate-free existing feature set, including baseline-derived summary/history features.",
            "physical_no_teacher_summary": "Coordinate-free physical/RF/geo features with bucket and direct baseline-summary/history removed.",
        },
        "model_paths": model_paths,
        "metrics_csv": str(METRICS_CSV),
        "planning_csv": str(PLANNING_CSV),
        "plots": plot_paths,
        "diagnostics_json": str(DIAGNOSTICS_JSON),
        "diagnostics": diagnostics,
        "key_metrics": {
            "test_last40_part3": metrics_df[metrics_df["split"] == "test_last40_part3"].to_dict(orient="records"),
            "test_dt_only_last40_part3": metrics_df[metrics_df["split"] == "test_dt_only_last40_part3"].to_dict(orient="records"),
            "planning_test_last40_part3": planning_df[planning_df["split"] == "test_last40_part3"].to_dict(orient="records"),
            "planning_test_dt_only_last40_part3": planning_df[planning_df["split"] == "test_dt_only_last40_part3"].to_dict(orient="records"),
        },
        "interpretation": (
            "Hybrid labels increase training coverage by filling unlabeled grids with baseline values while keeping DT values "
            "where available. Overall metrics must be read alongside DT-only metrics because baseline-filled rows are easier "
            "and can make aggregate scores look better than measured-DT behavior."
        ),
    }
    save_json(summary, SUMMARY_JSON)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
