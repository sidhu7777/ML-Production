"""
Model 1 bucket-sequence ablation retraining.

This script keeps the existing coordinate-free Model 1 artifacts untouched.
It trains two variants into models/model1_retrain_bucket_seq:
    - with_bucket_seq: coordinate-free baseline, keeps bucket_seq
    - remove_bucket_seq: coordinate-free baseline, removes bucket_seq
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


COORDINATE_FREE_METADATA = Path("models") / "model1_retrain" / "remove_coordinates" / "pred_rsrp" / "metadata.json"
DATASET_CSV = Path("data") / "model1_coverage_training.csv"
RETRAIN_ROOT = Path("models") / "model1_retrain_bucket_seq"
SUMMARY_JSON = RETRAIN_ROOT / "bucket_seq_ablation_summary.json"
SUMMARY_CSV = RETRAIN_ROOT / "bucket_seq_ablation_metrics.csv"
SHAP_COMPARE_PNG = RETRAIN_ROOT / "bucket_seq_shap_before_after.png"


def setup_logging() -> logging.Logger:
    RETRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("model1_bucket_seq_ablation")
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


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_coordinate_free_features() -> tuple[list[str], list[str]]:
    if not COORDINATE_FREE_METADATA.exists():
        raise FileNotFoundError(f"Missing coordinate-free metadata: {COORDINATE_FREE_METADATA}")
    metadata = json.loads(COORDINATE_FREE_METADATA.read_text(encoding="utf-8"))
    numeric = list(metadata["features"]["numeric"])
    categorical = list(metadata["features"]["categorical"])
    return numeric, categorical


def variant_features(remove_bucket_seq: bool) -> tuple[list[str], list[str]]:
    numeric, categorical = load_coordinate_free_features()
    if remove_bucket_seq:
        numeric = [feature for feature in numeric if feature != "bucket_seq"]
    return numeric, categorical


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


def load_dataset() -> pd.DataFrame:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing Model 1 dataset: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)
    numeric, categorical = load_coordinate_free_features()
    required = set(numeric + categorical + base.TARGETS + ["time_bucket"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"Model 1 dataset is missing required columns: {missing}")
    for col in numeric + base.TARGETS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    df["time_bucket"] = df["time_bucket"].astype(str)
    log.info("Loaded coordinate-free dataset rows=%d columns=%d", len(df), len(df.columns))
    return df


def shap_bucket_seq_summary(target_dir: Path) -> dict[str, object]:
    shap_path = target_dir / "shap_importance.csv"
    if not shap_path.exists():
        return {"available": False, "bucket_seq_rank": None, "bucket_seq_mean_abs_shap": None}
    shap_df = pd.read_csv(shap_path)
    bucket_rows = shap_df[shap_df["feature"].astype(str).str.contains("bucket_seq", na=False)].copy()
    if bucket_rows.empty:
        return {"available": True, "bucket_seq_rank": None, "bucket_seq_mean_abs_shap": 0.0}
    bucket_rows["rank"] = bucket_rows.index + 1
    return {
        "available": True,
        "bucket_seq_rank": int(bucket_rows["rank"].min()),
        "bucket_seq_mean_abs_shap": float(bucket_rows["mean_abs_shap"].sum()),
        "bucket_seq_rows": bucket_rows[["feature", "mean_abs_shap", "rank"]].to_dict(orient="records"),
    }


def load_shap_frame(target_dir: Path) -> pd.DataFrame:
    shap_path = target_dir / "shap_importance.csv"
    if not shap_path.exists():
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    return pd.read_csv(shap_path)


def create_before_after_shap_image() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before_root = RETRAIN_ROOT / "with_bucket_seq"
    after_root = RETRAIN_ROOT / "remove_bucket_seq"

    family_colors = {
        "RF": "#2f6f73",
        "Geo": "#7a5c22",
        "Temporal": "#5b5f97",
        "Other": "#777777",
    }

    def _family_for_feature(name: str) -> str:
        clean = str(name)
        for prefix in ("num__", "cat__"):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        if clean == "bucket_seq" or clean.startswith("prev"):
            return "Temporal"
        if clean in {"clutter_class", "dominant_band_class"}:
            return "RF"
        if clean in {
            "morphology_cluster",
            "terrain_elevation_m",
            "terrain_slope_deg",
            "grid_size_m",
            "grid_area_m2",
            "road_length_m",
            "road_density",
            "green_ratio",
            "water_ratio",
            "building_count",
            "building_area_ratio",
            "avg_building_area_m2",
            "park_open_area",
            "open_area_ratio",
            "mall_presence",
            "metro_presence",
            "clutter_transition_flag",
            "clutter_upgrade_score",
        }:
            return "Geo"
        return "RF"

    fig, axes = plt.subplots(len(base.TARGETS), 2, figsize=(18, 18), constrained_layout=True)
    panel_titles = ["Before: with bucket_seq", "After: bucket_seq removed"]

    for row_idx, target in enumerate(base.TARGETS):
        before_df = load_shap_frame(before_root / target).head(12).iloc[::-1].copy()
        after_df = load_shap_frame(after_root / target).head(12).iloc[::-1].copy()
        for col_idx, (variant_df, variant_title) in enumerate(zip([before_df, after_df], panel_titles)):
            ax = axes[row_idx, col_idx] if len(base.TARGETS) > 1 else axes[col_idx]
            if variant_df.empty:
                ax.set_axis_off()
                continue
            labels = variant_df["feature"].astype(str).tolist()
            colors = [family_colors.get(_family_for_feature(feature), family_colors["Other"]) for feature in labels]
            ax.barh(labels, variant_df["mean_abs_shap"], color=colors)
            ax.set_title(f"{target.replace('pred_', '').upper()} - {variant_title}", fontsize=13, fontweight="bold")
            ax.set_xlabel("mean |SHAP|")
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="x", alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=family_colors["RF"]),
        plt.Rectangle((0, 0), 1, 1, color=family_colors["Geo"]),
        plt.Rectangle((0, 0), 1, 1, color=family_colors["Temporal"]),
    ]
    fig.legend(handles, ["RF", "Geo", "Temporal"], loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Model 1 Bucket-Seq Ablation: SHAP Before vs After", fontsize=18, fontweight="bold")
    fig.savefig(SHAP_COMPARE_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return SHAP_COMPARE_PNG


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
        "bucket_seq_shap": shap_bucket_seq_summary(target_dir),
    }


def read_variant_result(variant_name: str, remove_bucket_seq: bool) -> dict[str, object]:
    numeric_features, categorical_features = variant_features(remove_bucket_seq)
    variant_root = RETRAIN_ROOT / variant_name
    return {
        "variant": variant_name,
        "remove_bucket_seq": remove_bucket_seq,
        "removed_features": ["bucket_seq"] if remove_bucket_seq else [],
        "feature_count": len(numeric_features) + len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "targets": {target: read_target_result(variant_root, target) for target in base.TARGETS},
    }


def train_variant(
    variant_name: str,
    remove_bucket_seq: bool,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    optuna_trials: int,
    optuna_timeout: int,
) -> dict[str, object]:
    numeric_features, categorical_features = variant_features(remove_bucket_seq)
    variant_root = RETRAIN_ROOT / variant_name
    variant_root.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("VARIANT: %s | features=%d | removed=%s", variant_name, len(numeric_features) + len(categorical_features), ["bucket_seq"] if remove_bucket_seq else [])
    log.info("=" * 70)

    result: dict[str, object] = {
        "variant": variant_name,
        "remove_bucket_seq": remove_bucket_seq,
        "removed_features": ["bucket_seq"] if remove_bucket_seq else [],
        "feature_count": len(numeric_features) + len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "targets": {},
    }

    with patched_base_context(variant_root, numeric_features, categorical_features, optuna_trials, optuna_timeout):
        for target in base.TARGETS:
            log.info("TARGET: %s", target)
            base.train_target(train_df, valid_df, test_df, target)
            result["targets"][target] = read_target_result(variant_root, target)

    return result


def metric_delta(with_value: object, without_value: object) -> float | None:
    try:
        return float(without_value) - float(with_value)
    except (TypeError, ValueError):
        return None


def build_comparison(with_bucket_seq: dict[str, object], without_bucket_seq: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    comparison: dict[str, object] = {}
    with_targets = with_bucket_seq.get("targets", {})
    without_targets = without_bucket_seq.get("targets", {})

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
                        "with_bucket_seq": with_value,
                        "remove_bucket_seq": without_value,
                        "delta_remove_minus_with": delta,
                    }
                )
                target_summary[f"{split}_{metric}_delta_remove_minus_with"] = delta

        with_shap = (with_targets.get(target, {}) or {}).get("bucket_seq_shap", {})
        without_shap = (without_targets.get(target, {}) or {}).get("bucket_seq_shap", {})
        target_summary["with_bucket_seq_rank"] = with_shap.get("bucket_seq_rank")
        target_summary["with_bucket_seq_mean_abs_shap"] = with_shap.get("bucket_seq_mean_abs_shap")
        target_summary["remove_bucket_seq_rank"] = without_shap.get("bucket_seq_rank")
        target_summary["remove_bucket_seq_mean_abs_shap"] = without_shap.get("bucket_seq_mean_abs_shap")
        comparison[target] = target_summary

    return comparison, pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain coordinate-free Model 1 with/without bucket_seq.")
    parser.add_argument("--trials", type=int, default=base.OPTUNA_TRIALS, help="Optuna trials per target and variant")
    parser.add_argument("--timeout", type=int, default=base.OPTUNA_TIMEOUT, help="Optuna timeout seconds per target and variant")
    parser.add_argument("--only-target", choices=base.TARGETS, help="Train only one target for a focused check")
    parser.add_argument("--summarize-only", action="store_true", help="Regenerate comparison files from existing artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_targets = list(base.TARGETS)
    if args.only_target:
        base.TARGETS = [args.only_target]

    try:
        log.info("Model 1 bucket_seq ablation started at %s", datetime.utcnow().isoformat() + "Z")
        log.info("Output root: %s", RETRAIN_ROOT.resolve())

        if args.summarize_only:
            with_bucket_seq = read_variant_result("with_bucket_seq", False)
            without_bucket_seq = read_variant_result("remove_bucket_seq", True)
        else:
            df = load_dataset()
            train_df = df[df["time_bucket"].isin(["PART_1", "PART_2"])].copy()
            part3 = (
                df[df["time_bucket"] == "PART_3"]
                .sort_values(["bucket_mid_timestamp", "bucket_max_timestamp", "grid_row", "grid_col"])
                .reset_index(drop=True)
                .copy()
            )
            split_idx = int(len(part3) * 0.60)
            valid_df = part3.iloc[:split_idx].copy()
            test_df = part3.iloc[split_idx:].copy()

            with_bucket_seq = train_variant(
                "with_bucket_seq",
                False,
                train_df,
                valid_df,
                test_df,
                args.trials,
                args.timeout,
            )
            without_bucket_seq = train_variant(
                "remove_bucket_seq",
                True,
                train_df,
                valid_df,
                test_df,
                args.trials,
                args.timeout,
            )

        comparison, comparison_df = build_comparison(with_bucket_seq, without_bucket_seq)
        comparison_df.to_csv(SUMMARY_CSV, index=False)

        summary = {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "dataset_csv": str(DATASET_CSV),
            "output_root": str(RETRAIN_ROOT),
            "optuna_trials": int(args.trials),
            "optuna_timeout": int(args.timeout),
            "removed_feature": "bucket_seq",
            "variants": {
                "with_bucket_seq": with_bucket_seq,
                "remove_bucket_seq": without_bucket_seq,
            },
            "comparison": comparison,
            "comparison_csv": str(SUMMARY_CSV),
            "shap_compare_png": str(create_before_after_shap_image()),
            "interpretation_rule": "For MAE/RMSE, positive delta means removing bucket_seq is worse. For R2, negative delta means removing bucket_seq is worse.",
        }
        save_json(summary, SUMMARY_JSON)
        log.info("Saved ablation summary to %s", SUMMARY_JSON)
        log.info("Saved ablation metric table to %s", SUMMARY_CSV)
    finally:
        base.TARGETS = original_targets


if __name__ == "__main__":
    main()
