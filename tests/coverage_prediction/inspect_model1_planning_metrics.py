"""
Model 1 planning-oriented inspection for the coordinate-free retrain.

This test case keeps the existing Model 1 artifacts untouched and adds
planning-specific evaluation in a new output folder:
    - hole detection / weak-coverage detection
    - worst-grid ranking overlap
    - residual bias by severity
    - combined report that answers the production review questions
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
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))


DATASET_CSV = Path("data") / "model1_coverage_training.csv"
MODEL_ROOT = Path("models") / "model1_retrain" / "remove_coordinates"
INSPECTION_ROOT = Path("models") / "model1_inspection"
OUTPUT_ROOT = Path("models") / "model1_planning_inspection"
TARGETS = ["pred_rsrp", "pred_rsrq", "pred_sinr"]

HOLE_THRESHOLDS = {"pred_rsrp": -100.0, "pred_rsrq": -13.0, "pred_sinr": 0.0}
WEAK_THRESHOLDS = {"pred_rsrp": -95.0, "pred_rsrq": -10.0, "pred_sinr": 3.0}
SEVERITY_SCALES = {"pred_rsrp": 20.0, "pred_rsrq": 5.0, "pred_sinr": 6.0}

COORDINATE_FEATURES = {"grid_centroid_lat", "grid_centroid_lon"}

RF_FEATURES = {
    "serving_distance_m",
    "nearest_site_distance_m",
    "site_count_250m",
    "site_count_500m",
    "azimuth_delta_deg",
    "interference_gap_db",
    "interference_ratio_linear",
    "los_blocked_ratio",
    "nlos_flag",
    "bandwidth_mhz_est",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "dominant_band_class",
    "carrier_count",
    "pred_rsrp_min",
    "pred_rsrp_max",
    "pred_rsrp_std",
    "pred_sinr_std",
    "measurement_count",
    "unique_cells",
    "unique_sites",
}
GEO_FEATURES = {
    "clutter_class",
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
}
TEMPORAL_FEATURE_PREFIXES = ("prev_obs_", "prev2_obs_", "prev_trend_")
TEMPORAL_FEATURES = {"bucket_seq"}


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_features() -> tuple[list[str], list[str], list[str]]:
    metadata_path = MODEL_ROOT / "pred_rsrp" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing coordinate-free metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    numeric = list(metadata["features"]["numeric"])
    categorical = list(metadata["features"]["categorical"])
    features = numeric + categorical
    unexpected_coords = sorted(COORDINATE_FEATURES.intersection(features))
    if unexpected_coords:
        raise RuntimeError(f"Coordinate-free metadata still contains coordinates: {unexpected_coords}")
    return numeric, categorical, features


def feature_family(feature: str) -> str:
    raw = clean_feature_name(feature)
    if raw in RF_FEATURES:
        return "RF"
    if raw in GEO_FEATURES:
        return "Geo"
    if raw in TEMPORAL_FEATURES or raw.startswith(TEMPORAL_FEATURE_PREFIXES):
        return "Temporal"
    if raw in COORDINATE_FEATURES:
        return "Coordinate"
    return "Other"


def clean_feature_name(feature: str) -> str:
    name = str(feature)
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for categorical in ("clutter_class", "dominant_band_class"):
        if name.startswith(f"{categorical}_"):
            return categorical
    return name


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return train_df, part3, valid_df, test_df


def load_dataset() -> pd.DataFrame:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing Model 1 dataset: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)
    numeric, categorical, _ = load_features()
    required = set(numeric + categorical + TARGETS + ["time_bucket"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"Model 1 dataset is missing required columns: {missing}")
    for col in numeric + TARGETS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    df["time_bucket"] = df["time_bucket"].astype(str)
    return df


def load_existing_inspection_summary() -> dict[str, Any]:
    path = INSPECTION_ROOT / "inspection_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_shap_summary() -> pd.DataFrame:
    path = INSPECTION_ROOT / "coordinate_free_shap_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=["target", "rank", "feature", "clean_feature", "family", "mean_abs_shap"])
    return pd.read_csv(path)


def planning_gap(value: pd.Series, threshold: float, scale: float) -> pd.Series:
    return np.maximum(0.0, (threshold - value.astype(float)) / scale)


def row_hole_label(frame: pd.DataFrame, suffix: str) -> pd.Series:
    return (
        (frame[f"pred_rsrp_{suffix}"] <= HOLE_THRESHOLDS["pred_rsrp"])
        | (frame[f"pred_rsrq_{suffix}"] <= HOLE_THRESHOLDS["pred_rsrq"])
        | (frame[f"pred_sinr_{suffix}"] <= HOLE_THRESHOLDS["pred_sinr"])
    )


def row_weak_label(frame: pd.DataFrame, suffix: str) -> pd.Series:
    return (
        (frame[f"pred_rsrp_{suffix}"] <= WEAK_THRESHOLDS["pred_rsrp"])
        | (frame[f"pred_rsrq_{suffix}"] <= WEAK_THRESHOLDS["pred_rsrq"])
        | (frame[f"pred_sinr_{suffix}"] <= WEAK_THRESHOLDS["pred_sinr"])
    )


def composite_severity(frame: pd.DataFrame, suffix: str) -> pd.Series:
    return (
        planning_gap(frame[f"pred_rsrp_{suffix}"], WEAK_THRESHOLDS["pred_rsrp"], SEVERITY_SCALES["pred_rsrp"])
        + planning_gap(frame[f"pred_rsrq_{suffix}"], WEAK_THRESHOLDS["pred_rsrq"], SEVERITY_SCALES["pred_rsrq"])
        + planning_gap(frame[f"pred_sinr_{suffix}"], WEAK_THRESHOLDS["pred_sinr"], SEVERITY_SCALES["pred_sinr"])
    )


def compute_binary_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual = actual.astype(bool)
    predicted = predicted.astype(bool)
    tp = int(((actual == True) & (predicted == True)).sum())
    fp = int(((actual == False) & (predicted == True)).sum())
    fn = int(((actual == True) & (predicted == False)).sum())
    tn = int(((actual == False) & (predicted == False)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def load_predictions(test_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = test_df[["grid_id", "grid_row", "grid_col", "time_bucket", "bucket_seq"]].copy()
    for target in TARGETS:
        model = joblib.load(MODEL_ROOT / target / f"{target}.joblib")
        pred = model.predict(test_df[features])
        out[f"{target}_actual"] = pd.to_numeric(test_df[target], errors="coerce")
        out[f"{target}_pred"] = pd.to_numeric(pd.Series(pred, index=test_df.index), errors="coerce")
        out[f"{target}_error"] = out[f"{target}_pred"] - out[f"{target}_actual"]
        out[f"{target}_abs_error"] = out[f"{target}_error"].abs()
        out[f"{target}_hole_actual"] = out[f"{target}_actual"] <= HOLE_THRESHOLDS[target]
        out[f"{target}_hole_pred"] = out[f"{target}_pred"] <= HOLE_THRESHOLDS[target]
        out[f"{target}_weak_actual"] = out[f"{target}_actual"] <= WEAK_THRESHOLDS[target]
        out[f"{target}_weak_pred"] = out[f"{target}_pred"] <= WEAK_THRESHOLDS[target]
        out[f"{target}_severity_actual"] = planning_gap(out[f"{target}_actual"], WEAK_THRESHOLDS[target], SEVERITY_SCALES[target])
        out[f"{target}_severity_pred"] = planning_gap(out[f"{target}_pred"], WEAK_THRESHOLDS[target], SEVERITY_SCALES[target])

    out["actual_hole_any"] = row_hole_label(out, "actual")
    out["pred_hole_any"] = row_hole_label(out, "pred")
    out["actual_weak_any"] = row_weak_label(out, "actual")
    out["pred_weak_any"] = row_weak_label(out, "pred")
    out["composite_severity_actual"] = composite_severity(out, "actual")
    out["composite_severity_pred"] = composite_severity(out, "pred")
    out["composite_error"] = out["composite_severity_pred"] - out["composite_severity_actual"]
    return out


def compute_target_metrics(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    planning_rows = []
    bias_rows = []

    for target in TARGETS:
        actual = pred_df[f"{target}_actual"]
        pred = pred_df[f"{target}_pred"]
        hole_actual = pred_df[f"{target}_hole_actual"]
        hole_pred = pred_df[f"{target}_hole_pred"]
        weak_actual = pred_df[f"{target}_weak_actual"]
        weak_pred = pred_df[f"{target}_weak_pred"]
        severity_actual = pred_df[f"{target}_severity_actual"]
        severity_pred = pred_df[f"{target}_severity_pred"]

        metric_rows.append(
            {
                "target": target,
                "mae": float(mean_absolute_error(actual, pred)),
                "rmse": rmse(actual, pred.to_numpy()),
                "r2": float(r2_score(actual, pred)),
                "residual_mean": float((pred - actual).mean()),
                "residual_std": float((pred - actual).std(ddof=0)),
                "spearman_actual_pred": float(spearmanr(actual, pred).correlation),
                "actual_hole_rate": float(hole_actual.mean()),
                "pred_hole_rate": float(hole_pred.mean()),
                "actual_weak_rate": float(weak_actual.mean()),
                "pred_weak_rate": float(weak_pred.mean()),
                "hole_metrics": compute_binary_metrics(hole_actual, hole_pred),
                "weak_metrics": compute_binary_metrics(weak_actual, weak_pred),
                "mean_severity_actual": float(severity_actual.mean()),
                "mean_severity_pred": float(severity_pred.mean()),
                "mean_abs_severity_gap": float((severity_pred - severity_actual).abs().mean()),
            }
        )

        target_df = pred_df[["grid_id", "grid_row", "grid_col"]].copy()
        target_df["target"] = target
        target_df["actual"] = actual
        target_df["predicted"] = pred
        target_df["residual"] = pred - actual
        target_df["actual_severity"] = severity_actual
        target_df["predicted_severity"] = severity_pred

        grid_df = (
            target_df.groupby("grid_id", as_index=False)
            .agg(
                grid_row=("grid_row", "first"),
                grid_col=("grid_col", "first"),
                row_count=("grid_id", "size"),
                actual_severity=("actual_severity", "mean"),
                predicted_severity=("predicted_severity", "mean"),
                mean_residual=("residual", "mean"),
                mae=("residual", lambda s: float(np.mean(np.abs(s)))),
            )
            .sort_values(["actual_severity", "row_count"], ascending=[False, False])
        )

        actual_top = grid_df.nlargest(20, "actual_severity")["grid_id"].astype(str).tolist()
        predicted_top = grid_df.nlargest(20, "predicted_severity")["grid_id"].astype(str).tolist()
        overlap = len(set(actual_top).intersection(predicted_top))
        planning_rows.append(
            {
                "target": target,
                "grid_count": int(len(grid_df)),
                "top20_actual_ids": actual_top,
                "top20_predicted_ids": predicted_top,
                "top20_overlap_count": int(overlap),
                "top20_overlap_ratio": float(overlap / 20.0),
                "spearman_grid_severity": float(spearmanr(grid_df["actual_severity"], grid_df["predicted_severity"]).correlation),
                "worst_actual_grid_id": str(grid_df.iloc[0]["grid_id"]) if not grid_df.empty else None,
                "worst_pred_grid_id": str(grid_df.sort_values("predicted_severity", ascending=False).iloc[0]["grid_id"]) if not grid_df.empty else None,
                "worst_actual_grid_severity": float(grid_df.iloc[0]["actual_severity"]) if not grid_df.empty else None,
                "worst_pred_grid_severity": float(grid_df.sort_values("predicted_severity", ascending=False).iloc[0]["predicted_severity"])
                if not grid_df.empty
                else None,
            }
        )

        target_df["severity_bin"] = pd.qcut(target_df["actual_severity"].rank(method="first"), 4, labels=False, duplicates="drop")
        for bin_id, group in target_df.groupby("severity_bin", dropna=True):
            bias_rows.append(
                {
                    "target": target,
                    "severity_bin": int(bin_id),
                    "rows": int(len(group)),
                    "actual_severity_mean": float(group["actual_severity"].mean()),
                    "predicted_severity_mean": float(group["predicted_severity"].mean()),
                    "residual_mean": float(group["residual"].mean()),
                    "residual_abs_mean": float(group["residual"].abs().mean()),
                }
            )

    overall = pred_df.copy()
    metric_rows.append(
        {
            "target": "composite",
            "mae": float(mean_absolute_error(overall["composite_severity_actual"], overall["composite_severity_pred"])),
            "rmse": rmse(overall["composite_severity_actual"], overall["composite_severity_pred"].to_numpy()),
            "r2": float(r2_score(overall["composite_severity_actual"], overall["composite_severity_pred"])),
            "residual_mean": float(overall["composite_error"].mean()),
            "residual_std": float(overall["composite_error"].std(ddof=0)),
            "spearman_actual_pred": float(spearmanr(overall["composite_severity_actual"], overall["composite_severity_pred"]).correlation),
            "actual_hole_rate": float(overall["actual_hole_any"].mean()),
            "pred_hole_rate": float(overall["pred_hole_any"].mean()),
            "actual_weak_rate": float(overall["actual_weak_any"].mean()),
            "pred_weak_rate": float(overall["pred_weak_any"].mean()),
            "hole_metrics": compute_binary_metrics(overall["actual_hole_any"], overall["pred_hole_any"]),
            "weak_metrics": compute_binary_metrics(overall["actual_weak_any"], overall["pred_weak_any"]),
            "mean_severity_actual": float(overall["composite_severity_actual"].mean()),
            "mean_severity_pred": float(overall["composite_severity_pred"].mean()),
            "mean_abs_severity_gap": float((overall["composite_severity_pred"] - overall["composite_severity_actual"]).abs().mean()),
        }
    )

    overall_grid = (
        overall.groupby("grid_id", as_index=False)
        .agg(
            grid_row=("grid_row", "first"),
            grid_col=("grid_col", "first"),
            row_count=("grid_id", "size"),
            actual_severity=("composite_severity_actual", "mean"),
            predicted_severity=("composite_severity_pred", "mean"),
            mean_residual=("composite_error", "mean"),
            mae=("composite_error", lambda s: float(np.mean(np.abs(s)))),
        )
        .sort_values(["actual_severity", "row_count"], ascending=[False, False])
    )
    actual_top = overall_grid.nlargest(20, "actual_severity")["grid_id"].astype(str).tolist()
    predicted_top = overall_grid.nlargest(20, "predicted_severity")["grid_id"].astype(str).tolist()
    overlap = len(set(actual_top).intersection(predicted_top))
    planning_rows.append(
        {
            "target": "composite",
            "grid_count": int(len(overall_grid)),
            "top20_actual_ids": actual_top,
            "top20_predicted_ids": predicted_top,
            "top20_overlap_count": int(overlap),
            "top20_overlap_ratio": float(overlap / 20.0),
            "spearman_grid_severity": float(spearmanr(overall_grid["actual_severity"], overall_grid["predicted_severity"]).correlation),
            "worst_actual_grid_id": str(overall_grid.iloc[0]["grid_id"]) if not overall_grid.empty else None,
            "worst_pred_grid_id": str(overall_grid.sort_values("predicted_severity", ascending=False).iloc[0]["grid_id"]) if not overall_grid.empty else None,
            "worst_actual_grid_severity": float(overall_grid.iloc[0]["actual_severity"]) if not overall_grid.empty else None,
            "worst_pred_grid_severity": float(overall_grid.sort_values("predicted_severity", ascending=False).iloc[0]["predicted_severity"])
            if not overall_grid.empty
            else None,
        }
    )

    overall_df = overall[["composite_severity_actual", "composite_severity_pred", "composite_error"]].copy()
    overall_df["severity_bin"] = pd.qcut(overall_df["composite_severity_actual"].rank(method="first"), 4, labels=False, duplicates="drop")
    for bin_id, group in overall_df.groupby("severity_bin", dropna=True):
        bias_rows.append(
            {
                "target": "composite",
                "severity_bin": int(bin_id),
                "rows": int(len(group)),
                "actual_severity_mean": float(group["composite_severity_actual"].mean()),
                "predicted_severity_mean": float(group["composite_severity_pred"].mean()),
                "residual_mean": float(group["composite_error"].mean()),
                "residual_abs_mean": float(group["composite_error"].abs().mean()),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    planning_df = pd.DataFrame(planning_rows)
    bias_df = pd.DataFrame(bias_rows)
    return metrics_df, planning_df, bias_df


def plot_confusion_panels(pred_df: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = OUTPUT_ROOT / "planning_hole_confusion.png"
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    panels = TARGETS + ["composite"]

    for ax, target in zip(axes.flat, panels):
        if target == "composite":
            actual = pred_df["actual_hole_any"]
            predicted = pred_df["pred_hole_any"]
            title = "COMPOSITE"
        else:
            actual = pred_df[f"{target}_hole_actual"]
            predicted = pred_df[f"{target}_hole_pred"]
            title = target.replace("pred_", "").upper()

        tp = int(((actual == True) & (predicted == True)).sum())
        fp = int(((actual == False) & (predicted == True)).sum())
        fn = int(((actual == True) & (predicted == False)).sum())
        tn = int(((actual == False) & (predicted == False)).sum())
        mat = np.array([[tn, fp], [fn, tp]], dtype=float)
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Pred safe", "Pred hole"])
        ax.set_yticks([0, 1], labels=["Actual safe", "Actual hole"])
        ax.set_title(
            f"{title}\nPrecision={tp / (tp + fp + 1e-9):.2f} | Recall={tp / (tp + fn + 1e-9):.2f} | F1={2 * tp / (2 * tp + fp + fn + 1e-9):.2f}",
            fontsize=12,
            fontweight="bold",
        )
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="black", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Model 1 Planning Inspection: Coverage-Hole Confusion", fontsize=18, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_severity_scatter(planning_df: pd.DataFrame, pred_df: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = OUTPUT_ROOT / "planning_grid_severity_scatter.png"
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    panels = TARGETS + ["composite"]

    for ax, target in zip(axes.flat, panels):
        if target == "composite":
            grid_df = (
                pred_df.groupby("grid_id", as_index=False)
                .agg(actual=("composite_severity_actual", "mean"), predicted=("composite_severity_pred", "mean"))
            )
            title = "COMPOSITE"
        else:
            grid_df = (
                pred_df.groupby("grid_id", as_index=False)
                .agg(actual=(f"{target}_severity_actual", "mean"), predicted=(f"{target}_severity_pred", "mean"))
            )
            title = target.replace("pred_", "").upper()

        if grid_df.empty:
            ax.set_axis_off()
            continue
        ax.scatter(grid_df["actual"], grid_df["predicted"], s=10, alpha=0.35, color="#2f6f73")
        lo = float(min(grid_df["actual"].min(), grid_df["predicted"].min()))
        hi = float(max(grid_df["actual"].max(), grid_df["predicted"].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#a23b72", linewidth=1.5)
        corr = spearmanr(grid_df["actual"], grid_df["predicted"]).correlation
        ax.set_title(f"{title}\nSpearman={corr:.2f} | grids={len(grid_df)}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual severity")
        ax.set_ylabel("Predicted severity")
        ax.grid(alpha=0.25)

    fig.suptitle("Model 1 Planning Inspection: Actual vs Predicted Grid Severity", fontsize=18, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_residual_bias(pred_df: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = OUTPUT_ROOT / "planning_residual_bias.png"
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    panels = TARGETS + ["composite"]

    for ax, target in zip(axes.flat, panels):
        if target == "composite":
            work = pred_df[["composite_severity_actual", "composite_error"]].copy()
            work = work.rename(columns={"composite_severity_actual": "severity", "composite_error": "residual"})
            title = "COMPOSITE"
        else:
            work = pred_df[[f"{target}_severity_actual", f"{target}_error"]].copy()
            work = work.rename(columns={f"{target}_severity_actual": "severity", f"{target}_error": "residual"})
            title = target.replace("pred_", "").upper()

        if work.empty:
            ax.set_axis_off()
            continue
        try:
            work["bin"] = pd.qcut(work["severity"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except Exception:
            work["bin"] = "All"
        summary = work.groupby("bin", dropna=True, observed=False).agg(
            residual_mean=("residual", "mean"),
            residual_abs_mean=("residual", lambda s: float(np.mean(np.abs(s)))),
            rows=("residual", "size"),
        )
        summary = summary.reset_index()
        ax.bar(summary["bin"].astype(str), summary["residual_mean"], color="#5b5f97", alpha=0.85)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{title} residual mean by severity bin", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean residual")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Model 1 Planning Inspection: Residual Bias by Severity", fontsize=18, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_report(
    summary: dict[str, Any],
    planning_metrics_df: pd.DataFrame,
    planning_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_drift_path: Path,
    target_drift_path: Path,
    shap_path: Path,
    interaction_path: Path,
    metric_paths: dict[str, Path],
) -> None:
    lines: list[str] = [
        "# Model 1 Planning Inspection",
        "",
        "This run evaluates the coordinate-free Model 1 as a planning tool, not just a regression model.",
        "",
        "## Data Quality",
        "",
    ]

    data_quality = summary.get("data_quality", {})
    lines.extend(
        [
            f"- rows: `{data_quality.get('rows')}`",
            f"- columns: `{data_quality.get('columns')}`",
            f"- missing cells: `{data_quality.get('missing_cells')}`",
            f"- duplicate rows: `{data_quality.get('duplicate_rows')}`",
            f"- PART_3 test rows: `{summary.get('row_counts', {}).get('test_last_40pct_part3')}`",
        ]
    )

    lines.extend(["", "## Drift", ""])
    drifted = summary.get("inspection_summary", {}).get("drifted_feature_count")
    lines.append(f"- material feature drift count from the existing coordinate-free inspection: `{drifted}`")
    lines.append(f"- feature drift file: `{feature_drift_path}`")
    lines.append(f"- target drift file: `{target_drift_path}`")

    lines.extend(["", "## SHAP", ""])
    top_shap = summary.get("top_shap_by_target", {})
    for target in TARGETS:
        rows = top_shap.get(target, [])[:8]
        formatted = ", ".join(f"{row['clean_feature']} ({row['family']}, {row['mean_abs_shap']:.3f})" for row in rows) if rows else "n/a"
        lines.append(f"- {target}: {formatted}")

    lines.extend(["", "## Planning Metrics", ""])
    for row in planning_metrics_df.itertuples(index=False):
        planning_row = planning_df[planning_df["target"] == row.target].iloc[0]
        lines.append(
            f"- {row.target}: hole precision `{row.hole_metrics['precision']:.3f}`, recall `{row.hole_metrics['recall']:.3f}`, "
            f"F1 `{row.hole_metrics['f1']:.3f}`, weak-coverage recall `{row.weak_metrics['recall']:.3f}`, "
            f"grid Spearman `{planning_row.spearman_grid_severity:.3f}`, top20 overlap `{planning_row.top20_overlap_count}/20`"
        )

    lines.extend(["", "## Largest Errors", ""])
    for target in TARGETS + ["composite"]:
        subset = pred_df if target == "composite" else pred_df.assign(**{"residual": pred_df[f"{target}_error"]})
        if target == "composite":
            subset = pred_df.assign(residual=pred_df["composite_error"])
        else:
            subset = pred_df.assign(residual=pred_df[f"{target}_error"])
        top = (
            subset.groupby("grid_id", as_index=False)
            .agg(
                residual_mean=("residual", "mean"),
                abs_residual=("residual", lambda s: float(np.mean(np.abs(s)))),
            )
            .sort_values("abs_residual", ascending=False)
            .head(5)
        )
        lines.append(f"- {target}:")
        for row in top.itertuples(index=False):
            lines.append(f"  - grid `{row.grid_id}` abs residual `{row.abs_residual:.3f}` mean residual `{row.residual_mean:.3f}`")

    lines.extend(["", "## Residual Bias", ""])
    for row in bias_df.itertuples(index=False):
        lines.append(
            f"- {row.target} severity bin `{row.severity_bin}`: residual mean `{row.residual_mean:.3f}`, abs mean `{row.residual_abs_mean:.3f}`"
        )

    lines.extend(["", "## Likely Root Cause", ""])
    lines.extend(
        [
            "- The model is not merely predicting unusual KPI values; it is also being asked to generalize across PART_3 conditions that drift from training.",
            "- Existing inspection shows strong target drift for `pred_rsrp` and `pred_sinr`, plus feature drift in RF and temporal features.",
            "- Interaction coverage analysis found many PART_3 feature combinations are unseen in training, so the issue is partly data support, not just model choice.",
            "- The current SHAP profile also shows shortcut-like dependencies on high-information derived features and `bucket_seq` in some variants.",
        ]
    )

    lines.extend(["", "## What To Investigate Next", ""])
    lines.extend(
        [
            "- Worst grids with the largest planning severity gaps.",
            "- Whether new RF descriptors like path-loss proxies or better interference summaries explain those grids.",
            "- Whether `bucket_seq` or derived summary features should remain in production if they do not transfer across projects.",
            "- Whether the planner should use a composite weak-coverage score in addition to regression metrics.",
        ]
    )

    lines.extend(["", "## Artifacts", ""])
    lines.extend(
        [
            f"- planning confusion plot: `{metric_paths['confusion_png']}`",
            f"- severity scatter plot: `{metric_paths['severity_png']}`",
            f"- residual bias plot: `{metric_paths['bias_png']}`",
            f"- planning metrics CSV: `{metric_paths['planning_metrics_csv']}`",
            f"- grid ranking CSV: `{metric_paths['planning_grid_csv']}`",
            f"- residual bias CSV: `{metric_paths['planning_bias_csv']}`",
            f"- summary JSON: `{metric_paths['summary_json']}`",
            f"- reference inspection summary: `{INSPECTION_ROOT / 'inspection_summary.json'}`",
            f"- reference interaction coverage summary: `{interaction_path}`",
        ]
    )

    (OUTPUT_ROOT / "planning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    numeric_features, categorical_features, features = load_features()
    df = load_dataset()
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    train_df, part3_df, valid_df, test_df = temporal_split(df)

    pred_df = load_predictions(test_df, features)
    planning_metrics_df, planning_df, bias_df = compute_target_metrics(pred_df)

    inspection_summary = load_existing_inspection_summary()
    shap_df = load_existing_shap_summary()
    interaction_summary_path = Path("models") / "model1_interaction_coverage" / "interaction_coverage_summary.json"
    interaction_summary = json.loads(interaction_summary_path.read_text(encoding="utf-8")) if interaction_summary_path.exists() else {}

    confusion_png = plot_confusion_panels(pred_df)
    severity_png = plot_severity_scatter(planning_df, pred_df)
    bias_png = plot_residual_bias(pred_df)

    # A compact combined image of the top SHAP features from the existing inspection.
    combined_shap_img = INSPECTION_ROOT / "combined_shap_top_features.png"

    metric_paths = {
        "confusion_png": confusion_png,
        "severity_png": severity_png,
        "bias_png": bias_png,
        "planning_metrics_csv": OUTPUT_ROOT / "planning_metrics.csv",
        "planning_grid_csv": OUTPUT_ROOT / "planning_grid_ranking.csv",
        "planning_bias_csv": OUTPUT_ROOT / "planning_residual_bias.csv",
        "summary_json": OUTPUT_ROOT / "planning_inspection_summary.json",
    }

    planning_metrics_df.to_csv(metric_paths["planning_metrics_csv"], index=False)
    planning_df.to_csv(metric_paths["planning_grid_csv"], index=False)
    bias_df.to_csv(metric_paths["planning_bias_csv"], index=False)

    top_shap = {
        target: shap_df[shap_df["target"] == target].head(10).to_dict(orient="records") for target in TARGETS
    }

    summary = {
        "dataset_csv": str(DATASET_CSV),
        "model_root": str(MODEL_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "row_counts": {
            "train_part1_part2": int(len(train_df)),
            "part3_total": int(len(part3_df)),
            "valid_first_60pct_part3": int(len(valid_df)),
            "test_last_40pct_part3": int(len(test_df)),
        },
        "data_quality": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
        },
        "inspection_summary": {
            "drifted_feature_count": inspection_summary.get("drifted_feature_count"),
            "drifted_features_by_family": inspection_summary.get("drifted_features_by_family"),
            "target_drift": inspection_summary.get("target_drift"),
            "test_metrics": inspection_summary.get("test_metrics"),
        },
        "feature_drift_csv": str(INSPECTION_ROOT / "feature_drift_train_vs_part3.csv"),
        "target_drift_csv": str(INSPECTION_ROOT / "target_drift_train_vs_part3.csv"),
        "shap_summary_csv": str(INSPECTION_ROOT / "coordinate_free_shap_summary.csv"),
        "combined_shap_image": str(combined_shap_img),
        "planning_artifacts": {
            "planning_metrics_csv": str(metric_paths["planning_metrics_csv"]),
            "planning_grid_csv": str(metric_paths["planning_grid_csv"]),
            "planning_bias_csv": str(metric_paths["planning_bias_csv"]),
            "confusion_png": str(confusion_png),
            "severity_png": str(severity_png),
            "bias_png": str(bias_png),
            "report": str(OUTPUT_ROOT / "planning_report.md"),
        },
        "top_shap_by_target": top_shap,
        "interaction_coverage_summary": {
            "path": str(interaction_summary_path),
            "train_unique_combinations": interaction_summary.get("train_unique_combinations"),
            "part3_unique_combinations": interaction_summary.get("part3_unique_combinations"),
            "shared_combinations": interaction_summary.get("shared_combinations"),
            "unseen_part3_row_fraction": interaction_summary.get("unseen_part3_row_fraction"),
        },
        "planning_metrics": planning_metrics_df.to_dict(orient="records"),
    }

    save_json(summary, metric_paths["summary_json"])
    build_report(
        summary,
        planning_metrics_df,
        planning_df,
        bias_df,
        pred_df,
        Path("models") / "model1_inspection" / "feature_drift_train_vs_part3.csv",
        Path("models") / "model1_inspection" / "target_drift_train_vs_part3.csv",
        Path("models") / "model1_inspection" / "coordinate_free_shap_summary.csv",
        interaction_summary_path,
        metric_paths,
    )

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
