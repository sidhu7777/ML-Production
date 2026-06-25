"""
Model 1 production inspection for the coordinate-free retrain.

Inputs:
    - data/model1_coverage_training.csv
    - models/model1_retrain/remove_coordinates/*

Outputs:
    - models/model1_inspection/*
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
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))


DATASET_CSV = Path("data") / "model1_coverage_training.csv"
MODEL_ROOT = Path("models") / "model1_retrain" / "remove_coordinates"
OUTPUT_ROOT = Path("models") / "model1_inspection"
TARGETS = ["pred_rsrp", "pred_rsrq", "pred_sinr"]
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


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float | None:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return None
    edges = np.unique(np.nanquantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        low = min(float(ref.min()), float(cur.min()))
        high = max(float(ref.max()), float(cur.max()))
        if high <= low:
            return 0.0
        edges = np.linspace(low, high, bins + 1)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    ref_pct = pd.cut(ref, edges, include_lowest=True).value_counts(sort=False).to_numpy(dtype=float) / len(ref)
    cur_pct = pd.cut(cur, edges, include_lowest=True).value_counts(sort=False).to_numpy(dtype=float) / len(cur)
    ref_pct = np.clip(ref_pct, 1e-6, 1.0)
    cur_pct = np.clip(cur_pct, 1e-6, 1.0)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_for_numeric(feature: str, train_df: pd.DataFrame, part3_df: pd.DataFrame) -> dict[str, Any]:
    train = pd.to_numeric(train_df[feature], errors="coerce").dropna()
    part3 = pd.to_numeric(part3_df[feature], errors="coerce").dropna()
    if train.empty or part3.empty:
        return {"feature": feature, "type": "numeric", "drifted": False, "reason": "empty"}
    ks_stat, p_value = ks_2samp(train, part3)
    psi = population_stability_index(train, part3)
    train_std = float(train.std(ddof=0)) or 1.0
    mean_shift_std = (float(part3.mean()) - float(train.mean())) / train_std
    drifted = bool((p_value < 0.01 and ks_stat >= 0.10) or (psi is not None and psi >= 0.20))
    return {
        "feature": feature,
        "family": feature_family(feature),
        "type": "numeric",
        "train_mean": float(train.mean()),
        "part3_mean": float(part3.mean()),
        "mean_shift": float(part3.mean() - train.mean()),
        "mean_shift_train_std": float(mean_shift_std),
        "train_median": float(train.median()),
        "part3_median": float(part3.median()),
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "psi": psi,
        "drifted": drifted,
    }


def drift_for_categorical(feature: str, train_df: pd.DataFrame, part3_df: pd.DataFrame) -> dict[str, Any]:
    train = train_df[feature].fillna("Unknown").astype(str)
    part3 = part3_df[feature].fillna("Unknown").astype(str)
    categories = sorted(set(train).union(set(part3)))
    if not categories:
        return {"feature": feature, "type": "categorical", "drifted": False, "reason": "empty"}
    table = np.array([[(train == cat).sum() for cat in categories], [(part3 == cat).sum() for cat in categories]])
    chi2, p_value, _, _ = chi2_contingency(table)
    total = float(table.sum())
    cramer_v = math.sqrt(float(chi2) / total) if total > 0 else 0.0
    train_dist = table[0] / max(table[0].sum(), 1)
    part3_dist = table[1] / max(table[1].sum(), 1)
    total_variation = float(np.abs(part3_dist - train_dist).sum() / 2.0)
    drifted = bool(p_value < 0.01 and (cramer_v >= 0.10 or total_variation >= 0.10))
    return {
        "feature": feature,
        "family": feature_family(feature),
        "type": "categorical",
        "categories": categories,
        "cramer_v": float(cramer_v),
        "p_value": float(p_value),
        "total_variation": total_variation,
        "drifted": drifted,
    }


def compute_feature_drift(
    train_df: pd.DataFrame,
    part3_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    rows = [drift_for_numeric(feature, train_df, part3_df) for feature in numeric_features]
    rows.extend(drift_for_categorical(feature, train_df, part3_df) for feature in categorical_features)
    drift_df = pd.DataFrame(rows)
    drift_df["sort_score"] = drift_df["psi"].fillna(0) + drift_df["ks_stat"].fillna(0) + drift_df["cramer_v"].fillna(0)
    return drift_df.sort_values(["drifted", "sort_score"], ascending=[False, False]).drop(columns=["sort_score"])


def compute_target_drift(train_df: pd.DataFrame, part3_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        row = drift_for_numeric(target, train_df, part3_df)
        row["target"] = target
        rows.append(row)
    return pd.DataFrame(rows).drop(columns=["feature", "family", "type"], errors="ignore")


def load_shap_summary() -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        shap_path = MODEL_ROOT / target / "shap_importance.csv"
        shap_df = pd.read_csv(shap_path)
        shap_df["target"] = target
        shap_df["rank"] = np.arange(1, len(shap_df) + 1)
        shap_df["clean_feature"] = shap_df["feature"].map(clean_feature_name)
        shap_df["family"] = shap_df["clean_feature"].map(feature_family)
        rows.append(shap_df)
    out = pd.concat(rows, ignore_index=True)
    return out[["target", "rank", "feature", "clean_feature", "family", "mean_abs_shap"]]


def create_combined_shap_image(shap_df: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = OUTPUT_ROOT / "combined_shap_top_features.png"
    family_colors = {
        "RF": "#2f6f73",
        "Geo": "#7a5c22",
        "Temporal": "#5b5f97",
        "Coordinate": "#9d2f2f",
        "Other": "#777777",
    }

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18, 8), sharex=False)
    for ax, target in zip(axes, TARGETS):
        top = shap_df[shap_df["target"] == target].head(12).iloc[::-1].copy()
        labels = top["clean_feature"].astype(str).tolist()
        colors = [family_colors.get(family, family_colors["Other"]) for family in top["family"]]
        ax.barh(labels, top["mean_abs_shap"], color=colors)
        ax.set_title(target.replace("pred_", "").upper(), fontsize=14, fontweight="bold")
        ax.set_xlabel("mean |SHAP|")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for family, color in family_colors.items()
        if family != "Coordinate"
    ]
    labels = [family for family in family_colors if family != "Coordinate"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Model 1 Coordinate-Free SHAP: Top Features by Target", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def predict_and_analyze_errors(df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    grid_rows = []
    for target in TARGETS:
        model_path = MODEL_ROOT / target / f"{target}.joblib"
        model = joblib.load(model_path)
        pred = model.predict(test_df[features])
        work = test_df[
            [
                "grid_id",
                "grid_row",
                "grid_col",
                "clutter_class",
                "dominant_band_class",
                "serving_distance_m",
                "nearest_site_distance_m",
                "interference_gap_db",
                "los_blocked_ratio",
                "nlos_flag",
                "clutter_transition_flag",
                "bucket_seq",
                target,
            ]
        ].copy()
        work["target"] = target
        work["actual"] = work[target]
        work["prediction"] = pred
        work["error"] = work["prediction"] - work["actual"]
        work["abs_error"] = work["error"].abs()
        work["squared_error"] = work["error"] ** 2
        prediction_rows.append(work.drop(columns=[target]))

        grid_summary = (
            work.groupby("grid_id", as_index=False)
            .agg(
                grid_row=("grid_row", "first"),
                grid_col=("grid_col", "first"),
                clutter_class=("clutter_class", "first"),
                dominant_band_class=("dominant_band_class", "first"),
                row_count=("abs_error", "size"),
                mae=("abs_error", "mean"),
                rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
                residual_mean=("error", "mean"),
                serving_distance_m=("serving_distance_m", "mean"),
                nearest_site_distance_m=("nearest_site_distance_m", "mean"),
                interference_gap_db=("interference_gap_db", "mean"),
                los_blocked_ratio=("los_blocked_ratio", "mean"),
                nlos_flag=("nlos_flag", "mean"),
                clutter_transition_flag=("clutter_transition_flag", "mean"),
            )
            .sort_values(["mae", "row_count"], ascending=[False, False])
        )
        grid_summary["target"] = target
        grid_rows.append(grid_summary)

    errors_df = pd.concat(prediction_rows, ignore_index=True)
    grid_errors_df = pd.concat(grid_rows, ignore_index=True)
    return errors_df, grid_errors_df


def metric_summary(errors_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in errors_df.groupby("target", sort=False):
        rows.append(
            {
                "target": target,
                "test_mae": float(mean_absolute_error(group["actual"], group["prediction"])),
                "test_rmse": rmse(group["actual"], group["prediction"].to_numpy()),
                "test_r2": float(r2_score(group["actual"], group["prediction"])),
                "residual_mean": float(group["error"].mean()),
                "residual_std": float(group["error"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def build_histograms(train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in train_df.columns or feature not in valid_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(train_df[feature]):
            train = pd.to_numeric(train_df[feature], errors="coerce").dropna()
            valid = pd.to_numeric(valid_df[feature], errors="coerce").dropna()
            if train.empty or valid.empty:
                continue
            edges = np.unique(np.nanquantile(train, np.linspace(0, 1, 11)))
            if len(edges) < 3:
                low = min(float(train.min()), float(valid.min()))
                high = max(float(train.max()), float(valid.max()))
                if high <= low:
                    continue
                edges = np.linspace(low, high, 11)
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            train_counts = pd.cut(train, edges, include_lowest=True).value_counts(sort=False)
            valid_counts = pd.cut(valid, edges, include_lowest=True).value_counts(sort=False)
            for interval, train_count, valid_count in zip(train_counts.index, train_counts.to_numpy(), valid_counts.to_numpy()):
                rows.append(
                    {
                        "feature": feature,
                        "bin": str(interval),
                        "train_pct": float(train_count / len(train)),
                        "valid_pct": float(valid_count / len(valid)),
                    }
                )
        else:
            train = train_df[feature].fillna("Unknown").astype(str)
            valid = valid_df[feature].fillna("Unknown").astype(str)
            for value in sorted(set(train).union(set(valid))):
                rows.append(
                    {
                        "feature": feature,
                        "bin": value,
                        "train_pct": float((train == value).mean()),
                        "valid_pct": float((valid == value).mean()),
                    }
                )
    return pd.DataFrame(rows)


def summarize_family_counts(drift_df: pd.DataFrame) -> dict[str, list[str]]:
    drifted = drift_df[drift_df["drifted"] == True].copy()
    return {
        family: drifted.loc[drifted["family"] == family, "feature"].astype(str).tolist()
        for family in ["RF", "Geo", "Temporal", "Other"]
    }


def format_metric(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_report(
    drift_df: pd.DataFrame,
    target_drift_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    grid_errors_df: pd.DataFrame,
    important_hist_features: list[str],
) -> None:
    drifted = drift_df[drift_df["drifted"] == True].copy()
    family_counts = summarize_family_counts(drift_df)
    lines = [
        "# Model 1 Coordinate-Free Inspection",
        "",
        "Input model: `models/model1_retrain/remove_coordinates`.",
        "Coordinates checked: `grid_centroid_lat` and `grid_centroid_lon` are not present in the model feature list.",
        "",
        "## 1. Drifted Features",
        "",
        f"Drifted feature count: `{len(drifted)}` using KS/PSI for numeric features and chi-square/Cramer's V for categorical features.",
    ]
    for family in ["RF", "Geo", "Temporal", "Other"]:
        values = family_counts.get(family, [])
        lines.append(f"- {family}: {', '.join(values) if values else 'none'}")

    lines.extend(["", "## 2. Target Drift", ""])
    for _, row in target_drift_df.iterrows():
        lines.append(
            f"- {row['target']}: train mean `{format_metric(row['train_mean'])}`, "
            f"PART_3 mean `{format_metric(row['part3_mean'])}`, "
            f"shift `{format_metric(row['mean_shift'])}`, PSI `{format_metric(row['psi'])}`, "
            f"drifted `{bool(row['drifted'])}`"
        )

    lines.extend(["", "## 3. Coordinate-Free SHAP", ""])
    for target in TARGETS:
        top = shap_df[shap_df["target"] == target].head(10)
        top_names = [f"{row.clean_feature} ({row.family}, {row.mean_abs_shap:.3f})" for row in top.itertuples()]
        lines.append(f"- {target}: {', '.join(top_names)}")

    lines.extend(["", "## 4. Test Errors", ""])
    for _, row in metrics_df.iterrows():
        lines.append(
            f"- {row['target']}: MAE `{format_metric(row['test_mae'])}`, RMSE `{format_metric(row['test_rmse'])}`, "
            f"R2 `{format_metric(row['test_r2'])}`, residual mean `{format_metric(row['residual_mean'])}`"
        )

    lines.extend(["", "Largest-error grid patterns:"])
    top_errors = grid_errors_df.groupby("target", group_keys=False).head(5)
    for _, row in top_errors.iterrows():
        lines.append(
            f"- {row['target']} grid `{int(row['grid_id'])}`: MAE `{format_metric(row['mae'])}`, "
            f"clutter `{row['clutter_class']}`, band `{row['dominant_band_class']}`, "
            f"serving distance `{format_metric(row['serving_distance_m'])}`, "
            f"LOS blocked `{format_metric(row['los_blocked_ratio'])}`, "
            f"transition `{format_metric(row['clutter_transition_flag'])}`"
        )

    lines.extend(
        [
            "",
            "## 5. Train Vs Validation Histograms",
            "",
            f"Histogram comparison files were written for: `{', '.join(important_hist_features)}`.",
            "",
            "## 6. Decision",
            "",
            "Do not switch algorithms or add broad new features yet. The coordinate-free model already relies on RF/geo/temporal features, but the PART_3 target distributions and several input distributions drift. The next useful improvement should be evidence-led: inspect the largest-error grids and add only missing RF descriptors that explain those failures, such as better path-loss/effective-distance/interference descriptors if the failed grids consistently show those patterns.",
        ]
    )
    (OUTPUT_ROOT / "inspection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    numeric_features, categorical_features, features = load_features()
    df = pd.read_csv(DATASET_CSV)
    for col in numeric_features + TARGETS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    df["time_bucket"] = df["time_bucket"].astype(str)

    train_df, part3_df, valid_df, test_df = temporal_split(df)
    drift_df = compute_feature_drift(train_df, part3_df, numeric_features, categorical_features)
    target_drift_df = compute_target_drift(train_df, part3_df)
    shap_df = load_shap_summary()
    combined_shap_path = create_combined_shap_image(shap_df)
    errors_df, grid_errors_df = predict_and_analyze_errors(df, test_df, features)
    metrics_df = metric_summary(errors_df)

    important_hist_features = (
        shap_df.groupby("clean_feature", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)
        .head(12)["clean_feature"]
        .tolist()
    )
    histogram_df = build_histograms(train_df, valid_df, important_hist_features)

    drift_df.to_csv(OUTPUT_ROOT / "feature_drift_train_vs_part3.csv", index=False)
    target_drift_df.to_csv(OUTPUT_ROOT / "target_drift_train_vs_part3.csv", index=False)
    shap_df.to_csv(OUTPUT_ROOT / "coordinate_free_shap_summary.csv", index=False)
    errors_df.to_csv(OUTPUT_ROOT / "test_prediction_errors.csv", index=False)
    grid_errors_df.to_csv(OUTPUT_ROOT / "top_error_grids_by_target.csv", index=False)
    metrics_df.to_csv(OUTPUT_ROOT / "test_metric_summary.csv", index=False)
    histogram_df.to_csv(OUTPUT_ROOT / "important_feature_histograms_train_vs_valid.csv", index=False)

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
        "coordinate_features_present": sorted(COORDINATE_FEATURES.intersection(features)),
        "drifted_feature_count": int(drift_df["drifted"].sum()),
        "drifted_features_by_family": summarize_family_counts(drift_df),
        "target_drift": target_drift_df.to_dict(orient="records"),
        "test_metrics": metrics_df.to_dict(orient="records"),
        "top_shap_by_target": {
            target: shap_df[shap_df["target"] == target].head(10).to_dict(orient="records")
            for target in TARGETS
        },
        "artifacts": {
            "feature_drift": str(OUTPUT_ROOT / "feature_drift_train_vs_part3.csv"),
            "target_drift": str(OUTPUT_ROOT / "target_drift_train_vs_part3.csv"),
            "shap_summary": str(OUTPUT_ROOT / "coordinate_free_shap_summary.csv"),
            "combined_shap_image": str(combined_shap_path),
            "prediction_errors": str(OUTPUT_ROOT / "test_prediction_errors.csv"),
            "top_error_grids": str(OUTPUT_ROOT / "top_error_grids_by_target.csv"),
            "histograms": str(OUTPUT_ROOT / "important_feature_histograms_train_vs_valid.csv"),
            "report": str(OUTPUT_ROOT / "inspection_report.md"),
        },
    }
    save_json(summary, OUTPUT_ROOT / "inspection_summary.json")
    write_report(drift_df, target_drift_df, shap_df, metrics_df, grid_errors_df, important_hist_features)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
