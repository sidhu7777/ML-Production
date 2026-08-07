from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


TARGETS = ["pred_rsrp", "pred_sinr", "pred_rsrq"]
TARGET_LABELS = {
    "pred_rsrp": "target_delta_rsrp",
    "pred_sinr": "target_delta_sinr",
    "pred_rsrq": "target_delta_rsrq",
}
CURRENT_KPI_COLUMNS = {
    "pred_rsrp": "current_rsrp",
    "pred_sinr": "current_sinr",
    "pred_rsrq": "current_rsrq",
}
EXCLUDED_FEATURES = {
    "time_bucket",
    "current_time_bucket",
    "future_time_bucket",
    "bucket_seq",
    "grid_id",
    "grid_row",
    "grid_col",
    "parent_grid_row_50m",
    "parent_grid_col_50m",
    "centroid_lat",
    "centroid_lon",
    "target_geometry_wkt",
    "bucket_start",
    "bucket_end",
    "geo_snapshot_ts_utc",
    "geo_snapshot_source_ts",
    "serving_cell_key",
    "serving_cell_id",
    "serving_site_id",
    "serving_sector",
    "serving_canonical_sector",
    "neighbor1_cell_key",
    "neighbor1_cell_id",
    "neighbor2_cell_key",
    "neighbor2_cell_id",
    "raw_pred_rsrp",
    "raw_pred_rsrq",
    "raw_pred_sinr",
    "raw_rsrp_delta_db",
    "raw_sinr_delta_db",
    "pred_rsrp_max",
    "_identity_anchor_row",
    "_split_name",
    "_holdout_bucket",
    "future_rsrp",
    "future_sinr",
    "future_rsrq",
    "future_rsrp_observed_from_bucket",
    "future_sinr_observed_from_bucket",
    "future_rsrq_observed_from_bucket",
}
CATEGORICAL_FEATURES = [
    "clutter_class",
    "dominant_band_class",
    "geo_snapshot_mode",
    "geo_assignment_source",
    "geo_assignment_method",
]
CHANGE_SIGNAL_COLUMNS = [
    "clutter_transition_flag",
    "geo_transition_source_score",
    "geo_impact_zone_score",
    "geo_impact_nearby_flag",
    "building_growth_ratio",
    "road_growth_ratio",
    "topology_serving_changed_flag",
    "topology_band_changed_flag",
    "rf_geo_blockage_penalty_db",
]


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true: pd.Series, pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": rmse(y_true.to_numpy(dtype=float), pred),
        "r2": float(r2_score(y_true, pred)),
        "mean_actual": float(pd.to_numeric(y_true, errors="coerce").mean()),
        "mean_prediction": float(np.mean(pred)),
        "residual_mean": float(np.mean(y_true.to_numpy(dtype=float) - pred)),
    }


def load_dataset(dataset_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    if "time_bucket" in df.columns:
        df["time_bucket"] = df["time_bucket"].astype(str)
    if "current_time_bucket" in df.columns:
        df["current_time_bucket"] = df["current_time_bucket"].astype(str)
        df["time_bucket"] = df["current_time_bucket"]
    has_delta_targets = all(col in df.columns for col in TARGET_LABELS.values())
    if not has_delta_targets:
        missing = [target for target in TARGETS if target not in df.columns]
        if missing:
            raise RuntimeError(f"Dataset is missing target columns: {missing}")
        raw_required = ["raw_pred_rsrp", "raw_pred_sinr", "raw_pred_rsrq"]
        missing_raw = [col for col in raw_required if col not in df.columns]
        if missing_raw:
            raise RuntimeError(f"Dataset is missing current KPI anchor columns: {missing_raw}")
        df["current_rsrp"] = pd.to_numeric(df["raw_pred_rsrp"], errors="coerce")
        df["current_sinr"] = pd.to_numeric(df["raw_pred_sinr"], errors="coerce")
        df["current_rsrq"] = pd.to_numeric(df["raw_pred_rsrq"], errors="coerce")
        df["target_delta_rsrp"] = pd.to_numeric(df["pred_rsrp"], errors="coerce") - df["current_rsrp"]
        df["target_delta_sinr"] = pd.to_numeric(df["pred_sinr"], errors="coerce") - df["current_sinr"]
        df["target_delta_rsrq"] = pd.to_numeric(df["pred_rsrq"], errors="coerce") - df["current_rsrq"]
    return df


def add_no_change_anchor_rows(df: pd.DataFrame) -> pd.DataFrame:
    anchors = df.copy()
    for col in CHANGE_SIGNAL_COLUMNS:
        if col in anchors.columns:
            anchors[col] = 0.0
    for col in TARGET_LABELS.values():
        anchors[col] = 0.0
    anchors["geo_snapshot_mode"] = "no_change_anchor"
    anchors["geo_assignment_source"] = "identity_constraint"
    anchors["geo_assignment_method"] = "current_kpi_plus_zero_delta"
    anchors["_identity_anchor_row"] = 1
    out = df.copy()
    out["_identity_anchor_row"] = 0
    return pd.concat([out, anchors], ignore_index=True)


def feature_columns(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    excluded = set(EXCLUDED_FEATURES) | set(TARGETS) | set(TARGET_LABELS.values())
    # Keep current_* KPI anchors and causal change/context features. The model learns KPI deltas,
    # then production adds the predicted delta back to the current Project KPI.
    cols = [col for col in df.columns if col not in excluded]
    categorical = [col for col in CATEGORICAL_FEATURES if col in cols]
    numeric = [col for col in cols if col not in categorical and pd.api.types.is_numeric_dtype(df[col])]
    return numeric, categorical


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    buckets = set(df["time_bucket"].astype(str).unique())
    if "PART_3" in buckets:
        train = df[df["time_bucket"].isin(["PART_1", "PART_2"])].copy()
        holdout = df[df["time_bucket"].eq("PART_3")].copy()
        holdout_name = "PART_3"
    else:
        train = df[df["time_bucket"].eq("PART_1")].copy()
        holdout = df[df["time_bucket"].eq("PART_2")].copy()
        holdout_name = "PART_2"
    sort_cols = [col for col in ["grid_row", "grid_col", "grid_id"] if col in holdout.columns]
    holdout = holdout.sort_values(sort_cols).reset_index(drop=True)
    split_idx = int(len(holdout) * 0.60)
    valid = holdout.iloc[:split_idx].copy()
    test = holdout.iloc[split_idx:].copy()
    for split_df, split_name in [(train, "train"), (valid, "valid"), (test, "test")]:
        split_df["_split_name"] = split_name
        split_df["_holdout_bucket"] = holdout_name
    return train, valid, test


def build_pipeline(numeric: List[str], categorical: List[str]) -> Pipeline:
    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=650,
        max_depth=6,
        learning_rate=0.035,
        subsample=0.90,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_alpha=0.05,
        reg_lambda=1.25,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("prep", prep), ("model", model)])


def transformed_feature_names(pipeline: Pipeline) -> List[str]:
    prep = pipeline.named_steps["prep"]
    try:
        return list(prep.get_feature_names_out())
    except Exception:
        model = pipeline.named_steps["model"]
        return [f"feature_{idx}" for idx in range(model.feature_importances_.shape[0])]


def feature_family(feature_name: str) -> str:
    clean = str(feature_name)
    for prefix in ("num__", "cat__"):
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
    if any(token in clean for token in ["raw_pred", "mean_candidate", "max_candidate", "std_candidate", "rsrp", "sinr", "rsrq"]):
        return "RF"
    if any(token in clean for token in ["neighbor", "interference", "gap", "load_pressure"]):
        return "Interference"
    if any(token in clean for token in ["serving_", "topology", "band", "earfcn", "azimuth", "carrier"]):
        return "Topology"
    if any(token in clean for token in ["clutter", "geo_", "building", "road", "park", "open", "mall", "metro", "green", "water"]):
        return "Geo"
    return "Other"


def save_feature_importance(pipeline: Pipeline, out_dir: Path) -> dict[str, object]:
    names = transformed_feature_names(pipeline)
    values = pipeline.named_steps["model"].feature_importances_
    imp = pd.DataFrame({"feature": names, "importance": values})
    imp["family"] = imp["feature"].map(feature_family)
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(out_dir / "feature_importance.csv", index=False)
    family = imp.groupby("family", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    family.to_csv(out_dir / "feature_family_importance.csv", index=False)
    return {
        "top_features": imp.head(20).to_dict(orient="records"),
        "family_importance": family.to_dict(orient="records"),
    }


def segment_metrics(df: pd.DataFrame, pred: np.ndarray, target: str) -> pd.DataFrame:
    work = df.copy()
    work["prediction"] = pred
    work["abs_error"] = (work[target] - work["prediction"]).abs()
    def num_col(name: str, default: float = 0.0) -> pd.Series:
        if name not in work.columns:
            return pd.Series(default, index=work.index, dtype="float64")
        return pd.to_numeric(work[name], errors="coerce").fillna(default)

    segments = {
        "all": pd.Series(True, index=work.index),
        "geo_impact_zone": num_col("geo_impact_zone_score") > 0.01,
        "clutter_transition": num_col("clutter_transition_flag").eq(1),
        "serving_changed": num_col("topology_serving_changed_flag").eq(1),
        "band_changed": num_col("topology_band_changed_flag").eq(1),
        "interference_high": num_col("neighbor_interference_index") > 1.0,
        "stable_low_change": (
            num_col("geo_impact_zone_score").le(0.01)
            & num_col("topology_serving_changed_flag").eq(0)
            & num_col("topology_band_changed_flag").eq(0)
        ),
    }
    rows = []
    for name, mask in segments.items():
        sub = work.loc[mask]
        if sub.empty:
            continue
        rows.append(
            {
                "segment": name,
                "rows": int(len(sub)),
                "mae": float(sub["abs_error"].mean()),
                "mean_actual": float(sub[target].mean()),
                "mean_prediction": float(sub["prediction"].mean()),
                "residual_mean": float((sub[target] - sub["prediction"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def train_target(
    target: str,
    label_col: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    numeric: List[str],
    categorical: List[str],
    run_dir: Path,
    prediction_mode: str,
) -> dict[str, object]:
    out_dir = run_dir / target
    out_dir.mkdir(parents=True, exist_ok=True)
    features = numeric + categorical
    pipeline = build_pipeline(numeric, categorical)
    pipeline.fit(train[features], train[label_col])

    split_metrics = {}
    predictions = {}
    for split_name, split_df in [("train", train), ("valid", valid), ("test", test)]:
        pred = pipeline.predict(split_df[features])
        predictions[split_name] = pred
        split_metrics[split_name] = metrics(split_df[label_col], pred)

    joblib.dump(pipeline, out_dir / f"{target}.joblib")
    save_json(split_metrics, out_dir / "metrics.json")
    importance = save_feature_importance(pipeline, out_dir)

    seg_df = test.copy()
    seg_df[target] = seg_df[label_col]
    seg = segment_metrics(seg_df, predictions["test"], target)
    seg.to_csv(out_dir / "test_segment_metrics.csv", index=False)

    metadata = {
        "target": target,
        "label_column": label_col,
        "prediction_mode": prediction_mode,
        "current_kpi_column": CURRENT_KPI_COLUMNS[target],
        "future_kpi_formula": f"{target} = {CURRENT_KPI_COLUMNS[target]} + predicted_delta",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "dataset": "model1_training_25m_rf_dataset.csv",
        "split": {
            "train": "PART_1 + PART_2",
            "valid": "first 60% of PART_3 sorted by grid_row/grid_col",
            "test": "last 40% of PART_3 sorted by grid_row/grid_col",
        },
        "excluded_features": sorted(EXCLUDED_FEATURES | set(TARGETS) | set(TARGET_LABELS.values())),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "metrics": split_metrics,
        "importance_summary": importance,
        "segment_metrics": seg.to_dict(orient="records"),
    }
    save_json(metadata, out_dir / "metadata.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model 1 25m RF XGBoost models without touching old weights.")
    parser.add_argument("--dataset-csv", default="tests/coverage_prediction/data/model1_training_25m_rf_dataset.csv")
    parser.add_argument("--weights-dir", default="tests/coverage_prediction/weights")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--no-identity-anchors", action="store_true")
    parser.add_argument("--prediction-mode", default="current_state_future_delta")
    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv)
    weights_root = Path(args.weights_dir)
    run_name = args.run_name.strip() or f"model1_25m_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = weights_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    df = load_dataset(dataset_csv)
    original_rows = int(len(df))
    if not args.no_identity_anchors:
        df = add_no_change_anchor_rows(df)
    numeric, categorical = feature_columns(df)
    train, valid, test = temporal_split(df)

    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "dataset_csv": str(dataset_csv),
        "rows": int(len(df)),
        "original_rows": original_rows,
        "identity_anchor_rows": int(len(df) - original_rows),
        "unique_grids": int(df["grid_id"].nunique()),
        "split_rows": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
        "targets": {},
        "numeric_features": numeric,
        "categorical_features": categorical,
    }

    for target in TARGETS:
        label_col = TARGET_LABELS[target]
        print(f"Training {target} as {label_col}...")
        summary["targets"][target] = train_target(target, label_col, train, valid, test, numeric, categorical, run_dir, args.prediction_mode)

    save_json(summary, run_dir / "training_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"targets", "numeric_features"}}, indent=2, default=str))


if __name__ == "__main__":
    main()
