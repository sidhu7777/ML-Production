from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


TARGETS = ["pred_rsrp", "pred_sinr", "pred_rsrq"]


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def model_features(run_dir: Path, target: str) -> List[str]:
    meta = load_json(run_dir / target / "metadata.json")
    return list(meta["numeric_features"]) + list(meta["categorical_features"])


def predict(run_dir: Path, df: pd.DataFrame, target: str) -> np.ndarray:
    model = joblib.load(run_dir / target / f"{target}.joblib")
    features = model_features(run_dir, target)
    return model.predict(df[features])


def base_p3_frame(df: pd.DataFrame) -> pd.DataFrame:
    p1 = df[df["time_bucket"].eq("PART_1")].set_index("grid_id")
    p3 = df[df["time_bucket"].eq("PART_3")].set_index("grid_id")
    common = p1.index.intersection(p3.index)
    p3 = p3.loc[common].copy()
    for target in TARGETS:
        p3[f"actual_delta_{target}"] = pd.to_numeric(p3[target], errors="coerce") - pd.to_numeric(p1.loc[common, target], errors="coerce")
    p3["serving_changed"] = p1.loc[common, "serving_cell_key"].astype(str).ne(p3["serving_cell_key"].astype(str)).to_numpy()
    p3["band_changed"] = pd.to_numeric(p1.loc[common, "serving_band"], errors="coerce").ne(
        pd.to_numeric(p3["serving_band"], errors="coerce")
    ).to_numpy()
    p3["clutter_changed"] = p1.loc[common, "clutter_class"].astype(str).ne(p3["clutter_class"].astype(str)).to_numpy()
    p3["geo_zone"] = pd.to_numeric(p3.get("geo_impact_zone_score", 0), errors="coerce").fillna(0.0).gt(0.01)
    p3["interference_high"] = pd.to_numeric(p3.get("neighbor_interference_index", 0), errors="coerce").fillna(0.0).gt(1.0)
    p3["stable_low_change"] = ~(p3["serving_changed"] | p3["band_changed"] | p3["geo_zone"])
    return p3.reset_index()


def metrics_for(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"rows": 0, "mae": None, "r2": None, "residual_mean": None}
    try:
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    except Exception:
        r2 = None
    return {
        "rows": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
        "residual_mean": float(np.mean(y_true.to_numpy(dtype=float) - y_pred)),
        "actual_mean": float(pd.to_numeric(y_true, errors="coerce").mean()),
        "pred_mean": float(np.mean(y_pred)),
    }


def direction_label(delta: pd.Series, stable_threshold: float) -> pd.Series:
    values = pd.to_numeric(delta, errors="coerce").fillna(0.0)
    return pd.Series(
        np.select(
            [values > stable_threshold, values < -stable_threshold],
            ["improve", "degrade"],
            default="stable",
        ),
        index=delta.index,
    )


def direction_metrics(actual_delta: pd.Series, pred_delta: pd.Series, stable_threshold: float) -> Dict[str, object]:
    actual = direction_label(actual_delta, stable_threshold)
    pred = direction_label(pred_delta, stable_threshold)
    labels = ["improve", "degrade", "stable"]
    result: Dict[str, object] = {
        "rows": int(len(actual)),
        "accuracy": float(actual.eq(pred).mean()) if len(actual) else None,
        "counts_actual": {label: int(actual.eq(label).sum()) for label in labels},
        "counts_pred": {label: int(pred.eq(label).sum()) for label in labels},
    }
    for label in labels:
        tp = int(actual.eq(label).mul(pred.eq(label)).sum())
        actual_count = int(actual.eq(label).sum())
        pred_count = int(pred.eq(label).sum())
        result[f"{label}_recall"] = float(tp / actual_count) if actual_count else None
        result[f"{label}_precision"] = float(tp / pred_count) if pred_count else None
    return result


def segment_masks(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "all_part3": pd.Series(True, index=frame.index),
        "stable_low_change": frame["stable_low_change"].astype(bool),
        "geo_impact_zone": frame["geo_zone"].astype(bool),
        "clutter_changed": frame["clutter_changed"].astype(bool),
        "serving_changed": frame["serving_changed"].astype(bool),
        "band_changed": frame["band_changed"].astype(bool),
        "interference_high": frame["interference_high"].astype(bool),
        "geo_zone_without_serving_change": frame["geo_zone"].astype(bool) & ~frame["serving_changed"].astype(bool),
        "serving_changed_without_geo_zone": frame["serving_changed"].astype(bool) & ~frame["geo_zone"].astype(bool),
    }


def permutation_drop_check(run_dir: Path, frame: pd.DataFrame, target: str, feature_groups: Dict[str, List[str]]) -> Dict[str, object]:
    features = model_features(run_dir, target)
    model = joblib.load(run_dir / target / f"{target}.joblib")
    base_pred = model.predict(frame[features])
    base_mae = mean_absolute_error(frame[target], base_pred)
    rng = np.random.default_rng(42)
    rows = []
    for group, cols in feature_groups.items():
        present = [col for col in cols if col in features and col in frame.columns]
        if not present:
            continue
        shuffled = frame.copy()
        for col in present:
            values = shuffled[col].to_numpy(copy=True)
            rng.shuffle(values)
            shuffled[col] = values
        pred = model.predict(shuffled[features])
        mae = mean_absolute_error(shuffled[target], pred)
        rows.append({"group": group, "features": present, "mae": float(mae), "mae_increase": float(mae - base_mae)})
    return {"base_mae": float(base_mae), "groups": rows}


def validate(dataset_csv: Path, run_dir: Path, output_json: Path) -> Dict[str, object]:
    df = pd.read_csv(dataset_csv)
    frame = base_p3_frame(df)
    p1_lookup = df[df["time_bucket"].eq("PART_1")].set_index("grid_id")
    result: Dict[str, object] = {
        "dataset_csv": str(dataset_csv),
        "run_dir": str(run_dir),
        "part3_rows": int(len(frame)),
        "targets": {},
    }
    feature_groups = {
        "geo": [
            "clutter_transition_flag",
            "geo_transition_source_score",
            "geo_impact_zone_score",
            "geo_impact_nearby_flag",
            "rf_geo_blockage_penalty_db",
            "building_growth_ratio",
            "road_growth_ratio",
            "building_area_ratio",
            "road_density",
            "clutter_class",
        ],
        "topology": [
            "topology_serving_changed_flag",
            "topology_band_changed_flag",
            "serving_band",
            "serving_earfcn",
            "serving_azimuth",
            "carrier_count",
            "dominant_band_class",
            "low_band_ratio",
            "mid_band_ratio",
            "high_band_ratio",
        ],
        "interference": [
            "neighbor_interference_index",
            "raw_neighbor_interference_index",
            "rf_load_pressure_proxy",
            "neighbor1_band",
        ],
        "rf_context": [
            "mean_candidate_rsrp",
            "mean_candidate_sinr",
            "std_candidate_rsrp",
            "std_candidate_sinr",
            "neighbor1_rsrp",
            "neighbor1_sinr",
            "neighbor2_rsrp",
            "neighbor2_sinr",
            "rsrp_gap_to_neighbor1",
            "sinr_gap_to_neighbor1",
        ],
    }
    thresholds = {"pred_rsrp": 1.0, "pred_sinr": 2.0, "pred_rsrq": 0.3}
    for target in TARGETS:
        pred = predict(run_dir, frame, target)
        frame[f"prediction_{target}"] = pred
        base_values = pd.to_numeric(p1_lookup.loc[frame["grid_id"], target].reset_index(drop=True), errors="coerce")
        pred_delta = pd.Series(pred, index=frame.index) - base_values
        target_result: Dict[str, object] = {
            "overall": metrics_for(frame[target], pred),
            "segments": {},
            "direction": direction_metrics(frame[f"actual_delta_{target}"], pred_delta, thresholds[target]),
            "permutation_drop": permutation_drop_check(run_dir, frame, target, feature_groups),
        }
        for name, mask in segment_masks(frame).items():
            sub = frame.loc[mask]
            sub_pred = pred[mask.to_numpy()]
            target_result["segments"][name] = metrics_for(sub[target], sub_pred)
        result["targets"][target] = target_result
    save_json(result, output_json)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Model 1 25m RF learning on changed-grid segments.")
    parser.add_argument("--dataset-csv", default="tests/coverage_prediction/data/model1_training_25m_rf_dataset.csv")
    parser.add_argument("--run-dir", default="tests/coverage_prediction/weights/model1_25m_rf_balanced_20260803_1510")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output_json) if args.output_json else run_dir / "learning_validation.json"
    result = validate(Path(args.dataset_csv), run_dir, output)
    print(json.dumps(result, indent=2, default=str)[:12000])


if __name__ == "__main__":
    main()
