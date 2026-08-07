from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


TARGETS = ["pred_rsrp", "pred_sinr", "pred_rsrq"]
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


def _classify_band(freq_value: object) -> str:
    freq = pd.to_numeric(pd.Series([freq_value]), errors="coerce").iloc[0]
    if pd.isna(freq) or freq <= 0:
        return "UNKNOWN"
    if freq <= 1000:
        return "LOW_BAND"
    if freq <= 2100:
        return "MID_BAND"
    return "HIGH_BAND"


def _load_features(run_dir: Path, target: str) -> tuple[List[str], List[str]]:
    meta = json.loads((run_dir / target / "metadata.json").read_text(encoding="utf-8"))
    return list(meta["numeric_features"]), list(meta["categorical_features"])


def _preferred_metric(df: pd.DataFrame, smooth_col: str, raw_col: str) -> pd.Series:
    if smooth_col in df.columns:
        smooth = pd.to_numeric(df[smooth_col], errors="coerce")
        return smooth.where(smooth.notna(), pd.to_numeric(df[raw_col], errors="coerce"))
    return pd.to_numeric(df[raw_col], errors="coerce")


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype("float64")


def _robust_norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index, dtype="float64")
    lo = float(valid.quantile(0.05))
    hi = float(valid.quantile(0.95))
    if hi <= lo:
        return pd.Series(0.0, index=series.index, dtype="float64")
    return ((values.fillna(lo) - lo) / (hi - lo)).clip(0.0, 1.0)


def add_current_state_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    clutter = out.get("clutter_class", pd.Series("", index=out.index)).astype(str).str.lower()
    clutter_risk = pd.Series(0.35, index=out.index, dtype="float64")
    clutter_risk = clutter_risk.where(~clutter.str.contains("dense"), 1.0)
    clutter_risk = clutter_risk.where(~(clutter.str.contains("urban") & ~clutter.str.contains("sub")), 0.72)
    clutter_risk = clutter_risk.where(~clutter.str.contains("sub"), 0.45)
    clutter_risk = clutter_risk.where(~clutter.str.contains("rural|open"), 0.18)
    clutter_risk = clutter_risk.where(~clutter.str.contains("water|green|vegetation"), 0.08)

    building_risk = (0.65 * _robust_norm(_num(out, "building_area_ratio")) + 0.35 * _robust_norm(_num(out, "building_count"))).clip(0.0, 1.0)
    road_activity = (0.60 * _robust_norm(_num(out, "road_density")) + 0.40 * _robust_norm(_num(out, "road_length_m"))).clip(0.0, 1.0)
    interference = (
        0.55 * _robust_norm(_num(out, "neighbor_interference_index"))
        + 0.25 * ((3.0 - _num(out, "rsrp_gap_to_neighbor1", 3.0)) / 6.0).clip(0.0, 1.0)
        + 0.20 * ((2.0 - _num(out, "sinr_gap_to_neighbor1", 2.0)) / 8.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    site_pressure = (0.55 * _robust_norm(_num(out, "candidate_cell_count")) + 0.45 * _robust_norm(_num(out, "unique_sites"))).clip(0.0, 1.0)
    growth_pressure = (
        0.30 * clutter_risk
        + 0.26 * building_risk
        + 0.16 * road_activity
        + 0.14 * site_pressure
        + 0.14 * interference
    ).clip(0.0, 1.0)
    out["current_state_growth_pressure"] = growth_pressure.round(6)
    out["current_state_interference_pressure"] = interference.round(6)
    out["current_state_building_pressure"] = building_risk.round(6)
    out["current_state_site_pressure"] = site_pressure.round(6)
    return out


def build_project196_grid_features(baseline_csv: Path) -> pd.DataFrame:
    usecols = [
        "grid_id",
        "Node_Cell_ID",
        "topology_match_id",
        "band",
        "earfcn",
        "sector",
        "site_id",
        "lat",
        "lon",
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "pred_rsrp_smoothed",
        "pred_rsrq_smoothed",
        "pred_sinr_smoothed",
        "clutter_class",
        "morphology_cluster",
        "building_count",
        "building_area_ratio",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "nlos_flag",
        "terrain_elevation_m",
        "terrain_slope_deg",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "nearest_site_distance_m",
        "azimuth_delta_deg",
    ]
    available_cols = set(pd.read_csv(baseline_csv, nrows=0).columns)
    df = pd.read_csv(baseline_csv, usecols=[col for col in usecols if col in available_cols])
    for col in usecols:
        if col not in df.columns:
            df[col] = np.nan
    df["band"] = pd.to_numeric(df["band"], errors="coerce")
    df["earfcn"] = pd.to_numeric(df["earfcn"], errors="coerce")
    df["before_rsrp"] = _preferred_metric(df, "pred_rsrp_smoothed", "pred_rsrp")
    df["before_rsrq"] = _preferred_metric(df, "pred_rsrq_smoothed", "pred_rsrq")
    df["before_sinr"] = _preferred_metric(df, "pred_sinr_smoothed", "pred_sinr")
    for col in ["pred_rsrp", "pred_rsrq", "pred_sinr", "before_rsrp", "before_rsrq", "before_sinr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cell_grid = (
        df.groupby(["grid_id", "Node_Cell_ID"], dropna=False)
        .agg(
            serving_cell_key=("topology_match_id", "first"),
            serving_cell_id=("Node_Cell_ID", "first"),
            serving_band=("band", "first"),
            serving_earfcn=("earfcn", "first"),
            serving_azimuth=("azimuth_delta_deg", "first"),
            before_rsrp=("before_rsrp", "mean"),
            before_rsrq=("before_rsrq", "mean"),
            before_sinr=("before_sinr", "mean"),
            candidate_rsrp=("pred_rsrp", "mean"),
            candidate_rsrq=("pred_rsrq", "mean"),
            candidate_sinr=("pred_sinr", "mean"),
            candidate_rsrp_max=("pred_rsrp", "max"),
            sample_count=("pred_rsrp", "size"),
            clutter_class=("clutter_class", "first"),
            building_count=("building_count", "first"),
            building_area_ratio=("building_area_ratio", "first"),
            road_length_m=("road_length_m", "first"),
            green_ratio=("green_ratio", "first"),
            water_ratio=("water_ratio", "first"),
            nlos_flag=("nlos_flag", "first"),
            terrain_elevation_m=("terrain_elevation_m", "first"),
            terrain_slope_deg=("terrain_slope_deg", "first"),
            site_count_250m=("site_count_250m", "first"),
            site_count_500m=("site_count_500m", "first"),
            serving_distance_m=("serving_distance_m", "first"),
            nearest_site_distance_m=("nearest_site_distance_m", "first"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
        .reset_index()
    )
    cell_grid = cell_grid.sort_values(["grid_id", "before_rsrp"], ascending=[True, False])
    cell_grid["rank"] = cell_grid.groupby("grid_id").cumcount() + 1
    serving = cell_grid[cell_grid["rank"].eq(1)].copy()
    neighbor1 = cell_grid[cell_grid["rank"].eq(2)][
        ["grid_id", "serving_band", "candidate_rsrp", "candidate_sinr"]
    ].rename(columns={"serving_band": "neighbor1_band", "candidate_rsrp": "neighbor1_rsrp", "candidate_sinr": "neighbor1_sinr"})
    neighbor2 = cell_grid[cell_grid["rank"].eq(3)][["grid_id", "candidate_rsrp", "candidate_sinr"]].rename(
        columns={"candidate_rsrp": "neighbor2_rsrp", "candidate_sinr": "neighbor2_sinr"}
    )
    grid_summary = (
        cell_grid.groupby("grid_id", as_index=False)
        .agg(
            candidate_cell_count=("Node_Cell_ID", "nunique"),
            unique_sites=("serving_cell_key", "nunique"),
            total_prediction_samples=("sample_count", "sum"),
            mean_candidate_rsrp=("candidate_rsrp", "mean"),
            max_candidate_rsrp=("candidate_rsrp_max", "max"),
            mean_candidate_rsrq=("candidate_rsrq", "mean"),
            mean_candidate_sinr=("candidate_sinr", "mean"),
            std_candidate_rsrp=("candidate_rsrp", "std"),
            std_candidate_sinr=("candidate_sinr", "std"),
        )
    )
    band_counts = (
        cell_grid.assign(band_class=cell_grid["serving_band"].map(_classify_band))
        .groupby(["grid_id", "band_class"])["sample_count"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for label in ["LOW_BAND", "MID_BAND", "HIGH_BAND"]:
        if label not in band_counts.columns:
            band_counts[label] = 0.0
    band_total = band_counts[["LOW_BAND", "MID_BAND", "HIGH_BAND"]].sum(axis=1).replace(0, np.nan)
    band_counts["low_band_ratio"] = (band_counts["LOW_BAND"] / band_total).fillna(0.0)
    band_counts["mid_band_ratio"] = (band_counts["MID_BAND"] / band_total).fillna(0.0)
    band_counts["high_band_ratio"] = (band_counts["HIGH_BAND"] / band_total).fillna(0.0)
    band_counts["carrier_count"] = (
        (band_counts["LOW_BAND"] > 0).astype(int)
        + (band_counts["MID_BAND"] > 0).astype(int)
        + (band_counts["HIGH_BAND"] > 0).astype(int)
    )
    band_counts["dominant_band_class"] = band_counts[["low_band_ratio", "mid_band_ratio", "high_band_ratio"]].idxmax(axis=1)
    band_counts["dominant_band_class"] = band_counts["dominant_band_class"].map(
        {"low_band_ratio": "LOW_BAND", "mid_band_ratio": "MID_BAND", "high_band_ratio": "HIGH_BAND"}
    )

    out = serving.merge(neighbor1, on="grid_id", how="left")
    out = out.merge(neighbor2, on="grid_id", how="left")
    out = out.merge(grid_summary, on="grid_id", how="left")
    out = out.merge(
        band_counts[["grid_id", "low_band_ratio", "mid_band_ratio", "high_band_ratio", "carrier_count", "dominant_band_class"]],
        on="grid_id",
        how="left",
    )
    out["rsrp_gap_to_neighbor1"] = out["candidate_rsrp"] - out["neighbor1_rsrp"]
    out["sinr_gap_to_neighbor1"] = out["candidate_sinr"] - out["neighbor1_sinr"]
    out["neighbor_interference_index"] = np.power(10.0, (out["neighbor1_rsrp"].fillna(-140.0) - out["candidate_rsrp"]) / 10.0)
    out["raw_neighbor1_rsrp"] = out["neighbor1_rsrp"]
    out["raw_neighbor1_sinr"] = out["neighbor1_sinr"]
    out["raw_neighbor_interference_index"] = out["neighbor_interference_index"]

    # Project 196 has current baseline context only; future geo/topology deltas are not available here.
    for col in [
        "clutter_transition_flag",
        "geo_transition_source_score",
        "geo_impact_zone_score",
        "geo_impact_nearby_flag",
        "building_growth_ratio",
        "road_growth_ratio",
        "topology_serving_changed_flag",
        "topology_band_changed_flag",
        "rf_geo_blockage_penalty_db",
    ]:
        out[col] = 0.0
    out["rf_load_pressure_proxy"] = (
        0.45 * out["neighbor_interference_index"].fillna(0.0).clip(0.0, 3.0)
        + 0.20 * (out["candidate_cell_count"].fillna(1.0).clip(1.0, 12.0) / 12.0)
    ).clip(0.0, 2.2)
    out["road_density"] = pd.to_numeric(out["road_length_m"], errors="coerce").fillna(0.0) / 625.0
    out["target_grid_area_m2"] = 625.0
    out["grid_size_m"] = 25.0
    out["target_grid_size_m"] = 25.0
    out["source_geo_tile_area_m2"] = 625.0
    out["park_open_area"] = 0.0
    out["open_area_ratio"] = 0.0
    out["mall_presence"] = 0.0
    out["metro_presence"] = 0.0
    out["building_area_sum_m2"] = pd.to_numeric(out["building_area_ratio"], errors="coerce").fillna(0.0) * 625.0
    out["avg_building_area_m2"] = out["building_area_sum_m2"] / pd.to_numeric(out["building_count"], errors="coerce").replace(0, np.nan)
    out["building_count_calc"] = out["building_count"]
    out["building_area_sum_m2_calc"] = out["building_area_sum_m2"]
    out["avg_building_area_m2_calc"] = out["avg_building_area_m2"]
    out["serving_sample_count"] = out["sample_count"]
    out["current_rsrp"] = out["before_rsrp"]
    out["current_sinr"] = out["before_sinr"]
    out["current_rsrq"] = out["before_rsrq"]
    out["geo_snapshot_mode"] = "project196_current"
    out["geo_assignment_source"] = "project196_model3_input"
    out["geo_assignment_method"] = "baseline_grid_aggregate"
    return add_current_state_pressure_features(out)


def ensure_features(df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in numeric:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in categorical:
        if col not in out.columns:
            out[col] = "Unknown"
        out[col] = out[col].fillna("Unknown").astype(str)
    return out


def metric_summary(df: pd.DataFrame, before_col: str, after_col: str, poor_threshold: float, lower_is_worse: bool = True) -> Dict[str, object]:
    before = pd.to_numeric(df[before_col], errors="coerce")
    after = pd.to_numeric(df[after_col], errors="coerce")
    before_poor = before < poor_threshold if lower_is_worse else before > poor_threshold
    after_poor = after < poor_threshold if lower_is_worse else after > poor_threshold
    delta = after - before
    return {
        "before_mean": float(before.mean()),
        "after_mean": float(after.mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "p05_delta": float(delta.quantile(0.05)),
        "p95_delta": float(delta.quantile(0.95)),
        "before_poor_count": int(before_poor.sum()),
        "after_poor_count": int(after_poor.sum()),
        "before_poor_pct": float(before_poor.mean() * 100.0),
        "after_poor_pct": float(after_poor.mean() * 100.0),
        "improved_out_of_poor_count": int((before_poor & ~after_poor).sum()),
        "new_poor_count": int((~before_poor & after_poor).sum()),
    }


def _series_stats(series: pd.Series) -> Dict[str, float]:
    values = pd.to_numeric(series, errors="coerce")
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def _training_target_stats(training_csv: Path | None) -> Dict[str, Dict[str, float]]:
    if training_csv is None or not training_csv.exists():
        return {}
    train = pd.read_csv(training_csv, usecols=[target for target in TARGETS])
    return {target: _series_stats(train[target]) for target in TARGETS}


def run_inference(baseline_csv: Path, run_dir: Path, output_dir: Path, training_csv: Path | None = None) -> Dict[str, object]:
    df = build_project196_grid_features(baseline_csv)
    no_future_change_mask = pd.Series(True, index=df.index)
    for col in CHANGE_SIGNAL_COLUMNS:
        no_future_change_mask &= pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0).abs().le(1e-9)
    for target in TARGETS:
        meta = json.loads((run_dir / target / "metadata.json").read_text(encoding="utf-8"))
        numeric = list(meta["numeric_features"])
        categorical = list(meta["categorical_features"])
        features = numeric + categorical
        model_input = ensure_features(df, numeric, categorical)
        model = joblib.load(run_dir / target / f"{target}.joblib")
        pred = model.predict(model_input[features])
        prediction_mode = meta.get("prediction_mode")
        if prediction_mode in {"delta_from_current_kpi", "current_state_future_delta"}:
            current_col = meta.get("current_kpi_column")
            if not current_col or current_col not in df.columns:
                raise RuntimeError(f"Delta model for {target} is missing current KPI column: {current_col}")
            if prediction_mode == "delta_from_current_kpi":
                pred = np.where(no_future_change_mask.to_numpy(), 0.0, pred)
            df[f"model1_delta_{target}"] = pred
            df[f"model1_after_{target}"] = pd.to_numeric(df[current_col], errors="coerce") + pred
        else:
            df[f"model1_after_{target}"] = pred

    df["delta_rsrp"] = df["model1_after_pred_rsrp"] - df["before_rsrp"]
    df["delta_sinr"] = df["model1_after_pred_sinr"] - df["before_sinr"]
    df["delta_rsrq"] = df["model1_after_pred_rsrq"] - df["before_rsrq"]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "project196_model1_grid_predictions.csv"
    df.to_csv(out_csv, index=False)
    train_stats = _training_target_stats(training_csv)
    project_stats = {
        "before_pred_rsrp": _series_stats(df["before_rsrp"]),
        "before_pred_sinr": _series_stats(df["before_sinr"]),
        "before_pred_rsrq": _series_stats(df["before_rsrq"]),
        "after_pred_rsrp": _series_stats(df["model1_after_pred_rsrp"]),
        "after_pred_sinr": _series_stats(df["model1_after_pred_sinr"]),
        "after_pred_rsrq": _series_stats(df["model1_after_pred_rsrq"]),
    }
    ood_flags = {}
    for target, before_col in {
        "pred_rsrp": "before_pred_rsrp",
        "pred_sinr": "before_pred_sinr",
        "pred_rsrq": "before_pred_rsrq",
    }.items():
        if target not in train_stats:
            continue
        ood_flags[target] = {
            "project196_before_mean_below_training_p05": project_stats[before_col]["mean"] < train_stats[target]["p05"],
            "project196_before_p05_below_training_min": project_stats[before_col]["p05"] < train_stats[target]["min"],
            "training_p05": train_stats[target]["p05"],
            "training_min": train_stats[target]["min"],
            "project196_before_mean": project_stats[before_col]["mean"],
            "project196_before_p05": project_stats[before_col]["p05"],
        }

    summary = {
        "baseline_csv": str(baseline_csv),
        "run_dir": str(run_dir),
        "training_csv": str(training_csv) if training_csv else None,
        "output_csv": str(out_csv),
        "grid_rows": int(len(df)),
        "no_future_change_rows_detected": int(no_future_change_mask.sum()),
        "project196_distribution": project_stats,
        "training_target_distribution": train_stats,
        "out_of_distribution_flags": ood_flags,
        "rsrp_lt_minus95": metric_summary(df, "before_rsrp", "model1_after_pred_rsrp", -95.0, lower_is_worse=True),
        "sinr_lt_0": metric_summary(df, "before_sinr", "model1_after_pred_sinr", 0.0, lower_is_worse=True),
        "rsrq_lt_minus10": metric_summary(df, "before_rsrq", "model1_after_pred_rsrq", -10.0, lower_is_worse=True),
        "top_rsrp_improvements": df.nlargest(10, "delta_rsrp")[
            ["grid_id", "lat", "lon", "before_rsrp", "model1_after_pred_rsrp", "delta_rsrp", "before_sinr", "model1_after_pred_sinr"]
        ].to_dict(orient="records"),
        "top_rsrp_degradations": df.nsmallest(10, "delta_rsrp")[
            ["grid_id", "lat", "lon", "before_rsrp", "model1_after_pred_rsrp", "delta_rsrp", "before_sinr", "model1_after_pred_sinr"]
        ].to_dict(orient="records"),
    }
    summary_json = output_dir / "project196_model1_inference_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run balanced Model 1 25m RF inference on Project 196 Model3 input.")
    parser.add_argument("--baseline-csv", default="models/model3_project196_input/project_196_model3_baseline_grid_input.csv")
    parser.add_argument("--run-dir", default="tests/coverage_prediction/weights/model1_25m_rf_balanced_20260803_1510")
    parser.add_argument("--output-dir", default="tests/output/project196_model1_inference")
    parser.add_argument("--training-csv", default="tests/coverage_prediction/data/model1_training_25m_rf_dataset.csv")
    args = parser.parse_args()
    summary = run_inference(Path(args.baseline_csv), Path(args.run_dir), Path(args.output_dir), Path(args.training_csv))
    print(json.dumps(summary, indent=2, default=str)[:12000])


if __name__ == "__main__":
    main()
