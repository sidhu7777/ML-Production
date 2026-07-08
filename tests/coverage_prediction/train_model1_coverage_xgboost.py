"""
Model 1 coverage training pipeline.

This test-side trainer reads the saved coverage archive, builds a grid-level
training dataset, persists that dataset into ML/data, and trains one regressor
per target:
    - pred_rsrp
    - pred_rsrq
    - pred_sinr
"""

from __future__ import annotations

import io
import json
import logging
import math
import subprocess
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
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from tests.coverage_prediction.coverage_artifact_locator import resolve_coverage_artifact_path

warnings.filterwarnings("ignore")


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = Path("data")
COVERAGE_ARCHIVE = resolve_coverage_artifact_path()
MODEL1_DATASET_CSV = DATA_DIR / "model1_coverage_training.csv"
MODEL1_DATASET_SUMMARY_JSON = DATA_DIR / "model1_coverage_training.summary.json"

MODEL_ROOT = Path("models") / "model1"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = MODEL_ROOT / "training.log"

TARGETS = ["pred_rsrp", "pred_rsrq", "pred_sinr"]

DEFAULT_BUCKET_BAND_MIX_PLAN = {
    "PART_1": {900: 0.15, 1800: 0.69, 850: 0.15, 2300: 0.01},
    "PART_2": {900: 0.11, 1800: 0.76, 850: 0.10, 2300: 0.03},
    "PART_3": {900: 0.08, 1800: 0.80, 850: 0.07, 2300: 0.05},
}

NUMERIC_FEATURES = [
    "grid_centroid_lat",
    "grid_centroid_lon",
    "bucket_seq",
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
    "measurement_count",
    "unique_cells",
    "unique_sites",
    "pred_rsrp_min",
    "pred_rsrp_max",
    "pred_rsrp_std",
    "pred_sinr_std",
    "serving_distance_m",
    "nearest_site_distance_m",
    "site_count_250m",
    "site_count_500m",
    "azimuth_delta_deg",
    "bandwidth_mhz_est",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "carrier_count",
    "clutter_transition_flag",
    "clutter_upgrade_score",
    "morphology_cluster",
    "interference_gap_db",
    "interference_ratio_linear",
    "terrain_elevation_m",
    "terrain_slope_deg",
    "los_blocked_ratio",
    "nlos_flag",
    "prev_obs_rsrp",
    "prev_obs_rsrq",
    "prev_obs_sinr",
    "prev2_obs_rsrp",
    "prev2_obs_rsrq",
    "prev2_obs_sinr",
    "prev_trend_rsrp",
    "prev_trend_rsrq",
    "prev_trend_sinr",
]

CATEGORICAL_FEATURES = ["clutter_class", "dominant_band_class"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT = 600
MAX_GENERALISATION_GAP = 0.10
MAX_RESIDUAL_BIAS = 0.50
EARTH_RADIUS_M = 6371000.0


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("model1_train")
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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)


def _archive_root(archive_path: Path) -> str:
    listed = subprocess.check_output(["tar", "-tf", str(archive_path)], text=True)
    first_file = next((line for line in listed.splitlines() if "/" in line and not line.endswith("/")), "")
    if not first_file:
        raise RuntimeError(f"No files found in archive: {archive_path}")
    return first_file.split("/", 1)[0]


def _read_csv_from_archive(archive_path: Path, member_name: str) -> pd.DataFrame:
    root = _archive_root(archive_path)
    raw = subprocess.check_output(["tar", "-xOf", str(archive_path), f"{root}/{member_name}"])
    return pd.read_csv(io.BytesIO(raw))


def _extract_site_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.rsplit("_", n=1).str[0]


def _haversine_distance_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _bearing_deg(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _safe_angle_delta_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _add_rf_topology_features(pred_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    work = pred_df.copy()
    work["Node_Cell_ID"] = work["Node_Cell_ID"].astype(str)
    work["site_id"] = _extract_site_id(work["Node_Cell_ID"])

    site_work = site_df.copy()
    site_work["Node_Cell_ID"] = site_work["Node_Cell_ID"].astype(str)
    site_work["site_id"] = _extract_site_id(site_work["Node_Cell_ID"])

    serving_lookup = (
        site_work[["Node_Cell_ID", "lat", "lon", "azimuth"]]
        .dropna(subset=["lat", "lon"])
        .drop_duplicates(subset=["Node_Cell_ID"])
        .rename(columns={"lat": "serving_site_lat", "lon": "serving_site_lon", "azimuth": "serving_site_azimuth"})
    )
    work = work.merge(serving_lookup, on="Node_Cell_ID", how="left")

    work["serving_distance_m"] = _haversine_distance_m(
        pd.to_numeric(work["lat"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(work["lon"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(work["serving_site_lat"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(work["serving_site_lon"], errors="coerce").to_numpy(dtype=float),
    )

    serving_bearing = _bearing_deg(
        pd.to_numeric(work["serving_site_lat"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(work["serving_site_lon"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(work["lat"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(work["lon"], errors="coerce").to_numpy(dtype=float),
    )
    work["azimuth_delta_deg"] = _safe_angle_delta_deg(
        serving_bearing,
        pd.to_numeric(work["serving_site_azimuth"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
    )

    unique_sites = (
        site_work.groupby("site_id", as_index=False)
        .agg(site_lat=("lat", "mean"), site_lon=("lon", "mean"))
        .dropna(subset=["site_lat", "site_lon"])
    )
    unique_points = work[["lat", "lon"]].drop_duplicates().copy()
    point_rad = np.radians(unique_points[["lat", "lon"]].to_numpy(dtype=float))
    site_rad = np.radians(unique_sites[["site_lat", "site_lon"]].to_numpy(dtype=float))
    tree = BallTree(site_rad, metric="haversine")

    nearest_dist, _ = tree.query(point_rad, k=1)
    unique_points["nearest_site_distance_m"] = nearest_dist[:, 0] * EARTH_RADIUS_M
    unique_points["site_count_250m"] = tree.query_radius(point_rad, r=250.0 / EARTH_RADIUS_M, count_only=True)
    unique_points["site_count_500m"] = tree.query_radius(point_rad, r=500.0 / EARTH_RADIUS_M, count_only=True)

    return work.merge(unique_points, on=["lat", "lon"], how="left")


def _add_temporal_history_features(df: pd.DataFrame) -> pd.DataFrame:
    bucket_seq_map = {"PART_1": 1, "PART_2": 2, "PART_3": 3}
    work = df.copy()
    work["bucket_seq"] = work["time_bucket"].map(bucket_seq_map).astype(float)
    work = work.sort_values(["grid_id", "bucket_seq", "bucket_mid_timestamp", "bucket_max_timestamp"]).reset_index(drop=True)

    for target, suffix in [("pred_rsrp", "rsrp"), ("pred_rsrq", "rsrq"), ("pred_sinr", "sinr")]:
        grouped = work.groupby("grid_id")[target]
        work[f"prev_obs_{suffix}"] = grouped.shift(1)
        work[f"prev2_obs_{suffix}"] = grouped.shift(2)
        work[f"prev_trend_{suffix}"] = work[f"prev_obs_{suffix}"] - work[f"prev2_obs_{suffix}"]

    return work


def _classify_mhz(freq_value: object) -> str:
    try:
        freq = float(freq_value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if not math.isfinite(freq) or freq <= 0:
        return "UNKNOWN"
    if freq <= 1000:
        return "LOW_BAND"
    if freq <= 2000:
        return "MID_BAND"
    return "HIGH_BAND"


def _classify_lte_earfcn(earfcn_value: object) -> str:
    try:
        earfcn = float(earfcn_value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if not math.isfinite(earfcn) or earfcn <= 0:
        return "UNKNOWN"
    if 2400 <= earfcn <= 2649:
        return "LOW_BAND"
    if 3450 <= earfcn <= 3799:
        return "LOW_BAND"
    if 1200 <= earfcn <= 1949:
        return "MID_BAND"
    if 0 <= earfcn <= 599:
        return "HIGH_BAND"
    if 38650 <= earfcn <= 39649:
        return "HIGH_BAND"
    if 39650 <= earfcn <= 41589:
        return "HIGH_BAND"
    if earfcn >= 600000:
        return "HIGH_BAND"
    if 600 <= earfcn <= 6000:
        return _classify_mhz(earfcn)
    return "UNKNOWN"


def _bucket_band_mix_ratios(bucket_label: object) -> dict[str, float]:
    plan = DEFAULT_BUCKET_BAND_MIX_PLAN.get(str(bucket_label), DEFAULT_BUCKET_BAND_MIX_PLAN["PART_3"])
    totals = {"LOW_BAND": 0.0, "MID_BAND": 0.0, "HIGH_BAND": 0.0}
    for band, ratio in plan.items():
        totals[_classify_mhz(band)] += float(ratio)
    total = sum(totals.values())
    if total <= 0:
        return {key: 0.0 for key in totals}
    return {key: value / total for key, value in totals.items()}


def _derive_band_features_from_kpi(kpi_df: pd.DataFrame) -> pd.DataFrame:
    if kpi_df.empty:
        return pd.DataFrame(
            columns=[
                "grid_id",
                "grid_row",
                "grid_col",
                "grid_centroid_lat",
                "grid_centroid_lon",
                "time_bucket",
                "bandwidth_mhz_est",
                "low_band_ratio",
                "mid_band_ratio",
                "high_band_ratio",
                "dominant_band_class",
                "carrier_count",
            ]
        )
    work = kpi_df.copy()
    work["dominant_band_class"] = work.get("dominant_earfcn", pd.Series(index=work.index, dtype=float)).map(_classify_lte_earfcn)
    work["bandwidth_mhz_est"] = pd.to_numeric(work.get("bandwidth_mhz_est"), errors="coerce")
    ratios = work["time_bucket"].astype(str).map(_bucket_band_mix_ratios)
    work["low_band_ratio"] = ratios.map(lambda item: item["LOW_BAND"] if isinstance(item, dict) else 0.0)
    work["mid_band_ratio"] = ratios.map(lambda item: item["MID_BAND"] if isinstance(item, dict) else 0.0)
    work["high_band_ratio"] = ratios.map(lambda item: item["HIGH_BAND"] if isinstance(item, dict) else 0.0)
    missing_band_mask = work["dominant_band_class"].eq("UNKNOWN")
    work.loc[missing_band_mask, "dominant_band_class"] = np.where(
        work.loc[missing_band_mask, "mid_band_ratio"] >= work.loc[missing_band_mask, ["low_band_ratio", "high_band_ratio"]].max(axis=1),
        "MID_BAND",
        np.where(
            work.loc[missing_band_mask, "high_band_ratio"] >= work.loc[missing_band_mask, "low_band_ratio"],
            "HIGH_BAND",
            "LOW_BAND",
        ),
    )
    work["carrier_count"] = np.where(
        work.get("dominant_earfcn").notna(),
        1.0,
        (
            (work["low_band_ratio"] > 0).astype(int)
            + (work["mid_band_ratio"] > 0).astype(int)
            + (work["high_band_ratio"] > 0).astype(int)
        ).astype(float),
    )
    return work[
        [
            "grid_id",
            "grid_row",
            "grid_col",
            "grid_centroid_lat",
            "grid_centroid_lon",
            "time_bucket",
            "bandwidth_mhz_est",
            "low_band_ratio",
            "mid_band_ratio",
            "high_band_ratio",
            "dominant_band_class",
            "carrier_count",
        ]
    ].copy()


def _derive_corrected_surface_features(corrected_pred_df: pd.DataFrame) -> pd.DataFrame:
    if corrected_pred_df.empty:
        return pd.DataFrame()
    work = corrected_pred_df.copy()
    group_cols = ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"]
    feature_df = (
        work.groupby(group_cols, as_index=False)
        .agg(
            morphology_cluster=("morphology_cluster", "median"),
            interference_gap_db=("interference_gap_db", "mean"),
            interference_ratio_linear=("interference_ratio_linear", "mean"),
            terrain_elevation_m=("terrain_elevation_m", "mean"),
            terrain_slope_deg=("terrain_slope_deg", "mean"),
            los_blocked_ratio=("los_blocked_ratio", "mean"),
            nlos_flag=("nlos_flag", "mean"),
        )
    )
    return feature_df


def _add_clutter_evolution_features(df: pd.DataFrame) -> pd.DataFrame:
    clutter_levels = {
        "Water": 0,
        "Vegetation": 0,
        "Rural/Open": 0,
        "Suburban": 1,
        "Urban": 2,
        "Dense Urban": 3,
    }
    work = df.copy()
    clutter_level = (
        work.get("clutter_class", pd.Series("", index=work.index))
        .fillna("")
        .astype(str)
        .map(clutter_levels)
        .fillna(0)
        .astype(int)
    )
    baseline_level = (
        work.assign(_clutter_level=clutter_level)
        .sort_values(["grid_id", "bucket_seq"])
        .groupby("grid_id", sort=False)["_clutter_level"]
        .transform("first")
        .reindex(work.index)
        .fillna(0)
        .astype(int)
    )
    work["clutter_transition_flag"] = clutter_level.ne(baseline_level).astype(float)
    work["clutter_upgrade_score"] = np.where(clutter_level > baseline_level, clutter_level, 0).astype(float)
    return work


def build_dataset() -> pd.DataFrame:
    log.info("Loading coverage archive: %s", COVERAGE_ARCHIVE)
    if not COVERAGE_ARCHIVE.exists():
        raise FileNotFoundError(f"Missing archive: {COVERAGE_ARCHIVE}")

    pred_df = _read_csv_from_archive(COVERAGE_ARCHIVE, "baseline_prediction_grid.csv")
    corrected_pred_df = _read_csv_from_archive(COVERAGE_ARCHIVE, "bucket_corrected_prediction_grid.csv")
    kpi_df = _read_csv_from_archive(COVERAGE_ARCHIVE, "grid_kpi_timeseries.csv")
    geo_df = _read_csv_from_archive(COVERAGE_ARCHIVE, "bucket_grid_geo_features.csv")
    site_df = _read_csv_from_archive(COVERAGE_ARCHIVE, "project_sites.csv")
    coverage_rows_df = _read_csv_from_archive(COVERAGE_ARCHIVE, "coverage_rows.csv")

    log.info(
        "Archive inputs loaded: baseline=%s corrected=%s kpi=%s geo=%s sites=%s coverage_rows=%s",
        pred_df.shape,
        corrected_pred_df.shape,
        kpi_df.shape,
        geo_df.shape,
        site_df.shape,
        coverage_rows_df.shape,
    )

    pred_df = _add_rf_topology_features(pred_df, site_df)

    feature_agg_df = (
        pred_df.groupby(
            ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"],
            as_index=False,
        )
        .agg(
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

    coverage_rows_df["timestamp"] = pd.to_datetime(coverage_rows_df["timestamp"], errors="coerce")
    observed_target_df = (
        coverage_rows_df.groupby(
            ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"],
            as_index=False,
        )
        .agg(
            pred_rsrp=("rsrp", "mean"),
            pred_rsrq=("rsrq", "mean"),
            pred_sinr=("sinr", "mean"),
        )
    )
    time_agg_df = (
        coverage_rows_df.dropna(subset=["timestamp"])
        .groupby(
            ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"],
            as_index=False,
        )
        .agg(
            bucket_min_timestamp=("timestamp", "min"),
            bucket_max_timestamp=("timestamp", "max"),
        )
    )
    time_agg_df["bucket_mid_timestamp"] = time_agg_df["bucket_min_timestamp"] + (
        (time_agg_df["bucket_max_timestamp"] - time_agg_df["bucket_min_timestamp"]) / 2
    )
    agg_df = feature_agg_df.merge(
        observed_target_df,
        on=["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"],
        how="left",
        validate="one_to_one",
    )
    agg_df = agg_df.merge(
        time_agg_df,
        on=["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"],
        how="left",
        validate="one_to_one",
    )

    corrected_feature_df = _derive_corrected_surface_features(corrected_pred_df)
    if not corrected_feature_df.empty:
        agg_df = agg_df.merge(
            corrected_feature_df,
            on=["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"],
            how="left",
            validate="one_to_one",
        )

    band_feature_df = _derive_band_features_from_kpi(kpi_df)
    if not band_feature_df.empty:
        agg_df = agg_df.merge(
            band_feature_df,
            on=["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"],
            how="left",
            validate="one_to_one",
        )

    geo_df = geo_df.drop(columns=["centroid_lat", "centroid_lon"], errors="ignore")
    geo_df = geo_df.drop(
        columns=[
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

    df = agg_df.merge(
        geo_df,
        on=["grid_id", "grid_row", "grid_col", "time_bucket"],
        how="left",
        validate="one_to_one",
    )

    for col in NUMERIC_FEATURES + TARGETS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    target_nulls = df[TARGETS].isnull().sum()
    if target_nulls.any():
        log.warning("Observed target nulls before filtering:\n%s", target_nulls.to_string())
        before_drop = len(df)
        df = df.dropna(subset=TARGETS).reset_index(drop=True)
        log.info("Dropped %d rows with missing observed targets; remaining rows=%d", before_drop - len(df), len(df))

    df = _add_temporal_history_features(df)
    df = _add_clutter_evolution_features(df)
    feature_nulls = df[ALL_FEATURES].isnull().sum()
    bad_feature_nulls = feature_nulls[feature_nulls > 0]
    if not bad_feature_nulls.empty:
        log.warning("Feature columns with nulls (imputed later):\n%s", bad_feature_nulls.to_string())

    temporal_nulls = df[
        [
            "prev_obs_rsrp",
            "prev_obs_rsrq",
            "prev_obs_sinr",
            "prev2_obs_rsrp",
            "prev2_obs_rsrq",
            "prev2_obs_sinr",
            "prev_trend_rsrp",
            "prev_trend_rsrq",
            "prev_trend_sinr",
        ]
    ].isnull().sum()
    log.info("Temporal history feature nulls (expected for early buckets, imputed later):\n%s", temporal_nulls.to_string())

    df = df.sort_values(["grid_id", "bucket_seq"]).reset_index(drop=True)
    MODEL1_DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MODEL1_DATASET_CSV, index=False)

    summary = {
        "source_archive": str(COVERAGE_ARCHIVE),
        "output_csv": str(MODEL1_DATASET_CSV),
        "rows": int(len(df)),
        "unique_grids": int(df["grid_id"].nunique()),
        "bucket_counts": {str(k): int(v) for k, v in df["time_bucket"].value_counts().sort_index().items()},
        "timestamp_columns_added": [
            "bucket_min_timestamp",
            "bucket_max_timestamp",
            "bucket_mid_timestamp",
        ],
        "features": ALL_FEATURES,
        "targets": TARGETS,
        "topology_features_added": [
            "serving_distance_m",
            "nearest_site_distance_m",
            "site_count_250m",
            "site_count_500m",
            "azimuth_delta_deg",
            "bandwidth_mhz_est",
            "low_band_ratio",
            "mid_band_ratio",
            "high_band_ratio",
            "dominant_band_class",
            "carrier_count",
        ],
        "corrected_surface_features_added": [
            "morphology_cluster",
            "interference_gap_db",
            "interference_ratio_linear",
            "terrain_elevation_m",
            "terrain_slope_deg",
            "los_blocked_ratio",
            "nlos_flag",
        ],
        "clutter_evolution_features_added": [
            "clutter_transition_flag",
            "clutter_upgrade_score",
        ],
        "temporal_features_added": [
            "bucket_seq",
            "prev_obs_rsrp",
            "prev_obs_rsrq",
            "prev_obs_sinr",
            "prev2_obs_rsrp",
            "prev2_obs_rsrq",
            "prev2_obs_sinr",
            "prev_trend_rsrp",
            "prev_trend_rsrq",
            "prev_trend_sinr",
        ],
        "target_source": "coverage_rows.csv aggregated by grid_id + time_bucket using observed rsrp/rsrq/sinr",
    }
    save_json(summary, MODEL1_DATASET_SUMMARY_JSON)
    log.info("Model 1 dataset saved to %s", MODEL1_DATASET_CSV)
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    log.info("Temporal split -> TRAIN:%d | VALID:%d | TEST:%d", len(train_df), len(valid_df), len(test_df))
    return train_df, valid_df, test_df


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
    target: str,
    out_dir: Path,
) -> dict:
    log.info("[%s] Optuna HPO: %d trials, timeout=%ds", target, OPTUNA_TRIALS, OPTUNA_TIMEOUT)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name=f"model1_{target}",
    )
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, X_valid, y_valid),
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
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
    target: str,
    out_dir: Path,
) -> dict:
    results: dict[str, object] = {}
    for split_name, X, y in [("TRAIN", X_train, y_train), ("VALID", X_valid, y_valid), ("TEST", X_test, y_test)]:
        pred = pipeline.predict(X)
        metrics = compute_metrics(y.values, pred)
        results[split_name] = metrics
        log.info("[%s] %s -> MAE=%.4f | RMSE=%.4f | R2=%.4f", target, split_name, metrics["mae"], metrics["rmse"], metrics["r2"])

    gap = abs(results["TRAIN"]["mae"] - results["VALID"]["mae"]) / (results["VALID"]["mae"] + 1e-9)
    results["generalisation_gap_pct"] = round(gap * 100, 2)
    if gap > MAX_GENERALISATION_GAP:
        log.warning("[%s] Generalisation gap %.1f%% exceeds threshold %.0f%%", target, gap * 100, MAX_GENERALISATION_GAP * 100)

    residuals = y_test.values - pipeline.predict(X_test)
    results["residual_mean"] = round(float(np.mean(residuals)), 4)
    results["residual_std"] = round(float(np.std(residuals)), 4)
    if abs(results["residual_mean"]) > MAX_RESIDUAL_BIAS:
        log.warning("[%s] Residual mean %.4f exceeds threshold %.2f", target, results["residual_mean"], MAX_RESIDUAL_BIAS)

    save_json(results, out_dir / "metrics.json")
    return results


def run_future_evolution_analysis(
    pipeline: Pipeline,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    out_dir: Path,
) -> dict[str, object]:
    log.info("[%s] Evaluating future evolution on PART_3 test rows", target)

    work = df_test.copy()
    work["prediction"] = pipeline.predict(df_test[ALL_FEATURES])
    work["abs_error"] = np.abs(work["prediction"] - work[target])
    work["squared_error"] = (work["prediction"] - work[target]) ** 2
    work["residual"] = work[target] - work["prediction"]

    overall = {
        "target": target,
        "reference_split": "PART_1 + PART_2",
        "future_split_evaluated": "PART_3 test subset",
        "test_rows": int(len(work)),
        "test_unique_grids": int(work["grid_id"].nunique()),
        "test_metrics": compute_metrics(work[target].to_numpy(dtype=float), work["prediction"].to_numpy(dtype=float)),
        "mean_actual": round(float(work[target].mean()), 4),
        "mean_prediction": round(float(work["prediction"].mean()), 4),
        "mean_abs_gap": round(float(work["abs_error"].mean()), 4),
        "median_abs_gap": round(float(work["abs_error"].median()), 4),
        "mean_residual": round(float(work["residual"].mean()), 4),
    }

    train_history = (
        df_train.groupby("grid_id", as_index=False)
        .agg(
            train_bucket_count=(target, "size"),
            train_actual_mean=(target, "mean"),
            train_actual_std=(target, "std"),
        )
    )
    evolution_df = work.merge(train_history, on="grid_id", how="left")
    evolution_df["future_actual_delta_vs_train_mean"] = evolution_df[target] - evolution_df["train_actual_mean"]
    evolution_df["future_pred_delta_vs_train_mean"] = evolution_df["prediction"] - evolution_df["train_actual_mean"]
    evolution_df["delta_alignment_gap"] = np.abs(
        evolution_df["future_actual_delta_vs_train_mean"] - evolution_df["future_pred_delta_vs_train_mean"]
    )

    clutter_summary = (
        evolution_df.groupby("clutter_class", dropna=False, as_index=False)
        .agg(
            row_count=(target, "size"),
            grid_count=("grid_id", "nunique"),
            actual_mean=(target, "mean"),
            predicted_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
            residual_mean=("residual", "mean"),
            actual_future_delta_mean=("future_actual_delta_vs_train_mean", "mean"),
            predicted_future_delta_mean=("future_pred_delta_vs_train_mean", "mean"),
            delta_alignment_gap_mean=("delta_alignment_gap", "mean"),
        )
        .sort_values(["mae", "row_count"], ascending=[True, False])
    )
    clutter_summary.to_csv(out_dir / "future_evolution_by_clutter.csv", index=False)

    grid_summary = (
        evolution_df.groupby("grid_id", as_index=False)
        .agg(
            grid_row=("grid_row", "first"),
            grid_col=("grid_col", "first"),
            clutter_class=("clutter_class", "first"),
            row_count=(target, "size"),
            actual_mean=(target, "mean"),
            predicted_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
            residual_mean=("residual", "mean"),
            train_actual_mean=("train_actual_mean", "first"),
            actual_future_delta_mean=("future_actual_delta_vs_train_mean", "mean"),
            predicted_future_delta_mean=("future_pred_delta_vs_train_mean", "mean"),
            delta_alignment_gap_mean=("delta_alignment_gap", "mean"),
        )
        .sort_values(["mae", "row_count"], ascending=[True, False])
    )
    grid_summary.to_csv(out_dir / "future_evolution_by_grid.csv", index=False)

    clutter_records = clutter_summary.to_dict(orient="records")
    grid_records = grid_summary.to_dict(orient="records")
    summary = {
        **overall,
        "clutter_alignment": {
            "group_count": int(len(clutter_records)),
            "mean_group_mae": round(float(clutter_summary["mae"].mean()), 4) if not clutter_summary.empty else None,
            "mean_group_delta_alignment_gap": round(float(clutter_summary["delta_alignment_gap_mean"].mean()), 4)
            if not clutter_summary.empty
            else None,
            "best_groups_by_mae": clutter_records[:5],
            "worst_groups_by_mae": list(reversed(clutter_records[-5:])),
        },
        "grid_alignment": {
            "grid_count": int(len(grid_records)),
            "mean_grid_mae": round(float(grid_summary["mae"].mean()), 4) if not grid_summary.empty else None,
            "mean_grid_delta_alignment_gap": round(float(grid_summary["delta_alignment_gap_mean"].mean()), 4)
            if not grid_summary.empty
            else None,
            "best_grids_by_mae": grid_records[:10],
            "worst_grids_by_mae": list(reversed(grid_records[-10:])),
        },
        "artifacts": {
            "clutter_csv": "future_evolution_by_clutter.csv",
            "grid_csv": "future_evolution_by_grid.csv",
            "summary_json": "future_evolution_summary.json",
        },
    }
    save_json(summary, out_dir / "future_evolution_summary.json")
    return summary


def run_shap(pipeline: Pipeline, X_test: pd.DataFrame, target: str, out_dir: Path) -> None:
    log.info("[%s] Computing SHAP values", target)
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
        fig = plt.gcf()
        fig.suptitle(f"Model 1 - {target} SHAP Summary", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:
        log.warning("[%s] SHAP step failed: %s", target, exc)


def run_yellowbrick(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target: str,
    out_dir: Path,
) -> None:
    log.info("[%s] Generating Yellowbrick plots", target)
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
        log.warning("[%s] Yellowbrick step failed: %s", target, exc)


def run_evidently(pipeline: Pipeline, df_train: pd.DataFrame, df_test: pd.DataFrame, target: str, out_dir: Path) -> None:
    log.info("[%s] Generating Evidently report", target)
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
        log.warning("[%s] Evidently step failed: %s", target, exc)


def save_model(pipeline: Pipeline, target: str, metrics: dict, best_params: dict, out_dir: Path) -> None:
    model_path = out_dir / f"{target}.joblib"
    joblib.dump(pipeline, model_path)
    metadata = {
        "target": target,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "XGBRegressor",
        "source_archive": str(COVERAGE_ARCHIVE),
        "training_dataset_csv": str(MODEL1_DATASET_CSV),
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES, "total": len(ALL_FEATURES)},
        "topology_features_added": [
            "serving_distance_m",
            "nearest_site_distance_m",
            "site_count_250m",
            "site_count_500m",
            "azimuth_delta_deg",
        ],
        "hyperparameters": best_params,
        "metrics": metrics,
        "split_logic": {
            "train": "PART_1 + PART_2",
            "valid": "first 60% of PART_3 sorted by bucket_mid_timestamp",
            "test": "last 40% of PART_3 sorted by bucket_mid_timestamp",
        },
        "random_seed": RANDOM_SEED,
        "optuna_trials": OPTUNA_TRIALS,
        "future_evolution_artifacts": [
            "future_evolution_summary.json",
            "future_evolution_by_clutter.csv",
            "future_evolution_by_grid.csv",
        ],
    }
    save_json(metadata, out_dir / "metadata.json")


def train_target(df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame, target: str) -> dict[str, object]:
    out_dir = MODEL_ROOT / target
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = df_train[ALL_FEATURES], df_train[target]
    X_valid, y_valid = df_valid[ALL_FEATURES], df_valid[target]
    X_test, y_test = df_test[ALL_FEATURES], df_test[target]

    best_params = run_optuna(X_train, y_train, X_valid, y_valid, target, out_dir)
    pipeline = fit_pipeline(X_train, y_train, best_params)
    metrics = evaluate(pipeline, X_train, y_train, X_valid, y_valid, X_test, y_test, target, out_dir)
    run_shap(pipeline, X_test, target, out_dir)
    run_yellowbrick(pipeline, X_train, y_train, X_test, y_test, target, out_dir)
    run_evidently(pipeline, df_train, df_test, target, out_dir)
    evolution_summary = run_future_evolution_analysis(pipeline, df_train, df_test, target, out_dir)
    save_model(pipeline, target, metrics, best_params, out_dir)
    return evolution_summary


def main() -> None:
    log.info("Model 1 training started at %s", datetime.utcnow().isoformat())
    df = build_dataset()
    train_df, valid_df, test_df = temporal_split(df)

    all_metrics: dict[str, object] = {}
    all_evolution: dict[str, object] = {}
    for target in TARGETS:
        log.info("=" * 70)
        log.info("TARGET: %s", target)
        log.info("=" * 70)
        all_evolution[target] = train_target(train_df, valid_df, test_df, target)
        metrics_path = MODEL_ROOT / target / "metrics.json"
        if metrics_path.exists():
            all_metrics[target] = json.loads(metrics_path.read_text(encoding="utf-8"))

    save_json(all_metrics, MODEL_ROOT / "all_metrics_summary.json")
    save_json(all_evolution, MODEL_ROOT / "all_future_evolution_summary.json")
    log.info("All artifacts saved under %s", MODEL_ROOT.resolve())


if __name__ == "__main__":
    main()
