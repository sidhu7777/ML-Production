from __future__ import annotations

import numpy as np
import pandas as pd


MODEL1_TARGETS = ["pred_rsrp", "pred_rsrq", "pred_sinr"]


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _num_on_index(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {str(col).lower(): col for col in df.columns}
    for name in names:
        col = lower.get(name.lower())
        if col is not None:
            return col
    return None


def _normalize_baseline(baseline_df: pd.DataFrame) -> pd.DataFrame:
    work = baseline_df.copy()
    lat_col = _first_col(work, ["lat", "latitude", "grid_centroid_lat"])
    lon_col = _first_col(work, ["lon", "lng", "longitude", "grid_centroid_lon"])
    if lat_col is None or lon_col is None:
        raise ValueError("Baseline rows must contain lat/lon columns for Model 1")
    work["lat"] = pd.to_numeric(work[lat_col], errors="coerce")
    work["lon"] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.loc[work["lat"].notna() & work["lon"].notna()].copy()

    for col in MODEL1_TARGETS:
        if col not in work.columns:
            source = col.replace("pred_", "")
            work[col] = _num(work, source)
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    grid_col = _first_col(work, ["grid_id", "gridId", "grid_label"])
    if grid_col is None:
        labels = work["lat"].round(6).astype(str) + "_" + work["lon"].round(6).astype(str)
        work["grid_id"] = pd.factorize(labels, sort=True)[0] + 1
    else:
        grid_num = pd.to_numeric(work[grid_col], errors="coerce")
        fallback = pd.factorize(work["lat"].round(6).astype(str) + "_" + work["lon"].round(6).astype(str), sort=True)[0] + 1
        work["grid_id"] = grid_num.fillna(pd.Series(fallback, index=work.index)).astype("int64")

    if "grid_row" not in work.columns:
        work["grid_row"] = np.nan
    if "grid_col" not in work.columns:
        work["grid_col"] = np.nan
    work["grid_row"] = pd.to_numeric(work["grid_row"], errors="coerce")
    work["grid_col"] = pd.to_numeric(work["grid_col"], errors="coerce")
    return work


def _prepare_geo(geo_df: pd.DataFrame) -> pd.DataFrame:
    if geo_df is None or geo_df.empty:
        return pd.DataFrame()
    work = geo_df.copy()
    lat_col = _first_col(work, ["lat", "latitude", "grid_centroid_lat"])
    lon_col = _first_col(work, ["lon", "lng", "longitude", "grid_centroid_lon"])
    if lat_col is None or lon_col is None:
        return pd.DataFrame()
    work["lat_6dp"] = pd.to_numeric(work[lat_col], errors="coerce").round(6)
    work["lon_6dp"] = pd.to_numeric(work[lon_col], errors="coerce").round(6)
    work = work.loc[work["lat_6dp"].notna() & work["lon_6dp"].notna()].copy()
    return work.drop_duplicates(["lat_6dp", "lon_6dp"], keep="first")


def build_model1_feature_frame(
    baseline_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    base = _normalize_baseline(baseline_df)
    base["lat_6dp"] = base["lat"].round(6)
    base["lon_6dp"] = base["lon"].round(6)

    node_col = _first_col(base, ["Node_Cell_ID", "nodeb_id_cell_id", "rf_identity_key", "cell_id"])
    site_col = _first_col(base, ["site_id", "Site ID", "node_b_id", "nodeb_id"])
    band_col = _first_col(base, ["band", "topology_band", "Band"])

    group_cols = ["grid_id"]
    grouped = base.groupby(group_cols, as_index=False).agg(
        grid_row=("grid_row", "first"),
        grid_col=("grid_col", "first"),
        grid_centroid_lat=("lat", "mean"),
        grid_centroid_lon=("lon", "mean"),
        pred_rsrp_min=("pred_rsrp", "min"),
        pred_rsrp_max=("pred_rsrp", "max"),
        pred_rsrp_std=("pred_rsrp", "std"),
        pred_sinr_std=("pred_sinr", "std"),
        measurement_count=("pred_rsrp", "size"),
        current_rsrp=("pred_rsrp", "mean"),
        current_rsrq=("pred_rsrq", "mean"),
        current_sinr=("pred_sinr", "mean"),
    )
    grouped["unique_cells"] = (
        base.groupby("grid_id")[node_col].nunique(dropna=True).reindex(grouped["grid_id"]).to_numpy()
        if node_col
        else 1
    )
    grouped["unique_sites"] = (
        base.groupby("grid_id")[site_col].nunique(dropna=True).reindex(grouped["grid_id"]).to_numpy()
        if site_col
        else 1
    )
    grouped["carrier_count"] = (
        base.groupby("grid_id")[band_col].nunique(dropna=True).reindex(grouped["grid_id"]).to_numpy()
        if band_col
        else 1
    )

    if band_col:
        band = pd.to_numeric(base[band_col], errors="coerce")
        band_group = base.assign(_band=band).groupby("grid_id")["_band"]
        counts = band_group.count().replace(0, np.nan)
        grouped["low_band_ratio"] = band_group.apply(lambda s: float((s <= 900).sum())).reindex(grouped["grid_id"]).to_numpy() / counts.reindex(grouped["grid_id"]).to_numpy()
        grouped["mid_band_ratio"] = band_group.apply(lambda s: float(((s > 900) & (s <= 2100)).sum())).reindex(grouped["grid_id"]).to_numpy() / counts.reindex(grouped["grid_id"]).to_numpy()
        grouped["high_band_ratio"] = band_group.apply(lambda s: float((s > 2100).sum())).reindex(grouped["grid_id"]).to_numpy() / counts.reindex(grouped["grid_id"]).to_numpy()
    else:
        grouped["low_band_ratio"] = 0.0
        grouped["mid_band_ratio"] = 1.0
        grouped["high_band_ratio"] = 0.0

    grouped["dominant_band_class"] = np.select(
        [
            grouped["low_band_ratio"] >= grouped[["mid_band_ratio", "high_band_ratio"]].max(axis=1),
            grouped["high_band_ratio"] >= grouped[["low_band_ratio", "mid_band_ratio"]].max(axis=1),
        ],
        ["LOW_BAND", "HIGH_BAND"],
        default="MID_BAND",
    )

    geo = _prepare_geo(geo_df)
    if not geo.empty:
        grouped["lat_6dp"] = grouped["grid_centroid_lat"].round(6)
        grouped["lon_6dp"] = grouped["grid_centroid_lon"].round(6)
        geo_cols = [
            "lat_6dp",
            "lon_6dp",
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
            "clutter_class",
            "morphology_cluster",
            "terrain_elevation_m",
            "terrain_slope_deg",
            "los_blocked_ratio",
            "nlos_flag",
            "serving_distance_m",
            "nearest_site_distance_m",
            "site_count_250m",
            "site_count_500m",
            "azimuth_delta_deg",
        ]
        grouped = grouped.merge(geo[[c for c in geo_cols if c in geo.columns]], on=["lat_6dp", "lon_6dp"], how="left")

    grouped["grid_size_m"] = _num_on_index(grouped, "grid_size_m").fillna(25.0)
    grouped["grid_area_m2"] = _num_on_index(grouped, "grid_area_m2").fillna(grouped["grid_size_m"] ** 2)
    grouped["cell_area_m2"] = grouped["grid_area_m2"]
    grouped["bandwidth_mhz_est"] = _num_on_index(grouped, "bandwidth_mhz_est").fillna(10.0)
    grouped["interference_gap_db"] = grouped["pred_rsrp_max"] - grouped["pred_rsrp_min"]
    grouped["interference_ratio_linear"] = np.power(10.0, grouped["interference_gap_db"].fillna(0) / 10.0)
    grouped["prev_obs_rsrp"] = grouped["current_rsrp"]
    grouped["prev_obs_rsrq"] = grouped["current_rsrq"]
    grouped["prev_obs_sinr"] = grouped["current_sinr"]
    grouped["prev2_obs_rsrp"] = grouped["current_rsrp"]
    grouped["prev2_obs_rsrq"] = grouped["current_rsrq"]
    grouped["prev2_obs_sinr"] = grouped["current_sinr"]
    grouped["prev_trend_rsrp"] = 0.0
    grouped["prev_trend_rsrq"] = 0.0
    grouped["prev_trend_sinr"] = 0.0
    grouped["clutter_transition_flag"] = 0.0
    grouped["clutter_upgrade_score"] = 0.0

    for col in numeric_features:
        if col not in grouped.columns:
            grouped[col] = 0.0
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce")
    for col in categorical_features:
        if col not in grouped.columns:
            grouped[col] = "UNKNOWN"
        grouped[col] = grouped[col].fillna("UNKNOWN").astype(str)

    return grouped
