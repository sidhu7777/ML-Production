from __future__ import annotations

import numpy as np
import pandas as pd


MODEL1_TARGETS = ["pred_rsrp", "pred_rsrq", "pred_sinr"]
BAND_CLASS_LABELS = ("LOW_BAND", "MID_BAND", "HIGH_BAND")


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _num_on_index(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


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


def _classify_band(value: object) -> str:
    band = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(band):
        return "MID_BAND"
    if float(band) <= 900:
        return "LOW_BAND"
    if float(band) >= 2300:
        return "HIGH_BAND"
    return "MID_BAND"


def _add_current_state_pressure_features(grouped: pd.DataFrame) -> pd.DataFrame:
    out = grouped.copy()
    clutter = out.get("clutter_class", pd.Series("", index=out.index)).astype(str).str.lower()
    clutter_risk = pd.Series(0.35, index=out.index, dtype="float64")
    clutter_risk = clutter_risk.where(~clutter.str.contains("dense"), 1.0)
    clutter_risk = clutter_risk.where(~(clutter.str.contains("urban") & ~clutter.str.contains("sub")), 0.72)
    clutter_risk = clutter_risk.where(~clutter.str.contains("sub"), 0.45)
    clutter_risk = clutter_risk.where(~clutter.str.contains("rural|open"), 0.18)
    clutter_risk = clutter_risk.where(~clutter.str.contains("water|green|vegetation"), 0.08)

    building_risk = (
        0.65 * _robust_norm(_num_on_index(out, "building_area_ratio", 0.0).fillna(0.0))
        + 0.35 * _robust_norm(_num_on_index(out, "building_count", 0.0).fillna(0.0))
    ).clip(0.0, 1.0)
    road_activity = (
        0.60 * _robust_norm(_num_on_index(out, "road_density", 0.0).fillna(0.0))
        + 0.40 * _robust_norm(_num_on_index(out, "road_length_m", 0.0).fillna(0.0))
    ).clip(0.0, 1.0)

    if "neighbor_interference_index" not in out.columns:
        gap = _num_on_index(out, "interference_gap_db", 0.0).fillna(0.0)
        out["neighbor_interference_index"] = np.power(10.0, (-gap.clip(lower=0.0)) / 10.0)
    if "rsrp_gap_to_neighbor1" not in out.columns:
        out["rsrp_gap_to_neighbor1"] = _num_on_index(out, "interference_gap_db", 3.0).fillna(3.0)
    if "sinr_gap_to_neighbor1" not in out.columns:
        out["sinr_gap_to_neighbor1"] = 2.0

    interference = (
        0.55 * _robust_norm(_num_on_index(out, "neighbor_interference_index", 0.0).fillna(0.0))
        + 0.25 * ((3.0 - _num_on_index(out, "rsrp_gap_to_neighbor1", 3.0).fillna(3.0)) / 6.0).clip(0.0, 1.0)
        + 0.20 * ((2.0 - _num_on_index(out, "sinr_gap_to_neighbor1", 2.0).fillna(2.0)) / 8.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    site_pressure = (
        0.55 * _robust_norm(
            (
                _num_on_index(out, "candidate_cell_count")
                if "candidate_cell_count" in out.columns
                else _num_on_index(out, "unique_cells", 1.0)
            ).fillna(1.0)
        )
        + 0.45 * _robust_norm(_num_on_index(out, "unique_sites", 1.0).fillna(1.0))
    ).clip(0.0, 1.0)
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
    sector_col = _first_col(base, ["sector", "sector_id", "canonical_sector_id"])
    topo_col = _first_col(base, ["topology_match_id", "canonical_physical_cell_id", "Node_Cell_ID", "nodeb_id_cell_id"])
    band_col = _first_col(base, ["band", "baseline_band", "topology_band", "Band"])
    earfcn_col = _first_col(base, ["earfcn", "earfcn_dl", "earfcn_downlink"])
    az_col = _first_col(base, ["azimuth_delta_deg", "azimuth", "antenna_azimuth"])
    if node_col is None:
        raise ValueError("Baseline rows must contain a cell identity column for Model 1")

    base["_node_cell_id"] = base[node_col].astype(str)
    base["_site_id"] = base[site_col].astype(str) if site_col else base["_node_cell_id"]
    base["_sector_id"] = base[sector_col].astype(str) if sector_col else base["_node_cell_id"]
    base["_topology_key"] = base[topo_col].astype(str) if topo_col else base["_node_cell_id"]
    base["_band"] = pd.to_numeric(base[band_col], errors="coerce") if band_col else np.nan
    base["_earfcn"] = pd.to_numeric(base[earfcn_col], errors="coerce") if earfcn_col else np.nan
    base["_azimuth"] = pd.to_numeric(base[az_col], errors="coerce") if az_col else np.nan
    base["_serving_score"] = _num(base, "pred_rsrp_smoothed").fillna(base["pred_rsrp"])

    geo_source_cols = [
        "clutter_class",
        "morphology_cluster",
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "los_blocked_ratio",
        "nlos_flag",
        "terrain_elevation_m",
        "terrain_slope_deg",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "azimuth_delta_deg",
    ]
    agg_spec = {
        "serving_cell_key": ("_topology_key", "first"),
        "serving_cell_id": ("_node_cell_id", "first"),
        "serving_site_id": ("_site_id", "first"),
        "serving_sector": ("_sector_id", "first"),
        "serving_band": ("_band", "first"),
        "serving_earfcn": ("_earfcn", "first"),
        "serving_azimuth": ("_azimuth", "first"),
        "current_rsrp": ("pred_rsrp", "mean"),
        "current_rsrq": ("pred_rsrq", "mean"),
        "current_sinr": ("pred_sinr", "mean"),
        "candidate_rsrp": ("pred_rsrp", "mean"),
        "candidate_rsrq": ("pred_rsrq", "mean"),
        "candidate_sinr": ("pred_sinr", "mean"),
        "candidate_rsrp_max": ("pred_rsrp", "max"),
        "serving_score": ("_serving_score", "mean"),
        "sample_count": ("pred_rsrp", "size"),
        "grid_row": ("grid_row", "first"),
        "grid_col": ("grid_col", "first"),
        "grid_centroid_lat": ("lat", "mean"),
        "grid_centroid_lon": ("lon", "mean"),
    }
    for col in geo_source_cols:
        if col in base.columns:
            agg_spec[col] = (col, "first")

    cell_grid = base.groupby(["grid_id", "_node_cell_id"], dropna=False).agg(**agg_spec).reset_index()
    cell_grid = cell_grid.sort_values(["grid_id", "serving_score"], ascending=[True, False])
    cell_grid["rank"] = cell_grid.groupby("grid_id").cumcount() + 1

    serving = cell_grid.loc[cell_grid["rank"].eq(1)].copy()
    neighbor1 = cell_grid.loc[cell_grid["rank"].eq(2), ["grid_id", "serving_band", "candidate_rsrp", "candidate_sinr"]].rename(
        columns={"serving_band": "neighbor1_band", "candidate_rsrp": "neighbor1_rsrp", "candidate_sinr": "neighbor1_sinr"}
    )
    neighbor2 = cell_grid.loc[cell_grid["rank"].eq(3), ["grid_id", "candidate_rsrp", "candidate_sinr"]].rename(
        columns={"candidate_rsrp": "neighbor2_rsrp", "candidate_sinr": "neighbor2_sinr"}
    )
    grouped = serving.merge(neighbor1, on="grid_id", how="left").merge(neighbor2, on="grid_id", how="left")

    grid_summary = cell_grid.groupby("grid_id", as_index=False).agg(
        candidate_cell_count=("serving_cell_id", "nunique"),
        unique_sites=("serving_site_id", "nunique"),
        total_prediction_samples=("sample_count", "sum"),
        mean_candidate_rsrp=("candidate_rsrp", "mean"),
        max_candidate_rsrp=("candidate_rsrp_max", "max"),
        mean_candidate_rsrq=("candidate_rsrq", "mean"),
        mean_candidate_sinr=("candidate_sinr", "mean"),
        std_candidate_rsrp=("candidate_rsrp", "std"),
        std_candidate_sinr=("candidate_sinr", "std"),
    )
    grouped = grouped.merge(grid_summary, on="grid_id", how="left")

    band_counts = (
        cell_grid.assign(band_class=cell_grid["serving_band"].map(_classify_band))
        .groupby(["grid_id", "band_class"])["sample_count"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for label in BAND_CLASS_LABELS:
        if label not in band_counts.columns:
            band_counts[label] = 0.0
    band_total = band_counts[list(BAND_CLASS_LABELS)].sum(axis=1).replace(0, np.nan)
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
    grouped = grouped.merge(
        band_counts[["grid_id", "low_band_ratio", "mid_band_ratio", "high_band_ratio", "carrier_count", "dominant_band_class"]],
        on="grid_id",
        how="left",
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

    for col in geo_source_cols:
        x_col = f"{col}_x"
        y_col = f"{col}_y"
        if x_col in grouped.columns or y_col in grouped.columns:
            left = grouped[x_col] if x_col in grouped.columns else pd.Series(np.nan, index=grouped.index)
            right = grouped[y_col] if y_col in grouped.columns else pd.Series(np.nan, index=grouped.index)
            grouped[col] = left.combine_first(right)
            grouped = grouped.drop(columns=[c for c in [x_col, y_col] if c in grouped.columns])

    grouped["grid_size_m"] = _num_on_index(grouped, "grid_size_m").fillna(25.0)
    grouped["target_grid_size_m"] = _num_on_index(grouped, "target_grid_size_m").fillna(grouped["grid_size_m"])
    grouped["target_grid_area_m2"] = _num_on_index(grouped, "target_grid_area_m2").fillna(grouped["target_grid_size_m"] ** 2)
    grouped["grid_area_m2"] = _num_on_index(grouped, "grid_area_m2").fillna(grouped["target_grid_area_m2"])
    grouped["cell_area_m2"] = grouped["grid_area_m2"]
    grouped["source_geo_tile_area_m2"] = _num_on_index(grouped, "source_geo_tile_area_m2").fillna(grouped["target_grid_area_m2"])
    grouped["bandwidth_mhz_est"] = _num_on_index(grouped, "bandwidth_mhz_est").fillna(10.0)
    grouped["rsrp_gap_to_neighbor1"] = grouped["candidate_rsrp"] - _num_on_index(grouped, "neighbor1_rsrp").fillna(-140.0)
    grouped["sinr_gap_to_neighbor1"] = grouped["candidate_sinr"] - _num_on_index(grouped, "neighbor1_sinr").fillna(-20.0)
    grouped["neighbor_interference_index"] = np.power(
        10.0,
        (_num_on_index(grouped, "neighbor1_rsrp").fillna(-140.0) - grouped["candidate_rsrp"]) / 10.0,
    )
    grouped["raw_neighbor1_rsrp"] = grouped["neighbor1_rsrp"]
    grouped["raw_neighbor1_sinr"] = grouped["neighbor1_sinr"]
    grouped["raw_neighbor_interference_index"] = grouped["neighbor_interference_index"]
    grouped["pred_rsrp_min"] = grouped["mean_candidate_rsrp"] - grouped["std_candidate_rsrp"].fillna(0.0)
    grouped["pred_rsrp_max"] = grouped["max_candidate_rsrp"]
    grouped["pred_rsrp_std"] = grouped["std_candidate_rsrp"].fillna(0.0)
    grouped["pred_sinr_std"] = grouped["std_candidate_sinr"].fillna(0.0)
    grouped["measurement_count"] = grouped["total_prediction_samples"]
    grouped["interference_gap_db"] = grouped["rsrp_gap_to_neighbor1"]
    grouped["interference_ratio_linear"] = np.power(10.0, grouped["interference_gap_db"].fillna(0) / 10.0)
    grouped["rf_load_pressure_proxy"] = (
        0.45 * grouped["neighbor_interference_index"].fillna(0.0).clip(0.0, 3.0)
        + 0.20 * (_num_on_index(grouped, "candidate_cell_count", 1.0).fillna(1.0).clip(1.0, 12.0) / 12.0)
    ).clip(0.0, 2.2)
    grouped["road_density"] = _num_on_index(grouped, "road_density").fillna(
        _num_on_index(grouped, "road_length_m", 0.0).fillna(0.0) / grouped["target_grid_area_m2"].replace(0, np.nan)
    )
    grouped["building_area_sum_m2"] = _num_on_index(grouped, "building_area_sum_m2").fillna(
        _num_on_index(grouped, "building_area_ratio", 0.0).fillna(0.0) * grouped["target_grid_area_m2"]
    )
    grouped["building_count_calc"] = _num_on_index(grouped, "building_count_calc").fillna(_num_on_index(grouped, "building_count", 0.0))
    grouped["building_area_sum_m2_calc"] = _num_on_index(grouped, "building_area_sum_m2_calc").fillna(grouped["building_area_sum_m2"])
    grouped["avg_building_area_m2_calc"] = _num_on_index(grouped, "avg_building_area_m2_calc").fillna(
        grouped["building_area_sum_m2_calc"] / grouped["building_count_calc"].replace(0, np.nan)
    )
    grouped["avg_building_area_m2"] = _num_on_index(grouped, "avg_building_area_m2").fillna(grouped["avg_building_area_m2_calc"])
    grouped["park_open_area"] = _num_on_index(grouped, "park_open_area", 0.0).fillna(0.0)
    grouped["open_area_ratio"] = _num_on_index(grouped, "open_area_ratio", 0.0).fillna(0.0)
    grouped["mall_presence"] = _num_on_index(grouped, "mall_presence", 0.0).fillna(0.0)
    grouped["metro_presence"] = _num_on_index(grouped, "metro_presence", 0.0).fillna(0.0)
    grouped["serving_sample_count"] = _num_on_index(grouped, "serving_sample_count").fillna(grouped["sample_count"])
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
    grouped["geo_transition_source_score"] = 0.0
    grouped["geo_impact_zone_score"] = 0.0
    grouped["geo_impact_nearby_flag"] = 0.0
    grouped["building_growth_ratio"] = 0.0
    grouped["road_growth_ratio"] = 0.0
    grouped["topology_serving_changed_flag"] = 0.0
    grouped["topology_band_changed_flag"] = 0.0
    grouped["rf_geo_blockage_penalty_db"] = 0.0
    grouped["future_rsrp_observed_from_bucket"] = 0.0
    grouped["future_sinr_observed_from_bucket"] = 0.0
    grouped["future_rsrq_observed_from_bucket"] = 0.0
    grouped["geo_snapshot_mode"] = "current_project"
    grouped["geo_assignment_source"] = "python_bridge"
    grouped["geo_assignment_method"] = "baseline_grid_aggregate"
    grouped = _add_current_state_pressure_features(grouped)

    for col in numeric_features:
        if col not in grouped.columns:
            grouped[col] = 0.0
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce")
    for col in categorical_features:
        if col not in grouped.columns:
            grouped[col] = "UNKNOWN"
        grouped[col] = grouped[col].fillna("UNKNOWN").astype(str)

    return grouped
