from __future__ import annotations

import numpy as np
import pandas as pd

CLUTTER_DEMAND_WEIGHT = {
    "Dense Urban": 1.0,
    "Urban": 0.78,
    "Suburban": 0.46,
    "Rural/Open": 0.20,
    "Vegetation": 0.18,
    "Water": 0.04,
}
CLUTTER_TRANSITION_LEVEL = {
    "Water": 0,
    "Vegetation": 0,
    "Rural/Open": 0,
    "Suburban": 1,
    "Urban": 2,
    "Dense Urban": 3,
}


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {str(col).lower(): col for col in df.columns}
    for name in names:
        col = lower.get(name.lower())
        if col is not None:
            return col
    return None


def _num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
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
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max())
    if hi <= lo:
        return pd.Series(0.0, index=series.index, dtype="float64")
    return ((values.fillna(lo) - lo) / (hi - lo)).clip(0.0, 1.0)


def _rename_current_columns(cell_df: pd.DataFrame) -> pd.DataFrame:
    work = cell_df.copy()
    rename_map = {
        "input_prb_utilization_pct": "current_prb_utilization_pct",
        "input_rrc_utilization_pct": "current_rrc_utilization_pct",
        "input_rrc_connected_users": "current_rrc_connected_users",
        "input_estimated_dl_capacity_mbps": "current_estimated_dl_capacity_mbps",
        "input_estimated_offered_traffic_mbps": "current_estimated_offered_traffic_mbps",
    }
    for src, dst in rename_map.items():
        if src in work.columns and dst not in work.columns:
            work[dst] = work[src]
    return work


def _baseline_cell_aggregates(baseline_df: pd.DataFrame) -> pd.DataFrame:
    if baseline_df is None or baseline_df.empty:
        return pd.DataFrame()
    work = baseline_df.copy()
    cell_col = _first_col(work, ["Node_Cell_ID", "node_cell_id", "rf_identity_key", "cell_id"])
    if cell_col is None:
        return pd.DataFrame()
    work["_node_cell_id"] = work[cell_col].astype(str)

    for target, candidates in {
        "rsrp": ["pred_rsrp", "rsrp", "rssi"],
        "rsrq": ["pred_rsrq", "rsrq"],
        "sinr": ["pred_sinr", "sinr"],
    }.items():
        col = _first_col(work, candidates)
        work[target] = pd.to_numeric(work[col], errors="coerce") if col else np.nan

    band_col = _first_col(work, ["band", "topology_band", "Band"])
    if band_col:
        work["_band_num"] = pd.to_numeric(work[band_col], errors="coerce")
    else:
        work["_band_num"] = np.nan

    grouped = work.groupby("_node_cell_id", as_index=False).agg(
        node_cell_id=("_node_cell_id", "first"),
        rsrp_mean=("rsrp", "mean"),
        rsrq_mean=("rsrq", "mean"),
        sinr_mean=("sinr", "mean"),
        corrected_rsrp_mean=("rsrp", "mean"),
        corrected_rsrq_mean=("rsrq", "mean"),
        corrected_sinr_mean=("sinr", "mean"),
        sample_count=("rsrp", "size"),
        carrier_count=("_band_num", "nunique"),
    )
    grouped["low_band_ratio"] = 0.0
    grouped["mid_band_ratio"] = 1.0
    grouped["high_band_ratio"] = 0.0
    if band_col:
        band_counts = work.groupby("_node_cell_id")["_band_num"].count().replace(0, np.nan)
        low = work.groupby("_node_cell_id")["_band_num"].apply(lambda s: float((s <= 900).sum()))
        mid = work.groupby("_node_cell_id")["_band_num"].apply(lambda s: float(((s > 900) & (s <= 2100)).sum()))
        high = work.groupby("_node_cell_id")["_band_num"].apply(lambda s: float((s > 2100).sum()))
        grouped = grouped.set_index("node_cell_id")
        grouped["low_band_ratio"] = (low / band_counts).reindex(grouped.index).fillna(0).to_numpy()
        grouped["mid_band_ratio"] = (mid / band_counts).reindex(grouped.index).fillna(1).to_numpy()
        grouped["high_band_ratio"] = (high / band_counts).reindex(grouped.index).fillna(0).to_numpy()
        grouped = grouped.reset_index()
    return grouped


def _mode_string(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return "UNKNOWN"
    return str(clean.mode().iloc[0])


def _geo_cell_aggregates(geo_df: pd.DataFrame) -> pd.DataFrame:
    if geo_df is None or geo_df.empty:
        return pd.DataFrame()
    work = geo_df.copy()
    cell_col = _first_col(work, ["nodeb_id_cell_id", "Node_Cell_ID", "node_cell_id", "rf_identity_key", "cell_id"])
    if cell_col is None:
        return pd.DataFrame()
    work["node_cell_id"] = work[cell_col].astype(str)

    numeric_cols = {
        "building_count",
        "building_area_ratio",
        "road_length_m",
        "road_density",
        "mall_presence",
        "metro_presence",
        "park_open_area",
        "open_area_ratio",
        "green_ratio",
        "water_ratio",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "azimuth_delta_deg",
        "nlos_flag",
        "terrain_slope_deg",
    }
    for col in numeric_cols:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = work.groupby("node_cell_id", as_index=False).agg(
        building_count=("building_count", "mean"),
        building_area_ratio=("building_area_ratio", "mean"),
        road_length_m=("road_length_m", "mean"),
        road_density=("road_density", "mean"),
        mall_presence=("mall_presence", "max"),
        metro_presence=("metro_presence", "max"),
        park_open_area=("park_open_area", "mean"),
        open_area_ratio=("open_area_ratio", "mean"),
        green_ratio=("green_ratio", "mean"),
        water_ratio=("water_ratio", "mean"),
        site_count_250m=("site_count_250m", "mean"),
        site_count_500m=("site_count_500m", "mean"),
        serving_distance_m=("serving_distance_m", "mean"),
        nearest_site_distance_m=("nearest_site_distance_m", "mean"),
        mean_nearest3_site_distance_m=("mean_nearest3_site_distance_m", "mean"),
        azimuth_delta_deg=("azimuth_delta_deg", "mean"),
        nlos_flag=("nlos_flag", "mean"),
        terrain_slope_deg=("terrain_slope_deg", "mean"),
        clutter_class=("clutter_class", _mode_string) if "clutter_class" in work.columns else ("node_cell_id", _mode_string),
    )
    return grouped


def _geo_project_context(geo_df: pd.DataFrame) -> dict[str, float | str]:
    cell_geo = _geo_cell_aggregates(geo_df)
    if cell_geo.empty:
        return {}
    context: dict[str, float | str] = {}
    for col in [
        "building_count",
        "building_area_ratio",
        "road_density",
        "mall_presence",
        "metro_presence",
        "park_open_area",
        "open_area_ratio",
        "green_ratio",
        "water_ratio",
    ]:
        if col in cell_geo.columns:
            context[col] = float(pd.to_numeric(cell_geo[col], errors="coerce").mean())
    if "clutter_class" in cell_geo.columns and not cell_geo["clutter_class"].dropna().empty:
        context["clutter_class"] = _mode_string(cell_geo["clutter_class"])
    return context


def build_model2_feature_frame(
    cell_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    work = _rename_current_columns(cell_df)
    if "Node_Cell_ID" not in work.columns:
        raise ValueError("Model 2 cell input must contain Node_Cell_ID")
    work["node_cell_id"] = work["Node_Cell_ID"].astype(str)

    baseline_agg = _baseline_cell_aggregates(baseline_df)
    if not baseline_agg.empty:
        drop_cols = [c for c in baseline_agg.columns if c in work.columns and c != "node_cell_id"]
        work = work.drop(columns=drop_cols, errors="ignore").merge(baseline_agg, on="node_cell_id", how="left")

    geo_agg = _geo_cell_aggregates(geo_df)
    if not geo_agg.empty:
        drop_cols = [c for c in geo_agg.columns if c in work.columns and c != "node_cell_id"]
        work = work.drop(columns=drop_cols, errors="ignore").merge(geo_agg, on="node_cell_id", how="left")

    for col, value in _geo_project_context(geo_df).items():
        if col not in work.columns or work[col].isna().all():
            work[col] = value

    current_prb = _num(work, "current_prb_utilization_pct", 0).fillna(0)
    current_rrc = _num(work, "current_rrc_utilization_pct", 0).fillna(0)
    capacity = _num(work, "current_estimated_dl_capacity_mbps", 1).replace(0, np.nan)
    traffic = _num(work, "current_estimated_offered_traffic_mbps", 0).fillna(0)

    work["prb_pressure_est"] = current_prb
    work["prb_outlier_flag"] = (current_prb > 70).astype(float)
    building_norm = _robust_norm(_num(work, "building_area_ratio", 0))
    building_count_norm = _robust_norm(_num(work, "building_count", 0))
    road_norm = _robust_norm(_num(work, "road_density", np.nan).combine_first(_num(work, "road_length_m", 0)))
    site_density_norm = _robust_norm(_num(work, "site_count_500m", 0))
    clutter_level = work.get("clutter_class", pd.Series("UNKNOWN", index=work.index)).map(CLUTTER_TRANSITION_LEVEL).fillna(1).astype(float)
    clutter_weight = work.get("clutter_class", pd.Series("UNKNOWN", index=work.index)).map(CLUTTER_DEMAND_WEIGHT).fillna(0.35).astype(float)
    activity_anchor = (
        (_num(work, "mall_presence", 0).fillna(0).gt(0).astype(float) * 0.45)
        + (_num(work, "metro_presence", 0).fillna(0).gt(0).astype(float) * 0.35)
        + (road_norm.ge(0.60).astype(float) * 0.20)
    ).clip(0.0, 1.0)
    work["geo_demand_score"] = (
        (
            0.30 * clutter_weight
            + 0.24 * building_norm
            + 0.16 * building_count_norm
            + 0.16 * road_norm
            + 0.08 * site_density_norm
            + 0.14 * activity_anchor
        ).clip(0, 1)
        * 100.0
    ).clip(0, 100)
    work["kpi_demand_score"] = np.maximum(current_prb, current_rrc)
    work["development_pressure_score"] = (
        0.36 * building_norm
        + 0.24 * building_count_norm
        + 0.20 * road_norm
        + 0.12 * site_density_norm
        + 0.08 * activity_anchor
    ).clip(0, 1) * 100.0
    work["growth_zone_score"] = work["geo_demand_score"] * 0.6 + work["kpi_demand_score"] * 0.4
    congestion_pressure = ((work["kpi_demand_score"] - 55.0) / 45.0).clip(0.0, 1.0)
    growth_intensity = (
        0.38 * (work["development_pressure_score"] / 100.0)
        + 0.24 * (work["geo_demand_score"] / 100.0)
        + 0.22 * congestion_pressure
        + 0.16 * activity_anchor
    ).clip(0.0, 1.0)
    work["growth_rate"] = (0.03 + 0.17 * growth_intensity).clip(0.03, 0.20)
    work["clutter_transition_flag"] = (clutter_level >= 2).astype(float)
    work["clutter_upgrade_score"] = ((work["development_pressure_score"] / 100.0) * (clutter_level / 3.0)).clip(0.0, 1.0)
    work["building_growth_ratio"] = (0.03 + 0.42 * building_norm * growth_intensity).clip(0.0, 0.45)
    work["road_growth_ratio"] = (0.02 + 0.30 * road_norm * growth_intensity).clip(0.0, 0.32)
    work["activity_anchor_score"] = (activity_anchor * 100.0).clip(0, 100)
    work["capacity_context_score"] = (traffic / capacity).fillna(0).clip(0, 2) * 50.0
    work["capacity_gap_score"] = (work["kpi_demand_score"] - 70).clip(lower=0)

    work["cqi_mean"] = (_num(work, "sinr_mean", 10).fillna(10) / 2.0 + 7.0).clip(1, 15)
    work["dl_tpt_mean"] = _num(work, "current_estimated_dl_capacity_mbps", 0).fillna(0)
    work["ul_tpt_mean"] = work["dl_tpt_mean"] * 0.25
    work["bandwidth_mhz_est"] = 10.0
    work["dominant_band_class"] = np.select(
        [
            _num(work, "low_band_ratio", 0).fillna(0) >= _num(work, "mid_band_ratio", 1).fillna(1),
            _num(work, "high_band_ratio", 0).fillna(0) >= _num(work, "mid_band_ratio", 1).fillna(1),
        ],
        ["LOW_BAND", "HIGH_BAND"],
        default="MID_BAND",
    )

    for col in numeric_features:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    for col in categorical_features:
        if col not in work.columns:
            work[col] = "UNKNOWN"
        work[col] = work[col].fillna("UNKNOWN").astype(str)

    return work
