from __future__ import annotations

import numpy as np
import pandas as pd


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


def _geo_project_context(geo_df: pd.DataFrame) -> dict[str, float | str]:
    if geo_df is None or geo_df.empty:
        return {}
    context: dict[str, float | str] = {}
    numeric_cols = [
        "building_count",
        "building_area_ratio",
        "road_density",
        "mall_presence",
        "metro_presence",
        "park_open_area",
        "open_area_ratio",
        "green_ratio",
        "water_ratio",
    ]
    for col in numeric_cols:
        if col in geo_df.columns:
            context[col] = float(pd.to_numeric(geo_df[col], errors="coerce").mean())
    for col in ["clutter_class"]:
        if col in geo_df.columns and not geo_df[col].dropna().empty:
            context[col] = str(geo_df[col].dropna().astype(str).mode().iloc[0])
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

    for col, value in _geo_project_context(geo_df).items():
        if col not in work.columns or work[col].isna().all():
            work[col] = value

    current_prb = _num(work, "current_prb_utilization_pct", 0).fillna(0)
    current_rrc = _num(work, "current_rrc_utilization_pct", 0).fillna(0)
    capacity = _num(work, "current_estimated_dl_capacity_mbps", 1).replace(0, np.nan)
    traffic = _num(work, "current_estimated_offered_traffic_mbps", 0).fillna(0)

    work["prb_pressure_est"] = current_prb
    work["prb_outlier_flag"] = (current_prb > 70).astype(float)
    work["growth_rate"] = 0.08
    work["geo_demand_score"] = (
        _num(work, "building_area_ratio", 0).fillna(0) * 35.0
        + _num(work, "road_density", 0).fillna(0) * 20.0
        + _num(work, "mall_presence", 0).fillna(0) * 15.0
        + _num(work, "metro_presence", 0).fillna(0) * 15.0
    ).clip(0, 100)
    work["kpi_demand_score"] = np.maximum(current_prb, current_rrc)
    work["development_pressure_score"] = work["geo_demand_score"]
    work["growth_zone_score"] = work["geo_demand_score"] * 0.6 + work["kpi_demand_score"] * 0.4
    work["clutter_transition_flag"] = 0.0
    work["clutter_upgrade_score"] = 0.0
    work["building_growth_ratio"] = 0.0
    work["road_growth_ratio"] = 0.0
    work["activity_anchor_score"] = work["geo_demand_score"]
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
