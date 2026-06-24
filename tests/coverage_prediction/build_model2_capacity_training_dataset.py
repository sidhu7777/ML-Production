from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


REQUIRED_ARCHIVE_MEMBERS = (
    "grid_kpi_timeseries.csv",
    "bucket_grid_geo_features.csv",
    "bucket_corrected_prediction_grid.csv",
    "summary.json",
)

BUCKET_ORDER = {"PART_1": 1, "PART_2": 2, "PART_3": 3}

CLUTTER_DEMAND_WEIGHT = {
    "Dense Urban": 1.0,
    "Urban": 0.78,
    "Suburban": 0.46,
    "Rural/Open": 0.2,
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

PRB_PRESSURE_CAP = 40.0
PRB_OUTLIER_THRESHOLD = 100.0

BAND_CLASS_LABELS = ("LOW_BAND", "MID_BAND", "HIGH_BAND")
BAND_FEATURE_COLUMNS = [
    "grid_id",
    "time_bucket",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "dominant_band_class",
    "carrier_count",
]
DEFAULT_BUCKET_BAND_MIX_PLAN = {
    "PART_1": {900: 0.15, 1800: 0.69, 850: 0.15, 2300: 0.01},
    "PART_2": {900: 0.11, 1800: 0.76, 850: 0.10, 2300: 0.03},
    "PART_3": {900: 0.08, 1800: 0.80, 850: 0.07, 2300: 0.05},
}


def _safe_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
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
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        hi = float(valid.max())
        lo = float(valid.min())
    if hi <= lo:
        return pd.Series(0.0, index=series.index, dtype="float64")
    return ((values.fillna(lo) - lo) / (hi - lo)).clip(0.0, 1.0)


def _archive_root(archive_path: Path) -> str:
    listed = subprocess.check_output(["tar", "-tf", str(archive_path)], text=True)
    first_file = next((line for line in listed.splitlines() if "/" in line and not line.endswith("/")), "")
    if not first_file:
        raise RuntimeError(f"No files found in archive: {archive_path}")
    return first_file.split("/", 1)[0]


def _extract_required_members(archive_path: Path, extract_dir: Path) -> Path:
    root = _archive_root(archive_path)
    target_root = extract_dir / root
    missing = [name for name in REQUIRED_ARCHIVE_MEMBERS if not (target_root / name).exists()]
    if not missing:
        return target_root

    extract_dir.mkdir(parents=True, exist_ok=True)
    members = [f"{root}/{name}" for name in REQUIRED_ARCHIVE_MEMBERS]
    subprocess.check_call(["tar", "-xf", str(archive_path), "-C", str(extract_dir), *members])
    return target_root


def _growth_component(current: pd.Series, previous: pd.Series) -> pd.Series:
    current = pd.to_numeric(current, errors="coerce")
    previous = pd.to_numeric(previous, errors="coerce")
    denom = previous.abs().replace(0, np.nan)
    raw = (current - previous) / denom
    cold_start = previous.isna() & current.notna() & (current > 0)
    raw = raw.where(~cold_start, 0.0)
    return raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 3.0)


def _derive_growth_rate(df: pd.DataFrame) -> pd.Series:
    work = df.sort_values(["grid_id", "bucket_seq"]).copy()
    grouped = work.groupby("grid_id", sort=False)
    prev_tpt = grouped["dl_tpt_mean"].shift(1)
    prb_col = "prb_pressure_est" if "prb_pressure_est" in work.columns else "estimated_prb_mean"
    prev_prb = grouped[prb_col].shift(1)
    prev_samples = grouped["sample_count"].shift(1)

    tpt_growth = _growth_component(work["dl_tpt_mean"], prev_tpt)
    prb_growth = _growth_component(work[prb_col], prev_prb)
    sample_growth = _growth_component(work["sample_count"], prev_samples)
    growth = (0.40 * tpt_growth) + (0.35 * prb_growth) + (0.25 * sample_growth)
    growth = growth.clip(-0.75, 2.5)
    return growth.reindex(work.index).sort_index()


def _empty_band_features() -> pd.DataFrame:
    return pd.DataFrame(columns=BAND_FEATURE_COLUMNS)


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

    # Common LTE/NR ARFCN ranges seen in Indian planning exports.
    # We classify by approximate downlink carrier family, not by bandwidth.
    if 2400 <= earfcn <= 2649:  # LTE band 5, ~850 MHz
        return "LOW_BAND"
    if 3450 <= earfcn <= 3799:  # LTE band 8, ~900 MHz
        return "LOW_BAND"
    if 1200 <= earfcn <= 1949:  # LTE band 3, ~1800 MHz
        return "MID_BAND"
    if 0 <= earfcn <= 599:  # LTE band 1, ~2100 MHz
        return "HIGH_BAND"
    if 38650 <= earfcn <= 39649:  # LTE band 40, ~2300 MHz
        return "HIGH_BAND"
    if 39650 <= earfcn <= 41589:  # LTE band 41, ~2500 MHz
        return "HIGH_BAND"
    if earfcn >= 600000:  # NR ARFCN, usually capacity/high-band in these exports
        return "HIGH_BAND"

    # Some sources store MHz in an EARFCN-named column.
    if 600 <= earfcn <= 6000:
        return _classify_mhz(earfcn)
    return "UNKNOWN"


def _classify_band_value(value: object, source_col: str) -> str:
    source = str(source_col).lower()
    if "earfcn" in source:
        return _classify_lte_earfcn(value)
    return _classify_mhz(value)


def _band_source_column(df: pd.DataFrame) -> str | None:
    for candidate in [
        "band",
        "frequency_mhz",
        "frequency",
        "downlink_frequency",
        "uplink_center_frequency",
        "earfcn",
        "dominant_earfcn",
        "corrected_dominant_earfcn",
    ]:
        if candidate in df.columns:
            return candidate
    return None


def _derive_band_features_from_rows(source_df: pd.DataFrame) -> pd.DataFrame:
    if source_df.empty or "grid_id" not in source_df.columns or "time_bucket" not in source_df.columns:
        return _empty_band_features()

    work = source_df.copy()
    work["grid_id"] = pd.to_numeric(work["grid_id"], errors="coerce").astype("Int64")
    work["time_bucket"] = work["time_bucket"].astype(str)
    band_source = _band_source_column(work)
    if band_source is None:
        out = work[["grid_id", "time_bucket"]].drop_duplicates().copy()
        out["low_band_ratio"] = 0.0
        out["mid_band_ratio"] = 0.0
        out["high_band_ratio"] = 0.0
        out["dominant_band_class"] = "UNKNOWN"
        out["carrier_count"] = 0
        return out

    work["band_class"] = work[band_source].map(lambda value: _classify_band_value(value, band_source))
    work = work.dropna(subset=["grid_id"])

    counts = (
        work.groupby(["grid_id", "time_bucket", "band_class"], dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
    )
    pivot = counts.pivot_table(
        index=["grid_id", "time_bucket"],
        columns="band_class",
        values="row_count",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    pivot.columns.name = None
    for label in BAND_CLASS_LABELS:
        if label not in pivot.columns:
            pivot[label] = 0.0

    total = pivot[list(BAND_CLASS_LABELS)].sum(axis=1).replace(0, np.nan)
    pivot["low_band_ratio"] = (pivot["LOW_BAND"] / total).fillna(0.0)
    pivot["mid_band_ratio"] = (pivot["MID_BAND"] / total).fillna(0.0)
    pivot["high_band_ratio"] = (pivot["HIGH_BAND"] / total).fillna(0.0)

    def _dominant_label(row: pd.Series) -> str:
        ratios = {
            "LOW_BAND": float(row["low_band_ratio"]),
            "MID_BAND": float(row["mid_band_ratio"]),
            "HIGH_BAND": float(row["high_band_ratio"]),
        }
        best_label = max(ratios, key=ratios.get)
        return best_label if ratios[best_label] > 0 else "UNKNOWN"

    pivot["dominant_band_class"] = pivot.apply(_dominant_label, axis=1)
    pivot["carrier_count"] = ((pivot["LOW_BAND"] > 0).astype(int) + (pivot["MID_BAND"] > 0).astype(int) + (pivot["HIGH_BAND"] > 0).astype(int))

    keep = [
        "grid_id",
        "time_bucket",
        "low_band_ratio",
        "mid_band_ratio",
        "high_band_ratio",
        "dominant_band_class",
        "carrier_count",
    ]
    return pivot[keep]


def _derive_band_features(pred_df: pd.DataFrame, kpi_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    pred_band_features = _derive_band_features_from_rows(pred_df)
    if not pred_band_features.empty and (pred_band_features[["low_band_ratio", "mid_band_ratio", "high_band_ratio"]].sum(axis=1) > 0).any():
        return pred_band_features, "prediction_grid"

    kpi_band_features = _derive_band_features_from_rows(kpi_df)
    if not kpi_band_features.empty and (kpi_band_features[["low_band_ratio", "mid_band_ratio", "high_band_ratio"]].sum(axis=1) > 0).any():
        return kpi_band_features, "grid_kpi_timeseries"

    if not pred_band_features.empty:
        return pred_band_features, "prediction_grid_empty"
    if not kpi_band_features.empty:
        return kpi_band_features, "grid_kpi_timeseries_empty"
    return _empty_band_features(), "missing"


def _normalise_band_mix_plan(raw_plan: object) -> Dict[str, Dict[float, float]]:
    if not isinstance(raw_plan, dict):
        raw_plan = DEFAULT_BUCKET_BAND_MIX_PLAN
    out: Dict[str, Dict[float, float]] = {}
    for bucket, plan in raw_plan.items():
        if not isinstance(plan, dict):
            continue
        bucket_plan: Dict[float, float] = {}
        for band, ratio in plan.items():
            try:
                band_value = float(band)
                ratio_value = float(ratio)
            except (TypeError, ValueError):
                continue
            if math.isfinite(band_value) and math.isfinite(ratio_value) and ratio_value > 0:
                bucket_plan[band_value] = ratio_value
        if bucket_plan:
            out[str(bucket)] = bucket_plan
    return out or _normalise_band_mix_plan(DEFAULT_BUCKET_BAND_MIX_PLAN)


def _bucket_mix_ratios(bucket_label: object, band_mix_plan: Dict[str, Dict[float, float]]) -> Dict[str, float]:
    plan = band_mix_plan.get(str(bucket_label), {})
    totals = {label: 0.0 for label in BAND_CLASS_LABELS}
    for band, ratio in plan.items():
        band_class = _classify_mhz(band)
        if band_class in totals:
            totals[band_class] += float(ratio)
    total = sum(totals.values())
    if total <= 0:
        return {label: 0.0 for label in BAND_CLASS_LABELS}
    return {label: totals[label] / total for label in BAND_CLASS_LABELS}


def _apply_bucket_band_mix_fallback(
    df: pd.DataFrame,
    source_summary: Dict[str, object],
) -> Tuple[pd.DataFrame, int, str]:
    out = df.copy()
    raw_plan = source_summary.get("band_mix_plan") or source_summary.get("topology_plan", {}).get("band_mix_plan")
    plan = _normalise_band_mix_plan(raw_plan)
    source = "source_summary_band_mix_plan" if raw_plan else "default_current_bucket_band_mix_plan"

    ratio_sum = (
        pd.to_numeric(out.get("low_band_ratio"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("mid_band_ratio"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("high_band_ratio"), errors="coerce").fillna(0.0)
    )
    missing_mask = ratio_sum.le(0.0)
    if not missing_mask.any():
        return out, 0, source

    filled = 0
    for bucket_label, bucket_idx in out.loc[missing_mask].groupby(out.loc[missing_mask, "time_bucket"].astype(str)).groups.items():
        ratios = _bucket_mix_ratios(bucket_label, plan)
        if sum(ratios.values()) <= 0:
            continue
        idx = list(bucket_idx)
        out.loc[idx, "low_band_ratio"] = ratios["LOW_BAND"]
        out.loc[idx, "mid_band_ratio"] = ratios["MID_BAND"]
        out.loc[idx, "high_band_ratio"] = ratios["HIGH_BAND"]
        best_label = max(ratios, key=ratios.get)
        out.loc[idx, "dominant_band_class"] = best_label if ratios[best_label] > 0 else "UNKNOWN"
        out.loc[idx, "carrier_count"] = int(sum(1 for value in ratios.values() if value > 0))
        filled += len(idx)
    return out, int(filled), source


def _ratio_vs_first_bucket(df: pd.DataFrame, value_col: str) -> pd.Series:
    work = df.sort_values(["grid_id", "bucket_seq"]).copy()
    current = pd.to_numeric(work[value_col], errors="coerce")
    baseline = work.groupby("grid_id", sort=False)[value_col].transform("first")
    baseline = pd.to_numeric(baseline, errors="coerce")
    denom = baseline.abs().replace(0, np.nan)
    ratio = (current - baseline) / denom
    cold_start = baseline.fillna(0.0).eq(0.0) & current.fillna(0.0).gt(0.0)
    ratio = ratio.where(~cold_start, 1.0)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 5.0)
    return ratio.reindex(work.index).sort_index()


def _add_model2_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bucket_seq"] = out["time_bucket"].astype(str).map(BUCKET_ORDER).fillna(0).astype(int)

    for col in [
        "sample_count",
        "building_count",
        "building_area_ratio",
        "road_density",
        "road_length_m",
        "park_open_area",
        "open_area_ratio",
        "mall_presence",
        "metro_presence",
    ]:
        out[col] = _safe_numeric(out, col, default=0.0)

    for col in [
        "dl_tpt_mean",
        "ul_tpt_mean",
        "cqi_mean",
        "estimated_prb_mean",
        "sinr_mean",
        "rsrp_mean",
        "rsrq_mean",
        "bandwidth_mhz_est",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["low_band_ratio", "mid_band_ratio", "high_band_ratio", "carrier_count"]:
        out[col] = _safe_numeric(out, col, default=0.0)
    out["dominant_band_class"] = out.get("dominant_band_class", pd.Series("UNKNOWN", index=out.index)).fillna("UNKNOWN").astype(str)

    raw_prb = out["estimated_prb_mean"]
    out["prb_pressure_est"] = raw_prb.clip(lower=0.0, upper=PRB_PRESSURE_CAP).fillna(0.0).round(6)
    out["prb_outlier_flag"] = (raw_prb > PRB_OUTLIER_THRESHOLD).fillna(False).astype(int)

    clutter_weight = (
        out.get("clutter_class", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .map(CLUTTER_DEMAND_WEIGHT)
        .fillna(0.35)
        .astype("float64")
    )

    building_count_norm = _robust_norm(out["building_count"])
    building_area_norm = _robust_norm(out["building_area_ratio"])
    road_norm = _robust_norm(out["road_density"].combine_first(out["road_length_m"]))
    place_anchor = ((out["mall_presence"] > 0).astype(float) * 0.10) + ((out["metro_presence"] > 0).astype(float) * 0.08)
    open_space_discount = (0.45 * _robust_norm(out["park_open_area"])) + (0.30 * _robust_norm(out["open_area_ratio"]))
    geo_demand_score = (
        (0.34 * clutter_weight)
        + (0.24 * building_area_norm)
        + (0.18 * building_count_norm)
        + (0.18 * road_norm)
        + place_anchor
        - open_space_discount
    ).clip(0.0, 1.0)

    throughput_present = out["dl_tpt_mean"].notna() & (out["dl_tpt_mean"] > 0)
    prb_present = out["prb_pressure_est"] > 0
    throughput_norm = _robust_norm(out["dl_tpt_mean"]).where(throughput_present, 0.0)
    prb_norm = _robust_norm(out["prb_pressure_est"]).where(prb_present, 0.0)
    sample_norm = _robust_norm(np.log1p(out["sample_count"].clip(lower=0.0)))
    cqi_valid = out["cqi_mean"].where(out["cqi_mean"].between(1, 15), np.nan)
    cqi_pressure = ((15.0 - cqi_valid) / 14.0).fillna(0.0).clip(0.0, 1.0)
    sinr_pressure = ((8.0 - out["sinr_mean"]) / 18.0).where(out["sinr_mean"].notna(), 0.0).clip(0.0, 1.0)
    kpi_demand_score = (
        (0.34 * throughput_norm)
        + (0.28 * prb_norm)
        + (0.24 * sample_norm)
        + (0.08 * cqi_pressure)
        + (0.06 * sinr_pressure)
    ).clip(0.0, 1.0)

    base_growth = _derive_growth_rate(out)
    geo_growth = (
        (0.40 * clutter_weight)
        + (0.30 * building_area_norm)
        + (0.20 * building_count_norm)
        + (0.10 * road_norm)
    ).clip(0.0, 1.0)
    out["growth_rate"] = ((0.60 * base_growth) + (0.40 * geo_growth)).clip(-0.75, 2.5)
    growth_signal = ((out["growth_rate"] + 0.75) / 3.25).clip(0.0, 1.0)

    out["geo_demand_score"] = (geo_demand_score * 100.0).round(3)
    out["kpi_demand_score"] = (kpi_demand_score * 100.0).round(3)
    activity_anchor_score = (
        ((out["mall_presence"] > 0).astype(float) * 0.45)
        + ((out["metro_presence"] > 0).astype(float) * 0.35)
        + ((road_norm >= 0.60).astype(float) * 0.20)
    ).clip(0.0, 1.0)

    development_pressure_score = (
        (0.24 * building_count_norm)
        + (0.26 * building_area_norm)
        + (0.16 * road_norm)
        + (0.14 * activity_anchor_score)
        + (0.20 * growth_signal)
    ).clip(0.0, 1.0)

    clutter_level = (
        out.get("clutter_class", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .map(CLUTTER_TRANSITION_LEVEL)
        .fillna(0)
        .astype(int)
    )
    baseline_clutter_level = (
        out.assign(_clutter_level=clutter_level)
        .sort_values(["grid_id", "bucket_seq"])
        .groupby("grid_id", sort=False)["_clutter_level"]
        .transform("first")
        .reindex(out.index)
        .fillna(0)
        .astype(int)
    )
    out["clutter_transition_flag"] = clutter_level.ne(baseline_clutter_level).astype(int)
    out["clutter_upgrade_score"] = np.where(clutter_level > baseline_clutter_level, clutter_level, 0).astype(float)

    out["building_growth_ratio"] = _ratio_vs_first_bucket(out, "building_area_ratio").round(6)
    out["road_growth_ratio"] = _ratio_vs_first_bucket(out, "road_density").round(6)

    bandwidth_norm = _robust_norm(out["bandwidth_mhz_est"].replace(0, np.nan).fillna(10.0))
    low_band_ratio = out["low_band_ratio"].clip(0.0, 1.0)
    mid_band_ratio = out["mid_band_ratio"].clip(0.0, 1.0)
    high_band_ratio = out["high_band_ratio"].clip(0.0, 1.0)
    carrier_count_norm = _robust_norm(out["carrier_count"])
    spare_capacity_signal = (1.0 - prb_norm).clip(0.0, 1.0)
    capacity_context_score = (
        (0.22 * bandwidth_norm)
        + (0.18 * spare_capacity_signal)
        + (0.12 * throughput_norm)
        + (0.08 * (1.0 - sinr_pressure))
        + (0.14 * low_band_ratio)
        + (0.16 * mid_band_ratio)
        + (0.18 * high_band_ratio)
        + (0.10 * carrier_count_norm)
    ).clip(0.0, 1.0)

    growth_zone_score = (
        (0.38 * geo_demand_score)
        + (0.32 * development_pressure_score)
        + (0.30 * kpi_demand_score)
    ).clip(0.0, 1.0)

    out["development_pressure_score"] = (development_pressure_score * 100.0).round(3)
    out["growth_zone_score"] = (growth_zone_score * 100.0).round(3)
    out["activity_anchor_score"] = (activity_anchor_score * 100.0).round(3)
    out["capacity_context_score"] = (capacity_context_score * 100.0).round(3)

    out["demand_index"] = (
        (
            (0.35 * geo_demand_score)
            + (0.40 * kpi_demand_score)
            + (0.15 * growth_signal)
            + (0.10 * capacity_context_score)
        )
        * 100.0
    ).round(3)

    area_factor = _robust_norm(out.get("grid_area_m2", pd.Series(2500.0, index=out.index))).replace(0, 0.35)
    users_base = 2.0 + (out["demand_index"] / 100.0) * 42.0
    observed_boost = np.log1p(out["sample_count"].clip(lower=0.0)) * 0.55
    out["active_users_est"] = (users_base * (0.70 + 0.60 * area_factor) + observed_boost).clip(lower=0.0).round(3)

    traffic_from_tpt = out["dl_tpt_mean"].fillna(0.0).clip(lower=0.0)
    capacity_gap_score = (
        (0.44 * (out["demand_index"] / 100.0).clip(0.0, 1.5))
        + (0.26 * prb_norm)
        + (0.12 * sinr_pressure)
        + (0.08 * (out["clutter_upgrade_score"] / 3.0).clip(0.0, 1.0))
        + (0.10 * np.clip(out["building_growth_ratio"], 0.0, 1.0))
        - (0.25 * capacity_context_score)
    ).clip(0.0, 1.5)
    traffic_proxy = (
        (out["demand_index"] / 100.0)
        * (out["bandwidth_mhz_est"].replace(0, np.nan).fillna(10.0))
        * (1.0 + 0.35 * capacity_gap_score)
        * 2.2
    )
    out["traffic_demand_est"] = (
        (0.40 * traffic_from_tpt)
        + (0.22 * out["active_users_est"])
        + (0.18 * traffic_proxy)
        + (0.12 * out["prb_pressure_est"])
        + (0.08 * out["capacity_context_score"])
    ).clip(lower=0.0).round(3)

    out["capacity_gap_score"] = (capacity_gap_score * 100.0).round(3)
    out["growth_rate"] = out["growth_rate"].round(6)
    out["demand_feature_source"] = "geo_kpi_growth_capacity_context"
    return out


def build_dataset(archive_path: Path, output_csv: Path, work_dir: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    extracted_root = _extract_required_members(archive_path, work_dir)
    kpi_path = extracted_root / "grid_kpi_timeseries.csv"
    geo_path = extracted_root / "bucket_grid_geo_features.csv"
    pred_path = extracted_root / "bucket_corrected_prediction_grid.csv"
    summary_path = extracted_root / "summary.json"

    kpi_df = pd.read_csv(kpi_path)
    geo_df = pd.read_csv(geo_path)
    pred_df = pd.read_csv(pred_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        source_summary = json.load(handle)

    join_keys = ["grid_id", "time_bucket"]
    for frame_name, frame in [("grid_kpi_timeseries", kpi_df), ("bucket_grid_geo_features", geo_df)]:
        missing_keys = [key for key in join_keys if key not in frame.columns]
        if missing_keys:
            raise RuntimeError(f"{frame_name} is missing join keys: {missing_keys}")
        frame["grid_id"] = pd.to_numeric(frame["grid_id"], errors="coerce").astype("Int64")
        frame["time_bucket"] = frame["time_bucket"].astype(str)

    geo_keep = [
        col
        for col in [
            "grid_id",
            "time_bucket",
            "grid_size_m",
            "grid_area_m2",
            "cell_area_m2",
            "road_length_m",
            "green_ratio",
            "water_ratio",
            "building_count",
            "building_area_sum_m2",
            "avg_building_area_m2",
            "building_area_ratio",
            "park_open_area",
            "open_area_ratio",
            "mall_presence",
            "metro_presence",
            "road_density",
            "clutter_class",
            "geo_snapshot_mode",
            "geo_snapshot_source_ts",
        ]
        if col in geo_df.columns
    ]
    band_features_df, band_feature_source = _derive_band_features(pred_df, kpi_df)
    merged = kpi_df.merge(geo_df[geo_keep], on=join_keys, how="left", validate="one_to_one")
    merged = merged.merge(band_features_df, on=join_keys, how="left", validate="one_to_one")
    enriched = _add_model2_features(merged)
    enriched, band_mix_fallback_rows, band_mix_fallback_source = _apply_bucket_band_mix_fallback(enriched, source_summary)

    preferred_cols = [
        "time_bucket",
        "bucket_seq",
        "grid_id",
        "grid_row",
        "grid_col",
        "grid_centroid_lat",
        "grid_centroid_lon",
        "clutter_class",
        "building_count",
        "building_area_ratio",
        "road_density",
        "road_length_m",
        "park_open_area",
        "open_area_ratio",
        "mall_presence",
        "metro_presence",
        "sample_count",
        "dl_tpt_mean",
        "ul_tpt_mean",
        "estimated_prb_mean",
        "prb_pressure_est",
        "prb_outlier_flag",
        "cqi_mean",
        "sinr_mean",
        "rsrp_mean",
        "rsrq_mean",
        "corrected_rsrp_mean",
        "corrected_rsrq_mean",
        "corrected_sinr_mean",
        "bandwidth_mhz_est",
        "low_band_ratio",
        "mid_band_ratio",
        "high_band_ratio",
        "dominant_band_class",
        "carrier_count",
        "geo_demand_score",
        "kpi_demand_score",
        "growth_rate",
        "development_pressure_score",
        "growth_zone_score",
        "clutter_transition_flag",
        "clutter_upgrade_score",
        "building_growth_ratio",
        "road_growth_ratio",
        "activity_anchor_score",
        "capacity_context_score",
        "capacity_gap_score",
        "demand_index",
        "active_users_est",
        "traffic_demand_est",
        "demand_feature_source",
    ]
    final_cols = [col for col in preferred_cols if col in enriched.columns]
    final_cols += [col for col in enriched.columns if col not in set(final_cols)]
    enriched = enriched[final_cols].sort_values(["grid_id", "bucket_seq"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_csv, index=False)

    expected_rows = int(source_summary.get("grid_cell_count", 0)) * len(source_summary.get("bucket_ranges", []))
    raw_prb = pd.to_numeric(enriched.get("estimated_prb_mean"), errors="coerce")
    prb_pressure = pd.to_numeric(enriched.get("prb_pressure_est"), errors="coerce")
    summary = {
        "source_archive": str(archive_path),
        "output_csv": str(output_csv),
        "rows": int(len(enriched)),
        "expected_rows_from_source_summary": expected_rows,
        "unique_grids": int(enriched["grid_id"].nunique(dropna=True)),
        "bucket_counts": {str(k): int(v) for k, v in enriched["time_bucket"].value_counts().sort_index().items()},
        "missing_geo_rows": int(enriched["clutter_class"].isna().sum()) if "clutter_class" in enriched.columns else int(len(enriched)),
        "demand_index_min": float(enriched["demand_index"].min()),
        "demand_index_max": float(enriched["demand_index"].max()),
        "active_users_est_min": float(enriched["active_users_est"].min()),
        "active_users_est_max": float(enriched["active_users_est"].max()),
        "traffic_demand_est_min": float(enriched["traffic_demand_est"].min()),
        "traffic_demand_est_max": float(enriched["traffic_demand_est"].max()),
        "estimated_prb_p99": float(raw_prb.quantile(0.99)) if raw_prb.notna().any() else None,
        "estimated_prb_max": float(raw_prb.max()) if raw_prb.notna().any() else None,
        "prb_pressure_cap": PRB_PRESSURE_CAP,
        "prb_pressure_est_max": float(prb_pressure.max()) if prb_pressure.notna().any() else None,
        "prb_outlier_threshold": PRB_OUTLIER_THRESHOLD,
        "prb_outlier_rows": int(enriched["prb_outlier_flag"].sum()) if "prb_outlier_flag" in enriched.columns else 0,
        "dominant_band_class_counts": {
            str(k): int(v) for k, v in enriched.get("dominant_band_class", pd.Series(dtype="object")).value_counts(dropna=False).items()
        },
        "band_feature_source": band_feature_source,
        "band_mix_fallback_source": band_mix_fallback_source,
        "band_mix_fallback_rows": band_mix_fallback_rows,
        "band_ratio_nonzero_rows": int(
            (
                pd.to_numeric(enriched.get("low_band_ratio"), errors="coerce").fillna(0.0)
                + pd.to_numeric(enriched.get("mid_band_ratio"), errors="coerce").fillna(0.0)
                + pd.to_numeric(enriched.get("high_band_ratio"), errors="coerce").fillna(0.0)
            ).gt(0.0).sum()
        ),
        "notes": [
            "Built from existing coverage artifacts only; coverage pipeline was not rerun.",
            "Demand features use geo context, KPI bucket history, and added development/capacity context features.",
            "Band-family ratios and dominant band class are derived from prediction-grid band columns when available, otherwise from grid_kpi_timeseries dominant EARFCN.",
            "estimated_prb_mean is retained as the raw unbounded proxy for audit only.",
            "Model 2 training should use prb_pressure_est, capped at 40, instead of raw estimated_prb_mean.",
            "active_users_est is an estimated proxy because real subscriber counters are not present in the source data.",
        ],
    }
    summary_path_out = output_csv.with_suffix(".summary.json")
    summary_path_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return enriched, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Model 2 capacity training dataset from saved coverage artifacts.")
    parser.add_argument("--archive", default="data/coverage_20260521_104406.7z", help="Coverage artifact .7z archive")
    parser.add_argument("--output", default="data/model2_capacity_training.csv", help="Output CSV path")
    parser.add_argument("--work-dir", default="data/tmp/model2_capacity_training_source", help="Temporary extract/read directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = build_dataset(Path(args.archive), Path(args.output), Path(args.work_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
