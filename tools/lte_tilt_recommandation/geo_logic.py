from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
from tools.lte_tilt_recommandation.cell_identity import (
    canonical_cell_id,
    canonical_pair_series,
)

MIN_BAD_SAMPLE_COUNT_FOR_ACTION = 25
MEDIUM_CONFIDENCE_BAD_SAMPLE_COUNT = 120
HIGH_CONFIDENCE_BAD_SAMPLE_COUNT = 250
VERY_HIGH_CONFIDENCE_BAD_SAMPLE_COUNT = 600

FAR_EDGE_MEAN_SERVING_DISTANCE_M = 175.0
FAR_EDGE_P90_SERVING_DISTANCE_M = 260.0
MEDIUM_EDGE_MEAN_SERVING_DISTANCE_M = 130.0
MEDIUM_EDGE_P90_SERVING_DISTANCE_M = 220.0

HIGH_NLOS_SHARE_GATE = 0.7046681119665223
HIGH_LOS_BLOCKED_RATIO_GATE = 0.2139142952055874
HIGH_BUILDING_AREA_RATIO_GATE = 0.24481831460941722
MEDIUM_NLOS_SHARE_GATE = 0.4834199705913421
MEDIUM_LOS_BLOCKED_RATIO_GATE = 0.13621508948087993

DENSE_OVERLAP_NEAREST_SITE_M = 180.0
DENSE_OVERLAP_SITE_COUNT_250M = 2.0
SMALL_AZIMUTH_DELTA_DEG = 35.0

MIN_AZIMUTH_MISMATCH_DEG = 15.0
MAX_AZIMUTH_MISMATCH_DEG = 45.0
MAX_AZIMUTH_STEP_DEG = 10.0
MIN_BEARING_SAMPLE_COUNT = 30
MAX_BEARING_SPREAD_DEG = 40.0
AZIMUTH_NLOS_HARD_BLOCK_GATE = 0.85
BEARING_BIN_SIZE_DEG = 10.0
MIN_PEAK_SHARE_FOR_AZIMUTH = 0.27944004818646667
MIN_SAFE_ETILT_DEG = 2.0
MAX_SAFE_ETILT_DEG = 12.0
MAX_ETILT_INCREASE_PER_RUN_DEG = 2.0
MAX_ETILT_DECREASE_PER_RUN_DEG = 2.0
RELAXED_AZIMUTH_BAD_SAMPLE_COUNT = 183
RELAXED_AZIMUTH_PEAK_SHARE = 0.5721666812548345
RELAXED_AZIMUTH_MAX_SPREAD_DEG = 32.264493787350155
MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH = 1.2734413438634005
STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH = 2.542748090042536
MIN_CANDIDATE_SCORE_GAP = 4.654858443626446
SITE_SYMMETRY_PENALTY_BAD_SAMPLE_RATIO = 0.055289002855823374

COVERAGE_ETILT_ACTION_SCORE = 51.49084372010145
OVERLAP_ETILT_ACTION_SCORE = 47.31004233364023
AZIMUTH_ACTION_SCORE = 71.61142299538227
TX_POWER_ACTION_SCORE = 46.25093121890758


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> str:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        key = c.strip().lower()
        if key in lower_map:
            return lower_map[key]
    if required:
        raise KeyError(f"Required column not found. Tried: {candidates}")
    return ""


def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _norm_cell_id(value) -> str:
    return canonical_cell_id(value)


def _cell_id_suffix(value) -> str:
    s = _norm_cell_id(value)
    return s.split("_")[-1] if s else ""


def _normalize_azimuth(val) -> float:
    if pd.isna(val):
        return np.nan
    return float(val) % 360.0


def _angular_diff(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    lat1_r = math.radians(float(lat1))
    lat2_r = math.radians(float(lat2))
    dlon_r = math.radians(float(lon2) - float(lon1))
    y = math.sin(dlon_r) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _circular_mean_deg(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if vals.empty:
        return np.nan
    radians = np.deg2rad(vals.values)
    return (math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean())) + 360.0) % 360.0


def _get_cell_key_from_log(log_df: pd.DataFrame) -> pd.Series:
    for col in ["Node_Cell_ID", "node_cell_id", "frontend_site_sector_key", "nodeb_id_cell_id"]:
        existing_col = _find_col(log_df, [col], required=False)
        if existing_col:
            existing = log_df[existing_col].map(canonical_cell_id)
            if existing.astype(str).str.strip().ne("").any():
                return existing

    nodeb_col = _find_col(log_df, ["nodeb_id", "nodeb", "enodeb_id", "gnodeb_id", "site_id"], required=False)
    cell_col = _find_col(log_df, ["cell_id", "eci", "ecgi_cell_id", "local_cell_id"], required=False)
    if nodeb_col and cell_col:
        return canonical_pair_series(log_df[nodeb_col], log_df[cell_col])
    if cell_col:
        return log_df[cell_col].map(canonical_cell_id)
    raise KeyError("Could not derive Cell ID from log data.")


def _derive_antenna_cell_key(antenna_df: pd.DataFrame) -> pd.Series:
    node_cell_col = _find_col(antenna_df, ["Node_Cell_ID", "node_cell_id", "frontend_site_sector_key", "nodeb_id_cell_id"], required=False)
    if node_cell_col:
        node_cell = antenna_df[node_cell_col].map(canonical_cell_id)
        if (node_cell != "").any():
            return node_cell
    nodeb_col = _find_col(antenna_df, ["nodeb_id", "nodeb", "site_id"], required=False)
    cell_col = _find_col(antenna_df, ["cell_id", "eci", "local_cell_id"], required=False)
    if nodeb_col and cell_col:
        combined = canonical_pair_series(antenna_df[nodeb_col], antenna_df[cell_col])
        if (combined != "").any():
            return combined
    if cell_col:
        return antenna_df[cell_col].map(canonical_cell_id)
    return pd.Series("", index=antenna_df.index)


def _mode_or_blank(series: pd.Series) -> str:
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return ""
    mode = vals.mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else ""


def _fmt_num(value: float, decimals: int) -> str:
    return "nan" if pd.isna(value) else f"{float(value):.{decimals}f}"


def _values_changed(curr, rec) -> bool:
    curr_num = pd.to_numeric(pd.Series([curr]), errors="coerce").iloc[0]
    rec_num = pd.to_numeric(pd.Series([rec]), errors="coerce").iloc[0]
    if pd.notna(curr_num) and pd.notna(rec_num):
        return not np.isclose(float(curr_num), float(rec_num), equal_nan=True)
    return str(curr) != str(rec)


def _changed_recommendation_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    return df.apply(lambda r: _values_changed(r.get("Current Value"), r.get("Recommended Value")), axis=1)


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(float(value)))))


def _confidence_label(score: int) -> str:
    return "high" if score >= 75 else "medium" if score >= 55 else "low"


def _circular_spread_deg(values: pd.Series, center_deg: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if vals.empty or pd.isna(center_deg):
        return np.nan
    diffs = vals.map(lambda v: _angular_diff(v, center_deg))
    return float(pd.to_numeric(diffs, errors="coerce").mean())


def _directional_peak_summary(values: pd.Series, weights: pd.Series, bin_size_deg: float = BEARING_BIN_SIZE_DEG) -> Dict[str, float]:
    bearings = pd.to_numeric(values, errors="coerce")
    bearing_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    work = pd.DataFrame({"bearing": bearings, "weight": bearing_weights}).dropna(subset=["bearing"])
    if work.empty:
        return {"peak_bearing_deg": np.nan, "peak_spread_deg": np.nan, "peak_share": np.nan, "peak_weight": 0.0, "second_peak_weight": 0.0, "directional_contrast": np.nan}
    work["bearing"] = work["bearing"].astype(float).map(_normalize_azimuth)
    work["weight"] = work["weight"].astype(float).clip(lower=0.0)
    work["weight"] = work["weight"].where(work["weight"] > 0.0, 1.0)
    bin_count = max(1, int(round(360.0 / float(bin_size_deg))))
    bin_idx = np.floor(work["bearing"].to_numpy() / float(bin_size_deg)).astype(int) % bin_count
    work["bin_idx"] = bin_idx
    bin_weights = np.bincount(bin_idx, weights=work["weight"].to_numpy(), minlength=bin_count).astype(float)
    smooth_weights = bin_weights + np.roll(bin_weights, 1) + np.roll(bin_weights, -1) if bin_count > 1 else bin_weights.copy()
    peak_bin = int(np.argmax(smooth_weights))
    bin_distance = ((work["bin_idx"] - peak_bin + bin_count / 2.0) % bin_count) - (bin_count / 2.0)
    local_lobe = work.loc[bin_distance.abs() <= 1].copy()
    if local_lobe.empty:
        local_lobe = work.copy()
    peak_bearing = _circular_mean_deg(local_lobe["bearing"])
    peak_spread = _circular_spread_deg(local_lobe["bearing"], peak_bearing)
    peak_weight = float(local_lobe["weight"].sum())
    total_weight = float(work["weight"].sum())
    peak_share = peak_weight / total_weight if total_weight > 0 else np.nan
    protected_bins = {(peak_bin - 1) % bin_count, peak_bin % bin_count, (peak_bin + 1) % bin_count}
    second_peak_weight = float(max((smooth_weights[idx] for idx in range(bin_count) if idx not in protected_bins), default=0.0))
    directional_contrast = peak_weight / second_peak_weight if second_peak_weight > 0 else np.inf
    return {"peak_bearing_deg": peak_bearing, "peak_spread_deg": peak_spread, "peak_share": peak_share, "peak_weight": peak_weight, "second_peak_weight": second_peak_weight, "directional_contrast": directional_contrast}


def compute_dominant_bearing_summary(log_df: pd.DataFrame, antenna_df: pd.DataFrame, rsrp_thresh: float, rsrq_thresh: float, sinr_thresh: float) -> pd.DataFrame:
    log_work = _normalize_columns(log_df).copy()
    ant_work = _normalize_columns(antenna_df).copy()
    log_work["Cell ID"] = _get_cell_key_from_log(log_work)
    ant_work["Cell ID"] = _derive_antenna_cell_key(ant_work)
    lat_col = _find_col(log_work, ["lat", "latitude"], required=False)
    lon_col = _find_col(log_work, ["lon", "longitude"], required=False)
    rsrp_col = _find_col(log_work, ["rsrp", "pred_rsrp", "csi_rsrp"], required=False)
    rsrq_col = _find_col(log_work, ["rsrq", "pred_rsrq", "csi_rsrq"], required=False)
    sinr_col = _find_col(log_work, ["sinr", "pred_sinr", "csi_sinr"], required=False)
    ant_lat_col = _find_col(ant_work, ["latitude", "lat"], required=False)
    ant_lon_col = _find_col(ant_work, ["longitude", "lon"], required=False)
    az_col = _find_col(ant_work, ["azimuth", "azi"], required=False)
    log_work["RSRP_eval"] = pd.to_numeric(log_work[rsrp_col], errors="coerce") if rsrp_col else np.nan
    log_work["RSRQ_eval"] = pd.to_numeric(log_work[rsrq_col], errors="coerce") if rsrq_col else np.nan
    log_work["SINR_eval"] = pd.to_numeric(log_work[sinr_col], errors="coerce") if sinr_col else np.nan
    log_work["Lat_eval"] = pd.to_numeric(log_work[lat_col], errors="coerce") if lat_col else np.nan
    log_work["Lon_eval"] = pd.to_numeric(log_work[lon_col], errors="coerce") if lon_col else np.nan
    log_work["severity_score"] = ((float(rsrp_thresh) - log_work["RSRP_eval"]).clip(lower=0).fillna(0) + (float(rsrq_thresh) - log_work["RSRQ_eval"]).clip(lower=0).fillna(0) + (float(sinr_thresh) - log_work["SINR_eval"]).clip(lower=0).fillna(0))
    ant_map = (
        ant_work.drop_duplicates(subset=["Cell ID"])
        .assign(Azimuth_cfg=lambda d: pd.to_numeric(d[az_col], errors="coerce").map(_normalize_azimuth) if az_col else np.nan, AntLat=lambda d: pd.to_numeric(d[ant_lat_col], errors="coerce") if ant_lat_col else np.nan, AntLon=lambda d: pd.to_numeric(d[ant_lon_col], errors="coerce") if ant_lon_col else np.nan)
        .set_index("Cell ID")[["Azimuth_cfg", "AntLat", "AntLon"]]
        .to_dict("index")
    )
    rows: List[Dict[str, object]] = []
    for cell_id, group in log_work.groupby("Cell ID", dropna=False):
        if not cell_id or cell_id not in ant_map:
            continue
        ant = ant_map[cell_id]
        if pd.isna(ant.get("AntLat")) or pd.isna(ant.get("AntLon")):
            continue
        g2 = group.dropna(subset=["RSRP_eval", "Lat_eval", "Lon_eval"]).copy()
        if g2.empty:
            continue
        use_dir = g2[g2["severity_score"] > 0].copy()
        if use_dir.empty:
            continue
        if len(use_dir) < MIN_BEARING_SAMPLE_COUNT:
            use_dir = g2.nlargest(min(len(g2), MIN_BEARING_SAMPLE_COUNT), "severity_score").copy()
            use_dir = use_dir[use_dir["severity_score"] > 0]
            if use_dir.empty:
                continue
        use_dir["bearing"] = use_dir.apply(lambda r: _bearing_deg(ant["AntLat"], ant["AntLon"], r["Lat_eval"], r["Lon_eval"]), axis=1)
        peak = _directional_peak_summary(use_dir["bearing"], use_dir["severity_score"])
        dominant_bearing = peak["peak_bearing_deg"]
        rows.append({"Cell ID": cell_id, "dominant_bearing_deg": dominant_bearing, "bearing_mismatch_deg": _angular_diff(dominant_bearing, ant.get("Azimuth_cfg", np.nan)), "bearing_sample_count": int(len(use_dir)), "bearing_spread_deg": peak["peak_spread_deg"], "bearing_peak_share": peak["peak_share"], "bearing_directional_contrast": peak["directional_contrast"]})
    return pd.DataFrame(rows)


def attach_geo_to_bad_samples(bad_df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    out = bad_df.copy()
    if geo_df.empty:
        return out
    geo_work = geo_df.copy()
    out["Cell ID"] = out["Cell ID"].map(_norm_cell_id)
    geo_work["Cell ID"] = geo_work["Node_Cell_ID"].astype(str).map(_norm_cell_id)
    lat_col = _find_col(out, ["lat", "latitude"], required=False)
    lon_col = _find_col(out, ["lon", "longitude"], required=False)
    if not lat_col or not lon_col:
        return out
    out["lat_6dp"] = pd.to_numeric(out[lat_col], errors="coerce").round(6)
    out["lon_6dp"] = pd.to_numeric(out[lon_col], errors="coerce").round(6)
    geo_work["lat_6dp"] = pd.to_numeric(geo_work["lat"], errors="coerce").round(6)
    geo_work["lon_6dp"] = pd.to_numeric(geo_work["lon"], errors="coerce").round(6)
    geo_cols = ["Cell ID", "lat_6dp", "lon_6dp", "clutter_class", "building_area_ratio", "los_blocked_ratio", "nlos_flag", "site_count_250m", "serving_distance_m", "nearest_site_distance_m", "mean_nearest3_site_distance_m", "azimuth_delta_deg"]
    geo_cols = [c for c in geo_cols if c in geo_work.columns]
    return out.merge(geo_work[geo_cols].drop_duplicates(subset=["Cell ID", "lat_6dp", "lon_6dp"], keep="last"), on=["Cell ID", "lat_6dp", "lon_6dp"], how="left")


def aggregate_bad_geo_context(bad_geo_df: pd.DataFrame) -> pd.DataFrame:
    if bad_geo_df.empty:
        return pd.DataFrame()
    work = bad_geo_df.copy()
    work["nlos_flag_num"] = pd.to_numeric(work.get("nlos_flag"), errors="coerce").fillna(0.0)
    return work.groupby(["Cell ID", "Technology"], dropna=False).agg(
        bad_sample_count=("Cell ID", "size"),
        mean_serving_distance_m=("serving_distance_m", "mean"),
        p90_serving_distance_m=("serving_distance_m", lambda s: pd.to_numeric(s, errors="coerce").dropna().quantile(0.90) if pd.to_numeric(s, errors="coerce").dropna().size else np.nan),
        mean_nearest_site_distance_m=("nearest_site_distance_m", "mean"),
        mean_site_count_250m=("site_count_250m", "mean"),
        mean_azimuth_delta_deg=("azimuth_delta_deg", "mean"),
        mean_building_area_ratio=("building_area_ratio", "mean"),
        mean_los_blocked_ratio=("los_blocked_ratio", "mean"),
        nlos_share=("nlos_flag_num", "mean"),
        clutter_mode=("clutter_class", _mode_or_blank),
    ).reset_index()


def _bounded_etilt_target(current_etilt: float, requested_etilt: float) -> float:
    if pd.isna(current_etilt) or pd.isna(requested_etilt):
        return np.nan
    lower = max(float(MIN_SAFE_ETILT_DEG), float(current_etilt) - float(MAX_ETILT_DECREASE_PER_RUN_DEG))
    upper = min(float(MAX_SAFE_ETILT_DEG), float(current_etilt) + float(MAX_ETILT_INCREASE_PER_RUN_DEG))
    return float(np.clip(float(requested_etilt), lower, upper))


def _signed_azimuth_delta(target_deg: float, current_deg: float) -> float:
    if pd.isna(target_deg) or pd.isna(current_deg):
        return np.nan
    return ((float(target_deg) - float(current_deg) + 180.0) % 360.0) - 180.0


def _select_best_candidate(candidates: List[Dict[str, object]]) -> Dict[str, object]:
    return max(candidates, key=lambda c: (float(c["score"]), -abs(float(c.get("delta", 0.0)))))


def _select_best_candidate_with_gap(candidates: List[Dict[str, object]], min_gap: float = MIN_CANDIDATE_SCORE_GAP) -> Dict[str, object]:
    ranked = sorted(candidates, key=lambda c: (float(c["score"]), -abs(float(c.get("delta", 0.0)))), reverse=True)
    best = ranked[0]
    if len(ranked) == 1 or str(best.get("mode")) == "hold":
        return best
    second = ranked[1]
    if float(best["score"]) - float(second["score"]) < float(min_gap):
        hold = next((c for c in candidates if str(c.get("mode")) == "hold"), None)
        if hold is not None:
            return hold
    return best


def _estimate_etilt_validation_gain(
    mode: str,
    delta: float,
    coverage_score: float,
    overlap_score: float,
    mean_dist: float,
    p90_dist: float,
    overlap_dense: bool,
    interference_signal_present: bool,
    coverage_signal_present: bool,
    coverage_dominant: bool,
    medium_edge_issue: bool,
    far_edge_issue: bool,
    nlos_share: float,
    los_blocked: float,
) -> float:
    gain = 0.0
    abs_delta = abs(float(delta))
    if mode == "coverage":
        gain += max(0.0, coverage_score - COVERAGE_ETILT_ACTION_SCORE)
        if far_edge_issue:
            gain += 8.0
        elif medium_edge_issue:
            gain += 5.0
        if coverage_dominant:
            gain += 4.0
        if not pd.isna(mean_dist):
            gain += min(6.0, max(0.0, (float(mean_dist) - MEDIUM_EDGE_MEAN_SERVING_DISTANCE_M) / 12.0))
        if not pd.isna(p90_dist):
            gain += min(6.0, max(0.0, (float(p90_dist) - MEDIUM_EDGE_P90_SERVING_DISTANCE_M) / 16.0))
        if overlap_dense and interference_signal_present:
            gain -= 5.0 * abs_delta
        if not pd.isna(nlos_share) and nlos_share >= HIGH_NLOS_SHARE_GATE:
            gain -= 5.0
        if not pd.isna(los_blocked) and los_blocked >= HIGH_LOS_BLOCKED_RATIO_GATE:
            gain -= 4.0
    elif mode == "overlap":
        gain += max(0.0, overlap_score - OVERLAP_ETILT_ACTION_SCORE)
        if overlap_dense:
            gain += 8.0
        if interference_signal_present:
            gain += 5.0
        if coverage_signal_present and (far_edge_issue or medium_edge_issue):
            gain -= 4.0 * abs_delta
        if not pd.isna(mean_dist) and mean_dist >= FAR_EDGE_MEAN_SERVING_DISTANCE_M:
            gain -= 4.0
        if not pd.isna(p90_dist) and p90_dist >= FAR_EDGE_P90_SERVING_DISTANCE_M:
            gain -= 4.0
        if not pd.isna(nlos_share) and nlos_share >= HIGH_NLOS_SHARE_GATE:
            gain -= 2.0
    return gain


def _estimate_azimuth_validation_gain(
    current_az: float,
    target_az: float,
    dominant_bearing: float,
    azimuth_score: float,
    bearing_mismatch: float,
    bearing_peak_share: float,
    directional_contrast: float,
    bearing_spread: float,
    nlos_share: float,
    los_blocked: float,
    overlap_dense: bool,
    interference_signal_present: bool,
) -> float:
    if pd.isna(current_az) or pd.isna(target_az) or pd.isna(dominant_bearing):
        return -999.0
    current_residual = _angular_diff(dominant_bearing, current_az)
    target_residual = _angular_diff(dominant_bearing, target_az)
    correction_gain = max(0.0, float(current_residual) - float(target_residual))
    gain = max(0.0, azimuth_score - AZIMUTH_ACTION_SCORE)
    gain += correction_gain * 1.8
    if not pd.isna(bearing_mismatch):
        gain += min(6.0, float(bearing_mismatch) / 8.0)
    if not pd.isna(bearing_peak_share):
        gain += max(0.0, (float(bearing_peak_share) - 0.25) * 20.0)
    if not pd.isna(directional_contrast):
        gain += max(0.0, (float(directional_contrast) - 1.0) * 5.0)
    if not pd.isna(bearing_spread):
        gain += max(0.0, (MAX_BEARING_SPREAD_DEG - float(bearing_spread)) / 5.0)
    if overlap_dense and interference_signal_present:
        gain += 2.0
    if not pd.isna(nlos_share) and nlos_share >= MEDIUM_NLOS_SHARE_GATE:
        gain -= 6.0
    if not pd.isna(los_blocked) and los_blocked >= HIGH_LOS_BLOCKED_RATIO_GATE:
        gain -= 5.0
    if target_residual > current_residual:
        gain -= 12.0
    return gain


def build_geo_aware_recommendations(bad_summary: pd.DataFrame, antenna_df: pd.DataFrame, swap_dict: Dict[str, str], geo_cell_summary: pd.DataFrame, bearing_summary: pd.DataFrame) -> pd.DataFrame:
    bad_summary = _normalize_columns(bad_summary).copy()
    antenna_df = _normalize_columns(antenna_df).copy()
    geo_cell_summary = geo_cell_summary.copy()
    bearing_summary = bearing_summary.copy()

    bad_summary["Cell ID"] = bad_summary["Cell ID"].map(_norm_cell_id)
    antenna_df["Cell ID"] = _derive_antenna_cell_key(antenna_df)
    if not geo_cell_summary.empty:
        geo_cell_summary["Cell ID"] = geo_cell_summary["Cell ID"].map(_norm_cell_id)
    if not bearing_summary.empty:
        bearing_summary["Cell ID"] = bearing_summary["Cell ID"].map(_norm_cell_id)

    ant_use = antenna_df.drop_duplicates(subset=["Cell ID"]).copy()
    ant_use["Cell ID Suffix"] = ant_use["Cell ID"].map(_cell_id_suffix)
    ant_local_cell_col = _find_col(antenna_df, ["cell_id", "eci", "local_cell_id"], required=False)
    ant_use["Antenna Local Cell"] = ant_use[ant_local_cell_col].map(_norm_cell_id) if ant_local_cell_col else ant_use["Cell ID Suffix"]

    tech_col = _find_col(antenna_df, ["Technology", "technology"], required=False)
    az_col = _find_col(antenna_df, ["azimuth", "azi"], required=False)
    etilt_col = _find_col(antenna_df, ["e_tilt", "etilt", "electrical_tilt"], required=False)
    power_col = _find_col(antenna_df, ["tx_power", "real_transmit_power_of_resource", "reference_signal_power"], required=False)

    geo_map = geo_cell_summary.set_index("Cell ID").to_dict("index") if not geo_cell_summary.empty else {}
    bearing_map = bearing_summary.set_index("Cell ID").to_dict("index") if not bearing_summary.empty else {}
    site_stats: Dict[str, Dict[str, float]] = {}
    if not bad_summary.empty:
        tmp = bad_summary.copy()
        tmp["site_key"] = tmp["Cell ID"].astype(str).str.split("_").str[0]
        tmp["total_bad_samples"] = (
            pd.to_numeric(tmp.get("Bad RSRP", 0), errors="coerce").fillna(0)
            + pd.to_numeric(tmp.get("Bad RSRQ", 0), errors="coerce").fillna(0)
            + pd.to_numeric(tmp.get("Bad SINR", 0), errors="coerce").fillna(0)
        )
        site_stats = tmp.groupby("site_key", dropna=False).agg(
            site_sector_count=("Cell ID", "nunique"),
            site_total_bad_samples=("total_bad_samples", "sum"),
            site_mean_bad_samples=("total_bad_samples", "mean"),
            site_max_bad_samples=("total_bad_samples", "max"),
        ).to_dict("index")

    rec_rows: List[Dict[str, object]] = []
    for _, row in bad_summary.iterrows():
        cell_id = _norm_cell_id(row["Cell ID"])
        site_key = cell_id.split("_")[0] if "_" in cell_id else cell_id
        tech = _safe_str(row.get("Technology", "")) or "UNKNOWN"
        bad_rsrp = int(row.get("Bad RSRP", 0) or 0)
        bad_rsrq = int(row.get("Bad RSRQ", 0) or 0)
        bad_sinr = int(row.get("Bad SINR", 0) or 0)
        total_bad = bad_rsrp + bad_rsrq + bad_sinr
        swap_flag = swap_dict.get(cell_id, "No")

        ant_row = ant_use.loc[ant_use["Cell ID"] == cell_id]
        if ant_row.empty:
            cell_suffix = _cell_id_suffix(cell_id)
            if cell_suffix:
                ant_row = ant_use.loc[ant_use["Antenna Local Cell"] == cell_suffix]
            if len(ant_row) != 1 and cell_suffix:
                ant_row = ant_use.loc[ant_use["Cell ID Suffix"] == cell_suffix]
        if ant_row.empty:
            continue
        if len(ant_row) > 1:
            ant_row = ant_row.iloc[[0]]
        ant = ant_row.iloc[0]

        ant_tech = _safe_str(ant[tech_col]) if tech_col and tech_col in ant else tech
        curr_az = pd.to_numeric(ant[az_col], errors="coerce") if az_col else np.nan
        curr_etilt = pd.to_numeric(ant[etilt_col], errors="coerce") if etilt_col else np.nan
        curr_power = pd.to_numeric(ant[power_col], errors="coerce") if power_col else np.nan

        geo_ctx = geo_map.get(cell_id, {})
        bearing_ctx = bearing_map.get(cell_id, {})
        bad_sample_count = int(pd.to_numeric(pd.Series([geo_ctx.get("bad_sample_count", total_bad)]), errors="coerce").fillna(total_bad).iloc[0])
        mean_dist = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_serving_distance_m")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        p90_dist = float(pd.to_numeric(pd.Series([geo_ctx.get("p90_serving_distance_m")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        mean_az_delta = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_azimuth_delta_deg")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        nlos_share = float(pd.to_numeric(pd.Series([geo_ctx.get("nlos_share")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        los_blocked = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_los_blocked_ratio")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        building_ratio = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_building_area_ratio")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        nearest_site = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_nearest_site_distance_m")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        site_count_250m = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_site_count_250m")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        clutter_mode = str(geo_ctx.get("clutter_mode", "") or "")

        dominant_bearing = float(pd.to_numeric(pd.Series([bearing_ctx.get("dominant_bearing_deg")]), errors="coerce").iloc[0]) if bearing_ctx else np.nan
        bearing_mismatch = float(pd.to_numeric(pd.Series([bearing_ctx.get("bearing_mismatch_deg")]), errors="coerce").iloc[0]) if bearing_ctx else np.nan
        bearing_sample_count = int(pd.to_numeric(pd.Series([bearing_ctx.get("bearing_sample_count", 0)]), errors="coerce").fillna(0).iloc[0]) if bearing_ctx else 0
        bearing_spread = float(pd.to_numeric(pd.Series([bearing_ctx.get("bearing_spread_deg")]), errors="coerce").iloc[0]) if bearing_ctx else np.nan
        bearing_peak_share = float(pd.to_numeric(pd.Series([bearing_ctx.get("bearing_peak_share")]), errors="coerce").iloc[0]) if bearing_ctx else np.nan
        directional_contrast = float(pd.to_numeric(pd.Series([bearing_ctx.get("bearing_directional_contrast")]), errors="coerce").iloc[0]) if bearing_ctx else np.nan

        site_ctx = site_stats.get(site_key, {})
        site_sector_count = int(pd.to_numeric(pd.Series([site_ctx.get("site_sector_count", 1)]), errors="coerce").fillna(1).iloc[0])
        site_mean_bad_samples = float(pd.to_numeric(pd.Series([site_ctx.get("site_mean_bad_samples", bad_sample_count)]), errors="coerce").fillna(bad_sample_count).iloc[0])
        site_total_bad_samples = float(pd.to_numeric(pd.Series([site_ctx.get("site_total_bad_samples", bad_sample_count)]), errors="coerce").fillna(bad_sample_count).iloc[0])

        low_signal = bad_sample_count < MIN_BAD_SAMPLE_COUNT_FOR_ACTION
        interference_signal_present = (bad_sinr + bad_rsrq) >= max(20, int(total_bad * 0.35))
        coverage_signal_ratio = bad_rsrp / max(total_bad, 1)
        interference_signal_ratio = (bad_sinr + bad_rsrq) / max(total_bad, 1)
        sinr_dominant = bad_sinr > max(bad_rsrp, bad_rsrq) and bad_sinr > 0
        rsrq_dominant = bad_rsrq > max(bad_rsrp, bad_sinr) and bad_rsrq > 0
        far_edge_issue = ((not pd.isna(mean_dist) and mean_dist >= FAR_EDGE_MEAN_SERVING_DISTANCE_M) or (not pd.isna(p90_dist) and p90_dist >= FAR_EDGE_P90_SERVING_DISTANCE_M))
        medium_edge_issue = ((not pd.isna(mean_dist) and mean_dist >= MEDIUM_EDGE_MEAN_SERVING_DISTANCE_M) or (not pd.isna(p90_dist) and p90_dist >= MEDIUM_EDGE_P90_SERVING_DISTANCE_M))
        high_blockage = ((not pd.isna(nlos_share) and nlos_share >= HIGH_NLOS_SHARE_GATE) or (not pd.isna(los_blocked) and los_blocked >= HIGH_LOS_BLOCKED_RATIO_GATE) or (not pd.isna(building_ratio) and building_ratio >= HIGH_BUILDING_AREA_RATIO_GATE))
        medium_blockage = ((not pd.isna(nlos_share) and nlos_share >= MEDIUM_NLOS_SHARE_GATE) or (not pd.isna(los_blocked) and los_blocked >= MEDIUM_LOS_BLOCKED_RATIO_GATE))
        close_site_overlap = not pd.isna(nearest_site) and nearest_site <= DENSE_OVERLAP_NEAREST_SITE_M
        dense_site_overlap = not pd.isna(site_count_250m) and site_count_250m >= DENSE_OVERLAP_SITE_COUNT_250M
        overlap_dense = close_site_overlap and dense_site_overlap
        small_az_delta = pd.isna(mean_az_delta) or mean_az_delta <= SMALL_AZIMUTH_DELTA_DEG
        site_persistent_issue = (site_sector_count >= 3 and site_mean_bad_samples >= MEDIUM_CONFIDENCE_BAD_SAMPLE_COUNT and site_total_bad_samples >= HIGH_CONFIDENCE_BAD_SAMPLE_COUNT * 3)
        persistent_coverage_signal = (site_persistent_issue and bad_rsrp >= max(12, int(total_bad * 0.20)))
        coverage_signal_present = (bad_rsrp >= max(20, int(total_bad * 0.30)) or persistent_coverage_signal)
        coverage_dominant = coverage_signal_present and bad_rsrp >= max(bad_sinr, bad_rsrq)
        high_bad_volume_override = (bad_sample_count >= VERY_HIGH_CONFIDENCE_BAD_SAMPLE_COUNT and medium_edge_issue and not high_blockage)
        stable_bearing_case = (not pd.isna(dominant_bearing) and not pd.isna(bearing_mismatch) and not pd.isna(bearing_spread) and not pd.isna(bearing_peak_share) and bearing_sample_count >= MIN_BEARING_SAMPLE_COUNT and MIN_AZIMUTH_MISMATCH_DEG <= bearing_mismatch <= MAX_AZIMUTH_MISMATCH_DEG and bearing_spread <= MAX_BEARING_SPREAD_DEG and bearing_peak_share >= MIN_PEAK_SHARE_FOR_AZIMUTH and (pd.isna(nlos_share) or nlos_share < AZIMUTH_NLOS_HARD_BLOCK_GATE))
        blockage_limited_case = coverage_signal_present and high_blockage and bad_rsrp >= max(10, int(total_bad * 0.20))
        site_symmetric_overlap_case = (site_sector_count >= 3 and overlap_dense and interference_signal_present and site_mean_bad_samples > 0 and abs(float(bad_sample_count) - float(site_mean_bad_samples)) <= float(site_mean_bad_samples) * SITE_SYMMETRY_PENALTY_BAD_SAMPLE_RATIO)

        coverage_etilt_score = 0.0
        overlap_etilt_score = 0.0
        azimuth_score = 0.0
        tx_power_score = 0.0
        if bad_sample_count >= VERY_HIGH_CONFIDENCE_BAD_SAMPLE_COUNT:
            coverage_etilt_score += 30.0
            overlap_etilt_score += 28.0
        elif bad_sample_count >= HIGH_CONFIDENCE_BAD_SAMPLE_COUNT:
            coverage_etilt_score += 22.0
            overlap_etilt_score += 20.0
        elif bad_sample_count >= MEDIUM_CONFIDENCE_BAD_SAMPLE_COUNT:
            coverage_etilt_score += 12.0
            overlap_etilt_score += 10.0
        if far_edge_issue:
            coverage_etilt_score += 22.0
            tx_power_score += 14.0
        elif medium_edge_issue:
            coverage_etilt_score += 14.0
            tx_power_score += 8.0
        if site_persistent_issue:
            coverage_etilt_score += 12.0
            overlap_etilt_score += 10.0
            azimuth_score += 8.0
        if coverage_signal_present:
            coverage_etilt_score += 12.0
            tx_power_score += 10.0
        if coverage_dominant:
            coverage_etilt_score += 8.0
            tx_power_score += 6.0
        if persistent_coverage_signal:
            coverage_etilt_score += 10.0
        if overlap_dense:
            overlap_etilt_score += 10.0
        if small_az_delta:
            overlap_etilt_score += 5.0
        if interference_signal_present:
            overlap_etilt_score += 10.0
            azimuth_score += 10.0
        if sinr_dominant:
            overlap_etilt_score += 6.0
            azimuth_score += 8.0
        elif rsrq_dominant:
            overlap_etilt_score += 3.0
            azimuth_score += 4.0
        if medium_edge_issue and coverage_signal_ratio >= 0.22 and not high_blockage:
            coverage_etilt_score += 12.0
        if medium_edge_issue and interference_signal_ratio < 0.72:
            coverage_etilt_score += 8.0
        if medium_edge_issue and not high_blockage and bad_sample_count >= HIGH_CONFIDENCE_BAD_SAMPLE_COUNT:
            coverage_etilt_score += 6.0
        if not pd.isna(p90_dist) and p90_dist >= 210.0 and not high_blockage:
            coverage_etilt_score += 4.0
        if overlap_dense and coverage_signal_ratio >= 0.28 and not high_blockage:
            coverage_etilt_score += 6.0
            overlap_etilt_score -= 6.0
        if medium_blockage and not high_blockage and medium_edge_issue:
            coverage_etilt_score += 5.0
        if interference_signal_ratio >= 0.72 and overlap_dense:
            overlap_etilt_score += 4.0
        if site_symmetric_overlap_case:
            overlap_etilt_score -= 2.0
        if stable_bearing_case:
            azimuth_score += 28.0
            azimuth_score += min(12.0, max(0.0, (MAX_BEARING_SPREAD_DEG - float(bearing_spread)) * 0.4))
            azimuth_score += min(12.0, max(0.0, (float(bearing_mismatch) - MIN_AZIMUTH_MISMATCH_DEG) * 0.4))
            azimuth_score += min(10.0, max(0.0, (float(bearing_peak_share) - MIN_PEAK_SHARE_FOR_AZIMUTH) * 40.0))
            if not pd.isna(directional_contrast):
                azimuth_score += min(12.0, max(0.0, (float(directional_contrast) - 1.0) * 6.0))
        elif bearing_sample_count > 0:
            azimuth_score -= 12.0
        if high_blockage:
            coverage_etilt_score -= 18.0
            tx_power_score -= 18.0
            if not overlap_dense:
                overlap_etilt_score -= 10.0
        elif medium_blockage:
            coverage_etilt_score -= 8.0
            tx_power_score -= 8.0
        if not pd.isna(nlos_share) and nlos_share >= AZIMUTH_NLOS_HARD_BLOCK_GATE:
            azimuth_score = -100.0
        if high_bad_volume_override:
            coverage_etilt_score += 10.0
        if interference_signal_present or overlap_dense:
            tx_power_score -= 18.0
        if not coverage_signal_present:
            tx_power_score -= 10.0
        if coverage_signal_present and not interference_signal_present and not overlap_dense:
            tx_power_score += 10.0
        if far_edge_issue and coverage_dominant and not high_blockage:
            tx_power_score += 8.0

        blockage_score = 0.0
        if high_blockage:
            blockage_score += 30.0
        elif medium_blockage:
            blockage_score += 16.0
        if not pd.isna(nlos_share):
            blockage_score += min(18.0, max(0.0, (float(nlos_share) - MEDIUM_NLOS_SHARE_GATE) * 60.0))

        directional_misalignment_case = (stable_bearing_case and not high_blockage and (pd.isna(nlos_share) or nlos_share < MEDIUM_NLOS_SHARE_GATE) and not pd.isna(directional_contrast) and directional_contrast >= MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH and bearing_peak_share >= max(MIN_PEAK_SHARE_FOR_AZIMUTH - 0.08, 0.30))
        if directional_misalignment_case:
            azimuth_score += 6.0
            if site_symmetric_overlap_case:
                azimuth_score += 4.0
        strong_azimuth_override = (stable_bearing_case and not pd.isna(directional_contrast) and directional_contrast >= 1.35 and not pd.isna(bearing_peak_share) and bearing_peak_share >= 0.34 and not pd.isna(bearing_mismatch) and 20.0 <= bearing_mismatch <= 35.0 and bad_sample_count >= 80 and (pd.isna(nlos_share) or nlos_share < AZIMUTH_NLOS_HARD_BLOCK_GATE) and (pd.isna(los_blocked) or los_blocked < HIGH_LOS_BLOCKED_RATIO_GATE))
        if strong_azimuth_override:
            azimuth_score += 4.0

        if low_signal:
            dominant_root_cause = "low_signal_confidence"
        else:
            root_scores = {
                "blockage_dominated": blockage_score,
                "coverage_candidate": coverage_etilt_score,
                "interference_candidate": overlap_etilt_score,
                "directional_misalignment": azimuth_score if directional_misalignment_case else 0.0,
                "interference_soft": 18.0 if interference_signal_present else 0.0,
                "coverage_soft": 16.0 if coverage_signal_present else 0.0,
            }
            dominant_root_cause = max(root_scores, key=root_scores.get)
            if dominant_root_cause == "interference_candidate" and overlap_etilt_score >= max(OVERLAP_ETILT_ACTION_SCORE, coverage_etilt_score + 10.0):
                dominant_root_cause = "interference_limited"

        action_family = "mixed"
        if dominant_root_cause == "coverage_candidate":
            action_family = "coverage"
        elif dominant_root_cause in {"interference_candidate", "interference_limited"}:
            action_family = "interference"
        elif dominant_root_cause == "directional_misalignment":
            action_family = "azimuth"
        elif dominant_root_cause == "blockage_dominated":
            action_family = "blockage"
        if strong_azimuth_override and azimuth_score >= AZIMUTH_ACTION_SCORE:
            action_family = "azimuth"

        relaxed_azimuth_case = (stable_bearing_case and bearing_sample_count >= max(20, MIN_BEARING_SAMPLE_COUNT - 10) and bearing_peak_share >= max(0.34, RELAXED_AZIMUTH_PEAK_SHARE - 0.12) and bearing_spread <= RELAXED_AZIMUTH_MAX_SPREAD_DEG and bad_sample_count >= RELAXED_AZIMUTH_BAD_SAMPLE_COUNT and not pd.isna(directional_contrast) and directional_contrast >= max(1.15, MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH - 0.15))
        azimuth_preferred_case = (directional_misalignment_case and azimuth_score >= AZIMUTH_ACTION_SCORE and action_family == "azimuth" and bearing_peak_share >= max(0.36, RELAXED_AZIMUTH_PEAK_SHARE - 0.10) and not pd.isna(directional_contrast) and directional_contrast >= max(1.35, STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH - 0.45) and (pd.isna(nlos_share) or nlos_share < MEDIUM_NLOS_SHARE_GATE) and azimuth_score >= max(overlap_etilt_score, coverage_etilt_score) + 3.0)
        family_priority = {"coverage": 1.5, "interference": 1.5, "azimuth": 1.5, "blockage": 0.5, "mixed": 1.0}

        def _family_bonus(candidate_family: str) -> float:
            if action_family == candidate_family:
                return 6.0
            if action_family == "mixed":
                return 2.0
            if action_family == "blockage":
                return -2.0 if candidate_family in {"coverage", "interference", "azimuth"} else 0.0
            return 0.0

        rec_etilt = curr_etilt
        etilt_reason = "No ETilt change required."
        etilt_status = "no_change"
        etilt_confidence_raw = max(coverage_etilt_score, overlap_etilt_score)
        etilt_confidence_floor = 0.0
        best_etilt_score = 0.0
        if not pd.isna(curr_etilt):
            if low_signal:
                etilt_reason = "Bad sample volume is too small for a reliable tilt change; collect more evidence before acting."
                etilt_status = "low_confidence_hold"
            else:
                etilt_candidates: List[Dict[str, object]] = [{"mode": "hold", "family": "hold", "delta": 0.0, "target": float(curr_etilt), "score": 0.0, "reason": "No ETilt change required."}]
                if overlap_etilt_score >= max(OVERLAP_ETILT_ACTION_SCORE, coverage_etilt_score + 2.0):
                    overlap_gain_1 = _estimate_etilt_validation_gain("overlap", 1.0, coverage_etilt_score, overlap_etilt_score, mean_dist, p90_dist, overlap_dense, interference_signal_present, coverage_signal_present, coverage_dominant, medium_edge_issue, far_edge_issue, nlos_share, los_blocked)
                    etilt_candidates.append({"mode": "overlap", "family": "interference", "delta": 1.0, "target": _bounded_etilt_target(curr_etilt, curr_etilt + 1.0), "score": overlap_etilt_score + overlap_gain_1 + _family_bonus("interference"), "reason": "Dense overlap and persistent interference evidence favor footprint tightening within safe ETilt bounds; increasing ETilt should reduce spillover."})
                    if bad_sinr >= max(8, bad_rsrp * 1.75) and overlap_etilt_score >= coverage_etilt_score + 8.0:
                        overlap_gain_2 = _estimate_etilt_validation_gain("overlap", 2.0, coverage_etilt_score, overlap_etilt_score, mean_dist, p90_dist, overlap_dense, interference_signal_present, coverage_signal_present, coverage_dominant, medium_edge_issue, far_edge_issue, nlos_share, los_blocked)
                        etilt_candidates.append({"mode": "overlap", "family": "interference", "delta": 2.0, "target": _bounded_etilt_target(curr_etilt, curr_etilt + 2.0), "score": overlap_etilt_score + overlap_gain_2 + _family_bonus("interference"), "reason": "Dense overlap and persistent interference evidence strongly favor footprint tightening within safe ETilt bounds; a larger ETilt increase should reduce spillover."})
                if coverage_etilt_score >= COVERAGE_ETILT_ACTION_SCORE:
                    coverage_gain_1 = _estimate_etilt_validation_gain("coverage", -1.0, coverage_etilt_score, overlap_etilt_score, mean_dist, p90_dist, overlap_dense, interference_signal_present, coverage_signal_present, coverage_dominant, medium_edge_issue, far_edge_issue, nlos_share, los_blocked)
                    etilt_candidates.append({"mode": "coverage", "family": "coverage", "delta": -1.0, "target": _bounded_etilt_target(curr_etilt, curr_etilt - 1.0), "score": coverage_etilt_score + coverage_gain_1 + _family_bonus("coverage"), "reason": "Persistent urban coverage weakness is present with edge-distance evidence, bounded by safe ETilt limits; reducing ETilt should open the footprint toward underserved users."})
                    if not pd.isna(p90_dist) and p90_dist >= 280.0 and coverage_etilt_score >= overlap_etilt_score + 6.0:
                        coverage_gain_2 = _estimate_etilt_validation_gain("coverage", -2.0, coverage_etilt_score, overlap_etilt_score, mean_dist, p90_dist, overlap_dense, interference_signal_present, coverage_signal_present, coverage_dominant, medium_edge_issue, far_edge_issue, nlos_share, los_blocked)
                        etilt_candidates.append({"mode": "coverage", "family": "coverage", "delta": -2.0, "target": _bounded_etilt_target(curr_etilt, curr_etilt - 2.0), "score": coverage_etilt_score + coverage_gain_2 + _family_bonus("coverage"), "reason": "Coverage-opening evidence is very strong at the cell edge, so a larger ETilt reduction is justified within safe bounds."})
                if action_family == "blockage" and blockage_limited_case and not high_blockage and coverage_etilt_score >= COVERAGE_ETILT_ACTION_SCORE - 4.0:
                    etilt_candidates.append({"mode": "coverage", "family": "coverage", "delta": -1.0, "target": _bounded_etilt_target(curr_etilt, curr_etilt - 1.0), "score": coverage_etilt_score + 1.0, "reason": "Blocked geometry is present, but a bounded 1 deg ETilt opening is allowed as a cautious secondary recovery path."})
                best_etilt = _select_best_candidate_with_gap(etilt_candidates, min_gap=2.0)
                rec_etilt = float(best_etilt["target"])
                best_etilt_score = float(best_etilt["score"])
                if best_etilt["mode"] != "hold" and not np.isclose(rec_etilt, float(curr_etilt), equal_nan=True):
                    etilt_reason = str(best_etilt["reason"])
                    etilt_status = "action_change"
                elif blockage_limited_case and overlap_etilt_score < OVERLAP_ETILT_ACTION_SCORE:
                    etilt_reason = "Bad RSRP occurs in blocked/NLOS geometry, so tilt is probably not the first fix; building or terrain blockage dominates the problem area."
                    etilt_status = "blocked_by_blockage"
                elif action_family == "blockage":
                    etilt_reason = "Blockage dominates this cell, so ETilt is held unless a cautious secondary recovery path shows enough value."
                    etilt_status = "blocked_by_blockage"

        rec_az = curr_az
        az_reason = "No azimuth change required."
        az_status = "no_change"
        az_confidence_raw = max(0.0, azimuth_score)
        best_az_score = 0.0
        if not pd.isna(curr_az):
            if low_signal:
                az_reason = "Bad sample volume is too small for a reliable azimuth change; hold azimuth until more evidence is available."
                az_status = "low_confidence_hold"
            elif str(swap_flag).strip().upper() == "YES":
                az_reason = "Swap sector is suspected, so azimuth should be held until mapping is validated."
                az_status = "hold_swap"
                az_confidence_raw = 0.0
            elif not pd.isna(nlos_share) and nlos_share >= AZIMUTH_NLOS_HARD_BLOCK_GATE:
                az_reason = "Extreme NLOS geometry makes dominant-bearing direction unreliable here, so azimuth is held."
                az_status = "blocked_by_blockage"
                az_confidence_raw = 0.0
            elif blockage_limited_case and not strong_azimuth_override:
                az_reason = "Blocked geometry makes directional steering unreliable here, so azimuth is held."
                az_status = "blocked_by_blockage"
                az_confidence_raw = 0.0
            elif azimuth_score >= AZIMUTH_ACTION_SCORE and (action_family == "azimuth" or site_persistent_issue or relaxed_azimuth_case or azimuth_preferred_case or strong_azimuth_override or directional_misalignment_case):
                signed_delta = _signed_azimuth_delta(dominant_bearing, curr_az)
                signed_delta = float(np.clip(signed_delta, -MAX_AZIMUTH_STEP_DEG, MAX_AZIMUTH_STEP_DEG))
                if abs(signed_delta) >= 5.0:
                    az_candidates: List[Dict[str, object]] = [
                        {"mode": "hold", "family": "hold", "delta": 0.0, "target": float(curr_az), "score": 0.0},
                        {"mode": "steer", "family": "azimuth", "delta": 5.0 if signed_delta > 0 else -5.0, "target": _normalize_azimuth(curr_az + (5.0 if signed_delta > 0 else -5.0)), "score": azimuth_score + _estimate_azimuth_validation_gain(curr_az, _normalize_azimuth(curr_az + (5.0 if signed_delta > 0 else -5.0)), dominant_bearing, azimuth_score, bearing_mismatch, bearing_peak_share, directional_contrast, bearing_spread, nlos_share, los_blocked, overlap_dense, interference_signal_present) + _family_bonus("azimuth")},
                    ]
                    if abs(signed_delta) > 7.5:
                        az_target_full = _normalize_azimuth(curr_az + float(np.clip(signed_delta, -MAX_AZIMUTH_STEP_DEG, MAX_AZIMUTH_STEP_DEG)))
                        az_candidates.append({"mode": "steer", "family": "azimuth", "delta": float(np.clip(signed_delta, -MAX_AZIMUTH_STEP_DEG, MAX_AZIMUTH_STEP_DEG)), "target": az_target_full, "score": azimuth_score + _estimate_azimuth_validation_gain(curr_az, az_target_full, dominant_bearing, azimuth_score, bearing_mismatch, bearing_peak_share, directional_contrast, bearing_spread, nlos_share, los_blocked, overlap_dense, interference_signal_present) + _family_bonus("azimuth")})
                    best_az = _select_best_candidate_with_gap(az_candidates, min_gap=2.0)
                    rec_az = float(best_az["target"])
                    best_az_score = float(best_az["score"])
                    az_reason = f"Stable degraded directional peak is offset from configured azimuth by {bearing_mismatch:.1f} deg with bearing spread {bearing_spread:.1f} deg, so a bounded azimuth correction is applied."
                    if np.isclose(rec_az, float(curr_az), equal_nan=True):
                        az_reason = "Observed dominant bearing is close enough to configured azimuth after candidate validation; no azimuth change needed."
                    else:
                        az_status = "action_change"
                else:
                    az_reason = "Observed dominant bearing is close enough to configured azimuth; no azimuth change needed."
            elif overlap_etilt_score >= OVERLAP_ETILT_ACTION_SCORE:
                az_reason = "Interference evidence is stronger for ETilt tightening than for azimuth steering, so azimuth is held."
                az_status = "prefer_tilt_first"
            elif not stable_bearing_case and bearing_sample_count > 0:
                az_reason = "Directional evidence is not stable enough for safe azimuth steering; hold azimuth until the corridor is narrower and more consistent."

        if etilt_status == "action_change" and az_status == "action_change":
            etilt_priority = best_etilt_score + family_priority.get(str(action_family), 1.0)
            az_priority = best_az_score + family_priority.get("azimuth" if action_family == "azimuth" else "mixed", 1.0)
            if action_family == "azimuth" or azimuth_preferred_case or (strong_azimuth_override and az_priority >= etilt_priority - 2.0):
                etilt_reason = "Azimuth candidate validated better than ETilt for the selected root-cause family, so tilt is held first."
                etilt_status = "prefer_azimuth_first"
                rec_etilt = curr_etilt
            elif best_etilt_score >= best_az_score + 4.0:
                az_reason = "ETilt candidate validated better than azimuth for this cell, so azimuth is held to avoid stacking a weaker change."
                az_status = "prefer_tilt_first"
                rec_az = curr_az
            elif best_az_score > best_etilt_score:
                etilt_reason = "Azimuth candidate showed the stronger validated directional correction, so ETilt is held first."
                etilt_status = "prefer_azimuth_first"
                rec_etilt = curr_etilt
            else:
                az_reason = "ETilt candidate showed the stronger validated improvement, so azimuth is held first."
                az_status = "prefer_tilt_first"
                rec_az = curr_az
        elif etilt_status == "action_change" and azimuth_preferred_case:
            etilt_reason = "Directional misalignment evidence is cleaner than ETilt evidence here, so tilt is held while azimuth is preferred first."
            etilt_status = "prefer_azimuth_first"
            rec_etilt = curr_etilt
        elif etilt_status == "action_change" and az_status == "no_change" and action_family == "azimuth" and best_az_score >= AZIMUTH_ACTION_SCORE - 2.0:
            etilt_reason = "Directional evidence points toward azimuth first, so ETilt is held until a clearer non-azimuth winner appears."
            etilt_status = "prefer_azimuth_first"
            rec_etilt = curr_etilt

        if etilt_confidence_raw >= COVERAGE_ETILT_ACTION_SCORE or etilt_confidence_raw >= OVERLAP_ETILT_ACTION_SCORE:
            if etilt_status in {"no_change", "prefer_azimuth_first", "prefer_tilt_first"}:
                etilt_confidence_floor = 55.0
        etilt_confidence_score = _clamp_score(max(0.0, etilt_confidence_raw if etilt_status == "action_change" else max(min(etilt_confidence_raw, 45.0), etilt_confidence_floor)))
        etilt_confidence_label = _confidence_label(etilt_confidence_score)

        az_confidence_floor = 0.0
        if az_confidence_raw >= AZIMUTH_ACTION_SCORE and az_status in {"no_change", "prefer_tilt_first"}:
            az_confidence_floor = 55.0
        az_confidence_score = _clamp_score(max(0.0, az_confidence_raw if az_status == "action_change" else max(min(az_confidence_raw, 45.0), az_confidence_floor)))
        az_confidence_label = _confidence_label(az_confidence_score)

        rec_power = curr_power
        pwr_reason = "No TX power change required."
        pwr_status = "no_change"
        pwr_confidence_raw = max(0.0, tx_power_score)
        if not pd.isna(curr_power):
            if low_signal:
                pwr_reason = "Bad sample volume is too small for a reliable power change recommendation."
                pwr_status = "low_confidence_hold"
            elif (tx_power_score >= TX_POWER_ACTION_SCORE and coverage_etilt_score >= COVERAGE_ETILT_ACTION_SCORE - 4.0 and overlap_etilt_score <= OVERLAP_ETILT_ACTION_SCORE - 10.0 and not interference_signal_present and not overlap_dense and not sinr_dominant and etilt_status != "action_change"):
                rec_power = curr_power + 1.0
                pwr_reason = "Coverage weakness looks distance-driven with low overlap risk and no stronger ETilt winner, so a small +1 dB increase is allowed as a secondary action."
                pwr_status = "action_change"
            elif interference_signal_present or overlap_dense:
                pwr_reason = "Power increase is avoided because the problem is interference/overlap driven."
                pwr_status = "held_for_interference"
        pwr_confidence_floor = 50.0 if pwr_confidence_raw >= TX_POWER_ACTION_SCORE and pwr_status == "no_change" else 0.0
        pwr_confidence_score = _clamp_score(max(0.0, pwr_confidence_raw if pwr_status == "action_change" else max(min(pwr_confidence_raw, 40.0), pwr_confidence_floor)))
        pwr_confidence_label = _confidence_label(pwr_confidence_score)

        base_context = (
            f"Root cause={dominant_root_cause}. Bad sample count={bad_sample_count}. "
            f"Geo context: mean_serving_distance_m={_fmt_num(mean_dist, 1)}, p90_serving_distance_m={_fmt_num(p90_dist, 1)}, "
            f"nlos_share={_fmt_num(nlos_share, 2)}, mean_los_blocked_ratio={_fmt_num(los_blocked, 2)}, clutter_mode={clutter_mode or 'n/a'}. "
            f"Scores: coverage_etilt={_fmt_num(coverage_etilt_score, 1)}, overlap_etilt={_fmt_num(overlap_etilt_score, 1)}, "
            f"azimuth={_fmt_num(azimuth_score, 1)}, tx_power={_fmt_num(tx_power_score, 1)}. "
            f"Bearing: peak_share={_fmt_num(bearing_peak_share, 2)}, spread_deg={_fmt_num(bearing_spread, 1)}, directional_contrast={_fmt_num(directional_contrast, 2)}."
        )

        rec_rows.extend([
            {"Cell ID": cell_id, "Technology": ant_tech or tech, "Parameter": "ETilt", "Current Value": "" if pd.isna(curr_etilt) else float(curr_etilt), "Recommended Value": "" if pd.isna(rec_etilt) else float(rec_etilt), "Reason": f"{etilt_reason} Confidence={etilt_confidence_label}({etilt_confidence_score}). {base_context}", "Swap Sector Detected": swap_flag, "Bad Sample Count": bad_sample_count, "P90 Serving Distance (m)": "" if pd.isna(p90_dist) else float(p90_dist), "Root Cause Category": dominant_root_cause, "Recommendation Status": etilt_status, "Recommendation Confidence": etilt_confidence_label, "Confidence Score": etilt_confidence_score, "Coverage ETilt Score": round(float(coverage_etilt_score), 2), "Overlap ETilt Score": round(float(overlap_etilt_score), 2), "Azimuth Score": round(float(azimuth_score), 2), "TX Power Score": round(float(tx_power_score), 2), "Bearing Peak Share": "" if pd.isna(bearing_peak_share) else round(float(bearing_peak_share), 4), "Bearing Spread Deg": "" if pd.isna(bearing_spread) else round(float(bearing_spread), 2), "Directional Contrast": "" if pd.isna(directional_contrast) else round(float(directional_contrast), 4)},
            {"Cell ID": cell_id, "Technology": ant_tech or tech, "Parameter": "Azimuth", "Current Value": "" if pd.isna(curr_az) else float(curr_az), "Recommended Value": "" if pd.isna(rec_az) else float(rec_az), "Reason": f"{az_reason} Confidence={az_confidence_label}({az_confidence_score}). {base_context}", "Swap Sector Detected": swap_flag, "Bad Sample Count": bad_sample_count, "P90 Serving Distance (m)": "" if pd.isna(p90_dist) else float(p90_dist), "Root Cause Category": dominant_root_cause, "Recommendation Status": az_status, "Recommendation Confidence": az_confidence_label, "Confidence Score": az_confidence_score, "Coverage ETilt Score": round(float(coverage_etilt_score), 2), "Overlap ETilt Score": round(float(overlap_etilt_score), 2), "Azimuth Score": round(float(azimuth_score), 2), "TX Power Score": round(float(tx_power_score), 2), "Bearing Peak Share": "" if pd.isna(bearing_peak_share) else round(float(bearing_peak_share), 4), "Bearing Spread Deg": "" if pd.isna(bearing_spread) else round(float(bearing_spread), 2), "Directional Contrast": "" if pd.isna(directional_contrast) else round(float(directional_contrast), 4)},
            {"Cell ID": cell_id, "Technology": ant_tech or tech, "Parameter": "TX Power", "Current Value": "" if pd.isna(curr_power) else float(curr_power), "Recommended Value": "" if pd.isna(rec_power) else float(rec_power), "Reason": f"{pwr_reason} Confidence={pwr_confidence_label}({pwr_confidence_score}). {base_context}", "Swap Sector Detected": swap_flag, "Bad Sample Count": bad_sample_count, "P90 Serving Distance (m)": "" if pd.isna(p90_dist) else float(p90_dist), "Root Cause Category": dominant_root_cause, "Recommendation Status": pwr_status, "Recommendation Confidence": pwr_confidence_label, "Confidence Score": pwr_confidence_score, "Coverage ETilt Score": round(float(coverage_etilt_score), 2), "Overlap ETilt Score": round(float(overlap_etilt_score), 2), "Azimuth Score": round(float(azimuth_score), 2), "TX Power Score": round(float(tx_power_score), 2), "Bearing Peak Share": "" if pd.isna(bearing_peak_share) else round(float(bearing_peak_share), 4), "Bearing Spread Deg": "" if pd.isna(bearing_spread) else round(float(bearing_spread), 2), "Directional Contrast": "" if pd.isna(directional_contrast) else round(float(directional_contrast), 4)},
        ])

    return pd.DataFrame(rec_rows, columns=["Cell ID", "Technology", "Parameter", "Current Value", "Recommended Value", "Reason", "Swap Sector Detected", "Bad Sample Count", "P90 Serving Distance (m)", "Root Cause Category", "Recommendation Status", "Recommendation Confidence", "Confidence Score", "Coverage ETilt Score", "Overlap ETilt Score", "Azimuth Score", "TX Power Score", "Bearing Peak Share", "Bearing Spread Deg", "Directional Contrast"])


def prepare_recommendation_exports(recommendations_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if recommendations_df.empty:
        return recommendations_df.copy(), recommendations_df.copy()
    full_df = recommendations_df.copy()
    changed_mask = _changed_recommendation_mask(full_df)
    status = full_df["Recommendation Status"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    filtered_df = full_df.loc[changed_mask | status.isin({"blocked_by_blockage", "hold_swap"})].copy()
    return full_df, filtered_df
