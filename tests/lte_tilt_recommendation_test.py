from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import text

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency for optimized candidate search
    optuna = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.lte_rf_debug_lab import DEFAULT_PROJECT_ID, DEFAULT_REGION, _write_json
from tools.lte_prediction_optimised import ml_engine as opt_ml


OUTPUT_ROOT = Path("tests/output")

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
MAX_ETILT_INCREASE_PER_RUN_DEG = 4.0
MAX_ETILT_DECREASE_PER_RUN_DEG = 4.0
RELAXED_AZIMUTH_BAD_SAMPLE_COUNT = 183
RELAXED_AZIMUTH_PEAK_SHARE = 0.5721666812548345
RELAXED_AZIMUTH_MAX_SPREAD_DEG = 32.264493787350155
MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH = 1.2734413438634005
STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH = 2.542748090042536
MIN_CANDIDATE_SCORE_GAP = 0.05
SITE_SYMMETRY_PENALTY_BAD_SAMPLE_RATIO = 0.055289002855823374

COVERAGE_ETILT_ACTION_SCORE = 51.49084372010145
OVERLAP_ETILT_ACTION_SCORE = 47.31004233364023
AZIMUTH_ACTION_SCORE = 71.61142299538227
TX_POWER_ACTION_SCORE = 46.25093121890758

OPTUNA_TRIAL_COUNT = 12
OPTUNA_N_JOBS = 4
COARSE_RF_VALIDATION_COUNT = 8
COARSE_MIN_SCORE_FOR_REFINEMENT = -0.05
COARSE_ETILT_DELTAS = (-6.0, -3.0, 3.0, 6.0)
COARSE_AZIMUTH_DELTAS = (-20.0, 20.0)
COARSE_TX_POWER_DELTAS = (-2.0, 2.0)
OPTUNA_PREFILTER_SCORE = -50.0
OPTUNA_PRUNE_GOOD_AREA_LOSS_PCT = 80.0
OPTUNA_PRUNE_MEAN_SINR_DELTA_DB = -8.0
SEVERE_GOOD_AREA_LOSS_PCT = 55.0
SEVERE_MEAN_SINR_DROP_DB = -5.0
ENABLE_GEOMETRY_PREFILTER_PRUNING = False
MAX_COORDINATED_ACTION_SITES = 3
MAX_COORDINATED_NEIGHBOR_SITES = 2
MAX_COORDINATED_ACTION_CELLS = 6
UPSTREAM_EXPLORATION_SCORE_FLOOR = -500.0
EXPLORATORY_ETILT_DELTA_DEG = 4.0
EXPLORATORY_AZIMUTH_STEP_DEG = 20.0
EXPLORATORY_TX_POWER_DELTA_DB = 2.0

INTERFERENCE_GAP_STRONG_DB = 3.0
INTERFERENCE_GAP_WEAK_DB = 6.0
SAME_EARFCN_INTERFERER_SHARE_GATE = 0.35
DOMINANT_INTERFERER_SHARE_GATE = 0.15
COVERAGE_LOW_RSRP_SHARE_GATE = 0.35
QUALITY_LOW_SINR_SHARE_GATE = 0.25

NET_BAD_REDUCTION_WEIGHT = 2.0
SINR_RECOVERY_WEIGHT = 1.2
RSRQ_RECOVERY_WEIGHT = 0.8
RSRP_RECOVERY_WEIGHT = 0.4
SINR_NEW_BAD_WEIGHT = 0.8
RSRQ_NEW_BAD_WEIGHT = 1.0
RSRP_NEW_BAD_WEIGHT = 0.5
SINR_SEVERITY_REDUCTION_WEIGHT = 8.0
RSRQ_SEVERITY_REDUCTION_WEIGHT = 4.0
RSRP_SEVERITY_REDUCTION_WEIGHT = 1.5
TOTAL_SEVERITY_REDUCTION_WEIGHT = 0.0
GOOD_AREA_LOSS_WEIGHT = 0.25
EXPLORATORY_REVIEW_SCORE_FLOOR = -0.15
SEVERE_CONSTRAINT_PENALTY = 25.0


@dataclass
class TiltRecommendationTestConfig:
    project_id: int = DEFAULT_PROJECT_ID
    region: str = DEFAULT_REGION
    operator: Optional[str] = None
    rsrp_threshold: float = -105.0
    rsrq_threshold: float = -15.0
    sinr_threshold: float = 0.0
    validate_candidates: bool = True
    radius_m: float = 500.0
    grid_resolution_m: float = 30.0
    workers: int = 4
    impact_radius_m: float = 800.0
    neighbor_site_count: int = 3
    max_interference_sites: int = 10
    max_good_area_loss_pct: float = 2.0
    max_mean_sinr_drop_db: float = 1.0
    min_score_gain: float = 0.0
    min_recovered_bad_samples: int = 3
    max_ranked_cells: int = 20
    max_ranked_sites: int = 8
    threshold_file_path: Optional[str] = None
    threshold_constraint_count: int = 0
    threshold_optimised_count: int = 0
    output_root: Path = OUTPUT_ROOT


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _derive_antenna_cell_key(antenna_df: pd.DataFrame) -> pd.Series:
    node_cell_col = TILT_SRC._find_col(antenna_df, ["Node_Cell_ID", "node_cell_id"], required=False)
    if node_cell_col:
        node_cell = antenna_df[node_cell_col].map(TILT_SRC._norm_cell_id)
        if (node_cell != "").any():
            return node_cell

    nodeb_col = TILT_SRC._find_col(antenna_df, ["nodeb_id", "nodeb", "site_id"], required=False)
    cell_col = TILT_SRC._find_col(antenna_df, ["cell_id", "eci", "local_cell_id"], required=False)

    if nodeb_col and cell_col:
        nodeb = antenna_df[nodeb_col].map(TILT_SRC._norm_cell_id)
        cell = antenna_df[cell_col].map(TILT_SRC._norm_cell_id)
        combined = pd.Series(
            np.where((nodeb != "") & (cell != ""), nodeb + "_" + cell, ""),
            index=antenna_df.index,
        )
        if (combined != "").any():
            return combined
        if (cell != "").any():
            return cell

    if cell_col:
        return antenna_df[cell_col].map(TILT_SRC._norm_cell_id)

    return pd.Series("", index=antenna_df.index)


def _load_tilt_module():
    module_path = PROJECT_ROOT / "tools" / "lte_tilt_recommandation" / "etilt_optimizer_cd2.py"
    module_name = "tests._tilt_source_module"
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(module_path),
            "dummy_log.csv",
            "dummy_antenna.csv",
            "-105",
            "-15",
            "0",
        ]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load tilt source module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv


TILT_SRC = _load_tilt_module()


def _fetch_baseline_log_df(project_id: int, region: str, operator: Optional[str]) -> pd.DataFrame:
    baseline_df = opt_ml.fetch_baseline(project_id, region=region).copy()
    baseline_job_id = _fetch_latest_baseline_job_id(project_id, region)
    baseline_df = baseline_df.loc[baseline_df["job_id"].astype(str) == str(baseline_job_id)].copy()
    if operator:
        operator_norm = str(operator).strip().lower()
        op_col = next((c for c in ["operator", "Operator"] if c in baseline_df.columns), None)
        if op_col:
            baseline_df = baseline_df.loc[
                baseline_df[op_col].astype(str).str.strip().str.lower() == operator_norm
            ].copy()

    if baseline_df.empty:
        raise FileNotFoundError(f"No baseline rows found for project_id={project_id} region={region}")

    baseline_df["node_b_id"] = baseline_df["Node_Cell_ID"].astype(str).str.split("_").str[0]
    baseline_df["nodeb_id"] = baseline_df["Node_Cell_ID"].astype(str).str.split("_").str[0]
    baseline_df["cell_id"] = baseline_df["Node_Cell_ID"].astype(str).str.split("_").str[-1]
    baseline_df["local_cell_id"] = baseline_df["Node_Cell_ID"].astype(str).str.split("_").str[-1]
    if "operator" not in baseline_df.columns:
        baseline_df["operator"] = operator or "Unknown"
    baseline_df["pred_rsrp"] = pd.to_numeric(baseline_df["pred_rsrp"], errors="coerce")
    baseline_df["pred_rsrq"] = pd.to_numeric(baseline_df["pred_rsrq"], errors="coerce")
    baseline_df["pred_sinr"] = pd.to_numeric(baseline_df["pred_sinr"], errors="coerce")
    baseline_df = _attach_baseline_topology_context_test_only(baseline_df, project_id, region, baseline_job_id)
    return baseline_df


def _fetch_antenna_df(project_id: int, region: str, operator: Optional[str]) -> pd.DataFrame:
    site_df = opt_ml.fetch_site_data(project_id, region=region, operator=operator).copy()
    if site_df.empty:
        raise FileNotFoundError(f"No site rows found for project_id={project_id} region={region}")
    return site_df


def _enrich_log_with_antenna_context(log_df: pd.DataFrame, antenna_df: pd.DataFrame) -> pd.DataFrame:
    out = log_df.copy()
    ant = antenna_df.copy()
    ant["Node_Cell_ID"] = ant["Node_Cell_ID"].astype(str).str.strip()
    merge_cols = [
        col for col in [
            "Node_Cell_ID",
            "Technology",
            "lat",
            "lon",
            "azimuth",
            "electrical_tilt",
            "mechanical_tilt",
            "tx_power",
            "antenna_height",
            "dashboard_site_id",
        ] if col in ant.columns
    ]
    ant = ant[merge_cols].drop_duplicates(subset=["Node_Cell_ID"], keep="last")
    out = out.merge(ant, on="Node_Cell_ID", how="left", suffixes=("", "_site"))
    if "Technology" not in out.columns:
        out["Technology"] = "4G"
    else:
        out["Technology"] = out["Technology"].fillna("4G")
    return out


def _fetch_geo_df(project_id: int, region: str, operator: Optional[str], antenna_df: pd.DataFrame) -> pd.DataFrame:
    affected_cells = sorted(antenna_df["Node_Cell_ID"].astype(str).unique().tolist())
    geo_df = opt_ml.fetch_geo_features(project_id, region=region, affected_cells=affected_cells).copy()
    if geo_df.empty:
        return geo_df
    if operator:
        site_cells = antenna_df[["Node_Cell_ID"]].copy()
        site_cells["Node_Cell_ID"] = site_cells["Node_Cell_ID"].astype(str)
        geo_df = geo_df.merge(site_cells.drop_duplicates(), on="Node_Cell_ID", how="inner")
    return geo_df


def _compute_dominant_bearing_summary(log_df: pd.DataFrame, antenna_df: pd.DataFrame) -> pd.DataFrame:
    log_work = TILT_SRC._normalize_columns(log_df).copy()
    ant_work = TILT_SRC._normalize_columns(antenna_df).copy()

    log_cell_col = TILT_SRC._find_col(log_work, ["Node_Cell_ID", "node_cell_id"], required=False)
    ant_cell_col = TILT_SRC._find_col(ant_work, ["Node_Cell_ID", "node_cell_id"], required=False)
    log_work["Cell ID"] = (
        log_work[log_cell_col].astype(str).map(TILT_SRC._norm_cell_id)
        if log_cell_col else TILT_SRC._get_cell_key_from_log(log_work)
    )
    ant_work["Cell ID"] = (
        ant_work[ant_cell_col].astype(str).map(TILT_SRC._norm_cell_id)
        if ant_cell_col else _derive_antenna_cell_key(ant_work)
    )

    lat_col = TILT_SRC._find_col(log_work, ["lat", "latitude"], required=False)
    lon_col = TILT_SRC._find_col(log_work, ["lon", "longitude"], required=False)
    rsrp_col = TILT_SRC._find_col(log_work, ["rsrp", "pred_rsrp", "csi_rsrp"], required=False)
    rsrq_col = TILT_SRC._find_col(log_work, ["rsrq", "pred_rsrq", "csi_rsrq"], required=False)
    sinr_col = TILT_SRC._find_col(log_work, ["sinr", "pred_sinr", "csi_sinr"], required=False)

    ant_lat_col = TILT_SRC._find_col(ant_work, ["latitude", "lat"], required=False)
    ant_lon_col = TILT_SRC._find_col(ant_work, ["longitude", "lon"], required=False)
    az_col = TILT_SRC._find_col(ant_work, ["azimuth", "azi"], required=False)

    log_work["RSRP_eval"] = pd.to_numeric(log_work[rsrp_col], errors="coerce") if rsrp_col else np.nan
    log_work["RSRQ_eval"] = pd.to_numeric(log_work[rsrq_col], errors="coerce") if rsrq_col else np.nan
    log_work["SINR_eval"] = pd.to_numeric(log_work[sinr_col], errors="coerce") if sinr_col else np.nan
    log_work["Lat_eval"] = pd.to_numeric(log_work[lat_col], errors="coerce") if lat_col else np.nan
    log_work["Lon_eval"] = pd.to_numeric(log_work[lon_col], errors="coerce") if lon_col else np.nan
    log_work["severity_score"] = (
        (float(TILT_SRC.RSRP_THRESH) - log_work["RSRP_eval"]).clip(lower=0).fillna(0)
        + (float(TILT_SRC.RSRQ_THRESH) - log_work["RSRQ_eval"]).clip(lower=0).fillna(0)
        + (float(TILT_SRC.SINR_THRESH) - log_work["SINR_eval"]).clip(lower=0).fillna(0)
    )

    ant_map = (
        ant_work.drop_duplicates(subset=["Cell ID"])
        .assign(
            Azimuth_cfg=lambda d: pd.to_numeric(d[az_col], errors="coerce").map(TILT_SRC._normalize_azimuth)
            if az_col else np.nan,
            AntLat=lambda d: pd.to_numeric(d[ant_lat_col], errors="coerce") if ant_lat_col else np.nan,
            AntLon=lambda d: pd.to_numeric(d[ant_lon_col], errors="coerce") if ant_lon_col else np.nan,
        )
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

        bad_dir = g2[g2["severity_score"] > 0].copy()
        if bad_dir.empty:
            continue
        if len(bad_dir) >= MIN_BEARING_SAMPLE_COUNT:
            use_dir = bad_dir
        else:
            use_dir = g2.nlargest(min(len(g2), MIN_BEARING_SAMPLE_COUNT), "severity_score").copy()
            use_dir = use_dir[use_dir["severity_score"] > 0]
            if use_dir.empty:
                continue

        use_dir["bearing"] = use_dir.apply(
            lambda r: TILT_SRC._bearing_deg(
                ant["AntLat"],
                ant["AntLon"],
                r["Lat_eval"],
                r["Lon_eval"],
            ),
            axis=1,
        )
        peak_summary = _directional_peak_summary(use_dir["bearing"], use_dir["severity_score"])
        dominant_bearing = peak_summary["peak_bearing_deg"]
        bearing_spread = peak_summary["peak_spread_deg"]
        rows.append(
            {
                "Cell ID": cell_id,
                "dominant_bearing_deg": dominant_bearing,
                "configured_azimuth_deg": ant.get("Azimuth_cfg", np.nan),
                "bearing_mismatch_deg": TILT_SRC._angular_diff(dominant_bearing, ant.get("Azimuth_cfg", np.nan)),
                "bearing_sample_count": int(len(use_dir)),
                "bearing_spread_deg": bearing_spread,
                "bearing_peak_share": peak_summary["peak_share"],
                "bearing_peak_weight": peak_summary["peak_weight"],
                "bearing_second_peak_weight": peak_summary["second_peak_weight"],
                "bearing_directional_contrast": peak_summary["directional_contrast"],
            }
        )

    return pd.DataFrame(rows)


def _attach_geo_to_bad_samples(bad_df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    out = bad_df.copy()
    if geo_df.empty:
        return out

    geo_work = geo_df.copy()
    out["Cell ID"] = out["Cell ID"].map(TILT_SRC._norm_cell_id)
    geo_work["Cell ID"] = geo_work["Node_Cell_ID"].astype(str).map(TILT_SRC._norm_cell_id)

    lat_col = TILT_SRC._find_col(out, ["lat", "latitude"], required=False)
    lon_col = TILT_SRC._find_col(out, ["lon", "longitude"], required=False)
    if not lat_col or not lon_col:
        return out

    out["lat_6dp"] = pd.to_numeric(out[lat_col], errors="coerce").round(6)
    out["lon_6dp"] = pd.to_numeric(out[lon_col], errors="coerce").round(6)
    geo_work["lat_6dp"] = pd.to_numeric(geo_work["lat"], errors="coerce").round(6)
    geo_work["lon_6dp"] = pd.to_numeric(geo_work["lon"], errors="coerce").round(6)

    geo_cols = [
        "Cell ID",
        "lat_6dp",
        "lon_6dp",
        "clutter_class",
        "morphology_cluster",
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "los_blocker_count",
        "los_blocked_ratio",
        "max_blocker_height_m",
        "diffraction_proxy_db",
        "nlos_flag",
        "terrain_elevation_m",
        "terrain_slope_deg",
        "proxy_site_elevation_m",
        "terrain_relief_to_site_m",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "azimuth_delta_deg",
    ]
    geo_cols = [c for c in geo_cols if c in geo_work.columns]
    return out.merge(
        geo_work[geo_cols].drop_duplicates(subset=["Cell ID", "lat_6dp", "lon_6dp"], keep="last"),
        on=["Cell ID", "lat_6dp", "lon_6dp"],
        how="left",
    )


def _mode_or_blank(series: pd.Series) -> str:
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return ""
    mode = vals.mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else ""


def _fmt_num(value: float, decimals: int) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.{decimals}f}"


def _norm_reason_token(value: str) -> str:
    token = str(value or "").strip().lower()
    token = token.replace("/", "_").replace("-", "_").replace(" ", "_")
    return token or "unknown"


def _values_changed(curr, rec) -> bool:
    curr_num = pd.to_numeric(pd.Series([curr]), errors="coerce").iloc[0]
    rec_num = pd.to_numeric(pd.Series([rec]), errors="coerce").iloc[0]
    if pd.notna(curr_num) and pd.notna(rec_num):
        return not np.isclose(float(curr_num), float(rec_num), equal_nan=True)
    return str(curr) != str(rec)


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(float(value)))))


def _confidence_label(score: int) -> str:
    if score >= 75:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def _changed_recommendation_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    return df.apply(
        lambda r: _values_changed(r.get("Current Value"), r.get("Recommended Value")),
        axis=1,
    )


def _circular_spread_deg(values: pd.Series, center_deg: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if vals.empty or pd.isna(center_deg):
        return np.nan
    diffs = vals.map(lambda v: TILT_SRC._angular_diff(v, center_deg))
    return float(pd.to_numeric(diffs, errors="coerce").mean())


def _directional_peak_summary(
    values: pd.Series,
    weights: pd.Series,
    bin_size_deg: float = BEARING_BIN_SIZE_DEG,
) -> Dict[str, float]:
    bearings = pd.to_numeric(values, errors="coerce")
    bearing_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    work = pd.DataFrame({"bearing": bearings, "weight": bearing_weights}).dropna(subset=["bearing"])
    if work.empty:
        return {
            "peak_bearing_deg": np.nan,
            "peak_spread_deg": np.nan,
            "peak_share": np.nan,
            "peak_weight": 0.0,
            "second_peak_weight": 0.0,
            "directional_contrast": np.nan,
            "total_weight": 0.0,
        }

    work["bearing"] = work["bearing"].astype(float).map(TILT_SRC._normalize_azimuth)
    work["weight"] = work["weight"].astype(float).clip(lower=0.0)
    work["weight"] = work["weight"].where(work["weight"] > 0.0, 1.0)

    bin_count = max(1, int(round(360.0 / float(bin_size_deg))))
    bin_idx = np.floor(work["bearing"].to_numpy() / float(bin_size_deg)).astype(int) % bin_count
    work["bin_idx"] = bin_idx
    bin_weights = np.bincount(bin_idx, weights=work["weight"].to_numpy(), minlength=bin_count).astype(float)
    smooth_weights = bin_weights.copy()
    if bin_count > 1:
        smooth_weights = bin_weights + np.roll(bin_weights, 1) + np.roll(bin_weights, -1)

    peak_bin = int(np.argmax(smooth_weights))
    bin_distance = ((work["bin_idx"] - peak_bin + bin_count / 2.0) % bin_count) - (bin_count / 2.0)
    local_lobe = work.loc[bin_distance.abs() <= 1].copy()
    if local_lobe.empty:
        local_lobe = work.copy()

    peak_bearing = TILT_SRC._circular_mean_deg(local_lobe["bearing"])
    peak_spread = _circular_spread_deg(local_lobe["bearing"], peak_bearing)
    peak_weight = float(local_lobe["weight"].sum())
    total_weight = float(work["weight"].sum())
    peak_share = peak_weight / total_weight if total_weight > 0 else np.nan
    protected_bins = {(peak_bin - 1) % bin_count, peak_bin % bin_count, (peak_bin + 1) % bin_count}
    second_peak_weight = float(
        max((smooth_weights[idx] for idx in range(bin_count) if idx not in protected_bins), default=0.0)
    )
    directional_contrast = peak_weight / second_peak_weight if second_peak_weight > 0 else np.inf
    return {
        "peak_bearing_deg": peak_bearing,
        "peak_spread_deg": peak_spread,
        "peak_share": peak_share,
        "peak_weight": peak_weight,
        "second_peak_weight": second_peak_weight,
        "directional_contrast": directional_contrast,
        "total_weight": total_weight,
    }


def _aggregate_bad_geo_context(bad_geo_df: pd.DataFrame) -> pd.DataFrame:
    if bad_geo_df.empty:
        return pd.DataFrame()

    work = bad_geo_df.copy()
    if "nlos_flag" in work.columns:
        work["nlos_flag_num"] = pd.to_numeric(work["nlos_flag"], errors="coerce").fillna(0.0)
    else:
        work["nlos_flag_num"] = 0.0
    required_numeric_cols = [
        "serving_distance_m",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "site_count_250m",
        "site_count_500m",
        "azimuth_delta_deg",
        "building_area_ratio",
        "building_count",
        "los_blocked_ratio",
        "los_blocker_count",
        "green_ratio",
        "water_ratio",
        "road_length_m",
        "terrain_slope_deg",
        "terrain_relief_to_site_m",
    ]
    for col in required_numeric_cols:
        if col not in work.columns:
            work[col] = np.nan
    if "clutter_class" not in work.columns:
        work["clutter_class"] = ""
    grouped = work.groupby(["Cell ID", "Technology"], dropna=False)
    summary = grouped.agg(
        bad_sample_count=("Cell ID", "size"),
        mean_serving_distance_m=("serving_distance_m", "mean"),
        p90_serving_distance_m=(
            "serving_distance_m",
            lambda s: pd.to_numeric(s, errors="coerce").dropna().quantile(0.90)
            if pd.to_numeric(s, errors="coerce").dropna().size
            else np.nan,
        ),
        mean_nearest_site_distance_m=("nearest_site_distance_m", "mean"),
        mean_nearest3_site_distance_m=("mean_nearest3_site_distance_m", "mean"),
        mean_site_count_250m=("site_count_250m", "mean"),
        mean_site_count_500m=("site_count_500m", "mean"),
        mean_azimuth_delta_deg=("azimuth_delta_deg", "mean"),
        mean_building_area_ratio=("building_area_ratio", "mean"),
        mean_building_count=("building_count", "mean"),
        mean_los_blocked_ratio=("los_blocked_ratio", "mean"),
        mean_los_blocker_count=("los_blocker_count", "mean"),
        mean_green_ratio=("green_ratio", "mean"),
        mean_water_ratio=("water_ratio", "mean"),
        mean_road_length_m=("road_length_m", "mean"),
        mean_terrain_slope_deg=("terrain_slope_deg", "mean"),
        mean_terrain_relief_to_site_m=("terrain_relief_to_site_m", "mean"),
        nlos_share=("nlos_flag_num", "mean"),
        clutter_mode=("clutter_class", _mode_or_blank),
    ).reset_index()
    return summary


def _signed_azimuth_delta(target_deg: float, current_deg: float) -> float:
    if pd.isna(target_deg) or pd.isna(current_deg):
        return np.nan
    return ((float(target_deg) - float(current_deg) + 180.0) % 360.0) - 180.0


def _normalize_constraint_bool_test_only(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _clean_threshold_cell_id_test_only(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text[:-2] if text.endswith(".0") else text
    return TILT_SRC._norm_cell_id(text)


def _resolve_threshold_file_path_test_only(path_value: Optional[str]) -> str:
    supplied = str(path_value or "").strip()
    if not supplied:
        return ""
    path = Path(supplied)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _load_threshold_constraints_test_only(file_path: Optional[str]) -> pd.DataFrame:
    resolved = _resolve_threshold_file_path_test_only(file_path)
    if not resolved:
        return pd.DataFrame()
    path = Path(resolved)
    if not path.exists():
        raise FileNotFoundError(f"Threshold constraint file not found: {path}")
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    if "cell_id" not in lower_map:
        raise ValueError("Threshold constraint file must contain a cell_id column")

    rename_map = {}
    for col in [
        "cell_id",
        "min_m_tilt", "max_m_tilt",
        "min_e_tilt", "max_e_tilt",
        "min_height", "max_height",
        "min_azimuth", "max_azimuth",
        "min_tx_power", "max_tx_power",
        "optimised",
    ]:
        if col in lower_map:
            rename_map[lower_map[col]] = col
    df = df.rename(columns=rename_map)
    df["cell_id"] = df["cell_id"].map(_clean_threshold_cell_id_test_only)
    if "optimised" not in df.columns:
        df["optimised"] = False
    df["optimised"] = df["optimised"].map(_normalize_constraint_bool_test_only)
    for col in [
        "min_m_tilt", "max_m_tilt",
        "min_e_tilt", "max_e_tilt",
        "min_height", "max_height",
        "min_azimuth", "max_azimuth",
        "min_tx_power", "max_tx_power",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(subset=["cell_id"], keep="last")


def _constraint_map_test_only(constraint_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    if constraint_df.empty:
        return {}
    eligible = constraint_df.loc[constraint_df["optimised"] == True].copy()
    if eligible.empty:
        return {}
    return eligible.set_index("cell_id").to_dict("index")


def _azimuth_in_constraint_range_test_only(value: float, min_az: float, max_az: float) -> bool:
    value = float(value) % 360.0
    min_az = float(min_az) % 360.0
    max_az = float(max_az) % 360.0
    if min_az <= max_az:
        return min_az <= value <= max_az
    return value >= min_az or value <= max_az


def _clamp_azimuth_to_constraint_test_only(value: float, min_az: float, max_az: float) -> float:
    value = float(value) % 360.0
    min_norm = float(min_az) % 360.0
    max_norm = float(max_az) % 360.0
    if _azimuth_in_constraint_range_test_only(value, min_norm, max_norm):
        return value
    min_dist = abs(_signed_azimuth_delta(value, min_norm))
    max_dist = abs(_signed_azimuth_delta(value, max_norm))
    return min_norm if min_dist <= max_dist else max_norm


def _clip_target_to_user_constraint_test_only(
    cell_id: str,
    parameter: str,
    target_value: float,
    constraint_map: Dict[str, Dict[str, object]],
) -> float:
    cfg = constraint_map.get(_clean_threshold_cell_id_test_only(cell_id))
    if not cfg or pd.isna(target_value):
        return float(target_value)
    param = str(parameter).strip().lower()
    if param == "etilt":
        min_col, max_col = "min_e_tilt", "max_e_tilt"
    elif param == "azimuth":
        min_col, max_col = "min_azimuth", "max_azimuth"
    elif param in {"tx power", "power"}:
        min_col, max_col = "min_tx_power", "max_tx_power"
    elif param in {"mechanical tilt", "mtilt"}:
        min_col, max_col = "min_m_tilt", "max_m_tilt"
    elif param in {"height", "antenna height"}:
        min_col, max_col = "min_height", "max_height"
    else:
        return float(target_value)

    min_allowed = cfg.get(min_col)
    max_allowed = cfg.get(max_col)
    if pd.isna(min_allowed) or pd.isna(max_allowed):
        return float(target_value)
    if param == "azimuth":
        return float(_clamp_azimuth_to_constraint_test_only(float(target_value), float(min_allowed), float(max_allowed)))
    return float(min(max(float(target_value), float(min_allowed)), float(max_allowed)))


def _bounded_etilt_target(current_etilt: float, requested_etilt: float) -> float:
    if pd.isna(current_etilt) or pd.isna(requested_etilt):
        return np.nan
    lower_bound = max(float(MIN_SAFE_ETILT_DEG), float(current_etilt) - float(MAX_ETILT_DECREASE_PER_RUN_DEG))
    upper_bound = min(float(MAX_SAFE_ETILT_DEG), float(current_etilt) + float(MAX_ETILT_INCREASE_PER_RUN_DEG))
    return float(np.clip(float(requested_etilt), lower_bound, upper_bound))


def _exploratory_etilt_target(current_etilt: float, requested_etilt: float) -> float:
    if pd.isna(current_etilt) or pd.isna(requested_etilt):
        return np.nan
    lower_bound = max(float(MIN_SAFE_ETILT_DEG), float(current_etilt) - float(EXPLORATORY_ETILT_DELTA_DEG))
    upper_bound = min(float(MAX_SAFE_ETILT_DEG), float(current_etilt) + float(EXPLORATORY_ETILT_DELTA_DEG))
    return float(np.clip(float(requested_etilt), lower_bound, upper_bound))

def _prepare_recommendation_exports(recommendations_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if recommendations_df.empty:
        return recommendations_df.copy(), recommendations_df.copy()

    full_df = recommendations_df.copy()
    changed_mask = _changed_recommendation_mask(full_df)
    status = full_df["Recommendation Status"].astype(str).map(_norm_reason_token)
    keep_mask = changed_mask | status.isin({"blocked_by_blockage", "hold_swap"})
    filtered_df = full_df.loc[keep_mask].copy()
    return full_df, filtered_df


def _resolve_engine(region: str):
    current_engine = opt_ml.engine.get(str(region).lower(), opt_ml.engine["india"])
    if current_engine is None:
        raise RuntimeError(f"No DB engine configured for region={region}")
    return current_engine


def _fetch_latest_baseline_job_id(project_id: int, region: str) -> str:
    current_engine = _resolve_engine(region)
    query = text(
        """
        SELECT job_id
        FROM lte_prediction_baseline_results
        WHERE project_id = :project_id
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    with current_engine.connect() as conn:
        row = conn.execute(query, {"project_id": int(project_id)}).fetchone()
    if not row or row[0] is None:
        raise FileNotFoundError(f"No baseline results found for project_id={project_id}")
    return str(row[0])


def _attach_baseline_topology_context_test_only(
    baseline_df: pd.DataFrame,
    project_id: int,
    region: str,
    baseline_job_id: str,
) -> pd.DataFrame:
    topology_cols = [
        "serving_pci",
        "serving_earfcn",
        "serving_frequency_mhz",
        "best_interferer_cell_id",
        "best_interferer_pci",
        "best_interferer_earfcn",
        "best_interferer_distance_m",
        "best_interferer_azimuth_delta_deg",
        "best_interferer_proxy_phys_dbm",
        "neighbor_1_cell_id",
        "neighbor_1_pci",
        "neighbor_1_earfcn",
        "neighbor_1_proxy_rsrp_dbm",
        "neighbor_1_distance_m",
        "neighbor_1_azimuth_delta_deg",
        "neighbor_2_cell_id",
        "neighbor_2_pci",
        "neighbor_2_earfcn",
        "neighbor_2_proxy_rsrp_dbm",
        "neighbor_2_distance_m",
        "neighbor_2_azimuth_delta_deg",
        "interference_gap_db",
        "interference_ratio_linear",
        "interference_sum_proxy_dbm",
        "sinr_proxy_db",
        "rsrq_proxy_db",
        "same_earfcn_interferer_count",
        "dominant_interferer_count",
        "interference_selection_mode",
    ]
    current_engine = _resolve_engine(region)
    schema_query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = 'lte_prediction_baseline_results'
        """
    )
    with current_engine.connect() as conn:
        schema_df = pd.read_sql(schema_query, conn)
    schema_col = "column_name" if "column_name" in schema_df.columns else "COLUMN_NAME"
    available_cols = set(schema_df[schema_col].astype(str))
    selected_topology_cols = [col for col in topology_cols if col in available_cols]
    if not selected_topology_cols:
        return baseline_df

    select_cols = ["lat", "lon", "nodeb_id_cell_id", *selected_topology_cols]
    query = text(
        f"""
        SELECT {", ".join(f"`{col}`" for col in select_cols)}
        FROM lte_prediction_baseline_results
        WHERE project_id = :project_id
          AND job_id = :baseline_job_id
        """
    )
    with current_engine.connect() as conn:
        topology_df = pd.read_sql(
            query,
            conn,
            params={"project_id": int(project_id), "baseline_job_id": str(baseline_job_id)},
        )
    if topology_df.empty:
        return baseline_df

    out = baseline_df.copy()
    out["_topo_lat_key"] = pd.to_numeric(out["lat"], errors="coerce").round(6)
    out["_topo_lon_key"] = pd.to_numeric(out["lon"], errors="coerce").round(6)
    out["_topo_cell_key"] = out["Node_Cell_ID"].astype(str).str.strip()
    topo = topology_df.copy()
    topo["_topo_lat_key"] = pd.to_numeric(topo["lat"], errors="coerce").round(6)
    topo["_topo_lon_key"] = pd.to_numeric(topo["lon"], errors="coerce").round(6)
    topo["_topo_cell_key"] = topo["nodeb_id_cell_id"].astype(str).str.strip()
    topo = topo.drop(columns=["lat", "lon", "nodeb_id_cell_id"], errors="ignore")
    topo = topo.drop_duplicates(subset=["_topo_lat_key", "_topo_lon_key", "_topo_cell_key"], keep="last")
    out = out.merge(topo, on=["_topo_lat_key", "_topo_lon_key", "_topo_cell_key"], how="left")
    out = out.drop(columns=["_topo_lat_key", "_topo_lon_key", "_topo_cell_key"], errors="ignore")
    print(
        f"[TILT_TEST][BASELINE_TOPOLOGY] baseline_job_id={baseline_job_id} "
        f"columns_loaded={len(selected_topology_cols)} serving_pci_non_null="
        f"{int(out['serving_pci'].notna().sum()) if 'serving_pci' in out.columns else 0}"
    )
    return out


def _clean_id(value) -> str:
    return TILT_SRC._norm_cell_id(value)


def _cell_suffix(value) -> str:
    return TILT_SRC._cell_id_suffix(value)


def _site_match_mask(site_df: pd.DataFrame, recommendation_cell_id: str) -> pd.Series:
    rec_id = _clean_id(recommendation_cell_id)
    rec_suffix = _cell_suffix(rec_id)
    node_cell = site_df["Node_Cell_ID"].astype(str).map(_clean_id)
    mask = node_cell == rec_id
    if not mask.any() and "cell_id" in site_df.columns:
        cell_id = site_df["cell_id"].astype(str).map(_clean_id)
        mask = cell_id == rec_id
    if not mask.any() and rec_suffix and "cell_id" in site_df.columns:
        cell_suffix = site_df["cell_id"].astype(str).map(_cell_suffix)
        mask = cell_suffix == rec_suffix
    return mask


def _is_bad_sample(df: pd.DataFrame, rsrp_threshold: float, rsrq_threshold: float, sinr_threshold: float) -> pd.Series:
    rsrp = pd.to_numeric(df["pred_rsrp"], errors="coerce")
    rsrq = pd.to_numeric(df["pred_rsrq"], errors="coerce")
    sinr = pd.to_numeric(df["pred_sinr"], errors="coerce")
    return (rsrp < float(rsrp_threshold)) | (rsrq < float(rsrq_threshold)) | (sinr < float(sinr_threshold))


def _kpi_severity(series: pd.Series, threshold: float, cap: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    severity = (float(threshold) - values).clip(lower=0.0, upper=float(cap))
    return severity.fillna(0.0)


def _best_serving_snapshot_by_location(
    df: pd.DataFrame,
    allowed_cells: Sequence[str],
) -> pd.DataFrame:
    allowed = {str(c) for c in allowed_cells}
    work = df.loc[df["Node_Cell_ID"].astype(str).isin(allowed)].copy()
    if work.empty:
        return pd.DataFrame()

    work["lat"] = pd.to_numeric(work["lat"], errors="coerce").round(6)
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce").round(6)
    work["pred_rsrp"] = pd.to_numeric(work["pred_rsrp"], errors="coerce")
    work["pred_rsrq"] = pd.to_numeric(work["pred_rsrq"], errors="coerce")
    work["pred_sinr"] = pd.to_numeric(work["pred_sinr"], errors="coerce")
    work = work.dropna(subset=["lat", "lon"])
    if work.empty:
        return pd.DataFrame()

    # Compare candidates on the same location universe while allowing serving-cell switching.
    ranked = work.sort_values(
        ["lat", "lon", "pred_rsrp", "pred_sinr", "pred_rsrq", "Node_Cell_ID"],
        ascending=[True, True, False, False, False, True],
    )
    serving = ranked.drop_duplicates(subset=["lat", "lon"], keep="first").copy()
    return serving[["lat", "lon", "Node_Cell_ID", "pred_rsrp", "pred_rsrq", "pred_sinr"]]


def _expand_cells_at_evaluation_locations(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    seed_cells: Sequence[str],
) -> List[str]:
    selected = {str(cell).strip() for cell in seed_cells if str(cell).strip()}
    if not selected:
        return []

    location_keys = set()
    for df in [baseline_df, candidate_df]:
        if df.empty or "Node_Cell_ID" not in df.columns:
            continue
        focus = df.loc[df["Node_Cell_ID"].astype(str).isin(selected), ["lat", "lon"]].copy()
        if focus.empty:
            continue
        focus["lat_key"] = pd.to_numeric(focus["lat"], errors="coerce").round(6)
        focus["lon_key"] = pd.to_numeric(focus["lon"], errors="coerce").round(6)
        focus = focus.dropna(subset=["lat_key", "lon_key"])
        location_keys.update(zip(focus["lat_key"], focus["lon_key"]))

    if not location_keys:
        return sorted(selected)

    for df in [baseline_df, candidate_df]:
        if df.empty or "Node_Cell_ID" not in df.columns:
            continue
        work = df[["lat", "lon", "Node_Cell_ID"]].copy()
        work["lat_key"] = pd.to_numeric(work["lat"], errors="coerce").round(6)
        work["lon_key"] = pd.to_numeric(work["lon"], errors="coerce").round(6)
        work = work.dropna(subset=["lat_key", "lon_key"])
        loc_mask = pd.MultiIndex.from_frame(work[["lat_key", "lon_key"]]).isin(location_keys)
        selected.update(work.loc[loc_mask, "Node_Cell_ID"].astype(str).tolist())
    return sorted(selected)


def _score_candidate_vs_baseline(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    affected_cells: Sequence[str],
    config: TiltRecommendationTestConfig,
) -> Dict[str, float]:
    evaluation_cells = _expand_cells_at_evaluation_locations(baseline_df, candidate_df, affected_cells)
    baseline_snapshot = _best_serving_snapshot_by_location(baseline_df, evaluation_cells)
    candidate_snapshot = _best_serving_snapshot_by_location(candidate_df, evaluation_cells)
    if baseline_snapshot.empty or candidate_snapshot.empty:
        return {
            "baseline_bad_count": 0.0,
            "candidate_bad_count": 0.0,
            "recovered_bad_samples": 0.0,
            "new_bad_samples": 0.0,
            "rsrp_recovered_bad": 0.0,
            "rsrp_new_bad": 0.0,
            "rsrq_recovered_bad": 0.0,
            "rsrq_new_bad": 0.0,
            "sinr_recovered_bad": 0.0,
            "sinr_new_bad": 0.0,
            "rsrp_severity_reduction": 0.0,
            "rsrq_severity_reduction": 0.0,
            "sinr_severity_reduction": 0.0,
            "total_severity_reduction": 0.0,
            "evaluation_sample_count": 0.0,
            "rsrp_severity_reduction_per_sample": 0.0,
            "rsrq_severity_reduction_per_sample": 0.0,
            "sinr_severity_reduction_per_sample": 0.0,
            "total_severity_reduction_per_sample": 0.0,
            "net_bad_reduction_share": 0.0,
            "recovered_bad_share": 0.0,
            "new_bad_share": 0.0,
            "rsrp_recovered_bad_share": 0.0,
            "rsrp_new_bad_share": 0.0,
            "rsrq_recovered_bad_share": 0.0,
            "rsrq_new_bad_share": 0.0,
            "sinr_recovered_bad_share": 0.0,
            "sinr_new_bad_share": 0.0,
            "good_area_loss_pct": 0.0,
            "mean_rsrp_delta": 0.0,
            "mean_rsrq_delta": 0.0,
            "mean_sinr_delta": 0.0,
            "score": -9999.0,
            "constraints_passed": 0.0,
        }

    merged = baseline_snapshot.merge(
        candidate_snapshot[["lat", "lon", "Node_Cell_ID", "pred_rsrp", "pred_rsrq", "pred_sinr"]],
        on=["lat", "lon"],
        how="left",
        suffixes=("_base", "_cand"),
    )
    if merged.empty:
        return {
            "baseline_bad_count": float(len(baseline_snapshot)),
            "candidate_bad_count": float(len(candidate_snapshot)),
            "recovered_bad_samples": 0.0,
            "new_bad_samples": 0.0,
            "rsrp_recovered_bad": 0.0,
            "rsrp_new_bad": 0.0,
            "rsrq_recovered_bad": 0.0,
            "rsrq_new_bad": 0.0,
            "sinr_recovered_bad": 0.0,
            "sinr_new_bad": 0.0,
            "rsrp_severity_reduction": 0.0,
            "rsrq_severity_reduction": 0.0,
            "sinr_severity_reduction": 0.0,
            "total_severity_reduction": 0.0,
            "evaluation_sample_count": 0.0,
            "rsrp_severity_reduction_per_sample": 0.0,
            "rsrq_severity_reduction_per_sample": 0.0,
            "sinr_severity_reduction_per_sample": 0.0,
            "total_severity_reduction_per_sample": 0.0,
            "net_bad_reduction_share": 0.0,
            "recovered_bad_share": 0.0,
            "new_bad_share": 0.0,
            "rsrp_recovered_bad_share": 0.0,
            "rsrp_new_bad_share": 0.0,
            "rsrq_recovered_bad_share": 0.0,
            "rsrq_new_bad_share": 0.0,
            "sinr_recovered_bad_share": 0.0,
            "sinr_new_bad_share": 0.0,
            "good_area_loss_pct": 100.0,
            "mean_rsrp_delta": 0.0,
            "mean_rsrq_delta": 0.0,
            "mean_sinr_delta": 0.0,
            "score": -9999.0,
            "constraints_passed": 0.0,
        }

    for col in ["pred_rsrp_cand", "pred_rsrq_cand", "pred_sinr_cand"]:
        base_col = col.replace("_cand", "_base")
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(pd.to_numeric(merged[base_col], errors="coerce"))

    rsrp_base = pd.to_numeric(merged["pred_rsrp_base"], errors="coerce")
    rsrq_base = pd.to_numeric(merged["pred_rsrq_base"], errors="coerce")
    sinr_base = pd.to_numeric(merged["pred_sinr_base"], errors="coerce")
    rsrp_cand = pd.to_numeric(merged["pred_rsrp_cand"], errors="coerce")
    rsrq_cand = pd.to_numeric(merged["pred_rsrq_cand"], errors="coerce")
    sinr_cand = pd.to_numeric(merged["pred_sinr_cand"], errors="coerce")

    base_rsrp_bad = rsrp_base < float(config.rsrp_threshold)
    cand_rsrp_bad = rsrp_cand < float(config.rsrp_threshold)
    base_rsrq_bad = rsrq_base < float(config.rsrq_threshold)
    cand_rsrq_bad = rsrq_cand < float(config.rsrq_threshold)
    base_sinr_bad = sinr_base < float(config.sinr_threshold)
    cand_sinr_bad = sinr_cand < float(config.sinr_threshold)
    base_bad = base_rsrp_bad | base_rsrq_bad | base_sinr_bad
    cand_bad = cand_rsrp_bad | cand_rsrq_bad | cand_sinr_bad

    rsrp_recovered = int((base_rsrp_bad & ~cand_rsrp_bad).sum())
    rsrp_new_bad = int((~base_rsrp_bad & cand_rsrp_bad).sum())
    rsrq_recovered = int((base_rsrq_bad & ~cand_rsrq_bad).sum())
    rsrq_new_bad = int((~base_rsrq_bad & cand_rsrq_bad).sum())
    sinr_recovered = int((base_sinr_bad & ~cand_sinr_bad).sum())
    sinr_new_bad = int((~base_sinr_bad & cand_sinr_bad).sum())

    rsrp_base_severity = _kpi_severity(rsrp_base, config.rsrp_threshold, cap=25.0)
    rsrp_cand_severity = _kpi_severity(rsrp_cand, config.rsrp_threshold, cap=25.0)
    rsrq_base_severity = _kpi_severity(rsrq_base, config.rsrq_threshold, cap=8.0)
    rsrq_cand_severity = _kpi_severity(rsrq_cand, config.rsrq_threshold, cap=8.0)
    sinr_base_severity = _kpi_severity(sinr_base, config.sinr_threshold, cap=15.0)
    sinr_cand_severity = _kpi_severity(sinr_cand, config.sinr_threshold, cap=15.0)
    rsrp_severity_reduction = float((rsrp_base_severity - rsrp_cand_severity).sum())
    rsrq_severity_reduction = float((rsrq_base_severity - rsrq_cand_severity).sum())
    sinr_severity_reduction = float((sinr_base_severity - sinr_cand_severity).sum())
    total_severity_reduction = (
        rsrp_severity_reduction
        + rsrq_severity_reduction
        + sinr_severity_reduction
    )
    evaluation_sample_count = max(float(len(merged)), 1.0)
    rsrp_severity_reduction_per_sample = rsrp_severity_reduction / evaluation_sample_count
    rsrq_severity_reduction_per_sample = rsrq_severity_reduction / evaluation_sample_count
    sinr_severity_reduction_per_sample = sinr_severity_reduction / evaluation_sample_count
    total_severity_reduction_per_sample = total_severity_reduction / evaluation_sample_count

    recovered_bad = int((base_bad & ~cand_bad).sum())
    new_bad = int((~base_bad & cand_bad).sum())
    net_bad_reduction = int(base_bad.sum()) - int(cand_bad.sum())
    baseline_good = int((~base_bad).sum())
    good_area_loss_pct = (float(new_bad) / float(baseline_good) * 100.0) if baseline_good > 0 else 0.0
    net_bad_reduction_share = float(net_bad_reduction) / evaluation_sample_count
    recovered_bad_share = float(recovered_bad) / evaluation_sample_count
    new_bad_share = float(new_bad) / evaluation_sample_count
    rsrp_recovered_share = float(rsrp_recovered) / evaluation_sample_count
    rsrp_new_bad_share = float(rsrp_new_bad) / evaluation_sample_count
    rsrq_recovered_share = float(rsrq_recovered) / evaluation_sample_count
    rsrq_new_bad_share = float(rsrq_new_bad) / evaluation_sample_count
    sinr_recovered_share = float(sinr_recovered) / evaluation_sample_count
    sinr_new_bad_share = float(sinr_new_bad) / evaluation_sample_count

    mean_rsrp_delta = float((rsrp_cand - rsrp_base).mean())
    mean_rsrq_delta = float((rsrq_cand - rsrq_base).mean())
    mean_sinr_delta = float((sinr_cand - sinr_base).mean())

    hard_good_area_loss_pct = max(float(config.max_good_area_loss_pct), float(SEVERE_GOOD_AREA_LOSS_PCT))
    hard_sinr_drop_db = max(float(config.max_mean_sinr_drop_db), abs(float(SEVERE_MEAN_SINR_DROP_DB)))
    constraints_passed = (
        good_area_loss_pct <= hard_good_area_loss_pct
        and mean_sinr_delta >= -hard_sinr_drop_db
    )
    good_area_soft_penalty = good_area_loss_pct * GOOD_AREA_LOSS_WEIGHT
    score = (
        sinr_severity_reduction_per_sample * SINR_SEVERITY_REDUCTION_WEIGHT
        + rsrq_severity_reduction_per_sample * RSRQ_SEVERITY_REDUCTION_WEIGHT
        + rsrp_severity_reduction_per_sample * RSRP_SEVERITY_REDUCTION_WEIGHT
        + total_severity_reduction_per_sample * TOTAL_SEVERITY_REDUCTION_WEIGHT
        + net_bad_reduction_share * NET_BAD_REDUCTION_WEIGHT
        + sinr_recovered_share * SINR_RECOVERY_WEIGHT
        + rsrq_recovered_share * RSRQ_RECOVERY_WEIGHT
        + rsrp_recovered_share * RSRP_RECOVERY_WEIGHT
        - sinr_new_bad_share * SINR_NEW_BAD_WEIGHT
        - rsrq_new_bad_share * RSRQ_NEW_BAD_WEIGHT
        - rsrp_new_bad_share * RSRP_NEW_BAD_WEIGHT
        - good_area_soft_penalty
    )
    if not constraints_passed:
        score -= SEVERE_CONSTRAINT_PENALTY

    return {
        "baseline_bad_count": float(base_bad.sum()),
        "candidate_bad_count": float(cand_bad.sum()),
        "recovered_bad_samples": float(recovered_bad),
        "new_bad_samples": float(new_bad),
        "rsrp_recovered_bad": float(rsrp_recovered),
        "rsrp_new_bad": float(rsrp_new_bad),
        "rsrq_recovered_bad": float(rsrq_recovered),
        "rsrq_new_bad": float(rsrq_new_bad),
        "sinr_recovered_bad": float(sinr_recovered),
        "sinr_new_bad": float(sinr_new_bad),
        "rsrp_severity_reduction": float(rsrp_severity_reduction),
        "rsrq_severity_reduction": float(rsrq_severity_reduction),
        "sinr_severity_reduction": float(sinr_severity_reduction),
        "total_severity_reduction": float(total_severity_reduction),
        "evaluation_sample_count": float(evaluation_sample_count),
        "rsrp_severity_reduction_per_sample": float(rsrp_severity_reduction_per_sample),
        "rsrq_severity_reduction_per_sample": float(rsrq_severity_reduction_per_sample),
        "sinr_severity_reduction_per_sample": float(sinr_severity_reduction_per_sample),
        "total_severity_reduction_per_sample": float(total_severity_reduction_per_sample),
        "net_bad_reduction": float(net_bad_reduction),
        "net_bad_reduction_share": float(net_bad_reduction_share),
        "recovered_bad_share": float(recovered_bad_share),
        "new_bad_share": float(new_bad_share),
        "rsrp_recovered_bad_share": float(rsrp_recovered_share),
        "rsrp_new_bad_share": float(rsrp_new_bad_share),
        "rsrq_recovered_bad_share": float(rsrq_recovered_share),
        "rsrq_new_bad_share": float(rsrq_new_bad_share),
        "sinr_recovered_bad_share": float(sinr_recovered_share),
        "sinr_new_bad_share": float(sinr_new_bad_share),
        "good_area_loss_pct": float(good_area_loss_pct),
        "mean_rsrp_delta": float(mean_rsrp_delta),
        "mean_rsrq_delta": float(mean_rsrq_delta),
        "mean_sinr_delta": float(mean_sinr_delta),
        "score": float(score),
        "constraints_passed": float(1 if constraints_passed else 0),
    }


def _candidate_cache_key(site_id: str, updates: Sequence[Dict[str, object]]) -> tuple:
    if not updates:
        return ((str(site_id), "__hold__", "__hold__", 0.0),)
    normalized = []
    for update in updates:
        normalized.append(
            (
                str(site_id),
                str(update.get("cell_id", "")).strip(),
                str(update.get("parameter", "")).strip(),
                round(float(update.get("target_value", 0.0)), 4),
            )
        )
    return tuple(sorted(normalized))


def _sample_distance_m(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    lat1_rad = np.radians(float(lat1))
    lon1_rad = np.radians(float(lon1))
    lat2_rad = np.radians(float(lat2))
    lon2_rad = np.radians(float(lon2))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return float(6371000.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def _baseline_topology_context(
    baseline_df: pd.DataFrame,
    target_cells: Sequence[str],
) -> Dict[str, object]:
    target_set = {str(c) for c in target_cells}
    focus = baseline_df.loc[baseline_df["Node_Cell_ID"].astype(str).isin(target_set)].copy()
    if focus.empty:
        return {
            "topology_root_cause": "unknown",
            "interference_limited": False,
            "coverage_limited": False,
            "weak_sinr_share": 0.0,
            "low_rsrp_share": 0.0,
            "same_earfcn_interferer_share": 0.0,
            "dominant_interferer_share": 0.0,
            "mean_interference_gap_db": np.nan,
            "p25_interference_gap_db": np.nan,
        }

    pred_rsrp = pd.to_numeric(focus.get("pred_rsrp"), errors="coerce")
    pred_sinr = pd.to_numeric(focus.get("pred_sinr"), errors="coerce")
    proxy_sinr = pd.to_numeric(focus.get("sinr_proxy_db"), errors="coerce") if "sinr_proxy_db" in focus.columns else pred_sinr
    signal_sinr = proxy_sinr.combine_first(pred_sinr)
    low_rsrp_share = float((pred_rsrp < -105.0).mean()) if len(pred_rsrp) else 0.0
    weak_sinr_share = float((signal_sinr < 0.0).mean()) if len(signal_sinr) else 0.0

    gap = pd.to_numeric(focus.get("interference_gap_db"), errors="coerce") if "interference_gap_db" in focus.columns else pd.Series(dtype=float)
    gap_non_null = gap.dropna()
    mean_gap = float(gap_non_null.mean()) if not gap_non_null.empty else np.nan
    p25_gap = float(gap_non_null.quantile(0.25)) if not gap_non_null.empty else np.nan

    same_count = (
        pd.to_numeric(focus.get("same_earfcn_interferer_count"), errors="coerce").fillna(0.0)
        if "same_earfcn_interferer_count" in focus.columns else pd.Series(0.0, index=focus.index)
    )
    dominant_count = (
        pd.to_numeric(focus.get("dominant_interferer_count"), errors="coerce").fillna(0.0)
        if "dominant_interferer_count" in focus.columns else pd.Series(0.0, index=focus.index)
    )
    best_interferer = (
        focus.get("best_interferer_cell_id").astype(str).str.strip()
        if "best_interferer_cell_id" in focus.columns else pd.Series("", index=focus.index)
    )
    same_earfcn_share = float((same_count > 0).mean()) if len(same_count) else 0.0
    dominant_share = float((dominant_count > 0).mean()) if len(dominant_count) else 0.0
    interferer_present_share = float((best_interferer != "").mean()) if len(best_interferer) else 0.0

    gap_is_tight = (
        (pd.notna(p25_gap) and p25_gap <= INTERFERENCE_GAP_STRONG_DB)
        or (pd.notna(mean_gap) and mean_gap <= INTERFERENCE_GAP_WEAK_DB)
    )
    topology_interference = (
        gap_is_tight
        or same_earfcn_share >= SAME_EARFCN_INTERFERER_SHARE_GATE
        or dominant_share >= DOMINANT_INTERFERER_SHARE_GATE
    )
    interference_limited = bool(
        topology_interference
        and (weak_sinr_share >= QUALITY_LOW_SINR_SHARE_GATE or interferer_present_share >= 0.50)
    )
    coverage_limited = bool(low_rsrp_share >= COVERAGE_LOW_RSRP_SHARE_GATE and not interference_limited)
    if interference_limited:
        root_cause = "interference_topology"
    elif coverage_limited:
        root_cause = "coverage_topology"
    elif weak_sinr_share >= QUALITY_LOW_SINR_SHARE_GATE:
        root_cause = "quality_topology"
    else:
        root_cause = "mixed_topology"

    return {
        "topology_root_cause": root_cause,
        "interference_limited": interference_limited,
        "coverage_limited": coverage_limited,
        "weak_sinr_share": weak_sinr_share,
        "low_rsrp_share": low_rsrp_share,
        "same_earfcn_interferer_share": same_earfcn_share,
        "dominant_interferer_share": dominant_share,
        "interferer_present_share": interferer_present_share,
        "mean_interference_gap_db": mean_gap,
        "p25_interference_gap_db": p25_gap,
    }


def _expand_evaluation_cells_from_topology(
    baseline_df: pd.DataFrame,
    seed_cells: Sequence[str],
) -> List[str]:
    selected = {str(cell).strip() for cell in seed_cells if str(cell).strip()}
    if not selected or baseline_df.empty:
        return sorted(selected)

    for _ in range(2):
        focus = baseline_df.loc[baseline_df["Node_Cell_ID"].astype(str).isin(selected)].copy()
        if focus.empty:
            break
        before_count = len(selected)
        for col in ["best_interferer_cell_id", "neighbor_1_cell_id", "neighbor_2_cell_id"]:
            if col not in focus.columns:
                continue
            values = focus[col].dropna().astype(str).str.strip()
            values = values.loc[~values.isin(["", "None", "nan", "NaN"])]
            selected.update(values.tolist())
        if len(selected) == before_count:
            break
    return sorted(selected)


def _fast_geometry_score(
    baseline_df: pd.DataFrame,
    site_df: pd.DataFrame,
    modified_site_df: pd.DataFrame,
    target_site_id: str,
    target_updates: Sequence[Dict[str, object]],
) -> float:
    site_map = _build_cell_site_map(site_df)
    target_cells = site_map.loc[site_map["Site ID"].astype(str) == str(target_site_id), "Cell ID"].astype(str).tolist()
    if not target_cells:
        return -999.0

    topology_context = _baseline_topology_context(baseline_df, target_cells)
    focus_df = baseline_df.loc[baseline_df["Node_Cell_ID"].astype(str).isin(target_cells)].copy()
    if focus_df.empty:
        return -999.0

    focus_df["lat_eval"] = pd.to_numeric(focus_df["lat"], errors="coerce")
    focus_df["lon_eval"] = pd.to_numeric(focus_df["lon"], errors="coerce")
    focus_df["pred_sinr_eval"] = pd.to_numeric(focus_df["pred_sinr"], errors="coerce")
    focus_df["pred_rsrp_eval"] = pd.to_numeric(focus_df["pred_rsrp"], errors="coerce")
    focus_df = focus_df.dropna(subset=["lat_eval", "lon_eval"])
    if focus_df.empty:
        return -999.0

    current_cells = _build_cell_site_map(site_df).set_index("Cell ID")
    modified_cells = _build_cell_site_map(modified_site_df).set_index("Cell ID")
    all_sites = _build_cell_site_map(site_df)[["Site ID", "lat", "lon"]].drop_duplicates("Site ID").copy()
    all_sites["lat"] = pd.to_numeric(all_sites["lat"], errors="coerce")
    all_sites["lon"] = pd.to_numeric(all_sites["lon"], errors="coerce")

    serving_distances: List[float] = []
    overlap_hits = 0
    azimuth_improvements = 0
    azimuth_regressions = 0
    bad_nlos_like = 0
    checked_samples = 0

    for _, sample in focus_df.iterrows():
        cell_id = str(sample["Node_Cell_ID"])
        if cell_id not in current_cells.index or cell_id not in modified_cells.index:
            continue
        current_cell = current_cells.loc[cell_id]
        modified_cell = modified_cells.loc[cell_id]
        if any(pd.isna(v) for v in [current_cell.get("lat"), current_cell.get("lon")]):
            continue
        sample_lat = float(sample["lat_eval"])
        sample_lon = float(sample["lon_eval"])
        current_dist = _sample_distance_m(sample_lat, sample_lon, current_cell.get("lat"), current_cell.get("lon"))
        if pd.notna(current_dist):
            serving_distances.append(current_dist)

        other_sites = all_sites.loc[all_sites["Site ID"].astype(str) != str(target_site_id)]
        if not other_sites.empty and pd.notna(current_dist):
            other_distances = other_sites.apply(
                lambda r: _sample_distance_m(sample_lat, sample_lon, r["lat"], r["lon"]),
                axis=1,
            )
            nearest_other = pd.to_numeric(other_distances, errors="coerce").min()
            if pd.notna(nearest_other) and nearest_other <= current_dist * 1.15:
                overlap_hits += 1

        bearing = TILT_SRC._bearing_deg(
            float(current_cell.get("lat")),
            float(current_cell.get("lon")),
            sample_lat,
            sample_lon,
        )
        current_mismatch = TILT_SRC._angular_diff(bearing, current_cell.get("azimuth"))
        modified_mismatch = TILT_SRC._angular_diff(bearing, modified_cell.get("azimuth"))
        if pd.notna(current_mismatch) and pd.notna(modified_mismatch):
            if modified_mismatch + 3.0 < current_mismatch:
                azimuth_improvements += 1
            elif modified_mismatch > current_mismatch + 3.0:
                azimuth_regressions += 1

        if pd.notna(sample.get("pred_sinr_eval")) and float(sample["pred_sinr_eval"]) < -1.0:
            bad_nlos_like += 1
        checked_samples += 1

    if checked_samples == 0:
        return -999.0

    mean_serving_distance = float(np.nanmean(serving_distances)) if serving_distances else np.nan
    overlap_share = float(overlap_hits) / float(checked_samples)
    weak_dominance = float((focus_df["pred_sinr_eval"] < 0.0).mean()) if "pred_sinr_eval" in focus_df.columns else 0.0
    low_rsrp_share = float((focus_df["pred_rsrp_eval"] < -105.0).mean()) if "pred_rsrp_eval" in focus_df.columns else 0.0
    nlos_like_share = float(bad_nlos_like) / float(checked_samples)
    interference_limited = bool(topology_context.get("interference_limited", False))
    coverage_limited = bool(topology_context.get("coverage_limited", False))
    same_earfcn_share = float(topology_context.get("same_earfcn_interferer_share", 0.0))
    dominant_interferer_share = float(topology_context.get("dominant_interferer_share", 0.0))

    score = 0.0
    for update in target_updates:
        parameter = str(update.get("parameter", "")).strip()
        cell_id = str(update.get("cell_id", "")).strip()
        if cell_id not in current_cells.index or cell_id not in modified_cells.index:
            continue
        current_row = current_cells.loc[cell_id]
        modified_row = modified_cells.loc[cell_id]

        if parameter == "ETilt":
            tilt_delta = float(modified_row.get("electrical_tilt", 0.0)) - float(current_row.get("electrical_tilt", 0.0))
            if tilt_delta < 0 and pd.notna(mean_serving_distance) and mean_serving_distance >= MEDIUM_EDGE_MEAN_SERVING_DISTANCE_M:
                score += 22.0
            if tilt_delta > 0 and overlap_share >= 0.35:
                score += 24.0
            if tilt_delta < 0 and overlap_share >= 0.45:
                score -= 18.0
            if tilt_delta > 0 and pd.notna(mean_serving_distance) and mean_serving_distance >= FAR_EDGE_MEAN_SERVING_DISTANCE_M:
                score -= 16.0
            if interference_limited and tilt_delta > 0:
                score += 20.0
            if interference_limited and tilt_delta < 0:
                score -= 24.0
            if coverage_limited and tilt_delta < 0:
                score += 18.0
            if coverage_limited and tilt_delta > 0:
                score -= 18.0
        elif parameter == "TX Power":
            tx_delta = float(modified_row.get("tx_power", 0.0)) - float(current_row.get("tx_power", 0.0))
            if tx_delta > 0 and low_rsrp_share >= 0.40 and overlap_share < 0.35:
                score += 18.0
            if tx_delta < 0 and overlap_share >= 0.35:
                score += 16.0
            if tx_delta > 0 and overlap_share >= 0.45:
                score -= 24.0
            if tx_delta < 0 and pd.notna(mean_serving_distance) and mean_serving_distance >= FAR_EDGE_MEAN_SERVING_DISTANCE_M:
                score -= 14.0
            if interference_limited and tx_delta < 0:
                score += 22.0
            if interference_limited and tx_delta > 0:
                score -= 28.0
            if coverage_limited and tx_delta > 0:
                score += 16.0
            if coverage_limited and tx_delta < 0:
                score -= 16.0
        elif parameter == "Azimuth":
            score += azimuth_improvements * 3.0
            score -= azimuth_regressions * 3.5
            if interference_limited:
                score += 6.0

    score -= weak_dominance * 10.0
    score -= nlos_like_share * 12.0
    score -= same_earfcn_share * 5.0
    score -= dominant_interferer_share * 8.0
    return float(score)


def _site_candidate_rank_tuple(result: Dict[str, object]) -> tuple:
    return (
        float(result.get("net_bad_reduction", -99999.0)),
        float(result.get("recovered_bad_samples", -99999.0)),
        -float(result.get("new_bad_samples", 99999.0)),
        -float(result.get("good_area_loss_pct", 99999.0)),
        float(result.get("mean_sinr_delta", -99999.0)),
        float(result.get("tie_break", 0.0)),
    )


def _site_candidate_reject_reason(
    result: Dict[str, object],
    config: TiltRecommendationTestConfig,
) -> str:
    error = str(result.get("error", "")).strip()
    if error:
        return f"error:{error}"
    constraints_passed = bool(int(result.get("constraints_passed", 0)))
    net_bad_reduction = float(result.get("net_bad_reduction", 0.0))
    recovered_bad_samples = float(result.get("recovered_bad_samples", 0.0))
    mean_sinr_delta = float(result.get("mean_sinr_delta", 0.0))
    score = float(result.get("score", -99999.0))
    if not constraints_passed:
        return "failed_severe_constraints"
    if score > 0.0:
        return "selected"
    tradeoff_positive = (
        net_bad_reduction > float(config.min_score_gain)
        or recovered_bad_samples >= float(config.min_recovered_bad_samples)
        or mean_sinr_delta > 0.75
    )
    if not tradeoff_positive:
        return "no_positive_net_bad_reduction"
    if recovered_bad_samples < float(config.min_recovered_bad_samples) and net_bad_reduction <= 0 and mean_sinr_delta <= 0.75:
        return "recovered_bad_below_minimum"
    return "selected"


def _validate_recommendation_candidates_test_only(
    recommendations_all_df: pd.DataFrame,
    site_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    config: TiltRecommendationTestConfig,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if recommendations_all_df.empty or not bool(config.validate_candidates):
        return recommendations_all_df.copy(), pd.DataFrame()

    if optuna is None:
        raise ImportError(
            "Optuna is required for tilt candidate validation. Install it with `pip install optuna`."
        )

    baseline_job_id = _fetch_latest_baseline_job_id(config.project_id, config.region)
    validated_df = recommendations_all_df.copy()
    evaluation_rows: List[Dict[str, object]] = []
    rf_cache: Dict[tuple, Dict[str, object]] = {}

    candidate_rows = validated_df.loc[
        validated_df["Parameter"].astype(str).str.strip().isin(["ETilt", "Azimuth", "TX Power"])
    ].copy()
    candidate_rows = candidate_rows.loc[_changed_recommendation_mask(candidate_rows)].copy()
    if candidate_rows.empty:
        return validated_df, pd.DataFrame()

    candidate_rows["recommendation_index"] = candidate_rows.index
    site_map = _build_cell_site_map(site_df)[["Cell ID", "Site ID"]].copy()
    candidate_rows = candidate_rows.merge(site_map, on="Cell ID", how="left")
    candidate_rows["Site ID"] = candidate_rows["Site ID"].fillna(
        candidate_rows["Cell ID"].astype(str).str.split("_").str[0]
    )

    for site_id, site_group in candidate_rows.groupby("Site ID", dropna=False):
        site_id = str(site_id)
        site_group = site_group.copy()
        param_rows: List[Dict[str, object]] = []
        for idx, rec in site_group.iterrows():
            current_value = pd.to_numeric(pd.Series([rec["Current Value"]]), errors="coerce").iloc[0]
            recommended_value = pd.to_numeric(pd.Series([rec["Recommended Value"]]), errors="coerce").iloc[0]
            if pd.isna(current_value):
                continue
            param_rows.append(
                {
                    "index": int(rec["recommendation_index"]),
                    "cell_id": str(rec["Cell ID"]),
                    "parameter": str(rec["Parameter"]).strip(),
                    "current_value": float(current_value),
                    "recommended_value": float(recommended_value) if pd.notna(recommended_value) else np.nan,
                    "reason": str(rec["Reason"]),
                }
            )
        if not param_rows:
            continue

        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def _suggest_target_value(trial: "optuna.trial.Trial", row: Dict[str, object]) -> float:
            parameter = str(row["parameter"])
            current_value = float(row["current_value"])
            recommended_value = row["recommended_value"]
            cell_id = str(row["cell_id"])
            if parameter == "ETilt":
                low = max(MIN_SAFE_ETILT_DEG, current_value - MAX_ETILT_DECREASE_PER_RUN_DEG)
                high = min(MAX_SAFE_ETILT_DEG, current_value + MAX_ETILT_INCREASE_PER_RUN_DEG)
                if pd.notna(recommended_value):
                    low = max(MIN_SAFE_ETILT_DEG, min(low, float(recommended_value)))
                    high = min(MAX_SAFE_ETILT_DEG, max(high, float(recommended_value)))
                return float(trial.suggest_float(f"{cell_id}__etilt", low, high))
            if parameter == "Azimuth":
                return float(
                    trial.suggest_float(
                        f"{cell_id}__azimuth_delta",
                        -MAX_AZIMUTH_STEP_DEG,
                        MAX_AZIMUTH_STEP_DEG,
                    )
                    + current_value
                )
            if parameter == "TX Power":
                low = current_value - 1.0
                high = current_value + 1.0
                if pd.notna(recommended_value):
                    low = min(low, float(recommended_value))
                    high = max(high, float(recommended_value))
                return float(trial.suggest_float(f"{cell_id}__tx_power", low, high))
            return float(current_value)

        def objective(trial: "optuna.trial.Trial") -> float:
            updates: List[Dict[str, object]] = []
            applied_values: Dict[str, float] = {}
            for row in param_rows:
                parameter = str(row["parameter"])
                current_value = float(row["current_value"])
                target_value = _suggest_target_value(trial, row)
                if parameter == "ETilt":
                    target_value = float(_bounded_etilt_target(current_value, target_value))
                elif parameter == "Azimuth":
                    signed_delta = _signed_azimuth_delta(target_value, current_value)
                    signed_delta = float(np.clip(signed_delta, -MAX_AZIMUTH_STEP_DEG, MAX_AZIMUTH_STEP_DEG))
                    target_value = float(TILT_SRC._normalize_azimuth(current_value + signed_delta))
                elif parameter == "TX Power":
                    target_value = float(round(target_value, 2))
                applied_values[f"{row['cell_id']}::{parameter}"] = float(target_value)
                if not np.isclose(float(target_value), current_value, equal_nan=True):
                    updates.append(
                        {
                            "cell_id": row["cell_id"],
                            "parameter": parameter,
                            "target_value": float(target_value),
                        }
                    )

            modified_site_df = (
                _apply_multiple_parameter_targets(site_df, updates)
                if updates else opt_ml._normalize_site_df(site_df, log_stage="TILT_TEST_OPTUNA_HOLD")
            )
            cheap_score = _fast_geometry_score(
                baseline_df=baseline_df,
                site_df=site_df,
                modified_site_df=modified_site_df,
                target_site_id=site_id,
                target_updates=updates,
            )
            trial.set_user_attr("cheap_score", float(cheap_score))
            trial.set_user_attr("updates", list(updates))
            trial.set_user_attr("applied_values", dict(applied_values))
            if ENABLE_GEOMETRY_PREFILTER_PRUNING and cheap_score < OPTUNA_PREFILTER_SCORE:
                raise optuna.TrialPruned()

            cache_key = _candidate_cache_key(site_id, updates)
            cached = rf_cache.get(cache_key)
            if cached is None:
                if updates:
                    affected_cells, affected_sites, changed_rows = opt_ml._compute_affected_cells(
                        modified_site_df,
                        float(config.impact_radius_m),
                        int(config.neighbor_site_count),
                    )
                    calibration_cells = (
                        sorted(changed_rows["Node_Cell_ID"].astype(str).unique().tolist())
                        if not changed_rows.empty else []
                    )
                    if not calibration_cells:
                        raise ValueError("No changed cells found for candidate")
                    k1k2_map = opt_ml.compute_k1k2_for_cells(baseline_df, modified_site_df, calibration_cells)
                    if not k1k2_map:
                        raise ValueError("No calibrated cells found for candidate")
                    optimized_df = opt_ml.run_prediction_only_optimized(
                        modified_site_df,
                        k1k2_map,
                        {
                            "project_id": int(config.project_id),
                            "region": config.region,
                            "radius": float(config.radius_m),
                            "grid_resolution": float(config.grid_resolution_m),
                            "n_workers": int(config.workers),
                            "impact_radius_m": float(config.impact_radius_m),
                            "neighbor_site_count": int(config.neighbor_site_count),
                            "max_interference_sites": int(config.max_interference_sites),
                            "baseline_job_id": baseline_job_id,
                        },
                    )
                    merged_df = opt_ml.replace_cells(baseline_df, optimized_df)
                    metrics = _score_candidate_vs_baseline(
                        baseline_df,
                        merged_df,
                        affected_cells,
                        config,
                    )
                else:
                    affected_cells = site_group["Cell ID"].astype(str).tolist()
                    affected_sites = [site_id]
                    calibration_cells = []
                    metrics = _score_candidate_vs_baseline(baseline_df, baseline_df, affected_cells, config)
                cached = {
                    "affected_cells": len(affected_cells),
                    "affected_sites": len(affected_sites),
                    "changed_cells": len(calibration_cells),
                    **metrics,
                }
                rf_cache[cache_key] = cached

            if float(cached["good_area_loss_pct"]) > OPTUNA_PRUNE_GOOD_AREA_LOSS_PCT:
                raise optuna.TrialPruned()
            if float(cached["mean_sinr_delta"]) < OPTUNA_PRUNE_MEAN_SINR_DELTA_DB:
                raise optuna.TrialPruned()

            evaluation_rows.append(
                {
                    "site_id": site_id,
                    "cell_id": ",".join(site_group["Cell ID"].astype(str).tolist()),
                    "parameter": "SITE_OPTUNA",
                    "candidate_name": f"trial_{trial.number}",
                    "target_value": json.dumps(applied_values, sort_keys=True),
                    "affected_cells": int(cached["affected_cells"]),
                    "affected_sites": int(cached["affected_sites"]),
                    "changed_cells": int(cached["changed_cells"]),
                    "cheap_score": float(cheap_score),
                    **cached,
                    "error": "",
                }
            )
            return float(cached["score"])

        study.optimize(objective, n_trials=OPTUNA_TRIAL_COUNT, n_jobs=1, show_progress_bar=False)

        completed_trials = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            continue

        best_trial = max(
            completed_trials,
            key=lambda trial: float(trial.value if trial.value is not None else -99999.0),
        )
        best_updates = list(best_trial.user_attrs.get("updates", []))
        best_values = dict(best_trial.user_attrs.get("applied_values", {}))
        best_result = rf_cache.get(_candidate_cache_key(site_id, best_updates))
        if best_result is None:
            continue

        constraints_passed = bool(int(best_result.get("constraints_passed", 0)))
        best_score = float(best_result.get("score", -99999.0))
        positive_reduction = float(best_result.get("net_bad_reduction", 0.0)) > float(config.min_score_gain)
        enough_recovery = float(best_result.get("recovered_bad_samples", 0.0)) >= float(config.min_recovered_bad_samples)

        for row in param_rows:
            idx = int(row["index"])
            parameter = str(row["parameter"])
            current_value = float(row["current_value"])
            best_value = float(best_values.get(f"{row['cell_id']}::{parameter}", current_value))
            keep_current = np.isclose(best_value, current_value, equal_nan=True)
            if constraints_passed and positive_reduction and enough_recovery and not keep_current:
                validated_df.at[idx, "Recommended Value"] = best_value
                validated_df.at[idx, "Recommendation Status"] = "action_change_validated"
                validated_df.at[idx, "Reason"] = (
                    f"{row['reason']} Optuna site validation selected {best_value:.2f} for {parameter} on site {site_id} "
                    f"with score={best_score:.2f}, recovered_bad={best_result['recovered_bad_samples']:.0f}, "
                    f"new_bad={best_result['new_bad_samples']:.0f}, good_area_loss_pct={best_result['good_area_loss_pct']:.2f}, "
                    f"mean_sinr_delta={best_result['mean_sinr_delta']:.2f}."
                )
            else:
                validated_df.at[idx, "Recommended Value"] = current_value
                validated_df.at[idx, "Recommendation Status"] = "no_safe_change_available"
                validated_df.at[idx, "Reason"] = (
                    f"{row['reason']} Optuna site validation rejected unsafe or weak site candidates for site {site_id}. "
                    f"Best score={best_score:.2f}, constraints_passed={constraints_passed}, "
                    f"net_bad_reduction={best_result.get('net_bad_reduction', np.nan):.0f}, "
                    f"good_area_loss_pct={best_result.get('good_area_loss_pct', np.nan):.2f}, "
                    f"mean_sinr_delta={best_result.get('mean_sinr_delta', np.nan):.2f}."
                )

    evaluation_df = pd.DataFrame(evaluation_rows)
    return validated_df, evaluation_df


def _rank_cells_by_bad_contribution(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["Cell ID", "Technology", "Bad RSRP", "Bad RSRQ", "Bad SINR", "total_bad_samples"])
    ranked = summary_df.copy()
    ranked["Cell ID"] = ranked["Cell ID"].map(TILT_SRC._norm_cell_id)
    ranked["Bad RSRP"] = pd.to_numeric(ranked.get("Bad RSRP"), errors="coerce").fillna(0)
    ranked["Bad RSRQ"] = pd.to_numeric(ranked.get("Bad RSRQ"), errors="coerce").fillna(0)
    ranked["Bad SINR"] = pd.to_numeric(ranked.get("Bad SINR"), errors="coerce").fillna(0)
    ranked["total_bad_samples"] = ranked["Bad RSRP"] + ranked["Bad RSRQ"] + ranked["Bad SINR"]
    ranked = ranked.sort_values(["total_bad_samples", "Bad RSRP", "Cell ID"], ascending=[False, False, True]).reset_index(drop=True)
    return ranked


def _build_cell_site_map(antenna_df: pd.DataFrame) -> pd.DataFrame:
    ant_work = opt_ml._normalize_site_df(antenna_df, log_stage="TILT_TEST_SITE_MAP")
    ant_work["Cell ID"] = ant_work["Node_Cell_ID"].astype(str).map(TILT_SRC._norm_cell_id)
    ant_work["Site ID"] = ant_work["dashboard_site_id"].astype(str).str.strip()
    ant_work["Sector Suffix"] = ant_work["Cell ID"].map(TILT_SRC._cell_id_suffix)
    return ant_work.drop_duplicates(subset=["Cell ID"], keep="last")


def _rank_sites_by_bad_contribution(summary_df: pd.DataFrame, antenna_df: pd.DataFrame) -> pd.DataFrame:
    ranked_cells = _rank_cells_by_bad_contribution(summary_df)
    if ranked_cells.empty:
        return pd.DataFrame()
    site_map = _build_cell_site_map(antenna_df)[["Cell ID", "Site ID"]].copy()
    work = ranked_cells.merge(site_map, on="Cell ID", how="left")
    work["Site ID"] = work["Site ID"].fillna(work["Cell ID"].astype(str).str.split("_").str[0])
    site_ranked = (
        work.groupby("Site ID", dropna=False)
        .agg(
            total_bad_samples=("total_bad_samples", "sum"),
            bad_rsrp=("Bad RSRP", "sum"),
            bad_rsrq=("Bad RSRQ", "sum"),
            bad_sinr=("Bad SINR", "sum"),
            sector_count=("Cell ID", "nunique"),
        )
        .reset_index()
        .sort_values(["total_bad_samples", "bad_sinr", "bad_rsrq", "bad_rsrp", "Site ID"], ascending=[False, False, False, False, True])
        .reset_index(drop=True)
    )
    return site_ranked


def _apply_multiple_parameter_targets(
    site_df: pd.DataFrame,
    target_updates: Sequence[Dict[str, object]],
) -> pd.DataFrame:
    modified = opt_ml._normalize_site_df(site_df, log_stage="TILT_TEST_CLUSTER_CANDIDATE_INPUT")
    for col in ["lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "tx_power", "antenna_height"]:
        modified[f"orig_{col}"] = pd.to_numeric(modified[col], errors="coerce")

    param_map = {
        "ETilt": "electrical_tilt",
        "Azimuth": "azimuth",
        "TX Power": "tx_power",
    }
    applied_mask = pd.Series(False, index=modified.index)
    for update in target_updates:
        target_col = param_map.get(str(update.get("parameter")))
        if not target_col:
            continue
        cell_id = str(update.get("cell_id", ""))
        mask = _site_match_mask(modified, cell_id)
        if not mask.any():
            raise ValueError(f"Could not match site rows for cell_id={cell_id}")
        modified.loc[mask, target_col] = float(update["target_value"])
        applied_mask = applied_mask | mask
    modified["optimization_applied"] = applied_mask.astype(bool)
    return modified


def _mark_site_cluster_for_scope(
    site_df: pd.DataFrame,
    site_cell_ids: Sequence[str],
) -> pd.DataFrame:
    modified = opt_ml._normalize_site_df(site_df, log_stage="TILT_TEST_CLUSTER_SCOPE_INPUT")
    for col in ["lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "tx_power", "antenna_height"]:
        modified[f"orig_{col}"] = pd.to_numeric(modified[col], errors="coerce")
    applied_mask = pd.Series(False, index=modified.index)
    for cell_id in site_cell_ids:
        mask = _site_match_mask(modified, str(cell_id))
        applied_mask = applied_mask | mask
    if not applied_mask.any():
        raise ValueError(f"Could not mark any site rows for fixed scope: site_cell_ids={list(site_cell_ids)}")
    # The engine detects changes by comparing orig_* columns to current values.
    # For fixed-scope discovery we need changed rows without altering the actual RF inputs.
    # Use a clearly non-close origin delta so np.isclose does not collapse it away.
    modified.loc[applied_mask, "orig_azimuth"] = (
        pd.to_numeric(modified.loc[applied_mask, "azimuth"], errors="coerce") + 0.123
    )
    modified["optimization_applied"] = applied_mask.astype(bool)
    return modified


def _build_coordinated_action_cluster(
    site_id: str,
    site_cell_ids: Sequence[str],
    antenna_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    ranked_cells: pd.DataFrame,
) -> pd.DataFrame:
    ant_use = _build_cell_site_map(antenna_df)
    primary_cells = {str(cell).strip() for cell in site_cell_ids if str(cell).strip()}
    selected_sites = [str(site_id)]
    related_cell_counts: Dict[str, int] = {}

    focus = baseline_df.loc[baseline_df["Node_Cell_ID"].astype(str).isin(primary_cells)].copy()
    for col in ["best_interferer_cell_id", "neighbor_1_cell_id", "neighbor_2_cell_id"]:
        if col not in focus.columns:
            continue
        values = focus[col].dropna().astype(str).str.strip()
        values = values.loc[~values.isin(["", "None", "nan", "NaN"])]
        for value in values:
            related_cell_counts[value] = related_cell_counts.get(value, 0) + 1

    if related_cell_counts:
        related_df = pd.DataFrame(
            [{"Cell ID": cell_id, "topology_ref_count": count} for cell_id, count in related_cell_counts.items()]
        )
        related_df = related_df.merge(
            ant_use[["Cell ID", "Site ID"]],
            on="Cell ID",
            how="inner",
        )
        related_df = related_df.loc[related_df["Site ID"].astype(str) != str(site_id)].copy()
        if not related_df.empty:
            site_counts = (
                related_df.groupby("Site ID", dropna=False)["topology_ref_count"]
                .sum()
                .reset_index()
                .sort_values(["topology_ref_count", "Site ID"], ascending=[False, True])
            )
            selected_sites.extend(
                site_counts["Site ID"].astype(str).head(MAX_COORDINATED_NEIGHBOR_SITES).tolist()
            )

    selected_sites = list(dict.fromkeys(selected_sites[:MAX_COORDINATED_ACTION_SITES]))
    cluster = ant_use.loc[ant_use["Site ID"].astype(str).isin(selected_sites)].copy()
    if cluster.empty:
        return ant_use.loc[ant_use["Site ID"].astype(str) == str(site_id)].copy()

    bad_rank = ranked_cells.copy()
    bad_rank["Cell ID"] = bad_rank["Cell ID"].astype(str)
    bad_rank["bad_rank_score"] = pd.to_numeric(bad_rank.get("total_bad_samples"), errors="coerce").fillna(0.0)
    topology_rank = pd.DataFrame(
        [{"Cell ID": cell_id, "topology_ref_count": count} for cell_id, count in related_cell_counts.items()],
        columns=["Cell ID", "topology_ref_count"],
    )
    cluster = cluster.merge(bad_rank[["Cell ID", "bad_rank_score"]], on="Cell ID", how="left")
    cluster = cluster.merge(topology_rank, on="Cell ID", how="left")
    cluster["bad_rank_score"] = pd.to_numeric(cluster["bad_rank_score"], errors="coerce").fillna(0.0)
    cluster["topology_ref_count"] = pd.to_numeric(cluster["topology_ref_count"], errors="coerce").fillna(0.0)
    cluster["primary_site_flag"] = (cluster["Site ID"].astype(str) == str(site_id)).astype(int)
    cluster["action_priority"] = (
        cluster["primary_site_flag"] * 100000.0
        + cluster["bad_rank_score"] * 10.0
        + cluster["topology_ref_count"] * 100.0
    )
    return cluster.sort_values(
        ["primary_site_flag", "action_priority", "Cell ID"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _site_candidate_targets(
    site_cells_df: pd.DataFrame,
    site_bad_cells_df: pd.DataFrame,
    swap_dict: Dict[str, str],
    topology_context: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = [{"name": "hold", "updates": [], "tie_break": 0.0, "root_cause": "hold"}]
    topology_context = topology_context or {}
    topology_root_cause = str(topology_context.get("topology_root_cause", "search"))
    interference_limited = bool(topology_context.get("interference_limited", False))
    coverage_limited = bool(topology_context.get("coverage_limited", False))
    site_bad_cells_df = site_bad_cells_df.sort_values(["total_bad_samples", "Bad SINR", "Bad RSRQ", "Bad RSRP"], ascending=[False, False, False, False]).reset_index(drop=True)
    bad_cells = site_bad_cells_df["Cell ID"].astype(str).tolist()
    if not bad_cells:
        return candidates

    def _topology_tie_break(updates: Sequence[Dict[str, object]]) -> float:
        adjustment = 0.0
        for update in updates:
            parameter = str(update.get("parameter", "")).strip()
            cell_id = str(update.get("cell_id", "")).strip()
            row = site_cells_df.loc[site_cells_df["Cell ID"] == cell_id]
            if row.empty:
                continue
            if parameter == "ETilt":
                curr = pd.to_numeric(pd.Series([row.iloc[0].get("electrical_tilt")]), errors="coerce").iloc[0]
                target = pd.to_numeric(pd.Series([update.get("target_value")]), errors="coerce").iloc[0]
                if pd.isna(curr) or pd.isna(target):
                    continue
                delta = float(target) - float(curr)
                if interference_limited:
                    adjustment += 0.35 if delta > 0 else -0.30
                if coverage_limited:
                    adjustment += 0.30 if delta < 0 else -0.25
            elif parameter == "TX Power":
                curr = pd.to_numeric(pd.Series([row.iloc[0].get("tx_power")]), errors="coerce").iloc[0]
                target = pd.to_numeric(pd.Series([update.get("target_value")]), errors="coerce").iloc[0]
                if pd.isna(curr) or pd.isna(target):
                    continue
                delta = float(target) - float(curr)
                if interference_limited:
                    adjustment += 0.30 if delta < 0 else -0.40
                if coverage_limited:
                    adjustment += 0.25 if delta > 0 else -0.20
            elif parameter == "Azimuth" and interference_limited:
                adjustment += 0.10
        return float(adjustment)

    def _etilt_updates(cell_ids: Sequence[str], delta: float, exploratory: bool = False) -> List[Dict[str, object]]:
        updates: List[Dict[str, object]] = []
        for cid in cell_ids:
            row = site_cells_df.loc[site_cells_df["Cell ID"] == cid]
            if row.empty:
                continue
            curr = pd.to_numeric(pd.Series([row.iloc[0].get("electrical_tilt")]), errors="coerce").iloc[0]
            if pd.isna(curr):
                continue
            target = (
                _exploratory_etilt_target(float(curr), float(curr) + float(delta))
                if exploratory else
                _bounded_etilt_target(float(curr), float(curr) + float(delta))
            )
            if pd.isna(target) or np.isclose(float(target), float(curr), equal_nan=True):
                continue
            updates.append({"cell_id": cid, "parameter": "ETilt", "target_value": float(target)})
        return updates

    def _tx_updates(cell_ids: Sequence[str], delta: float) -> List[Dict[str, object]]:
        updates: List[Dict[str, object]] = []
        for cid in cell_ids:
            row = site_cells_df.loc[site_cells_df["Cell ID"] == cid]
            if row.empty:
                continue
            curr = pd.to_numeric(pd.Series([row.iloc[0].get("tx_power")]), errors="coerce").iloc[0]
            if pd.isna(curr):
                continue
            target = float(curr) + float(delta)
            if np.isclose(float(target), float(curr), equal_nan=True):
                continue
            updates.append({"cell_id": cid, "parameter": "TX Power", "target_value": float(target)})
        return updates

    def _azimuth_candidate_specs(max_cells: int = 1, steps: Sequence[float] = (-10.0, -5.0, 5.0, 10.0)) -> List[tuple[str, List[Dict[str, object]], float]]:
        specs: List[tuple[str, List[Dict[str, object]], float]] = []
        for step in steps:
            updates: List[Dict[str, object]] = []
            for cid in bad_cells[:max_cells]:
                if str(swap_dict.get(cid, "No")).strip().upper() == "YES":
                    continue
                row = site_cells_df.loc[site_cells_df["Cell ID"] == cid]
                if row.empty:
                    continue
                curr_az = pd.to_numeric(pd.Series([row.iloc[0].get("azimuth")]), errors="coerce").iloc[0]
                if pd.isna(curr_az):
                    continue
                target = TILT_SRC._normalize_azimuth(float(curr_az) + float(step))
                if np.isclose(float(target), float(curr_az), equal_nan=True):
                    continue
                updates.append({"cell_id": cid, "parameter": "Azimuth", "target_value": float(target)})
            if updates:
                step_name = f"plus_{int(step)}" if step > 0 else f"minus_{abs(int(step))}"
                specs.append((f"worst_cell_azimuth_{step_name}", updates, 0.65 if abs(step) == 5.0 else 0.55))
        return specs

    # Simulation should decide winners, so build a broad candidate set instead of
    # pre-filtering on heuristic root-cause families.
    candidate_specs = [
        ("worst_cell_etilt_plus_1", _etilt_updates(bad_cells[:1], +1.0), 1.2),
        ("worst_cell_etilt_minus_1", _etilt_updates(bad_cells[:1], -1.0), 1.2),
        ("worst_cell_etilt_plus_2", _etilt_updates(bad_cells[:1], +2.0), 1.0),
        ("worst_cell_etilt_minus_2", _etilt_updates(bad_cells[:1], -2.0), 1.0),
        ("worst_cell_etilt_plus_3", _etilt_updates(bad_cells[:1], +3.0, exploratory=True), 0.95),
        ("worst_cell_etilt_minus_3", _etilt_updates(bad_cells[:1], -3.0, exploratory=True), 0.95),
        ("worst_cell_etilt_plus_4", _etilt_updates(bad_cells[:1], +4.0, exploratory=True), 0.9),
        ("worst_cell_etilt_minus_4", _etilt_updates(bad_cells[:1], -4.0, exploratory=True), 0.9),
        ("worst_cell_tx_plus_1", _tx_updates(bad_cells[:1], +1.0), 0.95),
        ("worst_cell_tx_minus_1", _tx_updates(bad_cells[:1], -1.0), 0.95),
        ("worst_cell_tx_plus_2", _tx_updates(bad_cells[:1], +EXPLORATORY_TX_POWER_DELTA_DB), 0.9),
        ("worst_cell_tx_minus_2", _tx_updates(bad_cells[:1], -EXPLORATORY_TX_POWER_DELTA_DB), 0.9),
        ("top2_etilt_plus_1", _etilt_updates(bad_cells[:2], +1.0), 0.85),
        ("top2_etilt_minus_1", _etilt_updates(bad_cells[:2], -1.0), 0.85),
        ("top2_etilt_plus_2", _etilt_updates(bad_cells[:2], +2.0), 0.8),
        ("top2_etilt_minus_2", _etilt_updates(bad_cells[:2], -2.0), 0.8),
        ("top2_etilt_plus_3", _etilt_updates(bad_cells[:2], +3.0, exploratory=True), 0.75),
        ("top2_etilt_minus_3", _etilt_updates(bad_cells[:2], -3.0, exploratory=True), 0.75),
        ("site_etilt_plus_1", _etilt_updates(bad_cells, +1.0), 0.5),
        ("site_etilt_minus_1", _etilt_updates(bad_cells, -1.0), 0.5),
        ("site_etilt_plus_2", _etilt_updates(bad_cells, +2.0), 0.45),
        ("site_etilt_minus_2", _etilt_updates(bad_cells, -2.0), 0.45),
        ("site_tx_plus_1", _tx_updates(bad_cells, +1.0), 0.45),
        ("site_tx_minus_1", _tx_updates(bad_cells, -1.0), 0.45),
        ("site_tx_plus_2", _tx_updates(bad_cells, +EXPLORATORY_TX_POWER_DELTA_DB), 0.4),
        ("site_tx_minus_2", _tx_updates(bad_cells, -EXPLORATORY_TX_POWER_DELTA_DB), 0.4),
        (
            "top2_etilt_plus_1_top1_tx_minus_1",
            _etilt_updates(bad_cells[:2], +1.0) + _tx_updates(bad_cells[:1], -1.0),
            0.8,
        ),
        (
            "top2_etilt_minus_1_top1_tx_plus_1",
            _etilt_updates(bad_cells[:2], -1.0) + _tx_updates(bad_cells[:1], +1.0),
            0.8,
        ),
        (
            "top2_etilt_plus_2_top1_tx_minus_2",
            _etilt_updates(bad_cells[:2], +2.0) + _tx_updates(bad_cells[:1], -EXPLORATORY_TX_POWER_DELTA_DB),
            0.75,
        ),
        (
            "top2_etilt_minus_2_top1_tx_plus_2",
            _etilt_updates(bad_cells[:2], -2.0) + _tx_updates(bad_cells[:1], +EXPLORATORY_TX_POWER_DELTA_DB),
            0.75,
        ),
    ]
    az_candidate_specs = _azimuth_candidate_specs(
        max_cells=min(2, len(bad_cells)),
        steps=(-EXPLORATORY_AZIMUTH_STEP_DEG, -10.0, -5.0, 5.0, 10.0, EXPLORATORY_AZIMUTH_STEP_DEG),
    )
    candidate_specs.extend(az_candidate_specs)
    for az_name, az_updates, _ in az_candidate_specs:
        candidate_specs.extend(
            [
                (f"{az_name}_etilt_plus_1", az_updates + _etilt_updates(bad_cells[:1], +1.0), 0.6),
                (f"{az_name}_etilt_minus_1", az_updates + _etilt_updates(bad_cells[:1], -1.0), 0.6),
                (f"{az_name}_etilt_plus_2", az_updates + _etilt_updates(bad_cells[:1], +2.0), 0.58),
                (f"{az_name}_etilt_minus_2", az_updates + _etilt_updates(bad_cells[:1], -2.0), 0.58),
                (f"{az_name}_tx_plus_1", az_updates + _tx_updates(bad_cells[:1], +1.0), 0.56),
                (f"{az_name}_tx_minus_1", az_updates + _tx_updates(bad_cells[:1], -1.0), 0.56),
            ]
        )

    for name, updates, tie_break in candidate_specs:
        candidates.append({
            "name": name,
            "updates": updates,
            "tie_break": float(tie_break) + _topology_tie_break(updates),
            "root_cause": topology_root_cause,
        })

    filtered: List[Dict[str, object]] = []
    for candidate in candidates:
        updates = list(candidate.get("updates", []))
        if candidate["name"] != "hold" and not updates:
            continue
        filtered.append(candidate)
    return filtered


def _optuna_site_cluster_search(
    site_id: str,
    site_cells_df: pd.DataFrame,
    tunable_cells_df: pd.DataFrame,
    site_bad_cells_df: pd.DataFrame,
    swap_dict: Dict[str, str],
    antenna_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    topology_context: Dict[str, object],
    fixed_affected_cells: Sequence[str],
    fixed_affected_sites: Sequence[str],
    fixed_changed_rows: pd.DataFrame,
    fixed_baseline_scope: pd.DataFrame,
    fixed_baseline_bad_count: int,
    config: TiltRecommendationTestConfig,
    baseline_job_id: str,
    constraint_map: Dict[str, Dict[str, object]],
) -> tuple[Optional[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    if optuna is None:
        raise ImportError(
            "Optuna is required for single-engine tilt candidate optimization. Install it with `pip install optuna`."
        )

    cluster_cells_df = tunable_cells_df.copy() if not tunable_cells_df.empty else site_cells_df.copy()
    sorted_bad = site_bad_cells_df.sort_values(
        ["total_bad_samples", "Bad SINR", "Bad RSRQ", "Bad RSRP"],
        ascending=[False, False, False, False],
    )
    bad_target_cells = sorted_bad["Cell ID"].astype(str).tolist()
    priority_cells = (
        cluster_cells_df.sort_values(
            ["primary_site_flag", "action_priority", "Cell ID"],
            ascending=[False, False, True],
        )["Cell ID"].astype(str).tolist()
        if "action_priority" in cluster_cells_df.columns else cluster_cells_df["Cell ID"].astype(str).tolist()
    )
    target_cells = list(dict.fromkeys([*bad_target_cells, *priority_cells]))[: min(MAX_COORDINATED_ACTION_CELLS, int(config.max_ranked_cells))]
    if not target_cells:
        return None, [], []

    interference_limited = bool(topology_context.get("interference_limited", False))
    coverage_limited = bool(topology_context.get("coverage_limited", False))
    topology_root_cause = str(topology_context.get("topology_root_cause", "mixed_topology"))
    site_cell_ids = cluster_cells_df["Cell ID"].astype(str).tolist()
    evaluation_rows: List[Dict[str, object]] = []
    rf_cache: Dict[tuple, Dict[str, object]] = {}
    eval_lock = threading.Lock()

    def _etilt_delta_choices() -> List[int]:
        if interference_limited:
            return [0, 1, 2, 3, 4, -1, -2]
        if coverage_limited:
            return [0, -1, -2, -3, -4, 1, 2]
        return [0, -1, 1, -2, 2, -3, 3, -4, 4]

    def _tx_delta_choices() -> List[int]:
        if interference_limited:
            return [0, -1, -2, 1]
        if coverage_limited:
            return [0, 1, 2, -1]
        return [0, -1, 1]

    def _azimuth_delta_choices(cell_id: str) -> List[int]:
        if str(swap_dict.get(cell_id, "No")).strip().upper() == "YES":
            return [0]
        return [0, -10, -5, 5, 10] if interference_limited else [0, -5, 5, -10, 10]

    def _cell_current(cell_id: str, column: str) -> float:
        row = cluster_cells_df.loc[cluster_cells_df["Cell ID"].astype(str) == str(cell_id)]
        if row.empty:
            return np.nan
        return pd.to_numeric(pd.Series([row.iloc[0].get(column)]), errors="coerce").iloc[0]

    def _target_with_user_bounds(cell_id: str, parameter: str, current_value: float, delta: float) -> Optional[Dict[str, object]]:
        if pd.isna(current_value):
            return None
        if parameter == "ETilt":
            target = _bounded_etilt_target(float(current_value), float(current_value) + float(delta))
        elif parameter == "Azimuth":
            if str(swap_dict.get(cell_id, "No")).strip().upper() == "YES":
                return None
            target = TILT_SRC._normalize_azimuth(float(current_value) + float(delta))
        elif parameter == "TX Power":
            target = float(current_value) + float(delta)
        else:
            return None
        target = _clip_target_to_user_constraint_test_only(cell_id, parameter, float(target), constraint_map)
        if pd.isna(target) or np.isclose(float(target), float(current_value), equal_nan=True):
            return None
        return {"cell_id": str(cell_id), "parameter": parameter, "target_value": float(target)}

    def _dedupe_updates(updates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        deduped: Dict[tuple, Dict[str, object]] = {}
        for update in updates:
            key = (str(update.get("cell_id", "")), str(update.get("parameter", "")))
            if key[0] and key[1]:
                deduped[key] = dict(update)
        return list(deduped.values())

    def _coarse_candidate_specs() -> List[tuple[str, List[Dict[str, object]], float]]:
        primary_cells = cluster_cells_df.loc[
            cluster_cells_df.get("primary_site_flag", 0).astype(int) == 1,
            "Cell ID",
        ].astype(str).tolist() if "primary_site_flag" in cluster_cells_df.columns else []
        primary_cells = [cell for cell in target_cells if cell in set(primary_cells)]
        neighbor_cells = [cell for cell in target_cells if cell not in set(primary_cells)]
        primary_focus = primary_cells[: min(3, len(primary_cells))]
        neighbor_focus = neighbor_cells[: min(3, len(neighbor_cells))]
        worst_focus = target_cells[: min(3, len(target_cells))]

        specs: List[tuple[str, List[Dict[str, object]], float]] = []

        def _add_spec(name: str, cells: Sequence[str], parameter: str, delta: float, priority: float) -> None:
            updates: List[Dict[str, object]] = []
            column = {"ETilt": "electrical_tilt", "Azimuth": "azimuth", "TX Power": "tx_power"}[parameter]
            for cell in cells:
                update = _target_with_user_bounds(cell, parameter, _cell_current(cell, column), delta)
                if update:
                    updates.append(update)
            updates = _dedupe_updates(updates)
            if updates:
                specs.append((name, updates, float(priority)))

        for delta in COARSE_ETILT_DELTAS:
            label = f"plus_{int(delta)}" if delta > 0 else f"minus_{abs(int(delta))}"
            _add_spec(f"coarse_worst_etilt_{label}", worst_focus[:1], "ETilt", delta, 1.0)
            _add_spec(f"coarse_primary_etilt_{label}", primary_focus, "ETilt", delta, 0.9)
            if neighbor_focus:
                _add_spec(f"coarse_neighbor_etilt_{label}", neighbor_focus, "ETilt", delta, 0.75)

        for delta in COARSE_AZIMUTH_DELTAS:
            label = f"plus_{int(delta)}" if delta > 0 else f"minus_{abs(int(delta))}"
            _add_spec(f"coarse_primary_azimuth_{label}", primary_focus[:2], "Azimuth", delta, 0.7)
            if neighbor_focus:
                _add_spec(f"coarse_neighbor_azimuth_{label}", neighbor_focus[:2], "Azimuth", delta, 0.65)

        for delta in COARSE_TX_POWER_DELTAS:
            label = f"plus_{int(delta)}" if delta > 0 else f"minus_{abs(int(delta))}"
            _add_spec(f"coarse_primary_tx_{label}", primary_focus, "TX Power", delta, 0.6)
            if neighbor_focus:
                _add_spec(f"coarse_neighbor_tx_{label}", neighbor_focus, "TX Power", delta, 0.55)

        if primary_focus and neighbor_focus:
            for primary_delta, neighbor_delta, name in [
                (4.0, -4.0, "coarse_primary_tighter_neighbor_looser"),
                (-4.0, 4.0, "coarse_primary_looser_neighbor_tighter"),
                (8.0, -4.0, "coarse_primary_hard_tighter_neighbor_looser"),
                (-8.0, 4.0, "coarse_primary_hard_looser_neighbor_tighter"),
            ]:
                updates = []
                for cell in primary_focus:
                    update = _target_with_user_bounds(cell, "ETilt", _cell_current(cell, "electrical_tilt"), primary_delta)
                    if update:
                        updates.append(update)
                for cell in neighbor_focus:
                    update = _target_with_user_bounds(cell, "ETilt", _cell_current(cell, "electrical_tilt"), neighbor_delta)
                    if update:
                        updates.append(update)
                updates = _dedupe_updates(updates)
                if updates:
                    specs.append((name, updates, 1.1))

        unique_specs: Dict[tuple, tuple[str, List[Dict[str, object]], float]] = {}
        for name, updates, priority in specs:
            unique_specs[_candidate_cache_key(site_id, updates)] = (name, updates, priority)
        return list(unique_specs.values())

    def _result_from_metrics(
        candidate_name: str,
        updates: Sequence[Dict[str, object]],
        metrics: Dict[str, float],
        affected_cells: Sequence[str],
        affected_sites: Sequence[str],
        calibration_cells: Sequence[str],
        cheap_score: float,
        error: str = "",
    ) -> Dict[str, object]:
        result = {
            "site_id": site_id,
            "cell_id": ",".join(site_cell_ids),
            "parameter": "SITE_OPTUNA_CLUSTER",
            "candidate_name": candidate_name,
            "current_value": "",
            "target_value": json.dumps(
                [
                    {
                        "cell_id": str(u.get("cell_id", "")),
                        "parameter": str(u.get("parameter", "")),
                        "target_value": float(u.get("target_value", 0.0)),
                    }
                    for u in updates
                ],
                sort_keys=True,
            ),
            "tie_break": float(cheap_score),
            "affected_cells": len(affected_cells),
            "affected_sites": len(affected_sites),
            "changed_cells": len(calibration_cells),
            "tunable_site_count": int(cluster_cells_df["Site ID"].astype(str).nunique()) if "Site ID" in cluster_cells_df.columns else 1,
            "tunable_cell_count": int(cluster_cells_df["Cell ID"].astype(str).nunique()) if "Cell ID" in cluster_cells_df.columns else len(target_cells),
            "tunable_sites": ",".join(cluster_cells_df["Site ID"].astype(str).drop_duplicates().tolist()) if "Site ID" in cluster_cells_df.columns else str(site_id),
            "target_action_cells": ",".join([str(c) for c in target_cells]),
            "cluster_eval_sample_count": int(len(fixed_baseline_scope)),
            "cluster_baseline_bad_count": int(fixed_baseline_bad_count),
            "root_cause": topology_root_cause,
            "topology_root_cause": topology_root_cause,
            "interference_limited": int(interference_limited),
            "coverage_limited": int(coverage_limited),
            "mean_interference_gap_db": topology_context.get("mean_interference_gap_db", np.nan),
            "same_earfcn_interferer_share": topology_context.get("same_earfcn_interferer_share", np.nan),
            "dominant_interferer_share": topology_context.get("dominant_interferer_share", np.nan),
            "update_count": len(updates),
            "selection_stage": "coordinated_optuna_site_cluster",
            "cheap_score": float(cheap_score),
            **metrics,
            "error": error,
        }
        result["reject_reason"] = _site_candidate_reject_reason(result, config)
        return result

    def _error_result(candidate_name: str, updates: Sequence[Dict[str, object]], error: Exception) -> Dict[str, object]:
        metrics = {
            "baseline_bad_count": np.nan,
            "candidate_bad_count": np.nan,
            "recovered_bad_samples": np.nan,
            "new_bad_samples": np.nan,
            "rsrp_recovered_bad": np.nan,
            "rsrp_new_bad": np.nan,
            "rsrq_recovered_bad": np.nan,
            "rsrq_new_bad": np.nan,
            "sinr_recovered_bad": np.nan,
            "sinr_new_bad": np.nan,
            "rsrp_severity_reduction": np.nan,
            "rsrq_severity_reduction": np.nan,
            "sinr_severity_reduction": np.nan,
            "total_severity_reduction": np.nan,
            "evaluation_sample_count": np.nan,
            "rsrp_severity_reduction_per_sample": np.nan,
            "rsrq_severity_reduction_per_sample": np.nan,
            "sinr_severity_reduction_per_sample": np.nan,
            "total_severity_reduction_per_sample": np.nan,
            "net_bad_reduction": np.nan,
            "net_bad_reduction_share": np.nan,
            "recovered_bad_share": np.nan,
            "new_bad_share": np.nan,
            "rsrp_recovered_bad_share": np.nan,
            "rsrp_new_bad_share": np.nan,
            "rsrq_recovered_bad_share": np.nan,
            "rsrq_new_bad_share": np.nan,
            "sinr_recovered_bad_share": np.nan,
            "sinr_new_bad_share": np.nan,
            "good_area_loss_pct": np.nan,
            "mean_rsrp_delta": np.nan,
            "mean_rsrq_delta": np.nan,
            "mean_sinr_delta": np.nan,
            "optimized_row_count": np.nan,
            "merged_row_count": np.nan,
            "baseline_row_count": np.nan,
            "optimized_distinct_cell_count": np.nan,
            "score": -99999.0,
            "constraints_passed": 0.0,
        }
        return _result_from_metrics(candidate_name, updates, metrics, [], [], [], -999.0, str(error))

    def _evaluate_updates(candidate_name: str, updates: Sequence[Dict[str, object]]) -> Dict[str, object]:
        updates = list(updates)
        if updates:
            updates = _dedupe_updates(updates)
            modified_site_df = _apply_multiple_parameter_targets(antenna_df, updates)
            cheap_score = _fast_geometry_score(
                baseline_df=baseline_df,
                site_df=antenna_df,
                modified_site_df=modified_site_df,
                target_site_id=site_id,
                target_updates=updates,
            )
            if ENABLE_GEOMETRY_PREFILTER_PRUNING and cheap_score < OPTUNA_PREFILTER_SCORE:
                raise optuna.TrialPruned()
            cache_key = _candidate_cache_key(site_id, updates)
            with eval_lock:
                cached = rf_cache.get(cache_key)
            if cached is None:
                affected_cells, affected_sites, changed_rows = opt_ml._compute_affected_cells(
                    modified_site_df,
                    float(config.impact_radius_m),
                    int(config.neighbor_site_count),
                )
                update_cells = sorted({str(update.get("cell_id", "")) for update in updates if str(update.get("cell_id", "")).strip()})
                calibration_cells = sorted(set(affected_cells).union(update_cells))
                if not calibration_cells:
                    raise ValueError("No changed cells found for candidate")
                k1k2_map = opt_ml.compute_k1k2_for_cells(baseline_df, modified_site_df, calibration_cells)
                if not k1k2_map:
                    raise ValueError("No calibrated cells found for candidate")
                optimized_df = opt_ml.run_prediction_only_optimized(
                    modified_site_df,
                    k1k2_map,
                    {
                        "project_id": int(config.project_id),
                        "region": config.region,
                        "radius": float(config.radius_m),
                        "grid_resolution": float(config.grid_resolution_m),
                        "n_workers": int(config.workers),
                        "impact_radius_m": float(config.impact_radius_m),
                        "neighbor_site_count": int(config.neighbor_site_count),
                        "max_interference_sites": int(config.max_interference_sites),
                        "baseline_job_id": baseline_job_id,
                    },
                )
                merged_df = opt_ml.replace_cells(baseline_df, optimized_df)
                evaluation_cells = _expand_evaluation_cells_from_topology(baseline_df, affected_cells)
                metrics = _score_candidate_vs_baseline(baseline_df, merged_df, evaluation_cells, config)
                metrics.update(
                    {
                        "optimized_row_count": float(len(optimized_df)),
                        "merged_row_count": float(len(merged_df)),
                        "baseline_row_count": float(len(baseline_df)),
                        "optimized_distinct_cell_count": float(optimized_df["Node_Cell_ID"].astype(str).nunique()) if "Node_Cell_ID" in optimized_df.columns else 0.0,
                    }
                )
                print(
                    f"[TILT_TEST][RF_REPLACE_DIAG] site_id={site_id} candidate={candidate_name} "
                    f"updates={len(updates)} affected_cells={len(affected_cells)} calibration_cells={len(calibration_cells)} "
                    f"optimized_rows={len(optimized_df)} merged_rows={len(merged_df)} baseline_rows={len(baseline_df)} "
                    f"score={float(metrics.get('score', -99999.0)):.4f} "
                    f"net_bad_reduction={float(metrics.get('net_bad_reduction', 0.0)):.0f}"
                )
                cached = {
                    "affected_cells_list": affected_cells,
                    "affected_sites_list": affected_sites,
                    "evaluation_cells_list": evaluation_cells,
                    "calibration_cells": calibration_cells,
                    **metrics,
                }
                with eval_lock:
                    rf_cache[cache_key] = cached
            affected_cells = cached["affected_cells_list"]
            affected_sites = cached["affected_sites_list"]
            calibration_cells = cached["calibration_cells"]
            metrics = {k: v for k, v in cached.items() if k not in {"affected_cells_list", "affected_sites_list", "evaluation_cells_list", "calibration_cells"}}
        else:
            cheap_score = 0.0
            affected_cells = list(fixed_affected_cells)
            affected_sites = list(fixed_affected_sites)
            calibration_cells = []
            evaluation_cells = _expand_evaluation_cells_from_topology(baseline_df, affected_cells)
            metrics = _score_candidate_vs_baseline(baseline_df, baseline_df, evaluation_cells, config)

        if float(metrics["good_area_loss_pct"]) > OPTUNA_PRUNE_GOOD_AREA_LOSS_PCT:
            raise optuna.TrialPruned()
        if float(metrics["mean_sinr_delta"]) < OPTUNA_PRUNE_MEAN_SINR_DELTA_DB:
            raise optuna.TrialPruned()
        return _result_from_metrics(candidate_name, updates, metrics, affected_cells, affected_sites, calibration_cells, cheap_score)

    hold_result = _evaluate_updates("hold", [])
    evaluation_rows.append(hold_result)

    coarse_specs = _coarse_candidate_specs()
    cheap_ranked_specs: List[tuple[float, str, List[Dict[str, object]]]] = []
    for name, updates, priority in coarse_specs:
        try:
            modified_site_df = _apply_multiple_parameter_targets(antenna_df, updates)
            cheap_score = _fast_geometry_score(
                baseline_df=baseline_df,
                site_df=antenna_df,
                modified_site_df=modified_site_df,
                target_site_id=site_id,
                target_updates=updates,
            )
            cheap_ranked_specs.append((float(cheap_score) + float(priority), name, updates))
        except Exception as exc:
            evaluation_rows.append(_error_result(name, updates, exc))

    coarse_results: List[Dict[str, object]] = []
    for _, name, updates in sorted(cheap_ranked_specs, key=lambda item: item[0], reverse=True)[:COARSE_RF_VALIDATION_COUNT]:
        try:
            result = _evaluate_updates(name, updates)
        except optuna.TrialPruned:
            continue
        except Exception as exc:
            result = _error_result(name, updates, exc)
        result["selection_stage"] = "coarse_rf_screen"
        coarse_results.append(result)
        evaluation_rows.append(result)

    coarse_positive = [
        result for result in coarse_results
        if bool(int(result.get("constraints_passed", 0)))
        and (
            float(result.get("score", -99999.0)) > float(COARSE_MIN_SCORE_FOR_REFINEMENT)
            or float(result.get("net_bad_reduction", 0.0)) > 0.0
            or float(result.get("recovered_bad_samples", 0.0)) > float(result.get("new_bad_samples", 0.0))
        )
    ]
    if not coarse_positive:
        completed_results = [hold_result, *coarse_results]
        chosen_result = max(
            completed_results,
            key=lambda result: (
                int(bool(int(result.get("constraints_passed", 0)))),
                float(result.get("score", -99999.0)),
                float(result.get("total_severity_reduction", -99999.0)),
                float(result.get("sinr_severity_reduction", -99999.0)),
                -float(result.get("new_bad_samples", 99999.0)),
            ),
        )
        selected_updates = [] if str(chosen_result.get("candidate_name")) == "hold" else json.loads(str(chosen_result.get("target_value", "[]") or "[]"))
        return chosen_result, selected_updates, evaluation_rows

    def objective(trial: "optuna.trial.Trial") -> float:
        updates: List[Dict[str, object]] = []
        for cell_id in target_cells:
            current_etilt = _cell_current(cell_id, "electrical_tilt")
            if pd.notna(current_etilt):
                etilt_delta = int(trial.suggest_categorical(f"{cell_id}__etilt_delta", _etilt_delta_choices()))
                target = float(_bounded_etilt_target(float(current_etilt), float(current_etilt) + float(etilt_delta)))
                target = _clip_target_to_user_constraint_test_only(cell_id, "ETilt", target, constraint_map)
                if not np.isclose(target, float(current_etilt), equal_nan=True):
                    updates.append({"cell_id": cell_id, "parameter": "ETilt", "target_value": target})

            current_tx = _cell_current(cell_id, "tx_power")
            if pd.notna(current_tx):
                tx_delta = int(trial.suggest_categorical(f"{cell_id}__tx_delta", _tx_delta_choices()))
                target = float(current_tx) + float(tx_delta)
                target = _clip_target_to_user_constraint_test_only(cell_id, "TX Power", target, constraint_map)
                if not np.isclose(target, float(current_tx), equal_nan=True):
                    updates.append({"cell_id": cell_id, "parameter": "TX Power", "target_value": target})

            current_az = _cell_current(cell_id, "azimuth")
            if pd.notna(current_az):
                az_delta = int(trial.suggest_categorical(f"{cell_id}__az_delta", _azimuth_delta_choices(cell_id)))
                target = float(TILT_SRC._normalize_azimuth(float(current_az) + float(az_delta)))
                target = _clip_target_to_user_constraint_test_only(cell_id, "Azimuth", target, constraint_map)
                if not np.isclose(target, float(current_az), equal_nan=True):
                    updates.append({"cell_id": cell_id, "parameter": "Azimuth", "target_value": target})
        updates = _dedupe_updates(updates)

        try:
            result = _evaluate_updates(f"trial_{trial.number}", updates)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            result = _error_result(f"trial_{trial.number}", updates, exc)
        trial.set_user_attr("updates", list(updates))
        trial.set_user_attr("result", dict(result))
        with eval_lock:
            evaluation_rows.append(result)
        return float(result["score"])

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, warn_independent_sampling=False)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=OPTUNA_TRIAL_COUNT, n_jobs=OPTUNA_N_JOBS, show_progress_bar=False)

    completed_results = [hold_result, *coarse_results]
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            result = trial.user_attrs.get("result")
            if isinstance(result, dict):
                completed_results.append(result)

    def _optuna_result_rank(result: Dict[str, object]) -> tuple:
        return (
            int(bool(int(result.get("constraints_passed", 0)))),
            float(result.get("score", -99999.0)),
            float(result.get("total_severity_reduction", -99999.0)),
            float(result.get("sinr_severity_reduction", -99999.0)),
            -float(result.get("new_bad_samples", 99999.0)),
        )

    chosen_result = max(completed_results, key=_optuna_result_rank)
    selected_updates = []
    if str(chosen_result.get("candidate_name")) != "hold":
        for trial in study.trials:
            result = trial.user_attrs.get("result")
            if isinstance(result, dict) and str(result.get("candidate_name")) == str(chosen_result.get("candidate_name")):
                selected_updates = list(trial.user_attrs.get("updates", []))
                break
        if not selected_updates:
            selected_updates = json.loads(str(chosen_result.get("target_value", "[]") or "[]"))

    return chosen_result, selected_updates, evaluation_rows


def _build_kpi_first_recommendations_test_only(
    summary_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    geo_cell_summary: pd.DataFrame,
    bearing_summary: pd.DataFrame,
    swap_dict: Dict[str, str],
    baseline_df: pd.DataFrame,
    config: TiltRecommendationTestConfig,
    run_dir: Path,
    constraint_map: Dict[str, Dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked_cells = _rank_cells_by_bad_contribution(summary_df)
    ranked_sites = _rank_sites_by_bad_contribution(summary_df, antenna_df)
    if ranked_cells.empty or ranked_sites.empty:
        return pd.DataFrame(), pd.DataFrame()

    ant_use = _build_cell_site_map(antenna_df)
    tech_col = "Technology" if "Technology" in ant_use.columns else ""

    geo_map = {}
    if not geo_cell_summary.empty:
        geo_work = geo_cell_summary.copy()
        geo_work["Cell ID"] = geo_work["Cell ID"].map(TILT_SRC._norm_cell_id)
        geo_map = geo_work.set_index("Cell ID").to_dict("index")

    baseline_job_id = _fetch_latest_baseline_job_id(config.project_id, config.region)
    evaluation_rows: List[Dict[str, object]] = []
    recommendation_rows: List[Dict[str, object]] = []

    target_sites = ranked_sites.head(int(config.max_ranked_sites)).copy()
    for _, site_row in target_sites.iterrows():
        site_id = str(site_row["Site ID"])
        site_cells_df = ant_use.loc[ant_use["Site ID"] == site_id].copy()
        if site_cells_df.empty:
            continue
        site_cell_ids = site_cells_df["Cell ID"].astype(str).tolist()
        tunable_cells_df = _build_coordinated_action_cluster(
            site_id=site_id,
            site_cell_ids=site_cell_ids,
            antenna_df=antenna_df,
            baseline_df=baseline_df,
            ranked_cells=ranked_cells,
        )
        tunable_cell_ids = tunable_cells_df["Cell ID"].astype(str).tolist()
        site_bad_cells_df = ranked_cells.loc[ranked_cells["Cell ID"].isin(tunable_cell_ids)].copy()
        primary_bad_cells_df = ranked_cells.loc[ranked_cells["Cell ID"].isin(site_cells_df["Cell ID"])].copy()
        if primary_bad_cells_df.empty:
            continue
        site_geo = geo_cell_summary.loc[
            geo_cell_summary["Cell ID"].astype(str).isin(site_cells_df["Cell ID"].astype(str))
        ].copy() if not geo_cell_summary.empty else pd.DataFrame()
        site_nlos_share = float(pd.to_numeric(site_geo.get("nlos_share"), errors="coerce").mean()) if not site_geo.empty else np.nan
        site_los_blocked = float(pd.to_numeric(site_geo.get("mean_los_blocked_ratio"), errors="coerce").mean()) if not site_geo.empty else np.nan
        site_clutter = str(site_geo["clutter_mode"].mode().iloc[0]) if not site_geo.empty and "clutter_mode" in site_geo.columns and not site_geo["clutter_mode"].dropna().empty else ""
        non_tunable = (
            pd.notna(site_nlos_share) and site_nlos_share > 0.75
            and pd.notna(site_los_blocked) and site_los_blocked > 0.20
        )

        fixed_affected_cells = _expand_evaluation_cells_from_topology(baseline_df, tunable_cell_ids)
        fixed_affected_sites = sorted(tunable_cells_df["Site ID"].astype(str).dropna().unique().tolist()) if "Site ID" in tunable_cells_df.columns else [site_id]
        fixed_changed_rows = pd.DataFrame()
        fixed_baseline_scope = baseline_df.loc[
            baseline_df["Node_Cell_ID"].astype(str).isin([str(c) for c in fixed_affected_cells])
        ].copy()
        fixed_baseline_bad_count = int(
            _is_bad_sample(
                fixed_baseline_scope,
                config.rsrp_threshold,
                config.rsrq_threshold,
                config.sinr_threshold,
            ).sum()
        ) if not fixed_baseline_scope.empty else 0
        topology_context = _baseline_topology_context(baseline_df, site_cell_ids)

        chosen_result, selected_updates, site_evaluation_rows = _optuna_site_cluster_search(
            site_id=site_id,
            site_cells_df=site_cells_df,
            tunable_cells_df=tunable_cells_df,
            site_bad_cells_df=site_bad_cells_df,
            swap_dict=swap_dict,
            antenna_df=antenna_df,
            baseline_df=baseline_df,
            topology_context=topology_context,
            fixed_affected_cells=fixed_affected_cells,
            fixed_affected_sites=fixed_affected_sites,
            fixed_changed_rows=fixed_changed_rows,
            fixed_baseline_scope=fixed_baseline_scope,
            fixed_baseline_bad_count=fixed_baseline_bad_count,
            config=config,
            baseline_job_id=baseline_job_id,
            constraint_map=constraint_map,
        )
        evaluation_rows.extend(site_evaluation_rows)
        if chosen_result is None:
            continue

        selected_by = "single_optuna_optimizer"
        site_root_cause = str(chosen_result.get("root_cause", "mixed"))
        site_total_bad = int(pd.to_numeric(pd.Series([site_row.get("total_bad_samples", 0)]), errors="coerce").fillna(0).iloc[0])
        status = "no_safe_change_available"
        best_safe_positive = (
            bool(int(chosen_result.get("constraints_passed", 0)))
            and float(chosen_result.get("score", -99999.0)) > 0.0
            and selected_updates
        )
        exploratory_review = (
            bool(int(chosen_result.get("constraints_passed", 0)))
            and selected_updates
            and float(chosen_result.get("score", -99999.0)) > float(EXPLORATORY_REVIEW_SCORE_FLOOR)
            and (
                float(chosen_result.get("total_severity_reduction_per_sample", 0.0)) > 0.0
                or float(chosen_result.get("recovered_bad_samples", 0.0)) >= float(config.min_recovered_bad_samples)
            )
        )
        if best_safe_positive:
            status = "action_change_validated"
        elif exploratory_review:
            status = "engineering_review_tradeoff"
        elif non_tunable:
            status = "non_tunable_physical_redesign_required"

        base_reason = (
            f"Site-level cluster optimization. site_id={site_id}. bad_samples={site_total_bad}. "
            f"root_cause={site_root_cause}. net_bad_reduction={float(chosen_result.get('net_bad_reduction', 0.0)):.0f}. "
            f"recovered_bad={float(chosen_result.get('recovered_bad_samples', 0.0)):.0f}. "
            f"new_bad={float(chosen_result.get('new_bad_samples', 0.0)):.0f}. "
            f"good_area_loss_pct={float(chosen_result.get('good_area_loss_pct', 0.0)):.2f}. "
            f"mean_sinr_delta={float(chosen_result.get('mean_sinr_delta', 0.0)):.2f}. "
            f"sinr_severity_reduction={float(chosen_result.get('sinr_severity_reduction', 0.0)):.2f}. "
            f"rsrq_severity_reduction={float(chosen_result.get('rsrq_severity_reduction', 0.0)):.2f}. "
            f"rsrp_severity_reduction={float(chosen_result.get('rsrp_severity_reduction', 0.0)):.2f}. "
            f"candidate={chosen_result.get('candidate_name')}. selected_by={selected_by}. "
            f"reject_reason={chosen_result.get('reject_reason', '')}. clutter_mode={site_clutter or 'n/a'}. "
            f"site_nlos_share={_fmt_num(site_nlos_share, 2)}. site_los_blocked_ratio={_fmt_num(site_los_blocked, 2)}."
        )
        if non_tunable:
            base_reason = f"{base_reason} This site looks non-tunable by simple RF parameter changes; physical redesign may be required."
        elif status == "engineering_review_tradeoff":
            base_reason = f"{base_reason} Candidate is not auto-deploy positive, but it has bounded risk and severity/recovery evidence; export for RF engineer review."
        elif status == "no_safe_change_available":
            base_reason = f"{base_reason} Single Optuna search did not find a tradeoff-positive safe candidate."

        updates_by_key = {
            (str(u["cell_id"]), str(u["parameter"])): float(u["target_value"])
            for u in selected_updates
        }
        recommendation_confidence_score = _clamp_score(
            max(
                0.0,
                min(
                    100.0,
                    float(chosen_result.get("score", 0.0)) * 4.0
                    + (15.0 if bool(int(chosen_result.get("constraints_passed", 0))) else 0.0),
                ),
            )
        )
        for _, sector_row in tunable_cells_df.iterrows():
            cell_id = str(sector_row["Cell ID"])
            ant_tech = TILT_SRC._safe_str(sector_row[tech_col]) if tech_col and tech_col in sector_row else "4G"
            for parameter, current_col in [("ETilt", "electrical_tilt"), ("Azimuth", "azimuth"), ("TX Power", "tx_power")]:
                current_value = pd.to_numeric(pd.Series([sector_row.get(current_col)]), errors="coerce").iloc[0]
                recommended_value = updates_by_key.get((cell_id, parameter), current_value)
                row_status = status if not np.isclose(float(recommended_value), float(current_value), equal_nan=True) else ("no_change" if status == "action_change_validated" else status)
                recommendation_rows.append(
                    {
                        "Cell ID": cell_id,
                        "Technology": ant_tech or "4G",
                        "Parameter": parameter,
                        "Current Value": current_value,
                        "Recommended Value": recommended_value,
                        "Reason": base_reason,
                        "Swap Sector Detected": swap_dict.get(cell_id, "No"),
                        "Bad Sample Count": int(pd.to_numeric(site_bad_cells_df.loc[site_bad_cells_df["Cell ID"] == cell_id, "total_bad_samples"], errors="coerce").fillna(0).sum()),
                        "P90 Serving Distance (m)": "",
                        "Root Cause Category": site_root_cause,
                        "Recommendation Status": row_status,
                        "Recommendation Confidence": _confidence_label(recommendation_confidence_score),
                        "Confidence Score": recommendation_confidence_score,
                        "Coverage ETilt Score": round(float(chosen_result.get("net_bad_reduction", 0.0)), 2),
                        "Overlap ETilt Score": round(float(chosen_result.get("good_area_loss_pct", 0.0)), 2),
                        "Azimuth Score": round(float(chosen_result.get("mean_sinr_delta", 0.0)), 2),
                        "TX Power Score": round(float(chosen_result.get("mean_rsrp_delta", 0.0)), 2),
                        "Bearing Peak Share": "",
                        "Bearing Spread Deg": "",
                        "Directional Contrast": "",
                    }
                )

    recommendations_df = pd.DataFrame(
        recommendation_rows,
        columns=[
            "Cell ID",
            "Technology",
            "Parameter",
            "Current Value",
            "Recommended Value",
            "Reason",
            "Swap Sector Detected",
            "Bad Sample Count",
            "P90 Serving Distance (m)",
            "Root Cause Category",
            "Recommendation Status",
            "Recommendation Confidence",
            "Confidence Score",
            "Coverage ETilt Score",
            "Overlap ETilt Score",
            "Azimuth Score",
            "TX Power Score",
            "Bearing Peak Share",
            "Bearing Spread Deg",
            "Directional Contrast",
        ],
    )
    evaluation_df = pd.DataFrame(evaluation_rows)
    return recommendations_df, evaluation_df


def _summarize_recommendation_run(
    config: TiltRecommendationTestConfig,
    log_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    bad_samples_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    geo_cell_summary: pd.DataFrame,
    bearing_summary: pd.DataFrame,
    recommendations_all_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    candidate_validation_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    swap_dict: Dict[str, str],
    run_dir: Path,
    excel_path: Path,
    total_runtime_sec: float,
) -> dict:
    swap_yes = int(sum(1 for v in swap_dict.values() if str(v).strip().upper() == "YES"))
    status_counts = (
        recommendations_all_df["Recommendation Status"].astype(str).map(_norm_reason_token).value_counts().to_dict()
        if not recommendations_all_df.empty else {}
    )
    root_cause_counts = (
        recommendations_all_df["Root Cause Category"].astype(str).map(_norm_reason_token).value_counts().to_dict()
        if not recommendations_all_df.empty else {}
    )
    confidence_counts = (
        recommendations_all_df["Recommendation Confidence"].astype(str).map(_norm_reason_token).value_counts().to_dict()
        if not recommendations_all_df.empty and "Recommendation Confidence" in recommendations_all_df.columns else {}
    )

    parameter_change_counts: dict[str, int] = {}
    top_parameter_changes: list[dict[str, object]] = []
    changed_rows = pd.DataFrame()
    if not recommendations_df.empty:
        changed_mask = _changed_recommendation_mask(recommendations_df)
        changed_rows = recommendations_df.loc[changed_mask].copy()
        if not changed_rows.empty:
            parameter_change_counts = changed_rows["Parameter"].astype(str).value_counts().to_dict()
            for _, rec in changed_rows.sort_values(["Bad Sample Count", "Cell ID"], ascending=[False, True]).head(12).iterrows():
                top_parameter_changes.append(
                    {
                        "cell_id": str(rec["Cell ID"]),
                        "parameter": str(rec["Parameter"]),
                        "current_value": rec["Current Value"],
                        "recommended_value": rec["Recommended Value"],
                        "bad_sample_count": int(pd.to_numeric(pd.Series([rec["Bad Sample Count"]]), errors="coerce").fillna(0).iloc[0]),
                        "root_cause_category": str(rec["Root Cause Category"]),
                        "recommendation_status": str(rec["Recommendation Status"]),
                        "reason": str(rec["Reason"]),
                    }
                )

    geo_context_overview = {}
    if not geo_cell_summary.empty:
        geo_context_overview = {
            "mean_serving_distance_m_range": {
                "min": round(float(pd.to_numeric(geo_cell_summary["mean_serving_distance_m"], errors="coerce").min()), 2),
                "max": round(float(pd.to_numeric(geo_cell_summary["mean_serving_distance_m"], errors="coerce").max()), 2),
            },
            "p90_serving_distance_m_range": {
                "min": round(float(pd.to_numeric(geo_cell_summary["p90_serving_distance_m"], errors="coerce").min()), 2),
                "max": round(float(pd.to_numeric(geo_cell_summary["p90_serving_distance_m"], errors="coerce").max()), 2),
            },
            "nlos_share_range": {
                "min": round(float(pd.to_numeric(geo_cell_summary["nlos_share"], errors="coerce").min()), 3),
                "max": round(float(pd.to_numeric(geo_cell_summary["nlos_share"], errors="coerce").max()), 3),
            },
            "los_blocked_ratio_range": {
                "min": round(float(pd.to_numeric(geo_cell_summary["mean_los_blocked_ratio"], errors="coerce").min()), 3),
                "max": round(float(pd.to_numeric(geo_cell_summary["mean_los_blocked_ratio"], errors="coerce").max()), 3),
            },
            "clutter_mode_counts": geo_cell_summary["clutter_mode"].astype(str).replace("", "UNKNOWN").value_counts().to_dict(),
        }

    top_bad_cells = []
    if not geo_cell_summary.empty:
        top_bad = geo_cell_summary.sort_values(["bad_sample_count", "p90_serving_distance_m"], ascending=[False, False]).head(10)
        for _, row in top_bad.iterrows():
            top_bad_cells.append(
                {
                    "cell_id": str(row["Cell ID"]),
                    "technology": str(row["Technology"]),
                    "bad_sample_count": int(pd.to_numeric(pd.Series([row["bad_sample_count"]]), errors="coerce").fillna(0).iloc[0]),
                    "mean_serving_distance_m": None if pd.isna(row["mean_serving_distance_m"]) else round(float(row["mean_serving_distance_m"]), 2),
                    "p90_serving_distance_m": None if pd.isna(row["p90_serving_distance_m"]) else round(float(row["p90_serving_distance_m"]), 2),
                    "nlos_share": None if pd.isna(row["nlos_share"]) else round(float(row["nlos_share"]), 3),
                    "mean_los_blocked_ratio": None if pd.isna(row["mean_los_blocked_ratio"]) else round(float(row["mean_los_blocked_ratio"]), 3),
                    "clutter_mode": str(row["clutter_mode"] or "UNKNOWN"),
                }
            )

    forecast_highlights = []
    if not forecast_df.empty:
        for _, row in forecast_df.sort_values(["Improvement %", "Pre-Change"], ascending=[False, False]).head(10).iterrows():
            forecast_highlights.append(
                {
                    "cell_id": str(row["Cell ID"]),
                    "kpi": str(row["KPI"]),
                    "pre_change": int(row["Pre-Change"]),
                    "est_post_change": int(row["Est. Post-Change"]),
                    "improvement_pct": float(row["Improvement %"]),
                    "forecast_type": "heuristic_not_rf_validated",
                }
            )

    candidate_validation_summary = {}
    if not candidate_validation_df.empty:
        candidate_validation_summary = {
            "evaluated_candidates": int(len(candidate_validation_df)),
            "cells_evaluated": int(candidate_validation_df["cell_id"].astype(str).nunique()),
            "parameters_evaluated": candidate_validation_df["parameter"].astype(str).value_counts().to_dict(),
            "safe_candidates": int((pd.to_numeric(candidate_validation_df["constraints_passed"], errors="coerce").fillna(0) > 0).sum()),
            "best_score_max": round(float(pd.to_numeric(candidate_validation_df["score"], errors="coerce").max()), 4),
            "best_score_min": round(float(pd.to_numeric(candidate_validation_df["score"], errors="coerce").min()), 4),
        }

    return {
        "run_type": "tilt_recommendation_test",
        "project_id": int(config.project_id),
        "region": config.region,
        "operator": config.operator,
        "thresholds": {
            "rsrp": float(config.rsrp_threshold),
            "rsrq": float(config.rsrq_threshold),
            "sinr": float(config.sinr_threshold),
        },
        "logic_profile": {
            "min_bad_sample_count_for_action": MIN_BAD_SAMPLE_COUNT_FOR_ACTION,
            "medium_confidence_bad_sample_count": MEDIUM_CONFIDENCE_BAD_SAMPLE_COUNT,
            "high_confidence_bad_sample_count": HIGH_CONFIDENCE_BAD_SAMPLE_COUNT,
            "very_high_confidence_bad_sample_count": VERY_HIGH_CONFIDENCE_BAD_SAMPLE_COUNT,
            "far_edge_mean_serving_distance_m": FAR_EDGE_MEAN_SERVING_DISTANCE_M,
            "far_edge_p90_serving_distance_m": FAR_EDGE_P90_SERVING_DISTANCE_M,
            "medium_edge_mean_serving_distance_m": MEDIUM_EDGE_MEAN_SERVING_DISTANCE_M,
            "medium_edge_p90_serving_distance_m": MEDIUM_EDGE_P90_SERVING_DISTANCE_M,
            "high_nlos_share_gate": HIGH_NLOS_SHARE_GATE,
            "high_los_blocked_ratio_gate": HIGH_LOS_BLOCKED_RATIO_GATE,
            "high_building_area_ratio_gate": HIGH_BUILDING_AREA_RATIO_GATE,
            "medium_nlos_share_gate": MEDIUM_NLOS_SHARE_GATE,
            "medium_los_blocked_ratio_gate": MEDIUM_LOS_BLOCKED_RATIO_GATE,
            "dense_overlap_nearest_site_m": DENSE_OVERLAP_NEAREST_SITE_M,
            "dense_overlap_site_count_250m": DENSE_OVERLAP_SITE_COUNT_250M,
            "small_azimuth_delta_deg": SMALL_AZIMUTH_DELTA_DEG,
            "min_azimuth_mismatch_deg": MIN_AZIMUTH_MISMATCH_DEG,
            "max_azimuth_mismatch_deg": MAX_AZIMUTH_MISMATCH_DEG,
            "max_azimuth_step_deg": MAX_AZIMUTH_STEP_DEG,
            "min_bearing_sample_count": MIN_BEARING_SAMPLE_COUNT,
            "max_bearing_spread_deg": MAX_BEARING_SPREAD_DEG,
            "azimuth_nlos_hard_block_gate": AZIMUTH_NLOS_HARD_BLOCK_GATE,
            "relaxed_azimuth_bad_sample_count": RELAXED_AZIMUTH_BAD_SAMPLE_COUNT,
            "relaxed_azimuth_peak_share": RELAXED_AZIMUTH_PEAK_SHARE,
            "relaxed_azimuth_max_spread_deg": RELAXED_AZIMUTH_MAX_SPREAD_DEG,
            "min_directional_contrast_for_azimuth": MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH,
            "strong_directional_contrast_for_azimuth": STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH,
            "min_candidate_score_gap": MIN_CANDIDATE_SCORE_GAP,
            "site_symmetry_penalty_bad_sample_ratio": SITE_SYMMETRY_PENALTY_BAD_SAMPLE_RATIO,
            "min_safe_etilt_deg": MIN_SAFE_ETILT_DEG,
            "max_safe_etilt_deg": MAX_SAFE_ETILT_DEG,
            "max_etilt_increase_per_run_deg": MAX_ETILT_INCREASE_PER_RUN_DEG,
            "max_etilt_decrease_per_run_deg": MAX_ETILT_DECREASE_PER_RUN_DEG,
            "impact_radius_m": float(config.impact_radius_m),
            "min_recovered_bad_samples": int(config.min_recovered_bad_samples),
            "max_ranked_cells": int(config.max_ranked_cells),
            "max_ranked_sites": int(config.max_ranked_sites),
            "threshold_file_path": str(config.threshold_file_path or ""),
            "threshold_constraint_rows": int(config.threshold_constraint_count),
            "threshold_optimised_rows": int(config.threshold_optimised_count),
            "threshold_constraints_applied_during_trials": bool(config.threshold_optimised_count > 0),
            "max_coordinated_action_sites": int(MAX_COORDINATED_ACTION_SITES),
            "max_coordinated_neighbor_sites": int(MAX_COORDINATED_NEIGHBOR_SITES),
            "max_coordinated_action_cells": int(MAX_COORDINATED_ACTION_CELLS),
            "coarse_rf_validation_count": int(COARSE_RF_VALIDATION_COUNT),
            "coarse_min_score_for_refinement": float(COARSE_MIN_SCORE_FOR_REFINEMENT),
            "coarse_refinement_gate": "constraints_passed AND (score_above_floor OR net_bad_reduction_positive OR recovered_bad_exceeds_new_bad)",
            "coarse_etilt_deltas": [float(v) for v in COARSE_ETILT_DELTAS],
            "coarse_azimuth_deltas": [float(v) for v in COARSE_AZIMUTH_DELTAS],
            "coarse_tx_power_deltas": [float(v) for v in COARSE_TX_POWER_DELTAS],
            "optuna_trial_count": int(OPTUNA_TRIAL_COUNT),
            "optuna_n_jobs": int(OPTUNA_N_JOBS),
            "optuna_prefilter_score": float(OPTUNA_PREFILTER_SCORE),
            "geometry_prefilter_pruning_enabled": bool(ENABLE_GEOMETRY_PREFILTER_PRUNING),
            "severe_good_area_loss_pct": float(SEVERE_GOOD_AREA_LOSS_PCT),
            "severe_mean_sinr_drop_db": float(SEVERE_MEAN_SINR_DROP_DB),
            "optuna_prune_good_area_loss_pct": float(OPTUNA_PRUNE_GOOD_AREA_LOSS_PCT),
            "optuna_prune_mean_sinr_delta_db": float(OPTUNA_PRUNE_MEAN_SINR_DELTA_DB),
            "net_bad_reduction_weight": float(NET_BAD_REDUCTION_WEIGHT),
            "sinr_recovery_weight": float(SINR_RECOVERY_WEIGHT),
            "rsrq_recovery_weight": float(RSRQ_RECOVERY_WEIGHT),
            "rsrp_recovery_weight": float(RSRP_RECOVERY_WEIGHT),
            "sinr_new_bad_weight": float(SINR_NEW_BAD_WEIGHT),
            "rsrq_new_bad_weight": float(RSRQ_NEW_BAD_WEIGHT),
            "rsrp_new_bad_weight": float(RSRP_NEW_BAD_WEIGHT),
            "sinr_severity_reduction_weight": float(SINR_SEVERITY_REDUCTION_WEIGHT),
            "rsrq_severity_reduction_weight": float(RSRQ_SEVERITY_REDUCTION_WEIGHT),
            "rsrp_severity_reduction_weight": float(RSRP_SEVERITY_REDUCTION_WEIGHT),
            "total_severity_reduction_weight": float(TOTAL_SEVERITY_REDUCTION_WEIGHT),
            "good_area_loss_weight": float(GOOD_AREA_LOSS_WEIGHT),
            "severe_constraint_penalty": float(SEVERE_CONSTRAINT_PENALTY),
            "score_normalization": "severity_and_bad_counts_are_scored_per_evaluation_sample",
            "exploratory_review_score_floor": float(EXPLORATORY_REVIEW_SCORE_FLOOR),
            "coverage_etilt_action_score": COVERAGE_ETILT_ACTION_SCORE,
            "overlap_etilt_action_score": OVERLAP_ETILT_ACTION_SCORE,
            "azimuth_action_score": AZIMUTH_ACTION_SCORE,
            "tx_power_action_score": TX_POWER_ACTION_SCORE,
        },
        "model_explanation": {
            "what_this_test_is_doing": (
                "This test harness identifies bad KPI samples, attributes them to serving cells, rolls them up to bad-serving sites, "
                "uses saved baseline topology to build bounded coordinated multi-site search spaces, lets one Optuna engine search those spaces, "
                "reruns localized optimized prediction for each candidate, and selects the best tradeoff-positive safe candidate."
            ),
            "reused_source_functions": [
                "tools.lte_prediction_optimised.ml_engine.fetch_baseline",
                "tools.lte_prediction_optimised.ml_engine.fetch_site_data",
                "tools.lte_prediction_optimised.ml_engine.fetch_geo_features",
                "tools.lte_tilt_recommandation.etilt_optimizer_cd2.filter_bad_samples",
                "tools.lte_tilt_recommandation.etilt_optimizer_cd2.detect_swap_sector",
                "tools.lte_tilt_recommandation.etilt_optimizer_cd2.build_forecast",
                "tools.lte_tilt_recommandation.etilt_optimizer_cd2.export_to_excel",
                "tools.lte_prediction_optimised.ml_engine.run_prediction_only_optimized",
            ],
            "decision_flow": [
                "Load current-state baseline prediction rows from the DB.",
                "Load current site and antenna settings from site_prediction.",
                "Load saved baseline geo features from lte_prediction_geo_features.",
                "Filter bad KPI samples using the configured RSRP/RSRQ/SINR thresholds.",
                "Attach geo context only to the bad-sample rows.",
                "Aggregate bad-sample geo context per cell.",
                "Rank serving sites by bad-sample contribution.",
                "Classify each site with saved baseline topology columns such as best interferer and interference gap.",
                "Use one bounded Optuna optimizer per ranked site cluster, with primary-site plus topology-linked neighbor/interferer action cells.",
                "Run a latency-safe coarse RF screen first; skip Optuna refinement if broad threshold-bounded candidates cannot beat HOLD.",
                "Re-run localized optimized prediction for each Optuna candidate.",
                "Rank candidates with tradeoff-aware KPI scoring and severe-failure hard constraints.",
                "Estimate forecast after validation with the existing heuristic export function.",
            ],
            "what_the_model_is_not_doing_yet": [
                "This is still a test harness, not a production deployment loop.",
                "It validates one interference cluster at a time rather than doing a full network-wide global optimizer pass.",
                "The forecast highlights remain heuristic export estimates, even though candidate selection now uses localized optimized prediction.",
            ],
        },
        "counts": {
            "baseline_rows": int(len(log_df)),
            "antenna_rows": int(len(antenna_df)),
            "geo_rows": int(len(geo_df)),
            "bad_samples": int(len(bad_samples_df)),
            "bad_cells": int(summary_df["Cell ID"].nunique()) if not summary_df.empty else 0,
            "bearing_cells": int(bearing_summary["Cell ID"].nunique()) if not bearing_summary.empty else 0,
            "geo_context_cells": int(geo_cell_summary["Cell ID"].nunique()) if not geo_cell_summary.empty else 0,
            "recommendation_rows_all": int(len(recommendations_all_df)),
            "recommendation_rows_actionable": int(len(changed_rows)),
            "recommendation_rows_exported": int(len(recommendations_df)),
            "forecast_rows": int(len(forecast_df)),
            "swap_sector_yes": swap_yes,
            "candidate_validation_rows": int(len(candidate_validation_df)),
        },
        "recommendation_summary": {
            "status_counts": status_counts,
            "root_cause_counts": root_cause_counts,
            "confidence_counts": confidence_counts,
            "parameter_change_counts": parameter_change_counts,
            "top_parameter_changes": top_parameter_changes,
        },
        "candidate_validation_summary": candidate_validation_summary,
        "geo_context_overview": geo_context_overview,
        "top_bad_cells": top_bad_cells,
        "forecast_highlights": forecast_highlights,
        "artifacts": {
            "baseline_log_input": str(run_dir / "baseline_log_input.csv"),
            "antenna_input": str(run_dir / "antenna_input.csv"),
            "geo_features_input": str(run_dir / "geo_features_input.csv"),
            "bad_samples": str(run_dir / "bad_samples.csv"),
            "bad_samples_with_geo": str(run_dir / "bad_samples_with_geo.csv"),
            "bad_summary": str(run_dir / "bad_summary.csv"),
            "bad_geo_cell_summary": str(run_dir / "bad_geo_cell_summary.csv"),
            "dominant_bearing_summary": str(run_dir / "dominant_bearing_summary.csv"),
            "recommendations_all": str(run_dir / "recommendations_all.csv"),
            "recommendations": str(run_dir / "recommendations.csv"),
            "site_candidate_evaluations": str(run_dir / "site_candidate_evaluations.csv"),
            "optuna_candidate_validation_results": str(run_dir / "optuna_candidate_validation_results.csv"),
            "candidate_validation_results": str(run_dir / "candidate_validation_results.csv"),
            "forecast": str(run_dir / "forecast.csv"),
            "excel_report": str(excel_path),
        },
        "total_runtime_sec": round(float(total_runtime_sec), 4),
    }


def run_tilt_recommendation_test(config: TiltRecommendationTestConfig) -> Path:
    start = time.perf_counter()
    run_dir = _ensure_dir(config.output_root / f"project_{config.project_id}" / f"tilt_{_timestamp()}")

    TILT_SRC.RSRP_THRESH = float(config.rsrp_threshold)
    TILT_SRC.RSRQ_THRESH = float(config.rsrq_threshold)
    TILT_SRC.SINR_THRESH = float(config.sinr_threshold)

    print(
        f"[TILT_TEST][START] project_id={config.project_id} region={config.region} "
        f"operator={config.operator} rsrp={config.rsrp_threshold} "
        f"rsrq={config.rsrq_threshold} sinr={config.sinr_threshold}"
    )
    threshold_path = _resolve_threshold_file_path_test_only(config.threshold_file_path)
    constraint_df = _load_threshold_constraints_test_only(threshold_path) if threshold_path else pd.DataFrame()
    constraint_map = _constraint_map_test_only(constraint_df)
    config.threshold_file_path = threshold_path or None
    config.threshold_constraint_count = int(len(constraint_df))
    config.threshold_optimised_count = int(constraint_df["optimised"].sum()) if "optimised" in constraint_df.columns else 0
    if threshold_path:
        print(
            f"[TILT_TEST][THRESHOLD_CONSTRAINTS] path={threshold_path} "
            f"rows={config.threshold_constraint_count} optimised_rows={config.threshold_optimised_count}"
        )
    else:
        print("[TILT_TEST][THRESHOLD_CONSTRAINTS] path=n/a rows=0 optimised_rows=0")

    log_df = _fetch_baseline_log_df(config.project_id, config.region, config.operator)
    antenna_df = _fetch_antenna_df(config.project_id, config.region, config.operator)
    log_df = _enrich_log_with_antenna_context(log_df, antenna_df)
    geo_df = _fetch_geo_df(config.project_id, config.region, config.operator, antenna_df)

    bad_samples_df, summary_df = TILT_SRC.filter_bad_samples(log_df.copy(), TILT_SRC.ALLOWED_TECHS)
    swap_dict = TILT_SRC.detect_swap_sector(log_df.copy(), antenna_df.copy())
    bearing_summary = _compute_dominant_bearing_summary(log_df, antenna_df)
    bad_geo_df = _attach_geo_to_bad_samples(bad_samples_df, geo_df)
    geo_cell_summary = _aggregate_bad_geo_context(bad_geo_df)
    recommendations_all_df, site_candidate_evaluation_df = _build_kpi_first_recommendations_test_only(
        summary_df,
        antenna_df,
        geo_cell_summary,
        bearing_summary,
        swap_dict,
        log_df,
        config,
        run_dir,
        constraint_map,
    )
    # The upstream stage is now the single Optuna optimizer; keep this artifact
    # empty so the report no longer represents a second optimization pass.
    optuna_validation_df = pd.DataFrame()
    candidate_validation_df = pd.concat(
        [site_candidate_evaluation_df, optuna_validation_df],
        ignore_index=True,
        sort=False,
    ) if (not site_candidate_evaluation_df.empty or not optuna_validation_df.empty) else pd.DataFrame()
    recommendations_all_df, recommendations_df = _prepare_recommendation_exports(recommendations_all_df)
    forecast_df = TILT_SRC.build_forecast(summary_df, recommendations_all_df)

    excel_path = run_dir / "RF_Optimization_Report_Test.xlsx"
    TILT_SRC.export_to_excel(
        summary_df=summary_df,
        recommendations_df=recommendations_df,
        forecast_df=forecast_df,
        bad_samples_df=bad_geo_df,
        output_path=str(excel_path),
    )

    log_df.to_csv(run_dir / "baseline_log_input.csv", index=False)
    antenna_df.to_csv(run_dir / "antenna_input.csv", index=False)
    geo_df.to_csv(run_dir / "geo_features_input.csv", index=False)
    bad_samples_df.to_csv(run_dir / "bad_samples.csv", index=False)
    bad_geo_df.to_csv(run_dir / "bad_samples_with_geo.csv", index=False)
    summary_df.to_csv(run_dir / "bad_summary.csv", index=False)
    geo_cell_summary.to_csv(run_dir / "bad_geo_cell_summary.csv", index=False)
    bearing_summary.to_csv(run_dir / "dominant_bearing_summary.csv", index=False)
    recommendations_all_df.to_csv(run_dir / "recommendations_all.csv", index=False)
    recommendations_df.to_csv(run_dir / "recommendations.csv", index=False)
    site_candidate_evaluation_df.to_csv(run_dir / "site_candidate_evaluations.csv", index=False)
    optuna_validation_df.to_csv(run_dir / "optuna_candidate_validation_results.csv", index=False)
    candidate_validation_df.to_csv(run_dir / "candidate_validation_results.csv", index=False)
    forecast_df.to_csv(run_dir / "forecast.csv", index=False)

    total_runtime_sec = time.perf_counter() - start
    summary_payload = _summarize_recommendation_run(
        config=config,
        log_df=log_df,
        antenna_df=antenna_df,
        geo_df=geo_df,
        bad_samples_df=bad_samples_df,
        summary_df=summary_df,
        geo_cell_summary=geo_cell_summary,
        bearing_summary=bearing_summary,
        recommendations_all_df=recommendations_all_df,
        recommendations_df=recommendations_df,
        candidate_validation_df=candidate_validation_df,
        forecast_df=forecast_df,
        swap_dict=swap_dict,
        run_dir=run_dir,
        excel_path=excel_path,
        total_runtime_sec=total_runtime_sec,
    )
    _write_json(run_dir / "summary.json", summary_payload)
    print(f"[TILT_TEST][DONE] run_dir={run_dir} total_runtime_sec={summary_payload['total_runtime_sec']}")
    return run_dir


def _parse_args() -> TiltRecommendationTestConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--rsrp", type=float, default=-105.0)
    parser.add_argument("--rsrq", type=float, default=-15.0)
    parser.add_argument("--sinr", type=float, default=0.0)
    parser.add_argument("--validate-candidates", action="store_true", default=True)
    parser.add_argument("--no-validate-candidates", action="store_false", dest="validate_candidates")
    parser.add_argument("--radius-m", type=float, default=500.0)
    parser.add_argument("--grid-resolution-m", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--impact-radius-m", type=float, default=800.0)
    parser.add_argument("--neighbor-site-count", type=int, default=3)
    parser.add_argument("--max-interference-sites", type=int, default=10)
    parser.add_argument("--max-good-area-loss-pct", type=float, default=2.0)
    parser.add_argument("--max-mean-sinr-drop-db", type=float, default=1.0)
    parser.add_argument("--min-score-gain", type=float, default=0.0)
    parser.add_argument("--min-recovered-bad-samples", type=int, default=3)
    parser.add_argument("--max-ranked-cells", type=int, default=20)
    parser.add_argument("--max-ranked-sites", type=int, default=8)
    parser.add_argument("--threshold-file", "--threshold-file-path", dest="threshold_file_path", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    return TiltRecommendationTestConfig(
        project_id=args.project_id,
        region=args.region,
        operator=args.operator,
        rsrp_threshold=args.rsrp,
        rsrq_threshold=args.rsrq,
        sinr_threshold=args.sinr,
        validate_candidates=args.validate_candidates,
        radius_m=args.radius_m,
        grid_resolution_m=args.grid_resolution_m,
        workers=args.workers,
        impact_radius_m=args.impact_radius_m,
        neighbor_site_count=args.neighbor_site_count,
        max_interference_sites=args.max_interference_sites,
        max_good_area_loss_pct=args.max_good_area_loss_pct,
        max_mean_sinr_drop_db=args.max_mean_sinr_drop_db,
        min_score_gain=args.min_score_gain,
        min_recovered_bad_samples=args.min_recovered_bad_samples,
        max_ranked_cells=args.max_ranked_cells,
        max_ranked_sites=args.max_ranked_sites,
        threshold_file_path=args.threshold_file_path,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    run_dir = run_tilt_recommendation_test(_parse_args())
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))

