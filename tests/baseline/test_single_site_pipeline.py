"""
Site -> sector, stage-by-stage pipeline trace for debugging prediction quality.

Runs the REAL production functions (imported directly from
tools.lte_prediction.ml_engine / geo_correction_pipeline - not a
reimplementation) for one project, then organizes every physical site's
sectors (site + sector + cell + band, matching how production actually
calculates - per site+sector+cell+band, not per whole site) so a dashboard
can pick a site, then one of its sectors, and inspect every stage the value
passes through:

  1. COST-231 raw          (run_rf_prediction_fast output, untouched)
  2. Geo-corrected          (apply_experimental_geo_adjustments output,
                             captured BEFORE DT calibration overwrites it -
                             the production code never persists this value
                             on its own, so it must be snapshotted here)
  3. DT-calibrated          (preserve_calibrated_kpis output)
  4. Smoothed / DT-blended  (apply_demo_dt_overlay output - what actually
                             gets saved as pred_rsrp_smoothed in production)

Each sector also carries its azimuth and the antenna's 3dB half-beamwidth
(65 degrees, the same constant COST-231's own antenna gain formula uses -
see grid_sampling.py:_antenna_gain_estimate) so a dashboard can draw the
true directional beam instead of treating the full 500m circular grid as
uniform coverage.

Writes one JSON file consumable by the companion Streamlit dashboard.

Usage:
    python tests/baseline/test_single_site_pipeline.py --project-id 210 --region taiwan
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads as load_wkt
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pyproj import Transformer
from shapely.geometry import LineString as _LineString, Point as _Point

load_dotenv(PROJECT_ROOT / ".env")

from tools.lte_prediction.ml_engine import (
    engine,
    fetch_site_data,
    fetch_drive_data,
    fetch_building_data,
    run_rf_prediction_fast,
    _resolve_prediction_polygons,
)
from tools.lte_prediction.geo_correction_pipeline import (
    normalize_site_for_geo,
    load_geo_weights,
    align_project_polygon_to_points,
    building_df_to_gdf,
    align_building_geometries_to_project,
    _enrich_buildings_with_osm_heights,
    _osm_enrichment_enabled,
    create_analysis_grid,
    attach_building_features,
    _attach_osm_context_features,
    build_grid_feature_frame,
    augment_grid_with_advanced_geo_features,
    assign_points_to_tiles,
    _attach_missing_grid_features_by_grid_id,
    attach_fixed_serving_sinr_rsrq_proxy,
    apply_experimental_geo_adjustments,
    split_drive_train_holdout,
    evaluate_geo_against_dt,
    fit_dt_holdout_calibration,
    apply_dt_holdout_calibration,
    preserve_calibrated_kpis,
    apply_demo_dt_overlay,
    _choose_utm_crs,
    DEFAULT_TILE_SIZE_M,
)

ANTENNA_3DB_BEAMWIDTH_DEG = 65.0  # matches grid_sampling.py::_antenna_gain_estimate


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2r - lon1r
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def off_axis_deg(bearing, azimuth):
    return np.abs((bearing - azimuth + 180.0) % 360.0 - 180.0)


def compute_3gpp_antenna_gain_unclipped(az_diff, elev_diff, max_gain=18.0,
                                         h_beamwidth=65.0, v_beamwidth=6.0,
                                         a_max=30.0, sla_v=20.0):
    """Exact copy of Sector_wise_prediction_code_copy.py::compute_3gpp_antenna_gain_vectorized -
    same real formula/defaults production uses. Reproduced here (not imported) only because the
    real function's caller clips its OUTPUT before returning it; this lets us see the real
    pre-clip RSRP for diagnosis. Test-case only - tools/lte_prediction/ is not touched."""
    ah = np.minimum(12.0 * (az_diff / h_beamwidth) ** 2, a_max)
    av = np.minimum(12.0 * (elev_diff / v_beamwidth) ** 2, sla_v)
    total_attenuation = np.minimum(ah + av, a_max)
    return max_gain - total_attenuation


def detect_indoor_loss(p_lat, p_lon, building_sindex, building_geoms):
    """Exact copy of Sector_wise_prediction_code_copy.py::detect_indoor's real logic:
    flat 15.0dB loss (production's own hardcoded constant, confirmed real - see
    load_building_polygons's meta.append({"loss": 15.0, ...})) for any point that falls
    INSIDE a real building polygon, 0 otherwise. Test-case only - not imported because
    production's version reads from a shared-memory worker-pool global, not a plain arg."""
    from shapely.geometry import Point
    pt = Point(p_lon, p_lat)
    candidate_idx = list(building_sindex.query(pt))
    for idx in candidate_idx:
        if building_geoms[idx].contains(pt):
            return 15.0
    return 0.0


def compute_sector_rsrp_unclipped(s_lat, s_lon, s_az, s_etilt, s_mtilt, s_htx, tx_pwr, freq,
                                   p_lat, p_lon, h_rx=1.5, antenna_gain=18.0, cable_loss=2.0,
                                   indoor_loss=0.0):
    """Exact copy of Sector_wise_prediction_code_copy.py::compute_sector_rsrp's real math
    (COST-231-Hata path loss + 3GPP antenna gain) PLUS the real indoor/building loss
    subtracted by process_chunk_3gpp_antenna right after calling compute_sector_rsrp
    (sec_rsrp = compute_sector_rsrp(...) - iloss) - confirmed real, not something invented:
    a flat 15.0dB penalty for any point inside a real building polygon. Vectorized over
    point arrays, WITHOUT the np.clip(-140,-44) production applies right after
    (Sector_wise_prediction_code_copy.py line 939). Test-case only."""
    d_m = np.maximum(haversine_m(s_lat, s_lon, p_lat, p_lon), 1.0)
    d_km = d_m / 1000.0

    a_hm = (1.1 * np.log10(freq) - 0.7) * h_rx - (1.56 * np.log10(freq) - 0.8)
    CM = 3.0
    base_PL = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(s_htx) - a_hm + CM
    slope_term = 44.9 - 6.55 * np.log10(s_htx)
    pathloss = base_PL + slope_term * np.log10(d_km)

    bearing = bearing_deg(s_lat, s_lon, p_lat, p_lon)
    az_diff = (bearing - s_az + 180.0) % 360.0 - 180.0
    elev_angle = np.degrees(np.arctan2(h_rx - s_htx, d_m))
    elev_diff = elev_angle + s_etilt + s_mtilt
    gain_3gpp = compute_3gpp_antenna_gain_unclipped(az_diff, elev_diff, antenna_gain)

    return tx_pwr + gain_3gpp - pathloss - cable_loss - indoor_loss


def attach_pred_cols_to_dt_points(dt_df: pd.DataFrame, pred_df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Test-case-only nearest-neighbour spatial join of real DT measurement points to the
    prediction grid, carrying an arbitrary set of columns. Same method (sjoin_nearest in a
    local UTM CRS, via the real production _choose_utm_crs helper) as
    geo_correction_pipeline._attach_prediction_grid_to_points, which is NOT reused directly
    because its own `keep_cols` list is hardcoded and doesn't carry the new
    stage1_raw_rsrp_unclipped / stage2b_spm_rsrp columns this test case adds. Production
    itself is not touched."""
    points = dt_df.copy()
    if points.empty or pred_df.empty or not {"lat", "lon"}.issubset(pred_df.columns):
        return points
    keep_cols = ["lat", "lon"] + [c for c in cols if c in pred_df.columns]
    preds = pred_df[keep_cols].dropna(subset=["lat", "lon"]).copy()
    if preds.empty:
        return points
    points_gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
    preds_gdf = gpd.GeoDataFrame(preds, geometry=gpd.points_from_xy(preds["lon"], preds["lat"]), crs="EPSG:4326")
    preds_gdf = preds_gdf.rename(columns={"lat": "grid_lat", "lon": "grid_lon"})
    utm_crs = _choose_utm_crs(preds_gdf)
    joined = gpd.sjoin_nearest(
        points_gdf.to_crs(utm_crs), preds_gdf.to_crs(utm_crs), how="left", distance_col="grid_match_distance_m",
    )
    joined = joined.to_crs("EPSG:4326").drop(columns=["geometry", "index_right"], errors="ignore")
    return pd.DataFrame(joined)


def fit_spm_style_clutter_offsets(dt_matched: pd.DataFrame, min_samples_per_class: int = 15,
                                   max_match_distance_m: float = 150.0,
                                   baseline_col: str = "stage1_raw_rsrp_unclipped",
                                   k1_clip_db: tuple = (0.0, 25.0),
                                   cal_clip_db: tuple = (-10.0, 10.0)):
    """SPM/Atoll-style Stage 2 replacement, now correctly split into the real SPM's TWO
    separate terms instead of one blended one:

      K1  - the real SPM constant-offset term. Confirmed via a real published SPM
            calibration case study (Purwokerto, 1800MHz): K1=22dB there - remarkably close to
            this project's own clean, non-imputed 4G Band-3 CDF evidence (~19.4dB gap). K1
            absorbs the broad, systematic Stage-1-vs-real gap and is applied EQUALLY to every
            point regardless of clutter class, so it cannot distort real antenna/distance
            differentiation (adding the same constant everywhere preserves every real relative
            difference exactly).

      CAL - the real SPM "Clutter Absorption Loss" term (K_clutter*f(clutter)) - a SMALL,
            genuine per-class LOSS differential relative to the class average, capped to a
            physically real range (default +/-10dB). This is deliberately NOT the same thing
            as the old single blended offset: previously this function returned one number per
            class that had to absorb BOTH the broad systematic bias AND the clutter-specific
            difference, which is exactly why a "loss" term came out as a large net GAIN (up to
            +62dB) - a real loss/absorption term should never do that. Splitting K1 out fixes
            it structurally, not by capping around the symptom.

    The returned `offsets` dict is still `clutter_class -> total additive dB` (K1 + CAL) for
    drop-in compatibility with apply_spm_style_stage2/apply_spm_style_stage2c, which just add
    it to physics - only the internal computation changed.

    max_match_distance_m excludes DT points whose nearest prediction grid point is farther
    than this - relevant because this trace can be scoped to a single site (only_sites), so a
    DT point elsewhere in the project could otherwise nearest-match a grid point far outside
    any real serving distance for that site."""
    work = dt_matched.copy()
    if "grid_match_distance_m" in work.columns:
        before_n = len(work)
        work = work[pd.to_numeric(work["grid_match_distance_m"], errors="coerce") <= max_match_distance_m]
        dropped_far = before_n - len(work)
    else:
        dropped_far = 0
    work = work.dropna(subset=["RSRP_meas", baseline_col, "clutter_class"])
    work["residual_db"] = pd.to_numeric(work["RSRP_meas"], errors="coerce") - pd.to_numeric(work[baseline_col], errors="coerce")
    global_mean_raw = float(work["residual_db"].mean()) if len(work) else 0.0
    k1_constant = float(np.clip(global_mean_raw, k1_clip_db[0], k1_clip_db[1]))

    counts = work.groupby("clutter_class").size()
    group_means = work.groupby("clutter_class")["residual_db"].mean()

    offsets: dict = {}
    debug: dict = {
        "method": "spm_k1_plus_cal_least_squares",
        "k1_constant_db": round(k1_constant, 3),
        "k1_clip_db": list(k1_clip_db),
        "cal_clip_db": list(cal_clip_db),
        "global_mean_residual_db_unclipped": round(global_mean_raw, 3),
        "matched_dt_rows_total": int(len(dt_matched)),
        "dropped_beyond_max_match_distance_m": int(dropped_far),
        "matched_dt_rows_used": int(len(work)),
        "per_class": {},
    }
    all_classes = set(group_means.index) | set(counts.index)
    for cls in sorted(all_classes):
        n = int(counts.get(cls, 0))
        if n >= min_samples_per_class:
            raw_class_mean = float(group_means[cls])
            raw_cal = raw_class_mean - global_mean_raw
            clipped_cal = float(np.clip(raw_cal, cal_clip_db[0], cal_clip_db[1]))
            total = k1_constant + clipped_cal
            offsets[cls] = total
            debug["per_class"][cls] = {
                "n": n, "k1_db": round(k1_constant, 3), "cal_db": round(clipped_cal, 3),
                "total_offset_db": round(total, 3), "raw_class_mean_residual_db": round(raw_class_mean, 3),
            }
        else:
            offsets[cls] = k1_constant
            debug["per_class"][cls] = {
                "n": n, "k1_db": round(k1_constant, 3), "cal_db": 0.0, "total_offset_db": round(k1_constant, 3),
                "note": f"fallback_cal_zero (n<{min_samples_per_class})",
            }
    return offsets, debug


def apply_spm_style_stage2(pred_df: pd.DataFrame, clutter_offsets: dict, global_mean_residual: float) -> pd.DataFrame:
    """physics (stage1_raw_rsrp_unclipped) + ONE calibrated per-clutter-class term - the SPM
    structure - producing stage2b_spm_rsrp_unclipped/stage2b_spm_rsrp. The -140/-44 clip is
    applied ONCE here as a final display step, never fed back into any further calculation -
    this is the "coverage threshold, not a loss mechanism" principle already established for
    Stage 1."""
    out = pred_df.copy()
    offset_series = out["clutter_class"].astype(str).map(clutter_offsets)
    offset_series = offset_series.fillna(float(global_mean_residual))
    out["stage2b_spm_rsrp_unclipped"] = pd.to_numeric(out["stage1_raw_rsrp_unclipped"], errors="coerce") + offset_series
    out["stage2b_spm_rsrp"] = out["stage2b_spm_rsrp_unclipped"].clip(-140, -44)
    return out


CONSERVATIVE_MIN_BUILDING_HEIGHT_M = 9.0  # ~3 real storeys - a real building of genuinely
# unknown height should not be treated as if it barely obstructs (confirmed via real
# published clutter-modelling guidance this session: missing-data assumptions should be
# conservative, not minimal, since underestimating loss overstates coverage).


MAX_SINGLE_KNIFE_EDGE_LOSS_DB = 25.0  # real single-obstruction diffraction loss in typical macro-
# cell geometries rarely exceeds this in practice - a first unclipped run produced outliers up
# to 60dB, which turned out to inflate the downstream clutter fit rather than help it (confirmed:
# capping this is what's needed, not the formula itself, which is real ITU-R physics).


def compute_diffraction_loss_knife_edge(s_lat: float, s_lon: float, s_htx: float, freq_mhz: float,
                                         p_lats: np.ndarray, p_lons: np.ndarray,
                                         building_sindex, building_geoms: list, building_heights: np.ndarray,
                                         utm_transformer: Transformer, h_rx: float = 1.5) -> np.ndarray:
    """Real single-dominant-knife-edge diffraction loss (ITU-R P.526-style Fresnel-Kirchhoff
    formula), computed from REAL per-path building intersections (same real-geometry method
    production's own _attach_building_path_features already uses - a real LineString from
    site to point, checked against real building polygons) and REAL (GHS-OBAT-imputed)
    building heights.

    This replaces production's diffraction_proxy_db, which converts the same real per-path
    data into dB using small hand-picked LINEAR coefficients
    (1.4*count + 9*blocked_ratio + 0.04*height) - not real diffraction physics. A single real
    building genuinely crossing the path contributes ~1.4dB there; real diffraction physics
    (below) can genuinely contribute 10-20dB+ for one significant obstruction, matching real
    RF engineering behaviour instead of a token, near-flat correction.

    For each point: finds every real building crossing the real site->point path, computes
    each one's real height-above-the-direct-line, keeps the single most-obstructing one (the
    dominant knife edge - a standard, real simplification of full multi-edge diffraction), and
    applies the real ITU-R J(v) diffraction-loss formula. Unknown/imputed building heights are
    floored at CONSERVATIVE_MIN_BUILDING_HEIGHT_M rather than allowed to sit near zero.

    Test-case only - not wired into tools/lte_prediction/."""
    n = len(p_lats)
    losses = np.zeros(n, dtype=float)
    if n == 0:
        return losses
    site_x, site_y = utm_transformer.transform(s_lon, s_lat)
    site_pt = _Point(site_x, site_y)
    wavelength_m = 299792458.0 / (float(freq_mhz) * 1e6)

    for i in range(n):
        point_x, point_y = utm_transformer.transform(float(p_lons[i]), float(p_lats[i]))
        total_d = float(np.hypot(point_x - site_x, point_y - site_y))
        if total_d <= 1.0:
            continue
        path = _LineString([(site_x, site_y), (point_x, point_y)])
        candidate_idx = list(building_sindex.query(path))
        if not candidate_idx:
            continue
        best_v = None
        for idx in candidate_idx:
            geom = building_geoms[idx]
            if not geom.intersects(path):
                continue
            inter = geom.intersection(path)
            if inter.is_empty:
                continue
            mid = inter.centroid
            d1 = site_pt.distance(mid)
            d2 = total_d - d1
            if d1 <= 0.5 or d2 <= 0.5:
                continue
            los_height_at_d1 = s_htx + (h_rx - s_htx) * (d1 / total_d)
            bld_h = building_heights[idx] if idx < len(building_heights) else np.nan
            if bld_h is None or (isinstance(bld_h, float) and np.isnan(bld_h)):
                bld_h = CONSERVATIVE_MIN_BUILDING_HEIGHT_M
            else:
                bld_h = max(float(bld_h), CONSERVATIVE_MIN_BUILDING_HEIGHT_M)
            h_above_los = bld_h - los_height_at_d1
            if h_above_los <= 0:
                continue
            v = h_above_los * np.sqrt(2.0 * total_d / (wavelength_m * d1 * d2))
            if best_v is None or v > best_v:
                best_v = v
        if best_v is None or best_v <= -0.78:
            continue
        j = 6.9 + 20.0 * np.log10(np.sqrt((best_v - 0.1) ** 2 + 1.0) + best_v - 0.1)
        losses[i] = min(max(0.0, j), MAX_SINGLE_KNIFE_EDGE_LOSS_DB)
    return losses


def apply_spm_style_stage2c(pred_df: pd.DataFrame, diffraction_loss_col: str,
                             clutter_offsets: dict, global_mean_residual: float) -> pd.DataFrame:
    """Stage 2c: physics + real diffraction (physics formula, computed directly - not DT-fit)
    + clutter offset (DT-calibrated, fit AFTER diffraction is subtracted, so the clutter term
    only has to explain what real diffraction physics doesn't already account for, instead of
    a single clutter constant being asked to absorb both effects)."""
    out = pred_df.copy()
    stage1_plus_diffraction = pd.to_numeric(out["stage1_raw_rsrp_unclipped"], errors="coerce") - pd.to_numeric(out[diffraction_loss_col], errors="coerce")
    out["stage1_plus_diffraction_unclipped"] = stage1_plus_diffraction
    offset_series = out["clutter_class"].astype(str).map(clutter_offsets)
    offset_series = offset_series.fillna(float(global_mean_residual))
    out["stage2c_spm_rsrp_unclipped"] = stage1_plus_diffraction + offset_series
    out["stage2c_spm_rsrp"] = out["stage2c_spm_rsrp_unclipped"].clip(-140, -44)
    return out


# Real production's apply_dt_holdout_calibration overwrites pred_rsrp_geo
# IN PLACE with the calibrated value (confirmed by reading
# geo_correction_pipeline.py directly: out[pred_col] = base + residual,
# where pred_col == "pred_rsrp_geo" for the RSRP model). So after the real
# Stage 3 runs, pred_rsrp_geo no longer holds Stage 2's original hand-tuned
# value - it holds Stage 3's output. This test file already snapshots the
# true, stable Stage 2 value into stage2_geo_rsrp right after Stage 2 runs
# (before that mutation happens) - use THAT everywhere a stable Stage-2
# reference is needed (features, MAE comparisons), never the mutable
# pred_rsrp_geo, to avoid silently comparing/training against a value that
# has already been overwritten by something else.
STAGE3_FEATURE_COLS = [
    "pred_rsrp", "pred_rsrq", "pred_sinr", "stage2_geo_rsrp", "pred_rsrq_geo", "pred_sinr_geo",
    "morphology_cluster", "building_count", "building_area_ratio", "avg_building_area_m2",
    "road_length_m", "green_ratio", "water_ratio", "los_blocker_count", "los_blocked_ratio",
    "max_blocker_height_m", "diffraction_proxy_db", "nlos_flag", "terrain_elevation_m",
    "terrain_slope_deg", "terrain_relief_to_site_m", "site_count_250m", "site_count_500m",
    "serving_distance_m", "nearest_site_distance_m", "mean_nearest3_site_distance_m",
    "azimuth_delta_deg", "serving_proxy_rsrp_phys_dbm", "rsrq_proxy_db", "sinr_proxy_db",
    "interference_gap_db", "interference_ratio_linear",
]


def build_dt_calibration_features_no_clutter(df: pd.DataFrame) -> pd.DataFrame:
    """Test-case replica of geo_correction_pipeline._build_dt_calibration_feature_frame,
    MINUS the clutter_class one-hot dummies. Stage 2b already calibrates a per-clutter-
    class term against real DT data; keeping clutter_class as a Stage-3 input feature too
    would let that same signal get a second, less-controlled adjustment through Ridge
    regression on top of an already-corrected baseline - the "duplicate DT training"
    overlap risk identified this session. Every other real feature is unchanged (using
    stage2_geo_rsrp, the stable snapshot, instead of production's mutable pred_rsrp_geo)."""
    numeric = pd.DataFrame(index=df.index)
    for col in STAGE3_FEATURE_COLS:
        if col in df.columns:
            numeric[col] = pd.to_numeric(df[col], errors="coerce")
    return numeric.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def fit_stage3_rewired(train_eval: pd.DataFrame, pred_col: str = "stage2b_spm_rsrp"):
    """Test-case replica of geo_correction_pipeline.fit_dt_holdout_calibration, rewired
    to fit the residual against Stage 2b's output (pred_col) instead of production's
    hardcoded pred_rsrp_geo (the old hand-tuned Stage 2). Real production's function
    can't be reused directly since its pred_col is hardcoded inside it. This is the
    "delta correction on top of an already-calibrated baseline" pattern confirmed via
    real published research this session (calibrate the propagation model once, then
    fit a separate residual model against ITS output, not the uncalibrated one)."""
    debug = {"enabled": False, "train_rows": int(len(train_eval)), "pred_col": pred_col}
    if pred_col not in train_eval.columns:
        debug["reason"] = f"pred_col '{pred_col}' not present"
        return None, debug
    valid = train_eval.dropna(subset=["RSRP_meas", pred_col]).copy()
    if len(valid) < 60:
        debug["reason"] = f"train_rows_lt_60 (got {len(valid)})"
        return None, debug
    features = build_dt_calibration_features_no_clutter(valid)
    if features.empty or features.shape[1] == 0:
        debug["reason"] = "no_features"
        return None, debug
    residual = (
        pd.to_numeric(valid["RSRP_meas"], errors="coerce") - pd.to_numeric(valid[pred_col], errors="coerce")
    ).clip(-12.0, 12.0)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)
    ridge = Ridge(alpha=3.0, random_state=42)
    ridge.fit(x_scaled, residual.to_numpy(dtype=float))
    bundle = {
        "pred_col": pred_col, "scaler": scaler, "model": ridge,
        "feature_columns": features.columns.tolist(), "low_clip": -12.0, "high_clip": 12.0,
    }
    debug.update({
        "enabled": True, "rows": int(len(valid)), "feature_count": int(features.shape[1]),
        "residual_mae": round(float(residual.abs().mean()), 4), "residual_bias": round(float(residual.mean()), 4),
    })
    return bundle, debug


RSRP_BIN_EDGES = [-140, -105, -95, -85, -75, -44]
RSRP_BIN_LABELS = ["-140 to -105", "-105 to -95", "-95 to -85", "-85 to -75", "-75 to -44"]


def mae_rmse(df: pd.DataFrame, col: str, meas_col: str = "RSRP_meas"):
    """Module-level (was a nested closure) so both the single-site trace and
    the project-wide trace can share the exact same metric definition."""
    v = df.dropna(subset=[col, meas_col]) if col in df.columns else pd.DataFrame()
    if v.empty:
        return None, None
    err = pd.to_numeric(v[meas_col], errors="coerce") - pd.to_numeric(v[col], errors="coerce")
    return round(float(err.abs().mean()), 3), round(float(np.sqrt((err ** 2).mean())), 3)


def binned_mae_rmse(df: pd.DataFrame, col: str, meas_col: str = "RSRP_meas") -> dict:
    """MAE/RMSE broken down by the REAL measured RSRP's signal-strength bin
    (not the predicted value) - shows whether a stage's accuracy is uniform
    across signal regimes or concentrated/degraded in a specific range."""
    if col not in df.columns or meas_col not in df.columns:
        return {}
    v = df.dropna(subset=[col, meas_col]).copy()
    if v.empty:
        return {}
    v["_bin"] = pd.cut(
        pd.to_numeric(v[meas_col], errors="coerce"), bins=RSRP_BIN_EDGES,
        labels=RSRP_BIN_LABELS, include_lowest=True,
    )
    out = {}
    for label in RSRP_BIN_LABELS:
        sub = v[v["_bin"] == label]
        if sub.empty:
            out[label] = {"n": 0, "mae": None, "rmse": None}
            continue
        err = pd.to_numeric(sub[meas_col], errors="coerce") - pd.to_numeric(sub[col], errors="coerce")
        out[label] = {
            "n": int(len(sub)),
            "mae": round(float(err.abs().mean()), 3),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        }
    return out


def apply_stage3_rewired(pred_df: pd.DataFrame, bundle) -> pd.Series:
    """Test-case replica of geo_correction_pipeline.apply_dt_holdout_calibration for the
    rewired Stage-3-on-Stage-2b bundle above."""
    fallback = pd.to_numeric(pred_df.get("stage2b_spm_rsrp"), errors="coerce").clip(-140.0, -44.0)
    if bundle is None:
        return fallback
    pred_col = bundle["pred_col"]
    if pred_col not in pred_df.columns:
        return fallback
    features = build_dt_calibration_features_no_clutter(pred_df)
    for col in bundle["feature_columns"]:
        if col not in features.columns:
            features[col] = 0.0
    features = features.reindex(columns=bundle["feature_columns"], fill_value=0.0)
    x_scaled = bundle["scaler"].transform(features)
    residual_pred = pd.Series(bundle["model"].predict(x_scaled), index=pred_df.index, dtype=float)
    residual_pred = residual_pred.clip(lower=bundle["low_clip"], upper=bundle["high_clip"])
    base = pd.to_numeric(pred_df[pred_col], errors="coerce")
    return (base + residual_pred).clip(-140.0, -44.0)


def run_trace(project_id: int, region: str, session_ids, operator, radius_m, grid_resolution_m,
              polygon_ids, output_path: Path, only_sites=None, data_dir: Path = None):
    cached_polygon_wkt_lonlat = None
    if data_dir is not None:
        # Load from the local cache built by fetch_and_cache_data.py - no
        # DB/bridge calls at all. Used when the C# bridge (port 5224) isn't
        # running; skips fetch_site_data/fetch_drive_data/fetch_building_data
        # entirely and reads the same CSVs run_baseline_from_cache.py uses.
        print(f"[TRACE] loading site/drive/building data from local cache: {data_dir}")
        site_df = pd.read_csv(data_dir / "site_df.csv", low_memory=False)
        # Prefer the n78-fixed local drive cache (built by
        # fix_drive_cache_n78.py: real DB n78 rows this project's original
        # cache was missing + N/A rows relabeled/imputed to n78 via nearest-
        # real-n78-point distance, explicitly flagged via band_source) over
        # the original drive_df.csv when it exists. Local file only - the
        # original drive_df.csv and the real DB are both untouched.
        drive_cache_path = data_dir / "drive_df_n78_fixed.csv"
        if drive_cache_path.exists():
            drive_df = pd.read_csv(drive_cache_path, low_memory=False)
            print(f"[TRACE] using n78-fixed drive cache ({drive_cache_path.name}): "
                  f"{len(drive_df)} rows, band_source={drive_df['band_source'].value_counts().to_dict()}")
        else:
            drive_df = pd.read_csv(data_dir / "drive_df.csv", low_memory=False)
        building_df = pd.read_csv(data_dir / "building_df.csv", low_memory=False)
        resolved_operator = operator
        # run_rf_prediction_fast clips its raw COST-231 output to whatever
        # polygon it's given (real production behaviour - scopes a real job
        # to the surveyed project area). The real project polygon is a
        # specific, irregular ~2.4x4.6km shape, much smaller/differently
        # shaped than a 2500m-radius circle - passing it here would chop the
        # antenna's true directional pattern into a shape that reflects the
        # SURVEY boundary, not the antenna, which is exactly wrong for a
        # trace whose whole point is to show the real antenna-pattern shape
        # per sector. So for THIS debug trace only, use a generous square
        # bounding box around the site (comfortably containing the full
        # request radius) instead of the tight real polygon - this is a
        # test-case-only substitution, never done for a real prediction job.
        site_lookup_df = pd.read_csv(data_dir / "site_df.csv", low_memory=False)
        if only_sites:
            site_lookup_df = site_lookup_df[site_lookup_df["Site ID"].isin(only_sites)]
        center_lat = float(site_lookup_df["lat"].astype(float).mean())
        center_lon = float(site_lookup_df["lon"].astype(float).mean())
        half_extent_deg_lat = (radius_m * 1.3) / 111320.0
        half_extent_deg_lon = (radius_m * 1.3) / (111320.0 * np.cos(np.radians(center_lat)))
        from shapely.geometry import box as _shp_box
        bbox_poly = _shp_box(
            center_lon - half_extent_deg_lon, center_lat - half_extent_deg_lat,
            center_lon + half_extent_deg_lon, center_lat + half_extent_deg_lat,
        )
        cached_polygon_wkt_lonlat = bbox_poly.wkt
        print(f"[TRACE] using a {radius_m*1.3:.0f}m half-extent bounding box (not the real project polygon) "
              f"so the antenna pattern isn't clipped by the survey boundary shape")
    else:
        print(f"[TRACE] fetching site/drive/building data for project_id={project_id} region={region}")
        site_df, resolved_operator = fetch_site_data(project_id, region=region, operator=operator, polygon_ids=polygon_ids)
        drive_df = fetch_drive_data(session_ids, resolved_operator, project_id, region=region)
        building_df = fetch_building_data(project_id, region=region)
    if only_sites:
        # Restrict the INPUT itself, not just the output - this is what
        # actually saves time, since COST-231's per-point cost scales with
        # both point count (radius^2) and number of sectors. Interference
        # candidates will only be computed from this reduced site set
        # instead of the full project, which is an acceptable trade-off for
        # a single-site/sector debug trace, not something to do for a real
        # baseline run.
        before_n = len(site_df)
        site_df = site_df[site_df["Site ID"].isin(only_sites)].copy()
        print(f"[TRACE] only_sites={sorted(only_sites)} -> site_df {before_n} -> {len(site_df)} rows (input scoped down, not just output)")
    print(f"[TRACE] site_df={len(site_df)} drive_df={len(drive_df)} building_df={len(building_df)}")

    dem_path = None
    candidate_dem = PROJECT_ROOT / "data" / "dem" / f"project_{project_id}_dem.tif"
    if candidate_dem.exists():
        dem_path = str(candidate_dem)

    # These are EXACTLY the defaults routes.py uses for a real /api/lte-prediction/run
    # call (see ML/tools/lte_prediction/routes.py) - no overrides, so this runs the
    # same code path a real production job would, including the frontend-grid-sampling
    # attempt (which may itself fall back to the circular grid if GetGridAnalytics
    # has no data for this project/region - that is production's real behavior, not
    # something this script is choosing).
    params = {
        "project_id": project_id,
        "region": region,
        "radius": radius_m,
        "grid": grid_resolution_m,
        "workers": 8,
        "max_interference_sites": 10,
        # Cache mode has no live bridge (port 5224) to call - frontend grid
        # sampling needs bridge.get_grid_analytics(). Falling back to the
        # deterministic circular grid here matches what production itself
        # already does for this project when frontend sampling has nothing
        # to return (confirmed earlier this session), so this isn't a
        # shortcut - it's the same real fallback path.
        "use_frontend_grid_sampling": data_dir is None,
        "samples_per_grid_axis": 3,
        "max_cells_per_grid": 3,
        "min_cells_per_grid": 1,
        "ensure_all_cells": True,
        "min_grids_per_cell": 1,
        "min_candidate_rsrp_dbm": -128,
        "candidate_safety_cap": 20,
    }
    if cached_polygon_wkt_lonlat:
        params["polygon_wkt"] = cached_polygon_wkt_lonlat

    # ---- STAGE 1: COST-231 raw (real production function, unmodified) ----
    print("[TRACE] STAGE 1: running COST-231 (run_rf_prediction_fast) for ALL sectors...")
    raw_pred_df = run_rf_prediction_fast(site_df, drive_df, building_df, params)
    print(f"[TRACE] STAGE 1 done: {len(raw_pred_df)} points across all sectors, "
          f"rsrp_range=({raw_pred_df['pred_rsrp'].min():.2f},{raw_pred_df['pred_rsrp'].max():.2f})")

    pred_df = raw_pred_df.copy()
    pred_df["stage1_raw_rsrp"] = pred_df["pred_rsrp"]
    pred_df["stage1_raw_rsrq"] = pred_df["pred_rsrq"]
    pred_df["stage1_raw_sinr"] = pred_df["pred_sinr"]

    # ---- Replicate apply_full_display_correction's real sequence, with a
    # snapshot inserted before the point where production code overwrites
    # the geo-only value with the DT-calibrated one. ----
    current_engine = engine.get(region.lower(), engine["india"])
    if data_dir is not None:
        polygons = [load_wkt(cached_polygon_wkt_lonlat)]
        print(f"[TRACE] using cached polygon (lon/lat, verified) from {data_dir / 'project_polygon.geojson'}")
    else:
        polygons = _resolve_prediction_polygons(params, current_engine)

    site_norm = normalize_site_for_geo(site_df)
    weights, weights_summary = load_geo_weights(project_id=project_id, weights_path=None)
    print(f"[TRACE] geo weights source={weights_summary}")

    polygon_list = list(polygons)
    polygon_gdf = gpd.GeoDataFrame({"geometry": polygon_list}, crs="EPSG:4326")
    polygon_gdf, polygon_alignment = align_project_polygon_to_points(polygon_gdf, site_norm)
    print(f"[TRACE] polygon_alignment={polygon_alignment}")

    osm_enabled = _osm_enrichment_enabled(params)
    building_gdf = building_df_to_gdf(building_df)
    building_gdf, building_alignment = align_building_geometries_to_project(building_gdf, polygon_gdf)
    print(f"[TRACE] building_alignment={building_alignment} building_rows={len(building_gdf)}")
    building_gdf = _enrich_buildings_with_osm_heights(building_gdf, polygon_gdf, enabled=osm_enabled)

    # Real building geometry + spatial index for the indoor-loss replica used
    # by both the whole-frame Stage 1 unclipped computation below and the
    # per-sector output loop further down (built once, reused - production's
    # own real buildings, not a separate fetch).
    building_geoms_list = list(building_gdf.geometry)
    building_sindex = building_gdf.sindex

    tile_size_m = float(params.get("tile_size_m") or max(float(params.get("grid", 25.0)), DEFAULT_TILE_SIZE_M))
    cluster_count = 5

    grid_gdf = create_analysis_grid(polygon_gdf, tile_size_m)
    grid_gdf = attach_building_features(grid_gdf, building_gdf)
    grid_gdf = _attach_osm_context_features(grid_gdf, polygon_gdf, enabled=osm_enabled)
    grid_df, _ = build_grid_feature_frame(grid_gdf, site_norm, cluster_count)
    grid_df, geo_status = augment_grid_with_advanced_geo_features(grid_df, building_gdf, site_norm, dem_raster_path=dem_path)
    print(f"[TRACE] OLD production clutter distribution (for comparison only): {grid_df['clutter_class'].value_counts().to_dict()}")

    building_gdf_h = None  # set below if the corrected-clutter override runs; used later for real diffraction heights too
    if data_dir is not None:
        # ---- TEST-CASE-ONLY OVERRIDE: replace production's old self-relative
        # quantile clutter_class (_derive_clutter_class, just printed above)
        # with the corrected classification already validated in
        # compute_clutter_final_v2.py this session - real Overture building/
        # road/water/green presence + real GHS-OBAT height + cited GHS-BUILT-C
        # height bands / DEGURBA water-share threshold, not invented cutoffs.
        # tools/lte_prediction/ is NOT touched - only grid_df/grid_gdf in this
        # trace script are overridden, after production's own functions have
        # already run.
        print("[TRACE] overriding clutter_class with corrected Overture+GHS-OBAT+cited-threshold classification (test-case only)...")
        from tests.baseline.compute_clutter_final_v2 import (
            GREEN_LC_SUBTYPES, GREEN_LU_SUBTYPES, GREEN_LU_CLASS,
            attach_building_features_fixed, attach_area_ratio, attach_road_length,
            impute_building_heights, attach_surrounding_height, classify as corrected_classify,
        )
        obat_csv = Path(r"C:\Users\PC\AppData\Local\Temp\claude\c--Users-PC-Desktop-S-Tracer-Exe-S-Tracer-Exe\bee613e1-06ad-457d-9fd7-900823b4b9ed\scratchpad\ghsobat_project210_bbox.csv")

        corrected_grid = attach_building_features_fixed(grid_gdf[["grid_id", "geometry", "cell_area_m2"]].copy(), building_gdf)

        roads = gpd.read_file(data_dir / "roads_segment.geojson")
        corrected_grid = attach_road_length(corrected_grid, roads)

        water = gpd.read_file(data_dir / "water.geojson")
        corrected_grid = attach_area_ratio(corrected_grid, water, "water_ratio")

        land_cover = gpd.read_file(data_dir / "land_cover.geojson")
        land_use = gpd.read_file(data_dir / "land_use.geojson")
        green_lc = land_cover[land_cover["subtype"].isin(GREEN_LC_SUBTYPES)]
        green_lu = land_use[land_use["subtype"].isin(GREEN_LU_SUBTYPES) | land_use["class"].isin(GREEN_LU_CLASS)]
        green = gpd.GeoDataFrame(pd.concat([green_lc, green_lu], ignore_index=True), crs="EPSG:4326")
        corrected_grid = attach_area_ratio(corrected_grid, green, "green_ratio")

        if obat_csv.exists():
            building_gdf_h = impute_building_heights(building_gdf, obat_csv)
            project_mean_height = float(building_gdf_h["height"].mean())
            surrounding_df = attach_surrounding_height(corrected_grid, building_gdf_h, radius_m=100.0)
            corrected_grid = corrected_grid.merge(surrounding_df, on="grid_id", how="left")
        else:
            print(f"[TRACE] GHS-OBAT csv not found at {obat_csv} - skipping height-based tiering, falling back to building presence only")
            corrected_grid["surrounding_height_m"] = np.nan
            project_mean_height = 20.0

        corrected_grid["clutter_class"] = corrected_grid.apply(lambda r: corrected_classify(r, project_mean_height), axis=1)
        print(f"[TRACE] CORRECTED clutter distribution (all sectors): {corrected_grid['clutter_class'].value_counts().to_dict()}")

        grid_df = grid_df.drop(columns=["clutter_class"]).merge(corrected_grid[["grid_id", "clutter_class"]], on="grid_id", how="left")

    grid_gdf = grid_gdf.merge(grid_df[["grid_id", "clutter_class", "morphology_cluster"]], on="grid_id", how="left")
    print(f"[TRACE] clutter distribution actually used downstream (all sectors): {grid_df['clutter_class'].value_counts().to_dict()}")

    pred_work = pred_df.copy()
    pred_work = assign_points_to_tiles(pred_work, grid_gdf)
    pred_work = _attach_missing_grid_features_by_grid_id(pred_work, grid_df)
    if "Node_Cell_ID" not in pred_work.columns and "node_cell_id" in pred_work.columns:
        pred_work["Node_Cell_ID"] = pred_work["node_cell_id"].astype(str)
    pred_work = attach_fixed_serving_sinr_rsrq_proxy(pred_work, site_norm)

    # ---- Whole-frame real pre-clip Stage 1 RSRP (same formula production
    # uses internally, captured BEFORE the np.clip(-140,-44) that hides the
    # true value once it drops below the floor - includes the real flat
    # 15dB indoor/building loss, detect_indoor's real mechanism). Computed
    # once here for every point/sector so it can feed BOTH the Stage 2b
    # SPM-style fit below AND the per-sector output loop further down
    # (previously recomputed per-sector; now computed once and reused). ----
    pred_work["_cell_id_key"] = pred_work["Node_Cell_ID"].astype(str).str.split("_").str[1]
    site_lookup = site_norm.set_index("cell_id")
    site_rf_cols = site_lookup[[
        "lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "antenna_height", "tx_power", "frequency_mhz",
    ]].rename(columns={
        "lat": "_s_lat", "lon": "_s_lon", "azimuth": "_s_az", "electrical_tilt": "_s_etilt",
        "mechanical_tilt": "_s_mtilt", "antenna_height": "_s_htx", "tx_power": "_s_txpwr", "frequency_mhz": "_s_freq",
    })
    pred_work = pred_work.merge(site_rf_cols, left_on="_cell_id_key", right_index=True, how="left")
    indoor_losses_all = np.array([
        detect_indoor_loss(lat, lon, building_sindex, building_geoms_list)
        for lat, lon in zip(pred_work["lat"].astype(float), pred_work["lon"].astype(float))
    ])
    pred_work["stage1_raw_rsrp_unclipped"] = compute_sector_rsrp_unclipped(
        s_lat=pred_work["_s_lat"], s_lon=pred_work["_s_lon"], s_az=pred_work["_s_az"],
        s_etilt=pred_work["_s_etilt"], s_mtilt=pred_work["_s_mtilt"], s_htx=pred_work["_s_htx"],
        tx_pwr=pred_work["_s_txpwr"], freq=pred_work["_s_freq"],
        p_lat=pred_work["lat"].astype(float), p_lon=pred_work["lon"].astype(float),
        indoor_loss=indoor_losses_all,
    )

    # ---- Real diffraction loss (test-case only): single-dominant-knife-edge
    # loss from REAL per-path building crossings + REAL (GHS-OBAT-imputed,
    # conservatively-floored) building heights - replaces production's
    # diffraction_proxy_db's small linear coefficients with real ITU-R
    # diffraction physics. Computed once per real serving site (grouped by
    # _cell_id_key, since diffraction depends on the real site->point
    # geometry, not azimuth) and reused for Stage 2c below. ----
    print("[TRACE] computing real knife-edge diffraction loss (test-case only)...")
    if building_gdf_h is not None:
        building_heights_arr = building_gdf_h["height"].to_numpy()
    else:
        print("[TRACE] no GHS-OBAT-imputed heights available - falling back to a flat "
              f"conservative {CONSERVATIVE_MIN_BUILDING_HEIGHT_M}m for every real building")
        building_heights_arr = np.full(len(building_geoms_list), np.nan)
    # IMPORTANT: building_sindex/building_geoms_list (used for the indoor-loss
    # check above) are in EPSG:4326 (lat/lon degrees). Diffraction physics
    # needs real METER distances, so path geometry below is built in a local
    # UTM projection - querying the lat/lon spatial index with a UTM-meter
    # LineString would never match (bounding boxes in degrees vs meters never
    # overlap) - confirmed exactly this bug on the first run (every point
    # returned 0 candidates, diffraction_loss was 0.00 for all 94,392
    # points). Fix: build a SEPARATE UTM-projected building index here,
    # matching production's own real _attach_building_path_features pattern
    # (building_utm = building_gdf.to_crs(utm_crs); sindex from that). ----
    utm_crs_for_diffraction = _choose_utm_crs(building_gdf)
    utm_transformer_for_diffraction = Transformer.from_crs("EPSG:4326", utm_crs_for_diffraction, always_xy=True)
    building_gdf_utm = building_gdf.to_crs(utm_crs_for_diffraction)
    building_geoms_utm_list = list(building_gdf_utm.geometry)
    building_sindex_utm = building_gdf_utm.sindex

    diffraction_loss_all = np.zeros(len(pred_work), dtype=float)
    for cell_key, group in pred_work.groupby("_cell_id_key"):
        g_s_lat = group["_s_lat"].iloc[0]
        g_s_lon = group["_s_lon"].iloc[0]
        g_s_htx = group["_s_htx"].iloc[0]
        g_s_freq = group["_s_freq"].iloc[0]
        if pd.isna(g_s_lat) or pd.isna(g_s_lon) or pd.isna(g_s_htx) or pd.isna(g_s_freq):
            continue
        group_losses = compute_diffraction_loss_knife_edge(
            s_lat=float(g_s_lat), s_lon=float(g_s_lon), s_htx=float(g_s_htx), freq_mhz=float(g_s_freq),
            p_lats=group["lat"].astype(float).to_numpy(), p_lons=group["lon"].astype(float).to_numpy(),
            building_sindex=building_sindex_utm, building_geoms=building_geoms_utm_list,
            building_heights=building_heights_arr, utm_transformer=utm_transformer_for_diffraction,
        )
        diffraction_loss_all[pred_work.index.get_indexer(group.index)] = group_losses
    pred_work["diffraction_loss_db"] = diffraction_loss_all
    print(f"[TRACE] real diffraction loss done: mean={pred_work['diffraction_loss_db'].mean():.2f}dB "
          f"max={pred_work['diffraction_loss_db'].max():.2f}dB "
          f"points_with_real_diffraction={(pred_work['diffraction_loss_db'] > 0).sum()}/{len(pred_work)}")

    # ---- STAGE 2: geo-corrected (captured BEFORE DT calibration overwrites pred_rsrp_geo) ----
    print("[TRACE] STAGE 2: apply_experimental_geo_adjustments...")
    pred_work, geo_summary = apply_experimental_geo_adjustments(pred_work, weights=weights)
    pred_work["stage2_geo_rsrp"] = pred_work["pred_rsrp_geo"].astype(float).copy()
    print(f"[TRACE] STAGE 2 done: rsrp_range=({pred_work['stage2_geo_rsrp'].min():.2f},{pred_work['stage2_geo_rsrp'].max():.2f})")

    drive_train_df, drive_holdout_df, split_summary = split_drive_train_holdout(drive_df, validation_fraction=0.25)
    train_eval, _, train_geo_metrics = evaluate_geo_against_dt(drive_train_df, pred_work)
    dt_calibration_models, dt_calibration_debug = fit_dt_holdout_calibration(train_eval)

    # ---- STAGE 2b: SPM/Atoll-style replacement (test-case-only, NOT wired
    # into production). Real Standard Propagation Model structure: physics
    # (stage1_raw_rsrp_unclipped - distance already modelled smoothly via
    # COST-231's own slope_term*log10(d_km)) + ONE calibrated per-clutter-
    # class term (K_clutter*f(clutter)), fit by least-squares against real
    # DT residuals - replacing the current stack of independently hand-
    # picked, independently distance-clipped feature weights that produced
    # the flat -140dBm plateau from ~1-2km. ----
    print("[TRACE] STAGE 2b: SPM-style clutter-only geo correction (test-case only)...")
    # drive_train_df/drive_holdout_df are already the output of
    # split_drive_train_holdout, which itself already ran
    # _prepare_drive_measurements(drive_df) internally (RSRP_meas/lat/lon
    # already present) - not re-normalizing here to avoid re-detecting the
    # measurement column against a frame that now also contains RSRP_meas.
    dt_train_matched = attach_pred_cols_to_dt_points(
        drive_train_df, pred_work, ["stage1_raw_rsrp_unclipped", "clutter_class"],
    )
    clutter_offsets, offsets_debug = fit_spm_style_clutter_offsets(dt_train_matched)
    print(f"[TRACE] STAGE 2b fitted per-class offsets (dB, real DT train split, n={offsets_debug['matched_dt_rows_used']}): {offsets_debug['per_class']}")
    pred_work = apply_spm_style_stage2(pred_work, clutter_offsets, offsets_debug["k1_constant_db"])
    print(f"[TRACE] STAGE 2b done: rsrp_range=({pred_work['stage2b_spm_rsrp'].min():.2f},{pred_work['stage2b_spm_rsrp'].max():.2f})")

    # ---- STAGE 2c: physics + REAL diffraction (physics formula, computed
    # directly from real per-path building geometry, NOT DT-fit) + clutter
    # offset - fit AFTER diffraction is subtracted, so the clutter term only
    # has to explain what real diffraction physics doesn't already account
    # for, instead of one clutter constant being asked to absorb both a
    # coarse average AND real per-point building obstruction. This is the
    # fix for Stage 2b's biggest identified weakness this session: a single
    # DT-fit constant per class pulls every point of that class toward the
    # class's average DT value regardless of real local obstruction,
    # smoothing away exactly the kind of sharp, building-driven differentiation
    # real production coverage maps should show. ----
    print("[TRACE] STAGE 2c: physics + real diffraction + DT-calibrated clutter (test-case only)...")
    dt_train_matched_2c = attach_pred_cols_to_dt_points(
        drive_train_df, pred_work, ["stage1_raw_rsrp_unclipped", "diffraction_loss_db", "clutter_class"],
    )
    dt_train_matched_2c["stage1_plus_diffraction_unclipped"] = (
        pd.to_numeric(dt_train_matched_2c["stage1_raw_rsrp_unclipped"], errors="coerce")
        - pd.to_numeric(dt_train_matched_2c["diffraction_loss_db"], errors="coerce")
    )
    clutter_offsets_2c, offsets_debug_2c = fit_spm_style_clutter_offsets(
        dt_train_matched_2c, baseline_col="stage1_plus_diffraction_unclipped",
    )
    print(f"[TRACE] STAGE 2c fitted per-class offsets (dB, AFTER real diffraction subtracted, "
          f"n={offsets_debug_2c['matched_dt_rows_used']}): {offsets_debug_2c['per_class']}")
    pred_work = apply_spm_style_stage2c(pred_work, "diffraction_loss_db", clutter_offsets_2c, offsets_debug_2c["k1_constant_db"])
    print(f"[TRACE] STAGE 2c done: rsrp_range=({pred_work['stage2c_spm_rsrp'].min():.2f},{pred_work['stage2c_spm_rsrp'].max():.2f})")

    # ---- Real MAE validation: current Stage 2 (hand-tuned weights) vs new
    # Stage 2b (SPM-style, clutter only) vs Stage 2c (SPM-style + real
    # diffraction), on the SAME held-out real DT points neither was fit on -
    # not a claim, a measured comparison. ----
    holdout_matched = attach_pred_cols_to_dt_points(
        drive_holdout_df, pred_work,
        ["pred_rsrp", "stage2_geo_rsrp", "stage2b_spm_rsrp", "stage2c_spm_rsrp"],
    )

    # _mae_rmse/binned_mae_rmse/RSRP_BIN_* are module-level (mae_rmse/
    # binned_mae_rmse above) so the project-wide trace script can reuse the
    # exact same metric definitions.
    mae_s1, rmse_s1 = mae_rmse(holdout_matched, "pred_rsrp")
    mae_s2, rmse_s2 = mae_rmse(holdout_matched, "stage2_geo_rsrp")
    mae_s2b, rmse_s2b = mae_rmse(holdout_matched, "stage2b_spm_rsrp")
    mae_s2c, rmse_s2c = mae_rmse(holdout_matched, "stage2c_spm_rsrp")
    stage2b_validation = {
        "holdout_dt_rows_matched": int(len(holdout_matched.dropna(subset=["RSRP_meas"]))) if "RSRP_meas" in holdout_matched.columns else 0,
        "mae_stage1_raw_physics_only": mae_s1, "rmse_stage1_raw_physics_only": rmse_s1,
        "mae_stage2_current_hand_tuned_weights": mae_s2, "rmse_stage2_current_hand_tuned_weights": rmse_s2,
        "mae_stage2b_spm_style_clutter_only": mae_s2b, "rmse_stage2b_spm_style_clutter_only": rmse_s2b,
        "mae_stage2c_spm_style_plus_diffraction": mae_s2c, "rmse_stage2c_spm_style_plus_diffraction": rmse_s2c,
    }
    print(f"[TRACE] STAGE 2b/2c real MAE validation (same DT holdout, not used for fitting either stage): {stage2b_validation}")

    # ---- STAGE 3-REWIRED: DT residual calibration on top of Stage 2b (not
    # Stage 2) - the "delta correction on an already-calibrated baseline"
    # pattern confirmed via real published research this session (calibrate
    # the propagation model once via Stage 2b, then fit a SEPARATE residual
    # model against ITS output, not the uncalibrated/hand-tuned one).
    # clutter_class is excluded from this model's own features since
    # Stage 2b already owns that signal - avoids the double-DT-training
    # overlap flagged this session. Test-case only - production's real
    # Stage 3 (fit_dt_holdout_calibration/apply_dt_holdout_calibration,
    # hardcoded to calibrate against pred_rsrp_geo) is untouched and still
    # runs separately right below, for a direct, honest before/after. ----
    print("[TRACE] STAGE 3-rewired: DT residual calibration on top of Stage 2b (test-case only)...")
    train_eval_2b = attach_pred_cols_to_dt_points(
        drive_train_df, pred_work, ["stage2b_spm_rsrp"] + STAGE3_FEATURE_COLS,
    )
    stage3_rewired_bundle, stage3_rewired_debug = fit_stage3_rewired(train_eval_2b, pred_col="stage2b_spm_rsrp")
    print(f"[TRACE] STAGE 3-rewired fit debug: {stage3_rewired_debug}")
    pred_work["stage3_rewired_rsrp"] = apply_stage3_rewired(pred_work, stage3_rewired_bundle)
    print(f"[TRACE] STAGE 3-rewired done: rsrp_range=({pred_work['stage3_rewired_rsrp'].min():.2f},{pred_work['stage3_rewired_rsrp'].max():.2f})")

    # ---- STAGE 3 (production, unchanged): DT-calibrated on top of Stage 2 (hand-tuned) ----
    print("[TRACE] STAGE 3: apply_dt_holdout_calibration + preserve_calibrated_kpis...")
    pred_work = apply_dt_holdout_calibration(pred_work, dt_calibration_models)
    pred_work = preserve_calibrated_kpis(pred_work)
    print(f"[TRACE] STAGE 3 done: rsrp_range=({pred_work['pred_rsrp_calibrated'].min():.2f},{pred_work['pred_rsrp_calibrated'].max():.2f})")

    # ---- Consolidated real MAE validation: every stage, same held-out DT
    # points, none of which were used to fit Stage 2b, Stage 3-rewired, or
    # the old Stage 3. Answers directly whether Stage 2b + rewired Stage 3
    # actually beats the current production stack, on real data. ----
    full_holdout_matched = attach_pred_cols_to_dt_points(
        drive_holdout_df, pred_work,
        ["pred_rsrp", "stage2_geo_rsrp", "stage2b_spm_rsrp", "stage3_rewired_rsrp", "pred_rsrp_calibrated"],
    )

    mae_f1, rmse_f1 = mae_rmse(full_holdout_matched, "pred_rsrp")
    mae_f2, rmse_f2 = mae_rmse(full_holdout_matched, "stage2_geo_rsrp")
    mae_f2b, rmse_f2b = mae_rmse(full_holdout_matched, "stage2b_spm_rsrp")
    mae_f3old, rmse_f3old = mae_rmse(full_holdout_matched, "pred_rsrp_calibrated")
    mae_f3r, rmse_f3r = mae_rmse(full_holdout_matched, "stage3_rewired_rsrp")
    stage3_validation = {
        "holdout_dt_rows_matched": int(len(full_holdout_matched.dropna(subset=["RSRP_meas"]))) if "RSRP_meas" in full_holdout_matched.columns else 0,
        "mae_stage1_raw_physics": mae_f1, "rmse_stage1_raw_physics": rmse_f1,
        "mae_stage2_current_hand_tuned": mae_f2, "rmse_stage2_current_hand_tuned": rmse_f2,
        "mae_stage2b_spm_style": mae_f2b, "rmse_stage2b_spm_style": rmse_f2b,
        "mae_stage3_old_calibrated_on_stage2": mae_f3old, "rmse_stage3_old_calibrated_on_stage2": rmse_f3old,
        "mae_stage3_rewired_calibrated_on_stage2b": mae_f3r, "rmse_stage3_rewired_calibrated_on_stage2b": rmse_f3r,
    }
    print(f"[TRACE] FULL STAGE VALIDATION (same DT holdout across every stage): {stage3_validation}")

    # ---- Same validation, broken down by the REAL measured RSRP's signal-
    # strength bin, so error isn't hidden by averaging together a strong-
    # signal regime with a weak-signal one. ----
    stage3_validation_by_rsrp_bin = {
        "stage1_raw_physics": binned_mae_rmse(full_holdout_matched, "pred_rsrp"),
        "stage2_current_hand_tuned": binned_mae_rmse(full_holdout_matched, "stage2_geo_rsrp"),
        "stage2b_spm_style": binned_mae_rmse(full_holdout_matched, "stage2b_spm_rsrp"),
        "stage3_old_calibrated_on_stage2": binned_mae_rmse(full_holdout_matched, "pred_rsrp_calibrated"),
        "stage3_rewired_calibrated_on_stage2b": binned_mae_rmse(full_holdout_matched, "stage3_rewired_rsrp"),
    }
    print(f"[TRACE] FULL STAGE VALIDATION BY RSRP BIN: {stage3_validation_by_rsrp_bin}")

    # ---- STAGE 4: smoothed / DT-anchor-blended ----
    print("[TRACE] STAGE 4: apply_demo_dt_overlay...")
    pred_work, demo_summary = apply_demo_dt_overlay(
        pred_work, drive_df, replace_radius_m=20.0, blend_sigma_m=60.0, blend_radius_m=140.0,
    )
    print(f"[TRACE] STAGE 4 done: demo_summary={demo_summary}")

    # ---- Robust site/sector matching: pred_work's Node_Cell_ID has no
    # region suffix ("LA200267_LA200267A2_A2_28"); site_norm's does
    # ("...A2_28_Taiwan"). Match on the parsed cell_id component instead of
    # the raw string - an earlier version of this script matched the raw
    # string, silently failed, and fell back to an arbitrary wrong site for
    # the bearing/distance reference. (_cell_id_key/site_lookup were already
    # built earlier, before Stage 2, to compute the whole-frame Stage 1
    # unclipped value - reused here, not rebuilt.) ----

    sites_out = {}
    for site_id, site_rows in site_norm.groupby("Site ID"):
        if only_sites and site_id not in only_sites:
            continue
        sectors_out = {}
        for _, srow in site_rows.iterrows():
            cell_id = srow["cell_id"]
            sector_pts = pred_work[pred_work["_cell_id_key"] == cell_id].copy()
            if sector_pts.empty:
                continue
            site_lat = float(srow["lat"])
            site_lon = float(srow["lon"])
            azimuth = float(srow["azimuth"]) if pd.notna(srow["azimuth"]) else 0.0
            sector_pts["distance_m"] = haversine_m(site_lat, site_lon, sector_pts["lat"].astype(float), sector_pts["lon"].astype(float))
            sector_pts["bearing_deg"] = bearing_deg(site_lat, site_lon, sector_pts["lat"].astype(float), sector_pts["lon"].astype(float))
            sector_pts["off_axis_deg"] = off_axis_deg(sector_pts["bearing_deg"], azimuth)
            # stage1_raw_rsrp_unclipped and stage2b_spm_rsrp(_unclipped) are
            # already on pred_work (computed once, whole-frame, before Stage
            # 2/2b above) and carried through the sector_pts slice - no
            # per-sector recomputation needed here anymore.

            radial = sector_pts.sort_values("distance_m").copy()
            radial["spacing_from_prev_m"] = radial["distance_m"].diff()
            for col in ["stage1_raw_rsrp", "stage2_geo_rsrp", "stage2b_spm_rsrp", "stage2c_spm_rsrp", "stage3_rewired_rsrp", "pred_rsrp_calibrated", "pred_rsrp_demo"]:
                radial[f"delta_{col}"] = radial[col].diff()
            # keep only the on-axis half of the grid for a cleaner radial profile
            radial_onaxis = radial[radial["off_axis_deg"] <= ANTENNA_3DB_BEAMWIDTH_DEG].sort_values("distance_m")

            keep_cols = [
                "lat", "lon", "distance_m", "bearing_deg", "off_axis_deg",
                "stage1_raw_rsrp", "stage1_raw_rsrp_unclipped", "stage1_raw_rsrq", "stage1_raw_sinr",
                "stage2_geo_rsrp", "stage2b_spm_rsrp", "stage2b_spm_rsrp_unclipped",
                "stage2c_spm_rsrp", "stage2c_spm_rsrp_unclipped", "diffraction_loss_db", "stage3_rewired_rsrp",
                "pred_rsrp_calibrated", "pred_rsrq_calibrated", "pred_sinr_calibrated",
                "pred_rsrp_demo", "pred_rsrq_demo", "pred_sinr_demo",
                "clutter_class", "morphology_cluster", "building_count", "building_area_ratio",
                "road_length_m", "green_ratio", "water_ratio",
                "los_blocker_count", "los_blocked_ratio", "max_blocker_height_m",
                "demo_dt_distance_m", "demo_blend_weight", "demo_dt_anchor",
            ]
            keep_cols = [c for c in keep_cols if c in sector_pts.columns]

            def df_to_records(df):
                return json.loads(df.replace({np.nan: None}).to_json(orient="records"))

            radial_cols = keep_cols + [
                "spacing_from_prev_m", "delta_stage1_raw_rsrp", "delta_stage2_geo_rsrp",
                "delta_stage2b_spm_rsrp", "delta_stage2c_spm_rsrp", "delta_stage3_rewired_rsrp",
                "delta_pred_rsrp_calibrated", "delta_pred_rsrp_demo",
            ]
            radial_cols = [c for c in radial_cols if c in radial_onaxis.columns]

            sectors_out[srow["sector"]] = {
                "cell_id": cell_id,
                "band": None if pd.isna(srow.get("band")) else (int(srow["band"]) if float(srow["band"]).is_integer() else float(srow["band"])),
                "azimuth": azimuth,
                "beamwidth_3db_deg": ANTENNA_3DB_BEAMWIDTH_DEG,
                "point_count": int(len(sector_pts)),
                "stage_summary": {
                    "stage1_raw_rsrp": {"min": float(sector_pts["stage1_raw_rsrp"].min()), "max": float(sector_pts["stage1_raw_rsrp"].max()), "mean": float(sector_pts["stage1_raw_rsrp"].mean()), "std": float(sector_pts["stage1_raw_rsrp"].std())},
                    "stage2_geo_rsrp": {"min": float(sector_pts["stage2_geo_rsrp"].min()), "max": float(sector_pts["stage2_geo_rsrp"].max()), "mean": float(sector_pts["stage2_geo_rsrp"].mean()), "std": float(sector_pts["stage2_geo_rsrp"].std())},
                    "stage2b_spm_rsrp": {"min": float(sector_pts["stage2b_spm_rsrp"].min()), "max": float(sector_pts["stage2b_spm_rsrp"].max()), "mean": float(sector_pts["stage2b_spm_rsrp"].mean()), "std": float(sector_pts["stage2b_spm_rsrp"].std())},
                    "stage2c_spm_rsrp": {"min": float(sector_pts["stage2c_spm_rsrp"].min()), "max": float(sector_pts["stage2c_spm_rsrp"].max()), "mean": float(sector_pts["stage2c_spm_rsrp"].mean()), "std": float(sector_pts["stage2c_spm_rsrp"].std())},
                    "stage3_rewired_rsrp": {"min": float(sector_pts["stage3_rewired_rsrp"].min()), "max": float(sector_pts["stage3_rewired_rsrp"].max()), "mean": float(sector_pts["stage3_rewired_rsrp"].mean()), "std": float(sector_pts["stage3_rewired_rsrp"].std())},
                    "stage3_calibrated_rsrp": {"min": float(sector_pts["pred_rsrp_calibrated"].min()), "max": float(sector_pts["pred_rsrp_calibrated"].max()), "mean": float(sector_pts["pred_rsrp_calibrated"].mean()), "std": float(sector_pts["pred_rsrp_calibrated"].std())},
                    "stage4_smoothed_rsrp": {"min": float(sector_pts["pred_rsrp_demo"].min()), "max": float(sector_pts["pred_rsrp_demo"].max()), "mean": float(sector_pts["pred_rsrp_demo"].mean()), "std": float(sector_pts["pred_rsrp_demo"].std())},
                },
                "clutter_distribution": sector_pts["clutter_class"].value_counts().to_dict() if "clutter_class" in sector_pts.columns else {},
                "all_points": df_to_records(sector_pts[keep_cols]),
                "radial_profile": df_to_records(radial_onaxis[radial_cols]),
            }
            print(f"[TRACE] site={site_id} sector={srow['sector']} cell_id={cell_id} azimuth={azimuth} "
                  f"points={len(sector_pts)} on_axis_radial_points={len(radial_onaxis)}")

        if sectors_out:
            first = site_rows.iloc[0]
            sites_out[site_id] = {
                "site_id": site_id,
                "lat": float(first["lat"]),
                "lon": float(first["lon"]),
                "sectors": sectors_out,
            }

    output = {
        "project_id": project_id,
        "region": region,
        "demo_summary": demo_summary,
        "dt_calibration_debug": dt_calibration_debug,
        "weights_summary": weights_summary,
        "stage2b_spm_offsets": offsets_debug,
        "stage2c_spm_offsets": offsets_debug_2c,
        "stage2b_validation": stage2b_validation,
        "stage3_rewired_fit_debug": stage3_rewired_debug,
        "stage3_validation": stage3_validation,
        "stage3_validation_by_rsrp_bin": stage3_validation_by_rsrp_bin,
        "sites": sites_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output), encoding="utf-8")
    print(f"[TRACE] wrote {output_path} ({output_path.stat().st_size} bytes), {len(sites_out)} sites")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Site -> sector stage-by-stage LTE prediction trace")
    parser.add_argument("--project-id", type=int, default=210)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--session-ids", type=int, nargs="+", default=None)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--polygon-ids", type=str, default=None)
    parser.add_argument("--radius-m", type=float, default=500.0)
    parser.add_argument("--grid-resolution-m", type=float, default=25.0)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "site_sector_trace.json")
    parser.add_argument("--data-dir", type=Path, default=None,
                         help="Load site/drive/building/polygon from this local cache dir instead of "
                              "the live DB/bridge (e.g. tests/baseline/data/project_210_taiwan)")
    parser.add_argument("--only-site", type=str, nargs="+", default=None,
                         help="Restrict the OUTPUT to these physical Site IDs (e.g. LA201565). "
                              "The full site set is still fetched and used for COST-231 interference "
                              "candidates, exactly like production - only what gets written is scoped down.")
    args = parser.parse_args(argv)

    session_ids = args.session_ids
    if session_ids is None and args.data_dir is None:
        from sqlalchemy import create_engine, text
        db_url = os.getenv("DATABASE_URL_Taiwan") if args.region == "taiwan" else os.getenv("DATABASE_URL")
        eng = create_engine(db_url)
        with eng.connect() as conn:
            ref = conn.execute(text("SELECT ref_session_id FROM tbl_project WHERE id=:pid"), {"pid": args.project_id}).scalar()
        session_ids = [int(s.strip()) for s in (ref or "").split(",") if s.strip()]
        print(f"[TRACE] resolved {len(session_ids)} session_ids from tbl_project.ref_session_id")

    run_trace(
        project_id=args.project_id,
        region=args.region,
        session_ids=session_ids,
        operator=args.operator,
        radius_m=args.radius_m,
        grid_resolution_m=args.grid_resolution_m,
        polygon_ids=args.polygon_ids,
        output_path=args.output,
        only_sites=set(args.only_site) if args.only_site else None,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
