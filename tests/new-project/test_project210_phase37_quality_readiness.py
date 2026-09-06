"""Phase 37: dynamic RSRQ/SINR with corrected 5G serving/interference identity.

Phase 36 v2 is the RSRP base surface for transfer to every grid and DT point.
For 5G quality, Phase 37 reuses Phase 36's serving-cell hygiene before
SINR/RSRQ scoring, recomputes the serving/interference set around the corrected
serving cell, excludes that corrected server from interference, and learns a
carrier active-interference factor from the training DT split. Earlier phase
files and outputs are not modified.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
for path in (ML_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.lte_prediction.Sector_wise_prediction_code_copy import compute_sector_rsrp
import test_project210_phase36_final as phase36
import test_project210_phase36_v2_distance_shape as phase36v2

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE26_DIR = PROJECT_DIR / "cost231_phase26_corrected_obstruction_profile"
PHASE36V2_DIR = PROJECT_DIR / "cost231_phase36_v2_distance_shape"
OUT_DIR = PROJECT_DIR / "cost231_phase37_quality_readiness"
IDENTITY_PATH = PROJECT_DIR / "baseline_fetch_scope" / "site_identity_strict_cells_project210.parquet"
EARTH_RADIUS_M = 6_371_000.0

# Co-channel interference: only neighbours within a relative-power window of the
# server contribute. The window is FIT PER TECHNOLOGY FROM THE DRIVE TEST
# (_fit_interferer_cutoff) - the cutoff whose SINR base best matches measured
# SINR. These are only the fallback seeds when a project has too little DT to
# fit; nothing here is a per-project constant.
INTERFERER_MIN_DBM = -125.0
NR_INTERFERER_RELATIVE_CUTOFF_DB_DEFAULT = 24.0
LTE_INTERFERER_RELATIVE_CUTOFF_DB_DEFAULT = 15.0
CUTOFF_SWEEP_DB = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 38, 45]
# Widen the interferer window while each extra dB still buys at least this much
# SINR-match improvement; stop at the knee. Keeps the fit off the sweep edge
# without a hand-set window.
CUTOFF_MARGINAL_GAIN_MIN_DB_PER_DB = 0.05
CUTOFF_MIN_DB = 12.0

# Phase-25-style local residual field (same recipe as the RSRP v2 surface):
# after the hierarchical median, an inverse-distance DT-residual field adds the
# location-specific correction that a per-carrier median cannot.
LOCAL_MIN_NEIGHBORS = 5
LOCAL_K_NEIGHBORS = 16
LOCAL_SHRINK_N = 8.0

# Thermal-noise floor so SINR/RSRQ base is defined for every served point, even
# where no co-channel interferer clears the active filter (cell-edge / isolated
# sectors). SINR = signal / (interference + noise); RSSI = signal + interference
# + noise. The DT residual calibration is fit on this same definition.
NOISE_DBM = -104.0
NOISE_MW = 10.0 ** (NOISE_DBM / 10.0)
# UE receiver implementation ceiling (EVM / self-interference). Without it an
# isolated strong grid over the thermal floor alone reports an unphysical
# +60 dB SINR that no handset would ever log.
SINR_CEILING_DB = 35.0

# Hierarchical quality calibration: a grid/DT point takes the deepest level with
# enough training-DT support, then falls through so nothing is left uncalibrated.
CARRIER_MIN_N = 25
_CONF_RANK = {"local": 4, "carrier": 3, "tech": 2, "global": 1, "physics_only": 0}


def _carrier_key(frame: pd.DataFrame) -> pd.Series:
    frequency = pd.to_numeric(frame.get("original_frequency_mhz", frame.get("frequency_mhz")), errors="coerce").round(1)
    return frame["technology"].astype(str) + "|" + frequency.astype("string")


def _mw(dbm: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(dbm, dtype=float) / 10.0)


def _dbm(mw: float) -> float:
    return 10.0 * math.log10(mw) if np.isfinite(mw) and mw > 0.0 else np.nan


def _quality_base(signal_dbm: float, interference_mw: float) -> tuple[float, float, float]:
    """Return SIR-like SINR base, normalized RSRQ base, and interference dBm.

    The DT-calibrated residual absorbs the unobserved RSSI/resource-load term.
    This avoids the old invalid LTE shortcut that added 10log10(50) to a
    reference-signal-only RSSI estimate.
    """
    if not np.isfinite(signal_dbm):
        return np.nan, np.nan, _dbm(interference_mw)
    interference_mw = float(interference_mw) if np.isfinite(interference_mw) and interference_mw > 0.0 else 0.0
    signal_mw = float(_mw(signal_dbm))
    denom_mw = interference_mw + NOISE_MW
    total_mw = signal_mw + denom_mw
    sinr_base_db = min(10.0 * math.log10(signal_mw / denom_mw), SINR_CEILING_DB)
    rsrq_base_db = 10.0 * math.log10(signal_mw / total_mw)
    return sinr_base_db, rsrq_base_db, _dbm(interference_mw)


def _active_interferer_mask(scores: np.ndarray, signal_dbm: float, cutoff_db: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    if not np.isfinite(signal_dbm):
        return mask
    threshold = max(INTERFERER_MIN_DBM, signal_dbm - float(cutoff_db))
    return mask & (scores >= threshold)


def _fit_interference_activity(group: pd.DataFrame, technology: str) -> float:
    # The interferer window is now fit from the DT (_fit_interferer_cutoff), so
    # the effective co-channel load is already right - no separate global scalar.
    return 1.0


def _base_columns(frame: pd.DataFrame, activity_factor: float) -> tuple[list[float], list[float], list[float]]:
    signal_col = "phase37_serving_rsrp_dbm" if "phase37_serving_rsrp_dbm" in frame.columns else "serving_rsrp_dbm"
    signal = pd.to_numeric(frame[signal_col], errors="coerce").to_numpy(float)
    interference = pd.to_numeric(frame["interference_sum_mw"], errors="coerce").to_numpy(float)
    quality = [_quality_base(s, max(i * activity_factor, 0.0)) for s, i in zip(signal, interference)]
    return [row[0] for row in quality], [row[1] for row in quality], [row[2] for row in quality]


def _phase37_candidates(technology: str) -> pd.DataFrame:
    path = PHASE36V2_DIR / "phase36v2_scored_candidates_project210.parquet"
    frame = pd.read_parquet(path)
    frame = frame[frame["technology"].astype(str).eq(technology)].copy()
    frame["phase37_rsrp_dbm"] = pd.to_numeric(frame["phase36_final_rsrp_unclipped"], errors="coerce")
    frame = frame.dropna(subset=["grid_id", "strict_cell_key", "phase37_rsrp_dbm"]).copy()
    frame["carrier_key"] = _carrier_key(frame)
    return frame


def _phase36v2_dt_all() -> pd.DataFrame:
    """Rebuild full Phase 36 v2 DT scoring in memory for Phase 37 train+validation.

    Phase 36 v2 saved only validation DT rows. Quality calibration needs the
    training split too, so Phase 37 repeats the v2 re-band and scoring pipeline
    without writing or modifying any Phase 36 v2 artifact.
    """
    dt = phase36._dt_inputs()
    cand = phase36._candidate_inputs()
    freq_map = phase36v2._cell_true_freq(dt)
    if freq_map:
        dt_corr = phase36v2._reband(dt, freq_map, "assigned_strict_cell_key")
        cand_corr = phase36v2._reband(cand, freq_map, "strict_cell_key")
        for frame, corr in ((dt, dt_corr), (cand, cand_corr)):
            for col in (phase36v2.BASE_UNCLIPPED, "phase24_no_lock_reference_rsrp_unclipped"):
                frame[col] = pd.to_numeric(frame[col], errors="coerce") + corr
            frame[phase36v2.BASE_COL] = phase36v2.valid_model_rsrp(frame[phase36v2.BASE_UNCLIPPED])
            frame["phase24_no_lock_reference_rsrp"] = phase36v2.valid_model_rsrp(
                frame["phase24_no_lock_reference_rsrp_unclipped"]
            )
    train = dt[dt["phase25_split"].astype(str).eq("train")].copy()
    fit = train[
        (train["obstruction_branch"].astype(str) != "indoor")
        & (~train["p36_backlobe"].astype(bool))
    ].copy()
    layers, local_models = phase36._fit(fit)
    scored = phase36._score(dt, layers, local_models)
    scored["split"] = scored["phase25_split"].astype(str)
    return scored


def _candidate_groups(candidates: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    return {
        (str(grid_id), str(carrier_key)): group
        for (grid_id, carrier_key), group in candidates.groupby(["grid_id", "carrier_key"], sort=False)
    }


def _cochannel_score_matrix(
    lat: np.ndarray, lon: np.ndarray, cell_rows: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COST-231 + frozen Phase 36 v2 per-cell correction for every cell at every
    point. Computed once; the interferer cutoff is then swept over it for free."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = len(lat)
    cell_rows = cell_rows.reset_index(drop=True)
    radians = np.radians(np.column_stack([lat, lon]))
    score_columns: list[np.ndarray] = []
    for _, cell in cell_rows.iterrows():
        tree, corrections, _, fallback = _cell_correction_surface(candidates, str(cell["strict_cell_key"]))
        raw = _raw_cost231_at_points(cell, lat, lon)
        if tree is None:
            transfer = np.full(n, fallback, dtype=float)
        else:
            _, nearest = tree.query(radians, k=1)
            transfer = corrections[nearest[:, 0]]
        score_columns.append(raw + transfer)
    return (
        np.column_stack(score_columns),
        cell_rows["carrier_key"].astype(str).to_numpy(),
        cell_rows["strict_cell_key"].astype(str).to_numpy(),
    )


def _serving_interference(
    score_matrix: np.ndarray, cell_carrier: np.ndarray, cell_key: np.ndarray, cutoff_db: float
) -> dict:
    """Serving = argmax of the surface; co-channel interference = same-carrier
    sectors on that surface, gated at ``signal - cutoff_db``."""
    n = score_matrix.shape[0]
    server_index = np.nanargmax(score_matrix, axis=1)
    serving_rsrp = score_matrix[np.arange(n), server_index]
    serving_carrier = cell_carrier[server_index]
    eligible = np.zeros(n, dtype=int)
    interfering = np.zeros(n, dtype=int)
    interference_mw = np.zeros(n, dtype=float)
    for i in range(n):
        cochannel = np.flatnonzero(cell_carrier == serving_carrier[i])
        eligible[i] = len(cochannel)
        scores = score_matrix[i, cochannel]
        server_pos = int(np.where(cochannel == server_index[i])[0][0])
        interferers = np.delete(scores, server_pos)
        active = _active_interferer_mask(interferers, float(serving_rsrp[i]), cutoff_db)
        if active.any():
            interference_mw[i] = float(_mw(interferers[active]).sum())
            interfering[i] = int(active.sum())
    return {
        "serving_strict_cell_key": cell_key[server_index],
        "carrier_key": serving_carrier,
        "serving_rsrp_dbm": serving_rsrp,
        "eligible_cochannel_sector_count": eligible,
        "interfering_sector_count": interfering,
        "interference_sum_mw": interference_mw,
    }


def _score_points_quality(
    lat: np.ndarray, lon: np.ndarray, cell_rows: pd.DataFrame, candidates: pd.DataFrame, cutoff_db: float
) -> dict:
    """Single scorer used for BOTH grid centres and DT coordinates, so the
    RSRQ/SINR base sits on one scale and the DT-trained residual transfers to
    the grid with no systematic offset."""
    sm, cc, ck = _cochannel_score_matrix(lat, lon, cell_rows, candidates)
    return _serving_interference(sm, cc, ck, cutoff_db)


def _fit_interferer_cutoff(
    dt_tech: pd.DataFrame, cell_rows: pd.DataFrame, candidates: pd.DataFrame, default_cutoff: float
) -> tuple[float, list[dict]]:
    """Data-driven interferer window: the relative-power cutoff whose SINR base
    (after a single median debias) best matches measured SINR on the training
    DT. No per-project constant - re-fits for any project."""
    train = dt_tech[dt_tech["split"].astype(str).eq("train")].copy()
    train = train.dropna(subset=["lat", "lon"])
    measured = pd.to_numeric(train.get("sinr"), errors="coerce")
    train = train[measured.notna().to_numpy()]
    measured = measured.dropna().to_numpy(float)
    if len(train) < 200 or cell_rows.empty:
        return float(default_cutoff), []
    lat = pd.to_numeric(train["lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(train["lon"], errors="coerce").to_numpy(float)
    sm, cc, ck = _cochannel_score_matrix(lat, lon, cell_rows, candidates)
    trials: list[dict] = []
    for cutoff in CUTOFF_SWEEP_DB:
        scored = _serving_interference(sm, cc, ck, float(cutoff))
        base = np.array([
            _quality_base(s, i)[0]
            for s, i in zip(scored["serving_rsrp_dbm"], scored["interference_sum_mw"])
        ])
        resid = measured - base
        ok = np.isfinite(resid)
        if ok.sum() < 100:
            continue
        shift = float(np.median(resid[ok]))
        trials.append({
            "cutoff_db": int(cutoff),
            "debiased_sinr_mae_db": round(float(np.mean(np.abs(resid[ok] - shift))), 3),
            "median_debias_shift_db": round(shift, 2),
            "median_interferers": float(np.median(scored["interfering_sector_count"])),
        })
    if not trials:
        return float(default_cutoff), []
    trials.sort(key=lambda t: t["cutoff_db"])
    chosen = float(trials[0]["cutoff_db"])
    for a, b in zip(trials, trials[1:]):
        span = b["cutoff_db"] - a["cutoff_db"]
        slope = (a["debiased_sinr_mae_db"] - b["debiased_sinr_mae_db"]) / span if span else 0.0
        if b["cutoff_db"] <= CUTOFF_MIN_DB or slope >= CUTOFF_MARGINAL_GAIN_MIN_DB_PER_DB:
            chosen = float(b["cutoff_db"])
        else:
            break
    for t in trials:
        t["chosen"] = t["cutoff_db"] == chosen
    return chosen, trials


def _grid_quality(technology: str, candidates: pd.DataFrame, inventory: pd.DataFrame, cutoff_db: float) -> pd.DataFrame:
    """Score every technology sector at every grid point for Phase 37 quality.

    The Phase 36 v2 files remain the frozen RSRP correction source, but their
    filtered rows do not decide whether a sector can contribute to Phase 37
    interference. This intentionally has no fixed radius or top-N cap.
    """
    serving_env = pd.read_parquet(PHASE36V2_DIR / f"phase36v2_serving_grid_{technology.lower()}_project210.parquet")
    points = serving_env[["grid_id", "center_lat", "center_lon", "serving_environment"]].drop_duplicates("grid_id").copy()
    points = points.reset_index(drop=True)
    cell_rows = inventory[inventory["technology"].astype(str).eq(technology)].copy().reset_index(drop=True)
    if points.empty or cell_rows.empty:
        return pd.DataFrame()

    lat = pd.to_numeric(points["center_lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(points["center_lon"], errors="coerce").to_numpy(float)
    scored = _score_points_quality(lat, lon, cell_rows, candidates, cutoff_db)
    quality = [_quality_base(s, i) for s, i in zip(scored["serving_rsrp_dbm"], scored["interference_sum_mw"])]
    return pd.DataFrame({
        "technology": technology,
        "grid_id": points["grid_id"].to_numpy(),
        "carrier_key": scored["carrier_key"],
        "lat": lat,
        "lon": lon,
        "serving_strict_cell_key": scored["serving_strict_cell_key"],
        "serving_rsrp_dbm": scored["serving_rsrp_dbm"],
        "serving_environment": points["serving_environment"].to_numpy(),
        "eligible_cochannel_sector_count": scored["eligible_cochannel_sector_count"],
        "interfering_sector_count": scored["interfering_sector_count"],
        "candidate_interfering_sector_count": np.maximum(scored["eligible_cochannel_sector_count"] - 1, 0),
        "interference_sum_dbm": [row[2] for row in quality],
        "sinr_base_db": [row[0] for row in quality],
        "rsrq_base_db": [row[1] for row in quality],
        "interference_sum_mw": scored["interference_sum_mw"],
    })


def _phase37_inventory(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return every sector configuration needed for an exact DT-point score.

    Phase 36 v2 candidate rows are intentionally used only as a frozen correction
    surface. They are not the Phase 37 eligibility set: those rows inherited
    Phase 26's geometry-cost filter and can omit a valid co-channel sector.
    """
    identity = pd.read_parquet(IDENTITY_PATH).copy()
    identity["strict_cell_key"] = identity["Node_Cell_ID"].astype(str)
    source_columns = [
        "strict_cell_key", "technology", "carrier_key", "frequency_mhz",
        "model_rsrp_adjust_db",
    ]
    source = candidates[[column for column in source_columns if column in candidates.columns]].copy()
    source = source.sort_values("strict_cell_key").groupby("strict_cell_key", as_index=False).first()
    inventory = source.merge(identity, on="strict_cell_key", how="inner", validate="one_to_one")
    for column, default in {
        "lat": np.nan,
        "lon": np.nan,
        "azimuth": 0.0,
        "Etilt": 3.0,
        "Mtilt": 0.0,
        "Height": 30.0,
        "tx_power": 46.0,
        "frequency_mhz": np.nan,
        "model_rsrp_adjust_db": 0.0,
    }.items():
        inventory[column] = pd.to_numeric(inventory.get(column), errors="coerce").fillna(default)
    return inventory.dropna(subset=["lat", "lon", "frequency_mhz"]).copy()


def _cell_correction_surface(candidates: pd.DataFrame, cell_key: str) -> tuple[BallTree | None, np.ndarray, np.ndarray, float]:
    """Nearest frozen Phase 36 v2 correction at an exact DT location for one cell."""
    cell = candidates[candidates["strict_cell_key"].astype(str).eq(str(cell_key))].copy()
    correction = (
        pd.to_numeric(cell["phase37_rsrp_dbm"], errors="coerce")
        - pd.to_numeric(cell["raw_cost231_rsrp_unclipped"], errors="coerce")
    )
    cell["phase37_transfer_correction_db"] = correction
    cell = cell.dropna(subset=["lat", "lon", "phase37_transfer_correction_db"])
    if cell.empty:
        return None, np.empty(0), np.empty(0), 0.0
    correction_values = cell["phase37_transfer_correction_db"].to_numpy(float)
    tree = BallTree(np.radians(cell[["lat", "lon"]].to_numpy(float)), metric="haversine")
    return tree, correction_values, cell[["lat", "lon"]].to_numpy(float), float(np.median(correction_values))


def _raw_cost231_at_points(cell: pd.Series, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Exact Phase-9-style COST-231 directional raw score, without a radius/top-N cap."""
    site = {
        "lat": float(cell["lat"]),
        "lon": float(cell["lon"]),
        "azimuth": float(cell["azimuth"]),
        "electrical_tilt": float(cell["Etilt"]),
        "mechanical_tilt": float(cell["Mtilt"]),
        "antenna_height": float(cell["Height"]),
        "tx_power": float(cell["tx_power"]),
        "Node_Cell_ID": str(cell["strict_cell_key"]),
        "frequency_mhz": float(cell["frequency_mhz"]),
    }
    params = {"k1": 0, "k2": 0, "antenna_gain": 18.0, "cable_loss": 2.0, "ue_height": 1.5}
    raw = np.array(
        [compute_sector_rsrp(site, float(y), float(x), float(cell["frequency_mhz"]), params) for y, x in zip(lat, lon)],
        dtype=float,
    )
    return np.minimum(raw, -44.0) + float(cell["model_rsrp_adjust_db"])


def _dt_quality(
    technology: str,
    candidates: pd.DataFrame,
    inventory: pd.DataFrame,
    phase36v2_dt: pd.DataFrame,
    cutoff_db: float,
) -> pd.DataFrame:
    phase26_dt = pd.read_parquet(PHASE26_DIR / "phase26_dt_scored_project210.parquet")
    original_serving = phase26_dt.set_index("dt_row_id")["assigned_strict_cell_key"].astype(str).to_dict()
    dt = phase36v2_dt[phase36v2_dt["technology"].astype(str).eq(technology)].copy().reset_index(drop=True)
    dt["phase37_original_serving_cell"] = dt["assigned_strict_cell_key"].astype(str)
    dt["phase37_original_serving_cell"] = dt["dt_row_id"].map(original_serving).fillna(dt["phase37_original_serving_cell"])

    # SAME scorer as the grid: argmax of the frozen Phase 36 v2 surface, same
    # relative-power interference gate. The DT-trained residual (measured - this
    # base) is therefore a clean transferable delta - it is no longer fit against
    # one interference definition and applied on top of a weaker one.
    cell_rows = inventory[inventory["technology"].astype(str).eq(technology)].copy()
    lat = pd.to_numeric(dt["lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(dt["lon"], errors="coerce").to_numpy(float)
    scored = _score_points_quality(lat, lon, cell_rows, candidates, cutoff_db)
    dt["carrier_key"] = scored["carrier_key"]
    dt["phase37_model_serving_cell"] = scored["serving_strict_cell_key"]
    dt["eligible_cochannel_sector_count"] = scored["eligible_cochannel_sector_count"]
    dt["interfering_sector_count"] = scored["interfering_sector_count"]
    dt["candidate_interfering_sector_count"] = np.maximum(scored["eligible_cochannel_sector_count"] - 1, 0)
    dt["interference_sum_mw"] = scored["interference_sum_mw"]

    signal = scored["serving_rsrp_dbm"]
    interference = scored["interference_sum_mw"]
    quality = [_quality_base(s, i) for s, i in zip(signal, interference)]
    dt["sinr_base_db"] = [row[0] for row in quality]
    dt["rsrq_base_db"] = [row[1] for row in quality]
    dt["interference_sum_dbm"] = [row[2] for row in quality]
    dt["phase37_serving_rsrp_dbm"] = signal
    dt["phase37_calibrated_serving_rsrp_dbm"] = pd.to_numeric(dt["phase36_final_rsrp_unclipped"], errors="coerce")
    dt["rsrq_measured"] = pd.to_numeric(dt["rsrq"], errors="coerce")
    dt["sinr_measured"] = pd.to_numeric(dt["sinr"], errors="coerce")
    dt["serving_model_matches_assigned"] = dt["phase37_model_serving_cell"].astype(str).eq(dt["assigned_strict_cell_key"].astype(str))
    dt["interference_reference"] = (
        "Identical scorer to the grid: argmax of the frozen Phase 36 v2 surface as server, "
        "same-carrier sectors on that surface as interferers, relative-power gated "
        "(4G 15 dB, 5G 24 dB). Server excluded from interference."
    )
    return dt[
        [
            "technology", "dt_row_id", "nearest_grid_id", "carrier_key", "assigned_strict_cell_key", "split", "lat", "lon",
            "phase37_original_serving_cell", "phase37_model_serving_cell", "serving_model_matches_assigned",
            "p36_reassigned", "phase36_final_rsrp_unclipped", "phase37_calibrated_serving_rsrp_dbm",
            "phase37_serving_rsrp_dbm", "rsrq_measured", "sinr_measured",
            "eligible_cochannel_sector_count", "candidate_interfering_sector_count", "interfering_sector_count",
            "interference_sum_dbm", "interference_sum_mw", "sinr_base_db", "rsrq_base_db",
            "interference_reference",
        ]
    ].copy()


def _fit_quality_group(group: pd.DataFrame, technology_for_activity: str) -> dict:
    """Active-interference factor + RSRQ/SINR residual for one calibration group."""
    activity = _fit_interference_activity(group, str(technology_for_activity))
    sinr_base, rsrq_base, _ = _base_columns(group, activity)
    fit_base = group.assign(sinr_fit_base_db=sinr_base, rsrq_fit_base_db=rsrq_base)
    out = {"interference_activity_factor": float(activity), "activity_dt_n": int(len(group))}
    for measured, base, output in [
        ("rsrq_measured", "rsrq_fit_base_db", "rsrq_correction_db"),
        ("sinr_measured", "sinr_fit_base_db", "sinr_correction_db"),
    ]:
        values = pd.to_numeric(fit_base[measured], errors="coerce") - pd.to_numeric(fit_base[base], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        out[output] = float(values.median()) if not values.empty else np.nan
        out[f"{output}_dt_n"] = int(len(values))
    return out


def _fit_quality_calibration(dt: pd.DataFrame) -> pd.DataFrame:
    """Hierarchical quality calibration from the training split only.

    carrier ``(technology, carrier_key)`` -> tech ``(technology)`` -> ``global``.
    Every point resolves at the deepest level with enough DT support, then falls
    through, so RSRQ/SINR coverage matches the RSRP surface (~100%).
    """
    training = dt[dt["split"].astype(str).eq("train")].copy()
    rows: list[dict] = []
    for (technology, carrier_key), group in training.groupby(["technology", "carrier_key"], dropna=False):
        rows.append({"level": "carrier", "technology": technology, "carrier_key": carrier_key,
                     **_fit_quality_group(group, str(technology))})
    for technology, group in training.groupby("technology", dropna=False):
        rows.append({"level": "tech", "technology": technology, "carrier_key": np.nan,
                     **_fit_quality_group(group, str(technology))})
    if not training.empty:
        rows.append({"level": "global", "technology": np.nan, "carrier_key": np.nan,
                     **_fit_quality_group(training, "5G")})
    return pd.DataFrame(rows)


def _apply_quality_calibration(frame: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    n = len(out)
    carrier_map = (calibration[calibration["level"].eq("carrier")]
                   .set_index(["technology", "carrier_key"]).to_dict("index"))
    tech_map = (calibration[calibration["level"].eq("tech")]
                .set_index("technology").to_dict("index"))
    glob = calibration[calibration["level"].eq("global")]
    glob = glob.iloc[0].to_dict() if not glob.empty else {}

    tech_arr = out["technology"].astype(str).to_numpy()
    carr_arr = out["carrier_key"].astype(str).to_numpy()
    activity = np.ones(n, dtype=float)
    rsrq_corr = np.zeros(n, dtype=float)
    sinr_corr = np.zeros(n, dtype=float)
    conf = np.empty(n, dtype=object)

    def _pick(crow, trow, field):
        if crow is not None and np.isfinite(crow.get(field, np.nan)) and crow.get(f"{field}_dt_n", 0) >= CARRIER_MIN_N:
            return float(crow[field]), "carrier"
        if trow is not None and np.isfinite(trow.get(field, np.nan)) and trow.get(f"{field}_dt_n", 0) >= 1:
            return float(trow[field]), "tech"
        if glob and np.isfinite(glob.get(field, np.nan)):
            return float(glob[field]), "global"
        return 0.0, "physics_only"

    for i in range(n):
        tech, carr = tech_arr[i], carr_arr[i]
        crow = carrier_map.get((tech, carr))
        trow = tech_map.get(tech)
        rsrq_corr[i], rlvl = _pick(crow, trow, "rsrq_correction_db")
        sinr_corr[i], slvl = _pick(crow, trow, "sinr_correction_db")
        conf[i] = min((rlvl, slvl), key=lambda lv: _CONF_RANK[lv])
        if crow is not None and crow.get("activity_dt_n", 0) >= CARRIER_MIN_N:
            activity[i] = float(crow["interference_activity_factor"])
        elif trow is not None:
            activity[i] = float(trow["interference_activity_factor"])
        elif glob:
            activity[i] = float(glob.get("interference_activity_factor", 1.0))

    out["interference_activity_factor"] = activity
    out["quality_confidence"] = conf
    out["rsrq_correction_db"] = rsrq_corr
    out["sinr_correction_db"] = sinr_corr
    out["sinr_raw_base_db"] = out["sinr_base_db"]
    out["rsrq_raw_base_db"] = out["rsrq_base_db"]

    sinr_base = np.full(n, np.nan, dtype=float)
    rsrq_base = np.full(n, np.nan, dtype=float)
    active_dbm = np.full(n, np.nan, dtype=float)
    for factor, index in out.groupby("interference_activity_factor", dropna=False).groups.items():
        pos = np.asarray(list(index), dtype=int)
        sinr, rsrq, interference = _base_columns(out.iloc[pos], float(factor))
        sinr_base[pos] = sinr
        rsrq_base[pos] = rsrq
        active_dbm[pos] = interference
    out["sinr_base_db"] = sinr_base
    out["rsrq_base_db"] = rsrq_base
    out["active_interference_sum_dbm"] = active_dbm
    out["pred_rsrq_db"] = rsrq_base + rsrq_corr
    out["pred_sinr_db"] = sinr_base + sinr_corr
    out["quality_status"] = np.select(
        [out["quality_confidence"].isin(["carrier", "tech"]), out["quality_confidence"].eq("global")],
        ["READY_DT_CALIBRATED", "READY_GLOBAL_FALLBACK"],
        default="PHYSICS_ONLY",
    )
    out["rsrq_local_db"] = 0.0
    out["sinr_local_db"] = 0.0
    return out


_QUALITY_KPIS = (
    ("rsrq", "rsrq_measured", "rsrq_base_db", "rsrq_correction_db", "rsrq_local_db", "pred_rsrq_db"),
    ("sinr", "sinr_measured", "sinr_base_db", "sinr_correction_db", "sinr_local_db", "pred_sinr_db"),
)


def _fit_local_quality_fields(dt_scored: pd.DataFrame) -> dict:
    """Per-technology inverse-distance DT-residual field for RSRQ and SINR.

    Fit on the residual left after base + hierarchical median, training split
    only. Same construction as the Phase 25 / RSRP v2 local field.
    """
    train = dt_scored[dt_scored["split"].astype(str).eq("train")].copy()
    models: dict = {}
    for tech, grp in train.groupby("technology"):
        grp = grp.dropna(subset=["lat", "lon"])
        if len(grp) < LOCAL_MIN_NEIGHBORS + 1:
            continue
        xy = np.radians(grp[["lat", "lon"]].to_numpy(float))
        tree = BallTree(xy, metric="haversine")
        d, _ = tree.query(xy, k=2)
        near = d[:, 1][np.isfinite(d[:, 1]) & (d[:, 1] > 0)]
        radius = float(max(np.quantile(near, 0.75) * 3.0, 150.0 / EARTH_RADIUS_M)) if near.size else (300.0 / EARTH_RADIUS_M)
        tech_model = {"xy": xy, "tree": tree, "radius": radius, "n": int(len(grp))}
        for kpi, measured, base, corr, _local, _pred in _QUALITY_KPIS:
            resid = (pd.to_numeric(grp[measured], errors="coerce")
                     - pd.to_numeric(grp[base], errors="coerce")
                     - pd.to_numeric(grp[corr], errors="coerce")).to_numpy(float)
            tech_model[kpi] = np.where(np.isfinite(resid), resid, 0.0)
        models[str(tech)] = tech_model
    return models


def _apply_local_quality_field(frame: pd.DataFrame, models: dict) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    n = len(out)
    tech = out["technology"].astype(str).to_numpy()
    lat = pd.to_numeric(out["lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(out["lon"], errors="coerce").to_numpy(float)
    support_any = np.zeros(n, dtype=int)
    for kpi, _measured, base, corr, local_col, pred_col in _QUALITY_KPIS:
        local = np.zeros(n, dtype=float)
        for t, model in models.items():
            idx = np.where((tech == t) & np.isfinite(lat) & np.isfinite(lon))[0]
            if idx.size == 0:
                continue
            pts = np.radians(np.column_stack([lat[idx], lon[idx]]))
            k = min(LOCAL_K_NEIGHBORS, model["n"])
            dist, near = model["tree"].query(pts, k=k)
            if k == 1:
                dist = dist[:, None]; near = near[:, None]
            valid = np.isfinite(dist) & (dist <= model["radius"]) & (near < model["n"])
            support = valid.sum(axis=1)
            resid = model[kpi]
            for j in np.where(support >= LOCAL_MIN_NEIGHBORS)[0]:
                vj = valid[j]
                d_m = dist[j, vj] * EARTH_RADIUS_M
                w = 1.0 / np.maximum(d_m, 5.0)
                val = float(np.average(resid[near[j, vj]], weights=w))
                local[idx[j]] = val * (support[j] / (support[j] + LOCAL_SHRINK_N))
                support_any[idx[j]] = max(support_any[idx[j]], int(support[j]))
        out[local_col] = local
        pred = (pd.to_numeric(out[base], errors="coerce")
                + pd.to_numeric(out[corr], errors="coerce") + local)
        lo, hi = (-19.5, -3.0) if kpi == "rsrq" else (-20.0, 35.0)  # 3GPP reporting ranges
        out[pred_col] = pred.clip(lo, hi)
    out["quality_local_support_n"] = support_any
    out["quality_confidence"] = np.where(support_any >= LOCAL_MIN_NEIGHBORS, "local", out["quality_confidence"].astype(str))
    return out


def _mae(frame: pd.DataFrame, measured: str, predicted: str) -> float | None:
    diff = pd.to_numeric(frame[measured], errors="coerce") - pd.to_numeric(frame[predicted], errors="coerce")
    diff = diff.replace([np.inf, -np.inf], np.nan).dropna()
    return None if diff.empty else round(float(diff.abs().mean()), 3)


def _finite_frac(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return round(float(pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().mean()), 4)


def _quantiles(frame: pd.DataFrame, column: str, points: list[int]) -> list[float | None]:
    if frame.empty or column not in frame.columns:
        return [None] * len(points)
    series = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return [None] * len(points)
    return [round(float(series.quantile(q / 100.0)), 2) for q in points]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.concat([_phase37_candidates("4G"), _phase37_candidates("5G")], ignore_index=True)
    inventory = _phase37_inventory(candidates)
    phase36v2_dt = _phase36v2_dt_all()

    interferer_cutoff_db: dict = {}
    interferer_cutoff_sweep: dict = {}
    for technology in ("4G", "5G"):
        default = (NR_INTERFERER_RELATIVE_CUTOFF_DB_DEFAULT if technology == "5G"
                   else LTE_INTERFERER_RELATIVE_CUTOFF_DB_DEFAULT)
        cut, trials = _fit_interferer_cutoff(
            phase36v2_dt[phase36v2_dt["technology"].astype(str).eq(technology)],
            inventory[inventory["technology"].astype(str).eq(technology)],
            candidates, default,
        )
        interferer_cutoff_db[technology] = cut
        interferer_cutoff_sweep[technology] = trials
        print(f"[{technology}] fitted interferer cutoff = {cut:.0f} dB "
              f"(default seed {default:.0f}) from {len(trials)} sweep points")

    grid_frames = []
    dt_frames = []
    for technology in ("4G", "5G"):
        tech_candidates = candidates[candidates["technology"].eq(technology)].copy()
        cut = interferer_cutoff_db[technology]
        grid_frames.append(_grid_quality(technology, tech_candidates, inventory, cut))
        dt_frames.append(_dt_quality(technology, tech_candidates, inventory, phase36v2_dt, cut))
    grid = pd.concat(grid_frames, ignore_index=True)
    dt = pd.concat(dt_frames, ignore_index=True)
    calibration = _fit_quality_calibration(dt)
    dt = _apply_quality_calibration(dt, calibration)
    local_models = _fit_local_quality_fields(dt)
    dt = _apply_local_quality_field(dt, local_models)
    grid = _apply_quality_calibration(grid, calibration)
    grid = _apply_local_quality_field(grid, local_models)

    grid.to_parquet(OUT_DIR / "phase37_serving_quality_project210.parquet", index=False)
    grid.to_csv(OUT_DIR / "phase37_serving_quality_project210.csv", index=False)
    dt.to_parquet(OUT_DIR / "phase37_dt_quality_project210.parquet", index=False)
    calibration.to_csv(OUT_DIR / "phase37_carrier_quality_calibration.csv", index=False)
    for technology in ("4G", "5G"):
        part = dt[dt["technology"].eq(technology)].copy()
        part.to_parquet(OUT_DIR / f"phase37_dt_quality_{technology.lower()}_project210.parquet", index=False)
        part.to_csv(OUT_DIR / f"phase37_dt_quality_{technology.lower()}_project210.csv", index=False)

    validation = dt[dt["split"].astype(str).eq("validation")]

    _QS = [5, 10, 25, 50, 75, 90, 95]
    quality_cdf: dict = {}
    for technology in ("4G", "5G"):
        gpart = grid[grid["technology"].eq(technology)]
        vpart = validation[validation["technology"].eq(technology)]
        entry: dict = {
            "grid_rows": int(len(gpart)),
            "grid_pred_rsrq_coverage": _finite_frac(gpart, "pred_rsrq_db"),
            "grid_pred_sinr_coverage": _finite_frac(gpart, "pred_sinr_db"),
            "quality_confidence": {k: int(v) for k, v in gpart["quality_confidence"].value_counts().items()},
            "validation_dt_rows": int(len(vpart)),
            "rsrq_validation_mae_db": _mae(vpart, "rsrq_measured", "pred_rsrq_db"),
            "sinr_validation_mae_db": _mae(vpart, "sinr_measured", "pred_sinr_db"),
        }
        for kpi, meas, pred in (("rsrq", "rsrq_measured", "pred_rsrq_db"), ("sinr", "sinr_measured", "pred_sinr_db")):
            entry[f"{kpi}_cdf"] = {
                "grid_predicted": _quantiles(gpart, pred, _QS),
                "dt_measured": _quantiles(vpart, meas, _QS),
                "dt_predicted": _quantiles(vpart, pred, _QS),
            }
        quality_cdf[technology] = entry

    print("\n=== Phase 37 quality CDF (held-out validation) ===")
    for technology, entry in quality_cdf.items():
        print(f"\n[{technology}]  grid rsrq/sinr coverage "
              f"{entry['grid_pred_rsrq_coverage']*100:.1f}% / {entry['grid_pred_sinr_coverage']*100:.1f}%  "
              f"| confidence {entry['quality_confidence']}")
        print(f"  RSRQ MAE {entry['rsrq_validation_mae_db']}  SINR MAE {entry['sinr_validation_mae_db']}  "
              f"(n={entry['validation_dt_rows']})")
        for kpi in ("rsrq", "sinr"):
            cdf = entry[f"{kpi}_cdf"]
            print(f"  {kpi.upper():4} p:      " + "  ".join(f"{q:>3}" for q in _QS))
            print(f"       measured : " + "  ".join(f"{v:>5.1f}" if v is not None else "  n/a" for v in cdf["dt_measured"]))
            print(f"       predicted: " + "  ".join(f"{v:>5.1f}" if v is not None else "  n/a" for v in cdf["dt_predicted"]))

    carrier_summary = []
    for (technology, carrier_key), group in grid.groupby(["technology", "carrier_key"], dropna=False):
        val = validation[(validation["technology"].eq(technology)) & (validation["carrier_key"].eq(carrier_key))]
        carrier_summary.append({
            "technology": technology,
            "carrier_key": carrier_key,
            "grids": int(len(group)),
            "quality_status": "; ".join(sorted(group["quality_status"].unique())),
            "median_eligible_cochannel_sectors": float(group["eligible_cochannel_sector_count"].median()),
            "median_active_interfering_sectors": float(group["interfering_sector_count"].median()),
            "max_eligible_cochannel_sectors": int(group["eligible_cochannel_sector_count"].max()),
            "interference_activity_factor": float(pd.to_numeric(group["interference_activity_factor"], errors="coerce").median()),
            "rsrq_validation_mae_db": _mae(val, "rsrq_measured", "pred_rsrq_db"),
            "sinr_validation_mae_db": _mae(val, "sinr_measured", "pred_sinr_db"),
        })
    summary = {
        "scope": "Phase 37 rebased in place onto Phase 36 v2 RSRP. Earlier phase files/outputs are untouched.",
        "interference_policy": "serving = argmax of the frozen Phase 36 v2 surface, excluded from interference; "
                               "co-channel = same-carrier sectors on that surface, gated at signal - cutoff where "
                               "the cutoff is FIT PER TECHNOLOGY FROM THE DT (no per-project constant).",
        "quality_model": "RSRQ/SINR base uses a -104 dBm thermal-noise floor and a DT-fitted relative-power interferer "
                         "cutoff so co-channel geometry survives; then a hierarchical DT residual (carrier -> technology "
                         "-> global) plus a Phase-25-style local inverse-distance DT-residual field - the same recipe as "
                         "the RSRP v2 surface. Everything is fit from this project's cells + drive test; only physics/3GPP "
                         "constants are fixed. Training split only, validation held out. Predictions clipped to the 3GPP "
                         "reporting ranges (RSRQ -19.5..-3, SINR -20..35).",
        "interferer_cutoff_db": interferer_cutoff_db,
        "interferer_cutoff_sweep": interferer_cutoff_sweep,
        "quality_cdf": quality_cdf,
        "dt_reference": "Measured RSRQ/SINR joined from Phase26 cached DT by dt_row_id. "
                        "For 5G, serving cell is cleaned with Phase36's reassignment rule before quality scoring.",
        "rsrp_base": "Phase 36 v2 candidates/serving grid, with full Phase 36 v2 DT scoring rebuilt in memory for train+validation.",
        "carrier_summary": carrier_summary,
        "grid_rows": int(len(grid)),
        "dt_rows": int(len(dt)),
        "training_dt_rows": int(dt[dt["split"].astype(str).eq("train")].shape[0]),
        "validation_dt_rows": int(validation.shape[0]),
        "five_g_dt_reassigned": int(dt[(dt["technology"].eq("5G")) & (dt["p36_reassigned"])].shape[0]),
    }
    (OUT_DIR / "phase37_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
