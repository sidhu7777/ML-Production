import math
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from shapely.geometry import Point
from shapely.ops import transform, unary_union
from shapely.wkt import loads as load_wkt
from sklearn.neighbors import BallTree
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError, OperationalError

from extensions import db

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

from tools.lte_prediction.Sector_wise_prediction_code_copy import compute_sector_rsrp
from tools.lte_prediction.grid_sampling import fetch_frontend_grid_cells
from tools.lte_prediction.ml_engine import (
    _resolve_prediction_polygons,
    engine,
    fetch_building_data,
    fetch_drive_data,
    fetch_site_data,
)
from tools.lte_prediction.services import LTEPredictionService
from tools.lte_prediction.dem_utils import ensure_project_dem
from tools.lte_prediction_offset.phase27_calibration import add_features, apply_outdoor, fit_outdoor
from tools.lte_prediction_offset.phase27_physical import _DemSampler, score_candidates
from tools.lte_prediction_offset import phase27_calibration as _calib
from tools.lte_prediction_offset import phase36_physical_upgrades as _p36
from tools.lte_prediction_offset import phase37_quality as _p37
from tools.lte_prediction_offset.geo_inputs import load_or_build_phase27_clutter
from utils.python_bridge import PythonBridgeError, get_bridge_client


JOBS = {}
EARTH_RADIUS_M = 6371000.0
CLIP_RSRP = (-140.0, -44.0)
BRIDGE_ENV_KEYS = ("PYTHON_BRIDGE_BASE_URL", "SIGNAL_TRACKERS_BRIDGE_URL")
_SAVE_ENGINES = {}
ML_ROOT = Path(__file__).resolve().parents[2]

# --- TEMPORARY: fixed tx_power override (Phase 39 equal-power) ---------------
# Per-cell tx_power fetched from the antenna table is not trusted right now.
# Until real per-cell power is confirmed, every cell (4G and 5G) is forced to
# this value instead of its fetched tx_power, matching Phase 39's equal-power
# diagnostic assumption. Remove this override (set to None) once real per-cell
# tx_power is validated and should drive the RF math again.
FIXED_TX_POWER_DBM_OVERRIDE = 46.0

# --- Phase 39: dynamic COST-231 frequency-anchor offset -----------------------
# COST-231/Hata is only calibrated at specific reference frequencies. When a
# cell's real/deployed frequency is not one of those, COST231 is run at the
# nearest valid anchor instead and the output is corrected back with the
# COST-231 frequency term: offset_db = -COST231_FREQ_COEFF * log10(f_real/f_anchor).
# Anchors are the frequencies validated so far; add to this list (not a new
# hardcoded band branch) as more get validated against drive test.
COST231_FREQ_COEFF = 33.9
COST231_VALID_ANCHORS_MHZ = (1500.0, 2600.0)
# Real deployed frequency for (technology, band, region) combinations where the
# antenna table's label does not match the true physical frequency (e.g.
# Taiwan's 5G N78 is labelled/stored as 2600 MHz but is actually deployed at
# 3300 MHz). Anything not listed here uses _frequency_from_site's value as-is
# (label already equals real for that band).
COST231_REAL_FREQUENCY_OVERRIDE_MHZ = {
    ("5G", "78", "taiwan"): 3300.0,
}


def _cost231_resolve_anchor_mhz(freq_mhz: pd.Series) -> pd.Series:
    """COST-231 is treated as directly valid across [min(anchors), max(anchors)]
    (today 1500-2600 MHz) -- a frequency inside that window runs at its own
    value, unchanged, exactly like before this function existed. Only a
    frequency outside the window is clamped to the nearest edge anchor, with
    _cost231_frequency_offset_db correcting the model output back to what
    freq_mhz itself should read."""
    lo, hi = min(COST231_VALID_ANCHORS_MHZ), max(COST231_VALID_ANCHORS_MHZ)
    freq = pd.to_numeric(freq_mhz, errors="coerce")
    return freq.clip(lower=lo, upper=hi)


def _cost231_frequency_offset_db(freq_real_mhz, freq_anchor_mhz) -> pd.Series:
    ratio = pd.to_numeric(freq_real_mhz, errors="coerce") / pd.to_numeric(freq_anchor_mhz, errors="coerce")
    return -COST231_FREQ_COEFF * np.log10(ratio)


def _with_db_retry(db_engine, action, stage: str):
    """Reconnect on transient remote-MySQL failures, without masking data errors."""
    for attempt in range(1, 4):
        try:
            return action()
        except (OperationalError, DBAPIError):
            if attempt == 3:
                raise
            if db_engine is not None:
                db_engine.dispose()
            delay_s = attempt * 2
            print(f"[LTE_OFFSET][DB_RETRY] stage={stage} attempt={attempt}/3 delay_s={delay_s}", flush=True)
            time.sleep(delay_s)


@contextmanager
def _without_python_bridge():
    old_values = {key: os.environ.get(key) for key in BRIDGE_ENV_KEYS}
    for key in BRIDGE_ENV_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _clean_text(series):
    text = series.astype("string").str.strip()
    lowered = text.str.lower()
    invalid = text.isna() | text.eq("") | lowered.isin({"nan", "none", "null", "<na>", "undefined"})
    return text.mask(invalid)


def _first_present(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _technology_from_site(site_df):
    tech = pd.Series(pd.NA, index=site_df.index, dtype="string")
    for col in ["Technology", "technology", "network_type", "rat", "tech"]:
        if col in site_df.columns:
            tech = tech.fillna(_clean_text(site_df[col]))
    band = _clean_text(site_df.get("band", pd.Series(index=site_df.index))).astype("string")
    text = tech.astype("string").str.upper()
    text = text.mask(text.str.contains("5G|NR", na=False), "5G")
    text = text.mask(text.str.contains("4G|LTE", na=False), "4G")
    text = text.mask(text.isna() & band.eq("78"), "5G")
    text = text.mask(text.isna(), "4G")
    return text.fillna("4G")


def _frequency_from_site(site_df):
    freq = pd.Series(np.nan, index=site_df.index, dtype=float)
    for col in ["frequency_mhz", "frequency", "downlink_frequency", "band"]:
        if col not in site_df.columns:
            continue
        candidate = pd.to_numeric(site_df[col], errors="coerce")
        freq = freq.where(pd.notna(freq), candidate)
    return freq.fillna(1800.0).clip(450.0, 3800.0)


def _prepare_site_rows(site_df, region):
    out = site_df.copy()
    if "Site ID" in out.columns and "site" not in out.columns:
        out["site"] = out["Site ID"]
    if "nodeb_id" not in out.columns:
        out["nodeb_id"] = out.get("site", out.get("Site ID", pd.Series(index=out.index)))

    _am_col = _first_present(out, ["antenna_model", "antenna_type", "antenna", "antenna_name"])
    out["antenna_model"] = _clean_text(out[_am_col]) if _am_col else pd.Series(pd.NA, index=out.index, dtype="string")

    out["strict_cell_key"] = _clean_text(out.get("Node_Cell_ID", pd.Series(index=out.index)))
    if "rf_identity_key" in out.columns:
        out["strict_cell_key"] = out["strict_cell_key"].fillna(_clean_text(out["rf_identity_key"]))
    if "cell_id" in out.columns:
        out["strict_cell_key"] = out["strict_cell_key"].fillna(_clean_text(out["cell_id"]))

    out["original_cell_id"] = _clean_text(out.get("legacy_nodeb_id_cell_id", pd.Series(index=out.index)))
    out["original_cell_id"] = out["original_cell_id"].fillna(_clean_text(out.get("cell_id", pd.Series(index=out.index))))
    out["site_key"] = _clean_text(out.get("site", out.get("Site ID", pd.Series(index=out.index)))).fillna("unknown-site")
    out["sector_key"] = _clean_text(out.get("sector", pd.Series(index=out.index))).fillna("unknown-sector")
    out["band_key"] = _clean_text(out.get("band", pd.Series(index=out.index))).fillna("unknown-band")
    out["technology_key"] = _technology_from_site(out)

    operator_col = _first_present(out, ["operator", "network", "cluster", "provider", "operator_name"])
    if operator_col:
        out["operator_key"] = _clean_text(out[operator_col]).fillna("unknown-operator")
    else:
        out["operator_key"] = "unknown-operator"

    out["site_sector_band_key"] = (
        out["site_key"].astype(str)
        + "|"
        + out["sector_key"].astype(str)
        + "|"
        + out["band_key"].astype(str)
    )
    out["sector_identity_key"] = (
        out["site_key"].astype(str) + "|" + out["original_cell_id"].astype(str) + "|" + out["sector_key"].astype(str)
    )

    for col, default in {
        "lat": None,
        "lon": None,
        "azimuth": 0.0,
        "Height": 30.0,
        "Etilt": 3.0,
        "Mtilt": 0.0,
        "tx_power": 46.0,
    }.items():
        out[col] = pd.to_numeric(out.get(col, pd.Series(index=out.index)), errors="coerce")
        if default is not None:
            out[col] = out[col].fillna(default)

    # Keep the real per-cell power BEFORE any equal-power override. The DT-side
    # normalisation needs it to shift measured RSRP into the same equal-power
    # space as the model (see _equal_power_shift_for_dt).
    out["original_tx_power_dbm"] = out["tx_power"]
    if FIXED_TX_POWER_DBM_OVERRIDE is not None:
        # TEMPORARY -- see FIXED_TX_POWER_DBM_OVERRIDE docstring above.
        out["tx_power"] = float(FIXED_TX_POWER_DBM_OVERRIDE)

    out["frequency_mhz"] = _frequency_from_site(out)
    out["original_frequency_mhz"] = out["frequency_mhz"]

    # Phase 39: real deployed frequency, for the (technology, band, region)
    # combinations where the antenna table's label doesn't match reality.
    freq_real = out["frequency_mhz"].copy()
    for (tech, band, override_region), real_mhz in COST231_REAL_FREQUENCY_OVERRIDE_MHZ.items():
        mask = (
            out["technology_key"].astype(str).eq(tech)
            & out["band_key"].astype(str).eq(band)
            & (str(region).lower() == str(override_region).lower())
        )
        freq_real.loc[mask] = real_mhz

    # Run COST231 at whichever frequency is valid (own frequency if already
    # inside the calibrated window, else the nearest edge anchor), and correct
    # the output back to freq_real with the analytic COST-231 frequency term.
    freq_anchor = _cost231_resolve_anchor_mhz(freq_real)
    out["frequency_mhz"] = freq_anchor
    out["model_rsrp_adjust_db"] = _cost231_frequency_offset_db(freq_real, freq_anchor).fillna(0.0)

    out = out.dropna(subset=["lat", "lon", "strict_cell_key"]).copy()
    return out.drop_duplicates(subset=["strict_cell_key"], keep="first").reset_index(drop=True)


def _site_record(row):
    return {
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "azimuth": float(row["azimuth"]),
        "electrical_tilt": float(row["Etilt"]),
        "mechanical_tilt": float(row["Mtilt"]),
        "antenna_height": float(row["Height"]),
        "tx_power": float(row["tx_power"]),
        "Node_Cell_ID": str(row["strict_cell_key"]),
        "frequency_mhz": float(row["frequency_mhz"]),
    }


def _cost231_for_points(site, lat_values, lon_values, freq_mhz):
    params = {"k1": 0, "k2": 0, "antenna_gain": 18.0, "cable_loss": 2.0, "ue_height": 1.5}
    # compute_sector_rsrp is fully vectorised internally (haversine_vectorized,
    # compute_bearing_vectorized, compute_3gpp_antenna_gain_vectorized, numpy
    # throughout), so score every point in ONE call. The previous per-point
    # list comprehension drove a vectorised function one scalar at a time and
    # was ~99% of every RF evaluation's runtime. Same math, same result.
    values = compute_sector_rsrp(
        site,
        np.asarray(lat_values, dtype=float),
        np.asarray(lon_values, dtype=float),
        float(freq_mhz),
        params,
    )
    # Keep the physical value below the display/no-coverage threshold.  A
    # candidate must not disappear before obstruction and calibration are
    # evaluated; only the upper display ceiling is applied here.
    return np.minimum(np.asarray(values, dtype=float), CLIP_RSRP[1])


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def _azimuth_delta_deg(bearing, azimuth):
    return np.abs((bearing - float(azimuth) + 180.0) % 360.0 - 180.0)


def _polygon_grid(polygons, grid_size_meters):
    valid_polygons = [poly for poly in polygons or [] if poly is not None and not poly.is_empty]
    if not valid_polygons:
        return pd.DataFrame()

    union = unary_union(valid_polygons)
    if union.is_empty:
        return pd.DataFrame()

    min_lon, min_lat, max_lon, max_lat = union.bounds
    center_lat = (min_lat + max_lat) / 2.0
    lat_step = float(grid_size_meters or 25.0) / 111320.0
    lon_step = float(grid_size_meters or 25.0) / (111320.0 * max(math.cos(math.radians(center_lat)), 1e-6))
    rows = int(math.ceil((max_lat - min_lat) / lat_step))
    cols = int(math.ceil((max_lon - min_lon) / lon_step))

    records = []
    for row in range(rows):
        cell_min_lat = min_lat + row * lat_step
        cell_max_lat = cell_min_lat + lat_step
        center_cell_lat = (cell_min_lat + cell_max_lat) / 2.0
        for col in range(cols):
            cell_min_lon = min_lon + col * lon_step
            cell_max_lon = cell_min_lon + lon_step
            center_cell_lon = (cell_min_lon + cell_max_lon) / 2.0
            if not union.covers(Point(center_cell_lon, center_cell_lat)):
                continue
            records.append(
                {
                    "grid_id": f"R{row}C{col}",
                    "center_lat": round(center_cell_lat, 8),
                    "center_lon": round(center_cell_lon, 8),
                    "min_lat": round(cell_min_lat, 8),
                    "max_lat": round(cell_max_lat, 8),
                    "min_lon": round(cell_min_lon, 8),
                    "max_lon": round(cell_max_lon, 8),
                    "grid_size_meters": float(grid_size_meters or 25.0),
                    "scenario_id": pd.NA,
                }
            )

    out = pd.DataFrame.from_records(records)
    print(
        f"[LTE_OFFSET][POLYGON_GRID] rows={len(out)} grid_size_meters={float(grid_size_meters or 25.0)}",
        flush=True,
    )
    return out


def _grid_from_bridge_or_db(cfg, current_engine, polygons, force_direct_db=False):
    polygon_grid = _polygon_grid(polygons, cfg.get("grid_resolution") or cfg.get("frontend_grid_size_meters") or 25.0)
    if not polygon_grid.empty:
        return polygon_grid.sort_values("grid_id").reset_index(drop=True)

    bridge = get_bridge_client()
    grid_df = pd.DataFrame()
    if bridge and not force_direct_db:
        try:
            grid_df, _ = bridge.get_grid_analytics(
                int(cfg["project_id"]),
                scenario_id=cfg.get("grid_analytics_scenario_id"),
                auth_header=cfg.get("grid_analytics_auth_header"),
                cookie_header=cfg.get("grid_analytics_cookie_header"),
                region=cfg.get("region"),
                country_code=cfg.get("country_code"),
            )
        except PythonBridgeError as exc:
            print(f"[LTE_OFFSET][GRID_FETCH] bridge_failed={exc}", flush=True)

    if grid_df.empty:
        grid_df, _ = fetch_frontend_grid_cells(
            current_engine,
            int(cfg["project_id"]),
            scenario_id=cfg.get("grid_analytics_scenario_id"),
            grid_size_meters=cfg.get("frontend_grid_size_meters"),
        )

    required = ["grid_id", "center_lat", "center_lon"]
    if grid_df.empty:
        grid_df = _polygon_grid(polygons, cfg.get("grid_resolution") or cfg.get("frontend_grid_size_meters") or 25.0)

    missing = [col for col in required if col not in grid_df.columns]
    if missing:
        raise ValueError(f"Grid Analytics data missing columns: {missing}")
    grid_df = grid_df.copy()
    for col in ["center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon", "grid_size_meters"]:
        if col in grid_df.columns:
            grid_df[col] = pd.to_numeric(grid_df[col], errors="coerce")
    grid_df = grid_df.dropna(subset=["grid_id", "center_lat", "center_lon"]).drop_duplicates("grid_id").copy()

    if polygons:
        union = unary_union(polygons)
        mask = [union.covers(Point(lon, lat)) for lat, lon in grid_df[["center_lat", "center_lon"]].to_numpy()]
        filtered = grid_df.loc[mask].copy()
        if not filtered.empty:
            grid_df = filtered
    if grid_df.empty:
        raise ValueError("No grid pixels available for offset LTE prediction")
    return grid_df.sort_values("grid_id").reset_index(drop=True)


def _surface_frame_for_site(
    row,
    candidate_grid,
    raw_values,
    distance_values,
    bearing_values,
    az_delta_values,
    ensure_all_cells_backfill=False,
):
    candidate_lat = candidate_grid["center_lat"].to_numpy(dtype=float)
    candidate_lon = candidate_grid["center_lon"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "grid_id": candidate_grid["grid_id"].astype(str).to_numpy(),
            "lat": candidate_lat,
            "lon": candidate_lon,
            "Node_Cell_ID": str(row["strict_cell_key"]),
            "node_cell_id": str(row["strict_cell_key"]),
            "strict_cell_key": str(row["strict_cell_key"]),
            "site": str(row["site_key"]),
            "nodeb_id": str(row.get("nodeb_id", row["site_key"])),
            "cell_id": str(row["original_cell_id"]),
            "sector": str(row["sector_key"]),
            "band": str(row["band_key"]),
            "Technology": str(row["technology_key"]),
            "operator": str(row["operator_key"]),
            "rf_identity_key": str(row["strict_cell_key"]),
            "sector_identity_key": str(row["sector_identity_key"]),
            "site_sector_band_key": str(row["site_sector_band_key"]),
            "legacy_nodeb_id_cell_id": str(row["original_cell_id"]),
            "serving_frequency_mhz": float(row["frequency_mhz"]),
            "original_frequency_mhz": float(row.get("original_frequency_mhz", row["frequency_mhz"])),
            "model_rsrp_adjust_db": float(row.get("model_rsrp_adjust_db", 0.0)),
            "distance_m": np.asarray(distance_values, dtype=float),
            "bearing_deg": np.asarray(bearing_values, dtype=float),
            "azimuth_delta_deg": np.asarray(az_delta_values, dtype=float),
            "raw_cost231_rsrp": np.asarray(raw_values, dtype=float),
            "grid_min_lat": pd.to_numeric(candidate_grid.get("min_lat", pd.Series(np.nan, index=candidate_grid.index)), errors="coerce").to_numpy(dtype=float),
            "grid_max_lat": pd.to_numeric(candidate_grid.get("max_lat", pd.Series(np.nan, index=candidate_grid.index)), errors="coerce").to_numpy(dtype=float),
            "grid_min_lon": pd.to_numeric(candidate_grid.get("min_lon", pd.Series(np.nan, index=candidate_grid.index)), errors="coerce").to_numpy(dtype=float),
            "grid_max_lon": pd.to_numeric(candidate_grid.get("max_lon", pd.Series(np.nan, index=candidate_grid.index)), errors="coerce").to_numpy(dtype=float),
            "ensure_all_cells_backfill": bool(ensure_all_cells_backfill),
        }
    )


def _nearest_backfill_rows(site_df, grid_row, k_nearest):
    lat = float(grid_row["center_lat"])
    lon = float(grid_row["center_lon"])
    candidates = []
    for _, row in site_df.iterrows():
        raw = _cost231_for_points(
            _site_record(row),
            np.asarray([lat], dtype=float),
            np.asarray([lon], dtype=float),
            float(row["frequency_mhz"]),
        )[0]
        raw = float(raw + float(row.get("model_rsrp_adjust_db", 0.0)))
        bearing = float(_bearing_deg(float(row["lat"]), float(row["lon"]), np.asarray([lat]), np.asarray([lon]))[0])
        distance = float(_haversine_m(float(row["lat"]), float(row["lon"]), lat, lon))
        candidates.append((distance, row, raw, bearing, float(_azimuth_delta_deg(np.asarray([bearing]), float(row["azimuth"]))[0])))

    if not candidates:
        return pd.DataFrame()
    grid_one = pd.DataFrame([grid_row])
    frames = []
    for distance, row, raw, bearing, delta in sorted(candidates, key=lambda item: item[0])[: max(1, int(k_nearest))]:
        frames.append(_surface_frame_for_site(row, grid_one, [raw], [distance], [bearing], [delta], ensure_all_cells_backfill=True))
    return pd.concat(frames, ignore_index=True)


def _grid_id_row_col(series):
    extracted = series.astype(str).str.extract(r"R(?P<row>\d+)C(?P<col>\d+)")
    return pd.to_numeric(extracted["row"], errors="coerce"), pd.to_numeric(extracted["col"], errors="coerce")


def _attach_gridanalytics_bucket_coords(surface):
    required = {"grid_id", "lat", "lon", "grid_min_lat", "grid_max_lat", "grid_min_lon", "grid_max_lon"}
    if surface.empty or not required.issubset(surface.columns):
        surface["analytics_lat"] = surface.get("lat", pd.Series(index=surface.index))
        surface["analytics_lon"] = surface.get("lon", pd.Series(index=surface.index))
        return surface

    grid_coords = (
        surface[
            [
                "grid_id",
                "lat",
                "lon",
                "grid_min_lat",
                "grid_max_lat",
                "grid_min_lon",
                "grid_max_lon",
            ]
        ]
        .drop_duplicates("grid_id")
        .copy()
    )
    grid_coords["analytics_lat"] = pd.to_numeric(grid_coords["lat"], errors="coerce")
    grid_coords["analytics_lon"] = pd.to_numeric(grid_coords["lon"], errors="coerce")
    grid_coords["grid_row"], grid_coords["grid_col"] = _grid_id_row_col(grid_coords["grid_id"])

    eps = 1e-9
    if grid_coords["grid_row"].notna().any():
        south_idx = grid_coords["grid_row"].idxmin()
        north_idx = grid_coords["grid_row"].idxmax()
        grid_coords.loc[south_idx, "analytics_lat"] = float(grid_coords.loc[south_idx, "grid_min_lat"]) + eps
        grid_coords.loc[north_idx, "analytics_lat"] = float(grid_coords.loc[north_idx, "grid_max_lat"]) - eps
    if grid_coords["grid_col"].notna().any():
        west_idx = grid_coords["grid_col"].idxmin()
        east_idx = grid_coords["grid_col"].idxmax()
        grid_coords.loc[west_idx, "analytics_lon"] = float(grid_coords.loc[west_idx, "grid_min_lon"]) + eps
        grid_coords.loc[east_idx, "analytics_lon"] = float(grid_coords.loc[east_idx, "grid_max_lon"]) - eps

    return surface.merge(
        grid_coords[["grid_id", "analytics_lat", "analytics_lon"]],
        on="grid_id",
        how="left",
    )


def _run_raw_surface(site_df, grid_df, cfg=None):
    cfg = cfg or {}
    grid_lat = grid_df["center_lat"].to_numpy(dtype=float)
    grid_lon = grid_df["center_lon"].to_numpy(dtype=float)
    frames = []
    total = len(site_df)
    radius_m = float(cfg.get("radius_m") or cfg.get("coverage_radius_m") or 500.0)
    backfill_k_nearest = max(1, int(cfg.get("out_of_radius_backfill_k_nearest", 8)))
    for idx, row in site_df.iterrows():
        distance_m = _haversine_m(float(row["lat"]), float(row["lon"]), grid_lat, grid_lon)
        candidate_pre = distance_m <= radius_m
        if not candidate_pre.any():
            if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == total:
                print(f"[LTE_OFFSET][COST231_DIRECTIONAL] cells_done={idx + 1}/{total} rows_so_far={sum(len(f) for f in frames)}", flush=True)
            continue

        raw = _cost231_for_points(
            _site_record(row),
            grid_lat[candidate_pre],
            grid_lon[candidate_pre],
            float(row["frequency_mhz"]),
        )
        raw = raw + float(row.get("model_rsrp_adjust_db", 0.0))
        # Distance is the candidate eligibility criterion. Do not use a raw
        # RSRP cutoff here: later physical layers may validly change a weak
        # raw candidate into the serving sector.
        candidate = np.isfinite(raw)
        if not candidate.any():
            if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == total:
                print(f"[LTE_OFFSET][COST231_DIRECTIONAL] cells_done={idx + 1}/{total} rows_so_far={sum(len(f) for f in frames)}", flush=True)
            continue

        candidate_grid = grid_df.loc[candidate_pre].iloc[np.flatnonzero(candidate)].copy()
        candidate_lat = candidate_grid["center_lat"].to_numpy(dtype=float)
        candidate_lon = candidate_grid["center_lon"].to_numpy(dtype=float)
        bearing = _bearing_deg(float(row["lat"]), float(row["lon"]), candidate_lat, candidate_lon)
        az_delta = _azimuth_delta_deg(bearing, float(row["azimuth"]))
        frames.append(
            _surface_frame_for_site(
                row,
                candidate_grid,
                raw[candidate],
                distance_m[candidate_pre][candidate],
                bearing,
                az_delta,
                ensure_all_cells_backfill=False,
            )
        )
        if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == total:
            print(f"[LTE_OFFSET][COST231_DIRECTIONAL] cells_done={idx + 1}/{total} rows_so_far={sum(len(f) for f in frames)}", flush=True)

    surface = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    ensure_all_cells = bool(cfg.get("ensure_all_cells", True))
    if ensure_all_cells:
        backfill_frames = []
        total_missing = 0
        for technology_key in sorted(site_df["technology_key"].astype(str).dropna().unique()):
            tech_sites = site_df.loc[site_df["technology_key"].astype(str) == technology_key].copy()
            if tech_sites.empty:
                continue
            if not surface.empty and "Technology" in surface.columns:
                existing = set(
                    surface.loc[
                        surface["Technology"].astype(str) == technology_key,
                        "grid_id",
                    ].astype(str)
                )
            else:
                existing = set()
            missing_grid = grid_df.loc[~grid_df["grid_id"].astype(str).isin(existing)].copy()
            total_missing += len(missing_grid)
            if missing_grid.empty:
                continue
            # Select nearest sectors in one distance matrix, then calculate RF
            # only for the selected sector-grid pairs.  This preserves Phase 9
            # nearest-eight behaviour without evaluating every sector for every
            # uncovered grid.
            grid_lat_m = missing_grid["center_lat"].to_numpy(dtype=float)
            grid_lon_m = missing_grid["center_lon"].to_numpy(dtype=float)
            site_lat_m = tech_sites["lat"].to_numpy(dtype=float)
            site_lon_m = tech_sites["lon"].to_numpy(dtype=float)
            distances = _haversine_m(grid_lat_m[:, None], grid_lon_m[:, None], site_lat_m[None, :], site_lon_m[None, :])
            nearest = np.argsort(distances, axis=1)[:, : min(backfill_k_nearest, len(tech_sites))]
            for site_pos in np.unique(nearest):
                grid_pos = np.flatnonzero(np.any(nearest == site_pos, axis=1))
                row = tech_sites.iloc[int(site_pos)]
                candidate_grid = missing_grid.iloc[grid_pos].copy()
                raw = _cost231_for_points(_site_record(row), candidate_grid["center_lat"].to_numpy(float), candidate_grid["center_lon"].to_numpy(float), float(row["frequency_mhz"])) + float(row.get("model_rsrp_adjust_db", 0.0))
                bearing = _bearing_deg(float(row["lat"]), float(row["lon"]), candidate_grid["center_lat"].to_numpy(float), candidate_grid["center_lon"].to_numpy(float))
                delta = _azimuth_delta_deg(bearing, float(row["azimuth"]))
                backfill_frames.append(_surface_frame_for_site(row, candidate_grid, raw, distances[grid_pos, site_pos], bearing, delta, ensure_all_cells_backfill=True))
        if total_missing:
            print(
                f"[LTE_OFFSET][ENSURE_ALL_CELLS] missing_grid_technology_pixels={total_missing} "
                f"action=k{backfill_k_nearest}_nearest_sector_backfill_per_technology",
                flush=True,
            )
            backfill_frames = [frame for frame in backfill_frames if not frame.empty]
            if backfill_frames:
                surface = pd.concat([surface, *backfill_frames], ignore_index=True)

    if surface.empty:
        raise ValueError("No directional Cost231 candidate rows generated")
    if "ensure_all_cells_backfill" not in surface.columns:
        surface["ensure_all_cells_backfill"] = False
    surface["ensure_all_cells_backfill"] = surface["ensure_all_cells_backfill"].fillna(False).astype(bool)
    return _attach_gridanalytics_bucket_coords(surface)


def _prepare_dt(drive_df):
    out = drive_df.copy()
    for col in ["lat", "lon"]:
        out[col] = pd.to_numeric(out.get(col, pd.Series(index=out.index)), errors="coerce")
    rsrp_col = _first_present(out, ["rsrp", "RSRP", "rssi", "RSSI", "reference_signal_received_power"])
    if rsrp_col is None:
        raise ValueError("Drive data missing RSRP/RSSI column")
    out["rsrp_measured"] = pd.to_numeric(out[rsrp_col], errors="coerce")
    out = out.dropna(subset=["lat", "lon", "rsrp_measured"]).copy()
    out = out[(out["rsrp_measured"] >= -150.0) & (out["rsrp_measured"] <= -30.0)].copy()
    out["dt_row_id"] = np.arange(len(out))
    return out.reset_index(drop=True)


def _run_cost231_at_dt(site_df, dt_df):
    out = dt_df.reset_index(drop=True).copy()
    network = _clean_text(out.get("network", out.get("technology", pd.Series(index=out.index)))).astype("string").str.upper()
    out["measured_technology"] = np.where(network.str.contains("5G|NR", na=False), "5G", "4G")
    out["assigned_strict_cell_key"] = pd.NA
    out["assigned_technology"] = pd.NA
    out["raw_cost231_at_dt_rsrp"] = np.nan
    for tech, positions in out.groupby("measured_technology", dropna=False).groups.items():
        sites = site_df.loc[site_df["technology_key"].astype(str).eq(str(tech))].reset_index(drop=True)
        if sites.empty:
            continue
        pos = np.asarray(list(positions), dtype=int)
        matrix = np.empty((len(pos), len(sites)), dtype=float)
        for col, (_, row) in enumerate(sites.iterrows()):
            matrix[:, col] = _cost231_for_points(_site_record(row), out.loc[pos, "lat"].to_numpy(float), out.loc[pos, "lon"].to_numpy(float), float(row["frequency_mhz"])) + float(row.get("model_rsrp_adjust_db", 0.0))
        best = np.nanargmax(matrix, axis=1)
        assigned = sites.iloc[best].reset_index(drop=True)
        out.loc[pos, "assigned_strict_cell_key"] = assigned["strict_cell_key"].astype(str).to_numpy()
        out.loc[pos, "assigned_technology"] = assigned["technology_key"].astype(str).to_numpy()
        out.loc[pos, "raw_cost231_at_dt_rsrp"] = matrix[np.arange(len(pos)), best]
    return out


def _attach_nearest_grid(dt_assigned, grid_df, replace_radius_m):
    tree = BallTree(np.radians(grid_df[["center_lat", "center_lon"]].to_numpy(dtype=float)), metric="haversine")
    dist_rad, idx = tree.query(np.radians(dt_assigned[["lat", "lon"]].to_numpy(dtype=float)), k=1)
    out = dt_assigned.copy()
    nearest = grid_df.iloc[idx[:, 0]].reset_index(drop=True)
    out["nearest_grid_id"] = nearest["grid_id"].astype(str).to_numpy()
    out["nearest_grid_distance_m"] = dist_rad[:, 0] * EARTH_RADIUS_M
    out["dt_replacement_eligible"] = out["nearest_grid_distance_m"] <= float(replace_radius_m)
    return out


def _offset_table(site_df, dt_assigned):
    valid = dt_assigned.dropna(subset=["dt_minus_cost231_db"]).copy()
    grouped = (
        valid.groupby("assigned_strict_cell_key", dropna=False)
        .agg(dt_count=("dt_minus_cost231_db", "size"), offset_db=("dt_minus_cost231_db", "median"))
        .reset_index()
        .rename(columns={"assigned_strict_cell_key": "strict_cell_key"})
    )
    out = site_df[["strict_cell_key", "technology_key"]].copy()
    out = out.merge(grouped, on="strict_cell_key", how="left")
    out["strict_dt_count"] = out["dt_count"].fillna(0).astype(int)
    out["offset_source"] = np.where(out["offset_db"].notna(), "cell_dt_median", "global_dt_median")

    if not valid.empty and "assigned_technology" in valid.columns:
        tech_stats = (
            valid.groupby("assigned_technology", dropna=False)["dt_minus_cost231_db"]
            .agg(technology_dt_count="size", technology_offset_db="median")
            .reset_index()
            .rename(columns={"assigned_technology": "technology_key"})
        )
        out = out.merge(tech_stats, on="technology_key", how="left")
        use_tech = out["offset_db"].isna() & out["technology_offset_db"].notna()
        out.loc[use_tech, "offset_db"] = out.loc[use_tech, "technology_offset_db"]
        out.loc[use_tech, "offset_source"] = "technology_dt_median"
        out["fallback_dt_count"] = out["technology_dt_count"].where(use_tech, out["strict_dt_count"])
    else:
        out["fallback_dt_count"] = out["strict_dt_count"]

    global_offset = float(valid["dt_minus_cost231_db"].median()) if not valid.empty else 0.0
    out["offset_db"] = pd.to_numeric(out["offset_db"], errors="coerce").fillna(global_offset)
    out["dt_count"] = out["dt_count"].fillna(0).astype(int)
    out["fallback_dt_count"] = out["fallback_dt_count"].fillna(len(valid)).astype(int)
    return out


def _apply_offset_and_replacement(surface, offsets, dt_with_grid):
    out = surface.merge(
        offsets[["strict_cell_key", "offset_db", "offset_source", "dt_count", "fallback_dt_count"]],
        on="strict_cell_key",
        how="left",
    )
    out["offset_db"] = pd.to_numeric(out["offset_db"], errors="coerce").fillna(0.0)
    out["offset_corrected_rsrp"] = (out["raw_cost231_rsrp"] + out["offset_db"]).clip(*CLIP_RSRP)
    grid_replacements = (
        dt_with_grid.loc[dt_with_grid["dt_replacement_eligible"]]
        .groupby("nearest_grid_id", dropna=False)
        .agg(
            dt_replacement_rsrp=("rsrp_measured", "mean"),
            dt_replacement_count=("rsrp_measured", "size"),
        )
        .reset_index()
        .rename(columns={"nearest_grid_id": "grid_id"})
    )

    out = out.merge(grid_replacements, on="grid_id", how="left")
    out["dt_replaced"] = out["dt_replacement_rsrp"].notna()
    # If a grid has DT, copy that DT average to every cell row in the grid so
    # production mean/best/worst all equal the measured pixel. Otherwise keep
    # each cell's own offset-corrected value for cell-level coverage.
    out["pred_rsrp"] = out["offset_corrected_rsrp"].where(
        out["dt_replacement_rsrp"].isna(),
        out["dt_replacement_rsrp"],
    )
    out["pred_rsrp"] = pd.to_numeric(out["pred_rsrp"], errors="coerce").clip(*CLIP_RSRP)
    out["pred_rsrp_smoothed"] = out["pred_rsrp"]
    out["pred_rsrq"] = np.nan
    out["pred_rsrq_smoothed"] = np.nan
    out["pred_sinr"] = np.nan
    out["pred_sinr_smoothed"] = np.nan
    out["dt_replacement_count"] = out["dt_replacement_count"].fillna(0).astype(int)
    return out


def _save_offset_baseline_results(save_delegate, final_df, project_id, job_id, operator, region):
    region_key = str(region).lower()
    save_engine = engine.get(region_key)
    if save_engine is None:
        env_key = "DATABASE_URL_Taiwan" if region_key == "taiwan" else "DATABASE_URL"
        db_url = os.getenv(env_key)
        if db_url:
            if region_key not in _SAVE_ENGINES:
                _SAVE_ENGINES[region_key] = create_engine(
                    db_url,
                    pool_size=10,
                    max_overflow=20,
                    pool_recycle=300,
                    pool_pre_ping=True,
                )
            save_engine = _SAVE_ENGINES[region_key]
        else:
            save_engine = db.engine
    if save_engine is None:
        raise ValueError(f"No database engine configured for region: {region}")

    out = final_df.copy()
    out["id"] = pd.NA
    out["project_id"] = int(project_id)
    out["job_id"] = str(job_id)
    out["created_at"] = datetime.now()

    save_lat = out.get("analytics_lat", out.get("lat", pd.Series(index=out.index)))
    save_lon = out.get("analytics_lon", out.get("lon", pd.Series(index=out.index)))
    out["lat"] = pd.to_numeric(save_lat, errors="coerce")
    out["lon"] = pd.to_numeric(save_lon, errors="coerce")
    out["lat_6dp"] = out["lat"].round(6)
    out["lon_6dp"] = out["lon"].round(6)

    for col, default in {
        "pred_rsrp": np.nan,
        "pred_rsrq": np.nan,
        "pred_sinr": np.nan,
        "pred_rsrp_smoothed": np.nan,
        "pred_rsrq_smoothed": np.nan,
        "pred_sinr_smoothed": np.nan,
    }.items():
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["pred_rsrp"] = out["pred_rsrp"].clip(*CLIP_RSRP)
    out["pred_rsrp_smoothed"] = out["pred_rsrp_smoothed"].fillna(out["pred_rsrp"]).clip(*CLIP_RSRP)
    out["pred_rsrq"] = out["pred_rsrq"].clip(-20, -3)
    out["pred_rsrq_smoothed"] = out["pred_rsrq_smoothed"].clip(-20, -3)
    out["pred_sinr"] = out["pred_sinr"].clip(-10, 30)
    out["pred_sinr_smoothed"] = out["pred_sinr_smoothed"].clip(-10, 30)

    defaults = {
        "node_b_id": out.get("nodeb_id", out.get("site", pd.Series(pd.NA, index=out.index))),
        "cell_id": out.get("legacy_nodeb_id_cell_id", out.get("strict_cell_key", pd.Series(pd.NA, index=out.index))),
        "operator": operator,
        "site_id": out.get("site", out.get("nodeb_id", pd.Series(pd.NA, index=out.index))),
        "nodeb_id_cell_id": out.get("rf_identity_key", out.get("strict_cell_key", pd.Series(pd.NA, index=out.index))),
        "legacy_nodeb_id_cell_id": out.get("legacy_nodeb_id_cell_id", out.get("cell_id", pd.Series(pd.NA, index=out.index))),
        "sector": out.get("sector", pd.Series(pd.NA, index=out.index)),
        "band": out.get("band", pd.Series(pd.NA, index=out.index)),
        "rf_identity_key": out.get("rf_identity_key", out.get("strict_cell_key", pd.Series(pd.NA, index=out.index))),
        "sector_identity_key": out.get("sector_identity_key", pd.Series(pd.NA, index=out.index)),
        "site_sector_band_key": out.get("site_sector_band_key", pd.Series(pd.NA, index=out.index)),
        "Technology": out.get("Technology", out.get("technology_key", pd.Series("4G", index=out.index))),
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
        out[col] = _clean_text(out[col] if isinstance(out[col], pd.Series) else pd.Series(value, index=out.index))

    out["operator"] = out["operator"].fillna(str(operator or "Unknown"))
    out["nodeb_id_cell_id"] = out["nodeb_id_cell_id"].fillna(out["strict_cell_key"] if "strict_cell_key" in out.columns else out["cell_id"])
    out["rf_identity_key"] = out["rf_identity_key"].fillna(out["nodeb_id_cell_id"])
    out["Technology"] = out["Technology"].fillna("4G")

    final_cols = [
        "id",
        "project_id",
        "job_id",
        "lat",
        "lat_6dp",
        "lon",
        "lon_6dp",
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "pred_rsrp_smoothed",
        "pred_rsrq_smoothed",
        "pred_sinr_smoothed",
        "node_b_id",
        "cell_id",
        "operator",
        "created_at",
        "site_id",
        "nodeb_id_cell_id",
        "legacy_nodeb_id_cell_id",
        "sector",
        "band",
        "rf_identity_key",
        "sector_identity_key",
        "site_sector_band_key",
        "Technology",
    ]
    out = out[final_cols]
    out = out.dropna(subset=["project_id", "nodeb_id_cell_id", "lat_6dp", "lon_6dp", "pred_rsrp"]).copy()
    out = out.drop_duplicates(
        subset=["project_id", "nodeb_id_cell_id", "lat_6dp", "lon_6dp"],
        keep="last",
    )

    print(
        f"[LTE_OFFSET][BASELINE_ONLY_SAVE] table=lte_prediction_baseline_results "
        f"rows={len(out)} project_id={project_id} job_id={job_id}",
        flush=True,
    )
    written_rows = save_delegate._replace_baseline_results(save_engine, out, project_id=int(project_id))
    geo_out = final_df.copy()
    geo_out["nodeb_id_cell_id"] = geo_out.get("rf_identity_key", geo_out.get("strict_cell_key"))
    geo_out["proxy_site_id"] = geo_out.get("site", geo_out.get("nodeb_id"))
    geo_out["building_count"] = geo_out.get("obstruction_branch", pd.Series("clear", index=geo_out.index)).isin(["indoor", "obstructed"]).astype(int)
    geo_out["los_blocker_count"] = geo_out["building_count"]
    geo_out["nlos_flag"] = geo_out["building_count"]
    geo_out["diffraction_proxy_db"] = (
        -pd.to_numeric(geo_out.get("building_obstruction_loss_db", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(geo_out.get("terrain_diffraction_loss_db", 0.0), errors="coerce").fillna(0.0)
    )
    geo_out["serving_distance_m"] = pd.to_numeric(geo_out.get("distance_m"), errors="coerce")
    geo_out["azimuth_delta_deg"] = pd.to_numeric(geo_out.get("azimuth_delta_deg"), errors="coerce")
    with _without_python_bridge():
        save_delegate._save_geo_features(
            geo_out,
            project_id=int(project_id),
            baseline_job_id=str(job_id),
            region=region,
            operator=operator,
            save_engine=save_engine,
            production_summary={"building_alignment": "phase27_dynamic", "polygon_alignment": "production_grid"},
        )
    print(
        f"[LTE_OFFSET][BASELINE_ONLY_SAVE_DONE] baseline_rows={written_rows} geo_feature_rows={len(geo_out)}",
        flush=True,
    )
    return written_rows


def _collapse_to_serving_grid_rows(surface):
    out = surface.copy()
    if "serving_strict_cell_key" in out.columns:
        serving_mask = out["strict_cell_key"].astype(str).eq(out["serving_strict_cell_key"].astype(str))
        serving = out.loc[serving_mask].copy()
    else:
        serving = pd.DataFrame()

    if serving.empty or serving["grid_id"].nunique(dropna=False) < out["grid_id"].nunique(dropna=False):
        best_idx = out.groupby("grid_id", dropna=False)["raw_cost231_rsrp"].idxmax()
        fallback = out.loc[best_idx].copy()
        if serving.empty:
            serving = fallback
        else:
            missing = set(out["grid_id"].astype(str).unique()) - set(serving["grid_id"].astype(str).unique())
            serving = pd.concat([serving, fallback.loc[fallback["grid_id"].astype(str).isin(missing)]], ignore_index=True)

    serving = serving.sort_values(["grid_id", "raw_cost231_rsrp"], ascending=[True, False])
    serving = serving.drop_duplicates(subset=["grid_id"], keep="first").copy()
    serving["node_cell_id"] = serving["strict_cell_key"]
    serving["Node_Cell_ID"] = serving["strict_cell_key"]
    serving["pred_rsrp_smoothed"] = serving["pred_rsrp"]
    print(
        f"[LTE_OFFSET][SERVING_GRID_COLLAPSE] rows_in={len(out)} rows_out={len(serving)} "
        f"grids={serving['grid_id'].nunique(dropna=False)} cells={serving['strict_cell_key'].nunique(dropna=True)}",
        flush=True,
    )
    return serving.reset_index(drop=True)


class LTEPredictionOffsetService:
    def __init__(self):
        self._save_delegate = LTEPredictionService()

    def submit(self, app, cfg):
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {"status": "queued"}
        threading.Thread(target=self._run_with_app_context, args=(app, job_id, cfg), daemon=True).start()
        return {"job_id": job_id}

    def get(self, job_id):
        return JOBS.get(job_id)

    def _run_with_app_context(self, app, job_id, cfg):
        with app.app_context():
            self._run(job_id, cfg)

    def _update(self, job_id, status, msg):
        JOBS[job_id]["status"] = status
        JOBS[job_id]["progress"] = msg

    def _storage_uri_to_path(self, storage_uri):
        raw = str(storage_uri or "").strip()
        if not raw:
            return None

        parsed = urlparse(raw)
        if parsed.scheme.lower() == "file":
            raw = unquote(parsed.path or "")
            if os.name == "nt" and raw.startswith("/") and len(raw) >= 4 and raw[2] == ":":
                raw = raw[1:]

        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return candidate
        if candidate.is_absolute():
            return candidate

        for base in (ML_ROOT, ML_ROOT.parent):
            resolved = (base / raw).resolve()
            if resolved.is_file():
                return resolved
        return candidate

    def _validate_dem_path(self, dem_path, site_df):
        path = Path(dem_path).expanduser()
        if not path.is_file():
            return False, None, "file_missing"

        dem = None
        try:
            dem = _DemSampler(path)
            lat = pd.to_numeric(site_df.get("lat"), errors="coerce")
            lon = pd.to_numeric(site_df.get("lon"), errors="coerce")
            valid = lat.notna() & lon.notna()
            if valid.any():
                sample = site_df.loc[valid, ["lat", "lon"]].drop_duplicates().head(100)
                values = dem.sample(sample["lat"].to_numpy(float), sample["lon"].to_numpy(float))
                finite_share = float(np.isfinite(values).mean()) if len(values) else 0.0
                if finite_share <= 0.0:
                    return False, dem.band, "no_site_samples_inside_dem"
            return True, dem.band, "ok"
        except Exception as exc:
            return False, None, str(exc)
        finally:
            if dem is not None:
                dem.close()

    def _active_dem_asset_rows(self, db_engine, project_id):
        if db_engine is None:
            return []

        def action():
            with db_engine.connect() as connection:
                table_exists = connection.execute(text("""
                    SELECT COUNT(*)
                    FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'tbl_project_dem_asset'
                """)).scalar()
                if not table_exists:
                    return []
                return connection.execute(text("""
                    SELECT id, source_name, storage_uri, selected_elevation_band,
                           crs, resolution_m, created_at
                    FROM tbl_project_dem_asset
                    WHERE project_id = :project_id
                      AND is_active = 1
                    ORDER BY created_at DESC, id DESC
                """), {"project_id": int(project_id)}).mappings().all()

        try:
            return _with_db_retry(db_engine, action, "dem_asset_lookup")
        except Exception as exc:
            print(f"[LTE_OFFSET][DEM_RESOLVE] source=project_dem_asset status=lookup_failed reason={exc}", flush=True)
            return []

    def _local_project_dem_candidates(self, project_id, region):
        candidates = []
        if int(project_id) == 210 or str(region).lower() == "taiwan":
            mapdata_root = ML_ROOT / "tests" / "new-project" / "data" / "mapdata"
            if mapdata_root.exists():
                candidates.extend(sorted(mapdata_root.rglob("height_5m.grd")))
        return candidates

    def _resolve_dem_path(self, project_id, region, site_df, db_engine, requested_path=None):
        requested = str(requested_path or "").strip()
        if requested:
            path = self._storage_uri_to_path(requested)
            ok, band, reason = self._validate_dem_path(path, site_df)
            print(
                f"[LTE_OFFSET][DEM_RESOLVE] source=request path={path} valid={ok} "
                f"selected_band={band} reason={reason}",
                flush=True,
            )
            if ok:
                return str(path)
            raise RuntimeError(f"Configured terrain DEM is unusable: {reason}")

        for row in self._active_dem_asset_rows(db_engine, project_id):
            path = self._storage_uri_to_path(row.get("storage_uri"))
            ok, band, reason = self._validate_dem_path(path, site_df)
            print(
                f"[LTE_OFFSET][DEM_RESOLVE] source=project_dem_asset asset_id={row.get('id')} "
                f"storage_uri={row.get('storage_uri')} path={path} valid={ok} "
                f"declared_band={row.get('selected_elevation_band')} selected_band={band} reason={reason}",
                flush=True,
            )
            if ok:
                return str(path)

        for path in self._local_project_dem_candidates(project_id, region):
            ok, band, reason = self._validate_dem_path(path, site_df)
            print(
                f"[LTE_OFFSET][DEM_RESOLVE] source=local_project_mapdata path={path} "
                f"valid={ok} selected_band={band} reason={reason}",
                flush=True,
            )
            if ok:
                return str(path)

        try:
            path = ensure_project_dem(
                project_id=int(project_id),
                region=str(region).lower(),
                site_df=site_df,
                timeout_sec=60,
                force=False,
            )
            ok, band, reason = self._validate_dem_path(path, site_df)
            print(
                f"[LTE_OFFSET][DEM_RESOLVE] source=auto_generated path={path} "
                f"valid={ok} selected_band={band} reason={reason}",
                flush=True,
            )
            if ok:
                return str(path)
        except Exception as exc:
            print(f"[LTE_OFFSET][DEM_RESOLVE] source=auto_generated valid=False reason={exc}", flush=True)

        print("[LTE_OFFSET][DEM_RESOLVE] source=none valid=False action=terrain_disabled", flush=True)
        return None

    def _run(self, job_id, cfg):
        try:
            region = str(cfg.get("region", "india")).lower()
            print(
                f"[LTE_OFFSET][JOB_START] job_id={job_id} project_id={cfg['project_id']} "
                f"region={region} sessions={cfg['session_ids']} grid={cfg['grid_resolution']}",
                flush=True,
            )

            current_engine = engine.get(region, engine["india"])
            self._update(job_id, "running", "Fetching site data")
            force_direct_db = False
            try:
                site_df_raw, operator = fetch_site_data(
                    cfg["project_id"],
                    region=region,
                    polygon_ids=cfg.get("polygon_ids"),
                    operator=cfg.get("operator"),
                )
            except PythonBridgeError as exc:
                if current_engine is None:
                    raise
                force_direct_db = True
                print(f"[LTE_OFFSET][BRIDGE_FALLBACK] stage=site_data reason={exc}", flush=True)
                with _without_python_bridge():
                    site_df_raw, operator = _with_db_retry(
                        current_engine,
                        lambda: fetch_site_data(
                            cfg["project_id"],
                            region=region,
                            polygon_ids=cfg.get("polygon_ids"),
                            operator=cfg.get("operator"),
                        ),
                        "site_data",
                    )
            bridge_context = _without_python_bridge() if force_direct_db else nullcontext()
            with bridge_context:
                polygons = _resolve_prediction_polygons(cfg, current_engine)
            if polygons:
                import geopandas as gpd

                polygon_gdf = gpd.GeoDataFrame({"geometry": polygons}, crs="EPSG:4326")
                site_mask_source = site_df_raw.copy()
                polygon_gdf, _ = transform_polygon_for_sites(polygon_gdf, site_mask_source)
                polygons = list(polygon_gdf.geometry)
                union = unary_union(polygons)
                mask = [union.covers(Point(lon, lat)) for lat, lon in site_df_raw[["lat", "lon"]].to_numpy()]
                site_df_raw = site_df_raw.loc[mask].copy()
            if site_df_raw.empty:
                raise ValueError("No site rows found inside project polygon")
            site_df = _prepare_site_rows(site_df_raw, region)
            dem_raster_path = self._resolve_dem_path(
                project_id=cfg["project_id"],
                region=region,
                site_df=site_df,
                db_engine=current_engine,
                requested_path=cfg.get("dem_raster_path"),
            )
            cfg["dem_raster_path"] = dem_raster_path

            self._update(job_id, "running", "Fetching drive data")
            bridge_context = _without_python_bridge() if force_direct_db else nullcontext()
            with bridge_context:
                drive_df = fetch_drive_data(
                    cfg["session_ids"],
                    operator,
                    cfg["project_id"],
                    region=region,
                    frontend_drive_rows=cfg.get("drive_rows"),
                    frontend_drive_rows_source=cfg.get("drive_rows_source"),
                )
            dt_df = _prepare_dt(drive_df)

            self._update(job_id, "running", "Fetching building geometry")
            with (_without_python_bridge() if force_direct_db else nullcontext()):
                building_df = (
                    _with_db_retry(
                        current_engine,
                        lambda: fetch_building_data(cfg["project_id"], region=region),
                        "building_geometry",
                    )
                    if cfg.get("building", True) else pd.DataFrame()
                )

            self._update(job_id, "running", "Fetching grid pixels")
            grid_df = _grid_from_bridge_or_db(cfg, current_engine, polygons, force_direct_db=force_direct_db)
            print(
                f"[LTE_OFFSET][INPUT] strict_cells={len(site_df)} grid_pixels={len(grid_df)} dt_rows={len(dt_df)}",
                flush=True,
            )

            self._update(job_id, "running", "Running Phase 9/26/27 production baseline")
            surface = _run_raw_surface(site_df, grid_df, cfg)
            with (_without_python_bridge() if force_direct_db else nullcontext()):
                grid_clutter, resolved_building_df, clutter_summary = load_or_build_phase27_clutter(
                    grid_df, building_df, cfg["project_id"], current_engine, cfg.get("ghs_obat_csv_path")
                )
            clutter_by_grid = grid_clutter.set_index("grid_id")["clutter_class"].to_dict() if not grid_clutter.empty else {}
            print(f"[LTE_OFFSET][PHASE27_CLUTTER] {clutter_summary}", flush=True)
            raw_best = surface.groupby(["Technology", "grid_id"], dropna=False)["raw_cost231_rsrp"].transform("max")
            physical_mask = (surface["raw_cost231_rsrp"] >= raw_best - 20.0) & (surface["raw_cost231_rsrp"] >= -145.0)
            surface = surface.loc[physical_mask].copy().reset_index(drop=True)
            print(f"[LTE_OFFSET][PHASE26_CANDIDATES] retained={len(surface)} margin_db=20 min_raw_dbm=-145", flush=True)
            dt_assigned = _run_cost231_at_dt(site_df, dt_df)
            surface = score_candidates(
                surface, site_df, resolved_building_df, cfg["project_id"], region,
                dem_raster_path=dem_raster_path,
                clutter_by_grid=clutter_by_grid,
                allow_auto_dem=False,
            )
            surface["technology"] = surface["Technology"].astype(str)
            surface = add_features(surface, "strict_cell_key")

            dt_points = dt_assigned.rename(columns={"assigned_strict_cell_key": "strict_cell_key"}).copy()
            dt_points["grid_id"] = "DT_" + dt_points["dt_row_id"].astype(str)
            dt_points["Technology"] = dt_points["assigned_technology"].astype(str)
            dt_points["technology"] = dt_points["Technology"]
            dt_points = dt_points.drop(columns=["band"], errors="ignore")
            dt_points = dt_points.merge(
                site_df[["strict_cell_key", "band_key", "sector_key", "frequency_mhz", "original_tx_power_dbm"]],
                on="strict_cell_key", how="left"
            ).rename(columns={"band_key": "band", "frequency_mhz": "serving_frequency_mhz"})

            # EQUAL-POWER: the model is forced to FIXED_TX_POWER_DBM_OVERRIDE, but the
            # drive test was recorded at each cell's REAL power. Every calibration below
            # (g5_anchor, fit_outdoor, fit_local) minimises (measured - model), so if only
            # the model side is normalised the calibration simply adds the real-power
            # difference straight back and the equal-power rule is cancelled. Shift the
            # measured RSRP into the same equal-power space first, so the residual carries
            # no power term. This mirrors Phase 39's _apply_equal_power_assumptions, which
            # shifts BOTH the DT rows and the candidates before fitting.
            if FIXED_TX_POWER_DBM_OVERRIDE is not None:
                _real_tx = pd.to_numeric(dt_points.get("original_tx_power_dbm"), errors="coerce")
                _dt_power_shift = (float(FIXED_TX_POWER_DBM_OVERRIDE) - _real_tx).fillna(0.0)
                dt_points["equal_power_shift_db"] = _dt_power_shift
                dt_points["rsrp_measured_real_power"] = pd.to_numeric(dt_points["rsrp_measured"], errors="coerce")
                dt_points["rsrp_measured"] = dt_points["rsrp_measured_real_power"] + _dt_power_shift
                print(
                    "[LTE_OFFSET][EQUAL_POWER_DT_NORMALISED] target_dbm="
                    f"{float(FIXED_TX_POWER_DBM_OVERRIDE):.1f} "
                    + "; ".join(
                        f"{t}: n={len(g)} median_shift_db={pd.to_numeric(g['equal_power_shift_db']).median():+.2f}"
                        for t, g in dt_points.groupby(dt_points["technology"].astype(str))
                    ),
                    flush=True,
                )

            dt_points["raw_cost231_rsrp"] = pd.to_numeric(dt_points["raw_cost231_at_dt_rsrp"], errors="coerce")
            dt_points = score_candidates(
                dt_points, site_df, resolved_building_df, cfg["project_id"], region,
                dem_raster_path=dem_raster_path,
                clutter_by_grid={},
                allow_auto_dem=False,
            )
            dt_points["technology"] = dt_points["Technology"].astype(str)
            dt_points = add_features(dt_points, "strict_cell_key")

            phase36_v2 = bool(cfg.get("enable_phase36_v2", True))
            quality_cal = pd.DataFrame()
            if phase36_v2:
                # --- Phase 36 v2 RSRP: per-RE reference + real antenna + Water override + local IDW field ---
                _ant_cols = [c for c in ("strict_cell_key", "Etilt", "Mtilt", "Height", "antenna_model",
                                         "lat", "lon", "azimuth") if c in site_df.columns]
                _ant = site_df[_ant_cols].drop_duplicates("strict_cell_key").rename(
                    columns={"lat": "_site_lat", "lon": "_site_lon", "azimuth": "_site_az"})
                surface = surface.drop(columns=[c for c in ("Etilt", "Mtilt", "Height", "antenna_model") if c in surface.columns],
                                       errors="ignore").merge(_ant, on="strict_cell_key", how="left")
                dt_points = dt_points.drop(columns=[c for c in ("Etilt", "Mtilt", "Height", "antenna_model") if c in dt_points.columns],
                                           errors="ignore").merge(_ant, on="strict_cell_key", how="left")
                # DT geometry (surface already carries distance_m / azimuth_delta_deg)
                _slat = pd.to_numeric(dt_points["_site_lat"], errors="coerce").to_numpy(float)
                _slon = pd.to_numeric(dt_points["_site_lon"], errors="coerce").to_numpy(float)
                _dlat = pd.to_numeric(dt_points["lat"], errors="coerce").to_numpy(float)
                _dlon = pd.to_numeric(dt_points["lon"], errors="coerce").to_numpy(float)
                dt_points["distance_m"] = np.maximum(_haversine_m(_slat, _slon, _dlat, _dlon), 1.0)
                _brg = _bearing_deg(_slat, _slon, _dlat, _dlon)
                _az = pd.to_numeric(dt_points["_site_az"], errors="coerce").fillna(0.0).to_numpy(float)
                dt_points["azimuth_delta_deg"] = np.abs((_brg - _az + 180.0) % 360.0 - 180.0)
                dt_points["rsrp_measured"] = pd.to_numeric(dt_points.get("rsrp_measured"), errors="coerce")
                dt_points = _p36.apply_reference_and_water(dt_points, "physical_rsrp_unclipped", g5_level_anchor_db=0.0)
                surface = _p36.apply_reference_and_water(surface, "physical_rsrp_unclipped", g5_level_anchor_db=0.0)
                clean5 = dt_points[
                    (dt_points["technology"].astype(str) == "5G")
                    & (dt_points["obstruction_branch"].astype(str) == "clear")
                    & (dt_points.get("clutter_class", pd.Series("", index=dt_points.index)).astype(str).str.lower() != "water")
                ]
                g5_anchor = float(
                    (pd.to_numeric(clean5["rsrp_measured"], errors="coerce")
                     - pd.to_numeric(clean5["phase36_physical_rsrp"], errors="coerce")).median()
                ) if len(clean5) >= 20 else 0.0
                g5_anchor = 0.0 if not np.isfinite(g5_anchor) else g5_anchor
                if g5_anchor:
                    for f in (dt_points, surface):
                        m5 = (f["technology"].astype(str) == "5G").to_numpy()
                        f.loc[m5, "phase36_physical_rsrp"] = pd.to_numeric(f.loc[m5, "phase36_physical_rsrp"], errors="coerce") + g5_anchor

                # DT train/validation split for the calibration + quality
                _h = pd.util.hash_pandas_object(dt_points.get("nearest_grid_id", dt_points["dt_row_id"]).astype(str),
                                                index=False).astype("uint64")
                dt_points["split"] = np.where((_h % 10) < 7, "train", "validation")
                dt_fit = dt_points[dt_points["split"].eq("train")]

                layers = fit_outdoor(dt_fit, "phase36_physical_rsrp")
                local_models = _calib.fit_local(dt_fit, layers, "phase36_physical_rsrp")
                final_df = _calib.apply_outdoor_v2(surface, layers, "phase36_physical_rsrp", local_models)

                # --- Phase 37 RSRQ / SINR ---
                try:
                    dt_scored_q = _calib.apply_outdoor_v2(dt_points, layers, "phase36_physical_rsrp", local_models)
                    dt_scored_q["rsrq_measured"] = pd.to_numeric(dt_points.get("rsrq"), errors="coerce")
                    dt_scored_q["sinr_measured"] = pd.to_numeric(dt_points.get("sinr"), errors="coerce")
                    # Authoritative carrier identity (technology + real/deployed
                    # frequency) straight from the prepared site rows, keyed by
                    # strict_cell_key. final_df / dt_scored_q don't reliably carry
                    # the same frequency column all the way through their own
                    # separate join paths (dt_scored_q in particular can end up
                    # with only the COST-231 model anchor for an out-of-range
                    # band), so compute_quality is given this instead of trusting
                    # either frame's own frequency column for the join key.
                    cell_carrier_map = site_df[["strict_cell_key", "technology_key", "original_frequency_mhz"]].drop_duplicates(
                        "strict_cell_key"
                    ).copy()
                    cell_carrier_map["carrier_key"] = (
                        cell_carrier_map["technology_key"].astype(str) + "|"
                        + pd.to_numeric(cell_carrier_map["original_frequency_mhz"], errors="coerce").round(1).astype("string")
                    )
                    final_df, quality_cal = _p37.compute_quality(
                        final_df, dt_scored_q, serving_col="final_rsrp", cell_carrier_map=cell_carrier_map
                    )
                except Exception as exc:  # quality is best-effort; RSRP must still ship
                    print(f"[LTE_OFFSET][PHASE37] disabled reason={exc}", flush=True)
                    final_df["pred_rsrq"] = np.nan
                    final_df["pred_sinr"] = np.nan
                model_tag = "cost231_phase9_phase26_phase36v2_phase37"
            else:
                layers = fit_outdoor(dt_points, "physical_rsrp_unclipped")
                final_df = apply_outdoor(surface, layers, "physical_rsrp_unclipped")
                model_tag = "cost231_phase9_phase26_phase27"

            final_df["pred_rsrp"] = final_df["final_rsrp"].fillna(-140.0)
            final_df["pred_rsrp_smoothed"] = final_df["pred_rsrp"]
            if "pred_rsrq" in final_df.columns:
                final_df["pred_rsrq_smoothed"] = final_df["pred_rsrq"]
            if "pred_sinr" in final_df.columns:
                final_df["pred_sinr_smoothed"] = final_df["pred_sinr"]
            final_df["dt_replaced"] = False
            final_df.attrs["production_summary"] = {
                "model": model_tag,
                "coverage_radius_m": float(cfg.get("radius_m") or cfg.get("coverage_radius_m") or 500.0),
                "candidate_filter_rule": "distance <= safe search radius; no pre-loss RSRP cutoff",
                "out_of_radius_backfill_k_nearest": int(cfg.get("out_of_radius_backfill_k_nearest", 8)),
                "ensure_all_cells": bool(cfg.get("ensure_all_cells", True)),
                "ensure_all_cells_backfill_rows": int(final_df.get("ensure_all_cells_backfill", pd.Series(False, index=final_df.index)).sum()),
                "dynamic_layers": [str(layer["layer"].iloc[0]) for layer in layers if not layer.empty],
                "phase27_clutter": clutter_summary,
                "dt_replaced_pixels": 0,
                "raw_directional_rows": int(len(final_df)),
                "grid_pixels": int(final_df["grid_id"].nunique(dropna=False)),
                "missing_grid_pixels_after_backfill": int(len(grid_df) - final_df["grid_id"].nunique(dropna=False)),
                "strict_cells": int(final_df["strict_cell_key"].nunique(dropna=True)),
            }

            self._update(job_id, "running", "Saving offset baseline to database")
            _save_offset_baseline_results(
                self._save_delegate,
                final_df,
                cfg["project_id"],
                job_id,
                operator,
                region,
            )

            os.makedirs("temp", exist_ok=True)
            output = f"temp/final_offset_{job_id}.csv"
            final_df.to_csv(output, index=False)
            JOBS[job_id]["output"] = output
            JOBS[job_id]["rows"] = len(final_df)
            JOBS[job_id]["metrics"] = final_df.attrs["production_summary"]
            self._update(job_id, "done", "Completed")
        except Exception as exc:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(exc)
            print(f"[LTE_OFFSET][JOB_FAILED] job_id={job_id} error={exc}", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()


def transform_polygon_for_sites(polygon_gdf, site_df):
    from tools.lte_prediction.geo_correction_pipeline import align_project_polygon_to_points

    return align_project_polygon_to_points(polygon_gdf, site_df)
