"""
Phase 20: builds a corrected DT-match dataset for 5G, using the REAL
NR-tagged drive-test rows (network='5G NSA'/'5G SA', confirmed 8,033 rows
correctly inside project 210's polygon in the live Taiwan DB) instead of
the 136 proximity-mislabeled LTE-anchor rows Phase 9/17/18/19 have used
until now.

Root cause chain that led here (all verified against the live DB, not
guessed):
  1. fetch_project_propagation_cache.py's _network_like_4g() filter drops
     any DT row whose `network` value doesn't contain "4G"/"LTE" - so
     every real 'network=5G NSA'/'5G SA' row was excluded at the very
     first fetch step, before any cached file ever saw it.
  2. The flat `band` column is unreliable for those rows (NaN/'N/A' for
     8,019 of the 8,033) - `network` + `primary_cell_info_1` ('nr_from_
     signalstrength') are the real signal, not `band`.
  3. The project polygon itself (`map_regions.region`) is stored with
     coordinates in (lat, lon) order, not the standard WKT (lon, lat) -
     shapely's default Point(lon, lat) covers-check silently returns zero
     matches against it unless you build Point(lat, lon) to match.
  4. pci/earfcn/cell_id are mostly 'N/A' for these rows too, so cell
     matching has to be nearest-site-by-distance (same fallback the rest
     of this pipeline already uses), not an ID join.

This script is read-only against the live DB (SELECT only) and writes a
new file - it does NOT overwrite phase9_dt_match_project210.parquet or
touch Phase 15/16/17/18/19's own files.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from shapely import wkt
from shapely.geometry import Point
from sqlalchemy import create_engine, text

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
for p in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit_project210_phase13_beam_check as phase13
import streamlit_project210_phase15_radius_progression as phase15
import test_project210_phase17_full_polygon_geo_dt_comparison as phase17

PROJECT_ID = 210
PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
OUT_DIR = PROJECT_DIR / "cost231_phase20_5g_real_dt_match"
DT_REPLACE_RADIUS_M = 25.0
MAX_CELL_MATCH_DISTANCE_M = 3000.0  # generous - reject only truly implausible matches
N78_TECHNOLOGY_OFFSET_DB = phase17.N78_TECHNOLOGY_OFFSET_DB
FREQ_MHZ_5G = 2600.0  # same established convention as the rest of this pipeline


def _fetch_real_5g_dt_rows() -> pd.DataFrame:
    load_dotenv(ML_ROOT / ".env")
    engine = create_engine(os.environ["DATABASE_URL_Taiwan"], pool_pre_ping=True, pool_recycle=300)
    with engine.connect() as conn:
        proj = pd.read_sql(text("SELECT * FROM tbl_project WHERE id = :pid"), conn, params={"pid": PROJECT_ID})
        sessions = [int(v) for v in re.findall(r"\d+", str(proj.iloc[0].get("ref_session_id") or ""))]
        in_clause = ",".join(str(s) for s in sessions)
        regions = pd.read_sql(
            text("SELECT ST_AsText(region) AS region_wkt FROM map_regions WHERE tbl_project_id=:pid AND status=1"),
            conn, params={"pid": PROJECT_ID},
        )
        poly = wkt.loads(str(regions.iloc[0]["region_wkt"]))
        q = text(
            f"""
            SELECT id, lat, lon, rsrp, rsrq, sinr, session_id, timestamp, network
            FROM tbl_network_log
            WHERE session_id IN ({in_clause})
              AND LOWER(COALESCE(`primary`, '')) = 'yes'
              AND rsrp IS NOT NULL
              AND network LIKE '%5G%'
            """
        )
        df = pd.read_sql(q, conn)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    # polygon is stored (lat, lon) order - Point(lat, lon) matches it, confirmed via a
    # 4G sanity check that reproduced the known-correct 11,458-row count exactly.
    mask = [poly.covers(Point(la, lo)) for la, lo in df[["lat", "lon"]].to_numpy()]
    inside = df.loc[mask].reset_index(drop=True)
    print(f"[PHASE20] real 5G-network DT rows inside project {PROJECT_ID}'s polygon: {len(inside)}")
    return inside


def _match_nearest_5g_cell(dt: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    cells = identity[identity["band"].astype(str) == "78"].reset_index(drop=True)
    print(f"[PHASE20] real 5G (band=78) identity cells available: {len(cells)}")
    cell_lat = cells["lat"].to_numpy(dtype=float)
    cell_lon = cells["lon"].to_numpy(dtype=float)

    assigned_keys = []
    assigned_dist = []
    for lat, lon in dt[["lat", "lon"]].to_numpy():
        d = _haversine_vec(lat, lon, cell_lat, cell_lon)
        j = int(np.argmin(d))
        assigned_keys.append(cells["Node_Cell_ID"].iloc[j])
        assigned_dist.append(float(d[j]))
    out = dt.copy()
    out["assigned_strict_cell_key"] = assigned_keys
    out["cell_match_distance_m"] = assigned_dist
    before = len(out)
    out = out[out["cell_match_distance_m"] <= MAX_CELL_MATCH_DISTANCE_M].reset_index(drop=True)
    print(f"[PHASE20] matched within {MAX_CELL_MATCH_DISTANCE_M:.0f}m of a real 5G cell: {len(out)} / {before}")
    return out


def _haversine_vec(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _compute_raw_cost231(dt: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    params_common = {"ue_height": 1.5, "k1": 0, "k2": 0, "cable_loss": 2.0, "antenna_gain": 18.0}
    out = dt.merge(
        identity[["Node_Cell_ID", "lat", "lon", "azimuth", "Etilt", "Mtilt", "Height", "tx_power"]].rename(
            columns={"lat": "site_lat", "lon": "site_lon"}
        ),
        left_on="assigned_strict_cell_key", right_on="Node_Cell_ID", how="left",
    )
    raw_pred = np.full(len(out), np.nan)
    for cell_key, group in out.groupby("assigned_strict_cell_key"):
        row0 = group.iloc[0]
        if pd.isna(row0.get("Etilt")):
            continue
        site_row = pd.Series({
            "lat": row0["site_lat"], "lon": row0["site_lon"], "azimuth": row0["azimuth"],
            "Etilt": row0["Etilt"], "Mtilt": row0["Mtilt"], "Height": row0["Height"], "tx_power": row0["tx_power"],
        })
        site_dict = phase15._row_to_site_dict_fixed(site_row)
        pred = np.array([
            phase15.compute_sector_rsrp(site_dict, la, lo, FREQ_MHZ_5G, params_common) + N78_TECHNOLOGY_OFFSET_DB
            for la, lo in zip(group["lat"].to_numpy(dtype=float), group["lon"].to_numpy(dtype=float))
        ])
        raw_pred[group.index] = pred
    out["raw_cost231_at_dt_rsrp"] = raw_pred
    return out


def _match_grid(dt: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    glat = grid["center_lat"].to_numpy(dtype=float)
    glon = grid["center_lon"].to_numpy(dtype=float)
    gid = grid["grid_id"].to_numpy()
    nearest_id, nearest_dist = [], []
    for lat, lon in dt[["lat", "lon"]].to_numpy():
        d = _haversine_vec(lat, lon, glat, glon)
        j = int(np.argmin(d))
        nearest_id.append(gid[j])
        nearest_dist.append(float(d[j]))
    out = dt.copy()
    out["nearest_grid_id"] = nearest_id
    out["nearest_grid_distance_m"] = nearest_dist
    out["dt_replacement_eligible"] = out["nearest_grid_distance_m"] <= DT_REPLACE_RADIUS_M
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw5g = _fetch_real_5g_dt_rows()
    identity = phase13.load_identity()
    matched = _match_nearest_5g_cell(raw5g, identity)
    matched = _compute_raw_cost231(matched, identity)
    missing = matched["raw_cost231_at_dt_rsrp"].isna()
    if missing.any():
        print(f"[PHASE20] {int(missing.sum())} rows had no identity match after cell-matching - dropped")
        matched = matched.loc[~missing].reset_index(drop=True)

    grid = pd.read_parquet(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210.parquet")
    matched = _match_grid(matched, grid)

    matched["rsrp_measured"] = matched["rsrp"].astype(float)
    matched["dt_minus_cost231_db"] = matched["rsrp_measured"] - matched["raw_cost231_at_dt_rsrp"]
    matched["assigned_technology"] = "5G"
    matched["primary"] = "yes"
    matched["source_table"] = "tbl_network_log"
    matched["m_alpha_long"] = None
    matched["m_alpha_short"] = None
    matched["rsrq"] = matched.get("rsrq")
    matched["sinr"] = matched.get("sinr")
    matched["mci"] = None
    matched["pci"] = None
    matched["earfcn"] = None
    matched["dt_row_id"] = matched["id"]
    matched["cell_id"] = None
    matched["nodeb_id"] = None

    final_cols = [
        "id", "lat", "lon", "rsrp_measured", "rsrq", "sinr", "mci", "pci", "earfcn", "session_id", "timestamp",
        "network", "m_alpha_long", "m_alpha_short", "primary", "source_table", "cell_id", "nodeb_id", "dt_row_id",
        "assigned_strict_cell_key", "assigned_technology", "raw_cost231_at_dt_rsrp", "dt_minus_cost231_db",
        "nearest_grid_id", "nearest_grid_distance_m", "dt_replacement_eligible",
    ]
    out = matched[final_cols].copy()
    out.to_parquet(OUT_DIR / "phase20_real_5g_dt_match_project210.parquet", index=False)
    print(f"[PHASE20] wrote {len(out)} real, correctly-matched 5G DT rows")
    print(out[["rsrp_measured", "raw_cost231_at_dt_rsrp", "dt_minus_cost231_db", "nearest_grid_distance_m"]].describe())

    original = pd.read_parquet(PHASE9_DIR / "phase9_dt_match_project210.parquet")
    kept_4g = original[original["assigned_technology"] == "4G"].copy()
    kept_4g["timestamp"] = kept_4g["timestamp"].astype(str)
    out["timestamp"] = out["timestamp"].astype(str)
    merged = pd.concat([kept_4g, out], ignore_index=True)
    merged.to_parquet(OUT_DIR / "phase9_dt_match_project210_corrected.parquet", index=False)
    print(f"[PHASE20] wrote corrected merged DT dataset: {len(kept_4g)} real 4G + {len(out)} real 5G = {len(merged)} rows")
    print(f"[PHASE20] (previous file had only 136 mislabeled 5G rows instead of these {len(out)} real ones)")


if __name__ == "__main__":
    main()
