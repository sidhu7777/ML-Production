"""
Local-cache-only fix for project 210's missing real n78 (5G NR, 3300MHz)
drive-test coverage.

Confirmed this session via direct, read-only queries against the real
Taiwan DB (DATABASE_URL_Taiwan):
  - project 210's own sessions have 188,244 raw rows in tbl_network_log,
    but the local cache (drive_df.csv) only has 19,491 - most rows,
    including ALL 561 real band='n78' rows, were dropped somewhere in the
    original cache-build step.
  - 561 real, properly band='n78'-labeled rows exist for project 210
    (558 within ~3km of site GA20000099) - genuine ground truth, just
    missing from the local cache.
  - 71,695 rows are band='N/A' but network='5G NSA' (real 5G measurements
    with no cell-identity metadata - no pci/earfcn/cell_id). Investigated
    multiple ways to recover their real band (primary_cell_info_1, PCI,
    same-timestamp LTE-anchor join) - none identify the true NR band.
    Per explicit user direction, these are relabeled n78 anyway and given
    an imputed pci/earfcn from their nearest REAL n78 measurement point
    (distance-based), to get enough n78-labeled volume to work with for
    both LTE and NR technologies. This is an explicit, labeled
    APPROXIMATION - not verified ground truth - and is kept distinguishable
    from the 561 real rows via the `band_source` column on every row this
    script touches.

DB access is READ-ONLY (SELECT only). Nothing is written back to the DB.
Output is a NEW local file - the original drive_df.csv is never modified.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(__file__).parent / "data" / "project_210_taiwan"
ORIGINAL_DRIVE_CSV = DATA_DIR / "drive_df.csv"
OUTPUT_DRIVE_CSV = DATA_DIR / "drive_df_n78_fixed.csv"


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def nearest_neighbor_assign(na_df: pd.DataFrame, ref_df: pd.DataFrame, ref_cols: list) -> pd.DataFrame:
    """For each row in na_df, find the nearest row in ref_df (by real lat/lon)
    and copy ref_cols from it. Brute-force O(n*m) - fine here (~71.7k x 561)."""
    out = na_df.copy()
    ref_lat = ref_df["lat"].to_numpy()
    ref_lon = ref_df["lon"].to_numpy()
    for col in ref_cols:
        out[col] = np.nan
    out["nearest_n78_distance_m"] = np.nan

    na_lat = out["lat"].to_numpy()
    na_lon = out["lon"].to_numpy()
    chunk = 2000
    for start in range(0, len(out), chunk):
        end = min(start + chunk, len(out))
        lat_chunk = na_lat[start:end][:, None]
        lon_chunk = na_lon[start:end][:, None]
        d = haversine_m(lat_chunk, lon_chunk, ref_lat[None, :], ref_lon[None, :])
        nearest_idx = np.argmin(d, axis=1)
        nearest_dist = d[np.arange(len(nearest_idx)), nearest_idx]
        for col in ref_cols:
            out.iloc[start:end, out.columns.get_loc(col)] = ref_df[col].to_numpy()[nearest_idx]
        out.iloc[start:end, out.columns.get_loc("nearest_n78_distance_m")] = nearest_dist
    return out


def main():
    db_url = os.getenv("DATABASE_URL_Taiwan")
    eng = create_engine(db_url)

    with eng.connect() as conn:
        ref = conn.execute(text("SELECT ref_session_id FROM tbl_project WHERE id=210")).scalar()
        session_ids = [s.strip() for s in (ref or "").split(",") if s.strip()]
        session_str = ",".join(session_ids)

        select_cols = """
            id, session_id, timestamp, lat, lon, battery, speed AS "Speed", level, apps,
            num_cells, network, m_alpha_long, m_alpha_short, pci, rssi, rsrp, rsrq, sinr,
            mos, jitter, latency, tac, packet_loss, dl_tpt, ul_tpt, band, image_path,
            indoor_outdoor, nodeb_id, cell_id, earfcn, `primary`, network AS technology
        """

        print("[FIX] pulling real band='n78' rows for project 210 (read-only)...")
        n78_df = pd.read_sql(
            text(f"SELECT {select_cols} FROM tbl_network_log WHERE session_id IN ({session_str}) AND band='n78'"),
            conn,
        )
        print(f"[FIX] real n78 rows pulled: {len(n78_df)}")

        print("[FIX] pulling band='N/A' (5G NSA, unlabeled) rows for project 210 (read-only)...")
        na_df = pd.read_sql(
            text(f"SELECT {select_cols} FROM tbl_network_log WHERE session_id IN ({session_str}) AND band='N/A'"),
            conn,
        )
        print(f"[FIX] N/A rows pulled: {len(na_df)}")

    n78_df = n78_df.dropna(subset=["lat", "lon"]).copy()
    na_df = na_df.dropna(subset=["lat", "lon"]).copy()

    n78_df["band_source"] = "verified_n78"

    print("[FIX] assigning nearest-real-n78 pci/earfcn to N/A rows by distance (explicit approximation)...")
    na_fixed = nearest_neighbor_assign(na_df, n78_df, ["pci", "earfcn", "cell_id", "nodeb_id"])
    na_fixed["band"] = "n78"
    na_fixed["band_source"] = "imputed_from_na_nearest_n78"
    print(f"[FIX] N/A->n78 imputation distance stats (m): "
          f"min={na_fixed['nearest_n78_distance_m'].min():.0f} "
          f"median={na_fixed['nearest_n78_distance_m'].median():.0f} "
          f"max={na_fixed['nearest_n78_distance_m'].max():.0f}")

    original = pd.read_csv(ORIGINAL_DRIVE_CSV, low_memory=False)
    original["band_source"] = "original_local_cache"

    combined = pd.concat(
        [original, n78_df.drop(columns=["nearest_n78_distance_m"], errors="ignore"), na_fixed],
        ignore_index=True, sort=False,
    )
    combined.to_csv(OUTPUT_DRIVE_CSV, index=False)
    print(f"[FIX] wrote {OUTPUT_DRIVE_CSV} ({len(combined)} rows total)")
    print(f"[FIX] band_source breakdown: {combined['band_source'].value_counts().to_dict()}")
    print(f"[FIX] band breakdown (post-fix): {combined['band'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
