"""Phase 39 v2 -- isolates ONE variable: does removing production's 500m
candidate-radius cap restore the 4G-vs-5G gap that Phase 39 (original) shows
and that production's actual saved job (radius-capped) does not?

Does NOT touch any production file and does NOT touch the original Phase 39
test script. Reuses production's own (already frequency/power-corrected)
_prepare_site_rows / _site_record / _cost231_for_points from
tools.lte_prediction_offset.services unmodified -- only this script's own
grid-scoring loop is new, and it deliberately scores every cell against every
grid point with no distance cutoff at all (matching original Phase 39's own
"no fixed radius or top-N cap" design, and production's own DT-scoring
_run_cost231_at_dt, which also has no cap).

Output: for 4G and 5G separately, the best (max) RSRP per grid point across
ALL cells of that technology, with no radius filtering -- directly comparable
to:
  - Phase39 (original) equal-power serving-cell numbers
  - production job 5b87c145's OWN radius-capped max-per-grid numbers
    (already computed: 4G median -84.90, 5G median -85.62, gap 0.72 dB)

Run:  ML/venv/Scripts/python.exe ML/tests/new-project/test_project39_v2_no_radius_cap.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
load_dotenv(ML_ROOT / ".env")

import tools.lte_prediction_offset.services as svc
from tools.lte_prediction.ml_engine import fetch_site_data
from utils.python_bridge import get_bridge_client

PROJECT_ID = 210
REGION = "taiwan"
OUT_DIR = THIS_DIR / "data" / "project_210_taiwan" / "phase39_v2_no_radius_cap"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Real project 210 site rows, via the SAME direct-DB path production
    #    uses, then through production's OWN _prepare_site_rows -- this is
    #    where the frequency-anchor offset + fixed 46 dBm power already live,
    #    unmodified from what the real run used.
    with svc._without_python_bridge():
        site_df_raw, operator = fetch_site_data(PROJECT_ID, region=REGION, polygon_ids=None, operator=None)
    site_df = svc._prepare_site_rows(site_df_raw, REGION)
    site_df = site_df.dropna(subset=["lat", "lon", "frequency_mhz"]).drop_duplicates("strict_cell_key")
    print(f"[P39V2] site rows prepared: {len(site_df)} cells "
          f"({(site_df['technology_key']=='4G').sum()} 4G, {(site_df['technology_key']=='5G').sum()} 5G)")

    # 2. The same 10,234-point grid production and Phase 39 both use, fetched
    #    from the live bridge (grid centers only -- geometry, not scores).
    bridge = get_bridge_client()
    grid_df, _ = bridge.get_grid_analytics(PROJECT_ID, region=REGION)
    grid_df = grid_df[["grid_id", "center_lat", "center_lon"]].drop_duplicates("grid_id").dropna()
    grid_lat = grid_df["center_lat"].to_numpy(float)
    grid_lon = grid_df["center_lon"].to_numpy(float)
    n_grid = len(grid_df)
    print(f"[P39V2] grid points: {n_grid}")

    # site_df from fetch_site_data(polygon_ids=None) is EVERY cell tagged to
    # this project across all of Taiwan (15,817 cells) -- production narrows
    # this by polygon + the 500m radius under test. To isolate the radius
    # variable alone (not re-litigate polygon membership), keep a generous
    # 15 km bounding box around the grid -- far larger than the 500m cap being
    # tested, but small enough to skip cells that could not matter regardless
    # of any radius policy (COST231 at that range is noise-floor already).
    pad_deg = 5_000.0 / 111_320.0
    lat_lo, lat_hi = grid_lat.min() - pad_deg, grid_lat.max() + pad_deg
    lon_lo, lon_hi = grid_lon.min() - pad_deg, grid_lon.max() + pad_deg
    before_n = len(site_df)
    site_df = site_df[
        site_df["lat"].between(lat_lo, lat_hi) & site_df["lon"].between(lon_lo, lon_hi)
    ].copy()
    print(f"[P39V2] site rows within 15km bounding box: {len(site_df)} of {before_n} "
          f"({(site_df['technology_key']=='4G').sum()} 4G, {(site_df['technology_key']=='5G').sum()} 5G)")

    # 3. Score EVERY cell against EVERY grid point -- no distance cap at all,
    #    exactly Phase 39's own "no fixed radius or top-N cap" design, and
    #    exactly production's own (uncapped) DT-scoring _run_cost231_at_dt.
    #    Reuses production's _site_record / _cost231_for_points unmodified.
    best = {"4G": np.full(n_grid, -np.inf), "5G": np.full(n_grid, -np.inf)}
    t_score = time.time()
    for i, (_, row) in enumerate(site_df.iterrows()):
        tech = str(row["technology_key"])
        if tech not in best:
            continue
        raw = svc._cost231_for_points(svc._site_record(row), grid_lat, grid_lon, float(row["frequency_mhz"]))
        raw = raw + float(row.get("model_rsrp_adjust_db", 0.0))
        np.maximum(best[tech], raw, out=best[tech])
        if (i + 1) % 500 == 0:
            print(f"[P39V2] scored {i + 1}/{len(site_df)} cells, elapsed={time.time() - t_score:.1f}s", flush=True)
    print(f"[P39V2] scoring done in {time.time() - t_score:.1f}s")

    bins = [-141, -115, -105, -95, -85, 0]
    labels = ["-140to-115", "-115to-105", "-105to-95", "-95to-85", "-85to0"]
    rows = []
    for tech, vals in best.items():
        vals = vals[np.isfinite(vals)]
        cats = pd.cut(vals, bins=bins, labels=labels)
        dist = (cats.value_counts(normalize=True).reindex(labels) * 100).round(1)
        rows.append({
            "technology": tech, "n_grids_covered": len(vals),
            "mean": round(float(np.mean(vals)), 2), "median": round(float(np.median(vals)), 2),
            **{f"pct_{l}": dist[l] for l in labels},
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "phase39_v2_no_radius_cap_summary.csv", index=False)
    print("\n=== Phase 39 v2 (no radius cap) -- max RSRP per grid per technology ===")
    print(result.to_string(index=False))
    print(f"\n[P39V2] total elapsed {time.time() - t0:.1f}s")
    print(f"[P39V2] saved: {OUT_DIR / 'phase39_v2_no_radius_cap_summary.csv'}")

    print("\n=== For comparison ===")
    print("Phase 39 (original, equal-power outdoor):        4G -85.6   5G -90.6   gap 5.0 dB")
    print("Phase 39 (original, calibrated_final outdoor):   4G -84.2   5G -86.2   gap 2.0 dB")
    print("Production job 5b87c145 (500m radius cap):        4G -84.90 5G -85.62  gap 0.72 dB")


if __name__ == "__main__":
    main()
