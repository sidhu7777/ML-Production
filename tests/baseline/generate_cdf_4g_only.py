"""
Same 3-curve CDF comparison as generate_cdf_graphs.py, but restricted to
ONLY the real 4G bands (Band 28 @775.5MHz, Band 3 @1840MHz) - deliberately
excluding n78/5G, which is the one band where real drive-test ground truth
is thin/approximated this session. Band 28 and Band 3 both have abundant,
genuinely real, correctly-labeled DT coverage already (no imputation), so
this is a clean test of Stage 1 (COST-231) alone against real data, with
no data-quality caveat attached.

Only Stage 1 (raw COST-231 + antenna, straight from run_rf_prediction_fast)
is computed here - no clutter classification, indoor-loss, or DT-fitting
needed, since the CDF comparison only ever used the raw baseline value.
Much faster than the full project-wide trace as a result.

Saves one combined image (matching cdf_4_combined.png's 3-curve style) to
the SAME cdf_graphs folder: cdf_5_4g_only_combined.png

Test-case only. Local cache only.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from tools.lte_prediction.ml_engine import run_rf_prediction_fast
from tools.lte_prediction.geo_correction_pipeline import (
    normalize_site_for_geo, align_project_polygon_to_points, _prepare_drive_measurements,
)
from tests.baseline.test_single_site_pipeline import attach_pred_cols_to_dt_points
from tests.baseline.generate_cdf_graphs import cdf_xy, plot_single_cdf, CDF_DIR

DATA_DIR = Path(__file__).parent / "data" / "project_210_taiwan"
FOUR_G_BANDS_SITE_DF = [28.0, 3.0]        # site_df's real "band" values (numeric)
FOUR_G_BANDS_DRIVE_DF = ["B28", "B3"]     # drive_df's real "band" text labels


def main():
    site_df = pd.read_csv(DATA_DIR / "site_df.csv", low_memory=False)
    drive_cache_path = DATA_DIR / "drive_df_n78_fixed.csv"
    drive_df = pd.read_csv(drive_cache_path if drive_cache_path.exists() else DATA_DIR / "drive_df.csv", low_memory=False)
    building_df = pd.read_csv(DATA_DIR / "building_df.csv", low_memory=False)

    site_df_4g = site_df[pd.to_numeric(site_df["band"], errors="coerce").isin(FOUR_G_BANDS_SITE_DF)].copy()
    print(f"[4G-CDF] 4G-only site_df: {len(site_df_4g)} rows (bands {FOUR_G_BANDS_SITE_DF}) "
          f"out of {len(site_df)} total, {site_df_4g['Site ID'].nunique()} real sites")

    # real project polygon, aligned the same way run_project_wide_trace.py does
    real_polygon_gdf = gpd.read_file(DATA_DIR / "project_polygon.geojson")
    real_polygon_gdf = real_polygon_gdf.set_crs("EPSG:4326") if real_polygon_gdf.crs is None else real_polygon_gdf.to_crs("EPSG:4326")
    if real_polygon_gdf.geometry.name != "geometry":
        real_polygon_gdf = real_polygon_gdf.rename_geometry("geometry")
    site_norm_4g = normalize_site_for_geo(site_df_4g)
    polygon_gdf, polygon_alignment = align_project_polygon_to_points(real_polygon_gdf[["geometry"]], site_norm_4g)
    print(f"[4G-CDF] polygon_alignment={polygon_alignment}")
    real_polygon_wkt = polygon_gdf.geometry.iloc[0].wkt

    params = {
        "project_id": 210, "region": "taiwan", "radius": 2500.0, "grid": 25.0,
        "workers": 8, "max_interference_sites": 10, "use_frontend_grid_sampling": False,
        "samples_per_grid_axis": 3, "max_cells_per_grid": 3, "min_cells_per_grid": 1,
        "ensure_all_cells": True, "min_grids_per_cell": 1,
        "min_candidate_rsrp_dbm": -128, "candidate_safety_cap": 20,
        "polygon_wkt": real_polygon_wkt,
    }

    print("[4G-CDF] STAGE 1: running COST-231 for the 4G-only site set...")
    pred_df = run_rf_prediction_fast(site_df_4g, drive_df, building_df, params)
    print(f"[4G-CDF] STAGE 1 done: {len(pred_df)} points, rsrp_range=({pred_df['pred_rsrp'].min():.2f},{pred_df['pred_rsrp'].max():.2f})")

    # Real DT measured RSRP, restricted to REAL (not imputed) 4G-band rows only
    dt_4g = drive_df[drive_df["band"].isin(FOUR_G_BANDS_DRIVE_DF)].copy()
    print(f"[4G-CDF] real 4G-band DT rows: {len(dt_4g)} (out of {len(drive_df)} total)")
    dt_4g_meas = _prepare_drive_measurements(dt_4g)
    dt_measured_4g = pd.to_numeric(dt_4g_meas["RSRP_meas"], errors="coerce").dropna().to_numpy()

    matched = attach_pred_cols_to_dt_points(dt_4g_meas, pred_df, ["pred_rsrp"])
    baseline_at_dt_points_4g = pd.to_numeric(matched["pred_rsrp"], errors="coerce").dropna().to_numpy()
    baseline_full_grid_4g = pd.to_numeric(pred_df["pred_rsrp"], errors="coerce").dropna().to_numpy()

    print(f"[4G-CDF] n: dt_measured={len(dt_measured_4g)}, baseline_full_grid={len(baseline_full_grid_4g)}, "
          f"baseline_at_dt_points={len(baseline_at_dt_points_4g)}")

    fig, ax = plt.subplots(figsize=(10, 7))
    for values, label, color in [
        (dt_measured_4g, f"Drive-test measured, REAL 4G only (n={len(dt_measured_4g):,})", "#1baf7a"),
        (baseline_full_grid_4g, f"Raw baseline, full 4G grid (n={len(baseline_full_grid_4g):,})", "#2a78d6"),
        (baseline_at_dt_points_4g, f"Raw baseline, at 4G DT points only (n={len(baseline_at_dt_points_4g):,})", "#eb6834"),
    ]:
        x, y = cdf_xy(values)
        ax.plot(x, y, label=label, color=color, linewidth=2)
    ax.set_title("CDF comparison — 4G ONLY (Band 28 @775.5MHz + Band 3 @1840MHz), real DT, no imputation")
    ax.set_xlabel("RSRP (dBm)")
    ax.set_ylabel("Cumulative probability")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.annotate(
        "4G (B28/B3) has abundant, genuinely real, correctly-labeled DT coverage - no imputation caveat.\n"
        "If the blue/orange curves track the green curve closely, Stage 1 (COST-231) itself is sound and\n"
        "the problem is downstream (Stage 2/2b correction), not the base physics.",
        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    out_path = CDF_DIR / "cdf_5_4g_only_combined.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[4G-CDF] wrote {out_path}")


if __name__ == "__main__":
    main()
