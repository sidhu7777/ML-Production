"""
Phase 18: DT-point-level diagnostic, isolating WHERE Phase 17's physical
model diverges from real DT measurements.

Read-only debug script. Does not modify Phase 15, 16, or 17 in any way -
only imports and reuses their already-written functions/output.

Why this exists: Phase 17's grid-level means show the new physical model
sitting ~18dB (4G) / ~9dB (5G) below Phase 9's DT-offset baseline, even
after fixing (a) the diffraction over-summation and (b) DT residual
leaking across clutter-class boundaries. A GRID-level mean can't tell you
whether that remaining gap is genuine dense-urban/building severity or a
base calibration problem, because it blends open, obstructed, and indoor
points together.

This script instead uses DT's own raw points (never the 25m grid) as
ground truth, and for each one:
  1. Takes the real measured RSRP and Phase 9's own assigned serving cell
     (assigned_strict_cell_key - the real production match, not
     re-derived here).
  2. Computes Phase 17's physical model prediction AT THAT EXACT POINT
     (tilt-fixed antenna + the de-duplicated single obstruction term from
     Phase 15's _geo_correction_db) - with NO DT residual applied, so
     this is the model's raw error before any calibration.
  3. Records which obstruction branch fired for that point (indoor /
     obstructed / clear) and the point's own clutter class.
  4. Groups (measured - predicted) by [technology, clutter_class, branch]
     and reports median/mean/IQR per group.

The diagnostic question this answers: is the physical model's error near
zero in clear/open/representative conditions (where nothing about this
model should differ much from Phase 9) but large in obstructed/indoor
conditions (genuine dense-urban severity, a magnitude-tuning problem) -
or is the error already large even in clear conditions (a base
calibration bug in the antenna/path-loss model itself, unrelated to
diffraction)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
for p in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit_project210_phase13_beam_check as phase13
import streamlit_project210_phase15_radius_progression as phase15
from phase_rsrp_guard import valid_model_rsrp
from test_project210_phase17_full_polygon_geo_dt_comparison import (  # reused, not reimplemented
    BASELINE_DATA_DIR,
    CLUTTER_TILES_PATH,
    N78_TECHNOLOGY_OFFSET_DB,
    PHASE9_DIR,
    RSRP_MAX,
    RSRP_MIN,
    OUT_DIR as PHASE17_OUT_DIR,
)

OUT_DIR = THIS_DIR / "data" / "project_210_taiwan" / "cost231_phase18_dt_point_diagnostic"


def _load_dt_with_clutter() -> pd.DataFrame:
    path = PHASE17_OUT_DIR / "phase17_dt_with_clutter_project210.parquet"
    return pd.read_parquet(path)


def _compute_physical_rsrp_per_dt_point(
    dt: pd.DataFrame, clutter_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Same physical model as Phase 17 (tilt-fixed antenna + the single
    non-overlapping obstruction term), evaluated at each DT point's own
    exact lat/lon using its own assigned serving cell - one row at a time
    so the obstruction branch (indoor/obstructed/clear) can be read back
    per point from _geo_correction_db's own counts dict, without needing
    to touch that function's return signature."""
    params_common = {"ue_height": 1.5, "k1": 0, "k2": 0, "cable_loss": 2.0, "antenna_gain": 18.0}
    clutter_weights = dict(phase15.DEFAULT_CLUTTER_WEIGHTS)

    physical_rsrp_unclipped = np.full(len(dt), np.nan)
    physical_rsrp = np.full(len(dt), np.nan)
    branch = np.array(["unknown"] * len(dt), dtype=object)

    n = len(dt)
    for i, row in enumerate(dt.itertuples(index=False)):
        row = row._asdict() if hasattr(row, "_asdict") else dict(zip(dt.columns, row))
        site_row = pd.Series({
            "lat": row["site_lat"], "lon": row["site_lon"], "azimuth": row["azimuth"],
            "Etilt": row["Etilt"], "Mtilt": row["Mtilt"], "Height": row["Height"], "tx_power": row["tx_power"],
        })
        site_dict = phase15._row_to_site_dict_fixed(site_row)
        freq = float(row["frequency_mhz"])
        tx_height_m = float(row["Height"]) if pd.notna(row["Height"]) else 30.0
        center_lat, center_lon = float(row["site_lat"]), float(row["site_lon"])

        raw = phase15.compute_sector_rsrp(site_dict, float(row["lat"]), float(row["lon"]), freq, params_common)
        if str(row.get("band")) == "78":
            raw = raw + N78_TECHNOLOGY_OFFSET_DB

        grid_df = pd.DataFrame({"lat": [float(row["lat"])], "lon": [float(row["lon"])]})
        correction, counts = phase15._geo_correction_db(
            grid_df, clutter_gdf, buildings_gdf, center_lat, center_lon,
            tx_height_m=tx_height_m, rx_height_m=1.5, freq_mhz=freq,
            clutter_weights=clutter_weights, building_area_weight=phase15.DEFAULT_BUILDING_AREA_WEIGHT,
            diffraction_multiplier=1.0, entry_loss_db=-15.0, entry_depth_slope_db_per_m=-0.5,
        )
        physical_rsrp_unclipped[i] = float(raw + correction[0])
        clipped = valid_model_rsrp(np.array([raw + correction[0]], dtype=float))[0]
        physical_rsrp[i] = float(clipped) if np.isfinite(clipped) else np.nan
        branch[i] = "indoor" if counts["indoor"] else ("obstructed" if counts["obstructed"] else "clear")

        if (i + 1) % 1000 == 0 or i == n - 1:
            print(f"[PHASE18] scored {i + 1}/{n} DT points", flush=True)

    out = dt.copy()
    out["physical_rsrp_unclipped"] = physical_rsrp_unclipped
    out["physical_rsrp"] = physical_rsrp
    out["obstruction_branch"] = branch
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dt = _load_dt_with_clutter()
    identity = phase13.load_identity()
    surface = pd.read_parquet(PHASE9_DIR / "phase9_directional_raw_corrected_surface_project210.parquet")
    clutter_gdf = gpd.read_file(CLUTTER_TILES_PATH)
    buildings_gdf = phase15.load_building_gdf(BASELINE_DATA_DIR)
    buildings_gdf = phase15.impute_building_heights(buildings_gdf, phase15.OBAT_CSV_PATH)

    freq_lookup = surface[["strict_cell_key", "frequency_mhz", "band"]].drop_duplicates(subset=["strict_cell_key"])
    dt = dt.merge(
        identity[["Node_Cell_ID", "lat", "lon", "azimuth", "Etilt", "Mtilt", "Height", "tx_power"]].rename(
            columns={"lat": "site_lat", "lon": "site_lon"}
        ),
        left_on="assigned_strict_cell_key", right_on="Node_Cell_ID", how="left",
    )
    dt = dt.merge(freq_lookup, left_on="assigned_strict_cell_key", right_on="strict_cell_key", how="left")

    before = len(dt)
    dt = dt.dropna(
        subset=["site_lat", "site_lon", "Etilt", "Mtilt", "Height", "tx_power", "frequency_mhz"]
    ).reset_index(drop=True)
    print(f"[PHASE18] DT points with a full identity match: {len(dt)} / {before}")

    dt = _compute_physical_rsrp_per_dt_point(dt, clutter_gdf, buildings_gdf)

    dt["phase18_error_db"] = dt["rsrp_measured"] - dt["physical_rsrp"]  # physical model only, no DT residual
    dt["phase9_error_db"] = dt["rsrp_measured"] - dt["raw_cost231_at_dt_rsrp"]  # Phase 9's own baseline error

    dt.to_parquet(OUT_DIR / "phase18_dt_point_diagnostic_project210.parquet", index=False)
    dt.to_csv(OUT_DIR / "phase18_dt_point_diagnostic_project210.csv", index=False)

    group_cols = ["assigned_technology", "clutter_class", "obstruction_branch"]
    report = (
        dt.groupby(group_cols)
        .agg(
            n=("phase18_error_db", "size"),
            phase18_error_median=("phase18_error_db", "median"),
            phase18_error_mean=("phase18_error_db", "mean"),
            phase18_error_p25=("phase18_error_db", lambda s: s.quantile(0.25)),
            phase18_error_p75=("phase18_error_db", lambda s: s.quantile(0.75)),
            phase9_error_median=("phase9_error_db", "median"),
        )
        .reset_index()
        .sort_values(["assigned_technology", "obstruction_branch", "clutter_class"])
    )
    report.to_csv(OUT_DIR / "phase18_summary_by_condition.csv", index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)
    print("\n[PHASE18] error (measured - predicted) by technology / clutter class / obstruction branch:")
    print(report.to_string(index=False))

    print("\n[PHASE18] error by technology / obstruction branch only (collapsed across clutter class):")
    coarse = (
        dt.groupby(["assigned_technology", "obstruction_branch"])
        .agg(n=("phase18_error_db", "size"), phase18_error_median=("phase18_error_db", "median"),
             phase18_error_mean=("phase18_error_db", "mean"))
        .reset_index()
    )
    print(coarse.to_string(index=False))


if __name__ == "__main__":
    main()
