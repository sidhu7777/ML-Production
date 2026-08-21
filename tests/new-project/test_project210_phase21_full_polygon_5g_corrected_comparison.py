"""
Phase 21: reruns Phase 17's full-polygon geo+DT-residual computation for
5G ONLY, sourcing DT from Phase 20's corrected dataset (8,033 real
network='5G NSA'/'5G SA' rows, confirmed inside project 210's polygon in
the live Taiwan DB) instead of the 136 proximity-mislabeled LTE-anchor
rows Phase 17/18/19 used until now.

4G is untouched - its DT was always correct (11,322 real rows), so this
script just copies Phase 17's already-computed 4G serving grid output
as-is rather than recomputing it.

Reuses Phase 17's own functions via import (same proven pattern Phase 18
and Phase 19 already use) - does not modify streamlit_project210_phase15_
radius_progression.py, test_project210_phase17_full_polygon_geo_dt_
comparison.py, or any other existing file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
for p in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit_project210_phase13_beam_check as phase13
import test_project210_phase17_full_polygon_geo_dt_comparison as phase17

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE20_DIR = PROJECT_DIR / "cost231_phase20_5g_real_dt_match"
OUT_DIR = PROJECT_DIR / "cost231_phase21_full_polygon_5g_corrected"
RSRP_MIN, RSRP_MAX = phase17.RSRP_MIN, phase17.RSRP_MAX


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corrected_dt_path = PHASE20_DIR / "phase9_dt_match_project210_corrected.parquet"
    dt_all = pd.read_parquet(corrected_dt_path)
    print(f"[PHASE21] corrected DT rows: {len(dt_all)} "
          f"({(dt_all['assigned_technology'] == '4G').sum()} 4G, {(dt_all['assigned_technology'] == '5G').sum()} 5G)")

    identity = phase13.load_identity()
    clutter_gdf, buildings_gdf = phase17._load_clutter_and_buildings()
    dt_all = phase17._classify_dt_clutter(dt_all, clutter_gdf)

    serving_all = phase17._build_serving_grid()  # same base grid Phase 17 uses - doesn't depend on DT source
    surface_all = pd.read_parquet(phase17.PHASE9_DIR / "phase9_directional_raw_corrected_surface_project210.parquet")

    # 4G: unchanged DT, so just reuse Phase 17's already-computed output directly.
    phase17_4g_path = phase17.OUT_DIR / "phase17_serving_grid_4g_project210.parquet"
    serving_4g = pd.read_parquet(phase17_4g_path)
    serving_4g.to_parquet(OUT_DIR / "phase21_serving_grid_4g_project210.parquet", index=False)
    print(f"[PHASE21] 4G: copied unchanged from Phase 17 ({len(serving_4g)} rows) - its DT was already correct")

    # 5G: recompute using the corrected, real 8,033-row DT dataset.
    technology = "5G"
    print(f"\n[PHASE21] ==== {technology} (corrected DT) ====")
    serving = serving_all[serving_all["technology"] == technology].copy().reset_index(drop=True)
    candidates = surface_all[surface_all["technology"] == technology].copy().reset_index(drop=True)

    candidates = phase17._compute_physical_and_geo_all_candidates(candidates, identity, clutter_gdf, buildings_gdf)
    dt_res_by_grid = phase17._dt_residual_by_grid_id(candidates[["grid_id", "lat", "lon"]], dt_all, technology, clutter_gdf)
    agg = phase17._aggregate_candidates(candidates, dt_res_by_grid)

    winner_detail = candidates.merge(dt_res_by_grid, on="grid_id", how="left")
    winner_detail = serving[["grid_id", "strict_cell_key"]].merge(
        winner_detail[["grid_id", "strict_cell_key", "physical_rsrp", "geo_correction_db", "dt_residual_db"]],
        on=["grid_id", "strict_cell_key"], how="left",
    )
    serving = serving.merge(winner_detail[["grid_id", "physical_rsrp", "geo_correction_db", "dt_residual_db"]], on="grid_id", how="left")
    serving = serving.merge(agg, on="grid_id", how="left")
    # _aggregate_candidates (reused verbatim from Phase 17's module) always names its
    # output columns phase17_rsrp_agg / phase17_frontend_mean_rsrp - rename to phase21_* here.
    serving["phase21_rsrp"] = serving["phase17_rsrp_agg"].astype(float).clip(RSRP_MIN, RSRP_MAX)
    serving["phase21_frontend_mean_rsrp"] = serving["phase17_frontend_mean_rsrp"].astype(float).clip(RSRP_MIN, RSRP_MAX)

    out_path = OUT_DIR / f"phase21_serving_grid_{technology.lower()}_project210.parquet"
    serving.to_parquet(out_path, index=False)
    serving.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"[PHASE21] wrote {out_path} ({len(serving)} rows)")

    # comparison against Phase 9 and the old (136-DT-point) Phase 17/19 5G numbers
    old_phase17 = pd.read_parquet(phase17.OUT_DIR / "phase17_serving_grid_5g_project210.parquet")
    summary = {
        "5G": {
            "grid_rows": int(len(serving)),
            "n_dt_points_used": int((dt_all["assigned_technology"] == "5G").sum()),
            "n_dt_points_used_OLD": 136,  # the old proximity-mislabeled count, for reference
            "mean_corrected_rsrp_phase9": float(serving["corrected_rsrp"].mean()),
            "mean_phase21_rsrp_NEW": float(serving["phase21_rsrp"].mean()),
            "mean_phase17_rsrp_OLD": float(old_phase17["phase17_rsrp"].mean()),
            "mean_phase21_frontend_mean_rsrp_NEW": float(serving["phase21_frontend_mean_rsrp"].mean()),
            "mean_phase17_frontend_mean_rsrp_OLD": float(old_phase17["phase17_frontend_mean_rsrp"].mean()),
            "mean_geo_correction_db": float(serving["geo_correction_db"].mean()),
            "mean_dt_residual_db_NEW": float(serving["dt_residual_db"].mean()),
            "mean_dt_residual_db_OLD": float(old_phase17["dt_residual_db"].mean()),
            "gap_to_phase9_NEW": float(serving["phase21_rsrp"].mean() - serving["corrected_rsrp"].mean()),
            "gap_to_phase9_OLD": float(old_phase17["phase17_rsrp"].mean() - old_phase17["corrected_rsrp"].mean()),
        }
    }
    (OUT_DIR / "phase21_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
    print("\n[PHASE21] summary (NEW = corrected 8,033-point DT, OLD = broken 136-point DT):")
    print(pd.Series(summary).to_json(indent=2))


if __name__ == "__main__":
    main()
