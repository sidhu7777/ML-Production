"""
Phase 28: RSRP reference-level fix + geo re-derivation (4G AND 5G).

Problem established in phases 9-27:
  compute_sector_rsrp() returns  tx_power + gain - pathloss - cable_loss  and calls it RSRP.
  That is total-carrier received power, not RSRP-per-resource-element.

  4G: the DB has NO 4G power/bandwidth (46 dBm is a hardcoded fallback), so the
      total->per-RE term  -10*log10(12*N_RB)  is applied from the RSRP definition + a
      per-band bandwidth assumption, verified on clean outdoor DT (~-27.8 @ 775, ~-24.8 @ 1840).
  5G: the DB has real per-cell power (50-55 dBm) already near an SSB/per-RE reference, so
      the 5G raw is only ~5-10 dB off. The 5G reference offset is taken directly from the
      clean (clear/LOS) DT residual per band - data-anchored, NOT the 4G formula.
      (5G raw at DT is empty in the phase-26 file, so it is reconstructed exactly from the
       phase-26 physical:  raw = physical_with_terrain - building_geo_corr + terrain_loss.)

  Then, for both techs:
    - outdoor building loss: DROPPED (COST231-Hata urban already contains it)
    - terrain diffraction: kept where real
    - indoor O2I: frequency wall + per-cell saturating depth (no indoor DT)
    - Water: open/LOS - NO terrain, NO O2I, NO residual (raw_after only)
    - light per-clutter residual (Water excluded from the fit)

This phase does NOT modify phases 9/19/22/24/25/26/27 or production. Outputs under
cost231_phase28_4g_rsrp_reference_fix/ , per technology.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
for p in (ML_ROOT, THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE26_DIR = PROJECT_DIR / "cost231_phase26_corrected_obstruction_profile"
OUT_DIR = PROJECT_DIR / "cost231_phase28_4g_rsrp_reference_fix"

# ---- Documented deployment assumptions (DB has none of this for 4G) -----------------
# tx_power fallback (46 dBm) is assumed to be TOTAL carrier power, uniformly spread across
# the carrier, with reference-signal power boost RS_BOOST_DB.
RS_BOOST_DB = 0.0
RB_MAP = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100}
# Per-band channel bandwidth assumption (Taiwan macro LTE). 775.5 MHz = APT700 (Band 28),
# 1840 MHz = Band 3. These are ASSUMPTIONS - the verification block below reports the
# residual each choice produces so they can be checked / revised.
# 775.5: clean-DT residual confirms 10 MHz (per-RE -27.8 -> LOS residual ~ -2 dB).
# 1840.0: clean-DT residual (-23.2) implies an effective ~5 MHz carrier OR a real +6 dB
#   band offset on a small (n=1681) sample; 5 MHz lands the LOS residual near 0.
BAND_BANDWIDTH_MHZ = {
    775.5: 10.0,
    1840.0: 5.0,
}
DEFAULT_BANDWIDTH_MHZ = 10.0


def _per_re_reference_offset_db(freq_mhz: float) -> float:
    """dB to ADD to the current 4G raw output to turn total-carrier power into per-RE RSRP.
    = -(10*log10(12 * N_RB)) + RS_BOOST_DB   (a negative number, ~ -28)."""
    bw = BAND_BANDWIDTH_MHZ.get(round(float(freq_mhz), 1), DEFAULT_BANDWIDTH_MHZ)
    n_rb = RB_MAP.get(bw, 50)
    return -(10.0 * math.log10(12 * n_rb)) + RS_BOOST_DB


# ---- indoor O2I (no indoor DT exists - pure physics, frequency + depth) --------------
def _o2i_wall_db(freq_mhz: float) -> float:
    return 8.5 + 9.5 * math.log10(max(float(freq_mhz), 1.0) / 1000.0)


def _indoor_o2i_db(freq_mhz: float, depth_m: float) -> float:
    depth = 8.0 * (1.0 - math.exp(-max(float(depth_m), 0.0) / 12.0))
    return _o2i_wall_db(freq_mhz) + depth


def _shrink_residual_table(train: pd.DataFrame, keys: list[str], resid_col: str, shrink_n: float) -> pd.DataFrame:
    t = (
        train.dropna(subset=[resid_col])
        .groupby(keys, dropna=False)
        .agg(n=(resid_col, "size"), med=(resid_col, "median"))
        .reset_index()
    )
    t["correction_db"] = t["med"] * t["n"] / (t["n"] + shrink_n)
    return t


def _metrics(measured: pd.Series, predicted: pd.Series) -> dict:
    err = pd.to_numeric(measured, errors="coerce") - pd.to_numeric(predicted, errors="coerce")
    a = err.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if a.size == 0:
        return {"n": 0, "mae": math.nan, "bias": math.nan, "median": math.nan, "p90_abs": math.nan}
    return {
        "n": int(a.size),
        "mae": float(np.mean(np.abs(a))),
        "bias": float(np.mean(a)),
        "median": float(np.median(a)),
        "p90_abs": float(np.quantile(np.abs(a), 0.9)),
    }


def _dt_raw_unclipped(dt_tech: pd.DataFrame) -> pd.Series:
    """Raw COST231 RSRP (unclipped, pre-geo) at each DT point.
    4G has it directly; the 5G column is empty in the phase-26 file, so reconstruct from
    the phase-26 physical (verified exact on 4G, MAE ~1e-15):
        raw = phase26_physical_with_terrain_unclipped - building_geo_correction + terrain_loss
    """
    direct = pd.to_numeric(dt_tech.get("raw_cost231_at_dt_rsrp_unclipped"), errors="coerce")
    if direct is not None and direct.notna().mean() > 0.5:
        return direct
    phys = pd.to_numeric(dt_tech["phase26_physical_with_terrain_rsrp_unclipped"], errors="coerce")
    bgc = pd.to_numeric(dt_tech["building_geo_correction_db"], errors="coerce").fillna(0.0)
    terr = pd.to_numeric(dt_tech["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return phys - bgc + terr


def _clean_branch_ref_offset(dt_tech: pd.DataFrame, raw_col: str) -> dict:
    """Data-anchored reference offset (5G): per-frequency median (measured - raw) on
    clear/LOS DT.  A negative number ~ -10 dB."""
    clr = dt_tech[dt_tech["obstruction_branch"].astype(str) == "clear"].copy()
    clr["freq_r"] = pd.to_numeric(clr["frequency_mhz"], errors="coerce").round(1)
    resid = pd.to_numeric(clr["rsrp_measured"], errors="coerce") - pd.to_numeric(clr[raw_col], errors="coerce")
    return {float(k): float(v) for k, v in resid.groupby(clr["freq_r"]).median().dropna().items()}


def _run_tech(tech: str, dt_all: pd.DataFrame, cand_all: pd.DataFrame, bounds: pd.DataFrame) -> dict:
    tl = tech.lower()
    dt = dt_all[dt_all["assigned_technology"].astype(str) == tech].copy()
    c = cand_all[cand_all["technology"].astype(str) == tech].copy()
    if dt.empty or c.empty:
        print(f"[PHASE28] {tech}: no data - skipped")
        return {}

    dt["freq_r"] = pd.to_numeric(dt["frequency_mhz"], errors="coerce").round(1)
    c["freq_r"] = pd.to_numeric(c["frequency_mhz"], errors="coerce").round(1)
    dt["dt_raw_unclipped"] = _dt_raw_unclipped(dt).to_numpy()

    if tech == "4G":
        ref_method = "-10*log10(12*N_RB) per-band bandwidth assumption, verified on clean DT"
        offset_map = {float(b): _per_re_reference_offset_db(float(b))
                      for b in dt["freq_r"].dropna().unique()}
    else:
        ref_method = "median clear/LOS DT residual (measured - reconstructed raw), per band"
        offset_map = _clean_branch_ref_offset(dt.rename(columns={"dt_raw_unclipped": "_raw"}), "_raw")
    default_off = float(np.median(list(offset_map.values()))) if offset_map else 0.0

    def _off(freq_series) -> np.ndarray:
        f = pd.to_numeric(freq_series, errors="coerce").round(1)
        return f.map(lambda x: offset_map.get(float(x), default_off) if pd.notna(x) else default_off).to_numpy(dtype=float)

    print(f"[PHASE28] {tech} reference offset (dB to add): "
          f"{ {k: round(v, 2) for k, v in offset_map.items()} }   [{ref_method}]")

    # ---------------- candidate physical ----------------
    c["phase28_per_re_db"] = _off(c["freq_r"])
    c["phase28_raw_rsrp"] = pd.to_numeric(c["raw_cost231_rsrp_unclipped"], errors="coerce") + c["phase28_per_re_db"]
    branch = c["obstruction_branch"].astype(str)
    terrain_loss = pd.to_numeric(c["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0).clip(lower=0.0)
    bgc = pd.to_numeric(c["building_geo_correction_db"], errors="coerce")
    depth_m = (-(bgc + 15.0) / 0.5).clip(lower=0.0, upper=40.0)
    o2i = np.array([_indoor_o2i_db(f, dm) for f, dm in zip(c["freq_r"], depth_m)])
    c["phase28_o2i_db"] = np.where(branch == "indoor", o2i, 0.0)
    c["phase28_terrain_db"] = terrain_loss

    water_c = c["clutter_class"].astype(str) == "Water"
    c.loc[water_c, "phase28_terrain_db"] = 0.0
    c.loc[water_c, "phase28_o2i_db"] = 0.0
    c["phase28_physical_rsrp"] = c["phase28_raw_rsrp"] - c["phase28_terrain_db"] - c["phase28_o2i_db"]

    # ---------------- DT physical + residual fit ----------------
    dtv = dt.copy()
    dtv["per_re_db"] = _off(dtv["freq_r"])
    dtv_terr = pd.to_numeric(dtv["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0).clip(lower=0.0)
    dtv_bgc = pd.to_numeric(dtv["building_geo_correction_db"], errors="coerce")
    dtv_depth = (-(dtv_bgc + 15.0) / 0.5).clip(lower=0.0, upper=40.0)
    dtv_o2i = np.array([_indoor_o2i_db(f, dm) for f, dm in zip(dtv["freq_r"], dtv_depth)])
    dtv_water = dtv["clutter_class"].astype(str) == "Water"
    br_dt = dtv["obstruction_branch"].astype(str)
    dtv_terr_eff = np.where(dtv_water, 0.0, dtv_terr)
    dtv_o2i_eff = np.where(dtv_water | (br_dt != "indoor"), 0.0, dtv_o2i)
    dtv["phase28_physical_rsrp"] = dtv["dt_raw_unclipped"] + dtv["per_re_db"] - dtv_terr_eff - dtv_o2i_eff
    dtv["raw_after"] = dtv["dt_raw_unclipped"] + dtv["per_re_db"]
    dtv["resid"] = pd.to_numeric(dtv["rsrp_measured"], errors="coerce") - dtv["phase28_physical_rsrp"]
    dtv["clutter_class"] = dtv["clutter_class"].astype("object").where(dtv["clutter_class"].notna(), "UNKNOWN")

    split_key = pd.util.hash_pandas_object(dtv["dt_row_id"].astype(str), index=False).astype("uint64")
    dtv["split"] = np.where((split_key % 10) < 7, "train", "validation")
    tr = dtv[dtv.split == "train"]

    # Water excluded from the residual fit (unreliable bridge/coast/GPS-drift DT).
    tr_fit = tr[tr["clutter_class"].astype(str) != "Water"]
    global_corr = float(tr_fit["resid"].median())
    clutter_tbl = _shrink_residual_table(tr_fit.assign(_r=tr_fit["resid"] - global_corr),
                                         ["clutter_class"], "_r", shrink_n=40.0)
    clutter_map = dict(zip(clutter_tbl["clutter_class"].astype(str), clutter_tbl["correction_db"]))

    def _residual_for(clut: pd.Series) -> np.ndarray:
        k = clut.astype(str)
        base = global_corr + k.map(lambda x: clutter_map.get(x, 0.0)).to_numpy(dtype=float)
        return np.where(k.to_numpy() == "Water", 0.0, base)   # Water: no residual, raw_after only

    c["phase28_residual_db"] = _residual_for(c["clutter_class"].astype("object").where(c["clutter_class"].notna(), "UNKNOWN"))
    c["phase28_final_rsrp_unclipped"] = c["phase28_physical_rsrp"] + c["phase28_residual_db"]
    c["phase28_final_rsrp"] = c["phase28_final_rsrp_unclipped"].where(c["phase28_final_rsrp_unclipped"] >= -140.0, np.nan)

    dtv["phase28_residual_db"] = _residual_for(dtv["clutter_class"])
    dtv["phase28_final_rsrp"] = dtv["phase28_physical_rsrp"] + dtv["phase28_residual_db"]
    va = dtv[dtv.split == "validation"]

    dtv_scored = dtv[[
        "dt_row_id", "lat", "lon", "rsrp_measured", "obstruction_branch", "clutter_class", "split",
        "dt_raw_unclipped", "per_re_db",
        "phase28_physical_rsrp", "phase28_residual_db", "phase28_final_rsrp",
    ]].rename(columns={"dt_raw_unclipped": "raw_cost231_at_dt_rsrp_unclipped"})
    dtv_scored.to_parquet(OUT_DIR / f"phase28_{tl}_dt_scored_project210.parquet", index=False)
    dtv_scored.to_csv(OUT_DIR / f"phase28_{tl}_dt_scored_project210.csv", index=False)

    dt_ref = dtv[[
        "dt_row_id", "lat", "lon", "rsrp_measured", "assigned_technology", "frequency_mhz",
        "obstruction_branch", "clutter_class", "dt_raw_unclipped", "per_re_db", "raw_after",
    ]].rename(columns={"dt_raw_unclipped": "raw_cost231_at_dt_rsrp_unclipped"})
    dt_ref.to_parquet(OUT_DIR / f"phase28_{tl}_dt_reference_check_project210.parquet", index=False)
    dt_ref.to_csv(OUT_DIR / f"phase28_{tl}_dt_reference_check_project210.csv", index=False)

    def by_branch(frame: pd.DataFrame, pred_col: str) -> dict:
        return {str(b): _metrics(g["rsrp_measured"], g[pred_col]) for b, g in frame.groupby("obstruction_branch")}

    verify = {
        "reference_method": ref_method,
        "reference_offset_db": {str(k): round(v, 2) for k, v in offset_map.items()},
        "residual_vs_RAW_before_fix": by_branch(dtv, "dt_raw_unclipped"),
        "residual_vs_RAW_after_fix": by_branch(dtv, "raw_after"),
    }

    # ---------------- serving grid: best server AND frontend (mean of candidates) -------
    c["_env"] = np.where(branch == "indoor", "indoor", "outdoor")
    best = c.sort_values("phase28_final_rsrp_unclipped").groupby("grid_id").tail(1)
    frontend = c.groupby("grid_id").agg(
        phase28_physical_mean_rsrp=("phase28_physical_rsrp", "mean"),
        phase28_final_mean_rsrp=("phase28_final_rsrp", "mean"),
    ).reset_index()
    serving = bounds.merge(
        best[["grid_id", "phase28_physical_rsrp", "phase28_final_rsrp", "phase28_final_rsrp_unclipped",
              "phase28_o2i_db", "phase28_terrain_db", "phase28_residual_db", "_env"]],
        on="grid_id", how="left",
    ).merge(frontend, on="grid_id", how="left").rename(
        columns={"_env": "serving_environment",
                 "phase28_physical_rsrp": "phase28_physical_best_rsrp",
                 "phase28_final_rsrp": "phase28_final_best_rsrp"}
    )
    serving.to_parquet(OUT_DIR / f"phase28_{tl}_serving_grid_project210.parquet", index=False)
    serving.to_csv(OUT_DIR / f"phase28_{tl}_serving_grid_project210.csv", index=False)
    c.to_parquet(OUT_DIR / f"phase28_{tl}_scored_candidates_project210.parquet", index=False)

    sv = pd.to_numeric(serving["phase28_final_best_rsrp"], errors="coerce")
    env = serving["serving_environment"]
    pipeline = {
        "reference_method": ref_method,
        "reference_offset_db": {str(k): round(v, 2) for k, v in offset_map.items()},
        "held_out_dt_final": _metrics(va["rsrp_measured"], va["phase28_final_rsrp"]),
        "held_out_dt_physical_no_residual": _metrics(va["rsrp_measured"], va["phase28_physical_rsrp"]),
        "held_out_dt_by_branch_final": by_branch(va, "phase28_final_rsrp"),
        "residual_correction": {
            "global_db": round(global_corr, 2),
            "per_clutter_db": {k: round(v, 2) for k, v in clutter_map.items()},
        },
        "serving_grid": {
            "rows": int(len(serving)),
            "no_coverage_rows": int(sv.isna().sum()),
            "median_rsrp": float(sv.median()),
            "outdoor_median": float(pd.to_numeric(serving.loc[env == "outdoor", "phase28_final_best_rsrp"], errors="coerce").median()),
            "indoor_median": float(pd.to_numeric(serving.loc[env == "indoor", "phase28_final_best_rsrp"], errors="coerce").median()),
            "indoor_o2i_db_median": float(pd.to_numeric(c.loc[branch == "indoor", "phase28_o2i_db"], errors="coerce").median()),
        },
        "verification": verify,
    }
    b_before = verify["residual_vs_RAW_before_fix"].get("clear", {}).get("median", float("nan"))
    b_after = verify["residual_vs_RAW_after_fix"].get("clear", {}).get("median", float("nan"))
    print(f"  {tech} clear-branch resid vs RAW  before {b_before:.1f}  ->  after {b_after:.1f}")
    print(f"  {tech} held-out DT MAE  physical {pipeline['held_out_dt_physical_no_residual']['mae']:.2f}"
          f"  ->  final {pipeline['held_out_dt_final']['mae']:.2f}  (bias {pipeline['held_out_dt_final']['bias']:.2f},"
          f" p90|err| {pipeline['held_out_dt_final']['p90_abs']:.1f})")
    print(f"  {tech} serving: {pipeline['serving_grid']['rows']} cells, {pipeline['serving_grid']['no_coverage_rows']} no-cov,"
          f" median {pipeline['serving_grid']['median_rsrp']:.1f}"
          f"  (outdoor {pipeline['serving_grid']['outdoor_median']:.1f} / indoor {pipeline['serving_grid']['indoor_median']:.1f})")
    return pipeline


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dt_all = pd.read_parquet(PHASE26_DIR / "phase26_dt_scored_project210.parquet")
    cand_all = pd.read_parquet(PHASE26_DIR / "phase26_scored_candidates_project210.parquet")
    grid = pd.read_parquet(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210.parquet")
    bounds = grid[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]]

    tech_summary = {}
    for tech in ("4G", "5G"):
        res = _run_tech(tech, dt_all, cand_all, bounds)
        if res:
            tech_summary[tech] = res

    summary = {
        "scope": "4G and 5G. RSRP reference-level fix + geo re-derivation + Water override. "
                 "No production or phase 9/19/22/24/25/26/27 file modified.",
        "reference_method": {
            "4G": "-10*log10(12*N_RB) from per-band bandwidth assumption (775.5=10 MHz, 1840=5 MHz), verified on clean DT",
            "5G": "median clear/LOS DT residual per band (data-anchored; 5G raw reconstructed from the phase-26 "
                  "physical, 5G tx_power is real and already near an SSB/per-RE reference)",
        },
        "assumptions": {"RS_boost_db": RS_BOOST_DB, "band_bandwidth_mhz": BAND_BANDWIDTH_MHZ},
        "geo_losses": {
            "outdoor_building_loss": "dropped (COST231-Hata urban already contains it; DT residual ~0)",
            "terrain_diffraction": "kept where >0 (real hills)",
            "indoor_o2i": "frequency wall + per-cell saturating depth (no indoor DT exists)",
            "water": "open/LOS - NO terrain, NO O2I, NO residual",
        },
        "technology": tech_summary,
        # back-compat: the existing dashboard block reads top-level pipeline/verification (== 4G)
        "pipeline": tech_summary.get("4G", {}),
        "verification": tech_summary.get("4G", {}).get("verification", {}),
    }
    (OUT_DIR / "phase28_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[PHASE28] wrote 4G + 5G outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
