"""
Phase 31: real per-electrical-tilt antenna pattern on TOP of the Phase 28 pipeline
(NOT Phase 27). 4G AND 5G.

Base = Phase 28:  raw COST231 + RSRP reference fix (4G: -10*log10(12*N_RB);
                  5G: data-anchored clear/LOS DT offset)
                  - terrain diffraction  - indoor O2I  + Water override
                  + light per-clutter residual (Water excluded)

Phase 31 adds, per candidate:
    gain_delta = real_pap_gain(CCVVPX308 for 4G / Kathrein 800109221 for 5G,
                               cell e-tilt, az-offset, depression)
               - generic_3gpp_gain(18 dBi / 65 H / 6 V)
to the raw-after level, then re-fits the per-clutter residual on the antenna-adjusted
physical. The antenna gain is a pure additive term, so this is exact.

Reuses:
  - Phase 28 helpers: _per_re_reference_offset_db, _clean_branch_ref_offset,
    _dt_raw_unclipped, _indoor_o2i_db, _shrink_residual_table, _metrics + summary offsets
  - Phase 29 antenna:  _antenna_gain_delta, _bearing_deg

Nothing in phases 9-30 or production is modified.
Output: data/project_210_taiwan/cost231_phase31_phase28_real_antenna/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
for p in (ML_ROOT, THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_project210_phase28_4g_rsrp_reference_fix as phase28
import test_project210_phase29_real_antenna_pattern as phase29

# -------------------------------------------------------------------------------------
# Phase 31 local fix (does NOT modify phase29's file): the raw .pap pattern has ~1 deg
# nulls of -40..-60 dB. Applied at a single ray angle they inject huge single-point
# errors. Real signal arrives over an angular spread + multipath, so a coverage model
# must not sample one bin. Replace phase29._pat_gain with a power-averaged (+/- 3 deg)
# read, and hard-cap the resulting antenna gain delta.
ANTENNA_DELTA_CLIP_DB = (-15.0, 12.0)
_PAT_HALF_WINDOW_DEG = 3


def _pat_gain_smoothed(start: int, arr, angle_deg):
    a = np.asarray(arr, dtype=float)
    base = np.round(np.asarray(angle_deg, dtype=float)).astype(int)
    stack = np.stack([a[(base + k - start) % len(a)] for k in range(-_PAT_HALF_WINDOW_DEG, _PAT_HALF_WINDOW_DEG + 1)])
    return 10.0 * np.log10(np.mean(10.0 ** (stack / 10.0), axis=0))


phase29._pat_gain = _pat_gain_smoothed   # phase29._antenna_gain_delta resolves this at call time

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE26_DIR = PROJECT_DIR / "cost231_phase26_corrected_obstruction_profile"
PHASE28_DIR = PROJECT_DIR / "cost231_phase28_4g_rsrp_reference_fix"
OUT_DIR = PROJECT_DIR / "cost231_phase31_phase28_real_antenna"


def _dt_geometry(dt: pd.DataFrame) -> pd.DataFrame:
    """distance_m + azimuth_delta_deg for DT points (Phase 26 DT has site+point but not these)."""
    out = dt.copy()
    slat = pd.to_numeric(out["site_lat"], errors="coerce").to_numpy()
    slon = pd.to_numeric(out["site_lon"], errors="coerce").to_numpy()
    rlat = pd.to_numeric(out["lat"], errors="coerce").to_numpy()
    rlon = pd.to_numeric(out["lon"], errors="coerce").to_numpy()
    cos0 = np.cos(np.radians(np.nanmean(rlat)))
    out["distance_m"] = np.maximum(
        np.sqrt(((slon - rlon) * 111320.0 * cos0) ** 2 + ((slat - rlat) * 110540.0) ** 2), 1.0
    )
    brg = phase29._bearing_deg(slat, slon, rlat, rlon)
    az = pd.to_numeric(out["azimuth"], errors="coerce").fillna(0.0).to_numpy()
    out["azimuth_delta_deg"] = np.abs((brg - az + 180.0) % 360.0 - 180.0)
    return out


def _offset_map(tech: str, p28sum: dict) -> tuple[dict, float]:
    """Reuse the exact Phase 28 reference offset per band (keeps Phase 31 a strict
    add-on to the Phase 28 base)."""
    raw = p28sum.get("technology", {}).get(tech, {}).get("reference_offset_db", {})
    m = {float(k): float(v) for k, v in raw.items()}
    default = float(np.median(list(m.values()))) if m else 0.0
    return m, default


def _run_tech(tech: str, cand_all: pd.DataFrame, dt_all: pd.DataFrame,
              bounds: pd.DataFrame, p28sum: dict) -> dict:
    tl = tech.lower()
    offset_map, default_off = _offset_map(tech, p28sum)
    if not offset_map:
        print(f"[PHASE31] {tech}: no Phase 28 reference offset - skipped")
        return {}

    def _off(freq_series) -> np.ndarray:
        f = pd.to_numeric(freq_series, errors="coerce").round(1)
        return f.map(lambda x: offset_map.get(float(x), default_off) if pd.notna(x) else default_off).to_numpy(float)

    # ---------------- candidates ----------------
    c = cand_all[cand_all["technology"].astype(str) == tech].copy()
    c["freq_r"] = pd.to_numeric(c["frequency_mhz"], errors="coerce").round(1)
    c["per_re_db"] = _off(c["freq_r"])
    c["antenna_gain_delta_db"] = np.clip(phase29._antenna_gain_delta(c), *ANTENNA_DELTA_CLIP_DB)
    c["raw_after"] = (
        pd.to_numeric(c["raw_cost231_rsrp_unclipped"], errors="coerce")
        + c["per_re_db"] + c["antenna_gain_delta_db"]
    )

    branch = c["obstruction_branch"].astype(str)
    terrain = pd.to_numeric(c["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0).clip(lower=0.0)
    bgc = pd.to_numeric(c["building_geo_correction_db"], errors="coerce")
    depth_m = (-(bgc + 15.0) / 0.5).clip(lower=0.0, upper=40.0)
    o2i = np.array([phase28._indoor_o2i_db(f, dm) for f, dm in zip(c["freq_r"], depth_m)])
    c["o2i_db"] = np.where(branch == "indoor", o2i, 0.0)
    c["terrain_db"] = terrain
    water_c = c["clutter_class"].astype(str) == "Water"
    c.loc[water_c, ["terrain_db", "o2i_db"]] = 0.0
    c["physical_rsrp"] = c["raw_after"] - c["terrain_db"] - c["o2i_db"]

    # ---------------- DT (same recipe) ----------------
    dt = dt_all[dt_all["assigned_technology"].astype(str) == tech].copy()
    dt["technology"] = tech
    dt = _dt_geometry(dt)
    dt["freq_r"] = pd.to_numeric(dt["frequency_mhz"], errors="coerce").round(1)
    dt["per_re_db"] = _off(dt["freq_r"])
    dt["dt_raw_unclipped"] = phase28._dt_raw_unclipped(dt).to_numpy()
    dt["antenna_gain_delta_db"] = np.clip(phase29._antenna_gain_delta(dt), *ANTENNA_DELTA_CLIP_DB)
    dt_terr = pd.to_numeric(dt["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0).clip(lower=0.0)
    dt_bgc = pd.to_numeric(dt["building_geo_correction_db"], errors="coerce")
    dt_depth = (-(dt_bgc + 15.0) / 0.5).clip(lower=0.0, upper=40.0)
    dt_o2i = np.array([phase28._indoor_o2i_db(f, dm) for f, dm in zip(dt["freq_r"], dt_depth)])
    dt_water = dt["clutter_class"].astype(str) == "Water"
    dt_terr_eff = np.where(dt_water, 0.0, dt_terr)
    dt_o2i_eff = np.where(dt_water | (dt["obstruction_branch"].astype(str) != "indoor"), 0.0, dt_o2i)
    dt["physical_rsrp"] = (
        dt["dt_raw_unclipped"] + dt["per_re_db"] + dt["antenna_gain_delta_db"] - dt_terr_eff - dt_o2i_eff
    )
    dt["resid"] = pd.to_numeric(dt["rsrp_measured"], errors="coerce") - dt["physical_rsrp"]
    dt["clutter_class"] = dt["clutter_class"].astype("object").where(dt["clutter_class"].notna(), "UNKNOWN")
    split_key = pd.util.hash_pandas_object(dt["dt_row_id"].astype(str), index=False).astype("uint64")
    dt["split"] = np.where((split_key % 10) < 7, "train", "validation")
    tr = dt[dt.split == "train"]
    tr_fit = tr[tr["clutter_class"].astype(str) != "Water"]

    global_corr = float(tr_fit["resid"].median())
    ctbl = phase28._shrink_residual_table(
        tr_fit.assign(_r=tr_fit["resid"] - global_corr), ["clutter_class"], "_r", shrink_n=40.0
    )
    cmap = dict(zip(ctbl["clutter_class"].astype(str), ctbl["correction_db"]))

    def _resid_for(clut: pd.Series) -> np.ndarray:
        k = clut.astype(str)
        base = global_corr + k.map(lambda x: cmap.get(x, 0.0)).to_numpy(float)
        return np.where(k.to_numpy() == "Water", 0.0, base)

    c["residual_db"] = _resid_for(c["clutter_class"].astype("object").where(c["clutter_class"].notna(), "UNKNOWN"))
    c["phase31_rsrp_unclipped"] = c["physical_rsrp"] + c["residual_db"]
    c["phase31_rsrp"] = c["phase31_rsrp_unclipped"].where(c["phase31_rsrp_unclipped"] >= -140.0, np.nan)
    dt["residual_db"] = _resid_for(dt["clutter_class"])
    dt["phase31_rsrp"] = dt["physical_rsrp"] + dt["residual_db"]
    va = dt[dt.split == "validation"]

    # ---------------- serving grid (best + frontend) ----------------
    c["_env"] = np.where(branch == "indoor", "indoor", "outdoor")
    best = c.sort_values("phase31_rsrp_unclipped").groupby("grid_id").tail(1)
    frontend = c.groupby("grid_id").agg(
        phase31_physical_mean_rsrp=("physical_rsrp", "mean"),
        phase31_final_mean_rsrp=("phase31_rsrp", "mean"),
        antenna_gain_delta_db_mean=("antenna_gain_delta_db", "mean"),
    ).reset_index()
    serving = (
        bounds.merge(
            best[["grid_id", "physical_rsrp", "phase31_rsrp", "phase31_rsrp_unclipped",
                  "o2i_db", "terrain_db", "residual_db", "antenna_gain_delta_db", "_env"]],
            on="grid_id", how="left",
        )
        .merge(frontend, on="grid_id", how="left")
        .rename(columns={"_env": "serving_environment",
                         "physical_rsrp": "phase31_physical_best_rsrp",
                         "phase31_rsrp": "phase31_final_best_rsrp"})
    )
    # Phase 28 reference for the before/after
    p28 = pd.read_parquet(PHASE28_DIR / f"phase28_{tl}_serving_grid_project210.parquet")
    serving = serving.merge(
        p28[["grid_id", "phase28_final_best_rsrp", "phase28_final_mean_rsrp"]], on="grid_id", how="left"
    )
    serving.to_parquet(OUT_DIR / f"phase31_serving_grid_{tl}_project210.parquet", index=False)
    serving.to_csv(OUT_DIR / f"phase31_serving_grid_{tl}_project210.csv", index=False)
    c.to_parquet(OUT_DIR / f"phase31_scored_candidates_{tl}_project210.parquet", index=False)
    dt[["dt_row_id", "lat", "lon", "rsrp_measured", "obstruction_branch", "clutter_class", "split",
        "antenna_gain_delta_db", "physical_rsrp", "residual_db", "phase31_rsrp"]].to_parquet(
        OUT_DIR / f"phase31_dt_scored_{tl}_project210.parquet", index=False)

    p28dt = pd.read_parquet(PHASE28_DIR / f"phase28_{tl}_dt_scored_project210.parquet")
    p28_va = p28dt[(p28dt.split == "validation") & (p28dt.obstruction_branch.astype(str) != "indoor")]
    va_out = va[va.obstruction_branch.astype(str) != "indoor"]
    gd = pd.to_numeric(c["antenna_gain_delta_db"], errors="coerce")
    svb = pd.to_numeric(serving["phase31_final_best_rsrp"], errors="coerce")
    tech_sum = {
        "antenna_pattern": "CCVVPX308" if tech == "4G" else "Kathrein 800109221",
        "antenna_gain_delta_db": {"median": round(float(gd.median()), 2),
                                  "p10": round(float(gd.quantile(0.1)), 2),
                                  "p90": round(float(gd.quantile(0.9)), 2)},
        "held_out_outdoor_dt": {
            "phase28_generic_antenna": phase28._metrics(p28_va["rsrp_measured"], p28_va["phase28_final_rsrp"]),
            "phase31_real_antenna": phase28._metrics(va_out["rsrp_measured"], va_out["phase31_rsrp"]),
        },
        "held_out_dt_all_final": phase28._metrics(va["rsrp_measured"], va["phase31_rsrp"]),
        "residual_correction": {"global_db": round(global_corr, 2),
                                "per_clutter_db": {k: round(v, 2) for k, v in cmap.items()}},
        "serving_grid": {
            "rows": int(len(serving)),
            "no_coverage_rows": int(svb.isna().sum()),
            "median_rsrp": round(float(svb.median()), 1),
            "outdoor_median": round(float(pd.to_numeric(serving.loc[serving.serving_environment == "outdoor", "phase31_final_best_rsrp"], errors="coerce").median()), 1),
            "indoor_median": round(float(pd.to_numeric(serving.loc[serving.serving_environment == "indoor", "phase31_final_best_rsrp"], errors="coerce").median()), 1),
        },
    }
    m28 = tech_sum["held_out_outdoor_dt"]["phase28_generic_antenna"]
    m31 = tech_sum["held_out_outdoor_dt"]["phase31_real_antenna"]
    print(f"[PHASE31] {tech} antenna gain delta median {tech_sum['antenna_gain_delta_db']['median']} dB "
          f"({tech_sum['antenna_pattern']})")
    print(f"[PHASE31] {tech} held-out outdoor DT MAE: Phase 28 generic {m28['mae']:.2f} -> Phase 31 real {m31['mae']:.2f} "
          f"(bias {m28['bias']:.2f} -> {m31['bias']:.2f}, p90 {m28['p90_abs']:.1f} -> {m31['p90_abs']:.1f})")
    print(f"[PHASE31] {tech} serving: {tech_sum['serving_grid']['rows']} cells, "
          f"{tech_sum['serving_grid']['no_coverage_rows']} no-cov, median {tech_sum['serving_grid']['median_rsrp']}  "
          f"(outdoor {tech_sum['serving_grid']['outdoor_median']} / indoor {tech_sum['serving_grid']['indoor_median']})")
    return tech_sum


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand_all = pd.read_parquet(PHASE26_DIR / "phase26_scored_candidates_project210.parquet")
    dt_all = pd.read_parquet(PHASE26_DIR / "phase26_dt_scored_project210.parquet")
    grid = pd.read_parquet(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210.parquet")
    bounds = grid[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]]
    p28sum = json.loads((PHASE28_DIR / "phase28_summary.json").read_text(encoding="utf-8"))

    tech_summary = {}
    for tech in ("4G", "5G"):
        res = _run_tech(tech, cand_all, dt_all, bounds, p28sum)
        if res:
            tech_summary[tech] = res

    summary = {
        "scope": "4G and 5G. Base = Phase 28 pipeline (RSRP reference fix + water fix). "
                 "Real antenna: CCVVPX308 (4G), Kathrein 800109221 (5G). Null-spike smoothed + capped.",
        "antenna_delta_clip_db": list(ANTENNA_DELTA_CLIP_DB),
        "technology": tech_summary,
        # back-compat: existing dashboard reads these top-level keys (== 4G)
        "antenna_gain_delta_db": tech_summary.get("4G", {}).get("antenna_gain_delta_db", {}),
        "held_out_outdoor_dt": tech_summary.get("4G", {}).get("held_out_outdoor_dt", {}),
        "residual_correction": tech_summary.get("4G", {}).get("residual_correction", {}),
        "serving_grid": tech_summary.get("4G", {}).get("serving_grid", {}),
    }
    (OUT_DIR / "phase31_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[PHASE31] wrote 4G + 5G outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
