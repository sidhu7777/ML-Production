"""Phase 36 (FINAL): per-technology physical + real antenna + full Phase 25 hierarchical calibration.

4G physical : COST-231 (production raw) + RSRP-per-RE reference fix + real CommScope
              CCVVPX308 pattern delta (null-smoothed via Phase 31, clipped +/-15/12 dB).
5G physical : SAME COST-231 core as 4G (production raw is computed at 2600 MHz) + the
              -2.58 dB 2600->3300 MHz n78 term + real Kathrein 800109221 pattern delta
              (all electrical tilts). COST-231 is used for 5G on purpose: 38.901 UMa's
              22 dB/decade LOS slope decays too slowly, so the calibrated 5G surface read
              STRONGER than 4G near the sites; sharing the COST-231 slope puts 5G correctly
              below 4G by the frequency delta. (Phases 33/34/35 keep the 38.901 experiment.)

Both techs : terrain diffraction is ALWAYS kept (Water included - only the indoor O2I term
             is skipped for Water, never terrain). Indoor O2I applied once. Then the full
             Phase 25 hierarchical dynamic calibration - tech_band -> clutter_terrain ->
             sector -> local IDW residual field, shrinkage-regularised, fit on 70% of the
             OUTDOOR DT grids and validated on the held-out 30%.

Serving-cell hygiene: DT points more than 135 deg off the assigned sector boresight (wrong
             serving assignment / deep backlobe) are flagged and excluded from the fit and
             from the headline metric.

Reuses, unmodified: Phase 25 calibration machinery (base aliased to
phase24_physical_with_terrain_rsrp), Phase 28 reference offset + O2I, Phase 29 antenna
patterns, Phase 31 null-smoothing, Phase 33 UMa constants + DT geometry, Phase 35 all-tilt
Kathrein selector. No earlier phase or production file is modified.
Output: data/project_210_taiwan/cost231_phase36_final/
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
for path in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import test_project210_phase22_terrain_diffraction_comparison as phase22
import test_project210_phase25_hierarchical_dynamic_calibration as phase25
import test_project210_phase28_4g_rsrp_reference_fix as phase28
import test_project210_phase29_real_antenna_pattern as phase29
import test_project210_phase31_phase28_real_antenna as phase31  # noqa: F401  (patches phase29._pat_gain -> smoothed)
import test_project210_phase33_5g_38_901 as phase33  # _geometry_for_dt + _bearing_deg
from phase_rsrp_guard import valid_model_rsrp

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE26_DIR = PROJECT_DIR / "cost231_phase26_corrected_obstruction_profile"
PHASE27_DIR = PROJECT_DIR / "cost231_phase27_dynamic_on_corrected_obstruction"
PHASE31_DIR = PROJECT_DIR / "cost231_phase31_phase28_real_antenna"
OUT_DIR = PROJECT_DIR / "cost231_phase36_final"
IMAGE_DIR = OUT_DIR / "images"

UE_H_M = 1.5
ANTENNA_DELTA_CLIP_DB = (-15.0, 12.0)   # real vendor pattern gain minus generic 3GPP 18/65/6
BACKLOBE_DROP_DEG = 135.0
# 5G n78 is modelled on the SAME COST-231 core as 4G (production raw is computed at
# 2600 MHz); this is the 2600 -> 3300 MHz frequency term. Using COST-231 for 5G keeps
# the propagation SLOPE identical to 4G, so the calibrated 5G surface sits below 4G by
# the frequency delta instead of above it (the 38.901 UMa 22 dB/decade LOS slope made
# 5G decay too slowly and read stronger than 4G near the sites).
N78_OFFSET_DB = -2.58

# Phase 25's machinery keys off this exact column name.
BASE_COL = "phase24_physical_with_terrain_rsrp"
BASE_UNCLIPPED = "phase24_physical_with_terrain_rsrp_unclipped"


# --------------------------------------------------------------------------- physical layer
def _raw_cost231(df: pd.DataFrame, tech: str, raw: np.ndarray) -> np.ndarray:
    """COST-231 production raw (unclipped) + RSRP reference term + real-antenna gain delta
    (real vendor pattern - generic 3GPP). Both techs share the COST-231 slope."""
    raw = np.asarray(raw, dtype=float)
    freq_r = pd.to_numeric(df["frequency_mhz"], errors="coerce").round(1)
    if tech == "4G":
        raw = raw + freq_r.map(phase28._per_re_reference_offset_db).to_numpy(float)
    else:
        raw = raw + N78_OFFSET_DB
    delta = np.clip(phase29._antenna_gain_delta(df.assign(technology=tech)), *ANTENNA_DELTA_CLIP_DB)
    return raw + np.asarray(delta, dtype=float)


def _dt_cost231_raw(dt: pd.DataFrame) -> np.ndarray:
    """Per-DT COST-231 raw (unclipped). Joined from the candidate surface by
    (nearest grid, assigned/re-assigned serving cell) so it is consistent with the
    serving grid and correct after serving-cell re-assignment; falls back to the stored
    4G column / the Phase 28 reconstruction where there is no candidate match."""
    cand = pd.read_parquet(PHASE26_DIR / "phase26_scored_candidates_project210.parquet")
    key = (cand.groupby(["grid_id", "strict_cell_key"], dropna=False)["raw_cost231_rsrp_unclipped"]
           .first().reset_index())
    key["grid_id"] = key["grid_id"].astype(str)
    key["strict_cell_key"] = key["strict_cell_key"].astype(str)
    m = dt[["nearest_grid_id", "assigned_strict_cell_key"]].copy()
    m["nearest_grid_id"] = m["nearest_grid_id"].astype(str)
    m["assigned_strict_cell_key"] = m["assigned_strict_cell_key"].astype(str)
    joined = m.merge(key, left_on=["nearest_grid_id", "assigned_strict_cell_key"],
                     right_on=["grid_id", "strict_cell_key"], how="left")["raw_cost231_rsrp_unclipped"]
    fallback = phase28._dt_raw_unclipped(dt).to_numpy(float)
    return pd.to_numeric(joined, errors="coerce").fillna(pd.Series(fallback)).to_numpy(float)


def _build_physical(df: pd.DataFrame, tech: str, raw: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["technology"] = tech
    raw = _raw_cost231(out, tech, raw)

    branch = out["obstruction_branch"].astype(str).to_numpy()
    water = out["clutter_class"].astype(str).to_numpy() == "Water"
    terr = pd.to_numeric(out["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()

    o2i_freq = 3300.0 if tech == "5G" else None
    bgc = pd.to_numeric(out["building_geo_correction_db"], errors="coerce")
    depth = (-(bgc + 15.0) / 0.5).clip(lower=0.0, upper=40.0).to_numpy()
    freq_r = pd.to_numeric(out["frequency_mhz"], errors="coerce").round(1).to_numpy()
    o2i = np.array([
        phase28._indoor_o2i_db(o2i_freq if o2i_freq is not None else f, dm)
        for f, dm in zip(freq_r, depth)
    ])
    o2i = np.where((branch == "indoor") & ~water, o2i, 0.0)

    out["phase36_raw_rsrp"] = raw
    out["phase36_terrain_db"] = terr           # kept for ALL clutter, Water included
    out["phase36_o2i_db"] = o2i
    out[BASE_UNCLIPPED] = raw - terr - o2i
    out[BASE_COL] = valid_model_rsrp(out[BASE_UNCLIPPED])
    out["phase24_phase19_style_bias_db"] = 0.0
    return out


# --------------------------------------------------------------------------- serving-cell hygiene
def _sector_table() -> pd.DataFrame:
    c = pd.read_parquet(PHASE26_DIR / "phase26_scored_candidates_project210.parquet")
    t = (
        c.groupby(["technology", "strict_cell_key"], dropna=False)
        .agg(site_lat=("site_lat", "first"), site_lon=("site_lon", "first"),
             azimuth=("azimuth_x", "first"), Etilt=("Etilt", "first"),
             Mtilt=("Mtilt", "first"), Height=("Height", "first"))
        .reset_index()
    )
    t["site_key"] = (pd.to_numeric(t["site_lat"], errors="coerce").round(5).astype(str)
                     + "_" + pd.to_numeric(t["site_lon"], errors="coerce").round(5).astype(str))
    return t


def _reassign_dt_serving(dt: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
    """The logged 5G PCI->cell map puts ~27% of DT points behind their assigned sector
    (median 84 deg off boresight vs 32 deg for the model's own best server). For each DT
    point, if another sector OF THE SAME SITE is far better aligned to the site->point
    bearing, re-assign it (azimuth, tilt, height, cell key) so DT is scored against the
    cell the model would actually pick. Points with no well-aligned same-site sector are
    left as-is and flagged as backlobe."""
    out = dt.copy()
    out["site_key"] = (pd.to_numeric(out["site_lat"], errors="coerce").round(5).astype(str)
                       + "_" + pd.to_numeric(out["site_lon"], errors="coerce").round(5).astype(str))
    slat = pd.to_numeric(out["site_lat"], errors="coerce").to_numpy(float)
    slon = pd.to_numeric(out["site_lon"], errors="coerce").to_numpy(float)
    rlat = pd.to_numeric(out["lat"], errors="coerce").to_numpy(float)
    rlon = pd.to_numeric(out["lon"], errors="coerce").to_numpy(float)
    bearing = np.asarray(phase29._bearing_deg(slat, slon, rlat, rlon), dtype=float)
    tech_arr = out["assigned_technology"].astype(str).to_numpy()
    sk_arr = out["site_key"].to_numpy()

    by_site = {key: g.reset_index(drop=True) for key, g in sectors.groupby(["technology", "site_key"])}
    az = pd.to_numeric(out["azimuth"], errors="coerce").to_numpy(float).copy()
    et = pd.to_numeric(out["Etilt"], errors="coerce").to_numpy(float).copy()
    mt = pd.to_numeric(out["Mtilt"], errors="coerce").to_numpy(float).copy()
    ht = pd.to_numeric(out["Height"], errors="coerce").to_numpy(float).copy()
    key = out["assigned_strict_cell_key"].astype(str).to_numpy().copy()
    reassigned = np.zeros(len(out), dtype=bool)

    for i in range(len(out)):
        g = by_site.get((tech_arr[i], sk_arr[i]))
        if g is None or len(g) < 2:
            continue
        cand_az = pd.to_numeric(g["azimuth"], errors="coerce").to_numpy(float)
        offs = np.abs((bearing[i] - cand_az + 180.0) % 360.0 - 180.0)
        j = int(np.argmin(offs))
        cur = abs((bearing[i] - az[i] + 180.0) % 360.0 - 180.0)
        if cur > 90.0 and offs[j] + 25.0 < cur:
            row = g.iloc[j]
            az[i], et[i], mt[i], ht[i] = (float(row["azimuth"]), float(row["Etilt"]),
                                          float(row["Mtilt"]), float(row["Height"]))
            key[i] = str(row["strict_cell_key"])
            reassigned[i] = True

    out["azimuth"], out["Etilt"], out["Mtilt"], out["Height"] = az, et, mt, ht
    out["assigned_strict_cell_key"] = key
    out["p36_reassigned"] = reassigned
    return out


# --------------------------------------------------------------------------- inputs
def _candidate_inputs() -> pd.DataFrame:
    cand = pd.read_parquet(PHASE26_DIR / "phase26_scored_candidates_project210.parquet")
    frames = []
    for tech in ("4G", "5G"):
        sub = cand[cand["technology"].astype(str) == tech].copy()
        if sub.empty:
            continue
        raw = pd.to_numeric(sub["raw_cost231_rsrp_unclipped"], errors="coerce").to_numpy(float)
        frames.append(_build_physical(sub, tech, raw))
    out = pd.concat(frames, ignore_index=True)
    out = phase25._add_common_features(out, "strict_cell_key")
    out["phase24_no_lock_reference_rsrp_unclipped"] = pd.to_numeric(out[BASE_UNCLIPPED], errors="coerce")
    out["phase24_no_lock_reference_rsrp"] = valid_model_rsrp(out["phase24_no_lock_reference_rsrp_unclipped"])
    return out


def _dt_inputs() -> pd.DataFrame:
    dt = pd.read_parquet(PHASE26_DIR / "phase26_dt_scored_project210.parquet")
    dt = _reassign_dt_serving(dt, _sector_table())   # serving-cell hygiene BEFORE geometry
    dt = phase33._geometry_for_dt(dt)                 # adds distance_m + azimuth_delta_deg
    frames = []
    for tech in ("4G", "5G"):
        sub = dt[dt["assigned_technology"].astype(str) == tech].copy()
        if sub.empty:
            continue
        raw = _dt_cost231_raw(sub)   # candidate-joined, correct after re-assignment
        frames.append(_build_physical(sub, tech, raw))
    out = pd.concat(frames, ignore_index=True)
    out["technology"] = out["assigned_technology"].astype(str)
    out = phase25._add_common_features(out, "assigned_strict_cell_key")
    out["phase24_no_lock_reference_rsrp_unclipped"] = pd.to_numeric(out[BASE_UNCLIPPED], errors="coerce")
    out["phase24_no_lock_reference_rsrp"] = valid_model_rsrp(out["phase24_no_lock_reference_rsrp_unclipped"])
    out["p36_backlobe"] = pd.to_numeric(out["azimuth_delta_deg"], errors="coerce").fillna(0.0) > BACKLOBE_DROP_DEG
    return phase25._split_dt_by_grid(out)


# --------------------------------------------------------------------------- calibration
#  Phase 25's proven hierarchy (tech_band -> clutter_terrain -> sector -> local IDW field),
#  fit on 70% of the outdoor DT grids. Same recipe as Phase 27. Indoor cells (no DT) get
#  only the group model-bias terms (tech_band + clutter_terrain), never the location-
#  specific sector/local field.
def _fit(fit_df: pd.DataFrame):
    train_scored, layers = phase25._fit_group_hierarchy(fit_df)
    local_models = phase25._fit_local_models(train_scored)
    return layers, local_models


def _score(df: pd.DataFrame, layers, local_models) -> pd.DataFrame:
    s = phase25._finalize(phase25._apply_local_model(phase25._apply_group_hierarchy(df, layers), local_models))
    is_indoor = (s["obstruction_branch"].astype(str) == "indoor").to_numpy()
    base_unclip = pd.to_numeric(s[BASE_UNCLIPPED], errors="coerce")
    full = pd.to_numeric(s["phase25_total_dynamic_correction_db"], errors="coerce").fillna(0.0).to_numpy()
    group_bias = (pd.to_numeric(s.get("tech_band_correction_db"), errors="coerce").fillna(0.0)
                  + pd.to_numeric(s.get("clutter_terrain_correction_db"), errors="coerce").fillna(0.0)).to_numpy()
    corr = np.where(is_indoor, group_bias, full)
    s["phase36_total_correction_db"] = corr
    s["phase36_local_corr_db"] = np.where(is_indoor, 0.0, pd.to_numeric(s["local_residual_correction_db"], errors="coerce").fillna(0.0))
    s["phase36_shape_adj_db"] = corr - s["phase36_local_corr_db"].to_numpy()
    s["phase36_final_rsrp_unclipped"] = base_unclip + corr
    s["phase36_final_rsrp"] = valid_model_rsrp(s["phase36_final_rsrp_unclipped"])
    s["phase36_confidence"] = pd.to_numeric(s.get("phase25_confidence"), errors="coerce").fillna(0.3)
    return s


def _aggregate(cand_scored: pd.DataFrame) -> pd.DataFrame:
    work = cand_scored.copy()
    work["_f"] = pd.to_numeric(work["phase36_final_rsrp_unclipped"], errors="coerce")
    best = work.sort_values("_f").groupby(["technology", "grid_id"], dropna=False).tail(1)
    best_env = best[["technology", "grid_id", "obstruction_branch"]].copy()
    best_env["serving_environment"] = np.where(
        best_env["obstruction_branch"].astype(str) == "indoor", "indoor", "outdoor"
    )
    agg = (
        work.groupby(["technology", "grid_id"], dropna=False)
        .agg(
            phase36_physical_best_rsrp=(BASE_COL, "max"),
            phase36_physical_mean_rsrp=(BASE_COL, "mean"),
            phase36_final_best_rsrp=("phase36_final_rsrp", "max"),
            phase36_final_mean_rsrp=("phase36_final_rsrp", "mean"),
            phase36_total_correction_db_mean=("phase36_total_correction_db", "mean"),
            phase36_shape_adj_db_mean=("phase36_shape_adj_db", "mean"),
            phase36_local_corr_db_mean=("phase36_local_corr_db", "mean"),
            phase36_confidence_mean=("phase36_confidence", "mean"),
        )
        .reset_index()
        .merge(best_env[["technology", "grid_id", "serving_environment"]], on=["technology", "grid_id"], how="left")
    )
    grid = pd.read_parquet(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210.parquet")
    bounds = grid[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]]
    full = pd.concat([bounds[["grid_id"]].assign(technology=t) for t in ("4G", "5G")], ignore_index=True)
    return full.merge(agg, on=["technology", "grid_id"], how="left").merge(bounds, on="grid_id", how="left")


def _cdf(items, title, path):
    phase22._plot_cdf(items, title, path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    dt = _dt_inputs()
    cand = _candidate_inputs()

    # 5G reference-level anchor. The production 5G raw is total-carrier power (tx_power is a
    # ~50 dBm total-power value), so - exactly like the 4G RSRP-per-RE bug - it needs a
    # wideband->per-RE reference term to become SS-RSRP. It is data-anchored here on ALL
    # outdoor training DT (not clear-only: clear DT sits close-in where the 5G raw happens
    # to match, which gave the wrong sign and left the mid/far surface ~7 dB hot).
    tr5 = dt[(dt["phase25_split"] == "train") & (dt["technology"] == "5G")
             & (dt["obstruction_branch"].astype(str) != "indoor")
             & (dt["clutter_class"].astype(str) != "Water")]
    off5 = float((pd.to_numeric(tr5["rsrp_measured"], errors="coerce")
                  - pd.to_numeric(tr5[BASE_COL], errors="coerce")).median())
    off5 = 0.0 if not np.isfinite(off5) else off5
    for frame in (dt, cand):
        m5 = (frame["technology"].astype(str) == "5G").to_numpy()
        frame.loc[m5, BASE_UNCLIPPED] = pd.to_numeric(frame.loc[m5, BASE_UNCLIPPED], errors="coerce") + off5
        frame.loc[m5, BASE_COL] = valid_model_rsrp(frame.loc[m5, BASE_UNCLIPPED])
        frame.loc[m5, "phase24_no_lock_reference_rsrp_unclipped"] = pd.to_numeric(
            frame.loc[m5, "phase24_no_lock_reference_rsrp_unclipped"], errors="coerce") + off5
        frame.loc[m5, "phase24_no_lock_reference_rsrp"] = valid_model_rsrp(
            frame.loc[m5, "phase24_no_lock_reference_rsrp_unclipped"])
    print(f"[PHASE36] 5G physical level anchor: {off5:+.1f} dB (all-outdoor training DT)")

    train = dt[dt["phase25_split"] == "train"].copy()
    valid = dt[dt["phase25_split"] == "validation"].copy()

    fit = train[(train["obstruction_branch"].astype(str) != "indoor") & (~train["p36_backlobe"])].copy()
    print(f"[PHASE36] dt train={len(train)} valid={len(valid)} fit(outdoor,no-backlobe)={len(fit)} "
          f"reassigned_serving={int(dt['p36_reassigned'].sum())} "
          f"still_backlobe_dropped={int(train['p36_backlobe'].sum())} candidates={len(cand)}")

    layers, local_models = _fit(fit)
    if layers:
        pd.concat(layers, ignore_index=True).to_csv(OUT_DIR / "phase36_group_corrections.csv", index=False)
        tb = pd.concat(layers, ignore_index=True)
        tb = tb[tb["layer"] == "tech_band"]
        print("[PHASE36] tech_band correction:",
              {f"{r.technology}/{r.band}": round(r.tech_band_correction_db, 1) for r in tb.itertuples()})

    valid_scored = _score(valid, layers, local_models)
    cand_scored = _score(cand, layers, local_models)
    serving_all = _aggregate(cand_scored)

    phase22._save_frame(valid_scored, OUT_DIR / "phase36_validation_dt_project210")
    phase22._save_frame(cand_scored, OUT_DIR / "phase36_scored_candidates_project210")

    # comparison references
    try:
        p27 = json.loads((PHASE27_DIR / "phase27_summary.json").read_text(encoding="utf-8"))["technology"]
    except Exception:
        p27 = {}
    try:
        p31 = json.loads((PHASE31_DIR / "phase31_summary.json").read_text(encoding="utf-8"))["technology"]
    except Exception:
        p31 = {}

    summary = {
        "scope": "FINAL. 4G = COST-231 + RSRP-per-RE reference fix + real CCVVPX308 delta. "
                 "5G = SAME COST-231 core (production raw @2600) + (-2.58 dB n78 2600->3300) + real Kathrein "
                 "delta + all-outdoor-DT level anchor. Both + Phase 25 hierarchical calibration "
                 "(tech_band -> clutter_terrain -> sector -> local IDW). Terrain kept for Water. "
                 "DT serving-cell re-assigned to best-aligned same-site sector; deep-backlobe dropped from fit.",
        "known_limitation": "4G and 5G measured DT are equal (+0.0 dB on 8033 co-located points), so the "
                 "calibrated map shows them similar; between the DT roads the 4G polygon reads ~5 dB below 5G "
                 "because the 4G 775 MHz COST-231-Hata raw (extrapolated below the model's 1500 MHz floor) is "
                 "too weak near the sites and the per-band tech_band constant cannot fix a distance-shaped error. "
                 "A true fix needs Okumura-Hata (150-1500 MHz) for the 775 MHz band in the production raw / Phase 9.",
        "params": {
            "n78_offset_db": N78_OFFSET_DB,
            "antenna_delta_clip_db": list(ANTENNA_DELTA_CLIP_DB),
            "backlobe_drop_deg": BACKLOBE_DROP_DEG,
            "g5_level_anchor_db": round(off5, 2),
            "dt_serving_reassigned": int(dt["p36_reassigned"].sum()),
        },
        "technology": {},
    }

    for tech in ("4G", "5G"):
        serv = serving_all[serving_all["technology"].astype(str) == tech].copy()
        vt = valid_scored[valid_scored["technology"].astype(str) == tech].copy()
        vt_out = vt[(vt["obstruction_branch"].astype(str) != "indoor")]
        vt_out_clean = vt_out[~vt_out["p36_backlobe"]]
        phase22._save_frame(serv, OUT_DIR / f"phase36_serving_grid_{tech.lower()}_project210")

        _cdf(
            [
                ("1 - DT measured (outdoor)", vt_out_clean["rsrp_measured"], "#111827"),
                ("2 - Phase 36 predicted at DT", vt_out_clean["phase36_final_rsrp"], "#2563eb"),
                ("3 - Phase 36 predicted, outdoor polygon",
                 serv.loc[serv["serving_environment"] == "outdoor", "phase36_final_best_rsrp"], "#16a34a"),
                ("4 - Phase 36 predicted, indoor polygon",
                 serv.loc[serv["serving_environment"] == "indoor", "phase36_final_best_rsrp"], "#f59e0b"),
            ],
            f"Project 210 {tech}: Phase 36 - DT accuracy vs whole-polygon prediction",
            IMAGE_DIR / f"phase36_{tech.lower()}_dt_vs_polygon_cdf.png",
        )
        _cdf(
            [
                ("Phase 36 physical abs err", (vt_out_clean["rsrp_measured"] - vt_out_clean[BASE_COL]).abs(), "#6b7280"),
                ("Phase 36 final abs err", (vt_out_clean["rsrp_measured"] - vt_out_clean["phase36_final_rsrp"]).abs(), "#16a34a"),
            ],
            f"Project 210 {tech}: Phase 36 held-out DT absolute error",
            IMAGE_DIR / f"phase36_{tech.lower()}_abs_error_cdf.png",
        )

        m_phys = phase25._metrics(vt_out_clean["rsrp_measured"], vt_out_clean[BASE_COL])
        m_final = phase25._metrics(vt_out_clean["rsrp_measured"], vt_out_clean["phase36_final_rsrp"])
        m_final_incl_backlobe = phase25._metrics(vt_out["rsrp_measured"], vt_out["phase36_final_rsrp"])
        nw = vt_out_clean[vt_out_clean["clutter_class"].astype(str) != "Water"]
        sv = pd.to_numeric(serv["phase36_final_best_rsrp"], errors="coerce")
        env = serv["serving_environment"]
        summary["technology"][tech] = {
            "held_out_outdoor_physical_no_calibration": m_phys,
            "held_out_outdoor_final": m_final,
            "held_out_outdoor_final_incl_backlobe_ref": m_final_incl_backlobe,
            "held_out_outdoor_final_non_water": phase25._metrics(nw["rsrp_measured"], nw["phase36_final_rsrp"]),
            "reference_phase27_dynamic": (p27.get(tech, {}) or {}).get("held_out_outdoor_phase27_dynamic", {}),
            "reference_phase31_real_antenna": ((p31.get(tech, {}) or {}).get("held_out_outdoor_dt", {}) or {}).get("phase31_real_antenna", {}),
            "serving_grid": {
                "rows": int(len(serv)),
                "no_coverage": int(sv.isna().sum()),
                "median_rsrp": round(float(sv.median()), 1),
                "outdoor_median": round(float(pd.to_numeric(serv.loc[env == "outdoor", "phase36_final_best_rsrp"], errors="coerce").median()), 1),
                "indoor_median": round(float(pd.to_numeric(serv.loc[env == "indoor", "phase36_final_best_rsrp"], errors="coerce").median()), 1),
            },
            "validation_dt_rows": int(len(vt)),
            "backlobe_dropped_from_metric": int(vt_out["p36_backlobe"].sum()),
        }
        p = summary["technology"][tech]
        print(f"[PHASE36] {tech}  physical MAE {m_phys['mae']:.2f} (bias {m_phys['bias']:+.1f}) -> "
              f"final MAE {m_final['mae']:.2f} clean / {m_final_incl_backlobe['mae']:.2f} incl-backlobe "
              f"(bias {m_final['bias']:+.2f}, p90 {m_final['p90_abs']:.1f})")
        print(f"[PHASE36] {tech}  serving median {p['serving_grid']['median_rsrp']}  "
              f"(outdoor {p['serving_grid']['outdoor_median']} / indoor {p['serving_grid']['indoor_median']})  "
              f"vs Phase27 {p['reference_phase27_dynamic'].get('mae', float('nan')):.2f} / Phase31 {p['reference_phase31_real_antenna'].get('mae', float('nan')):.2f}")

    (OUT_DIR / "phase36_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[PHASE36] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
