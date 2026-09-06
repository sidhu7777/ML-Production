"""
Phase 29: real measured per-electrical-tilt antenna patterns, replacing the generic
3GPP parametric antenna (max_gain 18 dBi, 65 deg H, 6 deg V) used up to Phase 26.

  4G 775.5 MHz -> CommScope CCVVPX308, "698 - 806 MHz" band, T0..T10, Port 1 (+45)
  4G 1840  MHz -> CommScope CCVVPX308, "1710 - 1880 MHz" band, T0..T10, Port 5 (+45)
  5G 3300  MHz -> Kathrein 800109221, "3300 - 3590 MHz" band, eTilt 2..12, Y1P45 Port1

The .pap pattern files are extracted from ML/Research/5G Antennas.rar into
data/project_210_taiwan/antenna_patterns/ .

Base pipeline = Phase 27 logic (Phase 26 corrected-obstruction physical + Phase 25
hierarchical dynamic calibration on outdoor DT). Phase 28 is NOT used.

Method (no Phase 9 / Phase 26 recompute): per candidate compute
    gain_delta = real_pattern_gain(band, cell tilt, az-offset, depression)
               - generic_3gpp_gain(18 / 65 / 6, same geometry)
add gain_delta to the Phase 26 physical value, then run the Phase 25 dynamic
calibration on the adjusted physical. Held-out DT validation and a before/after
comparison against Phase 27.

Nothing at or below Phase 28 is modified.
"""
from __future__ import annotations

import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
for p in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_project210_phase22_terrain_diffraction_comparison as phase22
import test_project210_phase25_hierarchical_dynamic_calibration as phase25
import test_project210_phase27_dynamic_on_corrected_obstruction as phase27
from phase_rsrp_guard import valid_model_rsrp

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE26_DIR = PROJECT_DIR / "cost231_phase26_corrected_obstruction_profile"
PAT_DIR = PROJECT_DIR / "antenna_patterns"
OUT_DIR = PROJECT_DIR / "cost231_phase29_real_antenna_pattern"
IMAGE_DIR = OUT_DIR / "images"

BASE_COL = phase27.BASE_COL                     # "phase24_physical_with_terrain_rsrp"
BASE_UNCLIP = "phase26_physical_with_terrain_rsrp_unclipped"
BASE_BIAS_COL = phase27.BASE_BIAS_COL
INDOOR_BRANCH = phase27.INDOOR_BRANCH

UE_HEIGHT_M = 1.5
TILT_SCALE = 10.0                               # raw DB Etilt/Mtilt are tenths of a degree

# generic 3GPP parametric antenna baked into the Phase 9/26 raw
GEN_MAX_GAIN = 18.0
GEN_H_BW = 65.0
GEN_V_BW = 6.0
GEN_A_MAX = 30.0
GEN_SLA_V = 20.0

# datasheet boresight gain per band (from the .paf headers)
BORESIGHT_GAIN_DBI = {
    "CCVVPX308_698": 14.5,
    "CCVVPX308_1710": 16.6,
    "K800109221_3300": 17.4,
}


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _bearing_deg(lat1, lon1, lat2, lon2):
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


# --------------------------------------------------------------------------- .pap loader
def _parse_pap(path: Path) -> tuple[int, np.ndarray, int, np.ndarray]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    hm = re.search(
        r"<HorizontalPatterns>.*?<StartAngle>(-?\d+)</StartAngle>.*?<Step>(\d+)</Step>.*?<Gains>([^<]+)</Gains>",
        txt, re.S,
    )
    vm = re.search(
        r"<VerticalPatterns>.*?<StartAngle>(-?\d+)</StartAngle>.*?<Step>(\d+)</Step>.*?<Gains>([^<]+)</Gains>",
        txt, re.S,
    )
    h = np.array([float(x) for x in hm.group(3).split(";")], dtype=float)
    v = np.array([float(x) for x in vm.group(3).split(";")], dtype=float)
    return int(hm.group(1)), h, int(vm.group(1)), v


def _pattern_file(tech: str, freq_mhz: float, etilt_deg: float) -> Path:
    if tech == "5G":
        et = int(min(12, max(2, round(etilt_deg))))
        return PAT_DIR / "K800109221" / f"3300 - 3590 MHz, eTilt {et}, Y1P45 - Port1.pap"
    et = int(min(10, max(0, round(etilt_deg))))
    if round(float(freq_mhz), 1) <= 1000.0:
        return PAT_DIR / "CCVVPX308" / f"698 - 806 MHz, T {et}, eAz 0, eBw 0, Port 1 +45.pap"
    return PAT_DIR / "CCVVPX308" / f"1710 - 1880 MHz, T {et}, eAz 0, eBw 0, Port 5 +45.pap"


def _boresight_key(tech: str, freq_mhz: float) -> str:
    if tech == "5G":
        return "K800109221_3300"
    return "CCVVPX308_698" if round(float(freq_mhz), 1) <= 1000.0 else "CCVVPX308_1710"


@lru_cache(maxsize=64)
def _pap_cached(path_str: str) -> tuple[int, tuple, int, tuple]:
    hs, h, vs, v = _parse_pap(Path(path_str))
    return hs, tuple(h), vs, tuple(v)


def _pat_gain(start: int, arr: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    idx = (np.round(angle_deg).astype(int) - start) % len(arr)
    return arr[idx]


# ----------------------------------------------------------------- antenna gain delta
def _generic_3gpp_gain(az_off: np.ndarray, elev_diff: np.ndarray) -> np.ndarray:
    ah = np.minimum(12.0 * (az_off / GEN_H_BW) ** 2, GEN_A_MAX)
    av = np.minimum(12.0 * (elev_diff / GEN_V_BW) ** 2, GEN_SLA_V)
    return GEN_MAX_GAIN - np.minimum(ah + av, GEN_A_MAX)


def _antenna_gain_delta(df: pd.DataFrame) -> np.ndarray:
    tech = df["technology"].astype(str).to_numpy()
    freq = pd.to_numeric(df["frequency_mhz"], errors="coerce").to_numpy()
    dist = np.maximum(pd.to_numeric(df["distance_m"], errors="coerce").to_numpy(), 1.0)
    htx = pd.to_numeric(df["Height"], errors="coerce").fillna(20.0).to_numpy()
    etilt = pd.to_numeric(df["Etilt"], errors="coerce").fillna(30.0).to_numpy() / TILT_SCALE
    mtilt = pd.to_numeric(df["Mtilt"], errors="coerce").fillna(0.0).to_numpy() / TILT_SCALE
    az_off = np.abs(pd.to_numeric(df["azimuth_delta_deg"], errors="coerce").fillna(0.0).to_numpy())

    elev_angle = np.degrees(np.arctan2(UE_HEIGHT_M - htx, dist))   # negative: point below antenna
    elev_diff = elev_angle + etilt + mtilt                          # 3GPP convention
    generic = _generic_3gpp_gain(az_off, elev_diff)

    depression = -elev_angle                                        # positive downward
    v_lookup = depression + mtilt                                   # electrical tilt is inside the .pap file

    real = np.full(len(df), np.nan, dtype=float)
    for (t, is_low), grp_idx in _group_by_config(tech, freq):
        rows = grp_idx
        # one .pap per (tech, band, rounded etilt) - iterate distinct tilt files
        et_round = np.clip(np.round(etilt[rows]).astype(int), 2 if t == "5G" else 0, 12 if t == "5G" else 10)
        bkey = _boresight_key(t, 900.0 if is_low else 1840.0 if t == "4G" else 3300.0)
        g0 = BORESIGHT_GAIN_DBI[bkey]
        for et_val in np.unique(et_round):
            sel = rows[et_round == et_val]
            path = _pattern_file(t, 900.0 if is_low else (1840.0 if t == "4G" else 3300.0), et_val)
            hs, h, vs, v = _pap_cached(str(path))
            h = np.asarray(h); v = np.asarray(v)
            hgain = _pat_gain(hs, h, az_off[sel])
            vgain = _pat_gain(vs, v, v_lookup[sel])
            real[sel] = g0 + hgain + vgain
    real = np.where(np.isfinite(real), real, generic)
    # floor: never below (boresight - 40) to avoid pattern-null spikes on the delta
    return real - generic


def _group_by_config(tech: np.ndarray, freq: np.ndarray):
    out = []
    for t in np.unique(tech):
        if t == "5G":
            idx = np.where(tech == t)[0]
            out.append(((t, False), idx))
            continue
        for is_low, mask in ((True, np.round(freq, 1) <= 1000.0), (False, np.round(freq, 1) > 1000.0)):
            idx = np.where((tech == t) & mask)[0]
            if idx.size:
                out.append(((t, is_low), idx))
    return out


# ------------------------------------------------------------------- inputs (Phase 27 + delta)
def _apply_delta_and_prep(df: pd.DataFrame, cell_col: str) -> pd.DataFrame:
    out = df.copy()
    delta = _antenna_gain_delta(out)
    out["phase29_antenna_gain_delta_db"] = delta
    for col in (BASE_COL, BASE_UNCLIP):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") + delta
    out["technology"] = out["technology"].astype(str)
    out = phase25._add_common_features(out, cell_col)
    out[BASE_BIAS_COL] = pd.to_numeric(out.get(BASE_BIAS_COL), errors="coerce").fillna(0.0)
    out["phase24_no_lock_reference_rsrp_unclipped"] = (
        pd.to_numeric(out[BASE_COL], errors="coerce") + out[BASE_BIAS_COL]
    )
    out["phase24_no_lock_reference_rsrp"] = valid_model_rsrp(out["phase24_no_lock_reference_rsrp_unclipped"])
    return out


def _candidate_inputs() -> pd.DataFrame:
    cand = phase22._read_frame(PHASE26_DIR / "phase26_scored_candidates_project210")
    cand = cand.rename(columns={"phase26_physical_with_terrain_rsrp": BASE_COL,
                                "phase26_phase19_bias_db": BASE_BIAS_COL})
    return _apply_delta_and_prep(cand, "strict_cell_key")


def _dt_inputs() -> pd.DataFrame:
    dt = phase22._read_frame(PHASE26_DIR / "phase26_dt_scored_project210")
    dt = dt.rename(columns={"phase26_physical_with_terrain_rsrp": BASE_COL})
    dt["technology"] = dt["assigned_technology"].astype(str)
    # geometry the delta needs: distance + azimuth_delta from site+point
    site_lat = pd.to_numeric(dt["site_lat"], errors="coerce"); site_lon = pd.to_numeric(dt["site_lon"], errors="coerce")
    rx_lat = pd.to_numeric(dt["lat"], errors="coerce"); rx_lon = pd.to_numeric(dt["lon"], errors="coerce")
    dt["distance_m"] = phase22._haversine_m(site_lat.to_numpy(), site_lon.to_numpy(), rx_lat.to_numpy(), rx_lon.to_numpy())
    brg = _bearing_deg(site_lat.to_numpy(), site_lon.to_numpy(), rx_lat.to_numpy(), rx_lon.to_numpy())
    az = pd.to_numeric(dt["azimuth"], errors="coerce").fillna(0.0).to_numpy()
    dt["azimuth_delta_deg"] = np.abs((brg - az + 180.0) % 360.0 - 180.0)

    bias = phase22._bias_table(dt, "dt_minus_with_terrain_physical_db")
    dt = phase22._attach_bias(dt, bias, BASE_BIAS_COL)
    dt = _apply_delta_and_prep(dt, "assigned_strict_cell_key")
    return phase25._split_dt_by_grid(dt)


# --------------------------------------------------------------------------------- score
def _fit_and_score():
    dt = _dt_inputs()
    candidates = _candidate_inputs()
    train = dt[dt["phase25_split"] == "train"].copy()
    validation = dt[dt["phase25_split"] == "validation"].copy()
    train_outdoor = train[train["obstruction_branch"].astype(str) != INDOOR_BRANCH].copy()
    print(f"[PHASE29] train_dt={len(train)} outdoor={len(train_outdoor)} val={len(validation)} candidates={len(candidates)}")

    tgs, layers = phase25._fit_group_hierarchy(train_outdoor)
    local_models = phase25._fit_local_models(tgs)

    def _score(df: pd.DataFrame) -> pd.DataFrame:
        s = phase27._rename_out(
            phase25._finalize(phase25._apply_local_model(phase25._apply_group_hierarchy(df, layers), local_models))
        )
        s = s.drop(columns=[c for c in ("phase27_dynamic_rsrp", "phase27_dynamic_rsrp_unclipped") if c in s.columns])
        base_u = pd.to_numeric(s[BASE_UNCLIP], errors="coerce")
        is_indoor = s["obstruction_branch"].astype(str) == INDOOR_BRANCH
        dyn = pd.to_numeric(s["phase27_total_dynamic_correction_db"], errors="coerce").fillna(0.0)
        s["phase27_total_dynamic_correction_db"] = np.where(is_indoor, 0.0, dyn)
        s["phase29_dynamic_rsrp_unclipped"] = np.where(is_indoor, base_u, base_u + dyn)
        s["phase29_dynamic_rsrp"] = valid_model_rsrp(s["phase29_dynamic_rsrp_unclipped"])
        return s

    return _score(train), _score(validation), _score(candidates), layers


def _metrics(m, p):
    return phase25._metrics(m, p)


def main() -> None:
    _ensure_dirs()
    train_s, val_s, cand_s, layers = _fit_and_score()

    # --- serving grid (best server per cell), reuse Phase 27 aggregation shape ---
    agg_in = cand_s.rename(columns={"phase29_dynamic_rsrp": "phase27_dynamic_rsrp",
                                    "phase29_dynamic_rsrp_unclipped": "phase27_dynamic_rsrp_unclipped"})
    serving_all = phase27._aggregate(agg_in).rename(columns={
        "phase27_dynamic_best_rsrp": "phase29_dynamic_best_rsrp",
        "phase27_dynamic_mean_rsrp": "phase29_dynamic_mean_rsrp",
    })

    phase22._save_frame(cand_s, OUT_DIR / "phase29_scored_candidates_project210")
    phase22._save_frame(val_s, OUT_DIR / "phase29_validation_dt_project210")

    # Phase 27 reference (already on disk) for the before/after comparison
    p27_val = phase22._read_frame(
        PROJECT_DIR / "cost231_phase27_dynamic_on_corrected_obstruction" / "phase27_validation_dt_project210"
    )

    summary = {
        "scope": "real per-tilt antenna patterns; 4G CCVVPX308, 5G Kathrein 800109221; base = Phase 27 logic",
        "generic_replaced": f"3GPP parametric {GEN_MAX_GAIN} dBi / {GEN_H_BW} H / {GEN_V_BW} V",
        "boresight_gain_dbi": BORESIGHT_GAIN_DBI,
        "technology": {},
    }
    for tech in ["4G", "5G"]:
        serving = serving_all[serving_all["technology"].astype(str) == tech].copy()
        vt = val_s[val_s["technology"].astype(str) == tech].copy()
        tr = train_s[train_s["technology"].astype(str) == tech].copy()
        vt_out = vt[vt["obstruction_branch"].astype(str) != INDOOR_BRANCH]
        tr_out = tr[tr["obstruction_branch"].astype(str) != INDOOR_BRANCH]
        p27 = p27_val[p27_val["technology"].astype(str) == tech].copy() if not p27_val.empty else pd.DataFrame()
        p27_out = p27[p27["obstruction_branch"].astype(str) != INDOOR_BRANCH] if not p27.empty else pd.DataFrame()

        phase22._save_frame(serving, OUT_DIR / f"phase29_serving_grid_{tech.lower()}_project210")
        env = serving.get("serving_environment")
        gd = pd.to_numeric(cand_s.loc[cand_s["technology"].astype(str) == tech, "phase29_antenna_gain_delta_db"], errors="coerce")
        sv = pd.to_numeric(serving["phase29_dynamic_best_rsrp"], errors="coerce")

        summary["technology"][tech] = {
            "antenna_gain_delta_db": {
                "median": round(float(gd.median()), 2),
                "p10": round(float(gd.quantile(0.1)), 2),
                "p90": round(float(gd.quantile(0.9)), 2),
            },
            "held_out_outdoor_phase27_generic": _metrics(p27_out["rsrp_measured"], p27_out["phase27_dynamic_rsrp"]) if not p27_out.empty else None,
            "held_out_outdoor_phase29_real_antenna": _metrics(vt_out["rsrp_measured"], vt_out["phase29_dynamic_rsrp"]),
            "insample_outdoor_phase29": _metrics(tr_out["rsrp_measured"], tr_out["phase29_dynamic_rsrp"]),
            "serving": {
                "rows": int(len(serving)),
                "no_coverage": int(sv.isna().sum()),
                "median": round(float(sv.median()), 1),
                "outdoor_median": round(float(pd.to_numeric(serving.loc[env == "outdoor", "phase29_dynamic_best_rsrp"], errors="coerce").median()), 1) if env is not None else None,
                "indoor_median": round(float(pd.to_numeric(serving.loc[env == "indoor", "phase29_dynamic_best_rsrp"], errors="coerce").median()), 1) if env is not None else None,
            },
        }

        phase22._plot_cdf(
            [
                ("DT measured (outdoor)", vt_out["rsrp_measured"], "#111827"),
                ("Phase 27 generic antenna", p27_out["phase27_dynamic_rsrp"] if not p27_out.empty else pd.Series(dtype=float), "#2563eb"),
                ("Phase 29 real antenna", vt_out["phase29_dynamic_rsrp"], "#16a34a"),
            ],
            f"Project 210 {tech}: Phase 29 held-out outdoor DT - generic vs real antenna",
            IMAGE_DIR / f"phase29_{tech.lower()}_dt_generic_vs_real.png",
        )
        phase22._plot_cdf(
            [
                ("Phase 27 generic abs err", (p27_out["rsrp_measured"] - p27_out["phase27_dynamic_rsrp"]).abs() if not p27_out.empty else pd.Series(dtype=float), "#2563eb"),
                ("Phase 29 real antenna abs err", (vt_out["rsrp_measured"] - vt_out["phase29_dynamic_rsrp"]).abs(), "#16a34a"),
            ],
            f"Project 210 {tech}: Phase 29 held-out outdoor DT absolute error",
            IMAGE_DIR / f"phase29_{tech.lower()}_abs_err_generic_vs_real.png",
        )
        m29 = summary["technology"][tech]["held_out_outdoor_phase29_real_antenna"]
        m27 = summary["technology"][tech]["held_out_outdoor_phase27_generic"]
        print(f"[PHASE29] {tech}  gain delta median {summary['technology'][tech]['antenna_gain_delta_db']['median']} dB")
        print(f"          held-out outdoor MAE: generic {m27['mae']:.2f} -> real antenna {m29['mae']:.2f}   "
              f"(bias {m27['bias']:.2f} -> {m29['bias']:.2f}, p90 {m27['p90_abs']:.1f} -> {m29['p90_abs']:.1f})")
        print(f"          serving: {summary['technology'][tech]['serving']['no_coverage']} no-cov, "
              f"outdoor med {summary['technology'][tech]['serving']['outdoor_median']}  indoor med {summary['technology'][tech]['serving']['indoor_median']}")

    (OUT_DIR / "phase29_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[PHASE29] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
