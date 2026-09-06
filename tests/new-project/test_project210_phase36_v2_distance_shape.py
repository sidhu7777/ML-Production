"""Phase 36 v2: DT-frequency RE-BAND of the 4G cells (keeps v1 intact).

The 4G drive test `earfcn` shows the phones were on Band 3 (1800 MHz, 5788 samples),
Band 1 (2100 MHz, 3432) and Band 7 (2600 MHz, 2174) - only 20 samples were on Band 28
(700 MHz). But Phase 26 stamped ~9777 rows as band 28 / 775.5 MHz and matched them to
700 MHz cells by nearest location, so the 4G physical runs ~13-18 dB too hot (700 vs
~1800-2600 MHz Hata frequency term). That is why the v1 map showed 4G ~6 dB below 5G
between the DT roads even though the clean co-located drive test has them within +1 dB.

v2 derives each 4G cell's REAL operating frequency from the median measured `earfcn` of
the DT on it and applies the COST-231-Hata frequency correction
  -33.9 * log10(f_true / f_labelled)
to that cell's raw - candidates and DT alike. Cells with < 8 DT points, or where measured
and label agree within 200 MHz, keep their label. (The ~5600 samples on 2100/2600 MHz
whose cells the model has NO inventory for still cannot be matched - that needs the
network cell database and is out of scope.)

Calibration = Phase 36 v1's exactly (Phase 25 hierarchy + local IDW). An earlier v2
also swapped in a distance-shaped ladder; it did not close the gap and cost ~0.2 dB MAE,
so v2 keeps v1's calibration and changes ONLY the cell frequencies. The distance-ladder
code is retained below (unused) for reference.

Everything else - physical core, real antenna, serving-cell re-assignment, Water handling,
DT split, backlobe drop - is Phase 36 v1, imported unchanged. No other phase is touched.
Output: data/project_210_taiwan/cost231_phase36_v2_distance_shape/  (v1 is untouched)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import test_project210_phase36_final as p36
import test_project210_phase25_hierarchical_dynamic_calibration as phase25

phase22 = p36.phase22
valid_model_rsrp = p36.valid_model_rsrp
BASE_COL, BASE_UNCLIPPED = p36.BASE_COL, p36.BASE_UNCLIPPED

OUT_DIR = p36.PROJECT_DIR / "cost231_phase36_v2_distance_shape"
IMAGE_DIR = OUT_DIR / "images"
V1_DIR = p36.OUT_DIR

# log-spaced (path loss is ~log distance)
_DIST_BINS = [0.0, 100.0, 175.0, 300.0, 500.0, 850.0, 1400.0, np.inf]
_DIST_LAB = ["d100", "d175", "d300", "d500", "d850", "d1400", "d1400p"]
_LAYERS = [
    ("l1_techband", ["technology", "band"], 120.0, 1),
    ("l2_distance", ["technology", "band", "dbk"], 40.0, 25),
    ("l3_clutter", ["technology", "band", "clutter_class", "obstruction_branch"], 45.0, 25),
]


def _dbk(df: pd.DataFrame) -> np.ndarray:
    return (pd.cut(pd.to_numeric(df["distance_m"], errors="coerce"), _DIST_BINS, labels=_DIST_LAB)
            .astype("object").fillna("d100").to_numpy())


# --------------------------------------------------------------------------- DT-frequency re-band
def _earfcn_to_mhz(e: float) -> float:
    """LTE FDD DL centre frequency from E-ARFCN (3GPP TS 36.101 Table 5.7.3-1, common bands)."""
    if not np.isfinite(e):
        return np.nan
    e = int(round(e))
    if 0 <= e <= 599:        # B1  2100
        return 2110.0 + 0.1 * e
    if 1200 <= e <= 1949:    # B3  1800
        return 1805.0 + 0.1 * (e - 1200)
    if 2750 <= e <= 3449:    # B7  2600
        return 2620.0 + 0.1 * (e - 2750)
    if 6150 <= e <= 6449:    # B20  800
        return 791.0 + 0.1 * (e - 6150)
    if 9210 <= e <= 9659:    # B28  700
        return 758.0 + 0.1 * (e - 9210)
    return np.nan


def _cell_true_freq(dt_scored: pd.DataFrame, min_n: int = 8) -> dict:
    """Per 4G cell: the real operating frequency, taken as the median measured E-ARFCN
    frequency of the (post-re-assignment) DT points on that cell. Returns
    {strict_cell_key: (f_labelled, f_true, n)} only where they disagree by > 200 MHz."""
    raw = pd.read_parquet(p36.PHASE26_DIR / "phase26_dt_scored_project210.parquet")
    ef = raw.set_index("dt_row_id")["earfcn"].to_dict()
    d = dt_scored[dt_scored["technology"].astype(str) == "4G"].copy()
    d["f_meas"] = d["dt_row_id"].map(ef).map(_earfcn_to_mhz)
    d["f_label"] = pd.to_numeric(d["frequency_mhz"], errors="coerce")
    out = {}
    for cell, g in d.dropna(subset=["f_meas"]).groupby("assigned_strict_cell_key"):
        if len(g) < min_n:
            continue
        f_true = float(g["f_meas"].median())
        f_label = float(g["f_label"].median())
        if abs(f_true - f_label) > 200.0:
            out[str(cell)] = (f_label, f_true, int(len(g)))
    return out


def _reband(df: pd.DataFrame, freq_map: dict, key_col: str) -> np.ndarray:
    """COST-231-Hata frequency correction to add to the raw for re-banded cells:
    -33.9*log10(f_true / f_labelled)  (a negative number - more loss at the higher band)."""
    keys = df[key_col].astype(str).to_numpy()
    corr = np.zeros(len(df), dtype=float)
    for i, k in enumerate(keys):
        hit = freq_map.get(k)
        if hit is not None:
            f_label, f_true, _ = hit
            corr[i] = -33.9 * np.log10(max(f_true, 1.0) / max(f_label, 1.0))
    return corr


def _fit_shape(fit_df: pd.DataFrame) -> list:
    w = fit_df.copy()
    w["dbk"] = _dbk(w)
    w["_r"] = pd.to_numeric(w["rsrp_measured"], errors="coerce") - pd.to_numeric(w[BASE_COL], errors="coerce")
    hier = []
    for name, keys, shr, mn in _LAYERS:
        t = (w.dropna(subset=["_r"]).groupby(keys, dropna=False)
             .agg(n=("_r", "size"), med=("_r", "median")).reset_index())
        t = t[t["n"] >= mn].copy()
        t["corr"] = t["med"] * t["n"] / (t["n"] + shr)
        hier.append((name, keys, t[keys + ["corr"]]))
        w = w.merge(t[keys + ["corr"]], on=keys, how="left")
        w["_r"] = w["_r"] - w["corr"].fillna(0.0)
        w = w.drop(columns="corr")
    return hier


def _apply_shape(df: pd.DataFrame, hier: list) -> tuple[np.ndarray, dict]:
    tmp = df.copy()
    tmp["dbk"] = _dbk(tmp)
    adj = np.zeros(len(df), dtype=float)
    parts = {}
    for name, keys, t in hier:
        c = tmp.merge(t, on=keys, how="left")["corr"].fillna(0.0).to_numpy(dtype=float)
        parts[name] = c
        adj = adj + c
    return adj, parts


def _fit(fit_df: pd.DataFrame):
    hier = _fit_shape(fit_df)
    gb = fit_df.copy()
    adj, _ = _apply_shape(gb, hier)
    gb["phase25_group_pred_rsrp"] = pd.to_numeric(gb[BASE_COL], errors="coerce") + adj
    local = phase25._fit_local_models(gb)
    return hier, local


def _score(df: pd.DataFrame, hier: list, local) -> pd.DataFrame:
    s = df.copy()
    adj, parts = _apply_shape(s, hier)
    s["phase25_group_pred_rsrp"] = pd.to_numeric(s[BASE_COL], errors="coerce") + adj
    s = phase25._apply_local_model(s, local)
    is_indoor = (s["obstruction_branch"].astype(str) == "indoor").to_numpy()
    lc = pd.to_numeric(s["local_residual_correction_db"], errors="coerce").fillna(0.0).to_numpy()
    corr = adj + np.where(is_indoor, 0.0, lc)
    s["phase36_shape_adj_db"] = adj
    s["phase36_l2_distance_db"] = parts.get("l2_distance", np.zeros(len(s)))
    s["phase36_local_corr_db"] = np.where(is_indoor, 0.0, lc)
    s["phase36_total_correction_db"] = corr
    s["phase36_final_rsrp_unclipped"] = pd.to_numeric(s[BASE_UNCLIPPED], errors="coerce") + corr
    s["phase36_final_rsrp"] = valid_model_rsrp(s["phase36_final_rsrp_unclipped"])
    sup = pd.to_numeric(s.get("local_residual_support_n"), errors="coerce").fillna(0.0).to_numpy()
    s["phase36_confidence"] = np.clip(0.45 + 0.35 * (sup >= phase25.LOCAL_MIN_NEIGHBORS) + 0.20 * (~is_indoor), 0.0, 1.0)
    return s


def _polygon_gap_by_distance(cand_scored: pd.DataFrame) -> dict:
    out = {}
    for tech in ("4G", "5G"):
        x = cand_scored[cand_scored["technology"].astype(str) == tech].copy()
        x["_u"] = pd.to_numeric(x["phase36_final_rsrp_unclipped"], errors="coerce")
        best = x.sort_values("_u").groupby("grid_id").tail(1)
        best["db"] = pd.cut(pd.to_numeric(best["distance_m"], errors="coerce"),
                            [0, 150, 300, 600, 1200, 9e9], labels=["<150", "150-300", "300-600", "600-1200", ">1200"])
        out[tech] = {str(k): round(float(pd.to_numeric(g["phase36_final_rsrp"], errors="coerce").median()), 1)
                     for k, g in best.groupby("db", observed=True)}
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    dt = p36._dt_inputs()
    cand = p36._candidate_inputs()
    # NB: no separate 5G level anchor - the L1 [technology, band] layer IS that anchor now.

    # ---- THE CHANGE: DT-frequency re-band of the 4G cells the earfcn shows are not on 700 MHz ----
    freq_map = _cell_true_freq(dt)
    reband_rows = []
    if freq_map:
        dt_corr = _reband(dt, freq_map, "assigned_strict_cell_key")
        cand_corr = _reband(cand, freq_map, "strict_cell_key")
        print(f"[36v2] re-banded {len(freq_map)} 4G cells; touched "
              f"{int((cand_corr != 0).sum())}/{len(cand)} candidate rows, {int((dt_corr != 0).sum())}/{len(dt)} DT rows")
        for k, (fl, ft, n) in sorted(freq_map.items(), key=lambda kv: -kv[1][2]):
            db = -33.9 * np.log10(ft / fl)
            reband_rows.append({"cell": k, "label_mhz": round(fl), "true_mhz": round(ft), "dt_n": n, "raw_shift_db": round(db, 1)})
            print(f"    {k}:  {fl:.0f} -> {ft:.0f} MHz  (n={n}, {db:+.1f} dB)")
        for frame, corr in ((dt, dt_corr), (cand, cand_corr)):
            for col in (BASE_UNCLIPPED, "phase24_no_lock_reference_rsrp_unclipped"):
                frame[col] = pd.to_numeric(frame[col], errors="coerce") + corr
            frame[BASE_COL] = valid_model_rsrp(frame[BASE_UNCLIPPED])
            frame["phase24_no_lock_reference_rsrp"] = valid_model_rsrp(frame["phase24_no_lock_reference_rsrp_unclipped"])
    else:
        print("[36v2] no 4G cell re-band triggered")

    train = dt[dt["phase25_split"] == "train"].copy()
    valid = dt[dt["phase25_split"] == "validation"].copy()
    fit = train[(train["obstruction_branch"].astype(str) != "indoor") & (~train["p36_backlobe"])].copy()
    print(f"[36v2] fit={len(fit)}  valid={len(valid)}  candidates={len(cand)}")

    # calibration = Phase 36 v1's, exactly
    layers, local_models = p36._fit(fit)
    if layers:
        pd.concat(layers, ignore_index=True).to_csv(OUT_DIR / "phase36v2_group_corrections.csv", index=False)
    valid_scored = p36._score(valid, layers, local_models)
    cand_scored = p36._score(cand, layers, local_models)
    serving_all = p36._aggregate(cand_scored)

    phase22._save_frame(valid_scored, OUT_DIR / "phase36v2_validation_dt_project210")
    phase22._save_frame(cand_scored, OUT_DIR / "phase36v2_scored_candidates_project210")

    v1 = json.loads((V1_DIR / "phase36_summary.json").read_text(encoding="utf-8")).get("technology", {})
    summary = {
        "scope": "Phase 36 v2 - DT-frequency RE-BAND of the 4G cells. Each 4G cell's frequency is taken from "
                 "the median measured E-ARFCN of its DT (the earfcn shows 1800/2100/2600 MHz, not the labelled "
                 "700 MHz); raw is shifted by -33.9*log10(f_true/f_label). Calibration = Phase 36 v1's exactly.",
        "reband": reband_rows,
        "technology": {},
        "polygon_median_by_distance": {"v2": _polygon_gap_by_distance(cand_scored)},
    }

    for tech in ("4G", "5G"):
        serv = serving_all[serving_all["technology"].astype(str) == tech].copy()
        vt = valid_scored[valid_scored["technology"].astype(str) == tech]
        vo = vt[(vt["obstruction_branch"].astype(str) != "indoor")]
        voc = vo[~vo["p36_backlobe"]]
        phase22._save_frame(serv, OUT_DIR / f"phase36v2_serving_grid_{tech.lower()}_project210")
        m_final = phase25._metrics(voc["rsrp_measured"], voc["phase36_final_rsrp"])
        m_bl = phase25._metrics(vo["rsrp_measured"], vo["phase36_final_rsrp"])
        sv = pd.to_numeric(serv["phase36_final_best_rsrp"], errors="coerce")
        env = serv["serving_environment"]
        v1t = v1.get(tech, {})
        summary["technology"][tech] = {
            "v2_held_out_outdoor_final": m_final,
            "v2_held_out_incl_backlobe": m_bl,
            "v1_held_out_outdoor_final": v1t.get("held_out_outdoor_final", {}),
            "v2_serving_grid": {
                "median": round(float(sv.median()), 1),
                "outdoor_median": round(float(pd.to_numeric(serv.loc[env == "outdoor", "phase36_final_best_rsrp"], errors="coerce").median()), 1),
                "indoor_median": round(float(pd.to_numeric(serv.loc[env == "indoor", "phase36_final_best_rsrp"], errors="coerce").median()), 1),
                "no_coverage": int(sv.isna().sum()),
            },
            "v1_serving_grid": v1t.get("serving_grid", {}),
        }
        p = summary["technology"][tech]
        print(f"[36v2] {tech}  final MAE {m_final['mae']:.2f} (bias {m_final['bias']:+.2f})  "
              f"v1 was {p['v1_held_out_outdoor_final'].get('mae', float('nan')):.2f}")
        print(f"[36v2] {tech}  serving outdoor {p['v2_serving_grid']['outdoor_median']}  "
              f"(v1 was {p['v1_serving_grid'].get('outdoor_median', '?')})")

    g = _polygon_gap_by_distance(cand_scored)
    print("\n[36v2] polygon median by distance (4G / 5G / gap):")
    for db in ["<150", "150-300", "300-600", "600-1200", ">1200"]:
        a, b = g["4G"].get(db), g["5G"].get(db)
        if a is not None and b is not None:
            print(f"    {db:9s}  4G {a:7.1f}   5G {b:7.1f}   5G-4G {b - a:+.1f}")

    (OUT_DIR / "phase36v2_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[36v2] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
