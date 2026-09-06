"""Phase 38: EARFCN-correct 4G DT re-match, then the Phase 36 v1 pipeline. No other phase touched.

The 4G drive test spans four LTE carriers (the `earfcn` field):
    700 MHz  B28  - 20 samples      - model HAS cells (39 sectors / 17 sites)
    1800 MHz B3   - 5788 samples     - model HAS cells (17 sectors / 6 sites)
    2100 MHz B1   - 3432 samples     - model has NO cells
    2600 MHz B7   - 2174 samples     - model has NO cells
Phase 26 matched almost everything to 700 MHz cells by nearest location, so the 4G
physical was compared to measurements from the wrong carrier.

Phase 38 re-matches every 4G DT sample by the band its `earfcn` reports:
  * B28  -> kept (already nearest-matched to a Band-28 cell).
  * B3   -> re-assigned to the strongest Band-3 (1840 MHz) candidate at that DT's grid;
            its obstruction / terrain / building-geo / site / tilt / height all come from
            that Band-3 candidate row. If the grid has no Band-3 candidate (out of the
            6-site 1800 MHz footprint) the sample is EXCLUDED.
  * B1 / B7 (2100 / 2600 MHz) -> EXCLUDED. You cannot calibrate a prediction that has no
            cell. Those areas simply have no 4G DT constraint here.
  * 5G  -> untouched.

Everything after the re-match is Phase 36 v1, imported unchanged: real antenna delta,
serving-cell geometry hygiene, Water handling, Phase 25 hierarchical calibration, DT
split, backlobe drop, 5G level anchor. Candidate / serving inventory is Phase 26's
(700 + 1800 MHz), unchanged - no cells are invented.

Output: data/project_210_taiwan/cost231_phase38_earfcn_rematch/   (v1 / v2 untouched)
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
import test_project210_phase33_5g_38_901 as phase33

phase22 = p36.phase22
valid_model_rsrp = p36.valid_model_rsrp
BASE_COL, BASE_UNCLIPPED = p36.BASE_COL, p36.BASE_UNCLIPPED

OUT_DIR = p36.PROJECT_DIR / "cost231_phase38_earfcn_rematch"
IMAGE_DIR = OUT_DIR / "images"
V1_DIR = p36.OUT_DIR


def _earfcn_band(e: float) -> str:
    if not np.isfinite(e):
        return "unknown"
    e = int(round(e))
    if 1200 <= e <= 1949:
        return "B3"                     # 1800 MHz
    if 9210 <= e <= 9659:
        return "B28"                    # 700 MHz
    if (0 <= e <= 599) or (2750 <= e <= 3799):
        return "B1B7"                   # 2100 / 2600 MHz - no cells
    return "other"


_CARRY = ["obstruction_branch", "terrain_diffraction_loss_db", "building_geo_correction_db",
          "clutter_class", "site_lat", "site_lon", "Etilt", "Mtilt", "Height",
          "frequency_mhz", "band"]


def _rematch_4g(raw_dt: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    d = raw_dt.copy()
    d["p38_true_band"] = pd.to_numeric(d["earfcn"], errors="coerce").map(_earfcn_band)
    d["p38_excluded"] = False
    d["p38_rematched"] = False
    is4g = d["assigned_technology"].astype(str) == "4G"

    b3 = cand[(cand["technology"].astype(str) == "4G") & (cand["band"].astype(str) == "3")].copy()
    b3["_r"] = pd.to_numeric(b3["raw_cost231_rsrp_unclipped"], errors="coerce")
    b3best = (b3.sort_values("_r").groupby("grid_id").tail(1)
              .rename(columns={"azimuth_x": "azimuth"}).set_index("grid_id"))

    for i in d.index[is4g]:
        tb = d.at[i, "p38_true_band"]
        if tb == "B28":
            continue
        if tb != "B3":                       # B1B7 / other / unknown
            d.at[i, "p38_excluded"] = True
            continue
        g = d.at[i, "nearest_grid_id"]
        if g not in b3best.index:
            d.at[i, "p38_excluded"] = True    # outside the 1800 MHz footprint
            continue
        row = b3best.loc[g]
        d.at[i, "assigned_strict_cell_key"] = row["strict_cell_key"]
        d.at[i, "azimuth"] = row["azimuth"]
        for c in _CARRY:
            d.at[i, c] = row[c]
        d.at[i, "p38_rematched"] = True
    return d


# -------- Phase 36 v1's _dt_inputs, but starting from the re-matched frame --------
def _dt_inputs_from(dt_rematched: pd.DataFrame) -> pd.DataFrame:
    dt = p36._reassign_dt_serving(dt_rematched, p36._sector_table())
    dt = phase33._geometry_for_dt(dt)
    frames = []
    for tech in ("4G", "5G"):
        sub = dt[dt["assigned_technology"].astype(str) == tech].copy()
        if sub.empty:
            continue
        raw = p36._dt_cost231_raw(sub)
        frames.append(p36._build_physical(sub, tech, raw))
    out = pd.concat(frames, ignore_index=True)
    out["technology"] = out["assigned_technology"].astype(str)
    out = phase25._add_common_features(out, "assigned_strict_cell_key")
    out["phase24_no_lock_reference_rsrp_unclipped"] = pd.to_numeric(out[BASE_UNCLIPPED], errors="coerce")
    out["phase24_no_lock_reference_rsrp"] = valid_model_rsrp(out["phase24_no_lock_reference_rsrp_unclipped"])
    out["p36_backlobe"] = pd.to_numeric(out["azimuth_delta_deg"], errors="coerce").fillna(0.0) > p36.BACKLOBE_DROP_DEG
    out["p38_excluded"] = out.get("p38_excluded", False)
    return phase25._split_dt_by_grid(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    raw_dt = pd.read_parquet(p36.PHASE26_DIR / "phase26_dt_scored_project210.parquet")
    cand_raw = pd.read_parquet(p36.PHASE26_DIR / "phase26_scored_candidates_project210.parquet")

    dt_rm = _rematch_4g(raw_dt, cand_raw)
    d4 = dt_rm[dt_rm["assigned_technology"].astype(str) == "4G"]
    mix = d4["p38_true_band"].value_counts().to_dict()
    print(f"[P38] 4G DT true-band mix: {mix}")
    print(f"[P38] 4G DT: kept-B28={int((d4.p38_true_band == 'B28').sum())}  "
          f"rematched-B3={int(d4.p38_rematched.sum())}  "
          f"excluded={int(d4.p38_excluded.sum())} "
          f"(of which B1/B7={int((d4.p38_excluded & (d4.p38_true_band == 'B1B7')).sum())}, "
          f"B3-no-coverage={int((d4.p38_excluded & (d4.p38_true_band == 'B3')).sum())})")
    dt_rm[["dt_row_id", "assigned_technology", "earfcn", "p38_true_band", "p38_rematched",
           "p38_excluded", "assigned_strict_cell_key", "frequency_mhz", "band"]].to_parquet(
        OUT_DIR / "phase38_dt_rematch_map_project210.parquet", index=False)

    dt = _dt_inputs_from(dt_rm)
    cand = p36._candidate_inputs()

    # 5G level anchor (Phase 36 v1), on non-excluded outdoor 5G train DT
    tr5 = dt[(dt["phase25_split"] == "train") & (dt["technology"] == "5G")
             & (dt["obstruction_branch"].astype(str) != "indoor")
             & (dt["clutter_class"].astype(str) != "Water") & (~dt["p38_excluded"])]
    off5 = float((pd.to_numeric(tr5["rsrp_measured"], errors="coerce")
                  - pd.to_numeric(tr5[BASE_COL], errors="coerce")).median())
    off5 = 0.0 if not np.isfinite(off5) else off5
    for frame in (dt, cand):
        m5 = (frame["technology"].astype(str) == "5G").to_numpy()
        for col in (BASE_UNCLIPPED, "phase24_no_lock_reference_rsrp_unclipped"):
            frame.loc[m5, col] = pd.to_numeric(frame.loc[m5, col], errors="coerce") + off5
        frame.loc[m5, BASE_COL] = valid_model_rsrp(frame.loc[m5, BASE_UNCLIPPED])
        frame.loc[m5, "phase24_no_lock_reference_rsrp"] = valid_model_rsrp(frame.loc[m5, "phase24_no_lock_reference_rsrp_unclipped"])
    print(f"[P38] 5G level anchor {off5:+.1f} dB")

    train = dt[dt["phase25_split"] == "train"].copy()
    valid = dt[dt["phase25_split"] == "validation"].copy()
    fit = train[(train["obstruction_branch"].astype(str) != "indoor")
                & (~train["p36_backlobe"]) & (~train["p38_excluded"])].copy()
    print(f"[P38] fit={len(fit)}  valid={len(valid)}  candidates={len(cand)}  "
          f"(excluded from fit: {int(train['p38_excluded'].sum())})")

    layers, local_models = p36._fit(fit)
    if layers:
        tb = pd.concat(layers, ignore_index=True)
        pd.concat(layers, ignore_index=True).to_csv(OUT_DIR / "phase38_group_corrections.csv", index=False)
        print("[P38] tech_band:", {f"{r.technology}/{r.band}": round(r.tech_band_correction_db, 1)
                                    for r in tb[tb.layer == "tech_band"].itertuples()})

    valid_scored = p36._score(valid, layers, local_models)
    cand_scored = p36._score(cand, layers, local_models)
    serving_all = p36._aggregate(cand_scored)

    phase22._save_frame(valid_scored, OUT_DIR / "phase38_validation_dt_project210")
    phase22._save_frame(cand_scored, OUT_DIR / "phase38_scored_candidates_project210")

    v1 = json.loads((V1_DIR / "phase36_summary.json").read_text(encoding="utf-8")).get("technology", {})
    summary = {
        "scope": "Phase 38 - EARFCN-correct 4G DT re-match (B3->1840 cells, B1/B7 excluded, B28 kept), "
                 "then Phase 36 v1 pipeline unchanged. Candidate inventory unchanged (700 + 1800 MHz); no cells invented.",
        "dt_4g_true_band_mix": mix,
        "dt_4g": {
            "kept_b28": int((d4.p38_true_band == "B28").sum()),
            "rematched_b3": int(d4.p38_rematched.sum()),
            "excluded_b1b7": int((d4.p38_excluded & (d4.p38_true_band == "B1B7")).sum()),
            "excluded_b3_no_coverage": int((d4.p38_excluded & (d4.p38_true_band == "B3")).sum()),
        },
        "g5_level_anchor_db": round(off5, 2),
        "technology": {},
        "polygon_median_by_distance": {"p38": _poly_gap(cand_scored)},
    }

    for tech in ("4G", "5G"):
        serv = serving_all[serving_all["technology"].astype(str) == tech].copy()
        vt = valid_scored[valid_scored["technology"].astype(str) == tech]
        vo = vt[(vt["obstruction_branch"].astype(str) != "indoor") & (~vt["p38_excluded"])]
        voc = vo[~vo["p36_backlobe"]]
        phase22._save_frame(serv, OUT_DIR / f"phase38_serving_grid_{tech.lower()}_project210")
        m_final = phase25._metrics(voc["rsrp_measured"], voc["phase36_final_rsrp"])
        m_phys = phase25._metrics(voc["rsrp_measured"], voc[BASE_COL])
        sv = pd.to_numeric(serv["phase36_final_best_rsrp"], errors="coerce")
        env = serv["serving_environment"]
        v1t = v1.get(tech, {})
        summary["technology"][tech] = {
            "p38_held_out_outdoor_physical": m_phys,
            "p38_held_out_outdoor_final": m_final,
            "v1_held_out_outdoor_final": v1t.get("held_out_outdoor_final", {}),
            "p38_serving_grid": {
                "median": round(float(sv.median()), 1),
                "outdoor_median": round(float(pd.to_numeric(serv.loc[env == "outdoor", "phase36_final_best_rsrp"], errors="coerce").median()), 1),
                "indoor_median": round(float(pd.to_numeric(serv.loc[env == "indoor", "phase36_final_best_rsrp"], errors="coerce").median()), 1),
                "no_coverage": int(sv.isna().sum()),
            },
            "v1_serving_grid": v1t.get("serving_grid", {}),
            "validation_dt_rows_scored": int(len(voc)),
        }
        p = summary["technology"][tech]
        print(f"[P38] {tech}  physical MAE {m_phys['mae']:.2f} -> final MAE {m_final['mae']:.2f} "
              f"(bias {m_final['bias']:+.2f})  v1 final was {p['v1_held_out_outdoor_final'].get('mae', float('nan')):.2f}")
        print(f"[P38] {tech}  serving outdoor {p['p38_serving_grid']['outdoor_median']}  "
              f"(v1 was {p['v1_serving_grid'].get('outdoor_median', '?')})")

    g = _poly_gap(cand_scored)
    print("\n[P38] polygon median by distance (4G / 5G / gap):")
    for db in ["<150", "150-300", "300-600", "600-1200", ">1200"]:
        a, b = g["4G"].get(db), g["5G"].get(db)
        if a is not None and b is not None:
            print(f"    {db:9s}  4G {a:7.1f}   5G {b:7.1f}   5G-4G {b - a:+.1f}")

    (OUT_DIR / "phase38_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[P38] wrote {OUT_DIR}")


def _poly_gap(cand_scored: pd.DataFrame) -> dict:
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


if __name__ == "__main__":
    main()
