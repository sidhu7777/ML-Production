"""Phase 39: equal-power 4G/5G diagnostic on the Phase 38 input.

This phase is a diagnostic only. It takes Phase 38's EARFCN-correct DT handling
and then normalizes the physical layer so both technologies use the same
46 dBm power reference:

* 4G: 46 dBm already. Band 28 gets the explicit COST-231 low-band floor
      approximation: use 1500 MHz floor plus -33.9*log10(775.5/1500), recorded
      as a +9.71 dB RSRP offset.
* 5G: tx_power is normalized to 46 dBm. Because the production raw is computed
      at 2600 MHz, the n78 correction is changed from Phase 36's -2.58 dB to
      the COST-231 frequency term -33.9*log10(3300/2600), about -3.51 dB.

The equal-power physical surface is the main output. A Phase-25-style calibrated
surface is also saved for reference, but it should not be used to answer the
same-power propagation question because calibration can absorb power errors.

No earlier phase file or output is modified.
Output: data/project_210_taiwan/cost231_phase39_equal_power_diagnostic/
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
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import test_project210_phase25_hierarchical_dynamic_calibration as phase25
import test_project210_phase36_final as p36
import test_project210_phase38_earfcn_rematch as p38

phase22 = p36.phase22
valid_model_rsrp = p36.valid_model_rsrp
BASE_COL, BASE_UNCLIPPED = p36.BASE_COL, p36.BASE_UNCLIPPED

OUT_DIR = p36.PROJECT_DIR / "cost231_phase39_equal_power_diagnostic"
IMAGE_DIR = OUT_DIR / "images"

TARGET_TX_POWER_DBM = 46.0
COST231_FREQ_COEFF = 33.9
B28_LABEL_MHZ = 775.5
COST231_LOW_FLOOR_MHZ = 1500.0
N78_RAW_MHZ = 2600.0
N78_REAL_MHZ = 3300.0

B28_LOW_BAND_OFFSET_DB = -COST231_FREQ_COEFF * math.log10(B28_LABEL_MHZ / COST231_LOW_FLOOR_MHZ)
N78_COST231_OFFSET_DB = -COST231_FREQ_COEFF * math.log10(N78_REAL_MHZ / N78_RAW_MHZ)
N78_REPLACE_PHASE36_DELTA_DB = N78_COST231_OFFSET_DB - p36.N78_OFFSET_DB


def _apply_equal_power_assumptions(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    out = df.copy()
    tech = out["technology"].astype(str)
    band = out["band"].astype(str)
    tx = pd.to_numeric(out.get("tx_power"), errors="coerce")
    tx_default = pd.Series(np.where(tech.eq("5G"), 50.0, TARGET_TX_POWER_DBM), index=out.index, dtype=float)
    tx = tx.fillna(tx_default)

    power_shift = TARGET_TX_POWER_DBM - tx
    freq_shift = pd.Series(0.0, index=out.index, dtype=float)
    freq_shift.loc[tech.eq("4G") & band.eq("28")] = B28_LOW_BAND_OFFSET_DB
    freq_shift.loc[tech.eq("5G")] = N78_REPLACE_PHASE36_DELTA_DB

    total_shift = power_shift + freq_shift
    out["phase39_target_tx_power_dbm"] = TARGET_TX_POWER_DBM
    out["phase39_original_tx_power_dbm"] = tx.astype(float)
    out["phase39_power_normalization_db"] = power_shift.astype(float)
    out["phase39_frequency_offset_db"] = freq_shift.astype(float)
    out["phase39_total_physical_shift_db"] = total_shift.astype(float)
    out["phase39_key_col"] = key_col

    for col in (BASE_UNCLIPPED, "phase24_no_lock_reference_rsrp_unclipped"):
        out[col] = pd.to_numeric(out[col], errors="coerce") + total_shift
    out[BASE_COL] = valid_model_rsrp(out[BASE_UNCLIPPED])
    out["phase24_no_lock_reference_rsrp"] = valid_model_rsrp(out["phase24_no_lock_reference_rsrp_unclipped"])
    if "phase36_raw_rsrp" in out.columns:
        out["phase36_raw_rsrp"] = pd.to_numeric(out["phase36_raw_rsrp"], errors="coerce") + total_shift

    out["phase39_equal_power_rsrp_unclipped"] = pd.to_numeric(out[BASE_UNCLIPPED], errors="coerce")
    out["phase39_equal_power_rsrp"] = valid_model_rsrp(out["phase39_equal_power_rsrp_unclipped"])
    return out


def _aggregate_p39(cand_scored: pd.DataFrame) -> pd.DataFrame:
    work = cand_scored.copy()
    work["_eq"] = pd.to_numeric(work["phase39_equal_power_rsrp_unclipped"], errors="coerce")
    work["_final"] = pd.to_numeric(work["phase39_final_rsrp_unclipped"], errors="coerce")
    best_eq = work.sort_values("_eq").groupby(["technology", "grid_id"], dropna=False).tail(1)
    best_final = work.sort_values("_final").groupby(["technology", "grid_id"], dropna=False).tail(1)

    env = best_eq[["technology", "grid_id", "obstruction_branch"]].copy()
    env["serving_environment"] = np.where(env["obstruction_branch"].astype(str).eq("indoor"), "indoor", "outdoor")
    agg = (
        work.groupby(["technology", "grid_id"], dropna=False)
        .agg(
            phase39_equal_power_best_rsrp=("phase39_equal_power_rsrp", "max"),
            phase39_equal_power_mean_rsrp=("phase39_equal_power_rsrp", "mean"),
            phase39_final_best_rsrp=("phase39_final_rsrp", "max"),
            phase39_final_mean_rsrp=("phase39_final_rsrp", "mean"),
            phase39_total_physical_shift_db_mean=("phase39_total_physical_shift_db", "mean"),
            phase39_total_correction_db_mean=("phase39_total_correction_db", "mean"),
            phase39_confidence_mean=("phase39_confidence", "mean"),
        )
        .reset_index()
        .merge(env[["technology", "grid_id", "serving_environment"]], on=["technology", "grid_id"], how="left")
    )
    final_cols = best_final[["technology", "grid_id", "strict_cell_key"]].rename(
        columns={"strict_cell_key": "phase39_final_serving_cell"}
    )
    agg = agg.merge(final_cols, on=["technology", "grid_id"], how="left")

    grid = pd.read_parquet(p36.PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210.parquet")
    bounds = grid[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]]
    full = pd.concat([bounds[["grid_id"]].assign(technology=t) for t in ("4G", "5G")], ignore_index=True)
    return full.merge(agg, on=["technology", "grid_id"], how="left").merge(bounds, on="grid_id", how="left")


def _distance_gap(cand_scored: pd.DataFrame, value_col: str) -> dict:
    out = {}
    sort_col = value_col.replace("_rsrp", "_rsrp_unclipped")
    for tech in ("4G", "5G"):
        x = cand_scored[cand_scored["technology"].astype(str).eq(tech)].copy()
        x["_u"] = pd.to_numeric(x[sort_col], errors="coerce")
        best = x.sort_values("_u").groupby("grid_id").tail(1)
        best["db"] = pd.cut(pd.to_numeric(best["distance_m"], errors="coerce"),
                            [0, 150, 300, 600, 1200, 9e9],
                            labels=["<150", "150-300", "300-600", "600-1200", ">1200"])
        out[tech] = {
            str(k): round(float(pd.to_numeric(g[value_col], errors="coerce").median()), 1)
            for k, g in best.groupby("db", observed=True)
        }
    return out


def _copy_phase39_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["phase39_total_correction_db"] = pd.to_numeric(out["phase36_total_correction_db"], errors="coerce")
    out["phase39_local_corr_db"] = pd.to_numeric(out["phase36_local_corr_db"], errors="coerce")
    out["phase39_shape_adj_db"] = pd.to_numeric(out["phase36_shape_adj_db"], errors="coerce")
    out["phase39_final_rsrp_unclipped"] = pd.to_numeric(out["phase36_final_rsrp_unclipped"], errors="coerce")
    out["phase39_final_rsrp"] = valid_model_rsrp(out["phase39_final_rsrp_unclipped"])
    out["phase39_confidence"] = pd.to_numeric(out["phase36_confidence"], errors="coerce").fillna(0.3)
    return out


def _cdf(items, title: str, path: Path) -> None:
    phase22._plot_cdf(items, title, path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    raw_dt = pd.read_parquet(p36.PHASE26_DIR / "phase26_dt_scored_project210.parquet")
    cand_raw = pd.read_parquet(p36.PHASE26_DIR / "phase26_scored_candidates_project210.parquet")
    dt_rm = p38._rematch_4g(raw_dt, cand_raw)
    dt = p38._dt_inputs_from(dt_rm)
    cand = p36._candidate_inputs()

    dt = _apply_equal_power_assumptions(dt, "assigned_strict_cell_key")
    cand = _apply_equal_power_assumptions(cand, "strict_cell_key")

    train = dt[dt["phase25_split"].astype(str).eq("train")].copy()
    valid = dt[dt["phase25_split"].astype(str).eq("validation")].copy()
    fit = train[
        (train["obstruction_branch"].astype(str) != "indoor")
        & (~train["p36_backlobe"].astype(bool))
        & (~train["p38_excluded"].astype(bool))
    ].copy()
    print(f"[P39] fit={len(fit)} valid={len(valid)} candidates={len(cand)}")
    print(f"[P39] target power={TARGET_TX_POWER_DBM:.1f} dBm  "
          f"B28 floor offset={B28_LOW_BAND_OFFSET_DB:+.2f} dB  "
          f"n78 COST231 offset={N78_COST231_OFFSET_DB:+.2f} dB")

    layers, local_models = p36._fit(fit)
    if layers:
        pd.concat(layers, ignore_index=True).to_csv(OUT_DIR / "phase39_group_corrections.csv", index=False)
        tb = pd.concat(layers, ignore_index=True)
        print("[P39] tech_band:", {
            f"{r.technology}/{r.band}": round(r.tech_band_correction_db, 1)
            for r in tb[tb.layer == "tech_band"].itertuples()
        })

    valid_scored = _copy_phase39_score_columns(p36._score(valid, layers, local_models))
    cand_scored = _copy_phase39_score_columns(p36._score(cand, layers, local_models))
    serving_all = _aggregate_p39(cand_scored)

    phase22._save_frame(valid_scored, OUT_DIR / "phase39_validation_dt_project210")
    phase22._save_frame(cand_scored, OUT_DIR / "phase39_scored_candidates_project210")

    summary = {
        "scope": "Phase 39 equal-power diagnostic. Input follows Phase 38 EARFCN-correct DT rematch. "
                 "Both technologies normalized to 46 dBm; 4G B28 uses explicit COST-231 1500 MHz floor "
                 "plus low-band offset; 5G uses 2600 MHz COST-231 base plus -3.51 dB 3300 MHz term.",
        "diagnostic_warning": "Use phase39_equal_power_* only for the same-power 4G-vs-5G coverage comparison. "
                              "Do not present equal-power DT MAE as model accuracy because the measured DT used real "
                              "network powers, while this phase intentionally forces both technologies to 46 dBm. "
                              "Use phase39_final_* for DT accuracy reference.",
        "params": {
            "target_tx_power_dbm": TARGET_TX_POWER_DBM,
            "b28_label_mhz": B28_LABEL_MHZ,
            "cost231_low_floor_mhz": COST231_LOW_FLOOR_MHZ,
            "b28_low_band_offset_db": round(B28_LOW_BAND_OFFSET_DB, 3),
            "n78_raw_mhz": N78_RAW_MHZ,
            "n78_real_mhz": N78_REAL_MHZ,
            "n78_cost231_offset_db": round(N78_COST231_OFFSET_DB, 3),
            "phase36_n78_offset_replaced_db": p36.N78_OFFSET_DB,
            "n78_extra_shift_vs_phase36_db": round(N78_REPLACE_PHASE36_DELTA_DB, 3),
        },
        "manager_conclusion": {},
        "technology": {},
        "polygon_median_by_distance": {
            "equal_power": _distance_gap(cand_scored, "phase39_equal_power_rsrp"),
            "calibrated_final": _distance_gap(cand_scored, "phase39_final_rsrp"),
        },
    }

    for tech in ("4G", "5G"):
        serv = serving_all[serving_all["technology"].astype(str).eq(tech)].copy()
        vt = valid_scored[valid_scored["technology"].astype(str).eq(tech)].copy()
        vo = vt[
            (vt["obstruction_branch"].astype(str) != "indoor")
            & (~vt["p36_backlobe"].astype(bool))
            & (~vt["p38_excluded"].astype(bool))
        ].copy()
        phase22._save_frame(serv, OUT_DIR / f"phase39_serving_grid_{tech.lower()}_project210")

        equal_metrics = phase25._metrics(vo["rsrp_measured"], vo["phase39_equal_power_rsrp"])
        final_metrics = phase25._metrics(vo["rsrp_measured"], vo["phase39_final_rsrp"])
        env = serv["serving_environment"]
        eq = pd.to_numeric(serv["phase39_equal_power_best_rsrp"], errors="coerce")
        fin = pd.to_numeric(serv["phase39_final_best_rsrp"], errors="coerce")
        summary["technology"][tech] = {
            "held_out_outdoor_equal_power_physical": equal_metrics,
            "held_out_outdoor_calibrated_final": final_metrics,
            "serving_grid_equal_power": {
                "rows": int(len(serv)),
                "no_coverage": int(eq.isna().sum()),
                "median": round(float(eq.median()), 1),
                "outdoor_median": round(float(pd.to_numeric(serv.loc[env == "outdoor", "phase39_equal_power_best_rsrp"], errors="coerce").median()), 1),
                "indoor_median": round(float(pd.to_numeric(serv.loc[env == "indoor", "phase39_equal_power_best_rsrp"], errors="coerce").median()), 1),
            },
            "serving_grid_calibrated_final": {
                "rows": int(len(serv)),
                "no_coverage": int(fin.isna().sum()),
                "median": round(float(fin.median()), 1),
                "outdoor_median": round(float(pd.to_numeric(serv.loc[env == "outdoor", "phase39_final_best_rsrp"], errors="coerce").median()), 1),
                "indoor_median": round(float(pd.to_numeric(serv.loc[env == "indoor", "phase39_final_best_rsrp"], errors="coerce").median()), 1),
            },
            "validation_dt_rows_scored": int(len(vo)),
        }
        _cdf(
            [
                ("1 - DT measured (outdoor)", vo["rsrp_measured"], "#111827"),
                ("2 - Phase 39 calibrated predicted at DT", vo["phase39_final_rsrp"], "#2563eb"),
                ("3 - Phase 39 calibrated outdoor polygon",
                 serv.loc[serv["serving_environment"] == "outdoor", "phase39_final_best_rsrp"], "#16a34a"),
                ("4 - Phase 39 calibrated indoor polygon",
                 serv.loc[serv["serving_environment"] == "indoor", "phase39_final_best_rsrp"], "#f59e0b"),
            ],
            f"Project 210 {tech}: Phase 39 calibrated DT accuracy reference",
            IMAGE_DIR / f"phase39_{tech.lower()}_calibrated_accuracy_cdf.png",
        )
        _cdf(
            [
                (f"{tech} equal-power outdoor polygon",
                 serv.loc[serv["serving_environment"] == "outdoor", "phase39_equal_power_best_rsrp"], "#16a34a"),
                (f"{tech} equal-power indoor polygon",
                 serv.loc[serv["serving_environment"] == "indoor", "phase39_equal_power_best_rsrp"], "#f59e0b"),
            ],
            f"Project 210 {tech}: Phase 39 equal-power polygon diagnostic",
            IMAGE_DIR / f"phase39_{tech.lower()}_equal_power_cdf.png",
        )
        print(f"[P39] {tech} equal-power MAE {equal_metrics['mae']:.2f} "
              f"(bias {equal_metrics['bias']:+.2f}); final MAE {final_metrics['mae']:.2f}")
        print(f"[P39] {tech} equal-power serving outdoor "
              f"{summary['technology'][tech]['serving_grid_equal_power']['outdoor_median']} dBm")

    g = summary["polygon_median_by_distance"]["equal_power"]
    s4 = summary["technology"]["4G"]["serving_grid_equal_power"]
    s5 = summary["technology"]["5G"]["serving_grid_equal_power"]
    summary["manager_conclusion"] = {
        "question": "If 4G and 5G are normalized to the same 46 dBm power, which technology is stronger?",
        "answer": "4G is stronger than 5G on the equal-power outdoor polygon.",
        "equal_power_4g_outdoor_median_dbm": s4["outdoor_median"],
        "equal_power_5g_outdoor_median_dbm": s5["outdoor_median"],
        "four_g_advantage_db": round(float(s4["outdoor_median"]) - float(s5["outdoor_median"]), 1),
    }
    serv4 = serving_all[serving_all["technology"].astype(str).eq("4G")]
    serv5 = serving_all[serving_all["technology"].astype(str).eq("5G")]
    _cdf(
        [
            ("4G equal-power outdoor polygon", serv4.loc[serv4["serving_environment"] == "outdoor", "phase39_equal_power_best_rsrp"], "#2563eb"),
            ("5G equal-power outdoor polygon", serv5.loc[serv5["serving_environment"] == "outdoor", "phase39_equal_power_best_rsrp"], "#dc2626"),
            ("4G equal-power indoor polygon", serv4.loc[serv4["serving_environment"] == "indoor", "phase39_equal_power_best_rsrp"], "#60a5fa"),
            ("5G equal-power indoor polygon", serv5.loc[serv5["serving_environment"] == "indoor", "phase39_equal_power_best_rsrp"], "#f97316"),
        ],
        "Project 210 Phase 39: equal-power 4G vs 5G polygon comparison",
        IMAGE_DIR / "phase39_equal_power_4g_vs_5g_polygon_cdf.png",
    )
    print("\n[P39] equal-power polygon median by distance (4G / 5G / gap):")
    for db in ["<150", "150-300", "300-600", "600-1200", ">1200"]:
        a, b = g["4G"].get(db), g["5G"].get(db)
        if a is not None and b is not None:
            print(f"    {db:9s}  4G {a:7.1f}   5G {b:7.1f}   5G-4G {b - a:+.1f}")

    (OUT_DIR / "phase39_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[P39] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
