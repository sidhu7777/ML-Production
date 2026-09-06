"""
Phase 27: Phase 25 hierarchical dynamic residual calibration, applied on top of the
Phase 26 corrected-obstruction physical layer (instead of the Phase 24 physical layer).

Rationale: Phase 26 fixed the outdoor obstruction physics (dominant-obstacle diffraction,
DT-point physical MAE 4G 14.2 vs Phase 22's 18.7) but only carries a coarse ~8-bucket
phase19-style bias, so its DT-point MAE sits at ~12 (4G) / ~16 (5G). Phase 25's dynamic
calibration (tech/band -> clutter/terrain/branch -> sector -> local residual field, all
shrinkage-regularised, trained on 70% of DT grids and validated on the held-out 30%)
took the Phase 24 layer from ~13 to ~9. This phase runs that exact machinery on the
Phase 26 physical layer.

Test-only. No DT replacement. Reuses Phase 25's fitting functions unchanged - only the
base physical column is swapped (phase26_physical_with_terrain_rsrp).
"""
from __future__ import annotations

import json
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
from phase_rsrp_guard import valid_model_rsrp

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE26_DIR = PROJECT_DIR / "cost231_phase26_corrected_obstruction_profile"
OUT_DIR = PROJECT_DIR / "cost231_phase27_dynamic_on_corrected_obstruction"
IMAGE_DIR = OUT_DIR / "images"

# Phase 25's group/local functions all key off this column name; we alias the Phase 26
# corrected-obstruction physical prediction onto it so the machinery runs unchanged.
BASE_COL = "phase24_physical_with_terrain_rsrp"
BASE_BIAS_COL = "phase24_phase19_style_bias_db"

INDOOR_BRANCH = "indoor"
# Phase 26 already applies its complete frequency-and-depth O2I term to indoor
# candidates. Phase 27 must not add a second indoor penetration loss. There is
# no indoor DT, so the dynamic hierarchy remains outdoor-only.


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _candidate_inputs() -> pd.DataFrame:
    cand = phase22._read_frame(PHASE26_DIR / "phase26_scored_candidates_project210")
    cand = cand.rename(
        columns={
            "phase26_physical_with_terrain_rsrp": BASE_COL,
            "phase26_phase19_bias_db": BASE_BIAS_COL,
        }
    )
    cand["technology"] = cand["technology"].astype(str)
    cand = phase25._add_common_features(cand, "strict_cell_key")
    cand[BASE_BIAS_COL] = pd.to_numeric(cand.get(BASE_BIAS_COL), errors="coerce").fillna(0.0)
    cand["phase24_no_lock_reference_rsrp_unclipped"] = (
        pd.to_numeric(cand[BASE_COL], errors="coerce") + cand[BASE_BIAS_COL]
    )
    cand["phase24_no_lock_reference_rsrp"] = valid_model_rsrp(
        cand["phase24_no_lock_reference_rsrp_unclipped"]
    )
    return cand


def _dt_inputs() -> pd.DataFrame:
    dt = phase22._read_frame(PHASE26_DIR / "phase26_dt_scored_project210")
    dt = dt.rename(columns={"phase26_physical_with_terrain_rsrp": BASE_COL})
    dt["technology"] = dt["assigned_technology"].astype(str)
    dt = phase25._add_common_features(dt, "assigned_strict_cell_key")

    # phase19-style bucket bias for the no-lock baseline reference (Water already excluded
    # inside phase22._bias_table); the dynamic layers below do not depend on it.
    bias = phase22._bias_table(dt, "dt_minus_with_terrain_physical_db")
    dt = phase22._attach_bias(dt, bias, BASE_BIAS_COL)
    dt[BASE_BIAS_COL] = pd.to_numeric(dt[BASE_BIAS_COL], errors="coerce").fillna(0.0)
    dt["phase24_no_lock_reference_rsrp_unclipped"] = (
        pd.to_numeric(dt[BASE_COL], errors="coerce") + dt[BASE_BIAS_COL]
    )
    dt["phase24_no_lock_reference_rsrp"] = valid_model_rsrp(
        dt["phase24_no_lock_reference_rsrp_unclipped"]
    )
    return phase25._split_dt_by_grid(dt)


_RENAME_OUT = {
    "phase25_dynamic_rsrp": "phase27_dynamic_rsrp",
    "phase25_dynamic_rsrp_unclipped": "phase27_dynamic_rsrp_unclipped",
    "phase25_total_dynamic_correction_db": "phase27_total_dynamic_correction_db",
    "phase25_group_pred_rsrp": "phase27_group_pred_rsrp",
    "phase25_confidence": "phase27_confidence",
}


def _rename_out(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in _RENAME_OUT.items() if k in df.columns})


def _aggregate(candidates_scored: pd.DataFrame) -> pd.DataFrame:
    work = candidates_scored.copy()
    work["_dyn"] = pd.to_numeric(work["phase27_dynamic_rsrp"], errors="coerce")
    # branch of the winning (best dynamic) candidate per grid, so the serving grid can be
    # split indoor vs outdoor downstream.
    best_rows = work.sort_values("_dyn").groupby(["technology", "grid_id"], dropna=False).tail(1)
    branch = best_rows[["technology", "grid_id", "obstruction_branch"]].rename(
        columns={"obstruction_branch": "serving_obstruction_branch"}
    )
    branch["serving_environment"] = np.where(
        branch["serving_obstruction_branch"].astype(str) == "indoor", "indoor", "outdoor"
    )
    agg = (
        candidates_scored.groupby(["technology", "grid_id"], dropna=False)
        .agg(
            phase26_physical_best_rsrp=(BASE_COL, "max"),
            phase26_physical_mean_rsrp=(BASE_COL, "mean"),
            phase27_no_lock_best_rsrp=("phase24_no_lock_reference_rsrp", "max"),
            phase27_dynamic_best_rsrp=("phase27_dynamic_rsrp", "max"),
            phase27_dynamic_mean_rsrp=("phase27_dynamic_rsrp", "mean"),
            phase27_total_dynamic_correction_db_mean=("phase27_total_dynamic_correction_db", "mean"),
            tech_band_correction_db_mean=("tech_band_correction_db", "mean"),
            clutter_terrain_correction_db_mean=("clutter_terrain_correction_db", "mean"),
            sector_correction_db_mean=("sector_correction_db", "mean"),
            local_residual_correction_db_mean=("local_residual_correction_db", "mean"),
            local_residual_support_n_mean=("local_residual_support_n", "mean"),
            phase27_confidence_mean=("phase27_confidence", "mean"),
        )
        .reset_index()
        .merge(branch, on=["technology", "grid_id"], how="left")
    )
    grid = phase22._read_frame(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210")
    bounds = grid[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]].copy()
    all_grid = pd.concat(
        [bounds[["grid_id"]].assign(technology=t) for t in ["4G", "5G"]], ignore_index=True
    )
    return all_grid.merge(agg, on=["technology", "grid_id"], how="left").merge(bounds, on="grid_id", how="left")


def main() -> None:
    _ensure_dirs()
    dt = _dt_inputs()
    candidates = _candidate_inputs()
    train = dt[dt["phase25_split"] == "train"].copy()
    validation = dt[dt["phase25_split"] == "validation"].copy()
    print(f"[PHASE27] train_dt={len(train)} validation_dt={len(validation)} candidates={len(candidates)}")

    # Indoor cells get NO DT calibration (there is no indoor drive test). Fit the dynamic
    # hierarchy on OUTDOOR training DT only, so the outdoor calibration is not contaminated
    # by the handful of misclassified "indoor" DT points (outdoor NLOS on/near buildings).
    train_outdoor = train[train["obstruction_branch"].astype(str) != INDOOR_BRANCH].copy()
    print(f"[PHASE27] train_outdoor={len(train_outdoor)} (dropped {len(train) - len(train_outdoor)} misclassified indoor DT)")
    train_group_scored, layers = phase25._fit_group_hierarchy(train_outdoor)
    local_models = phase25._fit_local_models(train_group_scored)

    def _score(df: pd.DataFrame) -> pd.DataFrame:
        scored = _rename_out(
            phase25._finalize(
                phase25._apply_local_model(phase25._apply_group_hierarchy(df, layers), local_models)
            )
        )
        base_unclipped = pd.to_numeric(
            scored["phase26_physical_with_terrain_rsrp_unclipped"], errors="coerce"
        )
        is_indoor = scored["obstruction_branch"].astype(str) == INDOOR_BRANCH
        dyn_corr = pd.to_numeric(scored["phase27_total_dynamic_correction_db"], errors="coerce").fillna(0.0)

        # Outdoor: corrected-obstruction physical + hierarchical dynamic correction, applied
        #   to the UNCLIPPED value so a near-threshold cell can still be recovered, clipped once.
        # Indoor: Phase 26 already includes O2I once; retain that physical value with
        #   NO second O2I term and NO DT-derived calibration.
        scored["phase27_indoor_o2i_extra_db"] = 0.0
        scored["phase27_total_dynamic_correction_db"] = np.where(is_indoor, 0.0, dyn_corr)
        scored["phase27_dynamic_rsrp_unclipped"] = np.where(
            is_indoor, base_unclipped, base_unclipped + dyn_corr
        )
        scored["phase27_dynamic_rsrp"] = valid_model_rsrp(scored["phase27_dynamic_rsrp_unclipped"])
        return scored

    train_scored = _score(train)
    validation_scored = _score(validation)
    candidates_scored = _score(candidates)

    phase22._save_frame(validation_scored, OUT_DIR / "phase27_validation_dt_project210")
    phase22._save_frame(candidates_scored, OUT_DIR / "phase27_scored_candidates_project210")

    layer_table = pd.concat(layers, ignore_index=True) if layers else pd.DataFrame()
    if not layer_table.empty:
        layer_table.to_csv(OUT_DIR / "phase27_group_corrections.csv", index=False)

    serving_all = _aggregate(candidates_scored)
    summary = {
        "base_physical_layer": "phase26_physical_with_terrain_rsrp (corrected dominant-obstacle obstruction)",
        "calibration": "phase25 hierarchical dynamic (tech_band -> clutter_terrain -> sector -> local), held-out 30% DT-grid validation, no DT replacement",
        "layers": {
            "tech_band": {"min_n": phase25.TECH_BAND_MIN_N, "shrink_n": phase25.TECH_BAND_SHRINK_N},
            "clutter_terrain": {"min_n": phase25.CLUTTER_TERRAIN_MIN_N, "shrink_n": phase25.CLUTTER_TERRAIN_SHRINK_N},
            "sector": {"min_n": phase25.SECTOR_MIN_N, "shrink_n": phase25.SECTOR_SHRINK_N},
            "local": {"min_neighbors": phase25.LOCAL_MIN_NEIGHBORS, "k_neighbors": phase25.LOCAL_K_NEIGHBORS},
        },
        "technology": {},
    }

    for tech in ["4G", "5G"]:
        serving = serving_all[serving_all["technology"].astype(str) == tech].copy()
        vt = validation_scored[validation_scored["technology"].astype(str) == tech].copy()
        tr = train[train["technology"].astype(str) == tech]
        phase22._save_frame(serving, OUT_DIR / f"phase27_serving_grid_{tech.lower()}_project210")
        phase22._plot_cdf(
            [
                ("Phase26 physical", serving["phase26_physical_best_rsrp"], "#6b7280"),
                ("Phase26 + phase19 bias", serving["phase27_no_lock_best_rsrp"], "#2563eb"),
                ("Phase27 dynamic", serving["phase27_dynamic_best_rsrp"], "#16a34a"),
            ],
            f"Project 210 {tech}: Phase 27 full-polygon serving CDF",
            IMAGE_DIR / f"phase27_{tech.lower()}_full_polygon_cdf.png",
        )
        phase22._plot_cdf(
            [
                ("DT measured", vt["rsrp_measured"], "#111827"),
                ("Phase26 + phase19 bias", vt["phase24_no_lock_reference_rsrp"], "#2563eb"),
                ("Phase27 dynamic", vt["phase27_dynamic_rsrp"], "#16a34a"),
            ],
            f"Project 210 {tech}: Phase 27 held-out DT CDF",
            IMAGE_DIR / f"phase27_{tech.lower()}_validation_dt_cdf.png",
        )
        phase22._plot_cdf(
            [
                ("Phase26+bias abs err", (vt["rsrp_measured"] - vt["phase24_no_lock_reference_rsrp"]).abs(), "#2563eb"),
                ("Phase27 dynamic abs err", (vt["rsrp_measured"] - vt["phase27_dynamic_rsrp"]).abs(), "#16a34a"),
            ],
            f"Project 210 {tech}: Phase 27 held-out DT absolute error CDF",
            IMAGE_DIR / f"phase27_{tech.lower()}_validation_abs_error_cdf.png",
        )
        vt_out = vt[vt["obstruction_branch"].astype(str) != "indoor"]
        vt_in = vt[vt["obstruction_branch"].astype(str) == "indoor"]
        serving_out = serving.loc[serving["serving_environment"] == "outdoor", "phase27_dynamic_best_rsrp"]
        serving_in = serving.loc[serving["serving_environment"] == "indoor", "phase27_dynamic_best_rsrp"]
        phase22._plot_cdf(
            [
                ("1 - DT measured (outdoor, at DT points)", vt_out["rsrp_measured"], "#4b5563"),
                ("2 - Predicted at those DT points", vt_out["phase27_dynamic_rsrp"], "#2563eb"),
                ("3 - Predicted outdoor (whole polygon)", serving_out, "#16a34a"),
                ("4 - Predicted indoor (whole polygon, 3GPP O2I, no DT)", serving_in, "#f59e0b"),
            ],
            f"Project 210 {tech}: Phase 27 - DT accuracy vs whole-polygon prediction",
            IMAGE_DIR / f"phase27_{tech.lower()}_dt_vs_polygon_indoor_outdoor_cdf.png",
        )
        # Matched O2I CDF: the same winning indoor candidate is evaluated with
        # and without its Phase 26 O2I term. Unlike the whole-polygon split,
        # this removes the different site-distance distribution of indoor and
        # outdoor grid populations.
        indoor_candidates = candidates_scored[
            (candidates_scored["technology"].astype(str) == tech)
            & (candidates_scored["obstruction_branch"].astype(str) == INDOOR_BRANCH)
        ].copy()
        indoor_winners = (
            indoor_candidates.sort_values("phase27_dynamic_rsrp_unclipped")
            .groupby("grid_id", dropna=False)
            .tail(1)
            .copy()
        )
        indoor_winners["outdoor_equivalent_rsrp"] = (
            pd.to_numeric(indoor_winners["raw_cost231_rsrp_unclipped"], errors="coerce")
            - pd.to_numeric(indoor_winners["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
        indoor_winners["matched_o2i_loss_db"] = (
            indoor_winners["outdoor_equivalent_rsrp"]
            - pd.to_numeric(indoor_winners["phase27_dynamic_rsrp_unclipped"], errors="coerce")
        )
        phase22._plot_cdf(
            [
                ("Outdoor-equivalent (same indoor cells, no O2I)", indoor_winners["outdoor_equivalent_rsrp"], "#16a34a"),
                ("Indoor prediction (same cells, one O2I term)", indoor_winners["phase27_dynamic_rsrp_unclipped"], "#f59e0b"),
            ],
            f"Project 210 {tech}: Phase 27 matched-cell O2I CDF",
            IMAGE_DIR / f"phase27_{tech.lower()}_matched_o2i_cdf.png",
        )
        tr_out = train_scored[
            (train_scored["technology"].astype(str) == tech)
            & (train_scored["obstruction_branch"].astype(str) != "indoor")
        ]
        serving_env = serving.get("serving_environment")
        summary["technology"][tech] = {
            "grid_rows": int(len(serving)),
            "no_coverage_grid_rows": int(pd.to_numeric(serving["phase27_dynamic_best_rsrp"], errors="coerce").isna().sum()),
            "indoor_grid_rows": int((serving_env == "indoor").sum()) if serving_env is not None else None,
            "train_dt_rows": int(len(tr)),
            "validation_dt_rows": int(len(vt)),
            # Headline accuracy is OUTDOOR held-out DT - the only real ground truth.
            "held_out_outdoor_phase26_plus_phase19_bias": phase25._metrics(
                vt_out["rsrp_measured"], vt_out["phase24_no_lock_reference_rsrp"]
            ),
            "held_out_outdoor_phase27_dynamic": phase25._metrics(vt_out["rsrp_measured"], vt_out["phase27_dynamic_rsrp"]),
            "insample_outdoor_phase27_dynamic": phase25._metrics(tr_out["rsrp_measured"], tr_out["phase27_dynamic_rsrp"]),
            # Indoor "DT" = misclassified outdoor-NLOS points; kept only for reference, not a target.
            "held_out_misclassified_indoor_dt_ref_only": phase25._metrics(vt_in["rsrp_measured"], vt_in["phase27_dynamic_rsrp"]),
            "mean_phase26_physical_best_rsrp": float(pd.to_numeric(serving["phase26_physical_best_rsrp"], errors="coerce").mean()),
            "mean_phase27_dynamic_best_rsrp": float(pd.to_numeric(serving["phase27_dynamic_best_rsrp"], errors="coerce").mean()),
            "mean_phase27_dynamic_outdoor_best_rsrp": float(
                pd.to_numeric(serving.loc[serving_env == "outdoor", "phase27_dynamic_best_rsrp"], errors="coerce").mean()
            ) if serving_env is not None else None,
            "mean_phase27_dynamic_indoor_best_rsrp": float(
                pd.to_numeric(serving.loc[serving_env == "indoor", "phase27_dynamic_best_rsrp"], errors="coerce").mean()
            ) if serving_env is not None else None,
            "matched_o2i_loss_db": {
                "n": int(pd.to_numeric(indoor_winners["matched_o2i_loss_db"], errors="coerce").notna().sum()),
                "mean": float(pd.to_numeric(indoor_winners["matched_o2i_loss_db"], errors="coerce").mean()),
                "median": float(pd.to_numeric(indoor_winners["matched_o2i_loss_db"], errors="coerce").median()),
                "p10": float(pd.to_numeric(indoor_winners["matched_o2i_loss_db"], errors="coerce").quantile(0.10)),
                "p90": float(pd.to_numeric(indoor_winners["matched_o2i_loss_db"], errors="coerce").quantile(0.90)),
            },
            "mean_confidence": float(pd.to_numeric(serving["phase27_confidence_mean"], errors="coerce").mean()),
        }
        print(f"[PHASE27] {tech} held-out OUTDOOR dynamic: {summary['technology'][tech]['held_out_outdoor_phase27_dynamic']}")
        print(f"[PHASE27] {tech} mean serving  outdoor={summary['technology'][tech]['mean_phase27_dynamic_outdoor_best_rsrp']:.1f}"
              f"  indoor={summary['technology'][tech]['mean_phase27_dynamic_indoor_best_rsrp']:.1f}")

    (OUT_DIR / "phase27_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("[PHASE27] summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
