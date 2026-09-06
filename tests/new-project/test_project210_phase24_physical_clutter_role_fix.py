"""
Phase 24: physical clutter-role fix on top of Phase 22.

This phase is test-only. It reuses Phase 22's COST-231, antenna, DEM terrain
diffraction, branch classification, and existing DT residual style. The only
physical correction changed here is the outdoor clutter role:

  - explicit indoor/building-obstructed paths keep the explicit building loss;
  - Dense Urban/Urban/Suburban remain obstruction proxies only when no explicit
    path obstruction exists;
  - independent clear-path clutter classes are counted once, not twice.

That makes the Phase 22 vs Phase 24 comparison isolate the double-counted
clear-path clutter issue without changing the terrain or indoor/O2I logic.
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

import streamlit_project210_phase15_radius_progression as phase15
import test_project210_phase17_full_polygon_geo_dt_comparison as phase17
import test_project210_phase22_terrain_diffraction_comparison as phase22
from phase_rsrp_guard import valid_model_rsrp


PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE20_DIR = PROJECT_DIR / "cost231_phase20_5g_real_dt_match"
PHASE22_DIR = PROJECT_DIR / "cost231_phase22_terrain_diffraction_comparison"
OUT_DIR = PROJECT_DIR / "cost231_phase24_physical_clutter_role_fix"
IMAGE_DIR = OUT_DIR / "images"

RSRP_MIN, RSRP_MAX = phase17.RSRP_MIN, phase17.RSRP_MAX
INDEPENDENT_CLEAR_CLUTTER = {
    cls for cls in phase15.DEFAULT_CLUTTER_WEIGHTS if cls not in phase15.OBSTRUCTION_PROXY_CLUTTER_CLASSES
}


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _read_frame(stem: Path) -> pd.DataFrame:
    return phase22._read_frame(stem)


def _save_frame(df: pd.DataFrame, stem: Path) -> None:
    phase22._save_frame(df, stem)


def _metrics(measured: pd.Series, predicted: pd.Series) -> dict:
    err = pd.to_numeric(measured, errors="coerce") - pd.to_numeric(predicted, errors="coerce")
    arr = err.dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mae": math.nan, "rmse": math.nan, "bias": math.nan, "p90_abs": math.nan}
    return {
        "mae": float(np.mean(np.abs(arr))),
        "rmse": float(np.sqrt(np.mean(np.square(arr)))),
        "bias": float(np.mean(arr)),
        "p90_abs": float(np.quantile(np.abs(arr), 0.90)),
    }


def _apply_phase24_clutter_role_fix(df: pd.DataFrame, raw_col: str) -> pd.DataFrame:
    out = df.copy()
    branch = out["obstruction_branch"].astype(str)
    clutter = out["clutter_class"].astype("object")
    clutter_key = clutter.where(clutter.notna(), "")
    weights = clutter_key.map(phase15.DEFAULT_CLUTTER_WEIGHTS).fillna(0.0).astype(float)

    clear = branch.eq("clear")
    proxy_class = clutter_key.isin(phase15.OBSTRUCTION_PROXY_CLUTTER_CLASSES)
    independent_clear = clear & clutter_key.isin(INDEPENDENT_CLEAR_CLUTTER)

    # Phase 19/22 clear branch applied env_adj + proxy. For non-proxy classes
    # those are the same physical clutter signal, so Phase 24 removes one copy.
    correction_delta = np.where(independent_clear, -weights, 0.0)
    old_correction = pd.to_numeric(out["building_geo_correction_db"], errors="coerce").fillna(0.0)
    out["phase24_building_clutter_correction_db"] = old_correction + correction_delta
    out["phase24_correction_delta_db"] = correction_delta
    out["phase24_removed_duplicate_clear_clutter_db"] = np.where(independent_clear, weights, 0.0)
    out["phase24_proxy_clutter_suppressed"] = branch.isin(["indoor", "obstructed"]) & proxy_class

    role = np.full(len(out), "clear_no_clutter_signal", dtype=object)
    role[branch.eq("indoor").to_numpy()] = "indoor_existing_o2i_unchanged"
    role[branch.eq("obstructed").to_numpy()] = "explicit_building_obstruction_proxy_suppressed"
    role[(clear & proxy_class).to_numpy()] = "clutter_proxy_no_path_obstruction"
    role[independent_clear.to_numpy()] = "independent_clear_clutter_single_count"
    out["phase24_clutter_role"] = role

    raw = pd.to_numeric(out[raw_col], errors="coerce")
    terrain = pd.to_numeric(out["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0)
    out["phase24_physical_no_terrain_rsrp_unclipped"] = raw + out["phase24_building_clutter_correction_db"]
    out["phase24_physical_with_terrain_rsrp_unclipped"] = (
        raw + out["phase24_building_clutter_correction_db"] - terrain
    )
    out["phase24_physical_no_terrain_rsrp"] = valid_model_rsrp(
        out["phase24_physical_no_terrain_rsrp_unclipped"]
    )
    out["phase24_physical_with_terrain_rsrp"] = valid_model_rsrp(
        out["phase24_physical_with_terrain_rsrp_unclipped"]
    )
    return out


def _attach_bias(df: pd.DataFrame, bias: pd.DataFrame, out_col: str) -> pd.DataFrame:
    return phase22._attach_bias(df, bias, out_col)


def _corrected_dt_replacements(dt: pd.DataFrame) -> pd.DataFrame:
    return phase22._corrected_dt_replacements(dt)


def _aggregate_by_grid(candidates: pd.DataFrame, replacements: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stale_replacement_cols = [
        col for col in ["dt_replacement_rsrp", "dt_replacement_count"] if col in candidates.columns
    ]
    out = candidates.drop(columns=stale_replacement_cols).merge(
        replacements,
        on=["technology", "grid_id"],
        how="left",
    )
    lock = out["dt_replacement_rsrp"].notna()
    out["phase24_with_terrain_calibrated_no_lock_unclipped"] = (
        out["phase24_physical_with_terrain_rsrp"] + out["phase24_phase19_style_bias_db"]
    )
    out["phase24_with_terrain_calibrated_no_lock"] = valid_model_rsrp(
        out["phase24_with_terrain_calibrated_no_lock_unclipped"]
    )
    out["phase24_with_terrain_calibrated_rsrp"] = out["phase24_with_terrain_calibrated_no_lock"].where(
        ~lock, out["dt_replacement_rsrp"]
    )

    agg = out.groupby(["technology", "grid_id"], dropna=False).agg(
        {
            "phase22_with_terrain_calibrated_rsrp": ["max", "mean"],
            "phase22_physical_with_terrain_rsrp": ["max", "mean"],
            "phase24_physical_with_terrain_rsrp": ["max", "mean"],
            "phase24_with_terrain_calibrated_rsrp": ["max", "mean"],
            "phase24_building_clutter_correction_db": ["mean"],
            "phase24_correction_delta_db": ["mean"],
            "phase24_proxy_clutter_suppressed": ["mean"],
            "terrain_diffraction_loss_db": ["mean", "max"],
        }
    )
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    agg = agg.rename(
        columns={
            "phase22_with_terrain_calibrated_rsrp_max": "phase22_with_terrain_best_rsrp",
            "phase22_with_terrain_calibrated_rsrp_mean": "phase22_with_terrain_mean_rsrp",
            "phase22_physical_with_terrain_rsrp_max": "phase22_physical_with_terrain_best_rsrp",
            "phase22_physical_with_terrain_rsrp_mean": "phase22_physical_with_terrain_mean_rsrp",
            "phase24_physical_with_terrain_rsrp_max": "phase24_physical_with_terrain_best_rsrp",
            "phase24_physical_with_terrain_rsrp_mean": "phase24_physical_with_terrain_mean_rsrp",
            "phase24_with_terrain_calibrated_rsrp_max": "phase24_with_terrain_best_rsrp",
            "phase24_with_terrain_calibrated_rsrp_mean": "phase24_with_terrain_mean_rsrp",
            "phase24_proxy_clutter_suppressed_mean": "phase24_proxy_clutter_suppressed_share",
            "terrain_diffraction_loss_db_mean": "terrain_diffraction_loss_db_mean",
            "terrain_diffraction_loss_db_max": "terrain_diffraction_loss_db_max",
        }
    )
    grid = _read_frame(PROJECT_DIR / "cost231_phase9_gridanalytics_compatible" / "phase9_gridanalytics_compatible_grid_project210")
    all_tech_grid = pd.concat(
        [grid[["grid_id"]].assign(technology=technology) for technology in ["4G", "5G"]],
        ignore_index=True,
    )
    return all_tech_grid.merge(agg, on=["technology", "grid_id"], how="left"), out


def _plot_outputs(serving: pd.DataFrame, dt_tech: pd.DataFrame, technology: str) -> None:
    phase22._plot_cdf(
        [
            ("Phase22 same bias + terrain", serving["phase22_with_terrain_best_rsrp"], "#2563eb"),
            ("Phase24 physical + terrain", serving["phase24_physical_with_terrain_best_rsrp"], "#f97316"),
            ("Phase24 same residual + terrain", serving["phase24_with_terrain_best_rsrp"], "#16a34a"),
        ],
        f"Project 210 {technology}: Phase 22 vs Phase 24 full polygon",
        IMAGE_DIR / f"phase24_{technology.lower()}_full_polygon_cdf.png",
    )
    phase22._plot_cdf(
        [
            ("DT measured", dt_tech["rsrp_measured"], "#111827"),
            ("Phase22 physical + terrain", dt_tech["phase22_physical_with_terrain_rsrp"], "#2563eb"),
            ("Phase24 physical + terrain", dt_tech["phase24_physical_with_terrain_rsrp"], "#f97316"),
            ("Phase24 same residual + terrain", dt_tech["phase24_with_terrain_calibrated_rsrp"], "#16a34a"),
        ],
        f"Project 210 {technology}: Phase 24 DT comparison",
        IMAGE_DIR / f"phase24_{technology.lower()}_dt_cdf.png",
    )
    phase22._plot_cdf(
        [
            ("Phase22 physical abs error", (dt_tech["rsrp_measured"] - dt_tech["phase22_physical_with_terrain_rsrp"]).abs(), "#2563eb"),
            ("Phase24 physical abs error", (dt_tech["rsrp_measured"] - dt_tech["phase24_physical_with_terrain_rsrp"]).abs(), "#f97316"),
            ("Phase24 calibrated abs error", (dt_tech["rsrp_measured"] - dt_tech["phase24_with_terrain_calibrated_rsrp"]).abs(), "#16a34a"),
        ],
        f"Project 210 {technology}: Phase 24 DT absolute error",
        IMAGE_DIR / f"phase24_{technology.lower()}_dt_abs_error_cdf.png",
    )


def main() -> None:
    _ensure_dirs()

    candidates = _read_frame(PHASE22_DIR / "phase22_scored_candidates_project210")
    dt_scored = _read_frame(PHASE22_DIR / "phase22_dt_terrain_scored_project210")
    grid = _read_frame(PROJECT_DIR / "cost231_phase9_gridanalytics_compatible" / "phase9_gridanalytics_compatible_grid_project210")
    dt_source = (
        PHASE20_DIR / "phase9_dt_match_project210_corrected"
        if (PHASE20_DIR / "phase9_dt_match_project210_corrected.parquet").exists()
        else PROJECT_DIR / "cost231_phase9_gridanalytics_compatible" / "phase9_dt_match_project210"
    )
    dt_replacement_source = _read_frame(dt_source)

    print(f"[PHASE24] candidates={len(candidates)} dt={len(dt_scored)}")
    candidates = _apply_phase24_clutter_role_fix(candidates, "raw_cost231_rsrp")
    dt_scored = _apply_phase24_clutter_role_fix(dt_scored, "raw_cost231_at_dt_rsrp")

    dt_scored["dt_minus_phase24_no_terrain_physical_db"] = (
        dt_scored["rsrp_measured"] - dt_scored["phase24_physical_no_terrain_rsrp"]
    )
    dt_scored["dt_minus_phase24_with_terrain_physical_db"] = (
        dt_scored["rsrp_measured"] - dt_scored["phase24_physical_with_terrain_rsrp"]
    )
    bias = phase22._bias_table(dt_scored, "dt_minus_phase24_no_terrain_physical_db")
    bias["bias_source"] = "phase24_no_terrain_physical_residual"
    bias.to_csv(OUT_DIR / "phase24_phase19_style_bias_by_condition.csv", index=False)

    candidates = _attach_bias(candidates, bias, "phase24_phase19_style_bias_db")
    dt_scored = _attach_bias(dt_scored, bias, "phase24_phase19_style_bias_db")
    dt_scored["phase24_with_terrain_calibrated_rsrp_unclipped"] = (
        dt_scored["phase24_physical_with_terrain_rsrp"] + dt_scored["phase24_phase19_style_bias_db"]
    )
    dt_scored["phase24_with_terrain_calibrated_rsrp"] = valid_model_rsrp(
        dt_scored["phase24_with_terrain_calibrated_rsrp_unclipped"]
    )

    replacements = _corrected_dt_replacements(dt_replacement_source)
    grid_agg, scored_candidates = _aggregate_by_grid(candidates, replacements)
    grid_bounds = grid[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]].copy()

    summary = {}
    for technology in ["4G", "5G"]:
        serving = grid_agg[grid_agg["technology"].astype(str) == technology].merge(grid_bounds, on="grid_id", how="left")
        dt_tech = dt_scored[dt_scored["assigned_technology"].astype(str) == technology].copy()

        _save_frame(serving, OUT_DIR / f"phase24_serving_grid_{technology.lower()}_project210")
        _plot_outputs(serving, dt_tech, technology)

        delta = pd.to_numeric(serving["phase24_with_terrain_best_rsrp"] - serving["phase22_with_terrain_best_rsrp"], errors="coerce")
        role_share = (
            scored_candidates[scored_candidates["technology"].astype(str) == technology]["phase24_clutter_role"]
            .value_counts(normalize=True)
            .round(4)
            .to_dict()
        )
        summary[technology] = {
            "grid_rows": int(len(serving)),
            "dt_rows": int(len(dt_tech)),
            "mean_phase22_same_bias_terrain_best_rsrp": float(serving["phase22_with_terrain_best_rsrp"].mean()),
            "mean_phase24_physical_terrain_best_rsrp": float(serving["phase24_physical_with_terrain_best_rsrp"].mean()),
            "mean_phase24_same_residual_terrain_best_rsrp": float(serving["phase24_with_terrain_best_rsrp"].mean()),
            "phase24_minus_phase22_best_db": {
                "mean": float(delta.mean()),
                "p50": float(delta.quantile(0.50)),
                "p90": float(delta.quantile(0.90)),
                "min": float(delta.min()),
                "max": float(delta.max()),
            },
            "phase24_correction_delta_db_mean": float(serving["phase24_correction_delta_db_mean"].mean()),
            "phase24_proxy_clutter_suppressed_share": float(serving["phase24_proxy_clutter_suppressed_share"].mean()),
            "role_share": role_share,
            "dt_phase22_physical_with_terrain": _metrics(dt_tech["rsrp_measured"], dt_tech["phase22_physical_with_terrain_rsrp"]),
            "dt_phase24_physical_with_terrain": _metrics(dt_tech["rsrp_measured"], dt_tech["phase24_physical_with_terrain_rsrp"]),
            "dt_phase24_same_residual_with_terrain": _metrics(dt_tech["rsrp_measured"], dt_tech["phase24_with_terrain_calibrated_rsrp"]),
            "representative_bias_rows": int(len(bias[bias["technology"].astype(str) == technology])),
            "images": {
                "full_polygon_cdf": str((IMAGE_DIR / f"phase24_{technology.lower()}_full_polygon_cdf.png").relative_to(THIS_DIR)),
                "dt_cdf": str((IMAGE_DIR / f"phase24_{technology.lower()}_dt_cdf.png").relative_to(THIS_DIR)),
                "dt_abs_error_cdf": str((IMAGE_DIR / f"phase24_{technology.lower()}_dt_abs_error_cdf.png").relative_to(THIS_DIR)),
            },
        }
        print(f"[PHASE24] wrote {technology} serving rows={len(serving)}")

    keep_cols = [
        "technology",
        "grid_id",
        "strict_cell_key",
        "lat",
        "lon",
        "raw_cost231_rsrp",
        "clutter_class",
        "obstruction_branch",
        "building_geo_correction_db",
        "phase24_building_clutter_correction_db",
        "phase24_correction_delta_db",
        "phase24_clutter_role",
        "phase24_proxy_clutter_suppressed",
        "terrain_diffraction_loss_db",
        "phase24_physical_with_terrain_rsrp_unclipped",
        "phase22_physical_with_terrain_rsrp",
        "phase22_with_terrain_calibrated_rsrp",
        "phase24_physical_with_terrain_rsrp",
        "phase24_phase19_style_bias_db",
        "phase24_with_terrain_calibrated_rsrp",
    ]
    _save_frame(scored_candidates[[col for col in keep_cols if col in scored_candidates.columns]], OUT_DIR / "phase24_scored_candidates_project210")
    _save_frame(dt_scored, OUT_DIR / "phase24_dt_scored_project210")
    (OUT_DIR / "phase24_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("[PHASE24] summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
