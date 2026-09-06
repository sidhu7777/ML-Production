"""Phase 42: Project 210 1500 m PAP coverage through the existing RF pipeline.

Test-only. No production files are modified.

The only Phase 42 experiment knobs are:
  1. regenerate the Phase 9 candidate surface with a 1500 m radius;
  2. keep the real PAP antenna replacement used by Phase 36/39.

Everything after candidate generation follows the established path:
Phase 26 terrain/building obstruction -> Phase 36 physical/PAP replacement ->
Phase 39 scoring/calibration -> Phase 42 display columns.
"""
from __future__ import annotations

import json
import os
import re
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

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE42_RADIUS_M = 1500.0

# Import the Phase 9 generator with Project 210 / Taiwan / 1500 m settings.
os.environ["PROP_PROJECT_ID"] = "210"
os.environ["PROP_REGION"] = "taiwan"
os.environ["PROP_PROJECT_SLUG"] = "project_210_taiwan"
os.environ["PROP_PROJECT_DIR"] = str(PROJECT_DIR)
os.environ["PHASE9_COVERAGE_RADIUS_M"] = str(int(PHASE42_RADIUS_M))
os.environ["PHASE9_CANDIDATE_MODE"] = "safe_radius"
os.environ["PHASE9_ENABLE_OUT_OF_RADIUS_BACKFILL"] = "1"

import streamlit_project210_phase13_beam_check as phase13
import test_project196_cost231_phase9_gridanalytics_compatible as phase9
import test_project210_phase17_full_polygon_geo_dt_comparison as phase17
import test_project210_phase22_terrain_diffraction_comparison as phase22
import test_project210_phase25_hierarchical_dynamic_calibration as phase25
import test_project210_phase26_corrected_obstruction_profile as phase26
import test_project210_phase36_final as p36
import test_project210_phase38_earfcn_rematch as p38
import test_project210_phase39_equal_power_diagnostic as phase39
from phase_rsrp_guard import RSRP_MAX_DBM, RSRP_NO_COVERAGE_DBM, valid_model_rsrp

PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE26_DIR = PROJECT_DIR / "cost231_phase26_corrected_obstruction_profile"
OUT_DIR = PROJECT_DIR / "cost231_phase42_1500m_pap_sector_coverage"
IMAGE_DIR = OUT_DIR / "images"
GRID_RE = re.compile(r"R(\d+)C(\d+)")


def _grid_bounds() -> pd.DataFrame:
    grid = pd.read_parquet(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210.parquet")
    keep = ["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]
    out = grid[keep].drop_duplicates("grid_id").copy()
    out["grid_id"] = out["grid_id"].astype(str)
    return out


def _save_parquet(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(stem.with_suffix(".parquet"), index=False)


def _component_count(grid_ids: pd.Series) -> int:
    coords = set()
    for gid in grid_ids.dropna().astype(str):
        match = GRID_RE.search(gid)
        if match:
            coords.add((int(match.group(1)), int(match.group(2))))
    components = 0
    while coords:
        components += 1
        stack = [coords.pop()]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in coords:
                    coords.remove(neighbor)
                    stack.append(neighbor)
    return components


def _build_phase9_1500_surface() -> pd.DataFrame:
    site_df = phase9._prepare_site_rows(
        phase9._read_first_existing(
            [
                phase9.BASELINE_SCOPE / "site_identity_strict_cells_project210.csv",
                phase9.BASELINE_SCOPE / "site_identity_102_strict_cells.csv",
            ]
        )
    )
    grid_df = _grid_bounds().rename(columns={"center_lat": "center_lat", "center_lon": "center_lon"})
    surface, _stats = phase9._run_directional_surface(site_df, grid_df)
    surface["grid_id"] = surface["grid_id"].astype(str)
    surface["strict_cell_key"] = surface["strict_cell_key"].astype(str)
    surface = surface[pd.to_numeric(surface["distance_m"], errors="coerce") <= PHASE42_RADIUS_M].copy()
    surface["phase42_candidate_source"] = "phase9_safe_radius_1500m"
    return surface


def _score_phase26_from_surface(surface: pd.DataFrame) -> pd.DataFrame:
    identity = phase13.load_identity()
    clutter_gdf, buildings_gdf = phase17._load_clutter_and_buildings()
    dem = phase22.TerrainSampler(phase22.DEM_PATH)
    try:
        scored = phase26._score_with_phase26_obstruction(
            surface,
            identity,
            clutter_gdf,
            buildings_gdf,
            dem,
            key_col="strict_cell_key",
            raw_col="raw_cost231_rsrp",
        )
    finally:
        dem.close()
    scored = phase26._rename_phase22_cols(scored)
    scored["phase26_terrain_delta_db"] = (
        pd.to_numeric(scored["phase26_physical_with_terrain_rsrp"], errors="coerce")
        - pd.to_numeric(scored["phase26_physical_no_terrain_rsrp"], errors="coerce")
    )
    return scored


def _candidate_inputs_from_phase26(scored: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for tech in ("4G", "5G"):
        sub = scored[scored["technology"].astype(str).eq(tech)].copy()
        if sub.empty:
            continue
        raw = pd.to_numeric(sub["raw_cost231_rsrp_unclipped"], errors="coerce").to_numpy(float)
        frames.append(p36._build_physical(sub, tech, raw))
    out = pd.concat(frames, ignore_index=True)
    out = phase25._add_common_features(out, "strict_cell_key")
    out["phase24_no_lock_reference_rsrp_unclipped"] = pd.to_numeric(out[p36.BASE_UNCLIPPED], errors="coerce")
    out["phase24_no_lock_reference_rsrp"] = valid_model_rsrp(out["phase24_no_lock_reference_rsrp_unclipped"])
    out["phase42_pap_replacement_db"] = np.clip(
        p36.phase29._antenna_gain_delta(out.assign(technology=out["technology"].astype(str))),
        *p36.ANTENNA_DELTA_CLIP_DB,
    )
    out["phase42_pattern_source"] = "phase36_real_pap_replacement"
    out["phase42_candidate_source"] = out.get("phase42_candidate_source", "phase9_safe_radius_1500m")
    return out


def _phase39_dt_inputs() -> pd.DataFrame:
    raw_dt = pd.read_parquet(PHASE26_DIR / "phase26_dt_scored_project210.parquet")
    cand_raw = pd.read_parquet(PHASE26_DIR / "phase26_scored_candidates_project210.parquet")
    dt_rm = p38._rematch_4g(raw_dt, cand_raw)
    return p38._dt_inputs_from(dt_rm)


def _score_phase39(candidate_inputs: pd.DataFrame) -> pd.DataFrame:
    dt = phase39._apply_equal_power_assumptions(_phase39_dt_inputs(), "assigned_strict_cell_key")
    cand = phase39._apply_equal_power_assumptions(candidate_inputs, "strict_cell_key")
    train = dt[dt["phase25_split"].astype(str).eq("train")].copy()
    fit = train[
        (train["obstruction_branch"].astype(str) != "indoor")
        & (~train["p36_backlobe"].astype(bool))
        & (~train["p38_excluded"].astype(bool))
    ].copy()
    print(f"[P42] phase39 fit rows={len(fit)} candidate rows={len(cand)}")
    layers, local_models = p36._fit(fit)
    return phase39._copy_phase39_score_columns(p36._score(cand, layers, local_models))


def _apply_phase42_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    signal = pd.to_numeric(out["phase39_final_rsrp_unclipped"], errors="coerce")
    out["phase42_raw_rsrp_dbm"] = signal
    out["phase42_potential_rsrp"] = valid_model_rsrp(signal)
    out["phase42_coverage_eligible"] = out["phase42_potential_rsrp"].notna()
    out["phase42_is_serving_cell"] = False
    out["phase42_serving_rsrp"] = np.nan
    out["phase42_hide_reason"] = np.select(
        [signal.isna().to_numpy(), signal.lt(RSRP_NO_COVERAGE_DBM).to_numpy()],
        ["missing_rsrp", "below_-140_dbm_floor"],
        default="usable_signal",
    )
    usable = out.dropna(subset=["grid_id", "phase42_potential_rsrp"]).copy()
    if not usable.empty:
        usable["_sort_value"] = pd.to_numeric(usable["phase42_raw_rsrp_dbm"], errors="coerce")
        best_idx = usable.sort_values("_sort_value").groupby(["technology", "grid_id"], dropna=False).tail(1).index
        out.loc[best_idx, "phase42_is_serving_cell"] = True
        out.loc[best_idx, "phase42_serving_rsrp"] = out.loc[best_idx, "phase42_potential_rsrp"]
        out.loc[out["phase42_is_serving_cell"], "phase42_hide_reason"] = "serving_cell"
        out.loc[out["phase42_coverage_eligible"] & ~out["phase42_is_serving_cell"], "phase42_hide_reason"] = (
            "usable_but_not_serving"
        )
    return out


def _best_per_grid(frame: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    work = frame.dropna(subset=["grid_id", value_col]).copy()
    if work.empty:
        return pd.DataFrame(columns=["technology", "grid_id"])
    work["_sort_value"] = pd.to_numeric(work[value_col], errors="coerce")
    keep_cols = [
        "technology", "grid_id", "strict_cell_key", "site", "sector", "band",
        "azimuth_delta_deg", "distance_m", "obstruction_branch", "clutter_class",
        "terrain_diffraction_loss_db", "phase42_pattern_source", value_col,
    ]
    best = work.sort_values("_sort_value").groupby(["technology", "grid_id"], dropna=False).tail(1)
    best = best[[col for col in keep_cols if col in best.columns]].rename(
        columns={
            "strict_cell_key": f"{prefix}_serving_cell",
            "site": f"{prefix}_site",
            "sector": f"{prefix}_sector",
            "band": f"{prefix}_band",
            "azimuth_delta_deg": f"{prefix}_azimuth_delta_deg",
            "distance_m": f"{prefix}_distance_m",
            "obstruction_branch": f"{prefix}_obstruction_branch",
            "clutter_class": f"{prefix}_clutter_class",
            "terrain_diffraction_loss_db": f"{prefix}_terrain_diffraction_loss_db",
            "phase42_pattern_source": f"{prefix}_pattern_source",
            value_col: f"{prefix}_rsrp",
        }
    )
    return best.reset_index(drop=True)


def _serving_grid(frame: pd.DataFrame) -> pd.DataFrame:
    bounds = _grid_bounds()
    full = pd.concat([bounds[["grid_id"]].assign(technology=tech) for tech in ("4G", "5G")], ignore_index=True)
    visible = _best_per_grid(frame, "phase42_serving_rsrp", "phase42")
    raw = _best_per_grid(frame, "phase42_raw_rsrp_dbm", "phase42_raw")
    out = full.merge(raw, on=["technology", "grid_id"], how="left").merge(
        visible, on=["technology", "grid_id"], how="left"
    )
    out["phase42_production_rsrp"] = (
        pd.to_numeric(out.get("phase42_raw_rsrp"), errors="coerce")
        .fillna(RSRP_NO_COVERAGE_DBM)
        .clip(lower=RSRP_NO_COVERAGE_DBM, upper=RSRP_MAX_DBM)
    )
    counts = (
        frame.groupby(["technology", "grid_id"], dropna=False)
        .agg(
            phase42_candidate_cells=("strict_cell_key", "nunique"),
            phase42_usable_candidate_cells=("phase42_coverage_eligible", "sum"),
            phase42_serving_candidate_cells=("phase42_is_serving_cell", "sum"),
        )
        .reset_index()
    )
    return out.merge(counts, on=["technology", "grid_id"], how="left").merge(bounds, on="grid_id", how="left")


def _sector_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(["technology", "strict_cell_key"], dropna=False):
        technology, cell = key
        potential = pd.to_numeric(group["phase42_potential_rsrp"], errors="coerce")
        serving = pd.to_numeric(group["phase42_serving_rsrp"], errors="coerce")
        raw = pd.to_numeric(group["phase42_raw_rsrp_dbm"], errors="coerce")
        rows.append(
            {
                "technology": technology,
                "strict_cell_key": cell,
                "site": group.get("site", pd.Series([""])).astype(str).iloc[0],
                "sector": group.get("sector", pd.Series([""])).astype(str).iloc[0],
                "band": group.get("band", pd.Series([""])).astype(str).iloc[0],
                "candidate_rows": int(len(group)),
                "usable_signal_rows": int(potential.notna().sum()),
                "serving_rows": int(serving.notna().sum()),
                "serving_pct": round(float(serving.notna().mean() * 100.0), 2) if len(group) else 0.0,
                "full_grid_components": int(_component_count(group["grid_id"])),
                "potential_components": int(_component_count(group.loc[potential.notna(), "grid_id"])),
                "mean_terrain_loss_db": round(float(pd.to_numeric(group["terrain_diffraction_loss_db"], errors="coerce").mean()), 2),
                "median_pap_replacement_db": round(float(pd.to_numeric(group["phase42_pap_replacement_db"], errors="coerce").median()), 2),
                "raw_median_dbm": round(float(raw.median()), 2) if raw.notna().any() else None,
                "potential_median_dbm": round(float(potential.median()), 2) if potential.notna().any() else None,
                "serving_median_dbm": round(float(serving.median()), 2) if serving.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def _cdf_inputs(serving: pd.DataFrame, candidates: pd.DataFrame, technology: str):
    grid = serving[serving["technology"].astype(str).eq(technology)].copy()
    cand = candidates[candidates["technology"].astype(str).eq(technology)].copy()
    return [
        ("1 - Phase 42 production capped/backfilled", grid["phase42_production_rsrp"], "#64748b"),
        ("2 - Phase 42 serving coverage", grid["phase42_rsrp"], "#16a34a"),
        ("3 - Phase 42 all usable candidate rows", cand["phase42_potential_rsrp"], "#2563eb"),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("[P42] building Phase9 1500m candidate surface")
    surface = _build_phase9_1500_surface()
    print(f"[P42] phase9_1500_surface rows={len(surface)} cells={surface['strict_cell_key'].nunique()}")
    surface_for_scoring = phase26._rf_plausible_candidates(surface)
    surface_for_scoring["phase42_candidate_source"] = "phase9_1500m_rf_plausible_phase26_filter"
    print(
        "[P42] phase26 RF-plausible candidate scope "
        f"rows={len(surface_for_scoring)} cells={surface_for_scoring['strict_cell_key'].nunique()}"
    )

    print("[P42] scoring Phase26 terrain/building obstruction on Phase42 surface")
    phase26_scored = _score_phase26_from_surface(surface_for_scoring)
    print(f"[P42] phase26_scored rows={len(phase26_scored)}")

    print("[P42] applying Phase36/39 production-style scoring")
    candidate_inputs = _candidate_inputs_from_phase26(phase26_scored)
    scored = _apply_phase42_columns(_score_phase39(candidate_inputs))
    serving = _serving_grid(scored)
    sector_summary = _sector_summary(scored)

    _save_parquet(scored, OUT_DIR / "phase42_cell_coverage_project210")
    _save_parquet(serving, OUT_DIR / "phase42_serving_grid_project210")
    sector_summary.to_csv(OUT_DIR / "phase42_sector_summary_project210.csv", index=False)

    for technology in ("4G", "5G"):
        tech_cells = scored[scored["technology"].astype(str).eq(technology)].copy()
        tech_serving = serving[serving["technology"].astype(str).eq(technology)].copy()
        _save_parquet(tech_cells, OUT_DIR / f"phase42_cell_coverage_{technology.lower()}_project210")
        _save_parquet(tech_serving, OUT_DIR / f"phase42_serving_grid_{technology.lower()}_project210")
        tech_serving.to_csv(OUT_DIR / f"phase42_serving_grid_{technology.lower()}_project210.csv", index=False)
        phase22._plot_cdf(
            _cdf_inputs(serving, scored, technology),
            f"Project 210 {technology}: Phase 42 1500 m PAP through production logic",
            IMAGE_DIR / f"phase42_{technology.lower()}_coverage_cdf.png",
        )

    summary = {
        "scope": "Phase 42 test-only. Phase9 safe-radius candidates regenerated at 1500 m, then Phase26 terrain/building and Phase36/39 scoring are reused.",
        "production_code_touched": False,
        "sector_radius_m": PHASE42_RADIUS_M,
        "pap_policy": "Phase9 raw contains generic antenna. Phase36 replaces that generic antenna with the real PAP pattern via PAP-minus-generic delta; this is the production-equivalent PAP insertion point.",
        "terrain_policy": "Phase26 terrain_diffraction_loss_db and corrected building obstruction are recomputed on the Phase42 candidate surface.",
        "technology": {},
    }
    for technology in ("4G", "5G"):
        tech_grid = serving[serving["technology"].astype(str).eq(technology)].copy()
        tech_cells = scored[scored["technology"].astype(str).eq(technology)].copy()
        visible_grid = pd.to_numeric(tech_grid["phase42_rsrp"], errors="coerce").notna()
        usable_cell = pd.to_numeric(tech_cells["phase42_potential_rsrp"], errors="coerce").notna()
        source_counts = tech_cells["phase42_candidate_source"].astype(str).value_counts(dropna=False).to_dict()
        summary["technology"][technology] = {
            "grid_rows": int(len(tech_grid)),
            "serving_grid_rows": int(visible_grid.sum()),
            "serving_grid_pct": round(float(visible_grid.mean() * 100.0), 2) if len(tech_grid) else 0.0,
            "cell_candidate_rows": int(len(tech_cells)),
            "usable_cell_candidate_rows": int(usable_cell.sum()),
            "usable_cell_candidate_pct": round(float(usable_cell.mean() * 100.0), 2) if len(tech_cells) else 0.0,
            "distinct_cells": int(tech_cells["strict_cell_key"].nunique(dropna=True)),
            "candidate_sources": source_counts,
            "max_full_grid_components": int(sector_summary.loc[
                sector_summary["technology"].astype(str).eq(technology), "full_grid_components"
            ].max()),
            "max_potential_components": int(sector_summary.loc[
                sector_summary["technology"].astype(str).eq(technology), "potential_components"
            ].max()),
            "median_terrain_loss_db": round(float(pd.to_numeric(tech_cells["terrain_diffraction_loss_db"], errors="coerce").median()), 2),
            "median_pap_replacement_db": round(float(pd.to_numeric(tech_cells["phase42_pap_replacement_db"], errors="coerce").median()), 2),
            "production_median_dbm": round(float(pd.to_numeric(tech_grid["phase42_production_rsrp"], errors="coerce").median()), 2),
        }
    (OUT_DIR / "phase42_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
