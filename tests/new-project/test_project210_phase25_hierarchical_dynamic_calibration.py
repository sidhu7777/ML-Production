"""
Phase 25: hierarchical dynamic residual calibration.

This is a test-only experiment. It does not replace predictions with DT.
It estimates residual corrections from DT training grids and validates on
held-out DT grids:

  physical Phase24 RSRP
    + technology/band residual
    + clutter/terrain/branch residual
    + sector residual, only when enough DT supports it
    + local residual field, only near supported training DT

The correction values are learned from the current project data. They are not
static clutter/building loss constants.
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
from scipy.spatial import cKDTree

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
for path in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import test_project210_phase17_full_polygon_geo_dt_comparison as phase17
import test_project210_phase22_terrain_diffraction_comparison as phase22
from phase_rsrp_guard import valid_model_rsrp


PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE24_DIR = PROJECT_DIR / "cost231_phase24_physical_clutter_role_fix"
OUT_DIR = PROJECT_DIR / "cost231_phase25_hierarchical_dynamic_calibration"
IMAGE_DIR = OUT_DIR / "images"

RSRP_MIN, RSRP_MAX = phase17.RSRP_MIN, phase17.RSRP_MAX

TECH_BAND_MIN_N = 80
CLUTTER_TERRAIN_MIN_N = 35
SECTOR_MIN_N = 30
LOCAL_MIN_NEIGHBORS = 5
LOCAL_K_NEIGHBORS = 16

TECH_BAND_SHRINK_N = 120.0
CLUTTER_TERRAIN_SHRINK_N = 60.0
SECTOR_SHRINK_N = 45.0
LOCAL_SHRINK_N = 8.0


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


def _add_common_features(df: pd.DataFrame, cell_col: str) -> pd.DataFrame:
    out = df.copy()
    out["technology"] = out["technology"].astype(str)
    out["band"] = out.get("band", pd.Series("UNKNOWN", index=out.index)).astype(str).fillna("UNKNOWN")
    out["cell_key"] = out[cell_col].astype(str).fillna("UNKNOWN")
    out["clutter_class"] = out["clutter_class"].astype("object").where(out["clutter_class"].notna(), "UNKNOWN")
    out["obstruction_branch"] = out["obstruction_branch"].astype(str).fillna("unknown")
    terrain = pd.to_numeric(out["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0)
    out["terrain_bucket"] = pd.cut(
        terrain,
        bins=[-np.inf, 0.5, 3.0, 10.0, np.inf],
        labels=["none", "low", "medium", "high"],
    ).astype(str)
    return out


def _split_dt_by_grid(dt: pd.DataFrame) -> pd.DataFrame:
    out = dt.copy()
    split_key = out["nearest_grid_id"].astype(str).fillna(out["dt_row_id"].astype(str))
    hashed = pd.util.hash_pandas_object(split_key, index=False).astype("uint64")
    out["phase25_split"] = np.where((hashed % 10) < 7, "train", "validation")
    return out


def _candidate_inputs() -> pd.DataFrame:
    candidates = _read_frame(PHASE24_DIR / "phase24_scored_candidates_project210")
    surface = pd.read_parquet(PHASE9_DIR / "phase9_directional_raw_corrected_surface_project210.parquet")
    meta_cols = [
        "technology",
        "grid_id",
        "strict_cell_key",
        "site",
        "sector",
        "band",
        "frequency_mhz",
        "distance_m",
        "azimuth_delta_deg",
    ]
    candidates = candidates.merge(
        surface[[col for col in meta_cols if col in surface.columns]],
        on=["technology", "grid_id", "strict_cell_key"],
        how="left",
    )
    candidates = _add_common_features(candidates, "strict_cell_key")
    candidates["phase24_no_lock_reference_rsrp_unclipped"] = (
        pd.to_numeric(candidates["phase24_physical_with_terrain_rsrp"], errors="coerce")
        + pd.to_numeric(candidates["phase24_phase19_style_bias_db"], errors="coerce").fillna(0.0)
    )
    candidates["phase24_no_lock_reference_rsrp"] = valid_model_rsrp(
        candidates["phase24_no_lock_reference_rsrp_unclipped"]
    )
    return candidates


def _dt_inputs() -> pd.DataFrame:
    dt = _read_frame(PHASE24_DIR / "phase24_dt_scored_project210")
    dt["technology"] = dt["assigned_technology"].astype(str)
    dt = _add_common_features(dt, "assigned_strict_cell_key")
    dt["phase24_no_lock_reference_rsrp_unclipped"] = (
        pd.to_numeric(dt["phase24_physical_with_terrain_rsrp"], errors="coerce")
        + pd.to_numeric(dt["phase24_phase19_style_bias_db"], errors="coerce").fillna(0.0)
    )
    dt["phase24_no_lock_reference_rsrp"] = valid_model_rsrp(dt["phase24_no_lock_reference_rsrp_unclipped"])
    return _split_dt_by_grid(dt)


def _fit_group_layer(
    train: pd.DataFrame,
    keys: list[str],
    predicted_col: str,
    layer_name: str,
    min_n: int,
    shrink_n: float,
) -> pd.DataFrame:
    work = train.copy()
    work["layer_residual_db"] = (
        pd.to_numeric(work["rsrp_measured"], errors="coerce")
        - pd.to_numeric(work[predicted_col], errors="coerce")
    )
    table = (
        work.dropna(subset=["layer_residual_db"])
        .groupby(keys, dropna=False)
        .agg(n_train=("layer_residual_db", "size"), median_residual_db=("layer_residual_db", "median"))
        .reset_index()
    )
    table = table[table["n_train"] >= min_n].copy()
    table[f"{layer_name}_shrink_factor"] = table["n_train"] / (table["n_train"] + shrink_n)
    table[f"{layer_name}_correction_db"] = (
        table["median_residual_db"] * table[f"{layer_name}_shrink_factor"]
    )
    table["layer"] = layer_name
    return table


def _attach_group_layer(df: pd.DataFrame, table: pd.DataFrame, keys: list[str], layer_name: str) -> pd.DataFrame:
    col = f"{layer_name}_correction_db"
    n_col = f"{layer_name}_n_train"
    if table.empty:
        out = df.copy()
        out[col] = 0.0
        out[n_col] = 0
        return out

    use_cols = keys + [col, "n_train"]
    out = df.merge(table[use_cols].rename(columns={"n_train": n_col}), on=keys, how="left")
    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out[n_col] = pd.to_numeric(out[n_col], errors="coerce").fillna(0).astype(int)
    return out


def _fit_group_hierarchy(train: pd.DataFrame) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    working = train.copy()
    working["phase25_group_pred_rsrp"] = pd.to_numeric(
        working["phase24_physical_with_terrain_rsrp"], errors="coerce"
    )
    layers: list[pd.DataFrame] = []
    layer_specs = [
        ("tech_band", ["technology", "band"], TECH_BAND_MIN_N, TECH_BAND_SHRINK_N),
        (
            "clutter_terrain",
            ["technology", "band", "clutter_class", "obstruction_branch", "terrain_bucket"],
            CLUTTER_TERRAIN_MIN_N,
            CLUTTER_TERRAIN_SHRINK_N,
        ),
        ("sector", ["technology", "band", "cell_key"], SECTOR_MIN_N, SECTOR_SHRINK_N),
    ]
    for layer_name, keys, min_n, shrink_n in layer_specs:
        table = _fit_group_layer(working, keys, "phase25_group_pred_rsrp", layer_name, min_n, shrink_n)
        working = _attach_group_layer(working, table, keys, layer_name)
        working["phase25_group_pred_rsrp"] = valid_model_rsrp(
            working["phase25_group_pred_rsrp"] + working[f"{layer_name}_correction_db"]
        )
        layers.append(table)
    return working, layers


def _apply_group_hierarchy(df: pd.DataFrame, layers: list[pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    out["phase25_group_pred_rsrp"] = pd.to_numeric(out["phase24_physical_with_terrain_rsrp"], errors="coerce")
    keys_by_layer = {
        "tech_band": ["technology", "band"],
        "clutter_terrain": ["technology", "band", "clutter_class", "obstruction_branch", "terrain_bucket"],
        "sector": ["technology", "band", "cell_key"],
    }
    for table in layers:
        if table.empty:
            continue
        layer_name = str(table["layer"].iloc[0])
        out = _attach_group_layer(out, table, keys_by_layer[layer_name], layer_name)
        out["phase25_group_pred_rsrp"] = valid_model_rsrp(
            out["phase25_group_pred_rsrp"] + out[f"{layer_name}_correction_db"]
        )
    for layer_name in keys_by_layer:
        col = f"{layer_name}_correction_db"
        n_col = f"{layer_name}_n_train"
        if col not in out.columns:
            out[col] = 0.0
        if n_col not in out.columns:
            out[n_col] = 0
    return out


def _project_xy(df: pd.DataFrame, lat0: float) -> np.ndarray:
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(dtype=float)
    x = lon * 111_320.0 * math.cos(math.radians(lat0))
    y = lat * 110_540.0
    return np.column_stack([x, y])


def _local_radius_m(train_tech: pd.DataFrame, lat0: float) -> float:
    if len(train_tech) < LOCAL_MIN_NEIGHBORS + 1:
        return 0.0
    xy = _project_xy(train_tech, lat0)
    tree = cKDTree(xy)
    distances, _ = tree.query(xy, k=min(2, len(train_tech)))
    nearest = distances[:, 1] if distances.ndim == 2 and distances.shape[1] > 1 else distances
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if nearest.size == 0:
        return 0.0
    return float(max(np.quantile(nearest, 0.75) * 3.0, 75.0))


def _fit_local_models(train_group_scored: pd.DataFrame) -> dict[str, dict]:
    models: dict[str, dict] = {}
    work = train_group_scored.copy()
    work["local_residual_db"] = (
        pd.to_numeric(work["rsrp_measured"], errors="coerce")
        - pd.to_numeric(work["phase25_group_pred_rsrp"], errors="coerce")
    )
    for tech, group in work.dropna(subset=["lat", "lon", "local_residual_db"]).groupby("technology"):
        if len(group) < LOCAL_MIN_NEIGHBORS:
            continue
        lat0 = float(pd.to_numeric(group["lat"], errors="coerce").mean())
        xy = _project_xy(group, lat0)
        radius_m = _local_radius_m(group, lat0)
        if radius_m <= 0:
            continue
        models[str(tech)] = {
            "lat0": lat0,
            "tree": cKDTree(xy),
            "residuals": group["local_residual_db"].to_numpy(dtype=float),
            "radius_m": radius_m,
            "n_train": int(len(group)),
        }
    return models


def _apply_local_model(df: pd.DataFrame, models: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()
    out["local_residual_correction_db"] = 0.0
    out["local_residual_support_n"] = 0
    out["local_residual_radius_m"] = 0.0
    for tech, idx in out.groupby("technology").groups.items():
        model = models.get(str(tech))
        if model is None:
            continue
        sub = out.loc[idx]
        points = _project_xy(sub, model["lat0"])
        k = min(LOCAL_K_NEIGHBORS, model["n_train"])
        distances, indices = model["tree"].query(points, k=k, distance_upper_bound=model["radius_m"])
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        valid = np.isfinite(distances) & (indices < model["n_train"])
        support = valid.sum(axis=1)
        corrections = np.zeros(len(sub), dtype=float)
        enough = support >= LOCAL_MIN_NEIGHBORS
        for row_i in np.where(enough)[0]:
            row_valid = valid[row_i]
            d = distances[row_i, row_valid]
            residual = model["residuals"][indices[row_i, row_valid]]
            weights = 1.0 / np.maximum(d, 5.0)
            local = float(np.average(residual, weights=weights))
            shrink = float(support[row_i] / (support[row_i] + LOCAL_SHRINK_N))
            corrections[row_i] = local * shrink
        out.loc[idx, "local_residual_correction_db"] = corrections
        out.loc[idx, "local_residual_support_n"] = support.astype(int)
        out.loc[idx, "local_residual_radius_m"] = float(model["radius_m"])
    return out


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["phase25_total_dynamic_correction_db"] = (
        out["tech_band_correction_db"]
        + out["clutter_terrain_correction_db"]
        + out["sector_correction_db"]
        + out["local_residual_correction_db"]
    )
    out["phase25_dynamic_rsrp_unclipped"] = (
        pd.to_numeric(out["phase24_physical_with_terrain_rsrp"], errors="coerce")
        + out["phase25_total_dynamic_correction_db"]
    )
    out["phase25_dynamic_rsrp"] = valid_model_rsrp(out["phase25_dynamic_rsrp_unclipped"])
    confidence = np.full(len(out), 0.30, dtype=float)
    confidence += np.where(out["tech_band_n_train"] > 0, 0.15, 0.0)
    confidence += np.where(out["clutter_terrain_n_train"] > 0, 0.15, 0.0)
    confidence += np.where(out["sector_n_train"] > 0, 0.20, 0.0)
    confidence += np.where(out["local_residual_support_n"] >= LOCAL_MIN_NEIGHBORS, 0.20, 0.0)
    out["phase25_confidence"] = np.clip(confidence, 0.0, 1.0)
    return out


def _aggregate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    agg = candidates.groupby(["technology", "grid_id"], dropna=False).agg(
        phase24_no_lock_best_rsrp=("phase24_no_lock_reference_rsrp", "max"),
        phase24_no_lock_mean_rsrp=("phase24_no_lock_reference_rsrp", "mean"),
        phase25_dynamic_best_rsrp=("phase25_dynamic_rsrp", "max"),
        phase25_dynamic_mean_rsrp=("phase25_dynamic_rsrp", "mean"),
        phase25_total_dynamic_correction_db_mean=("phase25_total_dynamic_correction_db", "mean"),
        tech_band_correction_db_mean=("tech_band_correction_db", "mean"),
        clutter_terrain_correction_db_mean=("clutter_terrain_correction_db", "mean"),
        sector_correction_db_mean=("sector_correction_db", "mean"),
        local_residual_correction_db_mean=("local_residual_correction_db", "mean"),
        local_residual_support_n_mean=("local_residual_support_n", "mean"),
        phase25_confidence_mean=("phase25_confidence", "mean"),
    ).reset_index()
    grid = _read_frame(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210")
    grid_bounds = grid[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]].copy()
    all_tech_grid = pd.concat(
        [grid_bounds[["grid_id"]].assign(technology=technology) for technology in ["4G", "5G"]],
        ignore_index=True,
    )
    return all_tech_grid.merge(agg, on=["technology", "grid_id"], how="left").merge(grid_bounds, on="grid_id", how="left")


def _plot_outputs(serving: pd.DataFrame, validation: pd.DataFrame, technology: str) -> None:
    phase22._plot_cdf(
        [
            ("Phase24 no-lock reference", serving["phase24_no_lock_best_rsrp"], "#2563eb"),
            ("Phase25 dynamic calibrated", serving["phase25_dynamic_best_rsrp"], "#16a34a"),
        ],
        f"Project 210 {technology}: Phase 24 vs Phase 25 full polygon",
        IMAGE_DIR / f"phase25_{technology.lower()}_full_polygon_cdf.png",
    )
    phase22._plot_cdf(
        [
            ("DT measured", validation["rsrp_measured"], "#111827"),
            ("Phase24 no-lock reference", validation["phase24_no_lock_reference_rsrp"], "#2563eb"),
            ("Phase25 dynamic calibrated", validation["phase25_dynamic_rsrp"], "#16a34a"),
        ],
        f"Project 210 {technology}: held-out DT validation",
        IMAGE_DIR / f"phase25_{technology.lower()}_validation_dt_cdf.png",
    )
    phase22._plot_cdf(
        [
            ("Phase24 no-lock abs error", (validation["rsrp_measured"] - validation["phase24_no_lock_reference_rsrp"]).abs(), "#2563eb"),
            ("Phase25 dynamic abs error", (validation["rsrp_measured"] - validation["phase25_dynamic_rsrp"]).abs(), "#16a34a"),
        ],
        f"Project 210 {technology}: held-out DT absolute error",
        IMAGE_DIR / f"phase25_{technology.lower()}_validation_abs_error_cdf.png",
    )


def main() -> None:
    _ensure_dirs()
    dt = _dt_inputs()
    candidates = _candidate_inputs()
    train = dt[dt["phase25_split"] == "train"].copy()
    validation = dt[dt["phase25_split"] == "validation"].copy()
    print(f"[PHASE25] train_dt={len(train)} validation_dt={len(validation)} candidates={len(candidates)}")

    train_group_scored, layers = _fit_group_hierarchy(train)
    local_models = _fit_local_models(train_group_scored)

    validation_scored = _apply_group_hierarchy(validation, layers)
    validation_scored = _finalize(_apply_local_model(validation_scored, local_models))
    candidates_scored = _apply_group_hierarchy(candidates, layers)
    candidates_scored = _finalize(_apply_local_model(candidates_scored, local_models))

    _save_frame(validation_scored, OUT_DIR / "phase25_validation_dt_project210")
    _save_frame(candidates_scored, OUT_DIR / "phase25_scored_candidates_project210")

    layer_table = pd.concat(layers, ignore_index=True) if layers else pd.DataFrame()
    if not layer_table.empty:
        layer_table.to_csv(OUT_DIR / "phase25_group_corrections.csv", index=False)
    pd.DataFrame(
        [
            {"technology": tech, "n_train": model["n_train"], "local_radius_m": model["radius_m"]}
            for tech, model in local_models.items()
        ]
    ).to_csv(OUT_DIR / "phase25_local_model_summary.csv", index=False)

    serving_all = _aggregate_candidates(candidates_scored)
    summary = {
        "calibration_policy": "train_on_dt_grids_validate_on_held_out_dt_grids_no_dt_replacement",
        "layers": {
            "tech_band": {"min_n": TECH_BAND_MIN_N, "shrink_n": TECH_BAND_SHRINK_N},
            "clutter_terrain": {"min_n": CLUTTER_TERRAIN_MIN_N, "shrink_n": CLUTTER_TERRAIN_SHRINK_N},
            "sector": {"min_n": SECTOR_MIN_N, "shrink_n": SECTOR_SHRINK_N},
            "local": {"min_neighbors": LOCAL_MIN_NEIGHBORS, "k_neighbors": LOCAL_K_NEIGHBORS},
        },
        "technology": {},
    }

    for technology in ["4G", "5G"]:
        serving = serving_all[serving_all["technology"].astype(str) == technology].copy()
        vtech = validation_scored[validation_scored["technology"].astype(str) == technology].copy()
        _save_frame(serving, OUT_DIR / f"phase25_serving_grid_{technology.lower()}_project210")
        _plot_outputs(serving, vtech, technology)
        delta = serving["phase25_dynamic_best_rsrp"] - serving["phase24_no_lock_best_rsrp"]
        summary["technology"][technology] = {
            "grid_rows": int(len(serving)),
            "train_dt_rows": int((train["technology"].astype(str) == technology).sum()),
            "validation_dt_rows": int(len(vtech)),
            "phase24_no_lock_validation": _metrics(vtech["rsrp_measured"], vtech["phase24_no_lock_reference_rsrp"]),
            "phase25_dynamic_validation": _metrics(vtech["rsrp_measured"], vtech["phase25_dynamic_rsrp"]),
            "mean_phase24_no_lock_best_rsrp": float(serving["phase24_no_lock_best_rsrp"].mean()),
            "mean_phase25_dynamic_best_rsrp": float(serving["phase25_dynamic_best_rsrp"].mean()),
            "phase25_minus_phase24_best_db": {
                "mean": float(delta.mean()),
                "p50": float(delta.quantile(0.50)),
                "p90": float(delta.quantile(0.90)),
                "min": float(delta.min()),
                "max": float(delta.max()),
            },
            "mean_total_dynamic_correction_db": float(serving["phase25_total_dynamic_correction_db_mean"].mean()),
            "mean_local_support_n": float(serving["local_residual_support_n_mean"].mean()),
            "mean_confidence": float(serving["phase25_confidence_mean"].mean()),
            "images": {
                "full_polygon_cdf": str((IMAGE_DIR / f"phase25_{technology.lower()}_full_polygon_cdf.png").relative_to(THIS_DIR)),
                "validation_dt_cdf": str((IMAGE_DIR / f"phase25_{technology.lower()}_validation_dt_cdf.png").relative_to(THIS_DIR)),
                "validation_abs_error_cdf": str((IMAGE_DIR / f"phase25_{technology.lower()}_validation_abs_error_cdf.png").relative_to(THIS_DIR)),
            },
        }
        print(f"[PHASE25] wrote {technology} serving rows={len(serving)}")

    (OUT_DIR / "phase25_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("[PHASE25] summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
