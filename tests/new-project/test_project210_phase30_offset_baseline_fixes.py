"""
Phase 30: verification harness for the defects in the PRODUCTION
`lte_prediction_offset` baseline (the code the app calls "Phase 27").

Nothing in `ML/tools/` is modified and no production file is touched.  The base
of this phase is the *actual production output* of a completed offset job:

    temp/final_offset_<job_id>.csv

which already carries every per-candidate intermediate column (raw_cost231_rsrp,
building_obstruction_loss_db, terrain_diffraction_loss_db, obstruction_branch,
clutter_class, tech_band/clutter_terrain/sector_correction_db,
physical_rsrp_unclipped, dynamic_residual_db, final_rsrp).

Fixes, each applied as a separate, individually-measured layer:

  F1  index-alignment fix ..... rebuild dynamic_residual_db as the clean sum of
                                the three (already-fitted, uncorrupted) layer
                                corrections.  Production loses ~36% of pixels to
                                a pandas index-misalignment NaN in
                                phase27_calibration.apply_outdoor().
  F2  terrain Fresnel gate .... recompute the path-profile diffraction term from
                                the 5 m New Taipei DEM with (a) endpoint
                                exclusion, (b) a knife-edge significance gate,
                                (c) water / void -> 0 dB.  Removes the phantom
                                10-37 dB "diffraction" over flat ground and
                                river banks.
  F3  WorldCover clutter ...... real clutter class per pixel from ESA WorldCover
                                polygons; over Water suppress the road-fitted
                                dynamic residual and any terrain extra loss.
  F5  no-data != -140 ......... genuine no-service pixels stay NaN (flagged),
                                not collapsed onto the same -140 as real
                                low-signal pixels.

There is deliberately NO spatial smoothing: no phase 9-29 does any, and the
residual pixel-to-pixel steps after F1/F3 are real serving-cell / knife-edge
effects that must stay visible.

Serving output mirrors the other phases: per (technology, grid) it carries both
`*_best_rsrp` (serving cell) and `*_mean_rsrp` (frontend = mean of candidates)
for the production baseline and for each fix stage, plus grid bounds, so the
shared dashboard can consume it exactly like phase27/phase29.

Output: data/project_210_taiwan/cost231_phase30_offset_baseline_fixes/
"""
from __future__ import annotations

import glob
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
for _p in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_project210_phase22_terrain_diffraction_comparison as phase22
from phase_rsrp_guard import RSRP_NO_COVERAGE_DBM, valid_model_rsrp

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
OUT_DIR = PROJECT_DIR / "cost231_phase30_offset_baseline_fixes"
IMAGE_DIR = OUT_DIR / "images"

SITE_RAW = PROJECT_DIR / "raw_db" / "site_prediction_project_210_raw_all.parquet"
GRID_STEM = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible" / "phase9_gridanalytics_compatible_grid_project210"
WORLDCOVER_GEOJSON = BASELINE_DIR / "data" / "project_210_taiwan" / "clutter_tiles_esa_worldcover.geojson"
TEMP_DIR = ML_ROOT / "temp"

UE_HEIGHT_M = 1.5
INDOOR_BRANCH = "indoor"

# ---- F2 gate parameters ---------------------------------------------------
EDGE_SKIP_M = 25.0                 # ignore terrain within this of either terminal
V_SIGNIFICANCE_GATE = -0.40        # ITU-R P.526 knife-edge parameter floor
TERRAIN_LOSS_CAP_DB = 45.0

# ---- F3 clutter rules ---------------------------------------------------
WATER_RESIDUAL_FLOOR_DB = -3.0
OPEN_RESIDUAL_FLOOR_DB = -8.0
OPEN_CLASSES = {"Rural/Open", "Vegetation"}


# --------------------------------------------------------------------------- io
def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_job_csv(job_id: str | None) -> Path:
    if job_id:
        p = TEMP_DIR / f"final_offset_{job_id}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = sorted(
        glob.glob(str(TEMP_DIR / "final_offset_*.csv")), key=lambda f: Path(f).stat().st_mtime
    )
    for f in reversed(candidates):
        head = pd.read_csv(f, nrows=1)
        if {"physical_rsrp_unclipped", "tech_band_correction_db"}.issubset(head.columns):
            return Path(f)
    raise FileNotFoundError("no temp/final_offset_*.csv carries the Phase 26/27 columns")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------------- F2 terrain
def _load_sites() -> pd.DataFrame:
    s = pd.read_parquet(SITE_RAW)
    s = s.rename(columns={"latitude": "site_lat", "longitude": "site_lon", "height": "Height"})
    s["strict_cell_key"] = (
        s["site"].astype(str) + "_" + s["cell_id"].astype(str) + "_"
        + s["sector"].astype(str) + "_" + s["band"].astype(str) + "_Taiwan"
    )
    s["Height"] = _num(s["Height"]).fillna(30.0).clip(3.0, 120.0)
    return s[["strict_cell_key", "site_lat", "site_lon", "Height"]].drop_duplicates(
        "strict_cell_key", keep="first"
    ).reset_index(drop=True)


def _gated_terrain_loss_one(
    dem: "phase22.TerrainSampler",
    site_lat: float, site_lon: float, site_ground_m: float, antenna_height_m: float,
    rx_lat: float, rx_lon: float, freq_mhz: float,
) -> tuple[float, float, str]:
    distance_m = float(phase22._haversine_m(site_lat, site_lon, rx_lat, rx_lon))
    if distance_m < 2.0 * EDGE_SKIP_M:
        return 0.0, -999.0, "too_short"

    sample_count = int(np.clip(math.ceil(distance_m / 20.0), 9, 128))
    fractions = np.linspace(0.0, 1.0, sample_count)
    lats = site_lat + (rx_lat - site_lat) * fractions
    lons = site_lon + (rx_lon - site_lon) * fractions
    terrain = dem.sample(lats, lons)

    if float(np.isnan(terrain).mean()) > 0.5:
        return 0.0, -999.0, "void_or_water"
    terrain = pd.Series(terrain).interpolate(limit_direction="both").to_numpy(dtype=float)

    tx_alt = float(site_ground_m) + float(antenna_height_m)
    rx_ground = terrain[-1] if np.isfinite(terrain[-1]) else site_ground_m
    rx_alt = float(rx_ground) + UE_HEIGHT_M
    los_alt = tx_alt + (rx_alt - tx_alt) * fractions
    clearance = terrain - los_alt

    along_m = distance_m * fractions
    inner_mask = (along_m >= EDGE_SKIP_M) & (along_m <= distance_m - EDGE_SKIP_M)
    if not inner_mask.any() or not np.isfinite(clearance[inner_mask]).any():
        return 0.0, -999.0, "no_inner_sample"

    inner_idx = int(np.nanargmax(np.where(inner_mask, clearance, -np.inf)))
    h_m = float(clearance[inner_idx])
    d1 = max(float(along_m[inner_idx]), 1.0)
    d2 = max(distance_m - d1, 1.0)
    wavelength_m = 300.0 / max(float(freq_mhz), 1.0)
    v = h_m * math.sqrt(2.0 * (d1 + d2) / (wavelength_m * d1 * d2))

    if v <= V_SIGNIFICANCE_GATE:
        return 0.0, v, "below_gate"

    loss_at_zero = 6.9 + 20.0 * math.log10(math.sqrt(0.1 ** 2 + 1.0) - 0.1)
    if v < 0.0:
        loss_db = (v - V_SIGNIFICANCE_GATE) / (0.0 - V_SIGNIFICANCE_GATE) * loss_at_zero
    else:
        loss_db = 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return float(np.clip(loss_db, 0.0, TERRAIN_LOSS_CAP_DB)), v, "gated_loss"


def _recompute_terrain(base: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    dem = phase22.TerrainSampler(phase22.DEM_PATH)
    site_lookup = sites.set_index("strict_cell_key")
    ground_cache: dict[str, float] = {}

    loss = np.zeros(len(base), dtype=float)
    knife_v = np.full(len(base), -999.0, dtype=float)
    reason = np.array(["unscored"] * len(base), dtype=object)
    freq = _num(base["serving_frequency_mhz"]).fillna(1800.0).to_numpy()
    rx_lat = _num(base["lat"]).to_numpy()
    rx_lon = _num(base["lon"]).to_numpy()

    groups = base.groupby("strict_cell_key", dropna=False).indices
    total = len(groups)
    for gi, (key, idx) in enumerate(groups.items()):
        if key not in site_lookup.index:
            reason[idx] = "no_site"
            continue
        srow = site_lookup.loc[key]
        if isinstance(srow, pd.DataFrame):
            srow = srow.iloc[0]
        s_lat, s_lon, s_h = float(srow["site_lat"]), float(srow["site_lon"]), float(srow["Height"])
        if key not in ground_cache:
            g = float(dem.sample(np.array([s_lat]), np.array([s_lon]))[0])
            ground_cache[key] = g if np.isfinite(g) else 0.0
        s_ground = ground_cache[key]
        for j in idx:
            loss[j], knife_v[j], reason[j] = _gated_terrain_loss_one(
                dem, s_lat, s_lon, s_ground, s_h, float(rx_lat[j]), float(rx_lon[j]), float(freq[j])
            )
        if (gi + 1) % 200 == 0 or gi + 1 == total:
            print(f"[PHASE30][F2] terrain cells {gi + 1}/{total}", flush=True)
    dem.close()

    out = base.copy()
    out["f2_terrain_diffraction_loss_db"] = loss
    out["f2_terrain_knife_edge_v"] = knife_v
    out["f2_terrain_reason"] = reason
    return out


# --------------------------------------------------------------- F3 clutter
def _worldcover_class(base: pd.DataFrame) -> pd.Series:
    import shapely
    from shapely.geometry import shape
    from shapely.strtree import STRtree

    payload = json.loads(WORLDCOVER_GEOJSON.read_text(encoding="utf-8"))
    geoms = [shape(f["geometry"]) for f in payload["features"]]
    classes = np.array(
        [str(f["properties"].get("clutter_class", "Unknown")) for f in payload["features"]], dtype=object
    )
    tree = STRtree(geoms)

    centroids = base.drop_duplicates("grid_id")[["grid_id", "lat", "lon"]].copy()
    pts = shapely.points(_num(centroids["lon"]).to_numpy(), _num(centroids["lat"]).to_numpy())
    hit = np.array([None] * len(centroids), dtype=object)
    for i, pt in enumerate(pts):
        for c in tree.query(pt):
            if geoms[c].covers(pt):
                hit[i] = classes[c]
                break
    centroids["clutter_class_wc"] = hit
    return base.merge(centroids[["grid_id", "clutter_class_wc"]], on="grid_id", how="left")["clutter_class_wc"]


# --------------------------------------------------------------- aggregation
def _grid_bounds() -> pd.DataFrame:
    df = phase22._read_frame(GRID_STEM)
    return df[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]]


STAGE_COLS = {
    "phase27_prod": "p27_display_rsrp",   # production baseline, as the app shows it
    "f1": "f1_display_rsrp",
    "f2": "f2_display_rsrp",
    "phase30": "f3_display_rsrp",          # full Phase 30 = F1 + F2 + F3
}


def _aggregate(scored: pd.DataFrame) -> pd.DataFrame:
    """Per (technology, grid): serving-cell (best) and frontend (mean of candidates)."""
    work = scored.copy()
    work["_raw"] = _num(work["raw_cost231_rsrp"])
    best_rows = work.sort_values("_raw").groupby(["technology", "grid_id"], dropna=False).tail(1)
    env = best_rows[["technology", "grid_id", "obstruction_branch", "clutter_class_wc"]].copy()
    env["serving_environment"] = np.where(
        env["obstruction_branch"].astype(str) == INDOOR_BRANCH, "indoor", "outdoor"
    )

    agg_spec = {}
    for stage, col in STAGE_COLS.items():
        agg_spec[f"{stage}_best_rsrp"] = (col, "max")
        agg_spec[f"{stage}_mean_rsrp"] = (col, "mean")
    agg_spec["f2_terrain_loss_db_mean"] = ("f2_terrain_diffraction_loss_db", "mean")
    agg_spec["prod_terrain_loss_db_mean"] = ("terrain_diffraction_loss_db", "mean")
    agg_spec["prod_to_phase30_delta_db_mean"] = ("_prod_to_p30_delta", "mean")

    agg = work.groupby(["technology", "grid_id"], dropna=False).agg(**agg_spec).reset_index()
    agg = agg.merge(
        env[["technology", "grid_id", "serving_environment", "clutter_class_wc"]],
        on=["technology", "grid_id"], how="left",
    )
    return agg.merge(_grid_bounds(), on="grid_id", how="left")


# --------------------------------------------------------------------- main
def main(job_id: str | None = None) -> None:
    _ensure_dirs()
    job_csv = _resolve_job_csv(job_id)
    print(f"[PHASE30] base production job = {job_csv.name}", flush=True)
    base = pd.read_csv(job_csv, low_memory=False)
    base["technology"] = base["Technology"].astype(str)
    is_indoor = base["obstruction_branch"].astype(str).eq(INDOOR_BRANCH)

    raw = _num(base["raw_cost231_rsrp"])
    building = _num(base["building_obstruction_loss_db"]).fillna(0.0)
    terr_prod = _num(base["terrain_diffraction_loss_db"]).fillna(0.0)
    phys_prod = _num(base["physical_rsrp_unclipped"])
    tb = _num(base["tech_band_correction_db"]).fillna(0.0)
    ct = _num(base["clutter_terrain_correction_db"]).fillna(0.0)
    sec = _num(base["sector_correction_db"]).fillna(0.0)

    # -------- production baseline (as the app shows it) --------
    base["p27_dynamic_residual_db"] = _num(base["dynamic_residual_db"])
    base["p27_display_rsrp"] = _num(base["final_rsrp"])

    # -------- F1  index-alignment fix --------
    f1_resid = np.where(is_indoor, 0.0, tb + ct + sec)
    base["f1_dynamic_residual_db"] = f1_resid
    f1_unclipped = np.where(is_indoor, phys_prod, phys_prod + f1_resid)
    base["f1_display_rsrp"] = valid_model_rsrp(pd.Series(f1_unclipped, index=base.index))

    # -------- F2  terrain Fresnel gate --------
    sites = _load_sites()
    base = _recompute_terrain(base, sites)
    f2_terr = _num(base["f2_terrain_diffraction_loss_db"]).fillna(0.0)
    f2_phys = raw + building - f2_terr
    f2_unclipped = np.where(is_indoor, f2_phys, f2_phys + base["f1_dynamic_residual_db"])
    base["f2_display_rsrp"] = valid_model_rsrp(pd.Series(f2_unclipped, index=base.index))

    # -------- F3  WorldCover clutter / water rule --------
    base["clutter_class_wc"] = _worldcover_class(base)
    is_water = base["clutter_class_wc"].eq("Water").to_numpy()
    is_open = base["clutter_class_wc"].isin(OPEN_CLASSES).to_numpy()

    f3_resid = np.asarray(f1_resid, dtype=float).copy()
    f3_resid = np.where(is_water, np.clip(f3_resid, WATER_RESIDUAL_FLOOR_DB, None), f3_resid)
    f3_resid = np.where(is_open, np.clip(f3_resid, OPEN_RESIDUAL_FLOOR_DB, None), f3_resid)
    f3_resid = np.where(is_indoor.to_numpy(), 0.0, f3_resid)
    base["f3_dynamic_residual_db"] = f3_resid

    f3_terr = f2_terr.to_numpy(dtype=float).copy()
    f3_terr = np.where(is_water, 0.0, f3_terr)
    base["f3_terrain_diffraction_loss_db"] = f3_terr

    f3_phys = raw + building - f3_terr
    f3_unclipped = np.where(is_indoor.to_numpy(), f3_phys, f3_phys + f3_resid)
    base["f3_rsrp_unclipped"] = f3_unclipped
    base["f3_display_rsrp"] = valid_model_rsrp(pd.Series(f3_unclipped, index=base.index))

    # -------- F5  genuine no-data flag (never collapsed to -140) --------
    base["phase30_no_data"] = ~np.isfinite(pd.Series(f3_unclipped)) | (
        pd.Series(f3_unclipped) < RSRP_NO_COVERAGE_DBM
    )

    base["_prod_to_p30_delta"] = _num(base["f3_display_rsrp"]) - _num(base["p27_display_rsrp"])

    phase22._save_frame(base, OUT_DIR / "phase30_scored_candidates_project210")

    # -------- serving grid (best + mean), per technology --------
    serving_all = _aggregate(base)
    for tech in ["4G", "5G"]:
        phase22._save_frame(
            serving_all[serving_all["technology"].eq(tech)].copy(),
            OUT_DIR / f"phase30_serving_grid_{tech.lower()}_project210",
        )

    # -------- summary --------
    def _stats(v: pd.Series) -> dict:
        v = _num(v)
        f = v[np.isfinite(v)]
        return {
            "n": int(len(v)),
            "nan_frac": round(float(v.isna().mean()), 4),
            "median": round(float(f.median()), 2) if len(f) else None,
            "p10": round(float(f.quantile(0.10)), 2) if len(f) else None,
            "frac_lt_120": round(float((f < -120).mean()), 4) if len(f) else None,
        }

    cl = {
        "rows": int(len(base)),
        "p27_dynamic_residual_nan_frac": round(float(base["p27_dynamic_residual_db"].isna().mean()), 4),
        "f1_dynamic_residual_nan_frac": round(float(pd.Series(f1_resid).isna().mean()), 4),
        "p27_final_rsrp_nan_frac": round(float(base["p27_display_rsrp"].isna().mean()), 4),
        "phase30_no_data_frac": round(float(base["phase30_no_data"].mean()), 4),
        "worldcover_class_counts": {
            str(k): int(v) for k, v in base["clutter_class_wc"].value_counts(dropna=False).items()
        },
        "terrain_gt10db_frac_prod": round(float((terr_prod > 10).mean()), 4),
        "terrain_gt10db_frac_f2": round(float((f2_terr > 10).mean()), 4),
    }
    summary = {
        "base_job": job_csv.name,
        "note": "production lte_prediction_offset untouched; F1 index fix + F2 terrain gate + F3 clutter/water + F5 no-data flag; NO smoothing",
        "candidate_level": cl,
        "technology": {},
    }
    for tech in ["4G", "5G"]:
        sv = serving_all[serving_all["technology"].eq(tech)]
        water = sv["clutter_class_wc"].eq("Water")
        summary["technology"][tech] = {
            "serving_grids": int(len(sv)),
            "phase27_prod": _stats(sv["phase27_prod_best_rsrp"]),
            "f1": _stats(sv["f1_best_rsrp"]),
            "f2": _stats(sv["f2_best_rsrp"]),
            "phase30": _stats(sv["phase30_best_rsrp"]),
            "water_only": {
                "n": int(water.sum()),
                "prod_median": round(float(_num(sv.loc[water, "phase27_prod_best_rsrp"]).median()), 2) if water.any() else None,
                "phase30_median": round(float(_num(sv.loc[water, "phase30_best_rsrp"]).median()), 2) if water.any() else None,
            },
        }
        print(
            f"[PHASE30] {tech}  dyn-resid NaN {cl['p27_dynamic_residual_nan_frac']:.1%} -> {cl['f1_dynamic_residual_nan_frac']:.1%}"
            f"   serving median prod {summary['technology'][tech]['phase27_prod']['median']}"
            f" -> Phase30 {summary['technology'][tech]['phase30']['median']}",
            flush=True,
        )
        phase22._plot_cdf(
            [
                ("Production (Phase 27)", sv["phase27_prod_best_rsrp"], "#dc2626"),
                ("F1 index fix", sv["f1_best_rsrp"], "#f59e0b"),
                ("F2 + terrain gate", sv["f2_best_rsrp"], "#2563eb"),
                ("Phase 30 (F1+F2+F3)", sv["phase30_best_rsrp"], "#16a34a"),
            ],
            f"Project 210 {tech}: Phase 30 serving RSRP - production vs staged fixes",
            IMAGE_DIR / f"phase30_{tech.lower()}_serving_cdf.png",
        )

    (OUT_DIR / "phase30_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("[PHASE30] summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
