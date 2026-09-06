"""Phase 41: production-style sector coverage for Project 210.

Phase 39/40 calculate RF values with the real antenna pattern already applied.
Phase 41 turns that RF surface into a client-facing sector coverage view:

* candidate RSRP remains auditable as `phase41_raw_rsrp_dbm`;
* potential sector signal above the floor is `phase41_potential_rsrp`;
* serving sector coverage is `phase41_serving_rsrp`;
* no lobe is removed by display logic; side/back lobes only count as coverage
  if that sector is the strongest serving cell at that grid.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import test_project210_phase22_terrain_diffraction_comparison as phase22
import test_project210_phase39_equal_power_diagnostic as phase39

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE39_DIR = PROJECT_DIR / "cost231_phase39_equal_power_diagnostic"
OUT_DIR = PROJECT_DIR / "cost231_phase41_pap_coverage_footprint"
IMAGE_DIR = OUT_DIR / "images"
RSRP_NO_COVERAGE_DBM = -140.0
RSRP_MAX_DBM = -44.0
SECTOR_COVERAGE_RADIUS_M = 500.0


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6_371_000.0 * np.arcsin(np.sqrt(a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def _azimuth_delta_deg(bearing, azimuth):
    return np.abs((bearing - float(azimuth) + 180.0) % 360.0 - 180.0)


def _generic_3gpp_gain(az_off_deg, elev_diff_deg, max_gain=18.0, h_bw=65.0, v_bw=6.0, a_max=30.0, sla_v=20.0):
    az_off = np.abs(np.asarray(az_off_deg, dtype=float))
    a_h = np.minimum(12.0 * (az_off / h_bw) ** 2, a_max)
    a_v = np.minimum(12.0 * (np.asarray(elev_diff_deg, dtype=float) / v_bw) ** 2, sla_v)
    return max_gain - np.minimum(a_h + a_v, a_max)


def _num(value, default: float) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) and np.isfinite(float(parsed)) else float(default)


def _cost231_raw_vector(row, lat_values, lon_values):
    freq = _num(row.get("frequency_mhz"), 1800.0)
    site_lat = _num(row.get("site_lat"), np.nan)
    site_lon = _num(row.get("site_lon"), np.nan)
    height = _num(row.get("Height"), 30.0)
    tx_power = _num(row.get("tx_power"), 46.0)
    azimuth = _num(row.get("azimuth_x", row.get("azimuth_y")), 0.0)
    etilt = _num(row.get("Etilt"), 3.0)
    mtilt = _num(row.get("Mtilt"), 0.0)

    distance_m = np.maximum(_haversine_m(site_lat, site_lon, lat_values, lon_values), 1.0)
    distance_km = distance_m / 1000.0
    ue_height = 1.5
    a_hm = (1.1 * np.log10(freq) - 0.7) * ue_height - (1.56 * np.log10(freq) - 0.8)
    base_pl = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(height) - a_hm + 3.0
    slope = 44.9 - 6.55 * np.log10(height)
    pathloss = base_pl + slope * np.log10(distance_km)
    bearing = _bearing_deg(site_lat, site_lon, lat_values, lon_values)
    az_delta = _azimuth_delta_deg(bearing, azimuth)
    elev_angle = np.degrees(np.arctan2(ue_height - height, distance_m))
    gain = _generic_3gpp_gain(az_delta, elev_angle + etilt + mtilt)
    return tx_power + gain - pathloss - 2.0, distance_m, bearing, az_delta


def _sector_table(candidates: pd.DataFrame) -> pd.DataFrame:
    first_cols = [
        "technology", "site", "sector_key", "sector", "band", "original_cell_id",
        "site_lat", "site_lon", "azimuth_x", "azimuth_y", "Etilt", "Mtilt",
        "Height", "tx_power", "frequency_mhz", "original_frequency_mhz",
    ]
    first_cols = [col for col in first_cols if col in candidates.columns]
    rows = []
    for cell, group in candidates.groupby("strict_cell_key", dropna=False):
        base = group.iloc[0][first_cols].to_dict()
        base["strict_cell_key"] = str(cell)
        base["phase41_existing_max_distance_m"] = float(pd.to_numeric(group["distance_m"], errors="coerce").max())
        rows.append(base)
    return pd.DataFrame(rows)


def _phase41_full_grid_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    bounds = _grid_bounds()
    grid_lat = pd.to_numeric(bounds["center_lat"], errors="coerce").to_numpy(float)
    grid_lon = pd.to_numeric(bounds["center_lon"], errors="coerce").to_numpy(float)
    exact_cols = [
        "strict_cell_key", "grid_id", "phase39_final_rsrp_unclipped",
        "phase39_final_rsrp", "phase39_equal_power_rsrp_unclipped",
    ]
    exact = candidates[[col for col in exact_cols if col in candidates.columns]].copy()
    exact["strict_cell_key"] = exact["strict_cell_key"].astype(str)
    exact["grid_id"] = exact["grid_id"].astype(str)

    frames = []
    for _, row in _sector_table(candidates).iterrows():
        raw, distance_m, bearing, az_delta = _cost231_raw_vector(row, grid_lat, grid_lon)
        in_radius = distance_m <= SECTOR_COVERAGE_RADIUS_M
        if not bool(in_radius.any()):
            continue
        cell = str(row["strict_cell_key"])
        frame = pd.DataFrame({
            "grid_id": bounds.loc[in_radius, "grid_id"].astype(str).to_numpy(),
            "lat": grid_lat[in_radius],
            "lon": grid_lon[in_radius],
            "center_lat": grid_lat[in_radius],
            "center_lon": grid_lon[in_radius],
            "strict_cell_key": cell,
            "technology": str(row.get("technology", "")),
            "site": str(row.get("site", "")),
            "sector_key": str(row.get("sector_key", row.get("sector", ""))),
            "sector": str(row.get("sector", row.get("sector_key", ""))),
            "band": str(row.get("band", "")),
            "original_cell_id": str(row.get("original_cell_id", "")),
            "distance_m": distance_m[in_radius],
            "bearing_deg": bearing[in_radius],
            "azimuth_delta_deg": az_delta[in_radius],
            "phase41_model_raw_rsrp_unclipped": raw[in_radius],
        })
        known = candidates[candidates["strict_cell_key"].astype(str).eq(cell)].copy()
        if known.empty:
            median_delta = 0.0
        else:
            known_raw, _, _, _ = _cost231_raw_vector(
                row,
                pd.to_numeric(known["lat"], errors="coerce").to_numpy(float),
                pd.to_numeric(known["lon"], errors="coerce").to_numpy(float),
            )
            known_final = pd.to_numeric(known["phase39_final_rsrp_unclipped"], errors="coerce").to_numpy(float)
            delta = known_final - known_raw
            median_delta = float(np.nanmedian(delta)) if np.isfinite(delta).any() else 0.0
        frame["phase41_cell_median_delta_db"] = median_delta
        frame["phase41_generated_rsrp_unclipped"] = frame["phase41_model_raw_rsrp_unclipped"] + median_delta
        frame = frame.merge(exact, on=["strict_cell_key", "grid_id"], how="left")
        exact_signal = pd.to_numeric(frame.get("phase39_final_rsrp_unclipped"), errors="coerce")
        frame["phase41_row_source"] = np.where(exact_signal.notna(), "exact_phase39_candidate", "generated_full_500m_grid")
        frame["phase41_signal_unclipped"] = exact_signal.fillna(frame["phase41_generated_rsrp_unclipped"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _apply_phase41_coverage_columns(frame: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    """Production-style coverage columns from the full RF prediction.

    The Phase 39 RF value already includes the complete PAP antenna influence.
    Phase 41 does not remove main/side/back lobes.  It only separates:
      * potential signal from this cell above the coverage floor;
      * serving coverage, added later where this cell is best at the grid.
    """
    out = frame.copy()
    n = len(out)
    if n == 0:
        out["phase41_coverage_eligible"] = pd.Series(dtype=bool)
        out["phase41_potential_rsrp"] = pd.Series(dtype=float)
        out["phase41_display_rsrp"] = pd.Series(dtype=float)
        return out

    signal = pd.to_numeric(out.get(signal_col, pd.Series(np.nan, index=out.index)), errors="coerce")
    raw_signal = signal.to_numpy(float)
    has_signal = raw_signal >= RSRP_NO_COVERAGE_DBM
    eligible = np.isfinite(raw_signal) & has_signal

    out["phase41_raw_rsrp_dbm"] = signal
    out["phase41_has_signal"] = has_signal
    out["phase41_coverage_eligible"] = eligible
    out["phase41_potential_rsrp"] = signal.where(eligible).clip(upper=RSRP_MAX_DBM)
    out["phase41_display_rsrp"] = out["phase41_potential_rsrp"]
    out["phase41_hide_reason"] = np.select(
        [~np.isfinite(raw_signal), ~has_signal],
        ["missing_rsrp", "below_-140_dbm_floor"],
        default="usable_signal",
    )
    return out


def _mark_serving_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["phase41_is_serving_cell"] = False
    out["phase41_serving_rsrp"] = np.nan
    usable = out.dropna(subset=["grid_id", "phase41_potential_rsrp"]).copy()
    if usable.empty:
        return out
    usable["_sort_value"] = pd.to_numeric(usable["phase41_raw_rsrp_dbm"], errors="coerce")
    best_idx = usable.sort_values("_sort_value").groupby(["technology", "grid_id"], dropna=False).tail(1).index
    out.loc[best_idx, "phase41_is_serving_cell"] = True
    out.loc[best_idx, "phase41_serving_rsrp"] = out.loc[best_idx, "phase41_potential_rsrp"]
    out["phase41_display_rsrp"] = out["phase41_serving_rsrp"]
    out.loc[out["phase41_is_serving_cell"], "phase41_hide_reason"] = "serving_cell"
    out.loc[out["phase41_coverage_eligible"] & ~out["phase41_is_serving_cell"], "phase41_hide_reason"] = "usable_but_not_serving"
    return out


def _load_phase39_candidates() -> pd.DataFrame:
    path = PHASE39_DIR / "phase39_scored_candidates_project210.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 39 candidates: {path}")
    return pd.read_parquet(path)


def _grid_bounds() -> pd.DataFrame:
    grid = pd.read_parquet(phase39.p36.PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210.parquet")
    return grid[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]].drop_duplicates("grid_id")


def _save_frame(df: pd.DataFrame, stem: Path) -> None:
    phase22._save_frame(df, stem)


def _best_per_grid(frame: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    work = frame.dropna(subset=["grid_id", value_col]).copy()
    if work.empty:
        return pd.DataFrame(columns=["technology", "grid_id"])
    work["_sort_value"] = pd.to_numeric(work[value_col], errors="coerce")
    keep_cols = [
        "technology",
        "grid_id",
        "strict_cell_key",
        "site",
        "sector_key",
        "band",
        "azimuth_delta_deg",
        "distance_m",
        "phase41_row_source",
        "phase41_hide_reason",
        value_col,
    ]
    keep_cols = [col for col in keep_cols if col in work.columns]
    best = work.sort_values("_sort_value").groupby(["technology", "grid_id"], dropna=False).tail(1)
    best = best[keep_cols].rename(
        columns={
            "strict_cell_key": f"{prefix}_serving_cell",
            "site": f"{prefix}_site",
            "sector_key": f"{prefix}_sector",
            "band": f"{prefix}_band",
            "azimuth_delta_deg": f"{prefix}_azimuth_delta_deg",
            "distance_m": f"{prefix}_distance_m",
            "phase41_row_source": f"{prefix}_row_source",
            "phase41_hide_reason": f"{prefix}_hide_reason",
            value_col: f"{prefix}_rsrp",
        }
    )
    return best.reset_index(drop=True)


def _serving_grid(frame: pd.DataFrame) -> pd.DataFrame:
    bounds = _grid_bounds()
    full = pd.concat([bounds[["grid_id"]].assign(technology=tech) for tech in ("4G", "5G")], ignore_index=True)
    visible = _best_per_grid(frame, "phase41_serving_rsrp", "phase41")
    raw = _best_per_grid(frame, "phase41_raw_rsrp_dbm", "phase41_raw")
    out = full.merge(raw, on=["technology", "grid_id"], how="left").merge(
        visible, on=["technology", "grid_id"], how="left"
    )
    out["phase41_production_rsrp"] = (
        pd.to_numeric(out.get("phase41_raw_rsrp"), errors="coerce")
        .fillna(RSRP_NO_COVERAGE_DBM)
        .clip(lower=RSRP_NO_COVERAGE_DBM, upper=RSRP_MAX_DBM)
    )
    counts = (
        frame.groupby(["technology", "grid_id"], dropna=False)
        .agg(
            phase41_candidate_cells=("strict_cell_key", "nunique"),
            phase41_usable_candidate_cells=("phase41_coverage_eligible", "sum"),
            phase41_serving_candidate_cells=("phase41_is_serving_cell", "sum"),
        )
        .reset_index()
    )
    out = out.merge(counts, on=["technology", "grid_id"], how="left").merge(bounds, on="grid_id", how="left")
    return out


def _sector_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(["technology", "strict_cell_key"], dropna=False):
        technology, cell = key
        signal = pd.to_numeric(group["phase41_serving_rsrp"], errors="coerce")
        potential = pd.to_numeric(group["phase41_potential_rsrp"], errors="coerce")
        raw = pd.to_numeric(group["phase41_raw_rsrp_dbm"], errors="coerce")
        visible = signal.notna()
        rows.append(
            {
                "technology": technology,
                "strict_cell_key": cell,
                "site": group.get("site", pd.Series([""])).astype(str).iloc[0],
                "sector": group.get("sector_key", pd.Series([""])).astype(str).iloc[0],
                "band": group.get("band", pd.Series([""])).astype(str).iloc[0],
                "antenna_model": group.get("antenna_model", pd.Series([""])).astype(str).iloc[0],
                "candidate_rows": int(len(group)),
                "usable_signal_rows": int(potential.notna().sum()),
                "serving_rows": int(visible.sum()),
                "serving_pct": round(float(visible.mean() * 100.0), 2),
                "exact_phase39_rows": int(group["phase41_row_source"].astype(str).eq("exact_phase39_candidate").sum()),
                "generated_full_grid_rows": int(group["phase41_row_source"].astype(str).eq("generated_full_500m_grid").sum()),
                "raw_median_dbm": round(float(raw.median()), 2) if raw.notna().any() else None,
                "usable_signal_median_dbm": round(float(potential.median()), 2) if potential.notna().any() else None,
                "serving_median_dbm": round(float(signal.median()), 2) if signal.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def _cdf_inputs(serving: pd.DataFrame, candidates: pd.DataFrame, technology: str):
    grid = serving[serving["technology"].astype(str).eq(technology)].copy()
    cand = candidates[candidates["technology"].astype(str).eq(technology)].copy()
    return [
        ("1 - Phase 41 production capped/backfilled", grid["phase41_production_rsrp"], "#64748b"),
        ("2 - Phase 41 serving coverage", grid["phase41_rsrp"], "#16a34a"),
        ("3 - Phase 41 all usable cell signal rows", cand["phase41_potential_rsrp"], "#2563eb"),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    retained_candidates = _load_phase39_candidates()
    candidates = _phase41_full_grid_candidates(retained_candidates)
    candidates = _apply_phase41_coverage_columns(candidates, "phase41_signal_unclipped")
    candidates = _mark_serving_coverage(candidates)
    serving = _serving_grid(candidates)
    sector_summary = _sector_summary(candidates)

    _save_frame(candidates, OUT_DIR / "phase41_cell_coverage_project210")
    _save_frame(serving, OUT_DIR / "phase41_serving_grid_project210")
    sector_summary.to_csv(OUT_DIR / "phase41_sector_summary_project210.csv", index=False)
    for technology in ("4G", "5G"):
        _save_frame(
            candidates[candidates["technology"].astype(str).eq(technology)].copy(),
            OUT_DIR / f"phase41_cell_coverage_{technology.lower()}_project210",
        )
        _save_frame(
            serving[serving["technology"].astype(str).eq(technology)].copy(),
            OUT_DIR / f"phase41_serving_grid_{technology.lower()}_project210",
        )
        phase22._plot_cdf(
            _cdf_inputs(serving, candidates, technology),
            f"Project 210 {technology}: Phase 41 serving coverage CDF",
            IMAGE_DIR / f"phase41_{technology.lower()}_coverage_cdf.png",
        )

    summary = {
        "scope": "Phase 41 expands every sector to a full 500 m project grid and builds production-style serving coverage.",
        "rsrp_policy": {
            "raw_signal_column": "phase41_signal_unclipped",
            "potential_signal_column": "phase41_potential_rsrp",
            "serving_signal_column": "phase41_serving_rsrp",
            "production_signal_column": "phase41_production_rsrp",
            "no_coverage_floor_dbm": RSRP_NO_COVERAGE_DBM,
            "display_upper_cap_dbm": RSRP_MAX_DBM,
            "below_floor_action": "NaN for coverage; raw value kept in phase41_raw_rsrp_dbm",
        },
        "full_grid_policy": {
            "sector_radius_m": SECTOR_COVERAGE_RADIUS_M,
            "exact_rows": "Existing Phase 39 candidate rows are preserved where available.",
            "generated_rows": "Missing per-sector grid rows are generated with direct COST231 geometry plus the sector median Phase 39 calibration delta.",
        },
        "coverage_policy": "Full PAP-influenced RF is retained. A sector coverage area is where that sector has usable signal and is the strongest serving cell for the technology/grid.",
        "lobe_policy": "No main-lobe-only filtering and no side/back-lobe deletion. Any lobe can appear only if the RF prediction makes it the serving cell above the coverage floor.",
        "technology": {},
    }
    for technology in ("4G", "5G"):
        tech_grid = serving[serving["technology"].astype(str).eq(technology)].copy()
        tech_cells = candidates[candidates["technology"].astype(str).eq(technology)].copy()
        visible_grid = pd.to_numeric(tech_grid["phase41_rsrp"], errors="coerce").notna()
        usable_cell = pd.to_numeric(tech_cells["phase41_potential_rsrp"], errors="coerce").notna()
        serving_cell = pd.to_numeric(tech_cells["phase41_serving_rsrp"], errors="coerce").notna()
        summary["technology"][technology] = {
            "grid_rows": int(len(tech_grid)),
            "serving_grid_rows": int(visible_grid.sum()),
            "serving_grid_pct": round(float(visible_grid.mean() * 100.0), 2) if len(tech_grid) else 0.0,
            "cell_candidate_rows": int(len(tech_cells)),
            "usable_cell_candidate_rows": int(usable_cell.sum()),
            "usable_cell_candidate_pct": round(float(usable_cell.mean() * 100.0), 2) if len(tech_cells) else 0.0,
            "serving_cell_candidate_rows": int(serving_cell.sum()),
            "serving_cell_candidate_pct": round(float(serving_cell.mean() * 100.0), 2) if len(tech_cells) else 0.0,
            "distinct_cells": int(tech_cells["strict_cell_key"].nunique(dropna=True)),
            "exact_phase39_rows": int(tech_cells["phase41_row_source"].astype(str).eq("exact_phase39_candidate").sum()),
            "generated_full_grid_rows": int(tech_cells["phase41_row_source"].astype(str).eq("generated_full_500m_grid").sum()),
        }
    (OUT_DIR / "phase41_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
