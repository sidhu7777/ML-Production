from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import transform
from sklearn.neighbors import BallTree

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tools.lte_prediction.Sector_wise_prediction_code_copy import compute_sector_rsrp

PROJECT_ID = int(os.getenv("PROP_PROJECT_ID", "196"))
PROJECT_SLUG = os.getenv("PROP_PROJECT_SLUG", "project_196_india")
PHASE_LABEL = os.getenv("COST231_PHASE_LABEL", "Cost231")
OUTPUT_SUBDIR = os.getenv("COST231_OUTPUT_SUBDIR", "cost231")
EARTH_RADIUS_M = 6371000.0
CLIP_RSRP = (-140.0, -44.0)
DT_REPLACE_RADIUS_M = 25.0

PROJECT_DIR = Path(os.getenv("PROP_PROJECT_DIR", str(THIS_DIR / "data" / PROJECT_SLUG)))
BASELINE_SCOPE = PROJECT_DIR / "baseline_fetch_scope"
GEO_DB = PROJECT_DIR / "geo_db"
GRID_DB = PROJECT_DIR / "grid_db"
DATA_DIR = PROJECT_DIR / OUTPUT_SUBDIR
COMBINED_DIR = DATA_DIR / "combined"
INDIVIDUAL_DIR = DATA_DIR / "individually"
WORK_DIR = DATA_DIR / "work"


def _parse_band_float_map(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in str(raw or "").split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        if not key:
            continue
        try:
            out[key] = float(value.strip())
        except ValueError:
            continue
    return out


BAND_FREQ_OVERRIDE_MHZ = _parse_band_float_map(os.getenv("COST231_BAND_FREQ_OVERRIDE_MHZ", ""))
BAND_RSRP_ADJUST_DB = _parse_band_float_map(os.getenv("COST231_BAND_RSRP_ADJUST_DB", ""))
OFFSET_FALLBACK_MODE = os.getenv("COST231_OFFSET_FALLBACK_MODE", "global").strip().lower()


def _ensure_dirs() -> None:
    for path in [DATA_DIR, COMBINED_DIR, INDIVIDUAL_DIR, WORK_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _read_first_existing(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return _read_csv(path)
    raise FileNotFoundError("None of these files exist: " + ", ".join(str(path) for path in paths))


def _clean_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.isna() | text.eq("") | text.str.lower().isin({"nan", "none", "<na>"}))


def _safe_token(value: object, fallback: str = "unknown") -> str:
    text = str(value if value is not None else fallback).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        text = fallback
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text[:150]


def _frequency_from_site(site_df: pd.DataFrame) -> pd.Series:
    freq = pd.Series(np.nan, index=site_df.index, dtype=float)
    for col in ["frequency_mhz", "frequency", "band"]:
        if col not in site_df.columns:
            continue
        candidate = pd.to_numeric(site_df[col], errors="coerce")
        freq = freq.where(pd.notna(freq), candidate)
    return freq.fillna(1800.0).clip(450.0, 3800.0)


def _technology_from_site(site_df: pd.DataFrame) -> pd.Series:
    tech = pd.Series(pd.NA, index=site_df.index, dtype="string")
    for col in ["Technology", "technology", "network_type", "rat", "tech"]:
        if col not in site_df.columns:
            continue
        tech = tech.fillna(_clean_text(site_df[col]))

    band = _clean_text(site_df.get("band", pd.Series(index=site_df.index))).astype("string")
    text = tech.astype("string").str.upper()
    text = text.mask(text.str.contains("5G|NR", na=False), "5G")
    text = text.mask(text.str.contains("4G|LTE", na=False), "4G")
    text = text.mask(text.isna() & band.eq("78"), "5G")
    text = text.mask(text.isna(), "4G")
    return text.fillna("4G")


def _prepare_site_rows(site_df: pd.DataFrame) -> pd.DataFrame:
    out = site_df.copy()
    out["strict_cell_key"] = _clean_text(out["Node_Cell_ID"]).fillna(_clean_text(out["rf_identity_key"]))
    out["original_cell_id"] = _clean_text(out["legacy_nodeb_id_cell_id"]).fillna(_clean_text(out["cell_id"]))
    out["site_key"] = _clean_text(out["site"]).fillna("unknown-site")
    out["sector_key"] = _clean_text(out["sector"]).fillna("unknown-sector")
    out["band_key"] = _clean_text(out["band"]).fillna("unknown-band")
    out["technology_key"] = _technology_from_site(out)
    out["operator_key"] = _clean_text(out["network"]).fillna(_clean_text(out.get("operator", pd.Series(index=out.index))))
    out["site_sector_band_key"] = (
        out["site_key"].astype(str)
        + "|"
        + out["sector_key"].astype(str)
        + "|"
        + out["band_key"].astype(str)
    )

    numeric_defaults = {
        "lat": None,
        "lon": None,
        "azimuth": 0.0,
        "Height": 30.0,
        "Etilt": 3.0,
        "Mtilt": 0.0,
        "tx_power": 46.0,
    }
    for col, default in numeric_defaults.items():
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if default is not None:
            out[col] = out[col].fillna(default)
    out["frequency_mhz"] = _frequency_from_site(out)
    out["original_frequency_mhz"] = out["frequency_mhz"]
    if BAND_FREQ_OVERRIDE_MHZ:
        band_text = out["band_key"].astype(str)
        for band, freq_mhz in BAND_FREQ_OVERRIDE_MHZ.items():
            out.loc[band_text == str(band), "frequency_mhz"] = float(freq_mhz)
    out["model_rsrp_adjust_db"] = 0.0
    if BAND_RSRP_ADJUST_DB:
        band_text = out["band_key"].astype(str)
        for band, adjust_db in BAND_RSRP_ADJUST_DB.items():
            out.loc[band_text == str(band), "model_rsrp_adjust_db"] = float(adjust_db)
    out = out.dropna(subset=["lat", "lon", "strict_cell_key"]).copy()
    out = out.drop_duplicates(subset=["strict_cell_key"], keep="first").reset_index(drop=True)
    return out


def _load_project_polygon() -> object | None:
    regions = _read_csv(GEO_DB / f"map_regions_project_{PROJECT_ID}_active.csv")
    if regions.empty or "region_wkt" not in regions.columns:
        return None
    geom = wkt.loads(str(regions.iloc[0]["region_wkt"]))
    # DB WKT is stored as lat lon; shapely expects x y, so swap to lon lat.
    geom = transform(lambda x, y, z=None: (y, x) if z is None else (y, x, z), geom)
    return geom if geom.is_valid else geom.buffer(0)


def _load_grid_points(project_polygon) -> pd.DataFrame:
    grid = _read_csv(GRID_DB / f"grid_analytics_project_{PROJECT_ID}_selected_scenario.csv")
    required = ["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]
    missing = [col for col in required if col not in grid.columns]
    if missing:
        raise ValueError(f"Grid data missing columns: {missing}")
    grid = grid[required + [col for col in ["grid_size_meters", "scenario_id"] if col in grid.columns]].copy()
    for col in ["center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon", "grid_size_meters"]:
        if col in grid.columns:
            grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid = grid.dropna(subset=["grid_id", "center_lat", "center_lon"]).drop_duplicates("grid_id").copy()
    if project_polygon is not None:
        mask = [project_polygon.covers(Point(lon, lat)) for lat, lon in grid[["center_lat", "center_lon"]].to_numpy()]
        filtered = grid.loc[mask].copy()
        if not filtered.empty:
            grid = filtered
    grid = grid.sort_values("grid_id").reset_index(drop=True)
    if grid.empty:
        raise ValueError("No grid points available inside project polygon")
    return grid


def _site_to_cost231_record(row: pd.Series) -> dict:
    return {
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "azimuth": float(row["azimuth"]),
        "electrical_tilt": float(row["Etilt"]),
        "mechanical_tilt": float(row["Mtilt"]),
        "antenna_height": float(row["Height"]),
        "tx_power": float(row["tx_power"]),
        "Node_Cell_ID": str(row["strict_cell_key"]),
        "frequency_mhz": float(row["frequency_mhz"]),
    }


def _cost231_for_points(site: dict, lat_values: np.ndarray, lon_values: np.ndarray, freq_mhz: float) -> np.ndarray:
    params = {"k1": 0, "k2": 0, "antenna_gain": 18.0, "cable_loss": 2.0, "ue_height": 1.5}
    values = [
        compute_sector_rsrp(site, float(lat), float(lon), float(freq_mhz), params)
        for lat, lon in zip(lat_values, lon_values)
    ]
    return np.clip(np.asarray(values, dtype=float), *CLIP_RSRP)


def _run_raw_cost231(site_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    grid_lat = grid_df["center_lat"].to_numpy(dtype=float)
    grid_lon = grid_df["center_lon"].to_numpy(dtype=float)
    frames: list[pd.DataFrame] = []
    for idx, row in site_df.iterrows():
        site = _site_to_cost231_record(row)
        raw = _cost231_for_points(site, grid_lat, grid_lon, float(row["frequency_mhz"]))
        raw = np.clip(raw + float(row.get("model_rsrp_adjust_db", 0.0)), *CLIP_RSRP)
        cell_frame = pd.DataFrame(
            {
                "project_id": PROJECT_ID,
                "grid_id": grid_df["grid_id"].astype(str).to_numpy(),
                "lat": grid_lat,
                "lon": grid_lon,
                "strict_cell_key": str(row["strict_cell_key"]),
                "site_sector_band_key": str(row["site_sector_band_key"]),
                "site": str(row["site_key"]),
                "sector": str(row["sector_key"]),
                "band": str(row["band_key"]),
                "technology": str(row["technology_key"]),
                "operator": str(row["operator_key"]),
                "original_cell_id": str(row["original_cell_id"]),
                "original_frequency_mhz": float(row.get("original_frequency_mhz", row["frequency_mhz"])),
                "frequency_mhz": float(row["frequency_mhz"]),
                "model_rsrp_adjust_db": float(row.get("model_rsrp_adjust_db", 0.0)),
                "raw_cost231_rsrp": np.clip(raw, *CLIP_RSRP),
            }
        )
        frames.append(cell_frame)
        if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == len(site_df):
            print(f"[COST231] cells_done={idx + 1}/{len(site_df)} rows_so_far={(idx + 1) * len(grid_df)}")
    return pd.concat(frames, ignore_index=True)


def _run_cost231_at_dt(site_df: pd.DataFrame, dt_df: pd.DataFrame) -> pd.DataFrame:
    dt_lat = dt_df["lat"].to_numpy(dtype=float)
    dt_lon = dt_df["lon"].to_numpy(dtype=float)
    pred_matrix = np.empty((len(dt_df), len(site_df)), dtype=float)
    for idx, row in site_df.iterrows():
        site = _site_to_cost231_record(row)
        pred_matrix[:, idx] = np.clip(
            _cost231_for_points(site, dt_lat, dt_lon, float(row["frequency_mhz"]))
            + float(row.get("model_rsrp_adjust_db", 0.0)),
            *CLIP_RSRP,
        )
    best_idx = np.argmax(pred_matrix, axis=1)
    assigned = site_df.iloc[best_idx].reset_index(drop=True)
    out = dt_df.reset_index(drop=True).copy()
    out["assigned_strict_cell_key"] = assigned["strict_cell_key"].astype(str).to_numpy()
    out["assigned_site_sector_band_key"] = assigned["site_sector_band_key"].astype(str).to_numpy()
    out["assigned_site"] = assigned["site_key"].astype(str).to_numpy()
    out["assigned_sector"] = assigned["sector_key"].astype(str).to_numpy()
    out["assigned_band"] = assigned["band_key"].astype(str).to_numpy()
    out["assigned_technology"] = assigned["technology_key"].astype(str).to_numpy()
    out["raw_cost231_at_dt_rsrp"] = pred_matrix[np.arange(len(out)), best_idx]
    out["dt_minus_cost231_db"] = out["rsrp_measured"] - out["raw_cost231_at_dt_rsrp"]
    return out


def _prepare_dt(drive_df: pd.DataFrame) -> pd.DataFrame:
    out = drive_df.copy()
    for col in ["lat", "lon", "rsrp"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["lat", "lon", "rsrp"]).copy()
    out = out[(out["rsrp"] >= -150.0) & (out["rsrp"] <= -30.0)].copy()
    out = out.rename(columns={"rsrp": "rsrp_measured"})
    out["dt_row_id"] = np.arange(len(out))
    return out.reset_index(drop=True)


def _attach_nearest_grid(dt_assigned: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    tree = BallTree(np.radians(grid_df[["center_lat", "center_lon"]].to_numpy(dtype=float)), metric="haversine")
    dist_rad, idx = tree.query(np.radians(dt_assigned[["lat", "lon"]].to_numpy(dtype=float)), k=1)
    out = dt_assigned.copy()
    nearest = grid_df.iloc[idx[:, 0]].reset_index(drop=True)
    out["nearest_grid_id"] = nearest["grid_id"].astype(str).to_numpy()
    out["nearest_grid_distance_m"] = dist_rad[:, 0] * EARTH_RADIUS_M
    out["dt_replacement_eligible"] = out["nearest_grid_distance_m"] <= DT_REPLACE_RADIUS_M
    return out


def _offset_table(site_df: pd.DataFrame, dt_assigned: pd.DataFrame) -> pd.DataFrame:
    valid = dt_assigned.dropna(subset=["dt_minus_cost231_db"]).copy()
    grouped = (
        valid.groupby("assigned_strict_cell_key", dropna=False)
        .agg(
            dt_count=("dt_minus_cost231_db", "size"),
            offset_db=("dt_minus_cost231_db", "median"),
            offset_mean_db=("dt_minus_cost231_db", "mean"),
            offset_p25_db=("dt_minus_cost231_db", lambda s: float(np.nanpercentile(s, 25))),
            offset_p75_db=("dt_minus_cost231_db", lambda s: float(np.nanpercentile(s, 75))),
        )
        .reset_index()
        .rename(columns={"assigned_strict_cell_key": "strict_cell_key"})
    )
    global_offset = float(valid["dt_minus_cost231_db"].median()) if not valid.empty else 0.0
    out = site_df[
        [
            "strict_cell_key",
            "site_sector_band_key",
            "site_key",
            "sector_key",
            "band_key",
            "technology_key",
            "operator_key",
        ]
    ].copy()
    out = out.merge(grouped, on="strict_cell_key", how="left")
    out["strict_dt_count"] = out["dt_count"].fillna(0).astype(int)
    out["offset_source"] = np.where(out["offset_db"].notna(), "cell_dt_median", "global_dt_median")

    if OFFSET_FALLBACK_MODE in {"technology", "tech"} and not valid.empty:
        valid["assigned_technology"] = _clean_text(valid["assigned_technology"]).fillna("unknown-technology")
        tech_stats = (
            valid.groupby("assigned_technology", dropna=False)["dt_minus_cost231_db"]
            .agg(
                technology_dt_count="size",
                technology_offset_db="median",
                technology_offset_mean_db="mean",
                technology_offset_p25_db=lambda s: float(np.nanpercentile(s, 25)),
                technology_offset_p75_db=lambda s: float(np.nanpercentile(s, 75)),
            )
            .reset_index()
            .rename(columns={"assigned_technology": "technology_key"})
        )
        out = out.merge(tech_stats, on="technology_key", how="left")
        use_tech = out["offset_db"].isna() & out["technology_offset_db"].notna()
        out.loc[use_tech, "offset_source"] = "technology_dt_median"
        out.loc[use_tech, "offset_db"] = out.loc[use_tech, "technology_offset_db"]
        out.loc[use_tech, "offset_mean_db"] = out.loc[use_tech, "technology_offset_mean_db"]
        out.loc[use_tech, "offset_p25_db"] = out.loc[use_tech, "technology_offset_p25_db"]
        out.loc[use_tech, "offset_p75_db"] = out.loc[use_tech, "technology_offset_p75_db"]
        out["fallback_dt_count"] = out["technology_dt_count"].where(use_tech, out["strict_dt_count"])
    else:
        out["fallback_dt_count"] = out["strict_dt_count"]

    out["dt_count"] = out["dt_count"].fillna(0).astype(int)
    out["offset_db"] = out["offset_db"].fillna(global_offset)
    out["offset_mean_db"] = out["offset_mean_db"].fillna(global_offset)
    out["offset_p25_db"] = out["offset_p25_db"].fillna(global_offset)
    out["offset_p75_db"] = out["offset_p75_db"].fillna(global_offset)
    out["fallback_dt_count"] = out["fallback_dt_count"].fillna(len(valid)).astype(int)
    return out


def _apply_offset_and_replacement(
    surface: pd.DataFrame,
    offsets: pd.DataFrame,
    dt_with_grid: pd.DataFrame,
) -> pd.DataFrame:
    out = surface.merge(
        offsets[["strict_cell_key", "offset_db", "offset_source", "dt_count", "fallback_dt_count"]],
        on="strict_cell_key",
        how="left",
    )
    out["offset_db"] = pd.to_numeric(out["offset_db"], errors="coerce").fillna(0.0)
    out["offset_corrected_rsrp"] = (out["raw_cost231_rsrp"] + out["offset_db"]).clip(*CLIP_RSRP)
    replacements = (
        dt_with_grid.loc[dt_with_grid["dt_replacement_eligible"]]
        .groupby(["assigned_strict_cell_key", "nearest_grid_id"], dropna=False)
        .agg(
            dt_replacement_rsrp=("rsrp_measured", "mean"),
            dt_replacement_count=("rsrp_measured", "size"),
            mean_dt_grid_distance_m=("nearest_grid_distance_m", "mean"),
        )
        .reset_index()
        .rename(columns={"assigned_strict_cell_key": "strict_cell_key", "nearest_grid_id": "grid_id"})
    )
    out = out.merge(replacements, on=["strict_cell_key", "grid_id"], how="left")
    out["dt_replaced"] = out["dt_replacement_rsrp"].notna()
    out["corrected_rsrp"] = out["offset_corrected_rsrp"].where(~out["dt_replaced"], out["dt_replacement_rsrp"])
    out["corrected_rsrp"] = pd.to_numeric(out["corrected_rsrp"], errors="coerce").clip(*CLIP_RSRP)
    out["dt_replacement_count"] = out["dt_replacement_count"].fillna(0).astype(int)
    return out


def _cdf_values(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    arr.sort()
    y = np.arange(1, arr.size + 1, dtype=float) / arr.size * 100.0
    return arr, y


def _plot_cdf(series_map: list[tuple[str, pd.Series | np.ndarray, str]], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plotted = 0
    for label, values, color in series_map:
        x, y = _cdf_values(values)
        if len(x) == 0:
            continue
        ax.plot(x, y, linewidth=2.2, label=f"{label} (n={len(x):,})", color=color)
        plotted += 1
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("RSRP (dBm)")
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.set_xlim(CLIP_RSRP[0], CLIP_RSRP[1])
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    if plotted:
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_outputs(surface: pd.DataFrame, dt_with_grid: pd.DataFrame, offsets: pd.DataFrame) -> None:
    _plot_cdf(
        [
            (f"{PHASE_LABEL} before - complete polygon", surface["raw_cost231_rsrp"], "#d94f3d"),
            ("After offset + DT pixel replacement - complete polygon", surface["corrected_rsrp"], "#168a52"),
        ],
        f"Project {PROJECT_ID} {PHASE_LABEL} CDF - Complete Polygon, {surface['strict_cell_key'].nunique()} Cells",
        COMBINED_DIR / "cdf_complete_polygon_cost231_before_after_offset.png",
    )
    _plot_cdf(
        [
            (f"{PHASE_LABEL} before at DT", dt_with_grid["raw_cost231_at_dt_rsrp"], "#d94f3d"),
            ("After at DT pixels", dt_with_grid["after_at_dt_pixel_rsrp"], "#168a52"),
            ("DT measured", dt_with_grid["rsrp_measured"], "#2b6cb0"),
        ],
        f"Project {PROJECT_ID} {PHASE_LABEL} CDF - DT Locations",
        COMBINED_DIR / "cdf_dt_locations_cost231_before_after_offset.png",
    )
    _plot_cdf(
        [
            ("Cell median offsets", offsets["offset_db"], "#805ad5"),
        ],
        f"Project {PROJECT_ID} Per-Cell Offset CDF",
        COMBINED_DIR / "cdf_per_cell_offset_db.png",
    )

    for idx, (cell_key, cell_df) in enumerate(surface.groupby("strict_cell_key", sort=True), start=1):
        dt_cell = dt_with_grid.loc[dt_with_grid["assigned_strict_cell_key"].astype(str) == str(cell_key)]
        first = cell_df.iloc[0]
        title = (
            f"Cell {idx:03d}: site {first['site']} sector {first['sector']} "
            f"band {first['band']} offset {float(first['offset_db']):+.2f} dB"
        )
        image_name = (
            f"{idx:03d}_site_{_safe_token(first['site'])}_sector_{_safe_token(first['sector'])}_"
            f"band_{_safe_token(first['band'])}_cdf.png"
        )
        series = [
            (f"{PHASE_LABEL} before", cell_df["raw_cost231_rsrp"], "#d94f3d"),
            ("After offset + DT replacement", cell_df["corrected_rsrp"], "#168a52"),
        ]
        if not dt_cell.empty:
            series.append(("DT measured assigned to this cell", dt_cell["rsrp_measured"], "#2b6cb0"))
        _plot_cdf(series, title, INDIVIDUAL_DIR / image_name)
        total_cells = int(surface["strict_cell_key"].nunique())
        if idx == 1 or idx % 20 == 0 or idx == total_cells:
            print(f"[PLOT] individual_cdf_done={idx}/{total_cells}")


def _save_outputs(surface: pd.DataFrame, dt_with_grid: pd.DataFrame, offsets: pd.DataFrame, summary: dict) -> None:
    surface_csv = DATA_DIR / f"cost231_offset_corrected_surface_project{PROJECT_ID}.csv"
    dt_csv = DATA_DIR / f"cost231_dt_match_project{PROJECT_ID}.csv"
    offsets_csv = DATA_DIR / f"cost231_offsets_cells_project{PROJECT_ID}.csv"
    surface.to_csv(surface_csv, index=False)
    dt_with_grid.to_csv(dt_csv, index=False)
    offsets.to_csv(offsets_csv, index=False)
    for df, path in [(surface, surface_csv), (dt_with_grid, dt_csv), (offsets, offsets_csv)]:
        try:
            df.to_parquet(path.with_suffix(".parquet"), index=False)
        except Exception as exc:
            path.with_suffix(".parquet.error.txt").write_text(str(exc), encoding="utf-8")
    summary.update(
        {
            "surface_csv": str(surface_csv.relative_to(THIS_DIR)),
            "dt_match_csv": str(dt_csv.relative_to(THIS_DIR)),
            "offsets_csv": str(offsets_csv.relative_to(THIS_DIR)),
            "combined_dir": str(COMBINED_DIR.relative_to(THIS_DIR)),
            "individual_dir": str(INDIVIDUAL_DIR.relative_to(THIS_DIR)),
        }
    )
    (DATA_DIR / "cost231_offset_cdf_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    print(f"[START] project={PROJECT_ID} project_dir={PROJECT_DIR}")
    site_df = _prepare_site_rows(
        _read_first_existing(
            [
                BASELINE_SCOPE / f"site_identity_strict_cells_project{PROJECT_ID}.csv",
                BASELINE_SCOPE / "site_identity_102_strict_cells.csv",
            ]
        )
    )
    drive_df = _prepare_dt(
        _read_first_existing(
            [
                BASELINE_SCOPE / f"drive_project_{PROJECT_ID}_baseline_primary_polygon.csv",
                BASELINE_SCOPE / f"drive_project_{PROJECT_ID}_baseline_primary_taiwan_polygon.csv",
                BASELINE_SCOPE / f"drive_project_{PROJECT_ID}_baseline_primary_airtel_polygon.csv",
            ]
        )
    )
    project_polygon = _load_project_polygon()
    grid_df = _load_grid_points(project_polygon)
    print(f"[INPUT] strict_cells={len(site_df)} dt_rows={len(drive_df)} grid_pixels={len(grid_df)}")

    surface = _run_raw_cost231(site_df, grid_df)
    dt_assigned = _run_cost231_at_dt(site_df, drive_df)
    dt_with_grid = _attach_nearest_grid(dt_assigned, grid_df)
    offsets = _offset_table(site_df, dt_with_grid)
    surface = _apply_offset_and_replacement(surface, offsets, dt_with_grid)

    dt_after_lookup = surface[
        ["strict_cell_key", "grid_id", "raw_cost231_rsrp", "corrected_rsrp", "offset_corrected_rsrp"]
    ].rename(
        columns={
            "strict_cell_key": "assigned_strict_cell_key",
            "grid_id": "nearest_grid_id",
            "raw_cost231_rsrp": "raw_cost231_nearest_grid_rsrp",
            "corrected_rsrp": "after_at_dt_pixel_rsrp",
            "offset_corrected_rsrp": "after_offset_only_nearest_grid_rsrp",
        }
    )
    dt_with_grid = dt_with_grid.merge(dt_after_lookup, on=["assigned_strict_cell_key", "nearest_grid_id"], how="left")

    summary = {
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "project_id": PROJECT_ID,
        "production_code_modified": False,
        "production_baseline_run": False,
        "phase_label": PHASE_LABEL,
        "output_subdir": OUTPUT_SUBDIR,
        "cost231_function": "tools.lte_prediction.Sector_wise_prediction_code_copy.compute_sector_rsrp",
        "band_frequency_override_mhz": BAND_FREQ_OVERRIDE_MHZ,
        "band_rsrp_adjust_db": BAND_RSRP_ADJUST_DB,
        "offset_fallback_mode": OFFSET_FALLBACK_MODE,
        "strict_cells": int(len(site_df)),
        "grid_pixels": int(len(grid_df)),
        "surface_rows": int(len(surface)),
        "dt_rows": int(len(drive_df)),
        "dt_rows_assigned": int(len(dt_with_grid)),
        "dt_replacement_radius_m": DT_REPLACE_RADIUS_M,
        "dt_replacement_rows": int(surface["dt_replaced"].sum()),
        "cells_with_dt_offset": int((offsets["dt_count"] > 0).sum()),
        "cells_using_technology_offset": int((offsets["offset_source"] == "technology_dt_median").sum()),
        "cells_using_global_offset": int((offsets["offset_source"] == "global_dt_median").sum()),
        "offset_source_counts": offsets["offset_source"].value_counts(dropna=False).to_dict(),
        "global_offset_db": float(dt_with_grid["dt_minus_cost231_db"].median()) if not dt_with_grid.empty else 0.0,
        "raw_cost231_rsrp": {
            "min": float(surface["raw_cost231_rsrp"].min()),
            "max": float(surface["raw_cost231_rsrp"].max()),
            "mean": float(surface["raw_cost231_rsrp"].mean()),
        },
        "corrected_rsrp": {
            "min": float(surface["corrected_rsrp"].min()),
            "max": float(surface["corrected_rsrp"].max()),
            "mean": float(surface["corrected_rsrp"].mean()),
        },
    }

    _plot_outputs(surface, dt_with_grid, offsets)
    _save_outputs(surface, dt_with_grid, offsets, summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
