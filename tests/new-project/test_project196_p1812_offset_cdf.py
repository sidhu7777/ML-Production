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
import rasterio
from rasterio.transform import rowcol
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import transform
from sklearn.neighbors import BallTree

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
PROJECT_ID = int(os.getenv("PROP_PROJECT_ID", "196"))
PROJECT_SLUG = os.getenv("PROP_PROJECT_SLUG", "project_196_india")
EARTH_RADIUS_M = 6371000.0
DT_REPLACE_RADIUS_M = 25.0
CLIP_RSRP = (-140.0, -44.0)

PROJECT_DIR = Path(os.getenv("PROP_PROJECT_DIR", str(THIS_DIR / "data" / PROJECT_SLUG)))
BASELINE_SCOPE = PROJECT_DIR / "baseline_fetch_scope"
GEO_DB = PROJECT_DIR / "geo_db"
GRID_DB = PROJECT_DIR / "grid_db"
DATA_DIR = PROJECT_DIR
P1812_DIR = PROJECT_DIR / "p1812"
COMBINED_DIR = P1812_DIR / "combined"
INDIVIDUAL_DIR = P1812_DIR / "individually"
INPUT_DIR = P1812_DIR / "input"
COMPARISON_DIR = PROJECT_DIR / "comparison"
DEM_PATH = Path(os.getenv("PROP_DEM_PATH", str(ML_ROOT / "data" / "dem" / f"project_{PROJECT_ID}_dem.tif")))


def _ensure_dirs() -> None:
    for path in [P1812_DIR, COMBINED_DIR, INDIVIDUAL_DIR, INPUT_DIR, COMPARISON_DIR]:
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
    return freq.fillna(1800.0).clip(450.0, 6000.0)


def _prepare_site_rows(site_df: pd.DataFrame) -> pd.DataFrame:
    out = site_df.copy()
    out["strict_cell_key"] = _clean_text(out["Node_Cell_ID"]).fillna(_clean_text(out["rf_identity_key"]))
    out["original_cell_id"] = _clean_text(out["legacy_nodeb_id_cell_id"]).fillna(_clean_text(out["cell_id"]))
    out["site_key"] = _clean_text(out["site"]).fillna("unknown-site")
    out["sector_key"] = _clean_text(out["sector"]).fillna("unknown-sector")
    out["band_key"] = _clean_text(out["band"]).fillna("unknown-band")
    out["operator_key"] = _clean_text(out["network"]).fillna(_clean_text(out.get("operator", pd.Series(index=out.index))))
    out["site_sector_band_key"] = (
        out["site_key"].astype(str) + "|" + out["sector_key"].astype(str) + "|" + out["band_key"].astype(str)
    )
    for col, default in {
        "lat": None,
        "lon": None,
        "azimuth": 0.0,
        "Height": 30.0,
        "Etilt": 3.0,
        "Mtilt": 0.0,
        "tx_power": 46.0,
    }.items():
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if default is not None:
            out[col] = out[col].fillna(default)
    out["frequency_mhz"] = _frequency_from_site(out)
    out = out.dropna(subset=["lat", "lon", "strict_cell_key"]).drop_duplicates("strict_cell_key").reset_index(drop=True)
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


def _load_project_polygon() -> object | None:
    regions = _read_csv(GEO_DB / f"map_regions_project_{PROJECT_ID}_active.csv")
    if regions.empty or "region_wkt" not in regions.columns:
        return None
    geom = wkt.loads(str(regions.iloc[0]["region_wkt"]))
    geom = transform(lambda x, y, z=None: (y, x) if z is None else (y, x, z), geom)
    return geom if geom.is_valid else geom.buffer(0)


def _load_grid_points(project_polygon) -> pd.DataFrame:
    grid = _read_csv(GRID_DB / f"grid_analytics_project_{PROJECT_ID}_selected_scenario.csv")
    required = ["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]
    for col in required + ["grid_size_meters", "scenario_id"]:
        if col in grid.columns:
            grid[col] = pd.to_numeric(grid[col], errors="coerce") if col != "grid_id" else grid[col]
    grid = grid.dropna(subset=["grid_id", "center_lat", "center_lon"]).drop_duplicates("grid_id").copy()
    if project_polygon is not None:
        mask = [project_polygon.covers(Point(lon, lat)) for lat, lon in grid[["center_lat", "center_lon"]].to_numpy()]
        filtered = grid.loc[mask].copy()
        if not filtered.empty:
            grid = filtered
    if grid.empty:
        raise ValueError("No grid points available inside project polygon")
    return grid.sort_values("grid_id").reset_index(drop=True)


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def antenna_gain(site: pd.Series, point_lat: np.ndarray, point_lon: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
    azimuth = float(site["azimuth"])
    height = float(site["Height"])
    etilt = float(site["Etilt"])
    mtilt = float(site["Mtilt"])
    bearing = bearing_deg(float(site["lat"]), float(site["lon"]), point_lat, point_lon)
    az_diff = np.abs((bearing - azimuth + 180.0) % 360.0 - 180.0)
    elev_angle = np.degrees(np.arctan2(1.5 - height, np.maximum(distance_m, 1.0)))
    elev_diff = elev_angle + etilt + mtilt
    ah = np.minimum(12.0 * np.square(az_diff / 65.0), 30.0)
    av = np.minimum(12.0 * np.square(elev_diff / 6.0), 20.0)
    return 18.0 - np.minimum(ah + av, 30.0)


class DemSampler:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        self.dataset = rasterio.open(path)
        self.nodata = self.dataset.nodata

    def close(self) -> None:
        self.dataset.close()

    def sample(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        coords = list(zip(lon.astype(float), lat.astype(float)))
        values = np.array([value[0] for value in self.dataset.sample(coords)], dtype=float)
        if self.nodata is not None:
            values = np.where(values == self.nodata, np.nan, values)
        return values


def _path_diffraction_loss(
    dem: DemSampler,
    site_lat: float,
    site_lon: float,
    site_ground_m: float,
    site_height_m: float,
    rx_lat: float,
    rx_lon: float,
    rx_ground_m: float,
    freq_mhz: float,
) -> tuple[float, float]:
    distance_m = float(haversine_m(site_lat, site_lon, rx_lat, rx_lon))
    if distance_m < 1.0:
        return 0.0, distance_m
    sample_count = int(np.clip(math.ceil(distance_m / 35.0), 5, 64))
    fractions = np.linspace(0.0, 1.0, sample_count)
    lats = site_lat + (rx_lat - site_lat) * fractions
    lons = site_lon + (rx_lon - site_lon) * fractions
    terrain = dem.sample(lats, lons)
    if np.isnan(terrain).all():
        return 0.0, distance_m
    terrain = pd.Series(terrain).interpolate(limit_direction="both").to_numpy(dtype=float)
    tx_alt = float(site_ground_m) + float(site_height_m)
    rx_alt = float(rx_ground_m) + 1.5
    los_alt = tx_alt + (rx_alt - tx_alt) * fractions
    clearance = terrain - los_alt
    if len(clearance) <= 2:
        return 0.0, distance_m
    inside = clearance[1:-1]
    if inside.size == 0:
        return 0.0, distance_m
    idx = int(np.nanargmax(inside)) + 1
    h = float(clearance[idx])
    d1 = max(distance_m * float(fractions[idx]), 1.0)
    d2 = max(distance_m - d1, 1.0)
    wavelength_m = 300.0 / max(float(freq_mhz), 1.0)
    v = h * math.sqrt(2.0 * (d1 + d2) / (wavelength_m * d1 * d2))
    if v <= -0.78:
        return 0.0, distance_m
    loss = 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return float(np.clip(loss, 0.0, 45.0)), distance_m


def _plain_p1812_rsrp_for_site(site: pd.Series, points: pd.DataFrame, dem: DemSampler) -> pd.DataFrame:
    freq_mhz = float(site["frequency_mhz"])
    point_lat = points["lat"].to_numpy(dtype=float)
    point_lon = points["lon"].to_numpy(dtype=float)
    distance_m = np.maximum(haversine_m(float(site["lat"]), float(site["lon"]), point_lat, point_lon), 1.0)
    fspl = 32.45 + 20.0 * np.log10(freq_mhz) + 20.0 * np.log10(np.maximum(distance_m / 1000.0, 0.001))
    gain = antenna_gain(site, point_lat, point_lon, distance_m)
    site_ground = float(dem.sample(np.array([float(site["lat"])]), np.array([float(site["lon"])]))[0])
    rx_ground = dem.sample(point_lat, point_lon)
    if not np.isfinite(site_ground):
        site_ground = 0.0
    rx_ground = np.where(np.isfinite(rx_ground), rx_ground, site_ground)
    diffraction = np.zeros(len(points), dtype=float)
    for i, (lat, lon, ground) in enumerate(zip(point_lat, point_lon, rx_ground)):
        diffraction[i], _ = _path_diffraction_loss(
            dem,
            float(site["lat"]),
            float(site["lon"]),
            site_ground,
            float(site["Height"]),
            float(lat),
            float(lon),
            float(ground),
            freq_mhz,
        )
    pathloss = fspl + diffraction
    rsrp = float(site["tx_power"]) + gain - pathloss - 2.0
    out = points[["grid_id", "lat", "lon"]].copy()
    out["plain_p1812_rsrp"] = np.clip(rsrp, *CLIP_RSRP)
    out["p1812_pathloss_db"] = pathloss
    out["p1812_diffraction_db"] = diffraction
    out["distance_m"] = distance_m
    return out


def _run_plain_p1812(site_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    dem = DemSampler(DEM_PATH)
    try:
        frames = []
        points = grid_df.rename(columns={"center_lat": "lat", "center_lon": "lon"}).copy()
        for idx, site in site_df.iterrows():
            pred = _plain_p1812_rsrp_for_site(site, points, dem)
            pred["project_id"] = PROJECT_ID
            pred["strict_cell_key"] = str(site["strict_cell_key"])
            pred["site_sector_band_key"] = str(site["site_sector_band_key"])
            pred["site"] = str(site["site_key"])
            pred["sector"] = str(site["sector_key"])
            pred["band"] = str(site["band_key"])
            pred["operator"] = str(site["operator_key"])
            pred["original_cell_id"] = str(site["original_cell_id"])
            pred["frequency_mhz"] = float(site["frequency_mhz"])
            frames.append(pred)
            if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == len(site_df):
                print(f"[P1812] cells_done={idx + 1}/{len(site_df)} rows_so_far={(idx + 1) * len(points)}")
        return pd.concat(frames, ignore_index=True)
    finally:
        dem.close()


def _run_p1812_at_dt(site_df: pd.DataFrame, dt_df: pd.DataFrame) -> pd.DataFrame:
    dem = DemSampler(DEM_PATH)
    try:
        points = dt_df[["lat", "lon"]].copy()
        pred_matrix = np.empty((len(dt_df), len(site_df)), dtype=float)
        for idx, site in site_df.iterrows():
            pred_matrix[:, idx] = _plain_p1812_rsrp_for_site(site, points.assign(grid_id=np.arange(len(points))), dem)[
                "plain_p1812_rsrp"
            ].to_numpy(dtype=float)
        best_idx = np.argmax(pred_matrix, axis=1)
        assigned = site_df.iloc[best_idx].reset_index(drop=True)
        out = dt_df.reset_index(drop=True).copy()
        out["assigned_strict_cell_key"] = assigned["strict_cell_key"].astype(str).to_numpy()
        out["assigned_site_sector_band_key"] = assigned["site_sector_band_key"].astype(str).to_numpy()
        out["assigned_site"] = assigned["site_key"].astype(str).to_numpy()
        out["assigned_sector"] = assigned["sector_key"].astype(str).to_numpy()
        out["assigned_band"] = assigned["band_key"].astype(str).to_numpy()
        out["plain_p1812_at_dt_rsrp"] = pred_matrix[np.arange(len(out)), best_idx]
        out["dt_minus_p1812_db"] = out["rsrp_measured"] - out["plain_p1812_at_dt_rsrp"]
        return out
    finally:
        dem.close()


def _attach_nearest_grid(dt_assigned: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    centers = grid_df.rename(columns={"center_lat": "lat", "center_lon": "lon"})
    tree = BallTree(np.radians(centers[["lat", "lon"]].to_numpy(dtype=float)), metric="haversine")
    dist_rad, idx = tree.query(np.radians(dt_assigned[["lat", "lon"]].to_numpy(dtype=float)), k=1)
    nearest = centers.iloc[idx[:, 0]].reset_index(drop=True)
    out = dt_assigned.copy()
    out["nearest_grid_id"] = nearest["grid_id"].astype(str).to_numpy()
    out["nearest_grid_distance_m"] = dist_rad[:, 0] * EARTH_RADIUS_M
    out["dt_replacement_eligible"] = out["nearest_grid_distance_m"] <= DT_REPLACE_RADIUS_M
    return out


def _offset_table(site_df: pd.DataFrame, dt_assigned: pd.DataFrame) -> pd.DataFrame:
    valid = dt_assigned.dropna(subset=["dt_minus_p1812_db"]).copy()
    grouped = (
        valid.groupby("assigned_strict_cell_key", dropna=False)
        .agg(
            dt_count=("dt_minus_p1812_db", "size"),
            offset_db=("dt_minus_p1812_db", "median"),
            offset_mean_db=("dt_minus_p1812_db", "mean"),
            offset_p25_db=("dt_minus_p1812_db", lambda s: float(np.nanpercentile(s, 25))),
            offset_p75_db=("dt_minus_p1812_db", lambda s: float(np.nanpercentile(s, 75))),
        )
        .reset_index()
        .rename(columns={"assigned_strict_cell_key": "strict_cell_key"})
    )
    global_offset = float(valid["dt_minus_p1812_db"].median()) if not valid.empty else 0.0
    out = site_df[
        ["strict_cell_key", "site_sector_band_key", "site_key", "sector_key", "band_key", "operator_key"]
    ].copy()
    out = out.merge(grouped, on="strict_cell_key", how="left")
    out["offset_source"] = np.where(out["offset_db"].notna(), "cell_dt_median", "global_dt_median")
    out["dt_count"] = out["dt_count"].fillna(0).astype(int)
    for col in ["offset_db", "offset_mean_db", "offset_p25_db", "offset_p75_db"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(global_offset)
    return out


def _apply_offset_and_replacement(surface: pd.DataFrame, offsets: pd.DataFrame, dt_with_grid: pd.DataFrame) -> pd.DataFrame:
    out = surface.merge(offsets[["strict_cell_key", "offset_db", "offset_source", "dt_count"]], on="strict_cell_key", how="left")
    out["offset_db"] = pd.to_numeric(out["offset_db"], errors="coerce").fillna(0.0)
    out["offset_corrected_rsrp"] = (out["plain_p1812_rsrp"] + out["offset_db"]).clip(*CLIP_RSRP)
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
    out["grid_id"] = out["grid_id"].astype(str)
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
    for label, values, color in series_map:
        x, y = _cdf_values(values)
        if len(x) == 0:
            continue
        ax.plot(x, y, linewidth=2.2, label=f"{label} (n={len(x):,})", color=color)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("RSRP (dBm)")
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.set_xlim(CLIP_RSRP[0], CLIP_RSRP[1])
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_outputs(surface: pd.DataFrame, dt_with_grid: pd.DataFrame, offsets: pd.DataFrame) -> None:
    _plot_cdf(
        [
            ("Plain P1812 before - complete polygon", surface["plain_p1812_rsrp"], "#7c3aed"),
            ("P1812 after offset + DT replacement", surface["corrected_rsrp"], "#168a52"),
        ],
        f"Project {PROJECT_ID} P1812 CDF - Complete Polygon, {surface['strict_cell_key'].nunique()} Cells",
        COMBINED_DIR / "cdf_complete_polygon_p1812_before_after_offset.png",
    )
    _plot_cdf(
        [
            ("Plain P1812 before at DT", dt_with_grid["plain_p1812_at_dt_rsrp"], "#7c3aed"),
            ("After at DT pixels", dt_with_grid["after_at_dt_pixel_rsrp"], "#168a52"),
            ("DT measured", dt_with_grid["rsrp_measured"], "#2563eb"),
        ],
        f"Project {PROJECT_ID} P1812 CDF - DT Locations",
        COMBINED_DIR / "cdf_dt_locations_p1812_before_after_offset.png",
    )
    _plot_cdf(
        [("Cell median offsets", offsets["offset_db"], "#805ad5")],
        f"Project {PROJECT_ID} P1812 Per-Cell Offset CDF",
        COMBINED_DIR / "cdf_per_cell_p1812_offset_db.png",
    )
    for idx, (cell_key, cell_df) in enumerate(surface.groupby("strict_cell_key", sort=True), start=1):
        dt_cell = dt_with_grid.loc[dt_with_grid["assigned_strict_cell_key"].astype(str) == str(cell_key)]
        first = cell_df.iloc[0]
        series = [
            ("P1812 before", cell_df["plain_p1812_rsrp"], "#7c3aed"),
            ("After offset + DT replacement", cell_df["corrected_rsrp"], "#168a52"),
        ]
        if not dt_cell.empty:
            series.append(("DT measured assigned to this cell", dt_cell["rsrp_measured"], "#2563eb"))
        _plot_cdf(
            series,
            f"Cell {idx:03d}: site {first['site']} sector {first['sector']} band {first['band']} offset {float(first['offset_db']):+.2f} dB",
            INDIVIDUAL_DIR
            / f"{idx:03d}_site_{_safe_token(first['site'])}_sector_{_safe_token(first['sector'])}_band_{_safe_token(first['band'])}_p1812_cdf.png",
        )
        total_cells = int(surface["strict_cell_key"].nunique())
        if idx == 1 or idx % 20 == 0 or idx == total_cells:
            print(f"[PLOT] individual_p1812_cdf_done={idx}/{total_cells}")


def _save_frame(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception as exc:
        path.with_suffix(".parquet.error.txt").write_text(str(exc), encoding="utf-8")


def _comparison_plots(p1812_surface: pd.DataFrame, p1812_dt: pd.DataFrame) -> dict:
    cost_dir = DATA_DIR / "cost231"
    cost_surface_path = cost_dir / f"cost231_offset_corrected_surface_project{PROJECT_ID}.parquet"
    cost_dt_path = cost_dir / f"cost231_dt_match_project{PROJECT_ID}.parquet"
    if not cost_surface_path.exists() or not cost_dt_path.exists():
        return {"comparison_available": False}
    cost_surface = pd.read_parquet(cost_surface_path)
    cost_dt = pd.read_parquet(cost_dt_path)
    _plot_cdf(
        [
            ("Cost231 before", cost_surface["raw_cost231_rsrp"], "#d94f3d"),
            ("P1812 before", p1812_surface["plain_p1812_rsrp"], "#7c3aed"),
            ("Cost231 after", cost_surface["corrected_rsrp"], "#f97316"),
            ("P1812 after", p1812_surface["corrected_rsrp"], "#168a52"),
        ],
        f"Project {PROJECT_ID} Complete Polygon - Cost231 vs P1812",
        COMPARISON_DIR / "cdf_complete_polygon_cost231_vs_p1812.png",
    )
    _plot_cdf(
        [
            ("DT measured", cost_dt["rsrp_measured"], "#2563eb"),
            ("Cost231 before at DT", cost_dt["raw_cost231_at_dt_rsrp"], "#d94f3d"),
            ("P1812 before at DT", p1812_dt["plain_p1812_at_dt_rsrp"], "#7c3aed"),
            ("Cost231 after at DT", cost_dt["after_at_dt_pixel_rsrp"], "#f97316"),
            ("P1812 after at DT", p1812_dt["after_at_dt_pixel_rsrp"], "#168a52"),
        ],
        f"Project {PROJECT_ID} DT Locations - Cost231 vs P1812",
        COMPARISON_DIR / "cdf_dt_locations_cost231_vs_p1812.png",
    )
    metrics = []
    for model, df, before_col, after_col in [
        ("cost231", cost_dt, "raw_cost231_at_dt_rsrp", "after_at_dt_pixel_rsrp"),
        ("p1812", p1812_dt, "plain_p1812_at_dt_rsrp", "after_at_dt_pixel_rsrp"),
    ]:
        for stage, pred_col in [("before", before_col), ("after", after_col)]:
            err = pd.to_numeric(df[pred_col], errors="coerce") - pd.to_numeric(df["rsrp_measured"], errors="coerce")
            err = err.dropna()
            metrics.append(
                {
                    "model": model,
                    "stage": stage,
                    "rows": int(len(err)),
                    "mae": float(err.abs().mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(err)))),
                    "bias_pred_minus_dt": float(err.mean()),
                }
            )
    metrics_df = pd.DataFrame(metrics)
    _save_frame(metrics_df, COMPARISON_DIR / "cost231_vs_p1812_dt_error_metrics")
    return {"comparison_available": True, "metrics": metrics}


def main() -> None:
    _ensure_dirs()
    print(f"[START] project={PROJECT_ID} p1812_test project_dir={PROJECT_DIR}")
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
    print(f"[INPUT] strict_cells={len(site_df)} dt_rows={len(drive_df)} grid_pixels={len(grid_df)} dem={DEM_PATH}")
    _save_frame(site_df, INPUT_DIR / "p1812_input_sites_102")
    _save_frame(grid_df, INPUT_DIR / "p1812_input_grid_polygon")
    _save_frame(drive_df, INPUT_DIR / "p1812_input_dt")

    surface = _run_plain_p1812(site_df, grid_df)
    dt_assigned = _run_p1812_at_dt(site_df, drive_df)
    dt_with_grid = _attach_nearest_grid(dt_assigned, grid_df)
    offsets = _offset_table(site_df, dt_with_grid)
    surface = _apply_offset_and_replacement(surface, offsets, dt_with_grid)

    dt_after_lookup = surface[
        ["strict_cell_key", "grid_id", "plain_p1812_rsrp", "corrected_rsrp", "offset_corrected_rsrp"]
    ].rename(
        columns={
            "strict_cell_key": "assigned_strict_cell_key",
            "grid_id": "nearest_grid_id",
            "plain_p1812_rsrp": "plain_p1812_nearest_grid_rsrp",
            "corrected_rsrp": "after_at_dt_pixel_rsrp",
            "offset_corrected_rsrp": "after_offset_only_nearest_grid_rsrp",
        }
    )
    dt_with_grid = dt_with_grid.merge(dt_after_lookup, on=["assigned_strict_cell_key", "nearest_grid_id"], how="left")

    _plot_outputs(surface, dt_with_grid, offsets)
    _save_frame(surface, P1812_DIR / f"p1812_offset_corrected_surface_project{PROJECT_ID}")
    _save_frame(dt_with_grid, P1812_DIR / f"p1812_dt_match_project{PROJECT_ID}")
    _save_frame(offsets, P1812_DIR / f"p1812_offsets_cells_project{PROJECT_ID}")
    comparison = _comparison_plots(surface, dt_with_grid)

    summary = {
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "project_id": PROJECT_ID,
        "production_code_modified": False,
        "production_baseline_run": False,
        "model": "plain_p1812_terrain_profile_test",
        "model_note": (
            "Test-local P.1812-family terrain/profile run: free-space loss plus DEM path diffraction, "
            "using the same cellular antenna/tx assumptions as the Cost231 test. No production code or DB write."
        ),
        "dem_path": str(DEM_PATH),
        "strict_cells": int(len(site_df)),
        "grid_pixels": int(len(grid_df)),
        "surface_rows": int(len(surface)),
        "dt_rows": int(len(drive_df)),
        "dt_replacement_radius_m": DT_REPLACE_RADIUS_M,
        "dt_replacement_rows": int(surface["dt_replaced"].sum()),
        "cells_with_dt_offset": int((offsets["dt_count"] > 0).sum()),
        "cells_using_global_offset": int((offsets["dt_count"] == 0).sum()),
        "global_offset_db": float(dt_with_grid["dt_minus_p1812_db"].median()) if not dt_with_grid.empty else 0.0,
        "plain_p1812_rsrp": {
            "min": float(surface["plain_p1812_rsrp"].min()),
            "max": float(surface["plain_p1812_rsrp"].max()),
            "mean": float(surface["plain_p1812_rsrp"].mean()),
        },
        "corrected_rsrp": {
            "min": float(surface["corrected_rsrp"].min()),
            "max": float(surface["corrected_rsrp"].max()),
            "mean": float(surface["corrected_rsrp"].mean()),
        },
        "comparison": comparison,
        "p1812_dir": str(P1812_DIR.relative_to(THIS_DIR)),
        "comparison_dir": str(COMPARISON_DIR.relative_to(THIS_DIR)),
    }
    (P1812_DIR / "p1812_offset_cdf_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
