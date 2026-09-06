"""
Phase 22: add one DEM path-profile terrain diffraction term to the
Phase 17/20 Cost231 + building/clutter workflow.

This is test-only. It does not modify production or previous phase outputs.

Principle:
  - Cost231 remains the base RF model.
  - Existing Phase 17/19 building branch logic remains the building/clutter
    obstruction model.
  - Terrain is added as one explicit path-profile diffraction component:
      terrain_physical = cost231 + building_branch_correction - terrain_loss
  - Terrain loss is computed from the DEM profile using the dominant
    Fresnel-Kirchhoff/P.526-style knife-edge parameter. Fresnel clearance is
    an input to that formula, not a standalone binary loss switch.

Outputs compare, per technology:
  - before terrain
  - after terrain
  - terrain loss distribution
  - DT CDF and full-polygon CDF plots
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
for path in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit_project210_phase13_beam_check as phase13
import test_project210_phase17_full_polygon_geo_dt_comparison as phase17
import test_project210_phase19_branch_calibrated_comparison as phase19
from phase_rsrp_guard import display_rsrp, valid_model_rsrp


PROJECT_ID = 210
PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE20_DIR = PROJECT_DIR / "cost231_phase20_5g_real_dt_match"
OUT_DIR = PROJECT_DIR / "cost231_phase22_terrain_diffraction_comparison"
IMAGE_DIR = OUT_DIR / "images"

MAPDATA_ROOT = (
    THIS_DIR
    / "data"
    / "mapdata"
    / "Dno19_0095_NewTaipeiCity_5m"
    / "Dno19_0095_NewTaipeiCity_5m"
    / "New_TaipeiCity_5m_UTM51N_planet"
)
DEM_PATH = MAPDATA_ROOT / "Heights" / "height_5m.grd"

RSRP_MIN, RSRP_MAX = phase17.RSRP_MIN, phase17.RSRP_MAX
UE_HEIGHT_M = 1.5
MIN_DT_FOR_REPRESENTATIVE_CLASS = phase17.MIN_DT_FOR_REPRESENTATIVE_CLASS
# Clutter classes that are not a real serving environment (no users) and whose drive-test
# samples are unreliable (GPS drift onto water polygons, bridge/causeway crossings, coastline).
# Excluded from calibration fitting: these cells fall back to bias 0 and report the physical value.
NON_SERVING_CLUTTER_FOR_CALIBRATION = {"Water"}
DEM_ELEVATION_MIN_M = -500.0
DEM_ELEVATION_MAX_M = 9000.0


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _read_frame(stem: Path) -> pd.DataFrame:
    parquet_path = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:
            if not csv_path.exists():
                raise
            print(f"[PHASE22] parquet read failed for {parquet_path.name}; using CSV fallback: {exc}")
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"No parquet/csv found for {stem}")


class TerrainSampler:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        self.dataset = rasterio.open(path)
        if self.dataset.crs is None:
            raise ValueError(f"DEM has no CRS: {path}")
        self.to_dem = Transformer.from_crs("EPSG:4326", self.dataset.crs, always_xy=True)
        self.band_index, self.band_stats = self._select_elevation_band()
        self.nodata = self.dataset.nodatavals[self.band_index - 1]
        self.band = self.dataset.read(self.band_index, masked=True)
        print(
            "[PHASE22][DEM] selected elevation raster band "
            f"{self.band_index}/{self.dataset.count}: {self.band_stats}",
            flush=True,
        )

    def _band_values(self, band_index: int) -> np.ndarray:
        band = self.dataset.read(band_index, masked=True)
        values = band.compressed() if np.ma.isMaskedArray(band) else band.reshape(-1)
        values = values.astype(float, copy=False)
        nodata = self.dataset.nodatavals[band_index - 1]
        if nodata is not None:
            values = values[~np.isclose(values, float(nodata), rtol=0.0, atol=1e-9)]
        values = values[np.isfinite(values)]
        if values.size > 500_000:
            step = int(math.ceil(values.size / 500_000))
            values = values[::step]
        return values

    def _score_band(self, band_index: int) -> tuple[float, dict]:
        values = self._band_values(band_index)
        if values.size == 0:
            return -1_000.0, {"reason": "empty_or_nodata"}

        p = np.percentile(values, [0, 1, 5, 50, 95, 99, 100])
        dtype = np.dtype(self.dataset.dtypes[band_index - 1])
        unique_count = int(len(np.unique(values[: min(values.size, 100_000)])))
        endpoint_share = float(np.mean(np.isclose(values, 0.0) | np.isclose(values, 255.0)))
        dynamic_range = float(p[5] - p[1])
        plausible_elevation = (
            DEM_ELEVATION_MIN_M <= p[1] <= DEM_ELEVATION_MAX_M
            and DEM_ELEVATION_MIN_M <= p[3] <= DEM_ELEVATION_MAX_M
            and p[5] <= DEM_ELEVATION_MAX_M
            and dynamic_range >= 1.0
        )
        display_like = (
            dtype == np.dtype("uint8")
            and 0.0 <= p[0] <= 255.0
            and 0.0 <= p[6] <= 255.0
            and endpoint_share > 0.25
        )

        score = 0.0
        if plausible_elevation:
            score += 20.0
        if np.issubdtype(dtype, np.floating):
            score += 12.0
        elif dtype.itemsize > 1 or np.issubdtype(dtype, np.signedinteger):
            score += 6.0
        score += min(dynamic_range / 20.0, 8.0)
        if display_like:
            score -= 30.0
        if unique_count < 32:
            score -= 10.0

        stats = {
            "dtype": str(dtype),
            "p0": round(float(p[0]), 3),
            "p1": round(float(p[1]), 3),
            "p5": round(float(p[2]), 3),
            "p50": round(float(p[3]), 3),
            "p95": round(float(p[4]), 3),
            "p99": round(float(p[5]), 3),
            "p100": round(float(p[6]), 3),
            "unique_sample": unique_count,
            "endpoint_0_255_share": round(endpoint_share, 4),
            "display_like": display_like,
            "plausible_elevation": plausible_elevation,
            "score": round(score, 3),
        }
        return score, stats

    def _select_elevation_band(self) -> tuple[int, dict]:
        scored = []
        for band_index in range(1, self.dataset.count + 1):
            score, stats = self._score_band(band_index)
            scored.append((score, band_index, stats))
            print(f"[PHASE22][DEM] band {band_index} candidate: {stats}", flush=True)

        scored.sort(reverse=True, key=lambda item: item[0])
        best_score, best_band, best_stats = scored[0]
        if best_score < 5.0 or not best_stats.get("plausible_elevation", False):
            raise ValueError(
                "No plausible DEM elevation band found. "
                f"Candidates: {[(idx, stats) for _score, idx, stats in scored]}"
            )
        return best_band, best_stats

    def close(self) -> None:
        self.dataset.close()

    def sample(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x, y = self.to_dem.transform(lon, lat)
        rows, cols = rowcol(self.dataset.transform, x, y)
        values = np.full(len(rows), np.nan, dtype=float)
        for idx, (row, col) in enumerate(zip(rows, cols)):
            if row < 0 or col < 0 or row >= self.dataset.height or col >= self.dataset.width:
                continue
            value = self.band[row, col]
            if np.ma.is_masked(value):
                continue
            value = float(value)
            if self.nodata is not None and value == self.nodata:
                continue
            if np.isfinite(value):
                values[idx] = value
        return values


def _haversine_m(lat1, lon1, lat2, lon2):
    radius_m = 6_371_000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_m * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _terrain_diffraction_loss_one(
    dem: TerrainSampler,
    site_lat: float,
    site_lon: float,
    site_ground_m: float,
    antenna_height_m: float,
    rx_lat: float,
    rx_lon: float,
    rx_ground_m: float,
    freq_mhz: float,
) -> tuple[float, float, float, float]:
    distance_m = float(_haversine_m(site_lat, site_lon, rx_lat, rx_lon))
    if distance_m < 1.0:
        return 0.0, distance_m, 0.0, -999.0

    sample_count = int(np.clip(math.ceil(distance_m / 25.0), 7, 96))
    fractions = np.linspace(0.0, 1.0, sample_count)
    lats = site_lat + (rx_lat - site_lat) * fractions
    lons = site_lon + (rx_lon - site_lon) * fractions
    terrain = dem.sample(lats, lons)
    if np.isnan(terrain).all():
        return 0.0, distance_m, 0.0, -999.0

    terrain = pd.Series(terrain).interpolate(limit_direction="both").to_numpy(dtype=float)
    tx_alt_m = float(site_ground_m) + float(antenna_height_m)
    rx_alt_m = float(rx_ground_m) + UE_HEIGHT_M
    los_alt_m = tx_alt_m + (rx_alt_m - tx_alt_m) * fractions
    clearance_m = terrain - los_alt_m
    inner = clearance_m[1:-1]
    if inner.size == 0 or not np.isfinite(inner).any():
        return 0.0, distance_m, 0.0, -999.0

    inner_idx = int(np.nanargmax(inner)) + 1
    h_m = float(clearance_m[inner_idx])
    d1_m = max(distance_m * float(fractions[inner_idx]), 1.0)
    d2_m = max(distance_m - d1_m, 1.0)
    wavelength_m = 300.0 / max(float(freq_mhz), 1.0)
    v = h_m * math.sqrt(2.0 * (d1_m + d2_m) / (wavelength_m * d1_m * d2_m))
    if v <= -0.78:
        return 0.0, distance_m, h_m, v

    loss_db = 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return float(np.clip(loss_db, 0.0, 45.0)), distance_m, h_m, v


def _terrain_losses_for_group(dem: TerrainSampler, group: pd.DataFrame) -> pd.DataFrame:
    row0 = group.iloc[0]
    site_lat = float(row0["site_lat"])
    site_lon = float(row0["site_lon"])
    site_ground = float(dem.sample(np.array([site_lat]), np.array([site_lon]))[0])
    if not np.isfinite(site_ground):
        site_ground = 0.0

    target_lat = group["lat"].to_numpy(dtype=float)
    target_lon = group["lon"].to_numpy(dtype=float)
    target_ground = dem.sample(target_lat, target_lon)
    target_ground = np.where(np.isfinite(target_ground), target_ground, site_ground)

    losses = np.zeros(len(group), dtype=float)
    distances = np.zeros(len(group), dtype=float)
    max_obstruction = np.zeros(len(group), dtype=float)
    knife_v = np.full(len(group), -999.0, dtype=float)
    freq = float(row0.get("frequency_mhz", 1800.0) or 1800.0)
    antenna_height = float(row0.get("Height", 30.0) or 30.0)

    for i, (lat, lon, ground) in enumerate(zip(target_lat, target_lon, target_ground)):
        losses[i], distances[i], max_obstruction[i], knife_v[i] = _terrain_diffraction_loss_one(
            dem,
            site_lat,
            site_lon,
            site_ground,
            antenna_height,
            float(lat),
            float(lon),
            float(ground),
            freq,
        )

    out = pd.DataFrame(index=group.index)
    out["site_ground_elevation_m"] = site_ground
    out["rx_ground_elevation_m"] = target_ground
    out["terrain_path_distance_m"] = distances
    out["terrain_max_obstruction_m"] = max_obstruction
    out["terrain_knife_edge_v"] = knife_v
    out["terrain_diffraction_loss_db"] = losses
    out["terrain_obstructed"] = losses > 0.1
    return out


def _frequency_lookup(surface: pd.DataFrame) -> pd.DataFrame:
    return surface[["strict_cell_key", "frequency_mhz", "band"]].drop_duplicates("strict_cell_key")


def _score_points(
    points: pd.DataFrame,
    identity: pd.DataFrame,
    clutter_gdf,
    buildings_gdf,
    dem: TerrainSampler,
    key_col: str,
    raw_col: str,
) -> pd.DataFrame:
    meta_cols = ["Node_Cell_ID", "lat", "lon", "azimuth", "Etilt", "Mtilt", "Height", "tx_power"]
    site_meta = identity[meta_cols].rename(columns={"lat": "site_lat", "lon": "site_lon"})
    out = points.merge(site_meta, left_on=key_col, right_on="Node_Cell_ID", how="left")
    out["building_geo_correction_db"] = 0.0
    out["obstruction_branch"] = "unknown"
    out["clutter_class"] = None

    n_cells = out[key_col].nunique(dropna=True)
    terrain_frames = []
    for idx, (cell_key, group) in enumerate(out.groupby(key_col, dropna=False)):
        if group.empty or pd.isna(group.iloc[0].get("site_lat")):
            continue
        row0 = group.iloc[0]
        grid_df = group[["lat", "lon"]].reset_index(drop=True)
        correction, branch, clutter = phase19._geo_correction_with_branch(
            grid_df,
            clutter_gdf,
            buildings_gdf,
            center_lat=float(row0["site_lat"]),
            center_lon=float(row0["site_lon"]),
            tx_height_m=float(row0.get("Height", 30.0) or 30.0),
            rx_height_m=UE_HEIGHT_M,
            freq_mhz=float(row0.get("frequency_mhz", 1800.0) or 1800.0),
        )
        out.loc[group.index, "building_geo_correction_db"] = correction
        out.loc[group.index, "obstruction_branch"] = branch
        out.loc[group.index, "clutter_class"] = clutter
        terrain_frames.append(_terrain_losses_for_group(dem, group))
        if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == n_cells:
            print(
                f"[PHASE22][SCORE] cells_done={idx + 1}/{n_cells} "
                f"rows={int(sum(len(frame) for frame in terrain_frames))}",
                flush=True,
            )

    if terrain_frames:
        terrain_df = pd.concat(terrain_frames).sort_index()
        out = out.join(terrain_df)
    else:
        for col in [
            "site_ground_elevation_m",
            "rx_ground_elevation_m",
            "terrain_path_distance_m",
            "terrain_max_obstruction_m",
            "terrain_knife_edge_v",
            "terrain_diffraction_loss_db",
        ]:
            out[col] = np.nan
        out["terrain_obstructed"] = False

    raw = pd.to_numeric(out[raw_col], errors="coerce")
    building = pd.to_numeric(out["building_geo_correction_db"], errors="coerce").fillna(0.0)
    terrain = pd.to_numeric(out["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0)
    out["phase22_physical_no_terrain_rsrp_unclipped"] = raw + building
    out["phase22_physical_with_terrain_rsrp_unclipped"] = raw + building - terrain
    out["phase22_physical_no_terrain_rsrp"] = valid_model_rsrp(
        out["phase22_physical_no_terrain_rsrp_unclipped"]
    )
    out["phase22_physical_with_terrain_rsrp"] = valid_model_rsrp(
        out["phase22_physical_with_terrain_rsrp_unclipped"]
    )
    return out


def _bias_table(dt: pd.DataFrame, residual_col: str) -> pd.DataFrame:
    work = dt.dropna(subset=[residual_col, "assigned_technology", "clutter_class", "obstruction_branch"]).copy()
    # Non-serving clutter (Water) is not calibrated - its DT residuals are noise, not a real bias.
    work = work[~work["clutter_class"].astype(str).isin(NON_SERVING_CLUTTER_FOR_CALIBRATION)].copy()
    table = (
        work.groupby(["assigned_technology", "clutter_class", "obstruction_branch"], dropna=False)
        .agg(n=(residual_col, "size"), bias_db=(residual_col, "median"))
        .reset_index()
        .rename(columns={"assigned_technology": "technology"})
    )
    table = table[table["n"] >= MIN_DT_FOR_REPRESENTATIVE_CLASS].copy()
    return table


def _attach_bias(df: pd.DataFrame, bias: pd.DataFrame, out_col: str) -> pd.DataFrame:
    out = df.merge(
        bias.rename(columns={"bias_db": out_col, "n": f"{out_col}_n"}),
        on=["technology", "clutter_class", "obstruction_branch"],
        how="left",
    )
    out[out_col] = pd.to_numeric(out[out_col], errors="coerce").fillna(0.0)
    out[f"{out_col}_n"] = pd.to_numeric(out[f"{out_col}_n"], errors="coerce").fillna(0).astype(int)
    return out


def _corrected_dt_replacements(dt: pd.DataFrame) -> pd.DataFrame:
    return (
        dt.loc[dt["dt_replacement_eligible"].fillna(False).astype(bool)]
        .groupby(["assigned_technology", "nearest_grid_id"], dropna=False)
        .agg(dt_replacement_rsrp=("rsrp_measured", "mean"), dt_replacement_count=("rsrp_measured", "size"))
        .reset_index()
        .rename(columns={"assigned_technology": "technology", "nearest_grid_id": "grid_id"})
    )


def _aggregate_by_grid(candidates: pd.DataFrame, replacements: pd.DataFrame) -> pd.DataFrame:
    stale_replacement_cols = [
        col
        for col in ["dt_replacement_rsrp", "dt_replacement_count"]
        if col in candidates.columns
    ]
    out = candidates.drop(columns=stale_replacement_cols).merge(
        replacements,
        on=["technology", "grid_id"],
        how="left",
    )
    lock = out["dt_replacement_rsrp"].notna()

    out["phase22_no_terrain_calibrated_no_lock_unclipped"] = (
        out["phase22_physical_no_terrain_rsrp"] + out["phase22_phase19_bias_db"]
    )
    out["phase22_with_terrain_calibrated_no_lock_unclipped"] = (
        out["phase22_physical_with_terrain_rsrp"] + out["phase22_phase19_bias_db"]
    )
    out["phase22_no_terrain_calibrated_no_lock"] = valid_model_rsrp(
        out["phase22_no_terrain_calibrated_no_lock_unclipped"]
    )
    out["phase22_with_terrain_calibrated_no_lock"] = valid_model_rsrp(
        out["phase22_with_terrain_calibrated_no_lock_unclipped"]
    )
    out["phase22_no_terrain_calibrated_rsrp"] = out["phase22_no_terrain_calibrated_no_lock"].where(
        ~lock, out["dt_replacement_rsrp"]
    )
    out["phase22_with_terrain_calibrated_rsrp"] = out["phase22_with_terrain_calibrated_no_lock"].where(
        ~lock, out["dt_replacement_rsrp"]
    )
    out["phase22_no_terrain_calibrated_rsrp"] = display_rsrp(out["phase22_no_terrain_calibrated_rsrp"])
    out["phase22_with_terrain_calibrated_rsrp"] = display_rsrp(out["phase22_with_terrain_calibrated_rsrp"])

    agg_specs = {
        "phase22_physical_no_terrain_rsrp": ["max", "mean"],
        "phase22_physical_with_terrain_rsrp": ["max", "mean"],
        "phase22_no_terrain_calibrated_rsrp": ["max", "mean"],
        "phase22_with_terrain_calibrated_rsrp": ["max", "mean"],
        "terrain_diffraction_loss_db": ["mean", "max"],
        "building_geo_correction_db": ["mean"],
        "terrain_obstructed": ["mean"],
    }
    agg = out.groupby(["technology", "grid_id"], dropna=False).agg(agg_specs)
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    agg = agg.rename(
        columns={
            "phase22_physical_no_terrain_rsrp_max": "phase22_physical_no_terrain_best_rsrp",
            "phase22_physical_no_terrain_rsrp_mean": "phase22_physical_no_terrain_mean_rsrp",
            "phase22_physical_with_terrain_rsrp_max": "phase22_physical_with_terrain_best_rsrp",
            "phase22_physical_with_terrain_rsrp_mean": "phase22_physical_with_terrain_mean_rsrp",
            "phase22_no_terrain_calibrated_rsrp_max": "phase22_no_terrain_best_rsrp",
            "phase22_no_terrain_calibrated_rsrp_mean": "phase22_no_terrain_mean_rsrp",
            "phase22_with_terrain_calibrated_rsrp_max": "phase22_with_terrain_best_rsrp",
            "phase22_with_terrain_calibrated_rsrp_mean": "phase22_with_terrain_mean_rsrp",
            "terrain_obstructed_mean": "terrain_obstructed_share",
        }
    )
    return agg, out


def _cdf_values(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    if arr.size == 0:
        return arr, arr
    return arr, np.arange(1, arr.size + 1, dtype=float) / arr.size * 100.0


def _plot_cdf(series_map: list[tuple[str, pd.Series, str]], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values, color in series_map:
        x, y = _cdf_values(values)
        ax.plot(x, y, label=f"{label} (n={len(x):,})", color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("RSRP / loss (dB)")
    ax.set_ylabel("Cumulative %")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_frame(df: pd.DataFrame, stem: Path) -> None:
    df.to_parquet(stem.with_suffix(".parquet"), index=False)
    df.to_csv(stem.with_suffix(".csv"), index=False)


def main() -> None:
    _ensure_dirs()
    print(f"[PHASE22] DEM path: {DEM_PATH}")

    surface = _read_frame(PHASE9_DIR / "phase9_directional_raw_corrected_surface_project210")
    grid = _read_frame(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210")
    dt_source = (
        PHASE20_DIR / "phase9_dt_match_project210_corrected"
        if (PHASE20_DIR / "phase9_dt_match_project210_corrected.parquet").exists()
        else PHASE9_DIR / "phase9_dt_match_project210"
    )
    dt = _read_frame(dt_source)
    identity = phase13.load_identity()
    clutter_gdf, buildings_gdf = phase17._load_clutter_and_buildings()
    dt = phase17._classify_dt_clutter(dt, clutter_gdf)

    freq_lookup = _frequency_lookup(surface)
    dt_points = dt.merge(freq_lookup, left_on="assigned_strict_cell_key", right_on="strict_cell_key", how="left")
    dt_points["technology"] = dt_points["assigned_technology"]

    dem = TerrainSampler(DEM_PATH)
    try:
        print(f"[PHASE22] surface rows={len(surface)} dt rows={len(dt_points)} identity rows={len(identity)}")
        candidates = _score_points(
            surface,
            identity,
            clutter_gdf,
            buildings_gdf,
            dem,
            key_col="strict_cell_key",
            raw_col="raw_cost231_rsrp",
        )
        dt_scored = _score_points(
            dt_points,
            identity,
            clutter_gdf,
            buildings_gdf,
            dem,
            key_col="assigned_strict_cell_key",
            raw_col="raw_cost231_at_dt_rsrp",
        )
    finally:
        dem.close()

    dt_scored["dt_minus_no_terrain_physical_db"] = (
        dt_scored["rsrp_measured"] - dt_scored["phase22_physical_no_terrain_rsrp"]
    )
    dt_scored["dt_minus_with_terrain_physical_db"] = (
        dt_scored["rsrp_measured"] - dt_scored["phase22_physical_with_terrain_rsrp"]
    )

    phase19_style_bias = _bias_table(dt_scored, "dt_minus_no_terrain_physical_db")
    phase19_style_bias["bias_source"] = "no_terrain_phase19_style"
    phase19_style_bias.to_csv(OUT_DIR / "phase22_phase19_style_bias_by_condition.csv", index=False)
    phase19_style_bias.to_csv(OUT_DIR / "phase22_no_terrain_bias_by_condition.csv", index=False)
    phase19_style_bias.to_csv(OUT_DIR / "phase22_with_terrain_bias_by_condition.csv", index=False)

    candidates = _attach_bias(candidates, phase19_style_bias, "phase22_phase19_bias_db")
    replacements = _corrected_dt_replacements(dt)
    grid_agg, scored_candidates = _aggregate_by_grid(candidates, replacements)
    scored_candidates["phase22_terrain_delta_db"] = (
        scored_candidates["phase22_physical_with_terrain_rsrp"]
        - scored_candidates["phase22_physical_no_terrain_rsrp"]
    )

    grid_bounds = grid[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]].copy()
    all_tech_grid = pd.concat(
        [grid_bounds[["grid_id"]].assign(technology=technology) for technology in ["4G", "5G"]],
        ignore_index=True,
    )
    grid_agg = all_tech_grid.merge(grid_agg, on=["technology", "grid_id"], how="left")
    serving_outputs = {}
    summary = {}
    for technology in ["4G", "5G"]:
        serving = grid_agg[grid_agg["technology"].astype(str) == technology].merge(grid_bounds, on="grid_id", how="left")
        out_stem = OUT_DIR / f"phase22_serving_grid_{technology.lower()}_project210"
        _save_frame(serving, out_stem)
        serving_outputs[technology] = serving

        dt_tech = dt_scored[dt_scored["assigned_technology"].astype(str) == technology].copy()
        dt_tech = _attach_bias(dt_tech, phase19_style_bias, "phase22_phase19_bias_db")
        dt_tech["phase22_no_terrain_calibrated_rsrp_unclipped"] = (
            dt_tech["phase22_physical_no_terrain_rsrp"] + dt_tech["phase22_phase19_bias_db"]
        )
        dt_tech["phase22_with_terrain_calibrated_rsrp_unclipped"] = (
            dt_tech["phase22_physical_with_terrain_rsrp"] + dt_tech["phase22_phase19_bias_db"]
        )
        dt_tech["phase22_no_terrain_calibrated_rsrp"] = valid_model_rsrp(
            dt_tech["phase22_no_terrain_calibrated_rsrp_unclipped"]
        )
        dt_tech["phase22_with_terrain_calibrated_rsrp"] = valid_model_rsrp(
            dt_tech["phase22_with_terrain_calibrated_rsrp_unclipped"]
        )

        _plot_cdf(
            [
                ("Physical before terrain", serving["phase22_physical_no_terrain_best_rsrp"], "#ef4444"),
                ("Physical after terrain", serving["phase22_physical_with_terrain_best_rsrp"], "#2563eb"),
                ("Phase19-style calibrated before terrain", serving["phase22_no_terrain_best_rsrp"], "#f97316"),
                ("Same bias + terrain", serving["phase22_with_terrain_best_rsrp"], "#16a34a"),
            ],
            f"Project 210 {technology}: full polygon terrain comparison",
            IMAGE_DIR / f"phase22_{technology.lower()}_full_polygon_cdf.png",
        )
        _plot_cdf(
            [
                ("DT measured", dt_tech["rsrp_measured"], "#111827"),
                ("Before terrain at DT", dt_tech["phase22_physical_no_terrain_rsrp"], "#ef4444"),
                ("After terrain at DT", dt_tech["phase22_physical_with_terrain_rsrp"], "#2563eb"),
                ("Phase19-style calibrated before terrain at DT", dt_tech["phase22_no_terrain_calibrated_rsrp"], "#f97316"),
                ("Same bias + terrain at DT", dt_tech["phase22_with_terrain_calibrated_rsrp"], "#16a34a"),
            ],
            f"Project 210 {technology}: DT-location terrain comparison",
            IMAGE_DIR / f"phase22_{technology.lower()}_dt_cdf.png",
        )
        _plot_cdf(
            [("Terrain diffraction loss", serving["terrain_diffraction_loss_db_mean"], "#7c3aed")],
            f"Project 210 {technology}: terrain diffraction loss",
            IMAGE_DIR / f"phase22_{technology.lower()}_terrain_loss_cdf.png",
        )

        terrain_loss = pd.to_numeric(serving["terrain_diffraction_loss_db_mean"], errors="coerce").fillna(0.0)
        summary[technology] = {
            "grid_rows": int(len(serving)),
            "dt_rows": int(len(dt_tech)),
            "mean_physical_before_terrain_best_rsrp": float(serving["phase22_physical_no_terrain_best_rsrp"].mean()),
            "mean_physical_after_terrain_best_rsrp": float(serving["phase22_physical_with_terrain_best_rsrp"].mean()),
            "mean_phase19_style_calibrated_before_terrain_best_rsrp": float(serving["phase22_no_terrain_best_rsrp"].mean()),
            "mean_calibrated_after_terrain_best_rsrp": float(serving["phase22_with_terrain_best_rsrp"].mean()),
            "mean_terrain_shift_physical_db": float(
                (
                    serving["phase22_physical_with_terrain_best_rsrp"]
                    - serving["phase22_physical_no_terrain_best_rsrp"]
                ).mean()
            ),
            "mean_terrain_shift_after_same_bias_db": float(
                (
                    serving["phase22_with_terrain_best_rsrp"]
                    - serving["phase22_no_terrain_best_rsrp"]
                ).mean()
            ),
            "terrain_loss_db": {
                "mean": float(terrain_loss.mean()),
                "p50": float(terrain_loss.quantile(0.50)),
                "p75": float(terrain_loss.quantile(0.75)),
                "p90": float(terrain_loss.quantile(0.90)),
                "max": float(terrain_loss.max()),
            },
            "terrain_obstructed_grid_share": float(serving["terrain_obstructed_share"].mean()),
            "mean_building_geo_correction_db": float(serving["building_geo_correction_db_mean"].mean()),
            "representative_phase19_style_bias_rows": int(
                len(phase19_style_bias[phase19_style_bias["technology"] == technology])
            ),
            "bias_source": "dt_minus_no_terrain_physical_db",
            "images": {
                "full_polygon_cdf": str((IMAGE_DIR / f"phase22_{technology.lower()}_full_polygon_cdf.png").relative_to(THIS_DIR)),
                "dt_cdf": str((IMAGE_DIR / f"phase22_{technology.lower()}_dt_cdf.png").relative_to(THIS_DIR)),
                "terrain_loss_cdf": str((IMAGE_DIR / f"phase22_{technology.lower()}_terrain_loss_cdf.png").relative_to(THIS_DIR)),
            },
        }
        print(f"[PHASE22] wrote {out_stem.with_suffix('.parquet')} ({len(serving)} rows)")

    keep_cols = [
        "technology",
        "grid_id",
        "strict_cell_key",
        "lat",
        "lon",
        "raw_cost231_rsrp",
        "building_geo_correction_db",
        "obstruction_branch",
        "clutter_class",
        "terrain_diffraction_loss_db",
        "terrain_max_obstruction_m",
        "terrain_knife_edge_v",
        "phase22_physical_no_terrain_rsrp_unclipped",
        "phase22_physical_with_terrain_rsrp_unclipped",
        "phase22_physical_no_terrain_rsrp",
        "phase22_physical_with_terrain_rsrp",
        "phase22_terrain_delta_db",
        "phase22_phase19_bias_db",
        "phase22_no_terrain_calibrated_rsrp",
        "phase22_with_terrain_calibrated_rsrp",
    ]
    _save_frame(scored_candidates[[col for col in keep_cols if col in scored_candidates.columns]], OUT_DIR / "phase22_scored_candidates_project210")
    _save_frame(dt_scored, OUT_DIR / "phase22_dt_terrain_scored_project210")
    (OUT_DIR / "phase22_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("[PHASE22] summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
