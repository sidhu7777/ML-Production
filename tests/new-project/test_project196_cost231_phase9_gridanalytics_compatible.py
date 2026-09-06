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
from shapely.ops import transform, unary_union
from sklearn.neighbors import BallTree

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from phase_rsrp_guard import RSRP_MAX_DBM, RSRP_NO_COVERAGE_DBM, display_rsrp, valid_model_rsrp
from tools.lte_prediction.Sector_wise_prediction_code_copy import compute_sector_rsrp


PROJECT_ID = int(os.getenv("PROP_PROJECT_ID", "196"))
PROJECT_REGION = os.getenv("PROP_REGION", "india").strip().lower()
PROJECT_SLUG = os.getenv("PROP_PROJECT_SLUG", "project_196_india")
PROJECT_DIR = Path(os.getenv("PROP_PROJECT_DIR", str(THIS_DIR / "data" / PROJECT_SLUG)))
BASELINE_SCOPE = PROJECT_DIR / "baseline_fetch_scope"
GEO_DB = PROJECT_DIR / "geo_db"
DATA_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
COMBINED_DIR = DATA_DIR / "combined"
WORK_DIR = DATA_DIR / "work"

GRID_SIZE_M = float(os.getenv("PHASE9_GRID_SIZE_M", "25"))
COVERAGE_RADIUS_M = float(os.getenv("PHASE9_COVERAGE_RADIUS_M", "500"))
MIN_CANDIDATE_RSRP_DBM = float(os.getenv("PHASE9_MIN_CANDIDATE_RSRP_DBM", str(RSRP_NO_COVERAGE_DBM)))
PHASE9_CANDIDATE_MODE = os.getenv("PHASE9_CANDIDATE_MODE", "safe_radius").strip().lower()
PHASE9_ENABLE_OUT_OF_RADIUS_BACKFILL = os.getenv("PHASE9_ENABLE_OUT_OF_RADIUS_BACKFILL", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
# Backfill keeps the k nearest sectors per uncovered cell (not just the single best-raw sector)
# so downstream phases can apply antenna/obstruction/terrain/bias per candidate and pick the
# true serving cell after all corrections. Same candidate schema as the in-radius path.
PHASE9_BACKFILL_K_NEAREST = max(1, int(float(os.getenv("PHASE9_BACKFILL_K_NEAREST", "8"))))
DT_REPLACE_RADIUS_M = float(os.getenv("PHASE9_DT_REPLACE_RADIUS_M", "25"))
EARTH_RADIUS_M = 6371000.0
CLIP_RSRP = (RSRP_NO_COVERAGE_DBM, RSRP_MAX_DBM)
ANTENNA_PATTERN_LOGIC = (
    "Cost231 helper compute_sector_rsrp uses 3GPP antenna gain: "
    "horizontal HPBW=65 deg, vertical HPBW=6 deg, max attenuation=30 dB, SLA_v=20 dB. "
    "Phase 9 applies no extra hard azimuth cutoff."
)


def _ensure_dirs() -> None:
    for path in [DATA_DIR, COMBINED_DIR, WORK_DIR]:
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
    return text.mask(text.isna() | text.eq("") | text.str.lower().isin({"nan", "none", "<na>", "null"}))


def _safe_token(value: object, fallback: str = "unknown") -> str:
    text = str(value if value is not None else fallback).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        text = fallback
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text)[:150]


def _frequency_from_site(site_df: pd.DataFrame) -> pd.Series:
    freq = pd.Series(np.nan, index=site_df.index, dtype=float)
    for col in ["frequency_mhz", "frequency", "band"]:
        if col in site_df.columns:
            candidate = pd.to_numeric(site_df[col], errors="coerce")
            freq = freq.where(pd.notna(freq), candidate)
    return freq.fillna(1800.0).clip(450.0, 3800.0)


def _technology_from_site(site_df: pd.DataFrame) -> pd.Series:
    tech = pd.Series(pd.NA, index=site_df.index, dtype="string")
    for col in ["Technology", "technology", "network_type", "rat", "tech"]:
        if col in site_df.columns:
            tech = tech.fillna(_clean_text(site_df[col]))
    band = _clean_text(site_df.get("band", pd.Series(index=site_df.index))).astype("string")
    text = tech.astype("string").str.upper()
    text = text.mask(text.str.contains("5G|NR", na=False), "5G")
    text = text.mask(text.str.contains("4G|LTE", na=False), "4G")
    text = text.mask(text.isna() & band.eq("78"), "5G")
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
    out["original_frequency_mhz"] = out["frequency_mhz"]
    out["model_rsrp_adjust_db"] = 0.0
    taiwan_n78 = out["band_key"].astype(str).eq("78") & (PROJECT_REGION == "taiwan")
    out.loc[taiwan_n78, "frequency_mhz"] = 2600.0
    out.loc[taiwan_n78, "model_rsrp_adjust_db"] = -2.58
    out = out.dropna(subset=["lat", "lon", "strict_cell_key"]).copy()
    return out.drop_duplicates(subset=["strict_cell_key"], keep="first").reset_index(drop=True)


def _prepare_dt(drive_df: pd.DataFrame) -> pd.DataFrame:
    out = drive_df.copy()
    for col in ["lat", "lon", "rsrp"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["lat", "lon", "rsrp"]).copy()
    out = out[(out["rsrp"] >= -150.0) & (out["rsrp"] <= -30.0)].copy()
    out = out.rename(columns={"rsrp": "rsrp_measured"})
    network_text = _clean_text(out.get("network", pd.Series(index=out.index))).astype("string").str.upper()
    out["measured_technology"] = np.where(network_text.str.contains("5G|NR", na=False), "5G", "4G")
    out["dt_row_id"] = np.arange(len(out))
    return out.reset_index(drop=True)


def _load_project_polygon():
    regions = _read_csv(GEO_DB / f"map_regions_project_{PROJECT_ID}_active.csv")
    if regions.empty or "region_wkt" not in regions.columns:
        raise ValueError("Project polygon not found in geo_db")
    geometries = []
    for raw in regions["region_wkt"].dropna():
        geom = wkt.loads(str(raw))
        # DB WKT is stored as lat lon; shapely expects x y, so swap to lon lat.
        geom = transform(lambda x, y, z=None: (y, x) if z is None else (y, x, z), geom)
        geometries.append(geom if geom.is_valid else geom.buffer(0))
    union = unary_union(geometries)
    return union if union.is_valid else union.buffer(0)


def _polygon_to_backend_latlon_vertices(geom) -> list[tuple[float, float]]:
    geoms = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    vertices: list[tuple[float, float]] = []
    for poly in geoms:
        for first, second in list(poly.exterior.coords):
            if 20 <= first <= 35 and 65 <= second <= 90:
                vertices.append((float(first), float(second)))
            elif 20 <= second <= 35 and 65 <= first <= 90:
                vertices.append((float(second), float(first)))
            elif -90 <= first <= 90 and -180 <= second <= 180:
                vertices.append((float(first), float(second)))
            elif -90 <= second <= 90 and -180 <= first <= 180:
                vertices.append((float(second), float(first)))
    return vertices


def _load_backend_region_vertices() -> list[tuple[float, float]]:
    regions = _read_csv(GEO_DB / f"map_regions_project_{PROJECT_ID}_active.csv")
    if regions.empty or "region_wkt" not in regions.columns:
        raise ValueError("Project polygon not found in geo_db")
    vertices: list[tuple[float, float]] = []
    for raw in regions["region_wkt"].dropna():
        vertices.extend(_polygon_to_backend_latlon_vertices(wkt.loads(str(raw))))
    vertices = [(lat, lon) for lat, lon in vertices if -90 <= lat <= 90 and -180 <= lon <= 180]
    if len(vertices) < 3:
        raise ValueError("Project polygon vertices not valid for GridAnalytics-compatible grid")
    return vertices


def _build_bounding_polygon_from_vertices(vertices: list[tuple[float, float]]) -> list[tuple[float, float]]:
    min_lat = min(lat for lat, _ in vertices)
    max_lat = max(lat for lat, _ in vertices)
    min_lon = min(lon for _, lon in vertices)
    max_lon = max(lon for _, lon in vertices)
    if abs(max_lat - min_lat) < 1e-9:
        min_lat -= 1e-6
        max_lat += 1e-6
    if abs(max_lon - min_lon) < 1e-9:
        min_lon -= 1e-6
        max_lon += 1e-6
    return [
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, max_lon),
        (max_lat, min_lon),
        (min_lat, min_lon),
    ]


def _point_in_polygon_backend(lat: float, lon: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    for i in range(len(poly)):
        j = len(poly) - 1 if i == 0 else i - 1
        yi, xi = poly[i]
        yj, xj = poly[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
    return inside


def _generate_gridanalytics_compatible_grid() -> tuple[pd.DataFrame, dict]:
    vertices = _load_backend_region_vertices()
    backend_origin_poly = _build_bounding_polygon_from_vertices(vertices)
    min_lat = min(lat for lat, _ in backend_origin_poly)
    max_lat = max(lat for lat, _ in backend_origin_poly)
    min_lon = min(lon for _, lon in backend_origin_poly)
    max_lon = max(lon for _, lon in backend_origin_poly)
    center_lat = (min_lat + max_lat) / 2.0
    lat_step = GRID_SIZE_M / 111320.0
    lon_step = GRID_SIZE_M / (111320.0 * max(abs(math.cos(math.radians(center_lat))), 1e-6))
    rows = int(math.ceil((max_lat - min_lat) / lat_step))
    cols = int(math.ceil((max_lon - min_lon) / lon_step))

    records: list[dict] = []
    for row in range(rows):
        cell_min_lat = min_lat + row * lat_step
        cell_max_lat = cell_min_lat + lat_step
        center_cell_lat = (cell_min_lat + cell_max_lat) / 2.0
        for col in range(cols):
            cell_min_lon = min_lon + col * lon_step
            cell_max_lon = cell_min_lon + lon_step
            center_cell_lon = (cell_min_lon + cell_max_lon) / 2.0
            if not _point_in_polygon_backend(center_cell_lat, center_cell_lon, vertices):
                continue
            records.append(
                {
                    "grid_id": f"R{row}C{col}",
                    "center_lat": round(center_cell_lat, 8),
                    "center_lon": round(center_cell_lon, 8),
                    "min_lat": round(cell_min_lat, 8),
                    "max_lat": round(cell_max_lat, 8),
                    "min_lon": round(cell_min_lon, 8),
                    "max_lon": round(cell_max_lon, 8),
                    "grid_size_meters": GRID_SIZE_M,
                }
            )
    grid = pd.DataFrame.from_records(records).sort_values("grid_id").reset_index(drop=True)
    if grid.empty:
        raise ValueError("GridAnalytics-compatible grid generated zero pixels")
    meta = {
        "grid_origin": "GridAnalyticsController.BuildBoundingPolygonFromVertices + GenerateGrid",
        "shape_filter": "actual active map_regions polygon center-point check",
        "source_vertices": len(vertices),
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
        "g_lat": lat_step,
        "g_lon": lon_step,
        "rows": rows,
        "cols": cols,
    }
    return grid, meta


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


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


def _site_record(row: pd.Series) -> dict:
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
    return np.minimum(np.asarray(values, dtype=float), RSRP_MAX_DBM)


def _run_directional_surface(site_df: pd.DataFrame, grid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_lat = grid_df["center_lat"].to_numpy(dtype=float)
    grid_lon = grid_df["center_lon"].to_numpy(dtype=float)
    frames: list[pd.DataFrame] = []
    cell_stats: list[dict] = []

    for idx, row in site_df.iterrows():
        dist_m = _haversine_m(float(row["lat"]), float(row["lon"]), grid_lat, grid_lon)
        bearing = _bearing_deg(float(row["lat"]), float(row["lon"]), grid_lat, grid_lon)
        az_delta = _azimuth_delta_deg(bearing, float(row["azimuth"]))
        full_cell_grid = PHASE9_CANDIDATE_MODE in {"full", "full_cell_grid", "all", "all_cells"}
        candidate_pre = np.ones(len(grid_df), dtype=bool) if full_cell_grid else dist_m <= COVERAGE_RADIUS_M

        raw = np.full(len(grid_df), np.nan, dtype=float)
        if candidate_pre.any():
            raw_unadjusted = _cost231_for_points(
                _site_record(row),
                grid_lat[candidate_pre],
                grid_lon[candidate_pre],
                float(row["frequency_mhz"]),
            )
            raw[candidate_pre] = raw_unadjusted + float(row.get("model_rsrp_adjust_db", 0.0))
        raw_valid = valid_model_rsrp(raw)
        candidate = candidate_pre & np.isfinite(raw)
        selected = grid_df.loc[candidate, ["grid_id", "center_lat", "center_lon"]].copy()

        cell_stats.append(
            {
                "strict_cell_key": str(row["strict_cell_key"]),
                "site": str(row["site_key"]),
                "sector": str(row["sector_key"]),
                "band": str(row["band_key"]),
                "azimuth": float(row["azimuth"]),
                "candidate_within_radius": int(candidate_pre.sum()),
                "candidate_saved_pixels": int(candidate.sum()),
                "candidate_mode": PHASE9_CANDIDATE_MODE,
                "coverage_grid_pct": float(candidate.sum() / len(grid_df) * 100.0),
                "valid_model_pixels": int(np.isfinite(raw_valid).sum()),
                "no_coverage_pixels": int(np.isnan(raw_valid).sum()),
                "min_raw_rsrp_unclipped": float(np.nanmin(raw[candidate])) if candidate.any() else None,
                "max_raw_rsrp_unclipped": float(np.nanmax(raw[candidate])) if candidate.any() else None,
                "mean_raw_rsrp_unclipped": float(np.nanmean(raw[candidate])) if candidate.any() else None,
            }
        )
        if not selected.empty:
            frames.append(
                pd.DataFrame(
                    {
                        "project_id": PROJECT_ID,
                        "grid_id": selected["grid_id"].astype(str).to_numpy(),
                        "lat": selected["center_lat"].to_numpy(dtype=float),
                        "lon": selected["center_lon"].to_numpy(dtype=float),
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
                        "azimuth": float(row["azimuth"]),
                        "distance_m": dist_m[candidate],
                        "bearing_deg": bearing[candidate],
                        "azimuth_delta_deg": az_delta[candidate],
                        "raw_cost231_rsrp_unclipped": raw[candidate],
                        "raw_cost231_rsrp": raw_valid[candidate],
                        "phase9_no_coverage": ~np.isfinite(raw_valid[candidate]),
                    }
                )
            )
        if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == len(site_df):
            print(
                f"[PHASE9][DIRECTIONAL] cells_done={idx + 1}/{len(site_df)} "
                f"candidate_rows={sum(len(frame) for frame in frames)}"
            )

    if frames:
        surface = pd.concat(frames, ignore_index=True)
    else:
        surface = pd.DataFrame()

    fallback_rows: list[dict] = []
    total_missing_tech_pixels = 0
    for technology_key in sorted(site_df["technology_key"].astype(str).dropna().unique()):
        tech_sites = site_df.loc[site_df["technology_key"].astype(str) == technology_key].copy()
        if tech_sites.empty:
            continue
        existing_grid_ids = (
            set(surface.loc[surface["technology"].astype(str) == technology_key, "grid_id"].astype(str))
            if not surface.empty
            else set()
        )
        missing_grid = grid_df.loc[~grid_df["grid_id"].astype(str).isin(existing_grid_ids)].copy()
        total_missing_tech_pixels += len(missing_grid)
        if not PHASE9_ENABLE_OUT_OF_RADIUS_BACKFILL:
            continue
        for grid_row in missing_grid.itertuples(index=False):
            lat = float(grid_row.center_lat)
            lon = float(grid_row.center_lon)
            raw_values: list[float] = []
            distances: list[float] = []
            bearings: list[float] = []
            deltas: list[float] = []
            for _, site_row in tech_sites.iterrows():
                raw = _cost231_for_points(
                    _site_record(site_row),
                    np.asarray([lat], dtype=float),
                    np.asarray([lon], dtype=float),
                    float(site_row["frequency_mhz"]),
                )[0]
                raw = float(raw + float(site_row.get("model_rsrp_adjust_db", 0.0)))
                bearing = float(_bearing_deg(float(site_row["lat"]), float(site_row["lon"]), np.asarray([lat]), np.asarray([lon]))[0])
                raw_values.append(float(raw))
                distances.append(float(_haversine_m(float(site_row["lat"]), float(site_row["lon"]), lat, lon)))
                bearings.append(bearing)
                deltas.append(float(_azimuth_delta_deg(np.asarray([bearing]), float(site_row["azimuth"]))[0]))

            # Keep the k nearest sectors as candidates (same schema as the in-radius path),
            # not just argmax(raw); downstream phases score antenna/obstruction/terrain/bias
            # per candidate and select the true serving cell after all corrections.
            order = list(np.argsort(np.asarray(distances, dtype=float)))
            keep_idx = order[: min(PHASE9_BACKFILL_K_NEAREST, len(order))]
            for cand_idx in keep_idx:
                row = tech_sites.iloc[int(cand_idx)]
                raw_valid = valid_model_rsrp(np.asarray([raw_values[cand_idx]], dtype=float))[0]
                fallback_rows.append(
                    {
                        "project_id": PROJECT_ID,
                        "grid_id": str(grid_row.grid_id),
                        "lat": lat,
                        "lon": lon,
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
                        "azimuth": float(row["azimuth"]),
                        "distance_m": distances[cand_idx],
                        "bearing_deg": bearings[cand_idx],
                        "azimuth_delta_deg": deltas[cand_idx],
                        "raw_cost231_rsrp_unclipped": raw_values[cand_idx],
                        "raw_cost231_rsrp": raw_valid,
                        "phase9_no_coverage": not np.isfinite(raw_valid),
                        "ensure_all_cells_backfill": True,
                    }
                )
    if total_missing_tech_pixels:
        print(
            "[PHASE9][ENSURE_ALL_CELLS] "
            f"missing_grid_technology_pixels={total_missing_tech_pixels} "
            f"action={f'k{PHASE9_BACKFILL_K_NEAREST}_nearest_sector_backfill_per_technology' if PHASE9_ENABLE_OUT_OF_RADIUS_BACKFILL else 'left_as_no_coverage'}"
        )
    if fallback_rows:
        fallback_df = pd.DataFrame.from_records(fallback_rows)
        surface["ensure_all_cells_backfill"] = False
        surface = pd.concat([surface, fallback_df], ignore_index=True)
    elif not surface.empty and "ensure_all_cells_backfill" not in surface.columns:
        surface["ensure_all_cells_backfill"] = False
    return surface, pd.DataFrame(cell_stats)


def _run_cost231_at_dt(site_df: pd.DataFrame, dt_df: pd.DataFrame) -> pd.DataFrame:
    dt_lat = dt_df["lat"].to_numpy(dtype=float)
    dt_lon = dt_df["lon"].to_numpy(dtype=float)
    out = dt_df.reset_index(drop=True).copy()
    out["assigned_strict_cell_key"] = pd.NA
    out["assigned_technology"] = pd.NA
    out["raw_cost231_at_dt_rsrp_unclipped"] = np.nan
    out["raw_cost231_at_dt_rsrp"] = np.nan

    for tech, dt_idx in out.groupby("measured_technology", dropna=False).groups.items():
        tech_sites = site_df[site_df["technology_key"].astype(str) == str(tech)].reset_index(drop=True)
        if tech_sites.empty:
            tech_sites = site_df.reset_index(drop=True)
        pred_matrix = np.empty((len(dt_idx), len(tech_sites)), dtype=float)
        group_pos = out.index.get_indexer(dt_idx)
        for idx, row in tech_sites.iterrows():
            site = _site_record(row)
            pred_matrix[:, idx] = (
                _cost231_for_points(site, dt_lat[group_pos], dt_lon[group_pos], float(row["frequency_mhz"]))
                + float(row.get("model_rsrp_adjust_db", 0.0))
            )
        best_idx = np.nanargmax(pred_matrix, axis=1)
        assigned = tech_sites.iloc[best_idx].reset_index(drop=True)
        best_raw = pred_matrix[np.arange(len(group_pos)), best_idx]
        out.loc[dt_idx, "assigned_strict_cell_key"] = assigned["strict_cell_key"].astype(str).to_numpy()
        out.loc[dt_idx, "assigned_technology"] = assigned["technology_key"].astype(str).to_numpy()
        out.loc[dt_idx, "raw_cost231_at_dt_rsrp_unclipped"] = best_raw
        out.loc[dt_idx, "raw_cost231_at_dt_rsrp"] = valid_model_rsrp(best_raw)
    out["dt_minus_cost231_db"] = out["rsrp_measured"] - out["raw_cost231_at_dt_rsrp"]
    return out


def _attach_nearest_grid(dt_assigned: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    tree = BallTree(np.radians(grid_df[["center_lat", "center_lon"]].to_numpy(dtype=float)), metric="haversine")
    dist_rad, idx = tree.query(np.radians(dt_assigned[["lat", "lon"]].to_numpy(dtype=float)), k=1)
    out = dt_assigned.copy()
    nearest = grid_df.iloc[idx[:, 0]].reset_index(drop=True)
    out["nearest_grid_id"] = nearest["grid_id"].astype(str).to_numpy()
    out["nearest_grid_distance_m"] = dist_rad[:, 0] * EARTH_RADIUS_M
    out["dt_replacement_eligible"] = out["nearest_grid_distance_m"] <= DT_REPLACE_RADIUS_M
    return out


def _offset_table(site_df: pd.DataFrame, dt_with_grid: pd.DataFrame) -> pd.DataFrame:
    valid = dt_with_grid.dropna(subset=["dt_minus_cost231_db"]).copy()
    cell_offsets = (
        valid.groupby("assigned_strict_cell_key", dropna=False)
        .agg(dt_count=("dt_minus_cost231_db", "size"), offset_db=("dt_minus_cost231_db", "median"))
        .reset_index()
        .rename(columns={"assigned_strict_cell_key": "strict_cell_key"})
    )
    out = site_df[["strict_cell_key", "technology_key", "site_key", "sector_key", "band_key"]].copy()
    out = out.merge(cell_offsets, on="strict_cell_key", how="left")
    out["offset_source"] = np.where(out["offset_db"].notna(), "cell_dt_median", "technology_dt_median")

    tech_offsets = (
        valid.groupby("assigned_technology", dropna=False)["dt_minus_cost231_db"]
        .agg(technology_dt_count="size", technology_offset_db="median")
        .reset_index()
        .rename(columns={"assigned_technology": "technology_key"})
    )
    out = out.merge(tech_offsets, on="technology_key", how="left")
    use_tech = out["offset_db"].isna() & out["technology_offset_db"].notna()
    out.loc[use_tech, "offset_db"] = out.loc[use_tech, "technology_offset_db"]
    global_offset = float(valid["dt_minus_cost231_db"].median()) if not valid.empty else 0.0
    use_global = out["offset_db"].isna()
    out.loc[use_global, "offset_db"] = global_offset
    out.loc[use_global, "offset_source"] = "global_dt_median"
    out["dt_count"] = out["dt_count"].fillna(0).astype(int)
    out["fallback_dt_count"] = out["technology_dt_count"].fillna(len(valid)).astype(int)
    return out


def _apply_offset_and_dt(surface: pd.DataFrame, offsets: pd.DataFrame, dt_with_grid: pd.DataFrame) -> pd.DataFrame:
    out = surface.merge(
        offsets[["strict_cell_key", "offset_db", "offset_source", "dt_count", "fallback_dt_count"]],
        on="strict_cell_key",
        how="left",
    )
    out["offset_db"] = pd.to_numeric(out["offset_db"], errors="coerce").fillna(0.0)
    out["offset_corrected_rsrp_unclipped"] = out["raw_cost231_rsrp"] + out["offset_db"]
    out["offset_corrected_rsrp"] = valid_model_rsrp(out["offset_corrected_rsrp_unclipped"])
    replacements = (
        dt_with_grid.loc[dt_with_grid["dt_replacement_eligible"]]
        .groupby(["assigned_technology", "nearest_grid_id"], dropna=False)
        .agg(dt_replacement_rsrp=("rsrp_measured", "mean"), dt_replacement_count=("rsrp_measured", "size"))
        .reset_index()
        .rename(columns={"assigned_technology": "technology", "nearest_grid_id": "grid_id"})
    )
    out = out.merge(replacements, on=["technology", "grid_id"], how="left")
    out["dt_replaced"] = out["dt_replacement_rsrp"].notna()
    out["corrected_rsrp"] = out["offset_corrected_rsrp"].where(~out["dt_replaced"], out["dt_replacement_rsrp"])
    out["corrected_rsrp"] = display_rsrp(out["corrected_rsrp"])
    out["dt_replacement_count"] = out["dt_replacement_count"].fillna(0).astype(int)
    return out


def _serving_grid(surface: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    valid = surface.dropna(subset=["corrected_rsrp"]).copy()
    technologies = sorted(surface["technology"].astype(str).dropna().unique()) if "technology" in surface.columns else []
    tech_grid = pd.concat(
        [grid_df[["grid_id", "center_lat", "center_lon"]].assign(technology=tech) for tech in technologies],
        ignore_index=True,
    ) if technologies else grid_df[["grid_id", "center_lat", "center_lon"]].copy()
    if valid.empty:
        serving = tech_grid.copy()
    else:
        best_idx = valid.groupby(["technology", "grid_id"], dropna=False)["corrected_rsrp"].idxmax()
        serving = valid.loc[best_idx].copy()
        serving = tech_grid.merge(serving, on=["technology", "grid_id"], how="left")
    serving["has_directional_candidate"] = serving["corrected_rsrp"].notna()
    return serving


def _cdf_values(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    if arr.size == 0:
        return arr, arr
    return arr, np.arange(1, arr.size + 1, dtype=float) / arr.size * 100.0


def _plot_cdf(series_map: list[tuple[str, pd.Series, str]], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label, values, color in series_map:
        x, y = _cdf_values(values)
        if len(x):
            ax.plot(x, y, linewidth=2.4, label=f"{label} (n={len(x):,})", color=color)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("RSRP (dBm)")
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.set_xlim(CLIP_RSRP)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_frame(df: pd.DataFrame, stem: str) -> None:
    csv_path = DATA_DIR / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(csv_path.with_suffix(".parquet"), index=False)
    except Exception as exc:
        csv_path.with_suffix(".parquet.error.txt").write_text(str(exc), encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    print(f"[PHASE9][START] project={PROJECT_ID} dir={PROJECT_DIR}")
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
                BASELINE_SCOPE / f"drive_project_{PROJECT_ID}_baseline_primary_airtel_polygon.csv",
            ]
        )
    )
    project_polygon = _load_project_polygon()
    grid_df, grid_meta = _generate_gridanalytics_compatible_grid()
    print(f"[PHASE9][INPUT] cells={len(site_df)} dt_rows={len(drive_df)} gridanalytics_pixels={len(grid_df)}")

    surface, cell_stats = _run_directional_surface(site_df, grid_df)
    dt_assigned = _run_cost231_at_dt(site_df, drive_df)
    dt_with_grid = _attach_nearest_grid(dt_assigned, grid_df)
    offsets = _offset_table(site_df, dt_with_grid)
    surface = _apply_offset_and_dt(surface, offsets, dt_with_grid)
    serving = _serving_grid(surface, grid_df)

    _save_frame(grid_df, f"phase9_gridanalytics_compatible_grid_project{PROJECT_ID}")
    _save_frame(surface, f"phase9_directional_raw_corrected_surface_project{PROJECT_ID}")
    _save_frame(serving, f"phase9_directional_serving_grid_project{PROJECT_ID}")
    _save_frame(dt_with_grid, f"phase9_dt_match_project{PROJECT_ID}")
    _save_frame(offsets, f"phase9_offsets_project{PROJECT_ID}")
    _save_frame(cell_stats, f"phase9_cell_directional_coverage_summary_project{PROJECT_ID}")

    _plot_cdf(
        [
            ("Directional raw Cost231 per-cell rows", surface["raw_cost231_rsrp"], "#d94f3d"),
            ("Directional after offset + DT replacement", surface["corrected_rsrp"], "#168a52"),
            ("Serving grid after offset + DT replacement", serving["corrected_rsrp"], "#2563eb"),
            ("DT measured", dt_with_grid["rsrp_measured"], "#805ad5"),
        ],
        "Project 196 Cost231 Phase 9 - GridAnalytics Compatible + Backfill",
        COMBINED_DIR / "cdf_phase9_gridanalytics_compatible_serving_dt.png",
    )

    candidate_counts = surface.groupby("grid_id")["strict_cell_key"].nunique()
    bins = [-140, -120, -110, -100, -95, -85, -44]
    labels = ["-140 to -120", "-120 to -110", "-110 to -100", "-100 to -95", "-95 to -85", "-85 to -44"]
    summary = {
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "project_id": PROJECT_ID,
        "production_code_modified": False,
        "phase_label": "Cost231 Phase 9 GridAnalytics-compatible safe-radius candidates",
        "grid_source": "GridAnalyticsController-compatible active map_regions bounding rectangle grid generated inside test case",
        "grid_meta": grid_meta,
        "grid_size_m": GRID_SIZE_M,
        "candidate_mode": PHASE9_CANDIDATE_MODE,
        "coverage_radius_m": COVERAGE_RADIUS_M,
        "antenna_pattern_logic": ANTENNA_PATTERN_LOGIC,
        "hard_azimuth_cutoff_used": False,
        "min_candidate_rsrp_dbm": MIN_CANDIDATE_RSRP_DBM,
        "candidate_filter_rule": (
            "distance <= coverage_radius_m only; no fixed candidate-count cap and no pre-loss RSRP cutoff"
        ),
        "out_of_radius_backfill_enabled": PHASE9_ENABLE_OUT_OF_RADIUS_BACKFILL,
        "out_of_radius_backfill_k_nearest": PHASE9_BACKFILL_K_NEAREST,
        "out_of_radius_backfill_rule": (
            f"cells with no sector within {COVERAGE_RADIUS_M:.0f} m keep their {PHASE9_BACKFILL_K_NEAREST} "
            "nearest same-technology sectors as candidates, scored by the same COST231+antenna model; "
            "downstream phases apply obstruction/terrain/bias per candidate and pick the serving cell"
        ),
        "coverage_threshold_dbm": RSRP_NO_COVERAGE_DBM,
        "dt_replace_radius_m": DT_REPLACE_RADIUS_M,
        "strict_cells": int(len(site_df)),
        "gridanalytics_grid_pixels": int(len(grid_df)),
        "directional_surface_rows": int(len(surface)),
        "full_unfiltered_cell_pixel_rows_would_be": int(len(site_df) * len(grid_df)),
        "surface_rows_vs_full_pct": float(len(surface) / max(len(site_df) * len(grid_df), 1) * 100.0),
        "grid_pixels_with_any_directional_candidate": int(
            surface.loc[~surface["ensure_all_cells_backfill"].astype(bool), "grid_id"].nunique()
        ),
        "grid_pixels_without_directional_candidate": int(
            len(grid_df) - surface.loc[~surface["ensure_all_cells_backfill"].astype(bool), "grid_id"].nunique()
        ),
        "ensure_all_cells_backfill_rows": int(surface["ensure_all_cells_backfill"].sum()),
        "grid_pixels_after_backfill": int(surface["grid_id"].nunique()),
        "missing_grid_pixels_after_backfill": int(len(grid_df) - surface["grid_id"].nunique()),
        "serving_grid_rows": int(len(serving)),
        "serving_grid_non_null": int(serving["corrected_rsrp"].notna().sum()),
        "dt_rows": int(len(drive_df)),
        "dt_replacement_eligible_rows": int(dt_with_grid["dt_replacement_eligible"].sum()),
        "dt_replaced_surface_rows": int(surface["dt_replaced"].sum()),
        "cells_with_any_directional_pixels": int(surface["strict_cell_key"].nunique()),
        "cells_without_directional_pixels": int(len(site_df) - surface["strict_cell_key"].nunique()),
        "cells_with_dt_offset": int((offsets["dt_count"] > 0).sum()),
        "offset_source_counts": offsets["offset_source"].value_counts(dropna=False).to_dict(),
        "candidate_cells_per_grid": {
            "min": int(candidate_counts.min()) if not candidate_counts.empty else 0,
            "max": int(candidate_counts.max()) if not candidate_counts.empty else 0,
            "mean": float(candidate_counts.mean()) if not candidate_counts.empty else 0.0,
        },
        "surface_corrected_bin_counts": {
            str(k): int(v)
            for k, v in pd.cut(
                surface["corrected_rsrp"],
                bins=bins,
                labels=labels,
                right=False,
                include_lowest=True,
            )
            .value_counts(sort=False)
            .items()
        },
        "serving_corrected_bin_counts": {
            str(k): int(v)
            for k, v in pd.cut(
                serving["corrected_rsrp"],
                bins=bins,
                labels=labels,
                right=False,
                include_lowest=True,
            )
            .value_counts(sort=False)
            .items()
        },
        "outputs": {
            "grid_csv": str((DATA_DIR / f"phase9_gridanalytics_compatible_grid_project{PROJECT_ID}.csv").relative_to(THIS_DIR)),
            "surface_csv": str((DATA_DIR / f"phase9_directional_raw_corrected_surface_project{PROJECT_ID}.csv").relative_to(THIS_DIR)),
            "serving_csv": str((DATA_DIR / f"phase9_directional_serving_grid_project{PROJECT_ID}.csv").relative_to(THIS_DIR)),
            "dt_csv": str((DATA_DIR / f"phase9_dt_match_project{PROJECT_ID}.csv").relative_to(THIS_DIR)),
            "offsets_csv": str((DATA_DIR / f"phase9_offsets_project{PROJECT_ID}.csv").relative_to(THIS_DIR)),
            "cell_summary_csv": str((DATA_DIR / f"phase9_cell_directional_coverage_summary_project{PROJECT_ID}.csv").relative_to(THIS_DIR)),
            "combined_dir": str(COMBINED_DIR.relative_to(THIS_DIR)),
        },
    }
    (DATA_DIR / "phase9_gridanalytics_compatible_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
