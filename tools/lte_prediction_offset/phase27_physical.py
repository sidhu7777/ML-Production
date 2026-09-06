"""Production-native Phase 26 physical path scorer.

This module contains no test-project data.  It applies one path-based terrain
diffraction term and one dominant-building/O2I term to Phase-9 candidates.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, box
from shapely.ops import transform

from tools.lte_prediction.dem_utils import ensure_project_dem
from tools.lte_prediction.geo_correction_pipeline import building_df_to_gdf

try:
    import rasterio
    from pyproj import Transformer
except ImportError:  # pragma: no cover - environment dependency
    rasterio = None
    Transformer = None


EARTH_RADIUS_M = 6_371_000.0
UE_HEIGHT_M = 1.5
DEFAULT_BUILDING_HEIGHT_M = 12.0
# A DEM sample must intrude materially into the first Fresnel zone before it
# is eligible as a terrain obstacle. This rejects sub-cell elevation noise;
# it is an input-quality gate before the P.526-style knife-edge calculation.
TERRAIN_FRESNEL_CLEARANCE_FRACTION = 0.60
TERRAIN_ENDPOINT_BUFFER_M = 20.0


def _swap_geometry_xy(geom):
    """Swap coordinate order while preserving multipart building geometry."""
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([_swap_geometry_xy(part) for part in geom.geoms])
    if geom.geom_type == "GeometryCollection":
        return GeometryCollection([_swap_geometry_xy(part) for part in geom.geoms])
    return transform(lambda x, y, z=None: (y, x) if z is None else (y, x, z), geom)


def _align_buildings_to_prediction_extent(buildings, candidates):
    """Use the geometry orientation that intersects the prediction extent.

    Legacy project building rows can be stored as x=latitude/y=longitude.
    Choose orientation from actual spatial overlap instead of assuming a DB
    convention, exactly as the production polygon alignment already does.
    """
    lat = pd.to_numeric(candidates.get("lat"), errors="coerce")
    lon = pd.to_numeric(candidates.get("lon"), errors="coerce")
    valid = lat.notna() & lon.notna()
    if buildings.empty or not valid.any():
        return buildings, "empty"

    extent = box(float(lon[valid].min()), float(lat[valid].min()), float(lon[valid].max()), float(lat[valid].max()))
    direct_hits = int(buildings.geometry.intersects(extent).sum())
    swapped = buildings.copy()
    swapped["geometry"] = swapped.geometry.apply(_swap_geometry_xy)
    swapped_hits = int(swapped.geometry.intersects(extent).sum())
    if swapped_hits > direct_hits:
        return swapped, f"swapped_xy direct_hits={direct_hits} swapped_hits={swapped_hits}"
    return buildings, f"original direct_hits={direct_hits} swapped_hits={swapped_hits}"


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _knife_edge_loss_db(height_m: float, d1_m: float, d2_m: float, freq_mhz: float) -> float:
    wavelength_m = 300.0 / max(float(freq_mhz), 1.0)
    v = height_m * math.sqrt(2.0 * (d1_m + d2_m) / (wavelength_m * d1_m * d2_m))
    if v <= -0.78:
        return 0.0
    return max(0.0, 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1))


class _DemSampler:
    def __init__(self, path: str | Path):
        if rasterio is None:
            raise RuntimeError("rasterio is unavailable")
        self.src = rasterio.open(path)
        self.to_dem = Transformer.from_crs("EPSG:4326", self.src.crs, always_xy=True)
        # A DEM can contain RGB/display bands. Prefer continuous floating or
        # signed/non-byte numeric bands, then choose the plausible elevation
        # surface with the widest range. This avoids treating 0/255 imagery as
        # terrain while remaining compatible with integer elevation rasters.
        candidates = []
        for band in range(1, self.src.count + 1):
            values = self.src.read(band, masked=True).compressed().astype(float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            if len(values) > 100_000:
                values = values[:: max(1, len(values) // 100_000)]
            p1, p50, p99 = np.percentile(values, [1, 50, 99])
            plausible = -500.0 <= p1 <= 9000.0 and -500.0 <= p50 <= 9000.0 and p99 <= 9000.0
            dtype = np.dtype(self.src.dtypes[band - 1])
            elevation_like = dtype.kind == "f" or dtype.kind == "i" or dtype.itemsize > 1
            if plausible and p99 - p1 >= 1.0:
                candidates.append((elevation_like, p99 - p1, band))
        preferred = [candidate for candidate in candidates if candidate[0]]
        best_pool = preferred or candidates
        if not best_pool:
            raise ValueError("No plausible elevation raster band in DEM")
        self.band = max(best_pool, key=lambda candidate: candidate[1])[2]

    def close(self):
        self.src.close()

    def sample(self, lat, lon):
        lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        x, y = self.to_dem.transform(lon, lat)
        values = np.asarray([value[0] for value in self.src.sample(zip(x, y), indexes=self.band)], dtype=float)
        nodata = self.src.nodatavals[self.band - 1]
        if nodata is not None:
            values[np.isclose(values, nodata)] = np.nan
        return values


def _terrain_loss_details(dem, tx_lat, tx_lon, tx_height, rx_lat, rx_lon, frequency_mhz):
    """Return terrain loss and the decision inputs used to produce it."""
    distance = float(_haversine_m(tx_lat, tx_lon, rx_lat, rx_lon))
    if distance < 2.0:
        return 0.0, 0.0, 0.0, "short_path"
    fractions = np.linspace(0.0, 1.0, int(np.clip(math.ceil(distance / 25.0), 7, 64)))
    lat = tx_lat + (rx_lat - tx_lat) * fractions
    lon = tx_lon + (rx_lon - tx_lon) * fractions
    profile = dem.sample(lat, lon)
    if np.isnan(profile).all():
        return 0.0, 0.0, np.nan, "dem_no_data"
    profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()
    los = (profile[0] + tx_height) + ((profile[-1] + UE_HEIGHT_M) - (profile[0] + tx_height)) * fractions
    clearance = profile - los
    d1 = distance * fractions
    d2 = distance * (1.0 - fractions)
    wavelength_m = 300.0 / max(float(frequency_mhz), 1.0)
    fresnel_r1 = np.sqrt(wavelength_m * d1 * d2 / np.maximum(d1 + d2, 1.0))
    endpoint_buffer = min(max(TERRAIN_ENDPOINT_BUFFER_M, 0.02 * distance), distance / 3.0)
    interior = (d1 >= endpoint_buffer) & (d2 >= endpoint_buffer)
    significant = clearance - TERRAIN_FRESNEL_CLEARANCE_FRACTION * fresnel_r1
    significant[~interior] = -np.inf
    i = int(np.nanargmax(significant))
    if not np.isfinite(significant[i]) or significant[i] <= 0.0:
        peak = float(np.nanmax(clearance[interior])) if interior.any() else np.nan
        return 0.0, float(significant[i]) if np.isfinite(significant[i]) else np.nan, peak, "below_fresnel_gate"
    loss = _knife_edge_loss_db(float(significant[i]), max(d1[i], 1.0), max(d2[i], 1.0), frequency_mhz)
    return loss, float(significant[i]), float(clearance[i]), "fresnel_obstructed"


def _terrain_loss(dem, tx_lat, tx_lon, tx_height, rx_lat, rx_lon, frequency_mhz):
    """Compatibility wrapper for scalar terrain loss callers/tests."""
    return _terrain_loss_details(dem, tx_lat, tx_lon, tx_height, rx_lat, rx_lon, frequency_mhz)[0]


def _indoor_depth_m(tx: Point, rx: Point, polygon) -> float:
    # The intersection length from receiver towards transmitter is the
    # available indoor path depth. Convert local degrees to metres by using
    # geodesic endpoint distance.
    line = LineString([rx, tx])
    inter = line.intersection(polygon)
    if inter.is_empty:
        return 0.0
    if inter.geom_type == "LineString":
        segments = [inter]
    elif inter.geom_type == "MultiLineString":
        segments = list(inter.geoms)
    elif inter.geom_type == "GeometryCollection":
        segments = [part for part in inter.geoms if part.geom_type == "LineString"]
    else:
        return 0.0
    depths = []
    for segment in segments:
        coords = list(segment.coords)
        if len(coords) >= 2:
            depths.append(float(_haversine_m(coords[0][1], coords[0][0], coords[-1][1], coords[-1][0])))
    return min(80.0, max(depths, default=0.0))


def _building_loss(buildings, sindex, tx_lat, tx_lon, tx_height, rx_lat, rx_lon, frequency_mhz):
    tx, rx = Point(tx_lon, tx_lat), Point(rx_lon, rx_lat)
    hits = list(sindex.query(rx, predicate="intersects")) if not buildings.empty else []
    containing = [i for i in hits if buildings.geometry.iloc[i].covers(rx)]
    if containing:
        depth = max((_indoor_depth_m(tx, rx, buildings.geometry.iloc[i]) for i in containing), default=0.0)
        wall = 8.5 + 9.5 * math.log10(max(float(frequency_mhz) / 1000.0, 0.1))
        indoor_depth = 13.0 * (1.0 - math.exp(-depth / 12.0))
        return -(wall + indoor_depth), "indoor", "Building"

    line = LineString([tx, rx])
    indices = list(sindex.query(line, predicate="intersects")) if not buildings.empty else []
    total = float(_haversine_m(tx_lat, tx_lon, rx_lat, rx_lon))
    dominant = 0.0
    for i in indices:
        geom = buildings.geometry.iloc[i]
        inter = line.intersection(geom)
        if inter.is_empty:
            continue
        p = inter.centroid
        d1 = max(float(_haversine_m(tx_lat, tx_lon, p.y, p.x)), 1.0)
        d2 = max(total - d1, 1.0)
        los_h = tx_height + (UE_HEIGHT_M - tx_height) * (d1 / total)
        height = float(buildings.iloc[i]["height_m"])
        dominant = max(dominant, _knife_edge_loss_db(height - los_h, d1, d2, frequency_mhz))
    return -dominant, ("obstructed" if dominant > 0.0 else "clear"), ("Building" if dominant > 0.0 else "Open")


def score_candidates(
    candidates: pd.DataFrame,
    site_df: pd.DataFrame,
    building_df: pd.DataFrame,
    project_id: int,
    region: str,
    dem_raster_path: str | Path | None = None,
    clutter_by_grid: dict | None = None,
    allow_auto_dem: bool = True,
) -> pd.DataFrame:
    """Apply Phase-26 physical corrections to already-selected candidates."""
    out = candidates.copy()
    sites = site_df.set_index("strict_cell_key")
    buildings = building_df_to_gdf(building_df)
    if not buildings.empty:
        buildings, building_alignment = _align_buildings_to_prediction_extent(buildings, out)
        print(f"[LTE_OFFSET][PHASE26_BUILDINGS] rows={len(buildings)} alignment={building_alignment}", flush=True)
        heights = pd.to_numeric(buildings.get("building_height_m"), errors="coerce")
        levels = pd.to_numeric(buildings.get("building_levels"), errors="coerce")
        buildings["height_m"] = heights.fillna(levels * 3.0).fillna(DEFAULT_BUILDING_HEIGHT_M).clip(3.0, 120.0)
        sindex = buildings.sindex
    else:
        sindex = None
        print("[LTE_OFFSET][PHASE26_BUILDINGS] rows=0 alignment=empty", flush=True)

    dem = None
    explicit_dem = Path(dem_raster_path).expanduser() if dem_raster_path else None
    try:
        if explicit_dem is not None:
            if not explicit_dem.is_file():
                raise FileNotFoundError(f"Configured DEM does not exist: {explicit_dem}")
            dem_path = explicit_dem
        else:
            if not allow_auto_dem:
                raise FileNotFoundError("No approved project terrain DEM resolved")
            dem_path = ensure_project_dem(int(project_id), str(region), site_df)
        dem = _DemSampler(dem_path)
        print(f"[LTE_OFFSET][PHASE26_DEM] enabled=True path={dem_path} selected_band={dem.band}", flush=True)
    except Exception as exc:
        if explicit_dem is not None:
            raise RuntimeError(f"Configured terrain DEM is unusable: {exc}") from exc
        print(f"[LTE_OFFSET][PHASE26_DEM] enabled=False reason={exc}", flush=True)

    out["building_obstruction_loss_db"] = 0.0
    out["terrain_diffraction_loss_db"] = 0.0
    out["terrain_fresnel_excess_m"] = np.nan
    out["terrain_peak_clearance_m"] = np.nan
    out["terrain_decision"] = "not_evaluated"
    out["obstruction_branch"] = "clear"
    out["clutter_class"] = "Open"
    clutter_by_grid = clutter_by_grid or {}
    for key, idx in out.groupby("strict_cell_key", dropna=False).groups.items():
        if key not in sites.index:
            continue
        site = sites.loc[key]
        if isinstance(site, pd.DataFrame):
            site = site.iloc[0]
        for row_index in idx:
            row = out.loc[row_index]
            building_loss, branch, clutter = _building_loss(
                buildings, sindex, float(site.lat), float(site.lon), float(site.Height), float(row.lat), float(row.lon), float(row.serving_frequency_mhz)
            ) if sindex is not None else (0.0, "clear", "Open")
            source_clutter = str(clutter_by_grid.get(str(row.grid_id), "")).strip()
            if source_clutter and source_clutter.lower() not in {"nan", "none", "unknown"}:
                clutter = source_clutter if branch != "indoor" else "Indoor"
            if dem is None:
                terrain, excess, peak, decision = 0.0, np.nan, np.nan, "dem_disabled"
            elif str(clutter).lower() == "water":
                # Water is a land-cover condition, not a DEM ridge. Do not
                # turn shoreline/void raster samples into a terrain obstacle.
                terrain, excess, peak, decision = 0.0, np.nan, np.nan, "water_land_cover"
            else:
                terrain, excess, peak, decision = _terrain_loss_details(
                    dem, float(site.lat), float(site.lon), float(site.Height), float(row.lat), float(row.lon), float(row.serving_frequency_mhz)
                )
            out.loc[row_index, [
                "building_obstruction_loss_db", "terrain_diffraction_loss_db", "terrain_fresnel_excess_m",
                "terrain_peak_clearance_m", "terrain_decision", "obstruction_branch", "clutter_class"
            ]] = [building_loss, terrain, excess, peak, decision, branch, clutter]
    if dem is not None:
        dem.close()
    terrain_counts = out["terrain_decision"].value_counts(dropna=False).to_dict()
    print(
        "[LTE_OFFSET][PHASE26_TERRAIN] "
        f"policy=fresnel_gate_{TERRAIN_FRESNEL_CLEARANCE_FRACTION:.2f} "
        f"nonzero={int((out['terrain_diffraction_loss_db'] > 0).sum())} decisions={terrain_counts}",
        flush=True,
    )
    out["physical_rsrp_unclipped"] = pd.to_numeric(out["raw_cost231_rsrp"], errors="coerce") + pd.to_numeric(out["building_obstruction_loss_db"], errors="coerce") - pd.to_numeric(out["terrain_diffraction_loss_db"], errors="coerce")
    return out
