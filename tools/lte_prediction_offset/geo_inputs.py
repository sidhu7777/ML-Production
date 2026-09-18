"""Project-scoped geospatial input cache for the offset baseline.

This module owns *input* provenance, not prediction output. Water/Vegetation
come from real Overture vector polygons; Open/Suburban/Urban/Dense Urban come
from a Canny-edge + K-Means image-texture read of one Google satellite tile
per project (ported from Research/open_map.py, fit only on tiles Overture
hasn't already claimed as Water/Vegetation). Building height is resolved
separately (GHS-OBAT / OSM) purely for indoor-penetration use elsewhere in
the baseline - it is not an input to this classifier any more; see the
per-function docstrings below for the real bugs this replaces and how each
was verified against real project data before landing here.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import requests
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from tools.lte_prediction.geo_correction_pipeline import _choose_utm_crs, building_df_to_gdf

PHASE27_CLUTTER_SOURCE = "Overture Maps + Image Texture"
# v3 -> v4: replaced the building-height/footprint-ratio density classifier
# (real bug: building_ratio's min(1.0, area/tile_area) silently returned 1.0
# for any tile touching an invalid/self-intersecting building polygon,
# because Python's min() does not special-case NaN - verified against real
# project 246 data: forced 91.9% of all non-water tiles into "Dense Urban")
# with the Overture-exclusion + image-texture method validated in
# tests/new-project/open_map_clutter. This version bump is what makes every
# existing project's wrong v3 cache stop matching and get recomputed - see
# _phase27_dataset below, no separate migration needed.
PHASE27_CLASSIFIER_VERSION = "phase27-v4"
OVERTURE_CONNECT_TIMEOUT_S = 20
OVERTURE_REQUEST_TIMEOUT_S = 90
# land_use only, deliberately not land_cover (forest/shrub/grass) - checked
# directly against real data: a single land_cover "forest" polygon measured
# 6.03 km2 against a 6.37 km2 project grid. Overture's land_cover theme is a
# coarse regional background classification that does not get updated when a
# city is built on what was broadly forest/grassland terrain; it is not a
# reliable signal for whether a specific tile is vegetated today. land_use
# (park/garden/pitch/etc, real individually-mapped features) is trustworthy.
GREEN_LU_SUBTYPES = {"park", "recreation", "horticulture", "agriculture"}
GREEN_LU_CLASS = {"park", "garden", "grass", "recreation_ground", "village_green", "pitch", "nature_reserve"}

# Same floor for both water and green - checked directly against real data:
# with water needing >=0.5 while green only needed >=0.30, a tile could be
# 48.5% water and still lose to green clearing its own, easier bar,
# converting a near-half-water tile into a 100% "Vegetation" label. Fixed by
# using one shared floor and then requiring whichever of water/green wins
# that comparison to ALSO be the genuine majority against the real built
# share left over (1 - water_ratio - green_ratio) - otherwise a tile that is
# 35% park and 65% real dense city block was being swallowed whole into
# Vegetation, because 0.35 only had to beat water's near-zero share, never
# the 65% that was actually built.
WATER_GREEN_FLOOR = 0.30

IMG_WIDTH = 640
IMG_HEIGHT = 640
SCALE = 2
# Research/open_map.py's own __main__ default ZOOM_LEVEL - the zoom its
# 151px blur kernel was implicitly calibrated for. See _scaled_blur_kernel_px.
REFERENCE_ZOOM = 14
TILE4_LABELS = {0: "Open", 1: "Suburban", 2: "Urban", 3: "Dense Urban"}


class _WebMercatorProjector:
    """Exact lon/lat -> Google Static Maps pixel-space mapping. Ported from
    Research/open_map.py's WebMercatorProjector (unmodified math) as
    first-class production code rather than importing a research/scratch
    module into the prediction path."""

    def __init__(self, center_lat: float, center_lon: float, zoom: int, img_w: int = IMG_WIDTH, img_h: int = IMG_HEIGHT, scale: int = SCALE):
        self.zoom = zoom
        self.scale = scale
        self.TILE_SIZE = 256
        self.cx_world, self.cy_world = self._project(center_lat, center_lon)
        self.tl_x_world = self.cx_world - (img_w / 2)
        self.tl_y_world = self.cy_world - (img_h / 2)

    def _project(self, lat: float, lon: float) -> tuple[float, float]:
        siny = min(max(math.sin(lat * math.pi / 180), -0.9999), 0.9999)
        x = self.TILE_SIZE * (0.5 + lon / 360)
        y = self.TILE_SIZE * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
        zoom_factor = 2 ** self.zoom
        return x * zoom_factor, y * zoom_factor

    def coord_to_pixel(self, lat: float, lon: float) -> tuple[int, int]:
        world_x, world_y = self._project(lat, lon)
        return int((world_x - self.tl_x_world) * self.scale), int((world_y - self.tl_y_world) * self.scale)


def _fetch_satellite_tile(center_lat: float, center_lon: float, zoom: int) -> np.ndarray:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set - required for Phase-27 image-texture clutter classification")
    params = {
        "center": f"{center_lat},{center_lon}", "zoom": zoom, "size": f"{IMG_WIDTH}x{IMG_HEIGHT}",
        "maptype": "satellite", "key": api_key, "scale": SCALE, "format": "png",
    }
    response = requests.get("https://maps.googleapis.com/maps/api/staticmap", params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Google Static Maps request failed: {response.status_code} {response.text[:200]}")
    image_array = np.asarray(bytearray(response.content), dtype="uint8")
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def _pick_zoom(min_lon: float, min_lat: float, max_lon: float, max_lat: float, margin: float = 1.08) -> int:
    """Smallest zoom whose single satellite tile still covers the whole grid
    bounding box. `scale` (Google's retina/DPI parameter) does NOT extend
    coverage - it only doubles pixel density for the same area (confirmed
    against _WebMercatorProjector itself: its top-left world offset is
    computed from the un-scaled logical width, scale is applied only
    afterwards at the world-to-output-pixel step). Treating scale as extra
    coverage picked a zoom one level too far in during early testing - the
    fetched image covered barely half the real bounding box."""
    mean_lat = (min_lat + max_lat) / 2.0
    width_m = (max_lon - min_lon) * 111320.0 * math.cos(math.radians(mean_lat)) * margin
    height_m = (max_lat - min_lat) * 111320.0 * margin
    needed_m = max(width_m, height_m)
    eff_px = min(IMG_WIDTH, IMG_HEIGHT)
    for zoom in range(20, 9, -1):
        meters_per_pixel = 156543.03392 * math.cos(math.radians(mean_lat)) / (2 ** zoom)
        if eff_px * meters_per_pixel >= needed_m:
            return zoom
    return 10


def _scaled_blur_kernel_px(zoom: int, base_kernel_px: int = 151, reference_zoom: int = REFERENCE_ZOOM) -> int:
    """Research/open_map.py hardcodes a 151px Gaussian blur kernel, implicitly
    calibrated for whatever real-world distance 151px is AT its own reference
    zoom (14). Meters-per-pixel halves per zoom level in, doubles per zoom
    level out (latitude and the scale/DPI parameter cancel out of this ratio
    identically for reference vs actual zoom), so the correct pixel count at
    any zoom is base_kernel_px * 2**(zoom - reference_zoom), rounded to the
    nearest odd integer (GaussianBlur requires odd kernel size). Without this,
    the blur's real-world footprint silently varies with project size alone -
    measured a 127x swing (41m to 5,230m) across realistic zoom levels."""
    scaled = base_kernel_px * (2.0 ** (zoom - reference_zoom))
    k = max(1, int(round(scaled)))
    return k if k % 2 == 1 else k + 1


def _edge_density_map(sat_img: np.ndarray, blur_kernel_px: int) -> np.ndarray:
    """Canny-edge + blur preprocessing, ported from Research/open_map.py's
    HybridZoneExtractor.extract_building_density - duplicated rather than
    imported because that method bundles this step together with a
    hardcoded, unmasked k=4 K-Means call with no hook to restrict which
    pixels get clustered. blur_kernel_px replaces the original's fixed 151 -
    see _scaled_blur_kernel_px."""
    gray = cv2.cvtColor(sat_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 80, 160), np.ones((2, 2), np.uint8), iterations=1)
    return np.clip(cv2.multiply(cv2.GaussianBlur(edges, (blur_kernel_px, blur_kernel_px), 0), 1.2), 0, 255).astype(np.uint8)


def _mode_or(values: np.ndarray, default: int = 0) -> int:
    if values.size == 0:
        return default
    return int(np.argmax(np.bincount(values.astype(np.int64))))


def _kmeans_tiers_on_mask(density_map: np.ndarray, candidate_mask: np.ndarray, k: int = 4) -> np.ndarray:
    """K-Means over only the pixels candidate_mask marks True, ranked so
    0=lowest edge density .. k-1=highest. Everywhere else is -1. Water and
    Vegetation tiles are excluded from this fit entirely (decided from
    Overture before this ever runs) - their pixels have no business
    influencing where the built-density cluster boundaries fall; a river's
    smooth surface and a sparse suburb both read as low edge density, and
    letting water/vegetation pixels sit in the same clustering pool skews
    the boundaries. k=4, not 3: a genuinely low-texture built-candidate tile
    (bare/open ground Overture never tagged as a park) still needs a real
    floor to land in ("Open") instead of being forced into "Suburban"."""
    tiers = np.full(density_map.shape, -1, dtype=np.int8)
    ys, xs = np.where(candidate_mask)
    if ys.size < k:
        return tiers
    values = np.float32(density_map[ys, xs].reshape(-1, 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    rank_of_label = {int(old): new for new, old in enumerate(np.argsort(centers.flatten()))}
    tiers[ys, xs] = np.array([rank_of_label[int(lbl)] for lbl in labels.flatten()], dtype=np.int8)
    return tiers


def _resolve_water_or_green(wr: float, gr: float) -> str | None:
    """'Water', 'Vegetation', or None (a built-density candidate) - see
    WATER_GREEN_FLOOR module docstring for why this compares both ratios
    against the real built share left over, not just against each other."""
    if wr < WATER_GREEN_FLOOR and gr < WATER_GREEN_FLOOR:
        return None
    built = max(0.0, 1.0 - wr - gr)
    if wr >= gr:
        return "Water" if wr >= built else None
    return "Vegetation" if gr >= built else None


def _with_db_retry(db_engine, action, label: str):
    """Retry only transient connection failures; SQL/data errors still fail fast."""
    last_error = None
    for attempt in range(1, 4):
        try:
            return action()
        except (OperationalError, DBAPIError) as exc:
            last_error = exc
            if attempt == 3:
                raise
            db_engine.dispose()
            delay = float(attempt * 2)
            print(f"[LTE_OFFSET][GEO_DB_RETRY] stage={label} attempt={attempt}/3 delay_s={delay:.0f} error={type(exc).__name__}", flush=True)
            time.sleep(delay)
    raise last_error  # pragma: no cover


def _read_cached(db_engine, project_id: int, dataset_id: int, grid_ids: pd.Series) -> pd.DataFrame:
    ids = grid_ids.astype(str).drop_duplicates().tolist()
    if not ids:
        return pd.DataFrame(columns=["grid_id", "clutter_class", "land_cover_class"])
    parts = []
    for start in range(0, len(ids), 1000):
        chunk = ids[start:start + 1000]
        params = {"project_id": project_id, "dataset_id": dataset_id}
        marks = []
        for index, grid_id in enumerate(chunk):
            key = f"g{index}"
            params[key] = grid_id
            marks.append(f":{key}")
        statement = text(f"""
            SELECT grid_id, clutter_class, land_cover_class
            FROM tbl_project_clutter_tile
            WHERE project_id = :project_id AND geo_dataset_id = :dataset_id
              AND is_active = 1 AND grid_id IN ({', '.join(marks)})
        """)
        parts.append(_with_db_retry(
            db_engine, lambda: pd.read_sql(statement, db_engine, params=params), "read_clutter_tiles"
        ))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _grid_geometry(grid_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build the exact production grid polygons used by the RF candidates."""
    rows = grid_df[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]].drop_duplicates("grid_id").copy()
    fallback_lat = 12.5 / 111320.0
    fallback_lon = fallback_lat / np.maximum(np.cos(np.radians(rows["center_lat"].to_numpy(float))), 1e-6)
    rows["min_lat"] = pd.to_numeric(rows["min_lat"], errors="coerce").fillna(rows["center_lat"] - fallback_lat)
    rows["max_lat"] = pd.to_numeric(rows["max_lat"], errors="coerce").fillna(rows["center_lat"] + fallback_lat)
    rows["min_lon"] = pd.to_numeric(rows["min_lon"], errors="coerce").fillna(rows["center_lon"] - fallback_lon)
    rows["max_lon"] = pd.to_numeric(rows["max_lon"], errors="coerce").fillna(rows["center_lon"] + fallback_lon)
    return gpd.GeoDataFrame(
        rows[["grid_id"]],
        geometry=[box(min_lon, min_lat, max_lon, max_lat) for min_lat, max_lat, min_lon, max_lon in rows[["min_lat", "max_lat", "min_lon", "max_lon"]].to_numpy()],
        crs="EPSG:4326",
    )


def _swap_xy(geom):
    return shapely_transform(lambda x, y, z=None: (y, x) if z is None else (y, x, z), geom)


def _align_buildings(buildings: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if buildings.empty:
        return buildings
    area = grid.geometry.union_all()
    direct = int(buildings.geometry.intersects(area).sum())
    swapped = buildings.copy()
    swapped["geometry"] = swapped.geometry.apply(_swap_xy)
    swapped_hits = int(swapped.geometry.intersects(area).sum())
    return swapped if swapped_hits > direct else buildings


def _safe_area_ratio(grid: gpd.GeoDataFrame, layer: gpd.GeoDataFrame, name: str) -> pd.Series:
    """Per-tile polygon-coverage ratio. Replaces the old _clip_area_ratio,
    which had two real bugs, both confirmed against live project data before
    this fix:

    1. NaN-from-invalid-geometry: an invalid (self-intersecting) polygon's
       .intersection().area can come back NaN, and Python's min(1.0, NaN)
       silently returns 1.0 instead of excluding it - this is what forced
       91.9% of project 246's tiles into "Dense Urban" under the old
       building-ratio logic. Fixed by repairing invalid rings with
       buffer(0) first and never passing a non-finite area to min().

    2. Double-counted overlap: Overture's land_use layer frequently
       represents one real feature (a park) as multiple overlapping
       polygons - a park outline plus a separately-tagged pitch/track/
       playground nested inside it. Summing each hit's intersection area
       independently counts that overlapping ground twice, inflating
       ratios well past their real value. Fixed by dissolving the whole
       layer into one non-overlapping geometry (unary_union) before
       measuring any tile's coverage.
    """
    grid_utm = grid.to_crs(_choose_utm_crs(grid))
    out = pd.Series(0.0, index=grid["grid_id"].astype(str))
    if layer is None or layer.empty:
        return out
    polys = layer[layer.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(grid_utm.crs).copy()
    if polys.empty:
        return out
    polys["geometry"] = polys.geometry.buffer(0)
    polys = polys[~polys.geometry.is_empty & polys.geometry.notna()]
    if polys.empty:
        return out

    dissolved = polys.geometry.union_all()
    dissolved_parts = [dissolved] if dissolved.geom_type == "Polygon" else list(dissolved.geoms)
    merged = gpd.GeoDataFrame({"geometry": dissolved_parts}, crs=polys.crs)

    index = merged.sindex
    values = []
    for position, row in enumerate(grid_utm.itertuples()):
        hits = list(index.query(row.geometry, predicate="intersects"))
        area = 0.0
        for h in hits:
            try:
                a = merged.geometry.iloc[h].intersection(row.geometry).area
            except Exception:
                a = 0.0
            if np.isfinite(a):
                area += a
        tile_area = row.geometry.area
        values.append(0.0 if not np.isfinite(tile_area) or tile_area <= 0 else min(1.0, area / tile_area))
        if (position + 1) % 2000 == 0:
            print(f"[LTE_OFFSET][PHASE27_OVERLAY] layer={name} tiles={position + 1}/{len(grid_utm)}", flush=True)
    return pd.Series(values, index=grid["grid_id"].astype(str))


def _impute_heights(buildings: gpd.GeoDataFrame, obat_csv_path: str | None) -> tuple[gpd.GeoDataFrame, str]:
    work = buildings.copy().reset_index(drop=True)
    direct = pd.to_numeric(work.get("building_height_m"), errors="coerce")
    levels = pd.to_numeric(work.get("building_levels"), errors="coerce") * 3.0
    work["height_m"] = direct.fillna(levels)
    source = "database_height"
    if obat_csv_path:
        path = Path(obat_csv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Configured GHS-OBAT extract does not exist: {path}")
        obat = pd.read_csv(path)
        expected = {"lat", "lon", "height"}
        if not expected.issubset(obat.columns):
            raise ValueError("GHS-OBAT extract must contain lat, lon, and height columns")
        points = gpd.GeoDataFrame(obat, geometry=gpd.points_from_xy(obat["lon"], obat["lat"]), crs="EPSG:4326")
        work["building_row_id"] = np.arange(len(work))
        joined = gpd.sjoin(points[["height", "geometry"]], work[["building_row_id", "geometry"]], how="inner", predicate="within")
        matched = joined.groupby("building_row_id")["height"].mean()
        work["height_m"] = work["building_row_id"].map(matched).fillna(work["height_m"])
        source = f"GHS-OBAT matched={int(work['building_row_id'].isin(matched.index).sum())}/{len(work)}"
    known = work["height_m"].notna()
    if known.any() and (~known).any():
        from scipy.spatial import cKDTree
        utm = work.to_crs(_choose_utm_crs(work))
        pts = np.asarray([[geom.centroid.x, geom.centroid.y] for geom in utm.geometry])
        tree = cKDTree(pts[known.to_numpy()])
        means = work.loc[known, "height_m"].to_numpy(float)
        missing_positions = np.flatnonzero((~known).to_numpy())
        for radius in (150.0, 300.0, 600.0):
            unresolved = [position for position in missing_positions if pd.isna(work.at[position, "height_m"])]
            if not unresolved:
                break
            for position, neighbours in zip(unresolved, tree.query_ball_point(pts[unresolved], radius)):
                if neighbours:
                    work.at[position, "height_m"] = float(means[neighbours].mean())
    # Whatever is still missing after direct height, GHS-OBAT, and neighbour
    # imputation falls back to this project's own real known-height mean -
    # never a fixed constant. A universal fallback (e.g. 12.0m always)
    # silently pins every such building into the same height tier
    # regardless of what the project actually looks like.
    known_mean = float(work.loc[known, "height_m"].mean()) if known.any() else 12.0
    work["height_m"] = pd.to_numeric(work["height_m"], errors="coerce").fillna(known_mean).clip(3.0, 120.0)
    return work, source


def _fetch_overture_context(grid: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    try:
        import overturemaps.core as overture
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("overturemaps is required to build a Phase-27 clutter cache") from exc
    polygon = grid.geometry.union_all()
    bbox = polygon.bounds
    layers = {}
    for kind in ("segment", "water", "land_cover", "land_use"):
        print(
            f"[LTE_OFFSET][OVERTURE_FETCH] layer={kind} state=start "
            f"connect_timeout_s={OVERTURE_CONNECT_TIMEOUT_S} request_timeout_s={OVERTURE_REQUEST_TIMEOUT_S}",
            flush=True,
        )
        layer = overture.geodataframe(
            kind,
            bbox=bbox,
            connect_timeout=OVERTURE_CONNECT_TIMEOUT_S,
            request_timeout=OVERTURE_REQUEST_TIMEOUT_S,
        )
        if layer.crs is None:
            layer = layer.set_crs("EPSG:4326")
        clipped = gpd.clip(layer, polygon)
        layers[kind] = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
        print(f"[LTE_OFFSET][OVERTURE_FETCH] layer={kind} state=done rows={len(layers[kind])}", flush=True)
    return layers


def _grid_resolution_m(grid: gpd.GeoDataFrame) -> float:
    """Tile edge length in metres, read back from the tile polygons themselves.

    Recorded on the dataset row so `resolution_m` describes the grid the tiles
    were actually classified on. The cache key is the boundary hash, never this
    value, so it is provenance only - but a 50 m dataset labelled 25 m makes the
    table unreadable to anyone auditing which grid a cached run used.
    """
    if grid.empty:
        return 0.0
    bounds = grid.geometry.bounds
    metres = (bounds["maxy"] - bounds["miny"]).to_numpy(float) * 111320.0
    metres = metres[np.isfinite(metres) & (metres > 0)]
    return round(float(np.median(metres)), 1) if metres.size else 0.0


def _phase27_dataset(db_engine, project_id: int, grid: gpd.GeoDataFrame) -> int:
    layout = "|".join(sorted(
        f"{grid_id}:{geometry.wkb_hex}" for grid_id, geometry in zip(grid["grid_id"].astype(str), grid.geometry)
    ))
    boundary_hash = hashlib.sha256(layout.encode("utf-8")).hexdigest()
    resolution_m = _grid_resolution_m(grid)
    def action():
      with db_engine.begin() as conn:
        existing = conn.execute(text("""
            SELECT id FROM tbl_project_geo_dataset
            WHERE project_id = :project_id AND dataset_type = 'phase27_clutter'
              AND source_name = :source_name AND source_version = :source_version
              AND boundary_hash = :boundary_hash AND is_active = 1
            ORDER BY id DESC LIMIT 1
        """), {"project_id": project_id, "source_name": PHASE27_CLUTTER_SOURCE,
               "source_version": PHASE27_CLASSIFIER_VERSION, "boundary_hash": boundary_hash}).scalar()
        if existing:
            return int(existing)

        # Every previously-active dataset for this project (any version) is
        # about to be superseded. Its child rows - tbl_project_clutter_tile,
        # the Overture context layers this saved into tbl_savepolygon, and
        # tbl_project_building_profile - must be deleted here, not just have
        # the dataset row itself deactivated. Without this they accumulate
        # forever: confirmed directly against the live DB, a project rerun
        # across classifier versions left 5 dataset rows and 77,032 orphaned
        # tile rows behind before this fix. Building rows in tbl_savepolygon
        # (source_name='overture_building'/'osm_building') are never at risk
        # here - they're inserted with no geo_dataset_id at all, so they can
        # never match this delete's geo_dataset_id filter.
        stale_ids = [row[0] for row in conn.execute(text("""
            SELECT id FROM tbl_project_geo_dataset
            WHERE project_id = :project_id AND dataset_type = 'phase27_clutter' AND is_active = 1
        """), {"project_id": project_id}).fetchall()]
        for stale_id in stale_ids:
            conn.execute(text("DELETE FROM tbl_project_clutter_tile WHERE project_id = :project_id AND geo_dataset_id = :dataset_id"),
                         {"project_id": project_id, "dataset_id": stale_id})
            conn.execute(text("DELETE FROM tbl_savepolygon WHERE project_id = :project_id AND geo_dataset_id = :dataset_id"),
                         {"project_id": project_id, "dataset_id": stale_id})
            conn.execute(text("DELETE FROM tbl_project_building_profile WHERE project_id = :project_id AND geo_dataset_id = :dataset_id"),
                         {"project_id": project_id, "dataset_id": stale_id})
        conn.execute(text("""UPDATE tbl_project_geo_dataset SET is_active = 0
            WHERE project_id = :project_id AND dataset_type = 'phase27_clutter' AND is_active = 1"""), {"project_id": project_id})
        result = conn.execute(text("""
            INSERT INTO tbl_project_geo_dataset
                (project_id, dataset_type, source_name, source_version, boundary_hash,
                 resolution_m, metadata_json, is_active)
            VALUES (:project_id, 'phase27_clutter', :source_name, :source_version,
                    :boundary_hash, :resolution_m,
                    JSON_OBJECT('classifier', 'overture-water-green->image-texture-k4(open/suburban/urban/dense-urban)'), 1)
        """), {"project_id": project_id, "source_name": PHASE27_CLUTTER_SOURCE,
               "source_version": PHASE27_CLASSIFIER_VERSION, "boundary_hash": boundary_hash,
               "resolution_m": resolution_m})
        return int(result.lastrowid)
    return _with_db_retry(db_engine, action, "phase27_dataset")


def _save_phase27_tiles(db_engine, project_id: int, dataset_id: int, grid: gpd.GeoDataFrame, values: pd.DataFrame) -> None:
    lookup = values.set_index("grid_id")
    rows = [(
        project_id, dataset_id, str(row.grid_id), row.geometry.wkt,
        str(lookup.at[str(row.grid_id), "clutter_class"]),
        str(lookup.at[str(row.grid_id), "land_cover_class"]),
    ) for row in grid.itertuples()]

    # Manual multi-row INSERT (not SQLAlchemy's execute(statement, list_of_dicts)
    # executemany path) - same reason as _save_overture_context_layers: pymysql's
    # executemany batching does not reliably merge rows when a VALUES expression
    # contains a function call like ST_GeomFromText(...). geometry_geom is a
    # NOT NULL, spatially-indexed column built from the same WKT already stored
    # (as portable text) in geometry_wkt.
    def make_action(batch):
        def action():
            placeholders = ", ".join(
                ["(%s, %s, %s, %s, %s, %s, 25.0, 1, ST_GeomFromText(%s, 4326, 'axis-order=long-lat'))"] * len(batch)
            )
            params = []
            for proj_id, ds_id, grid_id, geometry_wkt, clutter_class, land_cover_class in batch:
                params.extend([proj_id, ds_id, grid_id, geometry_wkt, clutter_class, land_cover_class, geometry_wkt])
            with db_engine.begin() as conn:
                conn.exec_driver_sql(
                    "INSERT INTO tbl_project_clutter_tile "
                    "(project_id, geo_dataset_id, grid_id, geometry_wkt, clutter_class, land_cover_class, "
                    "resolution_m, is_active, geometry_geom) "
                    f"VALUES {placeholders} "
                    "ON DUPLICATE KEY UPDATE geometry_wkt=VALUES(geometry_wkt), clutter_class=VALUES(clutter_class), "
                    "land_cover_class=VALUES(land_cover_class), is_active=1, geometry_geom=VALUES(geometry_geom)",
                    tuple(params),
                )
        return action

    # Keep transactions short.  A project cache has thousands of grid tiles and
    # one long upsert transaction can be blocked by unrelated DB activity.
    for start in range(0, len(rows), 250):
        batch = rows[start:start + 250]
        _with_db_retry(db_engine, make_action(batch), f"save_clutter_tiles_{start // 250 + 1}")


GHS_OBAT_MAX_BYTES = 100 * 1024 * 1024  # above this a country extract is treated as impractical; falls back to OSM


def resolve_building_heights(
    building_df: pd.DataFrame, grid_df: pd.DataFrame, project_id: int, db_engine, region: str, ghs_obat_csv_path: str | None
) -> tuple[pd.DataFrame, str]:
    """Baseline-time height resolution, decoupled from Phase-27 clutter classification.

    GHS-OBAT is only used where the country extract is a practical single
    file to load and match against (Taiwan). Everywhere else - large-country
    extracts such as India's, which routinely exceed 100MB - height falls
    back to whatever OSM/Overture already supplied at building-fetch time
    (tbl_savepolygon.height_m / building_levels), then neighbour imputation,
    then this project's own known-height mean. Resolved heights are written
    straight back into tbl_savepolygon.height_m via UPDATE - a plain column
    on the building's own row, not a new dataset-versioned table - so
    tbl_project_building_profile is no longer written to at all.
    """
    buildings = building_df_to_gdf(building_df)
    if buildings.empty or "id" not in buildings.columns:
        return building_df, "no_buildings"
    buildings = _align_buildings(buildings, _grid_geometry(grid_df))

    usable_obat_path = None
    if ghs_obat_csv_path and (region or "").lower() == "taiwan":
        path = Path(ghs_obat_csv_path).expanduser()
        if path.is_file() and path.stat().st_size <= GHS_OBAT_MAX_BYTES:
            usable_obat_path = str(path)

    buildings, height_source = _impute_heights(buildings, usable_obat_path)

    resolved = pd.to_numeric(buildings["height_m"], errors="coerce")
    ids = pd.to_numeric(buildings["id"], errors="coerce")
    valid = ids.notna() & resolved.notna()
    height_by_id = dict(zip(ids[valid].astype(int), resolved[valid].astype(float)))

    if height_by_id:
        rows = [{"building_id": bid, "height_m": height} for bid, height in height_by_id.items()]
        def action():
            with db_engine.begin() as conn:
                for start in range(0, len(rows), 500):
                    conn.execute(
                        text("UPDATE tbl_savepolygon SET height_m = :height_m WHERE id = :building_id AND project_id = :project_id"),
                        [{**row, "project_id": project_id} for row in rows[start:start + 500]],
                    )
        _with_db_retry(db_engine, action, "update_building_heights")

    out = building_df.copy()
    out["building_height_m"] = pd.to_numeric(out["id"], errors="coerce").map(height_by_id)
    return out, height_source


def _explode_polygons(layer: gpd.GeoDataFrame) -> list:
    """Singlepart Polygon geometries only - tbl_savepolygon.region is
    polygon-typed and MySQL rejects a MultiPolygon insert into it."""
    if layer is None or layer.empty:
        return []
    out = []
    for geom in layer.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            out.append(geom)
        elif geom.geom_type == "MultiPolygon":
            out.extend(part for part in geom.geoms if not part.is_empty)
    return out


def _save_overture_context_layers(db_engine, project_id: int, dataset_id: int, context: dict) -> None:
    """Persist the real Overture geometry this project's clutter was
    actually computed from - roads/highway/railway/water/land_cover/
    land_use - not just the derived clutter_class summary, so the
    frontend can render these as real, independent layers (matching the
    reference tool) instead of trying to reconstruct a road's shape from
    a tile label. Reuses tbl_savepolygon, which is already generic and
    source-tagged/dataset-versioned for exactly this - no new table.
    Line features (road/highway/railway) go in the table's existing
    generic `geometry` column; polygon features (water/land_cover/
    land_use) go in `region`, the same column buildings already use.
    Called only on a cache miss - once per dataset version, not per run.
    """
    segment = context.get("segment")
    if segment is not None and not segment.empty:
        subtype = segment.get("subtype", pd.Series(index=segment.index, dtype=str)).astype(str)
        cls = segment.get("class", pd.Series(index=segment.index, dtype=str)).astype(str)
        is_rail = subtype.eq("rail")
        is_highway = ~is_rail & cls.isin({"trunk", "primary", "secondary"})
        railway, highway, road = segment[is_rail], segment[is_highway], segment[~is_rail & ~is_highway]
    else:
        railway = highway = road = gpd.GeoDataFrame()

    polygon_layers = [
        ("overture_water", context.get("water")),
        ("overture_land_cover", context.get("land_cover")),
        ("overture_land_use", context.get("land_use")),
    ]
    line_layers = [
        ("overture_road", road),
        ("overture_highway", highway),
        ("overture_railway", railway),
    ]

    # A manually-built, single multi-row INSERT (not SQLAlchemy's
    # execute(statement, list_of_dicts) executemany path) - pymysql's
    # executemany batching does not reliably merge rows when a VALUES
    # expression contains a function call like ST_GeomFromText(...); it
    # can scramble parameters across rows. Every other geometry insert in
    # this codebase that survives real use (tools/buildings) already
    # avoids this the same way.
    def action():
        with db_engine.begin() as conn:
            for source_name, layer in polygon_layers:
                geoms = _explode_polygons(layer)
                for start in range(0, len(geoms), 250):
                    batch = geoms[start:start + 250]
                    if not batch:
                        continue
                    placeholders = ", ".join(
                        ["(%s, %s, ST_GeomFromText(%s, 4326, 'axis-order=long-lat'), %s, %s)"] * len(batch)
                    )
                    params = []
                    for g in batch:
                        params.extend([project_id, source_name, g.wkt, dataset_id, source_name])
                    conn.exec_driver_sql(
                        f"INSERT INTO tbl_savepolygon (project_id, name, region, geo_dataset_id, source_name) "
                        f"VALUES {placeholders}",
                        tuple(params),
                    )
                print(f"[LTE_OFFSET][PHASE27_CONTEXT_SAVE] source_name={source_name} rows={len(geoms)}", flush=True)

            for source_name, layer in line_layers:
                if layer is None or layer.empty:
                    print(f"[LTE_OFFSET][PHASE27_CONTEXT_SAVE] source_name={source_name} rows=0", flush=True)
                    continue
                geoms = [g for g in layer.geometry if g is not None and not g.is_empty]
                saved = 0
                for start in range(0, len(geoms), 250):
                    rows = []
                    for g in geoms[start:start + 250]:
                        # tbl_savepolygon.region is polygon-typed and NOT
                        # NULL - a line can't go there. A hairline buffer
                        # satisfies the constraint without claiming false
                        # precision; the real line geometry (what any real
                        # consumer should actually render) is in `geometry`.
                        region_poly = g.buffer(0.00001)
                        if region_poly.geom_type == "MultiPolygon":
                            # tbl_savepolygon.region is strictly POLYGON-typed
                            # (not generic GEOMETRY) - a disjoint MultiLineString
                            # (e.g. a road split across a tunnel/bridge gap)
                            # buffers into a MultiPolygon, which MySQL rejects
                            # for that column. It's only a NOT-NULL placeholder
                            # here (the real shape is in `geometry`), so keep
                            # just the largest part.
                            region_poly = max(region_poly.geoms, key=lambda p: p.area)
                        if region_poly.is_empty:
                            continue
                        rows.append((region_poly.wkt, g.wkt))
                    if not rows:
                        continue
                    placeholders = ", ".join(
                        ["(%s, %s, ST_GeomFromText(%s, 4326, 'axis-order=long-lat'), "
                         "ST_GeomFromText(%s, 4326, 'axis-order=long-lat'), %s, %s)"] * len(rows)
                    )
                    params = []
                    for region_wkt, geom_wkt in rows:
                        params.extend([project_id, source_name, region_wkt, geom_wkt, dataset_id, source_name])
                    conn.exec_driver_sql(
                        f"INSERT INTO tbl_savepolygon (project_id, name, region, geometry, geo_dataset_id, source_name) "
                        f"VALUES {placeholders}",
                        tuple(params),
                    )
                    saved += len(rows)
                print(f"[LTE_OFFSET][PHASE27_CONTEXT_SAVE] source_name={source_name} rows={saved}", flush=True)

    _with_db_retry(db_engine, action, "save_overture_context_layers")


def load_or_build_phase27_clutter(grid_df: pd.DataFrame, project_id: int, db_engine) -> tuple[pd.DataFrame, dict]:
    """Production implementation of Phase 27's vector clutter classifier.

    Building height is no longer resolved here at all - see
    resolve_building_heights, called separately at baseline time. This
    classifier never depended on building geometry/height for its decision;
    the only reason building_df was ever threaded through this function was
    to populate tbl_project_building_profile, which is no longer written to.
    """
    if db_engine is None:
        raise RuntimeError("Project database engine is required for the Phase-27 geospatial cache")
    grid = _grid_geometry(grid_df)
    dataset_id = _phase27_dataset(db_engine, int(project_id), grid)
    cached = _read_cached(db_engine, int(project_id), dataset_id, grid["grid_id"])
    if len(cached) == len(grid):
        return cached, {
            "enabled": True, "source": PHASE27_CLUTTER_SOURCE, "dataset_id": dataset_id,
            "cache": "hit", "tiles": len(cached),
        }

    context = _fetch_overture_context(grid)
    _save_overture_context_layers(db_engine, int(project_id), dataset_id, context)

    # --- Water / Vegetation: real Overture vector polygons, not height/density ---
    water_layer = context["water"]
    land_use = context["land_use"]
    green_layer = land_use[
        land_use.get("subtype", pd.Series(index=land_use.index, dtype=str)).isin(GREEN_LU_SUBTYPES)
        | land_use.get("class", pd.Series(index=land_use.index, dtype=str)).isin(GREEN_LU_CLASS)
    ].copy()
    if green_layer.empty:
        green_layer = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    # green_ratio must measure ONLY dry land, not the raw green polygon's
    # footprint - a park's Overture boundary routinely encloses a river
    # running through it, so the two are subtracted before measuring, not
    # compared as if independent. See WATER_GREEN_FLOOR module docstring.
    utm_crs = _choose_utm_crs(grid)
    green_layer_utm = green_layer.to_crs(utm_crs) if not green_layer.empty else green_layer
    water_layer_utm = water_layer.to_crs(utm_crs) if not water_layer.empty else water_layer
    if not green_layer_utm.empty and not water_layer_utm.empty:
        water_dissolved = water_layer_utm.geometry.buffer(0).union_all()
        green_layer_utm = green_layer_utm.copy()
        green_layer_utm["geometry"] = green_layer_utm.geometry.buffer(0).difference(water_dissolved)
        green_layer_utm = green_layer_utm[~green_layer_utm.geometry.is_empty & green_layer_utm.geometry.notna()]

    water_ratio = _safe_area_ratio(grid, water_layer, "water")
    green_ratio = _safe_area_ratio(grid, green_layer_utm, "green")

    # --- Open / Suburban / Urban / Dense Urban: image-texture K-Means, ---
    # --- fit only on tiles that aren't already Water/Vegetation ---
    min_lon, min_lat, max_lon, max_lat = grid.total_bounds
    center_lat, center_lon = (min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0
    zoom = _pick_zoom(min_lon, min_lat, max_lon, max_lat)
    print(f"[LTE_OFFSET][PHASE27_IMAGE] fetching satellite tile zoom={zoom} center=({center_lat:.5f},{center_lon:.5f})", flush=True)
    sat_img = _fetch_satellite_tile(center_lat, center_lon, zoom)
    projector = _WebMercatorProjector(center_lat, center_lon, zoom)
    img_h, img_w = sat_img.shape[:2]

    bboxes = []
    is_candidate = np.zeros(len(grid), dtype=bool)
    grid_ids = grid["grid_id"].astype(str).tolist()
    for position, row in enumerate(grid.itertuples()):
        tminx, tminy, tmaxx, tmaxy = row.geometry.bounds
        px1, py1 = projector.coord_to_pixel(tminy, tminx)
        px2, py2 = projector.coord_to_pixel(tmaxy, tmaxx)
        x1, x2 = sorted((max(0, min(img_w - 1, px1)), max(0, min(img_w - 1, px2))))
        y1, y2 = sorted((max(0, min(img_h - 1, py1)), max(0, min(img_h - 1, py2))))
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1
        bboxes.append((x1, y1, x2, y2))
        grid_id = grid_ids[position]
        is_candidate[position] = _resolve_water_or_green(water_ratio.get(grid_id, 0.0), green_ratio.get(grid_id, 0.0)) is None

    candidate_mask = np.zeros((img_h, img_w), dtype=bool)
    for position, (x1, y1, x2, y2) in enumerate(bboxes):
        if is_candidate[position]:
            candidate_mask[y1:y2, x1:x2] = True

    blur_kernel_px = _scaled_blur_kernel_px(zoom)
    density_map = _edge_density_map(sat_img, blur_kernel_px)
    tiers = _kmeans_tiers_on_mask(density_map, candidate_mask, k=4)
    print(f"[LTE_OFFSET][PHASE27_IMAGE] blur_kernel_px={blur_kernel_px} candidate_pixels={int(candidate_mask.sum())}/{img_h * img_w} "
          f"candidate_tiles={int(is_candidate.sum())}/{len(grid)}", flush=True)

    records = []
    for position, grid_id in enumerate(grid_ids):
        wr, gr = water_ratio.get(grid_id, 0.0), green_ratio.get(grid_id, 0.0)
        water_or_green = _resolve_water_or_green(wr, gr)
        if water_or_green is not None:
            label = water_or_green
            rule = "water_ratio_majority" if water_or_green == "Water" else "green_ratio_majority"
        else:
            x1, y1, x2, y2 = bboxes[position]
            valid = tiers[y1:y2, x1:x2]
            valid = valid[valid >= 0]
            tier = _mode_or(valid, default=-1) if valid.size else -1
            label = TILE4_LABELS.get(tier, "Open")
            rule = "image_texture_k4"
        records.append({"grid_id": grid_id, "clutter_class": label, "land_cover_class": rule})
    result = pd.DataFrame(records)
    _save_phase27_tiles(db_engine, int(project_id), dataset_id, grid, result)
    return result, {
        "enabled": True, "source": PHASE27_CLUTTER_SOURCE, "dataset_id": dataset_id, "cache": "miss",
        "tiles": len(result),
        "overture_features": {name: int(len(layer)) for name, layer in context.items()},
        "classes": {str(k): int(v) for k, v in result["clutter_class"].value_counts().items()},
    }
