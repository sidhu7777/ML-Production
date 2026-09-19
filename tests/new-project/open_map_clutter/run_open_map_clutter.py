"""
Runs Research/open_map.py's image-based clutter classifier (Canny edge
density -> 4-tier K-Means) against a project's real Taiwan coverage
area, unmodified, and samples its output onto the same grid tiles
production already classified - so the two can be compared tile-for-tile.

Production code is NOT imported or modified. The only production
touchpoint is a read-only SQL query (via the existing DB engine) to
(a) know which real grid tiles to sample onto and (b) pull production's
already-stored clutter_class for side-by-side reference in the
dashboard. Every classification value this script produces comes
entirely from Research/open_map.py's own classes, called as-is.

Output: clutter_comparison_<project_id>.csv + Density_Zones_Map_project<id>.jpg
in this same folder. Run with the project's venv:
    venv\\Scripts\\python.exe tests\\new-project\\open_map_clutter\\run_open_map_clutter.py
Pick the project with CLUTTER_PROJECT_ID (defaults to 210):
    set CLUTTER_PROJECT_ID=246 && venv\\Scripts\\python.exe tests\\new-project\\open_map_clutter\\run_open_map_clutter.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt as shapely_wkt
from shapely.geometry import box

ML_ROOT = Path(r"C:\Users\PC\Desktop\S-Tracer Exe\S-Tracer Exe\ML")
sys.path.insert(0, str(ML_ROOT))

from Research.open_map import GoogleMapsAPIHook, WebMercatorProjector, HybridZoneExtractor  # noqa: E402
from tools.lte_prediction.ml_engine import engine  # noqa: E402  (read-only DB access only)

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ID = int(os.environ.get("CLUTTER_PROJECT_ID", 210))
REGION = "taiwan"

# land_use-side of production's green taxonomy, copied here as plain data
# (not imported) so this script has no dependency on production's
# classification code. Production's land_cover-side (forest/shrub/grass)
# is deliberately NOT used here - see the comment in main() for why.
GREEN_LU_SUBTYPES = {"park", "recreation", "horticulture", "agriculture"}
GREEN_LU_CLASS = {"park", "garden", "grass", "recreation_ground", "village_green", "pitch", "nature_reserve"}

# Same as tests/baseline/clutter_map_streamlit.py's own HIGHWAY_CLASSES -
# real Overture segment.class values, used to split one segment fetch into
# roads/highway/railway for the reference layers (not the classification).
HIGHWAY_CLASSES = {"trunk", "primary", "secondary"}

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not GOOGLE_MAPS_API_KEY:
    raise RuntimeError("GOOGLE_MAPS_API_KEY is not set in ML/.env")

IMG_WIDTH = 640
IMG_HEIGHT = 640
SCALE = 2


def _load_production_grid() -> gpd.GeoDataFrame:
    """Real grid tiles production already classified for project 246 -
    read-only, used only so this script's output lines up tile-for-tile
    with what's already in the DB. Picks whichever phase27_clutter
    dataset has the most tiles (i.e. the one production actually
    finished writing), not just the newest id."""
    db_engine = engine["taiwan"]
    with db_engine.begin() as conn:
        best = conn.exec_driver_sql(
            """
            SELECT t.geo_dataset_id, COUNT(*) AS n
            FROM tbl_project_clutter_tile t
            JOIN tbl_project_geo_dataset d ON d.id = t.geo_dataset_id
            WHERE t.project_id = %s AND d.dataset_type = 'phase27_clutter'
            GROUP BY t.geo_dataset_id
            ORDER BY n DESC, t.geo_dataset_id DESC
            LIMIT 1
            """,
            (PROJECT_ID,),
        ).fetchone()
        if not best:
            raise RuntimeError(f"No phase27_clutter tiles found for project {PROJECT_ID} - run the baseline first.")
        dataset_id = best[0]
        print(f"[GRID] Using geo_dataset_id={dataset_id} ({best[1]} tiles) as the reference production grid.")

        rows = conn.exec_driver_sql(
            """
            SELECT grid_id, geometry_wkt AS wkt, clutter_class AS production_clutter_class
            FROM tbl_project_clutter_tile
            WHERE project_id = %s AND geo_dataset_id = %s
            """,
            (PROJECT_ID, dataset_id),
        ).fetchall()

    df = pd.DataFrame(rows, columns=["grid_id", "wkt", "production_clutter_class"])
    df["geometry"] = df["wkt"].apply(shapely_wkt.loads)
    grid = gpd.GeoDataFrame(df[["grid_id", "geometry", "production_clutter_class"]], crs="EPSG:4326")
    return grid


def _pick_zoom(min_lon, min_lat, max_lon, max_lat, img_w=IMG_WIDTH, img_h=IMG_HEIGHT, margin=1.08):
    """Smallest zoom level whose single 640x640 tile still covers the
    whole grid bounding box, so one Static Maps call is enough - matching
    how Research/open_map.py's own __main__ assumes a single image covers
    the target area.

    `scale` (Google's retina/DPI parameter) does NOT extend the
    geographic area a tile covers - it only doubles pixel density for the
    SAME area (confirmed against WebMercatorProjector itself: its
    top-left world offset is computed from img_logical_w un-scaled, and
    `scale` is applied only afterwards, at the world-to-output-pixel
    step). Using img_w*scale here as if it were extra coverage picked a
    zoom level too far in - the fetched image covered barely half the
    real bounding box, and every tile outside that got clamped to the
    image edge, producing long smeared/banded strips instead of real
    classification."""
    mean_lat = (min_lat + max_lat) / 2.0
    width_m = (max_lon - min_lon) * 111320.0 * math.cos(math.radians(mean_lat)) * margin
    height_m = (max_lat - min_lat) * 111320.0 * margin
    needed_m = max(width_m, height_m)
    eff_px = min(img_w, img_h)  # scale (retina/DPI) does not add coverage, see docstring
    for zoom in range(20, 9, -1):
        meters_per_pixel = 156543.03392 * math.cos(math.radians(mean_lat)) / (2 ** zoom)
        if eff_px * meters_per_pixel >= needed_m:
            return zoom
    return 10


def _mode_or(values: np.ndarray, default=0):
    if values.size == 0:
        return default
    counts = np.bincount(values.astype(np.int64))
    return int(np.argmax(counts))


# Water is a priority presence class for this visual clutter grid. A strict
# majority threshold made river-edge 25 m tiles with 45-49% real water fall
# through into image-based Urban/Suburban/Open, even though the same Overture
# water polygon is plainly visible in the baseline dashboard. Vegetation still
# needs a stronger dry-land share because park polygons are broader and can sit
# beside built-up areas.
WATER_PRESENCE_FLOOR = 0.25
GREEN_PRESENCE_FLOOR = 0.30


def _resolve_water_or_green(wr: float, gr: float, water_hit: bool = False) -> str | None:
    """'Water', 'Vegetation', or None (a built-density candidate).

    The water and green ratios are computed from mutually-exclusive vectors
    because water is subtracted from green before this function is called.
    That means any water share here is real water, not park/vegetation bleed.
    Classify water first so river-edge tiles do not fall into KMeans urban
    classes just because water is below 50% of the square.

    `water_hit` is the Google roadmap water mask for the same tile. It catches
    tiny ponds/streams that are visually real but too small to occupy 25% of a
    25 m grid square. Without this, those pond tiles can still be swallowed by
    park/vegetation polygons or by image-density KMeans.
    """
    if water_hit or wr >= WATER_PRESENCE_FLOOR:
        return "Water"
    if gr >= GREEN_PRESENCE_FLOOR:
        return "Vegetation"
    return None


# Research/open_map.py's own __main__ uses ZOOM_LEVEL=14 as its default -
# the 151px blur kernel it hardcodes was implicitly calibrated for whatever
# real-world distance 151px happens to be AT THAT zoom. At any other zoom,
# meters-per-pixel is different, so the same 151px means a different real
# distance. _scaled_blur_kernel_px converts the kernel size so it always
# represents that SAME original real-world footprint, regardless of which
# zoom a given project actually needs.
REFERENCE_ZOOM = 14


def _scaled_blur_kernel_px(zoom: int, base_kernel_px: int = 151, reference_zoom: int = REFERENCE_ZOOM) -> int:
    """Meters-per-pixel exactly halves per zoom level in, doubles per zoom
    level out (regardless of latitude or the scale/DPI parameter - both
    cancel out of this ratio, since they'd apply identically to the
    reference and the actual zoom). So the correct pixel count at any zoom
    is just base_kernel_px * 2**(zoom - reference_zoom), rounded to the
    nearest odd integer (GaussianBlur requires an odd kernel size).
    Verified: at zoom=14 (the reference itself) this returns exactly 151,
    unchanged - it only ever diverges from 151 when a project's zoom
    differs from Research/open_map.py's own default."""
    scaled = base_kernel_px * (2.0 ** (zoom - reference_zoom))
    k = max(1, int(round(scaled)))
    if k % 2 == 0:
        k += 1
    return k


def _edge_density_map(sat_img: np.ndarray, blur_kernel_px: int) -> np.ndarray:
    """Exactly the Canny-edge + blur preprocessing from
    Research/open_map.py's HybridZoneExtractor.extract_building_density -
    duplicated here (not imported) because that method bundles this step
    together with a hardcoded, unmasked k=4 K-Means call with no hook to
    restrict which pixels get clustered. We need the density map on its
    own so K-Means can run k=4, only over tiles Overture hasn't already
    identified as Water/Vegetation - see the comment in main() for why.
    blur_kernel_px replaces the original's fixed 151 - see
    _scaled_blur_kernel_px for why a fixed pixel count doesn't mean a
    fixed real-world distance once zoom varies by project."""
    gray = cv2.cvtColor(sat_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 80, 160), np.ones((2, 2), np.uint8), iterations=1)
    return np.clip(cv2.multiply(cv2.GaussianBlur(edges, (blur_kernel_px, blur_kernel_px), 0), 1.2), 0, 255).astype(np.uint8)


def _kmeans_tiers_on_mask(density_map: np.ndarray, candidate_mask: np.ndarray, k: int) -> np.ndarray:
    """K-Means over only the pixels candidate_mask marks True, ranked so
    0=lowest edge density .. k-1=highest. Everywhere else is -1 (not
    classified by this step - Water/Vegetation tiles were already decided
    from Overture before this ever runs, so their pixels have no business
    influencing where the built-density cluster boundaries fall)."""
    tiers = np.full(density_map.shape, -1, dtype=np.int8)
    ys, xs = np.where(candidate_mask)
    if ys.size < k:
        return tiers
    values = np.float32(density_map[ys, xs].reshape(-1, 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    rank_of_label = {int(old): new for new, old in enumerate(np.argsort(centers.flatten()))}
    ranked = np.array([rank_of_label[int(lbl)] for lbl in labels.flatten()], dtype=np.int8)
    tiers[ys, xs] = ranked
    return tiers


def _fetch_overture_layer(kind: str, polygon) -> gpd.GeoDataFrame:
    """Live Overture Maps fetch for one theme, self-contained (no import
    from production's geo_inputs.py) - same public overturemaps package
    production uses, called directly."""
    import overturemaps.core as overture
    layer = overture.geodataframe(kind, bbox=polygon.bounds, connect_timeout=20, request_timeout=90)
    if layer.crs is None:
        layer = layer.set_crs("EPSG:4326")
    clipped = gpd.clip(layer, polygon)
    return clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()


def _safe_area_ratio(grid_utm: gpd.GeoDataFrame, layer_utm: gpd.GeoDataFrame) -> np.ndarray:
    """Per-tile polygon-coverage ratio, guarded against two real traps:

    1. NaN-from-invalid-geometry (found in production's version of this
       same calculation): an invalid polygon's .intersection().area can
       come back NaN, and Python's min(1.0, NaN) silently returns 1.0
       instead of excluding it. Fixed by repairing invalid rings with
       buffer(0) first and never passing a non-finite area to min().

    2. Double-counted overlap: Overture's land_use layer frequently
       represents one real feature (e.g. a park) as multiple overlapping
       polygons - a park outline plus a separately-tagged pitch/track/
       playground nested inside it. Summing each hit's intersection area
       independently counts that overlapping ground twice. Confirmed
       against real project 210 data: summed green area was 1.979 km2,
       properly merged it was 1.855 km2 (only 6.3% less overall) - but
       that overlap is concentrated on specific tiles, not spread evenly,
       and was enough to push their ratio over the 0.30 threshold: tiles
       crossing that threshold dropped from 6,558 to 2,277 once the
       polygons were merged first. Fixed by dissolving the whole layer
       into one non-overlapping geometry (unary_union) before measuring
       any tile's coverage, so overlapping source polygons for the same
       real feature can never be counted more than once.
    """
    out = np.zeros(len(grid_utm))
    if layer_utm is None or layer_utm.empty:
        return out
    polys = layer_utm[layer_utm.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if polys.empty:
        return out
    polys["geometry"] = polys.geometry.buffer(0)
    polys = polys[~polys.geometry.is_empty & polys.geometry.notna()]
    if polys.empty:
        return out

    dissolved = polys.geometry.unary_union
    dissolved_parts = [dissolved] if dissolved.geom_type == "Polygon" else list(dissolved.geoms)
    merged = gpd.GeoDataFrame({"geometry": dissolved_parts}, crs=polys.crs)

    index = merged.sindex
    for i, geom in enumerate(grid_utm.geometry):
        hits = list(index.query(geom, predicate="intersects"))
        area = 0.0
        for h in hits:
            try:
                a = merged.geometry.iloc[h].intersection(geom).area
            except Exception:
                a = 0.0
            if np.isfinite(a):
                area += a
        tile_area = geom.area
        out[i] = 0.0 if not np.isfinite(tile_area) or tile_area <= 0 else min(1.0, area / tile_area)
    return out


def main():
    grid = _load_production_grid()
    min_lon, min_lat, max_lon, max_lat = grid.total_bounds
    center_lat, center_lon = (min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0
    zoom = _pick_zoom(min_lon, min_lat, max_lon, max_lat)
    print(f"[AREA] bounds lon=({min_lon:.5f},{max_lon:.5f}) lat=({min_lat:.5f},{max_lat:.5f}) "
          f"center=({center_lat:.5f},{center_lon:.5f}) zoom={zoom} tiles={len(grid)}")

    # --- Water / Vegetation / Rural-Open come from Overture, not the image ---
    boundary_polygon = box(min_lon, min_lat, max_lon, max_lat)
    print("[OVERTURE] fetching water/land_use for water + vegetation gating...")
    water_layer = _fetch_overture_layer("water", boundary_polygon)
    land_use = _fetch_overture_layer("land_use", boundary_polygon)
    print(f"[OVERTURE] water={len(water_layer)} land_use={len(land_use)}")

    # land_cover (GREEN_LC_SUBTYPES: forest/shrub/grass) is deliberately NOT
    # used here, even though production's own taxonomy includes it. Checked
    # directly against real data: one single "forest"-tagged land_cover
    # polygon for this project measured 6.03 km2 - almost the ENTIRE 6.37
    # km2 grid. Overture's land_cover theme is a coarse regional background
    # classification (closer to a satellite land-cover raster) that does
    # not get updated when a city is built on what was broadly forest/
    # grassland terrain - it is not a reliable signal for "is this specific
    # tile actually vegetated today". land_use (park/garden/pitch/etc, real
    # individually-mapped features - 164 rows here, 1.98 km2 total, sizes
    # that actually make sense for real parks) is the trustworthy source.
    green_layer = land_use[
        land_use.get("subtype", pd.Series(index=land_use.index, dtype=str)).isin(GREEN_LU_SUBTYPES)
        | land_use.get("class", pd.Series(index=land_use.index, dtype=str)).isin(GREEN_LU_CLASS)
    ].copy()
    if green_layer.empty:
        green_layer = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    utm_crs = grid.estimate_utm_crs()
    grid_utm = grid.to_crs(utm_crs)
    water_layer_utm = water_layer.to_crs(utm_crs) if not water_layer.empty else water_layer
    green_layer_utm = green_layer.to_crs(utm_crs) if not green_layer.empty else green_layer
    water_ratio = _safe_area_ratio(grid_utm, water_layer_utm)

    # green_ratio must measure ONLY dry land, not the raw green polygon's
    # footprint. A park's Overture boundary routinely encloses the river
    # running through it (e.g. Fuzhou Art Riverside Park) - measured
    # directly against real data: tiles with water_ratio=0.46-0.50 still
    # showed green_ratio=1.000, because the park polygon's boundary simply
    # contains the water, not because the tile is genuinely all dry park
    # land. Comparing that raw green_ratio fairly against water_ratio
    # doesn't fix this - a containing polygon's ratio is mechanically >=
    # the ratio of whatever's nested inside it, regardless of how much
    # real water is there. The actual fix is to subtract the water
    # geometry from the green geometry FIRST, so the two become exclusive,
    # non-overlapping measurements of the same tile instead of two
    # independently-measured, possibly-nested facts.
    if not green_layer_utm.empty and not water_layer_utm.empty:
        water_dissolved = water_layer_utm.geometry.buffer(0).unary_union
        green_layer_utm = green_layer_utm.copy()
        green_layer_utm["geometry"] = green_layer_utm.geometry.buffer(0).difference(water_dissolved)
        green_layer_utm = green_layer_utm[~green_layer_utm.geometry.is_empty & green_layer_utm.geometry.notna()]
    green_ratio = _safe_area_ratio(grid_utm, green_layer_utm)
    print(f"[OVERTURE] tiles with water_ratio>=0.5: {int((water_ratio >= 0.5).sum())}, "
          f"green_ratio>=0.30: {int((green_ratio >= 0.30).sum())}")

    # Save the real, whole Overture polygons (not tile-quantized) for the
    # dashboard to draw directly - same approach as tests/baseline's own
    # dashboard, which renders water.geojson at full vector precision
    # instead of reducing it to the clutter grid. Clipped to the grid's own
    # extent (not the wider bbox we fetched with) so the map doesn't show
    # water/vegetation area outside the actual project shape. green_layer
    # here is already water-subtracted, so the two never overlap.
    grid_extent_utm = grid_utm.geometry.unary_union
    water_vector = gpd.GeoSeries([water_layer_utm.geometry.buffer(0).unary_union.intersection(grid_extent_utm)], crs=utm_crs) \
        if not water_layer_utm.empty else gpd.GeoSeries([], crs=utm_crs)
    green_vector = gpd.GeoSeries([green_layer_utm.geometry.unary_union.intersection(grid_extent_utm)], crs=utm_crs) \
        if not green_layer_utm.empty else gpd.GeoSeries([], crs=utm_crs)
    water_vector_ll = water_vector[~water_vector.is_empty].to_crs("EPSG:4326") if len(water_vector) else water_vector
    green_vector_ll = green_vector[~green_vector.is_empty].to_crs("EPSG:4326") if len(green_vector) else green_vector
    water_geojson = THIS_DIR / f"water_vector_project{PROJECT_ID}.geojson"
    green_geojson = THIS_DIR / f"vegetation_vector_project{PROJECT_ID}.geojson"
    gpd.GeoDataFrame(geometry=water_vector_ll).to_file(water_geojson, driver="GeoJSON")
    gpd.GeoDataFrame(geometry=green_vector_ll).to_file(green_geojson, driver="GeoJSON")
    print(f"[SAVED] {water_geojson}")
    print(f"[SAVED] {green_geojson}")

    # Buildings + roads/highway/railway, purely for visual comparison against
    # the baseline dashboard (tests/baseline/clutter_map_streamlit.py) - not
    # used anywhere in the classification itself. Same real Overture fields
    # the baseline uses: subtype=='rail' for railway, class in
    # {trunk,primary,secondary} for highway-tier roads, everything else in
    # 'segment' is a plain road - one fetch, no double-plotting.
    print("[OVERTURE] fetching building/segment for reference layers (buildings/roads/highway/railway)...")
    building_layer = _fetch_overture_layer("building", boundary_polygon)
    segment_layer = _fetch_overture_layer("segment", boundary_polygon)
    is_rail = segment_layer.get("subtype", pd.Series(index=segment_layer.index, dtype=str)) == "rail"
    is_highway = (~is_rail) & segment_layer.get("class", pd.Series(index=segment_layer.index, dtype=str)).isin(HIGHWAY_CLASSES)
    railway_layer = segment_layer[is_rail]
    highway_layer = segment_layer[is_highway]
    roads_layer = segment_layer[~is_rail & ~is_highway]
    print(f"[OVERTURE] buildings={len(building_layer)} roads={len(roads_layer)} "
          f"highway={len(highway_layer)} railway={len(railway_layer)}")

    for name, layer in (("buildings", building_layer), ("roads", roads_layer), ("highway", highway_layer), ("railway", railway_layer)):
        clipped = layer.clip(boundary_polygon) if not layer.empty else layer
        out_path = THIS_DIR / f"{name}_vector_project{PROJECT_ID}.geojson"
        gpd.GeoDataFrame(geometry=clipped.geometry if not clipped.empty else gpd.GeoSeries([], crs="EPSG:4326")).to_file(out_path, driver="GeoJSON")
        print(f"[SAVED] {out_path} ({len(clipped)} features)")

    api = GoogleMapsAPIHook(api_key=GOOGLE_MAPS_API_KEY)
    sat_img = api.fetch_tile(center_lat, center_lon, zoom, width=IMG_WIDTH, height=IMG_HEIGHT, maptype="satellite")
    map_img = api.fetch_tile(center_lat, center_lon, zoom, width=IMG_WIDTH, height=IMG_HEIGHT, maptype="roadmap")

    projector = WebMercatorProjector(center_lat, center_lon, zoom, img_logical_w=IMG_WIDTH, img_logical_h=IMG_HEIGHT, scale=SCALE)
    extractor = HybridZoneExtractor()
    water_mask, green_mask, road_mask = extractor.extract_map_features(map_img)
    # Kept only for the zone_type_full diagnostic column below (a faithful,
    # unmodified reproduction of open_map.py's own k=4-over-everything
    # method, for comparison) - NOT used for final_class/urban_tier any more.
    density_tiers_k4_unmasked = extractor.extract_building_density(sat_img)

    img_h, img_w = sat_img.shape[:2]

    # --- Pass 1: pixel footprint + Overture exclusion per tile, first ---
    # Water and Vegetation are already decided from Overture (water_ratio/
    # green_ratio, computed above). Those tiles' pixels must NOT be part of
    # the K-Means fit that decides Open/Suburban/Urban/Dense Urban - a
    # river's smooth surface and a sparse suburb both read as "low edge
    # density", and letting the water/vegetation pixels sit in the same
    # clustering pool skews where the boundaries fall. So: figure out which
    # tiles are already excluded FIRST, build a pixel mask from that, then
    # run k=4 only on what's left - k=4, not k=3, because a real "no
    # meaningful built texture here at all" tile (bare/open ground Overture
    # never tagged as a park) still needs somewhere to land; forcing
    # everything into exactly 3 buckets (Suburban/Urban/Dense Urban) leaves
    # no floor under Suburban and wrongly promotes genuinely open ground
    # into it. This is different from the original bug (k=4 fit on the
    # WHOLE image, water/vegetation pixels included, skewing the tiers) -
    # here it's still k=4, but fit only on real candidates.
    bboxes = []
    is_candidate = np.zeros(len(grid), dtype=bool)
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
        tile_water_hit = bool(np.any(water_mask[y1:y2, x1:x2] == 255))
        is_candidate[position] = _resolve_water_or_green(
            water_ratio[position], green_ratio[position], water_hit=tile_water_hit
        ) is None

    candidate_mask = np.zeros((img_h, img_w), dtype=bool)
    for position, (x1, y1, x2, y2) in enumerate(bboxes):
        if is_candidate[position]:
            candidate_mask[y1:y2, x1:x2] = True

    blur_kernel_px = _scaled_blur_kernel_px(zoom)
    print(f"[GSD] zoom={zoom} -> blur kernel {blur_kernel_px}px (Research/open_map.py's own 151px, "
          f"scaled from its reference zoom={REFERENCE_ZOOM} to keep the same real-world smoothing footprint)")
    density_map = _edge_density_map(sat_img, blur_kernel_px)
    tiers_k4 = _kmeans_tiers_on_mask(density_map, candidate_mask, k=4)
    print(f"[K4] candidate pixels for Open/Suburban/Urban/Dense Urban clustering: {int(candidate_mask.sum())} "
          f"/ {img_h * img_w} ({int(is_candidate.sum())}/{len(grid)} tiles excluded as Water/Vegetation before fitting)")

    TIER4_LABELS = {0: "Open", 1: "Suburban", 2: "Urban", 3: "Dense Urban"}

    # --- Pass 2: build the actual per-tile records ---
    records = []
    for position, row in enumerate(grid.itertuples()):
        x1, y1, x2, y2 = bboxes[position]
        water_hit = bool(np.any(water_mask[y1:y2, x1:x2] == 255))
        green_hit = bool(np.any(green_mask[y1:y2, x1:x2] == 255))
        road_hit = bool(np.any(road_mask[y1:y2, x1:x2] == 255))

        # zone_type_full: faithful open_map.py reproduction, unchanged, for comparison only.
        tier_chunk_k4 = density_tiers_k4_unmasked[y1:y2, x1:x2]
        votes = {
            "Water Body": int(np.sum(water_mask[y1:y2, x1:x2] == 255)),
            "Greenery": int(np.sum(green_mask[y1:y2, x1:x2] == 255)),
            "Road": int(np.sum(road_mask[y1:y2, x1:x2] == 255)),
            "High Density": int(np.sum((tier_chunk_k4 == 3) | (tier_chunk_k4 == 2))),
            "Medium/Low Density": int(np.sum(tier_chunk_k4 == 1)),
        }
        zone_type_full = max(votes, key=votes.get) if any(votes.values()) else "Empty Terrain"

        wr, gr = water_ratio[position], green_ratio[position]
        water_or_green = _resolve_water_or_green(wr, gr, water_hit=water_hit)
        if water_or_green is not None:
            final_class = water_or_green
            tier4 = None
        else:
            valid = tiers_k4[y1:y2, x1:x2]
            valid = valid[valid >= 0]
            tier4 = _mode_or(valid, default=-1) if valid.size else -1
            final_class = TIER4_LABELS.get(tier4, "Open")  # -1 only if a candidate tile had no valid pixel (rare)

        urban_tier = final_class if final_class in TIER4_LABELS.values() else "N/A (Water/Vegetation)"

        centroid = row.geometry.centroid
        records.append({
            "grid_id": row.grid_id,
            "lat": centroid.y,
            "lon": centroid.x,
            "tile_wkt": row.geometry.wkt,
            "production_clutter_class": row.production_clutter_class,
            "k4_tier": tier4,
            "urban_tier": urban_tier,
            "zone_type_full": zone_type_full,
            "water_ratio": round(float(wr), 3),
            "green_ratio": round(float(gr), 3),
            "final_class": final_class,
            "water_hit": water_hit,
            "green_hit": green_hit,
            "road_hit": road_hit,
        })

    result = pd.DataFrame(records)
    out_csv = THIS_DIR / f"clutter_comparison_{PROJECT_ID}.csv"
    result.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv} ({len(result)} rows)")

    print("\n=== final_class (Overture water/vegetation/rural-open + image-based density tiers) - THE 6-class result ===")
    print(result["final_class"].value_counts())
    print(f"({(result['final_class'].value_counts() / len(result) * 100).round(1).to_dict()} %)")
    print("\n=== urban_tier (image-based tiers only) distribution ===")
    print(result["urban_tier"].value_counts())
    print(f"\n=== zone_type_full (faithful open_map.py category logic) distribution ===")
    print(result["zone_type_full"].value_counts())
    print("\n=== production_clutter_class (already in the DB, for reference) distribution ===")
    print(result["production_clutter_class"].value_counts())
    print("\n=== crosstab: final_class vs production_clutter_class ===")
    print(pd.crosstab(result["final_class"], result["production_clutter_class"]))

    # --- Save the same fused visual overlay open_map.py itself produces ---
    boundary_pixels = np.array([
        projector.coord_to_pixel(min_lat, min_lon),
        projector.coord_to_pixel(min_lat, max_lon),
        projector.coord_to_pixel(max_lat, max_lon),
        projector.coord_to_pixel(max_lat, min_lon),
    ], dtype=np.int32).reshape((-1, 1, 2))
    clipping_mask = np.zeros(sat_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(clipping_mask, [boundary_pixels], 255)
    overlay_img = extractor.fuse_and_render(sat_img, water_mask, green_mask, road_mask, density_tiers_k4_unmasked, poly_mask=clipping_mask)
    out_jpg = THIS_DIR / f"Density_Zones_Map_project{PROJECT_ID}.jpg"
    cv2.imwrite(str(out_jpg), overlay_img)
    print(f"[SAVED] {out_jpg}")

    # Raw source image + masks too, so the dashboard can show what the
    # classifier actually saw, not just the final overlay.
    raw_jpg = THIS_DIR / f"satellite_raw_project{PROJECT_ID}.jpg"
    cv2.imwrite(str(raw_jpg), sat_img)
    print(f"[SAVED] {raw_jpg}")


if __name__ == "__main__":
    main()
