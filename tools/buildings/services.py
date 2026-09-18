import concurrent.futures
import os
import json
import logging
import traceback
import zipfile
from pathlib import Path

import osmnx as ox
import geopandas as gpd
import pandas as pd
import requests
import sqlalchemy as db
from sqlalchemy.exc import OperationalError
from flask import current_app
from dotenv import load_dotenv

# -------------------------------------
# GLOBAL ENGINE (MULTI-REGION)
# -------------------------------------
load_dotenv()

engine_dict = {
    "india": db.create_engine(
        os.getenv("DATABASE_URL"),
        pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=3600
    ) if os.getenv("DATABASE_URL") else None,

    "taiwan": db.create_engine(
        os.getenv("DATABASE_URL_Taiwan"),
        pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=3600
    ) if os.getenv("DATABASE_URL_Taiwan") else None
}

def get_regional_engine(region_name):
    """Safely retrieves the requested region engine, falling back to India."""
    region = str(region_name).lower()
    current_engine = engine_dict.get(region, engine_dict.get("india"))

    if current_engine is None:
        raise RuntimeError(f"Database engine for region '{region}' is not initialized/configured.")

    return current_engine


# -------------------------------------
# GHS-OBAT (real building height) - downloaded once per region, cached
# outside python-runtime so it survives every installer/version upgrade
# (the NSIS installer wipes %APPDATA%\S-Tracer\python-runtime on every
# bump - see _dem_cache_dir in lte_prediction_offset/services.py, same
# reasoning, same pattern, reused here on purpose).
#
# GHS-OBAT (JRC, R2024A) is distributed as one file per COUNTRY, not
# queryable by bounding box remotely - see
# https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_OBAT_GLOBE_R2024A/
# Taiwan's file is 28MB - small enough to download once and cache locally.
# India's equivalent file is 11-23GB as a single country-wide archive (not
# split by state) - not practical to download/cache on a desktop app, so
# India is intentionally left out of GHS_OBAT_SOURCES below until a
# pre-filtered, properly-hosted India extract exists. Buildings for India
# keep using the existing OSM path with no real height (same as before).
# -------------------------------------
GHS_OBAT_SOURCES = {
    "taiwan": {
        "url": (
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_OBAT_GLOBE_R2024A/GHS_OBAT_GPKG_GLOBE_R2024A/V1-0/"
            "GHS_OBAT_GPKG_TWN_E2020_R2024A_V1_0.zip"
        ),
        "gpkg_name": "GHS_OBAT_GPKG_TWN_E2020_R2024A_V1_0.gpkg",
    },
}


def _ghs_obat_cache_dir() -> Path:
    base = os.getenv("APPDATA") or os.getenv("XDG_CACHE_HOME") or str(Path.home())
    path = Path(base) / "S-Tracer" / "ghs_obat"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_ghs_obat_file(region: str):
    """Local path to this region's cached GHS-OBAT GeoPackage, downloading
    it once (per region, ever - not per project) if not already cached.
    Returns None when no GHS-OBAT source is configured for this region."""
    region_key = str(region or "").strip().lower()
    source = GHS_OBAT_SOURCES.get(region_key)
    if source is None:
        return None

    cache_dir = _ghs_obat_cache_dir()
    gpkg_path = cache_dir / source["gpkg_name"]
    if gpkg_path.is_file():
        return gpkg_path

    zip_path = cache_dir / f"{region_key}_ghs_obat.zip"
    print(f"[BUILDINGS] GHS-OBAT not cached yet for {region_key} - downloading once "
          f"(subsequent projects in this region reuse this file)...", flush=True)
    try:
        with requests.get(source["url"], stream=True, timeout=(20, 300)) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(source["gpkg_name"], cache_dir)
        print(f"[BUILDINGS] GHS-OBAT cached at {gpkg_path}", flush=True)
        return gpkg_path
    except Exception as exc:
        print(f"[BUILDINGS] GHS-OBAT download failed for {region_key}: {exc}", flush=True)
        return None
    finally:
        zip_path.unlink(missing_ok=True)


def _match_ghs_obat_heights(buildings: gpd.GeoDataFrame, region: str) -> gpd.GeoDataFrame:
    """Real per-building height via a spatial join against the cached
    GHS-OBAT points (building centroids) for this region. Buildings with
    no match stay height_m=NaN - callers impute/fallback downstream, this
    function never invents a height."""
    buildings = buildings.copy()
    buildings["height_m"] = pd.NA

    gpkg_path = ensure_ghs_obat_file(region)
    if gpkg_path is None or buildings.empty:
        return buildings

    minx, miny, maxx, maxy = buildings.total_bounds
    pad = 0.01  # ~1km, generous margin around the project's own buildings
    try:
        obat = gpd.read_file(gpkg_path, bbox=(minx - pad, miny - pad, maxx + pad, maxy + pad))
    except Exception as exc:
        print(f"[BUILDINGS] GHS-OBAT read failed: {exc}", flush=True)
        return buildings
    if obat.empty:
        return buildings

    buildings["_row_id"] = range(len(buildings))
    joined = gpd.sjoin(
        obat[["height", "geometry"]], buildings[["_row_id", "geometry"]],
        how="inner", predicate="within",
    )
    if not joined.empty:
        height_by_row = joined.groupby("_row_id")["height"].mean()
        buildings["height_m"] = buildings["_row_id"].map(height_by_row)
    buildings = buildings.drop(columns=["_row_id"])

    matched = int(buildings["height_m"].notna().sum())
    print(f"[BUILDINGS] GHS-OBAT height matched: {matched}/{len(buildings)} buildings", flush=True)
    return buildings


# -------------------------------------
# BUILDING EXTRACTION + SAVE CLASS
# -------------------------------------
class BuildingService:

    def fetch_buildings_overture(self, polygon):
        """Fetch real building polygons from Overture Maps - primary source
        (more complete coverage than OSM). Returns (None, 0) on any failure
        or empty result so callers can fall back to OSM."""
        try:
            import overturemaps.core as overture
        except ImportError:
            print("[BUILDINGS] overturemaps package not available, skipping Overture", flush=True)
            return None, 0

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        try:
            layer = overture.geodataframe(
                "building",
                bbox=polygon.bounds,
                connect_timeout=20,
                request_timeout=90,
            )
        except Exception as exc:
            print(f"[BUILDINGS] Overture building fetch failed: {exc}", flush=True)
            return None, 0

        if layer is None or layer.empty:
            return None, 0
        if layer.crs is None:
            layer = layer.set_crs("EPSG:4326")

        try:
            clipped = gpd.clip(layer, polygon)
        except Exception as exc:
            print(f"[BUILDINGS] Overture clip failed: {exc}", flush=True)
            return None, 0

        clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty]
        clipped = clipped[clipped.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if clipped.empty:
            return None, 0

        return clipped, len(clipped)

    def fetch_buildings(self, polygon):
        """Fetch buildings from OSM - fallback only, used when Overture
        cannot return coverage for this polygon."""
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        ox.settings.timeout = 180
        ox.settings.use_cache = True

        try:
            buildings = ox.features_from_polygon(
                polygon,
                tags={"building": True, "residential": True}
            )

            buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]

            if buildings.empty:
                return None, 0

            return buildings, len(buildings)

        except Exception as e:
            if "No matching features" in str(e):
                return None, 0
            raise e

    # -------------------------------------
    # SAVE TO DATABASE
    # -------------------------------------
    # source_name records real provenance (overture_building / osm_building)
    # on every row, reusing the column tbl_savepolygon already carries for
    # exactly this purpose.
    def save_buildings_to_db(self, buildings, area_name, project_id, swap_output=False, region="india", source_name="osm_building"):

        # ✅ Fetch regional engine dynamically
        current_engine = get_regional_engine(region)

        # EXPLODE MULTIPOLYGONS
        buildings_exp = buildings.explode(index_parts=True, ignore_index=True)

        # --- FIX: Calculate Area FIRST ---
        buildings_exp["calc_area"] = buildings_exp.to_crs(epsg=3857).geometry.area

        # Safety net: Convert any mathematically impossible areas to 0
        buildings_exp["calc_area"] = buildings_exp["calc_area"].fillna(0)

        if "height_m" not in buildings_exp.columns:
            buildings_exp["height_m"] = pd.NA

        # swap_output intentionally does nothing to the geometry here. It used
        # to swap the already-correct lon/lat geometry back to lat/lon before
        # saving, on the assumption the DB wanted the ORIGINAL input's axis
        # order - but the INSERT below always declares
        # ST_GeomFromText(..., 'axis-order=long-lat'), so the column always
        # expects lon/lat regardless of how the input WKT was originally
        # given. Real bug, confirmed against project 263: for a project whose
        # input WKT needed swapping (swap_output=True), this used to swap the
        # geometry to lat/lon and then insert it declaring long/lat order,
        # producing an out-of-range latitude (MySQL error 3617) and silently
        # failing every building save whose input happened to be lat/lon.
        # swap_output is kept as a parameter only because callers still pass
        # it for the response's input_format_detected field.

        # Convert geometry to WKT
        buildings_exp["wkt_4326"] = buildings_exp.geometry.to_wkt()

        # NaN must become a real Python None here, not upstream on the
        # Series - pandas silently coerces None back to NaN when a column
        # is float64 dtype (confirmed: the previous .where(notna(), None)
        # on the Series never actually worked because of this), so pymysql
        # was handed a real float('nan') and rejected it outright
        # ("nan can not be used with MySQL"). Checking with pd.isna() per
        # row, right before the SQL parameter tuple is built, sidesteps
        # the dtype coercion entirely.
        values_list = [
            (area_name, row.wkt_4326, project_id, row.calc_area,
             None if pd.isna(row.height_m) else float(row.height_m), source_name)
            for row in buildings_exp.itertuples()
        ]

        # ✅ Grab the raw connection specifically from the regional engine
        raw = current_engine.raw_connection()
        cur = raw.cursor()

        try:
            cur.execute("SET autocommit=0")
            cur.execute("SET unique_checks=0")
            cur.execute("SET foreign_key_checks=0")

            batch_size = 1000
            total = 0

            for i in range(0, len(values_list), batch_size):
                batch = values_list[i : i + batch_size]

                placeholders = "(%s, ST_GeomFromText(%s, 4326, 'axis-order=long-lat'), %s, %s, %s, %s)"
                values_str = ", ".join([placeholders] * len(batch))

                insert_sql = f"""
                    INSERT INTO tbl_savepolygon (name, region, project_id, area, height_m, source_name)
                    VALUES {values_str}
                """

                flat_values = [item for sub in batch for item in sub]
                cur.execute(insert_sql, flat_values)
                total += len(batch)

            cur.execute("SET unique_checks=1")
            cur.execute("SET foreign_key_checks=1")
            raw.commit()

        finally:
            cur.close()
            raw.close()

        return total

    # -------------------------------------
    # MAIN EXTRACT + SAVE METHOD (UPDATED)
    # -------------------------------------
    # Overture is the primary building source (OSM is the fallback only,
    # used when Overture genuinely cannot return coverage for this
    # polygon). Height is deliberately NOT resolved here any more -
    # GHS-OBAT/OSM height matching moved to baseline run time (the only
    # place it's actually consumed - indoor penetration loss), so project
    # creation only ever saves geometry, never blocks on a height match.
    # height_m stays NULL for every building at this stage, real source or
    # not; a later baseline run is what fills it in, in place, once.
    def process_buildings(self, polygon, name, project_id, swap_output=False, region="india"):
        """
        Extract buildings and save them to DB
        """
        buildings, count = self.fetch_buildings_overture(polygon)
        source_name = "overture_building"

        if buildings is None or buildings.empty:
            print("[BUILDINGS] Overture returned no coverage for this polygon, falling back to OSM", flush=True)
            buildings, count = self.fetch_buildings(polygon)
            source_name = "osm_building"

        if buildings is None or buildings.empty:
            return None, 0, 0

        buildings["height_m"] = pd.NA

        # ✅ Pass region down to the save function
        saved_count = self.save_buildings_to_db(
            buildings, name, project_id, swap_output=swap_output, region=region, source_name=source_name
        )

        # geometry + height_m only, not the raw Overture attribute columns -
        # Overture's own fields (e.g. 'sources') can hold nested
        # numpy/array values that plain json.dumps can't serialize
        # (confirmed: this crashed AFTER a real, successful DB save,
        # turning a completed save into a reported failure for no reason -
        # the geojson response doesn't need those raw fields anyway).
        geojson = json.loads(buildings[["geometry", "height_m"]].to_json())

        return geojson, count, saved_count

    # -------------------------------------
    # UNIFIED PROJECT-CREATION GEO SETUP
    # -------------------------------------
    # Buildings and clutter classification both pull from Overture and
    # both belong at project creation - previously they were two separate,
    # disconnected pipelines (this file, and tools/lte_prediction_offset's
    # Phase-27 classifier, triggered separately at different times). This
    # is the single entry point: one project creation step, buildings and
    # clutter fetched CONCURRENTLY (not one after another - they're
    # independent network calls, and clutter classification does not read
    # building geometry or height at all any more, so there's no real
    # dependency forcing them to run in sequence), and height is not
    # touched at all - that's resolved later, once, at baseline time.
    def process_project_geo_setup(self, polygon, name, project_id, swap_output=False, region="india"):
        def _do_buildings():
            return self.process_buildings(polygon, name, project_id, swap_output=swap_output, region=region)

        def _do_clutter():
            from tools.lte_prediction.geo_correction_pipeline import create_analysis_grid
            from tools.lte_prediction_offset.geo_inputs import load_or_build_phase27_clutter

            db_engine = get_regional_engine(region)
            mask_gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")
            grid_gdf = create_analysis_grid(mask_gdf, cell_size_m=25.0)
            bounds = grid_gdf.geometry.bounds
            centroids = grid_gdf.geometry.centroid
            grid_df = pd.DataFrame({
                "grid_id": grid_gdf["grid_id"].astype(str),
                "center_lat": centroids.y, "center_lon": centroids.x,
                "min_lat": bounds["miny"], "max_lat": bounds["maxy"],
                "min_lon": bounds["minx"], "max_lon": bounds["maxx"],
            })
            # Clutter classification never reads building geometry/height
            # (see geo_inputs.py's classify loop) - building_df is no longer
            # even part of this function's signature. Height resolution is
            # entirely separate now (resolve_building_heights), called at
            # baseline time, not here.
            return load_or_build_phase27_clutter(grid_df, project_id, db_engine)

        # Check both futures independently, not one after the other - a
        # naive buildings_future.result() followed by clutter_future.result()
        # means a real exception in buildings hides whatever actually
        # happened to clutter (confirmed: this exact ordering once made a
        # genuine clutter-side failure invisible, reported only as a
        # buildings error, while the clutter thread's own partial work -
        # a dataset row with zero tiles - was left stranded with no
        # explanation). Both errors, if any, are surfaced together.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            buildings_future = pool.submit(_do_buildings)
            clutter_future = pool.submit(_do_clutter)
            buildings_result = buildings_error = None
            clutter_result = clutter_error = None
            try:
                buildings_result = buildings_future.result()
            except Exception as exc:
                buildings_error = exc
            try:
                clutter_result = clutter_future.result()
            except Exception as exc:
                clutter_error = exc

        if buildings_error or clutter_error:
            raise RuntimeError(
                f"process_project_geo_setup failed - buildings_error={buildings_error!r} clutter_error={clutter_error!r}"
            )
        return buildings_result, clutter_result
