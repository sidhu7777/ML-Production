"""Project-scoped geospatial input cache for the offset baseline.

This module owns *input* provenance, not prediction output. It implements the
same Overture + GHS-OBAT Phase-27 clutter classification in production and
caches the resulting project grid classes for reuse.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from tools.lte_prediction.geo_correction_pipeline import _choose_utm_crs, building_df_to_gdf

PHASE27_CLUTTER_SOURCE = "Overture Maps + GHS-OBAT"
PHASE27_CLASSIFIER_VERSION = "phase27-v2"
OVERTURE_CONNECT_TIMEOUT_S = 20
OVERTURE_REQUEST_TIMEOUT_S = 90
GREEN_LC_SUBTYPES = {"forest", "shrub", "grass"}
GREEN_LU_SUBTYPES = {"park", "recreation", "horticulture", "agriculture"}
GREEN_LU_CLASS = {"park", "garden", "grass", "recreation_ground", "village_green", "pitch", "nature_reserve"}


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


def _clip_area_ratio(grid: gpd.GeoDataFrame, layer: gpd.GeoDataFrame, name: str) -> pd.Series:
    grid_utm = grid.to_crs(_choose_utm_crs(grid))
    if layer.empty:
        return pd.Series(0.0, index=grid["grid_id"].astype(str))
    polygons = layer[layer.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(grid_utm.crs)
    if polygons.empty:
        return pd.Series(0.0, index=grid["grid_id"].astype(str))
    index = polygons.sindex
    values = []
    for position, row in enumerate(grid_utm.itertuples()):
        hits = list(index.query(row.geometry, predicate="intersects"))
        area = sum(polygons.geometry.iloc[i].intersection(row.geometry).area for i in hits)
        values.append(min(1.0, area / max(row.geometry.area, 1.0)))
        if (position + 1) % 2000 == 0:
            print(f"[LTE_OFFSET][PHASE27_OVERLAY] layer={name} tiles={position + 1}/{len(grid_utm)}", flush=True)
    return pd.Series(values, index=grid["grid_id"].astype(str))


def _building_context(grid: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> tuple[pd.Series, pd.Series]:
    grid_utm = grid.to_crs(_choose_utm_crs(grid))
    if buildings.empty:
        blank = pd.Series(0.0, index=grid["grid_id"].astype(str))
        return blank, blank
    bld = buildings.to_crs(grid_utm.crs).copy()
    bld["building_row_id"] = np.arange(len(bld))
    index = bld.sindex
    counts, ratios = [], []
    for position, row in enumerate(grid_utm.itertuples()):
        hits = list(index.query(row.geometry, predicate="intersects"))
        area = sum(bld.geometry.iloc[i].intersection(row.geometry).area for i in hits)
        counts.append(len(hits))
        ratios.append(min(1.0, area / max(row.geometry.area, 1.0)))
        if (position + 1) % 2000 == 0:
            print(f"[LTE_OFFSET][PHASE27_OVERLAY] layer=buildings tiles={position + 1}/{len(grid_utm)}", flush=True)
    keys = grid["grid_id"].astype(str)
    return pd.Series(counts, index=keys), pd.Series(ratios, index=keys)


def _road_length(grid: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.Series:
    grid_utm = grid.to_crs(_choose_utm_crs(grid))
    lines = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].to_crs(grid_utm.crs) if not roads.empty else roads
    if lines.empty:
        return pd.Series(0.0, index=grid["grid_id"].astype(str))
    index = lines.sindex
    values = []
    for position, row in enumerate(grid_utm.itertuples()):
        hits = list(index.query(row.geometry, predicate="intersects"))
        values.append(sum(lines.geometry.iloc[i].intersection(row.geometry).length for i in hits))
        if (position + 1) % 2000 == 0:
            print(f"[LTE_OFFSET][PHASE27_OVERLAY] layer=roads tiles={position + 1}/{len(grid_utm)}", flush=True)
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
    work["height_m"] = pd.to_numeric(work["height_m"], errors="coerce").fillna(12.0).clip(3.0, 120.0)
    return work, source


def _surrounding_height(grid: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> pd.Series:
    if buildings.empty:
        return pd.Series(np.nan, index=grid["grid_id"].astype(str))
    crs = _choose_utm_crs(grid)
    bld = buildings.to_crs(crs)
    tiles = grid.to_crs(crs)
    from scipy.spatial import cKDTree
    bld_points = np.asarray([[geom.centroid.x, geom.centroid.y] for geom in bld.geometry])
    tile_points = np.asarray([[geom.centroid.x, geom.centroid.y] for geom in tiles.geometry])
    nearby = cKDTree(bld_points).query_ball_point(tile_points, 100.0)
    heights = bld["height_m"].to_numpy(float)
    return pd.Series([float(heights[index].mean()) if index else np.nan for index in nearby], index=grid["grid_id"].astype(str))


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


def _phase27_dataset(db_engine, project_id: int, grid: gpd.GeoDataFrame) -> int:
    layout = "|".join(sorted(
        f"{grid_id}:{geometry.wkb_hex}" for grid_id, geometry in zip(grid["grid_id"].astype(str), grid.geometry)
    ))
    boundary_hash = hashlib.sha256(layout.encode("utf-8")).hexdigest()
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
        conn.execute(text("""UPDATE tbl_project_geo_dataset SET is_active = 0
            WHERE project_id = :project_id AND dataset_type = 'phase27_clutter' AND is_active = 1"""), {"project_id": project_id})
        result = conn.execute(text("""
            INSERT INTO tbl_project_geo_dataset
                (project_id, dataset_type, source_name, source_version, boundary_hash,
                 resolution_m, metadata_json, is_active)
            VALUES (:project_id, 'phase27_clutter', :source_name, :source_version,
                    :boundary_hash, 25.0,
                    JSON_OBJECT('classifier', 'water->building-height->road->green->open'), 1)
        """), {"project_id": project_id, "source_name": PHASE27_CLUTTER_SOURCE,
               "source_version": PHASE27_CLASSIFIER_VERSION, "boundary_hash": boundary_hash})
        return int(result.lastrowid)
    return _with_db_retry(db_engine, action, "phase27_dataset")


def _save_phase27_tiles(db_engine, project_id: int, dataset_id: int, grid: gpd.GeoDataFrame, values: pd.DataFrame) -> None:
    lookup = values.set_index("grid_id")
    rows = [{
        "project_id": project_id, "dataset_id": dataset_id, "grid_id": str(row.grid_id),
        "geometry_wkt": row.geometry.wkt, "clutter_class": str(lookup.at[str(row.grid_id), "clutter_class"]),
        "land_cover_class": str(lookup.at[str(row.grid_id), "land_cover_class"]),
    } for row in grid.itertuples()]
    statement = text("""
        INSERT INTO tbl_project_clutter_tile
          (project_id, geo_dataset_id, grid_id, geometry_wkt, clutter_class, land_cover_class, resolution_m, is_active)
        VALUES (:project_id, :dataset_id, :grid_id, :geometry_wkt, :clutter_class, :land_cover_class, 25.0, 1)
        ON DUPLICATE KEY UPDATE geometry_wkt=VALUES(geometry_wkt), clutter_class=VALUES(clutter_class),
          land_cover_class=VALUES(land_cover_class), is_active=1
    """)
    # Keep transactions short.  A project cache has thousands of grid tiles and
    # one long upsert transaction can be blocked by unrelated DB activity.
    for start in range(0, len(rows), 250):
        batch = rows[start:start + 250]

        def action(batch=batch):
            with db_engine.begin() as conn:
                conn.execute(statement, batch)

        _with_db_retry(db_engine, action, f"save_clutter_tiles_{start // 250 + 1}")


def _save_building_profiles(db_engine, project_id: int, dataset_id: int, buildings: gpd.GeoDataFrame, method: str) -> None:
    if buildings.empty:
        return
    ids = buildings.get("id", pd.Series(buildings.index, index=buildings.index)).astype(str)
    rows = [
        {"project_id": project_id, "building_geometry_id": identity, "dataset_id": dataset_id,
         "height_m": float(height), "method": method,
         "height_source": "GHS-OBAT" if method.startswith("GHS-OBAT") else "project_profile"}
        for identity, height in zip(ids, buildings["height_m"])
    ]
    statement = text("""
        INSERT INTO tbl_project_building_profile
          (project_id, building_geometry_id, geo_dataset_id, height_m, height_method, height_source, confidence, is_active)
        VALUES (:project_id, :building_geometry_id, :dataset_id, :height_m, :method, :height_source, 0.70, 1)
        ON DUPLICATE KEY UPDATE height_m=VALUES(height_m), height_method=VALUES(height_method),
          height_source=VALUES(height_source), confidence=VALUES(confidence), is_active=1
    """)
    def action():
        with db_engine.begin() as conn:
            for start in range(0, len(rows), 1000):
                conn.execute(statement, rows[start:start + 1000])
    _with_db_retry(db_engine, action, "save_building_profiles")


def _attach_resolved_heights(building_df: pd.DataFrame, buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return raw DB rows augmented with the same profile used for clutter."""
    out = building_df.copy()
    if buildings.empty or out.empty or len(buildings) != len(out):
        return out
    out["building_height_m"] = pd.to_numeric(buildings["height_m"], errors="coerce").to_numpy()
    return out


def load_or_build_phase27_clutter(grid_df: pd.DataFrame, building_df: pd.DataFrame, project_id: int, db_engine, obat_csv_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Production implementation of Phase 27's vector clutter classifier."""
    if db_engine is None:
        raise RuntimeError("Project database engine is required for the Phase-27 geospatial cache")
    grid = _grid_geometry(grid_df)
    dataset_id = _phase27_dataset(db_engine, int(project_id), grid)
    buildings = _align_buildings(building_df_to_gdf(building_df), grid)
    buildings, height_source = _impute_heights(buildings, obat_csv_path)
    _save_building_profiles(db_engine, int(project_id), dataset_id, buildings, height_source)
    resolved_building_df = _attach_resolved_heights(building_df, buildings)
    cached = _read_cached(db_engine, int(project_id), dataset_id, grid["grid_id"])
    if len(cached) == len(grid):
        return cached, resolved_building_df, {
            "enabled": True, "source": PHASE27_CLUTTER_SOURCE, "dataset_id": dataset_id,
            "cache": "hit", "tiles": len(cached), "height_source": height_source,
        }

    context = _fetch_overture_context(grid)
    building_count, building_ratio = _building_context(grid, buildings)
    water_ratio = _clip_area_ratio(grid, context["water"], "water")
    road_length = _road_length(grid, context["segment"])
    land_cover = context["land_cover"]
    land_use = context["land_use"]
    green_lc = land_cover[land_cover.get("subtype", pd.Series(index=land_cover.index, dtype=str)).isin(GREEN_LC_SUBTYPES)]
    green_lu = land_use[
        land_use.get("subtype", pd.Series(index=land_use.index, dtype=str)).isin(GREEN_LU_SUBTYPES)
        | land_use.get("class", pd.Series(index=land_use.index, dtype=str)).isin(GREEN_LU_CLASS)
    ]
    green = gpd.GeoDataFrame(pd.concat([green_lc, green_lu], ignore_index=True), crs="EPSG:4326")
    green_ratio = _clip_area_ratio(grid, green, "green")
    surrounding = _surrounding_height(grid, buildings)
    project_mean = float(buildings["height_m"].mean()) if not buildings.empty else 12.0
    records = []
    for grid_id in grid["grid_id"].astype(str):
        h = surrounding.get(grid_id)
        h = project_mean if pd.isna(h) else float(h)
        if water_ratio.get(grid_id, 0.0) >= 0.5:
            label, rule = "Water", "water_ratio>=0.5"
        elif building_count.get(grid_id, 0.0) > 0:
            label, rule = ("Dense Urban" if h > 15.0 else "Urban" if h > 6.0 else "Suburban"), "building_height_tier"
        elif road_length.get(grid_id, 0.0) > 0:
            label, rule = ("Dense Urban" if h > 15.0 else "Urban" if h > 6.0 else "Suburban"), "road_surrounding_height_tier"
        elif green_ratio.get(grid_id, 0.0) >= 0.30:
            label, rule = "Vegetation", "green_ratio>=0.30"
        else:
            label, rule = "Rural/Open", "no_context_feature"
        records.append({"grid_id": grid_id, "clutter_class": label, "land_cover_class": rule})
    result = pd.DataFrame(records)
    _save_phase27_tiles(db_engine, int(project_id), dataset_id, grid, result)
    return result, resolved_building_df, {
        "enabled": True, "source": PHASE27_CLUTTER_SOURCE, "dataset_id": dataset_id, "cache": "miss",
        "tiles": len(result), "height_source": height_source,
        "overture_features": {name: int(len(layer)) for name, layer in context.items()},
        "classes": {str(k): int(v) for k, v in result["clutter_class"].value_counts().items()},
    }
