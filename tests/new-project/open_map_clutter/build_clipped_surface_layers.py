"""
Build visual surface layers for the open_map clutter dashboard.

This does not call the DB, Overture, or Google. It only uses the saved
clutter_comparison_<project>.csv plus the saved vector GeoJSONs in this
folder. Overture water stays polygon-precise. For small water features that
Overture missed, the saved Google roadmap `water_hit` tile signal is added as
supplemental water, then removed from vegetation.

Run:
    venv\\Scripts\\python.exe tests\\new-project\\open_map_clutter\\build_clipped_surface_layers.py
"""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.ops import transform

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ID = int(os.environ.get("CLUTTER_PROJECT_ID", 210))

CSV_PATH = THIS_DIR / f"clutter_comparison_{PROJECT_ID}.csv"
WATER_GEOJSON = THIS_DIR / f"water_vector_project{PROJECT_ID}.geojson"
VEGETATION_GEOJSON = THIS_DIR / f"vegetation_vector_project{PROJECT_ID}.geojson"
BUILDINGS_GEOJSON = THIS_DIR / f"buildings_vector_project{PROJECT_ID}.geojson"

WATER_CLIPPED_GEOJSON = THIS_DIR / f"water_clipped_project{PROJECT_ID}.geojson"
VEGETATION_CLIPPED_GEOJSON = THIS_DIR / f"vegetation_clipped_project{PROJECT_ID}.geojson"
BUILDINGS_CLIPPED_GEOJSON = THIS_DIR / f"buildings_clipped_project{PROJECT_ID}.geojson"
WATER_SUPPLEMENTAL_GEOJSON = THIS_DIR / f"water_supplemental_project{PROJECT_ID}.geojson"
PROJECT_POLYGON_GEOJSON = THIS_DIR.parents[1] / "baseline" / "data" / f"project_{PROJECT_ID}_taiwan" / "project_polygon.geojson"


def _read_polygon_layer(path: Path, crs, project_extent):
    if not path.exists():
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    layer = gpd.read_file(path)
    if layer.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    layer = layer.to_crs(crs)
    layer = layer[layer.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if layer.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    layer["geometry"] = layer.geometry.buffer(0).intersection(project_extent)
    return layer[layer.geometry.apply(lambda geom: geom is not None and not geom.is_empty)]


def _overlay_grid(grid_utm: gpd.GeoDataFrame, layer: gpd.GeoDataFrame, out_path: Path) -> int:
    if layer.empty:
        out = gpd.GeoDataFrame({"grid_id": [], "geometry": []}, geometry="geometry", crs="EPSG:4326")
    else:
        out = gpd.overlay(grid_utm[["grid_id", "geometry"]], layer[["geometry"]], how="intersection", keep_geom_type=True)
        out = out[out.geometry.apply(lambda geom: geom is not None and not geom.is_empty)].to_crs("EPSG:4326")
    out.to_file(out_path, driver="GeoJSON")
    return len(out)


def _load_project_polygon():
    if not PROJECT_POLYGON_GEOJSON.exists():
        return None
    polygon_gdf = gpd.read_file(PROJECT_POLYGON_GEOJSON)
    if polygon_gdf.empty:
        return None
    return transform(lambda x, y: (y, x), polygon_gdf.geometry.iloc[0])


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found. Run run_open_map_clutter.py first.")

    df = pd.read_csv(CSV_PATH, usecols=["grid_id", "tile_wkt", "water_hit", "water_ratio"])
    grid = gpd.GeoDataFrame(
        df[["grid_id"]],
        geometry=df["tile_wkt"].apply(shapely_wkt.loads),
        crs="EPSG:4326",
    )
    project_polygon = _load_project_polygon()
    if project_polygon is not None:
        inside_mask = grid.geometry.intersects(project_polygon)
        df = df[inside_mask.to_numpy()].reset_index(drop=True)
        grid = grid[inside_mask].reset_index(drop=True)

    utm_crs = grid.estimate_utm_crs()
    grid_utm = grid.to_crs(utm_crs)
    project_extent = (
        gpd.GeoSeries([project_polygon], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
        if project_polygon is not None
        else grid_utm.geometry.union_all()
    )

    water = _read_polygon_layer(WATER_GEOJSON, utm_crs, project_extent)
    water_union = water.geometry.union_all() if not water.empty else None

    # Overture misses some small ponds/channels inside parks. The runner's
    # Google-roadmap mask already captured those cells as water_hit=True.
    # Add only cells where Overture did not already cover meaningful water.
    supplemental_tiles = grid_utm[df["water_hit"].astype(bool).to_numpy() & (df["water_ratio"].fillna(0).to_numpy() < 0.25)].copy()
    if water_union is not None and not supplemental_tiles.empty:
        supplemental_tiles["geometry"] = supplemental_tiles.geometry.difference(water_union)
        supplemental_tiles = supplemental_tiles[supplemental_tiles.geometry.apply(lambda geom: geom is not None and not geom.is_empty)]
    supplemental_tiles.to_crs("EPSG:4326").to_file(WATER_SUPPLEMENTAL_GEOJSON, driver="GeoJSON")

    if not supplemental_tiles.empty:
        supplemental_union = supplemental_tiles.geometry.union_all()
        if water.empty:
            water = gpd.GeoDataFrame({"geometry": [supplemental_union]}, geometry="geometry", crs=utm_crs)
        else:
            combined_water = water.geometry.union_all().union(supplemental_union)
            water = gpd.GeoDataFrame({"geometry": [combined_water]}, geometry="geometry", crs=utm_crs)
        water_union = water.geometry.union_all()

    vegetation = _read_polygon_layer(VEGETATION_GEOJSON, utm_crs, project_extent)
    if water_union is not None and not vegetation.empty:
        vegetation["geometry"] = vegetation.geometry.difference(water_union)
        vegetation = vegetation[vegetation.geometry.apply(lambda geom: geom is not None and not geom.is_empty)]

    buildings = _read_polygon_layer(BUILDINGS_GEOJSON, utm_crs, project_extent)

    water_count = _overlay_grid(grid_utm, water, WATER_CLIPPED_GEOJSON)
    vegetation_count = _overlay_grid(grid_utm, vegetation, VEGETATION_CLIPPED_GEOJSON)
    building_count = _overlay_grid(grid_utm, buildings, BUILDINGS_CLIPPED_GEOJSON)

    print(f"[SAVED] {WATER_SUPPLEMENTAL_GEOJSON} ({len(supplemental_tiles)} supplemental water cells)")
    print(f"[SAVED] {WATER_CLIPPED_GEOJSON} ({water_count} clipped pieces)")
    print(f"[SAVED] {VEGETATION_CLIPPED_GEOJSON} ({vegetation_count} clipped pieces)")
    print(f"[SAVED] {BUILDINGS_CLIPPED_GEOJSON} ({building_count} clipped pieces)")


if __name__ == "__main__":
    main()
