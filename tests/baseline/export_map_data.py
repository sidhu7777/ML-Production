"""
Exports project 210's classification result + underlying real layers
(buildings, roads, water, green, project boundary) as one compact JSON,
projected to local metres centred on the project, for a self-contained
static-map visualisation (no external tile server - browser CSP blocks
that anyway).

Read-only, no DB/bridge calls, no production code touched.

Usage:
    python tests/baseline/export_map_data.py --project-id 210 --region taiwan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.ops import transform


def rings(geom):
    """Return a list of exterior-ring coordinate lists (drops holes - fine for a visual)."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [list(geom.exterior.coords)]
    if geom.geom_type == "MultiPolygon":
        return [list(p.exterior.coords) for p in geom.geoms]
    if geom.geom_type == "LineString":
        return [list(geom.coords)]
    if geom.geom_type == "MultiLineString":
        return [list(ls.coords) for ls in geom.geoms]
    return []


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "map_export.json")
    args = parser.parse_args(argv)

    data_dir = args.data_root / f"project_{args.project_id}_{args.region}"

    tiles = gpd.read_file(data_dir / "clutter_tiles_final_v2.geojson")
    poly_gdf = gpd.read_file(data_dir / "project_polygon.geojson")
    poly = poly_gdf.geometry.iloc[0]
    poly_lonlat = transform(lambda x, y: (y, x), poly)  # stored swapped, fix to real lon/lat

    minx, miny, maxx, maxy = tiles.total_bounds
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2

    import math
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))

    def to_local_xy(lon, lat):
        return ((lon - center_lon) * m_per_deg_lon, (lat - center_lat) * m_per_deg_lat)

    def project_rings(geom):
        out = []
        for ring in rings(geom):
            out.append([[round(x, 1), round(y, 1)] for x, y in (to_local_xy(lon, lat) for lon, lat in ring)])
        return out

    # clutter tiles
    tile_features = []
    for _, row in tiles.iterrows():
        tile_features.append({
            "rings": project_rings(row.geometry),
            "cls": row["clutter_class"],
            "building_count": int(row.get("building_count", 0) or 0),
            "water_ratio": round(float(row.get("water_ratio", 0) or 0), 2),
            "surrounding_height_m": None if pd.isna(row.get("surrounding_height_m")) else round(float(row["surrounding_height_m"]), 1),
        })

    # buildings (real polygons, clipped to project polygon area only)
    building_df = pd.read_csv(data_dir / "building_df.csv", low_memory=False)
    geoms = building_df["region_wkt"].apply(shapely_wkt.loads)
    geoms_lonlat = geoms.apply(lambda g: transform(lambda x, y: (y, x), g) if g is not None else None)
    building_gdf = gpd.GeoDataFrame(building_df, geometry=geoms_lonlat, crs="EPSG:4326")
    building_gdf = building_gdf[building_gdf.geometry.notnull() & building_gdf.geometry.is_valid & ~building_gdf.geometry.is_empty]
    building_gdf = building_gdf[building_gdf.geometry.intersects(poly_lonlat)]
    building_rings = []
    for geom in building_gdf.geometry:
        building_rings.extend(project_rings(geom))

    # roads
    roads = gpd.read_file(data_dir / "roads_segment.geojson")
    roads = roads[roads.geometry.intersects(poly_lonlat)]
    road_lines = []
    for geom in roads.geometry:
        road_lines.extend(project_rings(geom))

    # water
    water = gpd.read_file(data_dir / "water.geojson")
    water = water[water.geometry.intersects(poly_lonlat)]
    water_rings = []
    for geom in water.geometry:
        water_rings.extend(project_rings(geom))

    # green (filtered to true green subtypes, same as classification)
    land_cover = gpd.read_file(data_dir / "land_cover.geojson")
    land_use = gpd.read_file(data_dir / "land_use.geojson")
    GREEN_LC_SUBTYPES = {"forest", "shrub", "grass"}
    GREEN_LU_SUBTYPES = {"park", "recreation", "horticulture", "agriculture"}
    GREEN_LU_CLASS = {"park", "garden", "grass", "recreation_ground", "village_green", "pitch", "nature_reserve"}
    green_lc = land_cover[land_cover["subtype"].isin(GREEN_LC_SUBTYPES)]
    green_lu = land_use[land_use["subtype"].isin(GREEN_LU_SUBTYPES) | land_use["class"].isin(GREEN_LU_CLASS)]
    green = pd.concat([green_lc, green_lu], ignore_index=True)
    green = gpd.GeoDataFrame(green, crs="EPSG:4326")
    green = green[green.geometry.intersects(poly_lonlat)]
    green_rings = []
    for geom in green.geometry:
        green_rings.extend(project_rings(geom))

    boundary_rings = project_rings(poly_lonlat)

    bundle = {
        "tiles": tile_features,
        "buildings": building_rings,
        "roads": road_lines,
        "water": water_rings,
        "green": green_rings,
        "boundary": boundary_rings,
        "meta": {
            "project_id": args.project_id,
            "region": args.region,
            "n_tiles": len(tiles),
            "n_buildings": len(building_gdf),
            "n_roads": len(roads),
            "n_water": len(water),
            "class_counts": tiles["clutter_class"].value_counts().to_dict(),
        },
    }

    args.out.write_text(json.dumps(bundle), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    print(f"buildings={len(building_gdf)} roads={len(roads)} water={len(water)} green={len(green)}")


if __name__ == "__main__":
    main()
