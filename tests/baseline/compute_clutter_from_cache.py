"""
Computes clutter_class distribution for a cached project using its 100m
tile grid, reading ONLY local cache files under tests/baseline/data/ - no
DB/bridge calls.

Fixes a real bug found in production's attach_building_features(): it
attributes each building's FULL polygon area to whichever single tile
contains the building's centroid, with no clipping to that tile's boundary.
Any building bigger than one 100m tile (10,000 m2) then inflates that one
tile's building_area_ratio past 1.0 - mathematically impossible for a real
ratio, and it happens more often now that Overture buildings include real
large footprints (malls, big residential blocks) that OSM's sparser set
rarely had. This script instead clips every building to each tile's
boundary before summing area (the same intersection-based pattern
production already uses correctly for green_ratio/water_ratio), so a large
building's area is split proportionally across every tile it actually
overlaps instead of dumped entirely into one.

This is test-case-only: it calls the real, unmodified
create_analysis_grid()/_derive_clutter_class() from production, and only
replaces the buggy building-area-attachment step locally. Nothing in
tools/lte_prediction/ is edited.

Usage:
    python tests/baseline/compute_clutter_from_cache.py --project-id 210 --region taiwan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.ops import transform

from tools.lte_prediction.geo_correction_pipeline import (
    _choose_utm_crs,
    _derive_clutter_class,
    create_analysis_grid,
)

GREEN_LC_SUBTYPES = {"forest", "shrub", "grass"}
GREEN_LU_SUBTYPES = {"park", "recreation", "horticulture", "agriculture"}
GREEN_LU_CLASS = {"park", "garden", "grass", "recreation_ground", "village_green", "pitch", "nature_reserve"}


def load_polygon(data_dir: Path):
    poly_gdf = gpd.read_file(data_dir / "project_polygon.geojson")
    poly = poly_gdf.geometry.iloc[0]
    return transform(lambda x, y: (y, x), poly)


def load_building_gdf(data_dir: Path) -> gpd.GeoDataFrame:
    building_df = pd.read_csv(data_dir / "building_df.csv", low_memory=False)
    geoms = building_df["region_wkt"].apply(shapely_wkt.loads)
    geoms_lonlat = geoms.apply(lambda g: transform(lambda x, y: (y, x), g) if g is not None else None)
    gdf = gpd.GeoDataFrame(building_df, geometry=geoms_lonlat, crs="EPSG:4326")
    return gdf[gdf.geometry.notnull() & gdf.geometry.is_valid & ~gdf.geometry.is_empty]


def attach_building_features_fixed(grid_gdf: gpd.GeoDataFrame, building_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Same output columns as production's attach_building_features(), but clips
    each building to each tile boundary before summing area, instead of
    attributing a building's full area to a single centroid-containing tile."""
    grid_utm = grid_gdf.to_crs(_choose_utm_crs(grid_gdf))
    grid_utm["building_count"] = 0.0
    grid_utm["building_area_sum_m2"] = 0.0
    grid_utm["avg_building_area_m2"] = 0.0
    grid_utm["building_area_ratio"] = 0.0

    if building_gdf.empty:
        return grid_utm.to_crs("EPSG:4326")

    bld_utm = building_gdf.to_crs(grid_utm.crs)[["geometry"]].copy()
    bld_utm["building_id"] = range(len(bld_utm))

    clipped = gpd.overlay(bld_utm, grid_utm[["grid_id", "geometry", "cell_area_m2"]], how="intersection", keep_geom_type=False)
    if clipped.empty:
        return grid_utm.to_crs("EPSG:4326")
    clipped["clip_area_m2"] = clipped.geometry.area

    agg = clipped.groupby("grid_id").agg(
        building_area_sum_m2=("clip_area_m2", "sum"),
        building_count=("building_id", "nunique"),
    )
    grid_utm = grid_utm.drop(columns=["building_count", "building_area_sum_m2", "avg_building_area_m2", "building_area_ratio"])
    grid_utm = grid_utm.merge(agg, on="grid_id", how="left")
    grid_utm["building_count"] = pd.to_numeric(grid_utm["building_count"], errors="coerce").fillna(0.0)
    grid_utm["building_area_sum_m2"] = pd.to_numeric(grid_utm["building_area_sum_m2"], errors="coerce").fillna(0.0)
    grid_utm["avg_building_area_m2"] = (grid_utm["building_area_sum_m2"] / grid_utm["building_count"].replace(0, np.nan)).fillna(0.0)
    grid_utm["building_area_ratio"] = (grid_utm["building_area_sum_m2"] / grid_utm["cell_area_m2"].replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)
    return grid_utm.to_crs("EPSG:4326")


def attach_road_length(grid_gdf: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid_utm = grid_gdf.to_crs(_choose_utm_crs(grid_gdf))
    grid_gdf = grid_gdf.copy()
    grid_gdf["road_length_m"] = 0.0
    roads_l = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].to_crs(grid_utm.crs)
    if roads_l.empty:
        return grid_gdf
    rj = gpd.overlay(roads_l[["geometry"]], grid_utm[["grid_id", "geometry"]], how="intersection", keep_geom_type=False)
    if rj.empty:
        return grid_gdf
    rj["road_seg_m"] = rj.geometry.length
    ragg = rj.groupby("grid_id")["road_seg_m"].sum()
    grid_gdf["road_length_m"] = grid_gdf["grid_id"].map(ragg).fillna(0.0)
    return grid_gdf


def attach_area_ratio(grid_gdf: gpd.GeoDataFrame, layer: gpd.GeoDataFrame, out_col: str) -> gpd.GeoDataFrame:
    grid_utm = grid_gdf.to_crs(_choose_utm_crs(grid_gdf))
    grid_gdf = grid_gdf.copy()
    grid_gdf[out_col] = 0.0
    layer = layer[layer.geometry.type.isin(["Polygon", "MultiPolygon"])].to_crs(grid_utm.crs)
    if layer.empty:
        return grid_gdf
    aj = gpd.overlay(layer[["geometry"]], grid_utm[["grid_id", "geometry", "cell_area_m2"]], how="intersection", keep_geom_type=False)
    if aj.empty:
        return grid_gdf
    aj["clip_area_m2"] = aj.geometry.area
    aagg = aj.groupby("grid_id")["clip_area_m2"].sum()
    ratios = (aagg / grid_utm.set_index("grid_id")["cell_area_m2"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    grid_gdf[out_col] = grid_gdf["grid_id"].map(ratios).fillna(0.0).clip(0.0, 1.0)
    return grid_gdf


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute clutter_class distribution from local cache, with the building-area-attachment bug fixed")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--tile-size-m", type=float, default=100.0)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args(argv)

    data_dir = args.data_root / f"project_{args.project_id}_{args.region}"

    poly_lonlat = load_polygon(data_dir)
    mask_gdf = gpd.GeoDataFrame({"geometry": [poly_lonlat]}, crs="EPSG:4326")

    building_gdf = load_building_gdf(data_dir)
    print(f"[CLUTTER] buildings: {len(building_gdf)}")

    grid_gdf = create_analysis_grid(mask_gdf, args.tile_size_m)
    print(f"[CLUTTER] tiles ({args.tile_size_m:.0f}m): {len(grid_gdf)}")

    grid_gdf = attach_building_features_fixed(grid_gdf, building_gdf)
    print(f"[CLUTTER] building_area_ratio: mean={grid_gdf['building_area_ratio'].mean():.4f} "
          f"max={grid_gdf['building_area_ratio'].max():.4f} (must be <= 1.0 now)")

    roads = gpd.read_file(data_dir / "roads_segment.geojson")
    water = gpd.read_file(data_dir / "water.geojson")
    land_cover = gpd.read_file(data_dir / "land_cover.geojson")
    land_use = gpd.read_file(data_dir / "land_use.geojson")

    green_lc = land_cover[land_cover["subtype"].isin(GREEN_LC_SUBTYPES)]
    green_lu = land_use[land_use["subtype"].isin(GREEN_LU_SUBTYPES) | land_use["class"].isin(GREEN_LU_CLASS)]
    green = gpd.GeoDataFrame(pd.concat([green_lc, green_lu], ignore_index=True), crs="EPSG:4326")
    print(f"[CLUTTER] green features (filtered to genuinely green subtypes): {len(green)}")

    grid_gdf = attach_road_length(grid_gdf, roads)
    grid_gdf = attach_area_ratio(grid_gdf, green, "green_ratio")
    grid_gdf = attach_area_ratio(grid_gdf, water, "water_ratio")

    print(f"[CLUTTER] road_length_m: mean={grid_gdf['road_length_m'].mean():.1f} max={grid_gdf['road_length_m'].max():.1f}")
    print(f"[CLUTTER] green_ratio: mean={grid_gdf['green_ratio'].mean():.4f} max={grid_gdf['green_ratio'].max():.4f}")
    print(f"[CLUTTER] water_ratio: mean={grid_gdf['water_ratio'].mean():.4f} max={grid_gdf['water_ratio'].max():.4f}")

    clutter = _derive_clutter_class(grid_gdf)
    counts = clutter.value_counts()
    pct = (clutter.value_counts(normalize=True) * 100).round(1)

    print()
    print("CLUTTER CLASS DISTRIBUTION (fixed building-area attachment, local cache only):")
    for cls in counts.index:
        print(f"  {cls:<12} {counts[cls]:>4} tiles  ({pct[cls]:.1f}%)")

    grid_gdf["clutter_class"] = clutter
    out_path = data_dir / "clutter_tiles_overture_fixed.geojson"
    grid_gdf.to_file(out_path, driver="GeoJSON")
    print(f"\n[CLUTTER] wrote {out_path}")


if __name__ == "__main__":
    main()
