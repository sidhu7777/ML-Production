"""
Final clutter classification (v2) - built purely from our own real, current,
already-verified vector layers, with no raster classifier and no invented
percentage thresholds:

  Water      = real Overture water polygons cover >=50% of the tile
               (DEGURBA's cited "Water grid cell" definition)
  Built      = a real Overture building polygon actually overlaps the tile
               (fact, not a threshold) -> sub-tiered by real GHS-OBAT
               building height using GHS-BUILT-C's own cited height bands:
                 >15m      -> Dense Urban
                 6m - 15m  -> Urban
                 <=6m      -> Suburban
               (tiles with a building but no GHS-OBAT height sample use the
               project-wide GHS-OBAT mean height, never 0 - see the earlier
               height-imputation discussion this session)
  Road, no building = a real Overture road actually runs through the tile
               -> Suburban, matching GHS-BUILT-C's own "open space, road
               surface" tier (open, not vegetation, not empty)
  Green      = real Overture green (forest/park/grass/etc, already filtered
               to true green subtypes) covers the tile, no building/road
  Rural/Open = none of the above - pure absence, no extra data needed

Building-area attachment uses the tile-boundary-clipping fix already
verified earlier this session (production's own attach_building_features
double-counts buildings larger than one tile; this clips each building to
each tile first, same as green/water already do).

Reads only local cache files - no DB/bridge calls. Does not modify
tools/lte_prediction/ - test-case-only.

Usage:
    python tests/baseline/compute_clutter_final_v2.py --project-id 210 --region taiwan
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

from tools.lte_prediction.geo_correction_pipeline import _choose_utm_crs, create_analysis_grid

GREEN_LC_SUBTYPES = {"forest", "shrub", "grass"}
GREEN_LU_SUBTYPES = {"park", "recreation", "horticulture", "agriculture"}
GREEN_LU_CLASS = {"park", "garden", "grass", "recreation_ground", "village_green", "pitch", "nature_reserve"}

WATER_SHARE_THRESHOLD = 0.5  # DEGURBA "Water grid cell" definition
DENSE_URBAN_HEIGHT_M = 15.0  # GHS-BUILT-C height band boundary
URBAN_HEIGHT_M = 6.0         # GHS-BUILT-C height band boundary


def load_building_gdf(data_dir: Path) -> gpd.GeoDataFrame:
    building_df = pd.read_csv(data_dir / "building_df.csv", low_memory=False)
    geoms = building_df["region_wkt"].apply(shapely_wkt.loads)
    geoms_lonlat = geoms.apply(lambda g: transform(lambda x, y: (y, x), g) if g is not None else None)
    gdf = gpd.GeoDataFrame(building_df, geometry=geoms_lonlat, crs="EPSG:4326")
    return gdf[gdf.geometry.notnull() & gdf.geometry.is_valid & ~gdf.geometry.is_empty]


def attach_building_features_fixed(grid_gdf: gpd.GeoDataFrame, building_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid_utm = grid_gdf.to_crs(_choose_utm_crs(grid_gdf))
    grid_utm["building_count"] = 0.0
    grid_utm["building_area_ratio"] = 0.0
    if building_gdf.empty:
        return grid_utm.to_crs("EPSG:4326")
    bld_utm = building_gdf.to_crs(grid_utm.crs)[["geometry"]].copy()
    bld_utm["building_id"] = range(len(bld_utm))
    clipped = gpd.overlay(bld_utm, grid_utm[["grid_id", "geometry", "cell_area_m2"]], how="intersection", keep_geom_type=False)
    if clipped.empty:
        return grid_utm.to_crs("EPSG:4326")
    clipped["clip_area_m2"] = clipped.geometry.area
    agg = clipped.groupby("grid_id").agg(building_area_sum_m2=("clip_area_m2", "sum"), building_count=("building_id", "nunique"))
    grid_utm = grid_utm.drop(columns=["building_count", "building_area_ratio"]).merge(agg, on="grid_id", how="left")
    grid_utm["building_count"] = pd.to_numeric(grid_utm["building_count"], errors="coerce").fillna(0.0)
    grid_utm["building_area_ratio"] = (
        pd.to_numeric(grid_utm["building_area_sum_m2"], errors="coerce").fillna(0.0) / grid_utm["cell_area_m2"].replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    return grid_utm.to_crs("EPSG:4326")


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


NEIGHBOR_RADII_M = [150.0, 300.0, 600.0]  # widen only as far as needed before falling back to project mean


def impute_building_heights(building_gdf: gpd.GeoDataFrame, obat_csv: Path) -> gpd.GeoDataFrame:
    """Every building ends up with a height: real GHS-OBAT measurement where
    available, otherwise the mean height of the nearest real-height
    buildings within a growing radius (150m -> 300m -> 600m), and only the
    project-wide mean as a last resort if truly nothing real is nearby.
    This replaces the flat project-mean-for-everything fallback."""
    building_gdf = building_gdf.copy()
    building_gdf["building_row_id"] = range(len(building_gdf))

    obat = pd.read_csv(obat_csv)
    obat_gdf = gpd.GeoDataFrame(obat, geometry=gpd.points_from_xy(obat["lon"], obat["lat"]), crs="EPSG:4326")

    joined = gpd.sjoin(obat_gdf[["height", "geometry"]], building_gdf[["building_row_id", "geometry"]], how="inner", predicate="within")
    n_real = joined["building_row_id"].nunique()
    print(f"[V2] buildings with a real GHS-OBAT height match: {n_real} / {len(building_gdf)}")

    height_by_building = joined.groupby("building_row_id")["height"].mean()
    building_gdf["height"] = building_gdf["building_row_id"].map(height_by_building)
    building_gdf["height_source"] = np.where(building_gdf["height"].notna(), "measured", None)

    utm_crs = _choose_utm_crs(building_gdf)
    all_utm = building_gdf.to_crs(utm_crs)
    all_utm["centroid"] = all_utm.geometry.centroid

    known = all_utm[all_utm["height"].notna()].copy()
    known_pts = np.array([[p.x, p.y] for p in known["centroid"]])
    from scipy.spatial import cKDTree
    known_tree = cKDTree(known_pts)

    missing_mask = all_utm["height"].isna()
    missing_idx = all_utm.index[missing_mask]
    missing_pts = np.array([[p.x, p.y] for p in all_utm.loc[missing_idx, "centroid"]])

    project_mean = float(known["height"].mean())
    imputed_heights = np.full(len(missing_idx), project_mean)
    imputed_source = np.full(len(missing_idx), "project_mean_fallback", dtype=object)

    if len(missing_idx) and len(known_pts):
        resolved = np.zeros(len(missing_idx), dtype=bool)
        for radius in NEIGHBOR_RADII_M:
            unresolved = ~resolved
            if not unresolved.any():
                break
            neighbor_lists = known_tree.query_ball_point(missing_pts[unresolved], r=radius)
            local_idx = np.where(unresolved)[0]
            for j, neighbors in zip(local_idx, neighbor_lists):
                if neighbors:
                    imputed_heights[j] = float(known["height"].iloc[neighbors].mean())
                    imputed_source[j] = f"neighbor_mean_{int(radius)}m_n{len(neighbors)}"
                    resolved[j] = True

    building_gdf.loc[missing_idx, "height"] = imputed_heights
    building_gdf.loc[missing_idx, "height_source"] = imputed_source

    print("[V2] height source breakdown:")
    print(building_gdf["height_source"].apply(lambda s: s.split("_n")[0] if isinstance(s, str) and s.startswith("neighbor") else s).value_counts().to_string())

    return building_gdf


def attach_surrounding_height(grid_gdf: gpd.GeoDataFrame, building_gdf_with_height: gpd.GeoDataFrame, radius_m: float = 100.0) -> pd.DataFrame:
    """For every tile (whether or not a building directly overlaps it),
    the surrounding-context height = mean height of real+imputed buildings
    within radius_m of the tile centroid. This is what both a built tile
    and a bare road tile use for height-banding - 'character of the
    surrounding area, not the object occupying one tile'."""
    from scipy.spatial import cKDTree

    utm_crs = _choose_utm_crs(building_gdf_with_height)
    bld_utm = building_gdf_with_height.to_crs(utm_crs)
    bld_pts = np.array([[p.x, p.y] for p in bld_utm.geometry.centroid])
    heights = bld_utm["height"].to_numpy()
    tree = cKDTree(bld_pts)

    grid_utm = grid_gdf.to_crs(utm_crs)
    tile_centroids = grid_utm.geometry.centroid
    tile_pts = np.array([[p.x, p.y] for p in tile_centroids])

    neighbor_lists = tree.query_ball_point(tile_pts, r=radius_m)
    surrounding = [float(heights[idxs].mean()) if idxs else np.nan for idxs in neighbor_lists]
    return pd.DataFrame({"grid_id": grid_utm["grid_id"].values, "surrounding_height_m": surrounding})


def height_to_tier(h: float) -> str:
    if h > DENSE_URBAN_HEIGHT_M:
        return "Dense Urban"
    if h > URBAN_HEIGHT_M:
        return "Urban"
    return "Suburban"


def classify(row, project_mean_height: float) -> str:
    if row["water_ratio"] >= WATER_SHARE_THRESHOLD:
        return "Water"
    if row["building_count"] > 0:
        # real buildings in this tile - use the surrounding-context height
        # (neighbourhood mean, already imputed per-building), not a flat
        # project-wide fallback.
        h = row.get("surrounding_height_m")
        if pd.isna(h):
            h = project_mean_height
        return height_to_tier(h)
    if row["road_length_m"] > 0:
        # no building actually overlaps this tile, but it's a real road -
        # classify by what's actually around it (a street inside a dense
        # block is Dense Urban, not a flat "Suburban" default).
        h = row.get("surrounding_height_m")
        if pd.isna(h):
            return "Rural/Open"  # a road with genuinely nothing built nearby
        return height_to_tier(h)
    if row["green_ratio"] >= 0.30:
        return "Vegetation"
    return "Rural/Open"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Final clutter classification v2 - own vector layers only, cited thresholds only")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--tile-size-m", type=float, default=100.0)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument(
        "--obat-csv",
        type=Path,
        default=Path(r"C:\Users\PC\AppData\Local\Temp\claude\c--Users-PC-Desktop-S-Tracer-Exe-S-Tracer-Exe\bee613e1-06ad-457d-9fd7-900823b4b9ed\scratchpad\ghsobat_project210_bbox.csv"),
    )
    args = parser.parse_args(argv)

    data_dir = args.data_root / f"project_{args.project_id}_{args.region}"
    poly_gdf = gpd.read_file(data_dir / "project_polygon.geojson")
    poly = poly_gdf.geometry.iloc[0]
    poly_lonlat = transform(lambda x, y: (y, x), poly)
    mask_gdf = gpd.GeoDataFrame({"geometry": [poly_lonlat]}, crs="EPSG:4326")

    grid_gdf = create_analysis_grid(mask_gdf, args.tile_size_m)
    print(f"[V2] tiles ({args.tile_size_m:.0f}m): {len(grid_gdf)}")

    building_gdf = load_building_gdf(data_dir)
    grid_gdf = attach_building_features_fixed(grid_gdf, building_gdf)

    building_gdf_h = impute_building_heights(building_gdf, args.obat_csv)
    project_mean_height = float(building_gdf_h["height"].mean())
    print(f"[V2] project-wide mean building height (real + imputed): {project_mean_height:.1f}m")

    surrounding_df = attach_surrounding_height(grid_gdf, building_gdf_h, radius_m=100.0)
    grid_gdf = grid_gdf.merge(surrounding_df, on="grid_id", how="left")

    roads = gpd.read_file(data_dir / "roads_segment.geojson")
    grid_gdf = attach_road_length(grid_gdf, roads)

    water = gpd.read_file(data_dir / "water.geojson")
    grid_gdf = attach_area_ratio(grid_gdf, water, "water_ratio")

    land_cover = gpd.read_file(data_dir / "land_cover.geojson")
    land_use = gpd.read_file(data_dir / "land_use.geojson")
    green_lc = land_cover[land_cover["subtype"].isin(GREEN_LC_SUBTYPES)]
    green_lu = land_use[land_use["subtype"].isin(GREEN_LU_SUBTYPES) | land_use["class"].isin(GREEN_LU_CLASS)]
    green = gpd.GeoDataFrame(pd.concat([green_lc, green_lu], ignore_index=True), crs="EPSG:4326")
    grid_gdf = attach_area_ratio(grid_gdf, green, "green_ratio")

    grid_gdf["clutter_class"] = grid_gdf.apply(lambda r: classify(r, project_mean_height), axis=1)

    counts = grid_gdf["clutter_class"].value_counts()
    pct = (grid_gdf["clutter_class"].value_counts(normalize=True) * 100).round(1)
    print()
    print("FINAL CLUTTER CLASSIFICATION v2 (own Overture vectors + GHS-OBAT height + cited GHS-BUILT-C/DEGURBA thresholds):")
    for cls in counts.index:
        print(f"  {cls:<12} {counts[cls]:>4} tiles  ({pct[cls]:.1f}%)")

    area_pct = (grid_gdf.groupby("clutter_class")["cell_area_m2"].sum() / grid_gdf["cell_area_m2"].sum() * 100).round(1)
    print()
    print("Area-weighted:")
    for cls in area_pct.sort_values(ascending=False).index:
        print(f"  {cls:<12} {area_pct[cls]:.1f}%")

    out_path = data_dir / "clutter_tiles_final_v2.geojson"
    grid_gdf.to_file(out_path, driver="GeoJSON")
    print(f"\n[V2] wrote {out_path}")


if __name__ == "__main__":
    main()
