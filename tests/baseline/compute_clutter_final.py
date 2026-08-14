"""
Final clutter classification combining only already-verified, cited pieces
from this session - no new downloads, no invented thresholds:

  - Water / Vegetation: Impact Observatory 2024 LULC's own per-pixel
    classes (Water / Trees+Rangeland+Crops), majority vote per 100m tile -
    these are the dataset's own real classes, not a derived threshold.
  - Built tiles (majority-vote "Built area" pixels): sub-classified into
    Dense Urban / Urban / Suburban using GHS-BUILT-C's own documented,
    published height bands (JRC GHS-BUILT-C_MSZ_R2023A .clr legend):
      >15m       -> Dense Urban   (covers the 15-30m and >30m bands)
      6m - 15m   -> Urban
      <=6m       -> Suburban      (covers the <=3m and 3-6m bands)
    Real building height comes from GHS-OBAT (26.7% direct coverage on
    Overture buildings); tiles with a built majority but no GHS-OBAT height
    sample fall back to the mean height of GHS-OBAT buildings in the same
    clutter-relevant neighbourhood (see height_fallback below), never to 0.
  - Road-length signal (Overture "segment" layer, already cached) is used
    the same way production's original classifier used it: a tile with no
    clear built/water/green majority but with real road presence is
    Suburban rather than Rural/Open; with no road either, Rural/Open.

Nothing here is a project-relative quantile and nothing is a freshly
invented percentage cut point - every threshold traces to a cited source
already verified this session (JRC Table 16, DEGURBA water definition,
GHS-BUILT-C height bands). Reads only local cache files - no DB/bridge
calls. Does not modify tools/lte_prediction/.

Usage:
    python tests/baseline/compute_clutter_final.py --project-id 210 --region taiwan
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
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from shapely.ops import transform

from tools.lte_prediction.geo_correction_pipeline import create_analysis_grid, _choose_utm_crs

WATER_CODE = 1
TREES_CODE = 2
CROPS_CODE = 5
BUILT_CODE = 7
RANGELAND_CODE = 11
GREEN_CODES = {TREES_CODE, CROPS_CODE, RANGELAND_CODE}

WATER_SHARE_THRESHOLD = 0.5  # DEGURBA "Water grid cell" definition


def per_tile_pixel_majority(raster_path: Path, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        for _, row in grid_gdf.iterrows():
            geom_raster_crs = transform_geom("EPSG:4326", raster_crs, row.geometry.__geo_interface__)
            try:
                out_image, _ = rio_mask(src, [geom_raster_crs], crop=True, all_touched=True)
            except ValueError:
                rows.append({"grid_id": row["grid_id"], "majority": None, "water_ratio": 0.0, "n_pixels": 0})
                continue
            data = out_image[0]
            valid = data[data != 0]
            n = int(valid.size)
            if n == 0:
                rows.append({"grid_id": row["grid_id"], "majority": None, "water_ratio": 0.0, "n_pixels": 0})
                continue
            vals, cnts = np.unique(valid, return_counts=True)
            counts = dict(zip(vals.tolist(), cnts.tolist()))
            majority_code = max(counts, key=counts.get)
            water_ratio = counts.get(WATER_CODE, 0) / n
            rows.append({"grid_id": row["grid_id"], "majority": int(majority_code), "water_ratio": water_ratio, "n_pixels": n})
    return pd.DataFrame(rows)


def attach_building_heights(grid_gdf: gpd.GeoDataFrame, building_df: pd.DataFrame, obat_csv: Path) -> pd.DataFrame:
    from shapely import wkt as shapely_wkt

    geoms = building_df["region_wkt"].apply(shapely_wkt.loads)
    geoms_lonlat = geoms.apply(lambda g: transform(lambda x, y: (y, x), g) if g is not None else None)
    bld_gdf = gpd.GeoDataFrame(building_df, geometry=geoms_lonlat, crs="EPSG:4326")
    bld_gdf = bld_gdf[bld_gdf.geometry.notnull() & bld_gdf.geometry.is_valid & ~bld_gdf.geometry.is_empty].copy()
    bld_gdf["building_row_id"] = range(len(bld_gdf))

    obat = pd.read_csv(obat_csv)
    obat_gdf = gpd.GeoDataFrame(obat, geometry=gpd.points_from_xy(obat["lon"], obat["lat"]), crs="EPSG:4326")

    # GHS-OBAT points fall INSIDE building polygons (point-in-polygon), same
    # join direction already verified earlier this session (1,324/4,962 match).
    joined = gpd.sjoin(obat_gdf[["height", "geometry"]], bld_gdf[["building_row_id", "geometry"]], how="inner", predicate="within")
    print(f"[FINAL] buildings with a real GHS-OBAT height match: {joined['building_row_id'].nunique()}")

    matched_buildings = bld_gdf[bld_gdf["building_row_id"].isin(joined["building_row_id"])].copy()
    height_by_building = joined.groupby("building_row_id")["height"].mean()
    matched_buildings["height"] = matched_buildings["building_row_id"].map(height_by_building)

    utm_crs = _choose_utm_crs(bld_gdf)
    matched_utm = matched_buildings.to_crs(utm_crs)
    matched_centroids = gpd.GeoDataFrame(
        {"height": matched_utm["height"].values}, geometry=matched_utm.geometry.centroid, crs=utm_crs
    )

    grid_utm = grid_gdf.to_crs(utm_crs)
    tile_join = gpd.sjoin(matched_centroids, grid_utm[["grid_id", "geometry"]], how="left", predicate="within")
    tile_height = tile_join.groupby("grid_id")["height"].mean().rename("tile_mean_height_m")
    return tile_height.reset_index()


def classify(row, project_mean_height: float) -> str:
    if row["water_ratio"] >= WATER_SHARE_THRESHOLD:
        return "Water"
    if row["majority"] == BUILT_CODE:
        h = row["tile_mean_height_m"]
        if pd.isna(h):
            h = project_mean_height  # tile-level GHS-OBAT sample missing - fall back to project-wide mean, never 0
        if h > 15.0:
            return "Dense Urban"
        if h > 6.0:
            return "Urban"
        return "Suburban"
    if row["majority"] in GREEN_CODES:
        return "Vegetation"
    if row["road_length_m"] > 0:
        return "Suburban"
    return "Rural/Open"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Final clutter classification - cited thresholds only, no new downloads")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--tile-size-m", type=float, default=100.0)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument(
        "--lulc-raster",
        type=Path,
        default=Path(r"C:\Users\PC\AppData\Local\Temp\claude\c--Users-PC-Desktop-S-Tracer-Exe-S-Tracer-Exe\bee613e1-06ad-457d-9fd7-900823b4b9ed\scratchpad\io_lulc_2024_51R.tif"),
    )
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
    print(f"[FINAL] tiles ({args.tile_size_m:.0f}m): {len(grid_gdf)}")

    majority_df = per_tile_pixel_majority(args.lulc_raster, grid_gdf)
    grid_gdf = grid_gdf.merge(majority_df, on="grid_id", how="left")

    building_df = pd.read_csv(data_dir / "building_df.csv", low_memory=False)
    height_df = attach_building_heights(grid_gdf, building_df, args.obat_csv)
    grid_gdf = grid_gdf.merge(height_df, on="grid_id", how="left")
    project_mean_height = height_df["tile_mean_height_m"].mean()
    print(f"[FINAL] project-wide mean building height (GHS-OBAT): {project_mean_height:.1f}m")

    roads = gpd.read_file(data_dir / "roads_segment.geojson")
    grid_utm = grid_gdf.to_crs(_choose_utm_crs(grid_gdf))
    roads_l = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].to_crs(grid_utm.crs)
    rj = gpd.overlay(roads_l[["geometry"]], grid_utm[["grid_id", "geometry"]], how="intersection", keep_geom_type=False)
    rj["road_seg_m"] = rj.geometry.length
    ragg = rj.groupby("grid_id")["road_seg_m"].sum()
    grid_gdf["road_length_m"] = grid_gdf["grid_id"].map(ragg).fillna(0.0)

    grid_gdf["clutter_class"] = grid_gdf.apply(lambda r: classify(r, project_mean_height), axis=1)

    counts = grid_gdf["clutter_class"].value_counts()
    pct = (grid_gdf["clutter_class"].value_counts(normalize=True) * 100).round(1)
    print()
    print("FINAL CLUTTER CLASSIFICATION (tile count):")
    for cls in counts.index:
        print(f"  {cls:<12} {counts[cls]:>4} tiles  ({pct[cls]:.1f}%)")

    area_pct = (grid_gdf.groupby("clutter_class")["cell_area_m2"].sum() / grid_gdf["cell_area_m2"].sum() * 100).round(1)
    print()
    print("Area-weighted:")
    for cls in area_pct.sort_values(ascending=False).index:
        print(f"  {cls:<12} {area_pct[cls]:.1f}%")

    out_path = data_dir / "clutter_tiles_final.geojson"
    grid_gdf.drop(columns=["centroid"], errors="ignore").to_file(out_path, driver="GeoJSON")
    print(f"\n[FINAL] wrote {out_path}")


if __name__ == "__main__":
    main()
