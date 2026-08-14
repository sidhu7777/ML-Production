"""
Classifies project 210's 100m tile grid using GHS-BUILT-C (Morphological
Settlement Zone, R2023A) - the real official JRC classification that
combines built-up surface fraction, vegetation intensity (NDVI), water,
roads, and building height into one product, at 10m resolution.

This is the graduated, multi-tier classification that ESA WorldCover /
Impact Observatory + JRC Table 16 could not provide on their own: JRC
publishes only ONE calibrated "dense built" cut point for third-party
datasets (Table 16), and DEGURBA's own SMOD is population-based at 1km
(too coarse). GHS-BUILT-C's MSZ layer is JRC's own published multi-class
legend, with real building-height bands, not an interpolation or a
project-specific hardcoded threshold.

Real class legend (from the raster's own .clr file, official JRC
GHS-BUILT-C_MSZ_R2023A product, CC BY 4.0):
  1  MSZ open space, low vegetation      (NDVI <= 0.3)
  2  MSZ open space, medium vegetation   (0.3 < NDVI <= 0.5)
  3  MSZ open space, high vegetation     (NDVI > 0.5)
  4  MSZ open space, water                (LAND share < 0.5)
  5  MSZ open space, road surface
  11 MSZ built, residential, height <=3m
  12 MSZ built, residential, 3-6m
  13 MSZ built, residential, 6-15m
  14 MSZ built, residential, 15-30m
  15 MSZ built, residential, >30m
  21 MSZ built, non-residential, height <=3m
  22 MSZ built, non-residential, 3-6m
  23 MSZ built, non-residential, 6-15m
  24 MSZ built, non-residential, 15-30m
  25 MSZ built, non-residential, >30m
(cells outside the Morphological Settlement Zone entirely are NoData/0 -
open countryside with no nearby built-up markers at all)

Grouping used here (built classes only, both residential/non-residential
share the same height bands so they are merged for a clutter/RF purpose -
height, not land use function, is what matters for path loss):
  Dense Urban: height bands 15-30m or >30m (classes 14,15,24,25)
  Urban:       height band 6-15m            (classes 13,23)
  Suburban:    height bands <=3m or 3-6m     (classes 11,12,21,22)
  Water:       class 4
  Vegetation:  classes 1-3 (any open-space vegetation tier)
  Rural/Open:  class 5 (road) or NoData (outside the MSZ - open countryside)

Source: JRC GHS-BUILT-C_GLOBE_R2023A, tile R6_C30 (covers Taiwan),
downloaded from jeodpp.jrc.ec.europa.eu open data FTP, CC BY 4.0.
Reads only local cache files - no DB/bridge calls. Does not modify
tools/lte_prediction/ - test-case-only alternative classifier.

Usage:
    python tests/baseline/compute_clutter_ghs_built_c.py --project-id 210 --region taiwan
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

from tools.lte_prediction.geo_correction_pipeline import create_analysis_grid

DENSE_URBAN_CODES = {14, 15, 24, 25}
URBAN_CODES = {13, 23}
SUBURBAN_CODES = {11, 12, 21, 22}
WATER_CODES = {4}
VEGETATION_CODES = {1, 2, 3}
ROAD_CODE = {5}


def classify_pixel_counts(counts: dict) -> str:
    dense = sum(counts.get(c, 0) for c in DENSE_URBAN_CODES)
    urban = sum(counts.get(c, 0) for c in URBAN_CODES)
    suburban = sum(counts.get(c, 0) for c in SUBURBAN_CODES)
    water = sum(counts.get(c, 0) for c in WATER_CODES)
    veg = sum(counts.get(c, 0) for c in VEGETATION_CODES)
    tiers = {"Dense Urban": dense, "Urban": urban, "Suburban": suburban, "Water": water, "Vegetation": veg}
    total_named = sum(tiers.values())
    if total_named == 0:
        return "Rural/Open"
    return max(tiers, key=tiers.get)


def tile_msz_classification(raster_path: Path, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata
        for _, row in grid_gdf.iterrows():
            geom_wgs84 = row.geometry.__geo_interface__
            geom_raster_crs = transform_geom("EPSG:4326", raster_crs, geom_wgs84)
            try:
                out_image, _ = rio_mask(src, [geom_raster_crs], crop=True, all_touched=True)
            except ValueError:
                rows.append({"grid_id": row["grid_id"], "clutter_class": "Rural/Open", "n_pixels": 0})
                continue
            data = out_image[0]
            valid = data[data != nodata] if nodata is not None else data[data != 0]
            n = int(valid.size)
            if n == 0:
                rows.append({"grid_id": row["grid_id"], "clutter_class": "Rural/Open", "n_pixels": 0})
                continue
            vals, cnts = np.unique(valid, return_counts=True)
            counts = dict(zip(vals.tolist(), cnts.tolist()))
            cls = classify_pixel_counts(counts)
            rows.append({"grid_id": row["grid_id"], "clutter_class": cls, "n_pixels": n})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify project tiles using GHS-BUILT-C MSZ (official JRC multi-tier classification)")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--tile-size-m", type=float, default=100.0)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument(
        "--msz-raster",
        type=Path,
        default=Path(r"C:\Users\PC\AppData\Local\Temp\claude\c--Users-PC-Desktop-S-Tracer-Exe-S-Tracer-Exe\bee613e1-06ad-457d-9fd7-900823b4b9ed\scratchpad\msz_R6_C30\GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A_54009_10_V1_0_R6_C30.tif"),
    )
    args = parser.parse_args(argv)

    data_dir = args.data_root / f"project_{args.project_id}_{args.region}"
    poly_gdf = gpd.read_file(data_dir / "project_polygon.geojson")
    poly = poly_gdf.geometry.iloc[0]
    poly_lonlat = transform(lambda x, y: (y, x), poly)
    mask_gdf = gpd.GeoDataFrame({"geometry": [poly_lonlat]}, crs="EPSG:4326")

    grid_gdf = create_analysis_grid(mask_gdf, args.tile_size_m)
    print(f"[GHS_BUILT_C] tiles ({args.tile_size_m:.0f}m): {len(grid_gdf)}")

    classified = tile_msz_classification(args.msz_raster, grid_gdf)
    grid_gdf = grid_gdf.merge(classified, on="grid_id", how="left")
    grid_gdf["clutter_class"] = grid_gdf["clutter_class"].fillna("Rural/Open")

    print(f"[GHS_BUILT_C] avg pixels/tile: {grid_gdf['n_pixels'].mean():.1f}")

    counts = grid_gdf["clutter_class"].value_counts()
    pct = (grid_gdf["clutter_class"].value_counts(normalize=True) * 100).round(1)

    print()
    print("GHS-BUILT-C MSZ CLUTTER CLASSIFICATION (official JRC multi-tier legend, tile count):")
    for cls in counts.index:
        print(f"  {cls:<12} {counts[cls]:>4} tiles  ({pct[cls]:.1f}%)")

    area_pct = (grid_gdf.groupby("clutter_class")["cell_area_m2"].sum() / grid_gdf["cell_area_m2"].sum() * 100).round(1)
    print()
    print("Area-weighted:")
    for cls in area_pct.sort_values(ascending=False).index:
        print(f"  {cls:<12} {area_pct[cls]:.1f}%")

    out_path = data_dir / "clutter_tiles_ghs_built_c.geojson"
    grid_gdf.to_file(out_path, driver="GeoJSON")
    print(f"\n[GHS_BUILT_C] wrote {out_path}")


if __name__ == "__main__":
    main()
