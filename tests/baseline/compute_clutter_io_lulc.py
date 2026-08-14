"""
Classifies project 210's 100m tile grid using Impact Observatory / Esri /
Microsoft's 10m Annual Land Use Land Cover (2024 edition) - a genuinely
current alternative to ESA WorldCover (frozen at 2021), still 10m
resolution, still CC BY-4.0, still a plain public file download (no Earth
Engine account, no paid tier - unlike Google Dynamic World).

Source: https://api.impactobservatory.com/stac-aws/collections/io-10m-annual-lulc
tile 51R, year 2024, downloaded from the public io-10m-annual-lulc S3 bucket.

9 classes: 1=Water, 2=Trees, 4=Flooded vegetation, 5=Crops, 7=Built area,
8=Bare ground, 9=Snow/ice, 10=Clouds, 11=Rangeland (shrub/scrub/grass).

Classification thresholds are NOT hand-picked - they come from the official
JRC GHS-DUG (Degree of Urbanisation Grid) User Guide, Version 6
(https://human-settlement.emergency.copernicus.eu/tools/GHS-DUG_User_Guide.pdf):
  - Water: >=0.5 (50%) permanent water share - DEGURBA's own published
    definition of a "Water grid cell" (User Guide p.6, section on GHSL SMOD
    Level 2 definitions).
  - Dense/highly-built-up: >=0.80 (80%) built-up share - Table 16
    ("Threshold values for several built-up datasets"), the row for
    "ESRI (2020)" (doi 10.1109/IGARSS47720.2021.9553499), the same
    Impact Observatory/Esri/Microsoft dataset family used here. JRC
    calibrates a different threshold per dataset because different
    built-up layers have different systematic saturation behaviour -
    0.80 is the value JRC itself found equivalent to "highly built-up"
    (office parks, malls, factories, dense settlement) for this dataset,
    not a threshold invented for this project.

The User Guide does not publish an official multi-tier (Urban/Suburban/
Rural) built-up-percentage breakdown the way it does for population
density (50/300/1500 people/km2) - only this one calibrated "dense/
highly-built-up" cut point per dataset. So only two tiers below are
cited; anything under the Dense Urban threshold that isn't Water is left
as a single "Other" bucket rather than inventing uncited sub-thresholds.

Reads only local cache files - no DB/bridge calls. Does not modify
tools/lte_prediction/ - test-case-only alternative classifier.

Usage:
    python tests/baseline/compute_clutter_io_lulc.py --project-id 210 --region taiwan
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
from shapely.geometry import shape
from shapely.ops import transform

from tools.lte_prediction.geo_correction_pipeline import create_analysis_grid

WATER = 1
BUILT = 7
GREEN_CLASSES = {2, 5, 11}  # Trees, Crops, Rangeland


def tile_landcover_fractions(raster_path: Path, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        for _, row in grid_gdf.iterrows():
            geom_wgs84 = row.geometry.__geo_interface__
            geom_raster_crs = transform_geom("EPSG:4326", raster_crs, geom_wgs84)
            try:
                out_image, _ = rio_mask(src, [geom_raster_crs], crop=True, all_touched=True)
            except ValueError:
                rows.append({"grid_id": row["grid_id"], "built_ratio": 0.0, "water_ratio": 0.0, "green_ratio": 0.0, "n_pixels": 0})
                continue
            data = out_image[0]
            valid = data[data != 0]  # 0 = No Data
            n = valid.size
            if n == 0:
                rows.append({"grid_id": row["grid_id"], "built_ratio": 0.0, "water_ratio": 0.0, "green_ratio": 0.0, "n_pixels": 0})
                continue
            built = float((valid == BUILT).sum()) / n
            water = float((valid == WATER).sum()) / n
            green = float(np.isin(valid, list(GREEN_CLASSES)).sum()) / n
            rows.append({"grid_id": row["grid_id"], "built_ratio": built, "water_ratio": water, "green_ratio": green, "n_pixels": int(n)})
    return pd.DataFrame(rows)


# Cited thresholds only - see module docstring for sources.
WATER_SHARE_THRESHOLD = 0.5   # DEGURBA "Water grid cell" definition (GHS-DUG User Guide)
DENSE_BUILT_THRESHOLD = 0.80  # Table 16, "ESRI (2020)" row - JRC-calibrated for this dataset family


def classify(row) -> str:
    if row["water_ratio"] >= WATER_SHARE_THRESHOLD:
        return "Water"
    if row["built_ratio"] >= DENSE_BUILT_THRESHOLD:
        return "Dense Urban"
    return "Other"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify project tiles using Impact Observatory 2024 LULC (current, free, no account)")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--tile-size-m", type=float, default=100.0)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument(
        "--lulc-raster",
        type=Path,
        default=Path(r"C:\Users\PC\AppData\Local\Temp\claude\c--Users-PC-Desktop-S-Tracer-Exe-S-Tracer-Exe\bee613e1-06ad-457d-9fd7-900823b4b9ed\scratchpad\io_lulc_2024_51R.tif"),
    )
    args = parser.parse_args(argv)

    data_dir = args.data_root / f"project_{args.project_id}_{args.region}"
    poly_gdf = gpd.read_file(data_dir / "project_polygon.geojson")
    poly = poly_gdf.geometry.iloc[0]
    poly_lonlat = transform(lambda x, y: (y, x), poly)
    mask_gdf = gpd.GeoDataFrame({"geometry": [poly_lonlat]}, crs="EPSG:4326")

    grid_gdf = create_analysis_grid(mask_gdf, args.tile_size_m)
    print(f"[IO_LULC] tiles ({args.tile_size_m:.0f}m): {len(grid_gdf)}")

    fractions = tile_landcover_fractions(args.lulc_raster, grid_gdf)
    grid_gdf = grid_gdf.merge(fractions, on="grid_id", how="left")

    print(f"[IO_LULC] built_ratio: mean={grid_gdf['built_ratio'].mean():.3f} max={grid_gdf['built_ratio'].max():.3f}")
    print(f"[IO_LULC] water_ratio: mean={grid_gdf['water_ratio'].mean():.3f} max={grid_gdf['water_ratio'].max():.3f}")
    print(f"[IO_LULC] green_ratio: mean={grid_gdf['green_ratio'].mean():.3f} max={grid_gdf['green_ratio'].max():.3f}")
    print(f"[IO_LULC] avg pixels/tile: {grid_gdf['n_pixels'].mean():.1f}")

    grid_gdf["clutter_class"] = grid_gdf.apply(classify, axis=1)
    counts = grid_gdf["clutter_class"].value_counts()
    pct = (grid_gdf["clutter_class"].value_counts(normalize=True) * 100).round(1)

    print()
    print("IMPACT OBSERVATORY 2024 LULC CLUTTER CLASSIFICATION (cited thresholds - JRC GHS-DUG Table 16 + DEGURBA water definition):")
    for cls in counts.index:
        print(f"  {cls:<12} {counts[cls]:>4} tiles  ({pct[cls]:.1f}%)")

    area_pct = (grid_gdf.groupby("clutter_class")["cell_area_m2"].sum() / grid_gdf["cell_area_m2"].sum() * 100).round(1)
    print()
    print("Area-weighted (not just tile count):")
    for cls in area_pct.sort_values(ascending=False).index:
        print(f"  {cls:<12} {area_pct[cls]:.1f}%")

    out_path = data_dir / "clutter_tiles_io_lulc_2024.geojson"
    grid_gdf.to_file(out_path, driver="GeoJSON")
    print(f"\n[IO_LULC] wrote {out_path}")


if __name__ == "__main__":
    main()
