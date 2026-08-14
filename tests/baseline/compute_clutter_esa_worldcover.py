"""
Classifies project 210's 100m tile grid using ESA WorldCover 10m (2021,
v200) instead of the project-relative quantile classifier in
_derive_clutter_class(), and instead of GHS-SMOD (which was tried and
rejected - too coarse at 1km to see water/green variation within a
project).

ESA WorldCover is the standard clutter-type source used by RF planning
tools implementing ITU-R P.1812: a global, 10m-resolution land-cover map
with 11 fixed classes, produced identically everywhere (not ranked against
this project's own tiles). At 10m resolution each 100m tile contains ~100
WorldCover pixels, so real per-tile land-cover percentages can be computed
directly (built-up %, water %, tree/grass/shrub % = green), then classified
against fixed, absolute thresholds instead of self-relative quantiles.

Source: ESA WorldCover 10m v200 (2021), tile N24E120, downloaded from the
public esa-worldcover S3 bucket (AWS Open Data, CC BY 4.0).
Reads only local cache files - no DB/bridge calls. Does not modify
tools/lte_prediction/ - test-case-only alternative classifier.

Usage:
    python tests/baseline/compute_clutter_esa_worldcover.py --project-id 210 --region taiwan
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
from shapely.ops import transform

from tools.lte_prediction.geo_correction_pipeline import create_analysis_grid

# ESA WorldCover standard class codes
BUILT_UP = 50
WATER = 80
WETLAND = 90
MANGROVES = 95
GREEN_CLASSES = {10, 20, 30}  # Tree cover, Shrubland, Grassland
CROPLAND = 40


def tile_landcover_fractions(raster_path: Path, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    with rasterio.open(raster_path) as src:
        for _, row in grid_gdf.iterrows():
            geom = [row.geometry.__geo_interface__]
            try:
                out_image, _ = rio_mask(src, geom, crop=True, all_touched=True)
            except ValueError:
                rows.append({"grid_id": row["grid_id"], "built_ratio": 0.0, "water_ratio": 0.0, "green_ratio": 0.0, "n_pixels": 0})
                continue
            data = out_image[0]
            valid = data[data != src.nodata] if src.nodata is not None else data.ravel()
            n = valid.size
            if n == 0:
                rows.append({"grid_id": row["grid_id"], "built_ratio": 0.0, "water_ratio": 0.0, "green_ratio": 0.0, "n_pixels": 0})
                continue
            built = float(np.isin(valid, [BUILT_UP]).sum()) / n
            water = float(np.isin(valid, [WATER, WETLAND, MANGROVES]).sum()) / n
            green = float(np.isin(valid, list(GREEN_CLASSES)).sum()) / n
            rows.append({"grid_id": row["grid_id"], "built_ratio": built, "water_ratio": water, "green_ratio": green, "n_pixels": int(n)})
    return pd.DataFrame(rows)


def classify(row) -> str:
    # Fixed, absolute thresholds calibrated against real land-cover
    # percentages (not ranked against this project's own tile population).
    if row["water_ratio"] >= 0.15:
        return "Water"
    if row["green_ratio"] >= 0.40 and row["built_ratio"] < 0.10:
        return "Vegetation"
    if row["built_ratio"] >= 0.60:
        return "Dense Urban"
    if row["built_ratio"] >= 0.30:
        return "Urban"
    if row["built_ratio"] >= 0.05:
        return "Suburban"
    return "Rural/Open"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify project tiles using ESA WorldCover (absolute, standard land-cover classes)")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--tile-size-m", type=float, default=100.0)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument(
        "--worldcover-raster",
        type=Path,
        default=Path(r"C:\Users\PC\AppData\Local\Temp\claude\c--Users-PC-Desktop-S-Tracer-Exe-S-Tracer-Exe\bee613e1-06ad-457d-9fd7-900823b4b9ed\scratchpad\esa_worldcover_N24E120.tif"),
    )
    args = parser.parse_args(argv)

    data_dir = args.data_root / f"project_{args.project_id}_{args.region}"
    poly_gdf = gpd.read_file(data_dir / "project_polygon.geojson")
    poly = poly_gdf.geometry.iloc[0]
    poly_lonlat = transform(lambda x, y: (y, x), poly)
    mask_gdf = gpd.GeoDataFrame({"geometry": [poly_lonlat]}, crs="EPSG:4326")

    grid_gdf = create_analysis_grid(mask_gdf, args.tile_size_m)
    print(f"[WORLDCOVER] tiles ({args.tile_size_m:.0f}m): {len(grid_gdf)}")

    fractions = tile_landcover_fractions(args.worldcover_raster, grid_gdf)
    grid_gdf = grid_gdf.merge(fractions, on="grid_id", how="left")

    print(f"[WORLDCOVER] built_ratio: mean={grid_gdf['built_ratio'].mean():.3f} max={grid_gdf['built_ratio'].max():.3f}")
    print(f"[WORLDCOVER] water_ratio: mean={grid_gdf['water_ratio'].mean():.3f} max={grid_gdf['water_ratio'].max():.3f}")
    print(f"[WORLDCOVER] green_ratio: mean={grid_gdf['green_ratio'].mean():.3f} max={grid_gdf['green_ratio'].max():.3f}")
    print(f"[WORLDCOVER] avg pixels/tile: {grid_gdf['n_pixels'].mean():.1f}")

    grid_gdf["clutter_class"] = grid_gdf.apply(classify, axis=1)
    counts = grid_gdf["clutter_class"].value_counts()
    pct = (grid_gdf["clutter_class"].value_counts(normalize=True) * 100).round(1)

    print()
    print("ESA WORLDCOVER CLUTTER CLASSIFICATION (fixed absolute thresholds, 10m real land-cover data):")
    for cls in counts.index:
        print(f"  {cls:<12} {counts[cls]:>4} tiles  ({pct[cls]:.1f}%)")

    out_path = data_dir / "clutter_tiles_esa_worldcover.geojson"
    grid_gdf.to_file(out_path, driver="GeoJSON")
    print(f"\n[WORLDCOVER] wrote {out_path}")


if __name__ == "__main__":
    main()
