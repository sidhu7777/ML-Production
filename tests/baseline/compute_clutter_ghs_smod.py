"""
Classifies project 210's 100m tile grid using GHS-SMOD (Degree of
Urbanisation) instead of the project-relative quantile classifier in
_derive_clutter_class(). GHS-SMOD is the UN Statistical Commission-endorsed
standard for urban/suburban/rural classification: fixed, absolute
population-density thresholds (50/300/1500 people/km2) with contiguity and
cluster-size rules, computed identically everywhere on Earth - not ranked
against this project's own tiles.

Source: GHSL R2023A, epoch E2025, 4326/30ss (~1km) raster, downloaded from
JRC's open data FTP (jeodpp.jrc.ec.europa.eu), CC BY 4.0.
Reads only local cache files - no DB/bridge calls. Does not modify
tools/lte_prediction/ - this is a test-case-only alternative classifier for
comparison against the current quantile-based one.

Usage:
    python tests/baseline/compute_clutter_ghs_smod.py --project-id 210 --region taiwan
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
import rasterio
from shapely.ops import transform

from tools.lte_prediction.geo_correction_pipeline import create_analysis_grid

SMOD_LABELS = {
    30: "Urban Centre",
    23: "Dense Urban Cluster",
    22: "Semi-dense Urban Cluster",
    21: "Suburban/Peri-urban",
    13: "Rural Cluster",
    12: "Low Density Rural",
    11: "Very Low Density Rural",
    10: "Water",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify project tiles using GHS-SMOD (absolute, standard urbanisation classes)")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--tile-size-m", type=float, default=100.0)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument(
        "--smod-raster",
        type=Path,
        default=Path(r"C:\Users\PC\AppData\Local\Temp\claude\c--Users-PC-Desktop-S-Tracer-Exe-S-Tracer-Exe\bee613e1-06ad-457d-9fd7-900823b4b9ed\scratchpad\ghs_smod\GHS_SMOD_E2025_GLOBE_R2023A_4326_30ss_V2_0.tif"),
    )
    args = parser.parse_args(argv)

    data_dir = args.data_root / f"project_{args.project_id}_{args.region}"
    poly_gdf = gpd.read_file(data_dir / "project_polygon.geojson")
    poly = poly_gdf.geometry.iloc[0]
    poly_lonlat = transform(lambda x, y: (y, x), poly)
    mask_gdf = gpd.GeoDataFrame({"geometry": [poly_lonlat]}, crs="EPSG:4326")

    grid_gdf = create_analysis_grid(mask_gdf, args.tile_size_m)
    print(f"[SMOD] tiles ({args.tile_size_m:.0f}m): {len(grid_gdf)}")

    centroids = grid_gdf.geometry.centroid
    coords = [(pt.x, pt.y) for pt in centroids]

    with rasterio.open(args.smod_raster) as src:
        print(f"[SMOD] raster crs={src.crs} bounds={src.bounds}")
        sampled = list(src.sample(coords))
    codes = np.array([int(v[0]) for v in sampled])

    grid_gdf["smod_code"] = codes
    grid_gdf["smod_class"] = [SMOD_LABELS.get(c, f"Unknown({c})") for c in codes]

    counts = grid_gdf["smod_class"].value_counts()
    pct = (grid_gdf["smod_class"].value_counts(normalize=True) * 100).round(1)

    print()
    print("GHS-SMOD CLASSIFICATION (fixed absolute thresholds, local raster, project polygon):")
    for cls in counts.index:
        print(f"  {cls:<26} {counts[cls]:>4} tiles  ({pct[cls]:.1f}%)")

    # Grouped into the same broad tiers used by the current classifier, for direct comparison
    def group(cls):
        if cls in ("Urban Centre", "Dense Urban Cluster"):
            return "Dense Urban"
        if cls == "Semi-dense Urban Cluster":
            return "Urban"
        if cls == "Suburban/Peri-urban":
            return "Suburban"
        if cls == "Water":
            return "Water"
        return "Rural/Open"

    grouped = grid_gdf["smod_class"].map(group)
    gcounts = grouped.value_counts()
    gpct = (grouped.value_counts(normalize=True) * 100).round(1)
    print()
    print("Grouped to match current clutter_class tiers:")
    for cls in gcounts.index:
        print(f"  {cls:<12} {gcounts[cls]:>4} tiles  ({gpct[cls]:.1f}%)")

    out_path = data_dir / "clutter_tiles_ghs_smod.geojson"
    grid_gdf.to_file(out_path, driver="GeoJSON")
    print(f"\n[SMOD] wrote {out_path}")


if __name__ == "__main__":
    main()
