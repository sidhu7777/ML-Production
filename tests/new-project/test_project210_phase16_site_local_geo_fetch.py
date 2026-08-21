"""
Phase 16: fetch REAL geo data (roads, water, land_cover, land_use, buildings)
for a 2km-radius circle around ONE site (LA201565) only, instead of relying
on the whole-project polygon from ML/tests/baseline (which only covers
~2.4km x 4.7km and was found to run out of data almost exactly where the
Phase 15 boresight/left-side effective-range scan hit its 2.5km cap).

Does NOT touch or modify anything in ML/tests/baseline - imports its
functions read-only and reuses them exactly as they are. Output goes to its
own folder under ML/tests/new-project/data, separate from the baseline
project-wide clutter output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import transform

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
BASELINE_DATA_DIR = BASELINE_DIR / "data" / "project_210_taiwan"
for p in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fetch_overture_geo_context import fetch_layer  # reused read-only, not modified
from compute_clutter_final_v2 import (  # reused read-only, not modified
    attach_road_length,
    attach_area_ratio,
    attach_building_features_fixed,
    attach_surrounding_height,
    impute_building_heights,
    classify,
    GREEN_LC_SUBTYPES,
    GREEN_LU_SUBTYPES,
    GREEN_LU_CLASS,
)
from tools.lte_prediction.geo_correction_pipeline import create_analysis_grid  # reused read-only

import streamlit_project210_phase13_beam_check as phase13

SITE_ID = "LA201565"
RADIUS_M = 2000.0
TILE_SIZE_M = 25.0
OUT_DIR = THIS_DIR / "data" / "project_210_taiwan" / "site_LA201565_2km_local_geo"
OBAT_CSV_PATH = BASELINE_DATA_DIR / "ghsobat_project210_bbox.csv"


def _circle_polygon(lat: float, lon: float, radius_m: float, n_points: int = 72):
    lat_step = radius_m / 111320.0
    lon_step = radius_m / (111320.0 * max(np.cos(np.radians(lat)), 1e-6))
    angles = np.linspace(0, 2 * np.pi, n_points)
    coords = [(lon + lon_step * np.sin(a), lat + lat_step * np.cos(a)) for a in angles]
    from shapely.geometry import Polygon
    return Polygon(coords)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    identity = phase13.load_identity()
    site_row = identity[identity["site"] == SITE_ID].iloc[0]
    site_lat, site_lon = float(site_row["lat"]), float(site_row["lon"])
    print(f"[PHASE16] site={SITE_ID} lat={site_lat:.6f} lon={site_lon:.6f} radius_m={RADIUS_M}")

    poly_lonlat = _circle_polygon(site_lat, site_lon, RADIUS_M)
    bbox = poly_lonlat.bounds
    mask_gdf = gpd.GeoDataFrame({"geometry": [poly_lonlat]}, crs="EPSG:4326")

    roads = fetch_layer("segment", bbox, poly_lonlat, "roads/transportation")
    water = fetch_layer("water", bbox, poly_lonlat, "water bodies")
    land_cover = fetch_layer("land_cover", bbox, poly_lonlat, "natural land cover / green")
    land_use = fetch_layer("land_use", bbox, poly_lonlat, "land use (parks, forest, etc.)")
    buildings_raw = fetch_layer("building", bbox, poly_lonlat, "buildings")
    print(
        f"[PHASE16] fetched: roads={len(roads)} water={len(water)} "
        f"land_cover={len(land_cover)} land_use={len(land_use)} buildings={len(buildings_raw)}"
    )

    building_gdf = buildings_raw[["geometry"]].copy().reset_index(drop=True)
    building_gdf = building_gdf[building_gdf.geometry.notnull() & building_gdf.geometry.is_valid & ~building_gdf.geometry.is_empty]

    grid_gdf = create_analysis_grid(mask_gdf, TILE_SIZE_M)
    print(f"[PHASE16] grid tiles ({TILE_SIZE_M:.0f}m): {len(grid_gdf)}")

    grid_gdf = attach_building_features_fixed(grid_gdf, building_gdf)

    # GHS-OBAT height data was only ever fetched for the original small
    # project polygon, so buildings outside it will mostly fall back to
    # neighbor-imputation or the project mean - real footprints/positions
    # (from Overture, fetched fresh above) still cover the full 2km circle,
    # only the height precision degrades outside the original polygon.
    building_gdf_h = impute_building_heights(building_gdf, OBAT_CSV_PATH)
    project_mean_height = float(building_gdf_h["height"].mean())
    print(f"[PHASE16] mean building height (real + imputed) in 2km circle: {project_mean_height:.1f}m")

    surrounding_df = attach_surrounding_height(grid_gdf, building_gdf_h, radius_m=100.0)
    grid_gdf = grid_gdf.merge(surrounding_df, on="grid_id", how="left")

    grid_gdf = attach_road_length(grid_gdf, roads)
    grid_gdf = attach_area_ratio(grid_gdf, water, "water_ratio")

    green_lc = land_cover[land_cover["subtype"].isin(GREEN_LC_SUBTYPES)] if "subtype" in land_cover.columns else land_cover.iloc[0:0]
    green_lu = (
        land_use[land_use["subtype"].isin(GREEN_LU_SUBTYPES) | land_use["class"].isin(GREEN_LU_CLASS)]
        if "subtype" in land_use.columns
        else land_use.iloc[0:0]
    )
    green = gpd.GeoDataFrame(pd.concat([green_lc, green_lu], ignore_index=True), crs="EPSG:4326")
    grid_gdf = attach_area_ratio(grid_gdf, green, "green_ratio")

    grid_gdf["clutter_class"] = grid_gdf.apply(lambda r: classify(r, project_mean_height), axis=1)

    counts = grid_gdf["clutter_class"].value_counts()
    print("\n[PHASE16] clutter classification for the 2km site-local circle:")
    for cls in counts.index:
        print(f"  {cls:<12} {counts[cls]:>5} tiles")

    grid_out = OUT_DIR / "clutter_tiles_site_local_2km.geojson"
    grid_gdf.to_file(grid_out, driver="GeoJSON")
    print(f"[PHASE16] wrote {grid_out}")

    building_gdf_h["height"] = pd.to_numeric(building_gdf_h["height"], errors="coerce")
    buildings_out = OUT_DIR / "buildings_site_local_2km.geojson"
    building_gdf_h[["geometry", "height", "height_source"]].to_file(buildings_out, driver="GeoJSON")
    print(f"[PHASE16] wrote {buildings_out}")

    (OUT_DIR / "meta.json").write_text(
        pd.Series(
            {
                "site_id": SITE_ID,
                "site_lat": site_lat,
                "site_lon": site_lon,
                "radius_m": RADIUS_M,
                "tile_size_m": TILE_SIZE_M,
                "grid_tiles": int(len(grid_gdf)),
                "buildings": int(len(building_gdf_h)),
                "mean_building_height_m": project_mean_height,
            }
        ).to_json(indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
