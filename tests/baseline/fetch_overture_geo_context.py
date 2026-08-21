"""
Fetches Overture Maps geo-context layers (roads, water, green/vegetation) for
one project's real polygon and caches them to local files under
tests/baseline/data/project_<id>_<region>/, alongside the existing
site/drive/building/polygon cache built by fetch_and_cache_data.py.

Building footprints already come from Overture (pushed to tbl_savepolygon
separately - see building_df.csv in the same cache dir). This script covers
the other geo-context inputs that production's OSM-based
_attach_osm_context_features() currently computes: road_length_m (from
Overture "segment"), water_ratio (from Overture "water"), green_ratio (from
Overture "land_cover"/"land_use"). Read-only against Overture's public S3
dataset - touches no DB, no production code.

Usage:
    python tests/baseline/fetch_overture_geo_context.py --project-id 210 --region taiwan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import overturemaps.core as core


def clip_to_polygon(gdf: gpd.GeoDataFrame, polygon) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    # Bug fix: this used to only *filter* to features whose geometry.intersects()
    # the project polygon, keeping each matched feature's FULL original geometry.
    # For land_cover/land_use, Overture ships large multi-part features (a single
    # feature id can span a huge area); one part touching the small project bbox
    # was enough to keep the whole untrimmed shape, which for this project (210,
    # Taiwan) pulled in "green" polygons spanning almost the entire island. A real
    # geometric clip (gpd.clip) trims every feature down to just the portion that
    # actually falls inside the project polygon, same as water/roads always needed.
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    clipped = gpd.clip(gdf, polygon)
    return clipped[clipped.geometry.notnull() & ~clipped.geometry.is_empty].copy()


def fetch_layer(overture_type: str, bbox, polygon, label: str) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox
    print(f"[GEO_CONTEXT] fetching Overture '{overture_type}' ({label}) in bbox, then clipping to true polygon ...")
    gdf = core.geodataframe(overture_type, bbox=(minx, miny, maxx, maxy))
    print(f"[GEO_CONTEXT]   {len(gdf)} features in bbox")
    clipped = clip_to_polygon(gdf, polygon)
    print(f"[GEO_CONTEXT]   {len(clipped)} features inside the true project polygon")
    return clipped


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fetch and cache Overture road/water/green geo-context layers for a project")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args(argv)

    data_dir = args.data_root / f"project_{args.project_id}_{args.region}"
    polygon_path = data_dir / "project_polygon.geojson"
    if not polygon_path.exists():
        raise SystemExit(f"No cached project polygon at {polygon_path} - run fetch_and_cache_data.py first")

    poly_gdf = gpd.read_file(polygon_path)
    poly = poly_gdf.geometry.iloc[0]
    # project_polygon.geojson is stored with swapped axes (x=lat, y=lon) - same
    # convention as project_polygon everywhere else in this cache. Overture's
    # own coordinates are real lon/lat, so build a lon/lat version to query/clip with.
    from shapely.ops import transform
    poly_lonlat = transform(lambda x, y: (y, x), poly)
    bbox = poly_lonlat.bounds
    print(f"[GEO_CONTEXT] project polygon bbox (lon,lat): {bbox}")

    roads = fetch_layer("segment", bbox, poly_lonlat, "roads/transportation")
    water = fetch_layer("water", bbox, poly_lonlat, "water bodies")
    land_cover = fetch_layer("land_cover", bbox, poly_lonlat, "natural land cover / green")
    land_use = fetch_layer("land_use", bbox, poly_lonlat, "land use (parks, forest, etc.)")

    out = {
        "roads_segment.geojson": roads,
        "water.geojson": water,
        "land_cover.geojson": land_cover,
        "land_use.geojson": land_use,
    }
    counts = {}
    for filename, gdf in out.items():
        path = data_dir / filename
        if gdf.empty:
            print(f"[GEO_CONTEXT] {filename}: empty, not writing a file")
            counts[filename] = 0
            continue
        # geoparquet-unfriendly nested/struct columns (sources, names) break
        # plain GeoJSON writers sometimes - keep only geometry + simple scalar
        # columns needed downstream, drop the rest.
        keep_cols = [c for c in gdf.columns if c == "geometry" or gdf[c].apply(lambda v: isinstance(v, (str, int, float, type(None)))).all()]
        trimmed = gdf[keep_cols]
        if trimmed.crs is None:
            trimmed = trimmed.set_crs("EPSG:4326")
        trimmed.to_file(path, driver="GeoJSON")
        counts[filename] = len(gdf)
        print(f"[GEO_CONTEXT] wrote {filename}: {len(gdf)} features")

    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["overture_geo_context"] = counts
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[GEO_CONTEXT] updated meta.json: {counts}")


if __name__ == "__main__":
    main()
