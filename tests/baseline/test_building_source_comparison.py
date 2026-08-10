"""
Standalone comparison: OSM vs Overture Maps building footprint coverage for
a project's polygon. Does NOT touch any production code (ml_engine.py /
geo_correction_pipeline.py are only imported read-only, for the existing
project-polygon loader and the same axis-order-safe insert pattern already
used by fetch_building_data's OSM cache-fill).

Reports what's currently in tbl_savepolygon, fetches Overture's building
layer for the same area, clips it to the real project polygon, and prints
a side-by-side comparison. Only pushes Overture buildings into the DB if
you pass --push, and only inserts - never deletes/touches the existing OSM
rows, so both sources stay visible and comparable in the DB afterward.

Usage:
    python tests/baseline/test_building_source_comparison.py --project-id 210 --region taiwan --polygon-ids 1883
    python tests/baseline/test_building_source_comparison.py --project-id 210 --region taiwan --polygon-ids 1883 --push
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
from dotenv import load_dotenv
from shapely.ops import transform

load_dotenv(PROJECT_ROOT / ".env")

from tools.lte_prediction.ml_engine import engine, _load_project_polygons, _swap_lon_lat_if_needed
from sqlalchemy import text


def report_current_db_state(current_engine, project_id: int):
    print("=" * 70)
    print(f"STEP 1: current tbl_savepolygon state for project_id={project_id}")
    print("=" * 70)
    with current_engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT name, COUNT(*) c FROM tbl_savepolygon WHERE project_id=:pid GROUP BY name"
        ), {"pid": project_id}).fetchall()
    if not rows:
        print("  (no rows for this project yet)")
    for name, count in rows:
        print(f"  name={name!r:25s} rows={count}")
    return dict(rows)


def fetch_overture_buildings(polygon):
    print()
    print("=" * 70)
    print("STEP 2: fetching Overture Maps buildings for this area")
    print("=" * 70)
    import overturemaps.core as core

    minx, miny, maxx, maxy = polygon.bounds
    print(f"  bbox={minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}  (querying, this can take 1-2 min)")
    gdf = core.geodataframe("building", bbox=(minx, miny, maxx, maxy))
    print(f"  fetched {len(gdf)} buildings in bbox (before clipping to actual polygon)")

    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    clipped = gdf[gdf.geometry.intersects(polygon)].copy()
    print(f"  {len(clipped)} buildings fall inside the actual project polygon")
    return clipped


def compare_coverage(osm_count: int, overture_gdf) -> dict:
    print()
    print("=" * 70)
    print("STEP 3: comparison")
    print("=" * 70)
    overture_count = len(overture_gdf)
    print(f"  {'Source':<12} {'Buildings':>10}")
    print(f"  {'-'*12} {'-'*10}")
    print(f"  {'OSM':<12} {osm_count:>10}")
    print(f"  {'Overture':<12} {overture_count:>10}")
    if osm_count > 0:
        print(f"  Overture has {overture_count / osm_count:.1f}x the building count OSM has for this area.")
    with_height = int(overture_gdf["height"].notna().sum()) if "height" in overture_gdf.columns else 0
    print(f"  Overture buildings with real height data: {with_height}/{overture_count}")
    return {"osm_count": osm_count, "overture_count": overture_count, "with_height": with_height}


def push_overture_to_db(current_engine, project_id: int, overture_gdf):
    print()
    print("=" * 70)
    print("STEP 4: pushing Overture buildings into tbl_savepolygon")
    print("=" * 70)
    if overture_gdf.empty:
        print("  nothing to push")
        return 0

    exploded = overture_gdf.explode(index_parts=True, ignore_index=True)
    # Same MySQL SRID-4326 axis-order fix already applied to the OSM path:
    # shapely/Overture geometries are (lon, lat); ST_GeomFromText(..., 4326)
    # needs (lat, lon) for this SRID.
    exploded["wkt_latlon"] = exploded.geometry.apply(
        lambda geom: transform(lambda x, y: (y, x), geom).wkt
    )
    exploded["calc_area"] = exploded.geometry.area
    area_name = f"overture_auto_{project_id}"

    values_list = [
        (area_name, row.wkt_latlon, int(project_id), float(row.calc_area))
        for row in exploded.itertuples()
    ]

    inserted = 0
    raw_conn = current_engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        cursor.execute("SET autocommit=0")
        batch_size = 1000
        for i in range(0, len(values_list), batch_size):
            batch = values_list[i:i + batch_size]
            placeholders = "(%s, ST_GeomFromText(%s, 4326), %s, %s)"
            values_str = ", ".join([placeholders] * len(batch))
            cursor.execute(
                f"INSERT INTO tbl_savepolygon (name, region, project_id, area) VALUES {values_str}",
                [item for row_vals in batch for item in row_vals],
            )
            inserted += len(batch)
        raw_conn.commit()
        cursor.close()
    finally:
        raw_conn.close()

    print(f"  inserted {inserted} rows tagged name={area_name!r}")
    return inserted


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare OSM vs Overture Maps building coverage for a project")
    parser.add_argument("--project-id", type=int, default=210)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--polygon-ids", type=str, default=None)
    parser.add_argument("--push", action="store_true", help="Insert Overture buildings into tbl_savepolygon (tagged overture_auto_<project_id>) if it has better coverage than OSM")
    args = parser.parse_args(argv)

    current_engine = engine.get(args.region.lower(), engine["india"])

    existing = report_current_db_state(current_engine, args.project_id)
    osm_count = existing.get(f"osm_auto_{args.project_id}", 0)

    polygons = _load_project_polygons(args.project_id, current_engine, args.region)
    if not polygons:
        print("No project polygon found - cannot fetch Overture buildings for this project.")
        return
    polygon = _swap_lon_lat_if_needed(polygons[0])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    overture_gdf = fetch_overture_buildings(polygon)
    stats = compare_coverage(osm_count, overture_gdf)

    already_pushed = f"overture_auto_{args.project_id}" in existing
    if args.push:
        if already_pushed:
            print(f"\noverture_auto_{args.project_id} rows already exist ({existing[f'overture_auto_{args.project_id}']}) - not re-inserting. Delete them first if you want a fresh push.")
        elif stats["overture_count"] > stats["osm_count"]:
            push_overture_to_db(current_engine, args.project_id, overture_gdf)
        else:
            print("\nOverture did not have more coverage than OSM for this area - not pushing.")
    else:
        print("\n(dry run - pass --push to actually insert Overture buildings into tbl_savepolygon)")


if __name__ == "__main__":
    main()
