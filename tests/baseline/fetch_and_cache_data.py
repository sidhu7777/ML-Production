"""
Fetches one project's real input data (site/drive/building/polygon) via the
REAL production functions (tools.lte_prediction.ml_engine - read-only calls,
nothing written back to any DB) and caches it to local files under
tests/baseline/data/project_<id>_<region>/.

Once cached, test scripts should load from these files instead of calling
fetch_site_data/fetch_drive_data/fetch_building_data/_load_project_polygons
directly - zero DB/bridge calls needed during iteration, and nothing here
ever touches production code or writes to any database.

Usage:
    python tests/baseline/fetch_and_cache_data.py --project-id 210 --region taiwan --polygon-ids 1883
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import geopandas as gpd
from sqlalchemy import create_engine, text

from tools.lte_prediction.ml_engine import (
    engine,
    fetch_site_data,
    fetch_drive_data,
    fetch_building_data,
    _load_project_polygons,
)


def resolve_session_ids(project_id: int, region: str) -> list[int]:
    db_url = os.getenv("DATABASE_URL_Taiwan") if region == "taiwan" else os.getenv("DATABASE_URL")
    eng = create_engine(db_url)
    with eng.connect() as conn:
        ref = conn.execute(text("SELECT ref_session_id FROM tbl_project WHERE id=:pid"), {"pid": project_id}).scalar()
    return [int(s.strip()) for s in (ref or "").split(",") if s.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fetch and cache one project's baseline input data locally")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--polygon-ids", type=str, default=None)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--session-ids", type=int, nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args(argv)

    out_dir = args.output_root / f"project_{args.project_id}_{args.region}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CACHE] writing to {out_dir}")

    session_ids = args.session_ids or resolve_session_ids(args.project_id, args.region)
    print(f"[CACHE] session_ids ({len(session_ids)}): {session_ids[:5]}{'...' if len(session_ids) > 5 else ''}")

    print("[CACHE] fetching site_df ...")
    site_df, resolved_operator = fetch_site_data(
        args.project_id, region=args.region, operator=args.operator, polygon_ids=args.polygon_ids,
    )
    site_df.to_csv(out_dir / "site_df.csv", index=False)
    print(f"[CACHE] site_df: {len(site_df)} rows, operator={resolved_operator}")

    print("[CACHE] fetching drive_df ...")
    drive_df = fetch_drive_data(session_ids, resolved_operator, args.project_id, region=args.region)
    drive_df.to_csv(out_dir / "drive_df.csv", index=False)
    print(f"[CACHE] drive_df: {len(drive_df)} rows")

    print("[CACHE] fetching building_df ...")
    building_df = fetch_building_data(args.project_id, region=args.region)
    building_df.to_csv(out_dir / "building_df.csv", index=False)
    print(f"[CACHE] building_df: {len(building_df)} rows")

    print("[CACHE] fetching project polygon ...")
    current_engine = engine.get(args.region.lower(), engine["india"])
    polygons = _load_project_polygons(args.project_id, current_engine, args.region)
    polygon_gdf = gpd.GeoDataFrame({"geometry": polygons}, crs="EPSG:4326")
    if not polygon_gdf.empty:
        polygon_gdf.to_file(out_dir / "project_polygon.geojson", driver="GeoJSON")
    print(f"[CACHE] project_polygon: {len(polygon_gdf)} polygon(s)")

    meta = {
        "project_id": args.project_id,
        "region": args.region,
        "operator": resolved_operator,
        "polygon_ids": args.polygon_ids,
        "session_ids": session_ids,
        "rows": {
            "site_df": len(site_df),
            "drive_df": len(drive_df),
            "building_df": len(building_df),
            "polygons": len(polygon_gdf),
        },
        "note": (
            "DEM is NOT duplicated here - it already has its own local cache with "
            "validity checks at ML/data/dem/project_<id>_dem.tif via ensure_project_dem()."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[CACHE] wrote meta.json")
    print(f"[CACHE] done: {out_dir}")


if __name__ == "__main__":
    main()
