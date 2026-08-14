"""
Runs the REAL production pipeline (ml_engine.run_rf_prediction_fast +
ml_engine.run_ml_fast, i.e. COST-231 + geo correction + DT calibration)
using LOCAL CACHED input data from tests/baseline/data/project_<id>_<region>/
(built by fetch_and_cache_data.py) instead of fetching from the DB.

Saves the resulting output as a frozen "baseline snapshot" - the reference
point every future comparison run diffs against. Nothing here writes to any
database or touches production code; it only reads local cache files and
writes local output files.

Usage:
    python tests/baseline/run_baseline_from_cache.py --project-id 210 --region taiwan
    python tests/baseline/run_baseline_from_cache.py --project-id 210 --region taiwan --label before_weight_fix
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import geopandas as gpd
import pandas as pd

from tools.lte_prediction.ml_engine import run_rf_prediction_fast, run_ml_fast


def load_cached_inputs(data_dir: Path):
    # NOTE: project polygon is intentionally NOT loaded from the cached geojson
    # and passed as polygon_wkt here. _resolve_prediction_polygons() treats an
    # explicitly-provided polygon_wkt as an "override" and skips the swapped-
    # axis auto-detection the normal DB-fetch path applies - and this project's
    # polygon needs that correction (confirmed: site alignment for this same
    # project reports alignment=swapped_xy). Passing the raw cached WKT through
    # the override path silently clipped every point to zero. So polygon
    # resolution still does its normal (fast, single-row) DB lookup; only the
    # heavy fetches (site/drive/building) are served from the local cache.
    site_df = pd.read_csv(data_dir / "site_df.csv", low_memory=False)
    drive_df = pd.read_csv(data_dir / "drive_df.csv", low_memory=False)
    building_df = pd.read_csv(data_dir / "building_df.csv", low_memory=False)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    return site_df, drive_df, building_df, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the real baseline pipeline against locally cached input data")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--region", type=str, default="taiwan")
    parser.add_argument("--radius-m", type=float, default=500.0)
    parser.add_argument("--grid-resolution-m", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-interference-sites", type=int, default=10)
    parser.add_argument("--tile-size-m", type=float, default=100.0)
    parser.add_argument("--cluster-count", type=int, default=5)
    parser.add_argument("--enable-osm", action="store_true")
    parser.add_argument("--dem-raster-path", type=str, default=None)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--snapshot-root", type=Path, default=Path(__file__).parent / "data" / "snapshots")
    parser.add_argument("--label", type=str, default="baseline", help="Name for this snapshot, e.g. 'before_weight_fix', 'after_weight_fix'")
    args = parser.parse_args(argv)

    data_dir = args.data_root / f"project_{args.project_id}_{args.region}"
    if not data_dir.exists():
        raise SystemExit(f"No cached data at {data_dir} - run fetch_and_cache_data.py first")

    print(f"[SNAPSHOT] loading cached inputs from {data_dir}")
    site_df, drive_df, building_df, meta = load_cached_inputs(data_dir)
    print(f"[SNAPSHOT] site_df={len(site_df)} drive_df={len(drive_df)} building_df={len(building_df)} "
          f"(project polygon resolved live via a lightweight DB lookup - see load_cached_inputs note)")

    dem_raster_path = args.dem_raster_path
    if dem_raster_path is None:
        candidate = PROJECT_ROOT / "data" / "dem" / f"project_{args.project_id}_dem.tif"
        if candidate.exists():
            dem_raster_path = str(candidate)
    print(f"[SNAPSHOT] dem_raster_path={dem_raster_path}")

    start = time.perf_counter()
    rf_params = {
        "project_id": args.project_id,
        "region": args.region,
        "radius": args.radius_m,
        "grid": args.grid_resolution_m,
        "workers": args.workers,
        "max_interference_sites": args.max_interference_sites,
        "use_frontend_grid_sampling": False,  # deterministic circular grid, same every run against this frozen cache
    }
    pred_df = run_rf_prediction_fast(site_df, drive_df, building_df, rf_params)
    print(f"[SNAPSHOT] COST-231 done rows={len(pred_df)} elapsed={time.perf_counter() - start:.1f}s")

    step = time.perf_counter()
    ml_params = {
        "project_id": args.project_id,
        "region": args.region,
        "grid": args.grid_resolution_m,
        "tile_size_m": args.tile_size_m,
        "cluster_count": args.cluster_count,
        "dem_raster_path": dem_raster_path,
        "enable_osm_enrichment": args.enable_osm,
        "dt_validation_fraction": 0.25,
    }
    final_df = run_ml_fast(pred_df, drive_df, site_df=site_df, building_df=building_df, params=ml_params)
    production_summary = dict(final_df.attrs.get("production_summary") or {})
    print(f"[SNAPSHOT] geo correction + DT calibration done rows={len(final_df)} elapsed={time.perf_counter() - step:.1f}s")

    snapshot_dir = args.snapshot_root / f"project_{args.project_id}_{args.region}" / args.label
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(snapshot_dir / "final_df.parquet", index=False)
    final_df.to_csv(snapshot_dir / "final_df_sample.csv", index=False) if len(final_df) <= 50000 else \
        final_df.iloc[::max(1, len(final_df) // 50000)].to_csv(snapshot_dir / "final_df_sample.csv", index=False)

    clutter_dist = final_df["clutter_class"].value_counts(dropna=False).to_dict() if "clutter_class" in final_df.columns else {}
    morph_dist = final_df["morphology_cluster"].value_counts(dropna=False).to_dict() if "morphology_cluster" in final_df.columns else {}

    summary = {
        "project_id": args.project_id,
        "region": args.region,
        "label": args.label,
        "params": {k: v for k, v in vars(args).items() if k not in ("data_root", "snapshot_root")},
        "input_meta": meta,
        "rows": len(final_df),
        "production_summary": production_summary,
        "clutter_distribution": clutter_dist,
        "morphology_distribution": {str(k): v for k, v in morph_dist.items()},
        "rsrp_stats": {
            "min": float(final_df["pred_rsrp"].min()) if "pred_rsrp" in final_df.columns else None,
            "max": float(final_df["pred_rsrp"].max()) if "pred_rsrp" in final_df.columns else None,
            "mean": float(final_df["pred_rsrp"].mean()) if "pred_rsrp" in final_df.columns else None,
            "std": float(final_df["pred_rsrp"].std()) if "pred_rsrp" in final_df.columns else None,
        },
        "total_runtime_sec": round(time.perf_counter() - start, 2),
    }
    (snapshot_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"[SNAPSHOT] clutter_distribution={clutter_dist}")
    print(f"[SNAPSHOT] rsrp_stats={summary['rsrp_stats']}")
    print(f"[SNAPSHOT] saved to {snapshot_dir}")


if __name__ == "__main__":
    main()
