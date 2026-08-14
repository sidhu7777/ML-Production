from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from tests.baseline.lte_rf_debug_lab import DEFAULT_GRID_RESOLUTION_M, DEFAULT_PROJECT_ID, DEFAULT_RADIUS_M, DEFAULT_REGION, DEFAULT_WORKERS, _write_json
from tools.lte_prediction_optimised import ml_engine as opt_ml


OUTPUT_ROOT = Path("tests/output")


@dataclass
class OptimizationTestConfig:
    project_id: int = DEFAULT_PROJECT_ID
    region: str = DEFAULT_REGION
    baseline_job_id: Optional[str] = None
    radius_m: float = DEFAULT_RADIUS_M
    grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M
    workers: int = DEFAULT_WORKERS
    target_type: str = "site"
    target_id: str = ""
    impact_radius_m: float = 1200.0
    neighbor_site_count: int = 3
    max_interference_sites: int = 10
    delta_lat: float = 0.0
    delta_lon: float = 0.0
    delta_azimuth: float = 0.0
    delta_electrical_tilt: float = 0.0
    delta_mechanical_tilt: float = 0.0
    delta_tx_power: float = 0.0
    delta_antenna_height: float = 0.0
    output_root: Path = OUTPUT_ROOT


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_engine(region: str):
    current_engine = opt_ml.engine.get(region.lower(), opt_ml.engine["india"])
    if current_engine is None:
        raise RuntimeError(f"No DB engine configured for region={region}")
    return current_engine


def _fetch_latest_baseline_job_id(project_id: int, region: str) -> str:
    current_engine = _resolve_engine(region)
    query = f"""
    SELECT job_id
    FROM lte_prediction_baseline_results
    WHERE project_id = {project_id}
    ORDER BY created_at DESC
    LIMIT 1
    """
    rows = pd.read_sql(query, current_engine)
    if rows.empty:
        raise FileNotFoundError(f"No baseline results found for project_id={project_id}")
    return str(rows.loc[0, "job_id"])


def _fetch_baseline_results(project_id: int, region: str, baseline_job_id: str) -> pd.DataFrame:
    # Baseline is now stored as current-state upsert rows, not a per-job snapshot.
    # Keep the reference job id for metadata only, but read the same project-wide
    # current state as production optimization does.
    df = opt_ml.fetch_baseline(project_id, region=region)
    if df.empty:
        raise FileNotFoundError(f"No baseline rows found for project_id={project_id}")
    return df


def _normalize_site_df(site_df: pd.DataFrame) -> pd.DataFrame:
    return opt_ml._normalize_site_df(site_df, log_stage="OPT_TEST_INPUT")


def _apply_site_changes(site_df: pd.DataFrame, config: OptimizationTestConfig) -> pd.DataFrame:
    return opt_ml.build_runtime_optimized_sites(
        site_df,
        {
            "target_type": config.target_type,
            "target_id": config.target_id,
            "delta_lat": config.delta_lat,
            "delta_lon": config.delta_lon,
            "delta_azimuth": config.delta_azimuth,
            "delta_electrical_tilt": config.delta_electrical_tilt,
            "delta_mechanical_tilt": config.delta_mechanical_tilt,
            "delta_tx_power": config.delta_tx_power,
            "delta_antenna_height": config.delta_antenna_height,
        },
    )


def _compute_affected_cells(
    site_df: pd.DataFrame,
    impact_radius_m: float,
    neighbor_site_count: int,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    return opt_ml._compute_affected_cells(site_df, impact_radius_m, neighbor_site_count)


def _compute_k1k2_for_cells(baseline_df: pd.DataFrame, site_df: pd.DataFrame, affected_cells: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    return opt_ml.compute_k1k2_for_cells(baseline_df, site_df, affected_cells)


def _run_affected_prediction(
    full_site_df: pd.DataFrame,
    affected_cells: Sequence[str],
    k1k2_map: Dict[str, Tuple[float, float]],
    config: OptimizationTestConfig,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    start = time.perf_counter()
    optimized_df = opt_ml.run_prediction_only_optimized(
        full_site_df,
        k1k2_map,
        {
            "project_id": int(config.project_id),
            "region": config.region,
            "radius": float(config.radius_m),
            "grid_resolution": float(config.grid_resolution_m),
            "n_workers": int(config.workers),
            "impact_radius_m": float(config.impact_radius_m),
            "neighbor_site_count": int(config.neighbor_site_count),
            "max_interference_sites": int(config.max_interference_sites),
        },
    )
    if optimized_df.empty:
        raise RuntimeError("No affected cells produced optimization predictions")

    timings: List[Dict[str, object]] = []
    for cid in sorted(set(str(x) for x in affected_cells)):
        cell_df = optimized_df[optimized_df["Node_Cell_ID"].astype(str) == cid].copy()
        if cell_df.empty:
            continue
        k1, k2 = k1k2_map.get(str(cid), (0.0, 0.0))
        timings.append(
            {
                "cell_id": str(cid),
                "grid_points": int(len(cell_df)),
                "elapsed_sec": None,
                "k1": round(float(k1), 6),
                "k2": round(float(k2), 6),
                "interference_site_rows": None,
            }
        )
    print(
        f"[OPT_TEST][RUN_DONE] optimized_rows={len(optimized_df)} "
        f"affected_cells={len(affected_cells)} total_elapsed_sec={round(time.perf_counter() - start, 4)}"
    )
    return optimized_df, timings


def _replace_cells_in_baseline(baseline_df: pd.DataFrame, optimized_df: pd.DataFrame) -> pd.DataFrame:
    return opt_ml.replace_cells(baseline_df, optimized_df)


def run_optimization_test(config: OptimizationTestConfig) -> Path:
    start = time.perf_counter()
    run_dir = _ensure_dir(config.output_root / f"project_{config.project_id}" / f"optimization_{_timestamp()}")
    baseline_job_id = config.baseline_job_id or _fetch_latest_baseline_job_id(config.project_id, config.region)
    print(
        f"[OPT_TEST][START] project_id={config.project_id} region={config.region} "
        f"baseline_job_id={baseline_job_id} target_type={config.target_type} target_id={config.target_id}"
    )
    baseline_df = _fetch_baseline_results(config.project_id, config.region, baseline_job_id)
    site_df = _normalize_site_df(opt_ml.fetch_site_data(config.project_id, region=config.region))
    modified_site_df = _apply_site_changes(site_df, config)
    affected_cells, affected_sites, changed_rows = _compute_affected_cells(
        modified_site_df,
        config.impact_radius_m,
        config.neighbor_site_count,
    )
    print(
        f"[OPT_TEST][AFFECTED] changed_cell_count={changed_rows['Node_Cell_ID'].nunique()} "
        f"affected_site_count={len(affected_sites)} affected_cell_count={len(affected_cells)} "
        f"impact_radius_m={config.impact_radius_m} neighbor_site_count={config.neighbor_site_count}"
    )
    calibration_cells = sorted(changed_rows["Node_Cell_ID"].astype(str).unique().tolist())
    print(
        f"[OPT_TEST][K1K2_LOCAL_SCOPE] changed_cells={len(calibration_cells)} "
        f"affected_cells={len(affected_cells)} calibration_cells={calibration_cells}"
    )
    k1k2_map = _compute_k1k2_for_cells(baseline_df, modified_site_df, calibration_cells)
    optimized_df, timings = _run_affected_prediction(modified_site_df, affected_cells, k1k2_map, config)
    merged_df = _replace_cells_in_baseline(baseline_df, optimized_df)

    baseline_df.to_parquet(run_dir / "baseline_smoothed_latest.parquet", index=False)
    modified_site_df.to_csv(run_dir / "site_after.csv", index=False)
    site_df.to_csv(run_dir / "site_before.csv", index=False)
    changed_rows.to_csv(run_dir / "site_changed_rows.csv", index=False)
    optimized_df.to_parquet(run_dir / "optimized_affected_predictions.parquet", index=False)
    merged_df.to_parquet(run_dir / "optimized_merged_predictions.parquet", index=False)
    pd.DataFrame(timings).to_csv(run_dir / "latency_log.csv", index=False)

    summary = {
        "run_type": "optimization_test",
        "project_id": int(config.project_id),
        "region": config.region,
        "baseline_job_id": baseline_job_id,
        "target_type": config.target_type,
        "target_id": str(config.target_id),
        "impact_radius_m": float(config.impact_radius_m),
        "neighbor_site_count": int(config.neighbor_site_count),
        "max_interference_sites": int(config.max_interference_sites),
        "changes": {
            "delta_lat": float(config.delta_lat),
            "delta_lon": float(config.delta_lon),
            "delta_azimuth": float(config.delta_azimuth),
            "delta_electrical_tilt": float(config.delta_electrical_tilt),
            "delta_mechanical_tilt": float(config.delta_mechanical_tilt),
            "delta_tx_power": float(config.delta_tx_power),
            "delta_antenna_height": float(config.delta_antenna_height),
        },
        "counts": {
            "baseline_rows": int(len(baseline_df)),
            "changed_rows": int(len(changed_rows)),
            "changed_cells": int(changed_rows["Node_Cell_ID"].nunique()),
            "affected_sites": int(len(affected_sites)),
            "affected_cells": int(len(affected_cells)),
            "optimized_rows": int(len(optimized_df)),
            "merged_rows": int(len(merged_df)),
            "k1k2_cells": int(len(k1k2_map)),
        },
        "affected_sites": list(affected_sites),
        "affected_cells": list(affected_cells),
        "cell_timings": timings,
        "total_runtime_sec": round(time.perf_counter() - start, 4),
        "artifacts": {
            "baseline_smoothed_latest": str(run_dir / "baseline_smoothed_latest.parquet"),
            "site_before": str(run_dir / "site_before.csv"),
            "site_after": str(run_dir / "site_after.csv"),
            "changed_rows": str(run_dir / "site_changed_rows.csv"),
            "optimized_affected_predictions": str(run_dir / "optimized_affected_predictions.parquet"),
            "optimized_merged_predictions": str(run_dir / "optimized_merged_predictions.parquet"),
            "latency_log": str(run_dir / "latency_log.csv"),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    print(f"[OPT_TEST][DONE] run_dir={run_dir} total_runtime_sec={summary['total_runtime_sec']}")
    return run_dir


def _parse_args() -> OptimizationTestConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--baseline-job-id", type=str, default=None)
    parser.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    parser.add_argument("--grid-resolution-m", type=float, default=DEFAULT_GRID_RESOLUTION_M)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--target-type", type=str, choices=["site", "cell"], default="site")
    parser.add_argument("--target-id", type=str, required=True)
    parser.add_argument("--impact-radius-m", type=float, default=1200.0)
    parser.add_argument("--neighbor-site-count", type=int, default=3)
    parser.add_argument("--max-interference-sites", type=int, default=10)
    parser.add_argument("--delta-lat", type=float, default=0.0)
    parser.add_argument("--delta-lon", type=float, default=0.0)
    parser.add_argument("--delta-azimuth", type=float, default=0.0)
    parser.add_argument("--delta-electrical-tilt", type=float, default=0.0)
    parser.add_argument("--delta-mechanical-tilt", type=float, default=0.0)
    parser.add_argument("--delta-tx-power", type=float, default=0.0)
    parser.add_argument("--delta-antenna-height", type=float, default=0.0)
    args = parser.parse_args()
    return OptimizationTestConfig(
        project_id=args.project_id,
        region=args.region,
        baseline_job_id=args.baseline_job_id,
        radius_m=args.radius_m,
        grid_resolution_m=args.grid_resolution_m,
        workers=args.workers,
        target_type=args.target_type,
        target_id=args.target_id,
        impact_radius_m=args.impact_radius_m,
        neighbor_site_count=args.neighbor_site_count,
        max_interference_sites=args.max_interference_sites,
        delta_lat=args.delta_lat,
        delta_lon=args.delta_lon,
        delta_azimuth=args.delta_azimuth,
        delta_electrical_tilt=args.delta_electrical_tilt,
        delta_mechanical_tilt=args.delta_mechanical_tilt,
        delta_tx_power=args.delta_tx_power,
        delta_antenna_height=args.delta_antenna_height,
    )


if __name__ == "__main__":
    run_dir = run_optimization_test(_parse_args())
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
