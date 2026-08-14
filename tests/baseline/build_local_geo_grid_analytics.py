from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.baseline.lte_rf_debug_lab import _write_json


DEFAULT_RUN_DIR = PROJECT_ROOT / "tests" / "output_nocache_20260528_234540" / "project_196" / "20260528_234545"
DEFAULT_REFERENCE_GRID = PROJECT_ROOT / "tests" / "output" / "project_196" / "tilt_rsrp_only_20260528_164338" / "baseline_grid_metrics.csv"


def _load_prediction_grid(run_dir: Path) -> pd.DataFrame:
    parquet_path = run_dir / "rf_prediction_grid.parquet"
    csv_path = run_dir / "rf_prediction_grid_full.csv"
    if parquet_path.exists():
        print(f"[LOCAL_GRID] loading_prediction={parquet_path}")
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        print(f"[LOCAL_GRID] loading_prediction={csv_path}")
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No prediction grid found under {run_dir}")


def _grid_row_col(grid_id: object) -> tuple[float, float]:
    match = re.match(r"^R(\d+)C(\d+)$", str(grid_id).strip())
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def _assign_reference_grid_ids(pred_df: pd.DataFrame, reference_grid_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    ref = reference_grid_df.copy()
    required_ref = {"grid_id", "min_lat", "max_lat", "min_lon", "max_lon"}
    if ref.empty or not required_ref.issubset(ref.columns):
        return out

    rc = ref["grid_id"].map(_grid_row_col)
    ref["_row"] = [value[0] for value in rc]
    ref["_col"] = [value[1] for value in rc]
    ref = ref.dropna(subset=["_row", "_col", "min_lat", "max_lat", "min_lon", "max_lon"]).copy()
    if ref.empty:
        return out

    row_bounds = (
        ref.groupby("_row", as_index=False)
        .agg(min_lat=("min_lat", "min"), max_lat=("max_lat", "max"))
        .sort_values("_row")
        .reset_index(drop=True)
    )
    col_bounds = (
        ref.groupby("_col", as_index=False)
        .agg(min_lon=("min_lon", "min"), max_lon=("max_lon", "max"))
        .sort_values("_col")
        .reset_index(drop=True)
    )
    lat = pd.to_numeric(out["lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(out["lon"], errors="coerce").to_numpy(dtype=float)
    row_idx = np.searchsorted(row_bounds["max_lat"].to_numpy(dtype=float), lat, side="left")
    col_idx = np.searchsorted(col_bounds["max_lon"].to_numpy(dtype=float), lon, side="left")
    valid = (
        (row_idx >= 0)
        & (row_idx < len(row_bounds))
        & (col_idx >= 0)
        & (col_idx < len(col_bounds))
    )
    assigned = np.full(len(out), pd.NA, dtype=object)
    if valid.any():
        row_min = row_bounds["min_lat"].to_numpy(dtype=float)[row_idx[valid]]
        row_max = row_bounds["max_lat"].to_numpy(dtype=float)[row_idx[valid]]
        col_min = col_bounds["min_lon"].to_numpy(dtype=float)[col_idx[valid]]
        col_max = col_bounds["max_lon"].to_numpy(dtype=float)[col_idx[valid]]
        inside = (lat[valid] >= row_min) & (lat[valid] <= row_max) & (lon[valid] >= col_min) & (lon[valid] <= col_max)
        valid_positions = np.flatnonzero(valid)
        inside_positions = valid_positions[inside]
        row_values = row_bounds["_row"].to_numpy(dtype=int)[row_idx[inside_positions]]
        col_values = col_bounds["_col"].to_numpy(dtype=int)[col_idx[inside_positions]]
        assigned[inside_positions] = [f"R{r}C{c}" for r, c in zip(row_values, col_values)]

    valid_grid_ids = set(ref["grid_id"].astype(str))
    assigned_series = pd.Series(assigned, index=out.index, dtype="string")
    assigned_series = assigned_series.where(assigned_series.isin(valid_grid_ids))
    out["grid_id"] = assigned_series
    print(
        f"[LOCAL_GRID][REFERENCE_ASSIGN] reference_grids={len(reference_grid_df)} "
        f"points={len(out)} assigned={int(out['grid_id'].notna().sum())}"
    )
    return out


def _build_local_grid_analytics(pred_df: pd.DataFrame, rsrp_threshold: float, reference_grid_df: pd.DataFrame) -> pd.DataFrame:
    required = {"lat", "lon"}
    missing = sorted(required - set(pred_df.columns))
    if missing:
        raise ValueError(f"Prediction artifact is missing required columns: {missing}")
    work = _assign_reference_grid_ids(pred_df, reference_grid_df) if not reference_grid_df.empty else pred_df.copy()
    if "grid_id" not in work.columns:
        raise ValueError("Prediction artifact/reference mapping did not produce grid_id")
    work["grid_id"] = work["grid_id"].astype("string").str.strip()
    work = work.loc[work["grid_id"].notna() & ~work["grid_id"].isin(["", "nan", "NaN", "None", "<NA>"])].copy()
    if work.empty:
        raise ValueError("Prediction artifact has no mapped grid_id rows")

    metric_map = {
        "pred_rsrp_geo": "baseline_avg_rsrp",
        "pred_rsrq_geo": "baseline_avg_rsrq",
        "pred_sinr_geo": "baseline_avg_sinr",
    }
    for source_col, target_col in metric_map.items():
        if source_col not in work.columns:
            fallback = source_col.replace("_geo", "")
            if fallback not in work.columns:
                raise ValueError(f"Prediction artifact is missing {source_col} and fallback {fallback}")
            source_col = fallback
        work[target_col] = pd.to_numeric(work[source_col], errors="coerce")

    for col in ["lat", "lon"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    agg = (
        work.groupby("grid_id", dropna=False)
        .agg(
            center_lat=("lat", "mean"),
            center_lon=("lon", "mean"),
            min_lat=("lat", "min"),
            max_lat=("lat", "max"),
            min_lon=("lon", "min"),
            max_lon=("lon", "max"),
            baseline_point_count=("baseline_avg_rsrp", "count"),
            baseline_avg_rsrp=("baseline_avg_rsrp", "mean"),
            baseline_avg_rsrq=("baseline_avg_rsrq", "mean"),
            baseline_avg_sinr=("baseline_avg_sinr", "mean"),
            distinct_node_cell_id=("Node_Cell_ID", "nunique") if "Node_Cell_ID" in work.columns else ("grid_id", "size"),
        )
        .reset_index()
    )
    if not reference_grid_df.empty:
        ref_cols = [
            col
            for col in ["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]
            if col in reference_grid_df.columns
        ]
        ref = reference_grid_df[ref_cols].drop_duplicates(subset=["grid_id"], keep="first").copy()
        agg = ref.merge(
            agg.drop(columns=[col for col in ["center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"] if col in agg.columns]),
            on="grid_id",
            how="left",
        )
        for col in ["baseline_point_count", "distinct_node_cell_id"]:
            if col in agg.columns:
                agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0).astype(int)
    agg["is_bad_rsrp"] = pd.to_numeric(agg["baseline_avg_rsrp"], errors="coerce") < float(rsrp_threshold)
    agg["rsrp_severity"] = (float(rsrp_threshold) - pd.to_numeric(agg["baseline_avg_rsrp"], errors="coerce")).clip(lower=0.0)
    for threshold in [-90.0, -95.0, -100.0, -105.0]:
        suffix = str(abs(int(threshold)))
        agg[f"is_bad_rsrp_lt_{suffix}"] = pd.to_numeric(agg["baseline_avg_rsrp"], errors="coerce") < threshold
    return agg.sort_values(["rsrp_severity", "baseline_avg_rsrp", "grid_id"], ascending=[False, True, True]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--reference-grid-metrics", type=Path, default=DEFAULT_REFERENCE_GRID)
    parser.add_argument("--rsrp", type=float, default=-90.0)
    parser.add_argument("--output-name", type=str, default="local_grid_analytics_geo.csv")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    pred_df = _load_prediction_grid(run_dir)
    reference_grid_df = pd.read_csv(args.reference_grid_metrics) if args.reference_grid_metrics and args.reference_grid_metrics.exists() else pd.DataFrame()
    if not reference_grid_df.empty:
        print(f"[LOCAL_GRID] reference_grid_metrics={args.reference_grid_metrics.resolve()} rows={len(reference_grid_df)}")
    grid_df = _build_local_grid_analytics(pred_df, args.rsrp, reference_grid_df)
    output_path = run_dir / args.output_name
    grid_df.to_csv(output_path, index=False)
    counts = {}
    for threshold in [-90.0, -95.0, -100.0, -105.0]:
        suffix = str(abs(int(threshold)))
        counts[f"bad_grid_count_lt_{suffix}"] = int(grid_df[f"is_bad_rsrp_lt_{suffix}"].fillna(False).sum())
    summary = {
        "run_dir": str(run_dir),
        "source": "rf_prediction_grid.pred_*_geo",
        "output_csv": str(output_path),
        "grid_count": int(len(grid_df)),
        "reference_grid_count": int(len(reference_grid_df)),
        "point_count": int(len(pred_df)),
        **counts,
    }
    _write_json(run_dir / "local_grid_analytics_geo_summary.json", summary)
    print(f"[LOCAL_GRID_DONE] {json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
