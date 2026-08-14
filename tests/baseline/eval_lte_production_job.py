from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import BallTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.lte_prediction import ml_engine


def _metric_bundle(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    valid = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if valid.empty:
        return {}

    err = valid["actual"] - valid["predicted"]
    abs_err = err.abs()
    return {
        "rows": int(len(valid)),
        "mae": round(float(mean_absolute_error(valid["actual"], valid["predicted"])), 4),
        "rmse": round(float(math.sqrt(mean_squared_error(valid["actual"], valid["predicted"]))), 4),
        "r2": round(float(r2_score(valid["actual"], valid["predicted"])), 4),
        "bias": round(float(err.mean()), 4),
        "p50_abs_err": round(float(abs_err.quantile(0.50)), 4),
        "p90_abs_err": round(float(abs_err.quantile(0.90)), 4),
    }


def _resolve_operator(project_id: int, region: str) -> str:
    site_df, operator = ml_engine.fetch_site_data(project_id, region=region)
    print(
        f"[EVAL] Resolved operator={operator} "
        f"site_rows={len(site_df)} project_id={project_id} region={region}"
    )
    return operator


def _fetch_drive_df(
    session_ids: Iterable[int],
    operator: str,
    project_id: int,
    region: str,
) -> pd.DataFrame:
    return ml_engine.fetch_drive_data(session_ids, operator, project_id, region=region)


def _load_prediction_csv(job_id: str, csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        path = Path(csv_path)
    else:
        path = PROJECT_ROOT / "temp" / f"final_{job_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    print(f"[EVAL] Using prediction artifact: {path}")
    return pd.read_csv(path)


def _match_dt_to_predictions(dt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    dt = dt_df.copy()
    preds = pred_df.copy()

    dt = dt.dropna(subset=["lat", "lon"]).copy()
    preds = preds.dropna(subset=["lat", "lon"]).copy()
    if dt.empty or preds.empty:
        raise ValueError("DT or prediction frame is empty after dropping missing coordinates")

    dt["lat"] = pd.to_numeric(dt["lat"], errors="coerce")
    dt["lon"] = pd.to_numeric(dt["lon"], errors="coerce")
    preds["lat"] = pd.to_numeric(preds["lat"], errors="coerce")
    preds["lon"] = pd.to_numeric(preds["lon"], errors="coerce")
    dt = dt.dropna(subset=["lat", "lon"]).copy()
    preds = preds.dropna(subset=["lat", "lon"]).copy()

    pred_rad = np.radians(preds[["lat", "lon"]].to_numpy(dtype=float))
    dt_rad = np.radians(dt[["lat", "lon"]].to_numpy(dtype=float))
    tree = BallTree(pred_rad, metric="haversine")
    dist_rad, indices = tree.query(dt_rad, k=1)
    earth_radius_m = 6371000.0

    matched_pred = preds.iloc[indices[:, 0]].reset_index(drop=True)
    matched = dt.reset_index(drop=True).copy()
    matched["match_distance_m"] = dist_rad[:, 0] * earth_radius_m

    for col in [
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "pred_rsrp_geo",
        "pred_rsrq_geo",
        "pred_sinr_geo",
        "pred_rsrp_demo",
        "pred_rsrq_demo",
        "pred_sinr_demo",
        "demo_visual_source",
        "demo_blend_weight",
        "demo_dt_anchor",
    ]:
        if col in matched_pred.columns:
            matched[col] = matched_pred[col].values
    return matched


def _print_metric_group(title: str, matched: pd.DataFrame, meas_col: str, pred_col: str) -> None:
    if meas_col not in matched.columns or pred_col not in matched.columns:
        print(f"[EVAL][{title}] skipped missing_column pred_col={pred_col} meas_col={meas_col}")
        return
    metrics = _metric_bundle(
        pd.to_numeric(matched[meas_col], errors="coerce"),
        pd.to_numeric(matched[pred_col], errors="coerce"),
    )
    if not metrics:
        print(f"[EVAL][{title}] skipped no_valid_rows pred_col={pred_col} meas_col={meas_col}")
        return
    print(f"[EVAL][{title}] {metrics}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate an already-completed LTE production job against DT")
    parser.add_argument("--job-id", required=True, type=str)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--session-ids", required=True, type=int, nargs="+")
    parser.add_argument("--region", default="india", type=str)
    parser.add_argument("--csv-path", default=None, type=str)
    parser.add_argument("--save-matched-csv", default=None, type=str)
    args = parser.parse_args(argv)

    pred_df = _load_prediction_csv(args.job_id, args.csv_path)
    operator = _resolve_operator(args.project_id, args.region)
    drive_df = _fetch_drive_df(args.session_ids, operator, args.project_id, args.region)
    matched = _match_dt_to_predictions(drive_df, pred_df)

    print(
        f"[EVAL] pred_rows={len(pred_df)} dt_rows={len(drive_df)} "
        f"matched_rows={len(matched)} mean_match_distance_m={round(float(matched['match_distance_m'].mean()), 4)}"
    )

    specs: Tuple[Tuple[str, str, str], ...] = (
        ("RSRP baseline", "rsrp", "pred_rsrp"),
        ("RSRQ baseline", "rsrq", "pred_rsrq"),
        ("SINR baseline", "sinr", "pred_sinr"),
        ("RSRP geo", "rsrp", "pred_rsrp_geo"),
        ("RSRQ geo", "rsrq", "pred_rsrq_geo"),
        ("SINR geo", "sinr", "pred_sinr_geo"),
        ("RSRP smooth", "rsrp", "pred_rsrp_demo"),
        ("RSRQ smooth", "rsrq", "pred_rsrq_demo"),
        ("SINR smooth", "sinr", "pred_sinr_demo"),
    )
    for title, meas_col, pred_col in specs:
        _print_metric_group(title, matched, meas_col, pred_col)

    if "demo_visual_source" in matched.columns:
        counts = matched["demo_visual_source"].astype(str).value_counts(dropna=False).to_dict()
        print(f"[EVAL] demo_visual_source_counts={counts}")
    if "demo_dt_anchor" in matched.columns:
        anchors = int(pd.Series(matched["demo_dt_anchor"]).fillna(False).astype(bool).sum())
        print(f"[EVAL] demo_dt_anchor_matches={anchors}")

    if args.save_matched_csv:
        out_path = Path(args.save_matched_csv)
        matched.to_csv(out_path, index=False)
        print(f"[EVAL] wrote_matched_csv={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
