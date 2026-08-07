from __future__ import annotations

import datetime as dt
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import db
from .feature_builder import build_model1_feature_frame
from .model_registry import TARGETS, load_latest_model1_bundle


JOBS: dict[str, dict[str, Any]] = {}


def _safe_float(value: Any) -> float | None:
    try:
        num = float(value)
    except Exception:
        return None
    if np.isnan(num) or np.isinf(num):
        return None
    return num


class FutureCoverageKpiPredictionService:
    def submit(self, cfg: dict[str, Any]) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": "queued",
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
            "updated_at": dt.datetime.utcnow().isoformat() + "Z",
            "config": {
                "project_id": cfg.get("project_id"),
                "region": cfg.get("region", "india"),
                "operator": cfg.get("operator"),
                "baseline_job_id": cfg.get("baseline_job_id"),
            },
        }
        thread = threading.Thread(target=self._run_job, args=(job_id, dict(cfg)), daemon=True)
        thread.start()
        return {"job_id": job_id, "status": "queued"}

    def get(self, job_id: str) -> dict[str, Any]:
        return JOBS.get(job_id, {"job_id": job_id, "status": "not_found"})

    def _update(self, job_id: str, status: str, progress: str, **extra: Any) -> None:
        job = JOBS.setdefault(job_id, {"job_id": job_id})
        job.update(extra)
        job["status"] = status
        job["progress"] = progress
        job["updated_at"] = dt.datetime.utcnow().isoformat() + "Z"
        print(f"[MODEL1][{job_id[:8]}] {status}: {progress}", flush=True)

    def _run_job(self, job_id: str, cfg: dict[str, Any]) -> None:
        started = dt.datetime.utcnow()
        try:
            project_id = int(cfg["project_id"])
            region = str(cfg.get("region", "india") or "india").lower()
            operator = str(cfg.get("operator", "") or "").strip() or None
            baseline_job_id = cfg.get("baseline_job_id")
            model_run_id = str(cfg.get("model_run_id") or f"future_coverage_kpi_{project_id}_{job_id[:8]}")

            self._update(job_id, "running", "Loading Model 1 weights")
            bundle = load_latest_model1_bundle()

            self._update(job_id, "running", "Fetching baseline rows from PythonBridge")
            baseline_df, resolved_baseline_job_id, input_source = db.fetch_baseline_rows(
                project_id,
                region=region,
                operator=operator,
                baseline_job_id=baseline_job_id,
            )

            self._update(job_id, "running", "Fetching geo features from PythonBridge")
            geo_df = db.fetch_geo_rows(project_id, region=region, baseline_job_id=resolved_baseline_job_id)

            self._update(job_id, "running", "Building Model 1 feature frame")
            feature_df = build_model1_feature_frame(
                baseline_df,
                geo_df,
                numeric_features=bundle.numeric_features,
                categorical_features=bundle.categorical_features,
            )
            if feature_df.empty:
                raise ValueError("Model 1 feature frame is empty")

            x_cols = bundle.numeric_features + bundle.categorical_features
            pred_df = feature_df[
                [
                    "grid_id",
                    "grid_row",
                    "grid_col",
                    "grid_centroid_lat",
                    "grid_centroid_lon",
                    "current_rsrp",
                    "current_rsrq",
                    "current_sinr",
                ]
            ].copy()

            self._update(job_id, "running", f"Running future coverage KPI inference for {len(feature_df)} grids")
            for target in TARGETS:
                delta_col = target.replace("pred_", "delta_")
                current_col = target.replace("pred_", "current_")
                pred_df[delta_col] = bundle.models[target].predict(feature_df[x_cols])
                pred_df[target] = pd.to_numeric(pred_df[current_col], errors="coerce") + pd.to_numeric(
                    pred_df[delta_col], errors="coerce"
                )

            out = pd.DataFrame(
                {
                    "project_id": int(project_id),
                    "baseline_job_id": str(resolved_baseline_job_id),
                    "model_run_id": model_run_id,
                    "grid_id": pd.to_numeric(pred_df["grid_id"], errors="coerce").astype("int64"),
                    "grid_row": pd.to_numeric(pred_df["grid_row"], errors="coerce").astype("Int64"),
                    "grid_col": pd.to_numeric(pred_df["grid_col"], errors="coerce").astype("Int64"),
                    "grid_centroid_lat": pd.to_numeric(pred_df["grid_centroid_lat"], errors="coerce"),
                    "grid_centroid_lon": pd.to_numeric(pred_df["grid_centroid_lon"], errors="coerce"),
                    "current_rsrp": pd.to_numeric(pred_df["current_rsrp"], errors="coerce"),
                    "current_rsrq": pd.to_numeric(pred_df["current_rsrq"], errors="coerce"),
                    "current_sinr": pd.to_numeric(pred_df["current_sinr"], errors="coerce"),
                    "delta_rsrp": pd.to_numeric(pred_df["delta_rsrp"], errors="coerce"),
                    "delta_rsrq": pd.to_numeric(pred_df["delta_rsrq"], errors="coerce"),
                    "delta_sinr": pd.to_numeric(pred_df["delta_sinr"], errors="coerce"),
                    "pred_rsrp": pd.to_numeric(pred_df["pred_rsrp"], errors="coerce"),
                    "pred_rsrq": pd.to_numeric(pred_df["pred_rsrq"], errors="coerce"),
                    "pred_sinr": pd.to_numeric(pred_df["pred_sinr"], errors="coerce"),
                    "model_name": "future_coverage_kpi_prediction",
                    "model_version": bundle.model_version,
                    "weights_rsrp_path": bundle.weights_paths["pred_rsrp"],
                    "weights_rsrq_path": bundle.weights_paths["pred_rsrq"],
                    "weights_sinr_path": bundle.weights_paths["pred_sinr"],
                    "input_source": input_source,
                    "status": "completed",
                }
            )
            out = out.loc[out["grid_centroid_lat"].notna() & out["grid_centroid_lon"].notna()].copy()

            self._update(job_id, "running", f"Saving {len(out)} future coverage KPI rows to DB")
            rows_written = db.save_results(out, region=region)
            elapsed = (dt.datetime.utcnow() - started).total_seconds()
            self._update(
                job_id,
                "completed",
                "Future coverage KPI prediction completed",
                project_id=project_id,
                baseline_job_id=str(resolved_baseline_job_id),
                model_run_id=model_run_id,
                input_source=input_source,
                baseline_rows=int(len(baseline_df)),
                geo_rows=int(len(geo_df)),
                feature_rows=int(len(feature_df)),
                rows_written=int(rows_written),
                runtime_sec=round(elapsed, 3),
                pred_ranges={
                    target: {
                        "min": _safe_float(out[target].min()),
                        "max": _safe_float(out[target].max()),
                    }
                    for target in TARGETS
                },
                delta_ranges={
                    target.replace("pred_", "delta_"): {
                        "min": _safe_float(out[target.replace("pred_", "delta_")].min()),
                        "max": _safe_float(out[target.replace("pred_", "delta_")].max()),
                    }
                    for target in TARGETS
                },
            )
        except Exception as exc:
            self._update(job_id, "error", str(exc), error=str(exc))


LTEModel1CoveragePredictionService = FutureCoverageKpiPredictionService
