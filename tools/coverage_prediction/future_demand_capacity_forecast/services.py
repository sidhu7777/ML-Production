from __future__ import annotations

import datetime as dt
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import db
from .feature_builder import build_model2_feature_frame
from .model_registry import load_latest_model2_bundle


ML_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_EXCEL = (
    ML_ROOT
    / "tools"
    / "coverage_prediction"
    / "future_demand_capacity_forecast"
    / "data"
    / "project_196_model2_demand_capacity_input.xlsx"
)

JOBS: dict[str, dict[str, Any]] = {}


def _safe_float(value: Any) -> float | None:
    try:
        num = float(value)
    except Exception:
        return None
    if np.isnan(num) or np.isinf(num):
        return None
    return round(num, 6)


def _load_cell_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model 2 input Excel not found: {path}")
    return pd.read_excel(path, sheet_name="Model2_Cell_Input")


def _future_forecast_from_model_outputs(feature_df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    current_prb = pd.to_numeric(feature_df["current_prb_utilization_pct"], errors="coerce").fillna(0)
    current_rrc = pd.to_numeric(feature_df["current_rrc_utilization_pct"], errors="coerce").fillna(0)
    current_users = pd.to_numeric(feature_df["current_rrc_connected_users"], errors="coerce").fillna(0)
    current_traffic = pd.to_numeric(feature_df["current_estimated_offered_traffic_mbps"], errors="coerce").fillna(0)

    demand = pd.to_numeric(pred["demand_index"], errors="coerce").fillna(current_prb)
    users = pd.to_numeric(pred["active_users_est"], errors="coerce").fillna(0)
    traffic = pd.to_numeric(pred["traffic_demand_est"], errors="coerce").fillna(0)

    def percentile_norm(series: pd.Series) -> pd.Series:
        lo = float(series.quantile(0.10))
        hi = float(series.quantile(0.90))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(0.0, index=series.index, dtype="float64")
        return ((series - lo) / (hi - lo)).clip(0.0, 1.0)

    demand_pressure = percentile_norm(demand)
    user_pressure = percentile_norm(users)
    traffic_pressure = percentile_norm(traffic)
    growth_rate = pd.to_numeric(feature_df.get("growth_rate", 0.05), errors="coerce").fillna(0.05).clip(0.03, 0.20)
    growth_zone = (pd.to_numeric(feature_df.get("growth_zone_score", 0), errors="coerce").fillna(0) / 100.0).clip(0.0, 1.0)
    capacity_gap = (pd.to_numeric(feature_df.get("capacity_gap_score", 0), errors="coerce").fillna(0) / 30.0).clip(0.0, 1.0)
    model_growth_pressure = (
        0.35 * demand_pressure
        + 0.25 * traffic_pressure
        + 0.15 * user_pressure
        + 0.25 * growth_zone
    ).clip(0.0, 1.0)

    horizon_load_scale = 90.0
    prb_delta = horizon_load_scale * growth_rate * model_growth_pressure + 4.0 * capacity_gap
    rrc_delta = horizon_load_scale * growth_rate * (0.55 * model_growth_pressure + 0.45 * user_pressure) + 2.0 * capacity_gap

    out = pd.DataFrame(index=feature_df.index)
    out["future_prb_utilization_pct"] = (current_prb + prb_delta).clip(0, 100)
    out["future_rrc_utilization_pct"] = (current_rrc + rrc_delta).clip(0, 100)
    out["future_rrc_connected_users"] = (current_users * 1.08 + users.clip(lower=0)).clip(lower=0)
    out["future_estimated_offered_traffic_mbps"] = (current_traffic * 1.08 + traffic.clip(lower=0)).clip(lower=0)
    out["future_congested_flag"] = (
        (out["future_prb_utilization_pct"] > 70) | (out["future_rrc_utilization_pct"] > 70)
    ).astype(int)
    return out


class FutureDemandCapacityForecastService:
    def submit(self, cfg: dict[str, Any]) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": "queued",
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
            "updated_at": dt.datetime.utcnow().isoformat() + "Z",
            "config": cfg,
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
        print(f"[FUTURE_DEMAND_CAPACITY][{job_id[:8]}] {status}: {progress}", flush=True)

    def _run_job(self, job_id: str, cfg: dict[str, Any]) -> None:
        started = dt.datetime.utcnow()
        try:
            project_id = int(cfg["project_id"])
            region = str(cfg.get("region", "india") or "india").lower()
            operator = str(cfg.get("operator", "") or "").strip() or None
            baseline_job_id = cfg.get("baseline_job_id")
            model_run_id = str(cfg.get("model_run_id") or f"future_demand_capacity_{project_id}_{job_id[:8]}")
            input_excel = Path(str(cfg.get("input_excel_path") or DEFAULT_INPUT_EXCEL))
            if not input_excel.is_absolute():
                input_excel = ML_ROOT / input_excel

            self._update(job_id, "running", "Loading future demand/capacity cell input Excel")
            cell_df = _load_cell_input(input_excel)
            if "project_id" in cell_df.columns:
                cell_df = cell_df.loc[pd.to_numeric(cell_df["project_id"], errors="coerce") == project_id].copy()
            if cell_df.empty:
                raise ValueError(f"No future demand/capacity cell rows found for project_id={project_id}")

            self._update(job_id, "running", "Loading future demand/capacity weights")
            bundle = load_latest_model2_bundle()

            self._update(job_id, "running", "Fetching baseline rows from PythonBridge")
            baseline_df, resolved_baseline_job_id, input_source = db.fetch_baseline_rows(
                project_id,
                region=region,
                operator=operator,
                baseline_job_id=baseline_job_id,
            )

            self._update(job_id, "running", "Fetching geo features from PythonBridge")
            geo_df = db.fetch_geo_rows(project_id, region=region, baseline_job_id=resolved_baseline_job_id)

            self._update(job_id, "running", "Building future demand/capacity feature frame")
            feature_df = build_model2_feature_frame(
                cell_df,
                baseline_df,
                geo_df,
                numeric_features=bundle.numeric_features,
                categorical_features=bundle.categorical_features,
            )

            x_cols = bundle.numeric_features + bundle.categorical_features
            pred = pd.DataFrame(index=feature_df.index)
            self._update(job_id, "running", f"Running future demand/capacity inference for {len(feature_df)} cells")
            pred["demand_index"] = bundle.models["model2a_demand"].predict(feature_df[x_cols])
            pred["active_users_est"] = bundle.models["model2b_users"].predict(feature_df[x_cols])
            pred["traffic_demand_est"] = bundle.models["model2c_traffic"].predict(feature_df[x_cols])

            forecast = _future_forecast_from_model_outputs(feature_df, pred)
            current_prb = pd.to_numeric(feature_df["current_prb_utilization_pct"], errors="coerce")
            current_rrc = pd.to_numeric(feature_df["current_rrc_utilization_pct"], errors="coerce")
            out = pd.DataFrame(
                {
                    "project_id": project_id,
                    "baseline_job_id": str(resolved_baseline_job_id),
                    "model_run_id": model_run_id,
                    "site_id": feature_df["site_id"].astype(str),
                    "sector_id": feature_df["sector_id"].astype(str),
                    "node_cell_id": feature_df["node_cell_id"].astype(str),
                    "canonical_physical_cell_id": feature_df.get("canonical_physical_cell_id", feature_df["node_cell_id"]).astype(str),
                    "band": feature_df["band"].astype(str),
                    "operator": operator,
                    "current_prb_utilization_pct": current_prb,
                    "current_rrc_utilization_pct": current_rrc,
                    "current_rrc_connected_users": pd.to_numeric(feature_df["current_rrc_connected_users"], errors="coerce"),
                    "current_estimated_dl_capacity_mbps": pd.to_numeric(feature_df["current_estimated_dl_capacity_mbps"], errors="coerce"),
                    "current_estimated_offered_traffic_mbps": pd.to_numeric(feature_df["current_estimated_offered_traffic_mbps"], errors="coerce"),
                    "current_congested_flag": ((current_prb > 70) | (current_rrc > 70)).astype(int),
                    "future_prb_utilization_pct": forecast["future_prb_utilization_pct"],
                    "future_rrc_utilization_pct": forecast["future_rrc_utilization_pct"],
                    "future_rrc_connected_users": forecast["future_rrc_connected_users"],
                    "future_estimated_offered_traffic_mbps": forecast["future_estimated_offered_traffic_mbps"],
                    "future_congested_flag": forecast["future_congested_flag"],
                    "model_name": "future_demand_capacity_forecast",
                    "model_version": bundle.model_version,
                    "weights_demand_path": bundle.weights_paths["model2a_demand"],
                    "weights_users_path": bundle.weights_paths["model2b_users"],
                    "weights_traffic_path": bundle.weights_paths["model2c_traffic"],
                    "input_source": input_source,
                    "status": "completed",
                }
            )

            self._update(job_id, "running", f"Saving {len(out)} future demand/capacity forecast rows to DB")
            rows_written = db.save_results(out, region=region)
            elapsed = (dt.datetime.utcnow() - started).total_seconds()
            self._update(
                job_id,
                "completed",
                "Future demand/capacity forecast completed",
                project_id=project_id,
                baseline_job_id=str(resolved_baseline_job_id),
                model_run_id=model_run_id,
                input_excel_path=str(input_excel),
                input_source=input_source,
                cell_rows=int(len(cell_df)),
                baseline_rows=int(len(baseline_df)),
                geo_rows=int(len(geo_df)),
                rows_written=int(rows_written),
                runtime_sec=round(elapsed, 3),
                current_congested_cells=int(out["current_congested_flag"].sum()),
                future_congested_cells=int(out["future_congested_flag"].sum()),
                future_prb_range={
                    "min": _safe_float(out["future_prb_utilization_pct"].min()),
                    "max": _safe_float(out["future_prb_utilization_pct"].max()),
                },
                future_rrc_range={
                    "min": _safe_float(out["future_rrc_utilization_pct"].min()),
                    "max": _safe_float(out["future_rrc_utilization_pct"].max()),
                },
            )
        except Exception as exc:
            self._update(job_id, "error", str(exc), error=str(exc))


LTEModel2DemandCapacityForecastService = FutureDemandCapacityForecastService
