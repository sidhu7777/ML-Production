from __future__ import annotations

import datetime as dt
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from . import db


ML_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL3_INPUT_EXCEL = ML_ROOT / "models" / "model3_project196_input" / "project_196_model3_input.xlsx"
DEFAULT_MODEL2_INPUT_EXCEL = (
    ML_ROOT
    / "tools"
    / "coverage_prediction"
    / "future_demand_capacity_forecast"
    / "data"
    / "project_196_model2_demand_capacity_input.xlsx"
)
DEFAULT_OUTPUT_ROOT = ML_ROOT / "outputs" / "future_capacity_recommendation"
DEFAULT_STABLE_OUTPUT_DIR = ML_ROOT / "models" / "future_capacity_recommendation_experiment"

JOBS: dict[str, dict[str, Any]] = {}


def _wait_for_job(service: Any, job_id: str, *, timeout_sec: int, label: str) -> dict[str, Any]:
    start = time.time()
    last_progress = None
    while True:
        status = service.get(job_id)
        progress = status.get("progress")
        if progress != last_progress:
            print(f"[FUTURE_CAPACITY_RECOMMENDATION][UPSTREAM] {label} {status.get('status')}: {progress}", flush=True)
            last_progress = progress
        if status.get("status") in {"completed", "error", "failed"}:
            if status.get("status") != "completed":
                raise RuntimeError(f"{label} failed: {status.get('error') or status.get('progress')}")
            return status
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"{label} did not finish within {timeout_sec} sec")
        time.sleep(2)


def _model2_future_csv(model2_df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "Node_Cell_ID": model2_df["node_cell_id"].astype(str),
            "canonical_physical_cell_id": model2_df.get("canonical_physical_cell_id", model2_df["node_cell_id"]).astype(str),
            "site_id": model2_df["site_id"].astype(str),
            "sector_id": model2_df["sector_id"].astype(str),
            "band": model2_df["band"].astype(str),
            "estimated_prb_utilization_pct": pd.to_numeric(model2_df["future_prb_utilization_pct"], errors="coerce"),
            "estimated_cell_rrc_utilization_pct": pd.to_numeric(model2_df["future_rrc_utilization_pct"], errors="coerce"),
            "estimated_cell_rrc_connected_users": pd.to_numeric(model2_df["future_rrc_connected_users"], errors="coerce"),
            "estimated_rrc_connected_users": pd.to_numeric(model2_df["future_rrc_connected_users"], errors="coerce"),
            "estimated_offered_traffic_mbps": pd.to_numeric(model2_df["future_estimated_offered_traffic_mbps"], errors="coerce"),
            "estimated_dl_capacity_mbps": pd.to_numeric(model2_df["current_estimated_dl_capacity_mbps"], errors="coerce"),
            "current_prb_utilization_pct": pd.to_numeric(model2_df["current_prb_utilization_pct"], errors="coerce"),
            "current_rrc_utilization_pct": pd.to_numeric(model2_df["current_rrc_utilization_pct"], errors="coerce"),
            "model2_future_congested_flag": pd.to_numeric(model2_df["future_congested_flag"], errors="coerce"),
        }
    )
    out.to_csv(output_path, index=False)
    return output_path


class FutureCapacityRecommendationService:
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
        print(f"[FUTURE_CAPACITY_RECOMMENDATION][{job_id[:8]}] {status}: {progress}", flush=True)

    def _run_job(self, job_id: str, cfg: dict[str, Any]) -> None:
        started = dt.datetime.utcnow()
        try:
            project_id = int(cfg["project_id"])
            region = str(cfg.get("region", "india") or "india").lower()
            operator = str(cfg.get("operator", "") or "").strip() or None
            baseline_job_id = str(cfg.get("baseline_job_id") or db.latest_baseline_job_id(project_id, region, operator))
            model_run_id = str(cfg.get("model_run_id") or f"future_capacity_recommendation_{project_id}_{job_id[:8]}")

            model3_input_excel = Path(str(cfg.get("model3_input_excel_path") or DEFAULT_MODEL3_INPUT_EXCEL))
            model2_input_excel = Path(str(cfg.get("model2_input_excel_path") or DEFAULT_MODEL2_INPUT_EXCEL))
            if not model3_input_excel.is_absolute():
                model3_input_excel = ML_ROOT / model3_input_excel
            if not model2_input_excel.is_absolute():
                model2_input_excel = ML_ROOT / model2_input_excel

            source_model1_run_id = str(cfg.get("source_model1_run_id") or f"{model_run_id}_future_coverage_kpi")
            source_model2_run_id = str(cfg.get("source_model2_run_id") or f"{model_run_id}_future_demand_capacity")

            if not bool(cfg.get("skip_model1", False)):
                self._update(job_id, "running", "Running upstream future coverage KPI prediction")
                from tools.coverage_prediction.future_coverage_kpi_prediction.services import FutureCoverageKpiPredictionService

                model1_service = FutureCoverageKpiPredictionService()
                model1_job = model1_service.submit(
                    {
                        "project_id": project_id,
                        "region": region,
                        "operator": operator,
                        "baseline_job_id": baseline_job_id,
                        "model_run_id": source_model1_run_id,
                    }
                )
                model1_status = _wait_for_job(
                    model1_service,
                    model1_job["job_id"],
                    timeout_sec=int(cfg.get("model1_timeout_sec", 600)),
                    label="Future coverage KPI prediction",
                )
                source_model1_run_id = str(model1_status.get("model_run_id") or source_model1_run_id)

            self._update(job_id, "running", "Running upstream future demand/capacity forecast")
            from tools.coverage_prediction.future_demand_capacity_forecast.services import LTEModel2DemandCapacityForecastService

            model2_service = LTEModel2DemandCapacityForecastService()
            model2_job = model2_service.submit(
                {
                    "project_id": project_id,
                    "region": region,
                    "operator": operator,
                    "baseline_job_id": baseline_job_id,
                    "model_run_id": source_model2_run_id,
                    "input_excel_path": str(model2_input_excel),
                }
            )
            model2_status = _wait_for_job(
                model2_service,
                model2_job["job_id"],
                timeout_sec=int(cfg.get("model2_timeout_sec", 600)),
                label="Future demand/capacity forecast",
            )
            source_model2_run_id = str(model2_status.get("model_run_id") or source_model2_run_id)

            self._update(job_id, "running", "Preparing future demand/capacity output for future recommendation")
            model2_df = db.fetch_model2_results(source_model2_run_id, region=region)
            future_csv = _model2_future_csv(
                model2_df,
                DEFAULT_OUTPUT_ROOT / model_run_id / "future_demand_capacity_for_recommendation.csv",
            )

            self._update(job_id, "running", "Running dynamic future capacity recommendations")
            from tests.coverage_prediction import model3_business_rule_recommendation_test as future_rules
            from tests.coverage_prediction.model4_future_recommendation_test import run_model4_future_recommendation

            output_root = Path(str(cfg.get("output_root") or DEFAULT_OUTPUT_ROOT))
            stable_output_dir = Path(str(cfg.get("stable_output_dir") or DEFAULT_STABLE_OUTPUT_DIR))
            if not output_root.is_absolute():
                output_root = ML_ROOT / output_root
            if not stable_output_dir.is_absolute():
                stable_output_dir = ML_ROOT / stable_output_dir

            run_cfg = future_rules.Model3RecommendationConfig(
                dataset_path=Path(""),
                summary_path=Path(""),
                output_root=output_root,
                stable_output_dir=stable_output_dir,
                congestion_threshold=float(cfg.get("congestion_threshold", 70.0)),
                rrc_sector_capacity=future_rules.DEFAULT_RRC_SECTOR_CAPACITY,
            )
            run_dir = run_model4_future_recommendation(
                run_cfg,
                excel_path=model3_input_excel,
                future_dataset_path=future_csv,
                max_congested_cells=cfg.get("max_congested_cells"),
                rf_workers=max(1, int(cfg.get("rf_workers", 2))),
                sector_parallelism=max(1, int(cfg.get("sector_parallelism", 1))),
                max_interference_sites=max(1, int(cfg.get("max_interference_sites", 10))),
                action_neighbor_cells=max(0, int(cfg.get("action_neighbor_cells", 2))),
                carrier_reselection_hysteresis_db=float(cfg.get("carrier_reselection_hysteresis_db", 0.0)),
            )

            summary_path = stable_output_dir / "model4_future_recommendation_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            files = summary.get("files", {}) if isinstance(summary, dict) else {}
            rec_path = Path(files.get("recommendations_csv") or stable_output_dir / "model4_future_recommendations.csv")
            after_rf_path = stable_output_dir / "model4_after_rf_surface_combined.csv"
            before_rf_path = stable_output_dir / "model4_before_rf_surface_combined.csv"
            if before_rf_path.is_file():
                try:
                    before_rf_path.unlink()
                except PermissionError:
                    pass

            recommendations = pd.read_csv(rec_path) if rec_path.is_file() else pd.DataFrame()
            after_rf = pd.read_csv(after_rf_path, low_memory=False) if after_rf_path.is_file() else pd.DataFrame()
            runtime_sec = (dt.datetime.utcnow() - started).total_seconds()

            self._update(job_id, "running", "Saving future capacity recommendation rows")
            rec_out = db.prepare_recommendation_results(
                recommendations,
                project_id=project_id,
                baseline_job_id=baseline_job_id,
                model_run_id=model_run_id,
                region=region,
                operator=operator,
                source_model1_run_id=source_model1_run_id,
                source_model2_run_id=source_model2_run_id,
                model_version="future_capacity_dynamic_rf",
                artifact_recommendations_path=str(rec_path) if rec_path.is_file() else None,
                artifact_after_rf_path=str(after_rf_path) if after_rf_path.is_file() else None,
                after_rf_rows=int(len(after_rf)),
                rf_runtime_sec=runtime_sec,
            )
            recommendation_rows = db.save_recommendation_results(rec_out, region=region)

            self._update(job_id, "running", "Saving future capacity after-RF rows")
            rf_out = db.prepare_after_rf_results(
                after_rf,
                project_id=project_id,
                baseline_job_id=baseline_job_id,
                model_run_id=model_run_id,
                region=region,
                operator=operator,
                artifact_path=str(after_rf_path) if after_rf_path.is_file() else None,
            )
            after_rf_rows = db.save_after_rf_results(rf_out, region=region)

            self._update(
                job_id,
                "completed",
                "Future capacity recommendation completed",
                project_id=project_id,
                baseline_job_id=baseline_job_id,
                model_run_id=model_run_id,
                source_model1_run_id=source_model1_run_id,
                source_model2_run_id=source_model2_run_id,
                model2_future_rows=int(len(model2_df)),
                run_dir=str(run_dir),
                recommendation_rows=int(recommendation_rows),
                after_rf_rows=int(after_rf_rows),
                resolved_rows=int(rec_out["resolved_flag"].sum()) if not rec_out.empty else 0,
                runtime_sec=round(runtime_sec, 3),
            )
        except Exception as exc:
            self._update(job_id, "error", str(exc), error=str(exc))


LTEModel4FutureRecommendationService = FutureCapacityRecommendationService
