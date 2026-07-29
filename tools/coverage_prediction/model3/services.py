from __future__ import annotations

import datetime as dt
import json
import threading
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from . import db


ML_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_EXCEL = ML_ROOT / "models" / "model3_project196_input" / "project_196_model3_input.xlsx"
DEFAULT_OUTPUT_ROOT = ML_ROOT / "outputs" / "model3_current_recommendation"
DEFAULT_STABLE_OUTPUT_DIR = ML_ROOT / "models" / "model3_current_recommendation_experiment"

JOBS: dict[str, dict[str, Any]] = {}


class LTEModel3CurrentRecommendationService:
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
        print(f"[MODEL3][{job_id[:8]}] {status}: {progress}", flush=True)

    def _run_job(self, job_id: str, cfg: dict[str, Any]) -> None:
        started = dt.datetime.utcnow()
        try:
            project_id = int(cfg["project_id"])
            region = str(cfg.get("region", "india") or "india").lower()
            operator = str(cfg.get("operator", "") or "").strip() or None
            baseline_job_id = str(cfg.get("baseline_job_id") or db.latest_baseline_job_id(project_id, region, operator))
            model_run_id = str(cfg.get("model_run_id") or f"model3_{project_id}_{job_id[:8]}")
            source_model2_run_id = cfg.get("source_model2_run_id")

            input_excel = Path(str(cfg.get("input_excel_path") or DEFAULT_INPUT_EXCEL))
            if not input_excel.is_absolute():
                input_excel = ML_ROOT / input_excel
            if not input_excel.exists():
                raise FileNotFoundError(f"Model 3 current input Excel not found: {input_excel}")

            self._update(job_id, "running", "Loading Model 3 current recommendation engine")
            from tests.coverage_prediction.model3_current_recommendation_test import (
                CurrentModel3Config,
                run_model3_current_recommendation_test,
            )

            output_root = Path(str(cfg.get("output_root") or DEFAULT_OUTPUT_ROOT))
            stable_output_dir = Path(str(cfg.get("stable_output_dir") or DEFAULT_STABLE_OUTPUT_DIR))
            if not output_root.is_absolute():
                output_root = ML_ROOT / output_root
            if not stable_output_dir.is_absolute():
                stable_output_dir = ML_ROOT / stable_output_dir

            run_config = CurrentModel3Config(
                dataset_path=input_excel,
                output_root=output_root,
                stable_output_dir=stable_output_dir,
                congestion_threshold=float(cfg.get("congestion_threshold", 70.0)),
                max_congested_cells=cfg.get("max_congested_cells"),
                carrier_reselection_hysteresis_db=float(cfg.get("carrier_reselection_hysteresis_db", 0.0)),
                rf_workers=max(1, int(cfg.get("rf_workers", 2))),
                max_interference_sites=max(1, int(cfg.get("max_interference_sites", 10))),
                action_neighbor_cells=max(0, int(cfg.get("action_neighbor_cells", 2))),
                sector_parallelism=max(1, int(cfg.get("sector_parallelism", 1))),
                stop_on_partial=bool(cfg.get("stop_on_partial", False)),
            )

            self._update(job_id, "running", "Running dynamic Model 3 recommendations")
            run_dir = run_model3_current_recommendation_test(run_config)
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            artifacts = summary.get("artifacts", {}) if isinstance(summary, dict) else {}

            rec_path = Path(artifacts.get("recommendations_csv") or run_dir / "model3_current_recommendations.csv")
            after_rf_path = Path(artifacts.get("after_rf_surface_combined_csv") or "")
            before_rf_path = Path(artifacts.get("before_rf_surface_combined_csv") or "")
            if before_rf_path.is_file():
                try:
                    before_rf_path.unlink()
                except PermissionError:
                    pass

            recommendations = pd.read_csv(rec_path) if rec_path.is_file() else pd.DataFrame()
            after_rf = pd.read_csv(after_rf_path, low_memory=False) if after_rf_path.is_file() else pd.DataFrame()

            runtime_sec = (dt.datetime.utcnow() - started).total_seconds()
            self._update(job_id, "running", "Saving Model 3 recommendation rows")
            rec_out = db.prepare_recommendation_results(
                recommendations,
                project_id=project_id,
                baseline_job_id=baseline_job_id,
                model_run_id=model_run_id,
                region=region,
                operator=operator,
                source_model2_run_id=source_model2_run_id,
                model_version="model3_current_dynamic_rf",
                artifact_recommendations_path=str(rec_path) if rec_path.is_file() else None,
                artifact_after_rf_path=str(after_rf_path) if after_rf_path.is_file() else None,
                after_rf_rows=int(len(after_rf)),
                rf_runtime_sec=runtime_sec,
            )
            recommendation_rows = db.save_recommendation_results(rec_out, region=region)

            self._update(job_id, "running", "Saving Model 3 after-RF rows")
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
                "Model 3 current recommendation completed",
                project_id=project_id,
                baseline_job_id=baseline_job_id,
                model_run_id=model_run_id,
                source_model2_run_id=source_model2_run_id,
                input_excel_path=str(input_excel),
                run_dir=str(run_dir),
                recommendation_rows=int(recommendation_rows),
                after_rf_rows=int(after_rf_rows),
                congested_cell_rows=int(summary.get("congested_cell_rows", len(recommendations))) if isinstance(summary, dict) else int(len(recommendations)),
                resolved_rows=int(rec_out["resolved_flag"].sum()) if not rec_out.empty else 0,
                runtime_sec=round(runtime_sec, 3),
            )
        except Exception as exc:
            self._update(job_id, "error", str(exc), error=str(exc))
