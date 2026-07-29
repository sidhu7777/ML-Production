from __future__ import annotations

from flask import Blueprint, jsonify, request

from .services import LTEModel3CurrentRecommendationService


lte_model3_current_recommendation_bp = Blueprint("lte_model3_current_recommendation", __name__)
service = LTEModel3CurrentRecommendationService()


@lte_model3_current_recommendation_bp.route("/run", methods=["POST"])
def run_model3_current_recommendation():
    data = request.get_json() or {}
    if "project_id" not in data:
        return jsonify({"error": "project_id is required"}), 400
    cfg = {
        "project_id": int(data["project_id"]),
        "region": str(data.get("region", "india") or "india").lower(),
        "operator": data.get("operator"),
        "baseline_job_id": data.get("baseline_job_id"),
        "model_run_id": data.get("model_run_id"),
        "source_model2_run_id": data.get("source_model2_run_id"),
        "input_excel_path": data.get("input_excel_path"),
        "congestion_threshold": data.get("congestion_threshold", 70.0),
        "max_congested_cells": data.get("max_congested_cells"),
        "carrier_reselection_hysteresis_db": data.get("carrier_reselection_hysteresis_db", 0.0),
        "rf_workers": data.get("rf_workers", 2),
        "max_interference_sites": data.get("max_interference_sites", 10),
        "action_neighbor_cells": data.get("action_neighbor_cells", 2),
        "sector_parallelism": data.get("sector_parallelism", 1),
        "stop_on_partial": data.get("stop_on_partial", False),
    }
    return jsonify(service.submit(cfg)), 202


@lte_model3_current_recommendation_bp.route("/status/<job_id>", methods=["GET"])
def status(job_id: str):
    job = service.get(job_id)
    if job.get("status") == "not_found":
        return jsonify(job), 404
    return jsonify(job)


@lte_model3_current_recommendation_bp.route("/result/<job_id>", methods=["GET"])
def result(job_id: str):
    job = service.get(job_id)
    if job.get("status") == "not_found":
        return jsonify(job), 404
    return jsonify(job)
