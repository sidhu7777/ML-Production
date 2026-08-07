from __future__ import annotations

from flask import Blueprint, jsonify, request

from .services import FutureCapacityRecommendationService


future_capacity_recommendation_bp = Blueprint("future_capacity_recommendation", __name__)
lte_model4_future_recommendation_bp = future_capacity_recommendation_bp
service = FutureCapacityRecommendationService()


@future_capacity_recommendation_bp.route("/run", methods=["POST"])
def run_model4_future_recommendation():
    data = request.get_json() or {}
    if "project_id" not in data:
        return jsonify({"error": "project_id is required"}), 400
    cfg = {
        "project_id": int(data["project_id"]),
        "region": str(data.get("region", "india") or "india").lower(),
        "operator": data.get("operator"),
        "baseline_job_id": data.get("baseline_job_id"),
        "model_run_id": data.get("model_run_id"),
        "source_model1_run_id": data.get("source_model1_run_id"),
        "source_model2_run_id": data.get("source_model2_run_id"),
        "model3_input_excel_path": data.get("model3_input_excel_path"),
        "model2_input_excel_path": data.get("model2_input_excel_path"),
        "congestion_threshold": data.get("congestion_threshold", 70.0),
        "max_congested_cells": data.get("max_congested_cells"),
        "carrier_reselection_hysteresis_db": data.get("carrier_reselection_hysteresis_db", 0.0),
        "rf_workers": data.get("rf_workers", 2),
        "max_interference_sites": data.get("max_interference_sites", 10),
        "action_neighbor_cells": data.get("action_neighbor_cells", 2),
        "sector_parallelism": data.get("sector_parallelism", 1),
        "skip_model1": data.get("skip_model1", False),
    }
    return jsonify(service.submit(cfg)), 202


@future_capacity_recommendation_bp.route("/status/<job_id>", methods=["GET"])
def status(job_id: str):
    job = service.get(job_id)
    if job.get("status") == "not_found":
        return jsonify(job), 404
    return jsonify(job)


@future_capacity_recommendation_bp.route("/result/<job_id>", methods=["GET"])
def result(job_id: str):
    job = service.get(job_id)
    if job.get("status") == "not_found":
        return jsonify(job), 404
    return jsonify(job)
