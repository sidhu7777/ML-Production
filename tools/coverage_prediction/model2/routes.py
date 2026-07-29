from __future__ import annotations

from flask import Blueprint, jsonify, request

from .services import LTEModel2DemandCapacityForecastService


lte_model2_demand_capacity_bp = Blueprint("lte_model2_demand_capacity_forecast", __name__)
service = LTEModel2DemandCapacityForecastService()


@lte_model2_demand_capacity_bp.route("/run", methods=["POST"])
def run_model2_demand_capacity_forecast():
    data = request.get_json() or {}
    if "project_id" not in data:
        return jsonify({"error": "project_id is required"}), 400
    cfg = {
        "project_id": int(data["project_id"]),
        "region": str(data.get("region", "india") or "india").lower(),
        "operator": data.get("operator"),
        "baseline_job_id": data.get("baseline_job_id"),
        "model_run_id": data.get("model_run_id"),
        "input_excel_path": data.get("input_excel_path"),
    }
    return jsonify(service.submit(cfg)), 202


@lte_model2_demand_capacity_bp.route("/status/<job_id>", methods=["GET"])
def status(job_id: str):
    job = service.get(job_id)
    if job.get("status") == "not_found":
        return jsonify(job), 404
    return jsonify(job)


@lte_model2_demand_capacity_bp.route("/result/<job_id>", methods=["GET"])
def result(job_id: str):
    job = service.get(job_id)
    if job.get("status") == "not_found":
        return jsonify(job), 404
    return jsonify(job)
