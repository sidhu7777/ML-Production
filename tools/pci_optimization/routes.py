from __future__ import annotations

from flask import Blueprint, jsonify, request

from .services import PciOptimizationService


pci_optimization_bp = Blueprint("pci_optimization", __name__)
service = PciOptimizationService()


@pci_optimization_bp.route("/run", methods=["POST"])
def run_pci_optimization_job():
    data = request.get_json() or {}
    if "project_id" not in data:
        return jsonify({"error": "project_id is required"}), 400
    cfg = {
        "project_id": int(data["project_id"]),
        "region": str(data.get("region", "india") or "india").lower(),
        "operator": data.get("operator", "all"),
        "primary_only": data.get("primary_only", True),
        "filter_sites_to_polygon": data.get("filter_sites_to_polygon", True),
        "filter_logs_to_polygon": data.get("filter_logs_to_polygon", False),
        "neighbor_distance_m": data.get("neighbor_distance_m", 500),
        "rules": data.get("rules") or {"collision": True, "confusion": True, "mod": [], "grouped": False, "co_centric": False},
        "run_optimizer": data.get("run_optimizer", True),
        "site_ids": data.get("site_ids") or [],
        "max_sites": data.get("max_sites", 0),
    }
    return jsonify(service.submit(cfg)), 202


@pci_optimization_bp.route("/status/<job_id>", methods=["GET"])
def status(job_id: str):
    job = service.get(job_id)
    if job.get("status") == "not_found":
        return jsonify(job), 404
    return jsonify(job)


@pci_optimization_bp.route("/result/<job_id>", methods=["GET"])
def result(job_id: str):
    job = service.get(job_id)
    if job.get("status") == "not_found":
        return jsonify(job), 404
    return jsonify(job)
