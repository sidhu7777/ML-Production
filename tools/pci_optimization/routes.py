from __future__ import annotations

import os

from flask import Blueprint, jsonify, request, send_file

from .engine import MOD_RULE_VALUES
from .services import OUTPUT_ROOT, PciOptimizationService


pci_optimization_bp = Blueprint("pci_optimization", __name__)
service = PciOptimizationService()


def _sanitize_mod_values(raw_mod) -> list[int]:
    """Never trust the client alone: MOD_RULE_VALUES = {1,3,6,7,8,9} is the
    only set the optimizer can safely use (Mod0 is a ZeroDivisionError, not
    a real rule). Anything outside that set -- 0, negative, out-of-range,
    non-numeric -- is silently dropped, never crashes the job. A frontend
    bug sending a stray 0 (confirmed: Number("") on a cleared field) can no
    longer take the whole run down."""
    if not isinstance(raw_mod, list):
        return []
    cleaned = []
    for value in raw_mod:
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if n in MOD_RULE_VALUES:
            cleaned.append(n)
    return cleaned


@pci_optimization_bp.route("/run", methods=["POST"])
def run_pci_optimization_job():
    data = request.get_json() or {}
    if "project_id" not in data:
        return jsonify({"error": "project_id is required"}), 400
    rules = data.get("rules") or {"collision": True, "confusion": True, "mod": [], "grouped": False, "co_centric": False}
    cfg = {
        "project_id": int(data["project_id"]),
        "region": str(data.get("region", "india") or "india").lower(),
        "operator": data.get("operator", "all"),
        "primary_only": data.get("primary_only", True),
        "filter_sites_to_polygon": data.get("filter_sites_to_polygon", True),
        "filter_logs_to_polygon": data.get("filter_logs_to_polygon", False),
        "neighbor_distance_m": data.get("neighbor_distance_m", 500),
        "rules": {**rules, "mod": _sanitize_mod_values(rules.get("mod"))},
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


@pci_optimization_bp.route("/download", methods=["GET"])
def download():
    """Same pattern as /api/lte-tilt-recommandation/download, plus a
    containment check the tilt route doesn't have -- the requested path
    must resolve inside OUTPUT_ROOT, so this can't be used to read an
    arbitrary file off the server via a crafted `file` query param."""
    file_path = request.args.get("file")
    if not file_path:
        return jsonify({"error": "file path required"}), 400

    resolved = os.path.realpath(file_path)
    output_root = os.path.realpath(str(OUTPUT_ROOT))
    if os.path.commonpath([resolved, output_root]) != output_root:
        return jsonify({"error": "file path not allowed"}), 400

    if not os.path.exists(resolved):
        return jsonify({"error": "File not found or expired on server"}), 404

    try:
        return send_file(resolved, as_attachment=True)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to download file: {exc}"}), 500
