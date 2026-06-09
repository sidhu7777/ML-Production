from flask import Blueprint, request, jsonify, current_app
from .services import LTEPredictionService
import multiprocessing
lte_prediction_bp = Blueprint("lte_prediction", __name__)
svc = LTEPredictionService()


@lte_prediction_bp.route("/run", methods=["POST"])
def run_prediction():

    try:
        data = request.get_json()
        app = current_app._get_current_object()

        cpu_count = multiprocessing.cpu_count()

        cfg = {
            "project_id": int(data["project_id"]),
            "session_ids": data["session_ids"],
            "region": str(data.get("region", "india")).lower(),
            "radius_m": float(data.get("radius", data.get("radius_m", 500))),
            "grid_resolution": float(data.get("grid_resolution", 25)),
            "building": bool(data.get("building", True)),
            "n_workers": int(data.get("n_workers", max(1, cpu_count - 1))),
            "max_interference_sites": int(data.get("max_interference_sites", 10)),
            "use_frontend_grid_sampling": bool(data.get("use_frontend_grid_sampling", True)),
            "samples_per_grid_axis": int(data.get("samples_per_grid_axis", 3)),
            "max_cells_per_grid": int(data.get("max_cells_per_grid", 3)),
            "min_cells_per_grid": int(data.get("min_cells_per_grid", 1)),
            "ensure_all_cells": bool(data.get("ensure_all_cells", True)),
            "min_grids_per_cell": int(data.get("min_grids_per_cell", 1)),
            "grid_analytics_scenario_id": data.get("grid_analytics_scenario_id"),
            "dem_raster_path": data.get("dem_raster_path"),
            "output_folder": current_app.config['OUTPUT_FOLDER']
        }

        return jsonify(svc.submit(app, cfg))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@lte_prediction_bp.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    return jsonify(svc.get(job_id))


@lte_prediction_bp.route("/result/<job_id>", methods=["GET"])
def result(job_id):
    return jsonify(svc.get(job_id))
