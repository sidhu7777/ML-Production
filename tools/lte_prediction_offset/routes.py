import multiprocessing

from flask import Blueprint, current_app, jsonify, request

from .services import LTEPredictionOffsetService


lte_prediction_offset_bp = Blueprint("lte_prediction_offset", __name__)
svc = LTEPredictionOffsetService()


def _resolve_region(data):
    raw_region = str(data.get("region") or "").strip().lower()
    if raw_region:
        if raw_region in {"tw", "twn"}:
            return "taiwan"
        if raw_region in {"in", "ind"}:
            return "india"
        return raw_region

    country_code = str(data.get("country_code") or data.get("countryCode") or "").strip().lower()
    if country_code in {"tw", "twn", "taiwan"}:
        return "taiwan"
    if country_code in {"in", "ind", "india"}:
        return "india"
    return "india"


@lte_prediction_offset_bp.route("/run", methods=["POST"])
def run_prediction():
    try:
        data = request.get_json() or {}
        app = current_app._get_current_object()
        cpu_count = multiprocessing.cpu_count()

        cfg = {
            "project_id": int(data["project_id"]),
            "session_ids": data["session_ids"],
            "region": _resolve_region(data),
            "country_code": data.get("country_code") or data.get("countryCode"),
            "polygon_ids": data.get("polygon_ids") or data.get("polygonIds"),
            "operator": str(data.get("operator", "") or "").strip(),
            "radius_m": float(data.get("radius", data.get("radius_m", 500))),
            "grid_resolution": float(data.get("grid_resolution", 25)),
            "building": bool(data.get("building", True)),
            # An approved project DEM asset. The physical scorer chooses its
            # elevation band dynamically; this only identifies the raster.
            "dem_raster_path": data.get("dem_raster_path") or data.get("demRasterPath"),
            # Optional project GHS-OBAT extract. When present it is used for
            # Phase-27 building-height matching before documented imputation.
            "ghs_obat_csv_path": data.get("ghs_obat_csv_path") or data.get("ghsObatCsvPath"),
            "n_workers": int(data.get("n_workers", max(1, cpu_count - 1))),
            "max_interference_sites": int(data.get("max_interference_sites", 10)),
            "use_frontend_grid_sampling": bool(data.get("use_frontend_grid_sampling", True)),
            "samples_per_grid_axis": int(data.get("samples_per_grid_axis", 1)),
            "max_cells_per_grid": int(data.get("max_cells_per_grid", data.get("max_viable_candidates_per_grid", 20))),
            "min_cells_per_grid": int(data.get("min_cells_per_grid", 1)),
            "ensure_all_cells": bool(data.get("ensure_all_cells", True)),
            "min_grids_per_cell": int(data.get("min_grids_per_cell", 1)),
            # Candidate eligibility is distance based.  No pre-loss RSRP
            # filter/candidate-count cap may decide serving coverage.
            "min_candidate_rsrp_dbm": data.get("min_candidate_rsrp_dbm"),
            "candidate_safety_cap": data.get("candidate_safety_cap"),
            "out_of_radius_backfill_k_nearest": int(data.get("out_of_radius_backfill_k_nearest", 8)),
            "grid_analytics_scenario_id": data.get("grid_analytics_scenario_id"),
            "grid_analytics_auth_header": request.headers.get("Authorization") or data.get("grid_analytics_auth_header"),
            "grid_analytics_cookie_header": request.headers.get("Cookie") or data.get("grid_analytics_cookie_header"),
            "drive_rows": data.get("drive_rows") or data.get("network_logs"),
            "drive_rows_source": data.get("drive_rows_source") or data.get("network_logs_source"),
            "dt_replace_radius_m": float(data.get("dt_replace_radius_m", 25)),
            "output_folder": current_app.config["OUTPUT_FOLDER"],
        }

        return jsonify(svc.submit(app, cfg))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@lte_prediction_offset_bp.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    return jsonify(svc.get(job_id))


@lte_prediction_offset_bp.route("/result/<job_id>", methods=["GET"])
def result(job_id):
    return jsonify(svc.get(job_id))
