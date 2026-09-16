import multiprocessing

from flask import Blueprint, current_app, jsonify, request

from .services import LTEPredictionOffsetService


# Service edge for the H Cell scope: the RSRP a cell must still deliver for the
# pixel to count as its coverage. Used only to solve each cell's own radius from
# its link budget, so the caller does not have to supply a distance per cell.
CELL_EDGE_RSRP_DBM = -110.0

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


def _prediction_scope(data):
    """('radius' | 'hcell', cell_edge_rsrp_dbm).

    radius : the operator fixes one distance and every cell is evaluated over it.
    hcell  : the model decides each cell's coverage itself, solving the distance
             at which that cell's own RSRP falls to the service edge from its
             link budget - tx power, real antenna gain, height, frequency. A
             700 MHz macro and a 3.5 GHz cell therefore get different radii
             with no per-cell input from the caller.

    The edge level is a planning constant, not something the operator has to
    supply per run, so it defaults to CELL_EDGE_RSRP_DBM. A caller may still
    override it to model a different service definition.
    """
    raw = str(data.get("prediction_scope") or data.get("predictionScope") or "radius").strip().lower()
    if raw.replace("_", "").replace("-", "").replace(" ", "") not in {"hcell", "cell", "cellh"}:
        return "radius", None

    edge = data.get("cell_edge_rsrp_dbm", data.get("cellEdgeRsrpDbm"))
    if edge is None or str(edge).strip() == "":
        return "hcell", CELL_EDGE_RSRP_DBM
    edge = float(edge)
    if not -160.0 <= edge <= -40.0:
        raise ValueError(f"cell_edge_rsrp_dbm={edge} is outside the plausible RSRP range -160..-40 dBm")
    return "hcell", edge


@lte_prediction_offset_bp.route("/run", methods=["POST"])
def run_prediction():
    try:
        data = request.get_json() or {}
        app = current_app._get_current_object()
        cpu_count = multiprocessing.cpu_count()

        scope, cell_edge_rsrp_dbm = _prediction_scope(data)

        cfg = {
            "project_id": int(data["project_id"]),
            "prediction_scope": scope,
            "session_ids": data["session_ids"],
            "region": _resolve_region(data),
            "country_code": data.get("country_code") or data.get("countryCode"),
            "polygon_ids": data.get("polygon_ids") or data.get("polygonIds"),
            "operator": str(data.get("operator", "") or "").strip(),
            # Radius scope: this value applies to every cell. H Cell scope: it
            # becomes a lower bound on the solved per-cell radius.
            "radius_m": float(data.get("radius", data.get("radius_m", 500))),
            # None keeps the flat radius; a value switches on the per-cell
            # link-budget radius in _run_raw_surface.
            "cell_edge_rsrp_dbm": cell_edge_rsrp_dbm,
            "cell_edge_radius_cap_m": data.get("cell_edge_radius_cap_m") or data.get("cellEdgeRadiusCapM"),
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
