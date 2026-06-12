from flask import Blueprint, request, jsonify, send_file
from .services import LTEPredictionService_optimised

lte_prediction_op = Blueprint("lte_prediction_optimized", __name__)

service = LTEPredictionService_optimised()


# ==========================================================
# RUN OPTIMIZED PREDICTION
# ==========================================================
@lte_prediction_op.route("/run", methods=["POST"])
@lte_prediction_op.route("/optimized", methods=["POST"])
def run_optimized():
    data = request.get_json()

    # Removed "region" from here so it doesn't crash if missing
    required_fields = ["project_id", "radius", "grid_resolution","operator"]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"{field} is required"}), 400

    # Explicitly set a default if the frontend forgot it
    data["region"] = str(data.get("region", "india")).lower()

    result = service.submit(data)
    return jsonify(result)


# ==========================================================
# RUN OPTIMIZED PREDICTION FROM SAVED TILT RECOMMENDATIONS
# ==========================================================
@lte_prediction_op.route("/recommendation-optimized", methods=["POST"])
def run_recommendation_optimized():
    data = request.get_json() or {}

    required_fields = ["project_id", "radius", "grid_resolution"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"{field} is required"}), 400

    data["region"] = str(data.get("region", "india")).lower()

    result = service.submit_recommendation_optimization(data)
    return jsonify(result), 202


# ==========================================================
# CHECK STATUS
# ==========================================================
@lte_prediction_op.route("/status/<job_id>", methods=["GET"])
def status(job_id):

    job = service.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job)


# ==========================================================
# DOWNLOAD FILE
# ==========================================================
@lte_prediction_op.route("/download", methods=["GET"])
def download():

    file_path = request.args.get("file")

    if not file_path:
        return jsonify({"error": "file path required"}), 400

    return send_file(file_path, as_attachment=True)
