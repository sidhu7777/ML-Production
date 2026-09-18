from flask import Blueprint, request, jsonify, current_app
import traceback
from shapely import wkt, ops
from .services import BuildingService

buildings_bp = Blueprint("buildings", __name__)
service = BuildingService()

# ================== HELPER: SMART WKT PARSER ==================
def clean_and_load_wkt(raw_string):
    """
    1. Adds 'POLYGON' if missing.
    2. Detects Swapped Coordinates (Lat/Lon) and fixes them to (Lon/Lat).
    Returns: (geometry, was_swapped_flag)
    """
    if not raw_string:
        raise ValueError("WKT string is empty")

    # 1. Fix missing 'POLYGON' prefix
    clean_str = raw_string.strip()
    if not clean_str.upper().startswith("POLYGON"):
        if clean_str.startswith("(("):
            clean_str = f"POLYGON {clean_str}"
        else:
            raise ValueError("Input must be a POLYGON or ((...))")

    # Load Geometry
    poly_geom = wkt.loads(clean_str)

    # 2. Check bounds to detect Lat/Lon swap
    min_x, min_y, max_x, max_y = poly_geom.bounds
    
    needs_swap = False
    
    # Strict Check: Latitude > 90 is impossible
    if min_y < -90 or max_y > 90:
        needs_swap = True
    # Heuristic: If Y (Lat) is > 60 and X (Lon) is < 60, user likely meant Lat/Lon
    elif (min_y > 60 and min_x < 60): 
        needs_swap = True

    if needs_swap:
        print("🔄 Buildings: Detected Lat/Lon input. Swapping to Lon/Lat...")
        poly_geom = ops.transform(lambda x, y: (y, x), poly_geom)
        return poly_geom, True # True = We swapped it

    return poly_geom, False # False = Input was already correct


# -------------------------------------
# CORS preflight
# -------------------------------------
@buildings_bp.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response, 204


# -------------------------------------
# GENERATE + SAVE BUILDINGS
# -------------------------------------
@buildings_bp.route("/generate", methods=["POST"])
def generate_buildings():
    try:
        if not request.is_json:
            return jsonify({"Status": 0, "Message": "JSON body required"}), 400

        data = request.get_json()
        raw_wkt = data.get("WKT") or data.get("wkt")
        name = data.get("Name")
        project_id = data.get("project_id")
        
        # ✅ EXTRACT REGION (Defaults to india)
        region = str(data.get("region", "india")).lower()

        if not raw_wkt or not name or not project_id:
             return jsonify({"Status": 0, "Message": "WKT, Name, and project_id are required"}), 400

        # --- SMART PARSING ---
        try:
            polygon, was_swapped = clean_and_load_wkt(raw_wkt)
        except Exception as e:
             return jsonify({"Status": 0, "Message": f"Invalid WKT: {str(e)}"}), 400

        # ✅ Pass region to the service. Buildings + Phase-27 clutter
        # classification both belong at project creation time, run
        # concurrently, not two separate calls - see process_project_geo_setup.
        buildings_result, clutter_result = service.process_project_geo_setup(
            polygon, name, project_id, swap_output=was_swapped, region=region
        )

        if not buildings_result:
            return jsonify({
                "Status": 0,
                "Message": "No buildings found in this area"
            })

        geojson, extracted_count, saved_count = buildings_result
        _, clutter_summary = clutter_result

        return jsonify({
            "Status": 1,
            "Message": f"Extracted {extracted_count}, Saved {saved_count} buildings",
            "Data": geojson,
            "Stats": {
                "extracted": extracted_count,
                "saved_to_db": saved_count
            },
            "Clutter": clutter_summary,
            "input_format_detected": "Lat/Lon" if was_swapped else "Lon/Lat"
        })

    except Exception as e:
        current_app.logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "Status": 0,
            "Message": str(e)
        }), 500


# -------------------------------------
# TEST ENDPOINT
# -------------------------------------
@buildings_bp.route("/test", methods=["GET"])
def test():
    # Simple test case (Standard Lon/Lat)
    wkt_str = "POLYGON((77.2090 28.6139, 77.2100 28.6139, 77.2100 28.6149, 77.2090 28.6149, 77.2090 28.6139))"
    poly = wkt.loads(wkt_str)
    
    # Defaults to India for tests
    geojson, count, saved = service.process_buildings(poly, "Test Area", 999, swap_output=False, region="india")

    return jsonify({
        "Status": 1,
        "Message": "Test successful",
        "Extracted": count,
        "Saved": saved,
        "Data": geojson
    })