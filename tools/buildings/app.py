import os
import json
import logging
import traceback
import sys
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

import osmnx as ox
import geopandas as gpd
from shapely.wkt import loads as wkt_loads
from shapely import ops as shapely_ops
import sqlalchemy as db
from sqlalchemy.exc import OperationalError
from geoalchemy2 import Geometry
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# --- 1. Load Environment Variables ---
load_dotenv()

# --- 2. Application and Logger Setup ---
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. OSMnx Settings ---
ox.settings.timeout = 180
ox.settings.use_cache = True

# --- 4. Database Connection (region-aware: india + taiwan) ---
BASE_DIR = os.getcwd()
DB_CERT_PATH = os.path.join(BASE_DIR, 'ca.pem')

if not os.path.exists(DB_CERT_PATH):
    logger.critical(f"CRITICAL ERROR: Database certificate 'ca.pem' not found in {BASE_DIR}")
    logger.critical("The 'ca.pem' file must be in the same directory as this script.")
    sys.exit()

DB_CONNECT_ARGS = {
    'ssl': {
        'ca': DB_CERT_PATH
    }
}

DB_URI_INDIA = os.environ.get('DATABASE_URL')
DB_URI_TAIWAN = os.environ.get('DATABASE_URL_Taiwan')

if not DB_URI_INDIA:
    logger.critical("CRITICAL ERROR: 'DATABASE_URL' not found.")
    logger.critical("Make sure it is set in your .env file.")
    sys.exit()


def _build_engine(db_uri, label):
    if not db_uri:
        logger.warning(f"⚠️ {label}: DB URI not set, region will be unavailable.")
        return None
    try:
        eng = db.create_engine(
            db_uri,
            connect_args=DB_CONNECT_ARGS,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        with eng.connect() as conn:
            logger.info(f"✅ {label}: database connection successful (with SSL).")
        return eng
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Could not create {label} database engine: {e}")
        return None


engine = _build_engine(DB_URI_INDIA, "India")
if engine is None:
    sys.exit()

engine_taiwan = _build_engine(DB_URI_TAIWAN, "Taiwan")

ENGINE_DICT = {
    "india": engine,
    "taiwan": engine_taiwan,
}


def resolve_region(data):
    raw = (data.get("region") or data.get("country_code") or data.get("countryCode") or "india")
    normalized = str(raw).strip().lower()
    if normalized in ("tw", "twn"):
        return "taiwan"
    if normalized in ("in", "ind"):
        return "india"
    return normalized


def get_regional_engine(region):
    """Falls back to India if the requested region has no configured/available engine."""
    eng = ENGINE_DICT.get(region)
    if eng is None:
        logger.warning(f"⚠️ No engine available for region '{region}', falling back to India.")
        return ENGINE_DICT["india"]
    return eng


# --- 5. Helper Functions ---

def _swap_lat_lon_if_needed(polygon):
    """
    OSM/shapely expects (lon, lat). Some callers (e.g. Taiwan region_wkt pulled
    straight from map_regions) send (lat, lon) instead, which OSM rejects with
    a nonsensical bounding box. Detect and fix it the same way routes.py does.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    needs_swap = False
    if min_y < -90 or max_y > 90:
        needs_swap = True
    elif min_y > 60 and min_x < 60:
        needs_swap = True
    if needs_swap:
        logger.info("🔄 Detected Lat/Lon input, swapping to Lon/Lat...")
        return shapely_ops.transform(lambda x, y: (y, x), polygon), True
    return polygon, False


def parse_geometry(data):
    """Parse geometry from request data"""
    logger.info(f"📥 Received data keys: {list(data.keys())}")

    if 'wkt' in data or 'WKT' in data:
        wkt = data.get('wkt') or data.get('WKT')
        logger.info(f"Parsing as WKT: {wkt[:100]}...")
        polygon = wkt_loads(wkt)
        polygon, _ = _swap_lat_lon_if_needed(polygon)
        return polygon
    else:
        raise ValueError("No valid geometry found")


def fetch_buildings(polygon, data, region="india"):
    """Fetch buildings from OpenStreetMap and save to DB"""
    save_engine = get_regional_engine(region)
    
    if not polygon.is_valid:
        logger.warning("Invalid polygon, attempting to fix...")
        polygon = polygon.buffer(0)
    
    logger.info(f"🌍 Polygon bounds: {polygon.bounds}")
    
    bounds = polygon.bounds
    width_deg = bounds[2] - bounds[0]
    height_deg = bounds[3] - bounds[1]
    width_m = width_deg * 111000
    height_m = height_deg * 111000
    logger.info(f"📏 Approximate size: {width_m:.1f}m × {height_m:.1f}m")
    
    logger.info("🔍 Fetching from OpenStreetMap...")
    
    try:
        buildings = ox.features_from_polygon(polygon, tags={"building": True, "residential": True})
        logger.info(f"📦 Fetched {len(buildings)} features from OSM")
        
        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
        logger.info(f"🏠 Filtered to {len(buildings)} building polygons")
        
        if buildings.empty:
            logger.warning("⚠️ No buildings found")
            return None, 0
            
        # --- OPTIMIZED DATABASE SAVE ---
        if save_engine:
            try:
                area_name = data.get('Name')
                project_id = data.get('project_id')

                if not area_name or not project_id:
                    logger.error("❌ Cannot save: 'Name' or 'project_id' missing from request.")
                    raise ValueError("'Name' and 'project_id' are required to save.")

                logger.info("Preparing building data for database insertion...")
                
                # Explode MultiPolygons
                buildings_exploded = buildings.explode(index_parts=True, ignore_index=True)
                total_polygons = len(buildings_exploded)
                logger.info(f"Exploded MultiPolygons. Total polygons to save: {total_polygons}")
                
                # OPTIMIZATION 1: Prepare all data in memory first (vectorized operations)
                logger.info("Converting geometries to WKT...")
                buildings_exploded['wkt_4326'] = buildings_exploded.geometry.to_wkt()
                
                # OPTIMIZATION 2: Fast area calculation (skip CRS conversion for speed)
                # Use approximate area in square degrees (faster than reprojection)
                logger.info("Calculating areas...")
                buildings_exploded['calc_area'] = buildings_exploded.geometry.area
                
                # OPTIMIZATION 3: Pre-build values list efficiently
                logger.info("Preparing values for bulk insert...")
                values_list = [
                    (area_name, row.wkt_4326, project_id, row.calc_area)
                    for row in buildings_exploded.itertuples()
                ]

                logger.info(f"Starting BULK database insert for {len(values_list)} polygons...")
                
                # OPTIMIZATION 4: Use multi-value INSERT for maximum speed
                raw_conn = save_engine.raw_connection()
                try:
                    cursor = raw_conn.cursor()
                    
                    # SUPER FAST: Disable autocommit and use batched multi-value INSERT
                    cursor.execute("SET autocommit=0")
                    cursor.execute("SET unique_checks=0")
                    cursor.execute("SET foreign_key_checks=0")
                    
                    batch_size = 1000  # Larger batches = faster (increased from 500)
                    total_inserted = 0
                    
                    for i in range(0, len(values_list), batch_size):
                        batch = values_list[i:i + batch_size]
                        
                        # Build multi-value INSERT statement
                        placeholders = "(%s, ST_GeomFromText(%s, 4326), %s, %s)"
                        values_str = ", ".join([placeholders] * len(batch))
                        
                        insert_query = f"""
                            INSERT INTO tbl_savepolygon (name, region, project_id, area)
                            VALUES {values_str}
                        """
                        
                        # Flatten the batch values
                        flat_values = [item for sublist in batch for item in sublist]
                        
                        cursor.execute(insert_query, flat_values)
                        total_inserted += len(batch)
                        
                        if total_inserted % 1000 == 0:
                            logger.info(f"Inserted {total_inserted}/{len(values_list)} polygons...")
                    
                    # Re-enable checks and commit
                    cursor.execute("SET unique_checks=1")
                    cursor.execute("SET foreign_key_checks=1")
                    raw_conn.commit()
                    cursor.close()
                    logger.info(f"✅ Committed all {total_inserted} polygons to database.")
                finally:
                    raw_conn.close()
                    
                logger.info(f"✅ Successfully saved {total_polygons} polygons to database in BULK.")
                
            except OperationalError as db_err:
                logger.error(f"❌ Database Error during save: {db_err}")
                logger.error(traceback.format_exc())
            except Exception as db_e:
                logger.error(f"❌ Database save error: {db_e}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("Database engine not configured. Skipping save.")

        geojson_str = buildings.to_json()
        geojson = json.loads(geojson_str)
        
        return geojson, len(buildings)
        
    except Exception as e:
        if "No matching features" in str(e) or "InsufficientResponseError" in str(type(e).__name__):
            logger.warning(f"⚠️ No buildings found in OpenStreetMap for this area")
            return None, 0
        else:
            raise


# --- 6. API Routes ---

@app.route('/', methods=['GET'])
def home():
    """Home endpoint to show service is running"""
    return jsonify({'service': 'Building Extraction Service', 'status': 'running', 'version': '3.0.0'})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database': {
            'india': 'connected' if ENGINE_DICT.get('india') else 'disconnected',
            'taiwan': 'connected' if ENGINE_DICT.get('taiwan') else 'disconnected',
        }
    })

@app.route('/api/generate-buildings', methods=['POST'])
def generate_buildings():
    """Main endpoint to generate buildings"""
    
    logger.info("=" * 60)
    logger.info("🚀 NEW REQUEST: /api/generate-buildings")
    logger.info("=" * 60)
    
    try:
        data = request.get_json()
        logger.info(f"📊 Request data type: {type(data)}")
        
        if not data:
            logger.error("❌ No data provided")
            return jsonify({'Status': 0, 'Message': 'No data provided'}), 400
        
        if 'Name' not in data or 'project_id' not in data:
            logger.warning("Request is missing 'Name' or 'project_id'.")
            return jsonify({'Status': 0, 'Message': "Input JSON must contain 'WKT', 'Name', and 'project_id' keys."}), 400

        region = resolve_region(data)
        logger.info(f"🌏 Region resolved: {region}")

        try:
            logger.info("🔄 Parsing geometry...")
            polygon = parse_geometry(data)
            logger.info(f"✅ Geometry parsed: {polygon.geom_type}")
        except Exception as e:
            logger.error(f"❌ Geometry parsing error: {str(e)}")
            return jsonify({'Status': 0, 'Message': f'Invalid geometry: {str(e)}'}), 400

        logger.info("🏗️ Fetching buildings from OpenStreetMap...")

        try:
            geojson, count = fetch_buildings(polygon, data, region=region)
            
            if count == 0 or geojson is None:
                return jsonify({
                    'Status': 0,
                    'Message': 'No buildings found in OpenStreetMap for this area.',
                    'Data': {'type': 'FeatureCollection', 'features': []}
                }), 200
            
            logger.info(f"✅ Successfully fetched {count} buildings")
            
            return jsonify({
                'Status': 1,
                'Message': f'Successfully fetched {count} buildings',
                'Data': geojson,
                'Stats': { 'total_buildings': count }
            }), 200
            
        except Exception as e:
            logger.error(f"❌ OSM fetch error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'Status': 0, 'Message': f'Error fetching buildings: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"💥 UNEXPECTED ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'Status': 0, 'Message': f'Server error: {str(e)}'}), 500

@app.route('/api/test-polygon', methods=['GET'])
def test():
    """Test endpoint with known working polygon (Delhi)"""
    sample_data = {
        "WKT": "POLYGON((77.2090 28.6139, 77.2100 28.6139, 77.2100 28.6149, 77.2090 28.6149, 77.2090 28.6139))",
        "project_id": 999,
        "Name": "Test Area"
    }
    
    try:
        polygon = parse_geometry(sample_data)
        geojson, count = fetch_buildings(polygon, sample_data)
        
        return jsonify({
            'Status': 1,
            'Message': f'Test successful - {count} buildings found',
            'Data': geojson or {'type': 'FeatureCollection', 'features': []}
        })
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        return jsonify({'Status': 0, 'Message': f'Test failed: {str(e)}'}), 500

# --- 7. Run the Application ---
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" Building Extraction Service (OPTIMIZED)")
    print(f" Running on: http://localhost:5001")
    print(f" India DB:  {'connected' if ENGINE_DICT.get('india') else 'NOT CONNECTED'}")
    print(f" Taiwan DB: {'connected' if ENGINE_DICT.get('taiwan') else 'NOT CONNECTED'}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)