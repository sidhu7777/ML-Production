from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
from werkzeug.exceptions import HTTPException

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
print(f"[PYTHON_BOOT] lte_patch=2026-06-10-latlon-trace loky_max_cpu_count={os.environ.get('LOKY_MAX_CPU_COUNT')}", flush=True)

# Import config, blueprints, and db
from config import config
from tools.buildings.routes import buildings_bp
from tools.cell_site.routes import cell_site_bp
from tools.prediction.routes import prediction_bp
from tools.area_breakup.routes import area_breakup_bp
from tools.report.routes import report_bp
from tools.lte_prediction.routes import lte_prediction_bp
from tools.lte_prediction_optimised.routes import lte_prediction_op
from tools.lte_tilt_recommandation.routes import rf_optimization_bp

from extensions import db
from flask_migrate import Migrate

migrate = Migrate()


# ==========================================================
# CREATE APP
# ==========================================================
def create_app(config_name='default'):

    app = Flask(__name__)

    # ---------------- LOGGING ----------------
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    app.logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Silence noisy logs
    for noisy in [
        "botocore", "boto3", "s3transfer", "httpx",
        "urllib3", "matplotlib", "PIL", "groq", "asyncio",
        "werkzeug",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ---------------- CONFIG ----------------
    env_config = config.get(config_name, config['default'])
    app.config.from_object(env_config)

    base_dir = os.path.dirname(__file__)

    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
    app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads')
    app.config['OUTPUT_FOLDER'] = os.path.join(base_dir, 'outputs')

    if hasattr(env_config, 'init_app'):
        env_config.init_app()

    # ---------------- EXTENSIONS ----------------
    db.init_app(app)
    migrate.init_app(app, db)

    # ---------------- CORS ----------------
    CORS(
        app,
        origins=["*", "http://localhost:5173", "https://singnaltracker.netlify.app"],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Accept"],
        supports_credentials=True,
        max_age=3600
    )

    # ---------------- CREATE FOLDERS (SAFE) ----------------
    with app.app_context():
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

    # ---------------- BLUEPRINTS ----------------
    app.register_blueprint(buildings_bp, url_prefix='/api/buildings')
    app.register_blueprint(cell_site_bp, url_prefix='/api/cell-site')
    app.register_blueprint(prediction_bp, url_prefix='/api/prediction')
    app.register_blueprint(area_breakup_bp, url_prefix='/api/area-breakup')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    app.register_blueprint(lte_prediction_bp, url_prefix="/api/lte-prediction")
    app.register_blueprint(lte_prediction_op, url_prefix="/api/lte-prediction-optimised")
    app.register_blueprint(rf_optimization_bp, url_prefix="/api/lte-tilt-recommandation")
    # ---------------- ROOT ----------------
    @app.route('/', methods=['GET'])
    def root():
        return jsonify({
            "message": "Python ML Backend is running",
            "services": {
                "buildings": "/api/buildings",
                "cell_site": "/api/cell-site",
                "prediction": "/api/prediction",
                "area_breakup": "/api/area-breakup",
                "report": "/api/report",
                "site_prediction": "/api/lte-prediction/run",
                "optimized_prediction": "/api/lte-prediction-optimised/run",
                "rf_optimization": "/api/lte-tilt-recommandation/optimize"
            }
        })

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'Python ML Backend',
            'lte_patch': '2026-06-10-latlon-trace',
            'loky_max_cpu_count': os.environ.get('LOKY_MAX_CPU_COUNT')
        }), 200

    # ---------------- ERROR HANDLERS ----------------
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({'error': 'File too large (max 100MB)'}), 413

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500

    @app.errorhandler(Exception)
    def handle_exception(e):
        if isinstance(e, HTTPException):
            return jsonify({
                "error": e.name,
                "message": e.description
            }), e.code

        app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True)

        try:
            db.session.rollback()
        except:
            pass

        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

    return app


# ==========================================================
# ENTRY POINT
# ==========================================================
app_env = os.getenv('FLASK_ENV', 'default')
app = create_app(app_env)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))

    app.run(
        host=os.getenv('HOST', '0.0.0.0'),
        port=port,
        debug=app.config.get('DEBUG', False),
        use_reloader=False
    )
