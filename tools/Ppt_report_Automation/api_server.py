"""Optional standalone server. Normal integration starts ML/app.py."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
from tools.Ppt_report_Automation.routes import ppt_report_bp

load_dotenv(ML_ROOT / '.env')
load_dotenv(Path(__file__).with_name('.env'))
app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = str(ML_ROOT / 'outputs')
CORS(app, origins='*')
app.register_blueprint(ppt_report_bp, url_prefix='/api/ppt-report')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5050))
    print(f'PPT API: http://localhost:{port}/api/ppt-report')
    app.run(host='0.0.0.0', port=port, debug=False)
