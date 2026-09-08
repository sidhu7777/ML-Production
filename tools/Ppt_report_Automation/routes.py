"""PPT API shared by the ML app and standalone server."""
import os
import tempfile
from pathlib import Path
from flask import Blueprint, current_app, jsonify, request, send_file, url_for
from werkzeug.exceptions import BadRequest

ppt_report_bp = Blueprint('ppt_report', __name__)


def generate_ppt_for_project(**kwargs):
    from .report_ppt_generator import generate_ppt_for_project as generate
    return generate(**kwargs)


def _output_dir():
    root = current_app.config.get('OUTPUT_FOLDER') or Path(__file__).resolve().parents[2] / 'outputs'
    return Path(root).resolve() / 'ppt_reports'


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f'{name} must be an integer >= {minimum}')
    try:
        result = int(value)
    except ValueError:
        raise ValueError(f'{name} must be an integer >= {minimum}') from None
    if result < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return result


@ppt_report_bp.get('/health')
def health_check():
    return jsonify(status='online', service='PowerPoint Report Generator API', version='1.0.0')


@ppt_report_bp.post('/generate')
def generate_ppt_endpoint():
    if not request.is_json:
        return jsonify(status='error', message='Content-Type must be application/json'), 415
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('JSON body must be an object')
        project_id = _integer(data.get('project_id'), 'project_id')
        user_id = _integer(data.get('user_id', 0), 'user_id', 0)
        session_ids = data.get('session_ids')
        if session_ids is not None:
            if isinstance(session_ids, str):
                session_ids = session_ids.split(',')
            if not isinstance(session_ids, list) or not session_ids:
                raise ValueError('session_ids must be a non-empty array or comma-separated string')
            session_ids = [_integer(item, 'session_ids') for item in session_ids]
        country_code = data.get('country_code', 'taiwan')
        region = data.get('region', country_code)
        for name, value in (('country_code', country_code), ('region', region)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a non-empty string')
        locked_bands = data.get('locked_bands')
        if locked_bands is not None and not (isinstance(locked_bands, str) or
                isinstance(locked_bands, list) and all(isinstance(b, str) for b in locked_bands)):
            raise ValueError('locked_bands must be a string or array of strings')
    except (ValueError, BadRequest) as exc:
        return jsonify(status='error', message=str(exc)), 400

    temporary_path = None
    try:
        directory = _output_dir()
        directory.mkdir(parents=True, exist_ok=True)
        filename = f'Mobility_DT_Project_{project_id}.pptx'
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.pptx', delete=False) as tmp:
            temporary_path = tmp.name
        generate_ppt_for_project(project_id=project_id, session_ids=session_ids,
            user_id=user_id, country_code=country_code.strip(), region=region.strip(),
            locked_bands=locked_bands, output_path=temporary_path)
        os.replace(temporary_path, directory / filename)
        return jsonify(status='success', message='PowerPoint presentation generated successfully',
            project_id=project_id, output_file=filename,
            download_url=url_for('ppt_report.download_ppt_endpoint', project_id=project_id))
    except Exception:
        current_app.logger.exception('Error generating PPT for project %s', project_id)
        return jsonify(status='error', message='PowerPoint generation failed. Check the ML server logs.'), 500
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


@ppt_report_bp.get('/download/<int:project_id>')
def download_ppt_endpoint(project_id):
    filename = f'Mobility_DT_Project_{project_id}.pptx'
    filepath = _output_dir() / filename
    if not filepath.is_file():
        return jsonify(status='error', message='Presentation not found. Generate it first.'), 404
    return send_file(filepath, as_attachment=True, download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation', max_age=0)
