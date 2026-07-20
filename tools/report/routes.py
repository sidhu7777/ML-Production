from flask import Blueprint, request, jsonify, current_app, send_file, Response, stream_with_context
import os
import threading
import uuid
import json
import tempfile
from queue import Queue, Empty

from tools.report_engine.main import main as generate_report
from tools.report_engine.db import get_project_by_id
from tools.report_engine.playwright_utils import check_chromium_rendering
from extensions import db

report_bp = Blueprint("report", __name__)
REPORT_JOBS = {}
REPORT_JOBS_LOCK = threading.Lock()
REPORT_SUBSCRIBERS = {}
REPORT_SUBSCRIBERS_LOCK = threading.Lock()


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _reports_root() -> str:
    root_path = getattr(current_app, "root_path", None)
    if not root_path:
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root_path, "data", "reports")


def _report_dir(report_id: str) -> str:
    return os.path.join(_reports_root(), report_id)


def _report_status_path(report_id: str) -> str:
    return os.path.join(_report_dir(report_id), "status.json")


def _write_job_status(report_id: str, state: dict):
    try:
        report_dir = _report_dir(report_id)
        os.makedirs(report_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix="status.", suffix=".json", dir=report_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str)
        os.replace(tmp_path, _report_status_path(report_id))
    except Exception as exc:
        current_app.logger.warning("[Report] Failed to persist status for %s: %s", report_id, exc)


def _read_job_status(report_id: str):
    try:
        path = _report_status_path(report_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return dict(state) if isinstance(state, dict) else None
    except Exception as exc:
        current_app.logger.warning("[Report] Failed to read status for %s: %s", report_id, exc)
        return None


def _set_job(report_id: str, **updates):
    with REPORT_JOBS_LOCK:
        state = REPORT_JOBS.get(report_id, {})
        state.update(updates)
        REPORT_JOBS[report_id] = state
        persisted = dict(state)
    _write_job_status(report_id, persisted)
    return persisted


def _get_job(report_id: str):
    with REPORT_JOBS_LOCK:
        state = REPORT_JOBS.get(report_id)
        if state:
            return dict(state)

    persisted = _read_job_status(report_id)
    if persisted:
        with REPORT_JOBS_LOCK:
            REPORT_JOBS[report_id] = dict(persisted)
        return persisted

    return None


def _status_payload(report_id: str, job: dict):
    payload = {
        "status": job.get("status", "processing"),
        "report_id": report_id,
        "project_id": job.get("project_id"),
        "user_id": job.get("user_id"),
    }
    if job.get("download_url"):
        payload["download_url"] = job["download_url"]
    if job.get("error"):
        payload["error"] = job["error"]
    if job.get("message"):
        payload["message"] = job["message"]
    return payload


def _report_pdf_path(report_id: str) -> str:
    return os.path.join(_report_dir(report_id), "report.pdf")


def _subscribe(report_id: str):
    q = Queue()
    with REPORT_SUBSCRIBERS_LOCK:
        REPORT_SUBSCRIBERS.setdefault(report_id, []).append(q)
    return q


def _unsubscribe(report_id: str, q: Queue):
    with REPORT_SUBSCRIBERS_LOCK:
        listeners = REPORT_SUBSCRIBERS.get(report_id, [])
        if q in listeners:
            listeners.remove(q)
        if not listeners and report_id in REPORT_SUBSCRIBERS:
            del REPORT_SUBSCRIBERS[report_id]


def _publish_status_event(report_id: str):
    job = _get_job(report_id)
    if not job:
        return
    payload = _status_payload(report_id, job)
    with REPORT_SUBSCRIBERS_LOCK:
        listeners = list(REPORT_SUBSCRIBERS.get(report_id, []))
    for q in listeners:
        q.put(payload)


def background_report_task(app, project_id, user_id, report_id):
    with app.app_context():
        try:
            _set_job(
                report_id,
                status="processing",
                project_id=project_id,
                user_id=user_id,
                message="Report generation is running",
            )
            _publish_status_event(report_id)
            current_app.logger.info(
                f"[Report] Starting generation: project_id={project_id}, user_id={user_id}, report_id={report_id}"
            )
            generate_report(
                project_id=project_id,
                user_id=user_id,
                report_id=report_id,
                db_engine=db.engine,
            )
            download_url = f"/api/report/download/{report_id}"
            _set_job(
                report_id,
                status="ready",
                project_id=project_id,
                user_id=user_id,
                download_url=download_url,
                message="Report generation completed",
            )
            _publish_status_event(report_id)
            current_app.logger.info(
                f"[Report] Completed generation: report_id={report_id}"
            )
        except Exception as e:
            _set_job(report_id, status="failed", error=str(e), message="Report generation failed")
            _publish_status_event(report_id)
            current_app.logger.exception(
                f"[Report] Failed generation: report_id={report_id}"
            )


@report_bp.route("/generate", methods=["POST"])
def generate():
    data = request.get_json() or {}

    project_id = _safe_int(data.get("project_id") or data.get("Project_id"))
    user_id = _safe_int(data.get("user_id") or data.get("User_id"))

    if not project_id:
        return jsonify({"error": "project_id is required"}), 400
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    report_id = str(uuid.uuid4())
    _set_job(
        report_id,
        status="processing",
        project_id=project_id,
        user_id=user_id,
        download_url=None,
        message="Report generation queued",
    )
    app = current_app._get_current_object()

    thread = threading.Thread(
        target=background_report_task,
        args=(app, project_id, user_id, report_id),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "message": "Report generation started",
        "status": "processing",
        "project_id": project_id,
        "user_id": user_id,
        "report_id": report_id
    }), 202


@report_bp.route("/render-health", methods=["GET"])
def render_health():
    ok, detail = check_chromium_rendering()
    status_code = 200 if ok else 503
    return jsonify({
        "status": "healthy" if ok else "unhealthy",
        "chromium_rendering": ok,
        "detail": detail,
    }), status_code


@report_bp.route("/status/<report_id>", methods=["GET"])
def status(report_id):
    job = _get_job(report_id)
    if not job:
        if os.path.exists(_report_pdf_path(report_id)):
            return jsonify({
                "status": "ready",
                "report_id": report_id,
                "download_url": f"/api/report/download/{report_id}",
            }), 200

        return jsonify({
            "status": "not_found",
            "report_id": report_id,
        }), 404

    payload = {
        "status": job.get("status", "processing"),
        "report_id": report_id,
        "project_id": job.get("project_id"),
        "user_id": job.get("user_id"),
    }
    if job.get("download_url"):
        payload["download_url"] = job["download_url"]
    if job.get("error"):
        payload["error"] = job["error"]
    if job.get("message"):
        payload["message"] = job["message"]
    return jsonify(payload), 200


@report_bp.route("/events/<report_id>", methods=["GET"])
def events(report_id):
    job = _get_job(report_id)
    if not job:
        if os.path.exists(_report_pdf_path(report_id)):
            return jsonify({
                "status": "ready",
                "report_id": report_id,
                "download_url": f"/api/report/download/{report_id}",
            }), 200

        return jsonify({
            "status": "not_found",
            "report_id": report_id,
        }), 404

    terminal = {"ready", "failed"}

    @stream_with_context
    def event_stream():
        initial = _get_job(report_id)
        if initial and initial.get("status") in terminal:
            payload = _status_payload(report_id, initial)
            yield f"event: report_status\ndata: {json.dumps(payload)}\n\n"
            return

        q = _subscribe(report_id)
        try:
            while True:
                try:
                    payload = q.get(timeout=20)
                    yield f"event: report_status\ndata: {json.dumps(payload)}\n\n"
                    if payload.get("status") in terminal:
                        break
                except Empty:
                    # Keep the connection alive for proxies/load balancers.
                    yield ": keep-alive\n\n"
        finally:
            _unsubscribe(report_id, q)

    return Response(event_stream(), mimetype="text/event-stream")


@report_bp.route("/download/<report_id>", methods=["GET"])
def download(report_id):
    pdf_path = _report_pdf_path(report_id)

    if os.path.exists(pdf_path):
        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="drive_test_report.pdf",
        )

    return jsonify({"error": "Report not found"}), 404
