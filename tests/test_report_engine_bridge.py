import pandas as pd

from tools.report_engine import db


class _BridgeStub:
    def __init__(self):
        self.updated = []

    def get_project(self, project_id):
        return {"id": project_id, "ref_session_id": "11,12", "project_name": "Bridge Project"}

    def get_project_regions(self, project_id):
        return [{"id": 1, "name": "R1", "region_wkt": "POLYGON((0 0,1 0,1 1,0 0))"}]

    def get_report_network_logs(self, session_ids, limit=50000, project_id=None, provider=None, start_date=None, end_date=None):
        return pd.DataFrame(
            [{"session_id": session_ids[0], "lat": 1.0, "lon": 2.0, "rsrp": -95.0, "timestamp": "2026-03-25T10:00:00"}]
        )

    def get_user(self, user_id):
        return {"id": user_id, "email": "bridge@example.com"}

    def get_user_thresholds(self, user_id):
        return {"user_id": user_id, "rsrp_json": "[]"}

    def update_project_download_path(self, project_id, download_path):
        self.updated.append((project_id, download_path))
        return True

    def get_sessions(self, session_ids):
        return pd.DataFrame(
            [{"id": session_ids[0], "start_time": "2026-03-25 10:00:00", "end_time": "2026-03-25 11:00:00", "distance": 12.5}]
        )


def test_report_db_helpers_use_bridge_when_available(monkeypatch):
    bridge = _BridgeStub()
    monkeypatch.setattr(db, "_bridge_client", lambda: bridge)

    project = db.get_project_by_id(196)
    regions = db.get_project_regions(196)
    logs = db.get_network_logs_for_sessions([11, 12])
    user = db.get_user_by_id(13)
    thresholds = db.get_user_thresholds(13)
    sessions = db.get_sessions_by_ids([11])
    db.update_project_download_path(196, "/api/report/download/abc")

    assert project["project_name"] == "Bridge Project"
    assert regions[0]["name"] == "R1"
    assert not logs.empty
    assert user["email"] == "bridge@example.com"
    assert thresholds["user_id"] == 13
    assert float(sessions.loc[0, "distance"]) == 12.5
    assert bridge.updated == [(196, "/api/report/download/abc")]
