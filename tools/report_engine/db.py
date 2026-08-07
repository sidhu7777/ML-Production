import os
import pandas as pd
from sqlalchemy import create_engine, text, bindparam
from dotenv import load_dotenv
from datetime import datetime
from utils.python_bridge import get_bridge_client, PythonBridgeError

load_dotenv()

_ENGINE = None
_REGIONAL_ENGINES = {}


def _bridge_client():
    try:
        return get_bridge_client()
    except PythonBridgeError:
        return None


def _coerce_bridge_datetime(value):
    """
    The bridge (PythonBridgeController) currently serializes MySQL DATETIME
    columns as a raw {"IsValidDateTime": .., "Year": .., "Month": .., ...}
    object instead of an ISO date string (a C#-side MySqlDateTime struct
    being dumped field-by-field rather than converted first). pandas can't
    parse that shape, so it's decoded here on the python side before any
    downstream code sees it.
    """
    if not isinstance(value, dict) or "Year" not in value:
        return value
    if value.get("IsValidDateTime") is False:
        return None
    try:
        microsecond = int(value.get("Microsecond") or 0) or int(value.get("Millisecond") or 0) * 1000
        return datetime(
            int(value["Year"]), int(value["Month"]), int(value["Day"]),
            int(value.get("Hour") or 0), int(value.get("Minute") or 0), int(value.get("Second") or 0),
            microsecond,
        )
    except (KeyError, ValueError, TypeError):
        return None


def _normalize_bridge_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(_coerce_bridge_datetime)
    return df


def init_engine(engine):
    """
    Initialize a shared SQLAlchemy engine from the main app.
    """
    global _ENGINE
    _ENGINE = engine


def _normalize_region(region: str | None = None, country_code: str | None = None) -> str:
    raw = str(region or country_code or "").strip().lower()
    if raw in {"tw", "twn", "taiwan"}:
        return "taiwan"
    if raw in {"in", "ind", "india"}:
        return "india"
    return raw or "india"


def _engine_url_for_region(region: str) -> str | None:
    if region == "taiwan":
        return os.getenv("DATABASE_URL_Taiwan") or os.getenv("DATABASE_URL_TAIWAN")
    return os.getenv("DATABASE_URL")


def get_engine(region: str | None = None, country_code: str | None = None):
    """
    Return the shared/default engine or create a regional engine from env.
    """
    global _ENGINE
    normalized_region = _normalize_region(region, country_code)
    if normalized_region == "india" and _ENGINE is not None:
        return _ENGINE

    if normalized_region in _REGIONAL_ENGINES:
        return _REGIONAL_ENGINES[normalized_region]

    db_url = _engine_url_for_region(normalized_region)
    if not db_url:
        raise RuntimeError(f"Missing database URL for report engine region={normalized_region}")

    # Keep connections healthy for long-running jobs
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
        connect_args={"connect_timeout": 30},
    )
    if normalized_region == "india":
        _ENGINE = engine
    _REGIONAL_ENGINES[normalized_region] = engine
    return engine


def _connect(region: str | None = None, country_code: str | None = None):
    return get_engine(region=region, country_code=country_code).connect()


# =====================================================
# DEBUG / INSPECTION HELPERS
# =====================================================

def list_tables():
    with _connect() as conn:
        rows = conn.execute(text("SHOW TABLES")).fetchall()
    print("Tables in database:")
    for t in rows:
        print("-", t[0])


def describe_table(table_name: str):
    with _connect() as conn:
        rows = conn.execute(text(f"DESCRIBE {table_name}")).fetchall()
    print(f"\nColumns in {table_name}:")
    for col in rows:
        print(col)


# =====================================================
# CORE DATA ACCESS FUNCTIONS
# =====================================================

def get_project_by_id(
    project_id: int,
    conn=None,
    region: str | None = None,
    country_code: str | None = None,
):
    bridge = _bridge_client()
    if bridge is not None:
        project = bridge.get_project(project_id, region=region, country_code=country_code)
        return dict(project) if project else None

    close_conn = False
    if conn is None:
        conn = _connect(region=region, country_code=country_code)
        close_conn = True

    try:
        query = text("""
            SELECT *
            FROM tbl_project
            WHERE id = :project_id
        """)
        row = conn.execute(query, {"project_id": project_id}).mappings().first()
        return dict(row) if row else None
    finally:
        if close_conn:
            conn.close()


def get_network_logs_for_sessions(
    session_ids: list[int],
    conn=None,
    project_id: int | None = None,
    provider: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    region: str | None = None,
    country_code: str | None = None,
) -> pd.DataFrame:
    if not session_ids:
        return pd.DataFrame()

    bridge = _bridge_client()
    if bridge is not None:
        df = bridge.get_report_network_logs(
            session_ids,
            project_id=project_id,
            provider=provider,
            start_date=start_date,
            end_date=end_date,
            region=region,
            country_code=country_code,
        )
        return _normalize_bridge_datetime_columns(df, ["timestamp"])

    close_conn = False
    if conn is None:
        conn = _connect(region=region, country_code=country_code)
        close_conn = True

    try:
        query = text("""
            SELECT *
            FROM tbl_network_log
            WHERE session_id IN :session_ids
        """).bindparams(bindparam("session_ids", expanding=True))

        df = pd.read_sql(query, conn, params={"session_ids": session_ids})
        df.attrs["report_data_source"] = "direct_db_raw"
        df.attrs["report_prefiltered"] = False
        return df
    finally:
        if close_conn:
            conn.close()


def get_project_regions(
    project_id: int,
    conn=None,
    region: str | None = None,
    country_code: str | None = None,
) -> list[dict]:
    bridge = _bridge_client()
    if bridge is not None:
        rows = bridge.get_project_regions(project_id, region=region, country_code=country_code)
        return [dict(r) for r in rows]

    close_conn = False
    if conn is None:
        conn = _connect(region=region, country_code=country_code)
        close_conn = True

    try:
        query = text("""
            SELECT
                id,
                name,
                ST_AsText(region) AS region_wkt
            FROM map_regions
            WHERE tbl_project_id = :project_id
              AND status = 1
        """)
        rows = conn.execute(query, {"project_id": project_id}).mappings().all()
        return [dict(r) for r in rows]
    finally:
        if close_conn:
            conn.close()


def get_user_thresholds(
    user_id: int,
    debug: bool = False,
    conn=None,
    region: str | None = None,
    country_code: str | None = None,
) -> dict | None:
    bridge = _bridge_client()
    if bridge is not None:
        data = bridge.get_user_thresholds(user_id, region=region, country_code=country_code)
        if debug:
            print("\n================ BRIDGE THRESHOLD ROW =================")
            print(f"user_id = {user_id}")
            if not data:
                print("NO ROW RETURNED FROM BRIDGE")
            else:
                for k, v in data.items():
                    print(f"{k}: {repr(v)}")
            print("=======================================================\n")
        return dict(data) if data else None

    close_conn = False
    if conn is None:
        conn = _connect(region=region, country_code=country_code)
        close_conn = True

    try:
        query = text("""
            SELECT *
            FROM thresholds
            WHERE user_id = :user_id
            LIMIT 1
        """)
        row = conn.execute(query, {"user_id": user_id}).mappings().first()
        data = dict(row) if row else None
    finally:
        if close_conn:
            conn.close()

    if debug:
        print("\n================ DB THRESHOLD ROW =================")
        print(f"user_id = {user_id}")
        if not data:
            print("NO ROW RETURNED FROM DB")
            return None
        for k, v in data.items():
            print(f"{k}: {repr(v)}")
        print("===================================================\n")

    return data


def get_user_by_id(
    user_id: int,
    conn=None,
    region: str | None = None,
    country_code: str | None = None,
) -> dict | None:
    bridge = _bridge_client()
    if bridge is not None:
        row = bridge.get_user(user_id, region=region, country_code=country_code)
        return dict(row) if row else None

    close_conn = False
    if conn is None:
        conn = _connect(region=region, country_code=country_code)
        close_conn = True

    try:
        query = text("""
            SELECT *
            FROM tbl_user
            WHERE id = :user_id
            LIMIT 1
        """)
        row = conn.execute(query, {"user_id": user_id}).mappings().first()
        return dict(row) if row else None
    finally:
        if close_conn:
            conn.close()


def update_project_download_path(
    project_id: int,
    download_path: str,
    conn=None,
    region: str | None = None,
    country_code: str | None = None,
) -> None:
    """
    Update tbl_project.Download_path for the given project.
    """
    bridge = _bridge_client()
    if bridge is not None:
        bridge.update_project_download_path(
            project_id,
            download_path,
            region=region,
            country_code=country_code,
        )
        return

    close_conn = False
    if conn is None:
        conn = _connect(region=region, country_code=country_code)
        close_conn = True

    try:
        query = text("""
            UPDATE tbl_project
            SET Download_path = :download_path
            WHERE id = :project_id
        """)
        conn.execute(query, {"download_path": download_path, "project_id": project_id})
        conn.commit()
    finally:
        if close_conn:
            conn.close()


def get_sessions_by_ids(
    session_ids: list[int],
    region: str | None = None,
    country_code: str | None = None,
) -> pd.DataFrame:
    if not session_ids:
        return pd.DataFrame()

    bridge = _bridge_client()
    if bridge is not None:
        df = bridge.get_sessions(session_ids, region=region, country_code=country_code)
        return _normalize_bridge_datetime_columns(df, ["start_time", "end_time"])

    with _connect(region=region, country_code=country_code) as conn:
        query = text("""
            SELECT id, start_time, end_time, distance
            FROM defaultdb.tbl_session
            WHERE id IN :session_ids
            ORDER BY start_time
        """).bindparams(bindparam("session_ids", expanding=True))
        return pd.read_sql(query, conn, params={"session_ids": session_ids})
