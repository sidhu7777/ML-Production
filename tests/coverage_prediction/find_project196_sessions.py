from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


ML_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ROOT))
load_dotenv(ML_ROOT / ".env")


def main() -> None:
    engine = create_engine(os.environ.get("DATABASE_URL") or os.environ.get("DB_URL"))
    cols = pd.read_sql(
        text(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'tbl_network_log' "
            "ORDER BY ORDINAL_POSITION"
        ),
        engine,
    )["COLUMN_NAME"].tolist()
    print("TBL_NETWORK_LOG_COLUMNS", cols)

    project_col = "project_id" if "project_id" in cols else ("tbl_project_id" if "tbl_project_id" in cols else None)
    session_col = "session_id" if "session_id" in cols else ("sessionId" if "sessionId" in cols else None)
    network_col = "network" if "network" in cols else ("Technology" if "Technology" in cols else None)
    if not project_col or not session_col:
        print(f"missing project/session columns project_col={project_col} session_col={session_col}")
        return

    where = [f"{project_col} = :project_id"]
    if network_col:
        where.append(f"({network_col} LIKE '%4G%' OR {network_col} LIKE '%LTE%')")
    query = (
        f"SELECT {session_col} AS session_id, COUNT(*) AS rows_count "
        "FROM tbl_network_log "
        f"WHERE {' AND '.join(where)} "
        f"GROUP BY {session_col} ORDER BY rows_count DESC LIMIT 30"
    )
    sessions = pd.read_sql(text(query), engine, params={"project_id": 196})
    print("\nPROJECT_196_SESSIONS")
    print(sessions.to_string(index=False))


if __name__ == "__main__":
    main()
