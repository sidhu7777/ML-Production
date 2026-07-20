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
    tables = pd.read_sql(
        text(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() "
            "AND (TABLE_NAME LIKE '%session%' OR TABLE_NAME LIKE '%project%' OR TABLE_NAME LIKE '%network%') "
            "ORDER BY TABLE_NAME"
        ),
        engine,
    )["TABLE_NAME"].tolist()
    print("TABLES")
    print("\n".join(tables))

    for table in tables:
        cols = pd.read_sql(
            text(
                "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table_name "
                "ORDER BY ORDINAL_POSITION"
            ),
            engine,
            params={"table_name": table},
        )["COLUMN_NAME"].tolist()
        joined = " ".join(cols).lower()
        if "project" in joined and "session" in joined:
            print(f"\nCANDIDATE {table}")
            print(cols)
            project_cols = [c for c in cols if "project" in c.lower()]
            session_cols = [c for c in cols if "session" in c.lower()]
            if project_cols and session_cols:
                pcol = project_cols[0]
                scol = session_cols[0]
                try:
                    sample = pd.read_sql(
                        text(
                            f"SELECT * FROM {table} "
                            f"WHERE {pcol} = :project_id "
                            "LIMIT 20"
                        ),
                        engine,
                        params={"project_id": 196},
                    )
                    print(sample.to_string(index=False))
                except Exception as exc:
                    print(f"sample_failed {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
