"""
Fetch the raw inputs for the sector-swap test case. READ-ONLY on the database.

For one project it saves, unfiltered:
  data/raw_site_prediction.csv   every site_prediction row. No dedup here: the old identity dedup
                                 (site|cell_id|sector|band|operator) silently dropped ~80% of project
                                 193's rows, because one cell_id often carries several PCIs.
  data/raw_serving_logs.csv      tbl_network_log serving rows (primary = Yes): eNodeB id, cell id,
                                 PCI, EARFCN, band, technology (network), operator, RSRP.
  data/raw_neighbour_logs.csv    tbl_network_log_neighbour rows: PCI, EARFCN, band, technology, RSRP.
                                 This table has no usable cell identity (nodeb_id / cell_id are only
                                 placeholders such as 0 / 8388607 / 131071), so it can only be matched
                                 by PCI.
  data/raw_fetch_info.json       project, region, session and row counts, fetch time.

Nothing is filtered by PCI, radius or site here -- all matching happens in build_dataset.py.

Run from the ML/ directory:
    venv\\Scripts\\python.exe -m tests.swap_sector.fetch_data --project-id 193 --region india
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, text

from tools.pci_optimization.db import get_engine
from tests.Pci_optimization.pci_optimization_dataset_test import _fetch_project_session_ids

DATA_DIR = Path(__file__).resolve().parent / "data"
LOG_COLUMNS = [
    "session_id", "timestamp", "lat", "lon", "network", "band", "earfcn", "pci",
    "nodeb_id", "cell_id", "m_alpha_long", "rsrp", "rsrq", "sinr",
]


def fetch_site_prediction(project_id: int, conn) -> pd.DataFrame:
    df = pd.read_sql(
        text("SELECT * FROM site_prediction WHERE tbl_project_id = :project_id"),
        conn, params={"project_id": int(project_id)},
    )
    for col in ("latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Some regions store lat/lon as an integer with the decimal point stripped
    # (25007083 -> 25.007083); same fix as tools/pci_optimization/engine.py.
    df.loc[df["latitude"].abs() > 90, "latitude"] /= 1_000_000.0
    df.loc[df["longitude"].abs() > 180, "longitude"] /= 1_000_000.0
    return df


def fetch_logs(table: str, session_ids: list[int], conn, serving_only: bool) -> pd.DataFrame:
    where = "session_id IN :session_ids"
    if serving_only:
        where += " AND LOWER(COALESCE(`primary`, '')) = 'yes'"
    select = ", ".join(f"`{col}`" for col in LOG_COLUMNS)
    query = text(f"SELECT {select} FROM `{table}` WHERE {where}").bindparams(
        bindparam("session_ids", expanding=True)
    )
    return pd.read_sql(query, conn, params={"session_ids": list(session_ids)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, default=193)
    parser.add_argument("--region", default="india")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    engine = get_engine(args.region)
    with engine.connect() as conn:
        session_ids, _meta = _fetch_project_session_ids(args.project_id, conn)
        site_df = fetch_site_prediction(args.project_id, conn)
        serving_df = fetch_logs("tbl_network_log", session_ids, conn, serving_only=True)
        neighbour_df = fetch_logs("tbl_network_log_neighbour", session_ids, conn, serving_only=False)

    site_df.to_csv(DATA_DIR / "raw_site_prediction.csv", index=False)
    serving_df.to_csv(DATA_DIR / "raw_serving_logs.csv", index=False)
    neighbour_df.to_csv(DATA_DIR / "raw_neighbour_logs.csv", index=False)

    info = {
        "project_id": args.project_id,
        "region": args.region,
        "sessions": len(session_ids),
        "site_prediction_rows": len(site_df),
        "serving_rows": len(serving_df),
        "neighbour_rows": len(neighbour_df),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (DATA_DIR / "raw_fetch_info.json").write_text(json.dumps(info, indent=2))
    for key, value in info.items():
        print(f"[fetch] {key}: {value}")


if __name__ == "__main__":
    main()
