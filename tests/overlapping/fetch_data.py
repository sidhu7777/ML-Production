"""
Fetch the raw inputs for the coverage-redundancy (overlapping) test case. READ-ONLY on the database.

For one project it saves under data/project_<id>/ (the model itself never touches the database):
  site_prediction.csv      every site_prediction row, unfiltered (cleaning is in site_cleaning.py)
  polygons.json            map_regions polygons as WKT, exactly as stored (project 193: "lat lon")
  forecast_users.parquet   lte_forecast_predictions users_per_grid/population/clutter (traffic weights)
  dt_serving.parquet       tbl_network_log serving rows (primary = Yes): calibration + validation of
                           the RF surface

lte_prediction_baseline_results is deliberately NOT used. Project 193 check (2026-09-11): its Airtel
pred_rsrp is flat at ~-91 dBm from 50 m to 500 m and from boresight to behind the antenna (it is
drive-test values carried over and smoothed), and each point holds only one site's own sectors.
Calibrating a directional per-antenna model to it added +25 dB and over-predicted the drive test
by 27 dB. The drive test itself is the ground truth.
  buildings.parquet        tbl_savepolygon building footprints as WKT (indoor points)
  fetch_info.json          row counts and fetch time

The database connection has been timing out intermittently, so every query is retried.

Run from the ML/ directory:
    venv\\Scripts\\python.exe -m tests.overlapping.fetch_data --project-id 193 --region india
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import shapely
from sqlalchemy import bindparam, text
from sqlalchemy.exc import DBAPIError, OperationalError

from tests.overlapping.config import DATA_DIR
from tools.pci_optimization.db import get_engine

LOG_COLUMNS = [
    "session_id", "timestamp", "lat", "lon", "network", "band", "earfcn", "pci",
    "nodeb_id", "cell_id", "m_alpha_long", "rsrp", "rsrq", "sinr",
]


def project_data_dir(project_id: int) -> Path:
    return DATA_DIR / f"project_{int(project_id)}"


def _with_retry(engine, what: str, fn: Callable, attempts: int = 4):
    for attempt in range(1, attempts + 1):
        try:
            with engine.connect() as conn:
                return fn(conn)
        except OperationalError as exc:
            if attempt == attempts:
                raise
            wait_s = 15 * attempt
            print(f"[OVERLAP][FETCH] {what}: attempt {attempt} failed ({str(exc.orig)[:120]}); retry in {wait_s}s")
            time.sleep(wait_s)


def _session_ids(project_id: int, conn) -> list[int]:
    raw = conn.execute(
        text("SELECT ref_session_id FROM tbl_project WHERE id = :p"), {"p": int(project_id)}
    ).scalar()
    ids = [int(s) for s in str(raw or "").replace(";", ",").split(",") if s.strip().isdigit()]
    if not ids:
        raise ValueError(f"project {project_id} has no ref_session_id sessions")
    return ids


def _geometry_to_wkt(value) -> str | None:
    """MySQL geometry comes back as WKT text via ST_AsText, or as raw bytes (4-byte SRID + WKB)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        blob = bytes(value)
        for payload in (blob[4:], blob):
            try:
                return shapely.from_wkb(payload).wkt
            except Exception:
                continue
        return None
    return str(value)


def fetch_polygons(project_id: int, conn) -> list[dict]:
    df = pd.read_sql(
        text("SELECT id, name, status, ST_AsText(region) AS wkt FROM map_regions WHERE tbl_project_id = :p"),
        conn,
        params={"p": int(project_id)},
    )
    return [
        {"id": int(r.id), "name": str(r.name), "status": None if pd.isna(r.status) else int(r.status), "wkt": r.wkt}
        for r in df.itertuples()
        if r.wkt
    ]


def fetch_forecast_users(project_id: int, conn) -> pd.DataFrame:
    return pd.read_sql(
        text(
            """
            SELECT lat, lon, operator, nodeb_id_cell_id, users_per_grid, population, clutter
            FROM lte_forecast_predictions WHERE project_id = :p
            """
        ),
        conn,
        params={"p": int(project_id)},
    )


def fetch_dt_serving(session_ids: list[int], conn) -> pd.DataFrame:
    select = ", ".join(f"`{c}`" for c in LOG_COLUMNS)
    query = text(
        f"SELECT {select} FROM tbl_network_log "
        "WHERE session_id IN :ids AND LOWER(COALESCE(`primary`, '')) = 'yes'"
    ).bindparams(bindparam("ids", expanding=True))
    return pd.read_sql(query, conn, params={"ids": list(session_ids)})


def fetch_buildings(project_id: int, conn) -> pd.DataFrame:
    params = {"p": int(project_id)}
    # Project 193 stores the footprint in `region` (POLYGON, SRID 4326) and leaves `geometry` NULL.
    try:
        df = pd.read_sql(
            text(
                "SELECT id, COALESCE(ST_AsText(geometry), ST_AsText(region)) AS wkt, is_active "
                "FROM tbl_savepolygon WHERE project_id = :p"
            ),
            conn,
            params=params,
        )
    except DBAPIError:
        # spatial functions unavailable: decode the raw bytes instead
        conn.rollback()
        df = pd.read_sql(
            text("SELECT id, COALESCE(geometry, region) AS wkt, is_active FROM tbl_savepolygon WHERE project_id = :p"),
            conn,
            params=params,
        )
    df["wkt"] = df["wkt"].map(_geometry_to_wkt)
    active = pd.to_numeric(df["is_active"], errors="coerce")
    return df[active.isna() | (active != 0)].drop(columns=["is_active"]).reset_index(drop=True)


def fetch_site_prediction(project_id: int, conn) -> pd.DataFrame:
    return pd.read_sql(
        text("SELECT * FROM site_prediction WHERE tbl_project_id = :p"), conn, params={"p": int(project_id)}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-id", type=int, default=193)
    parser.add_argument("--region", default="india")
    args = parser.parse_args()

    out_dir = project_data_dir(args.project_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = get_engine(args.region)
    started = time.time()

    session_ids = _with_retry(engine, "sessions", lambda c: _session_ids(args.project_id, c))
    sites = _with_retry(engine, "site_prediction", lambda c: fetch_site_prediction(args.project_id, c))
    sites.to_csv(out_dir / "site_prediction.csv", index=False)
    print(f"[OVERLAP][FETCH] site_prediction rows={len(sites)}")

    polygons = _with_retry(engine, "map_regions", lambda c: fetch_polygons(args.project_id, c))
    (out_dir / "polygons.json").write_text(json.dumps(polygons, indent=2))
    print(f"[OVERLAP][FETCH] polygons={len(polygons)}")

    buildings = _with_retry(engine, "buildings", lambda c: fetch_buildings(args.project_id, c))
    buildings.to_parquet(out_dir / "buildings.parquet", index=False)
    print(f"[OVERLAP][FETCH] buildings={len(buildings)}")

    forecast = _with_retry(engine, "forecast users", lambda c: fetch_forecast_users(args.project_id, c))
    forecast.to_parquet(out_dir / "forecast_users.parquet", index=False)
    print(f"[OVERLAP][FETCH] forecast rows={len(forecast)}")

    dt = _with_retry(engine, "drive test", lambda c: fetch_dt_serving(session_ids, c))
    dt.to_parquet(out_dir / "dt_serving.parquet", index=False)
    print(f"[OVERLAP][FETCH] dt serving rows={len(dt)}")

    info = {
        "project_id": args.project_id,
        "region": args.region,
        "sessions": len(session_ids),
        "site_prediction_rows": len(sites),
        "polygons": len(polygons),
        "buildings": len(buildings),
        "forecast_rows": len(forecast),
        "dt_serving_rows": len(dt),
        "fetch_seconds": round(time.time() - started, 1),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "fetch_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
