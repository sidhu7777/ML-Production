from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import transform
from sqlalchemy import create_engine, text

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from utils.python_bridge import _filter_complete_site_prediction_identity


PROJECT_ID = int(os.getenv("PROP_PROJECT_ID", "210"))
REGION = os.getenv("PROP_REGION", "taiwan").strip().lower()
PROJECT_SLUG = os.getenv("PROP_PROJECT_SLUG", f"project_{PROJECT_ID}_{REGION}")
PROJECT_DIR = Path(os.getenv("PROP_PROJECT_DIR", str(THIS_DIR / "data" / PROJECT_SLUG)))
ENV_PATH = ML_ROOT / ".env"

RAW_DB = PROJECT_DIR / "raw_db"
BASELINE_SCOPE = PROJECT_DIR / "baseline_fetch_scope"
GEO_DB = PROJECT_DIR / "geo_db"
GRID_DB = PROJECT_DIR / "grid_db"


def _ensure_dirs() -> None:
    for path in [PROJECT_DIR, RAW_DB, BASELINE_SCOPE, GEO_DB, GRID_DB]:
        path.mkdir(parents=True, exist_ok=True)


def _engine():
    load_dotenv(ENV_PATH)
    key = "DATABASE_URL_Taiwan" if REGION == "taiwan" else "DATABASE_URL"
    db_url = os.getenv(key)
    if not db_url:
        raise RuntimeError(f"{key} is not set in {ENV_PATH}")
    return create_engine(db_url, pool_pre_ping=True, pool_recycle=300)


def _save_frame(df: pd.DataFrame, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(path_stem.with_suffix(".parquet"), index=False)
    except Exception as exc:
        path_stem.with_suffix(".parquet.error.txt").write_text(str(exc), encoding="utf-8")


def _project_row(conn) -> pd.Series:
    df = pd.read_sql(text("SELECT * FROM tbl_project WHERE id = :pid"), conn, params={"pid": PROJECT_ID})
    if df.empty:
        raise ValueError(f"No tbl_project row found for project_id={PROJECT_ID}")
    return df.iloc[0]


def _session_ids(project: pd.Series) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", str(project.get("ref_session_id") or ""))]


def _load_polygon(conn):
    regions = pd.read_sql(
        text(
            """
            SELECT id, tbl_project_id, name, status, area, session_id, ST_AsText(region) AS region_wkt
            FROM map_regions
            WHERE tbl_project_id = :pid AND status = 1
            """
        ),
        conn,
        params={"pid": PROJECT_ID},
    )
    _save_frame(regions, GEO_DB / f"map_regions_project_{PROJECT_ID}_active")
    if regions.empty or "region_wkt" not in regions.columns:
        raise ValueError(f"No active polygon found for project_id={PROJECT_ID}")
    poly = wkt.loads(str(regions.iloc[0]["region_wkt"]))
    return regions, poly


def _choose_polygon_orientation(poly, point_df: pd.DataFrame):
    valid = point_df.dropna(subset=["lat", "lon"]).copy()
    direct_hits = int(sum(poly.covers(Point(lon, lat)) for lat, lon in valid[["lat", "lon"]].to_numpy()))
    swapped = transform(lambda x, y, z=None: (y, x) if z is None else (y, x, z), poly)
    swapped_hits = int(sum(swapped.covers(Point(lon, lat)) for lat, lon in valid[["lat", "lon"]].to_numpy()))
    chosen = poly if direct_hits >= swapped_hits else swapped
    return chosen if chosen.is_valid else chosen.buffer(0), direct_hits, swapped_hits


def _normalize_sites(raw_site: pd.DataFrame) -> pd.DataFrame:
    site = _filter_complete_site_prediction_identity(raw_site, endpoint=f"cache:site_prediction_project_{PROJECT_ID}")
    site = site.rename(
        columns={
            "latitude": "lat",
            "longitude": "lon",
            "e_tilt": "Etilt",
            "m_tilt": "Mtilt",
            "height": "Height",
            "pci": "PCI",
            "cluster": "network",
        }
    )
    if "provider" not in site.columns and "network" in site.columns:
        site["provider"] = site["network"]
    if "operator_name" not in site.columns and "network" in site.columns:
        site["operator_name"] = site["network"]
    if "operator" not in site.columns:
        site["operator"] = site.get("network", site.get("provider", pd.Series(index=site.index)))
    if "legacy_nodeb_id_cell_id" not in site.columns:
        site["legacy_nodeb_id_cell_id"] = site["cell_id"].astype(str).str.strip()
    if "Node_Cell_ID" not in site.columns:
        site["Node_Cell_ID"] = site["site_prediction_key"].astype(str).str.strip().str.replace("|", "_", regex=False)
    if "rf_identity_key" not in site.columns:
        site["rf_identity_key"] = site["Node_Cell_ID"]
    for col, default in {
        "lat": None,
        "lon": None,
        "azimuth": 0.0,
        "Height": 30.0,
        "Mtilt": 0.0,
        "Etilt": 3.0,
        "tx_power": 46.0,
    }.items():
        site[col] = pd.to_numeric(site[col], errors="coerce")
        if default is not None:
            site[col] = site[col].fillna(default)
    return site.dropna(subset=["lat", "lon", "Node_Cell_ID"]).copy()


def _fetch_sites(conn, polygon) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_sql(
        text(
            """
            SELECT site_prediction.*, cluster AS provider, cluster AS operator_name
            FROM site_prediction
            WHERE tbl_project_id = :pid
            """
        ),
        conn,
        params={"pid": PROJECT_ID},
    )
    _save_frame(raw, RAW_DB / f"site_prediction_project_{PROJECT_ID}_raw_all")
    sites = _normalize_sites(raw)
    chosen_poly, direct_hits, swapped_hits = _choose_polygon_orientation(polygon, sites)
    mask = [chosen_poly.covers(Point(lon, lat)) for lat, lon in sites[["lat", "lon"]].to_numpy()]
    polygon_sites = sites.loc[mask].copy()
    strict_sites = polygon_sites.drop_duplicates(subset=["Node_Cell_ID"], keep="first").sort_values(
        ["site", "sector", "band", "cell_id"]
    )
    _save_frame(polygon_sites, RAW_DB / f"site_prediction_project_{PROJECT_ID}_raw_polygon")
    _save_frame(sites, BASELINE_SCOPE / "site_baseline_complete_identity_all_project")
    _save_frame(polygon_sites, BASELINE_SCOPE / "site_baseline_complete_identity_polygon")
    _save_frame(strict_sites, BASELINE_SCOPE / f"site_identity_strict_cells_project{PROJECT_ID}")
    _save_frame(
        polygon_sites.drop_duplicates(subset=["site", "sector", "band"], keep="first"),
        BASELINE_SCOPE / "site_sector_band_unique_polygon",
    )
    stats = {
        "site_prediction_raw_all_project_rows": int(len(raw)),
        "site_complete_all_project_rows": int(len(sites)),
        "site_prediction_raw_polygon_rows": int(len(polygon_sites)),
        "site_identity_strict_keys_polygon": int(len(strict_sites)),
        "site_polygon_direct_hits": direct_hits,
        "site_polygon_swapped_hits": swapped_hits,
    }
    return strict_sites, chosen_poly, stats


def _fetch_grid(conn, polygon) -> tuple[pd.DataFrame, object, dict]:
    scenario_row = conn.execute(
        text(
            """
            SELECT scenario_id, MAX(created_at) AS max_created, COUNT(*) AS row_count
            FROM grid_analytics_results
            WHERE project_id = :pid
            GROUP BY scenario_id
            ORDER BY row_count DESC, max_created DESC
            LIMIT 1
            """
        ),
        {"pid": PROJECT_ID},
    ).fetchone()
    if scenario_row is None:
        raise ValueError(f"No grid_analytics_results rows found for project_id={PROJECT_ID}")
    selected_scenario = scenario_row[0]
    if selected_scenario is None:
        where = "project_id = :pid AND scenario_id IS NULL"
        params = {"pid": PROJECT_ID}
    else:
        where = "project_id = :pid AND scenario_id = :scenario_id"
        params = {"pid": PROJECT_ID, "scenario_id": int(selected_scenario)}
    all_grid = pd.read_sql(text("SELECT * FROM grid_analytics_results WHERE project_id = :pid"), conn, params={"pid": PROJECT_ID})
    selected = pd.read_sql(
        text(
            f"""
            SELECT grid_id, center_lat, center_lon, min_lat, max_lat, min_lon, max_lon, grid_size_meters, scenario_id
            FROM grid_analytics_results
            WHERE {where}
            ORDER BY grid_id
            """
        ),
        conn,
        params=params,
    )
    _save_frame(all_grid, GRID_DB / f"grid_analytics_project_{PROJECT_ID}_all")
    for col in ["center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon", "grid_size_meters"]:
        if col in selected.columns:
            selected[col] = pd.to_numeric(selected[col], errors="coerce")
    selected = selected.dropna(subset=["grid_id", "center_lat", "center_lon"]).drop_duplicates("grid_id").copy()
    mask = [polygon.covers(Point(lon, lat)) for lat, lon in selected[["center_lat", "center_lon"]].to_numpy()]
    selected_polygon = selected.loc[mask].copy()
    if selected_polygon.empty:
        selected_polygon = selected
    _save_frame(selected_polygon, GRID_DB / f"grid_analytics_project_{PROJECT_ID}_selected_scenario")
    stats = {
        "grid_analytics_all_rows": int(len(all_grid)),
        "grid_analytics_selected_scenario_rows": int(len(selected)),
        "grid_analytics_selected_polygon_rows": int(len(selected_polygon)),
        "selected_grid_scenario_id": None if selected_scenario is None else int(selected_scenario),
    }
    return selected_polygon, selected_scenario, stats


def _network_like_4g(series: pd.Series) -> pd.Series:
    return series.astype(str).str.contains("4G|LTE", case=False, na=False)


def _fetch_drive_table(conn, table_name: str, sessions: list[int], operator: str | None) -> pd.DataFrame:
    if not sessions:
        return pd.DataFrame()
    chunks = []
    session_chunks = [sessions[idx : idx + 80] for idx in range(0, len(sessions), 80)]
    for chunk in session_chunks:
        in_clause = ",".join(str(int(sid)) for sid in chunk)
        df = pd.read_sql(
            text(
                f"""
                SELECT id, lat, lon, rsrp, rsrq, sinr, mci, pci, earfcn, session_id, timestamp,
                       network, m_alpha_long, m_alpha_short, `primary`
                FROM {table_name}
                WHERE session_id IN ({in_clause})
                  AND LOWER(COALESCE(`primary`, '')) = 'yes'
                  AND rsrp IS NOT NULL
                """
            ),
            conn,
        )
        if not df.empty:
            chunks.append(df)
    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if out.empty:
        return out
    out = out.loc[_network_like_4g(out["network"])].copy()
    if operator:
        op = str(operator).strip().lower()
        op_mask = (
            out["m_alpha_long"].astype(str).str.strip().str.lower().eq(op)
            | out["m_alpha_short"].astype(str).str.strip().str.lower().eq(op)
            | out["network"].astype(str).str.strip().str.lower().eq(op)
        )
        if op_mask.any():
            out = out.loc[op_mask].copy()
    out["source_table"] = table_name
    out["cell_id"] = out.get("mci", pd.Series(index=out.index)).astype(str).replace({"nan": ""})
    out["nodeb_id"] = ""
    return out


def _fetch_drive(conn, project: pd.Series, polygon, operator: str | None) -> tuple[pd.DataFrame, dict]:
    sessions = _session_ids(project)
    main = _fetch_drive_table(conn, "tbl_network_log", sessions, operator)
    neighbour = _fetch_drive_table(conn, "tbl_network_log_neighbour", sessions, operator)
    raw = pd.concat([main, neighbour], ignore_index=True) if not main.empty or not neighbour.empty else pd.DataFrame()
    _save_frame(raw, RAW_DB / f"drive_project_{PROJECT_ID}_raw_sessions_combined")
    if raw.empty:
        filtered = raw
    else:
        for col in ["lat", "lon", "rsrp", "rsrq", "sinr"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw = raw.dropna(subset=["lat", "lon", "rsrp"]).copy()
        mask = [polygon.covers(Point(lon, lat)) for lat, lon in raw[["lat", "lon"]].to_numpy()]
        filtered = raw.loc[mask].copy()
    _save_frame(filtered, BASELINE_SCOPE / f"drive_project_{PROJECT_ID}_baseline_primary_polygon")
    stats = {
        "session_count": int(len(sessions)),
        "drive_raw_sessions_combined_rows": int(len(raw)),
        "drive_baseline_primary_polygon_rows": int(len(filtered)),
    }
    return filtered, stats


def _fetch_buildings(conn) -> dict:
    buildings = pd.read_sql(
        text("SELECT id, name, region, project_id, area, geometry FROM tbl_savepolygon WHERE project_id = :pid"),
        conn,
        params={"pid": PROJECT_ID},
    )
    _save_frame(buildings, GEO_DB / f"tbl_savepolygon_project_{PROJECT_ID}_buildings")
    return {"buildings_rows": int(len(buildings))}


def main() -> None:
    _ensure_dirs()
    eng = _engine()
    with eng.connect() as conn:
        project = _project_row(conn)
        _save_frame(pd.DataFrame([project.to_dict()]), RAW_DB / f"project_{PROJECT_ID}_tbl_project")
        _, raw_poly = _load_polygon(conn)
        strict_sites, project_polygon, site_stats = _fetch_sites(conn, raw_poly)
        grid_df, selected_scenario, grid_stats = _fetch_grid(conn, project_polygon)
        operator = None
        if not strict_sites.empty and "operator" in strict_sites.columns:
            values = strict_sites["operator"].dropna().astype(str).str.strip()
            operator = values.mode().iloc[0] if not values.empty else None
        drive_df, drive_stats = _fetch_drive(conn, project, project_polygon, operator)
        building_stats = _fetch_buildings(conn)

    manifest = {
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "source": f"direct {REGION} DB from {ENV_PATH}",
        "project_id": PROJECT_ID,
        "region": REGION,
        "production_code_modified": False,
        "production_baseline_run": False,
        "project_dir": str(PROJECT_DIR.relative_to(THIS_DIR)),
        "counts": {**site_stats, **grid_stats, **drive_stats, **building_stats},
        "operator": operator,
        "notes": [
            "Site scope is active project polygon plus complete site/cell/sector/band/operator identity.",
            "Strict identity is site|cell_id|sector|band|operator, stored as Node_Cell_ID/rf_identity_key for local propagation tests.",
            "Drive-test scope is primary 4G/LTE project session rows inside the same project polygon.",
            "No production baseline or DB write was run.",
        ],
    }
    (PROJECT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
