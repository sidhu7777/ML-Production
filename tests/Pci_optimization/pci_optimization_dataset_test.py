from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.lte_prediction import ml_engine


DEFAULT_PROJECT_ID = 193
DEFAULT_REGION = "india"
DEFAULT_OPERATOR = "Airtel"
OUTPUT_ROOT = Path("tests/output")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _write_json(payload: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _resolve_engine(region: str):
    current_engine = ml_engine.engine.get(str(region).lower(), ml_engine.engine["india"])
    if current_engine is None:
        raise RuntimeError(f"No DB engine configured for region={region}")
    return current_engine


def _table_columns(conn, table_name: str) -> set[str]:
    query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = :table_name
        """
    )
    df = pd.read_sql(query, conn, params={"table_name": table_name})
    col = "column_name" if "column_name" in df.columns else "COLUMN_NAME"
    return set(df[col].astype(str)) if col in df.columns else set()


def _parse_session_ids(raw_value: object) -> list[int]:
    if raw_value is None:
        return []
    out: list[int] = []
    for token in str(raw_value).replace(";", ",").split(","):
        token = token.strip()
        if token.isdigit():
            out.append(int(token))
    return out


def _fetch_project_session_ids(project_id: int, conn) -> tuple[list[int], dict]:
    project_cols = _table_columns(conn, "tbl_project")
    if "id" not in project_cols:
        raise RuntimeError("tbl_project.id not found; cannot resolve project sessions")
    select_cols = ["id"]
    for col in ["project_name", "name", "ref_session_id", "Session_ids", "session_ids"]:
        if col in project_cols:
            select_cols.append(col)

    query = text(
        f"""
        SELECT {", ".join(f"`{col}`" for col in select_cols)}
        FROM tbl_project
        WHERE id = :project_id
        LIMIT 1
        """
    )
    row = conn.execute(query, {"project_id": int(project_id)}).mappings().first()
    if not row:
        raise FileNotFoundError(f"No tbl_project row found for project_id={project_id}")
    meta = dict(row)
    raw_sessions = (
        meta.get("ref_session_id")
        or meta.get("Session_ids")
        or meta.get("session_ids")
        or ""
    )
    session_ids = _parse_session_ids(raw_sessions)
    if not session_ids:
        raise ValueError(f"No valid session IDs found in tbl_project for project_id={project_id}")
    return session_ids, meta


def _fetch_network_logs(
    session_ids: Iterable[int],
    operator: str,
    conn,
    primary_only: bool,
) -> pd.DataFrame:
    columns = _table_columns(conn, "tbl_network_log")
    required = ["session_id", "lat", "lon", "pci"]
    missing = [col for col in required if col not in columns]
    if missing:
        raise RuntimeError(f"tbl_network_log missing required columns: {missing}")

    wanted = [
        "id",
        "session_id",
        "timestamp",
        "lat",
        "lon",
        "pci",
        "earfcn",
        "band",
        "network",
        "m_alpha_long",
        "m_alpha_short",
        "rsrp",
        "rsrq",
        "sinr",
        "nodeb_id",
        "cell_id",
        "ta",
        "primary",
    ]
    selected = [col for col in wanted if col in columns]
    where_parts = ["session_id IN :session_ids"]
    params = {"session_ids": list(session_ids)}

    if operator and {"m_alpha_long", "m_alpha_short"}.intersection(columns):
        op_expr = []
        if "m_alpha_long" in columns:
            op_expr.append("LOWER(COALESCE(`m_alpha_long`, '')) = LOWER(:operator)")
        if "m_alpha_short" in columns:
            op_expr.append("LOWER(COALESCE(`m_alpha_short`, '')) = LOWER(:operator)")
        where_parts.append("(" + " OR ".join(op_expr) + ")")
        params["operator"] = str(operator)

    if primary_only and "primary" in columns:
        where_parts.append("LOWER(COALESCE(`primary`, '')) = 'yes'")

    query = text(
        f"""
        SELECT {", ".join(f"`{col}`" for col in selected)}
        FROM tbl_network_log
        WHERE {" AND ".join(where_parts)}
        """
    ).bindparams(bindparam("session_ids", expanding=True))
    df = pd.read_sql(query, conn, params=params)
    if "id" in df.columns and "log_id" not in df.columns:
        df = df.rename(columns={"id": "log_id"})
    return df


def _fetch_site_metadata(project_id: int, conn) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for table_name in ["site_ml", "site_noMl"]:
        columns = _table_columns(conn, table_name)
        if not columns or "project_id" not in columns:
            continue
        wanted = [
            "project_id",
            "network",
            "earfcn_or_narfcn",
            "pci_or_psi",
            "samples",
            "lat_pred",
            "lon_pred",
            "azimuth_deg_5",
            "azimuth_deg_5_soft",
            "azimuth_deg_label_soft",
            "azimuth_adjustment_deg",
            "beamwidth_deg_est",
            "median_sample_distance_m",
            "cell_id_representative",
            "site_key_inferred",
            "sector_count",
            "azimuth_reliability",
            "spacing_used",
        ]
        selected = [col for col in wanted if col in columns]
        query = text(
            f"""
            SELECT {", ".join(f"`{col}`" for col in selected)}
            FROM `{table_name}`
            WHERE project_id = :project_id
            """
        )
        df = pd.read_sql(query, conn, params={"project_id": int(project_id)})
        if not df.empty:
            df["site_source_table"] = table_name
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    site_df = pd.concat(frames, ignore_index=True)
    site_df = site_df.rename(
        columns={
            "lat_pred": "site_lat",
            "lon_pred": "site_lon",
            "pci_or_psi": "site_pci",
            "earfcn_or_narfcn": "site_earfcn",
            "azimuth_deg_5": "site_azimuth_deg",
            "azimuth_deg_5_soft": "site_azimuth_soft_deg",
            "cell_id_representative": "site_cell_id_representative",
            "site_key_inferred": "site_id_inferred",
        }
    )
    for col in ["site_lat", "site_lon", "site_pci", "site_earfcn", "samples"]:
        if col in site_df.columns:
            site_df[col] = pd.to_numeric(site_df[col], errors="coerce")
    return site_df


def _polygon_filter_logs(log_df: pd.DataFrame, project_id: int, current_engine, enabled: bool) -> tuple[pd.DataFrame, dict]:
    if not enabled:
        return log_df, {
            "enabled": False,
            "rows_before": int(len(log_df)),
            "rows_after": int(len(log_df)),
            "skipped": True,
        }
    filtered_df, stats = ml_engine._apply_drive_polygon_filter(log_df, int(project_id), current_engine)
    stats = dict(stats)
    stats["enabled"] = True
    return filtered_df.reset_index(drop=True), stats


def _polygon_filter_sites(site_df: pd.DataFrame, project_id: int, current_engine, enabled: bool) -> tuple[pd.DataFrame, dict]:
    if site_df.empty or not enabled or not {"site_lat", "site_lon"}.issubset(site_df.columns):
        return site_df, {
            "enabled": bool(enabled),
            "rows_before": int(len(site_df)),
            "rows_after": int(len(site_df)),
            "skipped": True,
        }
    work = site_df.rename(columns={"site_lat": "lat", "site_lon": "lon"}).copy()
    filtered, stats = ml_engine._apply_drive_polygon_filter(work, int(project_id), current_engine)
    filtered = filtered.rename(columns={"lat": "site_lat", "lon": "site_lon"})
    stats = dict(stats)
    stats["enabled"] = True
    return filtered.reset_index(drop=True), stats


def _norm_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().replace({"": np.nan, "nan": np.nan, "none": np.nan, "null": np.nan})


def _norm_number_key(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.round(0).astype("Int64").astype(str).replace({"<NA>": np.nan})


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = pd.to_numeric(lat1, errors="coerce").astype(float)
    lon1 = pd.to_numeric(lon1, errors="coerce").astype(float)
    lat2 = pd.to_numeric(lat2, errors="coerce").astype(float)
    lon2 = pd.to_numeric(lon2, errors="coerce").astype(float)
    radius_m = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return radius_m * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def _bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce").astype(float))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce").astype(float))
    dlon = np.radians(pd.to_numeric(lon2, errors="coerce").astype(float) - pd.to_numeric(lon1, errors="coerce").astype(float))
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def _detect_pci_transitions(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return pd.DataFrame()

    work = log_df.copy()
    work["_original_index"] = np.arange(len(work))
    for col in ["lat", "lon", "rsrp", "rsrq", "sinr"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "timestamp" in work.columns:
        work["_timestamp_sort"] = pd.to_datetime(work["timestamp"], errors="coerce")
    else:
        work["_timestamp_sort"] = pd.NaT
        work["timestamp"] = pd.NaT
    if "log_id" in work.columns:
        work["_log_id_sort"] = pd.to_numeric(work["log_id"], errors="coerce")
    else:
        work["log_id"] = pd.NA
        work["_log_id_sort"] = np.nan

    work["_session_key"] = _norm_key(work["session_id"])
    work["_pci_key"] = _norm_key(work["pci"])
    work = work.sort_values(
        ["_session_key", "_log_id_sort", "_timestamp_sort", "_original_index"],
        na_position="last",
    ).reset_index(drop=True)

    group = work.groupby("_session_key", dropna=False)
    prev_cols = [
        "log_id",
        "timestamp",
        "lat",
        "lon",
        "pci",
        "_pci_key",
        "rsrp",
        "rsrq",
        "sinr",
        "nodeb_id",
        "cell_id",
        "earfcn",
        "band",
        "network",
    ]
    for col in prev_cols:
        if col not in work.columns:
            work[col] = pd.NA
        work[f"prev_{col}"] = group[col].shift(1)

    valid = (
        work["lat"].notna()
        & work["lon"].notna()
        & work["_pci_key"].notna()
        & work["prev__pci_key"].notna()
        & (work["_pci_key"] != work["prev__pci_key"])
    )
    events = work.loc[valid].copy()
    if events.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "session_id": events["session_id"],
            "from_log_id": events["prev_log_id"],
            "to_log_id": events["log_id"],
            "from_timestamp": events["prev_timestamp"],
            "to_timestamp": events["timestamp"],
            "from_pci": events["prev_pci"],
            "to_pci": events["pci"],
            "from_lat": events["prev_lat"],
            "from_lon": events["prev_lon"],
            "to_lat": events["lat"],
            "to_lon": events["lon"],
            "event_lat": events["lat"],
            "event_lon": events["lon"],
            "from_nodeb_id_observed": events["prev_nodeb_id"],
            "to_nodeb_id_observed": events["nodeb_id"],
            "from_cell_id_observed": events["prev_cell_id"],
            "to_cell_id_observed": events["cell_id"],
            "from_earfcn": events["prev_earfcn"],
            "to_earfcn": events["earfcn"],
            "from_band": events["prev_band"],
            "to_band": events["band"],
            "from_network": events["prev_network"],
            "to_network": events["network"],
            "rsrp_before": events["prev_rsrp"],
            "rsrp_after": events["rsrp"],
            "rsrq_before": events["prev_rsrq"],
            "rsrq_after": events["rsrq"],
            "sinr_before": events["prev_sinr"],
            "sinr_after": events["sinr"],
        }
    )
    out["segment_distance_m"] = _haversine_m(out["from_lat"], out["from_lon"], out["to_lat"], out["to_lon"])
    out["event_bearing_deg"] = _bearing_deg(out["from_lat"], out["from_lon"], out["to_lat"], out["to_lon"])
    out["time_delta_sec"] = (
        pd.to_datetime(out["to_timestamp"], errors="coerce") - pd.to_datetime(out["from_timestamp"], errors="coerce")
    ).dt.total_seconds()
    out["speed_kmh"] = np.where(
        out["time_delta_sec"].gt(0),
        (out["segment_distance_m"] / out["time_delta_sec"]) * 3.6,
        np.nan,
    )
    out["rsrp_delta"] = pd.to_numeric(out["rsrp_after"], errors="coerce") - pd.to_numeric(out["rsrp_before"], errors="coerce")
    out["rsrq_delta"] = pd.to_numeric(out["rsrq_after"], errors="coerce") - pd.to_numeric(out["rsrq_before"], errors="coerce")
    out["sinr_delta"] = pd.to_numeric(out["sinr_after"], errors="coerce") - pd.to_numeric(out["sinr_before"], errors="coerce")
    out["same_site_observed"] = (
        _norm_key(out["from_nodeb_id_observed"]) == _norm_key(out["to_nodeb_id_observed"])
    ) & _norm_key(out["from_nodeb_id_observed"]).notna()
    out["same_earfcn"] = (_norm_number_key(out["from_earfcn"]) == _norm_number_key(out["to_earfcn"])) & _norm_number_key(out["from_earfcn"]).notna()
    return out.reset_index(drop=True)


def _prepare_site_lookup(site_df: pd.DataFrame) -> pd.DataFrame:
    if site_df.empty:
        return pd.DataFrame()
    work = site_df.copy()
    work["_site_pci_key"] = _norm_number_key(work["site_pci"])
    work["_site_earfcn_key"] = _norm_number_key(work["site_earfcn"]) if "site_earfcn" in work.columns else np.nan
    work["_site_network_key"] = _norm_key(work["network"]) if "network" in work.columns else np.nan
    work["_samples_sort"] = pd.to_numeric(work.get("samples", 0), errors="coerce").fillna(0.0)
    work = work.sort_values("_samples_sort", ascending=False)

    grouped = []
    for keys in [
        ["_site_network_key", "_site_earfcn_key", "_site_pci_key"],
        ["_site_earfcn_key", "_site_pci_key"],
        ["_site_pci_key"],
    ]:
        subset = work.dropna(subset=[keys[-1]]).drop_duplicates(subset=keys, keep="first").copy()
        subset["_lookup_level"] = "+".join(k.replace("_site_", "").replace("_key", "") for k in keys)
        grouped.append(subset)
    return pd.concat(grouped, ignore_index=True)


def _merge_site_side(events_df: pd.DataFrame, site_lookup: pd.DataFrame, side: str) -> pd.DataFrame:
    if events_df.empty or site_lookup.empty:
        return events_df
    out = events_df.copy()
    out[f"_{side}_pci_key"] = _norm_number_key(out[f"{side}_pci"])
    out[f"_{side}_earfcn_key"] = _norm_number_key(out[f"{side}_earfcn"]) if f"{side}_earfcn" in out.columns else np.nan
    out[f"_{side}_network_key"] = _norm_key(out[f"{side}_network"]) if f"{side}_network" in out.columns else np.nan

    site_cols = [
        "site_pci",
        "site_earfcn",
        "network",
        "samples",
        "site_lat",
        "site_lon",
        "site_azimuth_deg",
        "site_azimuth_soft_deg",
        "site_id_inferred",
        "site_cell_id_representative",
        "sector_count",
        "azimuth_reliability",
        "site_source_table",
        "_lookup_level",
        "_site_pci_key",
        "_site_earfcn_key",
        "_site_network_key",
    ]
    site_cols = [col for col in site_cols if col in site_lookup.columns]

    merged = None
    for event_keys, site_keys in [
        ([f"_{side}_network_key", f"_{side}_earfcn_key", f"_{side}_pci_key"], ["_site_network_key", "_site_earfcn_key", "_site_pci_key"]),
        ([f"_{side}_earfcn_key", f"_{side}_pci_key"], ["_site_earfcn_key", "_site_pci_key"]),
        ([f"_{side}_pci_key"], ["_site_pci_key"]),
    ]:
        candidate = out.merge(
            site_lookup[site_cols],
            left_on=event_keys,
            right_on=site_keys,
            how="left",
            suffixes=("", "_site"),
        )
        if merged is None:
            merged = candidate
        else:
            fill_cols = [col for col in candidate.columns if col in site_cols and col in merged.columns]
            missing_mask = merged["site_pci"].isna() if "site_pci" in merged.columns else pd.Series(True, index=merged.index)
            for col in fill_cols:
                merged.loc[missing_mask, col] = candidate.loc[missing_mask, col]

    if merged is None:
        return out
    rename_map = {
        col: f"{side}_{col}"
        for col in [
            "site_pci",
            "site_earfcn",
            "network",
            "samples",
            "site_lat",
            "site_lon",
            "site_azimuth_deg",
            "site_azimuth_soft_deg",
            "site_id_inferred",
            "site_cell_id_representative",
            "sector_count",
            "azimuth_reliability",
            "site_source_table",
            "_lookup_level",
        ]
        if col in merged.columns
    }
    merged = merged.rename(columns=rename_map)
    drop_cols = [col for col in merged.columns if col.startswith("_site_") or col in [f"_{side}_pci_key", f"_{side}_earfcn_key", f"_{side}_network_key"]]
    return merged.drop(columns=drop_cols, errors="ignore")


def _enrich_transition_sites(events_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    site_lookup = _prepare_site_lookup(site_df)
    out = _merge_site_side(events_df, site_lookup, "from")
    out = _merge_site_side(out, site_lookup, "to")
    if {"event_lat", "event_lon", "from_site_lat", "from_site_lon"}.issubset(out.columns):
        out["distance_to_from_site_m"] = _haversine_m(out["event_lat"], out["event_lon"], out["from_site_lat"], out["from_site_lon"])
    if {"event_lat", "event_lon", "to_site_lat", "to_site_lon"}.issubset(out.columns):
        out["distance_to_to_site_m"] = _haversine_m(out["event_lat"], out["event_lon"], out["to_site_lat"], out["to_site_lon"])
    if {"from_site_id_inferred", "to_site_id_inferred"}.issubset(out.columns):
        out["same_site_inferred"] = (
            _norm_key(out["from_site_id_inferred"]) == _norm_key(out["to_site_id_inferred"])
        ) & _norm_key(out["from_site_id_inferred"]).notna()
    return out


def _build_pair_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    group_cols = ["from_pci", "to_pci", "from_earfcn", "to_earfcn"]
    optional_cols = ["from_site_id_inferred", "to_site_id_inferred", "from_site_cell_id_representative", "to_site_cell_id_representative"]
    group_cols += [col for col in optional_cols if col in events_df.columns]
    summary = (
        events_df.groupby(group_cols, dropna=False)
        .agg(
            handover_count=("session_id", "size"),
            unique_sessions=("session_id", "nunique"),
            avg_segment_distance_m=("segment_distance_m", "mean"),
            median_segment_distance_m=("segment_distance_m", "median"),
            avg_time_delta_sec=("time_delta_sec", "mean"),
            avg_rsrp_before=("rsrp_before", "mean"),
            avg_rsrp_after=("rsrp_after", "mean"),
            avg_rsrp_delta=("rsrp_delta", "mean"),
            avg_rsrq_delta=("rsrq_delta", "mean"),
            avg_sinr_delta=("sinr_delta", "mean"),
            avg_distance_to_from_site_m=("distance_to_from_site_m", "mean") if "distance_to_from_site_m" in events_df.columns else ("segment_distance_m", "mean"),
            avg_distance_to_to_site_m=("distance_to_to_site_m", "mean") if "distance_to_to_site_m" in events_df.columns else ("segment_distance_m", "mean"),
        )
        .reset_index()
        .sort_values(["handover_count", "unique_sessions"], ascending=False)
    )
    return summary


def _build_cell_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    frames = []
    for role, pci_col, site_col, cell_col in [
        ("from", "from_pci", "from_site_id_inferred", "from_site_cell_id_representative"),
        ("to", "to_pci", "to_site_id_inferred", "to_site_cell_id_representative"),
    ]:
        cols = [pci_col]
        if site_col in events_df.columns:
            cols.append(site_col)
        if cell_col in events_df.columns:
            cols.append(cell_col)
        summary = (
            events_df.groupby(cols, dropna=False)
            .agg(
                handover_event_count=("session_id", "size"),
                unique_sessions=("session_id", "nunique"),
                avg_rsrp_before=("rsrp_before", "mean"),
                avg_rsrp_after=("rsrp_after", "mean"),
                avg_rsrp_delta=("rsrp_delta", "mean"),
                avg_sinr_delta=("sinr_delta", "mean"),
            )
            .reset_index()
        )
        summary["handover_role"] = role
        summary = summary.rename(columns={pci_col: "pci", site_col: "site_id_inferred", cell_col: "cell_id_representative"})
        frames.append(summary)
    return pd.concat(frames, ignore_index=True).sort_values("handover_event_count", ascending=False)


def _build_session_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    return (
        events_df.groupby("session_id", dropna=False)
        .agg(
            handover_count=("session_id", "size"),
            unique_from_pci=("from_pci", "nunique"),
            unique_to_pci=("to_pci", "nunique"),
            total_transition_distance_m=("segment_distance_m", "sum"),
            avg_rsrp_delta=("rsrp_delta", "mean"),
            avg_sinr_delta=("sinr_delta", "mean"),
        )
        .reset_index()
        .sort_values("handover_count", ascending=False)
    )


def run_pci_optimization_dataset_test(args: argparse.Namespace) -> Path:
    start = time.perf_counter()
    project_id = int(args.project_id)
    region = str(args.region)
    operator = str(args.operator or "")
    current_engine = _resolve_engine(region)
    run_dir = _ensure_dir(
        Path(args.output_root)
        / f"project_{project_id}"
        / "pci_optimization"
        / time.strftime("pci_optimization_%Y%m%d_%H%M%S")
    )

    with current_engine.connect() as conn:
        session_ids, project_meta = _fetch_project_session_ids(project_id, conn)
        if args.session_ids:
            session_ids = _parse_session_ids(args.session_ids)
        print(
            f"[PCI_OPT][PROJECT] project_id={project_id} region={region} operator={operator} "
            f"sessions={len(session_ids)} output={run_dir}"
        )

        raw_log_df = _fetch_network_logs(session_ids, operator, conn, primary_only=not args.include_non_primary)
        print(
            f"[PCI_OPT][LOG_FETCH] rows={len(raw_log_df)} sessions="
            f"{int(raw_log_df['session_id'].nunique()) if 'session_id' in raw_log_df.columns and not raw_log_df.empty else 0} "
            f"pci_unique={int(raw_log_df['pci'].nunique()) if 'pci' in raw_log_df.columns and not raw_log_df.empty else 0}"
        )
        log_df, log_polygon_stats = _polygon_filter_logs(
            raw_log_df, project_id, current_engine, enabled=not args.no_polygon_filter
        )
        print(
            f"[PCI_OPT][LOG_POLYGON] enabled={log_polygon_stats.get('enabled')} "
            f"rows_before={log_polygon_stats.get('rows_before')} rows_after={log_polygon_stats.get('rows_after')} "
            f"swapped={log_polygon_stats.get('swapped')} skipped={log_polygon_stats.get('skipped')}"
        )

        site_df = _fetch_site_metadata(project_id, conn)
        site_df, site_polygon_stats = _polygon_filter_sites(
            site_df, project_id, current_engine, enabled=not args.no_polygon_filter
        )
        print(
            f"[PCI_OPT][SITE_FETCH] rows={len(site_df)} pci_unique="
            f"{int(site_df['site_pci'].nunique()) if 'site_pci' in site_df.columns and not site_df.empty else 0} "
            f"polygon_rows_before={site_polygon_stats.get('rows_before')} polygon_rows_after={site_polygon_stats.get('rows_after')}"
        )

    if log_df.empty:
        raise ValueError("No network log rows available after fetch/polygon filter")

    events_df = _detect_pci_transitions(log_df)
    enriched_events_df = _enrich_transition_sites(events_df, site_df)
    pair_summary_df = _build_pair_summary(enriched_events_df)
    cell_summary_df = _build_cell_summary(enriched_events_df)
    session_summary_df = _build_session_summary(enriched_events_df)

    artifact_paths = {
        "raw_network_logs": _write_csv(log_df, run_dir / "pci_network_logs_polygon_filtered.csv"),
        "site_metadata": _write_csv(site_df, run_dir / "pci_site_metadata_polygon_filtered.csv"),
        "transition_events": _write_csv(enriched_events_df, run_dir / f"pci_handover_events_project_{project_id}.csv"),
        "transition_pair_summary": _write_csv(pair_summary_df, run_dir / f"pci_handover_pair_summary_project_{project_id}.csv"),
        "cell_handover_summary": _write_csv(cell_summary_df, run_dir / f"pci_cell_handover_summary_project_{project_id}.csv"),
        "session_handover_summary": _write_csv(session_summary_df, run_dir / f"pci_session_handover_summary_project_{project_id}.csv"),
    }

    if args.write_parquet:
        enriched_events_df.to_parquet(run_dir / f"pci_handover_events_project_{project_id}.parquet", index=False)
        pair_summary_df.to_parquet(run_dir / f"pci_handover_pair_summary_project_{project_id}.parquet", index=False)
        artifact_paths["transition_events_parquet"] = str(run_dir / f"pci_handover_events_project_{project_id}.parquet")
        artifact_paths["transition_pair_summary_parquet"] = str(run_dir / f"pci_handover_pair_summary_project_{project_id}.parquet")

    summary = {
        "project_id": project_id,
        "region": region,
        "operator": operator,
        "session_ids": session_ids,
        "project_meta": project_meta,
        "log_polygon_stats": log_polygon_stats,
        "site_polygon_stats": site_polygon_stats,
        "rows": {
            "network_logs": int(len(log_df)),
            "site_metadata": int(len(site_df)),
            "transition_events": int(len(enriched_events_df)),
            "transition_pairs": int(len(pair_summary_df)),
            "cell_summary": int(len(cell_summary_df)),
            "session_summary": int(len(session_summary_df)),
        },
        "distinct": {
            "sessions": int(log_df["session_id"].nunique()) if "session_id" in log_df.columns else 0,
            "pci_in_logs": int(log_df["pci"].nunique()) if "pci" in log_df.columns else 0,
            "from_pci": int(enriched_events_df["from_pci"].nunique()) if "from_pci" in enriched_events_df.columns else 0,
            "to_pci": int(enriched_events_df["to_pci"].nunique()) if "to_pci" in enriched_events_df.columns else 0,
        },
        "mapping": {
            "from_site_unmapped": int(enriched_events_df["from_site_pci"].isna().sum()) if "from_site_pci" in enriched_events_df.columns else int(len(enriched_events_df)),
            "to_site_unmapped": int(enriched_events_df["to_site_pci"].isna().sum()) if "to_site_pci" in enriched_events_df.columns else int(len(enriched_events_df)),
        },
        "artifact_paths": artifact_paths,
        "runtime_sec": round(time.perf_counter() - start, 3),
    }
    artifact_paths["summary"] = _write_json(summary, run_dir / "pci_optimization_dataset_summary.json")

    print(
        f"[PCI_OPT][DONE] run_dir={run_dir} events={len(enriched_events_df)} "
        f"pairs={len(pair_summary_df)} runtime_sec={summary['runtime_sec']}"
    )
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local PCI handover/optimization datasets from DB logs.")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=DEFAULT_OPERATOR)
    parser.add_argument("--session-ids", type=str, default="", help="Optional comma-separated session override.")
    parser.add_argument("--include-non-primary", action="store_true", help="Include rows where primary is not yes, if column exists.")
    parser.add_argument("--no-polygon-filter", action="store_true", help="Skip project polygon filtering.")
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run_dir = run_pci_optimization_dataset_test(_parse_args())
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
