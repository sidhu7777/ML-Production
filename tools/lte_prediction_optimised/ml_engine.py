import os
import time

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

from .Sector_wise_prediction_code_copy import (
    calibrate_site,
    compute_predictions_parallel,
    generate_grid,
    haversine_vectorized,
)
from ..lte_prediction.geo_correction_pipeline import (
    load_geo_weights,
    attach_site_context_features,
    attach_fixed_serving_sinr_rsrq_proxy,
    _refine_experimental_forward_features,
    apply_experimental_geo_adjustments,
)
from ..lte_tilt_recommandation.cell_identity import canonical_cell_id, canonical_pair


load_dotenv()

engine = {
    "india": create_engine(
        os.getenv("DATABASE_URL"),
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL") else None,
    "taiwan": create_engine(
        os.getenv("DATABASE_URL_Taiwan"),
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL_Taiwan") else None
}


def _safe_nunique(df, col):
    return int(df[col].nunique(dropna=True)) if col in df.columns else "n/a"


def _safe_non_null(df, col):
    return int(df[col].notna().sum()) if col in df.columns else "n/a"


def _safe_minmax(df, col):
    if col not in df.columns:
        return "n/a"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return "n/a"
    return f"{series.min():.4f}..{series.max():.4f}"


def _clean_token(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null", "<na>"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _rf_cell_id(value) -> str:
    text = _clean_token(value).replace("|", "_")
    while ".0_" in text:
        text = text.replace(".0_", "_")
    if text.endswith(".0"):
        text = text[:-2]
    return text.strip("_")


def _site_prefixed_cell_id(site, cell) -> str:
    site_text = _clean_token(site)
    cell_text = _rf_cell_id(cell)
    if not cell_text:
        return ""
    if not site_text:
        return cell_text
    return f"{site_text}_{cell_text}".strip("_")


def _print_fetch_summary(stage, table_name, filters, df, extra=None):
    print(f"[LTE_OPT][{stage}] source_table={table_name}")
    print(f"[LTE_OPT][{stage}] filters={filters}")
    print(f"[LTE_OPT][{stage}] row_count={len(df)}")
    print(f"[LTE_OPT][{stage}] columns={list(df.columns)}")
    if extra:
        for key, value in extra.items():
            print(f"[LTE_OPT][{stage}] {key}={value}")


def _ensure_canonical_identity(df):
    work = df.copy()
    if "Node_Cell_ID" in work.columns:
        work["Node_Cell_ID"] = work["Node_Cell_ID"].map(_rf_cell_id)
    elif "nodeb_id_cell_id" in work.columns:
        work["Node_Cell_ID"] = work["nodeb_id_cell_id"].map(_rf_cell_id)
    if "canonical_cell_id" in work.columns:
        work["canonical_cell_id"] = work["canonical_cell_id"].map(canonical_cell_id)
        return work
    if "Node_Cell_ID" in work.columns:
        work["canonical_cell_id"] = work["Node_Cell_ID"].map(canonical_cell_id)
    elif "nodeb_id_cell_id" in work.columns:
        work["canonical_cell_id"] = work["nodeb_id_cell_id"].map(canonical_cell_id)
    elif {"nodeb_id", "cell_id"}.issubset(work.columns):
        work["canonical_cell_id"] = [
            canonical_pair(nodeb, cell)
            for nodeb, cell in zip(work["nodeb_id"], work["cell_id"])
        ]
    elif "cell_id" in work.columns:
        work["canonical_cell_id"] = work["cell_id"].map(canonical_cell_id)
    else:
        work["canonical_cell_id"] = ""
    return work


def _normalize_site_df(site_df, log_stage="SITE_INPUT"):
    work = site_df.copy()
    work = work.rename(columns={
        "latitude": "lat",
        "longitude": "lon",
        "e_tilt": "electrical_tilt",
        "m_tilt": "mechanical_tilt",
        "height": "antenna_height",
        "Etilt": "electrical_tilt",
        "Mtilt": "mechanical_tilt",
        "Height": "antenna_height",
    })

    if "cell_id" in work.columns:
        if "Node_Cell_ID" in work.columns:
            work["Node_Cell_ID"] = work["Node_Cell_ID"].map(_rf_cell_id)
        elif "nodeb_id_cell_id" in work.columns:
            work["Node_Cell_ID"] = work["nodeb_id_cell_id"].map(_rf_cell_id)
        elif "site" in work.columns:
            work["Node_Cell_ID"] = [
                _site_prefixed_cell_id(site, cell)
                for site, cell in zip(work["site"], work["cell_id"])
            ]
        elif "nodeb_id" in work.columns:
            work["Node_Cell_ID"] = [
                _rf_cell_id(f"{_clean_token(nodeb)}_{_rf_cell_id(cell)}")
                for nodeb, cell in zip(work["nodeb_id"], work["cell_id"])
            ]
        else:
            work["Node_Cell_ID"] = work["cell_id"].map(_rf_cell_id)
        work["local_cell_id"] = work["cell_id"].map(_rf_cell_id)
    elif "Node_Cell_ID" in work.columns:
        work["Node_Cell_ID"] = work["Node_Cell_ID"].map(_rf_cell_id)
        work["local_cell_id"] = work["Node_Cell_ID"].map(canonical_cell_id)
    else:
        raise ValueError("Missing cell_id/Node_Cell_ID in optimized site input")
    work["canonical_cell_id"] = work["Node_Cell_ID"].map(canonical_cell_id)

    defaults = {
        "electrical_tilt": 0.0,
        "mechanical_tilt": 0.0,
        "antenna_height": 30.0,
        "azimuth": 0.0,
        "tx_power": 46.0,
        "lat": np.nan,
        "lon": np.nan,
        "frequency_mhz": 1800.0,
    }
    for col, default in defaults.items():
        if col not in work.columns:
            work[col] = default
        work[col] = pd.to_numeric(work[col], errors="coerce")
        if col in {"lat", "lon"}:
            continue
        missing_mask = work[col].isna()
        if missing_mask.any():
            work.loc[missing_mask, col] = float(default)
            if col == "tx_power":
                print(
                    f"[LTE_OPT][{log_stage}] {col}_source=default_fallback "
                    f"fallback_value={default} missing_rows={int(missing_mask.sum())}"
                )
        elif col == "tx_power":
            print(f"[LTE_OPT][{log_stage}] {col}_source=db missing_rows=0")

    work["nodeb_id"] = work["Node_Cell_ID"].str.split("_").str[0]

    dashboard_site_id = work["nodeb_id"].astype(str)
    dashboard_site_id = dashboard_site_id.where(
        ~dashboard_site_id.isin(["", "nan", "None"]),
        work["Node_Cell_ID"].str.split("_").str[0],
    )
    work["dashboard_site_id"] = dashboard_site_id.astype(str)
    return work


def _fetch_latest_baseline_job_id(project_id, region="india"):
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT job_id
    FROM lte_prediction_baseline_results
    WHERE project_id = {project_id}
    ORDER BY created_at DESC
    LIMIT 1
    """
    rows = pd.read_sql(query, current_engine)
    if rows.empty:
        raise FileNotFoundError(f"No baseline results found for project_id={project_id}")
    return str(rows.loc[0, "job_id"])


def fetch_baseline(project_id, region="india", operator=None, baseline_job_id=None):
    current_engine = engine.get(region.lower(), engine["india"])
    base_cols = ["lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr", "cell_id", "nodeb_id_cell_id", "job_id"]
    optional_identity_cols = ["operator"]
    topology_cols = ["best_interferer_cell_id", "neighbor_1_cell_id", "neighbor_2_cell_id"]
    select_cols = base_cols.copy()
    try:
        schema_df = pd.read_sql(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = DATABASE()
              AND table_name = 'lte_prediction_baseline_results'
            """,
            current_engine,
        )
        if not schema_df.empty:
            available = set(schema_df[schema_df.columns[0]].astype(str).tolist())
            select_cols.extend([col for col in optional_identity_cols + topology_cols if col in available])
    except Exception as exc:
        print(f"[LTE_OPT][BASELINE_FETCH] topology_column_probe_failed={exc}")
    if baseline_job_id is None:
        baseline_job_id = _fetch_latest_baseline_job_id(project_id, region=region)
    filters = ["project_id = %(project_id)s", "job_id = %(job_id)s"]
    params = {
        "project_id": int(project_id),
        "job_id": str(baseline_job_id),
    }
    if operator and "operator" in select_cols:
        filters.append("LOWER(TRIM(operator)) = %(operator)s")
        params["operator"] = str(operator).strip().lower()

    query = f"""
    SELECT {", ".join(select_cols)}
    FROM lte_prediction_baseline_results
    WHERE {" AND ".join(filters)}
    """

    df = pd.read_sql(query, current_engine, params=params)
    if "nodeb_id_cell_id" in df.columns:
        df["Node_Cell_ID"] = df["nodeb_id_cell_id"].map(_rf_cell_id)
        df["canonical_cell_id"] = df["nodeb_id_cell_id"].map(canonical_cell_id)
    else:
        df["Node_Cell_ID"] = df["cell_id"].map(_rf_cell_id)
        df["canonical_cell_id"] = df["cell_id"].map(canonical_cell_id)
    for col in topology_cols:
        if col in df.columns:
            df[col] = df[col].map(_rf_cell_id)
    df["rsrp"] = pd.to_numeric(df["pred_rsrp"], errors="coerce")
    _print_fetch_summary(
        "BASELINE_FETCH",
        "lte_prediction_baseline_results",
        {
            "project_id": project_id,
            "region": region,
            "operator": operator,
            "job_id": baseline_job_id,
            "mode": "latest_baseline_population",
        },
        df,
        extra={
            "distinct_cell_id": _safe_nunique(df, "cell_id"),
            "distinct_node_cell_id": _safe_nunique(df, "Node_Cell_ID"),
            "distinct_canonical_cell_id": _safe_nunique(df, "canonical_cell_id"),
            "distinct_job_id": _safe_nunique(df, "job_id"),
            "rsrp_range": _safe_minmax(df, "rsrp"),
        }
    )
    return df


def fetch_geo_features(project_id, region="india", affected_cells=None, baseline_job_id=None):
    current_engine = engine.get(region.lower(), engine["india"])
    query = """
    SELECT
        lat,
        lon,
        nodeb_id_cell_id,
        clutter_class,
        morphology_cluster,
        building_count,
        building_area_ratio,
        avg_building_area_m2,
        road_length_m,
        green_ratio,
        water_ratio,
        los_blocker_count,
        los_blocked_ratio,
        max_blocker_height_m,
        diffraction_proxy_db,
        nlos_flag,
        terrain_elevation_m,
        terrain_slope_deg,
        proxy_site_elevation_m,
        terrain_relief_to_site_m,
        site_count_250m,
        site_count_500m,
        serving_distance_m,
        nearest_site_distance_m,
        mean_nearest3_site_distance_m,
        azimuth_delta_deg
    FROM lte_prediction_geo_features
    WHERE project_id = %(project_id)s
      AND region = %(region)s
    """
    params = {
        "project_id": int(project_id),
        "region": str(region).lower(),
    }
    affected_cell_set = {_rf_cell_id(x) for x in (affected_cells or []) if _rf_cell_id(x)}

    df = pd.read_sql(query, current_engine, params=params)
    if "nodeb_id_cell_id" in df.columns:
        df["Node_Cell_ID"] = df["nodeb_id_cell_id"].map(_rf_cell_id)
    else:
        df["Node_Cell_ID"] = pd.Series(dtype=str)
    if affected_cell_set:
        df = df.loc[df["Node_Cell_ID"].astype(str).isin(affected_cell_set)].copy()
    _print_fetch_summary(
        "GEO_FETCH",
        "lte_prediction_geo_features",
        {
            "project_id": project_id,
            "region": region,
            "mode": "current_project_state",
            "affected_cell_count": len(affected_cells or []),
        },
        df,
        extra={
            "distinct_node_cell_id": _safe_nunique(df, "Node_Cell_ID"),
            "distinct_baseline_job_id": _safe_nunique(df, "baseline_job_id") if "baseline_job_id" in df.columns else "n/a",
            "lat_range": _safe_minmax(df, "lat"),
            "lon_range": _safe_minmax(df, "lon"),
        }
    )
    return df


def _apply_saved_geo_correction(pts_df, site_df, project_id=None):
    if pts_df.empty:
        return pts_df
    corrected = pts_df.copy()
    weights, weights_summary = load_geo_weights(project_id=project_id)
    corrected = attach_site_context_features(corrected, site_df)
    corrected = _refine_experimental_forward_features(corrected)
    corrected = attach_fixed_serving_sinr_rsrq_proxy(corrected, site_df)
    corrected, geo_summary = apply_experimental_geo_adjustments(corrected, weights=weights)
    corrected["pred_rsrp"] = pd.to_numeric(
        corrected.get("pred_rsrp_geo", corrected["pred_rsrp"]),
        errors="coerce",
    ).clip(-140, -44)
    corrected["pred_rsrq"] = pd.to_numeric(
        corrected.get("pred_rsrq_geo", corrected["pred_rsrq"]),
        errors="coerce",
    ).clip(-20, -3)
    corrected["pred_sinr"] = pd.to_numeric(
        corrected.get("pred_sinr_geo", corrected["pred_sinr"]),
        errors="coerce",
    ).clip(-10, 30)
    print(
        f"[LTE_OPT][GEO_CORRECTION] project_id={project_id} rows={len(corrected)} "
        f"mode={geo_summary.get('mode')} weights_project={weights_summary.get('project_id')}"
    )
    return corrected


def _expand_site_rows_to_allowed_cells(df: pd.DataFrame, allowed_cells) -> pd.DataFrame:
    allowed = {_rf_cell_id(cell) for cell in allowed_cells if _rf_cell_id(cell)}
    if not allowed or df.empty:
        return df.iloc[0:0].copy() if allowed_cells is not None else df

    expanded = []
    for _, row in df.iterrows():
        aliases = {
            _rf_cell_id(row.get("Node_Cell_ID")),
            _rf_cell_id(row.get("cell_id")),
            _rf_cell_id(row.get("local_cell_id")),
            _site_prefixed_cell_id(row.get("site"), row.get("cell_id")),
            _site_prefixed_cell_id(row.get("site"), row.get("local_cell_id")),
        }
        aliases.update({canonical_cell_id(alias) for alias in list(aliases) if alias})
        matched_aliases = sorted(alias for alias in aliases if alias and alias in allowed)
        for alias in matched_aliases:
            new_row = row.copy()
            new_row["Node_Cell_ID"] = alias
            new_row["canonical_cell_id"] = canonical_cell_id(alias)
            new_row["nodeb_id"] = alias.split("_")[0] if "_" in alias else alias
            new_row["dashboard_site_id"] = new_row["nodeb_id"]
            expanded.append(new_row)

    if not expanded:
        return df.iloc[0:0].copy()
    out = pd.DataFrame(expanded).reset_index(drop=True)
    return out


def fetch_site_data(project_id, region="india", operator=None, allowed_cells=None):
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT *
    FROM site_prediction
    WHERE tbl_project_id = {project_id}
    """

    raw_df = pd.read_sql(query, current_engine)
    df = _normalize_site_df(raw_df, log_stage="SITE_FETCH")
    if operator:
        operator_norm = str(operator).strip().lower()
        if "cluster" in df.columns:
            cluster_series = df["cluster"].astype(str).str.strip().str.lower()
            before_rows = len(df)
            df = df.loc[cluster_series == operator_norm].copy()
            print(
                f"[LTE_OPT][SITE_FETCH] operator_filter={operator} "
                f"rows_before={before_rows} rows_after={len(df)}"
            )
    if allowed_cells is not None:
        before_rows = len(df)
        before_cells = _safe_nunique(df, "Node_Cell_ID")
        df = _expand_site_rows_to_allowed_cells(df, allowed_cells)
        print(
            f"[LTE_OPT][SITE_FETCH] baseline_population_filter=True "
            f"rows_before={before_rows} rows_after={len(df)} "
            f"cells_before={before_cells} cells_after={_safe_nunique(df, 'Node_Cell_ID')}"
        )
    _print_fetch_summary(
        "SITE_FETCH",
        "site_prediction",
        {"project_id": project_id, "region": region, "operator": operator},
        df,
        extra={
            "distinct_pci": _safe_nunique(df, "pci"),
            "distinct_cell_id": _safe_nunique(df, "cell_id"),
            "distinct_nodeb_id": _safe_nunique(df, "nodeb_id"),
        }
    )
    return df


def fetch_optimized_sites(project_id, operator, region="india"):
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT *
    FROM site_prediction_optimized
    WHERE tbl_project_id = {project_id}
    AND cluster = '{operator}'
    """

    opt_raw = pd.read_sql(query, current_engine)
    if opt_raw.empty:
        _print_fetch_summary(
            "OPTIMIZED_SITE_FETCH",
            "site_prediction_optimized",
            {"project_id": project_id, "operator": operator, "region": region},
            opt_raw,
        )
        return opt_raw

    current_df = fetch_site_data(project_id, region=region, operator=operator)
    opt_df = _normalize_site_df(opt_raw, log_stage="OPTIMIZED_SITE_FETCH")
    sort_cols = [col for col in ["version", "updated_at", "created_at", "id", "Node_Cell_ID"] if col in opt_df.columns]
    if sort_cols:
        opt_df = opt_df.sort_values(sort_cols)
    opt_df = opt_df.drop_duplicates(subset=["Node_Cell_ID"], keep="last")

    compare_cols = [
        "lat",
        "lon",
        "azimuth",
        "electrical_tilt",
        "mechanical_tilt",
        "tx_power",
        "antenna_height",
    ]
    merged_df = current_df.copy()
    for col in compare_cols:
        merged_df[f"orig_{col}"] = pd.to_numeric(merged_df[col], errors="coerce")

    opt_df = opt_df.set_index("Node_Cell_ID", drop=False)
    mask = merged_df["Node_Cell_ID"].astype(str).isin(opt_df.index.astype(str))
    overlay_cols = compare_cols + ["frequency_mhz", "dashboard_site_id", "nodeb_id"]
    for col in overlay_cols:
        if col in opt_df.columns:
            mapping = opt_df[col].to_dict()
            merged_df.loc[mask, col] = merged_df.loc[mask, "Node_Cell_ID"].astype(str).map(mapping)

    merged_df["optimization_applied"] = mask
    changed_mask = _build_change_mask(merged_df)
    _print_fetch_summary(
        "OPTIMIZED_SITE_FETCH",
        "site_prediction_optimized",
        {"project_id": project_id, "operator": operator, "region": region},
        merged_df,
        extra={
            "optimized_rows": len(opt_df),
            "overlay_rows": int(mask.sum()),
            "changed_rows": int(changed_mask.sum()),
            "changed_cells": int(merged_df.loc[changed_mask, "Node_Cell_ID"].nunique()) if changed_mask.any() else 0,
            "distinct_pci": _safe_nunique(merged_df, "pci"),
            "distinct_cell_id": _safe_nunique(merged_df, "cell_id"),
            "distinct_nodeb_id": _safe_nunique(merged_df, "nodeb_id"),
        }
    )
    return merged_df


def compute_k1k2(baseline_df, site_df):
    k1k2_map = {}
    total_cells = int(site_df["Node_Cell_ID"].nunique())
    print(f"[LTE_OPT][K1K2] total_site_cells={total_cells}")

    baseline_work = _ensure_canonical_identity(baseline_df)
    site_work = _ensure_canonical_identity(site_df)
    for cid in site_work["Node_Cell_ID"].astype(str).unique():
        site_rows = site_work[site_work["Node_Cell_ID"].astype(str) == str(cid)].copy()
        dt_rows = baseline_work[baseline_work["Node_Cell_ID"].astype(str) == str(cid)].copy()
        print(
            f"[LTE_OPT][K1K2] cell={cid} site_rows={len(site_rows)} "
            f"baseline_rows={len(dt_rows)}"
        )

        if len(dt_rows) < 10:
            print(f"[LTE_OPT][K1K2] cell={cid} skipped_reason=baseline_rows_lt_10")
            continue

        freq = float(pd.to_numeric(site_rows["frequency_mhz"], errors="coerce").fillna(1800.0).iloc[0])
        tx_power = float(pd.to_numeric(site_rows["tx_power"], errors="coerce").fillna(46.0).iloc[0])
        k1, k2 = calibrate_site(
            dt_rows,
            site_rows,
            tx_power,
            18,
            2,
            freq
        )
        k1k2_map[str(cid)] = (float(k1), float(k2))
        print(f"[LTE_OPT][K1K2] cell={cid} calibrated_k1={k1:.4f} calibrated_k2={k2:.4f}")

    print(f"[LTE_OPT][K1K2] calibrated_cells={len(k1k2_map)}")
    return k1k2_map


def _build_change_mask(site_df):
    if site_df.empty:
        return pd.Series(dtype=bool)
    compare_cols = [
        "lat",
        "lon",
        "azimuth",
        "electrical_tilt",
        "mechanical_tilt",
        "tx_power",
        "antenna_height",
    ]
    changed_mask = pd.Series(False, index=site_df.index)
    for col in compare_cols:
        orig_col = f"orig_{col}"
        if orig_col not in site_df.columns or col not in site_df.columns:
            continue
        before = pd.to_numeric(site_df[orig_col], errors="coerce").fillna(-999999.0)
        after = pd.to_numeric(site_df[col], errors="coerce").fillna(-999999.0)
        changed_mask = changed_mask | (~np.isclose(before, after, equal_nan=True))
    return changed_mask


def _select_target_rows(site_df, target_type, target_id):
    target_id = str(target_id).strip()
    if not target_id:
        raise ValueError("target_id is required")
    if target_type == "cell":
        mask = site_df["Node_Cell_ID"].astype(str) == target_id
    elif target_type == "site":
        mask = site_df["dashboard_site_id"].astype(str) == target_id
    else:
        raise ValueError("target_type must be 'site' or 'cell'")
    if not mask.any():
        raise ValueError(f"No site rows matched target_type={target_type} target_id={target_id}")
    return mask


def build_runtime_optimized_sites(site_df, cfg):
    original = _normalize_site_df(site_df, log_stage="OPTIMIZED_RUNTIME_INPUT")
    modified = original.copy()
    target_type = str(cfg.get("target_type", "")).strip().lower()
    target_id = str(cfg.get("target_id", "")).strip()
    if not target_type or not target_id:
        raise ValueError("target_type and target_id are required for runtime optimization input")

    target_mask = _select_target_rows(modified, target_type, target_id)

    for col in ["lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "tx_power", "antenna_height"]:
        modified[f"orig_{col}"] = pd.to_numeric(original[col], errors="coerce")

    delta_map = {
        "lat": float(cfg.get("delta_lat", 0.0) or 0.0),
        "lon": float(cfg.get("delta_lon", 0.0) or 0.0),
        "azimuth": float(cfg.get("delta_azimuth", 0.0) or 0.0),
        "electrical_tilt": float(cfg.get("delta_electrical_tilt", 0.0) or 0.0),
        "mechanical_tilt": float(cfg.get("delta_mechanical_tilt", 0.0) or 0.0),
        "tx_power": float(cfg.get("delta_tx_power", 0.0) or 0.0),
        "antenna_height": float(cfg.get("delta_antenna_height", 0.0) or 0.0),
    }

    if delta_map["lat"]:
        modified.loc[target_mask, "lat"] = pd.to_numeric(modified.loc[target_mask, "lat"], errors="coerce") + delta_map["lat"]
    if delta_map["lon"]:
        modified.loc[target_mask, "lon"] = pd.to_numeric(modified.loc[target_mask, "lon"], errors="coerce") + delta_map["lon"]
    if delta_map["azimuth"]:
        modified.loc[target_mask, "azimuth"] = (
            pd.to_numeric(modified.loc[target_mask, "azimuth"], errors="coerce") + delta_map["azimuth"]
        ) % 360.0
    if delta_map["electrical_tilt"]:
        modified.loc[target_mask, "electrical_tilt"] = (
            pd.to_numeric(modified.loc[target_mask, "electrical_tilt"], errors="coerce") + delta_map["electrical_tilt"]
        )
    if delta_map["mechanical_tilt"]:
        modified.loc[target_mask, "mechanical_tilt"] = (
            pd.to_numeric(modified.loc[target_mask, "mechanical_tilt"], errors="coerce") + delta_map["mechanical_tilt"]
        )
    if delta_map["tx_power"]:
        modified.loc[target_mask, "tx_power"] = (
            pd.to_numeric(modified.loc[target_mask, "tx_power"], errors="coerce") + delta_map["tx_power"]
        )
    if delta_map["antenna_height"]:
        modified.loc[target_mask, "antenna_height"] = (
            pd.to_numeric(modified.loc[target_mask, "antenna_height"], errors="coerce") + delta_map["antenna_height"]
        )

    modified["optimization_applied"] = target_mask.astype(bool)
    changed_mask = _build_change_mask(modified)
    if not changed_mask.any():
        raise ValueError("No effective site change detected from the provided deltas")

    changed_rows = modified.loc[changed_mask].copy()
    print(
        f"[LTE_OPT][RUNTIME_CHANGE] target_type={target_type} target_id={target_id} "
        f"changed_rows={len(changed_rows)} changed_cells={changed_rows['Node_Cell_ID'].nunique()}"
    )
    return modified


def compute_k1k2_for_cells(baseline_df, site_df, target_cells):
    k1k2_map = {}
    baseline_work = _ensure_canonical_identity(baseline_df)
    site_work = _ensure_canonical_identity(site_df)
    for cid in sorted({str(x) for x in target_cells}):
        site_rows = site_work[site_work["Node_Cell_ID"].astype(str) == str(cid)].copy()
        dt_rows = baseline_work[baseline_work["Node_Cell_ID"].astype(str) == str(cid)].copy()
        print(
            f"[LTE_OPT][K1K2_LOCAL] cell={cid} site_rows={len(site_rows)} "
            f"baseline_rows={len(dt_rows)}"
        )
        if site_rows.empty or len(dt_rows) < 10:
            print(f"[LTE_OPT][K1K2_LOCAL] cell={cid} skipped_reason=baseline_rows_lt_10")
            continue
        freq = float(pd.to_numeric(site_rows["frequency_mhz"], errors="coerce").fillna(1800.0).iloc[0])
        tx_power = float(pd.to_numeric(site_rows["tx_power"], errors="coerce").fillna(46.0).iloc[0])
        k1, k2 = calibrate_site(dt_rows, site_rows, tx_power, 18, 2, freq)
        k1k2_map[str(cid)] = (float(k1), float(k2))
        print(f"[LTE_OPT][K1K2_LOCAL] cell={cid} calibrated_k1={k1:.4f} calibrated_k2={k2:.4f}")
    print(f"[LTE_OPT][K1K2_LOCAL] calibrated_cells={len(k1k2_map)}")
    return k1k2_map


def _clean_cell_id(value):
    return canonical_cell_id(value)


def _compute_affected_cells(
    site_df,
    impact_radius_m,
    neighbor_site_count,
    baseline_df=None,
    max_neighbors_per_update_cell=None,
):
    site_work = site_df.copy()
    changed_mask = _build_change_mask(site_work)
    changed_rows = site_work.loc[changed_mask].copy()
    if changed_rows.empty:
        raise ValueError("No effective optimized site change detected")

    changed_cell_ids = sorted(changed_rows["Node_Cell_ID"].astype(str).unique().tolist())
    changed_site_ids = sorted(changed_rows["dashboard_site_id"].astype(str).unique().tolist())
    selected_cells = set(changed_cell_ids)

    same_site_rows = site_work.loc[site_work["dashboard_site_id"].astype(str).isin(changed_site_ids)].copy()
    selected_cells.update(same_site_rows["Node_Cell_ID"].astype(str).tolist())

    available_cells = set(site_work["Node_Cell_ID"].astype(str).tolist())
    max_neighbors = max_neighbors_per_update_cell
    if max_neighbors is None:
        max_neighbors = neighbor_site_count
    max_neighbors = max(0, int(max_neighbors or 0)) * max(1, len(changed_cell_ids))

    neighbor_counts = {}
    topology_cols = []
    if isinstance(baseline_df, pd.DataFrame) and not baseline_df.empty:
        baseline_work = _ensure_canonical_identity(baseline_df)
        topology_cols = [
            col for col in ["best_interferer_cell_id", "neighbor_1_cell_id", "neighbor_2_cell_id"]
            if col in baseline_work.columns
        ]
        if topology_cols and "Node_Cell_ID" in baseline_work.columns:
            focus = baseline_work.loc[baseline_work["Node_Cell_ID"].astype(str).isin(changed_cell_ids)]
            for col in topology_cols:
                for value in focus[col].dropna().astype(str):
                    cell = _clean_cell_id(value)
                    if cell and cell not in selected_cells:
                        neighbor_counts[cell] = neighbor_counts.get(cell, 0) + 1

    selected_neighbor_cells = [
        cell for cell, _ in sorted(neighbor_counts.items(), key=lambda item: (-item[1], item[0]))
        if cell in available_cells
    ][:max_neighbors]
    selected_cells.update(selected_neighbor_cells)

    if not topology_cols or not selected_neighbor_cells:
        candidate_rows = site_work.loc[~site_work["Node_Cell_ID"].astype(str).isin(selected_cells)].copy()
        candidate_rows["lat"] = pd.to_numeric(candidate_rows["lat"], errors="coerce")
        candidate_rows["lon"] = pd.to_numeric(candidate_rows["lon"], errors="coerce")
        ranked_parts = []
        for _, row in changed_rows.iterrows():
            center_lat = pd.to_numeric(pd.Series([row.get("lat")]), errors="coerce").iloc[0]
            center_lon = pd.to_numeric(pd.Series([row.get("lon")]), errors="coerce").iloc[0]
            if pd.isna(center_lat) or pd.isna(center_lon) or candidate_rows.empty:
                continue
            ranked = candidate_rows.copy()
            ranked["distance_m"] = haversine_vectorized(
                float(center_lat),
                float(center_lon),
                ranked["lat"].to_numpy(dtype=float, copy=False),
                ranked["lon"].to_numpy(dtype=float, copy=False),
            )
            ranked = ranked.loc[ranked["distance_m"] <= float(impact_radius_m)].copy()
            if ranked.empty:
                continue
            ranked = ranked.sort_values(["distance_m", "Node_Cell_ID"], ascending=[True, True])
            ranked_parts.append(ranked)
        if ranked_parts and max_neighbors > 0:
            ranked_cells = (
                pd.concat(ranked_parts, ignore_index=True)
                .sort_values(["distance_m", "Node_Cell_ID"], ascending=[True, True])
                .drop_duplicates(subset=["Node_Cell_ID"], keep="first")
                .head(max_neighbors)
            )
            distance_neighbors = ranked_cells["Node_Cell_ID"].astype(str).tolist()
            selected_cells.update(distance_neighbors)
        else:
            distance_neighbors = []
    else:
        distance_neighbors = []

    affected_rows = site_work.loc[site_work["Node_Cell_ID"].astype(str).isin(selected_cells)].copy()
    affected_ids = sorted(affected_rows["Node_Cell_ID"].astype(str).unique().tolist())
    affected_site_ids = sorted(affected_rows["dashboard_site_id"].astype(str).unique().tolist())
    print(
        f"[LTE_OPT][AFFECTED_CELL_SCOPE] changed_cells={len(changed_cell_ids)} "
        f"same_site_cells={same_site_rows['Node_Cell_ID'].nunique()} "
        f"topology_neighbor_cells={len(selected_neighbor_cells)} "
        f"distance_neighbor_cells={len(distance_neighbors)} affected_cells={len(affected_ids)} "
        f"scope=changed_cells_plus_same_site_cells_plus_neighbor_cells"
    )
    return affected_ids, affected_site_ids, changed_rows


def _build_local_interference_records(full_site_df, site_rows, max_interference_sites):
    if max_interference_sites <= 0 or len(full_site_df) <= max_interference_sites:
        return full_site_df.to_dict("records")

    work = full_site_df.copy()
    clat = float(pd.to_numeric(site_rows["lat"], errors="coerce").mean())
    clon = float(pd.to_numeric(site_rows["lon"], errors="coerce").mean())
    work["_distance_m"] = haversine_vectorized(
        clat,
        clon,
        pd.to_numeric(work["lat"], errors="coerce").to_numpy(dtype=float, copy=False),
        pd.to_numeric(work["lon"], errors="coerce").to_numpy(dtype=float, copy=False),
    )
    serving_ids = set(site_rows["Node_Cell_ID"].astype(str).tolist())
    nearest_df = work.nsmallest(int(max_interference_sites), "_distance_m")
    serving_df = work[work["Node_Cell_ID"].astype(str).isin(serving_ids)]
    combined = (
        pd.concat([nearest_df, serving_df], ignore_index=True)
        .drop_duplicates(subset=["Node_Cell_ID"], keep="first")
        .drop(columns=["_distance_m"], errors="ignore")
    )
    return combined.to_dict("records")


def run_prediction_only_optimized(opt_sites, k1k2_map, params):
    if opt_sites.empty:
        return pd.DataFrame(columns=["lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr", "Node_Cell_ID"])

    work_df = _normalize_site_df(opt_sites, log_stage="OPTIMIZED_RUN")
    impact_radius_m = float(params.get("impact_radius_m", params.get("radius", 500)))
    neighbor_site_count = int(params.get("neighbor_site_count", 2))
    max_interference_sites = int(params.get("max_interference_sites", 10))
    prediction_points_df = params.get("prediction_points_df")
    strict_prediction_points = bool(params.get("strict_prediction_points", False))
    explicit_recompute_cells = params.get("recompute_cells")
    if isinstance(prediction_points_df, pd.DataFrame) and not prediction_points_df.empty:
        prediction_points_df = prediction_points_df.copy()
        prediction_points_df = _ensure_canonical_identity(prediction_points_df)
        if "Node_Cell_ID" in prediction_points_df.columns:
            prediction_points_df["Node_Cell_ID"] = prediction_points_df["Node_Cell_ID"].astype(str)
        print(
            f"[LTE_OPT][POINTS_OVERRIDE] enabled=True rows={len(prediction_points_df)} "
            f"distinct_node_cell_id={_safe_nunique(prediction_points_df, 'Node_Cell_ID')} "
            f"distinct_canonical_cell_id={_safe_nunique(prediction_points_df, 'canonical_cell_id')}"
        )
    else:
        prediction_points_df = pd.DataFrame()

    if explicit_recompute_cells is not None:
        affected_cells = sorted({str(cell).strip() for cell in explicit_recompute_cells if str(cell).strip()})
        available_cell_ids = set(work_df["Node_Cell_ID"].astype(str).tolist())
        affected_cells = [
            cell for cell in affected_cells
            if cell in available_cell_ids
        ]
        affected_rows = work_df.loc[work_df["Node_Cell_ID"].astype(str).isin(affected_cells)].copy()
        affected_sites = sorted(affected_rows["dashboard_site_id"].astype(str).dropna().unique().tolist())
        changed_rows = work_df.loc[_build_change_mask(work_df)].copy()
        print(
            f"[LTE_OPT][AFFECTED_OVERRIDE] update_cell_count={changed_rows['Node_Cell_ID'].nunique()} "
            f"recompute_cell_count={len(affected_cells)} affected_site_count={len(affected_sites)} "
            f"source=explicit_recompute_cells"
        )
    else:
        affected_cells, affected_sites, changed_rows = _compute_affected_cells(
            work_df,
            impact_radius_m,
            neighbor_site_count,
            baseline_df=params.get("baseline_df"),
            max_neighbors_per_update_cell=params.get("max_neighbors_per_update_cell"),
        )
    print(
        f"[LTE_OPT][AFFECTED] changed_cell_count={changed_rows['Node_Cell_ID'].nunique()} "
        f"affected_site_count={len(affected_sites)} affected_cell_count={len(affected_cells)} "
        f"impact_radius_m={impact_radius_m} neighbor_site_count={neighbor_site_count}"
    )

    final_list = []
    print(f"[LTE_OPT][RUN] total_cells_to_process={len(affected_cells)}")
    geo_features_df = params.get("geo_features_df")
    if isinstance(geo_features_df, pd.DataFrame) and not geo_features_df.empty:
        geo_features_df = geo_features_df.copy()
        if "nodeb_id_cell_id" in geo_features_df.columns and "Node_Cell_ID" not in geo_features_df.columns:
            geo_features_df["Node_Cell_ID"] = geo_features_df["nodeb_id_cell_id"].astype(str)
        print(
            f"[LTE_OPT][GEO_FETCH] source=local_geo_features rows={len(geo_features_df)} "
            f"distinct_node_cell_id={_safe_nunique(geo_features_df, 'Node_Cell_ID')}"
        )
    else:
        geo_features_df = pd.DataFrame()
    if geo_features_df.empty and params.get("project_id"):
        try:
            geo_features_df = fetch_geo_features(
                params["project_id"],
                region=str(params.get("region", "india")).lower(),
                affected_cells=affected_cells,
                baseline_job_id=params.get("baseline_job_id"),
            )
        except Exception as exc:
            print(f"[LTE_OPT][GEO_FETCH] enabled=False reason={exc}")
            geo_features_df = pd.DataFrame()

    for cid in affected_cells:
        print(f"[LTE_OPT][RUN] cell_start={cid}")
        site_rows = work_df[work_df["Node_Cell_ID"].astype(str) == str(cid)].copy()
        if site_rows.empty:
            continue
        k1, k2 = k1k2_map.get(str(cid), (0.0, 0.0))
        local_interference_records = _build_local_interference_records(
            work_df,
            site_rows,
            max_interference_sites,
        )

        print(
            f"[LTE_OPT][RUN] cell={cid} site_rows={len(site_rows)} "
            f"k1={k1} k2={k2} radius={params.get('radius')} "
            f"grid_resolution={params.get('grid_resolution')} "
            f"interference_site_rows={len(local_interference_records)}"
        )

        cell_params = params.copy()
        cell_params.update({
            "k1": k1,
            "k2": k2,
            "all_sites_rows": local_interference_records
        })
        point_source = "generated_grid"
        if not prediction_points_df.empty and {"Node_Cell_ID", "lat", "lon"}.issubset(prediction_points_df.columns):
            point_cols = [
                col for col in [
                    "lat",
                    "lon",
                    "Node_Cell_ID",
                    "nodeb_id_cell_id",
                    "cell_id",
                    "node_b_id",
                    "site_id",
                    "sector",
                    "frontend_site_sector_key",
                    "operator",
                    "Technology",
                    "canonical_cell_id",
                ] if col in prediction_points_df.columns
            ]
            pts = (
                prediction_points_df.loc[
                    prediction_points_df["Node_Cell_ID"].astype(str) == str(cid),
                    point_cols,
                ]
                .dropna(subset=["lat", "lon"])
                .drop_duplicates()
                .copy()
            )
            point_source = "baseline_prediction_points"
        else:
            pts = pd.DataFrame()
        if pts.empty and strict_prediction_points and not prediction_points_df.empty:
            print(f"[LTE_OPT][RUN] cell={cid} skipped_reason=no_prediction_points_strict")
            continue
        if pts.empty:
            pts = generate_grid(
                site_rows,
                cell_params["radius"],
                cell_params["grid_resolution"]
            )
            point_source = "generated_grid"
        cell_geo = geo_features_df[geo_features_df["Node_Cell_ID"].astype(str) == str(cid)].copy() if not geo_features_df.empty else pd.DataFrame()
        if point_source == "generated_grid" and not cell_geo.empty and not pts.empty:
            geo_mask = (
                cell_geo.loc[:, ["lat", "lon"]]
                .dropna(subset=["lat", "lon"])
                .drop_duplicates()
                .copy()
            )
            geo_mask["lat_6dp"] = pd.to_numeric(geo_mask["lat"], errors="coerce").round(6)
            geo_mask["lon_6dp"] = pd.to_numeric(geo_mask["lon"], errors="coerce").round(6)
            pts["lat_6dp"] = pd.to_numeric(pts["lat"], errors="coerce").round(6)
            pts["lon_6dp"] = pd.to_numeric(pts["lon"], errors="coerce").round(6)
            masked_pts = pts.merge(
                geo_mask[["lat_6dp", "lon_6dp"]],
                on=["lat_6dp", "lon_6dp"],
                how="inner",
            )
            if not masked_pts.empty:
                pts = masked_pts.drop(columns=["lat_6dp", "lon_6dp"], errors="ignore")
                point_source = "generated_grid_geo_mask"
            else:
                pts = pts.drop(columns=["lat_6dp", "lon_6dp"], errors="ignore")
        print(f"[LTE_OPT][RUN] cell={cid} grid_points={len(pts)} point_source={point_source}")

        start = time.time()
        rsrp, rsrq, sinr = compute_predictions_parallel(
            pts,
            site_rows,
            cell_params,
            n_workers=cell_params.get("n_workers")
        )
        elapsed = round(time.time() - start, 2)

        pts["pred_rsrp"] = np.clip(rsrp, -140, -44)
        pts["pred_rsrq"] = np.clip(rsrq, -20, -3)
        pts["pred_sinr"] = np.clip(sinr, -10, 30)
        if "Node_Cell_ID" not in pts.columns:
            pts["Node_Cell_ID"] = str(cid)
        if "canonical_cell_id" not in pts.columns:
            pts["canonical_cell_id"] = str(cid)
        if not cell_geo.empty:
            geo_merge_cols = [
                col for col in [
                    "Node_Cell_ID",
                    "lat",
                    "lon",
                    "clutter_class",
                    "morphology_cluster",
                    "building_count",
                    "building_area_ratio",
                    "avg_building_area_m2",
                    "road_length_m",
                    "green_ratio",
                    "water_ratio",
                    "los_blocker_count",
                    "los_blocked_ratio",
                    "max_blocker_height_m",
                    "diffraction_proxy_db",
                    "nlos_flag",
                    "terrain_elevation_m",
                    "terrain_slope_deg",
                    "proxy_site_elevation_m",
                    "terrain_relief_to_site_m",
                    "site_count_250m",
                    "site_count_500m",
                    "serving_distance_m",
                    "nearest_site_distance_m",
                    "mean_nearest3_site_distance_m",
                    "azimuth_delta_deg",
                ] if col in cell_geo.columns
            ]
            pts = pts.merge(
                cell_geo[geo_merge_cols].drop_duplicates(subset=["Node_Cell_ID", "lat", "lon"], keep="last"),
                on=["Node_Cell_ID", "lat", "lon"],
                how="left",
            )
            pts = _apply_saved_geo_correction(
                pts,
                work_df,
                project_id=params.get("project_id"),
            )

        print(
            f"[LTE_OPT][RUN] cell={cid} elapsed_sec={elapsed} "
            f"pred_rsrp_range={_safe_minmax(pts, 'pred_rsrp')} "
            f"pred_rsrq_range={_safe_minmax(pts, 'pred_rsrq')} "
            f"pred_sinr_range={_safe_minmax(pts, 'pred_sinr')}"
        )
        final_list.append(pts)

    final_df = pd.concat(final_list, ignore_index=True)
    _print_fetch_summary(
        "OPTIMIZED_RF_OUTPUT",
        "in_memory_optimized_prediction",
        {"cells_processed": len(affected_cells)},
        final_df,
        extra={
            "distinct_node_cell_id": _safe_nunique(final_df, "Node_Cell_ID"),
            "pred_rsrp_range": _safe_minmax(final_df, "pred_rsrp"),
            "pred_rsrq_range": _safe_minmax(final_df, "pred_rsrq"),
            "pred_sinr_range": _safe_minmax(final_df, "pred_sinr"),
        }
    )
    return final_df


def replace_cells(baseline_df, optimized_df):
    baseline_df = _ensure_canonical_identity(baseline_df)
    optimized_df = _ensure_canonical_identity(optimized_df)
    replace_ids = optimized_df["Node_Cell_ID"].astype(str).unique()
    baseline_df = baseline_df[
        ~baseline_df["Node_Cell_ID"].astype(str).isin(replace_ids)
    ]
    final_df = pd.concat([baseline_df, optimized_df], ignore_index=True)
    print(
        f"[LTE_OPT][REPLACE] replace_cell_count={len(replace_ids)} "
        f"remaining_baseline_rows={len(baseline_df)} final_rows={len(final_df)}"
    )
    return final_df
