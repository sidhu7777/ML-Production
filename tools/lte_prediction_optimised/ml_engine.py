import os
import time
import json
from functools import lru_cache
from pathlib import Path

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
from ..lte_prediction_offset.services import (
    _azimuth_delta_deg as _offset_azimuth_delta_deg,
    _bearing_deg as _offset_bearing_deg,
    _cost231_for_points as _offset_cost231_for_points,
    _haversine_m as _offset_haversine_m,
    _prepare_site_rows as _offset_prepare_site_rows,
    _site_record as _offset_site_record,
)
from ..lte_prediction_offset.phase27_calibration import add_features as _offset_add_features
from ..lte_prediction_offset.phase27_physical import score_candidates as _offset_score_candidates
from ..lte_prediction_offset.phase36_physical_upgrades import (
    apply_reference_and_water as _offset_apply_phase36,
    n_rb_for as _offset_n_rb_for,
)
from ..lte_prediction.geo_correction_pipeline import (
    load_geo_weights,
    attach_site_context_features,
    attach_fixed_serving_sinr_rsrq_proxy,
    _refine_experimental_forward_features,
    apply_experimental_geo_adjustments,
)
from ..lte_tilt_recommandation.cell_identity import (
    build_rf_identity,
    build_sector_identity,
    build_site_sector_band_identity,
    canonical_cell_id,
    canonical_pair,
)
from utils.python_bridge import _filter_complete_site_prediction_identity, get_bridge_client


ML_ROOT = Path(__file__).resolve().parents[2]


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


# Pure scalar id-normalisers, called millions of times per RF evaluation over a
# few hundred distinct ids (profiled: 5.9M _rf_cell_id calls). Memoise the str
# case only -- str hashes exactly so the cached result is identical, while
# non-str falls through uncached to avoid hash(1)==hash(True) style collisions.
def _clean_token_impl(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null", "<na>"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


_clean_token_cached = lru_cache(maxsize=200_000)(_clean_token_impl)


def _clean_token(value) -> str:
    if type(value) is str:
        return _clean_token_cached(value)
    return _clean_token_impl(value)


def _rf_cell_id_impl(value) -> str:
    text = _clean_token(value).replace("|", "_")
    while ".0_" in text:
        text = text.replace(".0_", "_")
    if text.endswith(".0"):
        text = text[:-2]
    return text.strip("_")


_rf_cell_id_cached = lru_cache(maxsize=200_000)(_rf_cell_id_impl)


def _rf_cell_id(value) -> str:
    if type(value) is str:
        return _rf_cell_id_cached(value)
    return _rf_cell_id_impl(value)


def _site_prefixed_cell_id(site, cell) -> str:
    site_text = _clean_token(site)
    cell_text = _rf_cell_id(cell)
    if not cell_text:
        return ""
    if not site_text:
        return cell_text
    return f"{site_text}_{cell_text}".strip("_")


def _pick_col(df: pd.DataFrame, candidates: list[str]):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _technology_series(df: pd.DataFrame) -> pd.Series:
    tech_col = _pick_col(df, ["Technology", "technology", "network_type", "rat", "tech"])
    if tech_col:
        tech = df[tech_col].astype("string").str.strip().str.upper()
    else:
        tech = pd.Series(pd.NA, index=df.index, dtype="string")

    tech = tech.mask(tech.str.contains("5G|NR", na=False), "5G")
    tech = tech.mask(tech.str.contains("4G|LTE", na=False), "4G")
    tech = tech.mask(tech.isna() | tech.isin(["", "NAN", "NONE", "NULL", "<NA>"]), "4G")
    return tech.astype(str)


def _clean_identity_series(series: pd.Series) -> pd.Series:
    return series.map(_rf_cell_id).replace("", pd.NA)


def _attach_rf_identity_columns(df: pd.DataFrame, prefer_rf_node_cell_id: bool = True) -> pd.DataFrame:
    if df.empty:
        for col in ["legacy_nodeb_id_cell_id", "sector_identity_key", "site_sector_band_key", "rf_identity_key"]:
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        return df

    work = df.copy()
    fallback_col = _pick_col(work, ["legacy_nodeb_id_cell_id", "Node_Cell_ID", "nodeb_id_cell_id", "node_cell_id"])
    fallback = work[fallback_col] if fallback_col else pd.Series([""] * len(work), index=work.index)
    site_col = _pick_col(work, ["site", "site_id", "node_b_id", "nodeb_id", "nodeb_id_raw", "dashboard_site_id"])
    cell_col = _pick_col(work, ["cell_id", "local_cell_id"])
    sector_col = _pick_col(work, ["sector", "sector_id", "Sector"])
    band_col = _pick_col(work, ["band", "Band", "carrier", "serving_frequency_mhz", "frequency_mhz", "frequency"])

    sites = work[site_col] if site_col else pd.Series([None] * len(work), index=work.index)
    cells = work[cell_col] if cell_col else pd.Series([None] * len(work), index=work.index)
    sectors = work[sector_col] if sector_col else pd.Series([None] * len(work), index=work.index)
    bands = work[band_col] if band_col else pd.Series([None] * len(work), index=work.index)

    legacy = fallback.map(_rf_cell_id)
    if "legacy_nodeb_id_cell_id" not in work.columns:
        work["legacy_nodeb_id_cell_id"] = legacy
    else:
        work["legacy_nodeb_id_cell_id"] = _clean_identity_series(work["legacy_nodeb_id_cell_id"]).fillna(legacy)

    work["sector_identity_key"] = [
        build_sector_identity(site, cell, sector, fallback=old)
        for site, cell, sector, old in zip(sites, cells, sectors, legacy)
    ]
    work["site_sector_band_key"] = [
        build_site_sector_band_identity(site, sector, band)
        for site, sector, band in zip(sites, sectors, bands)
    ]
    work["rf_identity_key"] = [
        build_rf_identity(site, cell, sector, band, fallback=old)
        for site, cell, sector, band, old in zip(sites, cells, sectors, bands, legacy)
    ]
    if prefer_rf_node_cell_id:
        rf = _clean_identity_series(work["rf_identity_key"])
        old = _clean_identity_series(work.get("Node_Cell_ID", legacy))
        work["Node_Cell_ID"] = rf.fillna(old).fillna("")
    return work


def _print_fetch_summary(stage, table_name, filters, df, extra=None):
    print(f"[LTE_OPT][{stage}] source_table={table_name}")
    print(f"[LTE_OPT][{stage}] filters={filters}")
    print(f"[LTE_OPT][{stage}] row_count={len(df)}")
    print(f"[LTE_OPT][{stage}] columns={list(df.columns)}")
    if extra:
        for key, value in extra.items():
            print(f"[LTE_OPT][{stage}] {key}={value}")


def _sortable_series(series: pd.Series) -> pd.Series:
    def _sortable_value(value):
        if isinstance(value, (dict, list, tuple, set)):
            try:
                return json.dumps(value, sort_keys=True, default=str)
            except TypeError:
                return str(value)
        return value

    return series.map(_sortable_value)


def _sort_site_rows(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    if not sort_cols:
        return df
    work = df.copy()
    helper_cols = []
    for col in sort_cols:
        helper_col = f"__sort_{col}"
        work[helper_col] = _sortable_series(work[col])
        helper_cols.append(helper_col)
    return work.sort_values(helper_cols).drop(columns=helper_cols)


def _ensure_canonical_identity(df):
    work = df.copy()
    if "rf_identity_key" in work.columns:
        rf_identity = _clean_identity_series(work["rf_identity_key"])
        fallback_col = _pick_col(work, ["Node_Cell_ID", "nodeb_id_cell_id", "legacy_nodeb_id_cell_id", "node_cell_id", "cell_id"])
        fallback = _clean_identity_series(work[fallback_col]) if fallback_col else pd.Series(pd.NA, index=work.index, dtype="object")
        work["Node_Cell_ID"] = rf_identity.fillna(fallback).fillna("")
        work["canonical_cell_id"] = work["Node_Cell_ID"].map(canonical_cell_id)
        return work
    elif "Node_Cell_ID" in work.columns:
        work["Node_Cell_ID"] = work["Node_Cell_ID"].map(_rf_cell_id)
    elif "nodeb_id_cell_id" in work.columns:
        work["Node_Cell_ID"] = work["nodeb_id_cell_id"].map(_rf_cell_id)
    if any(col in work.columns for col in ["site", "cell_id", "sector", "band", "frequency_mhz"]):
        work = _attach_rf_identity_columns(work, prefer_rf_node_cell_id="rf_identity_key" not in df.columns)
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


def _shared_identity_column(*dataframes: pd.DataFrame) -> str:
    preferred = [
        "rf_identity_key",
        "site_sector_band_key",
        "sector_identity_key",
        "Node_Cell_ID",
        "legacy_nodeb_id_cell_id",
        "canonical_cell_id",
        "cell_id",
    ]
    for col in preferred:
        if all(isinstance(df, pd.DataFrame) and col in df.columns for df in dataframes):
            return col
    return "Node_Cell_ID"


_IDENTITY_ALIAS_COLS = [
    "Node_Cell_ID",
    "rf_identity_key",
    "sector_identity_key",
    "site_sector_band_key",
    "legacy_nodeb_id_cell_id",
    "frontend_site_sector_key",
    "nodeb_id_cell_id",
    "canonical_cell_id",
    "cell_id",
    "local_cell_id",
]


def _row_identity_aliases(row) -> set[str]:
    aliases = {_rf_cell_id(row.get(col)) for col in _IDENTITY_ALIAS_COLS}
    aliases.update(
        {
            _site_prefixed_cell_id(row.get("site"), row.get("cell_id")),
            _site_prefixed_cell_id(row.get("site"), row.get("local_cell_id")),
            _site_prefixed_cell_id(row.get("site"), row.get("sector")),
            _site_prefixed_cell_id(row.get("dashboard_site_id"), row.get("sector")),
            _site_prefixed_cell_id(row.get("site_id"), row.get("sector")),
        }
    )
    aliases.update({canonical_cell_id(alias) for alias in list(aliases) if alias})
    return {alias for alias in aliases if alias}


def _identity_specificity(value: object) -> int:
    text = _rf_cell_id(value)
    if not text:
        return 0
    return len([part for part in text.split("_") if part])


def _identity_without_operator_suffix(value: object) -> str:
    text = _rf_cell_id(value)
    parts = [part for part in text.split("_") if part]
    if len(parts) >= 4 and not parts[-1].isdigit():
        return "_".join(parts[:-1])
    return text


def _tilt_degrees(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.abs().le(20.0), values / 10.0)


def _operator_suffix_match(values: pd.Series, target: str) -> pd.Series:
    if _identity_specificity(target) < 3:
        return pd.Series(False, index=values.index)
    text = values.astype("string").fillna("")
    target_base = _identity_without_operator_suffix(target)
    value_base = text.map(_identity_without_operator_suffix)
    return text.eq(target) | text.eq(target_base) | value_base.eq(target) | value_base.eq(target_base)


def _identity_match_mask(df: pd.DataFrame, identity: object) -> pd.Series:
    target = _rf_cell_id(identity)
    if df.empty or not target:
        return pd.Series(False, index=df.index)
    target_is_rich = _identity_specificity(target) >= 3
    target_aliases = {target} if target_is_rich else {target, canonical_cell_id(target)}
    for col in [c for c in _IDENTITY_ALIAS_COLS if c in df.columns]:
        values = df[col].map(_rf_cell_id)
        mask = values.isin(target_aliases)
        if not bool(mask.any()) and target_is_rich:
            mask = _operator_suffix_match(values, target)
        if bool(mask.any()):
            return mask
        if target_is_rich:
            continue
        canonical_mask = values.map(canonical_cell_id).isin(target_aliases)
        if bool(canonical_mask.any()):
            return canonical_mask
    return df.apply(lambda row: bool(_row_identity_aliases(row) & target_aliases), axis=1)


def _normalize_site_df(site_df, log_stage="SITE_INPUT"):
    work = site_df.copy()
    rename_map = {
        "latitude": "lat",
        "longitude": "lon",
        "e_tilt": "electrical_tilt",
        "m_tilt": "mechanical_tilt",
        "height": "antenna_height",
        "Etilt": "electrical_tilt",
        "Mtilt": "mechanical_tilt",
        "Height": "antenna_height",
        "frequency": "frequency_mhz",
        "Frequency": "frequency_mhz",
        "maximum_transmission_power_of_resource": "tx_power",
    }
    # If the target name already exists as its own column, a plain rename would
    # produce two columns with the same name (work[col] then returns a
    # DataFrame instead of a Series). Coalesce into the existing column instead.
    for src, dst in list(rename_map.items()):
        if src not in work.columns:
            continue
        if dst in work.columns:
            work[dst] = work[dst].combine_first(work[src])
            work = work.drop(columns=[src])
            rename_map.pop(src)
    work = work.rename(columns=rename_map)

    source_nodeb_id = (
        work["nodeb_id"].map(_rf_cell_id)
        if "nodeb_id" in work.columns
        else pd.Series(pd.NA, index=work.index, dtype="object")
    )
    source_site_id = None
    for site_candidate in ["site", "site_id", "dashboard_site_id"]:
        if site_candidate in work.columns:
            source_site_id = work[site_candidate].map(_rf_cell_id)
            break
    if source_site_id is None:
        source_site_id = pd.Series(pd.NA, index=work.index, dtype="object")

    strict_identity_col = _pick_col(work, ["site_prediction_key", "site_cell_sector_band_operator_key"])
    if "cell_id" in work.columns:
        if strict_identity_col:
            work["Node_Cell_ID"] = work[strict_identity_col].map(_rf_cell_id)
        elif "Node_Cell_ID" in work.columns:
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
    if strict_identity_col and "cell_id" in work.columns:
        work["legacy_nodeb_id_cell_id"] = work["cell_id"].map(_rf_cell_id)
        work["rf_identity_key"] = work["Node_Cell_ID"].map(_rf_cell_id)
    else:
        work["legacy_nodeb_id_cell_id"] = work["Node_Cell_ID"].map(_rf_cell_id)
    work["canonical_cell_id"] = work["legacy_nodeb_id_cell_id"].map(canonical_cell_id)

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

    work = _attach_rf_identity_columns(work, prefer_rf_node_cell_id=True)
    derived_nodeb_id = work["Node_Cell_ID"].str.split("_").str[0]
    work["nodeb_id"] = source_nodeb_id.fillna(derived_nodeb_id).astype(str)
    work["node_b_id"] = work["nodeb_id"]
    work["canonical_cell_id"] = work["legacy_nodeb_id_cell_id"].map(canonical_cell_id)

    dashboard_site_id = source_site_id.fillna(derived_nodeb_id).astype(str)
    dashboard_site_id = dashboard_site_id.where(
        ~dashboard_site_id.isin(["", "nan", "None", "<NA>"]),
        derived_nodeb_id,
    )
    work["dashboard_site_id"] = dashboard_site_id.astype(str)
    if "site_id" not in work.columns:
        work["site_id"] = work["dashboard_site_id"]
    return work


def _fetch_latest_baseline_job_id(project_id, region="india", operator=None):
    bridge = get_bridge_client()
    if bridge:
        params = _bridge_region_params(project_id, region)
        if operator and str(operator).strip().lower() != "all":
            params["operator"] = str(operator).strip()
        payload = bridge._request("GET", "GetLatestLteBaselineJobId", params=params)
        job_id = payload.get("JobId") or payload.get("jobId")
        if not job_id:
            op_msg = f" operator={operator}" if operator else ""
            raise FileNotFoundError(
                f"No baseline results found for project_id={project_id} region={region}{op_msg}"
            )
        return str(job_id)
    current_engine = engine.get(region.lower(), engine["india"])
    filters = ["project_id = %(project_id)s"]
    params = {"project_id": int(project_id)}
    if operator and str(operator).strip().lower() != "all":
        filters.append("LOWER(TRIM(operator)) = %(operator)s")
        params["operator"] = str(operator).strip().lower()
    query = f"""
    SELECT job_id
    FROM lte_prediction_baseline_results
    WHERE {" AND ".join(filters)}
    ORDER BY created_at DESC
    LIMIT 1
    """
    rows = pd.read_sql(query, current_engine, params=params)
    if rows.empty:
        op_msg = f" operator={operator}" if operator else ""
        raise FileNotFoundError(
            f"No baseline results found for project_id={project_id} region={region}{op_msg}"
        )
    return str(rows.loc[0, "job_id"])


def fetch_baseline(project_id, region="india", operator=None, baseline_job_id=None):
    bridge = get_bridge_client()
    if bridge:
        if baseline_job_id is None:
            baseline_job_id = _fetch_latest_baseline_job_id(
                project_id,
                region=region,
                operator=operator,
            )
        params = _bridge_region_params(project_id, region)
        params["jobId"] = str(baseline_job_id)
        if operator:
            params["operator"] = operator
        page_limit = int(os.getenv("PYTHON_BRIDGE_BASELINE_PAGE_SIZE", "50000"))
        df = bridge.get_rows("GetLteBaselineRows", params, limit=page_limit)
        if df.empty:
            raise FileNotFoundError(f"No baseline results found for project_id={project_id}")
        if "job_id" in df.columns:
            df = df.loc[df["job_id"].astype(str) == str(baseline_job_id)].copy()
        if operator and "operator" in df.columns:
            df = df.loc[df["operator"].astype(str).str.strip().str.lower() == str(operator).strip().lower()].copy()
        if df.empty:
            raise FileNotFoundError(f"No baseline results found for project_id={project_id} job_id={baseline_job_id}")
        if "rf_identity_key" in df.columns:
            df["rf_identity_key"] = df["rf_identity_key"].map(_rf_cell_id)
            df["Node_Cell_ID"] = df["rf_identity_key"]
        elif "nodeb_id_cell_id" in df.columns:
            df["Node_Cell_ID"] = df["nodeb_id_cell_id"].map(_rf_cell_id)
        else:
            df["Node_Cell_ID"] = df["cell_id"].map(_rf_cell_id)
        if "canonical_cell_id" not in df.columns:
            if "rf_identity_key" in df.columns:
                df["canonical_cell_id"] = df["rf_identity_key"].map(canonical_cell_id)
            elif "nodeb_id_cell_id" in df.columns:
                df["canonical_cell_id"] = df["nodeb_id_cell_id"].map(canonical_cell_id)
            else:
                df["canonical_cell_id"] = df["Node_Cell_ID"].map(canonical_cell_id)
        for col in ["best_interferer_cell_id", "neighbor_1_cell_id", "neighbor_2_cell_id"]:
            if col in df.columns:
                df[col] = df[col].map(_rf_cell_id)
        df["rsrp"] = pd.to_numeric(df["pred_rsrp"], errors="coerce")
        _print_fetch_summary(
            "BASELINE_FETCH",
            "lte_prediction_baseline_results via python_bridge",
            {"project_id": project_id, "region": region, "operator": operator, "job_id": baseline_job_id},
            df,
        )
        return df
    current_engine = engine.get(region.lower(), engine["india"])
    base_cols = ["lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr", "cell_id", "nodeb_id_cell_id", "job_id"]
    optional_identity_cols = [
        "operator",
        "Technology",
        "technology",
        "serving_frequency_mhz",
        "serving_earfcn",
        "serving_pci",
        "rf_identity_key",
        "sector_identity_key",
        "site_sector_band_key",
        "legacy_nodeb_id_cell_id",
        "frontend_site_sector_key",
    ]
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
        baseline_job_id = _fetch_latest_baseline_job_id(
            project_id,
            region=region,
            operator=operator,
        )
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
    if "rf_identity_key" in df.columns:
        df["rf_identity_key"] = df["rf_identity_key"].map(_rf_cell_id)
        df["Node_Cell_ID"] = df["rf_identity_key"]
    elif "nodeb_id_cell_id" in df.columns:
        df["Node_Cell_ID"] = df["nodeb_id_cell_id"].map(_rf_cell_id)
    else:
        df["Node_Cell_ID"] = df["cell_id"].map(_rf_cell_id)
    if "canonical_cell_id" not in df.columns:
        if "rf_identity_key" in df.columns:
            df["canonical_cell_id"] = df["rf_identity_key"].map(canonical_cell_id)
        elif "nodeb_id_cell_id" in df.columns:
            df["canonical_cell_id"] = df["nodeb_id_cell_id"].map(canonical_cell_id)
        else:
            df["canonical_cell_id"] = df["Node_Cell_ID"].map(canonical_cell_id)
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
    bridge = get_bridge_client()
    if bridge:
        df = bridge.get_rows("GetLtePredictionGeoFeatures", _bridge_region_params(project_id, region), limit=50000)
        if "nodeb_id_cell_id" in df.columns:
            df["Node_Cell_ID"] = df["nodeb_id_cell_id"].map(_rf_cell_id)
        else:
            df["Node_Cell_ID"] = pd.Series(dtype=str)
        affected_cell_set = {_rf_cell_id(x) for x in (affected_cells or []) if _rf_cell_id(x)}
        if affected_cell_set:
            df = df.loc[df["Node_Cell_ID"].astype(str).isin(affected_cell_set)].copy()
        _print_fetch_summary(
            "GEO_FETCH",
            "lte_prediction_geo_features via python_bridge",
            {"project_id": project_id, "region": region, "affected_cell_count": len(affected_cells or [])},
            df,
        )
        return df
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
    allowed.update({canonical_cell_id(cell) for cell in list(allowed) if cell})
    if not allowed or df.empty:
        return df.iloc[0:0].copy() if allowed_cells is not None else df

    expanded = []
    for _, row in df.iterrows():
        aliases = {
            _rf_cell_id(row.get("Node_Cell_ID")),
            _rf_cell_id(row.get("rf_identity_key")),
            _rf_cell_id(row.get("sector_identity_key")),
            _rf_cell_id(row.get("site_sector_band_key")),
            _rf_cell_id(row.get("legacy_nodeb_id_cell_id")),
            _rf_cell_id(row.get("frontend_site_sector_key")),
            _rf_cell_id(row.get("cell_id")),
            _rf_cell_id(row.get("local_cell_id")),
            _site_prefixed_cell_id(row.get("site"), row.get("cell_id")),
            _site_prefixed_cell_id(row.get("site"), row.get("local_cell_id")),
            _site_prefixed_cell_id(row.get("site"), row.get("sector")),
            _site_prefixed_cell_id(row.get("dashboard_site_id"), row.get("sector")),
        }
        aliases.update({canonical_cell_id(alias) for alias in list(aliases) if alias})
        matched_aliases = sorted(alias for alias in aliases if alias and alias in allowed)
        for alias in matched_aliases:
            new_row = row.copy()
            new_row["Node_Cell_ID"] = alias
            new_row["canonical_cell_id"] = canonical_cell_id(alias)
            if not _rf_cell_id(new_row.get("nodeb_id")):
                new_row["nodeb_id"] = alias.split("_")[0] if "_" in alias else alias
            if not _rf_cell_id(new_row.get("node_b_id")):
                new_row["node_b_id"] = new_row["nodeb_id"]
            if not _rf_cell_id(new_row.get("dashboard_site_id")):
                new_row["dashboard_site_id"] = _rf_cell_id(new_row.get("site")) or new_row["nodeb_id"]
            if not _rf_cell_id(new_row.get("site_id")):
                new_row["site_id"] = new_row["dashboard_site_id"]
            if alias == _rf_cell_id(row.get("rf_identity_key")):
                new_row["rf_identity_key"] = alias
            expanded.append(new_row)

    if not expanded:
        return df.iloc[0:0].copy()
    out = pd.DataFrame(expanded).reset_index(drop=True)
    return out


def _parse_polygon_ids(polygon_ids=None):
    if polygon_ids is None:
        return []
    if isinstance(polygon_ids, (list, tuple, set)):
        raw_items = polygon_ids
    else:
        raw_items = str(polygon_ids).split(",")
    ids = []
    for item in raw_items:
        try:
            value = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if value > 0 and value not in ids:
            ids.append(value)
    return ids


def _site_polygon_filter_sql(table_name, polygon_ids):
    if not polygon_ids:
        return ""
    id_list = ",".join(str(int(value)) for value in polygon_ids)
    return f"""
    AND EXISTS (
        SELECT 1
        FROM map_regions mr_filter
        WHERE mr_filter.tbl_project_id = {table_name}.tbl_project_id
          AND mr_filter.id IN ({id_list})
          AND CASE
            WHEN ({table_name}.latitude) BETWEEN -90 AND 90
              AND ({table_name}.longitude) BETWEEN -180 AND 180
            THEN ST_Contains(
              mr_filter.region,
              ST_SRID(POINT({table_name}.longitude, {table_name}.latitude), 4326)
            )
            WHEN ({table_name}.longitude) BETWEEN -90 AND 90
              AND ({table_name}.latitude) BETWEEN -180 AND 180
            THEN ST_Contains(
              mr_filter.region,
              ST_SRID(POINT({table_name}.latitude, {table_name}.longitude), 4326)
            )
            ELSE 0
          END = 1
    )
    """


def _country_code_for_region(region):
    normalized = str(region or "").strip().lower()
    if normalized == "taiwan":
        return "TW"
    if normalized == "india":
        return "IN"
    return normalized.upper() if normalized else None


def _bridge_region_params(project_id, region):
    params = {"projectId": int(project_id), "region": str(region or "india").lower()}
    country_code = _country_code_for_region(region)
    if country_code:
        params["countryCode"] = country_code
    return params


def fetch_site_data(project_id, region="india", operator=None, allowed_cells=None, polygon_ids=None):
    polygon_id_list = _parse_polygon_ids(polygon_ids)
    bridge = get_bridge_client()
    if bridge:
        params = _bridge_region_params(project_id, region)
        if polygon_id_list:
            params["polygon_ids"] = ",".join(str(value) for value in polygon_id_list)
        start_time = time.perf_counter()
        raw_df = bridge.get_rows("GetLteSitePredictionRows", params, limit=50000)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(
            f"[LTE_OPT][PYTHON_BRIDGE_TIMING] endpoint=GetLteSitePredictionRows "
            f"project_id={project_id} region={region} rows={len(raw_df)} "
            f"elapsed_ms={elapsed_ms:.0f} polygon_ids={params.get('polygon_ids', 'all')}"
        )
        df = _normalize_site_df(raw_df, log_stage="SITE_FETCH")
        if operator:
            operator_norm = str(operator).strip().lower()
            if "cluster" in df.columns:
                before_rows = len(df)
                df = df.loc[df["cluster"].astype(str).str.strip().str.lower() == operator_norm].copy()
                print(f"[LTE_OPT][SITE_FETCH] operator_filter={operator} rows_before={before_rows} rows_after={len(df)}")
        if allowed_cells is not None:
            before_rows = len(df)
            before_cells = _safe_nunique(df, "Node_Cell_ID")
            unfiltered_df = df.copy()
            df = _expand_site_rows_to_allowed_cells(df, allowed_cells)
            if df.empty and not unfiltered_df.empty:
                print(
                    "[LTE_OPT][SITE_FETCH] baseline_population_filter_empty=True "
                    "fallback=unfiltered_site_rows_for_recommendation_matching"
                )
                df = unfiltered_df
            print(
                f"[LTE_OPT][SITE_FETCH] baseline_population_filter=True rows_before={before_rows} rows_after={len(df)} "
                f"cells_before={before_cells} cells_after={_safe_nunique(df, 'Node_Cell_ID')}"
            )
        _print_fetch_summary("SITE_FETCH", "site_prediction via python_bridge", {"project_id": project_id, "region": region, "operator": operator}, df)
        return df
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT
        site_prediction.*,
        cluster AS provider,
        cluster AS operator_name
    FROM site_prediction
    WHERE tbl_project_id = {project_id}
    {_site_polygon_filter_sql("site_prediction", polygon_id_list)}
    """

    raw_df = pd.read_sql(query, current_engine)
    raw_df = _filter_complete_site_prediction_identity(raw_df, endpoint="direct:site_prediction")
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
        unfiltered_df = df.copy()
        df = _expand_site_rows_to_allowed_cells(df, allowed_cells)
        if df.empty and not unfiltered_df.empty:
            print(
                "[LTE_OPT][SITE_FETCH] baseline_population_filter_empty=True "
                "fallback=unfiltered_site_rows_for_recommendation_matching"
            )
            df = unfiltered_df
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


def _resolve_latest_site_prediction_scenario(project_id, operator=None, region="india"):
    current_engine = engine.get(region.lower(), engine["india"])
    where = [f"tbl_project_id = {int(project_id)}", "scenario IS NOT NULL", "scenario > 0"]
    if operator:
        safe_operator = str(operator).replace("'", "''")
        where.append(f"LOWER(TRIM(cluster)) = LOWER(TRIM('{safe_operator}'))")
    query = f"""
    SELECT MAX(scenario) AS scenario
    FROM site_prediction_optimized
    WHERE {" AND ".join(where)}
    """
    value = pd.read_sql(query, current_engine).iloc[0]["scenario"]
    if pd.isna(value):
        return None
    return int(value)


def resolve_site_prediction_scenario_operator(project_id, scenario, region="india"):
    try:
        selected_scenario = int(scenario)
    except (TypeError, ValueError):
        return None
    if selected_scenario <= 0:
        return None

    bridge = get_bridge_client()
    if bridge:
        params = _bridge_region_params(project_id, region)
        params["scenario"] = selected_scenario
        opt_rows = bridge.get_rows("GetSitePredictionOptimized", params, limit=50000)
        source = "python_bridge"
    else:
        current_engine = engine.get(region.lower(), engine["india"])
        query = f"""
        SELECT cluster
        FROM site_prediction_optimized
        WHERE tbl_project_id = {int(project_id)}
          AND scenario = {int(selected_scenario)}
        """
        opt_rows = pd.read_sql(query, current_engine)
        source = "database"

    if opt_rows.empty or "cluster" not in opt_rows.columns:
        print(
            f"[LTE_OPT][SCENARIO_OPERATOR] project_id={project_id} "
            f"scenario={selected_scenario} source={source} operator=None reason=no_rows"
        )
        return None

    operators = (
        opt_rows["cluster"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    operators = operators[operators.str.lower().ne("")]
    operator_by_key = {}
    for value in operators.tolist():
        key = str(value).strip().lower()
        if not key:
            continue
        operator_by_key.setdefault(key, str(value).strip())
    unique_ops = [operator_by_key[key] for key in sorted(operator_by_key)]
    if len(unique_ops) == 1:
        resolved = unique_ops[0]
        print(
            f"[LTE_OPT][SCENARIO_OPERATOR] project_id={project_id} "
            f"scenario={selected_scenario} source={source} operator={resolved}"
        )
        return resolved
    if len(unique_ops) > 1:
        print(
            f"[LTE_OPT][SCENARIO_OPERATOR] project_id={project_id} "
            f"scenario={selected_scenario} source={source} operators={unique_ops} "
            f"reason=multiple_operators"
        )
    return None


def _overlay_optimized_site_rows(current_df: pd.DataFrame, opt_df: pd.DataFrame, compare_cols: list[str]) -> pd.DataFrame:
    merged_df = current_df.copy()
    for col in compare_cols:
        if col in merged_df.columns:
            merged_df[f"orig_{col}"] = pd.to_numeric(merged_df[col], errors="coerce")

    overlay_cols = list(dict.fromkeys(
        compare_cols
        + [
            "cell_id",
            "local_cell_id",
            "band",
            "frequency_mhz",
            "dashboard_site_id",
            "nodeb_id",
            "rf_identity_key",
            "sector_identity_key",
            "site_sector_band_key",
            "legacy_nodeb_id_cell_id",
        ]
    ))
    merged_df["optimization_applied"] = False
    match_keys = ["rf_identity_key", "site_sector_band_key", "sector_identity_key", "Node_Cell_ID", "legacy_nodeb_id_cell_id"]

    for key in match_keys:
        if key not in merged_df.columns or key not in opt_df.columns:
            continue
        opt_keyed = opt_df.copy()
        opt_keyed[key] = opt_keyed[key].map(_rf_cell_id)
        opt_keyed = opt_keyed.loc[opt_keyed[key].astype(str).str.strip().ne("")]
        if opt_keyed.empty:
            continue
        opt_keyed = opt_keyed.drop_duplicates(subset=[key], keep="last").set_index(key, drop=False)
        current_key = merged_df[key].map(_rf_cell_id)
        mask = (~merged_df["optimization_applied"]) & current_key.isin(opt_keyed.index.astype(str))
        if not mask.any():
            continue
        for col in overlay_cols:
            if col in opt_keyed.columns:
                mapping = opt_keyed[col].to_dict()
                merged_df.loc[mask, col] = current_key.loc[mask].map(mapping)
        merged_df.loc[mask, "optimization_applied"] = True
        merged_df.loc[mask, "optimization_match_key"] = key

    merged_df = _attach_rf_identity_columns(merged_df, prefer_rf_node_cell_id=True)
    if "optimization_match_key" not in merged_df.columns:
        merged_df["optimization_match_key"] = ""
    return merged_df


def fetch_optimized_sites(project_id, operator, region="india", polygon_ids=None, scenario=None):
    polygon_id_list = _parse_polygon_ids(polygon_ids)
    selected_scenario = None
    if scenario is not None:
        try:
            selected_scenario = int(scenario)
        except (TypeError, ValueError):
            selected_scenario = None
        if selected_scenario is not None and selected_scenario <= 0:
            selected_scenario = None
    bridge = get_bridge_client()
    if bridge:
        params = _bridge_region_params(project_id, region)
        params["operator"] = operator
        if selected_scenario is not None:
            params["scenario"] = selected_scenario
        if polygon_id_list:
            params["polygon_ids"] = ",".join(str(value) for value in polygon_id_list)
        start_time = time.perf_counter()
        opt_raw = bridge.get_rows(
            "GetSitePredictionOptimized",
            params,
            limit=50000,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(
            f"[LTE_OPT][PYTHON_BRIDGE_TIMING] endpoint=GetSitePredictionOptimized "
            f"project_id={project_id} region={region} operator={operator} rows={len(opt_raw)} "
            f"elapsed_ms={elapsed_ms:.0f} scenario={params.get('scenario', 'latest')} "
            f"polygon_ids={params.get('polygon_ids', 'all')}"
        )
        if opt_raw.empty:
            _print_fetch_summary(
                "OPTIMIZED_SITE_FETCH",
                "site_prediction_optimized via python_bridge",
                {"project_id": project_id, "operator": operator, "region": region, "scenario": params.get("scenario", "latest")},
                opt_raw,
            )
            return opt_raw
        current_df = fetch_site_data(project_id, region=region, operator=operator, polygon_ids=polygon_id_list)
        opt_df = _normalize_site_df(opt_raw, log_stage="OPTIMIZED_SITE_FETCH")
        sort_cols = [col for col in ["version", "updated_at", "created_at", "id", "Node_Cell_ID"] if col in opt_df.columns]
        if sort_cols:
            opt_df = _sort_site_rows(opt_df, sort_cols)
        compare_cols = [
            "lat",
            "lon",
            "azimuth",
            "electrical_tilt",
            "mechanical_tilt",
            "tx_power",
            "antenna_height",
            "frequency_mhz",
        ]
        merged_df = _overlay_optimized_site_rows(current_df, opt_df, compare_cols)
        mask = merged_df["optimization_applied"].fillna(False)
        changed_mask = _build_change_mask(merged_df)
        _print_fetch_summary(
            "OPTIMIZED_SITE_FETCH",
            "site_prediction_optimized via python_bridge",
            {"project_id": project_id, "operator": operator, "region": region, "scenario": params.get("scenario", "latest")},
            merged_df,
            extra={
                "optimized_rows": len(opt_df),
                "overlay_rows": int(mask.sum()),
                "changed_rows": int(changed_mask.sum()),
                "changed_cells": int(merged_df.loc[changed_mask, "Node_Cell_ID"].nunique()) if changed_mask.any() else 0,
                "distinct_pci": _safe_nunique(merged_df, "pci"),
                "distinct_cell_id": _safe_nunique(merged_df, "cell_id"),
                "distinct_nodeb_id": _safe_nunique(merged_df, "nodeb_id"),
            },
        )
        return merged_df
    if selected_scenario is None:
        selected_scenario = _resolve_latest_site_prediction_scenario(project_id, operator=operator, region=region)
    if selected_scenario is None:
        empty = pd.DataFrame()
        _print_fetch_summary(
            "OPTIMIZED_SITE_FETCH",
            "site_prediction_optimized",
            {"project_id": project_id, "operator": operator, "region": region, "scenario": "latest"},
            empty,
        )
        return empty
    current_engine = engine.get(region.lower(), engine["india"])
    operator_filter_sql = ""
    if operator and str(operator).strip().lower() not in {"all", "auto"}:
        safe_operator = str(operator).replace("'", "''")
        operator_filter_sql = f"AND LOWER(TRIM(cluster)) = LOWER(TRIM('{safe_operator}'))"
    query = f"""
    SELECT
        site_prediction_optimized.*,
        cluster AS provider,
        cluster AS operator_name
    FROM site_prediction_optimized
    WHERE tbl_project_id = {project_id}
    {operator_filter_sql}
    AND scenario = {int(selected_scenario)}
    {_site_polygon_filter_sql("site_prediction_optimized", polygon_id_list)}
    """

    opt_raw = pd.read_sql(query, current_engine)
    opt_raw = _filter_complete_site_prediction_identity(opt_raw, endpoint="direct:site_prediction_optimized")
    if opt_raw.empty:
        _print_fetch_summary(
            "OPTIMIZED_SITE_FETCH",
            "site_prediction_optimized",
            {"project_id": project_id, "operator": operator, "region": region, "scenario": selected_scenario},
            opt_raw,
        )
        return opt_raw

    current_df = fetch_site_data(project_id, region=region, operator=operator, polygon_ids=polygon_id_list)
    opt_df = _normalize_site_df(opt_raw, log_stage="OPTIMIZED_SITE_FETCH")
    sort_cols = [col for col in ["version", "updated_at", "created_at", "id", "Node_Cell_ID"] if col in opt_df.columns]
    if sort_cols:
        opt_df = _sort_site_rows(opt_df, sort_cols)

    compare_cols = [
        "lat",
        "lon",
        "azimuth",
        "electrical_tilt",
        "mechanical_tilt",
        "tx_power",
        "antenna_height",
        "frequency_mhz",
    ]
    merged_df = _overlay_optimized_site_rows(current_df, opt_df, compare_cols)
    mask = merged_df["optimization_applied"].fillna(False)
    changed_mask = _build_change_mask(merged_df)
    _print_fetch_summary(
        "OPTIMIZED_SITE_FETCH",
        "site_prediction_optimized",
        {"project_id": project_id, "operator": operator, "region": region, "scenario": selected_scenario},
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
    baseline_work = _ensure_canonical_identity(baseline_df)
    site_work = _ensure_canonical_identity(site_df)
    identity_col = _shared_identity_column(baseline_work, site_work)
    total_cells = int(site_work[identity_col].nunique())
    print(f"[LTE_OPT][K1K2] total_site_cells={total_cells}")

    for cid in site_work[identity_col].astype(str).unique():
        site_rows = site_work[site_work[identity_col].astype(str) == str(cid)].copy()
        dt_rows = baseline_work[baseline_work[identity_col].astype(str) == str(cid)].copy()
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
        "frequency_mhz",
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
    identity_col = _shared_identity_column(baseline_work, site_work)
    candidate_cols = [
        "site_sector_band_key",
        "sector_identity_key",
        "legacy_nodeb_id_cell_id",
        "canonical_cell_id",
        identity_col,
        "Node_Cell_ID",
    ]
    baseline_identity_cache = {}
    baseline_canonical_cache = {}
    for candidate_col in dict.fromkeys(candidate_cols):
        if candidate_col not in baseline_work.columns:
            continue
        values = baseline_work[candidate_col].map(_rf_cell_id)
        baseline_identity_cache[candidate_col] = values
        baseline_canonical_cache[candidate_col] = values.map(canonical_cell_id)
    for cid in sorted({str(x) for x in target_cells}):
        site_rows = site_work.loc[_identity_match_mask(site_work, cid)].copy()
        dt_rows = pd.DataFrame()
        match_col = identity_col
        site_aliases = set()
        for _, site_row in site_rows.iterrows():
            site_aliases.update(_row_identity_aliases(site_row))
        site_aliases.update({_rf_cell_id(cid), canonical_cell_id(cid)})
        for candidate_col in candidate_cols:
            if site_rows.empty or candidate_col not in baseline_identity_cache:
                continue
            candidate_values = set(site_aliases)
            if candidate_col in site_rows.columns:
                candidate_values.update(
                    _rf_cell_id(value)
                    for value in site_rows[candidate_col].dropna().astype(str).tolist()
                    if _rf_cell_id(value)
                )
            if not candidate_values:
                continue
            baseline_values = baseline_identity_cache[candidate_col]
            baseline_canonical_values = baseline_canonical_cache[candidate_col]
            candidate_dt = baseline_work.loc[
                baseline_values.isin(candidate_values)
                | baseline_canonical_values.isin(candidate_values)
            ].copy()
            if not candidate_dt.empty:
                dt_rows = candidate_dt
                match_col = candidate_col
                break
        if dt_rows.empty:
            dt_rows = baseline_work.loc[_identity_match_mask(baseline_work, cid)].copy()
        print(
            f"[LTE_OPT][K1K2_LOCAL] cell={cid} site_rows={len(site_rows)} "
            f"baseline_rows={len(dt_rows)} match_col={match_col}"
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
    identity_col = _shared_identity_column(site_work, baseline_df if isinstance(baseline_df, pd.DataFrame) else pd.DataFrame())
    site_work["_scope_technology"] = _technology_series(site_work)
    changed_mask = _build_change_mask(site_work)
    changed_rows = site_work.loc[changed_mask].copy()
    if changed_rows.empty:
        raise ValueError("No effective optimized site change detected")

    changed_cell_ids = sorted(changed_rows[identity_col].astype(str).unique().tolist())
    changed_technologies = sorted(changed_rows["_scope_technology"].astype(str).unique().tolist())
    changed_site_technologies = (
        changed_rows.groupby(changed_rows["dashboard_site_id"].astype(str))["_scope_technology"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    cell_technology = {}
    for _, row in site_work.iterrows():
        tech = str(row.get("_scope_technology", "")).strip()
        if not tech:
            continue
        aliases = set(_row_identity_aliases(row))
        aliases.add(_rf_cell_id(row.get(identity_col)))
        for alias in aliases:
            if alias:
                cell_technology.setdefault(alias, tech)

    def _cell_matches_technology(cell_id, allowed_technologies):
        allowed = {str(value) for value in allowed_technologies if str(value).strip()}
        if not allowed:
            return True
        cell = _clean_cell_id(cell_id)
        tech = cell_technology.get(cell) or cell_technology.get(canonical_cell_id(cell))
        return tech in allowed if tech else False

    selected_cells = set(changed_cell_ids)

    same_site_mask = [
        str(row.get("_scope_technology", "")) in changed_site_technologies.get(str(row.get("dashboard_site_id", "")), set())
        for _, row in site_work.iterrows()
    ]
    same_site_rows = site_work.loc[same_site_mask].copy()
    selected_cells.update(same_site_rows[identity_col].astype(str).tolist())

    available_cells = set(site_work[identity_col].astype(str).tolist())

    # Count changed cells whose location actually moved. A moved cell is searched
    # around two centers (new + original position), so its neighbour budget must
    # be doubled — otherwise the closer new-location cells consume the whole cap
    # and the old-location neighbours are silently dropped.
    moved_cell_count = 0
    if {"orig_lat", "orig_lon"}.issubset(changed_rows.columns):
        new_lat_all = pd.to_numeric(changed_rows["lat"], errors="coerce")
        new_lon_all = pd.to_numeric(changed_rows["lon"], errors="coerce")
        orig_lat_all = pd.to_numeric(changed_rows["orig_lat"], errors="coerce")
        orig_lon_all = pd.to_numeric(changed_rows["orig_lon"], errors="coerce")
        moved_mask = (
            orig_lat_all.notna()
            & orig_lon_all.notna()
            & new_lat_all.notna()
            & new_lon_all.notna()
            & (
                ~np.isclose(orig_lat_all.fillna(0.0), new_lat_all.fillna(0.0))
                | ~np.isclose(orig_lon_all.fillna(0.0), new_lon_all.fillna(0.0))
            )
        )
        moved_cell_count = int(
            changed_rows.loc[moved_mask, identity_col].astype(str).nunique()
        )

    search_center_count = len(changed_cell_ids) + moved_cell_count

    max_neighbors = max_neighbors_per_update_cell
    if max_neighbors is None:
        max_neighbors = neighbor_site_count
    max_neighbors = max(0, int(max_neighbors or 0)) * max(1, search_center_count)

    neighbor_counts = {}
    topology_cols = []
    if isinstance(baseline_df, pd.DataFrame) and not baseline_df.empty:
        baseline_work = _ensure_canonical_identity(baseline_df)
        topology_cols = [
            col for col in ["best_interferer_cell_id", "neighbor_1_cell_id", "neighbor_2_cell_id"]
            if col in baseline_work.columns
        ]
        if topology_cols and identity_col in baseline_work.columns:
            baseline_work["_scope_technology"] = _technology_series(baseline_work)
            focus = baseline_work.loc[baseline_work[identity_col].astype(str).isin(changed_cell_ids)]
            for _, focus_row in focus.iterrows():
                source_tech = str(focus_row.get("_scope_technology", "")).strip()
                allowed_for_row = {source_tech} if source_tech else set(changed_technologies)
                for col in topology_cols:
                    value = focus_row.get(col)
                    cell = _clean_cell_id(value)
                    if cell and cell not in selected_cells and _cell_matches_technology(cell, allowed_for_row):
                        neighbor_counts[cell] = neighbor_counts.get(cell, 0) + 1

    selected_neighbor_cells = [
        cell for cell, _ in sorted(neighbor_counts.items(), key=lambda item: (-item[1], item[0]))
        if cell in available_cells and _cell_matches_technology(cell, changed_technologies)
    ][:max_neighbors]
    selected_cells.update(selected_neighbor_cells)

    if not topology_cols or not selected_neighbor_cells:
        candidate_rows = site_work.loc[
            ~site_work[identity_col].astype(str).isin(selected_cells)
            & site_work["_scope_technology"].astype(str).isin(changed_technologies)
        ].copy()
        candidate_rows["lat"] = pd.to_numeric(candidate_rows["lat"], errors="coerce")
        candidate_rows["lon"] = pd.to_numeric(candidate_rows["lon"], errors="coerce")
        ranked_parts = []
        for _, row in changed_rows.iterrows():
            row_tech = str(row.get("_scope_technology", "")).strip()
            tech_candidate_rows = candidate_rows.loc[candidate_rows["_scope_technology"].astype(str).eq(row_tech)].copy()
            if tech_candidate_rows.empty:
                continue
            # Search neighbours around BOTH the new location and the original
            # location. When a site is moved (or removed), the cells surrounding
            # its *old* position also lose its coverage/interference and must be
            # recomputed, not only the cells around the new position.
            new_lat = pd.to_numeric(pd.Series([row.get("lat")]), errors="coerce").iloc[0]
            new_lon = pd.to_numeric(pd.Series([row.get("lon")]), errors="coerce").iloc[0]
            orig_lat = pd.to_numeric(pd.Series([row.get("orig_lat")]), errors="coerce").iloc[0]
            orig_lon = pd.to_numeric(pd.Series([row.get("orig_lon")]), errors="coerce").iloc[0]

            centers = []
            if not (pd.isna(new_lat) or pd.isna(new_lon)):
                centers.append((float(new_lat), float(new_lon)))
            # Only add the original location as a separate center when it is
            # valid and the site actually moved (otherwise it duplicates the new
            # center and adds nothing).
            if not (pd.isna(orig_lat) or pd.isna(orig_lon)) and not (
                (not pd.isna(new_lat) and not pd.isna(new_lon))
                and np.isclose(orig_lat, new_lat)
                and np.isclose(orig_lon, new_lon)
            ):
                centers.append((float(orig_lat), float(orig_lon)))

            for center_lat, center_lon in centers:
                ranked = tech_candidate_rows.copy()
                ranked["distance_m"] = haversine_vectorized(
                    center_lat,
                    center_lon,
                    ranked["lat"].to_numpy(dtype=float, copy=False),
                    ranked["lon"].to_numpy(dtype=float, copy=False),
                )
                ranked = ranked.loc[ranked["distance_m"] <= float(impact_radius_m)].copy()
                if ranked.empty:
                    continue
                ranked = ranked.sort_values(["distance_m", identity_col], ascending=[True, True])
                ranked_parts.append(ranked)
        if ranked_parts and max_neighbors > 0:
            ranked_cells = (
                pd.concat(ranked_parts, ignore_index=True)
                .sort_values(["distance_m", identity_col], ascending=[True, True])
                .drop_duplicates(subset=[identity_col], keep="first")
                .head(max_neighbors)
            )
            distance_neighbors = ranked_cells[identity_col].astype(str).tolist()
            selected_cells.update(distance_neighbors)
        else:
            distance_neighbors = []
    else:
        distance_neighbors = []

    affected_rows = site_work.loc[site_work[identity_col].astype(str).isin(selected_cells)].copy()
    affected_ids = sorted(affected_rows[identity_col].astype(str).unique().tolist())
    affected_site_ids = sorted(affected_rows["dashboard_site_id"].astype(str).unique().tolist())
    print(
        f"[LTE_OPT][AFFECTED_CELL_SCOPE] changed_cells={len(changed_cell_ids)} "
        f"same_site_cells={same_site_rows[identity_col].nunique()} "
        f"topology_neighbor_cells={len(selected_neighbor_cells)} "
        f"distance_neighbor_cells={len(distance_neighbors)} affected_cells={len(affected_ids)} "
        f"technologies={changed_technologies} "
        f"scope=changed_cells_plus_same_site_cells_plus_neighbor_cells_technology_filtered"
    )
    return affected_ids, affected_site_ids, changed_rows.drop(columns=["_scope_technology"], errors="ignore")


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


def _first_clean_value(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str).str.strip()
        series = series[~series.str.lower().isin(["", "nan", "none", "null", "<na>"])]
        if not series.empty:
            return _rf_cell_id(series.iloc[0])
    return ""


def _attach_serving_identity_to_points(pts: pd.DataFrame, site_rows: pd.DataFrame, node_cell_id: str) -> pd.DataFrame:
    if pts.empty:
        return pts
    out = pts.copy()
    node_b_id = _first_clean_value(site_rows, ["node_b_id", "nodeb_id"])
    site_id = _first_clean_value(site_rows, ["site_id", "dashboard_site_id", "site"])
    local_cell_id = _first_clean_value(site_rows, ["cell_id", "local_cell_id"])
    band = _first_clean_value(site_rows, ["band", "Band", "frequency_band", "carrier"])
    operator = _first_clean_value(site_rows, ["operator", "Operator", "cluster", "network"])
    technology = _first_clean_value(site_rows, ["Technology", "technology"])

    out["Node_Cell_ID"] = str(node_cell_id)
    out["nodeb_id_cell_id"] = str(node_cell_id)
    if node_b_id:
        out["node_b_id"] = node_b_id
    if site_id:
        out["site_id"] = site_id
    if local_cell_id:
        out["cell_id"] = local_cell_id
    if band:
        out["band"] = band
    if operator:
        out["operator"] = operator
    if technology:
        out["Technology"] = technology
    out["canonical_cell_id"] = canonical_cell_id(str(node_cell_id))
    return out


def _restore_original_site_state(site_df: pd.DataFrame) -> pd.DataFrame:
    out = site_df.copy()
    for col in [
        "lat",
        "lon",
        "azimuth",
        "electrical_tilt",
        "mechanical_tilt",
        "tx_power",
        "antenna_height",
        "frequency_mhz",
    ]:
        orig_col = f"orig_{col}"
        if orig_col in out.columns and col in out.columns:
            before = pd.to_numeric(out[orig_col], errors="coerce")
            out[col] = before.where(before.notna(), out[col])
    return out


def _offset_ready_site_df(site_df: pd.DataFrame, region: str) -> pd.DataFrame:
    out = site_df.copy()
    rename_pairs = {
        "electrical_tilt": "Etilt",
        "mechanical_tilt": "Mtilt",
        "antenna_height": "Height",
        "technology": "Technology",
        "operator": "operator",
    }
    for src, dst in rename_pairs.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
        elif src in out.columns and dst in out.columns:
            out[dst] = out[dst].combine_first(out[src])
    if "frequency_mhz" not in out.columns and "serving_frequency_mhz" in out.columns:
        out["frequency_mhz"] = out["serving_frequency_mhz"]
    if "Node_Cell_ID" not in out.columns and "rf_identity_key" in out.columns:
        out["Node_Cell_ID"] = out["rf_identity_key"]
    if "Etilt" in out.columns:
        out["Etilt"] = _tilt_degrees(out["Etilt"])
    if "Mtilt" in out.columns:
        out["Mtilt"] = _tilt_degrees(out["Mtilt"])
    return _offset_prepare_site_rows(out, region)


def _baseline_points_for_cells(baseline_df: pd.DataFrame, cell_ids) -> pd.DataFrame:
    baseline_work = _ensure_canonical_identity(baseline_df)
    frames = []
    for cid in cell_ids:
        rows = baseline_work.loc[_identity_match_mask(baseline_work, cid)].copy()
        if rows.empty:
            continue
        rows["target_node_cell_id"] = str(cid)
        frames.append(rows)
    if not frames:
        return pd.DataFrame()
    pts = pd.concat(frames, ignore_index=True)
    pts["lat"] = pd.to_numeric(pts.get("lat"), errors="coerce")
    pts["lon"] = pd.to_numeric(pts.get("lon"), errors="coerce")
    pts = pts.dropna(subset=["lat", "lon"]).copy()
    if pts.empty:
        return pts
    if "grid_id" not in pts.columns:
        pts["grid_id"] = pd.NA
    clean_grid = pts["grid_id"].astype("string").str.strip()
    missing_grid = clean_grid.isna() | clean_grid.eq("") | clean_grid.str.lower().isin(["nan", "none", "null", "<na>"])
    pts.loc[missing_grid, "grid_id"] = [
        f"OPT_{cell}_{idx}"
        for idx, cell in zip(pts.index[missing_grid], pts.loc[missing_grid, "target_node_cell_id"].astype(str))
    ]
    pts["center_lat"] = pts["lat"]
    pts["center_lon"] = pts["lon"]
    pts["baseline_pred_rsrp"] = pd.to_numeric(pts.get("pred_rsrp"), errors="coerce")
    pts["baseline_pred_rsrq"] = pd.to_numeric(pts.get("pred_rsrq"), errors="coerce")
    pts["baseline_pred_sinr"] = pd.to_numeric(pts.get("pred_sinr"), errors="coerce")
    keep = ["grid_id", "center_lat", "center_lon", "target_node_cell_id", "baseline_pred_rsrp", "baseline_pred_rsrq", "baseline_pred_sinr"]
    for col in ["clutter_class", "Technology", "operator", "cell_id", "node_b_id", "nodeb_id", "site_id", "nodeb_id_cell_id"]:
        if col in pts.columns:
            keep.append(col)
    return pts[keep].drop_duplicates(subset=["grid_id", "target_node_cell_id"]).reset_index(drop=True)


def _generated_points_for_cell(site_rows: pd.DataFrame, cid: str, params: dict) -> pd.DataFrame:
    pts = generate_grid(site_rows, params.get("radius", 500), params.get("grid_resolution", 10))
    if pts.empty:
        return pts
    pts = pts.copy()
    pts["grid_id"] = [f"GEN_{cid}_{idx}" for idx in range(len(pts))]
    pts["center_lat"] = pd.to_numeric(pts["lat"], errors="coerce")
    pts["center_lon"] = pd.to_numeric(pts["lon"], errors="coerce")
    pts["target_node_cell_id"] = str(cid)
    pts["baseline_pred_rsrp"] = np.nan
    pts["baseline_pred_rsrq"] = np.nan
    pts["baseline_pred_sinr"] = np.nan
    return pts[["grid_id", "center_lat", "center_lon", "target_node_cell_id", "baseline_pred_rsrp", "baseline_pred_rsrq", "baseline_pred_sinr"]]


def _first_numeric_value(df: pd.DataFrame, col: str):
    if col not in df.columns or df.empty:
        return np.nan
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return np.nan
    return float(series.iloc[0])


def _location_change_summary(site_rows: pd.DataFrame) -> tuple[bool, float, float, float, float, float]:
    new_lat = _first_numeric_value(site_rows, "lat")
    new_lon = _first_numeric_value(site_rows, "lon")
    old_lat = _first_numeric_value(site_rows, "orig_lat")
    old_lon = _first_numeric_value(site_rows, "orig_lon")
    if not all(np.isfinite(value) for value in [new_lat, new_lon, old_lat, old_lon]):
        return False, 0.0, old_lat, old_lon, new_lat, new_lon
    moved_m = float(haversine_vectorized(old_lat, old_lon, new_lat, new_lon))
    return moved_m > 1.0, moved_m, old_lat, old_lon, new_lat, new_lon


def _local_project_dem_candidate(project_id, region) -> str | None:
    if int(project_id or 0) != 210 and str(region).lower() != "taiwan":
        return None
    mapdata_root = ML_ROOT / "tests" / "new-project" / "data" / "mapdata"
    if not mapdata_root.exists():
        return None
    for path in sorted(mapdata_root.rglob("height_5m.grd")):
        if path.is_file():
            return str(path)
    return None


def _manual_physical_context(params: dict) -> tuple[pd.DataFrame, str | None, dict]:
    building_df = params.get("building_df")
    if not isinstance(building_df, pd.DataFrame):
        building_df = pd.DataFrame()
    if building_df.empty and bool(params.get("building", True)) and params.get("project_id"):
        try:
            from ..lte_prediction.ml_engine import fetch_building_data as _fetch_building_data

            building_df = _fetch_building_data(
                int(params.get("project_id")),
                region=str(params.get("region", "india")).lower(),
            )
            print(
                f"[LTE_OPT][PHASE26_CONTEXT] building_rows={len(building_df)} source=project_buildings",
                flush=True,
            )
        except Exception as exc:
            print(f"[LTE_OPT][PHASE26_CONTEXT] building_rows=0 reason={exc}", flush=True)
            building_df = pd.DataFrame()

    dem_raster_path = params.get("dem_raster_path") or params.get("demRasterPath")
    if not dem_raster_path:
        dem_raster_path = _local_project_dem_candidate(
            params.get("project_id"),
            str(params.get("region", "india")).lower(),
        )
    if dem_raster_path:
        print(f"[LTE_OPT][PHASE26_CONTEXT] dem_raster_path={dem_raster_path}", flush=True)

    clutter_by_grid = params.get("clutter_by_grid")
    if not isinstance(clutter_by_grid, dict):
        clutter_by_grid = {}
    return building_df, dem_raster_path, clutter_by_grid


# Columns produced by phase27_physical.score_candidates (Phase 26 physical scoring).
_PHASE26_OUTPUT_COLS = [
    "building_obstruction_loss_db",
    "terrain_diffraction_loss_db",
    "terrain_fresnel_excess_m",
    "terrain_peak_clearance_m",
    "terrain_decision",
    "obstruction_branch",
    "clutter_class",
]


def _phase36_surface_for_points(
    site_df: pd.DataFrame,
    points_df: pd.DataFrame,
    project_id=None,
    region: str = "india",
    building_df: pd.DataFrame | None = None,
    dem_raster_path: str | None = None,
    clutter_by_grid: dict | None = None,
    phase26_cache: dict | None = None,
) -> pd.DataFrame:
    if site_df.empty or points_df.empty:
        return pd.DataFrame()
    grid = points_df[["grid_id", "center_lat", "center_lon"]].drop_duplicates("grid_id").copy()
    grid_lat = pd.to_numeric(grid["center_lat"], errors="coerce").to_numpy(float)
    grid_lon = pd.to_numeric(grid["center_lon"], errors="coerce").to_numpy(float)
    frames = []
    for _, row in site_df.iterrows():
        distance = _offset_haversine_m(float(row["lat"]), float(row["lon"]), grid_lat, grid_lon)
        raw = _offset_cost231_for_points(_offset_site_record(row), grid_lat, grid_lon, float(row["frequency_mhz"]))
        raw = raw + float(row.get("model_rsrp_adjust_db", 0.0))
        bearing = _offset_bearing_deg(float(row["lat"]), float(row["lon"]), grid_lat, grid_lon)
        az_delta = _offset_azimuth_delta_deg(bearing, float(row["azimuth"]))
        frames.append(pd.DataFrame({
            "grid_id": grid["grid_id"].astype(str).to_numpy(),
            "lat": grid_lat,
            "lon": grid_lon,
            "center_lat": grid_lat,
            "center_lon": grid_lon,
            "Node_Cell_ID": str(row["strict_cell_key"]),
            "node_cell_id": str(row["strict_cell_key"]),
            "strict_cell_key": str(row["strict_cell_key"]),
            "site": str(row["site_key"]),
            "nodeb_id": str(row.get("nodeb_id", row["site_key"])),
            "cell_id": str(row["original_cell_id"]),
            "sector": str(row["sector_key"]),
            "band": str(row["band_key"]),
            "Technology": str(row["technology_key"]),
            "technology": str(row["technology_key"]),
            "operator": str(row["operator_key"]),
            "rf_identity_key": str(row["strict_cell_key"]),
            "sector_identity_key": str(row["sector_identity_key"]),
            "site_sector_band_key": str(row["site_sector_band_key"]),
            "legacy_nodeb_id_cell_id": str(row["original_cell_id"]),
            "serving_frequency_mhz": float(row["frequency_mhz"]),
            "original_frequency_mhz": float(row.get("original_frequency_mhz", row["frequency_mhz"])),
            "model_rsrp_adjust_db": float(row.get("model_rsrp_adjust_db", 0.0)),
            "distance_m": np.asarray(distance, dtype=float),
            "bearing_deg": np.asarray(bearing, dtype=float),
            "azimuth_delta_deg": np.asarray(az_delta, dtype=float),
            "Height": float(row.get("Height", row.get("antenna_height", 30.0))),
            "Etilt": float(row.get("Etilt", row.get("electrical_tilt", 3.0))),
            "Mtilt": float(row.get("Mtilt", row.get("mechanical_tilt", 0.0))),
            "antenna_model": str(row.get("antenna_model", "")),
            "raw_cost231_rsrp": np.asarray(raw, dtype=float),
            "building_obstruction_loss_db": 0.0,
            "terrain_diffraction_loss_db": 0.0,
            "terrain_decision": "baseline_delta_preserved",
            "obstruction_branch": "clear",
            "clutter_class": "Open",
            "physical_rsrp_unclipped": np.asarray(raw, dtype=float),
        }))
    surface = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if surface.empty:
        return surface
    if not isinstance(clutter_by_grid, dict):
        clutter_by_grid = {}
    if "clutter_class" in points_df.columns:
        point_clutter = (
            points_df[["grid_id", "clutter_class"]]
            .dropna(subset=["grid_id"])
            .drop_duplicates("grid_id")
            .set_index("grid_id")["clutter_class"]
            .astype(str)
            .to_dict()
        )
        clutter_by_grid = {**point_clutter, **clutter_by_grid}
    # Phase 26 physical scoring (building obstruction + terrain diffraction +
    # clutter) depends ONLY on tx position/height, rx position and frequency --
    # phase27_physical.score_candidates takes (site.lat, site.lon, site.Height,
    # row.lat, row.lon, row.serving_frequency_mhz) and never reads tilt, azimuth
    # or the antenna pattern. All of those are fixed per (strict_cell_key,
    # grid_id) for the whole job, so trialling a tilt/azimuth delta cannot change
    # the result. Cache on that key and reuse across a cell's candidates instead
    # of re-running the per-row building/DEM intersection every candidate, which
    # profiled at ~279s of a 296s RF evaluation.
    cache = phase26_cache if isinstance(phase26_cache, dict) else None
    cache_keys = None
    if cache is not None:
        site_physical_key = {}
        for _, site_row in site_df.iterrows():
            strict_key = str(site_row.get("strict_cell_key", "")).strip()
            if not strict_key:
                continue
            site_physical_key[strict_key] = (
                round(float(site_row.get("lat", np.nan)), 7),
                round(float(site_row.get("lon", np.nan)), 7),
                round(float(site_row.get("Height", site_row.get("antenna_height", 30.0)) or 30.0), 2),
                round(float(site_row.get("frequency_mhz", np.nan)), 3),
            )

        cache_keys = []
        for surf_row in surface[["strict_cell_key", "grid_id", "center_lat", "center_lon"]].itertuples(index=False):
            grid_id = str(surf_row.grid_id)
            cache_keys.append((
                site_physical_key.get(str(surf_row.strict_cell_key), (str(surf_row.strict_cell_key),)),
                round(float(surf_row.center_lat), 7),
                round(float(surf_row.center_lon), 7),
                str(clutter_by_grid.get(grid_id, "")),
            ))
    if cache is not None and cache_keys and all(k in cache for k in cache_keys):
        cached_rows = [cache[k] for k in cache_keys]
        for pos, col in enumerate(_PHASE26_OUTPUT_COLS):
            surface[col] = [row[pos] for row in cached_rows]
        print(
            f"[LTE_OPT][PHASE26_PHYSICAL] rows={len(surface)} source=cache_hit",
            flush=True,
        )
    else:
        try:
            surface = _offset_score_candidates(
                surface,
                site_df,
                building_df if isinstance(building_df, pd.DataFrame) else pd.DataFrame(),
                int(project_id) if project_id is not None else 0,
                str(region).lower(),
                dem_raster_path=dem_raster_path,
                clutter_by_grid=clutter_by_grid,
                allow_auto_dem=False,
            )
            print(
                "[LTE_OPT][PHASE26_PHYSICAL] "
                f"rows={len(surface)} building_nonzero="
                f"{int((pd.to_numeric(surface.get('building_obstruction_loss_db'), errors='coerce').fillna(0.0) != 0.0).sum())} "
                f"terrain_nonzero={int((pd.to_numeric(surface.get('terrain_diffraction_loss_db'), errors='coerce').fillna(0.0) > 0.0).sum())} "
                f"source=computed",
                flush=True,
            )
            if cache is not None and all(col in surface.columns for col in _PHASE26_OUTPUT_COLS):
                stored = surface[_PHASE26_OUTPUT_COLS].to_numpy(dtype=object)
                for key, values in zip(cache_keys or [], stored):
                    cache[key] = tuple(values)
        except Exception as exc:
            print(f"[LTE_OPT][PHASE26_PHYSICAL] disabled reason={exc}", flush=True)
    surface = _offset_add_features(surface, "strict_cell_key")
    return _offset_apply_phase36(surface, "physical_rsrp_unclipped", g5_level_anchor_db=0.0)


def _carrier_key_for_quality(frame: pd.DataFrame) -> pd.Series:
    freq = pd.to_numeric(
        frame.get("original_frequency_mhz", frame.get("serving_frequency_mhz", frame.get("frequency_mhz"))),
        errors="coerce",
    ).round(1)
    return frame["technology"].astype(str) + "|" + freq.astype("string")


def _mw_from_dbm(values):
    return np.power(10.0, np.asarray(values, dtype=float) / 10.0)


def _row_matches_identity(row, identity_value) -> bool:
    target = _rf_cell_id(identity_value)
    if not target:
        return False
    if _identity_specificity(target) >= 3:
        target_base = _identity_without_operator_suffix(target)
        target_aliases = {target, target_base}
        for col in [
            "Node_Cell_ID",
            "node_cell_id",
            "strict_cell_key",
            "rf_identity_key",
            "nodeb_id_cell_id",
        ]:
            value = _rf_cell_id(row.get(col))
            if not value:
                continue
            value_base = _identity_without_operator_suffix(value)
            if value in target_aliases or value_base in target_aliases:
                return True
        return False
    return bool(_row_identity_aliases(row) & {target, canonical_cell_id(target)})


def _quality_from_signal(signal_dbm, interference_mw, technology):
    if not np.isfinite(signal_dbm):
        return np.nan, np.nan
    s = float(_mw_from_dbm(signal_dbm))
    noise = float(_mw_from_dbm(-104.0))
    rssi = s + max(float(interference_mw), 0.0) + noise
    sinr = 10.0 * np.log10(s / (max(float(interference_mw), 0.0) + noise))
    rsrq = 10.0 * np.log10(_offset_n_rb_for(str(technology), None)) + float(signal_dbm) - 10.0 * np.log10(rssi)
    return sinr, rsrq


def _apply_baseline_rsrp_residual(surface: pd.DataFrame, points_df: pd.DataFrame, residual_source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    joined = residual_source.merge(
        points_df[["grid_id", "target_node_cell_id", "baseline_pred_rsrp"]],
        on="grid_id",
        how="inner",
    )
    target_mask = joined.apply(lambda row: _row_matches_identity(row, row.get("target_node_cell_id")), axis=1)
    target = joined.loc[target_mask].copy()
    target["rsrp_residual_db"] = pd.to_numeric(target["baseline_pred_rsrp"], errors="coerce") - pd.to_numeric(target["phase36_physical_rsrp"], errors="coerce")
    target = target.dropna(subset=["rsrp_residual_db"])
    by_cell = target.groupby("strict_cell_key")["rsrp_residual_db"].median().to_dict()
    by_tech = target.groupby("technology")["rsrp_residual_db"].median().to_dict()

    out = surface.copy()
    cell_resid = out["strict_cell_key"].astype(str).map(by_cell)
    tech_resid = out["technology"].astype(str).map(by_tech)
    out["optimized_baseline_residual_db"] = pd.to_numeric(cell_resid, errors="coerce").fillna(
        pd.to_numeric(tech_resid, errors="coerce")
    ).fillna(0.0)
    out["optimized_final_rsrp_unclipped"] = pd.to_numeric(out["phase36_physical_rsrp"], errors="coerce") + out["optimized_baseline_residual_db"]
    out["optimized_final_rsrp"] = out["optimized_final_rsrp_unclipped"].clip(-140.0, -44.0)
    return out, {"cell_residuals": len(by_cell), "tech_residuals": len(by_tech)}


def _target_quality(surface: pd.DataFrame, points_df: pd.DataFrame, activity=None, corrections=None) -> pd.DataFrame:
    if surface.empty or points_df.empty:
        return pd.DataFrame()
    activity = activity or {}
    corrections = corrections or {}
    work = surface.copy()
    work["carrier_key"] = _carrier_key_for_quality(work)
    grouped = {key: grp for key, grp in work.groupby(["grid_id", "carrier_key"], sort=False)}

    point_targets = points_df[
        ["grid_id", "target_node_cell_id", "baseline_pred_rsrq", "baseline_pred_sinr"]
    ].drop_duplicates(["grid_id", "target_node_cell_id"]).copy()
    target_rows = work.merge(point_targets, on="grid_id", how="inner")
    target_mask = target_rows.apply(lambda row: _row_matches_identity(row, row.get("target_node_cell_id")), axis=1)
    target_rows = target_rows.loc[target_mask].copy()
    rows = []
    for _, row in target_rows.iterrows():
        key = (row["grid_id"], row["carrier_key"])
        grp = grouped.get(key)
        if grp is None or grp.empty:
            continue
        signal = float(row["optimized_final_rsrp"])
        others = grp.loc[grp["strict_cell_key"].astype(str) != str(row["strict_cell_key"]), "optimized_final_rsrp"]
        others = pd.to_numeric(others, errors="coerce").dropna().to_numpy(float)
        gate = others >= max(-125.0, signal - 20.0)
        raw_interference = float(_mw_from_dbm(others[gate]).sum()) if gate.any() else 0.0
        factor = float(activity.get(str(row["carrier_key"]), 1.0))
        base_sinr, base_rsrq = _quality_from_signal(signal, raw_interference * factor, row["technology"])
        corr = corrections.get(str(row["carrier_key"]), {})
        rows.append({
            "grid_id": row["grid_id"],
            "Node_Cell_ID": str(row["strict_cell_key"]),
            "pred_sinr": base_sinr + float(corr.get("sinr", 0.0) or 0.0),
            "pred_rsrq": base_rsrq + float(corr.get("rsrq", 0.0) or 0.0),
            "base_sinr": base_sinr,
            "base_rsrq": base_rsrq,
            "signal_mw": float(_mw_from_dbm(signal)),
            "raw_interference_mw": raw_interference,
            "carrier_key": str(row["carrier_key"]),
            "interfering_sector_count": int(gate.sum()),
            "baseline_pred_rsrq": row.get("baseline_pred_rsrq"),
            "baseline_pred_sinr": row.get("baseline_pred_sinr"),
        })
    return pd.DataFrame(rows)


def _fit_quality_delta(old_surface: pd.DataFrame, points_df: pd.DataFrame) -> tuple[dict, dict]:
    initial = _target_quality(old_surface, points_df)
    activity = {}
    for carrier, grp in initial.groupby("carrier_key", dropna=False):
        sinr = pd.to_numeric(grp["baseline_pred_sinr"], errors="coerce").to_numpy(float)
        sig = pd.to_numeric(grp["signal_mw"], errors="coerce").to_numpy(float)
        intf = pd.to_numeric(grp["raw_interference_mw"], errors="coerce").to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            needed = sig / (np.power(10.0, sinr / 10.0) * intf)
        needed = needed[np.isfinite(needed) & (needed > 0.0)]
        activity[str(carrier)] = float(np.clip(np.median(needed), 0.03, 1.0)) if needed.size else 1.0

    corrected_base = _target_quality(old_surface, points_df, activity=activity)
    corrections = {}
    for carrier, grp in corrected_base.groupby("carrier_key", dropna=False):
        sinr_res = pd.to_numeric(grp["baseline_pred_sinr"], errors="coerce") - pd.to_numeric(grp["base_sinr"], errors="coerce")
        rsrq_res = pd.to_numeric(grp["baseline_pred_rsrq"], errors="coerce") - pd.to_numeric(grp["base_rsrq"], errors="coerce")
        corrections[str(carrier)] = {
            "sinr": float(sinr_res.dropna().median()) if sinr_res.dropna().size else 0.0,
            "rsrq": float(rsrq_res.dropna().median()) if rsrq_res.dropna().size else 0.0,
        }
    return activity, corrections


def run_prediction_only_offset_manual(opt_sites, k1k2_map, params):
    """Manual optimization recompute using the Phase36/37 production baseline as
    the anchor.  The affected-cell selection remains in services/_compute_affected_cells;
    only the RF math for affected rows is replaced."""
    if opt_sites.empty:
        return pd.DataFrame(columns=["lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr", "Node_Cell_ID"])

    region = str(params.get("region", "india")).lower()
    work_df = _normalize_site_df(opt_sites, log_stage="OPTIMIZED_OFFSET_RUN")
    affected_cells = params.get("recompute_cells")
    if affected_cells is None:
        affected_cells, _, _ = _compute_affected_cells(
            work_df,
            float(params.get("impact_radius_m", params.get("radius", 500)) or 500),
            int(params.get("neighbor_site_count", 2) or 2),
            baseline_df=params.get("baseline_df"),
            max_neighbors_per_update_cell=params.get("max_neighbors_per_update_cell"),
        )
    affected_cells = sorted({str(cell).strip() for cell in affected_cells if str(cell).strip()})
    baseline_df = _ensure_canonical_identity(params.get("baseline_df", pd.DataFrame()))
    max_interference_sites = int(params.get("max_interference_sites", 10) or 10)
    site_scope_cells_raw = params.get("site_scope_cells") or affected_cells
    site_scope_cells = sorted({str(cell).strip() for cell in site_scope_cells_raw if str(cell).strip()})
    if not site_scope_cells:
        site_scope_cells = affected_cells
    affected_mask = pd.Series(False, index=work_df.index)
    for scope_cell in site_scope_cells:
        affected_mask = affected_mask | _identity_match_mask(work_df, scope_cell)
    scoped_work_df = work_df.loc[affected_mask].copy()
    if scoped_work_df.empty:
        print("[LTE_OPT][OFFSET_MANUAL_SCOPE] no affected site rows matched; using full site table")
        scoped_work_df = work_df
    else:
        print(
            f"[LTE_OPT][OFFSET_MANUAL_SCOPE] full_site_rows={len(work_df)} "
            f"scoped_site_rows={len(scoped_work_df)} affected_cells={len(affected_cells)} "
            f"site_scope_cells={len(site_scope_cells)}"
        )

    physical_building_df, physical_dem_path, physical_clutter_by_grid = _manual_physical_context(params)
    # Per-job Phase 26 cache (see _phase36_surface_for_points). Callers that
    # evaluate many candidates against the same cells (tilt coordinate search,
    # manual/recommendation optimisation) pass a dict in params and pay the
    # building/DEM scoring once per (cell, grid) instead of once per candidate.
    phase26_cache = params.get("phase26_cache")
    final_list = []
    for cid in affected_cells:
        site_rows = scoped_work_df.loc[_identity_match_mask(scoped_work_df, cid)].copy()
        if site_rows.empty:
            continue
        cell_technology = _technology_series(site_rows).iloc[0] if not site_rows.empty else ""
        same_technology_scope = scoped_work_df.loc[
            _technology_series(scoped_work_df).astype(str).eq(str(cell_technology))
        ].copy() if cell_technology else scoped_work_df
        if same_technology_scope.empty:
            same_technology_scope = scoped_work_df
        local_mod_records = _build_local_interference_records(same_technology_scope, site_rows, max_interference_sites)
        local_mod = pd.DataFrame(local_mod_records)
        local_old = _restore_original_site_state(local_mod)

        local_cell_ids = sorted(local_mod["Node_Cell_ID"].astype(str).dropna().unique().tolist())
        residual_points = _baseline_points_for_cells(baseline_df, local_cell_ids)
        location_changed, moved_m, old_lat, old_lon, new_lat, new_lon = _location_change_summary(site_rows)
        target_point_source = "baseline_prediction_points"
        if location_changed:
            target_points = _generated_points_for_cell(site_rows, cid, params)
            target_point_source = "generated_moved_site_grid"
        else:
            target_points = _baseline_points_for_cells(baseline_df, [cid])
        if target_points.empty:
            target_points = _generated_points_for_cell(site_rows, cid, params)
            target_point_source = "generated_grid_fallback"
        if target_points.empty:
            print(f"[LTE_OPT][OFFSET_MANUAL] cell={cid} skipped_reason=no_points")
            continue
        print(
            f"[LTE_OPT][OFFSET_MANUAL_TARGET_POINTS] cell={cid} "
            f"source={target_point_source} points={len(target_points)} "
            f"location_changed={location_changed} moved_m={moved_m:.2f} "
            f"old_lat={old_lat} old_lon={old_lon} new_lat={new_lat} new_lon={new_lon} "
            f"azimuth={_first_numeric_value(site_rows, 'azimuth')} "
            f"orig_azimuth={_first_numeric_value(site_rows, 'orig_azimuth')}"
        )
        all_points = pd.concat([target_points, residual_points], ignore_index=True).drop_duplicates(["grid_id", "target_node_cell_id"])

        # The "old" (pre-change) reference surface for cid depends only on the fixed
        # geographic neighbor set (local_old is always _restore_original_site_state,
        # i.e. the true production tilt/azimuth) and the static baseline points for
        # cid+neighbors. It does NOT depend on which tilt/azimuth trial is being
        # evaluated, so across many candidate calls in one job it is identical every
        # time. When the caller supplies old_surface_cache (candidate_validation.py's
        # per-job dict; None for the manual-optimization / recommendation-optimization
        # callers, which keep the original always-recompute behavior), reuse it instead
        # of re-running the COST231/phase36 surface + residual + quality-delta fit for
        # cid on every single candidate.
        old_surface_cache = params.get("old_surface_cache")
        cached_old = old_surface_cache.get(cid) if isinstance(old_surface_cache, dict) else None
        if cached_old is not None:
            old_surface_raw, old_surface, residual_summary, activity, quality_corrections = cached_old
        else:
            offset_old_sites = _offset_ready_site_df(local_old, region)
            old_surface_raw = _phase36_surface_for_points(
                offset_old_sites,
                all_points,
                project_id=params.get("project_id"),
                region=region,
                building_df=physical_building_df,
                dem_raster_path=physical_dem_path,
                clutter_by_grid=physical_clutter_by_grid,
                phase26_cache=phase26_cache,
            )
            if old_surface_raw.empty:
                print(f"[LTE_OPT][OFFSET_MANUAL] cell={cid} skipped_reason=no_surface")
                continue
            old_surface, residual_summary = _apply_baseline_rsrp_residual(old_surface_raw, all_points, old_surface_raw)
            activity, quality_corrections = _fit_quality_delta(old_surface, all_points)
            if isinstance(old_surface_cache, dict):
                old_surface_cache[cid] = (old_surface_raw, old_surface, residual_summary, activity, quality_corrections)

        offset_mod_sites = _offset_ready_site_df(local_mod, region)
        mod_surface_raw = _phase36_surface_for_points(
            offset_mod_sites,
            target_points,
            project_id=params.get("project_id"),
            region=region,
            building_df=physical_building_df,
            dem_raster_path=physical_dem_path,
            clutter_by_grid=physical_clutter_by_grid,
            phase26_cache=phase26_cache,
        )
        if mod_surface_raw.empty:
            print(f"[LTE_OPT][OFFSET_MANUAL] cell={cid} skipped_reason=no_surface")
            continue

        mod_surface, _ = _apply_baseline_rsrp_residual(mod_surface_raw, all_points, old_surface_raw)
        quality = _target_quality(mod_surface, target_points, activity=activity, corrections=quality_corrections)

        target_rows = mod_surface.merge(
            target_points[["grid_id", "target_node_cell_id"]],
            on="grid_id",
            how="inner",
        )
        target_mask = target_rows.apply(lambda row: _row_matches_identity(row, row.get("target_node_cell_id")), axis=1)
        target_rows = target_rows.loc[target_mask].copy()
        if target_rows.empty:
            print(f"[LTE_OPT][OFFSET_MANUAL] cell={cid} skipped_reason=no_target_candidate")
            continue
        pts = target_rows.copy()
        pts["pred_rsrp"] = pd.to_numeric(pts["optimized_final_rsrp"], errors="coerce").clip(-140, -44)
        pts = pts.merge(quality[["grid_id", "Node_Cell_ID", "pred_rsrq", "pred_sinr", "interfering_sector_count"]], on=["grid_id", "Node_Cell_ID"], how="left")
        pts["pred_rsrq"] = pd.to_numeric(pts["pred_rsrq"], errors="coerce").clip(-20, -3)
        pts["pred_sinr"] = pd.to_numeric(pts["pred_sinr"], errors="coerce").clip(-10, 30)
        pts = _attach_serving_identity_to_points(pts, site_rows, str(cid))
        print(
            f"[LTE_OPT][OFFSET_MANUAL] cell={cid} points={len(pts)} "
            f"technology={_first_clean_value(site_rows, ['Technology', 'technology']) or 'UNKNOWN'} "
            f"cell_residuals={residual_summary.get('cell_residuals')} "
            f"quality_carriers={len(activity)}"
        )
        final_list.append(pts)

    if not final_list:
        return pd.DataFrame(columns=["lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr", "Node_Cell_ID"])
    final_df = pd.concat(final_list, ignore_index=True)
    _print_fetch_summary(
        "OPTIMIZED_OFFSET_RF_OUTPUT",
        "phase36v2_phase37_baseline_delta",
        {"cells_processed": len(affected_cells)},
        final_df,
        extra={
            "distinct_node_cell_id": _safe_nunique(final_df, "Node_Cell_ID"),
            "pred_rsrp_range": _safe_minmax(final_df, "pred_rsrp"),
            "pred_rsrq_range": _safe_minmax(final_df, "pred_rsrq"),
            "pred_sinr_range": _safe_minmax(final_df, "pred_sinr"),
        },
    )
    return final_df


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
        affected_cells = [
            cell for cell in affected_cells
            if bool(_identity_match_mask(work_df, cell).any())
        ]
        affected_mask = pd.Series(False, index=work_df.index)
        for cell in affected_cells:
            affected_mask = affected_mask | _identity_match_mask(work_df, cell)
        affected_rows = work_df.loc[affected_mask].copy()
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
    progress_callback = params.get("progress_callback")
    progress_label = str(params.get("progress_label") or "optimized prediction")
    total_cells = len(affected_cells)
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

    for cell_index, cid in enumerate(affected_cells, start=1):
        if callable(progress_callback):
            progress_callback(progress_label, cell_index, total_cells, cid, "start", None, None)
        print(f"[LTE_OPT][RUN] cell_start={cid}")
        site_rows = work_df.loc[_identity_match_mask(work_df, cid)].copy()
        if site_rows.empty:
            if callable(progress_callback):
                progress_callback(progress_label, cell_index, total_cells, cid, "skipped", None, 0)
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
                    _identity_match_mask(prediction_points_df, cid),
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
        cell_geo = geo_features_df.loc[_identity_match_mask(geo_features_df, cid)].copy() if not geo_features_df.empty else pd.DataFrame()
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
        pts = _attach_serving_identity_to_points(pts, site_rows, str(cid))
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
        if callable(progress_callback):
            progress_callback(progress_label, cell_index, total_cells, cid, "done", elapsed, len(pts))
        final_list.append(pts)

    if not final_list:
        return pd.DataFrame(columns=["lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr", "Node_Cell_ID"])
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
