from __future__ import annotations

import contextlib
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tools.lte_prediction_optimised import ml_engine as opt_ml
from tools.lte_tilt_recommandation.cell_identity import canonical_cell_id
from tools.lte_tilt_recommandation.geo_logic import (
    MIN_BEARING_SAMPLE_COUNT,
    _normalize_azimuth,
    _signed_azimuth_delta,
    compute_dominant_bearing_summary,
)


SEVERE_CONSTRAINT_PENALTY = 100000.0
SEVERITY_CAPS = {
    "rsrp": 25.0,
    "rsrq": 8.0,
    "sinr": 15.0,
}
MEAN_DELTA_SCALES = {
    "rsrp": 10.0,
    "rsrq": 3.0,
    "sinr": 10.0,
}


@dataclass
class CandidateValidationConfig:
    project_id: int
    region: str
    radius_m: float = 500.0
    grid_resolution_m: float = 30.0
    workers: int = 1
    impact_radius_m: float = 500.0
    neighbor_site_count: int = 3
    max_interference_sites: int = 10
    max_candidates: int = 25
    baseline_job_id: Optional[str] = None
    coordinate_passes: int = 2
    candidate_workers: int = 1
    bad_grid_coverage_pct: float = 80.0
    max_group_cells: int = 0
    max_neighbors_per_update_cell: int = 2
    min_safe_etilt: float = 2.0
    max_safe_etilt: float = 12.0
    max_good_area_loss_pct: float = 15.0
    etilt_candidate_max_delta_deg: float = 4.0
    enable_azimuth_fallback: bool = True
    azimuth_fallback_max_delta_deg: float = 30.0
    azimuth_fallback_step_deg: float = 5.0
    azimuth_fallback_steps_deg: Optional[tuple[float, ...]] = None
    rf_debug_log_path: Optional[str] = None
    constraint_map: Optional[Dict[str, Dict[str, object]]] = None


def normalize_grid_id_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="string")
    out = series.astype("string").str.strip()
    return out.mask(out.isin(["", "nan", "NaN", "None", "<NA>"]))


def _clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _threshold_cell_id(value: object) -> str:
    return canonical_cell_id(value)


def _numeric_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _safe_float(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else float(default)


def _is_export_scalar(value: object) -> bool:
    if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
        return True
    return value is None


def _frontend_grid_component(metrics: Dict[str, object], kpi: str) -> float:
    severity_term = _safe_float(metrics.get(f"{kpi}_severity_reduction_per_grid")) / SEVERITY_CAPS[kpi]
    recovery_term = _safe_float(metrics.get(f"{kpi}_grid_recovered_share")) - _safe_float(metrics.get(f"{kpi}_grid_new_bad_share"))
    mean_delta_term = _safe_float(metrics.get(f"mean_{kpi}_delta")) / MEAN_DELTA_SCALES[kpi]
    return float(severity_term + recovery_term + mean_delta_term)


def _positive_delta_sequence(max_delta: object, step: object) -> list[float]:
    max_value = _safe_float(max_delta, 0.0)
    step_value = _safe_float(step, 0.0)
    if max_value <= 0.0 or step_value <= 0.0:
        return []
    max_value = abs(float(max_value))
    step_value = abs(float(step_value))
    values: list[float] = []
    current = step_value
    while current < max_value and not np.isclose(current, max_value):
        values.append(round(float(current), 6))
        current += step_value
    values.append(round(float(max_value), 6))
    deduped: list[float] = []
    for value in values:
        if value > 0.0 and not any(np.isclose(value, existing) for existing in deduped):
            deduped.append(float(value))
    return deduped


def _etilt_candidate_delta_sets(config: CandidateValidationConfig) -> tuple[list[float], Dict[str, list[float]]]:
    max_delta = abs(_safe_float(config.etilt_candidate_max_delta_deg, 4.0))
    if max_delta <= 0.0:
        max_delta = 4.0
    first_abs = min(1.0, max_delta)
    positives = _positive_delta_sequence(max_delta, 1.0)
    directional_positive = [value for value in positives if value > first_abs and not np.isclose(value, first_abs)]
    return (
        [-first_abs, first_abs],
        {
            "plus": directional_positive,
            "minus": [-value for value in directional_positive],
        },
    )


def _azimuth_fallback_steps(config: CandidateValidationConfig) -> list[float]:
    explicit_steps = config.azimuth_fallback_steps_deg
    if explicit_steps:
        steps = [abs(float(step)) for step in explicit_steps if abs(float(step)) > 0.0]
        return sorted(set(round(step, 6) for step in steps))
    max_delta = abs(_safe_float(config.azimuth_fallback_max_delta_deg, 30.0))
    step = abs(_safe_float(config.azimuth_fallback_step_deg, 5.0))
    if max_delta <= 0.0:
        max_delta = 30.0
    if step <= 0.0:
        step = 5.0
    return _positive_delta_sequence(max_delta, step)


@contextlib.contextmanager
def _rf_debug_capture(config: CandidateValidationConfig):
    path = str(config.rf_debug_log_path or "").strip()
    if not path:
        yield
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", errors="replace") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            yield


def _format_etilt_updates(updates: list[Dict[str, object]]) -> str:
    if not updates:
        return "[]"
    parts = []
    for update in updates:
        parts.append(
            "{cell}: {current:.1f}->{target:.1f} ({delta:+.1f})".format(
                cell=_clean_id(update.get("cell_id")),
                current=_safe_float(update.get("current_value"), np.nan),
                target=_safe_float(update.get("target_value"), np.nan),
                delta=_safe_float(update.get("actual_delta"), np.nan),
            )
        )
    return "[" + "; ".join(parts) + "]"


def _format_updates(updates: list[Dict[str, object]]) -> str:
    if not updates:
        return "[]"
    parts = []
    for update in updates:
        parameter = str(update.get("parameter", "") or "").strip() or "Update"
        parts.append(
            "{cell} {parameter}: {current:.1f}->{target:.1f} ({delta:+.1f})".format(
                cell=_clean_id(update.get("cell_id")),
                parameter=parameter,
                current=_safe_float(update.get("current_value"), np.nan),
                target=_safe_float(update.get("target_value"), np.nan),
                delta=_safe_float(update.get("actual_delta"), np.nan),
            )
        )
    return "[" + "; ".join(parts) + "]"


def _ensure_node_cell(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()
    if "Node_Cell_ID" in out.columns:
        out["Node_Cell_ID"] = out["Node_Cell_ID"].astype(str).str.strip()
        return out
    if {"nodeb_id", "cell_id"}.issubset(out.columns):
        out["Node_Cell_ID"] = out["nodeb_id"].map(_clean_id) + "_" + out["cell_id"].map(_clean_id)
    elif {"node_b_id", "cell_id"}.issubset(out.columns):
        out["Node_Cell_ID"] = out["node_b_id"].map(_clean_id) + "_" + out["cell_id"].map(_clean_id)
    return out


_MATCH_ALIAS_COLS = [
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


def _identity_match_mask(df: pd.DataFrame, cell_id: object) -> pd.Series:
    target = _clean_id(cell_id)
    if df.empty or not target:
        return pd.Series(False, index=df.index)
    target_aliases = {target, canonical_cell_id(target)}
    for col in [c for c in _MATCH_ALIAS_COLS if c in df.columns]:
        values = df[col].map(_clean_id)
        mask = values.isin(target_aliases)
        if bool(mask.any()):
            return mask
        canonical_mask = values.map(canonical_cell_id).isin(target_aliases)
        if bool(canonical_mask.any()):
            return canonical_mask
    return pd.Series(False, index=df.index)


def _prepare_optimizer_site_df(antenna_df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_node_cell(antenna_df)
    if "cell_id" in out.columns:
        original_cell_id = out["cell_id"].map(_clean_id)
        if "local_cell_id" not in out.columns:
            out["local_cell_id"] = original_cell_id
        else:
            local_cell_id = out["local_cell_id"].map(_clean_id)
            out["local_cell_id"] = local_cell_id.where(local_cell_id.astype(str).str.strip().ne(""), original_cell_id)
    elif "Node_Cell_ID" in out.columns:
        out["cell_id"] = out["Node_Cell_ID"].astype(str).str.strip()
        out["local_cell_id"] = out["cell_id"].map(_clean_id)
    if "nodeb_id_cell_id" not in out.columns and "local_cell_id" in out.columns:
        node_col = "nodeb_id" if "nodeb_id" in out.columns else ("node_b_id" if "node_b_id" in out.columns else None)
        if node_col:
            out["nodeb_id_cell_id"] = (
                out[node_col].map(_clean_id).astype(str).str.strip()
                + "_"
                + out["local_cell_id"].map(_clean_id).astype(str).str.strip()
            ).str.strip("_")
        elif "Node_Cell_ID" in out.columns:
            out["nodeb_id_cell_id"] = out["Node_Cell_ID"].map(_clean_id)
    alias_cols = {
        "latitude": "lat",
        "longitude": "lon",
        "e_tilt": "electrical_tilt",
        "m_tilt": "mechanical_tilt",
        "height": "antenna_height",
    }
    drop_aliases = [alias for alias, canonical in alias_cols.items() if alias in out.columns and canonical in out.columns]
    return out.drop(columns=drop_aliases) if drop_aliases else out


def _parameter_to_site_column(parameter: object) -> Optional[str]:
    text = str(parameter or "").strip().lower()
    if text in {"etilt", "e_tilt", "electrical tilt", "electrical_tilt"}:
        return "electrical_tilt"
    if text in {"azimuth", "azi"}:
        return "azimuth"
    if text in {"tx power", "power", "tx_power"}:
        return "tx_power"
    if text in {"mechanical tilt", "mtilt", "mechanical_tilt"}:
        return "mechanical_tilt"
    if text in {"height", "antenna height", "antenna_height"}:
        return "antenna_height"
    return None


def _apply_updates_to_site_df(
    antenna_df: pd.DataFrame,
    updates: list[Dict[str, object]],
) -> pd.DataFrame:
    if {"Node_Cell_ID", "electrical_tilt"}.issubset(antenna_df.columns):
        site_df = antenna_df.copy()
    else:
        site_df = opt_ml._normalize_site_df(_prepare_optimizer_site_df(antenna_df), log_stage="TILT_COORDINATE_SITE")
    for col in ["lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "tx_power", "antenna_height"]:
        if col in site_df.columns:
            site_df[f"orig_{col}"] = pd.to_numeric(site_df[col], errors="coerce")
    applied = pd.Series(False, index=site_df.index)
    for update in updates:
        cell_id = _clean_id(update.get("cell_id"))
        site_col = _parameter_to_site_column(update.get("parameter", "ETilt"))
        target = pd.to_numeric(pd.Series([update.get("target_value")]), errors="coerce").iloc[0]
        if not cell_id or site_col is None or pd.isna(target):
            continue
        mask = _identity_match_mask(site_df, cell_id)
        if not mask.any():
            suffix = cell_id.split("_")[-1]
            mask = site_df.get("cell_id", pd.Series("", index=site_df.index)).map(_clean_id) == suffix
        if mask.any():
            site_df.loc[mask, site_col] = float(target)
            applied = applied | mask
    site_df["optimization_applied"] = applied.astype(bool)
    return site_df


def _attach_grid_context(pred_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_node_cell(pred_df)
    if "grid_id" in out.columns:
        return out
    if context_df.empty or "grid_id" not in context_df.columns:
        return out
    ctx = _ensure_node_cell(context_df)
    required = {"Node_Cell_ID", "lat", "lon", "grid_id"}
    if not required.issubset(ctx.columns) or not {"Node_Cell_ID", "lat", "lon"}.issubset(out.columns):
        return out
    out["_lat_key"] = pd.to_numeric(out["lat"], errors="coerce").round(6)
    out["_lon_key"] = pd.to_numeric(out["lon"], errors="coerce").round(6)
    out["_cell_key"] = out["Node_Cell_ID"].astype(str).str.strip()
    ctx = ctx[["Node_Cell_ID", "lat", "lon", "grid_id"]].copy()
    ctx["_lat_key"] = pd.to_numeric(ctx["lat"], errors="coerce").round(6)
    ctx["_lon_key"] = pd.to_numeric(ctx["lon"], errors="coerce").round(6)
    ctx["_cell_key"] = ctx["Node_Cell_ID"].astype(str).str.strip()
    ctx = ctx.drop_duplicates(subset=["_cell_key", "_lat_key", "_lon_key"], keep="last")
    out = out.merge(ctx[["_cell_key", "_lat_key", "_lon_key", "grid_id"]], on=["_cell_key", "_lat_key", "_lon_key"], how="left")
    return out.drop(columns=["_lat_key", "_lon_key", "_cell_key"], errors="ignore")


def _attach_grid_context_from_analytics(pred_df: pd.DataFrame, grid_analytics_df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_node_cell(pred_df)
    if out.empty or "grid_id" in out.columns:
        return out
    required = {"grid_id", "min_lat", "max_lat", "min_lon", "max_lon"}
    if grid_analytics_df is None or grid_analytics_df.empty or not required.issubset(grid_analytics_df.columns):
        return out
    if not {"lat", "lon"}.issubset(out.columns):
        return out

    work = out.copy()
    work["grid_id"] = pd.NA
    lat = pd.to_numeric(work["lat"], errors="coerce")
    lon = pd.to_numeric(work["lon"], errors="coerce")
    grids = grid_analytics_df[list(required)].dropna(subset=["grid_id", "min_lat", "max_lat", "min_lon", "max_lon"]).copy()
    for col in ["min_lat", "max_lat", "min_lon", "max_lon"]:
        grids[col] = pd.to_numeric(grids[col], errors="coerce")
    grids = grids.dropna(subset=["min_lat", "max_lat", "min_lon", "max_lon"])
    for _, grid in grids.iterrows():
        missing = work["grid_id"].isna()
        if not bool(missing.any()):
            break
        mask = (
            missing
            & lat.ge(float(grid["min_lat"]))
            & lat.le(float(grid["max_lat"]))
            & lon.ge(float(grid["min_lon"]))
            & lon.le(float(grid["max_lon"]))
        )
        if bool(mask.any()):
            work.loc[mask, "grid_id"] = str(grid["grid_id"])
    mapped = int(work["grid_id"].notna().sum())
    print(f"[TILT_GRID_ATTACH] source=grid_analytics_bounds rows={len(work)} mapped_rows={mapped}")
    return work


def _apply_rf_delta(
    stored_baseline_df: pd.DataFrame,
    rf_baseline_df: pd.DataFrame,
    rf_candidate_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    out = _ensure_node_cell(stored_baseline_df)
    rf_base = _ensure_node_cell(rf_baseline_df)
    rf_cand = _ensure_node_cell(rf_candidate_df)
    metric_cols = [col for col in ["pred_rsrp", "pred_rsrq", "pred_sinr"] if col in out.columns]
    required = {"Node_Cell_ID", "lat", "lon"}
    if not metric_cols or not required.issubset(out.columns) or not required.issubset(rf_base.columns) or not required.issubset(rf_cand.columns):
        return out, {"rf_delta_matched_row_count": 0.0, "rf_delta_match_pct": 0.0}

    for frame in [out, rf_base, rf_cand]:
        for col in metric_cols:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    for frame in [out, rf_base, rf_cand]:
        frame["_cell_key"] = frame["Node_Cell_ID"].astype(str).str.strip()
        frame["_lat_key"] = pd.to_numeric(frame["lat"], errors="coerce").round(6)
        frame["_lon_key"] = pd.to_numeric(frame["lon"], errors="coerce").round(6)
    key_cols = ["_cell_key", "_lat_key", "_lon_key"]
    rf_base = rf_base.drop_duplicates(subset=key_cols, keep="last")
    rf_cand = rf_cand.drop_duplicates(subset=key_cols, keep="last")
    delta_source = rf_base[key_cols + metric_cols].merge(
        rf_cand[key_cols + metric_cols],
        on=key_cols,
        how="inner",
        suffixes=("_base", "_cand"),
    )
    if delta_source.empty:
        return out.drop(columns=key_cols, errors="ignore"), {"rf_delta_matched_row_count": 0.0, "rf_delta_match_pct": 0.0}
    out_indexed = out.set_index(key_cols, drop=False)
    delta_indexed = delta_source.set_index(key_cols, drop=False)
    common_index = delta_indexed.index.intersection(out_indexed.index)
    for col in metric_cols:
        delta = (
            pd.to_numeric(delta_indexed.loc[common_index, f"{col}_cand"], errors="coerce")
            - pd.to_numeric(delta_indexed.loc[common_index, f"{col}_base"], errors="coerce")
        ).fillna(0.0)
        out_indexed.loc[common_index, col] = pd.to_numeric(out_indexed.loc[common_index, col], errors="coerce") + delta
    out = out_indexed.reset_index(drop=True).drop(columns=key_cols, errors="ignore")
    match_pct = float(len(common_index) / max(len(rf_cand), 1) * 100.0)
    return out, {"rf_delta_matched_row_count": float(len(common_index)), "rf_delta_match_pct": match_pct}


def _aggregate_grids(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty or "grid_id" not in pred_df.columns:
        return pd.DataFrame()
    work = pred_df.copy()
    work["grid_id"] = normalize_grid_id_series(work["grid_id"])
    work = work.loc[work["grid_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    agg = {"point_count": ("grid_id", "count")}
    for kpi in ["rsrp", "rsrq", "sinr"]:
        col = f"pred_{kpi}"
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            agg[f"avg_{kpi}"] = (col, "mean")
    return work.groupby("grid_id", dropna=False).agg(**agg).reset_index()


def prepare_scope_export(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
    stage: str,
) -> pd.DataFrame:
    out = _ensure_node_cell(df)
    if out.empty:
        return out
    for kpi in ["rsrp", "rsrq", "sinr"]:
        col = f"pred_{kpi}"
        out[col] = _numeric_series(out, col)
        out[f"is_bad_{kpi}"] = (
            out[col] < float(thresholds[kpi])
            if float(weights.get(kpi, 0.0)) > 0.0
            else False
        )
        out[f"{kpi}_severity"] = _severity(out[col], thresholds[kpi])
    out["is_bad_combined"] = out[["is_bad_rsrp", "is_bad_rsrq", "is_bad_sinr"]].any(axis=1)
    out["combined_weighted_severity"] = (
        out["rsrp_severity"] * float(weights.get("rsrp", 0.0))
        + out["rsrq_severity"] * float(weights.get("rsrq", 0.0))
        + out["sinr_severity"] * float(weights.get("sinr", 0.0))
    )
    out["stage"] = stage
    return out


def prepare_grid_metrics_export(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
) -> pd.DataFrame:
    out = _aggregate_grids(df)
    if out.empty:
        return out
    score = pd.Series(0.0, index=out.index)
    bad_any = pd.Series(False, index=out.index)
    for kpi in ["rsrp", "rsrq", "sinr"]:
        avg_col = f"avg_{kpi}"
        if avg_col not in out.columns:
            out[avg_col] = np.nan
        out[f"is_bad_{kpi}"] = (
            pd.to_numeric(out[avg_col], errors="coerce") < float(thresholds[kpi])
            if float(weights.get(kpi, 0.0)) > 0.0
            else False
        )
        out[f"{kpi}_severity"] = _severity(out[avg_col], thresholds[kpi])
        bad_any = bad_any | out[f"is_bad_{kpi}"].fillna(False)
        score = score + out[f"{kpi}_severity"] * float(weights.get(kpi, 0.0))
    out["is_bad_combined"] = bad_any
    out["combined_weighted_severity"] = score
    return out


def build_combined_kpi_grid_impact(before_grid: pd.DataFrame, after_grid: pd.DataFrame, thresholds: Dict[str, float], weights: Dict[str, float]) -> pd.DataFrame:
    if before_grid.empty or after_grid.empty or "grid_id" not in before_grid.columns or "grid_id" not in after_grid.columns:
        return pd.DataFrame()
    merged = before_grid.merge(after_grid, on="grid_id", how="inner", suffixes=("_before", "_after"))
    rows = []
    for kpi in ["rsrp", "rsrq", "sinr"]:
        before_col = f"avg_{kpi}_before"
        after_col = f"avg_{kpi}_after"
        if before_col not in merged.columns or after_col not in merged.columns:
            continue
        before = pd.to_numeric(merged[before_col], errors="coerce")
        after = pd.to_numeric(merged[after_col], errors="coerce")
        active = float(weights.get(kpi, 0.0)) > 0.0
        was_bad = before < float(thresholds[kpi]) if active else pd.Series(False, index=merged.index)
        is_bad = after < float(thresholds[kpi]) if active else pd.Series(False, index=merged.index)
        rows.append(
            {
                "kpi": kpi.upper(),
                "threshold": float(thresholds[kpi]),
                "weight": float(weights.get(kpi, 0.0)),
                "before_bad_grids": int(was_bad.fillna(False).sum()),
                "after_bad_grids": int(is_bad.fillna(False).sum()),
                "net_bad_grid_reduction": int(was_bad.fillna(False).sum() - is_bad.fillna(False).sum()),
                "bad_to_good_grids": int((was_bad & ~is_bad).fillna(False).sum()),
                "good_to_bad_grids": int((~was_bad & is_bad).fillna(False).sum()),
                "mean_before": float(before.mean()) if len(before.dropna()) else np.nan,
                "mean_after": float(after.mean()) if len(after.dropna()) else np.nan,
                "mean_delta": float((after - before).mean()) if len(before.dropna()) and len(after.dropna()) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _frontend_reference_grid(grid_analytics_df: pd.DataFrame) -> pd.DataFrame:
    if grid_analytics_df.empty or "grid_id" not in grid_analytics_df.columns:
        return pd.DataFrame()
    out = grid_analytics_df.copy()
    out["grid_id"] = normalize_grid_id_series(out["grid_id"])
    for kpi in ["rsrp", "rsrq", "sinr"]:
        source = f"baseline_avg_{kpi}"
        out[f"avg_{kpi}"] = pd.to_numeric(out[source], errors="coerce") if source in out.columns else np.nan
    keep = [col for col in ["grid_id", "avg_rsrp", "avg_rsrq", "avg_sinr", "baseline_point_count"] if col in out.columns]
    return out.loc[out["grid_id"].notna(), keep].drop_duplicates(subset=["grid_id"], keep="first")


def _severity(values: pd.Series, threshold: float) -> pd.Series:
    return (float(threshold) - pd.to_numeric(values, errors="coerce")).clip(lower=0.0).fillna(0.0)


def _score_before_after(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    grid_analytics_df: pd.DataFrame,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
    config: Optional[CandidateValidationConfig] = None,
) -> Dict[str, float | str]:
    base_grid = _aggregate_grids(baseline_df)
    cand_grid = _aggregate_grids(candidate_df)
    reference = _frontend_reference_grid(grid_analytics_df)
    if not reference.empty and not base_grid.empty and not cand_grid.empty:
        delta = base_grid.merge(cand_grid, on="grid_id", how="inner", suffixes=("_model_base", "_model_cand"))
        for kpi in ["rsrp", "rsrq", "sinr"]:
            delta[f"avg_{kpi}_delta"] = _numeric_series(delta, f"avg_{kpi}_model_cand") - _numeric_series(delta, f"avg_{kpi}_model_base")
        merged = reference.rename(columns={f"avg_{kpi}": f"avg_{kpi}_base" for kpi in ["rsrp", "rsrq", "sinr"]})
        merged = merged.merge(delta[["grid_id", "avg_rsrp_delta", "avg_rsrq_delta", "avg_sinr_delta"]], on="grid_id", how="left")
        for kpi in ["rsrp", "rsrq", "sinr"]:
            merged[f"avg_{kpi}_cand"] = pd.to_numeric(merged[f"avg_{kpi}_base"], errors="coerce") + pd.to_numeric(merged[f"avg_{kpi}_delta"], errors="coerce").fillna(0.0)
        source = "frontend_grid_analytics_plus_candidate_rf_delta_weighted_kpi_formula"
    else:
        merged = base_grid.merge(cand_grid, on="grid_id", how="outer", suffixes=("_base", "_cand"))
        source = "recomputed_rf_grid_weighted_kpi_formula"
    if merged.empty:
        return {"score": -99999.0, "constraints_passed": 0.0, "grid_scoring_source": "no_grid_overlap"}

    evaluation_grid_count = max(float(len(merged)), 1.0)
    before_any = pd.Series(False, index=merged.index)
    after_any = pd.Series(False, index=merged.index)
    total_severity_reduction = 0.0
    weighted_severity_reduction = 0.0
    metric_fields: Dict[str, float] = {}
    for kpi in ["rsrp", "rsrq", "sinr"]:
        weight = float(weights.get(kpi, 0.0))
        base_values = _numeric_series(merged, f"avg_{kpi}_base")
        cand_values = _numeric_series(merged, f"avg_{kpi}_cand").fillna(base_values)
        active = weight > 0.0
        before_bad = base_values < float(thresholds[kpi]) if active else pd.Series(False, index=merged.index)
        after_bad = cand_values < float(thresholds[kpi]) if active else pd.Series(False, index=merged.index)
        before_any = before_any | before_bad
        after_any = after_any | after_bad
        severity_reduction = float((_severity(base_values, thresholds[kpi]) - _severity(cand_values, thresholds[kpi])).sum())
        severity_per_grid = severity_reduction / evaluation_grid_count
        mean_delta = float((cand_values - base_values).mean())
        recovered = int((before_bad & ~after_bad).sum())
        new_bad_kpi = int((~before_bad & after_bad).sum())
        before_bad_count = int(before_bad.sum())
        after_bad_count = int(after_bad.sum())
        total_severity_reduction += severity_reduction
        weighted_severity_reduction += severity_per_grid * weight / SEVERITY_CAPS[kpi] if active else 0.0
        component_metrics = {
            f"{kpi}_severity_reduction_per_grid": severity_per_grid,
            f"{kpi}_grid_recovered_share": float(recovered) / evaluation_grid_count,
            f"{kpi}_grid_new_bad_share": float(new_bad_kpi) / evaluation_grid_count,
            f"mean_{kpi}_delta": mean_delta,
        }
        metric_fields.update(
            {
                f"frontend_{kpi}_before_bad_grid_count": float(before_bad_count),
                f"frontend_{kpi}_after_bad_grid_count": float(after_bad_count),
                f"frontend_{kpi}_net_bad_grid_reduction": float(before_bad_count - after_bad_count),
                f"{kpi}_recovered_bad": float(recovered),
                f"{kpi}_new_bad": float(new_bad_kpi),
                f"{kpi}_severity_reduction": float(severity_reduction),
                f"{kpi}_severity_reduction_per_grid": float(severity_per_grid),
                f"{kpi}_grid_recovered_share": float(recovered) / evaluation_grid_count,
                f"{kpi}_grid_new_bad_share": float(new_bad_kpi) / evaluation_grid_count,
                f"mean_{kpi}_delta": float(mean_delta),
                f"combined_{kpi}_component": float(_frontend_grid_component(component_metrics, kpi)),
            }
        )

    active_weights = {kpi: float(weights.get(kpi, 0.0)) for kpi in ["rsrp", "rsrq", "sinr"] if float(weights.get(kpi, 0.0)) > 0.0}
    weight_sum = max(float(sum(active_weights.values())), 1.0)
    weighted_before_bad_count = sum(
        (weight / weight_sum) * _safe_float(metric_fields.get(f"frontend_{kpi}_before_bad_grid_count"))
        for kpi, weight in active_weights.items()
    )
    weighted_after_bad_count = sum(
        (weight / weight_sum) * _safe_float(metric_fields.get(f"frontend_{kpi}_after_bad_grid_count"))
        for kpi, weight in active_weights.items()
    )
    weighted_recovered_bad = sum(
        (weight / weight_sum) * _safe_float(metric_fields.get(f"{kpi}_recovered_bad"))
        for kpi, weight in active_weights.items()
    )
    weighted_new_bad = sum(
        (weight / weight_sum) * _safe_float(metric_fields.get(f"{kpi}_new_bad"))
        for kpi, weight in active_weights.items()
    )
    weighted_net_bad_reduction = weighted_before_bad_count - weighted_after_bad_count

    max_weight = max([float(weights.get(kpi, 0.0)) for kpi in ["rsrp", "rsrq", "sinr"]] or [0.0])
    priority_kpis = [kpi for kpi in ["rsrp", "rsrq", "sinr"] if float(weights.get(kpi, 0.0)) == max_weight and max_weight > 0.0]
    if len(priority_kpis) == 1:
        decision_kpi = priority_kpis[0]
        baseline_bad_count = _safe_float(metric_fields.get(f"frontend_{decision_kpi}_before_bad_grid_count"))
        candidate_bad_count = _safe_float(metric_fields.get(f"frontend_{decision_kpi}_after_bad_grid_count"))
        recovered_bad_count = _safe_float(metric_fields.get(f"{decision_kpi}_recovered_bad"))
        new_bad_count = _safe_float(metric_fields.get(f"{decision_kpi}_new_bad"))
        net_bad_reduction = baseline_bad_count - candidate_bad_count
        decision_scope = f"priority_kpi_frontend_{decision_kpi}"
    else:
        baseline_bad_count = weighted_before_bad_count
        candidate_bad_count = weighted_after_bad_count
        recovered_bad_count = weighted_recovered_bad
        new_bad_count = weighted_new_bad
        net_bad_reduction = weighted_net_bad_reduction
        decision_scope = "weighted_frontend_kpi_blend"

    recovered_combined = int((before_any & ~after_any).sum())
    new_bad_combined = int((~before_any & after_any).sum())
    combined_baseline_bad_count = int(before_any.sum())
    combined_candidate_bad_count = int(after_any.sum())
    combined_net_bad_reduction = combined_baseline_bad_count - combined_candidate_bad_count
    good_area_loss_pct = (float(weighted_new_bad) / evaluation_grid_count) * 100.0
    net_bad_reduction_share = float(weighted_net_bad_reduction) / evaluation_grid_count
    score = (
        float(weights.get("rsrp", 0.0)) * _frontend_grid_component(metric_fields, "rsrp")
        + float(weights.get("rsrq", 0.0)) * _frontend_grid_component(metric_fields, "rsrq")
        + float(weights.get("sinr", 0.0)) * _frontend_grid_component(metric_fields, "sinr")
        + net_bad_reduction_share * 2.0
        - good_area_loss_pct * 0.0025
    )
    priority_worsened = any(_safe_float(metric_fields.get(f"frontend_{kpi}_net_bad_grid_reduction")) < 0.0 for kpi in priority_kpis)
    max_loss = float(config.max_good_area_loss_pct) if config is not None else 15.0
    constraints_passed = good_area_loss_pct <= max(max_loss, 15.0) and not priority_worsened
    if not constraints_passed:
        score -= SEVERE_CONSTRAINT_PENALTY

    return {
        "score": float(score),
        "baseline_bad_count": float(baseline_bad_count),
        "candidate_bad_count": float(candidate_bad_count),
        "recovered_bad_samples": float(recovered_bad_count),
        "new_bad_samples": float(new_bad_count),
        "net_bad_reduction": float(net_bad_reduction),
        "net_bad_reduction_share": float(net_bad_reduction_share),
        "recovered_bad_share": float(recovered_bad_count) / evaluation_grid_count,
        "new_bad_share": float(new_bad_count) / evaluation_grid_count,
        "weighted_frontend_before_bad_grid_count": float(weighted_before_bad_count),
        "weighted_frontend_after_bad_grid_count": float(weighted_after_bad_count),
        "weighted_frontend_net_bad_grid_reduction": float(weighted_net_bad_reduction),
        "combined_any_before_bad_grid_count": float(combined_baseline_bad_count),
        "combined_any_after_bad_grid_count": float(combined_candidate_bad_count),
        "combined_any_net_bad_grid_reduction": float(combined_net_bad_reduction),
        "combined_any_recovered_bad": float(recovered_combined),
        "combined_any_new_bad": float(new_bad_combined),
        "decision_scope": decision_scope,
        "total_severity_reduction": float(total_severity_reduction),
        "good_area_loss_pct": float(good_area_loss_pct),
        "constraints_passed": float(1 if constraints_passed else 0),
        "priority_kpi_worsened": float(1 if priority_worsened else 0),
        "frontend_scoring_used": float(1 if source.startswith("frontend_") else 0),
        "grid_scoring_source": source,
        "validation_scope": "frontend_grid_analytics_population_with_candidate_rf_delta" if source.startswith("frontend_") else "recomputed_rf_grid_population",
        "evaluation_grid_count": float(evaluation_grid_count),
        "frontend_common_grid_count": float(len(merged)) if source.startswith("frontend_") else 0.0,
        "combined_weighted_severity_reduction": float(weighted_severity_reduction),
        "combined_weighted_tie_break": float(weighted_severity_reduction),
        "combined_rsrp_weight": float(weights.get("rsrp", 0.0)),
        "combined_rsrq_weight": float(weights.get("rsrq", 0.0)),
        "combined_sinr_weight": float(weights.get("sinr", 0.0)),
        "baseline_bad_grid_count": float(baseline_bad_count),
        "candidate_bad_grid_count": float(candidate_bad_count),
        "bad_to_good_grid_count": float(recovered_bad_count),
        "good_to_bad_grid_count": float(new_bad_count),
        "grid_mean_rsrp_delta_bad_baseline": metric_fields.get("mean_rsrp_delta", 0.0),
        **metric_fields,
    }


def _combined_grid_reference(
    baseline_df: pd.DataFrame,
    grid_analytics_df: pd.DataFrame,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
) -> pd.DataFrame:
    if grid_analytics_df is not None and not grid_analytics_df.empty:
        grid = _frontend_reference_grid(grid_analytics_df)
    else:
        grid = _aggregate_grids(baseline_df)
    if grid.empty or "grid_id" not in grid.columns:
        return pd.DataFrame()
    score = pd.Series(0.0, index=grid.index)
    bad_any = pd.Series(False, index=grid.index)
    for kpi in ["rsrp", "rsrq", "sinr"]:
        if float(weights.get(kpi, 0.0)) <= 0.0:
            continue
        avg_col = f"avg_{kpi}"
        if avg_col not in grid.columns:
            continue
        sev = _severity(grid[avg_col], thresholds[kpi])
        grid[f"{kpi}_severity"] = sev
        grid[f"is_bad_{kpi}"] = sev > 0
        score = score + sev * float(weights[kpi])
        bad_any = bad_any | grid[f"is_bad_{kpi}"].fillna(False)
    grid["combined_weighted_severity"] = score
    grid["is_bad_combined"] = bad_any
    source = "frontend_grid_analytics" if grid_analytics_df is not None and not grid_analytics_df.empty else "recomputed_baseline_grid"
    counts = {}
    for kpi in ["rsrp", "rsrq", "sinr"]:
        flag = f"is_bad_{kpi}"
        counts[kpi] = int(grid[flag].fillna(False).sum()) if flag in grid.columns else 0
    print(
        "[TILT_GRID_REFERENCE] "
        f"source={source} grid_rows={len(grid)} "
        f"rsrp_threshold={float(thresholds.get('rsrp', np.nan))} rsrp_bad_grids={counts['rsrp']} "
        f"rsrq_threshold={float(thresholds.get('rsrq', np.nan))} rsrq_bad_grids={counts['rsrq']} "
        f"sinr_threshold={float(thresholds.get('sinr', np.nan))} sinr_bad_grids={counts['sinr']} "
        f"combined_bad_grids={int(grid['is_bad_combined'].fillna(False).sum())} "
        f"weights={weights}"
    )
    return grid


def _weighted_kpi_cell_order(summary: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    if summary.empty:
        return summary
    total_cells = int(len(summary))
    active = [(kpi, float(weights.get(kpi, 0.0))) for kpi in ["rsrp", "rsrq", "sinr"] if float(weights.get(kpi, 0.0)) > 0.0]
    if not active:
        return summary
    max_weight = max(weight for _, weight in active)
    priority_kpis = [kpi for kpi, weight in active if weight == max_weight]

    if len(priority_kpis) == 1:
        kpi = priority_kpis[0]
        bad_col = f"Bad {kpi.upper()}"
        sev_col = f"{kpi}_bad_severity"
        priority = summary.loc[pd.to_numeric(summary.get(bad_col), errors="coerce").fillna(0) > 0].copy()
        if priority.empty:
            return priority
        priority["_selection_kpi"] = kpi.upper()
        priority["_selection_weight"] = float(max_weight)
        priority = priority.sort_values(
            [bad_col, sev_col, "Bad Grid Count", "combined_grid_severity", "Cell ID"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
        priority["_selection_rank"] = np.arange(1, len(priority) + 1)
        return priority

    selected: list[pd.DataFrame] = []
    selected_cells: set[str] = set()
    remaining_slots = total_cells
    for index, (kpi, weight) in enumerate(active):
        if remaining_slots <= 0:
            break
        budget = remaining_slots if index == len(active) - 1 else int(round(total_cells * weight))
        budget = max(1, min(budget, remaining_slots))
        bad_col = f"Bad {kpi.upper()}"
        sev_col = f"{kpi}_bad_severity"
        candidates = summary.loc[
            (~summary["Cell ID"].astype(str).isin(selected_cells))
            & (pd.to_numeric(summary.get(bad_col), errors="coerce").fillna(0) > 0)
        ].copy()
        if candidates.empty:
            continue
        candidates["_selection_kpi"] = kpi.upper()
        candidates["_selection_weight"] = float(weight)
        candidates = candidates.sort_values(
            [bad_col, sev_col, "Bad Grid Count", "combined_grid_severity", "Cell ID"],
            ascending=[False, False, False, False, True],
        ).head(budget)
        selected.append(candidates)
        selected_cells.update(candidates["Cell ID"].astype(str).tolist())
        remaining_slots = total_cells - len(selected_cells)

    if len(selected_cells) < total_cells:
        fill = summary.loc[~summary["Cell ID"].astype(str).isin(selected_cells)].copy()
        if not fill.empty:
            fill["_selection_kpi"] = "COMBINED_FILL"
            fill["_selection_weight"] = 0.0
            fill = fill.sort_values(
                ["combined_grid_severity", "Bad Grid Count", "Bad Samples", "Cell ID"],
                ascending=[False, False, False, True],
            )
            selected.append(fill)

    if not selected:
        return summary
    out = pd.concat(selected, ignore_index=True)
    out["_selection_rank"] = np.arange(1, len(out) + 1)
    return out.drop_duplicates(subset=["Cell ID"], keep="first").reset_index(drop=True)


def _select_coordinate_target_cells(
    baseline_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    grid_analytics_df: pd.DataFrame,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
    config: CandidateValidationConfig,
) -> list[str]:
    baseline = _ensure_node_cell(baseline_df)
    baseline = _attach_grid_context(baseline, baseline)
    baseline = _attach_grid_context_from_analytics(baseline, grid_analytics_df)
    mapped_rows = int(normalize_grid_id_series(baseline.get("grid_id")).notna().sum()) if "grid_id" in baseline.columns else 0
    baseline_ids = set(baseline.get("Node_Cell_ID", pd.Series(dtype=str)).astype(str).str.strip().tolist()) if "Node_Cell_ID" in baseline.columns else set()
    print(
        "[TILT_TARGET_SELECTION_INPUT] "
        f"baseline_rows={len(baseline)} grid_mapped_rows={mapped_rows} "
        f"grid_mapped_pct={mapped_rows / max(len(baseline), 1) * 100.0:.2f} "
        f"baseline_node_cell_ids={len(baseline_ids)}"
    )
    if baseline.empty or "grid_id" not in baseline.columns or "Node_Cell_ID" not in baseline.columns:
        print("[TILT_TARGET_SELECTION_EMPTY] reason=missing_baseline_grid_or_node_cell")
        return []
    grid = _combined_grid_reference(baseline, grid_analytics_df, thresholds, weights)
    if grid.empty:
        print("[TILT_TARGET_SELECTION_EMPTY] reason=empty_grid_reference")
        return []
    bad_grid_ids = set(grid.loc[grid["is_bad_combined"].fillna(False), "grid_id"].astype(str).tolist())
    if not bad_grid_ids:
        print("[TILT_TARGET_SELECTION_EMPTY] reason=no_bad_grids_in_grid_reference")
        return []
    ant = opt_ml._normalize_site_df(_prepare_optimizer_site_df(antenna_df), log_stage="TILT_COORDINATE_TARGETS")
    tunable = set()
    for alias_col in [col for col in _MATCH_ALIAS_COLS if col in ant.columns]:
        if alias_col in ant.columns:
            tunable.update(ant[alias_col].map(_clean_id).astype(str).str.strip().tolist())
    for node_col in ["nodeb_id", "node_b_id", "dashboard_site_id", "site_id"]:
        if node_col not in ant.columns:
            continue
        for cell_col in ["local_cell_id", "cell_id"]:
            if cell_col in ant.columns:
                tunable.update((ant[node_col].map(_clean_id) + "_" + ant[cell_col].map(_clean_id)).astype(str).str.strip("_").tolist())
    tunable = {cell for cell in tunable if cell and cell not in {"nan", "None", "<NA>"}}
    work = baseline.copy()
    work["grid_id"] = normalize_grid_id_series(work["grid_id"])
    bad_work = work.loc[work["grid_id"].astype(str).isin(bad_grid_ids)].copy()
    bad_node_ids = set(bad_work["Node_Cell_ID"].astype(str).str.strip().tolist())
    overlap_ids = bad_node_ids & tunable
    overlap_rows = int(bad_work["Node_Cell_ID"].astype(str).str.strip().isin(tunable).sum())
    missing_overlap_ids = bad_node_ids - tunable
    match_pct = len(overlap_ids) / max(len(bad_node_ids), 1) * 100.0
    print(
        "[TILT_TARGET_SELECTION_OVERLAP] "
        f"bad_grid_count={len(bad_grid_ids)} bad_grid_rows={len(bad_work)} "
        f"bad_grid_node_cell_ids={len(bad_node_ids)} antenna_tunable_ids={len(tunable)} "
        f"overlap_ids={len(overlap_ids)} overlap_rows={overlap_rows} "
        f"sample_bad_ids={sorted(list(bad_node_ids))[:5]} "
        f"sample_tunable_ids={sorted(list(tunable))[:5]}"
    )
    print(
        "[TILT_TARGET_SELECTION_EXACT_UPDATEABLE] "
        f"source_bad_ids={len(bad_node_ids)} exact_updateable_ids={len(overlap_ids)} "
        f"source_alias_only_ids={len(missing_overlap_ids)} exact_match_pct={match_pct:.2f} "
        "mode=good_ml_exact_antenna_key "
        f"sample_alias_only_ids={sorted(list(missing_overlap_ids))[:10]}"
    )
    work = bad_work.loc[bad_work["Node_Cell_ID"].astype(str).str.strip().isin(tunable)].copy()
    if work.empty:
        print("[TILT_TARGET_SELECTION_EMPTY] reason=no_bad_grid_rows_matching_tunable_antenna_ids")
        return []
    work["Cell ID"] = work["Node_Cell_ID"].astype(str).str.strip()
    work["_combined_severity"] = pd.Series(0.0, index=work.index)
    bad_cols: list[str] = []
    severity_cols: list[str] = []
    for kpi in ["rsrp", "rsrq", "sinr"]:
        if float(weights.get(kpi, 0.0)) <= 0.0 or f"pred_{kpi}" not in work.columns:
            continue
        bad_col = f"Bad {kpi.upper()}"
        sev_col = f"{kpi}_bad_severity"
        work[sev_col] = _severity(work[f"pred_{kpi}"], thresholds[kpi])
        work[bad_col] = work[sev_col] > 0
        work["_combined_severity"] += work[sev_col] * float(weights[kpi])
        bad_cols.append(bad_col)
        severity_cols.append(sev_col)
    work = work.loc[work["_combined_severity"] > 0].copy()
    if work.empty:
        print("[TILT_TARGET_SELECTION_EMPTY] reason=no_positive_combined_severity_after_kpi_thresholds")
        return []
    grouped = (
        work.groupby("Cell ID", dropna=False)
        .agg(
            **{
                "Bad RSRP": ("Bad RSRP", "sum") if "Bad RSRP" in bad_cols else ("_combined_severity", "size"),
                "Bad RSRQ": ("Bad RSRQ", "sum") if "Bad RSRQ" in bad_cols else ("_combined_severity", "size"),
                "Bad SINR": ("Bad SINR", "sum") if "Bad SINR" in bad_cols else ("_combined_severity", "size"),
                "rsrp_bad_severity": ("rsrp_bad_severity", "sum") if "rsrp_bad_severity" in severity_cols else ("_combined_severity", "sum"),
                "rsrq_bad_severity": ("rsrq_bad_severity", "sum") if "rsrq_bad_severity" in severity_cols else ("_combined_severity", "sum"),
                "sinr_bad_severity": ("sinr_bad_severity", "sum") if "sinr_bad_severity" in severity_cols else ("_combined_severity", "sum"),
                "combined_grid_severity": ("_combined_severity", "sum"),
                "Bad Grid Count": ("grid_id", "nunique"),
                "Bad Samples": ("_combined_severity", "count"),
            }
        )
        .reset_index()
    )
    for col in ["Bad RSRP", "Bad RSRQ", "Bad SINR", "rsrp_bad_severity", "rsrq_bad_severity", "sinr_bad_severity"]:
        if col not in grouped.columns:
            grouped[col] = 0.0
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0)
    grouped = _weighted_kpi_cell_order(grouped, weights)
    if grouped.empty:
        print("[TILT_TARGET_SELECTION_EMPTY] reason=no_weighted_kpi_target_cells")
        return []
    contribution_col = "combined_grid_severity"
    priority_kpis = [
        kpi for kpi in ["rsrp", "rsrq", "sinr"]
        if float(weights.get(kpi, 0.0)) == max(float(weights.get("rsrp", 0.0)), float(weights.get("rsrq", 0.0)), float(weights.get("sinr", 0.0)))
        and float(weights.get(kpi, 0.0)) > 0.0
    ]
    if len(priority_kpis) == 1:
        contribution_col = f"{priority_kpis[0]}_bad_severity"
    grouped["total_severity"] = pd.to_numeric(grouped[contribution_col], errors="coerce").fillna(0.0)
    total = float(grouped["total_severity"].sum())
    grouped["contribution_pct"] = grouped["total_severity"] / max(total, 1e-9) * 100.0
    grouped["cumulative_pct"] = grouped["contribution_pct"].cumsum()
    coverage = float(np.clip(float(config.bad_grid_coverage_pct), 1.0, 100.0))
    selected = grouped.loc[grouped["cumulative_pct"] <= coverage].copy()
    if selected.empty:
        selected = grouped.head(1).copy()
    elif len(selected) < len(grouped):
        selected = pd.concat([selected, grouped.iloc[[len(selected)]]], ignore_index=True)
    if int(config.max_group_cells or 0) > 0:
        selected = selected.head(int(config.max_group_cells)).copy()
    cells = selected["Cell ID"].astype(str).str.strip().tolist()
    print(
        f"[TILT_COORDINATE_TARGETS] bad_grids={len(bad_grid_ids)} contributor_cells={len(grouped)} "
        f"selected_cells={len(cells)} coverage_target={coverage:.1f} "
        f"selection=weighted_kpi_priority contribution={contribution_col} cells={cells}"
    )
    print(
        "[TILT_TARGET_SELECTION_FINAL] "
        f"contributors_after_match={len(grouped)} selected_for_coordinate_search={len(cells)} "
        f"coverage_target_pct={coverage:.1f} max_group_cells={int(config.max_group_cells or 0)}"
    )
    return cells


def _build_recompute_cells(
    antenna_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    update_cells: list[str],
    max_neighbors_per_update_cell: int,
    antenna_work_df: Optional[pd.DataFrame] = None,
) -> list[str]:
    update_set = {str(cell).strip() for cell in update_cells if str(cell).strip()}
    if not update_set:
        return []
    if antenna_work_df is not None and not antenna_work_df.empty:
        ant = antenna_work_df
    else:
        ant = opt_ml._normalize_site_df(_prepare_optimizer_site_df(antenna_df), log_stage="TILT_COORDINATE_RECOMPUTE")
    recompute = set(update_set)
    if {"Node_Cell_ID", "dashboard_site_id"}.issubset(ant.columns):
        update_mask = pd.Series(False, index=ant.index)
        for cell in update_set:
            update_mask = update_mask | _identity_match_mask(ant, cell)
        sites = set(ant.loc[update_mask, "dashboard_site_id"].astype(str).tolist())
        recompute.update(ant.loc[ant["dashboard_site_id"].astype(str).isin(sites), "Node_Cell_ID"].astype(str).tolist())
    topology_cols = [col for col in ["best_interferer_cell_id", "neighbor_1_cell_id", "neighbor_2_cell_id"] if col in baseline_df.columns]
    neighbor_counts: Dict[str, int] = {}
    if topology_cols and "Node_Cell_ID" in baseline_df.columns:
        focus = baseline_df.loc[baseline_df["Node_Cell_ID"].astype(str).isin(update_set)]
        for col in topology_cols:
            for value in focus[col].dropna().astype(str):
                cell = _clean_id(value)
                if cell and cell not in recompute:
                    neighbor_counts[cell] = neighbor_counts.get(cell, 0) + 1
    max_neighbors = max(0, int(max_neighbors_per_update_cell)) * max(1, len(update_set))
    recompute.update([cell for cell, _ in sorted(neighbor_counts.items(), key=lambda item: (-item[1], item[0]))[:max_neighbors]])
    return sorted(cell for cell in recompute if bool(_identity_match_mask(ant, cell).any()))


def _make_etilt_update(
    antenna_df: pd.DataFrame,
    cell_id: str,
    delta: float,
    config: CandidateValidationConfig,
    antenna_work_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, object]]:
    if antenna_work_df is not None and not antenna_work_df.empty:
        ant = antenna_work_df
    else:
        ant = opt_ml._normalize_site_df(_prepare_optimizer_site_df(antenna_df), log_stage="TILT_COORDINATE_UPDATE")
    row = ant.loc[_identity_match_mask(ant, cell_id)]
    if row.empty:
        return None
    current = pd.to_numeric(pd.Series([row.iloc[0].get("electrical_tilt")]), errors="coerce").iloc[0]
    if pd.isna(current):
        return None
    target = float(current) + float(delta)
    constraint = (config.constraint_map or {}).get(_threshold_cell_id(cell_id))
    if constraint and bool(constraint.get("optimised")):
        min_allowed = pd.to_numeric(pd.Series([constraint.get("min_e_tilt")]), errors="coerce").iloc[0]
        max_allowed = pd.to_numeric(pd.Series([constraint.get("max_e_tilt")]), errors="coerce").iloc[0]
        if pd.notna(min_allowed) and pd.notna(max_allowed):
            target = float(np.clip(target, float(min_allowed), float(max_allowed)))
    if np.isclose(target, float(current), equal_nan=True):
        return None
    return {
        "cell_id": str(cell_id).strip(),
        "parameter": "ETilt",
        "current_value": float(current),
        "target_value": float(target),
        "requested_delta": float(delta),
        "actual_delta": float(target) - float(current),
    }


def _azimuth_in_range(value: float, min_az: float, max_az: float) -> bool:
    value_norm = float(value) % 360.0
    min_norm = float(min_az) % 360.0
    max_norm = float(max_az) % 360.0
    if min_norm <= max_norm:
        return min_norm <= value_norm <= max_norm
    return value_norm >= min_norm or value_norm <= max_norm


def _clamp_azimuth_to_range(value: float, min_az: float, max_az: float) -> float:
    value_norm = float(value) % 360.0
    if _azimuth_in_range(value_norm, min_az, max_az):
        return value_norm
    min_norm = float(min_az) % 360.0
    max_norm = float(max_az) % 360.0
    dist_to_min = abs(((value_norm - min_norm + 180.0) % 360.0) - 180.0)
    dist_to_max = abs(((value_norm - max_norm + 180.0) % 360.0) - 180.0)
    return min_norm if dist_to_min <= dist_to_max else max_norm


def _bearing_context_map(
    baseline_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, object]]:
    try:
        bearing_df = compute_dominant_bearing_summary(
            baseline_df,
            antenna_df,
            float(thresholds.get("rsrp", -105.0)),
            float(thresholds.get("rsrq", -15.0)),
            float(thresholds.get("sinr", 0.0)),
        )
    except Exception as exc:
        print(f"[TILT_AZIMUTH_FALLBACK] bearing_context_failed reason={exc}")
        return {}
    if bearing_df.empty or "Cell ID" not in bearing_df.columns:
        return {}
    work = bearing_df.copy()
    work["_cell_key"] = work["Cell ID"].map(canonical_cell_id)
    return {
        str(row["_cell_key"]): row.drop(labels=["_cell_key"]).to_dict()
        for _, row in work.iterrows()
        if str(row.get("_cell_key", "")).strip()
    }


def _make_azimuth_update(
    antenna_df: pd.DataFrame,
    cell_id: str,
    step_deg: float,
    config: CandidateValidationConfig,
    bearing_map: Dict[str, Dict[str, object]],
    antenna_work_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, object]]:
    if antenna_work_df is not None and not antenna_work_df.empty:
        ant = antenna_work_df
    else:
        ant = opt_ml._normalize_site_df(_prepare_optimizer_site_df(antenna_df), log_stage="TILT_COORDINATE_AZIMUTH_UPDATE")
    cell_key = str(cell_id).strip()
    row = ant.loc[_identity_match_mask(ant, cell_key)]
    if row.empty:
        suffix = cell_key.split("_")[-1]
        row = ant.loc[ant.get("cell_id", pd.Series("", index=ant.index)).map(_clean_id) == suffix]
    if row.empty:
        return None

    current = pd.to_numeric(pd.Series([row.iloc[0].get("azimuth")]), errors="coerce").iloc[0]
    if pd.isna(current):
        return None
    current = _normalize_azimuth(float(current))
    if pd.isna(current):
        return None

    context = bearing_map.get(canonical_cell_id(cell_key)) or bearing_map.get(cell_key) or {}
    dominant_bearing = pd.to_numeric(pd.Series([context.get("dominant_bearing_deg")]), errors="coerce").iloc[0]
    bearing_samples = pd.to_numeric(pd.Series([context.get("bearing_sample_count")]), errors="coerce").fillna(0).iloc[0]
    if pd.isna(dominant_bearing) or int(bearing_samples) < int(MIN_BEARING_SAMPLE_COUNT):
        return None

    signed_delta = _signed_azimuth_delta(float(dominant_bearing), float(current))
    if pd.isna(signed_delta) or abs(float(signed_delta)) < 5.0:
        return None
    direction = 1.0 if float(signed_delta) > 0 else -1.0
    requested_delta = direction * abs(float(step_deg))
    if abs(requested_delta) > abs(float(signed_delta)):
        requested_delta = float(signed_delta)
    target = _normalize_azimuth(float(current) + float(requested_delta))

    constraint = (config.constraint_map or {}).get(_threshold_cell_id(cell_key))
    if constraint and bool(constraint.get("optimised")):
        min_allowed = pd.to_numeric(pd.Series([constraint.get("min_azimuth")]), errors="coerce").iloc[0]
        max_allowed = pd.to_numeric(pd.Series([constraint.get("max_azimuth")]), errors="coerce").iloc[0]
        if pd.notna(min_allowed) and pd.notna(max_allowed):
            target = _clamp_azimuth_to_range(float(target), float(min_allowed), float(max_allowed))

    actual_delta = _signed_azimuth_delta(float(target), float(current))
    if pd.isna(actual_delta) or abs(float(actual_delta)) < 0.5:
        return None

    return {
        "cell_id": cell_key,
        "parameter": "Azimuth",
        "current_value": float(current),
        "target_value": float(target),
        "requested_delta": float(requested_delta),
        "actual_delta": float(actual_delta),
        "dominant_bearing_deg": float(dominant_bearing),
        "bearing_sample_count": int(bearing_samples),
        "bearing_mismatch_deg": _safe_float(context.get("bearing_mismatch_deg"), np.nan),
        "bearing_peak_share": _safe_float(context.get("bearing_peak_share"), np.nan),
        "bearing_spread_deg": _safe_float(context.get("bearing_spread_deg"), np.nan),
        "bearing_directional_contrast": _safe_float(context.get("bearing_directional_contrast"), np.nan),
    }


def _rf_base_site_df(antenna_work: pd.DataFrame) -> pd.DataFrame:
    base_site = antenna_work.copy()
    for col in ["lat", "lon", "azimuth", "electrical_tilt", "mechanical_tilt", "tx_power", "antenna_height"]:
        if col in base_site.columns:
            base_site[f"orig_{col}"] = pd.to_numeric(base_site[col], errors="coerce")
    return base_site


def _rf_prediction_params(
    *,
    config: CandidateValidationConfig,
    baseline_work: pd.DataFrame,
    geo_df: pd.DataFrame,
    recompute_cells: list[str],
) -> Dict[str, object]:
    return {
        "project_id": int(config.project_id),
        "region": config.region,
        "radius": float(config.radius_m),
        "grid_resolution": float(config.grid_resolution_m),
        "n_workers": int(config.workers),
        "impact_radius_m": float(config.impact_radius_m),
        "neighbor_site_count": int(config.neighbor_site_count),
        "max_interference_sites": int(config.max_interference_sites),
        "baseline_job_id": config.baseline_job_id,
        "prediction_points_df": baseline_work,
        "geo_features_df": geo_df,
        "strict_prediction_points": True,
        "recompute_cells": recompute_cells,
    }


def _ensure_k1k2_for_recompute_cells(
    *,
    baseline_work: pd.DataFrame,
    antenna_work: pd.DataFrame,
    recompute_cells: list[str],
    k1k2_cache: Optional[Dict[str, tuple[float, float]]],
    seed_map: Optional[Dict[str, tuple[float, float]]] = None,
) -> Dict[str, tuple[float, float]]:
    k1k2_map: Dict[str, tuple[float, float]] = {}
    missing_k1k2_cells = []
    for cid in recompute_cells:
        key = str(cid)
        cached = seed_map.get(key) if seed_map is not None else None
        if cached is None and k1k2_cache is not None:
            cached = k1k2_cache.get(key)
        if cached is None:
            missing_k1k2_cells.append(key)
        else:
            pair = (float(cached[0]), float(cached[1]))
            k1k2_map[key] = pair
            if k1k2_cache is not None:
                k1k2_cache[key] = pair
    if missing_k1k2_cells:
        computed_k1k2 = opt_ml.compute_k1k2_for_cells(baseline_work, antenna_work, missing_k1k2_cells)
        for cid, value in computed_k1k2.items():
            pair = (float(value[0]), float(value[1]))
            key = str(cid)
            k1k2_map[key] = pair
            if k1k2_cache is not None:
                k1k2_cache[key] = pair
    return k1k2_map


def _filter_identity_rows(df: pd.DataFrame, cell_ids: list[str]) -> pd.DataFrame:
    if df.empty or not cell_ids:
        return df.iloc[0:0].copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    mask = pd.Series(False, index=df.index)
    for cid in cell_ids:
        mask = mask | opt_ml._identity_match_mask(df, cid)
    out = df.loc[mask].copy()
    return out if not out.empty else df.iloc[0:0].copy()


def _evaluate_update_set(
    *,
    updates: list[Dict[str, object]],
    baseline_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    grid_analytics_df: pd.DataFrame,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
    config: CandidateValidationConfig,
    baseline_rf_cache: Dict[tuple, pd.DataFrame],
    k1k2_cache: Optional[Dict[str, tuple[float, float]]] = None,
    baseline_work_df: Optional[pd.DataFrame] = None,
    antenna_work_df: Optional[pd.DataFrame] = None,
    rf_prediction_points_df: Optional[pd.DataFrame] = None,
    precomputed_k1k2_map: Optional[Dict[str, tuple[float, float]]] = None,
    precomputed_baseline_rf: Optional[pd.DataFrame] = None,
) -> tuple[Dict[str, object], pd.DataFrame]:
    eval_start = time.perf_counter()
    baseline_prep_sec = 0.0
    k1k2_sec = 0.0
    baseline_rf_sec = 0.0
    candidate_rf_sec = 0.0
    delta_apply_sec = 0.0
    score_sec = 0.0
    baseline_rf_cache_hit = False
    baseline_prep_start = time.perf_counter()
    if baseline_work_df is not None and not baseline_work_df.empty:
        baseline_work = baseline_work_df.copy()
    else:
        baseline_work = _attach_grid_context(_ensure_node_cell(baseline_df), baseline_df)
        baseline_work = _attach_grid_context_from_analytics(baseline_work, grid_analytics_df)
    if antenna_work_df is not None and not antenna_work_df.empty:
        antenna_work = antenna_work_df.copy()
    else:
        antenna_work = opt_ml._normalize_site_df(_prepare_optimizer_site_df(antenna_df), log_stage="TILT_COORDINATE_RF_ANTENNA")
    baseline_prep_sec = time.perf_counter() - baseline_prep_start
    update_cells = [_clean_id(update.get("cell_id")) for update in updates if _clean_id(update.get("cell_id"))]
    if not update_cells:
        metrics = _score_before_after(baseline_work, baseline_work, grid_analytics_df, thresholds, weights, config)
        metrics.update({"candidate_name": "hold", "update_count": 0, "updates": "[]"})
        return metrics, baseline_work
    recompute_cells = _build_recompute_cells(
        antenna_df,
        baseline_work,
        update_cells,
        config.max_neighbors_per_update_cell,
        antenna_work_df=antenna_work,
    )
    if not recompute_cells:
        raise ValueError("coordinate_recompute_cells_empty")
    k1k2_start = time.perf_counter()
    with _rf_debug_capture(config):
        candidate_site = _apply_updates_to_site_df(antenna_work, updates)
        k1k2_map = _ensure_k1k2_for_recompute_cells(
            baseline_work=baseline_work,
            antenna_work=antenna_work,
            recompute_cells=recompute_cells,
            k1k2_cache=k1k2_cache,
            seed_map=precomputed_k1k2_map,
        )
    k1k2_sec = time.perf_counter() - k1k2_start
    if not k1k2_map:
        raise ValueError("coordinate_dynamic_k1k2_not_available")
    scoped_rf_points = (
        rf_prediction_points_df
        if isinstance(rf_prediction_points_df, pd.DataFrame) and not rf_prediction_points_df.empty
        else _filter_identity_rows(baseline_work, recompute_cells)
    )
    if scoped_rf_points.empty:
        scoped_rf_points = baseline_work
    scoped_geo_df = _filter_identity_rows(geo_df, recompute_cells)
    if scoped_geo_df.empty and isinstance(geo_df, pd.DataFrame) and not geo_df.empty:
        scoped_geo_df = geo_df
    params = _rf_prediction_params(
        config=config,
        baseline_work=scoped_rf_points,
        geo_df=scoped_geo_df,
        recompute_cells=recompute_cells,
    )
    cache_key = tuple(sorted(recompute_cells))
    baseline_rf = precomputed_baseline_rf.copy() if isinstance(precomputed_baseline_rf, pd.DataFrame) and not precomputed_baseline_rf.empty else baseline_rf_cache.get(cache_key)
    if baseline_rf is None:
        base_site = _rf_base_site_df(antenna_work)
        baseline_rf_start = time.perf_counter()
        with _rf_debug_capture(config):
            baseline_rf = opt_ml.run_prediction_only_optimized(base_site, k1k2_map, params)
        baseline_rf_sec = time.perf_counter() - baseline_rf_start
        baseline_rf_cache[cache_key] = baseline_rf
    else:
        baseline_rf_cache_hit = True
    candidate_rf_start = time.perf_counter()
    with _rf_debug_capture(config):
        candidate_rf = opt_ml.run_prediction_only_optimized(candidate_site, k1k2_map, params)
    candidate_rf_sec = time.perf_counter() - candidate_rf_start
    delta_apply_start = time.perf_counter()
    after_df, delta_metrics = _apply_rf_delta(baseline_work, baseline_rf, candidate_rf)
    after_df = _attach_grid_context(after_df, baseline_work)
    delta_apply_sec = time.perf_counter() - delta_apply_start
    score_start = time.perf_counter()
    metrics = _score_before_after(baseline_work, after_df, grid_analytics_df, thresholds, weights, config)
    score_sec = time.perf_counter() - score_start
    metrics.update(delta_metrics)
    total_eval_sec = time.perf_counter() - eval_start
    metrics.update(
        {
            "candidate_name": f"coordinate_active_{len(updates)}",
            "update_count": len(updates),
            "updates": str(updates),
            "recompute_cell_count": float(len(recompute_cells)),
            "changed_cells": ",".join(update_cells),
            "timing_total_sec": float(total_eval_sec),
            "timing_baseline_prep_sec": float(baseline_prep_sec),
            "timing_k1k2_sec": float(k1k2_sec),
            "timing_baseline_rf_sec": float(baseline_rf_sec),
            "timing_baseline_rf_cache_hit": float(1 if baseline_rf_cache_hit else 0),
            "timing_candidate_rf_sec": float(candidate_rf_sec),
            "timing_delta_apply_sec": float(delta_apply_sec),
            "timing_score_sec": float(score_sec),
        }
    )
    print(
        "[TILT_CANDIDATE_RESULT] "
        f"updates={len(updates)} changes={_format_updates(updates)} "
        f"score={float(metrics.get('score', -99999.0)):.4f} "
        f"net={float(metrics.get('net_bad_reduction', 0.0)):.0f} "
        f"before_bad={float(metrics.get('baseline_bad_grid_count', metrics.get('baseline_bad_count', 0.0))):.0f} "
        f"after_bad={float(metrics.get('candidate_bad_grid_count', metrics.get('candidate_bad_count', 0.0))):.0f} "
        f"good_area_loss_pct={float(metrics.get('good_area_loss_pct', 0.0)):.2f} "
        f"constraints_passed={int(float(metrics.get('constraints_passed', 0.0)) >= 1.0)} "
        f"rf_delta_rows={float(metrics.get('rf_delta_matched_row_count', 0.0)):.0f} "
        f"recompute_cells={len(recompute_cells)} "
        f"timing_sec=total:{total_eval_sec:.2f},prep:{baseline_prep_sec:.2f},"
        f"k1k2:{k1k2_sec:.2f},base_rf:{baseline_rf_sec:.2f},"
        f"cand_rf:{candidate_rf_sec:.2f},delta:{delta_apply_sec:.2f},score:{score_sec:.2f},"
        f"base_cache_hit={int(baseline_rf_cache_hit)}"
    )
    return metrics, after_df


def _evaluate_trial_process(payload: Dict[str, object]) -> tuple[Dict[str, object], pd.DataFrame]:
    config = payload["config"]
    if not isinstance(config, CandidateValidationConfig):
        raise ValueError("invalid_candidate_validation_config")
    worker_idx = int(payload.get("worker_idx", 0) or 0)
    if config.rf_debug_log_path:
        root, ext = os.path.splitext(str(config.rf_debug_log_path))
        config = CandidateValidationConfig(
            **{
                **config.__dict__,
                "rf_debug_log_path": f"{root}.candidate_{os.getpid()}_{worker_idx}{ext or '.log'}",
                "workers": int(payload.get("rf_workers", 1) or 1),
            }
        )
    else:
        config = CandidateValidationConfig(
            **{
                **config.__dict__,
                "workers": int(payload.get("rf_workers", 1) or 1),
            }
        )
    return _evaluate_update_set(
        updates=payload["updates"],
        baseline_df=payload["baseline_df"],
        antenna_df=payload["antenna_df"],
        geo_df=payload["geo_df"],
        grid_analytics_df=payload["grid_analytics_df"],
        thresholds=payload["thresholds"],
        weights=payload["weights"],
        config=config,
        baseline_rf_cache={},
        k1k2_cache={},
        baseline_work_df=payload.get("baseline_work_df"),
        antenna_work_df=payload.get("antenna_work_df"),
        rf_prediction_points_df=payload.get("rf_prediction_points_df"),
        precomputed_k1k2_map=payload.get("precomputed_k1k2_map"),
        precomputed_baseline_rf=payload.get("precomputed_baseline_rf"),
    )


def coordinate_search_recommendations(
    *,
    baseline_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    grid_analytics_df: pd.DataFrame,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
    config: CandidateValidationConfig,
) -> Dict[str, pd.DataFrame]:
    target_cells = _select_coordinate_target_cells(baseline_df, antenna_df, grid_analytics_df, thresholds, weights, config)
    if not target_cells:
        return {
            "recommendations": pd.DataFrame(),
            "candidate_evaluations": pd.DataFrame(),
            "best_candidate_after": pd.DataFrame(),
            "best_candidate_metrics": pd.DataFrame(),
        }

    baseline_rf_cache: Dict[tuple, pd.DataFrame] = {}
    k1k2_cache: Dict[str, tuple[float, float]] = {}
    prepared_baseline_work = _attach_grid_context(_ensure_node_cell(baseline_df), baseline_df)
    prepared_baseline_work = _attach_grid_context_from_analytics(prepared_baseline_work, grid_analytics_df)
    prepared_antenna_work = opt_ml._normalize_site_df(
        _prepare_optimizer_site_df(antenna_df),
        log_stage="TILT_COORDINATE_RF_ANTENNA_PREPARED",
    )
    bearing_map = _bearing_context_map(baseline_df, antenna_df, thresholds) if bool(config.enable_azimuth_fallback) else {}
    evaluation_lock = threading.Lock()
    seen_coordinate_keys: set[tuple] = {tuple()}
    evaluation_rows: list[Dict[str, object]] = []
    current_updates: Dict[str, Dict[str, object]] = {}
    current_metrics, current_after = _evaluate_update_set(
        updates=[],
        baseline_df=baseline_df,
        antenna_df=antenna_df,
        geo_df=geo_df,
        grid_analytics_df=grid_analytics_df,
        thresholds=thresholds,
        weights=weights,
        config=config,
        baseline_rf_cache=baseline_rf_cache,
        k1k2_cache=k1k2_cache,
        baseline_work_df=prepared_baseline_work,
        antenna_work_df=prepared_antenna_work,
    )
    evaluation_rows.append(current_metrics)

    def rank(metrics: Dict[str, object]) -> tuple:
        return (
            int(float(metrics.get("constraints_passed", 0.0)) >= 1.0),
            float(metrics.get("score", -99999.0)),
            float(metrics.get("net_bad_reduction", -99999.0)),
            float(metrics.get("combined_weighted_tie_break", -99999.0)),
            -float(metrics.get("new_bad_samples", metrics.get("good_to_bad_grid_count", 99999.0))),
        )

    def candidate_key(updates: list[Dict[str, object]]) -> tuple:
        return tuple(
            sorted(
                (
                    _clean_id(update.get("cell_id")),
                    str(update.get("parameter", "")).strip(),
                    round(float(update.get("target_value", np.nan)), 6),
                )
                for update in updates
            )
        )

    def ordered_updates(state: Dict[str, Dict[str, object]]) -> list[Dict[str, object]]:
        return [dict(state[cell]) for cell in target_cells if cell in state]

    def evaluate_trial(
        *,
        pass_idx: int,
        cell_id: str,
        delta: float,
        state: Dict[str, Dict[str, object]],
        stage_name: str,
        parameter: str = "ETilt",
    ) -> Optional[tuple[Dict[str, Dict[str, object]], Dict[str, object], pd.DataFrame]]:
        if str(parameter).strip().lower() == "azimuth":
            update = _make_azimuth_update(
                antenna_df,
                cell_id,
                delta,
                config,
                bearing_map,
                antenna_work_df=prepared_antenna_work,
            )
        else:
            update = _make_etilt_update(antenna_df, cell_id, delta, config, antenna_work_df=prepared_antenna_work)
        if update is None:
            return None
        trial_state = dict(state)
        trial_state[cell_id] = update
        trial_updates = ordered_updates(trial_state)
        key = candidate_key(trial_updates)
        with evaluation_lock:
            if key in seen_coordinate_keys:
                return None
            seen_coordinate_keys.add(key)
        try:
            metrics, after_df = _evaluate_update_set(
                updates=trial_updates,
                baseline_df=baseline_df,
                antenna_df=antenna_df,
                geo_df=geo_df,
                grid_analytics_df=grid_analytics_df,
                thresholds=thresholds,
                weights=weights,
                config=config,
                baseline_rf_cache=baseline_rf_cache,
                k1k2_cache=k1k2_cache,
                baseline_work_df=prepared_baseline_work,
                antenna_work_df=prepared_antenna_work,
            )
        except Exception as exc:
            metrics, after_df = {
                "score": -99999.0,
                "constraints_passed": 0.0,
                "candidate_name": f"coordinate_{stage_name}_{parameter}_error_{cell_id}_{delta}",
                "error": str(exc),
            }, pd.DataFrame()
        metrics["candidate_name"] = f"coord_pass_{pass_idx}_{parameter.lower()}_cell_{cell_id}_delta_{delta:+.1f}_active_{len(trial_updates)}"
        return trial_state, metrics, after_df

    def evaluate_delta_stage(
        *,
        pass_idx: int,
        cell_id: str,
        deltas: list[float],
        state: Dict[str, Dict[str, object]],
        stage_name: str,
        parameter: str = "ETilt",
    ) -> list[tuple[float, Dict[str, Dict[str, object]], Dict[str, object], pd.DataFrame]]:
        valid_deltas = [float(delta) for delta in deltas]
        if not valid_deltas:
            return []
        candidate_workers = max(1, int(config.candidate_workers or 1))
        print(
            f"[TILT_COORDINATE_STAGE] pass={pass_idx} cell={cell_id} stage={stage_name} "
            f"parameter={parameter} deltas={valid_deltas} candidate_workers={min(candidate_workers, len(valid_deltas))}"
        )
        stage_results: list[tuple[float, Dict[str, Dict[str, object]], Dict[str, object], pd.DataFrame]] = []
        if candidate_workers <= 1 or len(valid_deltas) == 1:
            for delta in valid_deltas:
                evaluated = evaluate_trial(
                    pass_idx=pass_idx,
                    cell_id=cell_id,
                    delta=delta,
                    state=state,
                    stage_name=stage_name,
                    parameter=parameter,
                )
                if evaluated is None:
                    continue
                trial_state, metrics, after_df = evaluated
                evaluation_rows.append(metrics)
                stage_results.append((delta, trial_state, metrics, after_df))
            return stage_results
        process_jobs: list[tuple[float, Dict[str, Dict[str, object]], list[Dict[str, object]], str, tuple]] = []
        for delta in valid_deltas:
            if str(parameter).strip().lower() == "azimuth":
                update = _make_azimuth_update(
                    antenna_df,
                    cell_id,
                    delta,
                    config,
                    bearing_map,
                    antenna_work_df=prepared_antenna_work,
                )
            else:
                update = _make_etilt_update(antenna_df, cell_id, delta, config, antenna_work_df=prepared_antenna_work)
            if update is None:
                continue
            trial_state = dict(state)
            trial_state[cell_id] = update
            trial_updates = ordered_updates(trial_state)
            key = candidate_key(trial_updates)
            with evaluation_lock:
                if key in seen_coordinate_keys:
                    continue
                seen_coordinate_keys.add(key)
            update_cells = [_clean_id(update.get("cell_id")) for update in trial_updates if _clean_id(update.get("cell_id"))]
            recompute_cells = _build_recompute_cells(
                antenna_df,
                prepared_baseline_work,
                update_cells,
                config.max_neighbors_per_update_cell,
                antenna_work_df=prepared_antenna_work,
            )
            if not recompute_cells:
                continue
            cache_key = tuple(sorted(recompute_cells))
            candidate_name = f"coord_pass_{pass_idx}_{parameter.lower()}_cell_{cell_id}_delta_{delta:+.1f}_active_{len(trial_updates)}"
            process_jobs.append((delta, trial_state, trial_updates, candidate_name, cache_key))
        if not process_jobs:
            return []
        rf_workers_per_candidate = 1
        process_context_config = CandidateValidationConfig(
            **{
                **config.__dict__,
                "workers": rf_workers_per_candidate,
            }
        )
        process_context_by_key: Dict[tuple, Dict[str, object]] = {}
        for _, _, _, _, cache_key in process_jobs:
            if cache_key in process_context_by_key:
                continue
            recompute_cells = list(cache_key)
            scoped_baseline_work = _filter_identity_rows(prepared_baseline_work, recompute_cells)
            scoped_geo_df = _filter_identity_rows(geo_df, recompute_cells)
            k1k2_start = time.perf_counter()
            with _rf_debug_capture(config):
                k1k2_map = _ensure_k1k2_for_recompute_cells(
                    baseline_work=prepared_baseline_work,
                    antenna_work=prepared_antenna_work,
                    recompute_cells=recompute_cells,
                    k1k2_cache=k1k2_cache,
                )
            k1k2_sec = time.perf_counter() - k1k2_start
            baseline_rf = baseline_rf_cache.get(cache_key)
            baseline_rf_cache_hit = baseline_rf is not None
            baseline_rf_sec = 0.0
            if baseline_rf is None:
                parent_params = _rf_prediction_params(
                    config=process_context_config,
                    baseline_work=scoped_baseline_work,
                    geo_df=scoped_geo_df,
                    recompute_cells=recompute_cells,
                )
                baseline_rf_start = time.perf_counter()
                with _rf_debug_capture(config):
                    baseline_rf = opt_ml.run_prediction_only_optimized(
                        _rf_base_site_df(prepared_antenna_work),
                        k1k2_map,
                        parent_params,
                    )
                baseline_rf_sec = time.perf_counter() - baseline_rf_start
                baseline_rf_cache[cache_key] = baseline_rf
            process_context_by_key[cache_key] = {
                "recompute_cells": recompute_cells,
                "k1k2_map": k1k2_map,
                "baseline_rf": baseline_rf,
                "rf_prediction_points_df": scoped_baseline_work,
                "geo_df": scoped_geo_df,
            }
            print(
                "[TILT_COORDINATE_PROCESS_CONTEXT] "
                f"pass={pass_idx} cell={cell_id} stage={stage_name} "
                f"recompute_cells={len(recompute_cells)} "
                f"k1k2_cells={len(k1k2_map)} k1k2_sec={k1k2_sec:.2f} "
                f"base_cache_hit={int(baseline_rf_cache_hit)} base_rf_sec={baseline_rf_sec:.2f}"
            )
        max_workers = min(candidate_workers, len(process_jobs))
        print(
            f"[TILT_COORDINATE_PROCESS_POOL] pass={pass_idx} cell={cell_id} stage={stage_name} "
            f"parameter={parameter} process_workers={max_workers} rf_workers_per_candidate={rf_workers_per_candidate}"
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(
                    _evaluate_trial_process,
                    {
                        "updates": trial_updates,
                        "baseline_df": pd.DataFrame(),
                        "antenna_df": pd.DataFrame(),
                        "geo_df": process_context_by_key.get(cache_key, {}).get("geo_df", pd.DataFrame()),
                        "grid_analytics_df": grid_analytics_df,
                        "baseline_work_df": prepared_baseline_work,
                        "antenna_work_df": prepared_antenna_work,
                        "rf_prediction_points_df": process_context_by_key.get(cache_key, {}).get("rf_prediction_points_df", pd.DataFrame()),
                        "thresholds": thresholds,
                        "weights": weights,
                        "config": config,
                        "precomputed_k1k2_map": process_context_by_key.get(cache_key, {}).get("k1k2_map", {}),
                        "precomputed_baseline_rf": process_context_by_key.get(cache_key, {}).get("baseline_rf", pd.DataFrame()),
                        "rf_workers": rf_workers_per_candidate,
                        "worker_idx": idx,
                    },
                ): (delta, trial_state, candidate_name)
                for idx, (delta, trial_state, trial_updates, candidate_name, cache_key) in enumerate(process_jobs, start=1)
            }
            for future in as_completed(future_to_job):
                delta, trial_state, candidate_name = future_to_job[future]
                try:
                    metrics, after_df = future.result()
                except Exception as exc:
                    metrics, after_df = {
                        "score": -99999.0,
                        "constraints_passed": 0.0,
                        "candidate_name": f"coordinate_{stage_name}_{parameter}_error_{cell_id}_{delta}",
                        "error": str(exc),
                    }, pd.DataFrame()
                metrics["candidate_name"] = candidate_name
                with evaluation_lock:
                    evaluation_rows.append(metrics)
                stage_results.append((delta, trial_state, metrics, after_df))
        return sorted(stage_results, key=lambda item: valid_deltas.index(item[0]))

    first_probe, directional = _etilt_candidate_delta_sets(config)
    print(
        f"[TILT_CANDIDATE_CONFIG] etilt_first_probe={first_probe} "
        f"etilt_directional_plus={directional['plus']} "
        f"etilt_directional_minus={directional['minus']}"
    )
    for pass_idx in range(1, max(1, int(config.coordinate_passes)) + 1):
        changed = False
        print(f"[TILT_COORDINATE_PASS] pass={pass_idx} active_updates={len(current_updates)} score={float(current_metrics.get('score', 0.0)):.4f}")
        for cell_idx, cell_id in enumerate(target_cells, start=1):
            print(
                f"[TILT_COORDINATE_PROGRESS] pass={pass_idx} "
                f"cell_index={cell_idx}/{len(target_cells)} cell={cell_id} "
                f"completed={cell_idx - 1} remaining={len(target_cells) - cell_idx} "
                f"active_updates={len(current_updates)}"
            )
            best_updates = dict(current_updates)
            best_metrics = current_metrics
            best_after = current_after
            best_probe_delta = None
            for delta, trial_updates, metrics, after_df in evaluate_delta_stage(
                pass_idx=pass_idx,
                cell_id=cell_id,
                deltas=first_probe,
                state=current_updates,
                stage_name="first_probe",
            ):
                if rank(metrics) > rank(best_metrics):
                    best_updates = trial_updates
                    best_metrics = metrics
                    best_after = after_df
                    best_probe_delta = delta
            if best_probe_delta is not None:
                direction_key = "plus" if best_probe_delta > 0 else "minus"
                if directional[direction_key]:
                    step_results = evaluate_delta_stage(
                        pass_idx=pass_idx,
                        cell_id=cell_id,
                        deltas=directional[direction_key],
                        state=current_updates,
                        stage_name="directional_step_batch",
                    )
                    for _, trial_updates, metrics, after_df in sorted(
                        step_results,
                        key=lambda item: rank(item[2]),
                        reverse=True,
                    ):
                        if rank(metrics) > rank(best_metrics):
                            best_updates = trial_updates
                            best_metrics = metrics
                            best_after = after_df
                            break
            if rank(best_metrics) > rank(current_metrics):
                current_updates = best_updates
                current_metrics = best_metrics
                current_after = best_after
                changed = True
                print(
                    f"[TILT_COORDINATE_KEEP] pass={pass_idx} cell={cell_id} active_updates={len(current_updates)} "
                    f"score={float(current_metrics.get('score', 0.0)):.4f} net={float(current_metrics.get('net_bad_reduction', 0.0)):.0f}"
                )
        if not changed:
            print(f"[TILT_COORDINATE_STOP] pass={pass_idx} reason=no_improvement")
            break

    if bool(config.enable_azimuth_fallback) and bearing_map:
        remaining_cells = [cell_id for cell_id in target_cells if cell_id not in current_updates]
        azimuth_steps = _azimuth_fallback_steps(config)
        print(
            f"[TILT_AZIMUTH_FALLBACK_START] remaining_cells={len(remaining_cells)} "
            f"steps={azimuth_steps} active_etilt_updates={len(current_updates)}"
        )
        for cell_idx, cell_id in enumerate(remaining_cells, start=1):
            print(
                f"[TILT_AZIMUTH_FALLBACK_PROGRESS] cell_index={cell_idx}/{len(remaining_cells)} "
                f"cell={cell_id} active_updates={len(current_updates)}"
            )
            best_updates = dict(current_updates)
            best_metrics = current_metrics
            best_after = current_after
            fallback_results = evaluate_delta_stage(
                pass_idx=max(1, int(config.coordinate_passes)) + 1,
                cell_id=cell_id,
                deltas=azimuth_steps,
                state=current_updates,
                stage_name="azimuth_directional_fallback",
                parameter="Azimuth",
            )
            for _, trial_updates, metrics, after_df in sorted(
                fallback_results,
                key=lambda item: rank(item[2]),
                reverse=True,
            ):
                if rank(metrics) > rank(best_metrics):
                    best_updates = trial_updates
                    best_metrics = metrics
                    best_after = after_df
                    break
            if rank(best_metrics) > rank(current_metrics):
                current_updates = best_updates
                current_metrics = best_metrics
                current_after = best_after
                accepted = current_updates.get(cell_id, {})
                print(
                    f"[TILT_AZIMUTH_FALLBACK_KEEP] cell={cell_id} "
                    f"azimuth={_safe_float(accepted.get('current_value'), np.nan):.1f}->"
                    f"{_safe_float(accepted.get('target_value'), np.nan):.1f} "
                    f"score={float(current_metrics.get('score', 0.0)):.4f} "
                    f"net={float(current_metrics.get('net_bad_reduction', 0.0)):.0f}"
                )
        print(f"[TILT_AZIMUTH_FALLBACK_DONE] active_updates={len(current_updates)}")
    elif bool(config.enable_azimuth_fallback):
        print("[TILT_AZIMUTH_FALLBACK_SKIP] reason=no_bearing_context")

    evaluation_df = pd.DataFrame(evaluation_rows)
    actionable = (
        current_updates
        and float(current_metrics.get("constraints_passed", 0.0)) >= 1.0
        and (
            float(current_metrics.get("score", -99999.0)) > 0.0
            or float(current_metrics.get("net_bad_reduction", 0.0)) > 0.0
            or float(current_metrics.get("bad_to_good_grid_count", 0.0)) > float(current_metrics.get("good_to_bad_grid_count", 0.0))
        )
    )
    if not actionable:
        return {
            "recommendations": pd.DataFrame(),
            "candidate_evaluations": evaluation_df,
            "best_candidate_after": current_after,
            "best_candidate_metrics": pd.DataFrame([current_metrics]),
        }

    ant = opt_ml._normalize_site_df(_prepare_optimizer_site_df(antenna_df), log_stage="TILT_COORDINATE_EXPORT")
    rows = []
    for update in current_updates.values():
        rf_cell_id = _clean_id(update.get("cell_id"))
        cell_id = rf_cell_id
        ant_row = ant.loc[_identity_match_mask(ant, rf_cell_id)]
        tech = ant_row.iloc[0].get("Technology", "4G") if not ant_row.empty else "4G"
        parameter = str(update.get("parameter", "ETilt") or "ETilt").strip()
        if parameter.lower() == "azimuth":
            reason = (
                "Directional azimuth fallback selected this update after ETilt candidate passes did not produce a validated improvement "
                "for this cell. The azimuth candidate was evaluated with RF before/after validation. "
                f"dominant_bearing={_safe_float(update.get('dominant_bearing_deg'), np.nan):.1f}, "
                f"score={float(current_metrics.get('score', 0.0)):.4f}, "
                f"net_bad_grid_reduction={float(current_metrics.get('net_bad_reduction', 0.0)):.0f}."
            )
            root_cause = "directional_azimuth_fallback_validated"
        else:
            reason = (
                "Global combined weighted bad-grid coordinate search selected this ETilt update after RF before/after validation. "
                f"score={float(current_metrics.get('score', 0.0)):.4f}, "
                f"net_bad_grid_reduction={float(current_metrics.get('net_bad_reduction', 0.0)):.0f}."
            )
            root_cause = "global_bad_grid_combined_weighted"
        rows.append(
            {
                "Cell ID": cell_id,
                "Technology": tech,
                "Parameter": parameter,
                "Current Value": float(update["current_value"]),
                "Recommended Value": float(update["target_value"]),
                "Reason": reason,
                "Swap Sector Detected": "No",
                "Bad Sample Count": int(float(current_metrics.get("baseline_bad_grid_count", 0.0))),
                "Root Cause Category": root_cause,
                "Recommendation Status": "action_change_validated",
                "Recommendation Confidence": "High" if float(current_metrics.get("score", 0.0)) > 10 else "Medium",
                "Confidence Score": float(np.clip(float(current_metrics.get("score", 0.0)) * 4.0 + 15.0, 0.0, 100.0)),
                "Validation score": float(current_metrics.get("score", np.nan)),
                "Validation grid_scoring_source": current_metrics.get("grid_scoring_source"),
                "Validation validation_scope": current_metrics.get("validation_scope"),
                "Validation net_bad_reduction": float(current_metrics.get("net_bad_reduction", np.nan)),
                "Validation good_area_loss_pct": float(current_metrics.get("good_area_loss_pct", np.nan)),
                "Validation decision_scope": current_metrics.get("decision_scope"),
                "Validation combined_weighted_severity_reduction": float(current_metrics.get("combined_weighted_severity_reduction", np.nan)),
                "Validation combined_weighted_tie_break": float(current_metrics.get("combined_weighted_tie_break", np.nan)),
                "Validation weighted_frontend_net_bad_grid_reduction": float(current_metrics.get("weighted_frontend_net_bad_grid_reduction", np.nan)),
                "Validation combined_any_net_bad_grid_reduction": float(current_metrics.get("combined_any_net_bad_grid_reduction", np.nan)),
                "Validation priority_kpi_worsened": float(current_metrics.get("priority_kpi_worsened", np.nan)),
                "Validation reject_reason": "accepted_rf_validated_coordinate_search",
            }
        )
        for key, value in current_metrics.items():
            if _is_export_scalar(value):
                rows[-1][f"Metric {key}"] = value
    return {
        "recommendations": pd.DataFrame(rows),
        "candidate_evaluations": evaluation_df,
        "best_candidate_after": current_after,
        "best_candidate_metrics": pd.DataFrame([current_metrics]),
    }
