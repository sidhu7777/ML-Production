"""
Build a test-side Model 3 load-balancing prototype dataset.

This script intentionally stays outside production code. It uses the saved
Model 1 hybrid target experiment as the coverage source, re-applies the Model 2
capacity feature engineering to that hybrid coverage surface, and then adds
temporary planning proxy columns for PRB/RRC utilization.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tests.coverage_prediction.coverage_artifact_locator import resolve_coverage_artifact_path
from tests.coverage_prediction import build_model2_capacity_training_dataset as model2_builder


HYBRID_MODEL1_CSV = ML_ROOT / "models" / "model1_hybrid_target_experiment" / "hybrid_target_training.csv"
MODEL2_BASE_CSV = ML_ROOT / "data" / "model2_capacity_training.csv"
SOURCE_COVERAGE_ARCHIVE = resolve_coverage_artifact_path()
MODEL2_HYBRID_FULL_PREDICTIONS_CSV = (
    ML_ROOT / "models" / "model2_hybrid_target_experiment" / "model2_hybrid_full_predictions.csv"
)
OUTPUT_ROOT = ML_ROOT / "models" / "model3_hybrid_load_balancing_experiment"
HYBRID_MODEL2_CSV = OUTPUT_ROOT / "hybrid_model2_training.csv"
MODEL3_DATASET_CSV = OUTPUT_ROOT / "model3_load_balancing_dataset.csv"
SUMMARY_JSON = OUTPUT_ROOT / "model3_load_balancing_summary.json"

DEFAULT_RRC_SECTOR_CAPACITY = 400.0
DEFAULT_MIMO_LAYERS = 2.0
DEFAULT_CONTROL_OVERHEAD = 0.25
DEFAULT_MAX_CARRIERS_PER_SECTOR = 4
KNOWN_BAND_POOL_MHZ = [700.0, 850.0, 900.0, 1800.0, 2100.0, 2300.0]
ADD_BAND_PRIORITY = [2300.0, 2100.0, 1800.0, 900.0, 850.0, 700.0]
THRESHOLDS = [50.0, 60.0, 70.0, 80.0, 90.0]
MODEL3_PRB_TARGET_BANDS = [
    (0.00, 0.40, 20.0, 39.0),
    (0.40, 0.70, 40.0, 59.0),
    (0.70, 0.85, 60.0, 79.0),
    (0.85, 0.95, 80.0, 89.0),
    (0.95, 1.00, 90.0, 97.0),
]
MODEL3_RRC_TARGET_BANDS = [
    (0.00, 0.70, 35.0, 65.0),
    (0.70, 0.85, 65.0, 75.0),
    (0.85, 0.93, 75.0, 85.0),
    (0.93, 0.98, 85.0, 95.0),
    (0.98, 1.00, 95.0, 100.0),
]


def _save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _minmax_norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index, dtype="float64")
    lo = float(valid.min())
    hi = float(valid.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=series.index, dtype="float64")
    return ((values.fillna(lo) - lo) / (hi - lo)).clip(0.0, 1.0)


def _threshold_counts(series: pd.Series) -> dict[str, int]:
    values = pd.to_numeric(series, errors="coerce")
    return {f"gt_{int(threshold)}": int((values > threshold).sum()) for threshold in THRESHOLDS}


def _format_band_list(values: pd.Series | list[object]) -> str:
    if isinstance(values, pd.Series):
        series = pd.to_numeric(values, errors="coerce")
    else:
        series = pd.to_numeric(pd.Series(values), errors="coerce")
    cleaned = [float(v) for v in series.dropna().tolist() if np.isfinite(v)]
    if not cleaned:
        return ""
    return ",".join(str(int(v)) if float(v).is_integer() else f"{v:g}" for v in sorted(set(cleaned)))


def _available_band_options(existing_band_text: str) -> str:
    existing = set()
    if existing_band_text:
        existing = set(pd.to_numeric(pd.Series(existing_band_text.split(",")), errors="coerce").dropna().tolist())
    options = [band for band in KNOWN_BAND_POOL_MHZ if band not in existing]
    return ",".join(str(int(v)) if float(v).is_integer() else f"{v:g}" for v in options)


def _recommended_band_to_add(available_band_text: str, existing_carrier_count: int, sector_rank: float) -> str:
    if not available_band_text:
        return ""
    available = set(pd.to_numeric(pd.Series(available_band_text.split(",")), errors="coerce").dropna().tolist())
    if sector_rank < 0.25:
        priority = [2300.0, 2100.0, 1800.0, 900.0, 850.0, 700.0]
    elif sector_rank < 0.50:
        priority = [2100.0, 2300.0, 1800.0, 900.0, 850.0, 700.0]
    elif sector_rank < 0.75:
        priority = [1800.0, 900.0, 2100.0, 2300.0, 850.0, 700.0]
    else:
        priority = [700.0, 900.0, 1800.0, 2100.0, 2300.0, 850.0]

    if existing_carrier_count >= 3:
        priority = [band for band in [2300.0, 2100.0, 1800.0, 900.0, 850.0, 700.0] if band in priority]

    for band in priority:
        if band in available:
            return str(int(band)) if float(band).is_integer() else f"{band:g}"
    first = next(iter(sorted(available)), None)
    if first is None:
        return ""
    return str(int(first)) if float(first).is_integer() else f"{first:g}"


def _choose_sector_key(df: pd.DataFrame) -> str | None:
    for candidate in [
        "topology_frontend_site_sector_key",
        "topology_node_cell_sector_key",
        "topology_sector_identity_key",
        "topology_canonical_sector_id",
        "topology_site_sector_band_key",
    ]:
        if candidate in df.columns:
            return candidate
    return None


def _rank_percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index, dtype="float64")
    return values.rank(method="average", pct=True).fillna(0.0).clip(0.0, 1.0)


def _tiered_target_from_rank(rank: pd.Series, bands: list[tuple[float, float, float, float]]) -> pd.Series:
    target = pd.Series(np.nan, index=rank.index, dtype="float64")
    for idx, (lower, upper, target_lower, target_upper) in enumerate(bands):
        if upper <= lower:
            continue
        if idx == 0:
            mask = (rank >= lower) & (rank <= upper)
        else:
            mask = (rank > lower) & (rank <= upper)
        if not mask.any():
            continue
        local = (rank.loc[mask] - lower) / (upper - lower)
        target.loc[mask] = target_lower + local * (target_upper - target_lower)
    if target.isna().any():
        target = target.fillna(bands[0][2] if bands else 0.0)
    return target.clip(lower=min(b[2] for b in bands), upper=max(b[3] for b in bands)).round(3)


def _archive_root(archive_path: Path) -> str:
    listed = subprocess.check_output(["tar", "-tf", str(archive_path)], text=True)
    first_file = next((line for line in listed.splitlines() if "/" in line and not line.endswith("/")), "")
    if not first_file:
        raise RuntimeError(f"No files found in archive: {archive_path}")
    return first_file.split("/", 1)[0]


def _read_csv_from_archive(archive_path: Path, member_name: str) -> pd.DataFrame:
    root = _archive_root(archive_path)
    raw = subprocess.check_output(["tar", "-xOf", str(archive_path), f"{root}/{member_name}"])
    return pd.read_csv(io.BytesIO(raw))


def _extract_site_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.rsplit("_", n=1).str[0]


def _estimate_spectral_efficiency_bpshz(
    df: pd.DataFrame,
    bandwidth_mhz: pd.Series,
    mimo_layers: float,
    control_overhead: float,
) -> pd.Series:
    sinr = pd.to_numeric(df.get("sinr_mean"), errors="coerce")
    cqi = pd.to_numeric(df.get("cqi_mean"), errors="coerce")
    dl_tpt = pd.to_numeric(df.get("dl_tpt_mean"), errors="coerce")

    # Shannon-style estimate with an implementation loss factor, bounded to
    # practical LTE/NR planning ranges for a prototype capacity proxy.
    sinr_linear = np.power(10.0, sinr / 10.0)
    sinr_efficiency = 0.75 * np.log2(1.0 + sinr_linear)

    # CQI-to-spectral-efficiency approximation. It is intentionally smooth and
    # conservative, not a standards table replacement.
    cqi_efficiency = 0.1523 * cqi.clip(lower=1.0, upper=15.0)

    usable_resource_fraction = max(0.10, min(1.0, 1.0 - float(control_overhead)))
    layer_count = max(1.0, float(mimo_layers))
    observed_efficiency = dl_tpt / (bandwidth_mhz * layer_count * usable_resource_fraction)
    observed_efficiency = observed_efficiency.where(np.isfinite(observed_efficiency))
    band_labels = df.get("dominant_band_class", pd.Series("UNKNOWN", index=df.index)).fillna("UNKNOWN").astype(str)
    band_median_efficiency = observed_efficiency.groupby(band_labels).transform("median")
    global_median_efficiency = observed_efficiency.median()
    if not np.isfinite(global_median_efficiency):
        global_median_efficiency = 1.0
    observed_efficiency_fallback = observed_efficiency.combine_first(band_median_efficiency).fillna(global_median_efficiency)

    combined = pd.concat([sinr_efficiency, cqi_efficiency, observed_efficiency_fallback], axis=1).max(axis=1)
    combined = combined.fillna(1.0)
    return combined.clip(lower=0.15, upper=6.0).round(6)


def _apply_model3_load_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()

    score_components: list[pd.Series] = []
    score_weights: list[float] = []
    for col, weight in [
        ("demand_index_pred", 0.35),
        ("traffic_demand_est_pred", 0.30),
        ("active_users_est_pred", 0.15),
        ("prb_pressure_est", 0.10),
        ("capacity_gap_score", 0.05),
        ("growth_zone_score", 0.05),
    ]:
        if col in out.columns:
            score_components.append(_minmax_norm(out[col]))
            score_weights.append(weight)

    if score_components:
        total_weight = float(sum(score_weights)) or 1.0
        hotspot_score = sum(component * weight for component, weight in zip(score_components, score_weights)) / total_weight
    else:
        hotspot_score = pd.Series(0.0, index=out.index, dtype="float64")

    if "high_band_ratio" in out.columns:
        hotspot_score = (0.85 * hotspot_score) + (0.15 * _minmax_norm(out["high_band_ratio"]))
    if "carrier_count" in out.columns:
        hotspot_score = (0.90 * hotspot_score) + (0.10 * _minmax_norm(out["carrier_count"]))

    hotspot_score = hotspot_score.clip(0.0, 1.0).round(6)
    hotspot_rank = _rank_percentile(hotspot_score)
    prb_target_pct = _tiered_target_from_rank(hotspot_rank, MODEL3_PRB_TARGET_BANDS)

    capacity_mbps = pd.to_numeric(out.get("estimated_dl_capacity_mbps"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if capacity_mbps.notna().sum() == 0:
        capacity_mbps = pd.Series(1.0, index=out.index, dtype="float64")
    capacity_mbps = capacity_mbps.fillna(capacity_mbps.median() if capacity_mbps.notna().any() else 1.0).clip(lower=0.1)

    shaped_traffic_mbps = (capacity_mbps * (prb_target_pct / 100.0)).clip(lower=0.0).round(3)
    out["model3_hotspot_score"] = hotspot_score
    out["model3_hotspot_rank"] = hotspot_rank.round(6)
    out["model3_prb_target_pct"] = prb_target_pct
    out["estimated_offered_traffic_mbps"] = shaped_traffic_mbps
    out["estimated_prb_utilization_pct"] = prb_target_pct

    # Build a sector-level load profile so RRC concentration is realistic at
    # the true cell/carrier grouping rather than uniform per grid.
    rrc_group_cols = next(
        (
            cols
            for cols in [
                ["time_bucket", "site_id", "sector_id", "cell_id"],
                ["time_bucket", "Node_Cell_ID"],
                ["time_bucket", "cell_id"],
            ]
            if all(col in out.columns for col in cols)
        ),
        None,
    )

    if rrc_group_cols:
        group_profile = (
            out.groupby(rrc_group_cols, dropna=False, as_index=False)
            .agg(
                group_hotspot_score=("model3_hotspot_score", "mean"),
                group_grid_count=("grid_id", "nunique"),
                group_user_weight=("model3_hotspot_score", "sum"),
            )
        )
        group_profile["group_rank"] = _rank_percentile(group_profile["group_hotspot_score"])
        group_profile["group_rrc_target_pct"] = _tiered_target_from_rank(group_profile["group_rank"], MODEL3_RRC_TARGET_BANDS)
        group_profile["group_total_users_target"] = (
            pd.to_numeric(group_profile["group_rrc_target_pct"], errors="coerce").fillna(0.0) / 100.0
        ) * float(DEFAULT_RRC_SECTOR_CAPACITY)
        group_profile["group_user_weight"] = pd.to_numeric(group_profile["group_user_weight"], errors="coerce").fillna(0.0)
        group_profile["group_user_weight"] = group_profile["group_user_weight"].where(
            group_profile["group_user_weight"] > 0,
            1.0,
        )

        out = out.merge(group_profile, on=rrc_group_cols, how="left", validate="many_to_one")
        out["model3_rrc_group_rank"] = pd.to_numeric(out["group_rank"], errors="coerce").fillna(0.0).round(6)
        out["model3_rrc_target_pct"] = pd.to_numeric(out["group_rrc_target_pct"], errors="coerce").fillna(0.0).round(3)
        out["model3_rrc_group_total_users_target"] = pd.to_numeric(out["group_total_users_target"], errors="coerce").fillna(0.0).round(3)

        row_weight = (
            0.65 * _minmax_norm(out["model3_hotspot_score"])
            + 0.35 * _minmax_norm(out["traffic_demand_est_pred"] if "traffic_demand_est_pred" in out.columns else out["model3_hotspot_score"])
        )
        row_weight = row_weight.fillna(0.0) + 0.10
        group_index = out.groupby(rrc_group_cols, sort=False).ngroup()
        weight_sum = row_weight.groupby(group_index, sort=False).transform("sum")
        weight_sum = weight_sum.replace(0, np.nan).fillna(1.0)
        out["estimated_rrc_connected_users"] = (
            out["model3_rrc_group_total_users_target"] * (row_weight / weight_sum)
        ).clip(lower=0.0).round(3)
        out["estimated_cell_grid_count"] = pd.to_numeric(out["group_grid_count"], errors="coerce").fillna(1.0).round(3)
        out["estimated_cell_rrc_connected_users"] = (
            out["model3_rrc_group_total_users_target"]
        ).round(3)
        out["estimated_cell_rrc_utilization_pct"] = pd.to_numeric(out["model3_rrc_target_pct"], errors="coerce").fillna(0.0).round(3)

        profile_summary = {
            "rrc_grouping_columns": rrc_group_cols,
            "rrc_group_count": int(len(group_profile)),
            "rrc_group_rank_distribution": {
                "p50": float(group_profile["group_rank"].quantile(0.50)),
                "p80": float(group_profile["group_rank"].quantile(0.80)),
                "p90": float(group_profile["group_rank"].quantile(0.90)),
            },
            "rrc_target_pct_distribution": {
                "p50": float(group_profile["group_rrc_target_pct"].quantile(0.50)),
                "p80": float(group_profile["group_rrc_target_pct"].quantile(0.80)),
                "p90": float(group_profile["group_rrc_target_pct"].quantile(0.90)),
            },
            "rrc_target_threshold_counts": _threshold_counts(group_profile["group_rrc_target_pct"]),
        }
    else:
        out["model3_rrc_group_rank"] = 0.0
        out["model3_rrc_target_pct"] = 0.0
        out["model3_rrc_group_total_users_target"] = 0.0
        out["estimated_rrc_connected_users"] = (
            pd.to_numeric(out["active_users_est_pred"], errors="coerce").fillna(0.0).round(3)
            if "active_users_est_pred" in out.columns
            else pd.Series(0.0, index=out.index, dtype="float64")
        )
        out["estimated_cell_grid_count"] = np.nan
        out["estimated_cell_rrc_connected_users"] = np.nan
        out["estimated_cell_rrc_utilization_pct"] = np.nan
        profile_summary = {
            "rrc_grouping_columns": None,
            "rrc_group_count": 0,
            "rrc_group_rank_distribution": {},
            "rrc_target_pct_distribution": {},
            "rrc_target_threshold_counts": {},
        }

    out["model3_profile_source"] = "ranked_hotspot_targeting_on_model2_predictions"
    return out, profile_summary


def _add_sector_carrier_capability_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    sector_key = _choose_sector_key(out)
    if sector_key is None:
        out["existing_carriers"] = ""
        out["existing_carrier_count"] = 0
        out["available_bands_to_add"] = ""
        out["carrier_addition_options"] = ""
        out["available_earfcns_to_add"] = ""
        out["available_earfcn_options"] = ""
        out["recommended_band_to_add"] = ""
        out["available_band_options_count"] = 0
        out["max_supported_carriers"] = DEFAULT_MAX_CARRIERS_PER_SECTOR
        out["carrier_addition_possible"] = False
        out["sector_has_alternate_carrier"] = False
        out["sector_capacity_limit"] = DEFAULT_MAX_CARRIERS_PER_SECTOR
        out["carrier_addition_reason"] = "NO_SECTOR_KEY"
        return out, {
            "sector_key": None,
            "sector_count": 0,
            "carrier_addition_possible_rows": 0,
            "sector_has_alternate_carrier_rows": 0,
        }

    work = out[[sector_key, "topology_band"] + (["topology_earfcn"] if "topology_earfcn" in out.columns else [])].copy()
    work["topology_band"] = pd.to_numeric(work["topology_band"], errors="coerce")
    if "topology_earfcn" in work.columns:
        work["topology_earfcn"] = pd.to_numeric(work["topology_earfcn"], errors="coerce")

    sector_groups = work.groupby(sector_key, dropna=False)
    sector_df = sector_groups.agg(
        existing_carriers=("topology_band", _format_band_list),
        existing_carrier_count=("topology_band", lambda s: int(pd.to_numeric(s, errors="coerce").dropna().nunique())),
        existing_earfcns=("topology_earfcn", _format_band_list) if "topology_earfcn" in work.columns else (sector_key, "size"),
    ).reset_index()
    if "topology_earfcn" not in work.columns:
        sector_df["existing_earfcns"] = ""

    sector_df["_sector_rank"] = sector_df[sector_key].astype(str).rank(method="dense", pct=True).fillna(0.0)
    sector_df["available_bands_to_add"] = sector_df["existing_carriers"].apply(_available_band_options)
    sector_df["carrier_addition_options"] = sector_df["available_bands_to_add"]
    sector_df["available_band_options_count"] = sector_df["available_bands_to_add"].apply(
        lambda raw: 0 if not raw else len([token for token in raw.split(",") if token.strip()])
    )
    sector_df["available_earfcns_to_add"] = sector_df["available_bands_to_add"].apply(
        lambda raw: ",".join(
            {
                "700": "700",
                "850": "850",
                "900": "900",
                "1800": "1750",
                "2100": "2100",
                "2300": "2300",
            }.get(token.strip(), token.strip())
            for token in raw.split(",")
            if token.strip()
        )
    )
    sector_df["available_earfcn_options"] = sector_df["available_earfcns_to_add"]
    sector_df["recommended_band_to_add"] = sector_df.apply(
        lambda row: _recommended_band_to_add(
            row["available_bands_to_add"],
            int(row["existing_carrier_count"]),
            float(row["_sector_rank"]),
        ),
        axis=1,
    )

    def _tiered_sector_cap(row: pd.Series) -> int:
        existing = int(row["existing_carrier_count"])
        rank = float(row["_sector_rank"])
        if rank < 0.25:
            cap = existing
        elif rank < 0.50:
            cap = max(existing, min(DEFAULT_MAX_CARRIERS_PER_SECTOR, existing + 1))
        elif rank < 0.75:
            cap = max(existing, min(DEFAULT_MAX_CARRIERS_PER_SECTOR, existing + 2))
        else:
            cap = DEFAULT_MAX_CARRIERS_PER_SECTOR
        return int(max(existing, min(DEFAULT_MAX_CARRIERS_PER_SECTOR, cap)))

    sector_df["max_supported_carriers"] = sector_df.apply(_tiered_sector_cap, axis=1)
    sector_df["sector_capacity_limit"] = sector_df["max_supported_carriers"]
    sector_df["sector_has_alternate_carrier"] = sector_df["existing_carrier_count"] > 1
    sector_df["carrier_addition_possible"] = (
        (sector_df["existing_carrier_count"] < sector_df["max_supported_carriers"])
        & sector_df["available_bands_to_add"].astype(str).ne("")
    )
    sector_df["carrier_addition_blocked"] = ~sector_df["carrier_addition_possible"]
    sector_df["carrier_addition_reason"] = np.where(
        sector_df["carrier_addition_possible"],
        "AVAILABLE",
        np.where(sector_df["available_bands_to_add"].astype(str).eq(""), "NO_BAND_AVAILABLE", "AT_CAPACITY"),
    )
    sector_df = sector_df.drop(columns=["_sector_rank"])

    capability_summary = {
        "sector_key": sector_key,
        "sector_count": int(len(sector_df)),
        "carrier_addition_possible_rows": int(sector_df["carrier_addition_possible"].sum()),
        "carrier_addition_option_rows": int((sector_df["available_band_options_count"] > 0).sum()),
        "sector_has_alternate_carrier_rows": int(sector_df["sector_has_alternate_carrier"].sum()),
        "multi_carrier_sector_rows": int((sector_df["existing_carrier_count"] > 1).sum()),
        "single_carrier_sector_rows": int((sector_df["existing_carrier_count"] == 1).sum()),
        "carrier_addition_blocked_rows": int(sector_df["carrier_addition_blocked"].sum()),
        "max_supported_carriers_min": int(sector_df["max_supported_carriers"].min()),
        "max_supported_carriers_max": int(sector_df["max_supported_carriers"].max()),
        "recommended_band_examples": sector_df["recommended_band_to_add"].replace("", np.nan).dropna().value_counts().head(5).to_dict(),
    }

    merged = out.merge(sector_df, on=sector_key, how="left", validate="many_to_one")
    return merged, capability_summary


def _load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not HYBRID_MODEL1_CSV.exists():
        raise FileNotFoundError(f"Missing hybrid Model 1 dataset: {HYBRID_MODEL1_CSV}")
    if not MODEL2_BASE_CSV.exists():
        raise FileNotFoundError(f"Missing Model 2 base dataset: {MODEL2_BASE_CSV}")

    hybrid = pd.read_csv(HYBRID_MODEL1_CSV)
    model2_base = pd.read_csv(MODEL2_BASE_CSV)

    for frame in (hybrid, model2_base):
        frame["grid_id"] = pd.to_numeric(frame["grid_id"], errors="coerce").astype("Int64")
        frame["time_bucket"] = frame["time_bucket"].astype(str)

    return hybrid, model2_base


def _derive_grid_cell_keys_from_archive() -> pd.DataFrame:
    if not SOURCE_COVERAGE_ARCHIVE.exists():
        return pd.DataFrame()

    pred_df = _read_csv_from_archive(SOURCE_COVERAGE_ARCHIVE, "bucket_corrected_prediction_grid.csv")
    if pred_df.empty or "Node_Cell_ID" not in pred_df.columns:
        pred_df = _read_csv_from_archive(SOURCE_COVERAGE_ARCHIVE, "baseline_prediction_grid.csv")
    if pred_df.empty or "Node_Cell_ID" not in pred_df.columns:
        return pd.DataFrame()

    join_keys = ["grid_id", "time_bucket"]
    work = pred_df.copy()
    work["grid_id"] = pd.to_numeric(work["grid_id"], errors="coerce").astype("Int64")
    work["time_bucket"] = work["time_bucket"].astype(str)
    work["Node_Cell_ID"] = work["Node_Cell_ID"].astype(str)
    work["_rank_rsrp"] = pd.to_numeric(work.get("pred_rsrp"), errors="coerce")
    work = work.sort_values(join_keys + ["_rank_rsrp"], ascending=[True, True, False])
    best = work.dropna(subset=["grid_id"]).drop_duplicates(subset=join_keys, keep="first")
    best = best[join_keys + ["Node_Cell_ID"]].copy()
    best["site_id"] = _extract_site_id(best["Node_Cell_ID"])

    topology_keep = [
        col
        for col in [
            "Node_Cell_ID",
            "Site ID",
            "site_identity_key",
            "sector_identity",
            "sector_identity_key",
            "frontend_site_sector_key",
            "node_cell_sector_key",
            "sector",
            "band",
            "earfcn",
            "nodeb_id",
            "PCI",
            "azimuth",
            "canonical_sector_id",
            "site_sector_band_key",
            "rf_identity_key",
            "original_node_cell_id",
            "original_cell_id",
            "carrier_load_share",
        ]
        if col in pred_df.columns
    ]
    topology_df = pred_df[topology_keep].copy()
    if "Node_Cell_ID" in topology_df.columns:
        topology_df["Node_Cell_ID"] = topology_df["Node_Cell_ID"].astype(str)
        if "Site ID" in topology_df.columns:
            topology_df["Site ID"] = pd.to_numeric(topology_df["Site ID"], errors="coerce").astype("Int64")
        if "band" in topology_df.columns:
            topology_df["band"] = pd.to_numeric(topology_df["band"], errors="coerce")
        if "earfcn" in topology_df.columns:
            topology_df["earfcn"] = pd.to_numeric(topology_df["earfcn"], errors="coerce")
        if "nodeb_id" in topology_df.columns:
            topology_df["nodeb_id"] = pd.to_numeric(topology_df["nodeb_id"], errors="coerce")
        if "PCI" in topology_df.columns:
            topology_df["PCI"] = pd.to_numeric(topology_df["PCI"], errors="coerce")
        if "azimuth" in topology_df.columns:
            topology_df["azimuth"] = pd.to_numeric(topology_df["azimuth"], errors="coerce")
        if "carrier_load_share" in topology_df.columns:
            topology_df["carrier_load_share"] = pd.to_numeric(topology_df["carrier_load_share"], errors="coerce")
        topology_df = topology_df.drop_duplicates(subset=["Node_Cell_ID"], keep="first")
        best = best.merge(topology_df, on="Node_Cell_ID", how="left", validate="many_to_one")
        rename_map = {
            "Site ID": "topology_site_id",
            "site_identity_key": "topology_site_identity_key",
            "sector_identity": "topology_sector_identity",
            "sector_identity_key": "topology_sector_identity_key",
            "frontend_site_sector_key": "topology_frontend_site_sector_key",
            "node_cell_sector_key": "topology_node_cell_sector_key",
            "sector": "topology_sector",
            "band": "topology_band",
            "earfcn": "topology_earfcn",
            "nodeb_id": "topology_nodeb_id",
            "PCI": "topology_pci",
            "azimuth": "topology_azimuth",
            "canonical_sector_id": "topology_canonical_sector_id",
            "site_sector_band_key": "topology_site_sector_band_key",
            "rf_identity_key": "topology_rf_identity_key",
            "original_node_cell_id": "topology_original_node_cell_id",
            "original_cell_id": "topology_original_cell_id",
            "carrier_load_share": "topology_carrier_load_share",
        }
        best = best.rename(columns={k: v for k, v in rename_map.items() if k in best.columns})
        if "topology_site_id" in best.columns:
            best["site_id"] = best["topology_site_id"].astype("object").combine_first(best["site_id"].astype("object"))

    return best


def build_hybrid_model2_dataset() -> pd.DataFrame:
    hybrid, model2_base = _load_sources()
    join_keys = ["grid_id", "time_bucket"]

    base_keep = [
        "grid_id",
        "time_bucket",
        "grid_row",
        "grid_col",
        "grid_centroid_lat",
        "grid_centroid_lon",
        "sample_count",
        "dl_tpt_mean",
        "ul_tpt_mean",
        "estimated_prb_mean",
        "cqi_mean",
        "dominant_pci",
        "green_ratio",
        "water_ratio",
        "grid_size_m",
        "grid_area_m2",
        "cell_area_m2",
        "road_length_m",
        "building_count",
        "building_area_ratio",
        "park_open_area",
        "open_area_ratio",
        "mall_presence",
        "metro_presence",
        "road_density",
        "geo_snapshot_mode",
        "geo_snapshot_source_ts",
    ]
    base_keep = [col for col in base_keep if col in model2_base.columns]
    base_context = model2_base[base_keep].drop_duplicates(subset=join_keys)

    preferred_hybrid = [
        "grid_id",
        "time_bucket",
        "grid_row",
        "grid_col",
        "grid_centroid_lat",
        "grid_centroid_lon",
        "label_source",
        "dt_samples",
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "corrected_rsrp_mean",
        "corrected_rsrq_mean",
        "corrected_sinr_mean",
        "bandwidth_mhz_est",
        "low_band_ratio",
        "mid_band_ratio",
        "high_band_ratio",
        "dominant_band_class",
        "carrier_count",
        "clutter_class",
        "green_ratio",
        "water_ratio",
        "grid_size_m",
        "grid_area_m2",
        "cell_area_m2",
        "road_length_m",
        "building_count",
        "building_area_ratio",
        "park_open_area",
        "open_area_ratio",
        "mall_presence",
        "metro_presence",
        "road_density",
    ]
    hybrid_keep = [col for col in preferred_hybrid if col in hybrid.columns]
    work = hybrid[hybrid_keep].copy()

    work = work.rename(
        columns={
            "pred_rsrp": "rsrp_mean",
            "pred_rsrq": "rsrq_mean",
            "pred_sinr": "sinr_mean",
        }
    )

    merged = work.merge(base_context, on=join_keys, how="left", suffixes=("", "_base"))
    for col in list(merged.columns):
        if not col.endswith("_base"):
            continue
        original = col[:-5]
        if original not in merged.columns:
            merged = merged.rename(columns={col: original})
        else:
            merged[original] = merged[original].combine_first(merged[col])
            merged = merged.drop(columns=[col])

    if "sample_count" not in merged.columns:
        merged["sample_count"] = pd.to_numeric(merged.get("dt_samples"), errors="coerce").fillna(0.0)
    else:
        merged["sample_count"] = pd.to_numeric(merged["sample_count"], errors="coerce").combine_first(
            pd.to_numeric(merged.get("dt_samples"), errors="coerce")
        )

    for corrected_col, source_col in [
        ("corrected_rsrp_mean", "rsrp_mean"),
        ("corrected_rsrq_mean", "rsrq_mean"),
        ("corrected_sinr_mean", "sinr_mean"),
    ]:
        if corrected_col not in merged.columns:
            merged[corrected_col] = merged[source_col]
        else:
            merged[corrected_col] = pd.to_numeric(merged[corrected_col], errors="coerce").combine_first(
                pd.to_numeric(merged[source_col], errors="coerce")
            )

    cell_keys = _derive_grid_cell_keys_from_archive()
    if not cell_keys.empty:
        merged = merged.merge(cell_keys, on=join_keys, how="left", validate="one_to_one")

    enriched = model2_builder._add_model2_features(merged)
    enriched = enriched.sort_values(["grid_id", "bucket_seq"]).reset_index(drop=True)
    HYBRID_MODEL2_CSV.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(HYBRID_MODEL2_CSV, index=False)
    return enriched


def build_model3_dataset(
    rrc_sector_capacity: float = DEFAULT_RRC_SECTOR_CAPACITY,
    mimo_layers: float = DEFAULT_MIMO_LAYERS,
    control_overhead: float = DEFAULT_CONTROL_OVERHEAD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hybrid_model2 = build_hybrid_model2_dataset()
    out = hybrid_model2[hybrid_model2["time_bucket"].astype(str) == "PART_3"].copy()
    out, capability_summary = _add_sector_carrier_capability_fields(out)

    model2_prediction_source = "engineered_hybrid_model2_targets"
    demand_col = "demand_index"
    users_col = "active_users_est"
    traffic_col = "traffic_demand_est"
    if MODEL2_HYBRID_FULL_PREDICTIONS_CSV.exists():
        pred_df = pd.read_csv(MODEL2_HYBRID_FULL_PREDICTIONS_CSV)
        pred_df["grid_id"] = pd.to_numeric(pred_df["grid_id"], errors="coerce").astype("Int64")
        pred_df["time_bucket"] = pred_df["time_bucket"].astype(str)
        pred_keep = [
            col
            for col in [
                "grid_id",
                "time_bucket",
                "demand_index_pred",
                "active_users_est_pred",
                "traffic_demand_est_pred",
            ]
            if col in pred_df.columns
        ]
        if len(pred_keep) == 5:
            out = out.merge(pred_df[pred_keep], on=["grid_id", "time_bucket"], how="left", validate="one_to_one")
            demand_col = "demand_index_pred"
            users_col = "active_users_est_pred"
            traffic_col = "traffic_demand_est_pred"
            model2_prediction_source = str(MODEL2_HYBRID_FULL_PREDICTIONS_CSV)

    bandwidth_mhz = pd.to_numeric(out.get("bandwidth_mhz_est"), errors="coerce").replace(0, np.nan).fillna(10.0)
    usable_resource_fraction = max(0.10, min(1.0, 1.0 - float(control_overhead)))
    layer_count = max(1.0, float(mimo_layers))
    spectral_efficiency = _estimate_spectral_efficiency_bpshz(
        out,
        bandwidth_mhz=bandwidth_mhz,
        mimo_layers=layer_count,
        control_overhead=control_overhead,
    )
    out["estimated_spectral_efficiency_bpshz"] = spectral_efficiency
    out["estimated_dl_capacity_mbps"] = (
        bandwidth_mhz * spectral_efficiency * layer_count * usable_resource_fraction
    ).clip(lower=0.1).round(3)
    out, profile_summary = _apply_model3_load_profile(out)
    rrc_aggregation_note = (
        "RRC utilization is aggregated by the true cell/group key chosen by the load-profile step. "
        f"Target banding is derived from model3_hotspot_score and shaped to the requested congestion mix."
    )

    out["model3_proxy_source"] = "hybrid_model1_coverage_plus_model2_engineered_capacity_proxies"
    out.to_csv(MODEL3_DATASET_CSV, index=False)

    metric_cols = [
        "estimated_prb_utilization_pct",
        "estimated_rrc_connected_users",
        "estimated_cell_rrc_connected_users",
        "estimated_cell_rrc_utilization_pct",
        "demand_index",
        "active_users_est",
        "traffic_demand_est",
        "estimated_prb_mean",
        "prb_pressure_est",
    ]
    metrics: dict[str, Any] = {}
    for col in metric_cols:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        metrics[col] = {
            "non_null": int(values.notna().sum()),
            "min": float(values.min()) if values.notna().any() else None,
            "p50": float(values.quantile(0.50)) if values.notna().any() else None,
            "p60": float(values.quantile(0.60)) if values.notna().any() else None,
            "p70": float(values.quantile(0.70)) if values.notna().any() else None,
            "p80": float(values.quantile(0.80)) if values.notna().any() else None,
            "p90": float(values.quantile(0.90)) if values.notna().any() else None,
            "p95": float(values.quantile(0.95)) if values.notna().any() else None,
            "max": float(values.max()) if values.notna().any() else None,
            "threshold_counts": _threshold_counts(values),
        }

    rrc_group_cols = profile_summary.get("rrc_grouping_columns")
    rrc_cell_group_metrics = None
    if rrc_group_cols and "estimated_cell_rrc_utilization_pct" in out.columns:
        cell_groups = out[rrc_group_cols + [
            "estimated_cell_grid_count",
            "estimated_cell_rrc_connected_users",
            "estimated_cell_rrc_utilization_pct",
        ]].drop_duplicates(subset=rrc_group_cols)
        util = pd.to_numeric(cell_groups["estimated_cell_rrc_utilization_pct"], errors="coerce")
        rrc_cell_group_metrics = {
            "grouping_columns": rrc_group_cols,
            "groups": int(len(cell_groups)),
            "utilization_non_null": int(util.notna().sum()),
            "utilization_min": float(util.min()) if util.notna().any() else None,
            "utilization_p50": float(util.quantile(0.50)) if util.notna().any() else None,
            "utilization_p80": float(util.quantile(0.80)) if util.notna().any() else None,
            "utilization_p90": float(util.quantile(0.90)) if util.notna().any() else None,
            "utilization_max": float(util.max()) if util.notna().any() else None,
            "threshold_counts": _threshold_counts(util),
        }

    band_summary = {}
    if "dominant_band_class" in out.columns:
        band_summary = {
            str(k): int(v)
            for k, v in out["dominant_band_class"].fillna("UNKNOWN").value_counts(dropna=False).items()
        }

    topology_verification: dict[str, Any] = {}
    if "Node_Cell_ID" in out.columns:
        topology_verification = {
            "node_cell_id_non_null_rows": int(out["Node_Cell_ID"].notna().sum()),
            "unique_node_cell_ids": int(out["Node_Cell_ID"].nunique(dropna=True)),
            "node_cell_id_multi_site_count": int(out.groupby("Node_Cell_ID")["site_id"].nunique(dropna=True).gt(1).sum())
            if "site_id" in out.columns
            else None,
            "node_cell_id_multi_band_count": int(out.groupby("Node_Cell_ID")["topology_band"].nunique(dropna=True).gt(1).sum())
            if "topology_band" in out.columns
            else None,
            "node_cell_id_multi_earfcn_count": int(out.groupby("Node_Cell_ID")["topology_earfcn"].nunique(dropna=True).gt(1).sum())
            if "topology_earfcn" in out.columns
            else None,
            "topology_band_counts": {
                str(k): int(v)
                for k, v in out.get("topology_band", pd.Series(dtype="object")).value_counts(dropna=False).items()
            },
            "topology_earfcn_counts": {
                str(k): int(v)
                for k, v in out.get("topology_earfcn", pd.Series(dtype="object")).value_counts(dropna=False).items()
            },
            "note": (
                "Node_Cell_ID is verified as a cell/carrier key in this artifact: it maps to one site, one band, "
                "and one EARFCN. The topology fields are now sourced from the regenerated prediction grid, so the "
                "Model 3 dataset reflects the current synthetic carrier mix rather than the old project_sites export."
            ),
        }

    rrc_group_cols = profile_summary.get("rrc_grouping_columns")
    summary = {
        "hybrid_model1_csv": str(HYBRID_MODEL1_CSV),
        "base_model2_csv": str(MODEL2_BASE_CSV),
        "hybrid_model2_csv": str(HYBRID_MODEL2_CSV),
        "model3_dataset_csv": str(MODEL3_DATASET_CSV),
        "rows": int(len(out)),
        "unique_grids": int(out["grid_id"].nunique(dropna=True)),
        "bucket_counts": {str(k): int(v) for k, v in out["time_bucket"].value_counts().sort_index().items()},
        "rrc_sector_capacity_assumption": float(rrc_sector_capacity),
        "mimo_layers_assumption": float(mimo_layers),
        "control_overhead_assumption": float(control_overhead),
        "model2_prediction_source": model2_prediction_source,
        "model2_columns_used_for_proxy": {
            "demand": demand_col,
            "users": users_col,
            "traffic": traffic_col,
        },
        "new_model3_proxy_columns": [
            "model3_hotspot_score",
            "model3_hotspot_rank",
            "model3_prb_target_pct",
            "model3_rrc_group_rank",
            "model3_rrc_target_pct",
            "estimated_offered_traffic_mbps",
            "estimated_spectral_efficiency_bpshz",
            "estimated_dl_capacity_mbps",
            "estimated_prb_utilization_pct",
            "estimated_rrc_connected_users",
        "estimated_cell_rrc_connected_users",
        "estimated_cell_rrc_utilization_pct",
        "existing_carriers",
        "existing_carrier_count",
        "available_bands_to_add",
        "carrier_addition_options",
        "available_earfcns_to_add",
        "available_earfcn_options",
        "recommended_band_to_add",
        "available_band_options_count",
        "max_supported_carriers",
        "carrier_addition_possible",
        "carrier_addition_blocked",
        "carrier_addition_reason",
        "sector_has_alternate_carrier",
        "sector_capacity_limit",
    ],
        "proxy_formula_notes": [
            "estimated_spectral_efficiency_bpshz uses a conservative Shannon-style SINR estimate with CQI fallback.",
            "estimated_dl_capacity_mbps = bandwidth_mhz_est * estimated_spectral_efficiency_bpshz * mimo_layers * (1 - control_overhead).",
            "estimated_prb_utilization_pct is intentionally shaped to a realistic hotspot mix, then reconstructed via estimated_offered_traffic_mbps / estimated_dl_capacity_mbps * 100.",
            "estimated_offered_traffic_mbps is derived from the Model 2 traffic prediction and then redistributed into target load bands for the Model 3 prototype.",
            "estimated_rrc_connected_users is redistributed at the cell/group level from the Model 2 active-users prediction before aggregation.",
            "estimated_rrc_connected_users remains a row-level user estimate after group shaping.",
            "estimated_cell_rrc_connected_users = sum(estimated_rrc_connected_users) across the available cell-like group.",
            "estimated_cell_rrc_utilization_pct = estimated_cell_rrc_connected_users / sector_rrc_capacity * 100.",
            "existing_carriers / available_bands_to_add are synthetic sector-capability metadata derived from the current topology.",
            "recommended_band_to_add picks the highest-priority available band from the sector's spare-band list.",
            "carrier_addition_possible is True only when the sector has a spare band and the synthetic carrier cap is not exceeded.",
            "carrier_addition_blocked marks sectors that should skip carrier addition and move to the next optimization branch.",
            rrc_aggregation_note,
            "These are prototype engineering proxies, not industry-standard OSS counters.",
        ],
        "dominant_band_class_counts": band_summary,
        "topology_verification": topology_verification,
        "sector_carrier_capability": capability_summary,
        "model3_load_profile": profile_summary,
        "metrics": metrics,
        "rrc_cell_group_metrics": rrc_cell_group_metrics,
    }
    _save_json(summary, SUMMARY_JSON)
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Model 3 hybrid load-balancing prototype dataset.")
    parser.add_argument("--rrc-sector-capacity", type=float, default=DEFAULT_RRC_SECTOR_CAPACITY)
    parser.add_argument("--mimo-layers", type=float, default=DEFAULT_MIMO_LAYERS)
    parser.add_argument("--control-overhead", type=float, default=DEFAULT_CONTROL_OVERHEAD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = build_model3_dataset(args.rrc_sector_capacity, args.mimo_layers, args.control_overhead)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
