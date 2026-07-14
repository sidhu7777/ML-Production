from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook

ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tests.coverage_prediction import build_model3_hybrid_load_balancing_dataset as future_builder
from tests.coverage_prediction import build_model3_current_recommendation_dataset as current_builder
from tests.coverage_prediction import lte_coverage_test as coverage_test
from tests.coverage_prediction import model3_business_rule_recommendation_test as future_rules


DEFAULT_DATASET = current_builder.CURRENT_MODEL3_DATASET_CSV
DEFAULT_SUMMARY = current_builder.CURRENT_MODEL3_SUMMARY_JSON
DEFAULT_OUTPUT_ROOT = ML_ROOT / "tests" / "output" / "model3_current_recommendation"
DEFAULT_STABLE_OUTPUT_DIR = ML_ROOT / "models" / "model3_current_recommendation_experiment"
DEFAULT_CONGESTION_THRESHOLD = 80.0


@dataclass
class CurrentModel3Config:
    dataset_path: Path = DEFAULT_DATASET
    summary_path: Path = DEFAULT_SUMMARY
    output_root: Path = DEFAULT_OUTPUT_ROOT
    stable_output_dir: Path = DEFAULT_STABLE_OUTPUT_DIR
    congestion_threshold: float = DEFAULT_CONGESTION_THRESHOLD
    rrc_sector_capacity: float = future_builder.DEFAULT_RRC_SECTOR_CAPACITY
    sector_split_local_radius_m: float = 900.0
    max_sectors: int | None = None


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _pick_rrc_group_cols(df: pd.DataFrame) -> list[str] | None:
    return next(
        (
            cols
            for cols in [
                ["time_bucket", "site_id", "sector_id", "cell_id"],
                ["time_bucket", "Node_Cell_ID"],
                ["time_bucket", "cell_id"],
            ]
            if all(col in df.columns for col in cols)
        ),
        None,
    )


def _setup_logger(log_path: Path):
    return future_rules._setup_logger(log_path)


def _assign_current_grid_load_to_cells(enriched: pd.DataFrame, config: CurrentModel3Config) -> pd.DataFrame:
    out = enriched.copy()
    grid_traffic = pd.to_numeric(out.get("estimated_offered_traffic_mbps"), errors="coerce")
    if grid_traffic.notna().sum() == 0:
        grid_traffic = pd.to_numeric(out.get("traffic_demand_est"), errors="coerce")
    grid_users = pd.to_numeric(out.get("estimated_rrc_connected_users"), errors="coerce")
    if grid_users.notna().sum() == 0:
        grid_users = pd.to_numeric(out.get("active_users_est"), errors="coerce")

    out["grid_assigned_traffic_mbps"] = grid_traffic.fillna(0.0).clip(lower=0.0).round(3)
    out["grid_assigned_rrc_users"] = grid_users.fillna(0.0).clip(lower=0.0).round(3)

    cell_agg = (
        out.groupby(["Node_Cell_ID", "time_bucket"], dropna=False, as_index=False)
        .agg(
            cell_assigned_traffic_mbps=("grid_assigned_traffic_mbps", "sum"),
            cell_assigned_rrc_users=("grid_assigned_rrc_users", "sum"),
            cell_capacity_mbps=("estimated_dl_capacity_mbps", "max"),
            cell_serving_grid_count=("grid_id", "nunique"),
        )
    )
    out = out.merge(cell_agg, on=["Node_Cell_ID", "time_bucket"], how="left", validate="many_to_one")
    out["cell_capacity_mbps"] = pd.to_numeric(out["cell_capacity_mbps"], errors="coerce").replace(0.0, np.nan).fillna(0.1)
    out["estimated_prb_utilization_pct"] = (
        (pd.to_numeric(out["cell_assigned_traffic_mbps"], errors="coerce").fillna(0.0) / out["cell_capacity_mbps"]) * 100.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).round(3)
    out["estimated_cell_rrc_connected_users"] = pd.to_numeric(out["cell_assigned_rrc_users"], errors="coerce").fillna(0.0).round(3)
    out["estimated_cell_rrc_utilization_pct"] = (
        (out["estimated_cell_rrc_connected_users"] / float(config.rrc_sector_capacity)) * 100.0
    ).round(3)

    # Preserve grid-level demand while exposing cell-level totals for recommendation logic.
    out["estimated_rrc_connected_users"] = out["grid_assigned_rrc_users"]
    out["estimated_offered_traffic_mbps"] = pd.to_numeric(out["cell_assigned_traffic_mbps"], errors="coerce").fillna(0.0).round(3)
    out["estimated_dl_capacity_mbps"] = out["cell_capacity_mbps"].round(3)
    return out


def _current_dataset_base_context(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        col
        for col in df.columns
        if col
        in {
            "grid_id",
            "time_bucket",
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
            "clutter_class",
            "dominant_band_class",
            "bucket_seq",
            "bucket_min_timestamp",
            "bucket_max_timestamp",
            "bucket_mid_timestamp",
            "estimated_offered_traffic_mbps",
            "estimated_rrc_connected_users",
            "raw_estimated_offered_traffic_mbps",
            "raw_estimated_rrc_connected_users",
            "model3_hotspot_score",
            "model3_hotspot_rank",
        }
    ]
    out = df[keep].drop_duplicates(subset=["grid_id", "time_bucket"]).copy()
    out["grid_id"] = pd.to_numeric(out["grid_id"], errors="coerce").astype("Int64")
    out["time_bucket"] = out["time_bucket"].astype(str)
    return out


def _load_current_context(config: CurrentModel3Config, logger) -> dict[str, Any]:
    archive_path = future_rules.resolve_coverage_artifact_path()
    project_sites_df = future_rules._read_csv_from_archive(archive_path, "project_sites.csv")
    coverage_rows_df = future_rules._read_csv_from_archive(archive_path, "coverage_rows.csv")
    baseline_pred_df = future_rules._read_csv_from_archive(archive_path, "baseline_prediction_grid.csv")
    corrected_pred_df = future_rules._read_csv_from_archive(archive_path, "bucket_corrected_prediction_grid.csv")
    geo_df = future_rules._read_csv_from_archive(archive_path, "bucket_grid_geo_features.csv")
    kpi_df = future_rules._read_csv_from_archive(archive_path, "grid_kpi_timeseries.csv")
    summary = json.loads(
        __import__("subprocess").check_output(
            ["tar", "-xOf", str(archive_path), f"{future_rules._archive_root(archive_path)}/summary.json"],
            text=True,
        )
    )
    for frame in [coverage_rows_df, baseline_pred_df, corrected_pred_df, geo_df, kpi_df]:
        if "time_bucket" in frame.columns:
            frame["time_bucket"] = frame["time_bucket"].astype(str)
    part3_site_map, _ = coverage_test._build_bucket_site_topologies(
        site_df=project_sites_df,
        buckets=coverage_test.DEFAULT_BUCKETS,
        operator_name=str(summary.get("topology_operator") or "Airtel"),
    )
    part3_site_df = part3_site_map.get("PART_3", pd.DataFrame()).copy()
    current_df = pd.read_csv(config.dataset_path)
    current_df["time_bucket"] = current_df["time_bucket"].astype(str)
    current_df["grid_id"] = pd.to_numeric(current_df["grid_id"], errors="coerce").astype("Int64")
    logger.info(
        "current_context archive=%s part3_sites=%d current_rows=%d",
        archive_path,
        len(part3_site_df),
        len(current_df),
    )
    return {
        "archive_path": archive_path,
        "summary": summary,
        "project_sites_df": project_sites_df,
        "coverage_rows_df": coverage_rows_df,
        "baseline_pred_df": baseline_pred_df,
        "corrected_pred_df": corrected_pred_df,
        "geo_df": geo_df,
        "kpi_df": kpi_df,
        "part3_site_df": part3_site_df,
        "building_df": pd.DataFrame(columns=["geometry_wkt"]),
        "current_base_df": _current_dataset_base_context(current_df),
    }


def _build_current_cell_inventory(df: pd.DataFrame, config: CurrentModel3Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    sector_key = future_rules._pick_sector_key(df)
    site_key = future_rules._pick_site_key(df)
    rows: list[dict[str, Any]] = []
    for node_cell_id, group in df.groupby("Node_Cell_ID", dropna=False):
        prb = future_rules._to_num(group["estimated_prb_utilization_pct"])
        rrc = future_rules._to_num(group["estimated_cell_rrc_utilization_pct"])
        users = future_rules._to_num(group["estimated_cell_rrc_connected_users"])
        traffic = future_rules._to_num(group.get("estimated_offered_traffic_mbps", pd.Series(dtype=float)))
        capacity = future_rules._to_num(group.get("estimated_dl_capacity_mbps", pd.Series(dtype=float)))
        row = {
            "Node_Cell_ID": node_cell_id,
            "site_id": future_rules._first_non_empty(group[site_key]) if site_key and site_key in group.columns else "",
            "sector_id": future_rules._first_non_empty(group[sector_key]) if sector_key in group.columns else "",
            "band": future_rules._fmt_band(future_rules._first_non_empty(group.get("topology_band", pd.Series(dtype=object)))),
            "earfcn": future_rules._fmt_band(future_rules._first_non_empty(group.get("topology_earfcn", pd.Series(dtype=object)))),
            "topology_original_cell_id": future_rules._first_non_empty(group.get("topology_original_cell_id", pd.Series(dtype=object))),
            "topology_original_node_cell_id": future_rules._first_non_empty(group.get("topology_original_node_cell_id", pd.Series(dtype=object))),
            "topology_frontend_site_sector_key": future_rules._first_non_empty(group.get("topology_frontend_site_sector_key", pd.Series(dtype=object))),
            "grid_count": int(group["grid_id"].nunique(dropna=True)) if "grid_id" in group.columns else int(len(group)),
            "congested_grid_count": int(((prb > config.congestion_threshold) | (rrc > config.congestion_threshold)).sum()),
            "prb_before_pct": float(prb.max()) if prb.notna().any() else np.nan,
            "prb_p90_pct": float(prb.quantile(0.90)) if prb.notna().any() else np.nan,
            "rrc_before_pct": float(rrc.max()) if rrc.notna().any() else np.nan,
            "rrc_users_before": float(users.max()) if users.notna().any() else np.nan,
            "estimated_offered_traffic_mbps": float(traffic.max()) if traffic.notna().any() else np.nan,
            "estimated_dl_capacity_mbps": float(capacity.max()) if capacity.notna().any() else np.nan,
            "existing_carriers": future_rules._first_non_empty(group.get("existing_carriers", pd.Series(dtype=object))),
            "existing_carrier_count": int(future_rules._to_num(group.get("existing_carrier_count", pd.Series(dtype=float))).max())
            if "existing_carrier_count" in group.columns and future_rules._to_num(group["existing_carrier_count"]).notna().any()
            else 0,
            "available_bands_to_add": future_rules._first_non_empty(group.get("available_bands_to_add", pd.Series(dtype=object))),
            "carrier_addition_options": future_rules._first_non_empty(group.get("carrier_addition_options", pd.Series(dtype=object))),
            "available_earfcns_to_add": future_rules._first_non_empty(group.get("available_earfcns_to_add", pd.Series(dtype=object))),
            "available_earfcn_options": future_rules._first_non_empty(group.get("available_earfcn_options", pd.Series(dtype=object))),
            "recommended_band_to_add": future_rules._first_non_empty(group.get("recommended_band_to_add", pd.Series(dtype=object))),
            "available_band_options_count": int(future_rules._to_num(group.get("available_band_options_count", pd.Series(dtype=float))).max())
            if "available_band_options_count" in group.columns and future_rules._to_num(group["available_band_options_count"]).notna().any()
            else 0,
            "max_supported_carriers": int(future_rules._to_num(group.get("max_supported_carriers", pd.Series(dtype=float))).max())
            if "max_supported_carriers" in group.columns and future_rules._to_num(group["max_supported_carriers"]).notna().any()
            else 0,
            "carrier_addition_possible": bool(group.get("carrier_addition_possible", pd.Series(False)).astype(bool).any()),
            "carrier_addition_blocked": bool(group.get("carrier_addition_blocked", pd.Series(False)).astype(bool).all()),
            "carrier_addition_reason": future_rules._first_non_empty(group.get("carrier_addition_reason", pd.Series(dtype=object))),
            "sector_has_alternate_carrier": bool(group.get("sector_has_alternate_carrier", pd.Series(False)).astype(bool).any()),
            "sector_capacity_limit": int(future_rules._to_num(group.get("sector_capacity_limit", pd.Series(dtype=float))).max())
            if "sector_capacity_limit" in group.columns and future_rules._to_num(group["sector_capacity_limit"]).notna().any()
            else 0,
            "hotspot_score": float(future_rules._to_num(group.get("prb_pressure_est", pd.Series(dtype=float))).max())
            if "prb_pressure_est" in group.columns and future_rules._to_num(group["prb_pressure_est"]).notna().any()
            else np.nan,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"cell_count": 0, "sector_count": 0, "congested_cell_count": 0, "carrier_addition_candidate_count": 0}
    out["prb_rrc_pressure"] = out[["prb_before_pct", "rrc_before_pct"]].max(axis=1)
    out["congested"] = out["prb_rrc_pressure"] > float(config.congestion_threshold)
    sector_counts = out.groupby("sector_id")["Node_Cell_ID"].nunique(dropna=True)
    sector_congested_counts = out.groupby("sector_id")["congested"].sum()
    out["sector_cell_count"] = out["sector_id"].map(sector_counts).fillna(0).astype(int)
    out["sector_congested_count"] = out["sector_id"].map(sector_congested_counts).fillna(0).astype(int)
    summary = {
        "cell_count": int(len(out)),
        "sector_count": int(out["sector_id"].nunique(dropna=True)),
        "congested_cell_count": int(out["congested"].sum()),
        "carrier_addition_candidate_count": int(out["carrier_addition_possible"].sum()),
    }
    return out, summary


def _build_current_inventory_from_surface(
    *,
    baseline_local: pd.DataFrame,
    corrected_local: pd.DataFrame,
    local_site_df: pd.DataFrame,
    local_kpi: pd.DataFrame,
    local_geo: pd.DataFrame,
    context: dict[str, Any],
    config: CurrentModel3Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dense_part3 = future_rules._build_dense_part3_features(
        baseline_pred_df=baseline_local,
        corrected_pred_df=corrected_local,
        kpi_df=local_kpi,
        geo_df=local_geo,
        site_df=local_site_df.drop(columns=["_site_distance_m"], errors="ignore"),
    )
    dense_part3["time_bucket"] = "PART_3"
    dense_part3["grid_id"] = pd.to_numeric(dense_part3["grid_id"], errors="coerce").astype("Int64")

    base_context = context["current_base_df"]
    merged = dense_part3.merge(base_context, on=["grid_id", "time_bucket"], how="left", suffixes=("", "_base"))
    for col in list(merged.columns):
        if col.endswith("_base"):
            original = col[:-5]
            if original in merged.columns:
                merged[original] = merged[original].combine_first(merged[col])
                merged = merged.drop(columns=[col])
            else:
                merged = merged.rename(columns={col: original})
    merged = merged.sort_values(["grid_id", "time_bucket"]).drop_duplicates(subset=["grid_id", "time_bucket"], keep="first").copy()

    topo_work = corrected_local.copy()
    topo_work["grid_id"] = pd.to_numeric(topo_work["grid_id"], errors="coerce").astype("Int64")
    topo_work["_rank_rsrp"] = pd.to_numeric(topo_work.get("pred_rsrp"), errors="coerce")
    topo_best = topo_work.sort_values(["grid_id", "time_bucket", "_rank_rsrp"], ascending=[True, True, False]).drop_duplicates(subset=["grid_id", "time_bucket"], keep="first")
    topo_keep = [
        col
        for col in topo_best.columns
        if col
        in [
            "grid_id",
            "time_bucket",
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
            "carrier_load_share",
            "original_node_cell_id",
            "original_cell_id",
            "canonical_sector_id",
            "site_sector_band_key",
            "rf_identity_key",
        ]
    ]
    topo_best = topo_best[topo_keep].copy()
    topo_best = topo_best.rename(
        columns={
            "Site ID": "topology_site_id",
            "band": "topology_band",
            "earfcn": "topology_earfcn",
            "PCI": "topology_pci",
            "nodeb_id": "topology_nodeb_id",
            "sector": "topology_sector",
            "site_identity_key": "topology_site_identity_key",
            "sector_identity": "topology_sector_identity",
            "sector_identity_key": "topology_sector_identity_key",
            "frontend_site_sector_key": "topology_frontend_site_sector_key",
            "node_cell_sector_key": "topology_node_cell_sector_key",
            "canonical_sector_id": "topology_canonical_sector_id",
            "site_sector_band_key": "topology_site_sector_band_key",
            "rf_identity_key": "topology_rf_identity_key",
            "original_node_cell_id": "topology_original_node_cell_id",
            "original_cell_id": "topology_original_cell_id",
            "carrier_load_share": "topology_carrier_load_share",
        }
    )
    merged = merged.merge(topo_best, on=["grid_id", "time_bucket"], how="left", validate="many_to_one")
    enriched = future_builder.model2_builder._add_model2_features(merged)
    enriched, _ = future_builder._add_sector_carrier_capability_fields(enriched)

    bandwidth_mhz = pd.to_numeric(enriched.get("bandwidth_mhz_est"), errors="coerce").replace(0, np.nan).fillna(10.0)
    spectral = future_builder._estimate_spectral_efficiency_bpshz(
        enriched,
        bandwidth_mhz=bandwidth_mhz,
        mimo_layers=future_builder.DEFAULT_MIMO_LAYERS,
        control_overhead=future_builder.DEFAULT_CONTROL_OVERHEAD,
    )
    enriched["estimated_spectral_efficiency_bpshz"] = spectral.round(6)
    enriched["estimated_dl_capacity_mbps"] = (
        bandwidth_mhz * spectral * max(1.0, float(future_builder.DEFAULT_MIMO_LAYERS)) * max(0.1, 1.0 - float(future_builder.DEFAULT_CONTROL_OVERHEAD))
    ).clip(lower=0.1).round(3)
    enriched = _assign_current_grid_load_to_cells(enriched, config)
    cell_inventory, _ = _build_current_cell_inventory(enriched, config)
    return enriched, cell_inventory


def _choose_load_balance_candidate_current(sector_cells: pd.DataFrame, threshold: float) -> dict[str, Any] | None:
    if len(sector_cells) < 2:
        return None
    ordered = sector_cells.sort_values(["prb_rrc_pressure", "grid_count"], ascending=[False, False], na_position="last").reset_index(drop=True)
    source = ordered.iloc[0]
    peers = ordered.iloc[1:].copy()
    source_prb = float(source["prb_before_pct"]) if pd.notna(source["prb_before_pct"]) else np.nan
    source_rrc = float(source["rrc_before_pct"]) if pd.notna(source["rrc_before_pct"]) else np.nan
    source_prb_bad = bool(pd.notna(source_prb) and source_prb > threshold)
    source_rrc_bad = bool(pd.notna(source_rrc) and source_rrc > threshold)

    peers["prb_headroom"] = threshold - pd.to_numeric(peers["prb_before_pct"], errors="coerce")
    peers["rrc_headroom"] = threshold - pd.to_numeric(peers["rrc_before_pct"], errors="coerce")
    if source_prb_bad and not source_rrc_bad:
        peers = peers.loc[peers["prb_headroom"] > 0].copy()
        peers["lb_score"] = peers["prb_headroom"] + (0.10 * peers["rrc_headroom"].clip(lower=-200.0))
        congestion_mode = "PRB_ONLY"
    elif source_rrc_bad and not source_prb_bad:
        peers = peers.loc[peers["rrc_headroom"] > 0].copy()
        peers["lb_score"] = peers["rrc_headroom"] + (0.10 * peers["prb_headroom"].clip(lower=-200.0))
        congestion_mode = "RRC_ONLY"
    else:
        peers = peers.loc[(peers["prb_headroom"] > 0) | (peers["rrc_headroom"] > 0)].copy()
        peers["lb_score"] = peers["prb_headroom"].clip(lower=0.0) + peers["rrc_headroom"].clip(lower=0.0)
        congestion_mode = "PRB_AND_RRC"
    if peers.empty:
        return None
    peers = peers.sort_values(["lb_score", "estimated_dl_capacity_mbps", "grid_count"], ascending=[False, False, False], na_position="last")
    target = peers.iloc[0]
    return {
        "source_node_cell_id": str(source["Node_Cell_ID"]),
        "target_node_cell_id": str(target["Node_Cell_ID"]),
        "target_band": str(target["band"]),
        "source_row": source,
        "target_row": target,
        "congestion_mode": congestion_mode,
    }


def _run_load_balance_current(
    sector_cells: pd.DataFrame,
    config: CurrentModel3Config,
) -> dict[str, Any]:
    candidate = _choose_load_balance_candidate_current(sector_cells, config.congestion_threshold)
    if candidate is None:
        return {
            "status": "No Material Change",
            "selected_peer_node_cell_id": "",
            "selected_peer_band": "",
            "projected_prb_after_pct": np.nan,
            "projected_rrc_after_pct": np.nan,
            "projected_rrc_users_after": np.nan,
            "action_reason": "No same-sector carrier met the metric-aware current headroom check for the congested metric.",
            "next_step": "Add Carrier",
            "resimulation_required": False,
            "resimulation_flow": "Deterministic current-state load redistribution only",
            "should_fallthrough": True,
        }
    source = candidate["source_row"]
    target = candidate["target_row"]
    congestion_mode = str(candidate.get("congestion_mode") or "PRB_AND_RRC")
    source_traffic = float(source["estimated_offered_traffic_mbps"]) if pd.notna(source["estimated_offered_traffic_mbps"]) else 0.0
    target_traffic = float(target["estimated_offered_traffic_mbps"]) if pd.notna(target["estimated_offered_traffic_mbps"]) else 0.0
    source_users = float(source["rrc_users_before"]) if pd.notna(source["rrc_users_before"]) else 0.0
    target_users = float(target["rrc_users_before"]) if pd.notna(target["rrc_users_before"]) else 0.0
    source_cap = max(0.1, float(source["estimated_dl_capacity_mbps"])) if pd.notna(source["estimated_dl_capacity_mbps"]) else 0.1
    target_cap = max(0.1, float(target["estimated_dl_capacity_mbps"])) if pd.notna(target["estimated_dl_capacity_mbps"]) else 0.1
    threshold = float(config.congestion_threshold)

    source_prb = float(source["prb_before_pct"]) if pd.notna(source["prb_before_pct"]) else 0.0
    source_rrc = float(source["rrc_before_pct"]) if pd.notna(source["rrc_before_pct"]) else 0.0

    need_share_prb = max(0.0, (source_traffic - ((threshold / 100.0) * source_cap)) / max(source_traffic, 1e-6))
    need_share_rrc = max(0.0, (source_users - ((threshold / 100.0) * float(config.rrc_sector_capacity))) / max(source_users, 1e-6))
    max_share_target_prb = max(0.0, ((((threshold / 100.0) * target_cap) - target_traffic) / max(source_traffic, 1e-6))) if source_traffic > 0 else 0.0
    max_share_target_rrc = max(0.0, ((((threshold / 100.0) * float(config.rrc_sector_capacity)) - target_users) / max(source_users, 1e-6))) if source_users > 0 else 0.0

    if congestion_mode == "PRB_ONLY":
        desired_share = need_share_prb
        allowed_share = max_share_target_prb
    elif congestion_mode == "RRC_ONLY":
        desired_share = need_share_rrc
        allowed_share = max_share_target_rrc
    else:
        desired_share = max(need_share_prb, need_share_rrc)
        allowed_share = min(
            max_share_target_prb if source_traffic > 0 else 1.0,
            max_share_target_rrc if source_users > 0 else 1.0,
        )

    move_share = min(0.60, max(0.0, min(desired_share, allowed_share)))
    if move_share <= 0.0:
        return {
            "status": "Rejected",
            "selected_peer_node_cell_id": str(target["Node_Cell_ID"]),
            "selected_peer_band": str(target["band"]),
            "projected_prb_after_pct": np.nan,
            "projected_rrc_after_pct": np.nan,
            "projected_rrc_users_after": np.nan,
            "action_reason": f"Peer carrier exists, but the metric-aware headroom check found no safe movable share for {congestion_mode}.",
            "next_step": "Add Carrier",
            "resimulation_required": False,
            "resimulation_flow": "Deterministic current-state load redistribution only",
            "should_fallthrough": True,
        }
    moved_traffic = source_traffic * move_share
    moved_users = source_users * move_share
    source_prb_after = ((source_traffic - moved_traffic) / source_cap) * 100.0
    target_prb_after = ((target_traffic + moved_traffic) / target_cap) * 100.0
    source_rrc_after = ((source_users - moved_users) / float(config.rrc_sector_capacity)) * 100.0
    target_rrc_after = ((target_users + moved_users) / float(config.rrc_sector_capacity)) * 100.0
    projected_prb = max(source_prb_after, target_prb_after)
    projected_rrc = max(source_rrc_after, target_rrc_after)
    status = "Resolved" if projected_prb <= threshold and projected_rrc <= threshold else "Partially Resolved"
    if projected_prb >= max(float(source["prb_before_pct"]), float(target["prb_before_pct"])) and projected_rrc >= max(float(source["rrc_before_pct"]), float(target["rrc_before_pct"])):
        status = "Rejected"
    return {
        "status": status,
        "selected_peer_node_cell_id": str(target["Node_Cell_ID"]),
        "selected_peer_band": str(target["band"]),
        "projected_prb_after_pct": round(projected_prb, 3),
        "projected_rrc_after_pct": round(projected_rrc, 3),
        "projected_rrc_users_after": round(max(source_users - moved_users, target_users + moved_users), 3),
        "action_reason": f"Current-state load balancing used a metric-aware {congestion_mode} move share of {move_share:.3f}.",
        "next_step": "" if status == "Resolved" else "Add Carrier",
        "resimulation_required": False,
        "resimulation_flow": "Deterministic current-state load redistribution only",
        "should_fallthrough": status != "Resolved",
    }


def _rerun_current_topology(
    *,
    sector_cells: pd.DataFrame,
    config: CurrentModel3Config,
    context: dict[str, Any],
    logger,
    action_label: str,
    site_df: pd.DataFrame,
    source_rows: pd.DataFrame,
) -> dict[str, Any]:
    local_ctx = future_rules._prepare_local_action_context(
        sector_cells=sector_cells,
        site_df=site_df,
        context=context,
        config=future_rules.Model3RecommendationConfig(
            dataset_path=config.dataset_path,
            summary_path=config.summary_path,
            output_root=config.output_root,
            stable_output_dir=config.stable_output_dir,
            congestion_threshold=config.congestion_threshold,
            rrc_sector_capacity=config.rrc_sector_capacity,
            sector_split_local_radius_m=config.sector_split_local_radius_m,
        ),
        logger=logger,
        action_label=action_label,
        source_rows=source_rows,
    )
    if local_ctx is None:
        return {
            "status": "Recommended",
            "action_reason": f"{action_label} local context could not be prepared from PART_3 artifacts.",
            "projected_prb_after_pct": np.nan,
            "projected_rrc_after_pct": np.nan,
            "projected_rrc_users_after": np.nan,
            "next_step": "New Site",
            "resimulation_required": True,
            "resimulation_flow": f"{action_label} baseline rerun failed during local context preparation",
        }
    with open(logger.handlers[0].baseFilename, "a", encoding="utf-8") as log_stream:
        with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
            baseline_local = coverage_test._run_project_baseline_prediction(
                project_id=int(context["summary"].get("project_id", 196)),
                region=str(context["summary"].get("region", "india")).lower(),
                site_df=local_ctx["local_site_df"].drop(columns=["_site_distance_m"], errors="ignore"),
                drive_df=local_ctx["local_detail"],
                building_df=context["building_df"],
                baseline_radius_m=float(context["summary"].get("baseline_radius_m", 500.0)),
                grid_size_m=float(context["summary"].get("grid_size_m", 50.0)),
                workers=1,
                max_interference_sites=int(context["summary"].get("max_interference_sites", 50)),
                polygon_wkt=str(context["summary"].get("polygon_wkt", "")).strip() or None,
                use_frontend_grid_sampling=False,
                grid_analytics_scenario_id=context["summary"].get("grid_analytics_scenario_id"),
            )
            corrected_local = coverage_test._run_bucket_corrected_predictions(
                baseline_pred_df=baseline_local,
                detail_df=local_ctx["local_detail"].assign(time_bucket="PART_3"),
                site_df_by_bucket={"PART_3": local_ctx["local_site_df"].drop(columns=["_site_distance_m"], errors="ignore")},
                building_df=context["building_df"],
                project_id=int(context["summary"].get("project_id", 196)),
                region=str(context["summary"].get("region", "india")).lower(),
                grid_size_m=float(context["summary"].get("grid_size_m", 50.0)),
                buckets=[("PART_3", "2026-02-11 00:00:00", "2026-05-16 23:59:59")],
                polygon_wkt=str(context["summary"].get("polygon_wkt", "")).strip() or None,
            )
    baseline_local["time_bucket"] = "PART_3"
    baseline_local = future_rules._merge_point_map_by_coordinates(
        baseline_local,
        local_ctx["local_point_map"].drop(columns=["_site_distance_m"], errors="ignore"),
    )
    baseline_local = future_rules._ensure_grid_group_columns(baseline_local)
    if corrected_local.empty:
        corrected_local = baseline_local.copy()
    corrected_local["time_bucket"] = "PART_3"
    corrected_local = future_rules._merge_point_map_by_coordinates(
        corrected_local,
        local_ctx["local_point_map"].drop(columns=["_site_distance_m"], errors="ignore"),
    )
    corrected_local = future_rules._ensure_grid_group_columns(corrected_local)
    _, local_inventory = _build_current_inventory_from_surface(
        baseline_local=baseline_local,
        corrected_local=corrected_local,
        local_site_df=local_ctx["local_site_df"],
        local_kpi=local_ctx["local_kpi"],
        local_geo=local_ctx["local_geo"],
        context=context,
        config=config,
    )
    outcome = future_rules._evaluate_action_inventory(
        sector_cells=sector_cells,
        after_inventory=local_inventory,
        config=future_rules.Model3RecommendationConfig(
            dataset_path=config.dataset_path,
            summary_path=config.summary_path,
            output_root=config.output_root,
            stable_output_dir=config.stable_output_dir,
            congestion_threshold=config.congestion_threshold,
            rrc_sector_capacity=config.rrc_sector_capacity,
        ),
        logger=logger,
        action_label=action_label,
    )
    return {
        "status": outcome["status"],
        "action_reason": f"{action_label} reran PART_3 baseline and recalculated current-state KPIs deterministically.",
        "projected_prb_after_pct": outcome["projected_prb_after_pct"],
        "projected_rrc_after_pct": outcome["projected_rrc_after_pct"],
        "projected_rrc_users_after": outcome["projected_rrc_users_after"],
        "next_step": outcome["next_step"] or "New Site",
        "resimulation_required": True,
        "resimulation_flow": f"{action_label} topology -> baseline rerun -> deterministic current KPI rebuild -> Model 3 reevaluation",
    }


def _simulate_current_recommendation(
    sector_cells: pd.DataFrame,
    config: CurrentModel3Config,
    logger,
    context: dict[str, Any],
) -> dict[str, Any]:
    source_row = sector_cells.sort_values(["prb_rrc_pressure", "grid_count"], ascending=[False, False], na_position="last").iloc[0]
    source_prb = float(source_row["prb_before_pct"]) if pd.notna(source_row["prb_before_pct"]) else np.nan
    source_rrc = float(source_row["rrc_before_pct"]) if pd.notna(source_row["rrc_before_pct"]) else np.nan
    pressure = max(source_prb if pd.notna(source_prb) else 0.0, source_rrc if pd.notna(source_rrc) else 0.0)
    congested = pressure > config.congestion_threshold
    if not congested:
        return {
            "action": "Healthy",
            "status": "Healthy",
            "decision_path": "No action",
            "attempted_actions": "",
            "load_balance_possible": False,
            "selected_peer_node_cell_id": "",
            "selected_peer_band": "",
            "projected_prb_after_pct": source_prb,
            "projected_rrc_after_pct": source_rrc,
            "projected_rrc_users_after": float(source_row["rrc_users_before"]) if pd.notna(source_row["rrc_users_before"]) else np.nan,
            "action_reason": "Healthy.",
            "next_step": "",
            "resimulation_required": False,
            "resimulation_flow": "",
        }

    attempted_actions: list[str] = []

    load_result = _run_load_balance_current(sector_cells, config)
    if load_result["selected_peer_node_cell_id"]:
        attempted_actions.append(f"Load Balance[{load_result['status']}]")
    if load_result["selected_peer_node_cell_id"] and not bool(load_result.get("should_fallthrough")):
        return {
            "action": f"Load Balance -> {load_result['selected_peer_band']} MHz",
            "decision_path": "Congested | Current load balance branch",
            "attempted_actions": " -> ".join(attempted_actions),
            "load_balance_possible": True,
            **load_result,
        }

    can_add = bool(sector_cells["carrier_addition_possible"].any()) and not bool(sector_cells["carrier_addition_blocked"].all())
    carrier_add_attempted = False
    add_row = sector_cells.loc[sector_cells["carrier_addition_possible"] & ~sector_cells["carrier_addition_blocked"]].sort_values(
        ["prb_rrc_pressure", "grid_count"], ascending=[False, False], na_position="last"
    )
    if can_add and not add_row.empty and str(add_row.iloc[0]["recommended_band_to_add"]).strip():
        band = str(add_row.iloc[0]["recommended_band_to_add"]).strip()
        carrier_add_attempted = True
        site_df, source_rows = future_rules._build_carrier_addition_topology(sector_cells, context["part3_site_df"], band, logger)
        resim = _rerun_current_topology(
            sector_cells=sector_cells,
            config=config,
            context=context,
            logger=logger,
            action_label="current_carrier_add",
            site_df=site_df,
            source_rows=source_rows,
        )
        attempted_actions.append(f"Add Carrier {band}[{resim['status']}]")
        if str(resim["status"]) == "Resolved":
            return {
                "action": f"Add Carrier -> {band} MHz",
                "decision_path": "Congested | Carrier addition branch",
                "attempted_actions": " -> ".join(attempted_actions),
                "load_balance_possible": False,
                "selected_peer_node_cell_id": str(source_row["Node_Cell_ID"]),
                "selected_peer_band": "",
                **resim,
            }

    sector_split_possible = bool(
        int(source_row["sector_cell_count"]) >= 1
        and (not can_add or carrier_add_attempted)
        and (
            int(source_row["sector_congested_count"]) >= 2
            or (
                pressure >= 95.0
                and int(source_row["existing_carrier_count"]) >= int(source_row["max_supported_carriers"])
            )
            or bool(sector_cells["carrier_addition_blocked"].all())
        )
    )
    if sector_split_possible:
        site_df, source_rows = future_rules._build_sector_split_topology(sector_cells, context["part3_site_df"], logger)
        resim = _rerun_current_topology(
            sector_cells=sector_cells,
            config=config,
            context=context,
            logger=logger,
            action_label="current_sector_split",
            site_df=site_df,
            source_rows=source_rows,
        )
        attempted_actions.append(f"Sector Split[{resim['status']}]")
        if str(resim["status"]) == "Resolved":
            if resim["status"] == "Rejected":
                next_step = "New Site"
            else:
                next_step = resim["next_step"]
            return {
                "action": "Sector Split",
                "decision_path": "Congested | Sector split branch",
                "attempted_actions": " -> ".join(attempted_actions),
                "load_balance_possible": False,
                "selected_peer_node_cell_id": str(source_row["Node_Cell_ID"]),
                "selected_peer_band": "",
                **{**resim, "next_step": next_step},
            }

    site_df, source_rows = future_rules._build_new_site_topology(sector_cells, context["part3_site_df"], logger)
    resim = _rerun_current_topology(
        sector_cells=sector_cells,
        config=config,
        context=context,
        logger=logger,
        action_label="current_new_site",
        site_df=site_df,
        source_rows=source_rows,
    )
    attempted_actions.append(f"New Site[{resim['status']}]")
    return {
        "action": "New Site",
        "decision_path": "Congested | New site branch",
        "attempted_actions": " -> ".join(attempted_actions),
        "load_balance_possible": False,
        "selected_peer_node_cell_id": str(source_row["Node_Cell_ID"]),
        "selected_peer_band": "",
        **resim,
    }


def _build_recommendations(cell_df: pd.DataFrame, config: CurrentModel3Config, logger, context: dict[str, Any]) -> pd.DataFrame:
    rows = []
    sector_groups = []
    for sector_id, sector_cells in cell_df.groupby("sector_id", dropna=False):
        sector_groups.append((sector_id, sector_cells.copy()))
    sector_groups.sort(
        key=lambda item: float(item[1]["prb_rrc_pressure"].max()) if item[1]["prb_rrc_pressure"].notna().any() else -np.inf,
        reverse=True,
    )
    if config.max_sectors is not None:
        sector_groups = sector_groups[: max(0, int(config.max_sectors))]
        logger.info("sector_scope limited_to_top=%d", len(sector_groups))
    for sector_id, sector_cells in sector_groups:
        sector_cells = sector_cells.sort_values(["congested", "prb_rrc_pressure", "grid_count"], ascending=[False, False, False], na_position="last").copy()
        if not bool(sector_cells["congested"].any()):
            continue
        rec = _simulate_current_recommendation(sector_cells, config, logger, context)
        lead_row = sector_cells.iloc[0]
        sector_congested_ids = sector_cells.loc[sector_cells["congested"], "Node_Cell_ID"].astype(str).tolist()
        rows.append(
            {
                "site_id": lead_row["site_id"],
                "sector_id": sector_id,
                "Node_Cell_ID": lead_row["Node_Cell_ID"],
                "sector_congested_node_cell_ids": ", ".join(sector_congested_ids),
                "band": lead_row["band"],
                "earfcn": lead_row["earfcn"],
                "grid_count": int(sector_cells["grid_count"].sum()),
                "congested_grid_count": int(sector_cells["congested_grid_count"].sum()),
                "prb_before_pct": round(float(lead_row["prb_before_pct"]), 3) if pd.notna(lead_row["prb_before_pct"]) else np.nan,
                "rrc_before_pct": round(float(lead_row["rrc_before_pct"]), 3) if pd.notna(lead_row["rrc_before_pct"]) else np.nan,
                "rrc_users_before": round(float(lead_row["rrc_users_before"]), 3) if pd.notna(lead_row["rrc_users_before"]) else np.nan,
                "action": rec["action"],
                "status": rec["status"],
                "decision_path": rec["decision_path"],
                "attempted_actions": rec.get("attempted_actions", ""),
                "load_balance_possible": bool(rec["load_balance_possible"]),
                "selected_peer_node_cell_id": rec["selected_peer_node_cell_id"],
                "selected_peer_band": rec["selected_peer_band"],
                "recommended_band_to_add": lead_row["recommended_band_to_add"],
                "available_bands_to_add": lead_row["available_bands_to_add"],
                "carrier_addition_possible": bool(sector_cells["carrier_addition_possible"].any()),
                "carrier_addition_blocked": bool(sector_cells["carrier_addition_blocked"].all()),
                "max_supported_carriers": int(sector_cells["max_supported_carriers"].max()),
                "existing_carrier_count": int(sector_cells["existing_carrier_count"].max()),
                "existing_carriers": lead_row["existing_carriers"],
                "projected_prb_after_pct": rec["projected_prb_after_pct"],
                "projected_rrc_after_pct": rec["projected_rrc_after_pct"],
                "projected_rrc_users_after": rec["projected_rrc_users_after"],
                "action_reason": rec["action_reason"],
                "next_step": rec["next_step"],
                "resimulation_required": bool(rec["resimulation_required"]),
                "resimulation_flow": rec["resimulation_flow"],
            }
        )
    reco_df = pd.DataFrame(rows)
    if reco_df.empty:
        return reco_df
    reco_df["priority_score"] = reco_df[["prb_before_pct", "rrc_before_pct"]].max(axis=1)
    return reco_df.sort_values(["priority_score", "congested_grid_count", "grid_count"], ascending=[False, False, False], na_position="last").reset_index(drop=True)


def _write_workbook(run_dir: Path, cell_df: pd.DataFrame, reco_df: pd.DataFrame, sector_df: pd.DataFrame, summary: dict[str, Any]) -> Path:
    wb = Workbook()
    wb.remove(wb.active)
    ws_summary = wb.create_sheet("Summary")
    summary_rows = pd.DataFrame(
        [
            {"metric": "rows_in_model3_dataset", "value": summary["rows_in_model3_dataset"]},
            {"metric": "congested_cell_rows", "value": summary["congested_cell_rows"]},
            {"metric": "sector_rows", "value": summary["sector_rows"]},
            {"metric": "threshold", "value": summary["threshold"]},
            {"metric": "runtime_sec", "value": summary["runtime_sec"]},
        ]
    )
    future_rules._write_df_sheet(ws_summary, summary_rows)
    ws_reco = wb.create_sheet("Recommendations")
    future_rules._write_df_sheet(ws_reco, reco_df)
    future_rules._style_recommendation_sheet(ws_reco, reco_df)
    ws_cells = wb.create_sheet("Congested Cells")
    future_rules._write_df_sheet(ws_cells, cell_df.loc[cell_df["congested"]].copy())
    ws_sector = wb.create_sheet("Sector Inventory")
    future_rules._write_df_sheet(ws_sector, sector_df)
    workbook_path = run_dir / "model3_current_recommendations.xlsx"
    wb.save(workbook_path)
    return workbook_path


def run_model3_current_recommendation_test(config: CurrentModel3Config) -> Path:
    start = time.perf_counter()
    if not config.dataset_path.exists():
        current_builder.build_model3_current_dataset(config.rrc_sector_capacity)
    run_dir = _ensure_dir(config.output_root / f"model3_current_{_timestamp()}")
    log_path = run_dir / "log.txt"
    logger = _setup_logger(log_path)
    logger.info("start dataset=%s summary=%s", config.dataset_path, config.summary_path)
    source_df = pd.read_csv(config.dataset_path)
    cell_inventory, inventory_summary = _build_current_cell_inventory(source_df, config)
    context = _load_current_context(config, logger)
    recommendations = _build_recommendations(cell_inventory, config, logger, context)
    sector_inventory = future_rules._build_sector_inventory(cell_inventory)
    runtime_sec = time.perf_counter() - start
    summary = {
        "mode": "current_model3_recommendation",
        "dataset_path": str(config.dataset_path),
        "summary_path": str(config.summary_path),
        "rows_in_model3_dataset": int(len(cell_inventory)),
        "congested_cell_rows": int(len(recommendations)),
        "sector_rows": int(len(sector_inventory)),
        "inventory_summary": inventory_summary,
        "threshold": float(config.congestion_threshold),
        "max_sectors": int(config.max_sectors) if config.max_sectors is not None else None,
        "runtime_sec": round(float(runtime_sec), 4),
        "artifacts": {
            "recommendations_csv": str(run_dir / "model3_current_recommendations.csv"),
            "cell_inventory_csv": str(run_dir / "model3_current_cell_inventory.csv"),
            "sector_inventory_csv": str(run_dir / "model3_current_sector_inventory.csv"),
            "summary_json": str(run_dir / "summary.json"),
            "log": str(log_path),
        },
    }
    workbook_path = _write_workbook(run_dir, cell_inventory, recommendations, sector_inventory, summary)
    summary["workbook_path"] = str(workbook_path)
    summary["artifacts"]["workbook"] = str(workbook_path)
    recommendations.to_csv(run_dir / "model3_current_recommendations.csv", index=False)
    cell_inventory.to_csv(run_dir / "model3_current_cell_inventory.csv", index=False)
    sector_inventory.to_csv(run_dir / "model3_current_sector_inventory.csv", index=False)
    _save_json(run_dir / "summary.json", summary)
    config.stable_output_dir.mkdir(parents=True, exist_ok=True)
    for src, dest_name in [
        (run_dir / "summary.json", "model3_current_recommendation_summary.json"),
        (run_dir / "model3_current_recommendations.csv", "model3_current_recommendations.csv"),
        (workbook_path, "model3_current_recommendations.xlsx"),
        (log_path, "model3_current_recommendation.log"),
    ]:
        try:
            shutil.copy2(src, config.stable_output_dir / dest_name)
        except PermissionError:
            pass
    print(json.dumps(summary, indent=2, default=str))
    return run_dir


def parse_args() -> CurrentModel3Config:
    parser = argparse.ArgumentParser(description="Run current-state Model 3 recommendations without Model 1/2 inference.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stable-output-dir", type=Path, default=DEFAULT_STABLE_OUTPUT_DIR)
    parser.add_argument("--congestion-threshold", type=float, default=DEFAULT_CONGESTION_THRESHOLD)
    parser.add_argument("--rrc-sector-capacity", type=float, default=future_builder.DEFAULT_RRC_SECTOR_CAPACITY)
    parser.add_argument("--max-sectors", type=int, default=None)
    args = parser.parse_args()
    return CurrentModel3Config(
        dataset_path=args.dataset_path,
        summary_path=args.summary_path,
        output_root=args.output_root,
        stable_output_dir=args.stable_output_dir,
        congestion_threshold=args.congestion_threshold,
        rrc_sector_capacity=args.rrc_sector_capacity,
        max_sectors=args.max_sectors,
    )


if __name__ == "__main__":
    run_model3_current_recommendation_test(parse_args())
