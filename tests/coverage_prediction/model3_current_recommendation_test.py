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
    carrier_reselection_hysteresis_db: float = 3.0


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

    # Every identity string a PRE-EXISTING cell is known by, across all naming layers
    # (site table vs prediction surface use different formats for the same cell).
    # A cell seen in a local rerun whose ID is NOT in this universe must be a synthetic
    # action candidate (carrier addition / new site / sector split child), no matter how
    # the prediction tooling mangled its name.
    original_topology_cell_ids: set[str] = set()
    for col in ["Node_Cell_ID", "cell_id"]:
        if col in part3_site_df.columns:
            original_topology_cell_ids.update(part3_site_df[col].dropna().astype(str).str.strip())
    for col in ["Node_Cell_ID", "original_node_cell_id", "original_cell_id"]:
        if col in corrected_pred_df.columns:
            original_topology_cell_ids.update(corrected_pred_df[col].dropna().astype(str).str.strip())
    original_topology_cell_ids.discard("")

    logger.info(
        "current_context archive=%s part3_sites=%d current_rows=%d original_cell_id_universe=%d",
        archive_path,
        len(part3_site_df),
        len(current_df),
        len(original_topology_cell_ids),
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
        "current_df": current_df,
        "original_topology_cell_ids": original_topology_cell_ids,
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


def _select_serving_cell_with_hysteresis(
    topo_work: pd.DataFrame, hysteresis_db: float, original_cell_ids: set[str]
) -> pd.DataFrame:
    """Pick one serving cell per grid/time_bucket, but require a synthetic action
    candidate (carrier addition / new site / sector split child) to beat the grid's
    currently-serving original cell by more than `hysteresis_db` before the grid is
    allowed to reselect onto it.

    Synthetic-vs-original is decided by ID-universe membership, not name patterns:
    the prediction tooling mangles cell names between layers (site table
    '901053_1__ADD900' comes back as '1_ADD900'; pre-existing multiband cells
    '364_1__MB850' come back as '1_MB850'), so a suffix regex silently matches
    nothing. A candidate is original iff ANY of its identity columns (Node_Cell_ID,
    original_node_cell_id, original_cell_id) appears in `original_cell_ids` - the set
    of every identity string the pre-action network is known by.

    Without this gate, a colocated synthetic candidate (same site/power/height as the
    original, only a different band or a few hundred meters away) can win the plain
    best-RSRP comparison for every grid in the local resimulation footprint at once,
    causing the original cell to drop from ~95% PRB to 0% and dumping the entire load
    onto the new cell in one all-or-nothing swing instead of a realistic partial
    reselection.
    """
    work = topo_work.copy()
    is_original = pd.Series(False, index=work.index)
    for col in ["Node_Cell_ID", "original_node_cell_id", "original_cell_id"]:
        if col in work.columns:
            is_original = is_original | work[col].astype(str).str.strip().isin(original_cell_ids)
    work["_is_synthetic_candidate"] = ~is_original
    ranked = work.sort_values(["grid_id", "time_bucket", "_rank_rsrp"], ascending=[True, True, False])

    best_overall = ranked.drop_duplicates(subset=["grid_id", "time_bucket"], keep="first")
    best_original = ranked.loc[~ranked["_is_synthetic_candidate"]].drop_duplicates(
        subset=["grid_id", "time_bucket"], keep="first"
    )
    if best_original.empty:
        return best_overall.drop(columns=["_is_synthetic_candidate"], errors="ignore")

    prior_rsrp = best_original.set_index(["grid_id", "time_bucket"])["_rank_rsrp"]
    overall_keys = pd.MultiIndex.from_frame(best_overall[["grid_id", "time_bucket"]])
    prior_for_overall = prior_rsrp.reindex(overall_keys).to_numpy()

    stick_to_original = (
        best_overall["_is_synthetic_candidate"].to_numpy()
        & ~pd.isna(prior_for_overall)
        & (pd.to_numeric(best_overall["_rank_rsrp"], errors="coerce").to_numpy() < (prior_for_overall + hysteresis_db))
    )

    if stick_to_original.any():
        switch_keys = best_overall.loc[stick_to_original, ["grid_id", "time_bucket"]]
        fallback_rows = best_original.merge(switch_keys, on=["grid_id", "time_bucket"], how="inner")
        best_overall = pd.concat(
            [best_overall.loc[~stick_to_original], fallback_rows], ignore_index=True, sort=False
        )
    return best_overall.drop(columns=["_is_synthetic_candidate"], errors="ignore")


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
    topo_best = _select_serving_cell_with_hysteresis(
        topo_work, config.carrier_reselection_hysteresis_db, context.get("original_topology_cell_ids", set())
    )
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
    # Grids with no serving candidate in the corrected surface have no Node_Cell_ID;
    # keeping them would pool all their demand into a single phantom NaN "cell".
    enriched = enriched.loc[enriched["Node_Cell_ID"].notna()].copy()
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


_LOCAL_GRID_IDENTITY_COLS = ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon"]
_EARTH_RADIUS_M = 6371000.0


def _force_master_grid_identity(pred_df: pd.DataFrame, point_map: pd.DataFrame, tolerance_m: float) -> pd.DataFrame:
    """Give a locally-resimulated prediction surface the MASTER archive's grid numbering.

    The RF rerun (and `_ensure_grid_group_columns`) number grids by factorizing the
    run's own lat/lon pairs into sequential codes, so "grid 2600" in a local rerun is
    a different physical place than the master archive's grid 2600 - yet the codes
    overlap numerically. Downstream joins (demand attach, lineage lookup, before/after
    comparison) all key on grid_id, so without this the master demand lands on the
    wrong locations and every derived PRB number is physically meaningless.

    Exact lat/lon joins cannot align the two surfaces either: the RF engine lays out
    its own sampling lattice per run with a run-specific origin, so local prediction
    points never coincide exactly with master points. Instead, snap each local point
    to its nearest master point within `tolerance_m` (about half a grid cell) and
    adopt that master point's full grid identity. Points with no master point within
    tolerance are discarded - the master surface doesn't know them, no demand exists
    for them, and they only served as interference context during the RF run.
    """
    from sklearn.neighbors import BallTree

    if pred_df.empty or point_map.empty:
        return future_rules._ensure_grid_group_columns(pred_df.copy())
    work = pred_df.drop(columns=_LOCAL_GRID_IDENTITY_COLS, errors="ignore").copy()
    if not {"lat", "lon"}.issubset(work.columns):
        return future_rules._ensure_grid_group_columns(work)

    pm = point_map.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    work_lat = pd.to_numeric(work["lat"], errors="coerce")
    work_lon = pd.to_numeric(work["lon"], errors="coerce")
    valid = work_lat.notna() & work_lon.notna()
    work = work.loc[valid].copy()
    if work.empty:
        return future_rules._ensure_grid_group_columns(work)

    tree = BallTree(np.radians(pm[["lat", "lon"]].astype(float).to_numpy()), metric="haversine")
    dist_rad, nearest_idx = tree.query(np.radians(work[["lat", "lon"]].astype(float).to_numpy()), k=1)
    dist_m = dist_rad[:, 0] * _EARTH_RADIUS_M
    within = dist_m <= float(tolerance_m)
    work = work.loc[within].copy()
    if work.empty:
        return future_rules._ensure_grid_group_columns(work)
    matched = pm.iloc[nearest_idx[within, 0]].reset_index(drop=True)

    identity_cols = [c for c in _LOCAL_GRID_IDENTITY_COLS if c in matched.columns]
    work = work.reset_index(drop=True)
    for col in identity_cols:
        work[col] = matched[col].to_numpy()
    return future_rules._ensure_grid_group_columns(work)


def _affected_lineage_grid_ids(sector_cells: pd.DataFrame, current_df: pd.DataFrame) -> list[int]:
    """Grid ids that the archive's full-network surface actually assigned to the
    cells being acted on, independent of distance from the site coordinate."""
    lineage_ids: set[str] = set()
    for col in ["Node_Cell_ID", "topology_original_cell_id", "topology_original_node_cell_id"]:
        if col in sector_cells.columns:
            lineage_ids.update(
                str(v).strip() for v in sector_cells[col].dropna().astype(str).tolist() if str(v).strip()
            )
    if not lineage_ids or current_df.empty:
        return []
    mask = pd.Series(False, index=current_df.index)
    for col in ["Node_Cell_ID", "topology_original_cell_id", "topology_original_node_cell_id"]:
        if col in current_df.columns:
            mask = mask | current_df[col].astype(str).isin(lineage_ids)
    return pd.to_numeric(current_df.loc[mask, "grid_id"], errors="coerce").dropna().astype(int).unique().tolist()


def _prepare_local_action_context_current(
    *,
    sector_cells: pd.DataFrame,
    site_df: pd.DataFrame,
    context: dict[str, Any],
    config: CurrentModel3Config,
    logger,
    action_label: str,
    source_rows: pd.DataFrame,
) -> dict[str, Any] | None:
    """Model 3 current-recommendation variant of `future_rules._prepare_local_action_context`.

    The shared helper caps candidate interferers at the 25 nearest sites and grid
    coverage at a fixed radius, which makes the resimulated "after" PRB incomparable
    to the "before" PRB (computed from the full-network archive): the same physical
    cell gets a different SINR/capacity purely from interferers being dropped, and
    grids outside the radius silently disappear from the aggregation. This variant
    widens the interferer pool to match the original run's `max_interference_sites`
    cap and force-includes every grid the full-network surface actually assigned to
    the affected cells, so the local rerun stays apples-to-apples with "before".
    """
    baseline_all = context["baseline_pred_df"]
    corrected_all = context["corrected_pred_df"]
    detail_all = context["coverage_rows_df"]
    kpi_all = context["kpi_df"]
    geo_all = context["geo_df"]

    part3_baseline = baseline_all.loc[baseline_all["time_bucket"].astype(str) == "PART_3"].copy()
    part3_corrected = corrected_all.loc[corrected_all["time_bucket"].astype(str) == "PART_3"].copy()
    part3_detail = detail_all.loc[detail_all["time_bucket"].astype(str) == "PART_3"].copy()
    part3_kpi = kpi_all.loc[kpi_all["time_bucket"].astype(str) == "PART_3"].copy()
    part3_geo = geo_all.loc[geo_all["time_bucket"].astype(str) == "PART_3"].copy()

    if source_rows is None or source_rows.empty:
        source_rows = future_rules._extract_source_site_rows(sector_cells, site_df)
    if source_rows.empty:
        logger.info("%s_prepare_failed reason=no_source_rows", action_label)
        return None

    site_lat = pd.to_numeric(source_rows["lat"], errors="coerce").mean()
    site_lon = pd.to_numeric(source_rows["lon"], errors="coerce").mean()
    site_work = site_df.copy()
    site_work["_site_distance_m"] = future_rules._haversine_distance_m_series(site_work["lat"], site_work["lon"], site_lat, site_lon)
    site_distance_df = (
        site_work.groupby("Site ID", as_index=False)
        .agg(site_distance_m=("_site_distance_m", "min"))
        .sort_values("site_distance_m", ascending=True)
    )
    max_interference_sites = int(context["summary"].get("max_interference_sites", 50))
    nearest_site_count = max(60, max_interference_sites + 15)
    nearest_site_ids = site_distance_df["Site ID"].head(nearest_site_count).astype(str).tolist()
    local_site_df = site_work.loc[site_work["Site ID"].astype(str).isin(nearest_site_ids)].copy()
    if local_site_df.empty:
        local_site_df = site_work.copy()

    local_point_map = future_rules._make_local_point_map(part3_baseline)
    if local_point_map.empty:
        logger.info("%s_prepare_failed reason=no_point_map", action_label)
        return None
    local_point_map["_site_distance_m"] = future_rules._haversine_distance_m_series(local_point_map["lat"], local_point_map["lon"], site_lat, site_lon)

    lineage_grid_ids = _affected_lineage_grid_ids(sector_cells, context.get("current_df", pd.DataFrame()))
    radius_mask = local_point_map["_site_distance_m"] <= float(config.sector_split_local_radius_m)
    lineage_mask = pd.to_numeric(local_point_map["grid_id"], errors="coerce").astype("Int64").isin(lineage_grid_ids)
    local_point_map = local_point_map.loc[radius_mask | lineage_mask].copy()
    if local_point_map.empty:
        local_point_map = future_rules._make_local_point_map(
            part3_baseline.loc[part3_baseline["Node_Cell_ID"].astype(str).isin(sector_cells["Node_Cell_ID"].astype(str))]
        )
    if local_point_map.empty:
        logger.info("%s_prepare_failed reason=no_local_points", action_label)
        return None

    affected_grid_ids = pd.to_numeric(local_point_map["grid_id"], errors="coerce").dropna().astype("Int64").unique().tolist()
    local_detail = pd.DataFrame()
    if affected_grid_ids and "grid_id" in part3_detail.columns:
        detail_grid_ids = pd.to_numeric(part3_detail["grid_id"], errors="coerce").astype("Int64")
        local_detail = part3_detail.loc[detail_grid_ids.isin(affected_grid_ids)].copy()
    if local_detail.empty:
        local_detail = part3_detail.merge(local_point_map[["lat", "lon"]].drop_duplicates(), on=["lat", "lon"], how="inner")
    if local_detail.empty:
        local_detail = local_point_map[["lat", "lon", "grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon"]].drop_duplicates().copy()
        local_detail["time_bucket"] = "PART_3"

    local_kpi = part3_kpi.loc[pd.to_numeric(part3_kpi["grid_id"], errors="coerce").astype("Int64").isin(affected_grid_ids)].copy()
    local_geo = part3_geo.loc[pd.to_numeric(part3_geo["grid_id"], errors="coerce").astype("Int64").isin(affected_grid_ids)].copy()
    logger.info(
        "%s_prepare local_points=%d local_detail_rows=%d local_sites=%d lineage_grid_ids=%d",
        action_label,
        len(local_point_map),
        len(local_detail),
        len(local_site_df),
        len(lineage_grid_ids),
    )
    return {
        "part3_baseline": part3_baseline,
        "part3_corrected": part3_corrected,
        "local_site_df": local_site_df,
        "local_point_map": local_point_map,
        "local_detail": local_detail,
        "local_kpi": local_kpi,
        "local_geo": local_geo,
    }


def _evaluate_action_inventory_current(
    *,
    sector_cells: pd.DataFrame,
    after_inventory: pd.DataFrame,
    candidate_node_cell_ids: set[str],
    demand_serving_cell_ids: set[str],
    config: CurrentModel3Config,
    logger,
    action_label: str,
) -> dict[str, Any]:
    """Model 3 current-recommendation variant of `future_rules._evaluate_action_inventory`.

    The shared function's same-site fallback widens the after-scope to ANY cell at the
    site with nonzero load whenever the matched scope's PRB reads <= 0. That conflates
    two very different situations: (a) lineage matching genuinely failed to find the
    acted-on cell(s) in the after-surface, vs (b) the cell(s) were found and correctly
    show ~0 load because the action successfully evacuated them. Case (b) is exactly what
    a working carrier addition looks like, but the shared logic then substitutes an
    unrelated sibling cell's independent PRB as if it were this action's outcome -
    producing "before=96%, after=115%" results that describe two different physical
    cells, not a before/after of the same one.

    A cell that wins zero grids never gets a row in the collapsed `after_inventory` at
    all (`_build_current_cell_inventory` only emits rows for cells present in at least
    one grid's winning assignment) - so "absent from after_inventory" alone can't tell
    apart "genuinely evacuated to 0%" from "never a valid candidate in this rerun".
    `candidate_node_cell_ids` (the RF-predicted candidate surface, before winner
    selection collapses it) resolves that: if the acted-on cell was a real candidate,
    trust the 0% reading; only fall back to the site-wide search if it wasn't even a
    candidate here (e.g. removed by sector split, or dropped by the local topology).
    """
    before_cells = sector_cells["Node_Cell_ID"].astype(str).tolist()
    source_sector_id = str(future_rules._first_non_empty(sector_cells["sector_id"])).strip()
    source_site_id = future_rules._normalize_identity_text(future_rules._first_non_empty(sector_cells["site_id"]))
    source_frontend_sector_keys = {source_sector_id} if source_sector_id else set()
    source_topology_sector_keys: set[str] = set()
    for col in [
        "topology_frontend_site_sector_key",
        "topology_node_cell_sector_key",
        "topology_sector_identity_key",
        "topology_canonical_sector_id",
        "topology_site_sector_band_key",
        "sector_id",
    ]:
        if col in sector_cells.columns:
            source_topology_sector_keys.update(
                str(value).strip()
                for value in sector_cells[col].dropna().astype(str).tolist()
                if str(value).strip()
            )
    source_frontend_sector_keys.update(source_topology_sector_keys)
    original_lineage_ids = set(value for value in before_cells if str(value).strip())
    for col in ["topology_original_cell_id", "topology_original_node_cell_id"]:
        if col in sector_cells.columns:
            original_lineage_ids.update(
                str(value).strip()
                for value in sector_cells[col].dropna().astype(str).tolist()
                if str(value).strip()
            )

    def _scope_mask(df: pd.DataFrame, *, sector_keys: set[str], lineage_ids: set[str], site_id: str) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        if source_sector_id and "sector_id" in df.columns:
            mask = mask | df["sector_id"].astype(str).eq(source_sector_id)
        for col in [
            "sector_id",
            "topology_frontend_site_sector_key",
            "topology_node_cell_sector_key",
            "topology_sector_identity_key",
            "topology_canonical_sector_id",
            "topology_site_sector_band_key",
        ]:
            if col in df.columns and sector_keys:
                mask = mask | df[col].astype(str).isin(sector_keys)
        for col in ["Node_Cell_ID", "topology_original_cell_id", "topology_original_node_cell_id"]:
            if col in df.columns and lineage_ids:
                mask = mask | df[col].astype(str).isin(lineage_ids)
        if site_id and "site_id" in df.columns:
            mask = mask | df["site_id"].map(future_rules._normalize_identity_text).eq(site_id)
        return mask

    primary_mask = _scope_mask(after_inventory, sector_keys=source_frontend_sector_keys, lineage_ids=original_lineage_ids, site_id="")
    primary_matches = after_inventory.loc[primary_mask].copy()

    affected_sector_keys = set(source_frontend_sector_keys)
    if not primary_matches.empty:
        for col in [
            "sector_id",
            "topology_frontend_site_sector_key",
            "topology_node_cell_sector_key",
            "topology_sector_identity_key",
            "topology_canonical_sector_id",
            "topology_site_sector_band_key",
        ]:
            if col in primary_matches.columns:
                affected_sector_keys.update(
                    str(value).strip()
                    for value in primary_matches[col].dropna().astype(str).tolist()
                    if str(value).strip()
                )
        for col in ["Node_Cell_ID", "topology_original_cell_id", "topology_original_node_cell_id"]:
            if col in primary_matches.columns:
                original_lineage_ids.update(
                    str(value).strip()
                    for value in primary_matches[col].dropna().astype(str).tolist()
                    if str(value).strip()
                )

    final_mask = _scope_mask(after_inventory, sector_keys=affected_sector_keys, lineage_ids=original_lineage_ids, site_id="")
    # Follow the demand: cells that now serve the congested lineage grids decide the
    # outcome, whether they are the original cell, the synthetic candidate, or a sibling.
    if demand_serving_cell_ids and "Node_Cell_ID" in after_inventory.columns:
        final_mask = final_mask | after_inventory["Node_Cell_ID"].astype(str).str.strip().isin(demand_serving_cell_ids)
    after_candidates = after_inventory.loc[final_mask].copy()

    after_lineage_ids: set[str] = set()
    for col in ["Node_Cell_ID", "topology_original_cell_id", "topology_original_node_cell_id"]:
        if col in after_inventory.columns:
            after_lineage_ids.update(after_inventory[col].dropna().astype(str))
    before_cells_found_in_after = bool(original_lineage_ids & after_lineage_ids)
    before_cells_were_candidates = bool(original_lineage_ids & candidate_node_cell_ids)

    scope_mode = "matched_scope"
    evacuated_to_zero = False

    if after_candidates.empty and not before_cells_found_in_after and before_cells_were_candidates:
        # The acted-on cell(s) were legitimate RF candidates in this rerun (they had
        # predicted RSRP for at least one grid) but won zero grids after reselection -
        # `_build_current_cell_inventory` never emits a row for a cell with 0 assigned
        # grids, so this is a real "evacuated to 0%" outcome, not a matching gap. Trust
        # it directly instead of substituting an unrelated cell's independent load.
        evacuated_to_zero = True
        scope_mode = "evacuated_zero"
    elif after_candidates.empty and source_site_id:
        after_site_ids = after_inventory["site_id"].map(future_rules._normalize_identity_text) if "site_id" in after_inventory.columns else pd.Series("", index=after_inventory.index)
        after_candidates = after_inventory.loc[after_site_ids == source_site_id].copy()

    def _summarize_scope(df: pd.DataFrame) -> tuple[float, float, float]:
        prb = future_rules._to_num(df["prb_before_pct"])
        rrc = future_rules._to_num(df["rrc_before_pct"])
        users = future_rules._to_num(df["rrc_users_before"]) if "rrc_users_before" in df.columns else pd.Series(dtype=float)
        return (
            float(prb.max()) if prb.notna().any() else np.nan,
            float(rrc.max()) if rrc.notna().any() else np.nan,
            float(users.max()) if users.notna().any() else np.nan,
        )

    if evacuated_to_zero:
        projected_prb, projected_rrc, projected_users = 0.0, 0.0, 0.0
    else:
        projected_prb, projected_rrc, projected_users = _summarize_scope(after_candidates)
    before_prb = float(future_rules._to_num(sector_cells["prb_before_pct"]).max()) if future_rules._to_num(sector_cells["prb_before_pct"]).notna().any() else np.nan
    before_rrc = float(future_rules._to_num(sector_cells["rrc_before_pct"]).max()) if future_rules._to_num(sector_cells["rrc_before_pct"]).notna().any() else np.nan
    before_pressure = max(before_prb if pd.notna(before_prb) else 0.0, before_rrc if pd.notna(before_rrc) else 0.0)

    # Only widen scope when the acted-on cell(s) are genuinely absent from the after
    # surface AND were never even a valid candidate here (a real lineage/topology
    # failure) - not when they were a legitimate candidate that simply lost every grid.
    if (
        not evacuated_to_zero
        and source_site_id
        and not before_cells_found_in_after
        and pd.notna(before_prb)
        and before_prb > 0
        and (not pd.notna(projected_prb) or projected_prb <= 0.0)
        and "site_id" in after_inventory.columns
    ):
        site_scope = after_inventory.loc[after_inventory["site_id"].map(future_rules._normalize_identity_text).eq(source_site_id)].copy()
        site_scope = site_scope.loc[
            (future_rules._to_num(site_scope.get("prb_before_pct", pd.Series(dtype=float))) > 0.0)
            | (future_rules._to_num(site_scope.get("rrc_before_pct", pd.Series(dtype=float))) > 0.0)
            | (future_rules._to_num(site_scope.get("rrc_users_before", pd.Series(dtype=float))) > 0.0)
        ].copy()
        if not site_scope.empty:
            after_candidates = site_scope
            projected_prb, projected_rrc, projected_users = _summarize_scope(after_candidates)
            scope_mode = "same_site_nonzero_fallback"

    after_pressure = max(projected_prb if pd.notna(projected_prb) else 0.0, projected_rrc if pd.notna(projected_rrc) else 0.0)
    resolved = pd.notna(projected_prb) and pd.notna(projected_rrc) and projected_prb <= config.congestion_threshold and projected_rrc <= config.congestion_threshold
    improved = after_pressure < before_pressure - 0.5
    worsened = after_pressure > before_pressure + 0.5
    if resolved:
        status = "Resolved"
    elif worsened:
        status = "Rejected"
    elif improved:
        status = "Partially Resolved"
    else:
        status = "No Material Change"
    after_cells = after_candidates["Node_Cell_ID"].astype(str).dropna().tolist() if "Node_Cell_ID" in after_candidates.columns else []
    logger.info(
        "%s_done sector=%s before_cells=%s after_cells=%s demand_serving_cells=%s local_after_rows=%d scope_mode=%s before_cells_found_in_after=%s before_prb=%.3f before_rrc=%.3f actual_prb=%.3f actual_rrc=%.3f status=%s",
        action_label,
        source_sector_id,
        before_cells,
        after_cells,
        sorted(demand_serving_cell_ids),
        len(after_candidates),
        scope_mode,
        before_cells_found_in_after,
        before_prb if pd.notna(before_prb) else -1.0,
        before_rrc if pd.notna(before_rrc) else -1.0,
        projected_prb if pd.notna(projected_prb) else -1.0,
        projected_rrc if pd.notna(projected_rrc) else -1.0,
        status,
    )
    return {
        "status": status,
        "projected_prb_after_pct": round(projected_prb, 3) if pd.notna(projected_prb) else np.nan,
        "projected_rrc_after_pct": round(projected_rrc, 3) if pd.notna(projected_rrc) else np.nan,
        "projected_rrc_users_after": round(projected_users, 3) if pd.notna(projected_users) else np.nan,
        "next_step": "" if resolved else "New Site",
        "after_cells": after_cells,
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
    local_ctx = _prepare_local_action_context_current(
        sector_cells=sector_cells,
        site_df=site_df,
        context=context,
        config=config,
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
    point_map = local_ctx["local_point_map"].drop(columns=["_site_distance_m"], errors="ignore")
    snap_tolerance_m = float(context["summary"].get("grid_size_m", 50.0)) * 0.75
    baseline_local["time_bucket"] = "PART_3"
    baseline_local = _force_master_grid_identity(baseline_local, point_map, snap_tolerance_m)
    if corrected_local.empty:
        corrected_local = baseline_local.copy()
    else:
        corrected_local["time_bucket"] = "PART_3"
        corrected_local = _force_master_grid_identity(corrected_local, point_map, snap_tolerance_m)
    enriched_local, local_inventory = _build_current_inventory_from_surface(
        baseline_local=baseline_local,
        corrected_local=corrected_local,
        local_site_df=local_ctx["local_site_df"],
        local_kpi=local_ctx["local_kpi"],
        local_geo=local_ctx["local_geo"],
        context=context,
        config=config,
    )
    candidate_node_cell_ids: set[str] = set()
    for col in ["Node_Cell_ID", "original_node_cell_id", "original_cell_id"]:
        if col in corrected_local.columns:
            candidate_node_cell_ids.update(corrected_local[col].dropna().astype(str))

    # The congested demand lives on specific grids. After the action, whichever cells
    # now serve those grids carry that demand - THEY decide whether congestion is
    # actually resolved, regardless of how any cell is named across tooling layers.
    lineage_grid_ids = set(_affected_lineage_grid_ids(sector_cells, context.get("current_df", pd.DataFrame())))
    en_gid = pd.to_numeric(enriched_local.get("grid_id"), errors="coerce")
    demand_serving_cell_ids = set(
        enriched_local.loc[en_gid.isin(lineage_grid_ids), "Node_Cell_ID"].dropna().astype(str).str.strip()
    )
    demand_serving_cell_ids.discard("")

    outcome = _evaluate_action_inventory_current(
        sector_cells=sector_cells,
        after_inventory=local_inventory,
        candidate_node_cell_ids=candidate_node_cell_ids,
        demand_serving_cell_ids=demand_serving_cell_ids,
        config=config,
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
