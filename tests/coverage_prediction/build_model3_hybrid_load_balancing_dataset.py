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

from tests.coverage_prediction import build_model2_capacity_training_dataset as model2_builder


HYBRID_MODEL1_CSV = ML_ROOT / "models" / "model1_hybrid_target_experiment" / "hybrid_target_training.csv"
MODEL2_BASE_CSV = ML_ROOT / "data" / "model2_capacity_training.csv"
SOURCE_COVERAGE_ARCHIVE = ML_ROOT / "data" / "coverage_20260521_104406.7z"
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
THRESHOLDS = [50.0, 60.0, 70.0, 80.0, 90.0]


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

    site_df = _read_csv_from_archive(SOURCE_COVERAGE_ARCHIVE, "project_sites.csv")
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

    if not site_df.empty and "Node_Cell_ID" in site_df.columns:
        site_work = site_df.copy()
        site_work["Node_Cell_ID"] = site_work["Node_Cell_ID"].astype(str)
        site_keep = [
            col
            for col in [
                "Node_Cell_ID",
                "Site ID",
                "nodeb_id",
                "cell_id",
                "PCI",
                "earfcn",
                "band",
                "azimuth",
                "lat",
                "lon",
            ]
            if col in site_work.columns
        ]
        site_work = site_work[site_keep].drop_duplicates(subset=["Node_Cell_ID"], keep="first")
        best = best.merge(site_work, on="Node_Cell_ID", how="left", validate="many_to_one")
        rename_map = {
            "Site ID": "topology_site_id",
            "nodeb_id": "topology_nodeb_id",
            "cell_id": "topology_cell_id",
            "PCI": "topology_pci",
            "earfcn": "topology_earfcn",
            "band": "topology_band",
            "azimuth": "topology_azimuth",
            "lat": "topology_site_lat",
            "lon": "topology_site_lon",
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
    out = hybrid_model2.copy()

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
    offered_traffic_mbps = pd.to_numeric(out[traffic_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["estimated_offered_traffic_mbps"] = offered_traffic_mbps.round(3)
    out["estimated_prb_utilization_pct"] = (
        (offered_traffic_mbps / out["estimated_dl_capacity_mbps"]) * 100.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 100.0).round(3)

    out["estimated_rrc_connected_users"] = pd.to_numeric(out[users_col], errors="coerce").fillna(0.0).round(3)

    true_cell_key_candidates = [
        ["time_bucket", "site_id", "sector_id", "cell_id"],
        ["time_bucket", "Node_Cell_ID"],
        ["time_bucket", "cell_id"],
    ]
    rrc_group_cols = next(
        (
            cols
            for cols in true_cell_key_candidates
            if all(col in out.columns for col in cols)
        ),
        None,
    )
    if rrc_group_cols:
        rrc_group = (
            out.groupby(rrc_group_cols, dropna=False, as_index=False)
            .agg(
                estimated_cell_rrc_connected_users=("estimated_rrc_connected_users", "sum"),
                estimated_cell_grid_count=("grid_id", "nunique"),
            )
        )
        out = out.merge(rrc_group, on=rrc_group_cols, how="left", validate="many_to_one")
        out["estimated_cell_rrc_utilization_pct"] = (
            (out["estimated_cell_rrc_connected_users"] / float(rrc_sector_capacity)) * 100.0
        ).clip(lower=0.0).round(3)
        rrc_aggregation_note = f"RRC utilization is aggregated by true cell key columns: {rrc_group_cols}."
    else:
        out["estimated_cell_rrc_connected_users"] = np.nan
        out["estimated_cell_grid_count"] = np.nan
        out["estimated_cell_rrc_utilization_pct"] = np.nan
        rrc_aggregation_note = (
            "RRC utilization is not calculated because the current Model 3 dataset has no true "
            "site_id/sector_id/cell_id/Node_Cell_ID key. dominant_pci is intentionally not used "
            "because PCI is not globally unique and creates invalid sector-level aggregation."
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
                "and one EARFCN. However, this artifact's project_sites topology only contains LTE 1800/EARFCN 1750, "
                "so real inter-band load-shift simulation is not represented until multi-band topology exists."
            ),
        }

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
            "estimated_offered_traffic_mbps",
            "estimated_spectral_efficiency_bpshz",
            "estimated_dl_capacity_mbps",
            "estimated_prb_utilization_pct",
            "estimated_rrc_connected_users",
        "estimated_cell_rrc_connected_users",
        "estimated_cell_rrc_utilization_pct",
        ],
        "proxy_formula_notes": [
            "estimated_spectral_efficiency_bpshz uses a conservative Shannon-style SINR estimate with CQI fallback.",
            "estimated_dl_capacity_mbps = bandwidth_mhz_est * estimated_spectral_efficiency_bpshz * mimo_layers * (1 - control_overhead).",
            "estimated_prb_utilization_pct = estimated_offered_traffic_mbps / estimated_dl_capacity_mbps * 100.",
            "estimated_offered_traffic_mbps comes from the trained hybrid Model 2 traffic prediction when available.",
            "estimated_rrc_connected_users comes from the trained hybrid Model 2 active-users prediction when available.",
        "estimated_rrc_connected_users remains a grid-level user estimate.",
        "estimated_cell_rrc_connected_users = sum(estimated_rrc_connected_users) across the available cell-like group.",
        "estimated_cell_rrc_utilization_pct = estimated_cell_rrc_connected_users / sector_rrc_capacity * 100.",
        rrc_aggregation_note,
        "These are prototype engineering proxies, not industry-standard OSS counters.",
    ],
        "dominant_band_class_counts": band_summary,
        "topology_verification": topology_verification,
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
