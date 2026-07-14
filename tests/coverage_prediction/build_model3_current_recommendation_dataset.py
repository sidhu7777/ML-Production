from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tests.coverage_prediction import build_model3_hybrid_load_balancing_dataset as future_builder


OUTPUT_ROOT = ML_ROOT / "models" / "model3_current_recommendation_experiment"
CURRENT_MODEL3_DATASET_CSV = OUTPUT_ROOT / "model3_current_dataset.csv"
CURRENT_MODEL3_SUMMARY_JSON = OUTPUT_ROOT / "model3_current_summary.json"


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


def _apply_current_model3_load_profile(df: pd.DataFrame, rrc_sector_capacity: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()

    score_components: list[pd.Series] = []
    score_weights: list[float] = []
    for col, weight in [
        ("demand_index", 0.35),
        ("traffic_demand_est", 0.30),
        ("active_users_est", 0.15),
        ("prb_pressure_est", 0.10),
        ("capacity_gap_score", 0.05),
        ("growth_zone_score", 0.05),
    ]:
        if col in out.columns:
            score_components.append(future_builder._minmax_norm(out[col]))
            score_weights.append(weight)

    if score_components:
        total_weight = float(sum(score_weights)) or 1.0
        hotspot_score = sum(component * weight for component, weight in zip(score_components, score_weights)) / total_weight
    else:
        hotspot_score = pd.Series(0.0, index=out.index, dtype="float64")

    if "high_band_ratio" in out.columns:
        hotspot_score = (0.85 * hotspot_score) + (0.15 * future_builder._minmax_norm(out["high_band_ratio"]))
    if "carrier_count" in out.columns:
        hotspot_score = (0.90 * hotspot_score) + (0.10 * future_builder._minmax_norm(out["carrier_count"]))

    hotspot_score = hotspot_score.clip(0.0, 1.0).round(6)
    hotspot_rank = future_builder._rank_percentile(hotspot_score)
    prb_target_pct = future_builder._tiered_target_from_rank(hotspot_rank, future_builder.MODEL3_PRB_TARGET_BANDS)

    capacity_mbps = pd.to_numeric(out.get("estimated_dl_capacity_mbps"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if capacity_mbps.notna().sum() == 0:
        capacity_mbps = pd.Series(1.0, index=out.index, dtype="float64")
    capacity_mbps = capacity_mbps.fillna(capacity_mbps.median() if capacity_mbps.notna().any() else 1.0).clip(lower=0.1)

    out["model3_hotspot_score"] = hotspot_score
    out["model3_hotspot_rank"] = hotspot_rank.round(6)
    out["model3_prb_target_pct"] = prb_target_pct.round(3)
    out["estimated_offered_traffic_mbps"] = (capacity_mbps * (prb_target_pct / 100.0)).clip(lower=0.0).round(3)
    out["estimated_prb_utilization_pct"] = prb_target_pct.round(3)

    rrc_group_cols = _pick_rrc_group_cols(out)
    if rrc_group_cols:
        group_profile = (
            out.groupby(rrc_group_cols, dropna=False, as_index=False)
            .agg(
                group_hotspot_score=("model3_hotspot_score", "mean"),
                group_grid_count=("grid_id", "nunique"),
                group_user_weight=("model3_hotspot_score", "sum"),
            )
        )
        group_profile["group_rank"] = future_builder._rank_percentile(group_profile["group_hotspot_score"])
        group_profile["group_rrc_target_pct"] = future_builder._tiered_target_from_rank(
            group_profile["group_rank"],
            future_builder.MODEL3_RRC_TARGET_BANDS,
        )
        group_profile["group_total_users_target"] = (
            pd.to_numeric(group_profile["group_rrc_target_pct"], errors="coerce").fillna(0.0) / 100.0
        ) * float(rrc_sector_capacity)
        group_profile["group_user_weight"] = pd.to_numeric(group_profile["group_user_weight"], errors="coerce").fillna(0.0)
        group_profile["group_user_weight"] = group_profile["group_user_weight"].where(group_profile["group_user_weight"] > 0, 1.0)

        out = out.merge(group_profile, on=rrc_group_cols, how="left", validate="many_to_one")
        row_weight = (
            0.65 * future_builder._minmax_norm(out["model3_hotspot_score"])
            + 0.35 * future_builder._minmax_norm(out["traffic_demand_est"] if "traffic_demand_est" in out.columns else out["model3_hotspot_score"])
        )
        row_weight = row_weight.fillna(0.0) + 0.10
        group_index = out.groupby(rrc_group_cols, sort=False).ngroup()
        weight_sum = row_weight.groupby(group_index, sort=False).transform("sum").replace(0, np.nan).fillna(1.0)

        out["estimated_rrc_connected_users"] = (out["group_total_users_target"] * (row_weight / weight_sum)).clip(lower=0.0).round(3)
        out["estimated_cell_grid_count"] = pd.to_numeric(out["group_grid_count"], errors="coerce").fillna(1.0).round(3)
        out["estimated_cell_rrc_connected_users"] = pd.to_numeric(out["group_total_users_target"], errors="coerce").fillna(0.0).round(3)
        out["estimated_cell_rrc_utilization_pct"] = pd.to_numeric(out["group_rrc_target_pct"], errors="coerce").fillna(0.0).round(3)

        profile_summary = {
            "rrc_grouping_columns": rrc_group_cols,
            "rrc_group_count": int(len(group_profile)),
            "rrc_target_threshold_counts": future_builder._threshold_counts(group_profile["group_rrc_target_pct"]),
        }
    else:
        out["estimated_rrc_connected_users"] = pd.to_numeric(out.get("active_users_est"), errors="coerce").fillna(0.0).round(3)
        out["estimated_cell_grid_count"] = np.nan
        out["estimated_cell_rrc_connected_users"] = out["estimated_rrc_connected_users"].round(3)
        out["estimated_cell_rrc_utilization_pct"] = (
            (out["estimated_rrc_connected_users"] / float(rrc_sector_capacity)) * 100.0
        ).round(3)
        profile_summary = {
            "rrc_grouping_columns": None,
            "rrc_group_count": 0,
            "rrc_target_threshold_counts": {},
        }

    out["model3_profile_source"] = "ranked_hotspot_targeting_on_current_capacity_features"
    return out, profile_summary


def build_model3_current_dataset(
    rrc_sector_capacity: float = future_builder.DEFAULT_RRC_SECTOR_CAPACITY,
    mimo_layers: float = future_builder.DEFAULT_MIMO_LAYERS,
    control_overhead: float = future_builder.DEFAULT_CONTROL_OVERHEAD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hybrid_model2 = future_builder.build_hybrid_model2_dataset()
    out = hybrid_model2[hybrid_model2["time_bucket"].astype(str) == "PART_3"].copy()
    out, capability_summary = future_builder._add_sector_carrier_capability_fields(out)

    bandwidth_mhz = pd.to_numeric(out.get("bandwidth_mhz_est"), errors="coerce").replace(0, np.nan).fillna(10.0)
    usable_resource_fraction = max(0.10, min(1.0, 1.0 - float(control_overhead)))
    layer_count = max(1.0, float(mimo_layers))
    spectral_efficiency = future_builder._estimate_spectral_efficiency_bpshz(
        out,
        bandwidth_mhz=bandwidth_mhz,
        mimo_layers=layer_count,
        control_overhead=control_overhead,
    )
    out["estimated_spectral_efficiency_bpshz"] = spectral_efficiency.round(6)
    out["estimated_dl_capacity_mbps"] = (
        bandwidth_mhz * spectral_efficiency * layer_count * usable_resource_fraction
    ).clip(lower=0.1).round(3)

    traffic = pd.to_numeric(out.get("traffic_demand_est"), errors="coerce").fillna(0.0)
    users = pd.to_numeric(out.get("active_users_est"), errors="coerce").fillna(0.0)
    demand = pd.to_numeric(out.get("demand_index"), errors="coerce").fillna(0.0)
    estimated_prb_mean = pd.to_numeric(out.get("estimated_prb_mean"), errors="coerce").fillna(0.0)

    out["raw_estimated_offered_traffic_mbps"] = traffic.clip(lower=0.0).round(3)
    out["raw_estimated_rrc_connected_users"] = users.clip(lower=0.0).round(3)
    out["raw_estimated_prb_utilization_pct"] = (
        (out["raw_estimated_offered_traffic_mbps"] / out["estimated_dl_capacity_mbps"].replace(0.0, np.nan)) * 100.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).round(3)
    out["prb_pressure_est"] = (
        0.50 * future_builder._minmax_norm(out["raw_estimated_prb_utilization_pct"])
        + 0.30 * future_builder._minmax_norm(demand)
        + 0.20 * future_builder._minmax_norm(estimated_prb_mean)
    ).round(6)
    out["estimated_offered_traffic_mbps"] = out["raw_estimated_offered_traffic_mbps"]
    out["estimated_rrc_connected_users"] = out["raw_estimated_rrc_connected_users"]
    out, profile_summary = _apply_current_model3_load_profile(out, rrc_sector_capacity)

    out["model3_proxy_source"] = "current_part3_hybrid_surface_plus_engineered_capacity_features"
    out["model3_mode"] = "current"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.to_csv(CURRENT_MODEL3_DATASET_CSV, index=False)

    metrics = {}
    for col in [
        "estimated_prb_utilization_pct",
        "estimated_cell_rrc_utilization_pct",
        "estimated_offered_traffic_mbps",
        "estimated_rrc_connected_users",
        "estimated_dl_capacity_mbps",
    ]:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        metrics[col] = {
            "non_null": int(values.notna().sum()),
            "min": float(values.min()) if values.notna().any() else None,
            "p50": float(values.quantile(0.50)) if values.notna().any() else None,
            "p80": float(values.quantile(0.80)) if values.notna().any() else None,
            "p90": float(values.quantile(0.90)) if values.notna().any() else None,
            "max": float(values.max()) if values.notna().any() else None,
            "threshold_counts": future_builder._threshold_counts(values),
        }

    summary = {
        "mode": "current_model3_dataset",
        "rows": int(len(out)),
        "unique_grids": int(out["grid_id"].nunique(dropna=True)),
        "bucket_counts": {str(k): int(v) for k, v in out["time_bucket"].value_counts().sort_index().items()},
        "rrc_sector_capacity_assumption": float(rrc_sector_capacity),
        "mimo_layers_assumption": float(mimo_layers),
        "control_overhead_assumption": float(control_overhead),
        "dataset_csv": str(CURRENT_MODEL3_DATASET_CSV),
        "summary_json": str(CURRENT_MODEL3_SUMMARY_JSON),
        "proxy_formula_notes": [
            "This dataset is for current-state recommendation only.",
            "It does not use Model 1 future RF predictions.",
            "It does not use Model 2 future target predictions.",
            "Raw current traffic/users are preserved in raw_estimated_offered_traffic_mbps and raw_estimated_rrc_connected_users.",
            "Planning PRB is reshaped from a hotspot-ranked target profile rather than using raw traffic_demand_est directly.",
            "Planning RRC is shaped at the true cell/carrier grouping using sector-level target bands.",
            "estimated_prb_utilization_pct is the planning target utilization for current-state recommendation testing.",
            "estimated_cell_rrc_utilization_pct is the shaped planning RRC utilization for current-state recommendation testing.",
        ],
        "sector_carrier_capability": capability_summary,
        "load_profile_summary": profile_summary,
        "metrics": metrics,
    }
    _save_json(CURRENT_MODEL3_SUMMARY_JSON, summary)
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current-state Model 3 dataset without Model 1/2 predictions.")
    parser.add_argument("--rrc-sector-capacity", type=float, default=future_builder.DEFAULT_RRC_SECTOR_CAPACITY)
    parser.add_argument("--mimo-layers", type=float, default=future_builder.DEFAULT_MIMO_LAYERS)
    parser.add_argument("--control-overhead", type=float, default=future_builder.DEFAULT_CONTROL_OVERHEAD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = build_model3_current_dataset(args.rrc_sector_capacity, args.mimo_layers, args.control_overhead)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
