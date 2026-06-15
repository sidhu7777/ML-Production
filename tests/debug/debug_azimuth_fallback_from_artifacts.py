from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lte_tilt_recommandation.candidate_validation import (  # noqa: E402
    CandidateValidationConfig,
    MIN_BEARING_SAMPLE_COUNT,
    _apply_updates_to_site_df,
    _bearing_context_map,
    _evaluate_update_set,
    _make_azimuth_update,
)
from tools.lte_tilt_recommandation.cell_identity import canonical_cell_id  # noqa: E402
from tools.lte_tilt_recommandation.geo_logic import _signed_azimuth_delta  # noqa: E402


DEFAULT_ARTIFACT_DIR = ROOT / "outputs" / "temp_e637a811-abdc-4e52-b727-52f96d1d7845"


def _parse_updates(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    parsed = ast.literal_eval(text)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _extract_target_cells(candidate_df: pd.DataFrame) -> list[str]:
    cells: set[str] = set()
    pattern = re.compile(r"^coord_pass_\d+_cell_(.+?)_delta_")
    for value in candidate_df.get("candidate_name", pd.Series(dtype=str)).dropna().astype(str):
        match = pattern.search(value)
        if match:
            cell = canonical_cell_id(match.group(1))
            if cell:
                cells.add(cell)
    return sorted(cells)


def _build_antenna_df(scope_df: pd.DataFrame) -> pd.DataFrame:
    site_cols = [
        "Node_Cell_ID",
        "nodeb_id",
        "local_cell_id",
        "cell_id",
        "Technology_site",
        "lat_site",
        "lon_site",
        "azimuth",
        "electrical_tilt",
        "mechanical_tilt",
        "tx_power",
        "antenna_height",
        "dashboard_site_id",
    ]
    keep = [col for col in site_cols if col in scope_df.columns]
    antenna_df = scope_df[keep].copy()
    antenna_df["Node_Cell_ID"] = antenna_df["Node_Cell_ID"].map(canonical_cell_id)
    antenna_df = antenna_df.loc[antenna_df["Node_Cell_ID"].astype(str).str.strip().ne("")]
    if "local_cell_id" not in antenna_df.columns and "cell_id" in antenna_df.columns:
        antenna_df["local_cell_id"] = antenna_df["cell_id"].map(canonical_cell_id)
    antenna_df["_usable_site_score"] = 0
    for col in ["lat_site", "lon_site", "azimuth", "electrical_tilt"]:
        if col in antenna_df.columns:
            antenna_df["_usable_site_score"] += pd.to_numeric(antenna_df[col], errors="coerce").notna().astype(int)
    parts = antenna_df["Node_Cell_ID"].astype(str).str.split("_")
    antenna_df["_full_key_score"] = parts.map(lambda item: int(len(item) >= 3 and item[0] == item[1]))
    dedupe_key = "local_cell_id" if "local_cell_id" in antenna_df.columns else "Node_Cell_ID"
    antenna_df = (
        antenna_df.sort_values([dedupe_key, "_usable_site_score", "_full_key_score"], ascending=[True, False, False])
        .drop_duplicates(subset=[dedupe_key], keep="first")
        .drop(columns=["_usable_site_score", "_full_key_score"], errors="ignore")
    )
    rename_map = {}
    if "lat_site" in antenna_df.columns:
        rename_map["lat_site"] = "lat"
    if "lon_site" in antenna_df.columns:
        rename_map["lon_site"] = "lon"
    if "Technology_site" in antenna_df.columns:
        rename_map["Technology_site"] = "Technology"
    return antenna_df.rename(columns=rename_map)


def _artifact_cell_alias(cell_id: Any) -> str:
    cell = canonical_cell_id(cell_id)
    parts = cell.split("_")
    if len(parts) >= 3 and parts[0] == parts[1]:
        return "_".join([parts[0], *parts[2:]])
    return cell


def _artifact_full_alias(cell_id: Any) -> str:
    cell = canonical_cell_id(cell_id)
    parts = cell.split("_")
    if len(parts) == 2:
        return f"{parts[0]}_{parts[0]}_{parts[1]}"
    return cell


def _resolve_cell_for_artifact(cell_id: Any, available_cells: set[str]) -> str:
    cell = canonical_cell_id(cell_id)
    candidates = [cell, _artifact_full_alias(cell), _artifact_cell_alias(cell)]
    for candidate in candidates:
        if candidate in available_cells:
            return candidate
    return cell


def _resolve_update_ids_for_artifact(
    updates: list[dict[str, Any]],
    available_cells: set[str],
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for update in updates:
        out = dict(update)
        out["cell_id"] = _resolve_cell_for_artifact(out.get("cell_id"), available_cells)
        resolved.append(out)
    return resolved


def _safe_float(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else np.nan


def _rank(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(float(metrics.get("constraints_passed", 0.0)) >= 1.0),
        float(metrics.get("score", -99999.0)),
        float(metrics.get("net_bad_reduction", -99999.0)),
        float(metrics.get("combined_weighted_tie_break", -99999.0)),
        -float(metrics.get("new_bad_samples", metrics.get("good_to_bad_grid_count", 99999.0))),
    )


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "score",
        "constraints_passed",
        "net_bad_reduction",
        "candidate_bad_count",
        "new_bad_samples",
        "good_area_loss_pct",
        "combined_weighted_tie_break",
        "rf_delta_matched_row_count",
        "rf_delta_match_pct",
        "recompute_cell_count",
        "timing_total_sec",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug azimuth fallback candidate generation from a saved LTE tilt recommendation artifact."
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--steps", nargs="+", type=float, default=[5.0, 10.0, 15.0])
    parser.add_argument("--evaluate-rf", action="store_true", help="Run production RF validation for generated azimuth candidates.")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    summary_path = artifact_dir / "best_candidate_summary.json"
    candidate_path = artifact_dir / "candidate_validation_results.csv"
    before_scope_path = artifact_dir / "best_candidate_before_scope.csv.gz"
    grid_path = artifact_dir / "frontend_grid_scores.csv"

    for path in [summary_path, candidate_path, before_scope_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate_df = pd.read_csv(candidate_path)
    before_scope = pd.read_csv(before_scope_path)
    grid_analytics_df = pd.read_csv(grid_path) if grid_path.exists() else pd.DataFrame()
    antenna_df = _build_antenna_df(before_scope)

    thresholds = summary.get("thresholds") or {"rsrp": -90.0, "rsrq": -14.0, "sinr": 0.0}
    accepted_updates = _parse_updates(summary.get("updates"))
    available_artifact_cells = set(antenna_df["Node_Cell_ID"].astype(str).map(canonical_cell_id).tolist())
    target_cells = sorted(
        {
            _resolve_cell_for_artifact(cell, available_artifact_cells)
            for cell in _extract_target_cells(candidate_df)
        }
    )
    accepted_updates_for_eval = _resolve_update_ids_for_artifact(accepted_updates, available_artifact_cells)
    accepted_cells = {
        canonical_cell_id(update.get("cell_id"))
        for update in accepted_updates_for_eval
        if str(update.get("parameter", "ETilt")).strip().lower() == "etilt"
    }
    remaining_cells = [cell for cell in target_cells if cell not in accepted_cells]

    project_id = int(float(summary.get("project_id", before_scope.get("project_id", pd.Series([0])).iloc[0]) or 0))
    config = CandidateValidationConfig(
        project_id=project_id,
        region=str(summary.get("region", "") or ""),
        enable_azimuth_fallback=True,
        azimuth_fallback_steps_deg=tuple(abs(float(step)) for step in args.steps if abs(float(step)) > 0.0),
        candidate_workers=1,
        workers=1,
    )

    bearing_map = _bearing_context_map(before_scope, antenna_df, thresholds)
    antenna_after_etilt = _apply_updates_to_site_df(antenna_df, accepted_updates_for_eval)

    accepted_metrics: dict[str, Any] | None = None
    baseline_rf_cache: dict[tuple, pd.DataFrame] = {}
    k1k2_cache: dict[str, tuple[float, float]] = {}
    if args.evaluate_rf:
        print("[AZIMUTH_RF_DEBUG] evaluating accepted ETilt state first...")
        accepted_metrics, _ = _evaluate_update_set(
            updates=accepted_updates_for_eval,
            baseline_df=before_scope,
            antenna_df=antenna_df,
            geo_df=before_scope,
            grid_analytics_df=grid_analytics_df,
            thresholds=thresholds,
            weights=summary.get("weights") or {"rsrp": 0.2, "rsrq": 0.2, "sinr": 0.6},
            config=config,
            baseline_rf_cache=baseline_rf_cache,
            k1k2_cache=k1k2_cache,
        )
        print("[AZIMUTH_RF_DEBUG] accepted_etilt_metrics=", json.dumps(_metric_subset(accepted_metrics), default=str))

    rows: list[dict[str, Any]] = []
    rf_rows: list[dict[str, Any]] = []
    for cell in remaining_cells:
        cell_key = canonical_cell_id(cell)
        context = bearing_map.get(cell_key, {})
        ant_row = antenna_after_etilt.loc[
            antenna_after_etilt["Node_Cell_ID"].astype(str).map(canonical_cell_id) == cell_key
        ]
        current_azimuth = _safe_float(ant_row.iloc[0].get("azimuth")) if not ant_row.empty else np.nan
        dominant_bearing = _safe_float(context.get("dominant_bearing_deg"))
        signed_to_bearing = _signed_azimuth_delta(dominant_bearing, current_azimuth)
        generated: list[dict[str, Any]] = []
        for step in config.azimuth_fallback_steps_deg:
            update = _make_azimuth_update(
                antenna_df,
                cell_key,
                float(step),
                config,
                bearing_map,
                antenna_work_df=antenna_after_etilt,
            )
            if update:
                generated.append(update)
                if args.evaluate_rf and accepted_metrics is not None:
                    print(
                        "[AZIMUTH_RF_DEBUG] evaluating "
                        f"cell={cell_key} target={update.get('target_value')} delta={update.get('actual_delta')}"
                    )
                    try:
                        candidate_metrics, _ = _evaluate_update_set(
                            updates=accepted_updates_for_eval + [update],
                            baseline_df=before_scope,
                            antenna_df=antenna_df,
                            geo_df=before_scope,
                            grid_analytics_df=grid_analytics_df,
                            thresholds=thresholds,
                            weights=summary.get("weights") or {"rsrp": 0.2, "rsrq": 0.2, "sinr": 0.6},
                            config=config,
                            baseline_rf_cache=baseline_rf_cache,
                            k1k2_cache=k1k2_cache,
                        )
                        improved = _rank(candidate_metrics) > _rank(accepted_metrics)
                        rf_rows.append(
                            {
                                "cell_id": cell_key,
                                "current_azimuth": update.get("current_value"),
                                "target_azimuth": update.get("target_value"),
                                "actual_delta": update.get("actual_delta"),
                                "improved_vs_accepted_etilt": bool(improved),
                                "accepted_score": accepted_metrics.get("score"),
                                "candidate_score": candidate_metrics.get("score"),
                                "score_delta": _safe_float(candidate_metrics.get("score")) - _safe_float(accepted_metrics.get("score")),
                                "accepted_net_bad_reduction": accepted_metrics.get("net_bad_reduction"),
                                "candidate_net_bad_reduction": candidate_metrics.get("net_bad_reduction"),
                                "net_bad_reduction_delta": _safe_float(candidate_metrics.get("net_bad_reduction")) - _safe_float(accepted_metrics.get("net_bad_reduction")),
                                "accepted_new_bad_samples": accepted_metrics.get("new_bad_samples"),
                                "candidate_new_bad_samples": candidate_metrics.get("new_bad_samples"),
                                "constraints_passed": candidate_metrics.get("constraints_passed"),
                                "rf_delta_matched_row_count": candidate_metrics.get("rf_delta_matched_row_count"),
                                "rf_delta_match_pct": candidate_metrics.get("rf_delta_match_pct"),
                                "recompute_cell_count": candidate_metrics.get("recompute_cell_count"),
                                "timing_total_sec": candidate_metrics.get("timing_total_sec"),
                                "error": "",
                            }
                        )
                    except Exception as exc:
                        rf_rows.append(
                            {
                                "cell_id": cell_key,
                                "current_azimuth": update.get("current_value"),
                                "target_azimuth": update.get("target_value"),
                                "actual_delta": update.get("actual_delta"),
                                "improved_vs_accepted_etilt": False,
                                "error": str(exc),
                            }
                        )

        if ant_row.empty:
            skip_reason = "missing_antenna_row"
        elif not context:
            skip_reason = "missing_bearing_context"
        elif pd.isna(dominant_bearing):
            skip_reason = "missing_dominant_bearing"
        elif int(_safe_float(context.get("bearing_sample_count")) if pd.notna(_safe_float(context.get("bearing_sample_count"))) else 0) < MIN_BEARING_SAMPLE_COUNT:
            skip_reason = "bearing_sample_count_below_min"
        elif pd.isna(signed_to_bearing) or abs(float(signed_to_bearing)) < 5.0:
            skip_reason = "azimuth_already_close_to_dominant_bearing"
        elif not generated:
            skip_reason = "candidate_blocked_or_clamped_no_delta"
        else:
            skip_reason = ""

        rows.append(
            {
                "cell_id": cell_key,
                "current_azimuth": current_azimuth,
                "dominant_bearing_deg": dominant_bearing,
                "signed_delta_to_dominant_deg": signed_to_bearing,
                "bearing_sample_count": int(_safe_float(context.get("bearing_sample_count")) if pd.notna(_safe_float(context.get("bearing_sample_count"))) else 0),
                "bearing_mismatch_deg": _safe_float(context.get("bearing_mismatch_deg")),
                "bearing_peak_share": _safe_float(context.get("bearing_peak_share")),
                "bearing_spread_deg": _safe_float(context.get("bearing_spread_deg")),
                "bearing_directional_contrast": _safe_float(context.get("bearing_directional_contrast")),
                "generated_candidate_count": len(generated),
                "generated_candidates": json.dumps(
                    [
                        {
                            "step": update.get("requested_delta"),
                            "target": update.get("target_value"),
                            "actual_delta": update.get("actual_delta"),
                        }
                        for update in generated
                    ],
                    separators=(",", ":"),
                ),
                "skip_reason": skip_reason,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = artifact_dir / "azimuth_fallback_remaining_debug.csv"
    out_df.to_csv(out_path, index=False)

    generated_cells = int((out_df.get("generated_candidate_count", pd.Series(dtype=int)) > 0).sum()) if not out_df.empty else 0
    print("[AZIMUTH_DEBUG] artifact_dir=", artifact_dir)
    print(
        "[AZIMUTH_DEBUG] "
        f"target_cells={len(target_cells)} accepted_etilt_cells={len(accepted_cells)} "
        f"remaining_cells={len(remaining_cells)} bearing_context_cells={len(bearing_map)} "
        f"cells_with_azimuth_candidates={generated_cells}"
    )
    if not out_df.empty:
        print(out_df.to_string(index=False, max_colwidth=120))
    print("[AZIMUTH_DEBUG] wrote=", out_path)
    if args.evaluate_rf:
        rf_df = pd.DataFrame(rf_rows)
        rf_out_path = artifact_dir / "azimuth_fallback_rf_evaluation_debug.csv"
        rf_df.to_csv(rf_out_path, index=False)
        improved_count = int(rf_df.get("improved_vs_accepted_etilt", pd.Series(dtype=bool)).fillna(False).sum()) if not rf_df.empty else 0
        print(f"[AZIMUTH_RF_DEBUG] evaluated_candidates={len(rf_df)} improved_candidates={improved_count}")
        if not rf_df.empty:
            print(rf_df.to_string(index=False, max_colwidth=120))
        print("[AZIMUTH_RF_DEBUG] wrote=", rf_out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
