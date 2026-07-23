from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tests.coverage_prediction import model3_current_recommendation_test as current_rules
from tests.coverage_prediction import model3_business_rule_recommendation_test as future_rules
from tools.lte_prediction_optimised.ml_engine import run_prediction_only_optimized as production_run_prediction_only_optimized


INPUT_DIR = ML_ROOT / "models" / "model3_project196_input"
MODEL4_DIR = ML_ROOT / "models" / "model4_future_recommendation_experiment"
OUTPUT_CSV = MODEL4_DIR / "model4_full_after_baseline_rf_surface.csv"
OUTPUT_SUMMARY = MODEL4_DIR / "model4_full_after_baseline_summary.json"


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def _sector_number(sector_id: Any, node_cell_id: Any) -> str:
    sector_text = _clean(sector_id)
    if "|" in sector_text and not sector_text.lower().endswith("|nan"):
        return sector_text.split("|", 1)[1]
    parts = [part for part in re.split(r"[_|]", _clean(node_cell_id)) if part and part.lower() != "nan"]
    if len(parts) >= 3:
        return parts[-2]
    if len(parts) >= 2:
        return parts[-1]
    return ""


def _prepare_baseline_points() -> pd.DataFrame:
    baseline = pd.read_csv(INPUT_DIR / "project_196_model3_baseline_grid_input.csv", low_memory=False)
    baseline["time_bucket"] = "PART_3"
    baseline["lat"] = pd.to_numeric(baseline["lat"], errors="coerce")
    baseline["lon"] = pd.to_numeric(baseline["lon"], errors="coerce")
    baseline["original_full_node_cell_id"] = baseline["Node_Cell_ID"].map(_clean)
    baseline["nodeb_id_cell_id"] = baseline["Node_Cell_ID"]
    baseline["canonical_cell_id"] = baseline["Node_Cell_ID"].map(future_rules._normalize_identity_text)
    baseline["frontend_site_sector_key"] = baseline.apply(
        lambda row: f"{_clean(row.get('site_id')).removeprefix('s-')}|{_sector_number(row.get('sector'), row.get('Node_Cell_ID'))}",
        axis=1,
    )
    return baseline.dropna(subset=["lat", "lon"]).copy()


def _expand_site_rows_to_project196_rf_keys(site_df: pd.DataFrame, baseline_points: pd.DataFrame) -> pd.DataFrame:
    if site_df.empty or baseline_points.empty:
        return site_df.copy()
    keys = (
        baseline_points.loc[:, ["Node_Cell_ID", "cell_id", "site_id", "sector", "band"]]
        .dropna(subset=["Node_Cell_ID"])
        .drop_duplicates()
        .copy()
    )
    if keys.empty:
        return site_df.copy()

    work = site_df.copy()
    work["_join_cell_id"] = work.get("cell_id", work.get("Node_Cell_ID", "")).map(_clean)
    work["_join_site_id"] = work.get("site_id", work.get("Site ID", "")).map(_clean).str.removeprefix("s-")
    if not work["_join_site_id"].str.len().any() and "Site ID" in work.columns:
        work["_join_site_id"] = work["Site ID"].map(_clean).str.removeprefix("s-")
    work["_join_band"] = pd.to_numeric(work.get("band"), errors="coerce").astype("Int64").astype(str)

    keys["_join_cell_id"] = keys["cell_id"].map(_clean)
    keys["_join_site_id"] = keys["site_id"].map(_clean).str.removeprefix("s-")
    keys["_join_band"] = pd.to_numeric(keys["band"], errors="coerce").astype("Int64").astype(str)

    exact = work.drop_duplicates(subset=["_join_cell_id", "_join_site_id", "_join_band"], keep="first")
    fallback = work.drop_duplicates(subset=["_join_cell_id", "_join_site_id"], keep="first")
    exact_rows = keys.merge(
        exact,
        on=["_join_cell_id", "_join_site_id", "_join_band"],
        how="left",
        suffixes=("_project196", ""),
    )
    missing = exact_rows["Node_Cell_ID"].isna() if "Node_Cell_ID" in exact_rows.columns else pd.Series(True, index=exact_rows.index)
    if missing.any():
        fill_rows = keys.loc[missing.to_numpy()].merge(
            fallback,
            on=["_join_cell_id", "_join_site_id"],
            how="left",
            suffixes=("_project196", ""),
        )
        for col in fill_rows.columns:
            if col not in exact_rows.columns:
                exact_rows[col] = pd.NA
        common_cols = [col for col in fill_rows.columns if col in exact_rows.columns]
        exact_rows.loc[missing, common_cols] = fill_rows[common_cols].to_numpy()
    merged = exact_rows
    if merged.empty or "Node_Cell_ID_project196" not in merged.columns:
        return site_df.copy()
    merged["legacy_nodeb_id_cell_id"] = merged.get("legacy_nodeb_id_cell_id", merged["_join_cell_id"])
    merged["sector_identity_key"] = merged.get("Node_Cell_ID", merged["_join_cell_id"])
    merged["Node_Cell_ID"] = merged["Node_Cell_ID_project196"].map(_clean)
    merged["rf_identity_key"] = merged["Node_Cell_ID"]
    merged["nodeb_id_cell_id"] = merged["Node_Cell_ID"]
    merged["canonical_cell_id"] = merged["Node_Cell_ID"].map(future_rules._normalize_identity_text)
    if "band_project196" in merged.columns:
        merged["band"] = merged["band_project196"]
    if "site_id_project196" in merged.columns:
        merged["site_id"] = merged["site_id_project196"]
    if "sector_project196" in merged.columns:
        merged["sector"] = merged["sector_project196"]
    centroids = (
        baseline_points.groupby("Node_Cell_ID", dropna=False)[["lat", "lon"]]
        .mean(numeric_only=True)
        .rename(columns={"lat": "_centroid_lat", "lon": "_centroid_lon"})
        .reset_index()
    )
    merged = merged.merge(centroids, on="Node_Cell_ID", how="left")
    if "lat" in merged.columns:
        merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce").fillna(merged["_centroid_lat"])
    if "lon" in merged.columns:
        merged["lon"] = pd.to_numeric(merged["lon"], errors="coerce").fillna(merged["_centroid_lon"])
    for col, default in [("height", 25.0), ("azimuth", 0.0), ("m_tilt", 0.0), ("e_tilt", 2.0), ("tx_power", 43.0)]:
        if col not in merged.columns:
            merged[col] = default
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(default)
    merged = merged.drop(columns=[c for c in ["Node_Cell_ID_project196", "_join_cell_id", "_join_site_id", "_join_band"] if c in merged.columns])
    subset = [col for col in ["Node_Cell_ID", "lat", "lon", "azimuth"] if col in merged.columns]
    return merged.drop_duplicates(subset=subset or ["Node_Cell_ID"], keep="first").copy()


def _baseline_point_alias_mask(points: pd.DataFrame, source_rows: pd.DataFrame) -> pd.Series:
    aliases: set[str] = set()
    for frame in [source_rows]:
        for col in ["Node_Cell_ID", "cell_id", "original_node_cell_id", "original_cell_id", "rf_identity_key"]:
            if col in frame.columns:
                aliases.update(_clean(value) for value in frame[col].dropna().tolist())
                aliases.update(future_rules._normalize_identity_text(value) for value in frame[col].dropna().tolist())
    aliases = {value for value in aliases if value}
    return current_rules._vectorized_identity_point_mask(points, sorted(aliases))


def _append_prediction_points_for_new_rows(
    points: pd.DataFrame,
    baseline_points: pd.DataFrame,
    source_rows: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> pd.DataFrame:
    if new_rows.empty:
        return points
    lineage = baseline_points.loc[_baseline_point_alias_mask(baseline_points, source_rows)].copy()
    if lineage.empty:
        return points
    parts = [points]
    for _, row in new_rows.iterrows():
        new_id = _clean(row.get("Node_Cell_ID") or row.get("cell_id"))
        if not new_id:
            continue
        dup = lineage.copy()
        dup["Node_Cell_ID"] = new_id
        dup["nodeb_id_cell_id"] = new_id
        dup["canonical_cell_id"] = future_rules._normalize_identity_text(new_id)
        dup["cell_id"] = new_id
        dup["site_id"] = row.get("Site ID", row.get("site_id", dup.get("site_id", "")))
        if "band" in row.index:
            dup["band"] = row.get("band")
        if "earfcn" in row.index:
            dup["earfcn"] = row.get("earfcn")
        parts.append(dup)
    out = pd.concat(parts, ignore_index=True, sort=False)
    return out.drop_duplicates(subset=["Node_Cell_ID", "lat", "lon"], keep="last")


def _append_full_after_prediction_points(
    *,
    points: pd.DataFrame,
    baseline_points: pd.DataFrame,
    source_rows: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> pd.DataFrame:
    if new_rows.empty or source_rows.empty:
        return points
    lineage = baseline_points.loc[_baseline_point_alias_mask(baseline_points, source_rows)].copy()
    if lineage.empty:
        return points

    parts = [points]
    for _, row in new_rows.iterrows():
        new_id = _clean(row.get("Node_Cell_ID") or row.get("cell_id"))
        if not new_id:
            continue
        dup = lineage.copy()
        dup["Node_Cell_ID"] = new_id
        dup["nodeb_id_cell_id"] = new_id
        dup["canonical_cell_id"] = future_rules._normalize_identity_text(new_id)
        for col in ["rf_identity_key", "site_sector_band_key", "sector_identity_key"]:
            if col in dup.columns:
                dup[col] = new_id
        for col in ["site_id", "sector", "band", "earfcn", "cell_id", "operator", "Technology"]:
            if col in dup.columns and col in row.index:
                dup[col] = row.get(col)
        parts.append(dup)

    out = pd.concat(parts, ignore_index=True, sort=False)
    dedupe_cols = [col for col in ["Node_Cell_ID", "lat", "lon", "grid_id", "time_bucket"] if col in out.columns]
    return out.drop_duplicates(subset=dedupe_cols, keep="last") if dedupe_cols else out


def _merge_point_metadata(pred_df: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty or points.empty:
        return pred_df
    key_cols = [col for col in ["Node_Cell_ID", "lat", "lon"] if col in pred_df.columns and col in points.columns]
    if len(key_cols) < 3:
        return pred_df
    meta_cols = [
        col for col in [
            "Node_Cell_ID",
            "lat",
            "lon",
            "grid_id",
            "frontend_grid_id",
            "project_id",
            "baseline_band",
            "band",
            "sector",
            "site_id",
            "cell_id",
            "operator",
            "Technology",
            "time_bucket",
        ] if col in points.columns
    ]
    pred_work = pred_df.copy()
    point_work = points[meta_cols].copy()
    for frame in [pred_work, point_work]:
        frame["Node_Cell_ID"] = frame["Node_Cell_ID"].map(_clean)
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
        frame["_lat_key"] = frame["lat"].round(9)
        frame["_lon_key"] = frame["lon"].round(9)
    join_cols = ["Node_Cell_ID", "_lat_key", "_lon_key"]
    meta = point_work.dropna(subset=["Node_Cell_ID", "lat", "lon"]).drop_duplicates(subset=join_cols, keep="last")
    pred_work = pred_work.dropna(subset=["Node_Cell_ID", "lat", "lon"]).drop_duplicates(subset=join_cols, keep="last")
    out = pred_work.drop(columns=[col for col in meta.columns if col not in join_cols and col in pred_work.columns], errors="ignore")
    out = out.merge(meta, on=join_cols, how="inner", suffixes=("", "_point"), validate="one_to_one")
    for col in ["lat", "lon"]:
        point_col = f"{col}_point"
        if point_col in out.columns:
            out[col] = out[point_col]
    return out.drop(columns=[col for col in out.columns if col.endswith("_point") or col in {"_lat_key", "_lon_key"}], errors="ignore")


def _action_band(row: pd.Series) -> str:
    band = _clean(row.get("recommended_band_to_add"))
    action = _clean(row.get("action"))
    if not band and "->" in action:
        band = action.split("->", 1)[1].replace("MHz", "").strip()
    return band


def _action_sector_cell_count(row: pd.Series) -> int:
    action = _clean(row.get("action"))
    match = re.search(r"Add Sector\s*->\s*(\d+)", action, flags=re.IGNORECASE)
    if match:
        return max(1, int(match.group(1)))
    return 1


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()


def _sanitize_for_optimized_normalizer(df: pd.DataFrame) -> pd.DataFrame:
    out = _dedupe_columns(df)
    rename_pairs = [
        ("frequency", "frequency_mhz"),
        ("Frequency", "frequency_mhz"),
        ("e_tilt", "electrical_tilt"),
        ("Etilt", "electrical_tilt"),
        ("m_tilt", "mechanical_tilt"),
        ("Mtilt", "mechanical_tilt"),
        ("height", "antenna_height"),
        ("Height", "antenna_height"),
        ("latitude", "lat"),
        ("longitude", "lon"),
    ]
    for src, dest in rename_pairs:
        if src in out.columns and dest in out.columns:
            out[dest] = out[dest].where(out[dest].notna(), out[src])
            out = out.drop(columns=[src])
    return _dedupe_columns(out)


def main() -> None:
    started = time.perf_counter()
    MODEL4_DIR.mkdir(parents=True, exist_ok=True)
    rf_workers = min(31, max(1, (os.cpu_count() or 4) - 1))
    config = current_rules.CurrentModel3Config(
        dataset_path=MODEL4_DIR / "model4_project196_future_dataset.csv",
        summary_path=MODEL4_DIR / "model4_project196_future_dataset_summary.json",
        congestion_threshold=70.0,
        rf_workers=rf_workers,
        max_interference_sites=10,
        action_neighbor_cells=2,
    )
    logger = current_rules._setup_logger(MODEL4_DIR / "model4_full_after_baseline.log")
    source_df = pd.read_csv(config.dataset_path, low_memory=False)
    cell_inventory, _ = current_rules._build_current_cell_inventory(source_df, config)
    recommendations = pd.read_csv(MODEL4_DIR / "model4_future_recommendations.csv", low_memory=False)
    context = current_rules._load_current_context(config, logger)
    modified_site_df = context["part3_site_df"].copy()
    baseline_points = _prepare_baseline_points()
    modified_site_df = _expand_site_rows_to_project196_rf_keys(modified_site_df, baseline_points)
    full_after_points = baseline_points.copy()

    for _, rec in recommendations.iterrows():
        sector_id = _clean(rec.get("sector_id"))
        sector_cells = cell_inventory.loc[cell_inventory["sector_id"].map(_clean).eq(sector_id)].copy()
        if sector_cells.empty:
            continue
        before_ids = set(modified_site_df["Node_Cell_ID"].dropna().astype(str))
        action = _clean(rec.get("action"))
        if "Add Carrier" in action:
            modified_site_df, source_rows, _ = current_rules._build_current_carrier_addition_topology(
                sector_cells,
                modified_site_df,
                _action_band(rec),
                logger,
            )
        elif "Add Sector" in action:
            modified_site_df, source_rows = current_rules._build_current_add_sector_topology(
                sector_cells,
                modified_site_df,
                context,
                logger,
                carrier_count=_action_sector_cell_count(rec),
            )
        elif "Sector Split" in action:
            modified_site_df, source_rows = current_rules._build_current_sector_split_topology(
                sector_cells,
                modified_site_df,
                context,
                logger,
            )
        elif "New Site" in action:
            modified_site_df, source_rows = current_rules._build_current_new_site_topology(
                sector_cells,
                modified_site_df,
                context,
                logger,
            )
        else:
            continue
        new_rows = modified_site_df.loc[
            ~modified_site_df["Node_Cell_ID"].dropna().astype(str).isin(before_ids)
        ].copy()
        full_after_points = _append_full_after_prediction_points(
            points=full_after_points,
            baseline_points=baseline_points,
            source_rows=source_rows,
            new_rows=new_rows,
        )
        modified_site_df = _sanitize_for_optimized_normalizer(current_rules._fix_synthetic_frequency(modified_site_df))

    recompute_cells = sorted(
        {
            _clean(value)
            for value in modified_site_df.get("Node_Cell_ID", pd.Series(dtype=object)).dropna().astype(str).tolist()
            if _clean(value)
        }
    )
    full_after_points = full_after_points.loc[
        full_after_points["Node_Cell_ID"].astype(str).isin(recompute_cells)
    ].copy()
    run_params = {
        "radius": 500.0,
        "grid_resolution": 25.0,
        "n_workers": rf_workers,
        "antenna_gain": 18,
        "cable_loss": 2,
        "ue_height": 1.5,
        "frequency_mhz": 1800,
        "bandwidth_mhz": 10,
        "project_id": 196,
        "region": "india",
        "max_interference_sites": 10,
        "prediction_points_df": full_after_points,
        "strict_prediction_points": True,
        "baseline_df": full_after_points,
        "recompute_cells": recompute_cells,
    }
    modified_site_df = _sanitize_for_optimized_normalizer(modified_site_df)
    after = production_run_prediction_only_optimized(modified_site_df, {}, run_params)
    after = _merge_point_metadata(after, full_after_points)
    after.to_csv(OUTPUT_CSV, index=False)
    summary = {
        "mode": "model4_full_after_baseline_strict_saved_points",
        "baseline_point_rows_before": int(len(baseline_points)),
        "baseline_cells_before": int(baseline_points["Node_Cell_ID"].nunique()),
        "after_prediction_point_rows": int(len(full_after_points)),
        "after_prediction_point_cells": int(full_after_points["Node_Cell_ID"].nunique()),
        "site_cells_after": int(modified_site_df["Node_Cell_ID"].nunique()) if "Node_Cell_ID" in modified_site_df.columns else 0,
        "after_rf_rows": int(len(after)),
        "after_rf_cells": int(after["Node_Cell_ID"].nunique()) if "Node_Cell_ID" in after.columns else 0,
        "after_frontend_grids": int(after["grid_id"].nunique()) if "grid_id" in after.columns else 0,
        "rf_workers": int(rf_workers),
        "output_csv": str(OUTPUT_CSV),
        "runtime_sec": round(time.perf_counter() - started, 3),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
