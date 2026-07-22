from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tests.coverage_prediction import model3_business_rule_recommendation_test as future_rules
from tests.coverage_prediction import model3_current_recommendation_test as current_rules


DEFAULT_OUTPUT_ROOT = ML_ROOT / "tests" / "output" / "model4_future_recommendation"
DEFAULT_STABLE_OUTPUT_DIR = ML_ROOT / "models" / "model4_future_recommendation_experiment"
DEFAULT_EXCEL_INPUT = ML_ROOT / "models" / "model3_project196_input" / "project_196_model3_input_725dd154_full_grid.xlsx"
DEFAULT_PROJECT196_GRID_DATASET = current_rules.DEFAULT_DATASET
DEFAULT_MODEL3_CURRENT_RECOMMENDATIONS = current_rules.DEFAULT_STABLE_OUTPUT_DIR / "model3_current_recommendations.csv"
DEFAULT_FUTURE_DATASET = future_rules.DEFAULT_MODEL3_DATASET
DEFAULT_FUTURE_SUMMARY = future_rules.DEFAULT_MODEL3_SUMMARY


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _clean_text(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    if "_" in text or "|" in text:
        return text
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except Exception:
        pass
    return text


def _band_text(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        return str(int(round(float(text))))
    except Exception:
        return text


def _canonical_sector(site_id: Any, sector_id: Any, node_cell_id: Any, canonical_cell_id: Any) -> str:
    site = _clean_text(site_id)
    sector_text = _clean_text(sector_id)
    if "|" in sector_text and not sector_text.lower().endswith("|nan"):
        return sector_text
    if "|" in sector_text:
        site = _clean_text(sector_text.split("|", 1)[0]) or site

    for value in [canonical_cell_id, node_cell_id]:
        text = _clean_text(value)
        if not text:
            continue
        tokens = [token for token in re.split(r"[_|]", text) if token and token.lower() != "nan"]
        if len(tokens) >= 2:
            return f"{site or tokens[0]}|{tokens[1]}"
    return f"{site}|1" if site else "unknown|1"


def _identity_keys(row: pd.Series) -> set[str]:
    keys: set[str] = set()
    for col in [
        "Node_Cell_ID",
        "canonical_physical_cell_id",
        "topology_original_node_cell_id",
        "topology_original_cell_id",
        "topology_rf_identity_key",
        "topology_site_sector_band_key",
    ]:
        if col in row.index:
            text = _clean_text(row.get(col))
            if text:
                keys.add(text)
                keys.add(re.sub(r"__(MB|ADD)\d+", "", text))
                keys.add(re.sub(r"__MB\d+", "", text))
    site = _clean_text(row.get("site_id") if "site_id" in row.index else row.get("topology_site_id", ""))
    band = _band_text(row.get("band") if "band" in row.index else row.get("topology_band", ""))
    sector = _canonical_sector(site, row.get("sector_id", ""), row.get("Node_Cell_ID", ""), row.get("canonical_physical_cell_id", ""))
    sector_num = sector.split("|", 1)[1] if "|" in sector else ""
    if site and sector_num:
        keys.add(f"{site}_{sector_num}")
        if band:
            keys.add(f"{site}_{sector_num}_{band}")
            keys.add(f"{site}_{site}_{sector_num}_{sector_num}_{band}")
    return {key for key in keys if key}


def _future_lookup(future_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if future_df.empty:
        return rows
    work = future_df.copy()
    for _, row in work.iterrows():
        for key in _identity_keys(row):
            if key not in rows:
                rows[key] = row.to_dict()
    return rows


def _pick_future_row(excel_row: pd.Series, lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for key in _identity_keys(excel_row):
        if key in lookup:
            return lookup[key]
    return None


def _boolish(value: Any) -> bool:
    text = _clean_text(value).lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0", ""}:
        return False
    try:
        return float(text) > 0
    except Exception:
        return bool(value)


def _archive_grid_id_lookup() -> pd.DataFrame:
    archive_path = future_rules.resolve_coverage_artifact_path()
    baseline = future_rules._read_csv_from_archive(archive_path, "baseline_prediction_grid.csv")
    if "time_bucket" in baseline.columns:
        baseline = baseline.loc[baseline["time_bucket"].astype(str).eq("PART_3")].copy()
    baseline["lat_6dp"] = pd.to_numeric(baseline["lat"], errors="coerce").round(6)
    baseline["lon_6dp"] = pd.to_numeric(baseline["lon"], errors="coerce").round(6)
    baseline["archive_grid_id"] = pd.to_numeric(baseline["grid_id"], errors="coerce")
    return baseline[["lat_6dp", "lon_6dp", "archive_grid_id"]].dropna().drop_duplicates(
        subset=["lat_6dp", "lon_6dp"], keep="first"
    )


def _model3_selected_cell_ids(limit: int | None) -> list[str]:
    if not DEFAULT_MODEL3_CURRENT_RECOMMENDATIONS.exists():
        return []
    reco = pd.read_csv(DEFAULT_MODEL3_CURRENT_RECOMMENDATIONS)
    ids: list[str] = []
    if "sector_congested_node_cell_ids" in reco.columns:
        for value in reco["sector_congested_node_cell_ids"].dropna().astype(str).tolist():
            for part in value.split(","):
                text = _clean_text(part)
                if text and text not in ids:
                    ids.append(text)
                if limit is not None and len(ids) >= int(limit):
                    return ids
    if "Node_Cell_ID" in reco.columns:
        for value in reco["Node_Cell_ID"].dropna().astype(str).tolist():
            text = _clean_text(value)
            if text and text not in ids:
                ids.append(text)
            if limit is not None and len(ids) >= int(limit):
                return ids
    return ids


def _project196_selected_cell_ids(cell_df: pd.DataFrame, limit: int | None, threshold: float) -> list[str]:
    if cell_df.empty:
        return []
    work = cell_df.copy()
    work["_current_pressure"] = work[["model4_current_prb", "model4_current_rrc"]].max(axis=1)
    work = work.loc[work["_current_pressure"] > float(threshold)].copy()
    if work.empty:
        return []
    scenario_order = [
        "load_balance_success",
        "carrier_addition_required",
        "sector_split_candidate",
        "carrier_blocked_new_site",
        "new_site_required",
        "",
    ]
    sector_groups: dict[str, list[tuple[str, pd.DataFrame]]] = {key: [] for key in scenario_order}
    for sector_id, group in work.groupby("topology_frontend_site_sector_key", dropna=False):
        scenario = _clean_text(group.get("model3_scenario", pd.Series([""])).iloc[0])
        sector_groups.setdefault(scenario, []).append((str(sector_id), group.copy()))
    for groups in sector_groups.values():
        groups.sort(key=lambda item: float(item[1]["_current_pressure"].max()), reverse=True)

    selected: list[str] = []
    selected_set: set[str] = set()
    target = int(limit) if limit is not None and int(limit) > 0 else None
    while target is None or len(selected) < target:
        progressed = False
        for scenario in scenario_order:
            groups = sector_groups.get(scenario, [])
            if not groups:
                continue
            _, group = groups.pop(0)
            group = group.sort_values(["_current_pressure", "Node_Cell_ID"], ascending=[False, True])
            for value in group["Node_Cell_ID"].dropna().astype(str).tolist():
                text = _clean_text(value)
                if text and text not in selected_set:
                    selected.append(text)
                    selected_set.add(text)
                    if target is not None and len(selected) >= target:
                        return selected
            progressed = True
        if not progressed:
            break
    return selected


def build_model4_future_dataset_from_excel(
    *,
    excel_path: Path,
    output_dir: Path,
    future_dataset_path: Path = DEFAULT_FUTURE_DATASET,
    max_congested_cells: int | None = 18,
    congestion_threshold: float = future_rules.DEFAULT_CONGESTION_THRESHOLD,
) -> tuple[Path, Path, dict[str, Any]]:
    if not excel_path.exists():
        raise FileNotFoundError(f"Project 196 Excel input not found: {excel_path}")
    cell_input = pd.read_excel(excel_path, sheet_name="Model3_Cell_Input", dtype=str)
    base_grid = pd.read_excel(excel_path, sheet_name="Baseline_Grid_Input", dtype=str)
    if "project_id" in cell_input.columns:
        cell_input = cell_input.loc[cell_input["project_id"].map(_clean_text).eq("196")].copy()
    if "site_id" in cell_input.columns:
        cell_input = cell_input.loc[cell_input["site_id"].map(_clean_text).str.fullmatch(r"\d+", na=False)].copy()
    if "project_id" in base_grid.columns:
        base_grid = base_grid.loc[base_grid["project_id"].map(_clean_text).eq("196")].copy()
    if "Node_Cell_ID" in base_grid.columns:
        base_grid["Node_Cell_ID"] = base_grid["Node_Cell_ID"].map(_clean_text)
        base_grid = base_grid.loc[base_grid["Node_Cell_ID"].map(_clean_text).ne("")].copy()
    future_df = pd.read_csv(future_dataset_path) if future_dataset_path.exists() else pd.DataFrame()
    lookup = _future_lookup(future_df)

    cell_rows: list[dict[str, Any]] = []
    matched = 0
    for idx, row in cell_input.iterrows():
        future_row = _pick_future_row(row, lookup)
        if future_row is not None:
            matched += 1
        site_id = _clean_text(row.get("site_id"))
        sector_id = _canonical_sector(site_id, row.get("sector_id"), row.get("Node_Cell_ID"), row.get("canonical_physical_cell_id"))
        band = _band_text(row.get("band"))
        current_prb = float(pd.to_numeric(pd.Series([row.get("input_prb_utilization_pct")]), errors="coerce").fillna(0.0).iloc[0])
        current_rrc = float(pd.to_numeric(pd.Series([row.get("input_rrc_utilization_pct")]), errors="coerce").fillna(0.0).iloc[0])
        future_prb = float(future_row.get("estimated_prb_utilization_pct")) if future_row is not None and pd.notna(future_row.get("estimated_prb_utilization_pct")) else min(180.0, current_prb * 1.12)
        future_rrc = float(future_row.get("estimated_cell_rrc_utilization_pct")) if future_row is not None and pd.notna(future_row.get("estimated_cell_rrc_utilization_pct")) else min(180.0, current_rrc * 1.12)
        raw_future_prb = future_prb
        raw_future_rrc = future_rrc
        future_prb = max(current_prb, future_prb)
        future_rrc = max(current_rrc, future_rrc)
        users = float(future_row.get("estimated_cell_rrc_connected_users")) if future_row is not None and pd.notna(future_row.get("estimated_cell_rrc_connected_users")) else float(pd.to_numeric(pd.Series([row.get("input_rrc_connected_users")]), errors="coerce").fillna(0.0).iloc[0]) * 1.12
        traffic = float(future_row.get("estimated_offered_traffic_mbps")) if future_row is not None and pd.notna(future_row.get("estimated_offered_traffic_mbps")) else float(pd.to_numeric(pd.Series([row.get("input_estimated_offered_traffic_mbps")]), errors="coerce").fillna(0.0).iloc[0]) * 1.12
        capacity = float(future_row.get("estimated_dl_capacity_mbps")) if future_row is not None and pd.notna(future_row.get("estimated_dl_capacity_mbps")) else float(pd.to_numeric(pd.Series([row.get("input_estimated_dl_capacity_mbps")]), errors="coerce").replace(0, np.nan).fillna(0.1).iloc[0])
        existing_count = int(float(pd.to_numeric(pd.Series([row.get("input_existing_carrier_count", row.get("existing_carrier_count", 0))]), errors="coerce").fillna(0).iloc[0]))
        max_supported = int(float(pd.to_numeric(pd.Series([row.get("input_max_supported_carriers", max(existing_count, 1))]), errors="coerce").fillna(max(existing_count, 1)).iloc[0]))
        available = _clean_text(row.get("input_available_bands_to_add", row.get("available_bands_to_add", "")))
        can_add = _boolish(row.get("carrier_addition_possible")) and bool(available)

        excel_node_cell_id = _clean_text(row.get("Node_Cell_ID"))
        cell_rows.append(
            {
                "Node_Cell_ID": excel_node_cell_id,
                "site_id": site_id,
                "topology_site_id": site_id,
                "topology_frontend_site_sector_key": sector_id,
                "topology_original_node_cell_id": _clean_text(row.get("canonical_physical_cell_id")) or excel_node_cell_id,
                "topology_original_cell_id": _clean_text(row.get("canonical_physical_cell_id")) or excel_node_cell_id,
                "topology_band": float(band) if band else np.nan,
                "topology_earfcn": row.get("earfcn"),
                "topology_sector": sector_id.split("|", 1)[1] if "|" in sector_id else "",
                "topology_azimuth": row.get("azimuth"),
                "topology_rf_identity_key": excel_node_cell_id,
                "topology_site_sector_band_key": f"{sector_id}_{band}" if band else sector_id,
                "estimated_prb_utilization_pct": round(future_prb, 3),
                "estimated_cell_rrc_utilization_pct": round(future_rrc, 3),
                "estimated_cell_rrc_connected_users": round(users, 3),
                "estimated_rrc_connected_users": round(users, 3),
                "estimated_offered_traffic_mbps": round(traffic, 3),
                "estimated_dl_capacity_mbps": round(max(0.1, capacity), 3),
                "model3_hotspot_score": max(future_prb, future_rrc),
                "model3_hotspot_rank": max(future_prb, future_rrc),
                "model3_scenario": _clean_text(row.get("demo_scenario")),
                "model3_scenario_reason": _clean_text(row.get("demo_scenario_reason")),
                "model4_current_prb": round(current_prb, 3),
                "model4_current_rrc": round(current_rrc, 3),
                "model4_raw_future_prb": round(raw_future_prb, 3),
                "model4_raw_future_rrc": round(raw_future_rrc, 3),
                "existing_carriers": _clean_text(row.get("existing_carriers")),
                "existing_carrier_count": existing_count,
                "available_bands_to_add": available,
                "carrier_addition_options": available,
                "available_band_options_count": len([v for v in re.split(r"[,|;/\s]+", available) if v]),
                "available_earfcns_to_add": available,
                "available_earfcn_options": available,
                "recommended_band_to_add": _clean_text(row.get("recommended_band_to_add")) or (available.split(",")[0].strip() if available else ""),
                "max_supported_carriers": max_supported,
                "sector_capacity_limit": max_supported,
                "sector_has_alternate_carrier": existing_count > 1,
                "carrier_addition_possible": bool(can_add and max_supported > existing_count),
                "carrier_addition_blocked": not bool(can_add and max_supported > existing_count),
                "carrier_addition_reason": "PROJECT196_EXCEL_AVAILABLE_BAND" if bool(can_add and max_supported > existing_count) else "PROJECT196_EXCEL_NO_AVAILABLE_BAND_OR_LIMIT",
                "model4_future_source": "model1_model2_future_match" if future_row is not None else "excel_current_uplift_fallback",
                "model4_excel_source_path": str(excel_path),
            }
        )

    cell_out = pd.DataFrame(cell_rows)
    cell_out["model4_pressure"] = cell_out[["estimated_prb_utilization_pct", "estimated_cell_rrc_utilization_pct"]].max(axis=1)

    selected_model3_ids = _model3_selected_cell_ids(max_congested_cells)
    scope_source = "model4_future_pressure"
    selected_model3_found = 0
    if selected_model3_ids:
        selected_set = set(selected_model3_ids)
        scoped = cell_out.loc[cell_out["Node_Cell_ID"].astype(str).isin(selected_set)].copy()
        selected_model3_found = int(scoped["Node_Cell_ID"].nunique(dropna=True))
        if selected_model3_found == len(selected_model3_ids):
            cell_out = scoped.copy()
            scope_source = "model3_current_selected_congested_cells"
    if scope_source != "model3_current_selected_congested_cells":
        selected_model3_ids = _project196_selected_cell_ids(cell_out, max_congested_cells, congestion_threshold)
        selected_set = set(selected_model3_ids)
        scoped = cell_out.loc[cell_out["Node_Cell_ID"].astype(str).isin(selected_set)].copy()
        selected_model3_found = int(scoped["Node_Cell_ID"].nunique(dropna=True))
        if selected_model3_found:
            cell_out = scoped.copy()
            scope_source = "project196_excel_model3_selected_congested_cells"
    selected_model3_missing = [
        cell_id for cell_id in selected_model3_ids if cell_id not in set(cell_out["Node_Cell_ID"].astype(str).tolist())
    ]

    if max_congested_cells is not None and int(max_congested_cells) > 0:
        if scope_source == "model4_future_pressure":
            congested = cell_out.loc[cell_out["model4_pressure"] > float(congestion_threshold)].copy()
            selected_sectors: list[str] = []
            covered = 0
            for sector_id, group in congested.sort_values("model4_pressure", ascending=False).groupby("topology_frontend_site_sector_key", sort=False):
                selected_sectors.append(str(sector_id))
                covered += int(group["Node_Cell_ID"].nunique())
                if covered >= int(max_congested_cells):
                    break
            if selected_sectors:
                cell_out = cell_out.loc[cell_out["topology_frontend_site_sector_key"].astype(str).isin(selected_sectors)].copy()

    overlay_cols = [
        "Node_Cell_ID",
        "estimated_prb_utilization_pct",
        "estimated_cell_rrc_utilization_pct",
        "estimated_cell_rrc_connected_users",
        "estimated_rrc_connected_users",
        "estimated_offered_traffic_mbps",
        "estimated_dl_capacity_mbps",
        "model3_hotspot_score",
        "model3_hotspot_rank",
        "model3_scenario",
        "model3_scenario_reason",
        "model4_current_prb",
        "model4_current_rrc",
        "model4_raw_future_prb",
        "model4_raw_future_rrc",
        "available_bands_to_add",
        "carrier_addition_options",
        "available_band_options_count",
        "available_earfcns_to_add",
        "available_earfcn_options",
        "recommended_band_to_add",
        "max_supported_carriers",
        "sector_capacity_limit",
        "sector_has_alternate_carrier",
        "carrier_addition_possible",
        "carrier_addition_blocked",
        "carrier_addition_reason",
        "model4_future_source",
        "model4_excel_source_path",
    ]
    overlay = cell_out[overlay_cols].drop_duplicates("Node_Cell_ID", keep="first").copy()
    out = base_grid.merge(overlay, on="Node_Cell_ID", how="inner", suffixes=("", "__future"))
    for col in overlay_cols:
        future_col = f"{col}__future"
        if future_col in out.columns:
            out[col] = out[future_col].combine_first(out[col]) if col in out.columns else out[future_col]
            out = out.drop(columns=[future_col])

    out["lat_6dp"] = pd.to_numeric(out.get("lat_6dp", out.get("lat")), errors="coerce").round(6)
    out["lon_6dp"] = pd.to_numeric(out.get("lon_6dp", out.get("lon")), errors="coerce").round(6)
    out["grid_label"] = out["grid_id"].map(_clean_text)
    coord_label = out["lat_6dp"].astype(str) + "_" + out["lon_6dp"].astype(str)
    out["grid_label"] = out["grid_label"].where(out["grid_label"].ne(""), coord_label)
    out = out.loc[out["lat_6dp"].notna() & out["lon_6dp"].notna()].copy()
    grid_lookup = _archive_grid_id_lookup()
    out = out.merge(grid_lookup, on=["lat_6dp", "lon_6dp"], how="left")
    fallback_grid_id = pd.factorize(out["grid_label"], sort=True)[0] + 1
    out["grid_id"] = pd.to_numeric(out["archive_grid_id"], errors="coerce").fillna(pd.Series(fallback_grid_id, index=out.index)).astype(int)
    out = out.drop(columns=["archive_grid_id"], errors="ignore")
    grid_row_src = out["grid_row"] if "grid_row" in out.columns else pd.Series(np.nan, index=out.index)
    grid_col_src = out["grid_col"] if "grid_col" in out.columns else pd.Series(np.nan, index=out.index)
    out["grid_row"] = pd.to_numeric(grid_row_src, errors="coerce").fillna(out["grid_id"]).astype(int)
    out["grid_col"] = pd.to_numeric(grid_col_src, errors="coerce").fillna(1).astype(int)
    out["grid_centroid_lat"] = pd.to_numeric(out.get("lat"), errors="coerce")
    out["grid_centroid_lon"] = pd.to_numeric(out.get("lon"), errors="coerce")
    out["rsrp_mean"] = pd.to_numeric(out.get("pred_rsrp"), errors="coerce")
    out["rsrq_mean"] = pd.to_numeric(out.get("pred_rsrq"), errors="coerce")
    out["sinr_mean"] = pd.to_numeric(out.get("pred_sinr"), errors="coerce")
    out["corrected_rsrp_mean"] = pd.to_numeric(out.get("pred_rsrp_smoothed", out.get("pred_rsrp")), errors="coerce")
    out["corrected_rsrq_mean"] = pd.to_numeric(out.get("pred_rsrq_smoothed", out.get("pred_rsrq")), errors="coerce")
    out["corrected_sinr_mean"] = pd.to_numeric(out.get("pred_sinr_smoothed", out.get("pred_sinr")), errors="coerce")
    out["site_id"] = out["site_id"].map(lambda value: _clean_text(value).removeprefix("s-"))
    out["topology_site_id"] = out["site_id"]
    out["sector_id"] = out.apply(
        lambda row: _canonical_sector(row.get("site_id"), row.get("sector"), row.get("Node_Cell_ID"), row.get("topology_match_id")),
        axis=1,
    )
    out["topology_frontend_site_sector_key"] = out["sector_id"]
    out["topology_original_node_cell_id"] = out["cell_id"].map(_clean_text)
    out["topology_original_cell_id"] = out["cell_id"].map(_clean_text)
    out["topology_band"] = pd.to_numeric(out.get("band"), errors="coerce")
    out["topology_earfcn"] = pd.to_numeric(out.get("earfcn"), errors="coerce") if "earfcn" in out.columns else np.nan
    out["topology_sector"] = out["sector"].map(_clean_text) if "sector" in out.columns else ""
    out["topology_rf_identity_key"] = out["cell_id"].map(_clean_text)
    out["topology_site_sector_band_key"] = out["sector_id"].astype(str) + "_" + out["topology_band"].fillna("").astype(str)
    sample_src = out["sample_count"] if "sample_count" in out.columns else pd.Series(1, index=out.index)
    grid_size_src = out["grid_size_m"] if "grid_size_m" in out.columns else pd.Series(25.0, index=out.index)
    out["sample_count"] = pd.to_numeric(sample_src, errors="coerce").fillna(1)
    out["grid_size_m"] = pd.to_numeric(grid_size_src, errors="coerce").fillna(25.0)
    out["grid_area_m2"] = out["grid_size_m"] * out["grid_size_m"]
    out["estimated_dl_capacity_mbps"] = pd.to_numeric(out["estimated_dl_capacity_mbps"], errors="coerce").fillna(0.1).clip(lower=0.1)
    out["estimated_prb_utilization_pct"] = pd.to_numeric(out["estimated_prb_utilization_pct"], errors="coerce").fillna(0.0)
    out["estimated_cell_rrc_utilization_pct"] = pd.to_numeric(out["estimated_cell_rrc_utilization_pct"], errors="coerce").fillna(0.0)
    out["estimated_cell_rrc_connected_users"] = pd.to_numeric(out["estimated_cell_rrc_connected_users"], errors="coerce").fillna(
        out["estimated_cell_rrc_utilization_pct"] * (float(future_rules.DEFAULT_RRC_SECTOR_CAPACITY) / 100.0)
    )
    out["estimated_offered_traffic_mbps"] = (
        out["estimated_dl_capacity_mbps"] * (out["estimated_prb_utilization_pct"] / 100.0)
    ).round(3)
    per_cell_count = out.groupby("Node_Cell_ID")["grid_id"].transform("nunique").replace(0, 1)
    out["estimated_rrc_connected_users"] = (out["estimated_cell_rrc_connected_users"] / per_cell_count).round(3)
    out["time_bucket"] = "FUTURE"
    out["model3_mode"] = "future"
    out["model4_pressure"] = out[["estimated_prb_utilization_pct", "estimated_cell_rrc_utilization_pct"]].max(axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "model4_project196_future_dataset.csv"
    summary_path = output_dir / "model4_project196_future_dataset_summary.json"
    out.to_csv(dataset_path, index=False)
    summary = {
        "mode": "model4_project196_excel_future_dataset",
        "excel_path": str(excel_path),
        "source_grid_dataset_path": f"{excel_path}::Baseline_Grid_Input",
        "future_dataset_path": str(future_dataset_path),
        "rows": int(len(out)),
        "cell_count": int(out["Node_Cell_ID"].nunique(dropna=True)),
        "sector_count": int(out["topology_frontend_site_sector_key"].nunique(dropna=True)),
        "future_matched_rows_before_scope": int(matched),
        "scope_source": scope_source,
        "model3_selected_cell_count": int(len(selected_model3_ids)),
        "model3_selected_cells_found": int(selected_model3_found),
        "model3_selected_cells_missing": selected_model3_missing,
        "max_congested_cells_scope": int(max_congested_cells) if max_congested_cells is not None else None,
        "threshold": float(congestion_threshold),
        "congested_cell_count": int(
            (
                out.drop_duplicates("Node_Cell_ID")[["estimated_prb_utilization_pct", "estimated_cell_rrc_utilization_pct"]]
                .max(axis=1)
                > float(congestion_threshold)
            ).sum()
        ),
        "source_counts": {str(k): int(v) for k, v in cell_out["model4_future_source"].value_counts(dropna=False).items()},
        "dataset_csv": str(dataset_path),
        "summary_json": str(summary_path),
    }
    _save_json(summary_path, summary)
    return dataset_path, summary_path, summary


def run_model4_future_recommendation(
    config: future_rules.Model3RecommendationConfig,
    *,
    excel_path: Path,
    future_dataset_path: Path = DEFAULT_FUTURE_DATASET,
    max_congested_cells: int | None = 18,
) -> Path:
    stable_dir = config.stable_output_dir
    stable_dir.mkdir(parents=True, exist_ok=True)
    dataset_path, summary_path, model4_dataset_summary = build_model4_future_dataset_from_excel(
        excel_path=excel_path,
        output_dir=stable_dir,
        future_dataset_path=future_dataset_path,
        max_congested_cells=max_congested_cells,
        congestion_threshold=config.congestion_threshold,
    )
    run_config = current_rules.CurrentModel3Config(
        dataset_path=dataset_path,
        summary_path=summary_path,
        output_root=config.output_root,
        stable_output_dir=config.stable_output_dir,
        congestion_threshold=config.congestion_threshold,
        rrc_sector_capacity=config.rrc_sector_capacity,
        sector_split_local_radius_m=config.sector_split_local_radius_m,
        rf_workers=3,
        max_interference_sites=10,
        action_neighbor_cells=2,
        sector_parallelism=3,
    )
    run_dir = current_rules.run_model3_current_recommendation_test(run_config)

    rename_pairs = [
        ("model3_current_recommendations.xlsx", "model4_future_recommendations.xlsx"),
        ("model3_current_recommendations.csv", "model4_future_recommendations.csv"),
        ("summary.json", "model4_future_recommendation_summary.json"),
        ("log.txt", "model4_future_recommendation.log"),
    ]
    for src_name, dest_name in rename_pairs:
        src = run_dir / src_name
        if src.exists():
            try:
                shutil.copy2(src, stable_dir / dest_name)
            except PermissionError:
                pass
    payload = {
        "mode": "future_model4_recommendation",
        "run_dir": str(run_dir),
        "stable_output_dir": str(stable_dir),
        "excel_path": str(excel_path),
        "model4_dataset_summary": model4_dataset_summary,
        "engine_summary": json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if (run_dir / "summary.json").exists()
        else {},
        "files": {
            "workbook": str(stable_dir / "model4_future_recommendations.xlsx"),
            "recommendations_csv": str(stable_dir / "model4_future_recommendations.csv"),
            "summary_json": str(stable_dir / "model4_future_recommendation_summary.json"),
            "log": str(stable_dir / "model4_future_recommendation.log"),
        },
    }
    _save_json(stable_dir / "model4_future_recommendation_summary.json", payload)
    print(json.dumps(payload, indent=2, default=str))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run future-state Model 4 recommendations from the Project 196 Excel input.")
    parser.add_argument("--excel-path", type=Path, default=DEFAULT_EXCEL_INPUT)
    parser.add_argument("--future-dataset-path", type=Path, default=DEFAULT_FUTURE_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stable-output-dir", type=Path, default=DEFAULT_STABLE_OUTPUT_DIR)
    parser.add_argument("--congestion-threshold", type=float, default=future_rules.DEFAULT_CONGESTION_THRESHOLD)
    parser.add_argument("--rrc-sector-capacity", type=float, default=future_rules.DEFAULT_RRC_SECTOR_CAPACITY)
    parser.add_argument("--max-congested-cells", type=int, default=18)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = future_rules.Model3RecommendationConfig(
        dataset_path=Path(""),
        summary_path=Path(""),
        output_root=args.output_root,
        stable_output_dir=args.stable_output_dir,
        congestion_threshold=args.congestion_threshold,
        rrc_sector_capacity=args.rrc_sector_capacity,
    )
    run_model4_future_recommendation(
        cfg,
        excel_path=args.excel_path,
        future_dataset_path=args.future_dataset_path,
        max_congested_cells=args.max_congested_cells,
    )
