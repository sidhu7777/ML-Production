from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
load_dotenv(ML_ROOT / ".env")

from tests.debug.lte_tilt_recommendation_debug import (  # noqa: E402
    DEFAULT_PAYLOAD,
    _fetch_source,
    _redact_payload,
)
from tools.lte_prediction_optimised import ml_engine as opt_ml  # noqa: E402
from tools.lte_tilt_recommandation.candidate_validation import (  # noqa: E402
    _attach_grid_context,
    _attach_grid_context_from_analytics,
    _clean_id,
    _combined_grid_reference,
    _ensure_node_cell,
    _prepare_optimizer_site_df,
    _severity,
    _weighted_kpi_cell_order,
    normalize_grid_id_series,
)
from tools.lte_tilt_recommandation.cell_identity import canonical_cell_id  # noqa: E402
from tools.lte_tilt_recommandation.services import _rf_identity_series  # noqa: E402


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text_value = str(value).strip().lower()
    if text_value in {"1", "true", "yes", "y", "on"}:
        return True
    if text_value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _site_from_canonical(value: object) -> str:
    text = canonical_cell_id(value)
    parts = text.split("_")
    return parts[0] if len(parts) >= 2 else text


def _present_text(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.isna() | out.str.lower().isin(["", "none", "nan", "null", "<na>"]), "")


def _db_sector_key(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="string")
    sector = _present_text(df["sector"]) if "sector" in df.columns else pd.Series("", index=df.index, dtype="string")
    site_parts = []
    for col in ["site_id", "node_b_id", "nodeb_id"]:
        if col in df.columns:
            site_parts.append(_present_text(df[col]))
    site = site_parts[0] if site_parts else pd.Series("", index=df.index, dtype="string")
    for candidate in site_parts[1:]:
        site = site.where(site.ne(""), candidate)
    cell = _present_text(df["cell_id"]) if "cell_id" in df.columns else pd.Series("", index=df.index, dtype="string")
    key = site + "|" + sector + "|" + cell
    return key.mask(site.eq("") | sector.eq("") | cell.eq(""), "")


def _baseline_source_key(df: pd.DataFrame) -> pd.Series:
    if "nodeb_id_cell_id" in df.columns:
        return _rf_identity_series(df["nodeb_id_cell_id"])
    if {"node_b_id", "cell_id"}.issubset(df.columns):
        return _rf_identity_series(_present_text(df["node_b_id"]) + "|" + _present_text(df["cell_id"]))
    return pd.Series("", index=df.index, dtype="string")


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(DEFAULT_PAYLOAD)
    for key in DEFAULT_PAYLOAD:
        value = getattr(args, key, None)
        if value is not None:
            payload[key] = value
    for key in ["bridge_base_url", "bridge_api_key", "database_url", "database_url_taiwan"]:
        value = getattr(args, key, None)
        if value:
            payload[key] = value
    if args.threshold_file_path:
        payload["threshold_file_path"] = args.threshold_file_path
    return payload


def _build_tunable_sets(antenna_df: pd.DataFrame) -> tuple[pd.DataFrame, set[str], set[str]]:
    ant = opt_ml._normalize_site_df(_prepare_optimizer_site_df(antenna_df), log_stage="TARGET_DEDUPE_DEBUG")
    tunable_aliases: set[str] = set()
    for col in ["Node_Cell_ID", "local_cell_id", "cell_id", "source_local_cell_id", "canonical_cell_id"]:
        if col in ant.columns:
            values = ant[col].map(_clean_id)
            tunable_aliases.update(v for v in values.astype(str).str.strip().tolist() if v)
            tunable_aliases.update(v for v in values.map(canonical_cell_id).astype(str).str.strip().tolist() if v)
    if {"nodeb_id", "cell_id"}.issubset(ant.columns):
        tunable_aliases.update(
            (ant["nodeb_id"].map(_clean_id) + "_" + ant["cell_id"].map(_clean_id)).astype(str).tolist()
        )
    if {"node_b_id", "cell_id"}.issubset(ant.columns):
        tunable_aliases.update(
            (ant["node_b_id"].map(_clean_id) + "_" + ant["cell_id"].map(_clean_id)).astype(str).tolist()
        )
    tunable_aliases = {item for item in tunable_aliases if item and item.lower() not in {"nan", "none", "null"}}
    tunable_canonical = {canonical_cell_id(item) for item in tunable_aliases if canonical_cell_id(item)}
    return ant, tunable_aliases, tunable_canonical


def _add_bad_columns(work: pd.DataFrame, thresholds: dict[str, float], weights: dict[str, float]) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = work.copy()
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
    return work.loc[work["_combined_severity"] > 0].copy(), bad_cols, severity_cols


def _group_work(work: pd.DataFrame, group_col: str, bad_cols: list[str], severity_cols: list[str]) -> pd.DataFrame:
    grouped = (
        work.groupby(group_col, dropna=False)
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
    grouped = grouped.rename(columns={group_col: "Cell ID"})
    for col in ["Bad RSRP", "Bad RSRQ", "Bad SINR", "rsrp_bad_severity", "rsrq_bad_severity", "sinr_bad_severity"]:
        if col not in grouped.columns:
            grouped[col] = 0.0
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0)
    return grouped


def _select_by_coverage(grouped: pd.DataFrame, weights: dict[str, float], coverage_pct: float, max_group_cells: int) -> pd.DataFrame:
    ordered = _weighted_kpi_cell_order(grouped, weights)
    if ordered.empty:
        return ordered
    contribution_col = "combined_grid_severity"
    max_weight = max(float(weights.get("rsrp", 0.0)), float(weights.get("rsrq", 0.0)), float(weights.get("sinr", 0.0)))
    priority_kpis = [
        kpi for kpi in ["rsrp", "rsrq", "sinr"]
        if float(weights.get(kpi, 0.0)) == max_weight and float(weights.get(kpi, 0.0)) > 0.0
    ]
    if len(priority_kpis) == 1:
        contribution_col = f"{priority_kpis[0]}_bad_severity"
    ordered["total_severity"] = pd.to_numeric(ordered[contribution_col], errors="coerce").fillna(0.0)
    total = float(ordered["total_severity"].sum())
    ordered["contribution_pct"] = ordered["total_severity"] / max(total, 1e-9) * 100.0
    ordered["cumulative_pct"] = ordered["contribution_pct"].cumsum()
    coverage = float(np.clip(float(coverage_pct), 1.0, 100.0))
    selected = ordered.loc[ordered["cumulative_pct"] <= coverage].copy()
    if selected.empty:
        selected = ordered.head(1).copy()
    elif len(selected) < len(ordered):
        selected = pd.concat([selected, ordered.iloc[[len(selected)]]], ignore_index=True)
    if int(max_group_cells or 0) > 0:
        selected = selected.head(int(max_group_cells)).copy()
    return selected


def run_debug(payload: dict[str, Any], source: str, output_dir: Path) -> None:
    fetched = _fetch_source(source, payload)
    baseline_df = fetched["baseline_df"]
    antenna_df = fetched["antenna_df"]
    grid_df = fetched["grid_df"]
    thresholds = {
        "rsrp": float(payload.get("rsrp", -90)),
        "rsrq": float(payload.get("rsrq", -14)),
        "sinr": float(payload.get("sinr", 0)),
    }
    weights = {
        "rsrp": float(payload.get("rsrp_weight", 20)) / 100.0,
        "rsrq": float(payload.get("rsrq_weight", 20)) / 100.0,
        "sinr": float(payload.get("sinr_weight", 60)) / 100.0,
    }
    coverage_pct = float(payload.get("bad_grid_coverage_pct", 60))
    max_group_cells = int(payload.get("max_group_cells", 0) or 0)

    baseline = _ensure_node_cell(baseline_df)
    baseline = _attach_grid_context(baseline, baseline)
    baseline = _attach_grid_context_from_analytics(baseline, grid_df)
    grid = _combined_grid_reference(baseline, grid_df, thresholds, weights)
    bad_grid_ids = set(grid.loc[grid["is_bad_combined"].fillna(False), "grid_id"].astype(str).tolist())
    ant, tunable_aliases, tunable_canonical = _build_tunable_sets(antenna_df)

    work = baseline.copy()
    work["grid_id"] = normalize_grid_id_series(work["grid_id"])
    work["Raw Cell ID"] = work["Node_Cell_ID"].astype(str).str.strip()
    work["Canonical Cell ID"] = work["Raw Cell ID"].map(canonical_cell_id)
    work["Site ID"] = work["Canonical Cell ID"].map(_site_from_canonical)
    work["DB Sector Key"] = _db_sector_key(work)
    work["Baseline Source Key"] = _baseline_source_key(work)
    bad_work = work.loc[work["grid_id"].astype(str).isin(bad_grid_ids)].copy()
    bad_work, bad_cols, severity_cols = _add_bad_columns(bad_work, thresholds, weights)

    raw_match = bad_work.loc[bad_work["Raw Cell ID"].isin(tunable_aliases)].copy()
    canonical_match = bad_work.loc[bad_work["Canonical Cell ID"].isin(tunable_canonical)].copy()

    grouped_raw = _group_work(raw_match.assign(**{"Cell ID": raw_match["Raw Cell ID"]}), "Cell ID", bad_cols, severity_cols)
    selected_raw = _select_by_coverage(grouped_raw, weights, coverage_pct, max_group_cells)
    selected_raw["Canonical Cell ID"] = selected_raw["Cell ID"].map(canonical_cell_id)
    selected_raw["Site ID"] = selected_raw["Canonical Cell ID"].map(_site_from_canonical)
    antenna_node_ids = set(ant["Node_Cell_ID"].astype(str).str.strip().tolist()) if "Node_Cell_ID" in ant.columns else set()
    selected_raw["exact_antenna_node_key"] = selected_raw["Cell ID"].astype(str).str.strip().isin(antenna_node_ids)
    selected_raw_exact = selected_raw.loc[selected_raw["exact_antenna_node_key"]].copy()

    canonical_group_input = canonical_match.assign(**{"Cell ID": canonical_match["Canonical Cell ID"]})
    grouped_canonical = _group_work(canonical_group_input, "Cell ID", bad_cols, severity_cols)
    grouped_canonical["Site ID"] = grouped_canonical["Cell ID"].map(_site_from_canonical)
    selected_canonical = _select_by_coverage(grouped_canonical, weights, coverage_pct, max_group_cells)
    selected_canonical["Site ID"] = selected_canonical["Cell ID"].map(_site_from_canonical)

    source_key_match = bad_work.loc[bad_work["Baseline Source Key"].astype(str).str.strip().ne("")].copy()
    grouped_source_key = _group_work(source_key_match.assign(**{"Cell ID": source_key_match["Baseline Source Key"]}), "Cell ID", bad_cols, severity_cols)
    selected_source_key = _select_by_coverage(grouped_source_key, weights, coverage_pct, max_group_cells)
    selected_source_key["Canonical Cell ID"] = selected_source_key["Cell ID"].map(canonical_cell_id)
    selected_source_key["Site ID"] = selected_source_key["Canonical Cell ID"].map(_site_from_canonical)

    sector_key_match = bad_work.loc[bad_work["DB Sector Key"].astype(str).str.strip().ne("")].copy()
    grouped_sector_key = _group_work(sector_key_match.assign(**{"Cell ID": sector_key_match["DB Sector Key"]}), "Cell ID", bad_cols, severity_cols)
    selected_sector_key = _select_by_coverage(grouped_sector_key, weights, coverage_pct, max_group_cells)

    alias_map = (
        raw_match[["Raw Cell ID", "Canonical Cell ID", "Site ID", "Baseline Source Key", "DB Sector Key"]]
        .drop_duplicates()
        .sort_values(["Canonical Cell ID", "Raw Cell ID"])
    )
    alias_duplicates = (
        alias_map.groupby("Canonical Cell ID")
        .agg(
            site_id=("Site ID", "first"),
            alias_count=("Raw Cell ID", "nunique"),
            aliases=("Raw Cell ID", lambda s: "; ".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
    )
    alias_duplicates = alias_duplicates.loc[alias_duplicates["alias_count"] > 1].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    _safe_write_csv(grouped_raw, output_dir / "grouped_current_raw.csv")
    _safe_write_csv(selected_raw, output_dir / "selected_current_raw.csv")
    _safe_write_csv(selected_raw_exact, output_dir / "selected_good_ml_exact_updateable.csv")
    _safe_write_csv(grouped_canonical, output_dir / "grouped_proposed_canonical.csv")
    _safe_write_csv(selected_canonical, output_dir / "selected_proposed_canonical.csv")
    _safe_write_csv(grouped_source_key, output_dir / "grouped_baseline_source_key.csv")
    _safe_write_csv(selected_source_key, output_dir / "selected_baseline_source_key.csv")
    _safe_write_csv(grouped_sector_key, output_dir / "grouped_db_sector_key.csv")
    _safe_write_csv(selected_sector_key, output_dir / "selected_db_sector_key.csv")
    _safe_write_csv(alias_duplicates, output_dir / "alias_duplicates.csv")
    (output_dir / "payload.json").write_text(json.dumps(_redact_payload(payload), indent=2), encoding="utf-8")

    print("[TARGET_DEDUPE_DEBUG] source=", source)
    print(
        "[TARGET_DEDUPE_DEBUG_INPUT] "
        f"baseline_rows={len(baseline)} grid_rows={len(grid_df)} bad_grids={len(bad_grid_ids)} "
        f"bad_raw_cell_ids={bad_work['Raw Cell ID'].nunique()} "
        f"bad_canonical_cell_ids={bad_work['Canonical Cell ID'].nunique()} "
        f"bad_baseline_source_keys={bad_work['Baseline Source Key'].replace('', pd.NA).nunique()} "
        f"bad_db_sector_keys={bad_work['DB Sector Key'].replace('', pd.NA).nunique()} "
        f"antenna_rows={len(ant)} tunable_aliases={len(tunable_aliases)} "
        f"tunable_canonical={len(tunable_canonical)}"
    )
    print(
        "[TARGET_DEDUPE_DEBUG_CURRENT_RAW] "
        f"contributors={len(grouped_raw)} selected_cells={len(selected_raw)} "
        f"selected_physical_cells={selected_raw['Canonical Cell ID'].nunique()} "
        f"selected_sites={selected_raw['Site ID'].nunique()} "
        f"cells={selected_raw['Cell ID'].astype(str).tolist()}"
    )
    print(
        "[TARGET_DEDUPE_DEBUG_GOOD_ML_EXACT_UPDATEABLE] "
        f"selected_cells={len(selected_raw_exact)} "
        f"selected_physical_cells={selected_raw_exact['Canonical Cell ID'].nunique()} "
        f"selected_sites={selected_raw_exact['Site ID'].nunique()} "
        f"contribution_pct_sum={float(selected_raw_exact.get('contribution_pct', pd.Series(dtype=float)).sum()):.2f} "
        f"max_cumulative_pct={float(selected_raw_exact.get('cumulative_pct', pd.Series(dtype=float)).max() if not selected_raw_exact.empty else 0.0):.2f} "
        f"cells={selected_raw_exact['Cell ID'].astype(str).tolist()}"
    )
    print(
        "[TARGET_DEDUPE_DEBUG_PROPOSED_CANONICAL] "
        f"contributors={len(grouped_canonical)} selected_cells={len(selected_canonical)} "
        f"selected_sites={selected_canonical['Site ID'].nunique()} "
        f"cells={selected_canonical['Cell ID'].astype(str).tolist()}"
    )
    print(
        "[TARGET_DEDUPE_DEBUG_BASELINE_SOURCE_KEY] "
        f"contributors={len(grouped_source_key)} selected_cells={len(selected_source_key)} "
        f"selected_physical_cells={selected_source_key['Canonical Cell ID'].nunique() if 'Canonical Cell ID' in selected_source_key.columns else 0} "
        f"selected_sites={selected_source_key['Site ID'].nunique() if 'Site ID' in selected_source_key.columns else 0} "
        f"cells={selected_source_key['Cell ID'].astype(str).tolist()}"
    )
    print(
        "[TARGET_DEDUPE_DEBUG_DB_SECTOR_KEY] "
        f"usable_sector_rows={len(sector_key_match)} contributors={len(grouped_sector_key)} "
        f"selected_cells={len(selected_sector_key)} "
        f"cells={selected_sector_key['Cell ID'].astype(str).tolist() if not selected_sector_key.empty else []}"
    )
    print(
        "[TARGET_DEDUPE_DEBUG_ALIAS_DUPLICATES] "
        f"duplicate_physical_cells={len(alias_duplicates)} "
        f"current_selected_duplicate_physical_cells="
        f"{int(selected_raw.groupby('Canonical Cell ID')['Cell ID'].nunique().gt(1).sum())}"
    )
    print(f"[TARGET_DEDUPE_DEBUG_OUTPUT] {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current raw target selection vs canonical dedupe target selection.")
    parser.add_argument("--source", choices=["bridge", "direct"], default="bridge")
    parser.add_argument("--project-id", type=int, dest="project_id")
    parser.add_argument("--region")
    parser.add_argument("--operator")
    parser.add_argument("--rsrp", type=float)
    parser.add_argument("--rsrq", type=float)
    parser.add_argument("--sinr", type=float)
    parser.add_argument("--rsrp-weight", type=float, dest="rsrp_weight")
    parser.add_argument("--rsrq-weight", type=float, dest="rsrq_weight")
    parser.add_argument("--sinr-weight", type=float, dest="sinr_weight")
    parser.add_argument("--bad-grid-coverage-pct", type=float, dest="bad_grid_coverage_pct")
    parser.add_argument("--max-group-cells", type=int, dest="max_group_cells")
    parser.add_argument("--validate-candidates", type=_parse_bool, dest="validate_candidates")
    parser.add_argument("--threshold-file-path")
    parser.add_argument("--bridge-base-url")
    parser.add_argument("--bridge-api-key")
    parser.add_argument("--database-url")
    parser.add_argument("--database-url-taiwan")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    payload = _build_payload(args)
    if args.bridge_base_url:
        os.environ["PYTHON_BRIDGE_BASE_URL"] = args.bridge_base_url
    if args.bridge_api_key:
        os.environ["PYTHON_BRIDGE_API_KEY"] = args.bridge_api_key

    output_dir = Path(args.output_dir) if args.output_dir else ML_ROOT / "outputs" / "debug" / f"lte_tilt_target_dedupe_{_timestamp()}"
    print(f"[TARGET_DEDUPE_DEBUG] output_dir={output_dir}")
    print(f"[TARGET_DEDUPE_DEBUG] effective_payload={json.dumps(_redact_payload(payload))}")
    run_debug(payload, args.source, output_dir)


if __name__ == "__main__":
    main()
