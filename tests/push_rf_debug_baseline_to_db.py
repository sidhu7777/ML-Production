from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "tests" / "output_nocache_20260528_234540" / "project_196" / "20260528_234545"
BASELINE_TABLE = "lte_prediction_baseline_results"


def _clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def _pick_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lower_lookup = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        matched = lower_lookup.get(str(candidate).lower())
        if matched is not None:
            return matched
    return None


def _coalesce_columns(df: pd.DataFrame, target: str, candidates: Iterable[str], default=None) -> pd.DataFrame:
    out = df.copy()
    if target in out.columns:
        result = out[target].astype("object")
    else:
        result = pd.Series(default, index=out.index, dtype="object")
    for candidate in candidates:
        if candidate == target:
            continue
        if candidate in out.columns:
            candidate_values = out[candidate]
            fill_mask = result.isna() & candidate_values.notna()
            result = result.copy()
            result.loc[fill_mask] = candidate_values.loc[fill_mask]
    out[target] = result
    return out


def _prefer_columns(df: pd.DataFrame, target: str, candidates: Iterable[str], default=None) -> pd.DataFrame:
    out = df.copy()
    result = pd.Series(default, index=out.index, dtype="object")
    for candidate in candidates:
        if candidate in out.columns:
            candidate_values = out[candidate]
            take_mask = candidate_values.notna()
            result = result.copy()
            result.loc[take_mask] = candidate_values.loc[take_mask]
            if take_mask.all():
                break
    out[target] = result
    return out


def _load_prediction_grid(run_dir: Path) -> pd.DataFrame:
    parquet_path = run_dir / "rf_prediction_grid.parquet"
    csv_path = run_dir / "rf_prediction_grid_full.csv"
    if parquet_path.exists():
        print(f"[PUSH_BASELINE] loading_prediction={parquet_path}")
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        print(f"[PUSH_BASELINE] loading_prediction={csv_path}")
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No prediction grid found under {run_dir}")


def _load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _build_baseline_payload(
    pred_df: pd.DataFrame,
    site_df: pd.DataFrame,
    project_id: int,
    job_id: str,
    operator: Optional[str],
    kpi_stage: str = "demo",
) -> pd.DataFrame:
    out = pred_df.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    if "original_node_cell_id" in out.columns and "node_cell_id" not in out.columns:
        out["node_cell_id"] = out["original_node_cell_id"]
    elif "Node_Cell_ID" in out.columns and "node_cell_id" not in out.columns:
        out["node_cell_id"] = out["Node_Cell_ID"]
    if "node_cell_id" in out.columns:
        out["node_cell_id"] = _clean_text_series(out["node_cell_id"])

    kpi_stage = str(kpi_stage or "demo").strip().lower()
    if kpi_stage == "geo":
        stage_cols = {
            "pred_rsrp": ["pred_rsrp_geo", "pred_rsrp_demo", "pred_rsrp"],
            "pred_rsrq": ["pred_rsrq_geo", "pred_rsrq_demo", "pred_rsrq"],
            "pred_sinr": ["pred_sinr_geo", "pred_sinr_demo", "pred_sinr"],
        }
    elif kpi_stage == "raw":
        stage_cols = {
            "pred_rsrp": ["pred_rsrp", "pred_rsrp_geo", "pred_rsrp_demo"],
            "pred_rsrq": ["pred_rsrq", "pred_rsrq_geo", "pred_rsrq_demo"],
            "pred_sinr": ["pred_sinr", "pred_sinr_geo", "pred_sinr_demo"],
        }
    else:
        kpi_stage = "demo"
        stage_cols = {
            "pred_rsrp": ["pred_rsrp_demo", "pred_rsrp_geo", "pred_rsrp"],
            "pred_rsrq": ["pred_rsrq_demo", "pred_rsrq_geo", "pred_rsrq"],
            "pred_sinr": ["pred_sinr_demo", "pred_sinr_geo", "pred_sinr"],
        }
    print(f"[PUSH_BASELINE] kpi_stage={kpi_stage} stage_columns={stage_cols}")
    for target, candidates in stage_cols.items():
        out = _prefer_columns(out, target, candidates)

    if not site_df.empty:
        site_meta = site_df.copy()
        site_meta = site_meta.loc[:, ~site_meta.columns.duplicated()].copy()
        if "Node_Cell_ID" in site_meta.columns and "node_cell_id" not in site_meta.columns:
            site_meta["node_cell_id"] = site_meta["Node_Cell_ID"]
        elif "cell_id" in site_meta.columns:
            site_meta["node_cell_id"] = site_meta["cell_id"]

        if "node_cell_id" in site_meta.columns:
            site_meta["node_cell_id"] = _clean_text_series(site_meta["node_cell_id"])
            site_id_col = _pick_first_present(site_meta, ["site_id", "Site ID", "site"])
            operator_col = _pick_first_present(site_meta, ["operator", "network", "cluster", "Technology"])

            rename_map = {}
            if "nodeb_id" in site_meta.columns:
                rename_map["nodeb_id"] = "site_nodeb_id"
            if site_id_col:
                rename_map[site_id_col] = "site_site_id"
            if operator_col:
                rename_map[operator_col] = "site_operator"
            if "sector" in site_meta.columns:
                rename_map["sector"] = "site_sector"

            site_meta = site_meta.rename(columns=rename_map)
            keep_cols = ["node_cell_id"] + [
                col for col in ["site_nodeb_id", "site_site_id", "site_operator", "site_sector"] if col in site_meta.columns
            ]
            site_meta = site_meta[keep_cols].drop_duplicates(subset=["node_cell_id"], keep="first")
            out = out.merge(site_meta, on="node_cell_id", how="left")

    if "node_cell_id" in out.columns:
        split_cols = out["node_cell_id"].astype(str).str.split("_", n=1, expand=True)
        if split_cols.shape[1] >= 2:
            out["derived_nodeb_id"] = _clean_text_series(split_cols[0])
            out["derived_cell_id"] = _clean_text_series(split_cols[1])
        else:
            out["derived_nodeb_id"] = pd.NA
            out["derived_cell_id"] = pd.NA
    else:
        out["derived_nodeb_id"] = pd.NA
        out["derived_cell_id"] = pd.NA

    out["project_id"] = int(project_id)
    out["job_id"] = str(job_id)
    out["created_at"] = datetime.now()

    out = _coalesce_columns(out, "node_b_id", ["node_b_id", "nodeb_id", "site_nodeb_id", "derived_nodeb_id"])
    out = _coalesce_columns(out, "cell_id", ["original_cell_id", "derived_cell_id", "cell_id"])
    out = _coalesce_columns(out, "operator", ["operator", "site_operator"], default=operator)
    out = _coalesce_columns(out, "site_id", ["site_id", "site_site_id", "node_b_id"])
    out = _coalesce_columns(out, "sector", ["sector", "site_sector", "sector_identity"])
    out = _coalesce_columns(out, "frontend_site_sector_key", ["frontend_site_sector_key", "Node_Cell_ID"])
    out = _coalesce_columns(out, "original_node_cell_id", ["original_node_cell_id", "node_cell_id"])

    for col in ["node_b_id", "cell_id", "operator", "site_id", "sector", "frontend_site_sector_key", "original_node_cell_id"]:
        out[col] = _clean_text_series(out[col])

    legacy_nodeb_cell_id = np.where(
        out["node_b_id"].notna() & out["cell_id"].notna(),
        out["node_b_id"].astype(str) + "_" + out["cell_id"].astype(str),
        out.get("node_cell_id"),
    )
    out["legacy_nodeb_id_cell_id"] = _clean_text_series(pd.Series(legacy_nodeb_cell_id, index=out.index))
    out["nodeb_id_cell_id"] = _clean_text_series(out["frontend_site_sector_key"]).fillna(out["legacy_nodeb_id_cell_id"])

    out["id"] = pd.NA
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["lat_6dp"] = out["lat"].round(6)
    out["lon_6dp"] = out["lon"].round(6)
    if "Technology" not in out.columns:
        out["Technology"] = "4G"
    out["Technology"] = _clean_text_series(out["Technology"]).fillna("4G")

    numeric_context_cols = [
        "serving_frequency_mhz",
        "best_interferer_distance_m",
        "best_interferer_azimuth_delta_deg",
        "best_interferer_proxy_phys_dbm",
        "neighbor_1_proxy_rsrp_dbm",
        "neighbor_1_distance_m",
        "neighbor_1_azimuth_delta_deg",
        "neighbor_2_proxy_rsrp_dbm",
        "neighbor_2_distance_m",
        "neighbor_2_azimuth_delta_deg",
        "interference_gap_db",
        "interference_ratio_linear",
        "interference_sum_proxy_dbm",
        "sinr_proxy_db",
        "rsrq_proxy_db",
        "same_earfcn_interferer_count",
        "dominant_interferer_count",
    ]
    text_context_cols = [
        "serving_pci",
        "serving_earfcn",
        "best_interferer_cell_id",
        "best_interferer_pci",
        "best_interferer_earfcn",
        "neighbor_1_cell_id",
        "neighbor_1_pci",
        "neighbor_1_earfcn",
        "neighbor_2_cell_id",
        "neighbor_2_pci",
        "neighbor_2_earfcn",
        "interference_selection_mode",
    ]
    for col in numeric_context_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan
    for col in text_context_cols:
        if col in out.columns:
            out[col] = _clean_text_series(out[col])
        else:
            out[col] = pd.NA

    final_cols = [
        "id",
        "project_id",
        "job_id",
        "lat",
        "lat_6dp",
        "lon",
        "lon_6dp",
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "node_b_id",
        "cell_id",
        "operator",
        "created_at",
        "site_id",
        "nodeb_id_cell_id",
        "Technology",
        "sector",
        "frontend_site_sector_key",
        "serving_pci",
        "serving_earfcn",
        "serving_frequency_mhz",
        "best_interferer_cell_id",
        "best_interferer_pci",
        "best_interferer_earfcn",
        "best_interferer_distance_m",
        "best_interferer_azimuth_delta_deg",
        "best_interferer_proxy_phys_dbm",
        "neighbor_1_cell_id",
        "neighbor_1_pci",
        "neighbor_1_earfcn",
        "neighbor_1_proxy_rsrp_dbm",
        "neighbor_1_distance_m",
        "neighbor_1_azimuth_delta_deg",
        "neighbor_2_cell_id",
        "neighbor_2_pci",
        "neighbor_2_earfcn",
        "neighbor_2_proxy_rsrp_dbm",
        "neighbor_2_distance_m",
        "neighbor_2_azimuth_delta_deg",
        "interference_gap_db",
        "interference_ratio_linear",
        "interference_sum_proxy_dbm",
        "sinr_proxy_db",
        "rsrq_proxy_db",
        "same_earfcn_interferer_count",
        "dominant_interferer_count",
        "interference_selection_mode",
    ]
    out = out[final_cols]
    out = out.dropna(subset=["project_id", "nodeb_id_cell_id", "lat_6dp", "lon_6dp"]).copy()
    out = out.drop_duplicates(
        subset=["project_id", "nodeb_id_cell_id", "lat_6dp", "lon_6dp"],
        keep="last",
    )
    return out


def _upsert_baseline(engine, payload: pd.DataFrame, dry_run: bool) -> int:
    if payload.empty:
        print("[PUSH_BASELINE] rows=0 nothing_to_write=True")
        return 0
    if dry_run:
        print(f"[PUSH_BASELINE] dry_run=True rows_ready={len(payload)}")
        return 0

    staging_table = f"tmp_lte_baseline_stage_{uuid.uuid4().hex[:8]}"
    chunk_size = 5000
    with engine.begin() as conn:
        table_cols = {
            row[0]
            for row in conn.execute(
                text(
                    """
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = :table_name
                    """
                ),
                {"table_name": BASELINE_TABLE},
            )
        }
        required_cols = {"sector", "frontend_site_sector_key"}
        missing_cols = sorted(required_cols - table_cols)
        if missing_cols:
            raise RuntimeError(
                f"{BASELINE_TABLE} is missing required columns {missing_cols}. "
                "Run: ALTER TABLE lte_prediction_baseline_results "
                "ADD COLUMN sector VARCHAR(100) NULL, "
                "ADD COLUMN frontend_site_sector_key VARCHAR(255) NULL;"
            )

        next_id = int(conn.execute(text(f"SELECT COALESCE(MAX(id), 0) FROM {BASELINE_TABLE}")).scalar() or 0) + 1
        conn.execute(text(f"CREATE TEMPORARY TABLE {staging_table} LIKE {BASELINE_TABLE}"))
        for start_idx in range(0, len(payload), chunk_size):
            chunk = payload.iloc[start_idx:start_idx + chunk_size].copy()
            chunk["id"] = np.arange(next_id, next_id + len(chunk), dtype=np.int64)
            next_id += len(chunk)
            chunk.to_sql(
                staging_table,
                con=conn,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=chunk_size,
            )

        conn.execute(
            text(
                f"""
                INSERT INTO {BASELINE_TABLE} (
                    id, project_id, job_id, lat, lat_6dp, lon, lon_6dp,
                    pred_rsrp, pred_rsrq, pred_sinr, node_b_id, cell_id,
                    operator, created_at, site_id, nodeb_id_cell_id, Technology,
                    sector, frontend_site_sector_key,
                    serving_pci, serving_earfcn, serving_frequency_mhz,
                    best_interferer_cell_id, best_interferer_pci, best_interferer_earfcn,
                    best_interferer_distance_m, best_interferer_azimuth_delta_deg, best_interferer_proxy_phys_dbm,
                    neighbor_1_cell_id, neighbor_1_pci, neighbor_1_earfcn,
                    neighbor_1_proxy_rsrp_dbm, neighbor_1_distance_m, neighbor_1_azimuth_delta_deg,
                    neighbor_2_cell_id, neighbor_2_pci, neighbor_2_earfcn,
                    neighbor_2_proxy_rsrp_dbm, neighbor_2_distance_m, neighbor_2_azimuth_delta_deg,
                    interference_gap_db, interference_ratio_linear, interference_sum_proxy_dbm,
                    sinr_proxy_db, rsrq_proxy_db, same_earfcn_interferer_count,
                    dominant_interferer_count, interference_selection_mode
                )
                SELECT
                    id, project_id, job_id, lat, lat_6dp, lon, lon_6dp,
                    pred_rsrp, pred_rsrq, pred_sinr, node_b_id, cell_id,
                    operator, created_at, site_id, nodeb_id_cell_id, Technology,
                    sector, frontend_site_sector_key,
                    serving_pci, serving_earfcn, serving_frequency_mhz,
                    best_interferer_cell_id, best_interferer_pci, best_interferer_earfcn,
                    best_interferer_distance_m, best_interferer_azimuth_delta_deg, best_interferer_proxy_phys_dbm,
                    neighbor_1_cell_id, neighbor_1_pci, neighbor_1_earfcn,
                    neighbor_1_proxy_rsrp_dbm, neighbor_1_distance_m, neighbor_1_azimuth_delta_deg,
                    neighbor_2_cell_id, neighbor_2_pci, neighbor_2_earfcn,
                    neighbor_2_proxy_rsrp_dbm, neighbor_2_distance_m, neighbor_2_azimuth_delta_deg,
                    interference_gap_db, interference_ratio_linear, interference_sum_proxy_dbm,
                    sinr_proxy_db, rsrq_proxy_db, same_earfcn_interferer_count,
                    dominant_interferer_count, interference_selection_mode
                FROM {staging_table}
                ON DUPLICATE KEY UPDATE
                    job_id = VALUES(job_id),
                    pred_rsrp = VALUES(pred_rsrp),
                    pred_rsrq = VALUES(pred_rsrq),
                    pred_sinr = VALUES(pred_sinr),
                    node_b_id = VALUES(node_b_id),
                    cell_id = VALUES(cell_id),
                    operator = VALUES(operator),
                    created_at = VALUES(created_at),
                    site_id = VALUES(site_id),
                    nodeb_id_cell_id = VALUES(nodeb_id_cell_id),
                    Technology = VALUES(Technology),
                    sector = VALUES(sector),
                    frontend_site_sector_key = VALUES(frontend_site_sector_key),
                    lat = VALUES(lat),
                    lon = VALUES(lon),
                    lat_6dp = VALUES(lat_6dp),
                    lon_6dp = VALUES(lon_6dp),
                    serving_pci = VALUES(serving_pci),
                    serving_earfcn = VALUES(serving_earfcn),
                    serving_frequency_mhz = VALUES(serving_frequency_mhz),
                    best_interferer_cell_id = VALUES(best_interferer_cell_id),
                    best_interferer_pci = VALUES(best_interferer_pci),
                    best_interferer_earfcn = VALUES(best_interferer_earfcn),
                    best_interferer_distance_m = VALUES(best_interferer_distance_m),
                    best_interferer_azimuth_delta_deg = VALUES(best_interferer_azimuth_delta_deg),
                    best_interferer_proxy_phys_dbm = VALUES(best_interferer_proxy_phys_dbm),
                    neighbor_1_cell_id = VALUES(neighbor_1_cell_id),
                    neighbor_1_pci = VALUES(neighbor_1_pci),
                    neighbor_1_earfcn = VALUES(neighbor_1_earfcn),
                    neighbor_1_proxy_rsrp_dbm = VALUES(neighbor_1_proxy_rsrp_dbm),
                    neighbor_1_distance_m = VALUES(neighbor_1_distance_m),
                    neighbor_1_azimuth_delta_deg = VALUES(neighbor_1_azimuth_delta_deg),
                    neighbor_2_cell_id = VALUES(neighbor_2_cell_id),
                    neighbor_2_pci = VALUES(neighbor_2_pci),
                    neighbor_2_earfcn = VALUES(neighbor_2_earfcn),
                    neighbor_2_proxy_rsrp_dbm = VALUES(neighbor_2_proxy_rsrp_dbm),
                    neighbor_2_distance_m = VALUES(neighbor_2_distance_m),
                    neighbor_2_azimuth_delta_deg = VALUES(neighbor_2_azimuth_delta_deg),
                    interference_gap_db = VALUES(interference_gap_db),
                    interference_ratio_linear = VALUES(interference_ratio_linear),
                    interference_sum_proxy_dbm = VALUES(interference_sum_proxy_dbm),
                    sinr_proxy_db = VALUES(sinr_proxy_db),
                    rsrq_proxy_db = VALUES(rsrq_proxy_db),
                    same_earfcn_interferer_count = VALUES(same_earfcn_interferer_count),
                    dominant_interferer_count = VALUES(dominant_interferer_count),
                    interference_selection_mode = VALUES(interference_selection_mode)
                """
            )
        )
    return len(payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Push one RF debug lab prediction run into Stracer.lte_prediction_baseline_results with upsert semantics."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--project-id", type=int, default=196)
    parser.add_argument("--region", type=str, default="india")
    parser.add_argument("--job-id", type=str, default=None)
    parser.add_argument("--kpi-stage", choices=["demo", "geo", "raw"], default="demo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    db_url = os.getenv("DATABASE_URL" if args.region.lower() == "india" else "DATABASE_URL_Taiwan")
    if not db_url:
        raise RuntimeError("Database URL is missing from environment")

    run_dir = args.run_dir.resolve()
    summary = _load_summary(run_dir)
    pred_df = _load_prediction_grid(run_dir)
    site_path = run_dir / "site_df.csv"
    site_df = pd.read_csv(site_path) if site_path.exists() else pd.DataFrame()
    operator = summary.get("operator")
    job_id = args.job_id or f"rf_debug_{args.project_id}_{run_dir.name}"

    payload = _build_baseline_payload(
        pred_df=pred_df,
        site_df=site_df,
        project_id=args.project_id,
        job_id=job_id,
        operator=operator,
        kpi_stage=args.kpi_stage,
    )
    print(
        f"[PUSH_BASELINE] table=Stracer.{BASELINE_TABLE} mode=upsert "
        f"run_dir={run_dir} project_id={args.project_id} job_id={job_id} rows={len(payload)}"
    )
    if not payload.empty:
        print(
            f"[PUSH_BASELINE] unique_cells={payload['nodeb_id_cell_id'].nunique()} "
            f"lat_range={payload['lat'].min():.6f}..{payload['lat'].max():.6f} "
            f"lon_range={payload['lon'].min():.6f}..{payload['lon'].max():.6f}"
        )

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
    written = _upsert_baseline(engine, payload, dry_run=args.dry_run)
    print(f"[PUSH_BASELINE_DONE] dry_run={args.dry_run} upsert_rows={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
