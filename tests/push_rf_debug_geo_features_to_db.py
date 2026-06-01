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
GEO_TABLE = "lte_prediction_geo_features"


def _clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def _load_prediction_grid(run_dir: Path) -> pd.DataFrame:
    parquet_path = run_dir / "rf_prediction_grid.parquet"
    csv_path = run_dir / "rf_prediction_grid_full.csv"
    if parquet_path.exists():
        print(f"[PUSH_GEO] loading_prediction={parquet_path}")
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        print(f"[PUSH_GEO] loading_prediction={csv_path}")
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No prediction grid found under {run_dir}")


def _load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _build_geo_payload(
    pred_df: pd.DataFrame,
    project_id: int,
    baseline_job_id: str,
    region: str,
    operator: Optional[str],
    summary: dict,
) -> pd.DataFrame:
    out = pred_df.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    if "frontend_site_sector_key" in out.columns:
        out["nodeb_id_cell_id"] = out["frontend_site_sector_key"]
    elif "Node_Cell_ID" in out.columns:
        out["nodeb_id_cell_id"] = out["Node_Cell_ID"]
    elif "original_node_cell_id" in out.columns:
        out["nodeb_id_cell_id"] = out["original_node_cell_id"]
    else:
        raise ValueError("Prediction artifact is missing frontend_site_sector_key / Node_Cell_ID")

    if "proxy_site_id" not in out.columns:
        if "_proxy_site_id" in out.columns:
            out["proxy_site_id"] = out["_proxy_site_id"]
        elif "frontend_site_sector_key" in out.columns:
            out["proxy_site_id"] = out["frontend_site_sector_key"]
        else:
            out["proxy_site_id"] = out["nodeb_id_cell_id"]

    out["project_id"] = int(project_id)
    out["baseline_job_id"] = str(baseline_job_id)
    out["region"] = str(region).lower()
    out["operator"] = (
        _clean_text_series(out["operator"])
        if "operator" in out.columns
        else pd.Series(str(operator) if operator else pd.NA, index=out.index, dtype="object")
    )
    out["operator"] = out["operator"].fillna(str(operator) if operator else pd.NA)

    out["polygon_alignment"] = str(summary.get("project_polygon_alignment") or "")
    out["building_alignment"] = str(summary.get("building_alignment") or "")
    out["geo_source"] = "rf_debug_geo_correction"
    out["created_at"] = datetime.now()
    out["updated_at"] = datetime.now()

    schema_cols = [
        "project_id",
        "baseline_job_id",
        "region",
        "operator",
        "grid_id",
        "lat",
        "lon",
        "nodeb_id_cell_id",
        "proxy_site_id",
        "clutter_class",
        "morphology_cluster",
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "los_blocker_count",
        "los_blocked_ratio",
        "max_blocker_height_m",
        "diffraction_proxy_db",
        "nlos_flag",
        "terrain_elevation_m",
        "terrain_slope_deg",
        "proxy_site_elevation_m",
        "terrain_relief_to_site_m",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "azimuth_delta_deg",
        "polygon_alignment",
        "building_alignment",
        "geo_source",
        "created_at",
        "updated_at",
    ]
    out = _ensure_cols(out, schema_cols)

    numeric_cols = [
        "lat",
        "lon",
        "morphology_cluster",
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "los_blocker_count",
        "los_blocked_ratio",
        "max_blocker_height_m",
        "diffraction_proxy_db",
        "nlos_flag",
        "terrain_elevation_m",
        "terrain_slope_deg",
        "proxy_site_elevation_m",
        "terrain_relief_to_site_m",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "azimuth_delta_deg",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["lat"] = out["lat"].round(6)
    out["lon"] = out["lon"].round(6)
    for col in ["region", "operator", "grid_id", "nodeb_id_cell_id", "proxy_site_id", "clutter_class"]:
        out[col] = _clean_text_series(out[col])

    out = out.dropna(subset=["project_id", "region", "lat", "lon", "nodeb_id_cell_id"]).copy()
    out = out[schema_cols].drop_duplicates(
        subset=["project_id", "region", "nodeb_id_cell_id", "lat", "lon"],
        keep="last",
    )
    return out


def _push_geo_features(engine, payload: pd.DataFrame, project_id: int, region: str, dry_run: bool, replace_project: bool) -> int:
    if payload.empty:
        print("[PUSH_GEO] rows=0 nothing_to_write=True")
        return 0
    if dry_run:
        print(f"[PUSH_GEO] dry_run=True rows_ready={len(payload)} replace_project={replace_project}")
        return 0

    staging_table = f"tmp_lte_geo_stage_{uuid.uuid4().hex[:8]}"
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
                {"table_name": GEO_TABLE},
            )
        }
        missing = sorted(set(payload.columns) - table_cols)
        if missing:
            raise RuntimeError(f"{GEO_TABLE} is missing columns required by payload: {missing}")

        conn.execute(
            text(
                f"""
                CREATE TEMPORARY TABLE {staging_table} AS
                SELECT {", ".join(payload.columns)}
                FROM {GEO_TABLE}
                WHERE 1 = 0
                """
            )
        )
        for start_idx in range(0, len(payload), chunk_size):
            chunk = payload.iloc[start_idx:start_idx + chunk_size].copy()
            chunk.to_sql(
                staging_table,
                con=conn,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=chunk_size,
            )

        if replace_project:
            conn.execute(
                text(f"DELETE FROM {GEO_TABLE} WHERE project_id = :project_id AND region = :region"),
                {"project_id": int(project_id), "region": str(region).lower()},
            )
        else:
            conn.execute(
                text(
                    f"""
                    DELETE tgt
                    FROM {GEO_TABLE} AS tgt
                    INNER JOIN {staging_table} AS src
                        ON tgt.project_id = src.project_id
                       AND tgt.region = src.region
                       AND tgt.nodeb_id_cell_id = src.nodeb_id_cell_id
                       AND tgt.lat = src.lat
                       AND tgt.lon = src.lon
                    """
                )
            )

        conn.execute(text(f"INSERT INTO {GEO_TABLE} ({', '.join(payload.columns)}) SELECT {', '.join(payload.columns)} FROM {staging_table}"))
    return len(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Push RF debug lab geo features into Stracer.lte_prediction_geo_features.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--project-id", type=int, default=196)
    parser.add_argument("--region", type=str, default="india")
    parser.add_argument("--baseline-job-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-project", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    db_url = os.getenv("DATABASE_URL" if args.region.lower() == "india" else "DATABASE_URL_Taiwan")
    if not db_url:
        raise RuntimeError("Database URL is missing from environment")

    run_dir = args.run_dir.resolve()
    summary = _load_summary(run_dir)
    pred_df = _load_prediction_grid(run_dir)
    operator = summary.get("operator")
    baseline_job_id = args.baseline_job_id or f"rf_debug_{args.project_id}_{run_dir.name}"

    payload = _build_geo_payload(
        pred_df=pred_df,
        project_id=args.project_id,
        baseline_job_id=baseline_job_id,
        region=args.region,
        operator=operator,
        summary=summary,
    )
    print(
        f"[PUSH_GEO] table=Stracer.{GEO_TABLE} mode={'replace' if args.replace_project else 'key_delete_insert'} "
        f"run_dir={run_dir} project_id={args.project_id} baseline_job_id={baseline_job_id} rows={len(payload)}"
    )
    if not payload.empty:
        print(
            f"[PUSH_GEO] unique_cells={payload['nodeb_id_cell_id'].nunique()} "
            f"unique_grids={payload['grid_id'].nunique(dropna=True)} "
            f"clutter={payload['clutter_class'].value_counts(dropna=False).to_dict()}"
        )

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
    written = _push_geo_features(
        engine,
        payload=payload,
        project_id=args.project_id,
        region=args.region,
        dry_run=args.dry_run,
        replace_project=args.replace_project,
    )
    print(f"[PUSH_GEO_DONE] dry_run={args.dry_run} pushed_rows={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
