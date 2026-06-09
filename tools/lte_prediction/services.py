import uuid
import threading
import pandas as pd
import os
import numpy as np
from datetime import datetime
import time
import uuid as uuid_lib

from sqlalchemy import create_engine, text, Table, MetaData
from dotenv import load_dotenv

from .ml_engine import (
    run_rf_prediction_fast,
    run_ml_fast,
    fetch_site_data,
    fetch_drive_data,
    fetch_building_data
)
from .dem_utils import ensure_project_dem

from extensions import db

load_dotenv()
engine_dict = {
    "taiwan": create_engine(
        os.getenv("DATABASE_URL_Taiwan"),
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL_Taiwan") else None
}

JOBS = {}
BASELINE_SMOOTHED_COLUMNS = {
    "pred_rsrp_smoothed": "DOUBLE NULL",
    "pred_rsrq_smoothed": "DOUBLE NULL",
    "pred_sinr_smoothed": "DOUBLE NULL",
}


def _df_records_with_none(df: pd.DataFrame):
    safe_df = df.copy()
    safe_df = safe_df.replace({pd.NA: None})
    safe_df = safe_df.where(pd.notna(safe_df), None)
    return safe_df.to_dict(orient="records")


def _series_for_compare(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(["__MISSING__"] * len(df), index=df.index, dtype="object")
    series = df[col]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").round(6).astype(str)
    series = series.where(pd.notna(series), pd.NA)
    series = _clean_text_series(series)
    return series.fillna("").astype(str)


def _numeric_changed(new_series: pd.Series, old_series: pd.Series, tolerance: float) -> pd.Series:
    new_num = pd.to_numeric(new_series, errors="coerce")
    old_num = pd.to_numeric(old_series, errors="coerce")
    both_nan = new_num.isna() & old_num.isna()
    one_nan = new_num.isna() ^ old_num.isna()
    diff = (new_num - old_num).abs()
    changed = one_nan | (diff > tolerance)
    changed = changed & (~both_nan)
    return changed.fillna(False)


def _metric_range(df, col):
    if col not in df.columns:
        return "n/a"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return "n/a"
    return f"{series.min():.4f}..{series.max():.4f}"


def _job_df_summary(stage, df):
    print(f"[LTE][{stage}] shape={df.shape}")
    print(f"[LTE][{stage}] columns={list(df.columns)}")
    for col in ["cell_id", "nodeb_id", "Node_Cell_ID", "node_b_id", "operator", "site_id"]:
        if col in df.columns:
            print(f"[LTE][{stage}] distinct_{col}={int(df[col].nunique(dropna=True))}")
    for col in [
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "pred_rsrp_geo",
        "pred_rsrq_geo",
        "pred_sinr_geo",
        "pred_rsrp_demo",
        "pred_rsrq_demo",
        "pred_sinr_demo",
    ]:
        if col in df.columns:
            print(f"[LTE][{stage}] range_{col}={_metric_range(df, col)}")


def _clean_text_series(series):
    cleaned = series.astype(str).str.strip()
    return cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "none": pd.NA})


def _pick_first_present(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _coalesce_columns(df, target, candidates, default=None):
    out = pd.Series([default] * len(df), index=df.index, dtype="object")
    for col in candidates:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            series = _clean_text_series(series)
        out = out.where(out.notna(), series)
    df[target] = out
    return df


def _prefer_columns(df, target, candidates, default=None):
    result = pd.Series([default] * len(df), index=df.index, dtype="object")
    for col in candidates:
        if col not in df.columns:
            continue
        values = df[col]
        take_mask = result.isna() & values.notna()
        result.loc[take_mask] = values.loc[take_mask]
    df[target] = result
    return df


class LTEPredictionService:

    def submit(self, app, cfg):
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {"status": "queued"}

        threading.Thread(
            target=self._run_with_app_context,
            args=(app, job_id, cfg),
            daemon=True
        ).start()

        return {"job_id": job_id}

    def get(self, job_id):
        return JOBS.get(job_id)

    def _run_with_app_context(self, app, job_id, cfg):
        with app.app_context():
            self._run(job_id, cfg)

    def _run(self, job_id, cfg):
        try:
            region = str(cfg.get("region", "india")).lower()
            print(
                f"[LTE][JOB_START] job_id={job_id} project_id={cfg['project_id']} "
                f"region={region} session_ids={cfg['session_ids']} radius={cfg['radius_m']} "
                f"grid_resolution={cfg['grid_resolution']} n_workers={cfg['n_workers']} "
                f"max_interference_sites={cfg.get('max_interference_sites', 50)}"
            )

            self._update(job_id, "running", f"Fetching site data from {region.upper()} database")

            site_df, operator = fetch_site_data(cfg["project_id"], region=region)
            _job_df_summary("SITE_DF", site_df)

            self._update(job_id, "running", f"Operator: {operator}")

            dem_raster_path = self._resolve_dem_path(
                project_id=cfg["project_id"],
                region=region,
                site_df=site_df,
                requested_path=cfg.get("dem_raster_path"),
            )

            self._update(job_id, "running", "Fetching drive data")
            drive_df = fetch_drive_data(
                cfg["session_ids"], operator, cfg["project_id"], region=region
            )
            _job_df_summary("DRIVE_DF", drive_df)

            self._update(job_id, "running", "Fetching building data")
            building_df = fetch_building_data(cfg["project_id"], region=region)
            _job_df_summary("BUILDING_DF", building_df)

            self._update(job_id, "running", "RF Prediction")
            pred_df = run_rf_prediction_fast(
                site_df,
                drive_df,
                building_df,
                {
                    "project_id": cfg["project_id"],
                    "region": region,
                    "radius": cfg["radius_m"],
                    "grid": cfg["grid_resolution"],
                    "workers": cfg["n_workers"],
                    "max_interference_sites": cfg.get("max_interference_sites", 50),
                    "use_frontend_grid_sampling": cfg.get("use_frontend_grid_sampling", True),
                    "samples_per_grid_axis": cfg.get("samples_per_grid_axis", 3),
                    "max_cells_per_grid": cfg.get("max_cells_per_grid", 3),
                    "min_cells_per_grid": cfg.get("min_cells_per_grid", 1),
                    "ensure_all_cells": cfg.get("ensure_all_cells", True),
                    "min_grids_per_cell": cfg.get("min_grids_per_cell", 1),
                    "grid_analytics_scenario_id": cfg.get("grid_analytics_scenario_id"),
                }
            )

            _job_df_summary("RF_PRED_DF", pred_df)
            self._update(job_id, "running", "Geo correction and smoothing")
            final_df = run_ml_fast(
                pred_df,
                drive_df,
                site_df=site_df,
                building_df=building_df,
                params={
                    "project_id": cfg["project_id"],
                    "region": region,
                    "grid": cfg["grid_resolution"],
                    "tile_size_m": cfg.get("tile_size_m", 100),
                    "cluster_count": cfg.get("cluster_count", 5),
                    "dem_raster_path": dem_raster_path,
                    "optimizer_weights_path": cfg.get("optimizer_weights_path"),
                    "dt_replace_radius_m": cfg.get("dt_replace_radius_m", 20),
                    "dt_blend_sigma_m": cfg.get("dt_blend_sigma_m", 60),
                    "dt_blend_radius_m": cfg.get("dt_blend_radius_m", 140),
                },
            )
            _job_df_summary("DISPLAY_OUTPUT_DF", final_df)
            production_summary = dict(final_df.attrs.get("production_summary") or {})
            if production_summary:
                geo_metrics = production_summary.get("geo_validation_metrics")
                weights_summary = production_summary.get("weights_summary")
                if weights_summary:
                    print(f"[LTE][GEO_WEIGHTS] {weights_summary}")
                if geo_metrics:
                    print(f"[LTE][GEO_VALIDATION] {geo_metrics}")
                JOBS[job_id]["metrics"] = {
                    "baseline": production_summary.get("baseline_validation_metrics"),
                    "geo": geo_metrics,
                }
                JOBS[job_id]["weights"] = weights_summary

            self._update(job_id, "running", "Saving results to database")
            self._save_baseline_results(
                final_df,
                cfg["project_id"],
                job_id,
                site_df=site_df,
                operator=operator,
                region=region
            )

            output = f"temp/final_{job_id}.csv"
            os.makedirs("temp", exist_ok=True)
            final_df.to_csv(output, index=False)

            JOBS[job_id]["output"] = output
            JOBS[job_id]["rows"] = len(final_df)

            self._update(job_id, "done", "Completed")

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            print(f"Error in Job {job_id}: {str(e)}")

    def _update(self, job_id, status, msg):
        JOBS[job_id]["status"] = status
        JOBS[job_id]["progress"] = msg
        print(f"[{job_id[:6]}] {msg}")

    def _resolve_dem_path(self, project_id, region, site_df, requested_path=None):
        try:
            resolved_path = ensure_project_dem(
                project_id=int(project_id),
                region=str(region).lower(),
                site_df=site_df,
                output_path=requested_path,
                timeout_sec=60,
                force=False,
            )
            print(f"[LTE][DEM] auto_resolved=True path={resolved_path}")
            return str(resolved_path)
        except Exception as exc:
            print(f"[LTE][DEM] auto_resolved=False reason={exc}")
            return requested_path

    def _ensure_baseline_smoothed_columns(self, conn):
        table_name = "lte_prediction_baseline_results"
        existing_cols = {
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
                {"table_name": table_name},
            )
        }
        for col, sql_type in BASELINE_SMOOTHED_COLUMNS.items():
            if col in existing_cols:
                continue
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col} {sql_type}"))
            print(f"[LTE][BASELINE_SCHEMA] added_column={col} table={table_name}")

    def _compute_baseline_delta(self, conn, out: pd.DataFrame, project_id: int) -> pd.DataFrame:
        self._ensure_baseline_smoothed_columns(conn)
        existing = pd.read_sql(
            text(
                """
                SELECT
                    project_id,
                    nodeb_id_cell_id,
                    lat_6dp,
                    lon_6dp,
                    pred_rsrp,
                    pred_rsrq,
                    pred_sinr,
                    pred_rsrp_smoothed,
                    pred_rsrq_smoothed,
                    pred_sinr_smoothed,
                    node_b_id,
                    cell_id,
                    operator,
                    site_id,
                    Technology
                FROM lte_prediction_baseline_results
                WHERE project_id = :project_id
                """
            ),
            conn,
            params={"project_id": int(project_id)},
        )
        if existing.empty:
            print(f"[LTE][BASELINE_DELTA] project_id={project_id} existing_rows=0 delta_rows={len(out)} unchanged_rows=0")
            return out

        key_cols = ["project_id", "nodeb_id_cell_id", "lat_6dp", "lon_6dp"]
        compare_cols = [
            "pred_rsrp",
            "pred_rsrq",
            "pred_sinr",
            "pred_rsrp_smoothed",
            "pred_rsrq_smoothed",
            "pred_sinr_smoothed",
            "node_b_id",
            "cell_id",
            "operator",
            "site_id",
            "Technology",
        ]
        numeric_tolerances = {
            "pred_rsrp": 0.05,
            "pred_rsrq": 0.05,
            "pred_sinr": 0.05,
            "pred_rsrp_smoothed": 0.05,
            "pred_rsrq_smoothed": 0.05,
            "pred_sinr_smoothed": 0.05,
        }

        existing = existing.rename(columns={col: f"{col}__old" for col in compare_cols})
        merged = out.merge(existing, on=key_cols, how="left", indicator=True)
        is_new = merged["_merge"] == "left_only"
        is_changed = pd.Series(False, index=merged.index)
        changed_counts = {}
        for col in compare_cols:
            if col in numeric_tolerances:
                col_changed = _numeric_changed(
                    merged.get(col, pd.Series(index=merged.index, dtype=float)),
                    merged.get(f"{col}__old", pd.Series(index=merged.index, dtype=float)),
                    tolerance=float(numeric_tolerances[col]),
                )
            else:
                new_series = _series_for_compare(merged, col)
                old_series = _series_for_compare(merged, f"{col}__old")
                col_changed = new_series != old_series
            is_changed = is_changed | col_changed
        delta_mask = is_new | is_changed
        delta = merged.loc[delta_mask, out.columns].copy()
        unchanged = int((~delta_mask).sum())
        print(
            f"[LTE][BASELINE_DELTA] project_id={project_id} existing_rows={len(existing)} "
            f"delta_rows={len(delta)} unchanged_rows={unchanged}"
        )
        if unchanged:
            print(
                f"[LTE][BASELINE_DELTA] write_full_payload=True reason=latest_job_id_requires_complete_snapshot "
                f"rows={len(out)}"
            )
        return out

    def _upsert_baseline_results(self, save_engine, out: pd.DataFrame, project_id: int):
        table_name = "lte_prediction_baseline_results"
        metadata = MetaData()

        started_at = time.perf_counter()
        with save_engine.begin() as conn:
            out = self._compute_baseline_delta(conn, out, project_id=project_id)
            if out.empty:
                elapsed = time.perf_counter() - started_at
                print(
                    f"[LTE][BASELINE_DB_TIMING] table={table_name} rows=0 "
                    f"elapsed_sec={elapsed:.2f}"
                )
                return 0

            table = Table(table_name, metadata, autoload_with=conn)
            next_id = int(conn.execute(text(f"SELECT COALESCE(MAX(id), 0) FROM {table_name}")).scalar() or 0) + 1
            staging_table = f"tmp_lte_baseline_stage_{uuid_lib.uuid4().hex[:8]}"
            conn.execute(text(f"CREATE TEMPORARY TABLE {staging_table} LIKE {table_name}"))

            chunk_size = 5000
            total_rows = len(out)
            for start_idx in range(0, total_rows, chunk_size):
                chunk = out.iloc[start_idx:start_idx + chunk_size].copy()
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

            merge_sql = text(
                f"""
                INSERT INTO {table_name} (
                    id, project_id, job_id, lat, lat_6dp, lon, lon_6dp,
                    pred_rsrp, pred_rsrq, pred_sinr,
                    pred_rsrp_smoothed, pred_rsrq_smoothed, pred_sinr_smoothed,
                    node_b_id, cell_id,
                    operator, created_at, site_id, nodeb_id_cell_id, Technology
                )
                SELECT
                    id, project_id, job_id, lat, lat_6dp, lon, lon_6dp,
                    pred_rsrp, pred_rsrq, pred_sinr,
                    pred_rsrp_smoothed, pred_rsrq_smoothed, pred_sinr_smoothed,
                    node_b_id, cell_id,
                    operator, created_at, site_id, nodeb_id_cell_id, Technology
                FROM {staging_table}
                ON DUPLICATE KEY UPDATE
                    job_id = VALUES(job_id),
                    pred_rsrp = VALUES(pred_rsrp),
                    pred_rsrq = VALUES(pred_rsrq),
                    pred_sinr = VALUES(pred_sinr),
                    pred_rsrp_smoothed = VALUES(pred_rsrp_smoothed),
                    pred_rsrq_smoothed = VALUES(pred_rsrq_smoothed),
                    pred_sinr_smoothed = VALUES(pred_sinr_smoothed),
                    node_b_id = VALUES(node_b_id),
                    cell_id = VALUES(cell_id),
                    operator = VALUES(operator),
                    created_at = VALUES(created_at),
                    site_id = VALUES(site_id),
                    nodeb_id_cell_id = VALUES(nodeb_id_cell_id),
                    Technology = VALUES(Technology),
                    lat = VALUES(lat),
                    lon = VALUES(lon),
                    lat_6dp = VALUES(lat_6dp),
                    lon_6dp = VALUES(lon_6dp)
                """
            )
            conn.execute(merge_sql)
        elapsed = time.perf_counter() - started_at
        print(
            f"[LTE][BASELINE_DB_TIMING] table={table_name} rows={len(out)} "
            f"elapsed_sec={elapsed:.2f}"
        )
        return len(out)

    def _compute_geo_delta(
        self,
        conn,
        out: pd.DataFrame,
        project_id: int,
        region: str,
    ):
        existing = pd.read_sql(
            text(
                """
                SELECT
                    project_id,
                    region,
                    lat,
                    lon,
                    nodeb_id_cell_id,
                    operator,
                    grid_id,
                    proxy_site_id,
                    clutter_class,
                    morphology_cluster,
                    building_count,
                    building_area_ratio,
                    avg_building_area_m2,
                    road_length_m,
                    green_ratio,
                    water_ratio,
                    los_blocker_count,
                    los_blocked_ratio,
                    max_blocker_height_m,
                    diffraction_proxy_db,
                    nlos_flag,
                    terrain_elevation_m,
                    terrain_slope_deg,
                    proxy_site_elevation_m,
                    terrain_relief_to_site_m,
                    site_count_250m,
                    site_count_500m,
                    serving_distance_m,
                    nearest_site_distance_m,
                    mean_nearest3_site_distance_m,
                    azimuth_delta_deg,
                    polygon_alignment,
                    building_alignment,
                    geo_source
                FROM lte_prediction_geo_features
                WHERE project_id = :project_id
                  AND region = :region
                """
            ),
            conn,
            params={"project_id": int(project_id), "region": str(region).lower()},
        )

        key_cols = ["project_id", "region", "nodeb_id_cell_id", "lat", "lon"]
        compare_cols = [
            "operator",
            "grid_id",
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
        ]
        numeric_tolerances = {
            "morphology_cluster": 0.0,
            "building_count": 0.0,
            "building_area_ratio": 0.001,
            "avg_building_area_m2": 0.01,
            "road_length_m": 0.01,
            "green_ratio": 0.001,
            "water_ratio": 0.001,
            "los_blocker_count": 0.0,
            "los_blocked_ratio": 0.001,
            "max_blocker_height_m": 0.01,
            "diffraction_proxy_db": 0.01,
            "nlos_flag": 0.0,
            "terrain_elevation_m": 0.01,
            "terrain_slope_deg": 0.01,
            "proxy_site_elevation_m": 0.01,
            "terrain_relief_to_site_m": 0.01,
            "site_count_250m": 0.0,
            "site_count_500m": 0.0,
            "serving_distance_m": 0.01,
            "nearest_site_distance_m": 0.01,
            "mean_nearest3_site_distance_m": 0.01,
            "azimuth_delta_deg": 0.01,
        }

        if existing.empty:
            stale_keys = pd.DataFrame(columns=key_cols)
            print(
                f"[LTE][GEO_DELTA] project_id={project_id} region={region} "
                f"existing_rows=0 delta_rows={len(out)} unchanged_rows=0 stale_rows=0"
            )
            return out, stale_keys

        existing = existing.drop_duplicates(subset=key_cols, keep="last")
        out_keys = out[key_cols].drop_duplicates().copy()
        stale_keys = (
            existing[key_cols]
            .merge(out_keys, on=key_cols, how="left", indicator=True)
            .loc[lambda d: d["_merge"] == "left_only", key_cols]
            .copy()
        )

        existing = existing.rename(columns={col: f"{col}__old" for col in compare_cols})
        merged = out.merge(existing, on=key_cols, how="left", indicator=True)
        is_new = merged["_merge"] == "left_only"
        is_changed = pd.Series(False, index=merged.index)
        changed_counts = {}
        for col in compare_cols:
            if col in numeric_tolerances:
                col_changed = _numeric_changed(
                    merged.get(col, pd.Series(index=merged.index, dtype=float)),
                    merged.get(f"{col}__old", pd.Series(index=merged.index, dtype=float)),
                    tolerance=float(numeric_tolerances[col]),
                )
            else:
                new_series = _series_for_compare(merged, col)
                old_series = _series_for_compare(merged, f"{col}__old")
                col_changed = new_series != old_series
            changed_counts[col] = int(col_changed.sum())
            is_changed = is_changed | col_changed

        delta_mask = is_new | is_changed
        delta = merged.loc[delta_mask, out.columns].copy()
        unchanged = int((~delta_mask).sum())
        print(
            f"[LTE][GEO_DELTA] project_id={project_id} region={region} "
            f"existing_rows={len(existing)} delta_rows={len(delta)} "
            f"unchanged_rows={unchanged} stale_rows={len(stale_keys)}"
        )
        print(f"[LTE][GEO_DELTA_COLS] {changed_counts}")
        return delta, stale_keys

    def _upsert_geo_features(
        self,
        save_engine,
        out: pd.DataFrame,
        project_id: int,
        region: str,
    ):
        table_name = "lte_prediction_geo_features"
        metadata = MetaData()

        started_at = time.perf_counter()
        with save_engine.begin() as conn:
            delta, stale_keys = self._compute_geo_delta(
                conn,
                out,
                project_id=project_id,
                region=region,
            )
            if delta.empty and stale_keys.empty:
                elapsed = time.perf_counter() - started_at
                print(
                    f"[LTE][GEO_DB_TIMING] table={table_name} rows=0 "
                    f"elapsed_sec={elapsed:.2f}"
                )
                return 0

            table = Table(table_name, metadata, autoload_with=conn)
            staging_table = f"tmp_lte_geo_stage_{uuid_lib.uuid4().hex[:8]}"
            staging_key_table = f"tmp_lte_geo_stale_{uuid_lib.uuid4().hex[:8]}"
            staging_cols = [col.name for col in table.columns if col.name != "id"]
            conn.execute(
                text(
                    f"""
                    CREATE TEMPORARY TABLE {staging_table} AS
                    SELECT {", ".join(staging_cols)}
                    FROM {table_name}
                    WHERE 1 = 0
                    """
                )
            )

            chunk_size = 5000
            if not delta.empty:
                for start_idx in range(0, len(delta), chunk_size):
                    chunk = delta.iloc[start_idx:start_idx + chunk_size].copy()[staging_cols]
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
                        DELETE tgt
                        FROM {table_name} AS tgt
                        INNER JOIN {staging_table} AS src
                            ON tgt.project_id = src.project_id
                           AND tgt.region = src.region
                           AND tgt.nodeb_id_cell_id = src.nodeb_id_cell_id
                           AND tgt.lat = src.lat
                           AND tgt.lon = src.lon
                        """
                    )
                )
                conn.execute(
                    text(
                        f"""
                        INSERT INTO {table_name} (
                            project_id,
                            baseline_job_id,
                            region,
                            operator,
                            grid_id,
                            lat,
                            lon,
                            nodeb_id_cell_id,
                            proxy_site_id,
                            clutter_class,
                            morphology_cluster,
                            building_count,
                            building_area_ratio,
                            avg_building_area_m2,
                            road_length_m,
                            green_ratio,
                            water_ratio,
                            los_blocker_count,
                            los_blocked_ratio,
                            max_blocker_height_m,
                            diffraction_proxy_db,
                            nlos_flag,
                            terrain_elevation_m,
                            terrain_slope_deg,
                            proxy_site_elevation_m,
                            terrain_relief_to_site_m,
                            site_count_250m,
                            site_count_500m,
                            serving_distance_m,
                            nearest_site_distance_m,
                            mean_nearest3_site_distance_m,
                            azimuth_delta_deg,
                            polygon_alignment,
                            building_alignment,
                            geo_source,
                            created_at,
                            updated_at
                        )
                        SELECT
                            project_id,
                            baseline_job_id,
                            region,
                            operator,
                            grid_id,
                            lat,
                            lon,
                            nodeb_id_cell_id,
                            proxy_site_id,
                            clutter_class,
                            morphology_cluster,
                            building_count,
                            building_area_ratio,
                            avg_building_area_m2,
                            road_length_m,
                            green_ratio,
                            water_ratio,
                            los_blocker_count,
                            los_blocked_ratio,
                            max_blocker_height_m,
                            diffraction_proxy_db,
                            nlos_flag,
                            terrain_elevation_m,
                            terrain_slope_deg,
                            proxy_site_elevation_m,
                            terrain_relief_to_site_m,
                            site_count_250m,
                            site_count_500m,
                            serving_distance_m,
                            nearest_site_distance_m,
                            mean_nearest3_site_distance_m,
                            azimuth_delta_deg,
                            polygon_alignment,
                            building_alignment,
                            geo_source,
                            created_at,
                            updated_at
                        FROM {staging_table}
                        """
                    )
                )

            if not stale_keys.empty:
                conn.execute(
                    text(
                        f"""
                        CREATE TEMPORARY TABLE {staging_key_table} (
                            project_id BIGINT NOT NULL,
                            region VARCHAR(50) NOT NULL,
                            nodeb_id_cell_id VARCHAR(100) NOT NULL,
                            lat DOUBLE NOT NULL,
                            lon DOUBLE NOT NULL
                        )
                        """
                    )
                )
                for start_idx in range(0, len(stale_keys), chunk_size):
                    chunk = stale_keys.iloc[start_idx:start_idx + chunk_size].copy()
                    chunk.to_sql(
                        staging_key_table,
                        con=conn,
                        if_exists="append",
                        index=False,
                        method="multi",
                        chunksize=chunk_size,
                    )
                conn.execute(
                    text(
                        f"""
                        DELETE tgt
                        FROM {table_name} AS tgt
                        INNER JOIN {staging_key_table} AS stale
                            ON tgt.project_id = stale.project_id
                           AND tgt.region = stale.region
                           AND tgt.nodeb_id_cell_id = stale.nodeb_id_cell_id
                           AND tgt.lat = stale.lat
                           AND tgt.lon = stale.lon
                        """
                    )
                )

        elapsed = time.perf_counter() - started_at
        print(
            f"[LTE][GEO_DB_TIMING] table={table_name} rows={len(delta)} "
            f"elapsed_sec={elapsed:.2f}"
        )
        return len(delta)

    def _save_baseline_results(self, df, project_id, job_id, site_df=None, operator=None, region="india"):
        print(f"Saving baseline results to {region.upper()} DB...")

        if region.lower() == "taiwan" and engine_dict.get("taiwan"):
            save_engine = engine_dict["taiwan"]
        else:
            save_engine = db.engine

        source_summary = dict(df.attrs.get("production_summary") or {})
        out = df.copy()

        if "Node_Cell_ID" in out.columns and "node_cell_id" not in out.columns:
            out["node_cell_id"] = out["Node_Cell_ID"]
        if "node_cell_id" in out.columns:
            out["node_cell_id"] = _clean_text_series(out["node_cell_id"])

        # Existing baseline KPI columns store calibrated pre-smoothing values.
        # Smoothed/demo overlay values are preserved separately for display/audit.
        out = _prefer_columns(out, "pred_rsrp_smoothed", ["pred_rsrp_demo", "pred_rsrp"])
        out = _prefer_columns(out, "pred_rsrq_smoothed", ["pred_rsrq_demo", "pred_rsrq"])
        out = _prefer_columns(out, "pred_sinr_smoothed", ["pred_sinr_demo", "pred_sinr"])
        out = _prefer_columns(out, "pred_rsrp", ["pred_rsrp_calibrated", "pred_rsrp_geo", "pred_rsrp"])
        out = _prefer_columns(out, "pred_rsrq", ["pred_rsrq_calibrated", "pred_rsrq_geo", "pred_rsrq"])
        out = _prefer_columns(out, "pred_sinr", ["pred_sinr_calibrated", "pred_sinr_geo", "pred_sinr"])

        if site_df is not None and not site_df.empty:
            site_meta = site_df.copy()
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

                site_meta = site_meta.rename(columns=rename_map)
                keep_cols = ["node_cell_id"] + [
                    col for col in ["site_nodeb_id", "site_site_id", "site_operator"] if col in site_meta.columns
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

        out["project_id"] = project_id
        out["job_id"] = job_id
        out["created_at"] = datetime.now()

        out = _coalesce_columns(out, "node_b_id", ["node_b_id", "nodeb_id", "site_nodeb_id", "derived_nodeb_id"])
        out = _coalesce_columns(out, "cell_id", ["original_cell_id", "derived_cell_id", "cell_id"])
        out = _coalesce_columns(out, "operator", ["operator", "site_operator"], default=operator)
        out = _coalesce_columns(out, "site_id", ["site_id", "site_site_id", "node_b_id"])

        for col in ["node_b_id", "cell_id", "operator", "site_id"]:
            out[col] = _clean_text_series(out[col])

        legacy_nodeb_cell_id = np.where(
            out["node_b_id"].notna() & out["cell_id"].notna(),
            out["node_b_id"].astype(str) + "_" + out["cell_id"].astype(str),
            out.get("node_cell_id")
        )
        out["legacy_nodeb_id_cell_id"] = _clean_text_series(pd.Series(legacy_nodeb_cell_id, index=out.index))
        if "frontend_site_sector_key" in out.columns:
            out["nodeb_id_cell_id"] = _clean_text_series(out["frontend_site_sector_key"]).fillna(out["legacy_nodeb_id_cell_id"])
        else:
            out["nodeb_id_cell_id"] = out["legacy_nodeb_id_cell_id"]
        out["nodeb_id_cell_id"] = _clean_text_series(out["nodeb_id_cell_id"])

        self._save_geo_features(
            out,
            project_id=project_id,
            baseline_job_id=job_id,
            region=region,
            operator=operator,
            save_engine=save_engine,
            production_summary=source_summary,
        )

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
            "pred_rsrp_smoothed",
            "pred_rsrq_smoothed",
            "pred_sinr_smoothed",
            "node_b_id",
            "cell_id",
            "operator",
            "created_at",
            "site_id",
            "nodeb_id_cell_id",
            "Technology",
        ]

        out["id"] = pd.NA
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        out["lat_6dp"] = out["lat"].round(6)
        out["lon_6dp"] = out["lon"].round(6)
        if "Technology" not in out.columns:
            out["Technology"] = "4G"
        out["Technology"] = _clean_text_series(out["Technology"]).fillna("4G")
        for col, low, high in [
            ("pred_rsrp", -140, -44),
            ("pred_rsrq", -20, -3),
            ("pred_sinr", -10, 30),
            ("pred_rsrp_smoothed", -140, -44),
            ("pred_rsrq_smoothed", -20, -3),
            ("pred_sinr_smoothed", -10, 30),
        ]:
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(low, high)

        out = out[final_cols]
        out = out.dropna(subset=["project_id", "nodeb_id_cell_id", "lat_6dp", "lon_6dp"]).copy()
        out = out.drop_duplicates(
            subset=["project_id", "nodeb_id_cell_id", "lat_6dp", "lon_6dp"],
            keep="last",
        )
        _job_df_summary("BASELINE_DB_PAYLOAD", out)
        print(
            f"[LTE][BASELINE_DB_WRITE] table=lte_prediction_baseline_results "
            f"mode=upsert rows={len(out)} project_id={project_id} job_id={job_id}"
        )
        baseline_save_started = time.perf_counter()
        written_rows = self._upsert_baseline_results(save_engine, out, project_id=project_id)
        baseline_save_elapsed = time.perf_counter() - baseline_save_started
        print(
            f"[LTE][BASELINE_DB_WRITE_DONE] table=lte_prediction_baseline_results "
            f"rows={written_rows} elapsed_sec={baseline_save_elapsed:.2f}"
        )
        print(f"{written_rows} rows upserted into lte_prediction_baseline_results")

    def _save_geo_features(
        self,
        df,
        project_id,
        baseline_job_id,
        region,
        operator,
        save_engine,
        production_summary=None,
    ):
        geo_out = df.copy()
        production_summary = production_summary or {}

        geo_out["project_id"] = int(project_id)
        geo_out["baseline_job_id"] = str(baseline_job_id)
        geo_out["region"] = str(region).lower()
        geo_out["operator"] = _clean_text_series(
            geo_out["operator"] if "operator" in geo_out.columns else pd.Series([operator] * len(geo_out), index=geo_out.index)
        ).fillna(str(operator) if operator else None)

        if "proxy_site_id" not in geo_out.columns and "_proxy_site_id" in geo_out.columns:
            geo_out["proxy_site_id"] = geo_out["_proxy_site_id"]

        if "grid_id" not in geo_out.columns:
            geo_out["grid_id"] = pd.NA
        if "frontend_grid_id" in geo_out.columns:
            geo_out["grid_id"] = geo_out["frontend_grid_id"]

        geo_out["polygon_alignment"] = str(production_summary.get("polygon_alignment") or "")
        geo_out["building_alignment"] = str(production_summary.get("building_alignment") or "")
        geo_out["geo_source"] = "baseline_geo_correction"
        geo_out["created_at"] = datetime.now()
        geo_out["updated_at"] = datetime.now()

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

        for col in schema_cols:
            if col not in geo_out.columns:
                geo_out[col] = pd.NA

        for col in [
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
        ]:
            geo_out[col] = pd.to_numeric(geo_out[col], errors="coerce")

        geo_out["lat"] = geo_out["lat"].round(6)
        geo_out["lon"] = geo_out["lon"].round(6)
        geo_out["nodeb_id_cell_id"] = _clean_text_series(geo_out["nodeb_id_cell_id"])
        geo_out["proxy_site_id"] = _clean_text_series(geo_out["proxy_site_id"])
        geo_out["grid_id"] = _clean_text_series(geo_out["grid_id"])
        geo_out["clutter_class"] = _clean_text_series(geo_out["clutter_class"])

        geo_out = geo_out.dropna(subset=["lat", "lon", "nodeb_id_cell_id"]).copy()
        geo_out = geo_out[schema_cols].drop_duplicates(
            subset=["project_id", "nodeb_id_cell_id", "lat", "lon"],
            keep="last",
        )

        _job_df_summary("GEO_FEATURE_DB_PAYLOAD", geo_out)
        print(
            f"[LTE][GEO_DB_WRITE] table=lte_prediction_geo_features "
            f"mode=delta_upsert rows={len(geo_out)} project_id={project_id} baseline_job_id={baseline_job_id}"
        )

        geo_save_started = time.perf_counter()
        written_rows = self._upsert_geo_features(
            save_engine,
            geo_out,
            project_id=int(project_id),
            region=str(region).lower(),
        )
        geo_save_elapsed = time.perf_counter() - geo_save_started
        print(
            f"[LTE][GEO_DB_WRITE_DONE] table=lte_prediction_geo_features rows={written_rows} "
            f"elapsed_sec={geo_save_elapsed:.2f}"
        )

        print(f"{written_rows} rows written into lte_prediction_geo_features")
