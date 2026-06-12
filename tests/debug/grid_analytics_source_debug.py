from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
load_dotenv(ML_ROOT / ".env")


def _db_url(region: str) -> str:
    if region.strip().lower() == "taiwan":
        return os.getenv("DATABASE_URL_Taiwan", "")
    return os.getenv("DATABASE_URL", "")


def _fetch_json(url: str, api_key: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"X-Python-Bridge-Key": api_key})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _print_df(title: str, df: pd.DataFrame) -> None:
    print(f"\n[{title}] rows={len(df)}")
    if df.empty:
        return
    print(df.to_string(index=False))


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct DB vs C# bridge grid analytics selection.")
    parser.add_argument("--project-id", type=int, default=196)
    parser.add_argument("--region", default="india")
    parser.add_argument("--operator", default="Airtel")
    parser.add_argument("--bridge-base-url", default=os.getenv("PYTHON_BRIDGE_BASE_URL", "http://localhost:5224"))
    parser.add_argument("--bridge-api-key", default=os.getenv("PYTHON_BRIDGE_API_KEY", "s-tracer-local-bridge"))
    args = parser.parse_args()

    db_url = _db_url(args.region)
    if not db_url:
        raise RuntimeError(f"Database URL not configured for region={args.region}")

    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        cols = pd.read_sql(
            text(
                """
                SELECT COLUMN_NAME
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                  AND table_name = 'grid_analytics_results'
                """
            ),
            conn,
        )
        col_set = set(cols["COLUMN_NAME"].astype(str).tolist()) if "COLUMN_NAME" in cols.columns else set(cols.iloc[:, 0].astype(str).tolist())
        has_operator = "operator" in col_set

        where = ["project_id = :pid"]
        params: dict[str, Any] = {"pid": int(args.project_id)}
        if has_operator and args.operator.strip().lower() not in {"", "all", "none"}:
            where.append("LOWER(TRIM(operator)) = :op")
            params["op"] = args.operator.strip().lower()
        where_sql = " AND ".join(where)

        groups = pd.read_sql(
            text(
                f"""
                SELECT
                    scenario_id,
                    region_id,
                    ROUND(grid_size_meters, 3) AS grid_size_m,
                    COUNT(*) AS row_count,
                    MIN(created_at) AS min_created,
                    MAX(created_at) AS max_created,
                    SUM(CASE WHEN baseline_avg_rsrp < -90 THEN 1 ELSE 0 END) AS bad_rsrp_grids,
                    SUM(CASE WHEN baseline_avg_rsrq < -14 THEN 1 ELSE 0 END) AS bad_rsrq_grids,
                    SUM(CASE WHEN baseline_avg_sinr < 0 THEN 1 ELSE 0 END) AS bad_sinr_grids,
                    SUM(
                        CASE WHEN (
                            GREATEST(-90 - baseline_avg_rsrp, 0) * 0.2 +
                            GREATEST(-14 - baseline_avg_rsrq, 0) * 0.2 +
                            GREATEST(0 - baseline_avg_sinr, 0) * 0.6
                        ) > 0 THEN 1 ELSE 0 END
                    ) AS bad_combined_grids,
                    SUM(baseline_point_count) AS baseline_points,
                    SUM(optimized_point_count) AS optimized_points
                FROM grid_analytics_results
                WHERE {where_sql}
                GROUP BY scenario_id, region_id, ROUND(grid_size_meters, 3)
                ORDER BY max_created DESC, row_count DESC
                """
            ),
            conn,
            params=params,
        )

        direct_scenario = pd.read_sql(
            text(
                f"""
                SELECT
                    scenario_id,
                    MAX(created_at) AS max_created,
                    COUNT(*) AS row_count
                FROM grid_analytics_results
                WHERE {where_sql}
                GROUP BY scenario_id
                ORDER BY row_count DESC, max_created DESC
                LIMIT 1
                """
            ),
            conn,
            params=params,
        )

        bridge_candidates = pd.read_sql(
            text(
                f"""
                SELECT
                    scenario_id,
                    region_id,
                    ROUND(grid_size_meters, 3) AS grid_size_m,
                    COUNT(*) AS row_count,
                    MAX(created_at) AS max_created
                FROM grid_analytics_results
                WHERE {where_sql}
                  AND (region_id IS NULL OR region_id <= 0)
                GROUP BY scenario_id, region_id, ROUND(grid_size_meters, 3)
                ORDER BY max_created DESC, row_count DESC
                """
            ),
            conn,
            params=params,
        )

        latest_baseline_job = pd.read_sql(
            text(
                """
                SELECT job_id, COUNT(*) AS row_count, MAX(created_at) AS max_created
                FROM lte_prediction_baseline_results
                WHERE project_id = :pid
                GROUP BY job_id
                ORDER BY max_created DESC
                LIMIT 1
                """
            ),
            conn,
            params={"pid": int(args.project_id)},
        )
        baseline_summary = pd.DataFrame()
        if not latest_baseline_job.empty:
            job_id = latest_baseline_job.iloc[0]["job_id"]
            baseline_summary = pd.read_sql(
                text(
                    """
                    SELECT
                        job_id,
                        COUNT(*) AS row_count,
                        SUM(CASE WHEN pred_rsrp < -90 THEN 1 ELSE 0 END) AS bad_rsrp_points,
                        SUM(CASE WHEN pred_rsrq < -14 THEN 1 ELSE 0 END) AS bad_rsrq_points,
                        SUM(CASE WHEN pred_sinr < 0 THEN 1 ELSE 0 END) AS bad_sinr_points,
                        SUM(
                            CASE WHEN (
                                GREATEST(-90 - pred_rsrp, 0) * 0.2 +
                                GREATEST(-14 - pred_rsrq, 0) * 0.2 +
                                GREATEST(0 - pred_sinr, 0) * 0.6
                            ) > 0 THEN 1 ELSE 0 END
                        ) AS bad_combined_points,
                        MIN(created_at) AS min_created,
                        MAX(created_at) AS max_created
                    FROM lte_prediction_baseline_results
                    WHERE project_id = :pid
                      AND job_id = :job_id
                    GROUP BY job_id
                    """
                ),
                conn,
                params={"pid": int(args.project_id), "job_id": job_id},
            )

    _print_df("DB generation groups", groups)
    _print_df("Python direct selector: GROUP BY scenario_id ORDER BY row_count DESC", direct_scenario)
    _print_df("Bridge selector candidates: region_id NULL/0, latest scenario+grid_size generation", bridge_candidates.head(10))
    _print_df("Latest baseline job", latest_baseline_job)
    _print_df("Latest baseline KPI point counts", baseline_summary)

    print("\n[Code hashes]")
    for rel in [
        "tools/lte_tilt_recommandation/candidate_validation.py",
        "tools/lte_tilt_recommandation/recommendation_engine.py",
        "tools/lte_tilt_recommandation/services.py",
        "tools/lte_tilt_recommandation/etilt_optimizer_cd2.py",
    ]:
        path = ML_ROOT / rel
        print(f"{rel} sha256_16={_file_sha(path)}")

    url = (
        args.bridge_base_url.rstrip("/")
        + "/api/GridAnalytics/GetGridAnalytics?"
        + urllib.parse.urlencode({"projectId": int(args.project_id), "version": "original"})
    )
    try:
        payload = _fetch_json(url, args.bridge_api_key)
        data = payload.get("Data") or payload.get("data") or {}
        grids = data.get("grids") or []
        print("\n[Bridge API]")
        print(f"url={url}")
        print(f"status={payload.get('Status') or payload.get('status')}")
        print(f"grid_size_meters={data.get('grid_size_meters') or data.get('gridSizeMeters')}")
        print(f"total_grids_with_data={data.get('total_grids_with_data') or data.get('totalGridsWithData')}")
        print(f"actual_grid_rows={len(grids)}")
    except (urllib.error.URLError, TimeoutError) as exc:
        print("\n[Bridge API]")
        print(f"failed={exc}")

    print("\n[Diagnosis]")
    if not direct_scenario.empty and not bridge_candidates.empty:
        d = direct_scenario.iloc[0]
        b = bridge_candidates.iloc[0]
        print(
            "direct_python_selects="
            f"scenario_id={d.get('scenario_id')} rows={int(d.get('row_count') or 0)}"
        )
        print(
            "bridge_selects="
            f"scenario_id={b.get('scenario_id')} region_id={b.get('region_id')} "
            f"grid_size={b.get('grid_size_m')} rows={int(b.get('row_count') or 0)}"
        )
        if int(d.get("row_count") or 0) != int(b.get("row_count") or 0):
            print("cause=row_count_mismatch_from_different_selection_rules")
        else:
            print("cause=selection_counts_match")


if __name__ == "__main__":
    main()
