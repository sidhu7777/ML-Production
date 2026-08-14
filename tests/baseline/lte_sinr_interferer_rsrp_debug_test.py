from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.lte_prediction.geo_correction_pipeline import _compute_proxy_rsrp_arrays, _haversine_m_np


DEFAULT_PROJECT_ID = 196


def _latest_rf_debug_run(project_id: int, output_root: Path) -> Path:
    project_dir = output_root / f"project_{project_id}"
    candidates = [
        path
        for path in project_dir.iterdir()
        if path.is_dir() and (path / "rf_prediction_grid_full.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No RF debug run with rf_prediction_grid_full.csv under {project_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _str_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    return df[col].fillna("").astype(str).str.strip()


def _safe_float(value: object) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _series_stats(series: pd.Series) -> Dict[str, Optional[float]]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    quantiles = clean.quantile([0.05, 0.10, 0.50, 0.90, 0.95])
    return {
        "count": int(len(clean)),
        "mean": _safe_float(clean.mean()),
        "min": _safe_float(clean.min()),
        "p05": _safe_float(quantiles.loc[0.05]),
        "p10": _safe_float(quantiles.loc[0.10]),
        "p50": _safe_float(quantiles.loc[0.50]),
        "p90": _safe_float(quantiles.loc[0.90]),
        "p95": _safe_float(quantiles.loc[0.95]),
        "max": _safe_float(clean.max()),
    }


def _site_series_or_default(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _site_str_or_default(frame: pd.DataFrame, col: str) -> pd.Series:
    if col in frame.columns:
        return frame[col].fillna("").astype(str).str.strip()
    return pd.Series("", index=frame.index, dtype=object)


def _candidate_rank_debug(
    run_dir: Path,
    pred_df: pd.DataFrame,
    severe_work: pd.DataFrame,
    max_interferers: int = 18,
    top_n: int = 500,
) -> Dict[str, object]:
    site_path = run_dir / "site_df.csv"
    if not site_path.exists():
        return {"available": False, "reason": f"Missing site artifact: {site_path}"}

    site_df = pd.read_csv(site_path, low_memory=False)
    if not {"lat", "lon", "Node_Cell_ID"}.issubset(site_df.columns):
        return {"available": False, "reason": "site_df is missing lat/lon/Node_Cell_ID"}

    serving_sites = (
        site_df.copy()
        .assign(Node_Cell_ID=site_df["Node_Cell_ID"].fillna("").astype(str).str.strip())
        .sort_values("Node_Cell_ID")
        .drop_duplicates(subset=["Node_Cell_ID"], keep="first")
        .reset_index(drop=True)
    )
    serving_sites["lat"] = pd.to_numeric(serving_sites["lat"], errors="coerce")
    serving_sites["lon"] = pd.to_numeric(serving_sites["lon"], errors="coerce")
    serving_sites = serving_sites.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if serving_sites.empty:
        return {"available": False, "reason": "site_df has no usable site coordinates"}

    cell_ids = serving_sites["Node_Cell_ID"].astype(str).str.strip().to_numpy(dtype=object)
    serving_lookup = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    site_lat = serving_sites["lat"].to_numpy(dtype=float)
    site_lon = serving_sites["lon"].to_numpy(dtype=float)
    site_az = _site_series_or_default(serving_sites, "azimuth", 0.0).to_numpy(dtype=float)
    site_height = _site_series_or_default(serving_sites, "antenna_height", 30.0).to_numpy(dtype=float)
    site_tx = _site_series_or_default(serving_sites, "tx_power", 46.0).to_numpy(dtype=float)
    site_freq = _site_series_or_default(
        serving_sites,
        "frequency_mhz" if "frequency_mhz" in serving_sites.columns else "frequency",
        1800.0,
    ).to_numpy(dtype=float)
    site_etilt = _site_series_or_default(serving_sites, "electrical_tilt", 3.0).to_numpy(dtype=float)
    site_mtilt = _site_series_or_default(serving_sites, "mechanical_tilt", 0.0).to_numpy(dtype=float)
    site_earfcn = _site_str_or_default(serving_sites, "earfcn").to_numpy(dtype=object)

    severe_idx = severe_work.sort_values("selected_minus_serving_db", ascending=False).head(int(top_n)).index
    records = []
    candidate_k = max(1, min(int(max_interferers), len(serving_sites)))
    for row_idx in severe_idx:
        point = pred_df.loc[row_idx]
        point_lat = _safe_float(point.get("lat"))
        point_lon = _safe_float(point.get("lon"))
        serving_cell = str(point.get("Node_Cell_ID", "")).strip()
        if point_lat is None or point_lon is None or serving_cell not in serving_lookup:
            continue

        distances = _haversine_m_np(site_lat, site_lon, point_lat, point_lon)
        nearest_idx = np.argsort(distances)[:candidate_k].astype(int).tolist()
        serving_idx = serving_lookup[serving_cell]
        if serving_idx not in nearest_idx:
            nearest_idx.append(serving_idx)
        candidate_arr = np.array(sorted(set(nearest_idx)), dtype=int)

        rsrp_raw = _compute_proxy_rsrp_arrays(
            np.full(len(candidate_arr), point_lat, dtype=float),
            np.full(len(candidate_arr), point_lon, dtype=float),
            site_lat[candidate_arr],
            site_lon[candidate_arr],
            site_az[candidate_arr],
            site_height[candidate_arr],
            site_tx[candidate_arr],
            site_freq[candidate_arr],
            site_etilt[candidate_arr],
            site_mtilt[candidate_arr],
        )
        serving_local = int(np.flatnonzero(candidate_arr == serving_idx)[0])
        serving_raw = float(rsrp_raw[serving_local])
        order = np.argsort(rsrp_raw)[::-1]
        serving_rank = int(np.flatnonzero(order == serving_local)[0]) + 1
        strongest_local = int(order[0])
        strongest_idx = int(candidate_arr[strongest_local])
        selected_cell = str(severe_work.loc[row_idx, "selected_interferer_cell_id"]).strip()
        selected_candidate_matches = np.flatnonzero(cell_ids[candidate_arr] == selected_cell)
        selected_raw_rank = None
        selected_raw_rsrp = None
        selected_is_raw_strongest = False
        if selected_candidate_matches.size:
            selected_local = int(selected_candidate_matches[0])
            selected_raw_rank = int(np.flatnonzero(order == selected_local)[0]) + 1
            selected_raw_rsrp = float(rsrp_raw[selected_local])
            selected_is_raw_strongest = selected_local == strongest_local

        same_earfcn_mask = site_earfcn[candidate_arr] == str(point.get("serving_earfcn", "")).strip()
        same_earfcn_non_serving = same_earfcn_mask.copy()
        same_earfcn_non_serving[serving_local] = False
        if same_earfcn_non_serving.any():
            same_order = np.argsort(rsrp_raw[same_earfcn_non_serving])[::-1]
            same_locals = np.flatnonzero(same_earfcn_non_serving)
            strongest_same_local = int(same_locals[same_order[0]])
            strongest_same_idx = int(candidate_arr[strongest_same_local])
            strongest_same_cell = str(cell_ids[strongest_same_idx])
            strongest_same_rsrp = float(rsrp_raw[strongest_same_local])
        else:
            strongest_same_cell = ""
            strongest_same_rsrp = np.nan

        records.append(
            {
                "row_index": int(row_idx),
                "lat": point_lat,
                "lon": point_lon,
                "serving_cell_id": serving_cell,
                "selected_interferer_cell_id": selected_cell,
                "serving_raw_rsrp_dbm": serving_raw,
                "selected_adjusted_rsrp_dbm": _safe_float(severe_work.loc[row_idx, "selected_interferer_rsrp_dbm"]),
                "selected_minus_serving_adjusted_db": _safe_float(severe_work.loc[row_idx, "selected_minus_serving_db"]),
                "serving_raw_rank": serving_rank,
                "strongest_raw_cell_id": str(cell_ids[strongest_idx]),
                "strongest_raw_rsrp_dbm": float(rsrp_raw[strongest_local]),
                "strongest_raw_minus_serving_db": float(rsrp_raw[strongest_local] - serving_raw),
                "serving_is_raw_strongest": bool(serving_rank == 1),
                "selected_raw_rank": selected_raw_rank,
                "selected_raw_rsrp_dbm": selected_raw_rsrp,
                "selected_is_raw_strongest": bool(selected_is_raw_strongest),
                "strongest_same_earfcn_cell_id": strongest_same_cell,
                "strongest_same_earfcn_rsrp_dbm": strongest_same_rsrp,
                "strongest_same_earfcn_minus_serving_db": float(strongest_same_rsrp - serving_raw)
                if np.isfinite(strongest_same_rsrp)
                else np.nan,
                "candidate_count": int(len(candidate_arr)),
            }
        )

    rank_df = pd.DataFrame(records)
    rank_path = run_dir / "sinr_serving_candidate_rank_debug.csv"
    rank_df.to_csv(rank_path, index=False)
    if rank_df.empty:
        return {"available": True, "rows": 0, "artifact": str(rank_path)}

    summary = {
        "available": True,
        "rows": int(len(rank_df)),
        "artifact": str(rank_path),
        "serving_raw_strongest_count": int(rank_df["serving_is_raw_strongest"].sum()),
        "serving_raw_strongest_pct": float(rank_df["serving_is_raw_strongest"].mean() * 100.0),
        "selected_raw_strongest_count": int(rank_df["selected_is_raw_strongest"].sum()),
        "selected_raw_strongest_pct": float(rank_df["selected_is_raw_strongest"].mean() * 100.0),
        "strongest_raw_not_serving_count": int((~rank_df["serving_is_raw_strongest"]).sum()),
        "serving_raw_rank_stats": _series_stats(rank_df["serving_raw_rank"]),
        "strongest_raw_minus_serving_db_stats": _series_stats(rank_df["strongest_raw_minus_serving_db"]),
        "selected_raw_rank_stats": _series_stats(rank_df["selected_raw_rank"]),
        "strongest_same_earfcn_minus_serving_db_stats": _series_stats(
            rank_df["strongest_same_earfcn_minus_serving_db"]
        ),
    }
    return summary


def debug_interferer_rsrp(run_dir: Path, top_n: int = 200, rank_top_n: int = 500) -> Dict[str, object]:
    prediction_path = run_dir / "rf_prediction_grid_full.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing RF prediction artifact: {prediction_path}")

    df = pd.read_csv(prediction_path, low_memory=False)
    serving = _num(df, "serving_proxy_rsrp_phys_dbm").combine_first(_num(df, "serving_proxy_rsrp_dbm"))
    best_interferer = _num(df, "best_interferer_proxy_phys_dbm").combine_first(
        _num(df, "best_interferer_proxy_rsrp_dbm")
    )
    neighbor_1 = _num(df, "neighbor_1_proxy_rsrp_dbm")
    selected_interferer = best_interferer.combine_first(neighbor_1)

    work = pd.DataFrame(
        {
            "lat": _num(df, "lat"),
            "lon": _num(df, "lon"),
            "grid_id": _num(df, "grid_id"),
            "node_cell_id": _str_col(df, "Node_Cell_ID"),
            "serving_rsrp_dbm": serving,
            "selected_interferer_rsrp_dbm": selected_interferer,
            "selected_minus_serving_db": selected_interferer - serving,
            "sinr_proxy_db": _num(df, "sinr_proxy_db"),
            "pred_sinr": _num(df, "pred_sinr"),
            "pred_sinr_geo": _num(df, "pred_sinr_geo"),
            "interference_gap_db": _num(df, "interference_gap_db"),
            "same_earfcn_interferer_count": _num(df, "same_earfcn_interferer_count"),
            "dominant_interferer_count": _num(df, "dominant_interferer_count"),
            "serving_earfcn": _str_col(df, "serving_earfcn"),
            "selected_interferer_cell_id": _str_col(df, "best_interferer_cell_id").where(
                _str_col(df, "best_interferer_cell_id") != "",
                _str_col(df, "neighbor_1_cell_id"),
            ),
            "selected_interferer_earfcn": _str_col(df, "best_interferer_earfcn").where(
                _str_col(df, "best_interferer_earfcn") != "",
                _str_col(df, "neighbor_1_earfcn"),
            ),
            "selected_interferer_distance_m": _num(df, "best_interferer_distance_m").combine_first(
                _num(df, "neighbor_1_distance_m")
            ),
            "selected_interferer_azimuth_delta_deg": _num(df, "best_interferer_azimuth_delta_deg").combine_first(
                _num(df, "neighbor_1_azimuth_delta_deg")
            ),
            "interference_selection_mode": _str_col(df, "interference_selection_mode"),
        }
    )

    valid = work.dropna(subset=["serving_rsrp_dbm", "selected_interferer_rsrp_dbm"]).copy()
    delta = valid["selected_minus_serving_db"]
    summary: Dict[str, object] = {
        "run_dir": str(run_dir),
        "prediction_rows": int(len(df)),
        "valid_serving_interferer_rows": int(len(valid)),
        "missing_selected_interferer_rows": int(len(df) - len(valid)),
        "serving_rsrp_stats": _series_stats(valid["serving_rsrp_dbm"]),
        "selected_interferer_rsrp_stats": _series_stats(valid["selected_interferer_rsrp_dbm"]),
        "selected_minus_serving_db_stats": _series_stats(delta),
        "selected_interferer_stronger_count": int((delta > 0.0).sum()),
        "selected_interferer_stronger_pct": float((delta > 0.0).mean() * 100.0) if len(valid) else 0.0,
        "selected_interferer_gt_serving_by_3db_count": int((delta > 3.0).sum()),
        "selected_interferer_gt_serving_by_6db_count": int((delta > 6.0).sum()),
        "selected_interferer_within_3db_count": int((delta.abs() <= 3.0).sum()),
        "negative_interference_gap_count": int((_num(valid, "interference_gap_db") < 0.0).sum()),
        "sinr_proxy_negative_count": int((_num(valid, "sinr_proxy_db") < 0.0).sum()),
        "selection_mode_counts": valid["interference_selection_mode"].value_counts(dropna=False).head(20).to_dict(),
        "dominant_interferer_count_stats": _series_stats(valid["dominant_interferer_count"]),
        "same_earfcn_interferer_count_stats": _series_stats(valid["same_earfcn_interferer_count"]),
    }

    worst_rows = valid.sort_values("selected_minus_serving_db", ascending=False).head(int(top_n))
    worst_path = run_dir / "sinr_interferer_rsrp_worst_rows.csv"
    worst_rows.to_csv(worst_path, index=False)

    by_pair = (
        valid.assign(_is_stronger=delta > 0.0)
        .groupby(["node_cell_id", "selected_interferer_cell_id"], dropna=False)
        .agg(
            rows=("selected_minus_serving_db", "size"),
            mean_selected_minus_serving_db=("selected_minus_serving_db", "mean"),
            p90_selected_minus_serving_db=("selected_minus_serving_db", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.90)),
            stronger_rows=("_is_stronger", "sum"),
            mean_sinr_proxy_db=("sinr_proxy_db", "mean"),
            mean_interference_gap_db=("interference_gap_db", "mean"),
        )
        .reset_index()
        .sort_values(["stronger_rows", "mean_selected_minus_serving_db"], ascending=[False, False])
        .head(int(top_n))
    )
    pair_path = run_dir / "sinr_interferer_rsrp_worst_pairs.csv"
    by_pair.to_csv(pair_path, index=False)

    summary["artifacts"] = {
        "worst_rows": str(worst_path),
        "worst_pairs": str(pair_path),
    }
    severe = valid.loc[valid["selected_minus_serving_db"] > 6.0].copy()
    rank_summary = _candidate_rank_debug(
        run_dir,
        df,
        severe,
        max_interferers=18,
        top_n=rank_top_n,
    )
    summary["candidate_rank_debug"] = rank_summary
    if rank_summary.get("artifact"):
        summary["artifacts"]["candidate_rank"] = str(rank_summary["artifact"])
    summary_path = run_dir / "sinr_interferer_rsrp_debug_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["artifacts"]["summary"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug selected LTE SINR interferer RSRP vs serving RSRP.")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("tests/output"))
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--rank-top-n", type=int, default=500)
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_rf_debug_run(args.project_id, args.output_root)
    summary = debug_interferer_rsrp(run_dir, top_n=args.top_n, rank_top_n=args.rank_top_n)
    print("[SINR_INTERFERER_DEBUG] run_dir=", summary["run_dir"])
    print("[SINR_INTERFERER_DEBUG] valid_rows=", summary["valid_serving_interferer_rows"])
    print("[SINR_INTERFERER_DEBUG] selected_interferer_stronger_count=", summary["selected_interferer_stronger_count"])
    print("[SINR_INTERFERER_DEBUG] selected_interferer_stronger_pct=", round(float(summary["selected_interferer_stronger_pct"]), 2))
    print("[SINR_INTERFERER_DEBUG] gt_serving_by_3db=", summary["selected_interferer_gt_serving_by_3db_count"])
    print("[SINR_INTERFERER_DEBUG] gt_serving_by_6db=", summary["selected_interferer_gt_serving_by_6db_count"])
    print("[SINR_INTERFERER_DEBUG] selected_minus_serving_stats=", summary["selected_minus_serving_db_stats"])
    print("[SINR_INTERFERER_DEBUG] candidate_rank_debug=", summary["candidate_rank_debug"])
    print("[SINR_INTERFERER_DEBUG] artifacts=", summary["artifacts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
