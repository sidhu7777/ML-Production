from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ID = 196
PROJECT_DIR = THIS_DIR / "data" / "project_196_india"
SOURCE_DIR = PROJECT_DIR / "cost231"
DATA_DIR = PROJECT_DIR / "cost231_phase5_gridwide_dt_replacement"
COMBINED_DIR = DATA_DIR / "combined"
INDIVIDUAL_DIR = DATA_DIR / "individually"
CLIP_RSRP = (-140.0, -44.0)


def _ensure_dirs() -> None:
    for path in [DATA_DIR, COMBINED_DIR, INDIVIDUAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _read_frame(stem: str) -> pd.DataFrame:
    parquet_path = SOURCE_DIR / f"{stem}.parquet"
    csv_path = SOURCE_DIR / f"{stem}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path, low_memory=False)


def _cdf_values(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    if arr.size == 0:
        return arr, arr
    return arr, np.arange(1, arr.size + 1, dtype=float) / arr.size * 100.0


def _plot_cdf(series_map: list[tuple[str, pd.Series, str]], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label, values, color in series_map:
        x, y = _cdf_values(values)
        if len(x):
            ax.plot(x, y, linewidth=2.4, color=color, label=f"{label} (n={len(x):,})")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("RSRP (dBm)")
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.set_xlim(CLIP_RSRP)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_frame(df: pd.DataFrame, stem: str) -> None:
    csv_path = DATA_DIR / f"{stem}.csv"
    parquet_path = DATA_DIR / f"{stem}.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        parquet_path.with_suffix(".parquet.error.txt").write_text(str(exc), encoding="utf-8")


def _production_mean(surface: pd.DataFrame) -> pd.DataFrame:
    phase = (
        surface.groupby(["grid_id", "lat", "lon"], dropna=False)
        .agg(
            raw_cost231_rsrp=("raw_cost231_rsrp", "mean"),
            corrected_rsrp=("corrected_rsrp", "mean"),
            offset_corrected_rsrp=("offset_corrected_rsrp", "mean"),
            offset_db=("offset_db", "median"),
            dt_replacement_count=("dt_replacement_count", "sum"),
            dt_replaced=("dt_replaced", "max"),
            source_cell_count=("strict_cell_key", "nunique"),
            gridwide_dt_rsrp=("gridwide_dt_rsrp", "mean"),
        )
        .reset_index()
    )
    phase["project_id"] = PROJECT_ID
    phase["strict_cell_key"] = "production_mean_gridwide_dt"
    phase["site_sector_band_key"] = "production_mean_gridwide_dt"
    phase["site"] = "production_mean"
    phase["sector"] = "all"
    phase["band"] = "all"
    phase["operator"] = (
        surface["operator"].dropna().astype(str).mode().iloc[0]
        if "operator" in surface.columns and surface["operator"].notna().any()
        else "unknown"
    )
    phase["offset_source"] = "gridwide_dt_replacement_then_production_mean"
    phase["dt_count"] = 0
    phase["fallback_dt_count"] = 0
    phase["raw_model_rsrp"] = phase["raw_cost231_rsrp"]
    phase["corrected_rsrp"] = phase["corrected_rsrp"].clip(*CLIP_RSRP)
    phase["raw_cost231_rsrp"] = phase["raw_cost231_rsrp"].clip(*CLIP_RSRP)
    phase["offset_corrected_rsrp"] = phase["offset_corrected_rsrp"].clip(*CLIP_RSRP)
    phase["dt_replaced"] = phase["dt_replaced"].astype(bool)
    return phase


def main() -> None:
    _ensure_dirs()
    surface = _read_frame(f"cost231_offset_corrected_surface_project{PROJECT_ID}")
    dt = _read_frame(f"cost231_dt_match_project{PROJECT_ID}")
    offsets = _read_frame(f"cost231_offsets_102_cells_project{PROJECT_ID}")

    for col in ["raw_cost231_rsrp", "corrected_rsrp", "offset_corrected_rsrp", "offset_db", "dt_replacement_count"]:
        if col in surface.columns:
            surface[col] = pd.to_numeric(surface[col], errors="coerce")
    dt["rsrp_measured"] = pd.to_numeric(dt["rsrp_measured"], errors="coerce")

    eligible = dt.loc[dt["dt_replacement_eligible"].astype(bool)].dropna(subset=["nearest_grid_id", "rsrp_measured"]).copy()
    grid_dt = (
        eligible.groupby("nearest_grid_id", dropna=False)
        .agg(
            gridwide_dt_rsrp=("rsrp_measured", "mean"),
            gridwide_dt_count=("rsrp_measured", "size"),
        )
        .reset_index()
        .rename(columns={"nearest_grid_id": "grid_id"})
    )
    grid_dt["grid_id"] = grid_dt["grid_id"].astype(str)

    phase5_surface = surface.copy()
    phase5_surface["grid_id"] = phase5_surface["grid_id"].astype(str)
    phase5_surface = phase5_surface.merge(grid_dt, on="grid_id", how="left")
    phase5_surface["gridwide_dt_replaced"] = phase5_surface["gridwide_dt_rsrp"].notna()
    phase5_surface["corrected_rsrp"] = phase5_surface["corrected_rsrp"].where(
        ~phase5_surface["gridwide_dt_replaced"],
        phase5_surface["gridwide_dt_rsrp"],
    )
    phase5_surface["corrected_rsrp"] = pd.to_numeric(phase5_surface["corrected_rsrp"], errors="coerce").clip(*CLIP_RSRP)
    phase5_surface["dt_replaced"] = phase5_surface["gridwide_dt_replaced"]
    phase5_surface["dt_replacement_count"] = phase5_surface["gridwide_dt_count"].fillna(0).astype(int)

    phase5 = _production_mean(phase5_surface)

    _save_frame(phase5, f"cost231_offset_corrected_surface_project{PROJECT_ID}")
    _save_frame(dt, f"cost231_dt_match_project{PROJECT_ID}")
    _save_frame(offsets, f"cost231_offsets_102_cells_project{PROJECT_ID}")

    _plot_cdf(
        [
            ("Phase 4 production mean", _read_phase4(), "#ef4444"),
            ("Phase 5 gridwide DT then mean", phase5["corrected_rsrp"], "#168a52"),
            ("DT measured", dt["rsrp_measured"], "#2563eb"),
        ],
        "Project 196 Cost231 Phase 5 - Gridwide DT Replacement",
        COMBINED_DIR / "cdf_phase5_gridwide_dt_vs_phase4_mean.png",
    )
    _plot_cdf(
        [
            ("Before production mean", phase5["raw_cost231_rsrp"], "#d94f3d"),
            ("After gridwide DT + production mean", phase5["corrected_rsrp"], "#168a52"),
        ],
        "Project 196 Cost231 Phase 5 - Before/After",
        COMBINED_DIR / "cdf_phase5_before_after.png",
    )

    bins = [-140, -120, -110, -100, -95, -85, -44]
    labels = ["-140 to -120", "-120 to -110", "-110 to -100", "-100 to -95", "-95 to -85", "-85 to -44"]
    summary = {
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "project_id": PROJECT_ID,
        "production_code_modified": False,
        "phase_label": "Cost231 Phase 5 gridwide DT replacement then production mean aggregation",
        "source_subdir": "cost231",
        "source_rows": int(len(surface)),
        "phase5_rows": int(len(phase5)),
        "source_cells": int(surface["strict_cell_key"].nunique()),
        "phase5_grid_pixels": int(phase5["grid_id"].nunique()),
        "dt_rows": int(len(dt)),
        "dt_eligible_rows": int(len(eligible)),
        "gridwide_dt_pixels": int(grid_dt["grid_id"].nunique()),
        "gridwide_replaced_cell_rows": int(phase5_surface["gridwide_dt_replaced"].sum()),
        "aggregation": "replace every cell row at DT grid with DT mean, then mean across rows per grid",
        "phase5_corrected_mean": float(phase5["corrected_rsrp"].mean()),
        "phase5_corrected_min": float(phase5["corrected_rsrp"].min()),
        "phase5_corrected_max": float(phase5["corrected_rsrp"].max()),
        "phase5_bin_counts": {
            str(k): int(v)
            for k, v in pd.cut(
                phase5["corrected_rsrp"],
                bins=bins,
                labels=labels,
                right=False,
                include_lowest=True,
            )
            .value_counts(sort=False)
            .items()
        },
        "combined_dir": str(COMBINED_DIR.relative_to(THIS_DIR)),
        "individual_dir": str(INDIVIDUAL_DIR.relative_to(THIS_DIR)),
    }
    (DATA_DIR / "cost231_phase5_gridwide_dt_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


def _read_phase4() -> pd.Series:
    path = PROJECT_DIR / "cost231_phase4_production_mean_aggregation" / f"cost231_offset_corrected_surface_project{PROJECT_ID}.parquet"
    if path.exists():
        return pd.read_parquet(path, columns=["corrected_rsrp"])["corrected_rsrp"]
    path = path.with_suffix(".csv")
    return pd.read_csv(path, usecols=["corrected_rsrp"])["corrected_rsrp"]


if __name__ == "__main__":
    main()
