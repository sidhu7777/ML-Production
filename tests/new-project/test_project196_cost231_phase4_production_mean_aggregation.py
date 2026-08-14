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
DATA_DIR = PROJECT_DIR / "cost231_phase4_production_mean_aggregation"
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


def main() -> None:
    _ensure_dirs()
    surface = _read_frame(f"cost231_offset_corrected_surface_project{PROJECT_ID}")
    dt = _read_frame(f"cost231_dt_match_project{PROJECT_ID}")
    offsets = _read_frame(f"cost231_offsets_102_cells_project{PROJECT_ID}")

    for col in ["raw_cost231_rsrp", "corrected_rsrp", "offset_corrected_rsrp", "offset_db", "dt_replacement_count"]:
        if col in surface.columns:
            surface[col] = pd.to_numeric(surface[col], errors="coerce")

    group_cols = ["grid_id", "lat", "lon"]
    phase4 = (
        surface.groupby(group_cols, dropna=False)
        .agg(
            raw_cost231_rsrp=("raw_cost231_rsrp", "mean"),
            corrected_rsrp=("corrected_rsrp", "mean"),
            offset_corrected_rsrp=("offset_corrected_rsrp", "mean"),
            offset_db=("offset_db", "median"),
            dt_replacement_count=("dt_replacement_count", "sum"),
            dt_replaced=("dt_replaced", "max"),
            source_cell_count=("strict_cell_key", "nunique"),
        )
        .reset_index()
    )

    phase4["project_id"] = PROJECT_ID
    phase4["strict_cell_key"] = "production_mean_all_cells"
    phase4["site_sector_band_key"] = "production_mean_all_cells"
    phase4["site"] = "production_mean"
    phase4["sector"] = "all"
    phase4["band"] = "all"
    phase4["operator"] = (
        surface["operator"].dropna().astype(str).mode().iloc[0]
        if "operator" in surface.columns and surface["operator"].notna().any()
        else "unknown"
    )
    phase4["offset_source"] = "production_mean_aggregation"
    phase4["dt_count"] = 0
    phase4["fallback_dt_count"] = 0
    phase4["raw_model_rsrp"] = phase4["raw_cost231_rsrp"]
    phase4["corrected_rsrp"] = phase4["corrected_rsrp"].clip(*CLIP_RSRP)
    phase4["raw_cost231_rsrp"] = phase4["raw_cost231_rsrp"].clip(*CLIP_RSRP)
    phase4["offset_corrected_rsrp"] = phase4["offset_corrected_rsrp"].clip(*CLIP_RSRP)
    phase4["dt_replaced"] = phase4["dt_replaced"].astype(bool)

    _save_frame(phase4, f"cost231_offset_corrected_surface_project{PROJECT_ID}")
    _save_frame(dt, f"cost231_dt_match_project{PROJECT_ID}")
    _save_frame(offsets, f"cost231_offsets_102_cells_project{PROJECT_ID}")

    _plot_cdf(
        [
            ("Phase 1 all cell rows after", surface["corrected_rsrp"], "#168a52"),
            ("Phase 4 production mean grid after", phase4["corrected_rsrp"], "#ef4444"),
        ],
        "Project 196 Cost231 Phase 4 - Production Mean Aggregation",
        COMBINED_DIR / "cdf_phase4_production_mean_vs_phase1_all_rows.png",
    )
    _plot_cdf(
        [
            ("Production mean before", phase4["raw_cost231_rsrp"], "#d94f3d"),
            ("Production mean after", phase4["corrected_rsrp"], "#168a52"),
        ],
        "Project 196 Cost231 Phase 4 - Mean Before/After",
        COMBINED_DIR / "cdf_phase4_production_mean_before_after.png",
    )

    bins = [-140, -120, -110, -100, -95, -85, -44]
    labels = ["-140 to -120", "-120 to -110", "-110 to -100", "-100 to -95", "-95 to -85", "-85 to -44"]
    summary = {
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "project_id": PROJECT_ID,
        "production_code_modified": False,
        "phase_label": "Cost231 Phase 4 production mean aggregation",
        "source_subdir": "cost231",
        "source_rows": int(len(surface)),
        "phase4_rows": int(len(phase4)),
        "source_cells": int(surface["strict_cell_key"].nunique()),
        "phase4_grid_pixels": int(phase4["grid_id"].nunique()),
        "aggregation": "mean corrected_rsrp across all cell rows for each grid pixel",
        "source_corrected_mean": float(surface["corrected_rsrp"].mean()),
        "phase4_corrected_mean": float(phase4["corrected_rsrp"].mean()),
        "phase4_corrected_min": float(phase4["corrected_rsrp"].min()),
        "phase4_corrected_max": float(phase4["corrected_rsrp"].max()),
        "phase4_bin_counts": {
            str(k): int(v)
            for k, v in pd.cut(
                phase4["corrected_rsrp"],
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
    (DATA_DIR / "cost231_phase4_production_mean_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
