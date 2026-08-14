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
DATA_DIR = PROJECT_DIR / "cost231_phase7_serving_grid_production_average"
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


def _read_phase_values(subdir: str) -> pd.Series:
    base = PROJECT_DIR / subdir / f"cost231_offset_corrected_surface_project{PROJECT_ID}"
    parquet_path = base.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path, columns=["corrected_rsrp"])["corrected_rsrp"]
    return pd.read_csv(base.with_suffix(".csv"), usecols=["corrected_rsrp"])["corrected_rsrp"]


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

    work = surface.copy()
    work["grid_id"] = work["grid_id"].astype(str)
    best_idx = work.groupby("grid_id", dropna=False)["raw_cost231_rsrp"].idxmax()
    serving = work.loc[best_idx].copy()
    serving = serving.merge(grid_dt, on="grid_id", how="left")
    serving["final_grid_rsrp"] = serving["offset_corrected_rsrp"].where(
        serving["gridwide_dt_rsrp"].isna(),
        serving["gridwide_dt_rsrp"],
    )
    serving["final_grid_rsrp"] = pd.to_numeric(serving["final_grid_rsrp"], errors="coerce").clip(*CLIP_RSRP)
    serving["gridwide_dt_replaced"] = serving["gridwide_dt_rsrp"].notna()

    final_lookup = serving[
        [
            "grid_id",
            "strict_cell_key",
            "site_sector_band_key",
            "site",
            "sector",
            "band",
            "operator",
            "raw_cost231_rsrp",
            "offset_corrected_rsrp",
            "offset_db",
            "final_grid_rsrp",
            "gridwide_dt_rsrp",
            "gridwide_dt_count",
            "gridwide_dt_replaced",
        ]
    ].rename(
        columns={
            "strict_cell_key": "serving_strict_cell_key",
            "site_sector_band_key": "serving_site_sector_band_key",
            "site": "serving_site",
            "sector": "serving_sector",
            "band": "serving_band",
            "operator": "serving_operator",
            "raw_cost231_rsrp": "serving_raw_cost231_rsrp",
            "offset_corrected_rsrp": "serving_offset_corrected_rsrp",
            "offset_db": "serving_offset_db",
        }
    )

    phase7_surface = work.merge(final_lookup, on="grid_id", how="left")
    phase7_surface["raw_cost231_rsrp"] = phase7_surface["serving_raw_cost231_rsrp"]
    phase7_surface["offset_corrected_rsrp"] = phase7_surface["serving_offset_corrected_rsrp"]
    phase7_surface["offset_db"] = phase7_surface["serving_offset_db"]
    phase7_surface["corrected_rsrp"] = phase7_surface["final_grid_rsrp"]
    phase7_surface["dt_replaced"] = phase7_surface["gridwide_dt_replaced"].fillna(False).astype(bool)
    phase7_surface["dt_replacement_count"] = phase7_surface["gridwide_dt_count"].fillna(0).astype(int)
    phase7_surface["strict_cell_key"] = phase7_surface["serving_strict_cell_key"]
    phase7_surface["site_sector_band_key"] = phase7_surface["serving_site_sector_band_key"]
    phase7_surface["site"] = phase7_surface["serving_site"]
    phase7_surface["sector"] = phase7_surface["serving_sector"]
    phase7_surface["band"] = phase7_surface["serving_band"]
    phase7_surface["operator"] = phase7_surface["serving_operator"]
    phase7_surface["offset_source"] = "serving_grid_value_copied_to_all_rows"

    phase7 = (
        phase7_surface.groupby(["grid_id", "lat", "lon"], dropna=False)
        .agg(
            raw_cost231_rsrp=("raw_cost231_rsrp", "mean"),
            corrected_rsrp=("corrected_rsrp", "mean"),
            offset_corrected_rsrp=("offset_corrected_rsrp", "mean"),
            offset_db=("offset_db", "median"),
            dt_replacement_count=("dt_replacement_count", "max"),
            dt_replaced=("dt_replaced", "max"),
            source_cell_count=("strict_cell_key", "nunique"),
            strict_cell_key=("strict_cell_key", "first"),
            site_sector_band_key=("site_sector_band_key", "first"),
            site=("site", "first"),
            sector=("sector", "first"),
            band=("band", "first"),
            operator=("operator", "first"),
        )
        .reset_index()
    )
    phase7["project_id"] = PROJECT_ID
    phase7["raw_model_rsrp"] = phase7["raw_cost231_rsrp"]
    phase7["offset_source"] = "serving_grid_value_then_production_mean"
    phase7["dt_count"] = 0
    phase7["fallback_dt_count"] = 0
    phase7["corrected_rsrp"] = phase7["corrected_rsrp"].clip(*CLIP_RSRP)
    phase7["raw_cost231_rsrp"] = phase7["raw_cost231_rsrp"].clip(*CLIP_RSRP)
    phase7["offset_corrected_rsrp"] = phase7["offset_corrected_rsrp"].clip(*CLIP_RSRP)
    phase7["dt_replaced"] = phase7["dt_replaced"].astype(bool)

    _save_frame(phase7, f"cost231_offset_corrected_surface_project{PROJECT_ID}")
    _save_frame(dt, f"cost231_dt_match_project{PROJECT_ID}")
    _save_frame(offsets, f"cost231_offsets_102_cells_project{PROJECT_ID}")

    _plot_cdf(
        [
            ("Phase 4 production mean", _read_phase_values("cost231_phase4_production_mean_aggregation"), "#ef4444"),
            ("Phase 5 gridwide DT only", _read_phase_values("cost231_phase5_gridwide_dt_replacement"), "#f97316"),
            ("Phase 7 serving-grid corrected", phase7["corrected_rsrp"], "#168a52"),
            ("DT measured", dt["rsrp_measured"], "#2563eb"),
        ],
        "Project 196 Cost231 Phase 7 - Serving Grid Production Average",
        COMBINED_DIR / "cdf_phase7_serving_grid_vs_phase4_phase5.png",
    )
    _plot_cdf(
        [
            ("Serving raw Cost231", phase7["raw_cost231_rsrp"], "#d94f3d"),
            ("Serving offset/DT corrected", phase7["corrected_rsrp"], "#168a52"),
            ("DT measured", dt["rsrp_measured"], "#2563eb"),
        ],
        "Project 196 Cost231 Phase 7 - Before/After",
        COMBINED_DIR / "cdf_phase7_before_after.png",
    )

    bins = [-140, -120, -110, -100, -95, -85, -44]
    labels = ["-140 to -120", "-120 to -110", "-110 to -100", "-100 to -95", "-95 to -85", "-85 to -44"]
    summary = {
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "project_id": PROJECT_ID,
        "production_code_modified": False,
        "phase_label": "Cost231 Phase 7 serving-grid value copied before production average",
        "source_subdir": "cost231",
        "source_rows": int(len(surface)),
        "phase7_rows": int(len(phase7)),
        "source_cells": int(surface["strict_cell_key"].nunique()),
        "phase7_grid_pixels": int(phase7["grid_id"].nunique()),
        "dt_rows": int(len(dt)),
        "dt_eligible_rows": int(len(eligible)),
        "gridwide_dt_pixels": int(grid_dt["grid_id"].nunique()),
        "serving_cells_used": int(serving["strict_cell_key"].nunique()),
        "aggregation": "choose max raw Cost231 serving cell per grid, apply its offset, replace with DT if grid has DT, copy value to all rows, then production mean",
        "phase7_corrected_mean": float(phase7["corrected_rsrp"].mean()),
        "phase7_corrected_min": float(phase7["corrected_rsrp"].min()),
        "phase7_corrected_max": float(phase7["corrected_rsrp"].max()),
        "phase7_bin_counts": {
            str(k): int(v)
            for k, v in pd.cut(
                phase7["corrected_rsrp"],
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
    (DATA_DIR / "cost231_phase7_serving_grid_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
