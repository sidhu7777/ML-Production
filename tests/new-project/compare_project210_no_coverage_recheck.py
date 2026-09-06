from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase_rsrp_guard import RSRP_NO_COVERAGE_DBM


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
OUT_DIR = PROJECT_DIR / "no_coverage_recheck_snapshots"

PHASE_GRID_SPECS = [
    ("Phase17", "cost231_phase17_geo_dt_comparison", "phase17_serving_grid_{tech}_project210", "phase17_rsrp"),
    ("Phase19", "cost231_phase19_branch_calibrated_comparison", "phase19_serving_grid_{tech}_project210", "phase19_rsrp"),
    ("Phase22", "cost231_phase22_terrain_diffraction_comparison", "phase22_serving_grid_{tech}_project210", "phase22_with_terrain_best_rsrp"),
    ("Phase24", "cost231_phase24_physical_clutter_role_fix", "phase24_serving_grid_{tech}_project210", "phase24_with_terrain_best_rsrp"),
    ("Phase25", "cost231_phase25_hierarchical_dynamic_calibration", "phase25_serving_grid_{tech}_project210", "phase25_dynamic_best_rsrp"),
]

DT_SPECS = [
    (
        "Phase22 DT",
        "cost231_phase22_terrain_diffraction_comparison",
        "phase22_dt_terrain_scored_project210",
        "assigned_technology",
        "phase22_with_terrain_calibrated_rsrp",
    ),
    (
        "Phase24 DT",
        "cost231_phase24_physical_clutter_role_fix",
        "phase24_dt_scored_project210",
        "assigned_technology",
        "phase24_with_terrain_calibrated_rsrp",
    ),
    (
        "Phase25 validation",
        "cost231_phase25_hierarchical_dynamic_calibration",
        "phase25_validation_dt_project210",
        "technology",
        "phase25_dynamic_rsrp",
    ),
]


def _read_frame(stem: Path) -> pd.DataFrame:
    parquet_path = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    return pd.DataFrame()


def _cdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    if arr.size == 0:
        return arr, arr
    return arr, np.arange(1, arr.size + 1, dtype=float) / arr.size * 100.0


def _metrics(measured: pd.Series, predicted: pd.Series) -> dict:
    err = pd.to_numeric(measured, errors="coerce") - pd.to_numeric(predicted, errors="coerce")
    arr = err.dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "p90_abs": None}
    return {
        "n": int(arr.size),
        "mae": float(np.mean(np.abs(arr))),
        "rmse": float(np.sqrt(np.mean(np.square(arr)))),
        "bias": float(np.mean(arr)),
        "p90_abs": float(np.quantile(np.abs(arr), 0.90)),
    }


def _coverage_stats(values: pd.Series) -> dict:
    arr = pd.to_numeric(values, errors="coerce")
    n = int(len(arr))
    valid = arr.dropna()
    return {
        "rows": n,
        "valid_rows": int(valid.size),
        "no_coverage_nan_rows": int(arr.isna().sum()),
        "exact_floor_rows": int(np.isclose(valid.to_numpy(dtype=float), RSRP_NO_COVERAGE_DBM).sum()),
        "floor_dbm": RSRP_NO_COVERAGE_DBM,
        "valid_mean": float(valid.mean()) if valid.size else None,
        "valid_p10": float(valid.quantile(0.10)) if valid.size else None,
        "valid_p50": float(valid.quantile(0.50)) if valid.size else None,
        "valid_p90": float(valid.quantile(0.90)) if valid.size else None,
    }


def _plot_grid_cdf(label: str, tech: str, series_map: list[tuple[str, pd.Series]]) -> None:
    image_dir = OUT_DIR / label / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#2563eb", "#16a34a", "#f97316", "#7c3aed", "#111827"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (name, values) in enumerate(series_map):
        x, y = _cdf(values)
        ax.plot(x, y, label=f"{name} (n={len(x):,})", color=colors[idx % len(colors)], linewidth=2)
    ax.set_title(f"{label} {tech}: phase full-grid CDF")
    ax.set_xlabel("RSRP (dBm)")
    ax.set_ylabel("Cumulative %")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(image_dir / f"{tech.lower()}_phase_grid_cdf.png", dpi=160)
    plt.close(fig)


def save_snapshot(label: str) -> dict:
    snapshot_dir = OUT_DIR / label
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    summary = {"label": label, "grid": {}, "dt": {}}

    for tech in ["4g", "5g"]:
        tech_name = tech.upper()
        cdf_series = []
        summary["grid"][tech_name] = {}
        for phase, subdir, stem_pattern, value_col in PHASE_GRID_SPECS:
            df = _read_frame(PROJECT_DIR / subdir / stem_pattern.format(tech=tech))
            if df.empty or value_col not in df.columns:
                continue
            summary["grid"][tech_name][phase] = _coverage_stats(df[value_col])
            cdf_series.append((phase, df[value_col]))
        if cdf_series:
            _plot_grid_cdf(label, tech_name, cdf_series)

    for name, subdir, stem, tech_col, pred_col in DT_SPECS:
        df = _read_frame(PROJECT_DIR / subdir / stem)
        if df.empty or tech_col not in df.columns or pred_col not in df.columns:
            continue
        summary["dt"][name] = {}
        for tech in ["4G", "5G"]:
            sub = df[df[tech_col].astype(str) == tech]
            if sub.empty:
                continue
            summary["dt"][name][tech] = {
                "prediction": pred_col,
                "metrics": _metrics(sub["rsrp_measured"], sub[pred_col]),
                "coverage": _coverage_stats(sub[pred_col]),
            }

    (snapshot_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    summary = save_snapshot(args.label)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
