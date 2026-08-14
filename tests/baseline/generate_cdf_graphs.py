"""
Generates the 4 requested CDF images from run_project_wide_trace.py's raw
arrays (saved as .npy files, not baked into the JSON since they're large,
plain numeric arrays with nothing the dashboard needs to redraw itself):

  1. cdf_1_drive_test_measured_rsrp.png   - CDF of real DT measured RSRP, whole project
  2. cdf_2_raw_baseline_full_grid_rsrp.png - CDF of Stage 1 raw baseline predicted
                                              RSRP across the FULL predicted grid
                                              (every point, whole project)
  3. cdf_3_raw_baseline_at_drive_test_points_rsrp.png - CDF of Stage 1 raw baseline
                                              predicted RSRP restricted to ONLY the
                                              points matched to real DT locations -
                                              directly comparable to #1 (same location set)
  4. cdf_4_combined.png                    - all three overlaid on one chart

Test-case only. Run AFTER run_project_wide_trace.py has produced the
_raw_*.npy files in tests/baseline/output/cdf_graphs/.

Usage:
    python tests/baseline/generate_cdf_graphs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CDF_DIR = Path(__file__).parent / "output" / "cdf_graphs"


def cdf_xy(values: np.ndarray):
    values = np.sort(values)
    n = len(values)
    y = np.arange(1, n + 1) / n
    return values, y


def plot_single_cdf(values: np.ndarray, title: str, color: str, out_path: Path, xlabel: str = "RSRP (dBm)"):
    x, y = cdf_xy(values)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative probability")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    for p in (0.1, 0.5, 0.9):
        val = float(np.interp(p, y, x))
        ax.axhline(p, color="gray", linewidth=0.5, linestyle=":")
        ax.annotate(f"p{int(p*100)}={val:.1f}dBm", xy=(val, p), fontsize=8, color=color)
    ax.text(0.02, 0.98, f"n={len(values):,}\nmean={values.mean():.1f}dBm\nmedian={np.median(values):.1f}dBm",
             transform=ax.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[CDF] wrote {out_path} (n={len(values):,})")


def main():
    dt_measured = np.load(CDF_DIR / "_raw_dt_measured_rsrp.npy")
    baseline_full_grid = np.load(CDF_DIR / "_raw_baseline_full_grid_rsrp.npy")
    baseline_at_dt_points = np.load(CDF_DIR / "_raw_baseline_at_dt_points_rsrp.npy")

    plot_single_cdf(
        dt_measured, "CDF - Real drive-test measured RSRP (whole project)",
        "#1baf7a", CDF_DIR / "cdf_1_drive_test_measured_rsrp.png",
    )
    plot_single_cdf(
        baseline_full_grid, "CDF - Raw baseline (Stage 1 COST-231) predicted RSRP, FULL grid (whole project)",
        "#2a78d6", CDF_DIR / "cdf_2_raw_baseline_full_grid_rsrp.png",
    )
    plot_single_cdf(
        baseline_at_dt_points, "CDF - Raw baseline predicted RSRP, evaluated AT drive-test point locations only",
        "#eb6834", CDF_DIR / "cdf_3_raw_baseline_at_drive_test_points_rsrp.png",
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    for values, label, color in [
        (dt_measured, f"Drive-test measured (real, n={len(dt_measured):,})", "#1baf7a"),
        (baseline_full_grid, f"Raw baseline, full grid (n={len(baseline_full_grid):,})", "#2a78d6"),
        (baseline_at_dt_points, f"Raw baseline, at DT points only (n={len(baseline_at_dt_points):,})", "#eb6834"),
    ]:
        x, y = cdf_xy(values)
        ax.plot(x, y, label=label, color=color, linewidth=2)
    ax.set_title("Combined CDF comparison — drive test vs raw baseline (whole project 210)")
    ax.set_xlabel("RSRP (dBm)")
    ax.set_ylabel("Cumulative probability")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.annotate(
        "Curves 1 (green) and 3 (orange) cover the SAME real locations - the gap between them\n"
        "IS the raw baseline's real prediction error distribution. Curve 2 (blue) is the full\n"
        "predicted grid everywhere, not just where DT was collected.",
        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(CDF_DIR / "cdf_4_combined.png", dpi=150)
    plt.close(fig)
    print(f"[CDF] wrote {CDF_DIR / 'cdf_4_combined.png'}")


if __name__ == "__main__":
    main()
