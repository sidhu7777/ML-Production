from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.neighbors import BallTree

ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
sys.path.insert(0, str(ML_ROOT))
load_dotenv(ML_ROOT / ".env")
for bridge_env in ["PYTHON_BRIDGE_BASE_URL", "SIGNAL_TRACKERS_BRIDGE_URL"]:
    os.environ[bridge_env] = ""

from tools.lte_prediction import ml_engine  # noqa: E402

PROJECT_ID = 196
REGION = "india"
OPERATOR = "Airtel"
SESSION_IDS = [4187, 4178, 4180]
JOB_ID = "562b80d1-d166-4fb1-8b13-f5c52af8f4d3"
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / "project196_india_baseline_cdf.png"

KPI_SPECS = [
    ("RSRP", "pred_rsrp", "RSRP (dBm)"),
    ("RSRQ", "pred_rsrq", "RSRQ (dB)"),
    ("SINR", "pred_sinr", "SINR (dB)"),
]


def fetch_baseline(engine) -> pd.DataFrame:
    query = text(
        """
        SELECT lat, lon, pred_rsrp, pred_rsrq, pred_sinr
        FROM lte_prediction_baseline_results
        WHERE project_id = :project_id AND job_id = :job_id
        """
    )
    df = pd.read_sql(query, engine, params={"project_id": PROJECT_ID, "job_id": JOB_ID})
    print(f"[BASELINE] rows={len(df)}")
    return df


def match_dt_to_baseline(dt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    dt = dt_df.dropna(subset=["lat", "lon"]).copy()
    preds = pred_df.dropna(subset=["lat", "lon"]).copy()
    dt["lat"] = pd.to_numeric(dt["lat"], errors="coerce")
    dt["lon"] = pd.to_numeric(dt["lon"], errors="coerce")
    preds["lat"] = pd.to_numeric(preds["lat"], errors="coerce")
    preds["lon"] = pd.to_numeric(preds["lon"], errors="coerce")
    dt = dt.dropna(subset=["lat", "lon"]).copy()
    preds = preds.dropna(subset=["lat", "lon"]).copy()

    pred_rad = np.radians(preds[["lat", "lon"]].to_numpy(dtype=float))
    dt_rad = np.radians(dt[["lat", "lon"]].to_numpy(dtype=float))
    tree = BallTree(pred_rad, metric="haversine")
    dist_rad, indices = tree.query(dt_rad, k=1)
    earth_radius_m = 6371000.0

    matched_pred = preds.iloc[indices[:, 0]].reset_index(drop=True)
    matched = dt.reset_index(drop=True).copy()
    matched["match_distance_m"] = dist_rad[:, 0] * earth_radius_m
    for col in ["pred_rsrp", "pred_rsrq", "pred_sinr"]:
        matched[col] = matched_pred[col].values
    return matched


def plot_baseline_cdf_row(matched: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    color = "#d62728"

    for ax, (kpi_name, col, xlabel) in zip(axes, KPI_SPECS):
        values = pd.to_numeric(matched[col], errors="coerce").dropna().sort_values()
        y = np.arange(1, len(values) + 1) / len(values) * 100
        ax.plot(values.to_numpy(), y, linewidth=2.2, color=color, label="Mixed Prediction")
        ax.set_title(f"India Prediction - {kpi_name} CDF", fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative Percentage (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left")
        print(f"[{kpi_name}] rows={len(values)} min={values.min():.2f} max={values.max():.2f}")

    fig.suptitle(f"Project {PROJECT_ID} (India) - Baseline CDF at Drive-Test Locations", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def main():
    engine = create_engine(os.getenv("DATABASE_URL"))
    baseline_df = fetch_baseline(engine)

    drive_df = ml_engine.fetch_drive_data(SESSION_IDS, OPERATOR, PROJECT_ID, region=REGION)
    print(f"[DT] rows={len(drive_df)}")

    matched = match_dt_to_baseline(drive_df, baseline_df)
    print(f"[MATCHED] rows={len(matched)} mean_match_distance_m={matched['match_distance_m'].mean():.2f}")

    plot_baseline_cdf_row(matched, OUT_PATH)


if __name__ == "__main__":
    main()
