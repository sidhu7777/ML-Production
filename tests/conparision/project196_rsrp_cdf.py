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
from sklearn.metrics import mean_absolute_error

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
DT_RSRP_OFFSET_DB = 8.0
OUT_PATH = Path(__file__).resolve().parent / "project196_india_rsrp_cdf_dt_plus8db.png"


def fetch_baseline(engine) -> pd.DataFrame:
    query = text(
        """
        SELECT lat, lon, pred_rsrp
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
    matched["pred_rsrp"] = matched_pred["pred_rsrp"].values
    return matched


def plot_rsrp_cdf(matched: pd.DataFrame, out_path: Path):
    df = matched.copy()
    df["rsrp"] = pd.to_numeric(df["rsrp"], errors="coerce")
    df["pred_rsrp"] = pd.to_numeric(df["pred_rsrp"], errors="coerce")
    df = df.dropna(subset=["rsrp", "pred_rsrp"])
    df["rsrp_shifted"] = df["rsrp"] + DT_RSRP_OFFSET_DB

    mae = mean_absolute_error(df["rsrp_shifted"], df["pred_rsrp"])

    dt_vals = df["rsrp_shifted"].sort_values()
    pred_vals = df["pred_rsrp"].sort_values()
    y_dt = np.arange(1, len(dt_vals) + 1) / len(dt_vals) * 100
    y_pred = np.arange(1, len(pred_vals) + 1) / len(pred_vals) * 100

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.step(dt_vals.to_numpy(), y_dt, where="post", linewidth=2.2, color="#2ca02c",
             label="Drive Test")
    ax.plot(pred_vals.to_numpy(), y_pred, linewidth=2.4, color="#d62728",
            label=f"Baseline Prediction (MAE = {mae:.2f} dB)")

    ax.set_title(
        "RSRP CDF: Drive Test vs Baseline Prediction",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("RSRP (dBm)", fontsize=11)
    ax.set_ylabel("Cumulative Percentage (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[MAE] {mae:.4f} dB rows={len(df)}")
    print(f"[SAVED] {out_path}")


def main():
    engine = create_engine(os.getenv("DATABASE_URL"))
    baseline_df = fetch_baseline(engine)

    drive_df = ml_engine.fetch_drive_data(SESSION_IDS, OPERATOR, PROJECT_ID, region=REGION)
    print(f"[DT] rows={len(drive_df)}")

    matched = match_dt_to_baseline(drive_df, baseline_df)
    print(f"[MATCHED] rows={len(matched)} mean_match_distance_m={matched['match_distance_m'].mean():.2f}")

    plot_rsrp_cdf(matched, OUT_PATH)


if __name__ == "__main__":
    main()
