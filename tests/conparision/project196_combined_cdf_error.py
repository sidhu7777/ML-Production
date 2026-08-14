from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
OUT_PATH = Path(__file__).resolve().parent / "project196_india_combined_cdf_error.png"

KPI_SPECS = [
    ("RSRP", "rsrp", "pred_rsrp", "RSRP (dBm)"),
    ("RSRQ", "rsrq", "pred_rsrq", "RSRQ (dB)"),
    ("SINR", "sinr", "pred_sinr", "SINR (dB)"),
]

RSRP_BIN_EDGES = [-200, -105, -95, -85, -75, 20]
RSRP_BIN_LABELS = ["< -105", "-105 to -95", "-95 to -85", "-85 to -75", "-75 to 0"]


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


def build_rsrp_error_table(matched: pd.DataFrame) -> pd.DataFrame:
    df = matched.copy()
    df["rsrp"] = pd.to_numeric(df["rsrp"], errors="coerce")
    df["pred_rsrp"] = pd.to_numeric(df["pred_rsrp"], errors="coerce")
    df = df.dropna(subset=["rsrp", "pred_rsrp"])
    df["bias"] = df["pred_rsrp"] - df["rsrp"]
    df["abs_error"] = df["bias"].abs()
    df["bin"] = pd.cut(df["rsrp"], bins=RSRP_BIN_EDGES, labels=RSRP_BIN_LABELS, right=True)

    rows = []
    for label in RSRP_BIN_LABELS:
        sub = df.loc[df["bin"] == label]
        if sub.empty:
            rows.append({"RSRP Range (dBm)": label, "DT Points": 0, "Mean Abs Error (dB)": "n/a", "Bias (dB)": "n/a"})
            continue
        rows.append({
            "RSRP Range (dBm)": label,
            "DT Points": len(sub),
            "Mean Abs Error (dB)": round(float(sub["abs_error"].mean()), 2),
            "Bias (dB)": round(float(sub["bias"].mean()), 2),
        })
    return pd.DataFrame(rows[::-1])


def plot_combined(matched: pd.DataFrame, error_table: pd.DataFrame, out_path: Path):
    fig = plt.figure(figsize=(17, 9.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.1, 1.2], hspace=0.42, wspace=0.28)

    dt_color = "#2ca02c"
    pred_color = "#d62728"

    for idx, (kpi_name, dt_col, pred_col, xlabel) in enumerate(KPI_SPECS):
        ax = fig.add_subplot(gs[0, idx])
        dt_vals = pd.to_numeric(matched[dt_col], errors="coerce").dropna().sort_values()
        pred_vals = pd.to_numeric(matched[pred_col], errors="coerce").dropna().sort_values()
        y_dt = np.arange(1, len(dt_vals) + 1) / len(dt_vals) * 100
        y_pred = np.arange(1, len(pred_vals) + 1) / len(pred_vals) * 100

        ax.step(dt_vals.to_numpy(), y_dt, where="post", linewidth=2.0, color=dt_color, label="Drive Test")
        ax.plot(pred_vals.to_numpy(), y_pred, linewidth=2.2, color=pred_color, label="Baseline Prediction")
        ax.set_title(f"India Project 196 - {kpi_name} CDF", fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative Percentage (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", fontsize=9)

    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    ax_table.set_title(
        "RSRP Error by Signal-Strength Range (Baseline vs Drive Test, at matched DT locations)",
        fontsize=12, fontweight="bold", pad=14,
    )
    table = ax_table.table(
        cellText=error_table.values,
        colLabels=error_table.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.1)
    for i in range(len(error_table.columns)):
        cell = table[(0, i)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(weight="bold", color="white")

    fig.suptitle("Project 196 (India) - DT vs Baseline CDF Comparison + RSRP Error Breakdown", fontsize=14)
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

    error_table = build_rsrp_error_table(matched)
    print(error_table.to_string(index=False))

    plot_combined(matched, error_table, OUT_PATH)


if __name__ == "__main__":
    main()
