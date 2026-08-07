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

PROJECT_ID = 210
REGION = "taiwan"
OPERATOR = "遠傳電信"
SESSION_IDS = [4479, 4478, 4477, 4476, 4475, 4474, 4473, 4472, 4471, 4470, 4469, 4468, 4467, 4466, 4465, 4464,
               4463, 4462, 4461, 4460, 4459, 4458, 4457, 4456, 4455, 4454, 4453, 4452, 4451, 4450, 4449, 4448,
               4447, 4446, 4445, 4444, 4443, 4442, 4441, 4440, 4439, 4438, 4437, 4436, 4435, 4434, 4433, 4432,
               4431, 4430, 4429, 4428, 4427, 4426, 4425, 4424, 4423, 4422, 4421, 4420, 4419, 4418, 4417, 4416,
               4415, 4414, 4413, 4412, 4411, 4410, 4409, 4408, 4407, 4406, 4405, 4404, 4403, 4402, 4401, 4400,
               4399, 4398, 4397, 4396, 4395, 4394, 4393, 4392, 4391, 4390, 4389, 4388, 4387, 4386, 4385, 4384,
               4382, 4381, 4379, 4378, 4377, 4376, 4375, 4374, 4373, 4372, 4371, 4370, 4364, 4363, 4362, 4361,
               4360, 4359, 4358, 4357, 4356, 4352, 4351, 4350, 4348, 4347, 4346, 4345, 4344, 4343, 4322, 4318,
               4317, 4316, 4315, 4314, 4313, 4312, 4292, 4291, 4290, 4289, 4288, 4287, 4286, 4285, 4284, 4283,
               4282, 4281, 4280, 4254, 4253, 4252, 4247, 4246, 4245, 4244, 4243, 4242, 4241, 4240, 4239, 4238,
               4237, 4236, 4235, 4234, 4233, 4190, 4189, 4188, 4187, 4184, 4183, 4182, 4180, 4179, 4178, 4177,
               4175, 4174, 4173, 4171, 4170, 4163, 4159, 4157, 4156, 4155, 4154, 4153, 4152, 4151, 4150, 4149,
               4148, 4142, 4141, 4140, 4139, 4138, 4137, 4136, 4135, 4134, 4133, 4131, 4103, 4102, 4096, 4095,
               4094, 4093, 4092, 4091, 4090, 4089, 4088, 4087, 4086, 4085, 4084, 4083, 4082, 4081, 4080, 4079,
               4078, 4077, 4076, 4075, 4074, 4073, 4072, 4071, 4070, 4069, 4068, 4067, 4066, 4065, 4064, 4063,
               4062, 4061, 4060, 4059, 4058, 4057, 4056, 4055, 4054, 4052, 4051, 4050, 4049, 4048, 4045, 4044,
               4043, 4042, 4041, 4040, 3986, 3985, 3984, 3983, 3980, 3979]
JOB_ID = "e1566664-1705-48f9-8841-0e7af61d8878"
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / "project210_taiwan_baseline_cdf.png"

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
    color = "#1f77b4"

    for ax, (kpi_name, col, xlabel) in zip(axes, KPI_SPECS):
        values = pd.to_numeric(matched[col], errors="coerce").dropna().sort_values()
        y = np.arange(1, len(values) + 1) / len(values) * 100
        ax.plot(values.to_numpy(), y, linewidth=2.2, color=color, label="Mixed Prediction")
        ax.set_title(f"Taiwan Prediction - {kpi_name} CDF", fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative Percentage (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left")
        print(f"[{kpi_name}] rows={len(values)} min={values.min():.2f} max={values.max():.2f}")

    fig.suptitle(f"Project {PROJECT_ID} (Taiwan) - Baseline CDF at Drive-Test Locations", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def main():
    engine = create_engine(os.getenv("DATABASE_URL_Taiwan"))
    baseline_df = fetch_baseline(engine)

    drive_df = ml_engine.fetch_drive_data(SESSION_IDS, OPERATOR, PROJECT_ID, region=REGION)
    print(f"[DT] rows={len(drive_df)}")

    matched = match_dt_to_baseline(drive_df, baseline_df)
    print(f"[MATCHED] rows={len(matched)} mean_match_distance_m={matched['match_distance_m'].mean():.2f}")

    plot_baseline_cdf_row(matched, OUT_PATH)


if __name__ == "__main__":
    main()
