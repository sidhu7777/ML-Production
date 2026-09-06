"""
Phase 23: outdoor-only project-level calibration on top of Phase 22.

This phase intentionally does not touch indoor/O2I handling. It uses only
outdoor DT points (clear/obstructed branches) to choose bounded correction
parameters for the outdoor model, then applies the selected project
configuration to outdoor prediction candidates.

It reuses Phase 22's already-computed:
  - COST-231 + antenna baseline
  - building/obstruction branch classification
  - DEM terrain diffraction

The calibrated correction is deterministic and project-level, not per-grid
ML and not visual smoothing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase_rsrp_guard import RSRP_MAX_DBM, RSRP_NO_COVERAGE_DBM, valid_model_rsrp

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE22_DIR = PROJECT_DIR / "cost231_phase22_terrain_diffraction_comparison"
OUT_DIR = PROJECT_DIR / "cost231_phase23_outdoor_project_calibration"
IMAGE_DIR = OUT_DIR / "images"

RSRP_MIN = RSRP_NO_COVERAGE_DBM
RSRP_MAX = RSRP_MAX_DBM
MIN_TRAIN_POINTS = 20

CALIBRATION_GRID = [
    {"name": "tight", "alpha": 0.50, "clear_cap_db": 6.0, "obstructed_cap_db": 10.0},
    {"name": "balanced", "alpha": 0.75, "clear_cap_db": 8.0, "obstructed_cap_db": 14.0},
    {"name": "strong", "alpha": 1.00, "clear_cap_db": 10.0, "obstructed_cap_db": 18.0},
    {"name": "wide_obstruction", "alpha": 1.00, "clear_cap_db": 12.0, "obstructed_cap_db": 28.0},
    {"name": "phase19_like", "alpha": 1.00, "clear_cap_db": 16.0, "obstructed_cap_db": 40.0},
]


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _read_frame(stem: Path) -> pd.DataFrame:
    if stem.with_suffix(".parquet").exists():
        return pd.read_parquet(stem.with_suffix(".parquet"))
    if stem.with_suffix(".csv").exists():
        return pd.read_csv(stem.with_suffix(".csv"), low_memory=False)
    raise FileNotFoundError(f"No parquet/csv found for {stem}")


def _split_dt(dt: pd.DataFrame) -> pd.DataFrame:
    out = dt.copy()
    key = out.get("dt_row_id", out.get("id", pd.Series(out.index, index=out.index))).astype(str)
    hashed = pd.util.hash_pandas_object(key, index=False).astype("uint64")
    out["phase23_split"] = np.where((hashed % 10) < 7, "train", "validation")
    return out


def _outdoor_mask(df: pd.DataFrame) -> pd.Series:
    return df["obstruction_branch"].astype(str).isin(["clear", "obstructed"])


def _phase22_bias_table() -> pd.DataFrame:
    path = PHASE22_DIR / "phase22_phase19_style_bias_by_condition.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.read_csv(PHASE22_DIR / "phase22_no_terrain_bias_by_condition.csv")


def _attach_phase22_bias(dt: pd.DataFrame) -> pd.DataFrame:
    bias = _phase22_bias_table()[["technology", "clutter_class", "obstruction_branch", "bias_db"]]
    out = dt.merge(
        bias.rename(columns={"bias_db": "phase22_phase19_bias_db"}),
        on=["technology", "clutter_class", "obstruction_branch"],
        how="left",
    )
    out["phase22_phase19_bias_db"] = pd.to_numeric(out["phase22_phase19_bias_db"], errors="coerce").fillna(0.0)
    out["phase22_same_bias_with_terrain_rsrp_unclipped"] = (
        out["phase22_physical_with_terrain_rsrp"] + out["phase22_phase19_bias_db"]
    )
    out["phase22_same_bias_with_terrain_rsrp"] = valid_model_rsrp(
        out["phase22_same_bias_with_terrain_rsrp_unclipped"]
    )
    return out


def _calibration_table(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    grouped = train.groupby(["technology", "clutter_class", "obstruction_branch"], dropna=False)
    for (tech, clutter, branch), group in grouped:
        residual = group["rsrp_measured"] - group["phase22_physical_with_terrain_rsrp"]
        raw = float(residual.median())
        cap = float(config["obstructed_cap_db"] if branch == "obstructed" else config["clear_cap_db"])
        correction = float(np.clip(raw * float(config["alpha"]), -cap, cap))
        rows.append(
            {
                "technology": tech,
                "clutter_class": clutter,
                "obstruction_branch": branch,
                "n_train": int(len(group)),
                "raw_median_residual_db": raw,
                "phase23_outdoor_correction_db": correction,
                "alpha": float(config["alpha"]),
                "cap_db": cap,
                "config_name": config["name"],
            }
        )
    table = pd.DataFrame(rows)
    table = table[table["n_train"] >= MIN_TRAIN_POINTS].copy()
    return table.sort_values(["technology", "obstruction_branch", "clutter_class"])


def _attach_phase23_correction(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    correction_cols = [
        "technology",
        "clutter_class",
        "obstruction_branch",
        "phase23_outdoor_correction_db",
        "n_train",
        "config_name",
    ]
    out = df.merge(table[correction_cols], on=["technology", "clutter_class", "obstruction_branch"], how="left")
    outdoor = _outdoor_mask(out)
    out["phase23_outdoor_correction_db"] = pd.to_numeric(
        out["phase23_outdoor_correction_db"], errors="coerce"
    ).fillna(0.0)
    out.loc[~outdoor, "phase23_outdoor_correction_db"] = 0.0
    out["phase23_calibration_applied"] = outdoor & out["n_train"].notna()
    return out


def _score_validation(dt: pd.DataFrame, table: pd.DataFrame) -> dict:
    validation = dt[(dt["phase23_split"] == "validation") & _outdoor_mask(dt)].copy()
    scored = _attach_phase23_correction(validation, table)
    scored["phase23_rsrp_unclipped"] = (
        scored["phase22_physical_with_terrain_rsrp"] + scored["phase23_outdoor_correction_db"]
    )
    scored["phase23_rsrp"] = valid_model_rsrp(scored["phase23_rsrp_unclipped"])
    scored["phase22_error_db"] = scored["rsrp_measured"] - scored["phase22_same_bias_with_terrain_rsrp"]
    scored["phase23_error_db"] = scored["rsrp_measured"] - scored["phase23_rsrp"]

    def _metrics(error: pd.Series) -> dict:
        arr = pd.to_numeric(error, errors="coerce").dropna().to_numpy(dtype=float)
        if arr.size == 0:
            return {"mae": math.nan, "rmse": math.nan, "bias": math.nan, "p90_abs": math.nan}
        return {
            "mae": float(np.mean(np.abs(arr))),
            "rmse": float(np.sqrt(np.mean(np.square(arr)))),
            "bias": float(np.mean(arr)),
            "p90_abs": float(np.quantile(np.abs(arr), 0.90)),
        }

    by_tech = {}
    for tech, group in scored.groupby("technology"):
        by_tech[str(tech)] = {
            "n_validation": int(len(group)),
            "phase22": _metrics(group["phase22_error_db"]),
            "phase23": _metrics(group["phase23_error_db"]),
        }
    all_score = _metrics(scored["phase23_error_db"])
    return {
        "n_validation": int(len(scored)),
        "score": float(all_score["mae"] + 0.25 * abs(all_score["bias"])),
        "phase23": all_score,
        "phase22": _metrics(scored["phase22_error_db"]),
        "by_technology": by_tech,
    }


def _select_project_config(dt: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    train = dt[(dt["phase23_split"] == "train") & _outdoor_mask(dt)].copy()
    trial_rows = []
    tables = {}
    for config in CALIBRATION_GRID:
        table = _calibration_table(train, config)
        score = _score_validation(dt, table)
        tables[config["name"]] = table
        trial_rows.append(
            {
                "technology": "ALL",
                "config_name": config["name"],
                "alpha": config["alpha"],
                "clear_cap_db": config["clear_cap_db"],
                "obstructed_cap_db": config["obstructed_cap_db"],
                "n_validation": score["n_validation"],
                "score": score["score"],
                "phase22_mae": score["phase22"]["mae"],
                "phase23_mae": score["phase23"]["mae"],
                "phase22_bias": score["phase22"]["bias"],
                "phase23_bias": score["phase23"]["bias"],
                "phase23_rmse": score["phase23"]["rmse"],
                "phase23_p90_abs": score["phase23"]["p90_abs"],
            }
        )
        for tech, tech_score in score["by_technology"].items():
            phase23 = tech_score["phase23"]
            phase22 = tech_score["phase22"]
            trial_rows.append(
                {
                    "technology": tech,
                    "config_name": config["name"],
                    "alpha": config["alpha"],
                    "clear_cap_db": config["clear_cap_db"],
                    "obstructed_cap_db": config["obstructed_cap_db"],
                    "n_validation": tech_score["n_validation"],
                    "score": float(phase23["mae"] + 0.25 * abs(phase23["bias"])),
                    "phase22_mae": phase22["mae"],
                    "phase23_mae": phase23["mae"],
                    "phase22_bias": phase22["bias"],
                    "phase23_bias": phase23["bias"],
                    "phase23_rmse": phase23["rmse"],
                    "phase23_p90_abs": phase23["p90_abs"],
                }
            )

    trials_df = pd.DataFrame(trial_rows).sort_values(["technology", "score"])
    selected_configs = {}
    selected_tables = []
    for tech in sorted(trials_df.loc[trials_df["technology"] != "ALL", "technology"].unique()):
        tech_trials = trials_df[trials_df["technology"] == tech].sort_values("score")
        best_name = str(tech_trials.iloc[0]["config_name"])
        best_config = next(config for config in CALIBRATION_GRID if config["name"] == best_name)
        selected_configs[tech] = best_config
        selected_tables.append(tables[best_name][tables[best_name]["technology"].astype(str) == tech])

    selected_table = pd.concat(selected_tables, ignore_index=True) if selected_tables else pd.DataFrame()
    return selected_configs, selected_table, trials_df


def _aggregate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    outdoor = _outdoor_mask(candidates)
    out = _attach_phase23_correction(candidates, candidates.attrs["phase23_table"])
    out["phase23_candidate_rsrp"] = out["phase22_with_terrain_calibrated_rsrp"]
    out.loc[outdoor, "phase23_candidate_rsrp_unclipped"] = (
        out.loc[outdoor, "phase22_physical_with_terrain_rsrp"]
        + out.loc[outdoor, "phase23_outdoor_correction_db"]
    )
    out.loc[outdoor, "phase23_candidate_rsrp"] = valid_model_rsrp(
        out.loc[outdoor, "phase23_candidate_rsrp_unclipped"]
    )
    out["phase23_outdoor_model_used"] = outdoor

    agg = (
        out.groupby(["technology", "grid_id"], dropna=False)
        .agg(
            phase23_best_rsrp=("phase23_candidate_rsrp", "max"),
            phase23_mean_rsrp=("phase23_candidate_rsrp", "mean"),
            phase23_outdoor_correction_db_mean=("phase23_outdoor_correction_db", "mean"),
            phase23_outdoor_calibrated_share=("phase23_calibration_applied", "mean"),
        )
        .reset_index()
    )
    grid = _read_frame(PROJECT_DIR / "cost231_phase9_gridanalytics_compatible" / "phase9_gridanalytics_compatible_grid_project210")
    all_tech_grid = pd.concat(
        [grid[["grid_id"]].assign(technology=technology) for technology in ["4G", "5G"]],
        ignore_index=True,
    )
    return all_tech_grid.merge(agg, on=["technology", "grid_id"], how="left"), out


def _cdf_values(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    if arr.size == 0:
        return arr, arr
    return arr, np.arange(1, arr.size + 1, dtype=float) / arr.size * 100.0


def _plot_cdf(series_map: list[tuple[str, pd.Series, str]], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values, color in series_map:
        x, y = _cdf_values(values)
        ax.plot(x, y, label=f"{label} (n={len(x):,})", color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("RSRP / error (dB)")
    ax.set_ylabel("Cumulative %")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_frame(df: pd.DataFrame, stem: Path) -> None:
    df.to_parquet(stem.with_suffix(".parquet"), index=False)
    df.to_csv(stem.with_suffix(".csv"), index=False)


def main() -> None:
    _ensure_dirs()
    candidates = _read_frame(PHASE22_DIR / "phase22_scored_candidates_project210")
    dt = _read_frame(PHASE22_DIR / "phase22_dt_terrain_scored_project210")
    dt["technology"] = dt["assigned_technology"].astype(str)
    dt = _attach_phase22_bias(_split_dt(dt))

    selected_configs, table, trials = _select_project_config(dt)
    table.to_csv(OUT_DIR / "phase23_selected_outdoor_calibration_table.csv", index=False)
    trials.to_csv(OUT_DIR / "phase23_calibration_trials.csv", index=False)

    candidates.attrs["phase23_table"] = table
    grid_agg, scored_candidates = _aggregate_candidates(candidates)
    _save_frame(scored_candidates, OUT_DIR / "phase23_scored_candidates_project210")

    summary = {
        "selected_config_by_technology": selected_configs,
        "calibration_scope": "outdoor_only_clear_obstructed",
        "indoor_policy": "unchanged_from_phase22_not_calibrated_from_outdoor_dt",
        "trial_table": trials.to_dict("records"),
        "validation": _score_validation(dt, table),
        "technology": {},
    }

    validation = dt[(dt["phase23_split"] == "validation") & _outdoor_mask(dt)].copy()
    validation = _attach_phase23_correction(validation, table)
    validation["phase23_rsrp_unclipped"] = (
        validation["phase22_physical_with_terrain_rsrp"] + validation["phase23_outdoor_correction_db"]
    )
    validation["phase23_rsrp"] = valid_model_rsrp(validation["phase23_rsrp_unclipped"])
    validation["phase22_error_db"] = validation["rsrp_measured"] - validation["phase22_same_bias_with_terrain_rsrp"]
    validation["phase23_error_db"] = validation["rsrp_measured"] - validation["phase23_rsrp"]
    _save_frame(validation, OUT_DIR / "phase23_validation_dt_project210")

    for tech in ["4G", "5G"]:
        phase22 = _read_frame(PHASE22_DIR / f"phase22_serving_grid_{tech.lower()}_project210")
        serving = grid_agg[grid_agg["technology"].astype(str) == tech].merge(
            phase22[
                [
                    "grid_id",
                    "center_lat",
                    "center_lon",
                    "min_lat",
                    "max_lat",
                    "min_lon",
                    "max_lon",
                    "phase22_with_terrain_best_rsrp",
                    "phase22_with_terrain_mean_rsrp",
                    "phase22_physical_with_terrain_best_rsrp",
                    "terrain_diffraction_loss_db_mean",
                ]
            ],
            on="grid_id",
            how="left",
        )
        _save_frame(serving, OUT_DIR / f"phase23_serving_grid_{tech.lower()}_project210")

        vtech = validation[validation["technology"] == tech].copy()
        _plot_cdf(
            [
                ("Phase22 same-bias + terrain", serving["phase22_with_terrain_best_rsrp"], "#2563eb"),
                ("Phase23 outdoor calibrated", serving["phase23_best_rsrp"], "#16a34a"),
                ("Phase22 physical + terrain", serving["phase22_physical_with_terrain_best_rsrp"], "#ef4444"),
            ],
            f"Project 210 {tech}: Phase 22 vs Phase 23 full grid",
            IMAGE_DIR / f"phase23_{tech.lower()}_full_grid_cdf.png",
        )
        _plot_cdf(
            [
                ("DT measured", vtech["rsrp_measured"], "#111827"),
                ("Phase22 validation", vtech["phase22_same_bias_with_terrain_rsrp"], "#2563eb"),
                ("Phase23 validation", vtech["phase23_rsrp"], "#16a34a"),
            ],
            f"Project 210 {tech}: outdoor validation DT CDF",
            IMAGE_DIR / f"phase23_{tech.lower()}_validation_dt_cdf.png",
        )
        _plot_cdf(
            [
                ("Phase22 abs error", vtech["phase22_error_db"].abs(), "#2563eb"),
                ("Phase23 abs error", vtech["phase23_error_db"].abs(), "#16a34a"),
            ],
            f"Project 210 {tech}: outdoor validation absolute error",
            IMAGE_DIR / f"phase23_{tech.lower()}_validation_abs_error_cdf.png",
        )

        summary["technology"][tech] = {
            "grid_rows": int(len(serving)),
            "validation_rows": int(len(vtech)),
            "mean_phase22_best_rsrp": float(serving["phase22_with_terrain_best_rsrp"].mean()),
            "mean_phase23_best_rsrp": float(serving["phase23_best_rsrp"].mean()),
            "mean_phase23_vs_phase22_best_delta_db": float(
                (serving["phase23_best_rsrp"] - serving["phase22_with_terrain_best_rsrp"]).mean()
            ),
            "mean_phase22_error_db": float(vtech["phase22_error_db"].mean()) if len(vtech) else math.nan,
            "mean_phase23_error_db": float(vtech["phase23_error_db"].mean()) if len(vtech) else math.nan,
            "phase22_mae_db": float(vtech["phase22_error_db"].abs().mean()) if len(vtech) else math.nan,
            "phase23_mae_db": float(vtech["phase23_error_db"].abs().mean()) if len(vtech) else math.nan,
        }
        print(f"[PHASE23] wrote {tech} serving grid rows={len(serving)}")

    (OUT_DIR / "phase23_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("[PHASE23] selected config by technology:")
    print(json.dumps(selected_configs, indent=2))
    print("[PHASE23] summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
