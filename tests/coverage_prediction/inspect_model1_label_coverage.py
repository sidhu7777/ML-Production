"""
Inspect why Model 1 has fewer supervised training rows than full polygon grids.

The full prediction/geo surfaces contain every grid cell, but supervised
training can only score rows where measured rsrp/rsrq/sinr labels exist in
coverage_rows.csv. This inspection writes a separate artifact folder and does
not modify training data or model artifacts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tests.coverage_prediction import train_model1_coverage_xgboost as base


OUTPUT_ROOT = Path("models") / "model1_label_coverage_inspection"
TARGETS = ["rsrp", "rsrq", "sinr"]
MODEL_TARGETS = ["pred_rsrp", "pred_rsrq", "pred_sinr"]
KEYS = ["grid_id", "grid_row", "grid_col", "grid_centroid_lat", "grid_centroid_lon", "time_bucket"]


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_archive_csv(member: str) -> pd.DataFrame:
    return base._read_csv_from_archive(base.COVERAGE_ARCHIVE, member)


def group_surface(df: pd.DataFrame) -> pd.DataFrame:
    return df[KEYS].drop_duplicates().copy()


def group_observed_labels(coverage_rows: pd.DataFrame) -> pd.DataFrame:
    observed = (
        coverage_rows.groupby(KEYS, as_index=False)
        .agg(
            rsrp=("rsrp", "mean"),
            rsrq=("rsrq", "mean"),
            sinr=("sinr", "mean"),
            sample_count=("grid_id", "size"),
            timestamp_min=("timestamp", "min"),
            timestamp_max=("timestamp", "max"),
        )
    )
    observed["has_any_label"] = observed[TARGETS].notna().any(axis=1)
    observed["has_all_labels"] = observed[TARGETS].notna().all(axis=1)
    return observed


def bucket_count_frame(name: str, df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("time_bucket", as_index=False)
        .agg(rows=("grid_id", "size"), unique_grids=("grid_id", "nunique"))
        .assign(source=name)
    )


def create_coverage_plot(summary_df: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = OUTPUT_ROOT / "label_coverage_by_bucket.png"
    pivot = summary_df.pivot(index="time_bucket", columns="source", values="unique_grids").fillna(0)
    ordered_cols = [
        "baseline_prediction_surface",
        "geo_surface",
        "coverage_rows_any_label",
        "coverage_rows_all_labels",
        "final_model1_training_csv",
    ]
    ordered_cols = [col for col in ordered_cols if col in pivot.columns]
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    colors = ["#2f6f73", "#7a5c22", "#5b5f97", "#a23b72", "#50514f"][: len(ordered_cols)]
    pivot.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Model 1 Label Coverage: Full Grid Surface vs Observed KPI Labels", fontsize=15, fontweight="bold")
    ax.set_xlabel("time_bucket")
    ax.set_ylabel("unique grids")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Model 1 Label Coverage Inspection",
        "",
        "The polygon grid exists in the prediction/geo surfaces, but supervised training rows require measured KPI labels from `coverage_rows.csv`.",
        "",
        "## Main Finding",
        "",
        "The row loss happens because many full-surface grids do not have observed `rsrp`, `rsrq`, and `sinr` labels in the corresponding bucket.",
        "",
        "Using baseline `pred_rsrp`, `pred_rsrq`, and `pred_sinr` as targets would train on pseudo-labels, not measured truth.",
        "",
        "## Bucket Counts",
        "",
    ]
    for row in summary["bucket_label_coverage"]:
        lines.append(
            f"- {row['time_bucket']}: full baseline grids `{row['baseline_prediction_surface_unique_grids']}`, "
            f"observed any-label grids `{row['coverage_rows_any_label_unique_grids']}`, "
            f"observed all-label grids `{row['coverage_rows_all_labels_unique_grids']}`, "
            f"final training rows `{row['final_model1_training_csv_unique_grids']}`"
        )

    lines.extend(
        [
            "",
            "## Correct Resolution",
            "",
            "- Keep supervised training/evaluation only on grids with measured labels.",
            "- Use the full baseline/geo surface for inference after training, not as ground-truth training labels.",
            "- If you need 3k grids per bucket for validation, the missing piece is more measured KPI coverage in `coverage_rows.csv`, not a different regressor.",
            "- If you deliberately want to use baseline predictions as pseudo-labels, that should be a separate semi-supervised experiment and must be reported as pseudo-label training.",
        ]
    )
    (OUTPUT_ROOT / "label_coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    baseline = read_archive_csv("baseline_prediction_grid.csv")
    geo = read_archive_csv("bucket_grid_geo_features.csv")
    coverage_rows = read_archive_csv("coverage_rows.csv")
    final_training = pd.read_csv(base.MODEL1_DATASET_CSV)

    baseline_surface = group_surface(baseline)
    geo_surface = geo[["grid_id", "grid_row", "grid_col", "time_bucket"]].drop_duplicates().copy()
    observed = group_observed_labels(coverage_rows)
    observed_any = observed[observed["has_any_label"]].copy()
    observed_all = observed[observed["has_all_labels"]].copy()

    count_frames = [
        bucket_count_frame("baseline_prediction_surface", baseline_surface),
        bucket_count_frame("geo_surface", geo_surface),
        bucket_count_frame("coverage_rows_any_label", observed_any),
        bucket_count_frame("coverage_rows_all_labels", observed_all),
        bucket_count_frame("final_model1_training_csv", final_training),
    ]
    count_df = pd.concat(count_frames, ignore_index=True)

    wide_counts = (
        count_df.pivot(index="time_bucket", columns="source", values=["rows", "unique_grids"])
        .fillna(0)
        .astype(int)
    )
    wide_counts.columns = [f"{source}_{metric}" for metric, source in wide_counts.columns]
    wide_counts = wide_counts.reset_index()

    surface_with_labels = baseline_surface.merge(
        observed[KEYS + ["has_any_label", "has_all_labels", "sample_count"]],
        on=KEYS,
        how="left",
    )
    surface_with_labels["has_any_label"] = surface_with_labels["has_any_label"].fillna(False)
    surface_with_labels["has_all_labels"] = surface_with_labels["has_all_labels"].fillna(False)
    surface_with_labels["sample_count"] = surface_with_labels["sample_count"].fillna(0).astype(int)
    surface_with_labels["label_status"] = np.select(
        [
            surface_with_labels["has_all_labels"],
            surface_with_labels["has_any_label"],
        ],
        ["all_labels_present", "partial_labels_present"],
        default="no_observed_label",
    )

    label_status_df = (
        surface_with_labels.groupby(["time_bucket", "label_status"], as_index=False)
        .agg(unique_grids=("grid_id", "nunique"), rows=("grid_id", "size"))
        .sort_values(["time_bucket", "label_status"])
    )

    unlabeled_examples = surface_with_labels[surface_with_labels["label_status"] == "no_observed_label"].head(200)

    count_df.to_csv(OUTPUT_ROOT / "bucket_counts_long.csv", index=False)
    wide_counts.to_csv(OUTPUT_ROOT / "bucket_label_coverage_summary.csv", index=False)
    label_status_df.to_csv(OUTPUT_ROOT / "label_status_by_bucket.csv", index=False)
    unlabeled_examples.to_csv(OUTPUT_ROOT / "unlabeled_grid_examples.csv", index=False)
    plot_path = create_coverage_plot(count_df)

    summary = {
        "source_archive": str(base.COVERAGE_ARCHIVE),
        "final_training_csv": str(base.MODEL1_DATASET_CSV),
        "output_root": str(OUTPUT_ROOT),
        "bucket_label_coverage": wide_counts.to_dict(orient="records"),
        "label_status_by_bucket": label_status_df.to_dict(orient="records"),
        "artifacts": {
            "bucket_counts_long": str(OUTPUT_ROOT / "bucket_counts_long.csv"),
            "bucket_label_coverage_summary": str(OUTPUT_ROOT / "bucket_label_coverage_summary.csv"),
            "label_status_by_bucket": str(OUTPUT_ROOT / "label_status_by_bucket.csv"),
            "unlabeled_grid_examples": str(OUTPUT_ROOT / "unlabeled_grid_examples.csv"),
            "coverage_plot": str(plot_path),
            "report": str(OUTPUT_ROOT / "label_coverage_report.md"),
        },
        "answer": {
            "why_3k_grid_surface_becomes_1k_training_rows": (
                "The baseline/geo surfaces contain full-grid predictions/features, but Model 1 supervised labels come from "
                "coverage_rows.csv measured rsrp/rsrq/sinr. Rows without measured labels are dropped before training."
            ),
            "can_baseline_predictions_be_used_as_labels": (
                "Only as pseudo-labels. They are model-generated predictions, not measured KPI truth, so using them as targets "
                "would validate one prediction surface against another rather than against real drive-test measurements."
            ),
        },
    }
    save_json(summary, OUTPUT_ROOT / "label_coverage_summary.json")
    write_report(summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
