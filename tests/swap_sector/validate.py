"""Exploratory sensitivity of the sector-swap detectors on one built project snapshot.

Settings are NOT chosen by the best injected recall. The active model is method="pattern"; the
others are reported for comparison only. Synthetic swaps exchange whole antenna endpoints in the
configuration while every drive-test measurement stays real, so this tests recovery of crossed
configurations, not every physical effect of a feeder swap. Controls are unchanged configurations
with +-15 degree azimuth jitter, not field-verified healthy sites. One project only.

Writes data/validation_summary.csv (scorecard per configuration and site partition),
data/validation_by_pattern_quality.csv, data/validation_real_verdicts.csv and
data/validation_metadata.json.

Run from the ML/ directory (after build_dataset.py and make_synthetic_swap.py):
    venv\\Scripts\\python.exe -m tests.swap_sector.validate
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from tests.swap_sector.detect_sector_swap import DATA_DIR, add_ground_truth, detect, scorecard
from tests.swap_sector.evidence import Settings

CONFIGURATIONS = [
    ("pattern", Settings(method="pattern")),
    ("pattern_max_violation_2", Settings(method="pattern", max_violation_db=2.0)),
    ("pattern_max_violation_4", Settings(method="pattern", max_violation_db=4.0)),
    ("pattern_tolerance_1.5", Settings(method="pattern", comparison_tolerance_db=1.5)),
    ("pattern_tolerance_6", Settings(method="pattern", comparison_tolerance_db=6.0)),
    ("pattern_support_0.8", Settings(method="pattern", min_support=0.8)),
    ("combined", Settings(method="combined")),
    ("legacy", Settings(method="legacy")),
]


def partition(operator: str, site_id: str) -> str:
    """Fixed split by site, so carriers of one site never cross partitions."""
    return "held_out_sites" if int(hashlib.sha256(f"{operator}|{site_id}".encode()).hexdigest()[:8], 16) % 5 == 0 else "development_sites"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    data = args.data_dir
    cells = pd.read_csv(data / "cells.csv", dtype={"site_id": str})
    synthetic = pd.read_csv(data / "cells_synthetic.csv", dtype={"site_id": str})
    measurements = pd.read_csv(data / "measurements.csv")
    handovers = pd.read_csv(data / "handovers.csv") if (data / "handovers.csv").exists() else pd.DataFrame()

    summary, by_quality, real_counts = [], [], []
    for name, settings in CONFIGURATIONS:
        real = detect(cells, measurements, handovers, settings)
        test = add_ground_truth(detect(synthetic, measurements, handovers, settings), synthetic, real)
        test["partition"] = [partition(o, s) for o, s in zip(test["operator"], test["site_id"])]
        test.to_csv(data / f"validation_{name}.csv", index=False)
        for part, subset in [("all_sites", test)] + list(test.groupby("partition")):
            row = scorecard(subset).iloc[0].to_dict()
            row.update(configuration=name, partition=part, method=settings.method, max_violation_db=settings.max_violation_db,
                       comparison_tolerance_db=settings.comparison_tolerance_db, min_support=settings.min_support)
            summary.append(row)
        if "pattern_quality" in test:
            for quality, subset in test.groupby(test["pattern_quality"].fillna("")):
                row = scorecard(subset).iloc[0].to_dict()
                row.update(configuration=name, pattern_quality=quality or "none (not testable before pattern selection)")
                by_quality.append(row)
        real_counts.append({"configuration": name, **real["verdict"].value_counts().to_dict()})
        print(f"[validation] {name}: finished", flush=True)

    report = pd.DataFrame(summary)
    report.to_csv(data / "validation_summary.csv", index=False)
    pd.DataFrame(by_quality).to_csv(data / "validation_by_pattern_quality.csv", index=False)
    pd.DataFrame(real_counts).fillna(0).to_csv(data / "validation_real_verdicts.csv", index=False)
    metadata = {
        "project": json.loads((data / "raw_fetch_info.json").read_text()),
        "active_method": "pattern", "calibrated": False, "field_labels_available": False,
        "caveats": [
            "Synthetic crossed configurations do not model all physical swap effects",
            "Controls are not field-verified healthy sites",
            "No settings selected by these scores",
            "ASSUMED / APPROXIMATE antenna patterns cap verdicts at PROBABLE_SWAP",
            "One available project; external project validation remains outstanding",
        ],
    }
    (data / "validation_metadata.json").write_text(json.dumps(metadata, indent=2))
    pd.set_option("display.width", 250)
    print(report[report["partition"].eq("all_sites")].to_string(index=False))
    print(pd.DataFrame(by_quality).to_string(index=False))
    print(pd.DataFrame(real_counts).fillna(0).to_string(index=False))


if __name__ == "__main__":
    main()
