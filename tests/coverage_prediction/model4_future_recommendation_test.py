from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tests.coverage_prediction import model3_business_rule_recommendation_test as future_rules


DEFAULT_OUTPUT_ROOT = ML_ROOT / "tests" / "output" / "model4_future_recommendation"
DEFAULT_STABLE_OUTPUT_DIR = ML_ROOT / "models" / "model4_future_recommendation_experiment"
DEFAULT_EXISTING_FUTURE_OUTPUT_DIR = ML_ROOT / "models" / "model3_hybrid_load_balancing_experiment"

def _promote_existing_future_outputs(stable_dir: Path) -> bool:
    rename_pairs = [
        ("model3_business_rule_recommendations.xlsx", "model4_future_recommendations.xlsx"),
        ("model3_recommendations.csv", "model4_future_recommendations.csv"),
        ("model3_business_rule_recommendation_summary.json", "model4_future_recommendation_summary.json"),
        ("model3_business_rule_recommendation.log", "model4_future_recommendation.log"),
    ]
    copied_any = False
    for src_name, dest_name in rename_pairs:
        src = DEFAULT_EXISTING_FUTURE_OUTPUT_DIR / src_name
        if src.exists():
            shutil.copy2(src, stable_dir / dest_name)
            copied_any = True
    return copied_any


def run_model4_future_recommendation(config: future_rules.Model3RecommendationConfig, *, reuse_existing: bool = True) -> Path | None:
    stable_dir = config.stable_output_dir
    stable_dir.mkdir(parents=True, exist_ok=True)
    run_dir: Path | None = None

    if not reuse_existing or not _promote_existing_future_outputs(stable_dir):
        run_dir = future_rules.run_model3_business_rule_recommendation_test(config)

        rename_pairs = [
            ("model3_business_rule_recommendations.xlsx", "model4_future_recommendations.xlsx"),
            ("model3_recommendations.csv", "model4_future_recommendations.csv"),
            ("summary.json", "model4_future_recommendation_summary.json"),
            ("log.txt", "model4_future_recommendation.log"),
        ]
        for src_name, dest_name in rename_pairs:
            src = run_dir / src_name
            if src.exists():
                try:
                    shutil.copy2(src, stable_dir / dest_name)
                except PermissionError:
                    pass
    payload = {
        "mode": "future_model4_recommendation",
        "run_dir": str(run_dir) if run_dir is not None else None,
        "stable_output_dir": str(stable_dir),
        "reuse_existing": bool(reuse_existing),
        "files": {
            "workbook": str(stable_dir / "model4_future_recommendations.xlsx"),
            "recommendations_csv": str(stable_dir / "model4_future_recommendations.csv"),
            "summary_json": str(stable_dir / "model4_future_recommendation_summary.json"),
            "log": str(stable_dir / "model4_future_recommendation.log"),
        },
    }
    print(json.dumps(payload, indent=2, default=str))
    return run_dir


def parse_args() -> future_rules.Model3RecommendationConfig:
    parser = argparse.ArgumentParser(description="Run future-state Model 4 recommendations using Model 1 + Model 2 outputs.")
    parser.add_argument("--dataset-path", type=Path, default=future_rules.DEFAULT_MODEL3_DATASET)
    parser.add_argument("--summary-path", type=Path, default=future_rules.DEFAULT_MODEL3_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stable-output-dir", type=Path, default=DEFAULT_STABLE_OUTPUT_DIR)
    parser.add_argument("--congestion-threshold", type=float, default=future_rules.DEFAULT_CONGESTION_THRESHOLD)
    parser.add_argument("--rrc-sector-capacity", type=float, default=future_rules.DEFAULT_RRC_SECTOR_CAPACITY)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    config = future_rules.Model3RecommendationConfig(
        dataset_path=args.dataset_path,
        summary_path=args.summary_path,
        output_root=args.output_root,
        stable_output_dir=args.stable_output_dir,
        congestion_threshold=args.congestion_threshold,
        rrc_sector_capacity=args.rrc_sector_capacity,
    )
    config.force_rerun = bool(args.force_rerun)
    return config


if __name__ == "__main__":
    cfg = parse_args()
    run_model4_future_recommendation(cfg, reuse_existing=not getattr(cfg, "force_rerun", False))
