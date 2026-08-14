from __future__ import annotations

import json
from pathlib import Path

import test_project196_cost231_phase9_gridanalytics_compatible as phase9


PROJECT_ID = phase9.PROJECT_ID
DATA_DIR = phase9.PROJECT_DIR / "cost231_phase10_site_technology_refresh"
COMBINED_DIR = DATA_DIR / "combined"
WORK_DIR = DATA_DIR / "work"


RENAMES = {
    f"phase9_gridanalytics_compatible_grid_project{PROJECT_ID}": f"phase10_gridanalytics_compatible_grid_project{PROJECT_ID}",
    f"phase9_directional_raw_corrected_surface_project{PROJECT_ID}": f"phase10_directional_raw_corrected_surface_project{PROJECT_ID}",
    f"phase9_directional_serving_grid_project{PROJECT_ID}": f"phase10_directional_serving_grid_project{PROJECT_ID}",
    f"phase9_dt_match_project{PROJECT_ID}": f"phase10_dt_match_project{PROJECT_ID}",
    f"phase9_offsets_project{PROJECT_ID}": f"phase10_offsets_project{PROJECT_ID}",
    f"phase9_cell_directional_coverage_summary_project{PROJECT_ID}": f"phase10_cell_directional_coverage_summary_project{PROJECT_ID}",
}


def _replace_path(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    source.replace(target)


def _rename_phase_outputs() -> None:
    for old_stem, new_stem in RENAMES.items():
        for suffix in [".csv", ".parquet", ".parquet.error.txt"]:
            _replace_path(DATA_DIR / f"{old_stem}{suffix}", DATA_DIR / f"{new_stem}{suffix}")

    _replace_path(
        COMBINED_DIR / "cdf_phase9_gridanalytics_compatible_serving_dt.png",
        COMBINED_DIR / "cdf_phase10_site_technology_refresh_serving_dt.png",
    )

    old_summary = DATA_DIR / "phase9_gridanalytics_compatible_summary.json"
    new_summary = DATA_DIR / "phase10_site_technology_refresh_summary.json"
    if old_summary.exists():
        summary = json.loads(old_summary.read_text(encoding="utf-8"))
        summary["phase_label"] = "Cost231 Phase 10 site technology refresh from DB"
        summary["input_change"] = "Project 196 site cache refreshed after setting site_prediction Technology to 4G."
        summary["outputs"] = {
            key: value.replace("phase9_", "phase10_").replace(
                "cost231_phase9_gridanalytics_compatible",
                "cost231_phase10_site_technology_refresh",
            )
            for key, value in dict(summary.get("outputs", {})).items()
        }
        summary["outputs"]["combined_dir"] = str(COMBINED_DIR.relative_to(phase9.THIS_DIR))
        new_summary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        old_summary.unlink()


def main() -> None:
    phase9.DATA_DIR = DATA_DIR
    phase9.COMBINED_DIR = COMBINED_DIR
    phase9.WORK_DIR = WORK_DIR
    phase9.main()
    _rename_phase_outputs()


if __name__ == "__main__":
    main()
