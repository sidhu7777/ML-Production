import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.report_engine.kpi_analysis import run_kpi_analysis
from tools.report_engine.kpi_config import KPI_CONFIG
from tools.report_engine.load_data_db import load_project_data
from tools.report_engine.metadata_generator import build_metadata, write_metadata_file


def _parse_session_ids(ref_session_id: str) -> list[int]:
    return [int(s.strip()) for s in str(ref_session_id).split(",") if s.strip().isdigit()]


def _make_run_dir(base_dir: Path, project_id: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"report_engine_project_{project_id}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_summary(path: Path, payload: dict):
    lines = []
    for key, value in payload.items():
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_project_metadata_artifacts(project_id: int, out_dir: Path):
    run_dir = _make_run_dir(out_dir, project_id)
    images_dir = run_dir / "images" / "kpi_analysis"
    processed_dir = run_dir / "processed"
    images_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_df, filtered_df, project_meta = load_project_data(project_id)
    session_ids = _parse_session_ids(project_meta.get("ref_session_id", ""))

    kpi_metadata, drive_summary_metadata = run_kpi_analysis(
        filtered_df,
        user_id=None,
        kpi_config=KPI_CONFIG,
        session_ids=session_ids,
        image_dir=str(images_dir),
    )
    metadata = build_metadata(
        filtered_df,
        kpi_analysis_results=kpi_metadata,
        drive_summary_data=drive_summary_metadata,
    )

    raw_df.to_csv(run_dir / "raw_data.csv", index=False)
    filtered_df.to_csv(run_dir / "filtered_data.csv", index=False)
    _write_json(run_dir / "project_meta.json", project_meta)
    _write_json(run_dir / "kpi_metadata.json", kpi_metadata or {})
    _write_json(run_dir / "drive_summary_metadata.json", drive_summary_metadata or {})
    write_metadata_file(metadata, str(processed_dir / "report_metadata.json"))

    summary = {
        "project_id": project_id,
        "raw_rows": len(raw_df),
        "filtered_rows": len(filtered_df),
        "session_ids": session_ids,
        "kpis_in_metadata": sorted(list((metadata.get("kpi_summary") or {}).keys())),
        "output_dir": str(run_dir),
    }
    _write_summary(run_dir / "summary.txt", summary)

    preview = {
        "location": metadata.get("location"),
        "area_summary": metadata.get("area_summary"),
        "drive_summary": metadata.get("drive_summary"),
        "band_summary": metadata.get("band_summary"),
        "pci_summary": metadata.get("pci_summary"),
        "kpi_summary_keys": sorted(list((metadata.get("kpi_summary") or {}).keys())),
    }
    _write_json(run_dir / "metadata_preview.json", preview)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Build real report-engine metadata artifacts for a project.")
    parser.add_argument("--project-id", type=int, default=196)
    parser.add_argument(
        "--out-dir",
        default="tests/output/report_engine_real",
        help="Base output directory relative to ML/",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    run_dir = build_project_metadata_artifacts(args.project_id, out_dir)
    print(f"Metadata artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
