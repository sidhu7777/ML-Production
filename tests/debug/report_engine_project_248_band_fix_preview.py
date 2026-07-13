import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.report_engine import load_data_db as report_load
from tools.report_engine.kpi_analysis import build_native_table_data, run_kpi_analysis
from tools.report_engine.kpi_config import KPI_CONFIG
from tools.report_engine.load_data_db import filter_known_band_rows
from tools.report_engine.map_generator import normalize_band_name
from tools.report_engine.metadata_generator import build_metadata, write_metadata_file
from tools.report_engine.pdf_generator import generate_pdf_report


def _make_run_dir(base_dir: Path, project_id: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"report_engine_project_{project_id}_band_fix_preview_{ts}"
    (run_dir / "images" / "kpi_analysis").mkdir(parents=True, exist_ok=True)
    (run_dir / "processed").mkdir(parents=True, exist_ok=True)
    (run_dir / "report").mkdir(parents=True, exist_ok=True)
    return run_dir


def _band_summary(df):
    if "band" not in df.columns:
        return {}
    counts = df["band"].apply(normalize_band_name).value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def _patched_filter_primary_rows(df):
    if df.empty:
        return df

    masks = []
    if "primary_cell_info_1" in df.columns:
        primary_info = df["primary_cell_info_1"].fillna("").astype(str)
        masks.append(primary_info.str.contains("mRegistered=YES", case=False, na=False))
    if "primary" in df.columns:
        primary_values = df["primary"].fillna("").astype(str).str.strip().str.lower()
        masks.append(primary_values.isin({"yes", "y", "true", "1"}))
    if not masks:
        return df

    combined = masks[0].copy()
    for mask in masks[1:]:
        combined = combined | mask
    return df.loc[combined].reset_index(drop=True)


@contextmanager
def patched_primary_filter():
    original = report_load._filter_primary_rows
    report_load._filter_primary_rows = _patched_filter_primary_rows
    try:
        yield
    finally:
        report_load._filter_primary_rows = original


def _parse_session_ids(ref_session_id: str) -> list[int]:
    return [int(s.strip()) for s in str(ref_session_id).split(",") if s.strip().isdigit()]


def build_preview(project_id: int, user_id: int | None, out_dir: Path) -> Path:
    run_dir = _make_run_dir(out_dir, project_id)
    processed_dir = run_dir / "processed"
    images_dir = run_dir / "images"

    raw_df, original_filtered_df, project_meta = report_load.load_project_data(project_id)
    original_report_df = filter_known_band_rows(original_filtered_df)

    with patched_primary_filter():
        _, patched_filtered_df, project_meta = report_load.load_project_data(project_id)
        patched_report_df = filter_known_band_rows(patched_filtered_df)
        session_ids = _parse_session_ids(project_meta.get("ref_session_id", ""))

        kpi_metadata, drive_summary_metadata = run_kpi_analysis(
            patched_report_df,
            user_id=user_id,
            kpi_config=KPI_CONFIG,
            session_ids=session_ids,
            image_dir=str(images_dir / "kpi_analysis"),
        )
        metadata = build_metadata(
            patched_report_df,
            kpi_analysis_results=kpi_metadata,
            drive_summary_data=drive_summary_metadata,
        )
        write_metadata_file(metadata, str(processed_dir / "report_metadata.json"))

        report_text = {
            "Executive Summary": "Test-only preview generated with patched primary filter.",
            "Map View - Band": (
                "Band distribution preview after accepting both "
                "`primary_cell_info_1=mRegistered=YES` and `primary=Yes` as primary evidence."
            ),
        }
        (processed_dir / "report_text.json").write_text(
            json.dumps(report_text, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        native_tables = build_native_table_data(
            patched_report_df,
            metadata=metadata,
            band_summary=metadata.get("band_summary"),
            drive_summary=drive_summary_metadata,
        )

        pdf_path = generate_pdf_report(
            metadata_path=str(processed_dir / "report_metadata.json"),
            report_text_path=str(processed_dir / "report_text.json"),
            output_path=str(run_dir / "report" / "report_preview.pdf"),
            images_dir=str(images_dir),
            verbose=True,
            native_tables=native_tables,
            scratch_dir=str(run_dir / "report" / "_img_opt"),
        )

    summary = {
        "project_id": project_id,
        "raw_rows": int(len(raw_df)),
        "original_filtered_rows": int(len(original_filtered_df)),
        "patched_filtered_rows": int(len(patched_filtered_df)),
        "original_report_rows": int(len(original_report_df)),
        "patched_report_rows": int(len(patched_report_df)),
        "original_band_top": dict(list(_band_summary(original_report_df).items())[:8]),
        "patched_band_top": dict(list(_band_summary(patched_report_df).items())[:8]),
        "pdf_path": str(pdf_path),
    }
    (run_dir / "preview_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Band-fix preview PDF for project 248 without production changes.")
    parser.add_argument("--project-id", type=int, default=248)
    parser.add_argument("--user-id", type=int, default=13)
    parser.add_argument(
        "--out-dir",
        default="tests/output/report_engine_band_fix_preview",
        help="Base output directory.",
    )
    args = parser.parse_args()
    build_preview(args.project_id, args.user_id, Path(args.out_dir))


if __name__ == "__main__":
    main()
