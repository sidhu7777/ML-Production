import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests.debug.report_engine_base_map_debug as debug_report
from tests.debug.report_engine_base_map_debug import generate_full_report
from tools.report_engine import load_data_db as report_load
from tools.report_engine.load_data_db import filter_known_band_rows
from tools.report_engine.map_generator import normalize_band_name


def _make_run_dir(base_dir: Path, project_id: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"report_engine_project_{project_id}_primary_fix_preview_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _band_summary(df):
    if "band" not in df.columns:
        return {}
    counts = df["band"].apply(normalize_band_name).value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def _top_summary(df, limit: int = 8):
    items = list(_band_summary(df).items())[:limit]
    return dict(items)


def _patched_filter_primary_rows(df):
    if df.empty:
        return df

    primary_masks = []

    if "primary_cell_info_1" in df.columns:
        primary_info = df["primary_cell_info_1"].fillna("").astype(str)
        primary_masks.append(primary_info.str.contains("mRegistered=YES", case=False, na=False))

    if "primary" in df.columns:
        primary_values = df["primary"].fillna("").astype(str).str.strip().str.lower()
        primary_masks.append(primary_values.isin({"yes", "y", "true", "1"}))

    if not primary_masks:
        return df

    combined = primary_masks[0].copy()
    for mask in primary_masks[1:]:
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


@contextmanager
def patched_pdf_without_handover():
    original_has_image = debug_report._CrispPDFGenerator.has_image

    def _has_image_without_handover(self, filename, subdir="kpi_maps"):
        if filename == "handover_map.png":
            return False
        return original_has_image(self, filename, subdir=subdir)

    debug_report._CrispPDFGenerator.has_image = _has_image_without_handover
    try:
        yield
    finally:
        debug_report._CrispPDFGenerator.has_image = original_has_image


def build_preview(
    project_id: int,
    user_id: int | None,
    out_dir: Path,
    skip_llm: bool = False,
) -> Path:
    run_dir = _make_run_dir(out_dir, project_id)

    raw_df, original_filtered_df, _ = report_load.load_project_data(project_id)
    original_report_df = filter_known_band_rows(original_filtered_df)

    with patched_primary_filter():
        _, patched_filtered_df, _ = report_load.load_project_data(project_id)
        patched_report_df = filter_known_band_rows(patched_filtered_df)
        with patched_pdf_without_handover():
            report_dir = generate_full_report(
                project_id=project_id,
                user_id=user_id,
                out_dir=run_dir,
                skip_llm=skip_llm,
            )

    summary = {
        "project_id": project_id,
        "raw_rows": int(len(raw_df)),
        "original_filtered_rows": int(len(original_filtered_df)),
        "patched_filtered_rows": int(len(patched_filtered_df)),
        "original_report_rows": int(len(original_report_df)),
        "patched_report_rows": int(len(patched_report_df)),
        "original_band_top": _top_summary(original_report_df),
        "patched_band_top": _top_summary(patched_report_df),
        "original_n77": int(_band_summary(original_report_df).get("n77", 0)),
        "patched_n77": int(_band_summary(patched_report_df).get("n77", 0)),
        "original_n78": int(_band_summary(original_report_df).get("n78", 0)),
        "patched_n78": int(_band_summary(patched_report_df).get("n78", 0)),
        "report_dir": str(report_dir),
    }

    (run_dir / "primary_fix_preview_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Test-only preview for project 248 primary filter fix.")
    parser.add_argument("--project-id", type=int, default=248)
    parser.add_argument("--user-id", type=int, default=13)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM generation and use fallback text.",
    )
    parser.add_argument(
        "--out-dir",
        default="tests/output/report_engine_primary_fix_preview",
        help="Base output directory.",
    )
    args = parser.parse_args()
    build_preview(
        args.project_id,
        args.user_id,
        Path(args.out_dir),
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()
