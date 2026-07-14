"""
Standalone test case (test-case only -- new_report_sections.py and
test_new_pdf_report.py are NOT touched, and the full report pipeline in
either file is NOT run). Builds a MINIMAL PDF containing ONLY Section
"4.2 RSRP Analysis (Coverage)", reusing production's own NewFormatPDFReport
styles/table/flowable helpers unchanged, but with the per-technology map
images swapped for the GRID maps grid_rsrp_map_test.py already generated for
project 292 (a polygon project), instead of production's raw per-point maps.

Why this script exists: grid_rsrp_map_test.py's standalone PNG has a lot of
background/whitespace around the actual drive route. This script checks
whether that's still a problem once the image is placed into a PDF page at
the SAME size the real report uses -- reusing new_report_sections.py's own
4.2 image sizing (5.8in x 4in, via _sized_image/_compress_png) rather than
inventing a new size for this check.

Run directly with (from the ML/ directory), AFTER grid_rsrp_map_test.py has
generated its images:
    venv-win/Scripts/python.exe -m tests.new_pdf_report.grid_rsrp_map_test
    venv-win/Scripts/python.exe -m tests.new_pdf_report.grid_section_pdf_test
"""
from pathlib import Path

from reportlab.lib.units import inch
from reportlab.platypus import Spacer, Paragraph, KeepTogether

from tests.new_pdf_report.new_report_sections import (
    NewFormatPDFReport, _series, _technology_groups, _tech_slug,
    classify_coverage, build_rsrp_metric_table, make_native_table,
)
from tests.new_pdf_report.grid_rsrp_map_test import AGGREGATION

PROJECT_ID = 292


def build_grid_rsrp_section(report: NewFormatPDFReport, report_df, grid_maps_dir: Path) -> int:
    """Appends the 4.2 RSRP block (same content as the real report's 4.2)
    to `report.story`, but with grid map images instead of raw-point maps.
    Returns how many per-technology grid images were actually placed."""
    rsrp = _series(report_df, "rsrp")
    coverage_status, coverage_remarks = classify_coverage(rsrp)

    rsrp_flowables = [
        Paragraph(
            "Definition: Reference Signal Received Power (RSRP) is the primary LTE "
            "coverage KPI and indicates the received signal strength from the serving cell.",
            report.styles["Body"],
        ),
        Spacer(1, 4),
        Paragraph("<b>Acceptance Criteria</b>", report.styles["Body"]),
        *report._bullet_flowables([
            "Good: RSRP &gt; -95 dBm for more than 90% of samples",
            "Fair: Average RSRP -95 to -105 dBm",
            "Poor: Average RSRP &lt; -105 dBm, or coverage (% &gt; -95 dBm) &lt; 75%",
        ]),
        Spacer(1, 4),
        make_native_table(build_rsrp_metric_table(rsrp)),
        Spacer(1, 4),
        Paragraph(
            f"Observation: Coverage classified as <b>{coverage_status}</b>. {coverage_remarks} "
            f"Maps below are aggregated into the project's grid ({AGGREGATION} per cell) "
            f"instead of raw per-point markers.",
            report.styles["Body"],
        ),
    ]
    report.add_labeled_block("4.2 RSRP Analysis (Coverage) — GRID", rsrp_flowables)

    placed = 0
    for tech in _technology_groups(report_df):
        map_path = grid_maps_dir / f"rsrp_grid_map_{_tech_slug(tech)}.png"
        if not map_path.exists():
            continue
        report.story.append(KeepTogether([
            Paragraph(f"<b>{tech}</b>", report.styles["Body"]),
            Spacer(1, 2),
            report._sized_image(report._compress_png(str(map_path)), 5.8 * inch, 4 * inch),
        ]))
        report.story.append(Spacer(1, 6))
        placed += 1
    return placed


if __name__ == "__main__":
    from tests.new_pdf_report.test_new_pdf_report import _load_report_data

    _, report_df, _, project_meta = _load_report_data(PROJECT_ID)

    out_dir = Path(__file__).parent / "output" / f"project_{PROJECT_ID}"
    grid_maps_dir = out_dir / "images" / "grid_maps"
    if not any(grid_maps_dir.glob("rsrp_grid_map_*.png")):
        raise RuntimeError(
            f"No grid maps found in {grid_maps_dir} -- run "
            f"`python -m tests.new_pdf_report.grid_rsrp_map_test` first."
        )

    scratch_dir = out_dir / "_img_opt"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / "grid_section_4_2_only.pdf"
    report = NewFormatPDFReport(
        output_path=str(pdf_path),
        images_dir=str(out_dir / "images"),
        scratch_dir=str(scratch_dir),
    )
    placed = build_grid_rsrp_section(report, report_df, grid_maps_dir)
    report.build()

    print(f"[pdf] placed {placed} per-technology grid map(s)")
    print(f"[pdf] grid-only 4.2 section generated: {pdf_path}")
