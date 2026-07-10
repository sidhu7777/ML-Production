"""
Step-by-step test harness for the NEW PDF report format (report_VI.pdf).

Base example: project 248 (JigarVi, Mumbai — 3 sessions), loaded from the
live DB via DATABASE_URL in ML/.env. tools/report_engine (production) is
NOT modified; the primary-row filter step is rebuilt here to also include
5G NSA (see new_report_sections.filter_primary_rows_including_nr) — every
other step reuses production's own exported functions unchanged:
  - DB fetch (tbl_project / tbl_network_log / map_regions)
  - polygon filter (production's polygon_filter_all_cells)
  - NR-aware primary-row filter (test-case only, see above)
  - known-band filter (production's filter_known_band_rows)
  - drive summary metadata, area summary (spatial grid + geocoding),
    location, band/pci summaries (build_metadata)
  - base route map (folium + Playwright, same as production)

LLM policy (small API quota): only the Introduction narrative uses the
LLM (single small call, production fallback synthesizer if it fails);
Area Summary / Drive Summary / Executive Summary / Coverage KPI Analysis
are all rule-based — Area Summary is rule-based in production too.

Pages implemented so far (regenerates and REPLACES the same PDF each run):
  Page 1   Cover (production add_cover, geocoded location)
  Page 2   Table of Contents (production TOC)
  Page 3   1. Introduction (LLM) + 2. Area Summary (production logic + route map)
  Page 4+  3. Drive Summary in the NEW format
           (metric table + Session Details + Executive Summary block)
  Page 5+  4. Coverage KPI Analysis in the NEW format
           (4.1 Band Distribution, 4.2 RSRP [blended], 4.3 RSRQ and
           4.4 SINR [each per-technology, never blended], 4.5 Coverage
           KPI Summary, Overall Coverage Assessment) — reuses
           classify_coverage/classify_quality_by_technology from Section 3
           so the two sections can never disagree.
Next steps add: 5. Mobility KPI Analysis, 6. Handover KPI Analysis,
7. Service KPI Analysis — a few pages at a time.

Run standalone (from the ML folder):
    venv\\Scripts\\python.exe -B tests\\new_pdf_report\\test_new_pdf_report.py --project-id 248

Or with pytest (from the ML folder):
    venv\\Scripts\\python.exe -m pytest tests\\new_pdf_report\\test_new_pdf_report.py -s
"""

import argparse
import json
import os
import sys
from pathlib import Path

# --- Bootstrap: make `tools.` / `utils.` importable and load ML/.env ---
ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ML_ROOT / ".env")

import pytest  # noqa: E402

DEFAULT_PROJECT_ID = 248

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
PDF_NAME = "new_format_report.pdf"  # single evolving artifact, replaced each run


def _output_dir(project_id: int) -> Path:
    out = OUTPUT_ROOT / f"project_{project_id}"
    (out / "images" / "kpi_maps").mkdir(parents=True, exist_ok=True)
    (out / "images" / "kpi_analysis").mkdir(parents=True, exist_ok=True)
    (out / "html").mkdir(parents=True, exist_ok=True)
    return out


def _load_report_data(project_id: int):
    """
    Load project data using production's DB loaders, but rebuild the
    primary-row filter step ourselves (test-case only — see
    filter_primary_rows_including_nr in new_report_sections.py for why):
    production's load_project_data() applies _filter_primary_rows(), which
    only recognizes the LTE "mRegistered=YES" tag and silently drops every
    NR/5G-NSA row. We instead take raw_df, apply the polygon filter
    (production's own exported polygon_filter_all_cells — unchanged),
    then our NR-aware primary filter, then production's own
    filter_known_band_rows — unchanged. tools/report_engine is not
    modified anywhere in this chain; we're just calling its public
    functions in a corrected order/combination.
    """
    from tools.report_engine.load_data_db import (
        load_project_data,
        filter_known_band_rows,
        polygon_filter_all_cells,
    )
    from tests.new_pdf_report.new_report_sections import filter_primary_rows_including_nr

    raw_df, _production_filtered_df, project_meta = load_project_data(project_id)
    polygon_wkt = project_meta.get("region")

    # All cells, polygon-filtered only — production's own dataframe, used
    # here as the base for both handover detection (unchanged use) and the
    # corrected primary-row filter below.
    handover_df = polygon_filter_all_cells(raw_df, polygon_wkt)

    nr_aware_primary_df = filter_primary_rows_including_nr(handover_df)
    report_df = filter_known_band_rows(nr_aware_primary_df)

    print(f"[data] raw rows                                    : {len(raw_df)}")
    print(f"[data] production's primary+polygon rows (for ref) : {len(_production_filtered_df)}")
    print(f"[data] all-cells, polygon rows                     : {len(handover_df)}")
    print(f"[data] NR-aware primary rows                       : {len(nr_aware_primary_df)}")
    print(f"[data] report rows (+ known band, NR-aware)        : {len(report_df)}")

    if "network" in raw_df.columns and "network" in report_df.columns:
        raw_techs = set(raw_df["network"].dropna().astype(str).str.strip()) - {""}
        report_techs = set(report_df["network"].dropna().astype(str).str.strip()) - {""}
        dropped = raw_techs - report_techs
        if dropped:
            print(
                f"[data] NOTE: technology/network value(s) {sorted(dropped)} present in raw logs "
                f"but still absent from the NR-aware report_df (check band-normalization or "
                f"polygon filtering for these rows)."
            )
        else:
            print(f"[data] all raw technologies present in report_df: {sorted(raw_techs)}")
    return raw_df, report_df, handover_df, project_meta


def _render_base_route_map(report_df, polygon_wkt, out_dir: Path) -> bool:
    """Base route map exactly like production main.py (folium + Playwright)."""
    from tools.report_engine.map_generator import generate_base_route_map
    from tools.report_engine.playwright_utils import html_to_png

    html_path = out_dir / "html" / "base_route.html"
    png_path = out_dir / "images" / "kpi_maps" / "base_route_map.png"
    try:
        generate_base_route_map(report_df, polygon_wkt, str(html_path))
        html_to_png(
            str(html_path), str(png_path),
            width=1200, height=900, device_scale_factor=1,
        )
        print(f"[map] base route map generated: {png_path.name}")
        return True
    except Exception as exc:
        print(f"[map] WARNING: base route map failed (section renders without it): {exc}")
        return False


def _render_coverage_kpi_images(report_df, polygon_wkt, out_dir: Path, user_id=None) -> None:
    """
    Section 4 (Coverage KPI Analysis) images — Band map, RSRP/RSRQ/SINR
    maps, and the RSRP/RSRQ/SINR CDF charts — generated with production's
    OWN unmodified functions (generate_categorical_kpi_map /
    generate_kpi_map / resolve_kpi_ranges / generate_all_cdf_plots_from_df),
    exactly the way main.py's map-rendering step already does it. Only the
    3 numeric KPIs this section needs are rendered here; DL/UL/MOS/PCI
    images arrive when the Service/Mobility KPI Analysis sections
    (Steps 3/5) are built.

    RSRP is comparable in dBm terms across RATs (same as its blended 4.2
    table), so it keeps a single map coloring each point by its own actual
    measured value. RSRQ/SINR are classified PER TECHNOLOGY in their 4.3/4.4
    tables (their characteristic ranges differ by RAT), so their maps are
    now split the same way: one map per technology present in `network`,
    named "{rsrq,sinr}_map_{_tech_slug(tech)}.png", instead of a single map
    blending every RAT's points together. The CDF charts stay a single
    blended distribution per KPI, matching production's own
    generate_all_cdf_plots_from_df behaviour (unchanged).
    """
    from tools.report_engine.threshold_resolver import resolve_kpi_ranges
    from tools.report_engine.map_generator import (
        generate_kpi_map, generate_categorical_kpi_map, has_valid_numeric_data,
    )
    from tools.report_engine.playwright_utils import html_to_png
    from tools.report_engine.kpi_config import rsrp_colour_manual, rsrq_color_manual, sinr_color_manual
    from tools.report_engine import kpi_analysis
    from tests.new_pdf_report.new_report_sections import _technology_groups, _tech_slug

    html_dir = out_dir / "html"
    maps_dir = out_dir / "images" / "kpi_maps"

    # ---- Band categorical map + pie chart (kpi_analysis.generate_band_summary
    # writes band_pie.png as a side effect into kpi_analysis.IMAGE_DIR,
    # which run_report() already points at this project's images folder) ----
    try:
        kpi_analysis.generate_band_summary(report_df)
        html_path = html_dir / "band.html"
        png_path = maps_dir / "band_map.png"
        generate_categorical_kpi_map(
            df=report_df, kpi_column="band", output_html=str(html_path), polygon_wkt=polygon_wkt,
        )
        html_to_png(str(html_path), str(png_path), width=1200, height=900, device_scale_factor=1)
        print("[map] generated band_map.png + band_pie.png")
    except Exception as exc:
        print(f"[map] WARNING: failed to generate band map/pie: {exc}")

    # ---- RSRP map (single, blended across technologies) ----
    if "rsrp" in report_df.columns:
        df_kpi = report_df[report_df["rsrp"].notna() & report_df["lat"].notna() & report_df["lon"].notna()]
        if df_kpi.empty or not has_valid_numeric_data(report_df, "rsrp"):
            print("[map] skipped RSRP map: no valid data")
        else:
            try:
                ranges = resolve_kpi_ranges(kpi_name="RSRP", user_id=user_id, values=report_df["rsrp"])
                html_path = html_dir / "rsrp.html"
                png_path = maps_dir / "rsrp_map.png"
                generate_kpi_map(
                    df=df_kpi, kpi_column="rsrp", color_func=rsrp_colour_manual,
                    ranges=ranges, output_html=str(html_path), polygon_wkt=polygon_wkt,
                )
                html_to_png(str(html_path), str(png_path), width=1200, height=900, device_scale_factor=1)
                print("[map] generated rsrp_map.png")
            except Exception as exc:
                print(f"[map] WARNING: failed to generate RSRP map: {exc}")

    # ---- RSRQ / SINR maps — one per technology, never blended (their
    # KPI tables are already per-technology; see module docstring above) ----
    per_tech_kpi_specs = [
        ("RSRQ", "rsrq", rsrq_color_manual, "rsrq_map"),
        ("SINR", "sinr", sinr_color_manual, "sinr_map"),
    ]
    tech_list = _technology_groups(report_df)
    network_col = (
        report_df["network"].fillna("").astype(str).str.strip()
        if "network" in report_df.columns else None
    )
    for kpi_name, col, color_func, map_prefix in per_tech_kpi_specs:
        if col not in report_df.columns:
            continue
        techs = tech_list if (tech_list and network_col is not None) else [None]
        for tech in techs:
            df_tech = report_df if tech is None else report_df.loc[network_col == tech]
            df_kpi = df_tech[df_tech[col].notna() & df_tech["lat"].notna() & df_tech["lon"].notna()]
            tech_label = tech or "ALL"
            if df_kpi.empty or not has_valid_numeric_data(df_tech, col):
                print(f"[map] skipped {kpi_name} map for {tech_label}: no valid data")
                continue
            try:
                # Ranges stay based on the full KPI distribution (not the
                # per-technology subset) so color scales are comparable
                # across each technology's map for the same KPI.
                ranges = resolve_kpi_ranges(kpi_name=kpi_name, user_id=user_id, values=report_df[col])
                slug = _tech_slug(tech) if tech else "all"
                html_path = html_dir / f"{kpi_name.lower()}_{slug}.html"
                png_path = maps_dir / f"{map_prefix}_{slug}.png"
                generate_kpi_map(
                    df=df_kpi, kpi_column=col, color_func=color_func,
                    ranges=ranges, output_html=str(html_path), polygon_wkt=polygon_wkt,
                )
                html_to_png(str(html_path), str(png_path), width=1200, height=900, device_scale_factor=1)
                print(f"[map] generated {png_path.name} ({tech_label})")
            except Exception as exc:
                print(f"[map] WARNING: failed to generate {kpi_name} map for {tech_label}: {exc}")

    # ---- RSRP / RSRQ / SINR CDF charts (also generates DL/UL/MOS/PCI CDFs
    # for later steps, matching production's single-call behaviour) ----
    try:
        from tools.report_engine.cdf_kpi import generate_all_cdf_plots_from_df
        generate_all_cdf_plots_from_df(report_df, output_dir=str(out_dir / "images" / "kpi_analysis"))
    except Exception as exc:
        print(f"[map] WARNING: failed to generate CDF plots: {exc}")


def run_report(project_id: int = DEFAULT_PROJECT_ID, use_llm: bool = True) -> Path:
    """
    Generate the new-format report with everything implemented so far
    (pages 1-4: cover, TOC, Introduction, Area Summary, new Drive Summary).
    """
    from tools.report_engine import kpi_analysis
    from tools.report_engine.kpi_analysis import generate_drive_summary_images
    from tools.report_engine.map_generator import detect_handover_events
    from tools.report_engine.metadata_generator import build_metadata, write_metadata_file

    from tests.new_pdf_report.new_report_sections import (
        NewFormatPDFReport,
        derive_executive_summary,
        generate_introduction_text,
        compute_session_distances_km,
    )

    out_dir = _output_dir(project_id)

    # ---------- 1. DATA (production loaders / filters) ----------
    _, report_df, handover_df, project_meta = _load_report_data(project_id)
    assert not report_df.empty, f"Project {project_id} produced no report rows"

    # ---------- 2. DRIVE SUMMARY METADATA (production) ----------
    kpi_analysis.IMAGE_DIR = str(out_dir / "images" / "kpi_analysis")
    session_ids = [
        int(s.strip())
        for s in str(project_meta.get("ref_session_id", "")).split(",")
        if s.strip().isdigit()
    ]
    drive_summary = generate_drive_summary_images(
        session_ids, len(report_df), network_df=report_df
    )
    assert drive_summary is not None, "No drive summary metadata generated"
    assert drive_summary["total_samples"] == len(report_df)
    assert drive_summary["total_sessions"] == len(session_ids)

    # ---------- 2b. SESSION DISTANCE (Haversine sum-of-consecutive-points;
    #             tbl_session distance query is permission-denied so
    #             production silently reports 0.0 km) ----------
    distances_km = compute_session_distances_km(handover_df, session_ids)
    total_distance_km = round(sum(distances_km.values()), 2)
    drive_summary["distance_covered"] = total_distance_km
    for s in drive_summary.get("sessions", []):
        s["distance"] = distances_km.get(int(s["session_id"]), 0.0)
    print(f"[distance] per-session (km): {distances_km}")
    print(f"[distance] total (km): {total_distance_km}")

    # ---------- 3. METADATA (production build_metadata: location,
    #             area summary w/ geocoding, band + pci summaries) ----------
    print("[meta] building metadata (geocoding via Nominatim, ~10s)...")
    metadata = build_metadata(
        report_df,
        kpi_analysis_results={},          # KPI details arrive in later steps
        drive_summary_data=drive_summary,
    )
    write_metadata_file(metadata, str(out_dir / "report_metadata.json"))
    area_summary = metadata.get("area_summary")
    assert isinstance(area_summary, dict) and area_summary.get("Major Areas Covered"), \
        "Production area summary was not built"

    # ---------- 4. BASE ROUTE MAP (production folium + Playwright) ----------
    _render_base_route_map(report_df, project_meta.get("region"), out_dir)

    # ---------- 4b. COVERAGE KPI IMAGES (Section 4: Band/RSRP/RSRQ/SINR
    #             maps + CDF charts, production's own map/CDF generators) ----------
    _render_coverage_kpi_images(report_df, project_meta.get("region"), out_dir)

    # ---------- 5. INTRODUCTION (LLM, minimal prompt; synth fallback) ----------
    if use_llm:
        intro_text, intro_source = generate_introduction_text(metadata)
    else:
        from tools.report_engine.llm_integration import _fill_missing_sections_from_metadata
        intro_text = _fill_missing_sections_from_metadata(
            {}, metadata, missing_keys=["Introduction"]
        )["Introduction"]
        intro_source = "synth (--no-llm)"
    assert intro_text and intro_text.strip(), "Introduction text is empty"
    print(f"[text] Introduction source: {intro_source}")

    # ---------- 6. EXECUTIVE SUMMARY (rule-based) ----------
    # Handover EVENT DETECTION genuinely needs the broader all-cells,
    # polygon-only time series (transitions can pass through rows that
    # wouldn't survive a primary/band filter) — this matches production's
    # own stated intent for handover_df and is unchanged.
    try:
        handover_count = len(detect_handover_events(handover_df))
    except Exception as exc:
        print(f"[exec] handover detection skipped: {exc}")
        handover_count = None
    # Mobility (PCI dominance) now uses the SAME corrected report_df as
    # Coverage and Radio Quality — see filter_primary_rows_including_nr:
    # once report_df properly includes 5G NSA via the `primary` column,
    # its own top-PCI share/unique-PCI already match the frontend directly
    # (129 unique / 40.76% vs. frontend's 129 / 40.8%), so there is no
    # more reason to special-case Mobility onto a different dataframe.
    exec_summary = derive_executive_summary(report_df, handover_count, mobility_df=report_df)
    # Coverage + Mobility + Handover (1 row each) + one Radio Quality row per
    # technology present in `network` (2G/3G/4G/5G are never blended together).
    assert len(exec_summary["kpi_rows"]) >= 4
    assert exec_summary["observations"], "Executive observations are empty"

    # ---------- artifacts for review ----------
    band_summary = metadata.get("band_summary")
    (out_dir / "report_text.json").write_text(
        json.dumps(
            {
                "Introduction": intro_text,
                "Introduction_source": intro_source,
                "Area Summary": area_summary,
                "Drive Summary (new format)": exec_summary,
                "Session distances (km, Haversine consecutive-points)": distances_km,
                "Total distance (km)": total_distance_km,
                "Band Summary": band_summary,
            },
            indent=2, default=str, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---------- 7. PDF (same file replaced on every run) ----------
    pdf_path = out_dir / PDF_NAME
    report = NewFormatPDFReport(
        output_path=str(pdf_path),
        images_dir=str(out_dir / "images"),
        scratch_dir=str(out_dir / "_img_opt"),
    )
    report.add_cover(metadata)
    report.add_table_of_contents()
    report.add_introduction(intro_text)
    report.add_area_summary(area_summary)
    report.add_drive_summary_v2(drive_summary, exec_summary, distances_km=distances_km)
    report.add_coverage_kpi_analysis(report_df, exec_summary, band_summary)
    report.build()

    size_kb = pdf_path.stat().st_size / 1024
    assert pdf_path.exists() and size_kb > 1, "PDF was not generated"
    print(f"[pdf] generated (replaced): {pdf_path} ({size_kb:.1f} KB)")
    return pdf_path


# ------------------------------------------------------------------
# pytest entry point
# ------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"), reason="DATABASE_URL not configured"
)
def test_new_format_report_pages_1_to_4():
    run_report(DEFAULT_PROJECT_ID)


# ------------------------------------------------------------------
# standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="New-format PDF report test (step based)")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip the LLM call and use the production rule-based fallback")
    args = parser.parse_args()

    run_report(args.project_id, use_llm=not args.no_llm)
