"""
Production entry point for the "Per Technology" / expanded-format PDF
report (38 pages: per-technology KPI breakdown, deep Coverage/Mobility/
Handover/Service sections).

This module is the promoted, route-driven version of the step-based test
harness at tests/new_pdf_report/test_new_pdf_report.py (its run_report()
function). That harness is left completely UNCHANGED and stays available
as a standalone dev/test tool for re-testing this report format directly
against a hardcoded project id; this module is what tools/New_pdf_report/
routes.py calls for real requests, following tools/report_engine/main.py's
own project_id/user_id/report_id/db_engine/region/country_code convention
so both report types ("Combined" via tools.report_engine, "Per Technology"
via this module) are callable the exact same way from the app.

Differences from the test harness this was adapted from (test-only
behaviour that intentionally does NOT reach production -- see that file's
own comments for the full reasoning):
  - DEFAULT_THRESHOLD_USER_ID (=13) in the test harness was a placeholder
    threshold owner used only because project 248 (the test harness's
    default project) has no project -> user link in the DB. Production
    always receives a real `user_id` argument, so that is used directly
    everywhere KPI thresholds are resolved -- no placeholder constant
    exists in this module.
  - GRID_SIZE_METERS_OVERRIDE (=50) in the test harness was an explicit,
    documented test-only override of the project's own configured grid
    resolution. This module uses
    grid_maps.resolve_project_grid_size_meters(project_meta) directly,
    unmodified -- the project's real tbl_project.grid_size value.
  - Output directories: the test harness writes to
    tests/new_pdf_report/output/project_<id>/. This module instead
    follows tools/report_engine/main.py's own data/tmp/<report_id>/
    (working files) convention, but writes the final PDF under
    data/new_pdf_reports/<report_id>/ -- a SEPARATE namespace from
    tools/report_engine's own data/reports/<report_id>/, so the two
    report types never collide on the same report_id folder.
  - The Introduction section always uses the LLM (use_llm=True) --
    the test harness's --no-llm flag was a developer convenience for
    quota-limited manual runs and has no equivalent CLI here.
  - tbl_project.Download_path is updated after a successful run, same as
    tools/report_engine/main.py, but pointed at this module's own
    /api/new-pdf-report/download/<report_id> route instead of
    /api/report/download/<report_id>. Report-ready email is NOT sent --
    left commented out (same disabled-on-purpose state as
    tools/report_engine/main.py's own email block) for later reuse.

tools/report_engine and tests/new_pdf_report are NOT modified by this
module -- it only imports their public functions, exactly like the test
harness already did.
"""

import json
import os
import shutil
import uuid
from pathlib import Path

import pandas as pd

from tools.report_engine.db import init_engine, get_user_by_id, update_project_download_path
from tools.report_engine.email_service import send_report_ready_email
from tools.report_engine.load_data_db import (
    load_project_data,
    filter_known_band_rows,
    polygon_filter_all_cells,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
NEW_PDF_REPORTS_DIR = os.path.join(DATA_DIR, "new_pdf_reports")
PDF_NAME = "report.pdf"


def clean_directory(path):
    """Same cleanup helper as tools/report_engine/main.py: removes every
    file/subdirectory INSIDE `path`, but leaves `path` itself in place."""
    if os.path.exists(path):
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                os.remove(fp)
            else:
                shutil.rmtree(fp)


def _asset_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _load_report_data(
    project_id: int,
    region: str | None = None,
    country_code: str | None = None,
    technologies: list[str] | None = None,
):
    """
    Production adaptation of test_new_pdf_report.py's _load_report_data().

    Loads project data using production's DB loaders, but rebuilds the
    primary-row filter step (test-case-derived, ported unchanged into
    tools/New_pdf_report/new_report_sections.py as
    filter_primary_rows_including_nr): production's load_project_data()
    applies its own primary-row filter, which only recognizes the LTE
    "mRegistered=YES" tag and silently drops every NR/5G-NSA row. We
    instead take raw_df, apply the polygon filter (production's own
    exported polygon_filter_all_cells -- unchanged), then the NR-aware
    primary filter, then production's own filter_known_band_rows --
    unchanged. tools/report_engine is not modified anywhere in this
    chain; we're just calling its public functions in a corrected
    order/combination, same as the test harness did.
    """
    from shapely import contains as shp_contains, points as shp_points
    from shapely.ops import transform as shp_transform
    from shapely.wkt import loads as load_wkt, dumps as dump_wkt

    from tools.New_pdf_report.new_report_sections import filter_primary_rows_including_nr

    raw_df, _production_filtered_df, project_meta = load_project_data(
        project_id, region=region, country_code=country_code, technologies=technologies,
    )
    polygon_wkt = project_meta.get("region")

    # Detect and correct a lat/lon axis-order mismatch in the stored region
    # polygon (see test_new_pdf_report.py's _load_report_data for the full
    # background -- confirmed root cause for at least one project). This
    # affects every downstream consumer of project_meta["region"], not just
    # row filtering, so it's corrected once here rather than at each call
    # site.
    if polygon_wkt:
        poly = load_wkt(polygon_wkt)
        geo = raw_df.dropna(subset=["lat", "lon"])
        if not geo.empty:
            lon = pd.to_numeric(geo["lon"], errors="coerce").to_numpy(dtype=float)
            lat = pd.to_numeric(geo["lat"], errors="coerce").to_numpy(dtype=float)
            pts = shp_points(lon, lat)
            inside_as_is = int(shp_contains(poly, pts).sum())

            def _swap_xy(x, y, z=None):
                return (y, x) if z is None else (y, x, z)

            swapped_poly = shp_transform(_swap_xy, poly)
            inside_swapped = int(shp_contains(swapped_poly, pts).sum())
            if inside_swapped > inside_as_is:
                polygon_wkt = dump_wkt(swapped_poly)
                project_meta["region"] = polygon_wkt
                print(
                    f"[data] NOTE: project {project_id}'s stored region polygon was in "
                    f"(lat, lon) axis order -- corrected to standard WKT (lon, lat) order "
                    f"({inside_as_is} -> {inside_swapped} of {len(geo)} rows contained)."
                )

    # Bridge-prefiltered projects have already had polygon + primary
    # filtering applied server-side by GetDriveTestRows before Python ever
    # sees the rows -- mirror production and trust the bridge output as-is.
    bridge_prefiltered = bool(project_meta.get("report_data_prefiltered"))
    if bridge_prefiltered:
        handover_df = raw_df.reset_index(drop=True)
    else:
        handover_df = polygon_filter_all_cells(raw_df, polygon_wkt)

    nr_aware_primary_df = filter_primary_rows_including_nr(handover_df)
    report_df = filter_known_band_rows(nr_aware_primary_df)

    print(f"[data] raw rows                                    : {len(raw_df)}")
    print(f"[data] production's primary+polygon rows (for ref) : {len(_production_filtered_df)}")
    print(f"[data] bridge_prefiltered                          : {bridge_prefiltered}")
    print(f"[data] all-cells, polygon rows                     : {len(handover_df)}")
    print(f"[data] NR-aware primary rows                       : {len(nr_aware_primary_df)}")
    print(f"[data] report rows (+ known band, NR-aware)        : {len(report_df)}")

    return raw_df, report_df, handover_df, project_meta


def _load_neighbor_data(project_meta: dict):
    """
    Load dedicated neighbor logs from tbl_network_log_neighbour for the
    project's sessions, then apply the same project polygon filter used by
    the report pipeline. Ported as-is from test_new_pdf_report.py --
    production report_engine has no equivalent query, so this is new
    ground for the pipeline, not a duplicate of production logic.
    """
    from sqlalchemy import text, bindparam
    from tools.report_engine.db import get_engine

    session_ids = [
        int(s.strip())
        for s in str(project_meta.get("ref_session_id", "")).split(",")
        if s.strip().isdigit()
    ]
    if not session_ids:
        return pd.DataFrame()

    engine = get_engine()
    query = text("""
        SELECT *
        FROM tbl_network_log_neighbour
        WHERE session_id IN :session_ids
    """).bindparams(bindparam("session_ids", expanding=True))
    with engine.connect() as conn:
        neighbor_raw_df = pd.read_sql(query, conn, params={"session_ids": session_ids})

    neighbor_df = polygon_filter_all_cells(neighbor_raw_df, project_meta.get("region"))
    print(f"[neighbor] dedicated neighbor rows (raw)            : {len(neighbor_raw_df)}")
    print(f"[neighbor] dedicated neighbor rows (polygon)        : {len(neighbor_df)}")
    return neighbor_df


def _load_subsession_data(project_meta: dict) -> pd.DataFrame:
    """
    Load PS (data)/CS (voice) call-session records from tbl_sub_session for
    the project's sessions. Ported as-is from test_new_pdf_report.py --
    grep-confirmed that no existing tools/report_engine pipeline queries
    this table, so this is new ground, not a duplicate of production logic.
    """
    from sqlalchemy import text, bindparam
    from tools.report_engine.db import get_engine

    session_ids = [
        int(s.strip())
        for s in str(project_meta.get("ref_session_id", "")).split(",")
        if s.strip().isdigit()
    ]
    if not session_ids:
        return pd.DataFrame()

    engine = get_engine()
    query = text("""
        SELECT id, session_id, sub_session_id, type, status, start_time, end_time, json_data
        FROM tbl_sub_session
        WHERE session_id IN :session_ids
    """).bindparams(bindparam("session_ids", expanding=True))
    with engine.connect() as conn:
        subsession_df = pd.read_sql(query, conn, params={"session_ids": session_ids})
    print(f"[subsession] tbl_sub_session rows: {len(subsession_df)}")
    return subsession_df


def _render_base_route_map(
    report_df, polygon_wkt, out_dir: Path, grid_lattice=None, grid_size_meters=None,
) -> bool:
    """
    Base route map like production main.py (folium + Playwright). Grid vs
    raw-point split matches every other KPI map in this report: when the
    project has a polygon (grid_lattice populated), renders via
    generate_base_route_grid_map (solid route-coverage cells on the shared
    lattice); otherwise falls back to generate_base_route_map_polygon_aware
    (production's raw per-point circle markers, still framed to the full
    polygon when present).
    """
    from tools.New_pdf_report.new_report_sections import (
        generate_base_route_map_polygon_aware, generate_base_route_grid_map,
    )
    from tools.New_pdf_report.local_tiles import html_to_png_verified as html_to_png

    html_path = out_dir / "html" / "base_route.html"
    png_path = out_dir / "images" / "kpi_maps" / "base_route_map.png"
    if _asset_exists(png_path):
        print(f"[map] reused existing base route map: {png_path.name}")
        return True

    use_grid = bool(polygon_wkt) and grid_lattice is not None and not grid_lattice.empty
    try:
        if use_grid:
            from tools.New_pdf_report.google_tiles import use_gray_basemap

            with use_gray_basemap(polygon_wkt):
                ok = generate_base_route_grid_map(
                    report_df, grid_lattice, grid_size_meters, polygon_wkt, str(html_path),
                )
            if not ok:
                raise ValueError("no populated grid cells for base route map")
        else:
            generate_base_route_map_polygon_aware(report_df, polygon_wkt, str(html_path))
        html_to_png(
            str(html_path), str(png_path),
            width=1200, height=900, device_scale_factor=1,
        )
        print(f"[map] base route map generated: {png_path.name} ({'GRID' if use_grid else 'raw points'})")
        return True
    except Exception as exc:
        print(f"[map] WARNING: base route map failed (section renders without it): {exc}")
        return False


def _render_kpi_map_per_technology(
    report_df, polygon_wkt, grid_lattice, grid_size_meters,
    kpi_name, col, color_func, map_prefix, unit,
    html_dir: Path, maps_dir: Path, user_id=None,
) -> None:
    """
    One map PER TECHNOLOGY for a numeric KPI (RSRP/RSRQ/SINR in Section 4,
    DL/UL/MOS in Section 7) -- shared by both sections so the polygon/no
    polygon branch only lives in one place instead of being duplicated.

    If the project has a polygon (`grid_lattice` non-empty), renders the
    SAME grid every technology/KPI in this report shares (built once in
    main() via grid_maps.build_polygon_lattice) instead of production's raw
    per-point scatter map -- matching the frontend's own
    canEnableUnifiedGridView behaviour. No polygon -> unchanged raw-point
    rendering via production's own generate_kpi_map.
    """
    from tools.report_engine.threshold_resolver import resolve_kpi_ranges
    from tools.report_engine.map_generator import generate_kpi_map, has_valid_numeric_data
    from tools.New_pdf_report.local_tiles import html_to_png_verified as html_to_png
    from tools.New_pdf_report.new_report_sections import (
        _technology_groups, _tech_slug, technology_metric_label,
    )
    from tools.New_pdf_report.grid_maps import aggregate_grid_cells, generate_kpi_grid_map

    if col not in report_df.columns:
        return

    use_grid = bool(polygon_wkt) and grid_lattice is not None and not grid_lattice.empty

    # Grid cells actually touched by the drive route for this KPI, ALL
    # technologies combined -- the legend's "Total Grid Cells" figure.
    drive_route_total_cells = None
    if use_grid and col in report_df.columns:
        df_kpi_all_tech = report_df[
            report_df[col].notna() & report_df["lat"].notna() & report_df["lon"].notna()
        ]
        _, _, drive_route_total_cells = aggregate_grid_cells(df_kpi_all_tech, grid_lattice, value_col=col)

    tech_list = _technology_groups(report_df)
    network_col = (
        report_df["network"].fillna("").astype(str).str.strip()
        if "network" in report_df.columns else None
    )
    techs = tech_list if (tech_list and network_col is not None) else [None]

    # Per-technology display label for the map's own legend/tooltip text
    # ONLY (e.g. "RxLev" for a 2G RSRP map, "nrSINR" for a 5G SINR map) --
    # gated to RSRP/RSRQ/SINR specifically so the DL/UL/MOS calls into this
    # same function (Section 7) are completely unaffected, since those
    # KPIs aren't in TECHNOLOGY_METRIC_LABELS. This NEVER changes
    # `kpi_name` itself (still passed to resolve_kpi_ranges below exactly
    # as before) or `col`/`kpi_column` (the actual dataframe column read) --
    # only the human-facing label text.
    is_signal_kpi = col in ("rsrp", "rsrq", "sinr")

    for tech in techs:
        display_label = technology_metric_label(tech, col) if is_signal_kpi else kpi_name
        if display_label is None:
            # e.g. 2G RSRQ -- the frontend has no RSRQ concept for 2G at
            # all (TECHNOLOGY_METRIC_LABELS["2G"]["rsrq"] is None), so skip
            # this technology's map entirely rather than render an empty
            # or mislabeled one.
            print(f"[map] skipped {kpi_name} map for {tech or 'ALL'}: not applicable for this technology")
            continue
        df_tech = report_df if tech is None else report_df.loc[network_col == tech]
        df_kpi = df_tech[df_tech[col].notna() & df_tech["lat"].notna() & df_tech["lon"].notna()]
        tech_label = tech or "ALL"
        if df_kpi.empty or not has_valid_numeric_data(df_tech, col):
            print(f"[map] skipped {kpi_name} map for {tech_label}: no valid data")
            continue
        try:
            # Ranges stay based on the full KPI distribution (not the
            # per-technology subset) so color scales are comparable across
            # each technology's map for the same KPI.
            ranges = resolve_kpi_ranges(kpi_name=kpi_name, user_id=user_id, values=report_df[col])
            slug = _tech_slug(tech) if tech else "all"
            html_path = html_dir / f"{col}_{slug}.html"
            png_path = maps_dir / f"{map_prefix}_{slug}.png"
            if _asset_exists(png_path):
                print(f"[map] reused existing {png_path.name} ({tech_label})")
                continue

            if use_grid:
                from tools.New_pdf_report.google_tiles import use_gray_basemap

                cells, total, populated = aggregate_grid_cells(df_kpi, grid_lattice, value_col=col)
                if cells.empty:
                    print(f"[map] skipped {kpi_name} grid map for {tech_label}: no populated cells")
                    continue
                with use_gray_basemap(polygon_wkt):
                    generate_kpi_grid_map(
                        cells, ranges, str(html_path), polygon_wkt=polygon_wkt,
                        grid_size_meters=grid_size_meters, bounds_df=report_df,
                        metric_label=display_label.lower(), unit=unit, total_cells=drive_route_total_cells,
                    )
                html_to_png(str(html_path), str(png_path), width=1200, height=900, device_scale_factor=1)
                print(f"[map] generated {png_path.name} ({tech_label}, GRID: {populated}/{total} cells)")
            else:
                # generate_kpi_map (tools/report_engine/map_generator.py,
                # not modified) ties its legend title directly to
                # `kpi_column` -- there's no separate label parameter for
                # the raw-point path. To get a technology-correct legend
                # (e.g. "rxlev" instead of "rsrp" for 2G) without touching
                # report_engine, alias the KPI column under the display
                # label's own name in a local copy and point kpi_column at
                # that alias instead -- the underlying data/threshold
                # classification is untouched, only which column name the
                # map reads its legend title from.
                map_kpi_column = col
                map_df = df_kpi
                if display_label.lower() != col:
                    map_df = df_kpi.copy()
                    map_df[display_label.lower()] = map_df[col]
                    map_kpi_column = display_label.lower()
                generate_kpi_map(
                    df=map_df, kpi_column=map_kpi_column, color_func=color_func,
                    ranges=ranges, output_html=str(html_path), polygon_wkt=polygon_wkt,
                )
                html_to_png(str(html_path), str(png_path), width=1200, height=900, device_scale_factor=1)
                print(f"[map] generated {png_path.name} ({tech_label}, raw points)")
        except Exception as exc:
            print(f"[map] WARNING: failed to generate {kpi_name} map for {tech_label}: {exc}")


def _render_poor_region_maps_per_technology(
    report_df, polygon_wkt, grid_lattice, grid_size_meters,
    value_col: str, threshold: float, map_prefix: str,
    html_dir: Path, maps_dir: Path,
) -> None:
    """
    Per-technology counterpart of the single blended poor-region map
    (RSRP/RSRQ below the acceptance threshold), mirroring
    _render_kpi_map_per_technology's polygon/no-polygon branch and its
    per-technology skip logic: a technology is skipped entirely (no file
    generated) when technology_metric_label(tech, value_col) is None --
    e.g. 2G has no RSRQ concept at all, so no `rsrq_poor_regions_2g.png`
    is produced.

    Filenames follow `{map_prefix}_{_tech_slug(tech)}.png` (e.g.
    "rsrp_poor_regions_4g.png"), matching the naming convention
    `_render_kpi_map_per_technology` already uses for the main RSRP/RSRQ/
    SINR maps, so new_report_sections.py's rendering side can look them
    up the same way.
    """
    from tools.New_pdf_report.new_report_sections import (
        generate_poor_region_map_fixed, generate_poor_region_grid_map,
        _technology_groups, _tech_slug, technology_metric_label,
    )

    if value_col not in report_df.columns:
        return

    use_grid_poor = bool(polygon_wkt) and grid_lattice is not None and not grid_lattice.empty

    tech_list = _technology_groups(report_df)
    network_col = (
        report_df["network"].fillna("").astype(str).str.strip()
        if "network" in report_df.columns else None
    )
    techs = tech_list if (tech_list and network_col is not None) else [None]

    for tech in techs:
        display_label = technology_metric_label(tech, value_col)
        if display_label is None:
            # e.g. 2G under RSRQ -- not applicable, skip entirely.
            print(f"[map] skipped {map_prefix} map for {tech or 'ALL'}: not applicable for this technology")
            continue
        tech_label = tech or "ALL"
        slug = _tech_slug(tech) if tech else "all"
        png_path = maps_dir / f"{map_prefix}_{slug}.png"
        html_path = html_dir / f"{map_prefix}_{slug}.html"
        if _asset_exists(png_path):
            print(f"[map] reused existing {png_path.name} ({tech_label})")
            continue
        df_tech = report_df if tech is None else report_df.loc[network_col == tech]
        title = f"{display_label} < {threshold:g}"
        try:
            if use_grid_poor:
                from tools.New_pdf_report.google_tiles import use_gray_basemap

                with use_gray_basemap(polygon_wkt):
                    ok = generate_poor_region_grid_map(
                        df_tech, value_col, threshold, str(png_path), str(html_path),
                        title, grid_lattice=grid_lattice, grid_size_meters=grid_size_meters,
                        polygon_wkt=polygon_wkt,
                    )
                print(
                    f"[map] poor-region grid map {map_prefix} ({tech_label}): "
                    f"{'generated' if ok else 'SKIPPED (no cell median below threshold)'}"
                )
            else:
                ok = generate_poor_region_map_fixed(
                    df_tech, value_col, threshold, str(png_path), str(html_path),
                    title, polygon_wkt=polygon_wkt,
                )
                print(
                    f"[map] poor-region map {map_prefix} ({tech_label}): "
                    f"{'generated' if ok else 'SKIPPED (no poor samples)'}"
                )
        except Exception as exc:
            print(f"[map] WARNING: failed to generate {map_prefix} map for {tech_label}: {exc}")


def _render_coverage_kpi_images(
    report_df, polygon_wkt, out_dir: Path, user_id=None,
    grid_lattice=None, grid_size_meters=None,
) -> None:
    """
    Section 4 (Coverage KPI Analysis) images -- Band map (via
    generate_categorical_kpi_map_polygon_aware, so it frames the full
    polygon when one exists) and RSRP/RSRQ/SINR maps generated with
    production's OWN unmodified functions (generate_kpi_map /
    resolve_kpi_ranges) for the no-polygon raw-point case, plus the
    RSRP/RSRQ/SINR CDF charts generated by new_report_sections'
    generate_all_cdf_plots_from_df (adds the red acceptance-threshold
    crosshair; tools/report_engine is not modified).

    RSRP/RSRQ/SINR maps, and the RSRP/RSRQ poor-region maps below the
    acceptance threshold, are each generated ONE PER TECHNOLOGY present in
    `network` (see _render_kpi_map_per_technology /
    _render_poor_region_maps_per_technology) -- RSRP's own numeric
    classification in the 4.2 table stays blended across technologies
    (dBm is directly comparable across RATs), but each map plots per-
    point actual values, so a per-technology split is still useful there
    too. The CDF charts stay a single blended distribution per KPI.
    """
    from tools.New_pdf_report.new_report_sections import generate_categorical_kpi_map_polygon_aware
    from tools.New_pdf_report.local_tiles import html_to_png_verified as html_to_png
    from tools.report_engine.kpi_config import rsrp_colour_manual, rsrq_color_manual, sinr_color_manual
    from tools.report_engine import kpi_analysis

    html_dir = out_dir / "html"
    maps_dir = out_dir / "images" / "kpi_maps"

    # ---- Band categorical map + pie chart ----
    band_png = maps_dir / "band_map.png"
    band_pie = out_dir / "images" / "kpi_analysis" / "band_pie.png"
    if _asset_exists(band_png) and _asset_exists(band_pie):
        print("[map] reused existing band_map.png + band_pie.png")
    else:
        try:
            kpi_analysis.generate_band_summary(report_df)
            html_path = html_dir / "band.html"
            generate_categorical_kpi_map_polygon_aware(
                df=report_df, kpi_column="band", output_html=str(html_path), polygon_wkt=polygon_wkt,
            )
            html_to_png(str(html_path), str(band_png), width=1200, height=900, device_scale_factor=1)
            print("[map] generated band_map.png + band_pie.png")
        except Exception as exc:
            print(f"[map] WARNING: failed to generate band map/pie: {exc}")

    # ---- Poor-region maps (RSRP / RSRQ below acceptance threshold) --
    # one per technology, mirroring the main RSRP/RSRQ/SINR maps below
    # (_render_kpi_map_per_technology), not a single map blending every
    # technology's poor samples together. ----
    if "rsrp" in report_df.columns:
        from tools.New_pdf_report.new_report_sections import CDF_ACCEPTANCE_THRESHOLDS

        rsrp_threshold = CDF_ACCEPTANCE_THRESHOLDS["RSRP"]
        rsrq_threshold = CDF_ACCEPTANCE_THRESHOLDS["RSRQ"]
        _render_poor_region_maps_per_technology(
            report_df, polygon_wkt, grid_lattice, grid_size_meters,
            "rsrp", rsrp_threshold, "rsrp_poor_regions", html_dir, maps_dir,
        )
        _render_poor_region_maps_per_technology(
            report_df, polygon_wkt, grid_lattice, grid_size_meters,
            "rsrq", rsrq_threshold, "rsrq_poor_regions", html_dir, maps_dir,
        )

    # ---- RSRP / RSRQ / SINR maps -- one per technology, never blended ----
    per_tech_kpi_specs = [
        ("RSRP", "rsrp", rsrp_colour_manual, "rsrp_map", "dBm"),
        ("RSRQ", "rsrq", rsrq_color_manual, "rsrq_map", "dB"),
        ("SINR", "sinr", sinr_color_manual, "sinr_map", "dB"),
    ]
    for kpi_name, col, color_func, map_prefix, unit in per_tech_kpi_specs:
        _render_kpi_map_per_technology(
            report_df, polygon_wkt, grid_lattice, grid_size_meters,
            kpi_name, col, color_func, map_prefix, unit,
            html_dir, maps_dir, user_id=user_id,
        )

    # ---- RSRP / RSRQ / SINR CDF charts (also generates DL/UL/MOS/PCI CDFs
    # for later sections, matching production's single-call behaviour) ----
    cdf_dir = out_dir / "images" / "kpi_analysis"
    cdf_required = [
        cdf_dir / "cdf_rsrp.png",
        cdf_dir / "cdf_rsrq.png",
        cdf_dir / "cdf_sinr.png",
        cdf_dir / "cdf_dl_tpt.png",
        cdf_dir / "cdf_ul_tpt.png",
        cdf_dir / "cdf_mos.png",
        cdf_dir / "cdf_pci.png",
    ]
    if all(_asset_exists(p) for p in cdf_required):
        print("[map] reused existing CDF plots")
    else:
        try:
            from tools.New_pdf_report.new_report_sections import generate_all_cdf_plots_from_df
            generate_all_cdf_plots_from_df(report_df, output_dir=str(cdf_dir))
        except Exception as exc:
            print(f"[map] WARNING: failed to generate CDF plots: {exc}")


def _render_mobility_kpi_images(report_df, polygon_wkt, out_dir: Path) -> None:
    """
    Section 5 (Mobility KPI Analysis) visuals - PCI categorical map (via
    generate_categorical_kpi_map_polygon_aware, same polygon-framing
    treatment as the Band map) and PCI distribution artifacts generated
    using production's own helpers.
    """
    from tools.report_engine import kpi_analysis
    from tools.report_engine.map_generator import has_valid_categorical_data
    from tools.New_pdf_report.new_report_sections import generate_categorical_kpi_map_polygon_aware
    from tools.New_pdf_report.local_tiles import html_to_png_verified as html_to_png

    html_dir = out_dir / "html"
    maps_dir = out_dir / "images" / "kpi_maps"

    pci_dist = out_dir / "images" / "kpi_analysis" / "pci_distribution.png"
    pci_table = out_dir / "images" / "kpi_analysis" / "pci_table.png"
    if _asset_exists(pci_dist) and _asset_exists(pci_table):
        print("[map] reused existing pci_distribution.png + pci_table.png")
    else:
        try:
            kpi_analysis.generate_pci_distribution(report_df)
            print("[map] generated pci_distribution.png + pci_table.png")
        except Exception as exc:
            print(f"[map] WARNING: failed to generate PCI distribution artifacts: {exc}")

    try:
        if "pci" not in report_df.columns or not has_valid_categorical_data(report_df, "pci"):
            print("[map] skipped PCI map: no valid categorical PCI data")
            return
        html_path = html_dir / "pci.html"
        png_path = maps_dir / "pci_map.png"
        if _asset_exists(png_path):
            print("[map] reused existing pci_map.png")
            return
        generate_categorical_kpi_map_polygon_aware(
            df=report_df, kpi_column="pci", output_html=str(html_path), polygon_wkt=polygon_wkt,
        )
        html_to_png(str(html_path), str(png_path), width=1200, height=900, device_scale_factor=1)
        print("[map] generated pci_map.png")
    except Exception as exc:
        print(f"[map] WARNING: failed to generate PCI map: {exc}")


def _render_handover_kpi_images(handover_df, band_events, tech_events, polygon_wkt, out_dir: Path, grid_size_meters: float = 50.0) -> None:
    """
    Section 6 visuals: the band-handover route map
    (generate_handover_map_with_session_legend, which duplicates
    production's own generate_handover_map but also legends the
    per-session route colors) fed the frontend-style undeduped band_events
    from compute_handover_analysis (production's own location-based dedup
    undercounts real events), and the Inter-RAT technology-transition map
    (generate_tech_handover_map).
    """
    from tools.New_pdf_report.local_tiles import html_to_png_verified as html_to_png
    from tools.New_pdf_report.new_report_sections import (
        generate_tech_handover_map, generate_handover_map_with_session_legend,
    )

    html_dir = out_dir / "html"
    maps_dir = out_dir / "images" / "kpi_maps"

    band_png = maps_dir / "handover_map.png"
    if _asset_exists(band_png):
        print("[map] reused existing handover_map.png")
    else:
        try:
            html_path = html_dir / "handover_map.html"
            generate_handover_map_with_session_legend(
                handover_df, band_events, str(html_path), polygon_wkt=polygon_wkt,
                grid_size_meters=grid_size_meters,
            )
            html_to_png(str(html_path), str(band_png), width=1200, height=900, device_scale_factor=1)
            print(f"[map] generated handover_map.png ({len(band_events)} band events)")
        except Exception as exc:
            print(f"[map] WARNING: failed to generate handover map: {exc}")

    tech_png = maps_dir / "tech_handover_map.png"
    if _asset_exists(tech_png):
        print("[map] reused existing tech_handover_map.png")
    else:
        try:
            html_path = html_dir / "tech_handover_map.html"
            ok = generate_tech_handover_map(
                tech_events, str(html_path), str(tech_png), polygon_wkt=polygon_wkt,
                grid_size_meters=grid_size_meters,
            )
            if ok:
                print(f"[map] generated tech_handover_map.png ({len(tech_events)} tech events)")
        except Exception as exc:
            print(f"[map] WARNING: failed to generate tech handover map: {exc}")


def _render_key_findings_map(report_df, polygon_wkt, out_dir: Path) -> None:
    """
    Section 8 "Key Findings" map -- highlights the same best/worst eNodeB
    (_nodeb_coverage_extremes) and RSRP acceptance threshold
    (CDF_ACCEPTANCE_THRESHOLDS) the Overall Network Performance /
    Optimization Opportunities bullets above it already use, so the map
    and the bullet text can never name a different "best"/"worst" site.
    """
    from tools.New_pdf_report.new_report_sections import (
        generate_key_findings_map, _nodeb_coverage_extremes, CDF_ACCEPTANCE_THRESHOLDS,
    )

    html_dir = out_dir / "html"
    maps_dir = out_dir / "images" / "kpi_maps"

    png_path = maps_dir / "key_findings_map.png"
    if _asset_exists(png_path):
        print("[map] reused existing key_findings_map.png")
        return
    try:
        best_nodeb, worst_nodeb = _nodeb_coverage_extremes(report_df)
        html_path = html_dir / "key_findings_map.html"
        ok = generate_key_findings_map(
            report_df, best_nodeb, worst_nodeb, str(html_path), str(png_path),
            polygon_wkt=polygon_wkt, rsrp_threshold=CDF_ACCEPTANCE_THRESHOLDS["RSRP"],
        )
        if ok:
            print(
                f"[map] generated key_findings_map.png "
                f"(best={best_nodeb['nodeb_id'] if best_nodeb else None}, "
                f"worst={worst_nodeb['nodeb_id'] if worst_nodeb else None})"
            )
        else:
            print("[map] skipped key_findings_map.png: no best/worst eNodeB or below-threshold samples")
    except Exception as exc:
        print(f"[map] WARNING: failed to generate key findings map: {exc}")


def _render_service_kpi_images(
    report_df, polygon_wkt, out_dir: Path, user_id=None,
    grid_lattice=None, grid_size_meters=None,
) -> None:
    """
    Section 7 visuals - reuse existing assets where possible and only
    generate missing service maps/charts/tables.
    """
    from tools.report_engine.kpi_config import dl_colour_manual, ul_colour_manual, mos_colour_manual
    from tools.report_engine import kpi_analysis

    html_dir = out_dir / "html"
    maps_dir = out_dir / "images" / "kpi_maps"
    analysis_dir = out_dir / "images" / "kpi_analysis"

    # DL/UL/MOS maps -- one per technology, never blended, matching the
    # RSRP/RSRQ/SINR treatment in _render_coverage_kpi_images.
    service_specs = [
        ("DL", "dl_tpt", dl_colour_manual, "dl_map", "Mbps"),
        ("UL", "ul_tpt", ul_colour_manual, "ul_map", "Mbps"),
        ("MOS", "mos", mos_colour_manual, "mos_map", ""),
    ]
    for kpi_name, col, color_func, map_prefix, unit in service_specs:
        _render_kpi_map_per_technology(
            report_df, polygon_wkt, grid_lattice, grid_size_meters,
            kpi_name, col, color_func, map_prefix, unit,
            html_dir, maps_dir, user_id=user_id,
        )

    if not _asset_exists(analysis_dir / "network_quality_summary.png"):
        try:
            kpi_analysis.generate_network_quality_summary(report_df)
            print("[map] generated network_quality_summary.png")
        except Exception as exc:
            print(f"[map] WARNING: failed to generate network_quality_summary.png: {exc}")
    else:
        print("[map] reused existing network_quality_summary.png")

    qos_jobs = [
        ("latency", "Latency", "latency", "latency_hist.png"),
        ("jitter", "Jitter", "jitter", "jitter_hist.png"),
        ("speed", "Speed", "speed", "speed_hist.png"),
    ]
    for col, title, prefix, hist_name in qos_jobs:
        if _asset_exists(analysis_dir / hist_name):
            print(f"[map] reused existing {hist_name}")
            continue
        try:
            kpi_analysis.generate_qos_metrics(report_df, col, title, prefix)
            print(f"[map] generated {hist_name}")
        except Exception as exc:
            print(f"[map] WARNING: failed to generate {hist_name}: {exc}")


def main(
    project_id: int,
    user_id: int | None = None,
    report_id: str | None = None,
    db_engine=None,
    region: str | None = None,
    country_code: str | None = None,
    technologies: list[str] | None = None,
    map_view_type: str | None = None,
) -> None:
    """
    Generate the "Per Technology" / new-format PDF report for `project_id`
    and write it to data/new_pdf_reports/<report_id>/report.pdf.

    Mirrors tools/report_engine/main.py's own signature and directory
    convention exactly, but with its own report_id/output namespace
    (data/new_pdf_reports/<report_id>/ instead of data/reports/<report_id>/)
    so the two report types never collide on the same report_id.

    `technologies`: optional exact `network` column values to include (e.g.
    "4G", "4G (LTE Anchor - NSA)", "5G NSA") -- resolved by the caller (the
    frontend enumerates the real distinct values present for this project
    and the user checks the ones they want), not a generation bucket this
    function tries to expand itself. None/empty = every technology kept
    (unchanged default). Applied at the data-fetch layer (load_project_data
    -> GetDriveTestRows' Technologies field / the direct-DB fallback query),
    so a deselected technology is excluded everywhere in this report,
    including handover analysis -- not just from per-technology sections.

    `map_view_type`: "grid" or "raw", controls ONLY how the per-technology
    KPI maps / base route map / poor-region maps are drawn when the project
    has a polygon (has no effect otherwise -- raw points is the only option
    without a polygon). Does NOT affect the grid-cell-median KPI statistics
    used elsewhere in this report (Executive Summary, KPI tables) -- those
    still use the project's real grid lattice whenever a polygon exists,
    regardless of this choice, since that's a statistical-accuracy behavior,
    not a map-rendering style. Defaults to "grid" when a polygon exists and
    this is omitted, matching the report's original always-on-when-polygon
    behavior.
    """
    from tools.report_engine import kpi_analysis
    from tools.report_engine.kpi_analysis import generate_drive_summary_images
    from tools.report_engine.metadata_generator import build_metadata, write_metadata_file

    from tools.New_pdf_report.new_report_sections import (
        NewFormatPDFReport,
        derive_executive_summary,
        generate_introduction_text,
        compute_session_distances_km,
        compute_handover_analysis,
        build_poor_region_summary,
        CDF_ACCEPTANCE_THRESHOLDS,
    )
    from tools.New_pdf_report.grid_maps import (
        resolve_project_grid_size_meters, build_polygon_lattice,
    )

    if db_engine is not None:
        init_engine(db_engine)
    report_id = report_id or str(uuid.uuid4())

    out_dir = Path(f"{DATA_DIR}/tmp/{report_id}")
    html_dir = out_dir / "html"
    images_dir = out_dir / "images"
    kpi_maps_dir = images_dir / "kpi_maps"
    kpi_analysis_dir = images_dir / "kpi_analysis"
    processed_dir = out_dir / "processed"
    report_out_dir = Path(NEW_PDF_REPORTS_DIR) / report_id

    html_dir.mkdir(parents=True, exist_ok=True)
    kpi_maps_dir.mkdir(parents=True, exist_ok=True)
    kpi_analysis_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. LOAD DATA (production loaders / NR-aware filters)
    # --------------------------------------------------
    _, report_df, handover_df, project_meta = _load_report_data(
        project_id, region=region, country_code=country_code, technologies=technologies,
    )
    if report_df.empty:
        raise ValueError(
            f"No report rows found after filtering for project_id={project_id} "
            f"region={region or 'default'} country_code={country_code or 'default'}."
        )
    neighbor_df = _load_neighbor_data(project_meta)
    subsession_df = _load_subsession_data(project_meta)

    # --------------------------------------------------
    # 1b. HANDOVER ANALYSIS (Section 6 data; computed early so both the
    #     image rendering step and the Executive Summary's handover_count
    #     reuse the SAME event set).
    # --------------------------------------------------
    handover_data = compute_handover_analysis(handover_df)
    print(
        f"[handover] band={len(handover_data['band_events'])} "
        f"intra-freq={len(handover_data['intra_freq_events'])} "
        f"inter-RAT={len(handover_data['tech_events'])} "
        f"intra-eNodeB={len(handover_data['intra_enodeb_events'])} "
        f"inter-eNodeB={len(handover_data['inter_enodeb_events'])}"
    )

    # --------------------------------------------------
    # 2. DRIVE SUMMARY METADATA (production)
    # --------------------------------------------------
    kpi_analysis.IMAGE_DIR = str(kpi_analysis_dir)
    session_ids = [
        int(s.strip())
        for s in str(project_meta.get("ref_session_id", "")).split(",")
        if s.strip().isdigit()
    ]
    drive_summary = generate_drive_summary_images(
        session_ids, len(report_df), network_df=report_df
    )
    if drive_summary is None:
        raise ValueError("No drive summary metadata generated")

    # --------------------------------------------------
    # 2b. SESSION DISTANCE (Haversine sum-of-consecutive-points)
    # --------------------------------------------------
    distances_km = compute_session_distances_km(handover_df, session_ids)
    total_distance_km = round(sum(distances_km.values()), 2)
    drive_summary["distance_covered"] = total_distance_km
    for s in drive_summary.get("sessions", []):
        s["distance"] = distances_km.get(int(s["session_id"]), 0.0)
    print(f"[distance] per-session (km): {distances_km}")
    print(f"[distance] total (km): {total_distance_km}")

    # --------------------------------------------------
    # 3. METADATA (production build_metadata: location, area summary
    #    w/ geocoding, band + pci summaries)
    # --------------------------------------------------
    metadata = build_metadata(
        report_df,
        kpi_analysis_results={},
        drive_summary_data=drive_summary,
    )
    write_metadata_file(metadata, str(processed_dir / "report_metadata.json"))
    area_summary = metadata.get("area_summary")

    # --------------------------------------------------
    # 3a. GRID LATTICE (polygon projects only) -- built ONCE here and
    #     shared by every numeric per-technology KPI map below, exactly
    #     like the frontend's own grid view. Uses the project's REAL
    #     tbl_project.grid_size (resolve_project_grid_size_meters) -- no
    #     test-only override.
    # --------------------------------------------------
    polygon_wkt = project_meta.get("region")
    grid_lattice = None
    grid_size_meters = None
    if polygon_wkt:
        grid_size_meters = resolve_project_grid_size_meters(project_meta)
        grid_lattice = build_polygon_lattice(polygon_wkt, grid_size_meters)
        print(
            f"[grid] project {project_id} has a polygon -> grid lattice available "
            f"({grid_size_meters}m cells; {len(grid_lattice)} lattice cells inside the polygon)"
        )
    else:
        print(f"[grid] project {project_id} has no polygon -> raw points only (no grid option)")

    # `map_render_lattice` is what actually gets passed to the MAP-RENDERING
    # helpers below (base route / per-technology KPI / poor-region maps) --
    # deliberately a SEPARATE variable from `grid_lattice` itself, because
    # grid_lattice is also used for KPI STATISTICS (grid-cell-median
    # aggregation in _kpi_stat_series, feeding the Executive Summary/KPI
    # tables via add_coverage_kpi_analysis etc. below) which must stay
    # accurate regardless of which map style the user picked -- only the
    # map images should fall back to raw points when map_view_type="raw",
    # not the underlying KPI numbers. No polygon -> raw is the only option
    # either way, map_view_type has no effect.
    normalized_map_view_type = str(map_view_type or "grid").strip().lower()
    if normalized_map_view_type not in ("grid", "raw"):
        normalized_map_view_type = "grid"
    map_render_lattice = grid_lattice if normalized_map_view_type == "grid" else None
    print(
        f"[grid] map_view_type={normalized_map_view_type!r} -> rendering maps as "
        f"{'GRID' if (polygon_wkt and map_render_lattice is not None) else 'raw points'}"
    )

    # --------------------------------------------------
    # 3b. POOR RSRP / RSRQ LOCATION SUMMARY
    # --------------------------------------------------
    poor_rsrp_summary = build_poor_region_summary(
        report_df, "rsrp", CDF_ACCEPTANCE_THRESHOLDS["RSRP"], grid_lattice=grid_lattice,
    )
    poor_rsrq_summary = build_poor_region_summary(
        report_df, "rsrq", CDF_ACCEPTANCE_THRESHOLDS["RSRQ"], grid_lattice=grid_lattice,
    )

    # --------------------------------------------------
    # 4. BASE ROUTE MAP
    # --------------------------------------------------
    _render_base_route_map(
        report_df, polygon_wkt, out_dir,
        grid_lattice=map_render_lattice, grid_size_meters=grid_size_meters,
    )

    # --------------------------------------------------
    # 4b. COVERAGE KPI IMAGES (Section 4). user_id is the REAL caller-
    #     supplied user, so map color ranges come from the DB's own
    #     configured thresholds (resolve_kpi_ranges -> get_user_thresholds).
    # --------------------------------------------------
    _render_coverage_kpi_images(
        report_df, polygon_wkt, out_dir, user_id=user_id,
        grid_lattice=map_render_lattice, grid_size_meters=grid_size_meters,
    )

    # --------------------------------------------------
    # 4c. MOBILITY KPI IMAGES (Section 5)
    # --------------------------------------------------
    _render_mobility_kpi_images(report_df, project_meta.get("region"), out_dir)

    # --------------------------------------------------
    # 4c2. HANDOVER KPI IMAGES (Section 6)
    # --------------------------------------------------
    _render_handover_kpi_images(
        handover_df, handover_data["band_events"], handover_data["tech_events"],
        project_meta.get("region"), out_dir,
        grid_size_meters=grid_size_meters or 50.0,
    )

    # --------------------------------------------------
    # 4d. SERVICE KPI IMAGES (Section 7). Same real user_id as 4b.
    # --------------------------------------------------
    _render_service_kpi_images(
        report_df, polygon_wkt, out_dir, user_id=user_id,
        grid_lattice=map_render_lattice, grid_size_meters=grid_size_meters,
    )

    # --------------------------------------------------
    # 4e. KEY FINDINGS MAP (Section 8)
    # --------------------------------------------------
    _render_key_findings_map(report_df, polygon_wkt, out_dir)

    # --------------------------------------------------
    # 5. INTRODUCTION (LLM, minimal prompt; production rule-based
    #    fallback synthesizer if the LLM call fails)
    # --------------------------------------------------
    intro_text, intro_source = generate_introduction_text(metadata, report_df)
    if not intro_text or not intro_text.strip():
        raise ValueError("Introduction text is empty")
    print(f"[text] Introduction source: {intro_source}")

    # --------------------------------------------------
    # 6. EXECUTIVE SUMMARY (rule-based)
    # --------------------------------------------------
    handover_count = len(handover_data["band_events"])
    exec_summary = derive_executive_summary(
        report_df, handover_count, mobility_df=report_df, grid_lattice=grid_lattice,
    )
    # Minimum floor: Coverage + Handover (1 row each), plus at least one
    # Radio Quality row and at least one Mobility row -- both are now
    # broken out per technology (see classify_quality_by_technology /
    # classify_mobility_by_technology), so a multi-RAT project produces
    # more than one Radio Quality row and more than one Mobility row, but
    # >= 4 is still the correct floor for a normal single-technology
    # project (1 Coverage + 1 Radio Quality + 1 Mobility + 1 Handover).
    if len(exec_summary["kpi_rows"]) < 4:
        raise ValueError("Executive summary produced fewer than the expected KPI rows")
    if not exec_summary["observations"]:
        raise ValueError("Executive observations are empty")

    # --------------------------------------------------
    # artifacts for review
    # --------------------------------------------------
    band_summary = metadata.get("band_summary")
    (processed_dir / "report_text.json").write_text(
        json.dumps(
            {
                "Introduction": intro_text,
                "Introduction_source": intro_source,
                "Area Summary": area_summary,
                "Drive Summary (new format)": exec_summary,
                "Session distances (km, Haversine consecutive-points)": distances_km,
                "Total distance (km)": total_distance_km,
                "Band Summary": band_summary,
                # Mobility is now broken out per technology (one
                # "Mobility - <tech>" kpi_rows entry per RAT, or a single
                # unlabeled "Mobility" row when `network` is absent) --
                # collect every such row instead of assuming exactly one.
                "Mobility KPI Remarks": {
                    label: remarks
                    for label, _status, remarks in exec_summary["kpi_rows"]
                    if label == "Mobility" or label.startswith("Mobility - ")
                },
                "Neighbor Rows (tbl_network_log_neighbour)": int(len(neighbor_df)),
            },
            indent=2, default=str, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 7. PDF
    # --------------------------------------------------
    pdf_path = report_out_dir / PDF_NAME
    report = NewFormatPDFReport(
        output_path=str(pdf_path),
        images_dir=str(images_dir),
        scratch_dir=str(out_dir / "_img_opt"),
    )
    report.add_cover(metadata)
    report.add_table_of_contents()
    report.add_introduction(intro_text)
    report.add_area_summary(area_summary)
    report.add_drive_summary_v2(drive_summary, exec_summary, distances_km=distances_km)
    report.add_coverage_kpi_analysis(
        report_df, exec_summary, band_summary,
        poor_rsrp_summary=poor_rsrp_summary, poor_rsrq_summary=poor_rsrq_summary,
        grid_lattice=grid_lattice,
    )
    report.add_mobility_kpi_analysis(report_df, neighbor_df=neighbor_df)
    report.add_handover_kpi_analysis(handover_data, subsession_df=subsession_df)
    report.add_service_kpi_analysis(
        report_df, exec_summary=exec_summary, drive_summary=drive_summary,
        handover_data=handover_data, metadata=metadata,
        grid_lattice=grid_lattice,
    )
    report.add_key_findings_section(
        report_df, exec_summary=exec_summary, drive_summary=drive_summary,
        handover_data=handover_data, metadata=metadata,
    )
    report.build()

    size_kb = pdf_path.stat().st_size / 1024 if pdf_path.exists() else 0
    if not pdf_path.exists() or size_kb <= 1:
        raise ValueError("PDF was not generated")
    print(f"[pdf] generated: {pdf_path} ({size_kb:.1f} KB)")

    # --------------------------------------------------
    # 8. DOWNLOAD PATH (tbl_project.Download_path / bridge equivalent) --
    #    disabled on purpose: confirmed nothing in the frontend
    #    (StraceExeFron) reads this field -- the actual download flow the
    #    frontend uses is poll status -> fetch the PDF blob from this
    #    module's own /api/new-pdf-report/download/<report_id> route ->
    #    browser-side save, which does not depend on this DB field at all.
    #    Left here, commented, for later reuse if some other consumer of
    #    tbl_project.Download_path ever needs it for this report type.
    # --------------------------------------------------
    # base_url = os.getenv("BASE_URL", "").rstrip("/")
    # if base_url:
    #     download_link = f"{base_url}/api/new-pdf-report/download/{report_id}"
    # else:
    #     download_link = f"/api/new-pdf-report/download/{report_id}"
    # try:
    #     update_project_download_path(
    #         project_id,
    #         download_link,
    #         region=region,
    #         country_code=country_code,
    #     )
    #     print(f"Updated tbl_project.Download_path: {download_link}")
    # except Exception as e:
    #     print(f"Warning: failed to update Download_path: {e}")

    # Email-on-completion disabled (same as tools/report_engine/main.py --
    # not needed right now; left in place, commented, for later reuse).
    # if user_id is not None:
    #     user_row = get_user_by_id(user_id, region=region, country_code=country_code)
    #     if user_row and user_row.get("email"):
    #         print(f"[Email] Sending report link to: {user_row.get('email')}")
    #         user_name = (
    #             user_row.get("name")
    #             or user_row.get("user_name")
    #             or user_row.get("username")
    #             or "User"
    #         )
    #         project_name = project_meta.get("project_name") or "Project"
    #         try:
    #             send_report_ready_email(
    #                 to_email=user_row["email"],
    #                 user_name=user_name,
    #                 project_name=project_name,
    #                 report_id=report_id,
    #                 download_url=download_link,
    #             )
    #             print("[Email] Sent successfully.")
    #         except Exception as e:
    #             print(f"[Email] Failed to send: {e}")
    #     else:
    #         print(f"Warning: No email found for user_id={user_id}, skipping email send.")
    # else:
    #     print("Warning: user_id not provided, skipping email send.")

    # In production, remove temp files after successful report generation.
    # Keep during local debugging by setting REPORT_KEEP_TMP=1 -- same
    # convention as tools/report_engine/main.py.
    keep_tmp = os.getenv("REPORT_KEEP_TMP", "0") == "1"
    if not keep_tmp:
        clean_directory(str(out_dir))
