"""
Standalone test case (test-case only -- tools/report_engine is NOT modified
anywhere in this file). Renders RSRP aggregated into the SAME grid the
frontend's grid view uses for a POLYGON project, instead of production's raw
per-point scatter markers (generate_kpi_map in map_generator.py).

Ground truth for the grid math and defaults (checked directly against the
frontend source, not assumed or invented):
  StraceExeFron/src/pages/UnifiedMapView.jsx
    - canEnableUnifiedGridView = hasFilteringPolygons (line 2565): grid view
      is only ever possible when the project has a filtering polygon. No
      polygon -> the frontend never shows a grid, only raw points. This is
      why this script must run against a POLYGON project (292), not 248
      (248 has no polygon and no grid_size at all -- confirmed via
      tbl_project, see below -- so grid view could never trigger for it).
    - Grid view auto-enables (line 2576-2586) once the project has both a
      filtering polygon AND a `grid_size` value stored in the DB
      (tbl_project.grid_size, see Signal-Trackers-App/Models/EntityModel.cs
      -- `tbl_project.grid_size`). Grid cell size is therefore PER-PROJECT
      DB data, not a fixed constant -- confirmed project 292's grid_size is
      100 (not the 50m this script used to hardcode), project 248's is None.
    - Default aggregation for that same polygon-driven grid overlay is
      "median" (`gridAggregationMethod={lteGridAggregationMethod || "median"}`
      at line 7000, and `useState("median")` at line 1880-1881) -- NOT
      "average".
  StraceExeFron/src/components/unifiedMap/MapwithMultipleCircle.jsx
    - generateGridCellsOptimized() (line 598-937) -- TWO details this script
      previously got wrong, both fixed now:
        1. The lattice ORIGIN is the POLYGON's own bounding box
           (`globalBounds`, built from each polygon's `bbox` at line
           617-628, itself `computeBbox()` of the polygon's exterior ring
           -- StraceExeFron/src/utils/wkt.js:116-127), NOT the data points'
           bounding box. That lattice is computed ONCE and shared by every
           technology/metric filter. This script previously recomputed
           south/west from each per-technology data subset, so every
           technology got its OWN misaligned lattice -- always wrong
           relative to the frontend, and inconsistent between technologies
           in the same report.
        2. A lattice cell is only kept if its CENTER point is inside the
           polygon (`checker.isInside(centerLat, centerLng)`, line 747) --
           this is checked BEFORE looking at whether the cell has any data,
           and cells failing it are skipped entirely, even if they contain
           data points. This script previously had no such filter at all,
           so it counted every data-populated bucket regardless of whether
           that bucket's center actually sits inside the polygon shape --
           inflating cell counts, especially along a polygon's edges.
      generateGridCellsOptimized keeps EVERY polygon-interior cell in its
      returned array, populated or not (`cells.push(...)` at line 923 is
      unconditional). The frontend then reports two different counts
      (MapwithMultipleCircle.jsx:1865-1876, `onGridCellsStatsChange`):
        total     = visibleGridCells.length      (all polygon-interior cells)
        populated = cells with count > 0          (cells with actual data)
      "total" is identical for every technology filter (it only depends on
      the polygon + grid size, never on which rows are selected); only
      "populated" changes per technology. This script now reports both, per
      technology AND combined, instead of a single ambiguous number, so its
      output is directly comparable to what the frontend's own grid-cell
      stats show.
    - AGGREGATION_METHODS.median (line 344-348): standard sorted-midpoint
      median, matched exactly in Python below (no interpolation tricks).

KNOWN, INTENTIONAL count difference vs the frontend's raw grid stats --
confirmed root cause, not a bug (verified against project 292's actual DB
rows, not assumed):
  This script's raw per-technology sample counts (e.g. 1009 for "5G NSA")
  are lower than what the frontend's grid view counts (e.g. 1287 for the
  same technology) because report_df here has already been through
  production's OWN filter_known_band_rows() (tools/report_engine/
  load_data_db.py, unchanged) -- the same filter every other table/map in
  this report is built on (Band Distribution, RSRQ/SINR-per-technology,
  etc.). For project 292, exactly 278 of the 1287 raw "5G NSA" rows have
  band="N/A" (the device reported no band for those samples) ->
  normalize_band_name() maps that to "Unknown" -> filter_known_band_rows()
  drops them, leaving 1009. The frontend's grid apparently does NOT apply
  this same band filter, so it keeps all 1287. That 278-row gap is why this
  script's "5G NSA" populated-cell count (78) is lower than the frontend's
  (85) -- most of those extra band-less points land in cells that already
  have other 5G NSA points, but a handful land in otherwise-empty cells.
  Per explicit direction received: keep filter_known_band_rows applied here
  too, so this grid map stays on the exact same row population as every
  other KPI table/map in the report, rather than special-casing this one
  map onto a different (unfiltered) row set. Do not "fix" this gap by
  removing the band filter without re-confirming that direction.

Coloring/legend reuse production's OWN resolve_kpi_ranges / value_in_range /
build_legend_from_ranges (tools/report_engine, unchanged) so the grid map's
color buckets match the existing raw-point RSRP map bucket-for-bucket --
this is a rendering comparison, not a new/invented color scale. Ranges are
resolved once from the FULL (all-technology) RSRP distribution, matching
test_new_pdf_report.py's own per-technology RSRP maps (Section 4), so a
per-technology grid map's colors stay comparable across technologies. Only
POPULATED cells are drawn/legended, matching the frontend
(`visibleGridCells.filter(cell => cell.count > 0)`, line 1888-1889) -- empty
polygon-interior cells are tracked for the "total" stat but never rendered.

Image pixel size (1200x900, device_scale_factor=1) matches every other
map screenshot test_new_pdf_report.py takes -- unchanged here.

Run directly with (from the ML/ directory):
    venv-win/Scripts/python.exe -m tests.new_pdf_report.grid_rsrp_map_test
"""
import math
from pathlib import Path

import folium
import pandas as pd
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point

from tools.report_engine.map_generator import (
    new_report_map, add_fullscreen_css, draw_polygon_overlay,
    add_legend, value_in_range, build_legend_from_ranges,
    get_df_bounds, merge_bounds,
    REPORT_PAD_LEFT, REPORT_PAD_VERT, REPORT_PAD_RIGHT_BASE,
    REPORT_PAD_RIGHT_LEGEND, REPORT_MAP_MAX_ZOOM,
)
from tools.report_engine.threshold_resolver import resolve_kpi_ranges
from tools.report_engine.playwright_utils import html_to_png

FALLBACK_GRID_SIZE_METERS = 50  # only used if tbl_project.grid_size is missing/invalid
AGGREGATION = "median"  # UnifiedMapView.jsx:1880-1881/7000 default for the polygon-driven grid


def resolve_project_grid_size_meters(project_meta: dict) -> float:
    """
    Mirrors UnifiedMapView.jsx's `projectAreaGridSizeMeters` (line 2150-2156):
    read tbl_project.grid_size for THIS project, falling back only when it's
    missing or not a usable positive number.
    """
    raw = (project_meta or {}).get("grid_size")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return FALLBACK_GRID_SIZE_METERS
    return value if value > 0 else FALLBACK_GRID_SIZE_METERS


def _color_for_value(value, ranges: list) -> str:
    for idx, r in enumerate(ranges):
        is_last = idx == len(ranges) - 1
        if value_in_range(value, r, is_last):
            return r["color"]
    return "#808080"


def _median(values: pd.Series) -> float:
    sorted_vals = values.sort_values().to_numpy()
    n = len(sorted_vals)
    mid = n // 2
    return float(sorted_vals[mid]) if n % 2 else float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0)


def build_polygon_lattice(polygon_wkt: str, grid_size_meters: float) -> pd.DataFrame:
    """
    The SHARED grid lattice for this polygon+grid_size -- computed ONCE,
    reused for every technology/metric, exactly matching
    generateGridCellsOptimized (MapwithMultipleCircle.jsx:598-937):
      - origin/extent from the POLYGON's own bounding box (computeBbox of
        its exterior ring), not from any data points.
      - a cell is kept only if ITS CENTER lies inside the polygon
        (ray-casting containment, matching PolygonChecker.isInside).
    Returns one row per KEPT cell (this is the frontend's "total" count),
    with columns: row_idx, col_idx, south, west, north, east,
    center_lat, center_lon. Also returns cell_height/cell_width via
    DataFrame.attrs so callers can bucket data points with the identical
    cell size without recomputing it.
    """
    polygon = load_wkt(polygon_wkt)
    west, south, east, north = polygon.bounds  # shapely: (minx, miny, maxx, maxy) = (west, south, east, north)

    avg_lat_rad = math.radians((north + south) / 2.0)
    lat_deg_per_meter = 1 / 111320
    lng_deg_per_meter = 1 / (111320 * math.cos(avg_lat_rad))
    cell_height = grid_size_meters * lat_deg_per_meter
    cell_width = grid_size_meters * lng_deg_per_meter

    num_rows = max(1, math.ceil((north - south) / cell_height))
    num_cols = max(1, math.ceil((east - west) / cell_width))

    rows = []
    for row_idx in range(num_rows):
        cell_south = south + row_idx * cell_height
        cell_north = cell_south + cell_height
        center_lat = (cell_south + cell_north) / 2.0
        for col_idx in range(num_cols):
            cell_west = west + col_idx * cell_width
            cell_east = cell_west + cell_width
            center_lon = (cell_west + cell_east) / 2.0
            if not polygon.contains(Point(center_lon, center_lat)):
                continue
            rows.append((row_idx, col_idx, cell_south, cell_west, cell_north, cell_east, center_lat, center_lon))

    lattice = pd.DataFrame(rows, columns=["row_idx", "col_idx", "south", "west", "north", "east", "center_lat", "center_lon"])
    lattice.attrs["south"] = south
    lattice.attrs["west"] = west
    lattice.attrs["cell_height"] = cell_height
    lattice.attrs["cell_width"] = cell_width
    return lattice


def aggregate_grid_cells(df: pd.DataFrame, lattice: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, int, int]:
    """
    Buckets df's (lat, lon, value_col) rows onto the SHARED `lattice` (same
    row_idx/col_idx grid every technology AND every KPI uses) and
    aggregates each populated cell with the MEDIAN. Returns
    (populated_cells_df, total, populated) where `total`/`populated` match
    the frontend's own onGridCellsStatsChange stat
    (MapwithMultipleCircle.jsx:1865-1876). Generic over `value_col` (rsrp,
    rsrq, sinr, dl_tpt, ul_tpt, mos, ...) -- the output column is always
    named "value_agg" regardless of which KPI was aggregated.
    """
    empty = pd.DataFrame(columns=["row_idx", "col_idx", "south", "west", "north", "east", "value_agg", "sample_count"])
    if lattice.empty:
        return empty, 0, 0

    south, west = lattice.attrs["south"], lattice.attrs["west"]
    cell_height, cell_width = lattice.attrs["cell_height"], lattice.attrs["cell_width"]

    data = df.dropna(subset=["lat", "lon", value_col]).copy()
    total = len(lattice)
    if data.empty:
        return empty, total, 0

    data["lat"] = data["lat"].astype(float)
    data["lon"] = data["lon"].astype(float)
    data[value_col] = data[value_col].astype(float)
    data["row_idx"] = ((data["lat"] - south) / cell_height).apply(math.floor).astype(int)
    data["col_idx"] = ((data["lon"] - west) / cell_width).apply(math.floor).astype(int)

    # Only rows landing in a lattice cell whose CENTER is inside the polygon
    # count -- an inner merge against `lattice` enforces exactly that (the
    # same effect as generateGridCellsOptimized's isInside gate, applied
    # before any per-cell aggregation).
    matched = data.merge(lattice[["row_idx", "col_idx", "south", "west", "north", "east"]], on=["row_idx", "col_idx"], how="inner", suffixes=("", "_cell"))
    if matched.empty:
        return empty, total, 0

    grouped = matched.groupby(["row_idx", "col_idx"])[value_col].agg(value_agg=_median, sample_count="count").reset_index()
    grouped = grouped.merge(lattice[["row_idx", "col_idx", "south", "west", "north", "east"]], on=["row_idx", "col_idx"], how="left")
    populated = len(grouped)
    return grouped[["row_idx", "col_idx", "south", "west", "north", "east", "value_agg", "sample_count"]], total, populated


def fit_bounds_including_polygon(m, bounds_df: pd.DataFrame, polygon_wkt: str | None, reserve_legend_space: bool = True) -> None:
    """
    Shared viewport-fitting rule for every map in the new-format report:
    fit to the polygon's OWN bounds when the project has one, else fall
    back to the GPS data bounds (production's original fit_data_bounds
    behaviour). Used by every map-rendering function in this report (grid
    KPI maps, Band/PCI maps, base route map, poor-region maps, handover
    maps) via new_report_sections.py, not just the grid maps.

    Production's OWN fit_data_bounds() (map_generator.py:42-51) fits the
    viewport to the GPS DATA bounds ONLY, by its own docstring's admission:
    "the polygon is drawn as an overlay but does NOT expand the view".
    That's fine when there's no polygon, but once a project HAS one, the
    polygon -- not wherever GPS samples happened to reach -- defines the
    analysis area, and a project's polygon commonly extends a bit further
    than the drive route itself. Fitting to data-only bounds then clips
    the polygon outline (and, for grid maps, can crop lattice cells near
    its edge) at the screenshot boundary -- confirmed against a real
    project polygon (292's southern tip sat within ~20px of the 900px-tall
    screenshot's bottom edge, and ~2km/~20% narrower on its west edge,
    before this fix).

    Fix: union the data bounds with the polygon's own bounds (shapely
    .bounds) using production's own merge_bounds() (map_generator.py:220-224,
    previously unused for this) before fitting. Since polygon-filtered data
    is always a subset of the polygon, this union is equivalent to "polygon
    bounds when present, else data bounds" -- it's written as a union
    rather than a hard replacement only so a project whose data somehow
    exceeds a stale/misconfigured polygon still gets a frame that includes
    all of it, not a frame that silently clips real data.
    """
    bounds = get_df_bounds(bounds_df)
    if polygon_wkt:
        polygon = load_wkt(polygon_wkt)
        west, south, east, north = polygon.bounds
        bounds = merge_bounds(bounds, [[south, west], [north, east]])
    right = REPORT_PAD_RIGHT_LEGEND if reserve_legend_space else REPORT_PAD_RIGHT_BASE
    m.fit_bounds(
        bounds,
        padding_top_left=(REPORT_PAD_LEFT, REPORT_PAD_VERT),
        padding_bottom_right=(right, REPORT_PAD_VERT),
        max_zoom=REPORT_MAP_MAX_ZOOM,
    )


def generate_kpi_grid_map(
    cells: pd.DataFrame,
    ranges: list,
    output_html: str,
    polygon_wkt: str | None,
    grid_size_meters: float,
    bounds_df: pd.DataFrame,
    metric_label: str = "rsrp",
    unit: str = "dBm",
    total_cells: int | None = None,
) -> None:
    """`bounds_df` fits the viewport to the full (all-technology) route so a
    per-technology grid map frames the same area as its raw-point
    counterpart in Section 4/7. `metric_label`/`unit` only affect the
    tooltip/legend text (e.g. "sinr (grid, 100m cells, median)",
    "SINR median: 12.3 dB") -- the aggregation itself is identical for
    every KPI (see aggregate_grid_cells).

    `total_cells` is the TOTAL number of polygon-interior lattice cells
    (populated or not -- the same "total" aggregate_grid_cells returns,
    identical across every technology since it only depends on the
    polygon + grid size), added to the legend alongside this
    technology's own populated-cell count so a reader can see e.g.
    "Total Grid Cells: 4442" next to "Populated (this technology): 167"
    instead of only ever seeing the per-range breakdown of the populated
    subset with no sense of how big the full grid is. Omitted from the
    legend when not supplied.
    """
    m = new_report_map()
    add_fullscreen_css(m)

    for _, cell in cells.iterrows():
        color = _color_for_value(cell["value_agg"], ranges)
        folium.Rectangle(
            bounds=[(cell["south"], cell["west"]), (cell["north"], cell["east"])],
            color=color, weight=0, fill=True, fill_color=color, fill_opacity=0.85,
            tooltip=f"{metric_label.upper()} median: {cell['value_agg']:.1f} {unit} ({int(cell['sample_count'])} samples)",
        ).add_to(m)

    legend_items = build_legend_from_ranges(pd.DataFrame({metric_label: cells["value_agg"]}), metric_label, ranges)
    if total_cells is not None:
        legend_items = [
            ("Total Grid Cells", "#6b7280", total_cells),
            ("Populated (this technology)", "#374151", len(cells)),
            *legend_items,
        ]
    add_legend(m, f"{metric_label} (grid, {int(grid_size_meters)}m cells, {AGGREGATION})", legend_items)

    draw_polygon_overlay(m, polygon_wkt)
    fit_bounds_including_polygon(m, bounds_df.dropna(subset=["lat", "lon"]), polygon_wkt, reserve_legend_space=True)

    m.save(output_html)


if __name__ == "__main__":
    from tests.new_pdf_report.test_new_pdf_report import _load_report_data, DEFAULT_THRESHOLD_USER_ID
    from tests.new_pdf_report.new_report_sections import _technology_groups, _tech_slug
    from tools.report_engine.map_generator import has_valid_numeric_data

    project_id = 292  # same drive route as 248, but WITH a project polygon
    _, report_df, _, project_meta = _load_report_data(project_id)
    polygon_wkt = project_meta.get("region")
    if not polygon_wkt:
        raise RuntimeError(
            f"project {project_id} has no polygon -- grid view never applies "
            f"per the frontend's canEnableUnifiedGridView check; pick a project with one."
        )

    grid_size_meters = resolve_project_grid_size_meters(project_meta)
    print(f"[grid] project {project_id} grid_size (tbl_project.grid_size): {project_meta.get('grid_size')!r}"
          f" -> using {grid_size_meters}m cells")

    # Built ONCE from the polygon, shared by every technology below --
    # this "total" must be identical no matter which technology is filtered.
    lattice = build_polygon_lattice(polygon_wkt, grid_size_meters)
    print(f"[grid] polygon lattice: {len(lattice)} cells have a center inside the polygon "
          f"(this is the frontend's \"total\" stat -- constant across all technologies)")

    out_dir = Path(__file__).parent / "output" / f"project_{project_id}"
    html_dir = out_dir / "html"
    image_dir = out_dir / "images" / "grid_maps"
    html_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Ranges resolved once from the FULL (all-technology) RSRP distribution,
    # same convention as test_new_pdf_report.py's per-technology raw maps.
    ranges = resolve_kpi_ranges(kpi_name="RSRP", user_id=DEFAULT_THRESHOLD_USER_ID, values=report_df["rsrp"])

    tech_list = _technology_groups(report_df)
    network_col = (
        report_df["network"].fillna("").astype(str).str.strip()
        if "network" in report_df.columns else None
    )
    techs = tech_list if (tech_list and network_col is not None) else [None]

    print(f"\n{'Technology':<28}{'Raw samples':>12}{'Total cells':>13}{'Populated':>11}")
    print("-" * 64)

    generated = 0
    for tech in techs:
        df_tech = report_df if tech is None else report_df.loc[network_col == tech]
        df_kpi = df_tech[df_tech["rsrp"].notna() & df_tech["lat"].notna() & df_tech["lon"].notna()]
        tech_label = tech or "ALL"
        raw_count = len(df_kpi)
        if df_kpi.empty or not has_valid_numeric_data(df_tech, "rsrp"):
            print(f"{tech_label:<28}{raw_count:>12}{'--':>13}{'skipped (no data)':>18}")
            continue

        cells, total, populated = aggregate_grid_cells(df_kpi, lattice, value_col="rsrp")
        print(f"{tech_label:<28}{raw_count:>12}{total:>13}{populated:>11}")
        if cells.empty:
            continue

        slug = _tech_slug(tech) if tech else "all"
        html_path = html_dir / f"rsrp_grid_map_{slug}.html"
        png_path = image_dir / f"rsrp_grid_map_{slug}.png"

        generate_kpi_grid_map(
            cells, ranges, str(html_path), polygon_wkt=polygon_wkt,
            grid_size_meters=grid_size_meters, bounds_df=report_df,
            metric_label="rsrp", unit="dBm",
        )
        html_to_png(str(html_path), str(png_path), width=1200, height=900, device_scale_factor=1)
        print(f"   -> {png_path}")
        generated += 1

    # Combined (all technologies together) -- for comparison against
    # whatever the frontend shows with no technology filter applied.
    df_all = report_df[report_df["rsrp"].notna() & report_df["lat"].notna() & report_df["lon"].notna()]
    cells_all, total_all, populated_all = aggregate_grid_cells(df_all, lattice, value_col="rsrp")
    print("-" * 64)
    print(f"{'ALL (combined)':<28}{len(df_all):>12}{total_all:>13}{populated_all:>11}")

    if not generated:
        raise RuntimeError("No RSRP grid maps were generated -- check report_df has rsrp/lat/lon data")
