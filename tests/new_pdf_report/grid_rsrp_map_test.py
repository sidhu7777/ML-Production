"""
Standalone test case (test-case only -- tools/report_engine is NOT modified
anywhere in this file). Renders exactly ONE PNG: RSRP aggregated into the
SAME grid the frontend's grid view uses, instead of production's raw
per-point scatter markers (generate_kpi_map in map_generator.py).

Ground truth for the grid math (checked directly against the frontend
source, not assumed or invented):
  StraceExeFron/src/components/unifiedMap/MapwithMultipleCircle.jsx
    - generateGridCellsOptimized() default gridSizeMeters = 50 (line 1249)
    - latDegPerMeter = 1 / 111320
    - lngDegPerMeter = 1 / (111320 * cos(avgLat_radians))
    - cellHeight = gridSizeMeters * latDegPerMeter
    - cellWidth  = gridSizeMeters * lngDegPerMeter
  StraceExeFron/src/components/map/tools/DrawingToolsLayer.jsx:172
    - the Drawing Tools rectangle/polygon "pixelate" grid colors each cell
      via pickColorForValue(cellStats.mean, ...) -- i.e. AVERAGE, not
      median (computeStats() at line 43 computes both, but only .mean is
      used for cell coloring). This script matches that: per-cell average.
This script reproduces that exact bucketing + aggregation in Python.

Coloring/legend reuse production's OWN resolve_kpi_ranges / value_in_range /
build_legend_from_ranges (tools/report_engine, unchanged) so the grid map's
color buckets match the existing raw-point RSRP map bucket-for-bucket --
this is a rendering comparison, not a new/invented color scale.

Run directly with (from the ML/ directory):
    venv/Scripts/python.exe -m tests.new_pdf_report.grid_rsrp_map_test
"""
import math
from pathlib import Path

import folium
import pandas as pd

from tools.report_engine.map_generator import (
    new_report_map, add_fullscreen_css, draw_polygon_overlay,
    fit_data_bounds, add_legend, value_in_range, build_legend_from_ranges,
)
from tools.report_engine.threshold_resolver import resolve_kpi_ranges
from tools.report_engine.playwright_utils import html_to_png

GRID_SIZE_METERS = 50   # frontend default (MapwithMultipleCircle.jsx:1249)
AGGREGATION = "average"  # DrawingToolsLayer.jsx:172 colors cells by cellStats.mean


def _color_for_value(value, ranges: list) -> str:
    for idx, r in enumerate(ranges):
        is_last = idx == len(ranges) - 1
        if value_in_range(value, r, is_last):
            return r["color"]
    return "#808080"


def build_rsrp_grid_cells(df: pd.DataFrame, grid_size_meters: float = GRID_SIZE_METERS) -> pd.DataFrame:
    """
    Buckets df's (lat, lon, rsrp) rows into square grid cells using the
    identical meters-to-degrees math generateGridCellsOptimized uses, then
    aggregates each cell's rsrp with the average (matching
    DrawingToolsLayer.jsx:172's cellStats.mean cell-coloring). One row per
    non-empty cell.
    """
    data = df.dropna(subset=["lat", "lon", "rsrp"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["south", "west", "north", "east", "rsrp_avg", "sample_count"])

    data["lat"] = data["lat"].astype(float)
    data["lon"] = data["lon"].astype(float)
    data["rsrp"] = data["rsrp"].astype(float)

    south, north = data["lat"].min(), data["lat"].max()
    west = data["lon"].min()
    avg_lat_rad = math.radians((north + south) / 2.0)

    lat_deg_per_meter = 1 / 111320
    lng_deg_per_meter = 1 / (111320 * math.cos(avg_lat_rad))
    cell_height = grid_size_meters * lat_deg_per_meter
    cell_width = grid_size_meters * lng_deg_per_meter

    data["row_idx"] = ((data["lat"] - south) / cell_height).apply(math.floor)
    data["col_idx"] = ((data["lon"] - west) / cell_width).apply(math.floor)

    grouped = data.groupby(["row_idx", "col_idx"])["rsrp"].agg(["mean", "count"]).reset_index()
    grouped["south"] = south + grouped["row_idx"] * cell_height
    grouped["north"] = grouped["south"] + cell_height
    grouped["west"] = west + grouped["col_idx"] * cell_width
    grouped["east"] = grouped["west"] + cell_width
    grouped = grouped.rename(columns={"mean": "rsrp_avg", "count": "sample_count"})
    return grouped[["south", "west", "north", "east", "rsrp_avg", "sample_count"]]


def generate_rsrp_grid_map(
    df: pd.DataFrame,
    ranges: list,
    output_html: str,
    polygon_wkt: str | None = None,
    grid_size_meters: float = GRID_SIZE_METERS,
) -> pd.DataFrame:
    cells = build_rsrp_grid_cells(df, grid_size_meters)
    if cells.empty:
        raise ValueError("No data available to build RSRP grid cells")

    m = new_report_map()
    add_fullscreen_css(m)

    for _, cell in cells.iterrows():
        color = _color_for_value(cell["rsrp_avg"], ranges)
        folium.Rectangle(
            bounds=[(cell["south"], cell["west"]), (cell["north"], cell["east"])],
            color=color, weight=0, fill=True, fill_color=color, fill_opacity=0.85,
            tooltip=f"RSRP average: {cell['rsrp_avg']:.1f} dBm ({int(cell['sample_count'])} samples)",
        ).add_to(m)

    legend_items = build_legend_from_ranges(pd.DataFrame({"rsrp": cells["rsrp_avg"]}), "rsrp", ranges)
    add_legend(m, f"rsrp (grid, {int(grid_size_meters)}m cells, {AGGREGATION})", legend_items)

    draw_polygon_overlay(m, polygon_wkt)
    fit_data_bounds(m, df.dropna(subset=["lat", "lon"]), reserve_legend_space=True)

    m.save(output_html)
    return cells


if __name__ == "__main__":
    from tests.new_pdf_report.test_new_pdf_report import _load_report_data, DEFAULT_THRESHOLD_USER_ID

    project_id = 248
    _, report_df, _, project_meta = _load_report_data(project_id)
    polygon_wkt = project_meta.get("region")

    out_dir = Path(__file__).parent / "output" / f"project_{project_id}"
    html_dir = out_dir / "html"
    image_dir = out_dir / "images" / "grid_maps"
    html_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    ranges = resolve_kpi_ranges(kpi_name="RSRP", user_id=DEFAULT_THRESHOLD_USER_ID, values=report_df["rsrp"])

    html_path = html_dir / "rsrp_grid_map.html"
    png_path = image_dir / "rsrp_grid_map.png"

    cells = generate_rsrp_grid_map(report_df, ranges, str(html_path), polygon_wkt=polygon_wkt)
    html_to_png(str(html_path), str(png_path), width=1200, height=900, device_scale_factor=1)

    print(f"[grid] built {len(cells)} RSRP grid cells ({GRID_SIZE_METERS}m cells, {AGGREGATION} aggregation)")
    print(f"[grid] image saved: {png_path}")
