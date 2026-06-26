import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import folium
from shapely.wkt import loads

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.report_engine.load_data_db import load_project_data
from tools.report_engine.kpi_config import KPI_CONFIG
from tools.report_engine.map_generator import (
    add_fullscreen_css,
    add_legend,
    build_legend_from_ranges,
    get_df_bounds,
    value_in_range,
)
from tools.report_engine.playwright_utils import html_to_png
from tools.report_engine.threshold_resolver import resolve_kpi_ranges


def _make_run_dir(base_dir: Path, project_id: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"project_{project_id}_base_map_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _parse_session_ids(value) -> list[int]:
    return [int(s.strip()) for s in str(value or "").split(",") if s.strip().isdigit()]


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _route_summary(df: pd.DataFrame, polygon_wkt: str | None) -> dict:
    geo_df = df.dropna(subset=["lat", "lon"])
    bounds = get_df_bounds(geo_df) if not geo_df.empty else None
    return {
        "rows": int(len(df)),
        "valid_geo_rows": int(len(geo_df)),
        "has_polygon": bool(polygon_wkt),
        "bounds": bounds,
    }


def _fit_tight_bounds(
    m,
    bounds,
    *,
    left_padding_px: int,
    right_padding_px: int,
    top_padding_px: int,
    bottom_padding_px: int,
) -> None:
    m.fit_bounds(
        bounds,
        padding_top_left=(left_padding_px, top_padding_px),
        padding_bottom_right=(right_padding_px, bottom_padding_px),
        max_zoom=22,
    )


def _generate_base_route_map_with_tile(
    df: pd.DataFrame,
    polygon_wkt: str | None,
    output_html: Path,
    tile_name: str,
    left_padding_px: int,
    right_padding_px: int,
    top_padding_px: int,
    bottom_padding_px: int,
) -> None:
    df = df.dropna(subset=["lat", "lon"])
    if df.empty:
        raise ValueError("No GPS data to plot for base route map")

    m = folium.Map(
        tiles=tile_name,
        zoom_control=True,
        control_scale=False,
        prefer_canvas=True,
        max_zoom=22,
    )
    add_fullscreen_css(m)

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=(r["lat"], r["lon"]),
            radius=4,
            color="#2b8cbe",
            fill=True,
            fill_opacity=0.95,
        ).add_to(m)

    if polygon_wkt:
        geom = loads(polygon_wkt)
        polygon_latlon = [(coord[1], coord[0]) for coord in geom.exterior.coords]
        folium.Polygon(
            locations=polygon_latlon,
            color="red",
            weight=4,
            fill=False,
            opacity=1.0,
            tooltip="Polygon Boundary",
        ).add_to(m)

    _fit_tight_bounds(
        m,
        get_df_bounds(df),
        left_padding_px=left_padding_px,
        right_padding_px=right_padding_px,
        top_padding_px=top_padding_px,
        bottom_padding_px=bottom_padding_px,
    )
    m.save(str(output_html))


def _generate_range_kpi_map_with_tile(
    df: pd.DataFrame,
    kpi_column: str,
    ranges: list[dict],
    polygon_wkt: str | None,
    output_html: Path,
    tile_name: str,
    left_padding_px: int,
    legend_right_padding_px: int,
    top_padding_px: int,
    bottom_padding_px: int,
    legend_top_px: int,
    legend_right_px: int,
) -> None:
    df = df.dropna(subset=["lat", "lon", kpi_column])
    if df.empty:
        raise ValueError(f"No data available for KPI map: {kpi_column}")

    m = folium.Map(
        tiles=tile_name,
        zoom_control=True,
        control_scale=False,
        prefer_canvas=True,
        max_zoom=22,
    )
    add_fullscreen_css(m)

    for _, row in df.iterrows():
        try:
            value = float(row[kpi_column])
        except (TypeError, ValueError):
            continue

        color = "#808080"
        for idx, range_def in enumerate(ranges):
            if value_in_range(value, range_def, idx == len(ranges) - 1):
                color = range_def["color"]
                break

        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.9,
        ).add_to(m)

    legend_items = build_legend_from_ranges(df, kpi_column, ranges)
    add_legend(m, kpi_column, legend_items)
    m.get_root().header.add_child(folium.Element(f"""
    <style>
        .kpi-legend {{
            top: {legend_top_px}px !important;
            right: {legend_right_px}px !important;
        }}
    </style>
    """))

    if polygon_wkt:
        geom = loads(polygon_wkt)
        polygon_latlon = [(coord[1], coord[0]) for coord in geom.exterior.coords]
        folium.Polygon(
            locations=polygon_latlon,
            color="#FF0000",
            weight=5,
            fill=False,
            opacity=1.0,
            tooltip="Polygon Boundary",
        ).add_to(m)

    _fit_tight_bounds(
        m,
        get_df_bounds(df),
        left_padding_px=left_padding_px,
        right_padding_px=legend_right_padding_px,
        top_padding_px=top_padding_px,
        bottom_padding_px=bottom_padding_px,
    )
    m.save(str(output_html))


def generate_project_base_route_image(
    project_id: int,
    user_id: int | None,
    out_dir: Path,
    render_timeout_ms: int,
    candidate_width: int,
    candidate_height: int,
    tile_name: str,
    kpi_name: str,
    left_padding_px: int,
    base_right_padding_px: int,
    legend_right_padding_px: int,
    top_padding_px: int,
    bottom_padding_px: int,
    legend_top_px: int,
    legend_right_px: int,
) -> Path:
    """
    Real report base-map debug case.

    This intentionally imports the production loader and base-map renderer only.
    It does not modify production code and does not create synthetic/static images.
    """
    os.environ.setdefault("REPORT_RENDER_TIMEOUT_MS", str(render_timeout_ms))
    os.environ.setdefault("REPORT_RENDER_NAV_ATTEMPTS", "1")

    run_dir = _make_run_dir(out_dir, project_id)
    html_dir = run_dir / "html"
    image_dir = run_dir / "images" / "kpi_maps"
    processed_dir = run_dir / "processed"
    html_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_df, filtered_df, project_meta = load_project_data(project_id)
    polygon_wkt = project_meta.get("region")

    if filtered_df.empty:
        raise ValueError(f"Project {project_id} has no rows after report filtering.")

    safe_tile_name = tile_name.lower().replace(" ", "_")
    html_path = html_dir / f"base_route_map_{safe_tile_name}.html"
    compact_png_path = image_dir / f"base_route_map_{safe_tile_name}_compact.png"
    safe_kpi_name = kpi_name.lower()
    kpi_html_path = html_dir / f"{safe_kpi_name}_map_{safe_tile_name}.html"
    kpi_compact_png_path = image_dir / f"{safe_kpi_name}_map_{safe_tile_name}_compact.png"

    _generate_base_route_map_with_tile(
        filtered_df,
        polygon_wkt,
        html_path,
        tile_name,
        left_padding_px=left_padding_px,
        right_padding_px=base_right_padding_px,
        top_padding_px=top_padding_px,
        bottom_padding_px=bottom_padding_px,
    )
    html_to_png(
        str(html_path),
        str(compact_png_path),
        width=candidate_width,
        height=candidate_height,
    )

    kpi_cfg = KPI_CONFIG.get(kpi_name)
    if not kpi_cfg or kpi_cfg.get("type") != "range":
        raise ValueError(f"Only range KPI maps are supported in this debug script: {kpi_name}")

    kpi_column = kpi_cfg["column"]
    kpi_df = filtered_df[
        filtered_df[kpi_column].notna()
        & filtered_df["lat"].notna()
        & filtered_df["lon"].notna()
    ]
    ranges = resolve_kpi_ranges(kpi_name=kpi_name, user_id=user_id, values=kpi_df[kpi_column])
    _generate_range_kpi_map_with_tile(
        kpi_df,
        kpi_column=kpi_column,
        ranges=ranges,
        polygon_wkt=polygon_wkt,
        output_html=kpi_html_path,
        tile_name=tile_name,
        left_padding_px=left_padding_px,
        legend_right_padding_px=legend_right_padding_px,
        top_padding_px=top_padding_px,
        bottom_padding_px=bottom_padding_px,
        legend_top_px=legend_top_px,
        legend_right_px=legend_right_px,
    )
    html_to_png(
        str(kpi_html_path),
        str(kpi_compact_png_path),
        width=candidate_width,
        height=candidate_height,
    )

    raw_df.to_csv(processed_dir / "raw_data.csv", index=False)
    filtered_df.to_csv(processed_dir / "filtered_data_used_for_base_map.csv", index=False)
    _write_json(processed_dir / "project_meta.json", project_meta)

    summary = {
        "project_id": project_id,
        "user_id": user_id,
        "session_ids": _parse_session_ids(project_meta.get("ref_session_id")),
        "polygon_source": "project_meta.region",
        "raw": _route_summary(raw_df, polygon_wkt),
        "filtered_used_for_image": _route_summary(filtered_df, polygon_wkt),
        "tile_name": tile_name,
        "kpi_name": kpi_name,
        "kpi_column": kpi_column,
        "padding_px": {
            "left": left_padding_px,
            "base_right": base_right_padding_px,
            "kpi_legend_right": legend_right_padding_px,
            "top": top_padding_px,
            "bottom": bottom_padding_px,
            "geographic_expand_factor": 0,
        },
        "legend_position_px": {
            "top": legend_top_px,
            "right": legend_right_px,
        },
        "compact_viewport": {
            "width": candidate_width,
            "height": candidate_height,
            "png_path": str(compact_png_path),
            "why": (
                "Compact landscape preview keeps the route large, with the map "
                "content filling the frame instead of reserving empty margins."
            ),
        },
        "kpi_compact_viewport": {
            "width": candidate_width,
            "height": candidate_height,
            "png_path": str(kpi_compact_png_path),
            "legend_space": "legend floats over the map; no blank right column reserved",
        },
        "html_path": str(html_path),
        "kpi_html_path": str(kpi_html_path),
        "note": (
            "If has_polygon=false, the report loader does not polygon-filter the "
            "drive data and the base image is fit to route GPS bounds only."
        ),
    }
    _write_json(run_dir / "base_route_map_debug_summary.json", summary)

    print(f"Generated compact base route image: {compact_png_path}")
    print(f"Generated compact KPI image: {kpi_compact_png_path}")
    print(f"Debug summary: {run_dir / 'base_route_map_debug_summary.json'}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate the real report base-route image for one project."
    )
    parser.add_argument("--project-id", type=int, default=272)
    parser.add_argument("--user-id", type=int, default=13)
    parser.add_argument(
        "--out-dir",
        default="tests/output/report_engine_base_map_debug",
        help="Output directory relative to ML/ when run from ML, or current cwd.",
    )
    parser.add_argument(
        "--render-timeout-ms",
        type=int,
        default=20000,
        help="Playwright/Folium render timeout for this debug run.",
    )
    parser.add_argument("--candidate-width", type=int, default=1000)
    parser.add_argument("--candidate-height", type=int, default=760)
    parser.add_argument("--tile-name", default="CartoDB Voyager")
    parser.add_argument("--kpi-name", default="RSRP", choices=sorted(KPI_CONFIG.keys()))
    parser.add_argument("--left-padding-px", type=int, default=0)
    parser.add_argument("--base-right-padding-px", type=int, default=120)
    parser.add_argument("--legend-right-padding-px", type=int, default=340)
    parser.add_argument("--top-padding-px", type=int, default=0)
    parser.add_argument("--bottom-padding-px", type=int, default=10)
    parser.add_argument("--legend-top-px", type=int, default=44)
    parser.add_argument("--legend-right-px", type=int, default=6)
    args = parser.parse_args()

    generate_project_base_route_image(
        project_id=args.project_id,
        user_id=args.user_id,
        out_dir=Path(args.out_dir),
        render_timeout_ms=args.render_timeout_ms,
        candidate_width=args.candidate_width,
        candidate_height=args.candidate_height,
        tile_name=args.tile_name,
        kpi_name=args.kpi_name,
        left_padding_px=args.left_padding_px,
        base_right_padding_px=args.base_right_padding_px,
        legend_right_padding_px=args.legend_right_padding_px,
        top_padding_px=args.top_padding_px,
        bottom_padding_px=args.bottom_padding_px,
        legend_top_px=args.legend_top_px,
        legend_right_px=args.legend_right_px,
    )


if __name__ == "__main__":
    main()
