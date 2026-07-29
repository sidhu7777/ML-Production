"""
report_engine_base_map_debug.py
================================
Debug / validation test case for base-map and full PDF report generation.

NO production code is modified here.  All corrections live in this file only.

Three bugs are fixed compared to the production pipeline:

  FIX 1 – Map bounds
    The production generate_kpi_map merges the polygon's bounding box with
    the data bounds before calling fit_bounds.  When the project polygon is
    large but the drive only covers a small portion of it, the resulting
    viewport is padded with vast empty map tiles.
    Fix: local map-generation helpers fit the viewport to the data GPS points
    only.  The polygon boundary is still drawn on the map as a subtle dashed
    overlay for geographic context, but it does NOT expand the viewport.

  FIX 2 – Band % consistency (table vs LLM text)
    kpi_analysis.generate_band_summary uses len(filtered_df) as the
    denominator — this includes rows whose band value is NaN — so the %
    values in the table image are lower than those the LLM sees.
    metadata_generator.build_band_summary drops NaN first, so its total is
    higher, causing the LLM to report a different (higher) percentage.
    Additionally, raw band values like "0", "-1", "Unknown", "N/A", etc. are
    NOT normalised by either production function, so they appear as
    "Unknown" entries in the LLM text.
    Fix: after run_kpi_analysis generates band_table.png, it is overwritten
    here with a version whose denominator is the count of rows with a
    *known* band (after normalize_band_name), "Unknown" entries excluded.
    The same corrected list is injected into the metadata so the LLM-facing
    band_summary matches the table exactly.

  FIX 3 – Handover count (report ≪ frontend)
    load_project_data filters to primary-serving cells (mRegistered=YES)
    before returning filtered_df.  detect_handover_events then counts band
    transitions in that reduced set.  The frontend's GetNetworkLog C# endpoint
    does NOT apply a primary-cell filter — it returns all cells — so the
    frontend handover count is much higher.
    Fix: handover events are detected from a dataset that is polygon-filtered
    but NOT primary-cell filtered (polygon_filtered_raw_df), matching what the
    frontend backend produces.

Usage
-----
  Debug mode — base map + one KPI map only (original behaviour):
    python report_engine_base_map_debug.py --project-id 272

  Full report mode — generates complete PDF with all fixes:
    python report_engine_base_map_debug.py --project-id 272 --mode full

  Full report without LLM (faster, skips AI text generation):
    python report_engine_base_map_debug.py --project-id 272 --mode full --skip-llm
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import folium
import numpy as np
from PIL import Image as PILImage
from shapely import contains as shp_contains, points as shp_points
from shapely.wkt import loads
from reportlab.platypus import Image as RLImage, Spacer, Paragraph, Table, TableStyle, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Production imports (read-only — these files are NOT changed) ─────────────
from tools.report_engine.load_data_db import load_project_data
from tools.report_engine.kpi_config import KPI_CONFIG
from tools.report_engine.map_generator import (
    add_fullscreen_css,
    add_legend,
    build_legend_from_ranges,
    get_df_bounds,
    value_in_range,
    detect_handover_events,
    normalize_band_name,
    get_frontend_band_color,
    generate_distinct_colors,
)
from tools.report_engine.playwright_utils import html_to_png
from tools.report_engine.threshold_resolver import resolve_kpi_ranges
from tools.report_engine.kpi_analysis import run_kpi_analysis
from tools.report_engine.metadata_generator import build_metadata, write_metadata_file
from tools.report_engine.llm_integration import generate_report_text
from tools.report_engine.pdf_generator import PDFReportGenerator


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_run_dir(base_dir: Path, project_id: int, label: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"project_{project_id}_{label}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _parse_session_ids(value) -> list[int]:
    return [int(s.strip()) for s in str(value or "").split(",") if s.strip().isdigit()]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _join_unique_non_empty(values) -> str:
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned.ne("") & cleaned.str.lower().ne("none")]
    return ",".join(sorted(set(cleaned)))


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _route_summary(df: pd.DataFrame, polygon_wkt: str | None) -> dict:
    geo_df = df.dropna(subset=["lat", "lon"])
    bounds = get_df_bounds(geo_df) if not geo_df.empty else None
    return {
        "rows": int(len(df)),
        "valid_geo_rows": int(len(geo_df)),
        "has_polygon": bool(polygon_wkt),
        "bounds": bounds,
    }


def _filter_known_band_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only report rows with a known/valid band after production normalization."""
    if "band" not in df.columns:
        return df.iloc[0:0].copy()
    normalized = df["band"].apply(normalize_band_name)
    return df.loc[normalized.ne("Unknown")].copy()


def _cdf_filename_for_kpi(kpi_name: str, column: str) -> str:
    if kpi_name == "DL":
        return "cdf_dl_tpt.png"
    if kpi_name == "UL":
        return "cdf_ul_tpt.png"
    return f"cdf_{column.lower()}.png"


def _generate_local_cdf_plot(values: pd.Series, title: str, x_label: str, output_path: Path) -> tuple[int, float, float] | None:
    values = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if values.empty:
        return None

    y = np.arange(1, len(values) + 1) / len(values) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(values.to_numpy(), y, where="post", linewidth=2.0, color="#1f77b4")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Cumulative Probability (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="normal", pad=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.set_ylim(0, 100)

    offsets = {
        10: (18, 14),
        50: (18, -24),
        90: (-56, 14),
    }
    for p in (10, 50, 90):
        percentile_value = float(np.percentile(values, p))
        ax.scatter(
            [percentile_value],
            [p],
            s=34,
            color="#d62728",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.annotate(
            f"P{p} = {percentile_value:.2f}",
            xy=(percentile_value, p),
            xytext=offsets[p],
            textcoords="offset points",
            fontsize=8.5,
            color="#333333",
            arrowprops={
                "arrowstyle": "->",
                "color": "#d62728",
                "lw": 0.8,
                "alpha": 0.85,
            },
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "white",
                "ec": "#d0d7de",
                "alpha": 0.95,
            },
        )

    ax.tick_params(axis="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return len(values), float(values.min()), float(values.max())


def _generate_all_cdf_plots_local(report_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate every CDF from the exact report dataframe, not from the session API."""
    print("\n" + "=" * 80)
    print("GENERATING CDF DISTRIBUTION GRAPHS FROM REPORT FILTERED DATA")
    print("=" * 80)
    print(f"Rows used for CDF: {len(report_df)}")

    for kpi_name, cfg in KPI_CONFIG.items():
        col = cfg.get("column")
        if cfg.get("type") == "range" and col in report_df.columns:
            filename = _cdf_filename_for_kpi(kpi_name, col)
            stats = _generate_local_cdf_plot(
                report_df[col],
                f"CDF Distribution - {kpi_name}",
                f"{kpi_name} Value",
                output_dir / filename,
            )
            if stats:
                count, min_value, max_value = stats
                print(f"  {filename}: rows={count}, min={min_value:.2f}, max={max_value:.2f}")

    if "pci" in report_df.columns:
        stats = _generate_local_cdf_plot(
            report_df["pci"],
            "CDF Distribution - PCI",
            "PCI Value",
            output_dir / "cdf_pci.png",
        )
        if stats:
            count, min_value, max_value = stats
            print(f"  cdf_pci.png: rows={count}, min={min_value:.2f}, max={max_value:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1 — MAP GENERATION WITH DATA-ONLY BOUNDS
# These three helpers replicate the production map generators but do NOT
# merge the polygon's bounding box into the viewport.  The polygon boundary
# is still rendered on the map as a dashed overlay for geographic reference.
# ═══════════════════════════════════════════════════════════════════════════════

_REPORT_TILE = "CartoDB Voyager"
# Map screenshot resolution multiplier.  Production default is 2 (so a 1200×900
# map becomes a 2400×1800 PNG ≈ 4 MB each — that is what bloats the PDF to
# 40-60 MB).  At 1 the map is 1200×900 ≈ 207 DPI on the page — still sharp —
# and roughly 1/4 the bytes.
_DEVICE_SCALE = 1
# Tight padding (px) — small margins so the route fills the frame.
_PAD_LEFT    = 30   # px — left / top / bottom padding
_PAD_VERT    = 30
_PAD_RIGHT_BASE   = 40    # px — right padding for base route map (no legend)
_PAD_RIGHT_LEGEND = 340   # px — clears the 320px-wide legend overlay on the right


def _new_report_map() -> folium.Map:
    """
    Build a report map with FRACTIONAL ZOOM enabled.

    zoomSnap=0 lets fit_bounds settle on any fractional zoom level — exactly
    like Google Maps on the frontend — instead of snapping down to the next
    whole integer zoom.  Integer snapping is what left big gray/empty margins:
    when the route was a hair too large for zoom 13, Leaflet fell back to
    zoom 12 and filled only ~60% of the canvas.  With zoomSnap=0 the data
    fills the viewport right up to the padding on every map.
    """
    return folium.Map(
        tiles=_REPORT_TILE,
        zoom_control=True,
        control_scale=False,
        prefer_canvas=True,
        max_zoom=22,
        zoomSnap=0,      # continuous fractional zoom (no integer snapping)
        zoomDelta=0.25,
    )


def _fit_to_data(m: folium.Map, df: pd.DataFrame, *, legend: bool = False) -> None:
    """Fit the map viewport to the GPS data bounds only (fractional zoom)."""
    bounds = get_df_bounds(df)
    right = _PAD_RIGHT_LEGEND if legend else _PAD_RIGHT_BASE
    m.fit_bounds(
        bounds,
        padding_top_left=(_PAD_LEFT, _PAD_VERT),
        padding_bottom_right=(right, _PAD_VERT),
        max_zoom=22,
    )


def _draw_polygon_overlay(
    m: folium.Map,
    polygon_wkt: str | None,
    *,
    color: str = "#111827",
    weight: int = 2,
    opacity: float = 0.45,
    dash_array: str = "6,6",
) -> None:
    """Draw the project polygon as a subtle dashed outline (does not affect bounds)."""
    if not polygon_wkt:
        return
    geom = loads(polygon_wkt)
    latlon = [(c[1], c[0]) for c in geom.exterior.coords]
    folium.Polygon(
        locations=latlon,
        color=color,
        weight=weight,
        fill=False,
        opacity=opacity,
        dash_array=dash_array,
        tooltip="Polygon Boundary",
    ).add_to(m)


def _generate_base_route_map_fixed(
    df: pd.DataFrame,
    polygon_wkt: str | None,
    output_html: Path,
) -> None:
    """Base route map: route PolyLine per session, viewport = data bounds only.

    Uses PolyLine instead of 44K+ individual CircleMarkers — far faster to
    render in the headless browser and produces a clean continuous route line.
    """
    df = df.dropna(subset=["lat", "lon"])
    if df.empty:
        raise ValueError("No GPS data for base route map")

    m = _new_report_map()
    add_fullscreen_css(m)

    # Draw each GPS point as a filled circle (prefer_canvas=True routes all
    # CircleMarkers through the Leaflet Canvas renderer, so 44K dots render
    # fast without creating individual SVG DOM nodes).
    for _, r in df.iterrows():
        folium.CircleMarker(
            location=(r["lat"], r["lon"]),
            radius=4,
            color="#2b8cbe",
            fill=True,
            fill_opacity=0.95,
        ).add_to(m)

    # Polygon as solid red outline — does NOT expand viewport
    _draw_polygon_overlay(m, polygon_wkt, color="red", weight=4, opacity=1.0, dash_array="")
    _fit_to_data(m, df, legend=False)
    m.save(str(output_html))


def _generate_range_kpi_map_fixed(
    df: pd.DataFrame,
    kpi_column: str,
    ranges: list[dict],
    polygon_wkt: str | None,
    output_html: Path,
) -> None:
    """Range KPI map (RSRP/RSRQ/SINR/DL/UL/MOS): viewport = data bounds only."""
    df = df.dropna(subset=["lat", "lon", kpi_column])
    if df.empty:
        raise ValueError(f"No data for KPI map: {kpi_column}")

    m = _new_report_map()
    add_fullscreen_css(m)
    _draw_polygon_overlay(m, polygon_wkt)  # dashed overlay — does NOT shift bounds

    for _, row in df.iterrows():
        try:
            value = float(row[kpi_column])
        except (TypeError, ValueError):
            continue
        color = "#808080"
        for idx, r in enumerate(ranges):
            if value_in_range(value, r, idx == len(ranges) - 1):
                color = r["color"]
                break
        folium.CircleMarker(
            location=(row["lat"], row["lon"]), radius=4,
            color=color, fill=True, fill_opacity=0.9,
        ).add_to(m)

    legend_items = build_legend_from_ranges(df, kpi_column, ranges)
    add_legend(m, kpi_column, legend_items)
    _fit_to_data(m, df, legend=True)
    m.save(str(output_html))


def _generate_categorical_map_fixed(
    df: pd.DataFrame,
    kpi_column: str,
    polygon_wkt: str | None,
    output_html: Path,
) -> None:
    """
    Band / PCI categorical map: viewport = data bounds only.
    For Band maps, Unknown-normalised entries are excluded so the
    map colours match the corrected band table.
    """
    df = df.dropna(subset=["lat", "lon", kpi_column])
    if df.empty:
        raise ValueError(f"No data for categorical map: {kpi_column}")

    m = _new_report_map()
    add_fullscreen_css(m)
    _draw_polygon_overlay(m, polygon_wkt)

    is_band = kpi_column.lower() == "band"
    cat_col = "__report_cat"

    if is_band:
        df = df.copy()
        df[cat_col] = df[kpi_column].apply(normalize_band_name)
        # FIX 2 (map side): exclude Unknown so map matches corrected table
        df = df[df[cat_col] != "Unknown"]
        if df.empty:
            raise ValueError("No non-Unknown band data for band map")
        vcounts = df[cat_col].value_counts()
        color_map = {v: get_frontend_band_color(v) for v in vcounts.index}
        draw_df = df.assign(__cnt=df[cat_col].map(vcounts)).sort_values("__cnt", ascending=False)
        radius, fill_opacity = 3, 0.72
    else:
        unique_vals = sorted(df[kpi_column].unique())
        colors = generate_distinct_colors(len(unique_vals))
        color_map = {v: colors[i] for i, v in enumerate(unique_vals)}
        draw_df = df
        cat_col = kpi_column
        radius, fill_opacity = 4, 0.9

    for _, row in draw_df.iterrows():
        val = row[cat_col]
        if val not in color_map:
            continue
        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=radius, color=color_map[val],
            fill=True, fill_opacity=fill_opacity, opacity=0.85,
        ).add_to(m)

    vcounts = df[cat_col].value_counts()
    top_n = 6
    legend_items = [(str(v), color_map[v], c) for v, c in vcounts.head(top_n).items()]
    others = vcounts.iloc[top_n:].sum()
    if others > 0:
        legend_items.append(("Others", "#999999", others))
    add_legend(m, kpi_column, legend_items)

    _fit_to_data(m, df, legend=True)
    m.save(str(output_html))


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2 — BAND: CONSISTENT DENOMINATOR + EXCLUDE UNKNOWN
# Overwrites band_table.png and band_pie.png after run_kpi_analysis so
# both the image and the LLM-facing metadata use the same numbers.
# ═══════════════════════════════════════════════════════════════════════════════

def _regenerate_band_images_corrected(df: pd.DataFrame, image_dir: str) -> list[dict] | None:
    """
    Recompute band distribution excluding Unknown-normalised entries.
    Denominator = rows with a *known* band value (after normalization).
    Overwrites band_table.png and band_pie.png in image_dir.
    Returns the corrected band_summary list for injection into metadata.
    """
    if "band" not in df.columns:
        return None

    normalized = df["band"].apply(normalize_band_name)
    known = normalized[normalized != "Unknown"]
    if known.empty:
        return None

    total = len(known)
    band_counts = known.value_counts()

    rows = []
    summary = []
    for band, count in band_counts.items():
        pct = round((count / total) * 100, 2)
        rows.append({"Band": band, "Sample Count": int(count), "% Samples": pct})
        summary.append({"band": band, "sample_count": int(count), "sample_percentage": pct})

    band_df = pd.DataFrame(rows)

    # ── band_table.png ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, max(2.5, len(band_df) * 0.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=band_df.values, colLabels=band_df.columns,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.5)
    for i in range(len(band_df.columns)):
        cell = tbl[(0, i)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(weight="bold", color="white")
    plt.title("Band Distribution", pad=15, fontsize=14, weight="bold")
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(image_dir, "band_table.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── band_pie.png ──────────────────────────────────────────────
    plt.figure(figsize=(6, 6))
    plt.pie(band_df["% Samples"], labels=band_df["Band"], autopct="%1.1f%%")
    plt.title("Band Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "band_pie.png"), dpi=200)
    plt.close()

    bands_list = list(band_df["Band"])
    print(
        f"[FIX 2] band_table.png regenerated — "
        f"denominator={total} (non-Unknown), bands={bands_list}"
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3 — HANDOVER: POLYGON-FILTERED RAW (no primary-cell filter)
# ═══════════════════════════════════════════════════════════════════════════════

def _polygon_filter_raw(raw_df: pd.DataFrame, polygon_wkt: str | None) -> pd.DataFrame:
    """
    Apply spatial filtering to raw_df without any primary-cell restriction.
    This matches the C# GetNetworkLog backend which returns all cells for the
    session IDs within the project polygon.
    """
    if not polygon_wkt:
        return raw_df

    geo = raw_df.dropna(subset=["lat", "lon"])
    if geo.empty:
        return raw_df

    poly = loads(polygon_wkt)
    print(f"[FIX 3] Spatial-filtering {len(geo)} raw rows against polygon (vectorized) …")
    # Vectorized point-in-polygon: build all points at once and test in a single
    # shapely 2.0 call.  ~50-100× faster than df.apply row-by-row Point/contains.
    lon = pd.to_numeric(geo["lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(geo["lat"], errors="coerce").to_numpy(dtype=float)
    pts = shp_points(lon, lat)               # array of POINT geometries
    mask = shp_contains(poly, pts)           # boolean ndarray, all at once
    mask = np.where(np.isnan(lon) | np.isnan(lat), False, mask)
    result = geo[mask].reset_index(drop=True)
    print(f"[FIX 3] Polygon-filtered raw rows (all cells): {len(result)}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 4 — POOR REGION MAPS: SHOW ALL INDIVIDUAL POOR POINTS
# The production generate_poor_region_map clusters samples into top_regions=5
# centroids, so only 5-6 blobs appear even when there are 17 000 poor points.
# Fix: draw every poor sample as an individual dot — no clustering.
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_POOR_DOTS = 8000   # cap for rendering performance (canvas still fast at 8k)


def _generate_poor_region_map_fixed(
    df: pd.DataFrame,
    value_col: str,
    threshold: float,
    title: str,
    output_html: Path,
) -> int:
    """
    Draw ALL individual poor-sample points on a tight map.
    Caps at _MAX_POOR_DOTS via uniform subsampling when the dataset is very
    large (e.g. RSRQ has 17 000+ poor rows) — the visual result is the same
    because the dots overlap densely at this zoom level.
    Returns the true (unsampled) poor-sample count for the legend.
    """
    df = df.dropna(subset=["lat", "lon"]).copy()
    if value_col in df.columns:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    poor = df[df[value_col] < threshold]
    true_count = len(poor)
    print(f"  [FIX4] {title}: {true_count} poor samples out of {len(df)}")

    if true_count == 0:
        print(f"  [FIX4] No poor samples — skipping {title}")
        return 0

    # Subsample only for rendering speed; legend always shows true_count
    draw_df = (
        poor.iloc[:: max(1, len(poor) // _MAX_POOR_DOTS)]
        if len(poor) > _MAX_POOR_DOTS
        else poor
    )

    m = _new_report_map()   # fractional zoom + canvas renderer
    add_fullscreen_css(m)

    for _, r in draw_df.iterrows():
        folium.CircleMarker(
            location=(r["lat"], r["lon"]),
            radius=4,
            color="#e31a1c",
            fill=True,
            fill_opacity=0.85,
        ).add_to(m)

    legend_items = [(f"Poor Samples : {true_count}", "#e31a1c", true_count)]
    add_legend(m, title, legend_items)

    # Viewport fits the FULL route (df), not just poor points, for geographic context
    _fit_to_data(m, df, legend=True)
    m.save(str(output_html))
    return true_count


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 5 — HANDOVER MAP: data-only bounds + fractional zoom
# The production generate_handover_map merges the polygon bounds into the
# viewport AND relies on integer zoom, so it ends up zoomed out with gray
# margins like the old KPI maps.  This local version draws the same route +
# handover sparks but fits to the data bounds only with fractional zoom.
# ═══════════════════════════════════════════════════════════════════════════════

_HANDOVER_SPARK_SVG = (
    '<div style="transform: translate(-50%, -50%);">'
    '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24">'
    '<path fill="{color}" stroke="#222" stroke-width="0.6" '
    'd="M13 2 L3 14 H12 L11 22 L21 10 H12 L13 2 Z"/>'
    '</svg></div>'
)


def _generate_handover_map_fixed(
    df: pd.DataFrame,
    events: list[dict],
    polygon_wkt: str | None,
    output_html: Path,
) -> None:
    """Handover map: route points + band-handover sparks, data-bounds + fractional zoom."""
    df = df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        raise ValueError("No GPS data for handover map")

    band_events = [
        ev for ev in (events or [])
        if str(ev.get("type") or "").lower() == "band"
    ]

    m = _new_report_map()
    add_fullscreen_css(m)
    _draw_polygon_overlay(m, polygon_wkt)   # dashed overlay — does NOT expand bounds

    # Route points coloured per session (matches production look)
    if "session_id" in df.columns:
        sessions = list(df.groupby("session_id", sort=False))
        colors = generate_distinct_colors(len(sessions)) if sessions else ["#2b8cbe"]
        for i, (_, seg) in enumerate(sessions):
            c = colors[i % len(colors)]
            for _, r in seg.iterrows():
                folium.CircleMarker(
                    location=(r["lat"], r["lon"]), radius=4,
                    color=c, fill=True, fill_opacity=0.95,
                ).add_to(m)
    else:
        for _, r in df.iterrows():
            folium.CircleMarker(
                location=(r["lat"], r["lon"]), radius=4,
                color="#2b8cbe", fill=True, fill_opacity=0.95,
            ).add_to(m)

    # Handover sparks (blue lightning bolts)
    spark_color = "#3b82f6"
    for ev in band_events:
        html = _HANDOVER_SPARK_SVG.format(color=spark_color)
        icon = folium.DivIcon(html=html, icon_size=(28, 28), icon_anchor=(14, 14))
        tooltip = (
            f"Band: {ev.get('from_value')} -> {ev.get('to_value')} "
            f"(Session {ev.get('session_id')})"
        )
        folium.Marker(location=(ev["lat"], ev["lon"]), icon=icon, tooltip=tooltip).add_to(m)

    if band_events:
        add_legend(m, "Band Handover Events", [("Band", spark_color, len(band_events))])

    _fit_to_data(m, df, legend=bool(band_events))
    m.save(str(output_html))


def _render_kpi_map_task(
    kpi: str,
    cfg: dict,
    filtered_df: pd.DataFrame,
    html_dir: Path,
    kpi_maps_dir: Path,
    polygon_wkt: str | None,
    user_id: int | None,
    render_width: int,
    render_height: int,
) -> tuple[str, str | None, str | None]:
    """
    One KPI map generation task — safe to run in a thread pool.
    Returns (kpi_name, map_filename_or_None, error_or_None).
    """
    _ensure_dirs(html_dir, kpi_maps_dir)
    col = cfg["column"]
    if col not in filtered_df.columns:
        return kpi, None, "column missing"

    df_kpi = filtered_df[
        filtered_df[col].notna()
        & filtered_df["lat"].notna()
        & filtered_df["lon"].notna()
    ]
    if df_kpi.empty:
        return kpi, None, "no valid data"

    kpi_html = html_dir / f"{kpi.lower()}.html"
    kpi_png  = kpi_maps_dir / cfg["map_name"]

    try:
        if cfg["type"] == "range":
            df_num = df_kpi.copy()
            df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
            df_num = df_num.dropna(subset=[col])
            ranges = resolve_kpi_ranges(kpi_name=kpi, user_id=user_id, values=df_num[col])
            _generate_range_kpi_map_fixed(df_num, col, ranges, polygon_wkt, kpi_html)
        elif cfg["type"] == "categorical":
            _generate_categorical_map_fixed(df_kpi, col, polygon_wkt, kpi_html)
        else:
            return kpi, None, f"unknown type {cfg['type']}"

        html_to_png(str(kpi_html), str(kpi_png), width=render_width, height=render_height, device_scale_factor=_DEVICE_SCALE)
        return kpi, cfg["map_name"], None
    except Exception as exc:
        return kpi, None, str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 6 — CRISP TABLES + SMALL FILE SIZE
# Subclass of the production PDFReportGenerator that overrides ONLY add_image:
#   • Data tables  → rendered at full page width and NOT squeezed to the 4-inch
#     height cap, so the text stays large and sharp (the old cap shrank a 30-row
#     table to ~3.7 in wide → microscopic, "blurry" text).
#   • Map / chart images → palette-quantized + optimized before embedding, which
#     shrinks each map PNG from ~4 MB to a few hundred KB with no visible loss.
# All layout / TOC / section logic is inherited unchanged from production.
# ═══════════════════════════════════════════════════════════════════════════════

# Filenames that are DATA TABLES (render crisp/full-width, no height squeeze).
_TABLE_IMAGE_FILES = {
    "band_table.png", "pci_table.png", "session_table.png",
    "kpi_summary.png", "drive_summary.png", "poor_kpi_stats.png",
    "network_quality_summary.png", "pci_cdf_table.png",
    "speed_stats.png", "latency_stats.png", "jitter_stats.png",
    "rsrp_range_table.png", "rsrq_range_table.png", "sinr_range_table.png",
    "dl_range_table.png", "ul_range_table.png", "mos_range_table.png",
}

_CHART_IMAGE_FILES = {
    "pci_distribution.png",
    "cdf_rsrp.png", "cdf_rsrq.png", "cdf_sinr.png",
    "cdf_dl_tpt.png", "cdf_ul_tpt.png", "cdf_mos.png", "cdf_pci.png",
    "band_pie.png",
    "speed_hist.png", "latency_hist.png", "jitter_hist.png",
}

_TABLE_MAX_WIDTH  = 6.6 * inch   # full usable A4 text width (margins 40+40pt)
_TABLE_MAX_HEIGHT = 8.6 * inch   # nearly full page height (no 4-inch squeeze)
_POOR_PCI_LIMIT = 15

_TABLE_STYLE_SHEET = getSampleStyleSheet()
_TABLE_CELL_STYLE = ParagraphStyle(
    "DebugTableCell",
    parent=_TABLE_STYLE_SHEET["BodyText"],
    fontName="Helvetica",
    fontSize=8,
    leading=9,
    textColor=colors.HexColor("#1f2937"),
)
_TABLE_HEADER_STYLE = ParagraphStyle(
    "DebugTableHeader",
    parent=_TABLE_CELL_STYLE,
    fontName="Helvetica-Bold",
    textColor=colors.white,
)


def _make_native_table(df: pd.DataFrame) -> Table:
    """Render a compact native ReportLab table so the PDF stays crisp."""
    table_data = [[Paragraph(str(col), _TABLE_HEADER_STYLE) for col in df.columns]]
    for row in df.itertuples(index=False):
        table_data.append([
            Paragraph("" if pd.isna(value) else str(value), _TABLE_CELL_STYLE)
            for value in row
        ])

    col_widths = None
    if len(df.columns) == 5:
        col_widths = [
            0.55 * inch,
            0.9 * inch,
            0.95 * inch,
            0.95 * inch,
            3.25 * inch,
        ]
    elif len(df.columns) > 0:
        col_widths = [_TABLE_MAX_WIDTH / len(df.columns)] * len(df.columns)

    table = Table(table_data, repeatRows=1, hAlign="LEFT", colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa6b2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.whitesmoke,
            colors.HexColor("#eef2f7"),
        ]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _has_valid_series(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].dropna().shape[0] > 0


def _format_optional_avg(df: pd.DataFrame, column: str, decimals: int = 1, suffix: str = ""):
    if not _has_valid_series(df, column):
        return "N/A"
    value = round(pd.to_numeric(df[column], errors="coerce").mean(), decimals)
    return f"{value} {suffix}".strip()


def _drop_empty_na_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Hide columns that contain no real display value, such as all Category=N/A."""
    keep = []
    for col in df.columns:
        values = df[col].dropna().astype(str).str.strip()
        values = values[~values.str.lower().isin({"", "n/a", "na", "none", "nan"})]
        if not values.empty:
            keep.append(col)
    return df[keep] if keep else df


def _build_app_analytics_native_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    app_col = None
    if "apps" in df.columns and not df["apps"].isna().all():
        app_col = "apps"
    elif "app_name" in df.columns and not df["app_name"].isna().all():
        app_col = "app_name"
    if not app_col:
        return {}

    category_col = None
    if "category" in df.columns:
        category_col = "category"
    elif "app_category" in df.columns:
        category_col = "app_category"

    numeric_cols = ["rsrp", "rsrq", "sinr", "dl_tpt", "ul_tpt", "mos", "latency", "jitter", "packet_loss"]
    df_clean = df.copy()
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    part1_rows = []
    part2_rows = []
    for app in df_clean[app_col].dropna().unique():
        app_df = df_clean[df_clean[app_col] == app].copy()
        if app_df.empty:
            continue

        category = "N/A"
        if category_col and category_col in app_df.columns:
            cat_vals = app_df[category_col].dropna().astype(str).str.strip()
            cat_vals = cat_vals[~cat_vals.str.lower().isin({"", "n/a", "na", "none", "nan"})]
            if not cat_vals.empty:
                category = cat_vals.iloc[0]

        sessions = app_df["session_id"].nunique() if "session_id" in app_df.columns else "N/A"
        duration = "N/A"
        time_col = "timestamp" if "timestamp" in app_df.columns else "time" if "time" in app_df.columns else None
        if time_col:
            times = pd.to_datetime(app_df[time_col], errors="coerce")
            if times.notna().any():
                seconds = int((times.max() - times.min()).total_seconds())
                duration = f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

        part1_rows.append({
            "App Name": app,
            "Category": category,
            "Sessions": sessions,
            "Samples": len(app_df),
            "Duration": duration,
            "RSRP Avg": _format_optional_avg(app_df, "rsrp"),
            "RSRQ Avg": _format_optional_avg(app_df, "rsrq"),
            "SINR Avg": _format_optional_avg(app_df, "sinr"),
            "DL Avg": _format_optional_avg(app_df, "dl_tpt", suffix="Mbps"),
        })
        part2_rows.append({
            "App Name": app,
            "Category": category,
            "UL Avg": _format_optional_avg(app_df, "ul_tpt", suffix="Mbps"),
            "MOS Avg": _format_optional_avg(app_df, "mos", decimals=2),
            "Latency Avg": _format_optional_avg(app_df, "latency", suffix="ms"),
            "Jitter Avg": _format_optional_avg(app_df, "jitter", suffix="ms"),
            "Loss Avg": _format_optional_avg(app_df, "packet_loss", decimals=2, suffix="%"),
        })

    tables = {}
    if part1_rows:
        tables["app_analytics_part1.png"] = _drop_empty_na_columns(pd.DataFrame(part1_rows))
        tables["app_analytics_part2.png"] = _drop_empty_na_columns(pd.DataFrame(part2_rows))
    return tables


def _build_misc_native_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}

    kpi_rows = []
    for kpi, cfg in KPI_CONFIG.items():
        if cfg.get("type") != "range":
            continue
        col = cfg["column"]
        if not _has_valid_series(df, col):
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        kpi_rows.append({
            "KPI": kpi,
            "Average": round(values.mean(), 2),
            "Min": round(values.min(), 2),
            "Max": round(values.max(), 2),
            "Median": round(values.median(), 2),
        })
    if kpi_rows:
        tables["kpi_summary.png"] = pd.DataFrame(kpi_rows)

    if _has_valid_series(df, "pci"):
        pci_counts = df["pci"].dropna().value_counts().sort_values(ascending=False).head(30)
        pci_table = pci_counts.reset_index()
        pci_table.columns = ["PCI", "Sample Count"]
        tables["pci_table.png"] = pci_table

    quality = {}
    if _has_valid_series(df, "latency"):
        quality["Avg Latency (ms)"] = round(pd.to_numeric(df["latency"], errors="coerce").mean(), 2)
    if _has_valid_series(df, "jitter"):
        quality["Avg Jitter (ms)"] = round(pd.to_numeric(df["jitter"], errors="coerce").mean(), 2)
    if _has_valid_series(df, "packet_loss"):
        quality["Avg Packet Loss (%)"] = round(pd.to_numeric(df["packet_loss"], errors="coerce").mean(), 2)
    if quality:
        tables["network_quality_summary.png"] = pd.DataFrame([quality])

    poor_rows = []
    if _has_valid_series(df, "rsrp"):
        rsrp_values = pd.to_numeric(df["rsrp"], errors="coerce").dropna()
        poor_count = int((rsrp_values < -105).sum())
        poor_rows.append({
            "KPI": "RSRP",
            "Threshold": "< -105 dBm",
            "Poor Samples": poor_count,
            "Total Samples": len(rsrp_values),
            "Percentage": f"{(poor_count / len(rsrp_values) * 100):.2f}%" if len(rsrp_values) else "0.00%",
        })
    if _has_valid_series(df, "rsrq"):
        rsrq_values = pd.to_numeric(df["rsrq"], errors="coerce").dropna()
        poor_count = int((rsrq_values < -14).sum())
        poor_rows.append({
            "KPI": "RSRQ",
            "Threshold": "< -14 dB",
            "Poor Samples": poor_count,
            "Total Samples": len(rsrq_values),
            "Percentage": f"{(poor_count / len(rsrq_values) * 100):.2f}%" if len(rsrq_values) else "0.00%",
        })
    if poor_rows:
        tables["poor_kpi_stats.png"] = pd.DataFrame(poor_rows)

    for column, filename in [
        ("speed", "speed_stats.png"),
        ("latency", "latency_stats.png"),
        ("jitter", "jitter_stats.png"),
    ]:
        if _has_valid_series(df, column):
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            tables[filename] = pd.DataFrame([{
                "Average": round(values.mean(), 2),
                "Min": round(values.min(), 2),
                "Max": round(values.max(), 2),
            }])

    if "indoor_outdoor" in df.columns and not df["indoor_outdoor"].isna().all():
        rows = []
        numeric_df = df.copy()
        for col in ["rsrp", "rsrq", "sinr", "mos", "dl_tpt", "ul_tpt"]:
            if col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
        for env in ["Indoor", "Outdoor"]:
            env_df = numeric_df[numeric_df["indoor_outdoor"] == env]
            if not env_df.empty:
                rows.append({
                    "Environment": env,
                    "Samples": len(env_df),
                    "Avg RSRP": _format_optional_avg(env_df, "rsrp"),
                    "Avg RSRQ": _format_optional_avg(env_df, "rsrq"),
                    "Avg SINR": _format_optional_avg(env_df, "sinr", decimals=2),
                    "Avg MOS": _format_optional_avg(env_df, "mos", decimals=2),
                    "Avg DL (Mbps)": _format_optional_avg(env_df, "dl_tpt", decimals=2),
                    "Avg UL (Mbps)": _format_optional_avg(env_df, "ul_tpt", decimals=2),
                })
        if rows:
            tables["indoor_outdoor_stats.png"] = _drop_empty_na_columns(pd.DataFrame(rows))

    return tables


def _build_band_native_table(corrected_band_summary: list[dict] | None) -> dict[str, pd.DataFrame]:
    if not corrected_band_summary:
        return {}
    rows = [
        {
            "Band": item.get("band"),
            "Sample Count": item.get("sample_count"),
            "% Samples": item.get("sample_percentage"),
        }
        for item in corrected_band_summary
    ]
    return {"band_table.png": pd.DataFrame(rows)}


def _build_drive_native_tables(drive_summary: dict | None) -> dict[str, pd.DataFrame]:
    if not drive_summary:
        return {}

    sessions = drive_summary.get("sessions") or []
    distance_values = [row.get("distance") for row in sessions if row.get("distance") is not None]
    distance_available = any(float(value or 0) > 0 for value in distance_values)
    total_distance = drive_summary.get("distance_covered")
    distance_text = f"{float(total_distance):.2f} KM" if distance_available and total_distance is not None else "N/A"

    tables = {
        "drive_summary.png": pd.DataFrame([
            {"Metric": "Distance Covered", "Value": distance_text},
            {"Metric": "Total Samples", "Value": f"{drive_summary.get('total_samples', 0):,}"},
            {"Metric": "Total Sessions", "Value": drive_summary.get("total_sessions", "N/A")},
            {"Metric": "Number of Days", "Value": f"{drive_summary.get('number_of_days', 'N/A')} Days"},
        ])
    }

    if sessions:
        rows = []
        for row in sessions:
            distance = row.get("distance")
            distance_display = f"{float(distance):.6f}" if distance_available and distance is not None else "N/A"
            rows.append({
                "Session": row.get("session_id"),
                "Date": row.get("date"),
                "Start": row.get("start_time"),
                "End": row.get("end_time"),
                "Duration": row.get("duration"),
                "Distance": distance_display,
            })
        tables["session_table.png"] = pd.DataFrame(rows)

    return tables


def _build_kpi_range_native_tables(metadata: dict | None) -> dict[str, pd.DataFrame]:
    if not isinstance(metadata, dict):
        return {}

    kpi_summary = metadata.get("kpi_summary") or metadata.get("kpi_analysis") or {}
    if not isinstance(kpi_summary, dict):
        return {}

    filename_by_kpi = {
        "RSRP": "rsrp_range_table.png",
        "RSRQ": "rsrq_range_table.png",
        "SINR": "sinr_range_table.png",
        "DL": "dl_range_table.png",
        "DL Throughput": "dl_range_table.png",
        "UL": "ul_range_table.png",
        "UL Throughput": "ul_range_table.png",
        "MOS": "mos_range_table.png",
    }

    tables = {}
    for kpi, filename in filename_by_kpi.items():
        data = kpi_summary.get(kpi)
        if not isinstance(data, dict):
            continue
        distribution = data.get("distribution")
        if not isinstance(distribution, list) or not distribution:
            continue
        rows = []
        for item in distribution:
            if not isinstance(item, dict):
                continue
            rows.append({
                "Range": item.get("Range") or item.get("range"),
                "% Of Samples": item.get("percentage"),
                "CDF": item.get("cdf"),
            })
        if rows:
            tables[filename] = pd.DataFrame(rows)

    return tables


def _build_frontend_like_poor_pci_table(df: pd.DataFrame, metric_column: str, threshold: float) -> pd.DataFrame | None:
    """Group raw poor samples by PCI and cap the visible table to the top 15 rows."""
    if "pci" not in df.columns or metric_column not in df.columns:
        return None

    working = df[["pci", metric_column, "band"]].copy()
    working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce")
    working = working.dropna(subset=["pci", metric_column])
    working = working.loc[working[metric_column] < threshold]
    if working.empty:
        return None

    grouped = working.groupby("pci").agg(
        Poor_Samples=(metric_column, "count"),
        Avg_Value=(metric_column, "mean"),
        Worst_Value=(metric_column, "min"),
        Bands=("band", _join_unique_non_empty),
    ).reset_index()

    total_poor_pci = int(len(grouped))
    grouped = grouped.sort_values(
        by=["Poor_Samples", "Worst_Value", "pci"],
        ascending=[False, True, True],
    ).head(_POOR_PCI_LIMIT).copy()

    metric_label = metric_column.upper()
    grouped = grouped.rename(columns={
        "pci": "PCI",
        "Poor_Samples": "Poor Samples",
        "Avg_Value": f"Avg {metric_label}",
        "Worst_Value": f"Worst {metric_label}",
    })
    grouped[f"Avg {metric_label}"] = grouped[f"Avg {metric_label}"].round(2)
    grouped[f"Worst {metric_label}"] = grouped[f"Worst {metric_label}"].round(2)
    grouped.attrs["total_poor_pci"] = total_poor_pci
    return grouped


class _CrispPDFGenerator(PDFReportGenerator):
    """PDFReportGenerator with crisp tables and compressed map images."""

    def __init__(self, output_path, images_dir, scratch_dir, native_tables=None):
        super().__init__(output_path, images_dir)
        self._scratch_dir = scratch_dir
        os.makedirs(scratch_dir, exist_ok=True)
        self._opt_savings = 0  # bytes saved by image optimization (for logging)
        self._native_tables = native_tables or {}

    def has_image(self, filename, subdir="kpi_maps"):
        if filename in self._native_tables:
            return True
        return super().has_image(filename, subdir=subdir)

    def has_any_image(self, images):
        return any(self.has_image(**img) for img in images)

    def _append_visual(self, visual):
        """Bind a heading to the visual that immediately follows it."""
        if self.story and isinstance(self.story[-1], Paragraph) and hasattr(self.story[-1], "toc_level"):
            heading = self.story.pop()
            block = [
                heading,
                Spacer(1, 4),
                visual,
                Spacer(1, 6),
            ]
            keep = KeepTogether(block)
            keep.toc_level = heading.toc_level
            keep.toc_text = heading.toc_text
            keep.toc_key = heading.toc_key
            self.story.append(keep)
            return

        self.story.append(visual)
        self.story.append(Spacer(1, 6))

    # -- the single override --------------------------------------------------
    def add_image(self, filename, subdir="kpi_maps", max_height=4 * inch):
        if filename in self._native_tables:
            table_df = self._native_tables[filename]
            if table_df is not None and not table_df.empty:
                self._append_visual(_make_native_table(table_df))
            return

        path = os.path.join(self.images_dir, subdir, filename)
        if not os.path.exists(path):
            print(f"  Warning: missing {path}")
            return

        if filename in _TABLE_IMAGE_FILES:
            self._add_crisp_table(path)
        elif filename in _CHART_IMAGE_FILES:
            self._add_crisp_chart(path, max_height)
        else:
            self._add_compressed_image(path, max_height)

    def add_image_subsection(self, title, key, images, toc_text=None):
        if len(images) == 1 and images[0]["filename"] in self._native_tables:
            filename = images[0]["filename"]
            table_df = self._native_tables[filename]
            if table_df is None or table_df.empty:
                return False

            display_title = title
            if filename in {"pci_poor_rsrp.png", "pci_poor_rsrq.png"}:
                shown = len(table_df)
                total = int(table_df.attrs.get("total_poor_pci", shown))
                suffix = f"Top {shown} of {total} PCIs" if total > shown else f"{shown} PCIs"
                display_title = f"{title} ({suffix})"

            heading = Paragraph(display_title, self.styles["SubSection"])
            heading.toc_level = 1
            heading.toc_text = toc_text or display_title
            heading.toc_key = key

            block = [
                heading,
                Spacer(1, 4),
                _make_native_table(table_df),
                Spacer(1, 6),
            ]
            keep = KeepTogether(block)
            keep.toc_level = 1
            keep.toc_text = toc_text or display_title
            keep.toc_key = key
            self.story.append(keep)
            return True

        return super().add_image_subsection(title, key, images, toc_text=toc_text)

    # -- tables: full width, no height squeeze --------------------------------
    def _add_crisp_table(self, path):
        img = RLImage(path)
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(_TABLE_MAX_WIDTH / iw, _TABLE_MAX_HEIGHT / ih, 1.0)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        self._append_visual(img)

    # -- charts: keep original pixels; quantization softens axis labels/text ----
    def _add_crisp_chart(self, path, max_height):
        img = RLImage(path)
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(_TABLE_MAX_WIDTH / iw, max_height / ih, 1.0)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        self._append_visual(img)

    # -- maps: shrink bytes, then embed at normal sizing -----------------------
    def _add_compressed_image(self, path, max_height):
        embed_path = self._compress_png(path)
        img = RLImage(embed_path)
        iw, ih = img.imageWidth, img.imageHeight
        scale = min((5.8 * inch) / iw, max_height / ih, 1.0)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        self._append_visual(img)

    def _compress_png(self, path):
        """Palette-quantize + optimize a PNG to shrink file size; keep .png."""
        try:
            out = os.path.join(self._scratch_dir, os.path.basename(path))
            with PILImage.open(path) as im:
                rgb = im.convert("RGB")
                # Maps/charts have a limited palette (dots + legend + grey tiles)
                # so 256-colour quantization is visually lossless here.
                q = rgb.quantize(colors=256, method=PILImage.FASTOCTREE)
                q.save(out, format="PNG", optimize=True)
            before, after = os.path.getsize(path), os.path.getsize(out)
            if after < before:
                self._opt_savings += (before - after)
                return out
            return path  # quantization didn't help — embed original
        except Exception as exc:
            print(f"  Warning: image compress failed for {path}: {exc}")
            return path


def _build_pdf_crisp(metadata_path, report_text_path, output_path, images_dir,
                     scratch_dir, native_tables=None, verbose=False):
    """Drop-in for generate_pdf_report using the crisp/compressed subclass."""
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    with open(report_text_path, encoding="utf-8") as f:
        report_text = json.load(f)

    gen = _CrispPDFGenerator(output_path, images_dir, scratch_dir, native_tables=native_tables)
    gen.generate_report(report_text, metadata, verbose)
    saved_mb = gen._opt_savings / (1024 * 1024)
    print(f"  [FIX6] image optimization saved ~{saved_mb:.1f} MB")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# FULL REPORT PIPELINE (all fixes applied)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_full_report(
    project_id: int,
    user_id: int | None,
    out_dir: Path,
    render_width: int = 1200,
    render_height: int = 900,
    skip_llm: bool = False,
) -> Path:
    """
    Run the complete report pipeline with all three fixes applied.
    Outputs: all KPI maps, handover map, KPI analysis images, CDF plots,
             metadata JSON, report text JSON, and final report.pdf.
    """
    run_dir = _make_run_dir(out_dir, project_id, "full_report")
    html_dir       = run_dir / "html"
    images_dir     = run_dir / "images"
    kpi_maps_dir   = images_dir / "kpi_maps"
    kpi_analy_dir  = images_dir / "kpi_analysis"
    processed_dir  = run_dir / "processed"

    _ensure_dirs(html_dir, kpi_maps_dir, kpi_analy_dir, processed_dir)

    # ── 1. LOAD DATA ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[1/9] Loading project data")
    print("=" * 70)
    raw_df, filtered_df, project_meta = load_project_data(project_id)
    polygon_wkt = project_meta.get("region")
    session_ids = _parse_session_ids(project_meta.get("ref_session_id", ""))

    print(f"  raw rows      : {len(raw_df)}")
    print(f"  filtered rows : {len(filtered_df)}  (primary cells + polygon)")
    print(f"  session_ids   : {session_ids}")
    print(f"  polygon       : {'yes' if polygon_wkt else 'none'}")

    if filtered_df.empty:
        raise ValueError(f"Project {project_id} has no rows after report filtering.")

    report_df = _filter_known_band_rows(filtered_df)
    print(f"  report rows   : {len(report_df)}  (primary + polygon + known band)")
    if report_df.empty:
        raise ValueError(f"Project {project_id} has no rows with known band after report filtering.")

    # FIX 3: polygon-filtered raw (all cells, no primary-cell restriction)
    handover_df = _polygon_filter_raw(raw_df, polygon_wkt)

    # ── 2. BASE ROUTE MAP (FIX 1: data bounds only) ───────────────────────────
    print("\n" + "=" * 70)
    print("[2/9] Base route map")
    print("=" * 70)
    base_html = html_dir / "base_route.html"
    base_png  = kpi_maps_dir / "base_route_map.png"
    try:
        _ensure_dirs(html_dir, kpi_maps_dir)
        _generate_base_route_map_fixed(report_df, polygon_wkt, base_html)
        html_to_png(str(base_html), str(base_png), width=render_width, height=render_height, device_scale_factor=_DEVICE_SCALE)
        print("  base_route_map.png  ✓")
    except Exception as exc:
        print(f"  Warning: base route map failed: {exc}")

    # ── 3. KPI MAPS — parallel (FIX 1: data bounds only) ────────────────────
    print("\n" + "=" * 70)
    print("[3/9] KPI maps — 4 parallel workers (data-bounds only)")
    print("=" * 70)
    _MAP_WORKERS = 4
    with ThreadPoolExecutor(max_workers=_MAP_WORKERS) as pool:
        futures = {
            pool.submit(
                _render_kpi_map_task,
                kpi, cfg, report_df,
                html_dir, kpi_maps_dir,
                polygon_wkt, user_id,
                render_width, render_height,
            ): kpi
            for kpi, cfg in KPI_CONFIG.items()
        }
        for fut in as_completed(futures):
            kpi_name, map_file, err = fut.result()
            if err:
                print(f"  Warning: {kpi_name} map failed: {err}")
            else:
                print(f"  {kpi_name:8s} → {map_file}  ✓")

    # ── 4. POOR REGION MAPS — parallel (FIX 4: all individual dots) ──────────
    print("\n" + "=" * 70)
    print("[4/9] Poor region maps — all individual poor points (FIX 4)")
    print("=" * 70)
    poor_specs = [
        ("rsrp", -105, "RSRP < -105", "rsrp_poor_regions.png"),
        ("rsrq",  -14, "RSRQ < -14",  "rsrq_poor_regions.png"),
    ]

    def _render_poor_map(spec):
        col, thresh, title, fname = spec
        out_html = html_dir / f"{col}_poor_regions.html"
        out_png  = kpi_maps_dir / fname
        try:
            _ensure_dirs(html_dir, kpi_maps_dir)
            cnt = _generate_poor_region_map_fixed(report_df, col, thresh, title, out_html)
            html_to_png(str(out_html), str(out_png), width=render_width, height=render_height, device_scale_factor=_DEVICE_SCALE)
            return fname, cnt, None
        except Exception as exc:
            return fname, 0, str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for fname, cnt, err in pool.map(_render_poor_map, poor_specs):
            if err:
                print(f"  Warning: {fname} failed: {err}")
            else:
                print(f"  {fname}  ({cnt} poor samples)  ✓")

    # ── 5. HANDOVER MAP (FIX 3: all cells + FIX 5: data bounds, fractional zoom) ──
    print("\n" + "=" * 70)
    print("[5/9] Handover map (all cells — matches frontend count)")
    print("=" * 70)
    try:
        events = detect_handover_events(handover_df)
        print(f"  Handover events detected: {len(events)}")
        handover_html = html_dir / "handover_map.html"
        handover_png  = kpi_maps_dir / "handover_map.png"
        _ensure_dirs(html_dir, kpi_maps_dir)
        _generate_handover_map_fixed(handover_df, events, polygon_wkt, handover_html)
        html_to_png(str(handover_html), str(handover_png), width=render_width, height=render_height, device_scale_factor=_DEVICE_SCALE)
        print("  handover_map.png  ✓")
    except Exception as exc:
        print(f"  Warning: handover map failed: {exc}")

    # ── 6. KPI ANALYSIS IMAGES ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[6/9] KPI analysis images")
    print("=" * 70)
    _ensure_dirs(processed_dir, kpi_analy_dir)
    filtered_df.to_csv(processed_dir / "filtered_primary_polygon_data.csv", index=False)
    report_df.to_csv(processed_dir / "filtered_data.csv", index=False)

    kpi_summary, drive_summary = run_kpi_analysis(
        report_df,
        user_id=user_id,
        kpi_config=KPI_CONFIG,
        session_ids=session_ids,
        image_dir=str(kpi_analy_dir),
    )

    # FIX 2: overwrite band_table.png and band_pie.png with corrected values
    corrected_band_summary = _regenerate_band_images_corrected(report_df, str(kpi_analy_dir))
    native_tables = {
        "pci_poor_rsrp.png": _build_frontend_like_poor_pci_table(report_df, "rsrp", -105),
        "pci_poor_rsrq.png": _build_frontend_like_poor_pci_table(report_df, "rsrq", -14),
    }
    native_tables.update(_build_band_native_table(corrected_band_summary))
    native_tables.update(_build_drive_native_tables(drive_summary))
    native_tables.update(_build_app_analytics_native_tables(report_df))
    native_tables.update(_build_misc_native_tables(report_df))
    native_tables = {
        filename: table_df
        for filename, table_df in native_tables.items()
        if table_df is not None and not table_df.empty
    }

    # ── 7. CDF PLOTS ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[7/9] CDF plots (from report filtered data)")
    print("=" * 70)
    _generate_all_cdf_plots_local(report_df, kpi_analy_dir)

    # ── 8. METADATA (FIX 2: inject corrected band_summary) ───────────────────
    print("\n" + "=" * 70)
    print("[8/9] Building metadata")
    print("=" * 70)
    metadata = build_metadata(
        report_df,
        kpi_analysis_results=kpi_summary,
        drive_summary_data=drive_summary,
    )

    if corrected_band_summary:
        # Replace LLM-facing band_summary so text % matches the table exactly
        metadata["band_summary"] = corrected_band_summary
        print(
            f"  [FIX 2] band_summary replaced — "
            f"{len(corrected_band_summary)} known bands, Unknown excluded"
        )

    native_tables.update(_build_kpi_range_native_tables(metadata))
    native_tables = {
        filename: table_df
        for filename, table_df in native_tables.items()
        if table_df is not None and not table_df.empty
    }
    if native_tables:
        print(
            "  [FIX 6] native ReportLab tables prepared for: "
            + ", ".join(sorted(native_tables.keys()))
        )

    metadata_path = processed_dir / "report_metadata.json"
    write_metadata_file(metadata, str(metadata_path))
    print(f"  report_metadata.json written")

    # ── 9. REPORT TEXT + PDF ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[9/9] Report text + PDF")
    print("=" * 70)

    report_text_path = processed_dir / "report_text.json"

    if skip_llm:
        # Minimal fallback so PDF can still be built without an API call
        fallback = {
            "Introduction": metadata.get("introduction", ""),
            "Area Summary": metadata.get("area_summary") or {},
            "Drive Summary": "",
            "KPI Summary": "",
            "Map View - Band": "",
            "Map View - RSRP": "",
            "Map View - RSRQ": "",
            "Map View - SINR": "",
            "Map View - DL Throughput": "",
            "Map View - UL Throughput": "",
            "Map View - MOS": "",
            "PCI Summary": "",
        }
        _write_json(report_text_path, fallback)
        report_text = fallback
        print("  LLM skipped — minimal fallback text used")
    else:
        report_text = generate_report_text(
            metadata=metadata,
            output_path=str(report_text_path),
            verbose=True,
        )

    pdf_path = run_dir / "report.pdf"
    _build_pdf_crisp(
        metadata_path=str(metadata_path),
        report_text_path=str(report_text_path),
        output_path=str(pdf_path),
        images_dir=str(images_dir),
        scratch_dir=str(run_dir / "_img_opt"),
        native_tables=native_tables,
        verbose=True,
    )
    pdf_mb = pdf_path.stat().st_size / (1024 * 1024) if pdf_path.exists() else 0
    print(f"\n  PDF: {pdf_path}  ({pdf_mb:.1f} MB)  ✓")

    # ── DEBUG SUMMARY ─────────────────────────────────────────────────────────
    raw_df.to_csv(processed_dir / "raw_data.csv", index=False)
    _write_json(processed_dir / "project_meta.json", project_meta)

    summary = {
        "project_id": project_id,
        "user_id": user_id,
        "session_ids": session_ids,
        "fixes_applied": [
            "FIX1_maps_data_bounds_only__no_polygon_viewport_expansion",
            "FIX2_band_table_and_metadata_use_same_denominator__Unknown_excluded",
            "FIX3_handover_uses_all_cells_polygon_filtered__matches_frontend",
            "FIX4_pci_poor_rsrp_rsrq_use_raw_samples_top15_native_tables_keep_together",
            "FIX5_report_and_cdf_use_primary_polygon_known_band_dataframe",
        ],
        "row_counts": {
            "raw": int(len(raw_df)),
            "filtered_primary_polygon": int(len(filtered_df)),
            "report_primary_polygon_known_band": int(len(report_df)),
            "handover_polygon_only_all_cells": int(len(handover_df)),
        },
        "polygon_wkt_prefix": (polygon_wkt[:80] + "…") if polygon_wkt and len(polygon_wkt) > 80 else polygon_wkt,
        "pdf_path": str(pdf_path),
        "output_dir": str(run_dir),
    }
    _write_json(run_dir / "full_report_summary.json", summary)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Output: {run_dir}")
    print(
        f"Rows — raw: {len(raw_df)}, "
        f"filtered (primary+polygon): {len(filtered_df)}, "
        f"report (primary+polygon+known band): {len(report_df)}, "
        f"handover (all cells, polygon): {len(handover_df)}"
    )
    print("=" * 70)
    return run_dir


# ═══════════════════════════════════════════════════════════════════════════════
# DEBUG MODE — BASE ROUTE MAP + ONE KPI MAP
# (original behaviour preserved for quick per-KPI visual checks)
# ═══════════════════════════════════════════════════════════════════════════════

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
        tiles=tile_name, zoom_control=True, control_scale=False,
        prefer_canvas=True, max_zoom=22,
    )
    add_fullscreen_css(m)

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=(r["lat"], r["lon"]), radius=4,
            color="#2b8cbe", fill=True, fill_opacity=0.95,
        ).add_to(m)

    if polygon_wkt:
        geom = loads(polygon_wkt)
        polygon_latlon = [(c[1], c[0]) for c in geom.exterior.coords]
        folium.Polygon(
            locations=polygon_latlon, color="red", weight=4,
            fill=False, opacity=1.0, tooltip="Polygon Boundary",
        ).add_to(m)

    _fit_tight_bounds(
        m, get_df_bounds(df),
        left_padding_px=left_padding_px, right_padding_px=right_padding_px,
        top_padding_px=top_padding_px, bottom_padding_px=bottom_padding_px,
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
        tiles=tile_name, zoom_control=True, control_scale=False,
        prefer_canvas=True, max_zoom=22,
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
            location=(row["lat"], row["lon"]), radius=4,
            color=color, fill=True, fill_opacity=0.9,
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
        polygon_latlon = [(c[1], c[0]) for c in geom.exterior.coords]
        folium.Polygon(
            locations=polygon_latlon, color="#FF0000", weight=5,
            fill=False, opacity=1.0, tooltip="Polygon Boundary",
        ).add_to(m)

    _fit_tight_bounds(
        m, get_df_bounds(df),
        left_padding_px=left_padding_px, right_padding_px=legend_right_padding_px,
        top_padding_px=top_padding_px, bottom_padding_px=bottom_padding_px,
    )
    m.save(str(output_html))


def _range_index_for_value(value: float, ranges: list[dict]) -> int:
    for idx, range_def in enumerate(ranges):
        if value_in_range(value, range_def, idx == len(ranges) - 1):
            return idx
    return -1


def _range_color_for_index(range_index: int, ranges: list[dict]) -> str:
    if 0 <= range_index < len(ranges):
        return ranges[range_index]["color"]
    return "#808080"


def _generalize_range_kpi_points(
    df: pd.DataFrame,
    kpi_column: str,
    ranges: list[dict],
    *,
    render_width: int,
    render_height: int,
    cell_px: int,
    max_points: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Cartographic point thinning for printable report maps.

    The full dataframe remains the source for analysis and legends.  This only
    selects a deterministic representative set for drawing: one point per
    screen-space cell per KPI range/color, then coarsens the cell size until the
    rendered marker count is safe for Folium/Chromium.
    """
    work = df.dropna(subset=["lat", "lon", kpi_column]).copy()
    work[kpi_column] = pd.to_numeric(work[kpi_column], errors="coerce")
    work = work.dropna(subset=[kpi_column])
    if work.empty:
        return work, {"input_rows": 0, "draw_rows": 0, "cell_px": cell_px}

    work["__range_index"] = work[kpi_column].apply(lambda value: _range_index_for_value(float(value), ranges))

    min_lat, max_lat = float(work["lat"].min()), float(work["lat"].max())
    min_lon, max_lon = float(work["lon"].min()), float(work["lon"].max())
    lat_span = max(max_lat - min_lat, 1e-12)
    lon_span = max(max_lon - min_lon, 1e-12)

    chosen = work
    chosen_cell_px = max(1, int(cell_px))
    attempts = []

    while True:
        cols = max(1, int(render_width / chosen_cell_px))
        rows = max(1, int(render_height / chosen_cell_px))
        tmp = work.copy()
        tmp["__xbin"] = np.floor(((tmp["lon"].astype(float) - min_lon) / lon_span) * (cols - 1)).astype(int)
        tmp["__ybin"] = np.floor(((tmp["lat"].astype(float) - min_lat) / lat_span) * (rows - 1)).astype(int)
        tmp["__cell_center_lon"] = min_lon + ((tmp["__xbin"] + 0.5) / cols) * lon_span
        tmp["__cell_center_lat"] = min_lat + ((tmp["__ybin"] + 0.5) / rows) * lat_span
        tmp["__cell_dist"] = (
            (tmp["lon"].astype(float) - tmp["__cell_center_lon"]) ** 2
            + (tmp["lat"].astype(float) - tmp["__cell_center_lat"]) ** 2
        )
        idx = (
            tmp.sort_values(["__cell_dist", kpi_column])
            .groupby(["__ybin", "__xbin", "__range_index"], sort=False)
            .head(1)
            .index
        )
        chosen = tmp.loc[idx].copy()
        attempts.append({"cell_px": int(chosen_cell_px), "draw_rows": int(len(chosen))})

        if len(chosen) <= max_points or chosen_cell_px >= 96:
            break
        chosen_cell_px = int(chosen_cell_px * 1.35) + 1

    if len(chosen) > max_points:
        parts = []
        full_counts = work["__range_index"].value_counts(normalize=True)
        for range_index, share in full_counts.items():
            part = chosen[chosen["__range_index"] == range_index]
            take = max(1, int(round(max_points * float(share))))
            if len(part) > take:
                part = part.sort_values(["__ybin", "__xbin", "__cell_dist"]).iloc[:: max(1, len(part) // take)].head(take)
            parts.append(part)
        chosen = pd.concat(parts, ignore_index=False).head(max_points)

    helper_cols = [
        "__xbin", "__ybin", "__cell_center_lon", "__cell_center_lat", "__cell_dist",
    ]
    summary = {
        "method": "cartographic_point_generalization_grid_per_kpi_range",
        "input_rows": int(len(work)),
        "draw_rows": int(len(chosen)),
        "cell_px": int(chosen_cell_px),
        "max_points": int(max_points),
        "attempts": attempts,
        "full_range_counts": {
            str(int(k)): int(v)
            for k, v in work["__range_index"].value_counts().sort_index().items()
        },
        "draw_range_counts": {
            str(int(k)): int(v)
            for k, v in chosen["__range_index"].value_counts().sort_index().items()
        },
    }
    return chosen.drop(columns=[c for c in helper_cols if c in chosen.columns]), summary


def _generate_range_kpi_map_cartographic_with_tile(
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
    render_width: int,
    render_height: int,
    generalization_cell_px: int,
    generalized_max_points: int,
) -> dict:
    full_df = df.dropna(subset=["lat", "lon", kpi_column]).copy()
    full_df[kpi_column] = pd.to_numeric(full_df[kpi_column], errors="coerce")
    full_df = full_df.dropna(subset=[kpi_column])
    if full_df.empty:
        raise ValueError(f"No data available for KPI map: {kpi_column}")

    draw_df, summary = _generalize_range_kpi_points(
        full_df,
        kpi_column,
        ranges,
        render_width=render_width,
        render_height=render_height,
        cell_px=generalization_cell_px,
        max_points=generalized_max_points,
    )

    m = folium.Map(
        tiles=tile_name, zoom_control=True, control_scale=False,
        prefer_canvas=True, max_zoom=22,
    )
    add_fullscreen_css(m)

    for _, row in draw_df.iterrows():
        color = _range_color_for_index(int(row.get("__range_index", -1)), ranges)
        folium.CircleMarker(
            location=(row["lat"], row["lon"]), radius=4,
            color=color, fill=True, fill_opacity=0.9,
        ).add_to(m)

    # Legend intentionally uses the full dataframe, not the thinned draw set.
    legend_items = build_legend_from_ranges(full_df, kpi_column, ranges)
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
        polygon_latlon = [(c[1], c[0]) for c in geom.exterior.coords]
        folium.Polygon(
            locations=polygon_latlon, color="#FF0000", weight=5,
            fill=False, opacity=1.0, tooltip="Polygon Boundary",
        ).add_to(m)

    _fit_tight_bounds(
        m, get_df_bounds(full_df),
        left_padding_px=left_padding_px, right_padding_px=legend_right_padding_px,
        top_padding_px=top_padding_px, bottom_padding_px=bottom_padding_px,
    )
    m.save(str(output_html))
    summary["legend_source"] = "full_kpi_dataframe_not_generalized"
    summary["legend_items"] = [
        {"label": label, "color": color, "count": int(count)}
        for label, color, count in legend_items
    ]
    return summary


def generate_project_rsrp_cartographic_image(
    project_id: int,
    user_id: int | None,
    out_dir: Path,
    render_timeout_ms: int,
    candidate_width: int,
    candidate_height: int,
    tile_name: str,
    left_padding_px: int,
    legend_right_padding_px: int,
    top_padding_px: int,
    bottom_padding_px: int,
    legend_top_px: int,
    legend_right_px: int,
    generalization_cell_px: int,
    generalized_max_points: int,
) -> Path:
    """
    Project-level RSRP-only validation image using cartographic point thinning.
    This follows the production report data path, but only writes test artifacts.
    """
    os.environ.setdefault("REPORT_RENDER_TIMEOUT_MS", str(render_timeout_ms))
    os.environ.setdefault("REPORT_RENDER_NAV_ATTEMPTS", "1")

    run_dir = _make_run_dir(out_dir, project_id, "rsrp_cartographic")
    html_dir = run_dir / "html"
    image_dir = run_dir / "images" / "kpi_maps"
    processed_dir = run_dir / "processed"
    for d in (html_dir, image_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw_df, filtered_df, project_meta = load_project_data(project_id)
    polygon_wkt = project_meta.get("region")
    if filtered_df.empty:
        raise ValueError(f"Project {project_id} has no rows after report filtering.")

    if project_meta.get("report_data_prefiltered"):
        report_df = filtered_df.reset_index(drop=True)
    else:
        report_df = _filter_known_band_rows(filtered_df)
    if report_df.empty:
        raise ValueError(f"Project {project_id} has no known-band rows after report filtering.")

    kpi_cfg = KPI_CONFIG["RSRP"]
    kpi_column = kpi_cfg["column"]
    kpi_df = report_df[
        report_df[kpi_column].notna()
        & report_df["lat"].notna()
        & report_df["lon"].notna()
    ].copy()
    ranges = resolve_kpi_ranges(kpi_name="RSRP", user_id=user_id, values=kpi_df[kpi_column])

    safe_tile = tile_name.lower().replace(" ", "_")
    html_path = html_dir / f"rsrp_map_{safe_tile}_cartographic.html"
    png_path = image_dir / f"rsrp_map_{safe_tile}_cartographic.png"

    generalization_summary = _generate_range_kpi_map_cartographic_with_tile(
        kpi_df,
        kpi_column=kpi_column,
        ranges=ranges,
        polygon_wkt=polygon_wkt,
        output_html=html_path,
        tile_name=tile_name,
        left_padding_px=left_padding_px,
        legend_right_padding_px=legend_right_padding_px,
        top_padding_px=top_padding_px,
        bottom_padding_px=bottom_padding_px,
        legend_top_px=legend_top_px,
        legend_right_px=legend_right_px,
        render_width=candidate_width,
        render_height=candidate_height,
        generalization_cell_px=generalization_cell_px,
        generalized_max_points=generalized_max_points,
    )
    html_to_png(str(html_path), str(png_path), width=candidate_width, height=candidate_height)

    raw_df.to_csv(processed_dir / "raw_data.csv", index=False)
    filtered_df.to_csv(processed_dir / "filtered_primary_polygon_data.csv", index=False)
    report_df.to_csv(processed_dir / "report_primary_polygon_known_band_data.csv", index=False)
    _write_json(processed_dir / "project_meta.json", project_meta)

    summary = {
        "project_id": project_id,
        "user_id": user_id,
        "mode": "rsrp_cartographic_validation",
        "session_ids": _parse_session_ids(project_meta.get("ref_session_id")),
        "raw": _route_summary(raw_df, polygon_wkt),
        "filtered_primary_polygon": _route_summary(filtered_df, polygon_wkt),
        "report_primary_polygon_known_band": _route_summary(report_df, polygon_wkt),
        "kpi_name": "RSRP",
        "kpi_column": kpi_column,
        "tile_name": tile_name,
        "png_path": str(png_path),
        "html_path": str(html_path),
        "generalization": generalization_summary,
    }
    _write_json(run_dir / "rsrp_cartographic_summary.json", summary)

    print(f"Generated cartographic RSRP image: {png_path}")
    print(f"Full RSRP rows: {generalization_summary['input_rows']}")
    print(f"Drawn RSRP rows: {generalization_summary['draw_rows']}")
    print(f"Summary: {run_dir / 'rsrp_cartographic_summary.json'}")
    return run_dir


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
    Debug mode: generate base route image + one KPI map for visual inspection.
    Uses per-parameter tile and padding controls for rapid iteration.
    """
    os.environ.setdefault("REPORT_RENDER_TIMEOUT_MS", str(render_timeout_ms))
    os.environ.setdefault("REPORT_RENDER_NAV_ATTEMPTS", "1")

    run_dir = _make_run_dir(out_dir, project_id, "base_map_debug")
    html_dir      = run_dir / "html"
    image_dir     = run_dir / "images" / "kpi_maps"
    processed_dir = run_dir / "processed"
    for d in (html_dir, image_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw_df, filtered_df, project_meta = load_project_data(project_id)
    polygon_wkt = project_meta.get("region")

    if filtered_df.empty:
        raise ValueError(f"Project {project_id} has no rows after report filtering.")

    safe_tile  = tile_name.lower().replace(" ", "_")
    html_path  = html_dir / f"base_route_map_{safe_tile}.html"
    compact_png = image_dir / f"base_route_map_{safe_tile}_compact.png"
    safe_kpi   = kpi_name.lower()
    kpi_html   = html_dir / f"{safe_kpi}_map_{safe_tile}.html"
    kpi_png    = image_dir / f"{safe_kpi}_map_{safe_tile}_compact.png"

    _generate_base_route_map_with_tile(
        filtered_df, polygon_wkt, html_path, tile_name,
        left_padding_px=left_padding_px,
        right_padding_px=base_right_padding_px,
        top_padding_px=top_padding_px,
        bottom_padding_px=bottom_padding_px,
    )
    html_to_png(str(html_path), str(compact_png), width=candidate_width, height=candidate_height)

    kpi_cfg = KPI_CONFIG.get(kpi_name)
    if not kpi_cfg or kpi_cfg.get("type") != "range":
        raise ValueError(f"Only range KPI maps are supported in debug mode: {kpi_name}")

    kpi_column = kpi_cfg["column"]
    kpi_df = filtered_df[
        filtered_df[kpi_column].notna()
        & filtered_df["lat"].notna()
        & filtered_df["lon"].notna()
    ]
    ranges = resolve_kpi_ranges(kpi_name=kpi_name, user_id=user_id, values=kpi_df[kpi_column])
    _generate_range_kpi_map_with_tile(
        kpi_df, kpi_column=kpi_column, ranges=ranges,
        polygon_wkt=polygon_wkt, output_html=kpi_html, tile_name=tile_name,
        left_padding_px=left_padding_px,
        legend_right_padding_px=legend_right_padding_px,
        top_padding_px=top_padding_px,
        bottom_padding_px=bottom_padding_px,
        legend_top_px=legend_top_px,
        legend_right_px=legend_right_px,
    )
    html_to_png(str(kpi_html), str(kpi_png), width=candidate_width, height=candidate_height)

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
        },
        "legend_position_px": {"top": legend_top_px, "right": legend_right_px},
        "compact_viewport": {
            "width": candidate_width,
            "height": candidate_height,
            "png_path": str(compact_png),
        },
        "kpi_compact_viewport": {
            "width": candidate_width,
            "height": candidate_height,
            "png_path": str(kpi_png),
        },
        "html_path": str(html_path),
        "kpi_html_path": str(kpi_html),
    }
    _write_json(run_dir / "base_route_map_debug_summary.json", summary)

    print(f"Generated compact base route image: {compact_png}")
    print(f"Generated compact KPI image: {kpi_png}")
    print(f"Debug summary: {run_dir / 'base_route_map_debug_summary.json'}")
    return run_dir


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Report engine debug/validation test.\n"
            "  --mode debug  : generate base route map + one KPI map (default)\n"
            "  --mode rsrp-cartographic : generate one generalized RSRP map image\n"
            "  --mode full   : generate complete PDF report with all fixes applied"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project-id",  type=int, default=272)
    parser.add_argument("--user-id",     type=int, default=13)
    parser.add_argument(
        "--mode", choices=["debug", "rsrp-cartographic", "full"], default="debug",
        help="'debug' = base map + one KPI; 'rsrp-cartographic' = one generalized RSRP map; 'full' = complete PDF with fixes",
    )
    parser.add_argument(
        "--out-dir",
        default="tests/output/report_engine_base_map_debug",
        help="Output directory (relative to ML/ or absolute).",
    )

    # ── full-mode specific ────────────────────────────────────────────────────
    # Full-report map viewport.  1200×900 keeps the route filling the frame and
    # the PNGs small; 1920×1200 left big gray margins and bloated the PDF.
    parser.add_argument("--render-width",  type=int, default=1200)
    parser.add_argument("--render-height", type=int, default=900)
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM text generation (full mode only); use minimal fallback text.",
    )

    # ── debug-mode specific ───────────────────────────────────────────────────
    parser.add_argument("--render-timeout-ms", type=int, default=20000)
    parser.add_argument("--candidate-width",   type=int, default=1000)
    parser.add_argument("--candidate-height",  type=int, default=760)
    parser.add_argument("--tile-name",         default="CartoDB Voyager")
    parser.add_argument("--kpi-name",          default="RSRP", choices=sorted(KPI_CONFIG.keys()))
    parser.add_argument("--left-padding-px",          type=int, default=0)
    parser.add_argument("--base-right-padding-px",    type=int, default=120)
    parser.add_argument("--legend-right-padding-px",  type=int, default=340)
    parser.add_argument("--top-padding-px",           type=int, default=0)
    parser.add_argument("--bottom-padding-px",        type=int, default=10)
    parser.add_argument("--legend-top-px",            type=int, default=44)
    parser.add_argument("--legend-right-px",          type=int, default=6)
    parser.add_argument(
        "--generalization-cell-px",
        type=int,
        default=8,
        help="Screen-space cell size for cartographic point thinning.",
    )
    parser.add_argument(
        "--generalized-max-points",
        type=int,
        default=15000,
        help="Maximum marker count for cartographic validation image.",
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    if args.mode == "full":
        # Bound the per-map screenshot waits so a slow/blocked tile server can't
        # stall each map for ~100s.  Data dots render regardless of tiles.
        os.environ.setdefault("REPORT_RENDER_TIMEOUT_MS", str(args.render_timeout_ms))
        generate_full_report(
            project_id=args.project_id,
            user_id=args.user_id,
            out_dir=out_dir,
            render_width=args.render_width,
            render_height=args.render_height,
            skip_llm=args.skip_llm,
        )
    elif args.mode == "rsrp-cartographic":
        generate_project_rsrp_cartographic_image(
            project_id=args.project_id,
            user_id=args.user_id,
            out_dir=out_dir,
            render_timeout_ms=args.render_timeout_ms,
            candidate_width=args.candidate_width,
            candidate_height=args.candidate_height,
            tile_name=args.tile_name,
            left_padding_px=args.left_padding_px,
            legend_right_padding_px=args.legend_right_padding_px,
            top_padding_px=args.top_padding_px,
            bottom_padding_px=args.bottom_padding_px,
            legend_top_px=args.legend_top_px,
            legend_right_px=args.legend_right_px,
            generalization_cell_px=args.generalization_cell_px,
            generalized_max_points=args.generalized_max_points,
        )
    else:
        generate_project_base_route_image(
            project_id=args.project_id,
            user_id=args.user_id,
            out_dir=out_dir,
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
