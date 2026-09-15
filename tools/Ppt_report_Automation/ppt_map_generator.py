from altair import value
import folium
from shapely.wkt import loads
import colorsys
import json
import pandas as pd
import math
import os
import re

REPORT_MAP_MAX_ZOOM = 22
REPORT_MAP_PADDING_PX = 90
REPORT_MAP_LEGEND_RIGHT_PADDING_PX = 420

# -----------------------------------------------------
# PPT REPORT MAP HELPERS — Google Maps basemap + fractional zoom
# Uses clean Google Maps tiles without watermark, matching mobile & web map view.
# -----------------------------------------------------
DEFAULT_GOOGLE_MAPS_TILES = "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
REPORT_TILE = os.getenv("REPORT_TILE", DEFAULT_GOOGLE_MAPS_TILES)
REPORT_TILE_ATTR = os.getenv("REPORT_TILE_ATTR", "Google")
REPORT_PAD_LEFT = 30
REPORT_PAD_VERT = 30
REPORT_PAD_RIGHT_BASE = 40
REPORT_PAD_RIGHT_LEGEND = 340
REPORT_POINT_GENERALIZATION_THRESHOLD = int(os.getenv("REPORT_POINT_GENERALIZATION_THRESHOLD", "50000"))
REPORT_POOR_POINT_GENERALIZATION_THRESHOLD = int(os.getenv("REPORT_POOR_POINT_GENERALIZATION_THRESHOLD", "20000"))
REPORT_GENERALIZED_MAX_POINTS = int(os.getenv("REPORT_GENERALIZED_MAX_POINTS", "15000"))
REPORT_POOR_GENERALIZED_MAX_POINTS = int(os.getenv("REPORT_POOR_GENERALIZED_MAX_POINTS", "8000"))
REPORT_GENERALIZATION_CELL_PX = int(os.getenv("REPORT_GENERALIZATION_CELL_PX", "8"))
REPORT_RENDER_WIDTH_HINT = int(os.getenv("REPORT_RENDER_WIDTH", "1370"))
REPORT_RENDER_HEIGHT_HINT = int(os.getenv("REPORT_RENDER_HEIGHT", "900"))


def new_report_map():
    """Folium map with Google Maps tiles and fractional zoom (zoomSnap=0) for PPT reports."""
    if REPORT_TILE and (REPORT_TILE.startswith("http://") or REPORT_TILE.startswith("https://")):
        m = folium.Map(
            tiles=None,
            zoom_control=True,
            control_scale=False,
            prefer_canvas=True,
            max_zoom=REPORT_MAP_MAX_ZOOM,
            zoomSnap=0,       # continuous fractional zoom (no integer snapping)
            zoomDelta=0.25,
        )
        folium.TileLayer(
            tiles=REPORT_TILE,
            attr=REPORT_TILE_ATTR,
            name="Google Maps",
            overlay=False,
            control=False,
            max_zoom=REPORT_MAP_MAX_ZOOM,
        ).add_to(m)
        return m

    return folium.Map(
        tiles=REPORT_TILE,
        zoom_control=True,
        control_scale=False,
        prefer_canvas=True,
        max_zoom=REPORT_MAP_MAX_ZOOM,
        zoomSnap=0,       # continuous fractional zoom (no integer snapping)
        zoomDelta=0.25,
    )


def fit_data_bounds(m, df_or_bounds, reserve_legend_space=False):
    """Fit the viewport to the GPS data bounds only (not the polygon). Accepts DataFrame or [[min_lat, min_lon], [max_lat, max_lon]]."""
    if isinstance(df_or_bounds, (list, tuple)):
        bounds = df_or_bounds
    else:
        bounds = get_df_bounds(df_or_bounds)
    right = REPORT_PAD_RIGHT_LEGEND if reserve_legend_space else REPORT_PAD_RIGHT_BASE
    m.fit_bounds(
        bounds,
        padding_top_left=(REPORT_PAD_LEFT, REPORT_PAD_VERT),
        padding_bottom_right=(right, REPORT_PAD_VERT),
        max_zoom=REPORT_MAP_MAX_ZOOM,
    )


def draw_polygon_overlay(m, polygon_wkt, color="#111827", weight=2,
                         opacity=0.45, dash_array="6,6"):
    """Draw the project polygon as an outline that does NOT expand the viewport."""
    if not polygon_wkt:
        return
    geom = loads(polygon_wkt)
    latlon = [(c[1], c[0]) for c in geom.exterior.coords]
    folium.Polygon(
        locations=latlon, color=color, weight=weight, fill=False,
        opacity=opacity, dash_array=dash_array, tooltip="Polygon Boundary",
    ).add_to(m)


# Helper function to add legend to folium map

def add_fullscreen_css(m):
    css = """
    <style>
        html, body { width: 100%; height: 100%; margin: 0; padding: 0; }
        .folium-map { width: 100% !important; height: 100% !important; position: relative; }
        .leaflet-control-container { display: none !important; }
    </style>
    """
    m.get_root().header.add_child(folium.Element(css))


def fit_report_bounds(m, bounds, reserve_legend_space=False):
    """Fit report map content tightly with a print-friendly margin."""
    right_padding = (
        REPORT_MAP_LEGEND_RIGHT_PADDING_PX
        if reserve_legend_space
        else REPORT_MAP_PADDING_PX
    )
    m.fit_bounds(
        bounds,
        padding_top_left=(REPORT_MAP_PADDING_PX, REPORT_MAP_PADDING_PX),
        padding_bottom_right=(right_padding, REPORT_MAP_PADDING_PX),
        max_zoom=REPORT_MAP_MAX_ZOOM,
    )


def add_legend(m, title, items):
    """
    items = [(label, color, count)]
    """
    # Style is added to <head>; legend is injected into the map div so
    # Playwright clipping to the map captures the legend.
    legend_css = """
    <style>
        .kpi-legend {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 320px;
            max-height: calc(100% - 40px);
            overflow: auto;
            z-index: 9999;
            background-color: rgba(255, 255, 255, 0.98);
            color: #000;
            padding: 18px 16px;
            border-radius: 8px;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 18px;
            line-height: 1.6;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
            border: 2px solid rgba(0,0,0,0.15);
        }
        .kpi-legend-title {
            font-weight: 700;
            font-size: 20px;
            margin-bottom: 12px;
            color: #000;
            border-bottom: 2px solid #333;
            padding-bottom: 8px;
        }
        .kpi-legend-row {
            margin-top: 10px;
            display: flex;
            align-items: center;
            font-size: 18px;
        }
        .kpi-legend-swatch {
            width: 22px;
            height: 22px;
            border-radius: 3px;
            margin-right: 10px;
            flex: 0 0 auto;
            border: 1px solid rgba(0,0,0,0.2);
        }
    </style>
    """
    m.get_root().header.add_child(folium.Element(legend_css))

    rows_html = ""
    for label, color, count in items:
        rows_html += (
            f"<div class='kpi-legend-row'>"
            f"<span class='kpi-legend-swatch' style='background:{color};'></span>"
            f"<span>{label} : {count}</span>"
            f"</div>"
        )

    legend_inner_html = (
        f"<div class='kpi-legend'>"
        f"<div class='kpi-legend-title'>{title}</div>"
        f"{rows_html}"
        f"</div>"
    )

    payload = json.dumps(legend_inner_html)
    legend_js = f"""
    <script>
        (function() {{
            function injectLegend() {{
                var mapEl = document.querySelector('.folium-map');
                if (!mapEl) return;
                mapEl.style.position = 'relative';
                var existing = mapEl.querySelector('.kpi-legend');
                if (existing) existing.remove();
                var wrapper = document.createElement('div');
                wrapper.innerHTML = {payload};
                mapEl.appendChild(wrapper.firstElementChild);
            }}
            setTimeout(injectLegend, 250);
        }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(legend_js))





# heper function to get polygon bounds


def get_polygon_bounds(polygon_wkt):
    geom = loads(polygon_wkt)
    
    # WKT standard format is (lon, lat) from MySQL ST_AsText
    # Extract coordinates: shapely returns (x, y) = (lon, lat)
    coords = list(geom.exterior.coords)
    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]

    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)

    return [
        [min_lat, min_lon],
        [max_lat, max_lon],
    ]


def get_df_bounds(df):
    df = df.dropna(subset=["lat", "lon"])
    if df.empty:
        raise ValueError("No GPS data to compute bounds")
    return [
        [float(df["lat"].min()), float(df["lon"].min())],
        [float(df["lat"].max()), float(df["lon"].max())],
    ]


def merge_bounds(b1, b2):
    return [
        [min(b1[0][0], b2[0][0]), min(b1[0][1], b2[0][1])],
        [max(b1[1][0], b2[1][0]), max(b1[1][1], b2[1][1])],
    ]


def force_zoom_in(m, zoom_delta=2):
    """
    Force zoom-in AFTER fit_bounds.
    zoom_delta = how many levels to zoom in.
    """
    script = f"""
    <script>
        setTimeout(function() {{
            var el = document.querySelector('.folium-map');
            var mapId = el && el.id;
            var map = mapId && window[mapId];
            if (map && typeof map.getZoom === 'function') {{
                map.setZoom(map.getZoom() + {zoom_delta});
            }}
        }}, 300);
    </script>
    """
    m.get_root().html.add_child(folium.Element(script))


def expand_bounds(bounds, expand_factor=0.02):
    """
    Expands bounds by a small percentage to avoid clipping.
    bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    Reduced factor to minimize white space.
    """

    min_lat, min_lon = bounds[0]
    max_lat, max_lon = bounds[1]

    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    return [
        [min_lat - lat_range * expand_factor,
         min_lon - lon_range * expand_factor],
        [max_lat + lat_range * expand_factor,
         max_lon + lon_range * expand_factor],
    ]



# Helper function to build legend items from color function

def value_in_range(value, range_dict, is_last_range):
    """
    Check if value belongs to range using half-open intervals.
    - All ranges except last: min <= value < max
    - Last range: min <= value <= max
    This prevents double-counting at boundaries.
    """
    if is_last_range:
        return range_dict["min"] <= value <= range_dict["max"]
    else:
        return range_dict["min"] <= value < range_dict["max"]


def build_legend_from_ranges(df, kpi_column, ranges):
    legend_items = []

    values = pd.to_numeric(df[kpi_column], errors="coerce").dropna()
    if values.empty:
        return []

    for idx, r in enumerate(ranges):
        is_last = (idx == len(ranges) - 1)
        mask = values.apply(lambda v: value_in_range(v, r, is_last))
        count = int(mask.sum())

        if count == 0:
            continue

        label = r.get("range") or f'{r["min"]} to {r["max"]}'
        color = r["color"]

        legend_items.append((label, color, count))

    return legend_items


def _range_key_for_value(value, ranges):
    for idx, range_def in enumerate(ranges):
        if value_in_range(value, range_def, idx == len(ranges) - 1):
            return idx
    return -1


def _prepare_geo_numeric_df(df, value_col=None):
    subset = ["lat", "lon"] + ([value_col] if value_col else [])
    work = df.dropna(subset=subset).copy()
    if work.empty:
        return work

    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    if value_col:
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=["lat", "lon", value_col])
    else:
        work = work.dropna(subset=["lat", "lon"])

    return work[
        work["lat"].between(-90, 90)
        & work["lon"].between(-180, 180)
    ].copy()


def _cartographic_point_thinning(
    df,
    key_col=None,
    render_width=REPORT_RENDER_WIDTH_HINT,
    render_height=REPORT_RENDER_HEIGHT_HINT,
    cell_px=REPORT_GENERALIZATION_CELL_PX,
    max_points=REPORT_GENERALIZED_MAX_POINTS,
):
    """
    Deterministic map-scale point thinning.

    One representative point is kept per screen-space cell and optional
    attribute bucket. Bounds and legends still use the full source dataframe.
    """
    if df.empty:
        return df

    work = df.copy()
    min_lat, max_lat = float(work["lat"].min()), float(work["lat"].max())
    min_lon, max_lon = float(work["lon"].min()), float(work["lon"].max())
    lat_span = max(max_lat - min_lat, 1e-12)
    lon_span = max(max_lon - min_lon, 1e-12)

    chosen = work
    chosen_cell_px = max(1, int(cell_px))
    group_cols = ["__ybin", "__xbin"]
    if key_col and key_col in work.columns:
        group_cols.append(key_col)

    while True:
        cols = max(1, int(render_width / chosen_cell_px))
        rows = max(1, int(render_height / chosen_cell_px))
        tmp = work.copy()
        tmp["__xbin"] = (
            ((tmp["lon"].astype(float) - min_lon) / lon_span) * (cols - 1)
        ).clip(lower=0, upper=cols - 1).astype(int)
        tmp["__ybin"] = (
            ((tmp["lat"].astype(float) - min_lat) / lat_span) * (rows - 1)
        ).clip(lower=0, upper=rows - 1).astype(int)
        tmp["__cell_center_lon"] = min_lon + ((tmp["__xbin"] + 0.5) / cols) * lon_span
        tmp["__cell_center_lat"] = min_lat + ((tmp["__ybin"] + 0.5) / rows) * lat_span
        tmp["__cell_dist"] = (
            (tmp["lon"].astype(float) - tmp["__cell_center_lon"]) ** 2
            + (tmp["lat"].astype(float) - tmp["__cell_center_lat"]) ** 2
        )
        idx = (
            tmp.sort_values(["__cell_dist"])
            .groupby(group_cols, sort=False)
            .head(1)
            .index
        )
        chosen = tmp.loc[idx].copy()

        if len(chosen) <= max_points or chosen_cell_px >= 96:
            break
        chosen_cell_px = int(chosen_cell_px * 1.35) + 1

    helper_cols = [
        "__xbin", "__ybin", "__cell_center_lon", "__cell_center_lat", "__cell_dist",
    ]
    return chosen.drop(columns=[c for c in helper_cols if c in chosen.columns])


def _maybe_generalize_points(
    df,
    threshold=REPORT_POINT_GENERALIZATION_THRESHOLD,
    key_col=None,
    max_points=REPORT_GENERALIZED_MAX_POINTS,
):
    if len(df) <= threshold:
        return df
    draw_df = _cartographic_point_thinning(df, key_col=key_col, max_points=max_points)
    print(
        "[ReportMaps] cartographic point generalization "
        f"rows_before={len(df)} rows_drawn={len(draw_df)} threshold={threshold}"
    )
    return draw_df



# Data validation functions 

def has_valid_numeric_data(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False

    values = pd.to_numeric(df[column], errors="coerce")
    return values.notna().any()


def has_valid_categorical_data(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False

    return df[column].dropna().astype(str).str.strip().ne("").any()



# Debug map generation function

def generate_debug_map(df, polygon_wkt, output_path, sample_points=50):
    """
    Debug map to visually inspect:
    - GPS route
    - Polygon shape
    - Polygon vertex points
    """

    df = df.dropna(subset=["Latitude", "Longitude"])

    if df.empty:
        raise ValueError("No GPS data to plot")

    # Center map on GPS data
    m = new_report_map()

    add_fullscreen_css(m)

    # 1 GPS route (BLUE)
    folium.PolyLine(
        locations=list(zip(df["lat"], df["lon"])),
        color="blue",
        weight=3,
        opacity=0.7,
        tooltip="GPS Route"
    ).add_to(m)

    

    # 2 Polygon boundary + vertices
    geom = loads(polygon_wkt)

    # WKT format is (lon, lat), convert to (lat, lon) for folium
    polygon_latlon = [(coord[1], coord[0]) for coord in geom.exterior.coords]

    # Polygon outline (RED)
    folium.Polygon(
        locations=polygon_latlon,
        color="red",
        weight=4,
        fill=False,
        tooltip="Polygon Boundary"
    ).add_to(m)

    # 3 Polygon vertex markers (NUMBERED)
    for idx, (lat, lon) in enumerate(polygon_latlon):
        folium.Marker(
            location=(lat, lon),
            popup=f"Vertex {idx}\nLat: {lat}\nLon: {lon}",
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 10px;
                    color: red;
                    font-weight: bold;
                ">
                    {idx}
                </div>
                """
            )
        ).add_to(m)

    # Fit view to polygon + route with minimal padding
    bounds = merge_bounds(get_df_bounds(df), get_polygon_bounds(polygon_wkt))
    bounds = expand_bounds(bounds, expand_factor=0.02)
    fit_report_bounds(m, bounds)

    m.save(output_path)


# KPI map generation function

def generate_kpi_map(df, kpi_column, color_func, ranges, output_html, polygon_wkt=None, fixed_bounds=None):
    # Drop rows with missing lat/lon or KPI values
    df = _prepare_geo_numeric_df(df, kpi_column)

    if df.empty:
        raise ValueError("No data available for KPI map")

    df["__report_range_key"] = df[kpi_column].apply(lambda value: _range_key_for_value(float(value), ranges))
    draw_df = _maybe_generalize_points(
        df,
        threshold=REPORT_POINT_GENERALIZATION_THRESHOLD,
        key_col="__report_range_key",
        max_points=REPORT_GENERALIZED_MAX_POINTS,
    )

    m = new_report_map()

    add_fullscreen_css(m)

    for _, row in draw_df.iterrows():
        

        raw_value = row[kpi_column]

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        # Find color using half-open intervals
        color = "#808080"  # default gray
        for idx, r in enumerate(ranges):
            is_last = (idx == len(ranges) - 1)
            if value_in_range(value, r, is_last):
                color = r["color"]
                break

        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.9
        ).add_to(m)

    # 4 Build legend

    legend_items = build_legend_from_ranges(df, kpi_column, ranges)
    add_legend(m, kpi_column, legend_items)

    # Polygon drawn as a dashed overlay — it does NOT expand the viewport.
    draw_polygon_overlay(m, polygon_wkt)

    # Fit the viewport to the data only (fractional zoom keeps it tight).
    fit_data_bounds(m, fixed_bounds if fixed_bounds is not None else df, reserve_legend_space=False)

    m.save(output_html)



# Helper function to generate distinct colors

def generate_distinct_colors(n):
    """
    Generate n visually distinct colors using HSV space.
    Returns a list of hex color strings.
    """
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.85)
        colors.append(
            "#{:02x}{:02x}{:02x}".format(
                int(r * 255),
                int(g * 255),
                int(b * 255)
            )
        )
    return colors


FRONTEND_DYNAMIC_COLOR_PALETTE = [
    "#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#A133FF",
    "#33FFF5", "#FFD133", "#FF8C33", "#8CFF33", "#338CFF",
    "#FF3333", "#33FF8C", "#5733FF", "#FF33D1", "#33FFD1",
    "#D1FF33", "#FF6633", "#66FF33", "#3366FF", "#FF3366",
    "#C70039", "#900C3F", "#581845", "#1A5276", "#148F77",
    "#D4AC0D", "#AF601A", "#6C3483", "#1E8449", "#2874A6",
    "#CB4335", "#7D3C98", "#2E86C1", "#17A589", "#D68910",
    "#BA4A00", "#8E44AD", "#3498DB", "#16A085", "#F39C12",
]


FRONTEND_BAND_COLORS = {
    "GSM 900": "#DC2626",
    "DCS 1800": "#14B8A6",
    "WCDMA": "#7C3AED",
    "B1": "#EF4444",
    "B2": "#F59E0B",
    "B3": "#F97316",
    "B4": "#F59E0B",
    "B5": "#F59E0B",
    "B6": "#EF4444",
    "B7": "#10B981",
    "B8": "#10B981",
    "B9": "#F59E0B",
    "B12": "#3B82F6",
    "B13": "#3B82F6",
    "B17": "#3B82F6",
    "B18": "#10B981",
    "B19": "#EF4444",
    "B20": "#2563EB",
    "B25": "#8B5CF6",
    "B26": "#8B5CF6",
    "B28": "#EC4899",
    "B38": "#6366F1",
    "B39": "#6366F1",
    "B40": "#3B82F6",
    "B41": "#8B5CF6",
    "n5": "#F59E0B",
    "n28": "#EC4899",
    "n78": "#F472B6",
    "Unknown": "#a8a6a2",
}


def _frontend_hash_color(value):
    normalized = str(value or "").lower().strip()
    h = 0
    for ch in normalized:
        h = ((h << 5) - h) + ord(ch)
        h = ((h + 2**31) % 2**32) - 2**31
    return FRONTEND_DYNAMIC_COLOR_PALETTE[abs(h) % len(FRONTEND_DYNAMIC_COLOR_PALETTE)]


def _report_band_palette(count):
    """Build a high-separation palette for the bands present in one report."""
    if count <= 0:
        return []

    colors = []
    hue = 0.02
    hue_step = 0.618033988749895  # golden-ratio spacing keeps adjacent colors apart
    for _ in range(count):
        hue = (hue + hue_step) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 0.92)
        colors.append(
            "#{:02x}{:02x}{:02x}".format(
                int(r * 255),
                int(g * 255),
                int(b * 255)
            )
        )
    return colors


def build_report_band_color_map(bands):
    """Map the known bands in a single report to distinct, repeatable colors."""
    normalized = []
    seen = set()
    for band in bands or []:
        band_name = normalize_band_name(band)
        if band_name == "Unknown" or band_name in seen:
            continue
        seen.add(band_name)
        normalized.append(band_name)

    normalized.sort(key=lambda value: (0 if value.startswith("B") else 1 if value.startswith("n") else 2, value))
    palette = _report_band_palette(len(normalized))
    return {band: color for band, color in zip(normalized, palette)}


def normalize_band_name(band):
    """Python equivalent of the frontend normalizeBandName() helper."""
    if band is None:
        return "Unknown"

    raw = str(band).strip()
    if not raw:
        return "Unknown"

    upper_raw = raw.upper()
    if upper_raw in {"UNKNOWN", "N/A", "NA", "NULL", "UNDEFINED", "-1"}:
        return "Unknown"

    nr_match = re.match(r"^N\s*[-_\s]*([+-]?\d+)$", raw, flags=re.IGNORECASE)
    if nr_match:
        val = abs(int(nr_match.group(1)))
        return f"n{val}" if val > 0 else "Unknown"

    lte_match = re.match(r"^(?:B|BAND)\s*[-_\s]*([+-]?\d+)$", raw, flags=re.IGNORECASE)
    if lte_match:
        val = abs(int(lte_match.group(1)))
        return f"B{val}" if val > 0 else "Unknown"

    try:
        numeric = int(float(raw))
        if numeric != 0:
            return f"B{abs(numeric)}"
    except (TypeError, ValueError):
        pass

    return raw


def get_frontend_band_color(band):
    normalized = normalize_band_name(band)
    if normalized in FRONTEND_BAND_COLORS:
        return FRONTEND_BAND_COLORS[normalized]
    return _frontend_hash_color(normalized)


def normalize_tech_name(tech, band=None, network=None):
    tech_str = str(tech).strip() if tech is not None else ""
    net_str = str(network).strip().upper() if network is not None else ""
    t = tech_str.upper()
    combined = f"{t} {net_str}"

    if tech_str in {
        "000", "00", "Unknown/No Service", "Unknown / No Service",
        "UNKNOWN / NO SERVICE", "Unknown", "undefined", "null",
        "404440", "404011"
    } and not net_str and not band:
        return "Unknown"

    # 1. 4G(LTE Anchor NSA), NSA, and EN-DC are considered 5G
    if (
        "LTE ANCHOR" in combined
        or "LTE-ANCHOR" in combined
        or "LTE_ANCHOR" in combined
        or "ANCHOR" in combined
        or "NSA" in combined
        or "ENDC" in combined
        or "EN-DC" in combined
        or "5G" in combined
        or "NR" in combined
    ):
        return "5G"

    # 2. Check band (5G NR vs 4G LTE)
    if band is not None:
        band_str = str(band).strip().lower()
        if re.match(r"^n\d+", band_str) or band_str in {
            "n78", "n77", "n41", "n1", "n28", "n3", "n5", "n7",
            "n8", "n20", "n38", "n40", "n66", "n71", "n257", "n258", "n260", "n261"
        }:
            return "5G"
        if re.match(r"^[bl]\d+", band_str) or band_str in {
            "b1", "b2", "b3", "b4", "b5", "b7", "b8", "b12", "b13", "b17", "b18", "b19",
            "b20", "b25", "b26", "b28", "b38", "b39", "b40", "b41",
            "1800", "2600", "700", "2100", "900", "850", "2300", "2500"
        }:
            return "4G"

    # 3. Check network string for 4G / 3G / 2G
    if net_str:
        if "4G" in net_str or "LTE" in net_str:
            return "4G"
        if "3G" in net_str or "WCDMA" in net_str or "UMTS" in net_str:
            return "3G"
        if "2G" in net_str or "GSM" in net_str or "EDGE" in net_str or "GPRS" in net_str:
            return "2G"

    # 4. Check tech string
    if not tech_str:
        return "Unknown"

    if "LTE" in t or "4G" in t or "4G+" in t:
        return "4G"
    if "3G" in t or "WCDMA" in t or "UMTS" in t or "HSPA" in t:
        return "3G"
    if "2G" in t or "EDGE" in t or "GSM" in t or "GPRS" in t:
        return "2G"

    return tech_str

# Categorical KPI map generation function

def generate_categorical_kpi_map(df, kpi_column, output_html, polygon_wkt=None, fixed_bounds=None):
    """
    Categorical KPI map for Band / PCI:
    - Unlimited unique values
    - Dynamically generated distinct colors
    """

    df = _prepare_geo_numeric_df(df).dropna(subset=[kpi_column])

    if df.empty:
        raise ValueError(f"No data available for categorical KPI: {kpi_column}")

    m = new_report_map()

    add_fullscreen_css(m)

    # Polygon drawn as a subtle dashed overlay — does NOT expand the viewport.
    draw_polygon_overlay(m, polygon_wkt)

    is_band_map = kpi_column.lower() == "band"
    category_col = "__report_category"

    if is_band_map:
        df = df.copy()
        df[category_col] = df[kpi_column].apply(normalize_band_name)
        # Exclude Unknown so the band map colours match the corrected band table.
        df = df[df[category_col] != "Unknown"]
        if df.empty:
            raise ValueError("No non-Unknown band data for band map")
        value_counts = df[category_col].value_counts()
        value_color_map = build_report_band_color_map(value_counts.index.tolist())
        # Draw dominant bands first and smaller bands last so the PDF image does
        # not collapse visually into one color when points overlap.
        draw_df = df.assign(
            __category_count=df[category_col].map(value_counts)
        ).sort_values("__category_count", ascending=False)
        radius = 3
        fill_opacity = 0.72
    else:
        # 1 Get unique categorical values
        unique_values = sorted(df[kpi_column].unique())

        # 2 Generate distinct colors dynamically
        colors = generate_distinct_colors(len(unique_values))

        value_color_map = {
            val: colors[i]
            for i, val in enumerate(unique_values)
        }
        draw_df = df
        category_col = kpi_column
        radius = 4
        fill_opacity = 0.9

    draw_df = _maybe_generalize_points(
        draw_df,
        threshold=REPORT_POINT_GENERALIZATION_THRESHOLD,
        key_col=category_col,
        max_points=REPORT_GENERALIZED_MAX_POINTS,
    )

    # 3 Plot points
    for _, row in draw_df.iterrows():
        value = row[category_col]

        if value not in value_color_map:
            continue

        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=radius,
            color=value_color_map[value],
            fill=True,
            fill_opacity=fill_opacity,
            opacity=0.85,
            tooltip=f"{kpi_column}: {value}"
        ).add_to(m)

    
    # 4 Build legend with top N categories + "Others"
    value_counts = df[category_col].value_counts()

    top_n = 6
    legend_items = []
    for val, count in value_counts.head(top_n).items():
        
        legend_items.append((str(val), value_color_map[val], count))

    others = value_counts.iloc[top_n:].sum()
    if others > 0:
        legend_items.append(("Others", "#999999", others))
    add_legend(m, kpi_column, legend_items)

    # Fit the viewport to the data only (fractional zoom keeps it tight).
    fit_data_bounds(m, fixed_bounds if fixed_bounds is not None else df, reserve_legend_space=False)

    m.save(output_html)


# =====================================================
# POOR REGION MAPS (RSRP / RSRQ) - PNG OUTPUT
# =====================================================

def _haversine_m(lat1, lon1, lat2, lon2):
    """Distance between two lat/lon points in meters."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _select_non_overlapping_regions(candidates, min_distance_meters, top_regions):
    selected = []
    for cand in candidates:
        too_close = False
        for sel in selected:
            d = _haversine_m(cand["lat"], cand["lon"], sel["lat"], sel["lon"])
            if d < min_distance_meters:
                too_close = True
                break
        if not too_close:
            selected.append(cand)
        if len(selected) == top_regions:
            break
    return selected


def detect_handover_events(df: pd.DataFrame, use_global_detection=True, min_run_length=3):
    """
    Detect report handover events.

    The PDF report handover image intentionally follows the frontend's band
    handover logic only. Technology and PCI transitions are available in the
    frontend UI, but they should not be mixed into this report map.
    """
    frontend_events = _detect_frontend_style_handover_events(df)
    return frontend_events


def detect_endc_setup_events(session_ids, db_conn_or_engine=None, route_df=None, region=None, country_code=None):
    """
    Detect ENDC Setup events from tbl_l3_log and tbl_event_log for given session_id(s).

    Logic:
    1. In tbl_l3_log, check if message contains 'RRC Connection Reconfiguration Complete'
       (or 'RRCConnectionReconfigurationComplete').
    2. Check if 'ENDC AVAILABILITY' is also present (in message, detail, or in tbl_event_log).
    3. Category transition must be from LTE-RRC to NSA_SA.
       If LTE-RRC to LTE-RRC, it is NOT considered an ENDC Setup.
    4. Return event objects with icon_type='hand' and type='endc_setup'.
    """
    if isinstance(session_ids, (int, str)):
        session_ids = [int(session_ids)]
    elif isinstance(session_ids, (list, tuple, set)):
        session_ids = [int(s) for s in session_ids if pd.notna(s)]
    else:
        return []

    if not session_ids:
        return []

    df_l3 = pd.DataFrame()
    df_evt = pd.DataFrame()

    # 1. Fetch L3 logs via Python Bridge if available
    try:
        import sys
        ml_prod_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ML-Production"))
        if os.path.isdir(ml_prod_dir) and ml_prod_dir not in sys.path:
            sys.path.insert(0, ml_prod_dir)
        ml_prod_dir2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if os.path.isdir(os.path.join(ml_prod_dir2, "utils")) and ml_prod_dir2 not in sys.path:
            sys.path.insert(0, ml_prod_dir2)
        from utils.python_bridge import get_bridge_client, _bridge_region_body
        bridge_client = get_bridge_client()
    except Exception:
        bridge_client = None

    if bridge_client is not None:
        try:
            body = {"SessionIds": session_ids, **_bridge_region_body(region, country_code)}
            df_l3 = bridge_client.post_rows("GetL3LogRows", body, limit=50000)
            if not df_l3.empty:
                for col in ["latitude", "longitude", "session_id", "row_no"]:
                    if col in df_l3.columns:
                        df_l3[col] = pd.to_numeric(df_l3[col], errors="coerce")
        except Exception as e:
            print(f"[detect_endc_setup_events] PythonBridge GetL3LogRows error: {e}")

    # 2. Query direct DB if df_l3 is empty or to fetch tbl_event_log
    from sqlalchemy import create_engine, text, bindparam

    conn = None
    close_when_done = False
    if db_conn_or_engine is not None:
        if hasattr(db_conn_or_engine, "connect") and not hasattr(db_conn_or_engine, "execute"):
            conn = db_conn_or_engine.connect()
            close_when_done = True
        else:
            conn = db_conn_or_engine
    else:
        norm_reg = str(region or country_code or "").strip().lower()
        if norm_reg in {"tw", "twn", "taiwan"}:
            db_url = os.getenv("DATABASE_URL_Taiwan") or os.getenv("DATABASE_URL_TAIWAN")
        else:
            db_url = os.getenv("DATABASE_URL")

        if db_url:
            try:
                engine = create_engine(db_url)
                conn = engine.connect()
                close_when_done = True
            except Exception as e:
                print(f"[detect_endc_setup_events] DB connection error: {e}")
                conn = None

    if conn is not None:
        try:
            if df_l3.empty:
                q_l3 = text("""
                    SELECT id, session_id, row_no, timestamp_text, latitude, longitude, category, message, detail
                    FROM tbl_l3_log
                    WHERE session_id IN :sids
                    ORDER BY session_id ASC, id ASC
                """).bindparams(bindparam("sids", expanding=True))
                df_l3 = pd.read_sql(q_l3, conn, params={"sids": session_ids})

            q_evt = text("""
                SELECT id, session_id, row_no, timestamp_text, latitude, longitude, category, event_name as message, detail
                FROM tbl_event_log
                WHERE session_id IN :sids
                ORDER BY session_id ASC, id ASC
            """).bindparams(bindparam("sids", expanding=True))
            df_evt = pd.read_sql(q_evt, conn, params={"sids": session_ids})
        except Exception as e:
            print(f"[detect_endc_setup_events] DB query error: {e}")
        finally:
            if close_when_done and conn is not None:
                conn.close()

    if df_l3.empty and df_evt.empty:
        return []

    all_events = []
    for sid in session_ids:
        s_l3 = df_l3[df_l3["session_id"] == sid].copy() if not df_l3.empty else pd.DataFrame()
        s_evt = df_evt[df_evt["session_id"] == sid].copy() if not df_evt.empty else pd.DataFrame()

        if s_l3.empty:
            continue

        s_l3["src"] = "L3"
        if not s_evt.empty:
            s_evt["src"] = "EVT"
            combined = pd.concat([s_l3, s_evt], ignore_index=True)
        else:
            combined = s_l3

        sort_cols = [c for c in ["timestamp_text", "row_no", "id"] if c in combined.columns]
        combined = combined.sort_values(by=sort_cols).reset_index(drop=True)

        session_events = []
        for i, row in combined.iterrows():
            msg = str(row.get("message") or "")
            cat = str(row.get("category") or "")

            if "RRC Connection Reconfiguration Complete" in msg or "RRCConnectionReconfigurationComplete" in msg:
                t_str = str(row.get("timestamp_text") or "")
                window_ahead = combined.iloc[i + 1 : min(len(combined), i + 20)]
                window_behind = combined.iloc[max(0, i - 10) : i]

                has_endc_avail = (
                    window_ahead["message"].str.contains("ENDC_AVAILABILITY|ENDC AVAILABILITY", case=False, na=False).any()
                    or window_behind["message"].str.contains("ENDC_AVAILABILITY|ENDC AVAILABILITY", case=False, na=False).any()
                    or "ENDC AVAILABILITY" in str(row.get("detail") or "").upper()
                )

                next_tech_cats = [
                    r["category"]
                    for _, r in window_ahead.iterrows()
                    if r.get("category") in ["LTE-RRC", "NSA_SA", "NR-RRC"]
                ]
                is_nsa_transition = ("NSA_SA" in next_tech_cats)

                if has_endc_avail and is_nsa_transition:
                    lat = row.get("latitude")
                    lon = row.get("longitude")

                    # Fallback to route_df if coords missing in L3
                    if (pd.isna(lat) or pd.isna(lon) or float(lat or 0) == 0) and route_df is not None:
                        s_route = route_df[route_df["session_id"] == sid] if "session_id" in route_df.columns else route_df
                        if not s_route.empty and "lat" in s_route.columns:
                            valid_coords = s_route.dropna(subset=["lat", "lon"])
                            if not valid_coords.empty:
                                lat = valid_coords.iloc[0]["lat"]
                                lon = valid_coords.iloc[0]["lon"]

                    if pd.isna(lat) or pd.isna(lon) or float(lat or 0) == 0:
                        continue

                    session_events.append({
                        "type": "endc_setup",
                        "icon_type": "hand",
                        "time": t_str,
                        "lat": float(lat),
                        "lon": float(lon),
                        "session_id": sid,
                        "from_value": cat,
                        "to_value": "NSA_SA",
                        "tooltip": f"ENDC Setup ({t_str}): {cat} -> NSA_SA",
                    })

        # Debounce events occurring in close succession
        last_t = None
        for ev in session_events:
            t = str(ev.get("time") or "")
            t_key = t[:8] if len(t) >= 8 else t
            last_key = last_t[:8] if (last_t and len(last_t) >= 8) else last_t
            if last_key is None or t_key != last_key:
                all_events.append(ev)
                last_t = t

    return all_events


def detect_endc_setup_from_network_logs(
    network_log_df=None,
    session_ids=None,
    db_conn_or_engine=None,
    region=None,
    country_code=None,
):
    """
    Detect ENDC Setup events from tbl_network_log.extra_json.

    The extra_json column stores a JSON blob per row, e.g.:
        '{"endc_setup": "ok",     "nr_mac_dl_mbps": 45.2, ...}'
        '{"endc_setup": "failed", ...}'

    Rows where extra_json["endc_setup"] is recognised:
      - 'ok'     → green hand marker  (#22c55e)
      - 'failed' → red   hand marker  (#ef4444)

    TWO-STAGE approach:
      1. Parse network_log_df in-memory first (fast — no extra DB call).
         report_df is polygon-filtered, so some rows with endc_setup may be
         absent. If no events are found here, fall through to stage 2.
      2. Direct unfiltered DB query — fetches ALL rows for the session IDs
         where extra_json IS NOT NULL, then extracts endc_setup from JSON.
         This catches events that were filtered out of report_df.

    Args:
        network_log_df    : Already-loaded DataFrame (has 'extra_json', 'lat',
                            'lon', 'session_id', 'timestamp').  Tried first.
        session_ids       : list[int] used for direct DB fallback when
                            network_log_df yields no events.
        db_conn_or_engine : SQLAlchemy connection/engine for the DB fallback.
        region / country_code : Multi-region DB selection.

    Returns:
        list of event dicts for generate_handover_map().
        Each event has 'endc_status' ('ok'|'failed') and 'color' (green|red).
    """
    import json as _json

    COLOR_OK     = "#22c55e"   # green hand  ✅
    COLOR_FAILED = "#ef4444"   # red   hand  ❌

    def _parse_endc_status(extra_json_val):
        """Return 'ok'|'failed' or None if not present / unparseable.

        Keys checked (in priority order):
          endc_setup  – explicit 'ok' / 'failed'
          ENDC_SETUP, endcSetup, ENDCSetup – camelCase / upper variants
          endc_event  – used in Taiwan data: 'ok', 'setup', 'fail', 'failed', etc.
        """
        if extra_json_val is None:
            return None
        try:
            blob = _json.loads(extra_json_val) if isinstance(extra_json_val, str) else extra_json_val
            if not isinstance(blob, dict):
                return None
            # ── Try dedicated endc_setup key first ────────────────────────
            raw = (
                blob.get("endc_setup")
                or blob.get("ENDC_SETUP")
                or blob.get("endcSetup")
                or blob.get("ENDCSetup")
            )
            # ── Fallback to endc_event key (Taiwan / other regions) ───────
            if raw is None:
                raw = blob.get("endc_event") or blob.get("ENDC_EVENT") or blob.get("endcEvent")
            if raw is None:
                return None
            norm = str(raw).strip().lower()
            # Normalise to 'ok' | 'failed'
            # NOTE: 'setup' is intentionally NOT mapped — it means an ENDC
            # attempt is in progress and should NOT be plotted on the map.
            if norm in {"ok", "success"}:
                return "ok"
            if norm in {"failed", "fail", "failure", "error", "nok"}:
                return "failed"
            return None
        except Exception:
            return None


    def _df_to_events(df):
        """Parse extra_json from a DataFrame and return event list (may be [])."""
        if df is None or df.empty or "extra_json" not in df.columns:
            return []

        work = df.copy()

        # Normalise lat/lon aliases
        for alias, candidates in [
            ("lat", ["lat", "latitude", "gps_lat"]),
            ("lon", ["lon", "longitude", "gps_lon"]),
        ]:
            if alias not in work.columns:
                for cand in candidates:
                    if cand in work.columns:
                        work[alias] = work[cand]
                        break
        for col in ["lat", "lon"]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        work["_endc_status"] = work["extra_json"].apply(_parse_endc_status)
        df_endc = work[work["_endc_status"].notna()].copy()

        if df_endc.empty:
            return []

        # Deduplicate companion carrier rows (e.g. LTE Anchor + 5G NR companion rows)
        # that share the same timestamp/session_id so an ENDC event is counted
        # exactly once per event timestamp, not duplicated across RF carrier rows.
        if "timestamp" in df_endc.columns:
            ts_norm = df_endc["timestamp"].astype(str).str.strip().str[:19]
            df_endc = df_endc.assign(_ts_dedup=ts_norm)
            dedup_cols = ["_ts_dedup"]
            if "session_id" in df_endc.columns:
                dedup_cols.insert(0, "session_id")
            df_endc = df_endc.drop_duplicates(subset=dedup_cols)

        events = []
        for _, row in df_endc.iterrows():
            lat = row.get("lat")
            lon = row.get("lon")
            if pd.isna(lat) or pd.isna(lon) or float(lat or 0) == 0:
                continue
            status = row["_endc_status"]
            color  = COLOR_OK if status == "ok" else COLOR_FAILED
            t_str  = str(row.get("timestamp") or "")
            sid    = row.get("session_id", "")
            events.append({
                "type":        "endc_setup",
                "icon_type":   "hand",
                "endc_status": status,
                "color":       color,
                "time":        t_str,
                "lat":         float(lat),
                "lon":         float(lon),
                "session_id":  sid,
                "from_value":  "LTE",
                "to_value":    "NSA_SA",
                "tooltip":     f"ENDC Setup ({t_str}): {status.upper()}",
            })
        return events

    # ── STAGE 1: in-memory parse from network_log_df ──────────────────────
    if network_log_df is not None and not network_log_df.empty:
        events = _df_to_events(network_log_df)
        if events:
            ok_n     = sum(1 for e in events if e["endc_status"] == "ok")
            failed_n = sum(1 for e in events if e["endc_status"] == "failed")
            print(
                f"[detect_endc_setup_from_network_logs] stage=in_memory "
                f"total={len(events)} ok={ok_n} failed={failed_n}"
            )
            return events
        print(
            "[detect_endc_setup_from_network_logs] stage=in_memory "
            "no endc_setup found in report_df extra_json — trying direct DB query"
        )

    # ── STAGE 2: direct DB query (unfiltered — all rows for session IDs) ──
    # resolve session_ids
    sids = []
    if session_ids is not None:
        if isinstance(session_ids, (int, str)):
            sids = [int(session_ids)]
        elif isinstance(session_ids, (list, tuple, set)):
            sids = [int(s) for s in session_ids if pd.notna(s)]
    elif network_log_df is not None and "session_id" in network_log_df.columns:
        # derive from the df that was passed
        sids = [int(s) for s in network_log_df["session_id"].dropna().unique()]

    if not sids:
        print("[detect_endc_setup_from_network_logs] stage=db no session_ids — giving up")
        return []

    from sqlalchemy import create_engine, text, bindparam

    conn = None
    close_when_done = False
    if db_conn_or_engine is not None:
        if hasattr(db_conn_or_engine, "connect") and not hasattr(db_conn_or_engine, "execute"):
            conn = db_conn_or_engine.connect()
            close_when_done = True
        else:
            conn = db_conn_or_engine
    else:
        norm_reg = str(region or country_code or "").strip().lower()
        db_url = (
            (os.getenv("DATABASE_URL_Taiwan") or os.getenv("DATABASE_URL_TAIWAN"))
            if norm_reg in {"tw", "twn", "taiwan"}
            else os.getenv("DATABASE_URL")
        )
        if db_url:
            try:
                engine = create_engine(db_url)
                conn = engine.connect()
                close_when_done = True
            except Exception as e:
                print(f"[detect_endc_setup_from_network_logs] DB connection error: {e}")

    if conn is None:
        print("[detect_endc_setup_from_network_logs] stage=db no DB connection — giving up")
        return []

    try:
        q = text("""
            SELECT session_id, timestamp, lat, lon, extra_json
            FROM tbl_network_log
            WHERE session_id IN :sids
              AND extra_json IS NOT NULL
              AND extra_json != ''
            ORDER BY session_id ASC, id ASC
        """).bindparams(bindparam("sids", expanding=True))
        df_db = pd.read_sql(q, conn, params={"sids": sids})
    except Exception as e:
        print(f"[detect_endc_setup_from_network_logs] DB query error: {e}")
        return []
    finally:
        if close_when_done and conn is not None:
            conn.close()

    events = _df_to_events(df_db)
    ok_n     = sum(1 for e in events if e["endc_status"] == "ok")
    failed_n = sum(1 for e in events if e["endc_status"] == "failed")
    print(
        f"[detect_endc_setup_from_network_logs] stage=db "
        f"rows_with_extra_json={len(df_db)} total_events={len(events)} "
        f"ok={ok_n} failed={failed_n}"
    )
    return events


def _first_present(row, candidates):
    for col in candidates:
        if col in row.index:
            value = row.get(col)
            if isinstance(value, (dict, list, tuple, set)):
                if value:
                    return value
                continue
            if pd.notna(value) and str(value).strip() != "":
                return value
    return None


def _safe_scalar(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), sort_keys=True, default=str)
    return value


def _safe_sort_value(value):
    value = _safe_scalar(value)
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _clean_transition_value(value):
    value = _safe_scalar(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"unknown", "n/a", "na", "null", "undefined", "-"}:
        return None
    return text


def _safe_session_id(value):
    value = _safe_scalar(value)
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return str(value)


def _detect_frontend_style_handover_events(df: pd.DataFrame):
    """
    Detect band handover transitions using the same practical value
    normalization/order rules used by the frontend handover experience.
    """
    if df is None or df.empty:
        return []

    data = df.rename(columns={c: c.lower() for c in df.columns}).copy()
    if not {"lat", "lon"}.issubset(data.columns):
        return []

    data = data.dropna(subset=["lat", "lon"]).reset_index(drop=False)
    if data.empty:
        return []

    session_col = "session_id" if "session_id" in data.columns else None
    if session_col:
        data["__session_sort"] = data[session_col].apply(_safe_sort_value)
        data["__session_group"] = data[session_col].apply(_safe_sort_value)
    if "timestamp" in data.columns:
        data["__timestamp_sort"] = data["timestamp"].apply(_safe_sort_value)
    if "id" in data.columns:
        data["__id_sort"] = data["id"].apply(_safe_sort_value)

    sort_cols = []
    if session_col:
        sort_cols.append("__session_sort")
    if "timestamp" in data.columns:
        sort_cols.append("__timestamp_sort")
    elif "id" in data.columns:
        sort_cols.append("__id_sort")
    sort_cols.append("index")
    data = data.sort_values(sort_cols)

    groups = data.groupby("__session_group", dropna=False) if session_col else [(None, data)]
    events = []

    value_resolver = lambda r: normalize_band_name(
        _first_present(r, ("band", "neighbourband", "neighborband", "neighbour_band"))
    )

    for sid, group in groups:
        previous = None
        previous_row = None

        for _, row in group.iterrows():
            current = _clean_transition_value(value_resolver(row))
            if current is None:
                continue

            if previous is not None and previous != current and previous_row is not None:
                events.append({
                    "type": "band",
                    "session_id": _safe_session_id(sid),
                    "timestamp": row.get("timestamp"),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "from_value": previous,
                    "to_value": current,
                    "from_lat": float(previous_row["lat"]),
                    "from_lon": float(previous_row["lon"]),
                    "from_provider": _clean_transition_value(
                        _first_present(previous_row, ("m_alpha_long", "provider", "operator"))
                    ),
                    "to_provider": _clean_transition_value(
                        _first_present(row, ("m_alpha_long", "provider", "operator"))
                    ),
                    "from_network": None,
                    "to_network": None,
                })

            previous = current
            previous_row = row

    unique = []
    seen = set()
    for ev in events:
        key = (
            ev["type"],
            ev.get("session_id"),
            round(float(ev["lat"]), 6),
            round(float(ev["lon"]), 6),
            ev["from_value"],
            ev["to_value"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(ev)

    return unique


def _build_region_candidates(poor_df, grid_size):
    tmp = poor_df.copy()
    tmp["lat_bin"] = (tmp["lat"] / grid_size).round().astype(int)
    tmp["lon_bin"] = (tmp["lon"] / grid_size).round().astype(int)

    grid_counts = (
        tmp.groupby(["lat_bin", "lon_bin"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    candidates = []
    for _, r in grid_counts.iterrows():
        cell = tmp[(tmp.lat_bin == r.lat_bin) & (tmp.lon_bin == r.lon_bin)]
        candidates.append({
            "lat": cell.lat.mean(),
            "lon": cell.lon.mean(),
            "count": int(r["count"]),
            "points": cell
        })

    return candidates


def generate_poor_region_map(
    filtered_df,
    value_col,
    threshold,
    output_png,
    tmp_html,
    title,
    polygon_wkt=None,
    grid_size=0.0012,
    top_regions=5,
    min_distance_meters=400,
    point_radius=2,
    region_opacity=0.25,
    render_width=1370,
    render_height=900,
    device_scale_factor=1,
    fixed_bounds=None,
):
    """
    Generate poor region map PNG using ONLY filtered_df.
    No DB calls. No additional polygon filtering.
    """
    if value_col not in filtered_df.columns:
        print(f" Missing column: {value_col}")
        return

    df = _prepare_geo_numeric_df(filtered_df[["lat", "lon", value_col]].copy(), value_col)

    poor = df[df[value_col] < threshold]
    print(f"{title} | Filtered Samples: {len(df)} | Poor Samples: {len(poor)}")

    if poor.empty:
        print(f" No poor samples for {value_col}")
        return

    # Draw EVERY individual poor sample (not 5 clustered centroids).  The old
    # clustering collapsed 17 000 poor points into ~5 blobs; this shows the true
    # spatial spread.  Subsample only for render speed — the legend keeps the
    # true count and the dots overlap densely at this zoom anyway.
    true_count = len(poor)
    draw_df = _maybe_generalize_points(
        poor,
        threshold=REPORT_POOR_POINT_GENERALIZATION_THRESHOLD,
        max_points=REPORT_POOR_GENERALIZED_MAX_POINTS,
    )

    fmap = new_report_map()
    add_fullscreen_css(fmap)

    for _, p in draw_df.iterrows():
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=4,
            color="#e31a1c",
            fill=True,
            fill_opacity=0.85,
        ).add_to(fmap)

    add_legend(fmap, title, [(f"Poor Samples : {true_count}", "#e31a1c", true_count)])
    draw_polygon_overlay(fmap, polygon_wkt)

    # Viewport fits the full route (df) for geographic context.
    fit_data_bounds(fmap, fixed_bounds if fixed_bounds is not None else df, reserve_legend_space=False)

    fmap.save(tmp_html)

    # Convert saved HTML to PNG using Playwright utility (consistent with other maps)
    try:
        try:
            if __package__:
                from .playwright_utils import html_to_png
            else:
                from playwright_utils import html_to_png
        except ImportError:
            try:
                from playwright_utils import html_to_png
            except ImportError:
                from tools.report_engine.playwright_utils import html_to_png
        html_to_png(
            tmp_html, output_png,
            width=render_width, height=render_height,
            device_scale_factor=device_scale_factor,
        )
    except Exception as e:
        print(f" Warning: failed to convert poor region html to png: {e}")
    finally:
        try:
            if os.path.exists(tmp_html):
                os.remove(tmp_html)
        except Exception:
            pass


# =====================================================
# BASE ROUTE MAP (DRIVE ROUTE + POLYGON) - PNG OUTPUT
# =====================================================

def generate_base_route_map(df, polygon_wkt, output_html, fixed_bounds=None):
    """
    Generate a basic map showing the drive route and polygon boundary.
    No KPI overlays, just the route and boundary.
    """
    df = _prepare_geo_numeric_df(df)

    if df.empty:
        raise ValueError("No GPS data to plot for base route map")

    draw_df = _maybe_generalize_points(
        df,
        threshold=REPORT_POINT_GENERALIZATION_THRESHOLD,
        max_points=REPORT_GENERALIZED_MAX_POINTS,
    )

    m = new_report_map()

    add_fullscreen_css(m)

    # 1 Dense filled points for solid appearance (same style as KPI maps).
    # No polyline so nothing appears outside the polygon.
    for _, r in draw_df.iterrows():
        folium.CircleMarker(
            location=(r["lat"], r["lon"]),
            radius=4,
            color="#2b8cbe",
            fill=True,
            fill_opacity=0.95,
        ).add_to(m)

    # 2 Polygon boundary (solid red) — overlay only, does NOT expand the view.
    draw_polygon_overlay(m, polygon_wkt, color="red", weight=4, opacity=1.0, dash_array="")

    # Fit the viewport to the route only (fractional zoom keeps it tight).
    fit_data_bounds(m, fixed_bounds if fixed_bounds is not None else df, reserve_legend_space=False)

    m.save(output_html)


def generate_poor_region_maps(filtered_df, output_dir="data/images/maps", tmp_dir="data/tmp",
                              polygon_wkt=None, render_width=1370, render_height=900,
                              device_scale_factor=1, fixed_bounds=None):
    """Generate RSRP/RSRQ poor region maps using only filtered_df."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    generate_poor_region_map(
        filtered_df=filtered_df,
        value_col="rsrp",
        threshold=-105,
        output_png=os.path.join(output_dir, "rsrp_poor_regions.png"),
        tmp_html=os.path.join(tmp_dir, "rsrp_poor_regions.html"),
        title="RSRP < -105",
        polygon_wkt=polygon_wkt,
        render_width=render_width, render_height=render_height,
        device_scale_factor=device_scale_factor,
        fixed_bounds=fixed_bounds,
    )
    # Also generate RSRQ poor region map
    generate_poor_region_map(
        filtered_df=filtered_df,
        value_col="rsrq",
        threshold=-14,
        output_png=os.path.join(output_dir, "rsrq_poor_regions.png"),
        tmp_html=os.path.join(tmp_dir, "rsrq_poor_regions.html"),
        title="RSRQ < -14",
        polygon_wkt=polygon_wkt,
        render_width=render_width, render_height=render_height,
        device_scale_factor=device_scale_factor,
        fixed_bounds=fixed_bounds,
    )


def generate_handover_map(filtered_df, events, output_html, polygon_wkt=None, fixed_bounds=None):
    """
    Generate HTML for handover visualization. `events` is a list of handover dicts
    as returned by `detect_handover_events`.
    """
    df = filtered_df.dropna(subset=["lat", "lon"]).copy() if filtered_df is not None else pd.DataFrame()
    if not df.empty:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])
        df = df[
            df["lat"].between(-90, 90)
            & df["lon"].between(-180, 180)
        ].copy()
    if df.empty:
        raise ValueError("No GPS data to plot for handover map")
    events = [
        ev for ev in (events or [])
        if (
            str(ev.get("type") or "").lower() in ["band", "endc_setup", "handover"]
            or ev.get("icon_type") == "hand"
        )
        and pd.notna(ev.get("lat"))
        and pd.notna(ev.get("lon"))
    ]
    clean_events = []
    for ev in events:
        try:
            lat = float(ev.get("lat"))
            lon = float(ev.get("lon"))
        except (TypeError, ValueError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        clean_events.append({**ev, "lat": lat, "lon": lon})
    events = clean_events

    m = new_report_map()
    add_fullscreen_css(m)

    # Polygon drawn as a dashed overlay — does NOT expand the viewport.
    draw_polygon_overlay(m, polygon_wkt)

    # Draw dense route points only (no polylines).
    # This prevents line-like rendering and keeps handover focus on events.
    route_layers = []
    if "session_id" in df.columns:
        df["__session_sort"] = df["session_id"].apply(_safe_sort_value)
        sessions = sorted(df["__session_sort"].dropna().unique())
        if "timestamp" in df.columns:
            df["__timestamp_sort"] = df["timestamp"].apply(_safe_sort_value)
            df_route = df.sort_values(["__session_sort", "__timestamp_sort"]).copy()
        else:
            df_route = df.sort_values(["__session_sort"]).copy()

        route_palette = ["#2b8cbe", "#0284c7", "#6366f1", "#059669", "#d97706", "#7c3aed", "#0891b2"]
        colors = [route_palette[i % len(route_palette)] for i in range(len(sessions))] if sessions else ["#2b8cbe"]

        # Draw each session
        for i, sid in enumerate(sessions):
            seg = df_route[df_route["__session_sort"] == sid]
            if seg.empty:
                continue
            route_layers.append({
                "color": colors[i % len(colors)],
                "points": [
                    [float(r["lat"]), float(r["lon"])]
                    for _, r in seg.iterrows()
                ],
            })

        # Legend removed per user request - handover map should show only routes and handover events
    else:
        # Fallback: draw a single route backbone (pre-existing behavior)
        if "timestamp" in df.columns:
            df["__timestamp_sort"] = df["timestamp"].apply(_safe_sort_value)
            df_route = df.sort_values(["__timestamp_sort"])
        else:
            df_route = df

        route_layers.append({
            "color": "#2b8cbe",
            "points": [
                [float(r["lat"]), float(r["lon"])]
                for _, r in df_route.iterrows()
            ],
        })

    MAX_HANDOVER_ROUTE_DOTS = 8000
    route_point_count = sum(len(layer["points"]) for layer in route_layers)
    route_step = max(1, route_point_count // MAX_HANDOVER_ROUTE_DOTS)
    for layer in route_layers:
        if route_step > 1:
            layer["points"] = layer["points"][::route_step]

    # Add handover sparks / ENDC setup markers
    # Default color map for non-network-log events that don't carry a 'color' key
    event_default_colors = {
        "band": "#ef4444",
        "endc_setup": "#ef4444",
        "handover": "#ef4444",
    }
    event_points = []
    for ev in events:
        event_type = str(ev.get("type") or "endc_setup").lower()
        # Use per-event color if available (e.g. green/red from network logs)
        # otherwise fall back to the type-based default (always red)
        point_color = ev.get("color") or event_default_colors.get(event_type, "#ef4444")
        if ev.get("from_value") is not None and ev.get("to_value") is not None:
            tooltip = ev.get("tooltip") or (
                f"{event_type.title()}: {ev.get('from_value')} -> {ev.get('to_value')} "
                f"(Session {ev.get('session_id')})"
            )
        else:
            tooltip = ev.get("tooltip") or (
                f"{ev.get('from_provider')} -> {ev.get('to_provider')} (Session {ev.get('session_id')})"
            )
        event_points.append({
            "lat": ev["lat"],
            "lon": ev["lon"],
            "color": point_color,
            "tooltip": tooltip,
            "type": event_type,
            "icon_type": ev.get("icon_type", "hand"),
            "endc_status": ev.get("endc_status", ""),
        })

    payload = json.dumps({"routes": route_layers, "events": event_points})
    map_name = m.get_name()
    render_js = f"""
    <script>
        (function() {{
            var payload = {payload};
            function drawHandoverLayers() {{
                var map = window["{map_name}"];
                if (!map || !window.L) {{
                    window.setTimeout(drawHandoverLayers, 50);
                    return;
                }}
                var canvasRenderer = L.canvas();
                var routeOptions = {{
                    radius: 4,
                    stroke: true,
                    weight: 2,
                    opacity: 0.9,
                    fill: true,
                    fillOpacity: 0.92,
                    renderer: canvasRenderer
                }};
                payload.routes.forEach(function(layer) {{
                    layer.points.forEach(function(point) {{
                        L.circleMarker(point, Object.assign({{}}, routeOptions, {{
                            color: layer.color,
                            fillColor: layer.color
                        }})).addTo(map);
                    }});
                }});
                payload.events.forEach(function(ev) {{
                    if (ev.icon_type !== "circle_plain") {{
                        // Use the per-event color for the hand background
                        var bgColor = ev.color || "#dc2626";
                        var handIcon = L.divIcon({{
                            className: "endc-hand-marker",
                            html: '<div style="background:' + bgColor + ';border:2.5px solid #ffffff;border-radius:50%;width:30px;height:30px;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(0,0,0,0.6);cursor:pointer;"><svg width="18" height="18" viewBox="0 0 24 24" fill="#ffffff" style="display:block;"><path d="M18.5 8c-.83 0-1.5.67-1.5 1.5v4.25l-.47-.21a2.82 2.82 0 0 0-2.31-.08l-.22.09V3.5c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v8.5H10V1.5C10 .67 9.33 0 8.5 0S7 .67 7 1.5v10.5h-.5V4.5C6.5 3.67 5.83 3 5 3s-1.5.67-1.5 1.5v10.74c0 3.73 3.03 6.76 6.76 6.76h3.48c3.73 0 6.76-3.03 6.76-6.76V9.5c0-.83-.67-1.5-1.5-1.5z"/></svg></div>',
                            iconSize: [30, 30],
                            iconAnchor: [15, 15]
                        }});
                        var marker = L.marker([ev.lat, ev.lon], {{icon: handIcon, zIndexOffset: 1000}}).addTo(map);
                    }} else {{
                        var marker = L.circleMarker([ev.lat, ev.lon], {{
                            radius: 6,
                            color: "#1f2937",
                            weight: 1,
                            opacity: 0.95,
                            fill: true,
                            fillColor: ev.color,
                            fillOpacity: 0.95,
                            renderer: canvasRenderer
                        }}).addTo(map);
                    }}
                    if (ev.tooltip) {{
                        marker.bindTooltip(ev.tooltip);
                    }}
                }});
            }}
            drawHandoverLayers();
        }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(render_js))

    if events:
        # Build legend — separate OK (green) and Failed (red) entries when
        # the events carry endc_status, otherwise show a single combined entry.
        ok_count     = sum(1 for e in events if str(e.get("endc_status") or "").lower() == "ok")
        failed_count = sum(1 for e in events if str(e.get("endc_status") or "").lower() == "failed")
        other_count  = len(events) - ok_count - failed_count

        legend_items = []
        if ok_count or failed_count:
            legend_items.append(("ENDC Setup OK (✋)", "#22c55e", ok_count))
            legend_items.append(("ENDC Setup Failed (✋)", "#ef4444", failed_count))
        if other_count:
            legend_items.append(("Handover (✋)", "#ef4444", other_count))

        add_legend(
            m,
            "Handover / ENDC Events",
            legend_items,
        )

    # Fit the viewport to the data only (fractional zoom keeps it tight).
    bounds_target = fixed_bounds
    if bounds_target is None:
        if events:
            ev_df = pd.DataFrame([{"lat": e["lat"], "lon": e["lon"]} for e in events])
            bounds_df = pd.concat([df[["lat", "lon"]], ev_df], ignore_index=True)
            bounds_target = bounds_df
        else:
            bounds_target = df
    fit_data_bounds(m, bounds_target, reserve_legend_space=False)

    m.save(output_html)
