"""
Dashboard for run_open_map_clutter.py's output - visualizes the
Research/open_map.py image-based (Canny edge density + K-Means) clutter
classification for project 246, tile-for-tile, next to what's already
stored in production for the same tiles. Read-only: does not call any
production classification code, only reads the CSV this folder's
run_open_map_clutter.py already produced plus its saved images.

Run:
    venv\\Scripts\\python.exe -m streamlit run tests\\new-project\\open_map_clutter\\streamlit_open_map_clutter_dashboard.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely import wkt as shapely_wkt
from shapely.ops import transform

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ID = int(os.environ.get("CLUTTER_PROJECT_ID", 210))
CSV_PATH = THIS_DIR / f"clutter_comparison_{PROJECT_ID}.csv"
OVERLAY_JPG = THIS_DIR / f"Density_Zones_Map_project{PROJECT_ID}.jpg"
RAW_JPG = THIS_DIR / f"satellite_raw_project{PROJECT_ID}.jpg"
WATER_GEOJSON = THIS_DIR / f"water_vector_project{PROJECT_ID}.geojson"
VEGETATION_GEOJSON = THIS_DIR / f"vegetation_vector_project{PROJECT_ID}.geojson"
BUILDINGS_GEOJSON = THIS_DIR / f"buildings_vector_project{PROJECT_ID}.geojson"
ROADS_GEOJSON = THIS_DIR / f"roads_vector_project{PROJECT_ID}.geojson"
HIGHWAY_GEOJSON = THIS_DIR / f"highway_vector_project{PROJECT_ID}.geojson"
RAILWAY_GEOJSON = THIS_DIR / f"railway_vector_project{PROJECT_ID}.geojson"
WATER_CLIPPED_GEOJSON = THIS_DIR / f"water_clipped_project{PROJECT_ID}.geojson"
VEGETATION_CLIPPED_GEOJSON = THIS_DIR / f"vegetation_clipped_project{PROJECT_ID}.geojson"
BUILDINGS_CLIPPED_GEOJSON = THIS_DIR / f"buildings_clipped_project{PROJECT_ID}.geojson"
PROJECT_POLYGON_GEOJSON = THIS_DIR.parents[1] / "baseline" / "data" / f"project_{PROJECT_ID}_taiwan" / "project_polygon.geojson"

# Same palette as tests/baseline/clutter_map_streamlit.py's CLASS_COLORS,
# for visual consistency between the two dashboards.
FINAL_CLASS_COLORS = {
    "Dense Urban": "#b3261e",
    "Urban": "#e08a2b",
    "Suburban": "#d8b656",
    "Water": "#2a6fbd",
    "Vegetation": "#3f8f5c",
    "Open": "#c9c2b3",
}


def _build_raster(df: pd.DataFrame, color_field: str, class_colors: dict):
    """Rasterize the grid_id's own R{row}C{col} index into a 2D array and
    render with go.Heatmap instead of one filled polygon shape per tile.
    ~22k separate Scattergl fill='toself' shapes is what made the page
    unresponsive - fill rendering for that many disjoint polygons isn't
    what the browser's renderer is built for. A Heatmap is one canvas
    draw call regardless of cell count, so it stays instant at this size."""
    rc = df["grid_id"].str.extract(r"R(\d+)C(\d+)").astype(int)
    row, col = rc[0].to_numpy(), rc[1].to_numpy()
    row_min, row_max = row.min(), row.max()
    col_min, col_max = col.min(), col.max()
    n_rows, n_cols = row_max - row_min + 1, col_max - col_min + 1

    classes = list(class_colors.keys())
    code_of = {c: i for i, c in enumerate(classes)}
    codes = df[color_field].map(code_of).to_numpy(dtype=float)  # NaN for any label outside the palette

    z = np.full((n_rows, n_cols), np.nan)
    z[row - row_min, col - col_min] = codes
    text = np.full((n_rows, n_cols), "", dtype=object)
    text[row - row_min, col - col_min] = df[color_field].astype(str).to_numpy()

    # Real geographic position per row/col index, from this grid's own tiles.
    y_by_row = pd.Series(df["lat"].to_numpy(), index=row).groupby(level=0).mean().reindex(range(row_min, row_max + 1))
    x_by_col = pd.Series(df["lon"].to_numpy(), index=col).groupby(level=0).mean().reindex(range(col_min, col_max + 1))

    n = len(classes)
    colorscale = []
    for i, cls in enumerate(classes):
        colorscale.append([i / n, class_colors[cls]])
        colorscale.append([(i + 1) / n, class_colors[cls]])

    return z, x_by_col.to_numpy(), y_by_row.to_numpy(), text, colorscale, n


def _tile_polygon_trace(df: pd.DataFrame, color_field: str, class_name: str, color: str, clip_polygon, opacity: float):
    subset = df[df[color_field] == class_name]
    if subset.empty:
        return None
    xs, ys = [], []
    for tile_wkt in subset["tile_wkt"]:
        geom = shapely_wkt.loads(tile_wkt)
        if clip_polygon is not None:
            geom = geom.intersection(clip_polygon)
        if geom is None or geom.is_empty:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms) if geom.geom_type == "MultiPolygon" else []
        for poly in polys:
            coords = list(poly.exterior.coords)
            xs.extend([c[0] for c in coords] + [None])
            ys.extend([c[1] for c in coords] + [None])
    if not xs:
        return None
    return go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        fill="toself",
        fillcolor=color,
        line=dict(color=color, width=0),
        opacity=opacity,
        name=class_name,
        hoverinfo="skip",
        showlegend=False,
    )


def _preferred_layer(clipped_path: Path, raw_path: Path) -> Path:
    return clipped_path if clipped_path.exists() else raw_path


@st.cache_data
def _project_polygon_wkt(path: str) -> str | None:
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    gdf = gpd.read_file(path_obj)
    if gdf.empty:
        return None
    # This cached test polygon is stored as lat/lon coordinates even though
    # the CRS says EPSG:4326. Match the baseline dashboard's correction.
    return transform(lambda x, y: (y, x), gdf.geometry.iloc[0]).wkt


def _load_project_polygon():
    polygon_wkt = _project_polygon_wkt(str(PROJECT_POLYGON_GEOJSON))
    return shapely_wkt.loads(polygon_wkt) if polygon_wkt else None


@st.cache_data
def _polygon_coords(geojson_path: str, simplify_tolerance: float):
    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        return [], []
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        return [], []
    xs, ys = [], []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms) if geom.geom_type == "MultiPolygon" else []
        for p in polys:
            coords = list(p.exterior.coords)
            xs.extend([c[0] for c in coords] + [None])
            ys.extend([c[1] for c in coords] + [None])
    return xs, ys


def _vector_trace(geojson_path: Path, color: str, opacity=0.9, simplify_tolerance=0.0, line_color=None, line_width=0.0):
    """Real, whole Overture polygon (water or vegetation), drawn at full
    vector precision - same technique tests/baseline/clutter_map_streamlit.py
    uses for its own water/building layers. These are single large,
    already-dissolved shapes (one per file), not thousands of tiles, so
    the Scattergl fill='toself' performance problem from the old tile-by-
    tile rendering doesn't apply here.

    line_color/line_width default to no visible border (matches the
    original water/vegetation look, where a border doesn't matter - they're
    each one big dissolved shape). For buildings specifically, pass a real
    border (line_color="#111111", line_width=0.3, baseline's own values) -
    with ~5,000 individual small polygons and no defined edge, a 0-width
    border reads as scattered dark speckle under WebGL anti-aliasing rather
    than distinct building shapes; a thin, defined edge is what makes
    baseline's own buildings layer look smooth instead of "black spots"."""
    xs, ys = _polygon_coords(str(geojson_path), simplify_tolerance)
    if not xs:
        return None
    return go.Scattergl(
        x=xs, y=ys, mode="lines", fill="toself", fillcolor=color,
        line=dict(color=line_color or color, width=line_width), opacity=opacity, hoverinfo="skip", showlegend=False,
    )


@st.cache_data
def _line_coords(geojson_path: str, clip_polygon_wkt: str | None):
    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        return [], []
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        return [], []
    clip_polygon = shapely_wkt.loads(clip_polygon_wkt) if clip_polygon_wkt else None
    xs, ys = [], []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if clip_polygon is not None:
            geom = geom.intersection(clip_polygon)
        if geom is None or geom.is_empty:
            continue
        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms) if geom.geom_type == "MultiLineString" else []
        for ls in lines:
            coords = list(ls.coords)
            xs.extend([c[0] for c in coords] + [None])
            ys.extend([c[1] for c in coords] + [None])
    return xs, ys


def _line_trace(geojson_path: Path, color: str, width=1.5, dash=None, clip_polygon=None):
    """Roads/highway/railway - real LineString geometry, same technique as
    tests/baseline/clutter_map_streamlit.py's lines_to_trace."""
    xs, ys = _line_coords(str(geojson_path), clip_polygon.wkt if clip_polygon is not None else None)
    if not xs:
        return None
    return go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), hoverinfo="skip", showlegend=False)


st.set_page_config(page_title=f"Project {PROJECT_ID} - open_map.py Clutter Classification", layout="wide")
st.title(f"Project {PROJECT_ID} - Image-based Clutter Classification (Research/open_map.py)")
st.caption(
    "Every classification here comes from Research/open_map.py's Canny-edge + K-Means density "
    "method, run as-is against project 246's real coverage area, sampled onto production's own "
    "grid tiles. Production's stored value is shown alongside for reference only - nothing here "
    "was computed by production code."
)

if not CSV_PATH.exists():
    st.error(
        f"{CSV_PATH.name} not found. Run run_open_map_clutter.py first "
        f"(set CLUTTER_PROJECT_ID={PROJECT_ID} if not already the default):\n\n"
        r"venv\Scripts\python.exe tests\new-project\open_map_clutter\run_open_map_clutter.py"
    )
    st.stop()

df = pd.read_csv(CSV_PATH)
project_polygon = _load_project_polygon()
if project_polygon is not None:
    tile_geoms = df["tile_wkt"].apply(shapely_wkt.loads)
    df = df[tile_geoms.apply(lambda geom: geom.intersects(project_polygon))].copy()

col_a, col_b = st.columns([1, 3])

with col_a:
    map_mode = st.radio("Color tiles by", ["final_class (6-class result)", "production_clutter_class (reference)"])
    color_field = "final_class" if map_mode.startswith("final_class") else "production_clutter_class"

    st.markdown("**Layers**")
    show_tiles = st.checkbox("Clutter tiles (colored)", value=True)
    show_buildings = st.checkbox("Buildings (real Overture polygons)", value=False)
    show_roads = st.checkbox("Roads", value=True)
    show_highway = st.checkbox("Highway (real: class=trunk/primary/secondary)", value=True)
    show_railway = st.checkbox("Railway (real: subtype=rail)", value=True)
    show_water = st.checkbox("Water (real Overture polygons, clipped to grid)", value=True)
    show_vegetation = st.checkbox("Vegetation (real Overture polygons, clipped after water removal)", value=True)
    tile_opacity = st.slider("Tile fill opacity", 0.10, 1.00, 0.55)

    st.markdown("**Class distribution (tile count)**")
    counts = df[color_field].value_counts()
    for cls in FINAL_CLASS_COLORS:
        if cls in counts.index:
            pct = 100 * counts[cls] / len(df)
            st.markdown(
                f"<span style='display:inline-block;width:12px;height:12px;background:{FINAL_CLASS_COLORS[cls]};"
                f"border-radius:2px;margin-right:6px;'></span>{cls}: **{int(counts[cls])}** tiles ({pct:.1f}%)",
                unsafe_allow_html=True,
            )
    leftover = counts[~counts.index.isin(FINAL_CLASS_COLORS)]
    for cls, n in leftover.items():
        st.markdown(f"{cls}: **{int(n)}** tiles ({100*n/len(df):.1f}%)")

# Every tile gets a full (100%) fill in its own class color - including
# Water/Vegetation - exactly like tests/baseline/clutter_map_streamlit.py
# does for ITS tiles. Excluding Water/Vegetation from the fill (an earlier
# version of this dashboard) and relying solely on the real vector polygon
# to supply their color created real black gaps: a tile only needs to WIN
# the water/green/built comparison to be classified Water, not be 100%
# water - checked directly against real data, 822/1,916 Water tiles here
# have water_ratio < 0.9, so up to 60%+ of some of those tiles is not
# actually water at all. The raster excluded the whole tile while the real
# water polygon only covered its true (partial) share, leaving the
# remainder with nothing drawn - a black hole, confirmed by isolating the
# tile layer alone with buildings/water/vegetation all switched off and
# still seeing the same gaps. The real vector water/vegetation polygons
# (below) are still drawn ON TOP for precision - same as baseline - they
# just aren't the ONLY source of color for those tiles any more.
fig = go.Figure()
if show_tiles:
    if project_polygon is not None:
        for cls, color in FINAL_CLASS_COLORS.items():
            trace = _tile_polygon_trace(df, color_field, cls, color, project_polygon, tile_opacity)
            if trace:
                fig.add_trace(trace)
    else:
        z, x_vals, y_vals, text, colorscale, n_classes = _build_raster(df, color_field, FINAL_CLASS_COLORS)
        fig.add_trace(go.Heatmap(
            z=z, x=x_vals, y=y_vals, text=text, colorscale=colorscale,
            zmin=0, zmax=n_classes, showscale=False, hoverinfo="text",
            xgap=0, ygap=0, opacity=tile_opacity,
        ))
else:
    y_vals = df["lat"].to_numpy()

# Draw vegetation before water. Some mapped park/green polygons border or
# still visually overlap tiny water features, and Plotly paints later traces on
# top of earlier ones. Water must be the top layer so real ponds/river slivers
# do not appear green in the visual check.
if show_vegetation:
    trace = _vector_trace(
        _preferred_layer(VEGETATION_CLIPPED_GEOJSON, VEGETATION_GEOJSON),
        FINAL_CLASS_COLORS["Vegetation"],
        simplify_tolerance=0.000002,
    )
    if trace:
        fig.add_trace(trace)
if show_water:
    trace = _vector_trace(
        _preferred_layer(WATER_CLIPPED_GEOJSON, WATER_GEOJSON),
        FINAL_CLASS_COLORS["Water"],
        simplify_tolerance=0.000001,
    )
    if trace:
        fig.add_trace(trace)
if show_buildings:
    trace = _vector_trace(
        _preferred_layer(BUILDINGS_CLIPPED_GEOJSON, BUILDINGS_GEOJSON),
        "#2b2b2b", opacity=0.8, simplify_tolerance=0.000001,
        line_color="#111111", line_width=0.3,
    )
    if trace:
        fig.add_trace(trace)
if show_roads:
    trace = _line_trace(ROADS_GEOJSON, "#888888", width=1.0, clip_polygon=project_polygon)
    if trace:
        fig.add_trace(trace)
if show_highway:
    trace = _line_trace(HIGHWAY_GEOJSON, "#c48b17", width=2.2, clip_polygon=project_polygon)
    if trace:
        fig.add_trace(trace)
if show_railway:
    trace = _line_trace(RAILWAY_GEOJSON, "#7b2fbe", width=1.6, dash="dash", clip_polygon=project_polygon)
    if trace:
        fig.add_trace(trace)

if project_polygon is not None:
    boundary_coords = list(project_polygon.exterior.coords)
    bx, by = zip(*boundary_coords)
    fig.add_trace(go.Scatter(
        x=list(bx), y=list(by), mode="lines",
        line=dict(color="black", width=2),
        name="Project boundary", hoverinfo="skip", showlegend=False,
    ))

# One dummy legend marker per class - the Heatmap and vector traces above
# all have showlegend off, so this is the only legend source.
for cls, color in FINAL_CLASS_COLORS.items():
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=color), name=cls, showlegend=True))

mean_lat = float((project_polygon.bounds[1] + project_polygon.bounds[3]) / 2.0) if project_polygon is not None else float(np.nanmean(df["lat"].to_numpy()))
fig.update_layout(
    height=750,
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1.0),
    legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01),
    margin=dict(l=10, r=10, t=30, b=10),
    dragmode="pan",
)
if project_polygon is not None:
    minx, miny, maxx, maxy = project_polygon.bounds
    lon_pad = (maxx - minx) * 0.08
    lat_pad = (maxy - miny) * 0.08
    fig.update_xaxes(range=[minx - lon_pad, maxx + lon_pad])
    fig.update_yaxes(range=[miny - lat_pad, maxy + lat_pad])
fig.update_yaxes(scaleratio=1.0 / math.cos(math.radians(mean_lat)))
with col_b:
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

st.divider()
st.subheader("What the classifier saw")
img_col1, img_col2 = st.columns(2)
if RAW_JPG.exists():
    img_col1.image(str(RAW_JPG), caption="Raw satellite tile fetched for this area")
if OVERLAY_JPG.exists():
    img_col2.image(str(OVERLAY_JPG), caption="open_map.py's own fused density overlay (fuse_and_render)")
