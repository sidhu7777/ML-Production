"""
Interactive map to visually verify the clutter classification
(compute_clutter_final_v2.py) against real underlying geometry: buildings,
roads, highways, railways, water, project boundary.

Uses Plotly (not matplotlib) specifically for real pan/zoom/scroll in the
browser - the previous static-matplotlib version had no way to zoom into a
tile without regenerating a fixed-size PNG.

Railway and Highway are NOT separate fetches - they're already present in
the same roads_segment.geojson file already being loaded, distinguished by
real Overture fields: subtype=='rail' for railway, class in
('trunk','primary','secondary') for highway-tier roads. "Roads" covers
everything else (subtype=='road' and not highway-tier), so nothing is
plotted twice under two different layers.

Run:
    cd ML
    streamlit run tests/baseline/clutter_map_streamlit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely import wkt as shapely_wkt
from shapely.ops import transform

st.set_page_config(page_title="Clutter Classification Map", layout="wide", page_icon="🗺️")

DATA_ROOT = Path(__file__).parent / "data"

CLASS_COLORS = {
    "Dense Urban": "#b3261e",
    "Urban": "#e08a2b",
    "Suburban": "#d8b656",
    "Water": "#2a6fbd",
    "Vegetation": "#3f8f5c",
    "Rural/Open": "#c9c2b3",
}

HIGHWAY_CLASSES = {"trunk", "primary", "secondary"}


def polygons_to_trace(geoms, name, color, opacity=0.55, line_color=None, line_width=0.0, hoverinfo_skip=True):
    xs, ys = [], []
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms) if geom.geom_type == "MultiPolygon" else []
        for p in polys:
            coords = list(p.exterior.coords)
            xs.extend([c[0] for c in coords] + [None])
            ys.extend([c[1] for c in coords] + [None])
    if not xs:
        return None
    return go.Scattergl(
        x=xs, y=ys, mode="lines", fill="toself", fillcolor=color,
        line=dict(color=line_color or color, width=line_width),
        name=name, opacity=opacity, hoverinfo="skip" if hoverinfo_skip else None,
    )


def lines_to_trace(geoms, name, color, width=1.0, dash=None):
    xs, ys = [], []
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms) if geom.geom_type == "MultiLineString" else []
        for ls in lines:
            coords = list(ls.coords)
            xs.extend([c[0] for c in coords] + [None])
            ys.extend([c[1] for c in coords] + [None])
    if not xs:
        return None
    return go.Scattergl(
        x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash),
        name=name, hoverinfo="skip",
    )


@st.cache_data
def load_data(project_id: int, region: str):
    data_dir = DATA_ROOT / f"project_{project_id}_{region}"

    tiles = gpd.read_file(data_dir / "clutter_tiles_final_v2.geojson")

    poly_gdf = gpd.read_file(data_dir / "project_polygon.geojson")
    poly_lonlat = transform(lambda x, y: (y, x), poly_gdf.geometry.iloc[0])

    # `.intersects()` alone only FILTERS rows - a real railway/river line that
    # merely passes through the polygon still gets drawn in full, tails and
    # all, far outside it (confirmed: this is exactly what produced the long
    # purple/blue lines running way past the boundary). Clipping with
    # `.intersection(poly_lonlat)` keeps only the real portion of each
    # geometry that's actually inside the polygon.
    building_df = pd.read_csv(data_dir / "building_df.csv", low_memory=False)
    geoms = building_df["region_wkt"].apply(shapely_wkt.loads)
    geoms_lonlat = geoms.apply(lambda g: transform(lambda x, y: (y, x), g) if g is not None else None)
    building_gdf = gpd.GeoDataFrame(building_df, geometry=geoms_lonlat, crs="EPSG:4326")
    building_gdf = building_gdf[building_gdf.geometry.notnull() & building_gdf.geometry.is_valid & ~building_gdf.geometry.is_empty]
    building_gdf = building_gdf[building_gdf.geometry.intersects(poly_lonlat)].copy()
    building_gdf["geometry"] = building_gdf.geometry.intersection(poly_lonlat)
    building_gdf = building_gdf[~building_gdf.geometry.is_empty]

    roads_all = gpd.read_file(data_dir / "roads_segment.geojson")
    roads_all = roads_all[roads_all.geometry.intersects(poly_lonlat)].copy()
    roads_all["geometry"] = roads_all.geometry.intersection(poly_lonlat)
    roads_all = roads_all[~roads_all.geometry.is_empty]
    # Real Overture fields already distinguish these - no separate fetch needed.
    is_rail = roads_all.get("subtype", pd.Series("", index=roads_all.index)) == "rail"
    is_highway = (~is_rail) & roads_all.get("class", pd.Series("", index=roads_all.index)).isin(HIGHWAY_CLASSES)
    railway_gdf = roads_all[is_rail]
    highway_gdf = roads_all[is_highway]
    roads_gdf = roads_all[~is_rail & ~is_highway]

    water = gpd.read_file(data_dir / "water.geojson")
    water = water[water.geometry.intersects(poly_lonlat)].copy()
    water["geometry"] = water.geometry.intersection(poly_lonlat)
    water = water[~water.geometry.is_empty]

    return tiles, poly_lonlat, building_gdf, roads_gdf, highway_gdf, railway_gdf, water


st.title("🗺️ Clutter Classification — Visual Check")
st.caption("Interactive - scroll/drag to zoom and pan. Railway and Highway come from the same roads_segment.geojson already loaded (real subtype/class fields), not a separate fetch.")

col_a, col_b = st.columns([1, 3])
with col_a:
    project_id = st.number_input("Project ID", value=210, step=1)
    region = st.text_input("Region", value="taiwan")
    show_tiles = st.checkbox("Clutter tiles (colored)", value=True)
    show_buildings = st.checkbox("Buildings (real Overture polygons)", value=True)
    show_roads = st.checkbox("Roads", value=True)
    show_highway = st.checkbox("Highway (real: class=trunk/primary/secondary)", value=True)
    show_railway = st.checkbox("Railway (real: subtype=rail)", value=True)
    show_water = st.checkbox("Water (real Overture polygons)", value=True)
    tile_opacity = st.slider("Tile fill opacity", 0.1, 1.0, 0.55)

try:
    tiles, poly_lonlat, building_gdf, roads_gdf, highway_gdf, railway_gdf, water = load_data(int(project_id), region)
except FileNotFoundError as e:
    st.error(f"No cached data found for project {project_id}/{region}: {e}")
    st.stop()

with col_a:
    st.markdown("**Class distribution (tile count)**")
    counts = tiles["clutter_class"].value_counts()
    for cls in CLASS_COLORS:
        if cls in counts.index:
            st.markdown(
                f"<span style='display:inline-block;width:12px;height:12px;background:{CLASS_COLORS[cls]};"
                f"border-radius:2px;margin-right:6px;'></span>{cls}: **{counts[cls]}** tiles "
                f"({100*counts[cls]/len(tiles):.1f}%)",
                unsafe_allow_html=True,
            )
    st.markdown(
        f"---\nBuildings shown: **{len(building_gdf)}**  \n"
        f"Roads shown: **{len(roads_gdf)}**  \nHighway segments: **{len(highway_gdf)}**  \n"
        f"Railway segments: **{len(railway_gdf)}**  \nWater features: **{len(water)}**"
    )

fig = go.Figure()

if show_tiles:
    for cls, color in CLASS_COLORS.items():
        subset = tiles[tiles["clutter_class"] == cls]
        if subset.empty:
            continue
        trace = polygons_to_trace(subset.geometry, cls, color, opacity=tile_opacity)
        if trace:
            fig.add_trace(trace)

if show_water:
    trace = polygons_to_trace(water.geometry, "Water", "#1e5ac8", opacity=0.9, line_color="#123f8f", line_width=0.6)
    if trace:
        fig.add_trace(trace)

if show_buildings:
    trace = polygons_to_trace(building_gdf.geometry, "Buildings", "#2b2b2b", opacity=0.8, line_color="#111111", line_width=0.3)
    if trace:
        fig.add_trace(trace)

if show_roads:
    trace = lines_to_trace(roads_gdf.geometry, "Roads", "#888888", width=1.0)
    if trace:
        fig.add_trace(trace)

if show_highway:
    trace = lines_to_trace(highway_gdf.geometry, "Highway", "#c48b17", width=2.2)
    if trace:
        fig.add_trace(trace)

if show_railway:
    trace = lines_to_trace(railway_gdf.geometry, "Railway", "#7b2fbe", width=1.6, dash="dash")
    if trace:
        fig.add_trace(trace)

boundary_coords = list(poly_lonlat.exterior.coords)
bx, by = zip(*boundary_coords)
fig.add_trace(go.Scatter(x=list(bx), y=list(by), mode="lines", line=dict(color="black", width=2), name="Project boundary", hoverinfo="skip"))

minx, miny, maxx, maxy = poly_lonlat.bounds
mean_lat = (miny + maxy) / 2.0
# Fixed initial view = the real polygon's own bounds (+ a small margin) -
# without this, Plotly auto-fits to whatever's in the traces, and since
# geometries are now clipped to the polygon this isn't strictly needed for
# the "long tails" problem anymore, but it's still what keeps the default
# view centered on the project area regardless of which layers are toggled
# on/off (unchecking a layer used to shrink the auto-fit bounds and make
# the project area look tiny - a fixed range fixes that too).
pad = 0.08
lon_pad = (maxx - minx) * pad
lat_pad = (maxy - miny) * pad
fig.update_layout(
    height=800,
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[minx - lon_pad, maxx + lon_pad]),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1.0,
               range=[miny - lat_pad, maxy + lat_pad]),
    legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01),
    margin=dict(l=10, r=10, t=30, b=10),
    dragmode="pan",
)
# Real lon/lat aspect correction (matches the old ax.set_aspect(1/cos(lat)))
import math
fig.update_yaxes(scaleratio=1.0 / math.cos(math.radians(mean_lat)))

with col_b:
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
