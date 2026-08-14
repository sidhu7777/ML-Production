from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import transform
from streamlit_folium import st_folium

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
MAPDATA_DIR = DATA_DIR / "mapdata"
PROJECT_DIR = DATA_DIR / "project_210_taiwan"
PROJECT_ID = 210

VECTOR_COLORS = {
    "buildings": "#8b5cf6",
    "vegetation": "#16a34a",
    "water": "#0284c7",
    "riverlake": "#0ea5e9",
    "streets": "#64748b",
    "minorroads": "#94a3b8",
    "majorroads": "#f97316",
    "highways": "#ef4444",
    "railways": "#111827",
    "boundaries": "#2563eb",
    "cities": "#a855f7",
    "airports": "#eab308",
    "roads_in_tunnel": "#475569",
}

RASTER_PALETTE = np.array(
    [
        [31, 41, 55, 0],
        [22, 163, 74, 145],
        [132, 204, 22, 145],
        [250, 204, 21, 145],
        [249, 115, 22, 145],
        [239, 68, 68, 145],
        [14, 165, 233, 145],
        [168, 85, 247, 145],
        [20, 184, 166, 145],
        [244, 114, 182, 145],
        [148, 163, 184, 145],
    ],
    dtype=np.uint8,
)


def normalize_layer_name(path: Path) -> str:
    name = path.stem.lower()
    for prefix in ("taipeicity_",):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def dataset_label(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if any("newtaipeicity" in p for p in parts):
        return "New Taipei City 5m"
    if any("taipei_city" in p or "taipeicity" in p for p in parts):
        return "Taipei City 5m"
    return "Unknown"


@st.cache_data(show_spinner=False)
def build_inventory() -> pd.DataFrame:
    rows: list[dict] = []
    if not MAPDATA_DIR.exists():
        return pd.DataFrame(rows)

    for path in sorted(MAPDATA_DIR.rglob("*")):
        suffix = path.suffix.lower()
        if suffix not in {".tab", ".grd", ".grc", ".ecw"}:
            continue
        kind = "vector" if suffix == ".tab" else "raster" if suffix in {".grd", ".grc"} else "orthophoto"
        layer = normalize_layer_name(path)
        category = path.parent.name
        if kind == "vector" and layer in {"clutter_5m", "clutter_height_5m", "heights_5m", "image_05m"}:
            continue
        rows.append(
            {
                "dataset": dataset_label(path),
                "kind": kind,
                "category": category,
                "layer": layer,
                "file": path.name,
                "path": str(path),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["dataset", "kind", "category", "layer"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_project_polygon() -> tuple[list[list[float]], tuple[float, float, float, float]]:
    region_path = PROJECT_DIR / "geo_db" / f"map_regions_project_{PROJECT_ID}_active.csv"
    regions = pd.read_csv(region_path)
    if regions.empty or "region_wkt" not in regions.columns:
        raise ValueError(f"Project {PROJECT_ID} polygon not found: {region_path}")

    geom = wkt.loads(str(regions.iloc[0]["region_wkt"]))
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)

    # Project WKT is stored as x=lat, y=lon in this cache.
    coords = [[float(lat), float(lon)] for lat, lon in geom.exterior.coords]
    lats = [p[0] for p in coords]
    lons = [p[1] for p in coords]
    return coords, (min(lats), min(lons), max(lats), max(lons))


def project_polygon_lonlat() -> Polygon:
    coords, _ = load_project_polygon()
    return Polygon([(lon, lat) for lat, lon in coords])


def project_bbox_utm(buffer_m: float) -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    poly_utm = transform(transformer.transform, project_polygon_lonlat())
    return poly_utm.buffer(float(buffer_m)).bounds


def utm_bounds_to_latlon(bounds: tuple[float, float, float, float]) -> list[list[float]]:
    minx, miny, maxx, maxy = bounds
    transformer = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
    west, south = transformer.transform(minx, miny)
    east, north = transformer.transform(maxx, maxy)
    return [[south, west], [north, east]]


@st.cache_data(show_spinner=False)
def read_vector_layer(
    path_str: str,
    bbox: tuple[float, float, float, float],
    max_features: int,
    simplify_m: float,
) -> tuple[str, int, int, str]:
    path = Path(path_str)
    try:
        gdf = gpd.read_file(path, bbox=bbox)
    except Exception as exc:
        return "", 0, 0, f"{type(exc).__name__}: {exc}"

    original_count = len(gdf)
    if gdf.empty:
        return "", 0, 0, ""

    if simplify_m > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(float(simplify_m), preserve_topology=True)

    if len(gdf) > max_features:
        indices = np.linspace(0, len(gdf) - 1, max_features, dtype=int)
        gdf = gdf.iloc[indices].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:32651")
    gdf = gdf.to_crs("EPSG:4326")
    keep_cols = [c for c in gdf.columns if c.lower() in {"name", "type", "height", "class", "category"}]
    gdf = gdf[keep_cols + ["geometry"]] if keep_cols else gdf[["geometry"]]
    return gdf.to_json(), original_count, len(gdf), ""


def colorize_continuous(arr: np.ma.MaskedArray, opacity: int) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    mask = np.ma.getmaskarray(arr)
    data = np.where(mask, np.nan, data)
    valid = np.isfinite(data)
    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    if not valid.any():
        return rgba
    lo, hi = np.nanpercentile(data[valid], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(data[valid]), np.nanmax(data[valid]) + 1
    norm = np.clip((data - lo) / max(hi - lo, 1e-9), 0, 1)
    rgba[..., 0] = (40 + 215 * norm).astype(np.uint8)
    rgba[..., 1] = (80 + 150 * (1 - np.abs(norm - 0.55))).astype(np.uint8)
    rgba[..., 2] = (220 * (1 - norm)).astype(np.uint8)
    rgba[..., 3] = np.where(valid, opacity, 0).astype(np.uint8)
    return rgba


def colorize_categorical(arr: np.ma.MaskedArray, opacity: int) -> np.ndarray:
    data = np.asarray(arr.filled(0), dtype=np.int64)
    rgba = RASTER_PALETTE[np.mod(data, len(RASTER_PALETTE))].copy()
    rgba[..., 3] = np.where(np.ma.getmaskarray(arr), 0, opacity).astype(np.uint8)
    rgba[data == 0, 3] = 0
    return rgba


@st.cache_data(show_spinner=False)
def read_raster_overlay(
    path_str: str,
    bbox: tuple[float, float, float, float],
    max_pixels: int,
    opacity: int,
) -> tuple[str, list[list[float]], dict, str]:
    path = Path(path_str)
    try:
        with rasterio.open(path) as ds:
            full = Window(0, 0, ds.width, ds.height)
            window = from_bounds(*bbox, transform=ds.transform).round_offsets().round_lengths()
            window = window.intersection(full)
            width = max(1, int(window.width))
            height = max(1, int(window.height))
            scale = max(width / max_pixels, height / max_pixels, 1)
            out_width = max(1, int(width / scale))
            out_height = max(1, int(height / scale))
            arr = ds.read(
                1,
                window=window,
                out_shape=(out_height, out_width),
                masked=True,
                resampling=Resampling.nearest,
            )
            bounds = ds.window_bounds(window)
            layer_name = normalize_layer_name(path)
            if path.suffix.lower() == ".grc" or "clutter_5m" in layer_name:
                rgba = colorize_categorical(arr, opacity)
            else:
                rgba = colorize_continuous(arr, opacity)

            image = Image.fromarray(rgba, mode="RGBA")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            stats = {
                "source_width": ds.width,
                "source_height": ds.height,
                "shown_width": out_width,
                "shown_height": out_height,
                "dtype": ds.dtypes[0],
                "driver": ds.driver,
                "crs": str(ds.crs),
            }
            return uri, utm_bounds_to_latlon(bounds), stats, ""
    except Exception as exc:
        return "", [], {}, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def load_site_points() -> pd.DataFrame:
    path = PROJECT_DIR / "raw_db" / f"site_prediction_project_{PROJECT_ID}_raw_polygon.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    lat_col = next((c for c in ["Latitude", "latitude", "lat"] if c in df.columns), None)
    lon_col = next((c for c in ["Longitude", "longitude", "lon", "lng"] if c in df.columns), None)
    if not lat_col or not lon_col:
        return pd.DataFrame()
    cols = [lat_col, lon_col] + [c for c in ["site_id", "cell_id", "sector", "band", "technology"] if c in df.columns]
    return df[cols].drop_duplicates().rename(columns={lat_col: "lat", lon_col: "lon"})


def add_project_layers(fmap: folium.Map, show_sites: bool) -> None:
    coords, _ = load_project_polygon()
    folium.Polygon(
        locations=coords,
        color="#2563eb",
        weight=3,
        fill=False,
        tooltip=f"Project {PROJECT_ID} polygon",
        name=f"Project {PROJECT_ID} polygon",
    ).add_to(fmap)

    if show_sites:
        sites = load_site_points()
        layer = folium.FeatureGroup(name=f"Project {PROJECT_ID} site points", show=True)
        for row in sites.itertuples(index=False):
            label = " ".join(
                str(getattr(row, c, ""))
                for c in ["site_id", "cell_id", "sector", "band", "technology"]
                if hasattr(row, c)
            )
            folium.CircleMarker(
                location=[float(row.lat), float(row.lon)],
                radius=3,
                color="#2563eb",
                fill=True,
                fill_opacity=0.9,
                popup=label,
            ).add_to(layer)
        layer.add_to(fmap)


def add_vector_to_map(
    fmap: folium.Map,
    row: pd.Series,
    bbox: tuple[float, float, float, float],
    max_features: int,
    simplify_m: float,
) -> dict:
    geojson, original_count, shown_count, error = read_vector_layer(
        str(row["path"]),
        bbox,
        max_features,
        simplify_m,
    )
    result = {
        "layer": row["layer"],
        "kind": row["kind"],
        "loaded": original_count,
        "shown": shown_count,
        "status": error or "ok",
    }
    if error or not geojson:
        return result

    color = VECTOR_COLORS.get(str(row["layer"]).lower(), "#38bdf8")
    style = {
        "color": color,
        "weight": 1.4 if row["layer"] not in {"buildings", "vegetation"} else 0.6,
        "fillColor": color,
        "fillOpacity": 0.16 if row["layer"] in {"buildings", "vegetation"} else 0.05,
        "opacity": 0.85,
    }
    folium.GeoJson(
        geojson,
        name=f"{row['dataset']} - {row['layer']}",
        style_function=lambda _feature, style=style: style,
        show=True,
    ).add_to(fmap)
    return result


def add_raster_to_map(
    fmap: folium.Map,
    row: pd.Series,
    bbox: tuple[float, float, float, float],
    max_pixels: int,
    opacity: int,
) -> dict:
    uri, bounds, stats, error = read_raster_overlay(str(row["path"]), bbox, max_pixels, opacity)
    result = {
        "layer": row["layer"],
        "kind": row["kind"],
        "loaded": f"{stats.get('source_width', 0)}x{stats.get('source_height', 0)}" if stats else 0,
        "shown": f"{stats.get('shown_width', 0)}x{stats.get('shown_height', 0)}" if stats else 0,
        "status": error or "ok",
    }
    if error or not uri or not bounds:
        return result

    folium.raster_layers.ImageOverlay(
        image=uri,
        bounds=bounds,
        opacity=float(opacity) / 255.0,
        name=f"{row['dataset']} - {row['layer']}",
        interactive=True,
        cross_origin=False,
        zindex=300,
    ).add_to(fmap)
    return result


def main() -> None:
    st.set_page_config(page_title="Taiwan Mapdata Viewer", layout="wide")
    st.title("Taiwan Project 210 Mapdata Viewer")

    inventory = build_inventory()
    if inventory.empty:
        st.error(f"No mapdata files found under {MAPDATA_DIR}")
        return

    datasets = sorted(inventory["dataset"].dropna().unique().tolist())
    with st.sidebar:
        dataset = st.selectbox("Dataset", datasets, index=datasets.index("New Taipei City 5m") if "New Taipei City 5m" in datasets else 0)
        buffer_m = st.slider("Project bbox buffer", 0, 3000, 800, 100)
        max_features = st.slider("Max vector features per layer", 500, 30000, 7000, 500)
        simplify_m = st.slider("Vector simplify meters", 0.0, 20.0, 3.0, 0.5)
        max_raster_pixels = st.slider("Max raster width/height", 300, 1800, 900, 100)
        raster_opacity = st.slider("Raster opacity", 30, 220, 140, 5)
        show_sites = st.checkbox("Show project site points", value=True)

    selected_inventory = inventory[inventory["dataset"] == dataset].copy()
    vector_rows = selected_inventory[selected_inventory["kind"] == "vector"].copy()
    raster_rows = selected_inventory[selected_inventory["kind"] == "raster"].copy()

    default_vectors = [
        layer
        for layer in ["buildings", "vegetation", "water", "riverlake", "majorroads", "highways", "streets"]
        if layer in set(vector_rows["layer"])
    ]
    default_rasters = [
        layer
        for layer in ["clutter_5m", "clutter_height_5m", "height_5m", "heights_5m"]
        if layer in set(raster_rows["layer"])
    ][:1]

    col_a, col_b = st.columns([0.35, 0.65])
    with col_a:
        st.subheader("Folder Inventory")
        st.dataframe(
            selected_inventory[["kind", "category", "layer", "file", "path"]],
            use_container_width=True,
            height=260,
        )
        selected_vectors = st.multiselect(
            "Vector layers",
            vector_rows["layer"].tolist(),
            default=default_vectors,
        )
        selected_rasters = st.multiselect(
            "Raster layers",
            raster_rows["layer"].tolist(),
            default=default_rasters,
        )

    coords, latlon_bounds = load_project_polygon()
    center = [(latlon_bounds[0] + latlon_bounds[2]) / 2.0, (latlon_bounds[1] + latlon_bounds[3]) / 2.0]
    bbox = project_bbox_utm(buffer_m)
    fmap = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(fmap)
    add_project_layers(fmap, show_sites)

    results: list[dict] = []
    for _, row in vector_rows[vector_rows["layer"].isin(selected_vectors)].iterrows():
        results.append(add_vector_to_map(fmap, row, bbox, max_features, simplify_m))
    for _, row in raster_rows[raster_rows["layer"].isin(selected_rasters)].iterrows():
        results.append(add_raster_to_map(fmap, row, bbox, max_raster_pixels, raster_opacity))

    folium.LayerControl(collapsed=False).add_to(fmap)

    with col_b:
        st.subheader("Map View")
        map_key = "|".join(
            [
                dataset,
                str(buffer_m),
                str(max_features),
                str(simplify_m),
                str(max_raster_pixels),
                ",".join(selected_vectors),
                ",".join(selected_rasters),
            ]
        )
        st_folium(fmap, height=760, use_container_width=True, key=f"taiwan-mapdata-{map_key}")

    st.subheader("Loaded Layer Status")
    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("Select one or more layers to render them on the map.")


if __name__ == "__main__":
    main()
