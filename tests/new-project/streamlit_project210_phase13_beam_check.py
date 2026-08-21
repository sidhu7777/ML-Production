from __future__ import annotations

import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from tools.lte_prediction.Sector_wise_prediction_code_copy import compute_sector_rsrp

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
IDENTITY_PATH = PROJECT_DIR / "baseline_fetch_scope" / "site_identity_strict_cells_project210.parquet"
EARTH_RADIUS_M = 6_371_000.0

RSRP_BINS = [
    (-147, -115, "#991b1b", "-147 to -115"),
    (-115, -105, "#d97706", "-115 to -105"),
    (-105, -95, "#fef08a", "-105 to -95"),
    (-95, -85, "#22c55e", "-95 to -85"),
    (-85, 0, "#15803d", "-85 to 0"),
]

DEFAULT_SITE_ID = "LA201565"


@st.cache_data(show_spinner=False)
def load_identity() -> pd.DataFrame:
    if not IDENTITY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(IDENTITY_PATH)
    for col in ["lat", "lon", "azimuth", "Height", "Mtilt", "Etilt", "tx_power", "frequency", "band"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["site"] = df["site"].astype(str)
    df["sector"] = df["sector"].astype(str)
    df["Node_Cell_ID"] = df["Node_Cell_ID"].astype(str)
    return df.dropna(subset=["lat", "lon", "azimuth"]).copy()


def _build_grid(center_lat: float, center_lon: float, radius_m: float, resolution_m: float) -> pd.DataFrame:
    lat_step = resolution_m / 111320.0
    lon_step = resolution_m / (111320.0 * max(np.cos(np.radians(center_lat)), 1e-6))
    lat_span = radius_m / 111320.0
    lon_span = radius_m / (111320.0 * max(np.cos(np.radians(center_lat)), 1e-6))

    lat_range = np.arange(center_lat - lat_span, center_lat + lat_span, lat_step)
    lon_range = np.arange(center_lon - lon_span, center_lon + lon_span, lon_step)
    lat_grid, lon_grid = np.meshgrid(lat_range, lon_range, indexing="ij")
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()

    dlat = np.radians(lat_flat - center_lat)
    dlon = np.radians(lon_flat - center_lon)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(center_lat)) * np.cos(np.radians(lat_flat)) * np.sin(dlon / 2.0) ** 2
    )
    dist_m = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    mask = dist_m <= radius_m
    return pd.DataFrame(
        {
            "lat": lat_flat[mask],
            "lon": lon_flat[mask],
            "cell_lat_idx": np.round((lat_flat[mask] - center_lat) / lat_step).astype(int),
            "cell_lon_idx": np.round((lon_flat[mask] - center_lon) / lon_step).astype(int),
        }
    ), lat_step, lon_step


def _row_to_site_dict(row: pd.Series) -> dict:
    return {
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "azimuth": float(row["azimuth"]),
        "electrical_tilt": float(row.get("Etilt", 3.0) or 0.0),
        "mechanical_tilt": float(row.get("Mtilt", 0.0) or 0.0),
        "antenna_height": float(row.get("Height", 30.0) or 30.0),
        "tx_power": float(row.get("tx_power", 46.0) or 46.0),
    }


def render() -> None:
    st.title("Project 210 Taiwan - Phase 13: Single Cell+Sector+Band Beam Check")
    st.caption(
        "Debug tool: computes RSRP for ONE specific cell (site + sector + band) in "
        "total isolation, using the SAME production 3GPP antenna model "
        "(tools/lte_prediction/Sector_wise_prediction_code_copy.compute_sector_rsrp). "
        "No other sectors, no best-of merge, nothing else on the map. Just: is this "
        "one cell's own predicted coverage directional, or circular?"
    )

    identity = load_identity()
    if identity.empty:
        st.error(f"Site identity table not found: {IDENTITY_PATH}")
        return

    site_options = sorted(identity["site"].unique().tolist())
    default_index = site_options.index(DEFAULT_SITE_ID) if DEFAULT_SITE_ID in site_options else 0

    with st.sidebar:
        st.subheader("Phase 13 controls")
        site_id = st.selectbox("Site", site_options, index=default_index)

        site_cells = identity[identity["site"] == site_id].copy().reset_index(drop=True)
        cell_ids = site_cells["Node_Cell_ID"].tolist()
        cell_id = st.selectbox("Cell (site + sector + band)", cell_ids, index=0)

        radius_m = st.slider("Grid radius (m)", 200, 1500, 700, 50)
        resolution_m = st.slider("Grid resolution (m)", 10, 50, 25, 5)
        antenna_gain = st.slider("Antenna max gain (dBi)", 10.0, 22.0, 18.0, 0.5)

    row = site_cells.loc[site_cells["Node_Cell_ID"] == cell_id].iloc[0]
    st.subheader(f"Cell: {cell_id}")
    info_cols = st.columns(6)
    info_cols[0].metric("Sector", str(row.get("sector", "n/a")))
    info_cols[1].metric("Band", str(row.get("band", "n/a")))
    info_cols[2].metric("Azimuth", f"{float(row['azimuth']):.0f}")
    info_cols[3].metric("E-tilt", f"{float(row.get('Etilt', 0) or 0):.0f}")
    info_cols[4].metric("M-tilt", f"{float(row.get('Mtilt', 0) or 0):.0f}")
    info_cols[5].metric("Height (m)", f"{float(row.get('Height', 0) or 0):.0f}")

    center_lat = float(row["lat"])
    center_lon = float(row["lon"])
    grid_df, lat_step, lon_step = _build_grid(center_lat, center_lon, radius_m, resolution_m)
    if grid_df.empty:
        st.error("Empty grid - increase radius.")
        return

    site_dict = _row_to_site_dict(row)
    freq = float(row.get("frequency", 1800.0) or 1800.0)
    params_common = {"ue_height": 1.5, "k1": 0, "k2": 0, "cable_loss": 2.0, "antenna_gain": antenna_gain}

    grid_lats = grid_df["lat"].to_numpy(dtype=float)
    grid_lons = grid_df["lon"].to_numpy(dtype=float)
    # compute_sector_rsrp only supports scalar lat/lon per call (production
    # calls it once per point in a loop) - matched here for fidelity.
    rsrp = np.array(
        [
            compute_sector_rsrp(site_dict, float(lat), float(lon), freq, params_common)
            for lat, lon in zip(grid_lats, grid_lons)
        ],
        dtype=float,
    )
    rsrp = np.clip(rsrp, -147.0, -44.0)

    def _color_for(value: float) -> str:
        if not np.isfinite(value):
            return "#9ca3af"
        for lo, hi, color, _label in RSRP_BINS:
            if lo <= value < hi:
                return color
        return "#9ca3af"

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB positron", control_scale=True)
    folium.RegularPolygonMarker(
        location=[center_lat, center_lon],
        number_of_sides=3,
        radius=10,
        rotation=float(row["azimuth"]),
        color="#111827",
        fill=True,
        fill_color="#111827",
        fill_opacity=0.95,
        popup=folium.Popup(f"<b>{cell_id}</b><br>Azimuth: {row['azimuth']}", max_width=260),
    ).add_to(fmap)

    layer = folium.FeatureGroup(name=cell_id, show=True)
    for lat, lon, val in zip(grid_lats, grid_lons, rsrp):
        folium.Rectangle(
            bounds=[[lat - lat_step / 2, lon - lon_step / 2], [lat + lat_step / 2, lon + lon_step / 2]],
            color=_color_for(val),
            weight=0,
            fill=True,
            fill_color=_color_for(val),
            fill_opacity=0.75,
            tooltip=f"{val:.1f} dBm",
        ).add_to(layer)
    layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    st_folium(fmap, height=700, use_container_width=True, returned_objects=[], key=f"phase13-{cell_id}-{radius_m}-{resolution_m}")

    st.caption(
        "Front (boresight) vs. side/back RSRP at a fixed distance, computed for this "
        f"cell: boresight is strongest, azimuth offset weaker. Black triangle marker "
        "shows the antenna position and azimuth direction."
    )


def main() -> None:
    st.set_page_config(page_title="Project 210 Phase 13", layout="wide")
    render()


if __name__ == "__main__":
    main()
