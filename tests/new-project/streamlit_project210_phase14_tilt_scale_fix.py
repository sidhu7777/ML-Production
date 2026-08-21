from __future__ import annotations

import sys
from pathlib import Path

import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from tools.lte_prediction.Sector_wise_prediction_code_copy import compute_sector_rsrp

import streamlit_project210_phase13_beam_check as phase13

IDENTITY_PATH = phase13.IDENTITY_PATH
load_identity = phase13.load_identity
_build_grid = phase13._build_grid
RSRP_BINS = phase13.RSRP_BINS
DEFAULT_SITE_ID = phase13.DEFAULT_SITE_ID

TILT_SCALE = 10.0  # raw DB tilt values are in tenths of a degree (0,10,20...90 -> 0.0-9.0 deg)


def _row_to_site_dict_raw(row) -> dict:
    """Phase 13 behaviour: tilt used exactly as stored (the suspected bug)."""
    return {
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "azimuth": float(row["azimuth"]),
        "electrical_tilt": float(row.get("Etilt", 3.0) or 0.0),
        "mechanical_tilt": float(row.get("Mtilt", 0.0) or 0.0),
        "antenna_height": float(row.get("Height", 30.0) or 30.0),
        "tx_power": float(row.get("tx_power", 46.0) or 46.0),
    }


def _row_to_site_dict_fixed(row) -> dict:
    """Phase 14 fix: divide Etilt/Mtilt by 10 before using them as degrees."""
    return {
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "azimuth": float(row["azimuth"]),
        "electrical_tilt": float(row.get("Etilt", 3.0) or 0.0) / TILT_SCALE,
        "mechanical_tilt": float(row.get("Mtilt", 0.0) or 0.0) / TILT_SCALE,
        "antenna_height": float(row.get("Height", 30.0) or 30.0),
        "tx_power": float(row.get("tx_power", 46.0) or 46.0),
    }


def _predict_grid(site_dict: dict, freq: float, params: dict, grid_lats: np.ndarray, grid_lons: np.ndarray) -> np.ndarray:
    rsrp = np.array(
        [
            compute_sector_rsrp(site_dict, float(lat), float(lon), freq, params)
            for lat, lon in zip(grid_lats, grid_lons)
        ],
        dtype=float,
    )
    return np.clip(rsrp, -147.0, -44.0)


def _color_for(value: float) -> str:
    if not np.isfinite(value):
        return "#9ca3af"
    for lo, hi, color, _label in RSRP_BINS:
        if lo <= value < hi:
            return color
    return "#9ca3af"


def _add_rsrp_layer(fmap: folium.Map, name: str, show: bool, grid_lats, grid_lons, lat_step, lon_step, values) -> None:
    layer = folium.FeatureGroup(name=name, show=show)
    for lat, lon, val in zip(grid_lats, grid_lons, values):
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


def render() -> None:
    st.title("Project 210 Taiwan - Phase 14: Tilt Scale Fix (before / after)")
    st.caption(
        "Same cell, same grid, same production compute_sector_rsrp() as Phase 13. "
        "Only difference: Electrical/Mechanical tilt divided by 10 before being used "
        "as degrees (raw DB values are 0/10/20/.../90 - tenths-of-a-degree convention, "
        "not whole degrees). Toggle the two layers below to compare directly."
    )

    identity = load_identity()
    if identity.empty:
        st.error(f"Site identity table not found: {IDENTITY_PATH}")
        return

    site_options = sorted(identity["site"].unique().tolist())
    default_index = site_options.index(DEFAULT_SITE_ID) if DEFAULT_SITE_ID in site_options else 0

    with st.sidebar:
        st.subheader("Phase 14 controls")
        site_id = st.selectbox("Site", site_options, index=default_index, key="phase14_site")
        site_cells = identity[identity["site"] == site_id].copy().reset_index(drop=True)
        cell_ids = site_cells["Node_Cell_ID"].tolist()
        cell_id = st.selectbox("Cell (site + sector + band)", cell_ids, index=0, key="phase14_cell")

        radius_m = st.slider("Grid radius (m)", 200, 1500, 700, 50, key="phase14_radius")
        resolution_m = st.slider("Grid resolution (m)", 10, 50, 25, 5, key="phase14_res")
        antenna_gain = st.slider("Antenna max gain (dBi)", 10.0, 22.0, 18.0, 0.5, key="phase14_gain")

    row = site_cells.loc[site_cells["Node_Cell_ID"] == cell_id].iloc[0]
    st.subheader(f"Cell: {cell_id}")
    raw_etilt = float(row.get("Etilt", 0) or 0)
    raw_mtilt = float(row.get("Mtilt", 0) or 0)
    info_cols = st.columns(6)
    info_cols[0].metric("Sector", str(row.get("sector", "n/a")))
    info_cols[1].metric("Band", str(row.get("band", "n/a")))
    info_cols[2].metric("Azimuth", f"{float(row['azimuth']):.0f}")
    info_cols[3].metric("E-tilt raw -> fixed", f"{raw_etilt:.0f} -> {raw_etilt / TILT_SCALE:.1f}")
    info_cols[4].metric("M-tilt raw -> fixed", f"{raw_mtilt:.0f} -> {raw_mtilt / TILT_SCALE:.1f}")
    info_cols[5].metric("Height (m)", f"{float(row.get('Height', 0) or 0):.0f}")

    center_lat = float(row["lat"])
    center_lon = float(row["lon"])
    grid_df, lat_step, lon_step = _build_grid(center_lat, center_lon, radius_m, resolution_m)
    if grid_df.empty:
        st.error("Empty grid - increase radius.")
        return

    grid_lats = grid_df["lat"].to_numpy(dtype=float)
    grid_lons = grid_df["lon"].to_numpy(dtype=float)
    freq = float(row.get("frequency", 1800.0) or 1800.0)
    params_common = {"ue_height": 1.5, "k1": 0, "k2": 0, "cable_loss": 2.0, "antenna_gain": antenna_gain}

    rsrp_raw = _predict_grid(_row_to_site_dict_raw(row), freq, params_common, grid_lats, grid_lons)
    rsrp_fixed = _predict_grid(_row_to_site_dict_fixed(row), freq, params_common, grid_lats, grid_lons)

    dist_m = 111320.0 * np.sqrt(
        (grid_lats - center_lat) ** 2 + (grid_lons - center_lon) ** 2 * np.cos(np.radians(center_lat)) ** 2
    )
    ring = (dist_m > radius_m * 0.35) & (dist_m < radius_m * 0.55)
    if ring.any():
        raw_spread = float(np.nanmax(rsrp_raw[ring]) - np.nanmin(rsrp_raw[ring]))
        fixed_spread = float(np.nanmax(rsrp_fixed[ring]) - np.nanmin(rsrp_fixed[ring]))
    else:
        raw_spread = float("nan")
        fixed_spread = float("nan")

    metric_cols = st.columns(2)
    metric_cols[0].metric("Phase 13 (raw tilt) directional spread at mid-radius", f"{raw_spread:.1f} dB")
    metric_cols[1].metric("Phase 14 (tilt / 10) directional spread at mid-radius", f"{fixed_spread:.1f} dB")
    st.caption(
        "Directional spread = max-min RSRP among grid points sitting in a ring around "
        "the site (35%-55% of the chosen radius). A wider spread means a sharper, more "
        "visibly directional lobe; a narrow spread renders as a near-circle."
    )

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

    _add_rsrp_layer(fmap, "Phase 13: raw tilt (buggy)", False, grid_lats, grid_lons, lat_step, lon_step, rsrp_raw)
    _add_rsrp_layer(fmap, "Phase 14: tilt / 10 (fixed)", True, grid_lats, grid_lons, lat_step, lon_step, rsrp_fixed)
    folium.LayerControl(collapsed=False).add_to(fmap)
    st_folium(
        fmap,
        height=700,
        use_container_width=True,
        returned_objects=[],
        key=f"phase14-{cell_id}-{radius_m}-{resolution_m}",
    )

    st.caption(
        "Layer control (top-right): toggle between 'Phase 13: raw tilt (buggy)' and "
        "'Phase 14: tilt / 10 (fixed)' for the exact same cell and grid to see the "
        "before/after directly."
    )


def main() -> None:
    st.set_page_config(page_title="Project 210 Phase 14", layout="wide")
    render()


if __name__ == "__main__":
    main()
