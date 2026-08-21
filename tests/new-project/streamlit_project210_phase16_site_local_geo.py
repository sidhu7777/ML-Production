from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Reuses every prediction/diffraction/plotting function from Phase 15 as-is -
# the only thing this phase changes is WHERE the clutter/building data comes
# from (a real 2km-radius fetch centered on the site, instead of the
# whole-project polygon that was found to run out of data around 2.5km).
import streamlit_project210_phase15_radius_progression as phase15

SITE_ID_4G = phase15.SITE_ID  # LA201565
SITE_ID_5G = "GA20000541"  # co-located 5G n78 overlay at the exact same lat/lon, same 3 azimuths
LOCAL_DATA_DIR = THIS_DIR / "data" / "project_210_taiwan" / "site_LA201565_2km_local_geo"
LOCAL_CLUTTER_PATH = LOCAL_DATA_DIR / "clutter_tiles_site_local_2km.geojson"
LOCAL_BUILDINGS_PATH = LOCAL_DATA_DIR / "buildings_site_local_2km.geojson"

RADII_M = phase15.RADII_M
NO_COVERAGE_THRESHOLD_DBM = phase15.NO_COVERAGE_THRESHOLD_DBM


@st.cache_data(show_spinner=False)
def load_local_clutter() -> gpd.GeoDataFrame:
    if not LOCAL_CLUTTER_PATH.exists():
        return gpd.GeoDataFrame()
    return gpd.read_file(LOCAL_CLUTTER_PATH)


@st.cache_data(show_spinner=False)
def load_local_buildings() -> gpd.GeoDataFrame:
    if not LOCAL_BUILDINGS_PATH.exists():
        return gpd.GeoDataFrame()
    gdf = gpd.read_file(LOCAL_BUILDINGS_PATH)
    gdf["height"] = pd.to_numeric(gdf["height"], errors="coerce")
    return gdf


def _compute_band_results(
    row, clutter_gdf, buildings_gdf, geo_enabled, antenna_gain,
    clutter_weights, building_area_weight, diffraction_multiplier, entry_loss_db, entry_depth_slope,
):
    center_lat = float(row["lat"])
    center_lon = float(row["lon"])
    az = float(row["azimuth"])
    site_dict = phase15._row_to_site_dict_fixed(row)
    freq = float(row.get("frequency", 1800.0) or 1800.0)
    params_common = {"ue_height": 1.5, "k1": 0, "k2": 0, "cable_loss": 2.0, "antenna_gain": antenna_gain}
    tx_height_m = float(row.get("Height", 30.0) or 30.0)

    effective_range = phase15._effective_range_by_bearing(
        center_lat, center_lon, az, site_dict, freq, params_common,
        clutter_gdf, buildings_gdf, geo_enabled,
        tx_height_m=tx_height_m,
        clutter_weights=clutter_weights, building_area_weight=building_area_weight,
        diffraction_multiplier=diffraction_multiplier, entry_loss_db=entry_loss_db,
        entry_depth_slope_db_per_m=entry_depth_slope,
        max_distance_m=2500.0, step_m=25.0,
    )

    radius_results = []
    for radius_m in RADII_M:
        resolution_m = phase15._resolution_for(radius_m)
        grid_df, _lat_step, _lon_step = phase15._build_grid(center_lat, center_lon, radius_m, resolution_m)
        grid_lats = grid_df["lat"].to_numpy(dtype=float)
        grid_lons = grid_df["lon"].to_numpy(dtype=float)
        antenna_only = phase15._predict_grid(site_dict, freq, params_common, grid_lats, grid_lons)
        if geo_enabled:
            correction, counts = phase15._geo_correction_db(
                grid_df, clutter_gdf, buildings_gdf, center_lat, center_lon,
                tx_height_m=tx_height_m, rx_height_m=1.5, freq_mhz=freq,
                clutter_weights=clutter_weights, building_area_weight=building_area_weight,
                diffraction_multiplier=diffraction_multiplier, entry_loss_db=entry_loss_db,
                entry_depth_slope_db_per_m=entry_depth_slope,
            )
        else:
            correction = np.zeros(len(grid_df))
            counts = {"indoor": 0, "obstructed": 0, "clear": len(grid_df)}
        with_geo_raw = np.minimum(antenna_only + correction, -44.0)
        with_geo_masked = phase15._mask_no_coverage(with_geo_raw)
        pct_no_coverage = float(np.isnan(with_geo_masked).mean() * 100.0)
        radius_results.append(
            {"radius_m": radius_m, "grid_df": grid_df, "with_geo": with_geo_masked, "pct_no_coverage": pct_no_coverage, "counts": counts}
        )

    return {"az": az, "center_lat": center_lat, "center_lon": center_lon, "freq": freq, "effective_range": effective_range, "radius_results": radius_results}


def render() -> None:
    st.title("Project 210 Taiwan - Phase 16: Real 2km Site-Local Geo Data - 4G vs 5G (site LA201565)")
    st.caption(
        "Same prediction/diffraction code as Phase 15 (antenna, COST-231, multi-ray "
        "knife-edge diffraction, -140dBm no-coverage floor), same real 2km site-local "
        "geo data as before. NEW: compares 4G band 3 (1800MHz, site LA201565) against "
        "the co-located 5G n78 overlay (3300MHz, site GA20000541) - same lat/lon, same "
        "3 azimuths (30/90/180), same real buildings/clutter around them - so any "
        "difference in how far coverage reaches is coming from frequency + tx power, "
        "not from a different location or a different geo dataset. UPDATED diffraction "
        "model: every real building crossing a path now adds its own loss (summed, "
        "capped at 60dB) instead of only the single worst obstacle counting - paths "
        "through multiple buildings are no longer under-counted. Indoor loss now scales "
        "with depth (dB/m, field-tested range from the antenna research deck) instead "
        "of a flat wall-only penalty. Note: both bands still use the same COST-231-Hata "
        "path loss formula here; COST-231 was originally validated up to ~2GHz, so the "
        "5G/3.3GHz row is a known accuracy limitation worth a dedicated 3GPP UMa-NLOS "
        "model later, not yet implemented."
    )

    identity = phase15.load_identity()
    clutter_gdf = load_local_clutter()
    buildings_gdf = load_local_buildings()
    if identity.empty:
        st.error(f"Site identity table not found: {phase15.IDENTITY_PATH}")
        return
    if clutter_gdf.empty or buildings_gdf.empty:
        st.error(
            f"Local 2km geo data not found under {LOCAL_DATA_DIR}. Run "
            "test_project210_phase16_site_local_geo_fetch.py first."
        )
        return

    cells_4g = identity[identity["site"] == SITE_ID_4G].copy().reset_index(drop=True)
    cells_5g = identity[identity["site"] == SITE_ID_5G].copy().reset_index(drop=True)
    if cells_4g.empty or cells_5g.empty:
        st.error(f"Expected both {SITE_ID_4G} (4G) and {SITE_ID_5G} (5G) in the identity table.")
        return

    with st.sidebar:
        st.subheader("Phase 16 controls")
        sector_choice = st.selectbox("Sector (azimuth)", cells_4g["azimuth"].astype(int).tolist(), index=0, key="phase16_sector")
        antenna_gain = st.slider("Antenna max gain (dBi)", 10.0, 22.0, 18.0, 0.5, key="phase16_gain")
        geo_enabled = st.checkbox("Apply geo correction (clutter + building)", value=True, key="phase16_geo_enabled")
        clutter_weights = dict(phase15.DEFAULT_CLUTTER_WEIGHTS)
        if geo_enabled:
            for cls in phase15.DEFAULT_CLUTTER_WEIGHTS:
                clutter_weights[cls] = st.slider(
                    f"Clutter: {cls} (dB)", -10.0, 3.0, float(phase15.DEFAULT_CLUTTER_WEIGHTS[cls]), 0.5,
                    key=f"phase16_clutter_{cls}",
                )
            building_area_weight = st.slider("Building footprint weight (dB, x building_area_ratio)", -20.0, 2.0, phase15.DEFAULT_BUILDING_AREA_WEIGHT, 0.5, key="phase16_bld_area_w")
            diffraction_multiplier = st.slider("Path diffraction loss multiplier", 0.3, 2.0, 1.0, 0.1, key="phase16_diffraction_mult")
            entry_loss_db = st.slider("Building entry / indoor loss at the wall (dB)", -30.0, 0.0, -15.0, 1.0, key="phase16_entry")
            entry_depth_slope = st.slider("Indoor depth penetration slope (dB per metre)", -1.0, 0.0, -0.5, 0.1, key="phase16_depth_slope")
        else:
            building_area_weight = 0.0
            diffraction_multiplier = 0.0
            entry_loss_db = 0.0
            entry_depth_slope = 0.0

    row_4g = cells_4g.loc[cells_4g["azimuth"].astype(int) == int(sector_choice)].iloc[0]
    row_5g = cells_5g.loc[cells_5g["azimuth"].astype(int) == int(sector_choice)].iloc[0]
    st.subheader(f"Sector azimuth {int(sector_choice)} deg  |  4G: {row_4g['Node_Cell_ID']} ({row_4g['frequency']}MHz)  |  5G: {row_5g['Node_Cell_ID']} ({row_5g['frequency']}MHz)")

    with st.spinner("Computing 4G band..."):
        res_4g = _compute_band_results(row_4g, clutter_gdf, buildings_gdf, geo_enabled, antenna_gain, clutter_weights, building_area_weight, diffraction_multiplier, entry_loss_db, entry_depth_slope)
    with st.spinner("Computing 5G band..."):
        res_5g = _compute_band_results(row_5g, clutter_gdf, buildings_gdf, geo_enabled, antenna_gain, clutter_weights, building_area_weight, diffraction_multiplier, entry_loss_db, entry_depth_slope)

    st.subheader("Effective range by direction - 4G (1800MHz) vs 5G (3300MHz), same site, same geo data")
    range_rows = []
    for direction in res_4g["effective_range"]:
        range_rows.append(
            {
                "Direction": direction,
                "4G range (m)": int(res_4g["effective_range"][direction]),
                "5G range (m)": int(res_5g["effective_range"][direction]),
            }
        )
    st.dataframe(pd.DataFrame(range_rows), use_container_width=True, height=175)

    st.subheader("% genuinely no coverage (< -140dBm) by radius - 4G vs 5G")
    cov_rows = []
    for r4, r5 in zip(res_4g["radius_results"], res_5g["radius_results"]):
        cov_rows.append(
            {
                "radius_m": r4["radius_m"],
                "% no coverage (4G)": round(r4["pct_no_coverage"], 1),
                "% no coverage (5G)": round(r5["pct_no_coverage"], 1),
                "obstructed (4G)": r4["counts"]["obstructed"],
                "obstructed (5G)": r5["counts"]["obstructed"],
            }
        )
    st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, height=220)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(RADII_M), figsize=(4.0 * len(RADII_M), 8.8), constrained_layout=True)
    sc = None
    for col_idx, radius_m in enumerate(RADII_M):
        for row_idx, (res, label) in enumerate([(res_4g, f"4G {res_4g['freq']:.0f}MHz"), (res_5g, f"5G {res_5g['freq']:.0f}MHz")]):
            r = res["radius_results"][col_idx]
            ax = axes[row_idx, col_idx]
            grid_df = r["grid_df"]
            values = r["with_geo"]
            valid = np.isfinite(values)
            ax.scatter(grid_df["lon"][~valid], grid_df["lat"][~valid], c="#e5e7eb", s=10, marker="s")
            if valid.any():
                sc = ax.scatter(grid_df["lon"][valid], grid_df["lat"][valid], c=values[valid], s=10, cmap="RdYlGn", vmin=NO_COVERAGE_THRESHOLD_DBM, vmax=-70, marker="s")
            ax.plot(res["center_lon"], res["center_lat"], marker=(3, 0, 90 - res["az"]), markersize=13, color="black")
            ax.set_title(f"{label}\nr={radius_m}m", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
    if sc is not None:
        fig.colorbar(sc, ax=axes, label="RSRP (dBm)", shrink=0.6)
    fig.suptitle("4G (top) vs 5G (bottom) - same site, same real geo data (gray = no coverage)", fontsize=13, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="Project 210 Phase 16", layout="wide")
    render()


if __name__ == "__main__":
    main()
