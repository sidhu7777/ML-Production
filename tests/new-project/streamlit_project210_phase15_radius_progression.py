from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import LineString, Point

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from tools.lte_prediction.Sector_wise_prediction_code_copy import compute_sector_rsrp

import streamlit_project210_phase13_beam_check as phase13
import streamlit_project210_phase14_tilt_scale_fix as phase14
from compute_clutter_final_v2 import load_building_gdf, impute_building_heights  # reuse, not reimplement

IDENTITY_PATH = phase13.IDENTITY_PATH
load_identity = phase13.load_identity
_build_grid = phase13._build_grid
RSRP_BINS = phase13.RSRP_BINS
_row_to_site_dict_fixed = phase14._row_to_site_dict_fixed

SITE_ID = "LA201565"
RADII_M = [500, 800, 1000, 1500, 2000]
BASELINE_DATA_DIR = BASELINE_DIR / "data" / "project_210_taiwan"
CLUTTER_TILES_PATH = BASELINE_DATA_DIR / "clutter_tiles_final_v2.geojson"
OBAT_CSV_PATH = BASELINE_DATA_DIR / "ghsobat_project210_bbox.csv"
LIGHT_SPEED_M_S = 299_792_458.0

# Same starting values as ML/tests/baseline/lte_rf_geo_weight_optimizer.py
# CURRENT_WEIGHTS - reused, not reinvented, per the earlier discussion.
DEFAULT_CLUTTER_WEIGHTS = {
    "Dense Urban": -4.5,
    "Urban": -2.5,
    "Suburban": -1.0,
    "Vegetation": -1.8,
    "Water": 1.0,
    "Rural/Open": 0.8,
}
DEFAULT_BUILDING_AREA_WEIGHT = -9.0

# Anything below this is outside any realistic LTE receiver's sensitivity /
# COST-231-Hata's own calibrated validity range - treated as NO COVERAGE
# (excluded from the surface as NaN), not clipped to look like a real,
# identical-everywhere data point. This replaces the old -147dBm hard clip,
# which was silently flattening every deeply-shadowed point to one value.
NO_COVERAGE_THRESHOLD_DBM = -140.0


def _resolution_for(radius_m: float, target_cells_across: int = 40) -> float:
    return max(15.0, round(2.0 * radius_m / target_cells_across))


def _predict_grid(site_dict: dict, freq: float, params: dict, grid_lats: np.ndarray, grid_lons: np.ndarray) -> np.ndarray:
    rsrp = np.array(
        [
            compute_sector_rsrp(site_dict, float(lat), float(lon), freq, params)
            for lat, lon in zip(grid_lats, grid_lons)
        ],
        dtype=float,
    )
    return np.minimum(rsrp, -44.0)


def _mask_no_coverage(values: np.ndarray) -> np.ndarray:
    return np.where(values >= NO_COVERAGE_THRESHOLD_DBM, values, np.nan)


def _spread_at_ring(grid_df, values: np.ndarray, center_lat: float, center_lon: float, radius_m: float) -> float:
    dist_m = 111320.0 * np.sqrt(
        (grid_df["lat"].to_numpy() - center_lat) ** 2
        + (grid_df["lon"].to_numpy() - center_lon) ** 2 * np.cos(np.radians(center_lat)) ** 2
    )
    ring = (dist_m > radius_m * 0.35) & (dist_m < radius_m * 0.55)
    ring_values = values[ring]
    if not np.isfinite(ring_values).any():
        return float("nan")
    return float(np.nanmax(ring_values) - np.nanmin(ring_values))


def _effective_range_by_bearing(
    center_lat: float,
    center_lon: float,
    azimuth: float,
    site_dict: dict,
    freq: float,
    params_common: dict,
    clutter_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    geo_enabled: bool,
    tx_height_m: float,
    clutter_weights: dict,
    building_area_weight: float,
    diffraction_multiplier: float,
    entry_loss_db: float,
    max_distance_m: float = 2500.0,
    step_m: float = 25.0,
    entry_depth_slope_db_per_m: float = -0.5,
) -> dict:
    """For a handful of compass bearings relative to boresight, walk outward
    in step_m increments and report the furthest distance where the predicted
    value is still >= NO_COVERAGE_THRESHOLD_DBM - the real, direction-
    dependent coverage range, instead of one radius applied to every
    direction alike."""
    bearings = {"Front (boresight)": 0.0, "Right side": 90.0, "Back": 180.0, "Left side": 270.0}
    out = {}
    for label, offset in bearings.items():
        bearing = (azimuth + offset) % 360.0
        last_ok_m = 0.0
        distance = step_m
        while distance <= max_distance_m:
            dlat = (distance * np.cos(np.radians(bearing))) / 111320.0
            dlon = (distance * np.sin(np.radians(bearing))) / (111320.0 * max(np.cos(np.radians(center_lat)), 1e-6))
            lat_i = center_lat + dlat
            lon_i = center_lon + dlon
            value = compute_sector_rsrp(site_dict, lat_i, lon_i, freq, params_common)
            if geo_enabled:
                pt_df = pd.DataFrame({"lat": [lat_i], "lon": [lon_i]})
                correction, _ = _geo_correction_db(
                    pt_df, clutter_gdf, buildings_gdf, center_lat, center_lon,
                    tx_height_m=tx_height_m, rx_height_m=1.5, freq_mhz=freq,
                    clutter_weights=clutter_weights, building_area_weight=building_area_weight,
                    diffraction_multiplier=diffraction_multiplier, entry_loss_db=entry_loss_db,
                    entry_depth_slope_db_per_m=entry_depth_slope_db_per_m,
                )
                value = value + correction[0]
            if value < NO_COVERAGE_THRESHOLD_DBM:
                break
            last_ok_m = distance
            distance += step_m
        out[label] = last_ok_m
    return out


@st.cache_data(show_spinner=False)
def load_clutter_tiles() -> gpd.GeoDataFrame:
    if not CLUTTER_TILES_PATH.exists():
        return gpd.GeoDataFrame()
    return gpd.read_file(CLUTTER_TILES_PATH)


@st.cache_data(show_spinner=False)
def load_buildings() -> gpd.GeoDataFrame:
    if not (BASELINE_DATA_DIR / "building_df.csv").exists():
        return gpd.GeoDataFrame()
    buildings = load_building_gdf(BASELINE_DATA_DIR)
    if OBAT_CSV_PATH.exists():
        buildings = impute_building_heights(buildings, OBAT_CSV_PATH)
    else:
        buildings["height"] = 15.0  # fallback only if the height source is missing entirely
    return buildings


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * 6_371_000.0 * np.arcsin(np.sqrt(a)))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return float((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


def _destination_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    dlat = (distance_m * np.cos(np.radians(bearing_deg))) / 111320.0
    dlon = (distance_m * np.sin(np.radians(bearing_deg))) / (111320.0 * max(np.cos(np.radians(lat)), 1e-6))
    return lat + dlat, lon + dlon


def _knife_edge_loss_db(h_obstruction_m: float, d1_m: float, d2_m: float, wavelength_m: float) -> float:
    """Single knife-edge diffraction loss, ITU-R P.526 Fresnel-Kirchhoff
    approximation. h_obstruction_m is the obstacle height ABOVE the direct
    TX-RX line at its position; d1/d2 are its distances to TX and RX."""
    if d1_m <= 0 or d2_m <= 0:
        return 0.0
    v = h_obstruction_m * np.sqrt(2.0 * (d1_m + d2_m) / (wavelength_m * d1_m * d2_m))
    if v <= -0.78:
        return 0.0
    j = 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return float(max(0.0, j))


MAX_PATH_DIFFRACTION_LOSS_DB = 60.0
# Full linear summation of every obstacle's independent single-knife-edge
# loss was found to systematically over-penalize full-polygon runs (median
# shift ~-27 to -30dB, 25%+ of cells hitting the floor, best-server and
# mean-of-candidates converging to nearly the same value because almost
# every candidate got crushed). Naive summation double-counts overlapping
# Fresnel-zone effects between adjacent buildings - real multi-screen
# methods (Deygout/Epstein-Peterson) correct for exactly this. Practical
# compromise used here: the dominant (worst) obstacle counts in full, each
# next-most-severe obstacle contributes at a shrinking fraction of its own
# loss, instead of either "only the worst counts" (under-counts multi-
# building paths) or "every obstacle counts in full" (over-counts them).
DIMINISHING_OBSTACLE_WEIGHT = 0.4


def _path_diffraction_loss_db(
    site_pt: Point,
    target_pt: Point,
    total_dist_m: float,
    buildings_gdf: gpd.GeoDataFrame,
    sindex,
    tx_height_m: float,
    rx_height_m: float,
    wavelength_m: float,
) -> tuple[float, int]:
    """Samples every building the direct site->point line actually crosses,
    computes each one's real knife-edge diffraction loss from its measured/
    imputed height vs. the line-of-sight clearance at its position along the
    path, then combines them with diminishing weight (dominant obstacle in
    full, each subsequent one shrinking) - not a flat constant, not just the
    single worst obstacle, and not full unweighted summation either."""
    line = LineString([site_pt, target_pt])
    candidate_idx = list(sindex.query(line, predicate="intersects"))
    if not candidate_idx:
        return 0.0, 0

    obstacle_losses: list[float] = []
    for j in candidate_idx:
        poly = buildings_gdf.geometry.iloc[j]
        height = buildings_gdf["height"].iloc[j]
        if not np.isfinite(height) or height <= 0:
            continue
        inter = line.intersection(poly)
        if inter.is_empty:
            continue
        coords: list = []
        if inter.geom_type == "Point":
            coords = [(inter.x, inter.y)]
        elif inter.geom_type == "LineString":
            coords = list(inter.coords)
        elif inter.geom_type in ("MultiLineString", "GeometryCollection"):
            for g in getattr(inter, "geoms", []):
                if hasattr(g, "coords"):
                    coords.extend(list(g.coords))
        if not coords:
            continue
        entry_lon, entry_lat = min(coords, key=lambda c: site_pt.distance(Point(c)))
        d1 = _haversine_m(site_pt.y, site_pt.x, entry_lat, entry_lon)
        d1 = max(d1, 1.0)
        d2 = max(total_dist_m - d1, 1.0)
        frac = d1 / max(d1 + d2, 1.0)
        los_height_here = tx_height_m + frac * (rx_height_m - tx_height_m)
        h_obstruction = float(height) - los_height_here
        loss = _knife_edge_loss_db(h_obstruction, d1, d2, wavelength_m)
        if loss > 0:
            obstacle_losses.append(loss)

    if not obstacle_losses:
        return 0.0, 0
    obstacle_losses.sort(reverse=True)
    total_loss = 0.0
    weight = 1.0
    for loss in obstacle_losses:
        total_loss += loss * weight
        weight *= DIMINISHING_OBSTACLE_WEIGHT
    total_loss = min(total_loss, MAX_PATH_DIFFRACTION_LOSS_DB)
    return total_loss, len(obstacle_losses)


MAX_INDOOR_DEPTH_M = 25.0  # 3GPP TR 38.901 O2I convention: depth capped at 25m


def _indoor_depth_m(site_pt: Point, target_pt: Point, poly) -> float:
    """How far the target point sits past the building's own wall, measured
    along the site->target line - not just a flat 'inside, yes/no'. A point
    just past the wall and a point 20m deeper into the same building should
    not carry the same loss; field-tested penetration slopes (0.3-1 dB/m)
    confirm real signal keeps dropping with depth, not just at the wall."""
    line = LineString([site_pt, target_pt])
    inter = line.intersection(poly)
    if inter.is_empty:
        return 0.0
    coords: list = []
    if inter.geom_type == "Point":
        coords = [(inter.x, inter.y)]
    elif inter.geom_type == "LineString":
        coords = list(inter.coords)
    elif inter.geom_type in ("MultiLineString", "GeometryCollection"):
        for g in getattr(inter, "geoms", []):
            if hasattr(g, "coords"):
                coords.extend(list(g.coords))
    if not coords:
        return 0.0
    entry_lon, entry_lat = min(coords, key=lambda c: site_pt.distance(Point(c)))
    depth_m = _haversine_m(entry_lat, entry_lon, target_pt.y, target_pt.x)
    return float(min(depth_m, MAX_INDOOR_DEPTH_M))


FAN_OFFSETS_DEG = (-3.0, -1.5, 0.0, 1.5, 3.0)


def _multi_ray_diffraction_loss_db(
    site_lat: float,
    site_lon: float,
    site_pt: Point,
    target_lat: float,
    target_lon: float,
    total_dist_m: float,
    buildings_gdf: gpd.GeoDataFrame,
    sindex,
    tx_height_m: float,
    rx_height_m: float,
    wavelength_m: float,
) -> tuple[float, int]:
    """A single straight-line check is too fragile at 25m grid spacing in
    dense clutter: the ray to one point can slip through a gap between two
    buildings (near-zero loss) while the ray to the point right next to it
    clips a building corner (30-50dB loss), even though both points sit in
    the same generally-obstructed block. Standard fix used in real RF
    prediction tools: sample a small fan of nearby offset rays to the same
    point and average them, instead of trusting one deterministic line."""
    if total_dist_m <= 1.0:
        return 0.0, 0
    bearing = _bearing_deg(site_lat, site_lon, target_lat, target_lon)
    losses = []
    total_obstacles = 0
    for offset in FAN_OFFSETS_DEG:
        lat_i, lon_i = _destination_point(site_lat, site_lon, bearing + offset, total_dist_m)
        loss, n_obstacles = _path_diffraction_loss_db(
            site_pt, Point(lon_i, lat_i), total_dist_m, buildings_gdf, sindex, tx_height_m, rx_height_m, wavelength_m,
        )
        losses.append(loss)
        total_obstacles += n_obstacles
    return float(np.mean(losses)), total_obstacles


def _lookup_clutter(grid_df: pd.DataFrame, clutter_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if clutter_gdf.empty:
        return pd.DataFrame({"clutter_class": [None] * len(grid_df), "building_area_ratio": [0.0] * len(grid_df)})
    points = gpd.GeoDataFrame(
        {"row_idx": range(len(grid_df))},
        geometry=gpd.points_from_xy(grid_df["lon"], grid_df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, clutter_gdf[["clutter_class", "building_area_ratio", "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates(subset="row_idx").sort_values("row_idx")
    out = joined.set_index("row_idx").reindex(range(len(grid_df)))
    return out[["clutter_class", "building_area_ratio"]].reset_index(drop=True)


OBSTRUCTION_PROXY_CLUTTER_CLASSES = {"Dense Urban", "Urban", "Suburban"}
# These clutter classes are a coarse proxy for "how much building is here" -
# the SAME physical obstruction that indoor entry-loss / real per-path
# knife-edge diffraction below already model explicitly from real building
# footprints and heights. Once that real geometry has actually been found
# for a point, adding this proxy weight (and the building_area_ratio
# footprint weight) on top of it would double-count the same obstruction -
# exactly the "-75 to -140dBm over 25m" over-penalization this was flagged
# for. The proxy is used ONLY as a fallback for points where the path
# genuinely has no building geometry on it at all (truly clear line of
# sight, or a data gap), which is the one case with no explicit obstruction
# term to duplicate.


def _geo_correction_db(
    grid_df: pd.DataFrame,
    clutter_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    center_lat: float,
    center_lon: float,
    tx_height_m: float,
    rx_height_m: float,
    freq_mhz: float,
    clutter_weights: dict,
    building_area_weight: float,
    diffraction_multiplier: float,
    entry_loss_db: float,
    entry_depth_slope_db_per_m: float = -0.5,
) -> tuple[np.ndarray, dict]:
    n = len(grid_df)
    correction = np.zeros(n, dtype=float)
    counts = {"indoor": 0, "obstructed": 0, "clear": 0}
    wavelength_m = LIGHT_SPEED_M_S / (freq_mhz * 1e6)

    clutter_lookup = _lookup_clutter(grid_df, clutter_gdf)
    clutter_classes = clutter_lookup["clutter_class"].to_numpy()
    building_area_ratios = clutter_lookup["building_area_ratio"].to_numpy()

    def _env_adj(cls) -> float:
        # Independent physical effects (foliage absorption, water
        # reflection/multipath, open-area gain) that real building geometry
        # does not model either way - these always apply, regardless of
        # which obstruction branch a point falls into.
        if not cls or cls in OBSTRUCTION_PROXY_CLUTTER_CLASSES:
            return 0.0
        return clutter_weights.get(cls, 0.0)

    if buildings_gdf.empty:
        for i in range(n):
            cls = clutter_classes[i]
            bar = building_area_ratios[i]
            proxy = clutter_weights.get(cls, 0.0) if cls else 0.0
            correction[i] = proxy + (float(bar) if pd.notna(bar) else 0.0) * building_area_weight
            counts["clear"] += 1
        return correction, counts

    site_pt = Point(center_lon, center_lat)
    sindex = buildings_gdf.sindex
    for i in range(n):
        cls = clutter_classes[i]
        bar = building_area_ratios[i]
        env_adj = _env_adj(cls)
        lat_i = float(grid_df["lat"].iloc[i])
        lon_i = float(grid_df["lon"].iloc[i])
        pt = Point(lon_i, lat_i)
        candidate_idx = list(sindex.query(pt, predicate="intersects"))
        containing = [j for j in candidate_idx if buildings_gdf.geometry.iloc[j].contains(pt)]
        if containing:
            depth_m = max(
                (_indoor_depth_m(site_pt, pt, buildings_gdf.geometry.iloc[j]) for j in containing),
                default=0.0,
            )
            # Real geometric building-entry loss IS the obstruction term for
            # an indoor point - no proxy/footprint weight stacked on top.
            correction[i] = env_adj + entry_loss_db + depth_m * entry_depth_slope_db_per_m
            counts["indoor"] += 1
            continue
        total_dist_m = _haversine_m(center_lat, center_lon, lat_i, lon_i)
        diffraction_loss, n_obstacles = _multi_ray_diffraction_loss_db(
            center_lat, center_lon, site_pt, lat_i, lon_i, total_dist_m,
            buildings_gdf, sindex, tx_height_m, rx_height_m, wavelength_m,
        )
        if n_obstacles > 0:
            # Real per-path knife-edge diffraction IS the obstruction term
            # here - same reasoning, no proxy/footprint weight stacked on.
            correction[i] = env_adj - diffraction_loss * diffraction_multiplier
            counts["obstructed"] += 1
        else:
            # No explicit building geometry found on this path at all - the
            # coarse clutter-class/footprint proxy is the only obstruction
            # signal available for this point, so (and only here) it applies.
            proxy = clutter_weights.get(cls, 0.0) if cls else 0.0
            correction[i] = env_adj + proxy + (float(bar) if pd.notna(bar) else 0.0) * building_area_weight
            counts["clear"] += 1

    return correction, counts


def render() -> None:
    st.title("Project 210 Taiwan - Phase 15: Radius Progression + Geo Correction (site LA201565)")
    st.caption(
        "Fixed to site LA201565 only. Antenna model unchanged from Phase 14 (tilt-fixed "
        "3GPP). Building obstruction is real ITU-R P.526 knife-edge diffraction from real "
        "building footprints + GHS-OBAT-measured/imputed heights (both reused from "
        "ML/tests/baseline) - not a flat penalty. NEW in this run: a single straight line "
        "per point was too fragile at 25m spacing in dense clutter (one lucky ray slips "
        "through a gap, the ray to the next pixel clips a wall - unrealistic point-to-"
        "point jumps). Fixed with multi-ray sampling: every point now averages 5 nearby "
        "offset rays (+-1.5deg, +-3deg fan) instead of trusting one deterministic line, "
        "same approach real RF prediction tools use in dense clutter. Clutter class + "
        "proportional building-footprint weight stay as small additive terms, plus indoor "
        "entry loss where the point itself is inside a building. DT calibration is still "
        "NOT included - this stays a pure geometry/physics check."
    )

    identity = load_identity()
    clutter_gdf = load_clutter_tiles()
    buildings_gdf = load_buildings()
    if identity.empty:
        st.error(f"Site identity table not found: {IDENTITY_PATH}")
        return
    if clutter_gdf.empty:
        st.warning(f"Clutter tiles not found at {CLUTTER_TILES_PATH} - geo correction will be all zeros.")
    if buildings_gdf.empty:
        st.warning("Building footprints not found - obstruction/entry terms will be all zeros.")

    site_cells = identity[identity["site"] == SITE_ID].copy().reset_index(drop=True)
    if site_cells.empty:
        st.error(f"Site {SITE_ID} not found in identity table.")
        return
    cell_ids = site_cells["Node_Cell_ID"].tolist()

    with st.sidebar:
        st.subheader("Phase 15 controls")
        cell_id = st.selectbox("Cell (site + sector + band)", cell_ids, index=0, key="phase15_cell")
        antenna_gain = st.slider("Antenna max gain (dBi)", 10.0, 22.0, 18.0, 0.5, key="phase15_gain")

        st.markdown("**Geo correction (reused from baseline weight table)**")
        geo_enabled = st.checkbox("Apply geo correction (clutter + building)", value=True, key="phase15_geo_enabled")
        clutter_weights = dict(DEFAULT_CLUTTER_WEIGHTS)
        if geo_enabled:
            for cls in DEFAULT_CLUTTER_WEIGHTS:
                clutter_weights[cls] = st.slider(
                    f"Clutter: {cls} (dB)", -10.0, 3.0, float(DEFAULT_CLUTTER_WEIGHTS[cls]), 0.5,
                    key=f"phase15_clutter_{cls}",
                )
            building_area_weight = st.slider("Building footprint weight (dB, x building_area_ratio)", -20.0, 2.0, DEFAULT_BUILDING_AREA_WEIGHT, 0.5, key="phase15_bld_area_w")
            diffraction_multiplier = st.slider(
                "Path diffraction loss multiplier (x real knife-edge loss, not a flat dB)",
                0.3, 2.0, 1.0, 0.1, key="phase15_diffraction_mult",
                help="Scales the real per-path ITU-R P.526 knife-edge diffraction loss computed from actual building height vs. line-of-sight clearance - this is NOT a flat penalty, it varies per point by real obstruction geometry.",
            )
            entry_loss_db = st.slider("Building entry / indoor loss at the wall (dB)", -30.0, 0.0, -15.0, 1.0, key="phase15_entry")
            entry_depth_slope = st.slider(
                "Indoor depth penetration slope (dB per metre)", -1.0, 0.0, -0.5, 0.1, key="phase15_depth_slope",
                help="Field-tested range from the antenna research deck was 0.3-1.0 dB/m; loss grows the deeper a point sits inside a building instead of a flat wall-only penalty.",
            )
        else:
            building_area_weight = 0.0
            diffraction_multiplier = 0.0
            entry_loss_db = 0.0
            entry_depth_slope = 0.0

    row = site_cells.loc[site_cells["Node_Cell_ID"] == cell_id].iloc[0]
    st.subheader(f"Cell: {cell_id}  |  azimuth {float(row['azimuth']):.0f} deg  |  tilt (fixed) {float(row.get('Etilt', 0) or 0) / 10.0:.1f} deg")

    center_lat = float(row["lat"])
    center_lon = float(row["lon"])
    az = float(row["azimuth"])
    site_dict = _row_to_site_dict_fixed(row)
    freq = float(row.get("frequency", 1800.0) or 1800.0)
    params_common = {"ue_height": 1.5, "k1": 0, "k2": 0, "cable_loss": 2.0, "antenna_gain": antenna_gain}

    with st.spinner("Computing RSRP grids at 5 radii (antenna-only and with geo correction)..."):
        results = []
        for radius_m in RADII_M:
            resolution_m = _resolution_for(radius_m)
            grid_df, _lat_step, _lon_step = _build_grid(center_lat, center_lon, radius_m, resolution_m)
            grid_lats = grid_df["lat"].to_numpy(dtype=float)
            grid_lons = grid_df["lon"].to_numpy(dtype=float)
            antenna_only = _predict_grid(site_dict, freq, params_common, grid_lats, grid_lons)

            if geo_enabled:
                correction, counts = _geo_correction_db(
                    grid_df, clutter_gdf, buildings_gdf, center_lat, center_lon,
                    tx_height_m=float(row.get("Height", 30.0) or 30.0),
                    rx_height_m=1.5,
                    freq_mhz=freq,
                    clutter_weights=clutter_weights,
                    building_area_weight=building_area_weight,
                    diffraction_multiplier=diffraction_multiplier,
                    entry_loss_db=entry_loss_db,
                    entry_depth_slope_db_per_m=entry_depth_slope,
                )
            else:
                correction = np.zeros(len(grid_df))
                counts = {"indoor": 0, "obstructed": 0, "clear": len(grid_df)}

            with_geo_raw = np.minimum(antenna_only + correction, -44.0)
            antenna_only_masked = _mask_no_coverage(antenna_only)
            with_geo_masked = _mask_no_coverage(with_geo_raw)
            spread_antenna = _spread_at_ring(grid_df, antenna_only_masked, center_lat, center_lon, radius_m)
            spread_geo = _spread_at_ring(grid_df, with_geo_masked, center_lat, center_lon, radius_m)
            pct_above_85_antenna = float((antenna_only >= -85).mean() * 100.0)
            pct_above_85_geo = float((np.nan_to_num(with_geo_masked, nan=-999.0) >= -85).mean() * 100.0)
            pct_no_coverage_geo = float(np.isnan(with_geo_masked).mean() * 100.0)

            results.append(
                {
                    "radius_m": radius_m,
                    "grid_df": grid_df,
                    "antenna_only": antenna_only_masked,
                    "with_geo": with_geo_masked,
                    "spread_antenna": spread_antenna,
                    "spread_geo": spread_geo,
                    "pct_above_85_antenna": pct_above_85_antenna,
                    "pct_above_85_geo": pct_above_85_geo,
                    "pct_no_coverage_geo": pct_no_coverage_geo,
                    "counts": counts,
                }
            )

        effective_range = _effective_range_by_bearing(
            center_lat, center_lon, az, site_dict, freq, params_common,
            clutter_gdf, buildings_gdf, geo_enabled,
            tx_height_m=float(row.get("Height", 30.0) or 30.0),
            clutter_weights=clutter_weights, building_area_weight=building_area_weight,
            diffraction_multiplier=diffraction_multiplier, entry_loss_db=entry_loss_db,
            entry_depth_slope_db_per_m=entry_depth_slope,
        )

    st.subheader("Real, direction-dependent effective range (>= -140dBm), not one radius for every direction")
    st.dataframe(
        pd.DataFrame([{"Direction": k, "Effective range (m)": int(v)} for k, v in effective_range.items()]),
        use_container_width=True,
        height=175,
    )
    st.caption(
        f"Walked outward in 25m steps from the site along each bearing (relative to the "
        f"{az:.0f} deg boresight) and stopped at the first point where the real prediction "
        f"drops below {NO_COVERAGE_THRESHOLD_DBM:.0f}dBm. This is the actual coverage "
        "boundary per direction - not the radius slider, which is only a viewing window."
    )

    st.subheader("Is coverage actually dropping off with distance now?")
    table_rows = [
        {
            "radius_m": r["radius_m"],
            "% points >= -85dBm (antenna only)": round(r["pct_above_85_antenna"], 1),
            "% points >= -85dBm (with geo)": round(r["pct_above_85_geo"], 1),
            "% no coverage (< -140dBm, with geo)": round(r["pct_no_coverage_geo"], 1),
            "spread dB (antenna only)": round(r["spread_antenna"], 1) if np.isfinite(r["spread_antenna"]) else None,
            "spread dB (with geo)": round(r["spread_geo"], 1) if np.isfinite(r["spread_geo"]) else None,
            "obstructed points": r["counts"]["obstructed"],
            "indoor points": r["counts"]["indoor"],
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=220)

    fig, axes = plt.subplots(2, len(RADII_M), figsize=(4.0 * len(RADII_M), 8.6), constrained_layout=True)
    sc = None
    for col_idx, r in enumerate(results):
        grid_df = r["grid_df"]
        for row_idx, (key, label) in enumerate([("antenna_only", "Antenna only"), ("with_geo", "With geo correction")]):
            ax = axes[row_idx, col_idx]
            values = r[key]
            valid = np.isfinite(values)
            ax.scatter(
                grid_df["lon"][~valid], grid_df["lat"][~valid],
                c="#e5e7eb", s=10, marker="s", label="no coverage" if (col_idx == 0 and row_idx == 0) else None,
            )
            if valid.any():
                sc = ax.scatter(
                    grid_df["lon"][valid], grid_df["lat"][valid], c=values[valid], s=10, cmap="RdYlGn",
                    vmin=NO_COVERAGE_THRESHOLD_DBM, vmax=-70, marker="s",
                )
            ax.plot(center_lon, center_lat, marker=(3, 0, 90 - az), markersize=14, color="black")
            ax.set_title(f"{label}\nr={r['radius_m']}m", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
    if sc is not None:
        fig.colorbar(sc, ax=axes, label="RSRP (dBm)", shrink=0.6)
    fig.suptitle(
        f"{cell_id} - azimuth {az:.0f} deg - antenna only vs. with geo correction "
        f"(gray = below {NO_COVERAGE_THRESHOLD_DBM:.0f}dBm, no coverage)",
        fontsize=12, fontweight="bold",
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if geo_enabled:
        no_cov_trend = [r["pct_no_coverage_geo"] for r in results]
        growing = all(b >= a - 1e-6 for a, b in zip(no_cov_trend, no_cov_trend[1:]))
        st.caption(
            ("Share of points with genuinely NO coverage (gray, < -140dBm) grows with radius "
             "(" + " -> ".join(f"{v:.0f}%" for v in no_cov_trend) + ") and the effective-range "
             "table above shows a real, different distance per direction - coverage now has an "
             "actual boundary per direction instead of being clipped to look the same everywhere "
             "past a certain distance.")
            if growing else
            ("No-coverage share is not clearly growing with radius (" + " -> ".join(f"{v:.0f}%" for v in no_cov_trend) + ") "
             "- check the effective-range table above directly; if all four directions show "
             "similar numbers, the geometry still isn't differentiating direction correctly.")
        )


def main() -> None:
    st.set_page_config(page_title="Project 210 Phase 15", layout="wide")
    render()


if __name__ == "__main__":
    main()
