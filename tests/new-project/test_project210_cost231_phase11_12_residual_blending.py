from __future__ import annotations

import json
import math
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from shapely import wkt
from shapely.geometry import Polygon
from sklearn.neighbors import BallTree

THIS_DIR = Path(__file__).resolve().parent
DATA_ROOT = THIS_DIR / "data"
PROJECT_DIR = DATA_ROOT / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
OUT_DIR = PROJECT_DIR / "cost231_phase11_12_residual_blending"
IMAGE_DIR = OUT_DIR / "images"
HTML_DIR = OUT_DIR / "html"
PROJECT_ID = 210
EARTH_RADIUS_M = 6_371_000.0
CLIP_MIN = -147.0
CLIP_MAX = 0.0

MAPDATA_ROOT = DATA_ROOT / "mapdata" / "Dno19_0095_NewTaipeiCity_5m" / "Dno19_0095_NewTaipeiCity_5m" / "New_TaipeiCity_5m_UTM51N_planet"
CLUTTER_RASTER = MAPDATA_ROOT / "Clutter" / "clutter_5m.grc"
CLUTTER_HEIGHT_RASTER = MAPDATA_ROOT / "Clutter Height" / "clutter_height_5m.grd"
TERRAIN_HEIGHT_RASTER = MAPDATA_ROOT / "Heights" / "height_5m.grd"

RSRP_BINS = [
    (-147, -115, "#991b1b", "-147 to -115"),
    (-115, -105, "#d97706", "-115 to -105"),
    (-105, -95, "#fef08a", "-105 to -95"),
    (-95, -85, "#22c55e", "-95 to -85"),
    (-85, 0, "#15803d", "-85 to 0"),
]


def _ensure_dirs() -> None:
    for path in [OUT_DIR, IMAGE_DIR, HTML_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _save_frame(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[list[float]]]:
    surface = pd.read_parquet(PHASE9_DIR / f"phase9_directional_raw_corrected_surface_project{PROJECT_ID}.parquet")
    grid = pd.read_parquet(PHASE9_DIR / f"phase9_gridanalytics_compatible_grid_project{PROJECT_ID}.parquet")
    dt = pd.read_parquet(PHASE9_DIR / f"phase9_dt_match_project{PROJECT_ID}.parquet")

    regions = pd.read_csv(PROJECT_DIR / "geo_db" / f"map_regions_project_{PROJECT_ID}_active.csv")
    geom = wkt.loads(str(regions.iloc[0]["region_wkt"]))
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    # Cached WKT is lat/lon order.
    polygon_coords = [[float(lat), float(lon)] for lat, lon in geom.exterior.coords]

    return surface, grid, dt, polygon_coords


def _sample_raster(path: Path, lat: pd.Series, lon: pd.Series, default: float = 0.0) -> np.ndarray:
    if not path.exists() or len(lat) == 0:
        return np.full(len(lat), default, dtype=float)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    xs, ys = transformer.transform(lon.to_numpy(dtype=float), lat.to_numpy(dtype=float))
    coords = list(zip(xs, ys))
    out = np.full(len(coords), default, dtype=float)
    with rasterio.open(path) as ds:
        band = ds.read(1, masked=True)
        rows, cols = rowcol(ds.transform, xs, ys)
        for idx, (row, col) in enumerate(zip(rows, cols)):
            if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
                continue
            val = band[row, col]
            if np.ma.is_masked(val):
                continue
            fval = float(val)
            if np.isfinite(fval):
                out[idx] = fval
    return out


def _attach_geo_context(surface: pd.DataFrame, dt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    surface = surface.copy()
    dt = dt.copy()
    surface["clutter_class"] = _sample_raster(CLUTTER_RASTER, surface["lat"], surface["lon"], default=0).astype(int)
    surface["clutter_height_m"] = _sample_raster(CLUTTER_HEIGHT_RASTER, surface["lat"], surface["lon"], default=0)
    surface["terrain_height_m"] = _sample_raster(TERRAIN_HEIGHT_RASTER, surface["lat"], surface["lon"], default=0)

    dt["clutter_class"] = _sample_raster(CLUTTER_RASTER, dt["lat"], dt["lon"], default=0).astype(int)
    dt["clutter_height_m"] = _sample_raster(CLUTTER_HEIGHT_RASTER, dt["lat"], dt["lon"], default=0)
    dt["terrain_height_m"] = _sample_raster(TERRAIN_HEIGHT_RASTER, dt["lat"], dt["lon"], default=0)
    dt["height_bin"] = pd.cut(
        dt["clutter_height_m"].fillna(0),
        bins=[-1, 0, 5, 12, 25, 255],
        labels=["none", "low", "mid", "high", "very_high"],
    ).astype(str)
    surface["height_bin"] = pd.cut(
        surface["clutter_height_m"].fillna(0),
        bins=[-1, 0, 5, 12, 25, 255],
        labels=["none", "low", "mid", "high", "very_high"],
    ).astype(str)
    return surface, dt


def _idw_residual(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    source_delta: np.ndarray,
    decay_m: float,
    max_distance_m: float,
    k: int = 32,
) -> np.ndarray:
    result = np.zeros(len(target_lat), dtype=float)
    valid = (
        np.isfinite(source_lat)
        & np.isfinite(source_lon)
        & np.isfinite(source_delta)
        & (source_delta > -80)
        & (source_delta < 80)
    )
    if len(target_lat) == 0 or valid.sum() == 0:
        return result

    src_coords = np.radians(np.column_stack([source_lat[valid], source_lon[valid]]))
    tgt_coords = np.radians(np.column_stack([target_lat, target_lon]))
    k_eff = min(k, len(src_coords))
    tree = BallTree(src_coords, metric="haversine")
    dist_rad, idx = tree.query(tgt_coords, k=k_eff)
    dist_m = dist_rad * EARTH_RADIUS_M
    src_delta = source_delta[valid][idx]
    weights = np.exp(-dist_m / max(decay_m, 1.0)) / np.power(np.maximum(dist_m, 1.0), 0.35)
    weights = np.where(dist_m <= max_distance_m, weights, 0.0)
    denom = weights.sum(axis=1)
    good = denom > 1e-9
    result[good] = (weights[good] * src_delta[good]).sum(axis=1) / denom[good]
    return result


def _phase11_residual(surface: pd.DataFrame, dt: pd.DataFrame) -> pd.DataFrame:
    out = surface.copy()
    out["phase11_residual_db"] = 0.0
    valid_dt = dt.dropna(subset=["rsrp_measured", "dt_minus_cost231_db", "assigned_technology"]).copy()
    for technology, target_idx in out.groupby("technology", dropna=False).groups.items():
        tech_dt = valid_dt[valid_dt["assigned_technology"].astype(str) == str(technology)]
        if tech_dt.empty:
            continue
        target = out.loc[target_idx]
        out.loc[target_idx, "phase11_residual_db"] = _idw_residual(
            target["lat"].to_numpy(dtype=float),
            target["lon"].to_numpy(dtype=float),
            tech_dt["lat"].to_numpy(dtype=float),
            tech_dt["lon"].to_numpy(dtype=float),
            tech_dt["dt_minus_cost231_db"].to_numpy(dtype=float),
            decay_m=350.0,
            max_distance_m=1200.0,
            k=32,
        )
    out["phase11_rsrp_no_dt_lock"] = np.clip(
        out["raw_cost231_rsrp"].to_numpy(dtype=float) + out["phase11_residual_db"].to_numpy(dtype=float),
        CLIP_MIN,
        CLIP_MAX,
    )
    return _apply_dt_lock(out, "phase11_rsrp_no_dt_lock", "phase11_rsrp")


def _phase12_clutter_residual(phase11: pd.DataFrame, dt: pd.DataFrame) -> pd.DataFrame:
    out = phase11.copy()
    out["phase12_clutter_residual_db"] = np.nan
    valid_dt = dt.dropna(subset=["rsrp_measured", "dt_minus_cost231_db", "assigned_technology", "clutter_class"]).copy()
    global_delta = out["phase11_residual_db"].to_numpy(dtype=float)
    clutter_delta = np.full(len(out), np.nan, dtype=float)

    for (technology, clutter), target_idx in out.groupby(["technology", "clutter_class"], dropna=False).groups.items():
        tech_dt = valid_dt[
            (valid_dt["assigned_technology"].astype(str) == str(technology))
            & (valid_dt["clutter_class"].astype(int) == int(clutter))
        ]
        if len(tech_dt) < 8:
            continue
        target = out.loc[target_idx]
        median_height = float(np.nanmedian(target["clutter_height_m"].to_numpy(dtype=float))) if len(target) else 0.0
        if median_height >= 20:
            decay_m = 140.0
        elif median_height >= 8:
            decay_m = 230.0
        else:
            decay_m = 450.0
        local = _idw_residual(
            target["lat"].to_numpy(dtype=float),
            target["lon"].to_numpy(dtype=float),
            tech_dt["lat"].to_numpy(dtype=float),
            tech_dt["lon"].to_numpy(dtype=float),
            tech_dt["dt_minus_cost231_db"].to_numpy(dtype=float),
            decay_m=decay_m,
            max_distance_m=max(decay_m * 3.0, 350.0),
            k=24,
        )
        clutter_delta[out.index.get_indexer(target_idx)] = local

    use_local = np.isfinite(clutter_delta)
    blended_delta = global_delta.copy()
    blended_delta[use_local] = 0.75 * clutter_delta[use_local] + 0.25 * global_delta[use_local]
    out["phase12_clutter_residual_db"] = blended_delta
    out["phase12_rsrp_no_dt_lock"] = np.clip(
        out["raw_cost231_rsrp"].to_numpy(dtype=float) + blended_delta,
        CLIP_MIN,
        CLIP_MAX,
    )
    out = _apply_dt_lock(out, "phase12_rsrp_no_dt_lock", "phase12_rsrp")
    out["phase12_used_clutter_group"] = use_local
    return out


def _apply_dt_lock(surface: pd.DataFrame, source_col: str, output_col: str) -> pd.DataFrame:
    out = surface.copy()
    if "dt_replacement_rsrp" in out.columns:
        lock = out["dt_replacement_rsrp"].notna()
        out[output_col] = out[source_col].where(~lock, out["dt_replacement_rsrp"])
        out[f"{output_col}_dt_locked"] = lock
    else:
        out[output_col] = out[source_col]
        out[f"{output_col}_dt_locked"] = False
    out[output_col] = np.clip(out[output_col].astype(float), CLIP_MIN, CLIP_MAX)
    return out


def _serving_by_technology(surface: pd.DataFrame, grid: pd.DataFrame, phase_col: str) -> pd.DataFrame:
    idx = surface.groupby(["technology", "grid_id"], dropna=False)[phase_col].idxmax()
    cols = [
        "project_id",
        "grid_id",
        "technology",
        "strict_cell_key",
        "site",
        "sector",
        "band",
        "raw_cost231_rsrp",
        "corrected_rsrp",
        "phase11_rsrp",
        "phase12_rsrp",
        "phase11_residual_db",
        "phase12_clutter_residual_db",
        "clutter_class",
        "clutter_height_m",
        "terrain_height_m",
        "dt_replaced",
        "dt_replacement_rsrp",
    ]
    cols = [c for c in cols if c in surface.columns]
    serving = surface.loc[idx, cols].copy()
    serving = grid.merge(serving, on="grid_id", how="right")
    serving["phase_value_col"] = phase_col
    serving["rsrp"] = serving[phase_col]
    return serving.sort_values(["technology", "grid_id"]).reset_index(drop=True)


def _rsrp_color(value: float) -> str:
    if not np.isfinite(value):
        return "#9ca3af"
    for lo, hi, color, _label in RSRP_BINS:
        if lo <= value < hi:
            return color
    return "#9ca3af"


def _add_legend(fmap: folium.Map) -> None:
    rows = "".join(
        f"<div><span style='display:inline-block;width:12px;height:12px;background:{color};margin-right:6px;'></span>{label}</div>"
        for _lo, _hi, color, label in RSRP_BINS
    )
    html = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: white; border: 1px solid #d1d5db; border-radius: 6px;
                padding: 10px 12px; font-size: 12px;">
      <b>RSRP (dBm)</b>
      {rows}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(html))


def _add_grid_rectangles(fmap: folium.Map, df: pd.DataFrame, value_col: str, layer_name: str, show: bool) -> None:
    layer = folium.FeatureGroup(name=layer_name, show=show)
    for row in df.itertuples(index=False):
        val = float(getattr(row, value_col))
        popup = (
            f"<b>Grid:</b> {row.grid_id}<br>"
            f"<b>Technology:</b> {row.technology}<br>"
            f"<b>RSRP:</b> {val:.2f} dBm<br>"
            f"<b>Cell:</b> {row.strict_cell_key}<br>"
            f"<b>Site/Sector/Band:</b> {row.site} / {row.sector} / {row.band}<br>"
            f"<b>Clutter:</b> {row.clutter_class}<br>"
            f"<b>Clutter height:</b> {float(row.clutter_height_m):.1f}"
        )
        folium.Rectangle(
            bounds=[[row.min_lat, row.min_lon], [row.max_lat, row.max_lon]],
            color=_rsrp_color(val),
            weight=0,
            fill=True,
            fill_color=_rsrp_color(val),
            fill_opacity=0.74,
            popup=folium.Popup(popup, max_width=320),
        ).add_to(layer)
    layer.add_to(fmap)


def _make_html_maps(serving: pd.DataFrame, polygon_coords: list[list[float]]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    center = [
        float(np.mean([p[0] for p in polygon_coords])),
        float(np.mean([p[1] for p in polygon_coords])),
    ]
    for technology in ["4G", "5G"]:
        df = serving[serving["technology"].astype(str) == technology].copy()
        if df.empty:
            continue
        fmap = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron", control_scale=True)
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(fmap)
        folium.Polygon(
            polygon_coords,
            color="#2563eb",
            weight=3,
            fill=False,
            tooltip=f"Project {PROJECT_ID} polygon",
            name="Project polygon",
        ).add_to(fmap)
        _add_grid_rectangles(fmap, df, "corrected_rsrp", "Phase 9 offset baseline", show=False)
        _add_grid_rectangles(fmap, df, "phase11_rsrp", "Phase 11 residual blending", show=False)
        _add_grid_rectangles(fmap, df, "phase12_rsrp", "Phase 12 clutter-aware residual", show=True)
        folium.LayerControl(collapsed=False).add_to(fmap)
        _add_legend(fmap)
        out = HTML_DIR / f"project210_{technology.lower()}_phase11_phase12_interactive_map.html"
        fmap.save(out)
        outputs[technology] = str(out)
    return outputs


def _plot_cdf(ax, values: pd.Series, label: str, color: str) -> None:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return
    arr = np.sort(arr)
    y = np.arange(1, len(arr) + 1) / len(arr) * 100.0
    ax.plot(arr, y, color=color, linewidth=2.3, label=f"{label} (n={len(arr):,})")


def _make_static_images(serving: pd.DataFrame, dt: pd.DataFrame, polygon_coords: list[list[float]]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    colors = {"corrected_rsrp": "#ef4444", "phase11_rsrp": "#2563eb", "phase12_rsrp": "#16a34a"}
    for technology in ["4G", "5G"]:
        df = serving[serving["technology"].astype(str) == technology].copy()
        tech_dt = dt[dt["assigned_technology"].astype(str) == technology].copy()
        if df.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
        for ax, col, title in [
            (axes[0, 0], "corrected_rsrp", "Phase 9 offset baseline"),
            (axes[0, 1], "phase11_rsrp", "Phase 11 residual blending"),
            (axes[1, 0], "phase12_rsrp", "Phase 12 clutter-aware residual"),
        ]:
            sc = ax.scatter(df["center_lon"], df["center_lat"], c=df[col], s=9, cmap="RdYlGn", vmin=-120, vmax=-70, marker="s")
            poly_lon = [p[1] for p in polygon_coords]
            poly_lat = [p[0] for p in polygon_coords]
            ax.plot(poly_lon, poly_lat, color="#2563eb", linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, alpha=0.25)
            fig.colorbar(sc, ax=ax, label="RSRP (dBm)")

        diff = df["phase12_rsrp"] - df["phase11_rsrp"]
        sc = axes[1, 1].scatter(df["center_lon"], df["center_lat"], c=diff, s=9, cmap="coolwarm", vmin=-8, vmax=8, marker="s")
        axes[1, 1].plot([p[1] for p in polygon_coords], [p[0] for p in polygon_coords], color="#2563eb", linewidth=1.5)
        axes[1, 1].set_title("Phase 12 - Phase 11 delta")
        axes[1, 1].set_xlabel("Longitude")
        axes[1, 1].set_ylabel("Latitude")
        axes[1, 1].grid(True, alpha=0.25)
        fig.colorbar(sc, ax=axes[1, 1], label="Delta (dB)")
        fig.suptitle(f"Project 210 Taiwan {technology}: Cost231 Offset Residual Blending", fontsize=16, fontweight="bold")
        out = IMAGE_DIR / f"project210_{technology.lower()}_phase11_phase12_map_comparison.png"
        fig.savefig(out, dpi=170)
        plt.close(fig)
        outputs[f"{technology}_map"] = str(out)

        fig, ax = plt.subplots(figsize=(11, 7))
        _plot_cdf(ax, df["corrected_rsrp"], "Phase 9 offset baseline", colors["corrected_rsrp"])
        _plot_cdf(ax, df["phase11_rsrp"], "Phase 11 residual blending", colors["phase11_rsrp"])
        _plot_cdf(ax, df["phase12_rsrp"], "Phase 12 clutter-aware residual", colors["phase12_rsrp"])
        _plot_cdf(ax, tech_dt["rsrp_measured"], "DT measured", "#7c3aed")
        ax.set_title(f"Project 210 Taiwan {technology}: CDF comparison")
        ax.set_xlabel("RSRP (dBm)")
        ax.set_ylabel("Cumulative percentage (%)")
        ax.set_xlim(CLIP_MIN, -45)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        out = IMAGE_DIR / f"project210_{technology.lower()}_phase11_phase12_cdf_comparison.png"
        fig.savefig(out, dpi=170)
        plt.close(fig)
        outputs[f"{technology}_cdf"] = str(out)
    return outputs


def main() -> None:
    _ensure_dirs()
    surface, grid, dt, polygon_coords = _load_inputs()
    print(
        f"[PHASE11_12][INPUT] surface_rows={len(surface):,} "
        f"grid_pixels={len(grid):,} dt_rows={len(dt):,}",
        flush=True,
    )
    surface, dt = _attach_geo_context(surface, dt)
    print("[PHASE11_12][GEO] clutter/height context attached", flush=True)

    phase11 = _phase11_residual(surface, dt)
    phase12 = _phase12_clutter_residual(phase11, dt)
    serving = _serving_by_technology(phase12, grid, "phase12_rsrp")

    _save_frame(phase12, f"phase11_12_residual_surface_project{PROJECT_ID}")
    _save_frame(serving, f"phase11_12_serving_grid_by_technology_project{PROJECT_ID}")
    _save_frame(dt, f"phase11_12_dt_geo_context_project{PROJECT_ID}")

    images = _make_static_images(serving, dt, polygon_coords)
    html_maps = _make_html_maps(serving, polygon_coords)

    summary = {
        "project_id": PROJECT_ID,
        "phase11": "Technology-aware spatial residual blending: raw Cost231 + IDW(delta DT - raw Cost231), DT pixels locked.",
        "phase12": "Clutter/height-aware residual blending: same-clutter local residual blended with phase11 residual, DT pixels locked.",
        "surface_rows": int(len(phase12)),
        "serving_rows": int(len(serving)),
        "grid_pixels_by_technology": {
            str(k): int(v)
            for k, v in serving.groupby("technology")["grid_id"].nunique().to_dict().items()
        },
        "dt_rows_by_technology": {
            str(k): int(v)
            for k, v in dt.groupby("assigned_technology")["id"].count().to_dict().items()
        },
        "phase12_rows_using_clutter_group": int(phase12["phase12_used_clutter_group"].sum()),
        "images": images,
        "html_maps": html_maps,
    }
    (OUT_DIR / "phase11_12_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
