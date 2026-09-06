from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.colors import BoundaryNorm, ListedColormap

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import streamlit_project210_phase15_radius_progression as phase15

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE17_DIR = PROJECT_DIR / "cost231_phase17_geo_dt_comparison"
OUT_DIR = PROJECT_DIR / "cost231_phase19_branch_calibrated_comparison"
PHASE18_DIR = PROJECT_DIR / "cost231_phase18_dt_point_diagnostic"

RSRP_BINS = [
    (-147, -115, "#991b1b", "-147 to -115"),
    (-115, -105, "#d97706", "-115 to -105"),
    (-105, -95, "#fef08a", "-105 to -95"),
    (-95, -85, "#22c55e", "-95 to -85"),
    (-85, 0, "#15803d", "-85 to 0"),
]


@st.cache_data(show_spinner=False)
def load_serving(tech: str) -> pd.DataFrame:
    """Phase 19's own serving grid already carries Phase 9's values
    (corrected_rsrp, frontend_mean_rsrp) since it's built on the same
    _build_serving_grid() Phase 17 uses - only Phase 17's own phase17_rsrp /
    phase17_frontend_mean_rsrp need to be merged in from its separate
    output directory for the 3-way comparison."""
    p19_path = OUT_DIR / f"phase19_serving_grid_{tech.lower()}_project210.parquet"
    p17_path = PHASE17_DIR / f"phase17_serving_grid_{tech.lower()}_project210.parquet"
    if not p19_path.exists() or not p17_path.exists():
        return pd.DataFrame()
    p19 = pd.read_parquet(p19_path)
    p17 = pd.read_parquet(p17_path)[["grid_id", "phase17_rsrp", "phase17_frontend_mean_rsrp"]]
    serving = p19.merge(p17, on="grid_id", how="left")
    if not {"min_lat", "max_lat", "min_lon", "max_lon"}.issubset(serving.columns):
        bounds = load_grid_bounds()
        if not bounds.empty:
            serving = serving.merge(bounds, on="grid_id", how="left")
    return serving


@st.cache_data(show_spinner=False)
def load_dt() -> pd.DataFrame:
    path = PHASE18_DIR / "phase18_dt_point_diagnostic_project210.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_bias_table() -> pd.DataFrame:
    path = OUT_DIR / "phase19_summary.json"
    if not path.exists():
        return pd.DataFrame()
    dt = load_dt()
    if dt.empty:
        return pd.DataFrame()
    table = (
        dt.groupby(["assigned_technology", "clutter_class", "obstruction_branch"])
        .agg(n=("phase18_error_db", "size"), bias_db=("phase18_error_db", "median"))
        .reset_index()
        .rename(columns={"assigned_technology": "technology"})
    )
    table["representative"] = table["n"] >= 8
    return table.sort_values(["technology", "obstruction_branch", "clutter_class"])


@st.cache_data(show_spinner=False)
def load_grid_bounds() -> pd.DataFrame:
    path = PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)[["grid_id", "min_lat", "max_lat", "min_lon", "max_lon"]]


def _color_for(value: float) -> str:
    if not np.isfinite(value):
        return "#9ca3af"
    for lo, hi, color, _label in RSRP_BINS:
        if lo <= value < hi:
            return color
    return "#9ca3af"


@st.cache_data(show_spinner=False)
def _build_grid_map_html(serving: pd.DataFrame, value_col: str, title: str) -> str:
    """Returns the rendered HTML string, not the folium.Map object itself -
    a plain string is what st.cache_data can safely hash/store/deep-copy,
    where a live folium.Map (nested Jinja2 template refs) is not a good
    cache_data candidate. prefer_canvas=True switches Leaflet to canvas
    rendering instead of one SVG element per rectangle - the standard fix
    for maps with thousands of shapes, much faster to paint/pan/zoom."""
    required = ["min_lat", "max_lat", "min_lon", "max_lon", value_col]
    missing = [col for col in required if col not in serving.columns]
    if missing:
        return f"<p>Map cannot render because columns are missing: {', '.join(missing)}</p>"
    df = serving.dropna(subset=required)
    if df.empty:
        return "<p>Map cannot render because no grid cells have bounds and valid values.</p>"
    center = [float(df["lat"].mean()), float(df["lon"].mean())]
    fmap = folium.Map(
        location=center, zoom_start=14, tiles="CartoDB positron", control_scale=True, prefer_canvas=True,
    )
    layer = folium.FeatureGroup(name=title, show=True)
    for row in df.itertuples(index=False):
        val = float(getattr(row, value_col))
        popup = (
            f"<b>Grid:</b> {row.grid_id}<br>"
            f"<b>Serving cell:</b> {row.strict_cell_key}<br>"
            f"<b>Site/Sector/Band:</b> {row.site} / {row.sector} / {row.band}<br>"
            f"<b>Phase 9 (production offset):</b> {row.corrected_rsrp:.1f} dBm<br>"
            f"<b>Phase 17 (geo + IDW DT residual):</b> {row.phase17_rsrp:.1f} dBm<br>"
            f"<b>Phase 19 (geo + branch-calibrated bias):</b> {row.phase19_rsrp:.1f} dBm<br>"
            f"<b>Obstruction branch:</b> {row.obstruction_branch}<br>"
            f"<b>Bias applied:</b> {row.bias_db:+.1f} dB"
        )
        folium.Rectangle(
            bounds=[[row.min_lat, row.min_lon], [row.max_lat, row.max_lon]],
            color=_color_for(val),
            weight=0,
            fill=True,
            fill_color=_color_for(val),
            fill_opacity=0.85,
            tooltip=f"{val:.1f} dBm",
            popup=folium.Popup(popup, max_width=340),
        ).add_to(layer)
    layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap._repr_html_()


_GRID_ID_RE = re.compile(r"R(\d+)C(\d+)")


@st.cache_data(show_spinner=False)
def _build_static_image_png(serving: pd.DataFrame, value_col: str, title: str) -> bytes:
    """Vectorized raster instead of one MplRectangle.add_patch() per grid
    cell (10k+ individual patches was the actual bottleneck). grid_id
    already encodes the cell's row/col on the real 25m mesh (confirmed
    row<->lat and col<->lon correlation ~1.0 against the grid bounds
    table), so the whole surface can be built as one 2D array and drawn
    with a single imshow() call - same visual result, far fewer draw ops.
    Returns PNG bytes (not the Figure) so st.cache_data caches a plain,
    safely-copyable/hashable value instead of a live matplotlib object."""
    m = serving["grid_id"].str.extract(_GRID_ID_RE).astype(float)
    row, col = m[0].to_numpy(), m[1].to_numpy()
    valid = np.isfinite(row) & np.isfinite(col)
    row = row[valid].astype(int)
    col = col[valid].astype(int)
    values = pd.to_numeric(serving.loc[valid, value_col], errors="coerce").to_numpy()

    grid = np.full((int(row.max()) + 1, int(col.max()) + 1), np.nan, dtype=float)
    grid[row, col] = values

    boundaries = [b[0] for b in RSRP_BINS] + [RSRP_BINS[-1][1]]
    cmap = ListedColormap([b[2] for b in RSRP_BINS])
    cmap.set_bad(color="#9ca3af")
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(6, 7.5))
    # row 0 = southernmost (lowest lat) per the grid_bounds check, so
    # origin="lower" keeps south at the bottom, matching real geography.
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, origin="lower", aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c, _ in RSRP_BINS]
    labels = [lbl for _, _, _, lbl in RSRP_BINS]
    ax.legend(handles, labels, loc="lower left", fontsize=7, title="RSRP (dBm)", framealpha=0.9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_map(serving: pd.DataFrame, value_col: str, title: str, view_mode: str, height: int = 420) -> None:
    if view_mode == "Static image":
        st.image(_build_static_image_png(serving, value_col, title), use_container_width=True)
    else:
        components.html(_build_grid_map_html(serving, value_col, title), height=height, scrolling=False)


def _cdf_trace(values: pd.Series, name: str, color: str) -> go.Scatter:
    arr = pd.to_numeric(values, errors="coerce").dropna().sort_values().to_numpy()
    if len(arr) == 0:
        return go.Scatter(x=[], y=[], mode="lines", name=name)
    y = (pd.Series(range(1, len(arr) + 1)) / len(arr) * 100.0).to_numpy()
    return go.Scatter(x=arr, y=y, mode="lines", name=f"{name} (n={len(arr):,})", line=dict(color=color, width=2.5))


def render() -> None:
    st.title("Project 210 Taiwan - Phase 19: Branch-Calibrated Bias vs. Phase 17 vs. Phase 9 (production)")
    st.caption(
        "Phase 17 applied one flat DT-residual per clutter class (IDW-interpolated). "
        "Phase 18 (test_project210_phase18_dt_point_diagnostic.py) showed clutter class alone "
        "wasn't fine-grained enough - the SAME clutter class had opposite-signed errors "
        "depending on whether the path was clear, diffracted around buildings, or indoor. "
        "Phase 19 replaces the IDW residual with a bias read directly from Phase 18's own "
        "measured [clutter class x obstruction branch] error medians, applied only where "
        ">= 8 real DT points support that exact combination."
    )

    serving_4g = load_serving("4G")
    serving_5g = load_serving("5G")
    dt = load_dt()
    bias_table = load_bias_table()
    if serving_4g.empty or serving_5g.empty:
        st.error(
            f"Phase 19 output not found under {OUT_DIR} (or Phase 17 output missing under "
            f"{PHASE17_DIR}). Run test_project210_phase19_branch_calibrated_comparison.py "
            "(and Phase 17's script, if needed) first."
        )
        return

    with st.sidebar:
        view_mode = st.radio(
            "Map view", ["Interactive (folium)", "Static image"], index=0, key="phase19_view_mode",
            help="Interactive is a real Leaflet map (zoom/pan/hover); Static renders a flat PNG.",
        )
        st.subheader("Branch-conditioned bias table (from Phase 18 DT evidence)")
        st.caption("Only rows with 'representative=True' (>=8 DT points) were actually applied; others fall back to physical model only.")
        st.dataframe(bias_table, use_container_width=True, height=300)
        st.subheader("Clutter weights used (same as Phase 15/16/17)")
        for cls, w in phase15.DEFAULT_CLUTTER_WEIGHTS.items():
            st.text(f"{cls}: {w:+.1f} dB")
        st.text(f"Building footprint weight: {phase15.DEFAULT_BUILDING_AREA_WEIGHT:+.1f} dB (x building_area_ratio)")
        st.text("Building entry (wall): -15.0 dB / Indoor depth slope: -0.5 dB/m")
        st.text("5G n78 technology offset: -2.58 dB")

    st.subheader("Mean shift vs. Phase 9 (production baseline), whole polygon")
    for tech, serving in [("4G", serving_4g), ("5G", serving_5g)]:
        shift17 = float((serving["phase17_rsrp"] - serving["corrected_rsrp"]).mean())
        shift19 = float((serving["phase19_rsrp"] - serving["corrected_rsrp"]).mean())
        branch_share = serving["obstruction_branch"].value_counts(normalize=True)
        cols = st.columns(6)
        cols[0].metric(f"{tech} grid cells", f"{len(serving):,}")
        cols[1].metric(f"{tech} Phase 9 mean", f"{serving['corrected_rsrp'].mean():.1f} dBm")
        cols[2].metric(f"{tech} Phase 17 shift", f"{shift17:+.1f} dB")
        cols[3].metric(f"{tech} Phase 19 shift", f"{shift19:+.1f} dB")
        cols[4].metric(f"{tech} obstructed share", f"{branch_share.get('obstructed', 0.0) * 100:.0f}%")
        cols[5].metric(f"{tech} indoor share", f"{branch_share.get('indoor', 0.0) * 100:.0f}%")

    st.header("Phase 9 vs. Phase 17 vs. Phase 19")
    st.caption(
        "'Frontend' reproduces production's current default aggregation "
        "(lteGridAggregationMethod = \"mean\"). 'Serving cell' is the single best-server "
        "value - the real methodology comparison. Pick one technology + one aggregation at "
        "a time below (was rendering all 12 maps unconditionally before - each render is "
        "cached per exact combination, so revisiting one you've already viewed is instant)."
    )
    pick_cols = st.columns(2)
    with pick_cols[0]:
        tech_pick = st.radio("Technology", ["4G", "5G"], index=0, horizontal=True, key="phase19_map_tech")
    with pick_cols[1]:
        agg_pick = st.radio(
            "Aggregation", ["Frontend (mean of candidates)", "Serving cell (best server)"],
            index=1, horizontal=True, key="phase19_map_agg",
        )
    serving_pick = serving_4g if tech_pick == "4G" else serving_5g
    if agg_pick.startswith("Frontend"):
        cols_spec = [
            ("frontend_mean_rsrp", "Phase 9 (production)"),
            ("phase17_frontend_mean_rsrp", "Phase 17 (IDW DT residual)"),
            ("phase19_frontend_mean_rsrp", "Phase 19 (branch-calibrated bias)"),
        ]
    else:
        cols_spec = [
            ("corrected_rsrp", "Phase 9 (production)"),
            ("phase17_rsrp", "Phase 17 (IDW DT residual)"),
            ("phase19_rsrp", "Phase 19 (branch-calibrated bias)"),
        ]
    map_cols = st.columns(3)
    for (value_col, label), map_col in zip(cols_spec, map_cols):
        with map_col:
            st.caption(f"{tech_pick} - {label}")
            _render_map(serving_pick, value_col, label, view_mode)

    st.subheader("CDF: DT measured vs. Phase 9 vs. Phase 17 vs. Phase 19 (at each DT point's nearest grid)")
    for tech, serving in [("4G", serving_4g), ("5G", serving_5g)]:
        dt_tech = dt[dt["assigned_technology"] == tech].copy()
        joined = dt_tech.merge(
            serving[["grid_id", "corrected_rsrp", "phase17_rsrp", "phase19_rsrp"]],
            left_on="nearest_grid_id", right_on="grid_id", how="inner",
        )
        fig = go.Figure()
        fig.add_trace(_cdf_trace(joined["rsrp_measured"], "DT measured", "#2563eb"))
        fig.add_trace(_cdf_trace(joined["corrected_rsrp"], "Phase 9 predicted", "#ef4444"))
        fig.add_trace(_cdf_trace(joined["phase17_rsrp"], "Phase 17 predicted", "#f97316"))
        fig.add_trace(_cdf_trace(joined["phase19_rsrp"], "Phase 19 predicted", "#16a34a"))
        fig.update_layout(
            title=f"{tech} - DT vs Phase 9 vs Phase 17 vs Phase 19 (n_dt_joined={len(joined):,})",
            height=420, xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
            yaxis_range=[0, 100], xaxis_range=[-147, -45],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Phase 19 only: DT vs. predicted, split outdoor vs. indoor")
    st.caption(
        "Phase 19 predicted RSRP only (Phase 9/17 dropped here for clarity). Panel 1: real "
        "DT measurements. Panel 2: Phase 19 predicted at those same DT-covered grid cells "
        "(matched locations, same n as panel 1). Panel 3: Phase 19 predicted across the "
        "FULL grid's outdoor cells (clear + obstructed branches, not just where DT exists - "
        "a much larger n). Panel 4: Phase 19 predicted across the full grid's indoor cells - "
        "no DT ground truth exists for this one (DT is drive-test data, inherently outdoor)."
    )
    for tech, serving in [("4G", serving_4g), ("5G", serving_5g)]:
        dt_tech = dt[dt["assigned_technology"] == tech].copy()
        joined = dt_tech.merge(
            serving[["grid_id", "phase19_rsrp"]], left_on="nearest_grid_id", right_on="grid_id", how="inner",
        )
        outdoor = serving[serving["obstruction_branch"].isin(["clear", "obstructed"])]
        indoor = serving[serving["obstruction_branch"] == "indoor"]

        st.markdown(f"**{tech}**")
        panels = [
            ("DT measured", joined["rsrp_measured"], "#2563eb"),
            ("Phase 19 predicted (at DT locations)", joined["phase19_rsrp"], "#16a34a"),
            ("Phase 19 predicted (outdoor, full grid)", outdoor["phase19_rsrp"], "#f97316"),
            ("Phase 19 predicted (indoor, full grid)", indoor["phase19_rsrp"], "#7c3aed"),
        ]
        panel_cols = st.columns(4)
        for (label, values, color), col in zip(panels, panel_cols):
            with col:
                arr = pd.to_numeric(values, errors="coerce").dropna()
                fig = go.Figure()
                fig.add_trace(_cdf_trace(values, label, color))
                fig.update_layout(
                    title=f"{label}<br><sup>n={len(arr):,}</sup>", height=320,
                    xaxis_title="RSRP (dBm)", yaxis_title="Cum %",
                    yaxis_range=[0, 100], xaxis_range=[-147, -45], showlegend=False,
                    margin=dict(t=55, b=40, l=45, r=10),
                )
                st.plotly_chart(fig, use_container_width=True)

        combo = go.Figure()
        combo.add_trace(_cdf_trace(joined["rsrp_measured"], "DT measured", "#2563eb"))
        combo.add_trace(_cdf_trace(joined["phase19_rsrp"], "Phase 19 (at DT locations)", "#16a34a"))
        combo.add_trace(_cdf_trace(outdoor["phase19_rsrp"], "Phase 19 (outdoor, full grid)", "#f97316"))
        combo.add_trace(_cdf_trace(indoor["phase19_rsrp"], "Phase 19 (indoor, full grid)", "#7c3aed"))
        combo.update_layout(
            title=f"{tech} - all 4 combined",
            height=420, xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
            yaxis_range=[0, 100], xaxis_range=[-147, -45],
        )
        st.plotly_chart(combo, use_container_width=True)

    st.subheader("Serving cell detail (per grid cell, not an average)")
    tech_choice = st.selectbox("Technology", ["4G", "5G"], index=0, key="phase19_serving_table_tech")
    serving_show = serving_4g if tech_choice == "4G" else serving_5g
    st.dataframe(
        serving_show[
            ["grid_id", "strict_cell_key", "site", "sector", "band", "technology",
             "corrected_rsrp", "phase17_rsrp", "phase19_rsrp", "geo_correction_db",
             "bias_db", "obstruction_branch", "dt_replaced"]
        ].sort_values("grid_id").head(500),
        use_container_width=True, height=350,
    )


def main() -> None:
    st.set_page_config(page_title="Project 210 Phase 19", layout="wide")
    render()


if __name__ == "__main__":
    main()
