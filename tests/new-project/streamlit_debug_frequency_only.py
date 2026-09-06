"""
Streamlit view for the DEBUG "frequency-only 4G vs 5G" comparison.

Renders the products written by test_debug_frequency_only_4g_5g.py:
  1. Whole Taiwan network - every co-located 4G/5G pair, best-server vs
     frontend, 4G and 5G as separate surfaces (+ difference + CDF).
  2. One site's sectors on a local grid - 4G vs 5G.

Each 5G (n78) cell is paired with its co-located 4G cell and both are given the
4G side's parameters (location, azimuth, height, tilt, tx power). Every site
keeps its own real values; only the two halves of a pair are equalised. The
only remaining difference per pair is carrier frequency.

Two RSRP surfaces per technology:
  * "Frequency-only physics" = COST-231 + 3GPP antenna (+ N78 offset) +
    Phase 15 geo-correction - the like-for-like comparison.
  * "Phase 19 (+ DT branch bias)" = the above plus the Phase 18/19
    branch-calibrated DT bias. Its 5G rows come from pre-Phase-20 5G
    drive-test data and are large + positive, so it does NOT isolate
    frequency - shown only for completeness.
"""
from __future__ import annotations

import io
import json
import re
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
OUT_DIR = THIS_DIR / "data" / "debug_frequency_only"

RSRP_BINS = [
    (-147, -115, "#991b1b", "-147 to -115"),
    (-115, -105, "#d97706", "-115 to -105"),
    (-105, -95, "#fef08a", "-105 to -95"),
    (-95, -85, "#22c55e", "-95 to -85"),
    (-85, 0, "#15803d", "-85 to 0"),
]
DELTA_BINS = [
    (-40, -6, "#1e3a8a", "5G much lower (<= -6)"),
    (-6, -2, "#60a5fa", "5G lower (-6..-2)"),
    (-2, 2, "#e5e7eb", "about equal (-2..+2)"),
    (2, 6, "#fca5a5", "5G higher (+2..+6)"),
    (6, 40, "#991b1b", "5G much higher (>= +6)"),
]
_GRID_ID_RE = re.compile(r"R(\d+)C(\d+)")
TECH_COLOR = {"4G": "#2563eb", "5G": "#f97316"}

STAGE_FREQ_ONLY = "Frequency-only physics (recommended)"
STAGE_PHASE19 = "Phase 19 (+ DT branch bias)"
_STAGE_CAND_COL = {STAGE_FREQ_ONLY: "rsrp_freq_only", STAGE_PHASE19: "rsrp_phase19"}
_STAGE_SERV = {
    (STAGE_FREQ_ONLY, "serving"): "serving_rsrp",
    (STAGE_FREQ_ONLY, "frontend"): "frontend_rsrp",
    (STAGE_PHASE19, "serving"): "serving_rsrp_phase19",
    (STAGE_PHASE19, "frontend"): "frontend_rsrp_phase19",
}
_STAGE_DELTA = {
    (STAGE_FREQ_ONLY, "serving"): "serving_delta_5g_minus_4g",
    (STAGE_FREQ_ONLY, "frontend"): "frontend_delta_5g_minus_4g",
    (STAGE_PHASE19, "serving"): "serving_delta_5g_minus_4g_phase19",
    (STAGE_PHASE19, "frontend"): "frontend_delta_5g_minus_4g_phase19",
}
_STAGE_SUMM = {
    (STAGE_FREQ_ONLY, "serving"): "serving",
    (STAGE_FREQ_ONLY, "frontend"): "frontend",
    (STAGE_PHASE19, "serving"): "serving_phase19",
    (STAGE_PHASE19, "frontend"): "frontend_phase19",
}


@st.cache_data(show_spinner=False)
def _load(name: str) -> pd.DataFrame:
    path = OUT_DIR / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_summary() -> dict:
    path = OUT_DIR / "debug_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def _grid_bounds() -> pd.DataFrame:
    s = _load("debug_serving_grid_4g.parquet")
    cols = ["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]
    return s[cols].drop_duplicates() if not s.empty else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _aggregate_band(tech: str, band: str) -> pd.DataFrame:
    """Serving grid restricted to one 4G band, rebuilt from the candidate file
    (matches the script's _aggregate)."""
    cand = _load(f"debug_candidates_{tech.lower()}.parquet")
    bounds = _grid_bounds()
    if cand.empty or bounds.empty:
        return pd.DataFrame()
    if band != "All":
        cand = cand[cand["band_4g"] == int(band)]
    g = cand.groupby("grid_id")
    agg = pd.DataFrame(
        {
            "serving_rsrp": g["rsrp_freq_only"].max(),
            "frontend_rsrp": g["rsrp_freq_only"].mean(),
            "serving_rsrp_phase19": g["rsrp_phase19"].max(),
            "frontend_rsrp_phase19": g["rsrp_phase19"].mean(),
        }
    ).reset_index()
    best = cand.loc[g["rsrp_freq_only"].idxmax(), ["grid_id", "obstruction_branch"]].rename(
        columns={"obstruction_branch": "best_branch"}
    )
    out = bounds.merge(agg, on="grid_id", how="inner").merge(best, on="grid_id", how="left")
    out["lat"] = out["center_lat"]
    out["lon"] = out["center_lon"]
    return out


@st.cache_data(show_spinner=False)
def _delta_band(band: str) -> pd.DataFrame:
    if band == "All":
        return _load("debug_serving_grid_delta.parquet")
    s4 = _aggregate_band("4G", band)
    s5 = _aggregate_band("5G", band)
    if s4.empty or s5.empty:
        return pd.DataFrame()
    keep = ["grid_id", "serving_rsrp", "frontend_rsrp", "serving_rsrp_phase19", "frontend_rsrp_phase19"]
    d = s4[keep].merge(s5[keep], on="grid_id", suffixes=("_4g", "_5g"))
    d["serving_delta_5g_minus_4g"] = d["serving_rsrp_5g"] - d["serving_rsrp_4g"]
    d["frontend_delta_5g_minus_4g"] = d["frontend_rsrp_5g"] - d["frontend_rsrp_4g"]
    d["serving_delta_5g_minus_4g_phase19"] = d["serving_rsrp_phase19_5g"] - d["serving_rsrp_phase19_4g"]
    d["frontend_delta_5g_minus_4g_phase19"] = d["frontend_rsrp_phase19_5g"] - d["frontend_rsrp_phase19_4g"]
    return d.merge(_grid_bounds(), on="grid_id", how="left").assign(
        lat=lambda x: x["center_lat"], lon=lambda x: x["center_lon"]
    )


def _color_for(value: float, bins) -> str:
    if not np.isfinite(value):
        return "#9ca3af"
    for lo, hi, color, _label in bins:
        if lo <= value < hi:
            return color
    return "#9ca3af"


def _cdf_trace(values: pd.Series, name: str, color: str, dash: str | None = None) -> go.Scatter:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = np.sort(arr[np.isfinite(arr)])
    if arr.size == 0:
        return go.Scatter(x=[], y=[], mode="lines", name=name)
    y = np.arange(1, arr.size + 1, dtype=float) / arr.size * 100.0
    return go.Scatter(
        x=arr, y=y, mode="lines", name=f"{name} (n={arr.size:,})",
        line=dict(color=color, width=2.5, dash=dash),
    )


@st.cache_data(show_spinner=False)
def _grid_map_html(df: pd.DataFrame, value_col: str, title: str, bins_key: str) -> str:
    bins = DELTA_BINS if bins_key == "delta" else RSRP_BINS
    d = df.dropna(subset=["min_lat", "max_lat", "min_lon", "max_lon", value_col])
    if d.empty:
        return "<p>No rows.</p>"
    fmap = folium.Map(
        location=[float(d["lat"].mean()), float(d["lon"].mean())],
        zoom_start=13, tiles="CartoDB positron", control_scale=True, prefer_canvas=True,
    )
    layer = folium.FeatureGroup(name=title, show=True)
    for row in d.itertuples(index=False):
        val = float(getattr(row, value_col))
        color = _color_for(val, bins)
        folium.Rectangle(
            bounds=[[row.min_lat, row.min_lon], [row.max_lat, row.max_lon]],
            color=color, weight=0, fill=True, fill_color=color, fill_opacity=0.85,
            tooltip=f"{val:.1f}",
        ).add_to(layer)
    layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap._repr_html_()


@st.cache_data(show_spinner=False)
def _grid_static_png(df: pd.DataFrame, value_col: str, title: str, bins_key: str) -> bytes:
    bins = DELTA_BINS if bins_key == "delta" else RSRP_BINS
    d = df.dropna(subset=[value_col]).copy()
    m = d["grid_id"].astype(str).str.extract(_GRID_ID_RE).astype(float)
    valid = m[0].notna() & m[1].notna()
    rows = m.loc[valid, 0].astype(int).to_numpy()
    cols = m.loc[valid, 1].astype(int).to_numpy()
    vals = pd.to_numeric(d.loc[valid, value_col], errors="coerce").to_numpy(dtype=float)
    if rows.size == 0:
        return b""
    grid = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    grid[rows, cols] = vals

    boundaries = [b[0] for b in bins] + [bins[-1][1]]
    cmap = ListedColormap([b[2] for b in bins])
    cmap.set_bad("#9ca3af")
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(4.8, 5.6))
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, origin="lower", aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=b[2]) for b in bins]
    ax.legend(handles, [b[3] for b in bins], loc="lower left", fontsize=6.5, framealpha=0.9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_grid_map(df: pd.DataFrame, value_col: str, title: str, view_mode: str, bins_key: str = "rsrp") -> None:
    if df.empty or value_col not in df.columns:
        st.warning(f"Missing column: {value_col}")
        return
    if view_mode.startswith("Static"):
        png = _grid_static_png(df, value_col, title, bins_key)
        if png:
            st.image(png, use_container_width=True)
    else:
        components.html(_grid_map_html(df, value_col, title, bins_key), height=430, scrolling=False)


@st.cache_data(show_spinner=False)
def _local_static_png(local: pd.DataFrame, sector: str, value_col: str, center_lat: float, center_lon: float, title: str) -> bytes:
    d = local[local["sector"] == sector]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    boundaries = [b[0] for b in RSRP_BINS] + [RSRP_BINS[-1][1]]
    cmap = ListedColormap([b[2] for b in RSRP_BINS])
    norm = BoundaryNorm(boundaries, cmap.N)
    ax.scatter(d["lon"], d["lat"], c=d[value_col], cmap=cmap, norm=norm, s=9, marker="s")
    ax.plot(center_lon, center_lat, marker="*", markersize=16, color="#111827")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=b[2]) for b in RSRP_BINS]
    ax.legend(handles, [b[3] for b in RSRP_BINS], loc="lower left", fontsize=6.5, framealpha=0.9, title="RSRP (dBm)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _metric_row(pairs: list[tuple[str, str]]) -> None:
    cols = st.columns(len(pairs))
    for col, (label, value) in zip(cols, pairs):
        col.metric(label, value)


def render() -> None:
    st.title("Taiwan Debug - Frequency-only 4G vs 5G")
    summary = _load_summary()
    if not summary:
        st.error(
            f"Debug outputs not found under {OUT_DIR}. Run "
            "`python test_debug_frequency_only_4g_5g.py` first."
        )
        return

    meth = summary["method"]
    conv = summary["frequency_convention"]
    wp = summary["whole_project"]

    st.caption(
        f"Every 5G (n78) cell is paired with its co-located 4G cell "
        f"({meth['n_pairs']} pairs: {meth['n_pairs_band_3']} band-3, {meth['n_pairs_band_28']} band-28; "
        f"{meth['unpaired_4g']}+{meth['unpaired_5g']} unpaired, dropped). Both halves of a pair get the "
        f"**4G side's** parameters - location, azimuth, height, tilt, tx power. Every site keeps its own "
        f"real values; only the two halves of a pair are equalised. Same COST-231 + 3GPP antenna model "
        f"and Phase 15 geo-correction as the pipeline; candidate lists reused from Phase 9."
    )
    st.info(
        f"**The only variable per pair:** 4G = each cell's real frequency "
        f"(band 3 = {conv['4G_band3_mhz']:.0f} MHz, band 28 = {conv['4G_band28_mhz']:.0f} MHz)  |  "
        f"5G = {conv['5G_nominal_mhz']:.0f} MHz nominal, modelled as {conv['5G_modelled_as_mhz']:.0f} MHz "
        f"+ ({conv['5G_n78_offset_db']:.2f} dB) N78 offset. Use the **4G band filter** = 3 for the "
        f"clean \"~1800 vs 3300\" story."
    )

    with st.sidebar:
        st.subheader("Debug view controls")
        stage = st.radio("Model stage", [STAGE_FREQ_ONLY, STAGE_PHASE19], index=0, key="dbg_stage")
        band = st.radio("4G band", ["All", "3", "28"], index=0, horizontal=True, key="dbg_band",
                        help="3 = ~1840 MHz (closest to '1800'); 28 = ~776 MHz.")
        agg_pick = st.radio(
            "Aggregation", ["Best server (max over pairs)", "Frontend (mean over pairs)"],
            index=0, key="dbg_agg",
        )
        view_mode = st.radio("Whole-network map view", ["Static image", "Interactive (folium)"], index=0, key="dbg_view")
        sectors = [c["sector"] for c in summary["single_sector"]["cell_pairs"]]
        sector_pick = st.selectbox("Single-sector: which sector", sectors, index=0, key="dbg_sector")
    agg = "serving" if agg_pick.startswith("Best") else "frontend"

    if stage == STAGE_PHASE19:
        st.warning(
            "Phase 19 adds the branch-calibrated DT bias. Its 5G rows come from pre-Phase-20 5G "
            f"drive-test data and are large + positive (+{wp['5G']['mean_bias_db']:.1f} dB mean vs "
            f"+{wp['4G']['mean_bias_db']:.1f} dB for 4G), which reverses the sign of the frequency "
            "effect. Use 'Frequency-only physics' for a like-for-like comparison."
        )

    serving_4g = _aggregate_band("4G", band)
    serving_5g = _aggregate_band("5G", band)
    delta = _delta_band(band)

    serv_col = _STAGE_SERV[(stage, agg)]
    delta_col = _STAGE_DELTA[(stage, agg)]

    if band == "All":
        k4, k5 = wp["4G"][_STAGE_SUMM[(stage, agg)]], wp["5G"][_STAGE_SUMM[(stage, agg)]]
        k4_mean, k5_mean = k4["mean"], k5["mean"]
        k4_p95, k5_p95 = k4["pct_ge_minus95"], k5["pct_ge_minus95"]
    else:
        k4_mean = float(serving_4g[serv_col].mean())
        k5_mean = float(serving_5g[serv_col].mean())
        k4_p95 = float((serving_4g[serv_col] >= -95).mean() * 100)
        k5_p95 = float((serving_5g[serv_col] >= -95).mean() * 100)

    # ---------------- whole network ----------------
    st.header(f"1. Whole network - co-located pairs (4G band: {band})")
    _metric_row(
        [
            (f"4G mean ({agg})", f"{k4_mean:.1f} dBm"),
            (f"5G mean ({agg})", f"{k5_mean:.1f} dBm"),
            ("5G - 4G", f"{k5_mean - k4_mean:+.1f} dB"),
            ("4G % >= -95", f"{k4_p95:.0f}%"),
            ("5G % >= -95", f"{k5_p95:.0f}%"),
        ]
    )

    dec = wp["delta_5g_minus_4g"]["within_pair_candidate_decomposition"]
    st.caption(
        f"Within-pair 5G-minus-4G decomposition (candidate level, {dec['matched_candidate_rows']:,} matched rows): "
        f"COST-231 path loss + N78 offset **{dec['physical_cost231_plus_n78']:+.1f} dB**, "
        f"geo-correction (shorter wavelength -> more diffraction) **{dec['geo_correction']:+.1f} dB**, "
        f"Phase 19 DT bias **{dec['phase19_dt_bias']:+.1f} dB**. "
        f"Frequency-only total {dec['rsrp_freq_only_delta']:+.1f} dB; Phase 19 total {dec['rsrp_phase19_delta']:+.1f} dB."
    )

    map_cols = st.columns(3)
    with map_cols[0]:
        st.caption(f"4G - {agg}")
        _render_grid_map(serving_4g, serv_col, "4G", view_mode)
    with map_cols[1]:
        st.caption(f"5G - {agg}")
        _render_grid_map(serving_5g, serv_col, "5G", view_mode)
    with map_cols[2]:
        st.caption("5G - 4G difference (dB)")
        _render_grid_map(delta, delta_col, "5G - 4G", view_mode, bins_key="delta")

    fig = go.Figure()
    fig.add_trace(_cdf_trace(serving_4g[_STAGE_SERV[(stage, "serving")]], "4G best server", TECH_COLOR["4G"]))
    fig.add_trace(_cdf_trace(serving_5g[_STAGE_SERV[(stage, "serving")]], "5G best server", TECH_COLOR["5G"]))
    fig.add_trace(_cdf_trace(serving_4g[_STAGE_SERV[(stage, "frontend")]], "4G frontend mean", TECH_COLOR["4G"], dash="dot"))
    fig.add_trace(_cdf_trace(serving_5g[_STAGE_SERV[(stage, "frontend")]], "5G frontend mean", TECH_COLOR["5G"], dash="dot"))
    fig.update_layout(
        title=f"Whole-network CDF - 4G vs 5G ({stage}, band {band})",
        height=430, xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
        yaxis_range=[0, 100], xaxis_range=[-147, -45],
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("All-band numbers (both stages, both aggregations, and per 4G band)"):
        rows = []
        for tech in ["4G", "5G"]:
            for stg in [STAGE_FREQ_ONLY, STAGE_PHASE19]:
                for ag in ["serving", "frontend"]:
                    s = wp[tech][_STAGE_SUMM[(stg, ag)]]
                    rows.append(
                        {
                            "technology": tech,
                            "stage": "freq-only" if stg == STAGE_FREQ_ONLY else "phase19",
                            "aggregation": ag, "band": "All",
                            "mean": round(s["mean"], 1), "median": round(s["median"], 1),
                            "% >= -95": round(s["pct_ge_minus95"], 1), "% >= -105": round(s["pct_ge_minus105"], 1),
                        }
                    )
            for bkey, bstats in wp[tech].get("by_4g_band", {}).items():
                for ag in ["serving", "frontend"]:
                    s = bstats[ag]
                    rows.append(
                        {
                            "technology": tech, "stage": "freq-only", "aggregation": ag,
                            "band": bkey.replace("band_", ""),
                            "mean": round(s["mean"], 1), "median": round(s["median"], 1),
                            "% >= -95": round(s["pct_ge_minus95"], 1), "% >= -105": round(s["pct_ge_minus105"], 1),
                        }
                    )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            f"Branch share (best pair) - 4G: {wp['4G']['branch_share']}  |  5G: {wp['5G']['branch_share']}"
        )

    # ---------------- single sector ----------------
    ss = summary["single_sector"]
    sp = ss["shared_params"]
    st.header(f"2. Single sector ({ss['site_4g']} / {sector_pick}) - local grid")
    st.caption(
        f"Shared params: height {sp['antenna_height_m']:.0f} m, E-tilt {sp['electrical_tilt_deg']:.1f} deg, "
        f"M-tilt {sp['mechanical_tilt_deg']:.1f} deg, tx power {sp['tx_power_dbm']:.0f} dBm, "
        f"gain {sp['antenna_gain_dbi']:.0f} dBi. Star = site."
    )
    local_4g = _load("debug_local_grid_4g.parquet")
    local_5g = _load("debug_local_grid_5g.parquet")
    local_col = _STAGE_CAND_COL[stage]
    if local_4g.empty or local_5g.empty:
        st.warning("Local-grid outputs not found.")
    else:
        lg = ss["local_grid"]
        d4 = local_4g[local_4g["sector"] == sector_pick]
        d5 = local_5g[local_5g["sector"] == sector_pick]
        _metric_row(
            [
                ("4G mean", f"{d4[local_col].mean():.1f} dBm"),
                ("5G mean", f"{d5[local_col].mean():.1f} dBm"),
                ("5G - 4G", f"{d5[local_col].mean() - d4[local_col].mean():+.1f} dB"),
                ("4G % >= -95", f"{(d4[local_col] >= -95).mean() * 100:.0f}%"),
                ("5G % >= -95", f"{(d5[local_col] >= -95).mean() * 100:.0f}%"),
            ]
        )
        cols = st.columns(2)
        with cols[0]:
            st.caption(f"4G - sector {sector_pick}")
            st.image(
                _local_static_png(local_4g, sector_pick, local_col, lg["center_lat"], lg["center_lon"], f"4G - {sector_pick}"),
                use_container_width=True,
            )
        with cols[1]:
            st.caption(f"5G - sector {sector_pick}")
            st.image(
                _local_static_png(local_5g, sector_pick, local_col, lg["center_lat"], lg["center_lon"], f"5G - {sector_pick}"),
                use_container_width=True,
            )
        fig2 = go.Figure()
        fig2.add_trace(_cdf_trace(d4[local_col], "4G", TECH_COLOR["4G"]))
        fig2.add_trace(_cdf_trace(d5[local_col], "5G", TECH_COLOR["5G"]))
        fig2.add_trace(_cdf_trace(d4["physical_rsrp"], "4G physical only", TECH_COLOR["4G"], dash="dot"))
        fig2.add_trace(_cdf_trace(d5["physical_rsrp"], "5G physical only", TECH_COLOR["5G"], dash="dot"))
        fig2.update_layout(
            title=f"Single-sector CDF ({sector_pick}) - 4G vs 5G ({stage})",
            height=400, xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
            yaxis_range=[0, 100], xaxis_range=[-147, -45],
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            f"Local grid: {lg['n_points']:,} points, {lg['radius_m']:.0f} m radius, "
            f"{lg['resolution_m']:.0f} m resolution."
        )


def main() -> None:
    st.set_page_config(page_title="Taiwan Debug - Frequency-only 4G/5G", layout="wide")
    render()


if __name__ == "__main__":
    main()
