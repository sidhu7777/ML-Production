"""Phase 48 dashboard - read-only view of the corrected prediction chain.

Reads only what phase48_corrected_prediction.py wrote. No recomputation, no
database access, no production import. Nothing here writes anything.

Controls: technology (4G / 5G), radius mode (500 m / cell-h link budget),
serving-surface or single-cell view, and the outdoor/indoor/measured CDF.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap

ML = Path(__file__).resolve().parents[3]
PROJECT_DIR = ML / "tests" / "new-project" / "data" / "project_210_taiwan"
GRID_RE = re.compile(r"R(\d+)C(\d+)")

# Same legend as the existing Project 210 dashboards.
RSRP_BINS = [
    (-140, -115, "#991b1b", "-140 to -115"),
    (-115, -105, "#d97706", "-115 to -105"),
    (-105, -95, "#fef08a", "-105 to -95"),
    (-95, -85, "#22c55e", "-95 to -85"),
    (-85, 0, "#15803d", "-85 to 0"),
]

MODES = {"500 m (no backfill)": "fixed500", "Cell-h (link budget)": "cellh"}

st.set_page_config(page_title="Phase 48 corrected prediction", layout="wide")
st.title("Phase 48 - corrected prediction chain")
st.caption(
    "Absolute per-cell membership - no nearest-8 backfill, no competitor-relative 20 dB filter. "
    "Real PAP pattern applied absolutely. Per-technology TX power. 5G per-RE conversion applied. "
    "No smoothing, no joining of patches, no deletion of small components."
)


@st.cache_data(show_spinner=False)
def load(mode: str):
    d = PROJECT_DIR / f"phase48_{mode}"
    if not (d / "phase48_serving.parquet").is_file():
        return None
    return {
        "serving": pd.read_parquet(d / "phase48_serving.parquet"),
        "cells": pd.read_parquet(d / "phase48_cell_surface.parquet"),
        "components": pd.read_csv(d / "phase48_components.csv"),
        "radius": pd.read_csv(d / "phase48_link_budget_radius.csv"),
        "dt": pd.read_parquet(d / "phase48_dt_match.parquet"),
        "dt_uncal": (pd.read_parquet(d / "phase48_dt_match_uncalibrated.parquet")
                     if (d / "phase48_dt_match_uncalibrated.parquet").is_file() else None),
        "summary": json.loads((d / "phase48_summary.json").read_text(encoding="utf-8")),
        "dir": d,
    }


def components(grid_ids: pd.Series) -> int:
    """4-connected component count over R#C# grid ids - Phase 42's rule, the
    same one phase48_corrected_prediction.py uses, so counts are comparable."""
    coords = set()
    for gid in grid_ids.dropna().astype(str):
        m = GRID_RE.search(gid)
        if m:
            coords.add((int(m.group(1)), int(m.group(2))))
    n = 0
    while coords:
        n += 1
        stack = [coords.pop()]
        while stack:
            r, c = stack.pop()
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in coords:
                    coords.remove(nb)
                    stack.append(nb)
    return n


@st.cache_data(show_spinner="Deriving surface...")
def derive(mode: str, value_col: str, floor: float):
    """Apply the floor to the CHOSEN value column and rebuild the surface.

    Membership depends on the value, so the uncalibrated view is not just a
    recolour of the calibrated pixels - different pixels survive the floor.
    """
    full = pd.read_parquet(PROJECT_DIR / f"phase48_{mode}" / "phase48_scored_full.parquet")
    full = full.rename(columns={value_col: "value"})
    kept = full[full["value"] >= floor].copy()
    idx = kept.groupby(["technology", "grid_id"])["value"].idxmax()
    serving = kept.loc[idx].copy()
    comps = (kept.groupby(["strict_cell_key", "technology", "band"])
             .agg(rows=("grid_id", "size"), radius_used_m=("radius_used_m", "first"),
                  components=("grid_id", components))
             .reset_index())
    return kept, serving, comps


def static_png(df: pd.DataFrame, value_col: str, title: str) -> bytes:
    """Raster the grid by its R#C# indices. Missing pixels stay grey."""
    sub = df.dropna(subset=[value_col]).copy()
    rc = sub["grid_id"].astype(str).str.extract(GRID_RE).astype(float)
    ok = rc[0].notna() & rc[1].notna()
    rows = rc.loc[ok, 0].astype(int).to_numpy()
    cols = rc.loc[ok, 1].astype(int).to_numpy()
    vals = pd.to_numeric(sub.loc[ok, value_col], errors="coerce").to_numpy(float)
    if not len(rows):
        return b""
    grid = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    grid[rows, cols] = vals

    boundaries = [b[0] for b in RSRP_BINS] + [RSRP_BINS[-1][1]]
    cmap = ListedColormap([b[2] for b in RSRP_BINS])
    cmap.set_bad(color="#9ca3af")
    fig, ax = plt.subplots(figsize=(4.0, 5.0))
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=BoundaryNorm(boundaries, cmap.N),
              origin="lower", aspect="equal", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=b[2]) for b in RSRP_BINS] + \
              [plt.Rectangle((0, 0), 1, 1, color="#9ca3af")]
    labels = [b[3] for b in RSRP_BINS] + ["Not evaluated / below floor"]
    ax.legend(handles, labels, loc="lower left", fontsize=6, framealpha=0.9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def cdf_trace(values: np.ndarray, name: str, dash=None):
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return go.Scatter(x=v, y=np.arange(1, len(v) + 1) * 100.0 / len(v), mode="lines",
                      name=f"{name} (n={len(v):,})", line=dict(dash=dash) if dash else None)


# ------------------------------------------------------------------ sidebar
mode_label = st.sidebar.radio("Radius mode", list(MODES), index=0)
mode = MODES[mode_label]
data = load(mode)
if data is None:
    st.error(f"No Phase 48 output for mode '{mode}'. Run: "
             f"`python tests/new-project/optimization/phase48_corrected_prediction.py --mode {mode}`")
    st.stop()

tech = st.sidebar.radio("Technology", ["4G", "5G"], index=0)
view = st.sidebar.radio("View", ["Serving surface", "Single cell"], index=0)

serving = data["serving"]
cells = data["cells"]
comp = data["components"]
summary = data["summary"]

serving_t = serving[serving.technology.astype(str).eq(tech)].copy()
cells_t = cells[cells.technology.astype(str).eq(tech)].copy()
comp_t = comp[comp.technology.astype(str).eq(tech)].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Obstruction computed", str(summary.get("obstruction")))
st.sidebar.metric("Cells (all tech)", summary.get("cells"))
st.sidebar.metric("Cells split into islands", f"{summary.get('cells_split_into_islands')} / {len(comp)}")
st.sidebar.metric("Max pieces", summary.get("max_components"))
st.sidebar.caption(f"TX power: 4G {summary['tx_power_dbm']['4G']} dBm / 5G {summary['tx_power_dbm']['5G']} dBm")
st.sidebar.caption(f"Bandwidth: 4G {summary['bandwidth_mhz']['4G']} MHz / 5G {summary['bandwidth_mhz']['5G']} MHz")
st.sidebar.caption(f"Cell-edge threshold: {summary['edge_dbm']} dBm")

# ------------------------------------------------------------------ headline
c = st.columns(5)
c[0].metric("Grid pixels served", f"{serving_t.grid_id.nunique():,}")
c[1].metric(f"{tech} cells", f"{cells_t.strict_cell_key.nunique():,}")
c[2].metric("Split into islands", f"{int((comp_t.components > 1).sum())} / {len(comp_t)}")
c[3].metric("Median RSRP", f"{serving_t.phase48_rsrp.median():.1f} dBm")
rad = data["radius"]
rad_t = rad[rad.technology.astype(str).eq(tech)]
c[4].metric("Link-budget radius (median)", f"{rad_t.link_budget_radius_m.median():.0f} m")

tab_map, tab_cdf, tab_radius, tab_rows = st.tabs(
    ["Static coverage image", "CDF - outdoor / indoor / measured", "Link budget & radius", "Rows"])

# ------------------------------------------------------------------ map tab
STAGES = {"Calibrated (final)": "phase48_rsrp", "Uncalibrated (physical)": "phase48_rsrp_uncal"}

with tab_map:
    stage_label = st.radio("Stage", list(STAGES) + ["Side by side"], horizontal=True, index=0)
    floor = float(summary.get("floor_dbm", -120.0))
    st.caption(f"The floor ({floor:g} dBm) is applied to whichever stage is selected, so the pixel set "
               "changes too - not just the colours.")

    def surfaces(stage: str):
        kept_s, serv_s, comp_s = derive(mode, STAGES[stage], floor)
        return (kept_s[kept_s.technology.astype(str).eq(tech)],
                serv_s[serv_s.technology.astype(str).eq(tech)],
                comp_s[comp_s.technology.astype(str).eq(tech)])

    shown = list(STAGES) if stage_label == "Side by side" else [stage_label]

    if view == "Serving surface":
        st.subheader(f"{tech} best-server surface - {mode_label}")
        cols = st.columns(len(shown))
        stats = {}
        for col, stg in zip(cols, shown):
            _, sv, _ = surfaces(stg)
            with col:
                png = static_png(sv.rename(columns={"value": "v"}), "v", f"{tech} {stg}\n{mode_label}")
                if png:
                    st.image(png, width=460)
                st.metric("Pixels above floor", f"{len(sv):,}")
                st.metric("Median RSRP", f"{sv.value.median():.1f} dBm")
                st.metric("Green (>= -85 dBm)", f"{(sv.value >= -85).mean()*100:.1f} %")
            stats[stg] = sv
        if len(shown) == 2:
            a, b = stats["Uncalibrated (physical)"], stats["Calibrated (final)"]
            d = st.columns(3)
            d[0].metric("Pixels gained by calibration", f"{len(b) - len(a):+,}")
            d[1].metric("Median shift", f"{b.value.median() - a.value.median():+.1f} dB")
            d[2].metric("Green share shift",
                        f"{((b.value >= -85).mean() - (a.value >= -85).mean())*100:+.1f} pp")
        st.caption("Grey = pixel not evaluated for any cell of this technology, or below the floor. "
                   "It is never a filled-in or smoothed value.")
    else:
        st.subheader(f"{tech} single-cell footprint")
        _, _, comp_ref = surfaces(shown[-1])
        opts = comp_ref.sort_values(["components", "strict_cell_key"], ascending=[False, True])
        labels = [f"{r.strict_cell_key} | band={r.band} | {r.components} piece(s) | r={r.radius_used_m:.0f} m"
                  for r in opts.itertuples()]
        pick = st.selectbox("Cell (sorted worst-fragmentation first)", labels)
        key = str(opts.iloc[labels.index(pick)].strict_cell_key)

        cols = st.columns(len(shown))
        ones = {}
        for col, stg in zip(cols, shown):
            kp, _, cp = surfaces(stg)
            one = kp[kp.strict_cell_key.astype(str).eq(key)]
            row = cp[cp.strict_cell_key.astype(str).eq(key)]
            pieces = int(row.components.iloc[0]) if len(row) else 0
            with col:
                png = static_png(one.rename(columns={"value": "v"}), "v", f"{key}\n{stg}")
                if png:
                    st.image(png, width=460)
                st.metric("Pixels above floor", f"{len(one):,}")
                st.metric("Connected pieces", pieces)
                st.metric("Median RSRP", f"{one.value.median():.1f} dBm" if len(one) else "-")
                st.metric("Max distance", f"{one.distance_m.max():.0f} m" if len(one) else "-")
            ones[stg] = one
        if len(shown) == 2:
            a, b = ones["Uncalibrated (physical)"], ones["Calibrated (final)"]
            d = st.columns(2)
            d[0].metric("Pixels gained by calibration", f"{len(b) - len(a):+,}")
            if len(a) and len(b):
                d[1].metric("Median shift", f"{b.value.median() - a.value.median():+.1f} dB")
        st.caption("Every pixel shown carries this cell's own computed value. No pixel was added by "
                   "backfill or removed because another cell was stronger.")

        last = ones[shown[-1]]
        if len(last):
            st.markdown(f"**Directional check ({shown[-1]}) - median RSRP by |azimuth delta|**")
            bucket = pd.cut(last.azimuth_delta_deg.abs(), [0, 30, 60, 90, 120, 150, 180])
            st.dataframe(last.groupby(bucket, observed=True)
                         .agg(pixels=("value", "size"), median_rsrp=("value", "median"),
                              median_antenna_gain_dbi=("antenna_gain_dbi", "median")).round(1),
                         use_container_width=True)

# ------------------------------------------------------------------ cdf tab
with tab_cdf:
    st.subheader(f"{tech} - measured vs predicted (after calibration), and outdoor vs indoor")
    st.caption("Every predicted curve here is the FINAL calibrated value. Calibration is fitted only "
               "on train grids, outdoor, |az| <= 135 deg, band-matched by the Phase 38 EARFCN rule.")

    branch = serving_t.get("obstruction_branch", pd.Series("", index=serving_t.index)).astype(str)
    outdoor = serving_t.loc[~branch.eq("indoor"), "phase48_rsrp"]
    indoor = serving_t.loc[branch.eq("indoor"), "phase48_rsrp"]
    dt = data["dt"]
    dt_t = dt[dt.dt_tech.astype(str).eq(tech)]
    bands = sorted(dt_t.dt_band.astype(str).unique())
    band_pick = st.selectbox("Band (drive test is compared within its own band only)", ["All"] + bands)
    if band_pick != "All":
        dt_t = dt_t[dt_t.dt_band.astype(str).eq(band_pick)]
    fit_only = st.checkbox("Phase 39 fit subset only (outdoor, non-back-lobe)", value=True)
    if fit_only and "phase39_fit_row" in dt_t:
        dt_t = dt_t[dt_t.phase39_fit_row.astype(bool)]

    st.markdown("**Measured vs predicted at the drive-test points**")
    fig1 = go.Figure()
    for tr in [cdf_trace(dt_t.rsrp_measured.to_numpy(), "Measured DT", dash="dash"),
               cdf_trace(dt_t.predicted_rsrp.to_numpy(), "Predicted DT (calibrated)")]:
        if tr is not None:
            fig1.add_trace(tr)
    du = data["dt_uncal"]
    if du is not None:
        duf = du[du.dt_tech.astype(str).eq(tech)]
        if band_pick != "All":
            duf = duf[duf.dt_band.astype(str).eq(band_pick)]
        if fit_only and "phase39_fit_row" in duf:
            duf = duf[duf.phase39_fit_row.astype(bool)]
        tr = cdf_trace(duf.predicted_rsrp.to_numpy(), "Predicted DT (uncalibrated)", dash="dot")
        if tr is not None:
            fig1.add_trace(tr)
    fig1.update_layout(xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %", height=430,
                       legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig1, use_container_width=True)

    m = st.columns(4)
    if len(dt_t):
        for i, (lbl, sub) in enumerate([("all", dt_t),
                                        ("train", dt_t[dt_t.split.eq("train")] if "split" in dt_t else dt_t.iloc[:0]),
                                        ("test (held out)", dt_t[dt_t.split.eq("test")] if "split" in dt_t else dt_t.iloc[:0])]):
            if len(sub):
                m[i].metric(f"Bias {lbl}", f"{sub.error_db.median():+.2f} dB",
                            help=f"MAE {sub.error_db.abs().mean():.2f} dB over {len(sub):,} points")
        m[3].metric("Matched points", f"{len(dt_t):,}")

    st.markdown("---")
    st.markdown("**Outdoor vs indoor predicted coverage (whole serving surface)**")
    fig2 = go.Figure()
    for tr in [cdf_trace(outdoor.to_numpy(), "Predicted outdoor"),
               cdf_trace(indoor.to_numpy(), "Predicted indoor")]:
        if tr is not None:
            fig2.add_trace(tr)
    fig2.update_layout(xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %", height=430,
                       legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig2, use_container_width=True)

    k = st.columns(3)
    if len(outdoor) and len(indoor):
        k[0].metric("Outdoor median", f"{outdoor.median():.1f} dBm")
        k[1].metric("Indoor median", f"{indoor.median():.1f} dBm")
        k[2].metric("Outdoor - indoor (O2I)", f"{outdoor.median() - indoor.median():.1f} dB",
                    help="Expected O2I separation is roughly 10-15 dB. A negative value means "
                         "indoor is predicted stronger than outdoor, which is wrong.")
    else:
        k[0].info("No indoor rows - run without --no-obstruction to classify the indoor branch.")

    cal = summary.get("calibration_level1_db") or {}
    if cal:
        st.markdown("**Fitted calibration (dB added per technology/band)**")
        st.caption("This number is the evidence. A small residual means the physical model was already "
                   "close; a large one means calibration is hiding a level error.")
        st.dataframe(pd.Series(cal, name="correction_dB").to_frame(), use_container_width=True)

    st.info("All usable drive test is outdoor. The 2,754 rows labelled 'Indoor' in the source are a "
            "GPS-fix flag, not an indoor survey - they measure 3 dB STRONGER than outdoor, so they are "
            "not real O2I. The indoor curve above is therefore a predicted curve, not a measured one.")
    if summary.get("dt_rows_excluded_no_cells"):
        st.warning(f"{summary['dt_rows_excluded_no_cells']:,} drive-test rows excluded: bands B1/B7/B8 "
                   "have no cell in this project, so they cannot be matched.")

# ------------------------------------------------------------------ radius tab
with tab_radius:
    st.subheader("Per-cell link budget and solved cell-edge distance")
    st.caption("radius = COST-231 inverted for PL_max = (TX + boresight gain - cable + per-RE + anchor offset) "
               f"- {summary['edge_dbm']} dBm")
    st.dataframe(rad.groupby(["technology", "band"]).agg(
        cells=("strict_cell_key", "size"),
        tx_dbm=("tx_dbm", "first"),
        boresight_dbi=("boresight_dbi", "first"),
        per_re_db=("per_re_db", "first"),
        anchor_offset_db=("anchor_offset_db", "first"),
        deployed_mhz=("deployed_mhz", "first"),
        anchor_mhz=("anchor_mhz", "first"),
        eirp_re_dbm=("eirp_re_boresight_dbm", "median"),
        radius_m_median=("link_budget_radius_m", "median"),
        radius_m_min=("link_budget_radius_m", "min"),
        radius_m_max=("link_budget_radius_m", "max"),
    ).round(1), use_container_width=True)

    st.markdown("**PAP pattern file actually selected**")
    st.dataframe(rad.groupby(["technology", "band", "pap_file"]).size().rename("cells").reset_index(),
                 use_container_width=True, hide_index=True)
    st.dataframe(rad, use_container_width=True, height=320, hide_index=True)

# ------------------------------------------------------------------ rows tab
with tab_rows:
    st.subheader("Component count per cell")
    st.dataframe(comp.sort_values("components", ascending=False),
                 use_container_width=True, height=320, hide_index=True)
    st.subheader("Serving rows")
    show = [c for c in ["grid_id", "lat", "lon", "strict_cell_key", "band", "distance_m",
                        "azimuth_delta_deg", "antenna_gain_dbi", "pathloss_db", "per_re_db",
                        "anchor_offset_db", "building_obstruction_loss_db",
                        "terrain_diffraction_loss_db", "obstruction_branch", "clutter_class",
                        "phase48_rsrp"] if c in serving_t.columns]
    st.dataframe(serving_t[show], use_container_width=True, height=380, hide_index=True)
    st.caption(f"Source: {data['dir']}")
