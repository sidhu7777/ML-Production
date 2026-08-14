"""
Project-wide (all real sites/sectors) baseline dashboard page.

Streamlit multipage convention: any file under tests/baseline/pages/
automatically becomes its own page in the sidebar nav, alongside the main
dashboard_streamlit.py page - this file does not modify dashboard_streamlit.py
or any of its content at all.

Reads tests/baseline/output/project_wide_trace.json (produced by
run_project_wide_trace.py) plus the CDF PNGs in
tests/baseline/output/cdf_graphs/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEFAULT_TRACE_PATH = Path(__file__).resolve().parents[1] / "output" / "project_wide_trace.json"
CDF_DIR = Path(__file__).resolve().parents[1] / "output" / "cdf_graphs"

st.set_page_config(page_title="Project-Wide Baseline", layout="wide", page_icon="🗺️")

STAGE_META = {
    "stage1_raw_rsrp": {"label": "COST-231 raw", "color": "#2a78d6"},
    "stage2_geo_rsrp": {"label": "Geo-corrected (current, hand-tuned)", "color": "#eb6834"},
    "stage2b_spm_rsrp": {"label": "Geo-corrected (SPM-style, test-only)", "color": "#a349a4"},
    "stage3_rewired_rsrp": {"label": "DT-calibrated (rewired, on Stage 2b)", "color": "#2fb8c4"},
    "pred_rsrp_calibrated": {"label": "DT-calibrated (production, on Stage 2)", "color": "#1baf7a"},
}


@st.cache_data
def load_trace(path_str: str, _mtime: float) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_trace_from_path(path_str: str) -> dict:
    path = Path(path_str)
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return load_trace(path_str, mtime)


st.markdown(
    "<div style='font-size:11px;letter-spacing:0.08em;text-transform:uppercase;"
    "color:#eb6834;font-family:monospace;'>RF pipeline trace · whole project, all real sites</div>",
    unsafe_allow_html=True,
)
st.title("Project-Wide Baseline — Combined Coverage")
st.caption(
    "Every real site/sector in the project, run through the same Stage 1-3(+rewired) pipeline as the "
    "single-site trace - not scoped to one site. This is what makes the MAE/RMSE numbers below trustworthy: "
    "real drive-test points now nearest-match their OWN real serving site's grid, and every clutter class "
    "gets a real, adequately-sized sample, instead of being forced onto one unrelated site."
)

st.sidebar.header("Project-wide trace")
trace_path_str = st.sidebar.text_input("Trace JSON path", value=str(DEFAULT_TRACE_PATH))
data = load_trace_from_path(trace_path_str)

if not data or not data.get("sites_summary"):
    st.warning(
        "No project-wide trace data loaded yet. Run:\n\n"
        "`python tests/baseline/run_project_wide_trace.py --project-id 210 --region taiwan "
        "--data-dir tests/baseline/data/project_210_taiwan`"
    )
    st.stop()

sites_summary = data["sites_summary"]
map_points = pd.DataFrame(data.get("map_points", []))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sites", f"{len(sites_summary):,}")
n_sectors = sum(len(s["sectors"]) for s in sites_summary.values())
c2.metric("Sectors/cells", f"{n_sectors:,}")
c3.metric("Radius per site", f"{data.get('radius_m', '?'):.0f} m" if data.get("radius_m") else "?")
c4.metric("Map points (downsampled)", f"{len(map_points):,}")

st.caption(
    f"Radius is the same generous, non-cutoff distance used in the single-site trace (not production's "
    f"500m default) - real coverage past any one site is still real physics computed out to {data.get('radius_m','?')}m, "
    "the same 'compute far, derive boundary from threshold' principle, just applied to every site."
)

# ---- Real MAE/RMSE validation, project-wide - the trustworthy version of
# the single-site numbers, now with real per-site DT matching and real
# per-class sample sizes. ----
st.subheader("Project-wide real MAE/RMSE — every stage, same held-out DT points")
sv = data.get("stage3_validation", {})
if sv:
    rows = [
        {"stage": "Stage 1 (raw physics)", "mae": sv.get("mae_stage1_raw_physics"), "rmse": sv.get("rmse_stage1_raw_physics")},
        {"stage": "Stage 2 (current, hand-tuned)", "mae": sv.get("mae_stage2_current_hand_tuned"), "rmse": sv.get("rmse_stage2_current_hand_tuned")},
        {"stage": "Stage 2b (SPM-style)", "mae": sv.get("mae_stage2b_spm_style"), "rmse": sv.get("rmse_stage2b_spm_style")},
        {"stage": "Stage 3 (production, on Stage 2)", "mae": sv.get("mae_stage3_old_calibrated_on_stage2"), "rmse": sv.get("rmse_stage3_old_calibrated_on_stage2")},
        {"stage": "Stage 3-rewired (on Stage 2b)", "mae": sv.get("mae_stage3_rewired_calibrated_on_stage2b"), "rmse": sv.get("rmse_stage3_rewired_calibrated_on_stage2b")},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Held-out DT points matched project-wide: {sv.get('holdout_dt_rows_matched', 0):,}. Lower is better.")

with st.expander("MAE/RMSE by real measured RSRP signal-strength bin (project-wide)"):
    by_bin = data.get("stage3_validation_by_rsrp_bin", {})
    if by_bin:
        bin_labels = ["-140 to -105", "-105 to -95", "-95 to -85", "-85 to -75", "-75 to -44"]
        for stage_key, stage_label in [
            ("stage2b_spm_style", "Stage 2b (SPM-style)"),
            ("stage3_rewired_calibrated_on_stage2b", "Stage 3-rewired (on Stage 2b)"),
        ]:
            st.markdown(f"**{stage_label}**")
            bins = by_bin.get(stage_key, {})
            bin_rows = [{"range": lbl, **bins.get(lbl, {"n": 0, "mae": None, "rmse": None})} for lbl in bin_labels]
            st.dataframe(pd.DataFrame(bin_rows), use_container_width=True, hide_index=True)

with st.expander("Per-clutter-class fitted offsets (project-wide real DT data)"):
    offsets_dbg = data.get("stage2b_spm_offsets", {})
    per_class = offsets_dbg.get("per_class", {})
    if per_class:
        offsets_df = pd.DataFrame([
            {"clutter_class": cls, "fitted_offset_db": v.get("fitted_offset_db"), "n_dt_points": v.get("n"), "note": v.get("note", "")}
            for cls, v in per_class.items()
        ])
        st.dataframe(offsets_df, use_container_width=True, hide_index=True)
    st.caption(
        f"global_mean_residual_db={offsets_dbg.get('global_mean_residual_db')} · "
        f"matched_dt_rows_used={offsets_dbg.get('matched_dt_rows_used')} · "
        "compare per-class `n` here against the single-site trace's - project-wide sample sizes should be "
        "far more statistically adequate per class."
    )

st.subheader("Clutter distribution — whole project (corrected classification)")
clutter_dist = data.get("clutter_distribution_corrected", {})
if clutter_dist:
    st.bar_chart(pd.Series(clutter_dist))

# ---- Combined map: every site/sector's downsampled points, colored by
# whichever stage is selected. Real site markers shown too. ----
st.subheader("Combined coverage map — all real sites")
stage_col = st.selectbox(
    "Color by stage", list(STAGE_META.keys()), index=2,
    format_func=lambda k: STAGE_META[k]["label"],
)
if not map_points.empty:
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermapbox(
        lat=map_points["lat"], lon=map_points["lon"], mode="markers",
        marker=dict(size=6, color=map_points[stage_col], colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title="dBm")),
        text=map_points["site_id"] + " / " + map_points["sector"] + " / " + map_points["clutter_class"].astype(str),
        hovertemplate="%{text}<br>" + STAGE_META[stage_col]["label"] + ": %{marker.color:.1f} dBm<extra></extra>",
        name="Prediction points (downsampled)",
    ))
    site_lats = [s["lat"] for s in sites_summary.values()]
    site_lons = [s["lon"] for s in sites_summary.values()]
    fig_map.add_trace(go.Scattermapbox(
        lat=site_lats, lon=site_lons, mode="markers",
        marker=dict(size=10, color="black", symbol="circle"),
        text=list(sites_summary.keys()), hovertemplate="Site %{text}<extra></extra>",
        name="Real sites",
    ))
    center_lat = float(np.mean(site_lats))
    center_lon = float(np.mean(site_lons))
    fig_map.update_layout(
        height=650, mapbox=dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=12),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(
        f"{len(map_points):,} downsampled points across {len(sites_summary)} real sites, colored by "
        f"{STAGE_META[stage_col]['label']} on the real production RdYlGn scale. Points are evenly "
        "downsampled per sector (not every point) purely to keep this combined view responsive - the "
        "MAE/RMSE numbers above use the full, non-downsampled data."
    )
else:
    st.info("No map points in this trace file.")

# ---- CDF images (generated separately by generate_cdf_graphs.py) ----
st.subheader("CDF comparison — drive test vs raw baseline (whole project)")
cdf_files = [
    ("cdf_1_drive_test_measured_rsrp.png", "1. Drive-test measured RSRP (real, whole project)"),
    ("cdf_2_raw_baseline_full_grid_rsrp.png", "2. Raw baseline (Stage 1) predicted RSRP - full grid, everywhere"),
    ("cdf_3_raw_baseline_at_drive_test_points_rsrp.png", "3. Raw baseline predicted RSRP - AT drive-test point locations only"),
    ("cdf_4_combined.png", "4. Combined - all three overlaid"),
]
missing = [f for f, _ in cdf_files if not (CDF_DIR / f).exists()]
if missing:
    st.info(
        f"CDF images not generated yet (missing: {', '.join(missing)}). Run:\n\n"
        "`python tests/baseline/generate_cdf_graphs.py`\n\n"
        f"after the project-wide trace, images will be written to `{CDF_DIR}`."
    )
else:
    cdf_cols = st.columns(2)
    for i, (fname, caption) in enumerate(cdf_files):
        with cdf_cols[i % 2]:
            st.image(str(CDF_DIR / fname), caption=caption, use_container_width=True)
