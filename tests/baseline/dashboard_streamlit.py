"""
Streamlit dashboard for the site -> sector pipeline trace produced by
test_single_site_pipeline.py.

Run:
    cd ML
    venv\\Scripts\\activate
    streamlit run tests/baseline/dashboard_streamlit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEFAULT_TRACE_PATH = Path(__file__).parent / "output" / "site_sector_trace.json"

STAGE_META = {
    "s1": {"full": "stage1_raw_rsrp", "label": "COST-231 raw", "color": "#2a78d6",
           "note": "Pure physics path-loss output. No buildings, no terrain, no calibration."},
    "s2": {"full": "stage2_geo_rsrp", "label": "Geo-corrected", "color": "#eb6834",
           "note": "Physics + clutter/morphology/terrain penalty. Captured pre-calibration - "
                   "production overwrites this value in place before it's ever saved."},
    "s3": {"full": "pred_rsrp_calibrated", "label": "DT-calibrated", "color": "#1baf7a",
           "note": "Geo value corrected by a drive-test-trained residual model. "
                   "This is what the baseline table's main pred_rsrp column actually holds."},
    "s4": {"full": "pred_rsrp_demo", "label": "Smoothed", "color": "#eda100",
           "note": "Calibrated value blended toward nearby real drive-test measurements."},
}
STAGE_KEYS = list(STAGE_META.keys())
BEAMWIDTH = 65.0

st.set_page_config(page_title="Site x Sector Pipeline Trace", layout="wide", page_icon="📡")


@st.cache_data
def load_trace(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_new_trace(project_id: int, region: str, polygon_ids: str, output_path: Path):
    from tests.baseline.test_single_site_pipeline import run_trace
    import os
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    db_url = os.getenv("DATABASE_URL_Taiwan") if region == "taiwan" else os.getenv("DATABASE_URL")
    eng = create_engine(db_url)
    with eng.connect() as conn:
        ref = conn.execute(text("SELECT ref_session_id FROM tbl_project WHERE id=:pid"), {"pid": project_id}).scalar()
    session_ids = [int(s.strip()) for s in (ref or "").split(",") if s.strip()]

    with st.spinner(f"Running COST-231 -> geo -> calibration -> smoothing for ALL sectors of project {project_id}... this takes a couple of minutes."):
        run_trace(
            project_id=project_id, region=region, session_ids=session_ids, operator=None,
            radius_m=500.0, grid_resolution_m=25.0, polygon_ids=polygon_ids or None, output_path=output_path,
        )
    st.cache_data.clear()


# ---- Sidebar ----
st.sidebar.header("RF pipeline trace")
trace_path_str = st.sidebar.text_input("Trace JSON path", value=str(DEFAULT_TRACE_PATH))
data = load_trace(trace_path_str)

with st.sidebar.expander("Run a new trace", expanded=(not data)):
    project_id = st.number_input("Project ID", value=210, step=1)
    region = st.selectbox("Region", ["taiwan", "india"], index=0)
    polygon_ids = st.text_input("Polygon IDs (optional, comma-separated)", value="1883")
    if st.button("Run trace now", type="primary"):
        run_new_trace(int(project_id), region, polygon_ids, Path(trace_path_str))
        st.rerun()

if not data or not data.get("sites"):
    st.title("Site x Sector Pipeline Trace")
    st.warning("No trace data loaded yet. Use the sidebar to run a new trace.")
    st.stop()

sites = data["sites"]
site_ids = sorted(sites.keys(), key=lambda sid: -len(sites[sid]["sectors"]))

st.sidebar.markdown("---")
st.sidebar.subheader("1. Site")
site_id = st.sidebar.selectbox(
    "Physical site",
    site_ids,
    format_func=lambda sid: f"{sid}  ({len(sites[sid]['sectors'])} sector{'s' if len(sites[sid]['sectors']) != 1 else ''})",
)
site = sites[site_id]

st.sidebar.subheader("2. Sector (cell + band)")
sector_keys = sorted(site["sectors"].keys())
sector_key = st.sidebar.radio(
    "Sector",
    sector_keys,
    format_func=lambda sk: f"{sk} — azimuth {site['sectors'][sk]['azimuth']:.0f}° — band {site['sectors'][sk]['band']}",
    label_visibility="collapsed",
)
sector = site["sectors"][sector_key]

st.sidebar.subheader("3. Pipeline stage")
stage_key = st.sidebar.radio(
    "Stage", STAGE_KEYS, format_func=lambda k: STAGE_META[k]["label"], index=1, label_visibility="collapsed",
)
st.sidebar.caption(STAGE_META[stage_key]["note"])

st.sidebar.markdown("---")
st.sidebar.text(f"Site: {site_id}")
st.sidebar.text(f"Cell ID: {sector['cell_id']}")
st.sidebar.text(f"Azimuth: {sector['azimuth']:.0f}°  (±{BEAMWIDTH:.0f}° 3dB beam)")
st.sidebar.text(f"Band: {sector['band']}")
st.sidebar.text(f"Lat/Lon: {site['lat']:.6f}, {site['lon']:.6f}")
st.sidebar.text(f"Grid points (full circle): {sector['point_count']:,}")

st.sidebar.markdown("---")
st.sidebar.subheader("Clutter mix (this sector)")
for cls, count in sector.get("clutter_distribution", {}).items():
    st.sidebar.text(f"{cls}: {count}")


# ---- Main ----
st.markdown(
    "<div style='font-size:11px;letter-spacing:0.08em;text-transform:uppercase;"
    "color:#eb6834;font-family:monospace;'>RF pipeline trace · per site + sector + cell + band</div>",
    unsafe_allow_html=True,
)
st.title(f"Site {site_id} — Sector {sector_key} ({sector['cell_id']}, band {sector['band']})")
st.caption(
    f"Azimuth {sector['azimuth']:.0f}° · {sector['point_count']:,} points in the site's full 500m circular grid "
    f"(COST-231 generates the whole circle for every sector — direction comes entirely from antenna gain, "
    f"not from restricting which points get computed)"
)

st.info(
    "**Every point in the circle has a real value — direction only changes how strong it is, not whether "
    "it exists.** COST-231 computes RSRP everywhere in the 500m circle; the antenna gain formula "
    "(`_antenna_gain_estimate`, 65° half-power beamwidth, continuous — not a cutoff) weakens off-axis points "
    "gradually. The azimuth chart below shows this directly: signal peaks on-axis and tapers off, but never "
    "drops out. The map colors every point on the same continuous scale production uses, so it reads as a "
    "gradient, not a wedge."
)

meta = STAGE_META[stage_key]
stage_summary_key = {"s1": "stage1_raw_rsrp", "s2": "stage2_geo_rsrp", "s3": "stage3_calibrated_rsrp", "s4": "stage4_smoothed_rsrp"}[stage_key]
st_stats = sector["stage_summary"][stage_summary_key]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Min", f"{st_stats['min']:.1f} dBm")
c2.metric("Max", f"{st_stats['max']:.1f} dBm")
c3.metric("Mean", f"{st_stats['mean']:.1f} dBm")
c4.metric("Std dev", f"{st_stats['std']:.2f} dB")

all_points = pd.DataFrame(sector["all_points"])
radial = pd.DataFrame(sector["radial_profile"])

# Both all_points and radial_profile carry the real production column names
# (stage1_raw_rsrp, stage2_geo_rsrp, pred_rsrp_calibrated, pred_rsrp_demo).
stage_col_full = STAGE_META[stage_key]["full"]

# ---- Map: every point colored by its real value, no masking/fading ----
# An earlier version faded points past a hard 65deg cutoff and drew a wedge
# outline. That was a display artifact, not the data - the antenna gain
# formula is continuous, every point in the circle has a real computed
# value, and there is no cutoff. This just colors every point honestly.
st.subheader(f"Coverage map — colored by {meta['label']}")
azimuth = sector["azimuth"]
m_per_deg_lat = 111320.0
m_per_deg_lon = 111320.0 * np.cos(np.radians(site["lat"]))
plot_df = all_points.copy()
plot_df["x_m"] = (plot_df["lon"] - site["lon"]) * m_per_deg_lon
plot_df["y_m"] = (plot_df["lat"] - site["lat"]) * m_per_deg_lat
plot_df = plot_df.dropna(subset=[stage_col_full])

fig_map = go.Figure()
fig_map.add_trace(go.Scatter(
    x=plot_df["x_m"], y=plot_df["y_m"], mode="markers",
    marker=dict(size=7, color=plot_df[stage_col_full], colorscale="RdYlGn",
                colorbar=dict(title="dBm"), line=dict(width=0)),
    text=plot_df["clutter_class"],
    hovertemplate="%{text}<br>" + meta["label"] + ": %{marker.color:.1f} dBm<extra></extra>",
    showlegend=False,
))
fig_map.add_trace(go.Scatter(
    x=[0], y=[0], mode="markers", marker=dict(symbol="diamond", size=14, color="black"),
    name="Site", hovertemplate="Site<extra></extra>",
))
fig_map.update_layout(
    height=560, xaxis_title="meters east of site", yaxis_title="meters north of site",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig_map, use_container_width=True)
st.caption(
    "Every point in the full 500m circle, colored by its real computed value on a continuous scale - "
    "same as production. Signal is strongest on-axis and weakens gradually with distance and off-axis "
    "angle, but never disappears or gets masked; there's no hard directional cutoff in the actual data."
)

# ---- Azimuth vs RSRP: proves/disproves the directional physics ----
st.subheader("RSRP vs. off-axis angle — is the antenna pattern actually directional?")
fig_az = go.Figure()
for key in STAGE_KEYS:
    m = STAGE_META[key]
    d = all_points.dropna(subset=[m["full"]])
    fig_az.add_trace(go.Scatter(
        x=d["off_axis_deg"], y=d[m["full"]], mode="markers", marker=dict(size=3, color=m["color"], opacity=0.35),
        name=m["label"], visible=True if key == stage_key else "legendonly",
    ))
fig_az.add_vline(x=BEAMWIDTH, line_dash="dot", line_color="#898781", annotation_text="3dB beam edge")
fig_az.update_layout(
    height=360, xaxis_title="degrees off boresight (0 = straight ahead, 180 = directly behind)",
    yaxis_title="RSRP (dBm)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig_az, use_container_width=True)
st.caption(
    "Every point in the full 500m circle, by angle off boresight. A working directional model shows RSRP "
    "clearly falling as this angle grows. Toggle stages via the legend."
)

# ---- Radial profile: on-axis only ----
st.subheader("Radial profile — RSRP vs. distance from site (on-axis points only, all 4 stages)")
fig_line = go.Figure()
for key in STAGE_KEYS:
    m = STAGE_META[key]
    fig_line.add_trace(go.Scatter(
        x=radial["distance_m"], y=radial[m["full"]], mode="markers",
        marker=dict(color=m["color"], size=3, opacity=0.5 if key != stage_key else 0.85),
        name=m["label"],
    ))
fig_line.update_layout(
    height=420, xaxis_title="distance from site (m)", yaxis_title="RSRP (dBm)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
)
st.plotly_chart(fig_line, use_container_width=True)
st.caption(f"{len(radial):,} points within the {BEAMWIDTH:.0f}° 3dB beam, sorted by distance.")

# ---- Table ----
st.subheader("Point-by-point table — on-axis points")
table_cols = {
    "distance_m": "Dist (m)", "off_axis_deg": "Off-axis (deg)", "spacing_from_prev_m": "Spacing (m)",
    "clutter_class": "Clutter", "stage1_raw_rsrp": "Raw", "stage2_geo_rsrp": "Geo",
    "pred_rsrp_calibrated": "Calibrated", "pred_rsrp_demo": "Smoothed",
    "delta_stage2_geo_rsrp": "DeltaGeo", "delta_pred_rsrp_calibrated": "DeltaCalibrated",
}
table_df = radial[[c for c in table_cols if c in radial.columns]].rename(columns=table_cols)


def highlight_big_delta(val):
    if pd.isna(val):
        return ""
    return "font-weight: 700; color: #eb6834;" if abs(val) >= 3 else ""


delta_cols = [c for c in ["DeltaGeo", "DeltaCalibrated"] if c in table_df.columns]
styled = table_df.style.map(highlight_big_delta, subset=delta_cols) \
    .format({c: "{:.1f}" for c in table_df.columns if c != "Clutter"}, na_rep="—")
st.dataframe(styled, use_container_width=True, height=420)
st.caption("Delta columns = change from the previous point in this list. Rows with |Delta| >= 3dB are bolded.")

with st.expander("What each stage actually is (traced from the real code)"):
    st.markdown(
        """
**1. COST-231 raw** — pure physics path-loss model output (`run_rf_prediction_fast`). No geo context at all.

**2. Geo-corrected** — raw + a weighted offset from clutter class, morphology cluster, terrain, LOS blockers,
interference (`apply_experimental_geo_adjustments`). This value is normally overwritten in-place by DT
calibration before it's ever saved — this dashboard captures it before that happens.

**3. DT-calibrated** — a Ridge regression trained on real drive-test residuals corrects the geo value
(`apply_dt_holdout_calibration` -> `preserve_calibrated_kpis`). This is what the baseline table's main
`pred_rsrp` column actually holds.

**4. Smoothed** — stage 3 blended toward real nearby drive-test measurements by distance
(`apply_demo_dt_overlay`). Saved as `pred_rsrp_smoothed`.

**Antenna gain formula** (`grid_sampling.py::_antenna_gain_estimate`): a 3GPP-style horizontal pattern,
`Ah = min(12*(az_diff/65)^2, 30)`, i.e. a 65° 3dB half-beamwidth with up to 30dB of off-axis attenuation.
This is applied per-point during COST-231 itself — the raw stage already carries the directional shape,
it just isn't masked in a plain scatter of all points.
        """
    )
