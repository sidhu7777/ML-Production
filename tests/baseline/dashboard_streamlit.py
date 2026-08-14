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
    "s2": {"full": "stage2_geo_rsrp", "label": "Geo-corrected (current, hand-tuned)", "color": "#eb6834",
           "note": "Physics + a stack of independently hand-picked, independently distance-"
                   "clipped feature weights (DEFAULT_GEO_WEIGHTS). Captured pre-calibration - "
                   "production overwrites this value in place before it's ever saved. This is "
                   "the stage that produces the flat -140dBm plateau from ~1-2km."},
    "s2b": {"full": "stage2b_spm_rsrp", "label": "Geo-corrected (SPM-style, test-only)", "color": "#a349a4",
            "note": "Physics + ONE calibrated per-clutter-class loss term, fit by least-squares "
                    "against real DT data (the real Atoll/Forsk Standard Propagation Model "
                    "structure: K_clutter*f(clutter)). No distance re-penalization - COST-231's "
                    "own slope term already models distance smoothly. Test-case only, not in "
                    "production."},
    "s2c": {"full": "stage2c_spm_rsrp", "label": "Geo-corrected (SPM + real diffraction, test-only)", "color": "#d4534a",
            "note": "Physics + REAL single-knife-edge diffraction loss (ITU-R-style, computed "
                    "directly from real per-path building crossings + GHS-OBAT-imputed heights - "
                    "NOT DT-fit) + a DT-calibrated per-clutter-class term fit AFTER diffraction is "
                    "subtracted, so clutter only explains what real diffraction physics doesn't "
                    "already account for. Fixes Stage 2b's biggest flaw: a single DT-fit constant "
                    "per class was pulling every point toward the class average regardless of real "
                    "local building obstruction. Test-case only, not in production."},
    "s3": {"full": "pred_rsrp_calibrated", "label": "DT-calibrated (production, on Stage 2)", "color": "#1baf7a",
           "note": "Geo value corrected by a drive-test-trained residual model. "
                   "This is what the baseline table's main pred_rsrp column actually holds. "
                   "Calibrates against Stage 2 (hand-tuned) - unchanged, real production behavior."},
    "s3r": {"full": "stage3_rewired_rsrp", "label": "DT-calibrated (rewired, on Stage 2b)", "color": "#2fb8c4",
            "note": "The same kind of DT residual calibration as Stage 3, but correctly targeted: "
                    "fit against Stage 2b's (SPM-style) output instead of Stage 2's, and with "
                    "clutter_class removed from its own features (Stage 2b already owns that "
                    "signal - keeping it in both models risked double-correcting the same thing). "
                    "This is the real published 'delta correction on an already-calibrated "
                    "baseline' pattern. Test-case only."},
    "s4": {"full": "pred_rsrp_demo", "label": "Smoothed", "color": "#eda100",
           "note": "Calibrated value blended toward nearby real drive-test measurements."},
}
STAGE_KEYS = list(STAGE_META.keys())
BEAMWIDTH = 65.0

st.set_page_config(page_title="Site x Sector Pipeline Trace", layout="wide", page_icon="📡")


@st.cache_data
def load_trace(path_str: str, _mtime: float) -> dict:
    # _mtime busts the cache whenever the file on disk actually changes -
    # without it, st.cache_data only keys on path_str, so re-running a trace
    # to the SAME path (as this dashboard always does) silently kept serving
    # the old cached content even after the file was overwritten.
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_trace_from_path(path_str: str) -> dict:
    path = Path(path_str)
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return load_trace(path_str, mtime)


TRACE_RADIUS_M = 2500.0  # generous max extent - real coverage boundary is derived below from
# where RSRP actually crosses the threshold, not from this request radius. This just needs
# to be large enough that every sector's real signal has already decayed past any realistic
# threshold well before reaching it; it is not itself "the coverage radius" anywhere in this
# dashboard.


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
            radius_m=TRACE_RADIUS_M, grid_resolution_m=25.0, polygon_ids=polygon_ids or None, output_path=output_path,
        )
    st.cache_data.clear()


# ---- Sidebar ----
st.sidebar.header("RF pipeline trace")
trace_path_str = st.sidebar.text_input("Trace JSON path", value=str(DEFAULT_TRACE_PATH))
data = load_trace_from_path(trace_path_str)

st.sidebar.subheader("Compare against another trace")
compare_enabled = st.sidebar.checkbox("Show before/after comparison", value=False)
compare_path_str = ""
compare_data = {}
if compare_enabled:
    compare_path_str = st.sidebar.text_input(
        "Comparison trace JSON path",
        value=str(Path(__file__).parent / "output" / "site_sector_trace_BEFORE_500m_GA20000099.json"),
        help="e.g. the fixed-500m trace saved before the dynamic-radius fix",
    )
    compare_data = load_trace_from_path(compare_path_str)
    if not compare_data or not compare_data.get("sites"):
        st.sidebar.warning("No data at that comparison path.")

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

st.sidebar.subheader("4. Coverage threshold")
rsrp_threshold = st.sidebar.slider(
    "RSRP service threshold (dBm)", min_value=-140, max_value=-44, value=-128, step=1,
    help="Coverage extent below is DERIVED from where the real predicted RSRP crosses this "
         "value along each direction - it is not a fixed radius.",
)

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
    f"Azimuth {sector['azimuth']:.0f}° · {sector['point_count']:,} points computed out to a generous "
    f"{TRACE_RADIUS_M:.0f}m request extent (COST-231 generates every point in that circle — direction comes "
    f"entirely from antenna gain, not from restricting which points get computed). The actual coverage "
    f"extent shown below is derived from where RSRP crosses your chosen threshold, not from this request radius."
)

st.info(
    "**Every point out to the request extent has a real value — direction only changes how strong it is, not "
    "whether it exists, and distance is not capped at a fixed radius.** COST-231 computes RSRP everywhere out "
    "to a generous request extent; the antenna gain formula (`_antenna_gain_estimate`, real per-site azimuth/"
    "tilt, 65°/6° half-power beamwidth defaults where beamwidth data is missing, continuous — not a cutoff) "
    "weakens off-axis points gradually in both azimuth and elevation. The **coverage boundary itself is a "
    "threshold contour**: wherever real RSRP first drops below your selected threshold, moving outward — not "
    "an assumed radius. The map colors every point on the same continuous scale production uses, so it reads "
    "as a gradient, not a wedge."
)

meta = STAGE_META[stage_key]
stage_summary_key = {
    "s1": "stage1_raw_rsrp", "s2": "stage2_geo_rsrp", "s2b": "stage2b_spm_rsrp", "s2c": "stage2c_spm_rsrp",
    "s3": "stage3_calibrated_rsrp", "s3r": "stage3_rewired_rsrp", "s4": "stage4_smoothed_rsrp",
}[stage_key]
if stage_summary_key not in sector.get("stage_summary", {}):
    # Fails soft instead of an unhandled KeyError - this trace file (loaded
    # from trace_path_str) predates the selected stage (e.g. an older trace
    # saved before Stage 2b existed, or a stale cached read). st.cache_data
    # is keyed on file mtime (see load_trace above) so this should self-heal
    # on the next real file write + rerun; this guard just prevents a crash
    # if that hasn't happened yet.
    st.error(
        f"This trace file doesn't have '{meta['label']}' data for this sector "
        f"(missing stage_summary key '{stage_summary_key}'). The loaded trace at "
        f"`{trace_path_str}` likely predates this stage, or the page needs a hard "
        "refresh to pick up a just-regenerated file. Re-run the trace, then refresh "
        "this page (not just re-select a widget)."
    )
    st.stop()
st_stats = sector["stage_summary"][stage_summary_key]

all_points = pd.DataFrame(sector["all_points"])
radial = pd.DataFrame(sector["radial_profile"])

# ---- Dynamic coverage extent: the real distance (on-axis) where RSRP first
# drops below the chosen threshold, moving outward from the site. This is
# derived from real computed values every time the threshold changes - it is
# never a fixed/assumed radius. ----
def compute_coverage_extent(radial_df: pd.DataFrame, stage_col: str, threshold: float, request_extent_m: float):
    if stage_col not in radial_df.columns:
        # older comparison traces saved before Stage 2b existed won't have
        # stage2b_spm_rsrp - fail soft instead of a KeyError from dropna.
        return 0.0, f"'{stage_col}' not present in this trace file (older trace?)"
    radial_sorted = radial_df.sort_values("distance_m").dropna(subset=[stage_col])
    below = radial_sorted[radial_sorted[stage_col] < threshold]
    if not below.empty:
        return float(below["distance_m"].iloc[0]), f"real RSRP first drops below {threshold} dBm here"
    if not radial_sorted.empty:
        return float(radial_sorted["distance_m"].iloc[-1]), f"still above {threshold} dBm at the full {request_extent_m:.0f}m request extent - real boundary is farther out"
    return 0.0, "no on-axis points available"


stage_col_for_extent = STAGE_META[stage_key]["full"]
coverage_extent_m, extent_note = compute_coverage_extent(radial, stage_col_for_extent, rsrp_threshold, TRACE_RADIUS_M)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Min", f"{st_stats['min']:.1f} dBm")
c2.metric("Max", f"{st_stats['max']:.1f} dBm")
c3.metric("Mean", f"{st_stats['mean']:.1f} dBm")
c4.metric("Std dev", f"{st_stats['std']:.2f} dB")
c5.metric("Coverage extent (on-axis)", f"{coverage_extent_m:.0f} m", help=extent_note)
st.caption(f"Coverage extent: {extent_note}. This is computed fresh from real predicted values every time the threshold slider changes - not a stored or assumed radius.")

# ---- Real MAE validation: current Stage 2 (hand-tuned weights) vs new
# Stage 2b (SPM-style clutter term), measured on the SAME held-out real DT
# points neither stage was fit on. Not a claim - a measured comparison,
# computed once per trace run in test_single_site_pipeline.py. ----
stage2b_val = data.get("stage2b_validation")
if stage2b_val:
    st.subheader("Stage 2 vs Stage 2b — real MAE against held-out drive-test data")
    mv1, mv2, mv3, mv4 = st.columns(4)
    mv1.metric("Held-out DT points matched", f"{stage2b_val.get('holdout_dt_rows_matched', 0):,}")
    mae_s1 = stage2b_val.get("mae_stage1_raw_physics_only")
    mae_s2 = stage2b_val.get("mae_stage2_current_hand_tuned_weights")
    mae_s2b = stage2b_val.get("mae_stage2b_spm_style_clutter_only")
    mv2.metric("MAE — Stage 1 raw physics", f"{mae_s1:.2f} dB" if mae_s1 is not None else "—")
    mv3.metric("MAE — Stage 2 (current, hand-tuned)", f"{mae_s2:.2f} dB" if mae_s2 is not None else "—")
    delta = None if (mae_s2b is None or mae_s2 is None) else round(mae_s2b - mae_s2, 3)
    mv4.metric(
        "MAE — Stage 2b (SPM-style)", f"{mae_s2b:.2f} dB" if mae_s2b is not None else "—",
        delta=None if delta is None else f"{delta:+.2f} dB vs current Stage 2",
        delta_color="inverse",
    )
    st.caption(
        "Lower MAE is better. Both MAE values are measured on the SAME held-out real drive-test "
        "points, which neither Stage 2 (hand-tuned weights) nor Stage 2b (SPM-style clutter term) "
        "was fit on - this is a genuine comparison, not a display artifact. Note: this trace is "
        "scoped to a single site, so the number of real DT points close enough to fit/validate "
        "against is small - see `stage2b_spm_offsets` in the trace JSON for real per-class sample "
        "counts (`n`) and which classes fell back to the global mean for lack of local data."
    )
    with st.expander("Per-clutter-class fitted offsets (real DT data, this trace's train split)"):
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
            f"dropped_beyond_max_match_distance_m={offsets_dbg.get('dropped_beyond_max_match_distance_m')}"
        )

# ---- Real MAE validation: every stage, same held-out DT data - answers
# whether Stage 2b + the rewired Stage 3 (calibrated on Stage 2b's output,
# not Stage 2's) actually beats the current production stack. ----
stage3_val = data.get("stage3_validation")
if stage3_val:
    st.subheader("All stages — real MAE against held-out drive-test data")
    sv1, sv2, sv3 = st.columns(3)
    sv1.metric("Held-out DT points matched", f"{stage3_val.get('holdout_dt_rows_matched', 0):,}")
    mae_old3 = stage3_val.get("mae_stage3_old_calibrated_on_stage2")
    mae_new3 = stage3_val.get("mae_stage3_rewired_calibrated_on_stage2b")
    sv2.metric("MAE — Stage 3 (production, on Stage 2)", f"{mae_old3:.2f} dB" if mae_old3 is not None else "—")
    delta3 = None if (mae_new3 is None or mae_old3 is None) else round(mae_new3 - mae_old3, 3)
    sv3.metric(
        "MAE — Stage 3-rewired (on Stage 2b)", f"{mae_new3:.2f} dB" if mae_new3 is not None else "—",
        delta=None if delta3 is None else f"{delta3:+.2f} dB vs production Stage 3",
        delta_color="inverse",
    )
    full_mae_df = pd.DataFrame([
        {"stage": "Stage 1 (raw physics)", "mae_db": stage3_val.get("mae_stage1_raw_physics")},
        {"stage": "Stage 2 (current, hand-tuned)", "mae_db": stage3_val.get("mae_stage2_current_hand_tuned")},
        {"stage": "Stage 2b (SPM-style)", "mae_db": stage3_val.get("mae_stage2b_spm_style")},
        {"stage": "Stage 3 (production, on Stage 2)", "mae_db": mae_old3},
        {"stage": "Stage 3-rewired (on Stage 2b)", "mae_db": mae_new3},
    ])
    st.dataframe(full_mae_df, use_container_width=True, hide_index=True)
    st.caption(
        "Lower MAE is better. Every number here is measured on the same held-out real DT points, "
        "none of which were used to fit Stage 2b or Stage 3-rewired. Stage 3-rewired calibrates its "
        "residual against Stage 2b's output (not Stage 2's), and excludes clutter_class from its own "
        "features since Stage 2b already owns that signal - see the conversation for why calibrating "
        "against the wrong baseline, or double-using the same feature across two DT-fit stages, would "
        "both be real mistakes. Remember: for this site/band (n78), the held-out DT points include "
        "561 verified real n78 measurements plus a larger number of N/A-band rows relabeled to n78 "
        "and given an imputed pci/earfcn from their nearest real n78 point (explicit approximation, "
        "not verified ground truth) - see `stage2b_spm_offsets` / the local drive cache's "
        "`band_source` column to see which is which."
    )

# ---- Before/after comparison against a second trace file, same site+sector+stage ----
if compare_enabled and compare_data.get("sites"):
    st.subheader("Before / after comparison")
    cmp_site = compare_data["sites"].get(site_id)
    if cmp_site is None:
        st.warning(f"Site {site_id} is not present in the comparison trace ({compare_path_str}).")
    else:
        cmp_sector = cmp_site["sectors"].get(sector_key)
        if cmp_sector is None:
            st.warning(f"Sector {sector_key} is not present for site {site_id} in the comparison trace.")
        elif stage_summary_key not in cmp_sector.get("stage_summary", {}):
            # older comparison traces (saved before Stage 2b existed) don't
            # have this stage - fail soft with a clear message instead of a
            # KeyError, rather than blocking every other stage's comparison.
            st.info(f"'{STAGE_META[stage_key]['label']}' isn't present in the comparison trace "
                    f"({compare_path_str}) - it predates this stage. Pick another stage, or re-run "
                    "a trace to that path to get this stage in the comparison too.")
        else:
            cmp_radial = pd.DataFrame(cmp_sector["radial_profile"])
            cmp_extent_m, cmp_extent_note = compute_coverage_extent(cmp_radial, stage_col_for_extent, rsrp_threshold, TRACE_RADIUS_M)
            cmp_stats = cmp_sector["stage_summary"][stage_summary_key]

            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**Comparison trace** ({Path(compare_path_str).name})")
                st.metric("Points in this sector", f"{cmp_sector['point_count']:,}")
                st.metric("Coverage extent (on-axis)", f"{cmp_extent_m:.0f} m", help=cmp_extent_note)
                st.metric("Mean RSRP", f"{cmp_stats['mean']:.1f} dBm")
            with cols[1]:
                st.markdown(f"**Current trace** ({Path(trace_path_str).name})")
                delta_extent = coverage_extent_m - cmp_extent_m
                delta_mean = st_stats["mean"] - cmp_stats["mean"]
                st.metric("Points in this sector", f"{sector['point_count']:,}", delta=f"{sector['point_count'] - cmp_sector['point_count']:+,}")
                st.metric("Coverage extent (on-axis)", f"{coverage_extent_m:.0f} m", delta=f"{delta_extent:+.0f} m")
                st.metric("Mean RSRP", f"{st_stats['mean']:.1f} dBm", delta=f"{delta_mean:+.1f} dB")

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=cmp_radial.sort_values("distance_m")["distance_m"], y=cmp_radial.sort_values("distance_m")[stage_col_for_extent],
                mode="lines+markers", name="Comparison", line=dict(color="#999999"), marker=dict(size=3),
            ))
            fig_cmp.add_trace(go.Scatter(
                x=radial.sort_values("distance_m")["distance_m"], y=radial.sort_values("distance_m")[stage_col_for_extent],
                mode="lines+markers", name="Current", line=dict(color=meta["color"]), marker=dict(size=3),
            ))
            fig_cmp.add_hline(y=rsrp_threshold, line_dash="dot", line_color="#333333", annotation_text=f"{rsrp_threshold} dBm threshold")
            fig_cmp.update_layout(
                height=360, xaxis_title="distance from site (m)", yaxis_title=f"{meta['label']} (dBm)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.caption("Same site, sector, and pipeline stage in both traces - only the request radius / dynamic-extent logic differs.")
elif compare_enabled:
    st.info("Enter a valid comparison trace path in the sidebar to see a before/after view here.")

# Both all_points and radial_profile carry the real production column names
# (stage1_raw_rsrp, stage2_geo_rsrp, pred_rsrp_calibrated, pred_rsrp_demo).
stage_col_full = STAGE_META[stage_key]["full"]

# ---- Map: every point colored by its real value, no masking/fading ----
# An earlier version faded points past a hard 65deg cutoff and drew a wedge
# outline. That was a display artifact, not the data - the antenna gain
# formula is continuous, every point in the circle has a real computed
# value, and there is no cutoff. This just colors every point honestly.
#
# Below-threshold points are rendered as a faint, uncolored background
# instead of being painted onto the same continuous RdYlGn scale as the
# covered points. Without this split, a stage whose real values crowd
# against the -140dBm floor across most of the request extent (the current
# Stage 2 bug) reads as a "solid red disk" no matter what the true
# directional shape is, because the auto colorscale range gets dominated by
# the huge low-value mass. Splitting on the user's own threshold and giving
# color ONLY to points that are actually real coverage is what makes the
# true directional shape (or the current bug's lack of one) visible.
st.subheader(f"Coverage map — colored by {meta['label']}")
azimuth = sector["azimuth"]
m_per_deg_lat = 111320.0
m_per_deg_lon = 111320.0 * np.cos(np.radians(site["lat"]))
plot_df = all_points.copy()
plot_df["x_m"] = (plot_df["lon"] - site["lon"]) * m_per_deg_lon
plot_df["y_m"] = (plot_df["lat"] - site["lat"]) * m_per_deg_lat
plot_df = plot_df.dropna(subset=[stage_col_full])

covered_df = plot_df[plot_df[stage_col_full] >= rsrp_threshold]
uncovered_df = plot_df[plot_df[stage_col_full] < rsrp_threshold]

fig_map = go.Figure()
if not uncovered_df.empty:
    fig_map.add_trace(go.Scatter(
        x=uncovered_df["x_m"], y=uncovered_df["y_m"], mode="markers",
        marker=dict(size=5, color="#3a3a3a", opacity=0.18, line=dict(width=0)),
        text=uncovered_df["clutter_class"],
        hovertemplate="%{text}<br>below " + f"{rsrp_threshold}" + " dBm threshold — not real coverage<extra></extra>",
        name=f"Below {rsrp_threshold} dBm (not real coverage)",
    ))
if not covered_df.empty:
    color_floor = float(rsrp_threshold)
    color_ceiling = max(float(covered_df[stage_col_full].max()), color_floor + 1.0)
    fig_map.add_trace(go.Scatter(
        x=covered_df["x_m"], y=covered_df["y_m"], mode="markers",
        marker=dict(size=7, color=covered_df[stage_col_full], colorscale="RdYlGn",
                    cmin=color_floor, cmax=color_ceiling,
                    colorbar=dict(title="dBm"), line=dict(width=0)),
        text=covered_df["clutter_class"],
        hovertemplate="%{text}<br>" + meta["label"] + ": %{marker.color:.1f} dBm<extra></extra>",
        name=f"Above {rsrp_threshold} dBm (real coverage)",
    ))
fig_map.add_trace(go.Scatter(
    x=[0], y=[0], mode="markers", marker=dict(symbol="diamond", size=14, color="black"),
    name="Site", hovertemplate="Site<extra></extra>",
))
# Dynamic coverage-boundary ring - radius comes from the real threshold
# crossing computed above, not an assumed/fixed distance.
if coverage_extent_m > 0:
    ring_theta = np.linspace(0, 2 * np.pi, 200)
    fig_map.add_trace(go.Scatter(
        x=coverage_extent_m * np.cos(ring_theta), y=coverage_extent_m * np.sin(ring_theta),
        mode="lines", line=dict(color="black", width=1.5, dash="dot"),
        name=f"Coverage boundary ({rsrp_threshold} dBm, {coverage_extent_m:.0f}m on-axis)",
    ))

# ---- Direction-indicator triangle: SAME geometry method the real frontend
# uses (NetworkPlannerMap.jsx) - apex at the site, two base vertices offset
# from the site at (azimuth - beamwidth/2) and (azimuth + beamwidth/2). This
# is a pure orientation indicator, not a coverage shape - it carries no RSRP
# data, exactly like the frontend's version. Two differences from the
# frontend, both intentional and labeled: (1) length uses this sector's real
# threshold-derived coverage_extent_m instead of the frontend's fixed
# display default, so the triangle's size means something; (2) half-angle
# uses this dashboard's real antenna 3dB half-beamwidth (BEAMWIDTH=65deg,
# the same constant driving the actual antenna gain formula) rather than the
# frontend's separate 30deg display convention, since the constant here IS
# the real RF parameter.
if coverage_extent_m > 0:
    tri_radius = coverage_extent_m
    az_rad = np.radians(azimuth)
    half_bw_rad = np.radians(BEAMWIDTH / 2.0)
    # compass bearing -> (east, north) = (sin, cos)
    left_x, left_y = tri_radius * np.sin(az_rad - half_bw_rad), tri_radius * np.cos(az_rad - half_bw_rad)
    right_x, right_y = tri_radius * np.sin(az_rad + half_bw_rad), tri_radius * np.cos(az_rad + half_bw_rad)
    fig_map.add_trace(go.Scatter(
        x=[0, left_x, right_x, 0], y=[0, left_y, right_y, 0],
        mode="lines", fill="toself", fillcolor="rgba(0,0,0,0.08)",
        line=dict(color="black", width=1.5),
        name=f"Facing direction ({azimuth:.0f}°, ±{BEAMWIDTH/2:.0f}° half-beamwidth)",
    ))
fig_map.update_layout(
    height=560, xaxis_title="meters east of site", yaxis_title="meters north of site",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig_map, use_container_width=True)
st.caption(
    f"Every point out to the {TRACE_RADIUS_M:.0f}m request extent is still computed and still real (no hard "
    "directional cutoff anywhere in the data) - but only points at or above your chosen coverage threshold are "
    "colored; everything below it is shown as a faint gray background instead of being painted on the same "
    "color scale. The colored area IS the real coverage shape for this stage - a wide, roughly circular colored "
    "area means this stage isn't actually differentiating direction/distance well (the current Stage 2 bug); a "
    "colored area that tracks the facing-direction triangle means it is. The dotted ring is the real, "
    "threshold-derived coverage boundary on-axis, not an assumed radius."
)

# ---- Azimuth vs RSRP: proves/disproves the directional physics ----
st.subheader("RSRP vs. off-axis angle — is the antenna pattern actually directional?")
fig_az = go.Figure()
for key in STAGE_KEYS:
    m = STAGE_META[key]
    if m["full"] not in all_points.columns:
        continue  # older trace predating this stage - skip it, don't crash
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
show_unclipped = st.checkbox(
    "Show real pre-clip COST-231 value (unmasks what's really behind the -140dBm floor)",
    value=True,
)
fig_line = go.Figure()
for key in STAGE_KEYS:
    m = STAGE_META[key]
    if m["full"] not in radial.columns:
        continue  # older trace predating this stage - skip it, don't crash
    fig_line.add_trace(go.Scatter(
        x=radial["distance_m"], y=radial[m["full"]], mode="markers",
        marker=dict(color=m["color"], size=3, opacity=0.5 if key != stage_key else 0.85),
        name=m["label"],
    ))
if show_unclipped and "stage1_raw_rsrp_unclipped" in radial.columns:
    fig_line.add_trace(go.Scatter(
        x=radial["distance_m"], y=radial["stage1_raw_rsrp_unclipped"], mode="markers",
        marker=dict(color="#cccccc", size=3, symbol="x", opacity=0.7),
        name="COST-231 raw, UNCLIPPED (real value, no -140 floor)",
    ))
fig_line.add_hline(y=-140, line_dash="dash", line_color="#888888",
                    annotation_text="-140 dBm floor (production clips everything below this to this exact value)")
fig_line.update_layout(
    height=420, xaxis_title="distance from site (m)", yaxis_title="RSRP (dBm)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
)
fig_line.add_vline(x=coverage_extent_m, line_dash="dot", line_color="#333333",
                    annotation_text=f"{rsrp_threshold} dBm boundary")
st.plotly_chart(fig_line, use_container_width=True)
if show_unclipped and "stage1_raw_rsrp_unclipped" in radial.columns:
    below_floor = radial[radial["stage1_raw_rsrp_unclipped"] < -140]
    if not below_floor.empty:
        st.caption(
            f"{len(below_floor)} on-axis points have a REAL computed value below -140dBm (as low as "
            f"{below_floor['stage1_raw_rsrp_unclipped'].min():.1f}dBm) but all display as exactly -140 "
            f"once clipped - that's the flat plateau you're seeing on the map beyond the coverage boundary."
        )
st.caption(f"{len(radial):,} points within the {BEAMWIDTH:.0f}° 3dB beam, sorted by distance. "
           f"The dotted line marks where {STAGE_META[stage_key]['label']} crosses your chosen threshold.")

# ---- Table ----
st.subheader("Point-by-point table — on-axis points")
table_cols = {
    "distance_m": "Dist (m)", "off_axis_deg": "Off-axis (deg)", "spacing_from_prev_m": "Spacing (m)",
    "clutter_class": "Clutter", "stage1_raw_rsrp": "Raw", "stage2_geo_rsrp": "Geo (current)",
    "stage2b_spm_rsrp": "Geo (SPM-style)", "stage2c_spm_rsrp": "Geo (SPM+diffraction)",
    "diffraction_loss_db": "Diffraction loss", "pred_rsrp_calibrated": "Calibrated (on Stage 2)",
    "stage3_rewired_rsrp": "Calibrated (on Stage 2b)", "pred_rsrp_demo": "Smoothed",
    "delta_stage2_geo_rsrp": "DeltaGeo", "delta_stage2b_spm_rsrp": "DeltaGeoSPM",
    "delta_stage2c_spm_rsrp": "DeltaGeoSPMDiff",
    "delta_pred_rsrp_calibrated": "DeltaCalibrated", "delta_stage3_rewired_rsrp": "DeltaCalibratedRewired",
}
table_df = radial[[c for c in table_cols if c in radial.columns]].rename(columns=table_cols)


def highlight_big_delta(val):
    if pd.isna(val):
        return ""
    return "font-weight: 700; color: #eb6834;" if abs(val) >= 3 else ""


delta_cols = [c for c in ["DeltaGeo", "DeltaGeoSPM", "DeltaCalibrated", "DeltaCalibratedRewired"] if c in table_df.columns]
styled = table_df.style.map(highlight_big_delta, subset=delta_cols) \
    .format({c: "{:.1f}" for c in table_df.columns if c != "Clutter"}, na_rep="—")
st.dataframe(styled, use_container_width=True, height=420)
st.caption("Delta columns = change from the previous point in this list. Rows with |Delta| >= 3dB are bolded.")

with st.expander("What each stage actually is (traced from the real code)"):
    st.markdown(
        """
**1. COST-231 raw** — pure physics path-loss model output (`run_rf_prediction_fast`). No geo context at all.

**2. Geo-corrected (current, hand-tuned)** — raw + a weighted offset from clutter class, morphology cluster,
terrain, LOS blockers, interference (`apply_experimental_geo_adjustments`, `DEFAULT_GEO_WEIGHTS`). This value
is normally overwritten in-place by DT calibration before it's ever saved — this dashboard captures it before
that happens. Several of its feature weights are independently distance-clipped (e.g. `serving_distance_m`
capped at 1200m, `nearest_site_distance_m` at 1000m, `mean_nearest3_site_distance_m` at 1500m) — once real
distance passes those caps the added penalty stops growing and stays maxed-out, which is what produces the
flat -140dBm plateau from roughly 1-2km regardless of real direction or clutter.

**2b. Geo-corrected (SPM-style, test-only)** — raw + ONE calibrated per-clutter-class term, fit by
least-squares against real drive-test residuals (`fit_spm_style_clutter_offsets` /
`apply_spm_style_stage2`, test-case only, not wired into production). This mirrors the real Atoll/Forsk
Standard Propagation Model structure (`L = K1 + K2*log(d) + ... + K_clutter*f(clutter)`, confirmed via real
SPM documentation and academic calibration papers) — clutter contributes exactly one calibrated number per
class, and distance is never re-penalized a second time since COST-231's own `slope_term*log10(d_km)`
already models it smoothly inside the raw physics value. The "Stage 2 vs Stage 2b" panel above shows the
real MAE of each against the same held-out drive-test points.

**3. DT-calibrated** — a Ridge regression trained on real drive-test residuals corrects the geo value
(`apply_dt_holdout_calibration` -> `preserve_calibrated_kpis`). This is what the baseline table's main
`pred_rsrp` column actually holds.

**4. Smoothed** — stage 3 blended toward real nearby drive-test measurements by distance
(`apply_demo_dt_overlay`). Saved as `pred_rsrp_smoothed`.

**Antenna gain formula** (`grid_sampling.py::_antenna_gain_estimate`): a 3GPP-style pattern combining
BOTH horizontal and vertical components — `Ah = min(12*(az_diff/65)^2, 30)` (65° 3dB horizontal
half-beamwidth) and `Av = min(12*(elev_diff/6)^2, 20)` (6° 3dB vertical half-beamwidth, using each
site's real electrical + mechanical tilt), combined as `gain = max_gain - min(Ah+Av, 30)`. Real per-site
azimuth/tilt from `site_df` drive this; 65°/6° are documented fallback beamwidth assumptions since this
project's `bw` field is 100% missing. This is applied per-point during COST-231 itself for every point out
to the request extent — no point is ever deleted or masked based on angle; only the coverage *boundary*
shown above is threshold-derived, not the underlying prediction grid.

**Coverage extent** is not a fixed radius anywhere in this trace: COST-231 computes real RSRP out to a
generous request extent (2500m), and the boundary shown on the map/radial chart is wherever the real
predicted value crosses your chosen RSRP threshold, recomputed live as you move the slider.
        """
    )
