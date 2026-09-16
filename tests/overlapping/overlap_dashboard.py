"""
Coverage redundancy -- review dashboard (Streamlit). Reads output/ only: no database, no model run.

Page, top to bottom:
  1. One filter row: which run (project / operator / level), verdict and confidence.
  2. Whether the RF surface passed drive-test validation (if not, RSRP verdicts are withheld) and the
     capacity caveat.
  3. Verdict counts and network coverage.
  4. Candidate table (click a row) beside a map of what removing that site/cell ALONE does: its
     footprint points kept / lost / already weak, the sites that take over, and every candidate.
  5. The selected candidate in plain words: gate by gate, and who absorbs its area.
  6. Expanders: removal order, drive-test validation, run summary.

Run from the ML/ directory (after tests.overlapping.run_overlap):
    venv\\Scripts\\python.exe -m streamlit run tests/overlapping/overlap_dashboard.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from tests.overlapping.config import (
    OUTPUT_DIR,
    VERDICT_DUPLICATE,
    VERDICT_KEEP,
    VERDICT_NOT_TESTABLE,
    VERDICT_REDUNDANT,
)

VERDICT_ORDER = [VERDICT_REDUNDANT, VERDICT_KEEP, VERDICT_DUPLICATE, VERDICT_NOT_TESTABLE]
VERDICT_LABELS = {
    VERDICT_REDUNDANT: "Coverage-redundant",
    VERDICT_KEEP: "Keep",
    VERDICT_DUPLICATE: "Data duplicate",
    VERDICT_NOT_TESTABLE: "Not testable",
}
# Reference palette (dataviz skill). Map markers are a scatter, so only categorical slots 1-3 are
# used (they validate all-pairs); "not testable" is de-emphasised in muted gray, not a 4th hue.
VERDICT_COLORS = {
    VERDICT_KEEP: "#2a78d6",        # slot 1
    VERDICT_REDUNDANT: "#eb6834",   # slot 2
    VERDICT_DUPLICATE: "#1baf7a",   # slot 3
    VERDICT_NOT_TESTABLE: "#898781",
}
# Footprint point states mean good / bad, so they use the reserved status colors (always with a label).
POINT_STATES = {
    "kept": ("#0ca30c", "✓ stays covered"),
    "lost": ("#d03b3b", "✗ loses coverage"),
    "weak": ("#c3c2b7", "· weak before removal"),
}
INK_PRIMARY, INK_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SERIES_1 = "#2a78d6"

REASON_TEXT = {
    "RETENTION_BELOW_MIN": "loses reliably covered area",
    "EXPECTED_COVERAGE_EROSION": "wears coverage down across its whole area",
    "INDOOR_RETENTION_BELOW_MIN": "loses indoor coverage",
    "HOLE_TOO_LARGE": "leaves a coverage hole",
    "POOR_AREA_DEGRADES": "makes already-weak areas much weaker",
    "QUALITY_DEGRADES": "pushes SINR below the threshold",
    "HO_AMBIGUITY_INCREASE": "creates handover ping-pong areas",
    "NETWORK_LOSS_CAP": "total network coverage-loss limit reached",
    "MAX_REMOVALS_REACHED": "removal limit reached",
    "TOO_LITTLE_AREA": "too little of it inside the analysis area",
    "OUTSIDE_POLYGON": "outside the project polygon",
    "NO_RF_CELLS": "no usable antennas in the RF model",
    "RF_SURFACE_NOT_VALIDATED": "RF model does not reproduce this operator's drive test",
    "CONFIG_AMBIGUOUS": "site configuration is not trustworthy",
    "DUPLICATE_OF": "same antenna stored twice; kept copy",
}


def plain_reason(reason) -> str:
    if not isinstance(reason, str) or not reason:
        return ""
    for code in ("RF_SURFACE_NOT_VALIDATED", "CONFIG_AMBIGUOUS", "DUPLICATE_OF"):
        if reason.startswith(code + "("):
            return f"{REASON_TEXT[code]}: {reason[len(code) + 1:-1]}"
    prefix = ""
    if reason.startswith("NEEDED_AFTER_REMOVALS("):
        prefix = "Redundant on its own, but needed once the higher-ranked removals are applied - then it "
        reason = reason[len("NEEDED_AFTER_REMOVALS("):-1]
    return prefix + "; ".join(REASON_TEXT.get(p, p) for p in reason.split(";") if p)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_run(run_dir: str, stamp: float) -> dict:
    d = Path(run_dir)
    footprints = d / "candidate_footprints.parquet"
    return {
        "summary": json.loads((d / "summary.json").read_text()),
        "candidates": _read_csv(d / "candidates.csv"),
        "iterations": _read_csv(d / "iterations.csv"),
        "dt": _read_csv(d / "dt_validation.csv"),
        "grid": pd.read_parquet(d / "grid_states.parquet", columns=["point_id", "lat", "lon"]),
        "footprints": pd.read_parquet(footprints) if footprints.exists() else pd.DataFrame(),
    }


def _offset(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    b = math.radians(bearing_deg)
    return (
        lat + dist_m * math.cos(b) / 110_574.0,
        lon + dist_m * math.sin(b) / (111_320.0 * math.cos(math.radians(lat))),
    )


def _fmt(value, pattern: str = "{:.3f}") -> str:
    return "—" if value is None or (isinstance(value, float) and not math.isfinite(value)) else pattern.format(value)


def build_map(data: dict, cand: pd.DataFrame, row: pd.Series, level: str, location_probability: float) -> folium.Map:
    # OpenStreetMap tiles need no API key (CartoDB basemaps now render an "API KEY REQUIRED" watermark)
    m = folium.Map(location=[row["lat"], row["lon"]], zoom_start=16, tiles="OpenStreetMap", control_scale=True)

    fp = data["footprints"]
    if not fp.empty:
        pts = fp[fp["candidate_id"] == row["candidate_id"]].merge(data["grid"], on="point_id")
        for p in pts.itertuples(index=False):
            state = "lost" if p.lost else ("kept" if p.prob_before >= location_probability else "weak")
            color, label = POINT_STATES[state]
            after = "no signal" if not isinstance(p.serving_after, str) or not p.serving_after else p.serving_after
            folium.CircleMarker(
                [p.lat, p.lon],
                radius=4,
                weight=0,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=(
                    f"{label}<br>RSRP {_fmt(p.rsrp_before, '{:.0f}')} → {_fmt(p.rsrp_after, '{:.0f}')} dBm"
                    f"<br>P(covered) {p.prob_before:.2f} → {p.prob_after:.2f}<br>then served by {after}"
                ),
            ).add_to(m)

    shares = {}
    if isinstance(row.get("absorbers"), str) and row["absorbers"] not in ("", "[]"):
        for a in json.loads(row["absorbers"]):
            shares[str(a["site_key"])] = shares.get(str(a["site_key"]), 0.0) + a["share"]
    site_rows = cand.drop_duplicates("site_key") if level == "site" else cand.drop_duplicates("site_key")
    for site in site_rows.itertuples(index=False):
        if str(site.site_key) in shares:
            folium.CircleMarker(
                [site.lat, site.lon], radius=15, color=INK_SECONDARY, weight=2, fill=False,
                tooltip=f"Site {site.site_key} takes over {shares[str(site.site_key)]:.0%} of the footprint",
            ).add_to(m)

    if level == "cell":
        for c in cand.itertuples(index=False):
            az = float(str(c.azimuths).split(",")[0])
            end = _offset(c.lat, c.lon, az, 90.0)
            folium.PolyLine(
                [[c.lat, c.lon], list(end)], color=VERDICT_COLORS.get(c.verdict, INK_SECONDARY),
                weight=5 if c.candidate_id == row["candidate_id"] else 3, opacity=0.95,
                tooltip=f"{c.candidate_id} - {VERDICT_LABELS.get(c.verdict, c.verdict)}",
            ).add_to(m)
    else:
        for c in cand.itertuples(index=False):
            selected = c.candidate_id == row["candidate_id"]
            folium.CircleMarker(
                [c.lat, c.lon], radius=10 if selected else 7,
                color=INK_PRIMARY if selected else SURFACE, weight=3 if selected else 2,
                fill=True, fill_color=VERDICT_COLORS.get(c.verdict, INK_SECONDARY), fill_opacity=1.0,
                tooltip=f"{c.candidate_id} - {VERDICT_LABELS.get(c.verdict, c.verdict)}",
            ).add_to(m)
        for az in str(row["azimuths"]).split(","):
            if az:
                folium.PolyLine(
                    [[row["lat"], row["lon"]], list(_offset(row["lat"], row["lon"], float(az), 120.0))],
                    color=INK_PRIMARY, weight=2,
                ).add_to(m)
    return m


def legend_html(level: str) -> str:
    def swatch(color, label, round_=True):
        radius = "50%" if round_ else "2px"
        return (f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px">'
                f'<span style="width:10px;height:10px;border-radius:{radius};background:{color};display:inline-block"></span>'
                f"{label}</span>")
    verdicts = "".join(swatch(VERDICT_COLORS[v], VERDICT_LABELS[v], level == "site") for v in VERDICT_ORDER)
    states = "".join(swatch(color, label) for color, label in POINT_STATES.values())
    ring = swatch("transparent;border:2px solid " + INK_SECONDARY, "site that takes over")
    return f'<div style="font-size:0.85rem;line-height:1.9">{verdicts}<br>{states}{ring}</div>'


def gate_table(row: pd.Series, decision: dict) -> pd.DataFrame:
    def gate(name, value, limit, ok, unit=""):
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return {"Gate": name, "Value": "—", "Limit": limit, "Result": "— not applicable"}
        return {"Gate": name, "Value": f"{value:.3f}{unit}" if unit != " m²" else f"{value:,.0f}{unit}",
                "Limit": limit, "Result": "✓ pass" if ok else "✗ fail"}

    g = lambda key: row.get(key, float("nan"))  # noqa: E731
    uncovered_applies = g("uncovered_share") >= decision["min_uncovered_share"]
    rows = [
        gate("Reliably covered area kept", g("retention"), f"≥ {decision['min_retention']}", g("retention") >= decision["min_retention"]),
        gate("Expected coverage kept", g("expected_retention"), f"≥ {decision['min_expected_retention']}",
             g("expected_retention") >= decision["min_expected_retention"]),
        gate("Largest hole", g("largest_hole_m2"), f"≤ {decision['max_hole_m2']:,.0f} m²", g("largest_hole_m2") <= decision["max_hole_m2"], " m²"),
        gate("Area pushed below SINR threshold", g("new_low_sinr_share"), f"≤ {decision['max_new_low_sinr_share']}",
             g("new_low_sinr_share") <= decision["max_new_low_sinr_share"]),
        gate("Handover ping-pong area increase", g("ho_ambiguity_increase"), f"≤ {decision['max_ho_ambiguity_increase']}",
             g("ho_ambiguity_increase") <= decision["max_ho_ambiguity_increase"]),
        gate("Network coverage loss so far", g("cumulative_coverage_loss"), f"≤ {decision['max_cumulative_coverage_loss']}",
             g("cumulative_coverage_loss") <= decision["max_cumulative_coverage_loss"] + 1e-12),
        gate("RSRP drop where already weak (dB)", g("uncovered_rsrp_drop_db") if uncovered_applies else float("nan"),
             f"≤ {decision['max_uncovered_rsrp_drop_db']} dB", g("uncovered_rsrp_drop_db") <= decision["max_uncovered_rsrp_drop_db"]),
    ]
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Coverage redundancy", layout="wide")
    runs = sorted(p.parent for p in OUTPUT_DIR.glob("*/*/summary.json"))
    if not runs:
        st.warning("No runs yet. Run: venv\\Scripts\\python.exe -m tests.overlapping.run_overlap --operator all")
        st.stop()
    labels = {f"{p.parent.name} / {p.name}": p for p in runs}
    names = list(labels)
    default = next((i for i, n in enumerate(names) if n.endswith("airtel_site") and "193" in n), 0)

    f1, f2, f3 = st.columns([2, 3, 2])
    label = f1.selectbox("Run", names, index=default)
    data = load_run(str(labels[label]), (labels[label] / "summary.json").stat().st_mtime)
    summary, cand = data["summary"], data["candidates"]
    present = [v for v in VERDICT_ORDER if v in set(cand["verdict"])]
    verdicts = f2.multiselect("Verdict", present, default=present, format_func=lambda v: VERDICT_LABELS[v])
    confidences = f3.multiselect("Confidence (empty = all)", ["HIGH", "MEDIUM", "LOW"])

    level = summary["level"]
    st.title(f"Coverage redundancy - {summary['operator']}, {level} level, project {summary['project_id']}")
    validation = summary["rf_surface_validation"]
    if validation["status"] == "PASS":
        st.success("RF model validated on drive sessions it was not calibrated on - RSRP-based verdicts are published.")
    elif validation["status"] == "FAIL":
        st.error(
            "RF model NOT validated (" + ", ".join(validation["failures"]) + "). Every RSRP-based verdict is withheld "
            "as Not testable; the model's own answer is kept in the model_verdict column for review only."
        )
    else:
        st.warning("No drive test to validate the RF model - verdicts are model-only.")
    st.info(summary["note"])

    counts = cand["verdict"].value_counts()
    tiles = st.columns(6)
    for i, v in enumerate(VERDICT_ORDER):
        tiles[i].metric(VERDICT_LABELS[v], int(counts.get(v, 0)))
    net = summary["network"]
    before, after = net["expected_coverage_initial"], net["expected_coverage_after_removals"]
    tiles[4].metric("Expected coverage after removals", f"{after:.1%}", f"{(after - before) * 100:+.2f} pts")
    held = summary["calibration"].get("heldout") or {}
    bias = f"Median bias {held['bias_db']:+.1f} dB on held-out drive sessions." if held.get("bias_db") is not None else None
    tiles[5].metric("Held-out RSRP error σ", f"{summary['thresholds']['sigma_rsrp_db']:.1f} dB", help=bias)

    view = cand[cand["verdict"].isin(verdicts)]
    if confidences:
        view = view[view["confidence"].isin(confidences)]
    view = view.reset_index(drop=True)
    if view.empty:
        st.warning("No candidates match the filters.")
        st.stop()

    left, right = st.columns([5, 6])
    with left:
        rank = view["removal_rank"] if "removal_rank" in view else pd.Series(np.nan, index=view.index)
        table = view.assign(
            removal_rank=rank.map(lambda v: "" if pd.isna(v) else str(int(v))),
            verdict_label=view["verdict"].map(VERDICT_LABELS),
            why=view["reason"].map(plain_reason),
        )
        columns = {
            "removal_rank": "Rank",
            "candidate_id": "Candidate",
            "verdict_label": "Verdict",
            "confidence": "Confidence",
            "retention": st.column_config.NumberColumn("Area kept", format="%.3f"),
            "expected_retention": st.column_config.NumberColumn("Expected kept", format="%.3f"),
            "largest_hole_m2": st.column_config.NumberColumn("Hole m²", format="%.0f"),
            "footprint_users": st.column_config.NumberColumn("Users", format="%.0f"),
            "absorber_sites": "Taken over by",
            "why": "Why",
        }
        shown = [c for c in columns if c in table.columns]
        event = st.dataframe(
            table[shown], column_config=columns, hide_index=True, use_container_width=True, height=560,
            on_select="rerun", selection_mode="single-row",
        )
        picked = event.selection.rows if event and event.selection else []
        if picked:
            row = view.iloc[picked[0]]
        else:
            redundant = view[view["verdict"] == VERDICT_REDUNDANT]
            confident = redundant[redundant["confidence"] == "HIGH"]
            row = next(frame for frame in (confident, redundant, view) if not frame.empty).iloc[0]

    with right:
        location_probability = summary["thresholds"]["location_probability"]
        st_folium(build_map(data, cand, row, level, location_probability), height=520, use_container_width=True,
                  returned_objects=[], key=f"map-{label}-{row['candidate_id']}")
        st.markdown(legend_html(level), unsafe_allow_html=True)

    st.subheader(f"{row['candidate_id']} - {VERDICT_LABELS.get(row['verdict'], row['verdict'])}")
    if row["verdict"] == VERDICT_REDUNDANT:
        st.write("Every gate passes: coverage and quality are kept without it, given the removals ranked before it. "
                 "Capacity is not checked - confirm the load on the sites that take over first.")
    else:
        st.write(plain_reason(row.get("reason")) or "—")
    if row.get("model_verdict") and row["model_verdict"] != row["verdict"]:
        st.caption(f"Model's own answer (not published): {VERDICT_LABELS.get(row['model_verdict'], row['model_verdict'])}"
                   f" - {plain_reason(row.get('model_reason')) or 'all gates pass'}")

    if row["verdict"] in (VERDICT_REDUNDANT, VERDICT_KEEP) or row.get("model_verdict") in (VERDICT_REDUNDANT, VERDICT_KEEP):
        g, a = st.columns([3, 2])
        with g:
            st.markdown("**Gates** (state when it was last evaluated)")
            st.dataframe(gate_table(row, summary["config"]["decision"]), hide_index=True, use_container_width=True)
        with a:
            st.markdown("**Who takes over its area**")
            absorbers = json.loads(row["absorbers"]) if isinstance(row.get("absorbers"), str) else []
            if absorbers:
                st.dataframe(
                    pd.DataFrame(absorbers).rename(columns={"antenna_key": "Antenna", "site_key": "Site", "share": "Share"}),
                    column_config={"Share": st.column_config.NumberColumn(format="%.2f")},
                    hide_index=True, use_container_width=True,
                )
            else:
                st.write("—")
            share = row.get("absorber_handover_neighbour_share")
            st.caption(
                f"Footprint: {_fmt(row.get('footprint_area_m2'), '{:,.0f}')} m², {_fmt(row.get('footprint_users'), '{:.0f}')} users, "
                f"{int(row.get('dt_points_in_footprint') or 0)} drive-test points. "
                + (f"{share:.0%} of the takeover goes to sites the phones already hand over to."
                   if isinstance(share, float) and math.isfinite(share) else "No handover evidence for this site.")
            )

    with st.expander("Removal order"):
        it = data["iterations"]
        if it.empty:
            st.write("Nothing was removed.")
        else:
            chart = (
                alt.Chart(it)
                .mark_line(color=SERIES_1, strokeWidth=2, point=alt.OverlayMarkDef(color=SERIES_1, size=64))
                .encode(
                    x=alt.X("iteration:Q", title="Removal step", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("network_covered_share:Q", title="Expected network coverage", scale=alt.Scale(zero=False),
                            axis=alt.Axis(format=".1%")),
                    tooltip=[alt.Tooltip("iteration:Q", title="Step"), alt.Tooltip("removed_candidate:N", title="Removed"),
                             alt.Tooltip("network_covered_share:Q", title="Expected coverage", format=".2%")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True, theme="streamlit")
            st.dataframe(it, hide_index=True, use_container_width=True)

    with st.expander("Drive-test validation"):
        cal = summary["calibration"]
        st.write(
            f"Calibration source: **{cal['source']}** - {cal['rows_attributed']:,} of {cal['rows_in_area']:,} drive-test rows "
            f"matched to a configured antenna, {cal['sessions']} sessions, {cal['cv_folds']} folds. "
            f"Per-antenna offsets: {'yes' if cal['per_antenna_offsets'] else 'no'}. "
            f"Interference load factor {cal['interference_load_factor']} ({cal['load_source']})."
        )
        c1, c2 = st.columns(2)
        c1.markdown("**Held-out prediction error**")
        c1.json(held)
        c2.markdown("**Level per band (measured − model, dB)**")
        c2.json(cal.get("band_offsets_db") or {})
        dt = data["dt"]
        if not dt.empty and "heldout_residual_db" in dt:
            resid = dt["heldout_residual_db"].dropna().clip(-40, 40)
            bins = (np.floor(resid / 2.0) * 2.0).value_counts().sort_index()
            hist = pd.DataFrame({"residual_db": bins.index, "rows": bins.values})
            chart = (
                alt.Chart(hist)
                .mark_bar(color=SERIES_1, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=alt.RelativeBandSize(0.85))
                .encode(
                    x=alt.X("residual_db:O", title="Measured − predicted RSRP on held-out sessions (dB, 2 dB bins)"),
                    y=alt.Y("rows:Q", title="Drive-test rows"),
                    tooltip=[alt.Tooltip("residual_db:O", title="From dB"), alt.Tooltip("rows:Q", title="Rows", format=",")],
                )
                .properties(height=240)
            )
            st.altair_chart(chart, use_container_width=True, theme="streamlit")
            st.dataframe(hist, hide_index=True, use_container_width=True)

    with st.expander("Run summary (JSON)"):
        st.json(summary)


main()
