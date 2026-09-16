"""
Sector Swap Check -- test-case dashboard (Streamlit). Reads data/*.csv only; no database.

Active model: antenna-pattern swap detector (pattern_detector.py + antenna_profile.py). For every
carrier (site + operator + technology + EARFCN) the expected 36 x 10-degree antenna gains of each
configured antenna endpoint (azimuth + pattern) are compared, at the same drive-test locations, with
which sector the phone measured stronger -- for every way of assigning endpoints to PCIs. PCI numbers
never change. Every result shows the antenna pattern it rests on: EXACT, ASSUMED (technology rule) or
APPROXIMATE (generic 3GPP) -- the last two can never give a Confirmed swap.
  Demo            controlled synthetic sites (3 normal, a pair swap, a rotation)
  Whole project   one operator + technology at a time, never mixed

Run from the ML/ directory:
    venv\\Scripts\\python.exe -m streamlit run tests/swap_sector/swap_sector_dashboard.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

from tests.swap_sector.detect_sector_swap import (
    BIN_SIZE_DEG,
    DATA_DIR,
    angle_diff_deg,
    direction_profile,
    reference_provenance,
    run as run_detection,
    scorecard,
    strongest_per_spot,
)
from tests.swap_sector.evidence import Settings

RESULT_LABELS = {
    "CONFIRMED_SWAP": "🚨 Confirmed swap (rule-based)",
    "PROBABLE_SWAP": "⚠️ Probable swap",
    "AZIMUTH_MISMATCH": "🧭 Direction / RF anomaly",
    "AMBIGUOUS": "🔀 Ambiguous",
    "NORMAL": "✅ Normal",
    "NOT_ENOUGH_DATA": "❔ Insufficient data",
    "NOT_TESTABLE": "➖ Not testable",
}
QUALITY_LABELS = {
    "EXACT": "Exact antenna pattern",
    "ASSUMED": "Assumed antenna model",
    "APPROXIMATE": "Generic 3GPP pattern (approximate)",
    "": "—",
}
MATCH_LABELS = {"ID": "eNodeB ID", "PCI": "PCI fallback"}
OUTCOME_LABELS = {
    "FOUND": "✅ Swap found", "WRONG_SECTORS": "⚠️ Wrong sectors", "MISSED": "❌ Swap missed",
    "FALSE_ALARM": "❌ False alarm", "CORRECT": "✅ Correct", "UNDECIDED": "❔ Undecided",
}
# Distinct colours per sector inside one carrier (PCI-modulo palettes collide, e.g. PCI 51 / 151).
SECTOR_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#9a6324"]
NO_WINNER_COLOR = "#9ca3af"
WEDGE_RADIUS_M = 250.0
WEDGE_WIDTH_DEG = 60
RING_INNER_M = 290.0
RING_OUTER_M = 340.0
MAP_HEIGHT = 400
MAX_MAP_POINTS = 3000
VIEW_FILES = {
    "demo": ["cells_demo.csv", "measurements_demo.csv", "demo_metadata.json"],
    "project": ["cells.csv", "measurements.csv", "skipped.csv"],
}
HOW_IT_WORKS = """
**How it works**
1. **Expected (no drive-test RF):** every configured antenna endpoint's pattern is rotated to its configured azimuth and
   averaged into 36 slices of 10°. The carrier frequency comes from the drive-test EARFCN. Pattern source:
   **Exact** (real antenna model + in-band file) › **Assumed** (technology rule + in-band file) › **Approximate**
   (generic 3GPP). Another band's antenna file is never used.
2. **Observed (drive test):** at each spot, which sector the phone measured stronger (two sectors ≥ 3 dB apart, or the
   serving sector against sectors it did not even report). Comparing at the same spot cancels distance and propagation.
3. **Compare:** for every way of assigning the antenna endpoints to the PCIs, count how many dB the expected gains in that
   spot's slice put the weaker sector ahead (beyond 3 dB). PCI numbers never change.
4. **Decide:** the configured mapping fits → **Normal**. A mapping exchanging ≥ 2 sectors fits clearly and stably better →
   **Probable swap** (**Confirmed** only with exact patterns and handover support). One sector off with no exchange →
   **Direction / RF anomaly**. Unstable answer → **Ambiguous**. Thin data → **Insufficient data**.
"""


def file_stamp(names: list[str]) -> tuple:
    paths = [DATA_DIR / name for name in names + ["handovers.csv", "raw_fetch_info.json"]]
    paths += [DATA_DIR.parent / name for name in ("antenna_profile.py", "pattern_detector.py", "evidence.py",
                                                  "detect_sector_swap.py", "reference_provenance.json")]
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in paths)


@st.cache_data(show_spinner="Running the antenna-pattern detector...")
def load_results(config_name: str, operator: str, technology: str, stamp: tuple, max_violation_db: float) -> pd.DataFrame:
    return run_detection(config_name, operator, technology, Settings(method="pattern", max_violation_db=max_violation_db))


@st.cache_data
def load_csv(name: str, stamp: tuple) -> pd.DataFrame:
    dtype = None if name.startswith("measurements") else {"site_id": str}
    return pd.read_csv(DATA_DIR / name, dtype=dtype, low_memory=False)


def json_frame(text) -> pd.DataFrame:
    if not isinstance(text, str) or not text.strip():
        return pd.DataFrame()
    try:
        return pd.DataFrame(json.loads(text))
    except ValueError:
        return pd.DataFrame()


def number_text(value, unit: str = "", digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "–"
    return "–" if not math.isfinite(number) else f"{number:.{digits}f}{unit}"


def project_label() -> str:
    info = DATA_DIR / "raw_fetch_info.json"
    return str(json.loads(info.read_text()).get("project_id", "")) if info.exists() else ""


def demo_scope() -> tuple[str, str] | None:
    path = DATA_DIR / "cells_demo.csv"
    if not path.exists():
        return None
    demo = pd.read_csv(path, usecols=["operator", "technology"])
    return str(demo["operator"].iloc[0]), str(demo["technology"].iloc[0])


def offset_point(lat: float, lon: float, distance_m: float, bearing: float) -> tuple[float, float]:
    ang = distance_m / 6371000.0
    brg, phi1, lam1 = math.radians(bearing), math.radians(lat), math.radians(lon)
    phi2 = math.asin(math.sin(phi1) * math.cos(ang) + math.cos(phi1) * math.sin(ang) * math.cos(brg))
    lam2 = lam1 + math.atan2(math.sin(brg) * math.sin(ang) * math.cos(phi1), math.cos(ang) - math.sin(phi1) * math.sin(phi2))
    return math.degrees(phi2), math.degrees(lam2)


def wedge(lat: float, lon: float, azimuth: float) -> list[tuple[float, float]]:
    half = WEDGE_WIDTH_DEG // 2
    arc = [offset_point(lat, lon, WEDGE_RADIUS_M, azimuth + step) for step in range(-half, half + 1, 5)]
    return [(lat, lon)] + arc


def ring_slice(lat: float, lon: float, start_deg: float) -> list[tuple[float, float]]:
    angles = [start_deg, start_deg + BIN_SIZE_DEG / 2, start_deg + BIN_SIZE_DEG]
    outer = [offset_point(lat, lon, RING_OUTER_M, a) for a in angles]
    inner = [offset_point(lat, lon, RING_INNER_M, a) for a in reversed(angles)]
    return outer + inner


def carrier_label(row) -> str:
    frequency = number_text(row.get("frequency_mhz", float("nan")), " MHz", 0)
    return f"{row.get('band', '')} / EARFCN {row.get('earfcn', '')}" + ("" if frequency == "–" else f" / {frequency}")


def injected_change(group_cells: pd.DataFrame) -> str:
    moved = group_cells[angle_diff_deg(group_cells["azimuth"], group_cells["true_azimuth"]) >= 1]
    if moved.empty:
        return "Left as is"
    return "Antenna endpoints crossed on purpose: " + " ↔ ".join(f"PCI {int(p)}" for p in moved["pci"])


def select_row(table: pd.DataFrame, key: str) -> int:
    event = st.dataframe(table, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row", key=key)
    rows = event["selection"]["rows"]
    if not rows:
        st.caption("Click a row to open it. Showing the first row.")
    return rows[0] if rows else 0


def base_map(group_cells: pd.DataFrame) -> tuple[folium.Map, float, float]:
    site_lat, site_lon = float(group_cells["site_lat"].iloc[0]), float(group_cells["site_lon"].iloc[0])
    return folium.Map(location=[site_lat, site_lon], zoom_start=16, tiles="CartoDB positron"), site_lat, site_lon


def add_site_marker(fmap: folium.Map, lat: float, lon: float, site_id: str) -> None:
    folium.CircleMarker([lat, lon], radius=6, color="#111827", fill=True, fill_color="#111827", tooltip=f"Site {site_id}").add_to(fmap)


def build_configured_map(group_cells: pd.DataFrame, colors: dict, site_id: str) -> folium.Map:
    fmap, lat, lon = base_map(group_cells)
    for pci, azimuth in zip(group_cells["pci"], group_cells["azimuth"]):
        folium.Polygon(wedge(lat, lon, float(azimuth)), color=colors[pci], weight=2, fill=True, fill_opacity=0.3,
                       tooltip=f"PCI {int(pci)} · configured {azimuth:.0f}°").add_to(fmap)
    add_site_marker(fmap, lat, lon, site_id)
    return fmap


def build_drive_test_map(group_cells: pd.DataFrame, colors: dict, strongest: pd.DataFrame, profile: pd.DataFrame, site_id: str) -> folium.Map:
    fmap, lat, lon = base_map(group_cells)
    points = strongest.sample(MAX_MAP_POINTS, random_state=0) if len(strongest) > MAX_MAP_POINTS else strongest
    for p in points.itertuples():
        folium.CircleMarker([p.lat, p.lon], radius=3, weight=0, fill=True, fill_color=colors[p.pci], fill_opacity=0.85,
                            tooltip=f"Strongest: PCI {p.pci} ({p.rsrp:.0f} dBm)").add_to(fmap)
    for b in profile.itertuples():
        if b.spots == 0:
            continue
        has_winner = not pd.isna(b.dominant_pci)
        label = f"PCI {int(b.dominant_pci)} strongest at {b.share:.0%}" if has_winner else "no clear winner"
        folium.Polygon(ring_slice(lat, lon, b.bin_deg), color="white", weight=1, fill=True,
                       fill_color=colors[int(b.dominant_pci)] if has_winner else NO_WINNER_COLOR,
                       fill_opacity=0.9 if has_winner else 0.35,
                       tooltip=f"{b.bin_deg}–{b.bin_deg + BIN_SIZE_DEG}°: {b.spots} spots, {label}").add_to(fmap)
    add_site_marker(fmap, lat, lon, site_id)
    return fmap


def profile_figure(row: pd.Series, pcis: list[int], colors: dict) -> go.Figure | None:
    details = json_frame(row.get("profile_details", ""))
    if details.empty:
        return None
    fig = make_subplots(rows=1, cols=len(pcis), subplot_titles=[f"PCI {p}" for p in pcis], shared_yaxes=True)
    for k, pci in enumerate(pcis, start=1):
        d = details[details["pci"] == pci].sort_values("bin_deg")
        angle = d["bin_deg"] + BIN_SIZE_DEG / 2
        observed = pd.to_numeric(d["observed_db"], errors="coerce")
        shift = (observed - d["expected_gain_db"]).median() if observed.notna().any() else 0.0
        fig.add_trace(go.Scatter(x=angle, y=d["expected_gain_db"], name="Configured antenna (expected gain)",
                                 line={"color": "#6b7280", "dash": "dash"}, showlegend=k == 1, legendgroup="configured"), row=1, col=k)
        if "best_gain_db" in d and not d["best_gain_db"].equals(d["expected_gain_db"]):
            fig.add_trace(go.Scatter(x=angle, y=d["best_gain_db"], name="Best-fit antenna (expected gain)",
                                     line={"color": "#111827"}, showlegend=k == 1, legendgroup="best"), row=1, col=k)
        fig.add_trace(go.Scatter(x=angle, y=observed - shift, name="Drive test (distance-corrected, shifted)", mode="markers",
                                 marker={"color": colors[pci], "size": 6}, showlegend=k == 1, legendgroup="observed"), row=1, col=k)
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 40, "b": 10}, legend={"orientation": "h", "y": -0.25})
    fig.update_xaxes(range=[0, 360], dtick=90, title_text="Direction from site (°)")
    fig.update_yaxes(title_text="Relative dB", row=1, col=1)
    return fig


def pattern_text(pattern: pd.Series | None) -> str:
    if pattern is None:
        return "–"
    text = f"{QUALITY_LABELS.get(pattern['pattern_quality'], pattern['pattern_quality'])} · {pattern['antenna_model']}"
    file_tilt = number_text(pattern.get("file_e_tilt"), "°", 0)
    if file_tilt != "–":
        text += f" · file tilt {file_tilt} (site {number_text(pattern.get('requested_e_tilt'), '°', 0)})"
    return text


def render_detail(row: pd.Series, cells: pd.DataFrame, measurements: pd.DataFrame, view: str) -> None:
    show_truth = view == "demo"
    st.divider()
    st.subheader(f"Site {row['site_id']} · {row['operator']} {row['technology']} · {carrier_label(row)}")
    st.markdown(f"**{RESULT_LABELS.get(row['verdict'], row['verdict'])}** — {row['reason']}")
    quality = row.get("pattern_quality", "")
    quality = "" if pd.isna(quality) else str(quality)
    note = row.get("pattern_note", "")
    st.markdown(f"Antenna pattern: **{QUALITY_LABELS.get(quality, quality)}**" + (f" — {note}" if isinstance(note, str) and note else ""))
    if quality in ("ASSUMED", "APPROXIMATE"):
        st.caption("Without the real antenna model and its pattern file, this carrier can be a Probable swap at most, never Confirmed.")

    group_cells = cells[cells["group_id"] == row["group_id"]].sort_values("pci").reset_index(drop=True)
    if show_truth:
        st.info(f"Demo setup: {injected_change(group_cells)}.  Detector: {OUTCOME_LABELS.get(row.get('outcome', ''), row.get('outcome', ''))}")
    metrics = st.columns(4)
    support = row.get("support")
    for col, label, value in zip(metrics, ["Violation, configured mapping", "Violation, best mapping", "Improvement", "Same answer when resampled"],
                                 [number_text(row.get("original_violation_db"), " dB"), number_text(row.get("best_violation_db"), " dB"),
                                  number_text(row.get("violation_improvement_db"), " dB"),
                                  "–" if number_text(support) == "–" else f"{float(support):.0%}"]):
        col.metric(label, value)
    st.caption("Violation = on average, how many dB the expected antenna gains put the wrong sector ahead of the one the phone measured "
               "stronger at the same spot (beyond a 3 dB tolerance). 0 dB = every comparison agrees. PCI values are unchanged.")

    pcis = group_cells["pci"].astype(int).tolist()
    colors = {pci: SECTOR_COLORS[k % len(SECTOR_COLORS)] for k, pci in enumerate(pcis)}
    figure = profile_figure(row, pcis, colors)
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True, key=f"{view}_{row['group_id']}_profile")
        st.caption("Diagnostic view per PCI: dashed = expected gain of the antenna it is configured on; solid = the best-fit antenna "
                   "(only drawn when different); dots = drive-test RSRP per 10° slice, distance-corrected and shifted by one constant.")

    group_meas = measurements[(measurements["group_id"] == row["group_id"]) & measurements["pci"].isin(pcis)]
    profile = direction_profile(group_meas, pcis, require_comparison=True)
    strongest = strongest_per_spot(group_meas)
    left, right = st.columns(2)
    with left:
        st.markdown("**Configured directions (site table)**")
        st_folium(build_configured_map(group_cells, colors, row["site_id"]), width=None, height=MAP_HEIGHT,
                  returned_objects=[], key=f"{view}_{row['group_id']}_configured")
    with right:
        st.markdown("**Drive test**")
        st_folium(build_drive_test_map(group_cells, colors, strongest, profile, row["site_id"]), width=None, height=MAP_HEIGHT,
                  returned_objects=[], key=f"{view}_{row['group_id']}_drive_test")
    st.caption(f"Right map: dot = strongest measured PCI · ring = {BIN_SIZE_DEG}° slices coloured by the PCI that is strongest at most "
               f"shared locations (grey = no clear winner).")

    fits, patterns = json_frame(row.get("sector_fit_details", "")), json_frame(row.get("pattern_details", ""))
    sector_rows = []
    for k, pci in enumerate(pcis):
        fit = fits[fits["pci"] == pci].iloc[0] if not fits.empty and (fits["pci"] == pci).any() else None
        pattern = patterns[patterns["pci"] == pci].iloc[0] if not patterns.empty and (patterns["pci"] == pci).any() else None
        configured = float(group_cells.at[k, "azimuth"])
        best = float(fit["best_endpoint_azimuth"]) if fit is not None else configured
        entry = {
            "PCI": pci,
            "Configured": f"{configured:.0f}°",
            "Antenna pattern": pattern_text(pattern),
            "Violation on configured antenna": number_text(fit["violation_configured_db"], " dB") if fit is not None else "–",
            "Best-fit antenna": f"{best:.0f}°" + (" 🔁" if angle_diff_deg(best, configured) >= 1 else ""),
            "Violation on best-fit antenna": number_text(fit["violation_best_db"], " dB") if fit is not None else "–",
            "Comparisons": int(fit["comparisons"]) if fit is not None else 0,
            "Measured slices": int(fit["measured_bins"]) if fit is not None else 0,
        }
        if show_truth:
            entry["True antenna"] = f"{group_cells.at[k, 'true_azimuth']:.0f}°"
        sector_rows.append(entry)
    sector_df = pd.DataFrame(sector_rows)
    styled = sector_df.style.apply(lambda col: [f"background-color: {colors[p]}; color: white" for p in col], subset=["PCI"])
    st.dataframe(styled, hide_index=True, use_container_width=True)

    with st.expander("Coverage, handover and pattern evidence"):
        keys = ["comparison_locations", "comparisons", "serving_comparisons", "coverage_fraction", "locations",
                "original_profile_error_db", "best_profile_error_db", "ho_points", "ho_original_loss_db", "ho_best_loss_db",
                "confidence_score", "confidence_basis", "match_level", "provenance"]
        st.dataframe(pd.DataFrame({"Evidence": keys, "Value": [str(row.get(k, "")) for k in keys]}), hide_index=True)
        if not patterns.empty:
            st.dataframe(patterns, hide_index=True, use_container_width=True)
        st.caption("Serving comparisons = the serving sector against group sectors the phone did not report. Handover points are matched "
                   "serving-cell transitions (proxies, not protocol-confirmed handovers). Profile errors are diagnostics only.")


def render_demo(operator: str, technology: str, stamp: tuple) -> None:
    results = load_results("demo", operator, technology, stamp, st.session_state.max_violation)
    cells = load_csv("cells_demo.csv", stamp)
    measurements = load_csv("measurements_demo.csv", stamp)
    st.caption(f"All demo sites are {operator} {technology}. Each operator + technology is checked on its own, never mixed.")
    st.info(json.loads((DATA_DIR / "demo_metadata.json").read_text())["description"])
    st.markdown(HOW_IT_WORKS)
    view = results.assign(_swap=results["ground_truth"].eq("SWAPPED")).sort_values(["_swap", "site_id"], ascending=[False, True]).reset_index(drop=True)
    setups = {gid: injected_change(g) for gid, g in cells.groupby("group_id")}
    table = pd.DataFrame({
        "Site": view["site_id"],
        "Carrier": view.apply(carrier_label, axis=1),
        "Antenna pattern": view["pattern_quality"].fillna("").map(QUALITY_LABELS),
        "Demo setup": view["group_id"].map(setups),
        "Result": view["verdict"].map(RESULT_LABELS),
        "Detector right?": view["outcome"].map(OUTCOME_LABELS),
    })
    render_detail(view.iloc[select_row(table, "demo_table")], cells, measurements, "demo")


def render_reliability(operator: str, technology: str) -> None:
    names = ["cells.csv", "measurements.csv", "cells_synthetic.csv"]
    if not (DATA_DIR / "cells_synthetic.csv").exists():
        return
    with st.expander(f"How reliable is this for {operator} {technology}? (synthetic test)"):
        results = load_results("synthetic", operator, technology, file_stamp(names), st.session_state.max_violation)
        full = scorecard(results).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Swaps found", f"{full['found']} of {full['injected']}")
        c2.metric("Found, but wrong sectors", int(full["wrong_sectors"]))
        c3.metric("False alarms", f"{full['false_alarms']} of {full['untouched']}")
        rows = []
        for quality, subset in results.groupby(results["pattern_quality"].fillna("")):
            card = scorecard(subset).iloc[0]
            rows.append({"Antenna pattern": QUALITY_LABELS.get(quality, quality), "Swaps found": f"{card.found} of {card.injected}",
                         "Wrong sectors": int(card.wrong_sectors), "False alarms": f"{card.false_alarms} of {card.untouched}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Whole antenna endpoints were exchanged in the configuration of half the eligible carriers; every test carrier's "
                   "azimuths were also moved by up to 15°. Drive-test data is real. Controls are not field-verified healthy sites, and "
                   "recovering crossed configurations does not prove field accuracy.")


def render_project(stamp: tuple) -> None:
    cells_all = load_csv("cells.csv", stamp)
    scopes = list(cells_all.groupby(["operator", "technology"])["group_id"].nunique().sort_values(ascending=False).index)
    operator, technology = st.selectbox("Operator · Technology (each is checked on its own)", scopes, format_func=lambda s: f"{s[0]} · {s[1]}")
    results = load_results("real", operator, technology, stamp, st.session_state.max_violation)
    cells = cells_all[(cells_all["operator"] == operator) & (cells_all["technology"] == technology)]
    measurements = load_csv("measurements.csv", stamp)
    skipped = load_csv("skipped.csv", stamp)

    st.markdown(HOW_IT_WORKS)
    counts = results["verdict"].value_counts()
    for col, (key, label) in zip(st.columns(len(RESULT_LABELS)), RESULT_LABELS.items()):
        col.metric(label, int(counts.get(key, 0)))
    quality_counts = results["pattern_quality"].fillna("").value_counts()
    st.caption("Carriers by antenna pattern: " + " · ".join(f"{QUALITY_LABELS.get(q, q)}: {n}" for q, n in quality_counts.items() if q)
               + f" · no pattern (not testable earlier): {int(quality_counts.get('', 0))}")
    render_reliability(operator, technology)

    chosen = st.multiselect("Result", list(RESULT_LABELS), default=[k for k in RESULT_LABELS if k != "NOT_TESTABLE"], format_func=RESULT_LABELS.get)
    view = results[results["verdict"].isin(chosen)]
    view = view.assign(_rank=view["verdict"].map({k: i for i, k in enumerate(RESULT_LABELS)})).sort_values(["_rank", "site_id"]).reset_index(drop=True)
    if view.empty:
        st.info("No carriers match this filter.")
    else:
        table = pd.DataFrame({
            "Site": view["site_id"],
            "Carrier": view.apply(carrier_label, axis=1),
            "Antenna pattern": view["pattern_quality"].fillna("").map(QUALITY_LABELS),
            "Matched by": view["match_level"].map(MATCH_LABELS),
            "Sectors": view["n_sectors"],
            "Result": view["verdict"].map(RESULT_LABELS),
            "Why": view["reason"],
        })
        render_detail(view.iloc[select_row(table, f"project_table_{operator}_{technology}")], cells, measurements, "project")

    operator_skipped = skipped[skipped["operator"] == operator]
    with st.expander(f"{operator} site-table sectors that could not be checked ({len(operator_skipped)})"):
        st.dataframe(operator_skipped["reason_type"].value_counts().rename_axis("Reason").reset_index(name="Sectors"), hide_index=True)
        st.dataframe(operator_skipped, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Sector Swap Check", layout="wide")
    st.title("Sector Swap Check")
    st.sidebar.select_slider("Max mean violation (dB)", options=[2.0, 3.0, 4.0], value=3.0, key="max_violation")
    st.sidebar.caption("Sensitivity control. 3 dB (one tolerance step) is the starting value; no field-calibrated optimum exists yet.")
    st.caption(f"Project {project_label()} · Reference: {reference_provenance()['status']} · Local CSV snapshot · Antenna-pattern model")
    st.warning("Testing dashboard. Probable and rule-confirmed swaps need field verification. The demo sites are synthetic, "
               "not a field accuracy benchmark.")
    skipped_path = DATA_DIR / "skipped.csv"
    if skipped_path.exists():
        skipped_all = pd.read_csv(skipped_path)
        nr = skipped_all[skipped_all["technology"].eq("NR")]
        with st.expander(f"Data quality / not testable — {len(skipped_all)} excluded sector records; {len(nr)} NR records"):
            st.dataframe(skipped_all.groupby(["operator", "technology", "reason_type"], dropna=False).size().reset_index(name="Sector records"), hide_index=True)
            st.dataframe(skipped_all, hide_index=True)
    validation_path = DATA_DIR / "validation_summary.csv"
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        with st.expander("Validation: antenna-pattern model vs earlier methods (synthetic, exploratory)"):
            columns = [c for c in ("configuration", "partition", "injected", "found", "wrong_sectors", "untouched", "false_alarms",
                                   "max_violation_db", "comparison_tolerance_db", "min_support") if c in validation.columns]
            st.dataframe(validation[columns], hide_index=True)
            quality_path = DATA_DIR / "validation_by_pattern_quality.csv"
            if quality_path.exists():
                st.dataframe(pd.read_csv(quality_path), hide_index=True)
            st.caption("Sensitivity results, not threshold calibration. One project; no real field-labelled swaps.")
    demo = demo_scope()
    demo_label = f"Demo — {demo[0]} {demo[1]} sites (learn the concept)" if demo else "Demo"
    view = st.radio("View", ["demo", "project"], horizontal=True,
                    format_func=lambda v: demo_label if v == "demo" else f"Whole project {project_label()}")
    missing = [name for name in VIEW_FILES[view] if not (DATA_DIR / name).exists()]
    if missing:
        st.error(f"Missing {', '.join(missing)}. Run build_dataset, make_synthetic_swap and make_demo first (see README.md).")
        st.stop()
    stamp = file_stamp(VIEW_FILES[view])
    if view == "demo":
        render_demo(demo[0], demo[1], stamp)
    else:
        render_project(stamp)


if __name__ == "__main__":
    main()
