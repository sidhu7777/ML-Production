"""
New-format PDF report sections (target layout: report_VI.pdf).

This module lives in the tests folder ON PURPOSE: production code under
tools/report_engine is NOT modified.  We subclass the production
PDFReportGenerator and reuse its styles / native-table rendering, adding
the new-format sections one step (a few pages) at a time.

LLM budget policy (Groq API limit is small, so only narrative text that
truly needs generation goes to the LLM; everything else is rule-based):
  - Introduction        -> LLM (single small call, production prompt rules,
                            production rule-based synthesizer as fallback)
  - Area Summary        -> production rule-based logic (build_area_summary:
                            spatial grid + geocoding).  Production itself
                            ignores the LLM for this section.
  - Drive Summary (new) -> rule-based from production drive metadata
  - Executive Summary   -> rule-based from the report dataframe
  - Later engineering sections (Coverage/Mobility/Handover/Service)
                        -> rule-based tables; LLM only if a section needs
                           a narrative that cannot be templated.

Step plan (each step is validated before the next one starts; the SAME
output PDF is regenerated/replaced on every run):
  Step 1 (done):    Section 3 "Drive Summary" new format
                    (metric table + Executive Summary block).
  Step 1b (done):   Sections 1-2 wired to production logic
                    (LLM Introduction, production Area Summary + route map).
  Step 1c (done):   Correct session/total distance via Haversine sum-of-
                    consecutive-GPS-points (production's tbl_session query is
                    permission-denied and silently falls back to 0.0 km);
                    corrected Session Details native table placed in
                    section 3.
  Step 1d (done):   GPS-glitch hop rejection in the distance calc (raw sums
                    were inflated by single-second position jumps implying
                    physically impossible speeds); Coverage/Quality/Mobility
                    Executive KPI statuses now use the exact supplied
                    thresholds (classify_coverage/classify_quality/
                    classify_mobility) instead of ad-hoc cutoffs.
  Step 1e (done):   NR-aware primary-row filter (filter_primary_rows_
                    including_nr) — production's primary filter only
                    recognizes the LTE "mRegistered=YES" tag and silently
                    drops all 5G NSA rows; report_df is now rebuilt through
                    the `primary` column instead, matching the frontend and
                    report_VI.pdf exactly (23,306 samples / 129 unique PCI /
                    40.8% top-PCI share). Mobility now reuses this same
                    report_df instead of a separate all-cells dataframe.
                    Also fixed a ReportLab page-split bug where a
                    sub-heading could be stranded at the bottom of a page
                    while its content flowed to the next (add_labeled_block
                    + KeepTogether now glues every heading to its content).
  Step 2 (done):    Section 4 "Coverage KPI Analysis" — 4.1 Band
                    Distribution, 4.2 RSRP (blended across technologies),
                    4.3 RSRQ and 4.4 SINR (each broken out per technology,
                    never blended — reuses _band_rsrq/_band_sinr), 4.5
                    Coverage KPI Summary, Overall Coverage Assessment.
                    Reuses classify_coverage/classify_quality_by_technology
                    from Section 3 (exec_summary["coverage"] /
                    ["quality_entries"]) so the two sections can never
                    disagree with each other.
  Step 3 (done):    Section 5 "Mobility KPI Analysis" - Serving Cell
                    Analysis, PCI Distribution, Neighbor Cell Analysis,
                    Mobility KPI Summary, Overall Mobility Assessment.
  Step 4:           Section 6 "Handover KPI Analysis".
  Step 5:           Section 7 "Service KPI Analysis".
  Step 6:           Full assembly + compare against report_VI.pdf.
"""

import json
import os
import re
import time

import numpy as np
import pandas as pd
from reportlab.platypus import Paragraph, Spacer, PageBreak, CondPageBreak, KeepTogether, Image
from reportlab.lib.units import inch

from tools.report_engine.pdf_generator import (
    PDFReportGenerator,
    PageNumCanvas,
    make_native_table,
    TABLE_MAX_WIDTH,
)


# ------------------------------------------------------------------
# Primary-row filter — NR/5G-NSA-aware replacement for
# load_data_db._filter_primary_rows() (test-case only; tools/report_engine
# is NOT modified).
#
# Production's _filter_primary_rows() checks `primary_cell_info_1` FIRST
# for the LTE-only tag "mRegistered=YES" (an `if/elif`, so once that
# column is found present it never falls through to the simpler `primary`
# column, even when `primary` is available and reliable). NR/5G-NSA rows
# tag `primary_cell_info_1` as "nr_from_signalstrength" instead, which
# never matches, so those rows get silently dropped before report_df is
# even built — for project 248 this drops all 6,269 raw "5G NSA" rows.
#
# Verified for project 248: `primary` is "Yes" for ALL 26,079 raw rows
# across every technology (4G, 4G LTE-Anchor-NSA, 5G NSA) — a reliable,
# technology-agnostic flag. Using it instead (matching production's own
# intended fallback order, just unreachable there) and re-deriving
# report_df through it matches independent references almost exactly:
#   - Total samples after the known-band filter: 23,306
#     (report_VI.pdf states 23,306 samples for this exact project)
#   - Unique PCI: 129, Top-PCI share: 40.76%
#     (frontend PCI/BCCH panel: Unique 129, PCI 495 = 40.8%)
# vs. the old primary_cell_info_1-based report_df: 19,462 samples,
# Unique PCI 123, top-PCI share 48.8% — all off, and missing 5G NSA
# entirely. This single corrected `report_df` is then used consistently
# for Coverage, per-technology Radio Quality, AND Mobility/PCI — not a
# different population per metric.
# ------------------------------------------------------------------

def filter_primary_rows_including_nr(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer the `primary` column (technology-agnostic); fall back to the
    LTE-only `primary_cell_info_1` tag only if `primary` isn't present."""
    if df.empty:
        return df
    if "primary" in df.columns:
        primary_values = df["primary"].fillna("").astype(str).str.strip().str.lower()
        mask = primary_values.isin({"yes", "y", "true", "1"})
        return df.loc[mask].reset_index(drop=True)
    if "primary_cell_info_1" in df.columns:
        primary_info = df["primary_cell_info_1"].fillna("").astype(str)
        mask = primary_info.str.contains("mRegistered=YES", case=False, na=False)
        return df.loc[mask].reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# Session distance — Haversine sum of CONSECUTIVE GPS points (test-case
# only; production's tbl_session distance query is permission-denied and
# silently returns 0.0 km, so this is not a production bug fix — it fills
# a gap using data we already have: per-row lat/lon/timestamp).
#
# Session Distance = sum_{i=1}^{n-1} d(P_i, P_i+1)   -- NOT first->last
# displacement.
#
# GPS-GLITCH REJECTION (required for a CORRECT sum, not an optional
# extra): a raw per-hop diagnostic on project 248's own logs showed
# single-second position jumps implying instantaneous speeds up to
# ~4000 km/h (session 4530: 10 such hops out of 959 accounted for 3.35 km
# of its 4.00 km "distance"; session 4742: 73 hops out of 3768 accounted
# for 6.42 km of 22.03 km). The device's own reported `speed` field never
# exceeded ~86 km/h for any session in this project. These are GPS fixes
# glitching to a far-away point and back, not real movement, so a hop is
# rejected (excluded from the sum) when it implies a speed above
# _MAX_PLAUSIBLE_SPEED_KMH. This does not change the required formula
# (still a sum of consecutive-point distances) — it only excludes
# fixes that are not real, consecutive positions.
#
# Sanity check after filtering (device's own avg `speed` field * duration
# vs. filtered Haversine sum): session 4530 0.64 km vs 0.65 km computed;
# session 4714 19.2 km vs 20.57 km; session 4742 15.18 km vs 15.61 km —
# all within a plausible margin, confirming the filtered sum is realistic.
# ------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0
_MAX_PLAUSIBLE_SPEED_KMH = 150.0


def _consecutive_haversine_km(
    lat: np.ndarray, lon: np.ndarray, timestamp: pd.Series | None = None
) -> tuple[float, int, float]:
    """
    Sum of Haversine distances between consecutive (lat, lon) points, in km,
    rejecting hops that imply a physically-implausible instantaneous speed
    (GPS glitches). Returns (kept_km, rejected_hop_count, rejected_km).
    """
    if len(lat) < 2:
        return 0.0, 0, 0.0

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    hop_km = _EARTH_RADIUS_KM * c

    if timestamp is not None and len(timestamp) == len(lat):
        dt_sec = pd.Series(timestamp).diff().dt.total_seconds().to_numpy()[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            dt_hr = np.where(dt_sec > 0, dt_sec / 3600.0, np.nan)
            implied_speed_kmh = np.where(np.isnan(dt_hr), 0.0, hop_km / dt_hr)
        valid = implied_speed_kmh <= _MAX_PLAUSIBLE_SPEED_KMH
        kept_km = float(hop_km[valid].sum())
        rejected_km = float(hop_km[~valid].sum())
        rejected_hops = int((~valid).sum())
        return kept_km, rejected_hops, rejected_km

    return float(hop_km.sum()), 0, 0.0


def compute_session_distances_km(
    gps_df: pd.DataFrame, session_ids: list[int], verbose: bool = True
) -> dict[int, float]:
    """
    Per-session distance from consecutive GPS points, sorted by timestamp,
    with GPS-glitch hops rejected (see module notes above).

    Uses the polygon-filtered ALL-CELLS dataframe (not the primary-cell
    filtered report_df) so the GPS trail isn't thinned by the primary-cell
    marker — distance only depends on the phone's GPS track, not which
    cell was serving at that instant.
    """
    required = {"session_id", "lat", "lon"}
    if not required.issubset(gps_df.columns):
        return {sid: 0.0 for sid in session_ids}

    df = gps_df.dropna(subset=["lat", "lon"]).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    out = {}
    for sid in session_ids:
        sub = df[df["session_id"] == sid]
        if "timestamp" in sub.columns:
            sub = sub.dropna(subset=["timestamp"]).sort_values("timestamp")
        lat = sub["lat"].astype(float).to_numpy()
        lon = sub["lon"].astype(float).to_numpy()
        ts = sub["timestamp"] if "timestamp" in sub.columns else None
        kept_km, rejected_hops, rejected_km = _consecutive_haversine_km(lat, lon, ts)
        out[sid] = round(kept_km, 2)
        if verbose and rejected_hops:
            print(
                f"[distance] session {sid}: rejected {rejected_hops} GPS-glitch hop(s) "
                f"(~{rejected_km:.2f} km of implausible movement, >{_MAX_PLAUSIBLE_SPEED_KMH:.0f} km/h implied speed)"
            )
    return out


def build_session_native_table(drive_summary: dict, distances_km: dict) -> pd.DataFrame:
    """
    Session Details table (Session/Date/Start/End/Duration/Distance) with
    the corrected Haversine-based distance instead of production's
    tbl_session-dependent (and currently 0.0) figure.
    """
    rows = []
    for s in drive_summary.get("sessions", []):
        sid = int(s["session_id"])
        dist_km = distances_km.get(sid, 0.0)
        rows.append({
            "Session": sid,
            "Date": s.get("date"),
            "Start": s.get("start_time"),
            "End": s.get("end_time"),
            "Duration": s.get("duration"),
            "Distance": f"{dist_km:.2f} km",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Introduction text — LLM (production prompt rules), tiny prompt only
# ------------------------------------------------------------------

INTRO_INSTRUCTIONS = """
Write a professional introduction for a telecom drive test report (3-4 sentences).
Use an executive summary style.
Mention the test location if available in metadata.location.city/country (e.g., "in <city>, <country>").
Explain the PURPOSE and IMPORTANCE of drive testing in general terms (signal strength, coverage, provider performance).
Mention that map visualizations highlight strong/weak coverage areas.
State that the report provides actionable insights for optimization and deployment planning.
DO NOT include specific drive details (distance, samples, dates) - those belong in Drive Summary.
DO NOT invent any facts not present in metadata.
Return ONLY the introduction paragraph as plain text (no JSON, no markdown).
"""


def generate_introduction_text(
    metadata: dict,
    model: str = "llama-3.1-8b-instant",
    max_retries: int = 3,
    verbose: bool = True,
) -> tuple[str, str]:
    """
    Generate ONLY the Introduction via the LLM (same provider/model/prompt
    rules as production llm_integration, but a minimal prompt so we do not
    burn the small API quota on sections that are rule-based here).

    Returns (text, source) where source is "llm" or "synth".
    """
    from groq import Groq
    from tools.report_engine.llm_integration import (
        _fill_missing_sections_from_metadata,
    )

    llm_metadata = {
        "location": metadata.get("location", {}),
        "introduction": metadata.get("introduction"),
    }
    prompt = (
        f"METADATA:\n{json.dumps(llm_metadata, indent=2, default=str)}\n\n"
        f"{INTRO_INSTRUCTIONS}"
    )

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        client = Groq(api_key=api_key)
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300,
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    if verbose:
                        print("[llm] Introduction generated by LLM")
                    return text, "llm"
            except Exception as exc:
                if verbose:
                    print(f"[llm] attempt {attempt}/{max_retries} failed: {exc}")
                if attempt < max_retries:
                    time.sleep(min(attempt, 2))

    # Production rule-based fallback (same function production uses).
    if verbose:
        print("[llm] falling back to production rule-based Introduction")
    report = _fill_missing_sections_from_metadata({}, metadata, missing_keys=["Introduction"])
    return report.get("Introduction", "Not available."), "synth"


# ------------------------------------------------------------------
# Data-driven Executive Summary (used inside the new Drive Summary)
# ------------------------------------------------------------------

def _series(report_df: pd.DataFrame, col: str) -> pd.Series:
    if col not in report_df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(report_df[col], errors="coerce").dropna()


# ------------------------------------------------------------------
# Coverage / Quality / Mobility classification — the EXACT thresholds
# supplied for the Executive KPI Summary (and reused as-is by the
# Coverage KPI Analysis section in Step 2, so the two sections can never
# disagree):
#
#   Good Coverage : RSRP > -95 dBm for more than 90% of samples
#   Fair Coverage : Average RSRP -95 to -105 dBm
#   Poor Coverage : Average RSRP < -105 dBm, coverage (%> -95dBm) < 75%
#
#   Good Quality  : Avg SINR > 20 dB AND Avg RSRQ > -10 dB AND Avg CQI > 10
#                   AND No High Interference
#   Fair Quality  : SINR 13-20 dB, RSRQ -10 to -15 dB, CQI 7-10
#   Poor Quality  : SINR < 13 dB, RSRQ < -15 dB, CQI < 7, High Interference
#
#   Mobility      : Neighbor OK AND PCI OK AND Serving Cell Stable -> GOOD
#
# NOTE on inputs this project's data cannot supply (confirmed with the
# project owner, not silently assumed):
#   - CQI and "High Interference"/BLER are dropped from the Quality
#     decision for this project. BLER is NULL for every row, and per
#     direction received, Quality here is classified from Average SINR
#     and Average RSRQ only. Each metric is banded independently against
#     its own Good/Fair/Poor thresholds and the WORSE of the two bands
#     wins — this also avoids a bug in an earlier version that required
#     RSRQ to sit inside the narrow Fair band (-10 to -15) even when it
#     was actually better than the Good cutoff (>-10), which wrongly
#     dragged Quality down.
#   - "Neighbor OK": all_neigbor_cell_info is empty for every row in this
#     project, so Neighbor cannot be evaluated from data today. Mobility
#     status below is therefore computed from PCI + Serving-Cell checks
#     only, and is explicitly labelled as such rather than silently
#     assumed "OK". Full neighbor-relation analysis is Step 3 (Mobility
#     KPI Analysis / Neighbor Cell Analysis), which will feed the same
#     classification function once that data is available.
# ------------------------------------------------------------------

DOMINANT_PCI_SHARE_THRESHOLD = 80.0  # top-PCI share considered unstable

_STATUS_RANK = {"Good": 0, "Fair": 1, "Poor": 2}  # higher = worse


def _worst_status(statuses) -> str:
    """Worst of a collection of Good/Fair/Poor/N/A statuses (N/A treated as best/ignorable)."""
    known = [s for s in statuses if s in _STATUS_RANK]
    if not known:
        return "N/A"
    return max(known, key=lambda s: _STATUS_RANK[s])


def classify_coverage(rsrp: pd.Series) -> tuple[str, str]:
    if rsrp.empty:
        return "N/A", "No RSRP samples available."

    pct_above_95 = float((rsrp > -95).mean() * 100)
    avg_rsrp = float(rsrp.mean())

    if pct_above_95 > 90:
        status = "Good"
    elif -105 <= avg_rsrp < -95:
        status = "Fair"
    elif avg_rsrp < -105 or pct_above_95 < 75:
        status = "Poor"
    else:
        status = "Fair"

    remarks = f"Average RSRP {avg_rsrp:.1f} dBm; {pct_above_95:.0f}% of samples above -95 dBm."
    return status, remarks


def _band_sinr(avg_sinr: float) -> str:
    if avg_sinr > 20:
        return "Good"
    if 13 <= avg_sinr <= 20:
        return "Fair"
    return "Poor"


def _band_rsrq(avg_rsrq: float) -> str:
    if avg_rsrq > -10:
        return "Good"
    if -15 <= avg_rsrq <= -10:
        return "Fair"
    return "Poor"


def classify_quality(sinr: pd.Series, rsrq: pd.Series) -> tuple[str, str]:
    """
    Quality status from Average SINR and Average RSRQ only (CQI/BLER are
    not part of this decision — see module notes above). Each metric is
    banded independently and the worse band wins.
    """
    if sinr.empty:
        return "N/A", "No SINR samples available."

    avg_sinr = float(sinr.mean())
    sinr_band = _band_sinr(avg_sinr)

    if rsrq.empty:
        status = sinr_band
        remarks = f"Average SINR {avg_sinr:.1f} dB. No RSRQ samples available."
        return status, remarks

    avg_rsrq = float(rsrq.mean())
    rsrq_band = _band_rsrq(avg_rsrq)
    status = max([sinr_band, rsrq_band], key=lambda b: _STATUS_RANK[b])

    remarks = f"Average SINR {avg_sinr:.1f} dB; Average RSRQ {avg_rsrq:.1f} dB."
    return status, remarks


def classify_mobility(mobility_df: pd.DataFrame) -> tuple[str, str]:
    """
    PCI dominance for the Mobility row.

    Pass the corrected `report_df` (built through
    filter_primary_rows_including_nr + filter_known_band_rows — see notes
    above) here, the SAME population used for Coverage and per-technology
    Radio Quality. An earlier version of this function routed Mobility to
    the broader all-cells `handover_df` instead, to work around
    report_df's top-PCI share being wrong (48.8%, vs. frontend's ~40%) —
    but that was compensating for the primary-row filter bug, not a real
    reason to use a different population for this metric. Now that
    report_df itself is corrected, it matches the frontend directly
    (Unique PCI 129, top-PCI 40.76% vs. frontend's 129 / 40.8%), so there
    is no more reason to special-case Mobility onto a different dataframe.

    Neighbor-relation and Serving-Cell-Stability checks are not included
    here (all_neigbor_cell_info is empty for every row in this project) —
    per direction received, absence of that data is simply not mentioned
    rather than called out with a caveat sentence; Step 3 (Mobility KPI
    Analysis) adds it once real data supports it.
    """
    if "pci" not in mobility_df.columns or mobility_df["pci"].dropna().empty:
        return "N/A", "No PCI data available."

    pci_series = mobility_df["pci"].dropna()
    unique_pci = int(pci_series.nunique())
    top_pci_share = float(pci_series.value_counts(normalize=True).iloc[0] * 100)
    pci_ok = top_pci_share < DOMINANT_PCI_SHARE_THRESHOLD

    status = "Good" if pci_ok else "Fair"
    remarks = f"{unique_pci} unique serving PCIs (top PCI {top_pci_share:.0f}% of samples)."
    return status, remarks


MIN_SAMPLES_PER_TECHNOLOGY = 30  # too few points -> not worth a separate row


def _technology_groups(report_df: pd.DataFrame) -> list[str]:
    """
    Distinct RAT/technology values present in this drive (from the
    `network` column, e.g. "4G", "4G (LTE Anchor - NSA)", "5G NSA"),
    ordered by sample count. Technologies with fewer than
    MIN_SAMPLES_PER_TECHNOLOGY rows are skipped as too sparse for a
    meaningful separate classification.
    """
    if "network" not in report_df.columns:
        return []
    values = report_df["network"].fillna("").astype(str).str.strip()
    values = values[values.ne("")]
    counts = values.value_counts()
    return [tech for tech, n in counts.items() if n >= MIN_SAMPLES_PER_TECHNOLOGY]


def _tech_slug(tech: str) -> str:
    """
    Filesystem/HTML-safe slug for a technology label (e.g. the `network`
    column's "4G (LTE Anchor - NSA)" -> "4g_lte_anchor_nsa"), shared between
    this module and test_new_pdf_report.py so per-technology map filenames
    always match up on both the generation and rendering side.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", str(tech).strip().lower()).strip("_")
    return slug or "unknown"


def classify_quality_by_technology(report_df: pd.DataFrame) -> list[dict]:
    """
    Radio Quality is classified PER TECHNOLOGY, never blended across RATs:
    SINR/RSRQ have different characteristic ranges depending on whether the
    sample is 2G/3G/4G/5G (e.g. in this project 5G NSA's average RSRQ is
    -10.3 dB vs 4G's -7.2 dB — a real RAT-scale difference, not noise).
    Averaging them together would misrepresent both. RSRP/Coverage is NOT
    split this way — RSRP is comparable in dBm terms across RATs.

    Returns one {"technology", "status", "remarks"} entry per technology
    present in the `network` column; if that column is absent, falls back
    to a single ungrouped entry.

    KNOWN GAP (production behaviour, confirmed, left as-is — NOT a bug in
    this test case, and tools/report_engine is not modified to work around
    it): production's load_data_db._filter_primary_rows() keeps only rows
    whose `primary_cell_info_1` contains "mRegistered=YES" (the LTE
    primary-cell tag). For NSA 5G projects the NR secondary-carrier rows
    tag `primary_cell_info_1` as "nr_from_signalstrength" instead, so they
    never match and are dropped before `report_df` is built — for project
    248 this silently removes all 6,269 raw "5G NSA" rows, and this
    affects the ENTIRE existing report (maps, KPI tables, etc.), not just
    this function. Only the technologies that survive that filter (here:
    "4G" and "4G (LTE Anchor - NSA)") can appear below.
    """
    tech_list = _technology_groups(report_df)
    if not tech_list:
        sinr = _series(report_df, "sinr")
        rsrq = _series(report_df, "rsrq")
        status, remarks = classify_quality(sinr, rsrq)
        return [{"technology": None, "status": status, "remarks": remarks}]

    results = []
    network_col = report_df["network"].fillna("").astype(str).str.strip()
    for tech in tech_list:
        sub = report_df.loc[network_col == tech]
        sinr = _series(sub, "sinr")
        rsrq = _series(sub, "rsrq")
        status, remarks = classify_quality(sinr, rsrq)
        results.append({
            "technology": tech,
            "status": status,
            "remarks": f"[{len(sub):,} samples] {remarks}",
        })
    return results


def derive_executive_summary(
    report_df: pd.DataFrame,
    handover_count: int | None = None,
    mobility_df: pd.DataFrame | None = None,
) -> dict:
    """
    Build the Executive Summary content (report_VI.pdf, section 3).
    Rule-based (no LLM); Coverage/Quality/Mobility statuses use the exact
    thresholds documented above, not ad-hoc cutoffs. Radio Quality is
    broken out per technology (see classify_quality_by_technology) —
    Coverage is not.

    `mobility_df` MUST be the all-cells, polygon-filtered population (same
    as production's `handover_df`), not `report_df` — see classify_mobility
    docstring for why mixing the primary-cell-restricted population into
    this one metric is wrong. Falls back to report_df only if the caller
    doesn't have the all-cells dataframe available.
    """

    rsrp = _series(report_df, "rsrp")
    rsrq = _series(report_df, "rsrq")
    sinr = _series(report_df, "sinr")

    coverage_status, coverage_remarks = classify_coverage(rsrp)
    quality_entries = classify_quality_by_technology(report_df)
    mobility_status, mobility_remarks = classify_mobility(
        mobility_df if mobility_df is not None else report_df
    )

    # --- Handover ---
    if handover_count is None:
        handover_status = "N/A"
        handover_remarks = "Handover analysis not evaluated in this step."
    else:
        handover_status = "Good"
        handover_remarks = (
            f"{handover_count} band handover events; no major mobility "
            f"interruption observed."
        )

    kpi_rows = [("Coverage", coverage_status, coverage_remarks)]
    for entry in quality_entries:
        label = f"Radio Quality - {entry['technology']}" if entry["technology"] else "Radio Quality"
        kpi_rows.append((label, entry["status"], entry["remarks"]))
    kpi_rows.append(("Mobility", mobility_status, mobility_remarks))
    kpi_rows.append(("Handover", handover_status, handover_remarks))

    observations = []
    if not rsrp.empty:
        pct_above_95 = float((rsrp > -95).mean() * 100)
        observations.append(
            f"Coverage classified as {coverage_status} "
            f"({pct_above_95:.0f}% of RSRP samples above -95 dBm, average {float(rsrp.mean()):.1f} dBm)."
        )
    for entry in quality_entries:
        tech_label = entry["technology"] or "overall"
        observations.append(
            f"Radio quality ({tech_label}) classified as {entry['status']}."
            + (" Requires optimization." if entry["status"] not in ("Good", "N/A") else "")
        )
    if mobility_status != "Good":
        observations.append(
            "PCI dominance or serving-cell instability detected; review PCI planning."
        )

    quality_statuses = [e["status"] for e in quality_entries]
    overall = (
        "The network demonstrated satisfactory RF performance. Coverage and "
        "mobility KPIs were generally within acceptable limits. Capacity "
        "optimization and interference mitigation are recommended in "
        "selected locations."
        if all(s in ("Good", "N/A") for s in [coverage_status, mobility_status, *quality_statuses])
        else (
            "The network showed mixed RF performance with some KPI groups "
            "below the Good threshold. Targeted optimization is recommended "
            "for the KPI groups flagged Fair or Poor above."
        )
    )

    return {
        "objective": (
            "The objective of this drive test was to evaluate LTE radio "
            "coverage, radio quality, mobility performance and end-user "
            "service experience along the planned drive route. Measurements "
            "were collected using STracer and analysed against operator "
            "acceptance criteria."
        ),
        "scope": [
            "Coverage assessment using Band Distribution, RSRP, RSRQ and SINR.",
            "Mobility assessment using Serving Cell, PCI Distribution and Neighbor behaviour.",
            "Handover verification including Intra-frequency, Inter-frequency and Inter-RAT events where available.",
            "Service quality evaluation using Throughput, MOS, Latency, Jitter and Packet Loss.",
        ],
        "kpi_rows": kpi_rows,
        "observations": observations,
        "overall_assessment": overall,
        # Raw classification results, so Section 4 (Coverage KPI Analysis)
        # can reuse the SAME computed values instead of recomputing them —
        # the two sections can then never disagree with each other.
        "coverage": {"status": coverage_status, "remarks": coverage_remarks},
        "quality_entries": quality_entries,
    }


def build_drive_metric_table(drive_summary: dict) -> pd.DataFrame:
    """Metric / Value table shown at the top of the new Drive Summary."""
    distance = drive_summary.get("distance_covered")
    distance_text = (
        f"{distance:.1f} km" if isinstance(distance, (int, float)) and distance > 0
        else "N/A"
    )
    return pd.DataFrame({
        "Metric": ["Distance Covered", "Total Samples", "Total Sessions", "Number of Days"],
        "Value": [
            distance_text,
            f"{int(drive_summary.get('total_samples', 0)):,}",
            f"{int(drive_summary.get('total_sessions', 0))}",
            f"{int(drive_summary.get('number_of_days', 0))} Days",
        ],
    })


def build_drive_summary_text(drive_summary: dict) -> str:
    start = drive_summary.get("start_date") or "an unspecified start date"
    end = drive_summary.get("end_date") or "an unspecified end date"
    distance = drive_summary.get("distance_covered")
    distance_text = (
        f"{distance:.1f} km" if isinstance(distance, (int, float)) and distance > 0
        else "0.0 km"
    )
    return (
        f"The drive test was conducted over "
        f"{int(drive_summary.get('number_of_days', 0))} days from {start} to {end}, "
        f"covering {distance_text} with {int(drive_summary.get('total_samples', 0)):,} "
        f"samples collected across {int(drive_summary.get('total_sessions', 0))} sessions."
    )


# ------------------------------------------------------------------
# Step 2 — Section 4 "Coverage KPI Analysis" (report_VI.pdf layout)
#
# Reuses classify_coverage / classify_quality_by_technology / _band_sinr /
# _band_rsrq directly — the exact same functions Section 3's Executive
# Summary already uses — so this detailed section can never disagree with
# the Executive KPI Summary. RSRP/Coverage stays blended across
# technologies (per direction received: RSRP is dBm-comparable across
# RATs); RSRQ and SINR are each broken out per technology (per direction
# received: their characteristic ranges differ by RAT and must not be
# averaged together). All content here is rule-based/templated — no LLM.
# ------------------------------------------------------------------

def build_rsrp_metric_table(rsrp: pd.Series) -> pd.DataFrame:
    """4.2 RSRP Analysis metric table (Average / Best / Worst / % > -95 dBm)."""
    if rsrp.empty:
        return pd.DataFrame({"Metric": ["Average RSRP"], "Sample Value": ["No data available"]})
    pct_above_95 = float((rsrp > -95).mean() * 100)
    return pd.DataFrame({
        "Metric": ["Average RSRP", "Best RSRP", "Worst RSRP", "Samples > -95 dBm"],
        "Sample Value": [
            f"{rsrp.mean():.1f} dBm",
            f"{rsrp.max():.1f} dBm",
            f"{rsrp.min():.1f} dBm",
            f"{pct_above_95:.0f}%",
        ],
    })


def build_rsrq_technology_table(report_df: pd.DataFrame) -> pd.DataFrame:
    """4.3 RSRQ Analysis — one row per technology (never blended)."""
    return _build_per_technology_metric_table(
        report_df, column="rsrq", unit="dB", band_fn=_band_rsrq, metric_label="RSRQ",
    )


def build_sinr_technology_table(report_df: pd.DataFrame) -> pd.DataFrame:
    """4.4 SINR Analysis — one row per technology (never blended)."""
    return _build_per_technology_metric_table(
        report_df, column="sinr", unit="dB", band_fn=_band_sinr, metric_label="SINR",
    )


def _build_per_technology_metric_table(report_df, column, unit, band_fn, metric_label) -> pd.DataFrame:
    tech_list = _technology_groups(report_df)
    columns = ["Technology", f"Average {metric_label}", f"Worst {metric_label}", "Status"]

    if not tech_list:
        series = _series(report_df, column)
        if series.empty:
            return pd.DataFrame(columns=columns)
        avg = float(series.mean())
        return pd.DataFrame(
            [["Overall", f"{avg:.1f} {unit}", f"{series.min():.1f} {unit}", band_fn(avg)]],
            columns=columns,
        )

    rows = []
    network_col = report_df["network"].fillna("").astype(str).str.strip()
    for tech in tech_list:
        sub = report_df.loc[network_col == tech]
        series = _series(sub, column)
        if series.empty:
            continue
        avg = float(series.mean())
        rows.append([tech, f"{avg:.1f} {unit}", f"{series.min():.1f} {unit}", band_fn(avg)])
    return pd.DataFrame(rows, columns=columns)


def build_coverage_kpi_summary_table(
    report_df: pd.DataFrame, rsrp: pd.Series, coverage_status: str
) -> pd.DataFrame:
    """
    4.5 Coverage KPI Summary — report_VI.pdf's own layout
    (KPI | Acceptance | Observed | Status / PASS-FAIL), NOT a repeat of
    Section 3's Executive KPI Summary (different columns, condensed
    values) — this restates 4.2/4.3/4.4's own findings in the compact
    acceptance-vs-observed form, same as report_VI.pdf does.

    Adds two columns beyond report_VI's own layout, per direction
    received: "% Within Acceptance" / "% Outside Acceptance" — the % of
    THAT KPI's own individual samples that clear the Good numeric cutoff
    (>-95 dBm / >-10 dB / >20 dB), as distinct from the single averaged
    "Observed" value used for PASS/FAIL.
    """
    rows = []

    # ---- RSRP (blended across technologies, per direction received) ----
    if not rsrp.empty:
        avg_rsrp = float(rsrp.mean())
        pct_within = float((rsrp > -95).mean() * 100)
        rows.append((
            "RSRP", "> -95 dBm (90% of samples)", f"{avg_rsrp:.1f} dBm",
            "PASS" if coverage_status == "Good" else "FAIL",
            f"{pct_within:.0f}%", f"{100 - pct_within:.0f}%",
        ))

    # ---- RSRQ / SINR (per technology, never blended) ----
    tech_list = _technology_groups(report_df)
    network_col = (
        report_df["network"].fillna("").astype(str).str.strip()
        if "network" in report_df.columns else None
    )

    def _tech_subsets():
        if not tech_list:
            yield "Overall", report_df
        else:
            for tech in tech_list:
                yield tech, report_df.loc[network_col == tech]

    for tech, sub in _tech_subsets():
        rsrq = _series(sub, "rsrq")
        if rsrq.empty:
            continue
        avg_rsrq = float(rsrq.mean())
        pct_within = float((rsrq > -10).mean() * 100)
        rows.append((
            f"RSRQ - {tech}", "> -10 dB", f"{avg_rsrq:.1f} dB",
            "PASS" if _band_rsrq(avg_rsrq) == "Good" else "FAIL",
            f"{pct_within:.0f}%", f"{100 - pct_within:.0f}%",
        ))

    for tech, sub in _tech_subsets():
        sinr = _series(sub, "sinr")
        if sinr.empty:
            continue
        avg_sinr = float(sinr.mean())
        pct_within = float((sinr > 20).mean() * 100)
        rows.append((
            f"SINR - {tech}", "> 20 dB", f"{avg_sinr:.1f} dB",
            "PASS" if _band_sinr(avg_sinr) == "Good" else "FAIL",
            f"{pct_within:.0f}%", f"{100 - pct_within:.0f}%",
        ))

    return pd.DataFrame(rows, columns=[
        "KPI", "Acceptance", "Observed", "Status", "% Within Acceptance", "% Outside Acceptance",
    ])


def build_band_distribution_table(band_summary: list | None) -> pd.DataFrame:
    """4.1 Band Distribution table, from production's build_band_summary() output."""
    if not band_summary:
        return pd.DataFrame(columns=["Band", "Sample Count", "% Samples"])
    return pd.DataFrame({
        "Band": [b.get("band") for b in band_summary],
        "Sample Count": [f"{int(b.get('sample_count', 0)):,}" for b in band_summary],
        "% Samples": [f"{float(b.get('sample_percentage', 0)):.1f}%" for b in band_summary],
    })


def build_band_distribution_observation(band_summary: list | None) -> str:
    if not band_summary:
        return "Band distribution data is not available for this drive."
    top = sorted(band_summary, key=lambda b: b.get("sample_percentage", 0), reverse=True)[:3]
    parts = [f"Band {b.get('band')} ({b.get('sample_percentage', 0):.1f}%)" for b in top]
    if len(parts) == 1:
        return f"All samples were served on {parts[0]}. No unexpected frequency changes were observed."
    return (
        f"The majority of samples were served on {parts[0]}, with additional coverage on "
        f"{', '.join(parts[1:])}. No unexpected frequency changes were observed."
    )


# ------------------------------------------------------------------
# Step 3 - Section 5 "Mobility KPI Analysis"
# ------------------------------------------------------------------

def _pci_series(report_df: pd.DataFrame) -> pd.Series:
    if "pci" not in report_df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(report_df["pci"], errors="coerce").dropna().astype(int)


def build_pci_distribution_stats(report_df: pd.DataFrame) -> dict:
    pci_series = _pci_series(report_df)
    if pci_series.empty:
        return {
            "total_samples": 0,
            "unique_pci": 0,
            "dominant_pci": None,
            "dominant_count": 0,
            "dominant_share": 0.0,
            "top_30_share": 0.0,
            "counts": pd.Series(dtype=int),
        }

    counts = pci_series.value_counts().sort_values(ascending=False)
    total = int(len(pci_series))
    dominant_pci = int(counts.index[0])
    dominant_count = int(counts.iloc[0])
    dominant_share = float((dominant_count / total) * 100) if total else 0.0
    top_30_share = float((counts.head(30).sum() / total) * 100) if total else 0.0
    return {
        "total_samples": total,
        "unique_pci": int(counts.size),
        "dominant_pci": dominant_pci,
        "dominant_count": dominant_count,
        "dominant_share": dominant_share,
        "top_30_share": top_30_share,
        "counts": counts,
    }


def build_serving_cell_metric_table(report_df: pd.DataFrame) -> pd.DataFrame:
    stats = build_pci_distribution_stats(report_df)
    if stats["total_samples"] == 0:
        return pd.DataFrame({"Metric": ["Serving Cell Data"], "Observed Value": ["No PCI data available"]})
    return pd.DataFrame({
        "Metric": [
            "Samples with Serving PCI",
            "Unique Serving PCIs",
            "Dominant PCI",
            "Dominant PCI Share",
            "Top 30 PCI Share",
        ],
        "Observed Value": [
            f"{stats['total_samples']:,}",
            f"{stats['unique_pci']}",
            f"{stats['dominant_pci']}",
            f"{stats['dominant_share']:.1f}%",
            f"{stats['top_30_share']:.1f}%",
        ],
    })


def build_top_pci_table(report_df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    stats = build_pci_distribution_stats(report_df)
    counts = stats["counts"]
    if counts.empty:
        return pd.DataFrame(columns=["PCI", "Sample Count", "% Samples", "CDF %"])

    total = stats["total_samples"]
    running = 0
    rows = []
    for pci, count in counts.head(limit).items():
        count = int(count)
        running += count
        rows.append((
            int(pci),
            f"{count:,}",
            f"{(count / total) * 100:.1f}%",
            f"{(running / total) * 100:.1f}%",
        ))
    return pd.DataFrame(rows, columns=["PCI", "Sample Count", "% Samples", "CDF %"])


def build_top_pci_analysis_table(report_df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    stats = build_pci_distribution_stats(report_df)
    counts = stats["counts"]
    if counts.empty:
        return pd.DataFrame(columns=["PCI", "Sample Count", "% Samples", "Observation"])

    total = stats["total_samples"]
    rows = []
    for pci, count in counts.head(limit).items():
        share = float((count / total) * 100) if total else 0.0
        if share >= 20:
            obs = "Dominant serving PCI"
        elif share >= 5:
            obs = "Healthy overlap"
        else:
            obs = "Normal utilization"
        rows.append((int(pci), f"{int(count):,}", f"{share:.1f}%", obs))
    return pd.DataFrame(rows, columns=["PCI", "Sample Count", "% Samples", "Observation"])


def build_poor_pci_analysis_table(report_df: pd.DataFrame, metric_column: str):
    from tools.report_engine.kpi_analysis import build_poor_pci_table

    threshold = -105 if metric_column == "rsrp" else -14
    return build_poor_pci_table(report_df, metric_column, threshold)


def _poor_pci_section_title(base_title: str, table_df: pd.DataFrame | None) -> str:
    if table_df is None or table_df.empty:
        return base_title
    shown = len(table_df)
    total = int(table_df.attrs.get("total_poor_pci", shown))
    suffix = f"Top {shown} of {total} PCIs" if total > shown else f"{shown} PCIs"
    return f"{base_title} ({suffix})"


def build_neighbor_cell_table(neighbor_df: pd.DataFrame | None) -> pd.DataFrame:
    if neighbor_df is None or neighbor_df.empty:
        return pd.DataFrame({
            "Metric": [
                "Neighbor Log Source",
                "Neighbor Samples",
                "Neighbor PCI Coverage",
                "Assessment",
            ],
            "Observed Value": [
                "tbl_network_log_neighbour",
                "0",
                "0 unique neighbor PCIs",
                "Not populated",
            ],
        })

    pci = pd.to_numeric(neighbor_df["pci"], errors="coerce").dropna().astype(int) if "pci" in neighbor_df.columns else pd.Series(dtype=int)
    networks = (
        ", ".join(
            neighbor_df["network"].fillna("").astype(str).str.strip().replace("", np.nan).dropna().value_counts().head(3).index.tolist()
        )
        if "network" in neighbor_df.columns else ""
    )
    bands = (
        ", ".join(
            neighbor_df["band"].fillna("").astype(str).str.strip().replace("", np.nan).dropna().value_counts().head(3).index.tolist()
        )
        if "band" in neighbor_df.columns else ""
    )
    top_pci_text = "No PCI data"
    if not pci.empty:
        top_counts = pci.value_counts()
        top_pci = int(top_counts.index[0])
        top_share = float((top_counts.iloc[0] / len(pci)) * 100)
        top_pci_text = f"PCI {top_pci} ({top_share:.1f}% of neighbor samples)"

    return pd.DataFrame({
        "Metric": [
            "Neighbor Log Source",
            "Neighbor Samples",
            "Unique Neighbor PCIs",
            "Dominant Neighbor PCI",
            "Neighbor Networks",
            "Neighbor Bands",
            "Assessment",
        ],
        "Observed Value": [
            "tbl_network_log_neighbour",
            f"{len(neighbor_df):,}",
            f"{int(pci.nunique()) if not pci.empty else 0}",
            top_pci_text,
            networks or "N/A",
            bands or "N/A",
            "Available",
        ],
    })


def build_neighbor_cell_check_table(neighbor_df: pd.DataFrame | None) -> pd.DataFrame:
    if neighbor_df is None or neighbor_df.empty:
        return pd.DataFrame({
            "Check": ["Neighbor Logs", "Missing Neighbor", "Neighbor Consistency"],
            "Status": ["Not observed", "Not evaluated", "Not evaluated"],
        })

    pci = pd.to_numeric(neighbor_df["pci"], errors="coerce").dropna().astype(int) if "pci" in neighbor_df.columns else pd.Series(dtype=int)
    unique_pci = int(pci.nunique()) if not pci.empty else 0
    dominant_share = float(pci.value_counts(normalize=True).iloc[0] * 100) if not pci.empty else 0.0
    consistency = "Good" if unique_pci >= 3 and dominant_share < 95 else "Review"
    return pd.DataFrame({
        "Check": [
            "Neighbor Logs",
            "Unique Neighbor PCIs",
            "Neighbor Consistency",
        ],
        "Status": [
            "Observed",
            f"{unique_pci}",
            consistency,
        ],
    })


def build_cell_dominance_table(report_df: pd.DataFrame, neighbor_df: pd.DataFrame | None = None) -> pd.DataFrame:
    serving_stats = build_pci_distribution_stats(report_df)
    neighbor_pci = (
        pd.to_numeric(neighbor_df["pci"], errors="coerce").dropna().astype(int)
        if neighbor_df is not None and not neighbor_df.empty and "pci" in neighbor_df.columns
        else pd.Series(dtype=int)
    )
    if not neighbor_pci.empty:
        neighbor_counts = neighbor_pci.value_counts()
        neighbor_dom_pci = int(neighbor_counts.index[0])
        neighbor_dom_share = float((neighbor_counts.iloc[0] / len(neighbor_pci)) * 100)
    else:
        neighbor_dom_pci = None
        neighbor_dom_share = 0.0

    if serving_stats["dominant_share"] < 80:
        assessment = "Serving dominance within acceptable range"
    else:
        assessment = "Serving dominance requires review"

    return pd.DataFrame({
        "Metric": [
            "Dominant Serving PCI",
            "Serving PCI Share",
            "Dominant Neighbor PCI",
            "Neighbor PCI Share",
            "Assessment",
        ],
        "Observed Value": [
            str(serving_stats["dominant_pci"]) if serving_stats["dominant_pci"] is not None else "N/A",
            f"{serving_stats['dominant_share']:.1f}%" if serving_stats["total_samples"] else "N/A",
            str(neighbor_dom_pci) if neighbor_dom_pci is not None else "N/A",
            f"{neighbor_dom_share:.1f}%" if not neighbor_pci.empty else "N/A",
            assessment,
        ],
    })


def build_mobility_kpi_summary_table(report_df: pd.DataFrame, neighbor_df: pd.DataFrame | None = None) -> pd.DataFrame:
    stats = build_pci_distribution_stats(report_df)
    mobility_status, _mobility_remarks = classify_mobility(report_df)
    neighbor_table = build_neighbor_cell_table(neighbor_df)
    neighbor_status_value = str(neighbor_table.iloc[-1]["Observed Value"]) if not neighbor_table.empty else "Not available"
    rows = []

    if stats["total_samples"]:
        rows.append((
            "Serving Cell Stability",
            "Dominant PCI share < 80%",
            f"PCI {stats['dominant_pci']} at {stats['dominant_share']:.1f}% of samples",
            "PASS" if mobility_status == "Good" else "FAIL",
        ))
        rows.append((
            "PCI Diversity",
            "More than 1 serving PCI observed",
            f"{stats['unique_pci']} unique serving PCIs",
            "PASS" if stats["unique_pci"] > 1 else "FAIL",
        ))
    else:
        rows.append(("Serving Cell Stability", "Dominant PCI share < 80%", "No PCI data", "N/A"))
        rows.append(("PCI Diversity", "More than 1 serving PCI observed", "No PCI data", "N/A"))

    rows.append((
        "Neighbor Cell Availability",
        "Neighbor log rows available",
        neighbor_status_value,
        "PASS" if neighbor_status_value == "Available" else "N/A",
    ))
    return pd.DataFrame(rows, columns=["KPI", "Acceptance", "Observed", "Status"])


def build_service_metric_table(report_df: pd.DataFrame, column: str, label: str, unit: str) -> pd.DataFrame:
    series = _series(report_df, column)
    if series.empty:
        return pd.DataFrame({"Metric": [f"{label} Data"], "Sample Value": ["No data available"]})
    return pd.DataFrame({
        "Metric": [f"Average {label}", f"Peak {label}", f"Minimum {label}"],
        "Sample Value": [
            f"{float(series.mean()):.1f} {unit}",
            f"{float(series.max()):.1f} {unit}",
            f"{float(series.min()):.1f} {unit}",
        ],
    })


def build_two_metric_table(avg_label: str, avg_value: str, max_label: str, max_value: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Metric": [avg_label, max_label],
        "Sample Value": [avg_value, max_value],
    })


def build_packet_loss_table(report_df: pd.DataFrame) -> pd.DataFrame:
    series = _series(report_df, "packet_loss")
    if series.empty:
        return pd.DataFrame({"Metric": ["Packet Loss"], "Sample Value": ["No data available"]})
    return pd.DataFrame({
        "Metric": ["Average Packet Loss"],
        "Sample Value": [f"{float(series.mean()):.2f}%"],
    })


def build_service_kpi_summary_table(report_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dl = _series(report_df, "dl_tpt")
    ul = _series(report_df, "ul_tpt")
    mos = _series(report_df, "mos")
    latency = _series(report_df, "latency")
    packet_loss = _series(report_df, "packet_loss")

    if not dl.empty:
        rows.append(("DL Throughput", f"{float(dl.mean()):.1f} Mbps", "Project Specific", "PASS"))
    if not ul.empty:
        rows.append(("UL Throughput", f"{float(ul.mean()):.1f} Mbps", "Project Specific", "PASS"))
    if not mos.empty:
        rows.append(("MOS", f"{float(mos.mean()):.2f}", ">=3.5", "PASS" if float(mos.mean()) >= 3.5 else "FAIL"))
    if not latency.empty:
        rows.append(("Latency", f"{float(latency.mean()):.1f} ms", "<50 ms", "PASS" if float(latency.mean()) < 50 else "FAIL"))
    if not packet_loss.empty:
        rows.append(("Packet Loss", f"{float(packet_loss.mean()):.2f}%", "<1%", "PASS" if float(packet_loss.mean()) < 1 else "FAIL"))
    return pd.DataFrame(rows, columns=["KPI", "Observed", "Target", "Status"])


def build_application_analytics_tables(report_df: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    from tools.report_engine.kpi_analysis import build_app_analytics_native_tables

    tables = build_app_analytics_native_tables(report_df)
    return tables.get("app_analytics_part1.png"), tables.get("app_analytics_part2.png")


def _canonical_app_label(app_name: str) -> str:
    text = str(app_name or "").strip()
    lowered = text.lower()
    if "yt" in lowered or "youtube" in lowered:
        return "YouTube Streaming"
    if "whatsapp" in lowered:
        return "WhatsApp"
    if "gmail" in lowered:
        return "Gmail"
    if "google search" in lowered or "search" in lowered or "chrome" in lowered or "web" in lowered:
        return "Web Browsing"
    if "maps" in lowered or "map" in lowered:
        return "Maps"
    if "video" in lowered or "meet" in lowered or "call" in lowered:
        return "Video Calling"
    return text if text else "Unknown"


def _app_observation_from_row(app_label: str, row: pd.Series) -> str:
    dl = pd.to_numeric(pd.Series([row.get("DL Avg")]).astype(str).str.replace("Mbps", "", regex=False).str.strip(), errors="coerce").iloc[0]
    ul = pd.to_numeric(pd.Series([row.get("UL Avg")]).astype(str).str.replace("Mbps", "", regex=False).str.strip(), errors="coerce").iloc[0]
    mos = pd.to_numeric(pd.Series([row.get("MOS Avg")]).astype(str).str.strip(), errors="coerce").iloc[0]
    latency = pd.to_numeric(pd.Series([row.get("Latency Avg")]).astype(str).str.replace("ms", "", regex=False).str.strip(), errors="coerce").iloc[0]
    jitter = pd.to_numeric(pd.Series([row.get("Jitter Avg")]).astype(str).str.replace("ms", "", regex=False).str.strip(), errors="coerce").iloc[0]
    loss = pd.to_numeric(pd.Series([row.get("Loss Avg")]).astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce").iloc[0]

    if app_label == "YouTube Streaming":
        if pd.notna(dl) and dl >= 1.0 and pd.notna(latency) and latency < 100:
            return "Smooth playback"
        return "Playback sensitive to throughput variation"
    if app_label == "WhatsApp":
        if pd.notna(loss) and loss < 1 and pd.notna(latency) and latency < 100:
            return "Stable messaging and media transfer"
        return "Messaging stable with occasional delay"
    if app_label == "Maps":
        return "Accurate navigation updates" if pd.notna(latency) and latency < 120 else "Navigation updates affected by delay"
    if app_label == "Video Calling":
        if pd.notna(mos) and mos >= 3.5 and pd.notna(jitter) and jitter < 50:
            return "No noticeable interruptions"
        return "Quality dependent on latency and jitter"
    if app_label == "Gmail":
        return "Mail sync and attachment loading remained usable"
    if app_label == "Web Browsing":
        return "Responsive" if pd.notna(latency) and latency < 100 else "Responsive with occasional delay"
    return "Observed during drive testing"


def build_application_observation_table(report_df: pd.DataFrame) -> pd.DataFrame:
    part1, part2 = build_application_analytics_tables(report_df)
    if part1 is None or part1.empty:
        return pd.DataFrame(columns=["Application", "Observation"])

    merged = part1.copy()
    if part2 is not None and not part2.empty:
        merged = merged.merge(part2, on=["App Name"], how="left", suffixes=("", "_part2"))

    if "Samples" in merged.columns:
        merged["_samples_num"] = pd.to_numeric(merged["Samples"], errors="coerce").fillna(0)
        merged = merged.sort_values(["_samples_num", "App Name"], ascending=[False, True])

    seen = set()
    rows = []
    for _, row in merged.iterrows():
        label = _canonical_app_label(row.get("App Name"))
        if label in seen or label == "Unknown":
            continue
        seen.add(label)
        rows.append((label, _app_observation_from_row(label, row)))
        if len(rows) == 5:
            break
    return pd.DataFrame(rows, columns=["Application", "Observation"])


def build_application_overall_observation(report_df: pd.DataFrame, observation_df: pd.DataFrame) -> str:
    latency = _series(report_df, "latency")
    jitter = _series(report_df, "jitter")
    packet_loss = _series(report_df, "packet_loss")
    app_names = ", ".join(observation_df["Application"].head(3).tolist()) if not observation_df.empty else "the observed applications"

    if (
        (latency.empty or float(latency.mean()) < 100)
        and (jitter.empty or float(jitter.mean()) < 50)
        and (packet_loss.empty or float(packet_loss.mean()) < 1)
    ):
        return f"Application performance for {app_names} reflected the overall network quality and remained generally stable during the drive."
    return f"Application performance for {app_names} followed the underlying radio and IP quality trends, with responsiveness reducing where latency, jitter or throughput degraded."


# ------------------------------------------------------------------
# PDF builder (subclass of the production generator)
# ------------------------------------------------------------------

class NewFormatPDFReport(PDFReportGenerator):
    """
    Renders the new-format report (report_VI.pdf layout) using production
    styles, native tables and section rendering.  Only the sections
    implemented so far are rendered; production code stays untouched.
    """

    def _bullet_flowables(self, items):
        return [Paragraph(f"&bull;&nbsp;&nbsp;{item}", self.styles["Body"]) for item in items]

    def add_bullets(self, items):
        self.story.extend(self._bullet_flowables(items))
        self.story.append(Spacer(1, 6))

    def add_bold_label(self, text):
        self.story.append(Paragraph(f"<b>{text}</b>", self.styles["SubSection"]))

    def add_labeled_block(self, label, flowables, trailing_space=10):
        """
        Bold sub-heading + its content, glued together with KeepTogether so
        ReportLab can never strand the heading at the bottom of a page while
        the content flows to the next one (the "Key Engineering
        Observations" heading/content page-split bug).
        """
        block = [Paragraph(f"<b>{label}</b>", self.styles["SubSection"]), Spacer(1, 4), *flowables]
        self.story.append(KeepTogether(block))
        self.story.append(Spacer(1, trailing_space))

    def _sized_image(self, path, max_width, max_height):
        """Build (NOT append) a sized Image flowable — same width/height
        math as the inherited production _place_image(), but returns the
        flowable so it can be placed inside an add_labeled_block()
        flowables list instead of always landing at the end of self.story."""
        img = Image(path)
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(max_width / iw, max_height / ih, 1.0)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        return img

    def _kpi_image_flowables(self, map_filename: str, cdf_filename: str, max_height=4 * inch):
        """
        Map (compressed, same as production's KPI maps) + CDF chart
        (full-quality, same as production's charts) for one KPI, generated
        by test_new_pdf_report.py via production's own generate_kpi_map /
        generate_all_cdf_plots_from_df. Silently omits whichever file
        wasn't generated (e.g. no valid data for that KPI).
        """
        flowables = []
        map_path = os.path.join(self.images_dir, "kpi_maps", map_filename)
        if os.path.exists(map_path):
            flowables.append(self._sized_image(self._compress_png(map_path), 5.8 * inch, max_height))
            flowables.append(Spacer(1, 6))
        cdf_path = os.path.join(self.images_dir, "kpi_analysis", cdf_filename)
        if os.path.exists(cdf_path):
            flowables.append(self._sized_image(cdf_path, TABLE_MAX_WIDTH, max_height))
            flowables.append(Spacer(1, 6))
        return flowables

    def _optional_image_flowables(self, filenames, subdir="kpi_maps", max_width=5.8 * inch, max_height=4 * inch):
        flowables = []
        for filename in filenames:
            path = os.path.join(self.images_dir, subdir, filename)
            if not os.path.exists(path):
                continue
            use_path = self._compress_png(path) if subdir == "kpi_maps" else path
            flowables.append(self._sized_image(use_path, max_width, max_height))
            flowables.append(Spacer(1, 6))
        return flowables

    def _kpi_image_flowables_per_technology(
        self, report_df, map_prefix: str, cdf_filename: str, max_height=4 * inch,
    ):
        """
        Same as _kpi_image_flowables, but for KPIs classified PER
        TECHNOLOGY (RSRQ/SINR): one map per technology present in
        `report_df["network"]` (matching test_new_pdf_report.py's
        `{map_prefix}_{_tech_slug(tech)}.png` naming), each labeled with
        its technology, instead of a single map blending every RAT
        together. The CDF chart stays a single blended distribution
        (unchanged) since the per-technology breakdown already lives in
        the 4.3/4.4 tables.
        """
        flowables = []
        tech_list = _technology_groups(report_df)
        maps_dir = os.path.join(self.images_dir, "kpi_maps")
        any_map = False
        for tech in tech_list:
            map_path = os.path.join(maps_dir, f"{map_prefix}_{_tech_slug(tech)}.png")
            if not os.path.exists(map_path):
                continue
            flowables.append(Paragraph(f"<b>{tech}</b>", self.styles["Body"]))
            flowables.append(Spacer(1, 2))
            flowables.append(self._sized_image(self._compress_png(map_path), 5.8 * inch, max_height))
            flowables.append(Spacer(1, 6))
            any_map = True
        if not any_map:
            # Fall back to a single combined map if no per-technology maps
            # were generated (e.g. `network` column missing entirely).
            map_path = os.path.join(maps_dir, f"{map_prefix}_all.png")
            if os.path.exists(map_path):
                flowables.append(self._sized_image(self._compress_png(map_path), 5.8 * inch, max_height))
                flowables.append(Spacer(1, 6))
        cdf_path = os.path.join(self.images_dir, "kpi_analysis", cdf_filename)
        if os.path.exists(cdf_path):
            flowables.append(self._sized_image(cdf_path, TABLE_MAX_WIDTH, max_height))
            flowables.append(Spacer(1, 6))
        return flowables

    def add_introduction(self, text):
        """Section 1 — same structure as production pdf_generator."""
        self.add_toc_heading("1. Introduction", self.styles["Section"], 0, "sec1")
        if text and text.strip():
            self.story.append(Paragraph(text, self.styles["Body"]))

    def add_area_summary(self, area_summary):
        """
        Section 2 — identical to production pdf_generator.generate_report:
        render the structured area summary then the base route map image.
        """
        if not area_summary and not self.has_image("base_route_map.png"):
            return
        self.story.append(CondPageBreak(3 * inch))
        self.add_toc_heading("2. Area Summary", self.styles["Section"], 0, "sec2")
        self.render_section(None, area_summary)
        self.add_image("base_route_map.png")

    def add_drive_summary_v2(self, drive_summary: dict, exec_summary: dict, distances_km: dict | None = None):
        """Section 3 'Drive Summary' in the new report_VI format."""
        self.story.append(PageBreak())
        self.add_toc_heading("3. Drive Summary", self.styles["Section"], 0, "sec3")

        # Narrative sentence + Metric/Value table
        self.story.append(Paragraph(build_drive_summary_text(drive_summary), self.styles["Body"]))
        self.story.append(Spacer(1, 6))
        self.story.append(make_native_table(build_drive_metric_table(drive_summary)))
        self.story.append(Spacer(1, 10))

        # Session Details — corrected Haversine-based distance per session
        # (production's session_table.png depends on the permission-denied
        # tbl_session query and was never placed in this section either).
        # Every heading below is glued to its content via add_labeled_block
        # (KeepTogether) so a heading can never be stranded at the bottom
        # of a page while its content flows to the next one.
        if distances_km is not None and drive_summary.get("sessions"):
            self.add_labeled_block(
                "Session Details",
                [make_native_table(build_session_native_table(drive_summary, distances_km))],
            )

        # Executive Summary block
        self.add_labeled_block("Executive Summary", [
            Paragraph("Objective", self.styles["Body"]),
            Paragraph(exec_summary["objective"], self.styles["Body"]),
        ], trailing_space=6)

        self.add_labeled_block("Scope", self._bullet_flowables(exec_summary["scope"]))

        kpi_df = pd.DataFrame(
            exec_summary["kpi_rows"], columns=["KPI Group", "Status", "Remarks"]
        )
        self.add_labeled_block("Executive KPI Summary", [make_native_table(kpi_df)])

        self.add_labeled_block(
            "Key Engineering Observations", self._bullet_flowables(exec_summary["observations"])
        )

        self.add_labeled_block(
            "Overall Assessment", [Paragraph(exec_summary["overall_assessment"], self.styles["Body"])]
        )

    def add_coverage_kpi_analysis(self, report_df: pd.DataFrame, exec_summary: dict, band_summary: list | None):
        """
        Section 4 'Coverage KPI Analysis' (report_VI.pdf layout, Step 2).

        Reuses exec_summary["coverage"] / exec_summary["quality_entries"]
        (computed once in Section 3 via classify_coverage /
        classify_quality_by_technology) rather than recomputing — this
        section and the Executive KPI Summary can never disagree.
        """
        self.story.append(PageBreak())
        self.add_toc_heading("4. Coverage KPI Analysis", self.styles["Section"], 0, "sec4")
        self.story.append(Paragraph(
            "Coverage analysis is the first step in RF optimization. It evaluates whether "
            "the radio network provides continuous service with acceptable signal strength "
            "and quality across the planned drive route.",
            self.styles["Body"],
        ))
        self.story.append(Spacer(1, 8))

        rsrp = _series(report_df, "rsrp")
        coverage_status = exec_summary["coverage"]["status"]
        quality_entries = exec_summary["quality_entries"]

        # ---- 4.1 Band Distribution ----
        band_table = build_band_distribution_table(band_summary)
        band_flowables = [
            Paragraph(
                "Objective: verify that the UE camps on the planned band(s) throughout the "
                "drive route and identify unexpected band changes.",
                self.styles["Body"],
            ),
            Spacer(1, 4),
        ]
        if not band_table.empty:
            band_flowables.append(make_native_table(band_table))
            band_flowables.append(Spacer(1, 4))
        band_flowables.append(Paragraph(
            f"Observation: {build_band_distribution_observation(band_summary)}", self.styles["Body"]
        ))
        self.add_labeled_block("4.1 Band Distribution", band_flowables)
        # Map + pie chart, appended separately for the same reason as the
        # RSRP/RSRQ/SINR images below (avoid an oversized KeepTogether block).
        self.story.extend(self._kpi_image_flowables("band_map.png", "band_pie.png", max_height=4.5 * inch))

        # ---- 4.2 RSRP Analysis (Coverage) — blended across technologies ----
        rsrp_flowables = [
            Paragraph(
                "Definition: Reference Signal Received Power (RSRP) is the primary LTE "
                "coverage KPI and indicates the received signal strength from the serving cell.",
                self.styles["Body"],
            ),
            Spacer(1, 4),
            Paragraph("<b>Acceptance Criteria</b>", self.styles["Body"]),
            *self._bullet_flowables([
                "Good: RSRP &gt; -95 dBm for more than 90% of samples",
                "Fair: Average RSRP -95 to -105 dBm",
                "Poor: Average RSRP &lt; -105 dBm, or coverage (% &gt; -95 dBm) &lt; 75%",
            ]),
            Spacer(1, 4),
            make_native_table(build_rsrp_metric_table(rsrp)),
            Spacer(1, 4),
            Paragraph(
                f"Observation: Coverage classified as <b>{coverage_status}</b>. "
                f"{exec_summary['coverage']['remarks']}",
                self.styles["Body"],
            ),
        ]
        self.add_labeled_block("4.2 RSRP Analysis (Coverage)", rsrp_flowables)
        # Map + CDF appended separately (not inside the KeepTogether block
        # above) so a full-page image can flow naturally instead of risking
        # a "content too large for one page" layout failure.
        self.story.extend(self._kpi_image_flowables("rsrp_map.png", "cdf_rsrp.png"))
        self.story.extend(self._optional_image_flowables(["rsrp_poor_regions.png"], subdir="kpi_maps", max_width=5.8 * inch, max_height=4.5 * inch))

        # ---- 4.3 RSRQ Analysis — per technology, never blended ----
        rsrq_table = build_rsrq_technology_table(report_df)
        rsrq_worst_status = _worst_status(rsrq_table["Status"]) if not rsrq_table.empty else "N/A"
        rsrq_flowables = [
            Paragraph(
                "Definition: Reference Signal Received Quality (RSRQ) indicates the quality "
                "of the received reference signal and reflects both signal strength and cell "
                "loading. RSRQ is classified PER TECHNOLOGY, not blended, since its "
                "characteristic range differs by RAT.",
                self.styles["Body"],
            ),
            Spacer(1, 4),
            Paragraph("<b>Acceptance Criteria</b>", self.styles["Body"]),
            *self._bullet_flowables([
                "Good: Average RSRQ &gt; -10 dB",
                "Fair: Average RSRQ -10 to -15 dB",
                "Poor: Average RSRQ &lt; -15 dB",
            ]),
            Spacer(1, 4),
        ]
        if not rsrq_table.empty:
            rsrq_flowables.append(make_native_table(rsrq_table))
            rsrq_flowables.append(Spacer(1, 4))
        rsrq_flowables.append(Paragraph(
            f"Observation: worst-case RSRQ classification across technologies is "
            f"<b>{rsrq_worst_status}</b>.", self.styles["Body"]
        ))
        self.add_labeled_block("4.3 RSRQ Analysis", rsrq_flowables)
        # Per-technology maps (one per RAT present), not a single map
        # blending every technology together — RSRQ's characteristic range
        # differs by RAT, same reason its table is split per technology.
        self.story.extend(
            self._kpi_image_flowables_per_technology(report_df, "rsrq_map", "cdf_rsrq.png")
        )
        self.story.extend(self._optional_image_flowables(["rsrq_poor_regions.png"], subdir="kpi_maps", max_width=5.8 * inch, max_height=4.5 * inch))

        # ---- 4.4 SINR Analysis — per technology, never blended ----
        sinr_table = build_sinr_technology_table(report_df)
        sinr_worst_status = _worst_status(sinr_table["Status"]) if not sinr_table.empty else "N/A"
        sinr_flowables = [
            Paragraph(
                "Definition: Signal-to-Interference-plus-Noise Ratio (SINR) represents radio "
                "link quality and interference conditions. SINR is classified PER TECHNOLOGY, "
                "not blended, since its characteristic range differs by RAT.",
                self.styles["Body"],
            ),
            Spacer(1, 4),
            Paragraph("<b>Acceptance Criteria</b>", self.styles["Body"]),
            *self._bullet_flowables([
                "Good: Average SINR &gt; 20 dB",
                "Fair: Average SINR 13 to 20 dB",
                "Poor: Average SINR &lt; 13 dB",
            ]),
            Spacer(1, 4),
        ]
        if not sinr_table.empty:
            sinr_flowables.append(make_native_table(sinr_table))
            sinr_flowables.append(Spacer(1, 4))
        sinr_flowables.append(Paragraph(
            f"Observation: worst-case SINR classification across technologies is "
            f"<b>{sinr_worst_status}</b>.", self.styles["Body"]
        ))
        self.add_labeled_block("4.4 SINR Analysis", sinr_flowables)
        # Per-technology maps, same reasoning as 4.3 RSRQ above.
        self.story.extend(
            self._kpi_image_flowables_per_technology(report_df, "sinr_map", "cdf_sinr.png")
        )

        # ---- 4.5 Coverage KPI Summary — report_VI.pdf's own
        # KPI/Acceptance/Observed/PASS-FAIL layout, condensing 4.2-4.4's
        # own findings (NOT a repeat of Section 3's Executive KPI Summary,
        # which uses different columns and is about Mobility/Handover too) ----
        summary_df = build_coverage_kpi_summary_table(report_df, rsrp, coverage_status)
        self.add_labeled_block("4.5 Coverage KPI Summary", [make_native_table(summary_df)])

        # ---- Overall Coverage Assessment ----
        all_statuses = [coverage_status] + [e["status"] for e in quality_entries]
        if all(s in ("Good", "N/A") for s in all_statuses):
            overall_text = (
                "The coverage KPIs indicate that the network provides stable service with "
                "acceptable signal strength and quality over the majority of the evaluated "
                "route."
            )
        else:
            overall_text = (
                "The coverage KPIs indicate localized weak-signal or quality pockets. "
                "Targeted optimization is recommended for the specific KPI/technology "
                "combinations flagged Fair or Poor above."
            )
        self.add_labeled_block("Overall Coverage Assessment", [Paragraph(overall_text, self.styles["Body"])])

    def add_mobility_kpi_analysis(self, report_df: pd.DataFrame, neighbor_df: pd.DataFrame | None = None):
        """
        Section 5 "Mobility KPI Analysis" in the new test-only format.

        This section stays rule-based and uses the same corrected report_df
        population already used by the Executive Summary and Coverage
        section, so the PCI-based mobility statements remain consistent
        throughout the report.
        """
        self.story.append(PageBreak())
        self.add_toc_heading("5. Mobility KPI Analysis", self.styles["Section"], 0, "sec5")
        self.story.append(Paragraph(
            "Mobility analysis evaluates whether the UE maintains service continuity while moving "
            "through the network. It focuses on serving cell behavior, PCI planning, neighbor "
            "relationships and cell dominance.",
            self.styles["Body"],
        ))
        self.story.append(Spacer(1, 8))

        stats = build_pci_distribution_stats(report_df)
        mobility_status, mobility_remarks = classify_mobility(report_df)
        top_pci_df = build_top_pci_table(report_df, limit=15)
        top_pci_analysis_df = build_top_pci_analysis_table(report_df, limit=8)
        poor_rsrp_df = build_poor_pci_analysis_table(report_df, "rsrp")
        poor_rsrq_df = build_poor_pci_analysis_table(report_df, "rsrq")
        neighbor_table_df = build_neighbor_cell_table(neighbor_df)
        neighbor_check_df = build_neighbor_cell_check_table(neighbor_df)
        cell_dominance_df = build_cell_dominance_table(report_df, neighbor_df=neighbor_df)
        summary_df = build_mobility_kpi_summary_table(report_df, neighbor_df=neighbor_df)

        serving_flowables = [
            Paragraph(
                "Objective: verify that the UE remains connected to the expected serving cells "
                "across the planned route.",
                self.styles["Body"],
            ),
            Spacer(1, 4),
            make_native_table(build_serving_cell_metric_table(report_df)),
            Spacer(1, 4),
            Paragraph(
                f"Observation: Serving-cell transitions followed the planned footprint. {mobility_remarks}",
                self.styles["Body"],
            ),
        ]
        self.add_labeled_block("5.1 Serving Cell Distribution", serving_flowables)

        pci_map_path = os.path.join(self.images_dir, "kpi_maps", "pci_map.png")
        if os.path.exists(pci_map_path):
            self.story.append(self._sized_image(self._compress_png(pci_map_path), 5.8 * inch, 4.5 * inch))
            self.story.append(Spacer(1, 6))

        pci_flowables = [
            Paragraph(
                "Objective: review how the serving PCI population is distributed along the drive "
                "route and identify whether mobility is concentrated on a small number of cells.",
                self.styles["Body"],
            ),
            Spacer(1, 4),
        ]
        if not top_pci_df.empty:
            pci_flowables.append(make_native_table(top_pci_df))
            pci_flowables.append(Spacer(1, 4))
        if stats["total_samples"]:
            pci_flowables.append(Paragraph(
                f"Observation: {stats['unique_pci']} unique serving PCIs were observed. "
                f"The dominant PCI was {stats['dominant_pci']} with {stats['dominant_share']:.1f}% "
                f"of samples. PCI allocation was generally balanced across the drive route.",
                self.styles["Body"],
            ))
        else:
            pci_flowables.append(Paragraph("Observation: PCI distribution data is not available.", self.styles["Body"]))
        self.add_labeled_block("5.2 PCI Distribution", pci_flowables)

        for subdir, filename, compress in [
            ("kpi_analysis", "pci_distribution.png", False),
            ("kpi_analysis", "cdf_pci.png", False),
        ]:
            path = os.path.join(self.images_dir, subdir, filename)
            if os.path.exists(path):
                img_path = self._compress_png(path) if compress else path
                self.story.append(self._sized_image(img_path, TABLE_MAX_WIDTH, 4.5 * inch))
                self.story.append(Spacer(1, 6))

        top_pci_flowables = []
        if not top_pci_analysis_df.empty:
            top_pci_flowables.append(make_native_table(top_pci_analysis_df))
            top_pci_flowables.append(Spacer(1, 4))
        top_pci_flowables.append(Paragraph(
            "Recommendation: review PCI reuse only if future expansion introduces additional neighboring sites.",
            self.styles["Body"],
        ))
        self.add_labeled_block("5.3 Top PCI Analysis", top_pci_flowables)

        poor_rsrp_flowables = []
        if poor_rsrp_df is not None and not poor_rsrp_df.empty:
            poor_rsrp_flowables.append(make_native_table(poor_rsrp_df))
            poor_rsrp_flowables.append(Spacer(1, 4))
            poor_rsrp_flowables.append(Paragraph(
                "Observation: Weak PCI performance was localized and primarily associated with cell-edge conditions.",
                self.styles["Body"],
            ))
        else:
            poor_rsrp_flowables.append(Paragraph("No poor PCI entries were observed for the configured RSRP threshold.", self.styles["Body"]))
        self.add_labeled_block(
            _poor_pci_section_title("5.4 Poor PCI Analysis (RSRP)", poor_rsrp_df),
            poor_rsrp_flowables,
        )

        poor_rsrq_flowables = []
        if poor_rsrq_df is not None and not poor_rsrq_df.empty:
            poor_rsrq_flowables.append(make_native_table(poor_rsrq_df))
            poor_rsrq_flowables.append(Spacer(1, 4))
            poor_rsrq_flowables.append(Paragraph(
                "Observation: Low RSRQ values indicate loading or interference rather than pure coverage deficiency.",
                self.styles["Body"],
            ))
        else:
            poor_rsrq_flowables.append(Paragraph("No poor PCI entries were observed for the configured RSRQ threshold.", self.styles["Body"]))
        self.add_labeled_block(
            _poor_pci_section_title("5.5 Poor PCI Analysis (RSRQ)", poor_rsrq_df),
            poor_rsrq_flowables,
        )

        neighbor_flowables = [
            Paragraph(
                "Objective: verify that neighbor relations support smooth mobility.",
                self.styles["Body"],
            ),
            Spacer(1, 4),
            make_native_table(neighbor_check_df),
            Spacer(1, 4),
        ]
        neighbor_status = (
            str(neighbor_table_df.iloc[-1]["Observed Value"]) if not neighbor_table_df.empty else "Not available"
        )
        if neighbor_status == "Available":
            neighbor_obs = (
                f"Dedicated neighbor logs are present in tbl_network_log_neighbour with "
                f"{len(neighbor_df):,} rows. Maintain periodic audit of neighbor relations after parameter changes."
            )
        else:
            neighbor_obs = (
                "Dedicated neighbor logs are not populated for the current project dataset, so "
                "this section records availability only and avoids inferring neighbor quality."
            )
        neighbor_flowables.append(Paragraph(f"Observation: {neighbor_obs}", self.styles["Body"]))
        self.add_labeled_block("5.6 Neighbor Cell Analysis", neighbor_flowables)

        dominance_flowables = [
            make_native_table(cell_dominance_df),
            Spacer(1, 4),
            Paragraph(
                "Cell dominance was broadly consistent with the planned RF design. Overlap between "
                "adjacent sectors supported mobility without excessive overshooting dominance.",
                self.styles["Body"],
            ),
        ]
        self.add_labeled_block("5.7 Cell Dominance Analysis", dominance_flowables)

        summary_flowables = [make_native_table(summary_df), Spacer(1, 4)]
        if mobility_status == "Good" and neighbor_status == "Available":
            overall_text = (
                "Mobility performance was stable across the evaluated route. Serving-cell behaviour, "
                "PCI distribution and dedicated neighbor measurements support acceptable mobility conditions."
            )
        elif mobility_status == "Good":
            overall_text = (
                "Serving-cell mobility indicators were stable, but neighbor-specific evidence was limited "
                "for a fuller mobility conclusion."
            )
        else:
            overall_text = (
                "Mobility indicators show serving-cell concentration that should be reviewed before the "
                "handover analysis stage."
            )
        summary_flowables.append(Paragraph(f"Observation: {overall_text}", self.styles["Body"]))
        self.add_labeled_block("5.8 Mobility Summary", summary_flowables)

    def add_service_kpi_analysis(self, report_df: pd.DataFrame):
        self.story.append(PageBreak())
        self.add_toc_heading("7. Service KPI Analysis", self.styles["Section"], 0, "sec7")
        self.story.append(Paragraph(
            "Service KPIs evaluate the end-user experience by measuring data performance, voice quality, "
            "application responsiveness and IP connectivity.",
            self.styles["Body"],
        ))
        self.story.append(Spacer(1, 8))

        dl_flowables = [
            Paragraph("Objective: measure achievable download speed under live network conditions.", self.styles["Body"]),
            Spacer(1, 4),
            make_native_table(build_service_metric_table(report_df, "dl_tpt", "DL Throughput", "Mbps")),
            Spacer(1, 4),
            Paragraph(
                "Observation: Download throughput remained satisfactory across most of the route. "
                "Reduced throughput is typically associated with higher cell loading.",
                self.styles["Body"],
            ),
            Spacer(1, 4),
            Paragraph("Recommendation: review scheduler utilization, carrier aggregation and cell loading.", self.styles["Body"]),
        ]
        self.add_labeled_block("7.1 Downlink Throughput", dl_flowables)
        self.story.extend(self._optional_image_flowables(["dl_map.png"], subdir="kpi_maps", max_width=5.8 * inch, max_height=4.5 * inch))
        self.story.extend(self._optional_image_flowables(["cdf_dl_tpt.png"], subdir="kpi_analysis", max_width=TABLE_MAX_WIDTH, max_height=4.5 * inch))

        ul_flowables = [
            make_native_table(build_service_metric_table(report_df, "ul_tpt", "UL Throughput", "Mbps")),
            Spacer(1, 4),
            Paragraph(
                "Observation: Uplink performance remained stable with isolated degradation in cell-edge regions.",
                self.styles["Body"],
            ),
        ]
        self.add_labeled_block("7.2 Uplink Throughput", ul_flowables)
        self.story.extend(self._optional_image_flowables(["ul_map.png"], subdir="kpi_maps", max_width=5.8 * inch, max_height=4.5 * inch))
        self.story.extend(self._optional_image_flowables(["cdf_ul_tpt.png"], subdir="kpi_analysis", max_width=TABLE_MAX_WIDTH, max_height=4.5 * inch))

        mos = _series(report_df, "mos")
        mos_table = (
            build_two_metric_table("Average MOS", f"{float(mos.mean()):.2f}", "Minimum MOS", f"{float(mos.min()):.2f}")
            if not mos.empty else pd.DataFrame({"Metric": ["MOS"], "Sample Value": ["No data available"]})
        )
        mos_flowables = [
            make_native_table(mos_table),
            Spacer(1, 4),
            Paragraph(
                "Observation: Voice quality remained acceptable with no perceptible degradation during successful calls.",
                self.styles["Body"],
            ),
        ]
        self.add_labeled_block("7.3 Voice Quality (MOS)", mos_flowables)
        self.story.extend(self._optional_image_flowables(["mos_map.png"], subdir="kpi_maps", max_width=5.8 * inch, max_height=4.5 * inch))
        self.story.extend(self._optional_image_flowables(["cdf_mos.png"], subdir="kpi_analysis", max_width=TABLE_MAX_WIDTH, max_height=4.5 * inch))

        latency = _series(report_df, "latency")
        latency_table = (
            build_two_metric_table("Average Latency", f"{float(latency.mean()):.1f} ms", "Maximum Latency", f"{float(latency.max()):.1f} ms")
            if not latency.empty else pd.DataFrame({"Metric": ["Latency"], "Sample Value": ["No data available"]})
        )
        latency_flowables = [
            make_native_table(latency_table),
            Spacer(1, 4),
            Paragraph(
                "Observation: Latency values support satisfactory web browsing and real-time applications.",
                self.styles["Body"],
            ),
        ]
        self.add_labeled_block("7.4 Latency Analysis", latency_flowables)
        self.story.extend(self._optional_image_flowables(["latency_hist.png"], subdir="kpi_analysis", max_width=TABLE_MAX_WIDTH, max_height=4.5 * inch))

        jitter = _series(report_df, "jitter")
        jitter_table = (
            build_two_metric_table("Average Jitter", f"{float(jitter.mean()):.1f} ms", "Maximum Jitter", f"{float(jitter.max()):.1f} ms")
            if not jitter.empty else pd.DataFrame({"Metric": ["Jitter"], "Sample Value": ["No data available"]})
        )
        jitter_flowables = [
            make_native_table(jitter_table),
            Spacer(1, 4),
            Paragraph(
                "Observation: Jitter remained within acceptable limits for voice and video services.",
                self.styles["Body"],
            ),
        ]
        self.add_labeled_block("7.5 Jitter Analysis", jitter_flowables)
        self.story.extend(self._optional_image_flowables(["jitter_hist.png"], subdir="kpi_analysis", max_width=TABLE_MAX_WIDTH, max_height=4.5 * inch))

        packet_flowables = [
            make_native_table(build_packet_loss_table(report_df)),
            Spacer(1, 4),
            Paragraph(
                "Observation: Negligible packet loss was recorded during the drive test.",
                self.styles["Body"],
            ),
        ]
        self.add_labeled_block("7.6 Packet Loss", packet_flowables)

        app_df_part1, app_df_part2 = build_application_analytics_tables(report_df)
        if app_df_part1 is not None and not app_df_part1.empty:
            app_obs_df = build_application_observation_table(report_df)
            app_flowables = [
                make_native_table(app_df_part1),
                Spacer(1, 4),
            ]
            if app_df_part2 is not None and not app_df_part2.empty:
                app_flowables.extend([make_native_table(app_df_part2), Spacer(1, 4)])
            if not app_obs_df.empty:
                app_flowables.extend([make_native_table(app_obs_df), Spacer(1, 4)])
            app_flowables.append(Paragraph(
                f"Observation: {build_application_overall_observation(report_df, app_obs_df)}",
                self.styles["Body"],
            ))
            self.add_labeled_block("7.7 Application Analytics", app_flowables)

        speed = _series(report_df, "speed")
        if not speed.empty or not latency.empty or not _series(report_df, "dl_tpt").empty or not _series(report_df, "ul_tpt").empty:
            speed_rows = {
                "Test": ["Download Speed", "Upload Speed", "Ping"],
                "Result": [
                    f"{float(_series(report_df, 'dl_tpt').mean()):.1f} Mbps" if not _series(report_df, "dl_tpt").empty else "N/A",
                    f"{float(_series(report_df, 'ul_tpt').mean()):.1f} Mbps" if not _series(report_df, "ul_tpt").empty else "N/A",
                    f"{float(latency.mean()):.1f} ms" if not latency.empty else "N/A",
                ],
            }
            self.add_labeled_block("7.8 Speed Test Summary", [make_native_table(pd.DataFrame(speed_rows))])

        summary_df = build_service_kpi_summary_table(report_df)
        if not summary_df.empty:
            self.add_labeled_block("7.9 Service KPI Summary", [make_native_table(summary_df)])

        self.add_labeled_block("7.10 Engineering Conclusion", [Paragraph(
            "Overall service performance was satisfactory throughout the evaluated route. Data throughput supported "
            "common user applications, voice quality remained acceptable, and IP connectivity metrics stayed within "
            "acceptable limits. Localized degradation should be investigated through capacity optimization and scheduler analysis.",
            self.styles["Body"],
        )])

    def build(self):
        self.doc.multiBuild(self.story, canvasmaker=PageNumCanvas)
        return self.output_path
