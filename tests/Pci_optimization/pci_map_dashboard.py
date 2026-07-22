"""
PCI Optimization — test-only site/sector map (Streamlit).

Visualization only, for a small number of real sites (default 3). Sector
wedge geometry and PCI color palette are ported verbatim from the frontend
(StraceExeFron/src/components/unifiedMap/NetworkPlannerMap.jsx computeOffset
+ triangle build, StraceExeFron/src/components/map/layers/
MultiColorCirclesLayer.jsx PCI_COLOR_PALETTE) so this map reads the same way
the live app does. PCI handover/transition data comes from the corrected
`_detect_pci_transitions` in pci_optimization_dataset_test.py (this folder),
which itself delegates to the verified frontend-consistent
`_detect_column_transitions` in tests/new_pdf_report/new_report_sections.py.

No collision/confusion/mod-N conflict rules here — that is a later phase.
This step only shows PCI values, sector geometry, and observed neighbors.

Data flow: DB is only hit when you click "Refresh from DB", or on first-ever
load for a project (no local snapshot yet). Every other Load reads the
saved JSON snapshot in artifacts/project_<id>.json — plain, hand-editable
JSON, so synthetic sites/PCIs can be spliced in later for collision/
confusion testing without touching the DB at all.

Run: <venv>/python -m streamlit run tests/Pci_optimization/pci_map_dashboard.py
(from the ML/ directory, so the tests package resolves).
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from tests.Pci_optimization.pci_optimization_dataset_test import (
    DEFAULT_OPERATOR,
    DEFAULT_PROJECT_ID,
    DEFAULT_REGION,
    _build_pair_summary,
    _detect_pci_transitions,
    _enrich_transition_sites,
    _fetch_network_logs,
    _fetch_project_session_ids,
    _fetch_site_metadata,
    _polygon_filter_logs,
    _polygon_filter_sites,
    _resolve_engine,
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# Verbatim from MultiColorCirclesLayer.jsx (PCI_COLOR_PALETTE)
PCI_COLOR_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
    "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B739", "#52BE80",
    "#EC7063", "#5DADE2", "#F39C12", "#A569BD", "#48C9B0",
    "#E74C3C", "#3498DB", "#E67E22", "#9B59B6", "#1ABC9C",
]

# Fallbacks match useSiteData.js's normalizeBeamwidth/normalizeSectorRange defaults.
DEFAULT_BEAMWIDTH_DEG = 30.0
DEFAULT_RANGE_M = 220.0
SITE_MARKER_COLOR = "#1d4ed8"

# Real median_sample_distance_m can be a few meters (e.g. 1.7m) — geographically
# accurate but visually a single pixel at any zoom that also fits multiple
# sites. This is a rendering-only floor (does NOT change the real distance
# shown in the tooltip) so every wedge stays legible without zooming in.
MIN_DISPLAY_RADIUS_M = 90.0

# DataFrames stored in the artifact, by key -> in-memory dict key.
_ARTIFACT_TABLES = ("site_df", "events_df", "pair_summary_df")


def pci_color(pci: float | int | None) -> str:
    if pci is None or (isinstance(pci, float) and math.isnan(pci)):
        return "#9ca3af"
    return PCI_COLOR_PALETTE[int(pci) % len(PCI_COLOR_PALETTE)]


def compute_offset(lat: float, lon: float, distance_m: float, heading_deg: float) -> tuple[float, float]:
    """Geodesic destination point. Same formula as NetworkPlannerMap.jsx's computeOffset()."""
    earth_radius = 6371000.0
    lat1 = math.radians(lat)
    lng1 = math.radians(lon)
    heading = math.radians(heading_deg)
    ang_dist = distance_m / earth_radius

    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang_dist) + math.cos(lat1) * math.sin(ang_dist) * math.cos(heading)
    )
    lng2 = lng1 + math.atan2(
        math.sin(heading) * math.sin(ang_dist) * math.cos(lat1),
        math.cos(ang_dist) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lng2)


def normalize_beamwidth(value, fallback: float = DEFAULT_BEAMWIDTH_DEG) -> float:
    """Same clamp as useSiteData.js's normalizeBeamwidth: keep the real value
    (even a thin one) if present, just bound it to [5, 180]; only fall back
    to the default when the value is missing/non-positive."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(numeric) or numeric <= 0:
        return fallback
    return max(5.0, min(180.0, numeric))


def normalize_sector_range(value, fallback: float = DEFAULT_RANGE_M) -> float:
    """Same clamp as useSiteData.js's normalizeSectorRange: [20, 5000] m."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(numeric) or numeric <= 0:
        return fallback
    return max(20.0, min(5000.0, numeric))


def build_sector_triangle(
    site_lat: float, site_lon: float, azimuth_deg: float, beamwidth_deg: float, radius_m: float
) -> list[tuple[float, float]]:
    """[site, left edge, right edge] — same pie-slice shape as NetworkPlannerMap.jsx."""
    p1 = compute_offset(site_lat, site_lon, radius_m, azimuth_deg - beamwidth_deg / 2)
    p2 = compute_offset(site_lat, site_lon, radius_m, azimuth_deg + beamwidth_deg / 2)
    return [(site_lat, site_lon), p1, p2]


def get_artifact_path(project_id: int) -> Path:
    return ARTIFACT_DIR / f"project_{int(project_id)}.json"


def fetch_pci_map_data_from_db(project_id: int, region: str, operator: str, polygon_filter: bool) -> dict:
    """
    The only place this module touches the DB / the Electron PythonBridge.
    `polygon_filter` mirrors the old CLI script's own --no-polygon-filter
    escape hatch: _apply_drive_polygon_filter calls out to the local
    Electron/Node PythonBridge service (localhost:5224) for project region
    polygons, which is only reachable when this runs inside/alongside the
    desktop app. Turn it off to run standalone.
    """
    current_engine = _resolve_engine(region)
    with current_engine.connect() as conn:
        session_ids, project_meta = _fetch_project_session_ids(project_id, conn)
        raw_log_df = _fetch_network_logs(session_ids, operator, conn, primary_only=True)
        log_df, log_polygon_stats = _polygon_filter_logs(raw_log_df, project_id, current_engine, enabled=polygon_filter)
        site_df = _fetch_site_metadata(project_id, conn)
        site_df, site_polygon_stats = _polygon_filter_sites(site_df, project_id, current_engine, enabled=polygon_filter)

    if log_df.empty:
        raise ValueError(f"No network log rows for project_id={project_id} (session_ids={session_ids}).")
    if site_df.empty:
        raise ValueError(f"No site metadata rows for project_id={project_id}.")

    events_df = _detect_pci_transitions(log_df)
    enriched_events_df = _enrich_transition_sites(events_df, site_df)
    pair_summary_df = _build_pair_summary(enriched_events_df)

    return {
        "project_id": int(project_id),
        "region": region,
        "operator": operator,
        "polygon_filter": bool(polygon_filter),
        "session_ids": session_ids,
        "project_meta": project_meta,
        "log_polygon_stats": log_polygon_stats,
        "site_polygon_stats": site_polygon_stats,
        "site_df": site_df,
        "events_df": enriched_events_df,
        "pair_summary_df": pair_summary_df,
    }


def save_artifact(data: dict, path: Path) -> None:
    payload = {k: v for k, v in data.items() if k not in _ARTIFACT_TABLES}
    for table_key in _ARTIFACT_TABLES:
        df = data[table_key]
        payload[table_key] = json.loads(df.to_json(orient="records", date_format="iso"))
    payload["saved_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_artifact(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = {k: v for k, v in payload.items() if k not in _ARTIFACT_TABLES}
    for table_key in _ARTIFACT_TABLES:
        data[table_key] = pd.DataFrame(payload.get(table_key, []))
    return data


def _plausible_sites(site_df: pd.DataFrame) -> pd.DataFrame:
    """Drop the site_key_inferred="0" sentinel (bucket of failed per-row
    site inference, confirmed on project 193 to span ~1.5km of unrelated
    physical locations under one fake "site") and any remaining group whose
    lat/lon spread is too large to be one physical site."""
    if "site_id_inferred" not in site_df.columns:
        raise ValueError("site metadata has no site_id_inferred column; cannot group sectors into sites.")

    sites_with_id = site_df.dropna(subset=["site_id_inferred"]).copy()
    sites_with_id = sites_with_id[sites_with_id["site_id_inferred"].astype(str) != "0"]
    if sites_with_id.empty:
        raise ValueError("No site rows have a resolved (non-sentinel) site_id_inferred.")

    spread = sites_with_id.groupby("site_id_inferred").agg(
        lat_spread=("site_lat", lambda s: pd.to_numeric(s, errors="coerce").max() - pd.to_numeric(s, errors="coerce").min()),
        lon_spread=("site_lon", lambda s: pd.to_numeric(s, errors="coerce").max() - pd.to_numeric(s, errors="coerce").min()),
    )
    plausible_site_ids = spread[(spread["lat_spread"] <= 0.003) & (spread["lon_spread"] <= 0.003)].index
    sites_with_id = sites_with_id[sites_with_id["site_id_inferred"].isin(plausible_site_ids)]
    if sites_with_id.empty:
        raise ValueError("No site group has a physically plausible lat/lon spread.")
    return sites_with_id


def select_top_sites(site_df: pd.DataFrame, num_sites: int) -> tuple[pd.DataFrame, list]:
    """Fallback ranking: each site's smallest real PCI value, ascending —
    small, easy-to-reason-about numbers (e.g. 1, 3, 6) instead of whatever
    high-sample-count site happens to have large PCIs like 185/210."""
    sites_with_id = _plausible_sites(site_df)
    site_rank = (
        sites_with_id.assign(site_pci=pd.to_numeric(sites_with_id["site_pci"], errors="coerce"))
        .dropna(subset=["site_pci"])
        .groupby("site_id_inferred")["site_pci"]
        .min()
        .sort_values(ascending=True)
    )
    selected_site_ids = list(site_rank.head(num_sites).index)
    selected_sites_df = sites_with_id[sites_with_id["site_id_inferred"].isin(selected_site_ids)].copy()
    return selected_sites_df, selected_site_ids


def detect_pci_confusion(events_df: pd.DataFrame, site_ids_filter: set | None = None) -> list[dict]:
    """
    Confusion: a serving site has 2+ distinct neighbor sites that all
    report the same PCI on the same EARFCN (a UE can't tell those
    neighbors apart from the handover target's PCI alone), using real
    observed handovers (events_df, from the corrected
    _detect_pci_transitions -> _enrich_transition_sites pipeline).
    Confusion always implies an underlying Collision between the confused
    neighbor sites themselves (that IS what confusion is: a serving site's
    view of a collision between two of its own neighbors) — there is no
    such thing as confusion without collision. The reverse isn't true:
    two sites can collide (detect_pci_collisions) without any serving site
    observing them as shared neighbors, i.e. without confusion.

    `site_ids_filter`, if given, restricts to events where BOTH sides are
    in that set — pass the currently-selected site_id_inferred values to
    detect confusion only within what's actually being shown.
    """
    required_cols = {"from_site_id_inferred", "to_site_id_inferred", "to_pci", "to_site_earfcn"}
    if events_df.empty or not required_cols.issubset(events_df.columns):
        return []
    events = events_df.copy()
    events["to_pci"] = pd.to_numeric(events["to_pci"], errors="coerce")
    events = events.dropna(subset=["from_site_id_inferred", "to_site_id_inferred", "to_pci", "to_site_earfcn"])
    events = events[events["from_site_id_inferred"].astype(str) != "0"]
    events = events[events["to_site_id_inferred"].astype(str) != "0"]
    if site_ids_filter is not None:
        events = events[
            events["from_site_id_inferred"].isin(site_ids_filter) & events["to_site_id_inferred"].isin(site_ids_filter)
        ]
    # Group by (serving site, neighbor PCI, neighbor EARFCN) — not PCI
    # alone. PCI space is reused independently per carrier frequency, so
    # two cells sharing a PCI on different EARFCNs don't actually
    # collide/confuse; restricting to a shared EARFCN keeps this a real,
    # frequency-layer-correct case.
    pairs = events.drop_duplicates(["from_site_id_inferred", "to_site_id_inferred", "to_pci", "to_site_earfcn"])
    conflicts: list[dict] = []
    for (from_site, pci_val, earfcn_val), grp in pairs.groupby(["from_site_id_inferred", "to_pci", "to_site_earfcn"]):
        neighbor_sites = sorted(grp["to_site_id_inferred"].unique().tolist())
        if len(neighbor_sites) >= 2:
            conflicts.append(
                {
                    "type": "Confusion",
                    "pci": int(pci_val),
                    "earfcn": earfcn_val,
                    "serving_site": from_site,
                    "neighbor_sites": neighbor_sites,
                }
            )
    conflicts.sort(key=lambda c: len(c["neighbor_sites"]), reverse=True)
    return conflicts


def select_collision_confusion_cluster(
    site_df: pd.DataFrame, events_df: pd.DataFrame, num_sites: int
) -> tuple[pd.DataFrame, list, str, list]:
    """Picks sites around the strongest real Confusion cluster found by
    detect_pci_confusion(); falls back to select_top_sites() (smallest-PCI
    ranking) if this project's handover data has no such cluster."""
    groups = detect_pci_confusion(events_df)

    selected_ids: list = []
    explanation_parts: list[str] = []
    conflicts: list[dict] = []
    for g in groups:
        if len(selected_ids) >= num_sites:
            break
        cluster_candidates = [g["serving_site"]] + [s for s in g["neighbor_sites"] if s != g["serving_site"]]
        room = num_sites - len(selected_ids)
        new_sites = [s for s in cluster_candidates if s not in selected_ids][:room]
        if new_sites:
            selected_ids.extend(new_sites)
            explanation_parts.append(
                f"serving site {g['serving_site']} has {len(g['neighbor_sites'])} neighbor site(s) all reporting "
                f"PCI {g['pci']} on EARFCN {g['earfcn']} (confusion at the serving site + collision among those neighbors)"
            )
            conflicts.append(g)

    if not selected_ids:
        sites_df, ids = select_top_sites(site_df, num_sites)
        return sites_df, ids, "No PCI confusion/collision cluster found in this project's observed handovers — showing smallest-PCI sites instead.", []

    selected_sites_df = site_df[site_df["site_id_inferred"].isin(selected_ids)].copy()
    return selected_sites_df, selected_ids, " | ".join(explanation_parts), conflicts


def detect_pci_collisions(selected_sites_df: pd.DataFrame) -> list[dict]:
    """
    Independent Collision detector: 2+ of the currently-selected sites
    sharing the same (PCI, EARFCN) — no serving-site/neighbor concept at
    all, unlike Confusion (select_collision_confusion_cluster). Run
    separately so "Collision" and "Confusion" are genuinely independent
    toggles in the UI, not the same check shown under two names.
    """
    df = selected_sites_df.dropna(subset=["site_pci", "site_earfcn", "site_id_inferred"]).copy()
    if df.empty:
        return []
    df["site_pci"] = pd.to_numeric(df["site_pci"], errors="coerce")
    df["site_earfcn"] = pd.to_numeric(df["site_earfcn"], errors="coerce")
    df = df.dropna(subset=["site_pci", "site_earfcn"])

    collisions: list[dict] = []
    for (pci_val, earfcn_val), grp in df.groupby(["site_pci", "site_earfcn"]):
        sites = sorted(grp["site_id_inferred"].unique().tolist())
        if len(sites) >= 2:
            collisions.append({"type": "Collision", "pci": int(pci_val), "earfcn": earfcn_val, "sites": sites})
    return collisions


MOD_RULE_VALUES = (1, 3, 7, 9)


def detect_pci_mod_conflicts(selected_sites_df: pd.DataFrame, mod_n: int) -> list[dict]:
    """
    Generic PCI-mod-N conflict: 2+ of the currently-selected sites' real
    PCIs share the same (PCI % mod_n) remainder on the same EARFCN, even
    when their actual PCI values differ (unlike Collision, which needs an
    exact PCI match). Mod3 is the one with real 3GPP meaning here (PSS
    group — neighbor cells sharing a PSS degrade cross-correlation/sync).
    Mod1 is trivial (every PCI % 1 == 0, so it will flag almost every
    same-EARFCN site pair) and Mod7/Mod9 have no standard 3GPP planning
    meaning — all three are included because they were explicitly
    requested, not because they're meaningful network-planning checks.
    """
    df = selected_sites_df.dropna(subset=["site_pci", "site_earfcn", "site_id_inferred"]).copy()
    if df.empty:
        return []
    df["site_pci"] = pd.to_numeric(df["site_pci"], errors="coerce")
    df["site_earfcn"] = pd.to_numeric(df["site_earfcn"], errors="coerce")
    df = df.dropna(subset=["site_pci", "site_earfcn"])
    df["_mod"] = df["site_pci"].astype(int) % mod_n

    conflicts: list[dict] = []
    for (mod_val, earfcn_val), grp in df.groupby(["_mod", "site_earfcn"]):
        if grp["site_id_inferred"].nunique() < 2:
            continue  # one site with several same-mod sectors of its own isn't a neighbor-facing conflict
        members = sorted({(row.site_id_inferred, int(row.site_pci)) for row in grp.itertuples()})
        conflicts.append(
            {
                "type": f"Mod{mod_n}",
                "mod_n": mod_n,
                "mod_value": int(mod_val),
                "earfcn": earfcn_val,
                "members": members,
            }
        )
    conflicts.sort(key=lambda c: len(c["members"]), reverse=True)
    return conflicts


def inject_synthetic_mod_conflict(
    site_df: pd.DataFrame, selected_site_ids: list, mod_n: int, used_pcis: set
) -> tuple[pd.DataFrame, str]:
    """
    Fallback only — in practice Mod1/3/7/9 conflicts occur naturally among
    5+ real sites (coarse groupings collide far more often than an exact
    PCI match does), so this rarely fires. If detect_pci_mod_conflicts()
    still finds nothing for mod_n among the currently selected sites, adds
    ONE synthetic sector to an already-selected real site, with a PCI that
    shares mod_n's remainder with another already-selected site's real
    PCI on the same EARFCN — reuses the existing site set instead of
    pulling in new sites just for this.
    """
    candidates = site_df[site_df["site_id_inferred"].isin(selected_site_ids)].dropna(subset=["site_pci", "site_earfcn"]).copy()
    candidates["site_pci"] = pd.to_numeric(candidates["site_pci"], errors="coerce")
    candidates["site_earfcn"] = pd.to_numeric(candidates["site_earfcn"], errors="coerce")
    candidates = candidates.dropna(subset=["site_pci", "site_earfcn"])
    if len(candidates) < 2:
        raise ValueError(f"Not enough sector data among selected sites to synthesize a Mod{mod_n} example.")

    anchor = candidates.iloc[0]
    target_site = candidates.iloc[1]["site_id_inferred"]
    if target_site == anchor["site_id_inferred"] and len(candidates["site_id_inferred"].unique()) > 1:
        target_site = candidates[candidates["site_id_inferred"] != anchor["site_id_inferred"]].iloc[0]["site_id_inferred"]

    earfcn = anchor["site_earfcn"]
    target_mod = int(anchor["site_pci"]) % mod_n
    new_pci = target_mod
    while new_pci in used_pcis or new_pci == int(anchor["site_pci"]):
        new_pci += mod_n
        if new_pci > 503:
            raise ValueError(f"No free PCI value available to synthesize a Mod{mod_n} example.")

    site_df = site_df.copy()
    if "is_synthetic" not in site_df.columns:
        site_df["is_synthetic"] = False
    new_row = {
        "project_id": anchor.get("project_id"),
        "network": anchor.get("network"),
        "site_earfcn": earfcn,
        "site_pci": new_pci,
        "samples": 30,
        "site_lat": anchor.get("site_lat"),
        "site_lon": anchor.get("site_lon"),
        "site_azimuth_deg": 0.0,
        "site_azimuth_soft_deg": None,
        "beamwidth_deg_est": 30.0,
        "median_sample_distance_m": 150.0,
        "site_id_inferred": target_site,
        "site_source_table": SYNTHETIC_SOURCE_TABLE,
        "is_synthetic": True,
    }
    site_df = pd.concat([site_df, pd.DataFrame([new_row])], ignore_index=True)
    description = (
        f"Synthetic (Mod{mod_n}): real site {target_site} given a synthetic PCI {new_pci} on EARFCN {int(earfcn)} "
        f"(PCI % {mod_n} = {target_mod}), matching real site {anchor['site_id_inferred']}'s real PCI "
        f"{int(anchor['site_pci'])} on the same modulus — no natural Mod{mod_n} conflict existed among the "
        "selected sites."
    )
    return site_df, description


SYNTHETIC_SOURCE_TABLE = "synthetic"


def _pick_synthetic_pci(existing_pcis: set) -> int:
    for candidate in range(100, 504):
        if candidate not in existing_pcis:
            return candidate
    raise ValueError("No free PCI value (100-503) available for synthetic injection.")


def inject_synthetic_collision_confusion(site_df: pd.DataFrame, events_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Verified: neither project 193 nor 196 has a real same-EARFCN PCI
    collision or confusion case in the observed handover data (0 (PCI,
    EARFCN) pairs reused across sites, for any operator, in either
    project) — real PCI planning is per-frequency-layer, and reuse only
    matters within a shared EARFCN.

    This finds a REAL serving site and its two most-handed-over-to REAL
    neighbor sites (from real observed events), then adds ONE synthetic
    sector to each of those two real neighbor sites: same synthetic PCI,
    same EARFCN — plus two synthetic handover events tying them back to
    that real serving site. Everything except the injected PCI/EARFCN/
    handover numbers is real (real site locations, real serving site, real
    neighbor relationship) — only the "these two also share a PCI on this
    EARFCN" fact is fabricated, because that fact does not exist in either
    project's real data.

    Because the injected rows have the exact same shape as real ones,
    select_collision_confusion_cluster()'s existing, unmodified same-EARFCN
    detection logic finds and picks this cluster with no special-casing.
    Injected rows are marked site_source_table="synthetic" / is_synthetic=True
    (site_df) and is_synthetic=True (events_df) so the UI never mixes them
    up with real data — render_map() draws them with a dashed border and
    "(synthetic)" in the tooltip, and build_site_pci_table() flags them.
    """
    events = events_df.copy()
    events = events[events["from_site_id_inferred"].astype(str) != "0"]
    events = events[events["to_site_id_inferred"].astype(str) != "0"]
    events = events.dropna(subset=["from_site_id_inferred", "to_site_id_inferred"])
    if events.empty:
        raise ValueError("No real handover events available to anchor a synthetic injection to.")

    degree = events.groupby("from_site_id_inferred")["to_site_id_inferred"].nunique().sort_values(ascending=False)
    serving_site = degree.index[0]
    neighbor_counts = events.loc[events["from_site_id_inferred"] == serving_site, "to_site_id_inferred"].value_counts()
    if len(neighbor_counts) < 2:
        raise ValueError("Chosen serving site does not have 2 distinct real neighbors to attach a synthetic collision to.")
    site_b, site_c = neighbor_counts.index[0], neighbor_counts.index[1]

    site_df = site_df.copy()
    if "is_synthetic" not in site_df.columns:
        site_df["is_synthetic"] = False

    # Prefer an EARFCN both neighbor sites already broadcast on for
    # plausibility; fall back to site_b's first EARFCN.
    b_earfcns = set(pd.to_numeric(site_df.loc[site_df["site_id_inferred"] == site_b, "site_earfcn"], errors="coerce").dropna())
    c_earfcns = set(pd.to_numeric(site_df.loc[site_df["site_id_inferred"] == site_c, "site_earfcn"], errors="coerce").dropna())
    shared_earfcns = b_earfcns & c_earfcns
    synthetic_earfcn = sorted(shared_earfcns)[0] if shared_earfcns else sorted(b_earfcns)[0]

    existing_pcis_at_earfcn = set(
        pd.to_numeric(
            site_df.loc[
                (site_df["site_id_inferred"].isin([site_b, site_c])) & (site_df["site_earfcn"] == synthetic_earfcn),
                "site_pci",
            ],
            errors="coerce",
        ).dropna().astype(int)
    )
    synthetic_pci = _pick_synthetic_pci(existing_pcis_at_earfcn)

    new_site_rows = []
    for site_id in (site_b, site_c):
        anchor = site_df.loc[site_df["site_id_inferred"] == site_id].iloc[0]
        new_site_rows.append(
            {
                "project_id": anchor.get("project_id"),
                "network": anchor.get("network"),
                "site_earfcn": synthetic_earfcn,
                "site_pci": synthetic_pci,
                "samples": 50,
                "site_lat": anchor.get("site_lat"),
                "site_lon": anchor.get("site_lon"),
                "site_azimuth_deg": 0.0,
                "site_azimuth_soft_deg": None,
                "beamwidth_deg_est": 30.0,
                "median_sample_distance_m": 150.0,
                "site_id_inferred": site_id,
                "site_source_table": SYNTHETIC_SOURCE_TABLE,
                "is_synthetic": True,
            }
        )
    site_df = pd.concat([site_df, pd.DataFrame(new_site_rows)], ignore_index=True)

    serving_row = site_df.loc[site_df["site_id_inferred"] == serving_site].iloc[0]
    from_pci = serving_row.get("site_pci")
    from_earfcn = serving_row.get("site_earfcn")

    events_out = events_df.copy()
    if "is_synthetic" not in events_out.columns:
        events_out["is_synthetic"] = False

    # events_df's from_pci/to_pci/from_earfcn/to_earfcn are strings in real
    # rows (they pass through _clean_transition_text in _detect_column_
    # transitions), not numbers — mixing str and int in one "object" column
    # is exactly what breaks Streamlit's Arrow serialization for st.dataframe
    # (only to_site_earfcn is genuinely numeric in real rows, from the
    # site_df merge, so that one stays numeric here too).
    new_event_rows = []
    for idx, neighbor_site in enumerate((site_b, site_c), start=1):
        new_event_rows.append(
            {
                "session_id": f"SYNTH-{idx}",
                "from_pci": str(int(from_pci)),
                "to_pci": str(synthetic_pci),
                "from_earfcn": str(int(from_earfcn)),
                "to_earfcn": str(int(synthetic_earfcn)),
                "from_site_id_inferred": serving_site,
                "to_site_id_inferred": neighbor_site,
                "to_site_earfcn": synthetic_earfcn,
                "segment_distance_m": 100.0,
                "time_delta_sec": 5.0,
                "rsrp_before": -95.0,
                "rsrp_after": -90.0,
                "rsrq_before": -12.0,
                "rsrq_after": -11.0,
                "sinr_before": 8.0,
                "sinr_after": 10.0,
                "rsrp_delta": 5.0,
                "rsrq_delta": 1.0,
                "sinr_delta": 2.0,
                "is_synthetic": True,
            }
        )
    events_out = pd.concat([events_out, pd.DataFrame(new_event_rows)], ignore_index=True)
    events_out["is_synthetic"] = events_out["is_synthetic"].fillna(False)

    description = (
        f"Synthetic (Collision + Confusion): real serving site {serving_site} given 2 synthetic handover "
        f"events to its real neighbor sites {site_b} and {site_c}, both assigned synthetic PCI {synthetic_pci} "
        f"on EARFCN {synthetic_earfcn}."
    )
    return site_df, events_out, description


def inject_synthetic_pure_collision(
    site_df: pd.DataFrame, events_df: pd.DataFrame, exclude_site_ids: set, synthetic_pci: int
) -> tuple[pd.DataFrame, str, str, str]:
    """
    Confusion always implies collision (it's a serving site's view of a
    collision between two of its own neighbors), but the reverse isn't
    true — this builds a genuinely separate Collision-only example: two
    REAL sites that are NOT both real neighbors of any common serving
    site (checked against every real confusion group), given a shared
    synthetic PCI/EARFCN with `exclude_site_ids` (the sites already used
    by inject_synthetic_collision_confusion) kept out of the pool so the
    two examples stay clearly distinct sites. No handover events are
    added here, deliberately — nothing observes these two as shared
    neighbors, so detect_pci_confusion() will never flag them, only
    detect_pci_collisions() will.
    """
    confused_pairs: set = set()
    if not events_df.empty and {"from_site_id_inferred", "to_site_id_inferred"}.issubset(events_df.columns):
        for _, grp in events_df.dropna(subset=["from_site_id_inferred", "to_site_id_inferred"]).groupby("from_site_id_inferred"):
            neighbors = sorted(grp["to_site_id_inferred"].unique().tolist())
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    confused_pairs.add(frozenset((neighbors[i], neighbors[j])))

    candidate_ids = [
        sid for sid in site_df.dropna(subset=["site_id_inferred"])["site_id_inferred"].unique()
        if str(sid) != "0" and sid not in exclude_site_ids
    ]
    site_x = site_y = None
    for i in range(len(candidate_ids)):
        for j in range(i + 1, len(candidate_ids)):
            a, b = candidate_ids[i], candidate_ids[j]
            if frozenset((a, b)) not in confused_pairs:
                site_x, site_y = a, b
                break
        if site_x is not None:
            break
    if site_x is None:
        raise ValueError("No pair of sites available for a Collision-only synthetic example.")

    site_df = site_df.copy()
    if "is_synthetic" not in site_df.columns:
        site_df["is_synthetic"] = False

    anchor_x = site_df.loc[site_df["site_id_inferred"] == site_x].iloc[0]
    synthetic_earfcn = pd.to_numeric(anchor_x.get("site_earfcn"), errors="coerce")
    if pd.isna(synthetic_earfcn):
        synthetic_earfcn = 315.0

    new_rows = []
    for site_id in (site_x, site_y):
        anchor = site_df.loc[site_df["site_id_inferred"] == site_id].iloc[0]
        new_rows.append(
            {
                "project_id": anchor.get("project_id"),
                "network": anchor.get("network"),
                "site_earfcn": synthetic_earfcn,
                "site_pci": synthetic_pci,
                "samples": 40,
                "site_lat": anchor.get("site_lat"),
                "site_lon": anchor.get("site_lon"),
                "site_azimuth_deg": 0.0,
                "site_azimuth_soft_deg": None,
                "beamwidth_deg_est": 30.0,
                "median_sample_distance_m": 150.0,
                "site_id_inferred": site_id,
                "site_source_table": SYNTHETIC_SOURCE_TABLE,
                "is_synthetic": True,
            }
        )
    site_df = pd.concat([site_df, pd.DataFrame(new_rows)], ignore_index=True)

    description = (
        f"Synthetic (Collision only): real sites {site_x} and {site_y} — confirmed NOT linked as neighbors of "
        f"any common serving site in the real handover data — both assigned synthetic PCI {synthetic_pci} on "
        f"EARFCN {int(synthetic_earfcn)}. No handover events added, so Confusion will never flag this pair."
    )
    return site_df, site_x, site_y, description


def build_demo_selection(site_df: pd.DataFrame, events_df: pd.DataFrame, num_sites: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list]:
    """
    Builds TWO separate synthetic examples so Collision and Confusion can
    each be seen in isolation (not the same PCI satisfying both, as
    before): a Confusion+Collision cluster (3 sites) and a Collision-only
    pair (2 more, unrelated sites) — 5 sites minimum. If num_sites > 5,
    pads with extra real sites (select_top_sites, smallest-PCI ranking,
    no conflict of either kind) so the map isn't capped at 5.
    Returns (augmented_site_df, augmented_events_df, selected_sites_df, selected_site_ids, notes).
    """
    site_df2, events_df2, note1 = inject_synthetic_collision_confusion(site_df, events_df)
    confusion_groups = detect_pci_confusion(events_df2)
    if not confusion_groups:
        raise ValueError("No confusion cluster available even after synthetic injection.")
    top = confusion_groups[0]
    confusion_site_ids = [top["serving_site"]] + [s for s in top["neighbor_sites"] if s != top["serving_site"]]

    used_pcis = {top["pci"]} | set(
        pd.to_numeric(site_df2["site_pci"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    collision_pci = _pick_synthetic_pci(used_pcis)

    site_df3, site_x, site_y, note2 = inject_synthetic_pure_collision(
        site_df2, events_df2, exclude_site_ids=set(confusion_site_ids), synthetic_pci=collision_pci
    )
    selected_site_ids = confusion_site_ids + [site_x, site_y]

    if num_sites > len(selected_site_ids):
        remaining_pool = site_df3[~site_df3["site_id_inferred"].isin(selected_site_ids)]
        _, filler_ids = select_top_sites(remaining_pool, num_sites - len(selected_site_ids))
        selected_site_ids += filler_ids

    selected_sites_df = site_df3[site_df3["site_id_inferred"].isin(selected_site_ids)].copy()
    return site_df3, events_df2, selected_sites_df, selected_site_ids, [note1, note2]


CONFLICT_BORDER_COLOR = "#DC2626"


def assign_site_labels(site_ids: list) -> dict:
    """
    site_id_inferred is an internal join key, not a human-readable name —
    and for ~46% of real sites in this project it's partly asterisked
    (e.g. "1******6") because the Android OS itself redacts the Tracking
    Area Code in CellIdentityLte for privacy (confirmed directly in
    tbl_network_log.primary_cell_info_1: "mTac=6***5"), baked into the raw
    drive-test data before it ever reached the DB — not recoverable, and
    not something to show as if it were a real site name. Map each real id
    to a clean "Site N" label instead, in the given (already meaningful,
    e.g. serving-site-first) order.
    """
    labels: dict = {}
    for site_id in site_ids:
        if site_id not in labels:
            labels[site_id] = f"Site {len(labels) + 1}"
    return labels


def build_conflict_summary_table(conflicts: list, site_labels: dict) -> pd.DataFrame:
    """Handles three conflict shapes — Confusion (serving_site + neighbor_sites),
    Collision (flat sites list, no serving-site concept), and Mod-N
    (members: list of (site, pci) pairs sharing a PCI%N remainder, not an
    exact PCI) — kept as genuinely distinct rows/types, never merged."""
    rows = []
    for c in conflicts:
        detail = ""
        if c["type"] == "Confusion":
            serving_label = site_labels.get(c["serving_site"], c["serving_site"])
            involved = ", ".join(site_labels.get(s, s) for s in c["neighbor_sites"])
            pci = str(c["pci"])  # kept as str across all rows -- Mod-N rows have no single PCI (mixed int/"" breaks Arrow)
        elif c["type"] == "Collision":
            serving_label = ""
            involved = ", ".join(site_labels.get(s, s) for s in c["sites"])
            pci = str(c["pci"])
        else:  # Mod-N
            serving_label = ""
            involved = ", ".join(f"{site_labels.get(s, s)} (PCI {p})" for s, p in c["members"])
            pci = ""
            detail = f"PCI % {c['mod_n']} = {c['mod_value']}"
        rows.append(
            {
                "Type": c["type"],
                "PCI": pci,
                "EARFCN": c["earfcn"],
                "Serving Site": serving_label,
                "Sites Involved": involved,
                "Detail": detail,
            }
        )
    return pd.DataFrame(rows)


def render_map(
    selected_sites_df: pd.DataFrame,
    site_labels: dict | None = None,
    conflict_types: dict | None = None,
    filter_to_conflicts: bool = False,
) -> folium.Map:
    """
    `filter_to_conflicts=True` only draws a sector's wedge+PCI label if its
    (pci, earfcn) is a key in `conflict_types` — everything else is skipped
    entirely, so toggling Collision/Confusion in the sidebar actually
    changes what's shown, not just its border color. Site markers/labels
    still always draw (so you can see where a serving site is even if none
    of its own sectors match), and every wedge that IS drawn in this mode
    is, by construction, a conflict, so it's always given the red border.
    """
    site_labels = site_labels or {}
    conflict_types = conflict_types or {}
    conflict_keys = set(conflict_types.keys())
    center_lat = pd.to_numeric(selected_sites_df["site_lat"], errors="coerce").mean()
    center_lon = pd.to_numeric(selected_sites_df["site_lon"], errors="coerce").mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=17, tiles="CartoDB positron", control_scale=True)

    bounds_points: list[tuple[float, float]] = []

    for site_id, site_rows in selected_sites_df.groupby("site_id_inferred"):
        first = site_rows.iloc[0]
        site_lat = float(first["site_lat"])
        site_lon = float(first["site_lon"])
        bounds_points.append((site_lat, site_lon))
        site_label = site_labels.get(site_id, f"Site {site_id}")
        folium.CircleMarker(
            location=[site_lat, site_lon],
            radius=7,
            color=SITE_MARKER_COLOR,
            weight=3,
            fill=True,
            fill_color=SITE_MARKER_COLOR,
            fill_opacity=1.0,
            tooltip=site_label,
        ).add_to(fmap)
        folium.Marker(
            location=[site_lat, site_lon],
            icon=folium.DivIcon(
                html=(
                    '<div style="'
                    "font-size:12px; font-weight:800; color:#1d4ed8; "
                    "background:rgba(255,255,255,0.95); border:1.5px solid #1d4ed8; "
                    "border-radius:4px; padding:1px 5px; white-space:nowrap; "
                    'transform:translate(10px,-22px); box-shadow:0 1px 3px rgba(0,0,0,0.4);">'
                    f"{site_label}</div>"
                )
            ),
        ).add_to(fmap)

        # Sort sectors by azimuth so each PCI label can be nudged along its
        # own bisector without stacking directly on a neighbor's label when
        # two sectors point in nearly the same direction.
        sectors = site_rows.copy()
        sectors["_azimuth_sort"] = sectors.apply(
            lambda r: r["site_azimuth_soft_deg"] if pd.notna(r.get("site_azimuth_soft_deg")) else r.get("site_azimuth_deg", 0.0),
            axis=1,
        )
        sectors = sectors.sort_values("_azimuth_sort")

        for sector_idx, (_, sector) in enumerate(sectors.iterrows()):
            pci = sector.get("site_pci")
            if pd.isna(pci):
                continue
            azimuth = sector.get("site_azimuth_soft_deg")
            if pd.isna(azimuth):
                azimuth = sector.get("site_azimuth_deg", 0.0)
            azimuth = float(azimuth) if pd.notna(azimuth) else 0.0
            beamwidth = normalize_beamwidth(sector.get("beamwidth_deg_est"))
            real_radius = normalize_sector_range(sector.get("median_sample_distance_m"))
            radius = max(real_radius, MIN_DISPLAY_RADIUS_M)

            earfcn = sector.get("site_earfcn")
            pci_int = int(pci)
            key = (pci_int, int(earfcn)) if pd.notna(earfcn) else None
            is_conflict = key in conflict_keys
            if filter_to_conflicts and not is_conflict:
                continue

            triangle = build_sector_triangle(site_lat, site_lon, azimuth, beamwidth, radius)
            bounds_points.extend(triangle)
            color = pci_color(pci)
            samples = sector.get("samples")
            is_synthetic = bool(sector.get("is_synthetic", False))
            tooltip = (
                f"{site_label} | PCI {pci_int} | EARFCN {earfcn} | "
                f"Az {azimuth:.0f}° | Beamwidth {beamwidth:.0f}° | Samples {samples}"
            )
            if is_conflict:
                tooltip += f" | {'/'.join(sorted(conflict_types[key]))}"
            if is_synthetic:
                tooltip += " | SYNTHETIC (not real data)"
            folium.Polygon(
                locations=triangle,
                color=CONFLICT_BORDER_COLOR if is_conflict else color,
                weight=4 if is_conflict else 3,
                fill=True,
                fill_color=color,
                fill_opacity=0.55,
                dash_array="8,6" if is_synthetic else None,
                tooltip=tooltip,
            ).add_to(fmap)

            # Permanent PCI label on the sector itself (not just on hover),
            # placed along the sector's own bisector so it stays associated
            # with its wedge even when neighboring sectors point in nearly
            # the same direction and their fills visually overlap. Distance
            # is staggered by sector index within the site so labels for
            # tightly clustered azimuths don't sit exactly on top of each other.
            label_distance = radius * (0.45 + 0.15 * (sector_idx % 3))
            label_lat, label_lon = compute_offset(site_lat, site_lon, label_distance, azimuth)
            border_style = "dashed" if is_synthetic else "solid"
            border_color = CONFLICT_BORDER_COLOR if is_conflict else color
            border_width = "3px" if is_conflict else "2px"
            folium.Marker(
                location=[label_lat, label_lon],
                icon=folium.DivIcon(
                    html=(
                        '<div style="'
                        "font-size:14px; font-weight:800; color:#111827; "
                        "background:rgba(255,255,255,0.95); "
                        f"border:{border_width} {border_style} {border_color}; border-radius:4px; "
                        "padding:2px 6px; white-space:nowrap; "
                        "transform:translate(-50%,-50%); "
                        'box-shadow:0 1px 3px rgba(0,0,0,0.45);">'
                        f"{pci_int}</div>"
                    )
                ),
            ).add_to(fmap)

    if bounds_points:
        lats = [p[0] for p in bounds_points]
        lons = [p[1] for p in bounds_points]
        # fit_bounds() picks the tightest zoom that still shows every
        # selected site's wedges, instead of a fixed zoom_start that can
        # leave everything looking tiny when sites are spread out (or
        # crop sites out of view when they're close together).
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(40, 40))

    return fmap


def conflict_types_by_key(conflicts: list) -> dict:
    """(pci, earfcn) -> set of type strings ("Collision", "Confusion",
    "Mod1"/"Mod3"/"Mod7"/"Mod9") found for it among the given conflicts —
    a key can carry several independently. Mod-N conflicts have no single
    PCI (their members can have different real PCIs sharing one
    remainder), so they contribute one key per (member site's real PCI,
    EARFCN) instead of a single shared one."""
    out: dict = {}
    for c in conflicts:
        if pd.isna(c.get("earfcn")):
            continue
        earfcn = int(c["earfcn"])
        if c["type"] in ("Collision", "Confusion"):
            out.setdefault((c["pci"], earfcn), set()).add(c["type"])
        else:  # Mod-N
            for _site, pci in c["members"]:
                out.setdefault((pci, earfcn), set()).add(c["type"])
    return out


def build_site_pci_table(
    selected_sites_df: pd.DataFrame,
    site_labels: dict | None = None,
    conflict_types: dict | None = None,
    filter_to_conflicts: bool = False,
) -> pd.DataFrame:
    site_labels = site_labels or {}
    conflict_types = conflict_types or {}
    rows = []
    for _, sector in selected_sites_df.iterrows():
        pci = sector.get("site_pci")
        if pd.isna(pci):
            continue
        pci_int = int(pci)
        earfcn = sector.get("site_earfcn")
        site_id = sector.get("site_id_inferred")
        types_here = conflict_types.get((pci_int, int(earfcn)), set()) if pd.notna(earfcn) else set()
        if filter_to_conflicts and not types_here:
            continue
        rows.append(
            {
                "Site": site_labels.get(site_id, site_id),
                "PCI": pci_int,
                "Mod3 (PSS)": pci_int % 3,
                "Mod6": pci_int % 6,
                "Mod7": pci_int % 7,
                "Mod8": pci_int % 8,
                "Mod30 (SSS group)": pci_int % 30,
                "EARFCN": earfcn,
                "Samples": sector.get("samples"),
                "Conflict": ", ".join(sorted(types_here)),
                "Synthetic": bool(sector.get("is_synthetic", False)),
            }
        )
    return pd.DataFrame(rows)


def build_neighbor_table(selected_site_ids: list, pair_summary_df: pd.DataFrame, site_labels: dict | None = None) -> pd.DataFrame:
    site_labels = site_labels or {}
    if pair_summary_df.empty or "from_site_id_inferred" not in pair_summary_df.columns:
        return pd.DataFrame()

    mask = pair_summary_df["from_site_id_inferred"].isin(selected_site_ids)
    neighbors = pair_summary_df.loc[mask].copy()
    if neighbors.empty:
        return pd.DataFrame()

    neighbors["from_site_id_inferred"] = neighbors["from_site_id_inferred"].map(lambda s: site_labels.get(s, s))
    neighbors["to_site_id_inferred"] = neighbors["to_site_id_inferred"].map(lambda s: site_labels.get(s, s))
    neighbors = neighbors.rename(
        columns={
            "from_site_id_inferred": "Site",
            "from_pci": "From PCI",
            "to_site_id_inferred": "Neighbor Site",
            "to_pci": "Neighbor PCI",
            "handover_count": "Handover Count",
            "unique_sessions": "Unique Sessions",
        }
    )
    cols = ["Site", "From PCI", "Neighbor Site", "Neighbor PCI", "Handover Count", "Unique Sessions"]
    cols = [c for c in cols if c in neighbors.columns]
    return neighbors[cols].sort_values("Handover Count", ascending=False).reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="PCI Optimization Map", layout="wide")
    st.title("PCI Optimization — Site / Sector Map")
    st.caption(
        "Test-only visualization (ML/tests/Pci_optimization). Sector wedges + PCI colors mirror the "
        "live frontend map; PCI handover pairs use the corrected, frontend-consistent transition detector."
    )

    with st.sidebar.form("pci_map_controls"):
        st.header("Data source")
        project_id = st.number_input("Project ID", value=DEFAULT_PROJECT_ID, step=1, min_value=1)
        region = st.text_input("Region", value=DEFAULT_REGION)
        operator = st.text_input("Operator", value=DEFAULT_OPERATOR)
        # 5 minimum: a Confusion+Collision cluster (3 sites) and a separate
        # Collision-only pair (2 sites) need to coexist so each rule can be
        # shown in isolation — see build_demo_selection(). Extra sites (up
        # to 7) are real, conflict-free filler (smallest-PCI ranking).
        num_sites = st.number_input("Number of sites", value=5, min_value=5, max_value=7, step=1)
        # Synthetic injection is always on now (neither project has a real
        # same-EARFCN case — see inject_synthetic_collision_confusion
        # docstring) and always builds two SEPARATE examples, not a toggle
        # anymore. Polygon filter: always off (needs the Electron/Node
        # PythonBridge on localhost:5224, only reachable inside the desktop
        # app).
        polygon_filter = False
        col_load, col_refresh = st.columns(2)
        load_clicked = col_load.form_submit_button("Load")
        refresh_clicked = col_refresh.form_submit_button("Refresh from DB")

    st.sidebar.header("PCI rule view")
    # Separate, independent detectors/toggles (not one merged "Collision +
    # Confusion" label): Collision = 2+ shown sites sharing a PCI+EARFCN,
    # no serving-site concept. Confusion = a serving site with 2+ neighbors
    # reporting the same PCI+EARFCN. Outside the form so toggling either one
    # updates the view immediately, without needing to click Load again.
    show_collision = st.sidebar.checkbox(
        "Collision", value=True, help="Sites (from the ones shown) that share the same PCI on the same EARFCN."
    )
    show_confusion = st.sidebar.checkbox(
        "Confusion", value=True,
        help="A serving site with 2+ neighbor sites that all report the same PCI on the same EARFCN.",
    )
    # Mod3 = PSS group, the one with real 3GPP planning meaning. Mod1 is
    # trivial (every PCI % 1 == 0 — will flag almost every same-EARFCN
    # pair) and Mod7/Mod9 have no standard 3GPP meaning; all four are here
    # because they were explicitly requested, not because they're all
    # equally meaningful network-planning checks.
    # A single "Mod3" checkbox used to light up every remainder group (0,
    # 1, AND 2) at once, which is why the map looked like "everything" was
    # flagged — mod_n only has mod_n possible buckets, so with 10+ real
    # sectors per site most of them land in some bucket that also has a
    # cross-site match. Now each checked Mod-N rule gets its own remainder
    # picker (0..N-1) so only ONE specific group shows at a time.
    show_mod: dict[int, bool] = {}
    mod_remainder: dict[int, int] = {}
    for n in MOD_RULE_VALUES:
        show_mod[n] = st.sidebar.checkbox(
            f"Mod{n}" + (" (PSS group)" if n == 3 else ""),
            value=False,
            help=f"2+ shown sites whose PCI % {n} matches on the same EARFCN, even with different actual PCIs."
            + (" Trivial — PCI % 1 is always 0." if n == 1 else ""),
        )
        if show_mod[n]:
            mod_remainder[n] = st.sidebar.selectbox(
                f"  Mod{n} remainder to show",
                options=list(range(n)),
                key=f"mod_remainder_{n}",
                help=f"Only PCIs where PCI % {n} equals this value.",
            )

    if "pci_map_loaded" not in st.session_state:
        st.session_state["pci_map_loaded"] = False

    if load_clicked or refresh_clicked:
        st.session_state["pci_map_loaded"] = True
        st.session_state["pci_map_args"] = (int(project_id), region, operator, int(num_sites), bool(polygon_filter))
        st.session_state["force_refresh"] = bool(refresh_clicked)

    if not st.session_state["pci_map_loaded"]:
        st.info("Set project/region/operator in the sidebar and click Load.")
        return

    proj_id, region, operator, num_sites, polygon_filter = st.session_state["pci_map_args"]
    artifact_path = get_artifact_path(proj_id)

    try:
        if st.session_state.get("force_refresh") or not artifact_path.exists():
            with st.spinner(f"Fetching project {proj_id} from the DB..."):
                raw_data = fetch_pci_map_data_from_db(proj_id, region, operator, polygon_filter)
                save_artifact(raw_data, artifact_path)
            st.success(f"Saved snapshot to {artifact_path.relative_to(PROJECT_ROOT)}")
        else:
            raw_data = load_artifact(artifact_path)
            st.caption(
                f"Loaded local snapshot from {artifact_path.relative_to(PROJECT_ROOT)} "
                f"(saved {raw_data.get('saved_at', 'unknown time')}). Click 'Refresh from DB' for live data."
            )
    except Exception as exc:  # noqa: BLE001 - surface DB/data errors directly in the UI
        st.error(f"Failed to load PCI map data: {exc}")
        return

    try:
        site_df, events_df, selected_sites_df, selected_site_ids, demo_notes = build_demo_selection(
            raw_data["site_df"], raw_data["events_df"], num_sites
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to build demo site selection: {exc}")
        return
    for note in demo_notes:
        st.warning(note)

    confusion_conflicts = detect_pci_confusion(events_df, site_ids_filter=set(selected_site_ids))
    collision_conflicts = detect_pci_collisions(selected_sites_df)

    # Mod1/3/7/9: in practice these occur naturally among 5+ real sites
    # (coarse groupings collide far more easily than an exact PCI match),
    # so the synthetic fallback rarely fires — but if a project/selection
    # genuinely has none, inject one rather than showing an empty check.
    mod_conflicts: dict[int, list] = {}
    used_pcis = set(pd.to_numeric(selected_sites_df["site_pci"], errors="coerce").dropna().astype(int).tolist())
    for mod_n in MOD_RULE_VALUES:
        found = detect_pci_mod_conflicts(selected_sites_df, mod_n)
        if not found:
            try:
                selected_sites_df, mod_note = inject_synthetic_mod_conflict(
                    selected_sites_df, selected_site_ids, mod_n, used_pcis
                )
                st.warning(mod_note)
                found = detect_pci_mod_conflicts(selected_sites_df, mod_n)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not find or synthesize a Mod{mod_n} example: {exc}")
        mod_conflicts[mod_n] = found

    # site_id_inferred is an internal join key, not a display name — ~46% of
    # real sites in this project have it partly asterisked because the
    # Android OS itself redacts the Tracking Area Code for privacy
    # (confirmed in tbl_network_log.primary_cell_info_1: "mTac=6***5"),
    # baked into the raw drive-test data. Map to clean "Site N" labels
    # everywhere instead of showing that raw string as if it were a name.
    site_labels = assign_site_labels(selected_site_ids)

    # Filter each checked Mod-N rule down to just the ONE remainder picked
    # in the sidebar, instead of every remainder group at once (mod_n only
    # has mod_n buckets, so with 10+ real sectors per site, showing all of
    # them at once lights up most of a site's sectors — see mod_remainder).
    active_mod_conflicts: list = []
    for n in MOD_RULE_VALUES:
        if not show_mod[n]:
            continue
        remainder = mod_remainder[n]
        matching = [c for c in mod_conflicts[n] if c["mod_value"] == remainder]
        if not matching:
            st.info(f"No Mod{n} conflict with remainder {remainder} among the currently shown sites — try a different remainder in the sidebar.")
        active_mod_conflicts += matching

    active_conflicts = (
        (collision_conflicts if show_collision else [])
        + (confusion_conflicts if show_confusion else [])
        + active_mod_conflicts
    )
    conflict_types = conflict_types_by_key(active_conflicts)
    active_labels = (
        (["Collision"] if show_collision else [])
        + (["Confusion"] if show_confusion else [])
        + [f"Mod{n}={mod_remainder[n]}" for n in MOD_RULE_VALUES if show_mod[n]]
    )

    if not active_labels:
        st.warning("No PCI rule is checked in the sidebar — nothing is being highlighted.")
    elif active_conflicts:
        st.markdown(f"### PCI issues shown ({' + '.join(active_labels)})")
        st.dataframe(build_conflict_summary_table(active_conflicts, site_labels), use_container_width=True)
    else:
        st.info("No PCI issues of the selected type(s) found among the currently shown sites.")

    # Recompute pair_summary_df from events_df rather than reusing
    # raw_data["pair_summary_df"] as-is, so the synthetic handover pair (if
    # injected above) shows up in the neighbor table too.
    pair_summary_df = _build_pair_summary(events_df)

    st.subheader(
        f"{len(selected_site_ids)} site(s) shown "
        f"(of {raw_data['site_df']['site_id_inferred'].nunique()} available in project {proj_id})"
    )

    fmap = render_map(selected_sites_df, site_labels=site_labels, conflict_types=conflict_types, filter_to_conflicts=True)
    st_folium(fmap, width=None, height=700, key="pci_map")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Site PCI values (filtered to the active issue type(s))")
        st.dataframe(
            build_site_pci_table(selected_sites_df, site_labels=site_labels, conflict_types=conflict_types, filter_to_conflicts=True),
            use_container_width=True,
        )
    with col2:
        st.markdown("### Observed neighbors (real + synthetic PCI handovers)")
        neighbor_table = build_neighbor_table(selected_site_ids, pair_summary_df, site_labels=site_labels)
        if neighbor_table.empty:
            st.info("No observed handovers involving these sites in the filtered log data.")
        else:
            st.dataframe(neighbor_table, use_container_width=True)


if __name__ == "__main__":
    main()
