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
import random
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import folium
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from shapely.wkt import loads as load_wkt
from sklearn.neighbors import BallTree
from sqlalchemy import text
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
    _polygon_filter_logs,
    _resolve_engine,
)
from tools.lte_prediction import ml_engine
from utils.python_bridge import _filter_complete_site_prediction_identity

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


def _load_project_polygon_wkts(project_id: int, current_engine) -> list[str]:
    """
    Real project boundary from map_regions.region (tbl_project_id, status=1)
    -- the same table ml_engine._load_project_polygons() reads. That function
    tries the Electron/Node PythonBridge (localhost:5224) FIRST whenever
    PYTHON_BRIDGE_BASE_URL/SIGNAL_TRACKERS_BRIDGE_URL is set in .env, and
    does NOT fail over gracefully if nothing is actually listening there --
    confirmed directly: bridge_enabled() is True in this repo's env, but
    calling it standalone (no Electron running) raises a hard
    ConnectionRefusedError instead of falling back to the DB. Calling
    ml_engine._load_project_polygons_from_db() directly skips the bridge
    attempt entirely and hits map_regions the same way, so this works the
    same whether or not the desktop app is open.
    """
    polygons = ml_engine._load_project_polygons_from_db(int(project_id), current_engine)
    return [p.wkt for p in polygons]


def _filter_sites_within_polygon(site_df: pd.DataFrame, polygons: list) -> tuple[pd.DataFrame, dict]:
    """
    Restricts site_df to sectors whose (site_lat, site_lon) falls inside the
    project's real polygon(s), using the same contains-check +
    coordinate-swap-fallback ml_engine._apply_drive_polygon_filter already
    uses elsewhere in this repo (map_regions.region WKT points are stored
    as (lat, lon) for this data, confirmed directly against project 193 --
    non-standard vs. the (lon, lat) order _filter_df_by_polygons assumes,
    which is exactly what the swap fallback is for).
    """
    if not polygons or site_df.empty:
        return site_df, {"applied": False, "polygons_found": len(polygons), "rows_before": len(site_df), "rows_after": len(site_df)}

    work = site_df.rename(columns={"site_lat": "lat", "site_lon": "lon"}).copy()
    filtered = ml_engine._filter_df_by_polygons(work, polygons)
    swapped = False
    if filtered.empty:
        swapped_polygons = ml_engine._swap_polygon_coords(polygons)
        filtered = ml_engine._filter_df_by_polygons(work, swapped_polygons)
        swapped = True
    filtered = filtered.rename(columns={"lat": "site_lat", "lon": "site_lon"})
    stats = {
        "applied": True,
        "polygons_found": len(polygons),
        "rows_before": len(site_df),
        "rows_after": len(filtered),
        "swapped": swapped,
    }
    return filtered, stats


def _fetch_site_prediction_metadata(project_id: int, operator: str, current_engine) -> pd.DataFrame:
    """
    The REAL site/sector source — matches ml_engine.fetch_site_data(), what
    the live frontend map itself renders from. NOT site_ml/site_noMl (the
    old _fetch_site_metadata in pci_optimization_dataset_test.py): that pair
    of tables has no operator filter at all and silently mixes every
    carrier together (confirmed on project 193: 712 rows = 401 airtel + 185
    jio 4g + 124 vi india + 2 junk "000 000" rows, all merged into one pool).

    site_prediction stores ~6 duplicate re-inserted rows per real sector
    (one per past prediction/upload run for this project — group sizes are
    almost all exactly 6). The live frontend collapses these via a Map
    keyed by the 5-field identity (site|cell_id|sector|band|operator) the
    backend attaches as site_prediction_key (SiteLegend.jsx's
    readLegendIdentityKey / useSiteData.js's getSitePredictionMergeKey both
    consume this exact key) — utils.python_bridge.
    _filter_complete_site_prediction_identity() builds that same key and
    drops rows missing any of the 5 fields, but does NOT itself dedupe
    identical keys against each other; that final drop_duplicates step is
    what's actually needed here too. Verified end-to-end against project
    193: raw Airtel rows=1100 -> identity-complete=398 ->
    drop_duplicates(site_prediction_key)=199 sectors / 62 sites — matching
    the live map's own "Cell Sectors BY OPERATOR: Airtel 199" legend count
    exactly (previously site_ml/site_noMl gave a meaningless, mixed-
    operator "92").
    """
    query = text(
        """
        SELECT site_prediction.*, cluster AS provider, cluster AS operator_name
        FROM site_prediction
        WHERE tbl_project_id = :project_id
        """
    )
    raw_df = pd.read_sql(query, current_engine, params={"project_id": int(project_id)})
    if raw_df.empty:
        return pd.DataFrame()

    complete_df = _filter_complete_site_prediction_identity(raw_df, endpoint="pci_map_dashboard")
    if complete_df.empty:
        return pd.DataFrame()

    requested_operator = str(operator or "").strip().lower()
    if requested_operator and requested_operator not in {"all", "auto"}:
        operator_candidates = ["operator", "operator_name", "cluster", "provider", "network"]
        mask = pd.Series(False, index=complete_df.index)
        for col in operator_candidates:
            if col not in complete_df.columns:
                continue
            mask = mask | complete_df[col].astype(str).str.strip().str.lower().eq(requested_operator)
        complete_df = complete_df.loc[mask].copy()
    if complete_df.empty:
        return pd.DataFrame()

    deduped = complete_df.drop_duplicates(subset=["site_prediction_key"], keep="first").copy()
    deduped = deduped.rename(
        columns={
            "site": "site_id_inferred",
            "latitude": "site_lat",
            "longitude": "site_lon",
            "pci": "site_pci",
            "earfcn": "site_earfcn",
            "azimuth": "site_azimuth_deg",
            "cell_id": "site_cell_id_representative",
            "cluster": "network",
        }
    )
    for col in ("site_lat", "site_lon", "site_pci", "site_earfcn"):
        if col in deduped.columns:
            deduped[col] = pd.to_numeric(deduped[col], errors="coerce")
    # site_prediction has no per-sector sample-count/beamwidth/range fields
    # (that's a site_ml/site_noMl-only concept) -- "samples" here is just a
    # constant per-sector weight (1) so build_real_selection's ranking
    # degrades to "sites with more resolved sectors first" instead of
    # pretending to have real drive-sample counts this table doesn't carry.
    # azimuth_soft/beamwidth/median_sample_distance stay absent, so
    # normalize_beamwidth()/normalize_sector_range() fall back to their
    # documented defaults (30 deg / 220m) exactly as they already do for
    # any sector missing those fields.
    deduped["samples"] = 1
    deduped["site_azimuth_soft_deg"] = np.nan
    deduped["site_source_table"] = "site_prediction"
    deduped["is_synthetic"] = False
    return deduped.reset_index(drop=True)


CSV_DATASET_PATH = Path(__file__).resolve().parent / "Site_Data_Template_103815_updated.csv"


def _load_csv_site_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Pure site/sector inventory from a CSV — no DB, no drive-test/handover
    data at all (no session_id, no observed transitions, so events_df
    stays empty for this data source). This is exactly the "can we still
    do PCI optimization without handover" case: Collision/Mod-N/Grouped
    all work fine, since build_neighbor_graph already falls back to pure
    distance-based edges when there's no handover data to check against;
    Confusion is inherently defined by real observed handovers and will
    correctly always report 0 here, not an error.

    Columns confirmed from Site_Data_Template_103815_updated.csv: site,
    sector, cell_id, longitude, latitude, pci, azimuth, band, earfcn,
    height, m_tilt, e_tilt, transmit power (always empty), frequency,
    cluster (constant "Taiwan" — a region label, not an operator/carrier
    name), Technology (constant "4G"). No samples/beamwidth/range/soft-
    azimuth columns, same gap as site_prediction — same fallback handling
    (normalize_beamwidth/normalize_sector_range default to 30 deg/220m).
    longitude/latitude are stored as integers with the decimal point
    stripped (e.g. 25007083 -> 25.007083, 121456528 -> 121.456528) —
    confirmed against Taiwan's real lat/lon range once divided by 1e6.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={
            "site": "site_id_inferred",
            "longitude": "site_lon",
            "latitude": "site_lat",
            "pci": "site_pci",
            "earfcn": "site_earfcn",
            "azimuth": "site_azimuth_deg",
            "cluster": "network",
            "cell_id": "site_cell_id_representative",
        }
    )
    for col in ("site_lat", "site_lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce") / 1_000_000.0
    for col in ("site_pci", "site_earfcn", "site_azimuth_deg"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["samples"] = 1
    df["site_azimuth_soft_deg"] = np.nan
    df["site_source_table"] = f"csv:{csv_path.name}"
    df["is_synthetic"] = False
    return df.dropna(subset=["site_lat", "site_lon", "site_pci", "site_earfcn", "site_id_inferred"]).reset_index(drop=True)


def fetch_pci_map_data_from_db(project_id: int, region: str, operator: str, polygon_filter: bool) -> dict:
    """
    The only place this module touches the DB. Site/sector data comes from
    site_prediction (_fetch_site_prediction_metadata) — the same table and
    identity-dedup the live frontend map itself uses, operator-filtered
    correctly (see that function's docstring for the confirmed 199-sector
    parity check). Also fetches the project's real polygon (map_regions.
    region) via the direct-DB path (_load_project_polygon_wkts) so it can
    be drawn on the map and used to keep the site selection inside it.
    `polygon_filter` still gates the older log-row-level filter
    (_polygon_filter_logs, which goes through ml_engine.
    _apply_drive_polygon_filter and DOES attempt the Electron bridge first)
    for backward compatibility, but the polygon geometry itself and the
    site-level polygon filter (_filter_sites_within_polygon, called by
    main()) are both bridge-free.
    """
    current_engine = _resolve_engine(region)
    with current_engine.connect() as conn:
        session_ids, project_meta = _fetch_project_session_ids(project_id, conn)
        raw_log_df = _fetch_network_logs(session_ids, operator, conn, primary_only=True)
        log_df, log_polygon_stats = _polygon_filter_logs(raw_log_df, project_id, current_engine, enabled=polygon_filter)

    site_df = _fetch_site_prediction_metadata(project_id, operator, current_engine)
    site_polygon_stats = {"enabled": False, "skipped": True, "rows_before": len(site_df), "rows_after": len(site_df)}
    polygon_wkts = _load_project_polygon_wkts(project_id, current_engine)

    if log_df.empty:
        raise ValueError(f"No network log rows for project_id={project_id} (session_ids={session_ids}).")
    if site_df.empty:
        raise ValueError(f"No site_prediction rows for project_id={project_id}, operator={operator}.")

    events_df = _detect_pci_transitions(log_df)
    enriched_events_df = _enrich_transition_sites(events_df, site_df)
    pair_summary_df = _build_pair_summary(enriched_events_df)

    return {
        "project_id": int(project_id),
        "region": region,
        "operator": operator,
        "polygon_filter": bool(polygon_filter),
        "polygon_wkt": polygon_wkts,
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


def detect_pci_collisions(graph: nx.Graph) -> list[dict]:
    """
    Independent Collision detector: two NEIGHBOR sectors (a graph edge —
    real observed handover OR within the configured distance threshold,
    per build_neighbor_graph) sharing the exact same PCI on the same
    EARFCN. Evaluated over actual neighbor pairs, not blindly across the
    whole selection — this naturally includes same-site sector pairs
    (their distance is ~0m, so they're always graph-connected), matching
    real LTE PCI planning practice: a UE can hand over between two
    sectors of the same physical site just as easily as between two
    different sites, so both need checking. Run separately from
    Confusion so they're genuinely independent toggles in the UI, not
    the same check shown under two names.
    """
    # Grouped by CONNECTED COMPONENT of collision edges, not a flat union
    # of every site seen under this (pci, earfcn) -- confirmed necessary:
    # a flat union merges two totally unrelated collisions that happen to
    # reuse the same PCI number into one fake combined group, even when
    # the sites involved have no path to each other in the graph at all.
    edges_by_key: dict[tuple, list] = {}
    for u, v in graph.edges():
        if u[2] != v[2] or u[1] != v[1]:  # different EARFCN or different PCI -- not a collision
            continue
        key = (u[1], u[2])  # (pci, earfcn)
        edges_by_key.setdefault(key, []).append((u, v))

    collisions: list[dict] = []
    for (pci_val, earfcn_val), edges in edges_by_key.items():
        component_graph = nx.Graph()
        component_graph.add_edges_from(edges)
        for component in nx.connected_components(component_graph):
            sites = sorted({node[0] for node in component})
            collisions.append({"type": "Collision", "pci": pci_val, "earfcn": earfcn_val, "sites": sites})
    return collisions


def detect_pci_order_conflicts(graph: nx.Graph, max_order: int = 5) -> dict[int, list[dict]]:
    """
    Same-PCI+EARFCN reuse bucketed by exact hop-distance ("order") in the
    neighbor graph, order 1 through max_order. Order 1 = direct neighbor
    reuse (the same pairs detect_pci_collisions flags, one row per PAIR
    here instead of one row per PCI/EARFCN group). Order 2 is what
    Confusion structurally is (two cells both 1 hop from a common serving
    cell -> 2 hops from each other) -- Confusion itself is still the real,
    handover-verified detector; this is a purely graph-structural version.
    Orders 3-5 have no standard 3GPP/RF interference meaning -- a UE
    realistically never hears two cells that many hops apart at once, so
    these are a PCI reuse-DISTANCE audit metric, not a fault to fix.
    """
    nodes_by_pci_earfcn: dict[tuple, list] = {}
    for n in graph.nodes:
        nodes_by_pci_earfcn.setdefault((n[1], n[2]), []).append(n)

    results: dict[int, list[dict]] = {order: [] for order in range(1, max_order + 1)}
    for (pci, earfcn), members in nodes_by_pci_earfcn.items():
        if len(members) < 2:
            continue
        member_set = set(members)
        seen_pairs: set = set()
        for source in members:
            hop_lengths = nx.single_source_shortest_path_length(graph, source, cutoff=max_order)
            for other, hops in hop_lengths.items():
                if hops == 0 or other not in member_set:
                    continue
                pair_key = frozenset((source, other))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                results[hops].append({"type": f"Order{hops}", "pci": pci, "earfcn": earfcn, "sites": sorted({source[0], other[0]})})
    return results


def build_order_conflict_table(
    order_conflicts_by_order: dict[int, list[dict]],
    confusion_conflicts: list[dict],
) -> pd.DataFrame:
    """
    One row per order (1-5): total same-PCI+EARFCN pairs at that exact
    hop distance, plus how many of those same pairs are also flagged by
    the real, handover-verified Confusion detector -- cross-referenced by
    matching (pci, earfcn, site-pair), not assumed from the order number.

    "Also Collision" is NOT computed by re-deriving pairs from
    detect_pci_collisions' grouped output -- a connected COMPONENT can
    still be a chain (A-B-C with no direct A-C edge), so two members of
    the same reported collision group aren't necessarily a direct edge
    themselves. Order 1 IS the direct-edge/same-PCI definition (identical
    criteria to Collision), so "Also Collision" is exact and trivial by
    construction: equals Pairs at order 1, always 0 at order 2+ (Collision
    only ever fires on a direct 1-hop edge).
    """
    confusion_pairs: set = set()
    for c in confusion_conflicts:
        for a, b in combinations(c["neighbor_sites"], 2):
            confusion_pairs.add((c["pci"], c["earfcn"], frozenset((a, b))))

    rows = []
    for order in sorted(order_conflicts_by_order):
        pairs = order_conflicts_by_order[order]
        confusion_hits = sum(1 for p in pairs if (p["pci"], p["earfcn"], frozenset(p["sites"])) in confusion_pairs)
        rows.append(
            {
                "Order": order,
                "Pairs (same PCI+EARFCN)": len(pairs),
                "Also Collision": len(pairs) if order == 1 else 0,
                "Also Confusion": confusion_hits,
            }
        )
    return pd.DataFrame(rows)


MOD_RULE_VALUES = (1, 3, 6, 7, 8, 9)


def detect_pci_mod_conflicts(graph: nx.Graph, mod_n: int) -> list[dict]:
    """
    Generic PCI-mod-N conflict: two NEIGHBOR sectors (graph edge — real
    handover OR within the distance threshold, including same-site pairs)
    whose real PCIs share the same (PCI % mod_n) remainder on the same
    EARFCN, even when their actual PCI values differ (unlike Collision,
    which needs an exact match). Mod3 is the one with real 3GPP meaning
    here (PSS group — neighbor cells sharing a PSS degrade cross-
    correlation/sync). Mod1 is trivial (every PCI % 1 == 0, so it flags
    almost every neighbor pair) and Mod7/Mod9 have no standard 3GPP
    planning meaning — all three are included because they were
    explicitly requested, not because they're meaningful checks.
    """
    # Grouped by CONNECTED COMPONENT of same-remainder edges, not a flat
    # union of every (site, pci) seen under this remainder -- same fix as
    # detect_pci_collisions, same reason: a flat union merges unrelated
    # mod-N matches with no real path between them into one fake group.
    edges_by_key: dict[tuple, list] = {}
    for u, v in graph.edges():
        if u[2] != v[2]:
            continue
        earfcn = u[2]
        mod_u, mod_v = u[1] % mod_n, v[1] % mod_n
        if mod_u != mod_v:
            continue
        key = (mod_u, earfcn)
        edges_by_key.setdefault(key, []).append((u, v))

    conflicts: list[dict] = []
    for (mod_val, earfcn_val), edges in edges_by_key.items():
        component_graph = nx.Graph()
        component_graph.add_edges_from(edges)
        for component in nx.connected_components(component_graph):
            members = sorted({(node[0], node[1]) for node in component})
            conflicts.append(
                {
                    "type": f"Mod{mod_n}",
                    "mod_n": mod_n,
                    "mod_value": mod_val,
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


def detect_pci_group_conflicts(graph: nx.Graph) -> list[dict]:
    """
    Grouped PCI (3GPP PCI group, N_ID_1): the 504 PCIs (0-503) are divided
    into 168 groups of 3 consecutive values each. The group number is
    `PCI // 3` (integer division), 0-167 — NOT "PCI % 30", which is what
    the Site PCI table's earlier "Mod30 (SSS group)" column actually
    computed. That was a rough approximation, not the real 3GPP formula;
    this is the correct one. Two NEIGHBOR sectors (graph edge — real
    handover OR within the distance threshold, including same-site pairs)
    sharing a PCI group on the same EARFCN share the same underlying
    SSS-derived group even though their exact PCI (and PSS) differ.
    """
    # Grouped by CONNECTED COMPONENT, not a flat union -- same fix as
    # detect_pci_collisions/detect_pci_mod_conflicts.
    edges_by_key: dict[tuple, list] = {}
    for u, v in graph.edges():
        if u[2] != v[2]:
            continue
        earfcn = u[2]
        group_u, group_v = u[1] // 3, v[1] // 3
        if group_u != group_v:
            continue
        key = (group_u, earfcn)
        edges_by_key.setdefault(key, []).append((u, v))

    conflicts: list[dict] = []
    for (group_val, earfcn_val), edges in edges_by_key.items():
        component_graph = nx.Graph()
        component_graph.add_edges_from(edges)
        for component in nx.connected_components(component_graph):
            members = sorted({(node[0], node[1]) for node in component})
            conflicts.append({"type": "Grouped", "group_value": group_val, "earfcn": earfcn_val, "members": members})
    conflicts.sort(key=lambda c: len(c["members"]), reverse=True)
    return conflicts


def inject_synthetic_group_conflict(site_df: pd.DataFrame, selected_site_ids: list, used_pcis: set) -> tuple[pd.DataFrame, str]:
    """Fallback only (Grouped PCI conflicts occur naturally, like Mod-N) —
    same approach as inject_synthetic_mod_conflict but bucketed by PCI//3
    instead of PCI%N."""
    candidates = site_df[site_df["site_id_inferred"].isin(selected_site_ids)].dropna(subset=["site_pci", "site_earfcn"]).copy()
    candidates["site_pci"] = pd.to_numeric(candidates["site_pci"], errors="coerce")
    candidates["site_earfcn"] = pd.to_numeric(candidates["site_earfcn"], errors="coerce")
    candidates = candidates.dropna(subset=["site_pci", "site_earfcn"])
    if len(candidates) < 2:
        raise ValueError("Not enough sector data among selected sites to synthesize a Grouped PCI example.")

    anchor = candidates.iloc[0]
    target_site = candidates.iloc[1]["site_id_inferred"]
    if target_site == anchor["site_id_inferred"] and len(candidates["site_id_inferred"].unique()) > 1:
        target_site = candidates[candidates["site_id_inferred"] != anchor["site_id_inferred"]].iloc[0]["site_id_inferred"]

    earfcn = anchor["site_earfcn"]
    group_val = int(anchor["site_pci"]) // 3
    new_pci = group_val * 3
    while new_pci in used_pcis or new_pci == int(anchor["site_pci"]):
        new_pci += 1
        if new_pci >= group_val * 3 + 3:
            raise ValueError("No free PCI value in that group available to synthesize a Grouped PCI example.")

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
        f"Synthetic (Grouped PCI): real site {target_site} given a synthetic PCI {new_pci} on EARFCN {int(earfcn)} "
        f"(PCI group {group_val}, i.e. PCI//3), matching real site {anchor['site_id_inferred']}'s real PCI "
        f"{int(anchor['site_pci'])}'s group — no natural Grouped PCI conflict existed among the selected sites."
    )
    return site_df, description


def detect_co_centric_groups(selected_sites_df: pd.DataFrame, azimuth_tolerance_deg: float = 15.0) -> list[dict]:
    """
    Co-centric: sectors at the SAME physical site pointing in the SAME
    direction (within azimuth_tolerance_deg) but on DIFFERENT EARFCNs —
    same antenna location and coverage beam, different frequency layer/
    technology (e.g. one site/sector broadcasting both LTE1800 and
    LTE2100). This is structural/informational, not a fault by itself —
    operators typically want a *consistent, predictable* PCI relationship
    across a site's co-centric cells so they stay easy to identify and
    manage; shown here to make that structure visible on the map (drawn
    with a distinct color, not the red "conflict" border the other rules
    use), not flagged as wrong.
    """
    df = selected_sites_df.dropna(subset=["site_pci", "site_earfcn", "site_id_inferred"]).copy()
    if df.empty:
        return []
    df["site_pci"] = pd.to_numeric(df["site_pci"], errors="coerce")
    df["site_earfcn"] = pd.to_numeric(df["site_earfcn"], errors="coerce")
    df["_az"] = df.apply(
        lambda r: r["site_azimuth_soft_deg"] if pd.notna(r.get("site_azimuth_soft_deg")) else r.get("site_azimuth_deg", 0.0),
        axis=1,
    )
    df["_az"] = pd.to_numeric(df["_az"], errors="coerce")
    df = df.dropna(subset=["site_pci", "site_earfcn", "_az"])
    df["_az_bucket"] = (df["_az"] / azimuth_tolerance_deg).round().astype(int)

    groups: list[dict] = []
    for (site_id, az_bucket), grp in df.groupby(["site_id_inferred", "_az_bucket"]):
        if grp["site_earfcn"].nunique() < 2:
            continue
        members = sorted({(int(row.site_earfcn), int(row.site_pci)) for row in grp.itertuples()})
        groups.append(
            {
                "type": "Co-centric",
                "site": site_id,
                "azimuth_deg": float(grp["_az"].mean()),
                "members": members,  # (earfcn, pci) pairs
            }
        )
    groups.sort(key=lambda g: len(g["members"]), reverse=True)
    return groups


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
    #
    # Also adds a DIRECT event between site_b and site_c themselves, not
    # just servingsite->B and servingsite->C — confirmed necessary: under
    # the neighbor-graph model (build_neighbor_graph), being each only a
    # neighbor of a common third site does NOT make B and C neighbors of
    # EACH OTHER (they can be on opposite sides of the serving site, well
    # over the distance threshold apart) — detect_pci_collisions() found
    # 0 collisions here without this direct edge, because Site 2/Site 3
    # had no real OR distance-based edge connecting them despite both
    # being confused-neighbors of Site 1.
    new_event_rows = []
    pairs = [(serving_site, site_b), (serving_site, site_c), (site_b, site_c)]
    for idx, (from_site, to_site) in enumerate(pairs, start=1):
        from_pci_val = str(int(from_pci)) if from_site == serving_site else str(synthetic_pci)
        new_event_rows.append(
            {
                "session_id": f"SYNTH-{idx}",
                "from_pci": from_pci_val,
                "to_pci": str(synthetic_pci),
                "from_earfcn": str(int(from_earfcn)) if from_site == serving_site else str(int(synthetic_earfcn)),
                "to_earfcn": str(int(synthetic_earfcn)),
                "from_site_id_inferred": from_site,
                "to_site_id_inferred": to_site,
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
        f"events to its real neighbor sites {site_b} and {site_c} (plus one direct {site_b}<->{site_c} event "
        f"so they're graph-neighbors of each other too, not just both of {serving_site}), all three assigned "
        f"synthetic PCI {synthetic_pci} on EARFCN {synthetic_earfcn}."
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

    Picks the CLOSEST such pair (haversine), not just the first one found
    in arbitrary order — confirmed necessary: under the neighbor-graph
    model (build_neighbor_graph), Collision only fires for sectors that
    are graph-neighbors (real handover OR within the distance threshold).
    An arbitrary "not confused" pair can easily be many km apart, which
    made this example silently stop being detected as a collision at all
    once distance-based neighbor checking was added.
    """
    confused_pairs: set = set()
    if not events_df.empty and {"from_site_id_inferred", "to_site_id_inferred"}.issubset(events_df.columns):
        for _, grp in events_df.dropna(subset=["from_site_id_inferred", "to_site_id_inferred"]).groupby("from_site_id_inferred"):
            neighbors = sorted(grp["to_site_id_inferred"].unique().tolist())
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    confused_pairs.add(frozenset((neighbors[i], neighbors[j])))

    candidates = (
        site_df.dropna(subset=["site_id_inferred", "site_lat", "site_lon"])
        .loc[lambda d: (d["site_id_inferred"].astype(str) != "0") & (~d["site_id_inferred"].isin(exclude_site_ids))]
        .assign(
            site_lat=lambda d: pd.to_numeric(d["site_lat"], errors="coerce"),
            site_lon=lambda d: pd.to_numeric(d["site_lon"], errors="coerce"),
        )
        .dropna(subset=["site_lat", "site_lon"])
        .drop_duplicates("site_id_inferred")
    )
    candidate_ids = candidates["site_id_inferred"].tolist()
    latlon = dict(zip(candidates["site_id_inferred"], zip(candidates["site_lat"], candidates["site_lon"])))

    site_x = site_y = None
    best_distance = None
    for i in range(len(candidate_ids)):
        for j in range(i + 1, len(candidate_ids)):
            a, b = candidate_ids[i], candidate_ids[j]
            if frozenset((a, b)) in confused_pairs:
                continue
            lat1, lon1 = latlon[a]
            lat2, lon2 = latlon[b]
            dist = _haversine_scalar_m(lat1, lon1, lat2, lon2)
            if best_distance is None or dist < best_distance:
                best_distance, site_x, site_y = dist, a, b
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
        f"Synthetic (Collision only): real sites {site_x} and {site_y}, {best_distance:.0f}m apart (the closest "
        f"available pair) — confirmed NOT linked as neighbors of any common serving site in the real handover "
        f"data — both assigned synthetic PCI {synthetic_pci} on EARFCN {int(synthetic_earfcn)}. No handover "
        "events added, so Confusion will never flag this pair; Collision relies on the distance threshold."
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


def build_real_selection(site_df: pd.DataFrame, events_df: pd.DataFrame, num_sites: int) -> tuple[pd.DataFrame, list, str]:
    """
    Real-project mode: NO synthetic injection at all -- every sector shown
    is real data from this project/operator/polygon selection. Ranked by
    total observed samples (richest/most-driven sites first, the most
    representative real sites), not the demo's smallest-PCI cosmetic
    ranking. num_sites <= 0 means "every plausible real site" (no cap) --
    use this to test detection/optimizer behavior at full project scale
    (e.g. project 193/Airtel has ~92 real sites / ~166 sectors within its
    polygon), not just the small synthetic 5-site example.
    """
    sites_with_id = _plausible_sites(site_df)
    site_rank = sites_with_id.groupby("site_id_inferred")["samples"].sum().sort_values(ascending=False)
    selected_site_ids = list(site_rank.index) if num_sites <= 0 else list(site_rank.head(num_sites).index)
    selected_sites_df = sites_with_id[sites_with_id["site_id_inferred"].isin(selected_site_ids)].copy()
    note = (
        f"Real project mode: {len(selected_site_ids)} real site(s) / {len(selected_sites_df)} real sector(s) "
        "loaded (no synthetic data), ranked by total observed samples."
    )
    return selected_sites_df, selected_site_ids, note


CONFLICT_BORDER_COLOR = "#DC2626"
# Co-centric is structural/informational, not a fault -- distinct violet,
# not the red used for every genuine conflict type (Collision, Confusion,
# Mod-N, Grouped).
CO_CENTRIC_BORDER_COLOR = "#7C3AED"


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
    """Handles five conflict shapes — Confusion (serving_site + neighbor_sites),
    Collision (flat sites list), Mod-N (members: (site, pci) pairs sharing
    a PCI%N remainder), Grouped (members: (site, pci) pairs sharing a
    PCI//3 group), and Co-centric (single site + azimuth, members:
    (earfcn, pci) pairs — no single EARFCN, that's the whole point) — kept
    as genuinely distinct rows/types, never merged. PCI/EARFCN columns are
    always strings across every row (mixed int/"" in one column breaks
    Streamlit's Arrow serialization)."""
    rows = []
    for c in conflicts:
        detail = ""
        earfcn = str(int(c["earfcn"])) if c.get("earfcn") is not None and pd.notna(c.get("earfcn")) else ""
        if c["type"] == "Confusion":
            serving_label = site_labels.get(c["serving_site"], c["serving_site"])
            involved = ", ".join(site_labels.get(s, s) for s in c["neighbor_sites"])
            pci = str(c["pci"])
        elif c["type"] == "Collision":
            serving_label = ""
            involved = ", ".join(site_labels.get(s, s) for s in c["sites"])
            pci = str(c["pci"])
        elif c["type"] == "Grouped":
            serving_label = ""
            involved = ", ".join(f"{site_labels.get(s, s)} (PCI {p})" for s, p in c["members"])
            pci = ""
            detail = f"PCI Group {c['group_value']} (PCI // 3)"
        elif c["type"] == "Co-centric":
            serving_label = site_labels.get(c["site"], c["site"])
            involved = ", ".join(f"EARFCN {ef} (PCI {p})" for ef, p in c["members"])
            pci = ""
            earfcn = ""  # deliberately blank -- Co-centric spans multiple EARFCNs by definition
            detail = f"Azimuth ~{c['azimuth_deg']:.0f}°"
        else:  # Mod-N
            serving_label = ""
            involved = ", ".join(f"{site_labels.get(s, s)} (PCI {p})" for s, p in c["members"])
            pci = ""
            detail = f"PCI % {c['mod_n']} = {c['mod_value']}"
        rows.append(
            {
                "Type": c["type"],
                "PCI": pci,
                "EARFCN": earfcn,
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
    polygons: list | None = None,
) -> folium.Map:
    """
    `filter_to_conflicts=True` only draws a sector's wedge+PCI label if its
    (pci, earfcn) is a key in `conflict_types` — everything else is skipped
    entirely, so toggling Collision/Confusion in the sidebar actually
    changes what's shown, not just its border color. Site markers/labels
    still always draw (so you can see where a serving site is even if none
    of its own sectors match), and every wedge that IS drawn in this mode
    is, by construction, a conflict, so it's always given the red border.

    `polygons`, if given, is a list of shapely Polygons from the project's
    real map_regions.region boundary (see _load_project_polygon_wkts) —
    drawn as an unfilled dashed outline so the selected sites' containment
    inside the actual project drive area is visually checkable, not asserted.
    Their own exterior points are folded into the map's fit_bounds too, so
    the whole boundary stays in view together with the sites, not just
    whichever edge happens to be closest to them.
    """
    site_labels = site_labels or {}
    conflict_types = conflict_types or {}
    center_lat = pd.to_numeric(selected_sites_df["site_lat"], errors="coerce").mean()
    center_lon = pd.to_numeric(selected_sites_df["site_lon"], errors="coerce").mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=17, tiles="CartoDB positron", control_scale=True)

    bounds_points: list[tuple[float, float]] = []

    for polygon in polygons or []:
        # map_regions.region WKT points are stored as (lat, lon) for this
        # data (confirmed directly against project 193 — not the standard
        # (lon, lat) WKT convention), so exterior.coords can be handed to
        # folium's [lat, lon] `locations` directly, with no swap needed here
        # (the swap only matters for the shapely Point-in-polygon check in
        # _filter_sites_within_polygon, which assumes the opposite order).
        boundary = [(lat, lon) for lat, lon in polygon.exterior.coords]
        folium.Polygon(
            locations=boundary,
            color="#2563EB",
            weight=3,
            dash_array="10,8",
            fill=False,
            tooltip="Project drive area (map_regions)",
        ).add_to(fmap)
        bounds_points.extend(boundary)

    # In filter_to_conflicts mode, only draw markers/wedges for sites that
    # actually have a conflicted sector -- confirmed necessary at real
    # project scale: drawing all 3,000 site markers (Taiwan CSV) even
    # though only a handful of sites had any flagged sector was the actual
    # multi-minute render hang, not detection (which already finished).
    sites_to_draw = selected_sites_df
    if filter_to_conflicts:
        pci_num = pd.to_numeric(selected_sites_df["site_pci"], errors="coerce")
        earfcn_num = pd.to_numeric(selected_sites_df["site_earfcn"], errors="coerce")
        is_conflicted_row = [
            (int(p), int(e)) in conflict_types if pd.notna(p) and pd.notna(e) else False
            for p, e in zip(pci_num, earfcn_num)
        ]
        relevant_site_ids = set(selected_sites_df.loc[is_conflicted_row, "site_id_inferred"])
        sites_to_draw = selected_sites_df[selected_sites_df["site_id_inferred"].isin(relevant_site_ids)]

    for site_id, site_rows in sites_to_draw.groupby("site_id_inferred"):
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
            types_here = conflict_types.get(key, set()) if key is not None else set()
            is_conflict = bool(types_here)
            # Co-centric-only is structural, not a fault -- violet, not red.
            highlight_color = (
                CO_CENTRIC_BORDER_COLOR if types_here and types_here == {"Co-centric"} else CONFLICT_BORDER_COLOR
            )
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
                tooltip += f" | {'/'.join(sorted(types_here))}"
            if is_synthetic:
                tooltip += " | SYNTHETIC (not real data)"
            folium.Polygon(
                locations=triangle,
                color=highlight_color if is_conflict else color,
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
            border_color = highlight_color if is_conflict else color
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
    "Mod1"/"Mod3"/"Mod7"/"Mod9", "Grouped", "Co-centric") found for it
    among the given conflicts — a key can carry several independently.
    Mod-N/Grouped conflicts have no single PCI (their members can have
    different real PCIs sharing one remainder/group), so they contribute
    one key per (member site's real PCI, EARFCN). Co-centric has no
    single EARFCN either (that's the whole point), so it contributes one
    key per (member PCI, member EARFCN) pair directly."""
    out: dict = {}
    for c in conflicts:
        if c["type"] == "Co-centric":
            for earfcn, pci in c["members"]:
                out.setdefault((pci, earfcn), set()).add(c["type"])
            continue
        if pd.isna(c.get("earfcn")):
            continue
        earfcn = int(c["earfcn"])
        if c["type"] in ("Collision", "Confusion"):
            out.setdefault((c["pci"], earfcn), set()).add(c["type"])
        else:  # Mod-N or Grouped
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


# ============================================================
# Graph-based PCI recommendation (v1 heuristic, pre-NSGA-II)
#
# Everything above this point DETECTS problems. This section
# RECOMMENDS fixes: builds the real observed neighbor graph and
# proposes new PCI values that reduce conflict cost against real
# neighbors. Recommend-only — never mutates selected_sites_df, the
# map, or the artifact. A later phase can swap this greedy solver
# for a proper multi-objective one (e.g. NSGA-II via pymoo), reusing
# the same graph and the same per-candidate cost function as one of
# several competing fitness terms.
# ============================================================

PCI_MIN, PCI_MAX = 0, 503  # full 3GPP PCI range (504 values)


def _candidates_by_distance(current_pci: int) -> list[int]:
    """
    Search order: nearest-to-current-PCI first (current+1, current-1,
    current+2, current-2, ...), not ascending from 0. Real PCI replanning
    wants the smallest possible change, not the lowest free number
    anywhere in 0-503 — e.g. fixing PCI 100 should try 99/101 (same
    group of 3, {99,100,101} — already a different PCI%3 by construction,
    since a group's 3 members always have distinct remainders 0/1/2) and
    only reach further, different groups (98, 102, 97, 103, ...) if
    those don't clear the conflict. Falls back to the full 0-503 range
    only if nothing within it works (defensive; every value gets tried
    exactly once by construction, so this never actually happens).
    """
    candidates = []
    for delta in range(1, PCI_MAX - PCI_MIN + 1):
        up, down = current_pci + delta, current_pci - delta
        if up <= PCI_MAX:
            candidates.append(up)
        if down >= PCI_MIN:
            candidates.append(down)
    return candidates


def _haversine_scalar_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two lat/lon points — same
    formula as compute_offset()/the frontend's own distance math, just
    scalar instead of vectorized (called pairwise, node-by-node)."""
    radius_m = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


DEFAULT_NEIGHBOR_DISTANCE_M = 500.0


def build_neighbor_graph(
    selected_sites_df: pd.DataFrame, events_df: pd.DataFrame, distance_threshold_m: float = DEFAULT_NEIGHBOR_DISTANCE_M
) -> nx.Graph:
    """
    Nodes = every real (site, pci, earfcn) sector in the current
    selection. Two sectors are an EDGE ("neighbors") if EITHER:
      - there's a REAL observed handover between them (events_df, the
        same corrected, frequency-layer-aware data used by
        detect_pci_confusion) -- highest confidence, always trusted; OR
      - they're within `distance_threshold_m` of each other (haversine,
        each sector's own site_lat/site_lon) on the SAME EARFCN --
        the fallback for pairs with no observed handover, so the model
        still works when real mobility data is sparse or missing for a
        given pair.

    This is a single unified rule, not "same site vs different site":
    same-site sector pairs have distance ~0m, so they always satisfy the
    distance criterion automatically and need no special case. A UE can
    hand over between two sectors of the same physical site just as
    easily as between two different sites, so they must be checked for
    Collision/Mod-N/Grouped exactly like any other neighbor pair.

    Real edges get weight = handover count and observed=True; distance-
    only edges get weight 0 and observed=False (so callers can tell real
    vs inferred adjacency apart). A literal networkx.Graph, not an ad hoc
    dict, so later graph-theoretic work doesn't need a rewrite.
    """
    df = selected_sites_df.dropna(subset=["site_pci", "site_earfcn", "site_id_inferred", "site_lat", "site_lon"]).copy()
    df["site_pci"] = pd.to_numeric(df["site_pci"], errors="coerce")
    df["site_earfcn"] = pd.to_numeric(df["site_earfcn"], errors="coerce")
    df["site_lat"] = pd.to_numeric(df["site_lat"], errors="coerce")
    df["site_lon"] = pd.to_numeric(df["site_lon"], errors="coerce")
    df = df.dropna(subset=["site_pci", "site_earfcn", "site_lat", "site_lon"])

    graph = nx.Graph()
    node_by_site_pci: dict[tuple, tuple] = {}
    node_latlon: dict[tuple, tuple] = {}
    for row in df.itertuples():
        node = (row.site_id_inferred, int(row.site_pci), int(row.site_earfcn))
        graph.add_node(node)
        node_by_site_pci[(row.site_id_inferred, int(row.site_pci))] = node
        node_latlon[node] = (row.site_lat, row.site_lon)

    required = {"from_site_id_inferred", "to_site_id_inferred", "from_pci", "to_pci"}
    if not events_df.empty and required.issubset(events_df.columns):
        events = events_df.dropna(subset=list(required)).copy()
        for row in events.itertuples():
            try:
                from_pci, to_pci = int(float(row.from_pci)), int(float(row.to_pci))
            except (TypeError, ValueError):
                continue
            u = node_by_site_pci.get((row.from_site_id_inferred, from_pci))
            v = node_by_site_pci.get((row.to_site_id_inferred, to_pci))
            if u is None or v is None or u == v:
                continue
            if graph.has_edge(u, v):
                graph[u][v]["weight"] += 1
                graph[u][v]["observed"] = True
            else:
                graph.add_edge(u, v, weight=1, observed=True, distance_m=None)

    # Distance-based edges via a haversine BallTree radius query instead of
    # an all-pairs loop -- O(n log n) instead of O(n^2). Confirmed necessary:
    # the naive nested loop took ~17s at ~8,000 sectors (Taiwan CSV full
    # scale), which was the actual bottleneck, not the detectors.
    nodes_by_earfcn: dict[int, list] = {}
    for n in graph.nodes:
        nodes_by_earfcn.setdefault(n[2], []).append(n)
    earth_radius_m = 6371000.0
    radius_rad = distance_threshold_m / earth_radius_m
    for nodes in nodes_by_earfcn.values():
        if len(nodes) < 2:
            continue
        coords_rad = np.radians([node_latlon[n] for n in nodes])
        tree = BallTree(coords_rad, metric="haversine")
        indices, distances = tree.query_radius(coords_rad, r=radius_rad, return_distance=True)
        for i, (idx_list, dist_list) in enumerate(zip(indices, distances)):
            u = nodes[i]
            for j, dist_rad in zip(idx_list, dist_list):
                if j <= i:
                    continue
                v = nodes[j]
                if graph.has_edge(u, v):
                    continue
                graph.add_edge(u, v, weight=0, observed=False, distance_m=float(dist_rad) * earth_radius_m)

    return graph


def _make_pci_pair_cost(check_exact: bool, mod_values: list, check_grouped: bool):
    """
    Builds the cost function for exactly the rules currently checked in
    the sidebar — nothing else. 100 = exact PCI clash (only counted if
    Collision or Confusion is checked), +10 per checked Mod-N whose
    remainder also matches (only for the specific Mod values checked —
    Mod3 checked doesn't imply Mod7 is also being avoided), +1 = same
    PCI//3 (only if Grouped is checked). If a rule isn't checked, this
    function doesn't penalize matching it at all — the optimizer must not
    silently keep optimizing for a rule the user turned off.
    """
    def cost(pci_a: int, pci_b: int) -> int:
        total = 0
        if check_exact and pci_a == pci_b:
            total += 100
        for n in mod_values:
            if pci_a % n == pci_b % n:
                total += 10
        if check_grouped and pci_a // 3 == pci_b // 3:
            total += 1
        return total

    return cost


def _apply_pci_assignments(
    selected_sites_df: pd.DataFrame, events_df: pd.DataFrame, substitution: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Substitutes {(site_id, old_pci): new_pci} into REAL copies of the
    sector table and the observed-handover events table (from_pci/to_pci
    are strings there, per _detect_column_transitions' _clean_transition_
    text — matched as strings here too), so the real detectors
    (detect_pci_collisions/detect_pci_confusion/detect_pci_mod_conflicts/
    detect_pci_group_conflicts) can be rerun against a genuinely
    hypothetical post-change state, not just a cost formula re-evaluated
    on paper.
    """
    site_df2 = selected_sites_df.copy()
    events_df2 = events_df.copy()
    site_pci_numeric = pd.to_numeric(site_df2["site_pci"], errors="coerce")
    for (site_id, old_pci), new_pci in substitution.items():
        if new_pci == old_pci:
            continue
        site_mask = (site_df2["site_id_inferred"] == site_id) & (site_pci_numeric == old_pci)
        site_df2.loc[site_mask, "site_pci"] = new_pci
        if not events_df2.empty and {"from_site_id_inferred", "from_pci", "to_site_id_inferred", "to_pci"}.issubset(events_df2.columns):
            from_mask = (events_df2["from_site_id_inferred"] == site_id) & (events_df2["from_pci"] == str(old_pci))
            events_df2.loc[from_mask, "from_pci"] = str(new_pci)
            to_mask = (events_df2["to_site_id_inferred"] == site_id) & (events_df2["to_pci"] == str(old_pci))
            events_df2.loc[to_mask, "to_pci"] = str(new_pci)
    return site_df2, events_df2


def _count_remaining_conflicts(
    graph: nx.Graph,
    events_df: pd.DataFrame,
    selected_site_ids: list,
    check_collision: bool,
    check_confusion: bool,
    mod_values: list,
    check_grouped: bool,
) -> int:
    """Runs the ACTUAL detector functions (not the cost formula) against a
    hypothetical graph/events state and counts conflicts for only the
    currently-checked rules — the real verification step of the closed
    loop, independent of whatever the greedy search's own cost function
    assumed."""
    total = 0
    if check_collision:
        total += len(detect_pci_collisions(graph))
    if check_confusion:
        total += len(detect_pci_confusion(events_df, site_ids_filter=set(selected_site_ids)))
    for mod_n in mod_values:
        total += len(detect_pci_mod_conflicts(graph, mod_n))
    if check_grouped:
        total += len(detect_pci_group_conflicts(graph))
    return total


def _max_clique_size_per_earfcn(graph: nx.Graph, earfcns: set) -> dict:
    """
    Largest fully-mutually-connected group (clique) of sectors on each
    given EARFCN, restricted to graph edges (real observed handover OR
    within the distance threshold — see build_neighbor_graph). A clique
    of size K means those K sectors are ALL pairwise neighbors of each
    other, so a rule offering fewer than K distinct buckets (Mod-N's
    mod_n remainders, Grouped's 168 groups) can NEVER be fully satisfied
    there by any PCI assignment whatsoever — pigeonhole guarantees at
    least one pair keeps the same bucket. This is a hard mathematical
    property of the neighbor graph itself (independent of current PCI
    values), computed once up front rather than re-derived by watching
    the heuristic fail to converge.
    """
    sizes = {}
    for earfcn in earfcns:
        nodes_here = [n for n in graph.nodes if n[2] == earfcn]
        subgraph = graph.subgraph(nodes_here)
        best = 1
        for clique in nx.find_cliques(subgraph):
            if len(clique) > best:
                best = len(clique)
        sizes[earfcn] = best
    return sizes


def _check_rule_infeasibility(graph: nx.Graph, target_nodes: list, mod_values: list, check_grouped: bool) -> list:
    """
    Proves — not guesses — whether the checked Mod-N/Grouped rule(s) can
    ever reach zero conflicts on this neighbor graph, via the clique bound
    in _max_clique_size_per_earfcn. Collision isn't checked here: its
    "capacity" is the full 504-value PCI range, and a same-EARFCN clique
    that large essentially never occurs in practice. Returns a list of
    proof records (empty if nothing is provably infeasible) — each one
    names the rule, the EARFCN, the clique size found, and the rule's
    bucket capacity, so a "this can't converge" report is backed by a
    concrete, checkable fact instead of an assumption.
    """
    if not mod_values and not check_grouped:
        return []
    earfcns = {n[2] for n in target_nodes}
    clique_sizes = _max_clique_size_per_earfcn(graph, earfcns)
    problems = []
    for mod_n in mod_values:
        for earfcn, clique_size in clique_sizes.items():
            if clique_size > mod_n:
                problems.append({"rule": f"Mod{mod_n}", "earfcn": earfcn, "clique_size": clique_size, "capacity": mod_n})
    if check_grouped:
        for earfcn, clique_size in clique_sizes.items():
            if clique_size > 168:
                problems.append({"rule": "Grouped", "earfcn": earfcn, "clique_size": clique_size, "capacity": 168})
    return problems


def recommend_pci_reassignment(
    selected_sites_df: pd.DataFrame,
    events_df: pd.DataFrame,
    selected_site_ids: list,
    graph: nx.Graph,
    collision_conflicts: list,
    confusion_conflicts: list,
    mod_conflicts_by_n: dict,
    grouped_conflicts: list,
    check_collision: bool,
    check_confusion: bool,
    mod_values: list,
    check_grouped: bool,
    distance_threshold_m: float,
) -> tuple[pd.DataFrame, dict]:
    """
    v1 graph-based heuristic — explicitly not NSGA-II. Closed-loop: a
    single greedy pass is NOT trusted on its own. After every full pass
    over the touched sectors, the proposed PCI values are substituted
    into real copies of selected_sites_df/events_df (_apply_pci_
    assignments), the neighbor graph is rebuilt, and the SAME real
    detectors used everywhere else in this file are rerun against that
    hypothetical state (_count_remaining_conflicts) — not the cost
    formula recomputed on itself. This directly answers whether a
    recommendation is "just an assumption": a single pass is exactly
    that (see the earlier finding that a node's reported after_cost
    could go stale once later nodes were reassigned); this loop instead
    proves — or honestly reports failing to prove — that the FINAL
    suggested state has zero remaining conflicts for the checked rule(s).

    No arbitrary small iteration cap: this keeps going as long as real
    progress is being made (remaining conflict count still decreasing).
    Before looping, _check_rule_infeasibility proves — via a same-EARFCN
    clique-size bound, a hard graph-coloring fact — whether the checked
    Mod-N/Grouped rule(s) can EVER reach zero here at all. If so, the
    patience budget for "no further improvement" is small (a few extra
    passes to reach the best achievable state, since we already know 0 is
    mathematically unreachable — more looping wouldn't change that). If
    nothing is proven infeasible, the patience budget scales with problem
    size instead of a flat magic number, so a genuinely solvable case
    isn't cut off early just because it took a while.

    Only optimizes for rules actually checked in the sidebar: pass an
    empty list for any conflict type that's unchecked, and False/empty
    for the matching check_*/mod_values flag. `mod_conflicts_by_n` is
    {mod_n: conflicts} for each checked Mod value only.

    Collision/Mod-N/Grouped conflicts are defined by detect_pci_collisions/
    detect_pci_mod_conflicts/detect_pci_group_conflicts over NEIGHBOR
    PAIRS in `graph` (real observed handover OR within the distance
    threshold — see build_neighbor_graph), including same-site pairs.
    Returns (recommendation DataFrame, verification dict with keys
    iterations/converged/remaining_conflicts/verified_clean/
    remaining_by_iteration/infeasible_rules/stopped_reason).
    """
    touched_types: dict[tuple, set] = {}
    for c in collision_conflicts:
        for s in c["sites"]:
            touched_types.setdefault((s, c["pci"]), set()).add("Collision")
    for c in confusion_conflicts:
        for s in c["neighbor_sites"]:
            touched_types.setdefault((s, c["pci"]), set()).add("Confusion")
    for n, conflicts in mod_conflicts_by_n.items():
        for c in conflicts:
            for s, p in c["members"]:
                touched_types.setdefault((s, p), set()).add(f"Mod{n}")
    for c in grouped_conflicts:
        for s, p in c["members"]:
            touched_types.setdefault((s, p), set()).add("Grouped")

    if not touched_types:
        return pd.DataFrame(), {
            "iterations": 0, "converged": True, "remaining_conflicts": 0, "verified_clean": True,
            "remaining_by_iteration": [], "infeasible_rules": [], "stopped_reason": "nothing_to_do",
        }

    pair_cost = _make_pci_pair_cost(check_collision or check_confusion, mod_values, check_grouped)

    node_by_site_pci = {(n[0], n[1]): n for n in graph.nodes}
    target_nodes = [node_by_site_pci[k] for k in touched_types if k in node_by_site_pci]
    target_nodes.sort(key=lambda n: graph.degree(n), reverse=True)

    def neighbors_of(node):
        return [n for n in graph.neighbors(node) if n[2] == node[2]]

    # True baseline cost -- against the ORIGINAL, never-reassigned neighbor
    # PCIs, so "before" always means the real starting state, not whatever
    # a previous node happened to already be reassigned to.
    before_costs = {node: sum(pair_cost(node[1], n[1]) for n in neighbors_of(node)) for node in target_nodes}

    # Prove (not guess) whether the checked Mod-N/Grouped rule(s) can ever
    # reach zero on this graph at all, once, up front -- it's a structural
    # fact about the graph, unaffected by which PCIs anyone currently has.
    infeasible_rules = _check_rule_infeasibility(graph, target_nodes, mod_values, check_grouped)
    # If something is proven mathematically unreachable, a small patience
    # window is enough to settle on the best achievable state -- no amount
    # of extra looping changes a pigeonhole fact. If nothing is proven
    # infeasible, scale patience with problem size instead of a flat
    # magic cap, so a genuinely solvable case isn't cut off early.
    patience = 5 if infeasible_rules else max(30, 5 * len(target_nodes))

    assignments = {node: node[1] for node in target_nodes}
    iterations_used = 0
    converged = False
    remaining = None
    remaining_by_iteration: list[int] = []
    best_remaining = None
    no_improve_streak = 0
    stopped_reason = "converged"

    iteration = 0
    while True:
        iteration += 1
        iterations_used = iteration

        # Proposal pass: every touched node whose cost against the LATEST
        # known assignments of its neighbors is nonzero gets a fresh
        # candidate search. Nodes already at cost 0 are left untouched.
        for node in target_nodes:
            neighbors = neighbors_of(node)
            neighbor_pcis = [assignments.get(n, n[1]) for n in neighbors]
            current_pci = assignments[node]
            current_cost = sum(pair_cost(current_pci, npci) for npci in neighbor_pcis)
            if current_cost == 0:
                continue

            if check_grouped:
                search_order = _candidates_by_distance(node[1])
            else:
                search_order = [p for p in range(PCI_MIN, PCI_MAX + 1) if p != current_pci]
                random.shuffle(search_order)

            best_pci, best_cost = current_pci, current_cost
            for candidate in search_order:
                cost = sum(pair_cost(candidate, npci) for npci in neighbor_pcis)
                if cost < best_cost:
                    best_cost, best_pci = cost, candidate
                if cost == 0:
                    break
            assignments[node] = best_pci

        # Verification pass: substitute the CURRENT full proposal into
        # real data and rerun the actual detectors -- ground truth, not
        # the cost formula grading its own homework.
        substitution = {(node[0], node[1]): assignments[node] for node in target_nodes}
        hyp_site_df, hyp_events_df = _apply_pci_assignments(selected_sites_df, events_df, substitution)
        hyp_graph = build_neighbor_graph(hyp_site_df, hyp_events_df, distance_threshold_m=distance_threshold_m)
        remaining = _count_remaining_conflicts(
            hyp_graph, hyp_events_df, selected_site_ids, check_collision, check_confusion, mod_values, check_grouped
        )
        remaining_by_iteration.append(int(remaining))
        if remaining == 0:
            converged = True
            stopped_reason = "converged"
            break

        if best_remaining is None or remaining < best_remaining:
            best_remaining = remaining
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        if no_improve_streak >= patience:
            stopped_reason = "proven_infeasible" if infeasible_rules else "no_improvement_exhausted"
            break

    rows = []
    for node in target_nodes:
        site_id, current_pci, earfcn = node
        neighbors = neighbors_of(node)
        final_cost = sum(pair_cost(assignments[node], assignments.get(n, n[1])) for n in neighbors)
        rows.append(
            {
                "site": site_id,
                "current_pci": current_pci,
                "suggested_pci": assignments[node],
                "earfcn": earfcn,
                "before_cost": before_costs[node],
                "after_cost": final_cost,
                "num_same_earfcn_sectors": len(neighbors),
                "conflict_types": touched_types.get((site_id, current_pci), set()),
            }
        )

    verification = {
        "iterations": iterations_used,
        "converged": converged,
        "remaining_conflicts": int(remaining) if remaining is not None else 0,
        "verified_clean": bool(remaining == 0),
        "remaining_by_iteration": remaining_by_iteration,
        "infeasible_rules": infeasible_rules,
        "stopped_reason": stopped_reason,
    }
    return pd.DataFrame(rows), verification


# Highest severity wins when a sector is flagged by more than one rule.
# Any "Mod<n>" type (Mod1/Mod3/Mod7/Mod9, not just Mod3) -> Medium.
_SEVERITY_BY_TYPE = {"Collision": "Severe", "Confusion": "Severe", "Grouped": "Low"}
_SEVERITY_RANK = {"Severe": 3, "Medium": 2, "Low": 1}


def _severity_for_type(t: str) -> str:
    if t in _SEVERITY_BY_TYPE:
        return _SEVERITY_BY_TYPE[t]
    return "Medium" if t.startswith("Mod") else "Low"


def _severity_label(types_here: set) -> str:
    if not types_here:
        return ""
    return max((_severity_for_type(t) for t in types_here), key=lambda s: _SEVERITY_RANK[s])


def build_recommendation_table(recs: pd.DataFrame, site_labels: dict) -> pd.DataFrame:
    """Human-readable version: which rule(s) flagged this sector, how
    severe (Severe = Collision/Confusion, Medium = Mod3/PSS, Low =
    Grouped/SSS — same severity scale _pci_pair_cost uses: 100/10/1), and
    exactly what to change it to."""
    if recs.empty:
        return recs
    out = recs.copy()
    out["Site"] = out["site"].map(lambda s: site_labels.get(s, s))
    out["Issue Type"] = out["conflict_types"].map(lambda types: ", ".join(sorted(types)))
    out["Severity"] = out["conflict_types"].map(_severity_label)
    out["Changed"] = out["current_pci"] != out["suggested_pci"]
    out["Recommendation"] = out.apply(
        lambda r: (
            f"Change PCI {r['current_pci']} -> {r['suggested_pci']}"
            if r["Changed"]
            else f"Keep PCI {r['current_pci']} (already best option)"
        ),
        axis=1,
    )
    out = out.rename(
        columns={
            "current_pci": "Current PCI",
            "suggested_pci": "Suggested PCI",
            "earfcn": "EARFCN",
            "before_cost": "Conflict Cost Before",
            "after_cost": "Conflict Cost After",
            "num_same_earfcn_sectors": "Same-EARFCN Sectors Considered",
        }
    )
    cols = [
        "Site", "Issue Type", "Severity", "Recommendation", "Current PCI", "Suggested PCI", "EARFCN",
        "Changed", "Conflict Cost Before", "Conflict Cost After", "Same-EARFCN Sectors Considered",
    ]
    return out[cols].sort_values(
        by=["Severity", "Conflict Cost Before"], key=lambda s: s.map(_SEVERITY_RANK) if s.name == "Severity" else s,
        ascending=False,
    ).reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="PCI Optimization Map", layout="wide")
    st.title("PCI Optimization — Site / Sector Map")
    st.caption(
        "Test-only visualization (ML/tests/Pci_optimization). Sector wedges + PCI colors mirror the "
        "live frontend map; PCI handover pairs use the corrected, frontend-consistent transition detector."
    )

    st.sidebar.header("Data source")
    # A single top-level choice instead of manual Project ID/Region/
    # Operator entry: both options load from the SAME underlying project
    # (193, region india, operator Airtel) -- they only differ in which
    # sites they show and whether any data is synthetic.
    site_mode = st.sidebar.radio(
        "Data source",
        [
            "Demo (5 sites, synthetic conflicts)",
            "Project 193 (real Airtel data)",
            "Taiwan CSV (real site data, no handover)",
        ],
        index=0,
        help=(
            "Demo: the fixed 5-site synthetic Collision+Confusion example (2 real sites get a shared "
            "synthetic PCI injected, plus a separate synthetic Collision-only pair) -- always the same 5 "
            "sites, no inputs needed, for testing the rules/optimizer in isolation. Project 193: every real "
            "site/cell/sector that actually exists under this project (operator Airtel, region india) -- "
            "no synthetic data, no cap, whatever the DB has. Taiwan CSV: a local site/sector inventory file "
            "(Site_Data_Template_103815_updated.csv) with NO drive-test/handover data at all -- Collision/"
            "Mod-N/Grouped still work (pure geographic neighbor detection), but Confusion always reports 0 "
            "here since it needs real observed handovers, which this dataset doesn't have."
        ),
    )
    is_demo_mode = site_mode.startswith("Demo")
    is_csv_mode = site_mode.startswith("Taiwan")
    project_id, region, operator = DEFAULT_PROJECT_ID, DEFAULT_REGION, DEFAULT_OPERATOR
    if is_demo_mode:
        num_sites = 5
        st.sidebar.caption("Fixed 5-site synthetic demo — no additional inputs needed.")
    elif is_csv_mode:
        num_sites = 0
        st.sidebar.caption(
            f"Loads every site/sector in {CSV_DATASET_PATH.name} — no cap, no synthetic data, no handover "
            "data (Confusion will always show 0). ~3,000 sites / ~8,000 sectors, so Load/optimizer runs "
            "take tens of seconds, not milliseconds."
        )
    else:
        # 0 = no cap -- build_real_selection() loads every real,
        # plausible site under this project (see _plausible_sites/
        # _filter_sites_within_polygon), not an arbitrary chosen slice.
        num_sites = 0
        st.sidebar.caption("Loads every real site/cell/sector that exists under project 193 (Airtel) — no cap, no synthetic data.")

    # Polygon filter (map_regions.region, real DB query — see
    # _load_project_polygon_wkts) is always fetched and drawn; this only
    # gates the older row-level log/site filter that also runs through
    # ml_engine's Electron-bridge-first path.
    polygon_filter = False
    col_load, col_refresh = st.sidebar.columns(2)
    load_clicked = col_load.button("Load")
    refresh_clicked = col_refresh.button("Refresh from DB")

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
        label_suffix = " (PSS group)" if n == 3 else " (vendor 6-sector reuse heuristic)" if n == 6 else ""
        help_suffix = (
            " Trivial — PCI % 1 is always 0." if n == 1
            else " Not a 3GPP requirement — some vendors use it as a soft reuse-distance guideline for 6-sector sites." if n == 6
            else ""
        )
        show_mod[n] = st.sidebar.checkbox(
            f"Mod{n}{label_suffix}",
            value=False,
            help=f"2+ shown sites whose PCI % {n} matches on the same EARFCN, even with different actual PCIs.{help_suffix}",
        )
        if show_mod[n]:
            mod_remainder[n] = st.sidebar.selectbox(
                f"  Mod{n} remainder to show",
                options=list(range(n)),
                key=f"mod_remainder_{n}",
                help=f"Only PCIs where PCI % {n} equals this value.",
            )
    show_grouped = st.sidebar.checkbox(
        "Grouped PCI",
        value=False,
        help="2+ shown sites whose PCI falls in the same 3GPP PCI group (PCI // 3, 168 groups of 3) on the same EARFCN.",
    )
    show_co_centric = st.sidebar.checkbox(
        "Co-centric",
        value=False,
        help=(
            "Sectors at the SAME site pointing the SAME direction but on DIFFERENT EARFCNs "
            "(e.g. LTE1800 + LTE2100 on one sector) — informational, not a fault; shown in violet, not red."
        ),
    )
    neighbor_distance_m = st.sidebar.number_input(
        "Neighbor distance threshold (m)",
        value=int(DEFAULT_NEIGHBOR_DISTANCE_M),
        min_value=0,
        step=50,
        help=(
            "Two sectors count as neighbors (checkable for Collision/Mod-N/Grouped) if EITHER they have a "
            "real observed handover between them, OR they're within this distance of each other — same-site "
            "sectors always qualify (their distance is ~0m). Only used when real handover data doesn't "
            "already cover a pair; raise it if you want conflicts checked over a wider radius."
        ),
    )

    st.sidebar.header("Optimizer (beta)")
    run_optimizer_clicked = st.sidebar.button(
        "Run graph-based PCI optimizer",
        help=(
            "Builds the real observed neighbor graph (networkx) and proposes new PCI values "
            "for conflicted sectors via a greedy multi-constraint heuristic. Recommend-only — "
            "never changes the map or the sites shown above."
        ),
    )
    # st.button() only returns True on the exact rerun where it was
    # clicked -- any other widget interaction afterward (a rule checkbox,
    # anything) triggers a fresh rerun where it silently reverts to False
    # and the whole recommendation section vanishes, even though it was
    # run. Persist it in session_state (same pattern as pci_map_loaded)
    # so it stays visible until you reload the page.
    if "show_optimizer" not in st.session_state:
        st.session_state["show_optimizer"] = False
    if run_optimizer_clicked:
        st.session_state["show_optimizer"] = True
    if st.session_state["show_optimizer"]:
        hide_optimizer = st.sidebar.button("Hide recommendation")
        if hide_optimizer:
            st.session_state["show_optimizer"] = False

    if "pci_map_loaded" not in st.session_state:
        st.session_state["pci_map_loaded"] = False

    if load_clicked or refresh_clicked:
        st.session_state["pci_map_loaded"] = True
        st.session_state["pci_map_args"] = (int(project_id), region, operator, int(num_sites), bool(polygon_filter), site_mode)
        st.session_state["force_refresh"] = bool(refresh_clicked)

    if not st.session_state["pci_map_loaded"]:
        st.info("Set project/region/operator in the sidebar and click Load.")
        return

    proj_id, region, operator, num_sites, polygon_filter, site_mode = st.session_state["pci_map_args"]
    is_demo_mode = site_mode.startswith("Demo")
    is_csv_mode = site_mode.startswith("Taiwan")

    try:
        if is_csv_mode:
            with st.spinner(f"Reading {CSV_DATASET_PATH.name}..."):
                csv_site_df = _load_csv_site_dataset(CSV_DATASET_PATH)
            raw_data = {"site_df": csv_site_df, "events_df": pd.DataFrame(), "polygon_wkt": []}
            st.caption(
                f"Loaded {CSV_DATASET_PATH.name} directly (no DB, no artifact cache — a local CSV read is "
                f"already fast): {csv_site_df['site_id_inferred'].nunique()} sites / {len(csv_site_df)} sectors."
            )
        else:
            artifact_path = get_artifact_path(proj_id)
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

    # Real project boundary (map_regions.region for this project_id, see
    # _load_project_polygon_wkts) — restricts the site pool BEFORE
    # build_demo_selection runs, so every one of the "N sites" (real +
    # synthetic-on-real) is guaranteed to sit inside the project's actual
    # drive polygon, not just drawn alongside it as decoration. The CSV
    # data source has no project/DB linkage at all, so there's no polygon
    # to fetch for it -- skipped entirely rather than showing a confusing
    # "no polygon found for project 193" message that doesn't apply to it.
    polygons = []
    if is_csv_mode:
        st.info(f"{CSV_DATASET_PATH.name} is a standalone site inventory, not tied to a DB project — no polygon to draw.")
        site_df_pool = raw_data["site_df"]
        events_df_pool = raw_data["events_df"]
    else:
        for wkt in raw_data.get("polygon_wkt") or []:
            try:
                polygons.append(load_wkt(wkt))
            except Exception:  # noqa: BLE001
                continue

        site_df_pool = raw_data["site_df"]
        events_df_pool = raw_data["events_df"]
        if polygons:
            filtered_site_df, polygon_stats = _filter_sites_within_polygon(site_df_pool, polygons)
            if filtered_site_df.empty:
                st.warning(
                    f"Project polygon found ({len(polygons)} polygon(s)) but 0 of {polygon_stats['rows_before']} "
                    "sector rows fell inside it, even after the coordinate-swap fallback — falling back to all "
                    "project sites so the dashboard still loads. The polygon is still drawn on the map below."
                )
            else:
                site_df_pool = filtered_site_df
                pool_site_ids = set(site_df_pool["site_id_inferred"])
                events_df_pool = events_df_pool[
                    events_df_pool["from_site_id_inferred"].isin(pool_site_ids)
                    & events_df_pool["to_site_id_inferred"].isin(pool_site_ids)
                ]
                st.caption(
                    f"Project polygon loaded from map_regions ({len(polygons)} polygon(s)) — "
                    f"{polygon_stats['rows_after']}/{polygon_stats['rows_before']} sector rows fall inside it"
                    f"{' (coordinate-swap fallback used)' if polygon_stats.get('swapped') else ''}. "
                    "Site selection below is drawn only from those sectors."
                )
        else:
            st.info(f"No project polygon found in map_regions for project {proj_id} — showing all sites without a drawn boundary.")

    try:
        if is_demo_mode:
            site_df, events_df, selected_sites_df, selected_site_ids, demo_notes = build_demo_selection(
                site_df_pool, events_df_pool, 5
            )
        else:
            selected_sites_df, selected_site_ids, real_note = build_real_selection(
                site_df_pool, events_df_pool, int(num_sites)
            )
            site_df, events_df = site_df_pool, events_df_pool
            demo_notes = [real_note]
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to build site selection: {exc}")
        return
    for note in demo_notes:
        if is_demo_mode:
            st.warning(note)
        else:
            st.caption(note)

    _detection_t0 = time.perf_counter()
    confusion_conflicts = detect_pci_confusion(events_df, site_ids_filter=set(selected_site_ids))

    # Unified neighbor graph (real observed handover OR within
    # neighbor_distance_m — see build_neighbor_graph) drives Collision/
    # Mod-N/Grouped from here on, not a blanket "anyone sharing a value in
    # the selection" scan. Rebuilt after every injection below because
    # each one adds a new sector row to selected_sites_df.
    graph = build_neighbor_graph(selected_sites_df, events_df, distance_threshold_m=neighbor_distance_m)
    collision_conflicts = detect_pci_collisions(graph)

    # Mod1/3/7/9: in practice these occur naturally among 5+ real
    # neighbor sectors (coarse groupings collide far more easily than an
    # exact PCI match), so the synthetic fallback rarely fires — but if a
    # project/selection genuinely has none, inject one rather than
    # showing an empty check. Only in Demo mode — Project 193 (real) mode
    # never injects anything, by definition (see build_real_selection).
    mod_conflicts: dict[int, list] = {}
    used_pcis = set(pd.to_numeric(selected_sites_df["site_pci"], errors="coerce").dropna().astype(int).tolist())
    for mod_n in MOD_RULE_VALUES:
        found = detect_pci_mod_conflicts(graph, mod_n)
        if not found and is_demo_mode:
            try:
                selected_sites_df, mod_note = inject_synthetic_mod_conflict(
                    selected_sites_df, selected_site_ids, mod_n, used_pcis
                )
                st.warning(mod_note)
                graph = build_neighbor_graph(selected_sites_df, events_df, distance_threshold_m=neighbor_distance_m)
                found = detect_pci_mod_conflicts(graph, mod_n)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not find or synthesize a Mod{mod_n} example: {exc}")
        mod_conflicts[mod_n] = found

    # Grouped PCI (PCI // 3, the correct 3GPP formula) — same
    # naturally-occurring pattern as Mod-N, synthetic fallback rarely
    # fires, and only runs in Demo mode.
    grouped_conflicts = detect_pci_group_conflicts(graph)
    if show_grouped and not grouped_conflicts and is_demo_mode:
        try:
            selected_sites_df, grouped_note = inject_synthetic_group_conflict(selected_sites_df, selected_site_ids, used_pcis)
            st.warning(grouped_note)
            graph = build_neighbor_graph(selected_sites_df, events_df, distance_threshold_m=neighbor_distance_m)
            grouped_conflicts = detect_pci_group_conflicts(graph)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not find or synthesize a Grouped PCI example: {exc}")

    co_centric_groups = detect_co_centric_groups(selected_sites_df)
    _detection_elapsed_s = time.perf_counter() - _detection_t0

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
        + (grouped_conflicts if show_grouped else [])
        + (co_centric_groups if show_co_centric else [])
    )
    conflict_types = conflict_types_by_key(active_conflicts)
    active_labels = (
        (["Collision"] if show_collision else [])
        + (["Confusion"] if show_confusion else [])
        + [f"Mod{n}={mod_remainder[n]}" for n in MOD_RULE_VALUES if show_mod[n]]
        + (["Grouped PCI"] if show_grouped else [])
        + (["Co-centric"] if show_co_centric else [])
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

    total_available = raw_data["site_df"]["site_id_inferred"].nunique()
    if is_csv_mode:
        source_label = CSV_DATASET_PATH.name
    elif is_demo_mode:
        source_label = f"project {proj_id} (demo)"
    else:
        source_label = f"project {proj_id}"
    st.subheader(f"{len(selected_site_ids)} site(s) shown (of {total_available} available in {source_label})")
    st.caption(
        f"Graph build + Collision/Confusion/Mod/Grouped/Co-centric detection over "
        f"{graph.number_of_nodes()} sector(s) / {graph.number_of_edges()} neighbor edge(s): {_detection_elapsed_s:.2f}s."
    )

    with st.expander("PCI reuse by neighbor order (1-5 hops)"):
        st.caption(
            "Same PCI+EARFCN reuse bucketed by exact hop-distance in the neighbor graph. Order 1 = direct "
            "neighbor reuse (same pairs Collision flags). Order 2 is structurally what Confusion is (two "
            "cells both 1 hop from a common serving cell). Orders 3-5 have no standard 3GPP/RF interference "
            "meaning -- a UE realistically never hears cells that many hops apart at once; these are a PCI "
            "reuse-distance audit metric, not a fault to fix. \"Also Collision\"/\"Also Confusion\" are the "
            "REAL detectors cross-referenced against each order's pairs, not assumed from the order number."
        )
        order_conflicts = detect_pci_order_conflicts(graph, max_order=5)
        order_cols = st.columns(5)
        for order, col in zip(range(1, 6), order_cols):
            col.metric(f"Order {order}", len(order_conflicts[order]))
        st.dataframe(
            build_order_conflict_table(order_conflicts, confusion_conflicts),
            use_container_width=True,
            hide_index=True,
        )

    fmap = render_map(selected_sites_df, site_labels=site_labels, conflict_types=conflict_types, filter_to_conflicts=True, polygons=polygons)
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

    if st.session_state["show_optimizer"]:
        st.markdown("---")
        st.markdown("### Graph-based PCI recommendation (beta)")
        mod_values_checked = [n for n in MOD_RULE_VALUES if show_mod[n]]
        active_rule_labels = (
            (["Collision"] if show_collision else [])
            + (["Confusion"] if show_confusion else [])
            + [f"Mod{n}" for n in mod_values_checked]
            + (["Grouped"] if show_grouped else [])
        )
        st.caption(
            "Only optimizes for the rule(s) currently checked above: "
            + (", ".join(active_rule_labels) if active_rule_labels else "none — check a rule first")
            + f". Each candidate PCI is scored against every NEIGHBOR sector (real observed handover OR "
            f"within {int(neighbor_distance_m)}m — same-site sectors included, since they're always ~0m "
            "apart) on the same EARFCN — the same neighbor graph the rules above use. Closed-loop: after "
            "every proposal pass, the suggested PCIs are substituted into real copies of the site/event "
            "data and the ACTUAL Collision/Confusion/Mod/Grouped detectors are rerun against that "
            "hypothetical state to verify it, not just re-scored on paper — repeated until 0 conflicts "
            "remain or a max-iteration cap is hit. A first-pass greedy heuristic, not the eventual "
            "NSGA-II solver. Recommend-only: nothing above (map/tables) changes."
        )
        if not active_rule_labels:
            st.warning("No PCI rule is checked above — check at least one (Collision, Confusion, a Mod, or Grouped) to run the optimizer.")
        else:
            mod_conflicts_checked = {n: mod_conflicts.get(n, []) for n in mod_values_checked}
            t_start = time.perf_counter()
            recs, verification = recommend_pci_reassignment(
                selected_sites_df,
                events_df,
                selected_site_ids,
                graph,
                collision_conflicts if show_collision else [],
                confusion_conflicts if show_confusion else [],
                mod_conflicts_checked,
                grouped_conflicts if show_grouped else [],
                check_collision=show_collision,
                check_confusion=show_confusion,
                mod_values=mod_values_checked,
                check_grouped=show_grouped,
                distance_threshold_m=neighbor_distance_m,
            )
            elapsed_s = time.perf_counter() - t_start
            if recs.empty:
                st.info(f"No conflicts found for the currently checked rule(s) among the selected sites — nothing to recommend. ({elapsed_s:.2f}s)")
            else:
                st.dataframe(build_recommendation_table(recs, site_labels), use_container_width=True)
                total_before = int(recs["before_cost"].sum())
                total_after = int(recs["after_cost"].sum())
                st.caption(f"Total conflict cost across recommended sectors: {total_before} -> {total_after}.")
                st.caption(
                    f"Ran in {elapsed_s:.2f}s over {verification['iterations']} iteration(s) for "
                    f"{len(recs)} touched sector(s). Remaining conflicts per iteration: "
                    f"{verification['remaining_by_iteration']}."
                )
                if verification["verified_clean"]:
                    st.success(
                        f"Closed-loop verified: after {verification['iterations']} iteration(s), rerunning the "
                        "real Collision/Confusion/Mod/Grouped detectors on the substituted PCI values found 0 "
                        "remaining conflicts among the checked rule(s) for these sectors."
                    )
                elif verification["infeasible_rules"]:
                    proofs = "; ".join(
                        f"{p['rule']} on EARFCN {p['earfcn']}: {p['clique_size']} sectors are all mutual "
                        f"neighbors of each other, but {p['rule']} only offers {p['capacity']} distinct bucket(s)"
                        for p in verification["infeasible_rules"]
                    )
                    st.error(
                        f"Mathematically unsolvable, not a heuristic failure: {proofs}. By the pigeonhole "
                        "principle, no PCI assignment can avoid at least one same-bucket pair there — ran "
                        f"{verification['iterations']} iteration(s) to reach the best achievable state "
                        f"({verification['remaining_conflicts']} conflict(s) remaining), then stopped because "
                        "further looping cannot change a proven mathematical fact. The table above shows that "
                        "best-achievable state, not a guaranteed-clean fix. Reduce the neighbor distance "
                        "threshold, narrow which sectors are selected, or uncheck this rule to resolve it."
                    )
                else:
                    st.error(
                        f"NOT fully resolved after {verification['iterations']} iteration(s) — gave up after "
                        f"{verification['remaining_conflicts']} conflict(s) stopped improving with no proof "
                        "it's mathematically impossible; this heuristic simply couldn't find a better state. "
                        "The table above shows the best state found, but this is NOT a guaranteed-clean fix."
                    )


if __name__ == "__main__":
    main()
