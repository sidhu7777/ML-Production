from __future__ import annotations

import math
import random
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sqlalchemy import text

from tests.Pci_optimization.pci_optimization_dataset_test import (
    _build_pair_summary,
    _detect_pci_transitions,
    _enrich_transition_sites,
    _fetch_network_logs,
    _fetch_project_session_ids,
    _resolve_engine,
)
from tools.lte_prediction import ml_engine
from utils.python_bridge import _filter_complete_site_prediction_identity


MOD_RULE_VALUES = (1, 3, 6, 7, 8, 9)
DEFAULT_NEIGHBOR_DISTANCE_M = 500.0
PCI_MIN, PCI_MAX = 0, 503


# ============================================================
# Fetch -- real data only. No synthetic injection anywhere in
# this module; that's a test-dashboard-only concept
# (ML/tests/Pci_optimization/pci_map_dashboard.py).
# ============================================================


def fetch_site_prediction(project_id: int, operator: str, current_engine) -> pd.DataFrame:
    """
    Real site/sector source -- the same table + identity-dedup key the
    live frontend map uses (site_prediction, deduped by
    site|cell_id|sector|band|operator). Verified against project 193:
    matches the live map's own "Cell Sectors BY OPERATOR" count exactly.
    Direct DB query, not the Electron bridge -- the bridge's server-side
    spatial functions choke on malformed lat/lon in some regions (e.g.
    Taiwan longitude stored as an integer with the decimal point
    stripped), so this reads the raw value in Python and normalizes it
    here instead of letting a DB-side spatial call fail on it.
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

    complete_df = _filter_complete_site_prediction_identity(raw_df, endpoint="pci_optimization_engine")
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

    # Some regions (confirmed: Taiwan) store lat/lon as an integer with
    # the decimal point stripped (e.g. 25007083 -> needs /1e6 ->
    # 25.007083). A real Taiwan/India lat is never > 90 or < -90, so any
    # out-of-range magnitude is this exact bug, not a real coordinate.
    for col in ("site_lat",):
        bad = deduped[col].abs() > 90
        deduped.loc[bad, col] = deduped.loc[bad, col] / 1_000_000.0
    for col in ("site_lon",):
        bad = deduped[col].abs() > 180
        deduped.loc[bad, col] = deduped.loc[bad, col] / 1_000_000.0

    deduped["samples"] = 1
    deduped["site_azimuth_soft_deg"] = np.nan
    return deduped.reset_index(drop=True)


def load_project_polygons(project_id: int, current_engine) -> list[str]:
    """Real project boundary from map_regions.region -- direct DB, bypasses
    ml_engine._load_project_polygons()'s Electron-bridge-first attempt
    (which does not fail over gracefully when the bridge isn't reachable)."""
    polygons = ml_engine._load_project_polygons_from_db(int(project_id), current_engine)
    return [p.wkt for p in polygons]


def filter_sites_within_polygon(site_df: pd.DataFrame, polygons: list) -> tuple[pd.DataFrame, dict]:
    """Restricts site_df to sectors whose (site_lat, site_lon) falls inside
    the project's real polygon(s), with the coordinate-swap fallback
    map_regions.region WKT sometimes needs (points stored as (lat, lon),
    non-standard vs. the (lon, lat) order the underlying contains-check
    assumes)."""
    if not polygons or site_df.empty:
        return site_df, {"applied": False, "polygons_found": len(polygons), "rows_before": len(site_df), "rows_after": len(site_df)}

    from shapely.wkt import loads as load_wkt

    shapely_polygons = [load_wkt(p) if isinstance(p, str) else p for p in polygons]
    work = site_df.rename(columns={"site_lat": "lat", "site_lon": "lon"}).copy()
    filtered = ml_engine._filter_df_by_polygons(work, shapely_polygons)
    swapped = False
    if filtered.empty:
        swapped_polygons = ml_engine._swap_polygon_coords(shapely_polygons)
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


def fetch_network_events(project_id: int, region: str, operator: str, current_engine, primary_only: bool = True) -> pd.DataFrame:
    """Real observed handover transitions from tbl_network_log, via the
    verified-correct, frontend-consistent detector chain (_detect_pci_
    transitions -> _enrich_transition_sites), reused as-is from the
    already-validated ETL pipeline -- not reimplemented here."""
    with current_engine.connect() as conn:
        session_ids, _project_meta = _fetch_project_session_ids(project_id, conn)
        log_df = _fetch_network_logs(session_ids, operator, conn, primary_only=primary_only)
    if log_df.empty:
        return pd.DataFrame()
    events_df = _detect_pci_transitions(log_df)
    return events_df


def enrich_events(events_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df
    return _enrich_transition_sites(events_df, site_df)


# ============================================================
# Site selection -- real sites only, no synthetic padding.
# ============================================================


def plausible_sites(site_df: pd.DataFrame) -> pd.DataFrame:
    """Drops the site_id_inferred="0" sentinel (a bucket of failed per-row
    site inference that can span unrelated physical locations) and any
    site group whose lat/lon spread is too large to be one physical site."""
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


def select_sites(site_df: pd.DataFrame, max_sites: int, site_ids: list[str] | None = None) -> tuple[pd.DataFrame, list, str]:
    """Real-project selection, no synthetic injection. `site_ids`, if
    given, restricts to exactly those real sites; otherwise ranks by total
    resolved sectors (richest first) and takes the top `max_sites`
    (<= 0 means every plausible real site, no cap)."""
    sites_with_id = plausible_sites(site_df)
    if site_ids:
        wanted = set(str(s) for s in site_ids)
        selected_site_ids = [s for s in sites_with_id["site_id_inferred"].unique() if str(s) in wanted]
        note = f"{len(selected_site_ids)} of {len(wanted)} requested site_ids resolved to real, plausible sites."
    else:
        site_rank = sites_with_id.groupby("site_id_inferred")["samples"].sum().sort_values(ascending=False)
        selected_site_ids = list(site_rank.index) if max_sites <= 0 else list(site_rank.head(max_sites).index)
        note = f"{len(selected_site_ids)} real site(s) selected (no synthetic data), ranked by total resolved sectors."
    selected_sites_df = sites_with_id[sites_with_id["site_id_inferred"].isin(selected_site_ids)].copy()
    return selected_sites_df, selected_site_ids, note


# ============================================================
# Neighbor graph -- real observed handover OR within a
# configurable distance threshold, unified (no same-site
# special case needed: same-site sectors are always ~0m apart).
# ============================================================


def _haversine_scalar_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_neighbor_graph(
    selected_sites_df: pd.DataFrame, events_df: pd.DataFrame, distance_threshold_m: float = DEFAULT_NEIGHBOR_DISTANCE_M
) -> nx.Graph:
    df = selected_sites_df.dropna(subset=["site_pci", "site_earfcn", "site_id_inferred", "site_lat", "site_lon"]).copy()
    df["site_pci"] = pd.to_numeric(df["site_pci"], errors="coerce")
    df["site_earfcn"] = pd.to_numeric(df["site_earfcn"], errors="coerce")
    df["site_lat"] = pd.to_numeric(df["site_lat"], errors="coerce")
    df["site_lon"] = pd.to_numeric(df["site_lon"], errors="coerce")
    df = df.dropna(subset=["site_pci", "site_earfcn", "site_lat", "site_lon"])

    graph = nx.Graph()
    # Keyed by (site, pci, earfcn) -- NOT (site, pci) alone. A dual-band
    # site can legitimately reuse the same PCI number on two different
    # EARFCNs; a 2-field key would let the second overwrite the first,
    # silently dropping one real sector from every lookup below.
    node_by_site_pci_earfcn: dict[tuple, tuple] = {}
    node_latlon: dict[tuple, tuple] = {}
    for row in df.itertuples():
        node = (row.site_id_inferred, int(row.site_pci), int(row.site_earfcn))
        graph.add_node(node)
        node_by_site_pci_earfcn[node] = node
        node_latlon[node] = (row.site_lat, row.site_lon)

    required = {"from_site_id_inferred", "to_site_id_inferred", "from_pci", "to_pci", "from_site_earfcn", "to_site_earfcn"}
    if not events_df.empty and required.issubset(events_df.columns):
        events = events_df.dropna(subset=list(required)).copy()
        for row in events.itertuples():
            try:
                from_pci, to_pci = int(float(row.from_pci)), int(float(row.to_pci))
                from_earfcn, to_earfcn = int(float(row.from_site_earfcn)), int(float(row.to_site_earfcn))
            except (TypeError, ValueError):
                continue
            u = node_by_site_pci_earfcn.get((row.from_site_id_inferred, from_pci, from_earfcn))
            v = node_by_site_pci_earfcn.get((row.to_site_id_inferred, to_pci, to_earfcn))
            if u is None or v is None or u == v:
                continue
            if graph.has_edge(u, v):
                graph[u][v]["weight"] += 1
                graph[u][v]["observed"] = True
            else:
                graph.add_edge(u, v, weight=1, observed=True, distance_m=None)

    # Distance-based edges via a haversine BallTree radius query instead of
    # an all-pairs loop -- O(n log n) instead of O(n^2). Confirmed necessary:
    # the naive nested loop took ~17s at ~8,000 sectors (Taiwan project 210
    # full scale) and was the actual bottleneck behind multi-minute runs,
    # not the detectors or the optimizer (both operate on the much smaller
    # touched-node set once the graph exists).
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


# ============================================================
# Detectors -- Collision/Confusion/Mod-N/Grouped/Co-centric.
# ============================================================


def detect_pci_collisions(graph: nx.Graph) -> list[dict]:
    # Grouped by CONNECTED COMPONENT of collision edges, not a flat union
    # of every site seen under this (pci, earfcn) -- a flat union merges
    # two unrelated collisions that happen to reuse the same PCI number
    # into one fake combined group even when the sites have no path
    # between them in the graph at all.
    edges_by_key: dict[tuple, list] = {}
    for u, v in graph.edges():
        if u[2] != v[2] or u[1] != v[1]:
            continue
        key = (u[1], u[2])
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
    reuse (the same pairs detect_pci_collisions flags, just one row per
    PAIR here instead of one row per PCI/EARFCN group). Order 2 is what
    Confusion structurally is (two cells both 1 hop from a common serving
    cell -> 2 hops from each other) -- Confusion itself is still the real,
    handover-verified detector; this is a purely graph-structural version.
    Orders 3-5 have no standard 3GPP/RF interference meaning -- a UE
    realistically never hears two cells that many hops apart at once, so
    these are a PCI reuse-DISTANCE audit metric, not a fault to fix.

    Only searches from nodes that actually share a PCI+EARFCN with at
    least one other node (not BFS from every node in the graph), since
    same-PCI groups are typically a small fraction of all sectors.
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


def detect_pci_confusion(events_df: pd.DataFrame, site_ids_filter: set | None = None) -> list[dict]:
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
    pairs = events.drop_duplicates(["from_site_id_inferred", "to_site_id_inferred", "to_pci", "to_site_earfcn"])
    conflicts: list[dict] = []
    for (from_site, pci_val, earfcn_val), grp in pairs.groupby(["from_site_id_inferred", "to_pci", "to_site_earfcn"]):
        neighbor_sites = sorted(grp["to_site_id_inferred"].unique().tolist())
        if len(neighbor_sites) >= 2:
            conflicts.append(
                {"type": "Confusion", "pci": int(pci_val), "earfcn": earfcn_val, "serving_site": from_site, "neighbor_sites": neighbor_sites}
            )
    conflicts.sort(key=lambda c: len(c["neighbor_sites"]), reverse=True)
    return conflicts


def detect_pci_mod_conflicts(graph: nx.Graph, mod_n: int) -> list[dict]:
    # Same connected-component fix as detect_pci_collisions.
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
            conflicts.append({"type": f"Mod{mod_n}", "mod_n": mod_n, "mod_value": mod_val, "earfcn": earfcn_val, "members": members})
    conflicts.sort(key=lambda c: len(c["members"]), reverse=True)
    return conflicts


def detect_pci_group_conflicts(graph: nx.Graph) -> list[dict]:
    # Same connected-component fix as detect_pci_collisions.
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


def detect_co_centric_groups(selected_sites_df: pd.DataFrame, azimuth_tolerance_deg: float = 15.0) -> list[dict]:
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
        groups.append({"type": "Co-centric", "site": site_id, "azimuth_deg": float(grp["_az"].mean()), "members": members})
    groups.sort(key=lambda g: len(g["members"]), reverse=True)
    return groups


# ============================================================
# Closed-loop optimizer -- proposes, substitutes into real data,
# reruns the real detectors, and only stops on a real proof of
# convergence or of mathematical infeasibility (pigeonhole).
# ============================================================


def _candidates_by_distance(current_pci: int) -> list[int]:
    candidates = []
    for delta in range(1, PCI_MAX - PCI_MIN + 1):
        up, down = current_pci + delta, current_pci - delta
        if up <= PCI_MAX:
            candidates.append(up)
        if down >= PCI_MIN:
            candidates.append(down)
    return candidates


def _make_pci_pair_cost(check_exact: bool, mod_values: list, check_grouped: bool):
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


def _cost_for_single_rule(rule_label: str):
    """Isolates ONE rule's own contribution to the pair-cost. A sector
    touched by two rules at once (e.g. Collision AND Mod3 on the same
    physical sector) needs its own per-rule pass/fail, not just the
    combined total -- a sector can genuinely clear Collision (504 values
    of room) while still being stuck on Mod3 (3 buckets), and reporting
    only the combined cost would wrongly call the Collision side
    unresolved too."""
    if rule_label in ("Collision", "Confusion"):
        return _make_pci_pair_cost(check_exact=True, mod_values=[], check_grouped=False)
    if rule_label == "Grouped":
        return _make_pci_pair_cost(check_exact=False, mod_values=[], check_grouped=True)
    if rule_label.startswith("Mod"):
        return _make_pci_pair_cost(check_exact=False, mod_values=[int(rule_label[3:])], check_grouped=False)
    raise ValueError(f"Unknown rule label: {rule_label}")


def _apply_pci_assignments(selected_sites_df: pd.DataFrame, events_df: pd.DataFrame, substitution: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    graph: nx.Graph, events_df: pd.DataFrame, selected_site_ids: list,
    check_collision: bool, check_confusion: bool, mod_values: list, check_grouped: bool,
) -> int:
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
    """Proves, via a same-EARFCN clique-size bound (pigeonhole), whether
    the checked Mod-N/Grouped rule(s) can ever reach zero conflicts on
    this graph at all -- a hard mathematical fact, not a heuristic guess."""
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
    """Closed-loop v1 heuristic (not NSGA-II). No arbitrary iteration cap:
    keeps going while the real remaining-conflict count (rechecked via the
    real detectors after every pass, not a cost formula grading its own
    homework) keeps improving. Stops only on a verified 0, or on a
    same-EARFCN clique-size proof that 0 is mathematically unreachable."""
    # Keyed by (site, pci, earfcn) -- NOT (site, pci) alone. A site can
    # legitimately reuse the same PCI on two different EARFCNs (dual-band);
    # a 2-field key collapses those into one, silently dropping whichever
    # one loses the dict-overwrite race from ever being optimized at all
    # (confirmed on project 196: 22 of 50 touched (site, pci) pairs were
    # actually two distinct sectors on different EARFCNs sharing a number).
    touched_types: dict[tuple, set] = {}
    for c in collision_conflicts:
        for s in c["sites"]:
            touched_types.setdefault((s, c["pci"], c["earfcn"]), set()).add("Collision")
    for c in confusion_conflicts:
        for s in c["neighbor_sites"]:
            touched_types.setdefault((s, c["pci"], c["earfcn"]), set()).add("Confusion")
    for n, conflicts in mod_conflicts_by_n.items():
        for c in conflicts:
            for s, p in c["members"]:
                touched_types.setdefault((s, p, c["earfcn"]), set()).add(f"Mod{n}")
    for c in grouped_conflicts:
        for s, p in c["members"]:
            touched_types.setdefault((s, p, c["earfcn"]), set()).add("Grouped")

    if not touched_types:
        return pd.DataFrame(), {
            "iterations": 0, "converged": True, "remaining_conflicts": 0, "verified_clean": True,
            "remaining_by_iteration": [], "infeasible_rules": [], "stopped_reason": "nothing_to_do",
        }

    pair_cost = _make_pci_pair_cost(check_collision or check_confusion, mod_values, check_grouped)

    # touched_types keys are already (site, pci, earfcn) -- exactly the
    # graph's own node shape -- so a node is touched iff it's a key here,
    # no separate lookup dict needed (and no risk of it re-colliding).
    target_nodes = [node for node in touched_types if node in graph.nodes]
    target_nodes.sort(key=lambda n: graph.degree(n), reverse=True)

    def neighbors_of(node):
        return [n for n in graph.neighbors(node) if n[2] == node[2]]

    before_costs = {node: sum(pair_cost(node[1], n[1]) for n in neighbors_of(node)) for node in target_nodes}

    infeasible_rules = _check_rule_infeasibility(graph, target_nodes, mod_values, check_grouped)
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
        neighbor_final_pcis = [assignments.get(n, n[1]) for n in neighbors]
        final_cost = sum(pair_cost(assignments[node], npci) for npci in neighbor_final_pcis)
        conflict_types = touched_types.get((site_id, current_pci, earfcn), set())
        # Per-rule cost, not just the combined total -- a sector touched by
        # two rules at once can genuinely clear one while the other stays
        # stuck (e.g. Collision resolved, Mod3 still capacity-limited on
        # the same sector); reporting only `after_cost` would call both
        # unresolved just because the total isn't exactly zero.
        rule_costs = {
            rule_label: sum(_cost_for_single_rule(rule_label)(assignments[node], npci) for npci in neighbor_final_pcis)
            for rule_label in conflict_types
        }
        rows.append(
            {
                "site": site_id,
                "current_pci": current_pci,
                "suggested_pci": assignments[node],
                "earfcn": earfcn,
                "before_cost": before_costs[node],
                "after_cost": final_cost,
                "num_same_earfcn_sectors": len(neighbors),
                "conflict_types": conflict_types,
                "rule_costs": rule_costs,
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


_SEVERITY_BY_TYPE = {"Collision": "Severe", "Confusion": "Severe", "Grouped": "Low"}
_SEVERITY_RANK = {"Severe": 3, "Medium": 2, "Low": 1}


def _severity_for_type(t: str) -> str:
    if t in _SEVERITY_BY_TYPE:
        return _SEVERITY_BY_TYPE[t]
    return "Medium" if t.startswith("Mod") else "Low"


def severity_label(types_here: set) -> str:
    if not types_here:
        return ""
    return max((_severity_for_type(t) for t in types_here), key=lambda s: _SEVERITY_RANK[s])


# ============================================================
# Top-level orchestrator -- matches the frontend payload shape.
# ============================================================


def run_pci_optimization(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    cfg keys (matching the frontend /api/pci-optimization/run payload):
    project_id, region, operator, primary_only, filter_sites_to_polygon,
    filter_logs_to_polygon (accepted, currently unused -- log-row-level
    filtering isn't part of this pipeline), neighbor_distance_m, rules
    ({collision, confusion, mod: [...], grouped, co_centric}),
    run_optimizer, site_ids, max_sites.
    """
    project_id = int(cfg["project_id"])
    region = str(cfg.get("region") or "india").lower()
    operator = cfg.get("operator") or "all"
    primary_only = bool(cfg.get("primary_only", True))
    filter_sites_to_polygon = bool(cfg.get("filter_sites_to_polygon", True))
    neighbor_distance_m = float(cfg.get("neighbor_distance_m", DEFAULT_NEIGHBOR_DISTANCE_M))
    rules = cfg.get("rules") or {}
    check_collision = bool(rules.get("collision", True))
    check_confusion = bool(rules.get("confusion", True))
    mod_values = [int(n) for n in (rules.get("mod") or [])]
    check_grouped = bool(rules.get("grouped", False))
    check_co_centric = bool(rules.get("co_centric", False))
    run_optimizer = bool(cfg.get("run_optimizer", True))
    site_ids = cfg.get("site_ids") or None
    max_sites = int(cfg.get("max_sites", 0))

    current_engine = _resolve_engine(region)

    site_df = fetch_site_prediction(project_id, operator, current_engine)
    if site_df.empty:
        raise ValueError(f"No site_prediction rows for project_id={project_id}, operator={operator}, region={region}.")

    polygon_wkts = load_project_polygons(project_id, current_engine)
    polygon_stats = {"applied": False, "polygons_found": len(polygon_wkts)}
    if filter_sites_to_polygon and polygon_wkts:
        filtered_site_df, polygon_stats = filter_sites_within_polygon(site_df, polygon_wkts)
        if not filtered_site_df.empty:
            site_df = filtered_site_df

    events_df = fetch_network_events(project_id, region, operator, current_engine, primary_only=primary_only)
    events_df = enrich_events(events_df, site_df)

    if filter_sites_to_polygon and polygon_wkts and polygon_stats.get("applied"):
        pool_site_ids = set(site_df["site_id_inferred"])
        if not events_df.empty and {"from_site_id_inferred", "to_site_id_inferred"}.issubset(events_df.columns):
            events_df = events_df[
                events_df["from_site_id_inferred"].isin(pool_site_ids) & events_df["to_site_id_inferred"].isin(pool_site_ids)
            ]

    selected_sites_df, selected_site_ids, selection_note = select_sites(site_df, max_sites, site_ids)

    graph = build_neighbor_graph(selected_sites_df, events_df, distance_threshold_m=neighbor_distance_m)

    collision_conflicts = detect_pci_collisions(graph) if check_collision else []
    confusion_conflicts = detect_pci_confusion(events_df, site_ids_filter=set(selected_site_ids)) if check_confusion else []
    mod_conflicts_by_n = {n: detect_pci_mod_conflicts(graph, n) for n in mod_values}
    grouped_conflicts = detect_pci_group_conflicts(graph) if check_grouped else []
    co_centric_groups = detect_co_centric_groups(selected_sites_df) if check_co_centric else []
    order_conflicts_by_order = detect_pci_order_conflicts(graph, max_order=5)

    recs = pd.DataFrame()
    verification: dict[str, Any] = {}
    if run_optimizer:
        recs, verification = recommend_pci_reassignment(
            selected_sites_df, events_df, selected_site_ids, graph,
            collision_conflicts, confusion_conflicts, mod_conflicts_by_n, grouped_conflicts,
            check_collision=check_collision, check_confusion=check_confusion,
            mod_values=mod_values, check_grouped=check_grouped,
            distance_threshold_m=neighbor_distance_m,
        )

    return {
        "project_id": project_id,
        "region": region,
        "operator": operator,
        "neighbor_distance_m": neighbor_distance_m,
        "polygon_wkt": polygon_wkts,
        "polygon_stats": polygon_stats,
        "selection_note": selection_note,
        "site_count": len(selected_site_ids),
        "sector_count": len(selected_sites_df),
        "graph_edge_count": graph.number_of_edges(),
        "collision_conflicts": collision_conflicts,
        "confusion_conflicts": confusion_conflicts,
        "mod_conflicts_by_n": mod_conflicts_by_n,
        "grouped_conflicts": grouped_conflicts,
        "co_centric_groups": co_centric_groups,
        "order_conflicts_by_order": order_conflicts_by_order,
        "order_conflict_counts": {order: len(pairs) for order, pairs in order_conflicts_by_order.items()},
        "recommendations": recs,
        "verification": verification,
        "selected_sites_df": selected_sites_df,
    }
