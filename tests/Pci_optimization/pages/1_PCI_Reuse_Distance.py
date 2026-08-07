"""
PCI Optimization — Page 2: reuse-distance audit (test-only, Streamlit).

Two views, both independent of the neighbor-distance threshold used
elsewhere in this dashboard (that threshold only decides what Collision/
Confusion currently flag; this page shows the FULL same-PCI reuse picture
out to 15 km, regardless of whether it's currently flagged as a conflict):

1. A PCI x distance-bucket (1-15 km) table, filterable to All / Collision /
   Confusion pairs.
2. A before/after bar chart of Collision+Confusion pairs by distance,
   comparing real current PCI values against the closed-loop optimizer's
   suggested values.

Self-contained data-source sidebar (same three choices as the main page:
Demo / Project 193 / Taiwan CSV) so this page works standalone without
requiring the main page to have been loaded first in this session.

Run: <venv>/python -m streamlit run tests/Pci_optimization/pci_map_dashboard.py
(from the ML/ directory) — this page shows up automatically in the sidebar
nav once pci_map_dashboard.py is the entry point, since Streamlit picks up
any pages/ directory next to it.
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from shapely.wkt import loads as load_wkt

from tests.Pci_optimization.pci_map_dashboard import (
    CSV_DATASET_PATH,
    DEFAULT_NEIGHBOR_DISTANCE_M,
    DEFAULT_OPERATOR,
    DEFAULT_PROJECT_ID,
    DEFAULT_REGION,
    _apply_pci_assignments,
    _filter_sites_within_polygon,
    _haversine_scalar_m,
    _load_csv_site_dataset,
    build_demo_selection,
    build_neighbor_graph,
    build_real_selection,
    detect_pci_collisions,
    detect_pci_confusion,
    fetch_pci_map_data_from_db,
    get_artifact_path,
    load_artifact,
    recommend_pci_reassignment,
    save_artifact,
)

st.set_page_config(page_title="PCI Reuse Distance", layout="wide")
st.title("PCI Reuse Distance Audit (1-15 km)")
st.caption(
    "Every real pair of sites sharing the same PCI+EARFCN, bucketed by their REAL distance apart — "
    "not limited to the neighbor-distance threshold Collision/Mod-N/Grouped use on the main page. "
    "Shows the full reuse picture out to 15 km, so you can see reuse that isn't currently flagged as "
    "a conflict, alongside the pairs that are."
)

# ============================================================
# Data source — same three choices as the main page, own sidebar
# instance so this page works standalone.
# ============================================================
st.sidebar.header("Data source")
site_mode = st.sidebar.radio(
    "Data source",
    [
        "Demo (5 sites, synthetic conflicts)",
        "Project 193 (real Airtel data)",
        "Taiwan CSV (real site data, no handover)",
    ],
    index=0,
    help="Same three data sources as the main PCI map page — see there for details on each.",
)
is_demo_mode = site_mode.startswith("Demo")
is_csv_mode = site_mode.startswith("Taiwan")

neighbor_distance_m = st.sidebar.number_input(
    "Neighbor distance threshold (m)",
    value=int(DEFAULT_NEIGHBOR_DISTANCE_M),
    min_value=0,
    step=50,
    help="Only used to decide which reuse pairs currently count as Collision (used by the 'Collision' "
    "filter and the before/after chart) — the 1-15 km reuse table itself isn't limited by this.",
)

load_clicked = st.sidebar.button("Load")

if "reuse_page_loaded" not in st.session_state:
    st.session_state["reuse_page_loaded"] = False
if load_clicked:
    st.session_state["reuse_page_loaded"] = True
    st.session_state["reuse_page_args"] = (site_mode, float(neighbor_distance_m))

if not st.session_state["reuse_page_loaded"]:
    st.info("Pick a data source in the sidebar and click Load.")
    st.stop()

site_mode, neighbor_distance_m = st.session_state["reuse_page_args"]
is_demo_mode = site_mode.startswith("Demo")
is_csv_mode = site_mode.startswith("Taiwan")

try:
    if is_csv_mode:
        with st.spinner(f"Reading {CSV_DATASET_PATH.name}..."):
            csv_site_df = _load_csv_site_dataset(CSV_DATASET_PATH)
        raw_data = {"site_df": csv_site_df, "events_df": pd.DataFrame(), "polygon_wkt": []}
    else:
        artifact_path = get_artifact_path(DEFAULT_PROJECT_ID)
        if artifact_path.exists():
            raw_data = load_artifact(artifact_path)
        else:
            with st.spinner(f"Fetching project {DEFAULT_PROJECT_ID} from the DB..."):
                raw_data = fetch_pci_map_data_from_db(DEFAULT_PROJECT_ID, DEFAULT_REGION, DEFAULT_OPERATOR, False)
                save_artifact(raw_data, artifact_path)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load data: {exc}")
    st.stop()

polygons = []
site_df_pool = raw_data["site_df"]
events_df_pool = raw_data["events_df"]
if not is_csv_mode:
    for wkt in raw_data.get("polygon_wkt") or []:
        try:
            polygons.append(load_wkt(wkt))
        except Exception:  # noqa: BLE001
            continue
    if polygons:
        filtered_site_df, polygon_stats = _filter_sites_within_polygon(site_df_pool, polygons)
        if not filtered_site_df.empty:
            site_df_pool = filtered_site_df
            pool_site_ids = set(site_df_pool["site_id_inferred"])
            events_df_pool = events_df_pool[
                events_df_pool["from_site_id_inferred"].isin(pool_site_ids)
                & events_df_pool["to_site_id_inferred"].isin(pool_site_ids)
            ]

try:
    if is_demo_mode:
        site_df, events_df, selected_sites_df, selected_site_ids, notes = build_demo_selection(
            site_df_pool, events_df_pool, 5
        )
    else:
        selected_sites_df, selected_site_ids, note = build_real_selection(site_df_pool, events_df_pool, 0)
        site_df, events_df = site_df_pool, events_df_pool
        notes = [note]
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to build site selection: {exc}")
    st.stop()
for note in notes:
    st.caption(note)

st.subheader(f"{len(selected_site_ids)} site(s) / {len(selected_sites_df)} sector(s) loaded")

graph = build_neighbor_graph(selected_sites_df, events_df, distance_threshold_m=neighbor_distance_m)
collision_conflicts = detect_pci_collisions(graph)
confusion_conflicts = detect_pci_confusion(events_df, site_ids_filter=set(selected_site_ids))


# ============================================================
# Shared helpers
# ============================================================


def build_same_pci_pairs(sites_df: pd.DataFrame) -> pd.DataFrame:
    """Every real pair of DISTINCT sites sharing the same PCI+EARFCN, with
    their real haversine distance — independent of graph adjacency. Group
    sizes per PCI are small in practice (tens of sites, not thousands), so
    this stays cheap even at full project scale."""
    df = sites_df.dropna(subset=["site_pci", "site_earfcn", "site_id_inferred", "site_lat", "site_lon"]).copy()
    df["site_pci"] = pd.to_numeric(df["site_pci"], errors="coerce")
    df["site_earfcn"] = pd.to_numeric(df["site_earfcn"], errors="coerce")
    df["site_lat"] = pd.to_numeric(df["site_lat"], errors="coerce")
    df["site_lon"] = pd.to_numeric(df["site_lon"], errors="coerce")
    df = df.dropna(subset=["site_pci", "site_earfcn", "site_lat", "site_lon"])
    instances = df.drop_duplicates(subset=["site_id_inferred", "site_pci", "site_earfcn"])

    rows = []
    for (pci, earfcn), grp in instances.groupby(["site_pci", "site_earfcn"]):
        members = grp[["site_id_inferred", "site_lat", "site_lon"]].drop_duplicates(subset=["site_id_inferred"]).to_dict("records")
        for a, b in combinations(members, 2):
            if a["site_id_inferred"] == b["site_id_inferred"]:
                continue
            dist_km = _haversine_scalar_m(a["site_lat"], a["site_lon"], b["site_lat"], b["site_lon"]) / 1000.0
            rows.append(
                {
                    "pci": int(pci),
                    "earfcn": int(earfcn),
                    "site_a": a["site_id_inferred"],
                    "site_b": b["site_id_inferred"],
                    "distance_km": dist_km,
                    "pair_key": frozenset((a["site_id_inferred"], b["site_id_inferred"])),
                }
            )
    return pd.DataFrame(rows)


def collision_pair_keys(g) -> set:
    """(pci, earfcn, site-pair) keys for every direct graph-edge collision —
    same criteria detect_pci_collisions uses, at pair granularity."""
    keys = set()
    for u, v in g.edges():
        if u[2] != v[2] or u[1] != v[1]:
            continue
        keys.add((u[1], u[2], frozenset((u[0], v[0]))))
    return keys


def confusion_pair_keys(conflicts: list) -> set:
    """(pci, earfcn, site-pair) keys for every pair of neighbor sites within
    a real Confusion group — same pattern build_order_conflict_table uses."""
    keys = set()
    for c in conflicts:
        for a, b in combinations(c["neighbor_sites"], 2):
            keys.add((c["pci"], c["earfcn"], frozenset((a, b))))
    return keys


def bucket_km(distance_km: float) -> int:
    return max(1, math.ceil(distance_km))


all_pairs_df = build_same_pci_pairs(selected_sites_df)
if all_pairs_df.empty:
    st.warning("No two distinct sites share the same PCI+EARFCN in this selection — nothing to show.")
    st.stop()

col_keys = collision_pair_keys(graph)
conf_keys = confusion_pair_keys(confusion_conflicts)
all_pairs_df["is_collision"] = all_pairs_df.apply(lambda r: (r["pci"], r["earfcn"], r["pair_key"]) in col_keys, axis=1)
all_pairs_df["is_confusion"] = all_pairs_df.apply(lambda r: (r["pci"], r["earfcn"], r["pair_key"]) in conf_keys, axis=1)

# ============================================================
# 1. PCI x distance-bucket table, filterable All / Collision / Confusion
# ============================================================
st.header("1. PCI reuse by distance — PCI × 1-15 km")
filter_choice = st.selectbox(
    "Filter",
    ["All", "Collision", "Confusion"],
    help=(
        "All: every same-PCI+EARFCN pair regardless of distance or graph adjacency. Collision: only "
        "pairs that are also direct graph-neighbor collisions (real handover OR within the neighbor "
        "distance threshold above). Confusion: only pairs that are both real observed neighbors of a "
        "common serving site."
    ),
)
if filter_choice == "Collision":
    filtered_pairs = all_pairs_df[all_pairs_df["is_collision"]]
elif filter_choice == "Confusion":
    filtered_pairs = all_pairs_df[all_pairs_df["is_confusion"]]
else:
    filtered_pairs = all_pairs_df

within_15 = filtered_pairs[filtered_pairs["distance_km"] <= 15.0].copy()
beyond_15 = int((filtered_pairs["distance_km"] > 15.0).sum())

if within_15.empty:
    st.info(f"No {filter_choice} same-PCI pairs within 15 km.")
else:
    within_15["bucket_km"] = within_15["distance_km"].apply(bucket_km)
    pivot = within_15.pivot_table(index="pci", columns="bucket_km", values="site_a", aggfunc="count", fill_value=0)
    for col in range(1, 16):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[list(range(1, 16))]
    pivot.columns = [f"{c:02d}km" for c in pivot.columns]  # zero-padded so any column sort stays numeric order
    pivot = pivot.loc[pivot.sum(axis=1) > 0].sort_index()
    st.dataframe(pivot, use_container_width=True)
    st.caption(f"{len(within_15)} pair(s) shown across {len(pivot)} PCI value(s), 0-15km. {beyond_15} pair(s) beyond 15km not shown.")

# ============================================================
# 2. Before/after optimizer — Collision+Confusion pairs by distance
# ============================================================
st.header("2. Before vs after PCI optimization — Collision/Confusion pairs by distance")

before_conflict_pairs = all_pairs_df[all_pairs_df["is_collision"] | all_pairs_df["is_confusion"]]
if before_conflict_pairs.empty:
    st.info("No Collision/Confusion pairs found in this selection — nothing to optimize or chart.")
else:
    pci_options = ["All (aggregate)"] + sorted(before_conflict_pairs["pci"].unique().tolist())
    pci_choice = st.selectbox(
        "PCI",
        pci_options,
        help=(
            "A specific PCI's collision doesn't just 'move' after optimization -- the site gets a "
            "DIFFERENT PCI number entirely, so its old pair may simply stop existing, while its new "
            "PCI could (rarely) coincide with some unrelated site elsewhere. Picking a specific PCI "
            "here shows: Before = pairs that used to share exactly this PCI value; After = pairs that "
            "STILL share exactly this PCI value once optimization is done (only sites the optimizer "
            "did NOT move away from it) -- so a real shift in distance for the SAME PCI number is "
            "visible, not hidden inside an aggregate total. 'All' shows the combined total across every "
            "touched PCI, same as before."
        ),
    )

    with st.spinner("Running closed-loop optimizer (Collision + Confusion)..."):
        recs, verification = recommend_pci_reassignment(
            selected_sites_df,
            events_df,
            selected_site_ids,
            graph,
            collision_conflicts,
            confusion_conflicts,
            {},
            [],
            check_collision=True,
            check_confusion=True,
            mod_values=[],
            check_grouped=False,
            distance_threshold_m=neighbor_distance_m,
        )

    if recs.empty:
        st.info("Optimizer found nothing to touch.")
    else:
        substitution = {(row["site"], row["current_pci"]): row["suggested_pci"] for _, row in recs.iterrows()}
        after_site_df, after_events_df = _apply_pci_assignments(selected_sites_df, events_df, substitution)
        after_graph = build_neighbor_graph(after_site_df, after_events_df, distance_threshold_m=neighbor_distance_m)
        after_confusion = detect_pci_confusion(after_events_df, site_ids_filter=set(selected_site_ids))

        after_col_keys = collision_pair_keys(after_graph)
        after_conf_keys = confusion_pair_keys(after_confusion)
        # Sites don't move -- PCI reassignment is the only thing that
        # changes -- so re-derive pairs from after_site_df (new PCI values)
        # and re-flag against the NEW collision/confusion key sets.
        after_pairs_df = build_same_pci_pairs(after_site_df)
        after_pairs_df["is_collision"] = after_pairs_df.apply(
            lambda r: (r["pci"], r["earfcn"], r["pair_key"]) in after_col_keys, axis=1
        )
        after_pairs_df["is_confusion"] = after_pairs_df.apply(
            lambda r: (r["pci"], r["earfcn"], r["pair_key"]) in after_conf_keys, axis=1
        )
        after_conflict_pairs = after_pairs_df[after_pairs_df["is_collision"] | after_pairs_df["is_confusion"]]

        # Apply the PCI selector to BOTH sides. "After" is deliberately
        # filtered by the SAME pci number, not by "whatever these specific
        # sites ended up with" -- a resolved pair simply vanishes from
        # this PCI's after-count (correct: it's fixed), while pairs that
        # still legitimately share this same PCI post-optimization show
        # up at their real (possibly different) distance.
        if pci_choice != "All (aggregate)":
            before_for_chart = before_conflict_pairs[before_conflict_pairs["pci"] == pci_choice]
            after_for_chart = after_conflict_pairs[after_conflict_pairs["pci"] == pci_choice]
        else:
            before_for_chart = before_conflict_pairs
            after_for_chart = after_conflict_pairs

        before_counts = before_for_chart[before_for_chart["distance_km"] <= 15.0]["distance_km"].apply(bucket_km).value_counts()
        after_counts = after_for_chart[after_for_chart["distance_km"] <= 15.0]["distance_km"].apply(bucket_km).value_counts()

        buckets = list(range(1, 16))
        chart_df = pd.DataFrame(
            {
                "Before": [int(before_counts.get(b, 0)) for b in buckets],
                "After": [int(after_counts.get(b, 0)) for b in buckets],
            },
            # Zero-padded ("01km".."15km") so the chart's alphabetic label
            # sort lines up with numeric order -- "1km".."15km" sorts as
            # 10,11,...,15,1,2,...,9 alphabetically, which is wrong.
            index=[f"{b:02d}km" for b in buckets],
        )
        st.bar_chart(chart_df)
        if pci_choice != "All (aggregate)":
            st.caption(
                f"PCI {pci_choice} only: {len(before_for_chart)} pair(s) before, {len(after_for_chart)} pair(s) "
                "still sharing this exact PCI value after optimization."
            )

        if verification.get("verified_clean"):
            st.success(
                f"Closed-loop verified: 0 remaining Collision/Confusion conflicts after "
                f"{verification['iterations']} iteration(s)."
            )
        else:
            st.warning(
                f"NOT fully resolved: {verification.get('remaining_conflicts')} conflict(s) remain "
                f"({verification.get('stopped_reason')})."
            )
