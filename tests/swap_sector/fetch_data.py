"""
Fetch real data for the sector-swap test case: site/sector config, nearby
drive-test (DT) samples, and derived PCI-handover events, for a fixed set
of 6 known-good 3-sector sites in project 193 (India) — already verified
(separately) to have clean azimuth/PCI config and dense enough DT/HO
coverage to be usable as-is.

Deliberately reuses production's own data-fetch/derivation code instead of
re-implementing it, so this dataset matches what the live app would
compute:
  - fetch_site_prediction()      tools/pci_optimization/engine.py
  - get_engine()                 tools/pci_optimization/db.py
  - _fetch_project_session_ids() tests/Pci_optimization/pci_optimization_dataset_test.py
  - _fetch_network_logs()        tests/Pci_optimization/pci_optimization_dataset_test.py
  - _detect_pci_transitions()    tests/Pci_optimization/pci_optimization_dataset_test.py
    (delegates to new_pdf_report's _detect_column_transitions, the
    timestamp-aware version — production's own detect_handover_events()
    is known to undercount ~10x, see that module's docstring)
  - _enrich_transition_sites()   tests/Pci_optimization/pci_optimization_dataset_test.py
  - _haversine_m() / _bearing_deg()  tests/Pci_optimization/pci_optimization_dataset_test.py

Run from the ML/ directory so the `tools`/`tests` packages resolve:
    venv\\Scripts\\python.exe -m tests.swap_sector.fetch_data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools.pci_optimization.db import get_engine
from tools.pci_optimization.engine import fetch_site_prediction
from tests.Pci_optimization.pci_optimization_dataset_test import (
    _fetch_project_session_ids,
    _fetch_network_logs,
    _detect_pci_transitions,
    _enrich_transition_sites,
    _haversine_m,
    _bearing_deg,
)

PROJECT_ID = 193
REGION = "india"
OPERATOR = "all"
RADIUS_M = 1500.0

# 6 sites pre-validated (separately) to have 3 clean sectors each with
# spread-out azimuths, non-zero PCIs, and dense DT/HO coverage.
SITES = ["1.82", "2012", "2019", "358", "420162", "430493"]

DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    engine = get_engine(REGION)

    # ---- 1. Configured site/sector data (topology evidence) ----
    site_df = fetch_site_prediction(PROJECT_ID, OPERATOR, engine)
    site_df = site_df[site_df["site_id_inferred"].astype(str).isin(SITES)].reset_index(drop=True)
    site_df.to_csv(DATA_DIR / "site_config.csv", index=False)
    print(f"[site_config] {len(site_df)} sector rows across {site_df['site_id_inferred'].nunique()} sites")

    # site centroid (mean of its sectors' lat/lon — all 3 share one physical location)
    site_centers = (
        site_df.groupby("site_id_inferred")[["site_lat", "site_lon"]].mean().rename(
            columns={"site_lat": "center_lat", "site_lon": "center_lon"}
        )
    )

    # ---- 2. Drive-test samples (RF spatial evidence) ----
    with engine.connect() as conn:
        session_ids, _meta = _fetch_project_session_ids(PROJECT_ID, conn)
        # _fetch_network_logs (unlike fetch_site_prediction) has no "all"/"auto"
        # wildcard -- it filters m_alpha_long/m_alpha_short literally against
        # whatever operator string is passed, so an operator filter must be
        # omitted (None), not passed as the literal string "all".
        log_df = _fetch_network_logs(session_ids, None, conn, primary_only=True)
    print(f"[dt_raw] {len(log_df)} rows across {log_df['session_id'].nunique()} sessions (all of project {PROJECT_ID})")

    log_df["lat"] = pd.to_numeric(log_df["lat"], errors="coerce")
    log_df["lon"] = pd.to_numeric(log_df["lon"], errors="coerce")
    log_df = log_df.dropna(subset=["lat", "lon"])

    # Keep only DT points within RADIUS_M of one of the 6 sites; tag each
    # kept row with its nearest of the 6 and the calculated distance/bearing
    # from that site -- this is the "bearing_from_site" + "angle_bin" prep
    # the sector-swap directional profile needs downstream.
    kept_frames = []
    for site_id, row in site_centers.loc[site_centers.index.isin(SITES)].iterrows():
        dist = _haversine_m(row["center_lat"], row["center_lon"], log_df["lat"], log_df["lon"])
        nearby = log_df.loc[dist.values <= RADIUS_M].copy()
        if nearby.empty:
            print(f"[dt_site={site_id}] 0 points within {RADIUS_M:.0f}m")
            continue
        nearby["site_id_inferred"] = site_id
        nearby["distance_from_site_m"] = _haversine_m(row["center_lat"], row["center_lon"], nearby["lat"], nearby["lon"]).values
        nearby["bearing_from_site_deg"] = _bearing_deg(row["center_lat"], row["center_lon"], nearby["lat"], nearby["lon"])
        nearby["angle_bin_10deg"] = (np.floor(nearby["bearing_from_site_deg"] / 10.0) * 10).astype(int)
        kept_frames.append(nearby)
        print(f"[dt_site={site_id}] {len(nearby)} points within {RADIUS_M:.0f}m")

    dt_df = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()
    dt_df.to_csv(DATA_DIR / "dt_samples_6sites.csv", index=False)
    print(f"[dt_samples_6sites] {len(dt_df)} total rows saved")

    # ---- 3. Handover / PCI-transition events (mobility evidence) ----
    transitions = _detect_pci_transitions(log_df)
    if not transitions.empty:
        enriched = _enrich_transition_sites(transitions, site_df)
        touch_mask = (
            enriched.get("from_site_id_inferred", pd.Series(dtype=object)).astype(str).isin(SITES)
            | enriched.get("to_site_id_inferred", pd.Series(dtype=object)).astype(str).isin(SITES)
        )
        enriched = enriched.loc[touch_mask].reset_index(drop=True)
    else:
        enriched = transitions
    enriched.to_csv(DATA_DIR / "ho_events_6sites.csv", index=False)
    print(f"[ho_events_6sites] {len(enriched)} transition events touching the 6 sites")


if __name__ == "__main__":
    main()
