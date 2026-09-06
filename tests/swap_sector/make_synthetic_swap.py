"""
Build a synthetic sector-swap test case on top of REAL data.

Why synthetic injection is needed here at all: a prior check (real bearing +
dominant-PCI analysis against project 193's actual DT data, on all 6
candidate sites) found NO confirmed real sector-swap case -- the sites with
enough data all matched their configured azimuth->PCI mapping. To validate
that the detector actually fires on a real swap (a true-positive test, not
just a true-negative one), we need at least one confirmed case. Rather than
fabricate physics, we follow the SAME convention already established in
this codebase for PCI-optimization synthetic testing
(tests/Pci_optimization/pci_map_dashboard.py:
 inject_synthetic_mod_conflict / inject_synthetic_group_conflict /
 inject_synthetic_collision_confusion): keep everything real (site
 location, azimuth, cell_id, and every DT/RSRP/HO sample) and fabricate
 ONLY the single fact under test -- here, which PCI value a sector's
 config row reports -- exactly mimicking a real wiring/port-swap mistake
 where the physical antenna (fixed lat/lon/azimuth/cell hardware) ends up
 broadcasting a different PCI than intended. Rows are flagged
 is_synthetic=True / site_source_table="synthetic", same as that module,
 so synthetic rows can never be mistaken for real ones downstream.

Only run on the 3 sites from data/site_config.csv that survived
production's own site-identity filter (site+cell_id+sector+band+operator,
tools/pci_optimization/engine.py) with a COMPLETE 3-sector config --
2019, 358, 420162. (1.82, 2012, 430493 were left with 1-2 sectors after
that filter and are excluded from permutation-style testing here.)

For each of those 3 sites, 2 of its 3 sectors have their `site_pci` values
exchanged (azimuth/cell_id/lat/lon untouched -- that IS the swap); the
third sector is left untouched as a normal control. DT samples and HO
events are NOT touched at all -- they stay exactly as fetched by
fetch_data.py, so the "observed" side of any future comparison is 100%
real network behavior.

Run from the ML/ directory:
    venv\\Scripts\\python.exe -m tests.swap_sector.make_synthetic_swap
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# (site_id, cell_id_of_row_A, cell_id_of_row_B) -- PCIs of A and B get swapped.
SWAP_PLAN = [
    ("358", "358_6", "358_7"),
    ("2019", "2019_8", "2019_1"),
    ("420162", "420162_2", "420162_3"),
]


def main() -> None:
    site_df = pd.read_csv(DATA_DIR / "site_config.csv")
    site_df["is_synthetic"] = False
    site_df["swap_group_id"] = pd.NA
    site_df["ground_truth_swapped"] = False
    site_df["ground_truth_original_pci"] = pd.NA

    for group_id, (site_id, cell_a, cell_b) in enumerate(SWAP_PLAN, start=1):
        idx_a = site_df.index[site_df["site_cell_id_representative"] == cell_a]
        idx_b = site_df.index[site_df["site_cell_id_representative"] == cell_b]
        if len(idx_a) != 1 or len(idx_b) != 1:
            raise ValueError(f"Expected exactly one row each for {cell_a}/{cell_b}, got {len(idx_a)}/{len(idx_b)}")
        ia, ib = idx_a[0], idx_b[0]

        pci_a, pci_b = site_df.at[ia, "site_pci"], site_df.at[ib, "site_pci"]
        az_a, az_b = site_df.at[ia, "site_azimuth_deg"], site_df.at[ib, "site_azimuth_deg"]

        site_df.at[ia, "ground_truth_original_pci"] = pci_a
        site_df.at[ib, "ground_truth_original_pci"] = pci_b
        site_df.at[ia, "site_pci"] = pci_b
        site_df.at[ib, "site_pci"] = pci_a

        for i in (ia, ib):
            site_df.at[i, "is_synthetic"] = True
            site_df.at[i, "swap_group_id"] = group_id
            site_df.at[i, "ground_truth_swapped"] = True
            site_df.at[i, "site_source_table"] = "synthetic"

        print(
            f"[swap {group_id}] site {site_id}: {cell_a} (azimuth {az_a}) "
            f"PCI {pci_a}->{pci_b}  |  {cell_b} (azimuth {az_b}) PCI {pci_b}->{pci_a}"
        )

    out_path = DATA_DIR / "synthetic_swap_site_config.csv"
    site_df.to_csv(out_path, index=False)
    n_swapped = int(site_df["ground_truth_swapped"].sum())
    n_normal = int((~site_df["ground_truth_swapped"]).sum())
    print(f"\nSaved {out_path.name}: {n_swapped} synthetic-swapped sector rows, {n_normal} untouched (real/normal) rows")
    print("DT samples (dt_samples_6sites.csv) and HO events (ho_events_6sites.csv) are unchanged -- still 100% real.")


if __name__ == "__main__":
    main()
