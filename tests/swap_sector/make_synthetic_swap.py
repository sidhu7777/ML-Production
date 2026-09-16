"""
Synthetic sector swaps on REAL data -- the ground truth the detector is scored against.

No confirmed real swap exists in project 193, so swaps are injected into the CONFIG only: whole
antenna endpoints (azimuth + electrical / mechanical tilt + antenna model / port) are exchanged
between sectors of ONE carrier group (same site + operator + technology + EARFCN). This tests
recovery of crossed configurations, not every physical effect of a feeder swap. Every RSRP
measurement in measurements.csv stays 100% real.

Eligible group: >= 2 sectors, with at least one pair of sectors whose azimuths are
>= MIN_AZIMUTH_GAP_DEG apart (a swap between antennas pointing the same way is invisible).

Every eligible group first gets a small random azimuth error (+-AZIMUTH_JITTER_DEG per sector),
because a real config is never exact. Without it, an untouched control would be identical to the
real config and could never produce a false alarm, so the false-alarm number would be meaningless.
Then half of the eligible groups (seeded, reproducible) also get a swap; the other half are the
controls ("azimuths slightly off, but no swap" -- the detector must NOT call a swap there).
Swap types: 2-sector swap, and 3-sector rotation (groups with a suitable third sector).

Output: data/cells_synthetic.csv = cells.csv with the test endpoints, plus true_azimuth (the real
config value), ground_truth (SWAPPED / CONTROL / NOT_ELIGIBLE) and swap_type.

Run from the ML/ directory (after build_dataset.py):
    venv\\Scripts\\python.exe -m tests.swap_sector.make_synthetic_swap
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

MIN_AZIMUTH_GAP_DEG = 60.0
AZIMUTH_JITTER_DEG = 15.0
INJECT_SHARE = 0.5
RANDOM_SEED = 42
# Everything that belongs to the physical antenna endpoint moves together; the PCI stays.
ENDPOINT_COLUMNS = ("azimuth", "e_tilt", "m_tilt", "antenna_model", "pattern_port")


def gap_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def exchange_endpoints(cells: pd.DataFrame, targets: list, sources: list) -> None:
    """cells.loc[targets] receive the antenna endpoints currently on cells.loc[sources] (in place)."""
    columns = [c for c in ENDPOINT_COLUMNS if c in cells.columns]
    cells.loc[targets, columns] = cells.loc[sources, columns].to_numpy()


def inject_swaps(cells: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    cells = cells.copy()
    for column in ENDPOINT_COLUMNS[1:]:
        if column in cells.columns:
            cells[column] = cells[column].astype(object)
    cells["true_azimuth"] = cells["azimuth"]
    cells["ground_truth"] = "NOT_ELIGIBLE"
    cells["swap_type"] = ""

    eligible = []
    for group_id, group in cells.groupby("group_id"):
        az = group["azimuth"]
        pairs = [(a, b) for a, b in combinations(group.index, 2) if gap_deg(az[a], az[b]) >= MIN_AZIMUTH_GAP_DEG]
        triples = [
            t for t in combinations(group.index, 3)
            if all(gap_deg(az[a], az[b]) >= MIN_AZIMUTH_GAP_DEG for a, b in combinations(t, 2))
        ]
        if pairs:
            eligible.append((group_id, pairs, triples))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))
    n_inject = int(round(len(eligible) * INJECT_SHARE))
    for rank, k in enumerate(order):
        group_id, pairs, triples = eligible[k]
        members = cells.index[cells["group_id"] == group_id]
        jitter = rng.uniform(-AZIMUTH_JITTER_DEG, AZIMUTH_JITTER_DEG, len(members))
        cells.loc[members, "azimuth"] = ((cells.loc[members, "true_azimuth"] + jitter) % 360.0).round(1)
        if rank >= n_inject:
            cells.loc[members, "ground_truth"] = "CONTROL"
            continue
        if triples and rng.random() < 0.5:
            a, b, c = triples[rng.integers(len(triples))]
            exchange_endpoints(cells, [a, b, c], [b, c, a])
            swap_type = "3-sector rotation"
        else:
            a, b = pairs[rng.integers(len(pairs))]
            exchange_endpoints(cells, [a, b], [b, a])
            swap_type = "2-sector swap"
        cells.loc[members, "ground_truth"] = "SWAPPED"
        cells.loc[members, "swap_type"] = swap_type
    return cells


def main() -> None:
    cells = inject_swaps(pd.read_csv(DATA_DIR / "cells.csv", dtype={"site_id": str}))
    cells.to_csv(DATA_DIR / "cells_synthetic.csv", index=False)
    groups = cells.groupby("group_id").agg(ground_truth=("ground_truth", "first"), swap_type=("swap_type", "first"))
    print(f"[synthetic] groups: {len(groups)} | eligible: {int((groups['ground_truth'] != 'NOT_ELIGIBLE').sum())}")
    print(groups["ground_truth"].value_counts().to_string())
    print(groups.loc[groups["ground_truth"] == "SWAPPED", "swap_type"].value_counts().to_string())
    print(f"Whole antenna endpoints exchanged; every eligible carrier also has azimuth errors of up to "
          f"+-{AZIMUTH_JITTER_DEG:.0f} deg. Measurements are unchanged -- still 100% real.")


if __name__ == "__main__":
    main()
