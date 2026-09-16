"""
Synthetic known-answer scenario for the coverage-redundancy model.

A 3x3 hexagonal Airtel layout with 1400 m between sites (so every regular site is needed), plus
planted cases whose correct verdict is known from the geometry alone:
  OVERLAY    900001    second macro 35 m from S5 pointing the same ways (+2 deg): each is redundant while
                       the other is on -> exactly ONE is COVERAGE_REDUNDANT, the other KEEP with
                       NEEDED_AFTER_REMOVALS. (At 50 m apart the model already gives 900001 its own
                       ~4,400 m2 weak zone under its mast, so that spacing is no longer a clean answer.)
  DOMINATED  900006    36 dBm antenna 60 m from S2 aimed like S2's first sector, so S2 is stronger
                       almost everywhere it reaches                   -> COVERAGE_REDUNDANT
                       (not a small cell out in the beam: one 250 m out fills the weak area under
                       S2's mast, so removing it genuinely costs a few dB -- not a known answer)
  DUPLICATE  900002    copy of S1 stored 5 m away under another site id -> DATA_DUPLICATE
  PCI DUP    900003.5  decimal-id copy of S3 150 m away with S3's PCIs  -> DATA_DUPLICATE
  ISOLATED   900004    lone site 1400 m beyond the layout               -> KEEP
  AMBIGUOUS  900005    eight different azimuths on one site             -> NOT_TESTABLE
  regular    S1..S9 except S5                                          -> KEEP
Azimuths get +-15 deg and tilts +-1 deg jitter so regular sites are not a perfect lattice. Rows use
site_prediction's columns and the polygon is "lat lon" WKT like the database, so the same cleaning
and orientation code as the real run is exercised.

Run from the ML/ directory:
    venv\\Scripts\\python.exe -m tests.overlapping.make_synthetic_overlap
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint
from shapely.ops import transform

from tests.overlapping.config import (
    OUTPUT_DIR,
    VERDICT_DUPLICATE,
    VERDICT_KEEP,
    VERDICT_NOT_TESTABLE,
    VERDICT_REDUNDANT,
    OverlapConfig,
)
from tests.overlapping.geo import LocalProjection
from tests.overlapping.run_overlap import InputData, run_pipeline, write_outputs

ORIGIN_LAT, ORIGIN_LON = 28.40, 77.00
ISD_M = 1400.0
OPERATOR = "Airtel"
REGULAR_SITES = [100001 + i for i in range(9)]


@dataclass
class SyntheticCase:
    inputs: InputData
    expected: dict[str, str]
    mutual_pair: tuple[str, str]
    roles: pd.DataFrame


def candidate_id(site) -> str:
    return f"{OPERATOR}|{site}"


def build_synthetic_case(seed: int = 7) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    proj = LocalProjection(ORIGIN_LAT, ORIGIN_LON)
    pci_pool = iter(int(p) for p in rng.permutation(504))
    rows: list[dict] = []
    roles: list[dict] = []
    positions: dict[object, tuple[float, float]] = {}
    azimuth_of: dict[object, list[float]] = {}
    pcis_of: dict[object, list[int]] = {}
    tilts_of: dict[object, list[float]] = {}

    def add_site(site, x, y, azimuths, role, tx=46.0, height=30.0, e_tilts=None, pcis=None):
        lat, lon = proj.to_latlon(x, y)
        site_pcis = list(pcis) if pcis is not None else [next(pci_pool) for _ in azimuths]
        for k, az in enumerate(azimuths):
            rows.append(
                {
                    "id": len(rows) + 1,
                    "site": site,
                    "cell_id": f"{site}_{k + 1}",
                    "sector": k + 1,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "pci": site_pcis[k],
                    "azimuth": float(az % 360.0),
                    "height": height,
                    "m_tilt": 0.0,
                    "e_tilt": float(e_tilts[k] if e_tilts is not None else 3 + rng.integers(-1, 2)),
                    "tx_power": tx,
                    "reference_signal_power": tx - 5.0,
                    "earfcn": 1750,
                    "band": "B1800",
                    "cluster": OPERATOR,
                    "Technology": "4G",
                }
            )
        positions[site] = (x, y)
        azimuth_of[site] = list(azimuths)
        pcis_of[site] = site_pcis
        tilts_of[site] = [row["e_tilt"] for row in rows[-len(azimuths):]]
        roles.append({"site": str(site), "role": role, "x_m": x, "y_m": y})

    for r in range(3):
        for c in range(3):
            site = REGULAR_SITES[r * 3 + c]
            x = (c + 0.5 * (r % 2) - 1.0) * ISD_M
            y = (r - 1.0) * ISD_M * math.sqrt(3.0) / 2.0
            add_site(site, x, y, [base + rng.uniform(-15, 15) for base in (0.0, 120.0, 240.0)], "regular")

    s1, s2, s3, s5, s7, s9 = (REGULAR_SITES[i] for i in (0, 1, 2, 4, 6, 8))
    x5, y5 = positions[s5]
    add_site(900001, x5 + 28.0, y5 + 21.0, [a + 2.0 for a in azimuth_of[s5]], "overlay", e_tilts=tilts_of[s5])

    x1, y1 = positions[s1]
    add_site(900002, x1 + 3.0, y1 + 4.0, [a + 3.0 for a in azimuth_of[s1]], "duplicate_co_located")

    x3, y3 = positions[s3]
    add_site(900003.5, x3 + 150.0, y3, [a + 5.0 for a in azimuth_of[s3]], "duplicate_same_pci", pcis=pcis_of[s3])

    x9, y9 = positions[s9]
    add_site(900004, x9 + ISD_M, y9, [base + rng.uniform(-15, 15) for base in (0.0, 120.0, 240.0)], "isolated")

    x7, y7 = positions[s7]
    add_site(900005, x7, y7 - 600.0, [k * 45.0 for k in range(8)], "ambiguous_config")

    x2, y2 = positions[s2]
    beam = math.radians(azimuth_of[s2][0])
    add_site(
        900006,
        x2 + 60.0 * math.sin(beam),
        y2 + 60.0 * math.cos(beam),
        [azimuth_of[s2][0]],
        "dominated_low_power",
        tx=36.0,
        e_tilts=[tilts_of[s2][0]],
    )

    hull_xy = MultiPoint(list(positions.values())).convex_hull.buffer(500.0)
    polygon_lat_lon = transform(lambda x, y, z=None: proj.to_latlon(x, y), hull_xy)   # stored "lat lon" like map_regions

    expected = {candidate_id(s): VERDICT_KEEP for s in REGULAR_SITES if s != s5}
    expected.update(
        {
            candidate_id(900006): VERDICT_REDUNDANT,
            candidate_id(900002): VERDICT_DUPLICATE,
            candidate_id("900003.5"): VERDICT_DUPLICATE,
            candidate_id(900004): VERDICT_KEEP,
            candidate_id(900005): VERDICT_NOT_TESTABLE,
        }
    )
    inputs = InputData(site_rows=pd.DataFrame(rows), polygon_wkts=[polygon_lat_lon.wkt], source=f"synthetic(seed={seed})")
    return SyntheticCase(inputs, expected, (candidate_id(s5), candidate_id(900001)), pd.DataFrame(roles))


def score(case: SyntheticCase, candidates: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    verdicts = dict(zip(candidates["candidate_id"], candidates["verdict"]))
    reasons = dict(zip(candidates["candidate_id"], candidates["reason"].fillna("")))
    rows = [
        {"candidate_id": cid, "expected": exp, "got": verdicts.get(cid, "MISSING"), "reason": reasons.get(cid, "")}
        for cid, exp in case.expected.items()
    ]
    pair = [verdicts.get(cid, "MISSING") for cid in case.mutual_pair]
    for cid, got in zip(case.mutual_pair, pair):
        rows.append({"candidate_id": cid, "expected": "one of pair REDUNDANT", "got": got, "reason": reasons.get(cid, "")})
    table = pd.DataFrame(rows)
    table["ok"] = table["expected"] == table["got"]
    kept_reason = next((reasons.get(cid, "") for cid, got in zip(case.mutual_pair, pair) if got == VERDICT_KEEP), "")
    pair_ok = sorted(pair) == sorted([VERDICT_REDUNDANT, VERDICT_KEEP]) and kept_reason.startswith("NEEDED_AFTER_REMOVALS")
    table.loc[table["expected"] == "one of pair REDUNDANT", "ok"] = pair_ok
    return table, bool(table["ok"].all())


def main() -> int:
    case = build_synthetic_case()
    run = run_pipeline(OverlapConfig(project_id=0, operator=OPERATOR), case.inputs)
    out = write_outputs(run, OUTPUT_DIR / "synthetic" / "airtel_site")
    case.roles.to_csv(out / "site_roles.csv", index=False)
    table, ok = score(case, run.candidates)
    table.to_csv(out / "scorecard.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nSYNTHETIC {'PASS' if ok else 'FAIL'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
