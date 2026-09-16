"""
Demo set for learning the concept: 5 sites of ONE operator + technology (the detector checks each
operator + technology on its own, never mixed).

Default --source controlled: five fully synthetic LTE sites whose drive-test measurements are generated
from each site's TRUE antenna endpoints -- the antenna pattern the detector itself would select
(assumed CCVVPX308 file on B3, generic 3GPP pattern on B40), log-distance path loss (exponent 3) and
2 dB noise. Then the CONFIGURATION of two sites is crossed:
  DEMO-1, DEMO-2 (B3), DEMO-3 (B40)  normal
  DEMO-4 (B3)                        two antenna endpoints exchanged
  DEMO-5 (B40)                       three endpoints rotated, unequal azimuth spacing
These are not project 193 sites or field evidence.

Optional --source real: 5 real project-193 carriers of one operator + technology (3 left as they are,
2 with their two furthest-apart antenna endpoints exchanged in the configuration). Picked by coverage,
never by whether the injected swap is detected; measurements stay 100% real. Fails explicitly when
there are too few qualifying carriers.

Output: data/cells_demo.csv, data/measurements_demo.csv, data/demo_metadata.json

Run from the ML/ directory (after build_dataset.py for --source real):
    venv\\Scripts\\python.exe -m tests.swap_sector.make_demo
    venv\\Scripts\\python.exe -m tests.swap_sector.make_demo --source real --operator Airtel --technology LTE
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np
import pandas as pd

from tests.swap_sector.detect_sector_swap import DATA_DIR, angle_diff_deg, detect
from tests.swap_sector.make_synthetic_swap import exchange_endpoints

DEMO_NORMAL_SITES = 3
DEMO_SWAP_SITES = 2
MIN_GAP_DEG = 60.0
SWAP_MIN_GAP_DEG = 90.0
MIN_COVERED_DIRECTIONS = 6
MIN_STRONGEST_SPOTS = 10

# site, EARFCN, true azimuths, configured endpoint order (None = configured correctly)
CONTROLLED_SITES = [
    ("DEMO-1", 1346, (0.0, 120.0, 240.0), None),
    ("DEMO-2", 1346, (30.0, 150.0, 270.0), None),
    ("DEMO-3", 39150, (10.0, 130.0, 250.0), None),
    ("DEMO-4", 1346, (40.0, 160.0, 280.0), (1, 0, 2)),
    ("DEMO-5", 39150, (20.0, 100.0, 220.0), (1, 2, 0)),
]
DEMO_E_TILT, DEMO_M_TILT = 4.0, 2.0
DEMO_NOISE_DB = 2.0


def candidate_carriers(cells: pd.DataFrame, measurements: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    rows = []
    normal = real[(real["verdict"] == "NORMAL") & (real["n_sectors"] == 3) & (real["match_level"] == "ID")]
    for result in normal.itertuples():
        group = cells[cells["group_id"] == result.group_id]
        min_gap = min(float(angle_diff_deg(a, b)) for a, b in combinations(group["azimuth"], 2))
        meas = measurements[measurements["group_id"] == result.group_id]
        strongest = meas.loc[meas.groupby("location_id")["rsrp"].idxmax()]
        covered = strongest["bearing_deg"].floordiv(30).nunique()
        weakest_sector = strongest["pci"].value_counts().reindex(group["pci"], fill_value=0).min()
        if min_gap >= MIN_GAP_DEG and covered >= MIN_COVERED_DIRECTIONS and weakest_sector >= MIN_STRONGEST_SPOTS:
            rows.append({
                "group_id": result.group_id, "site_id": result.site_id, "operator": result.operator,
                "technology": result.technology, "min_gap": min_gap, "covered": covered, "locations": result.locations,
            })
    columns = ["group_id", "site_id", "operator", "technology", "min_gap", "covered", "locations"]
    ranked = pd.DataFrame(rows, columns=columns).sort_values(["covered", "locations"], ascending=False)
    return ranked.drop_duplicates(["operator", "site_id"]).reset_index(drop=True)


def with_swap(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    a, b = max(
        combinations(group.index, 2),
        key=lambda pair: float(angle_diff_deg(group.at[pair[0], "azimuth"], group.at[pair[1], "azimuth"])),
    )
    exchange_endpoints(group, [a, b], [b, a])
    group["ground_truth"] = "SWAPPED"
    group["swap_type"] = "2-sector swap"
    return group


def pick_demo(ranked: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame | None:
    """ranked = candidates of ONE operator + technology."""
    swap_groups = []
    for candidate in ranked[ranked["min_gap"] >= SWAP_MIN_GAP_DEG].itertuples():
        # Pick by coverage, never by whether the injected swap is detected.
        swap_groups.append(with_swap(base[base["group_id"] == candidate.group_id]))
        if len(swap_groups) == DEMO_SWAP_SITES:
            break
    swap_sites = {group["site_id"].iloc[0] for group in swap_groups}
    normal_ids = ranked[~ranked["site_id"].isin(swap_sites)].head(DEMO_NORMAL_SITES)["group_id"]
    if len(swap_groups) < DEMO_SWAP_SITES or len(normal_ids) < DEMO_NORMAL_SITES:
        return None
    return pd.concat(swap_groups + [base[base["group_id"].isin(normal_ids)]], ignore_index=True)


def make_real_demo(operator: str | None, technology: str | None) -> None:
    cells = pd.read_csv(DATA_DIR / "cells.csv", dtype={"site_id": str})
    measurements = pd.read_csv(DATA_DIR / "measurements.csv")
    real = detect(cells, measurements)
    ranked_all = candidate_carriers(cells, measurements, real)
    base = cells.assign(true_azimuth=cells["azimuth"], ground_truth="CONTROL", swap_type="")
    for column in ("e_tilt", "m_tilt", "antenna_model", "pattern_port"):
        base[column] = base[column].astype(object)
    scopes = ranked_all.groupby(["operator", "technology"]).size().sort_values(ascending=False).index
    for scope_operator, scope_technology in scopes:
        if (operator and scope_operator != operator) or (technology and scope_technology != technology):
            continue
        ranked = ranked_all[(ranked_all["operator"] == scope_operator) & (ranked_all["technology"] == scope_technology)]
        demo = pick_demo(ranked, base)
        if demo is None:
            print(f"[demo] {scope_operator} {scope_technology}: not enough clear carriers, trying the next operator + technology")
            continue
        demo.to_csv(DATA_DIR / "cells_demo.csv", index=False)
        measurements[measurements["group_id"].isin(demo["group_id"])].to_csv(DATA_DIR / "measurements_demo.csv", index=False)
        (DATA_DIR / "demo_metadata.json").write_text(json.dumps({
            "source": "real",
            "description": "Real project DT; carriers selected by coverage among detector-Normal ones; two with antenna "
                           "endpoints exchanged in the configuration. Not an accuracy benchmark."}, indent=2))
        summary = demo.groupby("group_id").agg(site=("site_id", "first"), band=("band", "first"), demo=("ground_truth", "first"))
        print(f"[demo] {scope_operator} {scope_technology}:\n{summary.to_string()}")
        return
    raise SystemExit("No operator + technology has 2 clear swap carriers and 3 clear normal carriers "
                     f"(detector-Normal 3-sector carriers qualifying: {len(ranked_all)}).")


def make_controlled_demo(operator: str = "Airtel") -> None:
    """Known physical antenna endpoints; the RF is generated before any configuration is crossed."""
    from tests.swap_sector.antenna_profile import carrier_frequency_mhz, horizontal_gain_db, select_pattern
    rng = np.random.default_rng(20260914)
    cell_rows, measurement_rows = [], []
    for number, (site, earfcn, true_azimuths, crossing) in enumerate(CONTROLLED_SITES, start=1):
        frequency, band = carrier_frequency_mhz(earfcn, "LTE")
        lat, lon = 28.62 + 0.02 * number, 77.36
        group_id = f"{operator}|{site}|LTE|{earfcn}"
        order = crossing or (0, 1, 2)
        moved = sum(order[k] != k for k in range(3))
        base = {"group_id": group_id, "operator": operator, "site_id": site, "technology": "LTE", "band": band,
                "earfcn": earfcn, "e_tilt": DEMO_E_TILT, "m_tilt": DEMO_M_TILT, "height": 30.0, "antenna_model": "",
                "pattern_port": "", "frequency_mhz": frequency, "frequency_band": band, "frequency_source": "DT EARFCN",
                "site_lat": lat, "site_lon": lon, "match_level": "ID"}
        patterns = [select_pattern({**base, "azimuth": azimuth}) for azimuth in true_azimuths]
        pcis = [number * 10 + k for k in range(3)]
        for k, pci in enumerate(pcis):
            cell_rows.append({**base, "pci": pci, "azimuth": true_azimuths[order[k]], "true_azimuth": true_azimuths[k],
                              "ground_truth": "SWAPPED" if crossing else "CONTROL",
                              "swap_type": "" if not crossing else "2-sector swap" if moved == 2 else "3-sector rotation"})
        for bearing in np.arange(5.0, 360.0, 10.0):
            for distance in (150.0, 300.0, 450.0, 600.0):
                point_lat = lat + distance * np.cos(np.radians(bearing)) / 111320.0
                point_lon = lon + distance * np.sin(np.radians(bearing)) / (111320.0 * np.cos(np.radians(lat)))
                powers = [-65.0 - 30.0 * np.log10(distance / 100.0) + float(horizontal_gain_db(p, bearing - azimuth))
                          + rng.normal(0.0, DEMO_NOISE_DB) for p, azimuth in zip(patterns, true_azimuths)]
                strongest = int(np.argmax(powers))
                for k, pci in enumerate(pcis):
                    measurement_rows.append({
                        "group_id": group_id, "location_id": f"{group_id}|{bearing:g}|{distance:g}", "session_id": number,
                        "lat": point_lat, "lon": point_lon, "pci": pci, "distance_m": distance, "bearing_deg": bearing,
                        "rsrp": powers[k], "samples": 1, "is_serving": k == strongest})
    pd.DataFrame(cell_rows).to_csv(DATA_DIR / "cells_demo.csv", index=False)
    pd.DataFrame(measurement_rows).to_csv(DATA_DIR / "measurements_demo.csv", index=False)
    (DATA_DIR / "demo_metadata.json").write_text(json.dumps({
        "source": "controlled",
        "description": "Fully synthetic sites. Drive-test RF is generated from each site's true antenna endpoints "
                       "(assumed CCVVPX308 pattern on B3, generic 3GPP pattern on B40) with log-distance loss and 2 dB noise. "
                       "DEMO-1..3 normal, DEMO-4 two antenna endpoints exchanged, DEMO-5 three endpoints rotated. "
                       "Not project 193 sites or field evidence."}, indent=2))
    print("[demo] Created five synthetic sites: three normal, one pair swap, one three-sector rotation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["controlled", "real"], default="controlled")
    parser.add_argument("--operator", help="e.g. Airtel, JIO, Vi")
    parser.add_argument("--technology", help="LTE or NR")
    args = parser.parse_args()
    if args.source == "controlled":
        make_controlled_demo(args.operator or "Airtel")
    else:
        make_real_demo(args.operator, args.technology)


if __name__ == "__main__":
    main()
