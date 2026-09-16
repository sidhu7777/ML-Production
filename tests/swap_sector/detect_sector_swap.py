"""
Sector-swap detector -- TEST CASE ONLY. Reads data/cells*.csv + data/measurements*.csv (built by
build_dataset.py / make_synthetic_swap.py / make_demo.py); never touches the database.

Active model: method="pattern" (pattern_detector.py + antenna_profile.py). The expected 36 x 10-degree
antenna-gain profile of every configured antenna endpoint (azimuth + EXACT / ASSUMED / APPROXIMATE
pattern) is compared with each PCI's drive-test profile, for every way of assigning endpoints to PCIs.
Verdicts: NORMAL, PROBABLE_SWAP, CONFIRMED_SWAP (rule-based, EXACT patterns only), AZIMUTH_MISMATCH
(direction / RF anomaly), AMBIGUOUS, NOT_ENOUGH_DATA, NOT_TESTABLE. No field calibration is claimed.

Kept for comparison only (validate.py): method="combined" / "direction" / "ranking" (evidence.py) and
method="legacy" (analyse_group_legacy below). Legacy baseline:
Per carrier group (site + operator + technology + EARFCN) it compares the PREDICTED direction of
each sector (site_prediction azimuth) with the REAL coverage direction measured in the drive test:

  1. Drive-test spots around the site are split by bearing into 36 bins of BIN_SIZE_DEG (10 deg).
  2. At each spot, the strongest of the group's PCIs is taken (serving + neighbour RSRP).
  3. A bin is won by a PCI when the bin has >= MIN_BIN_SPOTS spots and that PCI is the strongest
     at >= MIN_BIN_DOMINANCE of them.
  4. Real direction of a PCI = circular mean of the centres of the bins it wins, weighted by its
     strongest-spot count. A PCI needs >= MIN_PCI_BINS won bins to get a real direction.
  5. A predicted direction is right when the real direction is within +-DIRECTION_LIMIT_DEG.
  6. Every way of re-assigning the predicted directions across the PCIs is tried:
       NORMAL            the prediction as it is fits every sector that has a real direction
       SWAP_SUSPECTED    the prediction does not fit, but crossing the directions of sectors does
       AZIMUTH_MISMATCH  neither the prediction nor any crossing fits -- a direction is simply wrong
       NOT_ENOUGH_DATA   too few sectors with a real direction, or more than one answer fits
       NOT_TESTABLE      only one sector on the carrier, or sectors point the same way

PCI numbers are never changed (that would be PCI optimization). A swap means the coverage
directions of sectors are crossed -- e.g. feeder cables on the wrong antennas, or azimuths swapped
in the site table.

Run from the ML/ directory (production checks one operator + technology at a time):
    venv\\Scripts\\python.exe -m tests.swap_sector.detect_sector_swap --config real --operator Airtel --technology LTE
    venv\\Scripts\\python.exe -m tests.swap_sector.detect_sector_swap --config synthetic
    venv\\Scripts\\python.exe -m tests.swap_sector.detect_sector_swap --config demo
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

BIN_SIZE_DEG = 10
MIN_BIN_SPOTS = 3
MIN_BIN_DOMINANCE = 0.5
MIN_PCI_BINS = 2
DIRECTION_LIMIT_DEG = 30.0
MIN_SECTOR_SHARE_WITH_DIRECTION = 2 / 3
MIN_SWAP_ANGLE_DEG = 30.0       # re-assignments moving no sector this much are the same answer
MAX_FULL_PERMUTATION_SECTORS = 4
SWAP_VERDICTS = {"SWAP_SUSPECTED", "PROBABLE_SWAP", "CONFIRMED_SWAP"}


def angle_diff_deg(a, b) -> np.ndarray:
    return np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0)


def same_answer(azimuths_a, azimuths_b) -> bool:
    return bool(angle_diff_deg(azimuths_a, azimuths_b).max() < MIN_SWAP_ANGLE_DEG)


def circular_mean_deg(angles_deg, weights) -> float:
    rad = np.radians(np.asarray(angles_deg, dtype=float))
    w = np.asarray(weights, dtype=float)
    return float((np.degrees(np.arctan2((w * np.sin(rad)).sum(), (w * np.cos(rad)).sum())) + 360.0) % 360.0)


def format_values(values) -> str:
    return ";".join(f"{float(v):.1f}" for v in values)


def parse_values(text) -> list[float]:
    return [float(v) for v in str(text).split(";") if v != ""]


def candidate_assignments(azimuths: np.ndarray) -> list[tuple[int, ...]]:
    """perm[i] = index of the predicted direction given to sector i. Identity (the prediction) first."""
    n = len(azimuths)
    identity = tuple(range(n))
    if n <= MAX_FULL_PERMUTATION_SECTORS:
        options = list(itertools.permutations(range(n)))
    else:
        options = []
        for i, j in itertools.combinations(range(n), 2):
            swapped = list(identity)
            swapped[i], swapped[j] = j, i
            options.append(tuple(swapped))
    seen = {tuple(np.round(azimuths, 1))}
    result = [identity]
    for perm in options:
        moved = azimuths[list(perm)]
        signature = tuple(np.round(moved, 1))
        if signature in seen or same_answer(azimuths, moved):
            continue
        seen.add(signature)
        result.append(perm)
    return result


def strongest_per_spot(group_meas: pd.DataFrame) -> pd.DataFrame:
    if group_meas.empty:
        return group_meas
    return group_meas.loc[group_meas.groupby("location_id")["rsrp"].idxmax()]


def direction_profile(group_meas: pd.DataFrame, pcis: list[int], require_comparison=False) -> pd.DataFrame:
    """One row per 10-degree bin: spots in the bin, the PCI that wins it (NaN if none), its share."""
    bins = list(range(0, 360, BIN_SIZE_DEG))
    group_meas = group_meas[group_meas["pci"].isin(pcis)]
    if require_comparison and not group_meas.empty:
        group_meas = group_meas[group_meas.groupby("location_id").pci.transform("nunique") >= 2]
        # Exact RSRP ties are not wins for whichever PCI happens to appear first.
        peaks = group_meas.groupby("location_id").rsrp.transform("max")
        tied = group_meas[group_meas.rsrp.eq(peaks)].groupby("location_id").size()
        group_meas = group_meas[~group_meas.location_id.isin(tied[tied > 1].index)]
    strongest = strongest_per_spot(group_meas)
    if strongest.empty:
        counts = pd.DataFrame(0, index=bins, columns=pcis)
    else:
        bin_of_spot = (strongest["bearing_deg"] // BIN_SIZE_DEG * BIN_SIZE_DEG).astype(int) % 360
        counts = pd.crosstab(bin_of_spot, strongest["pci"]).reindex(index=bins, columns=pcis, fill_value=0)
    spots = counts.sum(axis=1)
    top_spots = counts.max(axis=1)
    share = (top_spots / spots.where(spots > 0)).fillna(0.0)
    won = (spots >= MIN_BIN_SPOTS) & (share >= MIN_BIN_DOMINANCE)
    if require_comparison:
        won &= counts.eq(top_spots, axis=0).sum(axis=1).eq(1)
    return pd.DataFrame({
        "bin_deg": bins,
        "spots": spots.to_numpy(),
        "dominant_pci": counts.idxmax(axis=1).where(won).to_numpy(),
        "dominant_spots": top_spots.where(won, 0).to_numpy(),
        "share": share.round(2).to_numpy(),
    })


def real_directions(profile: pd.DataFrame, pcis: list[int]) -> dict[int, float]:
    result = {}
    for pci in pcis:
        won = profile[profile["dominant_pci"] == pci]
        if len(won) >= MIN_PCI_BINS:
            result[pci] = circular_mean_deg(won["bin_deg"] + BIN_SIZE_DEG / 2, won["dominant_spots"])
    return result


def crossing_text(pcis: list[int], azimuths: np.ndarray, perm: tuple[int, ...], real: dict[int, float]) -> str:
    parts = []
    for k, pci in enumerate(pcis):
        if angle_diff_deg(azimuths[k], azimuths[perm[k]]) < 1:
            continue
        seen = f"really covers {real[pci]:.0f}°" if pci in real else "has no clear real direction"
        parts.append(f"PCI {pci} {seen} (predicted {azimuths[k]:.0f}°)")
    return "Coverage directions crossed: " + "; ".join(parts)


def analyse_group_legacy(group_cells: pd.DataFrame, group_meas: pd.DataFrame) -> dict:
    group_cells = group_cells.sort_values("pci")
    first = group_cells.iloc[0]
    pcis = group_cells["pci"].astype(int).tolist()
    azimuths = group_cells["azimuth"].astype(float).to_numpy()
    result = {
        "group_id": first["group_id"], "operator": first["operator"], "site_id": str(first["site_id"]),
        "technology": first["technology"], "band": first["band"], "earfcn": int(first["earfcn"]),
        "match_level": first["match_level"], "n_sectors": len(pcis),
        "pcis": ";".join(str(p) for p in pcis),
        "predicted_azimuths": format_values(azimuths), "real_directions": "", "bins_won": "",
        "sectors_with_direction": 0, "locations": 0, "largest_gap_deg": np.nan,
        "best_azimuths": format_values(azimuths), "changed_pcis": "",
    }

    def finish(verdict: str, reason: str) -> dict:
        result.update(verdict=verdict, reason=reason)
        return result

    if len(pcis) < 2:
        return finish("NOT_TESTABLE", "Only one sector on this carrier")
    candidates = candidate_assignments(azimuths)
    if len(candidates) == 1:
        return finish("NOT_TESTABLE", f"All sectors point within {MIN_SWAP_ANGLE_DEG:.0f}° of each other")

    profile = direction_profile(group_meas, pcis)
    real = real_directions(profile, pcis)
    result.update(
        locations=int(profile["spots"].sum()),
        sectors_with_direction=len(real),
        real_directions=format_values([real.get(p, np.nan) for p in pcis]),
        bins_won=";".join(str(int((profile["dominant_pci"] == p).sum())) for p in pcis),
    )
    if len(real) < max(2, MIN_SECTOR_SHARE_WITH_DIRECTION * len(pcis)):
        return finish("NOT_ENOUGH_DATA", f"Only {len(real)} of {len(pcis)} sectors have a clear real direction in the drive test")

    measured = [k for k, pci in enumerate(pcis) if pci in real]
    real_az = np.array([real[pcis[k]] for k in measured])
    assigned = [azimuths[list(perm)] for perm in candidates]
    gaps = [angle_diff_deg(real_az, a[measured]) for a in assigned]
    largest = np.array([g.max() for g in gaps])
    average = np.array([g.mean() for g in gaps])
    fits = largest <= DIRECTION_LIMIT_DEG
    result["largest_gap_deg"] = round(float(largest[0]), 1)

    if fits[0]:
        better = [i for i in range(1, len(candidates)) if fits[i] and average[i] < average[0]]
        if better:
            return finish("NOT_ENOUGH_DATA", "Both the prediction and a crossing of directions fit the drive test")
        return finish("NORMAL", f"Real directions match the prediction within ±{DIRECTION_LIMIT_DEG:.0f}° (largest gap {largest[0]:.0f}°)")

    fitting = [i for i in range(1, len(candidates)) if fits[i]]
    if not fitting:
        worst = measured[int(np.argmax(gaps[0]))]
        return finish(
            "AZIMUTH_MISMATCH",
            f"Direction wrong and no crossing explains it: PCI {pcis[worst]} really covers {real[pcis[worst]]:.0f}°, "
            f"predicted {azimuths[worst]:.0f}° ({largest[0]:.0f}° off)",
        )
    best = min(fitting, key=lambda i: average[i])
    if any(not same_answer(assigned[i], assigned[best]) for i in fitting):
        return finish("NOT_ENOUGH_DATA", "More than one crossing of directions fits the drive test")
    result.update(
        best_azimuths=format_values(assigned[best]),
        changed_pcis=";".join(str(pcis[k]) for k in range(len(pcis)) if candidates[best][k] != k),
    )
    return finish("SWAP_SUSPECTED", crossing_text(pcis, azimuths, candidates[best], real))


def detect(cells: pd.DataFrame, measurements: pd.DataFrame, handovers=None, settings=None) -> pd.DataFrame:
    from tests.swap_sector import pattern_detector
    from tests.swap_sector.evidence import analyse, Settings
    settings = settings or Settings(method="pattern")
    meas_by_group = dict(tuple(measurements.groupby("group_id")))
    empty = measurements.iloc[0:0]
    if handovers is None and (DATA_DIR / "handovers.csv").exists():
        handovers = pd.read_csv(DATA_DIR / "handovers.csv")
    ho_by_group = dict(tuple(handovers.groupby("group_id"))) if handovers is not None and not handovers.empty else {}
    rows = []
    for group_id, group_cells in cells.groupby("group_id"):
        group_meas, group_ho = meas_by_group.get(group_id, empty), ho_by_group.get(group_id)
        if settings.method == "legacy":
            rows.append(analyse_group_legacy(group_cells, group_meas))
        elif settings.method == "pattern":
            rows.append(pattern_detector.analyse(group_cells, group_meas, group_ho, settings))
        else:
            rows.append(analyse(group_cells, group_meas, group_ho, settings))
    return pd.DataFrame(rows)


def outcome(row: pd.Series) -> str:
    if row["ground_truth"] == "SWAPPED":
        if row["verdict"] in SWAP_VERDICTS:
            correct = same_answer(parse_values(row["best_azimuths"]), parse_values(row["true_azimuths"]))
            return "FOUND" if correct else "WRONG_SECTORS"
        return "MISSED" if row["verdict"] == "NORMAL" else "UNDECIDED"
    if row["ground_truth"] == "CONTROL":
        if row["verdict"] in SWAP_VERDICTS:
            return "FALSE_ALARM"
        return "CORRECT" if row["verdict"] == "NORMAL" else "UNDECIDED"
    return ""


def add_ground_truth(results: pd.DataFrame, synthetic_cells: pd.DataFrame, real_results: pd.DataFrame) -> pd.DataFrame:
    truth = synthetic_cells.sort_values("pci").groupby("group_id").agg(
        ground_truth=("ground_truth", "first"),
        swap_type=("swap_type", "first"),
        true_azimuths=("true_azimuth", format_values),
    )
    real = real_results[["group_id", "verdict"]].rename(columns={"verdict": "real_verdict"})
    out = results.merge(truth, on="group_id", how="left").merge(real, on="group_id", how="left")
    out["swap_type"] = out["swap_type"].fillna("")
    out["outcome"] = out.apply(outcome, axis=1)
    return out


def scorecard(results: pd.DataFrame) -> pd.DataFrame:
    """Synthetic test score. The second row only counts carriers whose REAL configuration was Normal, so
    real configuration problems do not blur the measurement of the detector itself (selection-biased)."""
    rows = []
    for scope, subset in (
        ("All eligible carriers", results),
        ("Detector-selected Normal subset (selection-biased)", results[results["real_verdict"] == "NORMAL"]),
    ):
        swapped = subset[subset["ground_truth"] == "SWAPPED"]["outcome"]
        control = subset[subset["ground_truth"] == "CONTROL"]["outcome"]
        rows.append({
            "scope": scope,
            "injected": len(swapped),
            "found": int((swapped == "FOUND").sum()),
            "wrong_sectors": int((swapped == "WRONG_SECTORS").sum()),
            "missed": int((swapped == "MISSED").sum()),
            "undecided": int((swapped == "UNDECIDED").sum()),
            "untouched": len(control),
            "false_alarms": int((control == "FALSE_ALARM").sum()),
            "correct_normal": int((control == "CORRECT").sum()),
            "untouched_undecided": int((control == "UNDECIDED").sum()),
        })
    return pd.DataFrame(rows)


TEST_CELL_FILES = {"synthetic": "cells_synthetic.csv", "demo": "cells_demo.csv"}


def in_scope(cells: pd.DataFrame, operator: str | None, technology: str | None) -> pd.DataFrame:
    """Production checks one operator + technology at a time, never mixed."""
    if operator:
        cells = cells[cells["operator"] == operator]
    if technology:
        cells = cells[cells["technology"] == technology]
    return cells


def reference_provenance():
    info = json.loads((DATA_DIR / "raw_fetch_info.json").read_text())
    references = json.loads((DATA_DIR.parent / "reference_provenance.json").read_text())
    return references.get(str(info["project_id"]), {"status": "UNKNOWN", "basis": "Reference independence has not been established for this project"})


def run(config_name: str, operator: str | None = None, technology: str | None = None, settings=None) -> pd.DataFrame:
    """real = the project's site_prediction; synthetic / demo = test configurations with known ground truth.
    Pass operator + technology to check one scope, the way production runs."""
    real_cells = in_scope(pd.read_csv(DATA_DIR / "cells.csv", dtype={"site_id": str}), operator, technology)
    if real_cells.empty and config_name != "demo":
        raise ValueError(f"No carriers for operator={operator!r} technology={technology!r}")
    measurements = pd.read_csv(DATA_DIR / "measurements.csv")
    measurements = measurements[measurements["group_id"].isin(real_cells["group_id"])]
    if config_name == "real":
        results = detect(real_cells, measurements, settings=settings)
    else:
        test_cells = in_scope(pd.read_csv(DATA_DIR / TEST_CELL_FILES[config_name], dtype={"site_id": str}), operator, technology)
        if test_cells.empty:
            raise ValueError(f"No {config_name} carriers for operator={operator!r} technology={technology!r}")
        if config_name == "demo":
            measurements = pd.read_csv(DATA_DIR / "measurements_demo.csv")
            real_cells = test_cells.assign(azimuth=test_cells["true_azimuth"])
        real_results = detect(real_cells[real_cells["group_id"].isin(test_cells["group_id"])], measurements, settings=settings)
        results = add_ground_truth(detect(test_cells, measurements, settings=settings), test_cells, real_results)
    provenance = reference_provenance()
    controlled = config_name == "demo" and json.loads((DATA_DIR / "demo_metadata.json").read_text()).get("source") == "controlled"
    results["provenance"] = "CONTROLLED_SYNTHETIC" if controlled else provenance["status"]
    if not controlled and provenance["status"] != "USER_CONFIRMED_INDEPENDENT":
        results["verdict"] = "NOT_ENOUGH_DATA"
        results["reason"] = provenance["basis"]
        results["confidence_score"] = 0
        if "outcome" in results:
            results["outcome"] = results.apply(outcome, axis=1)
    scope = "".join(f"_{value}" for value in (operator, technology) if value)
    results.to_csv(DATA_DIR / f"results_{config_name}{scope}.csv", index=False)
    return results


def write_run_summary(results: pd.DataFrame, config_name: str, operator, technology, settings) -> Path:
    info = json.loads((DATA_DIR / "raw_fetch_info.json").read_text())
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_id": info.get("project_id"), "config": config_name, "method": settings.method,
        "operator": operator or "all", "technology": technology or "all",
        "verdict_counts": results["verdict"].value_counts().to_dict(),
        "field_confirmed": False,
    }
    if "pattern_quality" in results:
        summary["pattern_quality_counts"] = results["pattern_quality"].replace("", "none").value_counts().to_dict()
        summary["verdicts_by_pattern_quality"] = {
            quality or "none": group["verdict"].value_counts().to_dict() for quality, group in results.groupby("pattern_quality")
        }
    scope = "".join(f"_{value}" for value in (operator, technology) if value)
    path = DATA_DIR / f"run_summary_{config_name}{scope}.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["real", "synthetic", "demo"], required=True)
    parser.add_argument("--operator", help="e.g. Airtel, JIO, Vi -- production runs one operator + technology at a time")
    parser.add_argument("--technology", help="LTE or NR")
    parser.add_argument("--method", choices=["pattern", "combined", "direction", "ranking", "legacy"], default="pattern")
    parser.add_argument("--max-violation-db", type=float, default=3.0, help="Pattern model: largest acceptable mean violation")
    parser.add_argument("--tolerance-deg", type=float, default=30.0, help="Direction models only; not a field-calibrated threshold")
    args = parser.parse_args()

    from tests.swap_sector.evidence import Settings
    try:
        settings = Settings(method=args.method, tolerance_deg=args.tolerance_deg, max_violation_db=args.max_violation_db)
    except ValueError as exc:
        parser.error(str(exc))
    df = run(args.config, args.operator, args.technology, settings)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 140)
    print(df["verdict"].value_counts().to_string())
    if "pattern_quality" in df:
        print(pd.crosstab(df["pattern_quality"].replace("", "none"), df["verdict"]).to_string())
    shown = df if args.config == "demo" else df[df["verdict"].isin(["PROBABLE_SWAP", "CONFIRMED_SWAP", "AZIMUTH_MISMATCH", "AMBIGUOUS", "SWAP_SUSPECTED"])]
    columns = ["site_id", "operator", "technology", "band", "earfcn", "n_sectors", "predicted_azimuths", "best_azimuths",
               "pattern_quality", "original_violation_db", "best_violation_db", "support", "verdict", "reason"]
    columns += ["ground_truth", "outcome"] if args.config == "demo" else []
    print(shown[[c for c in columns if c in shown.columns]].to_string(index=False))
    if args.config == "synthetic":
        print(scorecard(df).to_string(index=False))
    print(f"[summary] {write_run_summary(df, args.config, args.operator, args.technology, settings)}")
