"""
Sector-swap detector -- TEST CASE ONLY. Reads the files already produced by
fetch_data.py / make_synthetic_swap.py in tests/swap_sector/data/, does not
touch the database or write anything back to production tables.

Algorithm (per site), per the spec agreed in this test case:

  DT samples  -> quality-filtered directional PCI profile -> DT observed azimuth per PCI
  HO events   -> quality-filtered directional PCI profile -> HO observed azimuth per PCI
  DT + HO fusion (HO only weighted up when it has enough evidence on its own;
    sparse HO never overrides strong DT) -> final observed azimuth per PCI
  compare final observed azimuth against each sector's CONFIGURED azimuth
  test every PCI<->sector permutation for the site (not per-sector deltas),
    weighting each sector's error term by that PCI's evidence confidence
    (an implausibility penalty: a permutation that fights strong evidence
    costs more than one that fights weak evidence)
  confidence = evidence completeness x margin between best and 2nd-best permutation
  verdict = NORMAL / PROBABLE_SWAP / CONFIRMED_SWAP / INSUFFICIENT_DATA

Run from the ML/ directory:
    venv\\Scripts\\python.exe -m tests.swap_sector.detect_sector_swap --config real
    venv\\Scripts\\python.exe -m tests.swap_sector.detect_sector_swap --config synthetic
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

BIN_SIZE_DEG = 10
MIN_BIN_SAMPLES = 15          # a DT angle-bin needs this many total samples to count at all
MIN_BIN_DOMINANCE = 0.40      # top PCI (among the site's own sectors) must hold >=40% share -- meaningful
                               # plurality for up to 3 overlapping sectors, not an unreachable 50%+ majority
MIN_HO_EVENTS_FOR_PCI = 3     # a PCI needs at least this many HO-derived bearings to get an HO azimuth
MIN_EVIDENCE_COMPLETENESS = 0.66  # below this fraction of sectors with usable evidence -> INSUFFICIENT_DATA
CONFIRMED_CONFIDENCE = 0.70
PROBABLE_CONFIDENCE = 0.40


def circular_weighted_mean_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    rad = np.radians(angles_deg)
    x = np.sum(weights * np.cos(rad))
    y = np.sum(weights * np.sin(rad))
    if x == 0 and y == 0:
        return float("nan")
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def circular_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def dt_observed_azimuth(dt_df: pd.DataFrame, site_id: str, sector_pcis: set) -> dict:
    """Per PCI: (azimuth_deg, confidence 0-1, n_samples) from quality-filtered DT bins only."""
    site_dt = dt_df[dt_df["site_id_inferred"].astype(str) == str(site_id)]
    if site_dt.empty:
        return {}

    # Dominance is decided among this site's OWN configured PCIs only, not
    # against every PCI observed within the radius -- the radius also
    # catches nearby unrelated sites' traffic, which would otherwise dilute
    # "dominance" against signals that have nothing to do with this site's
    # own 3 sectors. A bin "belongs" to whichever of the site's own PCIs
    # has the most samples in it (plurality, not a fixed >=50% share --
    # with 3 overlapping sectors no single one reliably clears 50% even in
    # a bin it genuinely dominates), as long as the bin has enough total
    # own-PCI samples to be trustworthy at all.
    own_pci_dt = site_dt[site_dt["pci"].isin(sector_pcis)]
    if own_pci_dt.empty:
        return {}
    per_bin_pci_counts = own_pci_dt.groupby(["angle_bin_10deg", "pci"]).size()
    bin_totals = own_pci_dt.groupby("angle_bin_10deg").size()

    bin_winner: dict[float, tuple] = {}
    for (angle_bin, pci), count in per_bin_pci_counts.items():
        if bin_totals.get(angle_bin, 0) < MIN_BIN_SAMPLES:
            continue
        share = count / bin_totals[angle_bin]
        if share < MIN_BIN_DOMINANCE:
            continue
        current = bin_winner.get(angle_bin)
        if current is None or count > current[1]:
            bin_winner[angle_bin] = (pci, count)

    per_pci_bins: dict = {}
    for angle_bin, (pci, count) in bin_winner.items():
        per_pci_bins.setdefault(pci, {"bins": [], "weights": []})
        per_pci_bins[pci]["bins"].append(angle_bin + BIN_SIZE_DEG / 2.0)
        per_pci_bins[pci]["weights"].append(count)

    result: dict[float, tuple[float, float, int]] = {}
    for pci, data in per_pci_bins.items():
        weights = np.array(data["weights"], dtype=float)
        az = circular_weighted_mean_deg(np.array(data["bins"], dtype=float), weights)
        n_samples = int(weights.sum())
        confidence = min(1.0, n_samples / 200.0)  # saturates at 200 dominant-bin samples
        result[pci] = (az, confidence, n_samples)
    return result


def ho_observed_azimuth(ho_df: pd.DataFrame, site_id: str, site_lat: float, site_lon: float) -> dict:
    """Per PCI: (azimuth_deg, confidence 0-1, n_events) from HO transition locations
    where this site was the FROM side (serving that PCI right before the transition)
    or the TO side (serving that PCI right after)."""
    if ho_df.empty:
        return {}

    def bearing(lat, lon):
        phi1, phi2 = math.radians(site_lat), math.radians(lat)
        dlon = math.radians(lon - site_lon)
        y = math.sin(dlon) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    points: dict[float, list] = {}
    from_mask = ho_df["from_site_id_inferred"].astype(str) == str(site_id)
    to_mask = ho_df["to_site_id_inferred"].astype(str) == str(site_id)

    for _, row in ho_df[from_mask].iterrows():
        pci = row["from_pci"]
        b = bearing(row["event_lat"], row["event_lon"])
        points.setdefault(pci, []).append(b)
    for _, row in ho_df[to_mask].iterrows():
        pci = row["to_pci"]
        b = bearing(row["event_lat"], row["event_lon"])
        points.setdefault(pci, []).append(b)

    result = {}
    for pci, bearings in points.items():
        if len(bearings) < MIN_HO_EVENTS_FOR_PCI:
            continue
        arr = np.array(bearings, dtype=float)
        az = circular_weighted_mean_deg(arr, np.ones_like(arr))
        confidence = min(1.0, len(bearings) / 15.0)  # saturates at 15 HO points
        result[pci] = (az, confidence, len(bearings))
    return result


def fuse_observations(dt_obs: dict, ho_obs: dict) -> dict:
    """HO only outweighs DT for a given PCI when HO itself has enough evidence
    (per feedback: sparse HO must never override strong DT); otherwise DT-only."""
    fused = {}
    all_pcis = set(dt_obs) | set(ho_obs)
    for pci in all_pcis:
        dt = dt_obs.get(pci)
        ho = ho_obs.get(pci)
        if ho and dt:
            w_ho, w_dt = 0.6 * ho[1], 0.4 * dt[1]
            total_w = w_ho + w_dt
            az = circular_weighted_mean_deg(np.array([ho[0], dt[0]]), np.array([w_ho, w_dt]))
            confidence = max(ho[1], dt[1])
            n = ho[2] + dt[2]
        elif ho:
            az, confidence, n = ho
        elif dt:
            az, confidence, n = dt
        else:
            continue
        fused[pci] = (az, confidence, n)
    return fused


def permutation_test(sectors: list, fused_obs: dict) -> dict:
    """sectors: list of dicts {cell_id, azimuth, configured_pci}.
    Tries every bijection of the site's configured PCIs onto its sector
    azimuth slots; error term per sector is weighted by that PCI's evidence
    confidence (implausibility penalty: contradicting strong evidence costs
    more than contradicting weak/absent evidence)."""
    pcis = [s["configured_pci"] for s in sectors]
    azimuths = [s["azimuth"] for s in sectors]
    identity = tuple(pcis)

    results = []
    for perm in set(itertools.permutations(pcis)):
        total_error, total_weight, covered = 0.0, 0.0, 0
        for az, pci in zip(azimuths, perm):
            if pci not in fused_obs:
                continue
            obs_az, confidence, _n = fused_obs[pci]
            err = circular_diff_deg(az, obs_az)
            weight = max(confidence, 0.05)
            total_error += err * weight
            total_weight += weight
            covered += 1
        avg_error = (total_error / total_weight) if total_weight > 0 else float("inf")
        results.append({"perm": perm, "avg_error": avg_error, "covered": covered})

    results.sort(key=lambda r: r["avg_error"])
    best = results[0]
    second = results[1] if len(results) > 1 else None
    identity_result = next(r for r in results if r["perm"] == identity)

    evidence_completeness = sum(1 for pci in pcis if pci in fused_obs) / len(pcis)
    if second and best["avg_error"] < float("inf") and second["avg_error"] < float("inf"):
        denom = max(second["avg_error"], 1e-6)
        margin = max(0.0, (second["avg_error"] - best["avg_error"]) / denom)
    else:
        margin = 0.0
    confidence_score = round(evidence_completeness * margin, 3)

    if evidence_completeness < MIN_EVIDENCE_COMPLETENESS:
        verdict = "INSUFFICIENT_DATA"
    elif best["perm"] == identity:
        verdict = "NORMAL"
    elif confidence_score >= CONFIRMED_CONFIDENCE:
        verdict = "CONFIRMED_SWAP"
    elif confidence_score >= PROBABLE_CONFIDENCE:
        verdict = "PROBABLE_SWAP"
    else:
        verdict = "INSUFFICIENT_DATA"

    return {
        "identity_perm": identity,
        "best_perm": best["perm"],
        "best_avg_error_deg": round(best["avg_error"], 2) if best["avg_error"] != float("inf") else None,
        "identity_avg_error_deg": round(identity_result["avg_error"], 2) if identity_result["avg_error"] != float("inf") else None,
        "evidence_completeness": round(evidence_completeness, 2),
        "confidence_score": confidence_score,
        "verdict": verdict,
    }


def run(config_name: str) -> pd.DataFrame:
    config_file = "site_config.csv" if config_name == "real" else "synthetic_swap_site_config.csv"
    site_df = pd.read_csv(DATA_DIR / config_file)
    dt_df = pd.read_csv(DATA_DIR / "dt_samples_6sites.csv")
    ho_df = pd.read_csv(DATA_DIR / "ho_events_6sites.csv")

    rows = []
    for site_id, group in site_df.groupby("site_id_inferred"):
        site_id = str(site_id)
        group = group.reset_index(drop=True)
        sectors = [
            {
                "cell_id": r["site_cell_id_representative"],
                "azimuth": float(r["site_azimuth_deg"]),
                "configured_pci": float(r["site_pci"]),
                "ground_truth_swapped": bool(r.get("ground_truth_swapped", False)),
            }
            for _, r in group.iterrows()
        ]
        sector_pcis = {s["configured_pci"] for s in sectors}
        site_lat, site_lon = group["site_lat"].mean(), group["site_lon"].mean()

        if len(sectors) < 2:
            rows.append({
                "site_id": site_id, "n_sectors": len(sectors),
                "verdict": "INSUFFICIENT_DATA (config incomplete, <2 sectors)",
                "confidence_score": None, "best_perm": None, "identity_perm": None,
                "swapped_sectors_predicted": None,
                "ground_truth_has_swap": any(s["ground_truth_swapped"] for s in sectors),
            })
            continue

        dt_obs = dt_observed_azimuth(dt_df, site_id, sector_pcis)
        ho_obs = ho_observed_azimuth(ho_df, site_id, site_lat, site_lon)
        fused = fuse_observations(dt_obs, ho_obs)

        result = permutation_test(sectors, fused)

        predicted_swaps = []
        if result["best_perm"] != result["identity_perm"]:
            for cell, configured, predicted in zip([s["cell_id"] for s in sectors], result["identity_perm"], result["best_perm"]):
                if configured != predicted:
                    predicted_swaps.append(f"{cell}: configured_PCI={configured}->best_fit_PCI={predicted}")

        rows.append({
            "site_id": site_id,
            "n_sectors": len(sectors),
            "verdict": result["verdict"],
            "confidence_score": result["confidence_score"],
            "evidence_completeness": result["evidence_completeness"],
            "identity_avg_error_deg": result["identity_avg_error_deg"],
            "best_avg_error_deg": result["best_avg_error_deg"],
            "identity_perm": result["identity_perm"],
            "best_perm": result["best_perm"],
            "swapped_sectors_predicted": "; ".join(predicted_swaps) if predicted_swaps else None,
            "ground_truth_has_swap": any(s["ground_truth_swapped"] for s in sectors),
        })

    out_df = pd.DataFrame(rows)
    out_path = DATA_DIR / f"detection_results_{config_name}.csv"
    out_df.to_csv(out_path, index=False)
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["real", "synthetic"], required=True)
    args = parser.parse_args()

    df = run(args.config)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(df.to_string(index=False))
