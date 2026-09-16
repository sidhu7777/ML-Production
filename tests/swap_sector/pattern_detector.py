"""
Antenna-pattern sector-swap detector -- the active model of this test case. No database access.

Hypothesis tested: the cells (PCIs) of one carrier are connected to the wrong antenna endpoints.
An endpoint = configured azimuth + antenna pattern (+ its tilt); it moves as a whole. PCI numbers
never change.

Expected side (no drive-test RF values): antenna_profile.expected_bins of every configured endpoint
-> 36 x 10-degree relative antenna gain, with its pattern quality EXACT / ASSUMED / APPROXIMATE.

Observed side (drive test), compared at the SAME location so that distance, propagation and the
common power level cancel:
  pair     two PCIs measured at one location and one is >= pair_min_diff_db stronger
  serving  the phone was served by PCI s and did not report group PCI w at all (s should not be weaker)
Each comparison is placed in its 10-degree bin. Under a mapping of endpoints to PCIs, a comparison is
violated when the expected gains in that bin put the weaker PCI ahead by more than
comparison_tolerance_db (hysteresis / pattern error); the violation is that excess in dB.
Why not compare each PCI's profile shape on its own: in real drive tests a PCI is mostly measured only
where it serves; removing its own power offset there makes a wrong antenna (a flat -24 dB) look as good
as the right one. The per-PCI profile shape is kept only as a displayed diagnostic.

Every distinct assignment is scored by its mean violation (<= 5 sectors: all permutations; 6-7 sectors:
pair swaps and 3-rotations). Assignments that only move endpoints by < MIN_SWAP_ANGLE_DEG onto the same
pattern are the same answer. Support = share of bootstrap resamples (over locations) where that answer
strictly wins.

Decision, in this order:
  NOT_TESTABLE      NR, not 2..max_sectors unique PCIs, or a missing / invalid azimuth, tilt or frequency
  NOT_ENOUGH_DATA   too few comparison locations, 10-degree bins, or comparisons for some sector
  AZIMUTH_MISMATCH  no mapping explains which sector was stronger; or the configured mapping does but one
                    sector does not fit its own endpoint; or an exchange does not improve every moved
                    sector -> a direction / RF anomaly, never a swap
  AMBIGUOUS         the winning answer is not stable under resampling or on alternating 10-degree bins,
                    the improvement is too small, or handover locations contradict the exchange
  NORMAL            the configured mapping explains the measurements best
  PROBABLE_SWAP     a different mapping moving >= 2 sectors explains them clearly and stably better
  CONFIRMED_SWAP    PROBABLE_SWAP + every sector has an EXACT pattern with its own electrical tilt, the
                    sectors are eNodeB-ID matched, and handover locations support the exchange.
                    Rule-based; only a field check truly confirms. ASSUMED / APPROXIMATE patterns cap
                    the verdict at PROBABLE_SWAP.
"""
from __future__ import annotations

import itertools
import json
import zlib

import numpy as np
import pandas as pd

from tests.swap_sector.antenna_profile import QUALITY_ORDER, NotTestable, expected_bins, generic_pattern, select_pattern

BIN_COUNT = 36
RESIDUAL_CAP_DB = 20.0
MIN_SWAP_ANGLE_DEG = 30.0
FULL_PERMUTATION_SECTORS = 5
BOOTSTRAP_ROUNDS = 200
RANDOM_SEED = 7
MIN_HANDOVERS = 5
HO_SUPPORT_DB = 3.0          # handover spots must sit this much closer to the beams under the exchange
HO_CONTRADICTION_DB = 6.0
MAX_TILT_ERROR_FOR_CONFIRMATION_DEG = 1.0
QUALITY_SCORE_CAP = {"EXACT": 100, "ASSUMED": 75, "APPROXIMATE": 50}
SWAP_VERDICTS = ("PROBABLE_SWAP", "CONFIRMED_SWAP")


def angle_gap(a, b) -> np.ndarray:
    return np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0)


def same_mapping(first, second, azimuths: np.ndarray, keys: list[str]) -> bool:
    return all(angle_gap(azimuths[a], azimuths[b]) < MIN_SWAP_ANGLE_DEG and keys[a] == keys[b] for a, b in zip(first, second))


def distinct_assignments(azimuths: np.ndarray, keys: list[str]) -> list[tuple[int, ...]]:
    """perm[i] = endpoint given to PCI i. Identity first; answers equal to the identity or to an earlier
    exact signature are dropped."""
    n = len(azimuths)
    identity = tuple(range(n))
    if n <= FULL_PERMUTATION_SECTORS:
        options = itertools.permutations(identity)
    else:
        options = []
        for a, b in itertools.combinations(identity, 2):
            perm = list(identity)
            perm[a], perm[b] = b, a
            options.append(tuple(perm))
        for a, b, c in itertools.combinations(identity, 3):
            for x, y, z in ((b, c, a), (c, a, b)):
                perm = list(identity)
                perm[a], perm[b], perm[c] = x, y, z
                options.append(tuple(perm))
    result, signatures = [identity], {tuple((round(float(azimuths[k]), 1), keys[k]) for k in identity)}
    for perm in options:
        signature = tuple((round(float(azimuths[k]), 1), keys[k]) for k in perm)
        if signature in signatures or same_mapping(perm, identity, azimuths, keys):
            continue
        signatures.add(signature)
        result.append(tuple(perm))
    return result


def comparison_checks(measurements: pd.DataFrame, pcis: list[int], pair_min_diff_db: float) -> dict:
    """Same-location comparisons: arrays of location index, 10-degree bin, stronger and weaker sector."""
    empty = {"loc": np.zeros(0, int), "bin": np.zeros(0, int), "strong": np.zeros(0, int), "weak": np.zeros(0, int),
             "is_serving": np.zeros(0, bool), "n_locations": 0}
    m = measurements[measurements["pci"].isin(pcis)].dropna(subset=["rsrp", "bearing_deg"]) if not measurements.empty else measurements
    if m.empty:
        return empty
    spots = m.groupby(["location_id", "pci"], as_index=False).agg(
        rsrp=("rsrp", "median"), bearing_deg=("bearing_deg", "first"), is_serving=("is_serving", "max"))
    rsrp = spots.pivot(index="location_id", columns="pci", values="rsrp").reindex(columns=pcis)
    serving = spots.pivot(index="location_id", columns="pci", values="is_serving").reindex(index=rsrp.index, columns=pcis)
    bearing = spots.groupby("location_id")["bearing_deg"].first().reindex(rsrp.index).to_numpy(dtype=float)
    values = rsrp.to_numpy(dtype=float)
    measured = ~np.isnan(values)
    served = serving.fillna(False).astype(bool).to_numpy()
    location_bin = (bearing % 360 // 10).astype(int)
    loc, strong, weak, is_serving = [], [], [], []

    def add(rows: np.ndarray, s: int, w: int, serving_check: bool) -> None:
        loc.append(rows)
        strong.append(np.full(len(rows), s))
        weak.append(np.full(len(rows), w))
        is_serving.append(np.full(len(rows), serving_check))

    n = len(pcis)
    for i, j in itertools.combinations(range(n), 2):
        both = measured[:, i] & measured[:, j]
        diff = np.where(both, np.nan_to_num(values[:, i]) - np.nan_to_num(values[:, j]), 0.0)
        add(np.flatnonzero(both & (diff >= pair_min_diff_db)), i, j, False)
        add(np.flatnonzero(both & (diff <= -pair_min_diff_db)), j, i, False)
    for s, w in itertools.permutations(range(n), 2):
        add(np.flatnonzero(served[:, s] & ~measured[:, w]), s, w, True)
    loc_all = np.concatenate(loc).astype(int)
    return {"loc": loc_all, "bin": location_bin[loc_all], "strong": np.concatenate(strong).astype(int),
            "weak": np.concatenate(weak).astype(int), "is_serving": np.concatenate(is_serving).astype(bool),
            "n_locations": len(values)}


def violation_matrix(expected: np.ndarray, candidates, checks: dict, tolerance_db: float) -> np.ndarray:
    """violation[c, k] (dB) of comparison k under candidate mapping c."""
    perms = np.array(candidates)
    gain_strong = expected[perms[:, checks["strong"]], checks["bin"][None, :]]
    gain_weak = expected[perms[:, checks["weak"]], checks["bin"][None, :]]
    return np.maximum(0.0, -(gain_strong - gain_weak) - tolerance_db)


def sector_violation(violations: np.ndarray, checks: dict, n: int) -> np.ndarray:
    """Mean violation of the comparisons each sector takes part in (NaN when none)."""
    result = np.full(n, np.nan)
    for k in range(n):
        involved = (checks["strong"] == k) | (checks["weak"] == k)
        if involved.any():
            result[k] = float(violations[involved].mean())
    return result


def winning_answer(violations: np.ndarray, candidates, azimuths, keys) -> int:
    return int(np.argmin(violations.mean(axis=1)))


def bootstrap_support(violations: np.ndarray, checks: dict, candidates, best: int, azimuths, keys, seed_key: str) -> float:
    n_loc = checks["n_locations"]
    per_location = np.bincount(checks["loc"], minlength=n_loc).astype(float)
    loc_violation = np.vstack([np.bincount(checks["loc"], weights=row, minlength=n_loc) for row in violations])
    same = np.array([same_mapping(p, candidates[best], azimuths, keys) for p in candidates])
    if same.all():
        return 1.0
    rng = np.random.default_rng([RANDOM_SEED, zlib.crc32(seed_key.encode())])
    weights = rng.poisson(1.0, size=(BOOTSTRAP_ROUNDS, n_loc))
    scores = (weights @ loc_violation.T) / np.maximum(weights @ per_location, 1.0)[:, None]
    return float(np.mean(scores[:, same].min(axis=1) < scores[:, ~same].min(axis=1) - 1e-9))


def observed_bins(measurements: pd.DataFrame, pci: int, exponent: float, min_locations: int) -> tuple[np.ndarray, np.ndarray]:
    """Display only: median distance-corrected RSRP per 10-degree bin for one PCI (NaN = too few locations)."""
    m = measurements[measurements["pci"].eq(pci)].replace([np.inf, -np.inf], np.nan) if not measurements.empty else measurements
    if m.empty:
        return np.full(BIN_COUNT, np.nan), np.zeros(BIN_COUNT, dtype=int)
    m = m.dropna(subset=["bearing_deg", "distance_m", "rsrp"])
    m = m[m["distance_m"].between(30, 1000) & m["rsrp"].between(-150, -30)]
    if m.empty:
        return np.full(BIN_COUNT, np.nan), np.zeros(BIN_COUNT, dtype=int)
    spots = m.groupby("location_id", as_index=False).agg(
        bearing_deg=("bearing_deg", "first"), distance_m=("distance_m", "median"), rsrp=("rsrp", "median"))
    spots["bin"] = (spots["bearing_deg"] % 360 // 10).astype(int)
    spots["corrected"] = spots["rsrp"] + 10.0 * exponent * np.log10(spots["distance_m"] / 100.0)
    per_bin = spots.groupby("bin").agg(value=("corrected", "median"), count=("corrected", "size")).reindex(range(BIN_COUNT))
    counts = per_bin["count"].fillna(0).astype(int).to_numpy()
    values = per_bin["value"].to_numpy(dtype=float)
    values[counts < min_locations] = np.nan
    return values, counts


def shape_error(observed: np.ndarray, gain: np.ndarray) -> float:
    """Display only: RMS residual of one PCI's profile against one endpoint after removing a constant offset."""
    valid = np.isfinite(observed)
    if valid.sum() < 3:
        return float("nan")
    residual = observed[valid] - gain[valid]
    residual = residual - np.median(residual)
    return float(np.sqrt(np.mean(np.minimum(residual ** 2, RESIDUAL_CAP_DB ** 2))))


def handover_losses(handovers, pcis: list[int], expected: np.ndarray, candidates) -> tuple[np.ndarray, int]:
    """Mean dB loss per candidate. Leaving / entering spots should lie inside their sector's beam;
    a boundary spot lies where the two sectors' beams are equally strong."""
    if handovers is None or handovers.empty:
        return np.full(len(candidates), np.nan), 0
    index = {pci: k for k, pci in enumerate(pcis)}
    relative = expected - expected.max(axis=1, keepdims=True)
    perms = np.array(candidates)
    losses = []
    for event in handovers.itertuples():
        if event.pci not in index or not np.isfinite(event.bearing_deg):
            continue
        b, i = int(event.bearing_deg % 360 // 10), index[event.pci]
        if event.kind == "boundary":
            if event.other_pci not in index:
                continue
            losses.append(np.abs(relative[perms[:, i], b] - relative[perms[:, index[event.other_pci]], b]))
        elif event.kind in ("leaving", "entering"):
            losses.append(-relative[perms[:, i], b])
    if not losses:
        return np.full(len(candidates), np.nan), 0
    return np.mean(losses, axis=0), len(losses)


def evidence_score(coverage: float, best_violation: float, limit: float, support: float, quality: str) -> int:
    """Uncalibrated bottleneck rule (not a probability), capped by pattern quality."""
    raw = min(coverage, max(0.0, 1.0 - best_violation / limit), support)
    return int(round(min(100.0 * raw, QUALITY_SCORE_CAP[quality])))


def analyse(group_cells: pd.DataFrame, group_meas: pd.DataFrame, handovers=None, settings=None) -> dict:
    from tests.swap_sector.evidence import Settings
    settings = settings or Settings(method="pattern")
    cells = group_cells.sort_values("pci").reset_index(drop=True)
    first = cells.iloc[0]
    n = len(cells)
    pcis = cells["pci"].astype(int).tolist()
    azimuths = pd.to_numeric(cells["azimuth"], errors="coerce").to_numpy(dtype=float)
    fmt = lambda values: ";".join(f"{float(v):.1f}" for v in values)
    out = {
        "group_id": first["group_id"], "operator": first["operator"], "site_id": str(first["site_id"]),
        "technology": first["technology"], "band": first.get("band", ""), "earfcn": int(first["earfcn"]),
        "match_level": first.get("match_level", ""), "n_sectors": n, "pcis": ";".join(map(str, pcis)),
        "predicted_azimuths": fmt(azimuths), "best_azimuths": fmt(azimuths), "changed_pcis": "",
        "frequency_mhz": first.get("frequency_mhz", np.nan), "pattern_quality": "", "pattern_note": "",
        "locations": int(group_meas["location_id"].nunique()) if not group_meas.empty else 0,
        "comparison_locations": 0, "comparisons": 0, "serving_comparisons": 0, "coverage_fraction": 0.0,
        "original_violation_db": np.nan, "best_violation_db": np.nan, "violation_improvement_db": np.nan, "support": np.nan,
        "original_profile_error_db": np.nan, "best_profile_error_db": np.nan,
        "ho_points": 0, "ho_original_loss_db": np.nan, "ho_best_loss_db": np.nan,
        "confidence_score": 0, "confidence_basis": "Uncalibrated rule, capped by pattern quality",
        "pattern_details": "[]", "sector_fit_details": "[]", "profile_details": "[]",
        "method": "pattern", "max_violation_db": settings.max_violation_db,
    }

    def finish(verdict: str, reason: str) -> dict:
        out.update(verdict=verdict, reason=reason)
        return out

    if first["technology"] != "LTE":
        return finish("NOT_TESTABLE", "NR cell identity in the drive test is not validated, so NR sectors are not testable")
    if n < 2 or n > settings.max_sectors or len(set(pcis)) != n:
        return finish("NOT_TESTABLE", f"Need 2-{settings.max_sectors} uniquely identified sectors on one carrier")

    patterns, problems = [], []
    for row in cells.to_dict("records"):
        try:
            patterns.append(select_pattern(row))
        except NotTestable as exc:
            problems.append(f"PCI {int(row['pci'])}: {exc}")
    if problems:
        return finish("NOT_TESTABLE", "; ".join(problems))
    if min(QUALITY_ORDER[p["pattern_quality"]] for p in patterns) == QUALITY_ORDER["APPROXIMATE"] and any(p["path"] for p in patterns):
        patterns = [generic_pattern(p, "mixed pattern sources on this carrier: every sector compared with the generic 3GPP pattern")
                    for p in patterns]
    quality = min((p["pattern_quality"] for p in patterns), key=QUALITY_ORDER.get)
    keys = [f"{p['path']}|{p['file_e_tilt']}" for p in patterns]
    expected = np.array([expected_bins(p) for p in patterns])
    detail_keys = ("pattern_quality", "model_source", "antenna_model", "pattern_port", "file_e_tilt",
                   "requested_e_tilt", "tilt_error_deg", "frequency_mhz", "pattern_range_mhz", "note")
    out.update(pattern_quality=quality, pattern_note="; ".join(sorted({p["note"] for p in patterns if p["note"]})),
               pattern_details=json.dumps([{"pci": pci, **{k: p[k] for k in detail_keys}} for pci, p in zip(pcis, patterns)]))

    m = group_meas[group_meas["pci"].isin(pcis)] if not group_meas.empty else group_meas
    observed = [observed_bins(m, pci, settings.path_loss_exponent, settings.min_bin_locations) for pci in pcis]
    profiles = [{"pci": pci, "bin_deg": b * 10, "expected_gain_db": round(float(expected[i, b]), 2),
                 "observed_db": round(float(observed[i][0][b]), 2) if np.isfinite(observed[i][0][b]) else None,
                 "locations": int(observed[i][1][b])}
                for i, pci in enumerate(pcis) for b in range(BIN_COUNT)]
    out["profile_details"] = json.dumps(profiles)

    checks = comparison_checks(m, pcis, settings.pair_min_diff_db)
    check_locations = int(len(np.unique(checks["loc"])))
    check_bins = int(len(np.unique(checks["bin"])))
    per_sector_checks = [int(((checks["strong"] == k) | (checks["weak"] == k)).sum()) for k in range(n)]
    out.update(comparison_locations=check_locations, comparisons=int(len(checks["loc"])),
               serving_comparisons=int(checks["is_serving"].sum()), coverage_fraction=round(check_bins / BIN_COUNT, 3))
    if (check_locations < settings.min_check_locations or check_bins < settings.min_check_bins
            or min(per_sector_checks) < settings.min_sector_checks):
        thin = [str(pcis[k]) for k in range(n) if per_sector_checks[k] < settings.min_sector_checks]
        return finish("NOT_ENOUGH_DATA", f"{check_locations} comparison locations in {check_bins} 10-degree bins "
                                         f"(need {settings.min_check_locations} and {settings.min_check_bins})"
                                         + (f"; PCI {', '.join(thin)} in fewer than {settings.min_sector_checks} comparisons" if thin else ""))

    candidates = distinct_assignments(azimuths, keys)
    violations = violation_matrix(expected, candidates, checks, settings.comparison_tolerance_db)
    scores = violations.mean(axis=1)
    best_index = int(np.argmin(scores))
    best = candidates[best_index]
    improvement = float(scores[0] - scores[best_index])
    support = bootstrap_support(violations, checks, candidates, best_index, azimuths, keys, str(first["group_id"]))
    original_sector = sector_violation(violations[0], checks, n)
    best_sector = sector_violation(violations[best_index], checks, n)
    moved = [k for k in range(n) if angle_gap(azimuths[k], azimuths[best[k]]) >= MIN_SWAP_ANGLE_DEG or keys[k] != keys[best[k]]]
    for row in profiles:
        row["best_gain_db"] = round(float(expected[best[pcis.index(row["pci"])], row["bin_deg"] // 10]), 2)
    shape_original = [shape_error(observed[k][0], expected[k]) for k in range(n)]
    shape_best = [shape_error(observed[k][0], expected[best[k]]) for k in range(n)]
    out.update(
        best_azimuths=fmt(azimuths[list(best)]),
        original_violation_db=round(float(scores[0]), 2), best_violation_db=round(float(scores[best_index]), 2),
        violation_improvement_db=round(improvement, 2), support=round(support, 3),
        original_profile_error_db=round(float(np.nanmean(shape_original)), 2) if np.isfinite(shape_original).any() else np.nan,
        best_profile_error_db=round(float(np.nanmean(shape_best)), 2) if np.isfinite(shape_best).any() else np.nan,
        profile_details=json.dumps(profiles),
        sector_fit_details=json.dumps([{
            "pci": pci, "configured_azimuth": float(azimuths[k]), "pattern_quality": patterns[k]["pattern_quality"],
            "antenna_model": patterns[k]["antenna_model"], "file_e_tilt": patterns[k]["file_e_tilt"],
            "violation_configured_db": round(float(original_sector[k]), 2), "best_endpoint_azimuth": float(azimuths[best[k]]),
            "violation_best_db": round(float(best_sector[k]), 2), "comparisons": per_sector_checks[k],
            "profile_error_configured_db": round(shape_original[k], 2), "profile_error_best_db": round(shape_best[k], 2),
            "measured_bins": int(np.isfinite(observed[k][0]).sum()), "locations": int(observed[k][1].sum()),
        } for k, pci in enumerate(pcis)]),
    )
    ho_loss, ho_points = handover_losses(handovers, pcis, expected, candidates)
    if ho_points:
        out.update(ho_points=ho_points, ho_original_loss_db=round(float(ho_loss[0]), 2),
                   ho_best_loss_db=round(float(ho_loss[best_index]), 2))
    limit = settings.max_violation_db

    if scores[best_index] > limit:
        return finish("AZIMUTH_MISMATCH", f"No antenna-endpoint mapping explains which sector the phone measured stronger "
                                          f"(best mapping {scores[best_index]:.1f} dB > {limit:g} dB)")
    if best_index == 0:
        poor = [pcis[k] for k in range(n) if original_sector[k] > limit]
        if poor:
            return finish("AZIMUTH_MISMATCH", f"PCI {', '.join(map(str, poor))} does not match its own configured antenna "
                                              f"direction and no exchange explains it: a direction / RF anomaly, not a swap")
        if support < settings.min_support:
            return finish("AMBIGUOUS", f"Configured mapping fits best, but only in {support:.0%} of resampled drive tests")
        out["confidence_score"] = evidence_score(out["coverage_fraction"], float(scores[0]), limit, support, quality)
        return finish("NORMAL", f"Configured antenna mapping explains the drive test best ({scores[0]:.1f} dB mean violation, "
                                f"stable in {support:.0%} of resamples). Pattern: {quality}")

    changed = [pcis[k] for k in moved]
    if len(moved) < 2 or improvement < settings.min_violation_improvement_db:
        return finish("AMBIGUOUS", f"No clear exchange of at least two sectors (improvement {improvement:.2f} dB)")
    if support < settings.min_support:
        return finish("AMBIGUOUS", f"The best exchange wins in only {support:.0%} of resampled drive tests")
    if np.any((original_sector - best_sector)[moved] < settings.min_sector_improvement_db):
        return finish("AZIMUTH_MISMATCH", "The best exchange does not improve every moved sector: "
                                          "an isolated direction / RF anomaly, not a swap")
    if np.any(best_sector[moved] > limit):
        return finish("AMBIGUOUS", "At least one exchanged sector still fits its new antenna endpoint poorly")
    for parity in (0, 1):
        half = checks["bin"] % 2 == parity
        if len(np.unique(checks["bin"][half])) < max(2, settings.min_check_bins // 2):
            return finish("NOT_ENOUGH_DATA", "Too few 10-degree bins to cross-check the exchange on alternating bins")
        if not same_mapping(candidates[int(np.argmin(violations[:, half].mean(axis=1)))], best, azimuths, keys):
            return finish("AMBIGUOUS", "The exchange is not the best answer on both sets of alternating 10-degree bins")
    if ho_points >= MIN_HANDOVERS and ho_loss[best_index] - ho_loss[0] > HO_CONTRADICTION_DB:
        return finish("AMBIGUOUS", "Handover locations contradict the proposed exchange")

    ho_support = ho_points >= MIN_HANDOVERS and ho_loss[0] - ho_loss[best_index] >= HO_SUPPORT_DB
    exact = quality == "EXACT" and all(p["tilt_error_deg"] <= MAX_TILT_ERROR_FOR_CONFIRMATION_DEG for p in patterns)
    confirmed = exact and first.get("match_level", "") == "ID" and ho_support
    out.update(changed_pcis=";".join(map(str, changed)),
               confidence_score=evidence_score(out["coverage_fraction"], float(scores[best_index]), limit, support, quality))
    evidence = (f"mean violation {scores[0]:.1f} -> {scores[best_index]:.1f} dB (improvement {improvement:.1f} dB, "
                f"stable in {support:.0%} of resamples); pattern {quality}")
    if confirmed:
        return finish("CONFIRMED_SWAP", f"PCI {', '.join(map(str, changed))} fit exchanged antenna endpoints; {evidence}; "
                                        f"handover locations agree. Rule-based: verify in the field")
    missing = [] if exact else [f"pattern {quality} caps the verdict at Probable"]
    if exact and not ho_support:
        missing.append("no handover support")
    if exact and first.get("match_level", "") != "ID":
        missing.append("sectors not eNodeB-ID matched")
    return finish("PROBABLE_SWAP", f"PCI {', '.join(map(str, changed))} fit exchanged antenna endpoints; {evidence}; "
                                   f"not confirmed: {', '.join(missing)}. Verify in the field")
