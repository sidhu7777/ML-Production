"""Local experimental RF evidence model. Scores are rules, not probabilities.

No database access, ground-truth columns or configured azimuths enter direction estimation.
HO proxies are serving-cell transitions, not protocol-confirmed handover events.
"""
from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Settings:
    tolerance_deg: float = 30.0
    min_improvement_deg: float = 20.0
    min_margin_deg: float = 10.0
    min_rank_improvement: float = 0.15
    min_rank_margin: float = 0.08
    max_rank_loss: float = 0.30
    max_sectors: int = 7
    path_loss_exponent: float = 3.0
    method: str = "combined"
    # Antenna-pattern model (method="pattern", pattern_detector.py). Same-location comparisons against
    # expected antenna gains. Starting values come from RF reasoning (3 dB ~ handover hysteresis), not from
    # detection results; validate.py reports their sensitivity.
    comparison_tolerance_db: float = 3.0     # expected-gain error allowed before a comparison counts as violated
    pair_min_diff_db: float = 3.0            # smaller measured differences are fading, not a ranking
    max_violation_db: float = 3.0            # largest acceptable mean violation of a mapping (one tolerance step)
    min_violation_improvement_db: float = 1.0
    min_sector_improvement_db: float = 0.5
    min_support: float = 0.9                 # bootstrap share where the winning answer strictly wins
    min_check_locations: int = 15
    min_check_bins: int = 6
    min_sector_checks: int = 10
    min_bin_locations: int = 3               # profile chart only

    def __post_init__(self):
        if not 5 <= self.tolerance_deg <= 90 or not 1 <= self.path_loss_exponent <= 6:
            raise ValueError("Tolerance must be 5-90 degrees and path-loss exponent 1-6")
        if self.method not in {"combined", "direction", "ranking", "legacy", "pattern"}:
            raise ValueError("Unknown evidence method")
        if (min(self.comparison_tolerance_db, self.pair_min_diff_db, self.max_violation_db,
                self.min_violation_improvement_db, self.min_sector_improvement_db) <= 0
                or not 0 < self.min_support <= 1 or self.min_check_locations < 1 or self.min_check_bins < 2
                or self.min_sector_checks < 1 or self.min_bin_locations < 1):
            raise ValueError("Pattern-model thresholds must be positive; support within (0, 1]")


def gap(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def mean_angle(a, w=None):
    a = np.radians(a)
    w = np.ones(len(a)) if w is None else np.asarray(w)
    z = np.sum(w * np.exp(1j * a))
    return float(np.degrees(np.angle(z)) % 360)


def main_run(mask):
    """Longest circular connected component; tied disconnected lobes are ambiguous."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.array([], dtype=int)
    if mask.all():
        return np.arange(len(mask))
    starts = np.flatnonzero(mask & ~np.roll(mask, 1))
    runs = []
    for start in starts:
        run = []
        i = int(start)
        while mask[i]:
            run.append(i)
            i = (i + 1) % len(mask)
        runs.append(run)
    runs.sort(key=len, reverse=True)
    if len(runs) > 1 and len(runs[1]) >= 0.8 * len(runs[0]):
        return np.array([], dtype=int)
    return np.asarray(runs[0])


def observed_directions(measurements, pcis, settings):
    """Equal bin weighting prevents repeated roads dominating the estimate.

    Distance correction is an explicit sensitivity parameter, not a fitted truth.
    A directional first harmonic fits a smooth profile; disconnected dominance
    lobes are never averaged. Disagreeing estimators are reported as uncertain.
    """
    from tests.swap_sector.detect_sector_swap import direction_profile
    # A lone measured PCI is not evidence that it dominates the other sectors.
    shared = measurements.groupby("location_id").pci.transform("nunique") >= 2
    profile = direction_profile(measurements[shared], pcis, require_comparison=True)
    details = []
    for pci in pcis:
        run = main_run(profile.dominant_pci.eq(pci))
        dominance = mean_angle(profile.bin_deg.iloc[run].to_numpy() + 5) if len(run) >= 2 else np.nan
        m = measurements[measurements.pci == pci].copy()
        m["bin"] = (m.bearing_deg // 10).astype(int) % 36
        m["corrected"] = m.rsrp + 10 * settings.path_loss_exponent * np.log10(m.distance_m.clip(lower=30) / 100)
        bins = m.groupby("bin").agg(signal=("corrected", "median"), n=("location_id", "nunique"))
        bins = bins[bins.n >= 3]
        signal, residual, amplitude = np.nan, np.nan, 0.0
        if len(bins) >= 6:
            angles = np.radians(bins.index.to_numpy() * 10 + 5)
            design = np.column_stack([np.ones(len(angles)), np.cos(angles), np.sin(angles)])
            if np.linalg.cond(design) < 8:
                coef = np.linalg.lstsq(design, bins.signal.to_numpy(), rcond=None)[0]
                amplitude = float(np.hypot(coef[1], coef[2]))
                residual = float(np.sqrt(np.mean((design @ coef - bins.signal.to_numpy()) ** 2)))
                if amplitude >= 3 and amplitude >= residual:
                    signal = float(np.degrees(np.arctan2(coef[2], coef[1])) % 360)
        disagreement = float(gap(dominance, signal)) if np.isfinite(dominance) and np.isfinite(signal) else np.nan
        direction = dominance if np.isfinite(dominance) else signal
        if np.isfinite(disagreement):
            direction = mean_angle([dominance, signal]) if disagreement <= 45 else np.nan
        # Uncertainty widens the diagnostic band, but uncertain sectors do not qualify for confirmation.
        uncertainty = min(15.0, 30 / np.sqrt(max(len(bins), 1)))
        details.append(dict(pci=pci, dominance_deg=dominance, signal_deg=signal,
                            observed_deg=direction, uncertainty_deg=uncertainty,
                            disagreement_deg=disagreement, main_run_bins=len(run),
                            signal_bins=len(bins), signal_amplitude_db=amplitude,
                            signal_residual_db=residual))
    return pd.DataFrame(details), profile


def ranking_evidence(measurements, pcis):
    """Pairwise RSRP comparisons at shared locations, balanced by angular bin.

    Returns all sufficiently sampled sector pairs; single-PCI spots cannot vote.
    The ranking feature is correlated with direction evidence and is a gate,
    never an additional independent confidence vote.
    """
    pivot = measurements.pivot_table(index="location_id", columns="pci", values="rsrp", aggfunc="mean")
    bearing = measurements.groupby("location_id").bearing_deg.first()
    pairs = []
    for i, j in itertools.combinations(range(len(pcis)), 2):
        if pcis[i] not in pivot or pcis[j] not in pivot:
            continue
        values = pivot[[pcis[i], pcis[j]]].dropna()
        diff = values[pcis[i]] - values[pcis[j]]
        values = pd.DataFrame({"bearing": bearing.reindex(values.index), "diff": diff})
        values = values[values["diff"].abs() >= 3].copy()
        values["bin"] = (values.bearing // 10).astype(int)
        grouped = values.groupby("bin").agg(diff=("diff", "median"), n=("diff", "size"))
        grouped = grouped[grouped.n >= 2]
        if len(grouped) >= 4 and len(values) >= 12:
            pairs.append((i, j, grouped.index.to_numpy() * 10 + 5, grouped["diff"].to_numpy()))
    return pairs


def rank_scores(assignments, pairs):
    if not pairs:
        return np.full(len(assignments), np.nan)
    losses = []
    for i, j, bearings, differences in pairs:
        expected = np.cos(np.radians(bearings[None, :] - assignments[:, i, None])) - np.cos(np.radians(bearings[None, :] - assignments[:, j, None]))
        # Ties have half loss; compare ordering, not absolute RSRP or transmit power.
        losses.append(np.mean(np.where(np.abs(expected) < 0.05, 0.5,
                                      (expected * differences[None, :] < 0).astype(float)), axis=1))
    return np.mean(losses, axis=0)


def signal_rank_scores(assignments, measurements, pcis, settings):
    """Rank each PCI's distance-corrected signal across spatial bins.

    Unlike cross-PCI comparisons, this uses single-PCI locations too. Comparing
    bins within one PCI cancels its constant transmit-power offset. Equal bin
    pairs limit road sampling bias. No configured direction defines the bins.
    """
    losses = []
    supported = []
    for index, pci in enumerate(pcis):
        m = measurements[measurements.pci.eq(pci)].copy()
        m["bin"] = (m.bearing_deg // 10).astype(int)
        m["signal"] = m.rsrp + 10 * settings.path_loss_exponent * np.log10(m.distance_m.clip(lower=30) / 100)
        bins = m.groupby("bin").agg(signal=("signal", "median"), n=("location_id", "nunique"))
        bins = bins[bins.n >= 3]
        if len(bins) < 5:
            continue
        a, b = np.triu_indices(len(bins), 1)
        diff = bins.signal.to_numpy()[a] - bins.signal.to_numpy()[b]
        keep = np.abs(diff) >= 3
        if keep.sum() < 6:
            continue
        angles = bins.index.to_numpy() * 10 + 5
        gain = np.cos(np.radians(angles[None, :] - assignments[:, index, None]))
        expected = (gain[:, a] - gain[:, b])[:, keep]
        losses.append(np.mean(np.where(np.abs(expected) < .05, .5, (expected * diff[keep] < 0).astype(float)), axis=1))
        supported.append(index)
    return (np.mean(losses, axis=0) if losses else np.full(len(assignments), np.nan)), supported


def ho_scores(assignments, pcis, handovers):
    losses = []
    if handovers is None or handovers.empty:
        return np.full(len(assignments), np.nan), 0
    lookup = {p: i for i, p in enumerate(pcis)}
    used = 0
    for event in handovers.itertuples():
        if event.pci not in lookup:
            continue
        a = gap(assignments[:, lookup[event.pci]], event.bearing_deg)
        if event.kind == "boundary":
            if event.other_pci not in lookup:
                continue
            b = gap(assignments[:, lookup[event.other_pci]], event.bearing_deg)
            # Boundaries constrain relative beam distances, not boresight direction.
            losses.append(np.abs(a - b) / 180)
        elif event.kind in ("leaving", "entering"):
            losses.append(np.maximum(a - 60, 0) / 120)
        else:
            continue
        used += 1
    return (np.mean(losses, axis=0), used) if losses else (np.full(len(assignments), np.nan), 0)


def analyse(group_cells, group_meas, handovers=None, settings=None):
    settings = settings or Settings()
    cells = group_cells.sort_values("pci")
    first = cells.iloc[0]
    pcis = cells.pci.astype(int).tolist()
    az = cells.azimuth.to_numpy(dtype=float)
    n = len(pcis)
    fmt = lambda v: ";".join(f"{x:.1f}" for x in v)
    out = dict(group_id=first.group_id, operator=first.operator, site_id=str(first.site_id),
               technology=first.technology, band=first.band, earfcn=int(first.earfcn),
               match_level=first.match_level, n_sectors=n, pcis=";".join(map(str, pcis)),
               predicted_azimuths=fmt(az), best_azimuths=fmt(az), real_directions="",
               changed_pcis="", bins_won="", sectors_with_direction=0, locations=0,
               largest_gap_deg=np.nan, original_error_deg=np.nan, best_error_deg=np.nan,
               improvement_deg=np.nan, ambiguity_margin_deg=np.nan, rank_original_loss=np.nan,
               rank_best_loss=np.nan, rank_improvement=np.nan, rank_margin=np.nan,
               ho_points=0, ho_original_loss=np.nan, ho_best_loss=np.nan,
               coverage_fraction=0.0, confidence_score=0, confidence_basis="Uncalibrated evidence rule",
               tolerance_deg=settings.tolerance_deg, observed_uncertainty_deg="", evidence_details="",
               provenance="NOT_CHECKED", method=settings.method)

    def finish(verdict, reason):
        out.update(verdict=verdict, reason=reason)
        return out

    if first.technology != "LTE":
        return finish("NOT_TESTABLE", "NR requires independently validated cell and carrier matching")
    if n < 2 or n > settings.max_sectors or len(set(pcis)) != n:
        return finish("NOT_TESTABLE", "Need 2-7 uniquely identified sectors on one carrier")
    if not np.isfinite(az).all():
        return finish("NOT_TESTABLE", "Invalid configured azimuth")
    perms = np.array(list(itertools.permutations(range(n))))
    assignments = az[perms]
    _, unique = np.unique(np.round(assignments, 5), axis=0, return_index=True)
    assignments, perms = assignments[np.sort(unique)], perms[np.sort(unique)]
    if len(assignments) < 2:
        return finish("NOT_TESTABLE", "Configured directions are indistinguishable")
    m = group_meas[group_meas.pci.isin(pcis)].copy()
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["rsrp", "bearing_deg", "distance_m"])
    m = m[m.distance_m.between(30, 1000) & m.rsrp.between(-150, -30)]
    if m.empty:
        return finish("NOT_ENOUGH_DATA", "No usable RF measurements")
    directions, profile = observed_directions(m, pcis, settings)
    observed = directions.observed_deg.to_numpy()
    valid = np.isfinite(observed)
    pairs = ranking_evidence(m, pcis)
    shared_ranks = rank_scores(assignments, pairs)
    signal_ranks, signal_supported = signal_rank_scores(assignments, m, pcis, settings)
    # Require every sector to participate; a disconnected comparison graph is ambiguous.
    connected = {0}
    for _ in pcis:
        for i, j, _, _ in pairs:
            if i in connected or j in connected:
                connected.update([i, j])
    shared_usable = len(connected) == n and bool(pairs)
    signal_usable = len(signal_supported) == n
    rank_usable = shared_usable or signal_usable
    ranks = (signal_ranks + shared_ranks) / 2 if signal_usable and shared_usable else signal_ranks if signal_usable else shared_ranks
    coverage = float(m.assign(bin=(m.bearing_deg // 10).astype(int)).groupby("bin").location_id.nunique().ge(3).sum() / 36)
    errors = np.mean(gap(assignments[:, valid], observed[valid]), axis=1) if valid.any() else np.full(len(assignments), np.nan)
    out.update(real_directions=fmt(observed), observed_uncertainty_deg=fmt(directions.uncertainty_deg),
               sectors_with_direction=int(valid.sum()), locations=int(m.location_id.nunique()),
               bins_won=";".join(str(int(profile.dominant_pci.eq(p).sum())) for p in pcis),
               coverage_fraction=round(coverage, 3), evidence_details=directions.to_json(orient="records"))
    out.update(ranking_source="signal profile + shared locations" if signal_usable and shared_usable else "signal profile" if signal_usable else "shared locations",
               signal_rank_sectors=len(signal_supported), shared_rank_pairs=len(pairs))
    direction_usable = valid.all()
    if settings.method == "direction":
        rank_usable = False
    if settings.method == "ranking":
        direction_usable = False
    if not direction_usable and not rank_usable:
        return finish("NOT_ENOUGH_DATA", "Incomplete directions and insufficient shared-location sector comparisons")
    # Ranking preserves information discarded by bin winners. Direction is a consistency gate.
    scores = ranks if rank_usable else errors / 180
    order = np.argsort(scores, kind="stable")
    best = int(order[0])
    margin = float(scores[order[1]] - scores[best])
    ho, ho_count = ho_scores(assignments, pcis, handovers)
    uncertainty = directions.uncertainty_deg.to_numpy()
    directional_fit = bool(valid.all() and np.all(gap(assignments[best], observed) <= settings.tolerance_deg + uncertainty))
    out.update(best_azimuths=fmt(assignments[best]),
               original_error_deg=float(errors[0]), best_error_deg=float(errors[best]),
               improvement_deg=float(errors[0] - errors[best]),
               ambiguity_margin_deg=float(np.sort(errors)[1] - np.sort(errors)[0]) if valid.any() else np.nan,
               largest_gap_deg=float(gap(az[valid], observed[valid]).max()) if valid.any() else np.nan,
               rank_original_loss=float(ranks[0]), rank_best_loss=float(ranks[best]),
               rank_improvement=float(ranks[0] - ranks[best]), rank_margin=margin if rank_usable else np.nan,
               ho_points=ho_count, ho_original_loss=float(ho[0]), ho_best_loss=float(ho[best]))
    if coverage < 0.25 or out["locations"] < 30:
        return finish("NOT_ENOUGH_DATA", "Sparse angular coverage or fewer than 30 independent locations")
    threshold = settings.min_rank_margin if rank_usable else settings.min_margin_deg / 180
    if margin < threshold:
        return finish("NOT_ENOUGH_DATA", "Multiple sector mappings explain the RF evidence similarly")
    fit = bool(scores[best] <= settings.max_rank_loss) if rank_usable else directional_fit
    if not fit:
        return finish("AZIMUTH_MISMATCH", "No permutation adequately explains the RF evidence")
    if rank_usable and direction_usable and not directional_fit:
        return finish("NOT_ENOUGH_DATA", "Pairwise RF ranking and observed directions disagree")
    if best != 0 and settings.method == "combined" and valid.any() and np.any(gap(assignments[best, valid], observed[valid]) > settings.tolerance_deg + uncertainty[valid]):
        return finish("NOT_ENOUGH_DATA", "Alternative ranking conflicts with the available observed directions")
    if best != 0 and ho_count >= 5 and ho[best] - ho[0] > 0.15:
        return finish("NOT_ENOUGH_DATA", "HO transition locations contradict the RF swap candidate")
    if best == 0:
        out["confidence_score"] = round(100 * min(coverage, 1 - float(scores[0]), min(1, margin / (2 * threshold))))
        return finish("NORMAL", "Original mapping best explains available RF evidence; this is not field verification")
    improvement = float(scores[0] - scores[best])
    min_improvement = settings.min_rank_improvement if rank_usable else settings.min_improvement_deg / 180
    if improvement < min_improvement:
        return finish("NOT_ENOUGH_DATA", "Alternative mapping does not improve enough over the original")
    changed = np.flatnonzero(gap(az, assignments[best]) > 1)
    out["changed_pcis"] = ";".join(str(pcis[k]) for k in changed)
    # Conservative bottleneck score: correlated RF features do not add confidence points.
    quality = min(coverage, 1 - float(scores[best]), min(1, margin / (2 * threshold)))
    if first.match_level != "ID":
        quality = min(quality, 0.59)
    out["confidence_score"] = round(quality * 100)
    ho_support = ho_count >= 5 and ho[0] - ho[best] >= 0.10 and ho[best] <= 0.25
    confirmed = (settings.method == "combined" and rank_usable and directional_fit and ho_support
                 and quality >= 0.75 and first.match_level == "ID")
    label = "CONFIRMED_SWAP" if confirmed else "PROBABLE_SWAP"
    explanation = "Unique alternative mapping improves RF fit; verify configured sectors and antenna connections in the field"
    if confirmed:
        explanation = "Rule-confirmed by RF agreement, coverage and HO transition support; not field-confirmed"
    return finish(label, explanation)
