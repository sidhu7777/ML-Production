"""
Coverage-redundancy engine: "if this site/cell is switched off, does the rest of the network
still serve the area it served?"

Per analysis point p the calibrated matrix gives RSRP_c(p) for every cell c.
  serving   c*(p) = argmax over active cells of RSRP_c(p)
  SINR      RSRP_c* / (load * sum of the other active same-layer cells + noise)   (linear, per RE)
  P_cov     probability the true RSRP clears the KPI threshold given the prediction error:
            Phi((RSRP - T_rsrp) / sigma_rsrp), sigma from session-held-out drive-test residuals;
            0 below Qrxlevmin
  covered   P_cov >= location_probability (planning edge-coverage probability)
A hard "RSRP >= threshold" flag counts a 2 dB change at -99.5 dBm as a full loss (seen on the
synthetic overlay site, where the backup antenna 50 m away is ~2 dB weaker at the edge); P_cov
scores that as the small change it is, while a real hole still drops P_cov to ~0.
SINR is judged by its change, not multiplied into P_cov: predicted SINR has ~7 dB held-out error
on project 193, and folding that into every point's probability would bury the RSRP evidence.

For a candidate K (one cell, or every cell of a site) and the current active set A:
  footprint F   points whose serving cell is in K
  after         the same points re-evaluated with A minus K: next-best server, K's interference gone
Outside F the serving cell does not change and interference only drops, so coverage there can
only improve; F is the only place a removal can lose coverage.

Gates, all must pass:
  lost point                  covered before, not covered after, AND P_cov fell by >= min_probability_drop
                              (a 1-2 dB change at the coverage edge is prediction noise, not a loss)
  RETENTION_BELOW_MIN         1 - lost weight / covered weight in F              < min_retention
  EXPECTED_COVERAGE_EROSION   sum w*min(P_after, P_before) / sum w*P_before       < min_expected_retention
                              (catches a broad small drop that never makes single points "lost")
  INDOOR_RETENTION_BELOW_MIN  RETENTION on building points                       < min_indoor_retention
  HOLE_TOO_LARGE              largest 8-connected block of lost points           > max_hole_m2
  POOR_AREA_DEGRADES          median RSRP drop on F points that were not covered before
                              (when they are >= min_uncovered_share of F)        > max_uncovered_rsrp_drop_db
  QUALITY_DEGRADES            share of F whose predicted SINR goes from >= T_sinr to < T_sinr
                                                                                 > max_new_low_sinr_share
  HO_AMBIGUITY_INCREASE       rise in share of F where best and best other-site cell are within
                              the handover hysteresis                            > max_ho_ambiguity_increase
  NETWORK_LOSS_CAP            expected covered weight lost vs the original network > max_cumulative_coverage_loss
  extra gates                 callables(metrics) -> reason or None; the capacity check plugs in here

Removals are not additive (two overlapping sites can each look redundant only because of the
other), so candidates are removed one at a time: evaluate all, remove the safest passing one,
recompute the network, repeat until nothing passes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.special import ndtr

from tests.overlapping.config import (
    VERDICT_KEEP,
    VERDICT_NOT_TESTABLE,
    VERDICT_REDUNDANT,
    DecisionParams,
)

NO_SIGNAL_DBM = -250.0
_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)

ExtraGate = Callable[[dict], Optional[str]]


@dataclass
class Network:
    points: pd.DataFrame      # point_id, lat, lon, row, col, weight, indoor, users
    cells: pd.DataFrame       # antenna_key, site_key, layer; row i = column i of rsrp_dbm
    rsrp_dbm: np.ndarray      # (points, cells) calibrated RSRP, NO_SIGNAL_DBM out of range
    grid_resolution_m: float
    noise_per_re_dbm: float
    interference_load_factor: float
    sigma_rsrp_db: float

    def __post_init__(self) -> None:
        expected = (len(self.points), len(self.cells))
        if self.rsrp_dbm.shape != expected:
            raise ValueError(f"rsrp_dbm shape {self.rsrp_dbm.shape} != (points, cells) {expected}")


@dataclass(frozen=True)
class DesignThresholds:
    kpi_rsrp_dbm: float
    rsrp_margin_db: float        # z(location_probability) * sigma: the equivalent hard fade margin
    kpi_sinr_db: float
    qrxlevmin_dbm: float
    location_probability: float

    @property
    def rsrp_dbm(self) -> float:
        return self.kpi_rsrp_dbm + self.rsrp_margin_db


def design_thresholds(decision: DecisionParams, sigma_rsrp_db: float) -> DesignThresholds:
    z = NormalDist().inv_cdf(decision.location_probability)
    return DesignThresholds(
        kpi_rsrp_dbm=decision.rsrp_threshold_dbm,
        rsrp_margin_db=max(0.0, z * sigma_rsrp_db),
        kpi_sinr_db=decision.sinr_threshold_db,
        qrxlevmin_dbm=decision.qrxlevmin_dbm,
        location_probability=decision.location_probability,
    )


@dataclass
class PointState:
    serving: np.ndarray         # cell index, -1 = no signal
    rsrp: np.ndarray
    sinr: np.ndarray
    coverage_prob: np.ndarray   # P(true RSRP and SINR clear the KPI thresholds)
    covered: np.ndarray         # coverage_prob >= location probability
    other_site_gap: np.ndarray  # dB to the best active cell on another site (inf = none)

    def take(self, idx: np.ndarray) -> "PointState":
        return PointState(
            self.serving[idx], self.rsrp[idx], self.sinr[idx], self.coverage_prob[idx], self.covered[idx], self.other_site_gap[idx]
        )


@dataclass
class Candidate:
    candidate_id: str
    cells: np.ndarray
    verdict: str = ""   # preset verdict (DATA_DUPLICATE / NOT_TESTABLE) skips evaluation
    reason: str = ""


class CoverageEvaluator:
    def __init__(self, net: Network, thresholds: DesignThresholds, top_m: int = 8):
        self.net = net
        self.thresholds = thresholds
        self.rsrp = np.asarray(net.rsrp_dbm, dtype=np.float32)
        signal = self.rsrp > NO_SIGNAL_DBM + 1.0
        self.power_mw = np.where(signal, np.power(10.0, self.rsrp.astype(np.float64) / 10.0), 0.0)
        self.site_codes, self.site_names = pd.factorize(net.cells["site_key"].astype(str))
        self.layer_codes, _ = pd.factorize(net.cells["layer"].astype(str))
        self.noise_mw = 10.0 ** (net.noise_per_re_dbm / 10.0)
        self.load = float(net.interference_load_factor)
        self.weight = net.points["weight"].to_numpy(dtype=float)
        self.indoor = net.points["indoor"].to_numpy(dtype=bool)
        self.users = net.points["users"].to_numpy(dtype=float) if "users" in net.points else np.zeros(len(net.points))
        self.row = net.points["row"].to_numpy(dtype=int)
        self.col = net.points["col"].to_numpy(dtype=int)
        self.top_m = max(1, min(top_m, len(net.cells)))

    def state(self, active: np.ndarray, rows: np.ndarray | None = None) -> PointState:
        rsrp = self.rsrp if rows is None else self.rsrp[rows]
        power = self.power_mw if rows is None else self.power_mw[rows]
        masked = np.where(active[None, :], rsrp, np.float32(NO_SIGNAL_DBM))
        n = masked.shape[0]
        ar = np.arange(n)
        serving = masked.argmax(axis=1)
        srv_rsrp = masked[ar, serving].astype(float)
        has = srv_rsrp > NO_SIGNAL_DBM + 1.0
        srv_power = power[ar, serving]

        interference = np.zeros(n)
        serving_layer = self.layer_codes[serving]
        for layer in np.unique(serving_layer[has]):
            cols = active & (self.layer_codes == layer)
            on_layer = has & (serving_layer == layer)
            interference[on_layer] = power[on_layer][:, cols].sum(axis=1) - srv_power[on_layer]
        interference = np.maximum(interference, 0.0) * self.load + self.noise_mw
        with np.errstate(divide="ignore"):
            sinr = np.where(has, 10.0 * np.log10(np.maximum(srv_power, 1e-300) / interference), -np.inf)

        prob = self._coverage_probability(has, srv_rsrp)
        gap = self._other_site_gap(masked, serving, srv_rsrp)
        return PointState(
            serving=np.where(has, serving, -1),
            rsrp=np.where(has, srv_rsrp, NO_SIGNAL_DBM),
            sinr=sinr,
            coverage_prob=prob,
            covered=prob >= self.thresholds.location_probability,
            other_site_gap=np.where(has, gap, np.inf),
        )

    def _coverage_probability(self, has: np.ndarray, rsrp: np.ndarray) -> np.ndarray:
        thr = self.thresholds
        p_rsrp = ndtr((rsrp - thr.kpi_rsrp_dbm) / max(self.net.sigma_rsrp_db, 1e-3))
        return np.where(has & (rsrp >= thr.qrxlevmin_dbm), p_rsrp, 0.0)

    def _other_site_gap(self, masked: np.ndarray, serving: np.ndarray, srv_rsrp: np.ndarray) -> np.ndarray:
        n, n_cells = masked.shape
        if n == 0 or n_cells < 2:
            return np.full(n, np.inf)
        m = self.top_m
        idx = np.argpartition(-masked, m - 1, axis=1)[:, :m] if m < n_cells else np.tile(np.arange(n_cells), (n, 1))
        vals = np.take_along_axis(masked, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        other = (self.site_codes[idx] != self.site_codes[serving][:, None]) & (vals > NO_SIGNAL_DBM + 1.0)
        first = other.argmax(axis=1)
        found = other[np.arange(n), first]
        return np.where(found, srv_rsrp - vals[np.arange(n), first], np.inf)


def largest_hole_points(rows: np.ndarray, cols: np.ndarray) -> int:
    if rows.size == 0:
        return 0
    r = rows - rows.min()
    c = cols - cols.min()
    raster = np.zeros((r.max() + 1, c.max() + 1), dtype=bool)
    raster[r, c] = True
    labels, count = ndimage.label(raster, structure=_EIGHT_CONNECTED)
    return int(np.bincount(labels.ravel())[1:].max()) if count else 0


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & (weights > 0)
    if not ok.any():
        return float("nan")
    order = np.argsort(values[ok])
    v, w = values[ok][order], weights[ok][order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def _p10(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, 10)) if finite.size else float("nan")


def _gate_failures(m: dict, decision: DecisionParams, extra_gates: Sequence[ExtraGate]) -> list[str]:
    failures = []
    if np.isfinite(m["retention"]) and m["retention"] < decision.min_retention:
        failures.append("RETENTION_BELOW_MIN")
    if np.isfinite(m["expected_retention"]) and m["expected_retention"] < decision.min_expected_retention:
        failures.append("EXPECTED_COVERAGE_EROSION")
    if np.isfinite(m["indoor_retention"]) and m["indoor_retention"] < decision.min_indoor_retention:
        failures.append("INDOOR_RETENTION_BELOW_MIN")
    if m["largest_hole_m2"] > decision.max_hole_m2:
        failures.append("HOLE_TOO_LARGE")
    if (
        m["uncovered_share"] >= decision.min_uncovered_share
        and np.isfinite(m["uncovered_rsrp_drop_db"])
        and m["uncovered_rsrp_drop_db"] > decision.max_uncovered_rsrp_drop_db
    ):
        failures.append("POOR_AREA_DEGRADES")
    if m["new_low_sinr_share"] > decision.max_new_low_sinr_share:
        failures.append("QUALITY_DEGRADES")
    if m["ho_ambiguity_increase"] > decision.max_ho_ambiguity_increase:
        failures.append("HO_AMBIGUITY_INCREASE")
    if m["cumulative_coverage_loss"] > decision.max_cumulative_coverage_loss + 1e-12:
        failures.append("NETWORK_LOSS_CAP")
    for gate in extra_gates:
        reason = gate(m)
        if reason:
            failures.append(str(reason))
    return failures


def evaluate_candidate(
    ev: CoverageEvaluator,
    decision: DecisionParams,
    state: PointState,
    active: np.ndarray,
    cells: np.ndarray,
    covered_weight_now: float,
    covered_weight_initial: float,
    extra_gates: Sequence[ExtraGate] = (),
    keep_points: bool = False,
) -> dict:
    net = ev.net
    cells = np.asarray(cells, dtype=int)
    area = net.grid_resolution_m ** 2
    footprint = np.flatnonzero(np.isin(state.serving, cells))
    w = ev.weight[footprint]
    wsum = float(w.sum())
    nan = float("nan")
    m = {
        "footprint_points": int(footprint.size),
        "footprint_area_m2": float(footprint.size * area),
        "footprint_weight": wsum,
        "footprint_users": float(ev.users[footprint].sum()),
        "retention": nan,
        "expected_retention": nan,
        "hard_retention": nan,
        "indoor_retention": nan,
        "largest_hole_m2": 0.0,
        "lost_area_m2": 0.0,
        "gained_area_m2": 0.0,
        "uncovered_share": 0.0,
        "uncovered_rsrp_drop_db": nan,
        "covered_rsrp_drop_p50_db": nan,
        "rsrp_after_p10_dbm": nan,
        "sinr_before_p10_db": nan,
        "sinr_after_p10_db": nan,
        "low_sinr_share_before": 0.0,
        "new_low_sinr_share": 0.0,
        "ho_ambiguity_before": 0.0,
        "ho_ambiguity_after": 0.0,
        "ho_ambiguity_increase": 0.0,
        "absorbers": "[]",
        "absorber_sites": "",
    }
    new_expected = covered_weight_now
    if footprint.size:
        new_active = active.copy()
        new_active[cells] = False
        before = state.take(footprint)
        after = ev.state(new_active, footprint)
        cb, ca = before.covered, after.covered
        pb, pa = before.coverage_prob, after.coverage_prob
        lost = cb & ~ca & (pb - pa >= decision.min_probability_drop)
        gained = ~cb & ca

        w_covered = float(w[cb].sum())
        if w_covered > 0:
            m["retention"] = 1.0 - float(w[lost].sum()) / w_covered
            m["hard_retention"] = float(w[cb & ca].sum()) / w_covered
        expected_before = float((w * pb).sum())
        if expected_before > 0:
            m["expected_retention"] = float((w * np.minimum(pa, pb)).sum()) / expected_before
        indoor = ev.indoor[footprint]
        w_indoor = float(w[cb & indoor].sum())
        if w_indoor > 0:
            m["indoor_retention"] = 1.0 - float(w[lost & indoor].sum()) / w_indoor

        m["largest_hole_m2"] = float(largest_hole_points(ev.row[footprint][lost], ev.col[footprint][lost]) * area)
        m["lost_area_m2"] = float(lost.sum() * area)
        m["gained_area_m2"] = float(gained.sum() * area)

        drop = np.minimum(before.rsrp - after.rsrp, 60.0)
        uncovered = ~cb
        if wsum > 0:
            m["uncovered_share"] = float(w[uncovered].sum()) / wsum
        if uncovered.any():
            m["uncovered_rsrp_drop_db"] = weighted_median(drop[uncovered], w[uncovered])
        if cb.any():
            m["covered_rsrp_drop_p50_db"] = weighted_median(drop[cb], w[cb])
        m["rsrp_after_p10_dbm"] = _p10(np.where(after.serving >= 0, after.rsrp, np.nan))
        m["sinr_before_p10_db"] = _p10(before.sinr)
        m["sinr_after_p10_db"] = _p10(after.sinr)
        if wsum > 0:
            sinr_ok_before = before.sinr >= ev.thresholds.kpi_sinr_db
            m["low_sinr_share_before"] = float(w[~sinr_ok_before].sum()) / wsum
            m["new_low_sinr_share"] = float(w[sinr_ok_before & (after.sinr < ev.thresholds.kpi_sinr_db)].sum()) / wsum

        if wsum > 0:
            hyst = decision.ho_hysteresis_db
            m["ho_ambiguity_before"] = float(w[cb & (before.other_site_gap < hyst)].sum()) / wsum
            m["ho_ambiguity_after"] = float(w[ca & (after.other_site_gap < hyst)].sum()) / wsum
            m["ho_ambiguity_increase"] = m["ho_ambiguity_after"] - m["ho_ambiguity_before"]

        new_expected = covered_weight_now - float((w * (pb - pa)).sum())

        valid = after.serving >= 0
        if valid.any() and wsum > 0:
            share = np.bincount(after.serving[valid], weights=w[valid], minlength=len(net.cells)) / wsum
            top = [i for i in np.argsort(-share)[:3] if share[i] > 0]
            m["absorbers"] = json.dumps(
                [
                    {
                        "antenna_key": str(net.cells["antenna_key"].iat[i]),
                        "site_key": str(net.cells["site_key"].iat[i]),
                        "share": round(float(share[i]), 4),
                    }
                    for i in top
                ]
            )
            site_share = np.bincount(ev.site_codes[after.serving[valid]], weights=w[valid], minlength=len(ev.site_names))
            m["absorber_sites"] = "|".join(str(ev.site_names[i]) for i in np.argsort(-site_share)[:3] if site_share[i] > 0)
        if keep_points:
            m["_points"] = {
                "point_idx": footprint,
                "rsrp_before": before.rsrp,
                "rsrp_after": after.rsrp,
                "prob_before": pb,
                "prob_after": pa,
                "lost": lost,
                "serving_after": after.serving,
            }

    m["cumulative_coverage_loss"] = (
        (covered_weight_initial - new_expected) / covered_weight_initial if covered_weight_initial > 0 else 0.0
    )
    m["failures"] = _gate_failures(m, decision, extra_gates)
    m["passed"] = not m["failures"]
    return m


@dataclass
class RedundancyResult:
    candidates: pd.DataFrame
    iterations: pd.DataFrame
    initial_state: PointState
    final_state: PointState
    final_active: np.ndarray
    thresholds: DesignThresholds
    covered_weight_initial: float
    covered_weight_final: float
    # candidate -> per-point arrays of removing it alone from the full network (for review / maps)
    standalone_points: dict = field(default_factory=dict)


def _public(m: dict) -> dict:
    out = {k: v for k, v in m.items() if k not in ("failures", "passed") and not k.startswith("_")}
    out["failures"] = ";".join(m["failures"])
    return out


def _standalone(m: dict) -> dict:
    return {
        "standalone_passed": bool(m["passed"]),
        "standalone_failures": ";".join(m["failures"]),
        "standalone_retention": m["retention"],
        "standalone_expected_retention": m["expected_retention"],
        "standalone_largest_hole_m2": m["largest_hole_m2"],
    }


def _safety_key(m: dict, candidate_id: str) -> tuple:
    retention = m["retention"] if np.isfinite(m["retention"]) else 1.0
    expected = m["expected_retention"] if np.isfinite(m["expected_retention"]) else 1.0
    return (-retention, -expected, m["largest_hole_m2"], m["ho_ambiguity_increase"], m["footprint_weight"], candidate_id)


def run_redundancy(
    net: Network,
    candidates: Sequence[Candidate],
    decision: DecisionParams,
    thresholds: DesignThresholds,
    extra_gates: Sequence[ExtraGate] = (),
    log: Callable[[str], None] = print,
) -> RedundancyResult:
    ev = CoverageEvaluator(net, thresholds)
    active = np.ones(len(net.cells), dtype=bool)
    initial = ev.state(active)
    state = initial
    covered_initial = float((ev.weight * initial.coverage_prob).sum())   # expected covered weight
    covered_now = covered_initial
    total_weight = float(ev.weight.sum())
    log(
        f"[OVERLAP][NETWORK] points={len(net.points)} cells={len(net.cells)} "
        f"expected_coverage={covered_initial / max(total_weight, 1e-12):.3f} "
        f"covered_share(P>={thresholds.location_probability})={float(ev.weight[initial.covered].sum()) / max(total_weight, 1e-12):.3f} "
        f"equivalent_design_rsrp={thresholds.rsrp_dbm:.1f}dBm"
    )

    rows: dict[str, dict] = {}
    remaining: list[Candidate] = []
    for cand in candidates:
        base = {"candidate_id": cand.candidate_id, "n_cells": int(len(cand.cells))}
        if cand.verdict:
            rows[cand.candidate_id] = {**base, "verdict": cand.verdict, "reason": cand.reason}
        elif len(cand.cells) == 0:
            rows[cand.candidate_id] = {**base, "verdict": VERDICT_NOT_TESTABLE, "reason": "NO_RF_CELLS"}
        else:
            remaining.append(cand)

    standalone: dict[str, dict] = {}
    standalone_points: dict[str, dict | None] = {}
    last_eval: dict[str, dict] = {}
    accepted: dict[str, tuple[int, dict, float]] = {}
    limit_hit: set[str] = set()
    iterations: list[dict] = []
    iteration = 0
    while remaining:
        evals = {
            c.candidate_id: evaluate_candidate(
                ev, decision, state, active, c.cells, covered_now, covered_initial, extra_gates, keep_points=iteration == 0
            )
            for c in remaining
        }
        if iteration == 0:
            testable = []
            for c in remaining:
                e = evals[c.candidate_id]
                e["potential_points"] = int(np.count_nonzero((ev.rsrp[:, c.cells] >= thresholds.rsrp_dbm).any(axis=1)))
                if e["footprint_points"] < decision.min_footprint_points and e["potential_points"] < decision.min_potential_points:
                    rows[c.candidate_id] = {
                        "candidate_id": c.candidate_id,
                        "n_cells": int(len(c.cells)),
                        "verdict": VERDICT_NOT_TESTABLE,
                        "reason": "TOO_LITTLE_AREA",
                        **_public(e),
                    }
                else:
                    standalone_points[c.candidate_id] = e.pop("_points", None)
                    standalone[c.candidate_id] = e
                    testable.append(c)
            remaining = testable
            log(f"[OVERLAP][STANDALONE] evaluated={len(evals)} testable={len(remaining)} "
                f"pass_alone={sum(standalone[c.candidate_id]['passed'] for c in remaining)}")
        last_eval.update({c.candidate_id: evals[c.candidate_id] for c in remaining})
        passing = [c for c in remaining if evals[c.candidate_id]["passed"]]
        if not passing:
            break
        if decision.max_removals is not None and len(accepted) >= decision.max_removals:
            limit_hit.update(c.candidate_id for c in passing)
            break

        best = min(passing, key=lambda c: _safety_key(evals[c.candidate_id], c.candidate_id))
        active[best.cells] = False
        state = ev.state(active)
        covered_now = float((ev.weight * state.coverage_prob).sum())
        actual_loss = (covered_initial - covered_now) / covered_initial if covered_initial > 0 else 0.0
        iteration += 1
        accepted[best.candidate_id] = (iteration, evals[best.candidate_id], actual_loss)
        remaining = [c for c in remaining if c.candidate_id != best.candidate_id]
        e = evals[best.candidate_id]
        iterations.append(
            {
                "iteration": iteration,
                "removed_candidate": best.candidate_id,
                "passing_candidates": len(passing),
                "retention": e["retention"],
                "largest_hole_m2": e["largest_hole_m2"],
                "footprint_points": e["footprint_points"],
                "network_coverage_loss": actual_loss,
                "network_covered_share": covered_now / max(total_weight, 1e-12),
            }
        )
        log(f"[OVERLAP][ITER {iteration}] removed={best.candidate_id} passing={len(passing)} "
            f"remaining={len(remaining)} network_loss={actual_loss:.4f}")

    for cid, (rank, e, loss) in accepted.items():
        rows[cid] = {
            "candidate_id": cid,
            "n_cells": None,
            "verdict": VERDICT_REDUNDANT,
            "reason": "",
            "removal_rank": rank,
            "network_coverage_loss_after": loss,
            **_public(e),
            **_standalone(standalone[cid]),
        }
    for c in remaining:
        e, s = last_eval[c.candidate_id], standalone[c.candidate_id]
        if c.candidate_id in limit_hit:
            reason = "MAX_REMOVALS_REACHED"
        elif s["passed"]:
            reason = "NEEDED_AFTER_REMOVALS(" + ";".join(e["failures"]) + ")"
        else:
            reason = ";".join(e["failures"])
        rows[c.candidate_id] = {
            "candidate_id": c.candidate_id,
            "n_cells": int(len(c.cells)),
            "verdict": VERDICT_KEEP,
            "reason": reason,
            **_public(e),
            **_standalone(s),
        }

    n_cells_by_id = {c.candidate_id: int(len(c.cells)) for c in candidates}
    ordered = []
    for cand in candidates:
        row = rows[cand.candidate_id]
        row["n_cells"] = n_cells_by_id[cand.candidate_id]
        ordered.append(row)
    return RedundancyResult(
        candidates=pd.DataFrame(ordered),
        iterations=pd.DataFrame(iterations),
        initial_state=initial,
        final_state=state,
        final_active=active,
        thresholds=thresholds,
        covered_weight_initial=covered_initial,
        covered_weight_final=covered_now,
        standalone_points=standalone_points,
    )
