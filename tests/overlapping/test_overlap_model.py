"""
Unit + known-answer tests for the coverage-redundancy model. No database.

Run from the ML/ directory:
    venv\\Scripts\\python.exe -m pytest tests/overlapping -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.overlapping.config import (
    VERDICT_DUPLICATE,
    VERDICT_KEEP,
    VERDICT_NOT_TESTABLE,
    VERDICT_REDUNDANT,
    CleaningParams,
    DecisionParams,
    OverlapConfig,
    RFParams,
)
from tests.overlapping.geo import orient_lonlat, union_area
from tests.overlapping.make_synthetic_overlap import build_synthetic_case, score
from tests.overlapping.removal_model import (
    Candidate,
    CoverageEvaluator,
    Network,
    design_thresholds,
    evaluate_candidate,
    largest_hole_points,
    run_redundancy,
    weighted_median,
)
from tests.overlapping.rf_matrix import (
    VALIDATION_FAIL,
    VALIDATION_PASS,
    DriveTestCalibration,
    calibrate_with_drive_test,
    raw_rsrp_matrix,
)
from tests.overlapping.run_overlap import run_pipeline, withhold_unvalidated
from tests.overlapping.site_cleaning import clean_sites, cluster_azimuths, normalise_operator

TOY_DECISION = DecisionParams(level="cell", min_footprint_points=1, min_potential_points=1)


def toy_network(rsrp, sites, cols=None, sigma_rsrp=0.0, load=1.0) -> Network:
    rsrp = np.asarray(rsrp, dtype=np.float32)
    n_points, n_cells = rsrp.shape
    points = pd.DataFrame(
        {
            "point_id": np.arange(n_points),
            "lat": 0.0,
            "lon": 0.0,
            "row": 0,
            "col": np.arange(n_points) * 2 if cols is None else cols,   # not adjacent unless given
            "weight": 1.0,
            "indoor": False,
            "users": 0.0,
        }
    )
    cells = pd.DataFrame({"antenna_key": [f"c{i}" for i in range(n_cells)], "site_key": sites, "layer": "L1"})
    # sigma 0 (default) -> coverage probability is a 0/1 step exactly at the KPI threshold
    return Network(points, cells, rsrp, 25.0, RFParams().noise_per_re_dbm, load, sigma_rsrp)


def initial_metrics(net: Network, cell: int) -> dict:
    ev = CoverageEvaluator(net, design_thresholds(TOY_DECISION, net.sigma_rsrp_db))
    active = np.ones(len(net.cells), dtype=bool)
    state = ev.state(active)
    expected = float((ev.weight * state.coverage_prob).sum())
    return evaluate_candidate(ev, TOY_DECISION, state, active, np.array([cell]), expected, expected)


def run_toy(net: Network, extra_gates=()):
    thr = design_thresholds(TOY_DECISION, net.sigma_rsrp_db)
    cands = [Candidate(f"c{i}", np.array([i])) for i in range(len(net.cells))]
    result = run_redundancy(net, cands, TOY_DECISION, thr, extra_gates, log=lambda *_: None)
    return result, dict(zip(result.candidates["candidate_id"], result.candidates["verdict"]))


def test_cluster_azimuths_wraps_through_north():
    labels = cluster_azimuths(np.array([355.0, 5.0, 120.0, 125.0, 240.0]), 15.0)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert len(set(labels)) == 3


def test_largest_hole_is_eight_connected():
    assert largest_hole_points(np.array([0, 1, 5]), np.array([0, 1, 5])) == 2
    assert largest_hole_points(np.array([], dtype=int), np.array([], dtype=int)) == 0


def test_weighted_median():
    assert weighted_median(np.array([1.0, 2.0, 10.0]), np.array([1.0, 1.0, 5.0])) == 10.0


def test_operator_aliases():
    assert normalise_operator("airtel") == "Airtel"
    assert normalise_operator("JIO 4G") == "JIO 4G"
    assert normalise_operator("Vi India") == "Vi India"
    assert normalise_operator("000 000") == ""


def test_removing_an_interferer_raises_sinr():
    net = toy_network([[-80.0, -82.0]], sites=["A", "B"])
    ev = CoverageEvaluator(net, design_thresholds(TOY_DECISION, 0.0))
    both = ev.state(np.array([True, True]))
    alone = ev.state(np.array([True, False]))
    assert both.sinr[0] == pytest.approx(2.0, abs=0.05)   # noise (-125 dBm) is negligible here
    assert alone.sinr[0] > 40.0
    assert both.other_site_gap[0] == pytest.approx(2.0, abs=1e-4)


def test_cell_covered_by_neighbour_is_redundant_and_neighbour_is_kept():
    # c0 serves points 0-1 strongly; c1 reaches every point at -90 dBm; c0 is weak at points 2-3.
    net = toy_network([[-70.0, -90.0], [-72.0, -90.0], [-125.0, -90.0], [-125.0, -90.0]], sites=["A", "B"])
    result, verdicts = run_toy(net)
    assert verdicts == {"c0": VERDICT_REDUNDANT, "c1": VERDICT_KEEP}
    assert "RETENTION_BELOW_MIN" in result.candidates.set_index("candidate_id").loc["c1", "reason"]


def test_small_drop_at_coverage_edge_is_not_a_loss():
    # c0 serves 10 points; the other site's c1 is 1 dB weaker on 9 strong points and 3.5 dB weaker
    # on one edge point. A hard threshold calls the edge point lost (hard retention 0.9 < 0.97);
    # expected-coverage retention sees the small change it is.
    net = toy_network([[-80.0, -81.0]] * 9 + [[-97.0, -100.5]], sites=["A", "B"], sigma_rsrp=8.0)
    m = initial_metrics(net, 0)
    assert m["hard_retention"] == pytest.approx(0.9)
    assert m["retention"] == pytest.approx(1.0)
    assert m["expected_retention"] == pytest.approx(0.987, abs=0.002)
    assert m["largest_hole_m2"] == 0.0
    # c0 serves the whole toy network, so only the network-wide loss cap (1%) may trip here
    assert set(m["failures"]) <= {"NETWORK_LOSS_CAP"}, m["failures"]


def test_quality_gate_catches_sinr_collapse_with_rsrp_intact():
    # c0 serves 4 points at -70 dBm over four -80 dBm interferers (SINR ~4 dB). Without c0 the next
    # server is -80 dBm against three equal interferers (SINR ~-4.8 dB): RSRP fine, quality not.
    net = toy_network([[-70.0, -80.0, -80.0, -80.0, -80.0]] * 4, sites=["A", "B", "C", "D", "E"])
    m = initial_metrics(net, 0)
    assert m["retention"] == pytest.approx(1.0)
    assert m["new_low_sinr_share"] == pytest.approx(1.0)
    assert "QUALITY_DEGRADES" in m["failures"]
    assert "RETENTION_BELOW_MIN" not in m["failures"]


def test_mutual_backups_are_not_both_removed():
    net = toy_network([[-80.0, -80.0]] * 4, sites=["A", "B"])
    result, verdicts = run_toy(net)
    assert sorted(verdicts.values()) == sorted([VERDICT_KEEP, VERDICT_REDUNDANT])
    kept = result.candidates[result.candidates["verdict"] == VERDICT_KEEP].iloc[0]
    assert kept["reason"].startswith("NEEDED_AFTER_REMOVALS")


def test_hole_gate_uses_contiguous_area():
    # c0 serves 5 adjacent points; nothing else reaches them -> one 5-point hole (3125 m2 > 2500 m2).
    rsrp = [[-80.0, -130.0]] * 5 + [[-130.0, -80.0]]
    net = toy_network(rsrp, sites=["A", "B"], cols=np.array([0, 1, 2, 3, 4, 10]))
    result, verdicts = run_toy(net)
    row = result.candidates.set_index("candidate_id").loc["c0"]
    assert verdicts["c0"] == VERDICT_KEEP
    assert row["largest_hole_m2"] == pytest.approx(5 * 625.0)
    assert "HOLE_TOO_LARGE" in row["reason"]


def test_extra_gate_hook_can_block_a_removal():
    net = toy_network([[-70.0, -90.0], [-72.0, -90.0], [-125.0, -90.0], [-125.0, -90.0]], sites=["A", "B"])
    result, verdicts = run_toy(net, extra_gates=[lambda m: "CAPACITY_NOT_AVAILABLE"])
    assert verdicts["c0"] == VERDICT_KEEP
    assert "CAPACITY_NOT_AVAILABLE" in result.candidates.set_index("candidate_id").loc["c0", "reason"]


def test_duplicate_and_ambiguous_site_detection():
    def rows(site, lat, lon, azimuths, pcis, first_id):
        return [
            {"id": first_id + k, "site": site, "cell_id": f"{site}_{k}", "latitude": lat, "longitude": lon,
             "pci": pcis[k], "azimuth": az, "height": 30, "m_tilt": 2, "e_tilt": 4, "tx_power": 46,
             "cluster": "Airtel", "Technology": "4G"}
            for k, az in enumerate(azimuths)
        ]

    raw = pd.DataFrame(
        rows(1001, 28.60000, 77.30000, [0, 120, 240], [10, 11, 12], 1)
        + rows(1002, 28.60003, 77.30002, [5, 125, 245], [20, 21, 22], 10)     # ~4 m away, same directions
        + rows(7.5, 28.60000, 77.30150, [10, 130, 250], [10, 11, 12], 20)     # ~146 m away, same PCIs
        + rows(2001, 28.61000, 77.31000, [k * 45 for k in range(8)], list(range(40, 48)), 30)
    )
    ant, _ = clean_sites(raw, "Airtel", CleaningParams(), RFParams())
    by_site = ant.groupby("site_key")
    assert set(by_site.get_group("1002")["duplicate_rule"]) == {"CO_LOCATED"}
    assert set(by_site.get_group("7.5")["duplicate_rule"]) == {"SAME_PCI_NEARBY"}
    assert not by_site.get_group("1001")["is_duplicate"].any()
    assert by_site.get_group("2001")["site_config_issue"].iat[0] == "TOO_MANY_AZIMUTHS"
    assert not by_site.get_group("2001")["rf_include"].any()


def test_drive_test_calibration_recovers_injected_offset():
    # Drive test generated from the synthetic network: best-server RSRP - 5 dB + N(0, 4 dB), served
    # by the best site. Calibration must find -5 dB, a ~4 dB held-out sigma, and PASS.
    case = build_synthetic_case(seed=7)
    cfg = OverlapConfig(operator="Airtel")
    ant, _ = clean_sites(case.inputs.site_rows, "Airtel", cfg.cleaning, cfg.rf)
    cells = ant[ant["rf_include"]].reset_index(drop=True)
    area, _ = orient_lonlat(union_area(case.inputs.polygon_wkts), cells["lat"], cells["lon"])
    rng = np.random.default_rng(3)
    n = 3000
    minx, miny, maxx, maxy = area.bounds
    lon, lat = rng.uniform(minx, maxx, n), rng.uniform(miny, maxy, n)
    raw = raw_rsrp_matrix(cells, lat, lon, cfg.rf)
    best = raw.argmax(axis=1)
    dt = pd.DataFrame(
        {
            "session_id": np.arange(n) % 6,
            "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(n), unit="s"),
            "lat": lat,
            "lon": lon,
            "band": "B3",
            "pci": np.nan,
            "nodeb_id": cells["site_key"].to_numpy()[best],
            "m_alpha_long": "airtel",
            "rsrp": raw[np.arange(n), best] - 5.0 + rng.normal(0.0, 4.0, n),
            "sinr": 0.0,
        }
    )
    cal = calibrate_with_drive_test(cells, dt, "Airtel", area.buffer(0.01), cfg.rf, cfg.calibration)
    assert cal.validation_status == VALIDATION_PASS, cal.validation_failures
    assert cal.reference_offset_db == pytest.approx(-5.0, abs=1.0)
    assert cal.sigma_rsrp_db == pytest.approx(4.0, abs=1.5)
    assert cal.heldout["site_agreement"] > 0.9
    assert cal.load_source == "default"


def test_unvalidated_rf_surface_withholds_rf_verdicts():
    table = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "verdict": [VERDICT_REDUNDANT, VERDICT_KEEP, VERDICT_DUPLICATE],
            "reason": ["", "HOLE_TOO_LARGE", "DUPLICATE_OF(x)"],
            "confidence": ["HIGH", "LOW", ""],
        }
    )
    dt = DriveTestCalibration(
        offsets_db=np.zeros(1), sigma_rsrp_db=8.0, interference_load_factor=0.5,
        validation_status=VALIDATION_FAIL, validation_failures=["HELDOUT_BIAS(-12.5dB)"],
    )
    out = withhold_unvalidated(table, dt)
    assert out["verdict"].tolist() == [VERDICT_NOT_TESTABLE, VERDICT_NOT_TESTABLE, VERDICT_DUPLICATE]
    assert out["model_verdict"].tolist() == [VERDICT_REDUNDANT, VERDICT_KEEP, VERDICT_DUPLICATE]
    assert out["reason"].iat[0].startswith("RF_SURFACE_NOT_VALIDATED")


def test_synthetic_scenario_matches_known_answers():
    case = build_synthetic_case(seed=7)
    run = run_pipeline(OverlapConfig(project_id=0, operator="Airtel"), case.inputs, log=lambda *_: None)
    table, ok = score(case, run.candidates)
    assert ok, "\n" + table.to_string(index=False)
    assert run.summary["data"]["polygon_swapped_lat_lon"] is True
    assert run.summary["rf_surface_validation"]["status"] == "NOT_CHECKED"
    assert set(run.candidates["verdict"]) >= {VERDICT_REDUNDANT, VERDICT_KEEP, VERDICT_DUPLICATE}
