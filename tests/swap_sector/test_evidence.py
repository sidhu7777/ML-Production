"""Deterministic physical-direction fixtures, independent of configured directions."""
import unittest
import numpy as np
import pandas as pd

from tests.swap_sector.evidence import Settings, analyse, main_run, ho_scores


def fixture():
    az = [0., 120., 240.]
    cells = pd.DataFrame(dict(group_id=["fixture"] * 3, operator=["Test"] * 3,
                              technology=["LTE"] * 3, band=["B1"] * 3, earfcn=[100] * 3,
                              site_id=["123"] * 3, pci=[10, 20, 30], azimuth=az, match_level=["ID"] * 3))
    rng = np.random.default_rng(123)
    rows = []
    for angle in np.arange(5, 360, 10):
        for sample in range(5):
            distance = 100 + sample * 50
            for pci, direction in zip(cells.pci, az):
                rows.append(dict(group_id="fixture", location_id=f"{angle}-{sample}", session_id=1,
                                 pci=pci, bearing_deg=float(angle), distance_m=distance,
                                 rsrp=-75 - 30 * np.log10(distance / 100) + 16 * np.cos(np.radians(angle - direction)) + rng.normal(0, .2)))
    return cells, pd.DataFrame(rows)


class EvidenceTests(unittest.TestCase):
    def test_wrap_and_disconnected_lobes(self):
        mask = np.zeros(36, dtype=bool)
        mask[[34, 35, 0, 1]] = True
        self.assertEqual(set(main_run(mask)), {34, 35, 0, 1})
        mask[[16, 17, 18, 19]] = True
        self.assertEqual(len(main_run(mask)), 0)

    def test_normal_swap_rotation_and_no_mutation(self):
        cells, m = fixture()
        original = m.copy(deep=True)
        self.assertEqual(analyse(cells, m)["verdict"], "NORMAL")
        for assignment in ([120., 0., 240.], [120., 240., 0.]):
            swapped = cells.assign(azimuth=assignment)
            result = analyse(swapped, m)
            self.assertEqual(result["verdict"], "PROBABLE_SWAP")
            self.assertEqual(result["best_azimuths"], "0.0;120.0;240.0")
            self.assertGreater(result["improvement_deg"], 70)
            self.assertEqual(swapped.pci.tolist(), [10, 20, 30])
        pd.testing.assert_frame_equal(m, original)

    def test_sparse_nr_and_truth_is_not_used(self):
        cells, m = fixture()
        self.assertEqual(analyse(cells, m[m.bearing_deg < 25])["verdict"], "NOT_ENOUGH_DATA")
        self.assertEqual(analyse(cells.assign(technology="NR"), m)["verdict"], "NOT_TESTABLE")
        pd.testing.assert_series_equal(pd.Series(analyse(cells.assign(ground_truth="SWAPPED", true_azimuth=999), m)), pd.Series(analyse(cells, m)))

    def test_boundary_is_not_boresight(self):
        events = pd.DataFrame([dict(pci=10, other_pci=20, kind="boundary", bearing_deg=60)])
        scores, n = ho_scores(np.array([[0., 120.], [60., 180.]]), [10, 20], events)
        self.assertEqual(n, 1)
        self.assertEqual(scores[0], 0)
        self.assertGreater(scores[1], 0)

    def test_no_false_confirmation_without_ho(self):
        cells, m = fixture()
        result = analyse(cells.assign(azimuth=[120., 0., 240.]), m)
        self.assertNotEqual(result["verdict"], "CONFIRMED_SWAP")
        self.assertEqual(result["ho_points"], 0)

    def test_singletons_and_ties_are_not_dominance(self):
        from tests.swap_sector.detect_sector_swap import direction_profile
        cells, m = fixture()
        single = m[m.pci.eq(10)]
        self.assertTrue(direction_profile(single, cells.pci.tolist(), True).dominant_pci.isna().all())
        tied = m.assign(rsrp=-80)
        self.assertTrue(direction_profile(tied, cells.pci.tolist(), True).dominant_pci.isna().all())

    def test_partial_rf_can_use_profile_without_changing_truth(self):
        cells, m = fixture()
        # Interleaved locations: no PCI pair is observed together.
        m = m.assign(location_id=m.location_id + "|" + m.pci.astype(str))
        result = analyse(cells.assign(azimuth=[120., 0., 240.]), m)
        self.assertEqual(result["verdict"], "PROBABLE_SWAP")
        self.assertEqual(result["ranking_source"], "signal profile")

    def test_nr_and_invalid_parameters_fail_closed(self):
        with self.assertRaises(ValueError):
            Settings(tolerance_deg=0)
        cells, m = fixture()
        self.assertEqual(analyse(cells.assign(azimuth=0), m)["verdict"], "NOT_TESTABLE")

    def test_handover_requires_both_endpoints_and_no_invalid_bridge(self):
        from tests.swap_sector.build_dataset import build_handovers
        carriers = pd.DataFrame([
            dict(operator="Test", technology="LTE", site_id="123", earfcn=100, pci=10, group_id="a", site_lat=28., site_lon=77., match_level="ID"),
            dict(operator="Test", technology="LTE", site_id="124", earfcn=100, pci=20, group_id="b", site_lat=28.002, site_lon=77., match_level="ID")])
        rows = pd.DataFrame([
            dict(session_id=1, timestamp="2026-01-01 00:00:00", enb_id="123", cell_id=1, earfcn=100, pci=10, operator="Test", technology="LTE", lat=28.001, lon=77.001),
            dict(session_id=1, timestamp="2026-01-01 00:00:02", enb_id="124", cell_id=2, earfcn=100, pci=20, operator="Test", technology="LTE", lat=28.001, lon=77.001)])
        self.assertEqual(set(build_handovers(carriers, rows).kind), {"leaving", "entering"})
        self.assertTrue(build_handovers(carriers.iloc[:1], rows).empty)
        invalid = rows.iloc[:1].assign(timestamp="2026-01-01 00:00:01", enb_id="", cell_id=8388607)
        self.assertTrue(build_handovers(carriers, pd.concat([rows, invalid], ignore_index=True)).empty)


if __name__ == "__main__":
    unittest.main()
