"""Deterministic fixtures for the antenna-pattern swap detector (no database, no project data)."""
import unittest

import numpy as np
import pandas as pd

from tests.swap_sector import antenna_profile as ap
from tests.swap_sector.pattern_detector import SWAP_VERDICTS, analyse

B3_EARFCN, B40_EARFCN = 1346, 39150


def gap(a, b):
    return abs((a - b + 180) % 360 - 180)


def make_group(true_az, configured_az=None, earfcn=B3_EARFCN, antenna_model="", noise_db=1.0,
               bearings=np.arange(5, 360, 10), distances=(150, 300, 450, 600), seed=7):
    """Cells configured with `configured_az`; drive-test RF generated from antenna patterns pointing at `true_az`."""
    configured_az = list(true_az if configured_az is None else configured_az)
    frequency, band = ap.carrier_frequency_mhz(earfcn, "LTE")
    pcis = [101 + k for k in range(len(true_az))]
    cells = pd.DataFrame({
        "group_id": "g", "operator": "Test", "site_id": "1", "technology": "LTE", "band": band, "earfcn": earfcn,
        "pci": pcis, "azimuth": configured_az, "e_tilt": 4.0, "m_tilt": 2.0, "antenna_model": antenna_model,
        "pattern_port": "", "frequency_mhz": frequency, "match_level": "ID"})
    patterns = [ap.select_pattern({**cells.iloc[k].to_dict(), "azimuth": true_az[k]}) for k in range(len(true_az))]
    rng = np.random.default_rng(seed)
    rows = []
    for bearing in bearings:
        for distance in distances:
            for k, pci in enumerate(pcis):
                power = -65 - 30 * np.log10(distance / 100) + float(ap.horizontal_gain_db(patterns[k], bearing - true_az[k]))
                rows.append({"group_id": "g", "location_id": f"{bearing}|{distance}", "pci": pci, "bearing_deg": float(bearing),
                             "distance_m": float(distance), "rsrp": power + rng.normal(0, noise_db), "is_serving": False})
    return cells, pd.DataFrame(rows)


def leaving_events(pcis, true_az, per_sector=6):
    return pd.DataFrame([{"group_id": "g", "kind": "leaving", "pci": pci, "other_pci": -1,
                          "bearing_deg": (azimuth + 3 * (k - per_sector / 2)) % 360}
                         for pci, azimuth in zip(pcis, true_az) for k in range(per_sector)])


class FrequencyAndPatternSelection(unittest.TestCase):
    def test_earfcn_to_downlink_frequency(self):
        self.assertEqual(ap.carrier_frequency_mhz(1300, "LTE"), (1815.0, "B3"))
        self.assertEqual(ap.carrier_frequency_mhz(38950, "LTE"), (2330.0, "B40"))
        self.assertEqual(ap.carrier_frequency_mhz(315, "LTE"), (2141.5, "B1"))
        self.assertEqual(ap.carrier_frequency_mhz(3601, "LTE"), (940.1, "B8"))
        with self.assertRaises(ap.NotTestable):
            ap.carrier_frequency_mhz(99999, "LTE")

    def test_three_level_pattern_source(self):
        base = {"technology": "LTE", "azimuth": 90, "e_tilt": 4, "m_tilt": 2}
        exact = ap.select_pattern({**base, "frequency_mhz": 1819.6, "antenna_model": "CCVVPX308"})
        self.assertEqual((exact["pattern_quality"], exact["tilt_error_deg"]), ("EXACT", 0))
        assumed = ap.select_pattern({**base, "frequency_mhz": 1819.6})
        self.assertEqual(assumed["pattern_quality"], "ASSUMED")
        self.assertIn("CCVVPX308", assumed["path"])
        generic = ap.select_pattern({**base, "frequency_mhz": 2350.0})
        self.assertEqual((generic["pattern_quality"], generic["path"]), ("APPROXIMATE", ""))
        unknown = ap.select_pattern({**base, "frequency_mhz": 1819.6, "antenna_model": "NOT-A-MODEL"})
        self.assertEqual(unknown["pattern_quality"], "APPROXIMATE")

    def test_never_uses_an_out_of_band_vendor_file(self):
        for frequency in (940.1, 2141.5, 2350.0, 2625.0):
            for model in ("", "CCVVPX308"):
                chosen = ap.select_pattern({"technology": "LTE", "azimuth": 0, "e_tilt": 4, "m_tilt": 2,
                                            "frequency_mhz": frequency, "antenna_model": model})
                self.assertEqual((chosen["path"], chosen["pattern_quality"]), ("", "APPROXIMATE"))

    def test_nearest_tilt_is_reported(self):
        chosen = ap.select_pattern({"technology": "NR", "azimuth": 0, "e_tilt": 0, "m_tilt": 0, "frequency_mhz": 3500})
        self.assertEqual((chosen["pattern_quality"], chosen["file_e_tilt"], chosen["tilt_error_deg"]), ("ASSUMED", 2.0, 2.0))
        self.assertIn("nearest", chosen["note"])

    def test_missing_inputs_are_not_testable(self):
        for missing in ("azimuth", "e_tilt", "m_tilt", "frequency_mhz"):
            sector = {"technology": "LTE", "azimuth": 0, "e_tilt": 4, "m_tilt": 2, "frequency_mhz": 1819.6, missing: np.nan}
            with self.assertRaises(ap.NotTestable):
                ap.select_pattern(sector)

    def test_profile_is_rotated_by_the_configured_azimuth(self):
        for frequency in (1819.6, 2350.0):
            for azimuth in (0, 95, 250):
                pattern = ap.select_pattern({"technology": "LTE", "azimuth": azimuth, "e_tilt": 4, "m_tilt": 2, "frequency_mhz": frequency})
                peak = int(np.argmax(ap.expected_bins(pattern))) * 10 + 5
                self.assertLessEqual(gap(peak, azimuth), 10)


class DetectorVerdicts(unittest.TestCase):
    def test_normal(self):
        cells, m = make_group([0, 120, 240])
        result = analyse(cells, m)
        self.assertEqual((result["verdict"], result["pattern_quality"]), ("NORMAL", "ASSUMED"))

    def test_pair_swap_keeps_pcis_and_is_probable_with_assumed_pattern(self):
        cells, m = make_group([0, 120, 240], configured_az=[120, 0, 240])
        before = cells.copy(deep=True)
        result = analyse(cells, m, leaving_events(cells.pci, [0, 120, 240]))
        self.assertEqual(result["verdict"], "PROBABLE_SWAP")
        self.assertEqual(result["best_azimuths"], "0.0;120.0;240.0")
        self.assertEqual(result["changed_pcis"], "101;102")
        pd.testing.assert_frame_equal(cells, before)

    def test_exact_pattern_with_handover_support_is_rule_confirmed(self):
        cells, m = make_group([0, 120, 240], configured_az=[120, 0, 240], antenna_model="CCVVPX308")
        self.assertEqual(analyse(cells, m, leaving_events(cells.pci, [0, 120, 240]))["verdict"], "CONFIRMED_SWAP")
        self.assertEqual(analyse(cells, m)["verdict"], "PROBABLE_SWAP")
        self.assertEqual(analyse(cells.assign(match_level="PCI"), m, leaving_events(cells.pci, [0, 120, 240]))["verdict"], "PROBABLE_SWAP")

    def test_generic_pattern_is_never_confirmed(self):
        cells, m = make_group([0, 120, 240], configured_az=[120, 0, 240], earfcn=B40_EARFCN, antenna_model="CCVVPX308")
        result = analyse(cells, m, leaving_events(cells.pci, [0, 120, 240]))
        self.assertEqual((result["verdict"], result["pattern_quality"]), ("PROBABLE_SWAP", "APPROXIMATE"))

    def test_rotation_with_unequal_spacing(self):
        cells, m = make_group([0, 90, 200], configured_az=[90, 200, 0])
        result = analyse(cells, m)
        self.assertIn(result["verdict"], SWAP_VERDICTS)
        self.assertEqual(result["best_azimuths"], "0.0;90.0;200.0")

    def test_single_sector_anomaly_is_not_a_swap(self):
        cells, m = make_group([0, 120, 60], configured_az=[0, 120, 240])
        self.assertEqual(analyse(cells, m)["verdict"], "AZIMUTH_MISMATCH")

    def test_sparse_drive_test_is_not_enough_data(self):
        cells, m = make_group([0, 120, 240], bearings=np.arange(5, 50, 10))
        self.assertEqual(analyse(cells, m)["verdict"], "NOT_ENOUGH_DATA")

    def test_nr_missing_tilt_and_frequency_are_not_testable(self):
        cells, m = make_group([0, 120, 240])
        self.assertEqual(analyse(cells.assign(technology="NR"), m)["verdict"], "NOT_TESTABLE")
        self.assertEqual(analyse(cells.assign(e_tilt=np.nan), m)["verdict"], "NOT_TESTABLE")
        self.assertEqual(analyse(cells.assign(frequency_mhz=np.nan), m)["verdict"], "NOT_TESTABLE")

    def test_noise_only_is_never_a_swap(self):
        cells, m = make_group([0, 120, 240])
        noise = m.assign(rsrp=-90 + np.random.default_rng(3).normal(0, 8, len(m)))
        self.assertNotIn(analyse(cells, noise)["verdict"], SWAP_VERDICTS)

    def test_close_sectors_are_the_same_answer(self):
        cells, m = make_group([0, 15, 180])
        self.assertNotIn(analyse(cells, m)["verdict"], SWAP_VERDICTS)


if __name__ == "__main__":
    unittest.main()
