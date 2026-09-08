import unittest
from unittest.mock import patch
import pandas as pd
from tools.lte_prediction_optimised import ml_engine as engine
from tools.lte_prediction_optimised.services import (
    _build_site_prediction_update_rows, LTEPredictionService_optimised,
)


class RecommendationRecoveryTests(unittest.TestCase):
    def test_strict_baseline_scope_never_generates_fallback_grid(self):
        frame = pd.DataFrame([{'Node_Cell_ID': 'A_X_1_78', 'Technology': '5G'}])
        with patch.object(engine, '_normalize_site_df', return_value=frame), \
             patch.object(engine, '_ensure_canonical_identity', side_effect=lambda value: value), \
             patch.object(engine, '_manual_physical_context', return_value=(None, None, None)), \
             patch.object(engine, '_build_local_interference_records', return_value=frame.to_dict('records')), \
             patch.object(engine, '_restore_original_site_state', return_value=frame), \
             patch.object(engine, '_baseline_points_for_cells', return_value=pd.DataFrame()), \
             patch.object(engine, '_location_change_summary', return_value=(False, 0, 0, 0, 0, 0)), \
             patch.object(engine, '_generated_points_for_cell') as generate:
            result = engine.run_prediction_only_offset_manual(frame, {}, {
                'recompute_cells': ['A_X_1_78'], 'baseline_df': pd.DataFrame(),
                'strict_prediction_points': True,
            })
        self.assertTrue(result.empty)
        generate.assert_not_called()

    def test_site_save_preserves_full_cell_identity(self):
        sites = pd.DataFrame([
            {'id': 1, 'Node_Cell_ID': 'SITE1_CELL1_16_78', 'electrical_tilt': 7},
            {'id': 2, 'Node_Cell_ID': 'SITE2_CELL2_16_78', 'electrical_tilt': 3},
        ])
        applied = pd.DataFrame([{'status': 'applied', 'matched_node_cell_id': 'SITE1_CELL1_16_78_Taiwan'}])
        rows = _build_site_prediction_update_rows(210, 1, sites, applied)
        self.assertEqual([r['source_id'] for r in rows], [1])

    def test_out_of_range_slot_rejected_before_database_access(self):
        for slot in [0, 7, -1]:
            with self.assertRaisesRegex(ValueError, 'between 1 and 6'):
                LTEPredictionService_optimised()._create_scenario(
                    {'requested_public_scenario_id': slot}, 'test', 'taiwan')

    def test_repeated_identity_fallback_matches_rowwise_reference(self):
        frame = pd.DataFrame({
            'Node_Cell_ID': ['A_X_1_78', 'A_X_1_78', 'B_Y_1_78'],
            'site': ['A', 'A', 'B'], 'cell_id': ['X', 'X', 'Y'],
            'sector': [None, None, None],
        })
        for target in ['missing_long_cell_id', 'A_X', 'B_Y']:
            aliases = {target} if engine._identity_specificity(target) >= 3 else {target, engine.canonical_cell_id(target)}
            expected = frame.apply(lambda row: bool(engine._row_identity_aliases(row) & aliases), axis=1)
            pd.testing.assert_series_equal(engine._identity_match_mask(frame, target), expected)


if __name__ == '__main__':
    unittest.main()
