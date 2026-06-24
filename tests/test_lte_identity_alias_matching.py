import pandas as pd

from tools.lte_prediction_optimised import ml_engine as opt_ml
from tools.lte_prediction_optimised.services import LTEPredictionService_optimised


def test_k1k2_calibration_matches_site_and_baseline_aliases(monkeypatch):
    monkeypatch.setattr(opt_ml, "calibrate_site", lambda *_args, **_kwargs: (1.25, 4.5))
    baseline = pd.DataFrame(
        {
            "Node_Cell_ID": ["11625_11625_2"] * 12,
            "canonical_cell_id": ["11625_2"] * 12,
            "lat": [12.0] * 12,
            "lon": [77.0] * 12,
            "pred_rsrp": [-91.0] * 12,
        }
    )
    site = pd.DataFrame(
        {
            "Node_Cell_ID": ["11625_11625_2"],
            "cell_id": ["2"],
            "site": ["11625"],
            "sector": ["2"],
            "frequency_mhz": [1800.0],
            "tx_power": [46.0],
            "lat": [12.0],
            "lon": [77.0],
            "azimuth": [120.0],
            "electrical_tilt": [5.0],
            "mechanical_tilt": [0.0],
            "antenna_height": [30.0],
        }
    )

    k1k2 = opt_ml.compute_k1k2_for_cells(baseline, site, ["11625_2"])

    assert k1k2["11625_2"] == (1.25, 4.5)


def test_prediction_point_alias_mask_matches_canonical_and_rf_ids():
    points = pd.DataFrame(
        {
            "Node_Cell_ID": ["11625_11625_2"],
            "rf_identity_key": ["11625_11625_2"],
            "canonical_cell_id": ["11625_2"],
            "cell_id": ["2"],
            "lat": [12.0],
            "lon": [77.0],
        }
    )

    assert int(opt_ml._identity_match_mask(points, "11625_2").sum()) == 1
    assert int(opt_ml._identity_match_mask(points, "11625_11625_2").sum()) == 1


def test_empty_rf_identity_key_does_not_overwrite_node_cell_id():
    points = pd.DataFrame(
        {
            "Node_Cell_ID": ["11625_11625_2", "12248_12248_1"],
            "rf_identity_key": [pd.NA, pd.NA],
            "lat": [12.0, 12.1],
            "lon": [77.0, 77.1],
        }
    )

    normalized = opt_ml._ensure_canonical_identity(points)

    assert normalized["Node_Cell_ID"].tolist() == ["11625_11625_2", "12248_12248_1"]
    assert normalized["Node_Cell_ID"].nunique() == 2


def test_allowed_cell_filter_keeps_site_rows_matching_canonical_alias():
    site = pd.DataFrame(
        {
            "Node_Cell_ID": ["11625_2"],
            "cell_id": ["2"],
            "site": ["11625"],
            "sector": ["2"],
            "frequency_mhz": [1800.0],
            "tx_power": [46.0],
            "lat": [12.0],
            "lon": [77.0],
            "azimuth": [120.0],
            "electrical_tilt": [5.0],
            "mechanical_tilt": [0.0],
            "antenna_height": [30.0],
        }
    )

    expanded = opt_ml._expand_site_rows_to_allowed_cells(site, ["11625_11625_2"])

    assert not expanded.empty
    assert set(expanded["Node_Cell_ID"].astype(str)) == {"11625_2"}


def test_format_for_db_derives_canonical_identity_from_raw_node_cell():
    service = LTEPredictionService_optimised.__new__(LTEPredictionService_optimised)
    df = pd.DataFrame(
        {
            "lat": [12.0],
            "lon": [77.0],
            "pred_rsrp": [-90.0],
            "pred_rsrq": [-10.0],
            "pred_sinr": [5.0],
            "nodeb_id_cell_id": ["11625|11625_2"],
            "Node_Cell_ID": ["11625_11625_2"],
            "node_b_id": [pd.NA],
            "cell_id": [pd.NA],
            "canonical_cell_id": [pd.NA],
            "site_id": [pd.NA],
        }
    )

    out = service._format_for_db(df, project_id=270, job_id="job", operator="Airtel", scenario_id=1, public_scenario_id=3)

    assert out.loc[0, "nodeb_id_cell_id"] == "11625_2"
    assert out.loc[0, "cell_id"] == "2"
    assert out.loc[0, "node_b_id"] == "11625"
    assert out.loc[0, "site_id"] == "11625"


def test_optimized_prediction_returns_empty_when_strict_points_have_no_match():
    site = pd.DataFrame(
        {
            "Node_Cell_ID": ["11625_11625_2"],
            "cell_id": ["2"],
            "site": ["11625"],
            "sector": ["2"],
            "frequency_mhz": [1800.0],
            "tx_power": [46.0],
            "lat": [12.0],
            "lon": [77.0],
            "azimuth": [120.0],
            "electrical_tilt": [5.0],
            "mechanical_tilt": [0.0],
            "antenna_height": [30.0],
            "optimization_applied": [True],
            "orig_electrical_tilt": [4.0],
        }
    )
    params = {
        "project_id": 270,
        "region": "india",
        "radius": 500.0,
        "grid_resolution": 25.0,
        "impact_radius_m": 500.0,
        "neighbor_site_count": 2,
        "max_interference_sites": 10,
        "strict_prediction_points": True,
        "recompute_cells": ["11625_11625_2"],
        "prediction_points_df": pd.DataFrame(
            {
                "Node_Cell_ID": ["other_cell"],
                "lat": [12.0],
                "lon": [77.0],
            }
        ),
        "geo_features_df": pd.DataFrame(),
    }

    out = opt_ml.run_prediction_only_optimized(site, {"11625_11625_2": (1.0, 2.0)}, params)

    assert out.empty
    assert {"lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr", "Node_Cell_ID"}.issubset(out.columns)
