import pandas as pd

from tools.lte_prediction.geo_correction_pipeline import _derive_clutter_class
from tests.baseline.push_rf_debug_baseline_to_db import _build_baseline_payload


def test_derive_clutter_class_assigns_multiple_labels():
    df = pd.DataFrame(
        {
            "building_count": [0, 3, 8, 15],
            "building_area_ratio": [0.0, 0.05, 0.18, 0.35],
            "road_length_m": [0.0, 40.0, 120.0, 260.0],
            "green_ratio": [0.0, 0.1, 0.0, 0.0],
            "water_ratio": [0.0, 0.0, 0.0, 0.0],
        }
    )
    clutter = _derive_clutter_class(df)
    assert len(clutter) == len(df)
    assert clutter.nunique() >= 2


def test_build_baseline_payload_prefers_calibrated_geo_kpis_and_keeps_rf_context():
    pred_df = pd.DataFrame(
        {
            "Node_Cell_ID": ["11625_1"],
            "lat": [28.63],
            "lon": [77.35],
            "pred_rsrp": [-100.0],
            "pred_rsrq": [-12.0],
            "pred_sinr": [-5.0],
            "pred_rsrp_geo": [-88.5],
            "pred_rsrq_geo": [-7.2],
            "pred_sinr_geo": [4.3],
            "serving_pci": ["242"],
            "serving_earfcn": ["1750"],
            "best_interferer_cell_id": ["922266_1"],
            "best_interferer_pci": ["460"],
            "neighbor_1_pci": ["460"],
            "neighbor_2_pci": ["461"],
            "interference_gap_db": [7.5],
            "sinr_proxy_db": [3.8],
            "rsrq_proxy_db": [-8.1],
        }
    )
    site_df = pd.DataFrame(
        {
            "Node_Cell_ID": ["11625_1"],
            "nodeb_id": ["11625"],
            "site": ["11625"],
            "operator": ["Airtel"],
        }
    )
    out = _build_baseline_payload(
        pred_df=pred_df,
        site_df=site_df,
        project_id=196,
        job_id="rf_debug_196_test",
        operator="Airtel",
    )
    assert len(out) == 1
    assert out.loc[0, "pred_rsrp"] == -88.5
    assert out.loc[0, "pred_rsrq"] == -7.2
    assert out.loc[0, "pred_sinr"] == 4.3
    assert out.loc[0, "serving_pci"] == "242"
    assert out.loc[0, "best_interferer_pci"] == "460"
    assert out.loc[0, "neighbor_2_pci"] == "461"
    assert out.loc[0, "interference_gap_db"] == 7.5
