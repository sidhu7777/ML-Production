"""
One-off explanatory script (not a phase, not saved into the pipeline) -
walks the REAL COST-231 + 3GPP antenna + geo-correction calculation, term
by term, for the exact co-located site the user asked about:
  4G: LA201565_LA201565B2_B2_3_Taiwan   (band 3,  1840 MHz)
  5G: GA20000541_GA20000541mB6_26_78_Taiwan (band 78, real freq 3300 MHz,
      but this pipeline's established convention - confirmed correct by
      the user earlier - uses 2600 MHz + a -2.58dB N78 offset instead of
      the real 3300, as a COST-231-validity-range workaround)

Both sectors share the exact same site coordinates and azimuth (90 deg),
so three target points are built along that bearing at 25m/50m/75m and
every intermediate term of compute_sector_rsrp is printed for both
technologies at each distance, then the real geo-correction pipeline
(clutter + diffraction + indoor, from real clutter/building data) is run
on top, exactly as Phase 17/19/20/21 already do.

Read-only: imports production's own compute_sector_rsrp unmodified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
for p in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit_project210_phase13_beam_check as phase13
import streamlit_project210_phase15_radius_progression as phase15
import test_project210_phase17_full_polygon_geo_dt_comparison as phase17

ident = phase13.load_identity()


def _destination_point(lat, lon, bearing_deg, dist_m):
    R = 6371000.0
    br = np.radians(bearing_deg)
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(dist_m / R) + np.cos(lat1) * np.sin(dist_m / R) * np.cos(br))
    lon2 = lon1 + np.arctan2(
        np.sin(br) * np.sin(dist_m / R) * np.cos(lat1),
        np.cos(dist_m / R) - np.sin(lat1) * np.sin(lat2),
    )
    return np.degrees(lat2), np.degrees(lon2)


def explain(node_cell_id: str, freq_mhz: float, n78_offset: float, distances_m: list[float]) -> None:
    row = ident[ident["Node_Cell_ID"] == node_cell_id].iloc[0]
    site = phase15._row_to_site_dict_fixed(row)
    s_lat, s_lon, s_az = float(row["lat"]), float(row["lon"]), float(row["azimuth"])
    s_etilt, s_mtilt, s_htx, tx_pwr = site["electrical_tilt"], site["mechanical_tilt"], site["antenna_height"], site["tx_power"]

    print(f"\n{'='*90}\n{node_cell_id}  |  freq={freq_mhz} MHz  |  site=({s_lat:.6f},{s_lon:.6f})  az={s_az} "
          f"Etilt={s_etilt} Mtilt={s_mtilt} Height={s_htx}m tx_power={tx_pwr}dBm\n{'='*90}")

    params_common = {"ue_height": 1.5, "k1": 0, "k2": 0, "cable_loss": 2.0, "antenna_gain": 18.0}
    h_rx = params_common["ue_height"]

    for d_target in distances_m:
        p_lat, p_lon = _destination_point(s_lat, s_lon, s_az, d_target)

        d_m = phase15._haversine_m(s_lat, s_lon, p_lat, p_lon)
        d_km = d_m / 1000.0

        a_hm = (1.1 * np.log10(freq_mhz) - 0.7) * h_rx - (1.56 * np.log10(freq_mhz) - 0.8)
        CM = 3.0
        base_PL = 46.3 + 33.9 * np.log10(freq_mhz) - 13.82 * np.log10(s_htx) - a_hm + CM
        slope_term = 44.9 - 6.55 * np.log10(s_htx)
        pathloss = base_PL + slope_term * np.log10(d_km)

        bearing = s_az  # target built exactly on boresight, so bearing == azimuth
        az_diff = 0.0
        elev_angle = np.degrees(np.arctan2(h_rx - s_htx, d_m))
        total_tilt = s_etilt + s_mtilt
        elev_diff = elev_angle + total_tilt
        from tools.lte_prediction.Sector_wise_prediction_code_copy import compute_3gpp_antenna_gain_vectorized
        gain_3gpp = compute_3gpp_antenna_gain_vectorized(az_diff, elev_diff, params_common["antenna_gain"])

        cable_loss = params_common["cable_loss"]
        raw_rsrp = tx_pwr + gain_3gpp - pathloss - cable_loss
        physical_rsrp_lib = phase15.compute_sector_rsrp(site, p_lat, p_lon, freq_mhz, params_common)

        physical_with_offset = raw_rsrp + n78_offset

        print(f"\n--- distance = {d_target}m (actual haversine to target = {d_m:.2f}m) ---")
        print(f"  target point: ({p_lat:.6f}, {p_lon:.6f})")
        print(f"  a_hm (UE height correction term) = (1.1*log10({freq_mhz})-0.7)*{h_rx} - (1.56*log10({freq_mhz})-0.8) = {a_hm:.4f}")
        print(f"  base_PL = 46.3 + 33.9*log10({freq_mhz}) - 13.82*log10({s_htx}) - a_hm + 3.0 = {base_PL:.4f} dB")
        print(f"  slope_term = 44.9 - 6.55*log10({s_htx}) = {slope_term:.4f}")
        print(f"  pathloss = base_PL + slope_term*log10({d_km:.5f}) = {pathloss:.4f} dB")
        print(f"  bearing_to_point={bearing:.2f} az_diff={az_diff:.2f} elev_angle={elev_angle:.4f} "
              f"total_tilt={total_tilt} elev_diff={elev_diff:.4f}")
        print(f"  gain_3gpp (3GPP pattern, max 18dBi) = {gain_3gpp:.4f} dBi")
        print(f"  raw_rsrp = tx_pwr({tx_pwr}) + gain({gain_3gpp:.4f}) - pathloss({pathloss:.4f}) - cable_loss({cable_loss}) = {raw_rsrp:.4f} dBm")
        if n78_offset:
            print(f"  + N78 technology offset ({n78_offset:+.2f}dB, established pipeline convention) = {physical_with_offset:.4f} dBm")
        print(f"  [sanity check] production compute_sector_rsrp() library call directly = {physical_rsrp_lib:.4f} dBm (should equal raw_rsrp above)")

        clutter_gdf, buildings_gdf = phase17._load_clutter_and_buildings()
        grid_df = pd.DataFrame({"lat": [p_lat], "lon": [p_lon]})
        correction, counts = phase15._geo_correction_db(
            grid_df, clutter_gdf, buildings_gdf, s_lat, s_lon,
            tx_height_m=s_htx, rx_height_m=h_rx, freq_mhz=freq_mhz,
            clutter_weights=dict(phase15.DEFAULT_CLUTTER_WEIGHTS), building_area_weight=phase15.DEFAULT_BUILDING_AREA_WEIGHT,
            diffraction_multiplier=1.0, entry_loss_db=-15.0, entry_depth_slope_db_per_m=-0.5,
        )
        branch = "indoor" if counts["indoor"] else ("obstructed" if counts["obstructed"] else "clear")
        final_rsrp = physical_with_offset + correction[0]
        print(f"  geo_correction = {correction[0]:+.4f} dB  (branch={branch})")
        print(f"  FINAL predicted RSRP (physical + n78_offset + geo_correction) = {final_rsrp:.4f} dBm")


explain("LA201565_LA201565B2_B2_3_Taiwan", freq_mhz=1840.0, n78_offset=0.0, distances_m=[25, 50, 75])
explain("GA20000541_GA20000541mB6_26_78_Taiwan", freq_mhz=2600.0, n78_offset=phase17.N78_TECHNOLOGY_OFFSET_DB, distances_m=[25, 50, 75])
