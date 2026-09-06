"""Phase 33: isolated 5G 3GPP TR 38.901 UMa baseline at the physical 3300 MHz carrier.

Phase 33 changes only the 5G path-loss layer used by the Phase 31 comparison:

* COST-231 at 2600 MHz plus the -2.58 dB approximation is replaced by 3GPP
  TR 38.901 UMa at the source carrier frequency (3300 MHz).
* The stored Etilt/Mtilt values are tenths of a degree, so this phase uses
  Etilt/10 and Mtilt/10 consistently.
* Kathrein 800109221 PAP data is used only for supported 2..9 degree tilts.
  The 0/1 degree rows retain the standard 3GPP 18 dBi antenna as an explicit
  ``generic_fallback``; they are never silently mapped to the 2 degree file.

Terrain, water handling, O2I, the train/validation split, and the residual
fit follow Phase 28/31.  No earlier phase or production code is modified.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
for path in (ML_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import test_project210_phase28_4g_rsrp_reference_fix as phase28
import test_project210_phase29_real_antenna_pattern as phase29
import test_project210_phase31_phase28_real_antenna as phase31

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE26_DIR = PROJECT_DIR / "cost231_phase26_corrected_obstruction_profile"
PHASE31_DIR = PROJECT_DIR / "cost231_phase31_phase28_real_antenna"
OUT_DIR = PROJECT_DIR / "cost231_phase33_5g_38_901"

FC_GHZ = 3.3
UE_HEIGHT_M = 1.5
CABLE_LOSS_DB = 2.0
KATHREIN_MIN_ETILT_DEG = 2
KATHREIN_MAX_ETILT_DEG = 12


def _uma_pathloss_db(distance_m, h_bs_m, los) -> np.ndarray:
    """3GPP TR 38.901 UMa outdoor path loss.

    The deterministic h_E=1 m case is used for a 1.5 m outdoor UE.  This is
    the applicable low-UE-height branch of the standard's UMa breakpoint model.
    ``los`` is supplied by the existing path-obstruction classifier: clear is
    LOS, obstructed is NLOS.  Indoor remains outdoor-LOS plus the separate O2I
    term already present in Phase 28.
    """
    d2d = np.maximum(np.asarray(distance_m, dtype=float), 10.0)
    hbs = np.maximum(np.asarray(h_bs_m, dtype=float), UE_HEIGHT_M + 1.0)
    d3d = np.sqrt(d2d**2 + (hbs - UE_HEIGHT_M) ** 2)
    h_bs_eff = hbs - 1.0
    h_ut_eff = UE_HEIGHT_M - 1.0
    d_bp = 4.0 * h_bs_eff * h_ut_eff * FC_GHZ * 1e9 / 299792458.0
    pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * math.log10(FC_GHZ)
    pl2 = (
        28.0
        + 40.0 * np.log10(d3d)
        + 20.0 * math.log10(FC_GHZ)
        - 9.0 * np.log10(d_bp**2 + (hbs - UE_HEIGHT_M) ** 2)
    )
    pl_los = np.where(d2d <= d_bp, pl1, pl2)
    pl_nlos_prime = 13.54 + 39.08 * np.log10(d3d) + 20.0 * math.log10(FC_GHZ)
    return np.where(np.asarray(los, dtype=bool), pl_los, np.maximum(pl_los, pl_nlos_prime))


def _generic_gain(df: pd.DataFrame) -> np.ndarray:
    distance = np.maximum(pd.to_numeric(df["distance_m"], errors="coerce").to_numpy(float), 1.0)
    htx = pd.to_numeric(df["Height"], errors="coerce").fillna(25.0).to_numpy(float)
    etilt = pd.to_numeric(df["Etilt"], errors="coerce").fillna(30.0).to_numpy(float) / 10.0
    mtilt = pd.to_numeric(df["Mtilt"], errors="coerce").fillna(0.0).to_numpy(float) / 10.0
    az_off = np.abs(pd.to_numeric(df["azimuth_delta_deg"], errors="coerce").fillna(0.0).to_numpy(float))
    elevation = np.degrees(np.arctan2(UE_HEIGHT_M - htx, distance))
    return phase29._generic_3gpp_gain(az_off, elevation + etilt + mtilt)


def _kathrein_or_generic_gain(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute antenna gain and its per-row source without tilt substitution."""
    generic = _generic_gain(df)
    etilt = pd.to_numeric(df["Etilt"], errors="coerce").fillna(30.0).to_numpy(float) / 10.0
    mtilt = pd.to_numeric(df["Mtilt"], errors="coerce").fillna(0.0).to_numpy(float) / 10.0
    az_off = np.abs(pd.to_numeric(df["azimuth_delta_deg"], errors="coerce").fillna(0.0).to_numpy(float))
    distance = np.maximum(pd.to_numeric(df["distance_m"], errors="coerce").to_numpy(float), 1.0)
    htx = pd.to_numeric(df["Height"], errors="coerce").fillna(25.0).to_numpy(float)
    depression = -np.degrees(np.arctan2(UE_HEIGHT_M - htx, distance)) + mtilt

    rounded = np.round(etilt).astype(int)
    supported = (rounded >= KATHREIN_MIN_ETILT_DEG) & (rounded <= KATHREIN_MAX_ETILT_DEG)
    gain = generic.copy()
    source = np.where(supported, "Kathrein 800109221", "generic_fallback").astype(object)
    for tilt in np.unique(rounded[supported]):
        selected = np.where(supported & (rounded == tilt))[0]
        pattern = phase29.PAT_DIR / "K800109221" / f"3300 - 3590 MHz, eTilt {tilt}, Y1P45 - Port1.pap"
        if not pattern.is_file():
            source[selected] = "generic_fallback_pattern_missing"
            continue
        hs, h, vs, v = phase29._pap_cached(str(pattern))
        h_gain = phase29._pat_gain(hs, np.asarray(h), az_off[selected])
        v_gain = phase29._pat_gain(vs, np.asarray(v), depression[selected])
        gain[selected] = phase29.BORESIGHT_GAIN_DBI["K800109221_3300"] + h_gain + v_gain
    return gain, source


def _geometry_for_dt(dt: pd.DataFrame) -> pd.DataFrame:
    out = dt.copy()
    slat = pd.to_numeric(out["site_lat"], errors="coerce").to_numpy(float)
    slon = pd.to_numeric(out["site_lon"], errors="coerce").to_numpy(float)
    rlat = pd.to_numeric(out["lat"], errors="coerce").to_numpy(float)
    rlon = pd.to_numeric(out["lon"], errors="coerce").to_numpy(float)
    cos0 = np.cos(np.radians(np.nanmean(rlat)))
    out["distance_m"] = np.maximum(np.sqrt(((slon - rlon) * 111320.0 * cos0) ** 2 + ((slat - rlat) * 110540.0) ** 2), 1.0)
    bearing = phase29._bearing_deg(slat, slon, rlat, rlon)
    azimuth = pd.to_numeric(out["azimuth"], errors="coerce").fillna(0.0).to_numpy(float)
    out["azimuth_delta_deg"] = np.abs((bearing - azimuth + 180.0) % 360.0 - 180.0)
    return out


def _physical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    branch = out["obstruction_branch"].astype(str)
    # The outdoor path state feeds UMa. Indoor adds its separate O2I term;
    # it is not counted again as UMa-NLOS simply because the receiver is indoors.
    los = ~branch.eq("obstructed")
    pathloss = _uma_pathloss_db(
        pd.to_numeric(out["distance_m"], errors="coerce").to_numpy(float),
        pd.to_numeric(out["Height"], errors="coerce").fillna(25.0).to_numpy(float),
        los.to_numpy(bool),
    )
    gain, source = _kathrein_or_generic_gain(out)
    power = pd.to_numeric(out["tx_power"], errors="coerce").fillna(50.0).to_numpy(float)
    out["phase33_frequency_mhz"] = 3300.0
    out["phase33_pathloss_model"] = np.where(los, "3GPP 38.901 UMa LOS", "3GPP 38.901 UMa NLOS")
    out["phase33_pathloss_db"] = pathloss
    out["phase33_antenna_gain_db"] = gain
    out["phase33_antenna_source"] = source
    out["phase33_raw_rsrp"] = power + gain - pathloss - CABLE_LOSS_DB

    terrain = pd.to_numeric(out["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0).clip(lower=0.0)
    bgc = pd.to_numeric(out["building_geo_correction_db"], errors="coerce")
    depth = (-(bgc + 15.0) / 0.5).clip(lower=0.0, upper=40.0)
    o2i = np.array([phase28._indoor_o2i_db(3300.0, value) for value in depth])
    out["phase33_terrain_db"] = terrain
    out["phase33_o2i_db"] = np.where(branch.eq("indoor"), o2i, 0.0)
    water = out["clutter_class"].astype(str).eq("Water")
    out.loc[water, ["phase33_terrain_db", "phase33_o2i_db"]] = 0.0
    out["phase33_physical_rsrp"] = out["phase33_raw_rsrp"] - out["phase33_terrain_db"] - out["phase33_o2i_db"]
    return out


def _cdf(ax, label: str, data: pd.Series, color: str) -> None:
    values = pd.to_numeric(data, errors="coerce").dropna().sort_values().to_numpy(float)
    if len(values):
        ax.plot(values, np.arange(1, len(values) + 1) * 100.0 / len(values), label=f"{label} (n={len(values):,})", color=color)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(PHASE26_DIR / "phase26_scored_candidates_project210.parquet")
    dt_all = pd.read_parquet(PHASE26_DIR / "phase26_dt_scored_project210.parquet")
    grid = pd.read_parquet(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210.parquet")
    bounds = grid[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]]

    c = candidates[candidates["technology"].astype(str).eq("5G")].copy()
    dt = dt_all[dt_all["assigned_technology"].astype(str).eq("5G")].copy()
    dt["technology"] = "5G"
    dt = _geometry_for_dt(dt)
    c = _physical(c)
    dt = _physical(dt)
    dt["resid"] = pd.to_numeric(dt["rsrp_measured"], errors="coerce") - dt["phase33_physical_rsrp"]
    split_hash = pd.util.hash_pandas_object(dt["dt_row_id"].astype(str), index=False).astype("uint64")
    dt["split"] = np.where((split_hash % 10) < 7, "train", "validation")
    dt["clutter_class"] = dt["clutter_class"].astype("object").where(dt["clutter_class"].notna(), "UNKNOWN")
    train = dt[dt["split"].eq("train")]
    train_fit = train[~train["clutter_class"].astype(str).eq("Water")]
    global_correction = float(train_fit["resid"].median())
    table = phase28._shrink_residual_table(train_fit.assign(_r=train_fit["resid"] - global_correction), ["clutter_class"], "_r", 40.0)
    correction = dict(zip(table["clutter_class"].astype(str), table["correction_db"]))

    def residual_for(clutter: pd.Series) -> np.ndarray:
        keys = clutter.astype(str)
        result = global_correction + keys.map(lambda item: correction.get(item, 0.0)).to_numpy(float)
        return np.where(keys.to_numpy() == "Water", 0.0, result)

    c["phase33_residual_db"] = residual_for(c["clutter_class"].astype("object").where(c["clutter_class"].notna(), "UNKNOWN"))
    dt["phase33_residual_db"] = residual_for(dt["clutter_class"])
    c["phase33_final_rsrp_unclipped"] = c["phase33_physical_rsrp"] + c["phase33_residual_db"]
    c["phase33_final_rsrp"] = c["phase33_final_rsrp_unclipped"].where(c["phase33_final_rsrp_unclipped"] >= -140.0, np.nan)
    dt["phase33_final_rsrp"] = dt["phase33_physical_rsrp"] + dt["phase33_residual_db"]

    best = c.sort_values("phase33_final_rsrp_unclipped").groupby("grid_id").tail(1)
    serving = bounds.merge(best[["grid_id", "phase33_physical_rsrp", "phase33_final_rsrp", "phase33_final_rsrp_unclipped", "phase33_antenna_source", "phase33_pathloss_model", "phase33_o2i_db", "phase33_terrain_db", "phase33_residual_db"]], on="grid_id", how="left")
    p31_5g = pd.read_parquet(PHASE31_DIR / "phase31_serving_grid_5g_project210.parquet")
    p31_4g = pd.read_parquet(PHASE31_DIR / "phase31_serving_grid_4g_project210.parquet")
    serving = serving.merge(p31_5g[["grid_id", "phase31_final_best_rsrp"]], on="grid_id", how="left")

    validation = dt[dt["split"].eq("validation")]
    validation_outdoor = validation[~validation["obstruction_branch"].astype(str).eq("indoor")]
    p31_dt = pd.read_parquet(PHASE31_DIR / "phase31_dt_scored_5g_project210.parquet")
    p31_validation = p31_dt[(p31_dt["split"].astype(str).eq("validation")) & (~p31_dt["obstruction_branch"].astype(str).eq("indoor"))]
    p31_4g_summary = json.loads((PHASE31_DIR / "phase31_summary.json").read_text(encoding="utf-8"))["technology"]["4G"]

    c.to_parquet(OUT_DIR / "phase33_5g_scored_candidates_project210.parquet", index=False)
    dt.to_parquet(OUT_DIR / "phase33_5g_dt_scored_project210.parquet", index=False)
    serving.to_parquet(OUT_DIR / "phase33_5g_serving_grid_project210.parquet", index=False)
    c.to_csv(OUT_DIR / "phase33_5g_scored_candidates_project210.csv", index=False)
    dt.to_csv(OUT_DIR / "phase33_5g_dt_scored_project210.csv", index=False)
    serving.to_csv(OUT_DIR / "phase33_5g_serving_grid_project210.csv", index=False)
    table.to_csv(OUT_DIR / "phase33_5g_clutter_residuals.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 7))
    _cdf(ax, "5G DT measured", validation_outdoor["rsrp_measured"], "#111827")
    _cdf(ax, "Phase 31 5G (COST-231 approximation)", p31_validation["phase31_rsrp"], "#d97706")
    _cdf(ax, "Phase 33 5G (3GPP 38.901 at 3300 MHz)", validation_outdoor["phase33_final_rsrp"], "#2563eb")
    ax.set(title="Project 210 5G: held-out outdoor DT comparison", xlabel="RSRP (dBm)", ylabel="Cumulative %")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase33_5g_heldout_cdf.png", dpi=170)
    plt.close(fig)

    p33_metric = phase28._metrics(validation_outdoor["rsrp_measured"], validation_outdoor["phase33_final_rsrp"])
    p31_metric = phase28._metrics(p31_validation["rsrp_measured"], p31_validation["phase31_rsrp"])
    antenna_counts = c["phase33_antenna_source"].value_counts().to_dict()
    model_counts = c["phase33_pathloss_model"].value_counts().to_dict()
    summary = {
        "scope": "5G only; isolated Phase 33. 4G is read-only Phase 31 control.",
        "path_loss": {
            "model": "3GPP TR 38.901 UMa",
            "frequency_mhz": 3300.0,
            "los_rule": "clear and indoor -> UMa LOS outdoor path; obstructed -> UMa NLOS; indoor O2I is applied separately once",
            "tilt_units": "database Etilt/Mtilt are tenths of degree and are divided by 10",
        },
        "antenna": {
            "Kathrein_800109221": "used only for 2..9 degree tilts where the PAP file exists",
            "generic_fallback": "0/1 degree tilts use the standard 18 dBi, 65 degree horizontal, 6 degree vertical 3GPP antenna with the actual tilt",
            "candidate_rows_by_source": {str(k): int(v) for k, v in antenna_counts.items()},
        },
        "no_smoothing_or_delta_clip": True,
        "residual": {
            "fit": "70/30 held-out DT split; water excluded and receives zero residual",
            "global_db": round(global_correction, 3),
            "per_clutter_db": {str(k): round(float(v), 3) for k, v in correction.items()},
        },
        "5g_held_out_outdoor_dt": {
            "phase31_cost231_approximation": p31_metric,
            "phase33_3gpp_38_901": p33_metric,
        },
        "5g_candidate_path_states": {str(k): int(v) for k, v in model_counts.items()},
        "5g_serving_grid": {
            "rows": int(len(serving)),
            "no_coverage": int(serving["phase33_final_rsrp"].isna().sum()),
            "median_rsrp": float(pd.to_numeric(serving["phase33_final_rsrp"], errors="coerce").median()),
        },
        "4g_read_only_phase31_control": p31_4g_summary["held_out_outdoor_dt"]["phase31_real_antenna"],
    }
    (OUT_DIR / "phase33_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
