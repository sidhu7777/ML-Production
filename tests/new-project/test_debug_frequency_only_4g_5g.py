"""
DEBUG comparison (NOT a pipeline phase): isolate CARRIER FREQUENCY as the only
thing that differs between the 4G and the 5G prediction, across the WHOLE real
Taiwan network.

How the confound is removed
---------------------------
In this project every 5G (n78) cell is co-located with a 4G cell: same site
coordinates, same (or near-same) azimuth. They are separate rows in the
identity table (`GA...` = 5G, `LA...` = 4G) and differ slightly in tilt / tx
power / exact azimuth. Those per-pair differences - plus 5G's higher tx power -
are exactly what makes "5G looks better than 4G" when the two surfaces are
compared as-is.

This script pairs each 5G cell with its co-located 4G cell (by location, then
nearest azimuth) and gives BOTH cells of the pair ONE shared parameter set,
taken from the 4G (`LA`) side: location, azimuth, antenna height, electrical
tilt, mechanical tilt, tx power, antenna gain, cable loss. Every site keeps its
OWN real values (nothing is forced to a single global number) - only the two
halves of a co-located pair are made identical.

After that the ONLY difference between the 4G run and the 5G run, for every
pair, is the carrier frequency:
    4G : the cell's real frequency  (band 3 = 1840 MHz, band 28 = 775.5 MHz)
    5G : 3300 MHz nominal, modelled as 2600 MHz + (-2.58 dB) N78 offset
         (the established COST-231 validity-range convention for this project)

Model is identical to the real pipeline: production COST-231 + 3GPP antenna
(compute_sector_rsrp) + Phase 15 branch geo-correction. Two RSRP surfaces are
produced per technology:
    rsrp_freq_only : physical + geo-correction  (the like-for-like comparison)
    rsrp_phase19   : + Phase 18/19 branch-calibrated DT bias  (kept only for
                     completeness; its 5G rows come from pre-Phase-20 5G DT
                     and are large + positive, so it does NOT isolate frequency)

Candidate cell lists per grid tile are reused verbatim from Phase 9's own
directional surface (same ~5.7 candidates/tile), so the aggregation matches the
rest of the pipeline.

Products (all under data/debug_frequency_only/, no phase output touched):
  1. Whole-network polygon: every co-located pair, best-server + frontend-mean,
     4G and 5G as separate surfaces, 4G-band tag kept for filtering.
  2. Single sector: the three LA201565 / GA20000541 co-located pairs on a local
     grid, 4G vs 5G.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = ML_ROOT / "tests" / "baseline"
for _p in (ML_ROOT, THIS_DIR, BASELINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit_project210_phase13_beam_check as phase13
import streamlit_project210_phase15_radius_progression as phase15
import test_project210_phase17_full_polygon_geo_dt_comparison as phase17  # read-only
import test_project210_phase19_branch_calibrated_comparison as phase19  # read-only

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
OUT_DIR = THIS_DIR / "data" / "debug_frequency_only"

RSRP_MIN, RSRP_MAX = phase17.RSRP_MIN, phase17.RSRP_MAX
N78_OFFSET_DB = phase17.N78_TECHNOLOGY_OFFSET_DB  # -2.58
FREQ_5G_MODEL_MHZ = 2600.0
FREQ_5G_NOMINAL_MHZ = 3300.0
PARAMS_COMMON = {"ue_height": 1.5, "k1": 0, "k2": 0, "cable_loss": 2.0, "antenna_gain": 18.0}

# geo-correction (ray tracing) only where the bare physical value is still
# inside a realistic coverage range - elsewhere the cell is at the floor and a
# clutter/diffraction term changes nothing, and it can never be best server.
GEO_GATE_DBM = -135.0

BANDS_4G = {3, 28}
BAND_5G = 78
FIXED_PARAM_SIDE = "4G"  # each co-located pair uses its 4G (LA) cell's params

SINGLE_SECTOR_SITE_4G = "LA201565"  # canonical band-3 pair used all through the phases
LOCAL_RADIUS_M = 1800.0
LOCAL_RES_M = 30.0


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _build_pairs(identity: pd.DataFrame) -> pd.DataFrame:
    """One row per co-located 4G/5G pair. Shared params come from the 4G side."""
    ident = _num(identity.copy(), ["lat", "lon", "azimuth", "Height", "Etilt", "Mtilt", "tx_power", "band"])
    ident["loc"] = list(zip(ident["lat"].round(5), ident["lon"].round(5)))
    four = ident[ident["band"].isin(BANDS_4G)]
    five = ident[ident["band"] == BAND_5G]

    rows = []
    for loc, grp4 in four.groupby("loc"):
        grp5 = five[five["loc"] == loc]
        if grp5.empty:
            continue
        left = grp5.to_dict("records")
        for _, r4 in grp4.sort_values("azimuth").iterrows():
            if not left:
                break
            j = min(range(len(left)), key=lambda k: abs((left[k]["azimuth"] - r4["azimuth"] + 180) % 360 - 180))
            r5 = left.pop(j)
            rows.append(
                {
                    "pair_id": f"{r4['Node_Cell_ID']}|{r5['Node_Cell_ID']}",
                    "cell_4g": str(r4["Node_Cell_ID"]),
                    "cell_5g": str(r5["Node_Cell_ID"]),
                    "site_4g": str(r4["site"]),
                    "sector_4g": str(r4["sector"]),
                    "band_4g": int(r4["band"]),
                    "freq_4g_mhz": 1840.0 if int(r4["band"]) == 3 else 775.5,
                    "site_lat": float(r4["lat"]),
                    "site_lon": float(r4["lon"]),
                    "azimuth": float(r4["azimuth"]),
                    "Etilt": r4["Etilt"],
                    "Mtilt": r4["Mtilt"],
                    "Height": r4["Height"],
                    "tx_power": r4["tx_power"],
                }
            )
    pairs = pd.DataFrame(rows)
    print(f"[DEBUG-FREQ] built {len(pairs)} co-located pairs "
          f"(band 3: {(pairs['band_4g'] == 3).sum()}, band 28: {(pairs['band_4g'] == 28).sum()})")
    return pairs


def _site_dict(pair_row: pd.Series) -> dict:
    return phase15._row_to_site_dict_fixed(
        pd.Series(
            {
                "lat": pair_row["site_lat"],
                "lon": pair_row["site_lon"],
                "azimuth": pair_row["azimuth"],
                "Etilt": pair_row["Etilt"],
                "Mtilt": pair_row["Mtilt"],
                "Height": pair_row["Height"],
                "tx_power": pair_row["tx_power"],
            }
        )
    )


def _bias_lookup_for(bias_table: pd.DataFrame, technology: str) -> dict:
    sub = bias_table[bias_table["technology"] == technology]
    return {} if sub.empty else sub.set_index(["clutter_class", "obstruction_branch"])["bias_db"].to_dict()


def _predict_group(
    site_dict: dict,
    freq_mhz: float,
    n78_offset_db: float,
    lats: np.ndarray,
    lons: np.ndarray,
    clutter_gdf,
    buildings_gdf,
    bias_lookup: dict,
    center_lat: float,
    center_lon: float,
    tx_height_m: float,
) -> pd.DataFrame:
    """physical + branch geo-correction + branch-calibrated bias for one
    (site, frequency) over a set of target points."""
    n = len(lats)
    physical = np.array(
        [phase15.compute_sector_rsrp(site_dict, float(la), float(lo), freq_mhz, PARAMS_COMMON)
         for la, lo in zip(lats, lons)],
        dtype=float,
    ) + n78_offset_db

    correction = np.zeros(n, dtype=float)
    branch = np.array(["clear"] * n, dtype=object)
    clutter = np.array([""] * n, dtype=object)

    gate = physical >= GEO_GATE_DBM
    if gate.any():
        gdf = pd.DataFrame({"lat": lats[gate], "lon": lons[gate]})
        corr_g, branch_g, cls_g = phase19._geo_correction_with_branch(
            gdf, clutter_gdf, buildings_gdf, center_lat, center_lon,
            tx_height_m=tx_height_m, rx_height_m=1.5, freq_mhz=freq_mhz,
        )
        correction[gate] = corr_g
        branch[gate] = branch_g
        clutter[gate] = [c if isinstance(c, str) else "" for c in cls_g]

    bias = np.array(
        [float(bias_lookup.get((c, b), 0.0)) if c else 0.0 for c, b in zip(clutter, branch)],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "physical_rsrp": np.clip(physical, RSRP_MIN, RSRP_MAX),
            "geo_correction_db": correction,
            "bias_db": bias,
            "obstruction_branch": branch,
            "clutter_class": clutter,
            "rsrp_freq_only": np.clip(physical + correction, RSRP_MIN, RSRP_MAX),
            "rsrp_phase19": np.clip(physical + correction + bias, RSRP_MIN, RSRP_MAX),
        }
    )


def _run_technology(
    technology: str,
    surface: pd.DataFrame,
    pairs: pd.DataFrame,
    clutter_gdf,
    buildings_gdf,
    bias_lookup: dict,
) -> pd.DataFrame:
    """Long-form candidate prediction for one technology over the whole
    polygon, using Phase 9's own candidate lists but the paired/equalised
    site parameters."""
    key_col = "cell_4g" if technology == "4G" else "cell_5g"
    cand = surface[surface["technology"] == technology][["grid_id", "lat", "lon", "strict_cell_key"]].copy()
    cand = cand.merge(pairs, left_on="strict_cell_key", right_on=key_col, how="inner")
    print(f"[DEBUG-FREQ] {technology}: {len(cand)} candidate rows over {cand['grid_id'].nunique()} grid tiles, "
          f"{cand['pair_id'].nunique()} pairs")

    out = []
    groups = list(cand.groupby("pair_id"))
    for i, (pair_id, g) in enumerate(groups):
        pr = g.iloc[0]
        sd = _site_dict(pr)
        freq = float(pr["freq_4g_mhz"]) if technology == "4G" else FREQ_5G_MODEL_MHZ
        offset = 0.0 if technology == "4G" else N78_OFFSET_DB
        pred = _predict_group(
            sd, freq, offset,
            g["lat"].to_numpy(dtype=float), g["lon"].to_numpy(dtype=float),
            clutter_gdf, buildings_gdf, bias_lookup,
            float(pr["site_lat"]), float(pr["site_lon"]), float(pr["Height"]),
        )
        pred.index = g.index
        block = pd.concat(
            [g[["grid_id", "lat", "lon", "pair_id", "cell_4g", "cell_5g", "band_4g", "sector_4g", "azimuth"]], pred],
            axis=1,
        )
        block["frequency_mhz"] = freq
        out.append(block)
        if (i + 1) % 10 == 0 or i == len(groups) - 1:
            print(f"[DEBUG-FREQ] {technology}: {i + 1}/{len(groups)} pairs done", flush=True)
    res = pd.concat(out, ignore_index=True)
    res["technology"] = technology
    return res


def _aggregate(cand: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """serving (best over candidate pairs) + frontend (mean over candidate
    pairs) per grid tile, for both rsrp_freq_only and rsrp_phase19."""
    grp = cand.groupby("grid_id")
    agg = pd.DataFrame(
        {
            "serving_rsrp": grp["rsrp_freq_only"].max(),
            "frontend_rsrp": grp["rsrp_freq_only"].mean(),
            "serving_rsrp_phase19": grp["rsrp_phase19"].max(),
            "frontend_rsrp_phase19": grp["rsrp_phase19"].mean(),
            "n_candidates": grp["pair_id"].nunique(),
        }
    ).reset_index()
    best = cand.loc[grp["rsrp_freq_only"].idxmax()]
    best = best[["grid_id", "pair_id", "sector_4g", "band_4g", "azimuth", "obstruction_branch",
                 "geo_correction_db", "bias_db", "physical_rsrp"]].rename(
        columns={
            "pair_id": "best_pair_id", "sector_4g": "best_sector", "band_4g": "best_band_4g",
            "azimuth": "best_azimuth", "obstruction_branch": "best_branch",
            "geo_correction_db": "best_geo_correction_db", "bias_db": "best_bias_db",
            "physical_rsrp": "best_physical_rsrp",
        }
    )
    out = grid.merge(agg, on="grid_id", how="inner").merge(best, on="grid_id", how="left")
    out["lat"] = out["center_lat"]
    out["lon"] = out["center_lon"]
    return out


def _summ(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "mean": float(v.mean()),
        "median": float(v.median()),
        "pct_ge_minus85": float((v >= -85).mean() * 100.0),
        "pct_ge_minus95": float((v >= -95).mean() * 100.0),
        "pct_ge_minus105": float((v >= -105).mean() * 100.0),
        "pct_ge_minus120": float((v >= -120).mean() * 100.0),
    }


def _summarise_pair_of_candidates(c4: pd.DataFrame, c5: pd.DataFrame) -> dict:
    """Candidate-level mean decomposition of the 5G-minus-4G difference,
    matched on (grid_id, pair_id) so it is a true within-pair delta."""
    m = c4.merge(c5, on=["grid_id", "pair_id"], suffixes=("_4g", "_5g"))
    return {
        "matched_candidate_rows": int(len(m)),
        "physical_cost231_plus_n78": round(float((m["physical_rsrp_5g"] - m["physical_rsrp_4g"]).mean()), 2),
        "geo_correction": round(float((m["geo_correction_db_5g"] - m["geo_correction_db_4g"]).mean()), 2),
        "phase19_dt_bias": round(float((m["bias_db_5g"] - m["bias_db_4g"]).mean()), 2),
        "rsrp_freq_only_delta": round(float((m["rsrp_freq_only_5g"] - m["rsrp_freq_only_4g"]).mean()), 2),
        "rsrp_phase19_delta": round(float((m["rsrp_phase19_5g"] - m["rsrp_phase19_4g"]).mean()), 2),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    identity = phase13.load_identity()
    pairs = _build_pairs(identity)
    pairs.to_parquet(OUT_DIR / "debug_pairs.parquet", index=False)

    surface = pd.read_parquet(PHASE9_DIR / "phase9_directional_raw_corrected_surface_project210.parquet")
    grid = pd.read_parquet(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210.parquet")
    clutter_gdf, buildings_gdf = phase17._load_clutter_and_buildings()
    bias_table = phase19._load_bias_table()

    summary: dict = {
        "method": {
            "pairing": "co-located 4G/5G cells matched by location then nearest azimuth",
            "shared_params_from": FIXED_PARAM_SIDE,
            "n_pairs": int(len(pairs)),
            "n_pairs_band_3": int((pairs["band_4g"] == 3).sum()),
            "n_pairs_band_28": int((pairs["band_4g"] == 28).sum()),
            "unpaired_4g": 0,
            "unpaired_5g": 0,
        },
        "frequency_convention": {
            "4G_band3_mhz": 1840.0,
            "4G_band28_mhz": 775.5,
            "5G_nominal_mhz": FREQ_5G_NOMINAL_MHZ,
            "5G_modelled_as_mhz": FREQ_5G_MODEL_MHZ,
            "5G_n78_offset_db": N78_OFFSET_DB,
        },
        "whole_project": {},
        "single_sector": {},
    }

    cand_by_tech = {}
    for tech in ["4G", "5G"]:
        bl = _bias_lookup_for(bias_table, tech)
        cand = _run_technology(tech, surface, pairs, clutter_gdf, buildings_gdf, bl)
        cand.to_parquet(OUT_DIR / f"debug_candidates_{tech.lower()}.parquet", index=False)
        cand_by_tech[tech] = cand

        serving = _aggregate(cand, grid)
        serving["technology"] = tech
        p = OUT_DIR / f"debug_serving_grid_{tech.lower()}.parquet"
        serving.to_parquet(p, index=False)
        serving.to_csv(p.with_suffix(".csv"), index=False)
        print(f"[DEBUG-FREQ] {tech}: serving mean {serving['serving_rsrp'].mean():.1f} dBm  "
              f"frontend mean {serving['frontend_rsrp'].mean():.1f} dBm  ({len(serving)} tiles)")

        per_band = {}
        for band in sorted(cand["band_4g"].unique()):
            cb = cand[cand["band_4g"] == band]
            sb = _aggregate(cb, grid)
            per_band[f"band_{int(band)}"] = {
                "grid_tiles": int(len(sb)),
                "serving": _summ(sb["serving_rsrp"]),
                "frontend": _summ(sb["frontend_rsrp"]),
            }
        summary["whole_project"][tech] = {
            "grid_tiles": int(len(serving)),
            "serving": _summ(serving["serving_rsrp"]),
            "frontend": _summ(serving["frontend_rsrp"]),
            "serving_phase19": _summ(serving["serving_rsrp_phase19"]),
            "frontend_phase19": _summ(serving["frontend_rsrp_phase19"]),
            "mean_physical_rsrp": float(cand["physical_rsrp"].mean()),
            "mean_geo_correction_db": float(cand["geo_correction_db"].mean()),
            "mean_bias_db": float(cand["bias_db"].mean()),
            "branch_share": {
                str(k): round(float(v), 3)
                for k, v in serving["best_branch"].value_counts(normalize=True).items()
            },
            "by_4g_band": per_band,
        }

    # within-pair 5G - 4G delta on the serving grid + candidate decomposition
    dcols = ["grid_id", "serving_rsrp", "frontend_rsrp", "serving_rsrp_phase19", "frontend_rsrp_phase19"]
    s4 = pd.read_parquet(OUT_DIR / "debug_serving_grid_4g.parquet")[dcols]
    s5 = pd.read_parquet(OUT_DIR / "debug_serving_grid_5g.parquet")[dcols]
    delta = s4.merge(s5, on="grid_id", suffixes=("_4g", "_5g"))
    delta["serving_delta_5g_minus_4g"] = delta["serving_rsrp_5g"] - delta["serving_rsrp_4g"]
    delta["frontend_delta_5g_minus_4g"] = delta["frontend_rsrp_5g"] - delta["frontend_rsrp_4g"]
    delta["serving_delta_5g_minus_4g_phase19"] = delta["serving_rsrp_phase19_5g"] - delta["serving_rsrp_phase19_4g"]
    delta["frontend_delta_5g_minus_4g_phase19"] = delta["frontend_rsrp_phase19_5g"] - delta["frontend_rsrp_phase19_4g"]
    delta = delta.merge(grid, on="grid_id", how="left")
    delta["lat"] = delta["center_lat"]
    delta["lon"] = delta["center_lon"]
    delta.to_parquet(OUT_DIR / "debug_serving_grid_delta.parquet", index=False)

    summary["whole_project"]["delta_5g_minus_4g"] = {
        "mean_serving_delta_db_freq_only": float(delta["serving_delta_5g_minus_4g"].mean()),
        "median_serving_delta_db_freq_only": float(delta["serving_delta_5g_minus_4g"].median()),
        "mean_frontend_delta_db_freq_only": float(delta["frontend_delta_5g_minus_4g"].mean()),
        "mean_serving_delta_db_phase19": float(delta["serving_delta_5g_minus_4g_phase19"].mean()),
        "mean_frontend_delta_db_phase19": float(delta["frontend_delta_5g_minus_4g_phase19"].mean()),
        "within_pair_candidate_decomposition": _summarise_pair_of_candidates(cand_by_tech["4G"], cand_by_tech["5G"]),
    }

    # ---------------- single sector (LA201565 co-located pairs) ----------------
    sec_pairs = pairs[pairs["site_4g"] == SINGLE_SECTOR_SITE_4G].copy()
    if sec_pairs.empty:
        sec_pairs = pairs.head(3).copy()
    center_lat = float(sec_pairs.iloc[0]["site_lat"])
    center_lon = float(sec_pairs.iloc[0]["site_lon"])
    grid_df, lat_step, lon_step = phase13._build_grid(center_lat, center_lon, LOCAL_RADIUS_M, LOCAL_RES_M)
    llats = grid_df["lat"].to_numpy(dtype=float)
    llons = grid_df["lon"].to_numpy(dtype=float)
    print(f"[DEBUG-FREQ] single sector: {len(sec_pairs)} pairs on {SINGLE_SECTOR_SITE_4G}, "
          f"local grid {len(grid_df)} pts")

    for tech in ["4G", "5G"]:
        bl = _bias_lookup_for(bias_table, tech)
        frames = []
        for _, pr in sec_pairs.iterrows():
            sd = _site_dict(pr)
            freq = float(pr["freq_4g_mhz"]) if tech == "4G" else FREQ_5G_MODEL_MHZ
            offset = 0.0 if tech == "4G" else N78_OFFSET_DB
            pred = _predict_group(
                sd, freq, offset, llats, llons, clutter_gdf, buildings_gdf, bl,
                center_lat, center_lon, float(pr["Height"]),
            )
            pred["lat"] = llats
            pred["lon"] = llons
            pred["sector"] = pr["sector_4g"]
            pred["azimuth"] = pr["azimuth"]
            pred["frequency_mhz"] = freq
            frames.append(pred)
        local = pd.concat(frames, ignore_index=True)
        local["technology"] = tech
        local.to_parquet(OUT_DIR / f"debug_local_grid_{tech.lower()}.parquet", index=False)
        for sec in local["sector"].unique():
            sv = local.loc[local["sector"] == sec, "rsrp_freq_only"]
            print(f"[DEBUG-FREQ] {tech} sector {sec}: mean {sv.mean():.1f} dBm  "
                  f"pct>=-95 {(sv >= -95).mean() * 100:.0f}%")

    summary["single_sector"] = {
        "site_4g": SINGLE_SECTOR_SITE_4G,
        "cell_pairs": [
            {"sector": r["sector_4g"], "azimuth": r["azimuth"], "cell_4g": r["cell_4g"], "cell_5g": r["cell_5g"],
             "band_4g": int(r["band_4g"]), "freq_4g_mhz": r["freq_4g_mhz"]}
            for _, r in sec_pairs.iterrows()
        ],
        "shared_params": {
            "antenna_height_m": float(sec_pairs.iloc[0]["Height"]),
            "electrical_tilt_deg": float(sec_pairs.iloc[0]["Etilt"]) / 10.0,
            "mechanical_tilt_deg": float(sec_pairs.iloc[0]["Mtilt"]) / 10.0,
            "tx_power_dbm": float(sec_pairs.iloc[0]["tx_power"]),
            "antenna_gain_dbi": PARAMS_COMMON["antenna_gain"],
            "cable_loss_db": PARAMS_COMMON["cable_loss"],
        },
        "local_grid": {
            "radius_m": LOCAL_RADIUS_M, "resolution_m": LOCAL_RES_M,
            "center_lat": center_lat, "center_lon": center_lon, "n_points": int(len(grid_df)),
        },
    }

    (OUT_DIR / "debug_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DEBUG-FREQ] wrote outputs under {OUT_DIR}")
    print(json.dumps(summary["whole_project"]["delta_5g_minus_4g"], indent=2))


if __name__ == "__main__":
    main()
