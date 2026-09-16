"""Phase 47: does the absolute per-cell membership rule remove the islands?

Test-only. No production file is modified. The RF maths is imported from
production and called unchanged; only the rule deciding WHICH pixels belong to
a cell is replaced.

Arm A - production rule (reproduced exactly)
    membership = (distance <= 500 m)
                 + ensure_all_cells k=8 nearest-sector backfill
                 - competitor-relative filter (raw >= best_raw_at_pixel - 20 dB)

Arm B - absolute rule
    domain     = disc of max(500 m, farthest backfill pixel that arm A gave the
                 cell), clipped to the project grid
    membership = (this cell's own physical value >= -140 dBm)
    no competitor filter, no backfill deciding membership

No smoothing, no morphological closing, no interpolation, no dropping of small
components. Membership only. Components are counted with the same 4-connected
grid_id rule Phase 42 uses.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ML))

from tools.lte_prediction_offset import services as S
from tools.lte_prediction_offset import phase36_physical_upgrades as A

RUN = ML / "tests" / "output" / "baseline_210_profile_20260908_153817"
RADIUS_M = 500.0
BACKFILL_K = 8
MARGIN_DB = 20.0
RAW_MIN_DBM = -145.0
NO_COVERAGE_DBM = -140.0
GRID_RE = re.compile(r"R(\d+)C(\d+)")


def components(grid_ids: pd.Series) -> int:
    """4-connected component count over R#C# grid ids (Phase 42's rule)."""
    coords = set()
    for gid in grid_ids.dropna().astype(str):
        m = GRID_RE.search(gid)
        if m:
            coords.add((int(m.group(1)), int(m.group(2))))
    n = 0
    while coords:
        n += 1
        stack = [coords.pop()]
        while stack:
            r, c = stack.pop()
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in coords:
                    coords.remove(nb)
                    stack.append(nb)
    return n


def antenna_delta(frame: pd.DataFrame, corrected: bool) -> np.ndarray:
    """Production PAP delta. corrected=True fixes the two lookup-key bugs:
    deployed frequency instead of the COST-231 anchor, and tenths-of-a-degree
    tilt converted to degrees."""
    f = pd.DataFrame(index=frame.index)
    f["distance_m"] = frame["distance_m"].to_numpy(float)
    f["azimuth_delta_deg"] = frame["azimuth_delta_deg"].to_numpy(float)
    f["Height"] = frame["Height"].to_numpy(float)
    f["antenna_model"] = frame["antenna_model"].astype(str).to_numpy()
    f["technology"] = frame["technology_key"].astype(str).to_numpy()
    if corrected:
        f["Etilt"] = frame["Etilt"].to_numpy(float) / 10.0
        f["Mtilt"] = frame["Mtilt"].to_numpy(float) / 10.0
        f["serving_frequency_mhz"] = frame["original_frequency_mhz"].to_numpy(float)
    else:
        f["Etilt"] = frame["Etilt"].to_numpy(float)
        f["Mtilt"] = frame["Mtilt"].to_numpy(float)
        f["serving_frequency_mhz"] = frame["frequency_mhz"].to_numpy(float)
    return A.antenna_gain_delta_details(f)["phase36_antenna_delta_db"].to_numpy(float)


def main() -> None:
    sites = pd.read_parquet(RUN / "phase46_sites.parquet")
    grid = pd.read_parquet(RUN / "phase46_grid.parquet")
    print(f"[P47] cells={len(sites)} grid_pixels={len(grid)}")

    glat = grid["center_lat"].to_numpy(float)
    glon = grid["center_lon"].to_numpy(float)

    # ---------------- Arm A: production rule, reproduced ----------------
    cfg = {"radius_m": RADIUS_M, "ensure_all_cells": True,
           "out_of_radius_backfill_k_nearest": BACKFILL_K}
    surface = S._run_raw_surface(sites, grid, cfg)
    raw_best = surface.groupby(["Technology", "grid_id"], dropna=False)["raw_cost231_rsrp"].transform("max")
    keep = (surface["raw_cost231_rsrp"] >= raw_best - MARGIN_DB) & (surface["raw_cost231_rsrp"] >= RAW_MIN_DBM)
    arm_a = surface.loc[keep].copy()
    print(f"[P47] arm A: raw={len(surface)} after_20dB_filter={len(arm_a)}")

    # Farthest backfill pixel arm A handed each cell -> arm B's domain radius.
    bf = surface.loc[surface["ensure_all_cells_backfill"]]
    reach = bf.groupby("strict_cell_key")["distance_m"].max()

    # ---------------- Arm B: absolute rule ----------------
    rows = []
    for _, site in sites.iterrows():
        key = str(site["strict_cell_key"])
        domain_r = max(RADIUS_M, float(reach.get(key, 0.0)))
        dist = S._haversine_m(float(site["lat"]), float(site["lon"]), glat, glon)
        inside = dist <= domain_r
        if not inside.any():
            continue
        sub = grid.loc[inside].copy()
        d = dist[inside]
        bearing = S._bearing_deg(float(site["lat"]), float(site["lon"]),
                                 sub["center_lat"].to_numpy(float), sub["center_lon"].to_numpy(float))
        az_delta = S._azimuth_delta_deg(bearing, float(site["azimuth"]))
        raw = S._cost231_for_points(S._site_record(site),
                                    sub["center_lat"].to_numpy(float),
                                    sub["center_lon"].to_numpy(float),
                                    float(site["frequency_mhz"]))
        raw = raw + float(site.get("model_rsrp_adjust_db", 0.0))
        frame = pd.DataFrame({
            "grid_id": sub["grid_id"].astype(str).to_numpy(),
            "strict_cell_key": key,
            "technology_key": str(site["technology_key"]),
            "distance_m": d,
            "azimuth_delta_deg": az_delta,
            "raw_cost231_rsrp": raw,
            "Height": float(site["Height"]),
            "Etilt": float(site["Etilt"]),
            "Mtilt": float(site["Mtilt"]),
            "antenna_model": str(site["antenna_model"]),
            "frequency_mhz": float(site["frequency_mhz"]),
            "original_frequency_mhz": float(site["original_frequency_mhz"]),
            "domain_radius_m": domain_r,
        })
        frame["physical_current_pap"] = raw + antenna_delta(frame, corrected=False)
        frame["physical_fixed_pap"] = raw + antenna_delta(frame, corrected=True)
        rows.append(frame)

    arm_b_all = pd.concat(rows, ignore_index=True)
    print(f"[P47] arm B domain rows (before any threshold)={len(arm_b_all)}")

    # ---------------- component comparison ----------------
    report = []
    for key, site in sites.set_index("strict_cell_key").iterrows():
        a = arm_a.loc[arm_a["strict_cell_key"].astype(str).eq(str(key)), "grid_id"]
        dom = arm_b_all.loc[arm_b_all["strict_cell_key"].eq(str(key))]
        b_cur = dom.loc[dom["physical_current_pap"] >= NO_COVERAGE_DBM, "grid_id"]
        b_fix = dom.loc[dom["physical_fixed_pap"] >= NO_COVERAGE_DBM, "grid_id"]
        report.append({
            "strict_cell_key": str(key),
            "technology": str(site["technology_key"]),
            "domain_radius_m": round(float(dom["domain_radius_m"].iloc[0]), 0) if len(dom) else np.nan,
            "A_rows": int(len(a)), "A_components": components(a),
            "B_rows_current_pap": int(len(b_cur)), "B_components_current_pap": components(b_cur),
            "B_rows_fixed_pap": int(len(b_fix)), "B_components_fixed_pap": components(b_fix),
        })
    rep = pd.DataFrame(report)
    out = Path(__file__).resolve().parent / "phase47_component_report.csv"
    rep.to_csv(out, index=False)

    n = len(rep)
    print()
    print("=" * 74)
    print(f"ISLAND CHECK over {n} cells (no smoothing / no joining / no component deletion)")
    print("=" * 74)
    for label, col in [("A  production rule (500 m + backfill + 20 dB filter)", "A_components"),
                       ("B  absolute rule, current PAP lookup                ", "B_components_current_pap"),
                       ("B  absolute rule, PAP lookup fixed                  ", "B_components_fixed_pap")]:
        c = rep[col]
        print(f"{label} : split={int((c > 1).sum())}/{n}  max_pieces={int(c.max())}  single_piece={int((c == 1).sum())}")
    print()
    print("component-count distribution:")
    print(pd.DataFrame({
        "A_production": rep.A_components.value_counts().sort_index(),
        "B_current_pap": rep.B_components_current_pap.value_counts().sort_index(),
        "B_fixed_pap": rep.B_components_fixed_pap.value_counts().sort_index(),
    }).fillna(0).astype(int).to_string())

    still = rep[rep.B_components_fixed_pap > 1]
    if len(still):
        print()
        print(f"cells still split under arm B ({len(still)}):")
        print(still[["strict_cell_key", "technology", "domain_radius_m",
                     "B_rows_fixed_pap", "B_components_fixed_pap"]].to_string(index=False))

    # ---------------- directionality must survive ----------------
    print()
    print("directionality check - median physical RSRP by |azimuth delta| (arm B, fixed PAP):")
    d = arm_b_all[arm_b_all.physical_fixed_pap >= NO_COVERAGE_DBM]
    print(d.groupby(pd.cut(d.azimuth_delta_deg.abs(), [0, 30, 60, 90, 120, 150, 180]),
                    observed=True).physical_fixed_pap.agg(["count", "median"]).round(1).to_string())
    print()
    print(f"[P47] per-cell report written: {out}")


if __name__ == "__main__":
    main()
