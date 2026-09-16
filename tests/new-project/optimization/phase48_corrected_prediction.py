"""Phase 48: the corrected prediction chain, test-only.

No production file is modified and nothing is written to the database. The
propagation, obstruction and antenna-pattern code is imported from production
and called unchanged; only the inputs and the composition are corrected.

What Phase 48 changes vs production
  1  tx power per technology            4G 46 dBm, 5G 53 dBm (not a flat 46)
  2  antenna gain                       real PAP boresight + H + V, applied
                                        ABSOLUTELY (no real-minus-generic delta,
                                        no +/-15/12 clip)
  3  PAP file selection                 by DEPLOYED frequency (775.5 -> the
                                        698-806 MHz file) and tilt/10 (30 -> T 3)
  4  per-RE conversion                  applied to 5G as well (-35.15 dB at
                                        273 RB); production applies 0.0
  5  COST-231 anchor                    unchanged production behaviour: run at
                                        the anchor, correct with the analytic
                                        frequency term (775.5->1500 +9.71 dB,
                                        3300->2600 -3.51 dB)
  6  obstruction frequency              real deployed frequency, not the anchor
  7  membership                         ABSOLUTE: this cell's own value >= floor.
                                        No nearest-8 backfill, no competitor
                                        -relative 20 dB filter.
  8  radius                             mode 'fixed500' = 500 m, no backfill;
                                        mode 'cellh'    = per-cell link-budget
                                        radius solved from COST-231.

No smoothing, no joining of patches, no deletion of small components.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ML))

from tools.lte_prediction_offset import services as S
from tools.lte_prediction_offset import phase27_physical as P
from tools.lte_prediction_offset import phase36_physical_upgrades as A

RUN = ML / "tests" / "output" / "baseline_210_profile_20260908_153817"
PROJECT_DIR = ML / "tests" / "new-project" / "data" / "project_210_taiwan"
BUILDINGS = PROJECT_DIR / "geo_db" / "tbl_savepolygon_project_210_buildings.parquet"
DEM_PATH = (ML / "tests" / "new-project" / "data" / "mapdata"
            / "Dno19_0095_NewTaipeiCity_5m" / "Dno19_0095_NewTaipeiCity_5m"
            / "New_TaipeiCity_5m_UTM51N_planet" / "Heights" / "height_5m.grd")

TX_POWER_DBM = {"4G": 46.0, "5G": 53.0}
CABLE_LOSS_DB = 2.0
UE_HEIGHT_M = 1.5
RSRP_CEILING_DBM = -44.0
DT_SNAP_RADIUS_M = 25.0
# Phase 36's back-lobe rule, reused so the fit subset matches Phase 39's.
BACKLOBE_DROP_DEG = 135.0
MIN_FIT_SAMPLES = 30
GRID_RE = re.compile(r"R(\d+)C(\d+)")

# Phase 38 owns the authoritative 4G band decision: it is derived from EARFCN,
# never from the `band` text label. Imported rather than reimplemented.
sys.path.insert(0, str(ML / "tests" / "new-project"))
import test_project210_phase38_earfcn_rematch as p38  # noqa: E402

# Phase 38 band label -> this project's band_key
P38_BAND_TO_KEY = {"B3": "3", "B28": "28", "n78": "78"}


# ----------------------------------------------------------------- helpers
def load_buildings() -> pd.DataFrame:
    """Decode the cached building polygons into WKT production can parse.

    The cached table stores each polygon in the mislabelled `region` column as a
    raw MySQL geometry blob: 4-byte little-endian SRID followed by standard WKB.
    Production's `_parse_geometry_value` handles WKT/hex but not raw bytes, so it
    silently resolves 0 buildings. Decode here and hand production a
    `geometry_wkt` column; the resolved heights come from the profile capture.
    """
    from shapely import wkb as shapely_wkb

    raw = pd.read_parquet(BUILDINGS)
    wkts, keep = [], []
    for idx, value in raw["region"].items():
        if not isinstance(value, (bytes, bytearray)) or len(value) < 5:
            continue
        try:
            geom = shapely_wkb.loads(bytes(value)[4:])   # strip the SRID prefix
        except Exception:
            continue
        if geom is None or geom.is_empty:
            continue
        wkts.append(geom.wkt)
        keep.append(idx)
    out = raw.loc[keep].drop(columns=["region", "geometry"], errors="ignore").copy()
    out["geometry_wkt"] = wkts

    profiles = RUN / "building_profiles.csv"
    if profiles.is_file():
        prof = pd.read_csv(profiles)
        cols = [c for c in ("id", "building_height_m", "building_levels", "height_m") if c in prof.columns]
        out = out.merge(prof[cols].drop_duplicates("id"), on="id", how="left")
        if "building_height_m" in out and "height_m" in out:
            out["building_height_m"] = pd.to_numeric(out["building_height_m"], errors="coerce").fillna(
                pd.to_numeric(out["height_m"], errors="coerce"))
    print(f"[P48] buildings decoded={len(out)} of {len(raw)} rows in cache", flush=True)
    return out


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


def cost231_pathloss_db(freq_anchor_mhz: float, h_tx_m: float, d_m: np.ndarray) -> np.ndarray:
    """COST-231 Hata, identical to production's compute_sector_rsrp internals."""
    d_km = np.maximum(np.asarray(d_m, dtype=float), 1.0) / 1000.0
    f = float(freq_anchor_mhz)
    a_hm = (1.1 * math.log10(f) - 0.7) * UE_HEIGHT_M - (1.56 * math.log10(f) - 0.8)
    base = 46.3 + 33.9 * math.log10(f) - 13.82 * math.log10(h_tx_m) - a_hm + 3.0
    slope = 44.9 - 6.55 * math.log10(h_tx_m)
    return base + slope * np.log10(d_km)


def cost231_distance_m(freq_anchor_mhz: float, h_tx_m: float, pl_max_db: float) -> float:
    """Invert COST-231 for the distance at which path loss reaches pl_max."""
    f = float(freq_anchor_mhz)
    a_hm = (1.1 * math.log10(f) - 0.7) * UE_HEIGHT_M - (1.56 * math.log10(f) - 0.8)
    base = 46.3 + 33.9 * math.log10(f) - 13.82 * math.log10(h_tx_m) - a_hm + 3.0
    slope = 44.9 - 6.55 * math.log10(h_tx_m)
    return float(10.0 ** ((pl_max_db - base) / slope) * 1000.0)


def per_re_db(technology: str, bw_mhz: float) -> float:
    """Total carrier power -> per-RE (RSRP / SS-RSRP) level. Negative."""
    return A.per_re_offset_db(str(technology), float(bw_mhz))


def pap_absolute_gain(site, az_off_deg: np.ndarray, dist_m: np.ndarray) -> tuple[np.ndarray, dict]:
    """Absolute antenna gain from the real PAP pattern: g0 + G_h(az) + G_v(dep).

    The pattern is selected with the DEPLOYED frequency and the tilt converted
    out of tenths-of-a-degree. No generic 3GPP pattern is involved and no clip
    is applied - the pattern keeps its full dynamic range.
    """
    tech = str(site["technology_key"])
    deployed_mhz = float(site["original_frequency_mhz"])
    etilt_deg = float(site["Etilt"]) / 10.0
    mtilt_deg = float(site["Mtilt"]) / 10.0
    h_tx = float(site["Height"])

    model = site.get("antenna_model")
    model = "" if model is None or pd.isna(model) else str(model).strip()
    pat = A.default_pattern_lookup(model, deployed_mhz, etilt_deg, tech)
    elev = np.degrees(np.arctan2(UE_HEIGHT_M - h_tx, np.maximum(dist_m, 1.0)))
    depression = -elev + mtilt_deg
    if pat is None:
        raise RuntimeError(f"No PAP pattern resolved for {site['strict_cell_key']}")
    hs, h, vs, v, g0 = pat
    h_gain = A._pat_gain_smoothed(hs, np.asarray(h, float), np.abs(az_off_deg))
    v_gain = A._pat_gain_smoothed(vs, np.asarray(v, float), depression)
    path, _ = A._default_pap_path("5G" if tech == "5G" else "4G", deployed_mhz, int(round(etilt_deg)))
    meta = {"pap_file": path.name if path is not None else "",
            "boresight_dbi": float(g0), "etilt_deg": etilt_deg, "mtilt_deg": mtilt_deg}
    return g0 + h_gain + v_gain, meta


# ----------------------------------------------------------------- main
def build_surface(sites: pd.DataFrame, grid: pd.DataFrame, mode: str,
                  edge_dbm: float, bw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    glat = grid["center_lat"].to_numpy(float)
    glon = grid["center_lon"].to_numpy(float)
    frames, radius_rows = [], []

    for _, site in sites.iterrows():
        tech = str(site["technology_key"])
        h_tx = float(site["Height"])
        anchor_mhz = float(site["frequency_mhz"])          # production dynamic anchor
        deployed_mhz = float(site["original_frequency_mhz"])
        anchor_offset = float(site.get("model_rsrp_adjust_db", 0.0))
        tx = TX_POWER_DBM[tech]
        pre = per_re_db(tech, bw[tech])

        _, meta0 = pap_absolute_gain(site, np.array([0.0]), np.array([100.0]))
        g0 = meta0["boresight_dbi"]

        # Link-budget radius: boresight EIRP-per-RE against the edge threshold.
        eirp_re_boresight = tx + g0 - CABLE_LOSS_DB + pre + anchor_offset
        pl_max = eirp_re_boresight - edge_dbm
        d_edge = cost231_distance_m(anchor_mhz, h_tx, pl_max)
        radius_m = 500.0 if mode == "fixed500" else d_edge
        radius_rows.append({"strict_cell_key": str(site["strict_cell_key"]), "technology": tech,
                            "band": str(site["band_key"]), "height_m": h_tx,
                            "deployed_mhz": deployed_mhz, "anchor_mhz": anchor_mhz,
                            "tx_dbm": tx, "boresight_dbi": g0, "per_re_db": round(pre, 2),
                            "anchor_offset_db": round(anchor_offset, 2),
                            "eirp_re_boresight_dbm": round(eirp_re_boresight, 2),
                            "pl_max_db": round(pl_max, 2),
                            "link_budget_radius_m": round(d_edge, 1),
                            "radius_used_m": round(radius_m, 1),
                            "pap_file": meta0["pap_file"], "etilt_deg": meta0["etilt_deg"]})

        dist = S._haversine_m(float(site["lat"]), float(site["lon"]), glat, glon)
        inside = dist <= radius_m
        if not inside.any():
            continue
        sub = grid.loc[inside]
        d = dist[inside]
        bearing = S._bearing_deg(float(site["lat"]), float(site["lon"]),
                                 sub["center_lat"].to_numpy(float), sub["center_lon"].to_numpy(float))
        az_delta = S._azimuth_delta_deg(bearing, float(site["azimuth"]))
        gain, meta = pap_absolute_gain(site, az_delta, d)
        pl = cost231_pathloss_db(anchor_mhz, h_tx, d)

        # Absolute link budget, per resource element.
        raw = tx + gain - CABLE_LOSS_DB - pl + anchor_offset + pre

        frames.append(pd.DataFrame({
            "grid_id": sub["grid_id"].astype(str).to_numpy(),
            "lat": sub["center_lat"].to_numpy(float),
            "lon": sub["center_lon"].to_numpy(float),
            "strict_cell_key": str(site["strict_cell_key"]),
            "site": str(site["site_key"]), "sector": str(site["sector_key"]),
            "band": str(site["band_key"]), "technology": tech,
            "distance_m": d, "azimuth_delta_deg": az_delta,
            "antenna_gain_dbi": gain,
            "pathloss_db": pl,
            "serving_frequency_mhz": deployed_mhz,   # real frequency for obstruction physics
            "anchor_mhz": anchor_mhz,
            "tx_power_dbm": tx, "per_re_db": pre, "anchor_offset_db": anchor_offset,
            "raw_cost231_rsrp": raw,                 # name kept for production scorer
            "radius_used_m": radius_m,
            "pap_file": meta["pap_file"], "pap_boresight_dbi": meta["boresight_dbi"],
        }))

    return pd.concat(frames, ignore_index=True), pd.DataFrame(radius_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed500", "cellh"], default="fixed500")
    ap.add_argument("--edge-dbm", type=float, default=-110.0)
    ap.add_argument("--bw-4g", type=float, default=10.0)
    ap.add_argument("--bw-5g", type=float, default=100.0)
    ap.add_argument("--no-obstruction", action="store_true")
    ap.add_argument("--outdoor-building-loss", choices=["full", "none", "capped"], default="full",
                    help="How to treat the knife-edge building term on OUTDOOR pixels. "
                         "COST-231 Hata already contains urban clutter statistically, so 'full' "
                         "double-counts it. Indoor O2I is always kept.")
    ap.add_argument("--outdoor-cap-db", type=float, default=10.0)
    ap.add_argument("--floor-dbm", type=float, default=-120.0,
                    help="No-coverage floor. A pixel is kept only if this cell's own value "
                         "reaches it. Display range is floor .. -44 dBm.")
    ap.add_argument("--reuse-scored", type=str, default="",
                    help="Path to a phase48_scored_full.parquet to reuse obstruction from, "
                         "skipping the slow per-path recomputation.")
    args = ap.parse_args()
    bw = {"4G": args.bw_4g, "5G": args.bw_5g}
    started = time.perf_counter()

    out_dir = PROJECT_DIR / f"phase48_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sites = pd.read_parquet(RUN / "phase46_sites.parquet")
    grid = pd.read_parquet(RUN / "phase46_grid.parquet")
    production = pd.read_parquet(RUN / "baseline_predictions.parquet")
    # Production's own per-grid clutter classification, reused unchanged.
    clutter_by_grid = (production[["grid_id", "clutter_class"]].dropna()
                       .drop_duplicates("grid_id").astype(str)
                       .set_index("grid_id")["clutter_class"].to_dict())
    print(f"[P48] mode={args.mode} cells={len(sites)} grid={len(grid)} "
          f"bw4g={bw['4G']} bw5g={bw['5G']} edge={args.edge_dbm}", flush=True)

    surface, radii = build_surface(sites, grid, args.mode, args.edge_dbm, bw)
    radii.to_csv(out_dir / "phase48_link_budget_radius.csv", index=False)
    print(f"[P48] surface rows={len(surface)} (absolute membership, no backfill, no 20 dB filter)", flush=True)
    print("[P48] link-budget radius by band:", flush=True)
    print(radii.groupby(["technology", "band"]).link_budget_radius_m.agg(["min", "median", "max"]).round(0).to_string(), flush=True)

    if args.no_obstruction:
        surface["building_obstruction_loss_db"] = 0.0
        surface["terrain_diffraction_loss_db"] = 0.0
        surface["obstruction_branch"] = "not_evaluated"
        surface["clutter_class"] = surface["grid_id"].map(clutter_by_grid).fillna("Open")
        surface["physical_rsrp_unclipped"] = surface["raw_cost231_rsrp"]
    elif args.reuse_scored:
        # Obstruction depends only on the tx->rx geometry, never on power or
        # antenna, so a previous run's per-path values are reusable verbatim.
        prev = pd.read_parquet(args.reuse_scored)
        cols = ["strict_cell_key", "grid_id", "building_obstruction_loss_db",
                "terrain_diffraction_loss_db", "obstruction_branch", "clutter_class"]
        prev = prev[cols].drop_duplicates(["strict_cell_key", "grid_id"])
        before = len(surface)
        surface = surface.merge(prev, on=["strict_cell_key", "grid_id"], how="inner")
        print(f"[P48] reused obstruction from {args.reuse_scored}: {len(surface)} of {before} rows matched", flush=True)
    else:
        buildings = load_buildings()
        print(f"[P48] dem={DEM_PATH.name} exists={DEM_PATH.is_file()}", flush=True)
        surface = P.score_candidates(
            surface, sites, buildings, project_id=210, region="taiwan",
            dem_raster_path=DEM_PATH, clutter_by_grid=clutter_by_grid, allow_auto_dem=False)

    # COST-231 Hata is an empirical URBAN model: its measured path loss already
    # contains statistical building clutter. Adding an explicit knife-edge
    # building loss on top of it for an OUTDOOR pixel double-counts that
    # clutter. `--outdoor-building-loss none` keeps the indoor O2I term (a real
    # extra wall the empirical model never saw) and drops the outdoor term.
    bldg = pd.to_numeric(surface["building_obstruction_loss_db"], errors="coerce").fillna(0.0)
    branch = surface["obstruction_branch"].astype(str)
    if args.outdoor_building_loss == "none":
        bldg = bldg.where(branch.eq("indoor"), 0.0)
    elif args.outdoor_building_loss == "capped":
        bldg = bldg.where(branch.eq("indoor"), bldg.clip(lower=-float(args.outdoor_cap_db)))
    surface["building_obstruction_loss_applied_db"] = bldg
    surface["physical_rsrp_unclipped"] = (
        pd.to_numeric(surface["raw_cost231_rsrp"], errors="coerce")
        + bldg
        - pd.to_numeric(surface["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0)
    )

    # Uncalibrated physical, per resource element.
    surface["phase48_rsrp_uncal"] = np.minimum(
        pd.to_numeric(surface["physical_rsrp_unclipped"], errors="coerce"), RSRP_CEILING_DBM)
    floor = float(args.floor_dbm)

    # ---- drive test, matched by Phase 38's rule
    # The 4G band comes from EARFCN via phase38._earfcn_band, never from the
    # `band` text column. B1B7 / other / unknown have no cell in this project
    # and are excluded, exactly as Phase 38 sets p38_excluded. Every NR row is
    # taken as n78 (the DB labels them 'nr'/'n77'; there is only one NR band).
    dt = pd.read_parquet(RUN / "drive_test_rows.parquet")
    net = dt["network"].astype(str).str.upper()
    dt["dt_tech"] = np.where(net.str.contains("5G|NR", na=False), "5G", "4G")
    dt["rsrp_measured"] = pd.to_numeric(dt["rsrp"], errors="coerce")
    earfcn_band = pd.to_numeric(dt["earfcn"], errors="coerce").map(p38._earfcn_band)
    dt["p38_true_band"] = np.where(dt.dt_tech.eq("4G"), earfcn_band, "n78")
    dt["dt_band"] = dt["p38_true_band"].map(P38_BAND_TO_KEY)
    dt["p38_excluded"] = dt["dt_band"].isna()
    print("[P48] DT band from EARFCN (phase38 rule):", flush=True)
    print(dt.groupby(["dt_tech", "p38_true_band"]).size().to_string(), flush=True)
    print(f"[P48] DT rows excluded (no cell on that band): {int(dt.p38_excluded.sum())}", flush=True)
    usable = dt[(~dt.p38_excluded) & dt.rsrp_measured.notna()].copy()

    gy = grid["center_lat"].to_numpy(float) * 111000.0
    gx = grid["center_lon"].to_numpy(float) * 111000.0 * math.cos(math.radians(24.0))
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[gy, gx])
    q = np.c_[usable.lat.to_numpy(float) * 111000.0,
              usable.lon.to_numpy(float) * 111000.0 * math.cos(math.radians(24.0))]
    dist_snap, pos = tree.query(q, k=1)
    usable["grid_id"] = grid["grid_id"].astype(str).to_numpy()[pos]
    usable["snap_m"] = dist_snap
    usable = usable[usable.snap_m <= DT_SNAP_RADIUS_M].copy()

    def match_dt(value_col: str) -> pd.DataFrame:
        """Match every measurement to the best server WITHIN ITS OWN BAND, so a
        B3 measurement is never compared against a B28 prediction."""
        idxb = surface.groupby(["technology", "band", "grid_id"])[value_col].idxmax()
        per_band = surface.loc[idxb]
        pred = per_band[["technology", "band", "grid_id", value_col, "strict_cell_key",
                         "azimuth_delta_deg", "obstruction_branch", "clutter_class"]].rename(
            columns={"technology": "dt_tech", "band": "dt_band", value_col: "predicted_rsrp",
                     "strict_cell_key": "predicted_cell",
                     "azimuth_delta_deg": "predicted_az_delta",
                     "obstruction_branch": "predicted_branch",
                     "clutter_class": "predicted_clutter"})
        out = usable.merge(pred, on=["dt_tech", "dt_band", "grid_id"], how="inner")
        out["error_db"] = out.predicted_rsrp - out.rsrp_measured
        # Phase 39's fit subset: outdoor, not back lobe, band-valid.
        out["p36_backlobe"] = pd.to_numeric(out.predicted_az_delta, errors="coerce").fillna(0.0).abs() > BACKLOBE_DROP_DEG
        out["phase39_fit_row"] = (~out.predicted_branch.astype(str).eq("indoor")) & (~out.p36_backlobe)
        # Spatial split, like phase25's split-by-grid: a grid pixel is wholly in
        # train or wholly in test, so test points are never neighbours of train.
        h = out.grid_id.astype(str).map(lambda g: int(hashlib.md5(g.encode()).hexdigest(), 16) % 100)
        out["split"] = np.where(h < 70, "train", "test")
        return out

    def _stats(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        return (frame.groupby(["dt_tech", "dt_band"])
                .agg(rows=("error_db", "size"),
                     measured_median=("rsrp_measured", "median"),
                     predicted_median=("predicted_rsrp", "median"),
                     bias_db=("error_db", "median"),
                     mae_db=("error_db", lambda s: s.abs().mean()))
                .round(2))

    dtm_uncal = match_dt("phase48_rsrp_uncal")
    dt_stats_uncal = _stats(dtm_uncal[dtm_uncal.phase39_fit_row])

    # ---- calibration, fitted ONLY on train + phase39 fit rows
    fit_rows = dtm_uncal[dtm_uncal.phase39_fit_row & dtm_uncal.split.eq("train")].copy()
    lvl1 = (-fit_rows.groupby(["dt_tech", "dt_band"]).error_db.median()).rename("tech_band_correction_db")
    fit_rows = fit_rows.join(lvl1, on=["dt_tech", "dt_band"])
    fit_rows["resid1"] = fit_rows.error_db + fit_rows.tech_band_correction_db
    grp = fit_rows.groupby(["dt_tech", "dt_band", "predicted_clutter"])
    lvl2 = (-grp.resid1.median()).rename("clutter_correction_db")
    counts = grp.size().rename("n")
    lvl2 = pd.concat([lvl2, counts], axis=1)
    lvl2.loc[lvl2.n < MIN_FIT_SAMPLES, "clutter_correction_db"] = 0.0
    print("[P48] calibration level 1 (tech/band):", flush=True)
    print(lvl1.round(2).to_string(), flush=True)
    print("[P48] calibration level 2 (clutter refinement, min "
          f"{MIN_FIT_SAMPLES} samples):", flush=True)
    print(lvl2.round(2).to_string(), flush=True)

    surface = surface.join(lvl1.rename_axis(["technology", "band"]), on=["technology", "band"])
    surface["tech_band_correction_db"] = surface.tech_band_correction_db.fillna(0.0)
    surface = surface.join(lvl2["clutter_correction_db"].rename_axis(["technology", "band", "clutter_class"]),
                           on=["technology", "band", "clutter_class"])
    surface["clutter_correction_db"] = surface.clutter_correction_db.fillna(0.0)
    surface["phase48_rsrp"] = np.minimum(
        surface.phase48_rsrp_uncal + surface.tech_band_correction_db + surface.clutter_correction_db,
        RSRP_CEILING_DBM)
    # Saved AFTER calibration so the dashboard can show the uncalibrated and the
    # calibrated surface from one file, including how membership changes.
    surface.to_parquet(out_dir / "phase48_scored_full.parquet", index=False)

    # DT re-matched on the CALIBRATED surface - this is what the CDF must use.
    dtm = match_dt("phase48_rsrp")
    dt_stats = _stats(dtm[dtm.phase39_fit_row])
    dt_stats_train = _stats(dtm[dtm.phase39_fit_row & dtm.split.eq("train")])
    dt_stats_test = _stats(dtm[dtm.phase39_fit_row & dtm.split.eq("test")])
    dtm.to_parquet(out_dir / "phase48_dt_match.parquet", index=False)
    dtm_uncal.to_parquet(out_dir / "phase48_dt_match_uncalibrated.parquet", index=False)

    # ---- membership on the calibrated value
    kept = surface[surface["phase48_rsrp"] >= floor].copy()
    print(f"[P48] rows kept by absolute >= {floor} dBm: {len(kept)} of {len(surface)}", flush=True)
    kept.to_parquet(out_dir / "phase48_cell_surface.parquet", index=False)

    rep = []
    for key, grp2 in kept.groupby("strict_cell_key"):
        rep.append({"strict_cell_key": key, "technology": grp2.technology.iloc[0],
                    "band": grp2.band.iloc[0], "rows": len(grp2),
                    "radius_used_m": round(float(grp2.radius_used_m.iloc[0]), 1),
                    "components": components(grp2.grid_id)})
    rep = pd.DataFrame(rep)
    rep.to_csv(out_dir / "phase48_components.csv", index=False)

    idx = kept.groupby(["technology", "grid_id"])["phase48_rsrp"].idxmax()
    serving = kept.loc[idx].copy()
    serving.to_parquet(out_dir / "phase48_serving.parquet", index=False)

    serving["io"] = np.where(serving.get("obstruction_branch", pd.Series("", index=serving.index))
                             .astype(str).eq("indoor"), "Indoor", "Outdoor")
    sep = serving.groupby(["technology", "io"])["phase48_rsrp"].median().unstack()
    if {"Indoor", "Outdoor"}.issubset(sep.columns):
        sep["outdoor_minus_indoor_dB"] = (sep["Outdoor"] - sep["Indoor"]).round(1)

    summary = {
        "mode": args.mode, "edge_dbm": args.edge_dbm, "bandwidth_mhz": bw,
        "tx_power_dbm": TX_POWER_DBM, "obstruction": not args.no_obstruction,
        "outdoor_building_loss": args.outdoor_building_loss,
        "outdoor_cap_db": args.outdoor_cap_db if args.outdoor_building_loss == "capped" else None,
        "production_files_modified": False, "database_writes": False,
        "surface_rows": int(len(surface)), "rows_kept": int(len(kept)),
        "cells": int(kept.strict_cell_key.nunique()),
        "cells_split_into_islands": int((rep.components > 1).sum()),
        "max_components": int(rep.components.max()),
        "radius_m_by_band": {f"{t}_B{b}": float(v) for (t, b), v in
                             radii.groupby(["technology", "band"]).link_budget_radius_m.median().round(0).items()},
        "indoor_outdoor_median_dbm": json.loads(sep.round(1).to_json(orient="index")),
        "floor_dbm": floor,
        "dt_uncalibrated_fit_subset": json.loads(dt_stats_uncal.to_json(orient="index")) if len(dt_stats_uncal) else {},
        "dt_calibrated_fit_subset": json.loads(dt_stats.to_json(orient="index")) if len(dt_stats) else {},
        "dt_calibrated_train": json.loads(dt_stats_train.to_json(orient="index")) if len(dt_stats_train) else {},
        "dt_calibrated_test": json.loads(dt_stats_test.to_json(orient="index")) if len(dt_stats_test) else {},
        "calibration_level1_db": {f"{t}_B{b}": round(float(v),2) for (t,b),v in lvl1.items()},
        "dt_rows_excluded_no_cells": int(dt.p38_excluded.sum()),
        "dt_band_source": "phase38._earfcn_band (EARFCN, not the band text label)",
        "wall_s": round(time.perf_counter() - started, 1),
    }
    (out_dir / "phase48_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 78)
    print(f"PHASE 48 RESULT - mode={args.mode}")
    print("=" * 78)
    print(f"cells={summary['cells']}  rows kept={summary['rows_kept']}")
    print(f"ISLANDS: cells split={summary['cells_split_into_islands']}/{len(rep)}  max pieces={summary['max_components']}")
    print()
    print("link-budget radius (m), median by band:")
    print(radii.groupby(["technology", "band"]).link_budget_radius_m.median().round(0).to_string())
    print()
    print("PAP file actually selected:")
    print(radii.groupby(["technology", "band"]).pap_file.agg(lambda s: s.value_counts().index[0]).to_string())
    print()
    print("indoor vs outdoor median RSRP on the serving surface:")
    print(sep.round(1).to_string())
    print()
    print(f"floor = {floor} dBm  (display range {floor} .. {RSRP_CEILING_DBM} dBm)")
    print()
    print("DT phase39 fit subset - BEFORE calibration:")
    print(dt_stats_uncal.to_string())
    print()
    print("fitted calibration (dB added):")
    print(lvl1.round(2).to_string())
    print()
    print("DT phase39 fit subset - AFTER calibration (all rows):")
    print(dt_stats.to_string())
    print()
    print("AFTER calibration - TRAIN (fitted on these):")
    print(dt_stats_train.to_string())
    print()
    print("AFTER calibration - TEST (held out, spatially separate grids):")
    print(dt_stats_test.to_string())
    print()
    print(f"[P48] wrote {out_dir}")


if __name__ == "__main__":
    main()
