"""
RF surface for the redundancy model: a calibrated RSRP value for EVERY cell at EVERY analysis point.

Why it is rebuilt instead of read from lte_prediction_baseline_results: that table stores, per point,
only the sectors of the one site the point was sampled for (project 193 Airtel: always 3 rows of one
site), and its RSRP is flat with distance and direction (~-91 dBm from 50 m to 500 m, boresight to
behind the antenna) -- drive-test values carried over and smoothed. Cross-site overlap cannot be read
from it, and a directional model calibrated to it over-predicted the drive test by 27 dB.

  1. raw RSRP      tools/lte_prediction compute_sector_rsrp (COST-231 Hata + 3GPP antenna pattern),
                   converted from carrier power to per-RE RSRP
  2. calibration   the drive test is the ground truth. Each serving row is attributed to the antenna it
                   was measured on (serving eNB = site id, then PCI, else that site's strongest sector at
                   the spot). Per band: median(measured - raw). Reference level: pooled median over all
                   attributed rows (or reference_band). Per-antenna offsets (shrunk n/(n+n0)) are kept
                   only if they reduce the held-out error.
  3. validation    session-grouped cross-validation: calibrate on the other drive sessions, predict the
                   held-out session's best server. Held-out spread -> sigma of the coverage probability;
                   held-out bias, spread and serving-site agreement -> PASS / FAIL of the RF surface.
  4. interference  load factor fitted so the predicted SINR matches measured SINR (median); the default is
                   used when an operator logs no usable SINR (JIO logs 0).
  5. points        analysis grid over the polygon (+ buffer); building points are indoor; users_per_grid
                   (lte_forecast_predictions) weights points where it exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from tests.overlapping.config import CalibrationParams, OverlapConfig, RFParams
from tests.overlapping.geo import LocalProjection, build_grid, haversine_m, orient_many_lonlat, parse_geometry
from tests.overlapping.removal_model import NO_SIGNAL_DBM, Network
from tests.overlapping.site_cleaning import SENTINEL_SITE_IDS, normalise_operator
from tools.lte_prediction.Sector_wise_prediction_code_copy import compute_sector_rsrp

VALIDATION_PASS = "PASS"
VALIDATION_FAIL = "FAIL"
VALIDATION_NOT_CHECKED = "NOT_CHECKED"


def raw_rsrp_matrix(cells: pd.DataFrame, lat, lon, rf: RFParams) -> np.ndarray:
    """(points, cells) per-RE RSRP before calibration; NO_SIGNAL_DBM beyond max_cell_radius_m."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    out = np.full((lat.size, len(cells)), NO_SIGNAL_DBM, dtype=np.float32)
    params = {"ue_height": rf.ue_height_m, "antenna_gain": rf.antenna_gain_dbi, "cable_loss": rf.cable_loss_db}
    for j, cell in enumerate(cells.itertuples(index=False)):
        near = haversine_m(cell.lat, cell.lon, lat, lon) <= rf.max_cell_radius_m
        if not near.any():
            continue
        site = {
            "lat": cell.lat,
            "lon": cell.lon,
            "azimuth": cell.azimuth,
            "electrical_tilt": cell.e_tilt,
            "mechanical_tilt": cell.m_tilt,
            "antenna_height": cell.height,
            "tx_power": cell.tx_power,
        }
        out[near, j] = compute_sector_rsrp(site, lat[near], lon[near], rf.frequency_mhz, params) - rf.re_normalisation_db
    return out


def grid_point_ids(points: pd.DataFrame, resolution_m: float, proj: LocalProjection, lat, lon) -> np.ndarray:
    """Grid point id of the node nearest each (lat, lon); -1 when that node is outside the grid."""
    res = float(resolution_m)
    lookup = dict(
        zip(
            zip(np.rint(points["x_m"] / res).astype(np.int64), np.rint(points["y_m"] / res).astype(np.int64)),
            points["point_id"].astype(np.int64),
        )
    )
    x, y = proj.to_xy(lat, lon)
    keys = zip(np.rint(x / res).astype(np.int64), np.rint(y / res).astype(np.int64))
    return np.fromiter((lookup.get(k, -1) for k in keys), dtype=np.int64, count=len(x))


# --------------------------------------------------------------------------- drive-test helpers

def enb_keys(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    ok = num.notna() & (num > 0) & (num % 1 == 0)
    keys = num.where(ok).astype("Int64").astype(str)
    return keys.where(ok & ~keys.isin(SENTINEL_SITE_IDS))


def handover_site_pairs(dt: pd.DataFrame, max_gap_s: float) -> set[tuple[str, str]]:
    """Site pairs the phone actually handed over between (consecutive serving eNB change)."""
    d = pd.DataFrame(
        {
            "session_id": dt["session_id"],
            "ts": pd.to_datetime(dt["timestamp"], errors="coerce"),
            "enb": enb_keys(dt["nodeb_id"]),
        }
    ).dropna()
    if d.empty:
        return set()
    d = d.sort_values(["session_id", "ts"])
    prev_enb = d.groupby("session_id")["enb"].shift()
    gap = (d["ts"] - d.groupby("session_id")["ts"].shift()).dt.total_seconds()
    change = prev_enb.notna() & (d["enb"] != prev_enb) & (gap <= max_gap_s)
    return {tuple(sorted((a, b))) for a, b in zip(prev_enb[change], d["enb"][change])}


def attribute_serving_antenna(cells: pd.DataFrame, raw: np.ndarray, enb: np.ndarray, pci: np.ndarray) -> np.ndarray:
    """
    Antenna each drive-test row was measured on: the site whose id is the serving eNB, then the
    antenna carrying the row's PCI, else that site's strongest antenna at the spot. -1 when the eNB
    is not in the config or the antenna does not reach the spot.
    """
    site = cells["site_key"].astype(str).to_numpy()
    pci_sets = [set(p.split(",")) - {""} for p in cells["pcis"].astype(str)]
    by_site = {s: np.flatnonzero(site == s) for s in np.unique(site)}
    out = np.full(len(enb), -1, dtype=int)
    for r, (e, p) in enumerate(zip(enb, pci)):
        js = by_site.get(e) if isinstance(e, str) else None
        if js is None:
            continue
        if np.isfinite(p):
            match = [j for j in js if str(int(p)) in pci_sets[j]]
            if len(match) == 1:
                out[r] = match[0]
                continue
        out[r] = js[np.argmax(raw[r, js])]
    hit = np.flatnonzero(out >= 0)
    reaches = raw[hit, out[hit]] > NO_SIGNAL_DBM + 1.0
    out[hit[~reaches]] = -1
    return out


def _robust_sigma(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _band_offsets(resid: np.ndarray, band: np.ndarray, mask: np.ndarray, min_rows: int) -> tuple[pd.Series, float]:
    r = pd.Series(resid[mask])
    stats = r.groupby(band[mask]).agg(["median", "size"])
    return stats.loc[stats["size"] >= min_rows, "median"], float(r.median())


def _row_offsets(band: np.ndarray, band_offsets: pd.Series, pooled: float) -> np.ndarray:
    return pd.Series(band).map(band_offsets).fillna(pooled).to_numpy(dtype=float)


def _antenna_offsets(
    resid: np.ndarray, row_offset: np.ndarray, serving: np.ndarray, mask: np.ndarray, n_cells: int, cal: CalibrationParams
) -> tuple[np.ndarray, np.ndarray]:
    grouped = pd.Series(resid[mask] - row_offset[mask]).groupby(serving[mask])
    median, count = grouped.median(), grouped.size()
    rel = np.zeros(n_cells)
    counts = np.zeros(n_cells, dtype=int)
    idx = median.index.to_numpy(dtype=int)
    shrunk = (count / (count + cal.shrinkage_points) * median).to_numpy()
    rel[idx] = np.clip(shrunk, -cal.max_offset_deviation_db, cal.max_offset_deviation_db)
    counts[idx] = count.to_numpy()
    return rel, counts


def _sinr_parts(raw: np.ndarray, sig: np.ndarray, rel: np.ndarray, row_offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    power = np.where(sig, np.power(10.0, (raw + rel[None, :] + row_offset[:, None]) / 10.0), 0.0)
    serving = power.max(axis=1)
    return serving, np.maximum(power.sum(axis=1) - serving, 0.0)


# --------------------------------------------------------------------------- calibration + validation

@dataclass
class DriveTestCalibration:
    offsets_db: np.ndarray
    sigma_rsrp_db: float
    interference_load_factor: float
    source: str = "none"                    # "drive_test" | "none"
    sigma_source: str = "default"
    load_source: str = "default"
    reference_offset_db: float = 0.0
    band_offsets_db: dict = field(default_factory=dict)
    per_antenna_offsets: bool = False
    rows_in_area: int = 0
    rows_attributed: int = 0
    sessions: int = 0
    cv_folds: int = 0
    heldout: dict = field(default_factory=dict)
    sinr: dict = field(default_factory=dict)
    validation_status: str = VALIDATION_NOT_CHECKED
    validation_failures: list = field(default_factory=list)
    antenna_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    handover_site_pairs: set = field(default_factory=set)

    def summary(self) -> dict:
        skip = {"offsets_db", "antenna_table", "table", "handover_site_pairs"}
        out = {k: v for k, v in self.__dict__.items() if k not in skip}
        out["handover_site_pairs"] = len(self.handover_site_pairs)
        return out


def calibrate_with_drive_test(
    cells: pd.DataFrame,
    dt: pd.DataFrame | None,
    operator: str,
    area_eval_lonlat: BaseGeometry,
    rf: RFParams,
    cal: CalibrationParams,
) -> DriveTestCalibration:
    n_cells = len(cells)
    result = DriveTestCalibration(
        offsets_db=np.zeros(n_cells),
        sigma_rsrp_db=cal.default_sigma_rsrp_db,
        interference_load_factor=rf.interference_load_factor,
        antenna_table=pd.DataFrame(
            {"antenna_key": cells["antenna_key"], "dt_rows": 0, "relative_offset_db": 0.0, "offset_db": 0.0}
        ),
    )
    if dt is None or dt.empty or cells.empty:
        return result
    d = dt[dt["m_alpha_long"].map(normalise_operator) == normalise_operator(operator)].copy()
    if d.empty:
        return result
    result.handover_site_pairs = handover_site_pairs(d, cal.ho_max_gap_s)

    for col in ("lat", "lon", "rsrp", "sinr", "pci"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[d["rsrp"].between(-140.0, -44.0) & d["lat"].notna() & d["lon"].notna()]
    d = d[shapely.contains_xy(area_eval_lonlat, d["lon"].to_numpy(), d["lat"].to_numpy())].reset_index(drop=True)
    raw = raw_rsrp_matrix(cells, d["lat"], d["lon"], rf).astype(float)
    sig = raw > NO_SIGNAL_DBM + 1.0
    reach = sig.any(axis=1)
    d, raw, sig = d[reach].reset_index(drop=True), raw[reach], sig[reach]
    result.rows_in_area = int(len(d))
    if d.empty:
        return result

    result.source = "drive_test"
    measured = d["rsrp"].to_numpy(dtype=float)
    band = d["band"].fillna("?").astype(str).to_numpy()
    enb = enb_keys(d["nodeb_id"]).to_numpy(dtype=object)
    serving = attribute_serving_antenna(cells, raw, enb, d["pci"].to_numpy(dtype=float))
    attributed = serving >= 0
    result.rows_attributed = int(attributed.sum())
    if result.rows_attributed < cal.min_dt_points:
        result.validation_status = VALIDATION_FAIL
        result.validation_failures = [f"ATTRIBUTED_DT_ROWS({result.rows_attributed}<{cal.min_dt_points})"]
        return result

    resid = np.full(len(d), np.nan)
    resid[attributed] = measured[attributed] - raw[np.flatnonzero(attributed), serving[attributed]]

    sessions = sorted(d["session_id"].unique().tolist())
    result.sessions = len(sessions)
    folds = min(cal.cv_folds, len(sessions))
    if folds >= 2:
        fold = d["session_id"].map({s: i % folds for i, s in enumerate(sessions)}).to_numpy()
    else:
        folds, fold = 1, np.zeros(len(d), dtype=int)
    result.cv_folds = folds

    site_names = cells["site_key"].astype(str).to_numpy()
    checkable = pd.Series(enb).isin(set(site_names)).to_numpy()
    measured_sinr = d["sinr"].to_numpy(dtype=float)
    noise_mw = 10.0 ** (rf.noise_per_re_dbm / 10.0)
    loads = np.asarray(cal.load_factor_grid, dtype=float)

    variants = ("operator", "per_antenna")
    pred = {v: np.full(len(d), np.nan) for v in variants}
    pred_site = {v: np.full(len(d), "", dtype=object) for v in variants}
    pred_sinr = {v: np.full((loads.size, len(d)), np.nan) for v in variants}
    for f in range(folds):
        test = fold == f
        train = attributed & (fold != f) if folds > 1 else attributed
        if not test.any() or train.sum() < cal.min_band_rows:
            continue
        band_off, pooled = _band_offsets(resid, band, train, cal.min_band_rows)
        row_off = _row_offsets(band, band_off, pooled)
        rel_antenna, _ = _antenna_offsets(resid, row_off, serving, train, n_cells, cal)
        for variant, rel in (("operator", np.zeros(n_cells)), ("per_antenna", rel_antenna)):
            shifted = np.where(sig[test], raw[test] + rel[None, :], NO_SIGNAL_DBM)
            best = shifted.argmax(axis=1)
            pred[variant][test] = shifted[np.arange(best.size), best] + row_off[test]
            pred_site[variant][test] = site_names[best]
            srv, others = _sinr_parts(raw[test], sig[test], rel, row_off[test])
            for li, load in enumerate(loads):
                pred_sinr[variant][li, test] = 10.0 * np.log10(np.maximum(srv, 1e-300) / (load * others + noise_mw))

    def score(variant: str) -> dict:
        r = measured - pred[variant]
        ok = np.isfinite(r)
        check = ok & checkable
        return {
            "n": int(ok.sum()),
            "bias_db": float(np.median(r[ok])) if ok.any() else float("nan"),
            "robust_sd_db": _robust_sigma(r[ok]),
            "rmse_db": float(np.sqrt(np.mean(r[ok] ** 2))) if ok.any() else float("nan"),
            "site_agreement": float((pred_site[variant][check] == enb[check]).mean()) if check.any() else float("nan"),
        }

    scores = {v: score(v) for v in variants}
    use_antenna = (
        scores["per_antenna"]["robust_sd_db"] < scores["operator"]["robust_sd_db"] - cal.min_per_antenna_gain_db
        and abs(scores["per_antenna"]["bias_db"]) <= abs(scores["operator"]["bias_db"]) + 0.5
    )
    chosen = "per_antenna" if use_antenna else "operator"
    result.per_antenna_offsets = bool(use_antenna)
    result.heldout = {
        **scores[chosen],
        "variant": chosen,
        "in_sample": folds == 1,
        "robust_sd_operator_db": scores["operator"]["robust_sd_db"],
        "robust_sd_per_antenna_db": scores["per_antenna"]["robust_sd_db"],
    }

    band_off, pooled = _band_offsets(resid, band, attributed, cal.min_band_rows)
    row_off = _row_offsets(band, band_off, pooled)
    rel, counts = _antenna_offsets(resid, row_off, serving, attributed, n_cells, cal)
    if not use_antenna:
        rel = np.zeros(n_cells)
    reference = float(band_off[cal.reference_band]) if cal.reference_band and cal.reference_band in band_off.index else pooled
    result.reference_offset_db = reference
    result.band_offsets_db = {str(b): round(float(v), 2) for b, v in band_off.items()}
    result.offsets_db = reference + rel
    result.antenna_table = pd.DataFrame(
        {"antenna_key": cells["antenna_key"], "dt_rows": counts, "relative_offset_db": rel, "offset_db": result.offsets_db}
    )

    sd = result.heldout["robust_sd_db"]
    if np.isfinite(sd):
        result.sigma_rsrp_db = float(np.clip(sd, cal.min_sigma_db, cal.max_sigma_db))
        result.sigma_source = "drive_test_heldout"

    # Several operators log SINR as a constant 0 (JIO in project 193); only real values count.
    sinr_ok = np.isfinite(measured_sinr) & (measured_sinr != 0.0) & (np.abs(measured_sinr) <= 40.0)
    sinr_ok &= np.isfinite(pred_sinr[chosen][0])
    load_index = int(np.argmin(np.abs(loads - result.interference_load_factor)))
    if sinr_ok.sum() >= cal.min_dt_points and sinr_ok.mean() >= 0.5:
        biases = np.array([np.median(measured_sinr[sinr_ok] - pred_sinr[chosen][li, sinr_ok]) for li in range(loads.size)])
        load_index = int(np.argmin(np.abs(biases)))
        result.interference_load_factor = float(loads[load_index])
        result.load_source = "drive_test_sinr"
        result.sinr = {
            "n": int(sinr_ok.sum()),
            "measured_median_db": float(np.median(measured_sinr[sinr_ok])),
            "predicted_median_db": float(np.median(pred_sinr[chosen][load_index, sinr_ok])),
            "bias_db": float(biases[load_index]),
            "robust_sd_db": _robust_sigma(measured_sinr[sinr_ok] - pred_sinr[chosen][load_index, sinr_ok]),
            "bias_by_load_db": {str(l): round(float(b), 2) for l, b in zip(loads, biases)},
        }
    else:
        result.sinr = {"n": int(sinr_ok.sum()), "note": "no usable measured SINR (missing or logged as 0); default load factor"}

    failures = []
    bias = result.heldout["bias_db"]
    if not np.isfinite(bias) or abs(bias) > cal.max_abs_bias_db:
        failures.append(f"HELDOUT_BIAS({bias:.1f}dB)")
    if not np.isfinite(sd) or sd > cal.max_heldout_sigma_db:
        failures.append(f"HELDOUT_SIGMA({sd:.1f}dB)")
    agreement = result.heldout["site_agreement"]
    if np.isfinite(agreement) and agreement < cal.min_site_agreement:
        failures.append(f"SERVING_SITE_AGREEMENT({agreement:.2f})")
    result.validation_failures = failures
    result.validation_status = VALIDATION_FAIL if failures else VALIDATION_PASS

    antenna_names = cells["antenna_key"].to_numpy(dtype=object)
    result.table = pd.DataFrame(
        {
            "session_id": d["session_id"],
            "fold": fold,
            "timestamp": d["timestamp"],
            "lat": d["lat"],
            "lon": d["lon"],
            "band": band,
            "nodeb_id": enb,
            "attributed_antenna": np.where(attributed, antenna_names[np.maximum(serving, 0)], ""),
            "measured_rsrp_dbm": measured,
            "heldout_predicted_rsrp_dbm": pred[chosen],
            "heldout_residual_db": measured - pred[chosen],
            "heldout_predicted_site": pred_site[chosen],
            "measured_sinr_db": measured_sinr,
            "heldout_predicted_sinr_db": pred_sinr[chosen][load_index],
        }
    )
    return result


# --------------------------------------------------------------------------- points

def build_points(
    area_lonlat: BaseGeometry,
    cfg: OverlapConfig,
    building_wkts: list[str] | None,
    forecast: pd.DataFrame | None,
) -> tuple[pd.DataFrame, LocalProjection, dict]:
    grid, proj = build_grid(area_lonlat, cfg.grid_resolution_m, cfg.evaluation_buffer_m)
    report: dict = {"grid_points": int(len(grid)), "grid_points_in_polygon": int(grid["in_polygon"].sum())}

    grid["indoor"] = False
    geoms = [g for g in (parse_geometry(w) for w in (building_wkts or [])) if g is not None and g.area > 0]
    if geoms:
        geoms, swapped = orient_many_lonlat(geoms, area_lonlat)
        tree = shapely.STRtree(geoms)
        hit, _ = tree.query(shapely.points(grid["lon"].to_numpy(), grid["lat"].to_numpy()), predicate="within")
        grid.loc[np.unique(hit), "indoor"] = True
        report.update({"buildings": len(geoms), "buildings_swapped": swapped})
    report["indoor_share"] = float(grid["indoor"].mean())

    grid["users"] = 0.0
    weight_source = "area"
    if forecast is not None and not forecast.empty:
        f = forecast[forecast["operator"].map(normalise_operator) == normalise_operator(cfg.operator)].copy()
        for col in ("lat", "lon", "users_per_grid"):
            f[col] = pd.to_numeric(f[col], errors="coerce")
        f = f.dropna(subset=["lat", "lon", "users_per_grid"])
        if not f.empty:
            # forecast rows repeat per sector of the same location; users belong to the location
            f = f.assign(la=f["lat"].round(6), lo=f["lon"].round(6)).groupby(["la", "lo"], as_index=False)["users_per_grid"].max()
            pid = grid_point_ids(grid, cfg.grid_resolution_m, proj, f["la"], f["lo"])
            f = f[pid >= 0].assign(point_id=pid[pid >= 0])
            users = f.groupby("point_id")["users_per_grid"].mean()
            grid.loc[users.index.to_numpy(), "users"] = users.to_numpy()
            weight_source = "users_per_grid"
            report["points_with_users_data"] = int(len(users))

    mean_users = float(grid.loc[grid["users"] > 0, "users"].mean()) if (grid["users"] > 0).any() else 0.0
    base = 1.0 + cfg.weights.users_weight * grid["users"] / mean_users if mean_users > 0 else 1.0
    grid["weight"] = base * np.where(grid["indoor"], cfg.weights.indoor_multiplier, 1.0)
    report["weight_source"] = weight_source
    return grid, proj, report


# --------------------------------------------------------------------------- network

@dataclass
class NetworkBuild:
    network: Network
    antennas: pd.DataFrame        # every cleaned antenna + distance_to_area_m, in_polygon, net_index (-1 = not in RF)
    drive_test: DriveTestCalibration
    projection: LocalProjection
    area_lonlat: BaseGeometry
    area_eval_lonlat: BaseGeometry
    report: dict


def _rounded(d: dict) -> dict:
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def build_network(
    cfg: OverlapConfig,
    antennas: pd.DataFrame,
    area_lonlat: BaseGeometry,
    dt_serving: pd.DataFrame | None,
    forecast: pd.DataFrame | None,
    building_wkts: list[str] | None,
    log: Callable[[str], None] = print,
) -> NetworkBuild:
    points, proj, report = build_points(area_lonlat, cfg, building_wkts, forecast)
    area_xy = proj.geometry_to_xy(area_lonlat)
    eval_xy = area_xy.buffer(cfg.evaluation_buffer_m) if cfg.evaluation_buffer_m > 0 else area_xy
    area_eval_lonlat = transform(lambda x, y, z=None: proj.to_latlon(x, y)[::-1], eval_xy)

    ant = antennas.copy()
    x, y = proj.to_xy(ant["lat"], ant["lon"])
    ant["distance_to_area_m"] = shapely.distance(area_xy, shapely.points(x, y))
    ant["in_polygon"] = ant["distance_to_area_m"] <= 0.0
    in_net = ant["rf_include"] & (ant["distance_to_area_m"] <= cfg.site_buffer_m)
    ant["net_index"] = -1
    ant.loc[in_net, "net_index"] = np.arange(int(in_net.sum()))
    cells = ant[in_net].reset_index(drop=True)
    if cells.empty:
        raise ValueError("No antennas left for the RF surface after cleaning and the site buffer")
    cells["layer"] = f"{cfg.rf.frequency_mhz:.0f}MHz"

    log(f"[OVERLAP][RF] rsrp matrix points={len(points)} cells={len(cells)} sites={cells['site_key'].nunique()}")
    raw = raw_rsrp_matrix(cells, points["lat"], points["lon"], cfg.rf)
    dt = calibrate_with_drive_test(cells, dt_serving, cfg.operator, area_eval_lonlat, cfg.rf, cfg.calibration)
    cells["calibration_offset_db"] = dt.offsets_db
    cells["dt_rows"] = dt.antenna_table["dt_rows"].to_numpy()
    rsrp = np.where(raw > NO_SIGNAL_DBM + 1.0, raw + dt.offsets_db[None, :], NO_SIGNAL_DBM).astype(np.float32)
    log(
        f"[OVERLAP][CALIBRATION] source={dt.source} rows_in_area={dt.rows_in_area} attributed={dt.rows_attributed} "
        f"sessions={dt.sessions} reference_offset={dt.reference_offset_db:.1f}dB bands={dt.band_offsets_db} "
        f"per_antenna={dt.per_antenna_offsets}"
    )
    log(
        f"[OVERLAP][VALIDATION] status={dt.validation_status} failures={dt.validation_failures} "
        f"heldout={_rounded(dt.heldout)} sigma_rsrp={dt.sigma_rsrp_db:.1f}dB ({dt.sigma_source}) "
        f"load={dt.interference_load_factor} ({dt.load_source})"
    )

    network = Network(
        points=points,
        cells=cells,
        rsrp_dbm=rsrp,
        grid_resolution_m=cfg.grid_resolution_m,
        noise_per_re_dbm=cfg.rf.noise_per_re_dbm,
        interference_load_factor=dt.interference_load_factor,
        sigma_rsrp_db=dt.sigma_rsrp_db,
    )
    return NetworkBuild(network, ant, dt, proj, area_lonlat, area_eval_lonlat, report)
