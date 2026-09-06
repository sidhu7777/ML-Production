"""Phase 37 RSRQ / SINR for the production offset pipeline.

Consumes the Phase 36 v2 calibrated RSRP surface (``final_rsrp`` per grid x cell)
and the calibrated DT rows, and produces ``pred_rsrq`` / ``pred_sinr``.

RSRQ = 10*log10(N_RB) + RSRP - RSSI            (3GPP TS 36.214 / 38.215)
SINR = serving / (interference + noise)
RSSI = serving + sum(co-channel interference) + noise

Interference = linear sum of every OTHER co-channel sector's calibrated RSRP at
the point, excluding the serving cell, gated so a sector far below the serving
signal cannot dominate.  A per-carrier resource-load / active-interference factor
and a per-carrier RSRQ/SINR residual are learned from the training DT split only.

Carrier with no calibration DT -> pred_rsrq / pred_sinr stay NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NOISE_DBM = -104.0                       # thermal + NF, measurement-bandwidth reference
NOISE_MW = 10.0 ** (NOISE_DBM / 10.0)
INTERFERER_RELATIVE_CUTOFF_DB = 20.0     # a sector > this far below serving is not counted
INTERFERER_MIN_DBM = -125.0
ACTIVITY_MIN, ACTIVITY_MAX = 0.03, 1.0
# UE receiver implementation ceiling (EVM / self-interference). Without it an
# isolated strong grid over the thermal floor alone reports an unphysical
# +60 dB SINR that no handset would ever log.
SINR_CEILING_DB = 35.0


def _mw(dbm):
    return np.power(10.0, np.asarray(dbm, dtype=float) / 10.0)


def _carrier_key(frame: pd.DataFrame) -> pd.Series:
    freq = pd.to_numeric(frame.get("original_frequency_mhz", frame.get("serving_frequency_mhz", frame.get("frequency_mhz"))),
                         errors="coerce").round(1)
    return frame["technology"].astype(str) + "|" + freq.astype("string")


def _attach_carrier_key(frame: pd.DataFrame, cell_carrier_map) -> pd.Series:
    """carrier_key from the authoritative per-cell map (technology + real/deployed
    frequency, keyed by strict_cell_key) when available; falls back to the
    frame's own frequency columns otherwise.

    The map is the reliable source: a cell's real frequency is fixed once at
    site-row preparation (_prepare_site_rows), but the frame's own
    original_frequency_mhz/frequency_mhz columns do not consistently survive
    every join between there and here (e.g. the DT-scoring path can end up
    only carrying the COST-231 anchor frequency, not the real one), so two
    frames scoring the SAME cell can otherwise compute two different keys for
    it and silently fail to join.
    """
    fallback = _carrier_key(frame)
    if cell_carrier_map is None or getattr(cell_carrier_map, "empty", True) or "strict_cell_key" not in frame.columns:
        return fallback
    mapped = frame[["strict_cell_key"]].merge(
        cell_carrier_map[["strict_cell_key", "carrier_key"]], on="strict_cell_key", how="left"
    )
    mapped.index = frame.index
    return mapped["carrier_key"].where(mapped["carrier_key"].notna(), fallback)


def _point_quality(signal_dbm, interference_mw):
    """SIR-like SINR and a normalized RSRQ, both from signal/interference/noise only.

    The DT-calibrated residual (fit in compute_quality) absorbs the unobserved
    RSSI/resource-load term. This intentionally does not add 10*log10(N_RB) to a
    reference-signal-only RSSI estimate -- that shortcut assumed a fixed PRB
    count per technology (n_rb_for(tech, None) has no real per-carrier
    bandwidth) and is not a valid RSSI proxy.
    """
    if not np.isfinite(signal_dbm):
        return np.nan, np.nan
    interference_mw = float(interference_mw) if np.isfinite(interference_mw) and interference_mw > 0.0 else 0.0
    s = float(_mw(signal_dbm))
    denom_mw = interference_mw + NOISE_MW
    total_mw = s + denom_mw
    sinr = min(10.0 * np.log10(s / denom_mw), SINR_CEILING_DB)
    rsrq = 10.0 * np.log10(s / total_mw)
    return sinr, rsrq


def _score_points(scored: pd.DataFrame, serving_col: str) -> pd.DataFrame:
    """One row per (technology, carrier, grid) with serving + interference."""
    work = scored.copy()
    if "carrier_key" not in work.columns:
        work["carrier_key"] = _carrier_key(work)
    work["_rsrp"] = pd.to_numeric(work[serving_col], errors="coerce")
    work = work.dropna(subset=["_rsrp", "grid_id", "carrier_key"])
    rows = []
    for (carrier, grid), grp in work.groupby(["carrier_key", "grid_id"], sort=False):
        vals = grp["_rsrp"].to_numpy(float)
        srv = int(np.nanargmax(vals))
        signal = float(vals[srv])
        others = np.delete(vals, srv)
        gate = (others >= max(INTERFERER_MIN_DBM, signal - INTERFERER_RELATIVE_CUTOFF_DB))
        interference_mw = float(_mw(others[gate]).sum()) if gate.any() else 0.0
        rows.append({
            "technology": str(grp["technology"].iloc[0]),
            "carrier_key": carrier,
            "grid_id": grid,
            "serving_rsrp_dbm": signal,
            "interference_sum_mw": interference_mw,
            "interfering_sector_count": int(gate.sum()),
            "cochannel_sector_count": int(len(grp)),
        })
    return pd.DataFrame(rows)


def _fit_activity(group: pd.DataFrame) -> float:
    sinr = pd.to_numeric(group.get("sinr_measured"), errors="coerce").to_numpy(float)
    sig = _mw(pd.to_numeric(group.get("serving_rsrp_dbm"), errors="coerce").to_numpy(float))
    intf = pd.to_numeric(group.get("interference_sum_mw"), errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        needed = sig / (np.power(10.0, sinr / 10.0) * intf)
    needed = needed[np.isfinite(needed) & (needed > 0.0)]
    return float(np.clip(np.median(needed), ACTIVITY_MIN, ACTIVITY_MAX)) if needed.size else 1.0


def compute_quality(surface_scored: pd.DataFrame, dt_scored: pd.DataFrame,
                    serving_col: str = "final_rsrp",
                    cell_carrier_map: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (surface with pred_rsrq/pred_sinr, per-carrier calibration table).

    cell_carrier_map (optional): authoritative strict_cell_key -> carrier_key
    table (see _attach_carrier_key). Pass this whenever the caller's own
    surface_scored / dt_scored frames may not carry a consistent real-frequency
    column all the way through -- without it, a cell can silently fail to join
    its own DT calibration if surface_scored and dt_scored disagree on its
    frequency (e.g. one has the real deployed frequency, the other only the
    COST-231 model anchor).
    """
    surface_scored = surface_scored.copy()
    dt_scored = dt_scored.copy()
    surface_scored["carrier_key"] = _attach_carrier_key(surface_scored, cell_carrier_map)
    dt_scored["carrier_key"] = _attach_carrier_key(dt_scored, cell_carrier_map)

    grid_q = _score_points(surface_scored, serving_col)
    dt_q = _score_points(
        dt_scored.assign(grid_id=dt_scored.get("grid_id", "DT_" + dt_scored.get("dt_row_id", pd.Series(range(len(dt_scored)))).astype(str))),
        serving_col,
    )
    # bring measured rsrq/sinr + split onto the DT quality rows (carrier_key
    # already attached above, consistently with dt_q)
    meas = dt_scored.copy()
    keep = [c for c in ("grid_id", "carrier_key", "rsrq", "sinr", "rsrq_measured", "sinr_measured", "split") if c in meas.columns]
    dt_q = dt_q.merge(meas[keep].drop_duplicates(["grid_id", "carrier_key"]), on=["grid_id", "carrier_key"], how="left")
    dt_q["sinr_measured"] = pd.to_numeric(dt_q.get("sinr_measured", dt_q.get("sinr")), errors="coerce")
    dt_q["rsrq_measured"] = pd.to_numeric(dt_q.get("rsrq_measured", dt_q.get("rsrq")), errors="coerce")
    if "split" not in dt_q.columns:
        dt_q["split"] = "train"

    cal_rows = []
    for (tech, carrier), grp in dt_q[dt_q["split"].astype(str).eq("train")].groupby(["technology", "carrier_key"], dropna=False):
        activity = _fit_activity(grp)
        base = [_point_quality(s, max(i * activity, 0.0))
                for s, i in zip(grp["serving_rsrp_dbm"], grp["interference_sum_mw"])]
        sinr_b = np.array([b[0] for b in base]); rsrq_b = np.array([b[1] for b in base])
        sinr_res = (pd.to_numeric(grp["sinr_measured"], errors="coerce").to_numpy() - sinr_b)
        rsrq_res = (pd.to_numeric(grp["rsrq_measured"], errors="coerce").to_numpy() - rsrq_b)
        sinr_res = sinr_res[np.isfinite(sinr_res)]; rsrq_res = rsrq_res[np.isfinite(rsrq_res)]
        cal_rows.append({
            "technology": tech, "carrier_key": carrier,
            "activity_factor": activity,
            "sinr_correction_db": float(np.median(sinr_res)) if sinr_res.size else np.nan,
            "rsrq_correction_db": float(np.median(rsrq_res)) if rsrq_res.size else np.nan,
            "sinr_dt_n": int(sinr_res.size), "rsrq_dt_n": int(rsrq_res.size),
        })
    calibration = pd.DataFrame(cal_rows)

    def _apply(q: pd.DataFrame) -> pd.DataFrame:
        m = q.merge(calibration, on=["technology", "carrier_key"], how="left")
        act = pd.to_numeric(m["activity_factor"], errors="coerce").fillna(1.0).to_numpy()
        out_sinr = np.full(len(m), np.nan); out_rsrq = np.full(len(m), np.nan)
        for i, (s, inter) in enumerate(zip(m["serving_rsrp_dbm"], m["interference_sum_mw"])):
            sinr_b, rsrq_b = _point_quality(s, max(inter * act[i], 0.0))
            out_sinr[i] = sinr_b
            out_rsrq[i] = rsrq_b
        m["pred_sinr"] = out_sinr + pd.to_numeric(m["sinr_correction_db"], errors="coerce")
        m["pred_rsrq"] = out_rsrq + pd.to_numeric(m["rsrq_correction_db"], errors="coerce")
        return m

    grid_out = _apply(grid_q)
    # NOTE: must join on carrier_key too, not just technology + grid_id. A grid
    # can be covered by more than one carrier of the same technology (e.g. 4G
    # band 3 and band 28 both reaching the same point); joining on technology +
    # grid_id alone matches every carrier's surface row against every carrier's
    # quality row for that grid (a many-to-many fan-out), duplicating rows and
    # mixing one carrier's pred_rsrq/pred_sinr onto another carrier's surface row.
    surface = surface_scored  # carrier_key already attached above, consistently with grid_out
    surface = surface.merge(
        grid_out[["technology", "carrier_key", "grid_id", "pred_rsrq", "pred_sinr",
                  "interfering_sector_count", "serving_rsrp_dbm"]].rename(columns={"serving_rsrp_dbm": "quality_serving_rsrp_dbm"}),
        on=["technology", "carrier_key", "grid_id"], how="left",
        suffixes=("", "_q"),
    )
    return surface, calibration
