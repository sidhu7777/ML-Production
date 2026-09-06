"""Phase 40: RSRQ/SINR on the Phase 39 fixed-power frequency baseline.

Phase 39 is frozen.  Phase 40 reuses its fixed 46 dBm power policy and
frequency treatment, but evaluates quality independently:
  * grid serving cell = strongest Phase 39 candidate;
  * DT serving cell = the Phase 38/39 matched DT serving cell;
  * every same-carrier sector is eligible as interference, with no top-N cap;
  * activity and RSRQ/SINR residuals are fitted per carrier from training DT;
  * no KNN/local residual field or display smoothing is used.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import test_project210_phase37_quality_readiness as phase37
import test_project210_phase39_equal_power_diagnostic as phase39

PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
PHASE39_DIR = PROJECT_DIR / "cost231_phase39_equal_power_diagnostic"
OUT_DIR = PROJECT_DIR / "cost231_phase40_fixed_power_quality"

INTERFERER_MIN_DBM = -125.0
NR_INTERFERER_RELATIVE_CUTOFF_DB_DEFAULT = 24.0
LTE_INTERFERER_RELATIVE_CUTOFF_DB_DEFAULT = 15.0


def _carrier_key(frame: pd.DataFrame) -> pd.Series:
    frequency = pd.to_numeric(
        frame.get("original_frequency_mhz", frame.get("frequency_mhz")), errors="coerce"
    ).round(1)
    return frame["technology"].astype(str) + "|" + frequency.astype("string")


def _phase39_candidates(technology: str) -> pd.DataFrame:
    frame = pd.read_parquet(PHASE39_DIR / "phase39_scored_candidates_project210.parquet").copy()
    frame = frame[frame["technology"].astype(str).eq(technology)].copy()
    frame["phase40_rsrp_dbm"] = pd.to_numeric(frame["phase39_final_rsrp_unclipped"], errors="coerce")
    frame = frame.dropna(subset=["grid_id", "strict_cell_key", "phase40_rsrp_dbm"]).copy()
    frame["carrier_key"] = _carrier_key(frame)
    # Reuse the Phase 37 exact-coordinate transfer helpers without changing Phase 37.
    frame["phase37_rsrp_dbm"] = frame["phase40_rsrp_dbm"]
    return frame


def _phase39_dt_all() -> pd.DataFrame:
    """Rebuild Phase 39 train and validation scores in memory without changing it."""
    raw_dt = pd.read_parquet(phase39.p36.PHASE26_DIR / "phase26_dt_scored_project210.parquet")
    cand_raw = pd.read_parquet(phase39.p36.PHASE26_DIR / "phase26_scored_candidates_project210.parquet")
    dt_rematched = phase39.p38._rematch_4g(raw_dt, cand_raw)
    dt = phase39.p38._dt_inputs_from(dt_rematched)
    dt = phase39._apply_equal_power_assumptions(dt, "assigned_strict_cell_key")
    train = dt[dt["phase25_split"].astype(str).eq("train")].copy()
    fit = train[
        (train["obstruction_branch"].astype(str) != "indoor")
        & (~train["p36_backlobe"].astype(bool))
        & (~train["p38_excluded"].astype(bool))
    ].copy()
    layers, local_models = phase39.p36._fit(fit)
    scored = phase39._copy_phase39_score_columns(phase39.p36._score(dt, layers, local_models))
    scored["split"] = scored["phase25_split"].astype(str)
    scored["carrier_key"] = _carrier_key(scored)
    return scored


def _active_mask(values: np.ndarray, signal_dbm: float, cutoff_db: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.isfinite(values) & (values >= max(INTERFERER_MIN_DBM, signal_dbm - float(cutoff_db)))


def _grid_quality(technology: str, candidates: pd.DataFrame, inventory: pd.DataFrame, cutoff_db: float) -> pd.DataFrame:
    serving = pd.read_parquet(PHASE39_DIR / f"phase39_serving_grid_{technology.lower()}_project210.parquet")
    points = serving[["grid_id", "center_lat", "center_lon", "serving_environment"]].drop_duplicates("grid_id").reset_index(drop=True)
    cells = inventory[inventory["technology"].astype(str).eq(technology)].copy().reset_index(drop=True)
    lat = pd.to_numeric(points["center_lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(points["center_lon"], errors="coerce").to_numpy(float)
    matrix, carriers, keys = phase37._cochannel_score_matrix(lat, lon, cells, candidates)
    best = np.nanargmax(matrix, axis=1)
    rows = []
    for i, server_i in enumerate(best):
        carrier = carriers[server_i]
        members = np.flatnonzero(carriers == carrier)
        signal = float(matrix[i, server_i])
        other_indices = members[members != server_i]
        other = matrix[i, other_indices]
        active = _active_mask(other, signal, cutoff_db)
        interference = float(phase37._mw(other[active]).sum()) if active.any() else 0.0
        rows.append({
            "technology": technology, "grid_id": points.at[i, "grid_id"], "lat": lat[i], "lon": lon[i],
            "carrier_key": carrier, "serving_strict_cell_key": keys[server_i], "serving_rsrp_dbm": signal,
            "serving_environment": points.at[i, "serving_environment"],
            "eligible_cochannel_sector_count": int(len(members)),
            "candidate_interfering_sector_count": int(len(other_indices)),
            "interfering_sector_count": int(active.sum()), "interference_sum_mw": interference,
        })
    return pd.DataFrame(rows)


def _dt_quality(technology: str, candidates: pd.DataFrame, inventory: pd.DataFrame, dt_all: pd.DataFrame, cutoff_db: float) -> pd.DataFrame:
    dt = dt_all[dt_all["technology"].astype(str).eq(technology)].copy().reset_index(drop=True)
    cells = inventory[inventory["technology"].astype(str).eq(technology)].copy().reset_index(drop=True)
    key_to_index = {str(key): index for index, key in enumerate(cells["strict_cell_key"].astype(str))}
    lat = pd.to_numeric(dt["lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(dt["lon"], errors="coerce").to_numpy(float)
    matrix, carriers, keys = phase37._cochannel_score_matrix(lat, lon, cells, candidates)
    rows = []
    for i, row in dt.iterrows():
        assigned_key = str(row["assigned_strict_cell_key"])
        server_i = key_to_index.get(assigned_key)
        fallback = server_i is None
        if fallback:
            server_i = int(np.nanargmax(matrix[i]))
        carrier = carriers[server_i]
        members = np.flatnonzero(carriers == carrier)
        other_indices = members[members != server_i]
        # The matching DT scorer is the authoritative serving RSRP at this point.
        signal = float(pd.to_numeric(pd.Series([row.get("phase39_final_rsrp_unclipped")]), errors="coerce").iloc[0])
        if not np.isfinite(signal):
            signal = float(matrix[i, server_i])
        other = matrix[i, other_indices]
        active = _active_mask(other, signal, cutoff_db)
        interference = float(phase37._mw(other[active]).sum()) if active.any() else 0.0
        rows.append({
            "technology": technology, "dt_row_id": row["dt_row_id"], "split": row["split"],
            "lat": lat[i], "lon": lon[i], "carrier_key": carrier,
            "assigned_strict_cell_key": assigned_key, "phase40_model_serving_cell": keys[server_i],
            "serving_model_matches_assigned": not fallback,
            "phase40_serving_rsrp_dbm": signal,
            "rsrp_measured": pd.to_numeric(pd.Series([row.get("rsrp")]), errors="coerce").iloc[0],
            "rsrq_measured": pd.to_numeric(pd.Series([row.get("rsrq")]), errors="coerce").iloc[0],
            "sinr_measured": pd.to_numeric(pd.Series([row.get("sinr")]), errors="coerce").iloc[0],
            "eligible_cochannel_sector_count": int(len(members)),
            "candidate_interfering_sector_count": int(len(other_indices)),
            "interfering_sector_count": int(active.sum()), "interference_sum_mw": interference,
        })
    return pd.DataFrame(rows)


def _quality_values(signal: np.ndarray, interference: np.ndarray, activity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = [phase37._quality_base(float(s), max(float(i) * activity, 0.0)) for s, i in zip(signal, interference)]
    return (
        np.asarray([value[0] for value in values]),
        np.asarray([value[1] for value in values]),
        np.asarray([value[2] for value in values]),
    )


def _with_quality_base(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    signal_col = "phase40_serving_rsrp_dbm" if "phase40_serving_rsrp_dbm" in out else "serving_rsrp_dbm"
    signal = pd.to_numeric(out[signal_col], errors="coerce").to_numpy(float)
    interference = pd.to_numeric(out["interference_sum_mw"], errors="coerce").to_numpy(float)
    sinr, rsrq, interference_dbm = _quality_values(signal, interference, 1.0)
    out["sinr_base_db"] = sinr
    out["rsrq_base_db"] = rsrq
    out["interference_sum_dbm"] = interference_dbm
    out["quality_signal_bucket"] = pd.cut(
        signal, [-np.inf, -110, -100, -90, -80, -70, np.inf],
        labels=["lt_-110", "-110_-100", "-100_-90", "-90_-80", "-80_-70", "gt_-70"],
    ).astype("string")
    relative_interference = interference_dbm - signal
    out["quality_interference_bucket"] = pd.cut(
        relative_interference, [-np.inf, -25, -15, -8, -3, 3, np.inf],
        labels=["lt_-25", "-25_-15", "-15_-8", "-8_-3", "-3_3", "gt_3"],
    ).astype("string")
    serving_key = out.get("assigned_strict_cell_key", out.get("serving_strict_cell_key", pd.Series("", index=out.index)))
    out["quality_serving_key"] = serving_key.astype(str)
    return out


def _residual_table(
    frame: pd.DataFrame,
    keys: list[str],
    sinr_base_col: str,
    rsrq_base_col: str,
    min_n: int,
    shrink_n: float,
    layer: str,
) -> pd.DataFrame:
    rows = []
    usable = frame.dropna(subset=keys).copy()
    for key, group in usable.groupby(keys, dropna=False):
        sinr_res = (
            pd.to_numeric(group["sinr_measured"], errors="coerce")
            - pd.to_numeric(group[sinr_base_col], errors="coerce")
        ).replace([np.inf, -np.inf], np.nan).dropna()
        rsrq_res = (
            pd.to_numeric(group["rsrq_measured"], errors="coerce")
            - pd.to_numeric(group[rsrq_base_col], errors="coerce")
        ).replace([np.inf, -np.inf], np.nan).dropna()
        n = int(min(len(sinr_res), len(rsrq_res)))
        if n < min_n:
            continue
        shrink = float(n / (n + shrink_n))
        if not isinstance(key, tuple):
            key = (key,)
        row = {name: value for name, value in zip(keys, key)}
        row.update({
            "layer": layer,
            "sinr_correction_db": float(np.median(sinr_res) * shrink) if len(sinr_res) else np.nan,
            "rsrq_correction_db": float(np.median(rsrq_res) * shrink) if len(rsrq_res) else np.nan,
            "dt_n": n,
            "shrink": shrink,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_calibration(dt: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train = _with_quality_base(dt[dt["split"].astype(str).eq("train")])
    carrier = _residual_table(
        train, ["technology", "carrier_key"], "sinr_base_db", "rsrq_base_db",
        min_n=25, shrink_n=20.0, layer="carrier",
    )
    work = train.merge(
        carrier[["technology", "carrier_key", "sinr_correction_db", "rsrq_correction_db"]].rename(
            columns={"sinr_correction_db": "carrier_sinr_correction_db", "rsrq_correction_db": "carrier_rsrq_correction_db"}
        ),
        on=["technology", "carrier_key"], how="left",
    )
    work["sinr_after_carrier_db"] = work["sinr_base_db"] + pd.to_numeric(work["carrier_sinr_correction_db"], errors="coerce").fillna(0.0)
    work["rsrq_after_carrier_db"] = work["rsrq_base_db"] + pd.to_numeric(work["carrier_rsrq_correction_db"], errors="coerce").fillna(0.0)
    sector = _residual_table(
        work, ["technology", "carrier_key", "quality_serving_key"], "sinr_after_carrier_db", "rsrq_after_carrier_db",
        min_n=30, shrink_n=60.0, layer="serving_sector",
    )
    work = work.merge(
        sector[["technology", "carrier_key", "quality_serving_key", "sinr_correction_db", "rsrq_correction_db"]].rename(
            columns={"sinr_correction_db": "sector_sinr_correction_db", "rsrq_correction_db": "sector_rsrq_correction_db"}
        ),
        on=["technology", "carrier_key", "quality_serving_key"], how="left",
    )
    work["sinr_after_sector_db"] = work["sinr_after_carrier_db"] + pd.to_numeric(work["sector_sinr_correction_db"], errors="coerce").fillna(0.0)
    work["rsrq_after_sector_db"] = work["rsrq_after_carrier_db"] + pd.to_numeric(work["sector_rsrq_correction_db"], errors="coerce").fillna(0.0)
    bucket = _residual_table(
        work,
        ["technology", "carrier_key", "quality_signal_bucket", "quality_interference_bucket"],
        "sinr_after_sector_db", "rsrq_after_sector_db",
        min_n=45, shrink_n=90.0, layer="signal_interference_bucket",
    )
    return {"carrier": carrier, "sector": sector, "bucket": bucket}


def _apply_calibration(frame: pd.DataFrame, calibration: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = _with_quality_base(frame)
    carrier = calibration.get("carrier", pd.DataFrame())
    sector = calibration.get("sector", pd.DataFrame())
    bucket = calibration.get("bucket", pd.DataFrame())
    if not carrier.empty:
        out = out.merge(
            carrier[["technology", "carrier_key", "sinr_correction_db", "rsrq_correction_db"]].rename(
                columns={"sinr_correction_db": "carrier_sinr_correction_db", "rsrq_correction_db": "carrier_rsrq_correction_db"}
            ),
            on=["technology", "carrier_key"], how="left", validate="many_to_one",
        )
    if not sector.empty:
        out = out.merge(
            sector[["technology", "carrier_key", "quality_serving_key", "sinr_correction_db", "rsrq_correction_db"]].rename(
                columns={"sinr_correction_db": "sector_sinr_correction_db", "rsrq_correction_db": "sector_rsrq_correction_db"}
            ),
            on=["technology", "carrier_key", "quality_serving_key"], how="left", validate="many_to_one",
        )
    if not bucket.empty:
        out = out.merge(
            bucket[["technology", "carrier_key", "quality_signal_bucket", "quality_interference_bucket", "sinr_correction_db", "rsrq_correction_db"]].rename(
                columns={"sinr_correction_db": "bucket_sinr_correction_db", "rsrq_correction_db": "bucket_rsrq_correction_db"}
            ),
            on=["technology", "carrier_key", "quality_signal_bucket", "quality_interference_bucket"], how="left", validate="many_to_one",
        )
    for column in (
        "carrier_sinr_correction_db", "sector_sinr_correction_db", "bucket_sinr_correction_db",
        "carrier_rsrq_correction_db", "sector_rsrq_correction_db", "bucket_rsrq_correction_db",
    ):
        if column not in out:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["sinr_correction_db"] = out[["carrier_sinr_correction_db", "sector_sinr_correction_db", "bucket_sinr_correction_db"]].sum(axis=1)
    out["rsrq_correction_db"] = out[["carrier_rsrq_correction_db", "sector_rsrq_correction_db", "bucket_rsrq_correction_db"]].sum(axis=1)
    out["pred_sinr_db"] = (out["sinr_base_db"] + out["sinr_correction_db"]).clip(-20.0, 35.0)
    out["pred_rsrq_db"] = (out["rsrq_base_db"] + out["rsrq_correction_db"]).clip(-19.5, -3.0)
    out["quality_status"] = np.select(
        [
            out["bucket_sinr_correction_db"].ne(0.0) | out["bucket_rsrq_correction_db"].ne(0.0),
            out["sector_sinr_correction_db"].ne(0.0) | out["sector_rsrq_correction_db"].ne(0.0),
            out["carrier_sinr_correction_db"].ne(0.0) | out["carrier_rsrq_correction_db"].ne(0.0),
        ],
        ["READY_BUCKET_CALIBRATED", "READY_SECTOR_CALIBRATED", "READY_CARRIER_CALIBRATED"],
        default="INSUFFICIENT_DT",
    )
    return out


def _mae(frame: pd.DataFrame, measured: str, predicted: str) -> float | None:
    delta = pd.to_numeric(frame[measured], errors="coerce") - pd.to_numeric(frame[predicted], errors="coerce")
    delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
    return round(float(delta.abs().mean()), 3) if not delta.empty else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.concat([_phase39_candidates("4G"), _phase39_candidates("5G")], ignore_index=True)
    inventory = phase37._phase37_inventory(candidates)
    dt_all = _phase39_dt_all()
    interferer_cutoff_db = {}
    interferer_cutoff_sweep = {}
    for technology, default_cutoff in {
        "4G": LTE_INTERFERER_RELATIVE_CUTOFF_DB_DEFAULT,
        "5G": NR_INTERFERER_RELATIVE_CUTOFF_DB_DEFAULT,
    }.items():
        tech_candidates = candidates[candidates["technology"].astype(str).eq(technology)].copy()
        tech_inventory = inventory[inventory["technology"].astype(str).eq(technology)].copy()
        cutoff, trials = phase37._fit_interferer_cutoff(
            dt_all[dt_all["technology"].astype(str).eq(technology)],
            tech_inventory,
            tech_candidates,
            default_cutoff,
        )
        interferer_cutoff_db[technology] = float(cutoff)
        interferer_cutoff_sweep[technology] = trials
    grid = pd.concat([
        _grid_quality(tech, candidates[candidates["technology"].eq(tech)], inventory, interferer_cutoff_db[tech])
        for tech in ("4G", "5G")
    ], ignore_index=True)
    dt = pd.concat([
        _dt_quality(tech, candidates[candidates["technology"].eq(tech)], inventory, dt_all, interferer_cutoff_db[tech])
        for tech in ("4G", "5G")
    ], ignore_index=True)
    calibration = _fit_calibration(dt)
    grid = _apply_calibration(grid, calibration)
    dt = _apply_calibration(dt, calibration)
    validation = dt[dt["split"].astype(str).eq("validation")].copy()

    grid.to_parquet(OUT_DIR / "phase40_serving_quality_project210.parquet", index=False)
    grid.to_csv(OUT_DIR / "phase40_serving_quality_project210.csv", index=False)
    dt.to_parquet(OUT_DIR / "phase40_dt_quality_project210.parquet", index=False)
    calibration_frame = pd.concat(
        [table for table in calibration.values() if table is not None and not table.empty],
        ignore_index=True,
    ) if any(table is not None and not table.empty for table in calibration.values()) else pd.DataFrame()
    calibration_frame.to_csv(OUT_DIR / "phase40_carrier_quality_calibration.csv", index=False)
    pd.DataFrame().to_csv(OUT_DIR / "phase40_sector_activity_profiles.csv", index=False)
    for technology in ("4G", "5G"):
        dt[dt["technology"].eq(technology)].to_parquet(OUT_DIR / f"phase40_dt_quality_{technology.lower()}_project210.parquet", index=False)

    carrier_summary = []
    for (technology, carrier), group in grid.groupby(["technology", "carrier_key"], dropna=False):
        val = validation[(validation["technology"].eq(technology)) & (validation["carrier_key"].eq(carrier))]
        carrier_summary.append({
            "technology": technology, "carrier_key": carrier, "grids": int(len(group)),
            "median_cochannel_sectors": float(group["eligible_cochannel_sector_count"].median()),
            "median_active_interferers": float(group["interfering_sector_count"].median()),
            "median_interference_dbm": round(float(pd.to_numeric(group["interference_sum_dbm"], errors="coerce").median()), 3),
            "rsrq_validation_mae_db": _mae(val, "rsrq_measured", "pred_rsrq_db"),
            "sinr_validation_mae_db": _mae(val, "sinr_measured", "pred_sinr_db"),
        })
    summary = {
        "scope": "Phase 40 only. Frozen Phase 39 fixed-power/frequency candidate and DT baseline; Phase 39 unchanged.",
        "serving_policy": "Grid uses strongest Phase39 candidate. DT validation uses matched DT serving sector; that sector is excluded from same-carrier interference.",
        "interference_policy": "All same-carrier inventory sectors are evaluated, no top-N cap; active set is signal-cutoff dB with -125 dBm floor. Cutoff is fitted per technology from training DT, same Phase37 method.",
        "quality_status": "READY",
        "missing_quality_inputs": [],
        "quality_model": "Phase37-style DT-fitted interferer cutoff on the frozen Phase39 fixed-power RSRP baseline, plus hierarchical carrier/serving-sector/signal-interference residual calibration. No KNN/local residual and no display smoothing.",
        "quality_calibration_rows": int(len(calibration_frame)),
        "interferer_cutoff_db": interferer_cutoff_db,
        "interferer_cutoff_sweep": interferer_cutoff_sweep,
        "sector_activity_profiles": 0,
        "carrier_summary": carrier_summary,
        "validation_rows": int(len(validation)),
        "dt_serving_inventory_coverage": round(float(dt["serving_model_matches_assigned"].mean()), 4),
    }
    (OUT_DIR / "phase40_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
