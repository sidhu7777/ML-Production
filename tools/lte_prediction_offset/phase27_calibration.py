"""Project-scoped Phase-27-style residual calibration.

This module deliberately stores no country or project correction values.  It
fits compact, support-gated residual tables from the current project's outdoor
drive-test rows and applies them to candidate-sector predictions.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:  # optional - only needed for the Phase 36 v2 local residual field
    from sklearn.neighbors import BallTree
except Exception:  # pragma: no cover
    BallTree = None


TECH_BAND_MIN_N = 80
FEATURE_MIN_N = 35
SECTOR_MIN_N = 30
TECH_BAND_SHRINK_N = 120.0
FEATURE_SHRINK_N = 60.0
SECTOR_SHRINK_N = 45.0


def _fit_layer(train: pd.DataFrame, keys: list[str], residual_col: str, min_n: int, shrink_n: float, name: str) -> pd.DataFrame:
    table = (train.dropna(subset=[residual_col]).groupby(keys, dropna=False)[residual_col]
             .agg(n_train="size", median_residual_db="median").reset_index())
    table = table[table["n_train"] >= min_n].copy()
    table["shrink_factor"] = table["n_train"] / (table["n_train"] + shrink_n)
    table["correction_db"] = table["median_residual_db"] * table["shrink_factor"]
    table["layer"] = name
    return table


def _attach_layer(frame: pd.DataFrame, table: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    out = frame.copy()
    col = f"{name}_correction_db"
    support = f"{name}_n_train"
    if table.empty:
        out[col], out[support] = 0.0, 0
        return out
    joined = table[keys + ["n_train", "correction_db"]].rename(columns={"n_train": support, "correction_db": col})
    out = out.merge(joined, on=keys, how="left")
    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out[support] = pd.to_numeric(out[support], errors="coerce").fillna(0).astype(int)
    return out


def add_features(df: pd.DataFrame, sector_col: str) -> pd.DataFrame:
    out = df.copy()
    out["technology"] = out["technology"].astype(str)
    out["band"] = out.get("band", pd.Series("UNKNOWN", index=out.index)).astype(str)
    out["sector_key"] = out[sector_col].astype(str)
    out["clutter_class"] = out.get("clutter_class", pd.Series("UNKNOWN", index=out.index)).fillna("UNKNOWN").astype(str)
    out["obstruction_branch"] = out.get("obstruction_branch", pd.Series("clear", index=out.index)).fillna("clear").astype(str)
    terrain = pd.to_numeric(
        out.get("terrain_diffraction_loss_db", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    out["terrain_bucket"] = pd.cut(terrain, [-np.inf, .5, 3., 10., np.inf], labels=["none", "low", "medium", "high"]).astype(str)
    return out


def fit_outdoor(dt: pd.DataFrame, physical_col: str, measured_col: str = "rsrp_measured") -> list[pd.DataFrame]:
    work = dt[dt["obstruction_branch"].astype(str) != "indoor"].copy()
    work["residual_db"] = pd.to_numeric(work[measured_col], errors="coerce") - pd.to_numeric(work[physical_col], errors="coerce")
    specs = [
        ("tech_band", ["technology", "band"], TECH_BAND_MIN_N, TECH_BAND_SHRINK_N),
        ("clutter_terrain", ["technology", "band", "clutter_class", "obstruction_branch", "terrain_bucket"], FEATURE_MIN_N, FEATURE_SHRINK_N),
        ("sector", ["technology", "band", "sector_key"], SECTOR_MIN_N, SECTOR_SHRINK_N),
    ]
    layers = []
    for name, keys, min_n, shrink_n in specs:
        table = _fit_layer(work, keys, "residual_db", min_n, shrink_n, name)
        work = _attach_layer(work, table, keys, name)
        work["residual_db"] = work["residual_db"] - work[f"{name}_correction_db"]
        layers.append(table)
    return layers


# --------------------------------------------------------------------------------
# Phase 36 v2 additions: local IDW residual field + indoor group-bias.
# These are ADDITIVE - the existing fit_outdoor / apply_outdoor are unchanged.
# --------------------------------------------------------------------------------
LOCAL_MIN_NEIGHBORS = 5
LOCAL_K_NEIGHBORS = 16
LOCAL_SHRINK_N = 8.0


def _group_pred(frame: pd.DataFrame, layers: list[pd.DataFrame], physical_col: str) -> pd.Series:
    keys_by_layer = {
        "tech_band": ["technology", "band"],
        "clutter_terrain": ["technology", "band", "clutter_class", "obstruction_branch", "terrain_bucket"],
        "sector": ["technology", "band", "sector_key"],
    }
    out = frame.copy().reset_index(drop=True)
    pred = pd.to_numeric(out[physical_col], errors="coerce")
    for table in layers:
        if table.empty:
            continue
        name = str(table["layer"].iloc[0])
        out = _attach_layer(out, table, keys_by_layer[name], name)
        pred = pred + out[f"{name}_correction_db"]
    return pred


def fit_local(dt: pd.DataFrame, layers: list[pd.DataFrame], physical_col: str,
              measured_col: str = "rsrp_measured") -> dict:
    """Per-technology inverse-distance residual field, fit on the residual left
    after the group hierarchy. Outdoor DT only."""
    if BallTree is None:
        return {}
    work = dt[dt["obstruction_branch"].astype(str) != "indoor"].copy().reset_index(drop=True)
    work["_group_pred"] = _group_pred(work, layers, physical_col).to_numpy()
    work["_local_resid"] = pd.to_numeric(work[measured_col], errors="coerce") - work["_group_pred"]
    work = work.dropna(subset=["lat", "lon", "_local_resid"])
    models: dict = {}
    for tech, grp in work.groupby("technology"):
        if len(grp) < LOCAL_MIN_NEIGHBORS + 1:
            continue
        xy = np.radians(grp[["lat", "lon"]].to_numpy(float))
        tree = BallTree(xy, metric="haversine")
        d, _ = tree.query(xy, k=2)
        near = d[:, 1][np.isfinite(d[:, 1]) & (d[:, 1] > 0)]
        radius = float(max(np.quantile(near, 0.75) * 3.0, 75.0 / 6_371_000.0)) if near.size else (150.0 / 6_371_000.0)
        models[str(tech)] = {
            "tree": tree,
            "resid": grp["_local_resid"].to_numpy(float),
            "radius": radius,
            "n": int(len(grp)),
        }
    return models


def apply_local(candidates: pd.DataFrame, models: dict) -> np.ndarray:
    out = np.zeros(len(candidates), dtype=float)
    if not models:
        return out
    tech = candidates["technology"].astype(str).to_numpy()
    lat = pd.to_numeric(candidates["lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(candidates["lon"], errors="coerce").to_numpy(float)
    for t, model in models.items():
        idx = np.where(tech == t)[0]
        if idx.size == 0:
            continue
        pts = np.radians(np.column_stack([lat[idx], lon[idx]]))
        k = min(LOCAL_K_NEIGHBORS, model["n"])
        dist, near = model["tree"].query(pts, k=k)
        if k == 1:
            dist = dist[:, None]; near = near[:, None]
        valid = np.isfinite(dist) & (dist <= model["radius"]) & (near < model["n"])
        support = valid.sum(axis=1)
        for j in np.where(support >= LOCAL_MIN_NEIGHBORS)[0]:
            vj = valid[j]
            d = dist[j, vj] * 6_371_000.0
            r = model["resid"][near[j, vj]]
            w = 1.0 / np.maximum(d, 5.0)
            local = float(np.average(r, weights=w))
            out[idx[j]] = local * (support[j] / (support[j] + LOCAL_SHRINK_N))
    return out


def apply_outdoor_v2(candidates: pd.DataFrame, layers: list[pd.DataFrame],
                     physical_col: str, local_models: dict | None = None) -> pd.DataFrame:
    """apply_outdoor + local IDW field + indoor keeps the group model-bias
    (tech_band + clutter_terrain) instead of zero correction."""
    out = candidates.copy().reset_index(drop=True)
    keys_by_layer = {
        "tech_band": ["technology", "band"],
        "clutter_terrain": ["technology", "band", "clutter_class", "obstruction_branch", "terrain_bucket"],
        "sector": ["technology", "band", "sector_key"],
    }
    per_layer = {}
    for table in layers:
        if table.empty:
            continue
        name = str(table["layer"].iloc[0])
        out = _attach_layer(out, table, keys_by_layer[name], name)
        per_layer[name] = out[f"{name}_correction_db"].to_numpy(float)
    n = len(out)
    tb = per_layer.get("tech_band", np.zeros(n))
    ct = per_layer.get("clutter_terrain", np.zeros(n))
    sec = per_layer.get("sector", np.zeros(n))
    local = apply_local(out, local_models or {})
    indoor = out["obstruction_branch"].astype(str).eq("indoor").to_numpy()
    full = tb + ct + sec + local
    group_bias = tb + ct
    out["local_residual_correction_db"] = np.where(indoor, 0.0, local)
    out["dynamic_residual_db"] = np.where(indoor, group_bias, full)
    out["final_rsrp_unclipped"] = pd.to_numeric(out[physical_col], errors="coerce") + out["dynamic_residual_db"]
    out["final_rsrp"] = out["final_rsrp_unclipped"].where(out["final_rsrp_unclipped"] >= -140.0)
    # Guard: a (technology, band) with NO tech_band calibration support is not
    # validated - the applied physical (which now carries the -28 dB per-RE term)
    # has no correction to anchor it. Emit NaN rather than an unvalidated value.
    def _support(col):
        s = out[col] if col in out.columns else pd.Series(0.0, index=out.index)
        return pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy()
    uncalibrated = (_support("tech_band_n_train") <= 0) & (_support("clutter_terrain_n_train") <= 0)
    out["calibration_status"] = np.where(uncalibrated, "UNCALIBRATED_NO_DT", "DT_CALIBRATED")
    out.loc[uncalibrated, ["final_rsrp", "final_rsrp_unclipped"]] = np.nan
    return out


def apply_outdoor(candidates: pd.DataFrame, layers: list[pd.DataFrame], physical_col: str) -> pd.DataFrame:
    # Merges below create a new RangeIndex. Reset here and compute the total
    # only after every layer has been attached, so no residual can be lost to
    # pandas label alignment from a filtered/gappy candidate index.
    out = candidates.copy().reset_index(drop=True)
    keys_by_layer = {
        "tech_band": ["technology", "band"],
        "clutter_terrain": ["technology", "band", "clutter_class", "obstruction_branch", "terrain_bucket"],
        "sector": ["technology", "band", "sector_key"],
    }
    correction_cols = []
    for table in layers:
        if table.empty:
            continue
        name = str(table["layer"].iloc[0])
        out = _attach_layer(out, table, keys_by_layer[name], name)
        correction_cols.append(f"{name}_correction_db")
    indoor = out["obstruction_branch"].astype(str).eq("indoor")
    total = out[correction_cols].sum(axis=1) if correction_cols else pd.Series(0.0, index=out.index)
    out["dynamic_residual_db"] = total.where(~indoor, 0.0)
    out["final_rsrp_unclipped"] = pd.to_numeric(out[physical_col], errors="coerce") + out["dynamic_residual_db"]
    out["final_rsrp"] = out["final_rsrp_unclipped"].where(out["final_rsrp_unclipped"] >= -140.0)
    return out
