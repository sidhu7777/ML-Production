"""Phase 48 two-level residual calibration for the production offset path."""
from __future__ import annotations
import numpy as np
import pandas as pd

TECH_BAND_MIN_N = 1
CLUTTER_MIN_N = 30
CEILING_DBM = -44.0
FLOOR_DBM = -120.0

def fit_phase48(dt: pd.DataFrame, physical_col: str, measured_col: str = "rsrp_measured"):
    work = dt[dt.get("split", "train").astype(str).eq("train")].copy() if "split" in dt else dt.copy()
    work = work[work.get("obstruction_branch", pd.Series("clear", index=work.index)).astype(str).ne("indoor")].copy()
    # Phase 48 defines error as prediction minus measurement. Negating its
    # median below therefore moves the prediction towards the measurement.
    work["_residual"] = pd.to_numeric(work[physical_col], errors="coerce") - pd.to_numeric(work[measured_col], errors="coerce")
    work = work.dropna(subset=["_residual", "technology", "band"])
    tb = work.groupby(["technology", "band"], dropna=False)["_residual"].median().mul(-1).rename("tech_band_correction_db").to_frame()
    tech = work.groupby("technology", dropna=False)["_residual"].agg(n_train="size", median="median")
    tech["technology_correction_db"] = -tech["median"]
    work = work.join(tb, on=["technology", "band"])
    work["_residual_2"] = work["_residual"] + work["tech_band_correction_db"]
    ct = work.groupby(["technology", "band", "clutter_class"], dropna=False)["_residual_2"].agg(n_train="size", median="median")
    ct["clutter_correction_db"] = -ct["median"]
    ct.loc[ct["n_train"] < CLUTTER_MIN_N, "clutter_correction_db"] = 0.0
    return (
        tb.reset_index(),
        ct[["n_train", "clutter_correction_db"]].reset_index(),
        tech[["n_train", "technology_correction_db"]].reset_index(),
    )

def apply_phase48(
    frame: pd.DataFrame,
    physical_col: str,
    tech_band: pd.DataFrame,
    clutter: pd.DataFrame,
    technology: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out = out.merge(tech_band, on=["technology", "band"], how="left")
    has_exact = out["tech_band_correction_db"].notna()
    out = out.merge(technology[["technology", "technology_correction_db"]], on="technology", how="left")
    has_technology = out["technology_correction_db"].notna()
    out["tech_band_correction_db"] = pd.to_numeric(out["tech_band_correction_db"], errors="coerce")
    out["technology_correction_db"] = pd.to_numeric(out["technology_correction_db"], errors="coerce")
    out["base_calibration_correction_db"] = out["tech_band_correction_db"].where(
        has_exact, out["technology_correction_db"]
    ).fillna(0.0)
    out = out.merge(clutter, on=["technology", "band", "clutter_class"], how="left")
    out["clutter_correction_db"] = pd.to_numeric(out["clutter_correction_db"], errors="coerce").where(has_exact, 0.0).fillna(0.0)
    out["phase48_total_correction_db"] = out["base_calibration_correction_db"] + out["clutter_correction_db"]
    out["final_rsrp_unclipped"] = pd.to_numeric(out[physical_col], errors="coerce") + out["phase48_total_correction_db"]
    out["final_rsrp"] = out["final_rsrp_unclipped"].where(out["final_rsrp_unclipped"] >= FLOOR_DBM).clip(upper=CEILING_DBM)
    out["calibration_status"] = np.select(
        [has_exact, has_technology],
        ["DT_CALIBRATED_TECH_BAND", "DT_CALIBRATED_TECH_FALLBACK"],
        default="UNCALIBRATED_NO_DT",
    )
    return out
