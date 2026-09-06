from __future__ import annotations

import numpy as np
import pandas as pd


RSRP_NO_COVERAGE_DBM = -140.0
RSRP_MAX_DBM = -44.0


def valid_model_rsrp(values):
    """Upper-clip valid model RSRP and mark sub-floor predictions as no coverage.

    The old Phase17+ pipeline clipped every weak model value up to a floor.
    That made no-coverage look like a real red RSRP pixel and let
    later residual stages repair an invalid physical prediction. Keep values
    below the floor as NaN so maps/CDF/MAE can treat them separately.
    """
    if isinstance(values, pd.Series):
        out = pd.to_numeric(values, errors="coerce").astype(float)
        out = out.where(out >= RSRP_NO_COVERAGE_DBM, np.nan)
        return out.clip(upper=RSRP_MAX_DBM)

    arr = np.asarray(values, dtype=float)
    out = np.where(np.isfinite(arr) & (arr >= RSRP_NO_COVERAGE_DBM), arr, np.nan)
    return np.minimum(out, RSRP_MAX_DBM)


def display_rsrp(values):
    """Display-safe RSRP for measured/locked DT values without inventing coverage."""
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").astype(float).clip(upper=RSRP_MAX_DBM)
    arr = np.asarray(values, dtype=float)
    return np.minimum(arr, RSRP_MAX_DBM)
