from __future__ import annotations

import io
import json
import re
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.colors import BoundaryNorm, ListedColormap

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR / "data" / "project_210_taiwan"
IDENTITY_PATH = PROJECT_DIR / "baseline_fetch_scope" / "site_identity_strict_cells_project210.parquet"
PHASE9_DIR = PROJECT_DIR / "cost231_phase9_gridanalytics_compatible"
PHASE17_DIR = PROJECT_DIR / "cost231_phase17_geo_dt_comparison"
PHASE19_DIR = PROJECT_DIR / "cost231_phase19_branch_calibrated_comparison"
PHASE20_DIR = PROJECT_DIR / "cost231_phase20_5g_real_dt_match"
PHASE21_DIR = PROJECT_DIR / "cost231_phase21_full_polygon_5g_corrected"
PHASE22_DIR = PROJECT_DIR / "cost231_phase22_terrain_diffraction_comparison"
PHASE22_IMAGE_DIR = PHASE22_DIR / "images"
PHASE23_DIR = PROJECT_DIR / "cost231_phase23_outdoor_project_calibration"
PHASE23_IMAGE_DIR = PHASE23_DIR / "images"
PHASE24_DIR = PROJECT_DIR / "cost231_phase24_physical_clutter_role_fix"
PHASE24_IMAGE_DIR = PHASE24_DIR / "images"
PHASE25_DIR = PROJECT_DIR / "cost231_phase25_hierarchical_dynamic_calibration"
PHASE25_IMAGE_DIR = PHASE25_DIR / "images"
PHASE26_DIR = PROJECT_DIR / "cost231_phase26_corrected_obstruction_profile"
PHASE27_DIR = PROJECT_DIR / "cost231_phase27_dynamic_on_corrected_obstruction"
PHASE27_IMAGE_DIR = PHASE27_DIR / "images"
PHASE29_DIR = PROJECT_DIR / "cost231_phase29_real_antenna_pattern"
PHASE29_IMAGE_DIR = PHASE29_DIR / "images"
PHASE31_DIR = PROJECT_DIR / "cost231_phase31_phase28_real_antenna"
PHASE37_DIR = PROJECT_DIR / "cost231_phase37_quality_readiness"
STATIC_MAP_DISPLAY_WIDTH_PX = 430
SITE_TECH_COLORS = {"4G": "#1d4ed8", "5G": "#7c3aed"}

RSRP_BINS = [
    (-140, -115, "#991b1b", "-140 to -115"),
    (-115, -105, "#d97706", "-115 to -105"),
    (-105, -95, "#fef08a", "-105 to -95"),
    (-95, -85, "#22c55e", "-95 to -85"),
    (-85, 0, "#15803d", "-85 to 0"),
]

LOSS_BINS = [
    (0, 3, "#f8fafc", "0 to 3"),
    (3, 10, "#bfdbfe", "3 to 10"),
    (10, 20, "#60a5fa", "10 to 20"),
    (20, 35, "#2563eb", "20 to 35"),
    (35, 46, "#1e3a8a", "35 to 45"),
]

DELTA_BINS = [
    (-40, -20, "#991b1b", "-40 to -20"),
    (-20, -8, "#d97706", "-20 to -8"),
    (-8, 0, "#fef08a", "-8 to 0"),
    (0, 8, "#86efac", "0 to 8"),
    (8, 40, "#15803d", "8 to 40"),
]

CONFIDENCE_BINS = [
    (0.0, 0.4, "#991b1b", "0.0 to 0.4"),
    (0.4, 0.6, "#d97706", "0.4 to 0.6"),
    (0.6, 0.75, "#fef08a", "0.6 to 0.75"),
    (0.75, 0.9, "#22c55e", "0.75 to 0.9"),
    (0.9, 1.01, "#15803d", "0.9 to 1.0"),
]

RSRQ_BINS = [
    (-30, -16, "#ef4444", "-19 to -16 (bad)"),
    (-16, -13, "#facc15", "-16 to -13"),
    (-13, -9, "#4ade80", "-13 to -9"),
    (-9, -1, "#15803d", "-9 to -1 (good)"),
]

SINR_BINS = [
    (-20, 7, "#ef4444", "-13 to 7 (bad)"),
    (7, 10, "#facc15", "7 to 10"),
    (10, 14, "#fb923c", "10 to 14"),
    (14, 22, "#4ade80", "14 to 22"),
    (22, 40, "#15803d", "22 to 37 (good)"),
]

_GRID_ID_RE = re.compile(r"R(\d+)C(\d+)")


def _read_frame(path: Path) -> pd.DataFrame:
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"), low_memory=False)
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=5)
def load_grid_bounds() -> pd.DataFrame:
    return _read_frame(PHASE9_DIR / "phase9_gridanalytics_compatible_grid_project210")


def _attach_grid_bounds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or {"min_lat", "max_lat", "min_lon", "max_lon"}.issubset(df.columns):
        return df
    bounds = load_grid_bounds()
    needed = ["grid_id", "min_lat", "max_lat", "min_lon", "max_lon"]
    if bounds.empty or not set(needed).issubset(bounds.columns):
        return df
    return df.merge(bounds[needed], on="grid_id", how="left")


@st.cache_data(show_spinner=False)
def load_phase20_dt() -> pd.DataFrame:
    return _read_frame(PHASE20_DIR / "phase9_dt_match_project210_corrected")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase21_serving(tech: str) -> pd.DataFrame:
    serving = _read_frame(PHASE21_DIR / f"phase21_serving_grid_{tech.lower()}_project210")
    if serving.empty:
        return serving
    if "phase21_rsrp" not in serving.columns and "phase17_rsrp" in serving.columns:
        serving["phase21_rsrp"] = serving["phase17_rsrp"]
    if "phase21_frontend_mean_rsrp" not in serving.columns and "phase17_frontend_mean_rsrp" in serving.columns:
        serving["phase21_frontend_mean_rsrp"] = serving["phase17_frontend_mean_rsrp"]
    return _attach_grid_bounds(serving)


@st.cache_data(show_spinner=False, ttl=5)
def load_phase17_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE17_DIR / f"phase17_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase19_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE19_DIR / f"phase19_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase22_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE22_DIR / f"phase22_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase22_dt() -> pd.DataFrame:
    return _read_frame(PHASE22_DIR / "phase22_dt_terrain_scored_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase22_summary() -> dict:
    path = PHASE22_DIR / "phase22_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase23_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE23_DIR / f"phase23_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase23_validation_dt() -> pd.DataFrame:
    return _read_frame(PHASE23_DIR / "phase23_validation_dt_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase23_summary() -> dict:
    path = PHASE23_DIR / "phase23_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase24_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE24_DIR / f"phase24_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase24_dt() -> pd.DataFrame:
    return _read_frame(PHASE24_DIR / "phase24_dt_scored_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase24_summary() -> dict:
    path = PHASE24_DIR / "phase24_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase25_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE25_DIR / f"phase25_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase25_validation_dt() -> pd.DataFrame:
    return _read_frame(PHASE25_DIR / "phase25_validation_dt_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase25_summary() -> dict:
    path = PHASE25_DIR / "phase25_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase26_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE26_DIR / f"phase26_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase26_dt() -> pd.DataFrame:
    return _read_frame(PHASE26_DIR / "phase26_dt_scored_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase26_summary() -> dict:
    path = PHASE26_DIR / "phase26_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase27_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE27_DIR / f"phase27_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase27_validation_dt() -> pd.DataFrame:
    return _read_frame(PHASE27_DIR / "phase27_validation_dt_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase27_summary() -> dict:
    path = PHASE27_DIR / "phase27_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase27_group_corrections() -> pd.DataFrame:
    path = PHASE27_DIR / "phase27_group_corrections.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, ttl=5)
def load_phase29_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE29_DIR / f"phase29_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_sector_candidates() -> pd.DataFrame:
    """Per-candidate predictions for BOTH generic (Phase 27) and real-antenna (Phase 29),
    joined on the same (sector, grid cell) rows, for single-sector footprint comparison."""
    cols = ["strict_cell_key", "grid_id", "lat", "lon", "site", "sector", "band", "technology",
            "site_lat", "site_lon", "azimuth_x", "azimuth_delta_deg", "distance_m"]
    p27 = _read_frame(PHASE27_DIR / "phase27_scored_candidates_project210")
    p29 = _read_frame(PHASE29_DIR / "phase29_scored_candidates_project210")
    if p27.empty or p29.empty:
        return pd.DataFrame()
    a = p27[[c for c in cols if c in p27.columns] + ["phase27_dynamic_rsrp"]].copy()
    b = p29[["strict_cell_key", "grid_id", "phase29_dynamic_rsrp", "phase29_antenna_gain_delta_db"]].copy()
    m = a.merge(b, on=["strict_cell_key", "grid_id"], how="inner")
    return _attach_grid_bounds(m)


@st.cache_data(show_spinner=False, ttl=5)
def load_phase29_validation_dt() -> pd.DataFrame:
    return _read_frame(PHASE29_DIR / "phase29_validation_dt_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase29_candidates() -> pd.DataFrame:
    return _read_frame(PHASE29_DIR / "phase29_scored_candidates_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase29_summary() -> dict:
    path = PHASE29_DIR / "phase29_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


PHASE28_DIR = PROJECT_DIR / "cost231_phase28_4g_rsrp_reference_fix"


@st.cache_data(show_spinner=False, ttl=5)
def load_phase28_serving(tech: str = "4G") -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE28_DIR / f"phase28_{tech.lower()}_serving_grid_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase28_dt_check(tech: str = "4G") -> pd.DataFrame:
    return _read_frame(PHASE28_DIR / f"phase28_{tech.lower()}_dt_reference_check_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase28_dt_scored(tech: str = "4G") -> pd.DataFrame:
    return _read_frame(PHASE28_DIR / f"phase28_{tech.lower()}_dt_scored_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase28_summary() -> dict:
    path = PHASE28_DIR / "phase28_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase31_serving(tech: str = "4G") -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE31_DIR / f"phase31_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase31_dt_scored(tech: str = "4G") -> pd.DataFrame:
    return _read_frame(PHASE31_DIR / f"phase31_dt_scored_{tech.lower()}_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase31_summary() -> dict:
    path = PHASE31_DIR / "phase31_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


PHASE32_DIR = PROJECT_DIR / "cost231_phase32_real_antenna_audit"
PHASE33_DIR = PROJECT_DIR / "cost231_phase33_5g_38_901"
PHASE34_DIR = PROJECT_DIR / "cost231_phase34_ericsson_0_to_8"
PHASE35_DIR = PROJECT_DIR / "cost231_phase35_kathrein_all_tilts"

# Phase 33/34/35 all share the same 5G experiment shape: one path-loss / antenna
# swap on the Phase 33 engine. tag = "phase33" | "phase34" | "phase35".
_P3X = {
    "phase33": (PHASE33_DIR, "3GPP 38.901 UMa @ 3300 MHz, Kathrein 2-9 deg + generic 0/1"),
    "phase34": (PHASE34_DIR, "3GPP 38.901 UMa @ 3300 MHz, Ericsson AIR6468B42 0-8 deg + generic 9"),
    "phase35": (PHASE35_DIR, "3GPP 38.901 UMa @ 3300 MHz, Kathrein all tilts (0/1 -> 2 deg file)"),
}


@st.cache_data(show_spinner=False, ttl=5)
def load_p3x_serving(tag: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(_P3X[tag][0] / f"{tag}_5g_serving_grid_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_p3x_dt(tag: str) -> pd.DataFrame:
    return _read_frame(_P3X[tag][0] / f"{tag}_5g_dt_scored_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_p3x_summary(tag: str) -> dict:
    path = _P3X[tag][0] / f"{tag}_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False, ttl=5)
def load_phase32_audit() -> dict:
    path = PHASE32_DIR / "phase32_real_antenna_audit.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False, ttl=5)
def load_phase37_summary() -> dict:
    path = PHASE37_DIR / "phase37_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False, ttl=5)
def load_phase37_serving() -> pd.DataFrame:
    frame = _attach_grid_bounds(_read_frame(PHASE37_DIR / "phase37_serving_quality_project210"))
    if not frame.empty and {"min_lat", "max_lat", "min_lon", "max_lon"}.issubset(frame.columns):
        if "center_lat" not in frame.columns:
            frame["center_lat"] = (pd.to_numeric(frame["min_lat"], errors="coerce") + pd.to_numeric(frame["max_lat"], errors="coerce")) / 2.0
        if "center_lon" not in frame.columns:
            frame["center_lon"] = (pd.to_numeric(frame["min_lon"], errors="coerce") + pd.to_numeric(frame["max_lon"], errors="coerce")) / 2.0
    return frame


@st.cache_data(show_spinner=False, ttl=5)
def load_phase37_dt_quality(tech: str) -> pd.DataFrame:
    return _read_frame(PHASE37_DIR / f"phase37_dt_quality_{tech.lower()}_project210")


PHASE36_DIR = PROJECT_DIR / "cost231_phase36_final"


@st.cache_data(show_spinner=False, ttl=5)
def load_phase36_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE36_DIR / f"phase36_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase36_validation(tech: str) -> pd.DataFrame:
    df = _read_frame(PHASE36_DIR / "phase36_validation_dt_project210")
    return df[df["technology"].astype(str) == tech].copy() if not df.empty else df


@st.cache_data(show_spinner=False, ttl=5)
def load_phase36_summary() -> dict:
    path = PHASE36_DIR / "phase36_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


PHASE36V2_DIR = PROJECT_DIR / "cost231_phase36_v2_distance_shape"


@st.cache_data(show_spinner=False, ttl=5)
def load_phase36v2_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE36V2_DIR / f"phase36v2_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase36v2_validation(tech: str) -> pd.DataFrame:
    df = _read_frame(PHASE36V2_DIR / "phase36v2_validation_dt_project210")
    return df[df["technology"].astype(str) == tech].copy() if not df.empty else df


@st.cache_data(show_spinner=False, ttl=5)
def load_phase36v2_summary() -> dict:
    path = PHASE36V2_DIR / "phase36v2_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


PHASE38_DIR = PROJECT_DIR / "cost231_phase38_earfcn_rematch"


@st.cache_data(show_spinner=False, ttl=5)
def load_phase38_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE38_DIR / f"phase38_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase38_validation(tech: str) -> pd.DataFrame:
    df = _read_frame(PHASE38_DIR / "phase38_validation_dt_project210")
    return df[df["technology"].astype(str) == tech].copy() if not df.empty else df


@st.cache_data(show_spinner=False, ttl=5)
def load_phase38_summary() -> dict:
    path = PHASE38_DIR / "phase38_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


PHASE39_DIR = PROJECT_DIR / "cost231_phase39_equal_power_diagnostic"
PHASE40_DIR = PROJECT_DIR / "cost231_phase40_fixed_power_quality"
PHASE41_DIR = PROJECT_DIR / "cost231_phase41_pap_coverage_footprint"
PHASE42_DIR = PROJECT_DIR / "cost231_phase42_1500m_pap_sector_coverage"


@st.cache_data(show_spinner=False, ttl=5)
def load_phase39_serving(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE39_DIR / f"phase39_serving_grid_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase39_validation(tech: str) -> pd.DataFrame:
    df = _read_frame(PHASE39_DIR / "phase39_validation_dt_project210")
    return df[df["technology"].astype(str) == tech].copy() if not df.empty else df


@st.cache_data(show_spinner=False, ttl=5)
def load_phase39_summary() -> dict:
    path = PHASE39_DIR / "phase39_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False, ttl=5)
def load_phase40_summary() -> dict:
    path = PHASE40_DIR / "phase40_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False, ttl=5)
def load_phase40_serving() -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE40_DIR / "phase40_serving_quality_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase40_dt_quality(tech: str) -> pd.DataFrame:
    return _read_frame(PHASE40_DIR / f"phase40_dt_quality_{tech.lower()}_project210")


@st.cache_data(show_spinner=False, ttl=5)
def load_phase41_summary() -> dict:
    path = PHASE41_DIR / "phase41_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False, ttl=5)
def load_phase41_serving() -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE41_DIR / "phase41_serving_grid_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase41_cell_coverage(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE41_DIR / f"phase41_cell_coverage_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase41_sector_summary() -> pd.DataFrame:
    path = PHASE41_DIR / "phase41_sector_summary_project210.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=5)
def load_phase42_summary() -> dict:
    path = PHASE42_DIR / "phase42_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False, ttl=5)
def load_phase42_serving() -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE42_DIR / "phase42_serving_grid_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase42_cell_coverage(tech: str) -> pd.DataFrame:
    return _attach_grid_bounds(_read_frame(PHASE42_DIR / f"phase42_cell_coverage_{tech.lower()}_project210"))


@st.cache_data(show_spinner=False, ttl=5)
def load_phase42_sector_summary() -> pd.DataFrame:
    path = PHASE42_DIR / "phase42_sector_summary_project210.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=5)
def load_phase27_indoor_matched(tech: str) -> pd.DataFrame:
    """Matched-pixel O2I: for each indoor grid cell, the indoor prediction vs the
    outdoor-equivalent prediction AT THE SAME CELL. The gap = the real penetration loss,
    with geography removed (unlike the whole-polygon indoor-vs-outdoor CDF)."""
    df = _read_frame(PHASE27_DIR / "phase27_scored_candidates_project210")
    if df.empty:
        return pd.DataFrame()
    ind = df[(df["technology"].astype(str) == tech) & (df["obstruction_branch"].astype(str) == "indoor")].copy()
    if ind.empty:
        return pd.DataFrame()
    ind["indoor_pred"] = pd.to_numeric(ind["phase27_dynamic_rsrp"], errors="coerce")
    terr = pd.to_numeric(ind["terrain_diffraction_loss_db"], errors="coerce").fillna(0.0).clip(lower=0.0)
    # outdoor-equivalent at the same point = raw COST231 minus only terrain (no building / no O2I).
    ind["outdoor_equiv"] = pd.to_numeric(ind["raw_cost231_rsrp_unclipped"], errors="coerce") - terr
    best = ind.sort_values("indoor_pred").groupby("grid_id").tail(1)
    return best[["grid_id", "indoor_pred", "outdoor_equiv"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_site_sectors() -> pd.DataFrame:
    if not IDENTITY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(IDENTITY_PATH)
    if "Technology" in df.columns:
        df["technology"] = df["Technology"].astype(str)
    elif "technology" in df.columns:
        df["technology"] = df["technology"].astype(str)
    else:
        df["technology"] = "UNKNOWN"
    for col in ["lat", "lon", "azimuth", "band", "frequency"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [
        "technology",
        "Node_Cell_ID",
        "site",
        "sector",
        "band",
        "frequency",
        "lat",
        "lon",
        "azimuth",
    ]
    keep = [col for col in keep if col in df.columns]
    out = df[keep].dropna(subset=["lat", "lon", "azimuth"]).copy()
    return out.drop_duplicates(subset=["technology", "Node_Cell_ID", "lat", "lon", "azimuth"])


def _cdf_trace(values: pd.Series, name: str, color: str) -> go.Scatter:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    if arr.size == 0:
        return go.Scatter(x=[], y=[], mode="lines", name=name)
    y = np.arange(1, arr.size + 1, dtype=float) / arr.size * 100.0
    return go.Scatter(
        x=arr,
        y=y,
        mode="lines",
        name=f"{name} (n={arr.size:,})",
        line=dict(color=color, width=2.5),
    )


def _color_for_rsrp(value: float) -> str:
    if not np.isfinite(value):
        return "#9ca3af"
    for lo, hi, color, _label in RSRP_BINS:
        if lo <= value < hi:
            return color
    return "#9ca3af"


def _color_for_loss(value: float) -> str:
    if not np.isfinite(value):
        return "#9ca3af"
    for lo, hi, color, _label in LOSS_BINS:
        if lo <= value < hi:
            return color
    return "#9ca3af"


def _color_for_delta(value: float) -> str:
    if not np.isfinite(value):
        return "#9ca3af"
    for lo, hi, color, _label in DELTA_BINS:
        if lo <= value < hi:
            return color
    return "#9ca3af"


def _color_for_confidence(value: float) -> str:
    if not np.isfinite(value):
        return "#9ca3af"
    for lo, hi, color, _label in CONFIDENCE_BINS:
        if lo <= value < hi:
            return color
    return "#9ca3af"


def _color_for_bins(value: float, bins) -> str:
    if not np.isfinite(value):
        return "#9ca3af"
    for lo, hi, color, _label in bins:
        if lo <= value < hi:
            return color
    return bins[0][2] if np.isfinite(value) and value < bins[0][0] else bins[-1][2]


def _value_color(value: float, value_kind: str) -> str:
    if value_kind == "confidence":
        return _color_for_confidence(value)
    if value_kind == "delta":
        return _color_for_delta(value)
    if value_kind == "rsrq":
        return _color_for_bins(value, RSRQ_BINS)
    if value_kind == "sinr":
        return _color_for_bins(value, SINR_BINS)
    return _color_for_loss(value) if value_kind == "loss" else _color_for_rsrp(value)


def _legend_bins(value_kind: str):
    if value_kind == "confidence":
        return CONFIDENCE_BINS
    if value_kind == "delta":
        return DELTA_BINS
    if value_kind == "rsrq":
        return RSRQ_BINS
    if value_kind == "sinr":
        return SINR_BINS
    return LOSS_BINS if value_kind == "loss" else RSRP_BINS


def _center(df: pd.DataFrame) -> list[float]:
    lat_col = "center_lat" if "center_lat" in df.columns else "lat"
    lon_col = "center_lon" if "center_lon" in df.columns else "lon"
    return [float(df[lat_col].mean()), float(df[lon_col].mean())]


def _site_overlay_techs(selected_tech: str | None, overlay_mode: str) -> list[str]:
    if overlay_mode == "Off":
        return []
    if overlay_mode == "Both 4G and 5G":
        return ["4G", "5G"]
    return [selected_tech] if selected_tech else []


def _sector_tip(row) -> str:
    pieces = [
        f"<b>{getattr(row, 'technology', '')}</b>",
        f"<b>Cell:</b> {getattr(row, 'Node_Cell_ID', '')}",
        f"<b>Site:</b> {getattr(row, 'site', '')}",
        f"<b>Sector:</b> {getattr(row, 'sector', '')}",
        f"<b>Azimuth:</b> {float(getattr(row, 'azimuth', 0.0)):.0f}",
    ]
    if hasattr(row, "band") and pd.notna(getattr(row, "band")):
        pieces.append(f"<b>Band:</b> {getattr(row, 'band')}")
    if hasattr(row, "frequency") and pd.notna(getattr(row, "frequency")):
        pieces.append(f"<b>Identity frequency:</b> {float(getattr(row, 'frequency')):.0f} MHz")
    return "<br>".join(pieces)


def _sector_endpoint(lat: float, lon: float, azimuth: float, length_m: float = 170.0) -> tuple[float, float]:
    bearing = np.radians(float(azimuth))
    lat2 = lat + np.cos(bearing) * length_m / 110_540.0
    lon2 = lon + np.sin(bearing) * length_m / (111_320.0 * max(np.cos(np.radians(lat)), 1e-6))
    return float(lat2), float(lon2)


def _add_site_sector_layers(fmap: folium.Map, selected_tech: str | None, overlay_mode: str) -> None:
    techs = _site_overlay_techs(selected_tech, overlay_mode)
    if not techs:
        return
    sites = load_site_sectors()
    if sites.empty:
        return
    for tech in techs:
        sub = sites[sites["technology"].astype(str) == tech].copy()
        if sub.empty:
            continue
        color = SITE_TECH_COLORS.get(tech, "#111827")
        layer = folium.FeatureGroup(name=f"{tech} site sectors", show=True)
        for row in sub.itertuples(index=False):
            lat = float(row.lat)
            lon = float(row.lon)
            azimuth = float(row.azimuth)
            end_lat, end_lon = _sector_endpoint(lat, lon, azimuth)
            folium.RegularPolygonMarker(
                location=[lat, lon],
                number_of_sides=3,
                rotation=int(azimuth),
                radius=9,
                color="#ffffff",
                weight=1.2,
                fill=True,
                fill_color=color,
                fill_opacity=0.95,
                tooltip=f"{tech} {getattr(row, 'site', '')} {getattr(row, 'sector', '')}",
                popup=folium.Popup(_sector_tip(row), max_width=320),
            ).add_to(layer)
            folium.PolyLine([[lat, lon], [end_lat, end_lon]], color=color, weight=2, opacity=0.8).add_to(layer)
        layer.add_to(fmap)


def _static_site_handles(
    ax,
    valid_df: pd.DataFrame,
    grid_rows: np.ndarray,
    grid_cols: np.ndarray,
    selected_tech: str | None,
    overlay_mode: str,
) -> tuple[list, list]:
    techs = _site_overlay_techs(selected_tech, overlay_mode)
    if not techs or valid_df.empty or "center_lat" not in valid_df.columns or "center_lon" not in valid_df.columns:
        return [], []

    sites = load_site_sectors()
    if sites.empty:
        return [], []

    row_lat = pd.DataFrame({"row": grid_rows, "lat": pd.to_numeric(valid_df["center_lat"], errors="coerce")})
    col_lon = pd.DataFrame({"col": grid_cols, "lon": pd.to_numeric(valid_df["center_lon"], errors="coerce")})
    row_ref = row_lat.groupby("row")["lat"].median().dropna()
    col_ref = col_lon.groupby("col")["lon"].median().dropna()
    if len(row_ref) < 2 or len(col_ref) < 2:
        return [], []

    row_fit = np.polyfit(row_ref.to_numpy(dtype=float), row_ref.index.to_numpy(dtype=float), 1)
    col_fit = np.polyfit(col_ref.to_numpy(dtype=float), col_ref.index.to_numpy(dtype=float), 1)
    row_min, row_max = float(np.nanmin(grid_rows)), float(np.nanmax(grid_rows))
    col_min, col_max = float(np.nanmin(grid_cols)), float(np.nanmax(grid_cols))

    handles, labels = [], []
    for tech in techs:
        sub = sites[sites["technology"].astype(str) == tech].copy()
        if sub.empty:
            continue
        color = SITE_TECH_COLORS.get(tech, "#111827")
        lat = pd.to_numeric(sub["lat"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(sub["lon"], errors="coerce").to_numpy(dtype=float)
        row_pos = row_fit[0] * lat + row_fit[1]
        col_pos = col_fit[0] * lon + col_fit[1]
        inside = (
            np.isfinite(row_pos)
            & np.isfinite(col_pos)
            & (row_pos >= row_min - 2)
            & (row_pos <= row_max + 2)
            & (col_pos >= col_min - 2)
            & (col_pos <= col_max + 2)
        )
        sub = sub.loc[inside].copy()
        row_pos = row_pos[inside]
        col_pos = col_pos[inside]
        if sub.empty:
            continue
        for (_, row), x, y in zip(sub.iterrows(), col_pos, row_pos):
            marker = MarkerStyle("^").transformed(Affine2D().rotate_deg(-float(row["azimuth"])))
            ax.scatter(
                [x],
                [y],
                marker=marker,
                s=42,
                facecolors=color,
                edgecolors="#ffffff",
                linewidths=0.55,
                zorder=5,
            )
        handles.append(plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=color, markeredgecolor="#ffffff", markersize=7, linewidth=0))
        labels.append(f"{tech} sectors")
    return handles, labels


def _popup(row, value_col: str, label: str) -> str:
    pieces = [
        f"<b>Grid:</b> {getattr(row, 'grid_id', '')}",
        f"<b>{label}:</b> {float(getattr(row, value_col)):.1f}",
    ]
    for col, text in [
        ("strict_cell_key", "Cell"),
        ("site", "Site"),
        ("sector", "Sector"),
        ("band", "Band"),
        ("technology", "Technology"),
        ("terrain_diffraction_loss_db_mean", "Terrain loss mean"),
        ("terrain_obstructed_share", "Terrain obstructed share"),
    ]:
        if hasattr(row, col):
            val = getattr(row, col)
            if pd.notna(val):
                pieces.append(f"<b>{text}:</b> {val}")
    return "<br>".join(pieces)


@st.cache_data(show_spinner=False)
def _build_grid_map_html(
    serving: pd.DataFrame,
    value_col: str,
    label: str,
    value_kind: str,
    selected_tech: str | None,
    site_overlay: str,
) -> str:
    required = ["min_lat", "max_lat", "min_lon", "max_lon", value_col]
    missing = [col for col in required if col not in serving.columns]
    if missing:
        return f"<p>Map cannot render because columns are missing: {', '.join(missing)}</p>"
    df = serving.dropna(subset=required).copy()
    if df.empty:
        return "<p>Map cannot render because no grid cells have bounds and valid values.</p>"
    fmap = folium.Map(
        location=_center(df),
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )
    layer = folium.FeatureGroup(name=label, show=True)
    for row in df.itertuples(index=False):
        val = float(getattr(row, value_col))
        color = _value_color(val, value_kind)
        folium.Rectangle(
            bounds=[[row.min_lat, row.min_lon], [row.max_lat, row.max_lon]],
            color=color,
            weight=0,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"{val:.1f}",
            popup=folium.Popup(_popup(row, value_col, label), max_width=340),
        ).add_to(layer)
    layer.add_to(fmap)
    _add_site_sector_layers(fmap, selected_tech, site_overlay)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap._repr_html_()


@st.cache_data(show_spinner=False)
def _build_static_image_png(
    serving: pd.DataFrame,
    value_col: str,
    title: str,
    value_kind: str,
    selected_tech: str | None,
    site_overlay: str,
) -> bytes:
    df = serving.dropna(subset=[value_col]).copy()
    extracted = df["grid_id"].astype(str).str.extract(_GRID_ID_RE).astype(float)
    valid = extracted[0].notna() & extracted[1].notna()
    rows = extracted.loc[valid, 0].astype(int).to_numpy()
    cols = extracted.loc[valid, 1].astype(int).to_numpy()
    values = pd.to_numeric(df.loc[valid, value_col], errors="coerce").to_numpy(dtype=float)
    grid = np.full((int(rows.max()) + 1, int(cols.max()) + 1), np.nan, dtype=float)
    grid[rows, cols] = values

    bins = _legend_bins(value_kind)
    boundaries = [item[0] for item in bins] + [bins[-1][1]]
    cmap = ListedColormap([item[2] for item in bins])
    cmap.set_bad(color="#9ca3af")
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(3.5, 4.4))
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, origin="lower", aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=item[2]) for item in bins]
    labels = [item[3] for item in bins]
    if value_kind == "rsrp":
        handles.append(plt.Rectangle((0, 0), 1, 1, color="#9ca3af"))
        labels.append("No coverage")

    site_handles = _static_site_handles(ax, df.loc[valid].copy(), rows, cols, selected_tech, site_overlay)
    if site_handles:
        handles.extend(site_handles[0])
        labels.extend(site_handles[1])
    ax.legend(handles, labels, loc="lower left", fontsize=7, framealpha=0.9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_map(serving: pd.DataFrame, value_col: str, label: str, view_mode: str, value_kind: str = "rsrp") -> None:
    if value_col not in serving.columns:
        st.warning(f"Column missing: {value_col}")
        return
    if serving.empty:
        st.warning("No rows available.")
        return
    if view_mode == "Static image":
        st.image(
            _build_static_image_png(
                serving,
                value_col,
                label,
                value_kind,
                st.session_state.get("phase20_21_22_tech"),
                st.session_state.get("phase_site_overlay", "Selected technology"),
            ),
            width=STATIC_MAP_DISPLAY_WIDTH_PX,
        )
    else:
        components.html(
            _build_grid_map_html(
                serving,
                value_col,
                label,
                value_kind,
                st.session_state.get("phase20_21_22_tech"),
                st.session_state.get("phase_site_overlay", "Selected technology"),
            ),
            height=360,
            scrolling=False,
        )


def _metric_row(label: str, values: dict[str, float]) -> None:
    st.markdown(f"**{label}**")
    cols = st.columns(len(values))
    for col, (name, value) in zip(cols, values.items()):
        if isinstance(value, str):
            col.metric(name, value)
        else:
            col.metric(name, f"{value:.1f}")


def _render_phase20(dt: pd.DataFrame) -> None:
    st.header("Phase 20: Corrected DT Source")
    if dt.empty:
        st.error(f"Phase 20 corrected DT file not found under {PHASE20_DIR}.")
        return
    counts = dt["assigned_technology"].value_counts()
    eligible = dt.loc[dt["dt_replacement_eligible"].fillna(False).astype(bool), "assigned_technology"].value_counts()
    cols = st.columns(5)
    cols[0].metric("Total DT rows", f"{len(dt):,}")
    cols[1].metric("4G DT rows", f"{int(counts.get('4G', 0)):,}")
    cols[2].metric("5G DT rows", f"{int(counts.get('5G', 0)):,}")
    cols[3].metric("4G replacement rows", f"{int(eligible.get('4G', 0)):,}")
    cols[4].metric("5G replacement rows", f"{int(eligible.get('5G', 0)):,}")
    st.dataframe(
        dt[[
            "assigned_technology",
            "network",
            "assigned_strict_cell_key",
            "rsrp_measured",
            "raw_cost231_at_dt_rsrp",
            "dt_minus_cost231_db",
            "nearest_grid_id",
            "nearest_grid_distance_m",
            "dt_replacement_eligible",
        ]].head(500),
        use_container_width=True,
        height=260,
    )


def _render_phase21(view_mode: str, tech: str) -> None:
    st.header("Phase 21: Full Polygon With Corrected 5G DT")
    serving = load_phase21_serving(tech)
    phase17 = load_phase17_serving(tech)
    phase19 = load_phase19_serving(tech)
    if serving.empty:
        st.error(f"Phase 21 output not found under {PHASE21_DIR}.")
        return

    if tech == "5G":
        old_mean = float(phase17["phase17_rsrp"].mean()) if not phase17.empty else np.nan
        new_mean = float(serving["phase21_rsrp"].mean())
        _metric_row(
            "5G corrected DT impact",
            {
                "Grid cells": f"{len(serving):,}",
                "Phase 9 mean": float(serving["corrected_rsrp"].mean()),
                "Old Phase 17 mean": old_mean,
                "Phase 21 mean": new_mean,
                "Change vs old": new_mean - old_mean,
            },
        )
    else:
        _metric_row(
            "4G unchanged in Phase 21",
            {
                "Grid cells": f"{len(serving):,}",
                "Phase 9 mean": float(serving["corrected_rsrp"].mean()),
                "Phase 21 mean": float(serving["phase21_rsrp"].mean()),
                "Mean geo correction": float(serving["geo_correction_db"].mean()),
            },
        )

    map_cols = st.columns(3)
    with map_cols[0]:
        st.caption(f"{tech} Phase 9")
        _render_map(serving, "corrected_rsrp", "Phase 9", view_mode)
    with map_cols[1]:
        st.caption(f"{tech} Phase 21")
        _render_map(serving, "phase21_rsrp", "Phase 21", view_mode)
    with map_cols[2]:
        st.caption(f"{tech} Phase 21 frontend mean")
        _render_map(serving, "phase21_frontend_mean_rsrp", "Phase 21 frontend mean", view_mode)

    fig = go.Figure()
    fig.add_trace(_cdf_trace(serving["corrected_rsrp"], "Phase 9 full grid", "#ef4444"))
    if not phase17.empty and "phase17_rsrp" in phase17.columns:
        fig.add_trace(_cdf_trace(phase17["phase17_rsrp"], "Phase 17 full grid", "#f97316"))
    if not phase19.empty and "phase19_rsrp" in phase19.columns:
        fig.add_trace(_cdf_trace(phase19["phase19_rsrp"], "Phase 19 full grid", "#7c3aed"))
    fig.add_trace(_cdf_trace(serving["phase21_rsrp"], "Phase 21 full grid", "#16a34a"))
    fig.update_layout(
        title=f"{tech} full-polygon CDF: Phase 9/17/19/21",
        height=430,
        xaxis_title="RSRP (dBm)",
        yaxis_title="Cumulative %",
        yaxis_range=[0, 100],
        xaxis_range=[-147, -45],
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_phase22(view_mode: str, tech: str, aggregation: str) -> None:
    st.header("Phase 22: DEM Terrain Diffraction Before/After")
    serving = load_phase22_serving(tech)
    dt = load_phase22_dt()
    summary = load_phase22_summary().get(tech, {})
    if serving.empty:
        st.error(f"Phase 22 output not found under {PHASE22_DIR}.")
        return

    value_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    physical_before_col = f"phase22_physical_no_terrain_{value_suffix}_rsrp"
    physical_after_col = f"phase22_physical_with_terrain_{value_suffix}_rsrp"
    same_bias_before_col = f"phase22_no_terrain_{value_suffix}_rsrp"
    same_bias_after_col = f"phase22_with_terrain_{value_suffix}_rsrp"

    _metric_row(
        f"{tech} terrain impact",
        {
            "Grid cells": f"{len(serving):,}",
            "Aggregation": aggregation.split(" (", 1)[0],
            "Physical before": float(serving[physical_before_col].mean()),
            "Physical after": float(serving[physical_after_col].mean()),
            "Terrain shift": float(
                (
                    serving[physical_after_col]
                    - serving[physical_before_col]
                ).mean()
            ),
            "Same bias + terrain": float(serving[same_bias_after_col].mean()),
            "Obstructed share": f"{float(serving['terrain_obstructed_share'].mean()) * 100:.1f}%",
        },
    )
    if summary:
        loss = summary.get("terrain_loss_db", {})
        st.caption(
            f"Terrain loss mean={loss.get('mean', 0):.1f} dB, "
            f"p50={loss.get('p50', 0):.1f} dB, p90={loss.get('p90', 0):.1f} dB, "
            f"max={loss.get('max', 0):.1f} dB."
        )

    layer_options = {
        "Physical before terrain": (physical_before_col, "rsrp"),
        "Physical after terrain": (physical_after_col, "rsrp"),
        "Same bias before terrain": (same_bias_before_col, "rsrp"),
        "Same bias + terrain": (same_bias_after_col, "rsrp"),
        "Terrain loss mean": ("terrain_diffraction_loss_db_mean", "loss"),
    }
    selected = st.radio("Phase 22 map layer", list(layer_options.keys()), index=3, horizontal=True)
    value_col, value_kind = layer_options[selected]
    _render_map(serving, value_col, f"{tech} {selected}", view_mode, value_kind=value_kind)

    fig = go.Figure()
    fig.add_trace(_cdf_trace(serving[physical_before_col], "Physical before terrain", "#ef4444"))
    fig.add_trace(_cdf_trace(serving[physical_after_col], "Physical after terrain", "#2563eb"))
    fig.add_trace(_cdf_trace(serving[same_bias_before_col], "Same bias before terrain", "#f97316"))
    fig.add_trace(_cdf_trace(serving[same_bias_after_col], "Same bias + terrain", "#16a34a"))
    fig.update_layout(
        title=f"{tech} full-polygon CDF: terrain before/after - {aggregation}",
        height=430,
        xaxis_title="RSRP (dBm)",
        yaxis_title="Cumulative %",
        yaxis_range=[0, 100],
        xaxis_range=[-147, -45],
    )
    st.plotly_chart(fig, use_container_width=True)

    dt_tech = dt[dt["assigned_technology"].astype(str) == tech].copy() if not dt.empty else pd.DataFrame()
    if not dt_tech.empty:
        dt_fig = go.Figure()
        dt_fig.add_trace(_cdf_trace(dt_tech["rsrp_measured"], "DT measured", "#111827"))
        dt_fig.add_trace(_cdf_trace(dt_tech["phase22_physical_no_terrain_rsrp"], "Before terrain at DT", "#ef4444"))
        dt_fig.add_trace(_cdf_trace(dt_tech["phase22_physical_with_terrain_rsrp"], "After terrain at DT", "#2563eb"))
        if "phase22_no_terrain_calibrated_rsrp" in dt_tech.columns:
            dt_fig.add_trace(_cdf_trace(dt_tech["phase22_no_terrain_calibrated_rsrp"], "Same bias before terrain at DT", "#f97316"))
        if "phase22_with_terrain_calibrated_rsrp" in dt_tech.columns:
            dt_fig.add_trace(_cdf_trace(dt_tech["phase22_with_terrain_calibrated_rsrp"], "Same bias + terrain at DT", "#16a34a"))
        dt_fig.update_layout(
            title=f"{tech} DT-location CDF: measured vs terrain physics",
            height=430,
            xaxis_title="RSRP (dBm)",
            yaxis_title="Cumulative %",
            yaxis_range=[0, 100],
            xaxis_range=[-147, -45],
        )
        st.plotly_chart(dt_fig, use_container_width=True)

    image_paths = [
        PHASE22_IMAGE_DIR / f"phase22_{tech.lower()}_full_polygon_cdf.png",
        PHASE22_IMAGE_DIR / f"phase22_{tech.lower()}_dt_cdf.png",
        PHASE22_IMAGE_DIR / f"phase22_{tech.lower()}_terrain_loss_cdf.png",
    ]
    existing = [path for path in image_paths if path.exists()]
    if existing:
        st.subheader("Generated CDF PNGs")
        cols = st.columns(len(existing))
        for col, path in zip(cols, existing):
            col.image(str(path), use_container_width=True)


def _render_phase23(view_mode: str, tech: str, aggregation: str) -> None:
    st.header("Phase 23: Outdoor Project-Level Calibration")
    serving = load_phase23_serving(tech)
    validation = load_phase23_validation_dt()
    summary = load_phase23_summary()
    if serving.empty:
        st.error(f"Phase 23 output not found under {PHASE23_DIR}.")
        return

    tech_summary = summary.get("technology", {}).get(tech, {})
    selected_config = summary.get("selected_config_by_technology", {}).get(
        tech, summary.get("selected_config", {})
    )
    st.caption(
        "Outdoor-only calibration from held-out DT. Indoor/O2I is unchanged from Phase 22 "
        "and is not calibrated from outdoor DT."
    )
    cols = st.columns(5)
    cols[0].metric("Selected config", str(selected_config.get("name", "")))
    cols[1].metric("Clear cap", f"{float(selected_config.get('clear_cap_db', 0.0)):.0f} dB")
    cols[2].metric("Obstructed cap", f"{float(selected_config.get('obstructed_cap_db', 0.0)):.0f} dB")
    cols[3].metric("Phase22 MAE", f"{float(tech_summary.get('phase22_mae_db', np.nan)):.2f} dB")
    cols[4].metric("Phase23 MAE", f"{float(tech_summary.get('phase23_mae_db', np.nan)):.2f} dB")

    value_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    phase22_col = f"phase22_with_terrain_{value_suffix}_rsrp"
    phase23_col = f"phase23_{value_suffix}_rsrp"
    serving["phase23_vs_phase22_delta_db"] = serving[phase23_col] - serving[phase22_col]

    st.subheader(f"{tech} Phase 22 vs Phase 23 Map Comparison")
    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption("Phase 22 same bias + terrain")
        _render_map(serving, phase22_col, f"{tech} Phase 22", view_mode)
    with map_cols[1]:
        st.caption("Phase 23 outdoor calibrated")
        _render_map(serving, phase23_col, f"{tech} Phase 23", view_mode)

    diagnostic_options = {
        "Phase23 - Phase22 delta": ("phase23_vs_phase22_delta_db", "delta"),
        "Outdoor correction mean": ("phase23_outdoor_correction_db_mean", "delta"),
    }
    selected_diag = st.radio("Phase 23 diagnostic map", list(diagnostic_options.keys()), index=0, horizontal=True)
    diag_col, diag_kind = diagnostic_options[selected_diag]
    _render_map(serving, diag_col, f"{tech} {selected_diag}", view_mode, value_kind=diag_kind)

    fig = go.Figure()
    fig.add_trace(_cdf_trace(serving[phase22_col], "Phase22 same bias + terrain", "#2563eb"))
    fig.add_trace(_cdf_trace(serving[phase23_col], "Phase23 outdoor calibrated", "#16a34a"))
    fig.update_layout(
        title=f"{tech} full-polygon CDF: Phase 22 vs Phase 23 - {aggregation}",
        height=430,
        xaxis_title="RSRP (dBm)",
        yaxis_title="Cumulative %",
        yaxis_range=[0, 100],
        xaxis_range=[-147, -45],
    )
    st.plotly_chart(fig, use_container_width=True)

    vtech = validation[validation["technology"].astype(str) == tech].copy() if not validation.empty else pd.DataFrame()
    if not vtech.empty:
        dt_fig = go.Figure()
        dt_fig.add_trace(_cdf_trace(vtech["rsrp_measured"], "DT measured", "#111827"))
        dt_fig.add_trace(_cdf_trace(vtech["phase22_same_bias_with_terrain_rsrp"], "Phase22 validation", "#2563eb"))
        dt_fig.add_trace(_cdf_trace(vtech["phase23_rsrp"], "Phase23 validation", "#16a34a"))
        dt_fig.update_layout(
            title=f"{tech} outdoor validation CDF",
            height=430,
            xaxis_title="RSRP (dBm)",
            yaxis_title="Cumulative %",
            yaxis_range=[0, 100],
            xaxis_range=[-147, -45],
        )
        st.plotly_chart(dt_fig, use_container_width=True)

        err_fig = go.Figure()
        err_fig.add_trace(_cdf_trace(vtech["phase22_error_db"].abs(), "Phase22 abs error", "#2563eb"))
        err_fig.add_trace(_cdf_trace(vtech["phase23_error_db"].abs(), "Phase23 abs error", "#16a34a"))
        err_fig.update_layout(
            title=f"{tech} outdoor validation absolute error CDF",
            height=430,
            xaxis_title="Absolute error (dB)",
            yaxis_title="Cumulative %",
            yaxis_range=[0, 100],
        )
        st.plotly_chart(err_fig, use_container_width=True)

    trial_table = pd.DataFrame(summary.get("trial_table", []))
    if not trial_table.empty:
        if "technology" in trial_table.columns:
            trial_table = trial_table[trial_table["technology"].astype(str).isin([tech, "ALL"])].copy()
        st.subheader("Calibration Trials")
        st.dataframe(
            trial_table[
                [
                    "technology",
                    "config_name",
                    "alpha",
                    "clear_cap_db",
                    "obstructed_cap_db",
                    "phase22_mae",
                    "phase23_mae",
                    "phase22_bias",
                    "phase23_bias",
                    "score",
                ]
            ],
            use_container_width=True,
            height=220,
        )

    image_paths = [
        PHASE23_IMAGE_DIR / f"phase23_{tech.lower()}_full_grid_cdf.png",
        PHASE23_IMAGE_DIR / f"phase23_{tech.lower()}_validation_dt_cdf.png",
        PHASE23_IMAGE_DIR / f"phase23_{tech.lower()}_validation_abs_error_cdf.png",
    ]
    existing = [path for path in image_paths if path.exists()]
    if existing:
        st.subheader("Phase 23 Generated PNGs")
        cols = st.columns(len(existing))
        for col, path in zip(cols, existing):
            col.image(str(path), use_container_width=True)


def _render_phase24(view_mode: str, tech: str, aggregation: str) -> None:
    st.header("Phase 24: Physical Clutter Role Fix")
    serving = load_phase24_serving(tech)
    dt = load_phase24_dt()
    summary = load_phase24_summary().get(tech, {})
    if serving.empty:
        st.error(f"Phase 24 output not found under {PHASE24_DIR}.")
        return

    value_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    phase22_col = f"phase22_with_terrain_{value_suffix}_rsrp"
    phase24_physical_col = f"phase24_physical_with_terrain_{value_suffix}_rsrp"
    phase24_col = f"phase24_with_terrain_{value_suffix}_rsrp"
    serving["phase24_vs_phase22_delta_db"] = serving[phase24_col] - serving[phase22_col]

    _metric_row(
        f"{tech} Phase 22 vs Phase 24",
        {
            "Grid cells": f"{len(serving):,}",
            "Aggregation": aggregation.split(" (", 1)[0],
            "Phase22 final": float(serving[phase22_col].mean()),
            "Phase24 physical": float(serving[phase24_physical_col].mean()),
            "Phase24 final": float(serving[phase24_col].mean()),
            "Delta": float(serving["phase24_vs_phase22_delta_db"].mean()),
        },
    )
    if summary:
        pdelta = summary.get("phase24_minus_phase22_best_db", {})
        st.caption(
            "Phase 24 only changes the clear-path clutter role; terrain and indoor/O2I stay unchanged. "
            f"Best-server delta mean={pdelta.get('mean', 0):.2f} dB, "
            f"p90={pdelta.get('p90', 0):.2f} dB."
        )

    st.subheader(f"{tech} side-by-side map comparison")
    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption("Phase 22 same bias + terrain")
        _render_map(serving, phase22_col, f"{tech} Phase 22", view_mode)
    with map_cols[1]:
        st.caption("Phase 24 corrected clutter role")
        _render_map(serving, phase24_col, f"{tech} Phase 24", view_mode)

    diagnostic_options = {
        "Phase24 - Phase22 delta": ("phase24_vs_phase22_delta_db", "delta"),
        "Phase24 correction delta mean": ("phase24_correction_delta_db_mean", "delta"),
        "Proxy clutter suppressed share": ("phase24_proxy_clutter_suppressed_share", "loss"),
        "Terrain loss mean": ("terrain_diffraction_loss_db_mean", "loss"),
    }
    selected_diag = st.radio("Phase 24 diagnostic map", list(diagnostic_options.keys()), index=0, horizontal=True)
    diag_col, diag_kind = diagnostic_options[selected_diag]
    _render_map(serving, diag_col, f"{tech} {selected_diag}", view_mode, value_kind=diag_kind)

    fig = go.Figure()
    fig.add_trace(_cdf_trace(serving[phase22_col], "Phase22 same bias + terrain", "#2563eb"))
    fig.add_trace(_cdf_trace(serving[phase24_physical_col], "Phase24 physical + terrain", "#f97316"))
    fig.add_trace(_cdf_trace(serving[phase24_col], "Phase24 same residual + terrain", "#16a34a"))
    fig.update_layout(
        title=f"{tech} full-polygon CDF: Phase 22 vs Phase 24 - {aggregation}",
        height=430,
        xaxis_title="RSRP (dBm)",
        yaxis_title="Cumulative %",
        yaxis_range=[0, 100],
        xaxis_range=[-147, -45],
    )
    st.plotly_chart(fig, use_container_width=True)

    dt_tech = dt[dt["assigned_technology"].astype(str) == tech].copy() if not dt.empty else pd.DataFrame()
    if not dt_tech.empty:
        dt_fig = go.Figure()
        dt_fig.add_trace(_cdf_trace(dt_tech["rsrp_measured"], "DT measured", "#111827"))
        dt_fig.add_trace(_cdf_trace(dt_tech["phase22_physical_with_terrain_rsrp"], "Phase22 physical + terrain", "#2563eb"))
        dt_fig.add_trace(_cdf_trace(dt_tech["phase24_physical_with_terrain_rsrp"], "Phase24 physical + terrain", "#f97316"))
        dt_fig.add_trace(_cdf_trace(dt_tech["phase24_with_terrain_calibrated_rsrp"], "Phase24 same residual + terrain", "#16a34a"))
        dt_fig.update_layout(
            title=f"{tech} DT-location CDF: Phase 22 vs Phase 24",
            height=430,
            xaxis_title="RSRP (dBm)",
            yaxis_title="Cumulative %",
            yaxis_range=[0, 100],
            xaxis_range=[-147, -45],
        )
        st.plotly_chart(dt_fig, use_container_width=True)

        err_fig = go.Figure()
        err_fig.add_trace(_cdf_trace((dt_tech["rsrp_measured"] - dt_tech["phase22_physical_with_terrain_rsrp"]).abs(), "Phase22 physical abs error", "#2563eb"))
        err_fig.add_trace(_cdf_trace((dt_tech["rsrp_measured"] - dt_tech["phase24_physical_with_terrain_rsrp"]).abs(), "Phase24 physical abs error", "#f97316"))
        err_fig.add_trace(_cdf_trace((dt_tech["rsrp_measured"] - dt_tech["phase24_with_terrain_calibrated_rsrp"]).abs(), "Phase24 calibrated abs error", "#16a34a"))
        err_fig.update_layout(
            title=f"{tech} DT absolute error CDF",
            height=430,
            xaxis_title="Absolute error (dB)",
            yaxis_title="Cumulative %",
            yaxis_range=[0, 100],
        )
        st.plotly_chart(err_fig, use_container_width=True)

    image_paths = [
        PHASE24_IMAGE_DIR / f"phase24_{tech.lower()}_full_polygon_cdf.png",
        PHASE24_IMAGE_DIR / f"phase24_{tech.lower()}_dt_cdf.png",
        PHASE24_IMAGE_DIR / f"phase24_{tech.lower()}_dt_abs_error_cdf.png",
    ]
    existing = [path for path in image_paths if path.exists()]
    if existing:
        st.subheader("Phase 24 Generated PNGs")
        cols = st.columns(len(existing))
        for col, path in zip(cols, existing):
            col.image(str(path), use_container_width=True)


def _render_phase25(view_mode: str, tech: str, aggregation: str) -> None:
    st.header("Phase 25: Hierarchical Dynamic Calibration")
    serving = load_phase25_serving(tech)
    validation = load_phase25_validation_dt()
    summary = load_phase25_summary().get("technology", {}).get(tech, {})
    if serving.empty:
        st.error(f"Phase 25 output not found under {PHASE25_DIR}.")
        return

    value_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    phase24_col = f"phase24_no_lock_{value_suffix}_rsrp"
    phase25_col = f"phase25_dynamic_{value_suffix}_rsrp"
    serving["phase25_vs_phase24_delta_db"] = serving[phase25_col] - serving[phase24_col]

    phase24_metrics = summary.get("phase24_no_lock_validation", {})
    phase25_metrics = summary.get("phase25_dynamic_validation", {})
    cols = st.columns(6)
    cols[0].metric("Validation rows", f"{int(summary.get('validation_dt_rows', 0)):,}")
    cols[1].metric("Phase24 MAE", f"{float(phase24_metrics.get('mae', np.nan)):.2f} dB")
    cols[2].metric("Phase25 MAE", f"{float(phase25_metrics.get('mae', np.nan)):.2f} dB")
    cols[3].metric("Phase25 bias", f"{float(phase25_metrics.get('bias', np.nan)):.2f} dB")
    cols[4].metric("P90 abs err", f"{float(phase25_metrics.get('p90_abs', np.nan)):.2f} dB")
    cols[5].metric("Confidence", f"{float(summary.get('mean_confidence', np.nan)):.2f}")
    st.caption(
        "Validation is on held-out DT grids and does not use DT replacement. "
        "Corrections are estimated hierarchically from training DT: tech/band, "
        "clutter/terrain/branch, sector if enough samples, then local residual where supported."
    )

    st.subheader(f"{tech} side-by-side map comparison")
    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption("Phase 24 no-lock reference")
        _render_map(serving, phase24_col, f"{tech} Phase 24 no-lock", view_mode)
    with map_cols[1]:
        st.caption("Phase 25 dynamic calibrated")
        _render_map(serving, phase25_col, f"{tech} Phase 25 dynamic", view_mode)

    diagnostic_options = {
        "Phase25 - Phase24 delta": ("phase25_vs_phase24_delta_db", "delta"),
        "Total dynamic correction mean": ("phase25_total_dynamic_correction_db_mean", "delta"),
        "Tech/band correction mean": ("tech_band_correction_db_mean", "delta"),
        "Clutter/terrain correction mean": ("clutter_terrain_correction_db_mean", "delta"),
        "Sector correction mean": ("sector_correction_db_mean", "delta"),
        "Local residual correction mean": ("local_residual_correction_db_mean", "delta"),
        "Local support mean": ("local_residual_support_n_mean", "loss"),
        "Confidence mean": ("phase25_confidence_mean", "confidence"),
    }
    selected_diag = st.radio("Phase 25 diagnostic map", list(diagnostic_options.keys()), index=0, horizontal=True)
    diag_col, diag_kind = diagnostic_options[selected_diag]
    _render_map(serving, diag_col, f"{tech} {selected_diag}", view_mode, value_kind=diag_kind)

    fig = go.Figure()
    fig.add_trace(_cdf_trace(serving[phase24_col], "Phase24 no-lock reference", "#2563eb"))
    fig.add_trace(_cdf_trace(serving[phase25_col], "Phase25 dynamic calibrated", "#16a34a"))
    fig.update_layout(
        title=f"{tech} full-polygon CDF: Phase 24 vs Phase 25 - {aggregation}",
        height=430,
        xaxis_title="RSRP (dBm)",
        yaxis_title="Cumulative %",
        yaxis_range=[0, 100],
        xaxis_range=[-147, -45],
    )
    st.plotly_chart(fig, use_container_width=True)

    vtech = validation[validation["technology"].astype(str) == tech].copy() if not validation.empty else pd.DataFrame()
    if not vtech.empty:
        dt_fig = go.Figure()
        dt_fig.add_trace(_cdf_trace(vtech["rsrp_measured"], "DT measured", "#111827"))
        dt_fig.add_trace(_cdf_trace(vtech["phase24_no_lock_reference_rsrp"], "Phase24 no-lock reference", "#2563eb"))
        dt_fig.add_trace(_cdf_trace(vtech["phase25_dynamic_rsrp"], "Phase25 dynamic", "#16a34a"))
        dt_fig.update_layout(
            title=f"{tech} held-out DT CDF",
            height=430,
            xaxis_title="RSRP (dBm)",
            yaxis_title="Cumulative %",
            yaxis_range=[0, 100],
            xaxis_range=[-147, -45],
        )
        st.plotly_chart(dt_fig, use_container_width=True)

        err_fig = go.Figure()
        err_fig.add_trace(_cdf_trace((vtech["rsrp_measured"] - vtech["phase24_no_lock_reference_rsrp"]).abs(), "Phase24 no-lock abs error", "#2563eb"))
        err_fig.add_trace(_cdf_trace((vtech["rsrp_measured"] - vtech["phase25_dynamic_rsrp"]).abs(), "Phase25 dynamic abs error", "#16a34a"))
        err_fig.update_layout(
            title=f"{tech} held-out DT absolute error CDF",
            height=430,
            xaxis_title="Absolute error (dB)",
            yaxis_title="Cumulative %",
            yaxis_range=[0, 100],
        )
        st.plotly_chart(err_fig, use_container_width=True)

        sample_cols = [
            "phase25_split",
            "technology",
            "band",
            "clutter_class",
            "obstruction_branch",
            "terrain_bucket",
            "cell_key",
            "rsrp_measured",
            "phase24_no_lock_reference_rsrp",
            "phase25_dynamic_rsrp",
            "phase25_total_dynamic_correction_db",
            "local_residual_support_n",
            "phase25_confidence",
        ]
        st.subheader("Held-out DT validation rows")
        st.dataframe(vtech[[col for col in sample_cols if col in vtech.columns]].head(500), use_container_width=True)

    image_paths = [
        PHASE25_IMAGE_DIR / f"phase25_{tech.lower()}_full_polygon_cdf.png",
        PHASE25_IMAGE_DIR / f"phase25_{tech.lower()}_validation_dt_cdf.png",
        PHASE25_IMAGE_DIR / f"phase25_{tech.lower()}_validation_abs_error_cdf.png",
    ]
    existing = [path for path in image_paths if path.exists()]
    if existing:
        st.subheader("Phase 25 Generated PNGs")
        cols = st.columns(len(existing))
        for col, path in zip(cols, existing):
            col.image(str(path), use_container_width=True)


def _render_phase26(view_mode: str, tech: str, aggregation: str) -> None:
    st.header("Phase 26: Corrected Outdoor Obstruction Profile")
    serving = load_phase26_serving(tech)
    dt = load_phase26_dt()
    summary = load_phase26_summary().get("technology", {}).get(tech, {})
    if serving.empty:
        st.error(f"Phase 26 output not found under {PHASE26_DIR}.")
        return

    value_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    physical_col = f"phase26_physical_with_terrain_{value_suffix}_rsrp"
    final_col = f"phase26_with_terrain_{value_suffix}_rsrp"
    serving["phase26_final_minus_physical_db"] = serving[final_col] - serving[physical_col]

    metrics = summary.get("dt_phase26_calibrated_with_terrain", {})
    cols = st.columns(6)
    cols[0].metric("Grid rows", f"{int(summary.get('grid_rows', 0)):,}")
    cols[1].metric("Candidates", f"{int(summary.get('candidate_rows', 0)):,}")
    cols[2].metric("No coverage", f"{int(summary.get('no_coverage_grid_rows', 0)):,}")
    cols[3].metric("DT MAE", f"{float(metrics.get('mae', np.nan)):.2f} dB")
    cols[4].metric("DT bias", f"{float(metrics.get('bias', np.nan)):.2f} dB")
    cols[5].metric("Mean building loss", f"{float(summary.get('mean_building_geo_correction_db', np.nan)):.1f} dB")
    st.caption(
        "Phase 26 removes the old multi-building summed outdoor diffraction. "
        "Outdoor obstruction is dominant obstacle per ray with median fan selection; "
        "indoor entry/depth and DEM terrain stay separate."
    )

    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption("Corrected physical + terrain")
        _render_map(serving, physical_col, f"{tech} Phase 26 physical", view_mode)
    with map_cols[1]:
        st.caption("Final calibrated after terrain")
        _render_map(serving, final_col, f"{tech} Phase 26 final", view_mode)

    diagnostic_options = {
        "Final - physical delta": ("phase26_final_minus_physical_db", "delta"),
        "Building correction mean": ("building_geo_correction_db_mean", "delta"),
        "Terrain loss mean": ("terrain_diffraction_loss_db_mean", "loss"),
        "Terrain loss max": ("terrain_diffraction_loss_db_max", "loss"),
        "Terrain obstructed share": ("terrain_obstructed_share", "confidence"),
    }
    selected_diag = st.radio("Phase 26 diagnostic map", list(diagnostic_options.keys()), index=0, horizontal=True)
    diag_col, diag_kind = diagnostic_options[selected_diag]
    _render_map(serving, diag_col, f"{tech} {selected_diag}", view_mode, value_kind=diag_kind)

    fig = go.Figure()
    fig.add_trace(_cdf_trace(serving[physical_col], "Phase26 physical", "#2563eb"))
    fig.add_trace(_cdf_trace(serving[final_col], "Phase26 final calibrated", "#16a34a"))
    fig.update_layout(
        title=f"{tech} Phase 26 full-polygon CDF - {aggregation}",
        height=430,
        xaxis_title="RSRP (dBm)",
        yaxis_title="Cumulative %",
        yaxis_range=[0, 100],
        xaxis_range=[-140, -45],
    )
    st.plotly_chart(fig, use_container_width=True)

    dtech = dt[dt["assigned_technology"].astype(str) == tech].copy() if not dt.empty else pd.DataFrame()
    if not dtech.empty and "phase26_with_terrain_calibrated_rsrp" in dtech.columns:
        dt_fig = go.Figure()
        dt_fig.add_trace(_cdf_trace(dtech["rsrp_measured"], "DT measured", "#111827"))
        dt_fig.add_trace(_cdf_trace(dtech["phase26_with_terrain_calibrated_rsrp"], "Phase26 final at DT", "#16a34a"))
        dt_fig.update_layout(
            title=f"{tech} Phase 26 DT-location CDF",
            height=430,
            xaxis_title="RSRP (dBm)",
            yaxis_title="Cumulative %",
            yaxis_range=[0, 100],
            xaxis_range=[-140, -45],
        )
        st.plotly_chart(dt_fig, use_container_width=True)


def _render_phase27(view_mode: str, tech: str, aggregation: str) -> None:
    st.header("Phase 27: Dynamic Calibration on Corrected Obstruction")
    serving = load_phase27_serving(tech)
    vdt = load_phase27_validation_dt()
    summary = load_phase27_summary().get("technology", {}).get(tech, {})
    if serving.empty:
        st.error(f"Phase 27 output not found under {PHASE27_DIR}.")
        return

    value_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    physical_col = "phase26_physical_best_rsrp" if value_suffix == "best" else "phase26_physical_mean_rsrp"
    dynamic_col = f"phase27_dynamic_{value_suffix}_rsrp"

    ho_out = summary.get("held_out_outdoor_phase27_dynamic", {})
    ho_in = summary.get("held_out_misclassified_indoor_dt_ref_only", {})
    ins = summary.get("insample_outdoor_phase27_dynamic", {})
    ho = ho_out  # headline = outdoor held-out (the only real ground truth)
    cols = st.columns(6)
    cols[0].metric("Grid rows", f"{int(summary.get('grid_rows', 0)):,}")
    cols[1].metric("No coverage", f"{int(summary.get('no_coverage_grid_rows', 0)):,}")
    cols[2].metric("Outdoor held-out MAE", f"{float(ho.get('mae', np.nan)):.2f} dB")
    cols[3].metric("Outdoor in-sample MAE", f"{float(ins.get('mae', np.nan)):.2f} dB")
    cols[4].metric("Outdoor held-out bias", f"{float(ho.get('bias', np.nan)):.2f} dB")
    cols[5].metric("Outdoor held-out p90 |err|", f"{float(ho.get('p90_abs', np.nan)):.1f} dB")
    m_out = summary.get("mean_phase27_dynamic_outdoor_best_rsrp")
    m_in = summary.get("mean_phase27_dynamic_indoor_best_rsrp")
    if m_out is not None and m_in is not None:
        st.caption(
            f"Mean serving RSRP - outdoor {float(m_out):.1f} dBm, indoor {float(m_in):.1f} dBm "
            f"(indoor - outdoor gap {float(m_in) - float(m_out):.1f} dB)."
        )
    st.caption(
        "Outdoor: Phase 26 corrected-obstruction physical + Phase 25 hierarchical dynamic calibration "
        "(fit on OUTDOOR held-out DT only). Indoor: Phase 26 physical with its single frequency-and-depth "
        "O2I term, NO Phase 27 extra O2I and NO DT calibration - there is no indoor drive test."
    )

    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption("Phase 26 physical (corrected obstruction)")
        _render_map(serving, physical_col, f"{tech} Phase 26 physical", view_mode)
    with map_cols[1]:
        st.caption("Phase 27 dynamic-calibrated")
        _render_map(serving, dynamic_col, f"{tech} Phase 27 dynamic", view_mode)

    diagnostic_options = {
        "Total dynamic correction": ("phase27_total_dynamic_correction_db_mean", "delta"),
        "Local residual correction": ("local_residual_correction_db_mean", "delta"),
        "Sector correction": ("sector_correction_db_mean", "delta"),
        "Confidence": ("phase27_confidence_mean", "confidence"),
    }
    selected_diag = st.radio("Phase 27 diagnostic map", list(diagnostic_options.keys()), index=0, horizontal=True)
    diag_col, diag_kind = diagnostic_options[selected_diag]
    if diag_col in serving.columns:
        _render_map(serving, diag_col, f"{tech} {selected_diag}", view_mode, value_kind=diag_kind)

    # ---- Full-polygon serving CDF ----
    fig = go.Figure()
    fig.add_trace(_cdf_trace(serving[physical_col], "Phase 26 physical", "#6b7280"))
    fig.add_trace(_cdf_trace(serving["phase27_no_lock_best_rsrp"], "Phase 26 + phase19 bias", "#2563eb"))
    fig.add_trace(_cdf_trace(serving[dynamic_col], "Phase 27 dynamic", "#16a34a"))
    fig.update_layout(
        title=f"{tech} Phase 27 full-polygon serving CDF - {aggregation}",
        height=430, xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
        yaxis_range=[0, 100], xaxis_range=[-140, -45],
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Single 4-curve view -------------------------------------------------
    #  1) DT measured                    - ground truth, at DT point locations
    #  2) Predicted at those same DT pts  - accuracy where we have measurements
    #  3) Predicted - outdoor, whole polygon (incl. areas with no DT)
    #  4) Predicted - indoor,  whole polygon (no DT exists indoors)
    vt = vdt[vdt["technology"].astype(str) == tech].copy() if not vdt.empty else pd.DataFrame()
    env_col = "serving_environment" if "serving_environment" in serving.columns else None
    if not vt.empty and env_col is not None:
        vt_out = vt[vt["obstruction_branch"].astype(str) != "indoor"]
        dt_measured = pd.to_numeric(vt_out["rsrp_measured"], errors="coerce")
        dt_predicted = pd.to_numeric(vt_out["phase27_dynamic_rsrp"], errors="coerce")
        poly_out = serving.loc[serving[env_col] == "outdoor", dynamic_col]
        poly_in = serving.loc[serving[env_col] == "indoor", dynamic_col]

        st.subheader("DT vs predicted, and whole-polygon prediction (outdoor / indoor)")
        four = go.Figure()
        four.add_trace(_cdf_trace(dt_measured, "1 - DT measured (outdoor, at DT points)", "#e5e7eb"))
        four.add_trace(_cdf_trace(dt_predicted, "2 - Predicted at those DT points", "#3b82f6"))
        four.add_trace(_cdf_trace(poly_out, "3 - Predicted - outdoor (whole polygon)", "#22c55e"))
        four.add_trace(_cdf_trace(poly_in, "4 - Predicted - indoor (whole polygon, one O2I term, no DT)", "#f59e0b"))
        four.update_layout(
            title=f"{tech} Phase 27: DT accuracy vs whole-polygon prediction",
            height=470, xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
            yaxis_range=[0, 100], xaxis_range=[-140, -45],
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        )
        st.plotly_chart(four, use_container_width=True)
        st.caption(
            "1 vs 2: accuracy where measured (all DT is outdoor). 2 vs 3: is the DT route representative "
            "of the whole outdoor area (gap = DT driven on the better roads). 3 vs 4 is a geographic "
            "population comparison, not a penetration-loss measurement; use the matched-cell CDF below for O2I."
        )

        mc = st.columns(3)
        mc[0].metric("Outdoor held-out MAE", f"{float(ho_out.get('mae', np.nan)):.2f} dB",
                     help=f"bias {float(ho_out.get('bias', np.nan)):.2f} dB")
        mc[1].metric("Outdoor in-sample MAE", f"{float(ins.get('mae', np.nan)):.2f} dB")
        mc[2].metric("Indoor - outdoor gap (mean serving)",
                     f"{(float(m_in) - float(m_out)):.1f} dB" if (m_out is not None and m_in is not None) else "n/a")

        # ---- SEPARATE chart: matched-pixel O2I (geography removed) --------------------
        matched = load_phase27_indoor_matched(tech)
        if not matched.empty:
            gap = (matched["outdoor_equiv"] - matched["indoor_pred"]).replace([np.inf, -np.inf], np.nan).dropna()
            st.subheader("Matched-pixel O2I — indoor vs outdoor at the SAME cell")
            mfig = go.Figure()
            mfig.add_trace(_cdf_trace(matched["outdoor_equiv"], "Outdoor-equivalent (same indoor cells, no O2I)", "#22c55e"))
            mfig.add_trace(_cdf_trace(matched["indoor_pred"], "Indoor prediction (same cells, with O2I)", "#f59e0b"))
            mfig.update_layout(
                title=f"{tech} Phase 27: penetration loss on matched cells (n={len(matched):,})",
                height=430, xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                yaxis_range=[0, 100], xaxis_range=[-140, -45],
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            )
            st.plotly_chart(mfig, use_container_width=True)
            g1, g2, g3 = st.columns(3)
            g1.metric("Median O2I (matched)", f"{gap.median():.1f} dB")
            g2.metric("P10–P90 O2I", f"{gap.quantile(0.1):.0f} – {gap.quantile(0.9):.0f} dB")
            g3.metric("Whole-polygon CDF gap", f"{(float(m_in) - float(m_out)):.1f} dB" if (m_out is not None and m_in is not None) else "n/a",
                      help="Smaller than the matched gap because indoor cells sit closer to sites than the average outdoor cell — geography, not a modelling error.")
            st.caption(
                "This is the real penetration loss: same cells, with vs without O2I. The whole-polygon "
                "indoor-vs-outdoor CDF above shows a smaller gap only because indoor cells are nearer to sites."
            )

        err_fig = go.Figure()
        err_fig.add_trace(_cdf_trace(
            (vt["rsrp_measured"] - vt["phase24_no_lock_reference_rsrp"]).abs(), "Phase 26 + phase19 bias", "#2563eb"))
        err_fig.add_trace(_cdf_trace(
            (vt["rsrp_measured"] - vt["phase27_dynamic_rsrp"]).abs(), "Phase 27 dynamic", "#16a34a"))
        err_fig.update_layout(
            title=f"{tech} Phase 27 held-out DT absolute-error CDF",
            height=430, xaxis_title="Absolute error (dB)", yaxis_title="Cumulative %",
            yaxis_range=[0, 100], xaxis_range=[0, 40],
        )
        st.plotly_chart(err_fig, use_container_width=True)


def _render_phase29(view_mode: str, tech: str, aggregation: str) -> None:
    st.header("Phase 29: Real Per-Tilt Antenna Patterns")
    serving = load_phase29_serving(tech)
    vdt = load_phase29_validation_dt()
    p27_vdt = load_phase27_validation_dt()
    summary = load_phase29_summary().get("technology", {}).get(tech, {})
    if serving.empty or not summary:
        st.error(f"Phase 29 output not found under {PHASE29_DIR}.")
        return

    gen = summary.get("held_out_outdoor_phase27_generic", {}) or {}
    real = summary.get("held_out_outdoor_phase29_real_antenna", {}) or {}
    gd = summary.get("antenna_gain_delta_db", {}) or {}
    ant = "CommScope CCVVPX308 (698-806 / 1710-1880 MHz, T0-T10)" if tech == "4G" \
        else "Kathrein 800109221 (3300-3590 MHz, eTilt 2-12)"
    st.caption(
        f"Real measured per-electrical-tilt pattern for {tech}: **{ant}**, replacing the generic "
        f"3GPP parametric antenna (18 dBi / 65 deg H / 6 deg V). Base pipeline = Phase 27 "
        f"(Phase 26 physical + Phase 25 dynamic calibration on outdoor DT). Phase 28 not used."
    )

    c = st.columns(6)
    c[0].metric("Antenna gain delta (median)", f"{float(gd.get('median', np.nan)):+.2f} dB",
                help=f"P10 {float(gd.get('p10', np.nan)):.1f} / P90 {float(gd.get('p90', np.nan)):.1f} dB (real minus generic)")
    c[1].metric("Held-out MAE", f"{float(real.get('mae', np.nan)):.2f} dB",
                delta=f"{float(real.get('mae', np.nan)) - float(gen.get('mae', np.nan)):+.2f} vs generic", delta_color="inverse")
    c[2].metric("Held-out bias", f"{float(real.get('bias', np.nan)):+.2f} dB",
                delta=f"{float(real.get('bias', np.nan)) - float(gen.get('bias', np.nan)):+.2f}", delta_color="off")
    c[3].metric("Held-out p90 |err|", f"{float(real.get('p90_abs', np.nan)):.1f} dB",
                delta=f"{float(real.get('p90_abs', np.nan)) - float(gen.get('p90_abs', np.nan)):+.1f} vs generic", delta_color="inverse")
    c[4].metric("In-sample MAE", f"{float((summary.get('insample_outdoor_phase29', {}) or {}).get('mae', np.nan)):.2f} dB")
    c[5].metric("No coverage", f"{int((summary.get('serving', {}) or {}).get('no_coverage', 0)):,}")

    # ---- maps: Phase 29 dynamic + antenna-gain-delta diagnostic ----
    dyn_col = "phase29_dynamic_mean_rsrp" if aggregation.startswith("Frontend") else "phase29_dynamic_best_rsrp"
    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption("Phase 29 serving RSRP (real antenna)")
        _render_map(serving, dyn_col, f"{tech} Phase 29 RSRP", view_mode)
    with map_cols[1]:
        cand = load_phase29_candidates()
        if not cand.empty and "phase29_antenna_gain_delta_db" in cand.columns:
            ct = cand[cand["technology"].astype(str) == tech]
            gmap = ct.groupby("grid_id")["phase29_antenna_gain_delta_db"].mean().reset_index()
            gmap = serving[["grid_id", "center_lat", "center_lon", "min_lat", "max_lat", "min_lon", "max_lon"]].merge(gmap, on="grid_id", how="left")
            st.caption("Antenna gain delta (real − generic), serving-candidate mean")
            _render_map(gmap, "phase29_antenna_gain_delta_db", f"{tech} gain delta", view_mode, value_kind="delta")

    # ---- held-out outdoor DT: generic vs real ----
    vt = vdt[(vdt["technology"].astype(str) == tech) & (vdt["obstruction_branch"].astype(str) != "indoor")].copy()
    p27 = p27_vdt[(p27_vdt["technology"].astype(str) == tech) & (p27_vdt["obstruction_branch"].astype(str) != "indoor")].copy() \
        if not p27_vdt.empty else pd.DataFrame()
    if not vt.empty:
        st.subheader("Held-out outdoor DT — generic vs real antenna")
        f1 = go.Figure()
        f1.add_trace(_cdf_trace(vt["rsrp_measured"], "DT measured", "#e5e7eb"))
        if not p27.empty:
            f1.add_trace(_cdf_trace(p27["phase27_dynamic_rsrp"], "Phase 27 (generic 18/65/6)", "#2563eb"))
        f1.add_trace(_cdf_trace(vt["phase29_dynamic_rsrp"], "Phase 29 (real per-tilt pattern)", "#16a34a"))
        f1.update_layout(title=f"{tech} Phase 29 held-out outdoor DT CDF", height=430,
                         xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                         yaxis_range=[0, 100], xaxis_range=[-140, -45])
        st.plotly_chart(f1, use_container_width=True)

        f2 = go.Figure()
        if not p27.empty:
            f2.add_trace(_cdf_trace((p27["rsrp_measured"] - p27["phase27_dynamic_rsrp"]).abs(), "Phase 27 generic", "#2563eb"))
        f2.add_trace(_cdf_trace((vt["rsrp_measured"] - vt["phase29_dynamic_rsrp"]).abs(), "Phase 29 real antenna", "#16a34a"))
        f2.update_layout(title=f"{tech} Phase 29 held-out outdoor DT absolute-error CDF", height=430,
                         xaxis_title="Absolute error (dB)", yaxis_title="Cumulative %",
                         yaxis_range=[0, 100], xaxis_range=[0, 40])
        st.plotly_chart(f2, use_container_width=True)

    # ---- gain delta by azimuth offset ----
    cand = load_phase29_candidates()
    if not cand.empty:
        ct = cand[cand["technology"].astype(str) == tech].copy()
        ct["az_bin"] = pd.cut(pd.to_numeric(ct["azimuth_delta_deg"], errors="coerce").abs(),
                              bins=[0, 20, 45, 90, 180], labels=["0-20 (main beam)", "20-45", "45-90", "90-180 (back/side)"])
        g = ct.groupby("az_bin", observed=True)["phase29_antenna_gain_delta_db"].median().reset_index()
        bar = go.Figure(go.Bar(x=g["az_bin"].astype(str), y=g["phase29_antenna_gain_delta_db"], marker_color="#16a34a"))
        bar.update_layout(title=f"{tech} antenna gain delta (real − generic) by azimuth offset — median dB",
                          height=330, yaxis_title="dB", xaxis_title="azimuth offset from sector boresight")
        st.plotly_chart(bar, use_container_width=True)
        st.caption(
            "Near boresight the real pattern ≈ generic. Far off-axis (90-180 deg) the real antenna "
            "suppresses wrong-direction sectors much harder than the generic 65 deg cone — that's the "
            "'omni-like over-reach' fix, and why the error tail (p90) tightens while median MAE barely moves "
            "(the DT calibration was already compensating for the generic-antenna error)."
        )


def _render_phase31(view_mode: str, tech: str, aggregation: str) -> None:
    st.header(f"Phase 31: Real Antenna Pattern on the Phase 28 base ({tech}) — test only")
    serving = load_phase31_serving(tech)
    dts = load_phase31_dt_scored(tech)
    summary = load_phase31_summary()
    if serving.empty or not summary:
        st.error(f"Phase 31 {tech} output not found under {PHASE31_DIR}. Run test_project210_phase31_phase28_real_antenna.py.")
        return

    ts = summary.get("technology", {}).get(tech, {}) or summary
    ho = ts.get("held_out_outdoor_dt", {})
    g = ho.get("phase28_generic_antenna", {}) or {}
    r = ho.get("phase31_real_antenna", {}) or {}
    gd = ts.get("antenna_gain_delta_db", {}) or {}
    sg = ts.get("serving_grid", {}) or {}
    pat_name = ts.get("antenna_pattern", "CCVVPX308" if tech == "4G" else "Kathrein 800109221")
    st.caption(
        "Base = **Phase 28** pipeline (RSRP reference fix + Water override + light per-clutter residual). "
        f"Phase 31 adds the real **{pat_name}** per-electrical-tilt pattern (gain delta vs the generic "
        "3GPP 18/65/6), re-fitting the residual on the antenna-adjusted physical. NOT built on Phase 27."
    )

    agg_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    p31_col = f"phase31_final_{agg_suffix}_rsrp"
    p28_col = f"phase28_final_{agg_suffix}_rsrp"

    c = st.columns(5)
    c[0].metric("Antenna gain delta (median)", f"{float(gd.get('median', np.nan)):+.2f} dB",
                help=f"P10 {float(gd.get('p10', np.nan)):.1f} / P90 {float(gd.get('p90', np.nan)):.1f}")
    c[1].metric("Held-out outdoor DT MAE", f"{float(r.get('mae', np.nan)):.2f} dB",
                delta=f"{float(r.get('mae', np.nan)) - float(g.get('mae', np.nan)):+.2f} vs Phase 28 generic", delta_color="inverse")
    c[2].metric("Held-out bias", f"{float(r.get('bias', np.nan)):+.2f} dB")
    c[3].metric("Held-out p90 |err|", f"{float(r.get('p90_abs', np.nan)):.1f} dB",
                delta=f"{float(r.get('p90_abs', np.nan)) - float(g.get('p90_abs', np.nan)):+.1f} vs generic", delta_color="inverse")
    c[4].metric("Serving median / no-cov", f"{sg.get('median_rsrp', '?')} dBm / {sg.get('no_coverage_rows', 0)}")

    # ---- maps: Phase 28 (generic) vs Phase 31 (real antenna) ----
    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption(f"Phase 28 — generic 3GPP antenna ({agg_suffix})")
        _render_map(serving, p28_col, f"{tech} Phase 28 generic", view_mode)
    with map_cols[1]:
        st.caption(f"Phase 31 — real {pat_name} pattern ({agg_suffix})")
        _render_map(serving, p31_col, f"{tech} Phase 31 real antenna", view_mode)
    if "antenna_gain_delta_db_mean" in serving.columns:
        st.caption("Antenna gain delta (real − generic), serving-candidate mean")
        _render_map(serving, "antenna_gain_delta_db_mean", f"{tech} gain delta", view_mode, value_kind="delta")

    # ---- 4-curve view (same as Phase 28): DT accuracy vs whole-polygon prediction ----
    if not dts.empty and "serving_environment" in serving.columns:
        out_dt = dts[dts["obstruction_branch"].astype(str) != "indoor"]
        four = go.Figure()
        four.add_trace(_cdf_trace(out_dt["rsrp_measured"], "1 - DT measured (outdoor, at DT points)", "#e5e7eb"))
        four.add_trace(_cdf_trace(out_dt["phase31_rsrp"], "2 - Predicted at those DT points", "#3b82f6"))
        four.add_trace(_cdf_trace(serving.loc[serving["serving_environment"] == "outdoor", p31_col],
                                  f"3 - Predicted - outdoor (whole polygon, {agg_suffix})", "#22c55e"))
        four.add_trace(_cdf_trace(serving.loc[serving["serving_environment"] == "indoor", p31_col],
                                  f"4 - Predicted - indoor (whole polygon, {agg_suffix})", "#f59e0b"))
        four.update_layout(title=f"{tech} Phase 31: DT accuracy vs whole-polygon prediction", height=470,
                           xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                           yaxis_range=[0, 100], xaxis_range=[-140, -45],
                           legend=dict(orientation="h", yanchor="bottom", y=-0.35))
        st.plotly_chart(four, use_container_width=True)
        st.caption(
            "Same 4-curve view as Phase 28, now on the real-antenna prediction. 1 vs 2: accuracy where "
            "measured. 2 vs 3: DT route vs whole outdoor polygon. 3 vs 4: outdoor vs indoor (geographic "
            "populations, not a penetration-loss reading)."
        )

    # ---- CDF: Phase 28 vs Phase 31, full polygon ----
    fig = go.Figure()
    fig.add_trace(_cdf_trace(serving[p28_col], f"Phase 28 generic ({agg_suffix})", "#2563eb"))
    fig.add_trace(_cdf_trace(serving[p31_col], f"Phase 31 real antenna ({agg_suffix})", "#16a34a"))
    fig.update_layout(title=f"{tech} full-polygon serving CDF — Phase 28 vs Phase 31 ({aggregation})", height=430,
                      xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                      yaxis_range=[0, 100], xaxis_range=[-140, -45])
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Held-out outdoor DT MAE: Phase 28 generic {float(g.get('mae', np.nan)):.2f} → Phase 31 real "
        f"{float(r.get('mae', np.nan)):.2f} dB. On this lightly-calibrated base the real antenna adds per-sector "
        "directionality (±4–8 dB) but the light per-clutter residual can't correct where the pattern is off, "
        "so the DT fit gets slightly worse — unlike on Phase 27, where the sector/local calibration absorbs it."
    )

    with st.expander("phase31_summary.json"):
        st.json(summary)


def _p3x_heldout_mae(tag: str) -> dict:
    """Held-out outdoor 5G DT MAE for a phase33/34/35 summary, tolerant of key drift."""
    s = load_p3x_summary(tag)
    ho = s.get("5g_held_out_outdoor_dt", {})
    for k, v in ho.items():
        if k.startswith(tag):
            return v
    return {}


def _render_phase32_33_34(view_mode: str, tech: str) -> None:
    st.header("Phase 32 / 33 / 34 / 35 — 5G path-loss & antenna experiments")
    st.caption(
        "All three swap ONE thing on the Phase 28/31 5G base: Phase 33/35 replace COST-231 @ 2600 MHz with "
        "**3GPP TR 38.901 UMa @ the real 3300 MHz** carrier; Phase 34 does the same path loss but a different "
        "antenna. Phase 32 is a read-only audit of the Phase 31 antenna selection. 4G is untouched (read-only control)."
    )

    # ---------- ladder: held-out 5G outdoor DT MAE ----------
    p31 = load_phase31_summary().get("technology", {}).get("5G", {})
    p31_ho = (p31.get("held_out_outdoor_dt", {}) or {}).get("phase31_real_antenna", {})
    rows = [{
        "phase": "31  COST-231 @2600 + real antenna (current 5G base)",
        "n": p31_ho.get("n"), "MAE dB": round(p31_ho.get("mae", np.nan), 2),
        "bias dB": round(p31_ho.get("bias", np.nan), 2), "p90 |err|": round(p31_ho.get("p90_abs", np.nan), 1),
    }]
    for tag, label in [
        ("phase33", "33  38.901 @3300, Kathrein 2-9 + generic 0/1"),
        ("phase34", "34  38.901 @3300, Ericsson 0-8 + generic 9"),
        ("phase35", "35  38.901 @3300, Kathrein ALL tilts"),
    ]:
        m = _p3x_heldout_mae(tag)
        if m:
            rows.append({"phase": label, "n": m.get("n"), "MAE dB": round(m.get("mae", np.nan), 2),
                         "bias dB": round(m.get("bias", np.nan), 2), "p90 |err|": round(m.get("p90_abs", np.nan), 1)})
    st.subheader("Held-out 5G outdoor DT accuracy — lower MAE is better")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Switching to 38.901 @ 3300 MHz did **not** help — every 38.901 row is worse than the COST-231 base. "
        "Making the antenna 'more real' (33 → 34 → 35) moves MAE by ±1 dB with no consistent direction, so the "
        "antenna is not the cause. The 5G error is in the NLOS path-loss layer + the light residual, not the antenna."
    )

    # ---------- serving maps per phase ----------
    for tag, label in [("phase33", "Phase 33"), ("phase34", "Phase 34"), ("phase35", "Phase 35")]:
        serving = load_p3x_serving(tag)
        if serving.empty:
            continue
        phys_col, final_col = f"{tag}_physical_rsrp", f"{tag}_final_rsrp"
        st.markdown(f"### {label} — {_P3X[tag][1]}")
        cols = st.columns(3)
        with cols[0]:
            st.caption(f"{label} physical (no residual)")
            _render_map(serving, phys_col, f"5G {label} physical", view_mode)
        with cols[1]:
            st.caption(f"{label} final (+ per-clutter residual)")
            _render_map(serving, final_col, f"5G {label} final", view_mode)
        with cols[2]:
            if "phase31_final_best_rsrp" in serving.columns:
                st.caption("Phase 31 (COST-231) final — reference")
                _render_map(serving, "phase31_final_best_rsrp", "5G Phase 31 final", view_mode)

        dt = load_p3x_dt(tag)
        if not dt.empty:
            v = dt[(dt["split"].astype(str) == "validation") & (dt["obstruction_branch"].astype(str) != "indoor")]
            fig = go.Figure()
            fig.add_trace(_cdf_trace(v["rsrp_measured"], "1 - DT measured (outdoor)", "#e5e7eb"))
            fig.add_trace(_cdf_trace(v[f"{tag}_final_rsrp"], f"2 - {label} predicted at DT", "#3b82f6"))
            fig.add_trace(_cdf_trace(serving[final_col], f"3 - {label} predicted, whole polygon", "#22c55e"))
            fig.update_layout(title=f"5G {label}: DT accuracy vs whole-polygon prediction", height=430,
                              xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                              yaxis_range=[0, 100], xaxis_range=[-140, -45],
                              legend=dict(orientation="h", yanchor="bottom", y=-0.35))
            st.plotly_chart(fig, use_container_width=True)
            # Water is the single worst clutter on the 38.901 base
            vw = v[v["clutter_class"].astype(str) == "Water"]
            if not vw.empty:
                err = pd.to_numeric(vw["rsrp_measured"], errors="coerce") - pd.to_numeric(vw[f"{tag}_final_rsrp"], errors="coerce")
                st.caption(
                    f"Water DT points (n={len(vw)}): MAE {err.abs().mean():.1f} dB, bias {err.mean():+.1f} dB — "
                    "the Water override strips terrain diffraction, and 38.901's flat LOS slope lets those "
                    "shoreline points over-predict by ~40 dB. This is the largest single error source on the 38.901 base."
                )
        png = _P3X[tag][0] / f"{tag}_5g_heldout_cdf.png"
        if png.exists():
            with st.expander(f"{tag}_5g_heldout_cdf.png (script output)"):
                st.image(str(png))
        with st.expander(f"{tag}_summary.json"):
            st.json(load_p3x_summary(tag))

    # ---------- Phase 32 audit ----------
    st.markdown("### Phase 32 — read-only antenna-selection audit of Phase 31")
    audit = load_phase32_audit()
    if not audit:
        st.info("Phase 32 audit not found. Run test_project210_phase32_real_antenna_audit.py.")
    else:
        at = audit.get("technology", {})
        arows = []
        for t, d in at.items():
            arows.append({
                "tech": t,
                "candidate rows": d.get("rows"),
                "pattern freq-compatible": d.get("pattern_frequency_compatible_rows"),
                "freq-INCOMPATIBLE": d.get("pattern_frequency_incompatible_rows"),
                "tilt substituted": d.get("tilt_substituted_rows", 0),
                "delta clipped (low/high)": f"{d.get('delta_clipped_low_rows', 0)} / {d.get('delta_clipped_high_rows', 0)}",
            })
        st.dataframe(pd.DataFrame(arows), hide_index=True, use_container_width=True)
        st.caption(
            "The audit flags that **all 45,000 Phase 31 5G rows are frequency-incompatible**: the candidate carrier is "
            "labelled 2600 MHz but the Kathrein pattern file is 3300-3590 MHz, and 15,293 rows (Etilt 0/1) had no "
            "matching tilt file. This is why Phase 33 moved to the real 3300 MHz carrier."
        )
        with st.expander("phase32_real_antenna_audit.json"):
            st.json(audit)


def _render_phase36(view_mode: str, tech: str, aggregation: str) -> None:
    st.header(f"Phase 36 (FINAL) — {tech}: per-tech physical + real antenna + full Phase 25 calibration")
    serving = load_phase36_serving(tech)
    vt = load_phase36_validation(tech)
    summary = load_phase36_summary()
    if serving.empty or not summary:
        st.error(f"Phase 36 {tech} output not found under {PHASE36_DIR}. Run test_project210_phase36_final.py.")
        return
    ts = summary.get("technology", {}).get(tech, {})
    pr = summary.get("params", {})
    st.caption(
        f"**{tech} physical:** " + (
            "COST-231 + RSRP-per-RE reference fix + real CommScope CCVVPX308 pattern delta."
            if tech == "4G" else
            f"SAME COST-231 core as 4G (production raw @ 2600 MHz) + the −2.58 dB 2600→3300 MHz n78 term "
            f"+ real Kathrein 800109221 pattern delta + a {pr.get('g5_level_anchor_db', 0):+.1f} dB all-outdoor-DT "
            "level anchor. COST-231 (not 38.901) so the 5G slope matches 4G."
        )
        + " Terrain kept for Water. Then the Phase 25 hierarchical calibration "
        f"(tech_band → clutter_terrain → sector → local IDW field). {pr.get('dt_serving_reassigned', 0)} DT points "
        "re-assigned to a better-aligned same-site sector; deep-backlobe points dropped from the fit."
    )
    lim = summary.get("known_limitation")
    if lim:
        st.warning(lim)

    phys = ts.get("held_out_outdoor_physical_no_calibration", {})
    fin = ts.get("held_out_outdoor_final", {})
    fin_bl = ts.get("held_out_outdoor_final_incl_backlobe_ref", {})
    p27 = ts.get("reference_phase27_dynamic", {})
    p31 = ts.get("reference_phase31_real_antenna", {})
    sg = ts.get("serving_grid", {})
    c = st.columns(5)
    c[0].metric("Held-out DT MAE (final)", f"{float(fin.get('mae', np.nan)):.2f} dB",
                delta=f"{float(fin.get('mae', np.nan)) - float(phys.get('mae', np.nan)):+.2f} vs physical", delta_color="inverse")
    c[1].metric("incl. backlobe DT", f"{float(fin_bl.get('mae', np.nan)):.2f} dB")
    c[2].metric("Bias / p90 |err|", f"{float(fin.get('bias', np.nan)):+.2f} / {float(fin.get('p90_abs', np.nan)):.1f}")
    c[3].metric("vs Phase 27 dynamic", f"{float(p27.get('mae', np.nan)):.2f} dB",
                delta=f"{float(fin.get('mae', np.nan)) - float(p27.get('mae', np.nan)):+.2f}", delta_color="inverse")
    c[4].metric("vs Phase 31 real-ant", f"{float(p31.get('mae', np.nan)):.2f} dB",
                delta=f"{float(fin.get('mae', np.nan)) - float(p31.get('mae', np.nan)):+.2f}", delta_color="inverse")

    agg_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    phys_col = f"phase36_physical_{agg_suffix}_rsrp"
    final_col = f"phase36_final_{agg_suffix}_rsrp"
    st.caption(f"Aggregation: **{aggregation}** → `{final_col}`.")
    mcols = st.columns(2)
    with mcols[0]:
        st.caption(f"Phase 36 physical ({agg_suffix}) — before calibration")
        _render_map(serving, phys_col, f"{tech} Phase 36 physical", view_mode)
    with mcols[1]:
        st.caption(f"Phase 36 final ({agg_suffix}) — + hierarchical calibration")
        _render_map(serving, final_col, f"{tech} Phase 36 final", view_mode)

    if not vt.empty and "serving_environment" in serving.columns:
        clean = vt[(vt["obstruction_branch"].astype(str) != "indoor") & (~vt.get("p36_backlobe", False))]
        four = go.Figure()
        four.add_trace(_cdf_trace(clean["rsrp_measured"], "1 - DT measured (outdoor)", "#e5e7eb"))
        four.add_trace(_cdf_trace(clean["phase36_final_rsrp"], "2 - Phase 36 predicted at DT", "#3b82f6"))
        four.add_trace(_cdf_trace(serving.loc[serving["serving_environment"] == "outdoor", final_col],
                                  f"3 - Predicted outdoor polygon ({agg_suffix})", "#22c55e"))
        four.add_trace(_cdf_trace(serving.loc[serving["serving_environment"] == "indoor", final_col],
                                  f"4 - Predicted indoor polygon ({agg_suffix})", "#f59e0b"))
        four.update_layout(title=f"{tech} Phase 36: DT accuracy vs whole-polygon prediction", height=460,
                           xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                           yaxis_range=[0, 100], xaxis_range=[-140, -45],
                           legend=dict(orientation="h", yanchor="bottom", y=-0.35))
        st.plotly_chart(four, use_container_width=True)

        f2 = go.Figure()
        f2.add_trace(_cdf_trace((clean["rsrp_measured"] - clean["phase24_physical_with_terrain_rsrp"]).abs(),
                                "Physical abs err (no calibration)", "#6b7280"))
        f2.add_trace(_cdf_trace((clean["rsrp_measured"] - clean["phase36_final_rsrp"]).abs(),
                                "Final abs err", "#16a34a"))
        f2.update_layout(title=f"{tech} Phase 36 held-out DT absolute error", height=400,
                         xaxis_title="|error| (dB)", yaxis_title="Cumulative %",
                         yaxis_range=[0, 100], xaxis_range=[0, 40])
        st.plotly_chart(f2, use_container_width=True)

    with st.expander("phase36_summary.json"):
        st.json(summary)

    # ---------- v1 vs v2 (distance-shaped calibration experiment) ----------
    v2s = load_phase36v2_summary()
    v2serv = load_phase36v2_serving(tech)
    if v2s and not v2serv.empty:
        st.divider()
        st.subheader("Phase 36 v2 — 4G cell frequency RE-BAND")
        st.caption(
            "The 4G drive‑test `earfcn` shows the phones were on 1800 / 2100 / 2600 MHz, but Phase 26 labelled "
            "the cells 700 MHz and matched by nearest location — so the v1 4G physical ran ~13–18 dB too hot and "
            "the 4G polygon read ~6 dB below 5G between the DT roads. v2 takes each 4G cell's real frequency from "
            "the median measured E‑ARFCN of its DT and shifts that cell's raw by −33.9·log10(f_true/f_label). "
            "Calibration = Phase 36 v1's exactly. Nothing else and no other phase is changed."
        )
        rb = v2s.get("reband", [])
        if rb:
            with st.expander(f"{len(rb)} re-banded 4G cells"):
                st.dataframe(pd.DataFrame(rb), hide_index=True, use_container_width=True)
        v2t = v2s.get("technology", {}).get(tech, {})
        v2f = v2t.get("v2_held_out_outdoor_final", {})
        v1f = v2t.get("v1_held_out_outdoor_final", {})
        cc = st.columns(4)
        cc[0].metric("v2 held-out MAE", f"{float(v2f.get('mae', np.nan)):.2f} dB",
                     delta=f"{float(v2f.get('mae', np.nan)) - float(v1f.get('mae', np.nan)):+.2f} vs v1", delta_color="inverse")
        cc[1].metric("v2 bias", f"{float(v2f.get('bias', np.nan)):+.2f} dB")
        cc[2].metric("v2 serving outdoor", f"{v2t.get('v2_serving_grid', {}).get('outdoor_median', '?')} dBm")
        cc[3].metric("v1 serving outdoor", f"{v2t.get('v1_serving_grid', {}).get('outdoor_median', '?')} dBm")

        gap = (v2s.get("polygon_median_by_distance", {}) or {}).get("v2", {})
        if gap:
            rows = []
            for dbk in ["<150", "150-300", "300-600", "600-1200", ">1200"]:
                a, b = gap.get("4G", {}).get(dbk), gap.get("5G", {}).get(dbk)
                if a is not None and b is not None:
                    rows.append({"distance band": dbk, "4G median": a, "5G median": b, "5G − 4G": round(b - a, 1)})
            st.markdown("**v2 serving-grid median by distance — the 4G↔5G gap the re-band closes:**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption(
                "The v1 gap of **+6–7 dB** at 150–600 m collapses to **−2 dB** (4G now marginally stronger, which is "
                "physically right for 1860–2135 MHz vs 5G's 2600 MHz on the same towers), and the 4G held‑out DT MAE "
                "is unchanged (7.87 vs 7.86). Long‑range 4G (> 600 m) is now a few dB weak — few DT samples and the "
                "uniform −13…−18 dB shift bites the already‑weak far cells. The ~5,600 DT samples the phones took on "
                "2100/2600 MHz whose cells the model has no inventory for are still un‑fixable here."
            )
        v2clean = v2serv
        cmap = st.columns(2)
        with cmap[0]:
            st.caption(f"v1 final ({tech})")
            _render_map(serving, "phase36_final_best_rsrp", f"{tech} v1 final", view_mode)
        with cmap[1]:
            st.caption(f"v2 final ({tech}) — re-banded")
            _render_map(v2clean, "phase36_final_best_rsrp", f"{tech} v2 final", view_mode)

        # ---- v2 4-curve CDF: DT accuracy vs whole-polygon prediction ----
        v2vt = load_phase36v2_validation(tech)
        if not v2vt.empty and "serving_environment" in v2serv.columns:
            v2clean_dt = v2vt[(v2vt["obstruction_branch"].astype(str) != "indoor") & (~v2vt.get("p36_backlobe", False))]
            fourc = go.Figure()
            fourc.add_trace(_cdf_trace(v2clean_dt["rsrp_measured"], "1 - DT measured (outdoor)", "#e5e7eb"))
            fourc.add_trace(_cdf_trace(v2clean_dt["phase36_final_rsrp"], "2 - v2 predicted at DT", "#3b82f6"))
            fourc.add_trace(_cdf_trace(v2serv.loc[v2serv["serving_environment"] == "outdoor", "phase36_final_best_rsrp"],
                                       "3 - v2 predicted outdoor polygon (best)", "#22c55e"))
            fourc.add_trace(_cdf_trace(v2serv.loc[v2serv["serving_environment"] == "indoor", "phase36_final_best_rsrp"],
                                       "4 - v2 predicted indoor polygon (best)", "#f59e0b"))
            fourc.update_layout(title=f"{tech} Phase 36 v2: DT accuracy vs whole-polygon prediction", height=460,
                                xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                                yaxis_range=[0, 100], xaxis_range=[-140, -45],
                                legend=dict(orientation="h", yanchor="bottom", y=-0.35))
            st.plotly_chart(fourc, use_container_width=True)

            # ---- v1 vs v2 whole-polygon CDF, both techs, one chart ----
            v1_4g = load_phase36_serving("4G"); v1_5g = load_phase36_serving("5G")
            v2_4g = load_phase36v2_serving("4G"); v2_5g = load_phase36v2_serving("5G")
            comp = go.Figure()
            for df, lab, col in [(v1_4g, "v1 4G", "#93c5fd"), (v1_5g, "v1 5G", "#fca5a5"),
                                 (v2_4g, "v2 4G", "#2563eb"), (v2_5g, "v2 5G", "#dc2626")]:
                if not df.empty:
                    comp.add_trace(_cdf_trace(
                        pd.to_numeric(df.loc[df["serving_environment"] == "outdoor", "phase36_final_best_rsrp"], errors="coerce"),
                        lab, col))
            comp.update_layout(title="Outdoor serving CDF — v1 vs v2, 4G vs 5G", height=440,
                               xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                               yaxis_range=[0, 100], xaxis_range=[-130, -55],
                               legend=dict(orientation="h", yanchor="bottom", y=-0.3))
            st.plotly_chart(comp, use_container_width=True)
            st.caption(
                "v1 (pale): 4G sits well left of 5G — the mislabel bug. v2 (bold): 4G moves right onto 5G; the two "
                "curves cross near −90 dBm (4G stronger above, 5G stronger below), netting to ~equal — which is the "
                "+1 dB the clean co-located drive test measures."
            )
        with st.expander("phase36v2_summary.json"):
            st.json(v2s)

    # ---------- Phase 38: EARFCN-correct DT re-match (the honest version) ----------
    p38s = load_phase38_summary()
    p38serv = load_phase38_serving(tech)
    if p38s and not p38serv.empty:
        st.divider()
        st.subheader("Phase 38 — EARFCN‑correct 4G DT re‑match (honest, no re‑band)")
        d38 = p38s.get("dt_4g", {})
        st.caption(
            f"The 4G drive test spans 4 LTE carriers. v38 matches each sample by its `earfcn` band: "
            f"**B28 (700) kept: {d38.get('kept_b28', 0)}**, **B3 (1800) → 1840 MHz cells: {d38.get('rematched_b3', 0)}**, "
            f"**excluded — B1/B7 (2100/2600), no cells: {d38.get('excluded_b1b7', 0)}**, "
            f"**excluded — B3 out of 1800 footprint: {d38.get('excluded_b3_no_coverage', 0)}**. "
            "Calibration = Phase 36 v1's, on the correctly‑matched DT only. Candidate inventory unchanged — no cells invented."
        )
        t38 = p38s.get("technology", {}).get(tech, {})
        f38 = t38.get("p38_held_out_outdoor_final", {})
        cc = st.columns(4)
        cc[0].metric("v38 held‑out MAE", f"{float(f38.get('mae', np.nan)):.2f} dB",
                     help=f"scored on {t38.get('validation_dt_rows_scored', '?')} correctly‑matched DT")
        cc[1].metric("v38 bias", f"{float(f38.get('bias', np.nan)):+.2f} dB")
        cc[2].metric("v38 serving outdoor", f"{t38.get('p38_serving_grid', {}).get('outdoor_median', '?')} dBm")
        cc[3].metric("v1 serving outdoor", f"{t38.get('v1_serving_grid', {}).get('outdoor_median', '?')} dBm")
        gap38 = (p38s.get("polygon_median_by_distance", {}) or {}).get("p38", {})
        if gap38:
            rr = []
            for dbk in ["<150", "150-300", "300-600", "600-1200", ">1200"]:
                a, b = gap38.get("4G", {}).get(dbk), gap38.get("5G", {}).get(dbk)
                if a is not None and b is not None:
                    rr.append({"distance band": dbk, "4G median": a, "5G median": b, "5G − 4G": round(b - a, 1)})
            st.dataframe(pd.DataFrame(rr), hide_index=True, use_container_width=True)
        st.warning(
            "**Result: the 4G↔5G map gap is NOT resolved (+6–7 dB, same as v1).** "
            "v38 makes the calibration honest (held‑out MAE 7.31, the best of any version) and proves the gap is "
            "**not a modelling problem — it's the 4G cell inventory.** 73% of the 4G polygon is served by 700 MHz "
            "cells the drive test touched only 20 times, so their prediction (~−93.8 dBm) is uncalibrated and weak; "
            "the real serving layer (1800/2100/2600 MHz) is largely absent from the inventory. The map can only be "
            "fixed by adding the real 1800/2100/2600 MHz cells from the network cell database."
        )
        m38 = st.columns(2)
        with m38[0]:
            st.caption(f"v1 final ({tech})")
            _render_map(serving, "phase36_final_best_rsrp", f"{tech} v1 final", view_mode)
        with m38[1]:
            st.caption(f"v38 final ({tech}) — EARFCN‑matched DT")
            _render_map(p38serv, "phase36_final_best_rsrp", f"{tech} v38 final", view_mode)

        # ---- v38 4-curve CDF: DT accuracy vs whole-polygon prediction ----
        p38vt = load_phase38_validation(tech)
        if not p38vt.empty and "serving_environment" in p38serv.columns:
            clean38 = p38vt[(p38vt["obstruction_branch"].astype(str) != "indoor")
                            & (~p38vt.get("p36_backlobe", False))
                            & (~p38vt.get("p38_excluded", False))]
            f38c = go.Figure()
            f38c.add_trace(_cdf_trace(clean38["rsrp_measured"], "1 - DT measured (outdoor, EARFCN-matched)", "#e5e7eb"))
            f38c.add_trace(_cdf_trace(clean38["phase36_final_rsrp"], "2 - v38 predicted at DT", "#3b82f6"))
            f38c.add_trace(_cdf_trace(p38serv.loc[p38serv["serving_environment"] == "outdoor", "phase36_final_best_rsrp"],
                                      "3 - v38 predicted outdoor polygon (best)", "#22c55e"))
            f38c.add_trace(_cdf_trace(p38serv.loc[p38serv["serving_environment"] == "indoor", "phase36_final_best_rsrp"],
                                      "4 - v38 predicted indoor polygon (best)", "#f59e0b"))
            f38c.update_layout(title=f"{tech} Phase 38: DT accuracy vs whole-polygon prediction", height=460,
                               xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                               yaxis_range=[0, 100], xaxis_range=[-140, -45],
                               legend=dict(orientation="h", yanchor="bottom", y=-0.35))
            st.plotly_chart(f38c, use_container_width=True)
            st.caption(
                "Curve 2 tracks curve 1 tightly (held-out MAE 7.31, the best of any version) — the calibration is "
                "honest. But curve 3 (whole 4G polygon) sits left of curve 2 because 73% of the polygon is served "
                "by uncalibrated 700 MHz cells — the map limitation, not a fit error."
            )
        with st.expander("phase38_summary.json"):
            st.json(p38s)


def _render_phase39(view_mode: str, tech: str, aggregation: str) -> None:
    st.header(f"Phase 39 - {tech}: equal-power 4G/5G diagnostic")
    p39s = load_phase39_summary()
    p39serv = load_phase39_serving(tech)
    p39_4g = load_phase39_serving("4G")
    p39_5g = load_phase39_serving("5G")
    if not p39s or p39serv.empty or p39_4g.empty or p39_5g.empty:
        st.error(f"Phase 39 output not found under {PHASE39_DIR}. Run test_project210_phase39_equal_power_diagnostic.py.")
        return

    params39 = p39s.get("params", {})
    conclusion = p39s.get("manager_conclusion", {})
    st.caption(
        f"Diagnostic only: Phase 38 input, both technologies normalized to "
        f"**{params39.get('target_tx_power_dbm', 46)} dBm**. "
        f"4G B28 uses the COST-231 1500 MHz floor plus "
        f"**{params39.get('b28_low_band_offset_db', '?')} dB** low-band offset; "
        f"5G n78 uses 2600 MHz COST-231 plus "
        f"**{params39.get('n78_cost231_offset_db', '?')} dB** for 3300 MHz. "
        "Equal-power outputs answer the coverage comparison; calibrated outputs are shown only for DT accuracy reference."
    )
    if conclusion:
        st.success(
            f"{conclusion.get('answer', '4G is stronger than 5G on the equal-power outdoor polygon')} "
            f"Outdoor medians: 4G {conclusion.get('equal_power_4g_outdoor_median_dbm', '?')} dBm, "
            f"5G {conclusion.get('equal_power_5g_outdoor_median_dbm', '?')} dBm."
        )
    agg_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    equal_col = f"phase39_equal_power_{agg_suffix}_rsrp"
    final_col = f"phase39_final_{agg_suffix}_rsrp"
    agg_label = "frontend mean" if agg_suffix == "mean" else "serving cell"
    st.caption(f"Aggregation: **{aggregation}** -> `{equal_col}`.")
    t39 = p39s.get("technology", {}).get(tech, {})
    fin39 = t39.get("held_out_outdoor_calibrated_final", {})
    four_out = pd.to_numeric(p39_4g.loc[p39_4g["serving_environment"] == "outdoor", equal_col], errors="coerce")
    five_out = pd.to_numeric(p39_5g.loc[p39_5g["serving_environment"] == "outdoor", equal_col], errors="coerce")
    four_med = float(four_out.median())
    five_med = float(five_out.median())
    cc = st.columns(4)
    cc[0].metric(f"4G {agg_label} outdoor", f"{four_med:.1f} dBm")
    cc[1].metric(f"5G {agg_label} outdoor", f"{five_med:.1f} dBm")
    cc[2].metric("4G advantage", f"{four_med - five_med:+.1f} dB")
    cc[3].metric(f"{tech} calibrated DT MAE", f"{float(fin39.get('mae', np.nan)):.2f} dB",
                 help="DT accuracy is shown with the calibrated Phase 39 output, not the artificial equal-power raw diagnostic.")

    gap39 = (p39s.get("polygon_median_by_distance", {}) or {}).get("equal_power", {})
    if gap39 and agg_suffix == "best":
        rr = []
        for dbk in ["<150", "150-300", "300-600", "600-1200", ">1200"]:
            a, b = gap39.get("4G", {}).get(dbk), gap39.get("5G", {}).get(dbk)
            if a is not None and b is not None:
                rr.append({"distance band": dbk, "4G median": a, "5G median": b, "5G - 4G": round(b - a, 1)})
        st.dataframe(pd.DataFrame(rr), hide_index=True, use_container_width=True)
    elif agg_suffix == "mean":
        st.caption("Distance-band table is hidden for frontend mean because a mean-of-candidates grid has no single serving-cell distance.")

    maps = st.columns(2)
    with maps[0]:
        st.caption(f"4G equal-power {agg_label} map")
        _render_map(p39_4g, equal_col, f"4G Phase 39 equal-power {agg_label}", view_mode)
    with maps[1]:
        st.caption(f"5G equal-power {agg_label} map")
        _render_map(p39_5g, equal_col, f"5G Phase 39 equal-power {agg_label}", view_mode)

    comp = go.Figure()
    comp.add_trace(_cdf_trace(p39_4g.loc[p39_4g["serving_environment"] == "outdoor", equal_col],
                              f"4G equal-power outdoor polygon ({agg_label})", "#2563eb"))
    comp.add_trace(_cdf_trace(p39_5g.loc[p39_5g["serving_environment"] == "outdoor", equal_col],
                              f"5G equal-power outdoor polygon ({agg_label})", "#dc2626"))
    comp.add_trace(_cdf_trace(p39_4g.loc[p39_4g["serving_environment"] == "indoor", equal_col],
                              f"4G equal-power indoor polygon ({agg_label})", "#60a5fa"))
    comp.add_trace(_cdf_trace(p39_5g.loc[p39_5g["serving_environment"] == "indoor", equal_col],
                              f"5G equal-power indoor polygon ({agg_label})", "#f97316"))
    comp.update_layout(title=f"Phase 39 equal-power 4G vs 5G polygon comparison ({agg_label})", height=460,
                       xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                       yaxis_range=[0, 100], xaxis_range=[-140, -45],
                       legend=dict(orientation="h", yanchor="bottom", y=-0.35))
    st.plotly_chart(comp, use_container_width=True)

    p39vt = load_phase39_validation(tech)
    if not p39vt.empty and "serving_environment" in p39serv.columns:
        clean39 = p39vt[(p39vt["obstruction_branch"].astype(str) != "indoor")
                        & (~p39vt.get("p36_backlobe", False))
                        & (~p39vt.get("p38_excluded", False))]
        f39c = go.Figure()
        f39c.add_trace(_cdf_trace(clean39["rsrp_measured"], "1 - DT measured (outdoor)", "#e5e7eb"))
        f39c.add_trace(_cdf_trace(clean39["phase39_final_rsrp"], "2 - Phase 39 calibrated predicted at DT", "#3b82f6"))
        f39c.add_trace(_cdf_trace(p39serv.loc[p39serv["serving_environment"] == "outdoor", final_col],
                                  f"3 - Phase 39 calibrated outdoor polygon ({agg_label})", "#22c55e"))
        f39c.add_trace(_cdf_trace(p39serv.loc[p39serv["serving_environment"] == "indoor", final_col],
                                  f"4 - Phase 39 calibrated indoor polygon ({agg_label})", "#f59e0b"))
        f39c.update_layout(title=f"{tech} Phase 39 calibrated DT accuracy reference ({agg_label})", height=460,
                           xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                           yaxis_range=[0, 100], xaxis_range=[-140, -45],
                           legend=dict(orientation="h", yanchor="bottom", y=-0.35))
        st.plotly_chart(f39c, use_container_width=True)
    with st.expander("phase39_summary.json"):
        st.json(p39s)


def _render_phase28(view_mode: str, tech: str, aggregation: str) -> None:
    st.header(f"Phase 28: {tech} RSRP Reference Fix — UNDER REVIEW, test only")
    serving = load_phase28_serving(tech)
    dtc = load_phase28_dt_check(tech)
    summary = load_phase28_summary()
    if serving.empty or not summary:
        st.error(f"Phase 28 {tech} output not found under {PHASE28_DIR}.")
        return

    pipe = summary.get("technology", {}).get(tech, {}) or summary.get("pipeline", {})
    ho_f = pipe.get("held_out_dt_final", {}) or {}
    ho_p = pipe.get("held_out_dt_physical_no_residual", {}) or {}
    sg = pipe.get("serving_grid", {}) or {}
    ref_method = summary.get("reference_method", {}).get(tech, pipe.get("reference_method", ""))
    ref_off = pipe.get("reference_offset_db", {})
    ref_off_txt = ", ".join(f"{k}: {v:+.1f}" for k, v in ref_off.items()) if ref_off else "?"
    if tech == "4G":
        st.caption(
            "Adds the missing LTE RSRP definition term  −10·log10(12·N_RB)  (~−28 dB) to the 4G raw model — "
            "the DB has no 4G power/bandwidth, so 46 dBm is an assumed total-carrier value and 775/1840 MHz "
            "bandwidths are assumed (10 / 5 MHz). Outdoor building loss dropped (COST231 urban already contains it), "
            "terrain kept, indoor O2I = frequency + depth, then a light per-clutter residual. "
            "Phase 27's dynamic calibration is NOT applied here."
        )
    else:
        st.caption(
            "5G has real per-cell power in the DB (already near an SSB/per-RE reference), so the raw is only "
            f"~10 dB hot. The reference offset ({ref_off_txt} dB) is taken **directly from the clean/LOS DT "
            "residual** per band (data-anchored, not the 4G N_RB formula). Same geo recipe as 4G "
            "(building loss dropped, terrain kept, indoor O2I, light per-clutter residual, Water override). "
            "5G NLOS diffraction over-attenuation is a known model-level issue and is not fixed here — MAE stays high."
        )

    c = st.columns(5)
    c[0].metric("Held-out DT MAE (final)", f"{float(ho_f.get('mae', np.nan)):.2f} dB",
                delta=f"{float(ho_f.get('mae', np.nan)) - float(ho_p.get('mae', np.nan)):+.2f} vs physical", delta_color="inverse")
    c[1].metric("Held-out DT bias", f"{float(ho_f.get('bias', np.nan)):+.2f} dB")
    c[2].metric("Held-out p90 |err|", f"{float(ho_f.get('p90_abs', np.nan)):.1f} dB")
    c[3].metric("Serving median RSRP", f"{float(sg.get('median_rsrp', np.nan)):.1f} dBm")
    c[4].metric("No coverage", f"{int(sg.get('no_coverage_rows', 0)):,}")

    # ---- The verification: raw residual by branch BEFORE vs AFTER the reference fix ----
    ver = pipe.get("verification", {}) or summary.get("verification", {})
    before = ver.get("residual_vs_RAW_before_fix", {})
    after = ver.get("residual_vs_RAW_after_fix", {})
    rows = []
    for br in ("clear", "obstructed", "indoor"):
        b = before.get(br, {}); a = after.get(br, {})
        rows.append({
            "branch": br, "n": a.get("n"),
            "median residual — before fix": round(b.get("median", np.nan), 1),
            "median residual — after fix": round(a.get("median", np.nan), 1),
            "MAE after": round(a.get("mae", np.nan), 1),
        })
    st.subheader(f"Raw COST231 vs DT — before and after the reference fix ({ref_off_txt} dB)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        f"Before the fix the {tech} raw sits a near-constant offset above DT on every branch (a constant "
        "offset = a reference bug, not a propagation error). After the reference offset the clean/LOS branch "
        "matches DT to ~0 dB median with no calibration. "
        + ("(4G: offset from the RSRP-per-RE definition.)" if tech == "4G"
           else "(5G: offset anchored on the clean-DT residual; obstructed stays off because 5G NLOS "
                "diffraction is over-attenuated at model level.)")
    )

    agg_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    phys_col = f"phase28_physical_{agg_suffix}_rsrp"
    final_col = f"phase28_final_{agg_suffix}_rsrp"
    st.caption(f"Aggregation: **{aggregation}** — showing `{final_col}`. (Toggle in the sidebar.)")

    # ---- maps: physical vs final ----
    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption(f"Phase 28 physical ({agg_suffix}) — raw + reference fix − terrain − indoor O2I")
        _render_map(serving, phys_col, f"{tech} Phase 28 physical", view_mode)
    with map_cols[1]:
        st.caption(f"Phase 28 final ({agg_suffix}) — + per-clutter residual")
        _render_map(serving, final_col, f"{tech} Phase 28 final", view_mode)

    # ---- 4-curve view (same as Phase 27): DT accuracy vs whole-polygon prediction ----
    dts = load_phase28_dt_scored(tech)
    if not dts.empty and "serving_environment" in serving.columns:
        out_dt = dts[dts["obstruction_branch"].astype(str) != "indoor"]
        four = go.Figure()
        four.add_trace(_cdf_trace(out_dt["rsrp_measured"], "1 - DT measured (outdoor, at DT points)", "#e5e7eb"))
        four.add_trace(_cdf_trace(out_dt["phase28_final_rsrp"], "2 - Predicted at those DT points", "#3b82f6"))
        four.add_trace(_cdf_trace(serving.loc[serving["serving_environment"] == "outdoor", final_col],
                                  f"3 - Predicted - outdoor (whole polygon, {agg_suffix})", "#22c55e"))
        four.add_trace(_cdf_trace(serving.loc[serving["serving_environment"] == "indoor", final_col],
                                  f"4 - Predicted - indoor (whole polygon, {agg_suffix}, one O2I term, no DT)", "#f59e0b"))
        four.update_layout(title=f"{tech} Phase 28: DT accuracy vs whole-polygon prediction", height=470,
                           xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                           yaxis_range=[0, 100], xaxis_range=[-140, -45],
                           legend=dict(orientation="h", yanchor="bottom", y=-0.35))
        st.plotly_chart(four, use_container_width=True)
        st.caption(
            "Same 4-curve view as Phase 27. 1 vs 2: accuracy where measured. 2 vs 3: DT-route vs whole "
            "outdoor polygon. 3 vs 4: outdoor vs indoor (geographic populations, not a penetration-loss reading)."
        )

    # ---- CDF: Phase 28 vs Phase 27 serving ----
    p27 = load_phase27_serving(tech)
    p27_col = "phase27_dynamic_mean_rsrp" if agg_suffix == "mean" else "phase27_dynamic_best_rsrp"
    fig = go.Figure()
    fig.add_trace(_cdf_trace(serving[phys_col], f"Phase 28 physical ({agg_suffix})", "#6b7280"))
    fig.add_trace(_cdf_trace(serving[final_col], f"Phase 28 final ({agg_suffix})", "#16a34a"))
    if not p27.empty and p27_col in p27.columns:
        fig.add_trace(_cdf_trace(p27[p27_col], f"Phase 27 {agg_suffix} (for comparison)", "#2563eb"))
    fig.update_layout(title=f"{tech} full-polygon serving CDF — Phase 28 vs Phase 27", height=430,
                      xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                      yaxis_range=[0, 100], xaxis_range=[-140, -45])
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Phase 28 serving median {float(sg.get('median_rsrp', np.nan)):.1f} dBm vs Phase 27's ~ -86 dBm. "
        "Phase 28 is weaker / not 'all green' because the raw is now at the correct RSRP level and the big "
        "generic geo/calibration inflation is gone — this is the honest physical surface, before the dynamic "
        "calibration layer that Phase 27 has."
    )

    # ---- DT reference check: measured vs raw before/after ----
    if not dtc.empty:
        m = pd.to_numeric(dtc["rsrp_measured"], errors="coerce")
        rb = pd.to_numeric(dtc["raw_cost231_at_dt_rsrp_unclipped"], errors="coerce")
        ra = pd.to_numeric(dtc["raw_after"], errors="coerce")
        f2 = go.Figure()
        f2.add_trace(_cdf_trace(m, "DT measured", "#e5e7eb"))
        f2.add_trace(_cdf_trace(rb, "Raw COST231 — before fix", "#ef4444"))
        f2.add_trace(_cdf_trace(ra, f"Raw COST231 — after reference fix ({ref_off_txt} dB)", "#16a34a"))
        f2.update_layout(title=f"{tech} DT points: measured vs raw COST231, before/after the reference fix",
                         height=420, xaxis_title="RSRP (dBm)", yaxis_title="Cumulative %",
                         yaxis_range=[0, 100], xaxis_range=[-140, -40])
        st.plotly_chart(f2, use_container_width=True)

    with st.expander("phase28_summary.json"):
        st.json(summary)


def _render_sector_footprint(view_mode: str, tech: str) -> None:
    st.header("Single sector: generic antenna (Phase 27) vs real pattern (Phase 29)")
    df = load_sector_candidates()
    if df.empty:
        st.error("Sector candidate data not found (needs Phase 27 and Phase 29 scored candidates).")
        return
    df = df[df["technology"].astype(str) == tech].copy()
    if df.empty:
        st.warning(f"No {tech} sector candidates.")
        return

    # polygon centre + pick the sector whose site is nearest to it
    cy, cx = df["lat"].mean(), df["lon"].mean()
    sites = df.groupby("strict_cell_key").agg(
        site=("site", "first"), sector=("sector", "first"), band=("band", "first"),
        site_lat=("site_lat", "first"), site_lon=("site_lon", "first"),
        azimuth=("azimuth_x", "first"), n=("grid_id", "size"),
    ).reset_index()
    sites["dist_to_centre_m"] = 111320.0 * np.sqrt(
        (sites["site_lat"] - cy) ** 2 + ((sites["site_lon"] - cx) * np.cos(np.radians(cy))) ** 2
    )
    sites = sites.sort_values("dist_to_centre_m")
    sites["label"] = (sites["site"].astype(str) + " / sec " + sites["sector"].astype(str)
                      + " / " + sites["band"].astype(str) + "  (az " + sites["azimuth"].round(0).astype("Int64").astype(str)
                      + " deg, " + sites["dist_to_centre_m"].round(0).astype(int).astype(str) + " m from centre)")
    labels = sites["label"].tolist()
    choice = st.selectbox("Sector (default = nearest the polygon centre)", labels, index=0, key="sector_footprint_pick")
    key = sites.loc[sites["label"] == choice, "strict_cell_key"].iloc[0]
    az0 = float(sites.loc[sites["strict_cell_key"] == key, "azimuth"].iloc[0])

    s = df[df["strict_cell_key"] == key].copy()
    st.caption(
        f"Sector **{choice}**. Boresight azimuth **{az0:.0f} deg**. "
        f"{len(s)} grid cells are candidates for this sector. "
        "If the real pattern is working, Phase 29 should fall off away from the boresight while "
        "Phase 27 (generic 65 deg cone) stays flatter / more omni-like."
    )

    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption("Phase 27 — generic 3GPP antenna (18 / 65 / 6)")
        _render_map(s, "phase27_dynamic_rsrp", f"{tech} sector — generic", view_mode)
    with map_cols[1]:
        st.caption("Phase 29 — real per-tilt pattern")
        _render_map(s, "phase29_dynamic_rsrp", f"{tech} sector — real antenna", view_mode)

    # azimuth cut: mean predicted RSRP vs offset from boresight
    s["az_off"] = pd.to_numeric(s["azimuth_delta_deg"], errors="coerce").abs()
    s["az_bin"] = (s["az_off"] // 10 * 10).clip(upper=170)
    cut = s.groupby("az_bin").agg(
        p27=("phase27_dynamic_rsrp", "mean"), p29=("phase29_dynamic_rsrp", "mean"), n=("grid_id", "size")
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cut["az_bin"], y=cut["p27"], mode="lines+markers", name="Phase 27 generic", line=dict(color="#2563eb", width=2.5)))
    fig.add_trace(go.Scatter(x=cut["az_bin"], y=cut["p29"], mode="lines+markers", name="Phase 29 real antenna", line=dict(color="#16a34a", width=2.5)))
    fig.update_layout(
        title=f"{tech} sector: mean predicted RSRP vs azimuth offset from boresight",
        height=380, xaxis_title="azimuth offset from sector boresight (deg)", yaxis_title="mean predicted RSRP (dBm)",
    )
    st.plotly_chart(fig, use_container_width=True)

    front = s[s["az_off"] <= 60]
    back = s[s["az_off"] >= 120]
    m = st.columns(4)
    m[0].metric("Front (±60°) mean — generic", f"{front['phase27_dynamic_rsrp'].mean():.1f} dBm")
    m[1].metric("Front (±60°) mean — real", f"{front['phase29_dynamic_rsrp'].mean():.1f} dBm")
    m[2].metric("Back (>120°) mean — generic", f"{back['phase27_dynamic_rsrp'].mean():.1f} dBm")
    m[3].metric("Back (>120°) mean — real", f"{back['phase29_dynamic_rsrp'].mean():.1f} dBm",
                delta=f"{back['phase29_dynamic_rsrp'].mean() - back['phase27_dynamic_rsrp'].mean():+.1f} dB vs generic",
                delta_color="off")
    st.caption(
        "Front-to-back: generic "
        f"{front['phase27_dynamic_rsrp'].mean() - back['phase27_dynamic_rsrp'].mean():.1f} dB, "
        f"real antenna {front['phase29_dynamic_rsrp'].mean() - back['phase29_dynamic_rsrp'].mean():.1f} dB. "
        "A larger front-to-back on Phase 29 = the real pattern is suppressing the back lobe (not omni)."
    )


def _render_phase24_25_comparison(view_mode: str, tech: str, aggregation: str) -> None:
    st.header("Comparison: Phase 24 vs Phase 25")
    serving = load_phase25_serving(tech)
    validation = load_phase25_validation_dt()
    summary = load_phase25_summary().get("technology", {}).get(tech, {})
    if serving.empty:
        st.error(f"Phase 25 output not found under {PHASE25_DIR}.")
        return

    value_suffix = "mean" if aggregation.startswith("Frontend") else "best"
    phase24_col = f"phase24_no_lock_{value_suffix}_rsrp"
    phase25_col = f"phase25_dynamic_{value_suffix}_rsrp"
    serving["phase25_vs_phase24_delta_db"] = serving[phase25_col] - serving[phase24_col]

    phase24_metrics = summary.get("phase24_no_lock_validation", {})
    phase25_metrics = summary.get("phase25_dynamic_validation", {})
    cols = st.columns(5)
    cols[0].metric("Validation rows", f"{int(summary.get('validation_dt_rows', 0)):,}")
    cols[1].metric("Phase24 MAE", f"{float(phase24_metrics.get('mae', np.nan)):.2f} dB")
    cols[2].metric("Phase25 MAE", f"{float(phase25_metrics.get('mae', np.nan)):.2f} dB")
    cols[3].metric("MAE gain", f"{float(phase24_metrics.get('mae', np.nan)) - float(phase25_metrics.get('mae', np.nan)):.2f} dB")
    cols[4].metric("Mean correction", f"{float(summary.get('mean_total_dynamic_correction_db', np.nan)):.2f} dB")

    st.subheader(f"{tech} side-by-side map comparison")
    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption("Phase 24 no-lock reference")
        _render_map(serving, phase24_col, f"{tech} Phase 24 no-lock", view_mode)
    with map_cols[1]:
        st.caption("Phase 25 dynamic calibrated")
        _render_map(serving, phase25_col, f"{tech} Phase 25 dynamic", view_mode)

    diag_cols = st.columns(2)
    with diag_cols[0]:
        st.caption("Dynamic correction delta")
        _render_map(serving, "phase25_vs_phase24_delta_db", f"{tech} Phase25 - Phase24", view_mode, value_kind="delta")
    with diag_cols[1]:
        st.caption("Confidence")
        _render_map(serving, "phase25_confidence_mean", f"{tech} Phase25 confidence", view_mode, value_kind="confidence")

    fig = go.Figure()
    fig.add_trace(_cdf_trace(serving[phase24_col], "Phase24 no-lock reference", "#2563eb"))
    fig.add_trace(_cdf_trace(serving[phase25_col], "Phase25 dynamic calibrated", "#16a34a"))
    fig.update_layout(
        title=f"{tech} full-polygon CDF: Phase 24 vs Phase 25 - {aggregation}",
        height=430,
        xaxis_title="RSRP (dBm)",
        yaxis_title="Cumulative %",
        yaxis_range=[0, 100],
        xaxis_range=[-147, -45],
    )
    st.plotly_chart(fig, use_container_width=True)

    vtech = validation[validation["technology"].astype(str) == tech].copy() if not validation.empty else pd.DataFrame()
    if not vtech.empty:
        err_fig = go.Figure()
        err_fig.add_trace(_cdf_trace((vtech["rsrp_measured"] - vtech["phase24_no_lock_reference_rsrp"]).abs(), "Phase24 no-lock abs error", "#2563eb"))
        err_fig.add_trace(_cdf_trace((vtech["rsrp_measured"] - vtech["phase25_dynamic_rsrp"]).abs(), "Phase25 dynamic abs error", "#16a34a"))
        err_fig.update_layout(
            title=f"{tech} held-out DT absolute error CDF",
            height=430,
            xaxis_title="Absolute error (dB)",
            yaxis_title="Cumulative %",
            yaxis_range=[0, 100],
        )
        st.plotly_chart(err_fig, use_container_width=True)


def _render_phase37_quality(tech: str) -> None:
    st.header("Phase 37 - Dynamic RSRQ / SINR validation")
    summary = load_phase37_summary()
    serving = load_phase37_serving()
    if not summary or serving.empty:
        st.error(f"Phase 37 output not found under {PHASE37_DIR}. Run test_project210_phase37_quality_readiness.py.")
        return

    _cut = summary.get("interferer_cutoff_db", {})
    _cut_txt = ", ".join(f"{k} {v:.0f} dB" for k, v in _cut.items()) if _cut else "DT-fitted"
    st.caption(f"Phase 36 v2 re-banded real-antenna RSRP drives serving signal; the co-channel interferer window is "
               f"fit per technology from the drive test ({_cut_txt}) - no hand-set constant. RSRQ/SINR use a -104 dBm "
               f"noise floor, a hierarchical DT residual (carrier -> technology -> global) and a Phase-25-style local "
               f"inverse-distance DT-residual field - the same recipe as the RSRP v2 surface - so every served grid gets "
               f"a spatially-varying value. Corrections are learned from training DT only; CDF DT curves use the held-out "
               f"validation split.")

    cdf_summary = summary.get("quality_cdf", {})
    if cdf_summary:
        cov_cols = st.columns(2)
        for (cdf_tech, entry), holder in zip(sorted(cdf_summary.items()), cov_cols):
            conf = entry.get("quality_confidence", {})
            conf_txt = ", ".join(f"{k} {v:,}" for k, v in conf.items()) or "n/a"
            holder.markdown(
                f"**{cdf_tech}**  \n"
                f"RSRQ coverage {entry.get('grid_pred_rsrq_coverage', 0)*100:.1f}% · "
                f"SINR coverage {entry.get('grid_pred_sinr_coverage', 0)*100:.1f}%  \n"
                f"held-out RSRQ MAE **{entry.get('rsrq_validation_mae_db')}** dB · "
                f"SINR MAE **{entry.get('sinr_validation_mae_db')}** dB "
                f"(n={entry.get('validation_dt_rows', 0):,})  \n"
                f"confidence: {conf_txt}"
            )

    status_rows = summary.get("carrier_summary", [])
    if status_rows:
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.subheader("Predicted RSRQ / SINR distribution across the polygon")
    st.caption("Whole-polygon predicted quality per grid cell (every served cell has a value). "
               "Same grid geometry as the Phase 39 serving-cell map; sector triangles show the tech shown.")
    site_overlay = st.session_state.get("phase_site_overlay", "Selected technology")
    map_specs = [
        ("4G", "pred_rsrq_db", "rsrq", "4G Phase 37 predicted RSRQ (dB)"),
        ("5G", "pred_rsrq_db", "rsrq", "5G Phase 37 predicted RSRQ (dB)"),
        ("4G", "pred_sinr_db", "sinr", "4G Phase 37 predicted SINR (dB)"),
        ("5G", "pred_sinr_db", "sinr", "5G Phase 37 predicted SINR (dB)"),
    ]
    for row_specs in (map_specs[:2], map_specs[2:]):
        map_columns = st.columns(2)
        for (map_tech, value_col, value_kind, title), holder in zip(row_specs, map_columns):
            frame = serving[serving["technology"].astype(str).eq(map_tech)].copy()
            if frame.empty or value_col not in frame.columns:
                holder.warning(f"No {title} data.")
                continue
            holder.image(
                _build_static_image_png(frame, value_col, title, value_kind, map_tech, site_overlay),
                width=STATIC_MAP_DISPLAY_WIDTH_PX,
            )

    chart_specs = [
        ("4G", "rsrq_measured", "pred_rsrq_db", "RSRQ"),
        ("4G", "sinr_measured", "pred_sinr_db", "SINR"),
        ("5G", "rsrq_measured", "pred_rsrq_db", "RSRQ"),
        ("5G", "sinr_measured", "pred_sinr_db", "SINR"),
    ]
    for row_specs in (chart_specs[:2], chart_specs[2:]):
        chart_columns = st.columns(2)
        for (chart_tech, measured, predicted, metric), holder in zip(row_specs, chart_columns):
            dt = load_phase37_dt_quality(chart_tech)
            validation = dt[dt.get("split", pd.Series("", index=dt.index)).astype(str).eq("validation")].copy()
            grid = serving[serving["technology"].astype(str).eq(chart_tech)].copy()
            fig = go.Figure()
            fig.add_trace(_cdf_trace(validation.get(measured, pd.Series(dtype=float)), "DT measured validation", "#f8fafc"))
            fig.add_trace(_cdf_trace(validation.get(predicted, pd.Series(dtype=float)), "Phase 37 validation prediction", "#22d3ee"))
            environment = grid.get("serving_environment", pd.Series("UNKNOWN", index=grid.index)).astype(str)
            fig.add_trace(_cdf_trace(grid.loc[~environment.eq("indoor"), predicted], "Whole polygon outdoor", "#22c55e"))
            fig.add_trace(_cdf_trace(grid.loc[environment.eq("indoor"), predicted], "Whole polygon indoor", "#f59e0b"))
            fig.update_layout(
                title=f"{chart_tech} {metric}: DT accuracy vs whole-polygon prediction",
                height=430,
                xaxis_title=f"{metric} (dB)",
                yaxis_title="Cumulative %",
                yaxis_range=[0, 100],
                legend=dict(orientation="h", y=-0.24, font=dict(color="#f8fafc")),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#f8fafc"),
            )
            fig.update_xaxes(gridcolor="#374151", zerolinecolor="#6b7280")
            fig.update_yaxes(gridcolor="#374151", zerolinecolor="#6b7280")
            holder.plotly_chart(fig, use_container_width=True)

    work = serving[serving["technology"].astype(str) == tech].copy()
    if work.empty:
        return
    cols = st.columns(4)
    cols[0].metric(f"{tech} grid rows", f"{len(work):,}")
    cols[1].metric("Median co-channel sectors", f"{work['eligible_cochannel_sector_count'].median():.0f}")
    cols[2].metric("RSRQ median", f"{pd.to_numeric(work['pred_rsrq_db'], errors='coerce').median():.1f} dB")
    cols[3].metric("SINR median", f"{pd.to_numeric(work['pred_sinr_db'], errors='coerce').median():.1f} dB")

    dt = load_phase37_dt_quality(tech)
    dt_show_cols = [
        "dt_row_id", "carrier_key", "phase37_serving_rsrp_dbm", "rsrq_measured", "pred_rsrq_db",
        "sinr_measured", "pred_sinr_db", "interference_sum_dbm", "sinr_base_db", "rsrq_base_db",
        "eligible_cochannel_sector_count", "quality_confidence", "quality_status", "interference_reference",
    ]
    st.dataframe(dt[[column for column in dt_show_cols if column in dt.columns]], use_container_width=True, height=360, hide_index=True)


def _render_phase40_quality(tech: str) -> None:
    st.header("Phase 40 - Fixed-power RSRQ / SINR validation")
    summary = load_phase40_summary()
    serving = load_phase40_serving()
    if not summary or serving.empty:
        st.error(f"Phase 40 output not found under {PHASE40_DIR}. Run test_project210_phase40_fixed_power_quality.py.")
        return

    st.caption(
        "Phase 40 freezes the Phase 39 fixed-power, frequency-aware RSRP baseline. "
        "Grid serving is the strongest Phase 39 candidate; DT validation uses its matched serving sector. "
        "All same-carrier sectors are evaluated for interference, with no nearest-sector/top-N cap. "
        "The active-interferer cutoff is fitted per technology from training DT using the Phase 37 method, then quality is DT-calibrated. "
        "No KNN/local quality field and no display smoothing are used."
    )
    rows = summary.get("carrier_summary", [])
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Whole-polygon quality maps")
    site_overlay = st.session_state.get("phase_site_overlay", "Selected technology")
    map_specs = [
        ("4G", "pred_rsrq_db", "rsrq", "4G Phase 40 predicted RSRQ (dB)"),
        ("5G", "pred_rsrq_db", "rsrq", "5G Phase 40 predicted RSRQ (dB)"),
        ("4G", "pred_sinr_db", "sinr", "4G Phase 40 predicted SINR (dB)"),
        ("5G", "pred_sinr_db", "sinr", "5G Phase 40 predicted SINR (dB)"),
    ]
    for row_specs in (map_specs[:2], map_specs[2:]):
        holders = st.columns(2)
        for (map_tech, value_col, value_kind, title), holder in zip(row_specs, holders):
            frame = serving[serving["technology"].astype(str).eq(map_tech)].copy()
            if frame.empty:
                holder.warning(f"No {title} data.")
                continue
            holder.image(
                _build_static_image_png(frame, value_col, title, value_kind, map_tech, site_overlay),
                width=STATIC_MAP_DISPLAY_WIDTH_PX,
            )

    st.subheader("Held-out DT accuracy versus whole-polygon distribution")
    chart_specs = [
        ("4G", "rsrq_measured", "pred_rsrq_db", "RSRQ"),
        ("4G", "sinr_measured", "pred_sinr_db", "SINR"),
        ("5G", "rsrq_measured", "pred_rsrq_db", "RSRQ"),
        ("5G", "sinr_measured", "pred_sinr_db", "SINR"),
    ]
    for row_specs in (chart_specs[:2], chart_specs[2:]):
        holders = st.columns(2)
        for (chart_tech, measured, predicted, metric), holder in zip(row_specs, holders):
            dt = load_phase40_dt_quality(chart_tech)
            validation = dt[dt.get("split", pd.Series("", index=dt.index)).astype(str).eq("validation")].copy()
            grid = serving[serving["technology"].astype(str).eq(chart_tech)].copy()
            environment = grid.get("serving_environment", pd.Series("UNKNOWN", index=grid.index)).astype(str)
            figure = go.Figure()
            figure.add_trace(_cdf_trace(validation.get(measured, pd.Series(dtype=float)), "DT measured validation", "#f8fafc"))
            figure.add_trace(_cdf_trace(validation.get(predicted, pd.Series(dtype=float)), "Phase 40 validation prediction", "#22d3ee"))
            figure.add_trace(_cdf_trace(grid.loc[~environment.eq("indoor"), predicted], "Whole polygon outdoor", "#22c55e"))
            figure.add_trace(_cdf_trace(grid.loc[environment.eq("indoor"), predicted], "Whole polygon indoor", "#f59e0b"))
            figure.update_layout(
                title=f"{chart_tech} {metric}: DT accuracy vs whole-polygon prediction",
                height=430,
                xaxis_title=f"{metric} (dB)",
                yaxis_title="Cumulative %",
                yaxis_range=[0, 100],
                legend=dict(orientation="h", y=-0.24, font=dict(color="#f8fafc")),
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#f8fafc"),
            )
            figure.update_xaxes(gridcolor="#374151", zerolinecolor="#6b7280")
            figure.update_yaxes(gridcolor="#374151", zerolinecolor="#6b7280")
            holder.plotly_chart(figure, use_container_width=True)

    work = serving[serving["technology"].astype(str).eq(tech)].copy()
    if work.empty:
        return
    metrics = st.columns(4)
    metrics[0].metric(f"{tech} grid rows", f"{len(work):,}")
    metrics[1].metric("Median co-channel sectors", f"{work['eligible_cochannel_sector_count'].median():.0f}")
    metrics[2].metric("RSRQ median", f"{pd.to_numeric(work['pred_rsrq_db'], errors='coerce').median():.1f} dB")
    metrics[3].metric("SINR median", f"{pd.to_numeric(work['pred_sinr_db'], errors='coerce').median():.1f} dB")
    dt = load_phase40_dt_quality(tech)
    display_columns = [
        "dt_row_id", "carrier_key", "phase40_serving_rsrp_dbm", "rsrq_measured", "pred_rsrq_db",
        "sinr_measured", "pred_sinr_db", "interference_sum_dbm", "sinr_base_db", "rsrq_base_db",
        "eligible_cochannel_sector_count", "interfering_sector_count", "activity_factor", "quality_status",
    ]
    st.dataframe(dt[[column for column in display_columns if column in dt.columns]], use_container_width=True, height=360, hide_index=True)


def _render_phase41_coverage_footprint(view_mode: str, tech: str) -> None:
    st.header("Phase 41 - Sector serving coverage")
    summary = load_phase41_summary()
    serving = load_phase41_serving()
    sector_summary = load_phase41_sector_summary()
    if not summary or serving.empty:
        st.error(
            f"Phase 41 output not found under {PHASE41_DIR}. "
            "Run test_project210_phase41_pap_coverage_footprint.py."
        )
        return

    st.caption(
        "Phase 41 keeps the full RF prediction from Phase 39, where the antenna pattern has already affected signal "
        "strength. Sector coverage is not a visual lobe mask: it is the grid area where the selected sector has usable "
        "signal and is the strongest serving cell for that technology."
    )

    tech_summary = summary.get("technology", {})
    if tech_summary:
        st.dataframe(pd.DataFrame.from_dict(tech_summary, orient="index").reset_index(names="technology"),
                     use_container_width=True, hide_index=True)

    st.subheader("4G / 5G production capped maps")
    site_overlay = st.session_state.get("phase_site_overlay", "Selected technology")
    holders = st.columns(2)
    for map_tech, holder in zip(("4G", "5G"), holders):
        frame = serving[serving["technology"].astype(str).eq(map_tech)].copy()
        if frame.empty:
            holder.warning(f"No Phase 41 {map_tech} serving map.")
            continue
        if view_mode == "Static image":
            holder.image(
                _build_static_image_png(
                    frame,
                    "phase41_production_rsrp",
                    f"{map_tech} Phase 41 production capped/backfilled",
                    "rsrp",
                    map_tech,
                    site_overlay,
                ),
                width=STATIC_MAP_DISPLAY_WIDTH_PX,
            )
        else:
            with holder:
                components.html(
                    _build_grid_map_html(
                        frame,
                        "phase41_production_rsrp",
                        f"{map_tech} Phase 41 production capped/backfilled",
                        "rsrp",
                        map_tech,
                        site_overlay,
                    ),
                    height=620,
                    scrolling=False,
                )

    st.subheader("CDF: production surface vs serving coverage")
    cdf_cols = st.columns(2)
    for chart_tech, holder in zip(("4G", "5G"), cdf_cols):
        frame = serving[serving["technology"].astype(str).eq(chart_tech)].copy()
        fig = go.Figure()
        fig.add_trace(_cdf_trace(frame.get("phase41_production_rsrp", pd.Series(dtype=float)),
                                 "Production capped/backfilled", "#94a3b8"))
        fig.add_trace(_cdf_trace(frame.get("phase41_rsrp", pd.Series(dtype=float)),
                                 "Serving coverage", "#22c55e"))
        fig.update_layout(
            title=f"{chart_tech} RSRP CDF",
            height=410,
            xaxis_title="RSRP (dBm)",
            yaxis_title="Cumulative %",
            yaxis_range=[0, 100],
            legend=dict(orientation="h", y=-0.24, font=dict(color="#f8fafc")),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#f8fafc"),
        )
        fig.update_xaxes(gridcolor="#374151", zerolinecolor="#6b7280")
        fig.update_yaxes(gridcolor="#374151", zerolinecolor="#6b7280")
        holder.plotly_chart(fig, use_container_width=True)

    st.subheader("Sector / cell coverage")
    cells = load_phase41_cell_coverage(tech)
    if cells.empty or sector_summary.empty:
        st.warning(f"No Phase 41 {tech} cell coverage rows.")
        return
    sector_rows = sector_summary[sector_summary["technology"].astype(str).eq(tech)].copy()
    sector_rows = sector_rows.sort_values(["site", "sector", "band", "strict_cell_key"]).reset_index(drop=True)
    labels = [
        f"{row.strict_cell_key} | site={row.site} sector={row.sector} band={row.band} serving={row.serving_pct:.1f}%"
        for row in sector_rows.itertuples(index=False)
    ]
    if not labels:
        st.warning(f"No Phase 41 {tech} sector summary.")
        return
    selected_label = st.selectbox("Sector / cell", labels, index=0, key=f"phase41_sector_{tech}")
    selected_cell = str(sector_rows.iloc[labels.index(selected_label)]["strict_cell_key"])
    selected_rows = cells[cells["strict_cell_key"].astype(str).eq(selected_cell)].copy()
    metrics = st.columns(4)
    usable = pd.to_numeric(selected_rows.get("phase41_potential_rsrp"), errors="coerce").notna()
    serving = pd.to_numeric(selected_rows.get("phase41_serving_rsrp"), errors="coerce").notna()
    metrics[0].metric("Candidate rows", f"{len(selected_rows):,}")
    metrics[1].metric("Usable signal rows", f"{int(usable.sum()):,}")
    metrics[2].metric("Serving rows", f"{int(serving.sum()):,}")
    metrics[3].metric("Serving coverage", f"{float(serving.mean() * 100.0):.1f}%" if len(selected_rows) else "0.0%")

    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption("Potential RF from selected cell")
        _render_map(selected_rows, "phase41_potential_rsrp", f"{tech} selected-cell potential RF", view_mode)
    with map_cols[1]:
        st.caption("Serving coverage area")
        _render_map(selected_rows, "phase41_serving_rsrp", f"{tech} selected-cell serving coverage", view_mode)

    debug_cols = [
        "grid_id", "phase41_raw_rsrp_dbm", "phase41_potential_rsrp", "phase41_serving_rsrp",
        "phase41_hide_reason", "phase41_row_source", "phase41_cell_median_delta_db",
        "phase41_is_serving_cell", "azimuth_delta_deg", "phase36_antenna_delta_db",
        "distance_m",
    ]
    st.dataframe(
        selected_rows[[column for column in debug_cols if column in selected_rows.columns]]
        .sort_values(["phase41_hide_reason", "azimuth_delta_deg"], na_position="last"),
        use_container_width=True,
        height=340,
        hide_index=True,
    )


def _render_phase42_coverage_footprint(view_mode: str, tech: str) -> None:
    st.header("Phase 42 - 1500 m direct-PAP sector coverage")
    summary = load_phase42_summary()
    serving = load_phase42_serving()
    sector_summary = load_phase42_sector_summary()
    if not summary or serving.empty:
        st.error(
            f"Phase 42 output not found under {PHASE42_DIR}. "
            "Run test_project210_phase42_1500m_pap_sector_coverage.py."
        )
        return

    st.caption(
        "Phase 42 is test-only. It regenerates the Phase 9 candidate surface at 1500 m, then reuses Phase 26 "
        "terrain/building scoring and Phase 36/39 production-style scoring. PAP is applied at the Phase 36 "
        "production-equivalent replacement point: generic antenna in Phase 9 is replaced by real PAP-minus-generic gain."
    )

    tech_summary = summary.get("technology", {})
    if tech_summary:
        st.dataframe(
            pd.DataFrame.from_dict(tech_summary, orient="index").reset_index(names="technology"),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("4G / 5G production capped maps")
    site_overlay = st.session_state.get("phase_site_overlay", "Selected technology")
    holders = st.columns(2)
    for map_tech, holder in zip(("4G", "5G"), holders):
        frame = serving[serving["technology"].astype(str).eq(map_tech)].copy()
        if frame.empty:
            holder.warning(f"No Phase 42 {map_tech} serving map.")
            continue
        if view_mode == "Static image":
            holder.image(
                _build_static_image_png(
                    frame,
                    "phase42_production_rsrp",
                    f"{map_tech} Phase 42 1500 m direct-PAP capped/backfilled",
                    "rsrp",
                    map_tech,
                    site_overlay,
                ),
                width=STATIC_MAP_DISPLAY_WIDTH_PX,
            )
        else:
            with holder:
                components.html(
                    _build_grid_map_html(
                        frame,
                        "phase42_production_rsrp",
                        f"{map_tech} Phase 42 1500 m direct-PAP capped/backfilled",
                        "rsrp",
                        map_tech,
                        site_overlay,
                    ),
                    height=620,
                    scrolling=False,
                )

    st.subheader("CDF: production surface vs serving coverage")
    cdf_cols = st.columns(2)
    for chart_tech, holder in zip(("4G", "5G"), cdf_cols):
        frame = serving[serving["technology"].astype(str).eq(chart_tech)].copy()
        fig = go.Figure()
        fig.add_trace(_cdf_trace(frame.get("phase42_production_rsrp", pd.Series(dtype=float)),
                                 "Production capped/backfilled", "#94a3b8"))
        fig.add_trace(_cdf_trace(frame.get("phase42_rsrp", pd.Series(dtype=float)),
                                 "Serving coverage", "#22c55e"))
        fig.update_layout(
            title=f"{chart_tech} RSRP CDF",
            height=410,
            xaxis_title="RSRP (dBm)",
            yaxis_title="Cumulative %",
            yaxis_range=[0, 100],
            legend=dict(orientation="h", y=-0.24, font=dict(color="#f8fafc")),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#f8fafc"),
        )
        fig.update_xaxes(gridcolor="#374151", zerolinecolor="#6b7280")
        fig.update_yaxes(gridcolor="#374151", zerolinecolor="#6b7280")
        holder.plotly_chart(fig, use_container_width=True)

    st.subheader("Sector / cell coverage")
    cells = load_phase42_cell_coverage(tech)
    if cells.empty or sector_summary.empty:
        st.warning(f"No Phase 42 {tech} cell coverage rows.")
        return
    sector_rows = sector_summary[sector_summary["technology"].astype(str).eq(tech)].copy()
    sector_rows = sector_rows.sort_values(["site", "sector", "band", "strict_cell_key"]).reset_index(drop=True)
    labels = [
        f"{row.strict_cell_key} | site={row.site} sector={row.sector} band={row.band} serving={row.serving_pct:.1f}%"
        for row in sector_rows.itertuples(index=False)
    ]
    if not labels:
        st.warning(f"No Phase 42 {tech} sector summary.")
        return
    selected_label = st.selectbox("Sector / cell", labels, index=0, key=f"phase42_sector_{tech}")
    selected_cell = str(sector_rows.iloc[labels.index(selected_label)]["strict_cell_key"])
    selected_rows = cells[cells["strict_cell_key"].astype(str).eq(selected_cell)].copy()
    metrics = st.columns(5)
    usable = pd.to_numeric(selected_rows.get("phase42_potential_rsrp"), errors="coerce").notna()
    serving_mask = pd.to_numeric(selected_rows.get("phase42_serving_rsrp"), errors="coerce").notna()
    metrics[0].metric("Candidate rows", f"{len(selected_rows):,}")
    metrics[1].metric("Usable signal rows", f"{int(usable.sum()):,}")
    metrics[2].metric("Serving rows", f"{int(serving_mask.sum()):,}")
    metrics[3].metric("Full-grid components", str(sector_rows.iloc[labels.index(selected_label)].get("full_grid_components", "")))
    metrics[4].metric("Potential components", str(sector_rows.iloc[labels.index(selected_label)].get("potential_components", "")))

    map_cols = st.columns(2)
    with map_cols[0]:
        st.caption("Potential RF from selected cell")
        _render_map(selected_rows, "phase42_potential_rsrp", f"{tech} selected-cell direct-PAP potential RF", view_mode)
    with map_cols[1]:
        st.caption("Serving coverage area")
        _render_map(selected_rows, "phase42_serving_rsrp", f"{tech} selected-cell direct-PAP serving coverage", view_mode)

    debug_cols = [
        "grid_id", "phase42_raw_rsrp_dbm", "phase42_pap_raw_rsrp_unclipped",
        "phase42_potential_rsrp", "phase42_serving_rsrp", "phase42_hide_reason",
        "phase42_row_source", "phase42_pattern_source", "antenna_model",
        "phase42_pap_replacement_db", "phase42_cell_median_calibration_db", "phase42_is_serving_cell",
        "azimuth_delta_deg", "distance_m",
        "terrain_diffraction_loss_db", "building_geo_correction_db", "obstruction_branch", "clutter_class",
    ]
    st.dataframe(
        selected_rows[[column for column in debug_cols if column in selected_rows.columns]]
        .sort_values(["phase42_hide_reason", "azimuth_delta_deg"], na_position="last"),
        use_container_width=True,
        height=340,
        hide_index=True,
    )


def render() -> None:
    st.title("Project 210 Taiwan - Phase Validation")

    with st.sidebar:
        validation_view = st.selectbox(
            "Validation view",
            [
                "Comparison: Phase 24 vs Phase 25",
                "Phase 20 DT source",
                "Phase 21 corrected full polygon",
                "Phase 22 terrain",
                "Phase 23 outdoor calibration",
                "Phase 24 clutter role",
                "Phase 25 dynamic calibration",
                "Phase 26 corrected obstruction",
                "Phase 27 dynamic on corrected obstruction",
                "Phase 29 real antenna pattern",
                "Phase 28 RSRP reference fix (4G + 5G, under review)",
                "Sector footprint: generic vs real antenna",
                "Phase 31 real antenna on Phase 28 base",
                "Phase 32/33/34/35 5G path-loss & antenna experiments",
                "Phase 36 FINAL + v2 re-band + v38 EARFCN re-match",
                "Phase 39 equal-power diagnostic",
                "Phase 40 fixed-power RSRQ / SINR",
                "Phase 41 sector serving coverage",
                "Phase 42 1500m direct-PAP sector coverage",
                "Phase 37 RSRQ / SINR readiness",
                "All sections",
            ],
            index=0,
            key="phase_validation_view",
        )
        view_mode = st.radio(
            "Map view",
            ["Interactive (folium)", "Static image"],
            index=1,
            key="phase20_21_22_view_mode",
        )
        tech = st.radio("Technology", ["4G", "5G"], index=1, horizontal=True, key="phase20_21_22_tech")
        st.radio(
            "Site sector overlay",
            ["Selected technology", "Both 4G and 5G", "Off"],
            index=0,
            key="phase_site_overlay",
        )
        aggregation = st.radio(
            "Phase 22/24/25 aggregation",
            ["Serving cell (best server)", "Frontend (mean of candidates)"],
            index=0,
            key="phase22_aggregation",
        )
        if validation_view == "Phase 27 dynamic on corrected obstruction":
            corrections = load_phase27_group_corrections()
            clutter = corrections[
                (corrections.get("layer", pd.Series(index=corrections.index, dtype=object)).astype(str) == "clutter_terrain")
                & (corrections.get("technology", pd.Series(index=corrections.index, dtype=object)).astype(str) == tech)
            ].copy()
            st.divider()
            st.caption("Phase 27 dynamic clutter/terrain residuals")
            st.caption("These are DT-calibrated residual corrections, not fixed physical clutter losses.")
            if clutter.empty:
                st.caption("No supported clutter/terrain groups for this technology.")
            else:
                show_cols = [
                    "band", "clutter_class", "obstruction_branch", "terrain_bucket",
                    "n_train", "clutter_terrain_shrink_factor", "clutter_terrain_correction_db",
                ]
                clutter = clutter[[col for col in show_cols if col in clutter.columns]].rename(columns={
                    "n_train": "DT n",
                    "clutter_terrain_shrink_factor": "shrink",
                    "clutter_terrain_correction_db": "dynamic correction dB",
                })
                st.dataframe(
                    clutter.sort_values(["band", "clutter_class", "obstruction_branch", "terrain_bucket"]),
                    use_container_width=True,
                    height=300,
                    hide_index=True,
                )

    dt = load_phase20_dt()
    if validation_view == "Comparison: Phase 24 vs Phase 25":
        _render_phase24_25_comparison(view_mode, tech, aggregation)
    elif validation_view == "Phase 20 DT source":
        _render_phase20(dt)
    elif validation_view == "Phase 21 corrected full polygon":
        _render_phase21(view_mode, tech)
    elif validation_view == "Phase 22 terrain":
        _render_phase22(view_mode, tech, aggregation)
    elif validation_view == "Phase 23 outdoor calibration":
        _render_phase23(view_mode, tech, aggregation)
    elif validation_view == "Phase 24 clutter role":
        _render_phase24(view_mode, tech, aggregation)
    elif validation_view == "Phase 25 dynamic calibration":
        _render_phase25(view_mode, tech, aggregation)
    elif validation_view == "Phase 26 corrected obstruction":
        _render_phase26(view_mode, tech, aggregation)
    elif validation_view == "Phase 27 dynamic on corrected obstruction":
        _render_phase27(view_mode, tech, aggregation)
    elif validation_view == "Phase 29 real antenna pattern":
        _render_phase29(view_mode, tech, aggregation)
    elif validation_view == "Phase 28 RSRP reference fix (4G + 5G, under review)":
        _render_phase28(view_mode, tech, aggregation)
    elif validation_view == "Sector footprint: generic vs real antenna":
        _render_sector_footprint(view_mode, tech)
    elif validation_view == "Phase 31 real antenna on Phase 28 base":
        _render_phase31(view_mode, tech, aggregation)
    elif validation_view == "Phase 32/33/34/35 5G path-loss & antenna experiments":
        _render_phase32_33_34(view_mode, tech)
    elif validation_view == "Phase 36 FINAL + v2 re-band + v38 EARFCN re-match":
        _render_phase36(view_mode, tech, aggregation)
    elif validation_view == "Phase 39 equal-power diagnostic":
        _render_phase39(view_mode, tech, aggregation)
    elif validation_view == "Phase 40 fixed-power RSRQ / SINR":
        _render_phase40_quality(tech)
    elif validation_view == "Phase 41 sector serving coverage":
        _render_phase41_coverage_footprint(view_mode, tech)
    elif validation_view == "Phase 42 1500m direct-PAP sector coverage":
        _render_phase42_coverage_footprint(view_mode, tech)
    elif validation_view == "Phase 37 RSRQ / SINR readiness":
        _render_phase37_quality(tech)
    else:
        _render_phase20(dt)
        _render_phase21(view_mode, tech)
        _render_phase22(view_mode, tech, aggregation)
        _render_phase23(view_mode, tech, aggregation)
        _render_phase24(view_mode, tech, aggregation)
        _render_phase25(view_mode, tech, aggregation)
        _render_phase26(view_mode, tech, aggregation)
        _render_phase27(view_mode, tech, aggregation)
        _render_phase29(view_mode, tech, aggregation)
        _render_phase28(view_mode, tech, aggregation)
        _render_phase31(view_mode, tech, aggregation)
        _render_phase32_33_34(view_mode, tech)
        _render_phase36(view_mode, tech, aggregation)
        _render_phase39(view_mode, tech, aggregation)
        _render_phase40_quality(tech)
        _render_phase41_coverage_footprint(view_mode, tech)
        _render_phase42_coverage_footprint(view_mode, tech)
        _render_phase37_quality(tech)


def main() -> None:
    st.set_page_config(page_title="Project 210 Phase Validation", layout="wide")
    render()


if __name__ == "__main__":
    main()
