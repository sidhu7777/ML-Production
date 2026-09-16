"""Phase 46: read-only production cell audit dashboard.

This dashboard does not recompute or write predictions. It exposes the exact
rows emitted by the production offset run, per cell, so raw, antenna/physical,
and calibrated values can be inspected independently.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "tests" / "output"
DEFAULT_RUN = OUTPUT_ROOT / "baseline_210_profile_20260908_142428"

st.set_page_config(page_title="Phase 46 production cell audit", layout="wide")
st.title("Phase 46 — Project 210 production cell audit")
st.caption("Read-only view of the complete production offset output. No database writes and no recalculation.")

run_text = st.sidebar.text_input("Production run folder", str(DEFAULT_RUN))
run = Path(run_text)
path = run / "baseline_predictions.parquet"
if not path.exists():
    st.error(f"Missing production output: {path}")
    st.stop()

@st.cache_data(show_spinner=False)
def load(path_text: str) -> pd.DataFrame:
    frame = pd.read_parquet(path_text)
    if "technology" not in frame and "Technology" in frame:
        frame["technology"] = frame["Technology"]
    frame["technology"] = frame["technology"].astype(str)
    frame["strict_cell_key"] = frame["strict_cell_key"].astype(str)
    return frame

df = load(str(path))
if "operator" not in df:
    df["operator"] = "unknown"

st.sidebar.metric("Rows", f"{len(df):,}")
st.sidebar.metric("Distinct cells", f"{df.strict_cell_key.nunique():,}")
st.sidebar.metric("4G cells", f"{df.loc[df.technology.eq('4G'), 'strict_cell_key'].nunique():,}")
st.sidebar.metric("5G cells", f"{df.loc[df.technology.eq('5G'), 'strict_cell_key'].nunique():,}")

tech = st.sidebar.selectbox("Technology", ["4G", "5G"])
sub = df[df.technology.eq(tech)].copy()
operators = sorted(sub.operator.astype(str).unique())
operator = st.sidebar.selectbox("Operator", ["All"] + operators)
if operator != "All":
    sub = sub[sub.operator.astype(str).eq(operator)]

cells = (sub[["strict_cell_key", "site", "sector", "band"]]
         .drop_duplicates("strict_cell_key")
         .sort_values(["site", "sector", "band", "strict_cell_key"]))
labels = [f"{r.strict_cell_key} | site={r.site} sector={r.sector} band={r.band}" for r in cells.itertuples()]
if not labels:
    st.warning("No cells for the selected filters.")
    st.stop()
selected = st.sidebar.selectbox("Cell (all cells)", labels)
cell_key = str(cells.iloc[labels.index(selected)].strict_cell_key)
cell = sub[sub.strict_cell_key.eq(cell_key)].copy()

st.subheader(selected)
metric_cols = st.columns(5)
metric_cols[0].metric("Grid rows", f"{len(cell):,}")
metric_cols[1].metric("Raw COST-231 median", f"{pd.to_numeric(cell.get('raw_cost231_rsrp'), errors='coerce').median():.1f} dBm")
metric_cols[2].metric("Physical median", f"{pd.to_numeric(cell.get('phase36_physical_rsrp'), errors='coerce').median():.1f} dBm")
metric_cols[3].metric("Final median", f"{pd.to_numeric(cell.get('final_rsrp'), errors='coerce').median():.1f} dBm")
metric_cols[4].metric("PAP rows", f"{int(cell.get('phase36_antenna_source', pd.Series(dtype=str)).astype(str).eq('pap').sum()):,}")

value_cols = [
    ("Raw COST-231", "raw_cost231_rsrp", "#64748b"),
    ("Physical / PAP", "phase36_physical_rsrp", "#2563eb"),
    ("Final calibrated", "final_rsrp", "#16a34a"),
]
plot = cell[["grid_id", "lat", "lon", "azimuth_delta_deg", "obstruction_branch", "phase36_antenna_source"] + [c for _, c, _ in value_cols if c in cell]].copy()
plot["value"] = pd.to_numeric(plot["final_rsrp"], errors="coerce") if "final_rsrp" in plot else np.nan
plot["grid_label"] = plot.grid_id.astype(str)
st.plotly_chart(px.scatter_mapbox(plot.dropna(subset=["lat", "lon"]), lat="lat", lon="lon", color="value", hover_name="grid_label", hover_data=["value", "azimuth_delta_deg", "obstruction_branch", "phase36_antenna_source"], zoom=13, height=560, mapbox_style="open-street-map", color_continuous_scale="RdYlGn", title="Selected cell final calibrated RSRP"), use_container_width=True)

series = []
for label, col, color in value_cols:
    if col in cell:
        vals = pd.to_numeric(cell[col], errors="coerce").dropna().sort_values().to_numpy()
        if len(vals):
            series.append(pd.DataFrame({"RSRP (dBm)": vals, "Cumulative %": np.arange(1, len(vals)+1)*100/len(vals), "stage": label}))
if series:
    st.plotly_chart(px.line(pd.concat(series), x="RSRP (dBm)", y="Cumulative %", color="stage", title="Selected cell raw → physical/PAP → final CDF"), use_container_width=True)

st.subheader("Exact production rows")
show = ["grid_id", "lat", "lon", "raw_cost231_rsrp", "building_obstruction_loss_db", "terrain_diffraction_loss_db", "phase36_antenna_source", "phase36_pap_file", "phase36_antenna_delta_db", "phase36_physical_rsrp", "final_rsrp", "azimuth_delta_deg", "obstruction_branch", "calibration_status"]
show = [c for c in show if c in cell]
st.dataframe(cell[show].sort_values("grid_id"), use_container_width=True, height=420, hide_index=True)
