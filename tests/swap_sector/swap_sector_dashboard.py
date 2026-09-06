"""
Sector-Swap Detection — test-only validation dashboard (Streamlit).

Purpose: visually validate detect_sector_swap.py's output against the data
already fetched by fetch_data.py / make_synthetic_swap.py -- NOT a new data
path. Everything here reads the local CSV snapshots in tests/swap_sector/data/
(no DB connection, no production writes), the same "local snapshot" pattern
tests/Pci_optimization/pci_map_dashboard.py already uses for offline
iteration. Sector-wedge geometry and the PCI color palette are reused
directly from that dashboard so a sector reads the same visual way here as
it does in the live app / PCI-optimization dashboard.

Four things this version makes explicit (v1 buried or omitted all four):
  1. Full per-sector identity (cell_id, sector, band, EARFCN) -- not just a
     site-level label -- with the ".0" float-formatting bug on site ids fixed.
  2. A legend for what the map wedges mean (solid=configured, dashed=observed).
  3. The actual evidence behind a verdict: DT sample counts + RSRP/RSRQ/SINR
     per PCI, a polar chart of which PCI dominates each bearing bin, and the
     real HO transition events plotted on the map -- not just the final
     summarized wedges.
  4. An explicit CONFIGURED vs PREDICTED comparison table per sector, with a
     clear match/mismatch column, instead of one line of red text.

Run from the ML/ directory:
    venv\\Scripts\\python.exe -m streamlit run tests/swap_sector/swap_sector_dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from tests.Pci_optimization.pci_map_dashboard import build_sector_triangle, pci_color
from tests.swap_sector.detect_sector_swap import (
    BIN_SIZE_DEG,
    MIN_BIN_SAMPLES,
    dt_observed_azimuth,
    fuse_observations,
    ho_observed_azimuth,
    run as run_detection,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
WEDGE_RADIUS_M = 300.0
BEAMWIDTH_DEG = 40.0

VERDICT_BADGE = {
    "NORMAL": ("✅", "green"),
    "PROBABLE_SWAP": ("⚠️", "orange"),
    "CONFIRMED_SWAP": ("🚨", "red"),
    "INSUFFICIENT_DATA": ("❔", "gray"),
}


def clean_site_id(value) -> str:
    """358.0 -> '358', 1.82 -> '1.82' -- CSV round-trips read numeric-looking
    site ids as float64, which is a display artifact, not a real identity
    change; strip a trailing .0 only, keep genuine decimals like site 1.82."""
    s = str(value)
    return s[:-2] if s.endswith(".0") else s


@st.cache_data
def load_data(config_name: str):
    config_file = "site_config.csv" if config_name == "real" else "synthetic_swap_site_config.csv"
    site_df = pd.read_csv(DATA_DIR / config_file)
    site_df["site_id_display"] = site_df["site_id_inferred"].apply(clean_site_id)
    dt_df = pd.read_csv(DATA_DIR / "dt_samples_6sites.csv")
    dt_df["site_id_display"] = dt_df["site_id_inferred"].apply(clean_site_id)
    # dt_observed_azimuth() matches on "site_id_inferred" -- point it at the
    # cleaned id directly so callers can pass the same display string used
    # everywhere else on this page (no float-vs-string mismatch).
    dt_df["site_id_inferred"] = dt_df["site_id_display"]
    ho_df = pd.read_csv(DATA_DIR / "ho_events_6sites.csv")
    return site_df, dt_df, ho_df


@st.cache_data
def load_results(config_name: str) -> pd.DataFrame:
    df = run_detection(config_name)
    df["site_id_display"] = df["site_id"].apply(clean_site_id)
    return df


def badge(verdict: str) -> str:
    icon, color = VERDICT_BADGE.get(verdict.split(" ")[0], ("❔", "gray"))
    return f":{color}[{icon} {verdict}]"


def pci_stats_table(dt_df: pd.DataFrame, ho_df: pd.DataFrame, site_id_display: str, sector_pcis: set) -> pd.DataFrame:
    site_dt = dt_df[(dt_df["site_id_display"] == site_id_display) & (dt_df["pci"].isin(sector_pcis))]
    rows = []
    for pci in sorted(sector_pcis):
        sub = site_dt[site_dt["pci"] == pci]
        n_ho = 0
        if not ho_df.empty:
            n_ho = int(
                ((ho_df["from_pci"] == pci) & (ho_df["from_site_id_inferred"].astype(str).apply(clean_site_id) == site_id_display)).sum()
                + ((ho_df["to_pci"] == pci) & (ho_df["to_site_id_inferred"].astype(str).apply(clean_site_id) == site_id_display)).sum()
            )
        rows.append({
            "PCI": pci,
            "DT samples": len(sub),
            "avg RSRP": round(sub["rsrp"].mean(), 1) if len(sub) else None,
            "avg RSRQ": round(sub["rsrq"].mean(), 1) if len(sub) else None,
            "avg SINR": round(sub["sinr"].mean(), 1) if len(sub) else None,
            "HO events touching this PCI": n_ho,
        })
    return pd.DataFrame(rows)


def bearing_rose_chart(dt_df: pd.DataFrame, site_id_display: str, sector_pcis: set):
    """Polar bar chart: for every BIN_SIZE_DEG bearing bin, how many DT
    samples of each of the site's own PCIs fall in it. This IS the raw
    evidence dt_observed_azimuth() summarizes into one azimuth per PCI --
    shown directly so the summarized number can be checked against the
    actual distribution it came from."""
    site_dt = dt_df[(dt_df["site_id_display"] == site_id_display) & (dt_df["pci"].isin(sector_pcis))]
    fig = go.Figure()
    if site_dt.empty:
        return fig
    for pci in sorted(sector_pcis):
        sub = site_dt[site_dt["pci"] == pci]
        counts = sub.groupby("angle_bin_10deg").size()
        bins = list(range(0, 360, BIN_SIZE_DEG))
        r = [int(counts.get(b, 0)) for b in bins]
        fig.add_trace(go.Barpolar(
            r=r, theta=[b + BIN_SIZE_DEG / 2 for b in bins], width=[BIN_SIZE_DEG] * len(bins),
            name=f"PCI {int(pci)}", marker_color=pci_color(pci), opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90, tickmode="array", tickvals=list(range(0, 360, 30)))),
        showlegend=True, height=420, margin=dict(l=10, r=10, t=30, b=10),
        title=f"DT sample count per {BIN_SIZE_DEG}° bearing bin (min {MIN_BIN_SAMPLES} samples/bin to count as evidence)",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Sector Swap Detection — Validation", layout="wide")
    st.title("Sector Swap Detection — Validation Dashboard (test case)")
    st.caption(
        "Reads only the local snapshots in tests/swap_sector/data/ -- no DB hit, no production writes."
    )

    config_name = st.sidebar.radio(
        "Config source",
        options=["real", "synthetic"],
        format_func=lambda x: "Real (unmodified)" if x == "real" else "Synthetic swap (injected, ground truth known)",
    )
    site_df, dt_df, ho_df = load_data(config_name)
    results_df = load_results(config_name)

    st.subheader("All sites — detection summary")
    display_df = results_df.copy()
    display_df["site_id"] = display_df["site_id_display"]
    display_df["verdict"] = display_df["verdict"].apply(badge)
    st.dataframe(
        display_df[["site_id", "n_sectors", "verdict", "confidence_score", "evidence_completeness", "ground_truth_has_swap"]],
        use_container_width=True, hide_index=True,
    )

    site_ids = sorted(site_df["site_id_display"].unique())
    selected_site = st.sidebar.selectbox("Site", site_ids)

    site_rows = site_df[site_df["site_id_display"] == selected_site].reset_index(drop=True)
    site_result_rows = results_df[results_df["site_id_display"] == selected_site]
    site_result = site_result_rows.iloc[0] if not site_result_rows.empty else None
    sector_pcis = set(site_rows["site_pci"].tolist())

    site_lat = site_rows["site_lat"].mean()
    site_lon = site_rows["site_lon"].mean()
    dt_obs = dt_observed_azimuth(dt_df, selected_site, sector_pcis)
    ho_obs = ho_observed_azimuth(ho_df, selected_site, site_lat, site_lon)
    fused_obs = fuse_observations(dt_obs, ho_obs)

    st.markdown("---")
    st.subheader(f"Site {selected_site} — full sector identity")
    identity_cols = ["site_cell_id_representative", "sector", "band", "site_earfcn", "network", "site_pci", "site_azimuth_deg"]
    identity_cols = [c for c in identity_cols if c in site_rows.columns]
    st.dataframe(
        site_rows[identity_cols].rename(columns={
            "site_cell_id_representative": "cell_id", "site_earfcn": "earfcn",
            "network": "operator", "site_pci": "configured_pci", "site_azimuth_deg": "configured_azimuth",
        }),
        hide_index=True, use_container_width=True,
    )

    # ---- 4. Explicit CONFIGURED vs PREDICTED comparison (the main answer) ----
    st.subheader("Configured vs Predicted — the actual comparison")
    if site_result is not None and site_result["best_perm"] is not None:
        import ast
        identity_perm = site_result["identity_perm"]
        best_perm = site_result["best_perm"]
        if isinstance(identity_perm, str):
            identity_perm = ast.literal_eval(identity_perm)
        if isinstance(best_perm, str):
            best_perm = ast.literal_eval(best_perm)
        comp_rows = []
        for (_, sector), predicted_pci in zip(site_rows.iterrows(), best_perm):
            configured_pci = sector["site_pci"]
            configured_az = sector["site_azimuth_deg"]
            obs_of_configured = fused_obs.get(configured_pci)
            obs_of_predicted = fused_obs.get(predicted_pci)
            comp_rows.append({
                "cell_id": sector["site_cell_id_representative"],
                "configured_azimuth": configured_az,
                "configured_PCI": configured_pci,
                "predicted_PCI (best-fit)": predicted_pci,
                "match?": "✅ same" if configured_pci == predicted_pci else "❌ DIFFERENT",
                "observed_azimuth_of_configured_PCI": round(obs_of_configured[0], 1) if obs_of_configured else None,
                "observed_azimuth_of_predicted_PCI": round(obs_of_predicted[0], 1) if obs_of_predicted else None,
                "confidence_of_predicted_PCI": round(obs_of_predicted[1], 2) if obs_of_predicted else None,
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.markdown(
            f"**Verdict: {badge(str(site_result['verdict']))}**  |  "
            f"Confidence score: **{site_result['confidence_score']}**  |  "
            f"Evidence completeness: **{site_result['evidence_completeness']}**  |  "
            f"Total angular error — configured mapping: **{site_result['identity_avg_error_deg']}°**, "
            f"best-fit mapping: **{site_result['best_avg_error_deg']}°**"
        )
        if config_name == "synthetic" and bool(site_result["ground_truth_has_swap"]):
            st.info("Ground truth: this site DOES have an injected synthetic swap (for validation).")
    else:
        st.warning("Not enough sectors at this site to run a permutation comparison (config incomplete).")

    # ---- 3. Underlying evidence: RSRP/RSRQ/SINR + HO count per PCI ----
    st.subheader("Evidence behind the prediction — per-PCI stats")
    st.dataframe(pci_stats_table(dt_df, ho_df, selected_site, sector_pcis), hide_index=True, use_container_width=True)

    col_map, col_rose = st.columns([3, 2])

    with col_rose:
        st.plotly_chart(bearing_rose_chart(dt_df, selected_site, sector_pcis), use_container_width=True)

    with col_map:
        st.markdown(
            "**Map legend:** solid wedge = configured azimuth (colored by configured PCI) · "
            "dashed wedge = observed azimuth for that same PCI (from DT+HO evidence) · "
            "★ markers = real handover events touching this site, colored by the PCI on that side of the transition."
        )
        fmap = folium.Map(location=[site_lat, site_lon], zoom_start=17, tiles="CartoDB positron")
        folium.CircleMarker(
            location=[site_lat, site_lon], radius=6, color="#1d4ed8", fill=True, fill_color="#1d4ed8",
            tooltip=f"Site {selected_site}",
        ).add_to(fmap)

        for _, sector in site_rows.iterrows():
            pci = sector["site_pci"]
            color = pci_color(pci)
            configured_triangle = build_sector_triangle(site_lat, site_lon, sector["site_azimuth_deg"], BEAMWIDTH_DEG, WEDGE_RADIUS_M)
            folium.Polygon(
                locations=configured_triangle, color=color, weight=2, fill=True, fill_color=color, fill_opacity=0.35,
                tooltip=f"CONFIGURED: {sector['site_cell_id_representative']} — PCI {pci} @ {sector['site_azimuth_deg']:.0f}°",
            ).add_to(fmap)
            if pci in fused_obs:
                obs_az, confidence, n = fused_obs[pci]
                observed_triangle = build_sector_triangle(site_lat, site_lon, obs_az, BEAMWIDTH_DEG, WEDGE_RADIUS_M * 1.15)
                folium.Polygon(
                    locations=observed_triangle, color=color, weight=3, dash_array="8,6", fill=False,
                    tooltip=f"OBSERVED: PCI {pci} @ {obs_az:.0f}° (confidence {confidence:.2f}, n={n})",
                ).add_to(fmap)

        # Real HO transition events touching this site
        if not ho_df.empty:
            from_here = ho_df[ho_df["from_site_id_inferred"].astype(str).apply(clean_site_id) == selected_site]
            to_here = ho_df[ho_df["to_site_id_inferred"].astype(str).apply(clean_site_id) == selected_site]
            for _, ev in from_here.iterrows():
                folium.RegularPolygonMarker(
                    location=[ev["event_lat"], ev["event_lon"]], number_of_sides=5, radius=6,
                    color=pci_color(ev["from_pci"]), fill=True, fill_color=pci_color(ev["from_pci"]),
                    tooltip=f"HO from this site: PCI {ev['from_pci']} -> {ev['to_pci']}",
                ).add_to(fmap)
            for _, ev in to_here.iterrows():
                folium.RegularPolygonMarker(
                    location=[ev["event_lat"], ev["event_lon"]], number_of_sides=5, radius=6,
                    color=pci_color(ev["to_pci"]), fill=True, fill_color="white",
                    tooltip=f"HO into this site: PCI {ev['from_pci']} -> {ev['to_pci']}",
                ).add_to(fmap)

        st_folium(fmap, width=None, height=560, key=f"map_{config_name}_{selected_site}")


if __name__ == "__main__":
    main()
