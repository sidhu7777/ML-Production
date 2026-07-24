from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

PROJECT196_DIR = ML_ROOT / "models" / "model3_project196_input"
MODEL3_DIR = ML_ROOT / "models" / "model3_current_recommendation_experiment"
MODEL4_DIR = ML_ROOT / "models" / "model4_future_recommendation_experiment"
PROJECT196_CELL_INPUT = PROJECT196_DIR / "project_196_model3_cell_input.csv"
PROJECT196_BASELINE_INPUT = PROJECT196_DIR / "project_196_model3_baseline_grid_input.csv"

MODEL_CONFIG = {
    "Model 3": {
        "subtitle": "Current recommendation flow",
        "dataset": PROJECT196_BASELINE_INPUT,
        "scope_dataset": PROJECT196_CELL_INPUT,
        "scope_from_cell_input": True,
        "cell_input": PROJECT196_CELL_INPUT,
        "recommendations": MODEL3_DIR / "model3_current_recommendations.csv",
        "summary": MODEL3_DIR / "model3_current_recommendation_summary.json",
        "before_rf": None,
        "after_rf": MODEL3_DIR / "model3_after_rf_surface_combined.csv",
        "project_id": "196",
    },
    "Model 4": {
        "subtitle": "Future recommendation flow",
        "dataset": MODEL4_DIR / "model4_project196_future_dataset.csv",
        "scope_dataset": PROJECT196_CELL_INPUT,
        "scope_from_cell_input": True,
        "cell_input": PROJECT196_CELL_INPUT,
        "recommendations": MODEL4_DIR / "model4_future_recommendations.csv",
        "summary": MODEL4_DIR / "model4_future_recommendation_summary.json",
        "before_rf": None,
        "after_rf": MODEL4_DIR / "model4_full_after_baseline_rf_surface.csv",
        "project_id": "196",
    },
}


def load_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_json(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def clean_sector(value: Any, node_cell_id: Any = "") -> str:
    text = clean_text(value)
    if text and not text.lower().endswith("|nan"):
        return text
    node_text = clean_text(node_cell_id)
    parts = [part for part in node_text.replace("|", "_").split("_") if part and part.lower() != "nan"]
    if len(parts) >= 2:
        site = parts[0]
        sector = parts[-2] if parts[-1].isdigit() and len(parts) >= 3 else parts[-1]
        return f"{site}|{sector}"
    return ""


def has_value(value: Any) -> bool:
    return bool(clean_text(value))


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def pressure_color(value: float) -> str:
    if pd.isna(value):
        return "#6b7280"
    value = float(value)
    if value >= 90:
        return "#991b1b"
    if value >= 80:
        return "#dc2626"
    if value > 70:
        return "#f97316"
    if value >= 55:
        return "#facc15"
    return "#16a34a"


def rf_color(value: float, metric_name: str) -> str:
    if pd.isna(value):
        return "#6b7280"
    metric = metric_name.lower()
    value = float(value)
    if "rsrp" in metric or "rssi" in metric or "rxlev" in metric:
        if value <= -105:
            return "#dc2626"
        if value <= -95:
            return "#f97316"
        if value <= -85:
            return "#facc15"
        return "#16a34a"
    if "sinr" in metric:
        if value < 0:
            return "#dc2626"
        if value < 8:
            return "#f97316"
        if value < 15:
            return "#facc15"
        return "#16a34a"
    return pressure_color(value)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, str):
        return value
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    numeric_cols = [
        "estimated_prb_utilization_pct",
        "estimated_cell_rrc_utilization_pct",
        "model4_current_prb",
        "model4_current_rrc",
        "model4_pressure",
        "model4_raw_future_prb",
        "model4_raw_future_rrc",
        "topology_band",
        "input_prb_utilization_pct",
        "input_rrc_utilization_pct",
        "band",
        "grid_centroid_lat",
        "grid_centroid_lon",
        "lat",
        "lon",
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "pred_rsrp_geo",
        "pred_rsrq_geo",
        "pred_sinr_geo",
        "pred_rsrp_smoothed",
        "pred_rsrq_smoothed",
        "pred_sinr_smoothed",
        "rsrp_mean",
        "rsrq_mean",
        "sinr_mean",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = to_number(out[col])

    if "estimated_prb_utilization_pct" not in out.columns and "input_prb_utilization_pct" in out.columns:
        out["estimated_prb_utilization_pct"] = out["input_prb_utilization_pct"]
    if "estimated_cell_rrc_utilization_pct" not in out.columns and "input_rrc_utilization_pct" in out.columns:
        out["estimated_cell_rrc_utilization_pct"] = out["input_rrc_utilization_pct"]

    prb_col = "estimated_prb_utilization_pct"
    rrc_col = "estimated_cell_rrc_utilization_pct"
    if "model4_pressure" in out.columns:
        out["baseline_pressure"] = to_number(out["model4_pressure"])
    elif prb_col in out.columns and rrc_col in out.columns:
        out["baseline_pressure"] = out[[prb_col, rrc_col]].max(axis=1)
    else:
        out["baseline_pressure"] = pd.NA

    lat_col = first_existing(out, ["grid_centroid_lat", "lat"])
    lon_col = first_existing(out, ["grid_centroid_lon", "lon"])
    out["lat_for_map"] = to_number(out[lat_col]) if lat_col else pd.NA
    out["lon_for_map"] = to_number(out[lon_col]) if lon_col else pd.NA
    return out


def available_rf_metrics(*frames: pd.DataFrame) -> list[str]:
    preferred = [
        "pred_rsrp",
        "pred_rsrp_geo",
        "pred_rsrp_smoothed",
        "rsrp_mean",
        "pred_rsrq",
        "pred_rsrq_geo",
        "pred_sinr",
        "pred_sinr_geo",
        "pred_sinr_smoothed",
        "sinr_mean",
    ]
    available = []
    for col in preferred:
        if any(not frame.empty and col in frame.columns for frame in frames):
            available.append(col)
    return available


def resolve_metric_col(df: pd.DataFrame, metric_col: str) -> str | None:
    aliases = {
        "pred_rsrp_smoothed": ["pred_rsrp_smoothed", "pred_rsrp_geo", "pred_rsrp", "rsrp_mean"],
        "pred_rsrq_smoothed": ["pred_rsrq_smoothed", "pred_rsrq_geo", "pred_rsrq", "rsrq_mean"],
        "pred_sinr_smoothed": ["pred_sinr_smoothed", "pred_sinr_geo", "pred_sinr", "sinr_mean"],
    }
    for candidate in aliases.get(metric_col, [metric_col]):
        if candidate in df.columns:
            return candidate
    return None


def enrich_from_cell_input(dataset: pd.DataFrame, cell_input: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty or cell_input.empty or "Node_Cell_ID" not in dataset.columns or "Node_Cell_ID" not in cell_input.columns:
        return dataset

    cell = cell_input.copy()
    cell["Node_Cell_ID"] = cell["Node_Cell_ID"].map(clean_text)
    keep_map = {
        "input_prb_utilization_pct": "input_prb_utilization_pct",
        "input_rrc_utilization_pct": "input_rrc_utilization_pct",
        "input_rrc_connected_users": "input_rrc_connected_users",
        "input_available_bands_to_add": "input_available_bands_to_add",
        "available_bands_to_add": "cell_available_bands_to_add",
        "carrier_addition_possible": "cell_carrier_addition_possible",
        "site_id": "cell_site_id",
        "sector_id": "cell_sector_id",
        "band": "cell_band",
    }
    keep_cols = ["Node_Cell_ID"] + [col for col in keep_map if col in cell.columns]
    cell = cell[keep_cols].drop_duplicates(subset=["Node_Cell_ID"], keep="first").rename(columns=keep_map)

    out = dataset.copy()
    out["Node_Cell_ID"] = out["Node_Cell_ID"].map(clean_text)
    out = out.merge(cell, on="Node_Cell_ID", how="left", validate="many_to_one")
    for left, right in [
        ("estimated_prb_utilization_pct", "input_prb_utilization_pct"),
        ("estimated_cell_rrc_utilization_pct", "input_rrc_utilization_pct"),
        ("estimated_cell_rrc_connected_users", "input_rrc_connected_users"),
        ("available_bands_to_add", "input_available_bands_to_add"),
        ("carrier_addition_possible", "cell_carrier_addition_possible"),
        ("sector_id", "cell_sector_id"),
        ("topology_band", "cell_band"),
    ]:
        if right in out.columns:
            if left not in out.columns:
                out[left] = out[right]
            else:
                out[left] = out[left].where(out[left].map(has_value), out[right])
    if "cell_site_id" in out.columns:
        if "site_id" not in out.columns:
            out["site_id"] = out["cell_site_id"]
        else:
            out["site_id"] = out["site_id"].map(clean_text).str.replace(r"^s-", "", regex=True)
            out["site_id"] = out["site_id"].where(out["site_id"].map(has_value), out["cell_site_id"])
    if "sector_id" in out.columns:
        node_series = out["Node_Cell_ID"] if "Node_Cell_ID" in out.columns else pd.Series("", index=out.index)
        out["sector_id"] = [clean_sector(sector, node) for sector, node in zip(out["sector_id"], node_series)]
    return out


def recommendation_scope_warning(rec_df: pd.DataFrame, cell_input: pd.DataFrame, summary: dict[str, Any]) -> str:
    if rec_df.empty:
        return ""
    valid_sites = set()
    if "site_id" in cell_input.columns:
        valid_sites = set(cell_input["site_id"].map(clean_text).loc[lambda s: s.ne("")])
    rec_sites = set(rec_df["site_id"].map(clean_text).loc[lambda s: s.ne("")]) if "site_id" in rec_df.columns else set()
    outside_sites = sorted(rec_sites - valid_sites)
    dataset_path = str(summary.get("dataset_path", ""))
    if outside_sites or ("model3_current_dataset.csv" in dataset_path and "project_196" not in dataset_path):
        return (
            "The congested input scope is the corrected Project 196 18-cell input, but this "
            "recommendation artifact appears stale or generated from a different source. "
            f"Non-Project196 sites in recommendation CSV: {', '.join(outside_sites[:8]) or 'none detected'}."
        )
    return ""


def artifact_uses_project196_input(summary: dict[str, Any]) -> bool:
    text = json.dumps(summary, default=str).replace("\\", "/").lower()
    if not text:
        return False
    if "model3_current_dataset.csv" in text:
        return False
    if "model3_project196_input" in text or "project_196_model3" in text:
        return True
    return False


def valid_project196_cells(cell_input: pd.DataFrame) -> set[str]:
    if cell_input.empty or "Node_Cell_ID" not in cell_input.columns:
        return set()
    return set(cell_input["Node_Cell_ID"].map(clean_text).loc[lambda s: s.ne("")])


def valid_project196_sites(cell_input: pd.DataFrame) -> set[str]:
    if cell_input.empty or "site_id" not in cell_input.columns:
        return set()
    return set(cell_input["site_id"].map(clean_text).loc[lambda s: s.ne("")])


def artifact_has_only_project196_cells(df: pd.DataFrame, cell_input: pd.DataFrame) -> bool:
    if df.empty:
        return True
    valid_cells = valid_project196_cells(cell_input)
    valid_sites = valid_project196_sites(cell_input)
    if "Node_Cell_ID" in df.columns:
        cells = set(df["Node_Cell_ID"].map(clean_text).loc[lambda s: s.ne("")])
        if cells and valid_cells and not cells.issubset(valid_cells):
            return False
    if "site_id" in df.columns:
        sites = set(df["site_id"].map(clean_text).str.replace(r"^s-", "", regex=True).loc[lambda s: s.ne("")])
        synthetic_sites = {site for site in sites if site.startswith("MB") or site in {"1", "2", "3"}}
        if synthetic_sites:
            return False
        if sites and valid_sites and not sites.issubset(valid_sites):
            return False
    return True


def display_columns_for_dataset(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "site": first_existing(df, ["site_id", "topology_site_id", "node_b_id"]),
        "cell": first_existing(df, ["topology_original_cell_id", "topology_original_node_cell_id", "cell_id", "Node_Cell_ID"]),
        "sector": first_existing(df, ["sector_id", "topology_sector", "sector"]),
        "band": first_existing(df, ["topology_band", "band", "baseline_band"]),
        "node_cell": first_existing(df, ["Node_Cell_ID", "topology_rf_identity_key"]),
    }


def congested_cells_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = display_columns_for_dataset(df)
    keys = [col for col in [cols["site"], cols["cell"], cols["sector"], cols["band"], cols["node_cell"]] if col]
    keys = list(dict.fromkeys(keys))
    if not keys:
        return pd.DataFrame()

    prb_col = first_existing(df, ["estimated_prb_utilization_pct", "model4_current_prb"])
    rrc_col = first_existing(df, ["estimated_cell_rrc_utilization_pct", "model4_current_rrc"])
    available_col = first_existing(
        df,
        [
            "available_bands_to_add",
            "input_available_bands_to_add",
            "cell_available_bands_to_add",
            "carrier_addition_options",
            "recommended_band_to_add",
        ],
    )
    carrier_col = first_existing(df, ["carrier_addition_possible"])

    agg_spec: dict[str, tuple[str, str]] = {
        "prb": (prb_col, "max") if prb_col else ("baseline_pressure", "max"),
        "rrc": (rrc_col, "max") if rrc_col else ("baseline_pressure", "max"),
        "pressure": ("baseline_pressure", "max"),
    }
    if available_col:
        agg_spec["available_bands"] = (available_col, lambda s: ", ".join(sorted({clean_text(x) for x in s if has_value(x)})))
    if carrier_col:
        agg_spec["carrier_possible"] = (carrier_col, "max")

    grouped = df.groupby(keys, dropna=False).agg(**agg_spec).reset_index()
    grouped = grouped[grouped["pressure"] >= threshold].copy()
    if grouped.empty:
        return grouped

    grouped["site"] = grouped[cols["site"]].map(clean_text) if cols["site"] else ""
    grouped["cell"] = grouped[cols["cell"]].map(clean_text) if cols["cell"] else ""
    if cols["sector"]:
        node_col = cols["node_cell"] if cols["node_cell"] else None
        grouped["sector"] = [
            clean_sector(sector, node) for sector, node in zip(grouped[cols["sector"]], grouped[node_col] if node_col else "")
        ]
    else:
        grouped["sector"] = ""
    grouped["band"] = grouped[cols["band"]].map(clean_text) if cols["band"] else ""
    grouped["available_band"] = False
    if "available_bands" in grouped:
        grouped["available_band"] = grouped["available_bands"].map(has_value)
    if "carrier_possible" in grouped:
        carrier_possible = grouped["carrier_possible"].map(
            lambda value: clean_text(value).lower() in {"true", "1", "yes", "y"}
            or (pd.notna(value) and not isinstance(value, str) and bool(value))
        )
        grouped["available_band"] = grouped["available_band"].astype(bool) | carrier_possible

    view = grouped[["site", "cell", "sector", "band", "available_band", "prb", "rrc", "pressure"]].copy()
    return view.sort_values(["pressure", "prb", "rrc"], ascending=False).reset_index(drop=True)


def recommendations_table(rec_df: pd.DataFrame) -> pd.DataFrame:
    if rec_df.empty:
        return pd.DataFrame()

    out = rec_df.copy()
    for col in ["prb_before_pct", "rrc_before_pct", "projected_prb_after_pct", "projected_rrc_after_pct", "band"]:
        if col in out.columns:
            out[col] = to_number(out[col])

    out["site"] = out["site_id"].map(clean_text) if "site_id" in out else ""
    out["cell"] = out["Node_Cell_ID"].map(clean_text) if "Node_Cell_ID" in out else ""
    out["sector"] = out["sector_id"].map(clean_text) if "sector_id" in out else ""
    out["recommended_solution"] = out["action"].map(clean_text) if "action" in out else ""
    out["recommended_band"] = out["recommended_band_to_add"].map(clean_text) if "recommended_band_to_add" in out else ""
    out["new_sector_value"] = ""
    out["new_site_value"] = ""
    if "action" in out:
        sector_mask = out["action"].astype(str).str.contains("Sector", case=False, na=False)
        site_mask = out["action"].astype(str).str.contains("Site", case=False, na=False)
        out.loc[sector_mask, "new_sector_value"] = out.loc[sector_mask, "resimulation_flow"].map(clean_text) if "resimulation_flow" in out else "Sector Split"
        out.loc[site_mask, "new_site_value"] = out.loc[site_mask, "resimulation_flow"].map(clean_text) if "resimulation_flow" in out else "New Site"

    out["pressure_before_avg_pct"] = out[["prb_before_pct", "rrc_before_pct"]].mean(axis=1)
    out["pressure_after_avg_pct"] = out[["projected_prb_after_pct", "projected_rrc_after_pct"]].mean(axis=1)

    columns = [
        "site",
        "cell",
        "sector",
        "band",
        "recommended_solution",
        "prb_before_pct",
        "rrc_before_pct",
        "projected_prb_after_pct",
        "projected_rrc_after_pct",
        "recommended_band",
        "new_sector_value",
        "new_site_value",
        "pressure_before_avg_pct",
        "pressure_after_avg_pct",
        "status",
    ]
    existing = [col for col in columns if col in out.columns]
    return out[existing].sort_values("pressure_before_avg_pct", ascending=False, na_position="last").reset_index(drop=True)


def attach_after_pressure(dataset: pd.DataFrame, rec_df: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return dataset
    out = dataset.copy()
    out["after_pressure"] = out["baseline_pressure"]
    if rec_df.empty:
        return out

    rec = rec_df.copy()
    rec["recommendation_after_pressure"] = rec[["projected_prb_after_pct", "projected_rrc_after_pct"]].apply(
        lambda row: pd.to_numeric(row, errors="coerce").max(),
        axis=1,
    )

    if "sector_id" in out.columns and "sector_id" in rec.columns:
        sector_map = rec.dropna(subset=["sector_id"]).groupby("sector_id")["recommendation_after_pressure"].min()
        mapped = out["sector_id"].map(sector_map)
        out["after_pressure"] = mapped.combine_first(out["after_pressure"])

    if "Node_Cell_ID" in out.columns and "Node_Cell_ID" in rec.columns:
        cell_map = rec.dropna(subset=["Node_Cell_ID"]).groupby("Node_Cell_ID")["recommendation_after_pressure"].min()
        mapped = out["Node_Cell_ID"].map(cell_map)
        out["after_pressure"] = mapped.combine_first(out["after_pressure"])

    return out


def map_points(df: pd.DataFrame, value_col: str, max_points: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    value_col = resolve_metric_col(df, value_col) or ""
    if not value_col:
        return pd.DataFrame()
    required = ["lat_for_map", "lon_for_map", value_col]
    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    group_cols = ["grid_id"] if "grid_id" in df.columns else ["lat_for_map", "lon_for_map"]
    agg_spec = {
        "lat": ("lat_for_map", "mean"),
        "lon": ("lon_for_map", "mean"),
        "pressure": (value_col, "mean"),
    }
    if "site_id" in df.columns:
        agg_spec["site"] = ("site_id", lambda s: clean_text(s.iloc[0]) if len(s) else "")
    if "sector_id" in df.columns:
        agg_spec["sector"] = ("sector_id", lambda s: clean_text(s.iloc[0]) if len(s) else "")
    if "Node_Cell_ID" in df.columns:
        agg_spec["cells"] = ("Node_Cell_ID", "nunique")
    else:
        agg_spec["cells"] = (value_col, "count")
    agg = df.dropna(subset=required).groupby(group_cols, as_index=False).agg(**agg_spec)
    if len(agg) > max_points:
        agg = agg.sample(max_points, random_state=42)
    return agg


def recommendation_map_points(rec_df: pd.DataFrame, baseline_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if rec_df.empty or baseline_df.empty or value_col not in rec_df.columns:
        return pd.DataFrame()
    required = {"lat_for_map", "lon_for_map"}
    if not required.issubset(baseline_df.columns):
        return pd.DataFrame()

    site_lookup = pd.DataFrame()
    sector_lookup = pd.DataFrame()
    if "site_id" in baseline_df.columns:
        site_lookup = (
            baseline_df.dropna(subset=["lat_for_map", "lon_for_map"])
            .assign(site_key=lambda d: d["site_id"].map(clean_text).str.replace(r"^s-", "", regex=True))
            .groupby("site_key", as_index=False)
            .agg(lat=("lat_for_map", "median"), lon=("lon_for_map", "median"))
        )
    if "sector_id" in baseline_df.columns:
        sector_lookup = (
            baseline_df.dropna(subset=["lat_for_map", "lon_for_map"])
            .assign(sector_key=lambda d: d["sector_id"].map(clean_text))
            .groupby("sector_key", as_index=False)
            .agg(lat=("lat_for_map", "median"), lon=("lon_for_map", "median"))
        )

    out = rec_df.copy()
    out["pressure"] = to_number(out[value_col])
    out["site_key"] = out["site_id"].map(clean_text) if "site_id" in out.columns else ""
    out["sector_key"] = out["sector_id"].map(clean_text) if "sector_id" in out.columns else ""
    out["cell"] = out["Node_Cell_ID"].map(clean_text) if "Node_Cell_ID" in out.columns else ""
    out["action"] = out["action"].map(clean_text) if "action" in out.columns else ""

    if not sector_lookup.empty:
        out = out.merge(sector_lookup, on="sector_key", how="left", validate="many_to_one")
    else:
        out["lat"] = pd.NA
        out["lon"] = pd.NA
    if not site_lookup.empty:
        out = out.merge(site_lookup.rename(columns={"lat": "site_lat", "lon": "site_lon"}), on="site_key", how="left", validate="many_to_one")
        out["lat"] = out["lat"].combine_first(out["site_lat"])
        out["lon"] = out["lon"].combine_first(out["site_lon"])
    out = out.dropna(subset=["lat", "lon", "pressure"]).copy()
    out["cells"] = 1
    out["site"] = out["site_key"]
    out["sector"] = out["sector_key"]
    return out[["lat", "lon", "pressure", "site", "sector", "cells", "cell", "action"]].reset_index(drop=True)


def build_map(points: pd.DataFrame, title: str, key: str, metric_name: str = "pressure") -> None:
    if points.empty:
        st.info(f"No map-ready rows for {title}.")
        return

    fmap = folium.Map(
        location=[float(points["lat"].median()), float(points["lon"].median())],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
        width="100%",
        height=560,
    )
    for _, row in points.iterrows():
        pressure = float(row["pressure"]) if pd.notna(row["pressure"]) else float("nan")
        color = rf_color(pressure, metric_name) if metric_name != "pressure" else pressure_color(pressure)
        tooltip = (
            f"Grid: {row.get('grid_id', 'N/A')}<br>"
            f"Value: {fmt(pressure)}<br>"
            f"Site: {row.get('site', '')}<br>"
            f"Sector: {row.get('sector', '')}<br>"
            f"Cells: {row.get('cells', '')}<br>"
            f"Cell: {row.get('cell', '')}<br>"
            f"Action: {row.get('action', '')}"
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.78,
            opacity=0.9,
            weight=1,
            tooltip=tooltip,
        ).add_to(fmap)

    st.caption(title)
    st_folium(fmap, width=None, height=580, key=key)


def metric_row(dataset: pd.DataFrame, rec_df: pd.DataFrame, congested_df: pd.DataFrame, summary: dict[str, Any]) -> None:
    cols = st.columns(5)
    cols[0].metric("Congested Cells", f"{len(congested_df):,}")
    cols[1].metric("Recommendations", f"{len(rec_df):,}")
    cols[2].metric("Resolved", f"{int((rec_df.get('status', pd.Series(dtype=str)).astype(str) == 'Resolved').sum()):,}" if not rec_df.empty else "0")
    cols[3].metric("Avg Before", f"{dataset['baseline_pressure'].mean():.2f}%" if "baseline_pressure" in dataset else "N/A")
    cols[4].metric("Runtime", f"{summary.get('runtime_sec', 0):.1f}s" if summary.get("runtime_sec") else "N/A")


def page(model_name: str) -> None:
    config = MODEL_CONFIG[model_name]
    raw_baseline = load_csv(str(config["dataset"]))
    before_rf_path = config.get("before_rf")
    after_rf_path = config.get("after_rf")
    raw_before_rf = load_csv(str(before_rf_path)) if before_rf_path else pd.DataFrame()
    raw_after_rf = load_csv(str(after_rf_path)) if after_rf_path else pd.DataFrame()
    cell_input = load_csv(str(config["cell_input"]))
    if config.get("scope_from_cell_input"):
        raw_scope = cell_input.copy()
    else:
        raw_scope = load_csv(str(config.get("scope_dataset", config["dataset"])))
    baseline_dataset = normalize_dataset(enrich_from_cell_input(raw_baseline, cell_input))
    scope_dataset = normalize_dataset(raw_scope if config.get("scope_from_cell_input") else enrich_from_cell_input(raw_scope, cell_input))
    before_rf_dataset = normalize_dataset(raw_before_rf) if not raw_before_rf.empty else pd.DataFrame()
    after_rf_dataset = normalize_dataset(raw_after_rf) if not raw_after_rf.empty else pd.DataFrame()
    rec_df = load_csv(str(config["recommendations"]))
    summary = load_json(str(config["summary"]))
    threshold = float(summary.get("threshold") or 70.0)
    stale_recommendations = not artifact_uses_project196_input(summary) or not artifact_has_only_project196_cells(rec_df, cell_input)
    stale_after_rf = not artifact_has_only_project196_cells(raw_after_rf, cell_input)
    stale_before_rf = not artifact_has_only_project196_cells(raw_before_rf, cell_input)
    if stale_recommendations:
        rec_df = pd.DataFrame()
    if stale_after_rf:
        after_rf_dataset = pd.DataFrame()
    if stale_before_rf:
        before_rf_dataset = pd.DataFrame()

    st.subheader(config["subtitle"])
    st.caption(f"Project 196 RF baseline dataset: {config['dataset']}")
    st.caption(f"Project 196 selected congested scope: {config.get('scope_dataset', config['dataset'])}")
    if before_rf_path:
        st.caption(f"After-effect before RF: {before_rf_path}")
    if after_rf_path:
        st.caption(f"After-effect after RF: {after_rf_path}")
    st.caption(f"Cell input: {config['cell_input']}")
    st.caption(f"Recommendations: {config['recommendations']}")
    if stale_recommendations:
        st.warning(
            "Recommendation artifact is stale or not Project 196 aligned, so it is hidden. "
            "Run Model 3/Model 4 again on the corrected Project 196 18-cell Excel to populate this section."
        )
    if stale_after_rf:
        st.warning(
            "After-RF artifact is stale or contains non-Project196/synthetic-only sites, so it is hidden. "
            "This prevents outside-polygon points from being plotted as Project 196 results."
        )
    warning = recommendation_scope_warning(rec_df, cell_input, summary)
    if warning:
        st.warning(warning)

    if baseline_dataset.empty:
        st.error("Dataset CSV is missing or empty.")
        return
    if rec_df.empty:
        st.warning("Recommendation CSV is missing or empty. The before map will still render if the dataset has coordinates.")

    dataset_after = attach_after_pressure(scope_dataset, rec_df)
    congested_df = congested_cells_table(scope_dataset, threshold)
    rec_view = recommendations_table(rec_df)
    metric_row(dataset_after, rec_df, congested_df, summary)

    tabs = st.tabs(["Congested Cells", "Recommendations", "Project 196 RF Baseline", "Recommendation Pressure Before / After"])
    with tabs[0]:
        st.dataframe(congested_df.round(3), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.dataframe(rec_view.round(3), use_container_width=True, hide_index=True)

        if not rec_df.empty and "action" in rec_df.columns:
            action_counts = rec_df["action"].fillna("Unknown").value_counts().rename_axis("action").reset_index(name="count")
            st.bar_chart(action_counts.set_index("action"))

    with tabs[2]:
        if not after_rf_dataset.empty:
            st.info(
                "This is the real RF rerun after applying the recommendation topology changes. "
                "The before/after surfaces are affected-area RF artifacts from the optimized rerun, not a fake PRB projection."
            )
        else:
            st.info(
                "This page shows the saved Project 196 production baseline RF surface. "
                "No after-RF baseline surface file is configured for this model page yet."
            )
        rf_metrics = available_rf_metrics(baseline_dataset, before_rf_dataset, after_rf_dataset)
        if not rf_metrics:
            st.warning("No RF metric columns found in the configured baseline files.")
            return
        metric_col = st.selectbox("RF metric", rf_metrics, index=0)
        max_points = st.slider("RF map point limit", min_value=300, max_value=5000, value=1800, step=100)
        if not after_rf_dataset.empty:
            before_source = before_rf_dataset if not before_rf_dataset.empty else baseline_dataset
            left, right = st.columns(2)
            with left:
                before_rf_points = map_points(before_source, metric_col, max_points)
                build_map(before_rf_points, f"Before RF rerun: {metric_col}", f"{model_name}_rf_before_map", metric_col)
            with right:
                after_rf_points = map_points(after_rf_dataset, metric_col, max_points)
                build_map(after_rf_points, f"After RF rerun with added carrier/site/sector: {metric_col}", f"{model_name}_rf_after_map", metric_col)
        else:
            rf_points = map_points(baseline_dataset, metric_col, max_points)
            build_map(rf_points, f"Project 196 production baseline RF: {metric_col}", f"{model_name}_rf_baseline_map", metric_col)

    with tabs[3]:
        before_metric = "rec_before_pressure_pct"
        after_metric = "rec_after_pressure_pct"
        rec_for_map = rec_df.copy()
        if not rec_for_map.empty:
            rec_for_map[before_metric] = rec_for_map[["prb_before_pct", "rrc_before_pct"]].apply(
                lambda row: pd.to_numeric(row, errors="coerce").max(),
                axis=1,
            )
            rec_for_map[after_metric] = rec_for_map[["projected_prb_after_pct", "projected_rrc_after_pct"]].apply(
                lambda row: pd.to_numeric(row, errors="coerce").max(),
                axis=1,
            )
        before_points = recommendation_map_points(rec_for_map, baseline_dataset, before_metric)
        after_points = recommendation_map_points(rec_for_map, baseline_dataset, after_metric)
        missing_points = len(rec_for_map) - len(before_points)
        if missing_points > 0:
            st.warning(
                f"{missing_points} recommendation rows could not be placed on the Project 196 baseline map "
                "because their site/sector is not present in the Project 196 baseline artifact."
            )
        left, right = st.columns(2)
        with left:
            build_map(before_points, "Before recommendation pressure per recommended sector/cell: max(PRB%, RRC%)", f"{model_name}_before_map")
        with right:
            build_map(after_points, "After recommendation projected pressure per recommended sector/cell: max(PRB%, RRC%)", f"{model_name}_after_map")


def main() -> None:
    st.set_page_config(page_title="Model 3 / Model 4 Recommendation Dashboard", layout="wide")
    st.title("Model 3 / Model 4 Recommendation Dashboard")
    st.caption("Standalone Streamlit view for recommendation outputs. This is separate from the Excel workbook.")

    model_name = st.sidebar.radio("Page", list(MODEL_CONFIG.keys()))
    page(model_name)


if __name__ == "__main__":
    main()
