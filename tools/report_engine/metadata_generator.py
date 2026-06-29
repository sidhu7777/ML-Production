# src/metadata_generator.py

import json
import time
import logging
from typing import Dict, Optional, List
from math import radians, cos, sin, asin, sqrt

import pandas as pd
from geopy.geocoders import Nominatim

from .map_generator import normalize_band_name

logger = logging.getLogger(__name__)
geolocator = Nominatim(user_agent="pdf_report_generator", timeout=3)

# --------------------------------------------------
# GRID HELPERS
# --------------------------------------------------

def build_spatial_grid(df: pd.DataFrame, grid_size_deg: float = 0.002):
    df = df.dropna(subset=["lat", "lon"]).copy()
    df["lat_bin"] = (df["lat"] / grid_size_deg).astype(int)
    df["lon_bin"] = (df["lon"] / grid_size_deg).astype(int)

    return (
        df.groupby(["lat_bin", "lon_bin"])
        .agg(
            sample_count=("lat", "count"),
            avg_speed=("speed", "mean") if "speed" in df.columns else ("lat", "count"),
            center_lat=("lat", "mean"),
            center_lon=("lon", "mean"),
        )
        .reset_index()
    )

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def select_spatially_separated_cells(grid_df, min_distance_m=400, max_cells=6):
    selected = []
    for _, row in grid_df.iterrows():
        if len(selected) >= max_cells:
            break
        keep = True
        for sel in selected:
            if haversine(
                row.center_lat, row.center_lon,
                sel.center_lat, sel.center_lon
            ) < min_distance_m:
                keep = False
                break
        if keep:
            selected.append(row)
    return selected

# --------------------------------------------------
# GEOCODING
# --------------------------------------------------

def reverse_geocode_area(lat, lon, sleep_sec=1.0):
    try:
        loc = geolocator.reverse((lat, lon), zoom=17, language="en")
        time.sleep(sleep_sec)
        if not loc:
            return None
        addr = loc.raw.get("address", {})
        labels = [
            addr.get(k) for k in
            ["road", "neighbourhood", "suburb", "quarter", "city_district", "locality"]
            if addr.get(k)
        ]
        return {
            "labels": labels,
            "class": loc.raw.get("class"),
            "type": loc.raw.get("type"),
        }
    except Exception as e:
        logger.warning(f"Reverse geocode failed: {e}")
        return None


def reverse_geocode_location(lat, lon, sleep_sec=1.0):
    """Return coarse location info (city/country) for report metadata."""
    try:
        loc = geolocator.reverse((lat, lon), zoom=10, language="en")
        time.sleep(sleep_sec)
        if not loc:
            return None
        addr = loc.raw.get("address", {})
        city = addr.get("city") or addr.get("town") or addr.get("village")
        country = addr.get("country")
        if not (city or country):
            return None
        return {"city": city, "country": country}
    except Exception as e:
        logger.warning(f"Reverse geocode location failed: {e}")
        return None


def build_location(filtered_df: pd.DataFrame, sleep_sec: float = 1.0) -> Optional[Dict]:
    """Infer city/country from the first valid lat/lon in the dataset."""
    if not {"lat", "lon"}.issubset(filtered_df.columns):
        return None
    coords = filtered_df[["lat", "lon"]].dropna()
    if coords.empty:
        return None
    lat, lon = coords.iloc[0].tolist()
    return reverse_geocode_location(lat, lon, sleep_sec=sleep_sec)

# --------------------------------------------------
# AREA SUMMARY (STRUCTURED, NOT TEXT)
# --------------------------------------------------

def build_area_summary(filtered_df: pd.DataFrame, top_n: int = 6, sleep_sec: float = 1.0) -> Optional[Dict]:
    if not {"lat", "lon"}.issubset(filtered_df.columns):
        return None

    grid = build_spatial_grid(filtered_df)
    if grid.empty:
        return None

    grid = grid.sort_values("sample_count", ascending=False)
    selected_cells = select_spatially_separated_cells(grid, max_cells=top_n)

    speed_median = (
        filtered_df["speed"].median()
        if "speed" in filtered_df.columns else None
    )

    hotspots_list = []
    crowded_list = []
    covered_list = []

    for cell in selected_cells:
        geo = reverse_geocode_area(cell.center_lat, cell.center_lon, sleep_sec=sleep_sec)
        if not geo or not geo.get("labels"):
            continue

        name = geo["labels"][0]

        # Collect for paragraph summary
        covered_list.append({
            "name": name,
            "type": f"{geo.get('class','')} {geo.get('type','')}".strip(),
            "samples": int(cell.sample_count)
        })

        # Only add top 3 hotspots (just names)
        if len(hotspots_list) < 3:
            hotspots_list.append(name)

        # Only add top 3 crowded locations (just names)
        if speed_median is not None and cell.avg_speed <= speed_median and len(crowded_list) < 3:
            crowded_list.append(name)

    # Build Major Areas Covered as a paragraph
    if covered_list:
        area_parts = []
        for item in covered_list[:6]:  # Use up to 6 areas
            area_parts.append(f"{item['name']} ({item['type']}, {item['samples']} samples)")
        covered_paragraph = "The drive covered major areas including " + ", ".join(area_parts) + "."
    else:
        covered_paragraph = "Major areas covered during the drive."

    return {
        "Overview": (
            "Drive route covers key operational areas identified based on spatial "
            "distribution of samples and session density."
        ),
        "Hotspots & Marked Locations": ", ".join(hotspots_list) if hotspots_list else "Not identified",
        "Crowded & High-Traffic Locations": ", ".join(crowded_list) if crowded_list else "Not identified",
        "Major Areas Covered": covered_paragraph,
    }


def build_band_summary(filtered_df: pd.DataFrame) -> Optional[List[Dict]]:
    if "band" not in filtered_df.columns:
        return None

    # Normalize raw band values ("0", "-1", "Unknown", "N/A", …) and drop the
    # Unknown bucket so the denominator is the count of *known* bands.  This
    # keeps the LLM-facing band % identical to the band table / pie chart.
    normalized = filtered_df["band"].apply(normalize_band_name)
    known = normalized[normalized != "Unknown"]
    band_counts = known.value_counts()
    if band_counts.empty:
        return None

    total = int(band_counts.sum())
    summary = []
    for band, count in band_counts.items():
        pct = round((count / total) * 100, 2) if total > 0 else 0
        summary.append({
            "band": band,
            "sample_count": int(count),
            "sample_percentage": pct,
        })

    return summary


def summarize_kpi_details(kpi_details: Dict) -> Dict:
    """
    Create a compact KPI summary dict with only numeric headline fields.
    Remove distribution to keep only necessary content.
    """
    if not isinstance(kpi_details, dict):
        return {}

    keep_fields = {
        "average",
        "min",
        "max",
        "median",
        "percentile_25",
        "percentile_75",
        "poor_threshold",
        "poor_count",
        "poor_percentage",
    }

    out = {}
    for kpi, data in kpi_details.items():
        if not isinstance(data, dict):
            continue
        out[kpi] = {k: data.get(k) for k in keep_fields if k in data}

    return out


def add_fixed_threshold_analysis(filtered_df: pd.DataFrame, kpi_summary: Dict) -> Dict:
    """
    Add fixed threshold analysis for uniform reporting (not dependent on DB ranges).
    Uses thresholds from kpi_config.FIXED_THRESHOLD_CONFIG.
    """
    from .kpi_config import FIXED_THRESHOLD_CONFIG
    
    if not isinstance(kpi_summary, dict):
        return kpi_summary
    
    total_samples = len(filtered_df)
    
    # DL Throughput: poor and excellent thresholds from config
    if "DL" in kpi_summary or "DL Throughput" in kpi_summary:
        kpi_key = "DL" if "DL" in kpi_summary else "DL Throughput"
        if "dl_tpt" in filtered_df.columns:
            dl_data = pd.to_numeric(filtered_df["dl_tpt"], errors="coerce").dropna()
            if len(dl_data) > 0:
                config = FIXED_THRESHOLD_CONFIG.get("DL", {})
                poor_thresh = config.get("poor_threshold", 5)
                excellent_thresh = config.get("excellent_threshold", 19)
                
                below_poor = (dl_data <= poor_thresh).sum()
                below_poor_pct = round((below_poor / len(dl_data)) * 100, 2)
                above_excellent = (dl_data > excellent_thresh).sum()
                above_excellent_pct = round((above_excellent / len(dl_data)) * 100, 2)
                
                # Replace DB-based poor values with config-based values
                kpi_summary[kpi_key]["poor_threshold"] = poor_thresh
                kpi_summary[kpi_key]["poor_count"] = int(below_poor)
                kpi_summary[kpi_key]["poor_percentage"] = below_poor_pct
                kpi_summary[kpi_key]["excellent_threshold_value"] = excellent_thresh
                kpi_summary[kpi_key]["excellent_count"] = int(above_excellent)
                kpi_summary[kpi_key]["excellent_percentage"] = above_excellent_pct
    
    # UL Throughput: poor threshold and mid-range from config
    if "UL" in kpi_summary or "UL Throughput" in kpi_summary:
        kpi_key = "UL" if "UL" in kpi_summary else "UL Throughput"
        if "ul_tpt" in filtered_df.columns:
            ul_data = pd.to_numeric(filtered_df["ul_tpt"], errors="coerce").dropna()
            if len(ul_data) > 0:
                config = FIXED_THRESHOLD_CONFIG.get("UL", {})
                poor_thresh = config.get("poor_threshold", 5)
                range_min = config.get("range_min", 4)
                range_max = config.get("range_max", 6)
                
                below_poor = (ul_data <= poor_thresh).sum()
                below_poor_pct = round((below_poor / len(ul_data)) * 100, 2)
                in_range = ((ul_data >= range_min) & (ul_data <= range_max)).sum()
                in_range_pct = round((in_range / len(ul_data)) * 100, 2)
                
                # Replace DB-based poor values with config-based values
                kpi_summary[kpi_key]["poor_threshold"] = poor_thresh
                kpi_summary[kpi_key]["poor_count"] = int(below_poor)
                kpi_summary[kpi_key]["poor_percentage"] = below_poor_pct
                kpi_summary[kpi_key]["range_min"] = range_min
                kpi_summary[kpi_key]["range_max"] = range_max
                kpi_summary[kpi_key]["range_count"] = int(in_range)
                kpi_summary[kpi_key]["range_percentage"] = in_range_pct
    
    # MOS: poor threshold from config
    if "MOS" in kpi_summary:
        if "mos" in filtered_df.columns:
            mos_data = pd.to_numeric(filtered_df["mos"], errors="coerce").dropna()
            if len(mos_data) > 0:
                config = FIXED_THRESHOLD_CONFIG.get("MOS", {})
                poor_thresh = config.get("poor_threshold", 2.0)
                
                below_poor = (mos_data < poor_thresh).sum()
                below_poor_pct = round((below_poor / len(mos_data)) * 100, 2)
                
                # Replace DB-based poor values with config-based values
                kpi_summary["MOS"]["poor_threshold"] = poor_thresh
                kpi_summary["MOS"]["poor_count"] = int(below_poor)
                kpi_summary["MOS"]["poor_percentage"] = below_poor_pct
    
    # SINR: poor threshold and acceptable range from config
    if "SINR" in kpi_summary:
        if "sinr" in filtered_df.columns:
            sinr_data = filtered_df["sinr"].dropna()
            if len(sinr_data) > 0:
                config = FIXED_THRESHOLD_CONFIG.get("SINR", {})
                poor_thresh = config.get("poor_threshold", 0)
                range_min = config.get("range_min", 0)
                range_max = config.get("range_max", 10)
                
                below_poor = (sinr_data < poor_thresh).sum()
                below_poor_pct = round((below_poor / len(sinr_data)) * 100, 2)
                in_range = ((sinr_data >= range_min) & (sinr_data <= range_max)).sum()
                in_range_pct = round((in_range / len(sinr_data)) * 100, 2)
                
                # Replace DB-based poor values with config-based values
                kpi_summary["SINR"]["poor_threshold"] = poor_thresh
                kpi_summary["SINR"]["poor_count"] = int(below_poor)
                kpi_summary["SINR"]["poor_percentage"] = below_poor_pct
                kpi_summary["SINR"]["range_min"] = range_min
                kpi_summary["SINR"]["range_max"] = range_max
                kpi_summary["SINR"]["range_count"] = int(in_range)
                kpi_summary["SINR"]["range_percentage"] = in_range_pct
    
    return kpi_summary

# --------------------------------------------------
# METADATA
# --------------------------------------------------

def build_metadata(
    filtered_df: pd.DataFrame,
    kpi_details: Optional[Dict] = None,
    drive_summary: Optional[Dict] = None,
    location: Optional[Dict] = None,
    **kwargs,
) -> Dict:
    """
    Build metadata for report consumption.

    Backwards-compatible: accepts either positional `kpi_details`/`drive_summary`
    or keyword args `kpi_analysis_results` and `drive_summary_data` used by
    some callers.
    """

    # Support alternate caller parameter names
    if kpi_details is None:
        kpi_details = kwargs.get("kpi_analysis_results") or kwargs.get("kpi_details") or {}
    if drive_summary is None:
        drive_summary = kwargs.get("drive_summary_data") or kwargs.get("drive_summary") or {}

    kpi_summary = summarize_kpi_details(kpi_details)
    
    # Add fixed threshold analysis for uniform reporting
    kpi_summary = add_fixed_threshold_analysis(filtered_df, kpi_summary)

    # Build PCI summary if available
    pci_summary = None
    if "pci" in filtered_df.columns:
        pci_values = filtered_df["pci"].dropna()
        if not pci_values.empty:
            total_unique = pci_values.nunique()
            top_30_count = pci_values.value_counts().head(30).sum()
            top_30_pct = round((top_30_count / len(pci_values)) * 100, 2) if len(pci_values) > 0 else 0
            pci_summary = {
                "total_unique_pci": int(total_unique),
                "top_30_pci_percentage": float(top_30_pct),
            }

    if location is None:
        location = build_location(filtered_df)

    return {
        "location": location or {},
        "introduction": (
            f"Data includes {len(filtered_df)} samples "
            f"from {filtered_df['session_id'].nunique()} sessions."
            if "session_id" in filtered_df.columns else
            f"Data includes {len(filtered_df)} samples."
        ),
        "area_summary": build_area_summary(filtered_df),
        "drive_summary": drive_summary,
        "kpi_summary": kpi_summary,
        "band_summary": build_band_summary(filtered_df),
        "pci_summary": pci_summary,
    }

def _json_safe_metadata(value):
    if value is None:
        return None
    if value is pd.NA:
        return None
    if isinstance(value, dict):
        return {k: _json_safe_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_metadata(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe_metadata(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_safe_metadata(value.item())
        except Exception:
            pass
    if pd.isna(value):
        return None
    return value


def write_metadata_file(metadata: Dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe_metadata(metadata), f, indent=2, ensure_ascii=False)
