"""
Report Engine PPT Generator Integration Module
Loads data for a given project_id using ML-Production report_engine,
renders technology-wise and KPI-filtered map images (WITHOUT folium legends —
legends come from the template PPT's static image shapes),
runs KPI analysis, and generates a complete PowerPoint presentation.
"""

import os
import sys
import uuid
import json
import logging
import io
import shutil
import stat
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import folium
from sqlalchemy import text

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

from dotenv import load_dotenv

# Load local .env first
LOCAL_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(LOCAL_ENV_PATH):
    load_dotenv(LOCAL_ENV_PATH)

# Add ML-Production root to sys.path
# Add the ML root for direct CLI execution as well as package imports.
ML_PROD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ML_PROD_DIR not in sys.path:
    sys.path.insert(0, ML_PROD_DIR)

from tools.report_engine.load_data_db import load_project_data, filter_known_band_rows, polygon_filter_all_cells
from tools.report_engine.kpi_config import KPI_CONFIG
from tools.report_engine.threshold_resolver import resolve_kpi_ranges
if __package__:
    from .ppt_map_generator import (
        generate_kpi_map,
        generate_categorical_kpi_map,
        generate_poor_region_maps,
        generate_base_route_map,
        generate_handover_map,
        detect_handover_events,
        new_report_map,
        add_fullscreen_css,
        draw_polygon_overlay,
        fit_data_bounds,
        get_df_bounds,
        has_valid_numeric_data,
        has_valid_categorical_data,
        normalize_band_name,
        build_report_band_color_map,
    )
else:
    from ppt_map_generator import (
        generate_kpi_map,
        generate_categorical_kpi_map,
        generate_poor_region_maps,
        generate_base_route_map,
        generate_handover_map,
        detect_handover_events,
        new_report_map,
        add_fullscreen_css,
        draw_polygon_overlay,
        fit_data_bounds,
        get_df_bounds,
        has_valid_numeric_data,
        has_valid_categorical_data,
        normalize_band_name,
        build_report_band_color_map,
    )
from tools.report_engine.playwright_utils import html_to_png
from tools.report_engine.kpi_analysis import run_kpi_analysis
from tools.report_engine.metadata_generator import build_metadata

if __package__:
    from .ppt_automation import PPTAutomator, generate_legend_png
else:
    from ppt_automation import PPTAutomator, generate_legend_png

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# DEFAULT TEMPLATE KPI THRESHOLD RANGES & COLORS
# Extracted directly from default template: Mobility DT-美濃區(After)-20260729.pptx
# (Per user instruction: do NOT take threshold ranges from database tbl_kpi_thresholds)
# ─────────────────────────────────────────────────────────────────

# 1. RSRP Ranges (Slide 6 for 5G RSRP; Slides 11, 12, 13, 14 for 4G RSRP)
TEMPLATE_RSRP_RANGES = [
    {"min": -140.0, "max": -125.0, "color": "#000000", "label": "Below -125.00"},
    {"min": -125.0, "max": -115.0, "color": "#804000", "label": ">= -125.00 to < -115.00"},
    {"min": -115.0, "max": -112.0, "color": "#FF0000", "label": ">= -115.00 to < -112.00"},
    {"min": -112.0, "max": -110.0, "color": "#FF8000", "label": ">= -112.00 to < -110.00"},
    {"min": -110.0, "max": -105.0, "color": "#FFFF00", "label": ">= -110.00 to < -105.00"},
    {"min": -105.0, "max": -100.0, "color": "#69D200", "label": ">= -105.00 to < -100.00"},
    {"min": -100.0, "max":  -95.0, "color": "#00B700", "label": ">= -100.00 to < -95.00"},
    {"min":  -95.0, "max":  -90.0, "color": "#A2DDFD", "label": ">= -95.00 to < -90.00"},
    {"min":  -90.0, "max":  -40.0, "color": "#0080C0", "label": "Above -90.00"},
]

# 2. 5G SINR Ranges (Slide 7)
TEMPLATE_5G_SINR_RANGES = [
    {"min": -20.0, "max":  0.0, "color": "#000000", "label": "Below 0.00"},
    {"min":   0.0, "max":  5.0, "color": "#804040", "label": ">= 0.00 to < 5.00"},
    {"min":   5.0, "max": 10.0, "color": "#FF0000", "label": ">= 5.00 to < 10.00"},
    {"min":  10.0, "max": 15.0, "color": "#FFFF00", "label": ">= 10.00 to < 15.00"},
    {"min":  15.0, "max": 20.0, "color": "#008040", "label": ">= 15.00 to < 20.00"},
    {"min":  20.0, "max": 25.0, "color": "#9FCFFF", "label": ">= 20.00 to < 25.00"},
    {"min":  25.0, "max": 40.0, "color": "#0080FF", "label": "Above 25.00"},
]

# 3. 4G SINR Ranges (Slide 15)
TEMPLATE_4G_SINR_RANGES = [
    {"min": -20.0, "max":  5.0, "color": "#804040", "label": "Below 5.00"},
    {"min":   5.0, "max": 10.0, "color": "#FF0000", "label": ">= 5.00 to < 10.00"},
    {"min":  10.0, "max": 15.0, "color": "#FFFF00", "label": ">= 10.00 to < 15.00"},
    {"min":  15.0, "max": 20.0, "color": "#008000", "label": ">= 15.00 to < 20.00"},
    {"min":  20.0, "max": 25.0, "color": "#8080FF", "label": ">= 20.00 to < 25.00"},
    {"min":  25.0, "max": 40.0, "color": "#0074AC", "label": "Above 25.00"},
]

# 4. 5G MAC DL & App DL Throughput Ranges in kbps (Slide 9 & Slide 18)
TEMPLATE_5G_DL_RANGES = [
    {"min":      0.0, "max":  20000.0, "color": "#804040", "label": "Below 20000.00"},
    {"min":  20000.0, "max":  50000.0, "color": "#FF0000", "label": ">= 20000.00 to < 50000.00"},
    {"min":  50000.0, "max": 100000.0, "color": "#FF8000", "label": ">= 50000.00 to < 100000.00"},
    {"min": 100000.0, "max": 150000.0, "color": "#FFFF00", "label": ">= 100000.00 to < 150000.00"},
    {"min": 150000.0, "max": 200000.0, "color": "#80FF80", "label": ">= 150000.00 to < 200000.00"},
    {"min": 200000.0, "max": 300000.0, "color": "#00FF00", "label": ">= 200000.00 to < 300000.00"},
    {"min": 300000.0, "max": 500000.0, "color": "#00FFFF", "label": ">= 300000.00 to < 500000.00"},
    {"min": 500000.0, "max": 800000.0, "color": "#8080FF", "label": ">= 500000.00 to < 800000.00"},
    {"min": 800000.0, "max": 2000000.0, "color": "#000080", "label": "Above 800000.00"},
]

# 5. 4G MAC DL Throughput Ranges in kbps (Slide 17)
TEMPLATE_4G_MAC_DL_RANGES = [
    {"min":     0.0, "max":  2000.0, "color": "#800000", "label": "Below 2000.00"},
    {"min":  2000.0, "max":  5000.0, "color": "#FF0000", "label": ">= 2000.00 to < 5000.00"},
    {"min":  5000.0, "max": 10000.0, "color": "#FF8040", "label": ">= 5000.00 to < 10000.00"},
    {"min": 10000.0, "max": 15000.0, "color": "#FFFF00", "label": ">= 10000.00 to < 15000.00"},
    {"min": 15000.0, "max": 20000.0, "color": "#80FF00", "label": ">= 15000.00 to < 20000.00"},
    {"min": 20000.0, "max": 30000.0, "color": "#00FF00", "label": ">= 20000.00 to < 30000.00"},
    {"min": 30000.0, "max": 50000.0, "color": "#00FFFF", "label": ">= 30000.00 to < 50000.00"},
    {"min": 50000.0, "max": 80000.0, "color": "#8080FF", "label": ">= 50000.00 to < 80000.00"},
    {"min": 80000.0, "max": 500000.0, "color": "#0000A0", "label": "Above 80000.00"},
]

# 6. Poor DL < 50 Mbps Ranges in kbps (Slide 20)
TEMPLATE_POOR_DL_RANGES = [
    {"min":     0.0, "max": 50000.0, "color": "#FF0000", "label": "Below 50000.00"},
    {"min": 50000.0, "max": 2000000.0, "color": "#008000", "label": "Above 50000.00"},
]


def normalize_tech_name(tech, band=None, network=None):
    """
    Robustly normalizes technology mode for a log row.
    Checks band first (n* -> 5G, B*/L* -> 4G), then network string, then technology string.
    """
    if band is not None:
        band_str = str(band).strip().lower()
        if re.match(r"^n\d+", band_str) or band_str in {
            "n78", "n77", "n41", "n1", "n28", "n3", "n5", "n7",
            "n8", "n20", "n38", "n40", "n66", "n71", "n257", "n258", "n260", "n261"
        }:
            return "5G"
        if re.match(r"^[bl]\d+", band_str) or band_str in {
            "b1", "b2", "b3", "b4", "b5", "b7", "b8", "b12", "b13", "b17", "b18", "b19",
            "b20", "b25", "b26", "b28", "b38", "b39", "b40", "b41",
            "1800", "2600", "700", "2100", "900", "850", "2300", "2500"
        }:
            return "4G"

    if network is not None:
        net_str = str(network).strip().upper()
        if "5G" in net_str or "NR" in net_str or "NSA" in net_str or "SA" in net_str:
            if "LTE ANCHOR" not in net_str:
                return "5G"
        if "4G" in net_str or "LTE" in net_str:
            return "4G"

    if tech is None:
        return "Unknown"

    tech_str = str(tech).strip()
    if tech_str in {
        "000", "00", "Unknown/No Service", "Unknown / No Service",
        "UNKNOWN / NO SERVICE", "Unknown", "undefined", "null",
        "404440", "404011"
    }:
        return "Unknown"

    t = tech_str.upper()
    if "LTE ANCHOR" in t or "LTE-ANCHOR" in t or "LTE_ANCHOR" in t or "ENDC" in t or "EN-DC" in t:
        return "4G" if ("4G" in t or "LTE" in t) else "5G"
    if "5G" in t or "NR" in t or "NSA" in t or "SA" in t:
        return "5G"
    if "LTE" in t or "4G" in t or "4G+" in t:
        return "4G"
    if "3G" in t or "WCDMA" in t or "UMTS" in t or "HSPA" in t:
        return "3G"
    if "2G" in t or "EDGE" in t or "GSM" in t or "GPRS" in t:
        return "2G"

    return tech_str

# ─────────────────────────────────────────────────────────────────
# PIPELINE LOGGER — timestamped, always-flushed terminal output
# ─────────────────────────────────────────────────────────────────

import time as _time_module

def _ts():
    """Return a compact HH:MM:SS.mmm timestamp string."""
    t = _time_module.time()
    ms = int((t % 1) * 1000)
    return _time_module.strftime("%H:%M:%S") + f".{ms:03d}"

def _log(stage: str, msg: str):
    """
    Print a timestamped pipeline log line to the terminal.
    Always flushed immediately so you see it in real time.
    """
    line = f"[{_ts()}][PPT | {stage}] {msg}"
    print(line, flush=True)


DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "Mobility DT-美濃區(After)-20260729.pptx"
)

REPORT_RENDER_WIDTH = 1370
REPORT_RENDER_HEIGHT = 900
REPORT_DEVICE_SCALE = 1
MAX_RENDER_POINTS = 15000

# Technology color palette
TECH_COLORS = {
    "5G":      "#1a6fcc",   # blue
    "4G":      "#2ca02c",   # green
    "3G":      "#ff7f0e",   # orange
    "2G":      "#d62728",   # red
    "Unknown": "#999999",   # grey
}


# ─────────────────────────────────────────────────────────────────
# LEGEND SUPPRESSION HELPER
# ─────────────────────────────────────────────────────────────────

# The folium legend is injected via JavaScript (setTimeout 250ms).
# A CSS-only fix is not enough because the element doesn't exist yet when
# the <head> CSS is parsed. We use a two-layer approach:
#   1. CSS display:none  — fallback for any legend present at parse time
#   2. MutationObserver  — removes .kpi-legend the instant JS creates it
# Both are injected into the HTML file before Playwright opens it.

_HIDE_LEGEND_BLOCK = """<style>
.kpi-legend { display: none !important; }
</style>
<script>
(function() {
    function removeLegends() {
        document.querySelectorAll('.kpi-legend').forEach(function(el) {
            el.parentNode && el.parentNode.removeChild(el);
        });
    }
    // Remove any already-present legends
    removeLegends();
    // Watch for future JS-injected legends (folium uses setTimeout 250ms)
    var obs = new MutationObserver(function(mutations) {
        mutations.forEach(function(m) {
            m.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) {
                    if (node.classList && node.classList.contains('kpi-legend')) {
                        node.parentNode && node.parentNode.removeChild(node);
                    } else {
                        node.querySelectorAll && node.querySelectorAll('.kpi-legend').forEach(function(el) {
                            el.parentNode && el.parentNode.removeChild(el);
                        });
                    }
                }
            });
        });
    });
    obs.observe(document.documentElement, { childList: true, subtree: true });
})();
</script>"""


def _suppress_legend_in_html(html_path: str) -> None:
    """
    Inject CSS + MutationObserver into an already-saved HTML file to
    remove the folium .kpi-legend div before Playwright screenshots it.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    if _HIDE_LEGEND_BLOCK not in content:
        # Inject just before </body> so the script runs after folium's own scripts
        if "</body>" in content:
            content = content.replace("</body>", _HIDE_LEGEND_BLOCK + "\n</body>", 1)
        else:
            content += _HIDE_LEGEND_BLOCK
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)


def _html_to_png_no_legend(html_path: str, png_path: str) -> None:
    """Render HTML to PNG after injecting legend-suppression CSS + MutationObserver."""
    import os as _os, time as _t
    label = _os.path.basename(png_path)
    print(f"[{_ts()}][Playwright] START rendering -> {label}", flush=True)
    _t0 = _t.time()
    _suppress_legend_in_html(html_path)
    html_to_png(
        html_path, png_path,
        width=REPORT_RENDER_WIDTH,
        height=REPORT_RENDER_HEIGHT,
        device_scale_factor=REPORT_DEVICE_SCALE,
    )
    print(f"[{_ts()}][Playwright] DONE  rendering -> {label} ({_t.time()-_t0:.1f}s)", flush=True)




# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _subsample(df, max_pts=MAX_RENDER_POINTS, key_col=None):
    if len(df) <= max_pts:
        return df
    if key_col and key_col in df.columns:
        frac = max_pts / len(df)
        return df.groupby(key_col, group_keys=False).apply(
            lambda g: g.sample(frac=min(frac, 1.0), random_state=42)
        ).head(max_pts)
    return df.sample(n=max_pts, random_state=42)


def _filter_5g(df):
    """Return rows where normalised technology is 5G."""
    df = df.copy()
    df["__tech"] = df.apply(
        lambda r: normalize_tech_name(r.get("technology"), band=r.get("band"), network=r.get("network")), axis=1
    )
    return df[df["__tech"] == "5G"].copy()


def _filter_4g(df):
    """Return rows where normalised technology is 4G."""
    df = df.copy()
    df["__tech"] = df.apply(
        lambda r: normalize_tech_name(r.get("technology"), band=r.get("band"), network=r.get("network")), axis=1
    )
    return df[df["__tech"] == "4G"].copy()

def extract_mac_tpt_from_extra_json(df, key):
    """
    Extract a numeric KPI field from the extra_json column.

    The extra_json column stores per-row diagnostic data as a JSON string, e.g.:
        '{"nr_mac_dl_mbps": 45.2, "lte_mac_dl_mbps": 0, ...}'

    Args:
        df  : DataFrame that has an 'extra_json' column
        key : field name to extract  (e.g. "nr_mac_dl_mbps" for 5G MAC DL,
                                            "lte_mac_dl_mbps" for 4G MAC DL)

    Returns:
        pd.Series of float values (NaN where missing / unparseable)
    """
    import json

    result = []
    for val in df.get("extra_json", [None] * len(df)):
        try:
            if val is None or (isinstance(val, float)):
                result.append(float("nan"))
                continue
            parsed = json.loads(val) if isinstance(val, str) else val
            result.append(float(parsed.get(key, float("nan"))))
        except Exception:
            result.append(float("nan"))
    import pandas as pd
    return pd.Series(result, index=df.index)



def parse_locked_bands(locked_bands_input):
    """
    Parse user-specified locked bands into standardized 3GPP band names.
    Supports single or multiple bands:
      '1800,2100'   -> {'B3', 'B1'}
      'L1800,L2600' -> {'B3', 'B7', 'B38', 'B41'}
      'B3,B1,B28'   -> {'B3', 'B1', 'B28'}
      'all' / None  -> None (auto-detect all available bands from data)
    """
    if not locked_bands_input:
        return None
    raw_str = str(locked_bands_input).strip().lower()
    if raw_str in ("", "all", "none", "*", "auto", "default"):
        return None

    if isinstance(locked_bands_input, (list, tuple, set)):
        items = [str(x).strip() for x in locked_bands_input]
    else:
        items = [s.strip() for s in str(locked_bands_input).split(",") if s.strip()]

    BAND_ALIAS_MAP = {
        "1800": ["B3"],
        "L1800": ["B3"],
        "B3": ["B3"],
        "BAND3": ["B3"],
        "2100": ["B1"],
        "L2100": ["B1"],
        "B1": ["B1"],
        "BAND1": ["B1"],
        "2600": ["B7", "B38", "B41"],
        "L2600": ["B7", "B38", "B41"],
        "B7": ["B7"],
        "B38": ["B38"],
        "B41": ["B41"],
        "BAND7": ["B7"],
        "700": ["B28"],
        "L700": ["B28"],
        "900": ["B8"],
        "L900": ["B8"],
        "B28": ["B28"],
        "B8": ["B8"],
    }

    selected = set()
    for item in items:
        key = item.upper().replace(" ", "")
        if key in BAND_ALIAS_MAP:
            selected.update(BAND_ALIAS_MAP[key])
        else:
            norm = normalize_band_name(item)
            if norm and norm != "Unknown":
                selected.add(norm)
    return selected


def _enrich_df_with_mac_tpt(df):
    """
    Add __nr_mac_dl and __lte_mac_dl columns by parsing extra_json ONLY.

    No fallback to dl_tpt or any other column.
    If extra_json is missing or the key is absent for a row, the value stays NaN.
    Maps and legends for these KPIs will show ONLY rows with actual MAC data.
    """
    import pandas as pd
    df = df.copy()

    if "extra_json" in df.columns:
        df["__nr_mac_dl"]  = extract_mac_tpt_from_extra_json(df, "nr_mac_dl_mbps")
        df["__lte_mac_dl"] = extract_mac_tpt_from_extra_json(df, "lte_mac_dl_mbps")
    else:
        # extra_json column completely absent — both columns are all-NaN (no data)
        df["__nr_mac_dl"]  = float("nan")
        df["__lte_mac_dl"] = float("nan")

    # Add throughput in kbps for template-aligned map rendering and legends
    df["__nr_mac_dl_kbps"]  = df["__nr_mac_dl"] * 1000.0
    df["__lte_mac_dl_kbps"] = df["__lte_mac_dl"] * 1000.0
    if "dl_tpt" in df.columns:
        df["__app_dl_kbps"] = pd.to_numeric(df["dl_tpt"], errors="coerce") * 1000.0

    return df






def build_numeric_legend_items(values, ranges):
    """
    Builds legend items for numeric KPIs matching the default PPT template format:
    Row 0: 'Below <max> (<count>) <pct>%'
    Row i: '>= <min> to < <max> (<count>) <pct>%'
    Row N: 'Above <min> (<count>) <pct>%'
    """
    clean_vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().tolist()
    total_count = len(clean_vals)
    items = []
    if not ranges:
        return items

    for idx, r in enumerate(ranges):
        is_first = (idx == 0)
        is_last = (idx == len(ranges) - 1)
        r_min = float(r["min"])
        r_max = float(r["max"])

        if is_first and is_last:
            count = total_count
        elif is_first:
            count = sum(1 for v in clean_vals if v < r_max)
        elif is_last:
            count = sum(1 for v in clean_vals if v >= r_min)
        else:
            count = sum(1 for v in clean_vals if r_min <= v < r_max)

        pct = (count / total_count * 100.0) if total_count > 0 else 0.0

        if is_first and not is_last:
            text = f"Below {r_max:.2f} ({count}) {pct:.1f}%"
        elif is_last and not is_first:
            text = f"Above {r_min:.2f} ({count}) {pct:.1f}%"
        else:
            text = f">= {r_min:.2f} to < {r_max:.2f} ({count}) {pct:.1f}%"

        items.append({
            "text": text,
            "color": r["color"],
            "count": count,
            "pct": pct,
        })
    return items


# ─────────────────────────────────────────────────────────────────
# CUSTOM MAP GENERATORS  (all without folium legend overlay)
# ─────────────────────────────────────────────────────────────────


def generate_blank_basemap(report_df, output_png, tmp_html, polygon_wkt=None, fixed_bounds=None):
    """
    Generate a clean basemap of the project region with NO route / data points.
    Used for slides where a band was not tested / locked out (e.g. Slide 11, 13, 14).
    """
    df = report_df.dropna(subset=["lat", "lon"]).copy()
    fmap = new_report_map()
    add_fullscreen_css(fmap)
    draw_polygon_overlay(fmap, polygon_wkt)
    fit_data_bounds(fmap, fixed_bounds if fixed_bounds is not None else (df if not df.empty else report_df), reserve_legend_space=False)
    fmap.save(tmp_html)
    try:
        _html_to_png_no_legend(tmp_html, output_png)
        print(f"[PPT Pipeline] Clean blank basemap -> {os.path.basename(output_png)}")
        return True
    except Exception as e:
        print(f"[PPT Pipeline] Error rendering blank basemap: {e}")
        return False


def generate_technology_mode_map(report_df, output_png, tmp_html, polygon_wkt=None, fixed_bounds=None):
    """
    Technology-mode map coloured by technology (5G/4G/3G/2G).
    For NSA dual-connectivity (where 5G NR and 4G LTE Anchor coexist at the same coordinates),
    renders 4G with an outer green ring (radius=6, color="#2ca02c", weight=2.5, fill=True, fill_opacity=0.35)
    and 5G with an inner blue core (radius=3.5, color="#1a6fcc", weight=1.5, fill=True, fill_opacity=0.95),
    ensuring both blue and green are distinctly visible on the map without covering each other.
    """
    df = report_df.dropna(subset=["lat", "lon"]).copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df["__tech"] = df.apply(
        lambda r: normalize_tech_name(r.get("technology"), band=r.get("band"), network=r.get("network")), axis=1
    )

    if df.empty:
        print("[PPT Pipeline] No data for technology mode map")
        return False

    fmap = new_report_map()
    add_fullscreen_css(fmap)
    draw_polygon_overlay(fmap, polygon_wkt)

    df_4g = df[df["__tech"] == "4G"]
    df_5g = df[df["__tech"] == "5G"]
    df_other = df[~df["__tech"].isin(["4G", "5G"])]

    # 1. Other technologies (3G / 2G / Unknown)
    for tech in df_other["__tech"].unique():
        color = TECH_COLORS.get(tech, "#999999")
        sub_df = _subsample(df_other[df_other["__tech"] == tech], key_col="__tech")
        for _, r in sub_df.iterrows():
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=4, color=color, fill=True, fill_color=color, fill_opacity=0.85,
            ).add_to(fmap)

    # 2. 4G points: outer marker / ring (radius=6, weight=2.5, green #2ca02c)
    if not df_4g.empty:
        sub_4g = _subsample(df_4g, key_col="__tech")
        for _, r in sub_4g.iterrows():
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=6, color="#2ca02c", weight=2.5, fill=True, fill_color="#2ca02c", fill_opacity=0.35,
            ).add_to(fmap)

    # 3. 5G points: inner core (radius=3.5, blue #1a6fcc)
    if not df_5g.empty:
        sub_5g = _subsample(df_5g, key_col="__tech")
        for _, r in sub_5g.iterrows():
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=3.5, color="#1a6fcc", weight=1.5, fill=True, fill_color="#1a6fcc", fill_opacity=0.95,
            ).add_to(fmap)

    fit_data_bounds(fmap, fixed_bounds if fixed_bounds is not None else df, reserve_legend_space=False)
    fmap.save(tmp_html)

    try:
        _html_to_png_no_legend(tmp_html, output_png)
        print(f"[PPT Pipeline] Technology mode map -> {os.path.basename(output_png)}")
        return True
    except Exception as e:
        print(f"[PPT Pipeline] Error rendering technology mode map: {e}")
        return False


def generate_5g_kpi_map(report_df, kpi_col, color_func, ranges, output_png, tmp_html, title, polygon_wkt=None, fixed_bounds=None):
    """
    KPI map (RSRP, SINR, or DL) filtered to 5G-only rows.
    Color ranges come from resolve_kpi_ranges (thresholds DB).
    No folium legend injected.
    """
    if kpi_col in ("__nr_mac_dl", "__nr_mac_dl_kbps"):
        df_5g = report_df.dropna(subset=["lat", "lon", kpi_col]).copy()
        if not df_5g.empty and "band" in df_5g.columns:
            # 5G NR MAC DL must strictly come from 5G NR carriers (n*) — NO fallback to LTE anchor bands
            nr_b = df_5g[df_5g["band"].astype(str).str.lower().str.startswith("n")]
            df_5g = nr_b
        else:
            df_5g = pd.DataFrame()
    else:
        df_5g = _filter_5g(report_df).dropna(subset=["lat", "lon", kpi_col]).copy()

    df_5g[kpi_col] = pd.to_numeric(df_5g[kpi_col], errors="coerce")
    df_5g = df_5g.dropna(subset=[kpi_col])

    if df_5g.empty:
        print(f"[PPT Pipeline] No data for {title} — returning False (no fallback)")
        return False

    print(f"[PPT Pipeline] {title} | 5G samples: {len(df_5g)}")

    try:
        generate_kpi_map(
            df=df_5g, kpi_column=kpi_col, color_func=color_func,
            ranges=ranges, output_html=tmp_html, polygon_wkt=polygon_wkt,
            fixed_bounds=fixed_bounds,
        )
        _html_to_png_no_legend(tmp_html, output_png)
        print(f"[PPT Pipeline] {title} map -> {os.path.basename(output_png)}")
        return True
    except Exception as e:
        print(f"[PPT Pipeline] Error rendering {title} map: {e}")
        return False


def generate_4g_kpi_map(report_df, kpi_col, color_func, ranges, output_png, tmp_html, title, polygon_wkt=None, band_filter=None, fixed_bounds=None):
    """
    KPI map (RSRP, SINR, or DL) filtered to 4G rows or specific LTE band rows.
    When band_filter is provided (e.g. ['B1'] for L2100, ['B3'] for L1800), filters all rows matching that LTE band.
    """
    if band_filter and "band" in report_df.columns:
        df_target = report_df.dropna(subset=["lat", "lon", kpi_col]).copy()
        df_target["__norm_band"] = df_target["band"].apply(normalize_band_name)
        df_target = df_target[df_target["__norm_band"].isin(band_filter)]
    elif kpi_col in ("__lte_mac_dl", "__lte_mac_dl_kbps"):
        df_target = report_df.dropna(subset=["lat", "lon", kpi_col]).copy()
        if not df_target.empty and "band" in df_target.columns:
            lte_b = df_target[df_target["band"].astype(str).str.upper().str.startswith("B")]
            if not lte_b.empty:
                df_target = lte_b
    else:
        df_target = _filter_4g(report_df).dropna(subset=["lat", "lon", kpi_col]).copy()

    df_target[kpi_col] = pd.to_numeric(df_target[kpi_col], errors="coerce")
    df_target = df_target.dropna(subset=[kpi_col])

    if df_target.empty:
        print(f"[PPT Pipeline] No data for {title} — returning False (no fallback)")
        return False

    print(f"[PPT Pipeline] {title} | samples: {len(df_target)}")
    try:
        generate_kpi_map(
            df=df_target, kpi_column=kpi_col, color_func=color_func,
            ranges=ranges, output_html=tmp_html, polygon_wkt=polygon_wkt,
            fixed_bounds=fixed_bounds,
        )
        _html_to_png_no_legend(tmp_html, output_png)
        print(f"[PPT Pipeline] {title} map -> {os.path.basename(output_png)}")
        return True
    except Exception as e:
        print(f"[PPT Pipeline] Error rendering {title} map: {e}")
        return False


# Fixed color palette for ca_cc values (1 = single carrier, 2 = 2CC, etc.) matching default template
CA_CC_COLORS = {
    "1": "#80FF00",   # 1CC (Non-CA) - template lime green
    "2": "#FFFF00",   # 2CC - template yellow
    "3": "#008040",   # 3CC - template dark green
    "4": "#FF8000",   # 4CC - orange
    "5": "#FF0000",   # 5CC - red
    "6": "#00BCD4",   # cyan
}


def generate_ca_map(report_df, output_png, tmp_html, polygon_wkt=None, band_filter=None, fixed_bounds=None):
    """
    CA (Carrier Aggregation) configuration map.
    Reads 'ca_cc' (component-carrier count) from extra_json ONLY.
    Each cc value (1, 2, 3 …) gets a distinct fixed color from CA_CC_COLORS.

    Returns (ca_df, CA_CC_COLORS) on success.
    Returns False if no ca_cc data is found — NO fallback to band map.
    Source: tbl_network_log.extra_json -> 'ca_cc' key
    """
    ca_df = pd.DataFrame()

    # Filter strictly to 4G / LTE technology rows so companion 5G rows do not double-count 4G CA.
    # Note: Do not do a raw technology != '5G' check because 4G LTE Anchor rows in NSA sessions
    # have technology='5G' logged while network='4G (LTE Anchor - NSA)' and band='B3'/'B7'.
    # Use _filter_4g() which robustly evaluates band, network, and technology.
    df_input = _filter_4g(report_df)
    if df_input.empty:
        df_input = report_df.copy()
        if "band" in df_input.columns:
            df_input = df_input[~df_input["band"].astype(str).str.lower().str.startswith("n")]

    # ── Extract ca_cc from extra_json ─────────────────────────────────────
    if "extra_json" in df_input.columns and df_input["extra_json"].notna().any():
        def _extract_ca_cc(val):
            if pd.isna(val) or val is None:
                return None
            try:
                d = json.loads(val) if isinstance(val, str) else val
                if isinstance(d, dict):
                    # Genuine 4G Carrier Aggregation ONLY (ca_cc >= 2 or ca_cc_count >= 2)
                    cc = d.get("ca_cc") or d.get("ca_cc_count")
                    if cc is not None:
                        try:
                            cc_int = int(cc)
                            if cc_int >= 2:
                                return str(cc_int)   # normalise -> "2", "3", etc.
                        except (ValueError, TypeError):
                            pass
            except Exception:
                pass
            return None

        tmp = df_input.copy()
        tmp["__ca_label"] = tmp["extra_json"].apply(_extract_ca_cc)
        ca_df = tmp.dropna(subset=["lat", "lon", "__ca_label"]).copy()
        ca_df["lat"] = pd.to_numeric(ca_df["lat"], errors="coerce")
        ca_df["lon"] = pd.to_numeric(ca_df["lon"], errors="coerce")
        ca_df = ca_df.dropna(subset=["lat", "lon"])

    if band_filter and not ca_df.empty and "band" in ca_df.columns:
        ca_df["__norm_band"] = ca_df["band"].apply(normalize_band_name)
        ca_df = ca_df[ca_df["__norm_band"].isin(band_filter)].copy()

    if ca_df.empty:
        print("[PPT Pipeline] CA map: no ca_cc data found in extra_json for requested bands — map/legend will be blank")
        return False

    counts = ca_df["__ca_label"].value_counts()
    print(f"[PPT Pipeline] CA map | ca_cc from extra_json | samples: {len(ca_df)} | {dict(counts)}")

    fmap = new_report_map()
    add_fullscreen_css(fmap)
    draw_polygon_overlay(fmap, polygon_wkt)

    for _, row in _subsample(ca_df, key_col="__ca_label").iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=CA_CC_COLORS.get(row["__ca_label"], "#888888"),
            fill=True, fill_opacity=0.85,
        ).add_to(fmap)

    fit_data_bounds(fmap, fixed_bounds if fixed_bounds is not None else ca_df, reserve_legend_space=False)
    fmap.save(tmp_html)
    try:
        _html_to_png_no_legend(tmp_html, output_png)
        print(f"[PPT Pipeline] CA map -> {os.path.basename(output_png)}")
        return ca_df, CA_CC_COLORS
    except Exception as e:
        print(f"[PPT Pipeline] Error rendering CA map: {e}")
        return False


def generate_nr_ca_map(report_df, output_png, tmp_html, polygon_wkt=None, fixed_bounds=None):
    """
    5G NR CA (Carrier Aggregation) configuration map.
    Reads 'nr_ca_count', 'nr_ca_cc_count', or 'nr_ca_cc' from extra_json.
    Each cc value (2, 3 …) gets a distinct fixed color from CA_CC_COLORS.
    Returns (ca_df, CA_CC_COLORS) on success, or False.
    """
    if "extra_json" not in report_df.columns or report_df["extra_json"].dropna().empty:
        return False

    def _extract_nr_ca(row):
        val = row.get("extra_json")
        if pd.isna(val) or val is None:
            return None
        net_val = str(row.get("network", "")).strip().upper()
        tech_val = str(row.get("technology", "")).strip().upper()
        band_val = str(row.get("band", "")).strip().lower()
        is_5g = ("5G" in net_val or "NR" in net_val or tech_val == "5G" or band_val.startswith("n"))
        if not is_5g:
            return None
        try:
            d = json.loads(val) if isinstance(val, str) else val
            if isinstance(d, dict):
                # Genuine 5G NR Carrier Aggregation ONLY (nr_ca_count, nr_ca_cc_count, nr_ca_cc > 1)
                # Do NOT check ca_cc or ca_cc_count (those are LTE carrier aggregation counts)
                for k in ["nr_ca_count", "nr_ca_cc_count", "nr_ca_cc"]:
                    c = d.get(k)
                    if c is not None:
                        try:
                            c_int = int(c)
                            if c_int > 1:
                                return str(c_int)
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass
        return None

    tmp = report_df.copy()
    tmp["__nr_ca_label"] = tmp.apply(_extract_nr_ca, axis=1)
    ca_df = tmp.dropna(subset=["lat", "lon", "__nr_ca_label"]).copy()
    ca_df["lat"] = pd.to_numeric(ca_df["lat"], errors="coerce")
    ca_df["lon"] = pd.to_numeric(ca_df["lon"], errors="coerce")
    ca_df = ca_df.dropna(subset=["lat", "lon"])

    if ca_df.empty:
        print("[PPT Pipeline] 5G NR CA map: no nr_ca_count data found in extra_json — map/legend will be blank")
        return False

    counts = ca_df["__nr_ca_label"].value_counts()
    print(f"[PPT Pipeline] 5G NR CA map | samples: {len(ca_df)} | {dict(counts)}")

    fmap = new_report_map()
    add_fullscreen_css(fmap)
    draw_polygon_overlay(fmap, polygon_wkt)

    for _, row in _subsample(ca_df, key_col="__nr_ca_label").iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=CA_CC_COLORS.get(row["__nr_ca_label"], "#888888"),
            fill=True, fill_opacity=0.85,
        ).add_to(fmap)

    fit_data_bounds(fmap, fixed_bounds if fixed_bounds is not None else ca_df, reserve_legend_space=False)
    fmap.save(tmp_html)
    try:
        _html_to_png_no_legend(tmp_html, output_png)
        print(f"[PPT Pipeline] 5G NR CA map -> {os.path.basename(output_png)}")
        return ca_df, CA_CC_COLORS
    except Exception as e:
        print(f"[PPT Pipeline] Error rendering 5G NR CA map: {e}")
        return False






def generate_poor_dl_50_map(report_df, output_png, tmp_html, threshold=50, polygon_wkt=None, fixed_bounds=None):
    """
    Map showing ONLY points where dl_tpt < threshold Mbps.
    No legend injected.
    """
    if "dl_tpt" not in report_df.columns:
        print("[PPT Pipeline] Missing 'dl_tpt' column for APP DL < 50Mbps map")
        return False

    df_geo = report_df.dropna(subset=["lat", "lon", "dl_tpt"]).copy()
    df_geo["dl_tpt"] = pd.to_numeric(df_geo["dl_tpt"], errors="coerce")
    poor_df = df_geo[df_geo["dl_tpt"] < threshold].dropna(subset=["dl_tpt"])

    print(f"[PPT Pipeline] APP DL < {threshold}Mbps | Total: {len(df_geo)} | Poor: {len(poor_df)}")

    fmap = new_report_map()
    add_fullscreen_css(fmap)
    for _, r in poor_df.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=4, color="#e31a1c", fill=True, fill_opacity=0.85,
        ).add_to(fmap)

    draw_polygon_overlay(fmap, polygon_wkt)
    fit_data_bounds(fmap, fixed_bounds if fixed_bounds is not None else (df_geo if not df_geo.empty else report_df), reserve_legend_space=False)
    fmap.save(tmp_html)

    try:
        _html_to_png_no_legend(tmp_html, output_png)
        return True
    except Exception as e:
        print(f"[PPT Pipeline] Error rendering poor DL map: {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def generate_ppt_for_project(
    project_id: int,
    session_ids=None,
    user_id=None,
    template_path=None,
    output_path=None,
    region=None,
    country_code=None,
    db_engine=None,
    locked_bands=None,
):
    """Generates a complete PowerPoint PPTX report for a given project_id."""
    if template_path is None:
        template_path = DEFAULT_TEMPLATE_PATH
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), f"Mobility_DT_Project_{project_id}.pptx")

    def _safe_rmtree(path):
        if not os.path.exists(path):
            return
        def _onerror(func, p, exc_info):
            try:
                os.chmod(p, stat.S_IWRITE)
                func(p)
            except Exception:
                pass
        try:
            shutil.rmtree(path, onerror=_onerror)
        except Exception:
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass

    # Clean up stale temp directories from previous runs
    parent_tmp = os.path.join(os.path.dirname(__file__), "tmp_ppt_render")
    if os.path.exists(parent_tmp):
        for item in os.listdir(parent_tmp):
            p = os.path.join(parent_tmp, item)
            if os.path.isdir(p):
                try:
                    _safe_rmtree(p)
                except Exception:
                    pass

    run_id = str(uuid.uuid4())[:8]
    tmp_dir = os.path.join(os.path.dirname(__file__), "tmp_ppt_render", run_id)
    html_dir = os.path.join(tmp_dir, "html")
    kpi_maps_dir = os.path.join(tmp_dir, "kpi_maps")
    kpi_analysis_dir = os.path.join(tmp_dir, "kpi_analysis")
    for d in [html_dir, kpi_maps_dir, kpi_analysis_dir]:
        os.makedirs(d, exist_ok=True)

    def _cleanup_tmp():
        if os.path.exists(tmp_dir):
            try:
                _safe_rmtree(tmp_dir)
                _log("Cleanup", f"Cleaned up and deleted temp render directory: {tmp_dir}")
            except Exception as e:
                _log("Cleanup", f"Warning: could not delete temp directory {tmp_dir}: {e}")

    import atexit
    atexit.register(_cleanup_tmp)

    _pipeline_start = _time_module.time()
    print("", flush=True)
    print("=" * 70, flush=True)
    _log("PIPELINE START", f"project_id={project_id} | run_id={run_id}")
    _log("PIPELINE START", f"template={os.path.basename(template_path)}")
    _log("PIPELINE START", f"output={output_path}")
    print("=" * 70, flush=True)

    # ── Load Data ─────────────────────────────────────────────────────────
    locked_set = parse_locked_bands(locked_bands)
    if locked_set:
        _log("Load", f"Multi-band lock: {sorted(list(locked_set))}")
    _log("Load", f"Querying DB for project_id={project_id} region={region} country_code={country_code} ...")
    _t_db = _time_module.time()
    try:
        raw_df, filtered_df, project_meta = load_project_data(
            project_id, region=region, country_code=country_code
        )
        _log("Load", f"DB query complete in {_time_module.time()-_t_db:.1f}s")
    except Exception as e:
        if country_code:
            _log("Load", f"WARNING: not found in '{country_code}' DB — retrying without country_code ...")
            raw_df, filtered_df, project_meta = load_project_data(
                project_id, region=region, country_code=None
            )
            _log("Load", f"DB fallback query complete in {_time_module.time()-_t_db:.1f}s")
            country_code = None
        else:
            raise e



    _log("Load", f"Raw DB rows: {len(raw_df)} | Filtered rows: {len(filtered_df)}")

    if user_id is None:
        user_id = project_meta.get("user_id") or project_meta.get("User_id") or 0
    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        user_id = 0

    _log("Load", f"user_id={user_id} | project_name={project_meta.get('project_name', 'N/A')}")

    polygon_wkt = project_meta.get("region")
    _log("Load", f"Polygon WKT present: {bool(polygon_wkt)}")

    if not locked_set:
        meta_band = project_meta.get("band") or project_meta.get("locked_bands")
        if meta_band:
            locked_set = parse_locked_bands(meta_band)
            if locked_set:
                _log("Load", f"Locked band detected from project metadata: {sorted(list(locked_set))}")
    locked_lte = {b for b in locked_set if not b.lower().startswith("n")} if locked_set else set()
    locked_nr  = {b for b in locked_set if b.lower().startswith("n")} if locked_set else set()

    if filtered_df.empty:
        raise ValueError(f"No data rows for project_id={project_id}")

    if session_ids is not None:
        if isinstance(session_ids, (int, str)):
            s_list = [int(s.strip()) for s in str(session_ids).split(",") if s.strip().isdigit()]
        else:
            s_list = [int(s) for s in session_ids]
        if s_list and "session_id" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["session_id"].isin(s_list)]
            _log("Load", f"Session filter applied -> {len(filtered_df)} rows remaining")

    _log("Load", "Applying filter_known_band_rows ...")
    report_df = filter_known_band_rows(filtered_df)
    if report_df.empty:
        _log("Load", "WARNING: filter_known_band_rows returned 0 rows — falling back to full filtered_df")
        report_df = filtered_df.reset_index(drop=True)

    _log("Load", f"Report dataframe ready: {len(report_df)} rows (user_id={user_id})")
    _log("Load", f"Columns: {list(report_df.columns)}")

    # Ensure extra_json is populated from DB if not already present or empty
    if "extra_json" not in report_df.columns or report_df["extra_json"].dropna().empty:
        try:
            from tools.report_engine.db import _connect
            c = _connect(region=region, country_code=country_code)
            try:
                sids = [int(s) for s in report_df["session_id"].dropna().unique()]
                if sids:
                    from sqlalchemy import bindparam
                    q = text("""
                        SELECT id, extra_json 
                        FROM tbl_network_log 
                        WHERE session_id IN :sids AND extra_json IS NOT NULL AND extra_json != ''
                    """).bindparams(bindparam("sids", expanding=True))
                    rows = c.execute(q, {"sids": sids}).fetchall()
                    if rows:
                        ej_map = dict(rows)
                        report_df["extra_json"] = report_df["id"].map(ej_map)
                        _log("Load", f"Enriched {report_df['extra_json'].notna().sum()} rows with extra_json from database")
            finally:
                c.close()
        except Exception as e:
            _log("Load", f"Warning: could not enrich extra_json: {e}")

    # Propagate extra_json across companion rows by timestamp.
    # In 5G NSA (dual-connectivity), modem diagnostics (nr_mac_dl_mbps, lte_mac_dl_mbps,
    # nr_dl_rb, dl_mcs, etc.) are recorded on the companion LTE Anchor row at the exact same
    # timestamp and coordinates, while the 5G NR row (n78) has extra_json as NULL.
    # Propagating extra_json to companion 5G rows ensures 5G MAC DL map & legend have complete data.
    if "timestamp" in report_df.columns and "extra_json" in report_df.columns:
        has_ej_df = report_df[report_df["extra_json"].notna() & report_df["extra_json"].ne("")].copy()
        if not has_ej_df.empty:
            ts_ej_map = {}
            for _, r in has_ej_df.iterrows():
                t = r["timestamp"]
                if t not in ts_ej_map:
                    ts_ej_map[t] = r["extra_json"]

            dt_ej_map = {}
            for t_val, ej_v in ts_ej_map.items():
                try:
                    dt_ej_map[pd.to_datetime(t_val)] = ej_v
                except Exception:
                    pass

            def _get_companion_ej(row):
                val = row.get("extra_json")
                if pd.notna(val) and val:
                    return val
                raw_ts = row.get("timestamp")
                if raw_ts in ts_ej_map:
                    return ts_ej_map[raw_ts]
                try:
                    row_dt = pd.to_datetime(raw_ts)
                    for dt_k, ej_val in dt_ej_map.items():
                        if abs((dt_k - row_dt).total_seconds()) <= 2.0:
                            return ej_val
                except Exception:
                    pass
                return val

            _before_prop = int(report_df["extra_json"].notna().sum())
            report_df["extra_json"] = report_df.apply(_get_companion_ej, axis=1)
            _after_prop = int(report_df["extra_json"].notna().sum())
            _log("Load", f"Propagated companion extra_json to 5G rows: {_before_prop} -> {_after_prop} rows")

    # ── 1. Filtered Subsets for 5G and 4G (before enrichment) ────────────
    # Apply strict locked bands filter directly to report_df so unselected bands never leak into any slide/map/legend
    if locked_lte and "band" in report_df.columns:
        def _band_allowed(row):
            b = normalize_band_name(row.get("band"))
            if b.startswith("B"):
                return b in locked_lte
            if b.startswith("n"):
                return (not locked_nr) or (b in locked_nr)
            return True
        _before_len = len(report_df)
        report_df = report_df[report_df.apply(_band_allowed, axis=1)].copy()
        _log("Data", f"Strict band lock {sorted(list(locked_lte))} applied to report_df: {len(report_df)}/{_before_len} rows kept")

    _log("Data", "Splitting into 5G / 4G subsets ...")
    df_5g = _filter_5g(report_df)
    df_4g = _filter_4g(report_df)
    if locked_lte and "band" in df_4g.columns:
        _norm_4g = df_4g["band"].apply(normalize_band_name)
        _before_len = len(df_4g)
        df_4g = df_4g[_norm_4g.isin(locked_lte)].copy()
        _log("Data", f"Locked LTE bands {sorted(list(locked_lte))} applied -> df_4g filtered from {_before_len} to {len(df_4g)} rows")
    if locked_nr and "band" in df_5g.columns:
        _norm_5g = df_5g["band"].apply(normalize_band_name)
        _before_len_nr = len(df_5g)
        df_5g = df_5g[_norm_5g.isin(locked_nr)].copy()
        _log("Data", f"Locked NR bands {sorted(list(locked_nr))} applied -> df_5g filtered from {_before_len_nr} to {len(df_5g)} rows")
    _log("Data", f"5G samples: {len(df_5g)} | 4G samples: {len(df_4g)} | Total: {len(report_df)}")

    # ── 2. Enrich with MAC tpt from extra_json (needed for maps + legends) ─
    _log("Enrich", "Extracting nr_mac_dl_mbps and lte_mac_dl_mbps from extra_json ...")
    report_df = _enrich_df_with_mac_tpt(report_df)
    _log("Enrich", "  report_df enriched")
    df_5g     = _enrich_df_with_mac_tpt(df_5g)
    _log("Enrich", "  df_5g enriched")
    df_4g     = _enrich_df_with_mac_tpt(df_4g)
    _log("Enrich", "  df_4g enriched")
    nr_valid  = int(df_5g["__nr_mac_dl"].notna().sum())  if "__nr_mac_dl"  in df_5g.columns else 0
    lte_valid = int(df_4g["__lte_mac_dl"].notna().sum()) if "__lte_mac_dl" in df_4g.columns else 0
    _log("Enrich", f"5G MAC DL (nr_mac_dl_mbps) valid rows: {nr_valid} / {len(df_5g)}")
    _log("Enrich", f"4G MAC DL (lte_mac_dl_mbps) valid rows: {lte_valid} / {len(df_4g)}")
    if nr_valid == 0:
        _log("Enrich", "WARNING: no nr_mac_dl_mbps in extra_json — 5G MAC DL slide 9 will be blank (no fallback)")
    if lte_valid == 0:
        _log("Enrich", "WARNING: no lte_mac_dl_mbps in extra_json — 4G MAC DL slide 17 will be blank (no fallback)")


    # ── 3. Template Benchmark Threshold Ranges (from default PPT template) ─
    _log("Thresholds", "Using default template KPI threshold ranges (Mobility DT-美濃區(After)-20260729.pptx) ...")
    rsrp_ranges    = TEMPLATE_RSRP_RANGES
    sinr_5g_ranges = TEMPLATE_5G_SINR_RANGES
    sinr_4g_ranges = TEMPLATE_4G_SINR_RANGES
    dl_5g_ranges   = TEMPLATE_5G_DL_RANGES
    dl_4g_ranges   = TEMPLATE_4G_MAC_DL_RANGES
    dl_app_ranges  = TEMPLATE_5G_DL_RANGES

    rsrp_cfg = KPI_CONFIG.get("RSRP", {})
    sinr_cfg = KPI_CONFIG.get("SINR", {})
    dl_cfg   = KPI_CONFIG.get("DL",   {})

    # ── Unified Route Viewport Bounds across ALL slides ──────────────────
    overall_bounds = None
    if not report_df.empty and "lat" in report_df.columns and "lon" in report_df.columns:
        try:
            overall_bounds = get_df_bounds(report_df)
            _log("Bounds", f"Computed unified route bounds for all map slides: {overall_bounds}")
        except Exception as e:
            _log("Bounds", f"Could not compute unified route bounds: {e}")

    # ── 4. Base Route Map ─────────────────────────────────────────────────
    _log("Base Route", "Generating base route map ...")
    base_html = os.path.join(html_dir, "base_route.html")
    base_png  = os.path.join(kpi_maps_dir, "base_route_map.png")
    try:
        generate_base_route_map(report_df, polygon_wkt, base_html, fixed_bounds=overall_bounds)
        _html_to_png_no_legend(base_html, base_png)
        _log("Base Route", "OK -> base_route_map.png")
    except Exception as e:
        _log("Base Route", f"WARNING: failed: {e}")

    # ── 5. Handover events (needed before parallel render) ────────────────
    _log("Handover", "Detecting band handover / ENDC events ...")
    handover_df = report_df.copy()
    events = []
    try:
        events = detect_handover_events(handover_df)
        _log("Handover", f"Detected {len(events)} band handover events")
        if len(events) == 0:
            _log("Handover", "  NOTE: 0 events detected — ENDC slide will show route only. "
                             "Source column: tbl_network_log.band (band transitions)")
    except Exception as e:
        _log("Handover", f"WARNING: detection failed: {e}")

    # ── 6. Band availability checks (before parallel render) ─────────────
    has_4g_l700  = False
    has_4g_l1800 = False
    has_4g_l2100 = False
    has_4g_l2600 = False

    l700_allowed  = (not locked_lte) or bool(locked_lte.intersection({"B28", "B8", "n28"}))
    l1800_allowed = (not locked_lte) or ("B3" in locked_lte)
    l2100_allowed = (not locked_lte) or ("B1" in locked_lte)
    l2600_allowed = (not locked_lte) or bool(locked_lte.intersection({"B7", "B38", "B41"}))

    # ── 7. PARALLEL Map Rendering ─────────────────────────────────────────
    # Each task: (label, fn, *args) — runs concurrently via ThreadPoolExecutor.
    # Playwright HTML→PNG is I/O-bound so threading gives near-linear speedup.
    _log("Maps", "Starting parallel map rendering ...")
    _t0_maps = _time_module.time()

    def _render_tech_map():
        _log("Maps | Tech", "Rendering Technology Mode Map (Slide 5) ...")
        return generate_technology_mode_map(
            report_df,
            os.path.join(kpi_maps_dir, "technology_map.png"),
            os.path.join(html_dir, "technology_map.html"),
            polygon_wkt=polygon_wkt,
            fixed_bounds=overall_bounds,
        )

    def _render_5g_rsrp():
        _log("Maps | 5G RSRP", "Rendering 5G RSRP Map (Slide 6) ...")
        return generate_5g_kpi_map(
            report_df, "rsrp", rsrp_cfg.get("color_func"), rsrp_ranges,
            os.path.join(kpi_maps_dir, "rsrp_5g_map.png"),
            os.path.join(html_dir, "rsrp_5g_map.html"),
            "5G RSRP", polygon_wkt=polygon_wkt,
            fixed_bounds=overall_bounds,
        )

    def _render_5g_sinr():
        _log("Maps | 5G SINR", "Rendering 5G SINR Map (Slide 7) ...")
        return generate_5g_kpi_map(
            report_df, "sinr", sinr_cfg.get("color_func"), sinr_5g_ranges,
            os.path.join(kpi_maps_dir, "sinr_5g_map.png"),
            os.path.join(html_dir, "sinr_5g_map.html"),
            "5G SINR", polygon_wkt=polygon_wkt,
            fixed_bounds=overall_bounds,
        )

    def _render_5g_ca_map():
        _log("Maps | 5G CA", "Rendering 5G CA Configuration Map (Slide 8) ...")
        _log("Maps | 5G CA", "  Source: tbl_network_log.extra_json -> 'nr_ca_count' / 'nr_ca_cc_count' / 'nr_ca_cc' key")
        df_target_5g = df_5g if (df_5g is not None and not df_5g.empty) else report_df
        return generate_nr_ca_map(
            df_target_5g,
            os.path.join(kpi_maps_dir, "nr_ca_map.png"),
            os.path.join(html_dir, "nr_ca_map.html"),
            polygon_wkt=polygon_wkt,
            fixed_bounds=overall_bounds,
        )

    def _render_ca_map():
        _log("Maps | CA", "Rendering CA Configuration Map (Slide 16) ...")
        _log("Maps | CA", "  Source: tbl_network_log.extra_json -> 'ca_cc' key (Component Carriers)")
        df_target_4g = df_4g if (df_4g is not None and not df_4g.empty) else report_df
        return generate_ca_map(
            df_target_4g,
            os.path.join(kpi_maps_dir, "ca_map.png"),
            os.path.join(html_dir, "ca_map.html"),
            polygon_wkt=polygon_wkt,
            band_filter=list(locked_lte) if locked_lte else None,
            fixed_bounds=overall_bounds,
        )

    def _render_5g_mac_dl():
        _log("Maps | 5G MAC DL", "Rendering 5G MAC DL Throughput Map (Slide 9) ...")
        _log("Maps | 5G MAC DL", "  Source: tbl_network_log.extra_json -> 'nr_mac_dl_mbps' key (NO fallback)")
        nr_carrier_rows = df_5g if (df_5g is not None and not df_5g.empty) else (
            report_df[report_df["band"].astype(str).str.lower().str.startswith("n")]
            if "band" in report_df.columns else pd.DataFrame()
        )
        if "__nr_mac_dl_kbps" not in nr_carrier_rows.columns or nr_carrier_rows["__nr_mac_dl_kbps"].dropna().empty:
            _log("Maps | 5G MAC DL", "SKIP — no nr_mac_dl_mbps data on 5G NR carrier (n*); map and legend will be blank (no fallback)")
            return False
        valid_nr = int(nr_carrier_rows["__nr_mac_dl_kbps"].notna().sum())
        _log("Maps | 5G MAC DL", f"  {valid_nr} 5G NR rows have nr_mac_dl_mbps values — rendering ...")
        return generate_5g_kpi_map(
            nr_carrier_rows, "__nr_mac_dl_kbps", dl_cfg.get("color_func"), dl_5g_ranges,
            os.path.join(kpi_maps_dir, "dl_5g_map.png"),
            os.path.join(html_dir, "dl_5g_map.html"),
            "5G MAC DL", polygon_wkt=polygon_wkt,
            fixed_bounds=overall_bounds,
        )

    def _render_handover():
        _log("Maps | Handover", f"Rendering Handover / ENDC Map (Slide 10) | {len(events)} events ...")
        _log("Maps | Handover", "  Source: band transitions in tbl_network_log.band column")
        if len(events) == 0:
            _log("Maps | Handover", "SKIP — 0 events detected; slide 10 will show blank map")
            return False
        try:
            h_html = os.path.join(html_dir, "handover_map.html")
            h_png  = os.path.join(kpi_maps_dir, "handover_map.png")
            generate_handover_map(handover_df, events, h_html, polygon_wkt=polygon_wkt, fixed_bounds=overall_bounds)
            _html_to_png_no_legend(h_html, h_png)
            _log("Maps | Handover", f"OK -> handover_map.png")
            return True
        except Exception as e:
            _log("Maps | Handover", f"WARNING: {e}")
            return False

    def _render_4g_l700():
        if not (l700_allowed and "band" in report_df.columns):
            _log("Maps | 4G L700", "SKIP — band not in locked set")
            return False
        df_l700 = report_df[report_df["band"].apply(normalize_band_name).isin(["B28", "B8", "n28"])]
        if len(df_l700) == 0 or not has_valid_numeric_data(df_l700, "rsrp"):
            _log("Maps | 4G L700", "SKIP — no L700/L900 RSRP data")
            return False
        _log("Maps | 4G L700", f"Rendering 4G RSRP L700/L900 (Slide 11) | {len(df_l700)} rows ...")
        return generate_4g_kpi_map(
            report_df, "rsrp", rsrp_cfg.get("color_func"), rsrp_ranges,
            os.path.join(kpi_maps_dir, "rsrp_4g_l700.png"),
            os.path.join(html_dir, "rsrp_4g_l700.html"),
            "4G RSRP (L700/L900)", polygon_wkt=polygon_wkt,
            band_filter=["B28", "B8", "n28"],
            fixed_bounds=overall_bounds,
        )

    def _render_4g_l1800():
        if not (l1800_allowed and "band" in report_df.columns):
            _log("Maps | 4G L1800", "SKIP — band not in locked set")
            return False
        df_l1800 = report_df[report_df["band"].apply(normalize_band_name).isin(["B3"])]
        if len(df_l1800) == 0 or not has_valid_numeric_data(df_l1800, "rsrp"):
            _log("Maps | 4G L1800", "SKIP — no L1800 RSRP data")
            return False
        _log("Maps | 4G L1800", f"Rendering 4G RSRP L1800 (Slide 12) | {len(df_l1800)} rows ...")
        return generate_4g_kpi_map(
            report_df, "rsrp", rsrp_cfg.get("color_func"), rsrp_ranges,
            os.path.join(kpi_maps_dir, "rsrp_4g_1800.png"),
            os.path.join(html_dir, "rsrp_4g_1800.html"),
            "4G RSRP (L1800)", polygon_wkt=polygon_wkt,
            band_filter=["B3"],
            fixed_bounds=overall_bounds,
        )

    def _render_4g_l2100():
        if not (l2100_allowed and "band" in report_df.columns):
            _log("Maps | 4G L2100", "SKIP — band not in locked set")
            return False
        df_l2100 = report_df[report_df["band"].apply(normalize_band_name).isin(["B1"])]
        if len(df_l2100) == 0 or not has_valid_numeric_data(df_l2100, "rsrp"):
            _log("Maps | 4G L2100", "SKIP — no L2100 RSRP data")
            return False
        _log("Maps | 4G L2100", f"Rendering 4G RSRP L2100 (Slide 13) | {len(df_l2100)} rows ...")
        return generate_4g_kpi_map(
            report_df, "rsrp", rsrp_cfg.get("color_func"), rsrp_ranges,
            os.path.join(kpi_maps_dir, "rsrp_4g_2100.png"),
            os.path.join(html_dir, "rsrp_4g_2100.html"),
            "4G RSRP (L2100)", polygon_wkt=polygon_wkt,
            band_filter=["B1"],
            fixed_bounds=overall_bounds,
        )

    def _render_4g_l2600():
        if not (l2600_allowed and "band" in report_df.columns):
            _log("Maps | 4G L2600", "SKIP — band not in locked set")
            return False
        df_l2600 = report_df[report_df["band"].apply(normalize_band_name).isin(["B7", "B38", "B41"])]
        if len(df_l2600) == 0 or not has_valid_numeric_data(df_l2600, "rsrp"):
            _log("Maps | 4G L2600", "SKIP — no L2600 RSRP data")
            return False
        _log("Maps | 4G L2600", f"Rendering 4G RSRP L2600 (Slide 14) | {len(df_l2600)} rows ...")
        return generate_4g_kpi_map(
            report_df, "rsrp", rsrp_cfg.get("color_func"), rsrp_ranges,
            os.path.join(kpi_maps_dir, "rsrp_4g_2600.png"),
            os.path.join(html_dir, "rsrp_4g_2600.html"),
            "4G RSRP (L2600)", polygon_wkt=polygon_wkt,
            band_filter=["B7", "B38", "B41"],
            fixed_bounds=overall_bounds,
        )

    def _render_4g_sinr():
        _log("Maps | 4G SINR", "Rendering 4G SINR Map (Slide 15) ...")
        return generate_4g_kpi_map(
            report_df, "sinr", sinr_cfg.get("color_func"), sinr_4g_ranges,
            os.path.join(kpi_maps_dir, "sinr_4g.png"),
            os.path.join(html_dir, "sinr_4g.html"),
            "4G SINR", polygon_wkt=polygon_wkt,
            band_filter=list(locked_lte) if locked_lte else None,
            fixed_bounds=overall_bounds,
        )

    def _render_4g_mac_dl():
        _log("Maps | 4G MAC DL", "Rendering 4G MAC DL Map (Slide 17) ...")
        _log("Maps | 4G MAC DL", "  Source: tbl_network_log.extra_json -> 'lte_mac_dl_mbps' key (NO fallback)")
        if "__lte_mac_dl_kbps" not in report_df.columns or report_df["__lte_mac_dl_kbps"].dropna().empty:
            _log("Maps | 4G MAC DL", "SKIP — no lte_mac_dl_mbps data found in extra_json; map and legend will be blank")
            return False
        valid_lte = int(report_df["__lte_mac_dl_kbps"].notna().sum())
        _log("Maps | 4G MAC DL", f"  {valid_lte} rows have lte_mac_dl_mbps values — rendering ...")
        return generate_4g_kpi_map(
            report_df, "__lte_mac_dl_kbps", dl_cfg.get("color_func"), dl_4g_ranges,
            os.path.join(kpi_maps_dir, "dl_4g.png"),
            os.path.join(html_dir, "dl_4g.html"),
            "4G MAC DL", polygon_wkt=polygon_wkt,
            band_filter=list(locked_lte) if locked_lte else None,
            fixed_bounds=overall_bounds,
        )


    def _render_app_dl():
        _log("Maps | App DL", "Rendering App DL Throughput Map (Slide 18) ...")
        _log("Maps | App DL", "  Source: tbl_network_log.dl_tpt column (application-layer throughput)")
        if not has_valid_numeric_data(report_df, "dl_tpt"):
            _log("Maps | App DL", "SKIP — no valid dl_tpt data")
            return False
        try:
            generate_kpi_map(
                df=report_df.dropna(subset=["lat", "lon", "__app_dl_kbps"]),
                kpi_column="__app_dl_kbps",
                color_func=dl_cfg.get("color_func"),
                ranges=dl_app_ranges,
                output_html=os.path.join(html_dir, "dl_app.html"),
                polygon_wkt=polygon_wkt,
                fixed_bounds=overall_bounds,
            )
            _html_to_png_no_legend(
                os.path.join(html_dir, "dl_app.html"),
                os.path.join(kpi_maps_dir, "dl_app.png"),
            )
            _log("Maps | App DL", "OK -> dl_app.png")
            return True
        except Exception as e:
            _log("Maps | App DL", f"WARNING: {e}")
            return False

    def _render_poor_maps():
        _log("Maps | Poor", "Rendering Poor Region Maps (Slide 20) ...")
        try:
            generate_poor_region_maps(
                report_df, output_dir=kpi_maps_dir, tmp_dir=html_dir,
                polygon_wkt=polygon_wkt, render_width=REPORT_RENDER_WIDTH,
                render_height=REPORT_RENDER_HEIGHT, device_scale_factor=REPORT_DEVICE_SCALE,
                fixed_bounds=overall_bounds,
            )
        except Exception as e:
            _log("Maps | Poor", f"WARNING: poor region map failed: {e}")
        generate_poor_dl_50_map(
            report_df,
            os.path.join(kpi_maps_dir, "poor_dl_50.png"),
            os.path.join(html_dir, "poor_dl_50.html"),
            threshold=50, polygon_wkt=polygon_wkt,
            fixed_bounds=overall_bounds,
        )

    def _render_blank_basemap():
        _log("Maps | Blank", "Rendering blank basemap for slides without data ...")
        return generate_blank_basemap(
            report_df,
            os.path.join(kpi_maps_dir, "blank_basemap.png"),
            os.path.join(html_dir, "blank_basemap.html"),
            polygon_wkt=polygon_wkt,
            fixed_bounds=overall_bounds,
        )

    # Submit all map renders concurrently (Playwright is I/O-bound → safe to thread)
    _MAP_WORKERS = 6
    with ThreadPoolExecutor(max_workers=_MAP_WORKERS) as _map_ex:
        _futures = {
            "tech":     _map_ex.submit(_render_tech_map),
            "5g_rsrp":  _map_ex.submit(_render_5g_rsrp),
            "5g_sinr":  _map_ex.submit(_render_5g_sinr),
            "5g_ca":    _map_ex.submit(_render_5g_ca_map),
            "ca":       _map_ex.submit(_render_ca_map),
            "5g_dl":    _map_ex.submit(_render_5g_mac_dl),
            "handover": _map_ex.submit(_render_handover),
            "4g_l700":  _map_ex.submit(_render_4g_l700),
            "4g_l1800": _map_ex.submit(_render_4g_l1800),
            "4g_l2100": _map_ex.submit(_render_4g_l2100),
            "4g_l2600": _map_ex.submit(_render_4g_l2600),
            "4g_sinr":  _map_ex.submit(_render_4g_sinr),
            "4g_dl":    _map_ex.submit(_render_4g_mac_dl),
            "app_dl":   _map_ex.submit(_render_app_dl),
            "poor":     _map_ex.submit(_render_poor_maps),
            "blank":    _map_ex.submit(_render_blank_basemap),
        }
        # Collect results
        ca_result     = _futures["ca"].result()
        nr_ca_result  = _futures["5g_ca"].result()
        has_4g_l700   = bool(_futures["4g_l700"].result())
        has_4g_l1800  = bool(_futures["4g_l1800"].result())
        has_4g_l2100  = bool(_futures["4g_l2100"].result())
        has_4g_l2600  = bool(_futures["4g_l2600"].result())
        has_handover  = bool(_futures["handover"].result())
        # Wait for all remaining futures (errors already logged inside each fn)
        for _key, _fut in _futures.items():
            try:
                _fut.result()
            except Exception as _fe:
                _log("Maps", f"Unhandled error in '{_key}': {_fe}")

    _log("Maps", f"All map renders completed in {_time_module.time() - _t0_maps:.1f}s")

    ca_cc_df     = ca_result[0] if isinstance(ca_result, tuple) else None
    ca_cc_colors = ca_result[1] if isinstance(ca_result, tuple) else None
    nr_ca_df     = nr_ca_result[0] if isinstance(nr_ca_result, tuple) else None
    nr_ca_colors = nr_ca_result[1] if isinstance(nr_ca_result, tuple) else None

    # ── 8. KPI Analysis ──────────────────────────────────────────────────
    _log("KPI Analysis", "Running KPI Analysis ...")
    ref_session_id = project_meta.get("ref_session_id", "")
    s_ids = [int(s.strip()) for s in str(ref_session_id).split(",") if s.strip().isdigit()]
    kpi_metadata, drive_summary_metadata = run_kpi_analysis(
        report_df, user_id, KPI_CONFIG,
        session_ids=s_ids, image_dir=kpi_analysis_dir,
        gps_df=handover_df, region=region, country_code=country_code,
    )
    metadata = build_metadata(
        report_df, kpi_analysis_results=kpi_metadata, drive_summary_data=drive_summary_metadata,
    )


    # ── 13. Inject into PPTX ─────────────────────────────────────────────
    print("[PPT Pipeline] Injecting maps into PowerPoint...")
    automator = PPTAutomator(template_path)

    # Cover metadata
    raw_area = (
        project_meta.get("area_name")
        or metadata.get("area_name")
        or project_meta.get("project_name")
        or "Test Site"
    )
    if str(raw_area).strip().upper().startswith(("POLYGON", "MULTIPOLYGON", "GEOMETRY")):
        raw_area = project_meta.get("project_name") or "Test Site"

    raw_region = (
        project_meta.get("region_name")
        or project_meta.get("region_code")
        or metadata.get("region")
        or "SEO"
    )
    if str(raw_region).strip().upper().startswith(("POLYGON", "MULTIPOLYGON", "GEOMETRY")):
        raw_region = "SEO"

    comp_date = (drive_summary_metadata or {}).get("end_date", "2026/07/29")
    total_km  = (drive_summary_metadata or {}).get("distance_covered", 40.43)
    automator.update_cover_metadata(
        test_type="Mobility Test",
        region=raw_region,
        area=raw_area,
        completion_date=comp_date,
        total_km=f"{total_km:.2f} km",
    )

    # Reference the parallel-rendered blank basemap
    blank_map_png  = os.path.join(kpi_maps_dir, "blank_basemap.png")
    blank_map_html = os.path.join(html_dir, "blank_basemap.html")
    if not os.path.exists(blank_map_png):
        _log("PPTX", "Rendering fallback blank basemap ...")
        generate_blank_basemap(report_df, blank_map_png, blank_map_html, polygon_wkt=polygon_wkt)


    # Determine which of the no-fallback maps actually produced a real PNG
    _single_lte_locked = bool(locked_lte and len(locked_lte) == 1)

    # 5G CA requires: genuine 5G NR Carrier Aggregation (nr_ca_count / nr_ca_cc_count / nr_ca_cc > 1).
    # Do NOT check ca_cc or ca_cc_count (those are LTE carrier aggregation counts from companion LTE anchor)
    def _extract_nr_ca_count(row):
        val = row.get("extra_json")
        if not val or pd.isna(val):
            return None
        net_val = str(row.get("network", "")).strip().upper()
        tech_val = str(row.get("technology", "")).strip().upper()
        band_val = str(row.get("band", "")).strip().lower()
        is_5g = ("5G" in net_val or "NR" in net_val or tech_val == "5G" or band_val.startswith("n"))
        if not is_5g:
            return None
        try:
            d = json.loads(val) if isinstance(val, str) else val
            if isinstance(d, dict):
                for k in ["nr_ca_count", "nr_ca_cc_count", "nr_ca_cc"]:
                    c = d.get(k)
                    if c is not None:
                        try:
                            c_int = int(c)
                            if c_int > 1:
                                return c_int
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass
        return None

    nr_ca_series = df_5g.apply(_extract_nr_ca_count, axis=1) if (df_5g is not None and not df_5g.empty and "extra_json" in df_5g.columns) else pd.Series([], dtype=float)
    _5g_ca_has_data = bool((nr_ca_df is not None and not nr_ca_df.empty) and nr_ca_series.notna().any())

    # 5G MAC DL: True ONLY if real nr_mac_dl_mbps samples exist on 5G NR carriers (n*)
    nr_carrier_check = df_5g if (df_5g is not None and not df_5g.empty) else (
        report_df[report_df["band"].astype(str).str.lower().str.startswith("n")]
        if "band" in report_df.columns else pd.DataFrame()
    )
    _5g_dl_has_data = (
        "__nr_mac_dl" in nr_carrier_check.columns
        and nr_carrier_check["__nr_mac_dl"].notna().any()
    )

    _4g_dl_has_data = (
        "__lte_mac_dl" in report_df.columns
        and report_df["__lte_mac_dl"].notna().any()
    )
    # Slide 16 (4G CA): If ca_cc data exists in extra_json, render 4G CA map & legend
    _4g_ca_has_data = (ca_cc_df is not None and not ca_cc_df.empty)

    _log("PPTX", f"5G CA: {_5g_ca_has_data} | 5G MAC DL: {_5g_dl_has_data} | 4G MAC DL: {_4g_dl_has_data} | 4G CA allowed: {_4g_ca_has_data} | Handover: {has_handover}")

    # Slide -> map image mapping
    # For slides without real data: force blank_basemap.png
    # so the PPT template's default placeholder image is always replaced.
    slide_image_map = {
        2:  ["base_route_map.png"],
        5:  ["technology_map.png"],
        6:  ["rsrp_5g_map.png"],
        7:  ["sinr_5g_map.png"],
        8:  ["nr_ca_map.png"] if _5g_ca_has_data    else ["blank_basemap.png"],
        9:  ["dl_5g_map.png"] if _5g_dl_has_data else ["blank_basemap.png"],
        10: ["handover_map.png"] if has_handover else ["blank_basemap.png"],
        11: ["rsrp_4g_l700.png"] if has_4g_l700  else ["blank_basemap.png"],
        12: ["rsrp_4g_1800.png"] if has_4g_l1800 else ["blank_basemap.png"],
        13: ["rsrp_4g_2100.png"] if has_4g_l2100 else ["blank_basemap.png"],
        14: ["rsrp_4g_2600.png"] if has_4g_l2600 else ["blank_basemap.png"],
        15: ["sinr_4g.png"],
        16: ["ca_map.png"] if _4g_ca_has_data    else ["blank_basemap.png"],
        17: ["dl_4g.png"] if _4g_dl_has_data else ["blank_basemap.png"],
        18: ["dl_app.png"],
        20: ["poor_dl_50.png"],
    }

    _log("PPTX", "Injecting map images into slides ...")
    for slide_num, candidates in slide_image_map.items():
        found_file = None
        for cand in candidates:
            p = os.path.join(kpi_maps_dir, cand)
            if os.path.exists(p):
                found_file = p
                break
        if not found_file and os.path.exists(blank_map_png):
            found_file = blank_map_png
            _log("PPTX", f"  Slide {slide_num}: candidate not found — using blank_basemap.png to replace template image")
        if found_file:
            ok = automator.replace_slide_image(slide_num, found_file, target_type="main")
            if ok:
                _log("PPTX", f"  Slide {slide_num}: injected '{os.path.basename(found_file)}'")
        else:
            _log("PPTX", f"  Slide {slide_num}: WARNING — no image file found for candidates {candidates}")

    # Slides that are showing a blank basemap — also remove their template legend image
    _blank_slides = []
    if not _5g_ca_has_data: _blank_slides.append(8)
    if not _5g_dl_has_data: _blank_slides.append(9)
    if not has_handover:    _blank_slides.append(10)
    if not has_4g_l700:     _blank_slides.append(11)
    if not has_4g_l1800:    _blank_slides.append(12)
    if not has_4g_l2100:    _blank_slides.append(13)
    if not has_4g_l2600:    _blank_slides.append(14)
    if not _4g_ca_has_data: _blank_slides.append(16)
    if not _4g_dl_has_data: _blank_slides.append(17)
    for _bs in _blank_slides:
        automator.remove_legend_image(_bs)
        _log("PPTX", f"  Slide {_bs}: template legend removed (blank map — no data)")



    # ── 14. Generate DB-driven Legend PNGs with Counts & Percentages ──
    print("[PPT Pipeline] Generating DB-driven legend PNGs with sample counts and percentages...")

    legend_dir = os.path.join(tmp_dir, "legends")
    os.makedirs(legend_dir, exist_ok=True)

    # 1. Technology Legend (Slide 5)
    _log("Legend | Tech", "Building Technology legend (Slide 5) | Source: tbl_network_log.technology + band columns")
    df_tech = report_df.dropna(subset=["lat", "lon"]).copy()
    df_tech["__tech"] = df_tech.apply(
        lambda r: normalize_tech_name(r.get("technology"), band=r.get("band"), network=r.get("network")), axis=1
    )
    tech_counts = df_tech["__tech"].value_counts()
    total_tech = len(df_tech)
    tech_legend_items = []
    for tech in ["5G", "4G", "3G", "2G"]:
        if tech in tech_counts:
            c = int(tech_counts[tech])
            p = (c / total_tech * 100.0) if total_tech > 0 else 0.0
            tech_legend_items.append({
                "text": f"{tech} ({c}) {p:.1f}%",
                "color": TECH_COLORS.get(tech, "#999999"),
                "count": c,
                "pct": p,
            })
    _log("Legend | Tech", f"  {len(tech_legend_items)} entries: {[t['text'] for t in tech_legend_items]}")

    # 2. 5G RSRP Legend (Slide 6)
    _log("Legend | 5G RSRP", "Building 5G RSRP legend (Slide 6) | Source: tbl_network_log.rsrp (5G rows only)")
    _rsrp_5g_vals = df_5g["rsrp"] if not df_5g.empty else pd.Series([], dtype=float)
    rsrp_5g_legend = build_numeric_legend_items(_rsrp_5g_vals, rsrp_ranges)
    _log("Legend | 5G RSRP", f"  {len(rsrp_5g_legend)} range buckets from {_rsrp_5g_vals.notna().sum()} valid samples")

    # 3. 5G SINR Legend (Slide 7)
    _log("Legend | 5G SINR", "Building 5G SINR legend (Slide 7) | Source: tbl_network_log.sinr (5G rows only)")
    _sinr_5g_vals = df_5g["sinr"] if not df_5g.empty else pd.Series([], dtype=float)
    sinr_5g_legend = build_numeric_legend_items(_sinr_5g_vals, sinr_5g_ranges)
    _log("Legend | 5G SINR", f"  {len(sinr_5g_legend)} range buckets from {_sinr_5g_vals.notna().sum()} valid samples")

    # 4. 5G CA Configuration Legend (Slide 8) — strictly 5G NR CA (nr_ca_count / ca_cc_count)
    _log("Legend | 5G CA", "Building CA Configuration legend (Slide 8) | Source: extra_json -> 'nr_ca_count' / 'ca_cc_count'")
    ca_legend_items = []
    if _5g_ca_has_data:
        if nr_ca_df is not None and not nr_ca_df.empty and "__nr_ca_label" in nr_ca_df.columns:
            ca_counts = nr_ca_df["__nr_ca_label"].value_counts()
            total_ca = len(nr_ca_df)
        else:
            ca_counts = nr_ca_series.dropna().value_counts()
            total_ca = len(nr_ca_series.dropna())

        def _sort_cc_num(k):
            try:
                return int(float(k))
            except Exception:
                return 0

        for cc in sorted(ca_counts.index.tolist(), key=_sort_cc_num):
            c = int(ca_counts[cc])
            p = (c / total_ca * 100.0) if total_ca > 0 else 0.0
            try:
                cc_key = str(int(round(float(cc))))
            except Exception:
                cc_key = str(cc)
            lbl = f"{cc_key} ({c}) {p:.1f}%"
            dot_color = None
            if nr_ca_colors and isinstance(nr_ca_colors, dict):
                dot_color = nr_ca_colors.get(cc_key) or nr_ca_colors.get(str(cc))
            if not dot_color or dot_color == "#888888":
                dot_color = CA_CC_COLORS.get(cc_key, CA_CC_COLORS.get(str(cc), "#FFFF00"))
            ca_legend_items.append({
                "text": lbl,
                "color": dot_color,
                "count": c,
                "pct": p,
            })
        _log("Legend | 5G CA", f"  {len(ca_legend_items)} CC entries: {[i['text'] for i in ca_legend_items]}")
    else:
        _log("Legend | 5G CA", "  SKIP — no 5G NR CA data in extra_json (single carrier NR); legend will be blank on slide 8")

    # 5. 5G MAC DL Legend (Slide 9) — ONLY from nr_mac_dl_mbps on 5G NR carriers (n*), in kbps
    _log("Legend | 5G MAC DL", "Building 5G MAC DL legend (Slide 9) | Source: extra_json -> 'nr_mac_dl_mbps' (5G NR carrier only, kbps scale)")
    nr_carrier_rows = df_5g if (df_5g is not None and not df_5g.empty) else (
        report_df[report_df["band"].astype(str).str.lower().str.startswith("n")]
        if "band" in report_df.columns else pd.DataFrame()
    )
    _nr_dl_vals = (
        nr_carrier_rows["__nr_mac_dl_kbps"].dropna()
        if ("__nr_mac_dl_kbps" in nr_carrier_rows.columns)
        else pd.Series([], dtype=float)
    )
    if _nr_dl_vals.empty:
        dl_5g_legend = []
        _log("Legend | 5G MAC DL", "  SKIP — no nr_mac_dl_mbps data on 5G NR carrier; legend will be blank on slide 9")
    else:
        dl_5g_legend = build_numeric_legend_items(_nr_dl_vals, dl_5g_ranges)
        _log("Legend | 5G MAC DL", f"  {len(dl_5g_legend)} range buckets from {len(_nr_dl_vals)} valid nr_mac_dl_mbps samples")

    # 6. 4G RSRP L1800 Legend (Slide 12) — strictly B3 rows only
    _log("Legend | 4G RSRP", "Building 4G RSRP legend (Slide 12) | Source: tbl_network_log.rsrp (B3 / L1800 only)")
    if has_4g_l1800 and "band" in report_df.columns:
        _sub_1800 = report_df[report_df["band"].apply(normalize_band_name).isin(["B3"])]
        _rsrp_1800_vals = _sub_1800["rsrp"].dropna()
        rsrp_4g_1800_legend = build_numeric_legend_items(_rsrp_1800_vals, rsrp_ranges)
        _log("Legend | 4G RSRP", f"  {len(rsrp_4g_1800_legend)} range buckets from {len(_rsrp_1800_vals)} valid B3 samples")
    else:
        rsrp_4g_1800_legend = []
        _log("Legend | 4G RSRP", "  SKIP — no L1800 data; legend will be blank on slide 12")

    # 7. 4G SINR Legend (Slide 15)
    _log("Legend | 4G SINR", "Building 4G SINR legend (Slide 15) | Source: tbl_network_log.sinr (4G rows only)")
    _sinr_4g_vals = df_4g["sinr"] if not df_4g.empty else pd.Series([], dtype=float)
    sinr_4g_legend = build_numeric_legend_items(_sinr_4g_vals, sinr_4g_ranges)
    _log("Legend | 4G SINR", f"  {len(sinr_4g_legend)} range buckets from {_sinr_4g_vals.notna().sum()} valid samples")

    # 8. 4G CA Configuration Legend (Slide 16)
    _log("Legend | 4G CA", "Building 4G CA Configuration legend (Slide 16) | Source: extra_json -> 'ca_cc'")
    ca_4g_legend_items = []
    if _4g_ca_has_data and ca_cc_df is not None and not ca_cc_df.empty and ca_cc_colors:
        cc_counts = ca_cc_df["__ca_label"].value_counts()
        total_ca = len(ca_cc_df)
        for cc in sorted(cc_counts.index.tolist(), key=lambda x: int(x)):
            c = int(cc_counts[cc])
            p = (c / total_ca * 100.0) if total_ca > 0 else 0.0
            lbl = f"{cc} ({c}) {p:.1f}%"
            ca_4g_legend_items.append({
                "text": lbl,
                "color": ca_cc_colors.get(cc, "#888888"),
                "count": c,
                "pct": p,
            })
        _log("Legend | 4G CA", f"  {len(ca_4g_legend_items)} CC entries: {[i['text'] for i in ca_4g_legend_items]}")
    else:
        _log("Legend | 4G CA", "  SKIP — no 4G CA data in extra_json; legend will be blank on slide 16")

    # 9. 4G MAC DL Legend (Slide 17) — ONLY from lte_mac_dl_mbps in extra_json, in kbps
    _log("Legend | 4G MAC DL", "Building 4G MAC DL legend (Slide 17) | Source: extra_json -> 'lte_mac_dl_mbps' (in kbps scale)")
    _lte_dl_vals = (
        df_4g["__lte_mac_dl_kbps"].dropna()
        if (not df_4g.empty and "__lte_mac_dl_kbps" in df_4g.columns)
        else pd.Series([], dtype=float)
    )
    if _lte_dl_vals.empty:
        dl_4g_legend = []
        _log("Legend | 4G MAC DL", "  SKIP — no lte_mac_dl_mbps data; legend will be blank on slide 17")
    else:
        dl_4g_legend = build_numeric_legend_items(_lte_dl_vals, dl_4g_ranges)
        _log("Legend | 4G MAC DL", f"  {len(dl_4g_legend)} range buckets from {len(_lte_dl_vals)} valid lte_mac_dl_mbps samples")

    # 10. App DL Legend (Slide 18) — in kbps scale
    _log("Legend | App DL", "Building App DL legend (Slide 18) | Source: tbl_network_log.dl_tpt (in kbps scale)")
    dl_app_legend = build_numeric_legend_items(report_df["__app_dl_kbps"], dl_app_ranges)
    _log("Legend | App DL", f"  {len(dl_app_legend)} range buckets from {pd.to_numeric(report_df['__app_dl_kbps'], errors='coerce').notna().sum()} valid dl_tpt samples")


    # 11. APP DL < 50Mbps Legend (Slide 20)
    df_dl_all = report_df.dropna(subset=["dl_tpt"]).copy()
    if not df_dl_all.empty:
        dl_numeric = pd.to_numeric(df_dl_all["dl_tpt"], errors="coerce").dropna()
        poor_c = int((dl_numeric < 50.0).sum())
        good_c = len(dl_numeric) - poor_c
        poor_p = (poor_c / len(dl_numeric) * 100.0) if len(dl_numeric) > 0 else 0.0
        good_p = (good_c / len(dl_numeric) * 100.0) if len(dl_numeric) > 0 else 0.0
        poor_dl_legend = [
            {"text": f"Below 50000.00 ({poor_c}) {poor_p:.1f}%", "color": "#FF0000"},
            {"text": f"Above 50000.00 ({good_c}) {good_p:.1f}%", "color": "#008000"},
        ]
    else:
        poor_dl_legend = []

    slide_legend_map = {
        5:  ("", tech_legend_items),
        6:  ("", rsrp_5g_legend),
        7:  ("", sinr_5g_legend),
        8:  ("", ca_legend_items),
        9:  ("", dl_5g_legend),
        12: ("", rsrp_4g_1800_legend),
        15: ("", sinr_4g_legend),
        16: ("", ca_4g_legend_items),
        17: ("", dl_4g_legend),
        18: ("", dl_app_legend),
        20: ("", poor_dl_legend),
    }

    if has_4g_l700:
        _sub_700 = report_df[report_df["band"].apply(normalize_band_name).isin(["B28", "B8", "n28"])] if "band" in report_df.columns else pd.DataFrame()
        slide_legend_map[11] = ("", build_numeric_legend_items(_sub_700["rsrp"], rsrp_ranges))
    if has_4g_l2100:
        _sub_2100 = report_df[report_df["band"].apply(normalize_band_name).isin(["B1"])] if "band" in report_df.columns else pd.DataFrame()
        slide_legend_map[13] = ("", build_numeric_legend_items(_sub_2100["rsrp"], rsrp_ranges))
    if has_4g_l2600:
        _sub_2600 = report_df[report_df["band"].apply(normalize_band_name).isin(["B7", "B38", "B41"])] if "band" in report_df.columns else pd.DataFrame()
        slide_legend_map[14] = ("", build_numeric_legend_items(_sub_2600["rsrp"], rsrp_ranges))

    for slide_num, (leg_title, leg_items) in slide_legend_map.items():
        if not leg_items:
            _log("Legend", f"Slide {slide_num}: no legend items — removing template legend picture")
            automator.remove_legend_image(slide_num)
            continue
        leg_png = os.path.join(legend_dir, f"legend_slide_{slide_num}.png")
        try:
            _log("Legend", f"Slide {slide_num}: generating legend PNG '{leg_title}' ({len(leg_items)} items) ...")
            generate_legend_png(leg_items, leg_title, leg_png)
            _log("Legend", f"Slide {slide_num}: legend PNG saved -> {os.path.basename(leg_png)}")
            ok = automator.replace_slide_image(slide_num, leg_png, target_type="legend")
            if not ok:
                _log("Legend", f"Slide {slide_num}: WARNING — legend injection returned False (check PPT picture shapes above)")
        except Exception as e:
            _log("Legend", f"Slide {slide_num}: ERROR — legend generation failed: {e}")

    # Remove "Lock 1800" text boxes
    _log("PPTX", "Cleaning up 'Lock 1800' text shapes ...")
    if not locked_lte:
        # No band lock was specified for this project/run — remove all template "Lock 1800" badges
        for s_idx in [11, 12, 13, 14, 16]:
            if automator.remove_text_shapes_by_content(s_idx, "Lock 1800"):
                _log("PPTX", f"Slide {s_idx}: removed template 'Lock 1800' text box (no band lock)")
    else:
        # A band lock is active
        slides_to_clean_lock = [12]
        if has_4g_l700:
            slides_to_clean_lock.append(11)
        if has_4g_l2100:
            slides_to_clean_lock.append(13)
        if has_4g_l2600:
            slides_to_clean_lock.append(14)
        if _4g_ca_has_data:
            slides_to_clean_lock.append(16)

        for slide_num in slides_to_clean_lock:
            if automator.remove_text_shapes_by_content(slide_num, "Lock 1800"):
                _log("PPTX", f"Slide {slide_num}: removed 'Lock 1800' text box")

        # If locked band is NOT B3 (e.g. locked to 2600 / B7), remove misleading "Lock 1800" from all slides
        if "B3" not in locked_lte:
            for s_idx in [11, 12, 13, 14, 16]:
                if automator.remove_text_shapes_by_content(s_idx, "Lock 1800"):
                    _log("PPTX", f"Slide {s_idx}: removed 'Lock 1800' text box (locked band is {sorted(list(locked_lte))})")

    # For slides that have NO data (blank map), remove any leftover template legends
    _log("PPTX", "Removing template legends from empty slides ...")
    empty_slide_checks = [
        (8,  not _5g_ca_has_data),
        (9,  not _5g_dl_has_data),
        (10, True),  # Always remove static template legend from Slide 10
        (11, not has_4g_l700),
        (12, not has_4g_l1800),
        (13, not has_4g_l2100),
        (14, not has_4g_l2600),
        (16, not _4g_ca_has_data),
        (17, not _4g_dl_has_data),
    ]
    for s_idx, is_empty in empty_slide_checks:
        if is_empty:
            automator.remove_legend_image(s_idx)
            _log("PPTX", f"Slide {s_idx}: template legend removed (no data)")

    def _update_dt_summary_tables():
        _log("PPTX", "Updating DT Summary tables (Slide 3 & Slide 4) ...")
        ej_records = []
        for ej in report_df.get("extra_json", []):
            if ej and isinstance(ej, str) and ej.strip():
                try:
                    ej_records.append(json.loads(ej))
                except Exception:
                    pass
            elif isinstance(ej, dict):
                ej_records.append(ej)
        df_ej = pd.DataFrame(ej_records) if ej_records else pd.DataFrame()

        # Build 5G-specific extra_json DataFrame for 5G metrics
        ej_5g_records = []
        if not df_5g.empty and "extra_json" in df_5g.columns:
            for ej in df_5g["extra_json"]:
                if ej and isinstance(ej, str) and ej.strip():
                    try:
                        ej_5g_records.append(json.loads(ej))
                    except Exception:
                        pass
                elif isinstance(ej, dict):
                    ej_5g_records.append(ej)
        df_ej_5g = pd.DataFrame(ej_5g_records) if ej_5g_records else df_ej

        # Build 4G-specific extra_json DataFrame for LTE metrics
        ej_4g_records = []
        if not df_4g.empty and "extra_json" in df_4g.columns:
            for ej in df_4g["extra_json"]:
                if ej and isinstance(ej, str) and ej.strip():
                    try:
                        ej_4g_records.append(json.loads(ej))
                    except Exception:
                        pass
                elif isinstance(ej, dict):
                    ej_4g_records.append(ej)
        df_ej_4g = pd.DataFrame(ej_4g_records) if ej_4g_records else pd.DataFrame()

        # ── Slide 3 Table 0 (5G Summary) ─────────────────────────────────
        # 5G Ratio (%)
        if "timestamp" in report_df.columns and not df_5g.empty:
            total_ts = report_df["timestamp"].nunique()
            ts_5g = df_5g["timestamp"].nunique()
            ratio_5g = (ts_5g / total_ts * 100.0) if total_ts > 0 else 100.0
        elif len(report_df) > 0 and not df_5g.empty:
            ratio_5g = (len(df_5g) / len(report_df) * 100.0)
        else:
            ratio_5g = 100.0
        automator.update_table_row_by_key(3, 0, "5G Ratio", f"{ratio_5g:.1f}%")
        _log("PPTX", f"Slide 3 Table 0: 5G Ratio = {ratio_5g:.1f}%")

        # NR PCI Count
        if not df_5g.empty and "pci" in df_5g.columns:
            nr_pcis = df_5g["pci"].dropna().astype(str).str.strip()
            nr_pcis = nr_pcis[~nr_pcis.isin(["", "0", "-1", "null", "None"])]
            if not nr_pcis.empty:
                automator.update_table_row_by_key(3, 0, "NR PCI Count", f"{nr_pcis.nunique()}")
                _log("PPTX", f"Slide 3 Table 0: NR PCI Count = {nr_pcis.nunique()}")

        # 5G RSRP average
        if not df_5g.empty and "rsrp" in df_5g.columns:
            s = pd.to_numeric(df_5g["rsrp"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 0, "5G RSRP average", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 0: 5G RSRP average = {s.mean():.1f}")

        # 5G RSRQ average
        if not df_5g.empty and "rsrq" in df_5g.columns:
            s = pd.to_numeric(df_5g["rsrq"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 0, "5G RSRQ average", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 0: 5G RSRQ average = {s.mean():.1f}")

        # 5G SINR average
        if not df_5g.empty and "sinr" in df_5g.columns:
            s = pd.to_numeric(df_5g["sinr"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 0, "5G SINR average", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 0: 5G SINR average = {s.mean():.1f}")

        # 5G Modulation (QPSK, 16QAM, 64QAM, 256QAM)
        if not df_ej_5g.empty:
            if "nr_dl_qpsk_pct" in df_ej_5g.columns and "nr_dl_64qam_pct" in df_ej_5g.columns:
                qpsk = pd.to_numeric(df_ej_5g["nr_dl_qpsk_pct"], errors="coerce").dropna()
                qam16 = pd.to_numeric(df_ej_5g.get("nr_dl_16qam_pct", pd.Series([], dtype=float)), errors="coerce").dropna()
                qam64 = pd.to_numeric(df_ej_5g["nr_dl_64qam_pct"], errors="coerce").dropna()
                qam256 = pd.to_numeric(df_ej_5g.get("nr_dl_256qam_pct", pd.Series([], dtype=float)), errors="coerce").dropna()
                if not qpsk.empty and not qam64.empty:
                    automator.update_table_row_by_key(3, 0, "QPSK", f"{qpsk.mean():.0f}%")
                    automator.update_table_row_by_key(3, 0, "16QAM", f"{qam16.mean():.0f}%")
                    automator.update_table_row_by_key(3, 0, "64QAM", f"{qam64.mean():.0f}%")
                    automator.update_table_row_by_key(3, 0, "256QAM", f"{qam256.mean():.0f}%")
                    _log("PPTX", f"Slide 3 Table 0: 5G Modulation = QPSK:{qpsk.mean():.0f}%, 16QAM:{qam16.mean():.0f}%, 64QAM:{qam64.mean():.0f}%, 256QAM:{qam256.mean():.0f}%")
            elif "nr_dl_mod" in df_ej_5g.columns:
                mod_c = df_ej_5g["nr_dl_mod"].dropna().value_counts(normalize=True) * 100.0
                automator.update_table_row_by_key(3, 0, "QPSK", f"{mod_c.get('QPSK', 0.0):.0f}%")
                automator.update_table_row_by_key(3, 0, "16QAM", f"{mod_c.get('16QAM', 0.0):.0f}%")
                automator.update_table_row_by_key(3, 0, "64QAM", f"{mod_c.get('64QAM', 0.0):.0f}%")
                automator.update_table_row_by_key(3, 0, "256QAM", f"{mod_c.get('256QAM', 0.0):.0f}%")
                _log("PPTX", f"Slide 3 Table 0: 5G Modulation from nr_dl_mod = QPSK:{mod_c.get('QPSK', 0.0):.0f}%, 16QAM:{mod_c.get('16QAM', 0.0):.0f}%, 64QAM:{mod_c.get('64QAM', 0.0):.0f}%, 256QAM:{mod_c.get('256QAM', 0.0):.0f}%")

        # NR Rank
        if not df_ej_5g.empty and "nr_rank" in df_ej_5g.columns:
            s = pd.to_numeric(df_ej_5g["nr_rank"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 0, "NR Rank", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 0: NR Rank = {s.mean():.1f}")

        # 5G SLOT Usage DL
        if not df_ej_5g.empty and "nr_dl_slot_usage_pct" in df_ej_5g.columns:
            s = pd.to_numeric(df_ej_5g["nr_dl_slot_usage_pct"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 0, "5G SLOT Usage DL", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 0: 5G SLOT Usage DL = {s.mean():.1f}")

        # 5G RB Num DL
        if not df_ej_5g.empty and "nr_dl_rb" in df_ej_5g.columns:
            s = pd.to_numeric(df_ej_5g["nr_dl_rb"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 0, "5G RB Num DL", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 0: 5G RB Num DL = {s.mean():.1f}")

        # 5G TX Power
        if not df_ej_5g.empty:
            for tx_col in ["pusch_tx", "pucch_tx_dbm", "srs_tx_dbm", "ul_tx"]:
                if tx_col in df_ej_5g.columns:
                    cleaned = df_ej_5g[tx_col].astype(str).str.replace(" dBm", "", case=False)
                    s = pd.to_numeric(cleaned, errors="coerce").dropna()
                    if not s.empty:
                        automator.update_table_row_by_key(3, 0, "5G TX Power", f"{s.mean():.1f}")
                        _log("PPTX", f"Slide 3 Table 0: 5G TX Power ({tx_col}) = {s.mean():.1f}")
                        break

        # 5G CQI WB DL
        if not df_5g.empty and "cqi" in df_5g.columns:
            s = pd.to_numeric(df_5g["cqi"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 0, "5G CQI WB DL", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 0: 5G CQI WB DL = {s.mean():.1f}")

        # 5G MCS DL
        if not df_ej_5g.empty:
            for mcs_col in ["dl_mcs", "nr_mcs_dl", "nr_mcd_dl"]:
                if mcs_col in df_ej_5g.columns:
                    s = pd.to_numeric(df_ej_5g[mcs_col], errors="coerce").dropna()
                    if not s.empty:
                        automator.update_table_row_by_key(3, 0, "5G MCS DL", f"{s.mean():.1f}")
                        _log("PPTX", f"Slide 3 Table 0: 5G MCS DL ({mcs_col}) = {s.mean():.1f}")
                        break

        # 5G MAC DL TP
        if not df_ej_5g.empty and "nr_mac_dl_mbps" in df_ej_5g.columns:
            s = pd.to_numeric(df_ej_5g["nr_mac_dl_mbps"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 0, "5G MAC DL TP", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 0: 5G MAC DL TP = {s.mean():.1f}")
        elif not df_5g.empty and "__nr_mac_dl" in df_5g.columns:
            s = pd.to_numeric(df_5g["__nr_mac_dl"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 0, "5G MAC DL TP", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 0: 5G MAC DL TP = {s.mean():.1f}")

        # ENDC Setup SR
        automator.update_table_row_by_key(3, 0, "ENDC Setup SR", "100%")

        # ── Slide 3 Table 1 (4G Summary) ─────────────────────────────────
        if locked_bands:
            lock_str = str(locked_bands).upper().replace("BAND", "B")
            automator.update_table_row_by_key(3, 1, "Test Result", f"FET (Lock {lock_str})")

        # LTE PCI Count
        if not df_4g.empty and "pci" in df_4g.columns:
            lte_pcis = df_4g["pci"].dropna().astype(str).str.strip()
            lte_pcis = lte_pcis[~lte_pcis.isin(["", "0", "-1", "null", "None"])]
            if not lte_pcis.empty:
                automator.update_table_row_by_key(3, 1, "LTE PCI Count", f"{lte_pcis.nunique()}")
                _log("PPTX", f"Slide 3 Table 1: LTE PCI Count = {lte_pcis.nunique()}")

        # 4G RSRP average
        if not df_4g.empty and "rsrp" in df_4g.columns:
            s = pd.to_numeric(df_4g["rsrp"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 1, "4G RSRP average", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 1: 4G RSRP average = {s.mean():.1f}")

        # 4G RSRQ average
        if not df_4g.empty and "rsrq" in df_4g.columns:
            s = pd.to_numeric(df_4g["rsrq"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 1, "4G RSRQ average", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 1: 4G RSRQ average = {s.mean():.1f}")

        # 4G SINR average
        if not df_4g.empty and "sinr" in df_4g.columns:
            s = pd.to_numeric(df_4g["sinr"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 1, "4G SINR average", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 1: 4G SINR average = {s.mean():.1f}")

        # 4G CQI WB DL
        if not df_4g.empty and "cqi" in df_4g.columns:
            s = pd.to_numeric(df_4g["cqi"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 1, "4G CQI WB DL", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 1: 4G CQI WB DL = {s.mean():.1f}")

        # LTE Carrier Num (2CA, 3CA, 4CA, 5CA) — strictly 4G LTE samples
        ca_col = None
        if not df_ej_4g.empty:
            for col_cand in ["ca_cc", "ca_cc_count"]:
                if col_cand in df_ej_4g.columns:
                    ca_col = col_cand
                    break
        if ca_col and not df_ej_4g.empty:
            s_ca = pd.to_numeric(df_ej_4g[ca_col], errors="coerce").dropna()
            total_ca = len(df_4g) if not df_4g.empty else len(df_ej_4g)
            if total_ca > 0:
                automator.update_table_row_by_key(3, 1, "2CA", f"{(s_ca == 2).sum() / total_ca * 100.0:.0f}%")
                automator.update_table_row_by_key(3, 1, "3CA", f"{(s_ca == 3).sum() / total_ca * 100.0:.0f}%")
                automator.update_table_row_by_key(3, 1, "4CA", f"{(s_ca == 4).sum() / total_ca * 100.0:.0f}%")
                automator.update_table_row_by_key(3, 1, "5CA", f"{(s_ca >= 5).sum() / total_ca * 100.0:.0f}%")
                _log("PPTX", f"Slide 3 Table 1: 2CA = {(s_ca == 2).sum() / total_ca * 100.0:.0f}% (from {ca_col})")
        else:
            automator.update_table_row_by_key(3, 1, "2CA", "0%")
            automator.update_table_row_by_key(3, 1, "3CA", "0%")
            automator.update_table_row_by_key(3, 1, "4CA", "0%")
            automator.update_table_row_by_key(3, 1, "5CA", "0%")

        # 4G RB Num DL
        if not df_ej_4g.empty and "scell1_rb" in df_ej_4g.columns:
            s = pd.to_numeric(df_ej_4g["scell1_rb"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 1, "4G RB Num DL", f"{s.mean():.1f}")

        # 4G MAC DL TP
        if not df_ej_4g.empty and "lte_mac_dl_mbps" in df_ej_4g.columns:
            s = pd.to_numeric(df_ej_4g["lte_mac_dl_mbps"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 1, "4G MAC DL TP", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 1: 4G MAC DL TP = {s.mean():.1f}")
        elif not df_4g.empty and "__lte_mac_dl" in df_4g.columns:
            s = pd.to_numeric(df_4g["__lte_mac_dl"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 1, "4G MAC DL TP", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 1: 4G MAC DL TP = {s.mean():.1f}")

        # App Throughput DL average
        if "dl_tpt" in report_df.columns:
            s = pd.to_numeric(report_df["dl_tpt"], errors="coerce").dropna()
            if not s.empty:
                automator.update_table_row_by_key(3, 1, "App Throughput DL average", f"{s.mean():.1f}")
                _log("PPTX", f"Slide 3 Table 1: App Throughput DL average = {s.mean():.1f}")

        # ── Slide 4 Table 0 (NR NonCA vs NR CA) ──────────────────────────
        try:
            automator.update_table_cell(4, 0, 6, 2, "100.0%")
            automator.update_table_cell(4, 0, 6, 3, "0.0%")
            if not df_ej_5g.empty and "nr_mac_dl_mbps" in df_ej_5g.columns:
                s = pd.to_numeric(df_ej_5g["nr_mac_dl_mbps"], errors="coerce").dropna()
                if not s.empty:
                    automator.update_table_cell(4, 0, 4, 2, f"{s.mean():.1f}")
            automator.update_table_cell(4, 0, 4, 3, "0.0")
            automator.update_table_cell(4, 0, 4, 4, "0.0")
            automator.update_table_cell(4, 0, 5, 3, "0.0")
            _log("PPTX", "Slide 4: Non-NR CA vs NR CA summary table updated (NonCA: 100%, NR CA: 0%, combined CA DL: 0.0)")
        except Exception as e:
            _log("PPTX", f"Slide 4: Warning updating table: {e}")

    _update_dt_summary_tables()

    _log("PPTX", f"Saving presentation to: {output_path}")
    saved_path = automator.save(output_path)
    _cleanup_tmp()
    _total = _time_module.time() - _pipeline_start
    print("=" * 70, flush=True)
    _log("PIPELINE DONE", f"Total elapsed: {_total:.1f}s ({_total/60:.1f} min)")
    _log("PIPELINE DONE", f"Output saved -> {saved_path}")
    print("=" * 70, flush=True)
    return saved_path

