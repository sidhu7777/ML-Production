import os
import math
import numpy as np
from numpy import around
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from .kpi_config import KPI_CONFIG
from .threshold_resolver import resolve_kpi_ranges
from .map_generator import normalize_band_name, build_report_band_color_map

IMAGE_DIR = "data/images/kpi_analysis"

# Poor-PCI tables can include 100+ PCIs; cap the visible rows for the PDF.
POOR_PCI_LIMIT = 15


# =====================================================
# SAFETY HELPERS
# =====================================================
def has_valid_series(df, column):
    return column in df.columns and df[column].dropna().shape[0] > 0


def has_non_empty_df(df):
    return df is not None and not df.empty


def _join_unique_non_empty(values):
    cleaned = (
        values.dropna()
        .astype(str)
        .str.strip()
    )
    cleaned = cleaned[cleaned.ne("") & cleaned.str.lower().ne("none")]
    return ",".join(sorted(set(cleaned)))


def _nice_tick_step(min_value, max_value, max_ticks=10):
    span = max_value - min_value
    if span <= 0:
        return 1

    raw_step = span / max(1, max_ticks)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude

    for candidate in (1, 2, 5, 10):
        if normalized <= candidate:
            return candidate * magnitude
    return 10 * magnitude


# =====================================================
# UTILITY: SAVE TABLE IMAGE
# =====================================================
def save_table_image(df, title, filename):
    # Increase figure size and DPI for better visibility
    fig, ax = plt.subplots(figsize=(14, max(2.5, len(df) * 0.5)))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)  # Increased from 9 to 11
    table.scale(1, 1.5)  # Increased from 1.2 to 1.5
    
    # Style header row for better visibility
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Only draw title if provided (allow creating images without labels)
    if title:
        plt.title(title, pad=15, fontsize=14, weight='bold')
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(IMAGE_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


# =====================================================
# 1. KPI SUMMARY
# =====================================================
def generate_kpi_summary(df):
    rows = []
    kpi_metadata = {}

    for kpi, cfg in KPI_CONFIG.items():
        if cfg["type"] != "range":
            continue

        col = cfg["column"]
        if not has_valid_series(df, col):
            continue

        values = df[col].dropna()
        rows.append({
            "KPI": kpi,
            "Average": round(values.mean(), 2),
            "Min": round(values.min(), 2),
            "Max": round(values.max(), 2),
            "Median": round(values.median(), 2)
        })

        kpi_metadata[kpi] = {
            "average" : around(values.mean(), 2),
            "min"     : round(values.min(), 2),
            "max"     : round(values.max(), 2),
            "median"  : round(values.median(), 2),
            "percentile_25": round(values.quantile(0.25), 2),
            "percentile_75": round(values.quantile(0.75), 2)
        }

        # Add KPI-specific poor performance metrics
        if kpi == "RSRP":
            poor_threshold = -105
            poor_count = (values < poor_threshold).sum()
            poor_percentage = round(poor_count / len(values) * 100, 2) if len(values) > 0 else 0
            kpi_metadata[kpi]["poor_threshold"] = poor_threshold
            kpi_metadata[kpi]["poor_count"] = int(poor_count)
            kpi_metadata[kpi]["poor_percentage"] = poor_percentage
        elif kpi == "RSRQ":
            poor_threshold = -14
            poor_count = (values < poor_threshold).sum()
            poor_percentage = round(poor_count / len(values) * 100, 2) if len(values) > 0 else 0
            kpi_metadata[kpi]["poor_threshold"] = poor_threshold
            kpi_metadata[kpi]["poor_count"] = int(poor_count)
            kpi_metadata[kpi]["poor_percentage"] = poor_percentage
        elif kpi == "SINR":
            poor_threshold = 0
            poor_count = (values < poor_threshold).sum()
            poor_percentage = round(poor_count / len(values) * 100, 2) if len(values) > 0 else 0
            kpi_metadata[kpi]["poor_threshold"] = poor_threshold
            kpi_metadata[kpi]["poor_count"] = int(poor_count)
            kpi_metadata[kpi]["poor_percentage"] = poor_percentage
        elif kpi == "DL":
            poor_threshold = 5  # Mbps
            poor_count = (values <= poor_threshold).sum()
            poor_percentage = round(poor_count / len(values) * 100, 2) if len(values) > 0 else 0
            kpi_metadata[kpi]["poor_threshold"] = poor_threshold
            kpi_metadata[kpi]["poor_count"] = int(poor_count)
            kpi_metadata[kpi]["poor_percentage"] = poor_percentage
        elif kpi == "UL":
            poor_threshold = 2  # Mbps
            poor_count = (values <= poor_threshold).sum()
            poor_percentage = round(poor_count / len(values) * 100, 2) if len(values) > 0 else 0
            kpi_metadata[kpi]["poor_threshold"] = poor_threshold
            kpi_metadata[kpi]["poor_count"] = int(poor_count)
            kpi_metadata[kpi]["poor_percentage"] = poor_percentage
        elif kpi == "MOS":
            poor_threshold = 2
            poor_count = (values < poor_threshold).sum()
            poor_percentage = round(poor_count / len(values) * 100, 2) if len(values) > 0 else 0
            kpi_metadata[kpi]["poor_threshold"] = poor_threshold
            kpi_metadata[kpi]["poor_count"] = int(poor_count)
            kpi_metadata[kpi]["poor_percentage"] = poor_percentage


    summary_df = pd.DataFrame(rows)
    if not has_non_empty_df(summary_df):
        return

    save_table_image(summary_df, "KPI Summary", "kpi_summary.png")
    return kpi_metadata

# =====================================================
# 2. BAND SUMMARY
# =====================================================
def generate_band_summary(df):
    """
    Band distribution table + pie.

    Raw band values are normalized and the "Unknown" bucket is dropped, so the
    denominator is the count of *known* bands — this keeps the table, the pie
    and the LLM-facing metadata band % identical.  Returns the band summary
    list (band/sample_count/sample_percentage) for injection into metadata.
    """
    if not has_valid_series(df, "band"):
        return None

    normalized = df["band"].apply(normalize_band_name)
    known = normalized[normalized != "Unknown"]
    if known.empty:
        return None

    total = int(len(known))
    band_df = known.value_counts().reset_index()
    band_df.columns = ["band", "Sample Count"]
    band_df["% Samples"] = round((band_df["Sample Count"] / total) * 100, 2)

    save_table_image(band_df, "Band Distribution", "band_table.png")

    band_color_map = build_report_band_color_map(band_df["band"].tolist())
    pie_colors = [band_color_map[band] for band in band_df["band"]]
    plt.figure(figsize=(6, 6))
    plt.pie(
        band_df["% Samples"],
        labels=band_df["band"],
        autopct="%1.1f%%",
        colors=pie_colors,
    )
    plt.title("Band Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "band_pie.png"), dpi=200)
    plt.close()

    return [
        {
            "band": row["band"],
            "sample_count": int(row["Sample Count"]),
            "sample_percentage": float(row["% Samples"]),
        }
        for _, row in band_df.iterrows()
    ]


# =====================================================
# 3. KPI RANGE TABLES
# =====================================================
def generate_kpi_range_tables(df, user_id):
    for kpi, cfg in KPI_CONFIG.items():
        if cfg["type"] != "range":
            continue

        col = cfg["column"]
        if not has_valid_series(df, col):
            continue

        total = df[col].dropna().shape[0]
        if total == 0:
            continue

        rows = []
        ranges = resolve_kpi_ranges(kpi, user_id, values=df[col])
        cumulative = 0
        for idx, r in enumerate(ranges):
            is_last = (idx == len(ranges) - 1)
            # Use half-open intervals: [min, max) for all except last [min, max]
            if is_last:
                count = df[(df[col] >= r["min"]) & (df[col] <= r["max"])].shape[0]
            else:
                count = df[(df[col] >= r["min"]) & (df[col] < r["max"])].shape[0]
            if count == 0:
                continue

            percent = round((count / total) * 100, 2)
            cumulative += percent
            
            rows.append({
                "Range": r.get("range") or f'{r["min"]} to {r["max"]}',
                "Sample Count": count,
                "% Of Samples": percent,
                "CDF": round(cumulative, 2)
            })

        range_df = pd.DataFrame(rows)
        if not has_non_empty_df(range_df):
            continue

        save_table_image(
            range_df,
            f"{kpi} Distribution",
            f"{kpi.lower()}_range_table.png"
        )

# kpi range summary for metadata.json

def generate_kpi_range_summary(df, user_id):
    """
    Returns KPI range distribution for metadata.json
    (same logic as generate_kpi_range_tables, but data-only)
    """
    range_summary = {}

    for kpi, cfg in KPI_CONFIG.items():
        if cfg["type"] != "range":
            continue

        col = cfg["column"]
        if not has_valid_series(df, col):
            continue

        total = df[col].dropna().shape[0]
        if total == 0:
            continue

        ranges = resolve_kpi_ranges(kpi, user_id, values=df[col])
        range_data = []
        cumulative = 0
        for idx, r in enumerate(ranges):
            is_last = (idx == len(ranges) - 1)
            # Use half-open intervals: [min, max) for all except last [min, max]
            if is_last:
                count = df[(df[col] >= r["min"]) & (df[col] <= r["max"])].shape[0]
            else:
                count = df[(df[col] >= r["min"]) & (df[col] < r["max"])].shape[0]
            if count == 0:
                continue

            percent = round((count / total) * 100, 2)
            cumulative += percent
            
            # Include label from DB if available (e.g., "fair", "good", "excellent")
            label = r.get("label", "").strip()
            range_str = f'{label}: {r["min"]} to {r["max"]}' if label else f'{r["min"]} to {r["max"]}'
            
            range_data.append({
                "Range": range_str,
                "percentage": percent,
                "cdf": round(cumulative, 2)
            })

        if range_data:
            range_summary[kpi] = range_data

    return range_summary



# =====================================================
# 4. PCI DISTRIBUTION (STENOGRAPH)
# =====================================================
def generate_pci_distribution(df):
    if not has_valid_series(df, "pci"):
        return

    pci_series = pd.to_numeric(df["pci"], errors="coerce").dropna()
    if pci_series.empty:
        return

    # Count only existing PCI values. Plot on the actual PCI numeric axis so
    # hundreds of PCIs can be shown without writing every label.
    pci_counts = pci_series.astype(int).value_counts()
    if pci_counts.empty:
        return

    # Sort by PCI numeric value for visual continuity
    pci_counts = pci_counts.sort_index()

    x = pci_counts.index.to_numpy()
    y = pci_counts.values

    plt.figure(figsize=(14, 4))

    # Continuous curve
    plt.plot(x, y, color="red", linewidth=1)

    # Points on curve
    plt.scatter(x, y, color="red", s=30)

    min_pci = int(pci_counts.index.min())
    max_pci = int(pci_counts.index.max())
    step = int(max(1, _nice_tick_step(min_pci, max_pci, max_ticks=10)))
    tick_start = int(math.floor(min_pci / step) * step)
    tick_end = int(math.ceil(max_pci / step) * step)
    ticks = list(range(tick_start, tick_end + step, step))

    plt.xticks(ticks, [str(t) for t in ticks], rotation=0)
    plt.xlim(tick_start, tick_end)
    plt.title("PCI Distribution")
    plt.xlabel("PCI")
    plt.ylabel("Sample Count")
    plt.grid(True, axis="both", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "pci_distribution.png"), dpi=300)
    plt.close()

    # Top 30 table (separate and correct)
    top30 = pci_counts.sort_values(ascending=False).head(30).reset_index()
    top30.columns = ["pci", "Sample Count"]

    save_table_image(
        top30,
        "Top 30 PCI Distribution",
        "pci_table.png"
    )

# =====================================================
# 5/6. PCI – POOR RSRP / RSRQ  (per-sample, matches frontend)
# =====================================================
def build_poor_pci_table(df, metric_column, threshold, limit=POOR_PCI_LIMIT):
    """
    Per-sample poor-PCI table matching the frontend.

    The frontend flags PCIs that have *any* poor sample, not PCIs whose mean is
    poor.  So we count raw samples below the threshold, group by PCI, sort by
    poor-sample count and keep the worst `limit` rows (a city can have 100+
    poor PCIs which won't fit the PDF).  `df.attrs["total_poor_pci"]` carries
    the full count so the section title can show "Top N of M".
    """
    if "pci" not in df.columns or metric_column not in df.columns:
        return None

    cols = ["pci", metric_column] + (["band"] if "band" in df.columns else [])
    working = df[cols].copy()
    working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce")
    working = working.dropna(subset=["pci", metric_column])
    working = working.loc[working[metric_column] < threshold]
    if working.empty:
        return None

    agg = {
        "Poor_Samples": (metric_column, "count"),
        "Avg_Value": (metric_column, "mean"),
        "Worst_Value": (metric_column, "min"),
    }
    if "band" in working.columns:
        agg["Bands"] = ("band", _join_unique_non_empty)

    grouped = working.groupby("pci").agg(**agg).reset_index()
    total_poor_pci = int(len(grouped))
    grouped = grouped.sort_values(
        by=["Poor_Samples", "Worst_Value", "pci"],
        ascending=[False, True, True],
    ).head(limit).copy()

    label = metric_column.upper()
    grouped = grouped.rename(columns={
        "pci": "PCI",
        "Poor_Samples": "Poor Samples",
        "Avg_Value": f"Avg {label}",
        "Worst_Value": f"Worst {label}",
    })
    grouped[f"Avg {label}"] = grouped[f"Avg {label}"].round(2)
    grouped[f"Worst {label}"] = grouped[f"Worst {label}"].round(2)
    grouped.attrs["total_poor_pci"] = total_poor_pci
    return grouped


def generate_pci_poor_rsrp(df):
    table = build_poor_pci_table(df, "rsrp", -105)
    if table is None:
        return None
    save_table_image(table, "PCI with Poor RSRP (< -105 dBm)", "pci_poor_rsrp.png")
    return table


def generate_pci_poor_rsrq(df):
    table = build_poor_pci_table(df, "rsrq", -14)
    if table is None:
        return None
    save_table_image(table, "PCI with Poor RSRQ (< -14 dB)", "pci_poor_rsrq.png")
    return table


# =====================================================
# NATIVE TABLE DATA BUILDERS
# Return DataFrames so the PDF generator can render crisp, selectable,
# tiny native ReportLab tables instead of embedding blurry matplotlib PNGs.
# Keyed by the same filenames the PDF layout already references.
# =====================================================

def _format_optional_avg(df, column, decimals=1, suffix=""):
    if not has_valid_series(df, column):
        return "N/A"
    value = round(pd.to_numeric(df[column], errors="coerce").mean(), decimals)
    return f"{value} {suffix}".strip()


def _drop_empty_na_columns(df):
    """Hide columns that contain no real display value (e.g. all Category=N/A)."""
    keep = []
    for col in df.columns:
        values = df[col].dropna().astype(str).str.strip()
        values = values[~values.str.lower().isin({"", "n/a", "na", "none", "nan"})]
        if not values.empty:
            keep.append(col)
    return df[keep] if keep else df


def build_band_native_table(band_summary):
    if not band_summary:
        return {}
    rows = [
        {
            "Band": item.get("band"),
            "Sample Count": item.get("sample_count"),
            "% Samples": item.get("sample_percentage"),
        }
        for item in band_summary
    ]
    return {"band_table.png": pd.DataFrame(rows)}


def build_drive_native_tables(drive_summary):
    if not drive_summary:
        return {}

    sessions = drive_summary.get("sessions") or []
    distance_values = [row.get("distance") for row in sessions if row.get("distance") is not None]
    distance_available = any(float(value or 0) > 0 for value in distance_values)
    total_distance = drive_summary.get("distance_covered")
    distance_text = (
        f"{float(total_distance):.2f} KM"
        if distance_available and total_distance is not None else "N/A"
    )

    tables = {
        "drive_summary.png": pd.DataFrame([
            {"Metric": "Distance Covered", "Value": distance_text},
            {"Metric": "Total Samples", "Value": f"{drive_summary.get('total_samples', 0):,}"},
            {"Metric": "Total Sessions", "Value": drive_summary.get("total_sessions", "N/A")},
            {"Metric": "Number of Days", "Value": f"{drive_summary.get('number_of_days', 'N/A')} Days"},
        ])
    }

    if sessions:
        rows = []
        for row in sessions:
            distance = row.get("distance")
            distance_display = (
                f"{float(distance):.6f}"
                if distance_available and distance is not None else "N/A"
            )
            rows.append({
                "Session": row.get("session_id"),
                "Date": row.get("date"),
                "Start": row.get("start_time"),
                "End": row.get("end_time"),
                "Duration": row.get("duration"),
                "Distance": distance_display,
            })
        tables["session_table.png"] = pd.DataFrame(rows)

    return tables


def build_kpi_range_native_tables(metadata):
    if not isinstance(metadata, dict):
        return {}

    kpi_summary = metadata.get("kpi_summary") or metadata.get("kpi_analysis") or {}
    if not isinstance(kpi_summary, dict):
        return {}

    filename_by_kpi = {
        "RSRP": "rsrp_range_table.png",
        "RSRQ": "rsrq_range_table.png",
        "SINR": "sinr_range_table.png",
        "DL": "dl_range_table.png",
        "DL Throughput": "dl_range_table.png",
        "UL": "ul_range_table.png",
        "UL Throughput": "ul_range_table.png",
        "MOS": "mos_range_table.png",
    }

    tables = {}
    for kpi, filename in filename_by_kpi.items():
        data = kpi_summary.get(kpi)
        if not isinstance(data, dict):
            continue
        distribution = data.get("distribution")
        if not isinstance(distribution, list) or not distribution:
            continue
        rows = []
        for item in distribution:
            if not isinstance(item, dict):
                continue
            rows.append({
                "Range": item.get("Range") or item.get("range"),
                "% Of Samples": item.get("percentage"),
                "CDF": item.get("cdf"),
            })
        if rows:
            tables[filename] = pd.DataFrame(rows)

    return tables


def build_app_analytics_native_tables(df):
    app_col = None
    if "apps" in df.columns and not df["apps"].isna().all():
        app_col = "apps"
    elif "app_name" in df.columns and not df["app_name"].isna().all():
        app_col = "app_name"
    if not app_col:
        return {}

    category_col = "category" if "category" in df.columns else (
        "app_category" if "app_category" in df.columns else None
    )

    numeric_cols = ["rsrp", "rsrq", "sinr", "dl_tpt", "ul_tpt", "mos", "latency", "jitter", "packet_loss"]
    df_clean = df.copy()
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    part1_rows, part2_rows = [], []
    for app in df_clean[app_col].dropna().unique():
        app_df = df_clean[df_clean[app_col] == app].copy()
        if app_df.empty:
            continue

        category = "N/A"
        if category_col and category_col in app_df.columns:
            cat_vals = app_df[category_col].dropna().astype(str).str.strip()
            cat_vals = cat_vals[~cat_vals.str.lower().isin({"", "n/a", "na", "none", "nan"})]
            if not cat_vals.empty:
                category = cat_vals.iloc[0]

        sessions = app_df["session_id"].nunique() if "session_id" in app_df.columns else "N/A"
        duration = "N/A"
        time_col = "timestamp" if "timestamp" in app_df.columns else ("time" if "time" in app_df.columns else None)
        if time_col:
            times = pd.to_datetime(app_df[time_col], errors="coerce")
            if times.notna().any():
                seconds = int((times.max() - times.min()).total_seconds())
                duration = f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

        part1_rows.append({
            "App Name": app,
            "Category": category,
            "Sessions": sessions,
            "Samples": len(app_df),
            "Duration": duration,
            "RSRP Avg": _format_optional_avg(app_df, "rsrp"),
            "RSRQ Avg": _format_optional_avg(app_df, "rsrq"),
            "SINR Avg": _format_optional_avg(app_df, "sinr"),
            "DL Avg": _format_optional_avg(app_df, "dl_tpt", suffix="Mbps"),
        })
        part2_rows.append({
            "App Name": app,
            "Category": category,
            "UL Avg": _format_optional_avg(app_df, "ul_tpt", suffix="Mbps"),
            "MOS Avg": _format_optional_avg(app_df, "mos", decimals=2),
            "Latency Avg": _format_optional_avg(app_df, "latency", suffix="ms"),
            "Jitter Avg": _format_optional_avg(app_df, "jitter", suffix="ms"),
            "Loss Avg": _format_optional_avg(app_df, "packet_loss", decimals=2, suffix="%"),
        })

    tables = {}
    if part1_rows:
        tables["app_analytics_part1.png"] = _drop_empty_na_columns(pd.DataFrame(part1_rows))
        tables["app_analytics_part2.png"] = _drop_empty_na_columns(pd.DataFrame(part2_rows))
    return tables


def build_misc_native_tables(df):
    tables = {}

    kpi_rows = []
    for kpi, cfg in KPI_CONFIG.items():
        if cfg.get("type") != "range":
            continue
        col = cfg["column"]
        if not has_valid_series(df, col):
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        kpi_rows.append({
            "KPI": kpi,
            "Average": round(values.mean(), 2),
            "Min": round(values.min(), 2),
            "Max": round(values.max(), 2),
            "Median": round(values.median(), 2),
        })
    if kpi_rows:
        tables["kpi_summary.png"] = pd.DataFrame(kpi_rows)

    if has_valid_series(df, "pci"):
        pci_counts = df["pci"].dropna().value_counts().sort_values(ascending=False).head(30)
        pci_table = pci_counts.reset_index()
        pci_table.columns = ["PCI", "Sample Count"]
        tables["pci_table.png"] = pci_table

    quality = {}
    if has_valid_series(df, "latency"):
        quality["Avg Latency (ms)"] = round(pd.to_numeric(df["latency"], errors="coerce").mean(), 2)
    if has_valid_series(df, "jitter"):
        quality["Avg Jitter (ms)"] = round(pd.to_numeric(df["jitter"], errors="coerce").mean(), 2)
    if has_valid_series(df, "packet_loss"):
        quality["Avg Packet Loss (%)"] = round(pd.to_numeric(df["packet_loss"], errors="coerce").mean(), 2)
    if quality:
        tables["network_quality_summary.png"] = pd.DataFrame([quality])

    poor_rows = []
    if has_valid_series(df, "rsrp"):
        rsrp_values = pd.to_numeric(df["rsrp"], errors="coerce").dropna()
        poor_count = int((rsrp_values < -105).sum())
        poor_rows.append({
            "KPI": "RSRP",
            "Threshold": "< -105 dBm",
            "Poor Samples": poor_count,
            "Total Samples": len(rsrp_values),
            "Percentage": f"{(poor_count / len(rsrp_values) * 100):.2f}%" if len(rsrp_values) else "0.00%",
        })
    if has_valid_series(df, "rsrq"):
        rsrq_values = pd.to_numeric(df["rsrq"], errors="coerce").dropna()
        poor_count = int((rsrq_values < -14).sum())
        poor_rows.append({
            "KPI": "RSRQ",
            "Threshold": "< -14 dB",
            "Poor Samples": poor_count,
            "Total Samples": len(rsrq_values),
            "Percentage": f"{(poor_count / len(rsrq_values) * 100):.2f}%" if len(rsrq_values) else "0.00%",
        })
    if poor_rows:
        tables["poor_kpi_stats.png"] = pd.DataFrame(poor_rows)

    for column, filename in [
        ("speed", "speed_stats.png"),
        ("latency", "latency_stats.png"),
        ("jitter", "jitter_stats.png"),
    ]:
        if has_valid_series(df, column):
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            tables[filename] = pd.DataFrame([{
                "Average": round(values.mean(), 2),
                "Min": round(values.min(), 2),
                "Max": round(values.max(), 2),
            }])

    if "indoor_outdoor" in df.columns and not df["indoor_outdoor"].isna().all():
        rows = []
        numeric_df = df.copy()
        for col in ["rsrp", "rsrq", "sinr", "mos", "dl_tpt", "ul_tpt"]:
            if col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
        for env in ["Indoor", "Outdoor"]:
            env_df = numeric_df[numeric_df["indoor_outdoor"] == env]
            if not env_df.empty:
                rows.append({
                    "Environment": env,
                    "Samples": len(env_df),
                    "Avg RSRP": _format_optional_avg(env_df, "rsrp"),
                    "Avg RSRQ": _format_optional_avg(env_df, "rsrq"),
                    "Avg SINR": _format_optional_avg(env_df, "sinr", decimals=2),
                    "Avg MOS": _format_optional_avg(env_df, "mos", decimals=2),
                    "Avg DL (Mbps)": _format_optional_avg(env_df, "dl_tpt", decimals=2),
                    "Avg UL (Mbps)": _format_optional_avg(env_df, "ul_tpt", decimals=2),
                })
        if rows:
            tables["indoor_outdoor_stats.png"] = _drop_empty_na_columns(pd.DataFrame(rows))

    return tables


def build_native_table_data(report_df, metadata=None, band_summary=None, drive_summary=None):
    """
    Build the full {filename: DataFrame} dict of native tables for the PDF.
    Tables present here are rendered as crisp native ReportLab tables; anything
    absent falls back to its matplotlib PNG.
    """
    tables = {
        "pci_poor_rsrp.png": build_poor_pci_table(report_df, "rsrp", -105),
        "pci_poor_rsrq.png": build_poor_pci_table(report_df, "rsrq", -14),
    }
    tables.update(build_band_native_table(band_summary))
    tables.update(build_drive_native_tables(drive_summary))
    tables.update(build_app_analytics_native_tables(report_df))
    tables.update(build_misc_native_tables(report_df))
    tables.update(build_kpi_range_native_tables(metadata))
    return {
        filename: table_df
        for filename, table_df in tables.items()
        if table_df is not None and not table_df.empty
    }


# =====================================================
# 7. QOS METRICS + HISTOGRAMS (FIXED DOMAIN-SPECIFIC RANGES)
# =====================================================
def generate_qos_metrics(df, column, title, prefix):
    if not has_valid_series(df, column):
        return

    values = df[column].dropna()
    if values.empty:
        return

    stats_df = pd.DataFrame([{
        "Average": round(values.mean(), 2),
        "Min": round(values.min(), 2),
        "Max": round(values.max(), 2)
    }])

    save_table_image(stats_df, f"{title} Statistics", f"{prefix}_stats.png")

    # Fixed domain-specific ranges with legends
    if prefix == "speed":
        bins = [0, 5, 20, 40, 80, 100, float('inf')]
        labels = ['0-5\nWalking', '5-20\nCycling', '20-40\nModerate', '40-80\nFast', '80-100\nVery Fast', '100+\nHighway']
        colors = ['#8BC34A', '#4CAF50', '#FFC107', '#FF9800', '#FF5722', '#D32F2F']
        title_text = f"{title} Distribution (km/h)"
    elif prefix == "latency":
        bins = [0, 20, 50, 100, 150, float('inf')]
        labels = ['0-20\nExcellent', '20-50\nGood', '50-100\nFair', '100-150\nPoor', '150+\nBad']
        colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
        title_text = f"{title} Distribution (ms)"
    elif prefix == "jitter":
        bins = [0, 5, 10, 20, 30, float('inf')]
        labels = ['0-5\nExcellent', '5-10\nGood', '10-20\nFair', '20-30\nPoor', '30+\nBad']
        colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
        title_text = f"{title} Distribution (ms)"
    elif column == "packet_loss":
        bins = [0, 1, 3, 5, 10, float('inf')]
        labels = ['0-1%\nExcellent', '1-3%\nGood', '3-5%\nFair', '5-10%\nPoor', '10%+\nBad']
        colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
        title_text = f"{title} Distribution (%)"
    else:
        # Fallback to old behavior
        return

    # Categorize values into bins
    categorized = pd.cut(values, bins=bins, labels=range(len(labels)), include_lowest=True, right=False)
    counts = categorized.value_counts().sort_index()
    
    # Create figure with histogram and embedded legend
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)
    
    # Histogram subplot
    ax_hist = fig.add_subplot(gs[0])
    
    # Plot bars with counts
    bar_positions = range(len(labels))
    bars = ax_hist.bar(bar_positions, [counts.get(i, 0) for i in range(len(labels))], 
                       color=colors, edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for i, (bar, label) in enumerate(zip(bars, labels)):
        height = bar.get_height()
        if height > 0:
            ax_hist.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_hist.set_xticks(bar_positions)
    ax_hist.set_xticklabels(labels, fontsize=9)
    ax_hist.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax_hist.set_title(title_text, fontsize=13, fontweight='bold', pad=15)
    ax_hist.grid(axis='y', alpha=0.3, linestyle='--')
    ax_hist.set_axisbelow(True)
    
    # Legend subplot
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')
    
    # Create legend content
    legend_items = []
    for i, (label, color) in enumerate(zip(labels, colors)):
        count = counts.get(i, 0)
        # Extract just the range part (before newline)
        range_part = label.split('\n')[0]
        quality_part = label.split('\n')[1] if '\n' in label else ''
        legend_items.append((range_part, quality_part, color, int(count)))
    
    # Draw legend manually
    y_pos = 0.95
    ax_legend.text(0.5, y_pos, 'Legend', ha='center', va='top', 
                  fontsize=12, fontweight='bold', transform=ax_legend.transAxes)
    y_pos -= 0.08
    
    ax_legend.plot([0.1, 0.9], [y_pos, y_pos], 'k-', linewidth=1.5, transform=ax_legend.transAxes)
    y_pos -= 0.05
    
    for range_text, quality_text, color, count in legend_items:
        # Color box
        rect = plt.Rectangle((0.05, y_pos - 0.03), 0.15, 0.06, 
                            facecolor=color, edgecolor='black', linewidth=1,
                            transform=ax_legend.transAxes)
        ax_legend.add_patch(rect)
        
        # Text
        text_line = f"{range_text}"
        if quality_text:
            text_line += f" - {quality_text}"
        
        ax_legend.text(0.25, y_pos, text_line, 
                      va='center', fontsize=9, transform=ax_legend.transAxes)
        
        y_pos -= 0.10
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, f"{prefix}_hist.png"), dpi=200, bbox_inches='tight')
    plt.close()


# =====================================================
# 8. NETWORK QUALITY SUMMARY
# =====================================================
def generate_network_quality_summary(df):
    if not any([
        has_valid_series(df, "latency"),
        has_valid_series(df, "jitter"),
        "packet_loss" in df.columns
    ]):
        return

    summary = {
        "Avg Latency (ms)": round(df["latency"].mean(), 2) if has_valid_series(df, "latency") else "N/A",
        "Avg Jitter (ms)": round(df["jitter"].mean(), 2) if has_valid_series(df, "jitter") else "N/A",
        "Avg Packet Loss (%)": round(df["packet_loss"].mean(), 2) if has_valid_series(df, "packet_loss") else "N/A"
    }

    summary_df = pd.DataFrame([summary])
    save_table_image(summary_df, "Network Quality Summary", "network_quality_summary.png")


# =====================================================
# 9. POOR RSRP/RSRQ STATISTICS
# =====================================================
def generate_poor_kpi_stats(df):
    """Generate poor RSRP and RSRQ statistics"""
    stats = []
    
    # RSRP poor stats
    if has_valid_series(df, "rsrp"):
        rsrp_poor = df[df["rsrp"] < -105]
        rsrp_total = len(df[df["rsrp"].notna()])
        if rsrp_total > 0:
            rsrp_poor_pct = (len(rsrp_poor) / rsrp_total) * 100
            stats.append({
                "KPI": "RSRP",
                "Threshold": "< -105 dBm",
                "Poor Samples": len(rsrp_poor),
                "Total Samples": rsrp_total,
                "Percentage": f"{rsrp_poor_pct:.2f}%"
            })
    
    # RSRQ poor stats
    if has_valid_series(df, "rsrq"):
        rsrq_poor = df[df["rsrq"] < -14]
        rsrq_total = len(df[df["rsrq"].notna()])
        if rsrq_total > 0:
            rsrq_poor_pct = (len(rsrq_poor) / rsrq_total) * 100
            stats.append({
                "KPI": "RSRQ",
                "Threshold": "< -14 dB",
                "Poor Samples": len(rsrq_poor),
                "Total Samples": rsrq_total,
                "Percentage": f"{rsrq_poor_pct:.2f}%"
            })
    
    if stats:
        stats_df = pd.DataFrame(stats)
        save_table_image(stats_df, "Poor KPI Statistics", "poor_kpi_stats.png")


# =====================================================
# 10. APP ANALYTICS
# =====================================================
def generate_app_analytics(df):
    """Generate application-wise analytics with all KPI columns"""
    # Try 'apps' column first, fallback to 'app_name'
    app_col = None
    if "apps" in df.columns and not df["apps"].isna().all():
        app_col = "apps"
    elif "app_name" in df.columns and not df["app_name"].isna().all():
        app_col = "app_name"
        print("Warning: 'apps' column is null/missing, using 'app_name' instead")
    else:
        print("Warning: No valid app column found (apps or app_name)")
        return None
    
    # At this point app_col is set to valid column
    
    # Check for category column
    category_col = None
    if "category" in df.columns:
        category_col = "category"
    elif "app_category" in df.columns:
        category_col = "app_category"
    
    # Ensure numeric types for all KPI columns
    numeric_cols = ["rsrp", "rsrq", "sinr", "dl_tpt", "ul_tpt", "mos", "latency", "jitter", "packet_loss"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Calculate session count and duration per app
    app_stats_part1 = []
    app_stats_part2 = []
    
    # Create a clean copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    for app in df_clean[app_col].dropna().unique():
        app_df = df_clean[df_clean[app_col] == app].copy()  # Use .copy() to avoid SettingWithCopyWarning
        
        if len(app_df) > 0:
            # Get category
            category = "N/A"
            if category_col and category_col in df.columns:
                cat_vals = app_df[category_col].dropna().unique()
                if len(cat_vals) > 0:
                    category = str(cat_vals[0])
            
            # Calculate sessions and duration
            sessions = app_df["session_id"].nunique() if "session_id" in app_df.columns else "N/A"
            
            # Duration calculation
            duration = "N/A"
            if "timestamp" in app_df.columns or "time" in app_df.columns:
                time_col = "timestamp" if "timestamp" in app_df.columns else "time"
                try:
                    app_df[time_col] = pd.to_datetime(app_df[time_col], errors="coerce")
                    if app_df[time_col].notna().any():
                        time_range = app_df[time_col].max() - app_df[time_col].min()
                        total_seconds = int(time_range.total_seconds())
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        seconds = total_seconds % 60
                        duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                except:
                    pass
            
            # Part 1: Basic KPIs
            app_stats_part1.append({
                "App Name": app,
                "Category": category,
                "Sessions": sessions,
                "Samples": len(app_df),
                "Duration": duration,
                "RSRP (Avg)": round(app_df["rsrp"].mean(), 1) if has_valid_series(app_df, "rsrp") else "N/A",
                "RSRQ (Avg)": round(app_df["rsrq"].mean(), 1) if has_valid_series(app_df, "rsrq") else "N/A",
                "SINR (Avg)": round(app_df["sinr"].mean(), 1) if has_valid_series(app_df, "sinr") else "N/A",
                "DL (Avg)": f"{round(app_df['dl_tpt'].mean(), 1)} Mbps" if has_valid_series(app_df, "dl_tpt") else "N/A"
            })
            
            # Part 2: QoS KPIs
            app_stats_part2.append({
                "App Name": app,
                "Category": category,
                "UL (Avg)": f"{round(app_df['ul_tpt'].mean(), 1)} Mbps" if has_valid_series(app_df, "ul_tpt") else "N/A",
                "MOS (Avg)": round(app_df["mos"].mean(), 2) if has_valid_series(app_df, "mos") else "N/A",
                "Latency (Avg)": f"{round(app_df['latency'].mean(), 1)} ms" if has_valid_series(app_df, "latency") else "N/A",
                "Jitter (Avg)": f"{round(app_df['jitter'].mean(), 1)} ms" if has_valid_series(app_df, "jitter") else "N/A",
                "Loss % (Avg)": f"{round(app_df['packet_loss'].mean(), 2)}%" if has_valid_series(app_df, "packet_loss") else "N/A"
            })
    
    if app_stats_part1:
        # Generate Part 1 table
        app_df_1 = pd.DataFrame(app_stats_part1)
        # Save without title to avoid caption in PDF
        save_table_image(app_df_1, "", "app_analytics_part1.png")
        print(f"Generated app analytics part 1 using column: {app_col}")
        
        # Generate Part 2 table
        app_df_2 = pd.DataFrame(app_stats_part2)
        # Save without title to avoid caption in PDF
        save_table_image(app_df_2, "", "app_analytics_part2.png")
        print(f"Generated app analytics part 2 using column: {app_col}")
        
        return {"app_col": app_col, "apps_count": len(app_stats_part1)}
    else:
        print(f"Warning: No apps found in {app_col} column")
        return None


# =====================================================
# 11. INDOOR VS OUTDOOR STATISTICS
# =====================================================
def generate_indoor_outdoor_stats(df):
    """Generate indoor vs outdoor statistics"""
    if "indoor_outdoor" not in df.columns or df["indoor_outdoor"].isna().all():
        return
    
    required_cols = ["rsrp", "rsrq", "sinr", "mos", "dl_tpt", "ul_tpt"]
    if not all(col in df.columns for col in required_cols):
        return
    
    # Ensure numeric types
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    env_stats = []
    for env in ["Indoor", "Outdoor"]:
        env_df = df[df["indoor_outdoor"] == env]
        
        if len(env_df) > 0:
            env_stats.append({
                "Environment": env,
                "Samples": len(env_df),
                "Avg RSRP": round(env_df["rsrp"].mean(), 1) if has_valid_series(env_df, "rsrp") else "N/A",
                "Avg RSRQ": round(env_df["rsrq"].mean(), 1) if has_valid_series(env_df, "rsrq") else "N/A",
                "Avg SINR": round(env_df["sinr"].mean(), 2) if has_valid_series(env_df, "sinr") else "N/A",
                "Avg MOS": round(env_df["mos"].mean(), 2) if has_valid_series(env_df, "mos") else "N/A",
                "Avg DL (Mbps)": round(env_df["dl_tpt"].mean(), 2) if has_valid_series(env_df, "dl_tpt") else "N/A",
                "Avg UL (Mbps)": round(env_df["ul_tpt"].mean(), 2) if has_valid_series(env_df, "ul_tpt") else "N/A"
            })
    
    if env_stats:
        env_df = pd.DataFrame(env_stats)
        # Save without title to avoid caption in PDF
        save_table_image(env_df, "", "indoor_outdoor_stats.png")


# =====================================================
# 12. DRIVE SUMMARY (IMAGES)
# =====================================================
def format_duration(seconds):
    """Format duration from seconds to readable format"""
    if pd.isna(seconds) or seconds <= 0:
        return "NA"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if hours > 0:
        if minutes > 0:
            return f"{hours} hours {minutes} min"
        return f"{hours} hours"
    return f"{minutes} min"


def get_session_data_for_drive_summary(session_ids: list):
    """Fetch session data from database"""
    from .db import get_sessions_by_ids
    
    if not session_ids:
        return None
    
    try:
        return get_sessions_by_ids(session_ids)
    except Exception as e:
        print(f"ERROR: Failed to fetch session data: {e}")
        return None


def _coerce_report_datetime(value):
    if not isinstance(value, dict):
        return value

    keys = {str(k).lower(): k for k in value.keys()}
    if "year" in keys and value.get(keys["year"]) is not None:
        if value.get(keys.get("isvaliddatetime", "")) is False:
            return None
        try:
            microsecond = int(value.get(keys.get("microsecond", ""), 0) or 0)
            millisecond = int(value.get(keys.get("millisecond", ""), 0) or 0)
            if not microsecond and millisecond:
                microsecond = millisecond * 1000
            return datetime(
                int(value[keys["year"]]),
                int(value[keys["month"]]),
                int(value[keys["day"]]),
                int(value.get(keys.get("hour", ""), 0) or 0),
                int(value.get(keys.get("minute", ""), 0) or 0),
                int(value.get(keys.get("second", ""), 0) or 0),
                microsecond,
            )
        except (KeyError, TypeError, ValueError):
            return None

    for nested_key in ("value", "datetime", "dateTime", "date", "timestamp", "Timestamp"):
        if nested_key in value:
            return _coerce_report_datetime(value.get(nested_key))
    return None


def _coerce_report_datetime_series(series: pd.Series) -> pd.Series:
    return series.apply(_coerce_report_datetime)


def _build_session_df_from_network_logs(network_df: pd.DataFrame) -> pd.DataFrame | None:
    if network_df is None or network_df.empty:
        return None
    if "session_id" not in network_df.columns or "timestamp" not in network_df.columns:
        return None

    df = network_df[["session_id", "timestamp"]].copy()
    df["timestamp"] = _coerce_report_datetime_series(df["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return None

    grouped = df.groupby("session_id")["timestamp"].agg(["min", "max"]).reset_index()
    grouped.columns = ["id", "start_time", "end_time"]
    return grouped


def _normalize_session_times(session_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if session_df is None or session_df.empty:
        return None

    normalized = session_df.copy()
    normalized["start_time"] = _coerce_report_datetime_series(normalized["start_time"])
    normalized["end_time"] = _coerce_report_datetime_series(normalized["end_time"])
    normalized["start_time"] = pd.to_datetime(normalized["start_time"], errors="coerce")
    normalized["end_time"] = pd.to_datetime(normalized["end_time"], errors="coerce")
    normalized = normalized.dropna(subset=["start_time", "end_time"])
    if normalized.empty:
        return None
    return normalized


_EARTH_RADIUS_KM = 6371.0
_MAX_PLAUSIBLE_SPEED_KMH = 150.0


def _consecutive_haversine_km(lat: np.ndarray, lon: np.ndarray, timestamp: pd.Series | None = None):
    """
    Sum of Haversine distances between consecutive (lat, lon) points, in km,
    rejecting hops that imply a physically-implausible instantaneous speed
    (GPS glitches: single-second position jumps that aren't real movement).
    Returns (kept_km, rejected_hop_count, rejected_km).
    """
    if len(lat) < 2:
        return 0.0, 0, 0.0

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    hop_km = _EARTH_RADIUS_KM * c

    if timestamp is not None and len(timestamp) == len(lat):
        dt_sec = pd.Series(timestamp).diff().dt.total_seconds().to_numpy()[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            dt_hr = np.where(dt_sec > 0, dt_sec / 3600.0, np.nan)
            implied_speed_kmh = np.where(np.isnan(dt_hr), 0.0, hop_km / dt_hr)
        valid = implied_speed_kmh <= _MAX_PLAUSIBLE_SPEED_KMH
        return float(hop_km[valid].sum()), int((~valid).sum()), float(hop_km[~valid].sum())

    return float(hop_km.sum()), 0, 0.0


def compute_session_distances_km(gps_df: pd.DataFrame, session_ids: list, verbose: bool = True) -> dict:
    """
    Per-session distance computed from consecutive GPS points (lat/lon),
    sorted by timestamp, with GPS-glitch hops rejected. Computed entirely on
    the python side from data already in network_df, independent of
    tbl_session/GetSessions distance values.
    """
    required = {"session_id", "lat", "lon"}
    if gps_df is None or gps_df.empty or not required.issubset(gps_df.columns):
        return {sid: 0.0 for sid in session_ids}

    df = gps_df.dropna(subset=["lat", "lon"]).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    out = {}
    for sid in session_ids:
        sub = df[df["session_id"] == sid]
        if "timestamp" in sub.columns:
            sub = sub.dropna(subset=["timestamp"]).sort_values("timestamp")
        lat = sub["lat"].astype(float).to_numpy()
        lon = sub["lon"].astype(float).to_numpy()
        ts = sub["timestamp"] if "timestamp" in sub.columns else None
        kept_km, rejected_hops, rejected_km = _consecutive_haversine_km(lat, lon, ts)
        out[sid] = round(kept_km, 2)
        if verbose and rejected_hops:
            print(
                f"[distance] session {sid}: rejected {rejected_hops} GPS-glitch hop(s) "
                f"(~{rejected_km:.2f} km of implausible movement, >{_MAX_PLAUSIBLE_SPEED_KMH:.0f} km/h implied speed)"
            )
    return out


def generate_drive_summary_images(
    session_ids: list,
    total_samples: int,
    network_df: pd.DataFrame | None = None,
    gps_df: pd.DataFrame | None = None,
):
    """Generate drive summary and session table images"""

    # Prefer network log timestamps when available (more reliable than tbl_session)
    session_df = _normalize_session_times(_build_session_df_from_network_logs(network_df))
    if session_df is None:
        session_df = _normalize_session_times(get_session_data_for_drive_summary(session_ids))

    if session_df is None or session_df.empty:
        print("WARNING: No session data available for drive summary")
        return None

    # Create a copy to avoid SettingWithCopyWarning
    session_df = session_df.copy()

    # Distance is computed on the python side from consecutive GPS points
    # rather than trusted from tbl_session/GetSessions, so it's available
    # even when that path is missing or unreliable. Prefer gps_df (the
    # polygon-filtered ALL-CELLS dataframe) over network_df (primary-cell
    # filtered) so the GPS trail isn't thinned by the primary-cell marker —
    # distance only depends on the phone's GPS track, not which cell was
    # serving at that instant.
    distance_source = gps_df if gps_df is not None and not gps_df.empty else network_df
    distances_km = compute_session_distances_km(distance_source, session_df["id"].tolist())
    session_df["distance"] = session_df["id"].map(distances_km).fillna(0.0)
    total_distance = session_df["distance"].sum()
    total_sessions = len(session_df)
    session_df["date"] = session_df["start_time"].dt.date
    unique_days = session_df["date"].nunique()
    
    # Get start and end dates
    start_date = session_df["start_time"].min().strftime("%Y-%m-%d")
    end_date = session_df["end_time"].max().strftime("%Y-%m-%d")
    
    # Generate Image 1: Drive Summary
    summary_data = {
        "Metric": [
            "Distance Covered",
            "Total Samples",
            "Total Sessions",
            "Number of Days"
        ],
        "Value": [
            f"{total_distance:.2f} KM",
            f"{total_samples:,}",
            f"{total_sessions}",
            f"{unique_days} Days"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    save_table_image(summary_df, "Drive Summary", "drive_summary.png")
    
    # Generate Image 2: Session Table
    session_df["duration_sec"] = (
        session_df["end_time"] - session_df["start_time"]
    ).dt.total_seconds()
    
    display_df = pd.DataFrame({
        "Session": session_df["id"],
        "Date": session_df["start_time"].dt.strftime("%d-%m-%Y"),
        "Start": session_df["start_time"].dt.strftime("%H:%M"),
        "End": session_df["end_time"].dt.strftime("%H:%M"),
        "Duration": session_df["duration_sec"].apply(format_duration),
        "Distance": session_df["distance"].apply(lambda x: f"{x:.6f}")
    })
    
    save_table_image(display_df, "Session Details", "session_table.png")
    
    print("Generated drive summary images")
    
    # Return metadata structure
    return {
        "distance_covered": round(total_distance, 2),
        "total_samples": total_samples,
        "total_sessions": total_sessions,
        "number_of_days": unique_days,
        "start_date": start_date,
        "end_date": end_date,
        "sessions": [
            {
                "session_id": int(row["id"]),
                "date": row["start_time"].strftime("%d-%m-%Y"),
                "start_time": row["start_time"].strftime("%H:%M"),
                "end_time": row["end_time"].strftime("%H:%M"),
                "duration": format_duration(row["duration_sec"]),
                "distance": round(row["distance"], 6)
            }
            for _, row in session_df.iterrows()
        ]
    }


# =====================================================
# MASTER ENTRY
# =====================================================
def run_kpi_analysis(filtered_df, user_id, kpi_config, session_ids=None, image_dir: str | None = None, gps_df=None):
    global IMAGE_DIR
    if image_dir:
        IMAGE_DIR = image_dir
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    assert isinstance(kpi_config, dict), f"kpi_config corrupted: {type(kpi_config)}"

    # Create a copy to avoid SettingWithCopyWarning when converting to numeric
    filtered_df = filtered_df.copy()

    for kpi, cfg in kpi_config.items():
        if cfg["type"] == "range":
            col = cfg["column"]
            if col in filtered_df.columns:
                try:
                    filtered_df[col] = pd.to_numeric(
                        filtered_df[col],
                        errors="coerce"
                    )
                except Exception as e:
                    print(f"Warning: Failed to convert {col} to numeric: {e}")

    kpi_summary = generate_kpi_summary(filtered_df)
    kpi_ranges = generate_kpi_range_summary(filtered_df, user_id)
    generate_band_summary(filtered_df)
    # generate_kpi_range_tables(filtered_df, user_id)  # Commented out to avoid DB
    generate_pci_distribution(filtered_df)
    generate_pci_poor_rsrp(filtered_df)
    generate_pci_poor_rsrq(filtered_df)

    generate_qos_metrics(filtered_df, "latency", "Latency", "latency")
    generate_qos_metrics(filtered_df, "speed", "Speed", "speed")
    generate_qos_metrics(filtered_df, "jitter", "Jitter", "jitter")

    generate_network_quality_summary(filtered_df)
    generate_poor_kpi_stats(filtered_df)
    generate_app_analytics(filtered_df)
    generate_indoor_outdoor_stats(filtered_df)
    
    # Generate drive summary images if session_ids provided
    drive_summary_metadata = None
    if session_ids:
        drive_summary_metadata = generate_drive_summary_images(
            session_ids,
            len(filtered_df),
            network_df=filtered_df,
            gps_df=gps_df,
        )
    
    if kpi_summary:
        for kpi, ranges in kpi_ranges.items():
            if  kpi in kpi_summary:
                kpi_summary[kpi]["distribution"] = ranges

    return kpi_summary, drive_summary_metadata
