import os
from numpy import around
import pandas as pd
import matplotlib.pyplot as plt


from .kpi_config import KPI_CONFIG
from .threshold_resolver import resolve_kpi_ranges

IMAGE_DIR = "data/images/kpi_analysis"


# =====================================================
# SAFETY HELPERS
# =====================================================
def has_valid_series(df, column):
    return column in df.columns and df[column].dropna().shape[0] > 0


def has_non_empty_df(df):
    return df is not None and not df.empty


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
    if not has_valid_series(df, "band"):
        return

    total = len(df)
    band_df = df["band"].value_counts().reset_index()
    band_df.columns = ["band", "Sample Count"]

    if band_df.empty:
        return

    band_df["% Samples"] = round((band_df["Sample Count"] / total) * 100, 2)
    save_table_image(band_df, "Band Distribution", "band_table.png")

    plt.figure(figsize=(6, 6))
    plt.pie(band_df["% Samples"], labels=band_df["band"], autopct="%1.1f%%")
    plt.title("Band Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "band_pie.png"), dpi=200)
    plt.close()


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

    # Count only existing PCI values
    pci_counts = df["pci"].value_counts()
    if pci_counts.empty:
        return

    # Sort by PCI numeric value for visual continuity
    pci_counts = pci_counts.sort_index()

    x = range(len(pci_counts))            # positional index
    y = pci_counts.values                 # sample counts
    labels = pci_counts.index.astype(str)

    plt.figure(figsize=(14, 4))

    # Continuous curve
    plt.plot(x, y, color="red", linewidth=1)

    # Points on curve
    plt.scatter(x, y, color="red", s=30)

    plt.xticks(x, labels, rotation=90)
    plt.title("PCI Distribution")
    plt.xlabel("PCI")
    plt.ylabel("Sample Count")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "pci_distribution.png"), dpi=200)
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
# 5. PCI – POOR RSRP
# =====================================================
def generate_pci_poor_rsrp(df):
    if not has_valid_series(df, "rsrp"):
        return

    grouped = df.groupby("pci").agg(
        sample_count=("rsrp", "count"),
        avg_rsrp=("rsrp", "mean"),
        bands=("band", lambda x: ",".join(sorted(set(x)))),
        cell_ids=("cell_id", lambda x: ",".join(sorted(set(map(str, x)))))
    ).reset_index()

    poor = grouped.loc[grouped["avg_rsrp"] < -100].copy()
    if poor.empty:
        return

    poor["avg_rsrp"] = poor["avg_rsrp"].round(2)
    save_table_image(poor, "PCI with Poor RSRP (< -100 dBm)", "pci_poor_rsrp.png")


# =====================================================
# 6. PCI – POOR RSRQ
# =====================================================
def generate_pci_poor_rsrq(df):
    if not has_valid_series(df, "rsrq"):
        return

    grouped = df.groupby("pci").agg(
        sample_count=("rsrq", "count"),
        avg_rsrq=("rsrq", "mean"),
        bands=("band", lambda x: ",".join(sorted(set(x)))),
        cell_ids=("cell_id", lambda x: ",".join(sorted(set(map(str, x)))))
    ).reset_index()

    poor = grouped.loc[grouped["avg_rsrq"] < -15].copy()
    if poor.empty:
        return

    poor["avg_rsrq"] = poor["avg_rsrq"].round(2)
    save_table_image(poor, "PCI with Poor RSRQ (< -15 dB)", "pci_poor_rsrq.png")


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


def _build_session_df_from_network_logs(network_df: pd.DataFrame) -> pd.DataFrame | None:
    if network_df is None or network_df.empty:
        return None
    if "session_id" not in network_df.columns or "timestamp" not in network_df.columns:
        return None

    df = network_df[["session_id", "timestamp"]].copy()
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
    normalized["start_time"] = pd.to_datetime(normalized["start_time"], errors="coerce")
    normalized["end_time"] = pd.to_datetime(normalized["end_time"], errors="coerce")
    normalized = normalized.dropna(subset=["start_time", "end_time"])
    if normalized.empty:
        return None
    return normalized


def generate_drive_summary_images(session_ids: list, total_samples: int, network_df: pd.DataFrame | None = None):
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

    # Calculate statistics
    # Ensure distance comes from tbl_session if missing in network logs
    if "distance" not in session_df.columns or session_df["distance"].isna().all():
        dist_df = get_session_data_for_drive_summary(session_ids)
        if dist_df is not None and not dist_df.empty and "distance" in dist_df.columns:
            dist_df = dist_df[["id", "distance"]].dropna(subset=["id"])
            session_df = session_df.merge(dist_df, on="id", how="left", suffixes=("", "_session"))
        else:
            print("WARNING: distance column not found in session data")
            session_df["distance"] = 0.0
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
def run_kpi_analysis(filtered_df, user_id, kpi_config, session_ids=None, image_dir: str | None = None):
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
            network_df=filtered_df
        )
    
    if kpi_summary:
        for kpi, ranges in kpi_ranges.items():
            if  kpi in kpi_summary:
                kpi_summary[kpi]["distribution"] = ranges

    return kpi_summary, drive_summary_metadata
