# src/threshold_resolver.py

import json
from typing import List, Dict, Iterable
from .db import get_user_thresholds


# Explicit KPI → DB column mapping
KPI_DB_JSON_MAP = {
    "RSRP": "rsrp_json",
    "RSRQ": "rsrq_json",
    "SINR": "sinr_json",
    "DL": "dl_thpt_json",
    "UL": "ul_thpt_json",
    "MOS": "mos_json",
}


def resolve_kpi_ranges(
    kpi_name: str,
    user_id: int,
    values: Iterable | None = None,
    debug: bool = False,
    region: str | None = None,
    country_code: str | None = None,
) -> List[Dict]:
    """
    Resolve KPI ranges with FULL DATA COVERAGE.

    Design principles:
    - DB ranges are the base (user intent)
    - Auto ranges are added ONLY when data exceeds coverage
    - No duplicated / noisy logs
    - Static DB config is NOT printed repeatedly
    """

    # --------------------------------------------------
    # 0. Basic validation
    # --------------------------------------------------
    if kpi_name not in KPI_DB_JSON_MAP:
        raise RuntimeError(f"No DB mapping defined for KPI={kpi_name}")

    # --------------------------------------------------
    # 1. Load DB thresholds (ONCE per call, silent)
    # --------------------------------------------------
    db_row = get_user_thresholds(
        user_id,
        region=region,
        country_code=country_code,
    )
    if not db_row:
        return _build_fallback_ranges(kpi_name, values, debug=debug, reason="no_db_row")

    json_col = KPI_DB_JSON_MAP[kpi_name]
    raw_json = db_row.get(json_col)

    if not raw_json or raw_json.strip() in ("", "[]", "null"):
        return _build_fallback_ranges(kpi_name, values, debug=debug, reason="empty_db_json")

    try:
        parsed_json = json.loads(raw_json)
    except Exception as e:
        return _build_fallback_ranges(kpi_name, values, debug=debug, reason=f"invalid_db_json: {e}")

    # Thresholds are stored grouped by technology, e.g.
    # {"default": [{"min":..,"max":..,"color":..}, ...], "5g": [...], "4g": [...], ...}
    # Fall back to treating parsed_json as the ranges list directly for older/flat formats.
    if isinstance(parsed_json, dict):
        db_ranges = parsed_json.get("default") or []
    else:
        db_ranges = parsed_json

    if not db_ranges:
        return _build_fallback_ranges(kpi_name, values, debug=debug, reason="empty_default_ranges")

    # --------------------------------------------------
    # 2. Normalize & sort DB ranges (filter invalid)
    # --------------------------------------------------
    ranges: List[Dict] = []
    for r in db_ranges:
        if not {"min", "max", "color"} <= r.keys():
            raise RuntimeError(f"Invalid range definition: {r}")
        
        min_val = float(r["min"])
        max_val = float(r["max"])
        
        # Skip invalid ranges where min >= max
        if min_val >= max_val:
            print(f"  WARNING: Skipping invalid range {r} (min >= max)")
            continue

        ranges.append({
            "min": min_val,
            "max": max_val,
            "color": r["color"],
            "label": r.get("label", ""),
            "range": r.get("range") or f'{min_val} to {max_val}',
            "source": "db",
        })

    ranges.sort(key=lambda x: x["min"])

    if debug:
        print(f"\n[THRESHOLD RESOLVER] {kpi_name}")
        print("DB RANGES:")
        for r in ranges:
            print(f"  {r['range']}")

    # --------------------------------------------------
    # 3. If no data provided → return DB ranges as-is
    # --------------------------------------------------
    if values is None:
        return ranges

    clean_values = [float(v) for v in values if v is not None]
    if not clean_values:
        return ranges

    data_min = min(clean_values)
    data_max = max(clean_values)

    if debug:
        print(f"DATA RANGE: {data_min:.2f} → {data_max:.2f}")

    final_ranges: List[Dict] = []

    # --------------------------------------------------
    # 4. Lower-bound auto range
    # --------------------------------------------------
    if data_min < ranges[0]["min"]:
        final_ranges.append({
            "min": data_min,
            "max": ranges[0]["min"],
            "color": "#999999",
            "label": "< Min (Auto)",
            "range": f"< {ranges[0]['min']}",
            "source": "auto",
        })
        if debug:
            print("AUTO: lower-bound added")

    # --------------------------------------------------
    # 5. DB ranges + internal gaps
    # --------------------------------------------------
    for i, current in enumerate(ranges):
        final_ranges.append(current)

        if i == len(ranges) - 1:
            break

        next_range = ranges[i + 1]
        if current["max"] < next_range["min"]:
            final_ranges.append({
                "min": current["max"],
                "max": next_range["min"],
                "color": "#BBBBBB",
                "label": "Gap (Auto)",
                "range": f'{current["max"]} to {next_range["min"]}',
                "source": "auto",
            })
            if debug:
                print(f"AUTO: gap {current['max']} → {next_range['min']}")

    # --------------------------------------------------
    # 6. Upper-bound auto range
    # --------------------------------------------------
    if data_max > ranges[-1]["max"]:
        final_ranges.append({
            "min": ranges[-1]["max"],
            "max": data_max,
            "color": "#777777",
            "label": "> Max (Auto)",
            "range": f'> {ranges[-1]["max"]}',
            "source": "auto",
        })
        if debug:
            print("AUTO: upper-bound added")

    if debug:
        print("FINAL RANGES:")
        for r in final_ranges:
            print(f"  {r['range']} ({r['source']})")

    return final_ranges


def _build_fallback_ranges(
    kpi_name: str,
    values: Iterable | None,
    debug: bool = False,
    reason: str | None = None,
) -> List[Dict]:
    """
    Build simple, data-driven fallback ranges when DB thresholds are missing.
    Produces 5 equal-width bins between data min/max. If no data, returns 1 stub range.
    """
    # Coerce values to floats if provided
    clean_values = []
    if values is not None:
        for v in values:
            if v is None:
                continue
            try:
                clean_values.append(float(v))
            except Exception:
                continue

    if not clean_values:
        # Safe stub range so callers don't crash on empty ranges
        fallback = [{
            "min": 0.0,
            "max": 1.0,
            "color": "#999999",
            "label": "Auto",
            "range": "0 to 1",
            "source": "fallback",
        }]
        if debug:
            print(f"[THRESHOLD RESOLVER] {kpi_name} fallback ({reason}): no data, using stub range")
        return fallback

    data_min = min(clean_values)
    data_max = max(clean_values)
    if data_min == data_max:
        data_max = data_min + 1.0

    # 5 equal-width bins
    step = (data_max - data_min) / 5.0
    colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
    ranges: List[Dict] = []
    for i in range(5):
        r_min = data_min + step * i
        r_max = data_min + step * (i + 1)
        if i == 4:
            r_max = data_max
        ranges.append({
            "min": r_min,
            "max": r_max,
            "color": colors[i],
            "label": "",
            "range": f"{round(r_min, 2)} to {round(r_max, 2)}",
            "source": "fallback",
        })

    if debug:
        print(f"[THRESHOLD RESOLVER] {kpi_name} fallback ({reason}): {data_min:.2f} -> {data_max:.2f}")
    return ranges
