"""
Link the site table (site_prediction = the real configured sites) to the drive test for the
sector-swap test case. No database access -- reads the data/raw_*.csv files written by fetch_data.py.

Matching rules (each checked against project 193's real data):
  1. A sector GROUP is site + operator + technology + EARFCN: a swap can only happen between sectors
     of one carrier, and a PCI only identifies a cell within one carrier and technology. The carrier
     is taken from the drive test; the site table's earfcn/band are kept for reporting (in project 193
     every row says 1750 / B1800, which does not match the carriers the phone measured).
  2. Technology must match: 4G sectors only link to LTE drive-test rows. 5G is not testable yet --
     the drive test's 5G rows only carry placeholder cell ids.
  3. ID match: the site id is a real serving eNodeB id in the logs for that operator + technology.
     A sector's PCI is linked through serving rows with that eNodeB + operator + technology + PCI.
  4. PCI fallback for every other site: serving rows with the same operator + technology + PCI that
     do NOT come from an ID-matched eNodeB go to the nearest fallback site within MAX_DISTANCE_M.
     A fallback site whose PCIs are clearly served, right there, by an ID-matched eNodeB is the same
     site listed twice and is skipped.
  5. Neighbour rows have no usable cell identity: operator + technology + EARFCN + PCI -> nearest
     linked site within MAX_DISTANCE_M.
  6. Handovers: consecutive serving rows (<= MAX_HANDOVER_GAP_SEC apart) whose eNodeB / EARFCN / PCI
     changes, with the sides matched by eNodeB ID. Same eNodeB + same EARFCN = the border between two
     sectors of one site ("boundary"); a different eNodeB = the phone "leaving" one sector and
     "entering" another site's sector.
  7. Standing-still repeats are collapsed: one location = one session + one GPS position.
  8. Antenna-endpoint inputs for the expected profile are kept per sector: electrical and mechanical
     tilt, height and -- when the site table has such columns -- antenna model and port. The carrier
     frequency comes from the drive-test EARFCN (3GPP band table); only the channel number is used.

Outputs (data/):
  cells.csv          one row per sector per carrier that can be analysed (+ antenna-endpoint inputs)
  measurements.csv   one row per group + location + PCI: mean RSRP, whether it was the serving cell
  handovers.csv      one row per handover evidence point (boundary / leaving / entering)
  skipped.csv        site-table sectors that could not be checked, with the reason

Run from the ML/ directory:
    venv\\Scripts\\python.exe -m tests.swap_sector.build_dataset
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from tests.swap_sector.antenna_profile import NotTestable, carrier_frequency_mhz

DATA_DIR = Path(__file__).resolve().parent / "data"

MIN_DISTANCE_M = 30.0          # closer than this, GPS error makes the bearing from the site meaningless
MAX_DISTANCE_M = 1000.0
MIN_CARRIER_LOCATIONS = 5      # a PCI must be the serving cell at >= this many locations on a carrier
AZIMUTH_CONFLICT_DEG = 30.0    # site-table rows of one site+operator+technology+PCI further apart = conflict
DUPLICATE_RADIUS_M = 500.0
DUPLICATE_MIN_LOCATIONS = 10
DUPLICATE_SHARE = 0.8
MAX_HANDOVER_GAP_SEC = 3.0
TESTABLE_TECHNOLOGIES = {"LTE"}
# Site-table column names accepted as the real antenna model / port (none exist in project 193).
ANTENNA_MODEL_COLUMNS = ("antenna_model", "antenna_type", "antenna_name", "antenna")
ANTENNA_PORT_COLUMNS = ("pattern_port", "antenna_port", "port")
# Values phones report when the id is unavailable (all seen in project 193) -- never a real eNodeB.
PLACEHOLDER_IDS = {0, 131071, 8388607, 268435455, 562949953421311}
HANDOVER_COLUMNS = [
    "group_id", "kind", "pci", "other_pci", "session_id", "timestamp", "lat", "lon",
    "distance_m", "bearing_deg", "location_id",
]


def normalize_operator(value) -> str | None:
    name = str(value).strip().lower()
    if "jio" in name:
        return "JIO"
    if "airtel" in name:
        return "Airtel"
    if name.startswith("vi") or "vodafone" in name or "idea" in name:
        return "Vi"
    return None


def technology_from_network(value) -> str | None:
    """'4G' and '4G (LTE Anchor - NSA)' are LTE; '5G', '5G NSA', '5G (NR)' are NR; 2G/unknown are dropped."""
    name = str(value).strip().lower()
    if name.startswith("4g") or "lte" in name:
        return "LTE"
    if name.startswith("5g") or name.startswith("nr"):
        return "NR"
    return None


def clean_id(series: pd.Series) -> pd.Series:
    """358.0 -> '358', 1.82 -> '1.82'."""
    return series.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def max_azimuth_gap_deg(azimuths) -> float:
    values = np.unique(np.asarray(azimuths, dtype=float))
    return max((abs((a - b + 180.0) % 360.0 - 180.0) for a, b in combinations(values, 2)), default=0.0)


def circular_mean_deg(azimuths) -> float:
    rad = np.radians(np.asarray(azimuths, dtype=float))
    return float((np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) + 360.0) % 360.0)


def mode_or_empty(series: pd.Series):
    values = series.dropna()
    return values.mode().iat[0] if not values.empty else ""


def location_ids(df: pd.DataFrame) -> pd.Series:
    return df["session_id"].astype(str) + "|" + df["lat"].round(5).astype(str) + "|" + df["lon"].round(5).astype(str)


def nearest_candidate(points: pd.DataFrame, candidates: pd.DataFrame, max_distance_m: float) -> np.ndarray:
    """Row position in `candidates` of each point's nearest site within max_distance_m, else -1."""
    dist = haversine_m(
        candidates["site_lat"].to_numpy()[:, None], candidates["site_lon"].to_numpy()[:, None],
        points["lat"].to_numpy()[None, :], points["lon"].to_numpy()[None, :],
    )
    best = dist.argmin(axis=0)
    best[dist[best, np.arange(dist.shape[1])] > max_distance_m] = -1
    return best


def skipped_rows(df: pd.DataFrame, reason_type, reasons) -> pd.DataFrame:
    out = df[[c for c in ("operator", "site_id", "technology", "pci") if c in df.columns]].copy()
    out["reason_type"] = reason_type
    out["reason"] = reasons
    return out


def load_config_cells() -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """One sector per site + operator + technology + PCI. Rows of that key that disagree on azimuth are a conflict."""
    raw = pd.read_csv(DATA_DIR / "raw_site_prediction.csv", low_memory=False)
    raw["site_id"] = clean_id(raw["site"])
    raw["operator"] = raw["cluster"].map(normalize_operator)
    raw["technology"] = raw["Technology"].map(technology_from_network)
    for col in ("pci", "azimuth", "latitude", "longitude", "earfcn", "e_tilt", "m_tilt", "height"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    model_col = next((c for c in ANTENNA_MODEL_COLUMNS if c in raw.columns), None)
    port_col = next((c for c in ANTENNA_PORT_COLUMNS if c in raw.columns), None)
    usable = (
        raw["operator"].notna() & raw["technology"].notna()
        & ~raw["site_id"].isin(["", "nan", "None"])
        & raw[["pci", "azimuth", "latitude", "longitude"]].notna().all(axis=1)
        & raw["latitude"].between(-90, 90) & raw["longitude"].between(-180, 180)
        & raw["azimuth"].between(0, 360) & raw["pci"].between(0, 1007)
    )
    skipped = [skipped_rows(raw[~usable], "missing site-table values", "site-table row has no operator, technology, PCI, azimuth or location")]

    config = raw[usable].copy()
    config["pci"] = config["pci"].astype(int)
    centers = config.groupby(["operator", "site_id"], as_index=False).agg(
        site_lat=("latitude", "mean"), site_lon=("longitude", "mean")
    )
    kept, conflicts = [], []
    for (operator, site_id, technology, pci), rows in config.groupby(["operator", "site_id", "technology", "pci"]):
        entry = {
            "operator": operator, "site_id": site_id, "technology": technology, "pci": int(pci), "config_rows": len(rows),
            "site_table_earfcn": mode_or_empty(rows["earfcn"]), "site_table_band": mode_or_empty(rows["band"]),
            "e_tilt": mode_or_empty(rows["e_tilt"]) if "e_tilt" in rows else "",
            "m_tilt": mode_or_empty(rows["m_tilt"]) if "m_tilt" in rows else "",
            "height": mode_or_empty(rows["height"]) if "height" in rows else "",
            "antenna_model": mode_or_empty(rows[model_col]) if model_col else "",
            "pattern_port": mode_or_empty(rows[port_col]) if port_col else "",
        }
        if max_azimuth_gap_deg(rows["azimuth"]) > AZIMUTH_CONFLICT_DEG:
            conflicts.append({**entry, "reason_type": "conflicting azimuths",
                              "reason": f"site table gives this PCI different azimuths: {sorted({int(a) for a in rows['azimuth']})}"})
        else:
            kept.append({**entry, "azimuth": round(circular_mean_deg(rows["azimuth"]), 1)})
    skipped.append(pd.DataFrame(conflicts))
    cells = pd.DataFrame(kept).merge(centers, on=["operator", "site_id"], how="left")

    untestable = cells[~cells["technology"].isin(TESTABLE_TECHNOLOGIES)]
    skipped.append(skipped_rows(
        untestable, "5G not testable",
        "the drive test's 5G rows only carry placeholder cell ids, so 5G sectors cannot be matched yet",
    ))
    return cells[cells["technology"].isin(TESTABLE_TECHNOLOGIES)].reset_index(drop=True), skipped


def load_logs(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / file_name, low_memory=False)
    df["operator"] = df["m_alpha_long"].map(normalize_operator)
    df["technology"] = df["network"].map(technology_from_network)
    for col in ("lat", "lon", "pci", "earfcn", "rsrp", "nodeb_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["operator", "technology", "lat", "lon", "pci", "earfcn", "rsrp"])
    df = df[(df["lat"] != 0) & (df["lon"] != 0) & df["lat"].between(-90, 90)
            & df["lon"].between(-180, 180) & df["rsrp"].between(-150, -30)
            & df["pci"].between(0, 1007) & (df["earfcn"] >= 0)].copy()
    df = df[(df["technology"] != "LTE") | (df["pci"] <= 503)].copy()
    df["pci"] = df["pci"].astype(int)
    df["earfcn"] = df["earfcn"].astype(int)
    df["band"] = df["band"].fillna("").astype(str)
    real_enb = df["nodeb_id"].where(~df["nodeb_id"].isin(PLACEHOLDER_IDS) & df["nodeb_id"].mod(1).eq(0) & (df["nodeb_id"] > 0))
    df["enb_id"] = real_enb.astype("Int64").astype(str).replace("<NA>", "")
    df["location_id"] = location_ids(df)
    return df.drop_duplicates(subset=["session_id", "timestamp", "operator", "technology", "enb_id", "cell_id", "earfcn", "pci"]).reset_index(drop=True)


def assign_serving_rows(cells: pd.DataFrame, serving: pd.DataFrame):
    """Attach serving rows to site-table sectors: ID match first, then the PCI fallback."""
    cells = cells.copy()
    serving["op_enb"] = serving["operator"] + "|" + serving["technology"] + "|" + serving["enb_id"]
    cell_keys = cells["operator"] + "|" + cells["technology"] + "|" + cells["site_id"]
    served_enbs = set(serving.loc[serving["enb_id"] != "", "op_enb"])
    cells["match_level"] = np.where(cell_keys.isin(served_enbs), "ID", "PCI")
    id_site_keys = set(cell_keys[cells["match_level"] == "ID"])

    by_id = serving.merge(
        cells.loc[cells["match_level"] == "ID", ["operator", "technology", "site_id", "pci"]],
        left_on=["operator", "technology", "enb_id", "pci"], right_on=["operator", "technology", "site_id", "pci"],
    )
    by_id["match_level"] = "ID"

    fallback = cells[cells["match_level"] == "PCI"]
    known = serving[serving["enb_id"] != ""]
    duplicate_of: dict[tuple[str, str, str], str] = {}
    for (operator, technology, site_id), site_cells in fallback.groupby(["operator", "technology", "site_id"]):
        near = known[(known["operator"] == operator) & (known["technology"] == technology) & known["pci"].isin(site_cells["pci"])]
        near = near[haversine_m(site_cells["site_lat"].iloc[0], site_cells["site_lon"].iloc[0], near["lat"], near["lon"]) <= DUPLICATE_RADIUS_M]
        per_location = near.drop_duplicates(["location_id", "enb_id"])
        if per_location["location_id"].nunique() < DUPLICATE_MIN_LOCATIONS:
            continue
        shares = per_location["enb_id"].value_counts(normalize=True)
        if shares.iloc[0] >= DUPLICATE_SHARE and f"{operator}|{technology}|{shares.index[0]}" in id_site_keys:
            duplicate_of[(operator, technology, site_id)] = shares.index[0]
    is_duplicate = np.array(
        [(o, t, s) in duplicate_of for o, t, s in zip(fallback["operator"], fallback["technology"], fallback["site_id"])], dtype=bool
    )
    duplicates = fallback[is_duplicate]
    dup_skipped = skipped_rows(
        duplicates, "duplicate site",
        [f"duplicate of site {duplicate_of[(o, t, s)]}: its PCIs are served here by that eNodeB"
         for o, t, s in zip(duplicates["operator"], duplicates["technology"], duplicates["site_id"])],
    )
    fallback = fallback[~is_duplicate]
    cells = cells.drop(index=duplicates.index)

    unclaimed = serving[~serving["op_enb"].isin(id_site_keys)]
    unclaimed = unclaimed.merge(fallback[["operator", "technology", "pci"]].drop_duplicates(), on=["operator", "technology", "pci"])
    parts = []
    for (operator, technology, pci), points in unclaimed.groupby(["operator", "technology", "pci"]):
        candidates = fallback[(fallback["operator"] == operator) & (fallback["technology"] == technology) & (fallback["pci"] == pci)]
        best = nearest_candidate(points, candidates, MAX_DISTANCE_M)
        kept = points[best >= 0].copy()
        kept["site_id"] = candidates["site_id"].to_numpy()[best[best >= 0]]
        parts.append(kept)
    by_pci = pd.concat(parts, ignore_index=True) if parts else by_id.iloc[0:0].copy()
    by_pci["match_level"] = "PCI"
    return cells, pd.concat([by_id, by_pci], ignore_index=True), dup_skipped


def link_carriers(cells: pd.DataFrame, assigned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Which carriers (EARFCN) each site-table sector is actually served on in the drive test."""
    sector_key = ["operator", "site_id", "technology", "pci"]
    carriers = assigned.groupby(sector_key + ["earfcn"], as_index=False).agg(
        band=("band", lambda s: s.mode().iat[0]), serving_locations=("location_id", "nunique")
    )
    seen_locations = carriers.groupby(sector_key)["serving_locations"].sum()
    kept = carriers[carriers["serving_locations"] >= MIN_CARRIER_LOCATIONS]
    cell_carriers = cells.merge(kept, on=sector_key)
    cell_carriers["group_id"] = (
        cell_carriers["operator"] + "|" + cell_carriers["site_id"] + "|"
        + cell_carriers["technology"] + "|" + cell_carriers["earfcn"].astype(str)
    )

    linked = cells.merge(kept[sector_key].drop_duplicates(), how="left", indicator=True)
    unlinked = linked[linked["_merge"] == "left_only"]
    types, reasons = [], []
    for cell in unlinked.itertuples():
        seen = int(seen_locations.get((cell.operator, cell.site_id, cell.technology, cell.pci), 0))
        if seen == 0:
            types.append("not in drive test")
            reasons.append(
                f"eNodeB {cell.site_id} never served this PCI in the drive test" if cell.match_level == "ID"
                else f"no serving sample with this operator + PCI within {MAX_DISTANCE_M:.0f} m"
            )
        else:
            types.append("too few locations")
            reasons.append(f"served at only {seen} drive-test locations (need {MIN_CARRIER_LOCATIONS} on one carrier)")
    return cell_carriers, skipped_rows(unlinked, types, reasons)


def add_carrier_frequency(cell_carriers: pd.DataFrame) -> pd.DataFrame:
    """Carrier frequency / band from the drive-test EARFCN (channel number only, no RF values)."""
    frequencies, bands = [], []
    for earfcn, technology in zip(cell_carriers["earfcn"], cell_carriers["technology"]):
        try:
            frequency, band = carrier_frequency_mhz(earfcn, technology)
        except NotTestable:
            frequency, band = np.nan, ""
        frequencies.append(frequency)
        bands.append(band)
    return cell_carriers.assign(frequency_mhz=frequencies, frequency_band=bands, frequency_source="DT EARFCN")


def build_measurements(cell_carriers: pd.DataFrame, serving_assigned: pd.DataFrame, neighbour: pd.DataFrame) -> pd.DataFrame:
    carrier_key = ["operator", "site_id", "technology", "pci", "earfcn"]
    carrier_cols = carrier_key + ["group_id", "site_lat", "site_lon"]
    serving_part = serving_assigned.merge(cell_carriers[carrier_cols], on=carrier_key)
    serving_part["is_serving"] = True

    neighbour_key = ["operator", "technology", "earfcn", "pci"]
    candidates_by_key = dict(tuple(cell_carriers[carrier_cols].groupby(neighbour_key)))
    candidate_neighbours = neighbour.merge(cell_carriers[neighbour_key].drop_duplicates(), on=neighbour_key)
    parts = []
    for key, points in candidate_neighbours.groupby(neighbour_key):
        candidates = candidates_by_key[key]
        best = nearest_candidate(points, candidates, MAX_DISTANCE_M)
        kept = points[best >= 0].copy()
        chosen = candidates.iloc[best[best >= 0]]
        for col in ("site_id", "group_id", "site_lat", "site_lon"):
            kept[col] = chosen[col].to_numpy()
        parts.append(kept)
    neighbour_part = pd.concat(parts, ignore_index=True) if parts else serving_part.iloc[0:0].copy()
    neighbour_part["is_serving"] = False

    both = pd.concat([serving_part, neighbour_part], ignore_index=True)
    both["distance_m"] = haversine_m(both["site_lat"], both["site_lon"], both["lat"], both["lon"])
    both = both[both["distance_m"].between(MIN_DISTANCE_M, MAX_DISTANCE_M)].copy()
    both["bearing_deg"] = bearing_deg(both["site_lat"], both["site_lon"], both["lat"], both["lon"])
    return both.groupby(["group_id", "location_id", "pci"], as_index=False).agg(
        session_id=("session_id", "first"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        distance_m=("distance_m", "mean"),
        bearing_deg=("bearing_deg", "first"),
        rsrp=("rsrp", "mean"),
        samples=("rsrp", "size"),
        is_serving=("is_serving", "max"),
    )


def build_handovers(cell_carriers: pd.DataFrame, serving: pd.DataFrame) -> pd.DataFrame:
    """Serving-cell changes in the drive test, with both sides matched to sectors by eNodeB ID."""
    rows = serving.copy()
    rows["cell_key"] = clean_id(rows["cell_id"])
    cell_num = pd.to_numeric(rows["cell_key"], errors="coerce")
    rows["identity_valid"] = (rows["enb_id"] != "") & cell_num.notna() & ~cell_num.isin(PLACEHOLDER_IDS - {0}) & cell_num.ge(0)
    # A carrier/PCI must resolve to exactly one observed cell at this eNodeB.
    identity_keys = ["operator", "technology", "enb_id", "earfcn", "pci"]
    identity_count = rows.groupby(identity_keys)["cell_key"].transform("nunique")
    rows["identity_valid"] &= identity_count.eq(1)
    rows["ts"] = pd.to_datetime(rows["timestamp"], errors="coerce")
    rows = rows.dropna(subset=["ts"]).sort_values(["session_id", "ts"], kind="stable").reset_index(drop=True)
    stream = ["session_id", "operator", "technology"]
    # Ambiguous simultaneous serving records invalidate adjacency on both sides.
    rows["identity_valid"] &= ~rows.duplicated(stream + ["ts"], keep=False)
    prev = rows.groupby(stream)[["ts", "enb_id", "earfcn", "pci", "operator", "technology", "lat", "lon", "identity_valid"]].shift(1)
    gap_sec = (rows["ts"] - prev["ts"]).dt.total_seconds()
    changed = (
        prev["enb_id"].notna() & (gap_sec > 0) & (gap_sec <= MAX_HANDOVER_GAP_SEC)
        & rows["identity_valid"] & prev["identity_valid"].eq(True)
        & (prev["operator"] == rows["operator"]) & (prev["technology"] == rows["technology"])
        & ((prev["enb_id"] != rows["enb_id"]) | (prev["earfcn"] != rows["earfcn"]) | (prev["pci"] != rows["pci"]))
    )
    now, before = rows[changed], prev[changed]
    events = pd.DataFrame({
        "session_id": now["session_id"].to_numpy(),
        "timestamp": now["timestamp"].to_numpy(),
        "lat": (now["lat"].to_numpy() + before["lat"].to_numpy()) / 2,
        "lon": (now["lon"].to_numpy() + before["lon"].to_numpy()) / 2,
        "operator": now["operator"].to_numpy(),
        "technology": now["technology"].to_numpy(),
        "from_enb": before["enb_id"].to_numpy(),
        "from_earfcn": before["earfcn"].astype(int).to_numpy(),
        "from_pci": before["pci"].astype(int).to_numpy(),
        "to_enb": now["enb_id"].to_numpy(),
        "to_earfcn": now["earfcn"].to_numpy(),
        "to_pci": now["pci"].to_numpy(),
    })

    carriers = cell_carriers.loc[
        cell_carriers["match_level"] == "ID", ["operator", "technology", "site_id", "earfcn", "pci", "group_id", "site_lat", "site_lon"]
    ]
    for side in ("from", "to"):
        renamed = carriers.rename(columns={
            "site_id": f"{side}_enb", "earfcn": f"{side}_earfcn", "pci": f"{side}_pci",
            "group_id": f"{side}_group", "site_lat": f"{side}_site_lat", "site_lon": f"{side}_site_lon",
        })
        events = events.merge(renamed, on=["operator", "technology", f"{side}_enb", f"{side}_earfcn", f"{side}_pci"], how="left")

    parts = []
    boundary = events[
        (events["from_enb"] == events["to_enb"]) & (events["from_earfcn"] == events["to_earfcn"])
        & events["from_group"].notna() & (events["from_group"] == events["to_group"])
    ]
    parts.append(pd.DataFrame({
        "group_id": boundary["from_group"], "kind": "boundary",
        "pci": np.minimum(boundary["from_pci"], boundary["to_pci"]),
        "other_pci": np.maximum(boundary["from_pci"], boundary["to_pci"]),
        "session_id": boundary["session_id"], "timestamp": boundary["timestamp"],
        "lat": boundary["lat"], "lon": boundary["lon"],
        "site_lat": boundary["from_site_lat"], "site_lon": boundary["from_site_lon"],
    }))
    other_site = (events["from_enb"] != events["to_enb"]) & events["from_group"].notna() & events["to_group"].notna()
    for side, kind in (("from", "leaving"), ("to", "entering")):
        side_rows = events[other_site & events[f"{side}_group"].notna()]
        parts.append(pd.DataFrame({
            "group_id": side_rows[f"{side}_group"], "kind": kind,
            "pci": side_rows[f"{side}_pci"], "other_pci": -1,
            "session_id": side_rows["session_id"], "timestamp": side_rows["timestamp"],
            "lat": side_rows["lat"], "lon": side_rows["lon"],
            "site_lat": side_rows[f"{side}_site_lat"], "site_lon": side_rows[f"{side}_site_lon"],
        }))
    handovers = pd.concat(parts, ignore_index=True)
    if handovers.empty:
        return pd.DataFrame(columns=HANDOVER_COLUMNS)
    handovers["distance_m"] = haversine_m(handovers["site_lat"], handovers["site_lon"], handovers["lat"], handovers["lon"])
    handovers = handovers[handovers["distance_m"].between(MIN_DISTANCE_M, MAX_DISTANCE_M)].copy()
    handovers["bearing_deg"] = bearing_deg(handovers["site_lat"], handovers["site_lon"], handovers["lat"], handovers["lon"])
    handovers["location_id"] = location_ids(handovers)
    handovers = handovers.drop_duplicates(["group_id", "kind", "pci", "other_pci", "location_id"])
    return handovers[HANDOVER_COLUMNS].reset_index(drop=True)


def main() -> None:
    cells, skipped = load_config_cells()
    serving = load_logs("raw_serving_logs.csv")
    neighbour = load_logs("raw_neighbour_logs.csv")
    # The neighbour list often repeats the serving cell itself -- drop those copies.
    serving_keys = serving[["session_id", "timestamp", "technology", "earfcn", "pci"]]
    neighbour = neighbour.merge(serving_keys, how="left", indicator=True)
    neighbour = neighbour[neighbour["_merge"] == "left_only"].drop(columns="_merge")

    cells, serving_assigned, dup_skipped = assign_serving_rows(cells, serving)
    cell_carriers, unlinked_skipped = link_carriers(cells, serving_assigned)
    cell_carriers = add_carrier_frequency(cell_carriers)
    measurements = build_measurements(cell_carriers, serving_assigned, neighbour)
    handovers = build_handovers(cell_carriers, serving)

    cell_carriers = cell_carriers[[
        "group_id", "operator", "site_id", "technology", "band", "earfcn", "pci", "azimuth", "match_level",
        "serving_locations", "config_rows", "site_table_earfcn", "site_table_band",
        "e_tilt", "m_tilt", "height", "antenna_model", "pattern_port",
        "frequency_mhz", "frequency_band", "frequency_source", "site_lat", "site_lon",
    ]].sort_values(["group_id", "pci"])
    skipped_df = pd.concat(skipped + [dup_skipped, unlinked_skipped], ignore_index=True)
    cell_carriers.to_csv(DATA_DIR / "cells.csv", index=False)
    measurements.to_csv(DATA_DIR / "measurements.csv", index=False)
    handovers.to_csv(DATA_DIR / "handovers.csv", index=False)
    skipped_df.to_csv(DATA_DIR / "skipped.csv", index=False)

    group_sizes = cell_carriers.groupby("group_id").size()
    band_mismatch = (cell_carriers["band"] != cell_carriers["frequency_band"]).sum()
    print(f"[skipped] {len(skipped_df)} site-table sectors by reason:\n{skipped_df['reason_type'].value_counts().to_string()}")
    print(f"[linked] sectors on a carrier: {len(cell_carriers)} | by match: {cell_carriers['match_level'].value_counts().to_dict()}")
    print(f"[groups] site+operator+technology+EARFCN groups: {len(group_sizes)} | with >= 2 sectors: {(group_sizes >= 2).sum()}")
    print(f"[carrier] frequency from EARFCN: {cell_carriers.groupby('frequency_band').size().to_dict()} "
          f"| unknown EARFCN: {int(cell_carriers['frequency_mhz'].isna().sum())} | DT band label differs: {int(band_mismatch)}")
    print(f"[antenna inputs] e_tilt missing: {int(pd.to_numeric(cell_carriers['e_tilt'], errors='coerce').isna().sum())} "
          f"| m_tilt missing: {int(pd.to_numeric(cell_carriers['m_tilt'], errors='coerce').isna().sum())} "
          f"| real antenna model given: {int(cell_carriers['antenna_model'].astype(str).str.strip().replace('nan', '').ne('').sum())}")
    print(f"[measurements] rows: {len(measurements)} | locations: {measurements['location_id'].nunique()} "
          f"| serving share: {measurements['is_serving'].mean():.1%}")
    print(f"[handovers] evidence points: {len(handovers)} | by kind: {handovers['kind'].value_counts().to_dict()} "
          f"| groups with any: {handovers['group_id'].nunique()}")
    print(f"[5G] drive-test NR rows: {int((serving['technology'] == 'NR').sum() + (neighbour['technology'] == 'NR').sum())} "
          f"(placeholder cell ids only -> not testable)")


if __name__ == "__main__":
    main()
