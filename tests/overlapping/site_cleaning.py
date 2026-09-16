"""
Turn raw site_prediction rows into physical antennas, and flag what cannot be trusted.

Why antennas and not cell_id rows (project 193, checked 2026-09-11):
  * one cell_id often carries several rows with different PCI/azimuth; deduplicating on
    site|cell|sector|band|operator silently drops ~80% of rows and keeps an arbitrary azimuth;
  * coverage depends on where an antenna is and where it points, not on its cell id.
So rows of one operator + site are grouped by azimuth (circular, merge_deg apart) and every group
is one antenna with a circular-mean azimuth and median tilt/height/power.

Flags:
  site_config_issue  TOO_MANY_AZIMUTHS (more groups than max_sectors_per_site),
                     AZIMUTH_SMEAR (one group's rows spread wider than allowed),
                     LOCATION_SPREAD (rows of one site id far apart)
                     -> NOT_TESTABLE, and by default left out of the RF surface
  duplicate_of       antenna of a DIFFERENT site id that is the same physical antenna:
                     CO_LOCATED       <= duplicate_distance_m and <= duplicate_azimuth_deg apart
                     SAME_PCI_NEARBY  shares a PCI within same_pci_distance_m (clean PCI sets only)
                     -> DATA_DUPLICATE, excluded from the RF surface (it would double the power)
Representative of a duplicate group: clean config, real integer eNB id (not an Android sentinel),
most raw rows, lowest row id.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tests.overlapping.config import CleaningParams, RFParams
from tests.overlapping.geo import angle_diff_deg, circular_mean_deg, haversine_m

# Placeholder ids Android reports when the real one is unknown; never trust them as eNB ids.
SENTINEL_SITE_IDS = {"0", "1", "2", "65535", "131071", "268435455", "8388607", "2147483647"}


def normalise_operator(value) -> str:
    text_value = str(value if value is not None else "").strip()
    low = text_value.lower()
    if low in {"", "nan", "none", "null", "000 000"}:
        return ""
    if "airtel" in low:
        return "Airtel"
    if "jio" in low:
        return "JIO 4G"
    if low == "vi" or low.startswith("vi ") or "vodafone" in low or "idea" in low:
        return "Vi India"
    return text_value


def site_key(value) -> tuple[str, bool]:
    """(key, looks like a real integer eNB id)."""
    num = pd.to_numeric(value, errors="coerce")
    if pd.notna(num):
        num = float(num)
        if num.is_integer():
            key = str(int(num))
            return key, key not in SENTINEL_SITE_IDS
        return f"{num:.6f}".rstrip("0").rstrip("."), False
    key = str(value).strip()
    return key, key.isdigit() and key not in SENTINEL_SITE_IDS


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def prepare_rows(raw: pd.DataFrame, rf: RFParams, cleaning: CleaningParams) -> tuple[pd.DataFrame, dict]:
    df = raw.reset_index(drop=True)
    operator_source = df["cluster"] if "cluster" in df.columns else df.get("network", pd.Series("", index=df.index))
    keys = [site_key(v) for v in df["site"]]
    rows = pd.DataFrame(
        {
            "row_id": _numeric(df, "id").fillna(pd.Series(np.arange(len(df)), index=df.index)),
            "operator": operator_source.map(normalise_operator),
            "site_key": [k for k, _ in keys],
            "site_id_is_enb": [ok for _, ok in keys],
            "cell_id": df.get("cell_id", pd.Series("", index=df.index)).astype(str),
            "technology": df.get("Technology", pd.Series(cleaning.technology, index=df.index)).astype(str).str.strip(),
            "lat": _numeric(df, "latitude"),
            "lon": _numeric(df, "longitude"),
            "pci": _numeric(df, "pci"),
            "azimuth": _numeric(df, "azimuth") % 360.0,
            "e_tilt": _numeric(df, "e_tilt").fillna(rf.default_electrical_tilt_deg),
            "m_tilt": _numeric(df, "m_tilt").fillna(rf.default_mechanical_tilt_deg),
            "height": _numeric(df, "height"),
            "tx_power": _numeric(df, "tx_power"),
        }
    )
    # Some regions store coordinates as integers with the decimal point stripped.
    rows.loc[rows["lat"].abs() > 90, "lat"] /= 1_000_000.0
    rows.loc[rows["lon"].abs() > 180, "lon"] /= 1_000_000.0
    rows["height"] = rows["height"].where(rows["height"] > 0, rf.default_height_m)
    rows["tx_power_filled"] = rows["tx_power"].isna()
    rows["tx_power"] = rows["tx_power"].fillna(rf.default_tx_power_dbm)

    valid = rows["lat"].between(-90, 90) & rows["lon"].between(-180, 180) & rows["azimuth"].notna()
    valid &= (rows["lat"] != 0) & (rows["operator"] != "") & (rows["site_key"] != "")
    tech_ok = rows["technology"].str.upper() == cleaning.technology.upper()
    report = {
        "raw_rows": int(len(rows)),
        "rows_invalid_location_or_identity": int((~valid).sum()),
        "rows_other_technology": int((valid & ~tech_ok).sum()),
    }
    return rows[valid & tech_ok].reset_index(drop=True), report


def cluster_azimuths(azimuths: np.ndarray, merge_deg: float) -> np.ndarray:
    """Circular single-linkage clustering: consecutive azimuths more than merge_deg apart split."""
    n = azimuths.size
    if n <= 1:
        return np.zeros(n, dtype=int)
    order = np.argsort(azimuths)
    s = azimuths[order]
    gaps = np.diff(np.r_[s, s[0] + 360.0])       # gap after each sorted value, last one wraps
    start = (int(np.argmax(gaps)) + 1) % n       # begin right after the widest gap
    rotated = np.r_[s[start:], s[:start] + 360.0]
    labels_sorted = np.r_[0, np.cumsum(np.diff(rotated) > merge_deg)]
    labels = np.empty(n, dtype=int)
    labels[order[(np.arange(n) + start) % n]] = labels_sorted
    return labels


def _pci_text(values: pd.Series) -> str:
    pcis = sorted({int(v) for v in values.dropna() if 0 <= v <= 1007})
    return ",".join(str(p) for p in pcis)


def build_antennas(rows: pd.DataFrame, cleaning: CleaningParams) -> pd.DataFrame:
    records: list[dict] = []
    for (operator, skey), group in rows.groupby(["operator", "site_key"], sort=True):
        lat, lon = float(group["lat"].median()), float(group["lon"].median())
        location_spread = float(haversine_m(lat, lon, group["lat"], group["lon"]).max())
        labels = cluster_azimuths(group["azimuth"].to_numpy(dtype=float), cleaning.azimuth_merge_deg)
        issues: set[str] = set()
        if labels.max() + 1 > cleaning.max_sectors_per_site:
            issues.add("TOO_MANY_AZIMUTHS")
        if location_spread > cleaning.max_site_location_spread_m:
            issues.add("LOCATION_SPREAD")
        site_records = []
        for label in range(labels.max() + 1):
            member = group[labels == label]
            azimuth = circular_mean_deg(member["azimuth"])
            spread = float(angle_diff_deg(member["azimuth"], azimuth).max())
            if spread > cleaning.max_antenna_azimuth_spread_deg:
                issues.add("AZIMUTH_SMEAR")
            site_records.append(
                {
                    "operator": operator,
                    "site_key": skey,
                    "site_id_is_enb": bool(member["site_id_is_enb"].iloc[0]),
                    "antenna_key": f"{operator}|{skey}|{int(round(azimuth)) % 360:03d}",
                    "lat": lat,
                    "lon": lon,
                    "azimuth": azimuth,
                    "azimuth_spread_deg": spread,
                    "e_tilt": float(member["e_tilt"].median()),
                    "m_tilt": float(member["m_tilt"].median()),
                    "height": float(member["height"].median()),
                    "tx_power": float(member["tx_power"].median()),
                    "tx_power_filled": bool(member["tx_power_filled"].any()),
                    "pcis": _pci_text(member["pci"]),
                    "cell_ids": "|".join(sorted(set(member["cell_id"]))),
                    "raw_rows": int(len(member)),
                    "min_row_id": float(member["row_id"].min()),
                    "site_location_spread_m": location_spread,
                    "site_antenna_count": int(labels.max() + 1),
                }
            )
        for rec in site_records:
            rec["site_config_issue"] = ";".join(sorted(issues))
        records.extend(site_records)
    return pd.DataFrame.from_records(records)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def find_duplicates(antennas: pd.DataFrame, cleaning: CleaningParams) -> pd.DataFrame:
    ant = antennas.reset_index(drop=True).copy()
    ant["duplicate_of"] = ""
    ant["duplicate_rule"] = ""
    ant["duplicate_distance_m"] = np.nan
    for _, idx in ant.groupby("operator").groups.items():
        sub = ant.loc[idx]
        pos = sub.index.to_numpy()
        n = pos.size
        if n < 2:
            continue
        lat, lon, az = (sub[c].to_numpy(dtype=float) for c in ("lat", "lon", "azimuth"))
        dist = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        daz = angle_diff_deg(az[:, None], az[None, :])
        sites = sub["site_key"].to_numpy()
        other_site = sites[:, None] != sites[None, :]
        co_located = other_site & (dist <= cleaning.duplicate_distance_m) & (daz <= cleaning.duplicate_azimuth_deg)

        same_pci = np.zeros_like(co_located)
        pci_sets = [set(p.split(",")) - {""} for p in sub["pcis"]]
        clean_pci = np.array([0 < len(s) <= cleaning.same_pci_max_pcis_per_antenna for s in pci_sets])
        near = other_site & (dist <= cleaning.same_pci_distance_m) & (daz <= cleaning.same_pci_azimuth_deg)
        near &= clean_pci[:, None] & clean_pci[None, :]
        for i, j in np.argwhere(np.triu(near, k=1)):
            if pci_sets[i] & pci_sets[j]:
                same_pci[i, j] = same_pci[j, i] = True

        edges = co_located | same_pci
        uf = _UnionFind(n)
        for i, j in np.argwhere(np.triu(edges, k=1)):
            uf.union(int(i), int(j))
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(uf.find(i), []).append(i)
        for members in groups.values():
            if len(members) < 2:
                continue
            rep = min(
                members,
                key=lambda i: (
                    sub["site_config_issue"].iat[i] != "",
                    not sub["site_id_is_enb"].iat[i],
                    -sub["raw_rows"].iat[i],
                    sub["min_row_id"].iat[i],
                ),
            )
            for i in members:
                if i == rep:
                    continue
                ant.at[pos[i], "duplicate_of"] = sub["antenna_key"].iat[rep]
                ant.at[pos[i], "duplicate_rule"] = "CO_LOCATED" if co_located[i].any() else "SAME_PCI_NEARBY"
                ant.at[pos[i], "duplicate_distance_m"] = float(dist[i, rep])
    return ant


def clean_sites(raw: pd.DataFrame, operator: str, cleaning: CleaningParams, rf: RFParams) -> tuple[pd.DataFrame, dict]:
    rows, report = prepare_rows(raw, rf, cleaning)
    if operator and operator.lower() != "all":
        rows = rows[rows["operator"] == normalise_operator(operator)].reset_index(drop=True)
    report["rows_used"] = int(len(rows))
    if rows.empty:
        return pd.DataFrame(), report

    ant = find_duplicates(build_antennas(rows, cleaning), cleaning)
    ant["is_duplicate"] = ant["duplicate_of"] != ""
    ant["site_ambiguous"] = ant["site_config_issue"] != ""
    ant["rf_include"] = ~ant["is_duplicate"] & (~ant["site_ambiguous"] | cleaning.ambiguous_sites_as_alternatives)

    per_operator = {}
    for op, g in ant.groupby("operator"):
        per_operator[op] = {
            "sites": int(g["site_key"].nunique()),
            "antennas": int(len(g)),
            "duplicate_antennas": int(g["is_duplicate"].sum()),
            "ambiguous_sites": int(g.loc[g["site_ambiguous"], "site_key"].nunique()),
            "antennas_in_rf": int(g["rf_include"].sum()),
            "antennas_tx_power_defaulted": int(g["tx_power_filled"].sum()),
        }
    report["per_operator"] = per_operator
    return ant, report
