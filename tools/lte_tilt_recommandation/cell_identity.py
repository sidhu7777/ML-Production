from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


_EMPTY_VALUES = {"", "none", "nan", "null", "<na>"}


def clean_cell_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in _EMPTY_VALUES:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text.strip()


def canonical_cell_id(value: object) -> str:
    """Return the canonical client/reporting cell id: nodeb_sector."""
    text = clean_cell_token(value).replace("p0", "")
    if not text:
        return ""
    text = text.replace("|", "_")
    text = re.sub(r"\.0(?=_)|(?<=_)\.0", "", text)
    parts = [clean_cell_token(part).replace("p0", "") for part in text.split("_") if clean_cell_token(part)]
    if len(parts) >= 3:
        # Examples:
        # 2_618599_2 -> 618599_2
        # 618599_618599_2 -> 618599_2
        return f"{parts[-2]}_{parts[-1]}"
    if len(parts) == 2:
        return f"{parts[0]}_{parts[1]}"
    return text


def canonical_pair(nodeb: object, cell: object) -> str:
    nodeb_text = clean_cell_token(nodeb)
    cell_text = clean_cell_token(cell)
    if not cell_text:
        return ""
    if "_" in cell_text or "|" in cell_text:
        return canonical_cell_id(cell_text)
    if not nodeb_text:
        return canonical_cell_id(cell_text)
    return canonical_cell_id(f"{nodeb_text}_{cell_text}")


def _join_identity_parts(*parts: object) -> str:
    cleaned = [clean_cell_token(part).replace("|", "_").replace("p0", "") for part in parts]
    if any(not part for part in cleaned):
        return ""
    return "_".join(cleaned)


def build_rf_identity(site: object = None, cell_id: object = None, sector: object = None, band: object = None, fallback: object = None) -> str:
    """Return the carrier-level RF identity: site_cell_sector_band, with legacy fallback."""
    identity = _join_identity_parts(site, cell_id, sector, band)
    if identity:
        return identity
    return canonical_cell_id(fallback)


def build_sector_identity(site: object = None, cell_id: object = None, sector: object = None, fallback: object = None) -> str:
    """Return the sector-level identity: site_cell_sector, with legacy fallback."""
    identity = _join_identity_parts(site, cell_id, sector)
    if identity:
        return identity
    return canonical_cell_id(fallback)


def build_site_sector_band_identity(site: object = None, sector: object = None, band: object = None) -> str:
    """Return a stable match key for optimized rows where cell_id changed."""
    return _join_identity_parts(site, sector, band)


def canonical_cell_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="string")
    return series.map(canonical_cell_id)


def canonical_pair_series(nodeb_series: pd.Series, cell_series: pd.Series) -> pd.Series:
    return pd.Series(
        (canonical_pair(nodeb, cell) for nodeb, cell in zip(nodeb_series, cell_series)),
        index=cell_series.index,
        dtype="object",
    )


def first_non_empty_series(candidates: Iterable[pd.Series], index: pd.Index) -> pd.Series:
    out = pd.Series("", index=index, dtype="object")
    for series in candidates:
        if series is None:
            continue
        cleaned = canonical_cell_series(series)
        out = out.where(out.astype(str).str.strip().ne(""), cleaned)
    return out
