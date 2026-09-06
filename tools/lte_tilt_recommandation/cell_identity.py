from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import pandas as pd


_EMPTY_VALUES = {"", "none", "nan", "null", "<na>"}


# These are pure scalar functions called tens of millions of times per RF
# evaluation to normalise a few hundred distinct cell ids (profiled: 8.8M
# clean_cell_token + 29M str.strip calls for ~112 unique ids). Memoise the
# str case -- str hashes exactly, so a cached lookup is identical to
# recomputing. Non-str values fall through uncached, which avoids the
# int/float/bool hash-equality collisions (hash(1)==hash(True)) that a blanket
# cache would introduce.
def _clean_cell_token_impl(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in _EMPTY_VALUES:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text.strip()


_clean_cell_token_cached = lru_cache(maxsize=200_000)(_clean_cell_token_impl)


def clean_cell_token(value: object) -> str:
    if type(value) is str:
        return _clean_cell_token_cached(value)
    return _clean_cell_token_impl(value)


def _canonical_cell_id_impl(value: object) -> str:
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


_canonical_cell_id_cached = lru_cache(maxsize=200_000)(_canonical_cell_id_impl)


def canonical_cell_id(value: object) -> str:
    """Return the canonical client/reporting cell id: nodeb_sector."""
    if type(value) is str:
        return _canonical_cell_id_cached(value)
    return _canonical_cell_id_impl(value)


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
