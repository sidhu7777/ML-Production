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
    if "|" in text:
        text = text.split("|")[-1].strip()
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
