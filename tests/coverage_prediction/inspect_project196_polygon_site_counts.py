from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from shapely import wkt
from shapely.geometry import Point
from sqlalchemy import create_engine, text


ML_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ROOT))
load_dotenv(ML_ROOT / ".env")


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    text_value = str(value).strip()
    if text_value.lower() in {"", "none", "nan", "<na>", "null"}:
        return ""
    if text_value.endswith(".0"):
        text_value = text_value[:-2]
    return text_value


def canonical_cell(site: object, cell: object) -> str:
    site_text = clean(site)
    parts = [part for part in clean(cell).split("_") if part]
    if site_text and parts and parts[0] == site_text:
        parts = parts[1:]
    if len(parts) >= 2 and site_text and parts[0] == site_text:
        parts = parts[1:]
    if len(parts) >= 3:
        parts = parts[:2]
    return "_".join(parts)


def sector_value(site: object, cell: object, sector: object, sec_id: object) -> str:
    explicit = clean(sector) or clean(sec_id)
    if explicit:
        return explicit
    parts = [part for part in canonical_cell(site, cell).split("_") if part]
    return parts[-1] if parts else ""


def n(df: pd.DataFrame, cols: list[str]) -> int:
    return df[cols].fillna("").astype(str).agg("|".join, axis=1).nunique()


def main() -> None:
    engine = create_engine(os.environ.get("DATABASE_URL") or os.environ.get("DB_URL"))
    site = pd.read_sql(
        text(
            "SELECT id, site, nodeb_id, cell_id, sector, sec_id, band, frequency, earfcn, latitude, longitude "
            "FROM site_prediction WHERE tbl_project_id = :project_id"
        ),
        engine,
        params={"project_id": 196},
    )
    regions = pd.read_sql(
        text("SELECT ST_AsText(region) AS region_wkt FROM map_regions WHERE tbl_project_id = :project_id AND status = 1"),
        engine,
        params={"project_id": 196},
    )
    polygon = wkt.loads(str(regions.iloc[0]["region_wkt"]))
    lat = pd.to_numeric(site["latitude"], errors="coerce")
    lon = pd.to_numeric(site["longitude"], errors="coerce")
    site["inside_polygon"] = [
        polygon.contains(Point(y, x)) or polygon.touches(Point(y, x))
        for x, y in zip(lon, lat)
    ]
    inside = site.loc[site["inside_polygon"]].copy()
    inside["site_key"] = inside["site"].map(clean).where(inside["site"].map(clean).ne(""), inside["nodeb_id"].map(clean))
    inside["sector_key"] = [sector_value(s, c, sec, sec_id) for s, c, sec, sec_id in zip(inside["site"], inside["cell_id"], inside["sector"], inside["sec_id"])]
    inside["cell_key_raw"] = inside["cell_id"].map(clean)
    inside["cell_key_canonical"] = [canonical_cell(s, c) for s, c in zip(inside["site"], inside["cell_id"])]
    inside["band_key"] = inside["band"].map(clean).where(inside["band"].map(clean).ne(""), inside["frequency"].map(clean))

    valid = inside.loc[
        inside["site_key"].ne("")
        & inside["sector_key"].ne("")
        & inside["band_key"].ne("")
    ].copy()
    print("INSIDE_RAW_ROWS", len(inside))
    print("VALID_ROWS", len(valid))
    print("SITES", valid["site_key"].nunique())
    print("SITE_CELL_RAW", n(valid, ["site_key", "cell_key_raw"]))
    print("SITE_CELL_CANONICAL", n(valid, ["site_key", "cell_key_canonical"]))
    print("SITE_SECTOR", n(valid, ["site_key", "sector_key"]))
    print("SITE_CELL_SECTOR_CANONICAL", n(valid, ["site_key", "cell_key_canonical", "sector_key"]))
    print("SITE_CELL_SECTOR_BAND_CANONICAL", n(valid, ["site_key", "cell_key_canonical", "sector_key", "band_key"]))
    print("SITE_SECTOR_BAND", n(valid, ["site_key", "sector_key", "band_key"]))
    print("\nSITE_11625_SAMPLE")
    print(
        valid.loc[valid["site_key"].eq("11625"), ["id", "site", "cell_id", "sector", "sec_id", "band", "cell_key_canonical", "sector_key", "band_key"]]
        .sort_values(["cell_key_canonical", "sector_key", "band_key", "id"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
