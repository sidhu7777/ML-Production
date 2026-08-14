from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.wkt import loads as load_wkt

from tests.baseline.lte_rf_debug_lab import DEFAULT_PROJECT_ID, DEFAULT_REGION, _write_json


OUTPUT_ROOT = Path("tests/output")
DEFAULT_POLYGON_WKT = (
    "POLYGON(("
    "77.3493010211505 28.6451999446618,"
    "77.3760801959551 28.6563475183659,"
    "77.3798996615923 28.6493804236681,"
    "77.3790413547076 28.6309248012639,"
    "77.3383146930255 28.6320924980605,"
    "77.3493010211505 28.6451999446618"
    "))"
)
DEFAULT_BUCKET_LABEL = "PART_1"
DEFAULT_BUCKET_START = "2025-08-06 00:00:00"
DEFAULT_BUCKET_END = "2025-11-07 23:59:59"
DEFAULT_LIMIT = 1000
DEFAULT_TILE_SIDE_M = 1500.0


@dataclass
class OpenCellIdProbeConfig:
    project_id: int = DEFAULT_PROJECT_ID
    region: str = DEFAULT_REGION
    polygon_wkt: str = DEFAULT_POLYGON_WKT
    bucket_label: str = DEFAULT_BUCKET_LABEL
    bucket_start: str = DEFAULT_BUCKET_START
    bucket_end: str = DEFAULT_BUCKET_END
    api_key: str = ""
    radio: str = ""
    mcc: Optional[int] = None
    mnc: Optional[int] = None
    limit: int = DEFAULT_LIMIT
    max_pages: int = 20
    tile_side_m: float = DEFAULT_TILE_SIDE_M
    output_root: Path = OUTPUT_ROOT


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _polygon_bounds(polygon_wkt: str) -> Tuple[float, float, float, float]:
    polygon = load_wkt(polygon_wkt)
    minx, miny, maxx, maxy = polygon.bounds
    return float(miny), float(minx), float(maxy), float(maxx)


def _build_bbox_tiles(polygon_wkt: str, tile_side_m: float) -> List[Tuple[float, float, float, float]]:
    lat_min, lon_min, lat_max, lon_max = _polygon_bounds(polygon_wkt)
    center_lat = (lat_min + lat_max) / 2.0
    lat_step = float(tile_side_m) / 111320.0
    lon_step = float(tile_side_m) / max(111320.0 * abs(__import__("math").cos(__import__("math").radians(center_lat))), 1e-6)
    tiles: List[Tuple[float, float, float, float]] = []
    lat_cursor = lat_min
    while lat_cursor < lat_max:
        next_lat = min(lat_cursor + lat_step, lat_max)
        lon_cursor = lon_min
        while lon_cursor < lon_max:
            next_lon = min(lon_cursor + lon_step, lon_max)
            tiles.append((lat_cursor, lon_cursor, next_lat, next_lon))
            lon_cursor = next_lon
        lat_cursor = next_lat
    return tiles


def _fetch_page(
    config: OpenCellIdProbeConfig,
    bbox: Tuple[float, float, float, float],
    offset: int,
) -> Dict:
    lat_min, lon_min, lat_max, lon_max = bbox
    params: Dict[str, object] = {
        "key": config.api_key,
        "BBOX": f"{lat_min},{lon_min},{lat_max},{lon_max}",
        "limit": int(config.limit),
        "offset": int(offset),
        "format": "json",
    }
    if config.radio:
        params["radio"] = str(config.radio)
    if config.mcc is not None:
        params["mcc"] = int(config.mcc)
    if config.mnc is not None:
        params["mnc"] = int(config.mnc)
    url = "https://www.opencellid.org/cell/getInArea?" + urlencode(params)
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.opencellid.org/",
        },
    )
    try:
        with urlopen(req, timeout=60) as resp:
            payload = resp.read().decode("utf-8")
        return json.loads(payload)
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(
            f"OpenCellID HTTP {exc.code} {exc.reason}. "
            f"URL={url} BODY={body[:500]}"
        ) from exc


def _extract_rows(payload: Dict) -> pd.DataFrame:
    if isinstance(payload, dict):
        if isinstance(payload.get("cells"), list):
            return pd.DataFrame(payload["cells"])
        if isinstance(payload.get("cell"), list):
            return pd.DataFrame(payload["cell"])
        if isinstance(payload.get("response"), dict):
            nested = payload["response"]
            if isinstance(nested.get("cells"), list):
                return pd.DataFrame(nested["cells"])
    return pd.DataFrame()


def _clip_to_polygon(df: pd.DataFrame, polygon_wkt: str) -> pd.DataFrame:
    if df.empty or not {"lat", "lon"}.issubset(df.columns):
        return pd.DataFrame()
    work = df.copy()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work = work.dropna(subset=["lat", "lon"]).copy()
    if work.empty:
        return work
    polygon = load_wkt(polygon_wkt)
    gdf = gpd.GeoDataFrame(work, geometry=gpd.points_from_xy(work["lon"], work["lat"]), crs="EPSG:4326")
    gdf = gdf[gdf.geometry.within(polygon)].copy()
    return pd.DataFrame(gdf.drop(columns="geometry"))


def run_opencellid_probe(config: OpenCellIdProbeConfig) -> Path:
    if not config.api_key:
        raise ValueError("OpenCellID API key is required. Pass --api-key or set OPENCELLID_API_KEY.")

    run_dir = _ensure_dir(config.output_root / f"project_{config.project_id}" / f"opencellid_probe_{_timestamp()}")
    raw_dir = _ensure_dir(run_dir / "raw_pages")

    all_rows: List[pd.DataFrame] = []
    page_stats: List[Dict[str, object]] = []
    tiles = _build_bbox_tiles(config.polygon_wkt, config.tile_side_m)
    for tile_idx, bbox in enumerate(tiles, start=1):
        for page_idx in range(config.max_pages):
            offset = page_idx * int(config.limit)
            payload = _fetch_page(config, bbox, offset)
            _write_json(
                raw_dir / f"tile_{tile_idx:03d}_page_{page_idx + 1:03d}.json",
                payload if isinstance(payload, dict) else {"payload": payload},
            )
            if isinstance(payload, dict) and payload.get("code") == 3:
                page_stats.append(
                    {
                        "tile": tile_idx,
                        "page": page_idx + 1,
                        "offset": offset,
                        "api_rows": 0,
                        "polygon_rows": 0,
                        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                        "error": payload.get("error"),
                    }
                )
                print(f"[OPENCELLID_PROBE] tile={tile_idx} page={page_idx + 1} error={payload.get('error')}")
                break
            page_df = _extract_rows(payload)
            clipped_df = _clip_to_polygon(page_df, config.polygon_wkt)
            all_rows.append(clipped_df)
            page_stats.append(
                {
                    "tile": tile_idx,
                    "page": page_idx + 1,
                    "offset": offset,
                    "api_rows": int(len(page_df)),
                    "polygon_rows": int(len(clipped_df)),
                    "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    "error": "",
                }
            )
            print(
                f"[OPENCELLID_PROBE] tile={tile_idx} page={page_idx + 1} offset={offset} "
                f"api_rows={len(page_df)} polygon_rows={len(clipped_df)}"
            )
            if len(page_df) < int(config.limit):
                break

    result_df = pd.concat([df for df in all_rows if not df.empty], ignore_index=True) if any(not df.empty for df in all_rows) else pd.DataFrame()
    if not result_df.empty:
        result_df = result_df.drop_duplicates().reset_index(drop=True)
        result_df.to_csv(run_dir / "opencellid_polygon_rows.csv", index=False)

    pd.DataFrame(page_stats).to_csv(run_dir / "page_stats.csv", index=False)
    summary = {
        "run_type": "opencellid_probe_test",
        "project_id": int(config.project_id),
        "region": config.region,
        "bucket_label": config.bucket_label,
        "bucket_start": config.bucket_start,
        "bucket_end": config.bucket_end,
        "polygon_wkt": config.polygon_wkt,
        "radio": config.radio or None,
        "mcc": config.mcc,
        "mnc": config.mnc,
        "limit": int(config.limit),
        "max_pages": int(config.max_pages),
        "tile_side_m": float(config.tile_side_m),
        "tile_count": int(len(tiles)),
        "result_rows": int(len(result_df)),
        "artifacts": {
            "page_stats_csv": "page_stats.csv",
            "raw_pages_dir": "raw_pages",
            "result_csv": "opencellid_polygon_rows.csv" if not result_df.empty else None,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    return run_dir


def _parse_args() -> OpenCellIdProbeConfig:
    parser = argparse.ArgumentParser(description="Probe OpenCellID getInArea for polygon data.")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--polygon-wkt", type=str, default=DEFAULT_POLYGON_WKT)
    parser.add_argument("--bucket-label", type=str, default=DEFAULT_BUCKET_LABEL)
    parser.add_argument("--bucket-start", type=str, default=DEFAULT_BUCKET_START)
    parser.add_argument("--bucket-end", type=str, default=DEFAULT_BUCKET_END)
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENCELLID_API_KEY", ""))
    parser.add_argument("--radio", type=str, default="")
    parser.add_argument("--mcc", type=int, default=None)
    parser.add_argument("--mnc", type=int, default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--tile-side-m", type=float, default=DEFAULT_TILE_SIDE_M)
    args = parser.parse_args()
    return OpenCellIdProbeConfig(
        project_id=int(args.project_id),
        region=str(args.region),
        polygon_wkt=str(args.polygon_wkt),
        bucket_label=str(args.bucket_label),
        bucket_start=str(args.bucket_start),
        bucket_end=str(args.bucket_end),
        api_key=str(args.api_key).strip(),
        radio=str(args.radio).strip(),
        mcc=args.mcc,
        mnc=args.mnc,
        limit=max(1, int(args.limit)),
        max_pages=max(1, int(args.max_pages)),
        tile_side_m=max(200.0, float(args.tile_side_m)),
        output_root=OUTPUT_ROOT,
    )


def main() -> None:
    config = _parse_args()
    run_dir = run_opencellid_probe(config)
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
