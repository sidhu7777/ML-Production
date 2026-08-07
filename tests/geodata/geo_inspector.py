"""
Core (non-UI) logic for inspecting client "Maps Data" zip bundles before they
are used for saved-polygon / OSM geo-enrichment. Handles:

  - Listing a zip's contents without extracting it.
  - Cheaply detecting a truncated/corrupt nested .7z archive by comparing the
    7z header's declared size against the actual bytes present (reads only
    the first 32 bytes via a streaming read - no disk extraction needed).
  - Staging a nested archive to disk (only when the user asks for it) and
    listing what's inside it.
  - Extracting a small "snapshot" of specific files for preview.
  - Loading a geo file (shapefile/geojson/gpkg/kml) with geopandas for a
    quick preview: CRS, bounds, feature count, columns, sample rows, and a
    lightweight (simplified + capped) geometry set for map display.
"""
from __future__ import annotations

import shutil
import struct
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

SEVEN_ZIP_SIG = b"7z\xbc\xaf'\x1c"
ARCHIVE_EXTS = {".7z", ".zip", ".rar"}
GEO_FILE_EXTS = {".shp", ".geojson", ".json", ".gpkg", ".kml", ".gml"}
SHAPEFILE_SIDECAR_EXTS = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix"}


def human_size(n: int) -> str:
    size = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def find_7z_exe() -> Optional[str]:
    on_path = shutil.which("7z") or shutil.which("7za")
    if on_path:
        return on_path
    for candidate in (
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ):
        if Path(candidate).exists():
            return candidate
    return None


@dataclass
class ZipEntry:
    name: str
    size: int
    compressed_size: int
    compress_type: int
    is_dir: bool
    ext: str


def scan_zip(zip_path: str) -> list[ZipEntry]:
    entries = []
    with zipfile.ZipFile(zip_path) as z:
        for i in z.infolist():
            p = Path(i.filename)
            entries.append(
                ZipEntry(
                    name=i.filename,
                    size=i.file_size,
                    compressed_size=i.compress_size,
                    compress_type=i.compress_type,
                    is_dir=i.is_dir(),
                    ext=p.suffix.lower(),
                )
            )
    return entries


def entries_to_dataframe(entries: list[ZipEntry]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": e.name,
                "size": human_size(e.size),
                "size_bytes": e.size,
                "compressed": human_size(e.compressed_size),
                "type": "folder" if e.is_dir else (e.ext or "(no ext)"),
            }
            for e in entries
        ]
    )


@dataclass
class SevenZipIntegrity:
    signature_ok: bool
    actual_size: int
    expected_min_size: Optional[int]
    missing_bytes: Optional[int]
    missing_pct: Optional[float]
    is_complete: Optional[bool]
    note: str = ""


def precheck_7z_integrity_in_zip(zip_path: str, entry_name: str) -> SevenZipIntegrity:
    """Detect truncated/corrupt 7z archives stored inside a zip WITHOUT extracting.

    Reads only the first 32 bytes of the nested .7z (via a streaming read of
    the zip entry) and parses the 7z "start header", which declares the byte
    offset + size of the archive's metadata (the "next header"). Comparing
    that declared end-of-archive offset against the entry's actual
    (uncompressed) size tells us if the file was cut off mid-transfer.
    """
    with zipfile.ZipFile(zip_path) as z:
        info = z.getinfo(entry_name)
        with z.open(info) as f:
            head = f.read(32)

    actual_size = info.file_size
    if len(head) < 32 or head[:6] != SEVEN_ZIP_SIG:
        return SevenZipIntegrity(
            signature_ok=False,
            actual_size=actual_size,
            expected_min_size=None,
            missing_bytes=None,
            missing_pct=None,
            is_complete=None,
            note="Not a valid 7z signature (file may be a different format or fully corrupt).",
        )

    _sig, _ver_maj, _ver_min, _start_crc, next_offset, next_size, _next_crc = struct.unpack(
        "<6sBBIQQI", head
    )
    expected_min_size = 32 + next_offset + next_size
    missing = expected_min_size - actual_size
    is_complete = missing <= 0
    pct = (missing / expected_min_size * 100) if expected_min_size else None

    return SevenZipIntegrity(
        signature_ok=True,
        actual_size=actual_size,
        expected_min_size=expected_min_size,
        missing_bytes=max(missing, 0),
        missing_pct=pct,
        is_complete=is_complete,
        note="Archive header + data fully present." if is_complete else (
            f"Archive is truncated: only {actual_size/expected_min_size*100:.1f}% of the "
            f"expected bytes are present. The 7z metadata header (which lists every file "
            f"inside) lives at the END of the archive and is entirely missing, so the "
            f"contents cannot be listed or extracted at all. This is typically caused by "
            f"an interrupted upload/copy/download of the source file."
        ),
    )


def stage_entry(
    zip_path: str,
    entry_name: str,
    dest_dir: str,
    chunk_size: int = 4 * 1024 * 1024,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Stream-copy one zip entry to a local file (staging), reporting progress."""
    dest = Path(dest_dir) / Path(entry_name).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        info = z.getinfo(entry_name)
        total = info.file_size
        written = 0
        with z.open(info) as src, open(dest, "wb") as out:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if progress_cb:
                    progress_cb(written, total)
    return dest


def list_7z_contents(staged_7z_path: str) -> pd.DataFrame:
    """List the internal files of a (staged, on-disk) .7z archive without extracting them."""
    exe = find_7z_exe()
    if not exe:
        raise RuntimeError("7z.exe not found on PATH or in the usual Program Files locations.")
    result = subprocess.run(
        [exe, "l", "-slt", staged_7z_path],
        capture_output=True, text=True, check=True,
    )
    rows = []
    current: dict = {}
    for line in result.stdout.splitlines():
        line = line.rstrip()
        if not line:
            if current.get("Path"):
                rows.append(current)
            current = {}
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            current[key.strip()] = val.strip()
    if current.get("Path"):
        rows.append(current)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keep = [c for c in ["Path", "Size", "Packed Size", "Modified", "Attributes", "Folder"] if c in df.columns]
    df = df[keep]
    if "Size" in df.columns:
        df["Size"] = pd.to_numeric(df["Size"], errors="coerce").fillna(0).astype(int)
        df["size_human"] = df["Size"].map(human_size)
    return df


def extract_from_7z(staged_7z_path: str, targets: list[str], dest_dir: str) -> list[Path]:
    """Extract only specific files (a small 'snapshot') from a staged 7z archive."""
    exe = find_7z_exe()
    if not exe:
        raise RuntimeError("7z.exe not found on PATH or in the usual Program Files locations.")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "e", staged_7z_path, f"-o{dest}", "-y"] + list(targets)
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [dest / Path(t).name for t in targets]


@dataclass
class GeoPreview:
    driver: str
    crs: Optional[str]
    feature_count: int
    geometry_types: list[str]
    bounds: Optional[tuple[float, float, float, float]]
    columns: list[str]
    sample: pd.DataFrame
    sample_gdf: "object" = field(default=None, repr=False)


def load_geo_preview(path: str, max_features: int = 500):
    import geopandas as gpd

    gdf_full_len = None
    try:
        import fiona
        with fiona.open(path) as src:
            gdf_full_len = len(src)
            driver = src.driver
            crs = str(src.crs) if src.crs else None
    except Exception:
        driver = Path(path).suffix.lstrip(".").upper()
        crs = None

    gdf = gpd.read_file(path, rows=max_features)
    if crs is None and gdf.crs:
        crs = str(gdf.crs)

    bounds = tuple(gdf.total_bounds) if len(gdf) else None
    geom_types = sorted(gdf.geometry.geom_type.dropna().unique().tolist()) if len(gdf) else []
    non_geom_cols = [c for c in gdf.columns if c != "geometry"]

    return GeoPreview(
        driver=driver,
        crs=crs,
        feature_count=gdf_full_len if gdf_full_len is not None else len(gdf),
        geometry_types=geom_types,
        bounds=bounds,
        columns=non_geom_cols,
        sample=pd.DataFrame(gdf[non_geom_cols].head(50)),
        sample_gdf=gdf,
    )
