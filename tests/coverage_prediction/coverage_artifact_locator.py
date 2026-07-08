from __future__ import annotations

import os
from pathlib import Path


ML_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ML_ROOT / "data"
OUTPUT_ROOT = ML_ROOT / "tests" / "output"
LATEST_COVERAGE_POINTER = DATA_DIR / "latest_coverage_artifact.txt"
DEFAULT_COVERAGE_ARCHIVE = DATA_DIR / "coverage_20260521_104406.7z"


def write_latest_coverage_artifact(path: Path) -> Path:
    resolved = Path(path).resolve()
    LATEST_COVERAGE_POINTER.parent.mkdir(parents=True, exist_ok=True)
    LATEST_COVERAGE_POINTER.write_text(str(resolved), encoding="utf-8")
    return resolved


def _latest_data_archive() -> Path | None:
    archives = sorted(
        DATA_DIR.glob("coverage_*.7z"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return archives[0] if archives else None


def _latest_output_run_dir() -> Path | None:
    run_dirs = sorted(
        OUTPUT_ROOT.glob("project_*/coverage_*"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def resolve_coverage_artifact_path() -> Path:
    explicit = os.environ.get("COVERAGE_ARTIFACT_PATH") or os.environ.get("COVERAGE_ARCHIVE_PATH")
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return candidate.resolve()

    if LATEST_COVERAGE_POINTER.exists():
        candidate = Path(LATEST_COVERAGE_POINTER.read_text(encoding="utf-8").strip()).expanduser()
        if candidate.exists():
            return candidate.resolve()

    latest_archive = _latest_data_archive()
    if latest_archive is not None:
        return latest_archive.resolve()

    latest_run_dir = _latest_output_run_dir()
    if latest_run_dir is not None:
        return latest_run_dir.resolve()

    return DEFAULT_COVERAGE_ARCHIVE.resolve()
