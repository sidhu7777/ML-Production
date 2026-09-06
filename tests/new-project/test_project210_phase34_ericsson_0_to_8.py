"""Phase 34: isolated Ericsson AIR6468B42 fallback experiment for Project 210 5G.

This experiment reuses the frozen Phase 33 3GPP 38.901 pipeline and changes
only antenna-pattern selection:

* eTilt 0..8 degrees: Ericsson AIR6468B42 3400 MHz dlMacro MSI pattern.
* eTilt 9 degrees: the Phase 33 standard 3GPP generic antenna fallback.

It deliberately does not assert that AIR6468B42 is installed.  This is a
bounded evidence-gathering experiment requested for the available patterns.
No Phase 33 or prior phase files are changed.
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rarfile

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
for path in (ML_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import test_project210_phase33_5g_38_901 as phase33

OUT_DIR = phase33.PROJECT_DIR / "cost231_phase34_ericsson_0_to_8"
ARCHIVE_PATH = ML_ROOT / "Research" / "5G Antennas.rar"
ERICSSON_ZIP_NAME = "AIR6468B42_Rev709.zip"


@lru_cache(maxsize=16)
def _ericsson_pattern(tilt_deg: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Load the 3400 MHz Ericsson macro pattern directly from the source archive."""
    with rarfile.RarFile(ARCHIVE_PATH) as archive:
        nested_zip = zipfile.ZipFile(io.BytesIO(archive.read(ERICSSON_ZIP_NAME)))
        suffix = f"dlMacro_Downtilt{int(tilt_deg) * 10}_3400_PWR.msi"
        matches = [name for name in nested_zip.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one AIR6468B42 pattern for {tilt_deg} deg, found {matches}")
        text = nested_zip.read(matches[0]).decode("utf-8", errors="ignore")
    gain_match = re.search(r"^GAIN\s+([0-9.]+)dBi", text, flags=re.MULTILINE)
    if not gain_match:
        raise RuntimeError(f"Missing boresight gain in Ericsson pattern {tilt_deg} deg")

    def values(section: str) -> np.ndarray:
        part = text.split(section, 1)[1]
        if section == "HORIZONTAL 360":
            part = part.split("VERTICAL 360", 1)[0]
        rows = re.findall(r"^\s*([0-9.]+)\s+([0-9.+-]+)\s*$", part, flags=re.MULTILINE)
        result = np.full(360, np.nan, dtype=float)
        for angle, attenuation in rows:
            index = int(round(float(angle))) % 360
            result[index] = float(attenuation)
        if np.isnan(result).any():
            raise RuntimeError(f"Incomplete {section} pattern for Ericsson tilt {tilt_deg}")
        return result

    return float(gain_match.group(1)), values("HORIZONTAL 360"), values("VERTICAL 360")


def _ericsson_or_phase33_fallback(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Use Ericsson 0..8 degree patterns; retain Phase 33 generic fallback at 9 degrees."""
    generic = phase33._generic_gain(df)
    tilt = pd.to_numeric(df["Etilt"], errors="coerce").fillna(30.0).to_numpy(float) / 10.0
    mechanical = pd.to_numeric(df["Mtilt"], errors="coerce").fillna(0.0).to_numpy(float) / 10.0
    rounded = np.round(tilt).astype(int)
    azimuth = np.abs(pd.to_numeric(df["azimuth_delta_deg"], errors="coerce").fillna(0.0).to_numpy(float))
    distance = np.maximum(pd.to_numeric(df["distance_m"], errors="coerce").to_numpy(float), 1.0)
    height = pd.to_numeric(df["Height"], errors="coerce").fillna(25.0).to_numpy(float)
    depression = -np.degrees(np.arctan2(phase33.UE_HEIGHT_M - height, distance)) + mechanical

    gain = generic.copy()
    source = np.full(len(df), "generic_fallback_9deg", dtype=object)
    supported = (rounded >= 0) & (rounded <= 8)
    for tilt_value in np.unique(rounded[supported]):
        selected = np.where(supported & (rounded == tilt_value))[0]
        boresight, horizontal, vertical = _ericsson_pattern(int(tilt_value))
        h_loss = horizontal[np.round(azimuth[selected]).astype(int) % 360]
        v_loss = vertical[np.round(depression[selected]).astype(int) % 360]
        gain[selected] = boresight - h_loss - v_loss
        source[selected] = f"Ericsson AIR6468B42 3400 MHz eTilt {tilt_value}"
    return gain, source


def main() -> None:
    # Phase 33's main is deliberately reused for identical path loss, terrain,
    # O2I, residual split, metrics, and CDF.  Only its antenna selector/output
    # folder are temporarily replaced in this process.
    original_selector = phase33._kathrein_or_generic_gain
    original_out_dir = phase33.OUT_DIR
    try:
        phase33._kathrein_or_generic_gain = _ericsson_or_phase33_fallback
        phase33.OUT_DIR = OUT_DIR
        phase33.main()
    finally:
        phase33._kathrein_or_generic_gain = original_selector
        phase33.OUT_DIR = original_out_dir

    # Preserve Phase 33 files untouched while giving this run unambiguous
    # Phase 34 artifact names and an explicit experiment declaration.
    for path in list(OUT_DIR.glob("phase33_*")):
        target = path.with_name(path.name.replace("phase33_", "phase34_", 1))
        path.replace(target)

    # Phase 33's engine retains its internal column prefix. Rename it only in
    # Phase 34 artifacts so neither phase's data is ambiguous in a dashboard.
    for stem in ("phase34_5g_scored_candidates_project210", "phase34_5g_dt_scored_project210", "phase34_5g_serving_grid_project210"):
        parquet = OUT_DIR / f"{stem}.parquet"
        csv = OUT_DIR / f"{stem}.csv"
        frame = pd.read_parquet(parquet)
        frame = frame.rename(columns={column: column.replace("phase33_", "phase34_", 1) for column in frame.columns if column.startswith("phase33_")})
        frame.to_parquet(parquet, index=False)
        frame.to_csv(csv, index=False)

    summary_path = OUT_DIR / "phase34_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["scope"] = "5G-only Phase 34 experiment. Phase 33 3GPP 38.901 path-loss, terrain, O2I, residual and split are unchanged. Ericsson AIR6468B42 3400 MHz macro patterns are used for eTilt 0..8; 9 degrees keeps Phase 33 generic fallback."
    summary["antenna"]["experiment_status"] = "comparison only; not evidence that AIR6468B42 is installed"
    summary["antenna"]["pattern_selection"] = "0..8 -> Ericsson AIR6468B42 3400 MHz; 9 -> generic fallback"
    metrics = summary["5g_held_out_outdoor_dt"]
    metrics["phase34_3gpp_38_901"] = metrics.pop("phase33_3gpp_38_901")
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    dt = pd.read_parquet(OUT_DIR / "phase34_5g_dt_scored_project210.parquet")
    dt = dt[(dt["split"].astype(str) == "validation") & (~dt["obstruction_branch"].astype(str).eq("indoor"))]
    p31 = pd.read_parquet(phase33.PHASE31_DIR / "phase31_dt_scored_5g_project210.parquet")
    p31 = p31[(p31["split"].astype(str) == "validation") & (~p31["obstruction_branch"].astype(str).eq("indoor"))]
    fig, ax = phase33.plt.subplots(figsize=(11, 7))
    phase33._cdf(ax, "5G DT measured", dt["rsrp_measured"], "#111827")
    phase33._cdf(ax, "Phase 31 5G (COST-231 approximation)", p31["phase31_rsrp"], "#d97706")
    phase33._cdf(ax, "Phase 34 5G (3GPP 38.901 + Ericsson experiment)", dt["phase34_final_rsrp"], "#2563eb")
    ax.set(title="Project 210 5G: Phase 34 held-out outdoor DT comparison", xlabel="RSRP (dBm)", ylabel="Cumulative %")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase34_5g_heldout_cdf.png", dpi=170)
    phase33.plt.close(fig)


if __name__ == "__main__":
    main()
