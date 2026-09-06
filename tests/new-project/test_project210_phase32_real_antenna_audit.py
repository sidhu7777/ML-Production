"""Phase 32: read-only integrity audit for the Phase 31 real-antenna run.

This script never recalculates or overwrites Phases 27-31.  It checks whether
the selected vendor pattern is compatible with each candidate's carrier and
tilt configuration, and records how often Phase 31 altered the vendor pattern
with its clipping rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent / "data" / "project_210_taiwan"
PHASE31 = ROOT / "cost231_phase31_phase28_real_antenna"
OUT = ROOT / "cost231_phase32_real_antenna_audit"

PATTERN_RANGES_MHZ = {
    "4G_low": (698.0, 806.0),
    "4G_high": (1710.0, 1880.0),
    "5G": (3300.0, 3590.0),
}
DELTA_MIN_DB, DELTA_MAX_DB = -15.0, 12.0


def _audit_tech(tech: str) -> dict:
    path = PHASE31 / f"phase31_scored_candidates_{tech.lower()}_project210.parquet"
    frame = pd.read_parquet(path)
    freq = pd.to_numeric(frame["frequency_mhz"], errors="coerce")
    delta = pd.to_numeric(frame["antenna_gain_delta_db"], errors="coerce")
    etilt = pd.to_numeric(frame["Etilt"], errors="coerce") / 10.0

    if tech == "5G":
        low, high = PATTERN_RANGES_MHZ["5G"]
        selected_tilt = etilt.round().clip(2, 12)
    else:
        low_band = freq <= 1000.0
        low, high = PATTERN_RANGES_MHZ["4G_low"]
        in_low = low_band & freq.between(low, high)
        hlow, hhigh = PATTERN_RANGES_MHZ["4G_high"]
        in_high = (~low_band) & freq.between(hlow, hhigh)
        selected_tilt = etilt.round().clip(0, 10)
        return {
            "rows": int(len(frame)),
            "frequency_mhz": {str(k): int(v) for k, v in freq.value_counts().items()},
            "pattern_frequency_compatible_rows": int((in_low | in_high).sum()),
            "pattern_frequency_incompatible_rows": int((~(in_low | in_high)).sum()),
            "tilt_substituted_rows": int((selected_tilt != etilt.round()).sum()),
            "delta_clipped_low_rows": int((delta <= DELTA_MIN_DB).sum()),
            "delta_clipped_high_rows": int((delta >= DELTA_MAX_DB).sum()),
        }

    compatible = freq.between(low, high)
    return {
        "rows": int(len(frame)),
        "frequency_mhz": {str(k): int(v) for k, v in freq.value_counts().items()},
        "pattern_range_mhz": [low, high],
        "pattern_frequency_compatible_rows": int(compatible.sum()),
        "pattern_frequency_incompatible_rows": int((~compatible).sum()),
        "tilt_substituted_rows": int((selected_tilt != etilt.round()).sum()),
        "tilt_zero_or_one_forced_to_two_rows": int(etilt.isin([0.0, 1.0]).sum()),
        "delta_clipped_low_rows": int((delta <= DELTA_MIN_DB).sum()),
        "delta_clipped_high_rows": int((delta >= DELTA_MAX_DB).sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase31_summary = json.loads((PHASE31 / "phase31_summary.json").read_text(encoding="utf-8"))
    report = {
        "scope": "Read-only Phase 31 configuration audit. No prediction values were changed.",
        "phase31_pattern_selection": {
            "4G": "CCVVPX308 698-806 MHz or 1710-1880 MHz",
            "5G": "Kathrein 800109221 3300-3590 MHz",
        },
        "phase31_delta_clip_db": [DELTA_MIN_DB, DELTA_MAX_DB],
        "technology": {tech: _audit_tech(tech) for tech in ("4G", "5G")},
        "held_out_mae_from_phase31": {
            tech: phase31_summary["technology"][tech]["held_out_outdoor_dt"]
            for tech in ("4G", "5G")
        },
    }
    (OUT / "phase32_real_antenna_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
