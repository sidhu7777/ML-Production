"""Phase 35: a real measured 3.5 GHz antenna pattern for EVERY 5G electrical tilt.

Phase 33 leaves ~34 % of the 5G candidate rows (every Etilt 0 deg and 1 deg
sector - 15,293 rows) on the generic 3GPP 18/65/6 antenna, because the Kathrein
800109221 PAP set only covers 2..12 deg. Phase 35 removes that fallback: every
sector uses the real measured Kathrein 800109221 3300-3590 MHz envelope, with
Etilt 0/1 deg clamped to the measured 2 deg file (a ~6.5 deg vertical beam moves
< 1 dB at the horizon between 1 deg and 2 deg of electrical tilt). This is the
"one consistent real pattern across every tilt, no generic fallback" experiment.

Why NOT the CommScope S4-90M-R1-P8 that was proposed:
  * its published spec is a 90 deg HPBW, 15.4 dBi, 8-port beamforming array with a
    2..12 deg tilt range - a different antenna CLASS from these macro sectors
    (the Kathrein at ~65 deg / ~17.4 dBi is the right class),
  * it does not cover 0..1 deg either, so it would not remove the fallback,
  * CommScope publishes no machine-readable pattern file for it (portal login
    only; the public datasheet is image-only).
  Using it would be a worse proxy, not a better one.

Everything else is the frozen Phase 33 pipeline: 3GPP TR 38.901 UMa at 3300 MHz,
Etilt/10 + Mtilt/10, terrain, O2I, water override, the 70/30 DT split and the
per-clutter residual fit. Antenna sampling is single-bin like Phase 33/34 (no
smoothing / no delta clip) so the ONLY changed variable is real-pattern coverage.
No Phase 33 or earlier phase file is modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
for path in (ML_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import test_project210_phase33_5g_38_901 as phase33

phase29 = phase33.phase29
phase28 = phase33.phase28

OUT_DIR = phase33.PROJECT_DIR / "cost231_phase35_kathrein_all_tilts"
KATHREIN_FILE_MIN_DEG = 2
KATHREIN_FILE_MAX_DEG = 12


def _kathrein_all_tilts_gain(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Real Kathrein 800109221 gain for EVERY row. Rounded Etilt is clamped into
    the measured 2..12 deg file range; there is no generic 3GPP fallback."""
    generic = phase33._generic_gain(df)  # only if a .pap file is physically absent
    etilt = pd.to_numeric(df["Etilt"], errors="coerce").fillna(30.0).to_numpy(float) / 10.0
    mtilt = pd.to_numeric(df["Mtilt"], errors="coerce").fillna(0.0).to_numpy(float) / 10.0
    az_off = np.abs(pd.to_numeric(df["azimuth_delta_deg"], errors="coerce").fillna(0.0).to_numpy(float))
    distance = np.maximum(pd.to_numeric(df["distance_m"], errors="coerce").to_numpy(float), 1.0)
    htx = pd.to_numeric(df["Height"], errors="coerce").fillna(25.0).to_numpy(float)
    depression = -np.degrees(np.arctan2(phase33.UE_HEIGHT_M - htx, distance)) + mtilt

    rounded = np.round(etilt).astype(int)
    file_tilt = np.clip(rounded, KATHREIN_FILE_MIN_DEG, KATHREIN_FILE_MAX_DEG)
    gain = generic.copy()
    source = np.full(len(df), "Kathrein 800109221", dtype=object)
    source[rounded < KATHREIN_FILE_MIN_DEG] = "Kathrein 800109221 (Etilt 0/1 -> 2 deg file)"
    source[rounded > KATHREIN_FILE_MAX_DEG] = "Kathrein 800109221 (Etilt clamped to 12 deg file)"

    for tilt in np.unique(file_tilt):
        idx = np.where(file_tilt == tilt)[0]
        pattern = phase29.PAT_DIR / "K800109221" / f"3300 - 3590 MHz, eTilt {tilt}, Y1P45 - Port1.pap"
        if not pattern.is_file():
            source[idx] = "generic_fallback_pattern_missing"
            continue
        hs, h, vs, v = phase29._pap_cached(str(pattern))
        h_gain = phase29._pat_gain(hs, np.asarray(h), az_off[idx])
        v_gain = phase29._pat_gain(vs, np.asarray(v), depression[idx])
        gain[idx] = phase29.BORESIGHT_GAIN_DBI["K800109221_3300"] + h_gain + v_gain
    return gain, source


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse Phase 33's engine unchanged; swap only the antenna selector + out dir.
    original_selector = phase33._kathrein_or_generic_gain
    original_out_dir = phase33.OUT_DIR
    try:
        phase33._kathrein_or_generic_gain = _kathrein_all_tilts_gain
        phase33.OUT_DIR = OUT_DIR
        phase33.main()
    finally:
        phase33._kathrein_or_generic_gain = original_selector
        phase33.OUT_DIR = original_out_dir

    for path in list(OUT_DIR.glob("phase33_*")):
        path.replace(path.with_name(path.name.replace("phase33_", "phase35_", 1)))

    for stem in ("phase35_5g_scored_candidates_project210",
                 "phase35_5g_dt_scored_project210",
                 "phase35_5g_serving_grid_project210"):
        parquet = OUT_DIR / f"{stem}.parquet"
        frame = pd.read_parquet(parquet)
        frame = frame.rename(columns={c: c.replace("phase33_", "phase35_", 1)
                                      for c in frame.columns if c.startswith("phase33_")})
        frame.to_parquet(parquet, index=False)
        frame.to_csv(OUT_DIR / f"{stem}.csv", index=False)

    # ---- rewrite the summary as an explicit Phase 35 experiment ----
    summary_path = OUT_DIR / "phase35_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["scope"] = (
        "5G-only Phase 35 experiment. Phase 33 3GPP 38.901 UMa (3300 MHz) path loss, terrain, O2I, "
        "water, DT split and per-clutter residual are unchanged. The Kathrein 800109221 3300-3590 MHz "
        "measured pattern is used for EVERY tilt; Etilt 0/1 deg clamp to the 2 deg file; no generic fallback."
    )
    summary["antenna"] = {
        "pattern": "Kathrein 800109221, 3300-3590 MHz, Y1P45 Port1 (real measured PAP)",
        "coverage": "all electrical tilts; rounded Etilt clamped to the 2..12 deg measured file set",
        "generic_fallback": "none (Phase 33 fell back for the 15,293 Etilt 0/1 rows)",
        "sampling": "single 1 deg bin, no smoothing, no delta clip (identical to Phase 33/34)",
        "not_used": "CommScope S4-90M-R1-P8 - wrong antenna class (90 deg beamforming array), "
                    "2..12 deg tilt only, and no machine-readable pattern is published",
        "candidate_rows_by_source": summary.get("antenna", {}).get("candidate_rows_by_source", {}),
    }
    metrics = summary["5g_held_out_outdoor_dt"]
    metrics["phase35_kathrein_all_tilts"] = metrics.pop("phase33_3gpp_38_901")

    # ---- side-by-side ladder: 31 -> 33 -> 34 -> 35 ----
    def _heldout(path: Path, key: str) -> dict:
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
            return s["5g_held_out_outdoor_dt"][key]
        except Exception:
            return {}

    p33 = _heldout(phase33.OUT_DIR / "phase33_summary.json", "phase33_3gpp_38_901")
    p34 = _heldout(phase33.PROJECT_DIR / "cost231_phase34_ericsson_0_to_8" / "phase34_summary.json",
                   "phase34_3gpp_38_901")
    summary["ladder_5g_heldout_outdoor_dt_mae"] = {
        "phase31_cost231_real_antenna": round(metrics["phase31_cost231_approximation"]["mae"], 2),
        "phase33_38901_kathrein_2to9_generic_0to1": round(p33.get("mae", float("nan")), 2),
        "phase34_38901_ericsson_0to8_generic_9": round(p34.get("mae", float("nan")), 2),
        "phase35_38901_kathrein_all_tilts": round(metrics["phase35_kathrein_all_tilts"]["mae"], 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ---- held-out CDF: measured vs 31 / 33 / 35 ----
    dt = pd.read_parquet(OUT_DIR / "phase35_5g_dt_scored_project210.parquet")
    dt = dt[(dt["split"].astype(str) == "validation") & (~dt["obstruction_branch"].astype(str).eq("indoor"))]
    p31 = pd.read_parquet(phase33.PHASE31_DIR / "phase31_dt_scored_5g_project210.parquet")
    p31 = p31[(p31["split"].astype(str) == "validation") & (~p31["obstruction_branch"].astype(str).eq("indoor"))]
    p33dt = pd.read_parquet(phase33.OUT_DIR / "phase33_5g_dt_scored_project210.parquet")
    p33dt = p33dt[(p33dt["split"].astype(str) == "validation") & (~p33dt["obstruction_branch"].astype(str).eq("indoor"))]

    fig, ax = phase33.plt.subplots(figsize=(11, 7))
    phase33._cdf(ax, "5G DT measured", dt["rsrp_measured"], "#111827")
    phase33._cdf(ax, "Phase 31 (COST-231 real antenna)", p31["phase31_rsrp"], "#d97706")
    phase33._cdf(ax, "Phase 33 (38.901, Kathrein 2..9 + generic 0/1)", p33dt["phase33_final_rsrp"], "#9333ea")
    phase33._cdf(ax, "Phase 35 (38.901, Kathrein all tilts)", dt["phase35_final_rsrp"], "#2563eb")
    ax.set(title="Project 210 5G: Phase 35 held-out outdoor DT comparison",
           xlabel="RSRP (dBm)", ylabel="Cumulative %")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase35_5g_heldout_cdf.png", dpi=170)
    phase33.plt.close(fig)

    print(json.dumps(summary["ladder_5g_heldout_outdoor_dt_mae"], indent=2))
    print(f"[PHASE35] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
