"""Phase 36 v2 physical upgrades for the production offset pipeline.

Adds, on top of the existing Phase 26 physical (``physical_rsrp_unclipped``), the
three corrections proven in the Taiwan test phases:

1. RSRP-per-RE reference term.  ``compute_sector_rsrp`` returns total-carrier
   power; 3GPP RSRP is per resource element, so subtract ``10*log10(12*N_RB)``.
   4G uses the cell's real channel bandwidth (N_RB); 5G SS-RSRP is data-anchored
   by the caller on the clean drive-test residual (``g5_level_anchor_db``).

2. Real-antenna gain delta.  The raw uses a generic 3GPP 18/65/6 antenna;
   replace it with (real vendor pattern - generic) when an antenna pattern is
   available for the cell, else leave the generic (delta = 0).

3. Water override.  Open water gets terrain diffraction kept but the indoor /
   dominant-building O2I term removed.

Nothing here is country-specific.  Missing inputs degrade gracefully:
no bandwidth -> a per-band default N_RB; no antenna pattern -> generic (delta 0).
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Default vendor antenna patterns (same as Phase 36 v2). A client-supplied
# pattern for a given `antenna_model` overrides these; generic 3GPP is used only
# when no pattern file can be resolved.
PATTERN_DIR = Path(__file__).resolve().parent / "antenna_patterns"
_DEFAULT_BORESIGHT_DBI = {
    ("4G", "low"): 14.5,    # CCVVPX308 698-806 MHz
    ("4G", "high"): 16.6,   # CCVVPX308 1710-1880 MHz
    ("5G", "n78"): 17.4,    # K800109221 3300-3590 MHz
}

# LTE resource blocks by channel bandwidth (MHz).  NR n78 100 MHz / SCS 30 kHz = 273.
_RB_BY_BW_MHZ = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100, 100: 273}
_DEFAULT_N_RB = {"4G": 50, "5G": 273}
# The production RF core already contains the generic 3GPP gain. To apply the
# PAP absolutely, add the exact PAP-minus-generic difference with no artificial
# bound; this is algebraically identical to replacing generic gain by PAP gain.
ANTENNA_DELTA_CLIP_DB = None
ANTENNA_PATTERN_SUBSTITUTION_CLIP_DB = None


def n_rb_for(technology: str, bandwidth_mhz: float | None) -> int:
    if bandwidth_mhz is not None and np.isfinite(bandwidth_mhz):
        key = round(float(bandwidth_mhz), 1)
        if key in _RB_BY_BW_MHZ:
            return _RB_BY_BW_MHZ[key]
        # nearest known bandwidth
        nearest = min(_RB_BY_BW_MHZ, key=lambda b: abs(b - key))
        return _RB_BY_BW_MHZ[nearest]
    return _DEFAULT_N_RB.get(str(technology), 50)


def per_re_offset_db(technology: str, bandwidth_mhz: float | None) -> float:
    """dB to ADD to the raw to convert total-carrier power to per-RE RSRP (negative)."""
    return -(10.0 * math.log10(12 * n_rb_for(technology, bandwidth_mhz)))


def _bandwidth_series(frame: pd.DataFrame) -> pd.Series:
    for col in ("channel_bandwidth_mhz", "bandwidth_mhz", "channel_bw_mhz", "bandwidth", "bw_mhz"):
        if col in frame.columns:
            bw = pd.to_numeric(frame[col], errors="coerce")
            if bw.notna().any():
                return bw
    return pd.Series(np.nan, index=frame.index, dtype=float)


def apply_reference_and_water(
    frame: pd.DataFrame,
    physical_col: str = "physical_rsrp_unclipped",
    g5_level_anchor_db: float = 0.0,
    pattern_lookup=None,
) -> pd.DataFrame:
    """Return `frame` with `phase36_physical_rsrp` = physical + per-RE + 5G anchor
    + real-antenna gain delta.

    `pattern_lookup` defaults to the Phase 36 v2 vendor patterns (CCVVPX308 4G /
    K800109221 5G); generic 3GPP only where a pattern file is missing."""
    out = frame.copy()
    tech = out["technology"].astype(str)
    bw = _bandwidth_series(out)
    # Total carrier power -> per-resource-element level, for BOTH technologies.
    # RSRP and SS-RSRP are both defined per RE, so the conversion is not optional
    # for 5G. It was previously skipped on the grounds that `g5_level_anchor_db`
    # would close the 5G level instead, but that anchor is passed as 0.0 by the
    # production caller, so nothing closed it and 5G physical carried raw total
    # carrier power - roughly 35 dB hot at n78's 273 RB.
    per_re = np.array(
        [per_re_offset_db(t, b) for t, b in zip(tech.to_numpy(), bw)], dtype=float
    )
    anchor = np.where(tech.to_numpy() == "5G", float(g5_level_anchor_db), 0.0)

    # Phase 48 keeps the physical obstruction score as-is. Water is handled by
    # calibration/clutter grouping, not by a second post-physical addback.
    water_addback = np.zeros(len(out), dtype=float)

    antenna = antenna_gain_delta_details(out, pattern_lookup=pattern_lookup or default_pattern_lookup)

    # Emit evidence of the pattern source actually used for this calculation.
    # This distinguishes an applied vendor PAP from the intentional generic
    # 3GPP fallback without changing any RF value.
    pattern_log = pd.DataFrame({
        "technology": tech.to_numpy(),
        "source": antenna["phase36_antenna_source"].astype(str).to_numpy(),
        "file": antenna["phase36_pap_file"].astype(str).to_numpy(),
    })
    for (technology, source), group in pattern_log.groupby(["technology", "source"], dropna=False):
        files = sorted({value for value in group["file"] if value})
        detail = f" files={files}" if files else ""
        print(
            f"[LTE_OFFSET][PHASE36_ANTENNA_PATTERN] technology={technology} "
            f"source={source} rows={len(group)}{detail}",
            flush=True,
        )

    out["phase36_per_re_db"] = per_re
    out["phase36_5g_anchor_db"] = anchor
    out["phase36_water_o2i_addback_db"] = water_addback
    for col in antenna.columns:
        out[col] = antenna[col].to_numpy()
    out["phase36_physical_rsrp"] = (
        pd.to_numeric(out[physical_col], errors="coerce")
        + per_re
        + anchor
        + water_addback
        + pd.to_numeric(out["phase36_antenna_delta_db"], errors="coerce").fillna(0.0)
    )
    return out


# --------------------------------------------------------------------------- antenna
def _generic_3gpp_gain(az_off_deg: np.ndarray, elev_diff_deg: np.ndarray,
                       max_gain: float = 18.0, h_bw: float = 65.0, v_bw: float = 6.0,
                       a_max: float = 30.0, sla_v: float = 20.0) -> np.ndarray:
    """Must stay identical to Sector_wise_prediction_code_copy's
    compute_3gpp_antenna_gain_vectorized. `raw_cost231_rsrp` already contains
    that generic pattern, and the Phase 36 delta subtracts this one to replace
    it with the measured PAP pattern. Any difference between the two - sla_v
    was 30 here against 20 there - leaves a residue of the generic pattern in
    the result instead of cancelling it."""
    az_off = np.abs(np.asarray(az_off_deg, dtype=float))
    a_h = -np.minimum(12.0 * (az_off / h_bw) ** 2, a_max)
    a_v = -np.minimum(12.0 * (np.asarray(elev_diff_deg, dtype=float) / v_bw) ** 2, sla_v)
    return max_gain - np.minimum(-(a_h + a_v), a_max)


def antenna_gain_delta(
    frame: pd.DataFrame,
    pattern_lookup=None,
    ue_height_m: float = 1.5,
) -> np.ndarray:
    """(real vendor pattern gain) - (generic 3GPP gain), per row.

    `pattern_lookup(antenna_model, freq_mhz, etilt_deg) -> (h_start, h_arr, v_start, v_arr, boresight_dbi)`
    or None.  When it returns None (no model / not in library) the delta is 0 —
    the raw already contains the generic gain.
    """
    return antenna_gain_delta_details(frame, pattern_lookup=pattern_lookup, ue_height_m=ue_height_m)[
        "phase36_antenna_delta_db"
    ].to_numpy(float)


def antenna_gain_delta_details(
    frame: pd.DataFrame,
    pattern_lookup=None,
    ue_height_m: float = 1.5,
) -> pd.DataFrame:
    """PAP-vs-generic antenna gain terms used by Phase36.

    The returned delta is the only value added to RSRP. The other columns are
    kept with the prediction rows so production/debug output can prove the
    horizontal PAP gain, vertical PAP gain, and boresight value that produced it.
    """
    n = len(frame)
    if pattern_lookup is None:
        pattern_lookup = default_pattern_lookup

    dist = np.maximum(pd.to_numeric(frame.get("distance_m"), errors="coerce").fillna(1.0).to_numpy(float), 1.0)
    htx = pd.to_numeric(frame.get("Height", frame.get("antenna_height")), errors="coerce").fillna(25.0).to_numpy(float)
    etilt = pd.to_numeric(frame.get("Etilt", frame.get("electrical_tilt")), errors="coerce").fillna(3.0).to_numpy(float)
    mtilt = pd.to_numeric(frame.get("Mtilt", frame.get("mechanical_tilt")), errors="coerce").fillna(0.0).to_numpy(float)
    az_off = np.abs(pd.to_numeric(frame.get("azimuth_delta_deg"), errors="coerce").fillna(0.0).to_numpy(float))
    freq = pd.to_numeric(frame.get("serving_frequency_mhz", frame.get("frequency_mhz")), errors="coerce").fillna(1800.0).to_numpy(float)
    model = (frame["antenna_model"].astype(str).to_numpy() if "antenna_model" in frame.columns
             else np.full(n, "", dtype=object))
    tech = (frame["technology"].astype(str).to_numpy() if "technology" in frame.columns
            else np.full(n, "4G", dtype=object))

    elev = np.degrees(np.arctan2(ue_height_m - htx, dist))
    generic = _generic_3gpp_gain(az_off, elev + etilt + mtilt)
    depression = -elev + mtilt

    et_round = np.round(etilt).astype(int)
    real = np.full(n, np.nan, dtype=float)
    horizontal = np.full(n, np.nan, dtype=float)
    vertical = np.full(n, np.nan, dtype=float)
    boresight = np.full(n, np.nan, dtype=float)
    pap_model = np.full(n, "", dtype=object)
    pap_file = np.full(n, "", dtype=object)
    source = np.full(n, "generic_3gpp", dtype=object)

    freq_key = np.round(freq, 1)
    keys = pd.DataFrame({"m": model, "t": tech, "et": et_round, "f": freq_key})
    for (m, t, et, f), grp in keys.groupby(["m", "t", "et", "f"], sort=False):
        sel = grp.index.to_numpy()
        lookup_freq = float(f)
        pat = pattern_lookup(m, lookup_freq, int(et), t)
        if pat is None:
            continue
        hs, h, vs, v, g0 = pat
        h_gain = _pat_gain_smoothed(hs, np.asarray(h, float), az_off[sel])
        v_gain = _pat_gain_smoothed(vs, np.asarray(v, float), depression[sel])
        horizontal[sel] = h_gain
        vertical[sel] = v_gain
        boresight[sel] = g0
        real[sel] = g0 + h_gain + v_gain
        source[sel] = "pap"
        resolved_tech = "5G" if (str(t) == "5G" or str(m).upper().startswith("K800")) else "4G"
        default_path, _ = _default_pap_path(resolved_tech, lookup_freq, int(et))
        pap_model[sel] = str(m).strip() or ("K800109221" if resolved_tech == "5G" else "CCVVPX308")
        pap_file[sel] = str(default_path) if default_path is not None and default_path.is_file() else ""

    real_filled = np.where(np.isfinite(real), real, generic)
    delta = real_filled - generic
    return pd.DataFrame(
        {
            "phase36_antenna_delta_db": delta,
            "phase36_generic_antenna_gain_dbi": generic,
            "phase36_pap_gain_dbi": real_filled,
            "phase36_pap_horizontal_gain_db": horizontal,
            "phase36_pap_vertical_gain_db": vertical,
            "phase36_pap_boresight_gain_dbi": boresight,
            "phase36_pap_model": pap_model,
            "phase36_pap_file": pap_file,
            "phase36_antenna_source": source,
        },
        index=frame.index,
    )


_HALF_WINDOW_DEG = 3


def _pat_gain_smoothed(start: int, arr: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """Power-averaged pattern read over +/- 3 deg to suppress 1-deg null spikes."""
    a = np.asarray(arr, dtype=float)
    base = np.round(np.asarray(angle_deg, dtype=float)).astype(int)
    stack = np.stack([a[(base + k - start) % len(a)] for k in range(-_HALF_WINDOW_DEG, _HALF_WINDOW_DEG + 1)])
    return 10.0 * np.log10(np.mean(10.0 ** (stack / 10.0), axis=0))


# --------------------------------------------------------------------------- pattern library
def _normalise_pattern_axis(start: int, step: float, values: np.ndarray) -> tuple[int, np.ndarray]:
    if values.size == 360 and abs(float(step) - 1.0) < 1e-9:
        return int(start), values
    angles = (float(start) + float(step) * np.arange(values.size, dtype=float)) % 360.0
    order = np.argsort(angles)
    xp = angles[order]
    fp = values[order]
    xp = np.concatenate([xp - 360.0, xp, xp + 360.0])
    fp = np.concatenate([fp, fp, fp])
    return 0, np.interp(np.arange(360, dtype=float), xp, fp)


def _parse_pap(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    hm = re.search(r"<HorizontalPatterns>.*?<StartAngle>(-?\d+)</StartAngle>.*?<Step>(\d+)</Step>.*?<Gains>([^<]+)</Gains>", txt, re.S)
    vm = re.search(r"<VerticalPatterns>.*?<StartAngle>(-?\d+)</StartAngle>.*?<Step>(\d+)</Step>.*?<Gains>([^<]+)</Gains>", txt, re.S)
    h_start, h = _normalise_pattern_axis(
        int(hm.group(1)),
        float(hm.group(2)),
        np.array([float(x) for x in hm.group(3).split(";")], dtype=float),
    )
    v_start, v = _normalise_pattern_axis(
        int(vm.group(1)),
        float(vm.group(2)),
        np.array([float(x) for x in vm.group(3).split(";")], dtype=float),
    )
    return h_start, tuple(h), v_start, tuple(v)


@lru_cache(maxsize=256)
def _load_pap(path_str: str):
    return _parse_pap(Path(path_str))


def _pattern_family(technology: str, freq_mhz: float) -> tuple[Path, str, tuple]:
    """(directory, filename template with {et}, boresight key) for a cell.

    Selection is by the cell's DEPLOYED frequency, never by the COST-231
    anchor: the anchor is a propagation-model construct and says nothing about
    which physical antenna is installed."""
    if str(technology) == "5G":
        return (PATTERN_DIR / "K800109221",
                "3300 - 3590 MHz, eTilt {et}, Y1P45 - Port1.pap", ("5G", "n78"))
    if round(float(freq_mhz), 1) <= 1000.0:
        return (PATTERN_DIR / "CCVVPX308",
                "698 - 806 MHz, T {et}, eAz 0, eBw 0, Port 1 +45.pap", ("4G", "low"))
    return (PATTERN_DIR / "CCVVPX308",
            "1710 - 1880 MHz, T {et}, eAz 0, eBw 0, Port 5 +45.pap", ("4G", "high"))


@lru_cache(maxsize=64)
def available_tilts(technology: str, freq_mhz: float) -> tuple[int, ...]:
    """Electrical tilts this antenna actually ships, discovered from the files.

    Returned instead of a hardcoded range so adding or removing a .pap file
    changes the supported set with no code edit."""
    directory, template, _ = _pattern_family(technology, freq_mhz)
    if not directory.is_dir():
        return ()
    prefix, suffix = template.split("{et}")
    found = []
    for path in directory.glob("*.pap"):
        name = path.name
        if name.startswith(prefix) and name.endswith(suffix):
            middle = name[len(prefix):len(name) - len(suffix)]
            if middle.strip().isdigit():
                found.append(int(middle.strip()))
    return tuple(sorted(set(found)))


_TILT_SNAP_LOGGED: set = set()


def _default_pap_path(technology: str, freq_mhz: float, etilt_deg: int) -> tuple[Path | None, float]:
    """Resolve the pattern file + boresight gain, snapping to the NEAREST tilt
    the antenna actually offers and saying so.

    The previous silent clamp (min/max into a hardcoded range) is what hid the
    tenths-of-a-degree tilt unit error: every unconverted tilt of 20..90 landed
    on T 10 and the run still reported success."""
    directory, template, key = _pattern_family(technology, freq_mhz)
    boresight = _DEFAULT_BORESIGHT_DBI[key]
    offered = available_tilts(technology, freq_mhz)
    if not offered:
        return None, boresight
    requested = int(round(float(etilt_deg)))
    et = min(offered, key=lambda t: (abs(t - requested), t))
    if et != requested:
        token = (key, requested, et)
        if token not in _TILT_SNAP_LOGGED:
            _TILT_SNAP_LOGGED.add(token)
            print(
                f"[LTE_OFFSET][PHASE36_TILT_SNAP] {key[0]}/{key[1]} requested_etilt={requested} deg "
                f"-> using nearest available {et} deg (offered={list(offered)})",
                flush=True,
            )
    return directory / template.format(et=et), boresight


def make_pattern_lookup(client_patterns: dict | None = None):
    """Return a `pattern_lookup(antenna_model, freq_mhz, etilt_deg, technology)` callable.

    `client_patterns`: optional {antenna_model: {"dir": <path>, "boresight_dbi": <float>,
    "file_fmt": "<...{et}...>.pap"}} to override the defaults for named models.
    Falls back to the CCVVPX308 / K800109221 defaults, then to None (generic).
    """
    client_patterns = client_patterns or {}

    def _lookup(antenna_model, freq_mhz, etilt_deg, technology="4G"):
        model = str(antenna_model or "").strip()
        etilt_deg = int(round(float(etilt_deg))) if np.isfinite(etilt_deg) else 3
        override = client_patterns.get(model)
        if override:
            try:
                path = Path(override["dir"]) / override["file_fmt"].format(et=etilt_deg)
                if path.is_file():
                    hs, h, vs, v = _load_pap(str(path))
                    return hs, h, vs, v, float(override["boresight_dbi"])
            except Exception:
                pass
        # Technology is authoritative: production stores 5G n78 as 2600 MHz, so
        # frequency alone cannot tell 4G B7 from 5G n78.
        tech = "5G" if (str(technology) == "5G" or model.upper().startswith("K800")) else "4G"
        path, g0 = _default_pap_path(tech, freq_mhz, etilt_deg)
        if path is not None and path.is_file():
            hs, h, vs, v = _load_pap(str(path))
            return hs, h, vs, v, g0
        return None  # -> generic 3GPP (delta 0)

    return _lookup


default_pattern_lookup = make_pattern_lookup()
