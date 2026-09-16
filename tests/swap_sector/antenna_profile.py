"""Expected directional antenna profile of one sector -- no drive-test RF values.

Inputs: the site table (configured azimuth, electrical and mechanical tilt, and -- when
the site table has them -- the real antenna model and port) plus the carrier frequency of
the sector's carrier, derived from the drive-test EARFCN / NR-ARFCN. The channel number
only identifies the carrier; no measured signal enters this module.

Pattern source, in strict order; every result carries the one that was used:
  EXACT        real antenna model (+ port) from the site table and a .pap file of that
               model whose frequency range contains the carrier frequency
  ASSUMED      no real model; the technology rule (LTE -> CCVVPX308, NR -> K800109221)
               has a .pap file whose frequency range contains the carrier frequency
  APPROXIMATE  no in-band vendor file: generic 3GPP sector pattern
A vendor file of another frequency range (e.g. the 1710-1880 MHz file for B40 at
2300 MHz) is never used. Missing azimuth, tilt or frequency raises NotTestable.

Profile: horizontal gain around the site, rotated by the configured azimuth, at the
main-beam elevation. The .pap file chosen for the electrical tilt already contains that
tilt, so tilt is not applied a second time. Mechanical tilt and the elevation angle of
individual drive-test locations are not modelled (planned refinement). The profile is
relative antenna gain in dB, not received power.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

PATTERN_ROOT = Path(__file__).resolve().parents[2] / "tools" / "lte_prediction_offset" / "antenna_patterns"
ASSUMED_MODEL_BY_TECHNOLOGY = {"LTE": "CCVVPX308", "NR": "K800109221"}
QUALITY_ORDER = {"APPROXIMATE": 0, "ASSUMED": 1, "EXACT": 2}
GENERIC_3GPP = {
    "LTE": {"reference": "3GPP TR 36.814 Table A.2.1.1-2", "beamwidth_deg": 65.0, "max_attenuation_db": 25.0},
    "NR": {"reference": "3GPP TR 38.901 Table 7.3-1", "beamwidth_deg": 65.0, "max_attenuation_db": 30.0},
}
# LTE downlink channels, 3GPP TS 36.101 Table 5.7.3-1: band -> (F_DL_low MHz, N_Offs-DL, first EARFCN, last EARFCN)
LTE_DL_BANDS = {
    1: (2110.0, 0, 0, 599), 2: (1930.0, 600, 600, 1199), 3: (1805.0, 1200, 1200, 1949),
    4: (2110.0, 1950, 1950, 2399), 5: (869.0, 2400, 2400, 2649), 7: (2620.0, 2750, 2750, 3449),
    8: (925.0, 3450, 3450, 3799), 20: (791.0, 6150, 6150, 6449), 28: (758.0, 9210, 9210, 9659),
    38: (2570.0, 37750, 37750, 38249), 39: (1880.0, 38250, 38250, 38649), 40: (2300.0, 38650, 38650, 39649),
    41: (2496.0, 39650, 39650, 41589), 42: (3400.0, 41590, 41590, 43589), 43: (3600.0, 43590, 43590, 45589),
}
PAP_NAME = re.compile(r"(\d+)\s*-\s*(\d+) MHz, (?:eTilt|T) (\d+), (.+)\.pap$")


class NotTestable(ValueError):
    """A required input for the expected profile is missing, invalid or unusable."""


def carrier_frequency_mhz(channel, technology) -> tuple[float, str]:
    """Downlink carrier frequency and band from a drive-test EARFCN (LTE) or NR-ARFCN (NR)."""
    try:
        number = int(float(channel))
    except (TypeError, ValueError):
        raise NotTestable("carrier channel number is missing") from None
    if technology == "LTE":
        for band, (f_low, offset, first, last) in LTE_DL_BANDS.items():
            if first <= number <= last:
                return round(f_low + 0.1 * (number - offset), 1), f"B{band}"
        raise NotTestable(f"LTE EARFCN {number} is not in the supported band table")
    if technology == "NR":
        if 0 <= number < 600000:
            return round(0.005 * number, 3), "NR"
        if 600000 <= number < 2016667:
            return round(3000.0 + 0.015 * (number - 600000), 3), "NR"
        raise NotTestable(f"NR-ARFCN {number} is outside FR1")
    raise NotTestable(f"no channel rule for technology {technology!r}")


@lru_cache(maxsize=4)
def catalog(root: str = str(PATTERN_ROOT)) -> tuple[dict, ...]:
    records = []
    for path in sorted(Path(root).rglob("*.pap")):
        match = PAP_NAME.search(path.name)
        if match:
            low, high, tilt, port = match.groups()
            records.append({"path": str(path), "antenna_model": path.parent.name, "low_mhz": float(low),
                            "high_mhz": float(high), "e_tilt": float(tilt), "pattern_port": port.strip()})
    return tuple(records)


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _text(value) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _vendor_file(model: str, port: str, frequency: float, e_tilt: float, root) -> tuple[dict | None, str]:
    """(match, note): the in-band file of `model` with the nearest electrical tilt, or (None, why not)."""
    files = [p for p in catalog(str(root))
             if p["antenna_model"] == model and p["low_mhz"] <= frequency <= p["high_mhz"]
             and (not port or p["pattern_port"] == port)]
    if not files:
        return None, f"no {model} pattern file covers {frequency:g} MHz" + (f" on port {port}" if port else "")
    ports = sorted({p["pattern_port"] for p in files})
    if len(ports) > 1:
        return None, f"{model} has several ports covering {frequency:g} MHz ({', '.join(ports)}) and the port is unknown"
    tilt_error = min(abs(p["e_tilt"] - e_tilt) for p in files)
    chosen = min((p for p in files if abs(p["e_tilt"] - e_tilt) == tilt_error), key=lambda p: p["e_tilt"])
    match = {"antenna_model": model, "path": chosen["path"], "pattern_port": chosen["pattern_port"],
             "file_e_tilt": chosen["e_tilt"], "tilt_error_deg": tilt_error,
             "pattern_range_mhz": f"{chosen['low_mhz']:g}-{chosen['high_mhz']:g}"}
    return match, ("nearest available electrical tilt" if tilt_error else "")


def generic_pattern(pattern: dict, note: str = "") -> dict:
    """The generic 3GPP sector pattern for the same sector (azimuth, tilt, frequency, technology)."""
    generic = GENERIC_3GPP[pattern["technology"]]
    keep = {k: pattern[k] for k in ("technology", "azimuth", "requested_e_tilt", "m_tilt", "frequency_mhz")}
    return {**keep, "pattern_quality": "APPROXIMATE", "model_source": generic["reference"],
            "antenna_model": "generic 3GPP sector", "path": "", "pattern_port": "", "file_e_tilt": float("nan"),
            "tilt_error_deg": 0.0, "pattern_range_mhz": "", "note": note}


def select_pattern(sector: dict, root=PATTERN_ROOT) -> dict:
    azimuth, e_tilt = _number(sector.get("azimuth")), _number(sector.get("e_tilt"))
    m_tilt, frequency = _number(sector.get("m_tilt")), _number(sector.get("frequency_mhz"))
    missing = [name for name, value in (("azimuth", azimuth), ("electrical tilt", e_tilt),
                                        ("mechanical tilt", m_tilt), ("carrier frequency", frequency)) if value is None]
    if missing:
        raise NotTestable("missing " + ", ".join(missing))
    if not (0 <= azimuth <= 360 and -90 <= e_tilt <= 90 and -90 <= m_tilt <= 90 and frequency > 0):
        raise NotTestable("azimuth, tilt or carrier frequency is out of range")
    technology = _text(sector.get("technology"))
    if technology not in GENERIC_3GPP:
        raise NotTestable(f"no pattern rule for technology {technology or 'unknown'}")
    base = {"technology": technology, "azimuth": azimuth % 360.0, "requested_e_tilt": e_tilt,
            "m_tilt": m_tilt, "frequency_mhz": frequency}

    model, port = _text(sector.get("antenna_model")), _text(sector.get("pattern_port"))
    if model:
        match, note = _vendor_file(model, port, frequency, e_tilt, root)
        if match:
            return {**base, **match, "pattern_quality": "EXACT", "model_source": "site table", "note": note}
    else:
        assumed = ASSUMED_MODEL_BY_TECHNOLOGY[technology]
        match, note = _vendor_file(assumed, "", frequency, e_tilt, root)
        if match:
            return {**base, **match, "pattern_quality": "ASSUMED",
                    "model_source": f"technology rule ({technology} -> {assumed})", "note": note}
    return generic_pattern(base, note)


@lru_cache(maxsize=64)
def horizontal_cut(path: str) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    nodes = [n for n in root.iter("HorizontalPattern") if _number(n.findtext("Inclination")) == 0]
    if len(nodes) != 1:
        raise NotTestable(f"{Path(path).name} has no single horizontal cut")
    node = nodes[0]
    start, stop, step = (_number(node.findtext(key)) for key in ("StartAngle", "EndAngle", "Step"))
    values = [_number(v) for v in (node.findtext("Gains") or "").split(";") if v.strip()]
    gains = np.array([np.nan if v is None else v for v in values], dtype=float)
    if (None in (start, stop, step) or step <= 0 or len(gains) != round((stop - start) / step) + 1
            or not np.isfinite(gains).all() or len(gains) * step < 359):
        raise NotTestable(f"{Path(path).name} has a malformed horizontal cut")
    return start + np.arange(len(gains)) * step, gains - gains.max()


def horizontal_gain_db(pattern: dict, offset_deg) -> np.ndarray:
    """Relative gain (dB, 0 at the beam peak) at angles measured from the antenna boresight."""
    offset = (np.asarray(offset_deg, dtype=float) + 180.0) % 360.0 - 180.0
    if pattern["path"]:
        angles, gains = horizontal_cut(pattern["path"])
        return np.interp(offset, angles, gains, period=360.0)
    generic = GENERIC_3GPP[pattern["technology"]]
    return -np.minimum(12.0 * (offset / generic["beamwidth_deg"]) ** 2, generic["max_attenuation_db"])


def expected_bins(pattern: dict) -> np.ndarray:
    """36 x 10-degree relative gain (0-10, 10-20, ... degrees), power-averaged over 1-degree steps."""
    angles = np.arange(360, dtype=float) + 0.5
    gain = horizontal_gain_db(pattern, angles - pattern["azimuth"])
    return 10.0 * np.log10(np.mean(10.0 ** (gain.reshape(36, 10) / 10.0), axis=1))
