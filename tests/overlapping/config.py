"""
Thresholds and physical assumptions for the coverage-redundancy (overlapping) model.

Every number that can change a verdict lives here, and the full config is written into each
run's summary.json so a result can be reproduced. Defaults are planning conventions, not values
fitted to one project -- review them per market before trusting verdicts.

Capacity is deliberately absent: there is no PRB/RRC/load data yet (project 193 has none), so
the model decides coverage redundancy only. The capacity check plugs into
removal_model.run_redundancy(extra_gates=...) once load data exists.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = PACKAGE_DIR / "output"

VERDICT_REDUNDANT = "COVERAGE_REDUNDANT"   # coverage + quality kept without it; capacity NOT verified
VERDICT_KEEP = "KEEP"                      # removing it loses coverage/quality
VERDICT_DUPLICATE = "DATA_DUPLICATE"       # the same physical antenna stored twice
VERDICT_NOT_TESTABLE = "NOT_TESTABLE"      # config not trustworthy, or too little of it in the area

LEVEL_SITE = "site"
LEVEL_CELL = "cell"


@dataclass(frozen=True)
class RFParams:
    frequency_mhz: float = 1800.0          # site_prediction EARFCN is 1750 (B3) on every project 193 row
    antenna_gain_dbi: float = 18.0         # same defaults as tools/lte_prediction
    cable_loss_db: float = 2.0
    ue_height_m: float = 1.5
    default_tx_power_dbm: float = 46.0
    default_height_m: float = 30.0
    default_electrical_tilt_deg: float = 6.0
    default_mechanical_tilt_deg: float = 0.0
    # compute_sector_rsrp returns total carrier power; RSRP is per resource element.
    # 20 MHz LTE = 100 PRB x 12 subcarriers = 1200 RE.
    re_normalisation_db: float = 10.0 * math.log10(1200.0)
    # Thermal noise per 15 kHz RE (-174 dBm/Hz + 41.8 dB) plus a 7 dB UE noise figure.
    noise_per_re_dbm: float = -174.0 + 10.0 * math.log10(15000.0) + 7.0
    # Share of neighbour power that actually interferes. Fitted per operator from measured SINR
    # (rf_matrix); this default is used only when an operator logs no usable SINR. 0.5 is the
    # project 193 Airtel fit (1.0 predicted SINR 3 dB too low there).
    interference_load_factor: float = 0.5
    max_cell_radius_m: float = 3000.0      # beyond this a cell contributes nothing


@dataclass(frozen=True)
class CleaningParams:
    technology: str = "4G"
    azimuth_merge_deg: float = 15.0        # rows of one site closer than this in azimuth = one antenna
    max_antenna_azimuth_spread_deg: float = 30.0   # rows of one antenna smeared wider = config noise
    max_sectors_per_site: int = 6          # more distinct azimuths than this = config not trustworthy
    max_site_location_spread_m: float = 50.0
    duplicate_distance_m: float = 30.0     # two antennas of different site ids this close ...
    duplicate_azimuth_deg: float = 20.0    # ... and pointing the same way = one antenna stored twice
    same_pci_distance_m: float = 300.0     # same operator + PCI this close = same cell (PCI reuse is km-scale)
    same_pci_azimuth_deg: float = 45.0
    same_pci_max_pcis_per_antenna: int = 2  # antennas carrying many PCIs match anything by chance
    # Sites with an untrustworthy config are left out of the RF surface by default: a wrong
    # antenna would make its neighbours look redundant (the unsafe direction).
    ambiguous_sites_as_alternatives: bool = False


@dataclass(frozen=True)
class CalibrationParams:
    cv_folds: int = 5                      # drive sessions grouped into folds, each predicted from the others
    reference_band: str | None = None      # None = pooled level of every measured band
    min_band_rows: int = 30                # bands with fewer attributed rows use the pooled level
    shrinkage_points: int = 50             # per-antenna offset shrunk by n / (n + 50)
    max_offset_deviation_db: float = 15.0
    min_per_antenna_gain_db: float = 0.3   # per-antenna offsets kept only if they cut held-out spread this much
    min_dt_points: int = 100               # attributed drive-test rows needed to calibrate at all
    default_sigma_rsrp_db: float = 8.0     # typical urban shadowing; used when there is no drive test
    min_sigma_db: float = 3.0
    max_sigma_db: float = 12.0
    load_factor_grid: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    # RF-surface validation on held-out sessions. Failing any -> RSRP-based verdicts are withheld
    # (NOT_TESTABLE, RF_SURFACE_NOT_VALIDATED); the model's verdict stays in `model_verdict`.
    max_abs_bias_db: float = 6.0
    max_heldout_sigma_db: float = 12.0
    min_site_agreement: float = 0.40       # predicted best site == measured serving eNB
    ho_max_gap_s: float = 10.0             # consecutive serving rows further apart are not a handover


@dataclass(frozen=True)
class DecisionParams:
    level: str = LEVEL_SITE
    rsrp_threshold_dbm: float = -105.0     # coverage KPI threshold
    sinr_threshold_db: float = -3.0        # quality KPI threshold
    qrxlevmin_dbm: float = -124.0          # 3GPP TS 36.304 S-criterion floor (Srxlev > 0)
    # A point counts as covered when P(true RSRP >= rsrp_threshold) >= this, given the held-out
    # prediction sigma (the planning edge-coverage probability; equals a z*sigma fade margin).
    location_probability: float = 0.75
    # A covered point is "lost" (for holes) only if its coverage probability also fell by at least
    # this much; a 2 dB change at the coverage edge is prediction noise, not a hole.
    min_probability_drop: float = 0.25
    min_retention: float = 0.97            # reliably covered footprint weight NOT lost (see min_probability_drop)
    min_expected_retention: float = 0.90   # expected covered weight kept: guard against broad small erosion
    min_indoor_retention: float = 0.95
    max_hole_m2: float = 2500.0            # 4 grid points at 25 m
    min_uncovered_share: float = 0.10
    max_uncovered_rsrp_drop_db: float = 6.0
    max_new_low_sinr_share: float = 0.05   # footprint share pushed from >= sinr_threshold to below it
    ho_hysteresis_db: float = 3.0
    max_ho_ambiguity_increase: float = 0.25
    max_cumulative_coverage_loss: float = 0.01
    max_removals: int | None = None
    min_footprint_points: int = 8
    min_potential_points: int = 20


@dataclass(frozen=True)
class WeightParams:
    # point weight = (1 + users_weight * users / mean users) * (indoor_multiplier if in a building)
    users_weight: float = 1.0
    indoor_multiplier: float = 1.5


@dataclass(frozen=True)
class OverlapConfig:
    project_id: int = 193
    region: str = "india"
    operator: str = "Airtel"
    grid_resolution_m: float = 25.0
    evaluation_buffer_m: float = 300.0     # analysis grid extends this far outside the polygon
    site_buffer_m: float = 1500.0          # sites this far outside the polygon still serve/interfere
    rf: RFParams = field(default_factory=RFParams)
    cleaning: CleaningParams = field(default_factory=CleaningParams)
    calibration: CalibrationParams = field(default_factory=CalibrationParams)
    decision: DecisionParams = field(default_factory=DecisionParams)
    weights: WeightParams = field(default_factory=WeightParams)

    def to_dict(self) -> dict:
        return asdict(self)
