# Coverage redundancy (overlapping) test case

**Question:** if a site (or a single cell) is switched off, does the rest of the network still serve
the area it served?

**Scope:** coverage and quality only. There is no PRB / RRC / load data yet, so a
`COVERAGE_REDUNDANT` site is **not** a switch-off decision. The capacity check plugs in later as one
extra gate (see [Adding capacity later](#adding-capacity-later)); nothing else changes.

## Pipeline

```text
fetch_data.py        DB (read-only) -> data/project_<id>/   sites, polygon, drive test, buildings, users
site_cleaning.py     site_prediction rows -> physical antennas; duplicates; untrustworthy configs
rf_matrix.py         RSRP of EVERY antenna at EVERY 25 m point (production compute_sector_rsrp)
                     calibrated + validated against the drive test (session-held-out)
removal_model.py     coverage probability, per-candidate gates, one-at-a-time removal
run_overlap.py       orchestration, validation gate, outputs          -> output/project_<id>/<op>_<level>/
make_synthetic_overlap.py   known-answer scenario
test_overlap_model.py       unit + known-answer tests
overlap_dashboard.py        Streamlit review page (reads output/ only)
```

## Method

### 1. Physical antennas, not cell-id rows
`site_prediction` rows of one operator + site are grouped by azimuth (circular clustering, 15°).
Each group is one antenna: circular-mean azimuth, median tilt/height/power. Deduplicating on
`site|cell|sector|band|operator` instead drops ~80% of project 193 rows and keeps an arbitrary azimuth.

| Flag | Rule | Verdict |
|---|---|---|
| `CO_LOCATED` duplicate | another site id, ≤ 30 m, ≤ 20° apart | `DATA_DUPLICATE` (left out of RF) |
| `SAME_PCI_NEARBY` duplicate | another site id shares a PCI, ≤ 300 m, ≤ 45° (clean PCI sets only) | `DATA_DUPLICATE` |
| `TOO_MANY_AZIMUTHS` / `AZIMUTH_SMEAR` / `LOCATION_SPREAD` | > 6 azimuth groups, a group > 30° wide, rows > 50 m apart | `NOT_TESTABLE` (left out of RF) |

### 2. RF surface
For every antenna and every analysis point (polygon + 300 m, 25 m grid) RSRP comes from the
production propagation model `tools/lte_prediction` `compute_sector_rsrp` (COST-231 Hata + 3GPP
antenna pattern), converted to per-RE RSRP. SINR = serving / (load × other same-layer cells + noise).

**Why not `lte_prediction_baseline_results`:** each baseline point only holds the sectors of *one*
site (Airtel: always 3 rows of one site), so its "second best" is the sister sector. Its RSRP is also
flat at about −91 dBm from 50 m to 500 m and from boresight to behind the antenna. It is drive-test values
carried over and smoothed. Calibrating to it added +25 dB and over-predicted the drive test by 27 dB.

### 3. Drive-test calibration and validation
* Each serving drive-test row is attributed to the antenna it was measured on: serving eNB = site id,
  then PCI, else that site's strongest antenna at the spot.
* Level per band = median(measured − model); reference level = pooled over all bands.
  Per-antenna offsets (shrunk n/(n+50)) are used only if they reduce held-out error.
* **Session-grouped cross-validation** (5 folds): each drive session is predicted from the others.
  The held-out spread becomes σ, the prediction uncertainty used for coverage probability.
* The interference load factor is fitted so predicted SINR matches measured SINR (median).
* **Validation gate:** the RF surface must pass held-out |bias| ≤ 6 dB, spread ≤ 12 dB, and
  predicted best site = measured serving eNB ≥ 40% of rows. If it fails, every RSRP-based verdict is
  published as `NOT_TESTABLE (RF_SURFACE_NOT_VALIDATED)`. The model's own answer stays in `model_verdict`.

### 4. Coverage and the removal test
* P_cov(p) = Φ((RSRP − T_rsrp) / σ), and 0 below Qrxlevmin (3GPP TS 36.304 S-criterion).
  A point is covered when P_cov ≥ 0.75, the planning edge-coverage probability.
* **Footprint** F of candidate K = points whose best server is in K. Each point in F is re-evaluated
  without K: the next-best server takes over and K's interference is gone. Outside F coverage can only
  improve, so F is the only place a removal loses anything.
* **Lost point:** covered before, not covered after, **and** P_cov fell by ≥ 0.25. A 1–2 dB change at
  the coverage edge is prediction noise, not a loss. A hard threshold counted such a change as a full
  loss on the synthetic overlay site.

| Gate (all must pass) | Definition | Default |
|---|---|---|
| `RETENTION_BELOW_MIN` | 1 − lost weight / covered weight in F | ≥ 0.97 |
| `EXPECTED_COVERAGE_EROSION` | Σw·min(P_after, P_before) / Σw·P_before | ≥ 0.90 |
| `INDOOR_RETENTION_BELOW_MIN` | retention on building points | ≥ 0.95 |
| `HOLE_TOO_LARGE` | largest 8-connected block of lost points | ≤ 2,500 m² |
| `POOR_AREA_DEGRADES` | median RSRP drop where F was already uncovered (if ≥ 10% of F) | ≤ 6 dB |
| `QUALITY_DEGRADES` | share of F pushed from SINR ≥ −3 dB to below | ≤ 0.05 |
| `HO_AMBIGUITY_INCREASE` | rise in share of F where best and best other-site cell are < 3 dB apart | ≤ 0.25 |
| `NETWORK_LOSS_CAP` | cumulative expected covered-weight loss vs the original network | ≤ 1% |

SINR is judged by its change and not multiplied into P_cov, because predicted SINR has ~7 dB
held-out error. Point weights are `(1 + users_per_grid / mean) × 1.5 indoors`. User counts exist
for Airtel only; other operators get area weight.

### 5. One at a time
Removals do not add up: two overlapping sites can each look redundant only because the other is
on. Every iteration evaluates all remaining candidates, removes the safest passing one, recomputes
the network, and repeats until nothing passes. A site that passes alone but fails later is
`KEEP (NEEDED_AFTER_REMOVALS(...))`.

## Verdicts

| Verdict | Meaning |
|---|---|
| `COVERAGE_REDUNDANT` | coverage and quality kept without it, given the higher-ranked removals; capacity not verified |
| `KEEP` | removing it fails a gate (`reason`) |
| `DATA_DUPLICATE` | the same physical antenna stored twice |
| `NOT_TESTABLE` | config not trustworthy, outside the polygon, too little area, or RF surface not validated |

`confidence`: HIGH = footprint ≥ 20 points, ≥ 20 drive-test points in it, ≥ 50 attributed drive-test rows;
MEDIUM = footprint ≥ 8 points and ≥ 50 attributed rows; LOW otherwise.

## Project 193 (Indirapuram) results, data of 2026-09-11

| Operator | RF surface (held-out) | Published verdicts |
|---|---|---|
| **Airtel** site level | **PASS**: bias −0.2 dB, σ 10.1 dB, site match 0.58, load 0.5 | **18 COVERAGE_REDUNDANT**, 40 KEEP, 4 NOT_TESTABLE |
| Airtel cell level | PASS | 72 COVERAGE_REDUNDANT, 91 KEEP, 11 NOT_TESTABLE |
| JIO 4G | **FAIL**: bias −12.5 dB, σ 12.4 dB | 91 NOT_TESTABLE (withheld), 1 DATA_DUPLICATE |
| Vi India | **FAIL**: bias −19.3 dB, σ 13.8 dB, site match 0.28 | 152 NOT_TESTABLE (withheld), 117 DATA_DUPLICATE |

* Airtel: the 18 are the safest removals that fit in the **1% network coverage-loss budget**
  (0.93% used). Most of the 40 KEEPs hit that cap as well as a local gate. Redundant sites sit closer
  to their nearest neighbour (median 216 m vs 239 m for KEEP). Expected coverage goes from 81.0% to 80.2%.
* JIO / Vi: the calibrated model does not reproduce their drive test. Vi's config is mostly generated
  from logs (decimal site ids, 627 of 956 antennas are duplicates). JIO's measured serving cell is
  ~12 dB weaker than the model's best server. Fix the site data before trusting any verdict.

## Run

From the `ML/` directory:

```bash
venv\Scripts\python.exe -m tests.overlapping.fetch_data --project-id 193 --region india
venv\Scripts\python.exe -m tests.overlapping.run_overlap --project-id 193 --operator all --level site
venv\Scripts\python.exe -m tests.overlapping.run_overlap --project-id 193 --operator Airtel --level cell
venv\Scripts\python.exe -m tests.overlapping.make_synthetic_overlap
venv\Scripts\python.exe -m pytest tests/overlapping -q
venv\Scripts\python.exe -m streamlit run tests/overlapping/overlap_dashboard.py
```

Useful flags: `--rsrp-threshold -105 --sinr-threshold -3 --location-probability 0.75
--min-retention 0.97 --max-hole-m2 2500 --max-network-loss 0.01 --max-removals N --grid 25`.
All other parameters are in `config.py` and are written to each run's `summary.json`.

## Outputs (`output/project_<id>/<operator>_<level>/`)

| File | Content |
|---|---|
| `candidates.csv` | one row per site/cell: verdict, reason, rank, gate metrics, absorbers, confidence, `model_verdict` |
| `iterations.csv` | removal order and network expected coverage after each step |
| `candidate_footprints.parquet` | per candidate: its footprint points before/after removing it alone |
| `grid_states.parquet` | every point before / after all accepted removals |
| `antennas.csv`, `calibration.csv` | cleaned antennas; per-antenna drive-test rows and offsets |
| `dt_validation.csv` | drive-test rows: measured vs held-out prediction |
| `summary.json` | data counts, calibration, validation, thresholds, verdict counts, full config |

## Adding capacity later

`run_pipeline(cfg, inputs, extra_gates=[...])` takes callables that receive a candidate's metrics
(including `absorbers`, the takeover share per antenna, and `footprint_users`) and return a failure
reason or `None`:

```python
def capacity_gate(m: dict) -> str | None:
    for a in json.loads(m["absorbers"]):
        if prb_after(a["antenna_key"], a["share"]) > 0.70:   # from real PRB / load data
            return "CAPACITY_EXCEEDED"
    return None
```

A failing gate makes the site `KEEP` with that reason. When capacity is added,
`COVERAGE_REDUNDANT` becomes a switch-off recommendation.

## Assumptions and limits

* One reference layer (1800 MHz): `site_prediction` EARFCN is 1750 on every row, while the drive test
  shows B1/B3/B40/B8/B5. Band level differences are calibrated; carrier-level switch-off is not modelled.
* COST-231 + 3GPP pattern (as in production): no terrain or building path loss. Areas under a mast are
  weak in the model, which is why an overlay 50 m away already shows a small weak zone (see synthetic).
* σ ≈ 10 dB and SINR error ≈ 7 dB are large. Treat verdicts as ranked candidates for RF-engineer review
  and field check, not automatic actions.
* Thresholds are planning conventions checked on one project. Re-check them on another project.
* Handover evidence (`absorber_handover_neighbour_share`) is informative only; drive routes cover part of the area.
