# LTE Tilt Recommendation

This module creates RF optimization recommendations from the latest LTE baseline prediction. It reads baseline RF rows, antenna/site rows, geo features, optional frontend grid analytics, and optional per-cell constraints, then runs RF-validated candidate search before saving accepted rows.

The package path is intentionally spelled `lte_tilt_recommandation` in the current codebase. Keep that spelling in imports and URLs unless the whole package is renamed.

## Public API

Base prefix:

```text
/api/lte-tilt-recommandation
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/optimize` | `POST` | Starts an RF recommendation job. Accepts JSON or multipart form data. |
| `/status/<job_id>` | `GET` | Returns in-memory job state. |
| `/download?file=<xlsx_path>` | `GET` | Downloads the generated Excel report. |

Required field:

| Field | Meaning |
| --- | --- |
| `project_id` | Project id. |

Common optional fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `region` / `country_code` / `countryCode` | `india` | Region selector with India/Taiwan aliases. |
| `operator` | all | Operator filter; blank/all means all operators. |
| `rsrp` | `-105` | Bad RSRP threshold. |
| `rsrq` | `-15` | Bad RSRQ threshold. |
| `sinr` | `0` | Bad SINR threshold. |
| `mode` / `kpi_mode` / `recommendation_mode` | `combined_weighted` | One of `combined_weighted`, `rsrp_only`, `rsrq_only`, `sinr_only`. |
| `rsrp_weight`, `rsrq_weight`, `sinr_weight` | `34/33/33` | Combined KPI weights. |
| `validate_candidates` | `true` | Production requires this to be true. |
| `max_validation_candidates` / `max_candidates` | `25` | Candidate search limit. |
| `radius_m` / `radius` | `500` | RF recompute radius. |
| `grid_resolution_m` / `grid_resolution` | `30` | RF recompute grid size. |
| `n_workers` / `workers` | `1` | RF worker count. |
| `impact_radius_m` | radius | Affected-area radius for candidate validation. |
| `neighbor_site_count` | `3` | Neighbor sites included in validation. |
| `max_interference_sites` | `10` | Interference-site cap. |
| `coordinate_passes` | `2` | Coordinate/candidate search passes. |
| `candidate_workers` | `1` | Candidate evaluation workers. |
| `bad_grid_coverage_pct` | `80` | Bad-grid coverage requirement for target selection/scoring. |
| `max_group_cells` | `0` | Optional cap for grouped cell handling. |
| `max_neighbors_per_update_cell` | `2` | Neighbor expansion limit. |
| `threshold_file_path` / `threshold_file` | none | Constraint CSV path, or multipart uploaded constraint file. |

## Code Map

| File | Current role |
| --- | --- |
| `routes.py` | Accepts JSON/form requests, stores uploaded threshold files, resolves region, starts jobs. |
| `services.py` | Fetches antenna/baseline/geo/grid rows, loads constraints, runs engine, writes artifacts and DB rows. |
| `recommendation_engine.py` | Normalizes KPI modes/weights and orchestrates RF-validated recommendation generation. |
| `candidate_validation.py` | Target cell selection, candidate update generation, RF recompute, before/after scoring, debug exports. |
| `cell_identity.py` | Canonical identity helpers shared with baseline and optimized prediction. |
| `etilt_optimizer_cd2.py` | Shared ETilt/legacy optimizer functions used by the production engine. |
| `geo_logic.py` | Geo-aware helper logic used by older/supporting recommendation paths. |

## Production Flow

```text
POST /api/lte-tilt-recommandation/optimize
  -> route accepts JSON or multipart threshold upload
  -> RFOptimizationService.submit()
  -> fetch antenna rows from site_prediction
  -> normalize antenna identities and antenna fields
  -> resolve latest LTE baseline job
  -> fetch baseline rows from lte_prediction_baseline_results
  -> prepare baseline log rows and attach antenna context
  -> fetch lte_prediction_geo_features
  -> fetch optional frontend grid analytics
  -> allocate next RF recommendation scenario id
  -> load optional constraint CSV
  -> run_recommendation_engine()
  -> coordinate_search_recommendations()
  -> write Excel/debug artifacts
  -> apply constraint ranges to final recommendations
  -> append accepted rows to rf_optimization_results
  -> job status done/failed
```

## Input Data

### Antenna Rows

Antenna rows come from PythonBridge `GetLteTiltAntennaRows` or direct `site_prediction`.

The service normalizes:

- `Node_Cell_ID`
- `local_cell_id`
- `lat`, `lon`
- `azimuth`
- `electrical_tilt`
- `mechanical_tilt`
- `tx_power`
- `antenna_height`
- `dashboard_site_id`
- `Technology`

Strict identity columns such as `site_prediction_key` or `site_cell_sector_band_operator_key` are preferred when available.

### Baseline Rows

The service uses the latest baseline job id for the project and then loads rows from PythonBridge `GetLteBaselineRows` or direct `lte_prediction_baseline_results`.

Required baseline metrics:

```text
lat, lon, pred_rsrp, pred_rsrq, pred_sinr, job_id
```

When present, topology/interference columns are loaded too, including best interferer and neighbor fields. These improve candidate scope and validation.

### Geo Features

Geo rows come from PythonBridge `GetLtePredictionGeoFeatures` or direct `lte_prediction_geo_features`.

The recommendation engine uses geo context such as clutter, morphology, building density, LOS/NLOS blockage, terrain, site density, serving distance, nearest-site distance, and azimuth delta.

### Grid Analytics

Frontend grid analytics comes from the bridge grid analytics helper or direct `grid_analytics_results`. When multiple scenarios exist, direct DB mode selects a scenario by row count and recent timestamp. Grid analytics can be used as the reference grid population for before/after scoring.

## Candidate Validation

Production recommendation requires `validate_candidates=true`. If it is false, `run_recommendation_engine()` raises an error because the geo-only fallback is disabled.

The current validation flow:

1. Scores baseline prediction points against active KPI thresholds.
2. Scores frontend grid analytics when available.
3. Selects bad/priority cells from combined weighted severity or KPI-only mode.
4. Builds candidate antenna updates, mainly ETilt plus azimuth support in the validation helpers.
5. Applies candidate updates to a site DataFrame.
6. Runs optimized RF prediction for candidate cells using the optimized prediction engine.
7. Applies RF delta to the baseline population.
8. Scores before/after KPI impact.
9. Accepts candidates only when constraints pass and the candidate improves bad-grid/weighted impact.

## KPI Modes

| Mode | Active KPI thresholds | Weights |
| --- | --- | --- |
| `combined_weighted` | RSRP, RSRQ, SINR | Normalized `rsrp_weight`, `rsrq_weight`, `sinr_weight`. |
| `rsrp_only` | RSRP only | RSRP weight `1.0`. |
| `rsrq_only` | RSRQ only | RSRQ weight `1.0`. |
| `sinr_only` | SINR only | SINR weight `1.0`. |

Inactive thresholds are internally disabled with a very low sentinel value.

## Recommendation Parameters

The current code can normalize and apply several antenna parameters:

| Parameter | Support status |
| --- | --- |
| `ETilt` | Main production candidate path through RF validation. |
| `Azimuth` | Candidate helper exists and supports constraint bounds. |
| `TX Power` | Apply/constraint plumbing exists, but production candidate generation must be validated before broad rollout. |
| `Mechanical Tilt` | Apply/constraint plumbing exists, but production candidate generation must be validated before broad rollout. |
| `Height` | Apply/constraint plumbing exists, but production candidate generation must be validated before broad rollout. |

Rows are saved only after the engine returns accepted recommendations.

## Constraint File

The route accepts a multipart upload named `file` or `threshold_file`, or a path supplied as `threshold_file_path` / `threshold_file`.

Default fallback name:

```text
lte_tilt_recommendation_transformed.csv
```

Expected constraint concepts:

| Column concept | Meaning |
| --- | --- |
| `cell_id` | Target cell identity. |
| `optimised` | Whether the cell is eligible. |
| `min_e_tilt`, `max_e_tilt` | ETilt bounds. |
| `min_m_tilt`, `max_m_tilt` | Mechanical tilt bounds. |
| `min_height`, `max_height` | Antenna height bounds. |
| `min_azimuth`, `max_azimuth` | Azimuth bounds. |
| `min_tx_power`, `max_tx_power` | TX power bounds. |

The service can clamp/mark recommendations according to the parameter and writes constrained recommendation rows back to the Excel workbook.

## Output Artifacts

Each job writes into:

```text
ML/outputs/temp_<job_id>/
```

Important artifacts:

| Artifact | Meaning |
| --- | --- |
| `RF_Optimization_Report.xlsx` | Main downloadable workbook. |
| `candidate_validation_results.csv` | Candidate metrics and pass/fail context. |
| `frontend_grid_scores.csv` | Scored frontend grid rows when available. |
| `best_candidate_before_scope.csv.gz` | Before RF population for best candidate. |
| `best_candidate_after_scope.csv.gz` | After RF population for best candidate. |
| `best_candidate_before_bad_combined.csv.gz` | Bad-grid subset before. |
| `best_candidate_after_bad_combined.csv.gz` | Bad-grid subset after. |
| `best_candidate_before_grid_metrics.csv` | Grid metrics before. |
| `best_candidate_after_grid_metrics.csv` | Grid metrics after. |
| `combined_kpi_grid_impact.csv` | Combined KPI before/after grid impact. |
| `best_candidate_summary.json` | Selected candidate summary. |
| `tilt_rf_debug.log` | RF debug log path from candidate validation config. |

## Output Table

Accepted recommendations are appended to:

```text
rf_optimization_results
```

Saved columns include:

| Column | Meaning |
| --- | --- |
| `project_id` | Project id. |
| `scenario_id` | RF recommendation scenario id. |
| `operator` | Actual operator for the cell. |
| `cell_id` | Target cell identity. |
| `technology` | Usually `4G`. |
| `parameter` | Recommended parameter. |
| `current_value` | Current value. |
| `recommended_value` | Accepted recommended value. |
| `reason` | Engine reason text. |
| `swap_sector_detected` | Swap-sector suspicion flag. |
| `rsrp_threshold`, `rsrq_threshold`, `sinr_threshold` | Thresholds used for this job. |
| `created_at` | Save timestamp. |

`rf_optimization_results.scenario_id` is only the RF recommendation scenario id. It is not the optimized LTE scenario id.

## Scenario Numbering

Direct DB mode gets the next RF scenario with:

```sql
SELECT COALESCE(MAX(scenario_id), 0) + 1
FROM rf_optimization_results
WHERE project_id = :pid
```

Bridge mode uses `GetNextRfOptimizationScenarioId`.

## Bridge Endpoints Used

| Operation | Bridge endpoint |
| --- | --- |
| Antenna/site rows | `GetLteTiltAntennaRows` |
| Latest baseline job id | `GetLatestLteBaselineJobId` |
| Baseline rows | `GetLteBaselineRows` |
| Geo feature rows | `GetLtePredictionGeoFeatures` |
| Frontend grid analytics | `/api/GridAnalytics/GetGridAnalytics` |
| Next RF recommendation scenario | `GetNextRfOptimizationScenarioId` |
| Save RF recommendation rows | `SaveRfOptimizationResults` |

Without bridge mode, direct SQLAlchemy access is used.

## Downstream Link

The optimized prediction module consumes saved rows through:

```text
POST /api/lte-prediction-optimised/recommendation-optimized
```

It reads `rf_optimization_results.scenario_id` as `recommendation_scenario_id`, applies those recommendations to site rows, saves a public optimized site scenario, and writes optimized RF rows.

## Debug Markers

| Marker | Meaning |
| --- | --- |
| `[TILT][JOB_START]` | Job request, region, operator, and threshold summary. |
| `[TILT][ANTENNA_FETCH]` | Antenna/site input loaded. |
| `[TILT][BASELINE_FETCH]` | Baseline rows and latest baseline job loaded. |
| `[TILT][GEO_FETCH]` | Geo features loaded. |
| `[TILT][GRID_ANALYTICS_FETCH]` | Frontend grid analytics loaded or skipped. |
| `[TILT][CONSTRAINT_FILE]`, `[TILT][CONSTRAINT_FETCH]` | Constraint file selection and load. |
| `[TILT][ENGINE_CONFIG]` | Final engine config. |
| `[TILT_TARGET_SELECTION_EMPTY]` | No target cells survived bad-area selection. |
| `[TILT][CONSTRAINT_APPLY]` | Constraints applied to final rows. |
| `[TILT][DB_WRITE]`, `[TILT][DB_WRITE_DONE]` | RF recommendation persistence. |

## Common Failure Checks

1. Confirm a successful LTE baseline exists for the project.
2. Confirm the selected operator has matching site and baseline rows.
3. Check identity completeness: site, cell, sector, band, operator.
4. Check whether grid analytics is empty or using an unexpected scenario.
5. Check the constraint file for `optimised=false` or overly tight bounds.
6. Check candidate validation artifacts to see whether all candidates failed constraints or worsened priority areas.
7. Query downstream optimized prediction with `recommendation_scenario_id = rf_optimization_results.scenario_id`.
