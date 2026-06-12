# LTE Tilt Recommendation

This tool finds problematic LTE cells/areas and writes RF optimization recommendations into `rf_optimization_results`.

Note: the folder name is currently spelled `lte_tilt_recommandation` in the codebase. Keep that spelling in imports and paths unless the package is renamed everywhere.

## Where It Lives

| File | Role |
| --- | --- |
| `routes.py` | Flask endpoints under `/api/lte-tilt-recommandation` |
| `services.py` | Background job, DB/bridge reads, constraints, report/DB write |
| `recommendation_engine.py` | Production recommendation orchestration |
| `candidate_validation.py` | Candidate target selection, RF validation, before/after scoring |
| `geo_logic.py` | Geo-aware scoring and older recommendation logic helpers |
| `etilt_optimizer_cd2.py` | Shared/legacy ETilt optimizer functions used by the engine |
| `cell_identity.py` | Canonical cell id helpers |
| `candidate_validation.py` | Current candidate evaluation core |

## Code Architecture Map

| Layer | Code reference | What it does |
| --- | --- | --- |
| Optimize route | `tools/lte_tilt_recommandation/routes.py:13` | Registers `POST /optimize`, accepts JSON or uploaded threshold file |
| Status route | `tools/lte_tilt_recommandation/routes.py:42` | Returns job state |
| Download route | `tools/lte_tilt_recommandation/routes.py:52` | Downloads generated Excel report |
| Service class | `tools/lte_tilt_recommandation/services.py:708` | `RFOptimizationService` owns job lifecycle |
| Job submit | `tools/lte_tilt_recommandation/services.py:710` | Creates RF recommendation job id and starts background work |
| Main job runner | `tools/lte_tilt_recommandation/services.py:730` | Fetches inputs, runs recommendation engine, writes report and DB rows |
| Antenna normalization | `tools/lte_tilt_recommandation/services.py:169` | `_prepare_tilt_antenna_df()` normalizes `site_prediction` fields |
| Baseline/log normalization | `tools/lte_tilt_recommandation/services.py:212` | `_prepare_tilt_log_df()` enriches baseline rows with antenna context |
| Baseline direct fetch | `tools/lte_tilt_recommandation/services.py:277` | Reads latest baseline rows from `lte_prediction_baseline_results` |
| Grid analytics fetch | `tools/lte_tilt_recommandation/services.py:316` | Reads latest/selected `grid_analytics_results` |
| Bridge RF save | `tools/lte_tilt_recommandation/services.py:424` | Sends rows to bridge `SaveRfOptimizationResults` |
| Production engine | `tools/lte_tilt_recommandation/recommendation_engine.py:208` | `run_recommendation_engine()` orchestrates validated recommendation creation |
| Candidate search | `tools/lte_tilt_recommandation/candidate_validation.py:1109` | `coordinate_search_recommendations()` selects and evaluates candidate changes |
| Site update apply helper | `tools/lte_tilt_recommandation/candidate_validation.py:170` | `_apply_updates_to_site_df()` applies candidate updates to site DataFrame |
| ETilt candidate maker | `tools/lte_tilt_recommandation/candidate_validation.py:886` | `_make_etilt_update()` generates the current stable production candidate type |
| Geo-aware legacy/helper logic | `tools/lte_tilt_recommandation/geo_logic.py:442` | `build_geo_aware_recommendations()` contains geo scoring logic for ETilt/Azimuth/TX support |

## Recommendation Architecture

Current production recommendation is a validated candidate-search workflow.

```text
POST /api/lte-tilt-recommandation/optimize
  -> routes.py:13
  -> RFOptimizationService.submit()
  -> RFOptimizationService._run()
  -> fetch site_prediction antenna rows
  -> _prepare_tilt_antenna_df()
  -> fetch latest lte_prediction_baseline_results
  -> _prepare_tilt_log_df()
  -> fetch grid_analytics_results
  -> fetch lte_prediction_geo_features
  -> load optional threshold constraints
  -> run_recommendation_engine()
  -> coordinate_search_recommendations()
  -> candidate RF before/after scoring
  -> write Excel report
  -> append accepted rows to rf_optimization_results
```

The important production distinction:

| Capability | Current status |
| --- | --- |
| ETilt candidate generation | Production active through `_make_etilt_update()` |
| ETilt RF validation | Production active through candidate search and before/after scoring |
| Azimuth/TX/Mechanical/Height normalization | Present in service and candidate helpers |
| Azimuth/TX scoring support | Present in geo helper logic |
| Azimuth/TX/Mechanical/Height production candidate generation | Not fully production-enabled as the main coordinate search path |

The current production interpretation is therefore: ETilt recommendations are validated end to end, while additional antenna parameters require dedicated candidate generation and RF validation before being enabled as primary recommendation actions.

## API

### `POST /api/lte-tilt-recommandation/optimize`

The route accepts JSON or multipart form data.

Required field:

| Field | Meaning |
| --- | --- |
| `project_id` | Project id |

Common optional fields:

| Field | Meaning |
| --- | --- |
| `region` | Defaults to `india` |
| `operator` | Operator filter; empty/all means all operators |
| `rsrp` | Bad RSRP threshold, default `-105` |
| `rsrq` | Bad RSRQ threshold, default `-15` |
| `sinr` | Bad SINR threshold, default `0` |
| `mode` / `kpi_mode` / `recommendation_mode` | KPI weighting mode, default `combined_weighted` |
| `threshold_file_path` / `threshold_file` | Per-cell constraints file |
| `threshold_file` upload | Multipart uploaded constraints file |
| `validate_candidates` | Current production expects this to be true |
| `max_validation_candidates` / `max_candidates` | Candidate search limit |
| `candidate_workers` | Candidate evaluation worker count |

Status and download:

```text
GET /api/lte-tilt-recommandation/status/<job_id>
GET /api/lte-tilt-recommandation/download?file=<xlsx_path>
```

## Current Production Flow

1. The route validates `project_id` and submits a background job.
2. The service fetches antenna/site rows from `site_prediction`.
3. Site rows are normalized into antenna fields:
   - `Node_Cell_ID`
   - `lat`
   - `lon`
   - `azimuth`
   - `electrical_tilt`
   - `mechanical_tilt`
   - `tx_power`
   - `antenna_height`
4. The service loads the latest baseline job id from `lte_prediction_baseline_results`.
5. Baseline RF rows are fetched for the selected project/operator.
6. Baseline rows are prepared and joined with antenna context.
7. Grid analytics rows are fetched from `grid_analytics_results` when available.
8. Geo feature rows are fetched from `lte_prediction_geo_features`.
9. A threshold/constraint CSV is loaded if supplied.
10. `run_recommendation_engine()` runs the production candidate validation flow.
11. Candidate evaluation results and best-candidate debug artifacts are saved under a temp output folder.
12. Final recommendations are constrained by per-cell allowed min/max ranges.
13. An Excel report is generated.
14. Only RF-validated recommendation rows are written to `rf_optimization_results`.
15. The job status becomes `done` or `failed`.

## Current Candidate Evaluation Reality

The current production candidate search is ETilt-centered.

The code normalizes and can apply multiple antenna fields:

- `ETilt`
- `Azimuth`
- `TX Power`
- `Mechanical Tilt`
- `Height`

However, the coordinate candidate generation currently uses ETilt update generation as the main production path. The RF validation engine proves before/after improvement for those ETilt candidates before rows are saved.

Operational implications:

- ETilt recommendations are the established production path.
- Azimuth, TX power, mechanical tilt, and height are partially supported through normalization, constraints, and apply logic.
- Full production readiness for additional antenna parameters requires multi-parameter candidate generation and validation before those recommendations are persisted at scale.

Recommended rollout for future antenna expansion:

1. Keep ETilt stable.
2. Add Azimuth candidate generation and RF validation.
3. Add TX Power candidate generation and RF validation.
4. Add Mechanical Tilt and Height only with strict bounds and likely manual approval.

Additional antenna-parameter recommendations should not be enabled solely through the UI or output schema. Each parameter requires validated candidate generation before it is treated as production-ready.

## Input Tables

| Table | Used for |
| --- | --- |
| `site_prediction` | Antenna/site metadata |
| `lte_prediction_baseline_results` | Latest baseline RF prediction rows |
| `lte_prediction_geo_features` | Geo, terrain, clutter, blockage, distance, and azimuth context |
| `grid_analytics_results` | Frontend grid population and KPI context when available |
| `rf_optimization_results` | Used for scenario numbering and later downstream optimized prediction |

## Output

### Excel report

The service writes an Excel report under the ML output/temp area. The download endpoint returns that file.

The workbook includes recommendation data and supporting forecast/candidate information generated by the recommendation engine.

### `rf_optimization_results`

Accepted recommendations are appended to this table.

Important columns:

| Column | Meaning |
| --- | --- |
| `project_id` | Project id |
| `scenario_id` | RF recommendation scenario id |
| `operator` | Operator |
| `cell_id` | Target cell |
| `technology` | Usually `4G` |
| `parameter` | Recommended parameter, such as `ETilt` |
| `current_value` | Current antenna value |
| `recommended_value` | Recommended antenna value |
| `reason` | Human-readable reason |
| `swap_sector_detected` | Swap-sector suspicion flag |
| `rsrp_threshold`, `rsrq_threshold`, `sinr_threshold` | Thresholds used by the job |
| `created_at` | Save timestamp |

`rf_optimization_results.scenario_id` is not the optimized LTE scenario id. It is only the RF recommendation scenario id.

## Scenario Numbering

The service gets the next RF recommendation scenario id from:

```sql
SELECT COALESCE(MAX(scenario_id), 0) + 1
FROM rf_optimization_results
WHERE project_id = :pid
```

With the bridge enabled, the .NET endpoint handles this same scenario lookup/write path.

Downstream optimized prediction reads this RF scenario through `recommendation_scenario_id`.

## Constraint File Behavior

The threshold/constraint file can control whether a recommendation is allowed for a cell.

Expected constraint concepts include:

| Constraint | Used for |
| --- | --- |
| `optimised` | Whether the cell can be optimized |
| `min_e_tilt`, `max_e_tilt` | ETilt bounds |
| `min_m_tilt`, `max_m_tilt` | Mechanical tilt bounds |
| `min_height`, `max_height` | Antenna height bounds |
| `min_azimuth`, `max_azimuth` | Azimuth bounds |
| `min_tx_power`, `max_tx_power` | TX power bounds |

If a recommendation is outside bounds, the service clamps or marks it according to the parameter logic. Constraint application is logged with:

```text
[TILT][CONSTRAINT_APPLY]
```

## Bridge Behavior

When configured, the service uses the Python bridge for supported reads/writes.

| Operation | Bridge endpoint |
| --- | --- |
| Latest baseline job id | `GetLatestLteBaselineJobId` |
| Fetch baseline rows | `GetLteBaselineRows` |
| Fetch antenna rows | `GetLteTiltAntennaRows` |
| Fetch geo features | `GetLtePredictionGeoFeatures` |
| Save RF recommendation rows | `SaveRfOptimizationResults` |

Without the bridge, direct SQLAlchemy reads/writes are used through `DATABASE_URL` or `DATABASE_URL_Taiwan`.

## Debug Artifacts

During candidate validation the service writes debug artifacts such as:

| Artifact | Purpose |
| --- | --- |
| `candidate_validation_results.csv` | Evaluated candidates and scores |
| `best_candidate_before_scope.csv.gz` | RF population before selected candidate |
| `best_candidate_after_scope.csv.gz` | RF population after selected candidate |
| `best_candidate_before_bad_combined.csv.gz` | Bad-grid subset before |
| `best_candidate_after_bad_combined.csv.gz` | Bad-grid subset after |
| `best_candidate_before_grid_metrics.csv` | Grid metrics before |
| `best_candidate_after_grid_metrics.csv` | Grid metrics after |
| `best_candidate_summary.json` | Selected candidate summary |

These files are useful when explaining why a recommendation was accepted or rejected.

## Important Logs

| Marker | Meaning |
| --- | --- |
| `[TILT][JOB_START]` | Job config and thresholds |
| `[TILT][ANTENNA_FETCH]` | Site/antenna rows loaded |
| `[TILT][BASELINE_FETCH]` | Baseline rows loaded |
| `[TILT][GRID_ANALYTICS_FETCH]` | Grid analytics rows loaded |
| `[TILT][CONSTRAINT_FILE]` | Constraint file selected |
| `[TILT][CONSTRAINT_APPLY]` | Constraints applied to recommendations |
| `[TILT][DB_WRITE]` | RF recommendation DB write started or skipped |
| `[TILT][DB_WRITE_DONE]` | RF recommendation DB write completed |

## Common Debug Checks

If no RF rows are saved:

1. Check that baseline prediction exists for the project.
2. Check the selected operator; wrong operator filters can remove all rows.
3. Check candidate validation artifacts to see if all candidates were rejected.
4. Check whether the constraint file has `optimised = false` for target cells.
5. Check `[TILT][DB_WRITE] skipped=True` logs; this means no RF-validated rows survived to save.
6. Confirm `rf_optimization_results` is queried by the RF recommendation scenario id, not optimized LTE scenario id.

## Production Notes

- This module should write recommendations only after RF validation.
- The downstream optimized prediction module consumes rows from `rf_optimization_results`.
- A successful recommendation save does not automatically mean optimized LTE prediction has run; that is a separate endpoint.
- Keep RF recommendation scenario ids separate from LTE optimization scenario row ids.
