# LTE Prediction Optimised

This tool runs LTE prediction after a site configuration change. It has two production paths:

1. Manual optimized prediction from saved frontend site edits.
2. Recommendation optimized prediction from accepted RF recommendation rows.

The output is saved to `lte_prediction_optimised_results` and linked to `lte_optimization_scenarios`.

## Where It Lives

| File | Role |
| --- | --- |
| `routes.py` | Flask endpoints under `/api/lte-prediction-optimised` |
| `services.py` | Scenario creation, site-overlay logic, recommendation apply, DB writes |
| `ml_engine.py` | Optimized RF prediction, site normalization, affected-cell calculation |

## Code Architecture Map

| Layer | Code reference | What it does |
| --- | --- | --- |
| Manual route | `tools/lte_prediction_optimised/routes.py:12` | Registers `/run` and `/optimized` for manual optimized prediction |
| Recommendation route | `tools/lte_prediction_optimised/routes.py:34` | Registers `/recommendation-optimized` |
| Status route | `tools/lte_prediction_optimised/routes.py:52` | Returns job state |
| Download route | `tools/lte_prediction_optimised/routes.py:66` | Downloads generated CSV |
| Service class | `tools/lte_prediction_optimised/services.py:686` | `LTEPredictionService_optimised` owns both optimized paths |
| Manual submit | `tools/lte_prediction_optimised/services.py:688` | Creates/reuses scenario and starts manual optimized job |
| Recommendation submit | `tools/lte_prediction_optimised/services.py:735` | Creates recommendation-linked scenario and starts recommendation optimized job |
| Manual job runner | `tools/lte_prediction_optimised/services.py:783` | Loads baseline/site/optimized rows and predicts changed affected area |
| Recommendation job runner | `tools/lte_prediction_optimised/services.py:915` | Applies RF recommendation rows to sites and predicts before/after delta |
| Latest RF scenario | `tools/lte_prediction_optimised/services.py:116` | Finds latest `rf_optimization_results.scenario_id` when request omits it |
| RF recommendation fetch | `tools/lte_prediction_optimised/services.py:150` | Fetches rows from `rf_optimization_results` |
| Actionable filter | `tools/lte_prediction_optimised/services.py:250` | Keeps supported changed recommendation parameters |
| Apply recommendations | `tools/lte_prediction_optimised/services.py:295` | Applies recommendation values to normalized site rows |
| Site scenario row builder | `tools/lte_prediction_optimised/services.py:380` | Converts modified site rows into `site_prediction_optimized` update payload |
| Site API save | `tools/lte_prediction_optimised/services.py:443` | Saves site scenario through `/api/MapView/UpdateSitePrediction` when bridge is available |
| Site direct save | `tools/lte_prediction_optimised/services.py:504` | Direct DB fallback for `site_prediction_optimized` |
| Site scenario save wrapper | `tools/lte_prediction_optimised/services.py:612` | Runs API save first, direct DB fallback second |
| Baseline fetch | `tools/lte_prediction_optimised/ml_engine.py:245` | Loads baseline prediction rows |
| Site fetch | `tools/lte_prediction_optimised/ml_engine.py:538` | Loads baseline `site_prediction` rows |
| Optimized site fetch | `tools/lte_prediction_optimised/ml_engine.py:630` | Loads `site_prediction_optimized` scenario rows |
| K1/K2 calibration | `tools/lte_prediction_optimised/ml_engine.py:928` | Computes local calibration for target cells |
| Affected cells | `tools/lte_prediction_optimised/ml_engine.py:955` | Expands changed cells to neighbor/affected cells |
| Optimized RF model | `tools/lte_prediction_optimised/ml_engine.py:1076` | Runs optimized prediction for affected cells |
| DB formatter | `tools/lte_prediction_optimised/services.py:1156` | Sets `scenario_id` and `public_scenario_id` on output rows |
| Optimized DB write | `tools/lte_prediction_optimised/services.py:1085` | Writes `lte_prediction_optimised_results` |
| Next public scenario | `tools/lte_prediction_optimised/services.py:1212` | Finds available public scenario slot |
| Scenario prune | `tools/lte_prediction_optimised/services.py:1234` | Removes oldest scenario when max scenario count is reached |
| Scenario create | `tools/lte_prediction_optimised/services.py:1286` | Inserts/creates `lte_optimization_scenarios` row |
| Scenario status | `tools/lte_prediction_optimised/services.py:1374` | Updates scenario status by internal row id |

## Model Architecture

The optimized model is an affected-area re-prediction pipeline.

Manual path:

```text
Frontend saves site edits
  -> site_prediction_optimized.scenario = public scenario id
  -> POST /api/lte-prediction-optimised/optimized
  -> services.submit()
  -> _create_scenario()
  -> _run()
  -> fetch_baseline()
  -> fetch_site_data()
  -> fetch_optimized_sites()
  -> detect changed cells
  -> _compute_affected_cells()
  -> compute_k1k2_for_cells()
  -> run_prediction_only_optimized()
  -> _format_for_db()
  -> _save_to_db()
```

Recommendation path:

```text
RF recommendation rows exist
  -> rf_optimization_results.scenario_id = recommendation_scenario_id
  -> POST /api/lte-prediction-optimised/recommendation-optimized
  -> submit_recommendation_optimization()
  -> _create_scenario(target_id = rf_scenario_<id>)
  -> _run_recommendation_optimization()
  -> _fetch_recommendation_rows()
  -> _actionable_recommendations()
  -> _apply_recommendations_to_sites()
  -> _save_recommendation_site_prediction_scenario()
  -> run baseline and optimized RF
  -> save lte_prediction_optimised_results
```

The model does not recompute the whole project by default. It focuses on changed cells and nearby affected cells so planners can compare local before/after impact efficiently.

## API

### Manual optimized prediction

```text
POST /api/lte-prediction-optimised/run
POST /api/lte-prediction-optimised/optimized
```

Required fields:

| Field | Meaning |
| --- | --- |
| `project_id` | Project id |
| `radius` | Prediction radius |
| `grid_resolution` | Grid resolution |
| `operator` | Operator to run |

Common optional fields:

| Field | Meaning |
| --- | --- |
| `region` | Defaults to `india` |
| `site_prediction_scenario_id` / `sitePredictionScenarioId` / `scenario` | Public site scenario to read from `site_prediction_optimized.scenario` |
| `polygon_ids` / `polygonIds` | Optional region filter |
| `impact_radius_m` | Radius for affected-area expansion |
| `neighbor_site_count` | Number of neighbor sites included around changed cells |
| `max_interference_sites` | RF interference limit |
| `max_neighbors_per_update_cell` | Neighbor expansion limit around changed cells |
| `scenario_name` | Optional display name |
| `scenario_description` | Optional description |

### Recommendation optimized prediction

```text
POST /api/lte-prediction-optimised/recommendation-optimized
```

Required fields:

| Field | Meaning |
| --- | --- |
| `project_id` | Project id |
| `radius` | Prediction radius |
| `grid_resolution` | Grid resolution |

Common optional fields:

| Field | Meaning |
| --- | --- |
| `operator` | Optional recommendation/operator filter |
| `recommendation_scenario_id` | RF recommendation scenario id from `rf_optimization_results.scenario_id`; if missing, latest is selected |
| `region` | Defaults to `india` |
| `polygon_ids` / `polygonIds` | Optional region filter |
| `impact_radius_m` | Affected-area radius |
| `neighbor_site_count` | Neighbor expansion count |

Status and download:

```text
GET /api/lte-prediction-optimised/status/<job_id>
GET /api/lte-prediction-optimised/download?file=<csv_path>
```

## Scenario ID Rules

This is the most important part of the module.

| Table | Column | Meaning |
| --- | --- | --- |
| `lte_optimization_scenarios` | `id` | Internal optimization scenario row id |
| `lte_optimization_scenarios` | `scenario_id` | Public frontend scenario id |
| `lte_prediction_optimised_results` | `scenario_id` | Internal scenario row id |
| `lte_prediction_optimised_results` | `public_scenario_id` | Public frontend scenario id |
| `site_prediction_optimized` | `scenario` | Public frontend scenario id |
| `rf_optimization_results` | `scenario_id` | RF recommendation scenario id |

Example:

```text
rf_optimization_results.scenario_id = 19
lte_optimization_scenarios.id = 69
lte_optimization_scenarios.scenario_id = 1
lte_prediction_optimised_results.scenario_id = 69
lte_prediction_optimised_results.public_scenario_id = 1
site_prediction_optimized.scenario = 1
```

So frontend should display/select public scenario `1`, while optimized result storage uses internal row id `69`.

## Current Production Flow: Manual Site Scenario

1. The route validates `project_id`, `radius`, `grid_resolution`, and `operator`.
2. `submit()` creates or reuses an optimization scenario.
3. If `site_prediction_scenario_id` is provided, it is treated as the requested public scenario id.
4. Scenario metadata is written to `lte_optimization_scenarios`.
5. The job status is set to `running`.
6. Baseline RF rows are loaded from `lte_prediction_baseline_results`.
7. Baseline site rows are loaded from `site_prediction`.
8. Optimized site rows are loaded from `site_prediction_optimized`, using the public scenario id when supplied.
9. Changed cells are detected by comparing baseline site values against optimized site values.
10. Affected cells are expanded using impact radius and neighbor-site settings.
11. K1/K2 calibration is computed for target cells from the baseline prediction source.
12. `run_prediction_only_optimized()` predicts optimized RF for the affected population.
13. CSV output is saved under `outputs/`.
14. DB rows are formatted with:
    - `scenario_id = lte_optimization_scenarios.id`
    - `public_scenario_id = lte_optimization_scenarios.scenario_id`
15. Rows are written to `lte_prediction_optimised_results`.
16. Scenario status becomes `done` or `failed`.

## Current Production Flow: Recommendation Optimized

1. The route validates `project_id`, `radius`, and `grid_resolution`.
2. The service chooses `recommendation_scenario_id` from the request or latest `rf_optimization_results.scenario_id`.
3. A new row is created in `lte_optimization_scenarios` with:
   - `target_type = recommendation`
   - `target_id = rf_scenario_<recommendation_scenario_id>`
4. Recommendation rows are fetched from `rf_optimization_results`.
5. Only actionable supported parameters are kept:
   - `ETilt`
   - `Azimuth`
   - `TX Power`
   - `Mechanical Tilt`
   - `Height`
6. The recommendation values are applied to matching `site_prediction` rows in memory.
7. The applied site changes are saved into `site_prediction_optimized` using the public scenario id.
8. Baseline and optimized RF predictions are run.
9. RF deltas are applied so unchanged/affected populations can be compared consistently.
10. Optimized rows are saved to `lte_prediction_optimised_results`.
11. Scenario status is updated to `done`.

## Site Save Behavior

Recommendation optimized prediction saves site rows before writing optimized prediction rows.

Preferred path:

```text
POST /api/MapView/UpdateSitePrediction
```

Fallback path:

```text
direct DB insert/update into site_prediction_optimized
```

The log marker is:

```text
[LTE_OPT][SITE_SCENARIO_SAVE]
```

If this fails with zero rows affected, the job raises an error and should not continue to successful optimized DB write.

## Input Tables

| Table | Used for |
| --- | --- |
| `lte_prediction_baseline_results` | Baseline RF source for before/after comparison |
| `site_prediction` | Baseline site configuration |
| `site_prediction_optimized` | Saved frontend/manual or recommendation-applied site changes |
| `lte_prediction_geo_features` | Geo context used by optimized RF engine |
| `rf_optimization_results` | Source recommendations for recommendation optimized runs |
| `lte_optimization_scenarios` | Scenario metadata and lifecycle |

## Output Tables

| Table | Write behavior |
| --- | --- |
| `lte_optimization_scenarios` | Insert one row per optimized run; status updated by internal row id |
| `site_prediction_optimized` | Insert/update site overrides for recommendation optimized runs |
| `lte_prediction_optimised_results` | Append optimized RF prediction rows |

## Bridge Behavior

When the Python bridge is configured:

| Operation | Bridge endpoint |
| --- | --- |
| Latest baseline id | `GetLatestLteBaselineJobId` |
| Latest RF recommendation scenario | `GetLatestRfOptimizationScenarioId` |
| Fetch RF recommendation rows | `GetRfOptimizationRows` |
| Create optimization scenario | `CreateLteOptimizationScenario` |
| Update scenario status | `UpdateLteOptimizationScenarioStatus` |
| Save optimized rows | `SaveLtePredictionOptimisedResults` |
| Save recommendation site rows | `/api/MapView/UpdateSitePrediction` |

Without the bridge, SQLAlchemy direct DB access is used.

## Debug Logs

Important log markers:

| Marker | Meaning |
| --- | --- |
| `[LTE_OPT][SCENARIO_CREATE]` | Created optimization scenario; includes internal row id and public scenario id |
| `[LTE_OPT][SITE_SCENARIO_SAVE]` | Saved site changes to `site_prediction_optimized` |
| `[LTE_OPT][RECOMMENDATION_ROWS]` | Fetched RF recommendation rows |
| `[LTE_OPT][RECOMMENDATION_APPLIED]` | Recommendation rows matched and applied to site rows |
| `[LTE_OPT][RECOMMENDATION_OPTIMIZED_DB_PAYLOAD]` | Final optimized rows prepared for DB |
| `[LTE_OPT][DB_WRITE_DONE]` | Optimized rows written |
| `[LTE_OPT][SCENARIO_STATUS]` | Scenario marked `running`, `done`, or `failed` |

## Common Debug Checks

If optimized results are not visible in the expected view:

1. Check `lte_optimization_scenarios` first.
2. Use `lte_optimization_scenarios.id` to query `lte_prediction_optimised_results.scenario_id`.
3. Use `lte_optimization_scenarios.scenario_id` to query `site_prediction_optimized.scenario`.
4. Avoid querying optimized LTE rows with the RF recommendation scenario id; it belongs to `rf_optimization_results`, not `lte_prediction_optimised_results`.
5. If checking frontend-visible scenario numbers, use `public_scenario_id` or `site_prediction_optimized.scenario`.

## Production Notes

- Previous documentation described `lte_prediction_optimised_results.scenario_id` as the public scenario id. The current implementation uses a different mapping.
- The current implementation stores the internal optimized scenario row id in `lte_prediction_optimised_results.scenario_id`.
- The current implementation also stores the frontend-visible scenario id in `lte_prediction_optimised_results.public_scenario_id`.
- `site_prediction_optimized` in some live schemas has only `scenario`, not `scenario_id` or `public_scenario_id`.
- Treat `lte_optimization_scenarios` as the mapping table between frontend scenario and internal result rows.
