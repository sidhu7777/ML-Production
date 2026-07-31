# LTE Prediction Optimised

This module runs after-change LTE RF prediction. It does not simply rerun the whole baseline project. The production logic identifies changed cells, expands to affected same-site/neighbor cells, reuses baseline/geo context, predicts the affected population, and stores scenario-linked results.

There are two production paths:

1. Manual optimized prediction from saved frontend site edits.
2. Recommendation optimized prediction from saved RF tilt recommendation rows.

## Public API

Base prefix:

```text
/api/lte-prediction-optimised
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/run` | `POST` | Starts manual optimized prediction. |
| `/optimized` | `POST` | Alias for manual optimized prediction. |
| `/recommendation-optimized` | `POST` | Applies saved tilt recommendations and runs optimized prediction. |
| `/status/<job_id>` | `GET` | Returns in-memory job state. |
| `/download?file=<csv_path>` | `GET` | Downloads the generated CSV output. |

## Manual Optimized Request

Required fields:

| Field | Meaning |
| --- | --- |
| `project_id` | Project id. |
| `radius` | RF prediction radius. |
| `grid_resolution` | RF prediction grid resolution. |
| `operator` | Operator to run. |

Common optional fields:

| Field | Meaning |
| --- | --- |
| `region` / `country_code` / `countryCode` | Region selector; defaults to `india`. |
| `site_prediction_scenario_id` / `sitePredictionScenarioId` / `scenario` | Public frontend scenario id stored in `site_prediction_optimized.scenario`. |
| `scenario_id` plus `scenario_row_id` | Existing scenario mapping when a caller already created `lte_optimization_scenarios`. |
| `baseline_job_id` | Specific baseline job id. If missing, latest baseline is resolved. |
| `polygon_ids` / `polygonIds` | Optional site/polygon restriction. |
| `impact_radius_m` | Radius for affected-cell expansion. |
| `neighbor_site_count` | Neighbor sites included around changed cells. |
| `max_interference_sites` | Interferer cap during RF compute. |
| `max_neighbors_per_update_cell` | Limit for neighbor expansion per changed cell. |
| `target_type`, `target_id` | Optional scenario target metadata. |
| `scenario_name`, `scenario_description`, `created_by` | Scenario metadata. |

## Recommendation Optimized Request

Required fields:

| Field | Meaning |
| --- | --- |
| `project_id` | Project id. |
| `radius` | RF prediction radius. |
| `grid_resolution` | RF prediction grid resolution. |

Common optional fields:

| Field | Meaning |
| --- | --- |
| `operator` | Optional recommendation/operator filter. |
| `region` / `country_code` / `countryCode` | Region selector; defaults to `india`. |
| `recommendation_scenario_id` | RF recommendation scenario id from `rf_optimization_results.scenario_id`. If missing, latest is selected. |
| `baseline_job_id` | Specific baseline job id. If missing, latest baseline is resolved. |
| `impact_radius_m`, `neighbor_site_count`, `max_interference_sites` | Affected-area and RF compute controls. |
| `polygon_ids` / `polygonIds` | Optional site/polygon restriction. |

## Code Map

| File | Current role |
| --- | --- |
| `routes.py` | Validates request fields, resolves region/country aliases, exposes manual/recommendation/status/download endpoints. |
| `services.py` | Job lifecycle, scenario creation/pruning/status, recommendation fetch/apply, site scenario save, CSV and DB writes. |
| `ml_engine.py` | Baseline/site/optimized-site fetch, identity normalization, changed-cell detection, affected-cell expansion, K1/K2 calibration, optimized RF prediction. |
| `Sector_wise_prediction_code_copy.py` | RF prediction functions used by optimized RF compute. |

## Manual Scenario Flow

```text
frontend saves site edits into site_prediction_optimized
  -> POST /api/lte-prediction-optimised/optimized
  -> create/reuse lte_optimization_scenarios row
  -> fetch latest/specified baseline rows
  -> fetch base site_prediction rows
  -> fetch site_prediction_optimized rows for public scenario
  -> overlay optimized rows on base site rows
  -> detect changed cells
  -> expand changed cells to same-site and neighbor/affected cells
  -> compute K1/K2 calibration from baseline
  -> fetch matching geo features when available
  -> run optimized RF for affected cells
  -> replace affected baseline cells in output population
  -> save CSV under outputs/
  -> append lte_prediction_optimised_results
  -> mark lte_optimization_scenarios status done/failed
```

## Recommendation Optimized Flow

```text
rf_optimization_results contains accepted tilt recommendation rows
  -> POST /api/lte-prediction-optimised/recommendation-optimized
  -> resolve latest/requested RF recommendation scenario id
  -> create lte_optimization_scenarios row with target_id=rf_scenario_<id>
  -> fetch recommendation rows
  -> keep actionable supported parameter changes
  -> fetch baseline and base site_prediction rows
  -> apply recommendation values to matching site rows in memory
  -> save applied site rows into site_prediction_optimized using public scenario id
  -> run optimized RF over affected cells
  -> apply RF delta for before/after comparison consistency
  -> save CSV under outputs/
  -> append lte_prediction_optimised_results
  -> mark scenario done/failed
```

## Scenario ID Rules

This module uses both internal and public optimized scenario ids. Keep them separate.

| Table | Column | Meaning |
| --- | --- | --- |
| `lte_optimization_scenarios` | `id` | Internal scenario row id. |
| `lte_optimization_scenarios` | `scenario_id` | Public frontend scenario id, normally slot `1..6`. |
| `lte_prediction_optimised_results` | `scenario_id` | Internal scenario row id. |
| `lte_prediction_optimised_results` | `public_scenario_id` | Public frontend scenario id. |
| `site_prediction_optimized` | `scenario` | Public frontend scenario id. |
| `rf_optimization_results` | `scenario_id` | RF recommendation scenario id, not an optimized LTE scenario id. |

The service prunes the oldest optimized scenario when a project already has the configured maximum number of optimized scenarios. Direct DB mode uses slots `1..6` for public scenario ids.

## Identity And Matching

The optimized flow relies on robust identity matching because live data can contain multiple historical cell-id formats.

Current identity handling includes:

- `rf_identity_key`
- `site_sector_band_key`
- `sector_identity_key`
- `Node_Cell_ID`
- `legacy_nodeb_id_cell_id`
- `frontend_site_sector_key`
- `nodeb_id_cell_id`
- `canonical_cell_id`
- `cell_id`
- `local_cell_id`

The code normalizes pipe and underscore formats, strips decimal suffixes, builds site/cell/sector/band identities, and falls back through aliases before deciding that a recommendation or optimized row cannot be matched.

## Affected-Cell Recompute Logic

Optimized prediction recomputes a scoped population:

| Scope source | Meaning |
| --- | --- |
| Changed cells | Cells whose lat/lon/azimuth/tilt/power/height changed. |
| Same-site cells | Other cells on the same affected site. |
| Topology neighbors | Neighbor/interferer cells from baseline topology columns when available. |
| Distance neighbors | Nearby cells around both new and original locations when topology is missing or incomplete. |
| Explicit recompute cells | Optional override from request/config. |

The optimized RF engine prefers baseline prediction points for affected cells. If no points are available and strict point mode is not enabled, it generates a local grid. When geo features exist, generated grids can be masked to the baseline geo-feature footprint and then corrected using saved geo context.

## Supported Recommendation Parameters

The recommendation-optimized path only applies changed rows for supported parameters:

| Recommendation parameter | Site column applied |
| --- | --- |
| `ETilt`, `E Tilt`, `Electrical Tilt` | `electrical_tilt` |
| `Azimuth` | `azimuth` |
| `TX Power`, `Power` | `tx_power` |
| `Mechanical Tilt`, `MTilt` | `mechanical_tilt` |
| `Height`, `Antenna Height` | `antenna_height` |

Rows with unchanged values, unsupported parameters, invalid numeric recommendations, or no matching site row are skipped. If no recommendation row can be applied, the job fails rather than writing misleading optimized RF results.

## Site Scenario Save Behavior

Recommendation optimized prediction saves applied site changes before writing optimized RF rows.

Preferred path:

```text
POST /api/MapView/UpdateSitePrediction
```

Direct DB fallback:

```text
site_prediction_optimized
```

The direct fallback creates `site_prediction_optimized LIKE site_prediction` when necessary and ensures scenario/status/version helper columns exist. If both API and DB fallback produce zero affected rows, the job raises an error.

## Output Tables

| Table | Write behavior |
| --- | --- |
| `lte_optimization_scenarios` | One row per optimized run; status updated by internal row id. |
| `site_prediction_optimized` | Manual source rows or recommendation-applied site overrides keyed by public scenario. |
| `lte_prediction_optimised_results` | Appended optimized RF rows, including internal and public scenario ids. |

Important optimized result columns:

| Column | Meaning |
| --- | --- |
| `project_id`, `job_id` | Project/job identity. |
| `lat`, `lon` | Prediction point. |
| `pred_rsrp`, `pred_rsrq`, `pred_sinr` | Optimized RF metrics. |
| `node_b_id`, `cell_id`, `nodeb_id_cell_id` | Cell identities. |
| `Operator`, `Technology`, `site_id` | Display/filter fields. |
| `scenario_id` | Internal `lte_optimization_scenarios.id`. |
| `public_scenario_id` | Frontend-visible `lte_optimization_scenarios.scenario_id`. |

## Bridge Endpoints Used

| Operation | Bridge or API endpoint |
| --- | --- |
| Latest baseline id | `GetLatestLteBaselineJobId` |
| Fetch baseline rows | `GetLteBaselineRows` |
| Fetch site rows | `GetLteSitePredictionRows` |
| Fetch optimized site rows | `GetSitePredictionOptimized` |
| Fetch geo features | `GetLtePredictionGeoFeatures` |
| Latest RF recommendation scenario | `GetLatestRfOptimizationScenarioId` |
| Fetch RF recommendation rows | `GetRfOptimizationRows` |
| Create optimized scenario | `CreateLteOptimizationScenario` |
| Update optimized scenario status | `UpdateLteOptimizationScenarioStatus` |
| Save optimized rows | `SaveLtePredictionOptimisedResults` |
| Save recommendation site rows | `/api/MapView/UpdateSitePrediction` |

Without bridge mode, SQLAlchemy direct DB access is used.

## Debug Markers

| Marker | Meaning |
| --- | --- |
| `[LTE_OPT][SCENARIO_CREATE]` | Internal row id and public scenario id were created/resolved. |
| `[LTE_OPT][SCENARIO_PRUNE]` | Old optimized scenario was removed to free a public slot. |
| `[LTE_OPT][BASELINE_FETCH]` | Baseline rows loaded. |
| `[LTE_OPT][OPT_SITE_FETCH]` | Optimized site rows loaded. |
| `[LTE_OPT][AFFECTED_CELL_SCOPE]`, `[LTE_OPT][AFFECTED]` | Changed/neighbor/affected cell scope. |
| `[LTE_OPT][POINTS_OVERRIDE]` | Baseline prediction points are being reused. |
| `[LTE_OPT][GEO_FETCH]` | Saved geo features are available or skipped. |
| `[LTE_OPT][RECOMMENDATION_ROWS]` | RF recommendation rows loaded. |
| `[LTE_OPT][RECOMMENDATION_APPLIED]` | Recommendation rows applied to site rows. |
| `[LTE_OPT][SITE_SCENARIO_SAVE]` | Applied site changes saved to scenario rows. |
| `[LTE_OPT][DB_WRITE]`, `[LTE_OPT][DB_WRITE_DONE]` | Optimized rows persisted. |
| `[LTE_OPT][SCENARIO_STATUS]` | Scenario lifecycle update. |

## Common Failure Checks

1. Do not query `lte_prediction_optimised_results.scenario_id` with an RF recommendation scenario id.
2. Use `lte_optimization_scenarios.id` for optimized result rows.
3. Use `lte_optimization_scenarios.scenario_id` or `site_prediction_optimized.scenario` for frontend scenario selection.
4. Check that a baseline run exists for the project/operator.
5. Check that `site_prediction_optimized` contains rows for the requested public scenario.
6. Check identity fields when recommendation rows cannot match site rows.
7. Check bridge/API access for `/api/MapView/UpdateSitePrediction` if recommendation scenario saving fails.
