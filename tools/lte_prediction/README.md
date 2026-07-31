# LTE Prediction

This module builds the production baseline LTE RF prediction surface for a project. Its output is the reference layer used by tilt recommendation, optimized prediction, and the coverage Model 1-4 workflows.

## Public API

Base prefix:

```text
/api/lte-prediction
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/run` | `POST` | Starts a baseline LTE prediction job. |
| `/status/<job_id>` | `GET` | Returns in-memory job state. |
| `/result/<job_id>` | `GET` | Returns the same job payload for polling. |

Required request fields:

| Field | Meaning |
| --- | --- |
| `project_id` | Project id. |
| `session_ids` | Drive-test session ids used for calibration/input context. |

Common optional fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `region` / `country_code` / `countryCode` | `india` | Supports aliases such as `in`, `ind`, `tw`, `twn`. |
| `operator` | Auto/blank | Operator filter for site and drive rows. |
| `radius` / `radius_m` | `500` | Prediction radius in meters. |
| `grid_resolution` | `25` | Prediction grid resolution in meters. |
| `n_workers` | CPU count minus one | RF compute workers. |
| `max_interference_sites` | `10` | Interference-site cap during RF computation. |
| `use_frontend_grid_sampling` | `true` | Uses frontend grid analytics as the prediction surface when available. |
| `grid_analytics_scenario_id` | Latest/none | Frontend grid analytics scenario to reuse. |
| `samples_per_grid_axis` | `3` | Sample density inside each frontend grid cell. |
| `max_cells_per_grid` | `3` | Maximum relevant serving cells per frontend grid. |
| `min_cells_per_grid` | `1` | Minimum relevant cells per grid when possible. |
| `ensure_all_cells` | `true` | Ensures every site cell gets at least some samples. |
| `min_grids_per_cell` | `1` | Minimum grids sampled per cell. |
| `min_candidate_rsrp_dbm` | `-128` | RSRP viability threshold for candidate grid/cell pairing. |
| `candidate_safety_cap` | `20` | Hard cap for candidate cells per grid. |
| `drive_rows` / `network_logs` | none | Frontend-supplied drive rows instead of fetching from backend/DB. |
| `dem_raster_path` | auto/none | Optional DEM path; service also tries automatic project DEM resolution. |

## Code Map

| File | Current role |
| --- | --- |
| `routes.py` | Builds request config, resolves region/country aliases, starts jobs. |
| `services.py` | Job lifecycle, DEM resolution, baseline/geo DB formatting, delta/upsert writes, bridge writes. |
| `ml_engine.py` | Site/drive/building/polygon fetch, frontend grid sampling, RF engine call, post-processing entry. |
| `geo_correction_pipeline.py` | Identity normalization, project/building alignment, advanced geo/terrain features, DT calibration, display smoothing/correction. |
| `grid_sampling.py` | Frontend grid analytics loading and grid sample assignment to relevant cells. |
| `dem_utils.py` | Project DEM lookup/download/resolve helper. |
| `Sector_wise_prediction_code_copy.py` | RF propagation/prediction implementation called by `ml_engine.py`. |

## Production Flow

```text
POST /api/lte-prediction/run
  -> routes.py resolves region and builds cfg
  -> LTEPredictionService.submit()
  -> background thread enters app context
  -> fetch_site_data()
  -> ensure_project_dem()
  -> fetch_drive_data()
  -> fetch_building_data()
  -> run_rf_prediction_fast()
  -> run_ml_fast()
  -> apply_full_display_correction()
  -> _save_geo_features()
  -> _save_baseline_results()
  -> temp/final_<job_id>.csv
  -> job status done/failed
```

## Input Loading

### Site Rows

`fetch_site_data()` reads LTE site rows from PythonBridge endpoint `GetLteSitePredictionRows` when bridge mode is enabled. Without bridge mode it reads `site_prediction` directly.

Important behavior:

- Region parameters include country code (`IN` or `TW`) where possible.
- `polygon_ids` can restrict site rows to selected project regions.
- Site rows must have complete identity fields: site, cell, sector, band, and operator.
- The code normalizes aliases such as latitude/longitude, e_tilt/m_tilt/height, cluster/network/operator, PCI, tx power, band, EARFCN, and reference signal power.
- If `site_prediction_key` or `site_cell_sector_band_operator_key` exists, that strict identity is used as `Node_Cell_ID` and `rf_identity_key`.
- Operator filters are applied after normalization.

### Drive-Test Rows

`fetch_drive_data()` uses this order:

1. Frontend-supplied `drive_rows` / `network_logs`, if they contain usable `lat`, `lon`, `rsrp`, `rsrq`, and `sinr`.
2. Cached parquet under `cache/drive_<project>_<region>_<operator>_<hash>.parquet`, if valid.
3. PythonBridge `GetDriveTestRows`, with primary-only rows first and fallback to all rows/all operators when needed.
4. Direct DB reads from `tbl_network_log` and `tbl_network_log_neighbour`.

Drive rows are normalized to common lat/lon/KPI/cell/nodeb/PCI/EARFCN columns and filtered to project polygons. If polygon filtering returns no rows, the code retries with swapped polygon coordinates.

### Building, Polygon, Grid, And DEM Context

- Buildings come from PythonBridge `GetLteBuildingRows` or direct DB table `tbl_savepolygon`.
- Project polygons come from PythonBridge `GetProjectRegions` or direct `map_regions`.
- Frontend grid analytics comes from `/api/GridAnalytics/GetGridAnalytics` through the bridge, or direct `grid_analytics_results`.
- DEM is resolved by `ensure_project_dem()` unless an explicit `dem_raster_path` is supplied.

## RF Prediction And Geo Correction

`run_rf_prediction_fast()` prepares site/building CSV input, aligns project/building polygons, filters serving sites to the project polygon, and then calls the RF propagation engine.

Prediction points are selected in this order:

1. Frontend grid analytics samples, if enabled and available.
2. Circular/generated cell grids if frontend grid sampling is disabled or unavailable.

`run_ml_fast()` requires prediction rows, drive rows, site rows, building rows, and params. It calls `apply_full_display_correction()`, which adds production geo/display correction.

Current correction inputs include:

| Feature group | Examples |
| --- | --- |
| Building/LOS | building count, area ratio, blocker count, LOS blocked ratio, NLOS flag |
| Terrain | elevation, slope, site-relative relief |
| Morphology | clutter class, morphology cluster, green/water/road ratios |
| Site context | serving distance, nearest site distance, site density, azimuth delta |
| DT calibration | holdout calibration, drive-test replacement/blending, calibrated KPI preservation |
| Display output | smoothed/demo overlay KPI columns for display/audit |

The baseline metric columns store calibrated pre-smoothing values. Smoothed/display values are saved separately as `pred_rsrp_smoothed`, `pred_rsrq_smoothed`, and `pred_sinr_smoothed`.

## Output Tables

### `lte_prediction_baseline_results`

This is the main baseline RF table.

Important current columns:

| Column | Meaning |
| --- | --- |
| `project_id` | Project id. |
| `job_id` | Baseline job id generated by this run. |
| `lat`, `lon`, `lat_6dp`, `lon_6dp` | Prediction point and rounded key coordinates. |
| `pred_rsrp`, `pred_rsrq`, `pred_sinr` | Calibrated baseline KPI values. |
| `pred_rsrp_smoothed`, `pred_rsrq_smoothed`, `pred_sinr_smoothed` | Display/smoothed KPI values. |
| `node_b_id`, `cell_id`, `site_id` | Display/query identities. |
| `nodeb_id_cell_id` | Current RF identity key used by downstream modules. |
| `legacy_nodeb_id_cell_id` | Older node/cell identity fallback. |
| `sector`, `band` | Sector and band identity fields. |
| `rf_identity_key`, `sector_identity_key`, `site_sector_band_key` | Strict/current identity keys. |
| `operator`, `Technology` | Operator and RAT. |

Write behavior:

- Direct DB mode creates missing smoothed/identity columns where needed.
- Direct DB mode computes a baseline delta and upserts only changed/new rows.
- PythonBridge mode calls `SaveLtePredictionBaselineResults`.
- Bridge mode may set `replace_existing=true` when the project already has rows but no key overlap with the new result.

### `lte_prediction_geo_features`

This table stores environmental and geometric context for each baseline run.

Important current columns:

| Column | Meaning |
| --- | --- |
| `project_id`, `baseline_job_id`, `region`, `operator` | Scope and run id. |
| `grid_id` | Frontend/generated grid id where available. |
| `lat`, `lon`, `nodeb_id_cell_id` | Geo-feature key. |
| `proxy_site_id`, `clutter_class`, `morphology_cluster` | Site/environment labels. |
| `building_count`, `building_area_ratio`, `avg_building_area_m2` | Building density. |
| `road_length_m`, `green_ratio`, `water_ratio` | Land-use context. |
| `los_blocker_count`, `los_blocked_ratio`, `max_blocker_height_m`, `diffraction_proxy_db`, `nlos_flag` | Blockage/NLOS context. |
| `terrain_elevation_m`, `terrain_slope_deg`, `proxy_site_elevation_m`, `terrain_relief_to_site_m` | DEM context. |
| `site_count_250m`, `site_count_500m`, `serving_distance_m`, `nearest_site_distance_m`, `mean_nearest3_site_distance_m` | Site density/distance context. |
| `azimuth_delta_deg`, `polygon_alignment`, `building_alignment`, `geo_source` | Alignment and source metadata. |

Write behavior:

- Direct DB mode uses a delta/upsert pattern keyed by project, baseline job, region, cell identity, lat, and lon.
- PythonBridge mode calls `SaveLtePredictionGeoFeatures` with `replace_existing=true` for the baseline job.

## Bridge Endpoints Used

| Operation | Bridge endpoint |
| --- | --- |
| Project polygons | `GetProjectRegions` |
| Site rows | `GetLteSitePredictionRows` |
| Drive rows | `GetDriveTestRows` |
| Building rows | `GetLteBuildingRows` |
| Frontend grid analytics | `/api/GridAnalytics/GetGridAnalytics` |
| Save baseline rows | `SaveLtePredictionBaselineResults` |
| Save geo feature rows | `SaveLtePredictionGeoFeatures` |

## Downstream Dependencies

Tilt recommendation, optimized prediction, and coverage models depend on the latest successful baseline run.

| Downstream module | What it needs from baseline |
| --- | --- |
| Tilt recommendation | Latest baseline rows, topology/interference columns when present, and geo features. |
| Optimized prediction | Baseline rows, prediction-point population, K1/K2 calibration source, and geo features. |
| Coverage Model 1/2 | Baseline and geo rows for feature frames. |
| Coverage Model 3/4 | Latest baseline id and RF after-surface comparison context. |

## Debug Markers

Useful log markers:

| Marker | Meaning |
| --- | --- |
| `[LTE][JOB_START]` | Resolved request/job config. |
| `[LTE][SITE_FETCH_RAW]`, `[LTE][SITE_FETCH_READY]` | Site fetch and normalized site input. |
| `[LTE][DRIVE_FETCH_*]` | Drive source, cache, bridge, primary/all fallback, and polygon filtering. |
| `[LTE][FRONTEND_GRID_FETCH]`, `[LTE][RF_SAMPLE_SOURCE]` | Frontend grid sampling behavior. |
| `[LTE][DEM]` | DEM auto-resolution status. |
| `[LTE][POST_OUTPUT]` | Geo/display correction summary. |
| `[LTE][BASELINE_DB_WRITE]`, `[LTE][BASELINE_DB_WRITE_DONE]` | Baseline persistence. |
| `[LTE][GEO_DB_WRITE]`, `[LTE][GEO_DB_WRITE_DONE]` | Geo-feature persistence. |

## Common Failure Checks

1. Confirm `site_prediction` rows have complete site/cell/sector/band/operator identity.
2. Confirm `session_ids` resolve to usable drive rows with lat/lon and KPI values.
3. Check operator mismatch between site rows and drive rows.
4. Check polygon filters and swapped-coordinate fallback logs.
5. Check frontend grid analytics if prediction points look sparse or shifted.
6. Check DEM path or auto-resolution if terrain features are missing.
7. Check bridge URL/API key/timeouts when production uses PythonBridge.
