# LTE Prediction Offset

This module is the production physical/offset-calibrated LTE/5G prediction path. It starts from the normal LTE site, drive-test, building, polygon, grid, and DEM sources, then applies the newer physical correction stack before saving baseline-style output rows.

## Public API

Base prefix:

```text
/api/lte-prediction-offset
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/run` | `POST` | Starts an offset prediction job. |
| `/status/<job_id>` | `GET` | Returns in-memory job state. |
| `/result/<job_id>` | `GET` | Returns the same job payload for polling. |

Required request fields:

| Field | Meaning |
| --- | --- |
| `project_id` | Project id. |
| `session_ids` | Drive-test session ids, unless `drive_rows` / `network_logs` are supplied. |

Common optional fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `region` / `country_code` / `countryCode` | `india` | Region selector; supports India/Taiwan aliases. |
| `operator` | blank | Operator filter. |
| `radius` / `radius_m` | `500` | Flat prediction radius in meters. |
| `prediction_scope` / `predictionScope` | `radius` | Use `hcell`/`cell` to solve per-cell coverage radius from link budget. |
| `cell_edge_rsrp_dbm` / `cellEdgeRsrpDbm` | `-110` for H Cell | Service-edge RSRP used for per-cell radius. |
| `cell_edge_radius_cap_m` / `cellEdgeRadiusCapM` | none | Optional cap for solved per-cell radius. |
| `grid_resolution` | `25` | Prediction grid resolution in meters. |
| `building` | `true` | Enables building/geospatial context. |
| `dem_raster_path` / `demRasterPath` | auto/none | Optional project DEM raster. |
| `ghs_obat_csv_path` / `ghsObatCsvPath` | none | Optional GHS-OBAT building-height extract. |
| `drive_rows` / `network_logs` | none | Frontend-supplied drive rows. |
| `grid_analytics_scenario_id` | latest/none | Frontend grid analytics scenario. |
| `n_workers` | CPU count minus one | Worker count. |

## Code Map

| File | Current role |
| --- | --- |
| `routes.py` | Request parsing, region and prediction-scope handling, job submission. |
| `services.py` | Job lifecycle, site/drive/grid/building/DEM loading, raw surface generation, calibration, output persistence. |
| `geo_inputs.py` | Phase 27 clutter/building/land-use loading and cache support. |
| `phase27_physical.py` | Terrain, building, obstruction, and physical scoring features. |
| `phase27_calibration.py` | Feature assembly and older outdoor calibration helpers. |
| `phase36_physical_upgrades.py` | Per-RE RSRP reference correction, antenna pattern delta, and physical upgrades. |
| `phase37_quality.py` | RSRQ/SINR quality estimation. |
| `phase48_calibration.py` | Technology/band residual calibration from drive-test rows. |
| `antenna_patterns/` | Bundled `.pap` antenna pattern files used by Phase 36 where model/frequency/tilt match. |

## Current Flow

```text
POST /api/lte-prediction-offset/run
  -> fetch site rows
  -> prepare strict cell identity, technology, band, frequency, tilt, height, power
  -> resolve project DEM
  -> fetch drive-test rows
  -> fetch building geometry
  -> fetch frontend/grid pixels or build polygon grid
  -> generate directional COST-231 candidate surface
  -> classify clutter and score terrain/building context
  -> match DT rows by measured technology and measured band
  -> apply Phase 36 physical corrections
  -> fit/apply Phase 48 residual calibration
  -> estimate RSRQ/SINR where possible
  -> save baseline-style rows and geo features
```

## Site Data Requirements

The RF formula path uses the following site inputs:

| Field | Meaning |
| --- | --- |
| `lat`, `lon` | Site coordinates. |
| `nodeb_id`, `cell_id` | Client-supplied node/cell identity; derived strict identity fields are built by code. |
| `site` / `Site ID` | Site id. |
| `sector` | Sector label/id. |
| `Technology` / `technology` | `4G`, `LTE`, `5G`, or `NR`. |
| `band` | Deployed band, normalized as `B<number>` for LTE or `n<number>` for NR. |
| `azimuth` | Antenna azimuth in degrees. |
| `Etilt`, `Mtilt` | Electrical and mechanical tilt. |
| `Height` | Antenna height in meters. |
| `tx_power` | Transmit power in dBm. |
| `frequency_mhz` / `frequency` / `downlink_frequency` | Deployed carrier frequency in MHz. |
| `antenna_model` / `antenna_type` / `antenna` / `antenna_name` | Optional antenna pattern lookup key. |

Do not require clients to provide `rf_identity_key`, `legacy_nodeb_id_cell_id`, `Node_Cell_ID`, or `nodeb_id_cell_id` when `nodeb_id` and `cell_id` are available. Those identities are derived by the code.

## Drive-Test Calibration Rules

Drive-test rows must contain:

| Field | Meaning |
| --- | --- |
| `lat`, `lon` | Measurement point. |
| `rsrp` / `rssi` / `reference_signal_received_power` | Measured serving signal. |
| `technology` / `network` / `Technology` | Measured RAT. |
| `band` or `earfcn` | Measured carrier band. LTE band can be inferred from EARFCN; NR should provide a specific `n<number>` band when more than one NR band is deployed. |

Phase 48 only calibrates rows where measured technology and measured band match deployed site rows. Unmatched DT rows are dropped from calibration. If no matching DT exists, output rows remain physical predictions with calibration status `UNCALIBRATED_NO_DT`.

## Antenna Patterns

The supported antenna pattern file format is `.pap`. The bundled patterns currently live under:

```text
tools/lte_prediction_offset/antenna_patterns/
```

When a matching pattern is unavailable, the physical path falls back to the generic antenna behavior instead of failing.

## Outputs

This service writes through the same baseline persistence path as `tools/lte_prediction`:

```text
lte_prediction_baseline_results
lte_prediction_geo_features
```

It also writes a job CSV under the configured output folder.

## Debug Markers

Useful log markers:

| Marker | Meaning |
| --- | --- |
| `[LTE_OFFSET][INPUT]` | Prepared site/grid/DT counts. |
| `[LTE_OFFSET][CELL_EDGE_RADIUS]` | H Cell per-cell radius summary. |
| `[LTE_OFFSET][COST231_DIRECTIONAL]` | Raw candidate generation progress. |
| `[LTE_OFFSET][DT_BAND_MATCH]` | DT rows matched/unmatched by technology/band. |
| `[LTE_OFFSET][PHASE36_ANTENNA_PATTERN]` | Vendor PAP pattern or generic fallback usage. |
| `[LTE_OFFSET][PHASE48_CALIBRATION_MATCH]` | Rows used for Phase 48 calibration. |
| `[LTE_OFFSET][CALIBRATION_STATUS]` | Final calibrated vs uncalibrated row counts. |
