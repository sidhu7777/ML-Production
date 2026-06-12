# S-Tracer ML Backend

This directory contains the Python machine-learning and RF analytics services used by S-Tracer for LTE baseline prediction, optimized LTE prediction, RF tilt recommendation, and report generation.

This document provides a concise system overview and navigation index. Detailed production behavior, code architecture, database interactions, and operational guidance are documented in the README file within each tool directory.

## Tool Documentation

| Tool | Purpose | Detailed documentation |
| --- | --- | --- |
| LTE Prediction | Builds baseline LTE RF prediction and geo feature outputs from drive test sessions and site data. | [tools/lte_prediction/README.md](tools/lte_prediction/README.md) |
| LTE Prediction Optimised | Runs optimized LTE prediction from saved site edits or accepted RF recommendation scenarios. | [tools/lte_prediction_optimised/README.md](tools/lte_prediction_optimised/README.md) |
| LTE Tilt Recommendation | Evaluates bad RF areas and writes validated RF optimization recommendations. | [tools/lte_tilt_recommandation/README.md](tools/lte_tilt_recommandation/README.md) |
| Report Engine | Generates project PDF reports and exposes report status/download APIs. | [tools/report_engine/README.md](tools/report_engine/README.md) |



| Module | Endpoint |
| --- | --- |
| LTE Prediction | `POST /api/lte-prediction/run` |
| LTE Prediction Optimised | `POST /api/lte-prediction-optimised/run` or `POST /api/lte-prediction-optimised/optimized` |
| Recommendation Optimized Prediction | `POST /api/lte-prediction-optimised/recommendation-optimized` |
| LTE Tilt Recommendation | `POST /api/lte-tilt-recommandation/optimize` |
| Report Engine | `POST /api/report/generate` |

All long-running tools use in-memory background job state and expose status endpoints. The .NET backend can also be used through the Python bridge when `PYTHON_BRIDGE_BASE_URL` or `SIGNAL_TRACKERS_BRIDGE_URL` is configured.

## Current Production Flow

1. LTE baseline prediction reads `site_prediction` and drive-test rows, then saves:
   - `lte_prediction_baseline_results`
   - `lte_prediction_geo_features`
2. LTE tilt recommendation reads the latest baseline output plus site data and grid analytics, then saves accepted recommendation rows into:
   - `rf_optimization_results`
3. Recommendation optimized prediction reads `rf_optimization_results`, applies those recommendations to site rows, saves the scenario into:
   - `site_prediction_optimized`
   - `lte_optimization_scenarios`
   - `lte_prediction_optimised_results`
4. Frontend scenario selection should use the public scenario id from `lte_optimization_scenarios.scenario_id`. Internal optimized result rows use `lte_optimization_scenarios.id`.

## Important Scenario ID Rule

Different tables use different scenario identifiers. This is intentional in the current code, but it must be handled carefully.

| Table | Scenario meaning |
| --- | --- |
| `rf_optimization_results.scenario_id` | RF recommendation scenario id |
| `lte_optimization_scenarios.id` | Internal optimization scenario row id |
| `lte_optimization_scenarios.scenario_id` | Public frontend scenario id |
| `lte_prediction_optimised_results.scenario_id` | Internal optimization scenario row id |
| `lte_prediction_optimised_results.public_scenario_id` | Public frontend scenario id |
| `site_prediction_optimized.scenario` | Public frontend scenario id |

Example from a real run: RF recommendation scenario `19` created optimization scenario row `69` with public scenario `1`. Optimized result rows were saved with `scenario_id = 69` and `public_scenario_id = 1`; site rows were saved with `site_prediction_optimized.scenario = 1`.

## Local Configuration

Common environment variables:

| Variable | Used for |
| --- | --- |
| `DATABASE_URL` | India/default ML database connection |
| `DATABASE_URL_Taiwan` | Taiwan database connection |
| `PYTHON_BRIDGE_BASE_URL` or `SIGNAL_TRACKERS_BRIDGE_URL` | .NET PythonBridge API base URL |
| `PYTHON_BRIDGE_API_KEY` | Optional bridge authentication key |
| `PYTHON_BRIDGE_TIMEOUT_SECONDS` | Bridge request timeout |

## Reading Order

Recommended onboarding sequence:

1. This file.
2. [LTE Prediction](tools/lte_prediction/README.md).
3. [LTE Tilt Recommendation](tools/lte_tilt_recommandation/README.md).
4. [LTE Prediction Optimised](tools/lte_prediction_optimised/README.md).
5. [Report Engine](tools/report_engine/README.md), if working on reports.
