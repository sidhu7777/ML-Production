# LTE Tilt Recommendation Debug

This folder is for diagnosing why the frontend/Electron run gives empty LTE tilt recommendations while a local PowerShell/API run gives recommendations.

The debug script does not save recommendations to the database. It fetches inputs, runs the production recommendation engine, saves diagnostic CSVs, and compares direct DB vs Python bridge.

## Run Direct DB Vs Bridge

From `ML`:

```powershell
venv\Scripts\python.exe tests\debug\lte_tilt_recommendation_debug.py --source both
```

If your terminal does not already have DB/bridge environment variables, pass them explicitly:

```powershell
venv\Scripts\python.exe tests\debug\lte_tilt_recommendation_debug.py `
  --source both `
  --database-url "<your DATABASE_URL>" `
  --bridge-base-url "http://localhost:5224" `
  --bridge-api-key "<your PYTHON_BRIDGE_API_KEY>"
```

For Electron-style bridge debugging, start the C# backend first, then run with `--bridge-base-url http://localhost:5224`.

This uses the same default payload as the known local test:

```text
project_id=196
region=india
operator=Airtel
rsrp=-90
rsrq=-14
sinr=0
rsrp_weight=20
rsrq_weight=20
sinr_weight=60
radius_m=500
grid_resolution_m=25
n_workers=3
neighbor_site_count=2
max_interference_sites=10
candidate_workers=2
coordinate_passes=2
bad_grid_coverage_pct=60
max_group_cells=0
max_neighbors_per_update_cell=2
```

## Only Check Fetch/Input Data

Use this first if full recommendation takes too long:

```powershell
venv\Scripts\python.exe tests\debug\lte_tilt_recommendation_debug.py --source both --skip-engine
```

## Test Exact Frontend Payload

Copy the browser console object printed by:

```text
[LTE_TILT_RECOMMENDATION] POST /api/lte-tilt-recommandation/optimize
```

Save it as JSON, for example:

```text
ML\outputs\debug\frontend_payload.json
```

Then run:

```powershell
venv\Scripts\python.exe tests\debug\lte_tilt_recommendation_debug.py --source both --payload-json outputs\debug\frontend_payload.json
```

## What To Inspect

Each run writes a folder like:

```text
ML\outputs\debug\lte_tilt_recommendation_YYYYMMDD_HHMMSS
```

Important files:

```text
effective_payload.json
comparison.json
summary.json
direct\summary.json
bridge\summary.json
direct\candidate_evaluations.csv
bridge\candidate_evaluations.csv
direct\recommendations.csv
bridge\recommendations.csv
direct\recommendations_after_constraints.csv
bridge\recommendations_after_constraints.csv
```

Read `comparison.json` first. The first mismatch usually tells where the issue starts:

```text
baseline_job_id mismatch -> different LTE prediction result set
antenna_rows/fingerprint mismatch -> bridge site data differs from direct DB
baseline_rows/fingerprint mismatch -> bridge baseline data differs from direct DB
grid_rows/fingerprint mismatch -> frontend grid/bridge grid differs
raw_recommendations > 0 but after_constraints = 0 -> threshold file constraints removed all recommendations
candidate_evaluations = 0 -> target/candidate selection rejected everything before final recommendation
```
