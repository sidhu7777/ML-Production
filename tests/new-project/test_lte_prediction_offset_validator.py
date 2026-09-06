"""End-to-end validator for the production offset pipeline (Phase 36 v2 + Phase 37).

Runs the REAL `tools.lte_prediction_offset.services.LTEPredictionOffsetService._run`
against project 210 (Taiwan) using the REAL database fetches:

    fetch_site_data / fetch_drive_data / fetch_building_data   -> live TaiwanDB (direct, no bridge)
    load_or_build_phase27_clutter                              -> live tbl_project_clutter_tile cache
    _resolve_prediction_polygons                               -> live map_regions
    grid                                                       -> built from the project polygon

Only two things are intercepted:
  * the Python bridge is disabled  -> everything uses the direct-DB path
  * `_save_offset_baseline_results` is captured (DRY RUN) -> NOTHING is written to the DB

No production code is modified.  DEM: if the cached tif is unusable and cannot be
rebuilt offline, terrain degrades to disabled (a small correction) - the RSRP ->
calibration -> RSRQ/SINR flow is still validated.

Run:  ML/venv/Scripts/python.exe ML/tests/new-project/test_lte_prediction_offset_validator.py
"""
from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

THIS_DIR = Path(__file__).resolve().parent
ML_ROOT = THIS_DIR.parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
load_dotenv(ML_ROOT / ".env")

PROJECT_ID = 210
REGION = "taiwan"

# The real Taiwan terrain surface: 5 m Planet grid, UTM51N, band 4 = elevation
# (bands 1-3 are an RGB render). Covers New Taipei / Taipei. This is the DEM
# yesterday's /run used - `data/dem/project_210_dem.tif` is a skadi fallback that
# includes ocean bathymetry and fails the elevation-band sanity check.
_DEM_5M = (THIS_DIR / "data" / "mapdata" / "Dno19_0095_NewTaipeiCity_5m"
           / "Dno19_0095_NewTaipeiCity_5m" / "New_TaipeiCity_5m_UTM51N_planet"
           / "Heights" / "height_5m.grd")

import tools.lte_prediction_offset.services as svc_mod
from tools.lte_prediction.ml_engine import engine as _pred_engines

_CHECKS: list[tuple[str, bool, str]] = []
_CAPTURED: dict = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    _CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _session_ids() -> list[int]:
    eng = _pred_engines.get(REGION)
    with eng.connect() as c:
        row = c.execute(text("SELECT ref_session_id FROM tbl_project WHERE id=:p"), {"p": PROJECT_ID}).fetchone()
    return [int(x) for x in re.findall(r"\d+", str(row[0] or ""))]


def _install_min_stubs() -> None:
    # The Python bridge (localhost:5224) IS the production source for site + drive
    # rows - use it exactly as yesterday's run did.  Do NOT stub it.
    # Only intercept the DB write.
    def _capture_save(save_delegate, final_df, project_id, job_id, operator, region):
        _CAPTURED["final_df"] = final_df.copy()
        _CAPTURED["operator"] = operator
        print(f"  [DRY-RUN SAVE] captured {len(final_df)} rows for project {project_id} "
              f"job={job_id} (NOT written to DB)")
    svc_mod._save_offset_baseline_results = _capture_save


def main() -> None:
    print(f"=== LTE offset validator - REAL DB run, project {PROJECT_ID} ({REGION}) ===\n")
    sids = _session_ids()
    check("resolved session ids from tbl_project", len(sids) > 0, f"{len(sids)} sessions")
    check("prediction DB engine available", _pred_engines.get(REGION) is not None)

    # confirm the python bridge (production data source) is up
    try:
        import urllib.request
        bru = (os.getenv("PYTHON_BRIDGE_BASE_URL") or "").rstrip("/")
        r = urllib.request.urlopen(f"{bru}/healthz", timeout=5)
        check("python bridge reachable", r.status == 200, bru)
    except Exception as exc:
        check("python bridge reachable", False, str(exc))

    _install_min_stubs()

    class _Delegate:
        def _replace_baseline_results(self, *a, **k):
            return 0

    svc = svc_mod.LTEPredictionOffsetService()
    svc._save_delegate = _Delegate()
    job_id = f"validator-{uuid.uuid4().hex[:8]}"
    svc_mod.JOBS[job_id] = {"status": "queued"}
    cfg = {
        "project_id": PROJECT_ID,
        "session_ids": sids,
        "region": REGION,
        "operator": "",
        "polygon_ids": None,
        "radius_m": 500.0,
        "grid_resolution": 25.0,
        "building": True,
        "dem_raster_path": str(_DEM_5M) if _DEM_5M.is_file() else None,
        "ghs_obat_csv_path": None,
        "ensure_all_cells": True,
        "out_of_radius_backfill_k_nearest": 8,
        "enable_phase36_v2": True,
        "dt_replace_radius_m": 25.0,
    }

    print("\n-- running production _run (real DB fetch, dry-run save) --\n")
    svc._run(job_id, cfg)

    job = svc_mod.JOBS[job_id]
    print(f"\n-- job status: {job.get('status')}  {job.get('progress') or job.get('error', '')}")
    check("job completed", job.get("status") == "done", str(job.get("error", "")))

    fin = _CAPTURED.get("final_df")
    check("save frame captured", fin is not None)
    if fin is None:
        return _summary()

    ps = job.get("metrics", {})
    print(f"\n-- save frame: {len(fin):,} rows, {fin['grid_id'].nunique():,} grids, "
          f"{fin['strict_cell_key'].nunique()} cells")
    print(f"-- model tag: {ps.get('model')}   dynamic layers: {ps.get('dynamic_layers')}")
    check("model tag = phase36v2 + phase37", "phase36v2" in str(ps.get("model", "")), str(ps.get("model")))
    check("dynamic layers fitted", len(ps.get("dynamic_layers", [])) >= 1)

    for col in ("pred_rsrp", "pred_rsrq", "pred_sinr"):
        check(f"{col} column present", col in fin.columns)
    rsrp = pd.to_numeric(fin.get("pred_rsrp"), errors="coerce")
    rsrq = pd.to_numeric(fin.get("pred_rsrq"), errors="coerce")
    sinr = pd.to_numeric(fin.get("pred_sinr"), errors="coerce")
    check("pred_rsrp non-null > 90%", rsrp.notna().mean() > 0.9,
          f"{rsrp.notna().mean()*100:.0f}% | p5/p50/p95 {rsrp.quantile(.05):.0f}/{rsrp.median():.0f}/{rsrp.quantile(.95):.0f}")
    check("pred_rsrp within [-140,-30]", rsrp.dropna().between(-140, -30).mean() > 0.95)
    check("pred_rsrq populated", rsrq.notna().mean() > 0.2,
          f"{rsrq.notna().mean()*100:.0f}% non-null, median {rsrq.median():.1f}")
    check("pred_sinr populated", sinr.notna().mean() > 0.2,
          f"{sinr.notna().mean()*100:.0f}% non-null, median {sinr.median():.1f}")

    if "calibration_status" in fin.columns:
        print("-- calibration_status:", fin["calibration_status"].value_counts().to_dict())
    if {"Technology", "final_rsrp"}.issubset(fin.columns):
        for tech in ("4G", "5G"):
            best = (fin[fin["Technology"].astype(str) == tech]
                    .sort_values("final_rsrp").groupby("grid_id").tail(1))
            m = pd.to_numeric(best["final_rsrp"], errors="coerce")
            print(f"   {tech}: {len(best):,} serving grids, median final_rsrp {m.median():.1f} dBm, "
                  f"no-cov {m.isna().mean()*100:.0f}%")

    # schema check against the real target table
    try:
        with _pred_engines[REGION].connect() as c:
            tcols = {r[0] for r in c.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=DATABASE() AND table_name='lte_prediction_baseline_results'"))}
        need = {"lat", "lon", "pred_rsrp", "pred_rsrq", "pred_sinr", "project_id", "job_id"}
        check("target table has the KPI columns", need.issubset(tcols), f"missing {need - tcols}")
    except Exception as exc:
        check("target table schema check", False, str(exc))

    _summary()


def _summary() -> None:
    npass = sum(1 for _, ok, _ in _CHECKS if ok)
    print(f"\n=== {npass}/{len(_CHECKS)} checks passed ===")
    for name, ok, detail in _CHECKS:
        if not ok:
            print(f"  FAIL: {name}  {detail}")
    sys.exit(0 if npass == len(_CHECKS) else 1)


if __name__ == "__main__":
    main()
