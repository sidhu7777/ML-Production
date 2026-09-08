"""Run a real baseline project with timing and read-only persistence.

No production files are edited. Only persistence boundaries and observational
wrappers are replaced inside this standalone process. SQL writes fail closed.
"""
from __future__ import annotations

import functools
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

import numpy as np
import pandas as pd
import psutil
from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from tools.lte_prediction_offset import services as S
from tools.lte_prediction_offset import geo_inputs as G
from tools.lte_prediction_offset import phase27_physical as P
from tools.lte_prediction_offset import phase27_calibration as C
from tools.lte_prediction_offset import phase36_physical_upgrades as A
from tools.lte_prediction_offset import phase37_quality as Q
from tools.lte_prediction import ml_engine as M
from utils.python_bridge import PythonBridgeClient
from urllib.parse import urlparse

parser = argparse.ArgumentParser()
parser.add_argument('--project-id', type=int, default=210)
parser.add_argument('--region', default='taiwan')
parser.add_argument('--operator', default='')
parser.add_argument('--radius-m', type=float, default=500.0)
parser.add_argument('--grid-resolution', type=float, default=25.0)
parser.add_argument('--phase43-optimized-physical', action='store_true')
parser.add_argument('--phase43-v2', action='store_true')
parser.add_argument('--phase43-v3', action='store_true')
parser.add_argument('--phase43-workers', type=int, default=max(1, min(4, (os.cpu_count() or 1) - 1)))
parser.add_argument('--no-input-cache', action='store_true')
ARGS = parser.parse_args()

RUN = ROOT / 'tests' / 'output' / (f'baseline_{ARGS.project_id}_profile_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
RUN.mkdir(parents=True, exist_ok=True)
EVENTS = []
STACK = []
SAMPLES = Counter()
LEAVES = Counter()
SQL = {'reads': 0, 'blocked_writes': 0, 'elapsed_s': 0.0}
HTTP_READS = []
FINAL = {}
STOP = threading.Event()
MAIN_THREAD = threading.get_ident()
PROCESS = psutil.Process()
PEAK_RSS = 0

_request_url = PythonBridgeClient._request_url
def read_bridge(self, method, url, **kwargs):
    endpoint = urlparse(url).path.rstrip('/').rsplit('/', 1)[-1].lower()
    allowed = {'getltesitepredictionrows', 'getprojectregions', 'getltebuildingrows',
               'getdrivetestrows', 'getgridanalytics'}
    if endpoint not in allowed or method.upper() not in {'GET', 'POST'}:
        raise RuntimeError('Diagnostic blocked non-allowlisted bridge endpoint: ' + endpoint)
    started = time.perf_counter()
    try:
        return _request_url(self, method, url, **kwargs)
    finally:
        HTTP_READS.append({'endpoint':endpoint, 'wall_s':time.perf_counter()-started})
PythonBridgeClient._request_url = read_bridge


def dump(name, value):
    (RUN / name).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def emit(value):
    value['timestamp'] = datetime.now().isoformat()
    with (RUN / 'events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, default=str) + '\n')
    print('[PROFILE] ' + json.dumps(value, default=str), flush=True)


def load_phase43():
    path = ROOT / 'tests' / 'new-project' / 'optimization'
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import phase43_optimized_physical
    return phase43_optimized_physical


def load_phase43_geo():
    path = ROOT / 'tests' / 'new-project' / 'optimization'
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import phase43_optimized_geo
    return phase43_optimized_geo


@event.listens_for(Engine, 'before_cursor_execute')
def readonly_sql(conn, cursor, statement, parameters, context, executemany):
    operation = statement.lstrip().split(None, 1)[0].upper()
    if operation not in {'SELECT', 'SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN'}:
        SQL['blocked_writes'] += 1
        raise RuntimeError('Diagnostic blocked non-read SQL: ' + operation)
    context._profile_sql_start = time.perf_counter()
    SQL['reads'] += 1


@event.listens_for(Engine, 'after_cursor_execute')
def sql_done(conn, cursor, statement, parameters, context, executemany):
    SQL['elapsed_s'] += time.perf_counter() - context._profile_sql_start


def summary(value):
    if isinstance(value, pd.DataFrame):
        result = {'rows': len(value), 'columns': len(value.columns)}
        for col in ['Technology', 'technology', 'calibration_status', 'obstruction_branch', 'clutter_class']:
            if col in value:
                result[col] = value[col].value_counts(dropna=False).to_dict()
        return result
    if isinstance(value, tuple):
        return [summary(v) for v in value]
    if isinstance(value, (str, int, float, dict)) or value is None:
        return value
    return type(value).__name__


def timed(owner, name, label=None):
    original = getattr(owner, name)
    label = label or owner.__name__.split('.')[-1] + '.' + name
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        start, cpu = time.perf_counter(), time.process_time()
        STACK.append(label)
        emit({'stage': label, 'state': 'start'})
        result = None
        try:
            result = original(*args, **kwargs)
            if name == 'fetch_drive_data' and isinstance(result, pd.DataFrame) and result.empty:
                raise RuntimeError('No drive-test inputs; refusing an uncalibrated timing run')
            return result
        finally:
            row = {'stage': label, 'state': 'end', 'wall_s': time.perf_counter()-start,
                   'cpu_s': time.process_time()-cpu, 'result': summary(result)}
            EVENTS.append(row)
            STACK.pop()
            emit(row)
    setattr(owner, name, wrapper)


def sample_stacks():
    global PEAK_RSS
    heartbeat = time.perf_counter()
    while not STOP.wait(0.02):
        frame = sys._current_frames().get(MAIN_THREAD)
        keys = []
        while frame is not None:
            code = frame.f_code
            keys.append(f'{code.co_filename}:{code.co_firstlineno}:{code.co_name}')
            frame = frame.f_back
        if keys:
            LEAVES[keys[0]] += 1
            SAMPLES.update(set(keys))
        if time.perf_counter()-heartbeat >= 15:
            PEAK_RSS = max(PEAK_RSS, PROCESS.memory_info().rss)
            dump('progress.json', {'stage': list(STACK), 'rss_mb': PROCESS.memory_info().rss/2**20,
                                  'cpu_s': time.process_time(), 'sample_count': sum(LEAVES.values())})
            heartbeat = time.perf_counter()


def dataset_readonly(db_engine, project_id, grid):
    layout = '|'.join(sorted(f'{gid}:{geom.wkb_hex}' for gid, geom in zip(grid['grid_id'].astype(str), grid.geometry)))
    digest = hashlib.sha256(layout.encode()).hexdigest()
    with db_engine.connect() as conn:
        result = conn.execute(text("""SELECT id FROM tbl_project_geo_dataset
            WHERE project_id=:p AND dataset_type='phase27_clutter'
              AND source_name=:s AND source_version=:v AND boundary_hash=:h AND is_active=1
            ORDER BY id DESC LIMIT 1"""),
            {'p':project_id, 's':G.PHASE27_CLUTTER_SOURCE, 'v':G.PHASE27_CLASSIFIER_VERSION, 'h':digest}).scalar()
    emit({'stage':'dataset_lookup', 'existing_dataset':result, 'boundary_hash':digest})
    # A missing dataset is computed by the unchanged production classifier;
    # a local sentinel replaces only its database identity and persistence.
    return int(result) if result is not None else -1


def capture_buildings(db_engine, project_id, dataset_id, buildings, method):
    buildings.drop(columns='geometry', errors='ignore').to_csv(RUN / 'building_profiles.csv', index=False)
    emit({'stage':'suppressed_building_profile_write', 'rows':len(buildings), 'method':method})


def capture_tiles(db_engine, project_id, dataset_id, grid, values):
    values.to_csv(RUN / 'computed_clutter.csv', index=False)
    emit({'stage':'suppressed_clutter_write', 'rows':len(values)})


def cache_miss_geo(db_engine, project_id: int, dataset_id: int, grid_ids: pd.Series) -> pd.DataFrame:
    emit({'stage':'forced_geospatial_cache_miss', 'project_id':project_id, 'dataset_id':dataset_id, 'grid_ids':len(grid_ids)})
    return pd.DataFrame(columns=['grid_id', 'clutter_class', 'building_height_m'])


def capture_final(delegate, frame, project_id, job_id, operator, region):
    frame.to_parquet(RUN / 'baseline_predictions.parquet', index=False)
    FINAL.update(summary(frame))
    FINAL['production_summary'] = frame.attrs.get('production_summary', {})
    FINAL['metrics'] = {
        col: {'non_null':int(frame[col].notna().sum()), 'min':float(frame[col].min()),
              'median':float(frame[col].median()), 'max':float(frame[col].max())}
        for col in ['pred_rsrp', 'pred_rsrq', 'pred_sinr'] if col in frame
    }
    if 'phase36_pap_file' in frame:
        FINAL['pap_files'] = frame['phase36_pap_file'].value_counts().to_dict()
    emit({'stage':'suppressed_baseline_and_geo_write', 'rows':len(frame)})


def main():
    files = list((ROOT / 'tools' / 'lte_prediction_offset').rglob('*.py'))
    files += [ROOT / 'tools' / 'lte_prediction' / name for name in
              ['services.py', 'ml_engine.py', 'Sector_wise_prediction_code_copy.py', 'geo_correction_pipeline.py', 'dem_utils.py']]
    before = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    dump('production_hashes_before.json', before)
    region = str(ARGS.region or 'india').lower()
    with S.engine.get(region, S.engine['india']).connect() as conn:
        ref = conn.execute(text('SELECT ref_session_id FROM tbl_project WHERE id=:project_id'), {'project_id': ARGS.project_id}).scalar()
    sessions = [int(v) for v in re.findall(r'\d+', str(ref or ''))]
    assert sessions, 'Project has no reference sessions'
    cfg = dict(project_id=ARGS.project_id, session_ids=sessions, region=region, operator=ARGS.operator,
               polygon_ids=None, radius_m=ARGS.radius_m, grid_resolution=ARGS.grid_resolution, building=True,
               dem_raster_path=None, ghs_obat_csv_path=None, ensure_all_cells=True,
               out_of_radius_backfill_k_nearest=8, enable_phase36_v2=True,
               dt_replace_radius_m=25.0, n_workers=max(1, (os.cpu_count() or 1)-1))
    dump('config.json', cfg)
    emit({'stage':'run_start', 'run_dir':str(RUN), 'config':cfg,
          'data_path':'production bridge reads with guarded SQL reads; existing caches allowed', 'database_writes':False})
    if ARGS.phase43_optimized_physical or ARGS.phase43_v2 or ARGS.phase43_v3:
        phase43 = load_phase43()
        if ARGS.phase43_v2 or ARGS.phase43_v3:
            def phase43_v2_wrapper(*args, **kwargs):
                workers = min(ARGS.phase43_workers, 3) if ARGS.phase43_v3 else ARGS.phase43_workers
                return phase43.score_candidates_phase43_v2(*args, workers=workers, **kwargs)
            S.score_candidates = phase43_v2_wrapper
            emit({'stage':'phase43_v2_patch', 'target':'tools.lte_prediction_offset.services.score_candidates', 'workers':(min(ARGS.phase43_workers, 3) if ARGS.phase43_v3 else ARGS.phase43_workers)})
        else:
            S.score_candidates = phase43.score_candidates_phase43
            emit({'stage':'phase43_patch', 'target':'tools.lte_prediction_offset.services.score_candidates'})
    if ARGS.phase43_v3:
        G._fetch_overture_context = load_phase43_geo().fetch_overture_context_phase43_v3
        emit({'stage':'phase43_v3_patch', 'target':'tools.lte_prediction_offset.geo_inputs._fetch_overture_context'})
    if ARGS.no_input_cache:
        original_drive_cache_path = M._drive_cache_path
        def fresh_drive_cache_path(project_id, operator, region, session_ids):
            return str(RUN / ('fresh_drive_' + Path(original_drive_cache_path(project_id, operator, region, session_ids)).name))
        M._drive_cache_path = fresh_drive_cache_path
        G._read_cached = cache_miss_geo
        emit({'stage':'no_input_cache_patch', 'drive_cache':'run_local_unique_path', 'geospatial_cache':'forced_miss'})
    G._phase27_dataset = dataset_readonly
    G._save_building_profiles = capture_buildings
    G._save_phase27_tiles = capture_tiles
    S._save_offset_baseline_results = capture_final
    for name in ['fetch_site_data', '_resolve_prediction_polygons', '_prepare_site_rows', 'fetch_drive_data',
                 'fetch_building_data', '_grid_from_bridge_or_db', '_run_raw_surface',
                 'load_or_build_phase27_clutter', '_run_cost231_at_dt', 'score_candidates',
                 'fit_outdoor', '_save_offset_baseline_results']:
        timed(S, name)
    timed(S.LTEPredictionOffsetService, '_resolve_dem_path', 'dem.resolve_and_validate')
    for name in ['_read_cached', '_impute_heights', '_fetch_overture_context', '_building_context',
                 '_clip_area_ratio', '_road_length', '_surrounding_height']:
        timed(G, name)
    timed(P._DemSampler, '__init__', 'dem.open_select_band')
    for owner, names in [(C, ['fit_local', 'apply_local', 'apply_outdoor_v2']),
                         (A, ['apply_reference_and_water', 'antenna_gain_delta_details']),
                         (Q, ['compute_quality', '_score_points'])]:
        for name in names:
            timed(owner, name)
    worker = threading.Thread(target=sample_stacks, daemon=True)
    worker.start()
    job_id = 'profile-baseline-' + uuid.uuid4().hex[:10]
    S.JOBS[job_id] = {'status':'queued'}
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        S.LTEPredictionOffsetService()._run(job_id, cfg)
    finally:
        STOP.set()
        worker.join(timeout=2)
        after = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        dump('timings.json', EVENTS)
        dump('samples.json', {'interval_s':0.02, 'samples':sum(LEAVES.values()),
                             'leaf':LEAVES.most_common(80), 'inclusive':SAMPLES.most_common(100)})
        cache_caveat = 'Input caches bypassed: unique local drive cache path and forced geospatial cache miss' if ARGS.no_input_cache else 'Existing input caches permitted'
        result = {'job':S.JOBS[job_id], 'wall_s':time.perf_counter()-wall, 'cpu_s':time.process_time()-cpu,
                  'peak_rss_mb':max(PEAK_RSS, PROCESS.memory_info().rss)/2**20,
                  'sql':SQL, 'bridge_reads':HTTP_READS, 'production_files_unchanged':before==after, 'final':FINAL,
                  'caveats':['DB writes excluded', 'Production bridge read endpoints allowed', cache_caveat,
                             'Stack sampling and timing add observational overhead', 'One measured run; not a p95 estimate']}
        dump('summary.json', result)
        emit({'stage':'run_end', **result})
    assert before == after, 'Production files changed during diagnostic'
    assert SQL['blocked_writes'] == 0, 'Unexpected SQL write attempted'
    assert S.JOBS[job_id]['status'] == 'done' and FINAL, 'Baseline failed; see summary/log'
    assert FINAL['metrics']['pred_rsrq']['non_null'] > 0 and FINAL['metrics']['pred_sinr']['non_null'] > 0, 'Quality stage failed'
    print('PROFILE_SUCCESS ' + str(RUN), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        print('PROFILE_FAILED ' + str(RUN), flush=True)
        sys.exit(1)

