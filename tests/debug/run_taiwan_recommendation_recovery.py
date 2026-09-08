"""Recover project 210 recommendation scenario 2 and verify persisted output."""
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
import pandas as pd
from sqlalchemy import text, event
from tools.lte_prediction_offset.services import _without_python_bridge
from tools.lte_prediction_optimised import services as S


def main():
    out = ROOT / 'outputs' / 'recommendation_recovery_210'
    out.mkdir(parents=True, exist_ok=True)
    with _without_python_bridge():
        engine = S._resolve_engine('taiwan')
        @event.listens_for(engine, 'connect')
        def timeout(dbapi, record):
            dbapi._read_timeout = 180
            dbapi._write_timeout = 180
        sid, reco = S._fetch_recommendation_rows(210, 'taiwan', operator='Taiwan', recommendation_scenario_id=2)
        act = S._actionable_recommendations(reco)
        base = S.fetch_baseline(210, region='taiwan', operator='Taiwan', baseline_job_id='pap-baseline-b4580bac')
        sites = S.fetch_site_data(210, region='taiwan', operator='Taiwan', allowed_cells=base['Node_Cell_ID'].unique().tolist(), polygon_ids=None)
        mod, applied = S._apply_recommendations_to_sites(sites, act)
        rows = S._build_site_prediction_update_rows(210, 0, mod, applied)
        assert len(act) == len(applied) == len(rows) == 12, (len(act), len(applied), len(rows))
        assert applied['status'].eq('applied').all()
        reco.to_csv(out / 'source_recommendations.csv', index=False)
        applied.to_csv(out / 'applied_recommendations.csv', index=False)
        print('PREFLIGHT_PASS: 12 recommendations, 12 applied, 12 site updates', flush=True)
        svc = S.LTEPredictionService_optimised()
        # Resume the already allocated slot; its replaced data was backed up.
        with engine.connect() as conn:
            protected_before = pd.read_sql(text('SELECT * FROM site_prediction_optimized WHERE tbl_project_id=210 AND scenario=6 ORDER BY id'), conn)
            assert json.loads(protected_before.to_json(orient='records', date_format='iso')) == json.loads((out / 'protected_scenario_6_before.json').read_text())
        cfg = json.loads((out / 'run.json').read_text())
        job_id = cfg.pop('job_id')
        row_id, public_id = cfg['scenario_row_id'], cfg['scenario_id']
        assert public_id == 1
        with engine.connect() as conn:
            existing = conn.execute(text('SELECT COUNT(*) FROM lte_prediction_optimised_results WHERE project_id=210 AND scenario_id=:s'), {'s':row_id}).scalar()
            assert existing == 0, 'Results already exist; verify instead of appending duplicates'
        S.JOBS[job_id] = dict(status='queued', **cfg)
        (out / 'run.json').write_text(json.dumps(dict(job_id=job_id, **cfg), indent=2))
        svc._run_recommendation_optimization(job_id, cfg)
        job = S.JOBS[job_id]
        (out / 'job.json').write_text(json.dumps(job, indent=2, default=str))
        assert job['status'] == 'done', job
        with engine.connect() as conn:
            saved = pd.read_sql(text('SELECT * FROM site_prediction_optimized WHERE tbl_project_id=210 AND scenario=:s'), conn, params={'s':public_id})
            pred = pd.read_sql(text('SELECT lat,lon,nodeb_id_cell_id,pred_rsrp,pred_rsrq,pred_sinr,Technology FROM lte_prediction_optimised_results WHERE project_id=210 AND scenario_id=:s AND job_id=:j'), conn, params={'s':row_id,'j':job_id})
            status = conn.execute(text('SELECT status FROM lte_optimization_scenarios WHERE id=:s'), {'s':row_id}).scalar()
            protected_after = pd.read_sql(text('SELECT * FROM site_prediction_optimized WHERE tbl_project_id=210 AND scenario=6 ORDER BY id'), conn)
        pd.testing.assert_frame_equal(protected_before, protected_after)
        assert status == 'done'
        assert len(pred) == len(base) == job['rows']
        assert len(saved) == 12
        by_id = saved.set_index('site_prediction_id')
        for payload in rows:
            for key, col in [('e_tilt','e_tilt'),('azimuth','azimuth')]:
                assert abs(float(by_id.loc[payload['source_id'],col])-float(payload[key])) < 1e-6
        saved.to_csv(out / 'verified_saved_sites.csv', index=False)
        pred.to_csv(out / 'verified_saved_predictions.csv', index=False)
        summary = dict(job_id=job_id, scenario_row_id=row_id, scenario_id=public_id, source_recommendation_scenario=2,
                       status=status, recommendations=12, saved_sites=len(saved), saved_predictions=len(pred),
                       technologies=pred['Technology'].value_counts().to_dict(), output=job['output'], scenario_6_unchanged=True)
        (out / 'verification.json').write_text(json.dumps(summary, indent=2))
        print('VERIFIED_SUCCESS ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
