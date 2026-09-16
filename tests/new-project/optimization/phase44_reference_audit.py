"""Isolated Phase39 reconstruction and band-support audit; no DB writes.

Reuses frozen Phase26 inputs to isolate processing from input-fetch changes.
This is reference reproduction, not a production optimization benchmark.
"""
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_project210_phase39_equal_power_diagnostic as ref


def main():
    start = time.perf_counter()
    out = ROOT / 'data/project_210_taiwan/phase44_reference_audit'
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(ref.p36.PHASE26_DIR / 'phase26_dt_scored_project210.parquet')
    candidates = pd.read_parquet(ref.p36.PHASE26_DIR / 'phase26_scored_candidates_project210.parquet')
    dt = ref.p38._dt_inputs_from(ref.p38._rematch_4g(raw, candidates))
    dt = ref._apply_equal_power_assumptions(dt, 'assigned_strict_cell_key')
    cand = ref._apply_equal_power_assumptions(ref.p36._candidate_inputs(), 'strict_cell_key')
    eligible = dt.obstruction_branch.ne('indoor') & ~dt.p36_backlobe.astype(bool) & ~dt.p38_excluded.astype(bool)
    dt['phase44_fit_eligible'] = eligible & dt.phase25_split.eq('train')
    fit = dt[dt.phase44_fit_eligible].copy()
    dt.to_parquet(out / 'dt_assignment_and_selection.parquet', index=False)
    audit = dt.groupby(['technology', 'band', 'p38_true_band'], dropna=False).agg(total=('dt_row_id','size'), training_used=('phase44_fit_eligible','sum')).reset_index() if 'p38_true_band' in dt else dt.groupby(['technology','band']).agg(total=('dt_row_id','size'), training_used=('phase44_fit_eligible','sum')).reset_index()
    audit.to_csv(out / 'band_assignment_audit.csv', index=False)
    layers, local = ref.p36._fit(fit)
    pd.concat(layers, ignore_index=True).to_csv(out / 'fitted_corrections.csv', index=False)
    valid = ref._copy_phase39_score_columns(ref.p36._score(dt[dt.phase25_split.eq('validation')].copy(), layers, local))
    scored = ref._copy_phase39_score_columns(ref.p36._score(cand, layers, local))
    serving = ref._aggregate_p39(scored)
    scored.to_parquet(out / 'scored_candidates.parquet', index=False)
    valid.to_parquet(out / 'validation_dt.parquet', index=False)
    serving.to_parquet(out / 'serving_grid.parquet', index=False)
    report = {'scope':'Frozen-input Phase39 reproduction, not cold production benchmark', 'comparisons':{}}
    offset_path = ROOT.parents[1] / 'tests/output/baseline_210_profile_20260908_115936/dt_calibrated_predictions.parquet'
    offset = pd.read_parquet(offset_path)
    offset['earfcn_band'] = pd.to_numeric(offset.earfcn, errors='coerce').map(ref.p38._earfcn_band)
    offset['fit_eligible'] = offset['split'].eq('train') & offset.obstruction_branch.ne('indoor')
    offset.groupby(['technology','band','earfcn_band'],dropna=False).agg(total=('dt_row_id','size'),training_used=('fit_eligible','sum')).reset_index().to_csv(out/'offset_band_assignment_audit.csv',index=False)
    b28 = offset[offset.technology.eq('4G') & offset.band.astype(str).eq('28') & offset.fit_eligible]
    report['offset_b28_training'] = {'total':len(b28), 'earfcn_band_counts':b28.earfcn_band.value_counts().to_dict(), 'applied_correction_db':float(b28.tech_band_correction_db.iloc[0])}
    report['reference_b28_training'] = {'total':int((fit.technology.eq('4G') & fit.band.astype(str).eq('28')).sum()), 'minimum_required':ref.phase25.TECH_BAND_MIN_N, 'band_layer_present':bool(((pd.concat(layers).technology.eq('4G')) & (pd.concat(layers).band.astype(str).eq('28')) & pd.concat(layers).layer.eq('tech_band')).any())}
    for name, new, old, keys, cols in [
        ('candidates',scored,pd.read_parquet(ref.OUT_DIR/'phase39_scored_candidates_project210.parquet'),['technology','grid_id','strict_cell_key'],['phase39_final_rsrp','phase39_equal_power_rsrp']),
        ('validation',valid,pd.read_parquet(ref.OUT_DIR/'phase39_validation_dt_project210.parquet'),['technology','dt_row_id'],['rsrp_measured','phase39_final_rsrp'])]:
        merged = new.merge(old,on=keys,how='outer',suffixes=('_new','_old'),indicator=True,validate='one_to_one')
        stats = {'new_rows':len(new),'old_rows':len(old),'unmatched':int(merged['_merge'].ne('both').sum())}
        for col in cols:
            x,y=merged[col+'_new'],merged[col+'_old']
            stats[col]={'max_abs_diff':float((x-y).abs().max()),'different':int((~np.isclose(x,y,rtol=0,atol=1e-9,equal_nan=True)).sum())}
        report['comparisons'][name]=stats
    for tech in ['4G','5G']:
        oldserv=pd.read_parquet(ref.OUT_DIR/f'phase39_serving_grid_{tech.lower()}_project210.parquet')
        olddt=pd.read_parquet(ref.OUT_DIR/'phase39_validation_dt_project210.parquet')
        for mode in ['best','mean']:
            fig=make_subplots(rows=1,cols=2,subplot_titles=['Phase39 saved reference', 'Phase44 recomputed reference'])
            for panel,sv,dv in [(1,oldserv,olddt),(2,serving,valid)]:
                sv=sv[sv.technology.eq(tech)]
                dv=dv[dv.technology.eq(tech)&dv.obstruction_branch.ne('indoor')&~dv.p36_backlobe.astype(bool)&~dv.p38_excluded.astype(bool)]
                for label,values,color in [('DT measured',dv.rsrp_measured,'#dddddd'),('Predicted at DT',dv.phase39_final_rsrp,'#3b82f6'),('Outdoor',sv.loc[sv.serving_environment.eq('outdoor'),f'phase39_final_{mode}_rsrp'],'#22c55e'),('Indoor',sv.loc[sv.serving_environment.eq('indoor'),f'phase39_final_{mode}_rsrp'],'#f59e0b')]:
                    x=np.sort(pd.to_numeric(values,errors='coerce').dropna().to_numpy())
                    fig.add_trace(go.Scatter(x=x,y=np.arange(1,len(x)+1)*100/max(len(x),1),name=f'{panel}: {label} (n={len(x)})',line=dict(color=color)),row=1,col=panel)
            fig.update_layout(template='plotly_dark',title=f'{tech} / Taiwan / {mode}: reference reproduction',height=650)
            fig.update_xaxes(title='RSRP (dBm)',range=[-140,-45])
            fig.update_yaxes(title='Cumulative %',range=[0,100])
            fig.write_html(out/f'cdf_{tech}_{mode}.html',include_plotlyjs=True)
    report['elapsed_s']=time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(audit.to_string(index=False))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
