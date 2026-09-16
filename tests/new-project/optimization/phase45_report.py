"""Compare matched validation rows; explicitly report missing predictions."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

run=Path(sys.argv[1])
old=run.parent/'baseline_210_profile_20260908_115936'
a=pd.read_parquet(old/'dt_calibrated_predictions.parquet')
b=pd.read_parquet(run/'dt_calibrated_predictions.parquet')
m=a.merge(b,on='dt_row_id',suffixes=('_before','_after'),validate='one_to_one')
assert m['split_before'].eq(m['split_after']).all()
assert np.allclose(m.lat_before,m.lat_after) and np.allclose(m.lon_before,m.lon_after)
assert np.allclose(m.rsrp_measured_before,m.rsrp_measured_after)
assert b.band.astype(str).eq(b.phase45_measured_band.astype(str)).all()
v=m[m.split_before.eq('validation') & m.obstruction_branch_before.ne('indoor') & m.obstruction_branch_after.ne('indoor')]
report={'matched_dt':len(m),'common_outdoor_validation':len(v),'band_support':b.groupby(['technology','band'])[['tech_band_n_train','tech_band_correction_db']].first().reset_index().to_dict('records'),'metrics':{}}
ga=pd.read_parquet(old/'baseline_predictions.parquet')
gb=pd.read_parquet(run/'baseline_predictions.parquet')
gm=ga.merge(gb,on=['technology','grid_id','strict_cell_key'],suffixes=('_old','_new'),how='outer',indicator=True,validate='one_to_one')
report['physical_control']={'unmatched_grid_candidates':int(gm['_merge'].ne('both').sum())}
for col in ['raw_cost231_rsrp','building_obstruction_loss_db','terrain_diffraction_loss_db','phase36_antenna_delta_db']:
 report['physical_control'][col]={'max_abs_diff':float((gm[col+'_old']-gm[col+'_new']).abs().max()),'different':int((~np.isclose(gm[col+'_old'],gm[col+'_new'],rtol=0,atol=1e-9,equal_nan=True)).sum())}
for tech in ['4G','5G']:
 q=v[v.technology_before.eq(tech)]
 finite=q.final_rsrp_before.notna() & q.final_rsrp_after.notna()
 stats={'eligible_common_dt':len(q),'paired_finite_predictions':int(finite.sum()),'after_unsupported_or_missing':int(q.final_rsrp_after.isna().sum())}
 for side in ['before','after']:
  err=q.loc[finite,'final_rsrp_'+side]-q.loc[finite,'rsrp_measured_'+side]
  stats[side]={'mae':float(err.abs().mean()),'bias':float(err.mean()),'rmse':float(np.sqrt((err**2).mean()))} if len(err) else None
 report['metrics'][tech]=stats
 fig=make_subplots(rows=1,cols=2,subplot_titles=['Original offset','Phase45 band-constrained offset'])
 for i,(side,path) in enumerate([('before',old),('after',run)],1):
  grid=pd.read_parquet(path/'baseline_predictions.parquet')
  grid=grid[grid.technology.eq(tech)].copy()
  grid=grid.dropna(subset=['final_rsrp']).sort_values('final_rsrp').groupby('grid_id').tail(1)
  for label,series,color in [('Measured DT (common supported subset)',q.loc[finite,'rsrp_measured_'+side],'#dddddd'),('Predicted at same DT',q.loc[finite,'final_rsrp_'+side],'#3b82f6'),('Outdoor supported grids',grid.loc[grid.obstruction_branch.ne('indoor'),'final_rsrp'],'#22c55e'),('Indoor supported grids',grid.loc[grid.obstruction_branch.eq('indoor'),'final_rsrp'],'#f59e0b')]:
   x=np.sort(series.dropna().to_numpy())
   fig.add_trace(go.Scatter(x=x,y=np.arange(1,len(x)+1)*100/max(len(x),1),name=f'{i}: {label} n={len(x)}',line=dict(color=color)),row=1,col=i)
 fig.update_layout(template='plotly_dark',title=f'{tech} Phase45: supported rows only; see summary for exclusions',height=700)
 fig.update_xaxes(range=[-140,-45],title='RSRP (dBm)');fig.update_yaxes(range=[0,100],title='Cumulative %')
 fig.write_html(run/f'phase45_cdf_{tech}.html')
(run/'phase45_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
