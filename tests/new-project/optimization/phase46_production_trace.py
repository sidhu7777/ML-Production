"""Observe the unchanged production baseline; use read-only persistence harness."""
from pathlib import Path
import sys,json,functools
ML=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ML/'tests/debug'))
import profile_offset_baseline_readonly as h

def capture(name,filename):
    original=getattr(h.S,name)
    @functools.wraps(original)
    def wrapper(*args,**kwargs):
        result=original(*args,**kwargs)
        result.to_parquet(h.RUN/filename,index=False)
        return result
    setattr(h.S,name,wrapper)

for name,filename in [('_prepare_site_rows','phase46_sites.parquet'),('_grid_from_bridge_or_db','phase46_grid.parquet'),('_run_raw_surface','phase46_raw_candidates.parquet')]:
    capture(name,filename)

if __name__=='__main__':
    h.main()
    destination=ML/'tests/new-project/optimization/phase46_latest.json'
    destination.write_text(json.dumps({'run':str(h.RUN),'project_id':210},indent=2),encoding='utf-8')
