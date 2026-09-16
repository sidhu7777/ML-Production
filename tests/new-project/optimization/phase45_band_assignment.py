"""Production-path DT band assignment experiment. Production modules unchanged."""
from pathlib import Path
import sys
import re
import json
import pandas as pd

ML = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ML / 'tests/debug'))
import profile_offset_baseline_readonly as harness

S = harness.S
original = S._run_cost231_at_dt
SOURCE = ML / 'tests/output/baseline_210_profile_20260908_115936'
# Freeze DT input and preserve original dt_row_id so the existing validation
# split remains unchanged. All physical/calibration functions remain production.
S.fetch_drive_data = lambda *args, **kwargs: pd.read_parquet(SOURCE / 'drive_test_rows.parquet')


def measured_band(row):
    network = str(row.get('network', row.get('technology', ''))).upper()
    tech = '5G' if ('5G' in network or 'NR' in network) else '4G'
    label = str(row.get('band', '')).strip()
    if tech == '5G':
        match = re.fullmatch(r'n(\d+)', label, re.I)
        return tech, match.group(1) if match else None
    e = pd.to_numeric(row.get('earfcn'), errors='coerce')
    ranges = [(0,599,'1'),(1200,1949,'3'),(2750,3449,'7'),(3450,3799,'8'),(9210,9659,'28')]
    for lo,hi,band in ranges:
        if lo <= e <= hi:
            return tech,band
    return tech,None


def corrected(site_df, dt_df):
    dt = dt_df.copy()
    identities = dt.apply(measured_band,axis=1)
    dt['phase45_measured_technology'] = [x[0] for x in identities]
    dt['phase45_measured_band'] = [x[1] for x in identities]
    dt['phase45_exclusion'] = ''
    frames=[]
    for (tech,band), group in dt.groupby(['phase45_measured_technology','phase45_measured_band'],dropna=False):
        sites = site_df[site_df.technology_key.astype(str).eq(tech) & site_df.band_key.astype(str).eq(str(band))]
        if pd.isna(band) or sites.empty:
            dt.loc[group.index,'phase45_exclusion'] = 'unknown_measured_band' if pd.isna(band) else 'no_site_in_measured_band'
            continue
        scored=original(sites,group)
        assert scored.assigned_strict_cell_key.isin(sites.strict_cell_key).all()
        frames.append(scored)
    dt.to_parquet(harness.RUN/'phase45_selection.parquet',index=False)
    dt.groupby(['phase45_measured_technology','phase45_measured_band','phase45_exclusion'],dropna=False).size().to_csv(harness.RUN/'phase45_selection_counts.csv')
    if not frames:
        raise ValueError('No supported measured bands')
    return pd.concat(frames,ignore_index=True)


S._run_cost231_at_dt = corrected
if __name__ == '__main__':
    print('PHASE45_RUN '+str(harness.RUN),flush=True)
    harness.main()
