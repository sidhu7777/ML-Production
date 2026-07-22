from tools.report_engine.load_data_db import load_project_data
import pandas as pd, json

raw_df, filtered_df, project_meta = load_project_data(248)

for name, df in [('RAW', raw_df), ('FILTERED', filtered_df)]:
    pci = pd.to_numeric(df['pci'], errors='coerce') if 'pci' in df.columns else pd.Series(dtype=float)
    over = df.loc[pci.gt(503)].copy() if 'pci' in df.columns else df.iloc[0:0].copy()
    print(f'{name}_ROWS', len(df))
    print(f'{name}_PCI_MAX', None if pci.dropna().empty else int(pci.max()))
    print(f'{name}_PCI_GT503', len(over))
    top = pd.to_numeric(over['pci'], errors='coerce').value_counts().head(20).to_dict() if not over.empty else {}
    print(f'{name}_TOP_GT503', json.dumps({str(int(k)): int(v) for k, v in top.items()}))
    if not over.empty:
        print(f'{name}_GT503_BY_BAND', json.dumps(over['band'].astype(str).value_counts().head(20).to_dict()))
        print(f'{name}_GT503_BY_PRIMARY', json.dumps(over['primary'].astype(str).value_counts().to_dict()))
        cols = [c for c in ['session_id','band','pci','primary','primary_cell_info_1','network','cell_id'] if c in over.columns]
        print(f'{name}_SAMPLE')
        print(over[cols].head(20).to_string(index=False))
