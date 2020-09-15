import datacorral as dc
import numpy as np
import pandas as pd


dc_client = dc.Client()

df = pd.read_pickle(dc_client.get_path('DmNgdP'), compression='gzip')
df = df[['seq_id', 'data_partition', 'seq', 'len', 'verified']]

df.to_csv('tmp/bprna_seq_only.csv', index=False)
