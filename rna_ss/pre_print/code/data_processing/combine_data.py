import os
import sys
import pandas as pd


out_file = sys.argv[1]
in_dir = sys.argv[2]

dfs = []
for file in os.listdir(in_dir):
    if file.endswith(".pkl.gz"):
        x = os.path.join(in_dir, file)
        df = pd.read_pickle(x)
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df.to_pickle(out_file)
