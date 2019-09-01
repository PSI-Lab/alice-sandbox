import os
import re
import numpy as np
import pandas as pd
import tempfile


def db_to_mat(seq, db_notation):
    assert len(seq) == len(db_notation)
    db_file = tempfile.NamedTemporaryFile()
    out_file = tempfile.NamedTemporaryFile()
    with open(db_file.name, 'w') as f:
        # TODO need this extra 'energy' string for b2ct to work?
        f.write('{}\n{} (-100.0)'.format(seq, db_notation))
    cmd = 'b2ct < {} > {}'.format(db_file.name, out_file.name)
    os.system(cmd)
    # load header
    with open(out_file.name, 'r') as fp:
        tmp = fp.read()
        lines = tmp.splitlines()
    seq_len = int(re.match(string=lines[0], pattern=r'\s+(\d+)\s+ENERGY.*').group(1))
    assert len(seq) == seq_len
    # load data
    df = pd.read_csv(out_file.name, skiprows=1, header=None,
                     names=['i1', 'base', 'idx_i', 'i2', 'idx_j', 'i3'], sep=r"\s*")
    assert ''.join(df['base'].tolist()) == seq
    # matrix
    vals = np.zeros((len(seq), len(seq)))
    for _, row in df.iterrows():
        idx_i = row['idx_i']
        idx_j = row['idx_j'] - 1
        if idx_j != -1:
            vals[idx_i, idx_j] = 1
            vals[idx_j, idx_i] = 1
    return vals
