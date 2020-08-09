# combine valid category of single output
# to make mutually exclusive softmax
# original sigmoid (non mutually exclusive):
# not_ss, stem, i_loop, h_loop, corner
# mutually exclusive classes correspond to the following assignments:
# 1, 0, 0, 0, 0    # non-local structure region
# 0, 1, 0, 0, 0    # within stem
# 0, 1, 1, 0, 1    # stem - internal_loop boundary
# 0, 1, 0, 1, 1    # stem - hairpin_loop boundary
# 0, 1, 0, 0, 1    # stem - other non-local structure boundary
# 0, 1, 1, 1, 1    # special case where stem length = 1, and is also the boundary between i_loop and h_loop
# 0, 0, 1, 0, 0    # inside internal_loop
# 0, 0, 0, 1, 0    # inside hairpin loop
# 8 classes in total
import datacorral as dc
import numpy as np
import pandas as pd
from dgutils.pandas import add_column


def convert_multiclass(target, seq_id):
    assert len(target.shape) == 3
    assert target.shape[0] == target.shape[1]
    assert target.shape[2] == 5
    # we're doing naive processing by looking at each slice
    # so the code is more readable
    target_new = np.zeros((target.shape[0], target.shape[1], 8))
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            t = target[i, j, :]
            # check for all conditions
            if np.array_equal(t, [1, 0, 0, 0, 0]):
                target_new[i, j, 0] = 1
            elif np.array_equal(t, [0, 1, 0, 0, 0]):
                target_new[i, j, 1] = 1
            elif np.array_equal(t, [0, 1, 1, 0, 1]):
                target_new[i, j, 2] = 1
            elif np.array_equal(t, [0, 1, 0, 1, 1]):
                target_new[i, j, 3] = 1
            elif np.array_equal(t, [0, 1, 0, 0, 1]):
                target_new[i, j, 4] = 1
            elif np.array_equal(t, [0, 1, 1, 1, 1]):
                target_new[i, j, 5] = 1
            elif np.array_equal(t, [0, 0, 1, 0, 0]):
                target_new[i, j, 6] = 1
            elif np.array_equal(t, [0, 0, 0, 1, 0]):
                target_new[i, j, 7] = 1
            else:  # should not be here
                raise ValueError(seq_id, i, j, t)
    return target_new


dc_client = dc.Client()
df = pd.read_pickle(dc_client.get_path('MVRSGa'), compression='gzip')


df = add_column(df, 'target', ['target', 'seq_id'], convert_multiclass, pbar=True)
df.to_pickle('data/local_struct.bp_rna.multiclass.pkl.gz', compression='gzip')


