import numpy as np
import pandas as pd
import itertools
from collections import namedtuple
from utils.util_global_struct import process_bb_old_to_new
from utils.rna_ss_utils import arr2db, one_idx2arr, compute_fe


def range_overlap(r1, r2):
    if r2[0] < r1[1] <= r2[1] or r1[0] < r2[1] <= r1[1]:
        return True
    else:
        return False


def bb_conflict(bb1, bb2):
    r11 = (bb1.bb_x, bb1.bb_x + bb1.siz_x)
    r12 = (bb1.bb_y - bb1.siz_y, bb1.bb_y)
    r21 = (bb2.bb_x, bb2.bb_x + bb2.siz_x)
    r22 = (bb2.bb_y - bb2.siz_y, bb2.bb_y)
    if range_overlap(r11, r21) or range_overlap(r11, r22) or range_overlap(r12, r21) or range_overlap(r12, r22):
        return True
    else:
        return False


def get_total_bp(c):
    # find elements being included
    bb_inc = np.where(c)[0]
    # get size
    sizes = [bbs[x].siz_x for x in bb_inc]
    return sum(sizes)


BoundingBox = namedtuple("BoundingBox", ['bb_x', 'bb_y', 'siz_x', 'siz_y'])



df = pd.read_pickle('data/data_len40_1000_s1_stem_bb.pkl.gz')

# subset to example with bb sensitivity = 100%
df = df[df['bb_identical'] == 1]

data = []
for _, row in df.iterrows():
    df_stem = pd.DataFrame(row.pred_stem_bb)

    # for now skip those with too many bbs, since we're doing brute-force search
    if len(df_stem) > 15:
        print(f"Skip example with {len(df_stem)} bbs for now")
        continue

    # we use df index, make sure it's contiguous
    assert df_stem.iloc[-1].name == len(df_stem) - 1

    bbs = {}
    for idx, r in df_stem.iterrows():
        bbs[idx] = BoundingBox(bb_x=r['bb_x'],
                               bb_y=r['bb_y'],
                               siz_x=r['siz_x'],
                               siz_y=r['siz_y'])

    bb_conf_arr = np.zeros((len(bbs), len(bbs)))
    for i in bbs.keys():
        bb1 = bbs[i]
        for j in bbs.keys():
            bb2 = bbs[j]
            # TODO only need to compute half
            bb_conf_arr[i, j] = bb_conflict(bb1, bb2)
    assert np.all(bb_conf_arr.T == bb_conf_arr)

    # brute force way
    n_bbs = len(bbs)
    all_combos = list(itertools.product([0, 1], repeat=n_bbs))

    valid_combos = []
    for c in all_combos:
        # find elements being included
        bb_inc = np.where(c)[0]
        # all pairs of elements
        bb_pairs = list(itertools.combinations(bb_inc, 2))
        # check if any pair violate constraint
        is_valid = True
        for bb_pair in bb_pairs:
            if bb_conf_arr[bb_pair[0], bb_pair[1]]:  # only checking one-way since it's symmetric
                is_valid = False
                break
        # check if this is valid
        if is_valid:
            valid_combos.append(c)
        else:
            continue
    df_valid_combo = pd.DataFrame({'combo': valid_combos})
    # add in bb idx
    df_valid_combo['bb_inc'] = df_valid_combo['combo'].apply(lambda c: list(np.where(c)[0]))
    df_valid_combo['total_bps'] = df_valid_combo['combo'].apply(get_total_bp)

    # output
    row['valid_combos'] = df_valid_combo.to_dict(orient='list')
    data.append(row)

data = pd.DataFrame(data)

data.to_pickle('data/data_len40_1000_s1_stem_bb_le10_combos.pkl.gz')
