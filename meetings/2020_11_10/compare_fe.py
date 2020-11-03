import numpy as np
import pandas as pd
from collections import namedtuple
import dgutils.pandas as dgp
import sys
sys.path.insert(0, '../../rna_ss/')
from utils import compute_fe


def stem_bb_to_db_str(stem_bbs, seq_len):
    # conert stem bounding boxes to dot-bracket
    # hacky implementation, no pseudoknot support!

    # # find all stems, collect all base pairs
    # all_stems = []
    # for chain in global_struct:
    #     all_stems.extend([x for x in chain.chain if x.type == 'stem'])
    # bps = []  # list of (i, j) tuple
    db_str = list('.' * seq_len)
    for s in stem_bbs:
        for i, j in zip(range(s.tr_x, s.bl_x + 1), range(s.bl_y, s.tr_y + 1)[::-1]):
            # handle corner case, if (i, j) is out of bound, skip
            # this can happen in rare cases (which should have been cleaned up after stage 1 <- to be fixed)
            if i >= seq_len or j >= seq_len:
                continue
            db_str[i] = '('
            db_str[j] = ')'
            # bps.append((i, j))
    # bps = sorted(bps)
    return ''.join(db_str)


def struct_df_to_db_str(df_struct, seq_len):
    # structure: a df, no structure: None
    if not isinstance(df_struct, pd.DataFrame):
        assert df_struct is None
        return '.' * seq_len

    BoundingBox = namedtuple('BoundingBox', ['tr_x', 'tr_y', 'bl_x', 'bl_y'])

    df_struct = df_struct[df_struct['bb_type'] == 'stem']
    stems = []
    for _, row in df_struct.iterrows():
        bb_x = row['bb_x']
        bb_y = row['bb_y']
        siz_x = row['siz_x']
        siz_y = row['siz_y']
        bl_x = bb_x + siz_x - 1
        bl_y = bb_y - siz_y + 1
        stems.append(BoundingBox(tr_x=bb_x, tr_y=bb_y, bl_x=bl_x, bl_y=bl_y))

    db_str = stem_bb_to_db_str(stems, seq_len)
    return db_str


def ad_hoc_score(df_pred):
    df_stem = df_pred[df_pred['bb_type'] == 'stem']
    df_stem = dgp.add_column(df_stem, 'score', ['siz_x', 'prob_median', 'n_proposal_norm'],
                            lambda a, b, c: a * b * c)
    return df_stem['score'].sum()


def pick_best_ss(global_struct_dfs):
    if len(global_struct_dfs) == 0:
        return None
    else:
        scores = [ad_hoc_score(df_pred) for df_pred in global_struct_dfs]
        idx = np.argmax(scores)
        df_pred = global_struct_dfs[idx]
        return df_pred





# # test
# seq = 'GGGGATTACCCC'
# struct = '((((....))))'
# fe = compute_fe(seq, struct)
# print(fe)

in_file = sys.argv[1]
out_file = sys.argv[2]


df = pd.read_pickle(in_file)
df = dgp.add_column(df, 'best_struct', ['global_struct_dfs'], pick_best_ss)
df = dgp.add_column(df, 'pred_db_str', ['best_struct', 'len'], struct_df_to_db_str)
df = dgp.add_column(df, 'pred_fe', ['seq', 'pred_db_str'], compute_fe)

# output selected columns
df = df[['free_energy', 'len', 'mfe_frequency', 'seq', 'one_idx', 'gt_found', 'best_struct', 'pred_fe']]
df.to_pickle(out_file, compression='gzip')
