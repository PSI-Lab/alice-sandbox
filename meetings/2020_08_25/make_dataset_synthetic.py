import numpy as np
import pandas as pd
from dgutils.pandas import add_column
import sys
sys.path.insert(0, '../../rna_ss/')
from utils import db2pairs
from local_struct_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb


def add_target(seq, one_idx):
    pairs, structure_arr = one_idx2arr(one_idx, len(seq), remove_lower_triangular=True)
    pairs = sort_pairs(pairs)
    parser = LocalStructureParser(pairs)
    return parser.local_structure_bounding_boxes


df = pd.read_pickle('data/rand_seqs_var_len_sample_mfe_10_100_100000.pkl.gz')
df = add_column(df, 'bounding_boxes',
                 ['seq', 'one_idx'], add_target, pbar=True)

df.to_pickle('data/rand_seqs_var_len_sample_mfe_10_100_100000_pixel_bb.pkl.gz', compression='gzip')
