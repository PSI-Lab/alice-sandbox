import numpy as np
import pandas as pd
from dgutils.pandas import add_columns
import sys
sys.path.insert(0, '../../rna_ss/')
from utils import db2pairs
from local_struct_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target


def add_target(seq, one_idx):
    pairs, structure_arr = one_idx2arr(one_idx, len(seq), remove_lower_triangular=True)
    pairs = sort_pairs(pairs)
    parser = LocalStructureParser(pairs)
    target_vals, target_mask = make_target(structure_arr, parser.local_structure_bounding_boxes)
    return target_vals, target_mask


df = pd.read_pickle('../../rna_ss/data_processing/spot_rna/bp_rna/data/bp_rna.pkl.gz')
df = add_columns(df, ['target', 'mask'], ['seq', 'one_idx'], add_target, pbar=True)

df.to_pickle('data/local_struct.bp_rna.pkl.gz', compression='gzip')
