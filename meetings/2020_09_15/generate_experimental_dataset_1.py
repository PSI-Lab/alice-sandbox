
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../../rna_ss/')
from utils import get_fe_struct
from local_struct_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb
from dgutils.pandas import add_column


def add_target(seq, one_idx):
    pairs, structure_arr = one_idx2arr(one_idx, len(seq), remove_lower_triangular=True)
    pairs = sort_pairs(pairs)
    parser = LocalStructureParser(pairs)
    return parser.local_structure_bounding_boxes


df = pd.read_csv('tmp/bprna_seq_only.csv')

# drop sequence with non-standard characters
df = add_column(df, 'seq_valid', ['seq'],
                lambda s: set(s).issubset(set(list('ACGTUacgtu'))))
print("dropping rows with invalid nucleotide, before: {}".format(len(df)))
df = df[df['seq_valid']]
df = df.drop(columns=['seq_valid'])
print("After: {}".format(len(df)))

df = add_column(df, 'one_idx', ['seq'], lambda s: get_fe_struct(s.upper().replace('T', 'U'))[0])

df.to_pickle('tmp/bprna_rnafold_struct_tmp.pkl.gz', compression='gzip')

