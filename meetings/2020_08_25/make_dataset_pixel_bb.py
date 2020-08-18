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
    # target_stem_on, target_iloop_on, target_hloop_on, \
    # mask_stem_on, mask_iloop_on, mask_hloop_on, \
    # target_stem_location_x, target_stem_location_y, target_iloop_location_x, target_iloop_location_y, \
    # target_hloop_location_x, target_hloop_location_y, \
    # target_stem_size, target_iloop_size_x, target_iloop_size_y, target_hloop_size, \
    # mask_stem_location_size, mask_iloop_location_size, mask_hloop_location_size = make_target_pixel_bb(structure_arr, parser.local_structure_bounding_boxes)
    # return target_stem_on, target_iloop_on, target_hloop_on, \
    #        mask_stem_on, mask_iloop_on, mask_hloop_on, \
    #        target_stem_location_x, target_stem_location_y, target_iloop_location_x, target_iloop_location_y, \
    #        target_hloop_location_x, target_hloop_location_y, \
    #        target_stem_size, target_iloop_size_x, target_iloop_size_y, target_hloop_size, \
    #        mask_stem_location_size, mask_iloop_location_size, mask_hloop_location_size


df = pd.read_pickle('../../rna_ss/data_processing/spot_rna/bp_rna/data/bp_rna.pkl.gz')
df = add_column(df, 'bounding_boxes',
                 ['seq', 'one_idx'], add_target, pbar=True)
# df = add_columns(df, ['target_stem_on', 'target_iloop_on', 'target_hloop_on',
#            'mask_stem_on', 'mask_iloop_on', 'mask_hloop_on',
#            'target_stem_location_x', 'target_stem_location_y', 'target_iloop_location_x', 'target_iloop_location_y',
#             'target_hloop_location_x', 'target_hloop_location_y',
#            'target_stem_size', 'target_iloop_size_x', 'target_iloop_size_y', 'target_hloop_size',
#            'mask_stem_location_size', 'mask_iloop_location_size', 'mask_hloop_location_size'],
#                  ['seq', 'one_idx'], add_target, pbar=True)

df.to_pickle('data/local_struct_pixel_bb.bp_rna.pkl.gz', compression='gzip')
