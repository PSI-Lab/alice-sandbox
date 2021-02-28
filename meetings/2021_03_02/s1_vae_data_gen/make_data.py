# generate short rand sequence
# toy dataset
import argparse
import random
from tqdm import tqdm
import numpy as np
import pickle
# from subprocess import PIPE, Popen
import pandas as pd
from dgutils.pandas import add_columns, add_column
import sys
sys.path.insert(0, '../rna_ss_utils/')
from utils import sample_structures, db2pairs
from local_struct_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb


def add_target(seq, one_idx):
    pairs, structure_arr = one_idx2arr(one_idx, len(seq), remove_lower_triangular=True)
    pairs = sort_pairs(pairs)
    parser = LocalStructureParser(pairs)
    return parser.local_structure_bounding_boxes


def main(minlen=10, maxlen=100, num_seqs=100, num_sample=10, outfile=None):
    assert outfile
    data = []
    for i in tqdm(range(num_seqs)):
        seq = ''.join(random.choice(list('ACGU')) for _ in range(random.randint(minlen, maxlen)))

        # sample structures
        one_idxs = sample_structures(seq, num_sample)

        # find bounding boxes for each structure
        for one_idx in one_idxs: # right now we use -p option so there can be duplicates
            bounding_boxes = add_target(seq, one_idx)

            data.append({'seq_id': i,
                         'seq': seq,
                         'one_idx': one_idx,
                         'bounding_boxes': bounding_boxes})
    df = pd.DataFrame(data)
    df.to_pickle(outfile, compression='gzip')

        # seqs.append(seq)
    # print("Running rnafold...")
    # df = pd.DataFrame({'seq': seqs})
    # df = add_column(df, 'one_idx', ['seq'], lambda x: sample_structures(x, num_sample))
    # df.to_pickle(outfile, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--minlen', type=int, default=10, help='min sequence length')
    parser.add_argument('--maxlen', type=int, default=100, help='max sequence length')
    parser.add_argument('--num_seq', type=int, default=100, help='total number of sequences to generate')
    parser.add_argument('--num_sample', type=int, default=10, help='number of structures to sample per sequence')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()
    main(args.minlen, args.maxlen, args.num_seq, args.num_sample, args.out)


