# generate short rand sequence
# toy dataset
import argparse
import random
import cPickle as pickle
import pandas as pd
from utils import get_fe_struct
from dgutils.pandas import Column, get_metadata, write_dataframe, add_columns


def main(seq_len=50, num_seqs=100000):
    seqs = []  # TODO use set so we don't add duplicates (but the probability of getting duplicate is very low)
    for _ in range(num_seqs):
        seq = ''.join(random.choice(list('ACGU')) for _ in range(seq_len))
        seqs.append(seq)
    print("Running rnafold...")
    df = pd.DataFrame({'sequence': seqs})
    # add mid point pair arr
    df = add_columns(df, ['pair_matrix', 'free_energy', 'mfe_frequency', 'ensemble_diversity'],
                     ['sequence'], get_fe_struct)
    df.to_pickle('data/rand_seqs_fe_{}_{}.pkl.gz'.format(seq_len, num_seqs), compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--len', type=int, default=50, help='sequence length')
    parser.add_argument('--num', type=int, default=100000, help='total number of sequences to generate')
    args = parser.parse_args()
    main(args.len, args.num)

