# generate short rand sequence
# toy dataset
import argparse
import random
import cPickle as pickle
import pandas as pd
from utils import get_pair_prob_matrix
from dgutils.pandas import Column, get_metadata, write_dataframe, add_column


def main(seq_len=50, num_seqs=100000):
    seqs = []  # TODO use set so we don't add duplicates (but the probability of getting duplicate is very low)
    for _ in range(num_seqs):
        seq = ''.join(random.choice(list('ACGU')) for _ in range(seq_len))
        seqs.append(seq)
    print("Running rnafold...")
    df = pd.DataFrame({'sequence': seqs})
    # add mid point pair arr
    df = add_column(df, 'pair_matrix', ['sequence'], get_pair_prob_matrix)
    df.to_pickle('data/rand_seqs_2d_{}.csv.gz'.format(num_seqs), compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--len', type=int, default=50, help='sequence length')
    parser.add_argument('--num', type=int, default=100000, help='total number of sequences to generate')
    args = parser.parse_args()
    main(args.len, args.num)

