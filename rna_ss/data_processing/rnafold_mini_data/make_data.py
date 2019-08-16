# generate short rand sequence
# toy dataset
import argparse
import random
import pandas as pd
from utils import get_pair_prob_arr
from dgutils.pandas import Column, get_metadata, write_dataframe, add_column


def main(seq_len=51, num_seqs=100000):
    # checks
    assert seq_len % 2 == 1, "Sequence length should be odd integer"

    seqs = []  # TODO use set so we don't add duplicates (but the probability of getting duplicate is very low)
    for _ in range(num_seqs):
        seq = ''.join(random.choice(list('ACGU')) for _ in range(seq_len))
        seqs.append(seq)

    df = pd.DataFrame({'sequence': seqs})
    # add mid point pair arr
    df = add_column(df, 'mid_point_pair_prob', ['sequence'], get_pair_prob_arr)

    metadata = get_metadata(
        Column("mid_point_pair_prob", "LIST"))
    write_dataframe(metadata, df, 'data/rand_seqs_{}.csv'.format(num_seqs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--len', type=int, help='sequence length')
    parser.add_argument('--num', type=int, help='total number of sequences to generate')
    args = parser.parse_args()
    main(args.len, args.num)

