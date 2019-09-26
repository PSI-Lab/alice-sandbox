# generate short rand sequence
# toy dataset
import argparse
import random
import cPickle as pickle
import pandas as pd
from utils import get_fe_struct, one_idx
from dgutils.pandas import Column, get_metadata, write_dataframe, add_columns, add_column


def main(infile, outfile):
    # assert outfile
    # seqs = []  # TODO use set so we don't add duplicates (but the probability of getting duplicate is very low)
    # for _ in range(num_seqs):
    #     seq = ''.join(random.choice(list('ACGU')) for _ in range(random.randint(minlen, maxlen)))
    #     seqs.append(seq)
    # print("Running rnafold...")
    # df = pd.DataFrame({'sequence': seqs})
    df = pd.read_csv(infile)
    # add mid point pair arr
    df = add_columns(df, ['pair_matrix', 'free_energy', 'mfe_frequency', 'ensemble_diversity'],
                     ['sequence'], get_fe_struct)
    # replace matrix with 1 idx, since it's sparse
    df = add_column(df, 'one_idx', ['pair_matrix'], one_idx)
    df = df.drop(columns=['pair_matrix'])
    # add length
    df = add_column(df, 'len', ['sequence'], lambda x: len(x))
    # rename so match new data schema, to ease training setup
    df = df.rename(columns={'sequence': 'seq'})

    df.to_pickle(outfile, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inf', type=str, default=10, help='input file')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()
    main(args.inf, args.out)

