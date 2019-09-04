# generate short rand sequence
# toy dataset
import argparse
import random
import cPickle as pickle
import numpy as np
import pandas as pd
from utils import get_fe_struct, one_idx
from dgutils.pandas import Column, get_metadata, write_dataframe, add_columns, add_column


def main(minlen=10, maxlen=100, num_seqs=100000):
    data = []
    for k in range(num_seqs):
        if k % int(num_seqs//100) == 0:
            print("Generated {} data points".format(len(data)))

        # pick a length
        _len = random.randint(minlen, maxlen)
        # keep generating until success
        while True:
            seq = ''.join(random.choice(list('ACGU')) for _ in range(_len))
            pair_matrix, free_energy, mfe_frequency, ensemble_diversity = get_fe_struct(seq)
            # sample if mfe_frequency < 0.2
            if mfe_frequency >= 0.2 or (mfe_frequency < 0.2 and mfe_frequency > np.random.uniform(0, 1)):
                break  # terminate
            
        data.append({
            'seq': seq,
            'len': len(seq),
            'free_energy': free_energy,
            'mfe_frequency': mfe_frequency,
            'ensemble_diversity': ensemble_diversity,
            'one_idx': one_idx(pair_matrix),
        })

    df = pd.DataFrame(data)

    df.to_pickle('data/rand_seqs_var_len_sample_mfe_{}_{}_{}.pkl.gz'.format(minlen, maxlen, num_seqs), compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--minlen', type=int, default=10, help='min sequence length')
    parser.add_argument('--maxlen', type=int, default=100, help='max sequence length')
    parser.add_argument('--num', type=int, default=100000, help='total number of sequences to generate')
    args = parser.parse_args()
    main(args.minlen, args.maxlen, args.num)

