import logging
import argparse
import collections
import itertools
import pickle
import pandas as pd
import numpy as np


# had to define this to load the pickle
# see https://github.com/ml4bio/e2efold/blob/cc0189e4346a257627f76c3f6ab31c453db94383/data/preprocess_archiveii.py
RNA_SS_data = collections.namedtuple('RNA_SS_data',
                                     'seq ss_label length name pairs')


def main(input_datas):
    seq_order = list('AUCG')

    x = {input_data: None for input_data in input_datas}
    for input_data in input_datas:
        with open(input_data, 'rb') as f:
            _x = pickle.load(f)
            # print(input_data, len(_x))
            x[input_data] = _x
    df = []
    # for data_point in itertools.chain.from_iterable(xs):
    for input_data, x in x.items():
        print(input_data)
        # collect length statistics
        df = []
        for data_point in x:
            # find out the number of positions with encoding (all 0 is padding)
            seq = np.sum(data_point.seq, axis=1)
            # print('seq_len: {}  n_encoded_pos: {}, n_total: {}'.format(data_point.length, np.sum(seq), len(seq)))
            # typically the recorded seq length is the same as number of non-zero rows in seq
            # but in case there are N's, let's allow seq_len >= n_encoded_pos
            if data_point.length < np.sum(seq):
                print("Skipping suspicious data: {} data_point.length {} n_encoded_pos {}".format(data_point.name, data_point.length, np.sum(seq)))
                continue
            df.append({
                'n_encoded_pos': np.sum(seq),
                'n_total': len(seq),
            })
        # report statistics
        df = pd.DataFrame(df)
        print(df.describe())
        print('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', type=str, help='path to input data file')
    args = parser.parse_args()
    logging.debug(args)
    main(args.input)

