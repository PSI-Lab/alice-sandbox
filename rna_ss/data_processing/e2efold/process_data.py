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


# for sanity check
valid_pairs = [{'A', 'U'}, {'G', 'C'}, {'G', 'U'}]


def main(input_datas, output_data):
    # mapping used by the author
    # seq_dict = {
    #     'A': np.array([1, 0, 0, 0]),
    #     'U': np.array([0, 1, 0, 0]),
    #     'C': np.array([0, 0, 1, 0]),
    #     'G': np.array([0, 0, 0, 1])
    # }
    seq_order = list('AUCG')

    x = {input_data: None for input_data in input_datas}
    for input_data in input_datas:
        with open(input_data, 'rb') as f:
            _x = pickle.load(f)
            print(input_data, len(_x))
            x[input_data] = _x
    df = []
    # for data_point in itertools.chain.from_iterable(xs):
    for input_data, x in x.items():
        for data_point in x:
            seq = ''
            char_idx = np.argmax(data_point.seq, axis=1)
            for i in range(data_point.length):
                seq += seq_order[char_idx[i]]
            # convert to one_idx
            # tuple of 2 lists, for np indexing
            _idx = data_point.pairs
            one_idx = ([x[0] for x in _idx], [x[1] for x in _idx])
            # as a sanity check, make sure these indexes are compatible with the sequence length
            # log failure, skip and proceed if suspicious
            if len(one_idx[0]) > 0 and (np.min(one_idx[0]) < 0 or np.max(one_idx[0]) >= data_point.length):
                print("Skipping suspicious data: {}\n".format(data_point))
                continue
            if len(one_idx[1]) > 0 and (np.min(one_idx[1]) < 0 or np.max(one_idx[1]) >= data_point.length):
                print("Skipping suspicious data: {}\n".format(data_point))
                continue
            if len(one_idx[0]) > 0:
                n_invalid = 0
                for i, j in zip(one_idx[0], one_idx[1]):
                    # only process upper triangular matrix
                    # so we don't double count base pairs
                    if i < j:
                        continue
                    base_pair = {seq[i], seq[j]}
                    if base_pair not in valid_pairs:
                        n_invalid += 1
                if n_invalid > data_point.length * 0.2:
                    print("{} invalid pairs in sequence of length {}".format(n_invalid, data_point.length))
                    print("Skipping suspicious data: {}\n".format(data_point))
                    continue
            df.append({
                'seq': seq,
                'one_idx': one_idx,
                'seq_id': data_point.name,
                'source_file': input_data,
            })
    df = pd.DataFrame(df)
    print('output size: {}'.format(len(df)))

    df.to_pickle(output_data, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', type=str, help='path to input data file')
    parser.add_argument('--output', type=str, help='path to output data file')
    args = parser.parse_args()
    logging.debug(args)
    main(args.input, args.output)


