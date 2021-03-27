import argparse
import os
import logging
import subprocess
import yaml
import tqdm
import pandas as pd
import dgutils.pandas as dgp
import numpy as np
import math


logging.basicConfig(level=logging.INFO,
                   format="%(asctime)-15s %(message)s",
                   datefmt='%Y-%m-%d %H:%M:%S')


def encode_sequence(seq, order='ACGT', method='standard', allow_u=True, encode_n='zeros'):
    assert set(list(order)) == set(list('ACGT'))
    assert encode_n in ['zeros', 'average']
    assert method in ['standard', 'signed']
    ENCODING = np.zeros([90, 4], np.float32)
    ENCODING[[ord(base) for base in order]] = np.eye(4)
    if method == 'signed':
        ENCODING[[ord(x) for x in 'ACGT']] -= 0.25
    if encode_n == 'average':
        ENCODING[(ord('N'))].fill(0.25)
    valid_strs = b'ACGTN'
    if allow_u:
        ENCODING[ord('U')] = ENCODING[ord('T')]
        valid_strs += b'U'
    y = seq.encode('ascii').upper()
    assert all(x in valid_strs for x in y)
    return ENCODING[memoryview(y)]


def encode_stem_bb(x, seq, pad_len):
    # encode stem bb:
    # 8-channel encoding of the base pairs (TODO this really should be flip-invariant)
    bb_x, bb_y, siz_x, siz_y = x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y']
    assert siz_x == siz_y
    assert siz_x <= pad_len
    s1 = seq[bb_x:bb_x+siz_x]
    s2 = seq[bb_y+1-siz_y:bb_y+1][::-1]   # second sequence need to be flipped so we can align them along channel dimension
    assert len(s1) == siz_x
    assert len(s2) == siz_y
    feature = np.concatenate([encode_sequence(s1), encode_sequence(s2)], axis=1)
    feature_padded = np.zeros((pad_len, 8))
    feature_padded[:feature.shape[0], :] = feature
#     print(s1, s2, feature.shape, feature_padded.shape)
#     print(s1, encode_sequence(s1).shape)
#     print(s2, encode_sequence(s2).shape)
    return feature_padded, feature.shape[0]


def encode_iloop_bb(x, seq, pad_len):
    # encode iloop bb:
    # 5-channel encoding: base (4-ch) and 0/1 indicator of closing base pair
    bb_x, bb_y, siz_x, siz_y = x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y']
    assert siz_x + siz_y <= pad_len
    s1 = seq[bb_x:bb_x+siz_x]
    s2 = seq[bb_y+1-siz_y:bb_y+1][::-1]   # second sequence need to be flipped so we can align them along channel dimension
    assert len(s1) == siz_x
    assert len(s2) == siz_y
    feature_seq = np.concatenate([encode_sequence(s1), encode_sequence(s2)], axis=0)  # concat along length
    assert feature_seq.shape[0] == siz_x + siz_y
    feature_bin = np.zeros((siz_x + siz_y, 1))
    feature_bin[0, 0] = 1
    feature_bin[-1, 0] = 1
    feature_bin[siz_x - 1, 0] = 1
    feature_bin[siz_x, 0] = 1
    feature = np.concatenate([feature_seq, feature_bin], axis=1)
    # pad
    feature_padded = np.zeros((pad_len, 5))
    feature_padded[:feature.shape[0], :] = feature
    return feature_padded, siz_x + siz_y


def encode_hloop_bb(x, seq, pad_len):
    # encode hloop bb:
    # 4-channel encoding: just the bases
    bb_x, bb_y, siz_x, siz_y = x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y']
    assert siz_x == siz_y
    assert siz_x <= pad_len
    s1 = seq[bb_x:bb_x+siz_x]
    assert len(s1) == siz_x
    feature = encode_sequence(s1)
    # pad
    feature_padded = np.zeros((pad_len, 4))
    feature_padded[:feature.shape[0], :] = feature
    return feature_padded, siz_x


def find_match_bb(bb, df_target, bb_type):
    hit = df_target[(df_target['bb_type'] == bb_type) & (df_target['bb_x'] == bb['bb_x']) & (df_target['bb_y'] == bb['bb_y']) & (df_target['siz_x'] == bb['siz_x']) & (df_target['siz_y'] == bb['siz_y'])]
    if len(hit) > 0:
        assert len(hit) == 1
        return True
    else:
        return False


def encode_bb(x):
    return [x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y']]


def make_dataset(df):
    # for the sole purpose of training, subset to example where s2 label can be generated EXACTLY
    # i.e. subset to example where s1 bb sensitivity is 100%
    df = dgp.add_column(df, 'n_bb', ['bounding_boxes'], len)
    n_old = len(df)
    df = df[df['n_bb'] == df['n_bb_found']]
    # FIXME no need to do this
    logging.info("Subset to examples with 100% S1 bb sensitivity (for now). Before {}, after {}".format(n_old, len(df)))

    # putting together the dataset
    # for each row:
    # encode input: a list of:
    # bb_type, x, y, wx, wy,
    # + sequence padded
    # encode output: binary label for each input 'position'

    x_bb = []
    x_stem = []
    x_iloop = []
    x_hloop = []
    l_stem = []
    l_iloop = []
    l_hloop = []
    y_all = []

    for idx, row in df.iterrows():
        if len(x_bb) % 100 == 0:
            logging.info("Processed {} examples".format(idx))

        _x_bb = []
        _x_stem = []
        _x_iloop = []
        _x_hloop = []
        _l_stem = []
        _l_iloop = []
        _l_hloop = []
        _y = []
        df_target = pd.DataFrame(row['df_target'])

        seq = row['seq']

        # stems
        if row['bb_stem'] is not None:
            # encode sequence
            # find longest bb, for padding
            max_l_stem = max([x['siz_x'] for x in row['bb_stem']])
            for x in row['bb_stem']:
                #     print(encode_bb(x))
                feature, l = encode_stem_bb(x, seq, max_l_stem)
                _x_stem.append(feature)
                _l_stem.append(l)

            # bb feature and target
            for x in row['bb_stem']:
                if find_match_bb(x, df_target, 'stem'):
                    label = 1
                else:
                    label = 0
                # 100 for stem
                feature = [1, 0, 0]
                feature.extend(encode_bb(x))
                _x_bb.append(feature)
                _y.append(label)

        # iloops
        if row['bb_iloop'] is not None:
            # encode sequence
            # find longest bb, for padding
            max_l_iloop = max([x['siz_x'] + x['siz_y'] for x in row['bb_iloop']])
            for x in row['bb_iloop']:
                #     print(encode_bb(x))
                feature, l = encode_iloop_bb(x, seq, max_l_iloop)
                _x_iloop.append(feature)
                _l_iloop.append(l)

            # bb feature and target
            for x in row['bb_iloop']:
                if find_match_bb(x, df_target, 'iloop'):
                    label = 1
                else:
                    label = 0
                # 010 for iloop
                feature = [0, 1, 0]
                feature.extend(encode_bb(x))
                _x_bb.append(feature)
                _y.append(label)

        # hloops
        if row['bb_hloop'] is not None:
            # encode sequence
            # find longest bb, for padding
            max_l_hloop = max([x['siz_x'] + x['siz_y'] for x in row['bb_hloop']])
            for x in row['bb_hloop']:
                #     print(encode_bb(x))
                feature, l = encode_hloop_bb(x, seq, max_l_hloop)
                _x_hloop.append(feature)
                _l_hloop.append(l)

            # bb feature and target
            for x in row['bb_hloop']:
                if find_match_bb(x, df_target, 'hloop'):
                    label = 1
                else:
                    label = 0
                # 001 for hloop, note that we're NOT multiply normalized n_proposal by 2 to make upper limit 1
                feature = [0, 0, 1]
                feature.extend(encode_bb(x))
                _x_bb.append(feature)
                _y.append(label)


        x_bb.append(np.array(_x_bb))
        x_stem.append(np.array(_x_stem))
        x_iloop.append(np.array(_x_iloop))
        x_hloop.append(np.array(_x_hloop))
        l_stem.append(np.array(_l_stem))
        l_iloop.append(np.array(_l_iloop))
        l_hloop.append(np.array(_l_hloop))
        y_all.append(np.array(_y))
    return x_bb, x_stem, x_iloop, x_hloop, l_stem, l_iloop, l_hloop, y_all  # lists (each with variable number of bbs)


def main(in_file, out_file):

    # dataset
    logging.info("Loading {}".format(in_file))
    df = pd.read_pickle(in_file)
    logging.info("Loaded {} examples. Making dataset...".format(len(df)))
    x_bb, x_stem, x_iloop, x_hloop, l_stem, l_iloop, l_hloop, y_all = make_dataset(df)
    assert len(x_bb) == len(y_all)
    assert len(x_stem) == len(y_all)
    assert len(x_iloop) == len(y_all)
    assert len(x_hloop) == len(y_all)
    assert len(l_stem) == len(y_all)
    assert len(l_iloop) == len(y_all)
    assert len(l_hloop) == len(y_all)

    np.savez(out_file,
             x_bb=x_bb,
             x_stem=x_stem, x_iloop=x_iloop, x_hloop=x_hloop,
             l_stem=l_stem, l_iloop=l_iloop, l_hloop=l_hloop,
             y=y_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Path to input file, should be output from stage 1 with pruning (prune_stage_1.py)')
    parser.add_argument('--out_file', type=str, help='output dataset in npz format')

    args = parser.parse_args()

    # some basic logging
    logging.info("Cmd: {}".format(args))  # cmd args
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    logging.info("Current dir: {}, git hash: {}".format(cur_dir, git_hash))

    main(args.in_file, args.out_file)
