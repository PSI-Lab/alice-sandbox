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


def find_match_bb(bb, df_target, bb_type):
    hit = df_target[(df_target['bb_type'] == bb_type) & (df_target['bb_x'] == bb['bb_x']) & (df_target['bb_y'] == bb['bb_y']) & (df_target['siz_x'] == bb['siz_x']) & (df_target['siz_y'] == bb['siz_y'])]
    if len(hit) > 0:
        assert len(hit) == 1
        return True
    else:
        return False


def encode_bb(x):
    prob_sm = x['prob_sm']
    if len(prob_sm) == 0:  # avoid nan with np.median
        prob_sm_med = 0
    else:
        prob_sm_med = np.median(prob_sm)

    prob_sl = x['prob_sl']
    if len(prob_sl) == 0:  # avoid nan with np.median
        prob_sl_med = 0
    else:
        prob_sl_med = np.median(prob_sl)

    return [x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y'],
    prob_sm_med, len(x['prob_sm']) / (x['siz_x'] * x['siz_y']),
    prob_sl_med, len(x['prob_sl']) / (x['siz_x'] * x['siz_y'])]


def make_dataset(df):
    # for the sole purpose of training, subset to example where s2 label can be generated EXACTLY
    # i.e. subset to example where s1 bb sensitivity is 100%
    df = dgp.add_column(df, 'n_bb', ['bounding_boxes'], len)
    n_old = len(df)
    df = df[df['n_bb'] == df['n_bb_found']]
    logging.info("Subset to examples with 100% S1 bb sensitivity (for now). Before {}, after {}".format(n_old, len(df)))

    # putting together the dataset
    # for each row:
    # encode input: a list of:
    # bb_type, x, y, wx, wy, median_prob, n_proposal_normalized  (TODO add both corners?)
    # encode output: binary label for each input 'position'

    x_all = []
    y_all = []

    for idx, row in df.iterrows():
        if idx % 10000 == 0:  # FIXME idx is the original idx (not counter)
            logging.info("Processed {} examples".format(idx))

        _x = []
        _y = []
        df_target = pd.DataFrame(row['df_target'])
        if row['bb_stem'] is not None:
            for x in row['bb_stem']:
                if find_match_bb(x, df_target, 'stem'):
                    label = 1
                else:
                    label = 0
                # 100 for stem
                feature = [1, 0, 0]
                feature.extend(encode_bb(x))
                _x.append(feature)
                _y.append(label)
        if row['bb_iloop'] is not None:
            for x in row['bb_iloop']:
                if find_match_bb(x, df_target, 'iloop'):
                    label = 1
                else:
                    label = 0
                # 010 for iloop
                feature.extend(encode_bb(x))
                _x.append(feature)
                _y.append(label)
        if row['bb_hloop'] is not None:
            for x in row['bb_hloop']:
                if find_match_bb(x, df_target, 'hloop'):
                    label = 1
                else:
                    label = 0
                # 001 for hloop, also multiple normalized n_proposal by 2 to make upper limit 1
                feature.extend(encode_bb(x))
                _x.append(feature)
                _y.append(label)
        x_all.append(np.array(_x))
        y_all.append(np.array(_y))
    return x_all, y_all  # two lists


def main(in_file, out_file):

    # dataset
    logging.info("Loading {}".format(in_file))
    df = pd.read_pickle(in_file)
    logging.info("Loaded {} examples. Making dataset...".format(len(df)))
    x_all, y_all = make_dataset(df)
    assert len(x_all) == len(y_all)

    np.savez(out_file, x=x_all, y=y_all)


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
