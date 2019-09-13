import argparse
import re
import os
import tempfile
import numpy as np
import pandas as pd
from dgutils.pandas import add_columns
from utils import EvalMetric


def get_fe_struct(seq):
    # get MFE structure in binary matrix, and the free energy
    input_file = tempfile.NamedTemporaryFile().name
    ss_file = tempfile.NamedTemporaryFile().name
    ct_file = tempfile.NamedTemporaryFile().name
    with open(input_file, 'w') as f:
        f.write('>seq\n' + seq)

    cmd1 = 'RNAfold -p < {} > {}'.format(input_file, ss_file)
    cmd2 = 'b2ct < {} > {}'.format(ss_file, ct_file)
    os.system(cmd1)
    os.system(cmd2)

    # find MFE probability
    with open(ss_file, 'r') as fp:
        tmp = fp.read()
        match = re.match(string=tmp.replace('\n', ''),
                         pattern=r'.*frequency of mfe structure in ensemble (.*); ensemble diversity (.*)')
        mfe_freq = float(match.group(1).strip())
        ens_div = float(match.group(2).strip())

    # load header
    with open(ct_file, 'r') as fp:
        tmp = fp.read()
        lines = tmp.splitlines()
    seq_len = int(re.match(string=lines[0], pattern=r'\s+(\d+)\s+ENERGY =(.*)seq').group(1))
    fe = float(re.match(string=lines[0], pattern=r'\s+(\d+)\s+ENERGY =(.*)seq').group(2).strip())
    assert len(seq) == seq_len
    # load data
    df = pd.read_csv(ct_file, skiprows=1, header=None,
                     names=['i1', 'base', 'idx_i', 'i2', 'idx_j', 'i3'], sep=r"\s*")
    assert ''.join(df['base'].tolist()) == seq
    # matrix
    vals = np.zeros((len(seq), len(seq)))
    for _, row in df.iterrows():
        idx_i = row['idx_i']
        idx_j = row['idx_j'] - 1
        if idx_j != -1:
            vals[idx_i, idx_j] = 1
            vals[idx_j, idx_i] = 1
    return vals, fe, mfe_freq, ens_div


def process_row(seq, one_idx):
    target = np.zeros((len(seq), len(seq)))
    target[one_idx] = 1

    # run RNAfold
    pred, fe, mfe_freq, ens_div = get_fe_struct(seq)
    # set lower triangular to 0
    pred[np.tril_indices(pred.shape[0])] = 0

    sensitivity = EvalMetric.sensitivity(pred, target)
    ppv = EvalMetric.ppv(pred, target)
    f_measure = EvalMetric.f_measure(sensitivity, ppv)

    pred_idx = np.where(pred == 1)
    return pred_idx, sensitivity, ppv, f_measure, fe, mfe_freq, ens_div


def main(dataset_file, output):
    df = pd.read_pickle(dataset_file)

    df = add_columns(df, ['pred_idx', 'sensitivity', 'ppv',
                          'f_measure', 'fe', 'mfe_freq', 'ens_div'],
                     ['seq', 'one_idx'],
                     lambda x, y: process_row(x, y))

    df.to_pickle(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path to dataset file')
    parser.add_argument('--output', type=str, help='path to output file')
    args = parser.parse_args()

    main(args.dataset, args.output)
