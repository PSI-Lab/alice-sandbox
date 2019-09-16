import argparse
from StringIO import StringIO
import re
import os
import tempfile
from subprocess import PIPE, Popen
import numpy as np
import pandas as pd
from dgutils.pandas import add_columns
from utils import EvalMetric


def sample_structures(seq, n_samples):
    p = Popen(['RNAsubopt',  '-p', str(n_samples)], stdin=PIPE,
              stdout=PIPE, stderr=PIPE, universal_newlines=True)
    stdout, stderr = p.communicate(input=seq)
    rc = p.returncode
    if rc != 0:
        msg = 'RNAeval returned error code %d\nstdout:\n%s\nstderr:\n%s\n' % (
            rc, stdout, stderr)
        raise Exception(msg)
    # parse output
    lines = stdout.splitlines()
    assert len(lines) == n_samples + 1
    lines = lines[1:]
    # convert to idx array
    all_vals = []
    for s in lines:
        assert len(s) == len(seq)
        # convert to ct file (add a fake energy, otherwise b2ct won't run)
        input_str = '>seq\n{}\n{} (-0.0)'.format(seq, s)
        p = Popen(['b2ct'], stdin=PIPE,
                  stdout=PIPE, stderr=PIPE, universal_newlines=True)
        stdout, stderr = p.communicate(input=input_str)
        rc = p.returncode
        if rc != 0:
            msg = 'b2ct returned error code %d\nstdout:\n%s\nstderr:\n%s\n' % (
                rc, stdout, stderr)
            raise Exception(msg)
        # load data
        df = pd.read_csv(StringIO(stdout), skiprows=1, header=None,
                         names=['i1', 'base', 'idx_i', 'i2', 'idx_j', 'i3'],
                         sep=r"\s*")
        assert ''.join(df['base'].tolist()) == seq, ''.join(df['base'].tolist())
        # matrix
        vals = np.zeros((len(seq), len(seq)))
        for _, row in df.iterrows():
            idx_i = row['idx_i']
            idx_j = row['idx_j'] - 1
            if idx_j != -1:
                vals[idx_i, idx_j] = 1
                vals[idx_j, idx_i] = 1
        all_vals.append(vals)
    return all_vals


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


def process_row_mfe(seq, one_idx):
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


def process_row_sample_struct(seq, one_idx, n_sample):
    target = np.zeros((len(seq), len(seq)))
    target[one_idx] = 1

    preds = sample_structures(seq, n_sample)

    pred_idx = []
    sensitivity = []
    ppv = []
    f_measure = []
    for pred in preds:
        # set lower triangular to 0
        pred[np.tril_indices(pred.shape[0])] = 0
        _sensitivity = EvalMetric.sensitivity(pred, target)
        _ppv = EvalMetric.ppv(pred, target)
        _f_measure = EvalMetric.f_measure(_sensitivity, _ppv)
        _pred_idx = np.where(pred == 1)
        pred_idx.append(_pred_idx)
        sensitivity.append(_sensitivity)
        ppv.append(_ppv)
        f_measure.append(_f_measure)
    return pred_idx, sensitivity, ppv, f_measure


def main(dataset_file, output, n_sample):
    df = pd.read_pickle(dataset_file)

    df = add_columns(df, ['mfe_pred_idx', 'mfe_sensitivity', 'mfe_ppv',
                          'mfe_f_measure', 'mfe_fe', 'mfe_freq', 'ens_div'],
                     ['seq', 'one_idx'],
                     lambda x, y: process_row_mfe(x, y))

    # sample structure
    df = add_columns(df, ['pred_idx', 'sensitivity', 'ppv', 'f_measure'],
                     ['seq', 'one_idx'], lambda x, y: process_row_sample_struct(x, y, n_sample))

    df.to_pickle(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path to dataset file')
    parser.add_argument('--output', type=str, help='path to output file')
    parser.add_argument('--samples', type=int, help='number of structures to sample')
    args = parser.parse_args()

    main(args.dataset, args.output, args.samples)
