# generate short rand sequence
# toy dataset
import argparse
import random
import numpy as np
import cPickle as pickle
from StringIO import StringIO
from subprocess import PIPE, Popen
import pandas as pd
from utils import get_fe_struct, one_idx
from dgutils.pandas import add_columns, add_column


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
        # process into one-index
        # set lower triangular to 0
        vals[np.tril_indices(vals.shape[0])] = 0
        pred_idx = np.where(vals == 1)
        all_vals.append(pred_idx)
    return all_vals


def main(minlen=10, maxlen=100, num_seqs=100, outfile=None):
    assert outfile
    seqs = []  # TODO use set so we don't add duplicates (but the probability of getting duplicate is very low)
    for _ in range(num_seqs):
        seq = ''.join(random.choice(list('ACGU')) for _ in range(random.randint(minlen, maxlen)))
        seqs.append(seq)
    print("Running rnafold...")
    df = pd.DataFrame({'seq': seqs})
    df = add_column(df, 'one_idx', ['seq'], lambda x: sample_structures(x, num_seqs))
    df.to_pickle(outfile, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--minlen', type=int, default=10, help='min sequence length')
    parser.add_argument('--maxlen', type=int, default=100, help='max sequence length')
    parser.add_argument('--num', type=int, default=100, help='total number of sequences to generate')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()
    main(args.minlen, args.maxlen, args.num, args.out)


