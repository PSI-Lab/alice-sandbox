import re
import argparse
import numpy as np
import pandas as pd
from StringIO import StringIO
from subprocess import PIPE, Popen
from utils import db_to_mat


def file_gen(file_name):
    with open(file_name, 'r') as f:
        x = []
        for line in f:
            line = line.rstrip()
            if line == '' and x:
                yield x
                x = []
                continue
            x.append(line)
        if x:
            yield x


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


def main(input_file, outfile, num_seqs):
    # input_file = 'raw_data/S-151Rfam.txt'
    data = []

    for lines in file_gen(input_file):
        assert lines[0].startswith('>')
        seq_id = re.match(string=lines[0],
                          pattern=r'^> (.*)$').group(1)
        seq = lines[1]
        db_notation = lines[2]
        assert all([x in list('ACGUacgu') for x in seq])
        arr = db_to_mat(seq, db_notation, upper_triangular=True)
        assert arr.shape[0] == len(seq)
        assert arr.shape[1] == len(seq)

        # full matrix is too big, since it's sparse, we'll save the index of 1's
        idxes = np.where(arr == 1)

        data.append({
            'seq_id': seq_id,
            'seq': seq,
            'len': len(seq),
            'truth_one_idx': idxes,
            'rnafold_one_idx': sample_structures(seq, num_seqs),
        })
        print(len(data), seq_id)

    data = pd.DataFrame(data)
    data.to_pickle(outfile, compression='gzip')
    # data.to_pickle('data/rfam.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, help='input file')
    parser.add_argument('--outfile', type=str, help='output file')
    parser.add_argument('--num', type=int, default=100, help='total number of sequences to generate')
    args = parser.parse_args()
    main(args.infile, args.outfile, args.num)
