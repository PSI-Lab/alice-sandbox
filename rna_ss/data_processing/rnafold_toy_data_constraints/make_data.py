# generate short rand sequence
# toy dataset
# with constraint that certain motifs are unpaired
import argparse
import random
import re
import tempfile
import numpy as np
import cPickle as pickle
from StringIO import StringIO
from subprocess import PIPE, Popen
import pandas as pd
from dgutils.pandas import add_columns, add_column


def generate_reactivity_data(seq, accessible_motifs, accessible_prob=1.0):
    reactivities = np.empty(len(seq))
    reactivities[:] = np.nan
    for motif in accessible_motifs:
        for m in re.finditer(motif, seq):
            if accessible_prob == 1:
                reactivities[m.start():m.start()+len(motif)] = 2.0
            else:
                # TODO sample according to prob
                raise NotImplementedError
    df = pd.DataFrame({
        'position': range(1, len(seq) + 1),
        'reactivity': reactivities,
    })
    # keep rows with value
    df = df.dropna()
    # print(df)
    return df


def sample_structures(seq, n_samples, accessible_motifs=None, accessible_prob=1.0, unique=True):
    if accessible_motifs:
        df_reactivities = generate_reactivity_data(seq, accessible_motifs, accessible_prob)
        with tempfile.NamedTemporaryFile() as f:
            # print(df_reactivities)
            df_reactivities.to_csv(f.name, sep='\t', header=False, index=False)

            p = Popen(['RNAsubopt', '-p', str(n_samples),
                       '--shape={}'.format(f.name)], stdin=PIPE,
                      stdout=PIPE, stderr=PIPE, universal_newlines=True)
    else:
        p = Popen(['RNAsubopt', '-p', str(n_samples)], stdin=PIPE,
                  stdout=PIPE, stderr=PIPE, universal_newlines=True)
    stdout, stderr = p.communicate(input=seq)
    rc = p.returncode
    if rc != 0:
        msg = 'RNAsubopt returned error code %d\nstdout:\n%s\nstderr:\n%s\n' % (
            rc, stdout, stderr)
        raise Exception(msg)

    # parse output
    lines = stdout.splitlines()
    assert len(lines) == n_samples + 1
    lines = lines[1:]
    # convert to idx array
    one_idx = []
    prob_pair = []
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
        # matrix & paired prob
        vals_mat = np.zeros((len(seq), len(seq)))
        is_paired = np.zeros(len(seq))
        for _, row in df.iterrows():
            idx_i = row['idx_i']
            idx_j = row['idx_j'] - 1
            if idx_j != -1:
                vals_mat[idx_i, idx_j] = 1
                vals_mat[idx_j, idx_i] = 1
                is_paired[idx_i] = 1
                is_paired[idx_j] = 1
        # process into one-index
        # set lower triangular to 0
        vals_mat[np.tril_indices(vals_mat.shape[0])] = 0
        pred_idx = np.where(vals_mat == 1)
        one_idx.append(pred_idx)
        # add to pair prob
        prob_pair.append(is_paired)
    # compute pair prob
    prob_pair = np.asarray(prob_pair)
    prob_pair = np.mean(prob_pair, axis=0)
    # keep unique struct only
    if unique:
        one_idx = unique_struct(one_idx)
    return one_idx, prob_pair


def unique_struct(one_idx):
    # return set(one_idx)  # TODO does this work?
    df = pd.DataFrame(one_idx, columns=['left', 'right'])
    df = add_column(df, 'left', ['left'], lambda x: tuple(x))
    df = add_column(df, 'right', ['right'], lambda x: tuple(x))
    # get unique rows and their count
    df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'count'})
    return [tuple(x) for x in df.values]


def main(seq_len, num_seqs, num_structs_per_seq, outfile):
    # other parameters are hard-coded for now
    seqs = []  # TODO use set so we don't add duplicates (but the probability of getting duplicate is very low)
    for _ in range(num_seqs):
        seq = ''.join(random.choice(list('ACGU')) for _ in range(seq_len))
        seqs.append(seq)
    print("Running rnafold...")
    df = pd.DataFrame({'seq': seqs})
    # without constraints
    df = add_columns(df, ['one_idx_nc', 'pair_prob_nc'], ['seq'],
                     lambda x: sample_structures(x, num_structs_per_seq))
    # with constraints
    df = add_columns(df, ['one_idx_wc', 'pair_prob_wc'], ['seq'],
                     lambda x: sample_structures(x, num_structs_per_seq, accessible_motifs=['ACCA']))  # TODO input
    # df.to_pickle(outfile, compression='gzip')
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--len', type=int, default=20, help='sequence length')
    parser.add_argument('--num', type=int, default=100, help='total number of sequences to generate')
    parser.add_argument('--ns', type=int, default=100, help='number of structures per sequence to sample')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()
    main(args.len, args.num, args.ns,  args.out)


# # test
# one_idx, prob_pair = sample_structures('CGGCUCGCAACAGACCUAUUAGU',
#                   100, ['CUC'], accessible_prob=1.0)
#
# for x in unique_struct(one_idx):
#     print(x)
#
# print(prob_pair)
#
#
# one_idx, prob_pair = sample_structures('CGGCUCGCAACAGACCUAUUAGU', 100)
#
# for x in unique_struct(one_idx):
#     print(x)
#
# print(prob_pair)
