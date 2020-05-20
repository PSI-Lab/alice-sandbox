import os
import re
import tempfile
import argparse
import numpy as np
import pandas as pd


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
    # return upper triangular indexes
    idxes = [[], []]   # compatible withour internal format (list_of_i, list_of_j)
    for _, row in df.iterrows():
        idx_i = row['idx_i']
        idx_j = row['idx_j'] - 1
        if idx_j != -1:
            idxes[0].append(idx_i)
            idxes[1].append(idx_j)
    return idxes
    # # matrix
    # vals = np.zeros((len(seq), len(seq)))
    # for _, row in df.iterrows():
    #     idx_i = row['idx_i']
    #     idx_j = row['idx_j'] - 1
    #     if idx_j != -1:
    #         vals[idx_i, idx_j] = 1
    #         vals[idx_j, idx_i] = 1
    # return vals, fe, mfe_freq, ens_div


def main(df_in, out_file):
    df_out = []
    for seq in df_in['seq']:
        idxes = get_fe_struct(seq)
        df_out.append({
            'seq': seq,
            'pred_idx': idxes,
        })
    df_out = pd.DataFrame(df_out)
    # TODO need to pickle since output contains list
    df_out.to_pickle(args.out_file, protocol=2)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--in_file', type=str,
        help='path to input csv/pkl file (with column "seq")'
    )
    argparser.add_argument(
        '--out_file', type=str,
        help='path to output csv file'
    )
    argparser.add_argument(
        '--format', type=str,
        help='input format, csv or pkl'
    )
    args = argparser.parse_args()

    in_format = args.format
    if in_format == 'csv':
        df_in = pd.read_csv(args.in_file)
    elif in_format == 'pkl':
        df_in = pd.read_pickle(args.in_file)
    else:
        raise ValueError(in_format)

    main(df_in, args.out_file)

