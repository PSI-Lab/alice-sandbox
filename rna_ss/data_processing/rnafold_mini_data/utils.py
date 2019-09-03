import math
import errno
import logging
import os
import re
import shutil
import tempfile
import itertools
from collections import Iterable
from past.builtins import basestring
import subprocess
from subprocess import PIPE, Popen
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
import uuid


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


def get_pair_prob_matrix(seq):
    # get pair prob matrix
    # no restriction on length
    tmp_dir = tempfile.mkdtemp()
    p = Popen(['RNAfold', '--MEA'],
              cwd=tmp_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    stdout, stderr = p.communicate(input=seq)
    rc = p.returncode
    if rc != 0:
        msg = 'RNAplfold returned error code %d\nstdout:\n%s\nstderr:\n%s\n' % (
            rc, stdout, stderr)
        raise Exception(msg)
    out_filename = os.path.join(tmp_dir, 'dot.ps')
    with open(out_filename) as fp:
        lines = fp.read().splitlines()
    vals = np.zeros((len(seq), len(seq)))
    for line in lines:
        if not line.endswith('ubox') or 'sqrt' in line:  # skip non data, or data header
            continue
        p1, p2, sqrt_prob, _ = line.split(' ')
        p1 = int(p1) - 1
        p2 = int(p2) - 1
        vals[p1, p2] = float(sqrt_prob) ** 2
        vals[p2, p1] = float(sqrt_prob) ** 2
    return vals


def get_pair_prob_arr(seq):
    # for odd length sequence
    # get array of pair probability for every position with the center position
    assert len(seq) % 2 == 1

    tmp_dir = tempfile.mkdtemp()
    p = Popen(['RNAfold', '--MEA'],
              cwd=tmp_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    stdout, stderr = p.communicate(input=seq)
    rc = p.returncode
    if rc != 0:
        msg = 'RNAplfold returned error code %d\nstdout:\n%s\nstderr:\n%s\n' % (
            rc, stdout, stderr)
        raise Exception(msg)
    out_filename = os.path.join(tmp_dir, 'dot.ps')
    with open(out_filename) as fp:
        lines = fp.read().splitlines()
    assert len(seq) % 2 == 1
    position_of_interest = len(seq) // 2 + 1  # 1-based
    vals = np.zeros(len(seq))
    for line in lines:
        if not line.endswith('ubox') or 'sqrt' in line:  # skip non data, or data header
            continue
        # if not line.startswith(str(position_of_interest)):
        #     continue
        p1, p2, sqrt_prob, _ = line.split(' ')
        p1 = int(p1)
        p2 = int(p2)
        # assert int(p1) == position_of_interest
        if p1 == position_of_interest:
            vals[p2 - 1] = float(sqrt_prob) ** 2
        elif p2 == position_of_interest:
            vals[p1 - 1] = float(sqrt_prob) ** 2
    return vals.tolist()


def one_idx(arr):
    assert len(arr.shape) == 2
    assert arr.shape[0] == arr.shape[1]
    assert np.all((arr == 0) | (arr == 1))
    return np.where(arr == 1)
