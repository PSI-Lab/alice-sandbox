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
    tmp_dir = tempfile.mkdtemp()
    file_prefix = uuid.uuid4()
    # input_file = os.path.join(tmp_dir, '{}.input'.format(file_prefix))
    # output_file = os.path.join(tmp_dir, '{}.output'.format(file_prefix))
    input_file = tempfile.NamedTemporaryFile().name
    output_file = tempfile.NamedTemporaryFile().name
    # print(output_file.name)
    with open(input_file, 'w') as f:
        f.write('>seq\n' + seq)
    # p = Popen(['RNAfold', '-p', '<', input_file,  '|', 'b2ct', '>', output_file],
    #           cwd=tmp_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    # stdout, stderr = p.communicate()
    # p = Popen(['b2ct'],
    #           cwd=tmp_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    # p = Popen(['b2ct',  '>', output_file.name],
    #           cwd=tmp_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    # stdout, stderr = p.communicate('RNAfold -p < {}'.format(input_file))
    # rc = p.returncode
    # if rc != 0:
    #     msg = 'RNAplfold returned error code %d\nstdout:\n%s\nstderr:\n%s\n' % (
    #         rc, stdout, stderr)
    #     raise Exception(msg)

    cmd = 'RNAfold -p < {} | b2ct > {}'.format(input_file, output_file)
    # print(cmd)
    os.system(cmd)

    # load header
    with open(output_file, 'r') as fp:
        tmp = fp.read()
        # print(tmp)
        lines = tmp.splitlines()
    # lines = stdout.splitlines()
    # print(stdout)
    # print(lines)
    seq_len = int(re.match(string=lines[0], pattern=r'\s+(\d+)\s+ENERGY =(.*)seq').group(1))
    fe = float(re.match(string=lines[0], pattern=r'\s+(\d+)\s+ENERGY =(.*)seq').group(2).strip())
    assert len(seq) == seq_len
    # load data
    df = pd.read_csv(output_file, skiprows=1, header=None,
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
    return vals, fe


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

