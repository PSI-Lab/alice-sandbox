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
from subprocess import PIPE, Popen
import multiprocessing as mp
import numpy as np


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

