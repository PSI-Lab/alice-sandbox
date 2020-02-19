import os
import numpy as np
import pandas as pd
from utils import EvalMetric


def process_labels(fname):
    df = pd.read_csv(fname, sep="\s", skiprows=1)
    assert 'i' in df.columns, df.columns
    assert 'j' in df.columns, df.columns
    # convert to 0-based
    df['i'] = df['i'] - 1
    df['j'] = df['j'] - 1
    # validate (only on non-empty df)
    if len(df) > 0:
        assert df['i'].min() >= 0, fname
        assert df['j'].min() >= 0, fname
    # can't check max since we don't know sequence length...
    return [df['i'].tolist(), df['j'].tolist()]


def process_pred(fname):
    df = pd.read_csv(fname, sep="\s", skiprows=1, header=None, names=['i', 'base', 'j'])
    # sequence length
    seq_len = len(df)
    # those bases without any pairing have j = 0, drop them
    df = df[df['j'] > 0]
    # convert to 0-based
    df['i'] = df['i'] - 1
    df['j'] = df['j'] - 1
    # validate (only on non-empty df)
    if len(df) > 0:
        assert df['i'].min() >= 0, fname
        assert df['j'].min() >= 0, fname
    return [df['i'].tolist(), df['j'].tolist()], seq_len


# load labels
labels = {}
for dirpath, dirnames, filenames in os.walk('label/ts1/'):
    for filename in filenames:
        fname = os.path.join(dirpath, filename)
        labels[filename] = process_labels(fname)

# load predictions
preds = {}
seq_lens = {}
for dirpath, dirnames, filenames in os.walk('pred/ts1/'):
    for filename in filenames:
        fname = os.path.join(dirpath, filename)
        pred, seq_len = process_pred(fname)
        seq_id = filename.replace('.bpseq', '')
        preds[seq_id] = pred
        seq_lens[seq_id] = seq_len

    # make sure they have the same sequences
assert set(labels.keys()) == set(preds.keys())

# make into a df
df = []
for seq_id in labels.keys():
    df.append({'seq_id': seq_id,
               'label': labels[seq_id],
               'pred': preds[seq_id],
               'len': seq_lens[seq_id]})
df = pd.DataFrame(df)

# eval
eval = EvalMetric(bypass_pairing_check=True)
df_metric = []
for _, row in df.iterrows():
    seq_id = row['seq_id']
    seq_len = row['len']
    _label = row['label']
    _pred = row['pred']

    # make 2D array
    label = np.zeros((seq_len, seq_len))
    label[_label] = 1
    pred = np.zeros((seq_len, seq_len))
    pred[_pred] = 1

    # metric
    sensitivity = eval.sensitivity(pred, label)
    ppv = eval.ppv(pred, label)
    f_measure = eval.f_measure(sensitivity, ppv)
    df_metric.append({'seq_id': seq_id,
                      'len': seq_len,
                      'sensitivity': sensitivity,
                      'ppv': ppv,
                      'f_measure': f_measure})
df_metric = pd.DataFrame(df_metric)
print(df_metric.describe())

# report performance, compare with that in the paper
