import numpy as np
import pandas as pd
from dgutils.pandas import add_column
from utils import EvalMetric

# load result
df = pd.read_pickle('rfam_pred.pkl')

th = 0.003

# we stored our raw prediction, let's make it binary using an arbitrary threshold
df = add_column(df, 'pred', ['pred_val'], lambda x: (x > th).astype(np.int))

# eval
eval = EvalMetric(bypass_pairing_check=True)
df_metric = []
for _, row in df.iterrows():
    seq_id = row['seq_id']
    seq_len = row['len']
    _label = row['one_idx']
    pred = row['pred']

    label = np.zeros((seq_len, seq_len))
    label[_label] = 1

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
print(th)
print(df_metric.describe())

