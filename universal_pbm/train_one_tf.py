"""
Report cross validation and test performance for one TF dataset (one file).
"""
import sys
tf_name = sys.argv[1]
file_name = sys.argv[2]
in_dir = sys.argv[3]
out_dir = sys.argv[4]
gpu_id = int(sys.argv[5])
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # do not remove, this is to turn off X server so plot works on Linux
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
plt.ioff()
from scipy import stats
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
import cufflinks as cf
cf.go_offline()
cf.set_config_file(theme='ggplot')
import plotly
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from dgutils.pandas import add_column, add_columns
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import regularizers
import keras.backend as kb
from keras.losses import mean_squared_error, mean_absolute_error

assert file_name.endswith('.txt')

df = pd.read_csv(os.path.join(in_dir, tf_name, file_name),
                 names=['intensity', 'sequence'], delim_whitespace=True)

out_path = os.path.join(out_dir, tf_name)
if not os.path.isdir(out_path):
    os.mkdir(out_path)
out_file_metric = os.path.join(out_path, file_name.replace('.txt', '.csv'))
out_file_plot = os.path.join(out_path, file_name.replace('.txt', '.html'))


def _process_seq(s):
    if 'GTCTGTGTTCCGTTGTCCGTGCTG' in s:
        return s.replace('GTCTGTGTTCCGTTGTCCGTGCTG', '')
    else:
        return None


# if sequence length is 60, trim primer suffix
if len(df.iloc[0]['sequence']) == 60:
    df = add_column(df, 'sequence', ['sequence'], _process_seq)
    # drop those entries where primer is not present
    print("dropping %d rows" % len(df[df['sequence'].isna()]))
    print("before %d" % len(df))
    df = df.dropna(subset=['sequence'])
    print("after %d" % len(df))

# log intensity
df = add_column(df, 'log_intensity', ['intensity'], lambda x: np.log(x))

# some sequence might be of different length, drop them
print("lengths: %s" % df.sequence.str.len().unique())
print("median length: %d" % df.sequence.str.len().median())
print("Drop non-median length rows, before: %d" % len(df))
df = df[df.sequence.str.len() == df.sequence.str.len().median()]
print("After: %d" % len(df))


# split training/validation + testing
train_mask = np.random.rand(len(df)) < 0.8
df_train = df[train_mask]
df_test = df[~train_mask]

# encode data
IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
_data_input = []
_data_output = []
for i, row in df_train.iterrows():
    seq = row['sequence']
    seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')
    x = np.asarray(map(int, list(seq)))
    x = IN_MAP[x.astype('int8')]
    _data_input.append(x)

    _val = [row['log_intensity']]
    _data_output.append(_val)

X_train = np.swapaxes(np.swapaxes(np.stack(_data_input, axis=2), 0, 2), 1, 2)
Y_train = np.swapaxes(np.stack(_data_output, axis=1), 0, 1)

print('Training data: ', X_train.shape, Y_train.shape)

_data_input = []
_data_output = []
for i, row in df_test.iterrows():
    seq = row['sequence']
    seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')
    x = np.asarray(map(int, list(seq)))
    x = IN_MAP[x.astype('int8')]
    _data_input.append(x)

    _val = [row['log_intensity']]
    _data_output.append(_val)

X_test = np.swapaxes(np.swapaxes(np.stack(_data_input, axis=2), 0, 2), 1, 2)
Y_test = np.swapaxes(np.stack(_data_output, axis=1), 0, 1)

print('Test data: ', X_test.shape, Y_test.shape)


filter_param = [(100, 8, 1), (50, 8, 2)]


def conv_model(n_out=Y_train.shape[1]):
    model = Sequential()
    for i, (n_filter, filter_width, dilation_rate) in enumerate(filter_param):
        model.add(Conv1D(filters=n_filter, kernel_size=filter_width, strides=1, padding='valid',
                         dilation_rate=dilation_rate, activation='relu', use_bias=True,
                        name='conv_%d' % i))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(n_out, kernel_initializer='normal'))
    model.compile(loss=mean_squared_error, optimizer='adam')
#     opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
#                                     amsgrad=False)
#     model.compile(loss=mean_absolute_error, optimizer=opt)
    return model


estimator = KerasRegressor(build_fn=conv_model, epochs=100,       ## TODO decrease LR, increase epoch
                           batch_size=500, verbose=2)
kfold = KFold(n_splits=5, random_state=1234, shuffle=True)
result = cross_validate(estimator, X_train, Y_train, cv=kfold,
                        return_estimator=True, return_train_score=True,
                        scoring=('r2', 'neg_mean_squared_error'))


df_metric = []
fig = plotly.tools.make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True)

# cross validation
y_pred = np.empty(Y_train.shape)
for i, (train_index, test_index) in enumerate(kfold.split(X_train)):
    xt = X_train[test_index, :, :]
    y_pred[test_index, 0] = result['estimator'][i].predict(xt)
df_plot = pd.DataFrame({'target': Y_train[:, 0], 'pred': y_pred[:, 0]})

corr, pval = pearsonr(df_plot['target'], df_plot['pred'])
print('CV', corr, pval)
df_metric.append({'task': 'cross_validation',
                  'pearson_corr': corr,
                  'pearson_pval': pval})

fig.append_trace(plotly.graph_objs.Scatter(
    x=df_plot['pred'],
    y=df_plot['target'],
    name='cross-validation pearson corr: %.4f (%e)' % (corr, pval),
    mode='markers',
    marker={'size': 1},
), 1, 1)

# test set
y_pred = np.mean(np.stack([result['estimator'][i].predict(X_test) for i in range(len(result['estimator']))], axis=1), axis=1)

df_plot = pd.DataFrame({'target': Y_test[:, 0], 'pred': y_pred})

corr, pval = pearsonr(df_plot['target'], df_plot['pred'])
print('test set', corr, pval)
df_metric.append({'task': 'cross_validation',
                  'pearson_corr': corr,
                  'pearson_pval': pval})

fig.append_trace(plotly.graph_objs.Scatter(
    x=df_plot['pred'],
    y=df_plot['target'],
    name='test set pearson corr: %.4f (%e)' % (corr, pval),
    mode='markers',
    marker={'size': 1},
), 2, 1)

fig['layout'].update({'title': 'tmp',
                      'xaxis':
                          {
                              'title': 'prediction',
                          },
                      'yaxis':
                          {
                              'title': 'target',
                          }, })

pd.DataFrame(df_metric).to_csv(out_file_metric, index=False)
plotly.offline.plot(fig, filename=out_file_plot, auto_open=False)
