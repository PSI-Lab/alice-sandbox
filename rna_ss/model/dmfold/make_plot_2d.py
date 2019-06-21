import sys
import numpy as np
import datacorral as dc
import keras
from keras import backend as kb
from train_2d import custom_loss
import cPickle as pickle
import random
from sklearn.metrics import accuracy_score
from train_2d import DataGenerator
from utils import split_data_by_rna_type
import matplotlib
matplotlib.use('Agg')   # do not remove, this is to turn off X server so plot works on Linux
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
plt.ioff()
import cufflinks as cf
cf.go_offline()
cf.set_config_file(theme='ggplot')
import plotly
import plotly.graph_objs as go
import plotly.io as pio


split = sys.argv[1]


class Predictor2D(object):
    # works with sequence of any length! =)
    DNA_ENCODING = np.zeros((5, 4))
    DNA_ENCODING[1:, :] = np.eye(4)

    def __init__(self, model_file):
        self.model = keras.models.load_model(model_file, custom_objects={"kb": kb, "custom_loss": custom_loss})

    def predict_seq(self, seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('U', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = self.DNA_ENCODING[x.astype('int8')]
        _val_h = np.repeat(x[:, np.newaxis, :], len(seq), axis=1)
        _val_v = np.repeat(x[np.newaxis, :, :], len(seq), axis=0)
        _tmp_seq_map = np.concatenate((_val_h, _val_v), axis=2)
        _seq_map = _tmp_seq_map[np.newaxis, :, :, :]
        yp = self.model.predict(_seq_map)[0, :, :]
        return yp


def get_prediction(predictor, seq, pair_pos):
    assert all([x >= 0 for x in pair_pos])
    pair_pos = [x - 1 for x in pair_pos]

    # paired positions
    #     print pair_pos

    pair_map = np.zeros((len(seq), len(seq)))
    for i, j in enumerate(pair_pos):
        if j == -1:  # unpaired
            continue
        else:
            if i < j:
                # only set value in upper triangular matrix
                pair_map[i, j] = 1
    # mask lower diagonal part
    il1 = np.tril_indices(len(seq))
    pair_map[il1] = -1

    yp = predictor.predict_seq(seq)

    # set lower diagonal of yp_sliced to -1
    yp[np.tril_indices(yp.shape[0])] = -1

    # find locaitons where pair_map == 1
    idx_1 = np.where(pair_map == 1)

    idx_1_row = np.asarray(list(set(idx_1[0])), dtype=int)
    y_sliced = pair_map[idx_1_row, :]
    yp_sliced = yp[idx_1_row, :]

    # compute loss and accuracy for each row
    # losses.append(log_loss(y_sliced, yp_sliced))
    accuracy = accuracy_score(np.argmax(y_sliced, axis=1),
                              np.argmax(yp_sliced, axis=1))

    # also compute yp wit hard-max for rows with paired position
    yp_max = np.zeros(yp.shape)
    yp_max[range(yp.shape[0]), np.argmax(yp, axis=1)] = 1
    tmp = pair_map.copy()
    tmp[np.where(tmp == -1)] = 0
    yp_max[np.where(np.sum(tmp, axis=1) == 0), :] = 0  # set rows without paired position to all 0's
    yp_max[np.tril_indices(yp.shape[0])] = -1  # set lower diagonal to -1

    return seq, pair_pos, pair_map, yp, yp_max, accuracy


def make_heatmap(seq, pair_map, y_pred, y_pred_max, accuracy, out_prefix):
    # convert non missing val to logit
    y_pred_logit = y_pred.copy()
    y_pred_logit[np.where(y_pred_logit == -1)] = -20
    y_pred_logit[np.where(y_pred_logit != -20)] = np.log(
        y_pred_logit[np.where(y_pred_logit != -20)] / (1 - y_pred_logit[np.where(y_pred_logit != -20)]))

    # row/col axis labels
    axis_labels = ['%d%s' % (i, x) for i, x in enumerate(seq)]

    trace0 = go.Heatmap(z=y_pred, x=axis_labels, y=axis_labels, colorscale='Electric', name='Prediction',
                        showscale=False)
    trace1 = go.Heatmap(z=pair_map, x=axis_labels, y=axis_labels, colorscale='Greys', name='Target', showscale=False)
    trace2 = go.Heatmap(z=y_pred_logit, x=axis_labels, y=axis_labels, colorscale='Electric', name='Prediction',
                        showscale=False)
    trace3 = go.Heatmap(z=y_pred_max, x=axis_labels, y=axis_labels, colorscale='Electric', name='Prediction',
                        showscale=False)

    fig = plotly.tools.make_subplots(rows=2, cols=2, subplot_titles=(
        'Prediction - probability', 'Target', 'Prediction - logit', 'Prediction - binary'))

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace3, 2, 2)

    layout = dict(
        title='Accuracy: {0:.2%}'.format(accuracy),
        xaxis=dict(title='seq'),
        yaxis=dict(title='seq', autorange='reversed'),

        xaxis2=dict(title='seq'),
        yaxis2=dict(title='seq', autorange='reversed'),

        xaxis3=dict(title='seq'),
        yaxis3=dict(title='seq', autorange='reversed'),

        xaxis4=dict(title='seq'),
        yaxis4=dict(title='seq', autorange='reversed'),

        height=800,
        width=800,
    )
    fig['layout'].update(layout)
    plotly.offline.plot(fig, filename='%s.html' % out_prefix, auto_open=False)
    pio.write_image(fig, '%s.png' % out_prefix)  # note that this require X11 if running on Linux


with open('data/data.pkl', 'rb') as f:
    data_all = pickle.load(f)

if split == 'random':
    # note that this won't reproduce the split during training since we didn't fix the rand seed =(...
    # not a big problem for now since training and validation performance are close enough
    # TODO can be fixed in next iteration
    random.shuffle(data_all)
    num_training = int(len(data_all) * 0.8)
    data_validation = data_all[num_training:]
    datagen_v = DataGenerator(data_validation)
else:
    data_dict = split_data_by_rna_type(data_all)
    assert split in data_dict.keys()
    data_validation = data_dict[split]
    data_training = [data_dict[x] for x in data_dict.keys() if x != split]
    print("validation RNA type: %s" % split)
    print("validation RNAs: %d" % len(data_validation))
    datagen_v = DataGenerator(data_validation)

# load model
split2id = {
    'random': 'nIHGC4',
    '5s': '3oDyXh',
    'rnasep': 'uVQluV',
    'tmrna': 'k7o1cE',
    'trna': '1UIbXA',
}
predictor = Predictor2D(dc.Client().get_path(split2id[split]))

# find min, max and median (need to run this on GPU, otherwise too slow...)
acc = []
for idx in range(len(datagen_v.data)):
    _d = datagen_v.data[idx]
    _seq = _d[0]
    _pair_pos = _d[2]
    seq, pair_pos, pair_map, yp, yp_max, accuracy = get_prediction(predictor, _seq, _pair_pos)
    acc.append((idx, accuracy))
    print "%d/%d, %f" % (idx, len(datagen_v.data), accuracy)

idx_sorted = sorted(acc, key=lambda x: x[1])
idx_max = idx_sorted[-1][0]
idx_min = idx_sorted[0][0]
idx_median = idx_sorted[len(idx_sorted)/2][0]
print idx_max, idx_min, idx_median

for idx, label in zip([idx_max, idx_min, idx_median], ['max', 'min', 'median']):
    out_prefix = 'plot/heatmap_%s_%s' % (split, label)
    _d = datagen_v.data[idx]
    _seq = _d[0]
    _pair_pos = _d[2]
    seq, pair_pos, pair_map, yp, yp_max, accuracy = get_prediction(predictor, _seq, _pair_pos)
    make_heatmap(seq, pair_map, yp, yp_max, accuracy, out_prefix)
    print out_prefix
