import os
import argparse
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from scipy.special import softmax
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _make_mask(l):
    m = np.ones((l, l))
    m[np.tril_indices(l)] = 0
    return m


def make_plot_bb(target, pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y=None, title=None):
    thres = 0.5   # hard-coded threshold
    if pred_siz_y is None:
        pred_siz_y = pred_siz_x.copy()  # same size
    tmp = np.sum(target[0, :, :], axis=0)
    first_nonzero_idx_from_right = next(i for i in range(len(tmp)) if tmp[-(i + 1)] > 0)
    # crop
    if first_nonzero_idx_from_right == 0:  # can't index by -0 <- will be empty!
        # no crop
        pass
    else:
        target = target[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
        pred_on = pred_on[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
        pred_loc_x = pred_loc_x[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
        pred_loc_y = pred_loc_y[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
        pred_siz_x = pred_siz_x[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
        pred_siz_y = pred_siz_y[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]

    # apply 'hard-mask'  (lower triangle)
    assert target.shape == pred_on.shape  # channel x h x w
    assert target.shape[1] == target.shape[2]
    m = _make_mask(target.shape[1])
    # apply mask (for pred, only apply to pred_on since our processing starts from that array)
    target = target[0, :, :] * m
    pred_on = pred_on[0, :, :] * m

    fig = px.imshow(target)

    for i, j in np.transpose(np.where(pred_on > thres)):
        loc_x = np.argmax(pred_loc_x[:, i, j])
        loc_y = np.argmax(pred_loc_y[:, i, j])
        siz_x = np.argmax(pred_siz_x[:, i, j]) + 1  # size starts at 1 for index=0
        siz_y = np.argmax(pred_siz_y[:, i, j]) + 1
        # compute joint probability of taking the max value
        prob = pred_on[i, j] * softmax(pred_loc_x[:, i, j])[loc_x] * softmax(pred_loc_y[:, i, j])[loc_y] * \
               softmax(pred_siz_x[:, i, j])[siz_x - 1] * softmax(pred_siz_y[:, i, j])[siz_y - 1]  # FIXME multiplying twice for case where y is set to x
        # top right corner
        bb_x = i - loc_x
        bb_y = j + loc_y
        #     print(bb_x, bb_y, siz)
        # top left corner (for plot)
        x0 = bb_x
        y0 = bb_y - siz_y
        wx = siz_x
        wy = siz_y
        fig.add_shape(
            type='rect',
            y0=x0 - 0.5, y1=x0 + wx - 0.5, x0=y0 - 0.5, x1=y0 + wy - 0.5,  # image plot axis is swaped
            xref='x', yref='y',
            opacity=prob,  # opacity proportional to probability of bounding box
            line_color='red'
        )
        fig['layout'].update(height=800, width=800, title="{} threshold {}".format(title, thres))
    return fig


def make_plot_sigmoid(target, pred, title):
    tmp = np.sum(target[0, :, :], axis=0)
    first_nonzero_idx_from_right = next(i for i in range(len(tmp)) if tmp[-(i + 1)] > 0)
    # crop
    if first_nonzero_idx_from_right == 0:  # can't index by -0 <- will be empty!
        # no crop
        pass
    else:
        target = target[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
        pred = pred[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]

    # apply 'hard-mask'  (lower triangle)
    assert target.shape == pred.shape  # channel x h x w
    assert target.shape[1] == target.shape[2]
    m = _make_mask(target.shape[1])
    # repeat mask on channel
    # m = m[np.newaxis, :, :]
    # m = np.repeat(m, target.shape[0], axis=0)

    # apply mask
    target = target[0, :, :] * m
    pred = pred[0, :, :] * m

    fig = make_subplots(rows=1, cols=2, print_grid=False, shared_yaxes=True, shared_xaxes=True)

    # target
    tmp_fig = px.imshow(target)
    fig.append_trace(tmp_fig.data[0], 1, 1)
    fig['layout']['xaxis{}'.format(1)].update(title='target')
    fig['layout']['yaxis{}'.format(1)]['autorange'] = "reversed"
    # pred idx
    tmp_fig = px.imshow(pred)
    fig.append_trace(tmp_fig.data[0], 1, 2)
    fig['layout']['xaxis{}'.format(2)].update(title='pred')
    fig['layout']['yaxis{}'.format(1)]['autorange'] = "reversed"

    fig['layout'].update(height=400, width=400 * 2, title=title)

    fig['layout']['yaxis']['autorange'] = "reversed"

    return fig


def make_plot_softmax(target, pred, title):
    pred = softmax(pred, axis=0)

    tmp = np.sum(target[0, :, :], axis=0)
    first_nonzero_idx_from_right = next(i for i in range(len(tmp)) if tmp[-(i + 1)] > 0)
    # crop
    if first_nonzero_idx_from_right == 0:  # can't index by -0 <- will be empty!
        # no crop
        pass
    else:
        target = target[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
        pred = pred[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]

    # predicted index
    pred_idx = pred.argmax(axis=0)
    # predicted value
    pred_val = pred.max(axis=0)

    # apply 'hard-mask'  (lower triangle)
    # assert target.shape == pred.shape  # channel x h x w
    assert target.shape[1] == target.shape[2]
    m = _make_mask(target.shape[1])
    # # repeat mask on channel
    # m = m[np.newaxis, :, :]
    # m = np.repeat(m, target.shape[0], axis=0)

    # apply mask
    target = target[0, :, :] * m
    pred_idx = pred_idx * m
    pred_val = pred_val * m

    fig = make_subplots(rows=1, cols=3, print_grid=False, shared_yaxes=True, shared_xaxes=True)

    # target
    tmp_fig = px.imshow(target)
    fig.append_trace(tmp_fig.data[0], 1, 1)
    fig['layout']['xaxis{}'.format(1)].update(title='target')
    fig['layout']['yaxis{}'.format(1)]['autorange'] = "reversed"
    # pred idx
    tmp_fig = px.imshow(pred_idx)
    fig.append_trace(tmp_fig.data[0], 1, 2)
    fig['layout']['xaxis{}'.format(2)].update(title='pred_max_idx')
    fig['layout']['yaxis{}'.format(1)]['autorange'] = "reversed"
    # pred val
    tmp_fig = px.imshow(pred_val)
    fig.append_trace(tmp_fig.data[0], 1, 3)
    fig['layout']['xaxis{}'.format(3)].update(title='pred_max_val')
    fig['layout']['yaxis{}'.format(1)]['autorange'] = "reversed"

    fig['layout'].update(height=400, width=400 * 3, title=title)

    fig['layout']['yaxis']['autorange'] = "reversed"

    return fig


def main(in_file, out_file):
    df = pd.read_pickle(in_file)

    # col -> plot_func
    col2plot = OrderedDict()
    col2plot['stem_on'] = make_plot_sigmoid
    col2plot['stem_location_x'] = make_plot_softmax
    col2plot['stem_location_y'] = make_plot_softmax
    col2plot['stem_size'] = make_plot_softmax

    # assert set(df.columns) == {'target', 'pred', 'subset'}
    # output to a single html
    with open(out_file, 'w') as f:
        # pick random index of training data point
        row_tr = df[df['subset'] == 'training'].sample(n=1).iloc[0]
        # pick random index of validation data point
        row_va = df[df['subset'] == 'validation'].sample(n=1).iloc[0]
        for row in [row_tr, row_va]:
            # per-output plot
            for col, func in col2plot.items():
                fig = func(row['target_{}'.format(col)], row['pred_{}'.format(col)],
                           '{} name: {} type: {}'.format(in_file, col, row.subset))
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            # proposed bounding box plot
            fig = make_plot_bb(row['target_stem_on'], row['pred_stem_on'],
                               row['pred_stem_location_x'], row['pred_stem_location_y'],
                               row['pred_stem_size'], pred_siz_y=None,
                               title='bounding box prediction')
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Path to input prediction file')
    parser.add_argument('--out_file', type=str, help='Path to output plot html file')
    args = parser.parse_args()

    assert args.out_file.endswith('.html')
    # make result dir if non existing
    out_path = os.path.dirname(args.out_file)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    main(args.in_file, args.out_file)
