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


def predict_bounidng_box(pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y=None, thres=0.5):
    # hard-mask
    m = _make_mask(pred_on.shape[1])
    # apply mask (for pred, only apply to pred_on since our processing starts from that array)
    pred_on = pred_on * m
    # binary array with all 0's, we'll set the predicted bounding box region to 1
    # this will be used to calculate 'sensitivity'
    pred_box = np.zeros_like(pred_on)
    # also save box locations and probabilities
    proposed_boxes = []

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
        # save box
        proposed_boxes.append({
            'bb_x': bb_x,
            'bb_y': bb_y,
            'siz_x': siz_x,
            'siz_y': siz_y,
            'prob': prob,   # TODO shall we store 4 probabilities separately?
        })
        # set value in pred box, be careful with out of bound index
        x0 = bb_x
        y0 = bb_y - siz_y + 1  # 0-based
        wx = siz_x
        wy = siz_y
        ix0 = max(0, x0)
        iy0 = max(0, y0)
        ix1 = min(x0 + wx, pred_box.shape[0])
        iy1 = min(y0 + wy, pred_box.shape[1])
        pred_box[ix0:ix1, iy0:iy1] = 1

    # apply hard-mask to pred box
    pred_box = pred_box * m
    return proposed_boxes, pred_box


def make_plot_bb(target, pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y=None, title=None, thres=0.5):
    if pred_siz_y is None:
        pred_siz_y = pred_siz_x.copy()  # same size
    tmp = np.sum(target[0, :, :], axis=0)
    try:
        first_nonzero_idx_from_right = next(i for i in range(len(tmp)) if tmp[-(i + 1)] > 0)
    except StopIteration:
        # target all 0, can't infer index, don't crop FIXME we should save original sequence length in df then we don't need to infer
        first_nonzero_idx_from_right = 0
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

    # # binary array with all 0's, we'll set the predicted bounding box region to 1
    # # this will be used to calculate 'sensitivity'
    # pred_box = np.zeros_like(target)
    # # also save box locations and probabilities
    # proposed_boxes = []

    fig = px.imshow(target)

    proposed_boxes, pred_box = predict_bounidng_box(pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y, thres)

    for bb in proposed_boxes:
        bb_x = bb['bb_x']
        bb_y = bb['bb_y']
        siz_x = bb['siz_x']
        siz_y = bb['siz_y']
        prob = bb['prob']

    # for i, j in np.transpose(np.where(pred_on > thres)):
    #     loc_x = np.argmax(pred_loc_x[:, i, j])
    #     loc_y = np.argmax(pred_loc_y[:, i, j])
    #     siz_x = np.argmax(pred_siz_x[:, i, j]) + 1  # size starts at 1 for index=0
    #     siz_y = np.argmax(pred_siz_y[:, i, j]) + 1
    #     # compute joint probability of taking the max value
    #     prob = pred_on[i, j] * softmax(pred_loc_x[:, i, j])[loc_x] * softmax(pred_loc_y[:, i, j])[loc_y] * \
    #            softmax(pred_siz_x[:, i, j])[siz_x - 1] * softmax(pred_siz_y[:, i, j])[siz_y - 1]  # FIXME multiplying twice for case where y is set to x
    #     # top right corner
    #     bb_x = i - loc_x
    #     bb_y = j + loc_y
    #     # print(bb_x, bb_y, siz_x, siz_y)
    #     # top left corner (for plot)
        x0 = bb_x
        y0 = bb_y - siz_y + 1  # 0-based
        wx = siz_x
        wy = siz_y
        fig.add_shape(
            type='rect',
            y0=x0 - 0.5, y1=x0 + wx - 0.5, x0=y0 - 0.5, x1=y0 + wy - 0.5,  # image plot axis is swaped
            xref='x', yref='y',
            opacity=prob,  # opacity proportional to probability of bounding box
            line_color='red'
        )

        # # save box
        # proposed_boxes.append({
        #     'bb_x': bb_x,
        #     'bb_y': bb_y,
        #     'siz_x': siz_x,
        #     'siz_y': siz_y,
        #     'prob': prob,   # TODO shall we store 4 probabilities separately?
        # })

        # # set value in pred box, be careful with out of bound index
        # ix0 = max(0, x0)
        # iy0 = max(0, y0)
        # ix1 = min(x0 + wx, pred_box.shape[0])
        # iy1 = min(y0 + wy, pred_box.shape[1])
        # pred_box[ix0:ix1, iy0:iy1] = 1

    # # apply hard-mask to pred box
    # pred_box = pred_box * m
    # calculate metrics
    sensitivity = np.sum(pred_box * target) / np.sum(target)
    specificity = np.sum((1-pred_box) * (1-target) * m) / np.sum((1-target) * m)
    metric = {
        'sensitivity': sensitivity,
        'specificity': specificity,
    }

    # update figure
    fig_txt = ", ".join(["{}: {:.2f}".format(k, v) for k, v in metric.items()])
    fig['layout'].update(height=800, width=800, title="{} threshold {}<br>{}".format(title, thres, fig_txt))

    return fig, proposed_boxes, metric


def make_plot_sigmoid(target, pred, title):
    tmp = np.sum(target[0, :, :], axis=0)
    try:
        first_nonzero_idx_from_right = next(i for i in range(len(tmp)) if tmp[-(i + 1)] > 0)
    except StopIteration:
        # target all 0, can't infer index, don't crop FIXME we should save original sequence length in df then we don't need to infer
        first_nonzero_idx_from_right = 0
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
    try:
        first_nonzero_idx_from_right = next(i for i in range(len(tmp)) if tmp[-(i + 1)] > 0)
    except StopIteration:
        # target all 0, can't infer index, don't crop FIXME we should save original sequence length in df then we don't need to infer
        first_nonzero_idx_from_right = 0
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




