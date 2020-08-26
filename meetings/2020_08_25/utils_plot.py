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


def predict_bounidng_box(pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y, thres=0.5):
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


def array_clean_up(seq_len, target, pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y=None):
    # # remove padding
    # # FIXME we should save original sequence length in df then we don't need to infer
    if pred_siz_y is None:
        pred_siz_y = pred_siz_x.copy()  # same size
    # tmp = np.sum(target[0, :, :], axis=0)
    # try:
    #     first_nonzero_idx_from_right = next(i for i in range(len(tmp)) if tmp[-(i + 1)] > 0)
    # except StopIteration:
    #     # target all 0, can't infer index, don't crop
    #     first_nonzero_idx_from_right = 0
    # # crop
    # if first_nonzero_idx_from_right == 0:  # can't index by -0 <- will be empty!
    #     # no crop
    #     pass
    # else:
    #     target = target[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred_on = pred_on[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred_loc_x = pred_loc_x[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred_loc_y = pred_loc_y[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred_siz_x = pred_siz_x[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred_siz_y = pred_siz_y[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]

    # remove padding
    target = target[:, :seq_len, :seq_len]
    pred_on = pred_on[:, :seq_len, :seq_len]
    pred_loc_x = pred_loc_x[:, :seq_len, :seq_len]
    pred_loc_y = pred_loc_y[:, :seq_len, :seq_len]
    pred_siz_x = pred_siz_x[:, :seq_len, :seq_len]
    pred_siz_y = pred_siz_y[:, :seq_len, :seq_len]

    # apply 'hard-mask'  (lower triangle)
    assert target.shape == pred_on.shape  # channel x h x w
    assert target.shape[1] == target.shape[2]
    m = _make_mask(target.shape[1])
    # apply mask (for pred, only apply to pred_on since our processing starts from that array)
    target = target[0, :, :] * m
    pred_on = pred_on[0, :, :] * m
    return target, pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y, m



def make_plot_bb(seq, target, pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y=None, title=None, thres=0.5):
    # if pred_siz_y is None:
    #     pred_siz_y = pred_siz_x.copy()  # same size
    # tmp = np.sum(target[0, :, :], axis=0)
    # try:
    #     first_nonzero_idx_from_right = next(i for i in range(len(tmp)) if tmp[-(i + 1)] > 0)
    # except StopIteration:
    #     # target all 0, can't infer index, don't crop
    #     first_nonzero_idx_from_right = 0
    # # crop
    # if first_nonzero_idx_from_right == 0:  # can't index by -0 <- will be empty!
    #     # no crop
    #     pass
    # else:
    #     target = target[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred_on = pred_on[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred_loc_x = pred_loc_x[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred_loc_y = pred_loc_y[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred_siz_x = pred_siz_x[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred_siz_y = pred_siz_y[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]

    # # apply 'hard-mask'  (lower triangle)
    # assert target.shape == pred_on.shape  # channel x h x w
    # assert target.shape[1] == target.shape[2]
    # m = _make_mask(target.shape[1])
    # # apply mask (for pred, only apply to pred_on since our processing starts from that array)
    # target = target[0, :, :] * m
    # pred_on = pred_on[0, :, :] * m
    target, pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y, m = array_clean_up(len(seq), target, pred_on, pred_loc_x,
                                                                                     pred_loc_y, pred_siz_x, pred_siz_y)

    fig = px.imshow(target)

    # predict bounding boxes
    proposed_boxes, pred_box = predict_bounidng_box(pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y, thres)

    for bb in proposed_boxes:
        bb_x = bb['bb_x']
        bb_y = bb['bb_y']
        siz_x = bb['siz_x']
        siz_y = bb['siz_y']
        prob = bb['prob']

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


def make_target_df(target_bb):
    df_target_stem = []
    df_target_iloop = []
    df_target_hloop = []
    for (bb_x, bb_y), (siz_x, siz_y), bb_type in target_bb:
        row = {
            'bb_x': bb_x,
            'bb_y': bb_y,
            'siz_x': siz_x,
            'siz_y': siz_y,
        }
        if bb_type == 'stem':
            df_target_stem.append(row)
        elif bb_type in ['bulge', 'internal_loop']:
            df_target_iloop.append(row)
        elif bb_type == 'hairpin_loop':
            df_target_hloop.append(row)
        elif bb_type == 'pseudo_knot':
            pass  # do not process
        else:
            raise ValueError  # TODO pseudo knot?
    if len(df_target_stem) > 0:
        df_target_stem = pd.DataFrame(df_target_stem)
    if len(df_target_iloop) > 0:
        df_target_iloop = pd.DataFrame(df_target_iloop)
    if len(df_target_hloop) > 0:
        df_target_hloop = pd.DataFrame(df_target_hloop)
    return df_target_stem, df_target_iloop, df_target_hloop


def _calculate_bb_metrics(df_target, df_pred):

    def is_identical(bb1, bb2):
        return bb1 == bb2  # this should work? FIXME
        # bb1_x, bb1_y, siz1_x, siz1_y = bb1
        # bb2_x, bb2_y, siz2_x, siz2_y = bb2
        # # FIXME debug! any off-by-1 error?
        # return abs(bb1_x-bb2_x)<=1 and abs(bb1_y-bb2_y)<=1 and abs(siz1_x-siz2_x)<=1 and abs(siz1_y-siz2_y)<=1

    def is_overlap(bb1, bb2):
        bb1_x, bb1_y, siz1_x, siz1_y = bb1
        bb2_x, bb2_y, siz2_x, siz2_y = bb2
        # calculate overlap rectangle, check to see if it's empty
        x0 = max(bb1_x, bb2_x)
        x1 = min(bb1_x + siz1_x - 1, bb2_x + siz2_x - 1)  # note this is closed end
        y0 = max(bb1_y - siz1_y + 1, bb2_y - siz2_y + 1)  # closed end
        y1 = min(bb1_y, bb2_y)
        if x1 >= x0 and y1 >= y0:
            return True
        else:
            return False

    assert set(df_target.columns) == {'bb_x', 'bb_y', 'siz_x', 'siz_y'}
    assert set(df_pred.columns) == {'bb_x', 'bb_y', 'siz_x', 'siz_y'}

    # make sure all rows are unique
    assert not df_target.duplicated().any()
    assert not df_pred.duplicated().any()

    # w.r.t. target
    n_target_total = len(df_target)
    n_target_identical = 0
    n_target_overlap = 0
    n_target_nohit = 0
    for _, row1 in df_target.iterrows():
        # bb1 = (row1['bb_x'], row1['bb_y'], row1['siz_x'], row1['siz_y'])
        # FIXME data generator returns ledft corner of bb, convert to top right corner to be consistent
        bb1 = (row1['bb_x'], row1['bb_y'] + row1['siz_y'] -1, row1['siz_x'], row1['siz_y'])
        found_identical = False
        found_overlapping = False
        for _, row2 in df_pred.iterrows():
            bb2 = (row2['bb_x'], row2['bb_y'], row2['siz_x'], row2['siz_y'])
            if is_identical(bb1, bb2):
                found_identical = True
            elif is_overlap(bb1, bb2):  # note this is overlapping but NOT identical due to "elif"
                found_overlapping = True
            else:
                pass
        if found_identical:
            n_target_identical += 1
        elif found_overlapping:
            n_target_overlap += 1
        else:
            n_target_nohit += 1

    # FIXME there is some wasted comparison here (can be combined with last step)
    # w.r.t. pred
    n_pred_total = len(df_pred)
    n_pred_identical = 0
    n_pred_overlap = 0
    n_pred_nohit = 0
    for _, row1 in df_pred.iterrows():
        bb1 = (row1['bb_x'], row1['bb_y'], row1['siz_x'], row1['siz_y'])
        found_identical = False
        found_overlapping = False
        for _, row2 in df_target.iterrows():
            # bb2 = (row2['bb_x'], row2['bb_y'], row2['siz_x'], row2['siz_y'])
            # FIXME data generator returns ledft corner of bb, convert to top right corner to be consistent
            bb2 = (row2['bb_x'], row2['bb_y'] + row2['siz_y'] -1, row2['siz_x'], row2['siz_y'])
            if is_identical(bb1, bb2):
                found_identical = True
            elif is_overlap(bb1, bb2):  # note this is overlapping but NOT identical due to "elif"
                found_overlapping = True
            else:
                pass
        if found_identical:
            n_pred_identical += 1
        elif found_overlapping:
            n_pred_overlap += 1
        else:
            n_pred_nohit += 1
    result = {
        'n_target_total': n_target_total,
        'n_target_identical': n_target_identical,
        'n_target_overlap': n_target_overlap,
        'n_target_nohit': n_target_nohit,
        'n_pred_total': n_pred_total,
        'n_pred_identical': n_pred_identical,
        'n_pred_overlap': n_pred_overlap,
        'n_pred_nohit': n_pred_nohit,
    }
    return result


def calculate_bb_metrics(df_target, df_pred):
    if (df_target is None or len(df_target) == 0) and (df_pred is None or len(df_pred) == 0):
        return {
            'n_target_total': 0,
            'n_target_identical': 0,
            'n_target_overlap': 0,
            'n_target_nohit': 0,
            'n_pred_total': 0,
            'n_pred_identical': 0,
            'n_pred_overlap': 0,
            'n_pred_nohit': 0,
        }

    elif df_target is None or len(df_target) == 0:
        return {
            'n_target_total': 0,
            'n_target_identical': 0,
            'n_target_overlap': 0,
            'n_target_nohit': 0,
            'n_pred_total': len(df_pred),
            'n_pred_identical': 0,
            'n_pred_overlap': 0,
            'n_pred_nohit': 0,
        }
    elif df_pred is None or len(df_pred) == 0:
        return {
            'n_target_total': len(df_target),
            'n_target_identical': 0,
            'n_target_overlap': 0,
            'n_target_nohit': 0,
            'n_pred_total': 0,
            'n_pred_identical': 0,
            'n_pred_overlap': 0,
            'n_pred_nohit': 0,
        }
    else:
        return _calculate_bb_metrics(df_target, df_pred)


def sensitivity_specificity(target_on, pred_box, hard_mask):
    sensitivity = np.sum(pred_box * target_on) / np.sum(target_on)
    specificity = np.sum((1-pred_box) * (1-target_on) * hard_mask) / np.sum((1-target_on) * hard_mask)
    return sensitivity, specificity


def calculate_metrics(row, threshold):
    seq = row['seq']
    target_bb = row['bounding_boxes']
    target = row['target_stem_on']
    pred_on = row['pred_stem_on']
    pred_loc_x = row['pred_stem_location_x']
    pred_loc_y = row['pred_stem_location_y']
    pred_siz_x = row['pred_stem_size']
    # predict stem bb
    target_stem_on, pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y, m = array_clean_up(len(seq), target, pred_on, pred_loc_x,
                                                                                        pred_loc_y, pred_siz_x,
                                                                                        pred_siz_y=None)
    bb_stem, pred_box_stem = predict_bounidng_box(pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y,
                                                  thres=threshold)
    # predict iloop bb
    target = row['target_iloop_on']
    pred_on = row['pred_iloop_on']
    pred_loc_x = row['pred_iloop_location_x']
    pred_loc_y = row['pred_iloop_location_y']
    pred_siz_x = row['pred_iloop_size_x']
    pred_siz_y = row['pred_iloop_size_y']
    target_iloop_on, pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y, m = array_clean_up(len(seq), target, pred_on, pred_loc_x,
                                                                                        pred_loc_y, pred_siz_x,
                                                                                        pred_siz_y)
    bb_iloop, pred_box_iloop = predict_bounidng_box(pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y,
                                                    thres=threshold)
    # predict hloop bb
    target = row['target_hloop_on']
    pred_on = row['pred_hloop_on']
    pred_loc_x = row['pred_hloop_location_x']
    pred_loc_y = row['pred_hloop_location_y']
    pred_siz_x = row['pred_hloop_size']
    target_hloop_on, pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y, m = array_clean_up(len(seq), target, pred_on, pred_loc_x,
                                                                                        pred_loc_y, pred_siz_x,
                                                                                        pred_siz_y=None)
    bb_hloop, pred_box_hloop = predict_bounidng_box(pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y,
                                                    thres=threshold)
    # convert to dfs
    if len(bb_stem) > 0:
        df_stem = pd.DataFrame(bb_stem)
        df_stem = df_stem[['bb_x', 'bb_y', 'siz_x', 'siz_y']].drop_duplicates()
    else:
        df_stem = None
    if len(bb_iloop) > 0:
        df_iloop = pd.DataFrame(bb_iloop)
        df_iloop = df_iloop[['bb_x', 'bb_y', 'siz_x', 'siz_y']].drop_duplicates()
    else:
        df_iloop = None
    if len(bb_hloop) > 0:
        df_hloop = pd.DataFrame(bb_hloop)
        df_hloop = df_hloop[['bb_x', 'bb_y', 'siz_x', 'siz_y']].drop_duplicates()
    else:
        df_hloop = None
    # process target bb list into different types, store in df
    df_target_stem, df_target_iloop, df_target_hloop = make_target_df(target_bb)

    # # FIXME debug
    # print(df_target_stem)
    # print(df_stem)

    # metric for each bb type
    m_stem = calculate_bb_metrics(df_target_stem, df_stem)
    m_iloop = calculate_bb_metrics(df_target_iloop, df_iloop)
    m_hloop = calculate_bb_metrics(df_target_hloop, df_hloop)
    # calculate non-bb sensitivity and specificity
    se_stem, sp_stem = sensitivity_specificity(target_stem_on, pred_box_stem, m)
    se_iloop, sp_iloop = sensitivity_specificity(target_iloop_on, pred_box_iloop, m)
    se_hloop, sp_hloop = sensitivity_specificity(target_hloop_on, pred_box_hloop, m)
    # combine
    m_stem.update({'struct_type': 'stem', 'sensitivity': se_stem, 'specificity': sp_stem})
    m_iloop.update({'struct_type': 'iloop', 'sensitivity': se_iloop, 'specificity': sp_iloop})
    m_hloop.update({'struct_type': 'hloop', 'sensitivity': se_hloop, 'specificity': sp_hloop})
    df_result = pd.DataFrame([m_stem, m_iloop, m_hloop])
    df_result['bb_sensitivity_identical'] = df_result['n_target_identical'] / df_result['n_target_total']
    df_result['bb_sensitivity_overlap'] = (df_result['n_target_identical'] + df_result['n_target_overlap']) / df_result[
        'n_target_total']
    return df_result


def make_plot_sigmoid(seq, target, pred, title):
    # tmp = np.sum(target[0, :, :], axis=0)
    # try:
    #     first_nonzero_idx_from_right = next(i for i in range(len(tmp)) if tmp[-(i + 1)] > 0)
    # except StopIteration:
    #     # target all 0, can't infer index, don't crop FIXME we should save original sequence length in df then we don't need to infer
    #     first_nonzero_idx_from_right = 0
    # # crop
    # if first_nonzero_idx_from_right == 0:  # can't index by -0 <- will be empty!
    #     # no crop
    #     pass
    # else:
    #     target = target[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred = pred[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]

    target = target[:len(seq), :len(seq)]
    pred = pred[:len(seq), :len(seq)]

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


def make_plot_softmax(seq, target, pred, title):
    pred = softmax(pred, axis=0)

    # tmp = np.sum(target[0, :, :], axis=0)
    # try:
    #     first_nonzero_idx_from_right = next(i for i in range(len(tmp)) if tmp[-(i + 1)] > 0)
    # except StopIteration:
    #     # target all 0, can't infer index, don't crop FIXME we should save original sequence length in df then we don't need to infer
    #     first_nonzero_idx_from_right = 0
    # # crop
    # if first_nonzero_idx_from_right == 0:  # can't index by -0 <- will be empty!
    #     # no crop
    #     pass
    # else:
    #     target = target[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    #     pred = pred[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]

    target = target[:len(seq), :len(seq)]
    pred = pred[:len(seq), :len(seq)]

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




