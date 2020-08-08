import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _make_mask(l):
    m = np.ones((l, l))
    m[np.tril_indices(l)] = 0
    return m


def make_plot(target, pred, title):
    # hacky way to infer padding location
    # by finding out how many consecutive columns are all 0's in target channel 0
    tmp = np.sum(target[0, :, :], axis=0)
    first_nonzero_idx_from_right = next(i for i in range(len(tmp)) if tmp[-(i + 1)] > 0)

    # crop
    target = target[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]
    pred = pred[:, :-first_nonzero_idx_from_right, :-first_nonzero_idx_from_right]

    # apply 'hard-mask'  (lower triangle)
    assert target.shape == pred.shape  # channel x h x w
    assert target.shape[1] == target.shape[2]
    m = _make_mask(target.shape[1])
    # repeat mask on channel
    m = m[np.newaxis, :, :]
    m = np.repeat(m, target.shape[0], axis=0)

    # apply mask
    target = target * m
    pred = pred * m

    # this is known from data processing
    # not_local_structure, stem, internal_loop (include bulges), hairpin loop, is_corner
    target_names = ['not_ss', 'stem', 'i_loop', 'h_loop', 'corner']
    assert len(target_names) == target.shape[0]
    assert len(target_names) == pred.shape[0]

    fig = make_subplots(rows=len(target_names), cols=2, print_grid=False, shared_yaxes=True, shared_xaxes=True)

    for i in range(len(target_names)):
        # target
        tmp_fig = px.imshow(target[i, :, :])
        fig.append_trace(tmp_fig.data[0], i + 1, 1)
        fig['layout']['xaxis{}'.format(2 * i + 1)].update(title=target_names[i])
        fig['layout']['yaxis{}'.format(2 * i + 1)]['autorange'] = "reversed"
        # pred
        tmp_fig = px.imshow(pred[i, :, :])
        fig.append_trace(tmp_fig.data[0], i + 1, 2)
        fig['layout']['xaxis{}'.format(2 * i + 2)].update(title=target_names[i])
        fig['layout']['yaxis{}'.format(2 * i + 1)]['autorange'] = "reversed"
        # TODO apply hard mask?

    # fig['layout']['yaxis1'].update(title='y-axis')
    fig['layout'].update(height=400 * len(target_names), width=400 * 2, title=title)

    fig['layout']['yaxis']['autorange'] = "reversed"

    return fig


def main(in_file, out_path):
    df = pd.read_pickle(in_file)
    assert set(df.columns) == {'target', 'pred', 'subset'}
    # pick random index of training data point
    row_tr = df[df['subset'] == 'training'].sample(n=1).iloc[0]
    fig_tr = make_plot(row_tr['target'], row_tr['pred'],
                       '{} type: {}'.format(in_file, row_tr.subset))
    fig_tr.write_html(os.path.join(out_path, 'train.html'))
    # pick random index of validation data point
    row_va = df[df['subset'] == 'validation'].sample(n=1).iloc[0]
    fig_va = make_plot(row_va['target'], row_va['pred'],
                       '{} type: {}'.format(in_file, row_va.subset))
    fig_va.write_html(os.path.join(out_path, 'validation.html'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Prediction file')
    parser.add_argument('--out_path', type=str, help='Path to output result')
    args = parser.parse_args()

    # make result dir if non existing
    if not os.path.isdir(args.out_path):
        os.makedirs(args.out_path)

    main(args.in_file, args.out_path)

