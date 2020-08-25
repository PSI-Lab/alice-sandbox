import os
import argparse
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from scipy.special import softmax
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils_plot import _make_mask, make_plot_bb, make_plot_sigmoid, make_plot_softmax


def main(in_file, out_file, thres, opt_row, opt_tgt, opt_bb, verbose=False):
    df = pd.read_pickle(in_file)

    # col -> plot_func
    col2plot = OrderedDict()
    col2plot['stem_on'] = make_plot_sigmoid
    col2plot['stem_location_x'] = make_plot_softmax
    col2plot['stem_location_y'] = make_plot_softmax
    col2plot['stem_size'] = make_plot_softmax
    # TODO add other targets

    # which rows to plot
    if opt_row == 'all':
        row_ite = df.iterrows()
    elif opt_row == 'sample':
        # pick random index of training data point
        row_tr = df[df['subset'] == 'training'].sample(n=1).iloc[0]
        # pick random index of validation data point
        row_va = df[df['subset'] == 'validation'].sample(n=1).iloc[0]
        row_ite = enumerate([row_tr, row_va])
    else:
        raise ValueError

    # assert set(df.columns) == {'target', 'pred', 'subset'}
    # output to a single html
    with open(out_file, 'w') as f:
        # # pick random index of training data point
        # row_tr = df[df['subset'] == 'training'].sample(n=1).iloc[0]
        # # pick random index of validation data point
        # row_va = df[df['subset'] == 'validation'].sample(n=1).iloc[0]
        for _, row in tqdm(row_ite):
            if opt_tgt:
                # per-output plot
                for col, func in col2plot.items():
                    fig = func(row['seq'], row['target_{}'.format(col)], row['pred_{}'.format(col)],
                               '{} name: {} type: {}'.format(in_file, col, row.subset))
                    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            if opt_bb:
                # proposed bounding box plot
                # stem
                fig, bb, m = make_plot_bb(row['seq'], row['target_stem_on'], row['pred_stem_on'],
                                   row['pred_stem_location_x'], row['pred_stem_location_y'],
                                   row['pred_stem_size'], pred_siz_y=None,
                                   title='stem bounding box prediction, type: {} '.format(row.subset), thres=thres)
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                if verbose:
                    print('\nstem', m)
                # iloop
                fig, bb, m = make_plot_bb(row['seq'], row['target_iloop_on'], row['pred_iloop_on'],
                                   row['pred_iloop_location_x'], row['pred_iloop_location_y'],
                                   row['pred_iloop_size_x'], pred_siz_y=row['pred_iloop_size_y'],
                                   title='iloop bounding box prediction, type: {} '.format(row.subset), thres=thres)
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                if verbose:
                    print('iloop', m)
                # hloop
                fig, bb, m = make_plot_bb(row['seq'], row['target_hloop_on'], row['pred_hloop_on'],
                                   row['pred_hloop_location_x'], row['pred_hloop_location_y'],
                                   row['pred_hloop_size'], pred_siz_y=None,
                                   title='hloop bounding box prediction, type: {} '.format(row.subset), thres=thres)
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                if verbose:
                    print('hloop', m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Path to input prediction file')
    parser.add_argument('--out_file', type=str, help='Path to output plot html file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for calling bounding box.')
    parser.add_argument('--row', type=str, help='Whether to plot all rows, or sample.')
    parser.add_argument('--tgt', action='store_true', help='Whether to plot individual output predictions.')
    parser.add_argument('--bb', action='store_true', help='Whether to plot bounding box proposal predictions.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print progress.')
    args = parser.parse_args()

    assert 0 < args.threshold < 1
    assert args.out_file.endswith('.html')
    assert args.row in ['all', 'sample']
    # at least one needs to be specified
    assert args.tgt or args.bb

    # make result dir if non existing
    out_path = os.path.dirname(args.out_file)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    main(args.in_file, args.out_file, args.threshold, args.row, args.tgt, args.bb, args.verbose)

