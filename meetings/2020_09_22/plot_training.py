"""
Load training log and plot training progress.
"""
import argparse
import re
import dgutils.pandas as dgp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def main(in_log, out_plot):
    p_t = 'Epoch ([0-9]+)/[0-9]+, training loss \(running\) ([0-9.]+)$'
    p_v = 'Epoch ([0-9]+)/[0-9]+, validation loss ([0-9.]+)$'
    p_m = "({.*})"

    df_loss = []
    df_metric = []
    last_line = ''
    with open(in_log, 'r') as f:
        for line in f:
            # loss
            match_t = re.search(p_t, line)
            match_v = re.search(p_v, line)
            assert not (match_t and match_v)
            if match_t:
                df_loss.append(
                    {'epoch': int(match_t.group(1)), 'train_valid': 'training', 'loss': float(match_t.group(2))})
            if match_v:
                df_loss.append(
                    {'epoch': int(match_v.group(1)), 'train_valid': 'validation', 'loss': float(match_v.group(2))})

            # metric
            if "'stem_on" in line:
                match_m = re.search(p_m, line).group(1)
                metrics = eval(match_m)
                epoch = int(re.search('Epoch ([0-9]+)', last_line).group(1))
                if 'Training' in last_line:
                    t_v = 'training'
                elif 'Validation' in last_line:
                    t_v = 'validation'
                else:
                    raise ValueError
                df_metric.append({
                    'epoch': epoch,
                    'train_valid': t_v,

                    'stem_on_auroc': metrics['stem_on']['auroc'],
                    'stem_loc_x_accuracy': metrics['stem_location_x']['accuracy'],
                    'stem_loc_y_accuracy': metrics['stem_location_y']['accuracy'],
                    'stem_size_accuracy': metrics['stem_size']['accuracy'],

                    'iloop_on_auroc': metrics['iloop_on']['auroc'],
                    'iloop_loc_x_accuracy': metrics['iloop_location_x']['accuracy'],
                    'iloop_loc_y_accuracy': metrics['iloop_location_y']['accuracy'],
                    'iloop_size_x_accuracy': metrics['iloop_size_x']['accuracy'],
                    'iloop_size_y_accuracy': metrics['iloop_size_y']['accuracy'],

                    'hloop_on_auroc': metrics['hloop_on']['auroc'],
                    'hloop_loc_x_accuracy': metrics['hloop_location_x']['accuracy'],
                    'hloop_loc_y_accuracy': metrics['hloop_location_y']['accuracy'],
                    'hloop_size_accuracy': metrics['hloop_size']['accuracy'],
                })

            # keep track of last line so we know epoch & train/valid
            last_line = str(line)

    # plot
    fig = make_subplots(rows=6, cols=3, vertical_spacing=0.1, shared_yaxes=True,
                        subplot_titles=['loss', '', '',
                                        'stem_on_auroc', 'iloop_on_auroc', 'hloop_on_auroc',
                                        'stem_loc_x_accuracy', 'iloop_loc_x_accuracy', 'hloop_loc_x_accuracy',
                                        'stem_loc_y_accuracy', 'iloop_loc_y_accuracy', 'hloop_loc_y_accuracy',
                                        'stem_size_accuracy', 'iloop_size_x_accuracy', 'hloop_size_accuracy',
                                        '', 'iloop_size_y_accuracy', ''])

    tmp_fig = px.line(df_loss, x='epoch', y='loss', color='train_valid')
    for d in tmp_fig.data:
        fig.append_trace(d, 1, 1)

    # stem metrics
    for i, name in enumerate(['stem_on_auroc', 'stem_loc_x_accuracy', 'stem_loc_y_accuracy', 'stem_size_accuracy']):
        tmp_fig = px.line(df_metric, x='epoch', y=name, color='train_valid')
        tmp_fig.update_layout(showlegend=False)
        for d in tmp_fig.data:
            fig.append_trace(d, i + 2, 1)

    # iloop metrics
    for i, name in enumerate(['iloop_on_auroc', 'iloop_loc_x_accuracy', 'iloop_loc_y_accuracy', 'iloop_size_x_accuracy',
                              'iloop_size_y_accuracy']):
        tmp_fig = px.line(df_metric, x='epoch', y=name, color='train_valid')
        tmp_fig.update_layout(showlegend=False)
        for d in tmp_fig.data:
            fig.append_trace(d, i + 2, 2)

    # hloop metrics
    for i, name in enumerate(['hloop_on_auroc', 'hloop_loc_x_accuracy', 'hloop_loc_y_accuracy', 'hloop_size_accuracy']):
        tmp_fig = px.line(df_metric, x='epoch', y=name, color='train_valid')
        tmp_fig.update_layout(showlegend=False)
        for d in tmp_fig.data:
            fig.append_trace(d, i + 2, 3)

    fig.update_layout(
        autosize=False,
        width=900,
        height=1200,
    )

    pio.write_html(fig, file=out_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_log', type=str, help='Path to training log file')
    parser.add_argument('--out_plot', type=str, help='Path to output plot')
    args = parser.parse_args()
    main(args.in_log, args.out_plot)
