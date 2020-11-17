"""
Eval model via:
- load the mini batch prediction, apply threshold and find bounding boxes
- load model, use the inference pipeline to predict on sequence directly
(doing both just to verify)
"""
import argparse
import torch
from time import time
from utils_model import Predictor, Evaluator
# import datacorral as dc
import dgutils.pandas as dgp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def get_max_size(bbs):
    s_max = 0  # init with 0, in case no bb exist
    bb_sizes = []
    for bb in bbs:
        s = max(bb[1])  # max of siz_x, siz_y
        bb_sizes.append(s)
    if len(bb_sizes) > 0:
        s_max = max(bb_sizes)
    return s_max


def main(data_path, num_datapoints, max_len, max_bb_size, model_path, out_csv, out_plot):
    # dc_client = dc.Client()

    df_data = pd.read_pickle(data_path, compression='gzip')
    # drop those > max_len
    df_data = dgp.add_column(df_data, 'tmp', ['seq'], len)
    df_data = df_data[df_data['tmp'] <= max_len]
    df_data = df_data.drop(columns=['tmp'])
    # if max_bb_size is set, drop those where any bb size is > max_bb_size
    if max_bb_size > 0:
        n_before = len(df_data)
        df_data = dgp.add_column(df_data, 'tmp', ['bounding_boxes'], get_max_size)
        df_data = df_data[df_data['tmp'] <= max_bb_size]
        df_data = df_data.drop(columns=['tmp'])
        n_after = len(df_data)
        print("Dropping example with bb size > {}. Before: {}, after: {}".format(max_bb_size, n_before, n_after))

    # sample data points
    df_data = df_data.sample(n=min(num_datapoints, len(df_data)))

    predictor = Predictor(model_path)

    evaluator = Evaluator(predictor)
    result = []


    ctime = time()
    for idx, row in df_data.iterrows():
        seq = row['seq']
        one_idx = row['one_idx']

        # for now drop weird sequence
        if not set(seq.upper().replace('U', 'T')).issubset(set(list('ACGTN'))):
            continue
        # skip example with no structures
        if len(row['one_idx'][0]) == 0:
            assert len(row['one_idx'][1]) == 0
            continue

        evaluator.predict(seq, one_idx, 0.1)
        df_result, metrics = evaluator.calculate_metrics()
        print(idx, time() - ctime)
        ctime = time()
        result.append(metrics)

    result = pd.DataFrame(result)

    # restructure data to re-use previous plotting code
    # 'bb_sensitivity_identical', 'bb_sensitivity_overlap', 'specificity', `struct_type`
    df_result = []
    for _, m in result.iterrows():
        for struct_type in ['stem', 'iloop', 'hloop']:
            df_result.append({
                'struct_type': struct_type,
                'bb_sensitivity_identical': m['bb_{}_identical'.format(struct_type)],
                'bb_sensitivity_overlap': m['bb_{}_overlap'.format(struct_type)],
                'sensitivity': m['px_{}_sensitivity'.format(struct_type)],
                'specificity': m['px_{}_specificity'.format(struct_type)],
            })
    df_result = pd.DataFrame(df_result)

    fig = make_subplots(rows=4, cols=1, vertical_spacing=0.1,
                        subplot_titles=['Sensitivity (identical bounding box)',
                                        'Sensitivity (overlapping bounding box)',
                                        'Sensitivity (pixel) training',
                                        'Specificity (pixel) training'])
    bb2color = {
        'stem': px.colors.qualitative.Plotly[0],
        'iloop': px.colors.qualitative.Plotly[1],
        'hloop': px.colors.qualitative.Plotly[2],
    }

    for bb_type in ['stem', 'iloop', 'hloop']:
        for i, col_name in enumerate(
                ['bb_sensitivity_identical', 'bb_sensitivity_overlap', 'sensitivity', 'specificity']):
            df_plot = df_result[df_result['struct_type'] == bb_type][[col_name]]
            fig.append_trace(go.Histogram(x=df_plot[col_name], showlegend=True if i == 0 else False,
                                          name=bb_type, nbinsx=20, marker_color=bb2color[bb_type],
                                          histnorm='percent'), i + 1, 1)
    fig.update_xaxes(range=[-0.1, 1.1])  # don't cap at [0, 1] since some bars will be cut off (since the axis label is centered)
    fig.update_layout(
        autosize=False,
        width=600,
        height=900,
    )

    # export
    df_result.to_csv(out_csv, index=False)
    pio.write_html(fig, file=out_plot)
    print("Saved to:\n{}\n{}".format(out_csv, out_plot))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to dataset')
    parser.add_argument('--num', type=int, help='Number of data points to sample')
    parser.add_argument('--maxl', type=int, help='Max sequence length')
    parser.add_argument('--max_bb_size', type=int, default=0,
                        help='Max bounding box size. Example where any bb size exceed this limit will be ignored. Set to 0 to allow any sizes.')
    parser.add_argument('--model', type=str, help='Path to pytorch model params')
    parser.add_argument('--out_csv', type=str, help='Path to output csv')
    parser.add_argument('--out_plot', type=str, help='Path to output plot')
    args = parser.parse_args()
    main(args.data, args.num, args.maxl, args.max_bb_size, args.model, args.out_csv, args.out_plot)

