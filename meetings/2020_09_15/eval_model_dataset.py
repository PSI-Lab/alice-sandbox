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
import datacorral as dc
import dgutils.pandas as dgp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots


def main(data_path, num_datapoints, max_len, model_path, out_csv, out_plot):
    dc_client = dc.Client()

    df_data = pd.read_pickle(data_path, compression='gzip')
    # drop those > max_len
    df_data = dgp.add_column(df_data, 'tmp', ['seq'], len)
    df_data = df_data[df_data['tmp'] <= max_len]
    df_data = df_data.drop(columns=['tmp'])

    # sample data points
    df_data = df_data.sample(n=min(num_datapoints, len(df_data)))

    predictor = Predictor(model_path)

    evaluator = Evaluator(predictor)
    result = []


    ctime = time()
    for idx, row in df_data.iterrows():
        seq = row['seq']
        one_idx = row['one_idx']
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

    fig = make_subplots(rows=4, cols=2, vertical_spacing=0.1,
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

    fig.update_layout(
        autosize=False,
        width=600,
        height=900,
    )

    # export
    df_result.to_csv(out_csv, index=False)
    to_html(fig, out_plot)
    print("Saved to:\n{}\n{}".format(out_csv, out_plot))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to dataset')
    parser.add_argument('--num', type=int, help='Number of data points to sample')
    parser.add_argument('--maxl', type=int, help='Max sequence length')
    parser.add_argument('--model', type=str, help='Path to pytorch model params')
    parser.add_argument('--out_csv', type=str, help='Path to output csv')
    parser.add_argument('--out_plot', type=str, help='Path to output plot')
    args = parser.parse_args()
    main(args.data, args.num, args.maxl, args.model, args.out_csv, args.out_plot)

