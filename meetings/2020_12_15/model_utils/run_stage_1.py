"""
Predict all unique bounding boxes at a given threshold.
Store bounding box location and probabiliti(es).
"""
import argparse
import torch
from time import time
from utils_model import Predictor, Evaluator
import datacorral as dc
import dgutils.pandas as dgp
import pandas as pd


# def uniq_boxes(pred_bb):
#     # pred_bb: list
#     df = pd.DataFrame(pred_bb)
#     data = df.groupby(by=['bb_x', 'bb_y', 'siz_x', 'siz_y'], as_index=False).agg(list).to_dict('records')
#     return data


def main(data_path, num_datapoints, threshold, model_path, out_file):
    df_data = pd.read_pickle(data_path, compression='gzip')
    # # drop those > max_len
    # df_data = dgp.add_column(df_data, 'tmp', ['seq'], len)
    # df_data = df_data[df_data['tmp'] <= max_len]
    # df_data = df_data.drop(columns=['tmp'])

    # sample data points
    # if set to <= 0 do not resample
    if num_datapoints > 0:
        df_data = df_data.sample(n=min(num_datapoints, len(df_data)))

    predictor = Predictor(model_path)

    # evaluator = Evaluator(predictor)
    result = []


    ctime = time()
    for idx, row in df_data.iterrows():
        seq = row['seq']
        # one_idx = row['one_idx']

        # for now drop weird sequence
        if not set(seq.upper().replace('U', 'T')).issubset(set(list('ACGTN'))):
            continue
        # skip example with no structures
        if len(row['one_idx'][0]) == 0:
            assert len(row['one_idx'][1]) == 0
            continue

        # yp, pred_bb_stem, pred_bb_iloop, pred_bb_hloop, pred_box_stem, pred_box_iloop, pred_box_hloop = predictor.predict_bb(seq, threshold)
        #
        # new_row = row.copy()
        # if len(pred_bb_stem) > 0:
        #     uniq_stem = uniq_boxes(pred_bb_stem)
        #     new_row['bb_stem'] = uniq_stem
        # if len(pred_bb_iloop) > 0:
        #     uniq_iloop = uniq_boxes(pred_bb_iloop)
        #     new_row['bb_iloop'] = uniq_iloop
        # if len(pred_bb_hloop) > 0:
        #     uniq_hloop = uniq_boxes(pred_bb_hloop)
        #     new_row['bb_hloop'] = uniq_hloop

        new_row = row.copy()
        uniq_stem, uniq_iloop, uniq_hloop = predictor.predict_bb(seq, threshold)
        new_row['bb_stem'] = uniq_stem
        new_row['bb_iloop'] = uniq_iloop
        new_row['bb_hloop'] = uniq_hloop

        if len(result) % 100 == 0:
            print(len(result), time() - ctime)
            ctime = time()
        result.append(new_row)

    result = pd.DataFrame(result)
    result.to_pickle(out_file, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to dataset')
    parser.add_argument('--num', type=int, help='Number of data points to sample. Set to 0 to use all.')  # for debug use
    parser.add_argument('--threshold', type=float, help='threshold')
    parser.add_argument('--model', type=str, help='Path to pytorch model params')
    parser.add_argument('--out_file', type=str, help='Path to output csv pickle')
    args = parser.parse_args()
    assert  0 < args.threshold < 1
    main(args.data, args.num, args.threshold, args.model, args.out_file)

