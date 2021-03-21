"""
Predict all unique bounding boxes by thresholding on n_proposal_norm.
Store bounding box location and probabiliti(es).
"""
import argparse
# import torch
from time import time
from utils_model import Predictor, Evaluator
# import datacorral as dc
import dgutils.pandas as dgp
import pandas as pd


def filter_by_n_proposal(df_bb, threshold):
    if len(df_bb) == 0:
        return df_bb
    else:
        # handle cases where there's only softmax predicted or scalar predicted
        if 'prob_sm' not in df_bb.columns:
            df_bb = dgp.add_column(df_bb, 'prob_sm', ['siz_x'], lambda a: [])  # hacky way to create a column of empty lists
        if 'prob_sl' not in df_bb.columns:
            df_bb = dgp.add_column(df_bb, 'prob_sl', ['siz_x'],
                                   lambda a: [])  # hacky way to create a column of empty lists
        df_bb = dgp.add_column(df_bb, 'n_proposal_norm_sm', ['prob_sm', 'siz_x', 'siz_y'],
                          lambda a, b, c: len(a)/float(b * c))
        df_bb = dgp.add_column(df_bb, 'n_proposal_norm_sl', ['prob_sl', 'siz_x', 'siz_y'],
                          lambda a, b, c: len(a)/float(b * c))
        return df_bb[(df_bb['n_proposal_norm_sm'] > threshold) | (df_bb['n_proposal_norm_sl'] > threshold)]


def pred_threshold_on_n_proposal(seq, predictor, threshold):
    stems, iloops, hloops = predictor.predict_bb(seq, threshold=0, topk=1, perc_cutoff=0)
    stems = pd.DataFrame(stems)
    iloops = pd.DataFrame(iloops)
    hloops = pd.DataFrame(hloops)

    stems = filter_by_n_proposal(stems, threshold)
    iloops = filter_by_n_proposal(iloops, threshold)
    hloops = filter_by_n_proposal(hloops, threshold / 2)  # /2 threshold due to /2 upper bound
    return stems, iloops, hloops


def main(data_path, n_proposal_threshold, model_path, out_file):
    df_data = pd.read_pickle(data_path, compression='gzip')
    # # drop those > max_len
    # df_data = dgp.add_column(df_data, 'tmp', ['seq'], len)
    # df_data = df_data[df_data['tmp'] <= max_len]
    # df_data = df_data.drop(columns=['tmp'])

    # # sample data points
    # # if set to <= 0 do not resample
    # if num_datapoints > 0:
    #     print("Sampling {} data points with rand seed {}".format(num_datapoints, random_state))
    #     df_data = df_data.sample(n=min(num_datapoints, len(df_data)), random_state=random_state)

    predictor = Predictor(model_ckpt=model_path,
                          num_filters=[32, 32, 64, 64, 64, 128, 128],
                          filter_width=[9, 9, 9, 9, 9, 9, 9],
                          dropout=0.0)

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

        new_row = row.copy()

        stems, iloops, hloops = pred_threshold_on_n_proposal(seq, predictor, n_proposal_threshold)

        # if patch_size == 0:  # run on full sequence at once
        #     uniq_stem, uniq_iloop, uniq_hloop = predictor.predict_bb(seq=seq, threshold=threshold, topk=topk, perc_cutoff=perc_cutoff)
        # else:
        #     assert patch_size > 0
        #     uniq_stem, uniq_iloop, uniq_hloop = predictor.predict_bb_split(seq=seq, threshold=threshold, topk=topk, perc_cutoff=perc_cutoff, patch_size=patch_size)

        new_row['bb_stem'] = stems.to_dict('records')
        new_row['bb_iloop'] = iloops.to_dict('records')
        new_row['bb_hloop'] = hloops.to_dict('records')

        # if len(result) % 100 == 0:
        if len(result) % 10 == 0:  # FIXME debug
            print(len(result), time() - ctime)
            ctime = time()
        result.append(new_row)

    result = pd.DataFrame(result)
    result.to_pickle(out_file, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to dataset')
    # parser.add_argument('--num', type=int, help='Number of data points to sample. Set to 0 to use all.')  # for debug use
    # parser.add_argument('--random_state', type=int, default=5555, help='Used if --num is set, random seed for sampling rows from df.')
    parser.add_argument('--threshold', type=float, help='threshold on n_proposal_norm')
    # parser.add_argument('--topk', type=int, default=1, help='max number of predictions per pixel (only for pixels where prob_on > threshold, k per softmax and scalar)')
    # parser.add_argument('--perc_cutoff', type=float, default=1,
    #                     help='picked bb needs to have joint probability within perc_cutoff * p_top_hit (only for pixels where prob_on > threshold, per softmax and scalar)')
    # parser.add_argument('--patch_size', type=int, default=0, help='patch_size for splitting sequence, used to avoid OOM when running long sequence. Set to 0 to turn this off.')
    parser.add_argument('--model', type=str, help='Path to pytorch model params')
    parser.add_argument('--out_file', type=str, help='Path to output csv pickle')
    args = parser.parse_args()
    assert  0 < args.threshold < 1
    # assert args.topk >= 1  # for generating s2 training data we require specifying a fixed number (in the actual inference interface 0 is allowed for switching to only perc_cutoff)
    # assert 0 <= args.perc_cutoff <= 1
    main(args.data, args.threshold, args.model, args.out_file)

