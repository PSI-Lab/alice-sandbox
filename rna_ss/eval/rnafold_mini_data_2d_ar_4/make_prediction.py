import yaml
import argparse
import numpy as np
import pandas as pd
from dgutils.pandas import add_column, add_columns
from utils import PredictorSPlitModel, arr2db, EvalMetric
import logging


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def process_row(seq, threshold, model):
    pred = model.predict_non_ar(seq=seq, threshold=threshold)

    assert pred.shape[0] == 1
    # idx_sample = 0
    _pred = pred[0, :, :, 0].copy()  # need to copy! otherwise following code will overwrite the values
    # set lower triangular to 0
    _pred[np.tril_indices(_pred.shape[0])] = 0
    # make sure it only has 0s and 1s
    assert np.all((_pred == 1) | (_pred == 0))
    data_pred = np.where(_pred == 1)  # index of the sampled 1's
    return data_pred


def calculate_metric(one_idx, pred_idx, seq_len, eval):
    target = np.zeros((seq_len, seq_len))
    pred = np.zeros((seq_len, seq_len))
    target[one_idx] = 1
    pred[pred_idx] = 1
    sensitivity = eval.sensitivity(pred, target)
    ppv = eval.ppv(pred, target)
    return sensitivity, ppv, eval.f_measure(sensitivity, ppv)


def main(model_file, dataset_file, ml_threshold, output):
    model = PredictorSPlitModel(model_file)
    # TODO make sure dataset has unified format
    df = pd.read_pickle(dataset_file)

    df = add_column(df, 'pred_idx', ['seq'],
                    lambda x: process_row(x, ml_threshold, model), pbar=True)

    # add metric
    eval = EvalMetric(bypass_pairing_check=True)
    df = add_columns(df, ['sensitivity', 'ppv', 'f_measure'],
                     ['one_idx', 'pred_idx', 'len'],
                     lambda a, b, c: calculate_metric(a, b, c, eval))

    df.to_pickle(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to model file')
    parser.add_argument('--dataset', type=str, help='path to dataset file')
    parser.add_argument('--threshold', type=float, default=None, help='threshold to get binary output, default to not using threshold (real value output)')
    parser.add_argument('--output', type=str, help='path to output file')
    args = parser.parse_args()

    main(args.model, args.dataset, args.threshold, args.output)
