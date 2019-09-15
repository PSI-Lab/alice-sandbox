import yaml
import argparse
import numpy as np
import pandas as pd
from dgutils.pandas import add_columns
from utils import PredictorSPlitModel, arr2db, EvalMetric
import logging


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def process_row(seq, one_idx, n_sample, model):
    target = np.zeros((len(seq), len(seq)))
    target[one_idx] = 1
    pred, logp, fe = model.predict_one_step_ar(seq=seq, n_sample=n_sample, start_offset=2)  # hard-coded offset for now

    data_pred = []   # index of the sampled 1's
    data_sensitivity = []
    data_ppv = []
    data_f = []
    for idx_sample in range(n_sample):
        _pred = pred[idx_sample, :, :, 0].copy()  # need to copy! otherwise following code will overwrite the values
        # replace missing value by 0, for visualization
        _pred[_pred == -1] = 0
        # metric
        sensitivity = EvalMetric.sensitivity(_pred, target)
        ppv = EvalMetric.ppv(_pred, target)
        f_measure = EvalMetric.f_measure(sensitivity, ppv)
        data_pred.append(np.where(_pred == 1))
        data_sensitivity.append(sensitivity)
        data_ppv.append(ppv)
        data_f.append(f_measure)
    return data_pred, data_sensitivity, data_ppv, data_f, logp, fe


def main(model_file, dataset_file, n_sample, output):
    model = PredictorSPlitModel(model_file)
    # TODO make sure dataset has unified format
    df = pd.read_pickle(dataset_file)

    df = add_columns(df, ['pred_idx', 'sensitivity', 'ppv', 'f_measure', 'logp', 'fe'], ['seq', 'one_idx'],
                     lambda x, y: process_row(x, y, n_sample, model), pbar=False)

    df.to_pickle(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to model file')
    parser.add_argument('--dataset', type=str, help='path to dataset file')
    parser.add_argument('--samples', type=int, help='number of output to sample per each sequence')
    parser.add_argument('--output', type=str, help='path to output file')
    args = parser.parse_args()

    main(args.model, args.dataset, args.samples, args.output)

