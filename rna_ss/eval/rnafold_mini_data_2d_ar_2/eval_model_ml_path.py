import yaml
import argparse
import numpy as np
import pandas as pd
from dgutils.pandas import add_column
from utils import PredictorSPlitModel, arr2db, EvalMetric
import logging


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def process_row(seq, ml_threshold, model):
    pred, logp, fe = model.predict_one_step_ar(seq=seq, n_sample=1, start_offset=2,
                                               ml_path=True, ml_threshold=ml_threshold)  # hard-coded offset for now

    assert pred.shape[0] == 1
    idx_sample = 0
    _pred = pred[idx_sample, :, :, 0].copy()  # need to copy! otherwise following code will overwrite the values
    # replace missing value by 0, for visualization
    _pred[_pred == -1] = 0
    data_pred = np.where(_pred == 1)  # index of the sampled 1's
    return data_pred


def main(model_file, dataset_file, ml_threshold, output):
    model = PredictorSPlitModel(model_file)
    # TODO make sure dataset has unified format
    df = pd.read_pickle(dataset_file)

    df = add_column(df, 'pred_idx', ['seq'],
                    lambda x: process_row(x, ml_threshold, model), pbar=True)

    df.to_pickle(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to model file')
    parser.add_argument('--dataset', type=str, help='path to dataset file')
    parser.add_argument('--threshold', type=float, default=0.5, help='maximum lilelihood path threshold')
    parser.add_argument('--output', type=str, help='path to output file')
    args = parser.parse_args()

    main(args.model, args.dataset, args.threshold, args.output)
