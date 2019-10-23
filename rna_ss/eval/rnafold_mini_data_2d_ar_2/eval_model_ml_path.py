import yaml
import argparse
import numpy as np
import pandas as pd
from dgutils.pandas import add_column
from utils import PredictorSPlitModel, arr2db, EvalMetric
import logging


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def _process_row(seq, ml_threshold, model):
    pred, logp, fe = model.predict_one_step_ar(seq=seq, n_sample=1, start_offset=2,
                                               ml_path=True, ml_threshold=ml_threshold)  # hard-coded offset for now

    assert pred.shape[0] == 1
    idx_sample = 0
    _pred = pred[idx_sample, :, :, 0].copy()  # need to copy! otherwise following code will overwrite the values
    # replace missing value by 0, for visualization
    _pred[_pred == -1] = 0
    data_pred = np.where(_pred == 1)  # index of the sampled 1's
    return data_pred


def process_row(seq, n_sample, model):
    # TODO compute CHUNK_SIZE based on sequence length
    CHUNK_SIZE = 20
    _n = range(0, n_sample, CHUNK_SIZE)
    _n.append(n_sample)
    _n = [b - a for a, b in zip(_n[:-1], _n[1:])]  # batch sizes
    assert sum(_n) == n_sample
    if n_sample > CHUNK_SIZE:
        result = []
        for _l in _n:
            _r = _process_row(seq, _l, model)
            result.extend(_r)
    else:
        result = _process_row(seq, n_sample, model)
    return result


def main(model_file, dataset_file, n_sample, output):
    model = PredictorSPlitModel(model_file)
    # TODO make sure dataset has unified format
    df = pd.read_pickle(dataset_file)

    df = add_column(df, 'pred_idx', ['seq'],
                    lambda x: process_row(x, n_sample, model), pbar=True)

    df.to_pickle(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to model file')
    parser.add_argument('--dataset', type=str, help='path to dataset file')
    parser.add_argument('--threshold', type=float, default=0.5, help='maximum lilelihood path threshold')
    parser.add_argument('--output', type=str, help='path to output file')
    args = parser.parse_args()

    main(args.model, args.dataset, args.samples, args.output)
