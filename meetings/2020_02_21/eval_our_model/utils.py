import numpy as np


class EvalMetric(object):

    def __init__(self, bypass_pairing_check=False):
        # bypass checking that a single base can only be paired against 0 or 1 base
        self.bypass_pairing_check = bypass_pairing_check

    def _check_arr(self, arr):
        assert len(arr.shape) == 2
        assert arr.shape[0] == arr.shape[1]
        assert np.all((arr == 0) | (arr == 1))
        if not self.bypass_pairing_check:
            assert np.max(np.sum(arr, axis=0)) <= 1
            assert np.max(np.sum(arr, axis=1)) <= 1

    def sensitivity(self, _pred, _target):
        # numerator: number of correct predicted base pairs
        # denominator: number of true base pairs
        assert _pred.shape[0] == _target.shape[0]
        n = _pred.shape[0]
        # set lower triangular to be all 0's
        pred = _pred.copy()
        target = _target.copy()
        pred[np.tril_indices(n)] = 0
        target[np.tril_indices(n)] = 0
        # checks
        self._check_arr(pred)
        self._check_arr(target)
        # metric
        idx_true_base_pair = np.where(target == 1)
        return float(np.sum(pred[idx_true_base_pair]))/np.sum(target)

    def ppv(self, _pred, _target):
        # numerator: number of correct predicted base pairs
        # denominator: number of predicted base pairs
        assert _pred.shape[0] == _target.shape[0]
        n = _pred.shape[0]
        # set lower triangular to be all 0's
        pred = _pred.copy()
        target = _target.copy()
        pred[np.tril_indices(n)] = 0
        target[np.tril_indices(n)] = 0
        # checks
        self._check_arr(pred)
        self._check_arr(target)
        # metric
        idx_predicted_base_pair = np.where(pred == 1)
        return float(np.sum(target[idx_predicted_base_pair])/np.sum(pred))

    @staticmethod
    def f_measure(sensitivity, ppv):
        return (2 * sensitivity * ppv)/(sensitivity + ppv)
