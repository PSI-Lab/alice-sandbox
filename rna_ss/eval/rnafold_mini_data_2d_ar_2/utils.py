import keras.backend as kb
import tensorflow as tf
import logging
import tqdm
import re
import tempfile
from subprocess import PIPE, Popen
import numpy as np
from keras.layers import Convolution2D, Input
from keras.models import load_model, Model


def custom_loss(y_true, y_pred, mask_val=-1):  # mask val hard-coded for now
    # both are 3D array
    # num_examples x l1 x l2
    # find which values in yTrue (target) are the mask value
    is_mask = kb.equal(y_true, mask_val)  # true for all mask values
    is_mask = kb.cast(is_mask, dtype=kb.floatx())
    is_mask = 1 - is_mask  # now mask values are zero, and others are 1
    # reweight to account for proportion of missing value
    valid_entries = kb.cast(kb.sum(is_mask), dtype=kb.floatx())
    # total_entries = kb.cast(kb.prod(kb.shape(is_mask)), dtype=kb.floatx())

    def _loss(y_true, y_pred, is_mask):
        epsilon = tf.convert_to_tensor(kb.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        # return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), is_mask))
        return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred) + (1-y_true) * tf.log(1-y_pred), is_mask))

    # loss = kb.binary_crossentropy(kb.flatten(y_true), kb.flatten(y_pred)) * kb.flatten(is_mask)
    loss = _loss(y_true, y_pred, is_mask)
    loss = loss / valid_entries

    # loss = kb.mean(loss) * total_entries / valid_entries
    return loss


class TriangularConvolution2D(Convolution2D):

    def __init__(self, *args, **kwargs):
        # TODO input filter size should reflet the triangle length L
        # then we can initialize the base type using 2*L + 1
        super(TriangularConvolution2D, self).__init__(*args, **kwargs)

        self.mask = None

    def build(self, input_shape):
        super(TriangularConvolution2D, self).build(input_shape)

        # Create a numpy array of ones in the shape of our convolution weights.
        self.mask = np.ones(self.weights[0].shape)

        # We assert the height and width of our convolution to be equal as they should.
        assert self.mask.shape[0] == self.mask.shape[1]

        # for now let's make sure the filter width is odd number
        assert self.mask.shape[0] % 2 == 1

        # Since the height and width are equal, we can use either to represent the size of our convolution.
        filter_size = self.mask.shape[0]
        filter_center = filter_size // 2
        print(filter_center)

        # Zero out all weights above the center.
        self.mask[:filter_center, :, :, :] = 0

        # Zero out all weights to the right of the center.
        self.mask[:, (filter_center + 1):, :, :] = 0

        # zero out the little triangle to the left bottom
        # TODO right now this is being done in a stupid way
        for i in range(filter_center, self.mask.shape[0]):
            self.mask[i, :(i - filter_center), :, :] = 0

        # zero out the center weigths
        self.mask[filter_center, filter_center, :, :] = 0

        # Convert the numpy mask into a tensor mask.
        self.mask = kb.variable(self.mask)

    def call(self, x, mask=None):
        ''' I just copied the Keras Convolution2D call function so don't worry about all this code.
            The only important piece is: self.W * self.mask.
            Which multiplies the mask with the weights before calculating convolutions. '''
        output = kb.conv2d(x, self.weights[0] * self.mask, strides=(1, 1),
                           padding=self.padding, data_format=self.data_format,
                           dilation_rate=self.dilation_rate)

        if self.use_bias:
            output = kb.bias_add(
                output,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(output)

        return output


class DataEncoder(object):
    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    @staticmethod
    def _encode_seq(seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U',
                                                                                                          '4').replace(
            'N', '0')
        x = np.asarray(map(int, list(seq)))
        x = DataEncoder.DNA_ENCODING[x.astype('int8')]
        return x

    @staticmethod
    def encode_seqs(seqs):
        x = []
        assert all([len(s) == len(seqs[0]) for s in seqs])
        for s in seqs:
            x.append(DataEncoder._encode_seq(s))
        x = np.asarray(x)
        return x

    @staticmethod
    def _mask(x):
        assert len(x.shape) == 2
        assert x.shape[0] == x.shape[1]
        x[np.tril_indices(x.shape[0])] = -1
        return x

    @staticmethod
    def y_init(seq_len):
        y = np.zeros((seq_len, seq_len))
        y = DataEncoder._mask(y)
        return y[np.newaxis, :, :, np.newaxis]


def upper_band_index(l, n):
    assert n < l
    a = np.zeros((l, l))
    iu1 = np.triu_indices(l, n)
    iu2 = np.triu_indices(l, n+1)
    a[iu1] = 1
    a[iu2] = 0
    return np.where(a == 1)


def arr2db(arr):
    # TODO debug this code!!
    assert len(arr.shape) == 2
    assert arr.shape[0] == arr.shape[1]
    assert np.all((arr == 0) | (arr == 1))
    assert np.max(np.sum(arr, axis=0)) <= 1
    assert np.max(np.sum(arr, axis=1)) <= 1
    idx_pairs = np.where(arr == 1)
    idx_pairs = zip(idx_pairs[0], idx_pairs[1])

    db_str = ['.' for _ in range(len(arr))]
    for _i, _j in idx_pairs:
        i = min(_i, _j)
        j = max(_i, _j)
        db_str[i] = '('
        db_str[j] = ')'
    return ''.join(db_str)


def forna_url(seq, struct):
    url = "http://nibiru.tbi.univie.ac.at/forna/forna.html?id=url/name&sequence={}&structure={}".format(seq, struct)
    return url


def rna_eval_fe(seq, struct, verbose=True):
    # use RNAeval from ViennaRNA package to compute FE
    # checks
    assert len(seq) == len(struct)
    # call RNAeval
    p = Popen(['RNAeval'], stdin=PIPE,
              stdout=PIPE, stderr=PIPE, universal_newlines=True)
    stdout, stderr = p.communicate(input="{}\n{}".format(seq, struct))
    rc = p.returncode
    if rc != 0:
        msg = 'RNAeval returned error code %d\nstdout:\n%s\nstderr:\n%s\n' % (
            rc, stdout, stderr)
        # raise Exception(msg)
        if verbose:
            logging.warning(msg)
        return np.nan
    # parse output
    lines = stdout.splitlines()
    assert len(lines) == 2
    try:
        val = float(re.match(pattern=r".*\( *(-*\d+\.\d+)\)$", string=lines[1]).group(1))
    except AttributeError as e:
        # debug
        if verbose:
            print(lines)
        return np.nan
    return val


class EvalMetric(object):

    @staticmethod
    def _check_arr(arr):
        assert len(arr.shape) == 2
        assert arr.shape[0] == arr.shape[1]
        assert np.all((arr == 0) | (arr == 1))
        assert np.max(np.sum(arr, axis=0)) <= 1
        assert np.max(np.sum(arr, axis=1)) <= 1

    @staticmethod
    def sensitivity(_pred, _target):
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
        EvalMetric._check_arr(pred)
        EvalMetric._check_arr(target)
        # metric
        idx_true_base_pair = np.where(target == 1)
        return float(np.sum(pred[idx_true_base_pair]))/np.sum(target)

    @staticmethod
    def ppv(_pred, _target):
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
        EvalMetric._check_arr(pred)
        EvalMetric._check_arr(target)
        # metric
        idx_predicted_base_pair = np.where(pred == 1)
        return float(np.sum(target[idx_predicted_base_pair])/np.sum(pred))

    @staticmethod
    def f_measure(sensitivity, ppv):
        return (2 * sensitivity * ppv)/(sensitivity + ppv)


class PredictorSPlitModel(object):
    # split model into two parts
    # the first part is the 'representation', which only needs to be run once
    # second part is autoregressive, need to be run L-1 times

    def __init__(self, model_file):
        _model = load_model(model_file, custom_objects={'kb': kb, 'tf': tf,
                                                        'custom_loss': custom_loss,
                                                        'TriangularConvolution2D': TriangularConvolution2D})
        self.model_repr, self.model_ar = self.split_model(_model)

    def split_model(self, model):
        # check whether this trained model has named layers, if not print a warning
        is_named_layer = True
        if not any([l.name == 'concat_hid_target_prev' for l in model.layers]):
            logging.warning("Model missing names layers, will try to infer (unreliable!).")
            is_named_layer = False
        # find layers
        layer_input_org = next(l for l in model.layers if l.name == 'input_org')
        layer_target_prev = next(l for l in model.layers if l.name == 'target_prev')
        layer_ar_label = next(l for l in model.layers if l.name == 'ar_label')
        layer_fe = next(l for l in model.layers if l.name == 'fe')
        if is_named_layer:
            layer_final_hidden = next(l for l in model.layers if l.name == 'final_hidden')
            layer_concat = next(l for l in model.layers if l.name == 'concat_hid_target_prev')
            layer_tri_conv = next(l for l in model.layers if l.name == 'tri_conv')
            layer_conv_fe = next(l for l in model.layers if l.name == 'conv_fe')
            layer_mask_fe = next(l for l in model.layers if l.name == 'mask_fe')
        else:
            layer_final_hidden = next(l for l in model.layers if l.name == 'conv2d_6')
            layer_concat = next(l for l in model.layers if l.name == 'concatenate_3')
            layer_tri_conv = next(l for l in model.layers if l.name == 'triangular_convolution2d_1')
            layer_conv_fe = next(l for l in model.layers if l.name == 'conv2d_7')
            layer_mask_fe = next(l for l in model.layers if l.name == 'lambda_9')

        # first model
        model1 = Model(input=[layer_input_org.input, layer_target_prev.input],
                       output=[layer_final_hidden.output, layer_fe.output])
        # TODO layer_fe should be connected to the second model
        # in order to do so, we need extra named layers
        # training code to be updated
        # second model
        new_input = Input(layer_final_hidden.input_shape[1:])
        new_hid = layer_concat([new_input, layer_target_prev.input])
        new_output = layer_tri_conv(new_hid)
        new_output = layer_ar_label(new_output)
        model2 = Model(input=[new_input, layer_target_prev.input],
                       output=new_output)
        return model1, model2

    def predict_one_step_ar(self, seq, n_sample=1, start_offset=1, p_clip=1e-7):
        L = len(seq)
        # x = np.tile(DataEncoder.encode_seqs([seq]), [n_sample, 1, 1])
        x_single = DataEncoder.encode_seqs([seq])
        x = np.tile(x_single, [n_sample, 1, 1])
        y_single = DataEncoder.y_init(L)
        y = np.tile(y_single, [n_sample, 1, 1, 1])
        # log probabilities for sampled path
        logps = [[] for _ in range(n_sample)]

        # get representation
        # note that we can't use this fe, since it's on fake sequence
        logging.info("{}\nmodel_repr".format(seq))
        z_repr_single, _ = self.model_repr.predict([x_single, y_single])
        z_repr = np.tile(z_repr_single, [n_sample, 1, 1, 1])
        logging.info("model_ar")
        for n in tqdm.tqdm(range(start_offset, L)):
            tmp = self.model_ar.predict([z_repr, y])

            for idx_sample in range(n_sample):
                pred = tmp[idx_sample, :, :, 0]
                # sample n-th upper triangular band
                n_th_band_idx = upper_band_index(L, n)
                vals = pred[n_th_band_idx]
                threshold = np.random.uniform(0, 1, size=vals.shape)
                vals_sampled = (vals > threshold).astype(np.float32)

                # take into account that only a single 1 can show up in all rows/columns
                _y_old = y[idx_sample, :, :, 0]
                # FIXME slow + naive method
                already_paired = []

                # for position i, need to consider both row i and col i
                # similarly, for position j, need to consider both col j and row j
                # note that positions within the 'band' can also share the same index (e.g. one's row idx can be another's col idx)
                # so if one position is sampled to be 1, the other one cannot
                # collect these idxes while going through the positions
                new_sampled_idxes = set()
                for idx_in_band, (i, j, v) in enumerate(zip(range(0, L - n), range(n, L), vals_sampled)):
                    new_sampled_idxes.add(i)
                    new_sampled_idxes.add(j)
                    # lower triangular are all -1's don't sum over them!
                    i_row_sum = np.sum(_y_old[i, i + 1:])
                    j_col_sum = np.sum(_y_old[:j, j])
                    i_col_sum = np.sum(_y_old[:i, i])
                    j_row_sum = np.sum(_y_old[j, j + 1:])
                    assert 0 <= i_row_sum <= 1
                    assert 0 <= j_col_sum <= 1
                    assert 0 <= i_col_sum <= 1
                    assert 0 <= j_row_sum <= 1
                    total_sum = i_row_sum + j_col_sum + i_col_sum + j_row_sum
                    if total_sum > 0 or i in new_sampled_idxes or j in new_sampled_idxes:  # i and j cannot be paired, if at least one is paired with another position
                        already_paired.append(idx_in_band)

                # already_paired = np.asarray(already_paired)
                # setting those positions who has a already-paired base to 0
                # vals_sampled[np.where(already_paired)] = 0
                if len(already_paired) > 0:
                    vals_sampled[already_paired] = 0
                    # for those positions, set probability for y=1 to 0
                    vals[already_paired] = 0

                # log probability
                _vals = np.clip(vals, p_clip, 1 - p_clip)
                _lp = vals_sampled * np.log(_vals) + (1 - vals_sampled) * np.log(1 - _vals)
                logps[idx_sample].extend(_lp.tolist())

                # update y on n-th upper triangular band
                _y = y[idx_sample, :, :, 0]
                _y[n_th_band_idx] = vals_sampled
                y[idx_sample, :, :, 0] = _y

        # TODO layer_fe should be connected to the second model
        # for now if we want accurate fe, at the end of AR,
        # need to do another pass using the first model
        # after all sampling steps, run one more prediction using final y, to get predicted normalized energy
        logging.info("model_ar (for fe)")
        _, fe = self.model_repr.predict([x, y])
        assert len(fe.shape) == 2
        assert fe.shape[1] == 1

        logps = [np.sum(x) for x in logps]

        return y, logps, fe[:, 0]


# class Predictor(object):
#
#     def __init__(self, model_file):
#         self.model = load_model(model_file, custom_objects={'kb': kb, 'tf': tf,
#                                                             'custom_loss': custom_loss,
#                                                             'TriangularConvolution2D': TriangularConvolution2D})
#
#     def predict_one_step_ar(self, seq, n_sample=1, start_offset=1, p_clip=1e-7):
#         # (start_offset-1) determines the minimal size of a loop
#         L = len(seq)
#         x = np.tile(DataEncoder.encode_seqs([seq]), [n_sample, 1, 1])
#         y = np.tile(DataEncoder.y_init(L), [n_sample, 1, 1, 1])
#         # log probabilities for sampled path
#         logps = [[] for _ in range(n_sample)]
#
#         for n in range(start_offset, L):
#             # print(n)
#             tmp, _ = self.model.predict([x, y])
#             for idx_sample in range(n_sample):
#                 pred = tmp[idx_sample, :, :, 0]
#                 # sample n-th upper triangular band
#                 n_th_band_idx = upper_band_index(L, n)
#                 vals = pred[n_th_band_idx]
#                 threshold = np.random.uniform(0, 1, size=vals.shape)
#                 vals_sampled = (vals > threshold).astype(np.float32)
#
#                 # take into account that only a single 1 can show up in all rows/columns
#                 _y_old = y[idx_sample, :, :, 0]
#                 # FIXME slow + naive method
#                 already_paired = []
#
#                 # for position i, need to consider both row i and col i
#                 # similarly, for position j, need to consider both col j and row j
#                 for idx_in_band, (i, j) in enumerate(zip(range(0, L - n), range(n, L))):
#                     # lower triangular are all -1's don't sum over them!
#                     i_row_sum = np.sum(_y_old[i, i+1:])
#                     j_col_sum = np.sum(_y_old[:j, j])
#                     i_col_sum = np.sum(_y_old[:i, i])
#                     j_row_sum = np.sum(_y_old[j, j+1:])
#                     assert 0 <= i_row_sum <= 1
#                     assert 0 <= j_col_sum <= 1
#                     assert 0 <= i_col_sum <= 1
#                     assert 0 <= j_row_sum <= 1
#                     total_sum = i_row_sum + j_col_sum + i_col_sum + j_row_sum
#                     if total_sum > 0:   # i and j cannot be paired, if at least one is paired with another position
#                         already_paired.append(idx_in_band)
#
#                 # already_paired = np.asarray(already_paired)
#                 # setting those positions who has a already-paired base to 0
#                 # vals_sampled[np.where(already_paired)] = 0
#                 if len(already_paired) > 0:
#                     vals_sampled[already_paired] = 0
#                     # for those positions, set probability for y=1 to 0
#                     vals[already_paired] = 0
#
#                 # log probability
#                 _vals = np.clip(vals, p_clip, 1-p_clip)
#                 _lp = vals_sampled * np.log(_vals) + (1 - vals_sampled) * np.log(1 - _vals)
#                 logps[idx_sample].extend(_lp.tolist())
#
#                 # update y on n-th upper triangular band
#                 _y = y[idx_sample, :, :, 0]
#                 _y[n_th_band_idx] = vals_sampled
#                 y[idx_sample, :, :, 0] = _y
#
#         # after all sampling steps, run one more prediction using final y, to get predicted normalized energy
#         _, fe = self.model.predict([x, y])
#         assert len(fe.shape) == 2
#         assert fe.shape[1] == 1
#
#         logps = [np.sum(x) for x in logps]
#
#         return y, logps, fe[:, 0]
