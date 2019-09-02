import keras.backend as kb
import tensorflow as tf
import numpy as np
from keras.layers import Convolution2D
from keras.models import load_model


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


class Predictor(object):

    def __init__(self, model_file):
        self.model = load_model(model_file, custom_objects={'kb': kb, 'tf': tf,
                                                            'custom_loss': custom_loss,
                                                            'TriangularConvolution2D': TriangularConvolution2D})

    def predict_one_step_ar(self, seq, n_sample=1):
        L = len(seq)
        x = np.tile(DataEncoder.encode_seqs([seq]), [n_sample, 1, 1])
        y = np.tile(DataEncoder.y_init(L), [n_sample, 1, 1, 1])

        for n in range(1, L):
            print(n)
            tmp = self.model.predict([x, y])
            for idx_sample in range(n_sample):
                pred = tmp[idx_sample, :, :, 0]
                # sample n-th upper triangular band
                n_th_band_idx = upper_band_index(L, n)
                vals = pred[n_th_band_idx]
                threshold = np.random.uniform(0, 1, size=vals.shape)
                vals_sampled = (vals > threshold).astype(np.float32)

                # take into account that only a single 1 can show up in all rows/columns
                _y_old = y[0, :, :, 0]
                # FIXME slow + naive method
                already_paired = []
                for idx_in_band, (row_idx, col_idx) in enumerate(zip(range(0, L - n), range(n, L))):
                    row_sum = np.sum(_y_old[row_idx, row_idx + 1:col_idx])  # not including the current position
                    col_sum = np.sum(_y_old[row_idx + 1:col_idx, col_idx])  # not including the current position
                    assert row_sum <= 1
                    assert col_sum <= 1
                    total_sum = row_sum + col_sum
                    if total_sum > 0:   # i and j cannot be paired, if at least one is paired with another position
                        already_paired.append(idx_in_band)
                already_paired = np.asarray(already_paired)
                # setting those positions who has a already-paired base to 0
                # vals_sampled[np.where(already_paired)] = 0
                if len(already_paired) > 0:
                    vals_sampled[already_paired] = 0

                # update y on n-th upper triangular band
                _y = y[idx_sample, :, :, 0]
                _y[n_th_band_idx] = vals_sampled
                y[idx_sample, :, :, 0] = _y

        return y
