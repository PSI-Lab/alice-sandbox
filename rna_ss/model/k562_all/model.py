from keras.models import Model
from keras.layers import Input, Bidirectional
from keras.layers.core import Activation, Dense, Lambda
from keras.layers.convolutional import Conv1D, Cropping1D, ZeroPadding1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate, multiply
import keras.backend as kb
from keras.losses import binary_crossentropy, mean_squared_error, categorical_crossentropy
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np


def resolve_contex(residual_conv, n_repeat_in_residual_unit):
    n = 0
    for layer_config in residual_conv:
        n += (layer_config['filter_width'] - 1) * layer_config['dilation']
    return n * n_repeat_in_residual_unit


def residual_unit(l, w, ar, n_repeat_in_residual_unit, residual=True, gated=True):
    # Residual unit proposed in "Identity mappings in Deep Residual Networks"
    # by He et al.

    def f(input_node):

        conv = Lambda(lambda x: kb.identity(x))(input_node)

        for _ in range(n_repeat_in_residual_unit):
            bn = BatchNormalization()(conv)
            if not gated:
                act = Activation('relu')(bn)
                conv = Conv1D(l, w, dilation_rate=ar, padding='same')(act)
            else:
                # act_tanh = Activation('tanh')(bn)
                # act_sigmoid = Activation('sigmoid')(bn)
                # act = multiply([act_tanh, act_sigmoid])
                # conv = Conv1D(l, w, dilation_rate=ar, padding='same')(act)
                
                # TODO (shreshth)
                # Some people have reported gated linear units working better than gated tanh units on some tasks. I don't have any intuition or preference, but may be worth trying. Ref. https://arxiv.org/abs/1612.08083
                act_tanh = Conv1D(l, w, dilation_rate=ar, padding='same', activation='tanh')(bn)
                act_sigmoid = Conv1D(l, w, dilation_rate=ar, padding='same', activation='sigmoid')(bn)
                conv = multiply([act_tanh, act_sigmoid])

        if residual:
            output_node = add([conv, input_node])
        else:
            output_node = conv

        return output_node

    return f


def build_model(L, residual_conv, n_repeat_in_residual_unit, skip_conn_every_n,
                residual=True, skipconn=True, gated=True):
    context = resolve_contex(residual_conv, n_repeat_in_residual_unit)

    input0 = Input(shape=(None, 4), name='input0')
    conv = Conv1D(L, 1)(input0)
    skip = Conv1D(L, 1)(conv)

    for i, layer_config in enumerate(residual_conv):
        conv = residual_unit(layer_config['num_filter'], layer_config['filter_width'], layer_config['dilation'],
                             n_repeat_in_residual_unit, residual, gated)(conv)

        if (i + 1) % skip_conn_every_n == 0 or (i + 1) == len(residual_conv):
            if skipconn:
                # Skip connections to the output after every 4 residual units
                dense = Conv1D(L, 1)(conv)
                skip = add([skip, dense])
            else:
                skip = Conv1D(L, 1)(conv)

    # skip_cropped = Cropping1D(context / 2)(skip)
    #
    # # [0, 1]
    # output0 = Conv1D(3, 1, activation='sigmoid')(skip_cropped)

    hid = Cropping1D(context / 2)(skip)
    for n_units in [50, 10]:
        hid = Conv1D(n_units, 1, activation='relu')(hid)
    output0 = Conv1D(3, 1, activation='sigmoid')(hid)

    # # remove single dimension
    # output0 = Lambda(lambda x: kb.squeeze(x, axis=2), name='output0')(output0)

    model = Model(inputs=input0, outputs=output0)

    return model


def custom_loss(y_true, y_pred, mask_val=-1):
    # both are 3D array
    # num_examples x length
    # find which values in yTrue (target) are the mask value
    is_mask = kb.equal(y_true, mask_val)  # true for all mask values
    is_mask = kb.cast(is_mask, dtype=kb.floatx())
    is_mask = 1 - is_mask  # now mask values are zero, and others are 1
    # reweight to account for proportion of missing value
    valid_entries = kb.sum(is_mask)
    total_entries = kb.cast(kb.prod(kb.shape(is_mask)), dtype=kb.floatx())

    # need to multiply by the mask before averaging over minibatch and length dimension

    def _to_tensor(x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
        # Returns
            A tensor.
        """
        x = tf.convert_to_tensor(x)
        if x.dtype != dtype:
            x = tf.cast(x, dtype)
        return x

    def ce(output, target, mask):
        # manual computation of crossentropy
        epsilon = _to_tensor(10e-8, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        return - tf.reduce_mean(target * tf.log(output) * mask + (1-target) * tf.log(1-output) * mask)
        # return - tf.reduce_sum(target * tf.log(output) * mask,
        #                        reduction_indices=len(output.get_shape()) - 1)

    return ce(y_pred, y_true, is_mask) * total_entries / valid_entries
