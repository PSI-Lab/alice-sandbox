from keras.models import Model
from keras.models import Model
from keras.layers import Input, Bidirectional, Masking
from keras.layers.core import Activation, Dense, Lambda
from keras.layers.convolutional import Conv1D, Cropping1D, ZeroPadding1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate, multiply
import keras.backend as kb
import numpy as np


def resolve_contex(residual_conv, n_repeat_in_residual_unit=2):
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
                act_tanh = Conv1D(l, w, dilation_rate=ar, padding='same', activation='tanh')(bn)
                act_sigmoid = Conv1D(l, w, dilation_rate=ar, padding='same', activation='sigmoid')(bn)
                conv = multiply([act_tanh, act_sigmoid])

        if residual:
            output_node = add([conv, input_node])
        else:
            output_node = conv

        return output_node

    return f


def build_model_residual_conv(L, residual_conv, n_repeat_in_residual_unit=2, skip_conn_every_n=4,
                residual=True, skipconn=True, gated=True, nout=7):
    context = resolve_contex(residual_conv, n_repeat_in_residual_unit)

    input0 = Input(shape=(None, 4))
    # input0 = Masking(mask_value=0.)(input0)  # conv1d does not support masking -_-
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

    skip_cropped = Cropping1D(context / 2)(skip)

    hidden = Conv1D(50, 1, activation='sigmoid')(skip_cropped)
    if nout == 1:
        output0 = Conv1D(1, 1, activation='sigmoid')(hidden)
    else:
        output0 = Conv1D(nout, 1, activation='softmax')(hidden)

    model = Model(input0, output0)

    return model


def split_data_by_rna_type(data):
    # data is list of tuples, with the first element in tuple being the RNA type, e.g. tRNA, RNase, etc.
    # returns a dictionary of rna_type -> list (where the rna_type has been removed)
    data_dict = dict()
    for d in data:
        rna_type = d[-1].lower()
        if rna_type not in data_dict:
            data_dict[rna_type] = []
        data_dict[rna_type].append(d[:-1])
    return data_dict

