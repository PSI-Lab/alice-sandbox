from keras.models import Model
from keras.layers import Input, Bidirectional, Concatenate, Dot, LSTM
from keras.layers.core import Activation, Dense, Lambda
from keras.layers.convolutional import Conv1D, Cropping1D, ZeroPadding1D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, multiply
from keras import regularizers, initializers
import keras.backend as kb
from keras.losses import binary_crossentropy, mean_squared_error, categorical_crossentropy
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np


def build_model():
    input_org = Input(shape=(50, 4), name='input_org')

    # input_rev = Input(shape=(51, 4), name='input_rev')  # can also use rev comp
    reverse_layer = Lambda(lambda x: kb.reverse(x, axes=-2))
    input_rev = reverse_layer(input_org)

    conv_prods = []
    num_filters = [64, 64, 64, 64, 64]
    kernel_sizes = [7, 3, 3, 5, 9]

    conv_or = input_org
    conv_rv = input_rev

    for num_filter, kernel_size in zip(num_filters, kernel_sizes):
        conv_or = BatchNormalization()(conv_or)
        conv_or = Activation('relu')(conv_or)
        conv_or = Conv1D(filters=num_filter, kernel_size=kernel_size, padding='same', activation=None)(conv_or)

        conv_rv = BatchNormalization()(conv_rv)
        conv_rv = Activation('relu')(conv_rv)
        conv_rv = Conv1D(filters=num_filter, kernel_size=kernel_size, padding='same', activation=None)(conv_rv)
        # conv_rv_mid = Cropping1D(25)(conv_rv)

        # # transformation matrix - general
        # conv_or_transform = Conv1D(filters=64, kernel_size=1, activation=None)(conv_or)
        # conv_prod = Dot(axes=-1)([conv_or_transform, conv_rv_mid])
        # conv_prods.append(conv_prod)

        # # replace dot product
        # # by tiling the rev
        # # stack on org
        # # run a mini fully connected net
        # conv_rd_mid_tiled = Lambda(kb.tile, arguments={'n': (1, 51, 1)})(conv_rv_mid)
        # conv_fw_rd = Concatenate(axis=-1)([conv_or, conv_rd_mid_tiled])
        # # TODO hard coded 2 filters
        # # size=1, fully connected along all features at each position
        # hid_fw_rd = Conv1D(filters=2, kernel_size=1, activation='tanh')(conv_fw_rd)
        # conv_prods.append(hid_fw_rd)

        # dot product
        conv_prod = Dot(axes=-1)([conv_or, conv_rv])  # 2D map
        conv_prods.append(conv_prod)

    # stack 2D feature maps
    stack_layer = Lambda(lambda x: kb.stack(x, axis=-1))
    conv_prod_concat = stack_layer(conv_prods)

    # # add 2D conv - TODO how do we guarantee symmetry of output?
    # nf_2d = [128, 128, 128]
    # nk_2d = [[2, 2], [4, 4], [8, 8]]
    # conv_2d = conv_prod_concat
    # for nf, nk in zip(nf_2d, nk_2d):
    #     conv_2d = BatchNormalization()(conv_2d)
    #     conv_2d = Activation('relu')(conv_2d)
    #     conv_2d = Conv2D(filters=nf, kernel_size=nk, padding='same', activation=None)(conv_2d)
    #
    # output = Conv2D(1, [1, 1], activation='sigmoid')(conv_2d)

    hid = Conv2D(filters=10, kernel_size=[5, 5],
                 padding='same', activation='relu')(conv_prod_concat)
    output = Conv2D(filters=1, kernel_size=[2, 2], dilation_rate=2,
                    padding='same', activation='sigmoid')(hid)

    model = Model(input=input_org, output=output)

    return model


# def build_model():
#     input_org = Input(shape=(51, 4), name='input_org')
#     input_rev = Input(shape=(51, 4), name='input_rev')  # can also use rev comp
#
#     # TODO tie weights for org and rev - maybe not!
#     # TODO filter kernel size odd number!
#
#     conv_prods = []
#     num_filters = [8, 16, 16, 32, 32]
#     kernel_sizes = [1, 3, 5, 9, 17]
#     for num_filter, kernel_size in zip(num_filters, kernel_sizes):
#         # conv_or = Conv1D(filters=num_filter, kernel_size=kernel_size, padding='same', activation=None)(input_org)
#         # conv_rv = Conv1D(filters=num_filter, kernel_size=kernel_size, padding='same', activation=None)(input_rev)
#         conv_or = Conv1D(filters=num_filter, kernel_size=kernel_size, padding='same', activation='relu')(input_org)
#         conv_rv = Conv1D(filters=num_filter, kernel_size=kernel_size, padding='same', activation='relu')(input_rev)
#         conv_rv_mid = Cropping1D(25)(conv_rv)
#
#         # TODO replace dot product
#         # by tiling the rev
#         # stack on org
#         # run a mini fully connected net
#
#         conv_prod = Dot(axes=-1)([conv_or, conv_rv_mid])
#         conv_prods.append(conv_prod)
#     conv_prod_concat = Concatenate(axis=-1)(conv_prods)
#
#     # fully connected along feature dimension
#     output = Conv1D(1, 1, padding='same', activation='sigmoid')(conv_prod_concat)
#
#     model = Model(input=[input_org, input_rev], output=output)
#
#     return model



