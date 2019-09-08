from keras.models import Model
from keras.layers import Input, Bidirectional, Concatenate, Dot, LSTM, Convolution2D, GlobalAveragePooling2D
from keras.layers.core import Activation, Dense, Lambda
from keras.layers.convolutional import Conv1D, Cropping1D, ZeroPadding1D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, multiply, concatenate
from keras import regularizers, initializers
import keras.backend as kb
from keras.losses import binary_crossentropy, mean_squared_error, categorical_crossentropy
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np


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

        # zero out the center weights
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

    # def get_config(self):
    #     config = {'mask': self.mask}
    #     base_config = super(TriangularConvolution2D, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

    # def get_config(self):
    #     # Add the mask type property to the config.
    #     return dict(
    #         list(super(TriangularConvolution2D, self).get_config().items()))


def build_model():
    L12_P = 0.000005

    input_org = Input(shape=(None, 4), name='input_org')
    # target from the previous 'time stamp'
    target_ar = Input(shape=(None, None, 1), name='target_prev')

    conv_prods = []
    num_filters = [256, 256, 256, 256, 256]
    kernel_sizes = [7, 5, 5, 5, 5]
    dilation_sizes = [1, 2, 2, 4, 4]

    conv_or = input_org
    conv_rv = input_org

    def repeat_1(x):
        xr = kb.tile(kb.expand_dims(x, axis=-2), [1, 1, kb.shape(x)[1], 1])
        return xr

    def repeat_2(x):
        xr = kb.tile(kb.expand_dims(x, axis=-3), [1, kb.shape(x)[1], 1, 1])
        return xr

    # stacking single nucleotide
    # -1 is feature dim (d=4), -2 is length
    x1_2 = Lambda(repeat_1)(input_org)
    x2_2 = Lambda(repeat_2)(input_org)
    input_nt_stack = concatenate([x1_2, x2_2], axis=-1)

    for num_filter, kernel_size, dilation_size in zip(num_filters, kernel_sizes, dilation_sizes):
        conv_or = BatchNormalization()(conv_or)
        conv_or = Activation('relu')(conv_or)
        conv_or = Conv1D(filters=num_filter, kernel_size=kernel_size, dilation_rate=dilation_size,
                         kernel_regularizer=regularizers.l1_l2(l1=L12_P, l2=L12_P),
                         padding='same', activation=None)(conv_or)

        conv_rv = BatchNormalization()(conv_rv)
        conv_rv = Activation('relu')(conv_rv)
        conv_rv = Conv1D(filters=num_filter, kernel_size=kernel_size, dilation_rate=dilation_size,
                         kernel_regularizer=regularizers.l1_l2(l1=L12_P, l2=L12_P),
                         padding='same', activation=None)(conv_rv)

        # dot product
        conv_prod = Dot(axes=-1)([conv_or, conv_rv])  # 2D map
        # select upper triangular part (lower will be all 0's)
        upper_tri_layer = Lambda(lambda x: tf.matrix_band_part(x, 0, -1))
        conv_prod = upper_tri_layer(conv_prod)
        conv_prods.append(conv_prod)

    # stack 2D feature maps
    stack_layer = Lambda(lambda x: kb.stack(x, axis=-1))
    # we're doing this to merge multiple (?, L, L) layers to (?, L, L, K)
    conv_prod_concat = stack_layer(conv_prods)

    # add input nt stack
    conv_prod_concat = concatenate([conv_prod_concat, input_nt_stack], axis=-1)

    # add target label from previous time stamp
    # hid = Concatenate(axis=-1)([conv_prod_concat, target_ar])

    hid = Conv2D(20, (3, 3), padding='same', activation='relu')(conv_prod_concat)
    hid = Conv2D(20, (6, 6), padding='same', activation='relu')(hid)
    hid = Conv2D(20, (6, 6), dilation_rate=2,
                 padding='same', activation='relu')(hid)
    hid = Conv2D(20, (9, 9), dilation_rate=2,
                 padding='same', activation='relu')(hid)
    hid = Conv2D(20, (17, 17), dilation_rate=4,
                 padding='same', activation='relu')(hid)
    hid = Conv2D(20, (17, 17), dilation_rate=4,
                 padding='same', activation='relu')(hid)

    # auto regressive output label
    # triangular conv for ar label
    hid = Concatenate(axis=-1)([hid, target_ar])
    tri_conv = TriangularConvolution2D(20, (9, 9),
                                       padding='same', activation='relu')(hid)
    # output
    output1 = Conv2D(1, (1, 1), padding='same',
                     activation='sigmoid', name='ar_label')(tri_conv)

    # normalized energy

    def _mask_lower_tri_and_padding(input_nodes):
        x, y = input_nodes
        ones = tf.ones(kb.shape(x)[1:3])
        mask_a = tf.matrix_band_part(ones, 0, -1)   # diagonal + upper = 1
        mask_b = tf.matrix_band_part(ones, 0, 0)  # diagonal = 1
        tri_mask = mask_a - mask_b  # upper = 1
        # broadcast
        tri_mask = kb.tile(kb.expand_dims(kb.expand_dims(tri_mask, -1), 0), [kb.shape(x)[0], 1, 1, 1])

        pad_mask = kb.equal(y, -1)
        pad_mask = kb.cast(pad_mask, dtype=kb.floatx())
        pad_mask = 1 - pad_mask  # now mask values are zero, and others are 1

        # element wise multiply the two masks
        mask = tri_mask * pad_mask

        return x * mask

    # fc for fe, re-use same hid
    hid_fe = Conv2D(1, (6, 6), padding='same', activation='relu')(hid)
    hid_fe_masked = Lambda(_mask_lower_tri_and_padding)([hid_fe, target_ar])
    # global pooling
    output2 = GlobalAveragePooling2D(name='fe')(hid_fe_masked)

    model = Model(input=[input_org, target_ar], output=[output1, output2])

    return model


# def build_model():
#     L12_P = 0.000005
#
#     input_org = Input(shape=(None, 4), name='input_org')
#     # target from the previous 'time stamp'
#     target_ar = Input(shape=(None, None, 1), name='target_prev')
#
#     conv_prods = []
#     num_filters = [256, 256, 256, 256, 256]
#     kernel_sizes = [7, 5, 5, 5, 5]
#     dilation_sizes = [1, 2, 2, 4, 4]
#
#     conv_or = input_org
#     conv_rv = input_org
#
#     def repeat_1(x):
#         xr = kb.tile(kb.expand_dims(x, axis=-2), [1, 1, kb.shape(x)[1], 1])
#         return xr
#
#     def repeat_2(x):
#         xr = kb.tile(kb.expand_dims(x, axis=-3), [1, kb.shape(x)[1], 1, 1])
#         return xr
#
#     # stacking single nucleotide
#     # -1 is feature dim (d=4), -2 is length
#     x1_2 = Lambda(repeat_1)(input_org)
#     x2_2 = Lambda(repeat_2)(input_org)
#     input_nt_stack = concatenate([x1_2, x2_2], axis=-1)
#
#     for num_filter, kernel_size, dilation_size in zip(num_filters, kernel_sizes, dilation_sizes):
#         conv_or = BatchNormalization()(conv_or)
#         conv_or = Activation('relu')(conv_or)
#         conv_or = Conv1D(filters=num_filter, kernel_size=kernel_size, dilation_rate=dilation_size,
#                          kernel_regularizer=regularizers.l1_l2(l1=L12_P, l2=L12_P),
#                          padding='same', activation=None)(conv_or)
#
#         conv_rv = BatchNormalization()(conv_rv)
#         conv_rv = Activation('relu')(conv_rv)
#         conv_rv = Conv1D(filters=num_filter, kernel_size=kernel_size, dilation_rate=dilation_size,
#                          kernel_regularizer=regularizers.l1_l2(l1=L12_P, l2=L12_P),
#                          padding='same', activation=None)(conv_rv)
#
#         # dot product
#         conv_prod = Dot(axes=-1)([conv_or, conv_rv])  # 2D map
#         # select upper triangular part (lower will be all 0's)
#         upper_tri_layer = Lambda(lambda x: tf.matrix_band_part(x, 0, -1))
#         conv_prod = upper_tri_layer(conv_prod)
#         conv_prods.append(conv_prod)
#
#     # stack 2D feature maps
#     stack_layer = Lambda(lambda x: kb.stack(x, axis=-1))
#     conv_prod_concat = stack_layer(conv_prods)  # we're doing this to merge multiple (?, 50, 50) layers to (?, 50, 50, K)
#
#     # add input nt stack
#     conv_prod_concat = concatenate([conv_prod_concat, input_nt_stack], axis=-1)
#
#     # add target label from previous time stamp
#     hid = Concatenate(axis=-1)([conv_prod_concat, target_ar])
#
#     # triangular conv
#     # 2x2 (5//2 =2)
#     hid = TriangularConvolution2D(20, (5, 5), padding='same', activation='relu')(hid)
#     # 4x4 (9//2 = 4)
#     hid = TriangularConvolution2D(20, (9, 9), padding='same', activation='relu')(hid)
#     # 8x8 (17 //2 = 8)
#     hid = TriangularConvolution2D(20, (17, 17),
#                                   padding='same', activation='relu')(hid)
#     # output
#     output = TriangularConvolution2D(1, (17, 17),
#                                      padding='same', activation='sigmoid')(hid)
#
#     model = Model(input=[input_org, target_ar], output=output)
#
#     return model


# FIXME is this working?
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
