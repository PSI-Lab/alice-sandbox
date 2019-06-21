import argparse
import numpy as np
import cPickle as pickle
import pandas as pd
import os
import random
import keras
import tensorflow as tf
import keras.backend as kb
from keras.models import Sequential, Model
from keras.layers import Masking, Dense, Conv2D, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import log_loss, accuracy_score
from time import gmtime, strftime
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger
from utils import split_data_by_rna_type


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


class DataGenerator(keras.utils.Sequence):
    # N will be all 0's - for gradient masking
    DNA_ENCODING = np.zeros((5, 4))
    DNA_ENCODING[1:, :] = np.eye(4)

    def __init__(self, data, batch_size=10):
        self.data = data
        self.batch_size = batch_size
        self.all_idx = range(len(data))
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.all_idx) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.all_idx[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)
        return x, y

    def on_epoch_end(self):
        random.shuffle(self.all_idx)

    def debug_get_data(self, indexes):
        return self.__data_generation(indexes)

    def __data_generation(self, indexes):
        # # max len in minibatch - do not remove, in case we get dynamic dimension working for 2D training
        # len_mb = max([len(self.data[idx][0]) for idx in indexes])

        # FIXME hard-coded to fix batch dimension, for flatten
        len_mb = 500

        _data_input = np.zeros((len(indexes), len_mb, len_mb, 8))
        # _data_output = np.zeros((self.batch_size, len_mb, len_mb, 1))
        # # flattened output
        # _data_output = np.zeros((len(indexes), len_mb**2))

        # 2D output
        _data_output = np.zeros((len(indexes), len_mb, len_mb))

        for k, idx in enumerate(indexes):
            _d = self.data[idx]
            seq = _d[0]
            pair_idx = _d[2]
            pair_idx = [x - 1 for x in pair_idx]

            seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('U', '4').replace('N', '0')
            x = np.asarray(map(int, list(seq)))
            x = self.DNA_ENCODING[x.astype('int8')]
            # _seq_map = np.zeros((len_mb, len_mb, 8))
            _val_h = np.repeat(x[:, np.newaxis, :], len(seq), axis=1)
            _val_v = np.repeat(x[np.newaxis, :, :], len(seq), axis=0)
            _tmp_seq_map = np.concatenate((_val_h, _val_v), axis=2)
            _seq_map = np.zeros((len_mb, len_mb, 8))
            _seq_map[:len(seq), :len(seq), :] = _tmp_seq_map

            # encode output
            _pair_map = np.zeros((len_mb, len_mb))
            for i, j in enumerate(pair_idx):
                if j == -1:  # unpaired
                    continue
                else:
                    if i < j:
                        # only set value in upper triangular matrix
                        _pair_map[i, j] = 1
            # mask lower diagonal part
            il1 = np.tril_indices(len_mb)
            _pair_map[il1] = -1

            # _pair_map = _pair_map[:, :, np.newaxis]

            _data_input[k, :, :, :] = _seq_map
            # _data_output[k, :, :, :] = _pair_map
            # _data_output[k, :] = _pair_map.flatten()
            _data_output[k, :, :] = _pair_map

        return _data_input, _data_output


class Histories(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class ValidationSetLosses(Callback):
    def __init__(self, datagen_v, debug=False):
        self.datagen_v = datagen_v
        self.debug = debug
        super(ValidationSetLosses, self).__init__()

    def on_train_begin(self, logs={}):
        # self.losses = []
        self.accuracies = []
        self.top_3_accuracies = []
        self.top_5_accuracies = []
        self.top_10_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        # losses = []
        accuracies = []
        top_3_accuracies = []
        top_5_accuracies = []
        top_10_accuracies = []

        if self.debug is False:
            eval_idx = range(len(self.datagen_v))  # all
        else:
            eval_idx = [0]  # for debug, pick the first one

        for batch_idx in eval_idx:
            _x, _y = self.datagen_v[batch_idx]
            _yp = self.model.predict(_x)
            assert _x.shape[0] == _y.shape[0]
            assert _y.shape[0] == _yp.shape[0]

            for example_idx in range(_x.shape[0]):
                x = _x[example_idx, :, :, :]
                y = _y[example_idx, :, :]
                yp = _yp[example_idx, :, :]
                # reshape
                assert y.shape[0] == yp.shape[0]
                # img_size = int(np.sqrt(y.shape[0]))
                # y = np.reshape(y, (img_size, img_size))
                # yp = np.reshape(yp, (img_size, img_size))

                # find padding idx
                tmp = np.sum(np.sum(x, axis=2), axis=1)
                padding_idx = np.where(tmp == 0)[0][0]

                # # only report on positions that are paired
                # tmp_y = y.copy()  # replace mask with 0
                # tmp_y[np.tril_indices(tmp_y.shape[0])] = 0
                # paired_idx = np.where(np.sum(y, axis=1) != 0)[0]

                # # slice target and prediction
                # y_sliced = y[paired_idx, :padding_idx]
                # yp_sliced = yp[paired_idx, :padding_idx]

                # only need to focus on first half
                slice_idx = padding_idx/2  # FIXME this seems wrong!

                #y_sliced = y[:padding_idx, :padding_idx]
                #yp_sliced = yp[:padding_idx, :padding_idx]
                y_sliced = y[:slice_idx, :slice_idx]
                yp_sliced = yp[:slice_idx, :slice_idx]
                
                # set lower diagonal of yp_sliced to -1
                yp_sliced[np.tril_indices(yp_sliced.shape[0])] = -1

                # debug
                #print 'padding_idx', padding_idx
                #print 'y_sliced row max', np.argmax(y_sliced, axis=1)
                #print 'yp_sliced row max', np.argmax(yp_sliced, axis=1)
                #print 'sum', np.sum(y_sliced), np.sum(yp_sliced)
                #print 'y_sliced', y_sliced
                #print 'yp_sliced', yp_sliced

                # find locaitons where y_sliced == 1
                idx_1 = np.where(y_sliced == 1)
                #print 'idx_1', idx_1
                #print 'yp at idx_1', yp_sliced[idx_1]

                # pick rows where we have 1
                # row idx should be unique by design, but we're taking the set anyways
                idx_1_row = np.asarray(list(set(idx_1[0])), dtype=int)
                y_sliced = y_sliced[idx_1_row, :]
                yp_sliced = yp_sliced[idx_1_row, :]
                #print 'y_sliced row max', np.argmax(y_sliced, axis=1)
                #print 'yp_sliced row max', np.argmax(yp_sliced, axis=1)
                
                # compute loss and accuracy for each row
                # losses.append(log_loss(y_sliced, yp_sliced))
                accuracies.append(accuracy_score(np.argmax(y_sliced, axis=1),
                                                 np.argmax(yp_sliced, axis=1)))

                # top k accuracies
                target_idx = np.argmax(y_sliced, axis=1)
                # sort descending
                sort_idx = np.argsort(- yp_sliced, axis=1)

                def top_k_accuracy(k):
                    # repeat target idx k times
                    _target_idx = np.repeat(target_idx[:, np.newaxis], k, axis=1)
                    # slice the top k idx
                    _sort_idx = sort_idx[:, :k]
                    return np.sum(np.max(_target_idx == _sort_idx, axis=1)) / float(_sort_idx.shape[0])

                top_3_accuracies.append(top_k_accuracy(3))
                top_5_accuracies.append(top_k_accuracy(5))
                top_10_accuracies.append(top_k_accuracy(10))

        # losses = pd.Series(losses)
        accuracies = pd.Series(accuracies)
        # print 'cross entropy'
        # print losses.describe()
        print 'accuracy'
        print accuracies.describe()
        # self.losses.append(losses.median())
        self.accuracies.append(accuracies.median())
        top3_accuracies = pd.Series(top_3_accuracies)
        top5_accuracies = pd.Series(top_5_accuracies)
        top10_accuracies = pd.Series(top_10_accuracies)
        print 'top 3'
        print top3_accuracies.describe()
        print 'top 5'
        print top5_accuracies.describe()
        print 'top 10'
        print top10_accuracies.describe()
        self.top_3_accuracies.append(top3_accuracies.median())
        self.top_5_accuracies.append(top5_accuracies.median())
        self.top_10_accuracies.append(top10_accuracies.median())
        return


def main(split):
    # load dataset
    with open('data/data.pkl', 'rb') as f:
        data_all = pickle.load(f)

    # training/validation
    if split == 'random':
        random.shuffle(data_all)
        num_training = int(len(data_all) * 0.8)
        data_training = data_all[:num_training]
        data_validation = data_all[num_training:]
        datagen_t = DataGenerator(data_training)
        datagen_v = DataGenerator(data_validation)
    else:
        data_dict = split_data_by_rna_type(data_all)
        assert split in data_dict.keys()
        data_validation = data_dict[split]
        data_training = [data_dict[x] for x in data_dict.keys() if x != split]
        data_training = [x for y in data_training for x in y]
        print("validation RNA type: %s" % split)
        print("training RNAs: %d" % len(data_training))
        print("validation RNAs: %d" % len(data_validation))
        datagen_t = DataGenerator(data_training)
        datagen_v = DataGenerator(data_validation)

    model = Sequential()
    # model.add(Masking(mask_value=0., input_shape=(None, None, 8)))  # not supported for 2d conv =(
    # model.add(Conv2D(32, 8, padding='same', dilation_rate=1, activation='relu', input_shape=(None, None, 8),
    #                  batch_input_shape=(10, 500, 500, 8)))  # had to use this for flatten
    model.add(Conv2D(32, 8, padding='same', dilation_rate=1, activation='relu', input_shape=(None, None, 8)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 8, padding='same', dilation_rate=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 8, padding='same', dilation_rate=4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 8, padding='same', dilation_rate=8, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 8, padding='same', dilation_rate=16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # # upper triangular
    # model.add(Lambda(lambda x: tf.matrix_band_part(x, 0, -1)))
    # model.add(Flatten())
    model.add(Lambda(lambda x: kb.squeeze(x, axis=3)))

    # model.compile(loss='binary_crossentropy', optimizer='adam')
    model.compile(loss=custom_loss, optimizer='adam')

    # TODO row/col wise softmax
    # mask lower triangular in loss

    tictoc = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    log_dir = 'log_' + tictoc
    os.mkdir(log_dir)
    run_dir = 'run_' + tictoc
    os.mkdir(run_dir)

    callbacks = [
        Histories(),
        ValidationSetLosses(datagen_v),
        ValidationSetLosses(datagen_t, debug=True),  # FIXME print training metrics for debug
        # EarlyStopping(monitor='val_loss', patience=5),  # FIXME removed for debugging
        ModelCheckpoint(os.path.join(run_dir, 'checkpoint.{epoch:03d}.hdf5'),
                        save_best_only=False, period=1),
        CSVLogger(os.path.join(run_dir, 'history.csv')),
    ]

    model.fit_generator(generator=datagen_t,
                        validation_data=datagen_v,
                        validation_steps=10,
                        callbacks=callbacks,
                        shuffle=True,
                        epochs=100,
                        use_multiprocessing=True,
                        workers=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split",
                        default="random",
                        help="How to split training/validation set. "
                             "random: randomly select 20% RNAs to be validation data, regardless of type."
                             "5s: 5s RNAs used as validation data."
                             "rnasep: RNaseP used as validation data."
                             "trna: tRNA used as validation data."
                             "tmrna: tmRNA used as validation data.")
    args = parser.parse_args()
    main(args.split)




