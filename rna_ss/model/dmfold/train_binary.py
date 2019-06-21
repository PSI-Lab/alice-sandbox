"""paired v.s. unpaired"""
import argparse
import numpy as np
import cPickle as pickle
import pandas as pd
import os
import random
import keras
from time import gmtime, strftime
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import log_loss, accuracy_score
from utils import build_model_residual_conv, resolve_contex, split_data_by_rna_type


class DataGenerator(keras.utils.Sequence):
    # N will be all 0's - for gradient masking
    DNA_ENCODING = np.zeros((5, 4))
    DNA_ENCODING[1:, :] = np.eye(4)
    # 0 will be all 0's - for gradient masking
    OUTPUT_ENCODING = np.zeros((8, 7))
    OUTPUT_ENCODING[1:, :] = np.eye(7)

    LEN = 500  # this is > max length of sequence in dataset

    def __init__(self, data, context, batch_size=10):
        self.data = data
        self.batch_size = batch_size
        self.all_idx = range(len(data))
        self.on_epoch_end()
        self.context = context

    def __len__(self):
        return int(len(self.all_idx) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.all_idx[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)
        return x, y

    def on_epoch_end(self):
        random.shuffle(self.all_idx)

    def debug_get_data(self, indexes, pad=True):
        return self.__data_generation(indexes, pad)

    def __data_generation(self, indexes, pad=True):
        _data_input = []
        _data_output = []

        # max len in minibatch
        len_mb = max([len(self.data[idx][0]) for idx in indexes])

        for idx in indexes:
            _d = self.data[idx]
            seq = _d[0]
            pair_idx = _d[2]
            # 1 if paired, otherwise 0
            out = [1 if x > 0 else 0 for x in pair_idx]

            if pad:
                # pad to LEN
                assert len(seq) == len(out)
                assert len(seq) <= len_mb
                _diff = len_mb - len(seq)
                # add context/2 of N's on each side
                seq = 'N' * (self.context/2) + seq + 'N' * _diff + 'N' * (self.context/2)
                out.extend([0] * _diff)
                assert len(seq) == len(out) + self.context
            else:
                assert len(seq) == len(out)
                seq = 'N' * (self.context / 2) + seq + 'N' * (self.context / 2)

            # encode
            seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('U', '4').replace('N', '0')
            x = np.asarray(map(int, list(seq)))
            x = self.DNA_ENCODING[x.astype('int8')]
            y = np.asarray(out)
            _data_input.append(x)
            _data_output.append(y[:, np.newaxis])

        _data_input = np.swapaxes(np.swapaxes(np.stack(_data_input, axis=2), 0, 2), 1, 2)
        _data_output = np.swapaxes(np.swapaxes(np.stack(_data_output, axis=2), 0, 2), 1, 2)

        return _data_input, _data_output


class Histories(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class ValidationSetLosses(Callback):
    def __init__(self, datagen_v):
        self.datagen_v = datagen_v
        super(ValidationSetLosses, self).__init__()

    def on_train_begin(self, logs={}):
        self.losses = []
        self.aucs = []

    def on_epoch_end(self, epoch, logs=None):
        losses = []
        aucs = []
        for idx in range(len(self.datagen_v.all_idx)):
            x, y = self.datagen_v.debug_get_data([idx], pad=False)
            yp = self.model.predict(x)
            assert y.shape[0] == 1
            assert yp.shape[0] == 1
            y = y[0, :, 0]
            yp = yp[0, :, 0]
            assert y.shape[0] == yp.shape[0]
            losses.append(log_loss(y, yp))
            aucs.append(roc_auc_score(y, yp))

        losses = pd.Series(losses)
        aucs = pd.Series(aucs)
        print 'cross entropy'
        print losses.describe()
        print 'auc'
        print aucs.describe()
        self.losses.append(losses.median())
        self.aucs.append(aucs.median())
        return


def main(split):
    # load dataset
    with open('data/data.pkl', 'rb') as f:
        data_all = pickle.load(f)

    # config
    residual_conv = [{'num_filter': 32, 'filter_width': 11, 'dilation': 1},
                     {'num_filter': 32, 'filter_width': 11, 'dilation': 2},
                     # {'num_filter': 32, 'filter_width': 11, 'dilation': 4},
                     # {'num_filter': 32, 'filter_width': 11, 'dilation': 8},
                     ]
    context = resolve_contex(residual_conv)
    print context

    # training/validation
    if split == 'random':
        random.shuffle(data_all)
        num_training = int(len(data_all) * 0.8)
        data_training = data_all[:num_training]
        data_validation = data_all[num_training:]
        datagen_t = DataGenerator(data_training, context)
        datagen_v = DataGenerator(data_validation, context)
    else:
        data_dict = split_data_by_rna_type(data_all)
        assert split in data_dict.keys()
        data_validation = data_dict[split]
        data_training = [data_dict[x] for x in data_dict.keys() if x != split]
        data_training = [x for y in data_training for x in y]
        print("validation RNA type: %s" % split)
        print("training RNAs: %d" % len(data_training))
        print("validation RNAs: %d" % len(data_validation))
        datagen_t = DataGenerator(data_training, context)
        datagen_v = DataGenerator(data_validation, context)

    # model
    model = build_model_residual_conv(L=32, residual_conv=residual_conv, nout=1,
                                      residual=False, skipconn=False, gated=False)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0001))

    tictoc = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    log_dir = 'log_' + tictoc
    os.mkdir(log_dir)
    run_dir = 'run_' + tictoc
    os.mkdir(run_dir)

    callbacks = [
        Histories(),
        ValidationSetLosses(datagen_v),
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(os.path.join(run_dir, 'checkpoint.{epoch:03d}.hdf5'),
                        save_best_only=False, period=1),
        CSVLogger(os.path.join(run_dir, 'history.csv')),
    ]

    model.fit_generator(generator=datagen_t,
                        validation_data=datagen_v,
                        validation_steps=25,
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
