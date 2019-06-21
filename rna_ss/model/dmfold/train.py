import argparse
import numpy as np
import cPickle as pickle
import pandas as pd
import random
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, Masking, Bidirectional, Dense, Activation, TimeDistributed
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import log_loss, accuracy_score
from utils import split_data_by_rna_type


class DataGenerator(keras.utils.Sequence):
    # N will be all 0's - for gradient masking
    DNA_ENCODING = np.zeros((5, 4))
    DNA_ENCODING[1:, :] = np.eye(4)
    # 0 will be all 0's - for gradient masking
    OUTPUT_ENCODING = np.zeros((8, 7))
    OUTPUT_ENCODING[1:, :] = np.eye(7)

    LEN = 500  # this is > max length of sequence in dataset

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

    def debug_get_data(self, indexes, pad=True):
        return self.__data_generation(indexes, pad)

    def __data_generation(self, indexes, pad=True):
        _data_input = []
        _data_output = []

        for idx in indexes:
            _d = self.data[idx]

            seq = _d[0]
            out = _d[1]

            if pad:
                # pad to LEN
                assert len(seq) == len(out), "sequence len %d output len %d" % (len(seq), len(out))
                assert len(seq) < self.LEN, "sequence %d shorter than %d" % (len(seq), self.LEN)
                _diff = self.LEN - len(seq)
                seq = seq + 'N' * _diff
                out = out + [0] * _diff
                assert len(seq) == len(out), "after padding, sequence len %d output len %d" % (len(seq), len(out))
            else:
                assert len(seq) == len(out), "sequence len %d output len %d" % (len(seq), len(out))

            # encode
            seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('U', '4').replace('N', '0')
            x = np.asarray(map(int, list(seq)))
            x = self.DNA_ENCODING[x.astype('int8')]
            y = np.asarray(out)
            y = self.OUTPUT_ENCODING[y.astype('int8')]
            _data_input.append(x)
            _data_output.append(y)

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
        self.accuracies = []
        _ = self._get_accuracy()

    def on_epoch_end(self, epoch, logs=None):
        self.accuracies.append(self._get_accuracy())
        return

    def _get_accuracy(self):
        accuracies = []
        for idx in range(len(self.datagen_v.all_idx)):
            x, y = self.datagen_v.debug_get_data([idx], pad=False)  # no padding, since we're generating one example and need to computing accuracy
            yp = self.model.predict(x)
            assert y.shape[0] == 1, y.shape
            assert yp.shape[0] == 1, yp.shape
            y = y[0, :, :]
            yp = yp[0, :, :]
            assert y.shape[1] == yp.shape[1]
            accuracies.append(accuracy_score(np.argmax(y, axis=1), np.argmax(yp, axis=1)))
        accuracies = pd.Series(accuracies)
        print 'accuracy'
        print accuracies.describe()
        return accuracies.median()


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

    # mask with 0
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(None, 4)))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(7, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    callbacks = [
        Histories(),
        EarlyStopping(monitor='val_loss', patience=5),
        ValidationSetLosses(datagen_v),
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
