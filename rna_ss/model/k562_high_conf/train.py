import os
import sys
import shutil
import numpy as np
from time import gmtime, strftime
import tensorflow as tf
import keras
import datacorral as dc
import pandas as pd
from scipy.stats import pearsonr
from keras import objectives
import keras.backend as kb
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from genome_kit import Interval
from data_generator import DataGenerator
from dgutils.pandas import read_dataframe, add_column
from model import build_model, resolve_contex, custom_loss
from config import config


# def _load_interval(file):
#     transcript_itvs = []
#
#     with open(file, 'r') as f:
#         for line in f:
#             line = line.rstrip()
#             transcript_itvs.append(eval(line))
#     return transcript_itvs


class Histories(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


class ValidationSetMetrics(Callback):

    def __init__(self, val_data, batch_size=20):
        super(ValidationSetMetrics, self).__init__()
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.corr = []

    def on_epoch_end(self, epoch, logs=None):
        corr = []
        # get self.batch_size batch of validation data
        idx_start = np.random.randint(0, len(self.validation_data) - self.batch_size)

        _corr_data = []
        for batch_idx in range(idx_start, idx_start + self.batch_size):
            _x, _y, _w = self.validation_data[batch_idx]
            _yp = self.model.predict(_x)

            # compute correlation for each example
            for k in range(_y.shape[0]):
                _val = []
                # for each output dimension
                for j in range(_y.shape[2]):
                    y = _y[k, :, j]
                    yp = _yp[k, :, j]

                    # remove missing val
                    idx_valid = np.where(y != -1)  # TODO compute for each dataset
                    y = y[idx_valid]
                    yp = yp[idx_valid]
                    # compute correlation
                    c, p = pearsonr(y, yp)
                    _val.append(c)
                _corr_data.append(_val)
        _corr_data = pd.DataFrame(_corr_data)
        # print("Correlation")
        print(_corr_data.describe(percentiles=[0.75]))
        self.corr.append(_corr_data.median())
        return


def main(validation_fold_idx):
    chrom_folds = config['chrom_folds']
    _, df_intervals = read_dataframe(config['all_inervals'])
    df_intervals = add_column(df_intervals, 'chromosome', ['transcript'], lambda x: x.chromosome)

    # interval_folds = [[] for _ in range(len(chrom_folds))]
    #
    #
    # for itvs in all_intervals:
    #     itv_chrom = itvs[0].chromosome
    #     for fold_id, fold_chroms in enumerate(chrom_folds):
    #         if itv_chrom in fold_chroms:
    #             interval_folds[fold_id].append(itvs)
    #             break
    # print([len(x) for x in interval_folds])
    #
    # training_intervals = [item for idx, sublist in enumerate(interval_folds) for item in sublist if
    #                       idx != validation_fold_idx]
    # validation_intervals = interval_folds[validation_fold_idx]

    df_training_intervals = df_intervals[~df_intervals['chromosome'].isin(chrom_folds[validation_fold_idx])]
    df_validation_intervals = df_intervals[df_intervals['chromosome'].isin(chrom_folds[validation_fold_idx])]
    assert len(df_training_intervals) + len(df_validation_intervals) == len(df_intervals)

    print("Validation fold index: %d" % validation_fold_idx)
    print("Num disjoint intervals in training: %d" % len(df_training_intervals))
    print("Num disjoint intervals in validation: %d" % len(df_validation_intervals))

    training_dataset = DataGenerator(df_training_intervals)
    validation_dataset = DataGenerator(df_validation_intervals)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    kb.tensorflow_backend._get_available_gpus()

    model = build_model(config['n_filters'], config['residual_conv'], config['n_repeat_in_residual_unit'],
                        config['skip_conn_every_n'], config['residual'], config['skipconn'], config['gated'])
    opt = keras.optimizers.Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                amsgrad=False)
    model.compile(loss=custom_loss,
                  optimizer=opt)

    # callbacks
    tictoc = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    log_dir = 'log_' + tictoc
    os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir)

    run_dir = 'run_' + tictoc
    os.mkdir(run_dir)

    es_patience = config['es_patience']
    early_stopping_monitor = EarlyStopping(patience=es_patience, verbose=1)

    callbacks = [
        Histories(),
        ValidationSetMetrics(training_dataset),
        ValidationSetMetrics(validation_dataset),
        tensorboard,
        ReduceLROnPlateau(patience=5, cooldown=2, verbose=1),
        early_stopping_monitor,
        ModelCheckpoint(os.path.join(run_dir, 'checkpoint.{epoch:03d}.hdf5'),
                        save_best_only=False, period=1),
        CSVLogger(os.path.join(run_dir, 'history.csv')),
        Histories(),
    ]

    model.fit_generator(generator=training_dataset,
                        validation_data=validation_dataset,
                        validation_steps=config['num_batch_for_validation'],
                        callbacks=callbacks,
                        shuffle=True,
                        epochs=config['num_epoch'],
                        use_multiprocessing=True,
                        workers=2)

    # copy over the best model
    es_epoch = early_stopping_monitor.stopped_epoch
    best_epoch = es_epoch - es_patience
    model_file_src = os.path.join(run_dir, 'checkpoint.{epoch:03d}.hdf5'.format(epoch=best_epoch))
    model_file_des = os.path.join(config['model_dir'], 'fold_%d.hdf5' % validation_fold_idx)
    shutil.copy(model_file_src, model_file_des)
    print("Model from epoch %d is saved at: %s" % (best_epoch, model_file_des))


if __name__ == "__main__":
    fold_idx = int(sys.argv[1])
    assert 0 <= fold_idx < len(config['chrom_folds'])
    main(fold_idx)
