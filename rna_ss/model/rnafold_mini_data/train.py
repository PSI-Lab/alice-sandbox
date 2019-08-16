import os
# import sys
# import imp
import gzip
import csv
import yaml
import shutil
import argparse
import subprocess
import numpy as np
from time import gmtime, strftime
import tensorflow as tf
import keras
import datacorral as dc
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from keras import objectives
import keras.backend as kb
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from genome_kit import Interval
from data_generator import DataGenerator
from dgutils.pandas import Column, get_metadata, write_dataframe, add_column, read_dataframe
from model import build_model
# from config import config


class Histories(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


def main(config):
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

    # load data TODO hard-coded for now
    df_metadata, df_intervals = read_dataframe('data/rand_seqs_100000.csv')

    n_train = int(len(df_intervals) * 0.8)
    df_training = df_intervals[:n_train]
    df_validation = df_intervals[n_train:]

    print("Num sequences in training: %d" % len(df_training))
    print("Num sequences in validation: %d" % len(df_validation))

    training_dataset = DataGenerator(df_training, config['batch_size'])
    validation_dataset = DataGenerator(df_validation, config['batch_size'])

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    kb.tensorflow_backend._get_available_gpus()

    # model = build_model(config['n_filters'], config['residual_conv'], config['n_repeat_in_residual_unit'],
    #                     config['skip_conn_every_n'], config['residual'], config['skipconn'], config['gated'])

    model = build_model()

    opt = keras.optimizers.Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                amsgrad=False)  # TODO weight decay
    model.compile(loss='binary_crossentropy',optimizer=opt)

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
        # ValidationSetMetrics(training_dataset, os.path.join(run_dir, 'metric_training.csv'),
        #                      batch_size=config['num_batch_for_validation']),
        # ValidationSetMetrics(validation_dataset, os.path.join(run_dir, 'metric_validation.csv'),
        #                      batch_size=config['num_batch_for_validation']),
        tensorboard,
        ReduceLROnPlateau(patience=5, cooldown=2, verbose=1),
        early_stopping_monitor,
        ModelCheckpoint(os.path.join(run_dir, 'checkpoint.{epoch:03d}.hdf5'),
                        save_best_only=False, period=1),
        CSVLogger(os.path.join(run_dir, 'history.csv')),
        Histories(),
    ]

    # dump config
    with open(os.path.join(run_dir, 'config.yml'), 'w') as outfile:
        config['git_hash'] = git_hash
        yaml.dump(config, outfile)

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
    model_file_des = os.path.join(config['model_dir'], 'model.hdf5')
    shutil.copy(model_file_src, model_file_des)
    print("Model from epoch %d is saved at: %s" % (best_epoch, model_file_des))
    # also dump the config in that folder
    with open(os.path.join(config['model_dir'], 'config.yml'), 'w') as outfile:
        config['run_dir'] = run_dir
        config['git_hash'] = git_hash
        yaml.dump(config, outfile)

    # make prediction on validation set
    # TODO restore to the ES checkpoint
    data_pred = []
    for i, row in validation_dataset.df.iterrows():
        x1, x2, y = validation_dataset.get_data([i])
        pred = model.predict([x1, x2])
        row['pred'] = pred[0, :, 0].tolist()
        row['mid_point_pair_prob'] = row['mid_point_pair_prob'][:, 0].tolist()
        data_pred.append(row)
    data_pred = pd.DataFrame(data_pred)
    metadata = get_metadata(
        Column("mid_point_pair_prob", "LIST"),
        Column("pred", "LIST"))
    write_dataframe(metadata, data_pred, 'prediction/validation_data_prediction.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file')
    # parser.add_argument('--fold', type=int, help='validation fold ID')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # fold_idx = int(sys.argv[1])
    # assert 0 <= args.fold < len(config['chrom_folds'])
    main(config)
