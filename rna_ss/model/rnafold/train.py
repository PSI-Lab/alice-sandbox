import os
import itertools
import subprocess
# import sys
# import imp
import gzip
import csv
import yaml
import shutil
import argparse
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
from dgutils.pandas import read_dataframe, add_column
from model import build_model, resolve_contex, custom_loss


class Histories(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


def main(config, validation_fold_idx):
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

    chrom_folds = config['chrom_folds']
    chroms_train = list(itertools.chain.from_iterable([x for i, x in enumerate(chrom_folds) if i != validation_fold_idx]))
    chroms_valid = chrom_folds[validation_fold_idx]

    print("Validation fold index: %d" % validation_fold_idx)
    training_dataset = DataGenerator(chroms_train, config)
    validation_dataset = DataGenerator(chroms_valid, config)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    kb.tensorflow_backend._get_available_gpus()

    model = build_model(config)

    opt = keras.optimizers.Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                amsgrad=False)  # TODO weight decay
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
        tensorboard,
        ReduceLROnPlateau(patience=5, cooldown=2, verbose=1),
        early_stopping_monitor,
        ModelCheckpoint(os.path.join(run_dir, 'checkpoint.{epoch:03d}.hdf5'),
                        save_best_only=False, period=1),
        CSVLogger(os.path.join(run_dir, 'history.csv')),
    ]

    # dump config
    with open(os.path.join(run_dir, 'config.yml'), 'w') as outfile:
        config['validation_fold_idx'] = validation_fold_idx
        config['git_hash'] = git_hash
        yaml.dump(config, outfile)

    model.fit_generator(generator=training_dataset,
                        validation_data=validation_dataset,
                        validation_steps=config['num_batch_for_validation'],
                        callbacks=callbacks,
                        shuffle=True,
                        epochs=config['num_epoch'],
                        use_multiprocessing=True,
                        workers=4)

    # copy over the best model
    es_epoch = early_stopping_monitor.stopped_epoch
    best_epoch = es_epoch - es_patience
    model_file_src = os.path.join(run_dir, 'checkpoint.{epoch:03d}.hdf5'.format(epoch=best_epoch))
    model_file_des = os.path.join(config['model_dir'], 'fold_%d.hdf5' % validation_fold_idx)
    shutil.copy(model_file_src, model_file_des)
    print("Model from epoch %d is saved at: %s" % (best_epoch, model_file_des))
    # also dump the config in that folder
    with open(os.path.join(config['model_dir'], 'config_{}.yml'.format(validation_fold_idx)), 'w') as outfile:
        config['run_dir'] = run_dir
        config['git_hash'] = git_hash
        yaml.dump(config, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--fold', type=int, help='validation fold ID')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # fold_idx = int(sys.argv[1])
    assert 0 <= args.fold < len(config['chrom_folds'])
    main(config, args.fold)
