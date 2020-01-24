import os
# import sys
# import imp
import gzip
import csv
import yaml
import tqdm
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
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from genome_kit import Interval
# from data_generator import DataGeneratorFixedLen
from data_generator import DataGeneratorVarLen
# from data_generator import DataGeneratorInfinite
from dgutils.pandas import Column, get_metadata, write_dataframe, add_column, read_dataframe
from model import build_model, custom_loss, TriangularConvolution2D
# from config import config


class Histories(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


def main(config, data_file, output_dir):
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

    # load data TODO hard-coded for now
    # # random seq
    # df_intervals = pd.read_pickle('data/rand_seqs_fe_200_50000.pkl.gz')
    # # CG dataset
    # df_intervals = pd.read_pickle('data/s_processed.pkl')
    # # var length random seq dataset
    # df_intervals = pd.read_pickle('data/rand_seqs_var_len_10_100_100000.pkl.gz')
    df_intervals = pd.read_pickle(data_file)

    n_train = int(len(df_intervals) * 0.8)
    df_training = df_intervals[:n_train].reset_index(drop=True)
    df_validation = df_intervals[n_train:].reset_index(drop=True)

    print("Num sequences in training: %d" % len(df_training))
    print("Num sequences in validation: %d" % len(df_validation))

    # training_dataset = DataGeneratorFixedLen(df_training, config['batch_size'])
    # validation_dataset = DataGeneratorFixedLen(df_validation, config['batch_size'])
    # for training we group sequence of similar lengths in same minibatch to speed up
    # for validation we randomize the minibatch to make sure we have a good sampling of all lengths
    training_dataset = DataGeneratorVarLen(df_training, config['batch_size'], length_grouping=True)
    # training_dataset = DataGeneratorVarLen(df_training, config['batch_size'], length_grouping=False)
    validation_dataset = DataGeneratorVarLen(df_validation, config['batch_size'], length_grouping=False)

    # # debug
    # print("debug")
    # training_dataset = DataGeneratorInfinite(batch_size=config['batch_size'], num_batches=5,
    #                                          min_len=20, max_len=200, num_structures=10, name='training')
    #
    # # training_dataset = DataGeneratorInfinite(batch_size=config['batch_size'], num_batches=500,
    # #                                          min_len=20, max_len=200, num_structures=10, name='training')
    # validation_dataset = DataGeneratorInfinite(batch_size=config['batch_size'],
    #                                            num_batches=config['num_batch_for_validation'],
    #                                            min_len=20, max_len=200, num_structures=10, name='validation')

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    kb.tensorflow_backend._get_available_gpus()

    # model = build_model(config['n_filters'], config['residual_conv'], config['n_repeat_in_residual_unit'],
    #                     config['skip_conn_every_n'], config['residual'], config['skipconn'], config['gated'])

    model = build_model()

    opt = keras.optimizers.Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                amsgrad=False)  # TODO weight decay
    # model.compile(loss='binary_crossentropy',optimizer=opt)
    # model.compile(loss=custom_loss,
    #               optimizer=opt)
    # model.compile(loss={'ar_label': custom_loss, 'fe': 'mean_squared_error', 'base_pair_prob_output': custom_loss},
    #               loss_weights={'ar_label': 1.0, 'fe': 0.0, 'base_pair_prob_output': 1.0},  # FIXME turn off fe loss for now
    #               optimizer=opt)

    # FIXME debug: just to check whether custom_loss is causing the mismatched shape!
    model.compile(loss={'ar_label': custom_loss, 'fe': 'mean_squared_error', 'base_pair_prob_output': 'binary_crossentropy'},
                  loss_weights={'ar_label': 1.0, 'fe': 0.0, 'base_pair_prob_output': 1.0},  # FIXME turn off fe loss for now
                  optimizer=opt)


    # TODO loss weighting

    # debug
    print(model.summary())

    # callbacks
    tictoc = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    log_dir = 'log_' + tictoc
    os.mkdir(log_dir)
    # tensorboard = TensorBoard(log_dir=log_dir)

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
        # tensorboard,
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

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # also dump the config with run_dir in output folder
    with open(os.path.join(output_dir, 'config.yml'), 'w') as outfile:
        config['run_dir'] = run_dir
        config['git_hash'] = git_hash
        yaml.dump(config, outfile)

    print("run_dir: {}".format(run_dir))
    model.fit_generator(generator=training_dataset,
                        validation_data=validation_dataset,
                        validation_steps=config['num_batch_for_validation'],
                        callbacks=callbacks,
                        shuffle=True,
                        epochs=config['num_epoch'],
                        use_multiprocessing=True,
                        workers=8)

    print("run_dir: {}".format(run_dir))
    # copy over the best model
    es_epoch = early_stopping_monitor.stopped_epoch
    best_epoch = es_epoch - es_patience + 1
    print("ES epoch: {}, best epoch: {}".format(es_epoch, best_epoch))
    model_file_src = os.path.join(run_dir, 'checkpoint.{epoch:03d}.hdf5'.format(epoch=best_epoch))

    model_file_des = os.path.join(output_dir, 'model.hdf5')

    shutil.copy(model_file_src, model_file_des)
    print("Model from epoch %d is saved at: %s" % (best_epoch, model_file_des))
    # # also dump the config in that folder
    # with open(os.path.join(output_dir, 'config.yml'), 'w') as outfile:
    #     config['run_dir'] = run_dir
    #     config['git_hash'] = git_hash
    #     yaml.dump(config, outfile)

    # # make prediction on validation set
    # # restore to the ES checkpoint
    # print("Restoring model from epoch {}".format(best_epoch))
    # model = load_model(model_file_des, custom_objects={'kb': kb, 'tf': tf,
    #                                                    'custom_loss': custom_loss,
    #                                                    'TriangularConvolution2D': TriangularConvolution2D})
    # data_pred = []
    # print("Making predictions on validation data...")
    # for i, row in tqdm.tqdm(validation_dataset.df.iterrows(), total=len(validation_dataset.df)):
    #     x1, y = validation_dataset.get_data([i], len(row['seq']))
    #     pred, fe = model.predict(x1)
    #     row['pred'] = pred[0, :, :, 0]
    #     row['pfe'] = fe[0]
    #     data_pred.append(row)
    # data_pred = pd.DataFrame(data_pred)
    # if not os.path.isdir('prediction'):
    #     os.mkdir('prediction')
    # pd.to_pickle(data_pred, 'prediction/validation_data_prediction.pkl.gz', compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--data', type=str, help='training dataset')
    parser.add_argument('--output', type=str, help='output dir to save the best model')
    # parser.add_argument('--fold', type=int, help='validation fold ID')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # fold_idx = int(sys.argv[1])
    # assert 0 <= args.fold < len(config['chrom_folds'])
    main(config, args.data, args.output)
    # main(config, args.output)
