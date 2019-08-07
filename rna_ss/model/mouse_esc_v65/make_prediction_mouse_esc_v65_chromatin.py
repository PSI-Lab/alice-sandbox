"""
Make prediction on multi-compartment dataset, mouse ESC v6.5, chromatin
"""
import os
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
from genome_kit import Interval, Genome
import deepgenomics.pandas.v1 as dataframe
from data_generator import DataGenerator
from dgutils.interval import DisjointIntervalsSequence
from dgutils.pandas import read_dataframe, add_column, write_dataframe, add_columns
from model import build_model, resolve_contex, custom_loss


class Predictor(object):
    DNA_ENCODING = np.zeros((5, 4))
    DNA_ENCODING[1:, :] = np.eye(4)

    def __init__(self, model_file, context):
        self.model = keras.models.load_model(model_file, custom_objects={"kb": kb, "custom_loss": custom_loss})
        self.context = context

    def predict_seq(self, seq):
        # encode input
        seq = 'N' * (self.context / 2) + seq + 'N' * (self.context / 2)
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('U', '4').replace('T', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = self.DNA_ENCODING[x.astype('int8')]
        x = x[np.newaxis, :, :]
        yp = self.model.predict(x)
        return yp


def predict_row_data(seq, predictors, data_names):
    yps = []
    for predictor in predictors:
        yps.append(predictor.predict_seq(seq))
    yp = np.mean(np.concatenate(yps, axis=0), axis=0)  # TODO using mean for now
    assert len(yp.shape) == 2

    assert yp.shape[1] == len(data_names)

    # adjust precision for output
    # 1D output for each data prediction, for output (since we don't support list of lists)
    return tuple([np.around(yp[:, i], 4).tolist() for i in range(yp.shape[1])])


def main(config):

    # TODO hard-coded for now
    metadata, df = read_dataframe(gzip.open('data/mouse_esc_v65_icshape_ch_vivo.csv.gz'))

    context = resolve_contex(config['dense_conv'])
    print("Context: %d" % context)
    predictors = [Predictor('model/fold_{}.hdf5'.format(fold_idx), context)
                  for fold_idx in range(len(config['chrom_folds']))]

    # prediction
    df = add_columns(df, ['{}_pred'.format(x) for x in config['target_cols']],
                     ['sequence'],
                     lambda s: predict_row_data(s, predictors, config['target_cols']))

    # add new metadata, output
    for x in config['target_cols']:
        metadata.encoding['{}_pred'.format(x)] = dataframe.Metadata.LIST
    write_dataframe(metadata, df, 'prediction/mouse_esc_v65_icshape_ch_vivo_prediction.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO training script should also store config and custom loss per each trained model
    # since the hyperparam might be different for different fold

    parser.add_argument('--config', type=str, help='path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # tmp hack
    config['target_cols'] = ['data']

    main(config)

