"""
Make CV prediction on training dataset
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
from dgutils.pandas import read_dataframe, add_column, write_dataframe
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


def _find_fold(chrom_to_pred, chrom_folds):
    for i, chroms in enumerate(chrom_folds):
        if chrom_to_pred in chroms:
            return i
    return None


def predict_row_data(seq, fold_idx, predictors, gene_name, transcript_id):
    if not np.isnan(fold_idx):
        yp = predictors[int(fold_idx)].predict_seq(seq)[0, :, :]
    else:
        print("Processing {} {} using all models".format(gene_name, transcript_id))
        yps = []
        for _idx in range(len(config['chrom_folds'])):
            yps.append(predictors[_idx].predict_seq(seq))
        yp = np.mean(np.concatenate(yps, axis=0), axis=0)  # TODO using mean for now
        assert len(yp.shape) == 2
    return yp.tolist()  # list for df output


def _add_sequence(itvs, genome):
    diseq = DisjointIntervalsSequence(itvs, genome)
    return diseq.dna(diseq.interval)


def main(config):
    metadata, _df_intervals = read_dataframe(gzip.open(dc.Client().get_path(config['dataset_dc_id'])))
    _df_intervals = add_column(_df_intervals, 'chrom', ['transcript'], lambda x: x.chromosome)
    _df_intervals = add_column(_df_intervals, 'gene_name', ['transcript'], lambda x: x.gene.name)

    # use the data_generator only to process the df
    genome = Genome(config['genome_annotation'])
    df = add_column(_df_intervals, 'sequence', ['disjoint_intervals'], lambda x: _add_sequence(x, genome))
    df = add_column(df, 'log_tpm', ['tpm'], np.log)

    # find the model to use for making CV prediction
    df = add_column(df, 'fold_idx', ['chrom'], lambda x: _find_fold(x, config['chrom_folds']))

    context = resolve_contex(config['dense_conv'])
    print("Context: %d" % context)

    predictors = [Predictor('model/fold_{}.hdf5'.format(fold_idx), context)
                  for fold_idx in range(len(config['chrom_folds']))]

    # prediction
    df = add_column(df, 'pred', ['sequence', 'fold_idx', 'gene_name', 'transcript_id'],
                    lambda s, i, g, t: predict_row_data(s, i, predictors, g, t))

    # add new metadata, output
    metadata.encoding['pred'] = dataframe.Metadata.LIST
    write_dataframe(metadata, df, 'prediction/training_data_cv.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO training script should also store config and custom loss per each trained model
    # since the hyperparam might be different for different fold

    parser.add_argument('--config', type=str, help='path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    main(config)


