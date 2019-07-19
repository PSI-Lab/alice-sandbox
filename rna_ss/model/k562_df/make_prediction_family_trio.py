"""
Make prediction on family trio dataset
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
from genome_kit import Interval, Genome, Variant, VariantGenome
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


def _find_fold(chrom_to_pred, chrom_folds):
    for i, chroms in enumerate(chrom_folds):
        if chrom_to_pred in chroms:
            return i
    return None


def _predict_seq(seq, fold_idx, predictors, data_names):
    if not np.isnan(fold_idx):
        yp = predictors[int(fold_idx)].predict_seq(seq)[0, :, :]
    else:
        print("Processing {} using all models".format(seq))
        yps = []
        for _idx in range(len(config['chrom_folds'])):
            yps.append(predictors[_idx].predict_seq(seq))
        yp = np.mean(np.concatenate(yps, axis=0), axis=0)  # TODO using mean for now
        assert len(yp.shape) == 2
    assert yp.shape[1] == len(data_names)
    return yp


def predict_row_data(variant, transcript, fold_idx, genome, varg, predictors, data_names, flank=50):
    diseq_wt = DisjointIntervalsSequence(transcript.exons, genome)
    diseq_mt = DisjointIntervalsSequence(transcript.exons, varg)
    variant_lifted = diseq_wt.lift_interval(variant)

    seq_wt = diseq_wt.dna(variant_lifted.expand(flank))
    seq_mt = diseq_mt.dna(variant_lifted.expand(flank))

    yp_wt = _predict_seq(seq_wt, fold_idx, predictors, data_names)
    yp_mt = _predict_seq(seq_mt, fold_idx, predictors, data_names)

    # to list
    return tuple([seq_wt, seq_mt] + [np.around(yp_wt[:, i], 4).tolist() for i in range(yp_wt.shape[1])] + [
        np.around(yp_mt[:, i], 4).tolist() for i in range(yp_mt.shape[1])])


def main(config):
    df = pd.read_csv(dc.Client().get_path('aoLEcQ'))
    genome = Genome(config['genome_annotation'])
    df = add_column(df, 'transcript', ['transcript_id'], lambda tx_id: genome.transcripts[tx_id])
    df = add_column(df, 'chrom', ['transcript'], lambda x: x.chromosome)
    df = add_column(df, 'variant', ['variant'], lambda x: Variant.from_string(x, genome), pbar=False)
    df = add_column(df, 'varg', ['variant'], lambda x: VariantGenome(genome, x), pbar=False)

    # # add column to indicate if it's A/C nucleotide
    # df = add_column(df, 'is_ac', ['variant'], lambda x: x.ref in ['A', 'C'] or x.alt in ['A', 'C'])

    # find the model to use for making CV prediction
    df = add_column(df, 'fold_idx', ['chrom'], lambda x: _find_fold(x, config['chrom_folds']))

    context = resolve_contex(config['dense_conv'])
    print("Context: %d" % context)
    predictors = [Predictor('model/fold_{}.hdf5'.format(fold_idx), context)
                  for fold_idx in range(len(config['chrom_folds']))]

    # prediction
    df = add_columns(df, ['seq_wt', 'seq_mt'] + ['{}_pred_wt'.format(x) for x in config['target_cols']] + [
        '{}_pred_mt'.format(x) for x in config['target_cols']], ['variant', 'transcript', 'fold_idx', 'varg'],
                     lambda v, t, i, g: predict_row_data(v, t, i, genome, g, predictors, config['target_cols'], 50))

    # drop columns
    df = df.drop(columns=['varg'])

    # add metadata, output
    metadata = dataframe.Metadata()
    metadata.version = "1"
    for x in config['target_cols']:
        metadata.encoding['{}_pred_wt'.format(x)] = dataframe.Metadata.LIST
        metadata.encoding['{}_pred_mt'.format(x)] = dataframe.Metadata.LIST
    metadata.encoding["transcript"] = dataframe.Metadata.GENOMEKIT
    metadata.encoding["variant"] = dataframe.Metadata.GENOMEKIT
    write_dataframe(metadata, df, 'prediction/family_trio.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO training script should also store config and custom loss per each trained model
    # since the hyperparam might be different for different fold

    parser.add_argument('--config', type=str, help='path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    main(config)
