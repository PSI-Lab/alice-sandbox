"""
plot model
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
from keras.utils.vis_utils import plot_model


def main(config, fold_idx, out_file):
    model = keras.models.load_model('model/fold_{}.hdf5'.format(fold_idx),
                                    custom_objects={"kb": kb, "custom_loss": custom_loss})
    plot_model(model, to_file=out_file, show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO training script should also store config and custom loss per each trained model
    # since the hyperparam might be different for different fold

    parser.add_argument('--fold', type=int, help='validation fold ID')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--out', type=str, help='path to output file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # FIXME for now there is only one output, hack
    config['target_cols'] = ['data']

    main(config, args.fold, args.out)

