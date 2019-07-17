import os
import yaml
import numpy as np
import tensorflow as tf
import keras
import keras.backend as kb
from scipy.stats import pearsonr, spearmanr
from genome_kit import Interval, Genome, GenomeTrack, Variant, VariantGenome, GenomeTrackBuilder
import rnafoldr
from dgutils.interval import DisjointIntervalsSequence
from dgutils.pandas import add_column, add_columns, read_dataframe
from model import resolve_contex, custom_loss

with open('config.yml', 'r') as f:
    config = yaml.load(f)


def get_diseq_data(diseq, gtrack):
    return np.concatenate([gtrack(_i) for _i in diseq], axis=0)


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


# hard-coded for now
pred_track = GenomeTrackBuilder('pred/pred_track.gtrack', dim=3, etype='f16')
pred_track.set_default_value(-1)

# combined_gtrack = GenomeTrack('data/reactivity_combined.gtrack')

_, df_intervals = read_dataframe('data/intervals_combined.csv')


df_intervals = add_column(df_intervals, 'chrom', ['transcript'], lambda x: x.chromosome)
df_intervals = add_column(df_intervals, 'gene_name', ['transcript'], lambda x: x.gene.name)


# decide which model to use for each transcript

def _find_fold(chrom_to_pred, chrom_folds):
    for i, chroms in enumerate(chrom_folds):
        if chrom_to_pred in chroms:
            return i
    return None


df_intervals = add_column(df_intervals, 'fold_idx', ['chrom'], lambda x: _find_fold(x, config['chrom_folds']))
# # drop those without fold_idx
# print("Dropping {} rows without fold idx (use ensemble in the future)".format(df_intervals['fold_idx'].isna().sum()))
# df_intervals = df_intervals.dropna(subset=['fold_idx'])
# df_intervals['fold_idx'] = df_intervals['fold_idx'].astype(int)


context = resolve_contex(config['dense_conv'])
print("Context: %d" % context)

genome = Genome(config['genome_annotation'])

predictors = [Predictor('model/fold_{}.hdf5'.format(fold_idx), context)
              for fold_idx in range(len(config['chrom_folds']))]

# generate data
for _, row in df_intervals.iterrows():
    # print("{} {}".format(row['gene_name'], row['transcript_id']))

    itvs = row['disjoint_intervals']
    fold_idx = row['fold_idx']
    diseq = DisjointIntervalsSequence(itvs, genome)
    seq = diseq.dna(diseq.interval)
    if not np.isnan(fold_idx):
        yp = predictors[int(fold_idx)].predict_seq(seq)[0, :, :]
    else:
        print("Processing {} {} using all models".format(row['gene_name'], row['transcript_id']))
        yps = []
        for _idx in range(len(config['chrom_folds'])):
            yps.append(predictors[_idx].predict_seq(seq))
        yp = np.mean(np.concatenate(yps, axis=0), axis=0)
        assert len(yp.shape) == 2

    # set data for each sub interval
    for itv in diseq.intervals:
        itv_lifted = diseq.lift_interval(itv)
        data_start = len(Interval.spanning(diseq.interval.end5, itv_lifted.end5))
        data_end = data_start + len(itv)
        yp_slice = yp[data_start:data_end, :]

        # TODO there are still overlapping blocks?!
        try:
            pred_track.set_data(itv, yp_slice)
        except ValueError as e:
            print(e)
            continue

pred_track.finalize()


