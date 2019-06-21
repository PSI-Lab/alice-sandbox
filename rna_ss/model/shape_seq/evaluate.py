import os
import numpy as np
import keras
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from keras import backend as kb
from genome_kit import Interval, Genome
import rnafoldr
from dgutils.interval import DisjointIntervalsSequence
from data_generator import DataGenerator
from model import custom_loss, resolve_contex
from train import _load_interval
from config import config
import matplotlib
matplotlib.use('Agg')   # do not remove, this is to turn off X server so plot works on Linux
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
plt.ioff()
from scipy import stats
from sklearn import metrics
import cufflinks as cf
cf.go_offline()
cf.set_config_file(theme='ggplot')
import plotly


class Predictor(object):
    DNA_ENCODING = np.zeros((5, 4))
    DNA_ENCODING[1:, :] = np.eye(4)

    def __init__(self, model_file, context):
        self.model = keras.models.load_model(model_file, custom_objects={"kb": kb, "custom_loss": custom_loss})
        self.context = context

    def predict_seq(self, seq):
        # encode input
        seq = 'N' * (self.context / 2) + seq + 'N' * (self.context / 2)
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('U', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = self.DNA_ENCODING[x.astype('int8')]
        x = x[np.newaxis, :, :]
        yp = self.model.predict(x)
        return yp


context = resolve_contex(config['residual_conv'], config['n_repeat_in_residual_unit'])
print("Context: %d" % context)
num_folds = len(config['interval_folds'])
model_dir = config['model_dir']

plot_dir = 'plot'

genome = Genome(config['genome_annotation'])

predictor_folds = [Predictor(os.path.join(model_dir, 'fold_%d.hdf5' % idx),
                             context=400) for idx in range(num_folds)]
interval_folds = [_load_interval(x) for x in config['interval_folds']]
validation_dataset_folds = [
    DataGenerator(validation_intervals, gtrack=config['gtrack'], train_length=config['example_length'], context=context,
                  batch_size=config['batch_size'], genome=config['genome_annotation']) for validation_intervals in
    interval_folds]

# validation set performance
data = []
for fold_idx in range(num_folds):
    for k in range(len(validation_dataset_folds[fold_idx])):
        x, y = validation_dataset_folds[fold_idx][k]
        yp = predictor_folds[fold_idx].model.predict(x)

        for example_idx in range(y.shape[0]):
            df_plot = pd.DataFrame({'target': y[example_idx, :], 'pred': yp[example_idx, :]})
            # replace -1 with NaN in target
            df_plot = df_plot.replace(-1, np.nan)
            df_plot = df_plot.dropna()

            data.append({
                'validation_fold': fold_idx,
                'batch': k,
                'example': example_idx,
                'pearson': pearsonr(df_plot['target'], df_plot['pred'])[0],
                'spearman': spearmanr(df_plot['target'], df_plot['pred'])[0],
            })
df = pd.DataFrame(data)
print("Cross validation performance:")
print df[['pearson', 'spearman']].describe()

fig = df[['pearson', 'spearman']].iplot(kind='histogram', bins=100, title='Cross validation performance',
                                        xTitle='value', yTitle='count', asFigure=True)
plotly.offline.plot(fig, filename=os.path.join(plot_dir, 'cross_validation_performance.html'),
                    auto_open=False)

# selected transcripts from Omar
transcript_ids = ['NM_007355', 'NM_002819', 'NM_005762', 'NM_000291',
                  'NM_000700', 'NM_006230', 'NM_004104', 'NM_001605',
                  'NM_002047', 'NM_015629']

# for each transcript, find the model that wasn't trained on that transcript
transcript_cv_model_idx = dict()
for tx_id in transcript_ids:
    transcript = genome.transcripts[tx_id]

    validation_fold_flag = []
    for fold_idx in range(num_folds):
        is_in_validation = False
        for itvs in interval_folds[fold_idx]:
            for itv in itvs:
                if itv.overlaps(transcript):
                    is_in_validation = True
        validation_fold_flag.append(is_in_validation)

    # some transcript might be out of the training/validation set (i.s. not in the dataset)
    # drop those for now, since there is no 'ground truth' we can compare to
    if sum(validation_fold_flag) == 1:
        transcript_cv_model_idx[tx_id] = validation_fold_flag.index(True)
    elif sum(validation_fold_flag) == 0:
        print("Skip transcript %s since it's all missing value in gtrack" % tx_id)
    else:
        raise ValueError, validation_fold_flag


for FLANK in [50, 100, 150, 200, 250]:

    df_tx_exon = []

    for tx_id, cv_fold_idx in transcript_cv_model_idx.iteritems():

        transcript = genome.transcripts[tx_id]

        for exon in transcript.exons[1:-1]:
            # diseq
            diseq = DisjointIntervalsSequence(transcript.exons, genome)

            # skip exon if there is no enough flank on either side
            exon_lifted = diseq.lift_interval(exon)
            if len(Interval.spanning(diseq.interval.end5, exon_lifted.end5)) < FLANK or len(Interval.spanning(diseq.interval.end3, exon_lifted.end3)) < FLANK:
                print('Skip exon %s due to insufficient context (required: %d)' % (exon, FLANK))
                continue

            # diseq interval
            itv = exon_lifted.expand(FLANK)

            # get sequence
            seq = diseq.dna(itv)

            # get data from gtrack
            genomic_itvs = diseq.lower_interval(itv)
            y = np.concatenate([validation_dataset_folds[cv_fold_idx].gtrack(x) for x in genomic_itvs])[:, 0][FLANK:-FLANK]

            # prediction
            yp = predictor_folds[cv_fold_idx].predict_seq(seq.replace('T', 'U'))[0, :][FLANK:-FLANK]

            # RNAfoldr
            rnafold_scores = rnafoldr.fold_unpair(seq.replace('T', 'U'), winsize=len(seq), span=len(seq))[FLANK:-FLANK]

            # plot
            df_plot = pd.DataFrame({'target': y, 'pred':yp, 'rnafoldr': rnafold_scores})
            # replace -1 with NaN in target
            df_plot = df_plot.replace(-1, np.nan)
            df_plot = df_plot.dropna()
            corr_p_model = pearsonr(df_plot['target'], df_plot['pred'])
            corr_p_rnafoldr = pearsonr(df_plot['target'], df_plot['rnafoldr'])
            corr_s_model = spearmanr(df_plot['target'], df_plot['pred'])
            corr_s_rnafoldr = spearmanr(df_plot['target'], df_plot['rnafoldr'])

            df_tx_exon.append({'transcript': tx_id, 'exon': exon.index + 1, 'exon_interval': exon.interval,
                               'pearson_corr_model': corr_p_model[0], 'pearson_pval_model': corr_p_model[1],
                               'pearson_corr_rnafoldr': corr_p_rnafoldr[0], 'pearson_pval_rnafoldr': corr_p_rnafoldr[1],
                               'spearman_corr_model': corr_s_model.correlation, 'spearman_pval_model': corr_s_model.pvalue,
                               'spearman_corr_rnafoldr': corr_s_rnafoldr.correlation,
                               'spearman_pval_rnafoldr': corr_s_rnafoldr.pvalue})

    df_tx_exon = pd.DataFrame(df_tx_exon)
    print("Performance on transcript selected by Omar:")
    print df_tx_exon[
        ['pearson_corr_model', 'pearson_corr_rnafoldr', 'spearman_corr_model', 'spearman_corr_rnafoldr']].describe()

    fig = df_tx_exon[
        ['pearson_corr_model', 'pearson_corr_rnafoldr', 'spearman_corr_model', 'spearman_corr_rnafoldr']].iplot(
        kind='histogram', bins=100, title='Performance on transcripts: %s' % ','.join(transcript_cv_model_idx.keys()),
        xTitle='value', yTitle='count', asFigure=True)
    plotly.offline.plot(fig, filename=os.path.join(plot_dir, 'compare_rnafold_flank_%d.html' % FLANK),
                        auto_open=False)
