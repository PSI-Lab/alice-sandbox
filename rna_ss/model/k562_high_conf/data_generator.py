import numpy as np
import keras
import datacorral as dc
from genome_kit import Genome, GenomeTrack, Interval
from dgutils.interval import DisjointIntervalsSequence
from dgutils.pandas import add_column
from model import resolve_contex
from config import config


class DataGenerator(keras.utils.Sequence):

    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    def __init__(self, df_intervals, gtrack=config['gtrack'],
                 train_length=config['example_length'],
                 context=resolve_contex(config['dense_conv']),
                 batch_size=config['batch_size'], genome=config['genome_annotation'],
                 min_log_tpm=config['min_log_tpm'], min_rep_corr=config['min_rep_corr'],
                 example_reweighting=config['example_reweighting']):
        self.genome = Genome(genome)
        self.train_length = train_length
        self.context = context
        self.batch_size = batch_size
        self.gtrack = GenomeTrack(gtrack)
        self.diseqs, self.intervals = self._process_transcript_intervals(df_intervals, min_log_tpm, min_rep_corr,
                                                                         example_reweighting)
        # shuffle indexes before generating data
        self.on_epoch_end()

    def _example_weight(self, log_tpm, rep_corr):
        if np.isnan(log_tpm) or np.isinf(log_tpm) or np.isnan(rep_corr) or np.isinf(rep_corr):
            return 0.01  # arbitrary small number
        else:
            assert 0 <= rep_corr <= 1
            # tpm weight
            w1 = 1/(1 + np.exp(-1.5 * (log_tpm - 3)))
            # corr weight
            w2 = 1/(1 + np.exp(-10 * (rep_corr - 0.5)))
            return w1 * w2

    def _process_transcript_intervals(self, df_intervals, min_log_tpm, min_rep_corr, example_reweighting):
        """process transcript intervals to fixed length shorter intervals for training"""
        diseqs = []
        all_intervals = []  # list of tuples: diseq_interval, diseq_idx, example_weight

        n_skipped = 0

        # process df
        df_intervals = add_column(df_intervals, 'log_tpm', ['tpm'], np.log)

        for _, row in df_intervals.iterrows():
            itvs = row['disjoint_intervals']

            # filter
            # nan should be skipped
            if not (row['log_tpm'] >= min_log_tpm):
                print("Skip {} due to low log TPM {}".format(row['transcript_id'], row['log_tpm']))
                continue
            if not (row['pearson_corr'] >= min_rep_corr):
                print("Skip {} due to low rep corr {}".format(row['transcript_id'], row['pearson_corr']))
                continue

            if example_reweighting:
                # calculate example weight for all intervals within this transcript
                transcript_weight = self._example_weight(row['log_tpm'], row['pearson_corr'])
            else:
                transcript_weight = 1.0

            diseq = DisjointIntervalsSequence(itvs, self.genome)
            diseqs.append(diseq)
            diseq_idx = len(diseqs) - 1

            itv = diseq.interval.end5.expand(0, self.train_length)
            while itv.within(diseq.interval):
                # discard interval if all positions are missing output
                # or all positions == 0
                # or all positions == 1
                _y = self._get_y(itv, diseq)
                if np.all(_y == -1) or np.all(_y == 0) or np.all(_y == 1):
                    # print("Skip %s in diseq %s" % (itv, diseq.intervals))
                    n_skipped += 1
                else:
                    all_intervals.append((itv, diseq_idx, transcript_weight))
                itv = itv.shift(self.train_length)

                if n_skipped % 100 == 0 or len(all_intervals) % 1000 == 0:
                    print("Intervals: %d, skipped: %d" % (len(all_intervals), n_skipped))

        return diseqs, all_intervals

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.intervals) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        x, y, w = self.__data_generation(indexes)
        # note that this need to match the model, see model.py
        return x, y, w

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.intervals))
        np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""
        x, y, w = self.get_data([self.intervals[i] for i in indexes])

        return x, y, w

    def _get_x(self, itv, diseq):
        # middle part
        seq_mid = diseq.dna(itv)
        # upstream
        _ll = len(Interval.spanning(diseq.interval.end5, itv.end5))
        if _ll < self.context / 2:
            seq_up = 'N' * (self.context / 2 - _ll) + diseq.dna(itv.end5.expand(_ll, 0))
        else:
            seq_up = diseq.dna(itv.end5.expand(self.context / 2, 0))
        # downstream
        _lr = len(Interval.spanning(diseq.interval.end3, itv.end3))
        if _lr < self.context / 2:
            seq_dn = diseq.dna(itv.end3.expand(0, _lr)) + 'N' * (self.context / 2 - _lr)
        else:
            seq_dn = diseq.dna(itv.end3.expand(0, self.context / 2))
        seq = seq_up + seq_mid + seq_dn
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def _get_y(self, itv, diseq):
        # lower to genomic interval and get data
        genomic_itvs = diseq.lower_interval(itv)
        # last 3 channels, K562 dataset
        y = np.concatenate([self.gtrack(_i)[:, -3:] for _i in genomic_itvs], axis=0)
        return y

    def get_data_single(self, itv, diseq_idx, transcript_weight):
        diseq = self.diseqs[diseq_idx]
        # x
        x = self._get_x(itv, diseq)
        # y
        y = self._get_y(itv, diseq)
        return x, y, transcript_weight

    def get_data(self, intervals):
        x = []
        y = []
        w = []
        for itv, diseq_idx, transcript_weight in intervals:
            _x, _y, _w = self.get_data_single(itv, diseq_idx, transcript_weight)
            x.append(_x)
            y.append(_y)
            w.append(_w)
        x = np.asarray(x)
        y = np.asarray(y)
        w = np.asarray(w)
        return x, y, w
