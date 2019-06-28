import numpy as np
import keras
import datacorral as dc
from genome_kit import Genome, GenomeTrack, Interval
from dgutils.interval import DisjointIntervalsSequence
from model import resolve_contex
from config import config


class DataGenerator(keras.utils.Sequence):

    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    # def __init__(self, transcript_intervals, gtrack=config['dc_gtrack'],
    #              train_length=config['example_length'],
    #              context=resolve_contex(config['residual_conv'], config['n_repeat_in_residual_unit']),
    #              batch_size=config['batch_size'], genome=config['genome_annotation']):
    #     self.genome = Genome(genome)
    #     self.train_length = train_length
    #     self.context = context
    #     self.batch_size = batch_size
    #     self.gtrack = GenomeTrack(dc.Client().get_path(gtrack))
    #     self.diseqs, self.intervals = self._process_transcript_intervals(transcript_intervals)
    #     # shuffle indexes before generating data
    #     self.on_epoch_end()

    def __init__(self, transcript_intervals, gtrack=config['gtrack'],
                 train_length=config['example_length'],
                 context=resolve_contex(config['residual_conv'], config['n_repeat_in_residual_unit']),
                 batch_size=config['batch_size'], genome=config['genome_annotation']):
        self.genome = Genome(genome)
        self.train_length = train_length
        self.context = context
        self.batch_size = batch_size
        self.gtrack = GenomeTrack(gtrack)
        self.diseqs, self.intervals = self._process_transcript_intervals(transcript_intervals)
        # shuffle indexes before generating data
        self.on_epoch_end()

    def _process_transcript_intervals(self, transcript_intervals):
        """process transcript intervals to fixed length shorter intervals for training"""
        diseqs = []
        all_intervals = []  # list of tuples: diseq_interval, diseq_idx

        n_skipped = 0

        for itvs in transcript_intervals:
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
                    all_intervals.append((itv, diseq_idx))
                itv = itv.shift(self.train_length)

                if n_skipped % 10 == 0 or len(all_intervals) % 1000 == 0:
                    print("Intervals: %d, skipped: %d" % (len(all_intervals), n_skipped))

        return diseqs, all_intervals

        # # also collect transcript starts and ends, so we know when to pad input with N's
        # transcript_starts = []
        # transcript_ends = []
        # for transcript in transcript_intervals:
        #     transcript_starts.append(transcript.end5)
        #     transcript_ends.append(transcript.end3)
        #     itv = transcript.end5.expand(0, self.train_length)
        #
        #     while itv.within(transcript):
        #         all_intervals.append(itv)
        #         itv = itv.shift(self.train_length)
        # return all_intervals, transcript_starts, transcript_ends

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.intervals) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        x, y = self.__data_generation(indexes)
        # note that this need to match the model, see model.py
        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.intervals))
        np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""
        x, y = self.get_data([self.intervals[i] for i in indexes])

        return x, y

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
        y = np.concatenate([self.gtrack(_i) for _i in genomic_itvs])[:, 0]
        return y

    def get_data_single(self, itv, diseq_idx):
        diseq = self.diseqs[diseq_idx]
        # x
        x = self._get_x(itv, diseq)
        # y
        y = self._get_y(itv, diseq)
        return x, y

    def get_data(self, intervals):
        x = []
        y = []
        for itv, diseq_idx in intervals:
            _x, _y = self.get_data_single(itv, diseq_idx)
            x.append(_x)
            y.append(_y)
        x = np.asarray(x)
        y = np.asarray(y)
        # # 3-class encoding
        # y0 = self._convert_y(y[:, :, 0])
        # y1 = self._convert_y(y[:, :, 1])
        return x, y

    # def _convert_y(self, _y):
    #     # convert y to be 3 classes
    #     # class 0: y_i == 0
    #     # class 1: 0 < y_i < 1
    #     # class 2: y_i == 1
    #     # mask value -1 will be retained
    #     # input y should be 2-dimensional: N x L
    #     # output y will be 3-dimensional: N x L x 3
    #     assert len(_y.shape) == 2
    #     y = np.zeros((_y.shape[0], _y.shape[1], 3))
    #     assert np.all((0 <= _y) & (_y <= 1) | (_y == -1))
    #     # class 0
    #     y[:, :, 0][np.where(_y == 0)] = 1
    #     # class 1
    #     y[:, :, 1][np.where((0 < _y) & (_y < 1))] = 1
    #     # class 2
    #     y[:, :, 2][np.where(_y == 1)] = 1
    #     # mask
    #     for i in range(3):
    #         y[:, :, i][np.where(_y == -1)] = -1
    #     return y

    # def get_data(self, intervals):
    #     x = []
    #     y = []
    #     for itv, diseq_idx in intervals:
    #         _x, _y = self.get_data_single(itv, diseq_idx)
    #         x.append(_x)
    #         y.append(_y)
    #     x = np.asarray(x)
    #     y = np.asarray(y)
    #     return x, y[:, :, 0], y[:, :, 1]
