import numpy as np
import keras
import gzip
import datacorral as dc
from genome_kit import Genome, GenomeTrack, Interval
from dgutils.interval import DisjointIntervalsSequence
from dgutils.pandas import add_column, read_dataframe
from model import resolve_contex
# from config import config


class DataGenerator(keras.utils.Sequence):

    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    def __init__(self, df, config):

        # self.genome = Genome(config['genome_annotation'])
        self.train_length = config['example_length']
        self.context = resolve_contex(config['dense_conv'])
        self.batch_size = config['batch_size']
        # self.gtrack = GenomeTrack(config['gtrack'])

        # df = self._process_df_array(df, config['target_cols'], 'target_val')
        df = self._process_df_array(df, 'data', 'target_val')

        # df = add_column(df, 'sequence', ['disjoint_intervals'], self._add_sequence)
        # df = add_column(df, 'log_tpm', ['tpm'], np.log)

        self.df, self.intervals = self._process_transcript_intervals(df)

        # self.df, self.intervals = self._process_transcript_intervals(df, config['min_log_tpm'],
        #                                                                  config['min_rep_corr'],
        #                                                                  config['example_reweighting'])
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

    def _add_sequence(self, itvs):
        diseq = DisjointIntervalsSequence(itvs, self.genome)
        return diseq.dna(diseq.interval)

    def _process_df_array(self, df, old_col, new_col_name):
        # tmp, just add 1 dimension
        df = add_column(df, new_col_name, [old_col], lambda x: np.asarray(x)[:, np.newaxis])
        return df.drop(columns=[old_col])

    # def _process_df_array(self, df, val_cols, new_col_name):
    #     # combine all value columns into a single numpy array
    #
    #     def _combine_col(*cols):
    #         x = np.vstack(cols).T
    #         assert len(x.shape) == 2
    #         assert x.shape[0] == len(cols[0])
    #         assert x.shape[1] == len(cols)
    #         return x
    #
    #     df = add_column(df, new_col_name, val_cols, _combine_col)
    #     return df.drop(columns=val_cols)

    def _process_transcript_intervals(self, df):
        """process transcript intervals to fixed length shorter intervals for training"""
        # diseqs = []
        # all_intervals = []  # list of tuples: diseq_interval, diseq_idx, example_weight
        all_intervals = []  # list of tuples: df_idx, array_start, array_end

        n_skipped = 0

        for idx, row in df.iterrows():
            sequence = row['sequence']
            val_arr = row['target_val']
            assert len(sequence) == val_arr.shape[0]

            for _tmp_idx in range(len(sequence)//self.train_length):
                arr_start = _tmp_idx * self.train_length
                arr_end = (_tmp_idx + 1) * self.train_length

                _val = val_arr[arr_start:arr_end, :]
                if np.all(_val == -1) or np.all(_val == 0) or np.all(_val == 1):
                    # print("Skip %s in diseq %s" % (itv, diseq.intervals))
                    n_skipped += 1
                else:
                    all_intervals.append((idx, arr_start, arr_end))

                if n_skipped % 100 == 0 or len(all_intervals) % 1000 == 0:
                    print("Intervals: %d, skipped: %d" % (len(all_intervals), n_skipped))

        return df, all_intervals

    # def _process_transcript_intervals(self, df, min_log_tpm, min_rep_corr, example_reweighting):
    #     """process transcript intervals to fixed length shorter intervals for training"""
    #     # diseqs = []
    #     # all_intervals = []  # list of tuples: diseq_interval, diseq_idx, example_weight
    #     all_intervals = []  # list of tuples: df_idx, array_start, array_end, example_weight
    #
    #     n_skipped = 0
    #
    #     for idx, row in df.iterrows():
    #         itvs = row['disjoint_intervals']
    #
    #         # filter
    #         # nan should be skipped
    #         if not (row['log_tpm'] >= min_log_tpm):
    #             print("Skip {} due to low log TPM {}".format(row['transcript_id'], row['log_tpm']))
    #             continue
    #         if not (row['pearson_corr'] >= min_rep_corr):
    #             print("Skip {} due to low rep corr {}".format(row['transcript_id'], row['pearson_corr']))
    #             continue
    #
    #         if example_reweighting:
    #             # calculate example weight for all intervals within this transcript
    #             transcript_weight = self._example_weight(row['log_tpm'], row['pearson_corr'])
    #         else:
    #             transcript_weight = 1.0
    #
    #         sequence = row['sequence']
    #         val_arr = row['target_val']
    #         assert len(sequence) == val_arr.shape[0]
    #
    #         for _tmp_idx in range(len(sequence)//self.train_length):
    #             arr_start = _tmp_idx * self.train_length
    #             arr_end = (_tmp_idx + 1) * self.train_length
    #
    #             _val = val_arr[arr_start:arr_end, :]
    #             if np.all(_val == -1) or np.all(_val == 0) or np.all(_val == 1):
    #                 # print("Skip %s in diseq %s" % (itv, diseq.intervals))
    #                 n_skipped += 1
    #             else:
    #                 all_intervals.append((idx, arr_start, arr_end, transcript_weight))
    #
    #             if n_skipped % 100 == 0 or len(all_intervals) % 1000 == 0:
    #                 print("Intervals: %d, skipped: %d" % (len(all_intervals), n_skipped))
    #
    #         # diseq = DisjointIntervalsSequence(itvs, self.genome)
    #         # diseqs.append(diseq)
    #         # diseq_idx = len(diseqs) - 1
    #         #
    #         # itv = diseq.interval.end5.expand(0, self.train_length)
    #         # while itv.within(diseq.interval):
    #         #     # discard interval if all positions are missing output
    #         #     # or all positions == 0
    #         #     # or all positions == 1
    #         #     _y = self._get_y(itv, diseq)
    #         #     if np.all(_y == -1) or np.all(_y == 0) or np.all(_y == 1):
    #         #         # print("Skip %s in diseq %s" % (itv, diseq.intervals))
    #         #         n_skipped += 1
    #         #     else:
    #         #         all_intervals.append((itv, diseq_idx, transcript_weight))
    #         #     itv = itv.shift(self.train_length)
    #         #
    #         #     if n_skipped % 100 == 0 or len(all_intervals) % 1000 == 0:
    #         #         print("Intervals: %d, skipped: %d" % (len(all_intervals), n_skipped))
    #
    #     return df, all_intervals

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

    # def _get_x(self, itv, diseq):
    #     # middle part
    #     seq_mid = diseq.dna(itv)
    #     # upstream
    #     _ll = len(Interval.spanning(diseq.interval.end5, itv.end5))
    #     if _ll < self.context / 2:
    #         seq_up = 'N' * (self.context / 2 - _ll) + diseq.dna(itv.end5.expand(_ll, 0))
    #     else:
    #         seq_up = diseq.dna(itv.end5.expand(self.context / 2, 0))
    #     # downstream
    #     _lr = len(Interval.spanning(diseq.interval.end3, itv.end3))
    #     if _lr < self.context / 2:
    #         seq_dn = diseq.dna(itv.end3.expand(0, _lr)) + 'N' * (self.context / 2 - _lr)
    #     else:
    #         seq_dn = diseq.dna(itv.end3.expand(0, self.context / 2))
    #     seq = seq_up + seq_mid + seq_dn
    #     seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')
    #     x = np.asarray(map(int, list(seq)))
    #     x = self.DNA_ENCODING[x.astype('int8')]
    #     return x

    def _get_x(self, sequence, start, end):
        # middle part
        seq_mid = sequence[start:end]
        # upstream
        if start < self.context / 2:
            seq_up = 'N' * (self.context / 2 - start) + sequence[:start]
        else:
            seq_up = sequence[(start - self.context / 2):start]
        # downstream
        _lr = len(sequence) - end
        if _lr < self.context / 2:
            seq_dn = sequence[end:] + 'N' * (self.context / 2 - _lr)
        else:
            seq_dn = sequence[end:(end + self.context / 2)]
        seq = seq_up + seq_mid + seq_dn
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def _get_y(self, target, start, end):
        return target[start:end, :]

    def get_data_single(self, idx, arr_start, arr_end):
        row = self.df.iloc[idx].copy()
        # x
        x = self._get_x(row['sequence'], arr_start, arr_end)
        # y
        y = self._get_y(row['target_val'], arr_start, arr_end)
        return x, y

    # def get_data_single(self, idx, arr_start, arr_end, transcript_weight):
    #     row = self.df.iloc[idx].copy()
    #     # x
    #     x = self._get_x(row['sequence'], arr_start, arr_end)
    #     # y
    #     y = self._get_y(row['target_val'], arr_start, arr_end)
    #     return x, y, transcript_weight

    def get_data(self, intervals):
        x = []
        y = []
        for idx, arr_start, arr_end in intervals:
            _x, _y = self.get_data_single(idx, arr_start, arr_end)
            x.append(_x)
            y.append(_y)
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y
