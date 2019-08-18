import numpy as np
import keras
import gzip
import datacorral as dc
from genome_kit import Genome, GenomeTrack, Interval
from dgutils.interval import DisjointIntervalsSequence
from dgutils.pandas import add_column, read_dataframe


# class DataGenerator(keras.utils.Sequence):
#
#     DNA_ENCODING = np.asarray([[0, 0, 0, 0],
#                                [1, 0, 0, 0],
#                                [0, 1, 0, 0],
#                                [0, 0, 1, 0],
#                                [0, 0, 0, 1]])
#
#     complement_mapping = {
#         'A': 'U',
#         'C': 'G',
#         'G': 'C',
#         'U': 'A',
#     }
#
#     def __init__(self, df, batch_size):
#         self.batch_size = batch_size
#         # process df
#         self.df = self._process_df(df)
#         self.indexes = np.arange(len(self.df))
#
#     def _process_df(self, df):
#         # add 1 dimension
#         df = add_column(df, 'mid_point_pair_prob',
#                         ['mid_point_pair_prob'], lambda x: np.asarray(x)[:, np.newaxis])
#         # add sequence rev
#         df = add_column(df, 'sequence_rev', ['sequence'],
#                         lambda x: x[::-1])
#         # # add sequence rev comp
#         # df = add_column(df, 'sequence_rev_comp', ['sequence'],
#         #                 lambda x: ''.join(map(lambda y: self.complement_mapping[y], x)[::-1]))
#         return df
#
#     def __len__(self):
#         """Denotes the number of batches per epoch"""
#         return int(np.floor(len(self.df) / self.batch_size))
#
#     def __getitem__(self, index):
#         """Generate one batch of data"""
#         # Generate indexes of the batch
#         indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#         # Generate data
#         x, y = self.__data_generation(indexes)
#         # note that this need to match the model, see model.py
#         return x, y
#
#     def on_epoch_end(self):
#         """Updates indexes after each epoch"""
#         np.random.shuffle(self.indexes)
#
#     def __data_generation(self, indexes):
#         """Generates data containing batch_size samples"""
#         x1, x2, y = self.get_data(indexes)
#
#         return [x1, x2], y
#
#     def _encode_seq(self, seq):
#         seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
#         x = np.asarray(map(int, list(seq)))
#         x = self.DNA_ENCODING[x.astype('int8')]
#         return x
#
#     def get_data_single(self, idx):
#         row = self.df.iloc[idx].copy()
#         # x
#         x1 = self._encode_seq(row['sequence'])
#         x2 = self._encode_seq(row['sequence_rev'])
#         # x2 = self._encode_seq(row['sequence_rev_comp'])
#         # y
#         y = row['mid_point_pair_prob']
#         return x1, x2, y
#
#     def get_data(self, indexes):
#         x1 = []
#         x2 = []
#         y = []
#         for idx in indexes:
#             _x1, _x2, _y = self.get_data_single(idx)
#             x1.append(_x1)
#             x2.append(_x2)
#             y.append(_y)
#         x1 = np.asarray(x1)
#         x2 = np.asarray(x2)
#         y = np.asarray(y)
#         return x1, x2, y


class DataGenerator(keras.utils.Sequence):

    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    complement_mapping = {
        'A': 'U',
        'C': 'G',
        'G': 'C',
        'U': 'A',
    }

    def __init__(self, df, batch_size):
        self.batch_size = batch_size
        # process df
        self.df = self._process_df(df)
        self.indexes = np.arange(len(self.df))

    def _process_df(self, df):
        # add 1 dimension
        df = add_column(df, 'mid_point_pair_prob',
                        ['mid_point_pair_prob'], lambda x: np.asarray(x)[:, np.newaxis])
        # # add sequence rev
        # df = add_column(df, 'sequence_rev', ['sequence'],
        #                 lambda x: x[::-1])
        # # add sequence rev comp
        # df = add_column(df, 'sequence_rev_comp', ['sequence'],
        #                 lambda x: ''.join(map(lambda y: self.complement_mapping[y], x)[::-1]))
        return df

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.df) / self.batch_size))

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
        np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""
        x1, y = self.get_data(indexes)

        return x1, y

    def _encode_seq(self, seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def get_data_single(self, idx):
        row = self.df.iloc[idx].copy()
        # x
        x1 = self._encode_seq(row['sequence'])
        # x2 = self._encode_seq(row['sequence_rev'])
        # x2 = self._encode_seq(row['sequence_rev_comp'])
        # y
        y = row['mid_point_pair_prob']
        return x1, y

    def get_data(self, indexes):
        x1 = []
        y = []
        for idx in indexes:
            _x1, _y = self.get_data_single(idx)
            x1.append(_x1)
            y.append(_y)
        x1 = np.asarray(x1)
        y = np.asarray(y)
        return x1, y

