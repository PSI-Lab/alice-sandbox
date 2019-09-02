import numpy as np
import keras
import gzip
import datacorral as dc
from genome_kit import Genome, GenomeTrack, Interval
from dgutils.interval import DisjointIntervalsSequence
from dgutils.pandas import add_column, read_dataframe


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
        self.df = self._process_df(df)
        self.indexes = np.arange(len(self.df))

    def _process_df(self, df):
        # set lower triangular elements to -1

        def _mask(x):
            assert len(x.shape) == 2
            assert x.shape[0] == x.shape[1]
            x[np.tril_indices(x.shape[0])] = -1
            return x

        df = add_column(df, 'pair_matrix', ['pair_matrix'], _mask)
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
        y = row['pair_matrix'][:, :, np.newaxis]
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
        return [x1, y], y  # for autoregressive training

