import numpy as np
import keras
import random
import itertools
import pandas as pd
from StringIO import StringIO
from subprocess import PIPE, Popen
import gzip
import datacorral as dc
from genome_kit import Genome, GenomeTrack, Interval
from dgutils.interval import DisjointIntervalsSequence
from dgutils.pandas import add_column, read_dataframe


class DataGeneratorVarLen(keras.utils.Sequence):

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

    def __init__(self, df, batch_size, length_grouping):
        self.batch_size = batch_size
        self.df = self._process_df(df)  # process sparse array
        # self.indexes = np.arange(len(self.df))
        self.length_grouping = length_grouping
        if self.length_grouping:
            self.index_groups = self._split_indexes(self.df)
            self.indexes = self._combine_indexes(self.index_groups)
        else:
            self.indexes = np.arange(len(self.df))

    def _split_indexes(self, df, n_groups=20):
        # split df indexes into a couple of groups
        # such that indexes within the same group has sequences of similar lengths
        # first figure out the min and max length for each group
        min_len = df['len'].min()
        max_len = df['len'].max()
        assert max_len - min_len >= n_groups
        split_len = np.linspace(min_len, max_len, n_groups+1, dtype=np.int)
        # add 1 to last element (since we'll be doing [left, right))
        split_len[-1] += 1
        len_groups = [(split_len[i], split_len[i + 1]) for i in range(len(split_len) - 1)]
        print("Length groups: {}".format(len_groups))
        # collect indexes for each group
        index_groups = []
        for l1, l2 in len_groups:
            _idxes = df.index[(df['len'] >= l1) & (df['len'] < l2)].tolist()
            print("Length range [{}, {}), number of entries: {}".format(l1, l2, len(_idxes)))
            index_groups.append(_idxes)
        return index_groups

    def _combine_indexes(self, index_groups):
        # just concatenate
        #  TODO we can also shuffle groups, although some batch will have non-ideal
        # length combination, since group size is not garanteed to be integer multiple of batch_size
        return list(itertools.chain.from_iterable(index_groups))

    def _make_pair_arr(self, seq, one_idx):

        def _make_arr(seq, one_idx):
            target = np.zeros((len(seq), len(seq)))
            target[one_idx] = 1
            return target

        def _mask(x):
            assert len(x.shape) == 2
            assert x.shape[0] == x.shape[1]
            x[np.tril_indices(x.shape[0])] = -1
            return x

        pair_matrix = _make_arr(seq, one_idx)
        pair_matrix = _mask(pair_matrix)
        return pair_matrix

    def _process_df(self, df):
        # # set lower triangular elements to -1
        #
        # def _make_arr(seq, one_idx):
        #     target = np.zeros((len(seq), len(seq)))
        #     target[one_idx] = 1
        #     return target
        #
        # def _mask(x):
        #     assert len(x.shape) == 2
        #     assert x.shape[0] == x.shape[1]
        #     x[np.tril_indices(x.shape[0])] = -1
        #     return x
        #
        # df = add_column(df, 'pair_matrix', ['seq', 'one_idx'], _make_arr)
        # df = add_column(df, 'pair_matrix', ['pair_matrix'], _mask)

        # normalize free energy
        # energy / sequence_length
        df = add_column(df, 'fe', ['free_energy', 'len'], lambda x, y: x/float(y))

        return df

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # get max sequence length in this batch
        max_len = self.df.iloc[indexes]['len'].max()
        # Generate data
        x, y = self.__data_generation(indexes, max_len)
        # note that this need to match the model, see model.py
        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # np.random.shuffle(self.indexes)
        if self.length_grouping:
            # shuffle index within each group, and recombine
            # also shuffle group order
            _index_groups = list(self.index_groups)  # make a copy
            np.random.shuffle(_index_groups)
            index_groups = []
            for index_group in _index_groups:
                np.random.shuffle(index_group)
                index_groups.append(index_group)
            self.index_groups = index_groups
            self.indexes = self._combine_indexes(self.index_groups)
        else:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes, max_len):
        """Generates data containing batch_size samples"""
        x1, y = self.get_data(indexes, max_len)

        return x1, y

    def _encode_seq(self, seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def get_data_single(self, idx):
        row = self.df.iloc[idx].copy()
        # x
        x1 = self._encode_seq(row['seq'])
        # y
        # y = row['pair_matrix'][:, :, np.newaxis]
        y = self._make_pair_arr(row['seq'], row['one_idx'])
        y = y[:, :, np.newaxis]
        y2 = [row['fe']]
        return x1, y, y2

    def get_data(self, indexes, max_len):
        x1 = []
        y = []
        y2 = []
        for idx in indexes:
            _x1, _y, _y2 = self.get_data_single(idx)
            # pad to max length
            _x1 = np.pad(_x1, ((0, max_len - _x1.shape[0]), (0, 0)),
                         'constant', constant_values=(0, 0))
            _y = np.pad(_y, ((0, max_len - _y.shape[0]), (0, max_len - _y.shape[1]), (0, 0)),
                        'constant', constant_values=-1)  # only specify one value since it has to be the same if padding multiple axes
            x1.append(_x1)
            y.append(_y)
            y2.append(_y2)
        x1 = np.asarray(x1)
        y = np.asarray(y)
        y2 = np.asarray(y2)
        return [x1, y], [y, y2]


def sample_structures(seq, num_structures):
    # use -N for generating non-redundant samples <- only use this for generating training data
    # use "-e 2" to generate all structures within 2Kcal/mol within the MFE
    p = Popen(['RNAsubopt',  '-N',  '-e',  '2'], stdin=PIPE,
              stdout=PIPE, stderr=PIPE, universal_newlines=True)
    stdout, stderr = p.communicate(input=seq)
    rc = p.returncode
    if rc != 0:
        msg = 'RNAeval returned error code %d\nstdout:\n%s\nstderr:\n%s\n' % (
            rc, stdout, stderr)
        raise Exception(msg)
    # parse output
    lines = stdout.splitlines()
    # assert len(lines) == n_samples + 1
    lines = lines[1:]
    # convert to idx array
    all_vals = []

    # check if we have enough structures
    # if not, duplicate
    if len(lines) > num_structures:
        lines = lines[:num_structures]
    elif len(lines) < num_structures:
        # sample with replacement
        lines = np.random.choice(lines, num_structures)

    for line in lines:
        s, e = line.split(' ')
        fe = float(e)
        assert fe < 0  # just check it can be converted
        assert len(s) == len(seq)
        # convert to ct file (add a fake energy, otherwise b2ct won't run)
        input_str = '>seq\n{}\n{} ({})'.format(seq, s, e)
        p = Popen(['b2ct'], stdin=PIPE,
                  stdout=PIPE, stderr=PIPE, universal_newlines=True)
        stdout, stderr = p.communicate(input=input_str)
        rc = p.returncode
        if rc != 0:
            msg = 'b2ct returned error code %d\nstdout:\n%s\nstderr:\n%s\n' % (
                rc, stdout, stderr)
            raise Exception(msg)
        # load data
        df = pd.read_csv(StringIO(stdout), skiprows=1, header=None,
                         names=['i1', 'base', 'idx_i', 'i2', 'idx_j', 'i3'],
                         sep=r"\s*")
        assert ''.join(df['base'].tolist()) == seq, ''.join(df['base'].tolist())
        # matrix
        vals = np.zeros((len(seq), len(seq)))
        for _, row in df.iterrows():
            idx_i = row['idx_i']
            idx_j = row['idx_j'] - 1
            if idx_j != -1:
                vals[idx_i, idx_j] = 1
                vals[idx_j, idx_i] = 1
        # process into one-index
        # set lower triangular to 0
        vals[np.tril_indices(vals.shape[0])] = 0
        pred_idx = np.where(vals == 1)
        all_vals.append((pred_idx, fe))
    assert len(all_vals) == num_structures
    return all_vals


class FixedLengthDataBatch(object):
    """
    A 'batch' of sequences with the same length.
    Each sequence has multiple structures associated.
    It also initialize an counter
    """

    def __init__(self, num_seq, seq_len, num_structures):
        self.x = self._generate_sequences(num_seq, seq_len)
        self.y = self._generate_structures(self.x, num_structures)
        self._idx = num_structures - 1

    def is_valid(self):
        # check to see if we've counted down to 0
        if self._idx >= 0:
            return True
        elif self._idx == -1:
            return False
        else:
            raise ValueError  # shouldn't be here

    def pop_data(self):
        assert self.is_valid()
        self._idx -= 1
        return self._get_data(self._idx + 1)

    def _get_data(self, idx):
        structs = [ss[idx][0] for ss in self.y]
        fes = [ss[idx][1] for ss in self.y]
        return self.x, structs, fes

    def _generate_sequences(self, num_seq, seq_len):
        seqs = []  # TODO use set so we don't add duplicates (but the probability of getting duplicate is very low)
        for _ in range(num_seq):
            seq = ''.join(random.choice(list('ACGU')) for _ in seq_len)
            seqs.append(seq)
        return seqs

    def _generate_structures(self, seqs, num_structures):
        structs = []
        for seq in seqs:
            ss = sample_structures(seq, num_structures)
            structs.append(ss)
        return structs


class DataGeneratorInfinite(keras.utils.Sequence):

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

    def __init__(self, batch_size, num_batches, min_len=20, max_len=200, num_structures=10):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.min_len = min_len
        self.max_len = max_len
        self.num_structures = num_structures
        self._data = {i: None for i in num_batches}  # make it dict, easy to use

    def __len__(self):
        """number of batches per each epoch"""
        return self.num_batches

    def __getitem__(self, index):
        """Generate one batch of data"""
        # check if we need to generate new data
        if not self._data[index]:
            self._data[index] = FixedLengthDataBatch(np.random.randint(self.min_len, self.max_len), self.num_structures)
        _x, _y, _e = self._data[index].pop_data()
        x, y = self._encode_data(_x, _y, _e)
        return x, y

    def on_epoch_end(self):
        """decide whether to wipe data"""
        # wolg, check the first one
        if not self._data[0] or not self._data[0].is_valid():
            self._data = {i: None for i in self.num_batches}

    def _encode_seq(self, seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def _make_pair_arr(self, seq, one_idx):

        def _make_arr(seq, one_idx):
            target = np.zeros((len(seq), len(seq)))
            target[one_idx] = 1
            return target

        def _mask(x):
            assert len(x.shape) == 2
            assert x.shape[0] == x.shape[1]
            x[np.tril_indices(x.shape[0])] = -1
            return x

        pair_matrix = _make_arr(seq, one_idx)
        pair_matrix = _mask(pair_matrix)
        return pair_matrix

    def _encode_data(self, x, y, e):
        x_data = []
        y_data = []
        assert len(x) == len(y)
        assert len(x) == len(e)
        for seq, one_idx in zip(x, y):
            _x = self._encode_seq(seq)
            arr = self._make_pair_arr(seq, one_idx)
            x_data.append(_x)
            y_data.append(arr)
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        e_data = np.asarray(e)[np.newaxis, :]
        return [x_data, y_data], [y_data, e_data]


# class DataGeneratorFixedLen(keras.utils.Sequence):
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
#         self.df = self._process_df(df)
#         self.indexes = np.arange(len(self.df))
#
#     def _process_df(self, df):
#         # set lower triangular elements to -1
#
#         def _mask(x):
#             assert len(x.shape) == 2
#             assert x.shape[0] == x.shape[1]
#             x[np.tril_indices(x.shape[0])] = -1
#             return x
#
#         df = add_column(df, 'pair_matrix', ['pair_matrix'], _mask)
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
#         x1, y = self.get_data(indexes)
#
#         return x1, y
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
#         # x2 = self._encode_seq(row['sequence_rev'])
#         # x2 = self._encode_seq(row['sequence_rev_comp'])
#         # y
#         y = row['pair_matrix'][:, :, np.newaxis]
#         return x1, y
#
#     def get_data(self, indexes):
#         x1 = []
#         y = []
#         for idx in indexes:
#             _x1, _y = self.get_data_single(idx)
#             x1.append(_x1)
#             y.append(_y)
#         x1 = np.asarray(x1)
#         y = np.asarray(y)
#         return [x1, y], y  # for autoregressive training

