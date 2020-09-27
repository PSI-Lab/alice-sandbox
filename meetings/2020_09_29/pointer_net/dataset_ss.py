import itertools
import logging
import numpy as np
import torch
from torch.utils.data import Dataset


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class BoundingBoxDataset(Dataset):
    def __init__(self, df):
        # restrict to cases where stage 1 sensitivity is 100%
        logging.info(f"Dropping rows where stage 1 sensitivity is not 100%. Before: {len(df)}")
        df = df[df['bb_overlap'].apply(min) == 1]
        logging.info(f"After: {len(df)}")
        # TODO reindexing?

        # verify that feature dimensions are consistent, and store it (needed for initializing encode)
        feature_dims = df['features'].apply(lambda x: x.shape[1])
        assert feature_dims.nunique() == 1
        self.feature_dim = feature_dims.iloc[0]

        # save df
        self.df = df




        # self.prng = np.random.RandomState(seed=seed)
        # self.input_dim = high
        #
        # # Here, we assuming that the shape of each sample is a list of list of a single integer, e.g., [[10], [3], [5], [0]]
        # # It is for an easier extension later even though it is not necessary for this simple sorting example
        # self.seqs = [list(map(lambda x: [x], self.prng.choice(np.arange(low, high),
        #                                                       size=self.prng.randint(min_len, max_len + 1)).tolist()))
        #              for _ in range(num_samples)]
        # self.labels = [sorted(range(len(seq)), key=seq.__getitem__) for seq in self.seqs]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        features = row['features']
        target_idx = row['target_idx']
        input_len = row['input_len']
        return torch.FloatTensor(features), input_len, target_idx


        # seq = self.seqs[index]
        # label = self.labels[index]
        #
        # len_seq = len(seq)
        # row_col_index = list(zip(*[(i, number) for i, numbers in enumerate(seq) for number in numbers]))
        # num_values = len(row_col_index[0])
        #
        # i = torch.LongTensor(row_col_index)
        # v = torch.FloatTensor([1] * num_values)
        # data = torch.sparse.FloatTensor(i, v, torch.Size([len_seq, self.input_dim]))
        #
        # return data, len_seq, label

    def __len__(self):
        return len(self.df)


def bb_collate_fn(batch):
    batch_size = len(batch)

    # sort by sequence length, e.g. number of bounding boxes
    # largest -> smallest
    sorted_seqs, sorted_lengths, sorted_labels = zip(*sorted(batch, key=lambda x: x[1], reverse=True))

    # pad features
    padded_seqs = [seq.resize_as_(sorted_seqs[0]) for seq in sorted_seqs]
    # batch_size X max_seq_len X input_dim
    seq_tensor = torch.stack(padded_seqs)

    # sequence lengths
    length_tensor = torch.LongTensor(sorted_lengths)

    # pad target index
    padded_labels = list(zip(*(itertools.zip_longest(*sorted_labels, fillvalue=-1))))
    # batch_size X max_seq_len (-1 padding)
    label_tensor = torch.LongTensor(padded_labels).view(batch_size, -1)

    return seq_tensor, length_tensor, label_tensor
