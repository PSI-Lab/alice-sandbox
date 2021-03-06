import os
import sys
import subprocess
import logging
import pprint
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import datacorral as dc
# sys.path.insert(0, '../../rna_ss/')
# from utils import db2pairs
# from local_struct_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb
sys.path.insert(0, '../rna_ss_utils/')  # FIXME hacky
from utils import db2pairs
from local_struct_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb


def add_column(df, output_col, input_cols, func):
    # make a tuple of values of the requested input columns
    input_values = tuple(df[x].values for x in input_cols)

    # transpose to make a list of value tuples, one per row
    args = zip(*input_values)

    # evaluate the function to generate the values of the new column
    output_values = [func(*x) for x in args]

    # make a new dataframe with the new column added
    columns = {x: df[x].values for x in df.columns}
    columns[output_col] = output_values
    return pd.DataFrame(columns)


def set_up_logging(path_result):
    # make result dir if non existing
    if not os.path.isdir(path_result):
        os.makedirs(path_result)

    log_format = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    file_logger = logging.FileHandler(os.path.join(path_result, 'run.log'))
    file_logger.setFormatter(log_format)
    root_logger.addHandler(file_logger)
    console_logger = logging.StreamHandler()
    console_logger.setFormatter(log_format)
    root_logger.addHandler(console_logger)


def length_grouping(df, n):
    # split input df into multiple dfs,
    # according to the length of column `seq`
    dfs = []
    # first collect all lengths
    if 'len' not in df.columns:
        df = add_column(df, 'len', ['seq'], len)
    # sort
    df.sort_values(by=['len'], inplace=True)
    # split into bins
    idxes = np.linspace(0, len(df), num=n + 1, dtype=int)
    for start, end in zip(idxes[:-1], idxes[1:]):
        dfs.append(df[start:end])
    return dfs


class MyDataSet(Dataset):
    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    # TODO length grouping

    def __init__(self, df):
        self.len = len(df)
        self.df = df

    def _encode_seq(self, seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        x = np.asarray([int(x) for x in list(seq)])
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def tile_and_stack(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 4
        l = x.shape[0]
        x1 = x[:, np.newaxis, :]
        x2 = x[np.newaxis, :, :]
        x1 = np.repeat(x1, l, axis=1)
        x2 = np.repeat(x2, l, axis=0)
        return np.concatenate([x1, x2], axis=2)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        seq = row['seq']
        x = self._encode_seq(seq)
        x = self.tile_and_stack(x)

        # expand one_idx and bb into target arrays
        one_idx = row['one_idx']
        pairs, structure_arr = one_idx2arr(one_idx, len(seq), remove_lower_triangular=True)
        bounding_boxes = row['bounding_boxes']
        target_stem_on, target_iloop_on, target_hloop_on, \
        mask_stem_on, mask_iloop_on, mask_hloop_on, \
        target_stem_location_x, target_stem_location_y, target_iloop_location_x, target_iloop_location_y, \
        target_hloop_location_x, target_hloop_location_y, \
        target_stem_sm_size, target_iloop_sm_size_x, target_iloop_sm_size_y, target_hloop_sm_size, \
        target_stem_sl_size, target_iloop_sl_size_x, target_iloop_sl_size_y, target_hloop_sl_size, \
        mask_stem_location_size, mask_iloop_location_size, \
        mask_hloop_location_size = make_target_pixel_bb(structure_arr, bounding_boxes)

        # organize
        y = {
            # stem
            'stem_on': torch.from_numpy(target_stem_on[:, :, np.newaxis]).float(),  # add singleton dimension
            # FIXME these are int, no need to convert to float
            'stem_location_x': torch.from_numpy(target_stem_location_x[:, :, np.newaxis]).float(),   # add singleton dimension (these are integer index of softmax index)
            'stem_location_y': torch.from_numpy(target_stem_location_y[:, :, np.newaxis]).float(),
            'stem_sm_size': torch.from_numpy(target_stem_sm_size[:, :, np.newaxis]).float(),
            'stem_sl_size': torch.from_numpy(target_stem_sl_size[:, :, np.newaxis]).float(),
            
            # iloop
            'iloop_on': torch.from_numpy(target_iloop_on[:, :, np.newaxis]).float(),
            'iloop_location_x': torch.from_numpy(target_iloop_location_x[:, :, np.newaxis]).float(),
            'iloop_location_y': torch.from_numpy(target_iloop_location_y[:, :, np.newaxis]).float(),
            'iloop_sm_size_x': torch.from_numpy(target_iloop_sm_size_x[:, :, np.newaxis]).float(),
            'iloop_sm_size_y': torch.from_numpy(target_iloop_sm_size_y[:, :, np.newaxis]).float(),
            'iloop_sl_size_x': torch.from_numpy(target_iloop_sl_size_x[:, :, np.newaxis]).float(),
            'iloop_sl_size_y': torch.from_numpy(target_iloop_sl_size_y[:, :, np.newaxis]).float(),
            
            # hloop
            'hloop_on': torch.from_numpy(target_hloop_on[:, :, np.newaxis]).float(),
            'hloop_location_x': torch.from_numpy(target_hloop_location_x[:, :, np.newaxis]).float(),
            'hloop_location_y': torch.from_numpy(target_hloop_location_y[:, :, np.newaxis]).float(),
            'hloop_sm_size': torch.from_numpy(target_hloop_sm_size[:, :, np.newaxis]).float(),
            'hloop_sl_size': torch.from_numpy(target_hloop_sl_size[:, :, np.newaxis]).float(),
        }

        m = {
            # FIXME use int type to save memory
            # stem
            'stem_on': torch.from_numpy(mask_stem_on[:, :, np.newaxis]).float(),
            'stem_location_size': torch.from_numpy(mask_stem_location_size[:, :, np.newaxis]).float(),
            
            # iloop
            'iloop_on': torch.from_numpy(mask_iloop_on[:, :, np.newaxis]).float(),
            'iloop_location_size': torch.from_numpy(mask_iloop_location_size[:, :, np.newaxis]).float(),
            
            # hloop
            'hloop_on': torch.from_numpy(mask_hloop_on[:, :, np.newaxis]).float(),
            'hloop_location_size': torch.from_numpy(mask_hloop_location_size[:, :, np.newaxis]).float(),
        }

        # extra data saved for evaluation purpose (not used for training)
        md = {
            'seq': row['seq'],
            'one_idx': row['one_idx'],
            'bounding_boxes': row['bounding_boxes'],
        }

        # # debug
        # print(x.shape)
        # print({k: v.shape for k, v in y.items()})
        # print({k: v.shape for k, v in m.items()})
        return torch.from_numpy(x).float(), y, m, md


    def __len__(self):
        return self.len


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate2D:
    def __init__(self):
        pass

    def pad_collate(self, batch):
        """
        args:
            batch - list of (x, y, m, md), where y and m are dict

        reutrn:
            xs - x after padding
            ys - y after padding, dict of tensor
            ms - m after padding, dict of tensor
            md - list
        """
        # keys of dict
        keys_y = batch[0][1].keys()
        keys_m = batch[0][2].keys()
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[0], batch))
        # we expect it to be symmetric between dim 0 and 1
        assert max_len == max(map(lambda x: x[0].shape[1], batch))

        # naive implementation for easy debug
        xs = []
        ys = {k: [] for k in keys_y}
        ms = {k: [] for k in keys_m}
        mds = []
        for x, y, m, md in batch:
            # store metadata
            mds.append(md)
            # permute: prep for stacking: batch x channel x H x W
            # process x
            _x = pad_tensor(pad_tensor(x, pad=max_len, dim=0), pad=max_len, dim=1).permute(2, 0, 1)
            xs.append(_x)
            # process all y's
            for k in keys_y:
                assert y[k].shape[0] <= max_len
                assert y[k].shape[1] <= max_len
                _y = pad_tensor(pad_tensor(y[k], pad=max_len, dim=0), pad=max_len, dim=1).permute(2, 0, 1)
                ys[k].append(_y)
            # process all m's
            for k in keys_m:
                assert m[k].shape[0] <= max_len
                assert m[k].shape[1] <= max_len
                _m = pad_tensor(pad_tensor(m[k], pad=max_len, dim=0), pad=max_len, dim=1).permute(2, 0, 1)
                ms[k].append(_m)
        # stack tensor
        xs = torch.stack(xs, dim=0)
        for k in keys_y:
            ys[k] = torch.stack(ys[k], dim=0)
        for k in keys_m:
            ms[k] = torch.stack(ms[k], dim=0)
        return xs, ys, ms, mds


    def __call__(self, batch):
        return self.pad_collate(batch)



class LatentVarModel(nn.Module):
    OUTPUT_ORDER = ['stem_on', 'stem_location_x', 'stem_location_y', 'stem_sm_size', 'stem_sl_size',
                 'iloop_on', 'iloop_location_x', 'iloop_location_y', 'iloop_sm_size_x',
                 'iloop_sl_size_x', 'iloop_sm_size_y', 'iloop_sl_size_y',
                 'hloop_on', 'hloop_location_x', 'hloop_location_y', 'hloop_sm_size', 'hloop_sl_size']

    # softmax units
    # needed for one-hot encoding (for encoder input)
    NUM_CLASS = {
        'stem_location_x': 12,
        'stem_location_y': 12,
        'stem_sm_size': 11,
        'iloop_location_x': 12,
        'iloop_location_y': 12,
        'iloop_sm_size_x': 11,
        'iloop_sm_size_y': 11,
        'hloop_location_x': 12,
        'hloop_location_y': 12,
        'hloop_sm_size': 11,
    }

    def __init__(self, num_filters, num_output, filter_width, latent_dim, dropout):
        super(LatentVarModel, self).__init__()

        num_filters = [8] + num_filters
        filter_width = [None] + filter_width
        cnn_layers = []
        for i, (nf, fw) in enumerate(zip(num_filters[1:], filter_width[1:])):
            assert fw % 2 == 1  # odd
            cnn_layers.append(nn.Conv2d(num_filters[i], nf, kernel_size=fw, stride=1, padding=fw//2))
            cnn_layers.append(nn.BatchNorm2d(nf))
            cnn_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                cnn_layers.append(nn.Dropout(dropout))
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # self.fc = nn.Conv2d(num_filters[-1], 5, kernel_size=1)
        # self.fc = nn.Conv2d(num_filters[-1], 50, kernel_size=1)

        # posterior network
        # "encoder" of the cvae, x + y -> z's param
        self.posterior_fc = nn.Sequential(
            nn.Conv2d(num_filters[-1] + num_output, 50, kernel_size=1),
            nn.ReLU(),
        )
        # posterior mean and logvar
        # latent variable layer  # TODO revisit 50 dim?
        # input channel is 50 since it already passed through x+y -> 50 mapping
        self.posterior_mean = nn.Conv2d(50, latent_dim, kernel_size=1)
        self.posterior_logvar = nn.Conv2d(50, latent_dim, kernel_size=1)

        # prior network
        self.prior_fc = nn.Sequential(
            nn.Conv2d(num_filters[-1], 50, kernel_size=1),
            nn.ReLU(),
        )
        # TODO revisit 50 dim?
        self.prior_mean = nn.Conv2d(50, latent_dim, kernel_size=1)
        self.prior_logvar = nn.Conv2d(50, latent_dim, kernel_size=1)

        # add output specific hidden layers

        # stem
        self.out_stem_on = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_stem_loc_x = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_stem_loc_y = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        # self.out_stem_siz = nn.Sequential(
        #     nn.Conv2d(50, 20, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(20, 11, kernel_size=1),
        #     nn.LogSoftmax(dim=1),
        # )
        self.hid_stem_siz = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
        )
        self.out_stem_sm_siz = nn.Sequential(
            nn.Conv2d(20, 11, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_stem_sl_siz = nn.Sequential(
            nn.Conv2d(20, 1, kernel_size=1),
        )
        
        # iloop
        self.out_iloop_on = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_iloop_loc_x = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_iloop_loc_y = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        # self.out_iloop_siz_x = nn.Sequential(
        #     nn.Conv2d(50, 20, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(20, 11, kernel_size=1),
        #     nn.LogSoftmax(dim=1),
        # )
        self.hid_iloop_siz_x = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
        )
        self.out_iloop_sm_siz_x = nn.Sequential(
            nn.Conv2d(20, 11, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_iloop_sl_siz_x = nn.Sequential(
            nn.Conv2d(20, 1, kernel_size=1),
        )

        # self.out_iloop_siz_y = nn.Sequential(
        #     nn.Conv2d(50, 20, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(20, 11, kernel_size=1),
        #     nn.LogSoftmax(dim=1),
        # )
        self.hid_iloop_siz_y = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
        )
        self.out_iloop_sm_siz_y = nn.Sequential(
            nn.Conv2d(20, 11, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_iloop_sl_siz_y = nn.Sequential(
            nn.Conv2d(20, 1, kernel_size=1),
        )
        
        # hloop
        self.out_hloop_on = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_hloop_loc_x = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_hloop_loc_y = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        # self.out_hloop_siz = nn.Sequential(
        #     nn.Conv2d(50, 20, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(20, 11, kernel_size=1),
        #     nn.LogSoftmax(dim=1),
        # )
        self.hid_hloop_siz = nn.Sequential(
            nn.Conv2d(num_filters[-1] + latent_dim, 20, kernel_size=1),
            nn.ReLU(),
        )
        self.out_hloop_sm_siz = nn.Sequential(
            nn.Conv2d(20, 11, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_hloop_sl_siz = nn.Sequential(
            nn.Conv2d(20, 1, kernel_size=1),
        )

    def _collapse_y(self, y):
        # y is a dictionary, concat into one array
        # also convert softmax-output from int to one-hot
        # use fixed key ordering
        ys= []
        for k in self.OUTPUT_ORDER:
            # output names defined in self.NUM_CLASS are softmax-output type
            if k in self.NUM_CLASS:
                # shape: batch x channel x h x w
                assert y[k].shape[1] == 1
                # drop channel dim (which is 1) since we'll create one-hot dim and swap here
                # convert to int Tensor since one_hot only works with index
                target = y[k][:, 0, :, :].to(torch.int64)
                one_hot = torch.nn.functional.one_hot(target, num_classes=self.NUM_CLASS[k])
                # swap axis
                one_hot = one_hot.permute(0, 3, 1, 2)
                # convert to float
                ys.append(one_hot.to(torch.float32))
            else:
                ys.append(y[k])
        # concat along channel dimension
        return torch.cat(ys, 1)

    def process_x(self, x):
        return self.cnn_layers(x)

    def encode(self, x, y):
        # shape: batch x channel x h x w
        # input x is after CNN
        # encode and flatten y
        y = self._collapse_y(y)
        x = self.posterior_fc(torch.cat([x, y], 1))  # concat along channel dim
        return self.posterior_mean(x), self.posterior_mean(x)

    def prior_network(self, x):
        # shape: batch x channel x h x w
        # input x is after CNN
        x = self.prior_fc(x)
        return self.prior_mean(x), self.prior_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x):
        # stem
        y_stem_on = self.out_stem_on(x)
        y_stem_loc_x = self.out_stem_loc_x(x)
        y_stem_loc_y = self.out_stem_loc_y(x)
        # y_stem_siz = self.out_stem_siz(x)
        hid_stem_siz = self.hid_stem_siz(x)
        y_stem_sm_siz = self.out_stem_sm_siz(hid_stem_siz)
        y_stem_sl_siz = self.out_stem_sl_siz(hid_stem_siz)

        # iloop
        y_iloop_on = self.out_iloop_on(x)
        y_iloop_loc_x = self.out_iloop_loc_x(x)
        y_iloop_loc_y = self.out_iloop_loc_y(x)
        # y_iloop_siz_x = self.out_iloop_siz_x(x)
        # y_iloop_siz_y = self.out_iloop_siz_y(x)
        hid_iloop_siz_x = self.hid_iloop_siz_x(x)
        y_iloop_sm_siz_x = self.out_iloop_sm_siz_x(hid_iloop_siz_x)
        y_iloop_sl_siz_x = self.out_iloop_sl_siz_x(hid_iloop_siz_x)
        hid_iloop_siz_y = self.hid_iloop_siz_y(x)
        y_iloop_sm_siz_y = self.out_iloop_sm_siz_y(hid_iloop_siz_y)
        y_iloop_sl_siz_y = self.out_iloop_sl_siz_y(hid_iloop_siz_y)

        # hloop
        y_hloop_on = self.out_hloop_on(x)
        y_hloop_loc_x = self.out_hloop_loc_x(x)
        y_hloop_loc_y = self.out_hloop_loc_y(x)
        # y_hloop_siz = self.out_hloop_siz(x)
        hid_hloop_siz = self.hid_hloop_siz(x)
        y_hloop_sm_siz = self.out_hloop_sm_siz(hid_hloop_siz)
        y_hloop_sl_siz = self.out_hloop_sl_siz(hid_hloop_siz)

        # collect
        y = {
            # stem
            'stem_on': y_stem_on,
            'stem_location_x': y_stem_loc_x,
            'stem_location_y': y_stem_loc_y,
            # 'stem_size': y_stem_siz,
            'stem_sm_size': y_stem_sm_siz,
            'stem_sl_size': y_stem_sl_siz,

            # iloop
            'iloop_on': y_iloop_on,
            'iloop_location_x': y_iloop_loc_x,
            'iloop_location_y': y_iloop_loc_y,
            # 'iloop_size_x': y_iloop_siz_x,
            # 'iloop_size_y': y_iloop_siz_y,
            'iloop_sm_size_x': y_iloop_sm_siz_x,
            'iloop_sl_size_x': y_iloop_sl_siz_x,
            'iloop_sm_size_y': y_iloop_sm_siz_y,
            'iloop_sl_size_y': y_iloop_sl_siz_y,

            # hloop
            'hloop_on': y_hloop_on,
            'hloop_location_x': y_hloop_loc_x,
            'hloop_location_y': y_hloop_loc_y,
            # 'hloop_size': y_hloop_siz,
            'hloop_sm_size': y_hloop_sm_siz,
            'hloop_sl_size': y_hloop_sl_siz,
        }

        return y

    # Defining the forward pass
    def forward(self, x, y):
        x = self.process_x(x)
        # prior
        mu_p, logvar_p = self.prior_network(x)
        # posterior
        mu_q, logvar_q = self.encode(x, y)
        # sample z
        z = self.reparameterize(mu_q, logvar_q)
        # z has shape batch x latent_dim x h x w  TODO right now each pixel has different z's...hmmm
        # concat with x
        assert z.shape[0] == x.shape[0]  # batch dim
        assert z.shape[2] == x.shape[2]
        assert z.shape[3] == x.shape[3]
        xz = torch.cat([x, z], dim=1)
        # decoder
        return self.decode(xz), mu_q, logvar_q, mu_p, logvar_p

    def inference(self, x):  # using prior network
        x = self.process_x(x)
        # prior
        mu_p, logvar_p = self.prior_network(x)
        # sample z
        z = self.reparameterize(mu_p, logvar_p)
        # z has shape batch x latent_dim x h x w  TODO right now each pixel has different z's...hmmm
        # concat with x
        assert z.shape[0] == x.shape[0]  # batch dim
        assert z.shape[2] == x.shape[2]
        assert z.shape[3] == x.shape[3]
        xz = torch.cat([x, z], dim=1)
        # decoder
        return self.decode(xz), mu_p, logvar_p


# TODO move to class level
loss_b = torch.nn.BCELoss(reduction='none')
loss_m = torch.nn.NLLLoss(reduction='none')
loss_e = torch.nn.MSELoss(reduction='none')


def masked_loss_e(x, y, m):
    # batch x channel? x h x w
    l = loss_e(x, y)
    n_valid_output = torch.sum(torch.sum(m, dim=3), dim=2)  # vector of length = batch
    loss_spatial_sum = torch.sum(torch.sum(torch.mul(l, m), dim=3), dim=2)
    n_valid_output[n_valid_output == 0] = 1
    loss_spatial_mean = loss_spatial_sum / n_valid_output
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    return torch.mean(loss_batch_mean)


def masked_loss_b(x, y, m):
    # batch x channel? x h x w
    l = loss_b(x, y)
    n_valid_output = torch.sum(torch.sum(m, dim=3), dim=2)  # vector of length = batch
    loss_spatial_sum = torch.sum(torch.sum(torch.mul(l, m), dim=3), dim=2)
    n_valid_output[n_valid_output == 0] = 1
    loss_spatial_mean = loss_spatial_sum / n_valid_output
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    return torch.mean(loss_batch_mean)


def masked_loss_m(x, y, m):
    # remove singleton dimension in target & mask since NLLLoss doesn't need it
    assert y.shape[1] == 1
    assert m.shape[1] == 1
    y = y[:, 0, :, :]
    m = m[:, 0, :, :]
    l = loss_m(x, y.long())  # FIXME should use long type in data gen (also need to fix padding <- not happy with long?)
    # batch x h x w
    n_valid_output = torch.sum(torch.sum(m, dim=2), dim=1)  # vector of length = batch
    loss_spatial_sum = torch.sum(torch.sum(torch.mul(l, m), dim=2), dim=1)
    n_valid_output[n_valid_output == 0] = 1
    loss_spatial_mean = loss_spatial_sum / n_valid_output
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    return torch.mean(loss_batch_mean)


def masked_loss(x, y, m, maskw):
    # x: pred
    # y: target
    # maskw: mask weight, [0, 1], 0: hard mask (masked position loss ->0), >0: soft mask: masked position loss * maskw
    # x, y, m are all dicts

    # stem    
    # stem on
    _x = x['stem_on']
    _y = y['stem_on']
    _m1 = m['stem_on']
    if maskw > 0:
        _m1[_m1 == 0] = maskw
    loss_stem_on = masked_loss_b(_x, _y, _m1)
    logging.info("loss_stem_on: {}".format(loss_stem_on))

    # stem location x & y
    _x = x['stem_location_x']
    _y = y['stem_location_x']
    _m2 = m['stem_location_size']
    loss_stem_loc_x = masked_loss_m(_x, _y, _m2)
    logging.info("loss_stem_loc_x: {}".format(loss_stem_loc_x))
    _x = x['stem_location_y']
    _y = y['stem_location_y']
    loss_stem_loc_y = masked_loss_m(_x, _y, _m2)
    logging.info("loss_stem_loc_y: {}".format(loss_stem_loc_y))

    # stem size, softmax
    _x = x['stem_sm_size']
    _y = y['stem_sm_size']
    loss_stem_sm_siz = masked_loss_m(_x, _y, _m2)
    logging.info("loss_stem_sm_siz: {}".format(loss_stem_sm_siz))

    # stem size, scalar
    _x = x['stem_sl_size']
    _y = y['stem_sl_size']
    loss_stem_sl_siz = masked_loss_e(_x, _y, _m2)
    logging.info("loss_stem_sl_siz: {}".format(loss_stem_sl_siz))

    # iloop  
    # iloop on
    _x = x['iloop_on']
    _y = y['iloop_on']
    _m1 = m['iloop_on']
    if maskw > 0:
        _m1[_m1 == 0] = maskw
    loss_iloop_on = masked_loss_b(_x, _y, _m1)
    logging.info("loss_iloop_on: {}".format(loss_iloop_on))

    # iloop location x & y
    _x = x['iloop_location_x']
    _y = y['iloop_location_x']
    _m2 = m['iloop_location_size']
    loss_iloop_loc_x = masked_loss_m(_x, _y, _m2)
    logging.info("loss_iloop_loc_x: {}".format(loss_iloop_loc_x))
    _x = x['iloop_location_y']
    _y = y['iloop_location_y']
    loss_iloop_loc_y = masked_loss_m(_x, _y, _m2)
    logging.info("loss_iloop_loc_y: {}".format(loss_iloop_loc_y))

    # iloop size, softmax
    _x = x['iloop_sm_size_x']
    _y = y['iloop_sm_size_x']
    loss_iloop_sm_siz_x = masked_loss_m(_x, _y, _m2)
    logging.info("loss_iloop_sm_siz_x: {}".format(loss_iloop_sm_siz_x))
    _x = x['iloop_sm_size_y']
    _y = y['iloop_sm_size_y']
    loss_iloop_sm_siz_y = masked_loss_m(_x, _y, _m2)
    logging.info("loss_iloop_sm_siz_y: {}".format(loss_iloop_sm_siz_y))

    # iloop size, scalar
    _x = x['iloop_sl_size_x']
    _y = y['iloop_sl_size_x']
    loss_iloop_sl_siz_x = masked_loss_e(_x, _y, _m2)
    logging.info("loss_iloop_sl_siz_x: {}".format(loss_iloop_sl_siz_x))
    _x = x['iloop_sl_size_y']
    _y = y['iloop_sl_size_y']
    loss_iloop_sl_siz_y = masked_loss_e(_x, _y, _m2)
    logging.info("loss_iloop_sl_siz_y: {}".format(loss_iloop_sl_siz_y))
    
    # hloop
    # hloop on
    _x = x['hloop_on']
    _y = y['hloop_on']
    _m1 = m['hloop_on']
    if maskw > 0:
        _m1[_m1 == 0] = maskw
    loss_hloop_on = masked_loss_b(_x, _y, _m1)
    logging.info("loss_hloop_on: {}".format(loss_hloop_on))

    # hloop location x & y
    _x = x['hloop_location_x']
    _y = y['hloop_location_x']
    _m2 = m['hloop_location_size']
    loss_hloop_loc_x = masked_loss_m(_x, _y, _m2)
    logging.info("loss_hloop_loc_x: {}".format(loss_hloop_loc_x))
    _x = x['hloop_location_y']
    _y = y['hloop_location_y']
    loss_hloop_loc_y = masked_loss_m(_x, _y, _m2)
    logging.info("loss_hloop_loc_y: {}".format(loss_hloop_loc_y))

    # hloop size, softmax
    _x = x['hloop_sm_size']
    _y = y['hloop_sm_size']
    loss_hloop_sm_siz = masked_loss_m(_x, _y, _m2)
    logging.info("loss_hloop_sm_siz: {}".format(loss_hloop_sm_siz))
    
    # hloop size, scalar
    _x = x['hloop_sl_size']
    _y = y['hloop_sl_size']
    loss_hloop_sl_siz = masked_loss_e(_x, _y, _m2)
    logging.info("loss_hloop_sl_siz: {}".format(loss_hloop_sl_siz))

    # TODO scale down the MSE portion?
    return loss_stem_on + loss_stem_loc_x + loss_stem_loc_y + loss_stem_sm_siz + loss_stem_sl_siz + \
           loss_iloop_on + loss_iloop_loc_x + loss_iloop_loc_y + \
           loss_iloop_sm_siz_x + loss_iloop_sm_siz_y + loss_iloop_sl_siz_x + loss_iloop_sl_siz_y + \
           loss_hloop_on + loss_hloop_loc_x + loss_hloop_loc_y + loss_hloop_sm_siz + loss_hloop_sl_siz

    # return loss_stem_on + loss_stem_loc_x + loss_stem_loc_y + loss_stem_siz + \
    #        loss_iloop_on + loss_iloop_loc_x + loss_iloop_loc_y + loss_iloop_siz_x + loss_iloop_siz_y + \
    #        loss_hloop_on + loss_hloop_loc_x + loss_hloop_loc_y + loss_hloop_siz


def old(x, y, m):   # FIXME remove?
    # print(x.shape, y.shape, m.shape)
    l = torch.nn.BCELoss(reduction='none')(x, y)
    # number of valid entries = 1's in the mask
    # TODO later on we might have different mask per channel - currently implementation assumes same mask across all channels
    assert m.shape[1] == y.shape[1]
    n_valid_output = torch.sum(torch.sum(m, dim=3), dim=2)  # vector of length = batch
    # average over spatial dimension is achieved by summing then dividing by the above
    # (need to do this since we're padding + masking, can't naively do mean)
    # note that tensor shapes: batch x channel x H x W
    loss_spatial_sum = torch.sum(torch.sum(torch.mul(l, m), dim=3), dim=2)
    # for cases where all elements are masked, the corresponding element in n_valid_output will be 0
    # in such case, since the final numerator will be 0, wlog, we'll set the denominator to 1 (if leaving as 0 will result in NaN)
    n_valid_output[n_valid_output == 0] = 1
    loss_spatial_mean = loss_spatial_sum/n_valid_output  # element-wise division
    # average over batch, this is the per-channel loss
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    logging.info(loss_batch_mean)
    # average over channels
    return torch.mean(loss_batch_mean)


def kl_loss(mu_q, logvar_q, mu_p, logvar_p, m):
    # KL divergence in closed form, with mask
    # batch x ch x h x w
    n_valid_output = torch.sum(torch.sum(m, dim=3), dim=2)  # vector of length = batch
    # multivariate Gaussian KL divergence
    kl_spatial_sum = 0.5 * torch.sum(torch.sum(logvar_q.exp()/logvar_p.exp() + \
                          (mu_p - mu_q).pow(2)/logvar_p.exp() - 1 + \
                          logvar_p - logvar_q, dim=3), dim=2)

    # kl_spatial_sum = -0.5 * torch.sum(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=3), dim=2)
    n_valid_output[n_valid_output == 0] = 1
    loss_spatial_mean = kl_spatial_sum / n_valid_output
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    # sum over latent dimension
    loss_sum = torch.sum(loss_batch_mean)
    logging.info("kl loss: {}".format(loss_sum))
    return loss_sum


# def to_device(x, y, m, device):
#     return x.to(device), y.to(device), m.to(device)


def to_device(x, y, m, device):
    return x.to(device), {k: v.to(device) for k, v in y.items()}, {k: v.to(device) for k, v in m.items()}


def is_seq_valid(seq):
    seq = seq.upper()
    if set(seq).issubset(set('ACGTUN')):
        return True
    else:
        return False


class EvalMetric(object):

    def __init__(self):
        # hard-coded 2-level dict
        # output -> metric -> list
        self.metrics = {
            # stem
            'stem_on': {
                'auroc': [],
                'auprc': [],
            },
            'stem_location_x': {
                'accuracy': [],
            },
            'stem_location_y': {
                'accuracy': [],
            },
            # 'stem_size': {
            #     'accuracy': [],
            # },
            'stem_sm_size': {
                'accuracy': [],
            },
            'stem_sl_size': {
                'diff': [],
            },
            
            # iloop
            'iloop_on': {
                'auroc': [],
                'auprc': [],
            },
            'iloop_location_x': {
                'accuracy': [],
            },
            'iloop_location_y': {
                'accuracy': [],
            },
            # 'iloop_size_x': {
            #     'accuracy': [],
            # },
            # 'iloop_size_y': {
            #     'accuracy': [],
            # },
            'iloop_sm_size_x': {
                'accuracy': [],
            },
            'iloop_sm_size_y': {
                'accuracy': [],
            },
            'iloop_sl_size_x': {
                'diff': [],
            },
            'iloop_sl_size_y': {
                'diff': [],
            },
            
            # hloop
            'hloop_on': {
                'auroc': [],
                'auprc': [],
            },
            'hloop_location_x': {
                'accuracy': [],
            },
            'hloop_location_y': {
                'accuracy': [],
            },
            # 'hloop_size': {
            #     'accuracy': [],
            # },
            'hloop_sm_size': {
                'accuracy': [],
            },
            'hloop_sl_size': {
                'diff': [],
            },
        }

    def add_val(self, output, metric, val):
        assert output in self.metrics.keys()
        assert metric in self.metrics[output].keys(), "output {} metric {}".format(output, metric)
        self.metrics[output][metric].append(val)

    def merge(self, m):
        # merge inner list
        assert isinstance(m, EvalMetric)
        assert self.metrics.keys() == m.metrics.keys()
        for k1 in self.metrics.keys():
            assert self.metrics[k1].keys() == m.metrics[k1].keys()
            for k2 in self.metrics[k1].keys():
                assert isinstance(self.metrics[k1][k2], list)
                assert isinstance(m.metrics[k1][k2], list)
                self.metrics[k1][k2].extend(m.metrics[k1][k2])

    def aggregate(self, method=np.nanmean):
        x = self.metrics.copy()
        for k1 in x.keys():
            for k2 in x[k1].keys():
                x[k1][k2] = method(x[k1][k2])
        return x


def compute_metrics(x, y, m):
    # x, y, m are all dictionaries
    # x : true label, y: pred, m: binary mask, where 1 indicate valid

    def _roc_prc(key_target, key_mask):
        _x = x[key_target][idx_batch, 0, :, :]
        _y = y[key_target][idx_batch, 0, :, :]
        _m = m[key_mask][idx_batch, 0, :, :]
        mask_bool = _m.eq(1)
        _x2 = _x.masked_select(mask_bool).flatten().detach().cpu().numpy()
        _y2 = _y.masked_select(mask_bool).flatten().detach().cpu().numpy()
        # do not compute if empty (e.g. when all elements are being masked)
        # do not compute if there's only one class
        if len(_x2) > 0 and not np.all(_x2 == _x2[0]):
            roc = roc_auc_score(_x2, _y2)
            prc = average_precision_score(_x2, _y2)
        else:
            roc = np.NaN
            prc = np.NaN
        return roc, prc

    def _accuracy(key_target, key_mask):
        # for convenience
        _x = x[key_target][idx_batch, 0, :, :]
        _y = y[key_target][idx_batch, :, :, :].argmax(axis=0)
        _m = m[key_mask][idx_batch, 0, :, :]
        mask_bool = _m.eq(1)
        _x2 = _x.masked_select(mask_bool).flatten().detach().cpu().numpy()
        _y2 = _y.masked_select(mask_bool).flatten().detach().cpu().numpy()
        if len(_x2) == 0:
            return np.NaN
        else:
            return np.sum(_x2 == _y2)/float(len(_x2))

    def _diff(key_target, key_mask):
        _x = x[key_target][idx_batch, 0, :, :]
        _y = y[key_target][idx_batch, 0, :, :]
        _m = m[key_mask][idx_batch, 0, :, :]
        mask_bool = _m.eq(1)
        _x2 = _x.masked_select(mask_bool).flatten().detach().cpu().numpy()
        _y2 = _y.masked_select(mask_bool).flatten().detach().cpu().numpy()
        if len(_x2) == 0:
            return np.NaN
        else:
            return np.sum(np.abs(_x2 - _y2))/ float(len(_x2))

    evalm = EvalMetric()

    num_examples = y[list(y.keys())[0]].shape[0]  # wlog, check batch dimension using first output key

    for idx_batch in range(num_examples):
        # use mask to select non-zero entries
        
        # stem
        # stem on
        roc, prc = _roc_prc(key_target='stem_on', key_mask='stem_on')
        evalm.add_val(output='stem_on', metric='auroc', val=roc)
        evalm.add_val(output='stem_on', metric='auprc', val=prc)
        # stem location x & y
        evalm.add_val(output='stem_location_x', metric='accuracy',
                      val=_accuracy(key_target='stem_location_x', key_mask='stem_location_size'))
        evalm.add_val(output='stem_location_y', metric='accuracy',
                      val=_accuracy(key_target='stem_location_y', key_mask='stem_location_size'))
        # stem size, softmax
        evalm.add_val(output='stem_sm_size', metric='accuracy',
                      val=_accuracy(key_target='stem_sm_size', key_mask='stem_location_size'))
        # stem size, scalar
        evalm.add_val(output='stem_sl_size', metric='diff',
                      val=_diff(key_target='stem_sl_size', key_mask='stem_location_size'))
        
        # iloop
        # iloop on
        roc, prc = _roc_prc(key_target='iloop_on', key_mask='iloop_on')
        evalm.add_val(output='iloop_on', metric='auroc', val=roc)
        evalm.add_val(output='iloop_on', metric='auprc', val=prc)
        # iloop location x & y
        evalm.add_val(output='iloop_location_x', metric='accuracy',
                      val=_accuracy(key_target='iloop_location_x', key_mask='iloop_location_size'))
        evalm.add_val(output='iloop_location_y', metric='accuracy',
                      val=_accuracy(key_target='iloop_location_y', key_mask='iloop_location_size'))
        # iloop size x, softmax
        evalm.add_val(output='iloop_sm_size_x', metric='accuracy',
                      val=_accuracy(key_target='iloop_sm_size_x', key_mask='iloop_location_size'))
        # iloop size y, softmax
        evalm.add_val(output='iloop_sm_size_y', metric='accuracy',
                      val=_accuracy(key_target='iloop_sm_size_y', key_mask='iloop_location_size'))
        # iloop size x, scalar
        evalm.add_val(output='iloop_sl_size_x', metric='diff',
                      val=_diff(key_target='iloop_sl_size_x', key_mask='iloop_location_size'))
        # iloop size y, scalar
        evalm.add_val(output='iloop_sl_size_y', metric='diff',
                      val=_diff(key_target='iloop_sl_size_y', key_mask='iloop_location_size'))
        
        # hloop
        # stem on
        roc, prc = _roc_prc(key_target='hloop_on', key_mask='hloop_on')
        evalm.add_val(output='hloop_on', metric='auroc', val=roc)
        evalm.add_val(output='hloop_on', metric='auprc', val=prc)
        # hloop location x & y
        evalm.add_val(output='hloop_location_x', metric='accuracy',
                      val=_accuracy(key_target='hloop_location_x', key_mask='hloop_location_size'))
        evalm.add_val(output='hloop_location_y', metric='accuracy',
                      val=_accuracy(key_target='hloop_location_y', key_mask='hloop_location_size'))
        # hloop size, softmax
        evalm.add_val(output='hloop_sm_size', metric='accuracy',
                      val=_accuracy(key_target='hloop_sm_size', key_mask='hloop_location_size'))
        # hloop size, softmax
        evalm.add_val(output='hloop_sl_size', metric='diff',
                      val=_diff(key_target='hloop_sl_size', key_mask='hloop_location_size'))

    return evalm


def main(path_data, num_filters, filter_width, dropout, maskw, latent_dim, n_epoch, batch_size, max_length, out_dir, n_cpu):
    logging.info("Loading dataset: {}".format(path_data))
    dc_client = dc.Client()
    df = []
    for _p in path_data:
        if os.path.isfile(_p):
            df.append(pd.read_pickle(_p, compression='gzip'))
        else:
            print(dc_client.get_path(_p))
            df.append(pd.read_pickle(dc_client.get_path(_p), compression='gzip'))
    df = pd.concat(df)
    # subset to max length if specified
    if max_length:
        logging.info("Subsetting to max length, n_rows before: {}".format(len(df)))
        if 'len' not in df.columns:
            df = add_column(df, 'len', ['seq'], len)
            df = df[df['len'] <= max_length]
            df = df.drop(columns=['len'])
        else:
            df = df[df['len'] <= max_length]
        logging.info("After: {}".format(len(df)))
    # subset to sequence with valid nucleotides ACGTN
    df = add_column(df, 'is_seq_valid', ['seq'], is_seq_valid)
    n_invalid = (~df['is_seq_valid']).sum()
    logging.info("Subsetting to sequence with valid bases, n_rows before: {}".format(len(df)))
    logging.info("Dropping {} rows".format(n_invalid))
    df = df[df['is_seq_valid']]
    logging.info("After: {}".format(len(df)))
    df.drop(columns=['is_seq_valid'], inplace=True)

    # split into training+validation
    # shuffle rows
    df = df.sample(frac=1).reset_index(drop=True)
    # subset
    # tr_prop = 0.95
    tr_prop = 0.8
    _n_tr = int(len(df) * tr_prop)
    logging.info("Using {} data for training and {} for validation".format(_n_tr, len(df) - _n_tr))
    df_tr = df[:_n_tr]
    df_va = df[_n_tr:]
    # length group + chain dataset, to ensure that sequences in the same minibatch are of similar length
    n_groups = 20
    # data loaders
    data_loader_tr = DataLoader(torch.utils.data.ConcatDataset([MyDataSet(x) for x in length_grouping(df_tr, n_groups)]),
                                batch_size=batch_size,
                                shuffle=True, num_workers=n_cpu,
                                collate_fn=PadCollate2D())
    data_loader_va = DataLoader(torch.utils.data.ConcatDataset([MyDataSet(x) for x in length_grouping(df_va, n_groups)]),
                                batch_size=batch_size,
                                shuffle=True, num_workers=n_cpu,
                                collate_fn=PadCollate2D())

    # model
    model = LatentVarModel(num_filters=num_filters,
                           num_output=123,   # hard-coded
                           filter_width=filter_width,
                           latent_dim=latent_dim,
                           dropout=dropout)
    print(model)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: {}".format(device))

    model = model.to(device)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    # # naive guess is the mean of training target value
    # yp_naive = torch.mean(torch.stack([torch.mean(y) for _, y, _ in data_loader_tr]))
    # logging.info("Naive guess: {}".format(yp_naive))
    # # calculate loss using naive guess
    # logging.info("Naive guess performance")
    #
    # with torch.set_grad_enabled(False):
    #     # training
    #     loss_naive_tr = []
    #     auroc_naive_tr = []
    #     auprc_naive_tr = []
    #     for x, y, m in data_loader_tr:
    #         x, y, m = to_device(x, y, m, device)
    #         yp = torch.ones_like(y) * yp_naive
    #         # loss_naive_tr.append(masked_loss(yp, y, m).detach().cpu().numpy())
    #         loss_naive_tr.append(masked_loss(yp, y, m).item())
    #         _r, _p = compute_metrics(y, yp, m)
    #         auroc_naive_tr.extend(_r)
    #         auprc_naive_tr.extend(_p)
    #     logging.info("Training: loss {} au-ROC {} au-PRC {}".format(np.mean(np.stack(loss_naive_tr)),
    #                                                                 np.mean(np.stack(auroc_naive_tr)),
    #                                                                 np.mean(np.stack(auprc_naive_tr))))
    #     # validation
    #     loss_naive_va = []
    #     auroc_naive_va = []
    #     auprc_naive_va = []
    #     for x, y, m in data_loader_va:
    #         x, y, m = to_device(x, y, m, device)
    #         yp = torch.ones_like(y) * yp_naive
    #         # loss_naive_va.append(masked_loss(yp, y, m).detach().cpu().numpy())
    #         loss_naive_va.append(masked_loss(yp, y, m).item())
    #         _r, _p = compute_metrics(y, yp, m)
    #         auroc_naive_va.extend(_r)
    #         auprc_naive_va.extend(_p)
    #     logging.info("Validation: loss {} au-ROC {} au-PRC {}".format(np.mean(np.stack(loss_naive_va)),
    #                                                                   np.mean(np.stack(auroc_naive_va)),
    #                                                                   np.mean(np.stack(auprc_naive_va))))

    for epoch in range(n_epoch):
        running_loss_tr = []
        running_auroc_tr = []
        running_auprc_tr = []
        evalm_tr = EvalMetric()
        for x, y, m, md in data_loader_tr:

            x, y, m = to_device(x, y, m, device)
            # yp, mu, logvar = model(x)
            yp, mu_q, logvar_q, mu_p, logvar_p = model(x, y)

            loss_1 = masked_loss(yp, y, m, maskw)  # order: pred, target, mask, mask_weight
            loss_2 = kl_loss(mu_q, logvar_q, mu_p, logvar_p, m['stem_on'])  # w.o.l.g. use one of the hard masks


            # add in prior-based loss (TODO weight ths loss?)
            yp, mu_p, logvar_p = model.inference(x)
            loss_3 = masked_loss(yp, y, m, maskw)

            loss = loss_1 + loss_2 + loss_3
            # running_loss_tr.append(loss.detach().cpu().numpy())
            running_loss_tr.append(loss.item())

            # TODO return val
            evalm_tr.merge(compute_metrics(y, yp, m))


            # running_auroc_tr.extend(_r)
            # running_auprc_tr.extend(_p)
            logging.info("Epoch {} Training loss: {} ({} + {} + {})".format(epoch, loss, loss_1, loss_2, loss_3))

            model.zero_grad()

            loss.backward()
            optimizer.step()
        
        # report metric
        # logging.info(pprint.pformat(evalm_tr.aggregate(method=np.mean), indent=4))
        logging.info(evalm_tr.aggregate(method=np.nanmean))

        # save model
        _model_path = os.path.join(out_dir, 'model_ckpt_ep_{}.pth'.format(epoch))
        torch.save(model.state_dict(), _model_path)
        logging.info("Model checkpoint saved at: {}".format(_model_path))

        # # save the last minibatch prediction
        # df_pred = []
        # num_examples = y[list(y.keys())[0]].shape[0]   # wlog, check batch dimension using first output key
        # for i in range(num_examples):
        #     row = {'subset': 'training'}
        #     # store metadata
        #     row.update(md[i])
        #     for k in y.keys():
        #         #  batch x channel x H x W
        #         _y = y[k][i, :, :, :].detach().cpu().numpy()
        #         _yp = yp[k][i, :, :, :].detach().cpu().numpy()
        #         row.update({'target_{}'.format(k): _y,
        #                     'pred_{}'.format(k): _yp,
        #                     })
        #     df_pred.append(row)

        # # report training loss
        # logging.info(
        #     "Epoch {}/{}, training loss (running) {}, au-ROC {}, au-PRC {}".format(epoch, n_epoch,
        #                                                                            np.mean(
        #                                                                                np.stack(running_loss_tr)),
        #                                                                            np.mean(np.stack(running_auroc_tr)),
        #                                                                            np.mean(np.stack(running_auprc_tr))))
        logging.info(
            "Epoch {}/{}, training loss (running) {}".format(epoch, n_epoch,np.mean(np.stack(running_loss_tr))))

        with torch.set_grad_enabled(False):
            # report validation loss
            running_loss_va = []
            running_auroc_va = []
            running_auprc_va = []
            evalm_tr = EvalMetric()
            for x, y, m, md in data_loader_va:
                x, y, m = to_device(x, y, m, device)
                # yp, mu, logvar = model(x)
                yp, mu_q, logvar_q, mu_p, logvar_p = model(x, y)
                loss_1 = masked_loss(yp, y, m, maskw)
                loss_2 = kl_loss(mu_q, logvar_q, mu_p, logvar_p, m['stem_on'])  # w.o.l.g. use one of the hard masks
                loss = loss_1 + loss_2
                # running_loss_va.append(loss.detach().cpu().numpy())
                running_loss_va.append(loss.item())
                logging.info("Epoch {} Validation loss (posterior): {} ({} + {})".format(epoch, loss, loss_1, loss_2))

                evalm_tr.merge(compute_metrics(y, yp, m))
                # _r, _p = compute_metrics(y, yp, m)
                # running_auroc_va.extend(_r)
                # running_auprc_va.extend(_p)
            logging.info(evalm_tr.aggregate(method=np.nanmean))
            # logging.info(
            #     "Epoch {}/{}, validation loss {}, au-ROC {}, au-PRC {}".format(epoch, n_epoch,
            #                                                                    np.mean(np.stack(running_loss_va)),
            #                                                                    np.mean(np.stack(running_auroc_va)),
            #                                                                    np.mean(np.stack(running_auprc_va))))
            logging.info(
                "Epoch {}/{}, validation loss {}".format(epoch, n_epoch, np.mean(np.stack(running_loss_va))))

            # report validation loss using z sampled from prior network
            running_loss_va = []
            running_auroc_va = []
            running_auprc_va = []
            evalm_tr = EvalMetric()
            for x, y, m, md in data_loader_va:
                x, y, m = to_device(x, y, m, device)
                # yp, mu, logvar = model(x)
                yp, mu_p, logvar_p = model.inference(x)
                loss_1 = masked_loss(yp, y, m, maskw)
                # loss_2 = kl_loss(mu_q, logvar_q, mu_p, logvar_p, m['stem_on'])  # w.o.l.g. use one of the hard masks
                # loss = loss_1 + loss_2
                running_loss_va.append(loss_1.item())
                logging.info("Epoch {} Validation loss (prior, p(y|x) only): {}".format(epoch, loss_1))

                evalm_tr.merge(compute_metrics(y, yp, m))
            logging.info(evalm_tr.aggregate(method=np.nanmean))
            logging.info(
                "Epoch {}/{}, validation loss {}".format(epoch, n_epoch, np.mean(np.stack(running_loss_va))))


            # # save the last minibatch prediction
            # num_examples = y[list(y.keys())[0]].shape[0]  # wlog, check batch dimension using first output key
            # for i in range(num_examples):
            #     row = {'subset': 'validation'}
            #     # store metadata
            #     row.update(md[i])
            #     for k in y.keys():
            #         #  batch x channel x H x W
            #         _y = y[k][i, :, :, :].detach().cpu().numpy()
            #         _yp = yp[k][i, :, :, :].detach().cpu().numpy()
            #         row.update({'target_{}'.format(k): _y,
            #                     'pred_{}'.format(k): _yp,
            #                     })
            #     df_pred.append(row)

        # end of epoch
        # export prediction
        # out_file = os.path.join(out_dir, 'pred_ep_{}.pkl.gz'.format(epoch))
        # df_pred = pd.DataFrame(df_pred)
        # df_pred.to_pickle(out_file, compression='gzip')
        # logging.info("Exported prediction (one minibatch) to: {}".format(out_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=str, help='Path or DC ID to training data file, should be in pkl.gz format')
    parser.add_argument('--result', type=str, help='Path to output result')
    parser.add_argument('--num_filters', nargs='*', type=int, help='Number of conv filters for each layer.')
    parser.add_argument('--filter_width', nargs='*', type=int, help='Filter width for each layer.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--mask', type=float, default=0.0, help='Mask weight. Setting to 0 is equivalent to hard mask.')
    parser.add_argument('--latent_dim', type=int, help='Number of latent variables')
    parser.add_argument('--epoch', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Mini batch size')
    parser.add_argument('--max_length', type=int, default=0,
                        help='Max sequence length to train on. This is used to subset the dataaset.')
    parser.add_argument('--cpu', type=int, help='Number of CPU workers per data loader')
    args = parser.parse_args()

    # some basic logging
    set_up_logging(args.result)
    logging.debug("Cmd: {}".format(args))  # cmd args
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    logging.debug("Current dir: {}, git hash: {}".format(cur_dir, git_hash))
    # training
    assert 0 <= args.dropout <= 1
    assert 0 <= args.mask <= 1
    main(args.data, args.num_filters, args.filter_width, args.dropout, args.mask, args.latent_dim, args.epoch, args.batch_size, args.max_length, args.result,
         args.cpu)
