import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import norm
import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, '../../rna_ss/')
from utils import db2pairs
from local_struct_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datacorral as dc


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
            cnn_layers.append(nn.Conv2d(num_filters[i], nf, kernel_size=fw, stride=1, padding=fw // 2))
            cnn_layers.append(nn.BatchNorm2d(nf))
            cnn_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                cnn_layers.append(nn.Dropout(dropout))
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # self.fc = nn.Conv2d(num_filters[-1], 5, kernel_size=1)
        # self.fc = nn.Conv2d(num_filters[-1], 50, kernel_size=1)

        # "encoder" of the cvae, x + y -> z's param
        self.fc = nn.Sequential(
            nn.Conv2d(num_filters[-1] + num_output, 50, kernel_size=1),
            nn.ReLU(),
        )
        # self.fc = nn.Sequential(
        #     nn.Conv2d(num_filters[-1], 50, kernel_size=1),
        #     nn.ReLU(),
        # )

        # posterior mean and logvar
        # latent variable layer
        self.latent_mean = nn.Conv2d(50, latent_dim, kernel_size=1)
        self.latent_logvar = nn.Conv2d(50, latent_dim, kernel_size=1)

        # TODO add prior network

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
        ys = []
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
        x = self.fc(torch.cat([x, y], 1))  # concat along channel dim
        return self.latent_mean(x), self.latent_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
        # posterior
        mu, logvar = self.encode(x, y)
        # sample z
        z = self.reparameterize(mu, logvar)
        # z has shape batch x latent_dim x h x w  TODO right now each pixel has different z's...hmmm
        # concat with x
        assert z.shape[0] == x.shape[0]  # batch dim
        assert z.shape[2] == x.shape[2]
        assert z.shape[3] == x.shape[3]
        xz = torch.cat([x, z], dim=1)
        # decoder
        return self.decode(xz), mu, logvar


class SeqPairEncoder(object):
    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    def __init__(self, x1, x2):
        # x1 & x2: sequence, for now require to be same length (caller should pad with N if needed)
        assert len(x2) == len(x2), "For now require two seqs of same length. Please pad with N."
        x1 = x1.upper().replace('U', 'T')
        x2 = x2.upper().replace('U', 'T')
        assert set(x1).issubset(set(list('ACGTN')))
        assert set(x2).issubset(set(list('ACGTN')))
        self.x1 = x1
        self.x2 = x2
        # encode
        self.x1_1d = self.encode_seq(self.x1)
        self.x2_1d = self.encode_seq(self.x2)
        self.x_2d = self.encode_x(self.x1_1d, self.x2_1d)
        self.x_torch = self.encode_torch_input(self.x_2d)

    def encode_seq(self, x):
        seq = x.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        x = np.asarray([int(x) for x in list(seq)])
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def encode_x(self, x1, x2):
        # outer product
        assert len(x1.shape) == 2
        assert x1.shape[1] == 4
        assert len(x2.shape) == 2
        assert x2.shape[1] == 4
        l1 = x1.shape[0]
        l2 = x2.shape[0]
        # print(l1, l2)
        x1 = x1[:, np.newaxis, :]
        x2 = x2[np.newaxis, :, :]
        # print(x1.shape, x2.shape)
        x1 = np.repeat(x1, l2, axis=1)
        x2 = np.repeat(x2, l1, axis=0)
        # print(x1.shape, x2.shape)
        return np.concatenate([x1, x2], axis=2)

    def encode_torch_input(self, x):
        # add batch dim
        assert len(x.shape) == 3
        x = x[np.newaxis, :, :, :]
        # convert to torch tensor
        x = torch.from_numpy(x).float()
        # reshape: batch x channel x H x W
        x = x.permute(0, 3, 1, 2)
        return x


class DataEncoder(object):
    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    def __init__(self, x, y=None, bb_ref='top_right'):
        # x: sequence
        # y (optional): one_idx (two lists), zero-based
        x = x.upper().replace('U', 'T')
        assert set(x).issubset(set(list('ACGTN')))
        self.x = x
        if y:
            assert len(y) == 2
            assert len(y[0]) == len(y[1])
            assert max(y[0]) < len(x)
            assert max(y[1]) < len(x)
            self.y = y
        assert bb_ref in ['top_left', 'top_right']
        self.bb_ref = bb_ref
        # encode
        self.x_1d = self.encode_seq(self.x)
        self.x_2d = self.encode_x(self.x_1d)
        self.x_torch = self.encode_torch_input(self.x_2d)
        if y:
            self.y_bb, self.y_arrs = self.encode_y(self.y, len(self.x))


    def encode_seq(self, x):
        seq = x.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        x = np.asarray([int(x) for x in list(seq)])
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def encode_x(self, x):
        # outer product
        # tile and stack
        assert len(x.shape) == 2
        assert x.shape[1] == 4
        l = x.shape[0]
        x1 = x[:, np.newaxis, :]
        x2 = x[np.newaxis, :, :]
        x1 = np.repeat(x1, l, axis=1)
        x2 = np.repeat(x2, l, axis=0)
        return np.concatenate([x1, x2], axis=2)

    def encode_torch_input(self, x):
        # add batch dim
        assert len(x.shape) == 3
        x = x[np.newaxis, :, :, :]
        # convert to torch tensor
        x = torch.from_numpy(x).float()
        # reshape: batch x channel x H x W
        x = x.permute(0, 3, 1, 2)
        return x

    def encode_y(self, y, l):
        pairs, structure_arr = one_idx2arr(y, l, remove_lower_triangular=True)
        parser = LocalStructureParser(pairs)
        bounding_boxes = parser.parse_bounding_boxes()  # top left corner
        # extra data (needed for plot)
        target_stem_on, target_iloop_on, target_hloop_on, \
        mask_stem_on, mask_iloop_on, mask_hloop_on, \
        target_stem_location_x, target_stem_location_y, target_iloop_location_x, target_iloop_location_y, \
        target_hloop_location_x, target_hloop_location_y, \
        target_stem_size, target_iloop_size_x, target_iloop_size_y, target_hloop_size, \
        mask_stem_location_size, mask_iloop_location_size, \
        mask_hloop_location_size = make_target_pixel_bb(structure_arr, bounding_boxes)
        # only pick the useful ones
        # organize
        y = {
            # stem
            'stem_on': target_stem_on[:, :],
        #     'stem_location_x': torch.from_numpy(target_stem_location_x[:, :, np.newaxis]).float(),
        # # add singleton dimension (these are integer index of softmax index)
        #     'stem_location_y': torch.from_numpy(target_stem_location_y[:, :, np.newaxis]).float(),
        #     'stem_size': torch.from_numpy(target_stem_size[:, :, np.newaxis]).float(),

            # iloop
            'iloop_on': target_iloop_on,
            # 'iloop_location_x': torch.from_numpy(target_iloop_location_x[:, :, np.newaxis]).float(),
            # 'iloop_location_y': torch.from_numpy(target_hloop_location_y[:, :, np.newaxis]).float(),
            # 'iloop_size_x': torch.from_numpy(target_iloop_size_x[:, :, np.newaxis]).float(),
            # 'iloop_size_y': torch.from_numpy(target_iloop_size_y[:, :, np.newaxis]).float(),

            # hloop
            'hloop_on': target_hloop_on,
            # 'hloop_location_x': torch.from_numpy(target_hloop_location_x[:, :, np.newaxis]).float(),
            # 'hloop_location_y': torch.from_numpy(target_hloop_location_y[:, :, np.newaxis]).float(),
            # 'hloop_size': torch.from_numpy(target_hloop_size[:, :, np.newaxis]).float(),
        }
        if self.bb_ref == 'top_right':
            bbs = []
            for (x0, y0), (wx, wy), bb_name in bounding_boxes:
                bbs.append(((x0, y0 + wy - 1), (wx, wy), bb_name))
            return bbs, y
        else:
            return bounding_boxes, y


class Predictor(object):
    model_versions = {
        # # train on random sequence, after adding in scalar target for bb size, ep 11
        # # produced by: https://github.com/PSI-Lab/alice-sandbox/tree/f8df78da280b2a3ba16960a6226afaef2facd734/meetings/2021_01_05#s1-training
        # 'v1.0': 'KOE6Jb',
    }

    model_params = {
        # 'v1.0': {"num_filters": [32, 32, 64, 64, 64, 128, 128],
        #          "filter_width": [9, 9, 9, 9, 9, 9, 9],
        #          "dropout": 0.0},
    }

    def __init__(self, model_ckpt, num_filters=None, num_output=None, filter_width=None, latent_dim=None, dropout=None):
        # model_ckpt: model params checkpoint
        # can be any of the following:
        # version id
        # DC ID
        # path to the file

        # model file path
        dc_client = dc.Client()
        if model_ckpt in self.model_versions:
            assert model_ckpt in self.model_params
            model_file = dc_client.get_path(self.model_versions[model_ckpt])
            print("Loading know version {} with params {}".format(model_ckpt, self.model_params[model_ckpt]))
            num_filters = self.model_params[model_ckpt]['num_filters']
            filter_width = self.model_params[model_ckpt]['filter_width']
            dropout = self.model_params[model_ckpt]['dropout']
        elif os.path.isfile(model_ckpt):
            model_file = model_ckpt
        else:
            model_file = dc_client.get_path(model_ckpt)

        # default params
        if num_filters is None:
            num_filters = [32, 32, 64, 64, 64, 128, 128]
        if num_output is None:
            num_output = 123
        if filter_width is None:
            filter_width = [9, 9, 9, 9, 9, 9, 9]
        if latent_dim is None:
            latent_dim = 20
        if dropout is None:
            dropout = 0.0

        # needed for sampling z
        self.latent_dim = latent_dim

        # for computing trim_size
        self.filter_width = filter_width

        model = LatentVarModel(num_filters=num_filters, num_output=num_output,
                              filter_width=filter_width, latent_dim=latent_dim, dropout=dropout)
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        # set to be in inference mode
        model.eval()
        # TODO print model summary
        self.model = model

    @staticmethod
    def cleanup_hloop(bbs, l):
        # remove hloop that's not on diagonal
        bbs_new = []
        for bb in bbs:
            bb_x = bb['bb_x']
            bb_y = bb['bb_y']
            siz_x = bb['siz_x']
            siz_y = bb['siz_y']
            y0 = bb_y - siz_y + 1
            x1 = bb_x + siz_x - 1
            if bb_x == y0 and bb_y == x1:
                bbs_new.append(bb)
        return bbs_new

    # @staticmethod
    # def predict_bounidng_box(pred_on, pred_loc_x, pred_loc_y,
    #                          pred_sm_siz_x, pred_sm_siz_y,
    #                          pred_sl_siz_x, pred_sl_siz_y, thres=0.5, topk=1):
    #     if topk == 1:
    #         return Predictor.predict_bounidng_box_top_one(pred_on, pred_loc_x, pred_loc_y,
    #                                                       pred_sm_siz_x, pred_sm_siz_y,
    #                                                       pred_sl_siz_x, pred_sl_siz_y, thres)
    #     else:
    #         raise NotImplementedError

    @staticmethod
    def predict_bounding_box(pred_on, pred_loc_x, pred_loc_y,
                             pred_sm_siz_x, pred_sm_siz_y,
                             pred_sl_siz_x, pred_sl_siz_y, thres=0.5, topk=1, perc_cutoff=0):

        def _make_mask(l):
            m = np.ones((l, l))
            m[np.tril_indices(l)] = 0
            return m

        def _update(bb_x, bb_y, siz_x, siz_y, prob, proposed_boxes, pred_box, bb_source):
            assert bb_source in ['sm', 'sl']
            proposed_boxes.append({
                'bb_x': bb_x,
                'bb_y': bb_y,
                'siz_x': siz_x,
                'siz_y': siz_y,
                'prob_{}'.format(bb_source): prob,
            })
            # set value in pred box, be careful with out of bound index
            x0 = bb_x
            y0 = bb_y - siz_y + 1  # 0-based
            wx = siz_x
            wy = siz_y
            ix0 = max(0, x0)
            iy0 = max(0, y0)
            ix1 = min(x0 + wx, pred_box.shape[0])
            iy1 = min(y0 + wy, pred_box.shape[1])
            pred_box[ix0:ix1, iy0:iy1] = 1
            return proposed_boxes, pred_box

        def sm_top_one(p_on, pred_loc_x, pred_loc_y, pred_sm_siz_x, pred_sm_siz_y, i, j):
            loc_x = np.argmax(pred_loc_x[:, i, j])
            loc_y = np.argmax(pred_loc_y[:, i, j])
            # softmax size
            sm_siz_x = np.argmax(pred_sm_siz_x[:, i, j]) + 1  # size starts at 1 for index=0
            sm_siz_y = np.argmax(pred_sm_siz_y[:, i, j]) + 1
            # prob of on/off & location
            prob_1 = p_on * softmax(pred_loc_x[:, i, j])[loc_x] * softmax(pred_loc_y[:, i, j])[loc_y]
            # softmax: compute joint probability of taking the max value
            prob_sm = prob_1 *  softmax(pred_sm_siz_x[:, i, j])[sm_siz_x - 1] * softmax(pred_sm_siz_y[:, i, j])[
                       sm_siz_y - 1]  # FIXME multiplying twice for case where y is set to x
            bb_x = i - loc_x
            bb_y = j + loc_y
            # list of one tuple
            result = [(bb_x, bb_y, sm_siz_x, sm_siz_y, prob_sm)]
            return result

        def sm_top_k(p_on, pred_loc_x, pred_loc_y, pred_sm_siz_x, pred_sm_siz_y, i, j, k, cutoff=0):
            """take topk bbs,
            if cutoff is specified (i.e. !=0), only pick those whose prob >= cutoff * p_top"""
            assert k >= 2
            # outer product between 4 arrays: loc_x, loc_y, siz_x, siz_y
            loc_x = softmax(pred_loc_x[:, i, j])
            loc_y = softmax(pred_loc_y[:, i, j])
            siz_x = softmax(pred_sm_siz_x[:, i, j])
            siz_y = softmax(pred_sm_siz_y[:, i, j])
            joint_prob_all = p_on * loc_x[:, np.newaxis, np.newaxis, np.newaxis] * loc_y[np.newaxis, :, np.newaxis, np.newaxis] * siz_x[np.newaxis, np.newaxis, :, np.newaxis] * siz_y[np.newaxis, np.newaxis, np.newaxis, :]
            p_top = np.max(joint_prob_all)
            # sort along all axis (reverse idx so result is descending)
            idx_linear = np.argsort(joint_prob_all, axis=None)[::-1]
            # take top k
            idx_linear = idx_linear[:k]
            assert len(idx_linear)
            result = []
            # checks
            arr_shape = joint_prob_all.shape
            top_idx = np.unravel_index(idx_linear[0], arr_shape)
            # check whether it match the argmax, check value instead of index, in case there's a tie
            assert loc_x[top_idx[0]] == np.max(loc_x), "top_idx {} loc_x {}".format(top_idx, loc_x)
            assert loc_y[top_idx[1]] == np.max(loc_y), "top_idx {} loc_y {}".format(top_idx, loc_y)
            assert siz_x[top_idx[2]] == np.max(siz_x), "top_idx {} siz_x {}".format(top_idx, siz_x)
            assert siz_y[top_idx[3]] == np.max(siz_y), "top_idx {} loc_x {}".format(top_idx, siz_y)
            # assert top_idx[0] == np.argmax(loc_x)
            # assert top_idx[1] == np.argmax(loc_y)
            # assert top_idx[2] == np.argmax(siz_x)
            # assert top_idx[3] == np.argmax(siz_y)
            for _i in idx_linear:  # avoid clash with i
                idx = np.unravel_index(_i, arr_shape)  # this idx is 4-D
                # print(i, j, idx)
                bb_x = i - idx[0]
                bb_y = j + idx[1]
                sm_siz_x = idx[2] + 1
                sm_siz_y = idx[3] + 1
                prob_sm = joint_prob_all[idx]
                # if cutoff is specified, we also check the probability
                # break if it's lower than that (since loop is sorted descending)
                if cutoff > 0 and prob_sm < cutoff * p_top:
                    break
                result.append((bb_x, bb_y, sm_siz_x, sm_siz_y, prob_sm))
            return result

        def sm_top_perc(p_on, pred_loc_x, pred_loc_y, pred_sm_siz_x, pred_sm_siz_y, i, j, cutoff):
            """select those whose marginal probability >= cutoff*p_top,
            setting cutoff == 1 correspond to picking the argmax"""
            assert 0 < cutoff <= 1
            # outer product between 4 arrays: loc_x, loc_y, siz_x, siz_y
            loc_x = softmax(pred_loc_x[:, i, j])
            loc_y = softmax(pred_loc_y[:, i, j])
            siz_x = softmax(pred_sm_siz_x[:, i, j])
            siz_y = softmax(pred_sm_siz_y[:, i, j])
            joint_prob_all = p_on * loc_x[:, np.newaxis, np.newaxis, np.newaxis] * loc_y[np.newaxis, :, np.newaxis,
                                                                                   np.newaxis] * siz_x[np.newaxis,
                                                                                                 np.newaxis, :,
                                                                                                 np.newaxis] * siz_y[
                                                                                                               np.newaxis,
                                                                                                               np.newaxis,
                                                                                                               np.newaxis,
                                                                                                               :]
            joint_prob_max = np.max(joint_prob_all)
            # find index where p > cutoff * joint_prob_max
            idx_selected = np.where(joint_prob_all >= joint_prob_max * cutoff)
            result = []
            for idx in zip(*idx_selected):
                bb_x = i - idx[0]
                bb_y = j + idx[1]
                sm_siz_x = idx[2] + 1
                sm_siz_y = idx[3] + 1
                prob_sm = joint_prob_all[idx]
                result.append((bb_x, bb_y, sm_siz_x, sm_siz_y, prob_sm))
            return result

        def sl_top_one(p_on, pred_loc_x, pred_loc_y, pred_sl_siz_x, pred_sl_siz_y, i, j):
            loc_x = np.argmax(pred_loc_x[:, i, j])
            loc_y = np.argmax(pred_loc_y[:, i, j])
            # scalar size, round to int
            sl_siz_x = int(np.round(pred_sl_siz_x[i, j]))
            sl_siz_y = int(np.round(pred_sl_siz_y[i, j]))
            # avoid setting size 0 or negative # TODO adding logging warning
            if sl_siz_x < 1:
                sl_siz_x = 1
            if sl_siz_y < 1:
                sl_siz_y = 1
            # prob of on/off & location
            prob_1 = p_on * softmax(pred_loc_x[:, i, j])[loc_x] * softmax(pred_loc_y[:, i, j])[loc_y]
            # top right corner
            bb_x = i - loc_x
            bb_y = j + loc_y
            # list of one tuple
            result = [(bb_x, bb_y, sl_siz_x, sl_siz_y, prob_1)]
            return result

        def sl_top_k(p_on, pred_loc_x, pred_loc_y, pred_sl_siz_x, pred_sl_siz_y, i, j, k, cutoff=0):
            """take topk bbs,
            if cutoff is specified (i.e. !=0), only pick those whose prob >= cutoff * p_top"""
            assert k >= 2
            assert 0 <= cutoff <= 1
            # outer product between 2 arrays: loc_x, loc_y
            loc_x = softmax(pred_loc_x[:, i, j])
            loc_y = softmax(pred_loc_y[:, i, j])
            # scalar size, round to int
            sl_siz_x = int(np.round(pred_sl_siz_x[i, j]))
            sl_siz_y = int(np.round(pred_sl_siz_y[i, j]))
            # avoid setting size 0 or negative # TODO adding logging warning
            if sl_siz_x < 1:
                sl_siz_x = 1
            if sl_siz_y < 1:
                sl_siz_y = 1
            joint_prob_all = p_on * loc_x[:, np.newaxis] * loc_y[np.newaxis, :]
            p_top = np.max(joint_prob_all)
            # sort along all axis (reverse idx so result is descending)
            idx_linear = np.argsort(joint_prob_all, axis=None)[::-1]
            # take top k
            idx_linear = idx_linear[:k]
            assert len(idx_linear)
            result = []
            arr_shape = joint_prob_all.shape
            for _i in idx_linear:  # avoid clash with i
                idx = np.unravel_index(_i, arr_shape)  # this idx is 2-D
                bb_x = i - idx[0]
                bb_y = j + idx[1]
                prob_sl = joint_prob_all[idx]
                # if cutoff is specified, we also check the probability
                # break if it's lower than that (since loop is sorted descending)
                if cutoff > 0 and prob_sl < cutoff * p_top:
                    break
                result.append((bb_x, bb_y, sl_siz_x, sl_siz_y, prob_sl))
            return result

        def sl_top_perc(p_on, pred_loc_x, pred_loc_y, pred_sl_siz_x, pred_sl_siz_y, i, j, cutoff):
            """select those whose marginal probability >= cutoff*p_top,
                        setting cutoff == 1 correspond to picking the argmax"""
            assert 0 < cutoff <= 1
            # outer product between 2 arrays: loc_x, loc_y
            loc_x = softmax(pred_loc_x[:, i, j])
            loc_y = softmax(pred_loc_y[:, i, j])
            # scalar size, round to int
            sl_siz_x = int(np.round(pred_sl_siz_x[i, j]))
            sl_siz_y = int(np.round(pred_sl_siz_y[i, j]))
            # avoid setting size 0 or negative # TODO adding logging warning
            if sl_siz_x < 1:
                sl_siz_x = 1
            if sl_siz_y < 1:
                sl_siz_y = 1
            joint_prob_all = p_on * loc_x[:, np.newaxis] * loc_y[np.newaxis, :]
            joint_prob_max = np.max(joint_prob_all)
            # find index where p > cutoff * joint_prob_max
            idx_selected = np.where(joint_prob_all >= joint_prob_max * cutoff)
            result = []
            for idx in zip(*idx_selected):
                bb_x = i - idx[0]
                bb_y = j + idx[1]
                prob_sl = joint_prob_all[idx]
                result.append((bb_x, bb_y, sl_siz_x, sl_siz_y, prob_sl))
            return result

        # remove singleton dimensions
        pred_on = np.squeeze(pred_on)
        pred_loc_x = np.squeeze(pred_loc_x)
        pred_loc_y = np.squeeze(pred_loc_y)
        pred_sm_siz_x = np.squeeze(pred_sm_siz_x)
        if pred_sm_siz_y is None:
            pred_sm_siz_y = np.copy(pred_sm_siz_x)
        else:
            pred_sm_siz_y = np.squeeze(pred_sm_siz_y)
        pred_sl_siz_x = np.squeeze(pred_sl_siz_x)
        if pred_sl_siz_y is None:
            pred_sl_siz_y = np.copy(pred_sl_siz_x)
        else:
            pred_sl_siz_y = np.squeeze(pred_sl_siz_y)
        # TODO assert on input shape

        # hard-mask
        # note that we're also supporting case where pred_on is not square matrix (i.e. two input seqs are of different length)
        seq_len = pred_on.shape[1]
        m = _make_mask(seq_len)
        # apply mask (for pred, only apply to pred_on since our processing starts from that array)
        pred_on = pred_on * m
        # binary array with all 0's, we'll set the predicted bounding box region to 1
        # this will be used to calculate 'sensitivity'
        pred_box = np.zeros_like(pred_on)
        # also save box locations and probabilities
        proposed_boxes = []

        for i, j in np.transpose(np.where(pred_on > thres)):  # TODO vectorize
            # # save sm box # TODO some computation is duplicated in sm/sl
            if perc_cutoff == 0:
                if topk == 1:
                    result = sm_top_one(pred_on[i, j], pred_loc_x, pred_loc_y, pred_sm_siz_x, pred_sm_siz_y, i, j)
                else:
                    result = sm_top_k(pred_on[i, j], pred_loc_x, pred_loc_y, pred_sm_siz_x, pred_sm_siz_y, i, j, topk)
            elif topk == 0:
                result = sm_top_perc(pred_on[i, j], pred_loc_x, pred_loc_y, pred_sm_siz_x, pred_sm_siz_y, i, j, perc_cutoff)
            else:
                result = sm_top_k(pred_on[i, j], pred_loc_x, pred_loc_y, pred_sm_siz_x, pred_sm_siz_y, i, j, topk, perc_cutoff)
            for bb_x, bb_y, sm_siz_x, sm_siz_y, prob_sm in result:
                # assert 0 <= bb_x <= seq_len
                # assert 0 <= bb_y <= seq_len
                # ignore out of bound bbs # TODO print warning?
                if not (0 <= bb_x <= seq_len and 0 <= bb_y <= seq_len):
                    continue
                proposed_boxes, pred_box = _update(bb_x, bb_y, sm_siz_x, sm_siz_y, prob_sm, proposed_boxes, pred_box, bb_source='sm')

            # save sl box (it's ok if sl box is identical with sm box, since probabilities will be aggregated in the end)
            if perc_cutoff == 0:
                if topk == 1:
                    result = sl_top_one(pred_on[i, j], pred_loc_x, pred_loc_y, pred_sl_siz_x, pred_sl_siz_y, i, j)
                else:
                    result = sl_top_k(pred_on[i, j], pred_loc_x, pred_loc_y, pred_sl_siz_x, pred_sl_siz_y, i, j, topk)
            elif topk == 0:
                result = sl_top_perc(pred_on[i, j], pred_loc_x, pred_loc_y, pred_sl_siz_x, pred_sl_siz_y, i, j, perc_cutoff)
            else:
                result = sl_top_k(pred_on[i, j], pred_loc_x, pred_loc_y, pred_sl_siz_x, pred_sl_siz_y, i, j, topk, perc_cutoff)
            for bb_x, bb_y, sl_siz_x, sl_siz_y, prob_sl in result:
                # assert 0 <= bb_x <= seq_len
                # assert 0 <= bb_y <= seq_len
                # ignore out of bound bbs # TODO print warning?
                if not (0 <= bb_x <= seq_len and 0 <= bb_y <= seq_len):
                    continue
                proposed_boxes, pred_box = _update(bb_x, bb_y, sl_siz_x, sl_siz_y, prob_sl, proposed_boxes, pred_box, bb_source='sl')

        # apply hard-mask to pred box
        pred_box = pred_box * m
        return proposed_boxes, pred_box

    def _nn_pred_to_bb(self, seq, yp, threshold, topk=1, perc_cutoff=0, mask=None):
        # apply mask (if specified)
        # mask is applied to *_on output, masked entries set to 0 (thus those pixels won't predict anything)
        if mask is not None:
            # print(yp['stem_on'].shape, mask.shape)
            stem_on = yp['stem_on'] * mask
            iloop_on = yp['iloop_on'] * mask
            hloop_on = yp['hloop_on'] * mask
        else:
            stem_on = yp['stem_on']
            iloop_on = yp['iloop_on']
            hloop_on = yp['hloop_on']
        # bb
        pred_bb_stem, pred_box_stem = self.predict_bounding_box(pred_on=stem_on, pred_loc_x=yp['stem_location_x'],
                                                                pred_loc_y=yp['stem_location_y'],
                                                                pred_sm_siz_x=yp['stem_sm_size'],
                                                                pred_sm_siz_y=None,
                                                                pred_sl_siz_x=yp['stem_sl_size'],
                                                                pred_sl_siz_y=None,
                                                                thres=threshold, topk=topk, perc_cutoff=perc_cutoff)
        pred_bb_iloop, pred_box_iloop = self.predict_bounding_box(pred_on=iloop_on,
                                                                  pred_loc_x=yp['iloop_location_x'],
                                                                  pred_loc_y=yp['iloop_location_y'],
                                                                  pred_sm_siz_x=yp['iloop_sm_size_x'],
                                                                  pred_sm_siz_y=yp['iloop_sm_size_y'],
                                                                  pred_sl_siz_x=yp['iloop_sl_size_x'],
                                                                  pred_sl_siz_y=yp['iloop_sl_size_y'],
                                                                  thres=threshold, topk=topk,
                                                                  perc_cutoff=perc_cutoff)
        pred_bb_hloop, pred_box_hloop = self.predict_bounding_box(pred_on=hloop_on,
                                                                  pred_loc_x=yp['hloop_location_x'],
                                                                  pred_loc_y=yp['hloop_location_y'],
                                                                  pred_sm_siz_x=yp['hloop_sm_size'],
                                                                  pred_sm_siz_y=None,
                                                                  pred_sl_siz_x=yp['hloop_sl_size'],
                                                                  pred_sl_siz_y=None,
                                                                  thres=threshold, topk=topk,
                                                                  perc_cutoff=perc_cutoff)
        pred_bb_hloop = self.cleanup_hloop(pred_bb_hloop, len(seq))
        return yp, pred_bb_stem, pred_bb_iloop, pred_bb_hloop, pred_box_stem, pred_box_iloop, pred_box_hloop

    def _unique_bbs(self, pred_bb_stem, pred_bb_iloop, pred_bb_hloop):
        def uniq_boxes(pred_bb):
            # pred_bb: list
            # group rows correspond to the same bb
            # note that each row has only one of the values: prob_sm/ prob_sl
            # we would like to summerize each into a list, so dropping the NaN rows (for each independently)
            df = pd.DataFrame(pred_bb)
            data = df.groupby(by=['bb_x', 'bb_y', 'siz_x', 'siz_y'], as_index=False).agg(lambda x:  x.dropna().tolist()).to_dict('records')
            return data

        if len(pred_bb_stem) > 0:
            uniq_stem = uniq_boxes(pred_bb_stem)
        else:
            uniq_stem = None
        if len(pred_bb_iloop) > 0:
            uniq_iloop = uniq_boxes(pred_bb_iloop)
        else:
            uniq_iloop = None
        if len(pred_bb_hloop) > 0:
            uniq_hloop = uniq_boxes(pred_bb_hloop)
        else:
            uniq_hloop = None
        return uniq_stem, uniq_iloop, uniq_hloop

    def _predict_patch(self, seq, patch_row_start, patch_row_end, patch_col_start, patch_col_end, ext_patch_row_start, ext_patch_row_end, ext_patch_col_start, ext_patch_col_end):
        # # extract 'left' and 'right' sequence
        seq_1 = seq[ext_patch_row_start:ext_patch_row_end]
        seq_2 = seq[ext_patch_col_start:ext_patch_col_end]

        # index for picking output
        output_row_start = patch_row_start - ext_patch_row_start
        output_row_end = output_row_start + (patch_row_end - patch_row_start)
        output_col_start = patch_col_start - ext_patch_col_start
        output_col_end = output_col_start + (patch_col_end - patch_col_start)

        # predict
        de = SeqPairEncoder(seq_1, seq_2)
        yp = self.model(torch.tensor(de.x_torch))
        # select output
        # for k, v in yp.items():
        #     print(k, v.shape)
        print("Selecting array output range: {}-{}, {}-{}".format(output_row_start, output_row_end, output_col_start, output_col_end))
        yp = {k: v.detach().cpu().numpy()[0, :, output_row_start:output_row_end, output_col_start:output_col_end] for k, v in yp.items()}
        # for k, v in yp.items():
        #     print(k, v.shape)
        return yp

    def predict_bb_split(self, seq, threshold, topk=1, perc_cutoff=0, patch_size=100, trim_size=None):
        """ for predicting on long sequence.
        TODO more details

        :param seq:
        :param threshold:
        :param topk:
        :param perc_cutoff:
        :return:
        """
        # trim_size should be calculated automatically from model params
        if trim_size is not None:
            print("Warning: trim_size set by caller {}. Make sure you're debugging!".format(trim_size))
        else:
            trim_size = sum([x//2 for x in self.filter_width])

        assert topk >= 0  # 0 for unspecified
        assert 0 <= perc_cutoff <= 1  # 0 for unspecified

        seq_len = len(seq)
        n_splits = int(np.ceil(seq_len/patch_size))

        pred_all = {}

        for idx_row in range(n_splits):
            for idx_col in range(n_splits):
                # top left corner
                # this is the output range we'll be extracting
                patch_row_start = idx_row * patch_size
                patch_col_start = idx_col * patch_size
                if patch_row_start + patch_size > seq_len:
                    patch_row_end = seq_len
                else:
                    patch_row_end = patch_row_start + patch_size
                if patch_col_start + patch_size > seq_len:
                    patch_col_end = seq_len
                else:
                    patch_col_end = patch_col_start + patch_size
                # this is what we feed into the NN, with enough context for conv layers
                if patch_row_start - trim_size < 0:
                    ext_patch_row_start = 0
                else:
                    ext_patch_row_start = patch_row_start - trim_size
                if patch_col_start - trim_size < 0:
                    ext_patch_col_start = 0
                else:
                    ext_patch_col_start = patch_col_start - trim_size
                # size (make sure to not go beyond the whole seq)
                if patch_row_start + patch_size + trim_size > seq_len:
                    ext_patch_row_end = seq_len
                else:
                    ext_patch_row_end = patch_row_start + patch_size + trim_size
                if patch_col_start + patch_size + trim_size > seq_len:
                    ext_patch_col_end = seq_len
                else:
                    ext_patch_col_end = patch_col_start + patch_size + trim_size

                # debug FIXME
                print("Input region: {}-{}, {}-{}".format(ext_patch_row_start, ext_patch_row_end, ext_patch_col_start,
                                                          ext_patch_col_end))
                print("Output region: {}-{}, {}-{}".format(patch_row_start, patch_row_end, patch_col_start,
                                                           patch_col_end))
                # check top right corner of output region
                # if it's in lower triangular matrix, that means the whole patch is within lower triangular matrix
                # then we can safely skip predicting on this patch
                if patch_row_start > patch_col_end:
                    print("Patch fully contained in lower triangle matrix, skip.")
                    continue

                # get prediction for the patch
                # returns dict (already converted to np) (selected region)
                pred_patch = self._predict_patch(seq, patch_row_start, patch_row_end,
                                                 patch_col_start, patch_col_end,
                                                 ext_patch_row_start, ext_patch_row_end,
                                                 ext_patch_col_start, ext_patch_col_end)

                # merge into original size array
                for k, v in pred_patch.items():
                    if len(v.shape) == 3:
                        if k not in pred_all:
                            pred_all[k] = np.empty((v.shape[0], seq_len, seq_len))
                        pred_all[k][:, patch_row_start:patch_row_end, patch_col_start:patch_col_end] = v
                    elif len(v.shape) == 4:
                        if k not in pred_all:
                            pred_all[k] = np.empty((v.shape[0], seq_len, seq_len, v.shape[3]))
                        pred_all[k][:, patch_row_start:patch_row_end, patch_col_start:patch_col_end, :] = v

        yp, pred_bb_stem, pred_bb_iloop, pred_bb_hloop, pred_box_stem, pred_box_iloop, pred_box_hloop = self._nn_pred_to_bb(
            seq, pred_all, threshold, topk, perc_cutoff, mask=None)

        uniq_stem, uniq_iloop, uniq_hloop = self._unique_bbs(pred_bb_stem, pred_bb_iloop, pred_bb_hloop)
        return uniq_stem, uniq_iloop, uniq_hloop


    def predict_bb(self, seq, threshold, topk=1, perc_cutoff=0, seq2=None, mask=None, latent_var=None):

        # yp, pred_bb_stem, pred_bb_iloop, pred_bb_hloop, pred_box_stem, pred_box_iloop, pred_box_hloop = self._predict_bb(seq, threshold, topk=topk, perc_cutoff=perc_cutoff, seq2=seq2, mask=mask)

        """topk and perc_cutoff:
                - only topk is specified (perc_cutoff=0): use topk predicted bb
                - only perc_cutoff is specified (topk=0): use predicted bbs whose joint probability is within perc_cutoff * p(top_hit)
                - both topk and perc_cutoff are specified: use predicted bbs whose joint probability is within perc_cutoff * p(top_hit) AND
                is within topk
                """
        assert topk >= 0  # 0 for unspecified
        assert 0 <= perc_cutoff <= 1  # 0 for unspecified
        # if seq2 specified, predict bb for RNA-RNA
        if seq2:
            de = SeqPairEncoder(seq, seq2)
        else:
            de = DataEncoder(seq)

        # pass encoded sequence through CNN
        x = torch.tensor(de.x_torch)
        x_cnn = self.model.process_x(x)

        # if latent variable is not specified, sample it from fixed prior
        if latent_var is None:
            z = self.model.reparameterize(torch.zeros(1, self.latent_dim, x.shape[2], x.shape[3]),
                                                     torch.zeros(1, self.latent_dim, x.shape[2], x.shape[3]))
        # otherwise make sure the shape is correct
        else:
            assert z.shape[0] == 1
            assert z.shape[1] == self.latent_dim
            assert z.shape[2] == x.shape[2]
            assert z.shape[3] == x.shape[3]

        # run decoder
        xz = torch.cat([x_cnn, z], dim=1)
        yp = self.model.decode(xz)
        # yp = self.model(torch.tensor(de.x_torch))

        # TODO make new interface to sample N times (CNN only need to be run once <- faster)



        # single example, remove batch dimension
        yp = {k: v.detach().cpu().numpy()[0, :, :, :] for k, v in yp.items()}

        yp, pred_bb_stem, pred_bb_iloop, pred_bb_hloop, pred_box_stem, pred_box_iloop, pred_box_hloop = self._nn_pred_to_bb(
            seq, yp, threshold, topk, perc_cutoff, mask)

        uniq_stem, uniq_iloop, uniq_hloop = self._unique_bbs(pred_bb_stem, pred_bb_iloop, pred_bb_hloop)
        return uniq_stem, uniq_iloop, uniq_hloop


class Evaluator(object):

    def __init__(self, predictor):
        if predictor is None:
            print("Initializing evaluator without predictor")  # some class methods can still be used
        else:
            assert isinstance(predictor, Predictor)
        self.predictor = predictor
        # hold on to data for one example, for convenience
        self.data_encoder = None
        # predictions
        self.yp = None
        self.pred_bb_stem = None
        self.pred_bb_iloop = None
        self.pred_bb_hloop = None

    @staticmethod
    def make_plot_bb(target, pred_box):
        fig = px.imshow(target)
        for bb in pred_box:
            bb_x = bb['bb_x']
            bb_y = bb['bb_y']
            siz_x = bb['siz_x']
            siz_y = bb['siz_y']
            prob = bb['prob'] if 'prob' in bb else 1.0

            x0 = bb_x
            y0 = bb_y - siz_y + 1  # 0-based
            wx = siz_x
            wy = siz_y
            fig.add_shape(
                type='rect',
                y0=x0 - 0.5, y1=x0 + wx - 0.5, x0=y0 - 0.5, x1=y0 + wy - 0.5,  # image plot axis is swaped
                xref='x', yref='y',
                opacity=prob,  # opacity proportional to probability of bounding box
                line_color='red'
            )

        # # update figure
        # fig['layout'].update(height=800, width=800)

        return fig

    @staticmethod
    def make_target_bb_df(target_bb, convert_tl_to_tr=False):
        # convert_tl_to_tr: if set to True, target_bb has top left corner, will be converted to top right corner
        df_target_stem = []
        df_target_iloop = []
        df_target_hloop = []
        for (bb_x, bb_y), (siz_x, siz_y), bb_type in target_bb:
            if convert_tl_to_tr:
                bb_y = bb_y + siz_y - 1
            row = {
                'bb_x': bb_x,
                'bb_y': bb_y,
                'siz_x': siz_x,
                'siz_y': siz_y,
            }
            if bb_type == 'stem':
                df_target_stem.append(row)
            elif bb_type in ['bulge', 'internal_loop']:
                df_target_iloop.append(row)
            elif bb_type == 'hairpin_loop':
                df_target_hloop.append(row)
            elif bb_type in ['pseudo_knot', 'pesudo_knot']:  # allow for typo -_-
                pass  # do not process
            else:
                raise ValueError(bb_type)  # should never be here
        if len(df_target_stem) > 0:
            df_target_stem = pd.DataFrame(df_target_stem)
        if len(df_target_iloop) > 0:
            df_target_iloop = pd.DataFrame(df_target_iloop)
        if len(df_target_hloop) > 0:
            df_target_hloop = pd.DataFrame(df_target_hloop)
        return df_target_stem, df_target_iloop, df_target_hloop

    @staticmethod
    def _calculate_bb_metrics(df_target, df_pred):

        def is_identical(bb1, bb2):
            return bb1 == bb2  # this should work? FIXME
            # bb1_x, bb1_y, siz1_x, siz1_y = bb1
            # bb2_x, bb2_y, siz2_x, siz2_y = bb2
            # # FIXME debug! any off-by-1 error?
            # return abs(bb1_x-bb2_x)<=1 and abs(bb1_y-bb2_y)<=1 and abs(siz1_x-siz2_x)<=1 and abs(siz1_y-siz2_y)<=1

        def is_local_shift(bb1, bb2):
            """check if bb1 and bb2 are local shift/expand version of each other
            max diff <= 1, include case where bb1 == bb2"""
            bb1_x, bb1_y, siz1_x, siz1_y = bb1
            bb2_x, bb2_y, siz2_x, siz2_y = bb2
            max_diff = max(abs(bb1_x - bb2_x), abs(bb1_y - bb2_y), abs(siz1_x - siz2_x), abs(siz1_y - siz2_y))
            if max_diff <= 1:
                return True
            else:
                return False

        def area_overlap(bb1, bb2):
            bb1_x, bb1_y, siz1_x, siz1_y = bb1
            bb2_x, bb2_y, siz2_x, siz2_y = bb2
            # calculate overlap rectangle, check to see if it's empty
            x0 = max(bb1_x, bb2_x)
            x1 = min(bb1_x + siz1_x - 1, bb2_x + siz2_x - 1)  # note this is closed end
            y0 = max(bb1_y - siz1_y + 1, bb2_y - siz2_y + 1)  # closed end
            y1 = min(bb1_y, bb2_y)
            if x1 >= x0 and y1 >= y0:
                # return area of overlapping rectangle
                return (x1 - x0 + 1) * (y1 - y0 + 1)
            else:
                return 0

        assert set(df_target.columns) == {'bb_x', 'bb_y', 'siz_x', 'siz_y'}
        assert set(df_pred.columns) == {'bb_x', 'bb_y', 'siz_x', 'siz_y'}

        # make sure all rows are unique
        assert not df_target.duplicated().any()
        assert not df_pred.duplicated().any()

        # w.r.t. target
        n_target_total = len(df_target)
        n_target_local = 0
        n_target_identical = 0
        n_target_overlap = 0
        n_target_nohit = 0
        for _, row1 in df_target.iterrows():
            bb1 = (row1['bb_x'], row1['bb_y'], row1['siz_x'], row1['siz_y'])
            found_identical = False
            found_local_shift = False
            found_overlapping = False
            best_area_overlap = 0
            best_bb_overlap = None
            for _, row2 in df_pred.iterrows():
                bb2 = (row2['bb_x'], row2['bb_y'], row2['siz_x'], row2['siz_y'])
                if is_identical(bb1, bb2):
                    found_identical = True
                elif is_local_shift(bb1, bb2): # note this is overlapping but NOT local shift due to "elif"
                    found_local_shift = True
                elif area_overlap(bb1, bb2) > 0:  # note this is overlapping but NOT identical or local shift due to "elif"
                    found_overlapping = True
                    this_area = area_overlap(bb1, bb2)
                    if this_area > best_area_overlap:
                        best_area_overlap = this_area
                        best_bb_overlap = bb2
                else:
                    pass
            if found_identical:
                n_target_identical += 1
            elif found_local_shift:
                n_target_local += 1
            elif found_overlapping:
                n_target_overlap += 1
                # debug print closest pred bb
                print("target bb: {}".format(bb1))
                print("best overlapping bb: {}".format(best_bb_overlap))
                print("best overlapping area: {}".format(best_area_overlap))
            else:
                n_target_nohit += 1

        # FIXME there is some wasted comparison here (can be combined with last step)
        # w.r.t. pred
        n_pred_total = len(df_pred)
        n_pred_local = 0
        n_pred_identical = 0
        n_pred_overlap = 0
        n_pred_nohit = 0
        for _, row1 in df_pred.iterrows():
            bb1 = (row1['bb_x'], row1['bb_y'], row1['siz_x'], row1['siz_y'])
            found_identical = False
            found_local_shift = False
            found_overlapping = False
            for _, row2 in df_target.iterrows():
                bb2 = (row2['bb_x'], row2['bb_y'], row2['siz_x'], row2['siz_y'])
                if is_identical(bb1, bb2):
                    found_identical = True
                elif is_local_shift(bb1, bb2): # note this is overlapping but NOT local shift due to "elif"
                    found_local_shift = True
                elif area_overlap(bb1, bb2) > 0:  # note this is overlapping but NOT identical or local shift due to "elif"
                    found_overlapping = True
                else:
                    pass
            if found_identical:
                n_pred_identical += 1
            elif found_local_shift:
                n_pred_local += 1
            elif found_overlapping:
                n_pred_overlap += 1
            else:
                n_pred_nohit += 1
        result = {
            'n_target_total': n_target_total,
            'n_target_local': n_target_local,
            'n_target_identical': n_target_identical,
            'n_target_overlap': n_target_overlap,
            'n_target_nohit': n_target_nohit,
            'n_pred_total': n_pred_total,
            'n_pred_local': n_pred_local,
            'n_pred_identical': n_pred_identical,
            'n_pred_overlap': n_pred_overlap,
            'n_pred_nohit': n_pred_nohit,
        }
        return result

    def calculate_bb_metrics(self, df_target, df_pred):
        if (df_target is None or len(df_target) == 0) and (df_pred is None or len(df_pred) == 0):
            return {
                'n_target_total': 0,
                'n_target_local': 0,
                'n_target_identical': 0,
                'n_target_overlap': 0,
                'n_target_nohit': 0,
                'n_pred_total': 0,
                'n_pred_local': 0,
                'n_pred_identical': 0,
                'n_pred_overlap': 0,
                'n_pred_nohit': 0,
            }

        elif df_target is None or len(df_target) == 0:
            return {
                'n_target_total': 0,
                'n_target_local': 0,
                'n_target_identical': 0,
                'n_target_overlap': 0,
                'n_target_nohit': 0,
                'n_pred_total': len(df_pred),
                'n_pred_local': 0,
                'n_pred_identical': 0,
                'n_pred_overlap': 0,
                'n_pred_nohit': 0,
            }
        elif df_pred is None or len(df_pred) == 0:
            return {
                'n_target_total': len(df_target),
                'n_target_local': 0,
                'n_target_identical': 0,
                'n_target_overlap': 0,
                'n_target_nohit': 0,
                'n_pred_total': 0,
                'n_pred_local': 0,
                'n_pred_identical': 0,
                'n_pred_overlap': 0,
                'n_pred_nohit': 0,
            }
        else:
            return self._calculate_bb_metrics(df_target, df_pred)

    @staticmethod
    def sensitivity_specificity(target_on, pred_box, hard_mask):
        sensitivity = np.sum((pred_box * target_on) * hard_mask) / np.sum(target_on * hard_mask)
        specificity = np.sum((1 - pred_box) * (1 - target_on) * hard_mask) / np.sum((1 - target_on) * hard_mask)
        return sensitivity, specificity

    def calculate_metrics(self):
        # convert to dfs
        if len(self.pred_bb_stem) > 0:
            df_stem = pd.DataFrame(self.pred_bb_stem)
            df_stem = df_stem[['bb_x', 'bb_y', 'siz_x', 'siz_y']].drop_duplicates()
        else:
            df_stem = None
        if len(self.pred_bb_iloop) > 0:
            df_iloop = pd.DataFrame(self.pred_bb_iloop)
            df_iloop = df_iloop[['bb_x', 'bb_y', 'siz_x', 'siz_y']].drop_duplicates()
        else:
            df_iloop = None
        if len(self.pred_bb_hloop) > 0:
            df_hloop = pd.DataFrame(self.pred_bb_hloop)
            df_hloop = df_hloop[['bb_x', 'bb_y', 'siz_x', 'siz_y']].drop_duplicates()
        else:
            df_hloop = None
        # process target bb list into different types, store in df
        df_target_stem, df_target_iloop, df_target_hloop = self.make_target_bb_df(self.data_encoder.y_bb)

        # metric for each bb type
        m_stem = self.calculate_bb_metrics(df_target_stem, df_stem)
        m_iloop = self.calculate_bb_metrics(df_target_iloop, df_iloop)
        m_hloop = self.calculate_bb_metrics(df_target_hloop, df_hloop)
        # calculate non-bb sensitivity and specificity

        def _make_mask(l):
            m = np.ones((l, l))
            m[np.tril_indices(l)] = 0
            return m

        m = _make_mask(len(self.data_encoder.x))
        se_stem, sp_stem = self.sensitivity_specificity(self.data_encoder.y_arrs['stem_on'],
                                                        self.pred_box_stem, m)
        se_iloop, sp_iloop = self.sensitivity_specificity(self.data_encoder.y_arrs['iloop_on'],
                                                          self.pred_box_iloop, m)
        se_hloop, sp_hloop = self.sensitivity_specificity(self.data_encoder.y_arrs['hloop_on'],
                                                          self.pred_box_hloop, m)
        # combine
        m_stem.update({'struct_type': 'stem', 'pixel_sensitivity': se_stem, 'pixel_specificity': sp_stem})
        m_iloop.update({'struct_type': 'iloop', 'pixel_sensitivity': se_iloop, 'pixel_specificity': sp_iloop})
        m_hloop.update({'struct_type': 'hloop', 'pixel_sensitivity': se_hloop, 'pixel_specificity': sp_hloop})
        df_result = pd.DataFrame([m_stem, m_iloop, m_hloop])
        df_result['bb_sensitivity_identical'] = df_result['n_target_identical'] / df_result['n_target_total']
        df_result['bb_sensitivity_local_shift'] = (df_result['n_target_identical'] + df_result['n_target_local']) / \
                                              df_result['n_target_total']
        df_result['bb_sensitivity_overlap'] = (df_result['n_target_identical'] + df_result['n_target_local'] + df_result['n_target_overlap']) / \
                                              df_result['n_target_total']
        # also extract the sensitivities
        assert len(df_result) == 3
        metrics = {
            # bb sensitivity
            'bb_stem_identical': df_result[df_result['struct_type'] == 'stem'].iloc[0]['bb_sensitivity_identical'],
            'bb_stem_local_shift': df_result[df_result['struct_type'] == 'stem'].iloc[0]['bb_sensitivity_local_shift'],
            'bb_stem_overlap': df_result[df_result['struct_type'] == 'stem'].iloc[0]['bb_sensitivity_overlap'],
            'bb_iloop_identical': df_result[df_result['struct_type'] == 'iloop'].iloc[0]['bb_sensitivity_identical'],
            'bb_iloop_local_shift': df_result[df_result['struct_type'] == 'iloop'].iloc[0]['bb_sensitivity_local_shift'],
            'bb_iloop_overlap': df_result[df_result['struct_type'] == 'iloop'].iloc[0]['bb_sensitivity_overlap'],
            'bb_hloop_identical': df_result[df_result['struct_type'] == 'hloop'].iloc[0]['bb_sensitivity_identical'],
            'bb_hloop_local_shift': df_result[df_result['struct_type'] == 'hloop'].iloc[0]['bb_sensitivity_local_shift'],
            'bb_hloop_overlap': df_result[df_result['struct_type'] == 'hloop'].iloc[0]['bb_sensitivity_overlap'],
            # pixel
            'px_stem_sensitivity': df_result[df_result['struct_type'] == 'stem'].iloc[0]['pixel_sensitivity'],
            'px_stem_specificity': df_result[df_result['struct_type'] == 'stem'].iloc[0]['pixel_specificity'],
            'px_iloop_sensitivity': df_result[df_result['struct_type'] == 'iloop'].iloc[0]['pixel_sensitivity'],
            'px_iloop_specificity': df_result[df_result['struct_type'] == 'iloop'].iloc[0]['pixel_specificity'],
            'px_hloop_sensitivity': df_result[df_result['struct_type'] == 'hloop'].iloc[0]['pixel_sensitivity'],
            'px_hloop_specificity': df_result[df_result['struct_type'] == 'hloop'].iloc[0]['pixel_specificity'],
        }
        return df_result, metrics

    @staticmethod
    def bb_unique(bb):
        # convert list of bb to unique bb
        # input:
        # [{'bb_x': 0, 'bb_y': 16, 'siz_x': 3, 'siz_y': 3, 'prob': 0.1654079953181778},
        # {'bb_x': 0, 'bb_y': 72, 'siz_x': 8, 'siz_y': 8, 'prob': 0.11180505377843244}, ...]
        if len(bb) > 0:
            df_tmp = pd.DataFrame(bb)
            df_tmp = df_tmp[['bb_x', 'bb_y', 'siz_x', 'siz_y']].drop_duplicates()
            return df_tmp.to_dict('records')
        else:
            return bb

    def predict(self, seq, y, threshold):
        assert 0 <= threshold <= 1
        # y: in one_idx format, tuple of two lists of i's and j's
        self.data_encoder = DataEncoder(seq, y, bb_ref='top_right')
        yp, pred_bb_stem, pred_bb_iloop, pred_bb_hloop, pred_box_stem, pred_box_iloop, pred_box_hloop = self.predictor._predict_bb(seq, threshold)
        self.yp = yp
        self.pred_bb_stem = pred_bb_stem
        self.pred_bb_iloop = pred_bb_iloop
        self.pred_bb_hloop = pred_bb_hloop
        # extract unique ones, drop prob
        self.pred_bb_stem_uniq = self.bb_unique(self.pred_bb_stem)
        self.pred_bb_iloop_uniq = self.bb_unique(self.pred_bb_iloop)
        self.pred_bb_hloop_uniq = self.bb_unique(self.pred_bb_hloop)
        # pred box (filled in)
        self.pred_box_stem = pred_box_stem
        self.pred_box_iloop = pred_box_iloop
        self.pred_box_hloop = pred_box_hloop

    def plot_bb_prob(self):
        fig_stem = self.make_plot_bb(self.data_encoder.y_arrs['stem_on'], self.pred_bb_stem)
        fig_stem['layout'].update(title='Stem')
        fig_iloop = self.make_plot_bb(self.data_encoder.y_arrs['iloop_on'], self.pred_bb_iloop)
        fig_iloop['layout'].update(title='Internal loop')
        fig_hloop = self.make_plot_bb(self.data_encoder.y_arrs['hloop_on'], self.pred_bb_hloop)
        fig_hloop['layout'].update(title='Hairpin loop')
        return fig_stem, fig_iloop, fig_hloop

    def plot_bb_uniq(self):
        # # TODO bb counts etc. (also save in class)
        # # subplot
        # fig = make_subplots(rows=1, cols=3, print_grid=False, shared_yaxes=True, shared_xaxes=True)
        fig_stem = self.make_plot_bb(self.data_encoder.y_arrs['stem_on'], self.pred_bb_stem_uniq)
        # fig.append_trace(fig_stem.data[0], 1, 1)
        # fig['layout']['xaxis1'].update(title='stem: {}'.format(len(self.pred_bb_stem_uniq)))
        # fig['layout']['yaxis1']['autorange'] = "reversed"
        fig_iloop = self.make_plot_bb(self.data_encoder.y_arrs['iloop_on'], self.pred_bb_iloop_uniq)
        # fig.append_trace(fig_iloop.data[0], 1, 2)
        # fig['layout']['xaxis2'].update(title='iloop: {}'.format(len(self.pred_bb_iloop_uniq)))
        # fig['layout']['yaxis2']['autorange'] = "reversed"
        fig_hloop = self.make_plot_bb(self.data_encoder.y_arrs['hloop_on'], self.pred_bb_hloop_uniq)
        # fig.append_trace(fig_hloop.data[0], 1, 3)
        # fig['layout']['xaxis3'].update(title='hloop: {}'.format(len(self.pred_bb_hloop_uniq)))
        # fig['layout']['yaxis3']['autorange'] = "reversed"
        # fig['layout'].update(height=400, width=400 * 3, title="todo")
        # fig['layout']['yaxis']['autorange'] = "reversed"
        return fig_stem, fig_iloop, fig_hloop
