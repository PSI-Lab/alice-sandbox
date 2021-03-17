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


    def __init__(self, latent_dim):
        super(LatentVarModel, self).__init__()

        # num_filters = [8] + num_filters
        # filter_width = [None] + filter_width
        # cnn_layers = []
        # for i, (nf, fw) in enumerate(zip(num_filters[1:], filter_width[1:])):
        #     assert fw % 2 == 1  # odd
        #     cnn_layers.append(nn.Conv2d(num_filters[i], nf, kernel_size=fw, stride=1, padding=fw//2))
        #     cnn_layers.append(nn.BatchNorm2d(nf))
        #     cnn_layers.append(nn.ReLU(inplace=True))
        #     if dropout > 0:
        #         cnn_layers.append(nn.Dropout(dropout))
        # self.cnn_layers = nn.Sequential(*cnn_layers)

        self.encoder_2d_y = nn.Sequential(
            nn.Conv2d(8+1, 32, kernel_size=9, stride=1, padding=9 // 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(1, 1)),
            nn.Conv2d(32, 32, kernel_size=9, stride=1, padding=9 // 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(1, 1)),
            nn.Conv2d(32, 64, kernel_size=9, stride=1, padding=9 // 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(1, 1)),
        )

        # TODO hard-mask before global pooling!

        # posterior network - applied after global max pooling
        # "encoder" of the cvae, x + y -> z's param
        self.posterior_fc = nn.Sequential(
            nn.Linear(64, 50),
            nn.ReLU(),
        )
        # posterior mean and logvar
        self.posterior_mean = nn.Linear(50, latent_dim)
        self.posterior_logvar = nn.Linear(50, latent_dim)

        # prior network

        # self.encoder_1d = nn.LSTM(input_size=4, hidden_size=20, num_layers=2, batch_first=True, bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.project_1d = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1)
        self.encoder_1d = nn.TransformerEncoder(encoder_layer, num_layers=2)  # TODO this expects L x batch x feature


        # posterior mean and logvar - applied after taking LSTM first and last output

        #
        # self.prior_fc = nn.Sequential(
        #     nn.Conv2d(num_filters[-1], 50, kernel_size=1),
        #     nn.ReLU(),
        # )
        # applied after summing over transformer output
        self.prior_fc = nn.Sequential(
            nn.Linear(64, 50),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(50, latent_dim)
        self.prior_logvar = nn.Linear(50, latent_dim)

        # output - applied to x concat with z (broadcast)
        self.output = nn.Sequential(
            nn.Conv2d(8 + latent_dim, 32, kernel_size=9, padding=9 // 2),  # TODO different kernel size?
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=9, padding=9 // 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=9, padding=9 // 2),
            nn.Sigmoid(),
        )

    def posterior_network(self, x_2d, y):
        # x_2d: batch x 8 x L x L
        # y: batch x 1 x L x L   # TODO does it have the last dim?
        x = torch.cat([x_2d, y], dim=1)
        # conv layers
        x = self.encoder_2d_y(x)
        # global pool
        # TODO apply mask first
        # TODO max ?
        # x = x.max(dim=3).max(dim=2)
        x = x.mean(dim=3).mean(dim=2)
        # FC
        x = self.posterior_fc(x)
        # comute posterior params
        return self.posterior_mean(x), self.posterior_logvar(x)

    def prior_network(self, x_1d):
        # shape: batch x 4 x L
        # proj to transformer dimension
        x = self.project_1d(x_1d)
        # permute to fit transformer interface
        # print(x_1d.shape)
        # print(x.shape)
        x = x.permute(2, 0, 1)
        # print(x.shape)
        # transformer encoder
        x = self.encoder_1d(x)   # L x batch x feature
        # mean over length
        x = x.mean(0)
        # FC
        x = self.prior_fc(x)
        # compute prior params
        return self.prior_mean(x), self.prior_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def output_network(self, x_2d, z):
        # TODO broadcast z
        # x_2d is batch x 8 x L x L
        # print(z.shape)
        # manual broadcasting
        # x = torch.cat([x_2d, z], dim=1)
        x = torch.cat([x_2d, z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_2d.shape[2], x_2d.shape[3])], dim=1)
        # conv layers
        return self.output(x)

    # Defining the forward pass
    def forward(self, x_1d, x_2d, y):
        # prior
        mu_p, logvar_p = self.prior_network(x_1d)
        # posterior
        mu_q, logvar_q = self.posterior_network(x_2d, y)
        # sample z
        z = self.reparameterize(mu_q, logvar_q)
        # decoder
        return self.output_network(x_2d, z), mu_q, logvar_q, mu_p, logvar_p

    def inference(self, x_1d, x_2d):  # using prior network
        # prior
        mu_p, logvar_p = self.prior_network(x_1d)
        # sample z
        z = self.reparameterize(mu_p, logvar_p)

        # decoder
        return self.output_network(x_2d, z), mu_p, logvar_p, z


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
        self.x_1d_torch = self.encode_torch_input_1d(self.x_1d)
        self.x_2d_torch = self.encode_torch_input_2d(self.x_2d)
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

    def encode_torch_input_1d(self, x):
        # add batch dim
        assert len(x.shape) == 2
        x = x[np.newaxis, :, :]
        # convert to torch tensor
        x = torch.from_numpy(x).float()
        # reshape: batch x channel x L
        x = x.permute(0, 2, 1)
        return x

    def encode_torch_input_2d(self, x):
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

    def __init__(self, model_ckpt, latent_dim=None):
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

        if latent_dim is None:
            latent_dim = 20

        # needed for sampling z
        self.latent_dim = latent_dim

        model = LatentVarModel(latent_dim=latent_dim)
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        # set to be in inference mode
        model.eval()
        # TODO print model summary
        self.model = model

    def predict_matrix(self, seq):
        # FIXME seq2 won't work due to the way model was trained! (models takes a single x_1d for encoding prior)

        de = DataEncoder(seq)

        x_1d = torch.tensor(de.x_1d_torch)
        x_2d = torch.tensor(de.x_2d_torch)
        yp, mu_p, logvar_p, z = self.model.inference(x_1d, x_2d)


        # TODO make new interface to sample N times (CNN only need to be run once <- faster)
        assert yp.shape[0] == 1
        assert yp.shape[1] == 1
        # remove batch and channel dimension
        yp = yp.detach().cpu().numpy()[0, 0, :, :]

        # also return latent param and state
        assert mu_p.shape[0] == 1
        mu_p = mu_p.detach().cpu().numpy()[0, :]
        assert logvar_p.shape[0] == 1
        logvar_p = logvar_p.detach().cpu().numpy()[0, :]
        assert z.shape[0] == 1
        z = z.detach().cpu().numpy()[0, :]

        return yp, mu_p, logvar_p, z


