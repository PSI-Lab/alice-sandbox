import os
import logging
import subprocess
import yaml
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pandas as pd
import datacorral as dc
import dgutils.pandas as dgp
import numpy as np
import math



class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        # mask should be of shape batch x length, if specified
        if mask is not None:
            assert mask.size(0) == q.size(0)
            assert mask.size(1) == q.size(1)

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1 ,2)
        q = q.transpose(1 ,2)
        v = v.transpose(1 ,2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1 ,2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        if mask is not None:
            # mask out positions on output
            mask = mask.unsqueeze(-1)  # add feature dimension for broadcasting
            output = output.masked_fill(mask == 0, 0)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        #         print(mask.shape)
        mask = mask.unsqueeze(1).unsqueeze \
            (-1)  # add dimension for heads, so we can broadcast, this makes it batch x 1 x length x 1
        mask = torch.matmul(mask, mask.transpose(-2, -1))  # use outer product to conver length-wise mask to matrix, e.g. l=5 ones will correspond to 5x5 ones matrix, this makes batch x 1 x length x length
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2 ,x2 ,x2 ,mask))
        x = self.dropout_1(x)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))

        if mask is not None:
            # mask out positions on output
            mask = mask.unsqueeze(-1)  # add feature dimension for broadcasting
            x = x.masked_fill(mask == 0, 0)
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MyModel(nn.Module):
    def __init__(self, in_size, d_model, N, heads, n_hid):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(in_size, d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
        self.hid = nn.Linear(d_model, n_hid)
        self.out = nn.Linear(n_hid, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, src, mask):
        x = self.embed(src)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        # FC layers
        x = self.act1(self.hid(x))
        x = self.out(x)

        if mask is not None:
            # mask out positions on output before sigmoid (mask using -1e9 so the output will be ~0)
            mask = mask.unsqueeze(-1)  # add feature dimension for broadcasting
            x = x.masked_fill(mask == 0, -1e9)

        return self.act2(x)


# def find_match_bb(bb, df_target, bb_type):
#     hit = df_target[(df_target['bb_type'] == bb_type) & (df_target['bb_x'] == bb['bb_x']) & (df_target['bb_y'] == bb['bb_y']) &
#                 (df_target['siz_x'] == bb['siz_x']) & (df_target['siz_y'] == bb['siz_y'])]
#     if len(hit) > 0:
#         assert len(hit) == 1
#         return True
#     else:
#         return False


# def make_dataset(df):
#     # for the sole purpose of training, subset to example where s2 label can be generated EXACTLY
#     # i.e. subset to example where s1 bb sensitivity is 100%
#     df = dgp.add_column(df, 'n_bb', ['bounding_boxes'], len)
#     n_old = len(df)
#     df = df[df['n_bb'] == df['n_bb_found']]
#     logging.info("Subset to examples with 100% S1 bb sensitivity (for now). Before {}, after {}".format(n_old, len(df)))
#
#     # putting together the dataset
#     # for each row:
#     # encode input: a list of:
#     # bb_type, x, y, wx, wy, median_prob, n_proposal_normalized  (TODO add both corners?)
#     # encode output: binary label for each input 'position'
#
#     x_all = []
#     y_all = []
#
#     for idx, row in df.iterrows():
#         if idx % 10000 == 0:  # FIXME idx is the original idx (not counter)
#             logging.info("Processed {} examples".format(idx))
#
#         _x = []
#         _y = []
#         df_target = pd.DataFrame(row['df_target'])
#         if row['bb_stem'] is not None:
#             for x in row['bb_stem']:
#                 if find_match_bb(x, df_target, 'stem'):
#                     label = 1
#                 else:
#                     label = 0
#                 # 100 for stem
#                 _x.append([1, 0, 0, x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y'], np.median(x['prob']),
#                            len(x['prob']) / (x['siz_x'] * x['siz_y'])])
#                 _y.append(label)
#         if row['bb_iloop'] is not None:
#             for x in row['bb_iloop']:
#                 if find_match_bb(x, df_target, 'iloop'):
#                     label = 1
#                 else:
#                     label = 0
#                 # 010 for iloop
#                 _x.append([0, 1, 0, x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y'], np.median(x['prob']),
#                            len(x['prob']) / (x['siz_x'] * x['siz_y'])])
#                 _y.append(label)
#         if row['bb_hloop'] is not None:
#             for x in row['bb_hloop']:
#                 if find_match_bb(x, df_target, 'hloop'):
#                     label = 1
#                 else:
#                     label = 0
#                 # 001 for hloop, also multiple normalized n_proposal by 2 to make upper limit 1
#                 _x.append([0, 0, 1, x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y'], np.median(x['prob']),
#                            2 * len(x['prob']) / (x['siz_x'] * x['siz_y'])])
#                 _y.append(label)
#         x_all.append(np.array(_x))
#         y_all.append(np.array(_y))
#     return x_all, y_all  # two lists


# def make_single_pred(model, x, y):
#     model.eval()
#     # add batch dim, convert to torch tensor, make pred
#     x = torch.from_numpy(x[np.newaxis, :, :]).float()
#     y = torch.from_numpy(y[np.newaxis, :]).float()
#     preds = model(x, mask=None)  # no masking since parsing one example at a time for now
#     return preds


class Predictor(object):
    model_versions = {}  # nothing uploaded for now

    def __init__(self, model_ckpt, in_size=9, d_model=100, N=5, heads=5, n_hid=20):
        # model file path
        dc_client = dc.Client()
        if model_ckpt in self.model_versions:
            model_file = dc_client.get_path(self.model_versions[model_ckpt])
        elif os.path.isfile(model_ckpt):
            model_file = model_ckpt
        else:
            model_file = dc_client.get_path(model_ckpt)

        model = MyModel(in_size, d_model, N, heads, n_hid)
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        # set to be in inference mode
        model.eval()
        # TODO print model summary
        self.model = model

    @staticmethod
    def _validate_df(df):
        # validate S1 processed output
        # required fields
        assert set(df.columns) == {'bb_x', 'bb_y', 'siz_x', 'siz_y', 'prob_median', 'n_proposal_norm'}
        # values
        assert df['prob_median'].min() >= 0
        assert df['prob_median'].max() <= 1
        assert df['n_proposal_norm'].min() > 0
        assert df['n_proposal_norm'].max() <= 1
        # no duplicate bbs
        assert len(df[['bb_x', 'bb_y', 'siz_x', 'siz_y']].drop_duplicates()) == len(df)

    def encode_bbs(self, stems, iloops, hloops):
        # stems is a df with columns:
        # (bb_x, bb_y, siz_x, siz_y, prob_median, n_proposal_norm)
        # iloops and hloops are similar
        # can be None
        # these should be processed from S1 model output
        features = []
        if stems is not None:
            self._validate_df(stems)
            for _, x in stems.iterrows():
                features.append([1, 0, 0, x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y'], x['prob_median'], x['n_proposal_norm']])
        if iloops is not None:
            self._validate_df(iloops)
            for _, x in iloops.iterrows():
                features.append([0, 1, 0, x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y'], x['prob_median'], x['n_proposal_norm']])
        if hloops is not None:
            self._validate_df(hloops)
            for _, x in hloops.iterrows():
                features.append([0, 0, 1, x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y'], x['prob_median'], x['n_proposal_norm']])
        features = np.asarray(features)
        return features

    def predict(self, stems, iloops, hloops):
        features = self.encode_bbs(stems, iloops, hloops)
        # for single example, add in batch dimension
        x = torch.from_numpy(features[np.newaxis, :, :]).float()
        pred = self.model(x, mask=None)  # no masking since parsing one example at a time for now
        return pred

