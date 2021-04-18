import logging
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Optional
from torch_geometric.data import Data
from torch_geometric.typing import Adj, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, Parameter, GRUCell
from torch_geometric.utils import softmax
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot, zeros


# from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/attentive_fp.html?highlight=edge_attr
# this is the graph conv layer that uses edge feature
# i.e. x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
class GATEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 dropout: float = 0.0):
        super(GATEConv, self).__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out += self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


def roc_prc(x, y, m):
    # true, score, mask
    mask_bool = m.eq(1)
    _x2 = x.masked_select(mask_bool).flatten().detach().cpu().numpy()
    _y2 = y.masked_select(mask_bool).flatten().detach().cpu().numpy()
    # do not compute if empty (e.g. when all elements are being masked)
    # do not compute if there's only one class
    if len(_x2) > 0 and not np.all(_x2 == _x2[0]):
        roc = roc_auc_score(_x2, _y2)
        prc = average_precision_score(_x2, _y2)
    else:
        roc = np.NaN
        prc = np.NaN
    return roc, prc


def one_hot_single_base(seq):
    seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U',
                                                                                                      '4').replace(
        'N', '0')
    tmp = np.asarray(list(map(int, list(seq))), dtype=np.int16)
    # one-hot
    x = np.zeros((len(seq), 4))
    x[np.arange(len(seq)), tmp - 1] = 1
    return torch.from_numpy(x).float()


def make_dataset(df, fn_make_target, fn_encode_seq=one_hot_single_base):
    data_list = []

    for _, row in df.iterrows():
        seq = row['seq']
        stem_bb_bps = row['stem_bb_bps']
        target_bps = row['target_bps']

        # # use integer encoding for now
        # seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U',
        #                                                                                                   '4').replace(
        #     'N', '0')
        # tmp = np.asarray(list(map(int, list(seq))), dtype=np.int16)
        # # one-hot
        # # TODO instead of 1-hot encode each base, we can 1-hot encode the local k-mer
        # node_features = np.zeros((len(seq), 4))
        # node_features[np.arange(len(seq)), tmp - 1] = 1
        node_features = fn_encode_seq(seq)
        # node_features =

        # build edges
        edge_from = []
        edge_to = []
        # backbone - undirected edge for now
        node_left = range(0, len(seq) - 1)
        node_right = range(1, len(seq))
        edge_from.extend(node_left)
        edge_to.extend(node_right)
        edge_from.extend(node_right)
        edge_to.extend(node_left)
        assert len(edge_from) == len(edge_to)
        n_edge_1 = len(edge_from)  # number of 'backbone' edges
        # hydrogen bond candidates - undirected edge
        # for all predicted stem bbs
        for idx_left, idx_right in stem_bb_bps:
            edge_from.append(idx_left)
            edge_to.append(idx_right)
            edge_from.append(idx_right)
            edge_to.append(idx_left)
        assert len(edge_from) == len(edge_to)
        n_edge_2 = len(edge_from) - n_edge_1  # number of 'hydrogen bond' edges
        edge_index = torch.tensor([edge_from, edge_to], dtype=torch.long)

        # edge feature, 0 for "backbone", 1 for "hydrogen bond"
        edge_attr = torch.LongTensor([0] * n_edge_1 + [1] * n_edge_2).unsqueeze(1)

        y, m = fn_make_target(seq, stem_bb_bps, target_bps)
        # # edge-level target, encoded as 2D binary matrix, with masking
        # # binary matrix of size lxl
        # y = np.zeros((len(seq), len(seq)))
        # y[tuple(zip(*target_bps))] = 1
        # # mask: locations with 0 are don't-cares
        # # we only backprop from edges in pred stem bbs
        # m = np.zeros((len(seq), len(seq)))
        # m[tuple(zip(*stem_bb_bps))] = 1

        # make data point
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr,
                    y=y, m=m)

        data_list.append(data)
    return data_list


loss_bce = torch.nn.BCELoss(reduction='none')


def masked_loss_bce(x, y, m):
    # L x L
    l = loss_bce(x, y)
    n_valid_output = torch.sum(m)
    loss_spatial_sum = torch.sum(torch.mul(l, m))
    loss_spatial_mean = loss_spatial_sum / n_valid_output
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    return torch.mean(loss_batch_mean)


def get_logger(log_file):
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger
