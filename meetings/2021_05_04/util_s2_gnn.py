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
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot, zeros


class SimpleNodeEdgeConv(MessagePassing):
    # simplify GATEConv
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super(SimpleNodeEdgeConv, self).__init__(aggr='add')
        self.lin = Linear(in_channels + edge_dim, out_channels)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # FIXME how to add self loop, what about edge feature?
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # no edge features for these?
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor) -> Tensor:
        return F.leaky_relu_(self.lin(torch.cat([x_j, edge_attr], dim=-1)))


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


def masked_accuracy(x, y, m):
    # true, score, mask
    x = x * m
    y = y * m
    x2 = x.argmax(axis=1)
    y2 = y.argmax(axis=1)
    #TODO should ignore entries with all masks - right now this is overestimate?
    return (x2 == y2).sum()/len(x2)


def one_hot_single_base(seq):
    seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U',
                                                                                                      '4').replace(
        'N', '0')
    tmp = np.asarray(list(map(int, list(seq))), dtype=np.int16)
    # one-hot
    x = np.zeros((len(seq), 4))
    x[np.arange(len(seq)), tmp - 1] = 1
    return torch.from_numpy(x).float()


def make_edge_target(edge_index, stem_bb_bps, target_bps):
    y = np.zeros(edge_index.size(1))
    m = np.zeros(edge_index.size(1))

    # TODO better to vectorize this
    for i in range(edge_index.size(1)):
        edge_idx = (edge_index[0, i].item(), edge_index[1, i].item())
        if edge_idx in stem_bb_bps:
            m[i] = 1
        if edge_idx in target_bps:
            assert edge_idx in stem_bb_bps
            y[i] = 1
    return y, m


def make_dataset(df, fn_make_target, fn_encode_seq=one_hot_single_base, edge_feature='binary',
                 fn_make_target_edge=make_edge_target, include_s1_feature=False, s1_feature_dim=None):
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

        if include_s1_feature:
            s1_edge_feat = []

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
        if include_s1_feature:  # all 0 features for backbond edges
            s1_edge_feat = [[0] * s1_feature_dim for _ in range(n_edge_1)]

        # hydrogen bond candidates - undirected edge
        # for all predicted stem bbs
        for idx_left, idx_right in stem_bb_bps:
            edge_from.append(idx_left)
            edge_to.append(idx_right)
            edge_from.append(idx_right)
            edge_to.append(idx_left)
            if include_s1_feature:  # append twice for both direction
                s1_edge_feat.append(row['stem_bb_bps_features'][(idx_left, idx_right)])
                s1_edge_feat.append(row['stem_bb_bps_features'][(idx_left, idx_right)])
        assert len(edge_from) == len(edge_to)
        n_edge_2 = len(edge_from) - n_edge_1  # number of 'hydrogen bond' edges
        edge_index = torch.tensor([edge_from, edge_to], dtype=torch.long)

        # edge feature
        if edge_feature == 'binary':
            # 0 for "backbone", 1 for "hydrogen bond"
            edge_attr = torch.LongTensor([0] * n_edge_1 + [1] * n_edge_2).unsqueeze(1)
        elif edge_feature == 'one_hot':
            # [1, 0] for "backbone", [0, 1] for "hydrogen bond"
            edge_attr = torch.cat([torch.LongTensor([1] * n_edge_1 + [0] * n_edge_2).unsqueeze(1),
                                   torch.LongTensor([0] * n_edge_1 + [1] * n_edge_2).unsqueeze(1)],
                                  dim=1)
        else:
            raise NotImplementedError

        if include_s1_feature:
            s1_edge_feat = np.asarray(s1_edge_feat)
            s1_edge_feat = torch.from_numpy(s1_edge_feat).float()
            edge_attr = torch.cat([edge_attr, s1_edge_feat], dim=1)

        # 2D target and mask
        y, m = fn_make_target(seq, stem_bb_bps, target_bps)
        # equivalent edge-leve 1D target and mask
        ye, me = fn_make_target_edge(edge_index, stem_bb_bps, target_bps)

        # make data point
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr,
                    y=y, m=m, y_edge=ye, m_edge=me)

        data_list.append(data)
    return data_list


loss_bce = torch.nn.BCELoss(reduction='none')
loss_sce = torch.nn.CrossEntropyLoss(reduction='none')


def masked_loss_bce(x, y, m):
    # L x L
    l = loss_bce(x, y)
    n_valid_output = torch.sum(m)
    loss_spatial_sum = torch.sum(torch.mul(l, m))
    loss_spatial_mean = loss_spatial_sum / n_valid_output
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    return torch.mean(loss_batch_mean)


def masked_loss_sce(x, y, m):
    # mask out gradient for those nodes where all connections are masked
    l = loss_sce(x, y)
    m, _ = torch.max(m, dim=1)  # entries with 0 will be masked  # torch.max returns tuple
    n_valid_output = torch.sum(m)
    loss_sum = torch.sum(torch.mul(l, m))
    loss_mean = loss_sum / n_valid_output
    return loss_mean


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
