import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
import torch_geometric
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import random
from tqdm import tqdm
from typing import Optional
from torch_geometric.typing import Adj, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, Parameter, GRUCell
from torch_geometric.utils import softmax
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot, zeros


def make_dataset(df):
    data_list = []

    for _, row in df.iterrows():
        seq = row['seq']
        stem_bb_bps = row['stem_bb_bps']
        target_bps = row['target_bps']

        # use integer encoding for now
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U',
                                                                                                          '4').replace(
            'N', '0')
        tmp = np.asarray(list(map(int, list(seq))), dtype=np.int16)
        # one-hot
        # TODO instead of 1-hot encode each base, we can 1-hot encode the local k-mer
        node_features = np.zeros((len(seq), 4))
        node_features[np.arange(len(seq)), tmp - 1] = 1
        node_features = torch.from_numpy(node_features).float()

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

        # edge-level target, encoded as 2D binary matrix, with masking
        # binary matrix of size lxl
        y = np.zeros((len(seq), len(seq)))
        y[tuple(zip(*target_bps))] = 1
        # mask: locations with 0 are don't-cares
        # we only backprop from edges in pred stem bbs
        m = np.zeros((len(seq), len(seq)))
        m[tuple(zip(*stem_bb_bps))] = 1

        # make data point
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr,
                    y=y, m=m)

        data_list.append(data)
    return data_list


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


class Net(torch.nn.Module):
    def __init__(self, num_hids):
        super(Net, self).__init__()
        # FIXME layers are hard-coded here
        # graph conv layers for node message passing
        self.gcn = [GATEConv(4, num_hids[0], 1)]
        for num_hid_prev, num_hid in zip(num_hids[:-1], num_hids[1:]):
            self.gcn.append(GATEConv(num_hid_prev, num_hid, 1))

        # activations
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.Sigmoid()

        # node-node NN
        # this is the channel wide (thus 1x1) 2D conv net
        # that effectively applies a weight-tied fully connected NN to each node pair
        # FIXME hard-coded n hid
        self.node_pair_conv1 = torch.nn.Conv2d((np.sum(num_hids) + 4) * 2, 20, kernel_size=1, stride=1,
                                               padding=0)
        self.node_pair_conv2 = torch.nn.Conv2d(20, 1, kernel_size=1, stride=1,
                                               padding=0)

    def forward(self, data):
        x_all = [data.x]

        x = data.x
        for gcn in self.gcn:
            x = self.act1(gcn(x, data.edge_index, data.edge_attr))
            x_all.append(x)

        x = torch.cat(x_all, axis=-1) # concat node features from all layers, including input

        # outer concat: L x L x 2f
        x1 = x.unsqueeze(1).repeat(1, x.size(0), 1)
        x2 = x.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat([x1, x2], axis=2)

        # FC along last (channel) dim
        # note that conv_2d expects Input: (N, C_{in}, H_{in}, W_{in})
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = self.act1(self.node_pair_conv1(x))
        x = self.act2(self.node_pair_conv2(x))
        return x.squeeze()  # this is L x L


loss_b = torch.nn.BCELoss(reduction='none')


def masked_loss_b(x, y, m):
    # L x L
    l = loss_b(x, y)
    n_valid_output = torch.sum(m)
    loss_spatial_sum = torch.sum(torch.mul(l, m))
    loss_spatial_mean = loss_spatial_sum / n_valid_output
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    return torch.mean(loss_batch_mean)


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


def main(input_data, training_proportion, learning_rate, num_hids, epochs):
    df = pd.read_pickle(input_data)

    # train/valid dataset
    df = df.sample(frac=1)
    n_tr = int(len(df) * training_proportion)
    assert n_tr < len(df)
    df_tr = df[:n_tr]
    df_va = df[n_tr:]
    data_list_tr = make_dataset(df_tr)
    data_list_va = make_dataset(df_va)

    # init model
    model = Net(num_hids)
    model.train()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # shuffle training dataset
        random.shuffle(data_list_tr)

        # training dataset
        loss_all = []
        auc_all = []
        model.train()
        for data_idx, data in tqdm(enumerate(data_list_tr)):
            y = torch.from_numpy(data.y).float()
            m = torch.from_numpy(data.m).float()
            optimizer.zero_grad()
            pred = model(data)
            loss = masked_loss_b(pred, y, m)
            loss_all.append(loss.item())
            auc, prc = roc_prc(y, pred, m)
            auc_all.append(auc)
            # FIXME we're backprop after each example, should probably accumulate gradient
            loss.backward()
            optimizer.step()
        print("Epoch {}, training, mean loss {}, mean AUC {}".format(epoch, np.nanmean(loss_all), np.nanmean(auc_all)))

        # validation dataset
        loss_all = []
        auc_all = []
        model.eval()
        for data in data_list_va:
            y = torch.from_numpy(data.y).float()
            m = torch.from_numpy(data.m).float()
            optimizer.zero_grad()
            pred = model(data)
            loss = masked_loss_b(pred, y, m)
            loss_all.append(loss.item())
            auc, prc = roc_prc(y, pred, m)
            auc_all.append(auc)
        print("Epoch {}, testing, mean loss {}, mean AUC {}".format(epoch, np.nanmean(loss_all), np.nanmean(auc_all)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='Path to dataset')
    parser.add_argument('--training_proportion', type=float, default=0.9, help='proportion of training data')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learinng rate')
    parser.add_argument('--hid', type=int, nargs='+', default=[20, 20, 20], help='number of hidden units for each layer')
    parser.add_argument('--epochs', type=int, default=10, help='learinng rate')
    args = parser.parse_args()
    assert  0 < args.training_proportion < 1
    main(args.input_data, args.training_proportion, args.learning_rate, args.hid, args.epochs)





