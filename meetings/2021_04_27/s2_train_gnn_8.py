import argparse
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import logging
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
from util_s2_gnn import GATEConv, roc_prc, make_dataset, masked_loss_bce, get_logger


def make_target(seq, stem_bb_bps, target_bps):
    # edge-level target, encoded as 2D binary matrix, with masking
    # binary matrix of size lxl
    y = np.zeros((len(seq), len(seq)))
    y[tuple(zip(*target_bps))] = 1
    # mask: locations with 0 are don't-cares
    # we only backprop from edges in pred stem bbs
    m = np.zeros((len(seq), len(seq)))
    m[tuple(zip(*stem_bb_bps))] = 1
    return y, m


def kmer_int(seq, k):
    assert k % 2 == 1  # to make life easier! even padding on both side
    n_pad = (k-1)//2
    seq_len = len(seq)  # save length before padding
    seq = "N" * n_pad + seq + "N" * n_pad
    seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U',
                                                                                                      '4').replace(
        'N', '0')  # len = L + k - 1
    x = np.asarray(list(map(int, list(seq))), dtype=np.int16)  # 1D array of L + k - 1
    # make overlapping sliding window of size k
    x = sliding_window_view(x, k)  # L x k
    # convert each k-digit: represent k-digit base-5 number  (base 5 since we take into account N due to padding)
    b = np.array([5 ** i for i in range(k)])
    x = np.sum(x * b, axis=1)
    return torch.LongTensor(x)
    # # one-hot
    # data = np.zeros((seq_len, 5**k))
    # data[np.arange(seq_len), x] = 1
    # return data


class EdgeUpdate(torch.nn.Module):
    # "batch mode" edge update

    # TODO note that we're updating undirected edges twices
    # TODO backbone edges are also being updated

    def __init__(self, node_dim, edge_in_dim, edge_out_dim):
        super(EdgeUpdate, self).__init__()
        self.lin = Linear(node_dim + edge_in_dim, edge_out_dim)
        self.act = torch.nn.ReLU()

    def forward(self, node_feature, edge_index, edge_feature):
        # outer concat node feature
        # sum over node feature to be invariant to the two nodes
        node_feature1 = node_feature.unsqueeze(1).repeat(1, node_feature.size(0), 1)
        node_feature2 = node_feature.unsqueeze(0).repeat(node_feature.size(0), 1, 1)
        # each entry is the sum of node feature of node i and j
        node_feature_sum = node_feature1 + node_feature2  # n_node x n_node x n_node_dim

        # select entries that correspond to edges
        # note that each edge is directed (undirected edge has two entries)
        node_pair_feature_edge = node_feature_sum[(edge_index[0], edge_index[1])]  # n_edge x (n_node_dim*2)

        all_feature = torch.cat([node_pair_feature_edge, edge_feature], axis=1)  # n_edge x (n_node_dim*2 + edge_in_dim)
        return self.act(self.lin(all_feature))


class Net(torch.nn.Module):
    def __init__(self, num_hids, k, embed_dim):
        super(Net, self).__init__()
        # embedding layer
        self.node_embedding = torch.nn.Embedding(num_embeddings=5**k, embedding_dim=embed_dim)
        self.embed_dim = embed_dim
        # graph conv layers for node message passing
        # simple linear+relu layer for edge message passing
        self.gcn = [GATEConv(embed_dim, num_hids[0], 1)]
        self.edg = [EdgeUpdate(num_hids[0], 1, num_hids[0])]
        for num_hid_prev, num_hid in zip(num_hids[:-1], num_hids[1:]):
            self.gcn.append(GATEConv(num_hid_prev, num_hid, num_hid_prev))
            self.edg.append(EdgeUpdate(num_hid, num_hid_prev, num_hid))

        # activations
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.Sigmoid()

        # FC on edge feature
        # FIXME hard-coded hid layer
        self.edge_lin1 = torch.nn.Linear(np.sum(num_hids) + 1, 20)
        self.edge_lin2 = torch.nn.Linear(20, 1)

        # # node-node NN
        # # this is the channel wide (thus 1x1) 2D conv net
        # # that effectively applies a weight-tied fully connected NN to each node pair
        # # FIXME hard-coded n hid
        # self.node_pair_conv1 = torch.nn.Conv2d((np.sum(num_hids) + self.embed_dim) * 2, 20, kernel_size=1, stride=1,
        #                                        padding=0)
        # self.node_pair_conv2 = torch.nn.Conv2d(20, 1, kernel_size=1, stride=1,
        #                                        padding=0)

    def forward(self, data):
        # x_all = [data.x]

        x_node = data.x
        x_node = self.node_embedding(x_node)
        # x_node_all = [x_node]

        x_edge = data.edge_attr
        x_edge_all = [x_edge]  # TODO do we need the input edge feature

        for gcn, edg in zip(self.gcn, self.edg):
            x_node = self.act1(gcn(x_node, data.edge_index, x_edge))
            x_edge = edg(x_node, data.edge_index, x_edge)
            # x_node_all.append(x_node)
            x_edge_all.append(x_edge)

        # x_node = torch.cat(x_node_all, axis=-1) # concat node features from all layers, including input
        x_edge = torch.cat(x_edge_all, axis=-1) # concat edge features from all layers, including input

        x = self.act1(self.edge_lin1(x_edge))
        x = self.act2(self.edge_lin2(x))
        return x.squeeze()  # n_edge x 1?

        # # outer concat: L x L x 2f
        # x1 = x_node.unsqueeze(1).repeat(1, x_node.size(0), 1)
        # x2 = x_node.unsqueeze(0).repeat(x_node.size(0), 1, 1)
        # x_node = torch.cat([x1, x2], axis=2)
        #
        # # TODO indexing using edge index
        # # TODO use both node and edge embedding for final output
        #
        # # FC along last (channel) dim
        # # note that conv_2d expects Input: (N, C_{in}, H_{in}, W_{in})
        # x_node = x_node.permute(2, 0, 1).unsqueeze(0)
        # x_node = self.act1(self.node_pair_conv1(x_node))
        # x_node = self.act2(self.node_pair_conv2(x_node))
        # return x_node.squeeze()  # this is L x L


def main(input_data, training_proportion, learning_rate, num_hids, epochs, batch_size, log_file, kmer, embed_dim):
    logger = get_logger(log_file)

    df = pd.read_pickle(input_data)

    # train/valid dataset
    df = df.sample(frac=1)
    n_tr = int(len(df) * training_proportion)
    assert n_tr < len(df)
    df_tr = df[:n_tr]
    df_va = df[n_tr:]
    data_list_tr = make_dataset(df_tr, make_target, lambda x: kmer_int(x, k=kmer))
    data_list_va = make_dataset(df_va, make_target, lambda x: kmer_int(x, k=kmer))

    # init model
    model = Net(num_hids, k=kmer, embed_dim=embed_dim)
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

        # batching - thanks Andrew!
        loss = 0
        optimizer.zero_grad()

        for data_idx, data in tqdm(enumerate(data_list_tr)):
            y = torch.from_numpy(data.y_edge).float()
            m = torch.from_numpy(data.m_edge).float()
            optimizer.zero_grad()
            pred = model(data)
            # this should work (see gradient_check.ipynb)
            loss += masked_loss_bce(pred, y, m)
            auc, prc = roc_prc(y, pred.detach(), m)
            auc_all.append(auc)

            if (data_idx + 1) % batch_size == 0:  # TODO deal with last batch
                loss /= batch_size
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_all.append(loss.item())
                loss = 0

        logger.info("Epoch {}, training, mean loss {}, mean AUC {}".format(epoch, np.nanmean(loss_all), np.nanmean(auc_all)))

        # validation dataset
        loss_all = []
        auc_all = []
        model.eval()
        for data in data_list_va:
            y = torch.from_numpy(data.y_edge).float()
            m = torch.from_numpy(data.m_edge).float()
            optimizer.zero_grad()
            pred = model(data)
            loss = masked_loss_bce(pred, y, m)
            loss_all.append(loss.item())
            auc, prc = roc_prc(y, pred, m)
            auc_all.append(auc)
        logger.info("Epoch {}, testing, mean loss {}, mean AUC {}".format(epoch, np.nanmean(loss_all), np.nanmean(auc_all)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='Path to dataset')
    parser.add_argument('--training_proportion', type=float, default=0.9, help='proportion of training data')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learinng rate')
    parser.add_argument('--hid', type=int, nargs='+', default=[20, 20, 20], help='number of hidden units for each layer')
    parser.add_argument('--epochs', type=int, default=10, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--log', type=str, help='output log file')
    parser.add_argument('--kmer', type=int, default=3, help='kmer')
    parser.add_argument('--embed_dim', type=int, default=20, help='embedding dim for kmer')

    args = parser.parse_args()
    assert  0 < args.training_proportion < 1
    main(args.input_data, args.training_proportion, args.learning_rate, args.hid,
         args.epochs, args.batch_size, args.log, args.kmer, args.embed_dim)





