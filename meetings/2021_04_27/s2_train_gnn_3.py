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
from util_s2_gnn import GATEConv, masked_accuracy, make_dataset, masked_loss_sce, get_logger


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


class Net(torch.nn.Module):
    def __init__(self, num_hids, k, embed_dim):
        super(Net, self).__init__()
        # embedding layer
        self.node_embedding = torch.nn.Embedding(num_embeddings=5**k, embedding_dim=embed_dim)
        self.embed_dim = embed_dim
        # graph conv layers for node message passing
        self.gcn = [GATEConv(embed_dim, num_hids[0], 1)]
        for num_hid_prev, num_hid in zip(num_hids[:-1], num_hids[1:]):
            self.gcn.append(GATEConv(num_hid_prev, num_hid, 1))

        # activations
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.Softmax(dim=1)  # softmax along second dim (first is also fine)

        # node-node NN
        # this is the channel wide (thus 1x1) 2D conv net
        # that effectively applies a weight-tied fully connected NN to each node pair
        # FIXME hard-coded n hid
        self.node_pair_conv1 = torch.nn.Conv2d((np.sum(num_hids) + self.embed_dim) * 2, 20, kernel_size=1, stride=1,
                                               padding=0)
        self.node_pair_conv2 = torch.nn.Conv2d(20, 1, kernel_size=1, stride=1,
                                               padding=0)

    def forward(self, data):
        # x_all = [data.x]

        x = data.x
        x = self.node_embedding(x)
        x_all = [x]
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
        x = self.node_pair_conv2(x)

        # apply mask, so that we only take softmax for edges that's connected to each node
        x = x * torch.from_numpy(data.m).float()
        x = x.squeeze()  # this is L x L
        # softmax, since x is symmetric, this should also be symmetric (TODO check that!)
        x = self.act2(x)
        return x


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
        # auc_all = []
        acc_all = []
        model.train()

        # batching - thanks Andrew!
        loss = 0
        optimizer.zero_grad()

        for data_idx, data in tqdm(enumerate(data_list_tr)):
            y = torch.from_numpy(data.y).float()
            m = torch.from_numpy(data.m).float()
            optimizer.zero_grad()
            pred = model(data)
            # this should work (see gradient_check.ipynb)
            # using built-in loss, since the prediction is already masked
            # we're masking out those nodes without any hydrogen bond candidate connections
            loss += masked_loss_sce(pred, y.argmax(dim=1), m)
            # auc, prc = roc_prc(y, pred.detach(), m)
            # auc_all.append(auc)
            acc = masked_accuracy(y, pred.detach(), m)
            acc_all.append(acc)

            if (data_idx + 1) % batch_size == 0:  # TODO deal with last batch
                loss /= batch_size
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_all.append(loss.item())
                loss = 0

        logger.info("Epoch {}, training, mean loss {}, mean accuracy {}".format(epoch, np.nanmean(loss_all), np.nanmean(acc_all)))

        # validation dataset
        loss_all = []
        auc_all = []
        model.eval()
        for data in data_list_va:
            y = torch.from_numpy(data.y).float()
            m = torch.from_numpy(data.m).float()
            optimizer.zero_grad()
            pred = model(data)
            loss = masked_loss_sce(pred, y.argmax(dim=1), m)
            loss_all.append(loss.item())
            # auc, prc = roc_prc(y, pred, m)
            # auc_all.append(auc)
            acc = masked_accuracy(y, pred.detach(), m)
            acc_all.append(acc)

        logger.info("Epoch {}, testing, mean loss {}, mean accuracy {}".format(epoch, np.nanmean(loss_all), np.nanmean(acc_all)))


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





