"""
predict binary label for each node
"""
import argparse
import pickle
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import logging
import random
from tqdm import tqdm
import torch
from util_s2_gnn import GATEConv, compute_auc_f1, make_dataset, masked_loss_bce, get_logger


def make_target(seq, stem_bb_bps, one_idx):
    # edge-level target, encoded as 2D binary matrix, with masking
    # binary matrix of size lxl
    y = np.zeros((len(seq), len(seq)))
    y[tuple(one_idx)] = 1
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
        self.gcn = [GATEConv(embed_dim, num_hids[0], 1+6)]
        for num_hid_prev, num_hid in zip(num_hids[:-1], num_hids[1:]):
            self.gcn.append(GATEConv(num_hid_prev, num_hid, 1+6))
        self.gcn = torch.nn.ModuleList(self.gcn)

        # activations
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.Sigmoid()

        # node-node NN
        # this is the channel wide (thus 1x1) 2D conv net
        # that effectively applies a weight-tied fully connected NN to each node pair
        # FIXME hard-coded n hid
        self.node_fc1 = torch.nn.Linear(np.sum(num_hids) + self.embed_dim, 20)
        self.node_fc2 = torch.nn.Linear(20, 1)

    def forward(self, data):
        # x_all = [data.x]

        x = data.x
        x = self.node_embedding(x)
        x_all = [x]
        for gcn in self.gcn:
            x = self.act1(gcn(x, data.edge_index, data.edge_attr))
            x_all.append(x)

        x = torch.cat(x_all, axis=-1) # concat node features from all layers, including input

        # node output
        x = self.act1(self.node_fc1(x))
        x = self.act2(self.node_fc2(x))  # Lx1
        return x.squeeze()  # L

        # # outer concat: L x L x 2f
        # x1 = x.unsqueeze(1).repeat(1, x.size(0), 1)
        # x2 = x.unsqueeze(0).repeat(x.size(0), 1, 1)
        # x = torch.cat([x1, x2], axis=2)
        #
        # # FC along last (channel) dim
        # # note that conv_2d expects Input: (N, C_{in}, H_{in}, W_{in})
        # x = x.permute(2, 0, 1).unsqueeze(0)
        # x = self.act1(self.node_pair_conv1(x))
        # x = self.act2(self.node_pair_conv2(x))
        # return x.squeeze()  # this is L x L


def main(input_data, training_proportion, learning_rate, num_hids, epochs, batch_size,
         log_file, kmer, embed_dim, debug=False):
    logger = get_logger(log_file)

    df = pd.read_pickle(input_data)

    # train/valid dataset
    df = df.sample(frac=1)
    n_tr = int(len(df) * training_proportion)
    assert n_tr < len(df)
    df_tr = df[:n_tr]
    df_va = df[n_tr:]
    data_list_tr = make_dataset(df_tr, make_target,
                                lambda x: kmer_int(x, k=kmer),
                                include_s1_feature=True, s1_feature_dim=6)
    data_list_va = make_dataset(df_va, make_target,
                                lambda x: kmer_int(x, k=kmer),
                                include_s1_feature=True, s1_feature_dim=6)

    # init model
    model = Net(num_hids, k=kmer, embed_dim=embed_dim)
    model.train()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # loss
    loss_bce = torch.nn.BCELoss()

    for epoch in range(epochs):
        # shuffle training dataset
        random.shuffle(data_list_tr)

        # training dataset
        loss_all = []
        # auroc_all = []
        # auprc_all = []
        # f1s_all = []
        model.train()

        # batching - thanks Andrew!
        loss = 0
        optimizer.zero_grad()

        for data_idx, data in tqdm(enumerate(data_list_tr)):
            y = torch.from_numpy(data.y_node).float()
            m = torch.from_numpy(data.m).float()
            optimizer.zero_grad()
            pred = model(data)
            # this should work (see gradient_check.ipynb)
            loss += loss_bce(pred, y)
            # auc, prc, f1s = compute_auc_f1(y, pred.detach(), m)
            # auroc_all.append(auc)
            # auprc_all.append(prc)
            # f1s_all.append(f1s)

            if (data_idx + 1) % batch_size == 0:  # TODO deal with last batch
                loss /= batch_size
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_all.append(loss.item())
                loss = 0

        # logger.info("Epoch {}, training, mean loss {}, mean auROC {}, mean auPRC {}, mean f1 {}".format(epoch, np.nanmean(loss_all),
        #                                                                                     np.nanmean(auroc_all),
        #                                                                                     np.nanmean(auprc_all),
        #                                                                                     np.nanmean(f1s_all,
        #                                                                                                axis=0)))
        logger.info("Epoch {}, training, mean loss {}".format(epoch, np.nanmean(loss_all)))

        # validation dataset
        loss_all = []
        # auroc_all = []
        # auprc_all = []
        # f1s_all = []
        # data_debug = []
        model.eval()
        for data in data_list_va:
            y = torch.from_numpy(data.y_node).float()
            m = torch.from_numpy(data.m).float()
            # optimizer.zero_grad()
            pred = model(data)
            loss = loss_bce(pred, y)
            loss_all.append(loss.item())
            # auc, prc, f1s = compute_auc_f1(y, pred, m)
            # auroc_all.append(auc)
            # auprc_all.append(prc)
            # f1s_all.append(f1s)

            # data_debug.append({
            #     'x': data.x.detach().numpy(),
            #     'edge_index': data.edge_index.detach().numpy(),
            #     'edge_attr': data.edge_attr.detach().numpy(),
            #     'y': data.y,
            #     'm': data.m,
            #     'yp': pred.detach().numpy(),
            # })

        # logger.info("Epoch {}, validation, mean loss {}, mean auROC {}, mean auPRC {}, mean f1 {}".format(epoch,
        #                                                                                                 np.nanmean(
        #                                                                                                     loss_all),
        #                                                                                                 np.nanmean(
        #                                                                                                     auroc_all),
        #                                                                                                 np.nanmean(
        #                                                                                                     auprc_all),
        #                                                                                                 np.nanmean(
        #                                                                                                     f1s_all,
        #                                                                                                     axis=0)))
        logger.info("Epoch {}, validation, mean loss {}".format(epoch, np.nanmean(loss_all)))

        # FIXME hacky way to save model
        if (epoch + 1) % (max(1, epochs//10)) == 0:
            # _model_path = os.path.join(out_dir, 'model_ckpt_ep_{}.pth'.format(epoch))
            _model_path = log_file.replace('log', 'model_ckpt_ep_{}.pth'.format(epoch))
            torch.save(model.state_dict(), _model_path)
            logging.info("Model checkpoint saved at: {}".format(_model_path))
        # # if debug, also save prediction on validation set
        # if debug and (epoch + 1) % (max(1, epochs//10)) == 0:
        #     data_debug_export_path = log_file.replace('log', 'pred_va_ep_{}.pkl'.format(epoch))
        #     with open(data_debug_export_path, 'wb') as f:
        #         pickle.dump(data_debug, f, protocol=pickle.HIGHEST_PROTOCOL)
        #     logging.info("Prediction saved at: {}".format(data_debug_export_path))


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
    parser.add_argument('--debug', action='store_true', help='Set this to export validation set prediction for debugging')

    args = parser.parse_args()
    assert  0 < args.training_proportion < 1
    main(args.input_data, args.training_proportion, args.learning_rate, args.hid,
         args.epochs, args.batch_size, args.log, args.kmer, args.embed_dim, args.debug)





