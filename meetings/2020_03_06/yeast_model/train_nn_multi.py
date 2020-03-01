import os
import logging
import argparse
import pandas as pd
import numpy as np
# import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
# TODO also make plots


def add_column(df, output_col, input_cols, func):
    # make a tuple of values of the requested input columns
    input_values = tuple(df[x].values for x in input_cols)

    # transpose to make a list of value tuples, one per row
    args = zip(*input_values)

    # Attach a progress bar if required
    # if pbar:
    #     args = tqdm(args, total=len(df))
    #     args.set_description("Processing %s" % output_col)

    # evaluate the function to generate the values of the new column
    output_values = [func(*x) for x in args]

    # make a new dataframe with the new column added
    columns = {x: df[x].values for x in df.columns}
    columns[output_col] = output_values
    return pd.DataFrame(columns)


def load_data(f_name):
    # load raw pair wise data from cell map paper
    df = pd.read_csv(f_name,
                     sep='\t')[['Query Strain ID', 'Array Strain ID',
                                'Query single mutant fitness (SMF)', 'Array SMF',
                                'Genetic interaction score (ε)']].rename(columns={
        'Query Strain ID': 's1',
        'Array Strain ID': 's2',
        'Query single mutant fitness (SMF)': 'f1',
        'Array SMF': 'f2',
        'Genetic interaction score (ε)': 'interaction',
        'Double mutant fitness': 'fitness',
    })
    return df


def get_gene_id(strain_id):
    # extra gene ID from strain ID
    gene_id, _ = strain_id.split('_')
    return gene_id


def encode_x(g1, g2, gene_id2idx):
    return [gene_id2idx[g1], gene_id2idx[g2]]


# class MyDataSet(Dataset):
#
#     def __init__(self, x, y, num_genes):
#         self.num_genes = num_genes
#         assert x.shape[0] == y.shape[0]
#         self.len = x.shape[0]
#         self.x = x
#         # add new axis if needed
#         if len(y.shape) == 1:
#             self.y = y[:, np.newaxis]
#         else:
#             self.y = y
#
#     def __getitem__(self, index):
#         _x = self.x[index, :]
#         assert len(_x) == 2
#         # encode
#         x = np.ones(self.num_genes)
#         x[_x[0]] = 0
#         x[_x[1]] = 0
#         return torch.from_numpy(x).float(), torch.from_numpy(self.y[index]).float()
#
#     def __len__(self):
#         return self.len


class MyDataSet(Dataset):

    def __init__(self, x, y, num_genes):
        self.num_genes = num_genes
        assert x.shape[0] == y.shape[0]
        self.len = x.shape[0]
        self.x = x
        # # add new axis if needed
        # if len(y.shape) == 1:
        #     self.y = y[:, np.newaxis]
        # else:
        #     self.y = y
        assert y.shape[1] == 2  # 2 outputs
        self.y_gi = y[:, 0]
        self.y_fitness = y[:, 1]

    def __getitem__(self, index):
        _x = self.x[index, :]
        assert len(_x) == 2
        # encode
        xd = np.ones(self.num_genes)
        x1 = np.ones(self.num_genes)
        x2 = np.ones(self.num_genes)
        xd[_x[0]] = 0
        xd[_x[1]] = 0  # 2-hot encoding for double KO
        x1[_x[0]] = 0  # 1-hot encoding for first gene KO
        x2[_x[1]] = 0  # 1-hot encoding for second gene KO
        return torch.from_numpy(xd).float(), torch.from_numpy(x1).float(), torch.from_numpy(
            x2).float(), torch.from_numpy(self.y_fitness[index]).float(), torch.from_numpy(self.y_gi[index]).float()

    def __len__(self):
        return self.len


def make_model(num_input, hid_sizes):
    modules = []
    sizes = [num_input] + hid_sizes
    for last_size, this_size in zip(sizes[:-1], sizes[1:]):
        modules.append(torch.nn.Linear(last_size, this_size))
        modules.append(torch.nn.LeakyReLU())
        modules.append(torch.nn.BatchNorm1d(this_size))
    # add last layer
    modules.append(torch.nn.Linear(sizes[-1], 1))
    model = nn.Sequential(*modules)
    return model


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


def main(path_data, hid_sizes, n_epoch):
    # load data
    logging.info("Loading dataset: {}".format(path_data))
    df = []
    for _p in path_data:
        df.append(load_data(_p))
    df = pd.concat(df)

    # extract gene ID
    df = add_column(df, 'g1', ['s1'], get_gene_id)
    df = add_column(df, 'g2', ['s2'], get_gene_id)

    # take median of examples with the same gene pair, so that we don't have duplicates
    df = df[['g1', 'g2', 'interaction', 'fitness']].groupby(by=['g1', 'g2'], as_index=False).agg('median')
    # gene pair should be unique now
    # TODO check it

    # shuffle rows
    df = df.sample(frac=1).reset_index(drop=True)
    # subset
    _n_tr = int(len(df) * 0.8)
    df_tr = df[:_n_tr]
    df_ts = df[_n_tr:]
    # make sure genes are the same (for now drop rows that are not)
    genes_tr = set(df_tr.g1.unique().tolist() + df_tr.g2.unique().tolist())
    genes_ts = set(df_ts.g1.unique().tolist() + df_ts.g2.unique().tolist())
    genes_intersection = genes_tr.intersection(genes_ts)
    # print(len(genes_tr), len(genes_ts), len(genes_intersection))
    df_tr = df_tr[df_tr['g1'].isin(genes_intersection)]
    df_tr = df_tr[df_tr['g2'].isin(genes_intersection)]
    df_ts = df_ts[df_ts['g1'].isin(genes_intersection)]
    df_ts = df_ts[df_ts['g2'].isin(genes_intersection)]
    # print(len(df_tr), len(df_ts))
    logging.info("Number of training examples: {}, test: {}".format(len(df_tr), len(df_ts)))

    # for data encoding
    gene_ids = tuple(genes_intersection)
    num_genes = len(gene_ids)
    logging.info("Number of genes: {}".format(num_genes))
    # make it faster by create gene_id -> gene_idx mapping
    gene_id2idx = {x: i for i, x in enumerate(gene_ids)}
    df_tr = add_column(df_tr, 'x', ['g1', 'g2'], lambda g1, g2: encode_x(g1, g2, gene_id2idx))
    df_ts = add_column(df_ts, 'x', ['g1', 'g2'], lambda g1, g2: encode_x(g1, g2, gene_id2idx))

    # get data
    x_tr = np.asarray(df_tr['x'].to_list())
    y_tr = np.asarray(df_tr[['interaction', 'fitness']].to_list())

    x_ts = np.asarray(df_ts['x'].to_list())
    y_ts = np.asarray(df_ts[['interaction', 'fitness']].to_list())
    # print(x_tr.shape, y_tr.shape, x_ts.shape, y_ts.shape)
    assert x_tr.shape[0] == y_tr.shape[0]
    assert x_ts.shape[0] == y_ts.shape[0]

    # train + test data handler
    batch_size = 1000
    data_tr_loader = DataLoader(dataset=MyDataSet(x_tr, y_tr, num_genes),
                                batch_size=batch_size, shuffle=True)
    data_ts_loader = DataLoader(dataset=MyDataSet(x_ts, y_ts, num_genes),
                                batch_size=batch_size, shuffle=True)

    # model
    model = make_model(num_genes, hid_sizes)
    logging.info("model: \n{}".format(model))
    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # naive guess is the mean of training target value
    yp_naive = np.mean(y_tr)
    # calculate loss on naive guess
    # hard-coded to be MSE
    loss_naive_training = np.mean((yp_naive - y_tr)**2)
    loss_naive_test = np.mean((yp_naive - y_ts)**2)
    logging.info("Naive guess: {}, training loss: {}, test loss: {}".format(yp_naive, loss_naive_training, loss_naive_test))

    # training TODO GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # inital test performance
    with torch.set_grad_enabled(False):
        for xtd, xt1, xt2, ytd, ytgi in data_ts_loader:
            xtd = xtd.to(device)
            xt1 = xt1.to(device)
            xt2 = xt2.to(device)
            ytd = ytd.to(device)
            ytgi = ytgi.to(device)
            # double fitness
            ytd_pred = model(xtd)
            # single fitness & gi
            ytgi_pred = torch.add(ytd_pred, - torch.mul(model(xt1), model(xt2)))
            # yt_pred = model(xtd, xt1, xt2)
            loss_fitness = loss_fn(ytd, ytd_pred)
            loss_gi = loss_fn(ytgi, ytgi_pred)
            loss = loss_fitness + loss_gi
            logging.info('initial test batch loss: total {} fitness {} gi {}'.format(loss.item(), loss_fitness.item(),
                                                                                     loss_gi.item()))
            logging.info(
                'initial test batch fitness corr: {}'.format(
                    pearsonr(ytd.cpu().numpy()[:, 0], ytd_pred.cpu().numpy()[:, 0])))
            logging.info(
                'initial test batch gi corr: {}'.format(
                    pearsonr(ytgi.cpu().numpy()[:, 0], ytgi_pred.cpu().numpy()[:, 0])))
            # just run one batch (otherwise takes too long)
            break

    # for epoch in range(n_epoch):
    #     # Training
    #     for x_batch, y_batch in data_tr_loader:
    #         x_batch = x_batch.to(device)
    #         y_batch = y_batch.to(device)
    #         y_batch_pred = model(x_batch)
    #         loss = loss_fn(y_batch_pred, y_batch)
    #         # print(epoch, loss.item())
    #         model.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     # after epoch
    #     logging.info('[{}/{}] training batch loss: {}'.format(epoch, n_epoch, loss.item()))
    #     logging.info('[{}/{}] training batch corr: {}'.format(epoch, n_epoch, pearsonr(y_batch.detach().cpu().numpy()[:, 0], y_batch_pred.detach().cpu().numpy()[:, 0])))
    #
    #     # test
    #     with torch.set_grad_enabled(False):
    #         for xt, yt in data_ts_loader:
    #             xt = xt.to(device)
    #             yt = yt.to(device)
    #             yt_pred = model(xt)
    #             loss = loss_fn(yt_pred, yt)
    #             logging.info('[{}/{}] test batch loss: {}'.format(epoch, n_epoch, loss.item()))
    #             logging.info('[{}/{}] test batch corr: {}'.format(epoch, n_epoch, pearsonr(yt.cpu().numpy()[:, 0], yt_pred.cpu().numpy()[:, 0])))
    #             break
    #
    # logging.info('Done training')
    #
    # # re-run all data to compute final loss
    # with torch.set_grad_enabled(False):
    #     # training batches
    #     loss_training = []
    #     for x_batch, y_batch in data_tr_loader:
    #         x_batch = x_batch.to(device)
    #         y_batch = y_batch.to(device)
    #         y_batch_pred = model(x_batch)
    #         loss = loss_fn(y_batch_pred, y_batch)
    #         corr, pval = pearsonr(y_batch.detach().cpu().numpy()[:, 0], y_batch_pred.detach().cpu().numpy()[:, 0])
    #         loss_training.append({'loss': float(loss.detach().cpu().numpy()), 'corr': corr, 'pval': pval})
    #     loss_training = pd.DataFrame(loss_training)
    #     logging.info("Training data performance (summarized across batches):")
    #     logging.info(loss_training.describe())
    #
    #     # test batches
    #     loss_test = []
    #     for x_batch, y_batch in data_ts_loader:
    #         x_batch = x_batch.to(device)
    #         y_batch = y_batch.to(device)
    #         y_batch_pred = model(x_batch)
    #         loss = loss_fn(y_batch_pred, y_batch)
    #         corr, pval = pearsonr(y_batch.detach().cpu().numpy()[:, 0], y_batch_pred.detach().cpu().numpy()[:, 0])
    #         loss_test.append({'loss': float(loss.detach().cpu().numpy()), 'corr': corr, 'pval': pval})
    #     loss_test = pd.DataFrame(loss_test)
    #     logging.info("Test data performance (summarized across batches):")
    #     logging.info(loss_test.describe())

    # TODO make plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=str, help='path to training data file')
    parser.add_argument('--result', type=str, help='path to output result')
    parser.add_argument('--hid_sizes', nargs='+', type=int, help='hidden layer sizes, does not include first (input) and last (output) layer.')
    parser.add_argument('--epoch', type=int, help='number of epochs')
    # parser.add_argument('--shuffle_label', default=False, action='store_true', help='whether to shuffle training target values, this is a debugging option')
    args = parser.parse_args()
    set_up_logging(args.result)
    logging.debug(args)
    # main(args.data, args.hid_sizes, args.epoch, args.shuffle_label)
    main(args.data, args.hid_sizes, args.epoch)
