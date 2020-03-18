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
                                'Genetic interaction score (ε)', 'Double mutant fitness']].rename(columns={
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


class MyDataSet(Dataset):

    def __init__(self, x, y, num_genes):
        self.num_genes = num_genes
        assert x.shape[0] == y.shape[0]
        self.len = x.shape[0]
        self.x = x
        assert y.shape[1] == 4  # 2 outputs + 2 extra values for indirect evaluation
        self.y_fitness = y[:, [0]]
        self.y_gi = y[:, [1]]
        # extra target value (single fitness, do not used for training)
        self.f1 = y[:, [2]]
        self.f2 = y[:, [3]]

    def __getitem__(self, index):
        _x = self.x[index, :]
        assert len(_x) == 2
        # encode
        xd = np.ones(self.num_genes)
        x1 = np.ones(self.num_genes)
        x2 = np.ones(self.num_genes)
        xd[_x[0]] = 0
        xd[_x[1]] = 0  # 2-cold encoding for double KO
        x1[_x[0]] = 0  # 1-cold encoding for first gene KO
        x2[_x[1]] = 0  # 1-cold encoding for second gene KO
        return torch.from_numpy(xd).float(), torch.from_numpy(x1).float(), torch.from_numpy(
            x2).float(), torch.from_numpy(
            self.y_fitness[index]).float(), torch.from_numpy(self.y_gi[index]).float(), torch.from_numpy(
            self.f1[index]).float(), torch.from_numpy(self.f2[index]).float()

    def __len__(self):
        return self.len


def to_device(xd, x1, x2, yd, ygi, device):
    return xd.to(device), x1.to(device), x2.to(device), yd.to(device), ygi.to(device)


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


def m_wrapper(model, xd, x1, x2, yd=None, ygi=None, yf1=None, yf2=None, loss_fn=None, compute_loss=True,
              compute_corr=False, verbose=False):
    # convenience function
    # double fitness
    yd_pred = model(xd)
    yf1_pred = model(x1)
    yf2_pred = model(x2)
    # single fitness & gi
    ygi_pred = torch.add(yd_pred, - torch.mul(yf1_pred, yf2_pred))
    if compute_loss:
        loss_fitness = loss_fn(yd, yd_pred)
        loss_gi = loss_fn(ygi, ygi_pred)
        loss = loss_fitness + loss_gi
        if verbose:
            logging.info('loss: total {} fitness {} gi {}'.format(loss.item(), loss_fitness.item(),
                                                                  loss_gi.item()))
    else:
        loss = None
        loss_fitness = None
        loss_gi = None
    if compute_corr:
        corr_f, pval_f = pearsonr(yd.detach().cpu().numpy()[:, 0], yd_pred.detach().cpu().numpy()[:, 0])
        corr_gi, pval_gi = pearsonr(ygi.detach().cpu().numpy()[:, 0], ygi_pred.detach().cpu().numpy()[:, 0])
        corr_f1, pval_f1 = pearsonr(yf1.detach().cpu().numpy()[:, 0], yf1_pred.detach().cpu().numpy()[:, 0])
        corr_f2, pval_f2 = pearsonr(yf2.detach().cpu().numpy()[:, 0], yf2_pred.detach().cpu().numpy()[:, 0])
        if verbose:
            logging.info('[fitness] corr: {:.2f} ({:.2e})'.format(corr_f, pval_f))
            logging.info('[gi] corr: {:.2f} ({:.2e})'.format(corr_gi, pval_gi))
            logging.info('[f1 (indirect)] corr: {:.2f} ({:.2e})'.format(corr_f1, pval_f1))
            logging.info('[f2 (indirect)] corr: {:.2f} ({:.2e})'.format(corr_f2, pval_f2))
    return loss, loss_fitness, loss_gi, yd_pred, ygi_pred, yf1_pred, yf2_pred


def main(path_data, hid_sizes, n_epoch, out_dir):
    # load data
    logging.info("Loading dataset: {}".format(path_data))
    df = []
    for _p in path_data:
        df.append(load_data(_p))
    df = pd.concat(df)

    # extract gene ID
    df = add_column(df, 'g1', ['s1'], get_gene_id)
    df = add_column(df, 'g2', ['s2'], get_gene_id)

    # since we're loading f1 and f2 and some might be NaN, let's drop those rows for now
    df = df.dropna()

    # take median of examples with the same gene pair, so that we don't have duplicates
    # df = df[['g1', 'g2', 'interaction', 'fitness']].groupby(by=['g1', 'g2'], as_index=False).agg('median')
    df = df[['g1', 'g2', 'fitness', 'interaction', 'f1', 'f2']].groupby(by=['g1', 'g2'], as_index=False).agg('median')
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
    # # also create a reverse mapping, useful for mapping back to IDs for debugging after training
    # assert len(set(gene_id2idx.values())) == len(gene_id2idx)
    # gene_idx2id = {v: k for k, v in gene_id2idx.iteritems()}
    df_tr = add_column(df_tr, 'x', ['g1', 'g2'], lambda g1, g2: encode_x(g1, g2, gene_id2idx))
    df_ts = add_column(df_ts, 'x', ['g1', 'g2'], lambda g1, g2: encode_x(g1, g2, gene_id2idx))

    # get data
    x_tr = np.asarray(df_tr['x'].to_list())
    y_tr = df_tr[['fitness', 'interaction', 'f1', 'f2']].to_numpy()

    x_ts = np.asarray(df_ts['x'].to_list())
    y_ts = df_ts[['fitness', 'interaction', 'f1', 'f2']].to_numpy()
    assert x_tr.shape[0] == y_tr.shape[0]
    assert x_ts.shape[0] == y_ts.shape[0]

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train + test data handler
    batch_size = 2000
    data_tr_loader = DataLoader(dataset=MyDataSet(x_tr, y_tr, num_genes),
                                batch_size=batch_size, shuffle=True, num_workers=8)
    data_ts_loader = DataLoader(dataset=MyDataSet(x_ts, y_ts, num_genes),
                                batch_size=batch_size, shuffle=True, num_workers=8)

    # model
    model = make_model(num_genes, hid_sizes)
    logging.info("model: \n{}".format(model))
    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # naive guess is the mean of training target value
    yp_naive = np.mean(y_tr, axis=0)
    # calculate loss on naive guess
    # hard-coded to be MSE
    for i, target_name in enumerate(['fitness', 'gi']):   # slice 0 fitness, slice 1 gi
        loss_naive_training = np.mean((yp_naive[i] - y_tr[:, i])**2)
        loss_naive_test = np.mean((yp_naive[i] - y_ts[:, i])**2)
        logging.info("[{}] Naive guess: {}, training loss: {}, test loss: {}".format(target_name, yp_naive[i], loss_naive_training, loss_naive_test))

    # training
    model = model.to(device)

    # initial test performance
    with torch.set_grad_enabled(False):
        for xd, x1, x2, yd, ygi, yf1, yf2 in data_ts_loader:
            xd, x1, x2, yd, ygi = to_device(xd, x1, x2, yd, ygi, device)
            _ = m_wrapper(model, xd, x1, x2, yd, ygi, yf1, yf2, loss_fn=loss_fn, compute_loss=True, compute_corr=True,
                          verbose=True)
            # just run one batch (otherwise takes too long)
            break

    for epoch in range(n_epoch):
        # Training
        for idx, (xd, x1, x2, yd, ygi, yf1, yf2) in enumerate(data_tr_loader):
            xd, x1, x2, yd, ygi = to_device(xd, x1, x2, yd, ygi, device)
            # print if last minibatch
            if idx == len(data_tr_loader) - 1:
                logging.info("[{}/{}] Training last mini batch".format(epoch, n_epoch))
                loss, loss_fitness, loss_gi, _, _, _, _ = m_wrapper(model, xd, x1, x2, yd, ygi, yf1, yf2, loss_fn=loss_fn, compute_loss=True,
                                                        compute_corr=True, verbose=True)
            else:
                loss, loss_fitness, loss_gi, _, _, _, _ = m_wrapper(model, xd, x1, x2, yd, ygi, yf1, yf2, loss_fn=loss_fn, compute_loss=True,
                                                        compute_corr=False, verbose=False)
            # print(epoch, loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
        # test
        with torch.set_grad_enabled(False):
            for xd, x1, x2, yd, ygi, yf1, yf2 in data_ts_loader:   # TODO shall we use next()?
                xd, x1, x2, yd, ygi = to_device(xd, x1, x2, yd, ygi, device)
                logging.info("Testing")
                _ = m_wrapper(model, xd, x1, x2, yd, ygi, yf1, yf2, loss_fn=loss_fn, compute_loss=True, compute_corr=True,
                              verbose=True)
                break

    logging.info('Done training')

    # TODO report global correlation, since it's not additive

    # re-run all data to compute final loss
    with torch.set_grad_enabled(False):
        # training batches
        loss_training = []
        gene1_id_all = []
        gene2_id_all = []
        yd_all = []
        yd_pred_all = []
        ygi_all = []
        ygi_pred_all = []
        yf1_all = []
        yf1_pred_all = []
        yf2_all = []
        yf2_pred_all = []
        for xd, x1, x2, yd, ygi, yf1, yf2 in data_tr_loader:
            xd, x1, x2, yd, ygi = to_device(xd, x1, x2, yd, ygi, device)
            loss, loss_fitness, loss_gi, yd_pred, ygi_pred, yf1_pred, yf2_pred = m_wrapper(model, xd, x1, x2, yd, ygi, yf1, yf2, loss_fn=loss_fn, compute_loss=True,
                                                    compute_corr=False, verbose=False)
            loss_training.append({
                'total': float(loss.detach().cpu().numpy()),
                'fitness': float(loss_fitness.detach().cpu().numpy()),
                'gi': float(loss_gi.detach().cpu().numpy()),
            })
            gene1_id_all.extend(np.take(gene_ids, x1.detach().cpu().numpy().argmin(1)).tolist())  # map idx back to IDs
            gene2_id_all.extend(np.take(gene_ids, x2.detach().cpu().numpy().argmin(1)).tolist())
            yd_all.append(yd.detach().cpu().numpy())
            yd_pred_all.append(yd_pred.detach().cpu().numpy())
            ygi_all.append(ygi.detach().cpu().numpy())
            ygi_pred_all.append(ygi_pred.detach().cpu().numpy())
            yf1_all.append(yf1.detach().cpu().numpy())
            yf1_pred_all.append(yf1_pred.detach().cpu().numpy())
            yf2_all.append(yf2.detach().cpu().numpy())
            yf2_pred_all.append(yf2_pred.detach().cpu().numpy())
        loss_training = pd.DataFrame(loss_training)
        logging.info("Training data performance (summarized across batches):")
        logging.info(loss_training.describe())
        yd_all = np.concatenate(yd_all, axis=0)
        yd_pred_all = np.concatenate(yd_pred_all, axis=0)
        ygi_all = np.concatenate(ygi_all, axis=0)
        ygi_pred_all = np.concatenate(ygi_pred_all, axis=0)
        yf1_all = np.concatenate(yf1_all, axis=0)
        yf1_pred_all = np.concatenate(yf1_pred_all, axis=0)
        yf2_all = np.concatenate(yf2_all, axis=0)
        yf2_pred_all = np.concatenate(yf2_pred_all, axis=0)
        logging.info("correlation (fitness)")
        logging.info(pearsonr(yd_all[:, 0], yd_pred_all[:, 0]))
        logging.info("correlation (gi)")
        logging.info(pearsonr(ygi_all[:, 0], ygi_pred_all[:, 0]))
        logging.info("correlation (f1 indirect)")
        logging.info(pearsonr(yf1_all[:, 0], yf1_pred_all[:, 0]))
        logging.info("correlation (f2 indirect)")
        logging.info(pearsonr(yf2_all[:, 0], yf2_pred_all[:, 0]))
        # df for exporting
        df_train_pred = pd.DataFrame({
            'g1': gene1_id_all,
            'g2': gene2_id_all,
            'yd': yd_all[:, 0],
            'yd_pred': yd_pred_all[:, 0],
            'ygi': ygi_all[:, 0],
            'ygi_pred': ygi_pred_all[:, 0],
            'yf1': yf1_all[:, 0],
            'yf1_pred': yf1_pred_all[:, 0],
            'yf2': yf2_all[:, 0],
            'yf2_pred': yf2_pred_all[:, 0],
        })
        df_train_pred.to_csv(os.path.join(out_dir, 'pred_train.csv'), index=False)

        # test batches
        loss_test = []
        gene1_id_all = []
        gene2_id_all = []
        yd_all = []
        yd_pred_all = []
        ygi_all = []
        ygi_pred_all = []
        yf1_all = []
        yf1_pred_all = []
        yf2_all = []
        yf2_pred_all = []
        for xd, x1, x2, yd, ygi, yf1, yf2 in data_ts_loader:
            xd, x1, x2, yd, ygi = to_device(xd, x1, x2, yd, ygi, device)
            loss, loss_fitness, loss_gi, yd_pred, ygi_pred, yf1_pred, yf2_pred = m_wrapper(model, xd, x1, x2, yd, ygi, yf1, yf2, loss_fn=loss_fn, compute_loss=True,
                                                    compute_corr=False, verbose=False)
            loss_test.append({
                'total': float(loss.detach().cpu().numpy()),
                'fitness': float(loss_fitness.detach().cpu().numpy()),
                'gi': float(loss_gi.detach().cpu().numpy()),
            })
            gene1_id_all.extend(np.take(gene_ids, x1.detach().cpu().numpy().argmin(1)).tolist())  # map idx back to IDs
            gene2_id_all.extend(np.take(gene_ids, x2.detach().cpu().numpy().argmin(1)).tolist())
            yd_all.append(yd.detach().cpu().numpy())
            yd_pred_all.append(yd_pred.detach().cpu().numpy())
            ygi_all.append(ygi.detach().cpu().numpy())
            ygi_pred_all.append(ygi_pred.detach().cpu().numpy())
            yf1_all.append(yf1.detach().cpu().numpy())
            yf1_pred_all.append(yf1_pred.detach().cpu().numpy())
            yf2_all.append(yf2.detach().cpu().numpy())
            yf2_pred_all.append(yf2_pred.detach().cpu().numpy())
        loss_test = pd.DataFrame(loss_test)
        logging.info("Test data performance (summarized across batches):")
        logging.info(loss_test.describe())
        yd_all = np.concatenate(yd_all, axis=0)
        yd_pred_all = np.concatenate(yd_pred_all, axis=0)
        ygi_all = np.concatenate(ygi_all, axis=0)
        ygi_pred_all = np.concatenate(ygi_pred_all, axis=0)
        yf1_all = np.concatenate(yf1_all, axis=0)
        yf1_pred_all = np.concatenate(yf1_pred_all, axis=0)
        yf2_all = np.concatenate(yf2_all, axis=0)
        yf2_pred_all = np.concatenate(yf2_pred_all, axis=0)
        logging.info("correlation (fitness)")
        logging.info(pearsonr(yd_all[:, 0], yd_pred_all[:, 0]))
        logging.info("correlation (gi)")
        logging.info(pearsonr(ygi_all[:, 0], ygi_pred_all[:, 0]))
        logging.info("correlation (f1 indirect)")
        logging.info(pearsonr(yf1_all[:, 0], yf1_pred_all[:, 0]))
        logging.info("correlation (f2 indirect)")
        logging.info(pearsonr(yf2_all[:, 0], yf2_pred_all[:, 0]))
        # df for exporting
        df_test_pred = pd.DataFrame({
            'g1': gene1_id_all,
            'g2': gene2_id_all,
            'yd': yd_all[:, 0],
            'yd_pred': yd_pred_all[:, 0],
            'ygi': ygi_all[:, 0],
            'ygi_pred': ygi_pred_all[:, 0],
            'yf1': yf1_all[:, 0],
            'yf1_pred': yf1_pred_all[:, 0],
            'yf2': yf2_all[:, 0],
            'yf2_pred': yf2_pred_all[:, 0],
        })
        df_test_pred.to_csv(os.path.join(out_dir, 'pred_test.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=str, help='path to training data file')
    parser.add_argument('--result', type=str, help='path to output result')
    parser.add_argument('--hid_sizes', nargs='*', type=int, help='hidden layer sizes, does not include first (input) and last (output) layer.')
    parser.add_argument('--epoch', type=int, help='number of epochs')
    args = parser.parse_args()
    set_up_logging(args.result)
    logging.debug(args)
    main(args.data, args.hid_sizes, args.epoch, args.result)