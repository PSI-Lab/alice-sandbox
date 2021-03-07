import os
import sys
import subprocess
import logging
import pprint
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import datacorral as dc
# sys.path.insert(0, '../../rna_ss/')
# from utils import db2pairs
# from local_struct_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb
sys.path.insert(0, '../rna_ss_utils/')  # FIXME hacky
from utils import db2pairs
from local_struct_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb


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


def length_grouping(df, n):
    # split input df into multiple dfs,
    # according to the length of column `seq`
    dfs = []
    # first collect all lengths
    if 'len' not in df.columns:
        df = add_column(df, 'len', ['seq'], len)
    # sort
    df.sort_values(by=['len'], inplace=True)
    # split into bins
    idxes = np.linspace(0, len(df), num=n + 1, dtype=int)
    for start, end in zip(idxes[:-1], idxes[1:]):
        dfs.append(df[start:end])
    return dfs


class MyDataSet(Dataset):
    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    # TODO length grouping

    def __init__(self, df):
        self.len = len(df)
        self.df = df

    def _encode_seq(self, seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        x = np.asarray([int(x) for x in list(seq)])
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def tile_and_stack(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 4
        l = x.shape[0]
        x1 = x[:, np.newaxis, :]
        x2 = x[np.newaxis, :, :]
        x1 = np.repeat(x1, l, axis=1)
        x2 = np.repeat(x2, l, axis=0)
        return np.concatenate([x1, x2], axis=2)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        seq = row['seq']
        x_1d = self._encode_seq(seq)
        x_2d = self.tile_and_stack(x_1d)

        # expand one_idx and bb into target arrays
        one_idx = row['one_idx']
        y = np.zeros((len(seq), len(seq)))
        y[one_idx] = 1
        # add dimension
        y = y[:, :, np.newaxis]

        # TODO mask lower triangular?

        return torch.from_numpy(x_1d).float(), torch.from_numpy(x_2d).float(), torch.from_numpy(y).float()


    def __len__(self):
        return self.len


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate2D:
    def __init__(self):
        pass

    def pad_collate(self, batch):
        """
        args:
            batch - list of (x_1d, x_2d, y)

        reutrn:
            x_1d_new - x_1d after padding
            x_2d_new - x_2d after padding
            y_new - y after padding
            
        """
        # find longest sequence
        max_len = max(map(lambda x: x[1].shape[0], batch))
        # we expect it to be symmetric between dim 0 and 1
        assert max_len == max(map(lambda x: x[1].shape[1], batch))

        # pad
        x_1d_new = []
        x_2d_new = []
        y_new = []
        for x_1d, x_2d, y in batch:
            # permute: prep for stacking: batch x channel x L
            x_1d_new.append(pad_tensor(x_1d, pad=max_len, dim=0).permute(1, 0))
            # permute: prep for stacking: batch x channel x H x W
            x_2d_new.append(pad_tensor(pad_tensor(x_2d, pad=max_len, dim=0), pad=max_len, dim=1).permute(2, 0, 1))

            y_new.append(pad_tensor(pad_tensor(y, pad=max_len, dim=0), pad=max_len, dim=1).permute(2, 0, 1))  # TODO does y have dim 2?

        # stack tensor
        x_1d_new = torch.stack(x_1d_new, dim=0)
        x_2d_new = torch.stack(x_2d_new, dim=0)
        y_new = torch.stack(y_new, dim=0)

        return x_1d_new, x_2d_new, y_new


    def __call__(self, batch):
        return self.pad_collate(batch)



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
        return self.output_network(x_2d, z), mu_p, logvar_p


# TODO move to class level
loss_b = torch.nn.BCELoss(reduction='none')
loss_m = torch.nn.NLLLoss(reduction='none')
loss_e = torch.nn.MSELoss(reduction='none')



def masked_loss_e(x, y, m):
    # batch x channel? x h x w
    l = loss_e(x, y)
    n_valid_output = torch.sum(torch.sum(m, dim=3), dim=2)  # vector of length = batch
    loss_spatial_sum = torch.sum(torch.sum(torch.mul(l, m), dim=3), dim=2)
    n_valid_output[n_valid_output == 0] = 1
    loss_spatial_mean = loss_spatial_sum / n_valid_output
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    return torch.mean(loss_batch_mean)


def masked_loss_b(x, y, m):
    # batch x channel? x h x w
    l = loss_b(x, y)
    n_valid_output = torch.sum(torch.sum(m, dim=3), dim=2)  # vector of length = batch
    loss_spatial_sum = torch.sum(torch.sum(torch.mul(l, m), dim=3), dim=2)
    n_valid_output[n_valid_output == 0] = 1
    loss_spatial_mean = loss_spatial_sum / n_valid_output
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    return torch.mean(loss_batch_mean)


def masked_loss_m(x, y, m):
    # remove singleton dimension in target & mask since NLLLoss doesn't need it
    assert y.shape[1] == 1
    assert m.shape[1] == 1
    y = y[:, 0, :, :]
    m = m[:, 0, :, :]
    l = loss_m(x, y.long())  # FIXME should use long type in data gen (also need to fix padding <- not happy with long?)
    # batch x h x w
    n_valid_output = torch.sum(torch.sum(m, dim=2), dim=1)  # vector of length = batch
    loss_spatial_sum = torch.sum(torch.sum(torch.mul(l, m), dim=2), dim=1)
    n_valid_output[n_valid_output == 0] = 1
    loss_spatial_mean = loss_spatial_sum / n_valid_output
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    return torch.mean(loss_batch_mean)


def kl_loss(mu_q, logvar_q, mu_p, logvar_p):
    # KL divergence in closed form
    # batch x ch 
    # multivariate Gaussian KL divergence
    # sum over latent dim, mean over batch
    kl_val = 0.5 * torch.mean(torch.sum(logvar_q.exp()/logvar_p.exp() + \
                          (mu_p - mu_q).pow(2)/logvar_p.exp() - 1 + \
                          logvar_p - logvar_q, dim=1), dim=0)

    logging.info("kl loss: {}".format(kl_val))
    return kl_val


def is_seq_valid(seq):
    seq = seq.upper()
    if set(seq).issubset(set('ACGTUN')):
        return True
    else:
        return False


def main(path_data, latent_dim, n_epoch, batch_size, max_length, out_dir, n_cpu):
    logging.info("Loading dataset: {}".format(path_data))
    dc_client = dc.Client()
    df = []
    for _p in path_data:
        if os.path.isfile(_p):
            df.append(pd.read_pickle(_p, compression='gzip'))
        else:
            print(dc_client.get_path(_p))
            df.append(pd.read_pickle(dc_client.get_path(_p), compression='gzip'))
    df = pd.concat(df)
    # subset to max length if specified
    if max_length:
        logging.info("Subsetting to max length, n_rows before: {}".format(len(df)))
        if 'len' not in df.columns:
            df = add_column(df, 'len', ['seq'], len)
            df = df[df['len'] <= max_length]
            df = df.drop(columns=['len'])
        else:
            df = df[df['len'] <= max_length]
        logging.info("After: {}".format(len(df)))
    # subset to sequence with valid nucleotides ACGTN
    df = add_column(df, 'is_seq_valid', ['seq'], is_seq_valid)
    n_invalid = (~df['is_seq_valid']).sum()
    logging.info("Subsetting to sequence with valid bases, n_rows before: {}".format(len(df)))
    logging.info("Dropping {} rows".format(n_invalid))
    df = df[df['is_seq_valid']]
    logging.info("After: {}".format(len(df)))
    df.drop(columns=['is_seq_valid'], inplace=True)

    # split into training+validation
    # shuffle rows
    df = df.sample(frac=1).reset_index(drop=True)
    # subset
    # tr_prop = 0.95
    tr_prop = 0.8
    _n_tr = int(len(df) * tr_prop)
    logging.info("Using {} data for training and {} for validation".format(_n_tr, len(df) - _n_tr))
    df_tr = df[:_n_tr]
    df_va = df[_n_tr:]
    # length group + chain dataset, to ensure that sequences in the same minibatch are of similar length
    n_groups = 20
    # data loaders
    data_loader_tr = DataLoader(torch.utils.data.ConcatDataset([MyDataSet(x) for x in length_grouping(df_tr, n_groups)]),
                                batch_size=batch_size,
                                shuffle=True, num_workers=n_cpu,
                                collate_fn=PadCollate2D())
    data_loader_va = DataLoader(torch.utils.data.ConcatDataset([MyDataSet(x) for x in length_grouping(df_va, n_groups)]),
                                batch_size=batch_size,
                                shuffle=True, num_workers=n_cpu,
                                collate_fn=PadCollate2D())

    # model
    model = LatentVarModel(latent_dim=latent_dim)
    print(model)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: {}".format(device))

    model = model.to(device)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(n_epoch):
        running_loss_tr = []
        running_auroc_tr = []
        running_auprc_tr = []
        for i, (x_1d, x_2d, y) in enumerate(data_loader_tr):

            x_1d = x_1d.to(device)
            x_2d = x_2d.to(device)
            y = y.to(device)

            # yp, mu, logvar = model(x)
            yp, mu_q, logvar_q, mu_p, logvar_p = model(x_1d, x_2d, y)

            # FIXME not using mask for now! - should at least mask lower triangular
            m = torch.ones_like(y)

            loss_1 = masked_loss_b(yp, y, m)  # order: pred, target, mask, mask_weight
            loss_2 = kl_loss(mu_q, logvar_q, mu_p, logvar_p)
            loss = loss_1 + loss_2
            running_loss_tr.append(loss.item())
            logging.info(
                "Epoch {} ite {}/{} Training loss (posterior): {} ({} + {})".format(epoch, i, len(data_loader_tr), loss,
                                                                                    loss_1, loss_2))
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO calculate metric?
            y_true = y.detach().cpu().numpy().reshape(y.shape[0], -1)
            y_score = yp.detach().cpu().numpy().reshape(yp.shape[0], -1)
            aucs = []
            for idx_example in range(y_true.shape[0]):
                # skip example if only 1 class
                if np.max(y_true[idx_example, :]) == np.min(y_true[idx_example, :]):
                    continue
                aucs.append(roc_auc_score(y_true=y_true[idx_example, :], y_score=y_score[idx_example, :]))
            aucs = np.asarray(aucs)
            print("AUCs: mean {}, median: {}, max {}, min {}".format(np.mean(aucs), np.median(aucs), np.max(aucs), np.min(aucs)))

            # add in prior-based loss (TODO weight ths loss?)
            yp, mu_p, logvar_p = model.inference(x_1d, x_2d)
            loss_3 = masked_loss_b(yp, y, m)
            logging.info("Epoch {} Training loss (prior): {}".format(epoch, loss_3))
            model.zero_grad()
            loss_3.backward()
            optimizer.step()


        # save model
        _model_path = os.path.join(out_dir, 'model_ckpt_ep_{}.pth'.format(epoch))
        torch.save(model.state_dict(), _model_path)
        logging.info("Model checkpoint saved at: {}".format(_model_path))

        logging.info(
            "Epoch {}/{}, training loss (posterior) (running) {}".format(epoch, n_epoch,np.mean(np.stack(running_loss_tr))))

        with torch.set_grad_enabled(False):
            # report validation loss
            running_loss_va = []
            running_auroc_va = []
            running_auprc_va = []
            for i, (x_1d, x_2d, y) in enumerate(data_loader_va):
                x_1d = x_1d.to(device)
                x_2d = x_2d.to(device)
                y = y.to(device)
                # yp, mu, logvar = model(x)
                yp, mu_q, logvar_q, mu_p, logvar_p = model(x_1d, x_2d, y)

                # FIXME not using mask for now!
                m = torch.ones_like(y)

                loss_1 = masked_loss_b(yp, y, m)  # order: pred, target, mask, mask_weight
                loss_2 = kl_loss(mu_q, logvar_q, mu_p, logvar_p)
                loss = loss_1 + loss_2
                # running_loss_va.append(loss.detach().cpu().numpy())
                running_loss_va.append(loss.item())
                logging.info(
                    "Epoch {} ite {}/{} Validation loss (posterior): {} ({} + {})".format(epoch, i, len(data_loader_va),
                                                                                          loss, loss_1, loss_2))

                # TODO calculate metric?
                y_true = y.detach().cpu().numpy().reshape(y.shape[0], -1)
                y_score = yp.detach().cpu().numpy().reshape(yp.shape[0], -1)
                aucs = []
                for idx_example in range(y_true.shape[0]):
                    # skip example if only 1 class
                    if np.max(y_true[idx_example, :]) == np.min(y_true[idx_example, :]):
                        continue
                    aucs.append(roc_auc_score(y_true=y_true[idx_example, :], y_score=y_score[idx_example, :]))
                aucs = np.asarray(aucs)
                print("AUCs: mean {}, median: {}, max {}, min {}".format(np.mean(aucs), np.median(aucs), np.max(aucs),
                                                                         np.min(aucs)))

            logging.info(
                "Epoch {}/{}, validation loss (posterior): {}".format(epoch, n_epoch,
                                                                      np.mean(np.stack(running_loss_va))))

            # report validation loss using z sampled from prior network
            running_loss_va = []
            running_auroc_va = []
            running_auprc_va = []
            for i, (x_1d, x_2d, y) in enumerate(data_loader_va):
                x_1d = x_1d.to(device)
                x_2d = x_2d.to(device)
                y = y.to(device)
                # yp, mu, logvar = model(x)
                yp, mu_p, logvar_p = model.inference(x_1d, x_2d)
                # FIXME not using mask for now!
                m = torch.ones_like(y)
                loss_1 = masked_loss_b(yp, y, m)
                running_loss_va.append(loss_1.item())
                logging.info(
                    "Epoch {} ite {}/{} Validation loss (prior, p(y|x) only): {}".format(epoch, i, len(data_loader_va),
                                                                                         loss_1))

                # TODO calculate metric?
                y_true = y.detach().cpu().numpy().reshape(y.shape[0], -1)
                y_score = yp.detach().cpu().numpy().reshape(yp.shape[0], -1)
                aucs = []
                for idx_example in range(y_true.shape[0]):
                    # skip example if only 1 class
                    if np.max(y_true[idx_example, :]) == np.min(y_true[idx_example, :]):
                        continue
                    aucs.append(roc_auc_score(y_true=y_true[idx_example, :], y_score=y_score[idx_example, :]))
                aucs = np.asarray(aucs)
                print("AUCs: mean {}, median: {}, max {}, min {}".format(np.mean(aucs), np.median(aucs), np.max(aucs),
                                                                         np.min(aucs)))

            logging.info(
                "Epoch {}/{}, validation loss (prior, p(y|x) only): {}".format(epoch, n_epoch,
                                                                               np.mean(np.stack(running_loss_va))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=str, help='Path or DC ID to training data file, should be in pkl.gz format')
    parser.add_argument('--result', type=str, help='Path to output result')
    # parser.add_argument('--num_filters', nargs='*', type=int, help='Number of conv filters for each layer.')
    # parser.add_argument('--filter_width', nargs='*', type=int, help='Filter width for each layer.')
    # parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    # parser.add_argument('--mask', type=float, default=0.0, help='Mask weight. Setting to 0 is equivalent to hard mask.')
    parser.add_argument('--latent_dim', type=int, help='Number of latent variables')
    parser.add_argument('--epoch', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Mini batch size')
    parser.add_argument('--max_length', type=int, default=0,
                        help='Max sequence length to train on. This is used to subset the dataaset.')
    parser.add_argument('--cpu', type=int, help='Number of CPU workers per data loader')
    args = parser.parse_args()

    # some basic logging
    set_up_logging(args.result)
    logging.debug("Cmd: {}".format(args))  # cmd args
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    logging.debug("Current dir: {}, git hash: {}".format(cur_dir, git_hash))
    # training
    main(args.data, args.latent_dim, args.epoch, args.batch_size, args.max_length, args.result,
         args.cpu)
