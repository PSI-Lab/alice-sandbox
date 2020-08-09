import os
import subprocess
import logging
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

    # def _make_pair_arr(self, seq, one_idx):
    #
    #     def _make_arr(seq, one_idx):
    #         target = np.zeros((len(seq), len(seq)))
    #         target[one_idx] = 1
    #         return target
    #
    #     # def _mask(x):
    #     #     assert len(x.shape) == 2
    #     #     assert x.shape[0] == x.shape[1]
    #     #     x[np.tril_indices(x.shape[0])] = -1   # TODO how to mask gradient in pytorch?
    #     #     return x
    #
    #     def _make_mask(x):
    #         assert len(x.shape) == 2
    #         assert x.shape[0] == x.shape[1]
    #         m = np.ones_like(x)
    #         m[np.tril_indices(x.shape[0])] = 0
    #         return m
    #
    #     pair_matrix = _make_arr(seq, one_idx)
    #     # pair_matrix = _mask(pair_matrix)
    #     mask = _make_mask(pair_matrix)
    #     return pair_matrix, mask

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
        seq = self.df.iloc[index]['seq']
        # one_idx = self.df.iloc[index]['one_idx']
        x = self._encode_seq(seq)
        x = self.tile_and_stack(x)
        # _, m = self._make_pair_arr(seq, one_idx)
        m = self.df.iloc[index]['mask']
        # todo tmp
        y = self.df.iloc[index]['target']
        m = np.repeat(m[:, :, np.newaxis], y.shape[2], axis=2)
        # print(x.shape, y.shape, m.shape)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(m).float()

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
            batch - list of (x, y)

        reutrn:
            xs - x after padding
            ys - y after padding
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[0], batch))
        # we expect it to be symmetric between dim 0 and 1
        assert max_len == max(map(lambda x: x[0].shape[1], batch))
        # also for y
        assert max_len == max(map(lambda x: x[1].shape[0], batch))
        assert max_len == max(map(lambda x: x[1].shape[1], batch))
        # pad according to max_len
        batch = [(pad_tensor(pad_tensor(x, pad=max_len, dim=0), pad=max_len, dim=1),
                  pad_tensor(pad_tensor(y, pad=max_len, dim=0), pad=max_len, dim=1),
                  pad_tensor(pad_tensor(m, pad=max_len, dim=0), pad=max_len, dim=1))  # zero pad mask
                 for x, y, m in batch]
        # stack all, also make torch compatible shapes: batch x channel x H x W
        xs = torch.stack([x[0].permute(2, 0, 1) for x in batch], dim=0)
        ys = torch.stack([x[1].permute(2, 0, 1) for x in batch], dim=0)
        ms = torch.stack([x[2].permute(2, 0, 1) for x in batch], dim=0)
        return xs, ys, ms

    def __call__(self, batch):
        return self.pad_collate(batch)



class SimpleConvNet(nn.Module):
    def __init__(self, num_filters, filter_width, dropout):
        super(SimpleConvNet, self).__init__()

        num_filters = [8] + num_filters
        filter_width = [None] + filter_width
        cnn_layers = []
        for i, (nf, fw) in enumerate(zip(num_filters[1:], filter_width[1:])):
            assert fw % 2 == 1  # odd
            cnn_layers.append(nn.Conv2d(num_filters[i], nf, kernel_size=fw, stride=1, padding=fw//2))
            cnn_layers.append(nn.BatchNorm2d(nf))
            cnn_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                cnn_layers.append(nn.Dropout(dropout))
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # self.cnn_layers = nn.Sequential(
        #     nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        # )

        self.fc = nn.Conv2d(num_filters[-1], 5, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def masked_loss(x, y, m):
    # print(x.shape, y.shape, m.shape)
    l = torch.nn.BCELoss(reduction='none')(x, y)
    # number of valid entries = 1's in the mask
    # TODO later on we might have different mask per channel - currently implementation assumes same mask across all channels
    assert m.shape[1] == y.shape[1]
    n_valid_output = torch.sum(torch.sum(m, dim=3), dim=2)  # vector of length = batch
    # average over spatial dimension is achieved by summing then dividing by the above
    # (need to do this since we're padding + masking, can't naively do mean)
    # note that tensor shapes: batch x channel x H x W
    loss_spatial_sum = torch.sum(torch.sum(torch.mul(l, m), dim=3), dim=2)
    # for cases where all elements are masked, the corresponding element in n_valid_output will be 0
    # in such case, since the final numerator will be 0, wlog, we'll set the denominator to 1 (if leaving as 0 will result in NaN)
    n_valid_output[n_valid_output == 0] = 1
    loss_spatial_mean = loss_spatial_sum/n_valid_output  # element-wise division
    # average over batch, this is the per-channel loss
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    logging.info(loss_batch_mean)
    # average over channels
    return torch.mean(loss_batch_mean)


def to_device(x, y, m, device):
    return x.to(device), y.to(device), m.to(device)


def is_seq_valid(seq):
    seq = seq.upper()
    if set(seq).issubset(set('ACGTUN')):
        return True
    else:
        return False


# def compute_metrics(x, y, m):
#     # x : true label, y: pred, m: binary mask, where 1 indicate valid
#     # x, y, m are both torch tensors with batch x channel=1 x H x W
#     assert x.shape[1] == 1
#     assert y.shape[1] == 1
#     assert m.shape[1] == 1
#     for dim in [0, 1, 2]:
#         assert x.shape[dim] == y.shape[dim]
#         assert x.shape[dim] == m.shape[dim]
#     aurocs = []
#     auprcs = []
#     for idx_batch in range(x.shape[0]):
#         # use mask to select non-zero entries
#         _x = x[idx_batch, 0, :, :]
#         _y = y[idx_batch, 0, :, :]
#         _m = m[idx_batch, 0, :, :]
#         mask_bool = _m.eq(1)
#         _x2 = _x.masked_select(mask_bool).flatten().detach().cpu().numpy()
#         _y2 = _y.masked_select(mask_bool).flatten().detach().cpu().numpy()
#         # do not compute if there's only one class
#         if not np.all(_x2 == _x2[0]):
#             aurocs.append(roc_auc_score(_x2, _y2))
#             auprcs.append(average_precision_score(_x2, _y2))
#     return aurocs, auprcs


def main(path_data, num_filters, filter_width, dropout, n_epoch, batch_size, max_length, out_dir, n_cpu):
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

    model = SimpleConvNet(num_filters, filter_width, dropout)
    print(model)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    # # naive guess is the mean of training target value
    # yp_naive = torch.mean(torch.stack([torch.mean(y) for _, y, _ in data_loader_tr]))
    # logging.info("Naive guess: {}".format(yp_naive))
    # # calculate loss using naive guess
    # logging.info("Naive guess performance")
    #
    # with torch.set_grad_enabled(False):
    #     # training
    #     loss_naive_tr = []
    #     auroc_naive_tr = []
    #     auprc_naive_tr = []
    #     for x, y, m in data_loader_tr:
    #         x, y, m = to_device(x, y, m, device)
    #         yp = torch.ones_like(y) * yp_naive
    #         # loss_naive_tr.append(masked_loss(yp, y, m).detach().cpu().numpy())
    #         loss_naive_tr.append(masked_loss(yp, y, m).item())
    #         _r, _p = compute_metrics(y, yp, m)
    #         auroc_naive_tr.extend(_r)
    #         auprc_naive_tr.extend(_p)
    #     logging.info("Training: loss {} au-ROC {} au-PRC {}".format(np.mean(np.stack(loss_naive_tr)),
    #                                                                 np.mean(np.stack(auroc_naive_tr)),
    #                                                                 np.mean(np.stack(auprc_naive_tr))))
    #     # validation
    #     loss_naive_va = []
    #     auroc_naive_va = []
    #     auprc_naive_va = []
    #     for x, y, m in data_loader_va:
    #         x, y, m = to_device(x, y, m, device)
    #         yp = torch.ones_like(y) * yp_naive
    #         # loss_naive_va.append(masked_loss(yp, y, m).detach().cpu().numpy())
    #         loss_naive_va.append(masked_loss(yp, y, m).item())
    #         _r, _p = compute_metrics(y, yp, m)
    #         auroc_naive_va.extend(_r)
    #         auprc_naive_va.extend(_p)
    #     logging.info("Validation: loss {} au-ROC {} au-PRC {}".format(np.mean(np.stack(loss_naive_va)),
    #                                                                   np.mean(np.stack(auroc_naive_va)),
    #                                                                   np.mean(np.stack(auprc_naive_va))))

    for epoch in range(n_epoch):
        running_loss_tr = []
        running_auroc_tr = []
        running_auprc_tr = []
        for x, y, m in data_loader_tr:
            x, y, m = to_device(x, y, m, device)
            yp = model(x)
            loss = masked_loss(yp, y, m)  # order: pred, target, mask
            # running_loss_tr.append(loss.detach().cpu().numpy())

            running_loss_tr.append(loss.item())
            # _r, _p = compute_metrics(y, yp, m)
            # running_auroc_tr.extend(_r)
            # running_auprc_tr.extend(_p)
            logging.info("Epoch {} Training loss: {}".format(epoch, loss))

            model.zero_grad()
            loss.backward()
            optimizer.step()

        # # save model
        # _model_path = os.path.join(out_dir, 'model_ckpt_ep_{}.pth'.format(epoch))
        # torch.save(model.state_dict(), _model_path)
        # logging.info("Model checkpoint saved at: {}".format(_model_path))

        # save the last minibatch prediction
        df_pred = []
        for i in range(y.shape[0]):  #  batch x channel x H x W
            _y = y[i, :, :, :].detach().cpu().numpy()
            _yp = yp[i, :, :, :].detach().cpu().numpy()
            df_pred.append({'target': _y, 'pred': _yp, 'subset': 'training'})

        # # report training loss
        # logging.info(
        #     "Epoch {}/{}, training loss (running) {}, au-ROC {}, au-PRC {}".format(epoch, n_epoch,
        #                                                                            np.mean(
        #                                                                                np.stack(running_loss_tr)),
        #                                                                            np.mean(np.stack(running_auroc_tr)),
        #                                                                            np.mean(np.stack(running_auprc_tr))))
        logging.info(
            "Epoch {}/{}, training loss (running) {}".format(epoch, n_epoch,np.mean(np.stack(running_loss_tr))))

        with torch.set_grad_enabled(False):
            # report validation loss
            running_loss_va = []
            running_auroc_va = []
            running_auprc_va = []
            for x, y, m in data_loader_va:
                x, y, m = to_device(x, y, m, device)
                yp = model(x)
                loss = masked_loss(yp, y, m)
                # running_loss_va.append(loss.detach().cpu().numpy())
                running_loss_va.append(loss.item())
                logging.info("Epoch {} Validation loss: {}".format(epoch, loss))
                # _r, _p = compute_metrics(y, yp, m)
                # running_auroc_va.extend(_r)
                # running_auprc_va.extend(_p)
            # logging.info(
            #     "Epoch {}/{}, validation loss {}, au-ROC {}, au-PRC {}".format(epoch, n_epoch,
            #                                                                    np.mean(np.stack(running_loss_va)),
            #                                                                    np.mean(np.stack(running_auroc_va)),
            #                                                                    np.mean(np.stack(running_auprc_va))))
            logging.info(
                "Epoch {}/{}, validation loss {}".format(epoch, n_epoch, np.mean(np.stack(running_loss_va))))


            # save the last minibatch prediction
            for i in range(y.shape[0]):  #  batch x channel x H x W
                _y = y[i, :, :, :].detach().cpu().numpy()
                _yp = yp[i, :, :, :].detach().cpu().numpy()
                df_pred.append({'target': _y, 'pred': _yp, 'subset': 'validation'})

        # end pf epoch
        # export prediction
        out_file = os.path.join(out_dir, 'pred_ep_{}.pkl.gz'.format(epoch))
        df_pred = pd.DataFrame(df_pred)
        df_pred.to_pickle(out_file, compression='gzip')
        logging.info("Exported prediction (one minibatch) to: {}".format(out_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=str, help='Path or DC ID to training data file, should be in pkl.gz format')
    parser.add_argument('--result', type=str, help='Path to output result')
    parser.add_argument('--num_filters', nargs='*', type=int, help='Number of conv filters for each layer.')
    parser.add_argument('--filter_width', nargs='*', type=int, help='Filter width for each layer.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
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
    assert 0 <= args.dropout <= 1
    main(args.data, args.num_filters, args.filter_width, args.dropout, args.epoch, args.batch_size, args.max_length, args.result,
         args.cpu)
