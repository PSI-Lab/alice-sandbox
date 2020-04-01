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

    def _make_pair_arr(self, seq, one_idx):

        def _make_arr(seq, one_idx):
            target = np.zeros((len(seq), len(seq)))
            target[one_idx] = 1
            return target

        # def _mask(x):
        #     assert len(x.shape) == 2
        #     assert x.shape[0] == x.shape[1]
        #     x[np.tril_indices(x.shape[0])] = -1   # TODO how to mask gradient in pytorch?
        #     return x

        def _make_mask(x):
            assert len(x.shape) == 2
            assert x.shape[0] == x.shape[1]
            m = np.ones_like(x)
            m[np.tril_indices(x.shape[0])] = 0
            return m

        pair_matrix = _make_arr(seq, one_idx)
        # pair_matrix = _mask(pair_matrix)
        mask = _make_mask(pair_matrix)
        return pair_matrix, mask

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
        one_idx = self.df.iloc[index]['one_idx']
        x = self._encode_seq(seq)
        x = self.tile_and_stack(x)
        y, m = self._make_pair_arr(seq, one_idx)
        y = y[:, :, np.newaxis]
        m = m[:, :, np.newaxis]
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


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.resample = resample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.resample:
            residual = self.resample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_filters, num_stacks):
        assert len(num_filters) == len(num_stacks)
        assert len(num_filters) > 0
        super(ResNet, self).__init__()
        self.in_channels = num_filters[0]
        self.conv = conv3x3(8, num_filters[0])
        self.bn = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.res_layers = []

        for nf, ns in zip(num_filters, num_stacks):
            self.res_layers.append(self.make_layer(block, nf, ns))
        self.res_layers = nn.ModuleList(self.res_layers)

        self.fc = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def make_layer(self, block, out_channels, num_blocks):
        resample = None
        if self.in_channels != out_channels:
            resample = nn.Sequential(
                conv3x3(self.in_channels, out_channels),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, resample))
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        for layer in self.res_layers:
            out = layer(out)

        out = self.sigmoid(self.fc(out))
        return out


def masked_loss(x, y, m):
    l = torch.nn.BCELoss(reduction='none')(x, y)
    # note that tensor shapes: batch x channel x H x W
    return torch.mean(torch.sum(torch.sum(torch.mul(l, m), dim=3), dim=2))


def to_device(x, y, m, device):
    return x.to(device), y.to(device), m.to(device)


def compute_metrics(x, y, m):
    # x : true label, y: pred, m: binary mask, where 1 indicate valid
    # x, y, m are both torch tensors with batch x channel=1 x H x W
    assert x.shape[1] == 1
    assert y.shape[1] == 1
    assert m.shape[1] == 1
    for dim in [0, 1, 2]:
        assert x.shape[dim] == y.shape[dim]
        assert x.shape[dim] == m.shape[dim]
    aurocs = []
    auprcs = []
    for idx_batch in range(x.shape[0]):
        # use mask to select non-zero entries
        _x = x[idx_batch, 0, :, :]
        _y = y[idx_batch, 0, :, :]
        _m = m[idx_batch, 0, :, :]
        mask_bool = _m.eq(1)
        # _x2 = _x.masked_select(mask_bool).flatten().detach().cpu().numpy()
        _x2 = _x.masked_select(mask_bool).flatten().cpu().numpy()
        # _y2 = _y.masked_select(mask_bool).flatten().detach().cpu().numpy()
        _y2 = _y.masked_select(mask_bool).flatten().cpu().numpy()
        # do not compute if there's only one class
        if not np.all(_x2 == _x2[0]):
            aurocs.append(roc_auc_score(_x2, _y2))
            auprcs.append(average_precision_score(_x2, _y2))
    return aurocs, auprcs


def main(path_data, num_filters, num_stacks, n_epoch, batch_size, max_length, out_dir, n_cpu):
    logging.info("Loading dataset: {}".format(path_data))
    df = []
    for _p in path_data:
        df.append(pd.read_pickle(_p))
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

    model = ResNet(ResidualBlock, num_filters, num_stacks)
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
    tr_prop = 0.95
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

    # naive guess is the mean of training target value
    yp_naive = torch.mean(torch.stack([torch.mean(y) for _, y, _ in data_loader_tr]))
    logging.info("Naive guess: {}".format(yp_naive))
    # calculate loss using naive guess
    logging.info("Naive guess performance")

    with torch.set_grad_enabled(False):
        # training
        loss_naive_tr = []
        auroc_naive_tr = []
        auprc_naive_tr = []
        for x, y, m in data_loader_tr:
            x, y, m = to_device(x, y, m, device)
            yp = torch.ones_like(y) * yp_naive
            # loss_naive_tr.append(masked_loss(yp, y, m).detach().cpu().numpy())
            loss_naive_tr.append(masked_loss(yp, y, m).item())
            _r, _p = compute_metrics(y, yp, m)
            auroc_naive_tr.extend(_r)
            auprc_naive_tr.extend(_p)
        logging.info("Training: loss {} au-ROC {} au-PRC {}".format(np.mean(np.stack(loss_naive_tr)),
                                                                    np.mean(np.stack(auroc_naive_tr)),
                                                                    np.mean(np.stack(auprc_naive_tr))))
        # validation
        loss_naive_va = []
        auroc_naive_va = []
        auprc_naive_va = []
        for x, y, m in data_loader_va:
            x, y, m = to_device(x, y, m, device)
            yp = torch.ones_like(y) * yp_naive
            # loss_naive_va.append(masked_loss(yp, y, m).detach().cpu().numpy())
            loss_naive_va.append(masked_loss(yp, y, m).item())
            _r, _p = compute_metrics(y, yp, m)
            auroc_naive_va.extend(_r)
            auprc_naive_va.extend(_p)
        logging.info("Validation: loss {} au-ROC {} au-PRC {}".format(np.mean(np.stack(loss_naive_va)),
                                                                      np.mean(np.stack(auroc_naive_va)),
                                                                      np.mean(np.stack(auprc_naive_va))))

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

        # save model
        _model_path = os.path.join(out_dir, 'model_ckpt_ep_{}.pth'.format(epoch))
        torch.save(model.state_dict(), _model_path)
        logging.info("Model checkpoint saved at: {}".format(_model_path))

        # report training loss
        logging.info(
            "Epoch {}/{}, training loss (running) {}, au-ROC {}, au-PRC {}".format(epoch, n_epoch,
                                                                                   np.mean(
                                                                                       np.stack(running_loss_tr)),
                                                                                   np.mean(np.stack(running_auroc_tr)),
                                                                                   np.mean(np.stack(running_auprc_tr))))

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
                _r, _p = compute_metrics(y, yp, m)
                running_auroc_va.extend(_r)
                running_auprc_va.extend(_p)
            logging.info(
                "Epoch {}/{}, validation loss {}, au-ROC {}, au-PRC {}".format(epoch, n_epoch,
                                                                               np.mean(np.stack(running_loss_tr)),
                                                                               np.mean(np.stack(running_auroc_va)),
                                                                               np.mean(np.stack(running_auprc_va))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=str, help='Path to training data file')
    parser.add_argument('--result', type=str, help='Path to output result')
    parser.add_argument('--num_filters', nargs='*', type=int, help='Number of conv filters per each res block.')
    parser.add_argument('--num_stacks', nargs='*', type=int, help='Number of stacks per each res block.')
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
    main(args.data, args.num_filters, args.num_stacks, args.epoch, args.batch_size, args.max_length, args.result,
         args.cpu)
