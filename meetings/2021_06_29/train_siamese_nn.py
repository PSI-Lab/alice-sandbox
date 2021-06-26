import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from collections import namedtuple
from utils.misc import add_column
from utils.util_global_struct import process_bb_old_to_new
from utils.rna_ss_utils import arr2db, one_idx2arr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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


class MyDataSet(Dataset):
    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
    BoundingBox = namedtuple("BoundingBox", ['bb_x', 'bb_y', 'siz_x', 'siz_y'])

    def __init__(self, df, perc_bps_negative=0.8):
        df = self.process_data(df, perc_bps_negative)
        self.len = len(df)
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        seq = row['seq']
        bp_arr_best = row['bp_arr_best']
        # randomly sample a suboptimal one
        # TODO we should skip pseudoknot ones?
        idx = np.random.randint(0, len(row['bp_arrs_other']))
        bp_arr_other = row['bp_arrs_other'][idx]

        x = self._encode_seq(seq)
        x = self.tile_and_stack(x)

        bp_arr_best = bp_arr_best[:, :, np.newaxis]
        bp_arr_other = bp_arr_other[:, :, np.newaxis]

        x1 = np.concatenate([x, bp_arr_best], axis=2)
        x2 = np.concatenate([x, bp_arr_other], axis=2)
        y = np.asarray([1])  # always hard-coded, since we'd like score1 > score2

        # we want this: channel x H x W
        x1, x2, y = torch.from_numpy(x1).float(), torch.from_numpy(x2).float(), torch.from_numpy(y).float()
        x1 = x1.permute(2, 0, 1)
        x2 = x2.permute(2, 0, 1)

        return x1, x2, y

    def get_all_bp_arrs(self, index):
        # both bp_arr_best and all of bp_arrs_other of an example
        row = self.df.iloc[index]
        seq = row['seq']
        bp_arr_best = row['bp_arr_best']
        bp_arrs_other = row['bp_arrs_other']

        x = self._encode_seq(seq)
        x = self.tile_and_stack(x)

        # we want this: batch x channel x H x W
        arrs = []
        for bp_arr in [bp_arr_best] + bp_arrs_other:
            bp_arr = bp_arr[:, :, np.newaxis]
            bp_arr = np.concatenate([x, bp_arr], axis=2)
            bp_arr = np.transpose(bp_arr, [2, 0, 1])
            arrs.append(bp_arr)
        arrs = np.asarray(arrs)
        arrs = torch.from_numpy(arrs).float()
        return arrs

    def __len__(self):
        return self.len

    def _encode_seq(self, seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        x = np.asarray([int(x) for x in list(seq)])
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    @staticmethod
    def stem_bbs2arr(bbs, seq_len):
        # TODO validate there's no conflict!
        one_idx = []
        for bb in bbs:
            for offset in range(bb.siz_x):
                x = bb.bb_x + offset
                y = bb.bb_y - offset
                one_idx.append((x, y))
        # convert to list of 2 tuples
        one_idx = list(zip(*one_idx))

        # convert to arr
        pairs, arr = one_idx2arr(one_idx, seq_len)

        return arr

    def tile_and_stack(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 4
        l = x.shape[0]
        x1 = x[:, np.newaxis, :]
        x2 = x[np.newaxis, :, :]
        x1 = np.repeat(x1, l, axis=1)
        x2 = np.repeat(x2, l, axis=0)
        return np.concatenate([x1, x2], axis=2)

    def process_data(self, df, perc_bps_negative):
        data = []
        for _, row in df.iterrows():
            seq = row.seq
            df_target = process_bb_old_to_new(row.bounding_boxes)
            df_target = df_target[df_target['bb_type'] == 'stem']

            df_stem = pd.DataFrame(row.pred_stem_bb)
            # we use df index, make sure it's contiguous
            assert df_stem.iloc[-1].name == len(df_stem) - 1

            bbs = {}
            for idx, r in df_stem.iterrows():
                bbs[idx] = self.BoundingBox(bb_x=r['bb_x'],
                                       bb_y=r['bb_y'],
                                       siz_x=r['siz_x'],
                                       siz_y=r['siz_y'])

            target_bbs = []
            for idx, r in df_target.iterrows():
                target_bbs.append(self.BoundingBox(bb_x=r['bb_x'],
                                              bb_y=r['bb_y'],
                                              siz_x=r['siz_x'],
                                              siz_y=r['siz_y']))

            df_valid_combos = pd.DataFrame(row['bb_combos'])
            # get rid of no structure
            df_valid_combos = df_valid_combos[df_valid_combos['num_bps'] > 0]

            # # TODO to speed up, we can filter by top_bps_negative here
            # df_valid_combos = add_column(df_valid_combos, 'bp_arr', ['bb_inc'],
            #                              lambda bb_idx: self.stem_bbs2arr([bbs[i] for i in bb_idx], len(seq)))

            target_bb_inc = [next(i for i, bb in bbs.items() if bb == target_bb) for target_bb in target_bbs]
            target_bb_inc = set(target_bb_inc)
            # add in annotation of target global structure
            df_valid_combos = add_column(df_valid_combos, 'bb_inc', ['bb_inc'], set)
            df_valid_combos = add_column(df_valid_combos, 'is_mfe', ['bb_inc'], lambda x: x == target_bb_inc)
            assert len(df_valid_combos[df_valid_combos['is_mfe']]) == 1

            # bp_arr_best = df_valid_combos[df_valid_combos['is_mfe']].iloc[0]['bp_arr']
            num_bps_tgt = df_valid_combos[df_valid_combos['is_mfe']].iloc[0]['num_bps']
            if perc_bps_negative:
                # bp_arrs_other = df_valid_combos[~df_valid_combos['is_mfe']].sort_values(by=['num_bps'], ascending=False)[:top_bps_negative]['bp_arr'].tolist()
                df_tmp = df_valid_combos[(~df_valid_combos['is_mfe']) & (df_valid_combos['num_bps'] >= perc_bps_negative * num_bps_tgt)]

                # TODO to speed up, we can compute this later
                df_tmp = add_column(df_tmp, 'bp_arr', ['bb_inc'],
                                             lambda bb_idx: self.stem_bbs2arr([bbs[i] for i in bb_idx], len(seq)))

                # # subsample  FIXME hard-coded
                # df_tmp = df_tmp.sample(n=min(100, len(df_tmp)))

                # subsample from different parts:
                # - from those with n_bps == tgt
                # - from those with n_bps very close (+/-1) to tgt (to increase chance we compare with these)
                # - the rest
                bp_arrs_other = []
                df_tmp1 = df_tmp[(df_tmp['num_bps'] == num_bps_tgt)]
                if len(df_tmp1) > 0:
                    df_tmp1 = df_tmp1.sample(n=min(len(df_tmp1), 30)) # FIXME hard-coded numbers
                    bp_arrs_other.extend(df_tmp1['bp_arr'].tolist())
                df_tmp2 = df_tmp[(df_tmp['num_bps'] != num_bps_tgt) & (df_tmp['num_bps'] >= num_bps_tgt-1) & (df_tmp['num_bps'] <= num_bps_tgt+1)]
                if len(df_tmp2) > 0 :
                    df_tmp2 = df_tmp2.sample(n=min(len(df_tmp2), 30))  # FIXME hard-coded numbers
                    bp_arrs_other.extend(df_tmp2['bp_arr'].tolist())
                # rest
                n_left = 100 - len(bp_arrs_other)  # FIXME hard-coded numbers
                df_tmp3 = df_tmp[(df_tmp['num_bps'] > num_bps_tgt+1) | (df_tmp['num_bps'] < num_bps_tgt-1)]
                assert len(df_tmp3) > 0
                df_tmp3 = df_tmp3.sample(n=min(len(df_tmp3), n_left))
                bp_arrs_other.extend(df_tmp3['bp_arr'].tolist())

                # bp_arrs_other = df_tmp['bp_arr'].tolist()
            else:
                bp_arrs_other = df_valid_combos[~df_valid_combos['is_mfe']]['bp_arr'].tolist()

            bp_arr_best = df_valid_combos[df_valid_combos['is_mfe']].iloc[0]['bp_arr']

            # handle corner case where there's no 'other' combos
            if len(bp_arrs_other) == 0:
                logging.warning(f"Skipping example seq={seq} with len(bp_arrs_other)={len(bp_arrs_other)}")
                continue

            data.append({
                'seq': seq,
                'bp_arr_best': bp_arr_best,
                'bp_arrs_other': bp_arrs_other,
            })
        return pd.DataFrame(data)


class ScoreNetwork(nn.Module):

    def __init__(self, num_filters, filter_width, pooling_size, in_size):
        super(ScoreNetwork, self).__init__()

        num_filters = [9] + num_filters
        filter_width = [None] + filter_width
        pooling_size = [None] + pooling_size
        cnn_layers = []
        for i, (nf, fw, psize) in enumerate(zip(num_filters[1:], filter_width[1:], pooling_size[1:])):
            cnn_layers.append(nn.Conv2d(num_filters[i], nf, kernel_size=fw, stride=1))
            # cnn_layers.append(nn.BatchNorm2d(nf))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(psize))
        # extra layers

        # cnn_layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1))) # FIXME this is wrong in batch mode?

        # TODO this only works with fixed length for now!
        # cnn_layers.append(nn.AvgPool2d(kernel_size=in_size))  # kernel size == input size (to this layer), global pooling
        # FIXME hard-coded now! TODO compute from in_size and pooling_size
        logging.warning("hard-coded global pooling kernel size for now")
        cnn_layers.append(nn.AvgPool2d(kernel_size=5))

        # cnn_layers.append(nn.Conv2d(num_filters[-1], 1, kernel_size=1, stride=1))

        self.cnn = nn.Sequential(*cnn_layers)

        self.fc = nn.Linear(num_filters[-1], 1)

        self.out = nn.Sigmoid()

    def forward_single(self, x):
        # x = self.score_network(x)
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_pair(self, x1, x2, verbose=False):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x1 = torch.squeeze(x1)
        x2 = torch.squeeze(x2)

        out = self.out(x1 - x2)

        if verbose:
            logging.info(x1)
            logging.info(x2)
            logging.info(out)
        return out


# TODO move to class level
loss_b = torch.nn.BCELoss()


def compute_accuracy(yp, y, threshold=0.5):
    yp = yp.detach().numpy()
    y = y.detach().numpy()  # in fact this is all 1's
    yp = (yp >= threshold).astype(np.float16)
    return np.sum(yp == y)/len(y)


def main(path_data, num_filters, filter_width, pooling_size, in_size, n_epoch, learning_rate, batch_size,
         perc_bps_negative, out_dir, n_cpu, path_tdata=None):
    df = pd.read_pickle(path_data, compression='gzip')

    # TODO for now global pooling only works with fixed length input, verify df with in_size!

    model = ScoreNetwork(num_filters, filter_width, pooling_size, in_size)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # learning_rate = 0.002
    # batch_size = 10
    # n_epoch = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # split into training+validation
    # shuffle rows
    df = df.sample(frac=1, random_state=5555).reset_index(drop=True)  # fixed rand seed
    # subset
    # tr_prop = 0.95
    tr_prop = 0.8   # FIXME hard-coded
    _n_tr = int(len(df) * tr_prop)
    logging.info("Using {} data for training and {} for validation".format(_n_tr, len(df) - _n_tr))
    df_tr = df[:_n_tr]
    df_va = df[_n_tr:]
    logging.info("Initializing data loader...")
    data_loader_tr = DataLoader(MyDataSet(df_tr, perc_bps_negative),
                                batch_size=batch_size,
                                shuffle=True, num_workers=n_cpu)
    data_loader_va = DataLoader(MyDataSet(df_va, perc_bps_negative),
                                batch_size=batch_size,
                                shuffle=True, num_workers=n_cpu)
    if path_tdata:
        dfts = pd.read_pickle(path_tdata, compression='gzip')
        test_dataset = MyDataSet(dfts, perc_bps_negative)
        data_loader_ts = DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True, num_workers=n_cpu
                                    )
    else:
        data_loader_ts = None

    for epoch in range(n_epoch):
        loss_all = []
        acc_all = []
        for x1, x2, y in data_loader_tr:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            yp = model.forward_pair(x1, x2)  # nbx1
            # sue to: Using a target size (torch.Size([10, 1])) that is different to the input size (torch.Size([10])) is deprecated. Please ensure they have the same size.
            y = torch.squeeze(y)
            loss = loss_b(yp, y)
            loss_all.append(loss.item())
            acc_all.append(compute_accuracy(yp, y))
            # backprop
            model.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch {epoch}/{n_epoch}, training, loss {np.mean(loss_all)}, accuracy {np.mean(acc_all)}")

        # debug, print the last mini batch
        with torch.set_grad_enabled(False):
            logging.info("Traning set prediction on one mini batch:")
            yp = model.forward_pair(x1, x2, verbose=True)

        # save model every (n_epoch/10)-th epoch
        if (epoch + 1) % max(1, n_epoch//10) == 0:
            _model_path = os.path.join(out_dir, 'model_ckpt_ep_{}.pth'.format(epoch))
            torch.save(model.state_dict(), _model_path)
            logging.info("Model checkpoint saved at: {}".format(_model_path))

        with torch.set_grad_enabled(False):
            loss_all = []
            acc_all = []
            for x1, x2, y in data_loader_va:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)
                yp = model.forward_pair(x1, x2)  # nbx1
                # sue to: Using a target size (torch.Size([10, 1])) that is different to the input size (torch.Size([10])) is deprecated. Please ensure they have the same size.
                y = torch.squeeze(y)
                loss = loss_b(yp, y)
                loss_all.append(loss.item())
                acc_all.append(compute_accuracy(yp, y))
            logging.info(f"Epoch {epoch}/{n_epoch}, validation, loss {np.mean(loss_all)}, accuracy {np.mean(acc_all)}")

        # debug, print the last mini batch
        with torch.set_grad_enabled(False):
            logging.info("Validation set prediction on one mini batch:")
            yp = model.forward_pair(x1, x2, verbose=True)

        # debug use
        if data_loader_ts:
            with torch.set_grad_enabled(False):
                loss_all = []
                acc_all = []
                for x1, x2, y in data_loader_ts:
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    y = y.to(device)
                    yp = model.forward_pair(x1, x2)  # nbx1
                    # sue to: Using a target size (torch.Size([10, 1])) that is different to the input size (torch.Size([10])) is deprecated. Please ensure they have the same size.
                    y = torch.squeeze(y)
                    loss = loss_b(yp, y)
                    loss_all.append(loss.item())
                    acc_all.append(compute_accuracy(yp, y))
                logging.info(
                    f"Epoch {epoch}/{n_epoch}, test, loss {np.mean(loss_all)}, accuracy {np.mean(acc_all)}")

            # debug, print the last mini batch
            with torch.set_grad_enabled(False):
                logging.info("Test set prediction on one mini batch (from data loader):")
                yp = model.forward_pair(x1, x2, verbose=True)

                logging.info("Test set prediction on one pair of one example (from dataset):")
                x1, x2, y = test_dataset[0]   # TODO use 0 for now
                x1 = x1.to(device)
                x2 = x2.to(device)
                yp = model.forward_pair(torch.unsqueeze(x1, 0), torch.unsqueeze(x2, 0), verbose=True)

                logging.info("Test set prediction on all structures of one example (from dataset):")
                x = test_dataset.get_all_bp_arrs(0)  # TODO use 0 for now
                x = x.to(device)
                yp = model.forward_single(x)
                logging.info(yp.squeeze())

                logging.info("Test set prediction on pair selected from all structures of one example (from dataset):")
                x1 = x[0, :, :, :]
                x2 = x[1, :, :, :]
                x1 = x1.to(device)
                x2 = x2.to(device)
                yp = model.forward_pair(torch.unsqueeze(x1, 0), torch.unsqueeze(x2, 0), verbose=True)

                logging.info("Test set prediction on single struct, iterate subet from from all structures of one example (from dataset):")
                for k in range(x.shape[0])[:5]:
                    yp = model.forward_single(torch.unsqueeze(x[k, :, :, :], 0))
                    logging.info(f"{k}: {yp.squeeze()}")

                logging.info("Test set prediction on pair selected using forward_single (from dataset):")
                yp = model.forward_single(x[:2, :, :, :])
                logging.info(f"{k}: {yp.squeeze()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        help='Path to training data file, should be in pkl.gz format')
    parser.add_argument('--result', type=str, help='Path to output result')
    parser.add_argument('--cpu', type=int, help='Number of CPU workers per data loader')
    parser.add_argument('--num_filters', nargs='*', type=int, help='Number of conv filters for each layer.')
    parser.add_argument('--filter_width', nargs='*', type=int, help='Filter width for each layer.')
    parser.add_argument('--pooling_size', nargs='*', type=int, help='Pooling size for each layer.')
    parser.add_argument('--in_size', type=int, help='Input sequence length. Fixed length for now until we figure out proper global pooling.')
    parser.add_argument('--epoch', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, help='Mini batch size')
    parser.add_argument('--perc_bps_negative', type=float, default=0.8, help='if set, second structure in the pair will be sampled from difficult ones with large number of total bps (perc * num_bps_target)')
    parser.add_argument('--tdata', type=str, default=None,
                        help='debug use, separate test set')

    args = parser.parse_args()
    set_up_logging(args.result)

    main(args.data, args.num_filters, args.filter_width, args.pooling_size, args.in_size,
         args.epoch, args.lr, args.batch_size,
         args.perc_bps_negative, args.result, args.cpu, args.tdata)
