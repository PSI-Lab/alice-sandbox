import argparse
import random
import warnings
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# from dataset import IntegerSortDataset, sparse_seq_collate_fn
from dataset_ss import BoundingBoxDataset, bb_collate_fn
from model_ss import PointerNet


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='PtrNet-bounding-box')
#
# parser.add_argument('--low', type=int, default=0, help='lowest value in dataset (default: 0)')
# parser.add_argument('--high', type=int, default=100, help='highest value in dataset (default: 100)')
# parser.add_argument('--min-length', type=int, default=5, help='minimum length of sequences (default: 5)')
# parser.add_argument('--max-length', type=int, default=10, help='maximum length of sequences (default: 20)')
# parser.add_argument('--train-samples', type=int, default=100000,
#                     help='number of samples in train set (default: 100000)')
# parser.add_argument('--test-samples', type=int, default=1000, help='number of samples in test set (default: 1000)')

parser.add_argument('--dataset', type=str, help='path to training dataset')

# parser.add_argument('--emb-dim', type=int, default=8, help='embedding dimension (default: 8)')
parser.add_argument('--num-layers', type=int, default=1, help='RNN layers (default: 1)')
parser.add_argument('--hid-dim', type=int, default=50, help='hidden dimension (default: 50)')
parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay (default: 1e-5)')

parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def masked_accuracy(output, target, mask):
    """Computes a batch accuracy with a mask (for padded sequences) """
    with torch.no_grad():
        masked_output = torch.masked_select(output, mask)
        masked_target = torch.masked_select(target, mask)
        accuracy = masked_output.eq(masked_target).float().mean()

        return accuracy


def main():
    args = parser.parse_args()

    # if args.seed is not None:
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     cudnn.deterministic = True
    #     warnings.warn('You have chosen to seed training. '
    #                   'This will turn on the CUDNN deterministic setting, '
    #                   'which can slow down your training considerably! '
    #                   'You may see unexpected behavior when restarting '
    #                   'from checkpoints.')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = True if use_cuda else False

    # load and split dataset
    # assume to be gzip compressed pickle
    df = pd.read_pickle(args.dataset, compression='gzip')
    # TODO shuffle before splitting?
    len_train = int(len(df) * 0.8)
    df_tr = df[:len_train]
    df_va = df[len_train:]
    logging.info(f"Loading dataset {args.dataset}, use {len(df_tr)} for training and {len(df_va)} for validation")

    train_set = BoundingBoxDataset(df_tr)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              collate_fn=bb_collate_fn)

    test_set = BoundingBoxDataset(df_va)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             collate_fn=bb_collate_fn)

    # extra checks
    assert train_set.feature_dim == test_set.feature_dim

    # model = PointerNet(input_dim=train_set.feature_dim, embedding_dim=args.emb_dim, hidden_size=args.emb_dim).to(device)
    model = PointerNet(input_dim=train_set.feature_dim, hidden_size=args.hid_dim, num_layers=args.num_layers).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    test_loss = AverageMeter()
    test_accuracy = AverageMeter()

    for epoch in range(args.epochs):
        # Train
        model.train()
        for batch_idx, (seq, length, target) in enumerate(train_loader):
            seq, length, target = seq.to(device), length.to(device), target.to(device)

            optimizer.zero_grad()
            log_pointer_score, argmax_pointer, mask = model(seq, length)

            # FIXME debug
            # print(log_pointer_score)
            # print(log_pointer_score.shape)
            # print(argmax_pointer)
            # print(argmax_pointer.shape)
            # print(mask)
            # print(mask.shape)
            #
            # print(target)
            # print(target.shape)

            # print(log_pointer_score)
            # print(target)

            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            loss.backward()
            optimizer.step()

            # # FIXME debug
            # print('embedding gradient', model.embedding.weight.grad)
            # print('encoder gradient', model.encoder.rnn.weight_ih_l0.grad)

            train_loss.update(loss.item(), seq.size(0))

            mask = mask[:, 0, :]
            train_accuracy.update(masked_accuracy(argmax_pointer, target, mask).item(), mask.int().sum().item())

            if batch_idx % 20 == 0:
                print('Epoch {}: Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'
                      .format(epoch, batch_idx * len(seq), len(train_loader.dataset),
                              100. * batch_idx / len(train_loader), train_loss.avg, train_accuracy.avg))

        # Test
        model.eval()
        for seq, length, target in test_loader:
            seq, length, target = seq.to(device), length.to(device), target.to(device)

            log_pointer_score, argmax_pointer, mask = model(seq, length)
            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            test_loss.update(loss.item(), seq.size(0))

            mask = mask[:, 0, :]
            test_accuracy.update(masked_accuracy(argmax_pointer, target, mask).item(), mask.int().sum().item())
        print('Epoch {}: Test\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, test_loss.avg, test_accuracy.avg))


if __name__ == '__main__':
    main()
