import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# debug data
in_file = '/Users/alicegao/work/psi-lab-sandbox/rna_ss/data_processing/rnafold_mini_data/data/rand_seqs_var_len_5_20_10.pkl.gz'
df = pd.read_pickle(in_file)
df = df.iloc[[0,4]]
# print(df)


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


# class SequenceTile(nn.Module):
#     # tile 1D seq (b x L x k) into 2D array (b x L x L x 2k)
#     def __init__(self):
#         super(SequenceTile, self).__init__()
#         self.a = nn.Parameter(torch.zeros(1))
#         self.b = nn.Parameter(torch.zeros(1))
#         self.c = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         # unfortunately we don't have automatic broadcasting yet
#         a = self.a.expand_as(x)
#         b = self.b.expand_as(x)
#         c = self.c.expand_as(x)
#         return a * torch.exp((x - b)**2 / c)


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
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(8, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0])
        self.layer3 = self.make_layer(block, 64, layers[1])
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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.sigmoid(self.fc(out))
        return out


# def make_model():
#     modules = []
#     modules.append(torch.nn.Conv2d(8, 32, 5, padding=2))
#     modules.append(torch.nn.LeakyReLU())
#     modules.append(torch.nn.BatchNorm2d(32))
#     modules.append(torch.nn.Conv2d(32, 32, 5, padding=2))
#     modules.append(torch.nn.LeakyReLU())
#     modules.append(torch.nn.Conv2d(32, 1, 1))
#     modules.append(torch.nn.Sigmoid())
#     model = nn.Sequential(*modules)
#     return model


def masked_loss(x, y, m):
    l = torch.nn.BCELoss(reduce=False)(x, y)
    # note that tensor shapes: batch x channel x H x W
    return torch.mean(torch.sum(torch.sum(torch.mul(l, m), dim=3), dim=2))


def to_device(x, y, m, device):
    return x.to(device), y.to(device), m.to(device)


model = ResNet(ResidualBlock, [2, 2, 2])
print(model)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset = MyDataSet(df)
train_loader = DataLoader(dataset, batch_size=2,
                          shuffle=True, num_workers=1,
                          collate_fn=PadCollate2D())

for epoch in range(10):
    for x, y, m in train_loader:
        x, y, m = to_device(x, y, m, device)
        yp = model(x)
        # print(y.shape, yp.shape, m.shape)
        # print(y[0, 0, :])
        # print(yp[0, 0, :])
        loss = masked_loss(yp, y, m)  # order: pred, target, mask
        print(loss)
        model.zero_grad()
        loss.backward()
        optimizer.step()


