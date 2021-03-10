import argparse
import os
import logging
import subprocess
import yaml
import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import copy
import pandas as pd
import dgutils.pandas as dgp
import numpy as np
import math
from sklearn.metrics import roc_auc_score


logging.basicConfig(level=logging.INFO,
                   format="%(asctime)-15s %(message)s",
                   datefmt='%Y-%m-%d %H:%M:%S')


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        # mask should be of shape batch x length, if specified
        if mask is not None:
            assert mask.size(0) == q.size(0)
            assert mask.size(1) == q.size(1)
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        if mask is not None:
            # mask out positions on output
            mask = mask.unsqueeze(-1)  # add feature dimension for broadcasting
            # print(output)
            output = output.masked_fill(mask == 0, 0)
            # print(output)
        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
#         print(mask.shape)
        mask = mask.unsqueeze(1).unsqueeze(-1)  # add dimension for heads, so we can broadcast, this makes it batch x 1 x length x 1
        mask = torch.matmul(mask, mask.transpose(-2, -1))  # use outer product to conver length-wise mask to matrix, e.g. l=5 ones will correspond to 5x5 ones matrix, this makes batch x 1 x length x length
        # print(scores)
        scores = scores.masked_fill(mask == 0, -1e9)
        # print(scores)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output
    
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x = self.dropout_1(x)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
    
        if mask is not None:
            # mask out positions on output
            mask = mask.unsqueeze(-1)  # add feature dimension for broadcasting
            # print(x[0, :, 0])
            x = x.masked_fill(mask == 0, 0)
            # print(x[0, :, 0])
        return x    


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    
class MyModel(nn.Module):
    def __init__(self, in_size, d_model, N, heads, n_hid):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(in_size, d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
        self.hid = nn.Linear(d_model, n_hid)
        self.out = nn.Linear(n_hid, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        
    def forward(self, src, mask):
        x = self.embed(src)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        # FC layers
        x = self.act1(self.hid(x))
        x = self.out(x)
        
        if mask is not None:
            # mask out positions on output before sigmoid (mask using -1e9 so the output will be ~0)
            mask = mask.unsqueeze(-1)  # add feature dimension for broadcasting
            # print(x[0, :, 0])
            x = x.masked_fill(mask == 0, -1e9)
            # print(x[0, :, 0])
        return self.act2(x)


def bb_augmentation_shift(x, offset, idx_bb_x, idx_bb_y):
    # transform bb features to mimic the effect of shifting all bounding boxes by the same (small) offset
    # we make sure that the resulting new features do not contain negative indices
    # input encoding is as in 'make_dataset': bb_type, x, y, wx, wy, median_prob, n_proposal_normalized

    # first check that offset won't result in negative location, if so, cap it
    assert len(x.shape) == 2
    loc_min = min(np.min(x[:, idx_bb_x]), np.min(x[:, idx_bb_y]))
    if offset < 0 and loc_min < np.abs(offset):
        offset = - loc_min
    
    # apply offset
    x[:, idx_bb_x] += offset
    x[:, idx_bb_y] += offset
    
    return x

    
def make_single_pred(model, x, y):
    model.eval()
    # add batch dim, convert to torch tensor, make pred
    x = torch.from_numpy(x[np.newaxis, :, :]).float()
    y = torch.from_numpy(y[np.newaxis, :]).float()
    preds = model(x, mask=None)  # no masking since parsing one example at a time for now
    return preds


def run_one_epoch(model, dataset, device, training=False, optim=None):
    if not training:
        model.eval()
    else:
        assert optim is not None
    losses = []
    aucs = []
    for x_np, y_np, m_np in tqdm.tqdm(dataset):
        # convert to torch tensor
        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()
        m = torch.from_numpy(m_np).float()
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
        preds = model(x, mask=m)

        if training:
            optim.zero_grad()

        loss = masked_loss_b(preds.squeeze(), y.squeeze(), m)
        losses.append(loss.item())

        if training:
            loss.backward()
            optim.step()

        # au-ROC
        pred_np = preds.squeeze().detach().cpu().numpy()
        for j in range(y_np.shape[0]):
            mask = m_np[j, :]
            y_true = y_np[j, :]
            y_pred = pred_np[j, :]
            y_true = y_true[mask == 1]
            y_pred = y_pred[mask == 1]

            if np.max(y_true) == np.min(y_true):
                auc = np.NaN
            else:
                auc = roc_auc_score(y_true=y_true, y_score=y_pred)
            aucs.append(auc)
    return np.mean(losses), np.nanmean(aucs)


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


class PadBatch:
    def __init__(self):
        pass

    def pad(self, batch):
        """
        args:
            batch - list of (x, y)
        reutrn:
            x after padding
            y after padding
            mask
        """
        max_len = max(map(lambda x: x[0].shape[0], batch))
        x_all = []
        y_all = []
        m_all = []
        for x, y in batch:
            current_len = x.shape[0]
            assert current_len == y.shape[0]
            x_pad = np.zeros((max_len, x.shape[1]))
            x_pad[:current_len, :] = x
            y_pad = np.zeros(max_len)
            y_pad[:current_len] = y
            mask = np.zeros(max_len)
            mask[:current_len] = 1
            x_all.append(x_pad)
            y_all.append(y_pad)
            m_all.append(mask)
        x_all = np.asarray(x_all)
        y_all = np.asarray(y_all)
        m_all = np.asarray(m_all)
        assert x_all.shape[0] == y_all.shape[0]
        assert x_all.shape[0] == m_all.shape[0]
        return x_all, y_all, m_all

    def __call__(self, batch):
        return self.pad(batch)


class MyDataSet(Dataset):

    def __init__(self, x, y):
        # both x and y are list of np arr (since they are of variable length)
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.len = len(x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


    def __len__(self):
        return self.len


loss_b = torch.nn.BCELoss(reduction='none')


def masked_loss_b(x, y, m):
    # batch x len
    l = loss_b(x, y)
    n_valid_output = torch.sum(m, dim=1)  # vector of length = batch
    loss_spatial_sum = torch.sum(torch.mul(l, m), dim=1)
    n_valid_output[n_valid_output == 0] = 1  # in case some example has all 0
    loss_spatial_mean = loss_spatial_sum / n_valid_output
    loss_batch_mean = torch.mean(loss_spatial_mean, dim=0)
    return torch.mean(loss_batch_mean)


def group_by_length(x, n_group):
    # x: list of np arr with variable length in dim 0
    # return a list of n_group lists, of indexes
    # with equal number of elements in each group
    # TODO better version: with each group covering roughly the same range of length ( max_len - min_len roughly the same)
    all_len = [len(a) for a in x]
    idx_sorted = np.argsort(all_len)
    idx_group = np.array_split(idx_sorted, n_group)
    return idx_group

    
def main(in_file, config, out_dir):
    # load config
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info("Initializing model")
    model = MyModel(in_size=config['in_size'], d_model=config['n_dim'], N=config['n_attn_layer'], heads=config['n_heads'], n_hid=config['n_hid'])
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
   
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
    
    # dataset
    dataset = np.load(in_file, allow_pickle=True)
    x_all = dataset['x']
    y_all = dataset['y']
    
    # train/validation split
    logging.info("Spliting into training and validation")
    n_tr = int(len(x_all) * 0.8)
    x_tr = x_all[:n_tr]
    x_va = x_all[n_tr:]
    y_tr = y_all[:n_tr]
    y_va = y_all[n_tr:]

    # partition by length
    data_tr = []
    idx_group = group_by_length(x_tr, n_group=config['n_length_groups'])
    for idx in idx_group:
        data_tr.append(([x_tr[i] for i in idx], [y_tr[i] for i in idx]))
    data_va = []
    idx_group = group_by_length(x_va, n_group=config['n_length_groups'])
    for idx in idx_group:
        data_va.append(([x_va[i] for i in idx], [y_va[i] for i in idx]))
    
    # data loader
    data_loader_tr = DataLoader(torch.utils.data.ConcatDataset([MyDataSet(x, y) for x, y in data_tr]),
                                batch_size=config['batch_size'],
                                shuffle=True,
                                collate_fn=PadBatch())
    data_loader_va = DataLoader(torch.utils.data.ConcatDataset([MyDataSet(x, y) for x, y in data_va]),
                                batch_size=config['batch_size'],
                                shuffle=True,
                                collate_fn=PadBatch())

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Torch device: {}".format(device))
    model = model.to(device)
    # training
    model.train()

    # out dir and metric logging
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_log_metric = []

    logging.info("Training start")
    for epoch in range(config['epoch']):
        losses, aucs = run_one_epoch(model, data_loader_tr, device, training=True, optim=optim)

        _model_path = os.path.join(out_dir, 'model_ckpt_ep_{}.pth'.format(epoch))
        torch.save(model.state_dict(), _model_path)
        logging.info("Model checkpoint saved at: {}".format(_model_path))

        logging.info("End of epoch {}, training: mean loss {}, mean au-ROC {}".format(epoch, np.mean(losses), np.nanmean(aucs)))
        df_log_metric.append({'epoch': epoch, 'tv': 'training', 'loss': np.mean(losses), 'auc': np.nanmean(aucs)})
        total_loss = 0
        # pick a random training example and print the prediction
        idx = np.random.randint(0, len(x_tr))
        pred = make_single_pred(model, x_tr[idx], y_tr[idx])
        logging.info("Training dataset idx {}\ny: {}\npred: {}".format(idx, y_tr[idx].flatten(), pred.squeeze()))

        # validation
        loss_va, aucs_va = run_one_epoch(model, data_loader_va, device, training=False)
        logging.info("End of epoch {}, validation mean loss {}, mean au-ROC {}".format(epoch, np.mean(loss_va), np.nanmean(aucs_va)))
        df_log_metric.append({'epoch': epoch, 'tv': 'validation', 'loss': np.mean(loss_va), 'auc': np.nanmean(aucs_va)})
        # pick a random validation example and print the prediction
        idx = np.random.randint(0, len(x_va))
        pred = make_single_pred(model, x_va[idx], y_va[idx])
        logging.info("Validation dataset idx {}\ny: {}\npred: {}".format(idx, y_va[idx].flatten(), pred.squeeze()))

    # export metric
    df_log_metric = pd.DataFrame(df_log_metric)
    df_log_metric.to_csv(os.path.join(out_dir, 'run_log_metric.csv'), index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Path to input .npz file, should be output from stage 1 with pruning (prune_stage_1.py) then processed by make_dataset.py')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--out_dir', type=str, help='output dir for saving model checkpoint')
    
    args = parser.parse_args()
    
    # some basic logging
    logging.info("Cmd: {}".format(args))  # cmd args
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    logging.info("Current dir: {}, git hash: {}".format(cur_dir, git_hash))
    
    main(args.in_file, args.config, args.out_dir)
    

    

    
    
    