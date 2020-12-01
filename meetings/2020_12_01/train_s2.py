import argparse
import logging
import subprocess
import yaml
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pandas as pd
import dgutils.pandas as dgp
import numpy as np
import math


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
            output = output.masked_fill(mask == 0, 0)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
#         print(mask.shape)
        mask = mask.unsqueeze(1).unsqueeze(-1)  # add dimension for heads, so we can broadcast, this makes it batch x 1 x length x 1
        mask = torch.matmul(mask, mask.transpose(-2, -1))  # use outer product to conver length-wise mask to matrix, e.g. l=5 ones will correspond to 5x5 ones matrix, this makes batch x 1 x length x length
        scores = scores.masked_fill(mask == 0, -1e9)
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
            x = x.masked_fill(mask == 0, 0)
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
            x = x.masked_fill(mask == 0, -1e9)
        
        return self.act2(x)
    
    
def find_match_bb(bb, df_target, bb_type):
    hit = df_target[(df_target['bb_type'] == bb_type) & (df_target['bb_x'] == bb['bb_x']) & (df_target['bb_y'] == bb['bb_y']) & (df_target['siz_x'] == bb['siz_x']) & (df_target['siz_y'] == bb['siz_y'])]
    if len(hit) > 0:
        assert len(hit) == 1
        return True
    else:
        return False

    
def make_dataset(df):
     # for the sole purpose of training, subset to example where s2 label can be generated EXACTLY
    # i.e. subset to example where s1 bb sensitivity is 100%
    df = dgp.add_column(df, 'n_bb', ['bounding_boxes'], len)
    n_old = len(df)
    df = df[df['n_bb'] == df['n_bb_found']]
    logging.info("Subset to examples with 100% S1 bb sensitivity (for now). Before {}, after {}".format(n_old, len(df)))
    
    # putting together the dataset
    # for each row:
    # encode input: a list of:
    # bb_type, x, y, wx, wy, median_prob, n_proposal_normalized  (TODO add both corners?)
    # encode output: binary label for each input 'position'
    
    x_all = []
    y_all = []

    for idx, row in df.iterrows():
        if idx % 10000 == 0:   # FIXME idx is the original idx (not counter)
            logging.info("Processed {} examples".format(idx))
        
        _x = []
        _y = []
        df_target = pd.DataFrame(row['df_target'])
        if row['bb_stem'] is not None:
            for x in row['bb_stem']:
                if find_match_bb(x, df_target, 'stem'):
                    label = 1
                else:
                    label = 0
                # 100 for stem
                _x.append([1, 0, 0, x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y'], np.median(x['prob']), len(x['prob'])/(x['siz_x']*x['siz_y'])])
                _y.append(label)
        if row['bb_iloop'] is not None:
            for x in row['bb_iloop']:
                if find_match_bb(x, df_target, 'iloop'):
                    label = 1
                else:
                    label = 0
                # 010 for iloop
                _x.append([0, 1, 0, x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y'], np.median(x['prob']), len(x['prob'])/(x['siz_x']*x['siz_y'])])
                _y.append(label)
        if row['bb_hloop'] is not None:
            for x in row['bb_hloop']:
                if find_match_bb(x, df_target, 'hloop'):
                    label = 1
                else:
                    label = 0
                # 001 for hloop, also multiple normalized n_proposal by 2 to make upper limit 1
                _x.append([0, 0, 1, x['bb_x'], x['bb_y'], x['siz_x'], x['siz_y'], np.median(x['prob']), 2*len(x['prob'])/(x['siz_x']*x['siz_y'])])
                _y.append(label)
        x_all.append(np.array(_x))
        y_all.append(np.array(_y))
    return x_all, y_all   # two lists
    
    
def make_single_pred(model, x, y):
    model.eval()
    # add batch dim, convert to torch tensor, make pred
    x = torch.from_numpy(x[np.newaxis, :, :]).float()
    y = torch.from_numpy(y[np.newaxis, :]).float()
    preds = model(x, mask=None)  # no masking since parsing one example at a time for now
    return preds
    
    
def eval_model(model, _x, _y):
    model.eval()
    total_loss = 0
    for i, (x, y) in enumerate(zip(_x, _y)):
        # add batch dim, convert to torch tensor, make pred
        x = torch.from_numpy(x[np.newaxis, :, :]).float()
        y = torch.from_numpy(y[np.newaxis, :]).float()
        preds = model(x, mask=None)  # no masking since parsing one example at a time for now
        loss = F.binary_cross_entropy(preds.squeeze(), y.squeeze())  #FIXME make sure this works for multi-example batch!
        total_loss += loss.item()
    return total_loss/len(_x)
    
    
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
    # hacky - using s2 dataset since this one has the bb sensitivity (s1 datset does not, too lazy to recompute)
    # use rfam for debug - will replace wtih bigger dataset (synthetic)
    logging.info("Loading {}".format(in_file))
    df = pd.read_pickle(in_file)
    logging.info("Loaded {} examples. Making dataset...".format(len(df)))
    x_all, y_all = make_dataset(df)
    assert len(x_all) == len(y_all)
    
    # train/validation split
    logging.info("Spliting into training and validation")
    n_tr = int(len(x_all) * 0.8)
    x_tr = x_all[:n_tr]
    x_va = x_all[n_tr:]
    y_tr = y_all[:n_tr]
    y_va = y_all[n_tr:]
    

    # training
    model.train()
    total_loss = 0
    logging.info("Training start")
    for epoch in range(config['epoch']):
        # parse one example at a time for now FIXME
        for i, (x, y) in enumerate(tqdm.tqdm(zip(x_tr, y_tr))):
            x = torch.from_numpy(x[np.newaxis, :, :]).float()
            y = torch.from_numpy(y[np.newaxis, :]).float()

            preds = model(x, mask=None)  # no masking since parsing one example at a time for now

            optim.zero_grad()

            loss = F.binary_cross_entropy(preds.squeeze(), y.squeeze())  #FIXME make sure this works for multi-example batch!

            # TODO this ignore_index seems to be useful!
    #         loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
    #         results, ignore_index=target_pad)
    
    
            loss.backward()
            optim.step()
            total_loss += loss.item()
            
            if i % 1000 == 0:
                logging.info("Processed {} examples".format(i))

        _model_path = os.path.join(out_dir, 'model_ckpt_ep_{}.pth'.format(epoch))
        torch.save(model.state_dict(), _model_path)
        logging.info("Model checkpoint saved at: {}".format(_model_path))
                
        logging.info("End of epoch {}, training: mean loss {}".format(epoch, total_loss/len(x_tr)))
        total_loss = 0
        # pick a random training example and print the prediction
        idx = np.random.randint(0, len(x_tr))
        pred = make_single_pred(model, x_tr[idx], y_tr[idx])
        logging.info("Training dataset idx {}\ny: {}\npred: {}".format(idx, y_tr[idx].flatten(), pred.squeeze()))
              
        # validation
        loss_va = eval_model(model, x_va, y_va)
        logging.info("validation mean loss {}".format(loss_va))
        # pick a random validation example and print the prediction
        idx = np.random.randint(0, len(x_va))
        pred = make_single_pred(model, x_va[idx], y_va[idx])
        logging.info("Validation dataset idx {}\ny: {}\npred: {}".format(idx, y_va[idx].flatten(), pred.squeeze()))
              
#     # FIXME debug
#     logging.info(preds.squeeze())
#     logging.info(y.squeeze())
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Path to input file, should be output from stage 1 with pruning (prune_stage_1.py)')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--out_dir', type=str, help='output dir for saving model checkpoint')
    
    args = parser.parse_args()
    
    # some basic logging
    logging.info("Cmd: {}".format(args))  # cmd args
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    logging.info("Current dir: {}, git hash: {}".format(cur_dir, git_hash))
    
    main(args.in_file, args.config, args.out_dir)
    

    

    
    
    