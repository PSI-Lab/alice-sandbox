#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
# from dgutils.pandas import add_column


# In[36]:


# import matplotlib.pyplot as plt 
# import seaborn as sns
# import pandas as pd
# sns.set(color_codes=True)
# import cufflinks as cf
# cf.go_offline()
# cf.set_config_file(theme='ggplot')


# In[37]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[38]:


from scipy.stats import pearsonr


# In[39]:


from torch.utils.data import Dataset, DataLoader


# In[ ]:





# In[40]:


def add_column(df, output_col, input_cols, func, pbar=False):
    # make a tuple of values of the requested input columns
    input_values = tuple(df[x].values for x in input_cols)

    # transpose to make a list of value tuples, one per row
    args = zip(*input_values)

    # Attach a progress bar if required
    if pbar:
        args = tqdm(args, total=len(df))
        args.set_description("Processing %s" % output_col)

    # evaluate the function to generate the values of the new column
    output_values = [func(*x) for x in args]

    # make a new dataframe with the new column added
    columns = {x: df[x].values for x in df.columns}
    columns[output_col] = output_values
    return pd.DataFrame(columns)


# In[ ]:





# In[41]:
df_raw = pd.read_csv('data/training/raw_pair_wise/SGA_NxN.txt', 
                     sep='\t')[['Query Strain ID', 'Array Strain ID', 
                                'Query single mutant fitness (SMF)', 'Array SMF',
                                'Double mutant fitness']].rename(columns={
    'Query Strain ID': 's1',
    'Array Strain ID': 's2',
    'Query single mutant fitness (SMF)': 'f1',
    'Array SMF': 'f2',
    'Double mutant fitness': 'fitness',
})
#df_raw = pd.concat([df_raw_1, df_raw_2, df_raw_3])


def _get_gene_id(strain_id):
    gene_id, _ = strain_id.split('_')
    return gene_id


df_raw = add_column(df_raw, 'g1', ['s1'], _get_gene_id)
df_raw = add_column(df_raw, 'g2', ['s2'], _get_gene_id)

# df_tr = pd.read_csv('data/training/GO:0006281_training_data_2016_0', 
#                     header=None, names=['foo', 'g1', 'g2', 'fitness'], sep=' ')
# df_ts = pd.read_csv('data/training/GO_0006281_testing_data_2016_0.txt', 
#                     header=None, names=['foo', 'g1', 'g2', 'fitness'], sep=' ')


# In[42]:


len(df_raw)


# In[43]:


# len(df_raw[['g1', 'g2']].drop_duplicates())


# In[44]:


df_raw = df_raw[['g1', 'g2', 'fitness']].groupby(by=['g1', 'g2'], as_index=False).agg('median')
# TODO merge in f1 and f2


# In[45]:


len(df_raw)


# In[46]:


df_raw.head()


# In[ ]:





# In[47]:


# shuffle rows
df_raw = df_raw.sample(frac=1).reset_index(drop=True)
# subset
_n_tr = int(len(df_raw)*0.8)
print(_n_tr)
df_tr = df_raw[:_n_tr]
df_ts = df_raw[_n_tr:]
# make sure genes are the same (for now drop rows that are not)
genes_tr = set(df_tr.g1.unique().tolist() + df_tr.g2.unique().tolist())
genes_ts = set(df_ts.g1.unique().tolist() + df_ts.g2.unique().tolist())
genes_intersection = genes_tr.intersection(genes_ts)
print(len(genes_tr), len(genes_ts), len(genes_intersection))
df_tr = df_tr[df_tr['g1'].isin(genes_intersection)]
df_tr = df_tr[df_tr['g2'].isin(genes_intersection)]
df_ts = df_ts[df_ts['g1'].isin(genes_intersection)]
df_ts = df_ts[df_ts['g2'].isin(genes_intersection)]
print(len(df_tr), len(df_ts))


# In[ ]:





# In[48]:


df_tr.head()


# In[49]:


df_ts.head()


# In[ ]:





# In[50]:


# # FIXME for now use all data
# # later on subset to tr and ts (make sure both have all genes?)
# df_tr = df_raw.copy()
# gene_ids = tuple(set(df_raw.g1.tolist() + df_raw.g2.tolist()))
# num_genes = len(gene_ids)


# In[ ]:





# In[51]:


# encode data
# gene_ids = tuple(genes_tr)  # use this to assign a unique index for each gene
# num_genes = len(gene_ids)

gene_ids = tuple(genes_intersection)
num_genes = len(gene_ids)


# def encode_x(g1, g2, l, gene_ids):
#     x = np.ones(l)
#     x[gene_ids.index(g1)] = 0
#     x[gene_ids.index(g2)] = 0
#     return x

# make it faster by create gene_id -> gene_idx mapping
gene_id2idx = {x: i for i, x in enumerate(gene_ids)}

# save memory
def encode_x(g1, g2, l, gene_ids):
#     return [gene_ids.index(g1), gene_ids.index(g2)]
    return [gene_id2idx[g1], gene_id2idx[g2]]


df_tr = add_column(df_tr, 'x', ['g1', 'g2'], lambda g1, g2: encode_x(g1, g2, num_genes, gene_ids))
df_ts = add_column(df_ts, 'x', ['g1', 'g2'], lambda g1, g2: encode_x(g1, g2, num_genes, gene_ids))


# In[52]:


# get data
x_tr = np.asarray(df_tr['x'].to_list())
y_tr = np.asarray(df_tr['fitness'].to_list())
x_ts = np.asarray(df_ts['x'].to_list())
y_ts = np.asarray(df_ts['fitness'].to_list())
print(x_tr.shape, y_tr.shape, x_ts.shape, y_ts.shape)
assert x_tr.shape[0] == y_tr.shape[0]
assert x_ts.shape[0] == y_ts.shape[0]
# print(x_tr.shape, y_tr.shape)


# In[53]:


# TODO data pre-preoceesing, y log transformation?


# In[54]:


# dataset
class MyDataSet(Dataset):

    def __init__(self, x, y, num_genes):
        self.num_genes = num_genes
        assert x.shape[0] == y.shape[0]
        self.len = x.shape[0]
        self.x = x
        # add new axis if needed
        if len(y.shape) == 1:
            self.y = y[:, np.newaxis]
        else:
            self.y = y

    def __getitem__(self, index):
#         return self.x[index, :], self.y[index]
        _x = self.x[index, :]
        assert len(_x) == 2
        # encode
        x = np.ones(self.num_genes)
        x[_x[0]] = 0
        x[_x[1]] = 0
        return torch.from_numpy(x).float(), torch.from_numpy(self.y[index]).float()

    def __len__(self):
        return self.len


# In[55]:


# # a fully connected net
# class Net(nn.Module):

#     def __init__(self, n_in):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(n_in, 10)
#         self.fc2 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# In[56]:


model = torch.nn.Sequential(
    torch.nn.Linear(num_genes, 20),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(20),
    torch.nn.Linear(20, 10),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(10),
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(5),
    torch.nn.Linear(5, 1),
)

# model = torch.nn.Sequential(
#     torch.nn.Linear(num_genes, 50),
#     torch.nn.ReLU(),
#     torch.nn.Linear(50, 1),
# )


# In[57]:


data_tr_loader = DataLoader(dataset=MyDataSet(x_tr, y_tr, num_genes),
                         batch_size=200, shuffle=True)
data_ts_loader = DataLoader(dataset=MyDataSet(x_ts, y_ts, num_genes),
                         batch_size=1000, shuffle=True)


# In[58]:


# net = Net(num_genes)
# print(net)


# In[59]:


loss_fn = torch.nn.MSELoss(reduction='mean')


# In[60]:


learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[61]:


device = torch.device("cpu")


# In[62]:


# net_input = torch.randn(1, 1, 32, 32)
# net_output = net(net_input)
# target = torch.randn(10)  # a dummy target, for example
# target = target.view(1, -1)  # make it the same shape as output
# criterion = nn.MSELoss()

# loss = criterion(output, target)


# In[ ]:


# inital test performance
with torch.set_grad_enabled(False):
    for xt, yt in data_ts_loader:
        yt_pred = model(xt)
        loss = loss_fn(yt_pred, yt)
        print('initial test: ', loss.item())
        # just run one batch (otherwise takes too long)
        break
            
for epoch in range(20):
    # Training
    for x_batch, y_batch in data_tr_loader:
        y_batch_pred = model(x_batch)
        loss = loss_fn(y_batch_pred, y_batch)
        print(epoch, loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()
    # after epoch
    print('training batch corr')
    print(pearsonr(y_batch.detach().numpy()[:, 0], y_batch_pred.detach().numpy()[:, 0]))

    # test
    with torch.set_grad_enabled(False):
#         for xt, yt in data_ts_loader:
#             yt_pred = model(xt)
#             loss = loss_fn(yt_pred, yt)
#             print('test: ', loss.item())
        # using the first batch for now - this probably shuffles?
        for xt, yt in data_ts_loader:
#         xt, yt = data_ts_loader[0]
            yt_pred = model(xt)
            loss = loss_fn(yt_pred, yt)
            print('test: ', loss.item())
            print('test batch (1000 data points) corr')
            print(pearsonr(yt.numpy()[:, 0], yt_pred.numpy()[:, 0]))
            print('')
            break


print('done training')
print('training batch')
print(pearsonr(y_batch.detach().numpy()[:, 0], y_batch_pred.detach().numpy()[:, 0]))


print('test batch (1000 data points)')
print(pearsonr(yt.numpy()[:, 0], yt_pred.numpy()[:, 0]))


# In[ ]:




