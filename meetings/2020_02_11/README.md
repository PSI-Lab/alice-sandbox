

## Reproduce D-cell paper

### Small dataset

Train on single ontology term (data checked in to their github).

See https://github.com/PSI-Lab/alice-sandbox/blob/3eaa5dcbaf72772b4771cacfd8139bd53ef64f55/meetings/2020_02_11/d_cell/check_data_2.ipynb

scatter plot is weird.

- sklearn LR: https://github.com/PSI-Lab/alice-sandbox/blob/ac1412f634bcd4a250316e9e634eeeac6545a1b5/meetings/2020_02_11/d_cell/check_data_lr.ipynb
some performance, but not great

- sklearn NN: https://github.com/PSI-Lab/alice-sandbox/blob/ac1412f634bcd4a250316e9e634eeeac6545a1b5/meetings/2020_02_11/d_cell/check_data_2.ipynb
some performance, but not great

- pytorch NN: https://github.com/PSI-Lab/alice-sandbox/blob/ac1412f634bcd4a250316e9e634eeeac6545a1b5/meetings/2020_02_11/d_cell/check_data.ipynb
some performance, but not great

## check their prediction

Focus on data points where their p-val < 0.05.

- ExE - no pred pass 0.05 curoff?

- ExN 504 data points

Correlation 0.04,
see https://github.com/PSI-Lab/alice-sandbox/blob/f5f45ab9de527fa243a05a3aa4d8f3089b8b3e56/meetings/2020_02_11/d_cell/data_comparison.ipynb

- NxN 398664 (majority)

Correlation 0.23,
see https://github.com/PSI-Lab/alice-sandbox/blob/ac1412f634bcd4a250316e9e634eeeac6545a1b5/meetings/2020_02_11/d_cell/data_comparison.ipynb


- subset to "small dataset"

After intersection, 336 data points, correlation 0.16,
see https://github.com/PSI-Lab/alice-sandbox/blob/ac1412f634bcd4a250316e9e634eeeac6545a1b5/meetings/2020_02_11/d_cell/data_comparison_go.ipynb



## Train on full dataset

ExE only for now. Take median across strains with same gene but different alleles.

Also note that the dynamic range of the values

Performance is really good with fully connected net?
Suspicious?

Correlation on 1000 data points from test set 0.91,
see https://github.com/PSI-Lab/alice-sandbox/blob/ac1412f634bcd4a250316e9e634eeeac6545a1b5/meetings/2020_02_11/d_cell/full_data.ipynb


## TODOs

- make sure no bug in code

- train on whole dataset (ExE, ExN, NxN)

- add extra training example, single knockout fitness

- GNN (won't be useful if we can peak the performance using fully connected)
