

## Reproduce D-cell paper

### Small dataset

Train on single ontology term (data checked in to their github).

See https://github.com/PSI-Lab/alice-sandbox/blob/3eaa5dcbaf72772b4771cacfd8139bd53ef64f55/meetings/2020_02_11/d_cell/check_data_2.ipynb

scatter plot is weird.

TODO LR

TODO sklearn NN

TODO pytorch NN

## check their prediction

Focus on data points where their p-val < 0.05.

- ExE - no pred pass 0.05 curoff?

- ExN ~500 data points

TODO link

TODO correlation

- NxN 398664 (majority)

TODO link

TODO correlation

- subset to "small dataset"

TODO num data points

TODO link

TODO correlation


## Train on full dataset

ExE only for now. Take median across strains with same gene but different alleles.

Also note that the dynamic range of the values

Performance is really good with fully connected net?

TODO link

TODO plot

TODO correlation


## TODOs

- make sure no bug in code

- train on whole dataset (ExE, ExN, NxN)

- add extra training example, single knockout fitness

- GNN (won't be useful if we can peak the performance using fully connected)
