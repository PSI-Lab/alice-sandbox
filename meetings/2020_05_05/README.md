

## LT Dataset Comparison

### Make dataset in py2 compatible format

Done. See https://github.com/PSI-Lab/alice-sandbox/commit/d2c1e930aa56dab581ef354c554be4d13253777b.


### Dataset description

rfam_151: https://github.com/PSI-Lab/alice-sandbox/blob/e26f915f63ceabf5bcbf8db9c8d4a75aa9f9c091/rna_ss/data_processing/rna_cg/README.md#rfam_151

s_processed: https://github.com/PSI-Lab/alice-sandbox/blob/e26f915f63ceabf5bcbf8db9c8d4a75aa9f9c091/rna_ss/data_processing/rna_cg/README.md#s_processed

pdb_250: https://github.com/PSI-Lab/alice-sandbox/blob/76db250871e880f82af80349375e041746a859f9/rna_ss/data_processing/pdb_250/README.md#dataset-description


rnastralign: https://github.com/PSI-Lab/alice-sandbox/blob/a24db2ad855fb893b15dee13c954a073cfe5e42c/rna_ss/data_processing/e2efold/README.md#rnastralign

archive: https://github.com/PSI-Lab/alice-sandbox/blob/a24db2ad855fb893b15dee13c954a073cfe5e42c/rna_ss/data_processing/e2efold/README.md#archiveii

bp_rna: https://github.com/PSI-Lab/alice-sandbox/blob/f5886087dfe9afe7f0bbf7a60fb2bcdd25873715/rna_ss/data_processing/spot_rna/bp_rna/README.md#dataset-description

### Dataset Overlap annotation

TODO

### Dataset Overlap summary

TODO



## E2Efold

### Predictor setup

TODO

### Verify padding length

After 1-hot encoding,
all sequences in archiveII_all were padded to length 2968,
and all sequences in rnastralign were padded to length 1851.

See full report in [https://github.com/PSI-Lab/alice-sandbox/blob/9e33c2e3bb2cae7581f89c714d9e7c61ef40f5f7/rna_ss/data_processing/e2efold/README.md#extra---verify-padding-length-used-in-their-original-data](https://github.com/PSI-Lab/alice-sandbox/blob/9e33c2e3bb2cae7581f89c714d9e7c61ef40f5f7/rna_ss/data_processing/e2efold/README.md#extra---verify-padding-length-used-in-their-original-data).



