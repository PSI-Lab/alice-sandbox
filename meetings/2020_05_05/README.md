

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

Dataset size is shown on header. Two numbers in the cell: `n_similar\[n_identical\]`.

| overlap            | archive(3975) | bprna(13419) | pdb250(241) | rfam151(151) | rnastralign(37138) | sprocessed(5273) |
|--------------------|---------------|--------------|-------------|--------------|--------------------|------------------|
| archive(3975)      |               | 1873[1580]   | 35[22]      | 15[10]       | 13940[3796]        | 2029[1266]       |
| bprna(13419)       |               |              | 19[2]       | 120[60]      | 2000[1310]         | 1066[636]        |
| pdb250(241)        |               |              |             | 1[0]         | 55[22]             | 92[32]           |
| rfam151(151)       |               |              |             |              | 14[8]              | 6[2]             |
| rnastralign(37138) |               |              |             |              |                    | 1438[574]        |
| sprocessed(5273)   |               |              |             |              |                    |                  |

See https://github.com/PSI-Lab/alice-sandbox/tree/a9dacca82df1ebbd9375ddb10b3be2c9b464a2ed/rna_ss/data_processing/dataset_overlap


## E2Efold

### Paper summary

See https://github.com/deepgenomics/communal-sandbox/tree/addaae57ec9f512e8e114929b23ba1d98281212f/journal_club/ICLR_2020/tuesday/biology#rna-secondary-structure-prediction-by-learning-unrolled-algorithms

### Inference setup

`/Users/alicegao/work/other/e2efold`

Following instruction in their README with a few tweaks:

Create py3 env:

```
conda create --name e2efold python=3.7 ipython
conda activate e2efold
```

Installation:

```
pip install -e .
```

Other packages:

```
conda install scikit-learn pandas munch
```

download one of the dataset `rnastralign_all_600.tgz` and untar.

download trained models from https://drive.google.com/open?id=1m038Fw0HBGEzsvhS0mRxd0U7cGXqLAVt.

make code CPU-compatible, in experiment_rnastralign/e2e_learning_stage3.py L118, L130:

```
if not torch.cuda.is_available():
    foo = torch.load(model_path, map_location=torch.device('cpu'))
```


Their code is set up to run on the test set only, in their own format.
In order to run on other datasets or sequences, we need to extract the inference code and re-write the pipeline.



```
rm -rf .git/
# change env name in environment.yml (avoid conflict with my other envs): rna_ss -> e2efold
# why did they specify all the version IDs? env creation failed



conda env create -f environment.yml
conda activate e2efold

# downloa trained models from https://drive.google.com/open?id=1m038Fw0HBGEzsvhS0mRxd0U7cGXqLAVt
```


### Training setup

### Verify padding length

After 1-hot encoding,
all sequences in archiveII_all were padded to length 2968,
and all sequences in rnastralign were padded to length 1851.

See full report in [https://github.com/PSI-Lab/alice-sandbox/blob/9e33c2e3bb2cae7581f89c714d9e7c61ef40f5f7/rna_ss/data_processing/e2efold/README.md#extra---verify-padding-length-used-in-their-original-data](https://github.com/PSI-Lab/alice-sandbox/blob/9e33c2e3bb2cae7581f89c714d9e7c61ef40f5f7/rna_ss/data_processing/e2efold/README.md#extra---verify-padding-length-used-in-their-original-data).


## SPOT-RNA

### Inference setup
