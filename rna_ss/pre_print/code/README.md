# Reproduce model and result

This fold organize scripts we used to train and evaluate the model.

Version of model: `2019_09_25_1` (or `2019_09_25.1`)


## Data processing

code in [data_processing/](data_processing/) are from:
https://github.com/PSI-Lab/alice-sandbox/tree/2879d16052075d92ade1cde1fa4a0446da2cf9fc/rna_ss/data_processing/rnafold_mini_data/make_data_var_len

TODO copy over the code

```
cd data_processing
mkdir -p tmp_output
mkdir -p data
rm tmp_output/*
make all -j 16 NUM=1000000 MINLEN=10 MAXLEN=200 BATCHSIZE=5000

```

## Model Training

code in [training/](training/) are from:
https://github.com/PSI-Lab/alice-sandbox/tree/c098270b5df6be63c17ed779a2723f980c284283/rna_ss/model/rnafold_mini_data_2d_ar_2

TODO copy over the code

Make sure to run this with GPU.

```
cd training
CUDA_VISIBLE_DEVICES=0  python train.py  --config config.yml --data ../data_processing/data/rand_seqs_var_len_sample_mfe_10_200_1000000.pkl.gz
```

(in this version of the code, the best model will be stored in training/model/model.hdf5)

(We've also copied over the trained model in [model/](model/))

## Evaluation

### RFam

Prediction:

### Pseudo Knot


### Gradient Ascent

Find a non-trivial case


## My Notes

Scripts in this folder were moved from other folders where we did the original training and evaluation.
Here we note down the original commits and command, for self reference.

```
Re-write generator to make pair_matrix on-the-fly, to save memory:
2879d16052075d92ade1cde1fa4a0446da2cf9fc

Re-generate dataset, sample more sequences, but with shorter max length
rm tmp_output/*
make all -j 16 NUM=1000000 MINLEN=10 MAXLEN=200 BATCHSIZE=5000
Done, data at:
/home/alice/work/psi-lab-sandbox/rna_ss/data_processing/rnafold_mini_data/make_data_var_len/data/rand_seqs_var_len_sample_mfe_10_200_1000000.pkl.gz

```


```
Train model using new dataset (more sequences, longer)
CUDA_VISIBLE_DEVICES=0  python train.py  --config config.yml --data /home/alice/work/psi-lab-sandbox/rna_ss/data_processing/rnafold_mini_data/make_data_var_len/data/rand_seqs_var_len_sample_mfe_10_200_1000000.pkl.gz
Done, model bkup at:
/home/alice/work/psi-lab-sandbox/rna_ss/model/rnafold_mini_data_2d_ar_2/model/bkup/2019_09_25.1/
```



