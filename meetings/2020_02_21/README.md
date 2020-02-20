
## SPOT-RNA test set

Raw data from their Github.


Processed TS1 is a subset of https://github.com/PSI-Lab/alice-sandbox/tree/fad62c38770711a672a666bdce723b4d185a4a00/rna_ss/data_processing/spot_rna/fine_tuning/data


## Evaluate SPOT-RNA on their test set

Got the fasta file (TS1_sequences.zip from the author, or from their Github), and combine into one file:

```
unzip
rm -r TS1_sequences
cat TS1_sequences/* > spot_rna_ts1.fasta
```

See https://github.com/PSI-Lab/alice-sandbox/blob/8b75d5299542d958791439e28ac4577ca516a0dd/meetings/2020_02_21/rna_ss_test_set/spot_rna_ts1.fasta

Setup their prediction pipeline on worksrtation: /home/alice/work/SPOT-RNA

```
git clone git@github.com:jaswindersingh2/SPOT-RNA.git
cd SPOT-RNA/
rm -r .git
conda create -n spot_rna python=3.6 ipython
conda activate spot_rna
conda install tensorflow==1.14.0
while read p; do conda install --yes $p; done < requirements.txt
```

```
wget 'https://www.dropbox.com/s/dsrcf460nbjqpxa/SPOT-RNA-models.tar.gz' || wget -O SPOT-RNA-models.tar.gz 'https://app.nihaocloud.com/f/fbf3315a91d542c0bdc2/?dl=1'
tar -xvzf SPOT-RNA-models.tar.gz && rm SPOT-RNA-models.tar.gz
```

Run their model (batch mode) against all TS1 sequences:

```
scp spot_rna_ts1.fasta alice@alice-new.dg:/home/alice/work/SPOT-RNA/sample_inputs/.
```

```
mkdir outputs_ts1
python3 SPOT-RNA.py  --inputs sample_inputs/spot_rna_ts1.fasta  --outputs 'outputs_ts1/'
```

copy over the output, and evaluate the performance:

```
~/work/psi-lab-sandbox/meetings/2020_02_21/eval_spot_rna/pred/ts1(master)$ scp alice@alice-new.dg:/home/alice/work/SPOT-RNA/outputs_ts1/*bpseq .
```

Also downloaded the true labels, see https://github.com/PSI-Lab/alice-sandbox/tree/d0d8a3312cd8e20fc85019d7797cb9108c45bea0/meetings/2020_02_21/eval_spot_rna

eval their performance (see https://github.com/PSI-Lab/alice-sandbox/blob/7863ff87b1800dca3ca7dba322125b0d77e8a1f1/meetings/2020_02_21/eval_spot_rna/eval_spot_rna.py):

as reported in their paper Fig2b, their performance is really good on TS1:

```
              len  sensitivity        ppv  f_measure
count   67.000000    67.000000  67.000000  67.000000
mean    74.776119     0.618190   0.884527   0.714495
std     34.590164     0.192141   0.150040   0.172096
min     33.000000     0.166667   0.444444   0.242424
25%     48.000000     0.503163   0.830460   0.610229
50%     70.000000     0.666667   0.950000   0.777778
75%     87.500000     0.765686   1.000000   0.835271
max    189.000000     1.000000   1.000000   1.000000
```

In contrast, our model (trained on random sequences) doesn't perform well:
https://github.com/PSI-Lab/alice-sandbox/tree/c01772924dc474d37bfe7dd3595012cf778552b1/rna_ss/meetings/2020_02_07#check-their-test-set-ts1

Maybe fine-tuning the model will help?

## Evaluate SPOT-RNA on Rfam sequences

Converting to fasta file, using https://github.com/PSI-Lab/alice-sandbox/blob/c01772924dc474d37bfe7dd3595012cf778552b1/rna_ss/data_processing/rna_cg/data/rfam.pkl

```
OUT = open('rfam_150.fasta', 'w')
for _, row in df.iterrows():
    OUT.write('>{}\n{}\n'.format(row['seq_id'], row['seq']))
OUT.close()
```

Checked in at https://github.com/PSI-Lab/alice-sandbox/blob/b2ab264b43df6a8c74c48d56a99a706ceb19d273/meetings/2020_02_21/rna_ss_test_set/rfam_150.fasta

Run their model (batch mode) against rfam sequences:

```
scp rfam_150.fasta alice@alice-new.dg:/home/alice/work/SPOT-RNA/sample_inputs/.
```

```
mkdir outputs_rfam
python3 SPOT-RNA.py  --inputs sample_inputs/rfam_150.fasta  --outputs 'outputs_rfam/'
```

copy over the output, and evaluate the performance:

```
~/work/psi-lab-sandbox/meetings/2020_02_21/eval_spot_rna/pred/rfam(master)$ scp alice@alice-new.dg:/home/alice/work/SPOT-RNA/outputs_rfam/*bpseq .
```

(note that label were from already-generated rfam dataset, so we don't need to copy over)


```
        f_measure         len         ppv  sensitivity
count  151.000000  151.000000  151.000000   151.000000
mean     0.670549  136.298013    0.684496     0.693567
std      0.230741  102.036451    0.224764     0.269894
min      0.076336   23.000000    0.108696     0.053763
25%      0.485569   67.000000    0.555556     0.490000
50%      0.716049  104.000000    0.735849     0.729730
75%      0.865766  158.500000    0.870266     0.954545
max      1.000000  568.000000    1.000000     1.000000
```
(note that f_measure is first column this time)

## Evaluate our non-AR model on Rfam sequences

~/work/psi-lab-sandbox/rna_ss/eval/rnafold_mini_data_2d_ar_4

```
CUDA_VISIBLE_DEVICES=1 python make_prediction.py --model /home/alice/work/psi-lab-sandbox/rna_ss/model/rnafold_mini_data_2d_ar_4/run_2020_02_02_17_31_09/checkpoint.016.hdf5 --dataset /home/alice/work/psi-lab-sandbox/rna_ss/data_processing/rna_cg/data/rfam.pkl --output result/debug.2020_02_19.pkl
```

copy over:

```
~/work/psi-lab-sandbox/meetings/2020_02_21/eval_our_model(master)$ scp alice@alice-new.dg:/home/alice/work/psi-lab-sandbox/rna_ss/eval/rnafold_mini_data_2d_ar_4/result/debug.2020_02_19.pkl rfam_pred.pkl
```

performance:

```
0.003
        f_measure         len         ppv  sensitivity
count  144.000000  151.000000  151.000000   151.000000
mean     0.644914  136.298013    0.645528     0.625779
std      0.232888  102.036451    0.258563     0.297709
min      0.083333   23.000000    0.000000     0.000000
25%      0.457899   67.000000    0.489130     0.406300
50%      0.657413  104.000000    0.692308     0.652174
75%      0.836374  158.500000    0.820856     0.900000
max      1.000000  568.000000    1.000000     1.000000
```

(at th=0.003, worse than their model)

see

- wrap up yeast model training + code debug + pytorch GPU setup

- re-write RNA SS model in pytorch, compare with SPOT-RNA on TS0, TS1, TS2

- RNA SS toy dataset

- Vector cluster setup

- system bio papers