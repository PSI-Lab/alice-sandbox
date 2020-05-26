
## Performance evaluation

### Rfam151 - SPOT-RNA

Recall that last week we run E2Efold on rfam151 and the performance was not very good:
https://github.com/PSI-Lab/alice-sandbox/tree/8e38b7f3eb3e26d2892022f004efc073293627ca/meetings/2020_05_19#rfam151

As a comparison, we also run SPOT-RNA on rfam151.
For convenience, we've created a wrapper so we can run on dataset with out internal format.
TODO check in script.

Using env `spot_rna`, inside directory `rna_ss/tools/SPOT-RNA-master`, run:

```
python pred_seq.py --format pkl --in_file /Users/alicegao/work/psi-lab-sandbox/rna_ss/data_processing/rna_cg/data/rfam.pkl  --out_file tmp/rfam151.pkl
```


### Rfam151 - RNAfold

using env `rna_ss`, inside current directory, run:

```
python run_rnafold.py  --format pkl --in_file /Users/alicegao/work/psi-lab-sandbox/rna_ss/data_processing/rna_cg/data/rfam.pkl  --out_file data/rfam151.pkl
```

No need to adjust any parameters, since it's rnafold not plfold.


### Rfam11 - Comparison

See [check_rfam151.ipynb](check_rfam151.ipynb).

SPOT-RNA is a bit better than RNAfold, both are way better than E2Efold.

Note that both SPOT-RNA and E2Efold training dataset include a subset of Rfam151.
Out of 151 datapoints,
SPOT-RNA training data (bpRNA) has 120 similar (including 60 identical),
and E2Efold training data (rnastralign) has 14 similar (include 8 identical).
See https://github.com/PSI-Lab/alice-sandbox/tree/0d877854799d99fcf1c0802c193f1abde72e4254/rna_ss/data_processing/dataset_overlap

### evaluate on 'hold-out' portion (load set intersection from dataset comparison)


See [check_rfam151_held_out.ipynb](check_rfam151_held_out.ipynb).

Similar conclusion as above.
One interesting observation is that the max f1 score of E2Efold
dropped from 1 to 0.35 after subsetting to the held-out portion (not observed for SPOT-RNA or RNAfold),
which suggests that there is a possibility that E2Efold model memorized some structures in training data.

### Sprocessed - E2Efold

Using env `spot_rna`, inside directory `rna_ss/tools/SPOT-RNA-master`, run:

```
python pred_seq.py --format pkl --in_file /Users/alicegao/work/psi-lab-sandbox/rna_ss/data_processing/rna_cg/data/s_processed.pkl  --out_file tmp/s_processed.pkl
```

### Sprocessed - SPOT-RNA

Using env `spot_rna`, inside directory `rna_ss/tools/SPOT-RNA-master`, run:

```
python pred_seq.py --format pkl --in_file /Users/alicegao/work/psi-lab-sandbox/rna_ss/data_processing/rna_cg/data/s_processed.pkl  --out_file tmp/s_processed.pkl
```

Too slow on CPU, switch to run on work station GPU and copy back result:

```
CUDA_VISIBLE_DEVICES=0 python pred_seq.py --format pkl --in_file ../../data_processing/rna_cg/data/s_processed.pkl  --out_file tmp/s_processed.pkl
```

```
(spot_rna) alicegao@Alices-MacBook-Pro:~/work/psi-lab-sandbox/rna_ss/tools/SPOT-RNA-master/tmp(master)$ scp alice@alice-new.dg:/home/alice/work/psi-lab-sandbox/rna_ss/tools/SPOT-RNA-master/tmp/s_processed* .
```

### Rfam151 - RNAfold

using env `rna_ss`, inside current directory, run:

```
python run_rnafold.py  --format pkl --in_file /Users/alicegao/work/psi-lab-sandbox/rna_ss/data_processing/rna_cg/data/s_processed.pkl  --out_file data/s_processed.pkl
```

No need to adjust any parameters, since it's rnafold not plfold.


### Sprocessed - Comparison & held-out

See [check_sprocessed_held_out.ipynb](check_sprocessed_held_out.ipynb)


E2Efold performance much worse than SPOT-RNA and RNAfold,
it also has the most significant drop after subsetting to held-out sequences.
(although not as bad as rfam151)



## TODOs

Sprocessed - E2Efold  - 1800 model

e2efold: debug matrix index, make sure no bugs

SCFG & CLLM

contact e2efold author (select a few sequences, short toy example, good & bad performnce) be polite!

single sequence v.s. distribution

CVAE

"Grammar Variational Autoencoder"
https://arxiv.org/pdf/1703.01925.pdf

compare number of parameters of E2Efold v.s. SPOT-RNA

does SPOT-RNA's no-constraint setup make sense? Any weird result?

probing data: 293

read new paper

read paper "Linearly Constrained Neural Networks"
https://arxiv.org/pdf/2002.01600.pdf


Find short RNAs: short, so that we're confident enough that they fold into a fixed structure under a certain condition

293 dataset - in progress

non-coding RNAs?

report RNAfold performance

check SPOT-RNA & E2Efold performance

Modeling: CVAE, other, differentiable F1

CD-HIT-ES: get the tool set up, in case we need to run alignment in the future


Vector cluster setup: CPUs?


How to avoid DeepNN overfit


probing, different tissues, latent variable, captures 1. thermodynamic, 2. trans-factor

