
## E2Efold

### Contact author for running on new sequences


### Inference setup

`rna_ss/tools/e2efold`

conda env: [env_e2efold.yml](env_e2efold.yml)

Already modified some code to get it running. See last week's notes.

Use conda env `e2efold`.

Their code is set up to run on the test set only, in their own format.
In order to run on other datasets or sequences, we need to extract the inference code and re-write the pipeline.

Update code:

    - edited from experiment_rnastralign/e2e_learning_stage3.py

    - data generator requires sequences only (their data gen require target value)

    - return predictions

    - fixed most of the hard-coded path, so we can run prediction from packge-top-level

    - single script for running short and long sequences

    - note this script only works for sequences <= 600bp (will need to update the other script for longer seq O_O)

    - NN needs input to have fixed length:
        ```
        Traceback (most recent call last):
          File "pred_seqs.py", line 369, in <module>
            a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)
          File "/Users/alicegao/anaconda2/envs/e2efold/lib/python3.7/site-packages/torch/nn/modules/module.py", line 550, in __call__
            result = self.forward(*input, **kwargs)
          File "/Users/alicegao/work/psi-lab-sandbox/rna_ss/tools/e2efold/e2efold/models.py", line 793, in forward
            u, m, lmbd_tmp, a_tmp, a_hat_tmp, t)
          File "/Users/alicegao/work/psi-lab-sandbox/rna_ss/tools/e2efold/e2efold/models.py", line 823, in update_rule
            torch.abs(a_hat_updated) - self.rho_m*self.alpha * torch.pow(self.lr_decay_alpha,t))
        RuntimeError: The size of tensor a (23) must match the size of tensor b (600) at non-singleton dimension 2
        ```

Moved the updated code to: https://github.com/PSI-Lab/alice-sandbox/blob/92a0057ad62d85fa0689654945d346d3c6e68d23/rna_ss/tools/e2efold/pred_seqs.py

Provide input csv file with column 'seq':

```
python pred_seqs.py -c config.json --in_file {in_file} --out_file {out_file}
```

TODO: --long doesn't work, model still expect padding to be 600.

```
python pred_seqs.py -c experiment_rnastralign/config.json  --in_file {in.csv} --out_file {out.pkl}
```

Output is a pickled csv file with same column 'seq' and added column 'pred':

    ```
                           seq                   pred_idx
    0  AAAAAAAAAGGGGGUUUUUUUUU  [[0, 1, 5], [22, 20, 14]]
    1     GGGGGGGUUUUUUCCCCCCC         [[4, 5], [18, 13]]
    ````


### Toy examples

In this investigation, we put input sequence in tmp/test_seq_in.csv, and run:

```
python pred_seqs.py -c experiment_rnastralign/config.json --in_file tmp/test_seq_in.csv --out_file tmp/test_seq_out.pkl
```

convenient one-liner to quickly check output:

```
python -c "import pandas as pd; df=pd.read_pickle('tmp/test_seq_out.pkl'); print(df.iloc[0])"
```


We first check whether the model works for trivial RNA sequences.
G-C pairing (note that we make sure to respect their constraints with expected loop size >= 4):
```
GGGGGGAAAAAACCCCCC
```
Typically these shouldn't be any uncertainty regarding such sequence,
RNAfold predicts:
```
((((((......))))))
```
which makes intuitive sense. E2Efold predicts:
```
((.((.......).).))
```

Another trivial sequence with A-U pairs (note that we make sure to respect their constraints with expected loop size >= 4):
```
AAAAAAAACCCCCCUUUUUUUU
```
RNAfold:
```
((((((((......))))))))
```
E2Efold output:
```
(.(............).....)
```

G-C pairing, but alternating on each side of the stem:
```
GCGCGCAAAAAAGCGCGC
```
RNAfold:
```
((((((......))))))
```
E2Efold:
```
((.(........).)..)
```




Next, we try predicting miRNA from http://www.mirbase.org/cgi-bin/mirna_entry.pl?acc=hsa-mir-22.
Input sequence:
```
GGCUGAGCCGCAGUAGUUCUUCAGUGGCAAGCUUUAUGUCCUGACCCAGCUAAAGCUGCCAGUUGAAGAACUGUUGCCCUCUGCC
```
RNAfold predicts quite accurate structure:
```
(((.(((..((((((((((((((((((((.((((((.((.........))))))))))))).))))))))))))))).))).)))
```
E2Efold predicts:
```
(...(.....((((.......)).))(..(((.....)..))..)(........)..((.............))..)...)....
```



TODO also check SPOT-RNA

TODO make sure to include both in sandbox

archiveII other families

overfitting? test set sequence ->struct -> co-mutate paired bases

adversarial attack?


### Performance of rnastralign-model on archiveII

Any processing needed on archiveII since it has overlap?

Report performance of 2 other families. Is it included in the data file?

Do we need to pad sequence to a different fixed length?

### Training setup


## SPOT-RNA

### Inference setup

`rna_ss/tools/SPOT-RNA-master`

conda env: follow the author's README

Put test seq in 'tmp/test_seq_in.fasta' then run the following:

```
python SPOT-RNA.py --inputs tmp/test_seq_in.fasta --outputs tmp/
python tmp_process_output.py tmp/seq.bpseq
```

### Toy example

Same toy example as we did for E2Efold:

In:

```
GGGGGGAAAAAACCCCCC
```

Out:

```
((((((......))))))
```

In:

```
AAAAAAAACCCCCCUUUUUUUU
```

Out:

```
(((((((........)))))))
```


In:

```
GCGCGCAAAAAAGCGCGC
```

Out:

```
((((((......))))))
```


miRNA from http://www.mirbase.org/cgi-bin/mirna_entry.pl?acc=hsa-mir-22:

```
GGCUGAGCCGCAGUAGUUCUUCAGUGGCAAGCUUUAUGUCCUGACCCAGCUAAAGCUGCCAGUUGAAGAACUGUUGCCCUCUGCC
```

Out:

```
(((((((..(((((((((((((((((((.(((((((..............))))))))))).))))))))))))))).)))))))
```


Looks way more legit than E2Efold!

## TODOs

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




