

## TODOs


### E2Efold

#### New prediction pipeline

Author checked in new code: https://github.com/ml4bio/e2efold/tree/42847da56b59c2ee7620eee73fb7a1dac8326076/e2efold_productive
that can be run on any sequences.

Checked out their new repo at `rna_ss/tools/e2efold_2`, using env `e2efold`.
Copy over trained models `cp ../e2efold/models_ckpt/* models_ckpt/.`.

Under folder `e2efold_productive`, follow their instructions.
Had to fix the following:

`main_short.sh`:

```
PYTHONPATH=../ python e2efold_productive_short.py -c config.json
```


`e2efold_productive_short.py`:

```
if LOAD_MODEL and os.path.isfile(e2e_model_path):
    print('Loading e2e model...')
    if not torch.cuda.is_available():
        rna_ss_e2e.load_state_dict(torch.load(e2e_model_path, map_location=torch.device('cpu')))
    else:
        rna_ss_e2e.load_state_dict(torch.load(e2e_model_path))
```

Since they provided output as ct file, we copy over the processing script we had for spot-rna:
`cp ../../SPOT-RNA-master/tmp_process_output.py  .`.
Note that we changed the delimiter to TAB, as well as column ordering, to match e2efold output format.


We'ver backup=ed their example sequences and created clean input folder:

```
mv short_seqs/ bkup/.
mkdir short_seqs
```


Put input sequence in `short_seqs`:

```
echo GGGGAAAAACCCC > short_seqs/test_seq.seq
```

Then run:

```
sh main_short.sh
python tmp_process_output.py short_cts/test_seq.seq.ct
```

Output:

```
GGGGAAAAACCCC
(.(.......).)
```


#### Toy examples (re-testing using their new pipeline)

```
GGGGGGAAAAAACCCCCC
((.((.......).).))
```

```
AAAAAAAACCCCCCUUUUUUUU
(.(............).....)
```

```
GCGCGCAAAAAAGCGCGC
((.(........).)..)
```

```
GGCUGAGCCGCAGUAGUUCUUCAGUGGCAAGCUUUAUGUCCUGACCCAGCUAAAGCUGCCAGUUGAAGAACUGUUGCCCUCUGCC
(...(.....((((.......)).))(..(((.....)..))..)(........)..((.............))..)...)....
```

Conclusion: same as before (i.e. our own inference wrapper was bug-free)


#### Create python wrapper for convenience

`rna_ss/tools/e2efold_2/e2efold_productive/pred_seq.py`
(also added pythonpath override, so this can be run without command line prefix)

Put test sequences in `tmp/test_seq_in.csv`, then run:

```
python pred_seq.py --format csv --in_file tmp/test_seq_in.csv  --out_file tmp/test_seq_out.pkl
```

convenient one-liner to quickly check output:

```
python -c "import pandas as pd; df=pd.read_pickle('tmp/test_seq_out.pkl'); print(df.iloc[0])"
```


#### Examples for investigating positional embedding




#### Pipeline for long sequences

Author said it'll be added by end of May.

#### Rfam151

Inside directory `rna_ss/tools/e2efold_2/e2efold_productive`, run:

```
python pred_seq.py --format pkl --in_file /Users/alicegao/work/psi-lab-sandbox/rna_ss/data_processing/rna_cg/data/rfam.pkl  --out_file tmp/rfam151.pkl
```

back to this directory, check result using notebook `check_rfam151.ipynb`


#### TODO overfit?


predict on a real dataset, with different RNA families




compare number of parameters of E2Efold v.s. SPOT-RNA

setup SPOT-RNA running on other dataset (also run on same/similar sequences, so we can report stratified result)

does SPOT-RNA's no-constraint setup make sense? Any weird result?

probing data: 293

read new paper

## Carry-overs


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

