

## Probing dataset comparison

Per-transcript correlation between different datasets:
[dataset_comparison.html](dataset_comparison.html)


produced by: https://github.com/deepgenomics/communal-sandbox/blob/d963178391509f5d7db3631dff6b09adbca3c9a1/analysis/rna_ss/evaluate_probing_dataset/dataset_comparison_all.ipynb


TODO:

    - subset to transcripts with high coverage

    - 'sanity check' on short RNAs, ~20bp, compare with thermodynamic folding



## dataset overlapping

WIP, pending processing of bpRNA dataset.

Check identical sequence overlapping:

```
len(df_rfam)
151
len(df_rnastralign)
37138
len(df_s_processed)
6247
df_s_processed
len(df_pdb_250)
241
```

```
rfam rnastralign 17
rfam sprocessed 1
rfam pdb250 0
rnastralign sprocessed 878
rnastralign pdb250 71
sprocessed pdb250 69
```


TODO:

    - highly similar sequences


## bpRNA



