
1. Download the raw data, see [raw_data/](raw_data/).

2. Run:

```bash
python process_data.py --input raw_data/archiveII_all/train.pickle raw_data/archiveII_all/val.pickle raw_data/archiveII_all/test.pickle --output data/archiveII.pkl.gz
```

```
raw_data/archiveII_all/train.pickle 3180
raw_data/archiveII_all/val.pickle 398
raw_data/archiveII_all/test.pickle 397
output size: 3975
```


```bash
python process_data.py --input raw_data/rnastralign_all/train.pickle raw_data/rnastralign_all/val.pickle raw_data/rnastralign_all/test.pickle --output data/rnastralign.pkl.gz
```

```
raw_data/rnastralign_all/train.pickle 29719
raw_data/rnastralign_all/val.pickle 3715
raw_data/rnastralign_all/test.pickle 3715
output size: 37149
```


3. Upload data

```bash
dcl upload -s '{"description": "Processed dataset of RNA SS: archiveII."}' data/archiveII.pkl.gz
```

DC ID `7jsQW`

```bash
dcl upload -s '{"description": "Processed dataset of RNA SS: rnastralign."}' data/rnastralign.pkl.gz
```

DC ID ``


--------------

Processed data can be loaded into dataframe by `df = pd.read_pickle(file_name)`.
It looks like this:

```
                                                    seq                                            one_idx                                            seq_id                          source_file
0     GAGGAAAGUCCGGGCUCCAUAGGGCAGGGUGCCAGGUAACGCCUGG...  [[5, 299], [6, 298], [7, 297], [9, 296], [10, ...                                   RNaseP_CPB80.ct  raw_data/archiveII_all/train.pickle
1     GUUUCGGUGGUCAUAGCGUAGGGGAAACGCCCGGUUACAUUUCGAA...  [[0, 118], [1, 117], [3, 116], [4, 115], [5, 1...                   5s_Streptomyces-coelicolor-2.ct  raw_data/archiveII_all/train.pickle
2     GGGGAUGACAGGCUAUCGACAGGAUAGGUGUGAGAUGUCGUUGCAC...  [[0, 399], [1, 398], [2, 397], [3, 396], [4, 3...              tmRNA_Chlo.tepi._TRW-194439_1-404.ct  raw_data/archiveII_all/train.pickle
3     GCAGUACUGGUGUAAUGGUUAGCGCCUUAGAUUUCCAAUCUAAAGG...  [[0, 70], [1, 69], [2, 68], [3, 67], [4, 66], ...  tRNA_tdbR00000122-Codium_fragile-3133-Gly-UCC.ct  raw_data/archiveII_all/train.pickle
4     GGGGGCGAACGGGUUCGACGGGGAUGGAGUCCCCUGGGAAGCGAGC...  [[0, 352], [1, 351], [2, 350], [3, 349], [4, 3...              tmRNA_Ther.neap._TRW-309803_1-357.ct  raw_data/archiveII_all/train.pickle
...                                                 ...                                                ...                                               ...                                  ...
3970  GCCAACGUCCAUACCACGUUGAAAACACCGGUUCUCGUCCGAUCAC...  [[0, 117], [1, 116], [2, 115], [3, 114], [4, 1...                                  5s_P_cynthiaE.ct   raw_data/archiveII_all/test.pickle
3971  UGGGGCUCUGGUCCUCUCGCAACAAUAGUUCGUGAACUCGGUCAGG...  [[0, 103], [1, 102], [2, 101], [3, 100], [4, 9...                        srp_Vibr.chol._CP000627.ct   raw_data/archiveII_all/test.pickle
3972  GGGCAAAGCGUGAGGCUGGUUUCACAGAGCAGCGACAACCUCCCUC...  [[0, 39], [1, 38], [8, 33], [9, 32], [10, 31],...                        srp_Prun.arme._CV045835.ct   raw_data/archiveII_all/test.pickle
3973  GAGGAAAGUCCGGGCGCCGUCGAAACGCGGUGGUGGGUAACACCCA...  [[5, 376], [6, 375], [7, 374], [9, 373], [10, ...                                   RNaseP_CPA58.ct   raw_data/archiveII_all/test.pickle
3974  AUCCACGGCCAUAGGACUCUGAAAGCACCGCAUCCCGUCCGAUCUG...  [[0, 117], [2, 115], [3, 114], [4, 113], [5, 1...                 5s_Efibulobasidium-albescens-1.ct   raw_data/archiveII_all/test.pickle
```
