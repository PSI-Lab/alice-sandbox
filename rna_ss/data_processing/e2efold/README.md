
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

DC ID `qnck2`

```bash
dcl upload -s '{"description": "Processed dataset of RNA SS: rnastralign."}' data/rnastralign.pkl.gz
```

DC ID `OiEuA`


--------------

Processed data can be loaded into dataframe by `df = pd.read_pickle(file_name)`.
It looks like this:

```
                                                     seq                                            one_idx                                             seq_id                            source_file
0      CCUAGUGGCUAUGGCGGAGGGGAAACACCCGUUCCCAUCCCGAACA...  ([0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, ...  ./RNAStrAlign/5S_rRNA_database/Bacteria/B01868.ct  raw_data/rnastralign_all/train.pickle
1      GCCUACGACCAUACCACGUUGAAAACACCAGUUCUCGUCCGAUCAC...  ([0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 1...  ./RNAStrAlign/5S_rRNA_database/Eukaryota/E0024...  raw_data/rnastralign_all/train.pickle
2      GUCAACGACCAUACCAUGUUGAAAACAAAGGUUCUCGUCCCAUCAC...  ([0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 1...  ./RNAStrAlign/5S_rRNA_database/Eukaryota/E0076...  raw_data/rnastralign_all/train.pickle
3      GACGAACGCUGGCGGCGUGCUUAACACAUGCAAGUCGAACGGUGAU...  ([0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 1...  ./RNAStrAlign/16S_rRNA_database/Actinobacteria...  raw_data/rnastralign_all/train.pickle
4      UCCUCCUUAGUUCAGUCGGUAGAACGGUGGACUGUUAAUCCAUAUG...  ([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 21, 22, ...        ./RNAStrAlign/tRNA_database/tdbD00007819.ct  raw_data/rnastralign_all/train.pickle
...                                                  ...                                                ...                                                ...                                    ...
37144  UGUCGUAUUGAUAGCAUAGAGGACACACCUGUUCCCAUUCCGAACA...  ([0, 1, 2, 3, 4, 5, 6, 7, 13, 15, 16, 17, 18, ...  ./RNAStrAlign/5S_rRNA_database/Bacteria/B01325.ct   raw_data/rnastralign_all/test.pickle
37145  GACCUGGUGGCCAUGGCGGGGAAUGAUCCACCCGAUCCCAUCCCGA...  ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18...  ./RNAStrAlign/5S_rRNA_database/Bacteria/B04061.ct   raw_data/rnastralign_all/test.pickle
37146  UUGAUCCUGGCACAGUUUGAUCCUGGAACAGCUGGCGGCGUGCCUA...  ([0, 8, 34, 35, 36, 37, 38, 39, 40, 41, 42, 47...  ./RNAStrAlign/16S_rRNA_database/Bacilli/AY8266...   raw_data/rnastralign_all/test.pickle
37147  CUGCGCGUGGCUCAGCUUGGUAGAGCACUUGCUUGGGGUGCAAGAG...  ([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 22, 23, ...        ./RNAStrAlign/tRNA_database/tdbD00008387.ct   raw_data/rnastralign_all/test.pickle
37148  GCUCCCGUGGCCUAAUGGCUAGGGCAUUUGACUUCUAAUCAAGGGA...  ([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 21, 22, ...        ./RNAStrAlign/tRNA_database/tdbD00009158.ct   raw_data/rnastralign_all/test.pickle
```


--------

TODO some data points seem to have incompatible sequence length and matrix index... needs debugging.
