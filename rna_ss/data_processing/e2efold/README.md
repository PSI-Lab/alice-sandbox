
## Workflow

1. Download the raw data, see [raw_data/](raw_data/).

2. Process the raw data into out internal format.
Note that some data points seem to have incompatible sequence length and matrix index and will be dropped.
See log below for more details.

**archiveII**:

```bash
python process_data.py --input raw_data/archiveII_all/train.pickle raw_data/archiveII_all/val.pickle raw_data/archiveII_all/test.pickle --output data/archiveII.pkl.gz
```

```
raw_data/archiveII_all/train.pickle 3180
raw_data/archiveII_all/val.pickle 398
raw_data/archiveII_all/test.pickle 397
output size: 3975
```

**rnastralign**:

```bash
python process_data.py --input raw_data/rnastralign_all/train.pickle raw_data/rnastralign_all/val.pickle raw_data/rnastralign_all/test.pickle --output data/rnastralign.pkl.gz
```

```
raw_data/rnastralign_all/train.pickle 29719
raw_data/rnastralign_all/val.pickle 3715
raw_data/rnastralign_all/test.pickle 3715
Skipping suspicious data: RNA_SS_data(seq=array([[0, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 0],
       ...,
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]]), ss_label=array([[0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       ...,
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]), length=114, name='./RNAStrAlign/5S_rRNA_database/Bacteria/B00524.ct', pairs=[[1, 115], [2, 114], [3, 113], [5, 111], [6, 110], [7, 109], [8, 108], [14, 66], [15, 65], [16, 63], [17, 62], [18, 61], [19, 60], [20, 59], [27, 53], [28, 52], [29, 49], [30, 48], [31, 47], [32, 46], [46, 32], [47, 31], [48, 30], [49, 29], [52, 28], [53, 27], [59, 20], [60, 19], [61, 18], [62, 17], [63, 16], [65, 15], [66, 14], [68, 104], [69, 103], [71, 101], [72, 100], [77, 95], [78, 94], [79, 93], [80, 92], [81, 91], [82, 90], [83, 89], [84, 88], [88, 84], [89, 83], [90, 82], [91, 81], [92, 80], [93, 79], [94, 78], [95, 77], [100, 72], [101, 71], [103, 69], [104, 68], [108, 8], [109, 7], [110, 6], [111, 5], [113, 3], [114, 2], [115, 1]])

Skipping suspicious data: RNA_SS_data(seq=array([[1, 0, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0],
       ...,
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]]), ss_label=array([[0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       ...,
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]), length=117, name='./RNAStrAlign/5S_rRNA_database/Eukaryota/E00071.ct', pairs=[[0, 118], [1, 117], [2, 116], [3, 115], [4, 114], [5, 113], [6, 112], [7, 111], [8, 110], [13, 65], [14, 64], [15, 62], [16, 61], [19, 58], [20, 57], [26, 52], [27, 51], [28, 47], [29, 46], [30, 45], [31, 44], [44, 31], [45, 30], [46, 29], [47, 28], [51, 27], [52, 26], [57, 20], [58, 19], [61, 16], [62, 15], [64, 14], [65, 13], [67, 108], [68, 107], [69, 106], [70, 105], [71, 104], [78, 98], [79, 97], [80, 96], [81, 95], [82, 94], [84, 93], [85, 92], [86, 91], [91, 86], [92, 85], [93, 84], [94, 82], [95, 81], [96, 80], [97, 79], [98, 78], [104, 71], [105, 70], [106, 69], [107, 68], [108, 67], [110, 8], [111, 7], [112, 6], [113, 5], [114, 4], [115, 3], [116, 2], [117, 1], [118, 0]])

Skipping suspicious data: RNA_SS_data(seq=array([[1, 0, 0, 0],
       [0, 0, 1, 0],
       [1, 0, 0, 0],
       ...,
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]]), ss_label=array([[0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       ...,
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]), length=117, name='./RNAStrAlign/5S_rRNA_database/Eukaryota/E00049.ct', pairs=[[0, 118], [1, 117], [2, 116], [3, 115], [4, 114], [5, 113], [6, 112], [7, 111], [8, 110], [13, 65], [14, 64], [15, 62], [16, 61], [17, 60], [18, 59], [20, 57], [26, 52], [27, 51], [28, 47], [29, 46], [30, 45], [31, 44], [44, 31], [45, 30], [46, 29], [47, 28], [51, 27], [52, 26], [57, 20], [59, 18], [60, 17], [61, 16], [62, 15], [64, 14], [65, 13], [67, 108], [69, 106], [70, 105], [78, 98], [79, 97], [80, 96], [81, 95], [82, 94], [84, 93], [85, 92], [86, 91], [91, 86], [92, 85], [93, 84], [94, 82], [95, 81], [96, 80], [97, 79], [98, 78], [105, 70], [106, 69], [108, 67], [110, 8], [111, 7], [112, 6], [113, 5], [114, 4], [115, 3], [116, 2], [117, 1], [118, 0]])

Skipping suspicious data: RNA_SS_data(seq=array([[0, 0, 0, 1],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       ...,
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]]), ss_label=array([[0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       ...,
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]), length=116, name='./RNAStrAlign/5S_rRNA_database/Eukaryota/E02831.ct', pairs=[[0, 117], [2, 115], [3, 114], [5, 112], [7, 110], [8, 109], [13, 64], [14, 63], [15, 61], [16, 60], [17, 59], [18, 58], [19, 57], [20, 56], [26, 51], [27, 50], [28, 47], [29, 46], [30, 45], [31, 44], [44, 31], [45, 30], [46, 29], [47, 28], [50, 27], [51, 26], [56, 20], [57, 19], [58, 18], [59, 17], [60, 16], [61, 15], [63, 14], [64, 13], [66, 107], [67, 106], [68, 105], [69, 104], [70, 103], [71, 102], [77, 97], [78, 96], [80, 94], [81, 93], [83, 92], [84, 91], [85, 90], [90, 85], [91, 84], [92, 83], [93, 81], [94, 80], [96, 78], [97, 77], [102, 71], [103, 70], [104, 69], [105, 68], [106, 67], [107, 66], [109, 8], [110, 7], [112, 5], [114, 3], [115, 2], [117, 0]])

Skipping suspicious data: RNA_SS_data(seq=array([[0, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       ...,
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]]), ss_label=array([[0, 1, 0],
       [0, 1, 0],
       [1, 0, 0],
       ...,
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]), length=113, name='./RNAStrAlign/5S_rRNA_database/Bacteria/B00519.ct', pairs=[[2, 114], [3, 113], [5, 111], [6, 110], [7, 109], [8, 108], [14, 66], [15, 65], [16, 63], [17, 62], [18, 61], [19, 60], [20, 59], [27, 53], [28, 52], [29, 49], [30, 48], [31, 47], [32, 46], [46, 32], [47, 31], [48, 30], [49, 29], [52, 28], [53, 27], [59, 20], [60, 19], [61, 18], [62, 17], [63, 16], [65, 15], [66, 14], [68, 104], [69, 103], [71, 101], [72, 100], [77, 95], [78, 94], [79, 93], [80, 92], [81, 91], [82, 90], [83, 89], [84, 88], [88, 84], [89, 83], [90, 82], [91, 81], [92, 80], [93, 79], [94, 78], [95, 77], [100, 72], [101, 71], [103, 69], [104, 68], [108, 8], [109, 7], [110, 6], [111, 5], [113, 3], [114, 2]])

Skipping suspicious data: RNA_SS_data(seq=array([[0, 0, 0, 1],
       [0, 0, 0, 1],
       [0, 1, 0, 0],
       ...,
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]]), ss_label=array([[0, 1, 0],
       [1, 0, 0],
       [0, 1, 0],
       ...,
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]), length=119, name='./RNAStrAlign/5S_rRNA_database/Eukaryota/E01982.ct', pairs=[[0, 119], [2, 117], [3, 116], [4, 115], [5, 114], [6, 113], [7, 112], [8, 111], [13, 65], [14, 64], [15, 62], [16, 61], [17, 60], [18, 59], [19, 58], [20, 57], [26, 52], [27, 51], [29, 46], [30, 45], [31, 44], [44, 31], [45, 30], [46, 29], [51, 27], [52, 26], [57, 20], [58, 19], [59, 18], [60, 17], [61, 16], [62, 15], [64, 14], [65, 13], [67, 109], [68, 108], [69, 107], [70, 106], [71, 105], [79, 99], [80, 98], [81, 97], [82, 96], [83, 95], [85, 94], [86, 93], [87, 92], [92, 87], [93, 86], [94, 85], [95, 83], [96, 82], [97, 81], [98, 80], [99, 79], [105, 71], [106, 70], [107, 69], [108, 68], [109, 67], [111, 8], [112, 7], [113, 6], [114, 5], [115, 4], [116, 3], [117, 2], [119, 0]])

Skipping suspicious data: RNA_SS_data(seq=array([[0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0],
       ...,
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]]), ss_label=array([[1, 0, 0],
       [0, 1, 0],
       [0, 1, 0],
       ...,
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]), length=113, name='./RNAStrAlign/5S_rRNA_database/Bacteria/B04795.ct', pairs=[[1, 113], [2, 112], [3, 111], [4, 110], [5, 109], [7, 107], [13, 65], [14, 64], [15, 62], [16, 61], [17, 60], [18, 59], [19, 58], [20, 57], [26, 52], [27, 51], [28, 48], [29, 47], [30, 46], [31, 45], [45, 31], [46, 30], [47, 29], [48, 28], [51, 27], [52, 26], [57, 20], [58, 19], [59, 18], [60, 17], [61, 16], [62, 15], [64, 14], [65, 13], [67, 103], [68, 102], [70, 100], [71, 99], [76, 94], [77, 93], [79, 91], [80, 90], [81, 89], [82, 88], [88, 82], [89, 81], [90, 80], [91, 79], [93, 77], [94, 76], [99, 71], [100, 70], [102, 68], [103, 67], [107, 7], [109, 5], [110, 4], [111, 3], [112, 2], [113, 1]])

Skipping suspicious data: RNA_SS_data(seq=array([[0, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 0],
       ...,
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]]), ss_label=array([[0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       ...,
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]), length=114, name='./RNAStrAlign/5S_rRNA_database/Bacteria/B00528.ct', pairs=[[2, 116], [3, 115], [4, 114], [6, 112], [7, 111], [8, 110], [9, 109], [15, 67], [16, 66], [17, 64], [18, 63], [19, 62], [20, 61], [21, 60], [28, 54], [29, 53], [30, 50], [31, 49], [32, 48], [33, 47], [47, 33], [48, 32], [49, 31], [50, 30], [53, 29], [54, 28], [60, 21], [61, 20], [62, 19], [63, 18], [64, 17], [66, 16], [67, 15], [69, 105], [70, 104], [72, 102], [73, 101], [78, 96], [79, 95], [80, 94], [81, 93], [82, 92], [83, 91], [84, 90], [90, 84], [91, 83], [92, 82], [93, 81], [94, 80], [95, 79], [96, 78], [101, 73], [102, 72], [104, 70], [105, 69], [109, 9], [110, 8], [111, 7], [112, 6], [114, 4], [115, 3], [116, 2]])

Skipping suspicious data: RNA_SS_data(seq=array([[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0],
       ...,
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]]), ss_label=array([[0, 1, 0],
       [1, 0, 0],
       [0, 1, 0],
       ...,
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]), length=117, name='./RNAStrAlign/5S_rRNA_database/Eukaryota/E00054.ct', pairs=[[0, 117], [3, 114], [4, 113], [5, 112], [6, 111], [7, 110], [8, 109], [13, 64], [14, 63], [15, 61], [16, 60], [17, 59], [18, 58], [19, 57], [20, 56], [26, 51], [27, 50], [28, 47], [29, 46], [30, 45], [31, 44], [44, 31], [45, 30], [46, 29], [47, 28], [50, 27], [51, 26], [56, 20], [57, 19], [58, 18], [59, 17], [60, 16], [61, 15], [63, 14], [64, 13], [66, 107], [67, 106], [68, 105], [69, 104], [70, 103], [77, 97], [78, 96], [79, 95], [80, 94], [81, 93], [83, 92], [84, 91], [85, 90], [90, 85], [91, 84], [92, 83], [93, 81], [94, 80], [95, 79], [96, 78], [97, 77], [103, 70], [104, 69], [105, 68], [106, 67], [107, 66], [109, 8], [110, 7], [111, 6], [112, 5], [113, 4], [114, 3], [117, 0]])

Skipping suspicious data: RNA_SS_data(seq=array([[0, 0, 0, 1],
       [0, 1, 0, 0],
       [0, 1, 0, 0],
       ...,
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]]), ss_label=array([[0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       ...,
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]), length=116, name='./RNAStrAlign/5S_rRNA_database/Eukaryota/E00288.ct', pairs=[[0, 117], [2, 115], [4, 113], [5, 112], [6, 111], [7, 110], [8, 109], [13, 64], [14, 63], [15, 61], [16, 60], [17, 59], [18, 58], [19, 57], [20, 56], [26, 51], [27, 50], [28, 47], [29, 46], [30, 45], [31, 44], [44, 31], [45, 30], [46, 29], [47, 28], [50, 27], [51, 26], [56, 20], [57, 19], [58, 18], [59, 17], [60, 16], [61, 15], [63, 14], [64, 13], [66, 107], [67, 106], [68, 105], [69, 104], [70, 103], [77, 97], [78, 96], [79, 95], [80, 94], [81, 93], [83, 92], [84, 91], [85, 90], [90, 85], [91, 84], [92, 83], [93, 81], [94, 80], [95, 79], [96, 78], [97, 77], [103, 70], [104, 69], [105, 68], [106, 67], [107, 66], [109, 8], [110, 7], [111, 6], [112, 5], [113, 4], [115, 2], [117, 0]])

Skipping suspicious data: RNA_SS_data(seq=array([[0, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 0],
       ...,
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]]), ss_label=array([[0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       ...,
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]]), length=115, name='./RNAStrAlign/5S_rRNA_database/Bacteria/B00520.ct', pairs=[[2, 116], [3, 115], [4, 114], [6, 112], [7, 111], [8, 110], [9, 109], [15, 67], [16, 66], [17, 64], [18, 63], [19, 62], [20, 61], [21, 60], [28, 54], [29, 53], [30, 50], [31, 49], [32, 48], [33, 47], [47, 33], [48, 32], [49, 31], [50, 30], [53, 29], [54, 28], [60, 21], [61, 20], [62, 19], [63, 18], [64, 17], [66, 16], [67, 15], [69, 105], [70, 104], [72, 102], [73, 101], [78, 96], [79, 95], [80, 94], [81, 93], [82, 92], [83, 91], [84, 90], [90, 84], [91, 83], [92, 82], [93, 81], [94, 80], [95, 79], [96, 78], [101, 73], [102, 72], [104, 70], [105, 69], [109, 9], [110, 8], [111, 7], [112, 6], [114, 4], [115, 3], [116, 2]])

output size: 37138
```


3. Upload data

```bash
dcl upload -s '{"description": "Processed dataset of RNA SS: archiveII."}' data/archiveII.pkl.gz
```

DC ID `EJStGP`

```bash
dcl upload -s '{"description": "Processed dataset of RNA SS: rnastralign."}' data/rnastralign.pkl.gz
```

DC ID `277S5a`


## Use the data

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


## Extra - verify padding length used in their original data

Result:


```
raw_data/archiveII_all/train.pickle
       n_encoded_pos  n_total
count    3180.000000   3180.0
mean      212.846855   2968.0
std       190.055349      0.0
min        28.000000   2968.0
25%       110.000000   2968.0
50%       120.000000   2968.0
75%       313.250000   2968.0
max      2968.000000   2968.0

raw_data/archiveII_all/val.pickle
       n_encoded_pos  n_total
count     398.000000    398.0
mean      224.479899   2968.0
std       215.416656      0.0
min        33.000000   2968.0
25%       104.250000   2968.0
50%       121.000000   2968.0
75%       323.000000   2968.0
max      2915.000000   2968.0

raw_data/archiveII_all/test.pickle
       n_encoded_pos  n_total
count     397.000000    397.0
mean      205.876574   2968.0
std       189.977517      0.0
min        30.000000   2968.0
25%       111.000000   2968.0
50%       120.000000   2968.0
75%       318.000000   2968.0
max      2923.000000   2968.0

raw_data/rnastralign_all/train.pickle
       n_encoded_pos  n_total
count   29719.000000  29719.0
mean      499.815875   1851.0
std       559.815606      0.0
min        30.000000   1851.0
25%        93.000000   1851.0
50%       122.000000   1851.0
75%       897.500000   1851.0
max      1829.000000   1851.0

raw_data/rnastralign_all/val.pickle
       n_encoded_pos  n_total
count    3715.000000   3715.0
mean      509.972005   1851.0
std       561.982915      0.0
min        36.000000   1851.0
25%        95.500000   1851.0
50%       122.000000   1851.0
75%       963.000000   1851.0
max      1689.000000   1851.0

raw_data/rnastralign_all/test.pickle
       n_encoded_pos  n_total
count    3715.000000   3715.0
mean      511.974159   1851.0
std       563.327558      0.0
min        30.000000   1851.0
25%       107.500000   1851.0
50%       122.000000   1851.0
75%       961.500000   1851.0
max      1829.000000   1851.0

```

As shown above, after 1-hot encoding,
all sequences in archiveII_all were padded to length 2968,
and all sequences in rnastralign were padded to length 1851.


To reproduce:

```bash
python inspect_raw_data.py --input raw_data/archiveII_all/train.pickle raw_data/archiveII_all/val.pickle raw_data/archiveII_all/test.pickle raw_data/rnastralign_all/train.pickle raw_data/rnastralign_all/val.pickle raw_data/rnastralign_all/test.pickle
```



--------

TODO some data points seem to have incompatible sequence length and matrix index... needs debugging.
