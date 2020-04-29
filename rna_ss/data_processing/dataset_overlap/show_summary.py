import os
import pandas as pd


# hard-coded dataset size
# our internal version dataset, might be slightly smaller since some sequence/structure could not be processed
dataset_size = {
    "archive": 3975,
    "rnastralign": 37138,
    "rfam151": 151,
    "bprna": 13419,
    "sprocessed": 5273,
    "pdb250": 241,
}

# dataset names
dataset_names = sorted(dataset_size.keys())

# dictionary to store overlap values
data_overlap = {x: {y: [None, None] for y in dataset_names} for x in dataset_names}
# print(data_overlap)

for dirpath, dirnames, filenames in os.walk('data'):
    for filename in filenames:
        if not filename.endswith('_overlap.csv.gz'):
            print("skipping file {}".format(filename))
            continue
        # extract name of the two dataset
        d1, d2 = filename.replace('_overlap.csv.gz', '').split('_')
        fname = os.path.join(dirpath, filename)
        print(fname, d1, d2)

        df = pd.read_csv(fname)
        n_similar = len(df)
        n_identical = len(df[df['normalized_score'] == 1])
        data_overlap[d1][d2] = [n_similar, n_identical]
        data_overlap[d2][d1] = [n_similar, n_identical]

# report (csv format, for converting to markdown)
print(",".join(["overlap"] + ["{}({})".format(x, dataset_size[x]) for x in dataset_names]))
for i, x in enumerate(dataset_names):
    print(",".join(["{}({})".format(x, dataset_size[x])] + [""] * (i+1) + ["{}[{}]".format(*data_overlap[x][dataset_names[j]]) for j in range(i+1, len(dataset_names))]))


