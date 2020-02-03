import os
import os.path
import pandas as pd


input_dir = 'PDB_dataset'
sequences = {}
labels = {}


def process_labels(fname):
    df = pd.read_csv(fname, sep=r"\s*", skiprows=1)
    assert 'i' in df.columns, df.columns
    assert 'j' in df.columns, df.columns
    # convert to 0-based
    df['i'] = df['i'] - 1
    df['j'] = df['j'] - 1
    # validate (only on non-empty df)
    if len(df) > 0:
        assert df['i'].min() >= 0, fname
        assert df['j'].min() >= 0, fname
    # can't check max since we don't know sequence length...
    return [df['i'].tolist(), df['j'].tolist()]


for dirpath, dirnames, filenames in os.walk(input_dir):
    for filename in filenames:
        # skip hacky files
        if filename in ['output.clstr', 'val2_seq', 'val_seq', 'train_seq.txt', 'combined_seq', 'output']:
            continue

        fname = os.path.join(dirpath, filename)
        # read in one byte to determine the file type
        with open(fname, 'r') as f:
            first_char = f.read(1)
            if first_char == '>':
                sequences[filename] = f.readlines()[-1].strip()
            elif first_char == '#':
                labels[filename] = process_labels(fname)
            else:
                raise ValueError(fname)

# validate
assert set(sequences.keys()) == set(labels.keys())

# make dataset
df = []
for seq_id in sequences.keys():
    df.append({
        'seq_id': seq_id,
        'sequence': sequences[seq_id],
        'len': len(sequences[seq_id]),
        'one_idx': labels[seq_id],
    })

df = pd.DataFrame(df)
df.to_pickle('data/spot_rna_fine_tuning.pkl')
