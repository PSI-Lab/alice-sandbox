from StringIO import StringIO
import zipfile
import pandas as pd


def load_single_fasta(data):
    data = data.splitlines()
    assert len(data) == 2, data
    assert data[0].startswith('>')
    seq = data[1]
    if not all([x in list('ACGU') for x in seq]):
        print("Skipping seq with ambiguious symbol: {}".format(seq))
        return None
    else:
        return seq


def load_single_struct(data):
    df = pd.read_csv(StringIO(data), skiprows=2, delimiter='\t',
                     header=None, names=['i', 'j'])  # avoid loading extra empty col
    one_idx = (df['i'].tolist(), df['j'].tolist())
    return one_idx


seq_all = dict()
label_all = dict()
anno_all = dict()


with zipfile.ZipFile('raw_data/PDB_dataset.zip') as f:
    for name in f.namelist():
        'PDB_dataset/TS2_sequences/1s9s'
        assert name.startswith('PDB_dataset')
        fields = name.split('/')
        if len(fields) != 3:
            print("Skipping {}".format(name))
            continue  # skip folder pointers
        _, category_name, seq_id = fields
        if not seq_id:
            print("Skipping {}".format(name))
            continue  # skip folder pointers
        if 'seq' in seq_id:
            print("Skipping {}".format(name))
            continue  # skip combined data
        category, data_type = category_name.split('_')
        assert data_type in ['sequences', 'labels']

        # data
        data = f.read(name)
        if data_type == 'sequences':
            seq = load_single_fasta(data)
            seq_all[seq_id] = seq
        elif data_type == 'labels':
            one_idx = load_single_struct(data)
            label_all[seq_id] = one_idx

        # anno
        if seq_id in anno_all:
            assert anno_all[seq_id] == category
        else:
            anno_all[seq_id] = category


# combine data
assert set(seq_all.keys()) == set(label_all.keys())
assert set(seq_all.keys()) == set(anno_all.keys())
df = []
for seq_id in seq_all.keys():
    seq = seq_all[seq_id]
    if not seq:
        print("Skipping seq {}".format(seq_id))
        continue
    df.append({
        'seq_id': seq_id,
        'seq': seq,
        'len': len(seq),
        'one_idx': label_all[seq_id],
        'category': anno_all[seq_id],
    })
df = pd.DataFrame(df)
df.to_pickle('data/pdb_250.pkl')
