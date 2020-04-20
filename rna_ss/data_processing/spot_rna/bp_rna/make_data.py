import pandas as pd
import os
from utils import db2pairs, arr2db, pairs2idx, idx2arr


input_dir = 'raw_data/bpRNA_dataset'
df = []

for dirpath, dirnames, filenames in os.walk(input_dir):
    for filename in filenames:
        # extract dataset partition name (i.e. whether it's TR1/TS1/etc.)
        data_part_name = dirpath.split('/')[1]
        data_part_name, _ = data_part_name.split('_')

        fname = os.path.join(dirpath, filename)
        # print(fname)
        # process file name to get sequence ID and data partition (used by bpRNA)
        fname_parts = fname.split('/')
        data_partition = fname_parts[-2]
        seq_id = fname_parts[-1].replace('bpRNA_', '').replace('.st', '')

        with open(fname, 'r') as f:
            lines = f.readlines()
            # # check that first 3 lines are comments
            # assert all([x.startswith('#') for x in lines[:3]])
            # skip comment lines
            idx = next(i for i, x in enumerate(lines) if not x.startswith('#'))
            # next line is sequence
            seq = lines[idx].rstrip()
            # next line is dot-bracket notation
            db_str = lines[idx+1].rstrip()
            # print(seq)
            # print(db_str)
            pairs = db2pairs(db_str)
            # print(pairs)
            db2, result_ambiguous = arr2db(idx2arr(pairs2idx(pairs), len(seq)), verbose=True)
            # print(db2)
            if not result_ambiguous:
                pairs2 = db2pairs(db2)
                # print(pairs2)
                assert pairs == pairs2
                verified = True
            else:
                print("Can not verify conversion since db2 is ambiguous: {}\n{}\n{}\n{}\n{}".format(fname, seq, db_str, pairs, db2))
                verified = False

            df.append({
                'seq_id': seq_id,
                'data_partition': data_partition,
                'seq': seq,
                'len': len(seq),
                'one_idx': [[x[0] for x in pairs], [x[1] for x in pairs]],
                'verified': verified,
            })

df = pd.DataFrame(df)
df.to_pickle('data/bp_rna.pkl', protocol=2)  # make it py2 compatible

