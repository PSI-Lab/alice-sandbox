import re
import numpy as np
import pandas as pd
from utils import db_to_mat

input_file = 'raw_data/S-Processed-TES.txt'
data = []


def file_gen(file_name):
    with open(file_name, 'r') as f:
        x = []
        for line in f:
            line = line.rstrip()
            if line == '' and x:
                yield x
                x = []
                continue
            x.append(line)
        if x:
            yield x


for lines in file_gen(input_file):
    assert lines[0].startswith('>')
    seq_id = re.match(string=lines[0],
                      pattern=r'^>.*SSTRAND_ID=([^;]+);').group(1)  # sometimes data has double '>'
    seq = lines[1]
    db_notation = lines[2]
    assert all([x in list('ACGUacgu') for x in seq])
    arr = db_to_mat(seq, db_notation, upper_triangular=True)
    assert arr.shape[0] == len(seq)
    assert arr.shape[1] == len(seq)

    # full matrix is too big, since it's sparse, we'll save the index of 1's
    idxes = np.where(arr == 1)

    data.append({
        'seq_id': seq_id,
        'seq': seq,
        'len': len(seq),
        'one_idx': idxes,
    })
    print(len(data), seq_id)

data = pd.DataFrame(data)

data.to_pickle('data/s_processed_test.pkl')
