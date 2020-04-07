import os
import sys
import os.path


input_dir = 'raw_data/bpRNA_dataset'

for dirpath, dirnames, filenames in os.walk(input_dir):
    for filename in filenames:
        # extract dataset partition name (i.e. whether it's TR1/TS1/etc.)
        data_part_name = dirpath.split('/')[1]
        data_part_name, _ = data_part_name.split('_')

        fname = os.path.join(dirpath, filename)
        print(fname)

        with open(fname, 'r') as f:
            lines = f.readlines()
            # check that first 3 lines are comments
            assert all([x.startswith('#') for x in lines[:3]])
            # next line is sequence
            seq = lines[3].rstrip()
            # next line is dot-bracket notation
            db_str = lines[4].rstrip()
            print(seq)
            print(db_str)
            print('')

