import sys
import pandas as pd

in_file = sys.argv[1]


df = pd.read_csv(in_file, skiprows=1, header=None, names=['s', 'dummy_1', 'dummy_2', 'j', 'i'], sep='\t')
print(df)

seq = ''.join(df['s'].tolist())
db_str = ['.'] * len(seq)

for _, row in df.iterrows():
    if row['j'] == 0:  # unpaired
        continue
    elif row['i'] > row['j']:  # do not consider lower triangular matrix since it's redundant
        continue
    # extract paired position in 0-base, set left and right bracket
    # TODO doesn't work if theres's triplet or pseudo knot
    i = row['i'] - 1
    j = row['j'] - 1
    db_str[i] = '('
    db_str[j] = ')'

print(seq)
db_str = ''.join(db_str)
print(db_str)


