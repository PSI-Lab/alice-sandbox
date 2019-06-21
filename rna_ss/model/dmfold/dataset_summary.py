import cPickle as pickle
import pandas as pd

with open('data/data.pkl', 'rb') as f:
    data = pickle.load(f)


print("Number of RNAs: %d" % len(data))

df = pd.Series([len(x[0]) for x in data])
print("Sequence length:")
print df.describe()

