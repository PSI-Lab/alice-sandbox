import _pickle as pickle
from common.data_generator import RNASSDataGenerator
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data', 
    'seq ss_label length name pairs')

all_600 = RNASSDataGenerator('./','test_no_redundant_600') 
all_1800 = RNASSDataGenerator('./','test_no_redundant_1800')

name_600 = list(map(lambda x: all_600.data[x].name, range(all_600.len))) 
seq_600 = list(map(lambda x: x.replace('.', ''), all_600.seq))

name_1800 = list(map(lambda x: all_1800.data[x].name, range(all_1800.len))) 
seq_1800 = list(map(lambda x: x.replace('.', ''), all_1800.seq))

name = name_600+name_1800 
seq = seq_600 + seq_1800

with open('rnastralign_test_no_redundant.seq', 'w') as f:
	for i in range(len(name)):
		f.write('>'+name[i]+'\n'+seq[i]+'\n')



val = RNASSDataGenerator('./','val') 
name_val = list(map(lambda x: val.data[x].name, range(val.len))) 
seq_val = list(map(lambda x: x.replace('.', ''), val.seq))
with open('rnastralign_val_no_redundant.seq', 'w') as f:
	for i in range(len(name_val)):
		f.write('>'+name_val[i]+'\n'+seq_val[i]+'\n')

train = RNASSDataGenerator('./','train') 
name_train = list(map(lambda x: train.data[x].name, range(train.len))) 
seq_train = list(map(lambda x: x.replace('.', ''), train.seq))
with open('rnastralign_train_no_redundant.seq', 'w') as f:
	for i in range(len(name_train)):
		f.write('>'+name_train[i]+'\n'+seq_train[i]+'\n')