import os
import cPickle as pickle
import numpy as np


data = []
all_files = []
for filename in os.listdir("raw_data/"):
    if filename.endswith(".ct"):
        all_files.append(os.path.join("raw_data", filename))

for filename in all_files:
    # TODO re-imeplement this segment - hard to read and inefficient!
    List = []
    fileRead = open(filename, 'r')
    line = fileRead.readline()
    while line and line.split()[0] != '1':
        line = fileRead.readline()
    while line:
        if (len(line.split()) == 6):
            list = [line.split()[0], line.split()[1], line.split()[-2], 0]
            List.append(list)
        line = fileRead.readline()
    List = np.array(List)
    flag = -1
    Stack = []
    for i in range(len(List)):
        if (i == flag):
            Stack.pop()
            if (not Stack):
                flag = -1
            else:
                flag = Stack[-1]
        if (List[i][2] != '0' and int(List[i][0]) < int(List[i][2]) and List[i][3] == '0' and int(List[i][2]) < len(
                List)):
            if (flag < 0):
                List[i][3] = '1'
                List[int(List[i][2]) - 1][3] = '2'
                flag = int(List[i][2]) - 1
                Stack.append(int(List[i][2]) - 1)
            else:
                if (int(List[i][2]) - 1 < flag):
                    List[i][3] = '1'
                    List[int(List[i][2]) - 1][3] = '2'
                    flag = int(List[i][2]) - 1
                    if (Stack[-1] == int(List[i][2])):
                        Stack.pop()
                    Stack.append(int(List[i][2]) - 1)
    flag = -1
    Stack = []
    for i in range(len(List)):
        if (i == flag):
            Stack.pop()
            if (not Stack):
                flag = -1
            else:
                flag = Stack[-1]
        if (List[i][2] != '0' and int(List[i][0]) < int(List[i][2]) and List[i][3] == '0' and int(List[i][2]) < len(
                List)):
            if (flag < 0):
                List[i][3] = '3'
                List[int(List[i][2]) - 1][3] = '4'
                flag = int(List[i][2]) - 1
                Stack.append(int(List[i][2]) - 1)
            else:
                if (int(List[i][2]) - 1 < flag):
                    List[i][3] = '3'
                    List[int(List[i][2]) - 1][3] = '4'
                    flag = int(List[i][2]) - 1
                    if (Stack[-1] == int(List[i][2])):
                        Stack.pop()
                    Stack.append(int(List[i][2]) - 1)
    for i in range(len(List)):
        if (List[i][2] != '0' and int(List[i][0]) < int(List[i][2]) and List[i][3] == '0' and int(List[i][2]) < len(
                List)):
            List[i][3] = '5'
            List[int(List[i][2]) - 1][3] = '6'

    # TODO end of to be re-implemented

    # just store the sequence array and 7-class output array
    # we'll do the encoding at training time
    assert int(List[-1][0]) == len(List), len(List)
    seq = ''.join([x[1] for x in List])
    # classes will be 1 - 7 (reserve 0 for gradient masking)
    db_class = [int(x[3]) + 1 for x in List]
    paired_pos = [int(x[2]) for x in List]
    assert all(1 <= x <= 7 for x in db_class)

    # also store the prefix of the file name, which indicates the RNA type
    # we can use this to split training/validation set to avoid 'cheating'
    rna_prefix = os.path.basename(filename).split('_')[0]
    data.append((seq, db_class, paired_pos, rna_prefix))

with open('data/data.pkl', 'wb') as f:
    pickle.dump(data, f)
