import os
import pandas as pd
import deepgenomics.pandas.v1 as dataframe
from dgutils.pandas import read_dataframe, write_dataframe, add_column, add_columns


data = []

# comparative rna web data
for filename in os.listdir('raw_data/comparative_rna_web'):
    if filename.endswith(".txt"):
        data_name = filename.replace('.txt', '')
        _df = pd.read_csv(os.path.join('raw_data/comparative_rna_web', filename),
                          skiprows=4, sep='\s+', header=None, names=['position', 'base', 'paired_position'])
        sequence = ''.join(_df['base'])
        paired_position = _df['paired_position'].tolist()
        paired = [0 if x == 0 else 1 for x in paired_position]
        data.append({
            'dataset': 'comparative_rna_web',
            'data_name': data_name,
            'sequence': sequence,
            'paired_position': paired_position,
            'paired': paired,
        })


# dm fold data
for filename in os.listdir('raw_data/dm_fold'):
    if filename.endswith(".ct"):
        data_name = filename.replace('.ct', '')
        _df = pd.read_csv(os.path.join('raw_data/dm_fold', filename),
                          skiprows=1, sep='\t', header=None,
                          names=['position', 'base', 'position_m1', 'position_p1', 'paired_position', 'position_2'])
        sequence = ''.join(_df['base'])
        paired_position = _df['paired_position'].tolist()
        paired = [0 if x == 0 else 1 for x in paired_position]
        data.append({
            'dataset': 'dm_fold',
            'data_name': data_name,
            'sequence': sequence,
            'paired_position': paired_position,
            'paired': paired,
        })

df = pd.DataFrame(data)

# output
metadata = dataframe.Metadata()
metadata.version = "1"
metadata.encoding['paired_position'] = dataframe.Metadata.LIST
metadata.encoding['paired'] = dataframe.Metadata.LIST
write_dataframe(metadata, df, 'data/yeast_solved_structure.csv')
