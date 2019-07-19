import gzip
import yaml
import pandas as pd
import numpy as np
import deepgenomics.pandas.v1 as dataframe
import datacorral as dc
from scipy.stats import pearsonr
from dgutils.pandas import read_dataframe, write_dataframe, add_column, add_columns


input_data_names = [
    ('k562_vitro', 'dms'),
    ('k562_vivo1', 'dms'),
    ('k562_vivo2', 'dms'),
]

# columns for computing rep correlation
for_corr = ['k562_vivo1', 'k562_vivo2']

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)


dfs_input = []
for data_name, data_type in input_data_names:
    _, _df = read_dataframe(gzip.open('data/{}.csv.gz'.format(data_name)))

    assert _df['transcript'].nunique() == len(_df)
    print(data_name, len(_df))

    dfs_input.append(_df)

# make a "master table" that has the transcript and disjoint intervals
# then we can join all data onto this
df_master = pd.concat([x[['disjoint_intervals', 'transcript', 'transcript_id']] for x in dfs_input]).drop_duplicates(
    subset=['transcript', 'transcript_id'])  # can't use the list
print("master table len: {}".format(len(df_master)))

for _d, _df in zip(input_data_names, dfs_input):
    data_name = _d[0]
    # only keep the join key and the value column
    _df = _df[['transcript', 'reactivity_data']].rename(columns={'reactivity_data': data_name})
    df_master = pd.merge(df_master, _df, on=['transcript'], how='left')


# combine all data rows into one


def _combine_array(*arr_in):
    # at least one of them will be non-missing, use that one to figure out the length
    # in order to initialize array of all missing values
    a = next(x for x in arr_in if type(x) != float)
    arr_len = len(a)
    arr_out = [x if type(x) != float else [config['missing_val_reactivity'] for _ in
                                                range(arr_len)] for x in arr_in]
    return tuple(arr_out)


# I don't think write_dataframe can do list of list....
# so let's just fill in the missing arr, but keep them separate
data_names = [x[0] for x in input_data_names]
df_master = add_columns(df_master, data_names, data_names, _combine_array)


# rep correlation


def compute_transcript_correlation(col1, col2):
    _y1 = np.asarray(col1)
    _y2 = np.asarray(col2)
    # select positions where both are not -1 (missing val indicator)
    idx = np.where((_y1 != -1) & (_y2 != -1))
    y1 = _y1[idx]
    y2 = _y2[idx]
    corr, pval = pearsonr(y1, y2)
    return corr, pval


df_master = add_columns(df_master, ['pearson_corr', 'pearson_pval'], for_corr, compute_transcript_correlation)


# add expression


def parse_expression_data(expression_file, cell_line):
    # parse gene expression file downloaded from protein atlas
    # rna_celline.tsv
    df = pd.read_csv(expression_file, sep='\t')
    df = df[['Gene name', 'Sample', 'Value']].rename(columns={'Gene name': 'gene_name',
                                                              'Sample': 'cell_line',
                                                              'Value': 'tpm'})
    assert cell_line in df['cell_line'].unique()
    df = df[df['cell_line'] == cell_line].drop(columns=['cell_line'])
    # convert the two column df to dict, gene_name -> tpm
    return pd.Series(df.tpm.values, index=df.gene_name).to_dict()


gene2tpm = parse_expression_data(dc.Client().get_path(config['cell_line_gene_expression']), config['cell_line_name'])
df_master = add_column(df_master, 'tpm', ['transcript'],
                       lambda x: gene2tpm.get(x.gene.name, None))

# output
metadata = dataframe.Metadata()
metadata.version = "1"
for data_name in data_names:
    metadata.encoding[data_name] = dataframe.Metadata.LIST
metadata.encoding["transcript"] = dataframe.Metadata.GENOMEKIT
metadata.encoding["disjoint_intervals"] = dataframe.Metadata.LIST_OF_GENOMEKIT

write_dataframe(metadata, df_master, 'data/k562_combined.csv')
