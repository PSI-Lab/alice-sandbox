import yaml
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import datacorral as dc
import deepgenomics.pandas.v1 as dataframe
from genome_kit import Genome, GenomeTrack, Interval, GenomeTrackBuilder
from dgutils.pandas import read_dataframe, write_dataframe, add_column, add_columns


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


def get_diseq_data(diseq, gtrack):
    return np.concatenate([gtrack(_i) for _i in diseq], axis=0)


def compute_transcript_correlation(diseq, gtrack, idx_for_corr):
    assert len(idx_for_corr) == 2
    _y = get_diseq_data(diseq, gtrack)
    _y1 = _y[:, idx_for_corr[0]]
    _y2 = _y[:, idx_for_corr[1]]
    # select positions where both are not -1 (missing val indicator)
    idx = np.where((_y1 != -1) & (_y2 != -1))
    y1 = _y1[idx]
    y2 = _y2[idx]
    corr, pval = pearsonr(y1, y2)
    return corr, pval


with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# set of gene symbols passing min TPM
gene2tpm = parse_expression_data(dc.Client().get_path(config['cell_line_gene_expression']), config['cell_line_name'])


with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)


# list of tuples (data_set_name, type)
_names = [
    ('k562_vitro', 'dms'),
    ('k562_vivo_300', 'dms'),
    ('k562_vivo_400', 'dms'),
]

# to compute transcript level correlation
for_corr = ['k562_vivo_300', 'k562_vivo_400']
assert len(for_corr) == 2

df_data_info = pd.DataFrame(_names, columns=['data_name', 'data_type'])
df_data_info['track_index'] = range(len(_names))
idx_for_corr = [df_data_info[df_data_info['data_name'] == _name].iloc[0]['track_index'] for _name in for_corr]

# combine all diseqs
dfs_diseq = []
for data_name in df_data_info['data_name']:
    _, df_diseq = read_dataframe('data/intervals_{}.csv'.format(data_name))
    df_diseq = df_diseq[['transcript_id', 'disjoint_intervals']]
    df_diseq[data_name] = True
    print("{}: {} diseqs".format(data_name, len(df_diseq)))
    dfs_diseq.append(df_diseq)
# df_all = reduce(lambda x, y: pd.merge(x, y, on=['transcript_id', 'disjoint_intervals'], how='outer'), dfs_diseq)
_df_diseq = pd.concat(dfs_diseq).drop_duplicates(subset=['transcript_id'])
_dfs_id = [y[['transcript_id', x]] for x, y in zip(df_data_info['data_name'], dfs_diseq)]
df_all = reduce(lambda x, y: pd.merge(x, y, on='transcript_id', how='outer'), _dfs_id)
df_all = df_all.fillna(value={x: False for x in df_data_info['data_name']})
assert len(df_all) == len(_df_diseq)
df_all = pd.merge(df_all, _df_diseq[['transcript_id', 'disjoint_intervals']], on='transcript_id')

# transcript_id should be unique
print(len(df_all))
print(df_all['transcript_id'].nunique())
assert len(df_all) == df_all['transcript_id'].nunique()

print('All intervals: {}'.format(len(df_all)))

# load gtracks
reactivity_tracks = [GenomeTrack('data/reactivity_{}.gtrack'.format(data_name)) for data_name in
                     df_data_info['data_name']]
coverage_tracks = [GenomeTrack('data/coverage_{}.gtrack'.format(data_name)) for data_name in
                   df_data_info['data_name']]

# output tracks
track_reactivity = GenomeTrackBuilder('data/reactivity_combined.gtrack', dim=len(df_data_info),
                                      etype=config['gtrack_encoding_reactivity'])
track_reactivity.set_default_value(config['default_val_reactivity'])
track_coverage = GenomeTrackBuilder('data/coverage_combined.gtrack', dim=len(df_data_info),
                                    etype=config['gtrack_encoding_coverage'])
track_coverage.set_default_value(config['default_val_coverage'])

for _, row in df_all.iterrows():

    transcript_id = row['transcript_id']
    diseq = row['disjoint_intervals']

    for itv in diseq:
        # reactivity
        combine_array = []
        for i, t in enumerate(reactivity_tracks):
            _d = t(itv)
            assert _d.shape[1] == 1  # make sure it's 1D

            # normalize value to 0-1 for PARS dataset
            # except for 'missing values' (-1)
            data_type = df_data_info.iloc[i]['data_type']
            if data_type == 'pars' and not np.all(_d == config['missing_val_reactivity']):
                # clip at 2, then divide by 2
                assert np.min(_d[_d != config['missing_val_reactivity']]) >= 0
                _d = np.clip(_d, -1, 2)
                _d[_d != config['missing_val_reactivity']] = _d[_d != config['missing_val_reactivity']] / 2.0
            assert np.max(_d) <= 1

            combine_array.append(_d)
        combine_array = np.concatenate(combine_array, axis=1)

        try:
            assert np.max(combine_array) <= 1
            track_reactivity.set_data(itv, combine_array)
        except ValueError as e:
            print(str(e))
            continue

        # coverage
        combine_array = []
        for t in coverage_tracks:
            _d = t(itv)
            assert _d.shape[1] == 1  # make sure it's 1D
            combine_array.append(_d)
        combine_array = np.concatenate(combine_array, axis=1)
        try:
            track_coverage.set_data(itv, combine_array)
        except ValueError as e:
            print(str(e))
            continue

# output
track_reactivity.finalize()
track_coverage.finalize()

genome = Genome(config['genome_annotation'])

# transcript level correlation
loaded_track = GenomeTrack('data/reactivity_combined.gtrack')
df_all = add_columns(df_all, ['pearson_corr', 'pearson_pval'], ['disjoint_intervals'],
                     lambda x: compute_transcript_correlation(x, loaded_track, idx_for_corr))

# add gene expression
df_all = add_column(df_all, 'transcript', ['transcript_id'], lambda x: genome.transcripts[x])
df_all = add_column(df_all, 'tpm', ['transcript'],
                    lambda x: gene2tpm.get(x.gene.name, None))

df_data_info.to_csv('data/data_info.csv', index=False)

metadata = dataframe.Metadata(
    version="1",
    encoding={
        "disjoint_intervals": dataframe.Metadata.GENOMEKIT,
        "transcript": dataframe.Metadata.GENOMEKIT,
    })
write_dataframe(metadata, df_all, 'data/intervals_combined.csv')

