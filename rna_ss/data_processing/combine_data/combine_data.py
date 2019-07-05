import yaml
import pandas as pd
import numpy as np
from genome_kit import GenomeTrack, Interval, GenomeTrackBuilder


with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)


# list of tuples (data_set_name, type)
_names = [
    # wan 2014
    ('human_lymphoblastoid_family_child', 'pars'),
    ('human_lymphoblastoid_family_father', 'pars'),
    ('human_lymphoblastoid_family_mother', 'pars'),
    # solomon 2017
    ('hepg2_rep1', 'pars'),
    ('hepg2_rep2', 'pars'),
    # zubradt 2016
    ('hek293', 'dms'),
    # rouskin 2014
    ('fibroblast_vitro', 'dms'),
    ('fibroblast_vivo', 'dms'),
    ('k562_vitro', 'dms'),
    ('k562_vivo_300', 'dms'),
    ('k562_vivo_400', 'dms'),
]

df_data_info = pd.DataFrame(_names, columns=['data_name', 'data_type'])
df_data_info['track_index'] = range(len(_names))


def _load_interval(file):
    transcript_itvs = []

    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip()
            transcript_itvs.append(tuple(eval(line)))
    return set(transcript_itvs)


# combine all diseqs
diseqs = []
for data_name in df_data_info['data_name']:
    itvs = _load_interval('data/intervals_{}.txt'.format(data_name))
    print(data_name, len(itvs))
    diseqs.append(itvs)
diseqs = set.union(*diseqs)

print('All intervals: {}'.format(len(diseqs)))

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

for diseq in diseqs:
    # print(diseq)
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
df_data_info.to_csv('data/data_info.csv', index=False)
with open('data/intervals_combined.txt', 'w') as f:
    for d in diseqs:
        f.write(str(list(d)) + '\n')
