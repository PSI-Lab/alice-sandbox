import pandas as pd
from genome_kit import GenomeTrack, Interval


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

print(len(diseqs))
