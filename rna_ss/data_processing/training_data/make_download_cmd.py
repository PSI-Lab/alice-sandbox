import yaml
import os
import logging
import pandas as pd
from dgutils.pandas import add_column


logging.getLogger().setLevel(logging.INFO)

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

OUT = open(config['download_cmd'], 'w')

for study_name in config['studies']:
    logging.info(study_name)
    df = pd.read_csv(config['studies'][study_name]['sra_run_table'], sep='\t')
    runs = config['studies'][study_name]['runs']
    df = df[df['Run'].isin(runs)]

    df = add_column(df, 'f6', ['Run'], lambda x: x[:6])
    df = add_column(df, 'l1', ['Run'], lambda x: x[-1])
    df = add_column(df, 'url_single', ['f6', 'l1', 'Run'],
                    lambda f6, l1, run: 'era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/{}/00{}/{}/{}.fastq.gz'.format(f6, l1,
                                                                                                                run,
                                                                                                                run))
    df = add_column(df, 'url_pair_1', ['f6', 'Run'],
                    lambda f6, run: 'era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/{}/{}/{}_1.fastq.gz'.format(f6, run, run))
    df = add_column(df, 'url_pair_2',  ['f6', 'Run'],
                    lambda f6, run: 'era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/{}/{}/{}_2.fastq.gz'.format(f6, run, run))

    out_dir = 'data/{}'.format(study_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for _, row in df.iterrows():
        if row['LibraryLayout'] == 'SINGLE':
            OUT.write('{} -QT -l 300m -P33001 -i {} {} {}\n'.format(config['ascp_script'], config['ascp_key'],
                                                                  row['url_single'], out_dir))
        elif row['LibraryLayout'] == 'PAIRED':
            OUT.write('{} -QT -l 300m -P33001 -i {} {} {}\n'.format(config['ascp_script'], config['ascp_key'],
                                                                  row['url_pair_1'], out_dir))
            OUT.write('{} -QT -l 300m -P33001 -i {} {} {}\n'.format(config['ascp_script'], config['ascp_key'],
                                                                  row['url_pair_2'], out_dir))
        else:
            raise ValueError

OUT.close()
