"""
Process transcripts matching those imn ModSeq paper
"""
import numpy as np
import pandas as pd
import deepgenomics.pandas.v1 as dataframe
from dgutils.pandas import add_column, write_dataframe, add_columns
from utils import Interval, FastaGenome, WigTrack, _sort_ditvs, _add_data


# only processing specific transcript for now
transcript_ids = ['NR_132209.1',  # 25S
                  'NR_132213.1',  # 18S
                  'NR_132211.1',  # 5.8S

                'NR_132214.1',  # ETS1
                  'NR_132212.1',  # ITS1
                  'NR_132219.1',  # ITS2
                  'NR_132251.1',  # NME1
                  'NR_132215.1',  # RDN5
                  'NR_132171.1',  # SCR1
                  'NR_132179.1',  # snR10
                  'NR_132246.1',  # snR11
                  'NR_132193.1',  # snR128
                  'NR_132262.1',  # snR17a
                  'NR_132194.1',  # snR190
                  'NR_132198.1',  # snR3
                  'NR_132204.1',  # snR30
                  'NR_132195.1',  # snR37
                  'NR_132249.1',  # snR40
                  'NR_132203.1',  # snR42
                  'NR_132170.1',  # snR52
                  'NR_132242.1',  # snR73
                  'NR_132163.1',  # snR80
                  ]

# above transcript ID was from sacCer3, strip off the version number to match sacCer2 refseq file
transcript_ids = [x.split('.')[0] for x in transcript_ids]

genome = FastaGenome('raw_data/fasta/')
wig_track = WigTrack(genome.chrom_sizes, 'raw_data/dms/GSE45803_Feb13_VivoAllextra_1_15_PLUS.wig.gz',
                     'raw_data/dms/GSE45803_Feb13_VivoAllextra_1_15_Minus.wig.gz', np.nan)
df_anno = pd.read_csv('raw_data/annotation/xenoRefGene.txt.gz', sep='\t', header=None,
                      names=['bin', 'name', 'chrom', 'strand', 'tx_start', 'tx_end',
                             'cds_start', 'cds_end', 'exon_count', 'exon_starts', 'exon_ends',
                             'score', 'name_2', 'cds_start_stat', 'cds_end_stat', 'exon_frames'])

df_anno = df_anno[df_anno['name'].isin(transcript_ids)]
print("Found {} out of {} transcripts".format(len(df_anno), len(transcript_ids)))
# drop duplicates # TODO better solution?
df_anno = df_anno.drop_duplicates(subset=['name'])


# get itv
df_anno = add_column(df_anno, 'itv', ['chrom', 'strand', 'tx_start', 'tx_end'],
                     lambda chrom, strand, s, e: Interval(chrom, strand, s, e))
# add sequence
df_anno = add_column(df_anno, 'sequence', ['itv'], lambda x: genome.dna(x))
# add data
df_anno = add_columns(df_anno, ['data', 'ac_coverage'], ['itv', 'sequence', 'name_2', 'name'],
                      lambda x, y, n1, n2: _add_data(wig_track, x, y, n1, n2, w=100))

df_anno = df_anno[['name', 'chrom', 'strand', 'tx_start', 'tx_end',
                   'name_2', 'sequence', 'data', 'ac_coverage']].rename(
    columns={'name': 'transcript_id', 'name_2': 'gene_name'})

metadata = dataframe.Metadata()
metadata.version = "1"
metadata.encoding['data'] = dataframe.Metadata.LIST
write_dataframe(metadata, df_anno, 'data/yeast_dms_match_mod_seq_transcripts.csv')
