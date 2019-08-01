import numpy as np
import pandas as pd
import deepgenomics.pandas.v1 as dataframe
from dgutils.pandas import add_column, write_dataframe, add_columns
from utils import Interval, FastaGenome, ModSeqBedGraph, _sort_ditvs


def get_transcript_itv(chrom, strand, tx_start, tx_end):
    return Interval(chrom, strand, tx_start, tx_end)


def _align_data(itv, seq, gene_name, transcript_id, dtrack):
    # just reporting
    data = dtrack[itv]
    idx = np.where(~np.isnan(data))[0]  # index of non missing values
    mean_ac_val = np.mean([data[i] for i in idx if seq[i] in ['A', 'C', 'a', 'c']])
    mean_gt_val = np.mean([data[i] for i in idx if seq[i] in ['G', 'T', 'g', 't']])
    print("{} {} A/C mean {} G/T mean {}".format(gene_name, transcript_id, mean_ac_val, mean_gt_val))
    return data


def _add_data(itv, seq, gene_name, transcript_id, dtrack):
    x = _align_data(itv, seq, gene_name, transcript_id, dtrack)
    return x.tolist()


# Note we need SacCer3 fasta and annotation!!!
genome = FastaGenome('raw_data/fasta/')
data_track = ModSeqBedGraph(genome.chrom_sizes,
                            'raw_data/mod_seq/WT-100mM_R1_AdaptStop_sorted_Pos.bedGraph.gz',
                            'raw_data/mod_seq/WT-100mM_R1_AdaptStop_sorted_Neg.bedGraph.gz',
                            np.nan)
df_anno = pd.read_csv('raw_data/annotation/ncbiRefSeqCurated.txt.gz', sep='\t', header=None,
                      names=['bin', 'name', 'chrom', 'strand', 'tx_start', 'tx_end',
                             'cds_start', 'cds_end', 'exon_count', 'exon_starts', 'exon_ends',
                             'score', 'name_2', 'cds_start_stat', 'cds_end_stat', 'exon_frames'])


# only processing specific transcript for now
transcript_ids = ['NR_132209.1',  # 25S
                  'NR_132213.1',  # 18S
                  'NR_132211.1',  # 5.8S

                  ]

df_anno = df_anno[df_anno['name'].isin(transcript_ids)]


df_anno = add_column(df_anno, 'itv', ['chrom', 'strand', 'tx_start', 'tx_end'], get_transcript_itv)

df_anno = add_column(df_anno, 'sequence', ['itv'], lambda x: genome.dna(x))

df_anno = add_column(df_anno, 'data', ['itv', 'sequence', 'name_2', 'name'],
                     lambda x, y, n1, n2: _add_data(x, y, n1, n2, data_track))

df_anno = df_anno[['name', 'chrom', 'strand', 'tx_start', 'tx_end',
                   'name_2', 'sequence', 'data']].rename(
    columns={'name': 'transcript_id', 'name_2': 'gene_name'})


metadata = dataframe.Metadata()
metadata.version = "1"
metadata.encoding['data'] = dataframe.Metadata.LIST
write_dataframe(metadata, df_anno, 'data/yeast_modseq_test.csv')
