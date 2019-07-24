import numpy as np
import pandas as pd
import deepgenomics.pandas.v1 as dataframe
from dgutils.pandas import add_column, write_dataframe
from utils import Interval, FastaGenome, WigTrack


def get_transcript_ditv(chrom, strand, exon_starts, exon_ends):
    exon_starts = [int(x) for x in exon_starts.rstrip(',').split(',')]
    exon_ends = [int(x) for x in exon_ends.rstrip(',').split(',')]
    return [Interval(chrom, strand, s, e) for s, e in zip(exon_starts, exon_ends)]


genome = FastaGenome('raw_data/fasta/')

wig_track = WigTrack(genome.chrom_sizes, 'raw_data/dms/GSE45803_Feb13_VivoAllextra_1_15_PLUS.wig.gz',
                     'raw_data/dms/GSE45803_Feb13_VivoAllextra_1_15_Minus.wig.gz', np.nan)

df_anno = pd.read_csv('raw_data/annotation/ncbiRefSeqCurated.txt.gz', sep='\t', header=None,
                      names=['bin', 'name', 'chrom', 'strand', 'tx_start', 'tx_end',
                             'cds_start', 'cds_end', 'exon_count', 'exon_starts', 'exon_ends',
                             'score', 'name_2', 'cds_start_stat', 'cds_end_stat', 'exon_frames'])

# select transcripts with complete cds status
df_anno = df_anno[(df_anno['cds_start_stat'] == 'cmpl') & (df_anno['cds_end_stat'] == 'cmpl')]
# get ditv
df_anno = add_column(df_anno, 'ditv', ['chrom', 'strand', 'exon_starts', 'exon_ends'], get_transcript_ditv)
# add sequence
df_anno = add_column(df_anno, 'sequence', ['ditv'], lambda x: genome.dna(x))

# add data


def _norm(x, w=50, check_len=True):
    # Each reactivity value above the 95th percentile is set to the 95th percentile
    # and each reactivity value below the 5th percentile is set to the 5th percentile,
    # then the reactivity at each position of the transcript is divided by the value of the 95th percentile
    if check_len:
        assert np.sum(~np.isnan(x)) == w, (x, np.sum(~np.isnan(x)))
    q95 = np.nanquantile(x, 0.95)
    q05 = np.quantile(x, 0.05)
    x = np.clip(x, q05, q95)
    x = x/q95
    assert np.nanmin(x) >= 0, np.nanmin(x)
    assert np.nanmax(x) <= 1, np.nanmin(x)
    return x


def _add_data(itv, w=50):
    x = wig_track[itv]
    # normalize to 0 - 1
    # window normalization?
    # use window size 50 (50 non missing values)
    idx = np.where(~np.isnan(x))[0]  # index of non missing values

    if len(idx) > 0:
        y = []
        ks = (len(idx) - 1)//w + 1
        print(len(idx), ks)
        for k in range(ks):
            # first batch, use first index in x
            if k == 0:
                start = 0
            else:
                start = idx[k * w]
            # last batch, use the last index in x
            if k == ks -1:
                end = len(x)
                check_len = False
            else:
                end = idx[(k + 1) * w]
                check_len = True

            # print(start, end, end - start)
            yp = _norm(x[start:end], w, check_len)

            # if k == 0:
            #     start = 0
            #     end = idx[(k + 1) * w]
            #     print(start, end, end - start)
            #     yp = _norm(x[start:end], w)
            # # last batch, use the last index in x
            # elif k == ks -1:
            #     start = idx[k * w]
            #     end = len(x)
            #     print(start, end, end-start)
            #     yp = _norm(x[start:end], w, check_len=False)
            # else:
            #     start = idx[k * w]
            #     end = idx[(k + 1) * w]
            #     print(start, end, end - start)
            #     yp = _norm(x[start:end], w)
            # end = (k + 1) * w
            # if end > len(idx):
            #     end = idx[-1]

            y.append(yp)
        y = np.concatenate(y)
        assert len(y) == len(x), (len(y), len(x))
    else:
        y = x
    # replace nan with -1
    y[np.where(np.isnan(y))] = -1
    return y.tolist()


df_anno = add_column(df_anno, 'data', ['ditv'], lambda x: _add_data(x, w=50))

# output
df_anno = df_anno[['name', 'chrom', 'strand', 'tx_start', 'tx_end', 'name_2', 'sequence', 'data']].rename(
    columns={'name': 'transcript_id', 'name_2': 'gene_name'})

metadata = dataframe.Metadata()
metadata.version = "1"
metadata.encoding['data'] = dataframe.Metadata.LIST
write_dataframe(metadata, df_anno, 'data/yeast_test.csv')
