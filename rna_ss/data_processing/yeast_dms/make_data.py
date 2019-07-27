import numpy as np
import pandas as pd
import deepgenomics.pandas.v1 as dataframe
from dgutils.pandas import add_column, write_dataframe, add_columns
from utils import Interval, FastaGenome, WigTrack, _sort_ditvs


def get_transcript_ditv(chrom, strand, exon_starts, exon_ends):
    exon_starts = [int(x) for x in exon_starts.rstrip(',').split(',')]
    exon_ends = [int(x) for x in exon_ends.rstrip(',').split(',')]
    return [Interval(chrom, strand, s, e) for s, e in zip(exon_starts, exon_ends)]


genome = FastaGenome('raw_data/fasta/')

wig_track = WigTrack(genome.chrom_sizes, 'raw_data/dms/GSE45803_Feb13_VivoAllextra_1_15_PLUS.wig.gz',
                     'raw_data/dms/GSE45803_Feb13_VivoAllextra_1_15_Minus.wig.gz', np.nan)

# df_anno = pd.read_csv('raw_data/annotation/ncbiRefSeqCurated.txt.gz', sep='\t', header=None,
#                       names=['bin', 'name', 'chrom', 'strand', 'tx_start', 'tx_end',
#                              'cds_start', 'cds_end', 'exon_count', 'exon_starts', 'exon_ends',
#                              'score', 'name_2', 'cds_start_stat', 'cds_end_stat', 'exon_frames'])
df_anno = pd.read_csv('raw_data/annotation/xenoRefGene.txt.gz', sep='\t', header=None,
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


# def _align_data(itv, seq, n=10):
#     # need to do this since I was using sacCer3 fasta and annotation but the wig files were on sacCer2
#     # no longer needed if switch to sacCer2 fasta and annotation
#     # print('before ', itv)
#     if type(itv) == list:
#         itv = _sort_ditvs(itv)
#         l = sum([len(x) for x in itv])
#         itv[0] = itv[0].expand(n, 0)
#         itv[-1] = itv[-1].expand(0, n)
#         new_l = sum([len(x) for x in itv])
#         assert new_l == l + 2*n, (new_l, l)
#     else:
#         l = len(itv)
#         itv = itv.expand(n, n)
#         new_l = len(itv)
#         assert new_l == l + 2 * n, (new_l, l)
#     # print('after ', itv)
#     _data = wig_track[itv]
#     nac = []
#     for i in range(2 * n):
#         data = _data[i:i+l]
#         idx = np.where(~np.isnan(data))[0]  # index of non missing values
#         n_ac_covered = len([i for i in idx if seq[i] in ['A', 'C', 'a', 'c']])
#         nac.append(n_ac_covered)
#     i_selected = nac.index(max(nac))  # argmax
#     relative_shift = i_selected - n
#     best_coverage = float(nac[i_selected])/(seq.count('A') + seq.count('C') + seq.count('a') + seq.count('c'))
#     print("relative shift {}, A/C coverage {}".format(relative_shift, best_coverage))
#     return _data[i_selected:i_selected+l], best_coverage, relative_shift


def _align_data(itv, seq, gene_name, transcript_id):
    # just reporting
    data = wig_track[itv]
    idx = np.where(~np.isnan(data))[0]  # index of non missing values
    n_ac_covered = len([i for i in idx if seq[i] in ['A', 'C', 'a', 'c']])
    n_gt_covered = len([i for i in idx if seq[i] in ['G', 'T', 'g', 't']])
    ac_coverage = float(n_ac_covered)/(seq.count('A') + seq.count('C') + seq.count('a') + seq.count('c'))
    gt_coverage = float(n_gt_covered)/(seq.count('G') + seq.count('T') + seq.count('g') + seq.count('t'))
    print("{} {} A/C coverage {} G/T coverage {}".format(gene_name, transcript_id, ac_coverage, gt_coverage))
    return data, ac_coverage


def _add_data(itv, seq, gene_name, transcript_id, w=50):
    # raw data seems to be random shifted
    # re-align by looking for the offset that maximize coverage on A/C bases
    # x, ac_coverage, relative_shift = _align_data(itv, seq)
    x, ac_coverage = _align_data(itv, seq, gene_name, transcript_id)

    # normalize to 0 - 1
    # window normalization?
    # use window size 50 (50 non missing values)
    idx = np.where(~np.isnan(x))[0]  # index of non missing values

    if len(idx) > 0:
        y = []
        ks = (len(idx) - 1)//w + 1
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

            yp = _norm(x[start:end], w, check_len)

            y.append(yp)
        y = np.concatenate(y)
        assert len(y) == len(x), (len(y), len(x))
    else:
        y = x
    # replace nan with -1
    y[np.where(np.isnan(y))] = -1
    # return y.tolist(), ac_coverage, relative_shift
    return y.tolist(), ac_coverage


# df_anno = add_columns(df_anno, ['data', 'ac_coverage', 'relative_shift'], ['ditv', 'sequence'], lambda x, y: _add_data(x, y, w=50))
df_anno = add_columns(df_anno, ['data', 'ac_coverage'], ['ditv', 'sequence', 'name_2', 'name'],
                      lambda x, y, n1, n2: _add_data(x, y, n1, n2, w=50))

# output
# df_anno = df_anno[['name', 'chrom', 'strand', 'tx_start', 'tx_end',
#                    'name_2', 'sequence', 'data', 'ac_coverage', 'relative_shift']].rename(
#     columns={'name': 'transcript_id', 'name_2': 'gene_name'})
df_anno = df_anno[['name', 'chrom', 'strand', 'tx_start', 'tx_end',
                   'name_2', 'sequence', 'data', 'ac_coverage']].rename(
    columns={'name': 'transcript_id', 'name_2': 'gene_name'})

metadata = dataframe.Metadata()
metadata.version = "1"
metadata.encoding['data'] = dataframe.Metadata.LIST
write_dataframe(metadata, df_anno, 'data/yeast_test.csv')
