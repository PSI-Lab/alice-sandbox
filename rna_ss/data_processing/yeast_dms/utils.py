import os
import gzip
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from bx.wiggle import IntervalReader


def _sort_ditvs(itvs):
    assert all([x.chromosome == itvs[0].chromosome for x in itvs])
    assert all([x.strand == itvs[0].strand for x in itvs])
    assert len(itvs) > 0
    assert all([len(x) > 0 for x in itvs])
    # sort 5' -> 3'
    itvs = sorted(itvs, key=lambda x: x.as_rna1())
    # check that they do not overlap
    assert not any([x.overlaps(y) for x, y in zip(itvs[:-1], itvs[1:])])
    # # make sure there is no adjacent contiguous intervals
    # assert not any([x.end == y.start for x, y in zip(itvs[:-1], itvs[1:])]), itvs
    # print a warning if there are adjacent contiguous intervals
    if any([x.end == y.start for x, y in zip(itvs[:-1], itvs[1:])]):
        print("Warning: adjacent contiguous intervals in {}".format(itvs))
    return itvs


class Interval(object):

    def __init__(self, chromosome, strand, start, end):
        assert strand in ['+', '-']
        self.chromosome = chromosome
        self.strand = strand
        self.start = start
        self.end = end
        self._data_len = end - start

    def __len__(self):
        return self._data_len

    def as_rna1(self):
        if self.start == self.end:
            raise ValueError("Empty intervals cannot be represented using RNA1 convention.")

        if self.strand == '+':  # Forward strand
            return self.start + 1, self.end
        else:  # Reverse strand
            return -self.end, -(self.start + 1)

    def expand(self, up, dn):
        if self.strand == '+':
            return Interval(self.chromosome, self.strand, self.start - up, self.end + dn)
        else:
            return Interval(self.chromosome, self.strand, self.start - dn, self.end + up)

    def as_opposite_strand(self):
        strand = '-' if self.strand == '+' else '+'
        return Interval(self.chromosome, strand, self.start, self.end)

    def overlaps(self, other):
        if self.chromosome != other.chromosome:
            return False
        if self.strand != other.strand:
            return False
        if max(self.start, other.start) < min(self.end, other.end):
            return True
        else:
            return False

    def __repr__(self):
        return 'Interval("{}", "{}", {}, {})'.format(self.chromosome, self.strand, self.start, self.end)


class FastaGenome(object):

    def __init__(self, data_path):
        self.genome = self._load_fasta_files(data_path)
        self._chrom_sizes = None

    def dna(self, itv):
        if type(itv) == list:
            return self._dna_ditv(itv)
        else:
            return self._dna_itv(itv)

    def _dna_itv(self, itv):
        assert itv.chromosome in self.genome.keys()
        assert itv.strand in ['+', '-']
        assert itv.start >= 0
        assert itv.end < len(self.genome[itv.chromosome])
        seq = self.genome[itv.chromosome][itv.start:itv.end]
        if itv.strand == '-':
            seq = self._rc(seq)
        return str(seq)

    def _dna_ditv(self, itvs):
        itvs = _sort_ditvs(itvs)
        return ''.join([self._dna_itv(x) for x in itvs])

    @staticmethod
    def _rc(seq):
        return str(seq.reverse_complement())

    def _load_fasta_files(self, data_path):
        genome_dict = dict()
        for filename in os.listdir(data_path):
            if filename.endswith(".fa.gz"):
                with gzip.open(os.path.join(data_path, filename), 'r') as f:
                    for x in SeqIO.parse(f, "fasta"):
                        assert x.id not in genome_dict
                        genome_dict[x.id] = x.seq
        return genome_dict

    @property
    def chrom_sizes(self):
        if self._chrom_sizes is None:
            self._chrom_sizes = {c: len(s) for c, s in self.genome.iteritems()}
        return self._chrom_sizes


class WigTrack(object):

    def __init__(self, chrom_sizes, data_plus, data_minus, missing_val=np.nan):
        self.data = self._init_array(chrom_sizes, missing_val)
        self.missing_val = missing_val
        self._add_data(data_plus, strand='+')
        self._add_data(data_minus, strand='-')

    def __getitem__(self, item):
        # lookup interval or ditv
        if type(item) == list:
            return self._get_ditv(item)
        else:
            return self._get_itv(item)

    def _get_itv(self, itv):
        val = self.data[itv.chromosome][itv.strand][itv.start:itv.end]
        if itv.strand == '-':
            val = val[::-1]
        return val

    def _get_ditv(self, itvs):
        itvs = _sort_ditvs(itvs)
        return np.concatenate([self._get_itv(x) for x in itvs])

    def _init_array(self, chrom_sizes, missing_val):
        data_dict = {}
        for chrom, arr_len in chrom_sizes.iteritems():
            arr_plus = np.empty(arr_len)
            arr_plus.fill(missing_val)
            arr_mnus = np.empty(arr_len)
            arr_mnus.fill(missing_val)
            data_dict[chrom] = {'+': arr_plus, '-': arr_mnus}
        return data_dict

    def _add_data(self, data_file, strand, allow_overwrite=False):
        with gzip.open(data_file) as f:
            for chrom, start, end, _, val in IntervalReader(f):  # ignore strand returned by bx (always +?)
                if not allow_overwrite:
                    current_val = self.data[chrom][strand][start:end]
                    if np.isnan(self.missing_val):  # FIXME this doesn't deal with inf
                        assert np.all(np.isnan(current_val)), current_val
                    else:
                        assert np.all(current_val == self.missing_val), current_val
                self.data[chrom][strand][start:end] = val


def _norm(x, w=50, check_len=True):
    # Each reactivity value above the 95th percentile is set to the 95th percentile
    # and each reactivity value below the 5th percentile is set to the 5th percentile,
    # then the reactivity at each position of the transcript is divided by the value of the 95th percentile
    if check_len:
        assert np.sum(~np.isnan(x)) == w, (x, np.sum(~np.isnan(x)))
    q95 = np.nanquantile(x, 0.95)
    q05 = np.nanquantile(x, 0.05)
    x = np.clip(x, q05, q95)
    # some times values are all 0's, do not divide
    if q95 != 0:
        x = x/q95
    assert np.nanmin(x) >= 0, np.nanmin(x)
    assert np.nanmax(x) <= 1, np.nanmin(x)
    return x


def _align_data(wig_track, itv, seq, gene_name, transcript_id, remove_non_ac_vals=False):
    # TODO set A/C bases with missing values to 0!!!!
    data = wig_track[itv]
    # all missing values
    if np.all(np.isnan(data)):
        return None

    idx_ac = [i for i, base in enumerate(seq) if base in ['A', 'C', 'a', 'c']]
    idx_nan = np.where(np.isnan(data))[0]  # index of missing values
    _idx = list(set(idx_ac).intersection(set(idx_nan)))  # index of A/C bases with missing value
    print("{} {} setting {} (total {}) A/C bases with missing value to 0".format(gene_name, transcript_id, len(_idx),
                                                                                 len(idx_ac)))
    data[_idx] = 0

    # idx_non_nan = np.where(~np.isnan(data))[0]  # index of non missing values
    # n_ac_covered = len([i for i in idx if seq[i] in ['A', 'C', 'a', 'c']])
    # n_gt_covered = len([i for i in idx if seq[i] in ['G', 'T', 'g', 't']])
    # ac_total = seq.count('A') + seq.count('C') + seq.count('a') + seq.count('c')
    # if ac_total > 0:
    #     ac_coverage = float(n_ac_covered)/ac_total
    # else:
    #     ac_coverage = 0.0
    # gt_total = seq.count('G') + seq.count('T') + seq.count('g') + seq.count('t')
    # if gt_total > 0:
    #     gt_coverage = float(n_gt_covered)/gt_total
    # else:
    #     gt_coverage = 0.0
    if remove_non_ac_vals:
        idx_non_ac = [i for i in range(len(seq)) if seq[i] not in ['A', 'C', 'a', 'c']]
        data[idx_non_ac] = np.nan
        idx = np.where(~np.isnan(data))[0]  # index of non missing values
        assert len([i for i in idx if seq[i] in ['G', 'T', 'g', 't']]) == 0
    # print("{} {} A/C coverage {} G/T coverage {}".format(gene_name, transcript_id, ac_coverage, gt_coverage))
    # return data, ac_coverage
    return data


def _add_data(wig_track, itv, seq, gene_name, transcript_id, w=100):
    if len(seq) == 0:
        return [], 0.0

    # just reporting
    x = _align_data(wig_track, itv, seq, gene_name, transcript_id, remove_non_ac_vals=True)
    if x is None:
        return None

    # normalize to 0 - 1
    # window normalization?
    # use window size W W non missing values)
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

            # print start, end, x[start:end]
            yp = _norm(x[start:end], w, check_len)

            y.append(yp)
        y = np.concatenate(y)
        assert len(y) == len(x), (len(y), len(x))
    else:
        y = x
    # replace nan with -1
    y[np.where(np.isnan(y))] = -1
    # return y.tolist(), ac_coverage, relative_shift
    return y.tolist()

