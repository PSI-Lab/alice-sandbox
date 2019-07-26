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
    # make sure there is no adjacent contiguous intervals
    assert not any([x.end == y.start for x, y in zip(itvs[:-1], itvs[1:])])
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


