import gzip
import csv
import numpy as np
import pandas as pd
import datacorral as dc
from genome_kit import Genome, Interval, GenomeTrackBuilder
from dgutils.interval import DisjointIntervalsSequence, IntervalData
from config import config


def parse_expression_data(expression_file, cell_line):
    # parse gene expression file downloaded from protein atlas
    # rna_celline.tsv
    df = pd.read_csv(expression_file, sep='\t')
    df = df[['Gene name', 'Sample', 'Value']].rename(columns={'Gene name': 'gene_name',
                                                              'Sample': 'cell_line',
                                                              'Value': 'tpm'})
    assert cell_line in df['cell_line'].unique()
    return df[df['cell_line'] == cell_line].drop(columns=['cell_line'])


def split_list_equal_sum(a, k):
    assert k > 1
    assert k < len(a)
    result = [[] for _ in range(k)]
    # sort descending
    a = sorted(a)[::-1]
    # go through items in a, add it to the sublist with smallest sum
    for x in a:
        current_sum = map(sum, result)
        idx_sublist = np.argmin(current_sum)
        result[idx_sublist].append(x)
    return result


# TODO chromosome size might not be the best proxy, maybe mRNA size?
# split chromosomes into K folds
chrom_sizes = config['chrom_sizes']
interval_fold_file_names = config['interval_folds']
fold_chrom_sizes = split_list_equal_sum(map(lambda x: x[1], chrom_sizes), len(interval_fold_file_names))
fold_chroms = map(lambda x: map(lambda z: next(y[0] for y in chrom_sizes if y[1] == z), x), fold_chrom_sizes)
print "chromosomes for each fold:", fold_chroms
print "chromosome sizes for each fold:", fold_chrom_sizes


genome = Genome(config['genome_annotation'])

reader = csv.DictReader(gzip.open(dc.Client().get_path(config['raw_data'])), delimiter='\t')
diseqs = dict()
default_val = -1

track_ss = GenomeTrackBuilder(config['gtrack'], 'f16')  # single value
track_ss.set_default_value(default_val)

# set of gene symbols passing min TPM
df_exp = parse_expression_data(dc.Client().get_path(config['cell_line_gene_expression']), 'Hep G2')
expressed_genes = set(df_exp[df_exp['tpm'] >= config['min_tpm']]['gene_name'].tolist())
print("Using {} (out of {}, TPM >= {}) highly expressed genes in HepG2".format(len(expressed_genes), len(df_exp),
                                                                               config['min_tpm']))

skipped_transcripts = set()

for row in reader:

    tx_id = row['tx_id']
    chrom = row['chr']
    strand = row['strand']

    if tx_id not in diseqs:
        try:
            transcript = genome.transcripts[tx_id]

            # deal with case where multiple transcripts having the same transcript ID
            if transcript.chromosome != chrom or transcript.strand != strand:
                transcript = next(
                    t for t in genome.transcripts if t.chromosome == chrom and t.strand == strand and t.id == tx_id)
        except (KeyError, StopIteration):
            if tx_id not in skipped_transcripts:
                print('Cannot get GK transcript %s' % tx_id)
                skipped_transcripts.add(tx_id)
            continue

        # check whether this gene is expressed in the cell line
        if transcript.gene.name not in expressed_genes:
            if tx_id not in skipped_transcripts:
                print("{} {} not well expressed in HepG2, skipping".format(transcript.gene.name, transcript.id))
                skipped_transcripts.add(tx_id)
            continue

        diseqs[tx_id] = DisjointIntervalsSequence(map(lambda x: x.interval, transcript.exons), genome)

    diseq = diseqs[tx_id]

    # validate
    # skip non training/validation chromosomes
    if chrom not in config['chrom_all']:
        continue

    assert diseq.intervals[0].chromosome == chrom, (row, chrom)
    assert diseq.intervals[0].strand == strand, (row, diseq.intervals)

    transcript_position = int(row['transcript_position'])
    genomic_position = int(row['genomic_position'])
    base = row['base']

    # skip transcript position that's outside of GK transcript length
    if transcript_position < 1 or transcript_position > len(diseq.interval):
        print('Skip transcript %s position %d' % (tx_id, transcript_position))
        continue

    # first try using diseq interval
    diseq_itv = diseq.interval.end5.shift(transcript_position - 1).expand(0, 1)
    if diseq.dna(diseq_itv) == base:
        genom_itv = diseq.lower_interval(diseq_itv)[0]
        if genom_itv.as_dna1()[0] != genomic_position:
            genom_itv = None
            pass

    # if the above is unsuccessful, fall back to genome position
    # need to do this since some transcript definition in the raw data is inconsistent with GK
    # e.g. NM_003742 (GK is shorted of 1bp on 5'end)
    genom_itv = Interval.from_dna1(chrom, strand, genomic_position, genomic_position, genome)
    assert genome.dna(genom_itv) == base

    # set gtrack value, make sure use default_val for missing value
    if row['reactivity'] != 'NA':
        val = float(row['reactivity'])
    else:
        val = default_val

    # report if setting overlapping data
    try:
        track_ss.set_data(genom_itv, np.asarray([val], dtype=np.float16))
    except ValueError as e:
        print(str(e))
        continue


track_ss.finalize()

# output disjoint intervals for each of the K folds
for chroms, file_name in zip(fold_chroms, interval_fold_file_names):
    fold_intervals = [x.intervals for x in diseqs.values() if x.intervals[0].chromosome in chroms]
    with open(file_name, 'w') as f:
        for itv in fold_intervals:
            f.write(str(itv) + '\n')
