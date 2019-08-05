"""
Process rRNA data from yeast DMS paper
note that we still need to email the author on:
- why sequence length shorter than that in the annotation DB
- why do G/T bases also have value
"""
import pandas as pd
from Bio import SeqIO
import deepgenomics.pandas.v1 as dataframe
from dgutils.pandas import read_dataframe, write_dataframe, add_column, add_columns


def process_data(wig_file, fasta_file, gene_name):
    # it's not really wig file, just 2 columns (from paper SI)
    df = pd.read_csv(wig_file, skiprows=2, sep='\s+',
                     header=None, names=['position', 'val'])
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = record.seq

    df['base'] = list(sequences[gene_name][:len(df)])

    print(gene_name)
    for b in list('ACGT'):
        print(b)
        print(df[df['base'] == b]['val'].median())
    print('')

    return ''.join(df['base'].tolist()), df['val'].tolist()


name1 = 'RDN18-1'
seq1, val1 = process_data('raw_data/gse45803/GSE45803_ACviv1mset1mag_sc-rrna.align18sRNAnoMiss.wig.gz',
                          'raw_data/gse45803/S288C_RDN18-1_RDN18-1_genomic.fsa', name1)
name2 = 'RDN25-1'
seq2, val2 = process_data('raw_data/gse45803/GSE45803_r25sDMS_GCvitro2_LateJune12res_sc-rrna.align.wigFiveCount.wig.gz',
                          'raw_data/gse45803/EC1118_RDN25-1_RDN25-1_genomic.fsa', name2)

df = pd.DataFrame([
    {
        'gene_name': name1,
        'sequence': seq1,
        'data': val1,
    },
    {
        'gene_name': name2,
        'sequence': seq2,
        'data': val2,
    },
])

# output
metadata = dataframe.Metadata()
metadata.version = "1"
metadata.encoding['data'] = dataframe.Metadata.LIST
write_dataframe(metadata, df, 'data/yeast_dms_ribosome_rna.csv')
