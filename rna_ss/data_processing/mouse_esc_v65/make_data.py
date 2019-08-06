import re
import numpy as np
import pandas as pd
import genome_kit as gk
import deepgenomics.pandas.v1 as dataframe
from dgutils.interval import DisjointIntervalsSequence
from dgutils.pandas import write_dataframe


genome = gk.Genome('gencode.vM19')
wig_file = 'raw_data/GSE60034_v65polyA%2BicSHAPE_NAI-N3_vivo.wig'

gtrack = gk.GenomeTrackBuilder('data/test.gtrack', 'f16')
gtrack.set_default_value(-1.0)

# gtrack.set_data_from_wig(genome, wig_file)
gtrack.set_data_from_bedgraph(genome, wig_file)
gtrack.finalize()


# load data
gtrack = gk.GenomeTrack('data/test.gtrack')

# # test + strand
# itv = gk.Interval('chr5', '+', 142904500, 142904510, 'm38')
# print gtrack(itv)
# print gtrack(itv).shape
# # test - strand - works! GK revert the track, so data is 5'->3'
# itv = gk.Interval('chr5', '-', 142904500, 142904510, 'm38')
# print gtrack(itv)
# print gtrack(itv).shape


def _appris(genome, transcript):
    x = genome.appris_principality(transcript)
    if x is None:
        x = re.sub('[^0-9]', '', transcript.id)
    return x


def get_data_ditv(ditv, gtrack):
    assert type(ditv) == list
    # make sure it's sorted
    if len(ditv) > 1:
        assert all([x.upstream_of(y) for x, y in zip(ditv[:-1], ditv[1:])])
    data = []
    for itv in ditv:
        _d = gtrack(itv)
        assert len(_d.shape) == 2
        assert _d.shape[1] == 1
        data.append(_d[:, 0])
    return np.concatenate(data)


data = []
# go through all genes, pick one transcript per gene (TODO double check how the authors picked their transcripts for alignment)
for gene in genome.genes:
    transcripts = sorted(gene.transcripts, key=lambda x: _appris(genome, x))
    transcript = transcripts[0]
    diseq = DisjointIntervalsSequence(map(lambda x: x.interval, transcript.exons), genome)
    vals = get_data_ditv(diseq.intervals, gtrack)

    # skip all missing vals
    if np.all(vals == -1):
        continue

    # for statistics, replace missing val with nan
    _tmp = vals.copy()
    _tmp[_tmp == -1] = np.nan
    print("{} min {} max {}".format(gene, np.nanmin(_tmp), np.nanmax(_tmp)))

    data.append({
        'gene_name': gene.name,
        'transcript_id': transcript.id,
        'transcript': transcript,
        'disjoint_interval': diseq.intervals,
        'sequence': diseq.dna(diseq.interval),
        'data': vals.tolist(),
    })


df = pd.DataFrame(data)
metadata = dataframe.Metadata()
metadata.version = "1"
metadata.encoding['transcript'] = dataframe.Metadata.GENOMEKIT
metadata.encoding['disjoint_interval'] = dataframe.Metadata.LIST_OF_GENOMEKIT
metadata.encoding['data'] = dataframe.Metadata.LIST
write_dataframe(metadata, df, 'data/mouse_esc_v65_vivo.csv')

#  SHAPE data, not just A/C! =)
