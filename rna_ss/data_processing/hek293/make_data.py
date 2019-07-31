import gzip
import csv
import yaml
import argparse
import numpy as np
import pandas as pd
import deepgenomics.pandas.v1 as dataframe
from genome_kit import Genome, Interval, GenomeTrackBuilder
from dgutils.pandas import add_column, add_columns, write_dataframe
from dgutils.interval import DisjointIntervalsSequence, IntervalData


def main(input_data, output_data, data_type, config):
    genome = Genome(config['genome_annotation'])
    reader = csv.DictReader(gzip.open(input_data), delimiter='\t')
    transcripts = dict()
    diseqs = dict()
    vals = dict()

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

            # initialize transcript
            transcripts[tx_id] = transcript
            # initialize diseq
            diseqs[tx_id] = DisjointIntervalsSequence(map(lambda x: x.interval, transcript.exons), genome)
            # initialize the val array
            vals[tx_id] = config['missing_val_reactivity'] * np.ones(len(diseqs[tx_id].interval))

        diseq = diseqs[tx_id]

        assert diseq.intervals[0].chromosome == chrom, (row, chrom)
        assert diseq.intervals[0].strand == strand, (row, diseq.intervals)

        transcript_position = int(row['transcript_position'])
        genomic_position = int(row['genomic_position'])
        base = row['base']

        # skip transcript position that's outside of GK transcript length
        if transcript_position < 1 or transcript_position > len(diseq.interval):
            print('Skip transcript %s position %d' % (tx_id, transcript_position))
            continue

        genomic_itv = Interval(chrom, strand, genomic_position-1, genomic_position, genome)
        assert genome.dna(genomic_itv) == base

        # skip genomic positions outside of transcript
        if not any([x.overlaps(genomic_itv) for x in diseq.intervals]):
            print('Skip transcript {} position {} {}'.format(tx_id, transcript_position, genomic_itv))
            continue

        # reactivity
        if row['reactivity'] != 'NA':

            val = float(row['reactivity'])
            array_position = len(Interval.spanning(diseq.interval.end5, diseq.lift_interval(genomic_itv).end5))
            if transcript_position != array_position:
                pass
                # print("{} different transcript position in raw data {}, computed from GK {}".format(tx_id,
                #                                                                                     transcript_position,
                #                                                                                     array_position))
            try:
                vals[tx_id][array_position] = val
            except IndexError as e:
                print(e)
                print(tx_id, array_position, vals[tx_id].shape)
                print(diseq.interval, diseq.intervals, genomic_itv, diseq.lift_interval(genomic_itv))
                raise
        else:
            pass

    df_data = []
    for tx_id in diseqs.keys():
        if np.all(vals[tx_id] == config['missing_val_reactivity']):
            print("Skip {} (all missing val)".format(tx_id))
            continue

        df_data.append({
            'transcript_id': tx_id,
            'transcript': transcripts[tx_id],
            'disjoint_intervals': diseqs[tx_id].intervals,
            'reactivity_data': vals[tx_id].tolist(),
        })

    df_data = pd.DataFrame(df_data)

    metadata = dataframe.Metadata()
    metadata.version = "1"
    metadata.encoding["reactivity_data"] = dataframe.Metadata.LIST
    metadata.encoding["transcript"] = dataframe.Metadata.GENOMEKIT
    metadata.encoding["disjoint_intervals"] = dataframe.Metadata.LIST_OF_GENOMEKIT

    write_dataframe(metadata, df_data, output_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str,
                        help='Path to input data (after mapping, counting and normalization, more details later)')
    parser.add_argument('--output_data', type=str, help='Output df')
    parser.add_argument('--data_type', type=str, help='Source data type, supports: pars, dms')
    parser.add_argument('--config', type=str, help='Config file')
    args = parser.parse_args()

    assert args.data_type in ['pars', 'dms']

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(args.input_data, args.output_data, args.data_type, config)
