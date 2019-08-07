import genome_kit as gk
import deepgenomics.pandas.v1 as dataframe
import pandas as pd
import math
from dgutils.interval import DisjointIntervalsSequence
from dgutils.pandas import write_dataframe, add_column, add_columns
import gzip


genome = gk.Genome('gencode.vM19')

data = []
# TODO this is the chromatin dataset, shall we also process the nucleo and cytoplasm one?
with gzip.open('raw_data/GSM3310478_mes_ch_vivo.out.txt.gz') as f:
    for line in f:
        line = line.rstrip()
        fields = line.split('\t')
        transcript_id = fields[0]
        transcript_len = int(fields[1])
        unknown_val = float(fields[2])
        vals = [float(x) if x != 'NULL' else float('nan') for x in fields[3:]]
        # skip of all NaN's
        if all([math.isnan(x) for x in vals]):
            print("Skipping {} due to: all NaN vals".format(transcript_id))
            continue
        data.append({
            'transcript_id': transcript_id,
            'transcript_len': transcript_len,
            'unknown_val': unknown_val,
            'vals': vals,
        })
df = pd.DataFrame(data)


# GK transcript

def _get_transcript(transcript_id, transcript_len):
    try:
        transcript = genome.transcripts[transcript_id]
        if sum([len(x) for x in transcript.exons]) == transcript_len:
            return transcript
        else:
            print("Transcript {} length mismatch: GK {} data source {}".format(transcript, sum([len(x) for x in transcript.exons]), transcript_len))
            return None
    except KeyError:
        print("Cannot reconstruct GK transcript {}".format(transcript_id))
        return None


df = add_column(df, 'transcript', ['transcript_id', 'transcript_len'], _get_transcript)


# drop transcript that's not compatible
df = df.dropna(subset=['transcript'])


def _get_seq(transcript):
    diseq = DisjointIntervalsSequence(transcript.exons, genome)
    return diseq.dna(diseq.interval)


# add sequence
df = add_column(df, 'sequence', ['transcript'], _get_seq)

# output
metadata = dataframe.Metadata()
metadata.version = "1"
metadata.encoding["transcript"] = dataframe.Metadata.GENOMEKIT
metadata.encoding["vals"] = dataframe.Metadata.LIST

write_dataframe(metadata, df, 'data/mouse_esc_v65_icshape_ch_vivo.csv')
