# generate short rand sequence
# toy dataset
import argparse
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../utils/')
import genome_kit as gk
from rna_ss_utils import get_fe_struct, db2pairs
from rna_ss_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb


def add_target(seq, one_idx):
    pairs, structure_arr = one_idx2arr(one_idx, len(seq), remove_lower_triangular=True)
    pairs = sort_pairs(pairs)
    parser = LocalStructureParser(pairs)
    return parser.local_structure_bounding_boxes


def get_interval_segments(gene, seq_len_min, seq_len_max, n_seq_per_gene_max, rand_step_max):
    itvs = []
    n_seq = np.random.randint(1, n_seq_per_gene_max + 1)
    position = 0
    while len(itvs) < n_seq:
        position += np.random.randint(1, rand_step_max + 1)
        if position > len(gene) - seq_len_max:
            break
        seq_len = np.random.randint(seq_len_min, seq_len_max + 1)
        itvs.append(gene.end5.shift(position).expand(0, seq_len))
    return itvs


def generate_data(genome, chroms, n_data, seq_len_min, seq_len_max, n_seq_per_gene_max, rand_step_max, threshold_mfe_freq):
    data_all = []

    for gene in genome.genes:

        if gene.chromosome not in chroms:
            continue

        itvs = get_interval_segments(gene, seq_len_min, seq_len_max, n_seq_per_gene_max, rand_step_max)
        seqs = [genome.dna(x) for x in itvs]

        for seq in seqs:
            one_idx, fe, mfe_freq, ens_div = get_fe_struct(seq)
            if mfe_freq < threshold_mfe_freq:
                continue
            elif len(one_idx[0]) == 0:  # no structure
                continue
            else:
                bounding_boxes = add_target(seq, one_idx)
                data = {'seq': seq,
                             'one_idx': one_idx,
                             'fe': fe,
                             'mfe_freq': mfe_freq,
                             'ens_div': ens_div,
                             'bounding_boxes': bounding_boxes}

            if len(data_all) == n_data:
                return data_all
            else:
                data_all.append(data)

            # debug
            if len(data_all) % max(1, n_data//100) == 0:
                print("len(data_all) {}".format(len(data_all)))


def main(seq_len_min, seq_len_max, num_seq, threshold_mfe_freq, chromosomes, outfile):
    assert outfile

    # hard-coded params
    n_seq_per_gene_max = 1000
    rand_step_max = 100

    genome = gk.Genome('gencode.v29')

    data = generate_data(genome, chromosomes,
                         num_seq, seq_len_min, seq_len_max, n_seq_per_gene_max, rand_step_max, threshold_mfe_freq)

    df = pd.DataFrame(data)
    df.to_pickle(outfile, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--len_min', type=int, default=20, help='min sequence length')
    parser.add_argument('--len_max', type=int, default=200, help='max sequence length')
    parser.add_argument('--num_seq', type=int, default=100, help='total number of sequences to generate')
    parser.add_argument('--threshold_mfe_freq', type=float, default=0.1, help='Minimum frequency of MFE structure (so we have less uncertainty)')
    parser.add_argument('--chromosomes', type=str, nargs='+', help='chromosome names')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()
    main(args.len_min, args.len_max, args.num_seq, args.threshold_mfe_freq, args.chromosomes, args.out)


