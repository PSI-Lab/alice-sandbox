# generate short rand sequence
# toy dataset
import argparse
import random
from tqdm import tqdm
import numpy as np
import pickle
# from subprocess import PIPE, Popen
import pandas as pd
from dgutils.pandas import add_columns, add_column
import sys
sys.path.insert(0, '../utils/')
import genome_kit as gk
from rna_ss_utils import sample_structures, db2pairs, one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb


def add_target(seq, one_idx):
    pairs, structure_arr = one_idx2arr(one_idx, len(seq), remove_lower_triangular=True)
    pairs = sort_pairs(pairs)
    parser = LocalStructureParser(pairs)
    return parser.local_structure_bounding_boxes


def get_interval_segments(gene, seq_len, n_seq_per_gene_max, rand_step_max):
    itvs = []
    n_seq = np.random.randint(1, n_seq_per_gene_max + 1)
    position = 0
    while len(itvs) < n_seq:
        position += np.random.randint(1, rand_step_max + 1)
        if position > len(gene) - seq_len:
            break
        itvs.append(gene.end5.shift(position).expand(0, seq_len))
    return itvs


def generate_data(genome, chroms, n_data, n_sample, seq_len, n_seq_per_gene_max, rand_step_max):
    data_all = []

    for gene in genome.genes:

        if gene.chromosome not in chroms:
            continue

        itvs = get_interval_segments(gene, seq_len, n_seq_per_gene_max, rand_step_max)
        seqs = [genome.dna(x) for x in itvs]

        for seq in seqs:
            all_one_idx = sample_structures(seq, n_samples=n_sample, remove_dup=True)
            for one_idx in all_one_idx:
                if len(one_idx[0]) == 0:  # no structure
                    continue
                else:
                    bounding_boxes = add_target(seq, one_idx)
                    data = {'seq': seq,
                                 'one_idx': one_idx,
                                 # 'fe': fe,
                                 # 'mfe_freq': mfe_freq,
                                 # 'ens_div': ens_div,
                                 'bounding_boxes': bounding_boxes}

                if len(data_all) == n_data:
                    return data_all
                else:
                    data_all.append(data)

            # debug
            if len(data_all) % max(1, n_data//100) == 0:
                print("len(data_all) {}".format(len(data_all)))


def main(seq_len, num_data, n_sample, chromosomes, outfile):
    assert outfile

    # hard-coded params
    n_seq_per_gene_max = 1000
    rand_step_max = 100

    genome = gk.Genome('gencode.v29')

    data = generate_data(genome, chromosomes,
                         num_data, n_sample, seq_len, n_seq_per_gene_max, rand_step_max)

    df = pd.DataFrame(data)
    df.to_pickle(outfile, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--len', type=int, default=50, help='sequence length (fixed)')
    parser.add_argument('--num_data', type=int, default=100, help='total number of data points (seq * sample_per_seq) to generate')
    parser.add_argument('--num_sample', type=int, default=10,
                        help='number of (unique) structures per sequence to sample. total number of examples is num_seq*num_sample')
    # parser.add_argument('--threshold_mfe_freq', type=float, default=0.1, help='Minimum frequency of MFE structure (so we have less uncertainty)')
    parser.add_argument('--chromosomes', type=str, nargs='+', help='chromosome names')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()
    main(args.len, args.num_data, args.num_sample, args.chromosomes, args.out)


