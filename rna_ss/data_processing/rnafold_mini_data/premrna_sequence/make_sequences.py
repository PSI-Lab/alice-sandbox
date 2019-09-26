import genome_kit as gk
import pandas as pd
import random
import argparse


def main(min_len, max_len, out_file_prefix, n_parts):
    random.seed(5555)

    genome = gk.Genome('gencode.v29')
    seqs = []
    print("Generating random length sequences")
    for gene in genome.genes:
        seq = genome.dna(gene)
        # split into non-overlapping substrings
        # of random lengths within [min_len, max_len]
        idx_start = 0
        while len(seq) - idx_start >= min_len:
            offset = random.randint(min_len, max_len)
            _s = seq[idx_start:idx_start+offset]
            _s = _s.upper()
            if 'N' in _s:
                continue
            _s = _s.replace('T', 'U')
            seqs.append(_s)
            idx_start += offset
        # debug FIXME remove
        print("debug")
        break

    print("Generated {} sequences".format(len(seqs)))
    # TODO remove duplicates

    # split into parts
    batch_size = len(seqs) // n_parts
    for idx_part in range(n_parts):
        part_start = idx_part * batch_size
        if idx_part == n_parts - 1:
            part_end = len(seqs)
        else:
            part_end = (idx_part + 1) * batch_size
        df = pd.DataFrame({'sequence': seqs[part_start:part_end]})
        out_file_name = "{}_{}.csv".format(out_file_prefix, idx_part+1)
        df.to_csv(out_file_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--minlen', type=int, default=10, help='min sequence length')
    parser.add_argument('--maxlen', type=int, default=100, help='max sequence length')
    parser.add_argument('--parts', type=int, default=100, help='number of output files')
    parser.add_argument('--out', type=str, help='output file_prefix')
    args = parser.parse_args()
    main(args.minlen, args.maxlen, args.out, args.parts)

