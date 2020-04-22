import argparse
from functools import reduce
import pandas as pd
from Bio import pairwise2
from Bio.pairwise2 import format_alignment


def main(d1, d2, out_file):
    # store identical and similar sequence
    df_overlap = []

    df1 = pd.read_pickle(d1)
    df2 = pd.read_pickle(d2)
    print("Loaded data:")
    print("{} {}".format(d1, len(df1)))
    print("{} {}".format(d2, len(df2)))

    # de-dup!
    df1 = df1.drop_duplicates(subset=['seq'])
    df2 = df2.drop_duplicates(subset=['seq'])
    print("After dedup:")
    print("{} {}".format(d1, len(df1)))
    print("{} {}".format(d2, len(df2)))

    n_similar = 0
    n_identical = 0

    for _, row1 in df1.iterrows():
        id1 = row1['seq_id']
        s1 = row1['seq']
        # finding best hit for s1
        # TODO should also start with s2
        # print(s1)
        print(id1, len(s1))
        best_score = 0
        best_id2 = None
        best_alignment = None
        best_overlap = None
        for _, row2 in df2.iterrows():
            id2 = row2['seq_id']
            s2 = row2['seq']
            # for debug
            if s2 == s1:
                n_identical += 1
                df_overlap.append({
                    'd1': d1,
                    'seq_id1': row1['seq_id'],
                    'seq1': s1,
                    'd2': d2,
                    'seq_id2': row2['seq_id'],
                    'seq2': s2,
                    'score': len(s1),
                    'normalized_score': 1.0,
                })

            # only align sequence with comparable length
            if not (0.7 * len(s1) <= len(s2) <= 1.3 * len(s1)):
                continue
            # these penalty term ensure that best case score = seq_len (when 2 seqs are identical)
            alignments = pairwise2.align.globalms(s1, s2, 1, -0.5, -1, -.1,  # match, mismatch, gap open, gap extension
                                                  one_alignment_only=True)
            score = alignments[0][2]
            normalized_score = score / min(len(s1), len(s2))
            if normalized_score > best_score:
                # normalized score, [0, 1]
                best_score = normalized_score
                best_id2 = id2
                best_alignment = alignments[0]
                best_overlap = {
                    'd1': d1,
                    'seq_id1': row1['seq_id'],
                    'seq1': s1,
                    'd2': d2,
                    'seq_id2': row2['seq_id'],
                    'seq2': s2,
                    'score': score,
                    'normalized_score': normalized_score,
                }
        if best_score >= 0.8:
            print(format_alignment(*best_alignment))
            print(best_score)
            print(best_id2)
            n_similar += 1
            df_overlap.append(best_overlap)
        else:
            print("No similar sequence found!\n")

    print("{} {}".format(d1, len(df1)))
    print("{} {}".format(d2, len(df2)))
    print("Found {} similar out of a total {}".format(n_similar, len(df1)))
    print("n_identical {}".format(n_identical))

    # output
    df_overlap = pd.DataFrame(df_overlap)
    df_overlap.to_csv(out_file, compression='gzip', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d1', type=str, help='path to dataframe pkl for dataset 1')
    parser.add_argument('--d2', type=str, help='path to dataframe pkl for dataset 2')
    parser.add_argument('--out_file', help='path to output file')
    args = parser.parse_args()
    main(args.d1, args.d2, args.out_file)


